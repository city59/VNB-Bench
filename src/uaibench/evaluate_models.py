"""Evaluate all scorers on UAIBench and on two standard protocols.

python -m src.uaibench.evaluate_models <run> [n_boot]

Scorers
  random, recency, frequency, cart_first, popularity   behavior heuristics (as in build.scores)
  gbdt        gradient-boosted trees trained on training-period UAI candidates to predict conversion
  sasrec_t, sasrec_mb, mbstr, pbat                   neural models (scores from models_seq.py)

Protocols
  uai        UAIBench test queries: conversion / re-engagement ccNDCG@10, conditional-conversion AUC
  std        first test-period purchase per user, 1 positive + 99 uniform negatives excluding the user's history
  stdfull    same instances, full-catalog ranking with previously purchased items masked
All standard metrics are tie-aware (expected value under random tie-breaking).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .build import CONVERTED, RETURNED, SILENT, SHIFT, Config, pass1, scores as heuristic_scores, user_macro
from .metrics import query_metrics

ROOT = Path(__file__).resolve().parents[2]
NEURAL = ["mbstr", "pbat", "mbht", "missl", "mbsrec", "mbgen"]
HEUR = ["random", "recency", "frequency", "cart_first", "popularity"]


def gbdt_features(df: pd.DataFrame) -> np.ndarray:
    q = df.u.to_numpy().astype(np.int64) * SHIFT + df.t.to_numpy()
    g = pd.DataFrame({"q": q, "last": df.last_ts.to_numpy(np.float64), "cnt": df.cnt_all.to_numpy(np.float64)})
    qmax = g.groupby("q")["last"].transform("max").to_numpy()
    n = g.groupby("q")["last"].transform("size").to_numpy()
    rk_last = g.groupby("q")["last"].rank(ascending=False, method="average").to_numpy()
    rk_cnt = g.groupby("q")["cnt"].rank(ascending=False, method="average").to_numpy()
    return np.column_stack([
        np.log1p(df.cnt_w.to_numpy()), np.log1p(df.cnt_all.to_numpy()), df.cart_w.to_numpy(), df.cart_all.to_numpy(),
        df.fav_all.to_numpy(), np.log1p(np.maximum(qmax - df.last_ts.to_numpy(), 0) / 3600.0),
        df.t.to_numpy() - df.first_day.to_numpy(), np.log1p(df["pop"].to_numpy()), np.log1p(n),
        (rk_last - 1) / np.maximum(n - 1, 1), (rk_cnt - 1) / np.maximum(n - 1, 1),
    ]).astype(np.float32)


def eval_uai(df: pd.DataFrame, score_cols: dict, k: int = 10) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    q = df.u.to_numpy().astype(np.int64) * SHIFT + df.t.to_numpy()
    order = np.argsort(q, kind="stable")
    q, lab, u = q[order], df.label.to_numpy()[order], df.u.to_numpy()[order]
    sc = {n: s[order] for n, s in score_cols.items()}
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    rows = []
    for a, b in zip(starts, ends):
        L_ = lab[a:b]
        if (L_ == L_[0]).all():
            continue
        r = {"u": int(u[a])}
        for n, s in sc.items():
            for m, v in query_metrics(L_, s[a:b], None, k).items():
                if m in ("cc_ndcg_conv", "cc_ndcg_engage", "auc_conditional", "ndcg_conv"):
                    r[f"{n}:{m}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def tie_aware(gt: np.ndarray, ties: np.ndarray, k: int = 10) -> tuple:
    """Expected HR@k and NDCG@k when the positive is uniformly placed among its ties."""
    hr, nd = np.zeros(len(gt)), np.zeros(len(gt))
    for j, (g, t) in enumerate(zip(gt.astype(np.int64), ties.astype(np.int64))):
        pos = np.arange(g, g + t + 1)
        hr[j] = np.mean(pos < k)
        nd[j] = np.mean(np.where(pos < k, 1.0 / np.log2(pos + 2.0), 0.0))
    return hr, nd


def boot(x: np.ndarray, n_boot: int, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return {"mean": float(x.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "n": int(len(x))}


def std_features(d, items_needed, uids, tss, W):
    """Per (instance, candidate) history features from events strictly before the purchase time."""
    e = d["events"]
    key = e["u"].astype(np.int64) * (1 << 31) + (e["ts"] - e["ts"].min())
    order = np.argsort(key, kind="stable")
    u, i, b, ts, day = e["u"][order], e["i"][order], e["b"][order], e["ts"][order], e["day"][order]
    key = key[order]
    starts = np.searchsorted(key, uids.astype(np.int64) * (1 << 31), side="left")
    ends = np.searchsorted(key, uids.astype(np.int64) * (1 << 31) + (tss - e["ts"].min()), side="left")
    feats = []
    for r, (s, t_end) in enumerate(zip(starts, ends)):
        ii, bb, tt, dd = i[s:t_end], b[s:t_end], ts[s:t_end], day[s:t_end]
        aux = bb < 4
        cand = items_needed[r]
        f = np.zeros((len(cand), 6))
        if aux.any():
            ia, ba, ta, da = ii[aux], bb[aux], tt[aux], dd[aux]
            dmax = dd.max() if len(dd) else 0
            hit = np.isin(cand, ia)
            for c in np.flatnonzero(hit):
                m = ia == cand[c]
                f[c] = [m.sum(), (ba[m] == 3).any(), (ba[m] == 2).any(), ta[m].max(), (m & (da > dmax - W)).sum(), 1]
        feats.append(f)
    return feats


def main() -> None:
    run = sys.argv[1]
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    d_dir = ROOT / "data/bench" / run
    meta = json.loads((d_dir / "meta.json").read_text())
    W = meta["window"]
    tr = pd.DataFrame(dict(np.load(d_dir / "uai_train.npz")))
    ev = pd.DataFrame(dict(np.load(d_dir / "uai_eval.npz")))
    std = dict(np.load(d_dir / "std_eval.npz"))
    events = dict(np.load(d_dir / "events.npz"))
    # --- GBDT on training-period candidates. Training-period popularity would include each training anchor's
    # own future purchase, so training rows use point-in-time popularity (all users' purchases before day t).
    bc = np.load(ROOT / "data/processed" / run / "buy_counts_all_users.npz")
    o = np.lexsort((bc["day"], bc["item"]))
    bkey = bc["item"][o].astype(np.int64) * 65536 + bc["day"][o]
    bcum = np.r_[0, np.cumsum(bc["count"][o])]
    lo = np.searchsorted(bkey, tr.i.to_numpy().astype(np.int64) * 65536, side="left")
    hi = np.searchsorted(bkey, tr.i.to_numpy().astype(np.int64) * 65536 + tr.t.to_numpy(), side="left")
    tr["pop"] = bcum[hi] - bcum[lo]
    y = (tr.label.to_numpy() == CONVERTED).astype(int)
    gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True,
                                        validation_fraction=0.1, n_iter_no_change=20, random_state=2026)
    gb.fit(gbdt_features(tr), y)
    result = {"run": run, "gbdt": {"n_train": int(len(tr)), "pos_rate": float(y.mean()), "iters": int(gb.n_iter_)}}
    # --- UAIBench
    test = ev[ev.split == 2].reset_index(drop=True)
    sc = heuristic_scores(test, 2026)
    sc["gbdt"] = gb.predict_proba(gbdt_features(test))[:, 1]
    test_mask = (ev.split == 2).to_numpy()
    for k in NEURAL:
        f = d_dir / f"scores_{k}_s2026.npz"
        if f.exists():
            z = np.load(f)
            sc[k] = z["uai_split2"].astype(np.float64)
            assert len(sc[k]) == test_mask.sum()
    qm = eval_uai(test, sc)
    result["uai"] = {col: user_macro(qm, col, n_boot, 2026) for col in qm.columns if ":" in col}
    qm.to_csv(d_dir / "uai_query_metrics_test.csv.gz", index=False)
    # --- diagnostic: does a model's score carry information beyond the behavior features? Model scores exist
    # for the validation period, so both GBDTs are fit there (features only / features + model score and its
    # within-query rank) and evaluated on the test period.
    val = ev[ev.split == 1].reset_index(drop=True)
    yv = (val.label.to_numpy() == CONVERTED).astype(int)
    qv = val.u.to_numpy().astype(np.int64) * SHIFT + val.t.to_numpy()
    qt = test.u.to_numpy().astype(np.int64) * SHIFT + test.t.to_numpy()

    def _fit(X):
        g = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=20, random_state=2026)
        return g.fit(X, yv)

    def _with(df_feat, s, q):
        r = pd.Series(s).groupby(q).rank(pct=True).to_numpy()
        return np.column_stack([df_feat, s, r]).astype(np.float32)

    Fv, Ft = gbdt_features(val), gbdt_features(test)
    stack = {"gbdt_val": _fit(Fv).predict_proba(Ft)[:, 1]}
    for k in NEURAL:
        f = d_dir / f"scores_{k}_s2026.npz"
        if f.exists():
            z = np.load(f)
            sv, st_ = z["uai_split1"].astype(np.float64), z["uai_split2"].astype(np.float64)
            stack[f"gbdt_val+{k}"] = _fit(_with(Fv, sv, qv)).predict_proba(_with(Ft, st_, qt))[:, 1]
    if len(stack) > 1:
        qs = eval_uai(test, stack)
        result["stack"] = {col: user_macro(qs, col, n_boot, 2026) for col in qs.columns if col.endswith(":cc_ndcg_conv")}
    # --- standard protocols
    m2 = std["split"] == 2
    uids, tss, pos, negs = std["uid"][m2], std["ts"][m2], std["pos"][m2], std["negs"][m2]
    cands = np.concatenate([pos[:, None], negs], 1)
    feats = std_features({"events": events}, cands, uids, tss, W)
    # popularity over all users' training purchases, mapped to the model vocabulary
    g1 = pass1(sorted((ROOT / "data/processed" / run).glob("bucket_*.npz")), Config())
    pop_full = np.zeros(meta["n_items"] + 1)
    idx = np.searchsorted(g1["train_items"], g1["pop_items"])
    ok = (idx < len(g1["train_items"])) & (g1["train_items"][np.minimum(idx, len(g1["train_items"]) - 1)] == g1["pop_items"])
    pop_full[idx[ok] + 1] = g1["pop_counts"][ok]
    std_scores = {n: np.zeros(cands.shape) for n in HEUR + ["gbdt"]}
    for r, f in enumerate(feats):
        cnt, fav, cart, last, cnt_w, has = f.T
        std_scores["recency"][r] = np.where(has > 0, last, -1)
        std_scores["frequency"][r] = cnt
        std_scores["cart_first"][r] = cart * 1e12 + fav * 1e11 + cnt * 1e7 + np.where(has > 0, last - 1.4e9, 0) / 1e3
        std_scores["popularity"][r] = pop_full[cands[r]]
        dfr = pd.DataFrame({"u": np.arange(len(cnt)), "t": 0, "cnt_w": cnt_w, "cnt_all": cnt, "cart_w": cart, "cart_all": cart, "fav_all": fav,
                            "last_ts": np.where(has > 0, last, 0), "first_day": 0, "pop": pop_full[cands[r]]})
        std_scores["gbdt"][r] = gb.predict_proba(gbdt_features(dfr))[:, 1]
    for k in NEURAL:
        f = d_dir / f"scores_{k}_s2026.npz"
        if f.exists():
            std_scores[k] = np.load(f)["std_split2"].astype(np.float64)
    res_std, res_full = {}, {}
    for n, s in std_scores.items():
        gt = (s[:, 1:] > s[:, :1]).sum(1)
        ties = (s[:, 1:] == s[:, :1]).sum(1)
        hr, nd = tie_aware(gt, ties)
        res_std[n] = {"hr10": boot(hr, n_boot), "ndcg10": boot(nd, n_boot)}
    # full-catalog ranking for heuristics/GBDT. Untouched items carry a catalog score (constant, popularity, or the
    # GBDT score of an item without history); history items are scored exactly; previous purchases are masked.
    zero = pd.DataFrame({"u": np.arange(len(pop_full)), "t": 0, "cnt_w": 0.0, "cnt_all": 0.0, "cart_w": 0.0, "cart_all": 0.0,
                         "fav_all": 0.0, "last_ts": 0.0, "first_day": 0, "pop": pop_full})
    upop, inv = np.unique(pop_full, return_inverse=True)
    zu = pd.DataFrame({"u": np.arange(len(upop)), "t": 0, "cnt_w": 0.0, "cnt_all": 0.0, "cart_w": 0.0, "cart_all": 0.0,
                       "fav_all": 0.0, "last_ts": 0.0, "first_day": 0, "pop": upop})
    gz_full = gb.predict_proba(gbdt_features(zu))[:, 1][inv]
    catalog = {"random": np.zeros(len(pop_full)), "recency": np.full(len(pop_full), -1.0),
               "frequency": np.zeros(len(pop_full)), "cart_first": np.zeros(len(pop_full)),
               "popularity": pop_full, "gbdt": gz_full}
    hist = history_index(events)
    for n in HEUR + ["gbdt"]:
        cat = catalog[n]
        cat_sorted = np.sort(cat[1:])
        hrs, nds = [], []
        for r in range(len(uids)):
            gt, ties = full_rank_counts(n, std_scores[n][r][0], cands[r][0], uids[r], tss[r], hist, cat, cat_sorted,
                                        pop_full, gb, W)
            h_, n_ = tie_aware(np.array([gt]), np.array([ties]))
            hrs.append(h_[0]); nds.append(n_[0])
        res_full[n] = {"hr10": boot(np.array(hrs), n_boot), "ndcg10": boot(np.array(nds), n_boot)}
    for k in NEURAL:
        f = d_dir / f"scores_{k}_s2026.npz"
        if f.exists():
            rk = np.load(f)["stdfull_rank_split2"]
            hr = (rk < 10).astype(float)
            nd = np.where(rk < 10, 1.0 / np.log2(rk + 2.0), 0.0)
            res_full[k] = {"hr10": boot(hr, n_boot), "ndcg10": boot(nd, n_boot)}
    result["std"], result["stdfull"] = res_std, res_full
    result["std_instances"] = int(m2.sum())
    (d_dir / "eval_results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"uai_conv": {n: round(result["uai"].get(f"{n}:cc_ndcg_conv", {}).get("mean", float("nan")), 3) for n in sc},
                      "std_ndcg": {n: round(v["ndcg10"]["mean"], 3) for n, v in res_std.items()},
                      "stdfull_ndcg": {n: round(v["ndcg10"]["mean"], 3) for n, v in res_full.items()}}, indent=1))


def history_index(e):
    k_ = e["u"].astype(np.int64) * (1 << 31) + (e["ts"] - e["ts"].min())
    o = np.argsort(k_, kind="stable")
    return k_[o], e["i"][o], e["b"][o], e["ts"][o], e["day"][o], e["ts"].min()


def full_rank_counts(n, ps, pos_item, uid, ts, hist, cat, cat_sorted, pop_full, gb, W):
    """Number of catalog items scored above the positive and tied with it (positive excluded)."""
    kk, ii, bb, tt, dd, t0 = hist
    s = np.searchsorted(kk, int(uid) * (1 << 31), side="left")
    t_end = np.searchsorted(kk, int(uid) * (1 << 31) + (int(ts) - t0), side="left")
    hi, hb, ht, hd = ii[s:t_end], bb[s:t_end], tt[s:t_end], dd[s:t_end]
    bought = set(hi[hb == 4].tolist()) - {int(pos_item)}
    aux = hb < 4
    dmax = hd.max() if len(hd) else 0
    h_items = np.unique(hi[aux])
    h_items = np.array([x for x in h_items if x not in bought], dtype=np.int64)
    feats = []
    for x in h_items:
        m = aux & (hi == x)
        feats.append([m.sum(), (hb[m] == 3).any(), (hb[m] == 2).any(), ht[m].max(), (m & (hd > dmax - W)).sum()])
    f = np.array(feats, dtype=float).reshape(-1, 5)
    cnt, cart, fav, last, cntw = f.T if len(f) else (np.zeros(0),) * 5
    if n == "random":
        hs = np.zeros(len(f))
    elif n == "recency":
        hs = last
    elif n == "frequency":
        hs = cnt
    elif n == "cart_first":
        hs = cart * 1e12 + fav * 1e11 + cnt * 1e7 + (last - 1.4e9) / 1e3
    elif n == "popularity":
        hs = pop_full[h_items]
    else:
        dfh = pd.DataFrame({"u": np.arange(len(f)), "t": 0, "cnt_w": cntw, "cnt_all": cnt, "cart_w": cart, "cart_all": cart,
                            "fav_all": fav, "last_ts": last, "first_day": 0, "pop": pop_full[h_items]})
        hs = gb.predict_proba(gbdt_features(dfh))[:, 1] if len(f) else np.zeros(0)
    excl = np.array(sorted(set(h_items.tolist()) | bought), dtype=np.int64)
    gt = (len(cat_sorted) - np.searchsorted(cat_sorted, ps, side="right")) - int((cat[excl] > ps).sum()) + int((hs > ps).sum())
    eq = (np.searchsorted(cat_sorted, ps, side="right") - np.searchsorted(cat_sorted, ps, side="left"))         - int((cat[excl] == ps).sum()) + int((hs == ps).sum())
    return gt, max(eq - 1, 0)


if __name__ == "__main__":
    main()
