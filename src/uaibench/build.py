"""Construct prospective UAI queries from canonical buckets and evaluate scorers.

Pass 1 (global): split boundaries, training item vocabulary, training purchase
popularity, auxiliary-to-purchase delays (for the horizon rule) and purchase
statistics. Pass 2 (per bucket): activity-triggered daily anchors, candidate
sets, features and outcome labels. Pass 3: metrics with the training-estimated
Returned gain g.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import BUY, CART, FAV
from .metrics import CONVERTED, RETURNED, SILENT, query_metrics, random_ndcg

SHIFT = 4096  # day slots per key; all logs are far shorter
IBITS = 23     # item ids must be < 2**23


@dataclass
class Config:
    split: tuple = (0.6, 0.2, 0.2)
    delta: int | None = None          # None -> derived by the coverage rule
    delta_grid: tuple = (1, 2, 3, 7, 14)
    coverage: float = 0.8
    window: int | None = None         # None -> W = delta
    warmup: int | None = None         # None -> W
    exclude_repurchase: bool = True
    k: int = 10
    n_boot: int = 1000
    seed: int = 2026
    derived: dict = field(default_factory=dict)


def _load(path: Path) -> pd.DataFrame:
    z = np.load(path)
    return pd.DataFrame({k: z[k] for k in z.files})


def _pk(u, i):
    return (np.asarray(u, np.int64) << IBITS) | np.asarray(i, np.int64)


class PairIndex:
    """Sorted per-pair auxiliary day records of one bucket."""

    def __init__(self, ev: pd.DataFrame):
        assert ev.i.max() < 2**IBITS
        aux = ev[ev.b != BUY]
        pk = _pk(aux.u, aux.i)
        g = pd.DataFrame({"pk": pk, "day": aux.day.to_numpy(), "cart": (aux.b == CART).to_numpy(),
                          "fav": (aux.b == FAV).to_numpy(), "ts": aux.ts.to_numpy()})
        g = g.groupby(["pk", "day"], sort=True).agg(n=("ts", "size"), cart=("cart", "max"),
                                                    fav=("fav", "max"), last=("ts", "max")).reset_index()
        self.pk = g.pk.to_numpy()
        self.day = g.day.to_numpy()
        self.comb = self.pk * SHIFT + self.day
        self.cnt = np.r_[0, np.cumsum(g.n.to_numpy())]
        self.cart = np.r_[0, np.cumsum(g.cart.to_numpy())]
        self.fav = np.r_[0, np.cumsum(g.fav.to_numpy())]
        self.last = g["last"].to_numpy()
        buys = ev[ev.b == BUY]
        fb = pd.Series(buys.day.to_numpy(), index=_pk(buys.u, buys.i)).groupby(level=0).min()
        self.fb_pk = fb.index.to_numpy()
        self.fb_day = fb.to_numpy()
        act = np.unique(ev.u.to_numpy(np.int64) * SHIFT + ev.day.to_numpy())
        self.active = act

    def first_buy(self, pk: np.ndarray) -> np.ndarray:
        out = np.full(len(pk), np.iinfo(np.int64).max)
        if len(self.fb_pk) == 0:
            return out
        idx = np.minimum(np.searchsorted(self.fb_pk, pk), len(self.fb_pk) - 1)
        hit = self.fb_pk[idx] == pk
        out[hit] = self.fb_day[idx[hit]]
        return out

    def is_active(self, u: np.ndarray, t: np.ndarray) -> np.ndarray:
        key = u * SHIFT + t
        idx = np.minimum(np.searchsorted(self.active, key), len(self.active) - 1)
        return self.active[idx] == key


def split_bounds(n_days: int, split: tuple) -> list:
    a = int(round(n_days * split[0]))
    b = int(round(n_days * (split[0] + split[1])))
    return [(0, a), (a, b), (b, n_days)]  # half-open day ranges


# ---------------------------------------------------------------- pass 1
def pass1(buckets: list, cfg: Config) -> dict:
    n_days = 0
    for p in buckets:
        n_days = max(n_days, int(np.load(p)["day"].max()) + 1)
    bounds = split_bounds(n_days, cfg.split)
    tr_end = bounds[0][1]
    deriv_start = min(7, tr_end // 2)
    items, pops, delays = [], [], []
    users = set()
    purchase_stats = {s: np.zeros(3, np.int64) for s in ("train", "val", "test")}
    n_events = 0
    for p in buckets:
        ev = _load(p)
        n_events += len(ev)
        users.update(np.unique(ev.u).tolist())
        tr = ev[ev.day < tr_end]
        items.append(np.unique(tr.i))
        pops.append(tr[tr.b == BUY].i.to_numpy())
        idx = PairIndex(ev)
        # first purchases and the last auxiliary day strictly before the purchase day
        hi = np.searchsorted(idx.comb, idx.fb_pk * SHIFT + idx.fb_day, side="left")
        prev_ok = (hi > 0) & (idx.comb[np.maximum(hi - 1, 0)] // SHIFT == idx.fb_pk)
        prev_day = np.where(prev_ok, idx.comb[np.maximum(hi - 1, 0)] % SHIFT, -1)
        # any auxiliary event at or before the purchase timestamp is approximated at day level
        same_or_before = np.searchsorted(idx.comb, idx.fb_pk * SHIFT + idx.fb_day, side="right")
        any_prior = (same_or_before > 0) & (idx.comb[np.maximum(same_or_before - 1, 0)] // SHIFT == idx.fb_pk)
        for name, (lo, hi_d) in zip(("train", "val", "test"), bounds):
            m = (idx.fb_day >= max(lo, deriv_start)) & (idx.fb_day < hi_d)
            purchase_stats[name] += [m.sum(), (m & any_prior).sum(), (m & prev_ok).sum()]
        m = (idx.fb_day >= deriv_start) & (idx.fb_day < tr_end) & prev_ok
        delays.append(idx.fb_day[m] - prev_day[m])
    train_items = np.unique(np.concatenate(items))
    full = Path(buckets[0]).parent / "buy_counts_all_users.npz"
    if full.exists():  # population-level training purchases (independent of user sampling)
        z = np.load(full)
        m = z["day"] < tr_end
        s = pd.Series(z["count"][m]).groupby(z["item"][m]).sum()
        pop_items, pop_counts = s.index.to_numpy(np.int64), s.to_numpy(np.int64)
    else:
        pop_items, pop_counts = np.unique(np.concatenate(pops), return_counts=True)
    delays = np.concatenate(delays)
    test_len = bounds[2][1] - bounds[2][0]
    if cfg.delta is None:
        feasible = [d for d in cfg.delta_grid if test_len >= 2 * d] or [min(cfg.delta_grid)]
        cover = {d: float((delays <= d).mean()) if len(delays) else math.nan for d in cfg.delta_grid}
        meet = [d for d in feasible if cover[d] >= cfg.coverage]
        delta = min(meet) if meet else max(feasible)
    else:
        delta = cfg.delta
        cover = {d: float((delays <= d).mean()) if len(delays) else math.nan for d in cfg.delta_grid}
    window = cfg.window or delta
    warmup = cfg.warmup if cfg.warmup is not None else window
    return {
        "n_days": n_days, "bounds": bounds, "n_events": n_events, "n_users": len(users),
        "train_items": train_items, "pop_items": pop_items, "pop_counts": pop_counts,
        "delay_coverage": cover, "n_delays": int(len(delays)),
        "delay_quantiles": {q: float(np.quantile(delays, q)) for q in (0.5, 0.8, 0.9)} if len(delays) else {},
        "delta": delta, "window": window, "warmup": warmup, "test_len": test_len,
        "purchase_stats": {k: dict(zip(("first_purchases", "with_prior_aux_same_or_earlier_day",
                                        "considered_aux_on_earlier_day"), v.tolist()))
                           for k, v in purchase_stats.items()},
    }


# ---------------------------------------------------------------- pass 2
def candidates(ev: pd.DataFrame, g1: dict) -> tuple[pd.DataFrame, dict]:
    idx = PairIndex(ev)
    W, D = g1["window"], g1["delta"]
    split_of = np.full(g1["n_days"] + W + D + 2, -1)
    last_anchor = np.zeros(3, np.int64)
    for s, (lo, hi) in enumerate(g1["bounds"]):
        split_of[lo:hi] = s
        last_anchor[s] = hi - 1 - D  # window (t, t+D] must stay inside the split
    keys = []
    u_all = idx.pk >> IBITS
    for k in range(W):
        t = idx.day + k
        s = split_of[t]
        ok = (s >= 0) & (t >= g1["warmup"])
        ok &= t <= last_anchor[np.maximum(s, 0)]
        ok &= idx.is_active(u_all, t)
        keys.append(idx.pk[ok] * SHIFT + t[ok])
    key = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
    pk, t = key // SHIFT, key % SHIFT
    u, i = pk >> IBITS, pk & ((1 << IBITS) - 1)
    stats = {"generated": len(pk)}
    fb = idx.first_buy(pk)
    keep = fb > t
    stats["excluded_already_purchased"] = int((~keep).sum())
    s = split_of[t]
    warm = np.isin(i, g1["train_items"])
    cold = (~warm) & (s > 0) & keep
    stats["excluded_cold_item"] = int(cold.sum())
    keep &= ~cold
    pk, t, u, i, fb, s = pk[keep], t[keep], u[keep], i[keep], fb[keep], s[keep]
    lo = np.searchsorted(idx.comb, pk * SHIFT + (t - W), side="right")
    hi = np.searchsorted(idx.comb, pk * SHIFT + t, side="right")
    first = np.searchsorted(idx.comb, pk * SHIFT, side="left")
    nxt = np.minimum(hi, len(idx.comb) - 1)
    has_next = (hi < len(idx.comb)) & (idx.comb[nxt] // SHIFT == pk)
    next_day = np.where(has_next, idx.comb[nxt] % SHIFT, np.iinfo(np.int64).max)
    conv = (fb > t) & (fb <= t + D)
    ret = (~conv) & (next_day <= t + D)
    label = np.where(conv, CONVERTED, np.where(ret, RETURNED, SILENT))
    pop = np.zeros(len(i))
    pi = np.searchsorted(g1["pop_items"], i)
    pi_c = np.minimum(pi, len(g1["pop_items"]) - 1)
    hitp = g1["pop_items"][pi_c] == i
    pop[hitp] = g1["pop_counts"][pi_c[hitp]]
    df = pd.DataFrame({
        "u": u, "i": i, "t": t, "split": s, "label": label,
        "cnt_w": idx.cnt[hi] - idx.cnt[lo], "cnt_all": idx.cnt[hi] - idx.cnt[first],
        "cart_w": (idx.cart[hi] - idx.cart[lo]) > 0, "cart_all": (idx.cart[hi] - idx.cart[first]) > 0,
        "fav_all": (idx.fav[hi] - idx.fav[first]) > 0,
        "last_ts": idx.last[hi - 1], "first_day": idx.comb[first] % SHIFT, "pop": pop,
        # Returned at t, converted in the following window: used only on training anchors for g
        "conv_next": (fb > t + D) & (fb <= t + 2 * D),
        "next_window_ok": t + 2 * D <= last_anchor[s] + D,
    })
    return df, stats


def scores(df: pd.DataFrame, seed: int) -> dict:
    cols = np.column_stack([df.cart_all.to_numpy(np.int64), df.fav_all.to_numpy(np.int64),
                            df.cnt_all.to_numpy(np.int64), df.last_ts.to_numpy(np.int64)])
    strength = np.lexsort(cols.T[::-1])
    # equal keys must share a score for tie-aware metrics
    key = cols[strength]
    newgrp = np.r_[True, (key[1:] != key[:-1]).any(axis=1)] if len(key) else np.zeros(0, bool)
    rank = np.empty(len(df))
    rank[strength] = np.cumsum(newgrp)
    return {
        # a constant score ties every item; tie-aware metrics then equal the exact random-order expectation
        "random": np.zeros(len(df)),
        "recency": df.last_ts.to_numpy(float),
        "frequency": df.cnt_all.to_numpy(float),
        "cart_first": rank,
        "popularity": df["pop"].to_numpy(float),
    }


# ---------------------------------------------------------------- pass 3
def evaluate(df: pd.DataFrame, g: float, k: int, seed: int) -> pd.DataFrame:
    df = df.sort_values(["u", "t"], kind="stable").reset_index(drop=True)
    sc = scores(df, seed)
    q = (df.u.to_numpy() * SHIFT + df.t.to_numpy())
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    lab = df.label.to_numpy()
    rows = []
    for a, b in zip(starts, ends):
        L = lab[a:b]
        if (L == L[0]).all():
            continue
        base = {"u": int(df.u[a]), "t": int(df.t[a]), "split": int(df.split[a]), "n": b - a,
                "n_c": int((L == CONVERTED).sum()), "n_r": int((L == RETURNED).sum())}
        conv = (L == CONVERTED).astype(float)
        base["random_ndcg_conv"] = random_ndcg(conv, k) if 0 < conv.sum() < len(conv) else math.nan
        for name, s in sc.items():
            for m, v in query_metrics(L, s[a:b], g, k).items():
                base[f"{name}:{m}"] = v
        rows.append(base)
    return pd.DataFrame(rows)


def user_macro(qm: pd.DataFrame, col: str, n_boot: int, seed: int) -> dict:
    d = qm[["u", col]].dropna()
    if d.empty:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "queries": 0, "users": 0}
    um = d.groupby("u")[col].mean().to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.array([um[rng.integers(0, len(um), len(um))].mean() for _ in range(n_boot)]) if n_boot else um.mean(keepdims=True)
    return {"mean": float(um.mean()), "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
            "queries": int(len(d)), "users": int(len(um))}


def _support(c: pd.DataFrame) -> tuple[dict, pd.Series]:
    lab = c.label.to_numpy()
    qs = pd.DataFrame({"u": c.u.to_numpy(), "t": c.t.to_numpy(), "c": lab == CONVERTED,
                       "r": lab == RETURNED, "s_": lab == SILENT}).groupby(["u", "t"]).sum()
    qs["n"] = qs.c + qs.r + qs.s_
    masks = {
        "any_informative": ((qs.c > 0).astype(int) + (qs.r > 0) + (qs.s_ > 0)) >= 2,
        "binary_informative": (qs.c > 0) & (qs.c < qs.n),
        "engage_informative": ((qs.c + qs.r) > 0) & (qs.s_ > 0),
        "conditional": (qs.c > 0) & (qs.r > 0),
    }
    out = {"candidates": len(c), "queries": len(qs), "users": int(c.u.nunique()),
           "converted": int(qs.c.sum()), "returned": int(qs.r.sum()), "silent": int(qs.s_.sum())}
    for k_, m in masks.items():
        out[f"{k_}_queries"] = int(m.sum())
        out[f"{k_}_users"] = int(qs[m].index.get_level_values(0).nunique())
    return out, qs.n.value_counts()


def run(buckets: list, cfg: Config) -> tuple[dict, pd.DataFrame]:
    g1 = pass1(buckets, cfg)
    gen, keep_frames = {}, []
    sup = {n: {} for n in ("train", "val", "test")}
    sizes = {n: pd.Series(dtype=float) for n in sup}
    g_num = g_den = gs_num = gs_den = 0
    test_conv = 0
    # users are disjoint across buckets, so query/user counts add up
    for p in buckets:
        df, st = candidates(_load(p), g1)
        for k_, v in st.items():
            gen[k_] = gen.get(k_, 0) + v
        tr = df[(df.split == 0) & df.next_window_ok]
        rr, ss = tr[tr.label == RETURNED], tr[tr.label == SILENT]
        g_num += int(rr.conv_next.sum()); g_den += len(rr)
        gs_num += int(ss.conv_next.sum()); gs_den += len(ss)
        for s, name in enumerate(sup):
            o, vc = _support(df[df.split == s])
            for k_, v in o.items():
                sup[name][k_] = sup[name].get(k_, 0) + v
            sizes[name] = sizes[name].add(vc, fill_value=0)
        test_conv += len(df[(df.split == 2) & (df.label == CONVERTED)][["u", "i"]].drop_duplicates())
        ev = df[df.split > 0]
        q = ev.u.to_numpy() * SHIFT + ev.t.to_numpy()
        nlev = pd.Series(ev.label.to_numpy()).groupby(q).transform("nunique").to_numpy()
        keep_frames.append(ev[nlev >= 2].drop(columns=["conv_next", "next_window_ok"]))
        del df, tr, ev
    g = g_num / g_den if g_den else math.nan
    g_silent = gs_num / gs_den if gs_den else math.nan
    for name in sup:
        vc = sizes[name].sort_index()
        sup[name]["mean_size"] = float((vc.index * vc).sum() / vc.sum()) if vc.sum() else math.nan
        cum = vc.cumsum() / vc.sum() if vc.sum() else vc
        sup[name]["median_size"] = float(cum.index[np.searchsorted(cum.to_numpy(), 0.5)]) if vc.sum() else math.nan
        sup[name]["p90_size"] = float(cum.index[np.searchsorted(cum.to_numpy(), 0.9)]) if vc.sum() else math.nan
    support = sup
    qm = evaluate(pd.concat(keep_frames, ignore_index=True), g, cfg.k, cfg.seed)
    results = {}
    for s, name in ((1, "val"), (2, "test")):
        part = qm[qm.split == s]
        results[name] = {col: user_macro(part, col, cfg.n_boot, cfg.seed)
                         for col in part.columns if ":" in col}
    summary = {
        "config": {k: v for k, v in asdict(cfg).items() if k != "derived"},
        "derived": {"n_days": g1["n_days"], "bounds": g1["bounds"], "delta": g1["delta"], "window": g1["window"],
                    "warmup": g1["warmup"], "test_len": g1["test_len"], "delay_coverage": g1["delay_coverage"],
                    "n_delays": g1["n_delays"], "delay_quantiles": g1["delay_quantiles"], "g": g,
                    "g_support_returned_train": g_den,
                    "next_window_conversion_rate_silent": g_silent},
        "log": {"events": g1["n_events"], "users": g1["n_users"], "train_items": int(len(g1["train_items"]))},
        "purchase_stats": g1["purchase_stats"],
        "test_converted_pairs": int(test_conv),
        "generation": gen, "support": support, "results": results,
    }
    return summary, qm
