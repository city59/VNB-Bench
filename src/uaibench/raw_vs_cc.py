"""Raw NDCG@10 vs ccNDCG@10 on the VNB-Bench conversion and re-engagement tasks (test period).

python -m src.uaibench.raw_vs_cc <run> -> data/bench/<run>/raw_vs_cc.json

Same queries, scorers, GBDT fit, and user-macro averaging as evaluate_models. Also records the per-query chance
level of raw NDCG (its value under a uniformly random order) so the two metrics can be compared.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .build import CONVERTED, SILENT, SHIFT, scores as heuristic_scores, user_macro
from .evaluate_models import NEURAL, gbdt_features
from .metrics import cc_ndcg, ndcg, random_ndcg

ROOT = Path(__file__).resolve().parents[2]
run = sys.argv[1]
d_dir = ROOT / "data/bench" / run
tr = pd.DataFrame(dict(np.load(d_dir / "uai_train.npz")))
ev = pd.DataFrame(dict(np.load(d_dir / "uai_eval.npz")))
bc = np.load(ROOT / "data/processed" / run / "buy_counts_all_users.npz")
o = np.lexsort((bc["day"], bc["item"]))
bkey = bc["item"][o].astype(np.int64) * 65536 + bc["day"][o]
bcum = np.r_[0, np.cumsum(bc["count"][o])]
lo = np.searchsorted(bkey, tr.i.to_numpy().astype(np.int64) * 65536, side="left")
hi = np.searchsorted(bkey, tr.i.to_numpy().astype(np.int64) * 65536 + tr.t.to_numpy(), side="left")
tr["pop"] = bcum[hi] - bcum[lo]
y = (tr.label.to_numpy() == CONVERTED).astype(int)
gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True,
                                    validation_fraction=0.1, n_iter_no_change=20, random_state=2026).fit(gbdt_features(tr), y)
test = ev[ev.split == 2].reset_index(drop=True)
sc = heuristic_scores(test, 2026)
sc["gbdt"] = gb.predict_proba(gbdt_features(test))[:, 1]
for k in NEURAL:
    f = d_dir / f"scores_{k}_s2026.npz"
    if f.exists():
        sc[k] = np.load(f)["uai_split2"].astype(np.float64)

q = test.u.to_numpy().astype(np.int64) * SHIFT + test.t.to_numpy()
order = np.argsort(q, kind="stable")
q, lab, u = q[order], test.label.to_numpy()[order], test.u.to_numpy()[order]
sc = {n: s[order] for n, s in sc.items()}
starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
ends = np.r_[starts[1:], len(q)]
rows = {"conv": [], "engage": []}
for a, b in zip(starts, ends):
    L_ = lab[a:b]
    for task, g in (("conv", (L_ == CONVERTED).astype(float)), ("engage", (L_ != SILENT).astype(float))):
        if not 0 < g.sum() < len(g):
            continue
        r = {"u": int(u[a]), "n": int(b - a), "share": float(g.mean()), "chance": random_ndcg(g)}
        for n, s in sc.items():
            r[f"{n}:ndcg"] = ndcg(g, s[a:b])
            r[f"{n}:cc"] = cc_ndcg(g, s[a:b])
        rows[task].append(r)
out = {"run": run}
for task, rs in rows.items():
    df = pd.DataFrame(rs)
    df.to_csv(d_dir / f"raw_vs_cc_{task}.csv.gz", index=False)
    res = {"queries": len(df), "chance": user_macro(df, "chance", 0, 0)["mean"],
           "median_n": float(df.n.median()), "share": user_macro(df, "share", 0, 0)["mean"]}
    for n in sc:
        res[n] = {m: user_macro(df, f"{n}:{m}", 1000, 2026) for m in ("ndcg", "cc")}
    out[task] = res
(d_dir / "raw_vs_cc.json").write_text(json.dumps(out, indent=1))
print(run, {t: {"chance": round(out[t]["chance"], 3),
                **{n: (round(out[t][n]["ndcg"]["mean"], 3), round(out[t][n]["cc"]["mean"], 3)) for n in sc}}
            for t in ("conv", "engage")})
