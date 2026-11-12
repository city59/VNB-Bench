"""Export model-ready data for one processed run into data/bench/<run>/.

Files
  meta.json      configuration, split bounds, vocabulary sizes
  events.npz     all events of the sampled users with training-period items: u (dense), i (1..N), b (1..4), ts, day
  uai_eval.npz   val/test candidates of UAIBench queries with at least two outcome classes (+ features)
  uai_train.npz  training-period candidates for supervised UAI scorers (all candidates of queries with a
                 conversion, plus a 5% sample of the remaining candidates)
  std_eval.npz   standard protocol: first val/test-period purchase per user, 1 positive + 99 negatives that
                 exclude every item the user interacted with before the purchase

python -m src.uaibench.export taobao_r0.2
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import BUY
from .build import SHIFT, Config, _load, candidates, pass1

ROOT = Path(__file__).resolve().parents[2]
FEATS = ["cnt_w", "cnt_all", "cart_w", "cart_all", "fav_all", "last_ts", "first_day", "pop"]


def main() -> None:
    run = sys.argv[1]
    n_neg = 99
    rng = np.random.default_rng(2026)
    proc = ROOT / "data/processed" / run
    out = ROOT / "data/bench" / run
    out.mkdir(parents=True, exist_ok=True)
    buckets = sorted(proc.glob("bucket_*.npz"))
    g1 = pass1(buckets, Config())
    items = g1["train_items"]  # sorted raw ids
    n_items = len(items)
    bounds = g1["bounds"]
    ev_parts, eval_parts, train_parts, std_parts = [], [], [], []
    user_ids = []
    for p in buckets:
        ev = _load(p)
        df, _ = candidates(ev, g1)
        df["item"] = np.searchsorted(items, df.i.to_numpy()) + 1
        q = df.u.to_numpy() * SHIFT + df.t.to_numpy()
        nlev = pd.Series(df.label.to_numpy()).groupby(q).transform("nunique").to_numpy()
        hasc = pd.Series((df.label == 2).to_numpy()).groupby(q).transform("max").to_numpy()
        cols = ["u", "i", "item", "t", "split", "label"] + FEATS
        eval_parts.append(df.loc[(df.split > 0) & (nlev >= 2), cols])
        tr = df[(df.split == 0)]
        keep = hasc[(df.split == 0).to_numpy()].astype(bool) | (rng.random(len(tr)) < 0.05)
        train_parts.append(tr.loc[keep, cols])
        # model-ready events: training-vocabulary items only
        pos = np.searchsorted(items, ev.i.to_numpy())
        pos_c = np.minimum(pos, n_items - 1)
        known = items[pos_c] == ev.i.to_numpy()
        e = pd.DataFrame({"u": ev.u.to_numpy()[known], "i": pos_c[known] + 1, "b": ev.b.to_numpy()[known] + 1,
                          "ts": ev.ts.to_numpy()[known], "day": ev.day.to_numpy()[known]})
        e = e.sort_values(["u", "ts", "b"], kind="stable")
        ev_parts.append(e)
        user_ids.append(np.unique(ev.u.to_numpy()))
        # standard protocol instances
        for s in (1, 2):
            lo, hi = bounds[s]
            buys = e[(e.b == BUY + 1) & (e.day >= lo) & (e.day < hi)]
            first = buys.sort_values(["u", "ts"]).drop_duplicates("u")
            if first.empty:
                continue
            hist = e.merge(first[["u", "ts"]].rename(columns={"ts": "pts"}), on="u")
            hist = hist[hist.ts < hist.pts]
            seen = hist.groupby("u").i.apply(lambda x: set(x.tolist())).to_dict()
            negs = np.zeros((len(first), n_neg), np.int64)
            for k, (u, it) in enumerate(zip(first.u.to_numpy(), first.i.to_numpy())):
                bad = seen.get(u, set()) | {it}
                row, filled = [], 0
                while filled < n_neg:
                    cand = rng.integers(1, n_items + 1, size=2 * n_neg)
                    for c in cand:
                        if c not in bad and c not in row:
                            row.append(int(c))
                            filled += 1
                            if filled == n_neg:
                                break
                negs[k] = row
            std_parts.append(pd.DataFrame({"u": first.u.to_numpy(), "ts": first.ts.to_numpy(), "day": first.day.to_numpy(),
                                           "split": s, "pos": first.i.to_numpy(),
                                           "seen_before": [it in seen.get(u, set()) for u, it in zip(first.u, first.i)]})
                             .assign(**{f"n{j}": negs[:, j] for j in range(n_neg)}))
    users = np.unique(np.concatenate(user_ids))
    E = pd.concat(ev_parts, ignore_index=True)
    umap = lambda x: np.searchsorted(users, x) + 1  # dense user ids, 0 = padding
    np.savez(out / "events.npz", u=umap(E.u.to_numpy()), i=E.i.to_numpy(np.int64), b=E.b.to_numpy(np.int8),
             ts=E.ts.to_numpy(np.int64), day=E.day.to_numpy(np.int32))
    for name, parts in (("uai_eval", eval_parts), ("uai_train", train_parts)):
        D = pd.concat(parts, ignore_index=True)
        np.savez(out / f"{name}.npz", **{c: D[c].to_numpy() for c in D.columns}, uid=umap(D.u.to_numpy()))
    S = pd.concat(std_parts, ignore_index=True)
    np.savez(out / "std_eval.npz", uid=umap(S.u.to_numpy()), ts=S.ts.to_numpy(), day=S.day.to_numpy(), split=S.split.to_numpy(),
             pos=S.pos.to_numpy(), seen_before=S.seen_before.to_numpy(),
             negs=S[[f"n{j}" for j in range(n_neg)]].to_numpy(np.int64))
    meta = {"run": run, "n_items": int(n_items), "n_users": int(len(users)), "bounds": bounds, "delta": g1["delta"],
            "window": g1["window"], "n_events": int(len(E)),
            "uai_eval_rows": int(sum(len(x) for x in eval_parts)), "uai_train_rows": int(sum(len(x) for x in train_parts)),
            "std_instances": {str(s): int((S.split == s).sum()) for s in (1, 2)},
            "std_pos_seen_before_share": float(S.seen_before.mean())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
