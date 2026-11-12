"""Aggregate eval_results.json of all datasets and compute rank correlations.

python -m src.uaibench.analyze_results  -> evidence/uaibench_v1/results_summary.json (+ printed tables)
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
RUNS = {"Taobao": "taobao_r0.2", "Tmall": "tmall_r0.015", "IJCAI-15": "ijcai15_r0.3", "RetailRocket": "retailrocket"}
SCORERS = ["recency", "frequency", "cart_first", "popularity", "gbdt", "mbstr", "pbat", "mbht", "missl", "mbsrec", "mbgen"]
NAMES = {"recency": "Recency", "frequency": "Frequency", "cart_first": "Cart-first", "popularity": "Popularity",
         "gbdt": "GBDT-UAI", "mbstr": "MB-STR", "pbat": "PBAT", "mbht": "MBHT", "missl": "MISSL", "mbsrec": "MBSRec", "mbgen": "MBGen"}


def main():
    out = {"datasets": {}}
    for ds, run in RUNS.items():
        f = ROOT / "data/bench" / run / "eval_results.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        rows = {}
        for s in SCORERS:
            if f"{s}:cc_ndcg_conv" not in r["uai"]:
                continue
            rows[s] = {
                "conv": r["uai"][f"{s}:cc_ndcg_conv"], "engage": r["uai"][f"{s}:cc_ndcg_engage"],
                "cond_auc": r["uai"].get(f"{s}:auc_conditional"),
                "std_ndcg": r["std"][s]["ndcg10"], "stdfull_ndcg": r["stdfull"][s]["ndcg10"],
                "std_hr": r["std"][s]["hr10"], "stdfull_hr": r["stdfull"][s]["hr10"],
            }
        present = list(rows)
        tau = {}
        for a, b in [("std_ndcg", "conv"), ("stdfull_ndcg", "conv"), ("std_ndcg", "engage"), ("stdfull_ndcg", "engage"),
                     ("conv", "engage"), ("std_ndcg", "stdfull_ndcg")]:
            x = [rows[s][a]["mean"] for s in present]
            y = [rows[s][b]["mean"] for s in present]
            tau[f"{a}~{b}"] = float(kendalltau(x, y)[0])
        neural = [s for s in present if s in ("mbstr", "pbat", "mbht", "missl", "mbsrec", "mbgen", "gbdt")]
        tau_learned = {}
        if len(neural) >= 3:
            for a, b in [("std_ndcg", "conv"), ("stdfull_ndcg", "conv")]:
                tau_learned[f"{a}~{b}"] = float(kendalltau([rows[s][a]["mean"] for s in neural],
                                                           [rows[s][b]["mean"] for s in neural])[0])
        out["datasets"][ds] = {"run": run, "rows": rows, "kendall_all": tau, "kendall_learned": tau_learned,
                               "n_scorers": len(present), "std_instances": r.get("std_instances"),
                               "gbdt": r.get("gbdt")}
        print(f"\n== {ds} ({len(present)} scorers, std instances {r.get('std_instances')})")
        print(f"{'scorer':14s} {'conv':>16s} {'engage':>16s} {'condAUC':>8s} {'std@10':>8s} {'full@10':>8s}")
        for s in present:
            v = rows[s]
            ca = v["cond_auc"]["mean"] if v["cond_auc"] else float("nan")
            print(f"{NAMES[s]:14s} {v['conv']['mean']:.3f} [{v['conv']['lo']:.2f},{v['conv']['hi']:.2f}] "
                  f"{v['engage']['mean']:.3f} [{v['engage']['lo']:.2f},{v['engage']['hi']:.2f}] {ca:8.3f} "
                  f"{v['std_ndcg']['mean']:8.3f} {v['stdfull_ndcg']['mean']:8.3f}")
        print("kendall:", {k: round(v, 2) for k, v in tau.items()}, "learned:", {k: round(v, 2) for k, v in tau_learned.items()})
    (ROOT / "evidence/uaibench_v1").mkdir(parents=True, exist_ok=True)
    (ROOT / "evidence/uaibench_v1/results_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
