"""CLI: python -m src.uaibench.run <dataset> [options]."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .adapters import convert
from .build import Config, run

ROOT = Path(__file__).resolve().parents[2]
RAW = {
    "retailrocket": ROOT / "data/raw/retailrocket/events.csv",
    "taobao": ROOT / "data/raw/taobao/UserBehavior.csv.zip",
    "ijcai15": ROOT / "data/raw/ijcai15/data_format1.zip",
    "tmall": ROOT / "data/raw/tmall",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(RAW))
    ap.add_argument("--rate", type=float, default=1.0, help="user sample rate (hash-based)")
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--delta", type=int)
    ap.add_argument("--window", type=int)
    ap.add_argument("--double11", action="store_true", help="IJCAI-15: keep the Double 11 day")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--reconvert", action="store_true")
    a = ap.parse_args()

    name = a.dataset + ("_d11" if a.double11 else "") + (f"_r{a.rate:g}" if a.rate < 1 else "")
    proc = ROOT / "data/processed" / name
    meta_path = proc / "convert.json"
    t0 = time.time()
    if a.reconvert or not meta_path.exists():
        meta = convert(a.dataset, RAW[a.dataset], proc, a.rate, a.buckets, include_double11=a.double11)
        meta_path.write_text(json.dumps(meta, indent=2))
    meta = json.loads(meta_path.read_text())
    cfg = Config(delta=a.delta, window=a.window, n_boot=a.n_boot, seed=a.seed)
    summary, qm = run(sorted(proc.glob("bucket_*.npz")), cfg)
    summary["convert"] = meta
    summary["runtime_sec"] = round(time.time() - t0, 1)
    out = ROOT / "evidence/uaibench_v0" / (name + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    qm.to_csv(out / "query_metrics.csv.gz", index=False)
    print(json.dumps({k: summary[k] for k in ("derived", "support", "purchase_stats")}, indent=1, default=float))


if __name__ == "__main__":
    main()
