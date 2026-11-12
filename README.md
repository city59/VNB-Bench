# VNB-Bench

A benchmark of multi-behavior sequential recommenders on unconverted auxiliary interactions.

## Setup

```bash
pip install -r requirements.txt
bash scripts/fetch_third_party.sh   # official model code at pinned commits
```

## Data

The raw logs are not redistributed. Download them into `data/raw/`:

| Log | Source | Files |
|---|---|---|
| Taobao | https://tianchi.aliyun.com/dataset/649 | `taobao/UserBehavior.csv.zip` |
| Tmall | https://tianchi.aliyun.com/dataset/140281 | `tmall/tianchi_2014002_rec_tmall_log.part{a,b,c}.zip` |
| IJCAI-15 | https://tianchi.aliyun.com/dataset/42 | `ijcai15/data_format1.zip` |
| RetailRocket | https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset | `retailrocket/events.csv` |

Check the files with `cd data/raw && sha256sum -c SHA256SUMS`.

## Run

```bash
# build the benchmark
python -m src.uaibench.run taobao --rate 0.2
python -m src.uaibench.run tmall --rate 0.015
python -m src.uaibench.run ijcai15 --rate 0.3
python -m src.uaibench.run retailrocket
for r in taobao_r0.2 tmall_r0.015 ijcai15_r0.3 retailrocket; do python -m src.uaibench.export $r; done

# train and score a model (mbstr, pbat, mbht, missl, mbgen)
python -m src.uaibench.train_mbsr tmall_r0.015 mbgen 2026

# evaluate
python -m src.uaibench.evaluate_models tmall_r0.015 1000
python -m src.uaibench.raw_vs_cc tmall_r0.015
python -m src.uaibench.analyze_results
```

`python -m src.uaibench.test_metrics` checks the metric implementation. MBSRec is not included here; see its official
release at https://github.com/Shereen-Elsayed/MBSRec.

Additional results (standard-protocol ranks, conversion fine-tuning, static-graph models) are in
[results/README.md](results/README.md).
