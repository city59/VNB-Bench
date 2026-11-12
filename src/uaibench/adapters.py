"""Raw-log adapters: map each dataset to canonical events.

Canonical behavior codes: VIEW=0, FAV=1, CART=2 (auxiliary) and BUY=3 (target).
`day` is a local-calendar day index starting at 0; `ts` is seconds (day-level
logs use the start of the day). Users are sampled and bucketed by a fixed
multiplicative hash of the raw user id, so sampling never looks at outcomes.
"""

from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

VIEW, FAV, CART, BUY = 0, 1, 2, 3


def user_hash(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)


class BuyCounter:
    """Purchase counts per (item, epoch day) over ALL users, before user sampling.

    Item popularity is a global training-period statistic; computing it on the
    sampled users only would make it degenerate at small sampling rates.
    """

    def __init__(self):
        self.keys, self.counts = [], []

    def add(self, items: np.ndarray, days: np.ndarray) -> None:
        if len(items) == 0:
            return
        k, c = np.unique(np.asarray(items, np.int64) * 65536 + np.asarray(days, np.int64), return_counts=True)
        self.keys.append(k)
        self.counts.append(c)
        if len(self.keys) > 64:
            self._compact()

    def _compact(self) -> None:
        k = np.concatenate(self.keys)
        c = np.concatenate(self.counts)
        uk, inv = np.unique(k, return_inverse=True)
        self.keys, self.counts = [uk], [np.bincount(inv, weights=c).astype(np.int64)]

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.keys:
            z = np.zeros(0, np.int64)
            return z, z, z
        self._compact()
        k, c = self.keys[0], self.counts[0]
        return k // 65536, k % 65536, c


def _emit(chunk: pd.DataFrame, rate: float, n_buckets: int, store: dict) -> None:
    h = user_hash(chunk["u"].to_numpy())
    keep = (h % np.uint64(10_000)) < np.uint64(int(round(rate * 10_000)))
    chunk = chunk.loc[keep]
    bucket = ((h[keep] >> np.uint64(14)) % np.uint64(n_buckets)).astype(np.int64)
    for b in np.unique(bucket):
        part = chunk.loc[bucket == b]
        store.setdefault(int(b), []).append(
            (
                part["u"].to_numpy(np.int64),
                part["i"].to_numpy(np.int64),
                part["b"].to_numpy(np.int8),
                part["day"].to_numpy(np.int16),
                part["ts"].to_numpy(np.int64),
            )
        )


def _chunks_retailrocket(path: Path, chunksize: int):
    codes = {"view": VIEW, "addtocart": CART, "transaction": BUY}
    for c in pd.read_csv(path, usecols=["timestamp", "visitorid", "event", "itemid"], chunksize=chunksize):
        ts = c["timestamp"].to_numpy(np.int64) // 1000
        yield pd.DataFrame(
            {"u": c["visitorid"], "i": c["itemid"], "b": c["event"].map(codes).astype(np.int8), "ts": ts}
        ).assign(day=lambda d: (d.ts // 86400).astype(np.int64))  # UTC days


TAOBAO_START = int(dt.datetime(2017, 11, 25, tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp())
TAOBAO_END = TAOBAO_START + 9 * 86400  # official window: 2017-11-25 .. 2017-12-03 (UTC+8)


def _chunks_taobao(path: Path, chunksize: int):
    codes = {"pv": VIEW, "fav": FAV, "cart": CART, "buy": BUY}
    zf = zipfile.ZipFile(path)
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    with zf.open(name) as fh:
        reader = pd.read_csv(
            fh, header=None, names=["u", "i", "cat", "beh", "ts"], usecols=["u", "i", "beh", "ts"],
            dtype={"u": np.int64, "i": np.int64, "beh": "category", "ts": np.int64}, chunksize=chunksize,
        )
        for c in reader:
            c = c[(c.ts >= TAOBAO_START) & (c.ts < TAOBAO_END)]
            yield pd.DataFrame(
                {"u": c.u, "i": c.i, "b": c.beh.astype(str).map(codes).astype(np.int8), "ts": c.ts}
            ).assign(day=lambda d: (d.ts + 8 * 3600) // 86400)  # Beijing calendar days


def _chunks_ijcai15(path: Path, chunksize: int, include_double11: bool):
    codes = {0: VIEW, 1: CART, 2: BUY, 3: FAV}
    zf = zipfile.ZipFile(path)
    name = [n for n in zf.namelist() if n.endswith("user_log_format1.csv")][0]
    with zf.open(name) as fh:
        for c in pd.read_csv(fh, usecols=["user_id", "item_id", "time_stamp", "action_type"], chunksize=chunksize):
            mmdd = c["time_stamp"].to_numpy(np.int64)
            # explicit log end: a handful of stray 11-12 records would otherwise extend the
            # last split with empty days and silently turn late windows into Silent labels
            keep = mmdd <= (1111 if include_double11 else 1110)
            c, mmdd = c[keep], mmdd[keep]
            days = pd.to_datetime({"year": 2015, "month": mmdd // 100, "day": mmdd % 100})
            day = (days - pd.Timestamp("1970-01-01")).dt.days.to_numpy(np.int64)
            yield pd.DataFrame(
                {"u": c.user_id.to_numpy(), "i": c.item_id.to_numpy(), "b": c.action_type.map(codes).astype(np.int8).to_numpy(),
                 "ts": day * 86400, "day": day}
            )


def _chunks_tmall(raw_dir: Path, chunksize: int, rate: float, buys: "BuyCounter | None" = None):
    """Rec-Tmall log (Tianchi 140281), full files: '\\x01'-separated, no header,
    item_id (int), user_id ('u…'), action, vtime 'YYYY-mm-dd HH:MM:SS' (Beijing time).
    (The 155 KB web sample uses a different, comma-separated layout.)

    The log has ~1.33 billion rows, so users are filtered by hash before the
    timestamp column is parsed.
    """
    codes = {"click": VIEW, "collect": FAV, "cart": CART, "alipay": BUY}
    thr = np.uint64(int(round(rate * 10_000)))
    for part in sorted(raw_dir.glob("tianchi_2014002_rec_tmall_log.part*.zip")):
        zf = zipfile.ZipFile(part)
        for name in sorted(n for n in zf.namelist() if not n.endswith("/")):
            with zf.open(name) as fh:
                for c in pd.read_csv(fh, sep="\x01", header=None, names=["item_id", "user_id", "action", "vtime"],
                                     dtype={"item_id": np.int64, "user_id": str, "action": str, "vtime": str},
                                     chunksize=chunksize):
                    if buys is not None:
                        bb = c[c.action == "alipay"]
                        bd = pd.to_datetime(bb.vtime, format="%Y-%m-%d %H:%M:%S", errors="coerce")
                        okb = bd.notna().to_numpy()
                        buys.add(bb.item_id.to_numpy(np.int64)[okb],
                                 ((bd[okb] - pd.Timestamp("1970-01-01")).dt.days).to_numpy(np.int64))
                    u = c.user_id.str.slice(1).astype(np.int64).to_numpy()
                    keep = (user_hash(u) % np.uint64(10_000)) < thr
                    c, u = c[keep], u[keep]
                    ts = pd.to_datetime(c.vtime, format="%Y-%m-%d %H:%M:%S", errors="coerce")
                    ok = ts.notna().to_numpy() & c.action.isin(list(codes)).to_numpy()
                    c, u, ts = c[ok], u[ok], ts[ok]
                    sec = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(np.int64)
                    yield pd.DataFrame({"u": u, "i": c.item_id.to_numpy(np.int64),
                                        "b": c.action.map(codes).astype(np.int8).to_numpy(),
                                        "ts": sec - 8 * 3600, "day": sec // 86400})  # local calendar days


def convert(dataset: str, raw: Path, out_dir: Path, rate: float, n_buckets: int,
            chunksize: int = 2_000_000, include_double11: bool = False, min_day_share: float = 0.01) -> dict:
    buys = BuyCounter()
    if dataset == "retailrocket":
        chunks = _chunks_retailrocket(raw, chunksize)
    elif dataset == "taobao":
        chunks = _chunks_taobao(raw, chunksize)
    elif dataset == "ijcai15":
        chunks = _chunks_ijcai15(raw, chunksize, include_double11)
    elif dataset == "tmall":
        chunks = _chunks_tmall(raw, chunksize, rate, buys)
    else:
        raise ValueError(dataset)
    store: dict = {}
    n_raw = 0
    for c in chunks:
        n_raw += len(c)
        if dataset != "tmall":  # tmall counts purchases inside the generator, before sampling
            m = (c["b"] == BUY).to_numpy()
            buys.add(c["i"].to_numpy()[m], c["day"].to_numpy()[m])
        _emit(c, rate, n_buckets, store)
    out_dir.mkdir(parents=True, exist_ok=True)
    # data-quality guard: drop leading/trailing days whose volume is < min_day_share of the median day
    all_days = np.concatenate([p[3].astype(np.int64) for parts in store.values() for p in parts])
    lo_day = int(all_days.min())
    vol = np.bincount(all_days - lo_day)
    core = np.flatnonzero(vol >= min_day_share * np.median(vol[vol > 0]))
    day_first, day_last = lo_day + int(core[0]), lo_day + int(core[-1])
    dropped = {"leading_days": int(core[0]), "trailing_days": int(len(vol) - 1 - core[-1]),
               "events": int(((all_days < day_first) | (all_days > day_last)).sum())}
    del all_days
    day0 = day_first
    n_kept = 0
    for b, parts in store.items():
        u, i, beh, day, ts = (np.concatenate([p[k] for p in parts]) for k in range(5))
        day = day.astype(np.int64)
        m = (day >= day_first) & (day <= day_last)
        df = pd.DataFrame({"u": u[m], "i": i[m], "b": beh[m], "day": day[m] - day0, "ts": ts[m]})
        df = df.drop_duplicates(["u", "i", "b", "ts"])  # exact duplicate tuples only
        n_kept += len(df)
        np.savez(out_dir / f"bucket_{b:03d}.npz", **{k: df[k].to_numpy() for k in df.columns})
    bi, bd, bc = buys.result()
    m = (bd >= day_first) & (bd <= day_last)
    np.savez(out_dir / "buy_counts_all_users.npz", item=bi[m], day=bd[m] - day0, count=bc[m])
    return {"dataset": dataset, "rows_in_window": n_raw, "rows_kept_after_sampling_dedup": n_kept,
            "user_sample_rate": rate, "n_buckets": n_buckets, "day0_epoch_day": day0,
            "n_days": day_last - day_first + 1, "trimmed": dropped,
            "purchases_all_users": int(bc[m].sum()), "include_double11": include_double11}
