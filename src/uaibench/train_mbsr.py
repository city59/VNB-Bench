"""Train an MBSR model and score VNB-Bench and the standard protocols.

  python -m src.uaibench.train_mbsr <run> <model> [seed]
  model in {mbstr, pbat, mbht, missl, mbgen}

Training follows each official release (mbsr_models.*.hp): per-dataset configuration, training-instance
sampling, loss, gradient accumulation, and early stopping after hp["patience"] epochs without improvement on
each protocol's validation metric, within hp["max_epochs"]. Validation uses fixed subsamples of at most 10,000
queries / instances; test scoring uses all of them. Each protocol keeps the checkpoint that is best on its own
validation metric.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .mbsr_models import BUY, MODELS
from .metrics import cc_ndcg

ROOT = Path(__file__).resolve().parents[2]
MIN_DELTA = 0.0  # Lightning EarlyStopping default; RecBole counts any increase
VAL_CAP = 10000
MEM_FRACTION = {"mbgen": 0.4, "mbht": 0.35, "mbstr": 0.3, "pbat": 0.3}  # of 16 GB
SMOKE = int(os.environ.get("UAI_SMOKE_STEPS", 0))  # >0: time this many training batches and one validation, save nothing


class Data:
    def __init__(self, run, seed):
        d = ROOT / "data/bench" / run
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text())
        e = np.load(d / "events.npz")
        o = np.lexsort((e["b"], e["ts"], e["u"]))
        self.u, self.i, self.b = e["u"][o].astype(np.int64), e["i"][o].astype(np.int64), e["b"][o].astype(np.int64)
        self.ts, self.day = e["ts"][o].astype(np.int64), e["day"][o].astype(np.int64)
        self.n_items, self.n_users = self.meta["n_items"], self.meta["n_users"]
        self.train_end = self.meta["bounds"][0][1]
        self.ts0 = int(self.ts.min())
        self.key_day = self.u * 100000 + self.day
        self.key_ts = self.u * (1 << 31) + (self.ts - self.ts0)
        self.ustart = np.searchsorted(self.u, np.arange(self.n_users + 2))
        tr = self.day < self.train_end
        self.tr_users, first = np.unique(self.u[tr], return_index=True)
        tr_pos = np.flatnonzero(tr)
        self.tr_s = tr_pos[first]
        self.tr_e = np.r_[tr_pos[first[1:] - 1] + 1, tr_pos[-1] + 1]
        self.tr_buy = np.flatnonzero(tr & (self.b == BUY))
        self.uai = dict(np.load(d / "uai_eval.npz"))
        self.std = dict(np.load(d / "std_eval.npz"))
        rng = np.random.default_rng(seed)
        # fixed validation subsamples
        m = self.uai["split"] == 1
        q = self.uai["uid"].astype(np.int64) * 100000 + self.uai["t"]
        uq = np.unique(q[m])
        keep = set(rng.choice(uq, min(VAL_CAP, len(uq)), replace=False).tolist())
        self.uai_val_mask = m & np.isin(q, list(keep))
        sv = np.flatnonzero(self.std["split"] == 1)
        self.std_val_idx = np.sort(rng.choice(sv, min(VAL_CAP, len(sv)), replace=False))

    def gather(self, uids, ends, L):
        """Left-padded last-L events of each user before global position `ends`."""
        uids, ends = np.asarray(uids, np.int64), np.asarray(ends, np.int64)
        idx = ends[:, None] - L + np.arange(L)[None, :]
        valid = idx >= self.ustart[uids][:, None]
        idx = np.clip(idx, 0, len(self.i) - 1)
        return np.where(valid, self.i[idx], 0), np.where(valid, self.b[idx], 0)

    def _train_users(self, min_len=2):
        ok = (self.tr_e - self.tr_s) >= min_len
        return self.tr_users[ok], self.tr_s[ok], self.tr_e[ok]

    def _window(self, users, s, ends, L):
        it, bh = self.gather(users, ends, L)
        it = np.where(np.arange(L)[None, :] >= L - (ends - s)[:, None], it, 0)  # never before the training start
        return it, np.where(it > 0, bh, 0)

    def cloze_windows(self, L, batch, rng):
        """MB-STR / PBAT RecTrainDataset: every training user once per epoch in random order; the last L events of
        the training sequence, or for longer sequences with probability 0.2 a uniformly placed window of L."""
        users, s, e = self._train_users()
        n = e - s
        rand = (n > L) & (rng.random(len(n)) >= 0.8)
        start = np.where(rand, (rng.random(len(n)) * np.maximum(n - L + 1, 1)).astype(np.int64), 0)
        ends = np.where(rand, s + start + L, e)
        perm = rng.permutation(len(users))
        for k in range(0, len(perm), batch):
            sel = perm[k:k + batch]
            it, bh = self._window(users[sel], s[sel], ends[sel], L)
            yield it, bh, users[sel]

    def purchase_instances(self, L, batch, rng):
        """One random training purchase per user per epoch, with its preceding events as history."""
        pos = self.tr_buy[rng.permutation(len(self.tr_buy))]
        u = self.u[pos]
        _, first = np.unique(u, return_index=True)
        pos = pos[first]
        u = self.u[pos]
        ok = pos > self.ustart[u]
        pos, u = pos[ok], u[ok]
        perm = rng.permutation(len(pos))
        for k in range(0, len(perm), batch):
            sel = perm[k:k + batch]
            it, bh = self.gather(u[sel], pos[sel], L)
            yield it, bh, u[sel], self.i[pos[sel]]


def to_dev(batch, dev):
    return tuple(torch.as_tensor(np.asarray(x, np.int64), device=dev) for x in batch)


def _is_oom(e):
    return isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(e)


def _state_halving(model, it_t, bh_t, u_t):
    """model.state on a batch; when the shared GPU runs short of memory, retry on halves (same result)."""
    try:
        return model.state(it_t, bh_t, u_t)
    except RuntimeError as e:
        if not _is_oom(e) or len(it_t) == 1:
            raise
        torch.cuda.empty_cache()
        m = len(it_t) // 2
        return torch.cat([_state_halving(model, it_t[:m], bh_t[:m], u_t[:m]),
                          _state_halving(model, it_t[m:], bh_t[m:], u_t[m:])])


@torch.no_grad()
def states(model, data, uids, ends, dev, bs=None):
    model.eval()
    bs = bs or getattr(model, "infer_bs", 512)
    out = []
    for k in range(0, len(uids), bs):
        it, bh = data.gather(uids[k:k + bs], ends[k:k + bs], model.L)
        it_t, bh_t, u_t = to_dev((it, bh, uids[k:k + bs]), dev)
        h = torch.zeros((len(it), 1), device=dev)
        nz = (it_t > 0).any(1)
        if nz.any():
            hz = _state_halving(model, it_t[nz], bh_t[nz], u_t[nz])
            h = torch.zeros((len(it), hz.shape[1]), device=dev)
            h[nz] = hz
        out.append(h)  # users without any prior event keep a zero state: constant (tied) scores
    width = max(x.shape[1] for x in out)
    return torch.cat([torch.nn.functional.pad(x, (0, width - x.shape[1])) for x in out])


@torch.no_grad()
def score_rows(model, h, qidx, cands, dev, bs=32768):
    out = []
    cands = np.asarray(cands, np.int64)
    if cands.ndim == 1:
        cands = cands[:, None]
    for k in range(0, len(qidx), bs):
        hh = h[torch.as_tensor(np.asarray(qidx[k:k + bs], np.int64), device=dev)]
        cc = torch.as_tensor(cands[k:k + bs], device=dev)
        zero = (hh == 0).all(1)
        s = _score_halving(model, hh, cc) if (~zero).any() else torch.zeros(cc.shape, device=dev)
        s[zero] = 0.0
        out.append(s.float().cpu())
    return torch.cat(out).numpy()


def _full_halving(model, hh):
    try:
        return model.full(hh)
    except RuntimeError as e:
        if not _is_oom(e) or len(hh) == 1:
            raise
        torch.cuda.empty_cache()
        m = len(hh) // 2
        return torch.cat([_full_halving(model, hh[:m]), _full_halving(model, hh[m:])])


def _score_halving(model, hh, cc):
    try:
        return model.score(hh, cc)
    except RuntimeError as e:
        if not _is_oom(e) or len(hh) == 1:
            raise
        torch.cuda.empty_cache()
        m = len(hh) // 2
        return torch.cat([_score_halving(model, hh[:m], cc[:m]), _score_halving(model, hh[m:], cc[m:])])


def uai_scores(model, data, dev, mask):
    if hasattr(model, "uai_scores"):  # MBGen scores by generation, see mbgen.py
        return model.uai_scores(data, dev, mask)
    U = data.uai
    uid, t, items = U["uid"][mask].astype(np.int64), U["t"][mask], U["item"][mask]
    q = uid * 100000 + t
    uq, qidx = np.unique(q, return_inverse=True)
    ends = np.searchsorted(data.key_day, uq, side="right")
    h = states(model, data, uq // 100000, ends, dev)
    return score_rows(model, h, qidx, items, dev).reshape(-1)


def std_scores(model, data, dev, idx):
    if hasattr(model, "std_scores"):  # MBGen scores by generation, see mbgen.py
        return model.std_scores(data, dev, idx)
    S = data.std
    uid, ts = S["uid"][idx].astype(np.int64), S["ts"][idx].astype(np.int64)
    ends = np.searchsorted(data.key_ts, uid * (1 << 31) + (ts - data.ts0), side="left")
    h = states(model, data, uid, ends, dev)
    cands = np.concatenate([S["pos"][idx][:, None], S["negs"][idx]], 1)
    return score_rows(model, h, np.arange(len(idx)), cands, dev), h, ends


@torch.no_grad()
def std_full_rank(model, data, dev, idx):
    if hasattr(model, "std_full_rank"):  # MBGen scores by generation, see mbgen.py
        return model.std_full_rank(data, dev, idx)
    S = data.std
    _, h, ends = std_scores(model, data, dev, idx)
    uid, pos = S["uid"][idx].astype(np.int64), S["pos"][idx].astype(np.int64)
    ranks = np.zeros(len(idx))
    for k in range(0, len(idx), 32):
        hh = h[k:k + 32]
        sc = _full_halving(model, hh)
        p = torch.as_tensor(pos[k:k + 32] - 1, device=dev)
        ps = sc.gather(1, p.view(-1, 1))
        for r in range(len(hh)):
            s0, e0 = data.ustart[uid[k + r]], ends[k + r]
            bought = data.i[s0:e0][data.b[s0:e0] == BUY] - 1
            bought = bought[bought != p[r].item()]
            if len(bought):
                sc[r, torch.as_tensor(bought, device=dev)] = -float("inf")
        ranks[k:k + len(hh)] = ((sc > ps).sum(1).float() + 0.5 * ((sc == ps).sum(1).float() - 1)).cpu().numpy()
    return ranks


def uai_val_metric(data, scores, mask):
    U = data.uai
    uid, t, lab = U["uid"][mask], U["t"][mask], U["label"][mask]
    q = uid.astype(np.int64) * 100000 + t
    o = np.argsort(q, kind="stable")
    q, lab, sc, uid = q[o], lab[o], scores[o], uid[o]
    st = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    en = np.r_[st[1:], len(q)]
    per = {}
    for a, b in zip(st, en):
        y = (lab[a:b] == 2).astype(float)
        if 0 < y.sum() < len(y):
            per.setdefault(uid[a], []).append(cc_ndcg(y, sc[a:b]))
    return float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")


def std_val_metric(s):
    gt = (s[:, 1:] > s[:, :1]).sum(1)
    ties = (s[:, 1:] == s[:, :1]).sum(1)
    nd = np.zeros(len(s))
    for j, (g, t) in enumerate(zip(gt, ties)):
        pos = np.arange(g, g + t + 1)
        nd[j] = np.mean(np.where(pos < 10, 1 / np.log2(pos + 2.0), 0.0))
    return float(nd.mean())


def _cpu_state(model):
    return {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}


def train_epoch(model, opt, data, rng, dev):
    """One epoch of the release's training loop; hp["accum"] micro-batches per optimizer step (Lightning's
    accumulate_grad_batches times the number of DDP processes, whose gradients are averaged)."""
    if hasattr(model, "fit_epoch"):  # MBGen keeps its own optimizer, schedule, and mixed precision
        return model.fit_epoch(data, rng, dev)
    accum = model.hp.get("accum", 1)
    losses, pending = [], 0
    opt.zero_grad()
    for nb, batch in enumerate(model.batches(data, rng)):
        if SMOKE and nb == SMOKE:
            break
        loss = model.train_loss(to_dev(batch, dev), rng)
        if loss is not None:
            (loss / accum).backward()
            losses.append(loss.item())
        pending += 1
        if pending == accum:
            opt.step()
            opt.zero_grad()
            pending = 0
    if pending:
        opt.step()
        opt.zero_grad()
    return float(np.mean(losses))


def main():
    run, kind = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda")
    # several runs share one 16 GB GPU; a cap keeps PyTorch's cache from overflowing into (slow) shared memory
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("UAI_MEM_FRACTION", MEM_FRACTION.get(kind, 0.35))))
    t0 = time.time()
    data = Data(run, seed)
    ds = run.split("_")[0]
    model = MODELS[kind](kind, data.n_items, data.n_users, ds).to(dev)
    hp = model.hp
    opt = None
    if not hasattr(model, "fit_epoch"):
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"], betas=hp["betas"])
    else:
        model.setup_training(data, dev)  # sets hp["max_epochs"] from the release's step budget
    max_epochs = int(os.environ.get("UAI_MAX_EPOCHS", hp["max_epochs"]))
    patience = int(os.environ.get("UAI_PATIENCE", hp["patience"]))
    best = {"uai": [-1.0, None, -1], "std": [-1.0, None, -1]}
    stale = {"uai": 0, "std": 0}
    log, start = [], 0
    # resumable: a multi-day schedule must survive restarts of the machine
    ck = data.dir / f"ckpt_{kind}_s{seed}.pt"
    if ck.exists():
        c = torch.load(ck, map_location="cpu")
        model.load_state_dict(c["model"])
        if opt is not None:
            opt.load_state_dict(c["opt"])
        else:
            model.load_state_dict_train(c["opt"])
        best, stale, log, start = c["best"], c["stale"], c["log"], c["epoch"] + 1
        rng = c["rng"]
        try:
            torch.set_rng_state(c["torch_rng"])
            torch.cuda.set_rng_state(c["cuda_rng"])
        except RuntimeError:  # checkpoint written by another PyTorch version (run moved between machines)
            torch.manual_seed(seed + c["epoch"] + 1)
            print("torch RNG state incompatible; reseeded", flush=True)
        t0 -= log[-1]["sec"]
        print("resumed after epoch", c["epoch"], flush=True)
    for ep in range(start, max_epochs):
        if min(stale.values()) >= patience:
            break
        model.train()
        t_ep = time.time()
        loss = train_epoch(model, opt, data, rng, dev)
        t_tr = time.time() - t_ep
        v_uai = uai_val_metric(data, uai_scores(model, data, dev, data.uai_val_mask), data.uai_val_mask)
        v_std = std_val_metric(std_scores(model, data, dev, data.std_val_idx)[0])
        for name, v in (("uai", v_uai), ("std", v_std)):
            stale[name] = 0 if v > best[name][0] + MIN_DELTA else stale[name] + 1
            if v > best[name][0]:
                best[name] = [v, _cpu_state(model), ep]
        torch.cuda.empty_cache()
        log.append({"epoch": ep, "loss": loss, "val_uai": v_uai, "val_std": v_std, "sec": round(time.time() - t0, 1)})
        print(json.dumps(log[-1]), flush=True)
        if SMOKE:
            n_b = getattr(model, "steps_per_epoch", None) or sum(1 for _ in model.batches(data, np.random.default_rng(0)))
            print(json.dumps({"smoke_train_sec": round(t_tr, 1), "smoke_batches": SMOKE, "batches_per_epoch": n_b,
                              "val_sec": round(time.time() - t_ep - t_tr, 1),
                              "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}), flush=True)
            return
        torch.save({"model": _cpu_state(model), "opt": opt.state_dict() if opt else model.state_dict_train(),
                    "best": best, "stale": stale, "log": log, "epoch": ep, "rng": rng,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, str(ck) + ".tmp")
        os.replace(str(ck) + ".tmp", ck)
    out = {}
    model.load_state_dict(best["uai"][1])
    for split in (1, 2):
        out[f"uai_split{split}"] = uai_scores(model, data, dev, data.uai["split"] == split)
    if kind == "mbstr":
        np.save(data.dir / "mbstr_item_emb.npy", model.item_embeddings())
    model.load_state_dict(best["std"][1])
    for split in (1, 2):
        idx = np.flatnonzero(data.std["split"] == split)
        out[f"std_split{split}"] = std_scores(model, data, dev, idx)[0]
        out[f"stdfull_rank_split{split}"] = std_full_rank(model, data, dev, idx)
    for k_, v_ in out.items():
        if np.isnan(v_).any():
            raise RuntimeError(f"NaN in {k_}")
    np.savez(data.dir / f"scores_{kind}_s{seed}.npz", **out)
    hp_log = {k: (v if not isinstance(v, tuple) else list(v)) for k, v in hp.items()}
    (data.dir / f"log_{kind}_s{seed}.json").write_text(json.dumps(
        {"kind": kind, "seed": seed, "best_epoch_uai": best["uai"][2], "best_epoch_std": best["std"][2],
         "best_val_uai": best["uai"][0], "best_val_std": best["std"][0], "epochs": log, "hp": hp_log,
         "max_epochs": max_epochs, "patience": patience, "min_delta": MIN_DELTA, "val_cap": VAL_CAP,
         "stopped_early": min(stale.values()) >= patience,
         "params": sum(p.numel() for p in model.parameters()), "runtime_sec": round(time.time() - t0, 1)},
        indent=1, default=str))
    ck.unlink(missing_ok=True)
    print("done", kind, run, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
