"""MBGen (Liu, Hou, and McAuley, CIKM 2024) with the official PBA transformer (third_party/mbgen, Apache-2.0).

Kept from the release (config/CID/{ijcai,retail}/main.yaml, run.py, trainer/Trainer.py, tokenizer/utils.py,
trainer/evaluation.py at commit 268a5d0):
  items      balanced chunked IDs (CID): a seeded (2024) random permutation of the item ids written as 3 base-K
             digits behind the behavior token; K = 96 for the IJCAI configuration and 64 for Retail. The Taobao
             and Tmall catalogs exceed 96^3 items, so there K is the smallest value with K^3 > n_items.
  sequences  user token (id mod 2000) + the last 50 events x 4 tokens + EOS = 202 tokens; target
             [behavior, 3 digits, EOS]; every prefix of each user's last 51 training events is a training sample
             (SequenceGenerator.generate_training_sequence). The release drops items with fewer than 5
             occurrences; the benchmark keeps every item so that all candidates can be scored.
  model      PBA transformer, d_model 256, d_ff 512, 6 heads of 64, 4 encoder and 4 sparse decoder layers,
             behavior injection in layers 0-1, dropout 0.
  training   AdamW (lr 1.2e-3 IJCAI configuration / 1e-3 Retail, weight decay 0.005), cosine schedule with
             15k warm-up steps over 350k steps, batch 512, fp16 autocast with GradScaler(2^14), gradient-norm
             clipping at 1.0 (applied, as in the release, before unscaling), evaluation every epoch, patience 10
             (stops once 11 epochs fail to improve), at most ceil(350k / steps per epoch) epochs.
  ranking    full-catalog ranking by beam search with 50 beams after the decoder prefix [BOS, buy] (the
             release's test setting).
Candidate scoring for VNB-Bench and the sampled protocol: teacher-forced log-likelihood of a candidate's three
digits after [BOS, buy], the quantity that beam search maximizes.
"""

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import hf_path

ROOT = Path(__file__).resolve().parents[2]
hf_path.add()


def _load(name):
    spec = importlib.util.spec_from_file_location(f"mbgen_{name}", ROOT / "third_party" / "mbgen" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PBA = _load("PBA_transformer")
EVAL = _load("evaluation")
from transformers.optimization import get_scheduler  # noqa: E402

N_EV, N_TOK = 50, 4          # events in the input, tokens per event
SEQ_LEN = 2 + N_EV * N_TOK   # max_seq_len 202
N_USER_TOK = 2000
BEH = np.array([0, 2, 3, 4, 1])  # benchmark ids (1 view, 2 fav, 3 cart, 4 buy) -> MBGen (buy 1, pv 2, fav 3, cart 4)
BUY_TOK = 1


class MBGen(torch.nn.Module):
    infer_bs = 256
    micro = 256  # memory only: gradients of a 512 batch accumulated over two halves

    def __init__(self, kind, n_items, n_users, ds):
        super().__init__()
        self.n_items = n_items
        retail = ds == "retailrocket"
        K = 64 if retail else 96
        while K ** 3 <= n_items:
            K += 1
        self.K = K
        self.offsets = [0, 5, 5 + K, 5 + 2 * K, 5 + 3 * K]
        self.eos = self.offsets[-1] + N_USER_TOK
        self.L = N_EV
        self.hp = {"config": "retail" if retail else "ijcai", "K": K, "id_offsets": self.offsets,
                   "num_user_tokens": N_USER_TOK, "batch": 512, "lr": 1e-3 if retail else 1.2e-3,
                   "wd": 0.005, "warmup_steps": 15000, "steps": 350000, "patience": 11, "max_epochs": None,
                   "beams_full": 50, "d_model": 256, "d_ff": 512, "heads": 6, "d_kv": 64, "layers": "4+4 sparse",
                   "loss": "token cross-entropy (seq2seq)"}
        cfg = PBA.PBATransformerConfig(
            behavior_injection=True, behavior_embedding_dim=64, num_positions=4, num_behavior=4,
            num_layers=4, num_decoder_layers=4, num_sparse_encoder_layers=4, num_sparse_decoder_layers=4,
            sparse_layers_encoder=[], sparse_layers_decoder=[0, 1, 2, 3], behavior_injection_encoder=[0, 1],
            behavior_injection_decoder=[0, 1], Moe_behavior_only=False, shared_expert=False, d_model=256, d_ff=512,
            num_heads=6, d_kv=64, dropout_rate=0.0, activation_function="relu", vocab_size=self.eos + 1,
            pad_token_id=0, eos_token_id=self.eos, decoder_start_token_id=0, feed_forward_proj="relu", n_positions=50)
        self.m = PBA.PBATransformersForConditionalGeneration(cfg)
        # CID: random permutation of the item ids (generate_random_map, seed 2024), then base-K digits
        ids = np.arange(1, n_items + 1)
        perm = ids.copy()
        np.random.seed(2024)
        np.random.shuffle(perm)
        value = np.zeros(n_items + 1, np.int64)
        value[ids] = perm
        digits = np.stack([value // K ** 2, (value // K) % K, value % K], 1)
        self.codes = digits + np.array(self.offsets[1:4])  # (n_items + 1) x 3 token ids
        self.codes[0] = 0
        self.value_to_item = np.zeros(K ** 3, np.int64)
        self.value_to_item[value[ids]] = ids

    # ---------------------------------------------------------------- tokenization
    def _encoder_inputs(self, items, behs, uids):
        """items/behs: B x 50 left-padded events -> input_ids, attention_mask (B x 202), as pad_sequence."""
        B = len(items)
        n = (items > 0).sum(1)
        tok = np.concatenate([BEH[behs][:, :, None], self.codes[items]], 2).reshape(B, N_EV * N_TOK)
        j = np.arange(N_EV * N_TOK)[None, :]
        src = np.clip((N_EV - n)[:, None] * N_TOK + j, 0, N_EV * N_TOK - 1)
        body = np.where(j < N_TOK * n[:, None], np.take_along_axis(tok, src, 1), 0)
        ids = np.zeros((B, SEQ_LEN), np.int64)
        ids[:, 0] = np.asarray(uids) % N_USER_TOK + self.offsets[-1]
        ids[:, 1:1 + N_EV * N_TOK] = body
        ids[np.arange(B), 1 + N_TOK * n] = self.eos
        mask = (np.arange(SEQ_LEN)[None, :] <= (1 + N_TOK * n)[:, None]).astype(np.int64)
        return ids, mask

    # ---------------------------------------------------------------- training
    def setup_training(self, data, dev):
        users, s, e = data._train_users()
        ws = np.maximum(s, e - (N_EV + 1))  # the last 51 training events of each user
        cnt = e - ws - 1
        self.samp_u = np.repeat(users, cnt)
        self.samp_ws = np.repeat(ws, cnt)
        self.samp_t = np.repeat(ws + 1, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        self.steps_per_epoch = math.ceil(len(self.samp_t) / self.hp["batch"])
        self.hp["max_epochs"] = math.ceil(self.hp["steps"] / self.steps_per_epoch)
        self.hp["train_samples"] = int(len(self.samp_t))
        self.opt = torch.optim.AdamW(self.parameters(), lr=self.hp["lr"], weight_decay=self.hp["wd"])
        self.sched = get_scheduler(name="cosine", optimizer=self.opt, num_warmup_steps=self.hp["warmup_steps"],
                                   num_training_steps=self.hp["steps"])
        self.scaler = torch.cuda.amp.GradScaler(init_scale=2 ** 14)

    def state_dict_train(self):
        return {"opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "scaler": self.scaler.state_dict()}

    def load_state_dict_train(self, st):
        self.opt.load_state_dict(st["opt"])
        self.sched.load_state_dict(st["sched"])
        self.scaler.load_state_dict(st["scaler"])

    def _train_batch(self, data, sel):
        u, t, ws = self.samp_u[sel], self.samp_t[sel], self.samp_ws[sel]
        it, bh = data.gather(u, t, N_EV)
        keep = np.arange(N_EV)[None, :] >= N_EV - (t - ws)[:, None]  # history starts at the window start
        it, bh = np.where(keep, it, 0), np.where(keep, bh, 0)
        ids, mask = self._encoder_inputs(it, bh, u)
        lab = np.concatenate([BEH[data.b[t]][:, None], self.codes[data.i[t]], np.full((len(t), 1), self.eos)], 1)
        return ids, mask, lab

    def fit_epoch(self, data, rng, dev):
        self.train()
        perm = rng.permutation(len(self.samp_t))
        total, nb = 0.0, 0
        bs = self.hp["batch"]
        smoke = int(os.environ.get("UAI_SMOKE_STEPS", 0))
        for k in range(0, len(perm), bs):
            if smoke and nb == smoke:
                break
            ids, mask, lab = (torch.as_tensor(x, device=dev) for x in self._train_batch(data, perm[k:k + bs]))
            self.opt.zero_grad()
            loss = 0.0
            # the 512-sample batch is processed in two halves whose size-weighted mean losses add up to the
            # batch's mean token loss (every target has 5 tokens), so the gradient equals the full batch's
            for h in range(0, len(ids), self.micro):
                with torch.cuda.amp.autocast():
                    part = self.m(input_ids=ids[h:h + self.micro], attention_mask=mask[h:h + self.micro],
                                  labels=lab[h:h + self.micro]).loss * (len(ids[h:h + self.micro]) / len(ids))
                self.scaler.scale(part).backward()
                loss = loss + part.detach()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            total += loss.item()
            nb += 1
        return total / max(nb, 1)

    # ---------------------------------------------------------------- scoring
    @torch.no_grad()
    def _score(self, data, dev, uids, ends, qidx, cands, row_bs=2048):
        """Log-likelihood of each candidate's digits for the query (uids[q], prefix ending before ends[q])."""
        self.eval()
        qidx = np.asarray(qidx, np.int64)
        cands = np.asarray(cands, np.int64)
        flat = cands.reshape(len(qidx), -1)
        C = flat.shape[1]
        out = np.zeros(flat.shape, np.float32)
        order = np.argsort(qidx, kind="stable")
        qs = np.asarray(qidx)[order]
        bounds = np.searchsorted(qs, np.arange(len(uids) + 1))
        for a in range(0, len(uids), self.infer_bs):
            b = min(a + self.infer_bs, len(uids))
            it, bh = data.gather(uids[a:b], ends[a:b], N_EV)
            empty = (it > 0).sum(1) == 0
            ids, mask = (torch.as_tensor(x, device=dev) for x in self._encoder_inputs(it, bh, uids[a:b]))
            with torch.cuda.amp.autocast():
                enc = self.m.get_encoder()(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
            rows = order[bounds[a]:bounds[b]]
            for r0 in range(0, len(rows), max(1, row_bs // C)):
                rr = rows[r0:r0 + max(1, row_bs // C)]
                ql = torch.as_tensor(np.asarray(qidx)[rr] - a, device=dev).repeat_interleave(C)
                code = torch.as_tensor(self.codes[flat[rr].reshape(-1)], device=dev)
                dec = torch.cat([torch.zeros_like(code[:, :1]), torch.full_like(code[:, :1], BUY_TOK), code[:, :2]], 1)
                with torch.cuda.amp.autocast():
                    logits = self.m(encoder_outputs=(enc[ql],), attention_mask=mask[ql], decoder_input_ids=dec).logits
                lp = F.log_softmax(logits[:, 1:4].float(), -1).gather(2, code.unsqueeze(-1)).sum((1, 2))
                out[rr] = lp.view(len(rr), C).cpu().numpy()
            e_rows = rows[empty[np.asarray(qidx)[rows] - a]]
            out[e_rows] = 0.0  # no prior event: constant (tied) scores, as for the other models
        return out.reshape(cands.shape)

    def uai_scores(self, data, dev, mask):
        U = data.uai
        uid, t, items = U["uid"][mask].astype(np.int64), U["t"][mask], U["item"][mask]
        q = uid * 100000 + t
        uq, qidx = np.unique(q, return_inverse=True)
        ends = np.searchsorted(data.key_day, uq, side="right")
        return self._score(data, dev, uq // 100000, ends, qidx, items[:, None]).reshape(-1)

    def _std_ends(self, data, idx):
        S = data.std
        uid, ts = S["uid"][idx].astype(np.int64), S["ts"][idx].astype(np.int64)
        return uid, np.searchsorted(data.key_ts, uid * (1 << 31) + (ts - data.ts0), side="left")

    def std_scores(self, data, dev, idx):
        S = data.std
        uid, ends = self._std_ends(data, idx)
        cands = np.concatenate([S["pos"][idx][:, None], S["negs"][idx]], 1)
        return self._score(data, dev, uid, ends, np.arange(len(idx)), cands), None, None

    @torch.no_grad()
    def std_full_rank(self, data, dev, idx, bs=16):
        """Rank of the purchased item among the 50 beams, earlier purchases removed; 1e9 if it is not generated."""
        self.eval()
        S = data.std
        uid, ends = self._std_ends(data, idx)
        pos = S["pos"][idx].astype(np.int64)
        nb = self.hp["beams_full"]
        ranks = np.full(len(idx), 1e9)
        for a in range(0, len(idx), bs):
            b = min(a + bs, len(idx))
            it, bh = data.gather(uid[a:b], ends[a:b], N_EV)
            ids, mask = (torch.as_tensor(x, device=dev) for x in self._encoder_inputs(it, bh, uid[a:b]))
            dec = torch.tensor([[0, BUY_TOK]], device=dev).repeat(b - a, 1)
            with torch.cuda.amp.autocast():
                seqs = EVAL.beam_search(self.m, ids, mask, decoder_input_ids=dec, max_length=6, num_beams=nb,
                                        num_return_sequences=nb)
            dig = seqs[:, 2:5].cpu().numpy().reshape(b - a, nb, 3) - np.array(self.offsets[1:4])
            ok = ((dig >= 0) & (dig < self.K)).all(2)
            val = np.where(ok, dig[..., 0] * self.K ** 2 + dig[..., 1] * self.K + dig[..., 2], 0)
            gen = np.where(ok, self.value_to_item[val], 0)
            for r in range(b - a):
                if (it[r] > 0).sum() == 0:
                    ranks[a + r] = (self.n_items - 1) / 2  # empty history: expected rank of a random order
                    continue
                s0, e0 = data.ustart[uid[a + r]], ends[a + r]
                bought = set(data.i[s0:e0][data.b[s0:e0] == 4].tolist()) - {pos[a + r]}
                seen, lst = set(), []
                for g in gen[r]:
                    if g > 0 and g not in bought and g not in seen:
                        seen.add(g)
                        lst.append(g)
                if pos[a + r] in seen:
                    ranks[a + r] = lst.index(pos[a + r])
        return ranks
