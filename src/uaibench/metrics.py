"""Query-level metrics for UAI ranking.

All ranking metrics are tie-aware: items with equal scores share the average
discount of the positions they jointly occupy, which equals the expected metric
under a uniformly random tie order (McSherry and Najork, 2008).
"""

from __future__ import annotations

import math

import numpy as np

CONVERTED, RETURNED, SILENT = 2, 1, 0


def _discounts(n: int, k: int) -> np.ndarray:
    pos = np.arange(n)
    return np.where(pos < k, 1.0 / np.log2(pos + 2.0), 0.0)


def tie_aware_dcg(gains: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    g = gains[order]
    disc = _discounts(len(s), k)
    # boundaries of equal-score groups in sorted order
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    ends = np.r_[starts[1:], len(s)]
    csum = np.r_[0.0, np.cumsum(disc)]
    group_disc = (csum[ends] - csum[starts]) / (ends - starts)
    group_gain = np.add.reduceat(g, starts)
    return float(np.dot(group_gain, group_disc))


def cc_ndcg(gains: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """Chance-corrected NDCG: 0 for a random order, 1 for the ideal order."""
    n = len(gains)
    disc = _discounts(n, k)
    ideal = float(np.dot(np.sort(gains)[::-1], disc))
    expected = float(gains.mean() * disc.sum())
    if ideal - expected <= 1e-12:
        return math.nan
    return (tie_aware_dcg(gains, scores, k) - expected) / (ideal - expected)


def ndcg(gains: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    disc = _discounts(len(gains), k)
    ideal = float(np.dot(np.sort(gains)[::-1], disc))
    if ideal <= 0:
        return math.nan
    return tie_aware_dcg(gains, scores, k) / ideal


def random_ndcg(gains: np.ndarray, k: int = 10) -> float:
    disc = _discounts(len(gains), k)
    ideal = float(np.dot(np.sort(gains)[::-1], disc))
    if ideal <= 0:
        return math.nan
    return float(gains.mean() * disc.sum()) / ideal


def pair_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def query_metrics(labels: np.ndarray, scores: np.ndarray, g: float | None = None, k: int = 10) -> dict:
    """labels use CONVERTED/RETURNED/SILENT codes.

    Conversion task: Converted vs the rest. Re-engagement task: Converted or
    Returned vs Silent. Conditional diagnostic: Converted vs Returned. The
    graded variant (gain g for Returned) is kept only as a sensitivity check:
    per-query normalization rescales any g > 0 to full weight on queries
    without a conversion.
    """
    conv = (labels == CONVERTED).astype(float)
    eng = (labels != SILENT).astype(float)
    c, r, s = (scores[labels == x] for x in (CONVERTED, RETURNED, SILENT))
    out = {
        "cc_ndcg_conv": cc_ndcg(conv, scores, k),
        "ndcg_conv": ndcg(conv, scores, k) if 0 < conv.sum() < len(conv) else math.nan,
        "cc_ndcg_engage": cc_ndcg(eng, scores, k),
        "auc_target": pair_auc(c, np.r_[r, s]),
        "auc_return": pair_auc(np.r_[c, r], s),
        "auc_conditional": pair_auc(c, r),
    }
    if g is not None:
        graded = np.where(labels == CONVERTED, 1.0, np.where(labels == RETURNED, g, 0.0))
        out["cc_ndcg_graded"] = cc_ndcg(graded, scores, k)
    return out
