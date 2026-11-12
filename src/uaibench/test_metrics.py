import itertools
import math

import numpy as np

from .metrics import cc_ndcg, ndcg, pair_auc, tie_aware_dcg


def test_tie_aware_equals_permutation_average():
    gains = np.array([1.0, 0.3, 0.0, 0.0, 0.3])
    scores = np.array([2.0, 2.0, 1.0, 2.0, 0.0])
    tied = [0, 1, 3]
    vals = []
    for perm in itertools.permutations(tied):
        s = scores.copy()
        for rank, j in enumerate(perm):
            s[j] = 2.0 + (3 - rank) * 1e-6
        vals.append(tie_aware_dcg(gains, s, 3))
    assert math.isclose(tie_aware_dcg(gains, scores, 3), np.mean(vals))


def test_cc_ndcg_bounds():
    gains = np.array([1.0, 0.5, 0.0, 0.0])
    assert math.isclose(cc_ndcg(gains, np.array([4.0, 3, 2, 1])), 1.0)
    assert math.isclose(cc_ndcg(gains, np.zeros(4)), 0.0, abs_tol=1e-12)  # all tied == random
    assert math.isnan(cc_ndcg(np.zeros(3), np.arange(3.0)))
    vals = [cc_ndcg(gains, np.array(p, float)) for p in itertools.permutations(range(4))]
    assert math.isclose(np.mean(vals), 0.0, abs_tol=1e-12)


def test_auc_and_ndcg():
    assert pair_auc(np.array([1.0]), np.array([0.0, 1.0])) == 0.75
    assert math.isclose(ndcg(np.array([0.0, 1.0]), np.array([0.0, 1.0])), 1.0)


if __name__ == "__main__":
    test_tie_aware_equals_permutation_average()
    test_cc_ndcg_bounds()
    test_auc_and_ndcg()
    print("metric tests passed")
