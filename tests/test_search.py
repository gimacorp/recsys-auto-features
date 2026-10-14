import numpy as np

from recsys_auto.search import residual_scores


def test_proxy_prefers_residual_aligned_feature_and_ignores_constants():
    residual = np.array([-1.0, 1.0, -1.0, 1.0])
    matrix = np.column_stack([residual, [1, 1, 1, 1], [-1, -1, 1, 1]])
    scores = residual_scores(matrix, residual, {"aligned": [0], "constant": [1], "other": [2]})
    assert scores["aligned"] == 1
    assert scores["constant"] == 0
    assert scores["other"] == 0
