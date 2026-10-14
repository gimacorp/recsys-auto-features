import numpy as np
import pytest

from recsys_auto.catalogue import retrieval_metrics


def test_catalogue_recall_uses_full_positive_set():
    ndcg, recall = retrieval_metrics(np.arange(20), np.arange(20, dtype=float), {19, 0})
    assert recall == 0.5
    assert ndcg == pytest.approx(1 / (1 + 1 / np.log2(3)))


def test_catalogue_ties_are_deterministic_by_item_id():
    ndcg, recall = retrieval_metrics(np.array([3, 1, 2]), np.array([0.5, 0.5, 0.5]), {1}, k=1)
    assert ndcg == 1
    assert recall == 1
