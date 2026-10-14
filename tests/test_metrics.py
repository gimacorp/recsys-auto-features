import numpy as np
import pandas as pd
import pytest

from recsys_auto.metrics import evaluate, paired_bootstrap, ranking_per_user


def test_known_ranking_and_user_filter():
    events = pd.DataFrame(
        {"user": [1, 1, 1, 2, 2], "item": [1, 2, 3, 1, 2], "label": [1, 0, 1, 1, 1]}
    )
    frame = ranking_per_user(events, np.array([0.9, 0.8, 0.7, 0.2, 0.3]))
    assert len(frame) == 1
    assert frame.iloc[0]["ndcg@10"] == pytest.approx((1 + 0.5) / (1 + 1 / np.log2(3)))
    assert frame.iloc[0]["recall@10"] == 1


def test_identical_models_have_zero_paired_interval():
    events = pd.DataFrame({"user": [1, 1, 2, 2], "item": [1, 2, 1, 2], "label": [1, 0, 0, 1]})
    prediction = np.array([0.8, 0.2, 0.4, 0.6])
    result = paired_bootstrap(events, prediction, prediction, samples=50)
    assert result["logloss_ci95"] == [0.0, 0.0]
    assert result["ndcg@10_ci95"] == [0.0, 0.0]


def test_empty_ranking_is_explicit():
    events = pd.DataFrame({"user": [1, 2], "item": [1, 1], "label": [0, 1]})
    result = evaluate(events, np.array([0.2, 0.8]))
    assert result["ranking_users"] == 0
    assert result["ndcg@10"] is None
