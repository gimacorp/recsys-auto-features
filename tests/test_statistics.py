import json

import numpy as np
import pandas as pd
import pytest

from recsys_auto.statistics import macro_cluster_interval, practical_criteria


def test_common_user_draws_preserve_opposite_period_effects():
    first = pd.DataFrame({"sum": [1.0, -1.0], "count": [1, 1]}, index=[1, 2])
    second = pd.DataFrame({"sum": [-1.0, 1.0], "count": [1, 1]}, index=[1, 2])
    result = macro_cluster_interval([first, second], samples=100)
    assert result["ci95"] == [0, 0]
    assert result["delta"] == 0


def test_cluster_bootstrap_estimate_matches_event_weighted_difference():
    grouped = pd.DataFrame({"sum": [0.5, 1.0], "count": [1, 4]}, index=[1, 2])
    result = macro_cluster_interval([grouped], samples=100)
    assert result["delta"] == pytest.approx(0.3)
    assert result["ci95"][0] <= 0.3 <= result["ci95"][1]


def test_preregistered_numpy_results_serialize_as_json_booleans():
    criteria = practical_criteria({"ci95": np.array([0.001, 0.012])}, np.float64(0.5))
    restored = json.loads(json.dumps(criteria, allow_nan=False))
    assert restored["loss_passed"] is False
    assert restored["compactness_passed"] is True
