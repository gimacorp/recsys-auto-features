import pandas as pd
import pytest

from recsys_auto.statistics import macro_cluster_interval


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
