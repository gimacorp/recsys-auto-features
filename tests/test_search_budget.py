from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from recsys_auto.grammar import FeaturePack
from recsys_auto.search import search
from recsys_auto.split import TemporalFold


@pytest.mark.parametrize("strategy", ["random", "residual"])
def test_all_rejected_proposals_still_use_identical_budget(monkeypatch, strategy):
    candidates = [FeaturePack("user", days) for days in [1, 3, 10, 14]]
    frame = pd.DataFrame(
        {column: np.arange(10, dtype=float) for pack in candidates for column in pack.columns}
    )
    frame["base"] = 0
    labels = pd.Series([0, 1] * 5)
    fold = TemporalFold(4, 8, 10)
    calls = []

    def fake_fit(features, y, selected_fold, columns, params, config, seed):
        calls.append(columns)
        return SimpleNamespace(
            logloss=0.6,
            auc=0.5,
            seconds=0.1,
            prediction=np.full(4, 0.5),
            columns=columns,
            model=SimpleNamespace(tree_count_=1),
        )

    monkeypatch.setattr("recsys_auto.search.fit_validation", fake_fit)
    baseline = SimpleNamespace(logloss=0.5, prediction=np.full(4, 0.5), columns=["base"])
    config = {"search_rounds": 2, "proposals_per_round": 2, "min_improvement": 0.0001}
    result = search(frame, labels, fold, baseline, candidates, {}, config, 17, strategy)
    assert len(calls) == 4
    assert result.selected_packs == []
    assert result.fitted is baseline
    assert len({trial["pack"] for trial in result.trials}) == 4
