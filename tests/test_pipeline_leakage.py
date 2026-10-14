import numpy as np
import pandas as pd

from recsys_auto.data import GENRES
from recsys_auto.features import STATIC_COLUMNS, build_features
from recsys_auto.grammar import packs
from recsys_auto.model import fit_validation
from recsys_auto.search import search
from recsys_auto.split import TemporalFold


def test_test_label_mutation_preserves_model_and_feature_selection():
    rng = np.random.default_rng(19)
    count = 64
    ratings = rng.integers(1, 6, count)
    events = pd.DataFrame(
        {
            "event_id": np.arange(count),
            "timestamp": np.arange(count) // 2 * 86400,
            "user": np.arange(count) % 4,
            "item": np.arange(count) % 11,
            "rating": ratings,
            "label": (ratings >= 4).astype(int),
            "genres": [(1, 8)] * count,
            "age": 30,
            "gender": "M",
            "occupation": "student",
            "release_year": 1990,
            **{genre: int(genre in ["Action", "Drama"]) for genre in GENRES},
        }
    )
    fold = TemporalFold(32, 48, 64)
    config = {
        "max_iterations": 12,
        "early_stopping_rounds": 3,
        "thread_count": 2,
        "search_rounds": 1,
        "proposals_per_round": 2,
        "min_improvement": 0.0001,
    }
    params = {"depth": 3, "learning_rate": 0.1, "l2_leaf_reg": 3}
    results = []
    predictions = []
    for poison in [False, True]:
        current = events.copy()
        if poison:
            current.loc[fold.test, "rating"] = 6 - current.loc[fold.test, "rating"]
            current["label"] = (current.rating >= 4).astype(int)
        matrix, _ = build_features(current, [7, None, 3, 10])
        columns = STATIC_COLUMNS + [column for pack in packs([7, None]) for column in pack.columns]
        fitted = fit_validation(matrix, current.label, fold, columns, params, config, 17)
        predictions.append(fitted.prediction)
        results.append(
            search(
                matrix, current.label, fold, fitted, packs([3, 10]), params, config, 17, "residual"
            )
        )
    np.testing.assert_array_equal(predictions[0], predictions[1])
    assert results[0].selected_packs == results[1].selected_packs
    assert results[0].fitted.logloss == results[1].fitted.logloss
