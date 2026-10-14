import numpy as np
import pandas as pd
import pytest

from recsys_auto.features import history_pack
from recsys_auto.grammar import ENTITIES, FeaturePack
from recsys_auto.snapshot import snapshot_features


@pytest.mark.parametrize("entity", ENTITIES)
@pytest.mark.parametrize("days", [1, None])
def test_snapshot_queries_match_event_engine(entity, days):
    from recsys_auto.data import GENRES

    events = pd.DataFrame(
        {
            "event_id": range(5),
            "timestamp": [0, 86400, 86400, 172800, 172800],
            "user": [1, 1, 2, 1, 2],
            "item": [1, 2, 2, 3, 1],
            "rating": [5, 2, 4, 0, 0],
            "label": [1, 0, 1, 0, 0],
            "genres": [(0, 1), (1,), (0, 1), (0, 1), (1,)],
            "gender": "M",
            "occupation": "student",
            "age": 20,
            "release_year": 1990,
            **{genre: 0 for genre in GENRES},
        }
    )
    pack = FeaturePack(entity, days)
    expected = history_pack(events, pack).iloc[3:]
    actual = snapshot_features(events.iloc[:3], events.iloc[3:], [pack])[pack.columns]
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
