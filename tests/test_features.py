import numpy as np
import pandas as pd
import pytest

from recsys_auto.features import history_pack
from recsys_auto.grammar import ENTITIES, FeaturePack


def example():
    return pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3, 4],
            "timestamp": [0, 86400, 86400, 172800, 259200],
            "user": [1, 1, 1, 2, 1],
            "item": [1, 2, 3, 1, 1],
            "rating": [5, 1, 4, 2, 3],
            "label": [1, 0, 1, 0, 0],
            "genres": [(0, 1), (0,), (1,), (0, 1), (0, 1)],
        }
    )


def oracle(events, pack):
    rows = []
    for query in events.itertuples():
        stats = []
        genres = query.genres if "genre" in pack.entity else [None]
        for genre in genres:
            past = events[events.timestamp < query.timestamp]
            if pack.days is not None:
                past = past[past.timestamp >= query.timestamp - pack.days * 86400]
            if pack.entity in ["user", "user_item", "user_genre"]:
                past = past[past.user == query.user]
            if pack.entity in ["item", "user_item"]:
                past = past[past.item == query.item]
            if genre is not None:
                past = past.loc[past.genres.map(lambda gs: genre in gs).astype(bool)]
            count = len(past)
            stats.append(
                [
                    np.log1p(count),
                    (past.label.sum() + 2.5) / (count + 5),
                    (past.rating.sum() + 15) / (count + 5),
                    (query.timestamp - past.timestamp.max()) / 86400 if count else 3650,
                ]
            )
        values = np.array(stats)
        if "genre" in pack.entity:
            rows.append(
                [
                    values[:, 0].mean(),
                    values[:, 0].max(),
                    values[:, 1].mean(),
                    values[:, 1].min(),
                    values[:, 2].mean(),
                    values[:, 3].min(),
                ]
            )
        else:
            rows.append(values[0])
    return np.array(rows)


@pytest.mark.parametrize("entity", ENTITIES)
@pytest.mark.parametrize("days", [1, 2, None])
def test_vectorized_engine_matches_independent_oracle(entity, days):
    events = example()
    pack = FeaturePack(entity, days)
    np.testing.assert_allclose(history_pack(events, pack), oracle(events, pack), rtol=1e-6)


@pytest.mark.parametrize("entity", ENTITIES)
def test_future_and_simultaneous_labels_do_not_change_features(entity):
    events = example()
    original = history_pack(events, FeaturePack(entity, None))
    events.loc[events.timestamp >= 86400, ["rating", "label"]] = [5, 1]
    changed = history_pack(events, FeaturePack(entity, None))
    np.testing.assert_array_equal(original.iloc[:3], changed.iloc[:3])


def test_window_includes_left_boundary_and_excludes_query():
    frame = history_pack(example(), FeaturePack("user", 1))
    assert frame.iloc[1, 0] == pytest.approx(np.log(2))
    assert frame.iloc[2, 0] == pytest.approx(np.log(2))
    assert frame.iloc[4, 0] == 0


def test_all_genres_contribute_to_user_affinity():
    frame = history_pack(example(), FeaturePack("user_genre", None))
    assert frame.iloc[4]["user_genre__all__log_count_mean"] == pytest.approx(np.log(3))


@pytest.mark.parametrize("entity", ENTITIES)
def test_randomized_repeated_interactions_match_oracle(entity):
    rng = np.random.default_rng(7)
    ratings = rng.integers(1, 6, 35)
    events = pd.DataFrame(
        {
            "event_id": np.arange(35),
            "timestamp": np.sort(rng.integers(0, 8, 35)) * 86400,
            "user": rng.integers(1, 4, 35),
            "item": rng.integers(1, 5, 35),
            "rating": ratings,
            "label": (ratings >= 4).astype(int),
            "genres": [(0, 1) if index % 2 else (1, 2) for index in range(35)],
        }
    )
    pack = FeaturePack(entity, 2)
    np.testing.assert_allclose(history_pack(events, pack), oracle(events, pack), rtol=1e-6)


@pytest.mark.parametrize("entity", ENTITIES)
def test_feature_rows_follow_input_order(entity):
    events = example().iloc[[4, 1, 3, 0, 2]]
    pack = FeaturePack(entity, None)
    np.testing.assert_allclose(history_pack(events, pack), oracle(events, pack), rtol=1e-6)


def test_duplicate_event_ids_rejected():
    events = example()
    events.loc[1, "event_id"] = events.loc[0, "event_id"]
    with pytest.raises(ValueError, match="event_id"):
        history_pack(events, FeaturePack("user", None))
