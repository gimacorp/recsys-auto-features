"""Признаки кандидатов в фиксированный момент с отдельной историей feedback."""

import numpy as np
import pandas as pd

from .features import static_features
from .grammar import ENTITIES, FeaturePack

KEYS = {
    "user": ["user"],
    "item": ["item"],
    "user_item": ["user", "item"],
    "genre": ["genre"],
    "user_genre": ["user", "genre"],
}


def snapshot_features(
    history: pd.DataFrame, queries: pd.DataFrame, requested: list[FeaturePack]
) -> pd.DataFrame:
    if queries.timestamp.nunique() != 1:
        raise ValueError("Все кандидаты должны иметь один timestamp")
    timestamp = int(queries.timestamp.iloc[0])
    history = history[history.timestamp < timestamp]
    frames = [static_features(queries)]
    for entity in ENTITIES:
        entity_packs = [pack for pack in requested if pack.entity == entity]
        if not entity_packs:
            continue
        keys = KEYS[entity]
        past = history.copy()
        current = queries.copy()
        if "genre" in entity:
            past = past.explode("genres").rename(columns={"genres": "genre"})
            current = current.explode("genres").rename(columns={"genres": "genre"})
        for pack in entity_packs:
            window = (
                past if pack.days is None else past[past.timestamp >= timestamp - pack.days * 86400]
            )
            grouped = (
                window.groupby(keys, sort=False)
                .agg(
                    count=("label", "size"),
                    positive=("label", "sum"),
                    rating_sum=("rating", "sum"),
                    latest=("timestamp", "max"),
                )
                .reset_index()
            )
            merged = current[["event_id", *keys]].merge(
                grouped, how="left", on=keys, validate="many_to_one", sort=False
            )
            count = merged["count"].fillna(0)
            values = pd.DataFrame(
                {
                    "event_id": merged.event_id,
                    "count": np.log1p(count),
                    "rate": (merged.positive.fillna(0) + 2.5) / (count + 5),
                    "rating": (merged.rating_sum.fillna(0) + 15) / (count + 5),
                    "gap": ((timestamp - merged.latest) / 86400).fillna(3650),
                }
            )
            if "genre" in entity:
                aggregated = values.groupby("event_id", sort=True).agg(
                    log_count_mean=("count", "mean"),
                    log_count_max=("count", "max"),
                    positive_rate_mean=("rate", "mean"),
                    positive_rate_min=("rate", "min"),
                    rating_mean=("rating", "mean"),
                    gap_days_min=("gap", "min"),
                )
                result = aggregated.reindex(queries.event_id).to_numpy(np.float32)
            else:
                result = values[["count", "rate", "rating", "gap"]].to_numpy(np.float32)
            frames.append(pd.DataFrame(result, columns=pack.columns, index=queries.index))
    return pd.concat(frames, axis=1)
