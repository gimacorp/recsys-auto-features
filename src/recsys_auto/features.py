"""Векторизованные исторические агрегаты с условием history.timestamp < query.timestamp."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data import GENRES
from .grammar import ENTITIES, FeaturePack, packs

CAT_COLUMNS = ["user", "item", "gender", "occupation"]
STATIC_COLUMNS = [*CAT_COLUMNS, "age", "release_year", *GENRES, "hour", "weekday", "day"]


def static_features(events: pd.DataFrame) -> pd.DataFrame:
    result = events[[*CAT_COLUMNS, "age", "release_year", *GENRES]].copy()
    for column in CAT_COLUMNS:
        result[column] = result[column].astype(str)
    result["hour"] = (events.timestamp // 3600 % 24).astype(np.int8)
    result["weekday"] = ((events.timestamp // 86400 + 3) % 7).astype(np.int8)
    result["day"] = (events.timestamp // 86400).astype(np.int32)
    return result


def grouped_history(events: pd.DataFrame, entity: str) -> pd.DataFrame:
    result = events[["event_id", "timestamp", "rating", "label", "user", "item"]].copy()
    if "genre" in entity:
        result["genre"] = events.genres
        result = result.explode("genre", ignore_index=True)
    keys = {
        "user": ["user"],
        "item": ["item"],
        "user_item": ["user", "item"],
        "genre": ["genre"],
        "user_genre": ["user", "genre"],
    }[entity]
    result["group"] = result.groupby(keys, sort=False).ngroup()
    return result.sort_values(["group", "timestamp", "event_id"], kind="stable").reset_index(
        drop=True
    )


def history_pack(
    events: pd.DataFrame, pack: FeaturePack, expanded: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Окно [t-window, t). Сглаживание использует фиксированные априорные значения."""
    expanded = grouped_history(events, pack.entity) if expanded is None else expanded
    length = len(expanded)
    values = np.zeros((length, 4), dtype=np.float64)
    values[:, 1:] = [0.5, 3.0, 3650.0]
    timestamps = expanded.timestamp.to_numpy(np.int64)
    rating = expanded.rating.to_numpy(np.float64)
    label = expanded.label.to_numpy(np.float64)
    groups = expanded.group.to_numpy()
    boundaries = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, length]
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end - start == 1:
            continue
        t = timestamps[start:end]
        right = np.searchsorted(t, t, side="left")
        left = (
            np.zeros(len(t), dtype=int)
            if pack.days is None
            else np.searchsorted(t, t - pack.days * 86400, side="left")
        )
        count = right - left
        positives = np.r_[0.0, np.cumsum(label[start:end])]
        ratings = np.r_[0.0, np.cumsum(rating[start:end])]
        values[start:end, 0] = np.log1p(count)
        values[start:end, 1] = (positives[right] - positives[left] + 2.5) / (count + 5)
        values[start:end, 2] = (ratings[right] - ratings[left] + 15) / (count + 5)
        gap = (t - t[np.maximum(right - 1, 0)]) / 86400
        values[start:end, 3] = np.where(count > 0, gap, 3650.0)
    if "genre" in pack.entity:
        frame = pd.DataFrame(values, columns=["count", "rate", "rating", "gap"])
        frame["event_id"] = expanded.event_id.to_numpy()
        aggregated = frame.groupby("event_id", sort=True).agg(
            log_count_mean=("count", "mean"),
            log_count_max=("count", "max"),
            positive_rate_mean=("rate", "mean"),
            positive_rate_min=("rate", "min"),
            rating_mean=("rating", "mean"),
            gap_days_min=("gap", "min"),
        )
        result = aggregated.reindex(events.event_id).to_numpy(dtype=np.float32)
    else:
        order = np.argsort(expanded.event_id.to_numpy())
        result = values[order].astype(np.float32)
    return pd.DataFrame(result, index=events.index, columns=pack.columns)


def build_features(events: pd.DataFrame, days: list[int | None]) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    frames = [static_features(events)]
    timings = {}
    for entity in ENTITIES:
        expanded = grouped_history(events, entity)
        for pack in packs(days):
            if pack.entity != entity:
                continue
            tick = time.perf_counter()
            frames.append(history_pack(events, pack, expanded))
            timings[pack.name] = time.perf_counter() - tick
        print(f"features: {entity}", flush=True)
    frame = pd.concat(frames, axis=1)
    return frame, {
        "seconds": time.perf_counter() - started,
        "packs_seconds": timings,
        "rows": len(frame),
        "columns": len(frame.columns),
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
    }
