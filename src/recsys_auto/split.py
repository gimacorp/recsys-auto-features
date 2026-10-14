"""Временные границы с атомарными группами одинакового timestamp."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    train_end: int
    validation_end: int
    test_end: int

    @property
    def train(self) -> slice:
        return slice(0, self.train_end)

    @property
    def validation(self) -> slice:
        return slice(self.train_end, self.validation_end)

    @property
    def test(self) -> slice:
        return slice(self.validation_end, self.test_end)

    @property
    def development(self) -> slice:
        return slice(0, self.validation_end)

    def describe(self, events: pd.DataFrame) -> dict:
        result = {}
        for name in ["train", "validation", "test"]:
            block = events.iloc[getattr(self, name)]
            result[name] = {
                "rows": len(block), "users": int(block.user.nunique()),
                "positive_fraction": float(block.label.mean()),
                "start_utc": str(pd.to_datetime(block.timestamp.min(), unit="s", utc=True)),
                "end_utc": str(pd.to_datetime(block.timestamp.max(), unit="s", utc=True)),
            }
        return result


def temporal_fold(events: pd.DataFrame, fractions: list[float]) -> TemporalFold:
    timestamps = events.timestamp.to_numpy()
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("События должны быть упорядочены по времени")
    if len(fractions) != 3 or not (0 < fractions[0] < fractions[1] < fractions[2] <= 1):
        raise ValueError("Ожидаются три строго возрастающие доли в (0, 1]")
    boundaries = []
    for fraction in fractions:
        if fraction == 1:
            boundaries.append(len(events))
        else:
            index = int(len(events) * fraction)
            boundaries.append(int(np.searchsorted(timestamps, timestamps[index], side="left")))
    fold = TemporalFold(*boundaries)
    if not (0 < fold.train_end < fold.validation_end < fold.test_end):
        raise ValueError("Разбиение создало пустой период")
    return fold
