"""Загрузка официального архива и нормализация схемы MovieLens."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror", "Musical", "Mystery",
    "Romance", "SciFi", "Thriller", "War", "Western",
]


def download(root: Path) -> dict:
    """Сохраняет архив и только необходимые файлы; исходные данные исключены из Git."""
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "ml-100k.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as source:
        for name in ["u.data", "u.item", "u.user", "README"]:
            target = root / "ml-100k" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(f"ml-100k/{name}"))
    manifest = {"url": URL, "sha256": digest, "bytes": archive.stat().st_size}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load(root: Path) -> pd.DataFrame:
    directory = root / "ml-100k"
    events = pd.read_csv(directory / "u.data", sep="\t", header=None,
                         names=["user", "item", "rating", "timestamp"])
    users = pd.read_csv(directory / "u.user", sep="|", header=None,
                        names=["user", "age", "gender", "occupation", "zip"])
    items = pd.read_csv(directory / "u.item", sep="|", header=None, encoding="latin-1",
                        names=["item", "title", "release", "video_release", "url", *GENRES])
    items["release_year"] = pd.to_datetime(items["release"], format="%d-%b-%Y",
                                            errors="coerce").dt.year.fillna(0).astype(int)
    events = events.merge(users.drop(columns="zip"), on="user", validate="many_to_one")
    events = events.merge(items[["item", "release_year", *GENRES]], on="item",
                          validate="many_to_one")
    events = events.sort_values(["timestamp", "user", "item"], kind="stable").reset_index(drop=True)
    events["event_id"] = np.arange(len(events))
    events["label"] = (events["rating"] >= 4).astype(np.int8)
    events["genres"] = [tuple(np.flatnonzero(row)) or (0,)
                       for row in events[GENRES].to_numpy()]
    if len(events) != 100_000 or events[["user", "item"]].duplicated().any():
        raise ValueError("Нарушена ожидаемая схема MovieLens 100K")
    return events


def describe(events: pd.DataFrame) -> dict:
    return {
        "events": len(events), "users": int(events.user.nunique()),
        "items": int(events.item.nunique()), "positive_fraction": float(events.label.mean()),
        "start_utc": str(pd.to_datetime(events.timestamp.min(), unit="s", utc=True)),
        "end_utc": str(pd.to_datetime(events.timestamp.max(), unit="s", utc=True)),
        "distinct_timestamps": int(events.timestamp.nunique()),
        "max_events_same_timestamp": int(events.groupby("timestamp").size().max()),
        "user_item_repeated_events": int(events[["user", "item"]].duplicated().sum()),
    }
