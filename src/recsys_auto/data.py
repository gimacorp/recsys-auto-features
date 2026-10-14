"""Загрузка официального архива и нормализация схемы MovieLens."""

from __future__ import annotations

import hashlib
import json
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ARCHIVE_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
DATASETS = {
    "ml-100k": (ARCHIVE_SHA256, ["u.data", "u.item", "u.user", "README"]),
    "ml-1m": (
        "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
        ["ratings.dat", "movies.dat", "users.dat", "README"],
    ),
}
GENRES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "FilmNoir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "SciFi",
    "Thriller",
    "War",
    "Western",
]


def download(root: Path, dataset: str = "ml-100k") -> dict:
    """Сохраняет архив и только необходимые файлы; исходные данные исключены из Git."""
    root.mkdir(parents=True, exist_ok=True)
    if dataset not in DATASETS:
        raise ValueError("Неизвестный датасет")
    expected_digest, members = DATASETS[dataset]
    url = f"https://files.grouplens.org/datasets/movielens/{dataset}.zip"
    archive = root / f"{dataset}.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context, timeout=60) as response:
            temporary.write_bytes(response.read())
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected_digest:
            temporary.unlink()
            raise ValueError("Контрольная сумма загрузки отличается от зафиксированной")
        temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected_digest:
        raise ValueError(f"Контрольная сумма архива отличается от зафиксированной: {digest}")
    with zipfile.ZipFile(archive) as source:
        for name in members:
            target = root / dataset / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(f"{dataset}/{name}"))
    manifest = {"url": url, "sha256": digest, "bytes": archive.stat().st_size}
    manifest_path = root / ("manifest.json" if dataset == "ml-100k" else f"{dataset}-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_100k(root: Path) -> pd.DataFrame:
    directory = root / "ml-100k"
    events = pd.read_csv(
        directory / "u.data", sep="\t", header=None, names=["user", "item", "rating", "timestamp"]
    )
    users = pd.read_csv(
        directory / "u.user",
        sep="|",
        header=None,
        names=["user", "age", "gender", "occupation", "zip"],
    )
    items = pd.read_csv(
        directory / "u.item",
        sep="|",
        header=None,
        encoding="latin-1",
        names=["item", "title", "release", "video_release", "url", *GENRES],
    )
    items["release_year"] = (
        pd.to_datetime(items["release"], format="%d-%b-%Y", errors="coerce")
        .dt.year.fillna(0)
        .astype(int)
    )
    events = events.merge(users.drop(columns="zip"), on="user", validate="many_to_one")
    events = events.merge(
        items[["item", "release_year", *GENRES]], on="item", validate="many_to_one"
    )
    events = events.sort_values(["timestamp", "user", "item"], kind="stable").reset_index(drop=True)
    events["event_id"] = np.arange(len(events))
    events["label"] = (events["rating"] >= 4).astype(np.int8)
    events["genres"] = [tuple(np.flatnonzero(row)) or (0,) for row in events[GENRES].to_numpy()]
    if len(events) != 100_000 or events[["user", "item"]].duplicated().any():
        raise ValueError("Нарушена ожидаемая схема MovieLens 100K")
    return events


def _load_1m(root: Path) -> pd.DataFrame:
    directory = root / "ml-1m"
    options = {"sep": "::", "engine": "python", "header": None, "encoding": "latin-1"}
    events = pd.read_csv(
        directory / "ratings.dat", names=["user", "item", "rating", "timestamp"], **options
    )
    users = pd.read_csv(
        directory / "users.dat", names=["user", "gender", "age", "occupation", "zip"], **options
    )
    items = pd.read_csv(directory / "movies.dat", names=["item", "title", "genre_names"], **options)
    items["release_year"] = items.title.str.extract(r"\((\d{4})\)\s*$")[0].fillna(0).astype(int)
    items[GENRES] = encode_genres(items.genre_names)
    events = events.merge(users.drop(columns="zip"), on="user", validate="many_to_one")
    events = events.merge(
        items[["item", "release_year", *GENRES]], on="item", validate="many_to_one"
    )
    events = events.sort_values(["timestamp", "user", "item"], kind="stable").reset_index(drop=True)
    events["event_id"] = np.arange(len(events))
    events["label"] = (events.rating >= 4).astype(np.int8)
    events["genres"] = [tuple(np.flatnonzero(row)) or (0,) for row in events[GENRES].to_numpy()]
    if len(events) != 1_000_209 or events[["user", "item"]].duplicated().any():
        raise ValueError("Нарушена ожидаемая схема MovieLens 1M")
    return events


def encode_genres(names: pd.Series) -> pd.DataFrame:
    aliases = {"Children": "Children's", "FilmNoir": "Film-Noir", "SciFi": "Sci-Fi"}
    genre_sets = names.fillna("").str.split("|").map(set)
    result = pd.DataFrame(
        {
            genre: genre_sets.map(lambda values: int(aliases.get(genre, genre) in values))
            for genre in GENRES
            if genre != "unknown"
        }
    )
    result["unknown"] = (result.sum(axis=1) == 0).astype(int)
    return result[GENRES]


def load(root: Path, dataset: str = "ml-100k") -> pd.DataFrame:
    if dataset == "ml-100k":
        return _load_100k(root)
    if dataset == "ml-1m":
        return _load_1m(root)
    raise ValueError("Неизвестный датасет")


def load_for_config(root: Path, config: dict) -> pd.DataFrame:
    events = load(root, config["dataset"])
    modulus = config.get("cohort_modulus", 1)
    if isinstance(modulus, bool) or not isinstance(modulus, int) or modulus < 1:
        raise ValueError("cohort_modulus должен быть положительным целым")
    if modulus > 1:
        events = events[events.user % modulus == 0].copy().reset_index(drop=True)
        events["event_id"] = np.arange(len(events))
    if events.empty:
        raise ValueError("Пустая когорта пользователей")
    return events


def describe(events: pd.DataFrame) -> dict:
    return {
        "events": len(events),
        "users": int(events.user.nunique()),
        "items": int(events.item.nunique()),
        "positive_fraction": float(events.label.mean()),
        "start_utc": str(pd.to_datetime(events.timestamp.min(), unit="s", utc=True)),
        "end_utc": str(pd.to_datetime(events.timestamp.max(), unit="s", utc=True)),
        "distinct_timestamps": int(events.timestamp.nunique()),
        "max_events_same_timestamp": int(events.groupby("timestamp").size().max()),
        "user_item_repeated_events": int(events[["user", "item"]].duplicated().sum()),
    }
