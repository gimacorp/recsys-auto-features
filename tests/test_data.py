import numpy as np
import pandas as pd
import pytest

from recsys_auto.data import encode_genres, load_for_config


def test_genre_aliases_and_unknown_fallback():
    flags = encode_genres(pd.Series(["Children's|Sci-Fi|Film-Noir", "Drama|Comedy", ""]))
    assert flags.loc[0, ["Children", "SciFi", "FilmNoir"]].tolist() == [1, 1, 1]
    assert flags.loc[1, ["Drama", "Comedy"]].tolist() == [1, 1]
    assert flags.loc[2, "unknown"] == 1
    assert flags.loc[2].sum() == 1


def test_cohort_selection_preserves_complete_user_histories(monkeypatch):
    events = pd.DataFrame(
        {"user": [1, 6, 6, 12, 2], "event_id": np.arange(5), "label": [1, 0, 1, 1, 0]}
    )
    monkeypatch.setattr("recsys_auto.data.load", lambda root, dataset: events.copy())
    cohort = load_for_config(None, {"dataset": "ml-1m", "cohort_modulus": 6})
    assert cohort.user.tolist() == [6, 6, 12]
    assert cohort.event_id.tolist() == [0, 1, 2]
    assert cohort.label.tolist() == [0, 1, 1]
    events["label"] = 1 - events.label
    assert load_for_config(None, {"dataset": "ml-1m", "cohort_modulus": 6}).user.tolist() == [
        6,
        6,
        12,
    ]


def test_empty_cohort_rejected(monkeypatch):
    monkeypatch.setattr(
        "recsys_auto.data.load", lambda root, dataset: pd.DataFrame({"user": [1], "event_id": [0]})
    )
    with pytest.raises(ValueError, match="Пустая"):
        load_for_config(None, {"dataset": "ml-1m", "cohort_modulus": 6})


def test_bad_download_keeps_cache_retryable(tmp_path, monkeypatch):
    import io

    from recsys_auto.data import download

    monkeypatch.setattr(
        "recsys_auto.data.urllib.request.urlopen",
        lambda *args, **kwargs: io.BytesIO(b"bad payload"),
    )
    with pytest.raises(ValueError, match="Контрольная сумма"):
        download(tmp_path)
    assert not (tmp_path / "ml-100k.zip").exists()
    assert not (tmp_path / "ml-100k.part").exists()
