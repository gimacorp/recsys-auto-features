from pathlib import Path

import pytest

from recsys_auto.artifacts import artifact_path


def test_equal_directory_names_have_independent_storage(tmp_path):
    first = artifact_path(tmp_path / "first" / "run", "models", "model.cbm")
    second = artifact_path(tmp_path / "second" / "run", "models", "model.cbm")
    assert first != second
    assert first.parent == tmp_path / "first" / "run" / "models"


def test_legacy_fallback_is_read_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy = Path("artifacts/models/run/model.cbm")
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy")
    experiment = Path("new/run")
    assert artifact_path(experiment, "models", "model.cbm", reading=True) == legacy
    assert artifact_path(experiment, "models", "model.cbm") == experiment / "models/model.cbm"


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        artifact_path(tmp_path, "models", "../model.cbm")
