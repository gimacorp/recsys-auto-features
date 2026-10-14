from types import SimpleNamespace

from recsys_auto.experiment import provenance


def test_provenance_works_without_git_checkout(tmp_path, monkeypatch):
    configuration = tmp_path / "config.json"
    configuration.write_text("{}")
    monkeypatch.setattr(
        "recsys_auto.experiment.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", returncode=128),
    )
    result = provenance(configuration, {"sha256": "test"})
    assert result["git_revision"] is None
    assert "captured_utc" in result
    assert "started_utc" not in result
    assert result["source_sha256"]
