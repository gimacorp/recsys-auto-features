"""Изоляция моделей и предсказаний внутри каталога каждого запуска."""

import json
from pathlib import Path


def artifact_path(experiment: Path, kind: str, filename: str, *, reading: bool = False) -> Path:
    if kind not in {"models", "predictions"} or Path(filename).name != filename:
        raise ValueError("Недопустимый тип или имя артефакта")
    path = experiment / kind / filename
    if reading and not path.exists():
        provenance = experiment / "provenance.json"
        if provenance.exists():
            metadata = json.loads(provenance.read_text())
            if metadata.get("artifact_layout") == "run-local-v2" or "artifacts.py" in metadata.get(
                "source_sha256", {}
            ):
                return path
        # Совместимость с опубликованными результатами первой версии.
        legacy = Path("artifacts") / kind / experiment.name / filename
        if legacy.exists():
            return legacy
    return path
