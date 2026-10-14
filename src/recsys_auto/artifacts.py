"""Изоляция моделей и предсказаний внутри каталога каждого запуска."""

from pathlib import Path


def artifact_path(experiment: Path, kind: str, filename: str, *, reading: bool = False) -> Path:
    if kind not in {"models", "predictions"} or Path(filename).name != filename:
        raise ValueError("Недопустимый тип или имя артефакта")
    path = experiment / kind / filename
    if reading and not path.exists():
        # Совместимость с опубликованными результатами первой версии.
        legacy = Path("artifacts") / kind / experiment.name / filename
        if legacy.exists():
            return legacy
    return path
