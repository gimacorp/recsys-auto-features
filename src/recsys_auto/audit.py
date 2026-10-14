"""Проверка целостности артефактов и пересчёт метрик из локальных предсказаний."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import artifact_path
from .data import load
from .metrics import evaluate
from .split import temporal_fold


def audit(data_root: Path, experiment_root: Path) -> dict:
    result = json.loads((experiment_root / "results.json").read_text())
    config = result["config"]
    events = load(data_root)
    expected_records = len(config["folds"]) * len(config["seeds"]) * 6
    if len(result["records"]) != expected_records:
        raise ValueError("Ожидается полный набор методов, fold и seed")
    keys = [(row["fold"], row["seed"], row["method"]) for row in result["records"]]
    if len(set(keys)) != len(keys):
        raise ValueError("В итоговых записях есть дубли")
    metric_checks = 0
    search_checks = 0
    for fold_index, fractions in enumerate(config["folds"]):
        fold = temporal_fold(events, fractions)
        if events.iloc[fold.train].timestamp.max() >= events.iloc[fold.validation].timestamp.min():
            raise ValueError("Пересечение времён train и validation")
        if events.iloc[fold.validation].timestamp.max() >= events.iloc[fold.test].timestamp.min():
            raise ValueError("Пересечение времён validation и test")
        for seed in config["seeds"]:
            prediction_path = artifact_path(
                experiment_root, "predictions", f"fold{fold_index}-seed{seed}.npz", reading=True
            )
            with np.load(prediction_path) as archive:
                for row in result["records"]:
                    if row["fold"] != fold_index or row["seed"] != seed:
                        continue
                    actual = evaluate(events.iloc[fold.test], archive[row["method"]])
                    for name, expected in row["test"].items():
                        if expected is None:
                            if actual[name] is not None:
                                raise ValueError(f"Ожидается пустая метрика {name}")
                        elif not np.isclose(actual[name], expected, rtol=0, atol=1e-12):
                            raise ValueError(f"Метрика {name} отличается для {row['method']}")
                        metric_checks += 1
                    if row["method"] in ["random", "residual"]:
                        if (
                            len(row["trials"])
                            != config["search_rounds"] * config["proposals_per_round"]
                        ):
                            raise ValueError("Нарушен бюджет поиска")
                        accepted = [trial["pack"] for trial in row["trials"] if trial["accepted"]]
                        if accepted != row["selected_packs"]:
                            raise ValueError("Выбранные пакеты отличаются от журнала поиска")
                        for trial in row["trials"]:
                            if trial["accepted"] and not (
                                trial["logloss"]
                                < trial["starting_logloss"] - config["min_improvement"]
                            ):
                                raise ValueError(
                                    "Принят пакет с недостаточным улучшением validation"
                                )
                        search_checks += 1
    output = {
        "status": "passed",
        "records": expected_records,
        "metric_checks": metric_checks,
        "search_runs_checked": search_checks,
        "timestamp_boundaries_checked": len(config["folds"]) * 2,
    }
    from .experiment import write_json

    write_json(experiment_root / "audit.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Пересчитать и проверить результаты")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/full"))
    args = parser.parse_args()
    print(json.dumps(audit(args.data, args.experiment), ensure_ascii=False, indent=2))
