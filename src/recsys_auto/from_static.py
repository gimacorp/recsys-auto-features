"""Исследовательский поиск исторических признаков с исходной статической моделью."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .experiment import prepare, provenance, write_json
from .features import STATIC_COLUMNS
from .grammar import packs
from .metrics import evaluate, paired_bootstrap
from .model import fit_validation, refit
from .search import search
from .split import temporal_fold


def run_from_static(data_root: Path, experiment_root: Path, additional_path: Path) -> dict:
    tick = time.perf_counter()
    original = json.loads((experiment_root / "results.json").read_text())
    config = {**original["config"], **json.loads(additional_path.read_text())}
    events, features, manifest, _ = prepare(data_root, config, Path("artifacts/cache"))
    candidates = packs(list(dict.fromkeys(config["manual_days"] + config["candidate_days"])))
    records = []
    for fold_index, fractions in enumerate(config["folds"]):
        fold = temporal_fold(events, fractions)
        for seed in config["seeds"]:
            manual = next(
                row
                for row in original["records"]
                if row["fold"] == fold_index and row["seed"] == seed and row["method"] == "manual"
            )
            params = manual["params"]
            baseline = fit_validation(
                features, events.label, fold, STATIC_COLUMNS, params, config, seed
            )
            prediction_path = (
                Path("artifacts/predictions")
                / experiment_root.name
                / f"fold{fold_index}-seed{seed}.npz"
            )
            manual_prediction = np.load(prediction_path)["manual"]
            for strategy in ["random", "residual"]:
                found = search(
                    features,
                    events.label,
                    fold,
                    baseline,
                    candidates,
                    params,
                    config,
                    seed,
                    strategy,
                )
                _, prediction, seconds = refit(
                    features, events.label, fold, found.fitted, params, config, seed
                )
                record = {
                    "fold": fold_index,
                    "seed": seed,
                    "method": "auto_" + strategy,
                    "feature_count": len(found.fitted.columns),
                    "columns": found.fitted.columns,
                    "selected_packs": found.selected_packs,
                    "trials": found.trials,
                    "search_evaluations": len(found.trials),
                    "search_fit_seconds": found.fit_seconds,
                    "search_scoring_seconds": found.scoring_seconds,
                    "refit_seconds": seconds,
                    "iterations": found.fitted.model.tree_count_,
                    "validation_logloss": found.fitted.logloss,
                    "test": evaluate(events.iloc[fold.test], prediction),
                    "vs_manual": paired_bootstrap(
                        events.iloc[fold.test],
                        manual_prediction,
                        prediction,
                        config["bootstrap_samples"],
                    ),
                }
                records.append(record)
                write_json(experiment_root / "from_static.partial.json", {"records": records})
                print(
                    f"auto {strategy} fold={fold_index} seed={seed}: test logloss={record['test']['logloss']:.6f}",
                    flush=True,
                )
    result = {
        "config": config,
        "records": records,
        "elapsed_seconds": time.perf_counter() - tick,
        "provenance": provenance(additional_path, manifest),
    }
    write_json(experiment_root / "from_static.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Автоматический поиск с исходными статическими признаками"
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/full"))
    parser.add_argument("--config", type=Path, default=Path("configs/from_static.json"))
    args = parser.parse_args()
    run_from_static(args.data, args.experiment, args.config)
