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
from .model import fit_validation, make_model, refit
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
                model, prediction, seconds = refit(
                    features, events.label, fold, found.fitted, params, config, seed
                )
                model_root = Path("artifacts/models") / experiment_root.name
                model_root.mkdir(parents=True, exist_ok=True)
                model.save_model(
                    str(model_root / f"fold{fold_index}-seed{seed}-auto_{strategy}.cbm")
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


def export_saved_models(data_root: Path, experiment_root: Path) -> dict:
    """Восстановить выбранные модели из зафиксированных параметров без повторного поиска."""
    original = json.loads((experiment_root / "results.json").read_text())
    automatic = json.loads((experiment_root / "from_static.json").read_text())
    config = automatic["config"]
    events, features, manifest, _ = prepare(data_root, config, Path("artifacts/cache"))
    model_root = Path("artifacts/models") / experiment_root.name
    model_root.mkdir(parents=True, exist_ok=True)
    checks = []
    for row in automatic["records"]:
        fold = temporal_fold(events, config["folds"][row["fold"]])
        manual = next(
            record
            for record in original["records"]
            if record["fold"] == row["fold"]
            and record["seed"] == row["seed"]
            and record["method"] == "manual"
        )
        model = make_model(manual["params"], config, row["seed"], row["iterations"])
        model.fit(
            features.iloc[fold.development][row["columns"]], events.label.iloc[fold.development]
        )
        prediction = model.predict_proba(features.iloc[fold.test][row["columns"]])[:, 1]
        actual = evaluate(events.iloc[fold.test], prediction)
        for metric, expected in row["test"].items():
            if expected is not None and not np.isclose(
                actual[metric], expected, atol=1e-12, rtol=0
            ):
                raise ValueError(f"Восстановленная модель отличается по метрике {metric}")
        model.save_model(
            str(model_root / f"fold{row['fold']}-seed{row['seed']}-{row['method']}.cbm")
        )
        checks.append(
            {"fold": row["fold"], "seed": row["seed"], "method": row["method"], "status": "passed"}
        )
    output = {
        "records": checks,
        "status": "passed",
        "provenance": provenance(experiment_root / "config.json", manifest),
    }
    write_json(experiment_root / "model_export_audit.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Автоматический поиск с исходными статическими признаками"
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/full"))
    parser.add_argument("--config", type=Path, default=Path("configs/from_static.json"))
    parser.add_argument(
        "--export-only", action="store_true", help="Восстановить модели из готового журнала"
    )
    args = parser.parse_args()
    if args.export_only:
        export_saved_models(args.data, args.experiment)
    else:
        run_from_static(args.data, args.experiment, args.config)
