"""Абляции ручных признаков и чувствительность к обновлению истории."""

import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier

from .experiment import prepare, provenance, write_json
from .grammar import packs
from .metrics import evaluate, paired_bootstrap
from .model import fit_validation, refit
from .snapshot import snapshot_features
from .split import temporal_fold


def diagnostics(data_root: Path, experiment_root: Path) -> dict:
    result = json.loads((experiment_root / "results.json").read_text())
    config = result["config"]
    events, features, manifest, _ = prepare(data_root, config, Path("artifacts/cache"))
    rows = []
    for fold_index, fractions in enumerate(config["folds"]):
        fold = temporal_fold(events, fractions)
        baseline = next(
            row
            for row in result["records"]
            if row["fold"] == fold_index
            and row["seed"] == config["seeds"][0]
            and row["method"] == "manual"
        )
        model_root = Path("artifacts/models") / experiment_root.name
        model = CatBoostClassifier()
        model.load_model(str(model_root / f"fold{fold_index}-seed{baseline['seed']}-manual.cbm"))
        online_prediction = model.predict_proba(features.iloc[fold.test][baseline["columns"]])[:, 1]
        for entity in ["user", "item", "user_genre", "genre"]:
            columns = [
                column for column in baseline["columns"] if not column.startswith(entity + "__")
            ]
            fitted = fit_validation(
                features, events.label, fold, columns, baseline["params"], config, baseline["seed"]
            )
            _, prediction, seconds = refit(
                features, events.label, fold, fitted, baseline["params"], config, baseline["seed"]
            )
            rows.append(
                {
                    "fold": fold_index,
                    "seed": baseline["seed"],
                    "method": "drop_" + entity,
                    "validation_logloss": fitted.logloss,
                    "feature_count": len(columns),
                    "iterations": fitted.model.tree_count_,
                    "seconds": fitted.seconds + seconds,
                    "test": evaluate(events.iloc[fold.test], prediction),
                    "vs_manual": paired_bootstrap(
                        events.iloc[fold.test],
                        online_prediction,
                        prediction,
                        config["bootstrap_samples"],
                    ),
                }
            )
        query = events.iloc[fold.test].copy()
        query["timestamp"] = int(events.iloc[fold.test].timestamp.min())
        windows = list(dict.fromkeys(config["manual_days"] + config["candidate_days"]))
        frozen = snapshot_features(events.iloc[fold.development], query, packs(windows))
        for method in ["manual", "random", "residual"]:
            record = next(
                row
                for row in result["records"]
                if row["fold"] == fold_index
                and row["seed"] == baseline["seed"]
                and row["method"] == method
            )
            model = CatBoostClassifier()
            model.load_model(
                str(model_root / f"fold{fold_index}-seed{baseline['seed']}-{method}.cbm")
            )
            prediction = model.predict_proba(frozen[record["columns"]])[:, 1]
            rows.append(
                {
                    "fold": fold_index,
                    "seed": baseline["seed"],
                    "method": "frozen_" + method,
                    "test": evaluate(events.iloc[fold.test], prediction),
                    "online_test": record["test"],
                }
            )
        print(f"diagnostics fold={fold_index} completed", flush=True)
    output = {
        "records": rows,
        "scope": "exploratory-first-seed-only",
        "provenance": provenance(experiment_root / "config.json", manifest),
    }
    write_json(experiment_root / "diagnostics.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Абляции и замороженная история")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/full"))
    args = parser.parse_args()
    diagnostics(args.data, args.experiment)
