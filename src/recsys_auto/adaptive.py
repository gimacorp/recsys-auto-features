"""Генерация временных окон из распределения интервалов обучающей истории."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .artifacts import artifact_path
from .experiment import prepare, provenance, write_json
from .features import grouped_history
from .grammar import ENTITIES, FeaturePack
from .metrics import evaluate, paired_bootstrap
from .model import fit_validation, refit
from .search import search
from .split import temporal_fold

QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]


def propose_windows(training, excluded_days: list) -> tuple[list[FeaturePack], dict]:
    """Только train timestamp/ключи; метки и будущие события исключены из предложения."""
    candidates = []
    description = {}
    for entity in ENTITIES:
        expanded = grouped_history(training, entity)
        distinct = expanded.drop_duplicates(["group", "timestamp"])
        gaps = distinct.groupby("group", sort=False).timestamp.diff().dropna().to_numpy() / 86400
        gaps = gaps[gaps > 0]
        windows = (
            []
            if not len(gaps)
            else sorted(
                {max(1, int(np.ceil(value))) for value in np.quantile(gaps, QUANTILES)}
                - set(excluded_days)
            )
        )
        candidates.extend(FeaturePack(entity, days) for days in windows)
        description[entity] = {"positive_gaps": len(gaps), "days": windows}
    return candidates, description


def run_adaptive(experiment: Path, data_root: Path) -> dict:
    started = time.perf_counter()
    original = json.loads((experiment / "results.json").read_text())
    config = original["config"]
    from .data import load_for_config

    events = load_for_config(data_root, config)
    candidates = {}
    descriptions = {}
    for index, fractions in enumerate(config["folds"]):
        fold = temporal_fold(events, fractions)
        candidates[index], descriptions[index] = propose_windows(
            events.iloc[fold.train], config["manual_days"]
        )
    derived_days = sorted({pack.days for pool in candidates.values() for pack in pool})
    derived_config = {**config, "candidate_days": derived_days}
    events, features, manifest, _ = prepare(data_root, derived_config, Path("artifacts/cache"))
    records = []
    for index, fractions in enumerate(config["folds"]):
        fold = temporal_fold(events, fractions)
        for seed in config["seeds"]:
            manual = next(
                row
                for row in original["records"]
                if row["fold"] == index and row["seed"] == seed and row["method"] == "manual"
            )
            baseline = fit_validation(
                features, events.label, fold, manual["columns"], manual["params"], config, seed
            )
            found = search(
                features,
                events.label,
                fold,
                baseline,
                candidates[index],
                manual["params"],
                config,
                seed,
                "residual",
            )
            _, prediction, seconds = refit(
                features, events.label, fold, found.fitted, manual["params"], config, seed
            )
            with np.load(
                artifact_path(
                    experiment, "predictions", f"fold{index}-seed{seed}.npz", reading=True
                )
            ) as archive:
                reference = archive["manual"]
            records.append(
                {
                    "fold": index,
                    "seed": seed,
                    "method": "adaptive_residual",
                    "feature_count": len(found.fitted.columns),
                    "selected_packs": found.selected_packs,
                    "trials": found.trials,
                    "available_packs": found.available_packs,
                    "search_evaluations": len(found.trials),
                    "search_fit_seconds": found.fit_seconds,
                    "search_scoring_seconds": found.scoring_seconds,
                    "refit_seconds": seconds,
                    "validation_logloss": found.fitted.logloss,
                    "test": evaluate(events.iloc[fold.test], prediction),
                    "vs_manual": paired_bootstrap(
                        events.iloc[fold.test], reference, prediction, config["bootstrap_samples"]
                    ),
                }
            )
            write_json(experiment / "adaptive.partial.json", {"records": records})
            print(
                f"adaptive fold={index} seed={seed}: {records[-1]['test']['logloss']:.6f}",
                flush=True,
            )
    result = {
        "scope": "exploratory on previously evaluated MovieLens 100K",
        "quantiles": QUANTILES,
        "training_windows": descriptions,
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
        "provenance": provenance(experiment / "config.json", manifest),
    }
    write_json(experiment / "adaptive.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Окна из train-интервалов и направляемый поиск")
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/full"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    run_adaptive(args.experiment, args.data)
