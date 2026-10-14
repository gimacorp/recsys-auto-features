"""Исполнение зафиксированного протокола и сохранение проверяемых артефактов."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import artifact_path
from .config import validate_config
from .data import describe, download, load_for_config
from .features import STATIC_COLUMNS, build_features
from .grammar import packs
from .metrics import evaluate, paired_bootstrap, segment_metrics
from .model import fit_validation, refit, tune
from .search import search
from .split import temporal_fold


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def provenance(config_path: Path, manifest: dict) -> dict:
    source = Path(__file__).parent
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version(name) for name in ["numpy", "pandas", "catboost", "scikit-learn", "scipy"]
        },
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.glob("*.py"))
        },
        "dataset": manifest,
        "artifact_layout": "run-local-v2",
        "started_utc": str(pd.Timestamp.now(tz="UTC")),
    }


def prepare(data_root: Path, config: dict, cache_root: Path) -> tuple:
    manifest = download(data_root, config["dataset"])
    events = load_for_config(data_root, config)
    windows = list(dict.fromkeys(config["manual_days"] + config["candidate_days"]))
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "windows": windows,
                "data": manifest,
                "cohort_modulus": config.get("cohort_modulus", 1),
                "engine": Path(__file__).with_name("features.py").read_text(),
                "grammar": Path(__file__).with_name("grammar.py").read_text(),
                "loader": Path(__file__).with_name("data.py").read_text(),
            }
        ).encode()
    ).hexdigest()[:16]
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / f"features-{fingerprint}.pkl"
    stats_path = cache.with_suffix(".json")
    if cache.exists() and stats_path.exists():
        features = pd.read_pickle(cache)
        stats = json.loads(stats_path.read_text())
        stats["cache_hit"] = True
    else:
        features, stats = build_features(events, windows)
        features.to_pickle(cache)
        stats["cache_hit"] = False
        write_json(stats_path, stats)
    return events, features, manifest, stats


def run(config_path: Path, data_root: Path, output: Path, resume: bool = False) -> dict:
    config = json.loads(config_path.read_text())
    validate_config(config)
    if (output / "provenance.json").exists() and not resume:
        raise ValueError("Каталог уже содержит запуск. Выберите новый --output или --resume")
    if resume and not (output / "provenance.json").exists():
        raise ValueError("Для --resume нужен каталог существующего запуска с provenance.json")
    started = time.perf_counter()
    events, features, manifest, feature_stats = prepare(data_root, config, Path("artifacts/cache"))
    provenance_record = provenance(config_path, manifest)
    if resume and (output / "provenance.json").exists():
        old = json.loads((output / "provenance.json").read_text())
        for key in ["config_sha256", "source_sha256", "dataset"]:
            if old[key] != provenance_record[key]:
                raise ValueError(f"Продолжение требует идентичного {key}")
    else:
        write_json(output / "provenance.json", provenance_record)
    write_json(output / "config.json", config)
    write_json(output / "dataset_summary.json", describe(events))
    write_json(output / "feature_stats.json", feature_stats)
    manual = packs(config["manual_days"])
    candidates = packs(config["candidate_days"])
    records = []
    fold_records = []
    for fold_index, fractions in enumerate(config["folds"]):
        fold = temporal_fold(events, fractions)
        fold_records.append({"fold": fold_index, "fractions": fractions, **fold.describe(events)})
        manual_columns = STATIC_COLUMNS + [column for pack in manual for column in pack.columns]
        nonconstant = set(features.iloc[fold.train].nunique()[lambda counts: counts > 1].index)
        manual_columns = [column for column in manual_columns if column in nonconstant]
        all_columns = list(
            dict.fromkeys(
                manual_columns
                + [
                    column
                    for pack in candidates
                    for column in pack.columns
                    if column in nonconstant
                ]
            )
        )
        tuning_path = output / f"tuning-fold{fold_index}.json"
        if resume and tuning_path.exists():
            tuning_record = json.loads(tuning_path.read_text())
            params = tuning_record["best_params"]
        else:
            params, trials = tune(features, events.label, fold, manual_columns, config)
            write_json(tuning_path, {"best_params": params, "trials": trials})
        for seed in config["seeds"]:
            checkpoint = output / f"fold{fold_index}-seed{seed}.json"
            if resume and checkpoint.exists():
                records.extend(json.loads(checkpoint.read_text()))
                continue
            baseline = fit_validation(
                features, events.label, fold, manual_columns, params, config, seed
            )
            static = fit_validation(
                features, events.label, fold, STATIC_COLUMNS, params, config, seed
            )
            all_packs = fit_validation(
                features, events.label, fold, all_columns, params, config, seed
            )
            methods = {"manual": baseline, "static": static, "all_packs": all_packs}
            searches = {}
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
                methods[strategy] = found.fitted
                searches[strategy] = found
            predictions = {}
            seed_records = []
            for method, fitted in methods.items():
                model, prediction, seconds = refit(
                    features, events.label, fold, fitted, params, config, seed
                )
                predictions[method] = prediction
                model_path = artifact_path(
                    output, "models", f"fold{fold_index}-seed{seed}-{method}.cbm"
                )
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save_model(str(model_path))
                record = {
                    "fold": fold_index,
                    "seed": seed,
                    "method": method,
                    "params": params,
                    "iterations": fitted.model.tree_count_,
                    "columns": fitted.columns,
                    "feature_count": len(fitted.columns),
                    "validation_logloss": fitted.logloss,
                    "validation_auc": fitted.auc,
                    "validation_fit_seconds": fitted.seconds,
                    "refit_seconds": seconds,
                    "test": evaluate(events.iloc[fold.test], prediction),
                    "segments": segment_metrics(
                        events.iloc[fold.development], events.iloc[fold.test], prediction
                    ),
                }
                if method in searches:
                    found = searches[method]
                    record.update(
                        {
                            "selected_packs": found.selected_packs,
                            "trials": found.trials,
                            "search_scoring_seconds": found.scoring_seconds,
                            "search_fit_seconds": found.fit_seconds,
                            "search_evaluations": len(found.trials),
                            "available_packs": found.available_packs,
                        }
                    )
                seed_records.append(record)
                print(
                    f"test fold={fold_index} seed={seed} {method}: "
                    f"logloss={record['test']['logloss']:.6f}, auc={record['test']['auc']:.6f}",
                    flush=True,
                )
            popularity = features.iloc[fold.test]["item__all__positive_rate"].to_numpy()
            predictions["item_prior"] = popularity
            seed_records.append(
                {
                    "fold": fold_index,
                    "seed": seed,
                    "method": "item_prior",
                    "feature_count": 1,
                    "test": evaluate(events.iloc[fold.test], popularity),
                    "segments": segment_metrics(
                        events.iloc[fold.development], events.iloc[fold.test], popularity
                    ),
                }
            )
            for record in seed_records:
                if record["method"] != "manual":
                    record["vs_manual"] = paired_bootstrap(
                        events.iloc[fold.test],
                        predictions["manual"],
                        predictions[record["method"]],
                        samples=config["bootstrap_samples"],
                        seed=2026,
                    )
                if record["method"] == "residual":
                    record["vs_random"] = paired_bootstrap(
                        events.iloc[fold.test],
                        predictions["random"],
                        predictions["residual"],
                        samples=config["bootstrap_samples"],
                        seed=2026,
                    )
            prediction_path = artifact_path(
                output, "predictions", f"fold{fold_index}-seed{seed}.npz"
            )
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(prediction_path, **predictions)
            write_json(checkpoint, seed_records)
            records.extend(seed_records)
    result = {
        "config": config,
        "folds": fold_records,
        "records": records,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    write_json(output / "results.json", result)
    write_json(output / "folds.json", fold_records)
    return result
