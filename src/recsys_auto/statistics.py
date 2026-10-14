"""Сравнение средней потери по seed с парным bootstrap пользователей."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .artifacts import artifact_path
from .experiment import prepare, provenance, write_json
from .split import temporal_fold


def macro_cluster_interval(
    groups: list[pd.DataFrame], samples: int = 1000, seed: int = 2026
) -> dict:
    """Общие draws пользователей для всех fold сохраняют межпериодную зависимость."""
    users = sorted(set().union(*(set(frame.index) for frame in groups)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(users), size=(samples, len(users)))
    values = []
    means = []
    for grouped in groups:
        aligned = grouped.reindex(users, fill_value=0)
        sums = aligned["sum"].to_numpy()
        counts = aligned["count"].to_numpy()
        values.append(sums[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1))
        means.append(float(sums.sum() / counts.sum()))
    distribution = np.mean(values, axis=0)
    return {
        "delta": float(np.mean(means)),
        "ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
        "users": len(users),
        "samples": samples,
        "folds": len(groups),
    }


def analyse(experiment: Path, data_root: Path) -> dict:
    primary = json.loads((experiment / "results.json").read_text())
    automatic = json.loads((experiment / "from_static.json").read_text())
    config = primary["config"]
    events, features, manifest, _ = prepare(data_root, config, Path("artifacts/cache"))
    pairs = [("auto_residual", "manual"), ("auto_residual", "auto_random"), ("residual", "manual")]
    grouped_pairs = {pair: [] for pair in pairs}
    summaries = []
    for fold_index, fractions in enumerate(config["folds"]):
        fold = temporal_fold(events, fractions)
        test = events.iloc[fold.test]
        all_records = [
            row for row in primary["records"] + automatic["records"] if row["fold"] == fold_index
        ]
        losses = {}
        for method in {name for pair in pairs for name in pair}:
            seed_losses = []
            for seed in config["seeds"]:
                record = next(
                    row for row in all_records if row["method"] == method and row["seed"] == seed
                )
                model = CatBoostClassifier()
                model.load_model(
                    str(
                        artifact_path(
                            experiment,
                            "models",
                            f"fold{fold_index}-seed{seed}-{method}.cbm",
                            reading=True,
                        )
                    )
                )
                probability = np.clip(
                    model.predict_proba(features.iloc[fold.test][record["columns"]])[:, 1],
                    1e-15,
                    1 - 1e-15,
                )
                y = test.label.to_numpy()
                loss = -(y * np.log(probability) + (1 - y) * np.log1p(-probability))
                if not np.isclose(loss.mean(), record["test"]["logloss"], atol=1e-12, rtol=0):
                    raise ValueError("Пересчитанная потеря модели отличается от журнала")
                seed_losses.append(loss)
            losses[method] = np.mean(seed_losses, axis=0)
            summaries.append(
                {
                    "fold": fold_index,
                    "method": method,
                    "mean_seed_logloss": float(losses[method].mean()),
                }
            )
        for pair in pairs:
            frame = pd.DataFrame(
                {"user": test.user.to_numpy(), "delta": losses[pair[0]] - losses[pair[1]]}
            )
            grouped_pairs[pair].append(frame.groupby("user").delta.agg(["sum", "count"]))
    comparisons = [
        {
            "method": method,
            "reference": reference,
            **macro_cluster_interval(groups, config["bootstrap_samples"]),
        }
        for (method, reference), groups in grouped_pairs.items()
    ]
    manual_count = np.mean(
        [row["feature_count"] for row in primary["records"] if row["method"] == "manual"]
    )
    auto_count = np.mean(
        [row["feature_count"] for row in automatic["records"] if row["method"] == "auto_residual"]
    )
    result = {
        "comparisons": comparisons,
        "summaries": summaries,
        "mean_manual_features": float(manual_count),
        "mean_auto_features": float(auto_count),
        "feature_ratio": float(auto_count / manual_count),
        "scope": "conditional on fixed seeds and test periods; shared user draws across folds",
        "provenance": provenance(experiment / "config.json", manifest),
    }
    if config["dataset"] == "ml-1m":
        comparison = next(
            row
            for row in comparisons
            if row["method"] == "auto_residual" and row["reference"] == "manual"
        )
        result["preregistered_criteria"] = {
            "logloss_margin": 0.01,
            "feature_ratio_limit": 0.6,
            "loss_passed": comparison["ci95"][1] < 0.01,
            "compactness_passed": auto_count / manual_count <= 0.6,
        }
    write_json(experiment / "statistics.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Средняя по seed потеря и совместный cluster bootstrap"
    )
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/full"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(analyse(args.experiment, args.data), ensure_ascii=False, indent=2))
