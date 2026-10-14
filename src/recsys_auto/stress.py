"""Диагностика временного сдвига без изменения выбранных моделей."""

import argparse
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from .artifacts import artifact_path
from .experiment import prepare, write_json
from .split import temporal_fold


def temporal_stress(experiment: Path, data_root: Path) -> dict:
    primary = json.loads((experiment / "results.json").read_text())
    automatic = json.loads((experiment / "from_static.json").read_text())
    config = primary["config"]
    events, features, _, _ = prepare(data_root, config, Path("artifacts/cache"))
    rows = []
    for fold_index, fractions in enumerate(config["folds"]):
        fold = temporal_fold(events, fractions)
        test = events.iloc[fold.test]
        elapsed = (test.timestamp.to_numpy() - test.timestamp.min()) / 86400
        labels = test.label.to_numpy()
        losses = {}
        for method in ["manual", "auto_random", "auto_residual"]:
            values = []
            for seed in config["seeds"]:
                record = next(
                    row
                    for row in primary["records"] + automatic["records"]
                    if row["fold"] == fold_index and row["method"] == method and row["seed"] == seed
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
                p = np.clip(
                    model.predict_proba(features.iloc[fold.test][record["columns"]])[:, 1],
                    1e-15,
                    1 - 1e-15,
                )
                values.append(-(labels * np.log(p) + (1 - labels) * np.log1p(-p)))
            losses[method] = np.mean(values, axis=0)
        for left, right in [(0, 7), (7, 30), (30, 90), (90, 365), (365, np.inf)]:
            mask = (elapsed >= left) & (elapsed < right)
            if not mask.any():
                continue
            metrics = {method: float(loss[mask].mean()) for method, loss in losses.items()}
            rows.append(
                {
                    "fold": fold_index,
                    "period_days": f"{left}–{right if np.isfinite(right) else '∞'}",
                    "events": int(mask.sum()),
                    "users": int(test.iloc[np.flatnonzero(mask)].user.nunique()),
                    "positive_fraction": float(labels[mask].mean()),
                    "logloss": metrics,
                    "auto_residual_minus_manual": metrics["auto_residual"] - metrics["manual"],
                }
            )
    result = {
        "scope": "exploratory temporal diagnostics; models and selections remain fixed",
        "records": rows,
    }
    write_json(experiment / "temporal_stress.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Потеря по интервалам после начала test")
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/external"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(temporal_stress(args.experiment, args.data), ensure_ascii=False, indent=2))
