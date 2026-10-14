"""Единый бюджет обучения и подбор параметров сильного базового бустинга."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from .features import CAT_COLUMNS
from .split import TemporalFold


@dataclass
class FitResult:
    model: CatBoostClassifier
    prediction: np.ndarray
    logloss: float
    auc: float
    seconds: float
    columns: list[str]


def fit_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    fold: TemporalFold,
    columns: list[str],
    params: dict,
    config: dict,
    seed: int,
) -> FitResult:
    tick = time.perf_counter()
    model = make_model(params, config, seed, config["max_iterations"])
    model.fit(
        features.iloc[fold.train][columns],
        labels.iloc[fold.train],
        eval_set=(features.iloc[fold.validation][columns], labels.iloc[fold.validation]),
        early_stopping_rounds=config["early_stopping_rounds"],
        use_best_model=True,
    )
    prediction = model.predict_proba(features.iloc[fold.validation][columns])[:, 1]
    y = labels.iloc[fold.validation]
    return FitResult(
        model,
        prediction,
        float(log_loss(y, prediction)),
        float(roc_auc_score(y, prediction)),
        time.perf_counter() - tick,
        columns,
    )


def make_model(params: dict, config: dict, seed: int, iterations: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        **params,
        iterations=iterations,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=seed,
        thread_count=config["thread_count"],
        verbose=False,
        allow_writing_files=False,
        cat_features=CAT_COLUMNS,
        has_time=True,
        one_hot_max_size=2000,
    )


def tune(
    features: pd.DataFrame, labels: pd.Series, fold: TemporalFold, columns: list[str], config: dict
) -> tuple[dict, list[dict]]:
    trials = []
    for params in config["tuning_grid"]:
        result = fit_validation(features, labels, fold, columns, params, config, config["seeds"][0])
        trials.append(
            {
                "params": params,
                "logloss": result.logloss,
                "auc": result.auc,
                "iterations": result.model.tree_count_,
                "seconds": result.seconds,
            }
        )
        print(f"tune {params}: logloss={result.logloss:.6f}", flush=True)
    best = min(trials, key=lambda trial: trial["logloss"])
    return best["params"], trials


def refit(
    features: pd.DataFrame,
    labels: pd.Series,
    fold: TemporalFold,
    selected: FitResult,
    params: dict,
    config: dict,
    seed: int,
) -> tuple:
    tick = time.perf_counter()
    model = make_model(params, config, seed, selected.model.tree_count_)
    model.fit(features.iloc[fold.development][selected.columns], labels.iloc[fold.development])
    prediction = model.predict_proba(features.iloc[fold.test][selected.columns])[:, 1]
    return model, prediction, time.perf_counter() - tick
