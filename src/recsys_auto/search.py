"""Поиск пакетов: одинаковая процедура принятия, разные стратегии предложения."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .grammar import FeaturePack
from .model import FitResult, fit_validation
from .split import TemporalFold


@dataclass
class SearchResult:
    fitted: FitResult
    selected_packs: list[str]
    trials: list[dict]
    scoring_seconds: float
    fit_seconds: float


def residual_scores(matrix: np.ndarray, residual: np.ndarray,
                    slices: dict[str, list[int]]) -> dict[str, float]:
    """Максимальная абсолютная корреляция стандартизованного признака с y-p."""
    centered = matrix - matrix.mean(axis=0)
    residual = residual - residual.mean()
    norms = np.linalg.norm(centered, axis=0) * np.linalg.norm(residual)
    correlation = np.abs(centered.T @ residual) / np.maximum(norms, 1e-12)
    return {name: float(correlation[index].max()) for name, index in slices.items()}


def search(features: pd.DataFrame, labels: pd.Series, fold: TemporalFold,
           baseline: FitResult, candidates: list[FeaturePack], params: dict,
           config: dict, seed: int, strategy: str) -> SearchResult:
    if strategy not in ["random", "residual"]:
        raise ValueError("Неизвестная стратегия поиска")
    # Доступность признака определяется по train, независимо от меток test.
    available = {
        pack.name: [column for column in pack.columns
                    if features.iloc[fold.train][column].nunique() > 1]
        for pack in candidates
    }
    available = {name: columns for name, columns in available.items() if columns}
    candidate_columns = list(dict.fromkeys(column for columns in available.values() for column in columns))
    matrix = features.iloc[fold.validation][candidate_columns].to_numpy(np.float64)
    positions = {column: index for index, column in enumerate(candidate_columns)}
    slices = {name: [positions[column] for column in columns] for name, columns in available.items()}
    rng = np.random.default_rng(seed)
    current = baseline
    selected = []
    tried = set()
    trials = []
    scoring_seconds = 0.0
    fit_seconds = 0.0
    for round_index in range(config["search_rounds"]):
        tick = time.perf_counter()
        remaining = sorted(set(available) - tried)
        count = min(config["proposals_per_round"], len(remaining))
        if not count:
            break
        if strategy == "residual":
            scores = residual_scores(matrix, labels.iloc[fold.validation].to_numpy() - current.prediction,
                                     {name: slices[name] for name in remaining})
            proposals = sorted(remaining, key=lambda name: (-scores[name], name))[:count]
        else:
            scores = {}
            proposals = rng.choice(remaining, size=count, replace=False).tolist()
        scoring_seconds += time.perf_counter() - tick
        fitted = []
        for name in proposals:
            result = fit_validation(features, labels, fold, current.columns + available[name],
                                    params, config, seed)
            fit_seconds += result.seconds
            fitted.append((name, result))
            trials.append({"round": round_index + 1, "pack": name, "strategy": strategy,
                           "proxy_score": scores.get(name), "logloss": result.logloss,
                           "auc": result.auc, "seconds": result.seconds,
                           "iterations": result.model.tree_count_, "accepted": False,
                           "starting_logloss": current.logloss, "starting_packs": selected.copy()})
            tried.add(name)
        best_name, best = min(fitted, key=lambda pair: pair[1].logloss)
        if best.logloss < current.logloss - config["min_improvement"]:
            selected.append(best_name)
            current = best
            next(trial for trial in reversed(trials)
                 if trial["round"] == round_index + 1 and trial["pack"] == best_name)["accepted"] = True
        print(f"search {strategy} seed={seed} round={round_index + 1}: "
              f"logloss={current.logloss:.6f}, selected={selected}", flush=True)
    return SearchResult(current, selected, trials, scoring_seconds, fit_seconds)
