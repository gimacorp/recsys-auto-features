"""Проверка экспериментальной конфигурации до загрузки данных и обучения."""

import math

from .data import DATASETS


def validate_config(config: dict) -> None:
    if config.get("dataset") not in DATASETS:
        raise ValueError("Неизвестный dataset")
    seeds = config.get("seeds", [])
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("seeds должны быть уникальными неотрицательными целыми")
    folds = config.get("folds", [])
    if not folds or any(
        len(fold) != 3 or not (0 < fold[0] < fold[1] < fold[2] <= 1) for fold in folds
    ):
        raise ValueError("Ожидаются возрастающие границы train/validation/test")
    for key in ["manual_days", "candidate_days"]:
        values = config.get(key, [])
        if len(values) != len(set(values)) or any(
            value is not None and (type(value) is not int or value < 1) for value in values
        ):
            raise ValueError(f"{key}: ожидаются уникальные положительные целые окна или null")
    if None not in config["manual_days"]:
        raise ValueError("manual_days должен включать null для контроля item_prior")
    if set(config["manual_days"]) & set(config["candidate_days"]):
        raise ValueError("Ручные и дополнительные окна должны различаться")
    for key in [
        "search_rounds",
        "proposals_per_round",
        "max_iterations",
        "early_stopping_rounds",
        "thread_count",
        "bootstrap_samples",
    ]:
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} должен быть положительным целым")
    improvement = config.get("min_improvement", -1)
    if (
        not isinstance(improvement, (int, float))
        or not math.isfinite(improvement)
        or improvement < 0
    ):
        raise ValueError("min_improvement должен быть конечным неотрицательным числом")
    if not config.get("tuning_grid"):
        raise ValueError("tuning_grid должен содержать хотя бы одну конфигурацию")
