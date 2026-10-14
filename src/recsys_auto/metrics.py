"""Классификационные метрики и ранжирование наблюдаемых кандидатов."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def ranking_per_user(events: pd.DataFrame, prediction: np.ndarray, k: int = 10) -> pd.DataFrame:
    frame = events[["user", "item", "label"]].copy()
    frame["prediction"] = prediction
    rows = []
    discounts = 1 / np.log2(np.arange(2, k + 2))
    for user, group in frame.groupby("user", sort=True):
        positives = int(group.label.sum())
        if len(group) < 2 or positives == 0 or positives == len(group):
            continue
        ordered = group.sort_values(["prediction", "item"], ascending=[False, True], kind="stable")
        hits = ordered.label.to_numpy()[:k]
        ideal = discounts[:min(positives, k)].sum()
        rows.append({"user": user, "ndcg@10": float((hits * discounts[:len(hits)]).sum() / ideal),
                     "recall@10": float(hits.sum() / positives), "candidates": len(group)})
    return pd.DataFrame(rows, columns=["user", "ndcg@10", "recall@10", "candidates"])


def evaluate(events: pd.DataFrame, prediction: np.ndarray) -> dict:
    y = events.label.to_numpy()
    ranking = ranking_per_user(events, prediction)
    return {
        "logloss": float(log_loss(y, prediction, labels=[0, 1])),
        "auc": float(roc_auc_score(y, prediction)) if np.unique(y).size == 2 else None,
        "brier": float(brier_score_loss(y, prediction)),
        "ndcg@10": float(ranking["ndcg@10"].mean()) if len(ranking) else None,
        "recall@10": float(ranking["recall@10"].mean()) if len(ranking) else None,
        "ranking_users": len(ranking), "rows": len(events),
    }


def paired_bootstrap(events: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray,
                     samples: int = 1000, seed: int = 2026) -> dict:
    """Парный cluster bootstrap: пользователи ресемплируются целиком."""
    epsilon = 1e-15
    y = events.label.to_numpy()
    def losses(pred):
        p = np.clip(pred, epsilon, 1 - epsilon)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    frame = pd.DataFrame({"user": events.user.to_numpy(),
                          "delta": losses(candidate) - losses(baseline)})
    grouped = frame.groupby("user").delta.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    sums = grouped["sum"].to_numpy()
    counts = grouped["count"].to_numpy()
    draws = rng.integers(0, len(grouped), size=(samples, len(grouped)))
    deltas = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    ranking_base = ranking_per_user(events, baseline).set_index("user")
    ranking_candidate = ranking_per_user(events, candidate).set_index("user")
    result = {"logloss_delta": float(frame.delta.mean()),
              "logloss_ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
              "users": len(grouped), "bootstrap_samples": samples}
    for metric in ["ndcg@10", "recall@10"]:
        difference = (ranking_candidate[metric] - ranking_base[metric]).to_numpy()
        if len(difference):
            index = rng.integers(0, len(difference), size=(samples, len(difference)))
            result[metric + "_delta"] = float(difference.mean())
            result[metric + "_ci95"] = np.quantile(difference[index].mean(axis=1), [0.025, 0.975]).tolist()
    return result


def segment_metrics(history: pd.DataFrame, test: pd.DataFrame, prediction: np.ndarray) -> dict:
    seen_users = set(history.user)
    seen_items = set(history.item)
    warm_user = test.user.isin(seen_users).to_numpy()
    warm_item = test.item.isin(seen_items).to_numpy()
    masks = {"warm_both": warm_user & warm_item, "cold_user": ~warm_user,
             "cold_item": ~warm_item, "cold_any": ~warm_user | ~warm_item}
    return {name: evaluate(test.iloc[np.flatnonzero(mask)], prediction[mask])
            for name, mask in masks.items() if mask.any()}
