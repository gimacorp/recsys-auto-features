"""Дополнительный fixed-snapshot тест по всему доступному каталогу."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .data import GENRES, load
from .experiment import provenance, write_json
from .grammar import packs
from .snapshot import snapshot_features
from .split import temporal_fold


def retrieval_metrics(items: np.ndarray, scores: np.ndarray, positives: set[int], k=10) -> tuple:
    order = np.lexsort((items, -scores))[:k]
    hits = np.array([int(item in positives) for item in items[order]])
    discount = 1 / np.log2(np.arange(2, len(hits) + 2))
    ideal = (1 / np.log2(np.arange(2, min(len(positives), k) + 2))).sum()
    return float((hits * discount).sum() / ideal), float(hits.sum() / len(positives))


def catalogue_evaluation(data_root: Path, experiment_root: Path, batch_users: int = 8) -> dict:
    started = time.perf_counter()
    experiment = json.loads((experiment_root / "results.json").read_text())
    config = experiment["config"]
    events = load(data_root)
    records = []
    fold_descriptions = []
    intervals = []
    for fold_index, fractions in enumerate(config["folds"]):
        fold = temporal_fold(events, fractions)
        history = events.iloc[fold.development]
        test = events.iloc[fold.test]
        timestamp = int(test.timestamp.min())
        catalogue = history.drop_duplicates("item")[["item", "release_year", *GENRES, "genres"]]
        catalogue_items = set(catalogue.item)
        positive = test[(test.label == 1) & test.item.isin(catalogue_items)]
        truth = positive.groupby("user").item.apply(set).to_dict()
        users = sorted(truth)
        demographics = events.drop_duplicates("user").set_index("user")
        seen = history.groupby("user").item.apply(set).to_dict()
        warm = set(history.user)
        requested = packs(list(dict.fromkeys(config["manual_days"] + config["candidate_days"])))
        models = {}
        fold_records = [
            row
            for row in experiment["records"]
            if row["fold"] == fold_index and row["method"] != "item_prior"
        ]
        for record in fold_records:
            key = (record["method"], record["seed"])
            model = CatBoostClassifier()
            model.load_model(
                str(
                    Path("artifacts/models")
                    / experiment_root.name
                    / f"fold{fold_index}-seed{record['seed']}-{record['method']}.cbm"
                )
            )
            models[key] = (model, record["columns"])
        per_user = {key: [] for key in models}
        per_user[("item_prior", 0)] = []
        feature_seconds = 0.0
        for start in range(0, len(users), batch_users):
            queries = []
            for user in users[start : start + batch_users]:
                user_items = catalogue[~catalogue.item.isin(seen.get(user, set()))].copy()
                user_items["user"] = user
                for name in ["age", "gender", "occupation"]:
                    user_items[name] = demographics.loc[user, name]
                queries.append(user_items)
            query = pd.concat(queries, ignore_index=True)
            query["event_id"] = np.arange(len(query))
            query["timestamp"] = timestamp
            tick = time.perf_counter()
            features = snapshot_features(history, query, requested)
            feature_seconds += time.perf_counter() - tick
            prediction = {
                key: model.predict_proba(features[columns])[:, 1]
                for key, (model, columns) in models.items()
            }
            prediction[("item_prior", 0)] = features["item__all__positive_rate"].to_numpy()
            for user, indices in query.groupby("user").indices.items():
                items = query.iloc[indices].item.to_numpy()
                for key, scores in prediction.items():
                    ndcg, recall = retrieval_metrics(items, scores[indices], truth[user])
                    per_user[key].append(
                        {
                            "user": int(user),
                            "ndcg@10": ndcg,
                            "recall@10": recall,
                            "cold_user": user not in warm,
                            "candidates": len(items),
                        }
                    )
            print(
                f"catalogue fold={fold_index}: users {min(start + batch_users, len(users))}/{len(users)}",
                flush=True,
            )
        for (method, seed), values in per_user.items():
            frame = pd.DataFrame(values).set_index("user")
            records.append(
                {
                    "fold": fold_index,
                    "seed": seed,
                    "method": method,
                    "users": len(frame),
                    "ndcg@10": float(frame["ndcg@10"].mean()),
                    "recall@10": float(frame["recall@10"].mean()),
                    "mean_candidates": float(frame.candidates.mean()),
                    "segments": {
                        name: {
                            "users": int(mask.sum()),
                            "ndcg@10": float(frame.loc[mask, "ndcg@10"].mean()),
                            "recall@10": float(frame.loc[mask, "recall@10"].mean()),
                        }
                        for name, mask in {
                            "cold_user": frame.cold_user,
                            "warm_user": ~frame.cold_user,
                        }.items()
                        if mask.any()
                    },
                }
            )
        for seed in config["seeds"]:
            base = pd.DataFrame(per_user[("manual", seed)]).set_index("user")
            rng = np.random.default_rng(2026)
            draws = rng.integers(0, len(base), size=(config["bootstrap_samples"], len(base)))
            for method in ["random", "residual", "all_packs"]:
                candidate = pd.DataFrame(per_user[(method, seed)]).set_index("user")
                for metric in ["ndcg@10", "recall@10"]:
                    delta = (candidate[metric] - base[metric]).to_numpy()
                    intervals.append(
                        {
                            "fold": fold_index,
                            "seed": seed,
                            "method": method,
                            "metric": metric,
                            "delta_vs_manual": float(delta.mean()),
                            "ci95": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist(),
                        }
                    )
        fold_descriptions.append(
            {
                "fold": fold_index,
                "snapshot_utc": str(pd.to_datetime(timestamp, unit="s", utc=True)),
                "catalogue_items": len(catalogue),
                "users": len(users),
                "positive_events_in_catalogue": len(positive),
                "positive_events_total": int(test.label.sum()),
                "feature_seconds": feature_seconds,
            }
        )
    manifest = json.loads((data_root / "manifest.json").read_text())
    result = {
        "protocol": "fixed-snapshot-known-catalogue",
        "folds": fold_descriptions,
        "records": records,
        "paired_intervals": intervals,
        "elapsed_seconds": time.perf_counter() - started,
        "provenance": provenance(experiment_root / "config.json", manifest),
    }
    write_json(experiment_root / "catalogue.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Ранжирование каталога в фиксированный момент")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/full"))
    parser.add_argument("--batch-users", type=int, default=8)
    args = parser.parse_args()
    if args.batch_users < 1:
        parser.error("--batch-users должен быть положительным")
    catalogue_evaluation(args.data, args.experiment, args.batch_users)


if __name__ == "__main__":
    main()
