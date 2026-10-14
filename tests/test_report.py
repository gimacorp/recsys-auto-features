import json

from recsys_auto.report import generate


def test_report_supports_one_fold_and_one_seed(tmp_path):
    # Искусственные значения нужны только для проверки формы входов и генерации файлов.
    metrics = {
        "logloss": 0.5,
        "auc": 0.7,
        "brier": 0.2,
        "ndcg@10": 0.4,
        "recall@10": 0.2,
        "rows": 10,
        "ranking_users": 2,
    }

    def record(method):
        return {
            "fold": 0,
            "seed": 17,
            "method": method,
            "feature_count": 40,
            "test": metrics,
            "vs_manual": {"logloss_delta": 0.0, "logloss_ci95": [0.0, 0.0]},
            "selected_packs": [],
            "search_evaluations": 2,
            "search_fit_seconds": 1.0,
            "search_scoring_seconds": 0.01,
        }

    methods = ["manual", "static", "random", "residual", "all_packs", "item_prior"]
    primary = [record(name) for name in methods]
    automatic = [record("auto_random"), record("auto_residual")]
    catalogue = [
        {
            "fold": 0,
            "seed": 17,
            "method": name,
            "ndcg@10": 0.1,
            "recall@10": 0.05,
            "users": 2,
            "mean_candidates": 100,
        }
        for name in [*methods, "auto_random", "auto_residual"]
    ]
    inputs = {
        "results": {"config": {"folds": [[0.5, 0.65, 0.8]], "seeds": [17]}, "records": primary},
        "from_static": {"records": automatic},
        "catalogue": {"records": catalogue},
        "diagnostics": {"records": [record("drop_user")]},
        "audit": {"status": "passed"},
    }
    source = tmp_path / "input"
    source.mkdir()
    for name, data in inputs.items():
        (source / f"{name}.json").write_text(json.dumps(data))
    destination = tmp_path / "report"
    generate(source, destination)
    assert (destination / "generated_results.md").exists()
    for name in ["paired_logloss", "feature_quality", "catalogue_ndcg"]:
        assert (destination / "figures" / f"{name}.png").stat().st_size > 1000
