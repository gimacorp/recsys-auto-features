"""Генерация таблиц и научных графиков непосредственно из результатов запусков."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/recsys-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = {
    "item_prior": "История фильма",
    "static": "Статические признаки",
    "manual": "Ручные признаки",
    "all_packs": "Все пакеты",
    "random": "Случайное дополнение",
    "residual": "Направляемое дополнение",
    "auto_random": "Поиск с нуля: случайный",
    "auto_residual": "Поиск с нуля: направляемый",
}


def table(frame: pd.DataFrame, digits=6) -> str:
    """Markdown без дополнительной зависимости tabulate."""
    headers = [str(name) for name in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        formatted = [
            f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def flatten(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": row["fold"],
                "seed": row["seed"],
                "method": row["method"],
                "features": row.get("feature_count", 0),
                **row["test"],
            }
            for row in records
        ]
    )


def generate(experiment: Path, output: Path) -> None:
    inputs = {
        name: json.loads((experiment / f"{name}.json").read_text())
        for name in ["results", "catalogue", "diagnostics", "from_static", "audit"]
    }
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    primary = flatten(inputs["results"]["records"])
    automatic = flatten(inputs["from_static"]["records"])
    summary = (
        primary.groupby("method", sort=False)[
            ["logloss", "auc", "brier", "ndcg@10", "recall@10", "features"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(experiment / "summary.csv", index=False)
    sections = [
        "# Таблицы, рассчитанные из артефактов",
        "",
        "Таблицы генерируются командой `python -m recsys_auto.report`. Основная таблица — равновзвешенное среднее по двум fold и трём seed. Пользовательское ранжирование относится к наблюдаемым кандидатам.",
        "",
        "## Основной эксперимент",
        "",
        table(summary),
        "",
        "## Все запуски основного эксперимента",
        "",
        table(primary.drop(columns=["features", "rows", "ranking_users"])),
        "",
        "## Поиск с исходными статическими признаками",
        "",
        table(
            automatic.groupby("method")[["logloss", "auc", "ndcg@10", "recall@10", "features"]]
            .mean()
            .reset_index()
        ),
        "",
        table(automatic[["fold", "seed", "method", "features", "logloss", "auc"]]),
        "",
        "## Выбранные пакеты",
        "",
    ]
    chosen = []
    for source in ["results", "from_static"]:
        for row in inputs[source]["records"]:
            if "selected_packs" in row:
                chosen.append(
                    {
                        "fold": row["fold"],
                        "seed": row["seed"],
                        "method": row["method"],
                        "selected": ", ".join(row["selected_packs"]) or "∅",
                    }
                )
    sections += [
        table(pd.DataFrame(chosen)),
        "",
        "## Парные интервалы разности log loss",
        "",
        "Разность = метод − ручная модель. Отрицательная разность соответствует улучшению. Интервалы рассчитаны cluster bootstrap по пользователям отдельно для каждого fold/seed; множественные сравнения требуют осторожной интерпретации.",
        "",
    ]
    differences = []
    for row in inputs["results"]["records"]:
        if row["method"] in ["random", "residual", "all_packs"]:
            delta = row["vs_manual"]
            differences.append(
                {
                    "fold": row["fold"],
                    "seed": row["seed"],
                    "method": row["method"],
                    "delta": delta["logloss_delta"],
                    "low": delta["logloss_ci95"][0],
                    "high": delta["logloss_ci95"][1],
                }
            )
    delta_frame = pd.DataFrame(differences)
    sections += [table(delta_frame), "", "## Фиксированный каталог", ""]
    catalogue = pd.DataFrame(inputs["catalogue"]["records"])
    catalogue_table = (
        catalogue.groupby(["fold", "method"])[["ndcg@10", "recall@10", "users", "mean_candidates"]]
        .mean()
        .reset_index()
    )
    sections += [
        table(catalogue_table),
        "",
        "## Абляции и замороженная история",
        "",
        table(
            flatten(inputs["diagnostics"]["records"])[
                ["fold", "method", "logloss", "auc", "ndcg@10", "recall@10"]
            ]
        ),
        "",
        "## Стоимость поиска",
        "",
    ]
    costs = []
    for source in ["results", "from_static"]:
        for row in inputs[source]["records"]:
            if "search_evaluations" in row:
                costs.append(
                    {
                        "method": row["method"],
                        "evaluations": row["search_evaluations"],
                        "fit_seconds": row["search_fit_seconds"],
                        "scoring_seconds": row["search_scoring_seconds"],
                        "features": row["feature_count"],
                    }
                )
    sections += [
        table(pd.DataFrame(costs).groupby("method").mean().reset_index(), digits=3),
        "",
        "Время измерено на одной машине и включает текущую нагрузку. Основной эксперимент, дополнительные проверки и поиск с нуля запускались в разных условиях нагрузки; сравнение времени между экспериментами требует отдельного изолированного benchmark.",
        "",
    ]
    (output / "generated_results.md").write_text("\n".join(sections), encoding="utf-8")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for fold, ax in enumerate(axes):
        for offset, method, color in [(-0.12, "random", "#868686"), (0.12, "residual", "#007f89")]:
            group = delta_frame[
                (delta_frame.fold == fold) & (delta_frame.method == method)
            ].sort_values("seed")
            x = np.arange(len(group)) + offset
            ax.errorbar(
                x,
                group.delta,
                yerr=[group.delta - group.low, group.high - group.delta],
                fmt="o",
                capsize=4,
                color=color,
                label=NAMES[method],
            )
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xticks(np.arange(3), inputs["results"]["config"]["seeds"])
        ax.set_title(f"Период {fold + 1}")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Δ log loss к ручной модели")
        ax.legend(fontsize=8)
    fig.savefig(figures / "paired_logloss.png", dpi=180)
    plt.close(fig)
    merged = pd.concat([primary, automatic], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = {
        "manual": "#222222",
        "all_packs": "#777777",
        "auto_random": "#dd8c31",
        "auto_residual": "#007f89",
    }
    for fold, ax in enumerate(axes):
        for method, color in colors.items():
            group = merged[(merged.fold == fold) & (merged.method == method)]
            ax.scatter(group.features, group.logloss, color=color, alpha=0.4, s=25)
            ax.scatter(
                group.features.mean(),
                group.logloss.mean(),
                color=color,
                s=90,
                label=NAMES[method],
                edgecolors="white",
            )
        ax.set_title(f"Период {fold + 1}")
        ax.set_xlabel("Число признаков")
        ax.set_ylabel("Test log loss")
        ax.legend(fontsize=8)
    fig.savefig(figures / "feature_quality.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    methods = [
        "item_prior",
        "static",
        "manual",
        "auto_random",
        "auto_residual",
        "random",
        "residual",
        "all_packs",
    ]
    for fold, ax in enumerate(axes):
        group = (
            catalogue[catalogue.fold == fold].groupby("method")["ndcg@10"].mean().reindex(methods)
        )
        ax.barh(
            [NAMES[name] for name in methods],
            group,
            color=["#aaa", "#ccc", "#222", "#dd8c31", "#007f89", "#dd8c31", "#007f89", "#777"],
        )
        ax.set_title(f"Фиксированный каталог: период {fold + 1}")
        ax.set_xlabel("NDCG@10")
    fig.savefig(figures / "catalogue_ndcg.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сгенерировать таблицы и графики отчёта")
    parser.add_argument("--experiment", type=Path, default=Path("artifacts/full"))
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()
    generate(args.experiment, args.output)
