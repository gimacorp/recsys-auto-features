"""Сводный график компромисса компактности и средней потери на трёх проверках."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/recsys-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def transfer_plot(experiments: list[Path], output: Path):
    labels, deltas, lows, highs, ratios = [], [], [], [], []
    for path in experiments:
        summary = json.loads((path / "statistics.json").read_text())
        config = json.loads((path / "config.json").read_text())
        paired = json.loads((path / "from_static.json").read_text())["config"].get(
            "pair_long_history", False
        )
        labels.append(
            "ML100K\nразработка"
            if config["dataset"] == "ml-100k"
            else f"ML1M: когорта {config.get('cohort_offset', 0)}\n"
            + ("парные горизонты" if paired else "исходный поиск")
        )
        row = next(
            row
            for row in summary["comparisons"]
            if row["method"] == "auto_residual" and row["reference"] == "manual"
        )
        deltas.append(row["delta"])
        lows.append(row["ci95"][0])
        highs.append(row["ci95"][1])
        ratios.append(summary["feature_ratio"])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    indices = np.arange(len(labels))
    colors = ["#777777", "#b94935", "#007f89"][: len(labels)]
    for index, (delta, low, high, color) in enumerate(
        zip(deltas, lows, highs, colors, strict=True)
    ):
        axes[0].errorbar(
            index, delta, yerr=[[delta - low], [high - delta]], fmt="o", capsize=5, color=color
        )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].axhline(0.01, color="#b94935", linestyle="--", label="Допуск для проверок ML1M: 0.01")
    axes[0].set_xticks(indices, labels)
    axes[0].set_ylabel("Δ log loss: автоматический − ручной")
    axes[0].set_title("Средняя по seed потеря и 95% интервал")
    axes[0].legend(fontsize=8)
    axes[1].bar(indices, ratios, color=colors)
    axes[1].axhline(0.6, color="#b94935", linestyle="--", label="Предел числа столбцов: 60%")
    axes[1].set_xticks(indices, labels)
    axes[1].set_ylim(0, 0.75)
    axes[1].set_ylabel("Относительное число входных столбцов")
    axes[1].set_title("Компактность автоматического представления")
    axes[1].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сводная визуализация переноса качества")
    parser.add_argument(
        "--experiments",
        type=Path,
        nargs="+",
        default=[Path("artifacts/full"), Path("artifacts/external"), Path("artifacts/replication")],
    )
    parser.add_argument("--output", type=Path, default=Path("docs/figures/transfer_validation.png"))
    args = parser.parse_args()
    transfer_plot(args.experiments, args.output)
