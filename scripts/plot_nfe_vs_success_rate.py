#!/usr/bin/env python3
"""Plot success rate vs NFE for seven Franka Kitchen sub-tasks.

Compares FlowPolicy, DP-CNN, and DP-Transformer across NFE={1, 8, 32, 100}
(mean ± min–max band over 3 seeds).

Microwave / Slide Cabinet / Hinge Cabinet: hardcode dari Tabel 4.1, 4.3, 4.5
(+ Lampiran 4-21). Kettle / Top Burner / Bottom Burner / Light Switch: dari
eval kitchen_eval_nfe100 (seed 42/43/44 dan train0/1/2).

Usage:
  python scripts/plot_nfe_vs_success_rate.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator

ROOT = Path(__file__).resolve().parents[1]

NFE = [1, 8, 32, 100]
TASKS = [
    "Microwave",
    "Slide Cabinet",
    "Hinge Cabinet",
    "Kettle",
    "Top Burner",
    "Bottom Burner",
    "Light Switch",
]
MODELS = ["FlowPolicy", "DP-CNN", "DP-Transformer"]

MODEL_STYLE = {
    "FlowPolicy": {
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "o",
    },
    "DP-CNN": {
        "color": "#1f77b4",
        "linestyle": "--",
        "marker": "s",
    },
    "DP-Transformer": {
        "color": "#ff7f0e",
        "linestyle": ":",
        "marker": "^",
    },
}

data = {
    "FlowPolicy": {
        "Microwave": {
            1: [0.60, 0.64, 0.72],
            8: [0.73, 0.70, 0.73],
            32: [0.70, 0.66, 0.72],
            100: [0.72, 0.66, 0.72],
        },
        "Slide Cabinet": {
            1: [0.57, 0.64, 0.46],
            8: [0.59, 0.66, 0.61],
            32: [0.57, 0.58, 0.54],
            100: [0.65, 0.56, 0.59],
        },
        "Hinge Cabinet": {
            1: [0.02, 0.07, 0.07],
            8: [0.30, 0.32, 0.31],
            32: [0.42, 0.35, 0.40],
            100: [0.34, 0.43, 0.42],
        },
        "Kettle": {
            1: [0.62, 0.74, 0.67],
            8: [0.38, 0.35, 0.30],
            32: [0.41, 0.40, 0.33],
            100: [0.35, 0.37, 0.31],
        },
        "Top Burner": {
            1: [0.29, 0.26, 0.28],
            8: [0.58, 0.47, 0.62],
            32: [0.53, 0.56, 0.61],
            100: [0.55, 0.60, 0.56],
        },
        "Bottom Burner": {
            1: [0.82, 0.82, 0.82],
            8: [0.75, 0.84, 0.88],
            32: [0.73, 0.80, 0.78],
            100: [0.77, 0.81, 0.79],
        },
        "Light Switch": {
            1: [0.20, 0.22, 0.24],
            8: [0.47, 0.51, 0.34],
            32: [0.43, 0.44, 0.45],
            100: [0.45, 0.43, 0.41],
        },
    },
    "DP-CNN": {
        "Microwave": {
            1: [0.00, 0.00, 0.00],
            8: [0.01, 0.00, 0.00],
            32: [0.03, 0.00, 0.13],
            100: [0.73, 0.64, 0.70],
        },
        "Slide Cabinet": {
            1: [0.08, 0.21, 0.15],
            8: [0.02, 0.03, 0.38],
            32: [0.00, 0.00, 0.02],
            100: [0.57, 0.56, 0.58],
        },
        "Hinge Cabinet": {
            1: [0.00, 0.00, 0.00],
            8: [0.00, 0.00, 0.00],
            32: [0.00, 0.00, 0.00],
            100: [0.54, 0.51, 0.49],
        },
        "Kettle": {
            1: [0.00, 0.01, 0.01],
            8: [0.03, 0.02, 0.02],
            32: [0.00, 0.01, 0.02],
            100: [0.37, 0.44, 0.48],
        },
        "Top Burner": {
            1: [0.00, 0.01, 0.00],
            8: [0.00, 0.00, 0.01],
            32: [0.01, 0.00, 0.02],
            100: [0.53, 0.58, 0.44],
        },
        "Bottom Burner": {
            1: [0.01, 0.00, 0.00],
            8: [0.01, 0.01, 0.00],
            32: [0.01, 0.00, 0.03],
            100: [0.71, 0.69, 0.68],
        },
        "Light Switch": {
            1: [0.08, 0.06, 0.06],
            8: [0.05, 0.00, 0.04],
            32: [0.00, 0.00, 0.02],
            100: [0.59, 0.62, 0.65],
        },
    },
    "DP-Transformer": {
        "Microwave": {
            1: [0.00, 0.00, 0.00],
            8: [0.00, 0.00, 0.00],
            32: [0.08, 0.07, 0.00],
            100: [0.73, 0.63, 0.71],
        },
        "Slide Cabinet": {
            1: [0.11, 0.11, 0.15],
            8: [0.06, 0.04, 0.04],
            32: [0.10, 0.07, 0.07],
            100: [0.67, 0.54, 0.54],
        },
        "Hinge Cabinet": {
            1: [0.00, 0.00, 0.00],
            8: [0.00, 0.00, 0.00],
            32: [0.00, 0.00, 0.00],
            100: [0.32, 0.50, 0.64],
        },
        "Kettle": {
            1: [0.01, 0.01, 0.02],
            8: [0.01, 0.00, 0.02],
            32: [0.05, 0.02, 0.03],
            100: [0.60, 0.48, 0.59],
        },
        "Top Burner": {
            1: [0.00, 0.00, 0.01],
            8: [0.01, 0.00, 0.00],
            32: [0.09, 0.06, 0.07],
            100: [0.41, 0.51, 0.32],
        },
        "Bottom Burner": {
            1: [0.00, 0.00, 0.00],
            8: [0.02, 0.01, 0.03],
            32: [0.14, 0.07, 0.10],
            100: [0.65, 0.67, 0.64],
        },
        "Light Switch": {
            1: [0.09, 0.06, 0.14],
            8: [0.10, 0.06, 0.08],
            32: [0.05, 0.06, 0.18],
            100: [0.56, 0.61, 0.53],
        },
    },
}


def _stats_for_task(model: str, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.array([data[model][task][n] for n in NFE], dtype=float)
    return values.mean(axis=1), values.min(axis=1), values.max(axis=1)


def _style_axis(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(NFE))
    ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in NFE]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.grid(True, color="#d0d0d0", linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(15, 12), dpi=300, sharey=True, sharex=True
    )
    axes_flat = axes.ravel()

    fig.suptitle(
        "Success Rate terhadap NFE pada Tujuh Sub-Tugas Franka Kitchen",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    x = np.array(NFE, dtype=float)

    for idx, task in enumerate(TASKS):
        ax = axes_flat[idx]
        for model in MODELS:
            style = MODEL_STYLE[model]
            mean, lo, hi = _stats_for_task(model, task)
            ax.plot(
                x,
                mean,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=7,
                linewidth=1.8,
                label=model,
            )
            ax.fill_between(x, lo, hi, color=style["color"], alpha=0.18, linewidth=0)

        _style_axis(ax)
        ax.set_title(task, fontweight="bold", fontsize=12)
        if idx % ncols == 0:
            ax.set_ylabel("Success Rate")

    for idx in range(len(TASKS), nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.supxlabel("NFE (Number of Function Evaluations)", fontsize=11)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        columnspacing=2.0,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.98))

    out_png = ROOT / "gambar_4x_nfe_vs_success_rate.png"
    out_pdf = ROOT / "gambar_4x_nfe_vs_success_rate.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.show()


if __name__ == "__main__":
    main()
