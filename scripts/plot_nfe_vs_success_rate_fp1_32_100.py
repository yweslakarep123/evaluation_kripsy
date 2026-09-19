#!/usr/bin/env python3
"""Success rate vs NFE for 7 Kitchen tasks — FP NFE 1–32+100, DP @ NFE=100 only.

Same layout as gambar_4x_nfe_vs_success_rate.png, but FlowPolicy uses the
dense NFE sweep from kitchen_eval_nfe100 (summary.csv). DP is shown only at
NFE=100 (low-NFE DP is excluded from fair comparison).

Usage:
  python scripts/plot_nfe_vs_success_rate_fp1_32_100.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data/kitchen_eval_plots/nfe100/summary.csv"
OUT_DIR = ROOT / "data/kitchen_eval_plots/nfe100"
OUT_STEM = "nfe_vs_success_rate_fp1-32-100"

FP_NFE = list(range(1, 33)) + [100]
DP_NFE = [100]
# X-axis ticks (readable on log scale)
X_TICKS = [1, 2, 4, 8, 16, 32, 100]

TASKS = [
    ("Microwave", "sr_microwave"),
    ("Slide Cabinet", "sr_slide cabinet"),
    ("Hinge Cabinet", "sr_hinge cabinet"),
    ("Kettle", "sr_kettle"),
    ("Top Burner", "sr_top burner"),
    ("Bottom Burner", "sr_bottom burner"),
    ("Light Switch", "sr_light switch"),
]

MODEL_META = {
    "flowpolicy": {
        "label": "FlowPolicy",
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "o",
        "nfes": FP_NFE,
        "markersize": 4.5,
    },
    "diffusion_policy_cnn": {
        "label": "DP-CNN",
        "color": "#1f77b4",
        "linestyle": "--",
        "marker": "s",
        "nfes": DP_NFE,
        "markersize": 7,
    },
    "diffusion_policy_transformer": {
        "label": "DP-Transformer",
        "color": "#ff7f0e",
        "linestyle": ":",
        "marker": "^",
        "nfes": DP_NFE,
        "markersize": 7,
    },
}
MODEL_ORDER = (
    "flowpolicy",
    "diffusion_policy_cnn",
    "diffusion_policy_transformer",
)


def load_summary(path: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            out[(r["model"], int(r["nfe"]))] = r
    return out


def series_for(
    table: dict[tuple[str, int], dict],
    model: str,
    col: str,
    nfes: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, los, his = [], [], []
    for nfe in nfes:
        row = table.get((model, nfe))
        if row is None:
            raise KeyError(f"Missing summary row for {model} NFE={nfe}")
        mu = float(row[col])
        sd = float(row.get(f"{col}_std") or 0.0)
        means.append(mu)
        los.append(max(0.0, mu - sd))
        his.append(min(1.0, mu + sd))
    return np.asarray(means), np.asarray(los), np.asarray(his)


def _style_axis(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(X_TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in X_TICKS]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.grid(True, color="#d0d0d0", linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    table = load_summary(SUMMARY)
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
        "Success Rate terhadap NFE pada Tujuh Sub-Tugas Franka Kitchen\n"
        "(FlowPolicy NFE 1–32 & 100; DP @ 1/8/32/100)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    for idx, (task_name, col) in enumerate(TASKS):
        ax = axes_flat[idx]
        for model in MODEL_ORDER:
            meta = MODEL_META[model]
            mean, lo, hi = series_for(table, model, col, meta["nfes"])
            x = np.asarray(meta["nfes"], dtype=float)
            ax.plot(
                x,
                mean,
                color=meta["color"],
                linestyle=meta["linestyle"],
                marker=meta["marker"],
                markersize=meta["markersize"],
                linewidth=1.8,
                label=meta["label"],
                markevery=1,
            )
            ax.fill_between(x, lo, hi, color=meta["color"], alpha=0.18, linewidth=0)

        _style_axis(ax)
        ax.set_title(task_name, fontweight="bold", fontsize=12)
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

    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / f"{OUT_STEM}.png"
    out_pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
