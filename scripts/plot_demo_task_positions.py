#!/usr/bin/env python3
"""Plot per-subtask position in Kitchen demo sequences (positions 1–4).

Reads demo_sequences.csv. Each demo is a fixed 4-of-7 task chain; this figure
shows where each sub-task sits in that chain (weighted by n_demos).

Outputs:
  data/kitchen_eval_plots/demo_stats/demo_task_positions.csv
  data/kitchen_eval_plots/demo_stats/04_demo_task_positions.png/.pdf

Usage:
  python scripts/plot_demo_task_positions.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data/kitchen_eval_plots/demo_stats/demo_sequences.csv"
OUT_DIR = ROOT / "data/kitchen_eval_plots/demo_stats"
OUT_BASE = OUT_DIR / "04_demo_task_positions"

TASKS = [
    "microwave",
    "kettle",
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
]
SHORT = {
    "microwave": "Microwave",
    "kettle": "Kettle",
    "bottom burner": "Bottom Burner",
    "top burner": "Top Burner",
    "light switch": "Light Switch",
    "slide cabinet": "Slide Cabinet",
    "hinge cabinet": "Hinge Cabinet",
}
# Distinct, print-friendly palette for positions 1–4
POS_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
POS_LABELS = ["Position 1 (first)", "Position 2", "Position 3", "Position 4 (last)"]


def load_position_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts[7,4], means[7]) weighted by n_demos."""
    counts = np.zeros((len(TASKS), 4), dtype=np.float64)
    task_idx = {t: i for i, t in enumerate(TASKS)}
    with path.open() as f:
        for row in csv.DictReader(f):
            n = int(row["n_demos"])
            for pos, key in enumerate(["task1", "task2", "task3", "task4"]):
                counts[task_idx[row[key]], pos] += n
    totals = counts.sum(axis=1)
    means = np.zeros(len(TASKS))
    for i in range(len(TASKS)):
        if totals[i] > 0:
            means[i] = np.sum(counts[i] * np.arange(1, 5)) / totals[i]
    return counts, means


def save_csv(counts: np.ndarray, means: np.ndarray, path: Path) -> None:
    totals = counts.sum(axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "task",
                "short",
                "n_demos_containing",
                "n_pos1",
                "n_pos2",
                "n_pos3",
                "n_pos4",
                "pct_pos1",
                "pct_pos2",
                "pct_pos3",
                "pct_pos4",
                "mean_position",
            ]
        )
        for i, t in enumerate(TASKS):
            tot = totals[i]
            pct = (100.0 * counts[i] / tot) if tot else np.zeros(4)
            w.writerow(
                [
                    t,
                    SHORT[t],
                    int(tot),
                    int(counts[i, 0]),
                    int(counts[i, 1]),
                    int(counts[i, 2]),
                    int(counts[i, 3]),
                    round(float(pct[0]), 2),
                    round(float(pct[1]), 2),
                    round(float(pct[2]), 2),
                    round(float(pct[3]), 2),
                    round(float(means[i]), 3),
                ]
            )
    print(f"Saved: {path}")


def plot_positions(counts: np.ndarray, means: np.ndarray, out_base: Path) -> None:
    totals = counts.sum(axis=1)
    # Sort tasks by mean position (early → late) for readability
    order = np.argsort(means)
    tasks_ord = [TASKS[i] for i in order]
    counts_ord = counts[order]
    means_ord = means[order]
    totals_ord = totals[order]
    pct = np.zeros_like(counts_ord)
    for i in range(len(tasks_ord)):
        if totals_ord[i] > 0:
            pct[i] = 100.0 * counts_ord[i] / totals_ord[i]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.8),
        dpi=200,
        gridspec_kw={"width_ratios": [1.35, 0.85]},
    )

    # ── Left: stacked % by position ─────────────────────────────────────
    ax = axes[0]
    y = np.arange(len(tasks_ord))
    left = np.zeros(len(tasks_ord))
    for p in range(4):
        ax.barh(
            y,
            pct[:, p],
            left=left,
            height=0.72,
            color=POS_COLORS[p],
            edgecolor="white",
            linewidth=0.5,
            label=POS_LABELS[p],
        )
        for i in range(len(tasks_ord)):
            w = pct[i, p]
            if w >= 8:
                ax.text(
                    left[i] + w / 2,
                    y[i],
                    f"{w:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if w >= 25 else "#222222",
                    fontweight="medium",
                )
        left = left + pct[:, p]

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{SHORT[t]}  (n={int(totals_ord[i])})" for i, t in enumerate(tasks_ord)],
        fontsize=10,
    )
    ax.set_xlabel("% of demos containing this task", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_title(
        "A. Where does each sub-task appear in the 4-step demo?\n"
        "(stacked share of positions 1–4)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)

    # ── Right: mean position ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.barh(
        y,
        means_ord,
        height=0.72,
        color="#72B7B2",
        edgecolor="white",
        linewidth=0.5,
    )
    for i, m in enumerate(means_ord):
        ax2.text(m + 0.06, y[i], f"{m:.2f}", va="center", fontsize=9)
    ax2.set_yticks(y)
    ax2.set_yticklabels([SHORT[t] for t in tasks_ord], fontsize=10)
    ax2.set_xlabel("Mean position in demo chain", fontsize=11)
    ax2.set_xlim(0.8, 4.4)
    ax2.set_xticks([1, 2, 3, 4])
    ax2.axvline(1, color="#cccccc", linewidth=0.8, linestyle="--")
    ax2.axvline(4, color="#cccccc", linewidth=0.8, linestyle="--")
    ax2.set_title(
        "B. Mean position\n(1 = first, 4 = last)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.xaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax2.set_axisbelow(True)

    fig.suptitle(
        "Kitchen demos — sub-task position in the 4-of-7 demonstration chain",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Source: demo_sequences.csv · counts weighted by number of .mjl demos per sequence folder",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_base.with_suffix('.png')}")
    print(f"Saved: {out_base.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=OUT_BASE)
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    out_base = args.out if args.out.is_absolute() else ROOT / args.out

    counts, means = load_position_matrix(csv_path)
    save_csv(counts, means, OUT_DIR / "demo_task_positions.csv")
    plot_positions(counts, means, out_base)


if __name__ == "__main__":
    main()
