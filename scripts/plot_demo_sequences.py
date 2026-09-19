#!/usr/bin/env python3
"""Plot Kitchen demo sequences from demo_sequences.csv.

Exports:
  1) Combined overview (legacy)
  2) Separate panels A / B / C for Word
  3) Word-optimized sequences panel (narrower page width, 300 dpi, clearer labels)

Usage:
  python scripts/plot_demo_sequences.py
  python scripts/plot_demo_sequences.py \\
    --csv data/kitchen_eval_plots/demo_stats/demo_sequences.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data/kitchen_eval_plots/demo_stats/demo_sequences.csv"
DEFAULT_OUT_DIR = ROOT / "data/kitchen_eval_plots/demo_stats"

SOURCE_COLOR = {"friday": "#4C78A8", "postcorl": "#F58518"}
FIRST_COLOR = {
    "microwave": "#59A14F",
    "kettle": "#E15759",
    "top burner": "#B07AA1",
    "bottom burner": "#76B7B2",
    "light switch": "#EDC948",
    "slide cabinet": "#9C755F",
    "hinge cabinet": "#BAB0AC",
}
SHORT = {
    "microwave": "Microwave",
    "kettle": "Kettle",
    "bottom burner": "Bottom Burner",
    "top burner": "Top Burner",
    "light switch": "Light Switch",
    "slide cabinet": "Slide Cabinet",
    "hinge cabinet": "Hinge Cabinet",
}


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["n_demos"] = int(r["n_demos"])
    return rows


def _save(fig: plt.Figure, stem: Path, *, dpi: int = 200) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stem.with_suffix('.png')}")
    print(f"Saved: {stem.with_suffix('.pdf')}")


def plot_sequences_panel(
    rows: list[dict],
    stem: Path,
    *,
    figsize: tuple[float, float] = (7.2, 9.2),
    dpi: int = 300,
    title: str | None = None,
    word_friendly: bool = False,
) -> None:
    """Horizontal bar chart of all demo sequences (standalone)."""
    total = sum(r["n_demos"] for r in rows)
    rows_sorted = sorted(rows, key=lambda r: r["n_demos"])  # bottom→top

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(rows_sorted))
    counts = [r["n_demos"] for r in rows_sorted]
    colors = [SOURCE_COLOR[r["source"]] for r in rows_sorted]
    ax.barh(y, counts, color=colors, edgecolor="white", linewidth=0.5, height=0.78)

    # Cleaner labels for Word: sequence name only; counts annotated on bars
    if word_friendly:
        labels = [r["sequence"] for r in rows_sorted]
        label_fs = 10
        for yi, r in zip(y, rows_sorted):
            pct = 100.0 * r["n_demos"] / total
            ax.text(
                r["n_demos"] + max(counts) * 0.012,
                yi,
                f"{r['n_demos']}  ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=9,
                color="#333333",
            )
        ax.set_xlim(0, max(counts) * 1.28)
    else:
        labels = []
        for r in rows_sorted:
            pct = 100.0 * r["n_demos"] / total
            labels.append(f"{r['sequence']}   ({r['n_demos']}, {pct:.1f}%)")
        label_fs = 9
        ax.set_xlim(0, max(counts) * 1.12)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_fs)
    ax.set_xlabel("Number of demonstrations", fontsize=11)
    if title is None:
        title = (
            f"Demo sequences ranked by count\n"
            f"(25 folders, {total} MJL demos total)"
        )
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=SOURCE_COLOR["friday"], label="friday"),
            Patch(facecolor=SOURCE_COLOR["postcorl"], label="postcorl"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
        title="Source",
        title_fontsize=9,
    )
    fig.text(
        0.5,
        0.01,
        "Each demo is a fixed 4-of-7 task order · bar length = # .mjl files in that folder",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, stem, dpi=dpi)


def plot_first_task(rows: list[dict], stem: Path, *, dpi: int = 300) -> None:
    total = sum(r["n_demos"] for r in rows)
    first = Counter()
    for r in rows:
        first[r["task1"]] += r["n_demos"]
    order = sorted(first.keys(), key=lambda t: -first[t])
    vals = [first[t] for t in order]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(
        x,
        vals,
        color=[FIRST_COLOR.get(t, "#888888") for t in order],
        edgecolor="white",
        width=0.68,
    )
    for i, v in enumerate(vals):
        ax.text(
            i,
            v + total * 0.015,
            f"{v}\n({100 * v / total:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[t] for t in order], fontsize=10)
    ax.set_ylabel("Number of demos", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title(
        "Which task starts the demo?\n(first of the 4-task chain)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, stem, dpi=dpi)


def plot_last_task(rows: list[dict], stem: Path, *, dpi: int = 300) -> None:
    total = sum(r["n_demos"] for r in rows)
    last = Counter()
    for r in rows:
        last[r["task4"]] += r["n_demos"]
    order = sorted(last.keys(), key=lambda t: -last[t])
    vals = [last[t] for t in order]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(
        x,
        vals,
        color=[FIRST_COLOR.get(t, "#888888") for t in order],
        edgecolor="white",
        width=0.68,
    )
    for i, v in enumerate(vals):
        ax.text(
            i,
            v + total * 0.015,
            f"{v}\n({100 * v / total:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[t] for t in order], fontsize=10)
    ax.set_ylabel("Number of demos", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.32)
    ax.set_title(
        "Which task ends the demo?\n(4th / last in the chain)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, stem, dpi=dpi)


def plot_overview(rows: list[dict], out_base: Path) -> None:
    """Legacy combined A+B+C figure (kept for reference)."""
    total = sum(r["n_demos"] for r in rows)
    rows_sorted = sorted(rows, key=lambda r: r["n_demos"])

    fig = plt.figure(figsize=(14, 10), dpi=200)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[3.2, 1.2],
        width_ratios=[1.15, 1.0],
        hspace=0.32,
        wspace=0.28,
    )

    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(rows_sorted))
    counts = [r["n_demos"] for r in rows_sorted]
    colors = [SOURCE_COLOR[r["source"]] for r in rows_sorted]
    ax.barh(y, counts, color=colors, edgecolor="white", linewidth=0.4, height=0.82)
    labels = []
    for r in rows_sorted:
        pct = 100.0 * r["n_demos"] / total
        labels.append(f"{r['sequence']}   ({r['n_demos']}, {pct:.1f}%)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Number of demonstrations", fontsize=11)
    ax.set_title(
        f"A. Demo sequences ranked by count\n(25 folders, {total} MJL demos total)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax.set_xlim(0, max(counts) * 1.18)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=SOURCE_COLOR["friday"], label="friday"),
            Patch(facecolor=SOURCE_COLOR["postcorl"], label="postcorl"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
        title="Source",
        title_fontsize=9,
    )

    ax_b = fig.add_subplot(gs[0, 1])
    first = Counter()
    for r in rows:
        first[r["task1"]] += r["n_demos"]
    order_first = sorted(first.keys(), key=lambda t: -first[t])
    x = np.arange(len(order_first))
    vals = [first[t] for t in order_first]
    ax_b.bar(
        x,
        vals,
        color=[FIRST_COLOR.get(t, "#888888") for t in order_first],
        edgecolor="white",
        width=0.7,
    )
    for i, v in enumerate(vals):
        ax_b.text(
            i,
            v + total * 0.012,
            f"{v}\n({100 * v / total:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(
        [SHORT[t] for t in order_first], fontsize=9, rotation=15, ha="right"
    )
    ax_b.set_ylabel("Number of demos", fontsize=10)
    ax_b.set_ylim(0, max(vals) * 1.28)
    ax_b.set_title(
        "B. Which task starts the demo?\n(first of the 4-task chain)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax_b.yaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax_b.set_axisbelow(True)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    ax_c = fig.add_subplot(gs[1, 1])
    last = Counter()
    for r in rows:
        last[r["task4"]] += r["n_demos"]
    order_last = sorted(last.keys(), key=lambda t: -last[t])
    x = np.arange(len(order_last))
    vals = [last[t] for t in order_last]
    ax_c.bar(
        x,
        vals,
        color=[FIRST_COLOR.get(t, "#888888") for t in order_last],
        edgecolor="white",
        width=0.65,
    )
    for i, v in enumerate(vals):
        ax_c.text(
            i,
            v + total * 0.01,
            f"{v}\n({100 * v / total:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([SHORT[t] for t in order_last], fontsize=9)
    ax_c.set_ylabel("Number of demos", fontsize=10)
    ax_c.set_ylim(0, max(vals) * 1.32)
    ax_c.set_title(
        "C. Which task ends the demo?\n(4th / last in the chain)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax_c.yaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax_c.set_axisbelow(True)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    fig.suptitle(
        "Kitchen demonstration sequences (Relay Policy Learning MJL demos)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "Source: demo_sequences.csv · each demo is a fixed 4-of-7 task order · "
        "bar length = number of .mjl files in that folder",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )

    _save(fig, out_base, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir

    rows = load_rows(csv_path)
    total = sum(r["n_demos"] for r in rows)
    print(f"Loaded {len(rows)} sequences, {total} demos from {csv_path}")

    # Legacy combined overview
    plot_overview(rows, out_dir / "03_demo_sequences_overview")

    # Separate panels (Word-friendly sizes)
    plot_sequences_panel(
        rows,
        out_dir / "03a_demo_sequences_ranked",
        figsize=(7.2, 9.2),
        dpi=300,
        word_friendly=True,
        title=(
            f"Demo sequences ranked by count\n"
            f"(25 folders, {total} MJL demos total)"
        ),
    )
    plot_first_task(rows, out_dir / "03b_demo_first_task", dpi=300)
    plot_last_task(rows, out_dir / "03c_demo_last_task", dpi=300)


if __name__ == "__main__":
    main()
