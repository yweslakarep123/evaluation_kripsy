#!/usr/bin/env python3
"""Plot FlowPolicy vs Diffusion Policy (CNN + Transformer) on Kitchen eval.

Reads summary.json (+ completion counts from eval_metrics.json) and writes:
  1. Grouped per-task success rate
  2. Dual delta charts (FP - DP)
  3. Multi-stage p_k grouped bars
  4. Timing (latency / episode duration / task duration with n)

Usage:
  python scripts/plot_kitchen_fp_vs_dp.py
  python scripts/plot_kitchen_fp_vs_dp.py --out-dir data/kitchen_eval_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

MODEL_SPECS = {
    "FlowPolicy": {
        "summary": ROOT
        / "kripsy12/FlowPolicy/data/kitchen_eval/flowpolicy/summary.json",
        "seed_dirs": [
            ROOT
            / "kripsy12/FlowPolicy/data/kitchen_eval/flowpolicy"
            / f"seed_baseline_{s}"
            for s in (42, 43, 44)
        ],
        "color": "#2ca02c",
        "short": "FP",
    },
    "DP-CNN": {
        "summary": ROOT
        / "diffusion_policy/data/kitchen_eval/diffusion_policy_cnn/summary.json",
        "seed_dirs": [
            ROOT
            / "diffusion_policy/data/kitchen_eval/diffusion_policy_cnn"
            / f"seed_train{s}"
            for s in (0, 1, 2)
        ],
        "color": "#1f77b4",
        "short": "CNN",
    },
    "DP-Transformer": {
        "summary": ROOT
        / "diffusion_policy/data/kitchen_eval/diffusion_policy_transformer"
        / "summary.json",
        "seed_dirs": [
            ROOT
            / "diffusion_policy/data/kitchen_eval/diffusion_policy_transformer"
            / f"seed_train{s}"
            for s in (0, 1, 2)
        ],
        "color": "#ff7f0e",
        "short": "Trans",
    },
}

MODEL_ORDER = list(MODEL_SPECS.keys())
TASKS = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]
TASK_LABELS = {
    "bottom burner": "bottom\nburner",
    "top burner": "top\nburner",
    "light switch": "light\nswitch",
    "slide cabinet": "slide\ncabinet",
    "hinge cabinet": "hinge\ncabinet",
    "microwave": "microwave",
    "kettle": "kettle",
}
LOW_N_THRESHOLD = 30  # gray-out task duration bars below this total completions


def _save(fig, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path_base.with_suffix('.png')}")
    print(f"  wrote {path_base.with_suffix('.pdf')}")


def load_summaries() -> dict[str, dict]:
    out = {}
    for name, spec in MODEL_SPECS.items():
        path = spec["summary"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing summary for {name}: {path}")
        out[name] = json.loads(path.read_text())
    return out


def load_completion_counts() -> dict[str, dict[str, int]]:
    """Total completion_count across seeds per model/task."""
    counts: dict[str, dict[str, int]] = {m: {t: 0 for t in TASKS} for m in MODEL_ORDER}
    for name, spec in MODEL_SPECS.items():
        for seed_dir in spec["seed_dirs"]:
            metrics_path = seed_dir / "eval_metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(metrics_path)
            em = json.loads(metrics_path.read_text())
            stats = em.get("completion_order_stats", {})
            for task in TASKS:
                counts[name][task] += int(stats.get(task, {}).get("completion_count", 0))
    return counts


def episode_n(summaries: dict[str, dict]) -> int:
    """Total episodes used for success-rate denominators (same for all models)."""
    # summary n_samples is #seeds; per-seed n_episodes is in summary
    n_seeds = summaries["FlowPolicy"]["success_rate"]["bottom burner"]["n_samples"]
    n_ep = summaries["FlowPolicy"]["n_episodes_per_checkpoint"]
    return int(n_seeds * n_ep)


def print_text_summary(
    summaries: dict[str, dict], completions: dict[str, dict[str, int]], n_ep: int
) -> None:
    print("=" * 72)
    print(f"Kitchen eval comparison | {n_ep} episodes per model (3 seeds × 100)")
    print("=" * 72)

    print("\nPer-task success rate (mean % across seeds) and total completions:")
    header = f"{'task':<16}" + "".join(f"{m:>16}" for m in MODEL_ORDER)
    print(header)
    for task in TASKS:
        cells = []
        for m in MODEL_ORDER:
            mean = summaries[m]["success_rate"][task]["mean"] * 100
            n = completions[m][task]
            cells.append(f"{mean:5.1f}% n={n:<3}")
        print(f"{task:<16}" + "".join(f"{c:>16}" for c in cells))

    fp = summaries["FlowPolicy"]
    print("\nFlowPolicy excels vs DP-CNN / DP-Transformer (pp = percentage points):")
    for task in TASKS:
        fp_m = fp["success_rate"][task]["mean"]
        d_cnn = (fp_m - summaries["DP-CNN"]["success_rate"][task]["mean"]) * 100
        d_tr = (fp_m - summaries["DP-Transformer"]["success_rate"][task]["mean"]) * 100
        if d_cnn > 1 and d_tr > 1:
            print(f"  + {task}: FP−CNN {d_cnn:+.1f}pp, FP−Trans {d_tr:+.1f}pp")

    print("\nDiffusion Policy excels (FP behind both):")
    for task in TASKS:
        fp_m = fp["success_rate"][task]["mean"]
        d_cnn = (fp_m - summaries["DP-CNN"]["success_rate"][task]["mean"]) * 100
        d_tr = (fp_m - summaries["DP-Transformer"]["success_rate"][task]["mean"]) * 100
        if d_cnn < -1 and d_tr < -1:
            print(f"  - {task}: FP−CNN {d_cnn:+.1f}pp, FP−Trans {d_tr:+.1f}pp")

    print("\nTiming (mean ms):")
    for key, label in (
        ("inference_latency", "inference latency"),
        ("episode_duration", "episode duration"),
    ):
        vals = [summaries[m]["timing_ms"][key]["mean"] for m in MODEL_ORDER]
        print(
            f"  {label}: "
            + ", ".join(f"{m}={v:.1f}" for m, v in zip(MODEL_ORDER, vals))
        )
    print()


def plot_grouped_success(
    summaries: dict[str, dict], n_ep: int, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(TASKS))
    width = 0.25
    offsets = np.linspace(-(len(MODEL_ORDER) - 1) / 2, (len(MODEL_ORDER) - 1) / 2, len(MODEL_ORDER))

    for i, model in enumerate(MODEL_ORDER):
        means = [summaries[model]["success_rate"][t]["mean"] * 100 for t in TASKS]
        stds = [summaries[model]["success_rate"][t]["std"] * 100 for t in TASKS]
        ax.bar(
            x + offsets[i] * width,
            means,
            width,
            yerr=stds,
            label=model,
            color=MODEL_SPECS[model]["color"],
            alpha=0.9,
            capsize=3,
            error_kw={"elinewidth": 1},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        f"Per-task success rate (mean ± std across 3 seeds)\n"
        f"Denominator: n={n_ep} episodes per model (fair comparison)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "01_grouped_success_rate")


def plot_delta(summaries: dict[str, dict], n_ep: int, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    baselines = ["DP-CNN", "DP-Transformer"]

    for ax, baseline in zip(axes, baselines):
        deltas = []
        for task in TASKS:
            fp = summaries["FlowPolicy"]["success_rate"][task]["mean"]
            base = summaries[baseline]["success_rate"][task]["mean"]
            deltas.append((fp - base) * 100)

        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        y = np.arange(len(TASKS))
        ax.barh(y, deltas, color=colors, alpha=0.85, height=0.7)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(TASKS)
        ax.set_xlabel("Δ success rate (pp)")
        ax.set_title(f"FlowPolicy − {baseline}")
        ax.grid(True, axis="x", alpha=0.3)

        for yi, d in zip(y, deltas):
            ha = "left" if d >= 0 else "right"
            offset = 0.8 if d >= 0 else -0.8
            ax.text(d + offset, yi, f"{d:+.1f}", va="center", ha=ha, fontsize=8)

    fig.suptitle(
        f"Where each method excels (green = FlowPolicy ahead, red = DP ahead)\n"
        f"n={n_ep} episodes per model",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "02_delta_success_rate")


def plot_multistage(summaries: dict[str, dict], n_ep: int, out_dir: Path) -> None:
    ks = [f"p{k}" for k in range(1, 8)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(ks))
    width = 0.25
    offsets = np.linspace(-(len(MODEL_ORDER) - 1) / 2, (len(MODEL_ORDER) - 1) / 2, len(MODEL_ORDER))

    for i, model in enumerate(MODEL_ORDER):
        means = [
            summaries[model]["multistage_metrics"]["all_7_tasks"]["px"][k]["mean"] * 100
            for k in ks
        ]
        stds = [
            summaries[model]["multistage_metrics"]["all_7_tasks"]["px"][k]["std"] * 100
            for k in ks
        ]
        ax.bar(
            x + offsets[i] * width,
            means,
            width,
            yerr=stds,
            label=model,
            color=MODEL_SPECS[model]["color"],
            alpha=0.9,
            capsize=3,
            error_kw={"elinewidth": 1},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"≥{k} tasks" for k in range(1, 8)])
    ax.set_ylabel("Fraction of episodes (%)")
    ax.set_ylim(0, 110)
    ax.set_title(
        f"Multi-stage p_k: fraction of episodes completing ≥ k of 7 tasks\n"
        f"n={n_ep} episodes per model"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "03_multistage_pk")


def plot_timing(
    summaries: dict[str, dict],
    completions: dict[str, dict[str, int]],
    n_ep: int,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))

    # Panel 1: inference latency
    ax = axes[0]
    means = [summaries[m]["timing_ms"]["inference_latency"]["mean"] for m in MODEL_ORDER]
    stds = [summaries[m]["timing_ms"]["inference_latency"]["std"] for m in MODEL_ORDER]
    colors = [MODEL_SPECS[m]["color"] for m in MODEL_ORDER]
    bars = ax.bar(MODEL_ORDER, means, yerr=stds, color=colors, alpha=0.9, capsize=4)
    ax.set_ylabel("ms")
    ax.set_title("Inference latency")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.tick_params(axis="x", rotation=15)

    # Panel 2: episode duration
    ax = axes[1]
    means = [summaries[m]["timing_ms"]["episode_duration"]["mean"] for m in MODEL_ORDER]
    stds = [summaries[m]["timing_ms"]["episode_duration"]["std"] for m in MODEL_ORDER]
    bars = ax.bar(MODEL_ORDER, means, yerr=stds, color=colors, alpha=0.9, capsize=4)
    ax.set_ylabel("ms")
    ax.set_title("Episode duration")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v/1000:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.tick_params(axis="x", rotation=15)

    # Panel 3: task duration (only completed tasks) with n labels
    ax = axes[2]
    x = np.arange(len(TASKS))
    width = 0.25
    offsets = np.linspace(-(len(MODEL_ORDER) - 1) / 2, (len(MODEL_ORDER) - 1) / 2, len(MODEL_ORDER))

    for i, model in enumerate(MODEL_ORDER):
        means = []
        stds = []
        alphas = []
        for task in TASKS:
            td = summaries[model]["timing_ms"]["task_duration"][task]
            means.append(td["mean"])
            stds.append(td["std"])
            n = completions[model][task]
            alphas.append(0.35 if n < LOW_N_THRESHOLD else 0.9)

        # draw per-bar with individual alpha (low-n grayed)
        for j, (mean, std, alpha) in enumerate(zip(means, stds, alphas)):
            task_j = TASKS[j]
            ax.bar(
                x[j] + offsets[i] * width,
                mean,
                width,
                yerr=std,
                color=MODEL_SPECS[model]["color"],
                alpha=alpha,
                capsize=2,
                error_kw={"elinewidth": 0.8},
                label=model if j == 0 else None,
            )
            n = completions[model][task_j]
            ax.text(
                x[j] + offsets[i] * width,
                mean,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=8)
    ax.set_ylabel("ms")
    ax.set_title(
        f"Task duration (completed only)\n"
        f"faded if n<{LOW_N_THRESHOLD} completions"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Timing comparison | success-rate denom n={n_ep} episodes/model",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "04_timing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "kitchen_eval_plots",
        help="Directory for PNG/PDF figures",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    summaries = load_summaries()
    completions = load_completion_counts()
    n_ep = episode_n(summaries)

    # Sanity: all models same episode budget
    for m in MODEL_ORDER:
        n_seeds = summaries[m]["success_rate"]["bottom burner"]["n_samples"]
        n_per = summaries[m]["n_episodes_per_checkpoint"]
        assert n_seeds * n_per == n_ep, f"{m} episode budget mismatch"

    print_text_summary(summaries, completions, n_ep)

    print(f"Writing figures to {out_dir}")
    plot_grouped_success(summaries, n_ep, out_dir)
    plot_delta(summaries, n_ep, out_dir)
    plot_multistage(summaries, n_ep, out_dir)
    plot_timing(summaries, completions, n_ep, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
