#!/usr/bin/env python3
"""General + WHY plots from kitchen_eval_nfe100 full eval.

Data roots:
  FP:  kripsy12/FlowPolicy/data/kitchen_eval_nfe100/flowpolicy/
  DP:  diffusion_policy/data/kitchen_eval_nfe100_diffusion/   (fallback: kitchen_eval_nfe100/)

Outputs:
  data/kitchen_eval_plots/nfe100/general/   — per-task SR, p_k, timing, deltas
  data/kitchen_eval_plots/nfe100/why/       — completion-order WHY (FP@8 vs DP@100)
  data/kitchen_eval_plots/nfe100/why_nfe8/  — WHY at equal NFE=8
  data/kitchen_eval_plots/nfe100/why_nfe100/— WHY at equal NFE=100

Usage:
  python scripts/plot_kitchen_nfe100_why_and_general.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_kitchen_completion_order as why  # noqa: E402

OUT_BASE = ROOT / "data/kitchen_eval_plots/nfe100"
FP_ROOT = ROOT / "kripsy12/FlowPolicy/data/kitchen_eval_nfe100/flowpolicy"
DP_CANDIDATES = [
    ROOT / "diffusion_policy/data/kitchen_eval_nfe100_diffusion",
    ROOT / "diffusion_policy/data/kitchen_eval_nfe100",
]

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
MODEL_ORDER = ["FlowPolicy", "DP-CNN", "DP-Transformer"]
COLORS = {
    "FlowPolicy": "#2ca02c",
    "DP-CNN": "#1f77b4",
    "DP-Transformer": "#ff7f0e",
}
SHORT = {"FlowPolicy": "FP", "DP-CNN": "CNN", "DP-Transformer": "Trans"}
NFES = (1, 8, 32, 100)


def find_dp_root() -> Path:
    for p in DP_CANDIDATES:
        if p.is_dir() and any(p.rglob("eval_metrics.json")):
            return p
    raise FileNotFoundError(f"No DP nfe100 root in {DP_CANDIDATES}")


def seed_dirs(model_key: str, nfe: int, dp_root: Path) -> List[Path]:
    if model_key == "FlowPolicy":
        return [
            FP_ROOT / f"seed_baseline_{s}_nfe{nfe}_sseed0" for s in (42, 43, 44)
        ]
    mid = (
        "diffusion_policy_cnn"
        if model_key == "DP-CNN"
        else "diffusion_policy_transformer"
    )
    return [
        dp_root / mid / f"seed_train{s}_nfe{nfe}_sseed0" for s in (0, 1, 2)
    ]


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def load_summary(model_key: str, nfe: int, dp_root: Path) -> Dict[str, Any]:
    """Aggregate 3 seeds like summary.json."""
    dirs = seed_dirs(model_key, nfe, dp_root)
    metrics_list = []
    for d in dirs:
        p = d / "eval_metrics.json"
        if not p.is_file():
            raise FileNotFoundError(p)
        metrics_list.append(json.loads(p.read_text()))

    out: Dict[str, Any] = {
        "success_rate": {},
        "timing_ms": {"inference_latency": {}, "episode_duration": {}, "task_duration": {}},
        "multistage_metrics": {"all_7_tasks": {"px": {}}},
        "n_episodes_per_checkpoint": metrics_list[0].get("n_episodes", 100),
        "n_seeds": len(metrics_list),
    }
    for t in TASKS + ["all_7_tasks"]:
        means = [m["success_rate"][t]["mean"] for m in metrics_list]
        mu, sd = _mean_std(means)
        out["success_rate"][t] = {"mean": mu, "std": sd, "n_samples": len(means)}

    for key in ("inference_latency", "episode_duration"):
        means = [m["timing_ms"][key]["mean"] for m in metrics_list]
        mu, sd = _mean_std(means)
        out["timing_ms"][key] = {"mean": mu, "std": sd, "n_samples": len(means)}

    overall = [
        m["timing_ms"]["task_duration"]["overall"]["mean"] for m in metrics_list
    ]
    mu, sd = _mean_std(overall)
    out["timing_ms"]["task_duration"]["overall"] = {
        "mean": mu,
        "std": sd,
        "n_samples": len(overall),
    }
    for t in TASKS:
        means = []
        for m in metrics_list:
            v = m["timing_ms"]["task_duration"].get(t, {}).get("mean")
            if v is not None:
                means.append(v)
        mu, sd = _mean_std(means)
        out["timing_ms"]["task_duration"][t] = {
            "mean": mu,
            "std": sd,
            "n_samples": len(means),
        }

    for k in range(1, 8):
        pk = f"p{k}"
        means = [
            m["multistage_metrics"]["all_7_tasks"]["px"][pk] for m in metrics_list
        ]
        mu, sd = _mean_std(means)
        out["multistage_metrics"]["all_7_tasks"]["px"][pk] = {
            "mean": mu,
            "std": sd,
            "n_samples": len(means),
        }

    # completion counts across all episodes
    completions = {t: 0 for t in TASKS}
    n_ep_total = 0
    for m in metrics_list:
        for ep in m["episodes"]:
            n_ep_total += 1
            for t in TASKS:
                completions[t] += int(ep["task_success"].get(t, 0))
    out["_completions"] = completions
    out["_n_ep_total"] = n_ep_total
    return out


def _save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {base.with_suffix('.png')}")


def plot_general_panel(
    summaries: Dict[str, Dict[str, Any]],
    title_suffix: str,
    out_dir: Path,
    prefix: str,
) -> None:
    # 1 grouped success
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(TASKS))
    w = 0.25
    for i, name in enumerate(MODEL_ORDER):
        s = summaries[name]
        means = [s["success_rate"][t]["mean"] * 100 for t in TASKS]
        stds = [s["success_rate"][t]["std"] * 100 for t in TASKS]
        ax.bar(
            x + (i - 1) * w,
            means,
            w,
            yerr=stds,
            label=name,
            color=COLORS[name],
            capsize=3,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title(f"Per-task success {title_suffix}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / f"{prefix}01_grouped_success_rate")

    # 2 delta FP - DP
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fp = summaries["FlowPolicy"]
    for ax, dp_name in zip(axes, ["DP-CNN", "DP-Transformer"]):
        dp = summaries[dp_name]
        deltas = [
            (fp["success_rate"][t]["mean"] - dp["success_rate"][t]["mean"]) * 100
            for t in TASKS
        ]
        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
        ax.bar(np.arange(len(TASKS)), deltas, color=colors, edgecolor="black", lw=0.4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(TASKS)))
        ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=8)
        ax.set_title(f"FP − {SHORT[dp_name]} (pp) {title_suffix}")
        ax.set_ylabel("Δ success (percentage points)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / f"{prefix}02_delta_success_rate")

    # 3 multistage pk
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pks = [f"p{k}" for k in range(1, 8)]
    x = np.arange(len(pks))
    w = 0.25
    for i, name in enumerate(MODEL_ORDER):
        s = summaries[name]
        means = [
            s["multistage_metrics"]["all_7_tasks"]["px"][pk]["mean"] * 100 for pk in pks
        ]
        stds = [
            s["multistage_metrics"]["all_7_tasks"]["px"][pk]["std"] * 100 for pk in pks
        ]
        ax.bar(
            x + (i - 1) * w,
            means,
            w,
            yerr=stds,
            label=name,
            color=COLORS[name],
            capsize=3,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pks)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title(f"Multi-stage p_k {title_suffix}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / f"{prefix}03_multistage_pk")

    # 4 timing
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    # latency
    ax = axes[0]
    for i, name in enumerate(MODEL_ORDER):
        s = summaries[name]["timing_ms"]["inference_latency"]
        ax.bar(
            i,
            s["mean"],
            yerr=s["std"],
            color=COLORS[name],
            capsize=4,
            edgecolor="black",
            lw=0.4,
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER])
    ax.set_ylabel("ms")
    ax.set_title("Inference latency")
    ax.grid(True, axis="y", alpha=0.3)

    # episode duration
    ax = axes[1]
    for i, name in enumerate(MODEL_ORDER):
        s = summaries[name]["timing_ms"]["episode_duration"]
        ax.bar(
            i,
            s["mean"] / 1000.0,
            yerr=s["std"] / 1000.0,
            color=COLORS[name],
            capsize=4,
            edgecolor="black",
            lw=0.4,
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER])
    ax.set_ylabel("s")
    ax.set_title("Episode duration")
    ax.grid(True, axis="y", alpha=0.3)

    # overall task duration
    ax = axes[2]
    for i, name in enumerate(MODEL_ORDER):
        s = summaries[name]["timing_ms"]["task_duration"]["overall"]
        ax.bar(
            i,
            s["mean"],
            yerr=s["std"],
            color=COLORS[name],
            capsize=4,
            edgecolor="black",
            lw=0.4,
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels([SHORT[m] for m in MODEL_ORDER])
    ax.set_ylabel("ms")
    ax.set_title("Mean task duration")
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Timing {title_suffix}", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / f"{prefix}04_timing")


def write_general_report(
    panels: Dict[str, Dict[str, Dict[str, Any]]], out_dir: Path
) -> None:
    lines = [
        "=" * 78,
        "NFE100 general comparison report",
        "=" * 78,
        "",
    ]
    for label, summaries in panels.items():
        lines.append(f"--- {label} ---")
        for name in MODEL_ORDER:
            s = summaries[name]
            p3 = s["multistage_metrics"]["all_7_tasks"]["px"]["p3"]
            p4 = s["multistage_metrics"]["all_7_tasks"]["px"]["p4"]
            lat = s["timing_ms"]["inference_latency"]
            lines.append(
                f"  {name}: p3={p3['mean']:.3f}±{p3['std']:.3f}  "
                f"p4={p4['mean']:.3f}±{p4['std']:.3f}  "
                f"lat={lat['mean']:.1f}±{lat['std']:.1f} ms"
            )
            sr = ", ".join(
                f"{t.split()[0]}={s['success_rate'][t]['mean']*100:.0f}%"
                for t in TASKS
            )
            lines.append(f"    SR: {sr}")
        lines.append("")
    path = out_dir / "general_report.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def run_why(
    nfe_map: Dict[str, int],
    dp_root: Path,
    out_dir: Path,
    label: str,
) -> None:
    """nfe_map: model display name -> nfe."""
    # Monkey-patch MODEL_SPECS seed_dirs for why module
    specs = {}
    for name in MODEL_ORDER:
        nfe = nfe_map[name]
        specs[name] = {
            "seed_dirs": seed_dirs(name, nfe, dp_root),
            "color": COLORS[name],
        }
    why.MODEL_SPECS = specs
    why.MODEL_ORDER = list(MODEL_ORDER)

    print(f"\nWHY [{label}] -> {out_dir}")
    for name in MODEL_ORDER:
        for d in specs[name]["seed_dirs"]:
            if not (d / "eval_metrics.json").is_file():
                raise FileNotFoundError(d / "eval_metrics.json")
        print(f"  {name} @NFE={nfe_map[name]}")

    stats = {name: why.load_model_stats(name) for name in MODEL_ORDER}
    out_dir.mkdir(parents=True, exist_ok=True)
    why.plot_position_hist(stats, out_dir)
    why.plot_transition_heatmaps(stats, out_dir)
    why.plot_path_conditional(stats, out_dir)
    why.plot_stop_multistage(stats, out_dir)
    why.plot_speed(stats, out_dir)
    text = why.write_report(stats, out_dir)
    # prepend label
    report_path = out_dir / "why_report.txt"
    report_path.write_text(
        f"WHY analysis from NFE100 full eval — {label}\n"
        + f"NFE map: {nfe_map}\n\n"
        + text
    )
    print(f"  wrote {report_path}")


def main() -> None:
    dp_root = find_dp_root()
    print(f"DP root: {dp_root}")
    print(f"FP root: {FP_ROOT}")

    general_dir = OUT_BASE / "general"
    general_dir.mkdir(parents=True, exist_ok=True)

    panels: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Operating-point panel: FP@8 vs DP@100 (best practical configs)
    op = {
        name: load_summary(
            name, 8 if name == "FlowPolicy" else 100, dp_root
        )
        for name in MODEL_ORDER
    }
    panels["operating FP@8 vs DP@100"] = op
    print("\nGeneral: operating point FP@8 vs DP@100")
    plot_general_panel(op, "(FP@8 vs DP@100)", general_dir, "op_")

    # Equal-NFE panels
    for nfe in NFES:
        print(f"\nGeneral: equal NFE={nfe}")
        summ = {name: load_summary(name, nfe, dp_root) for name in MODEL_ORDER}
        panels[f"equal NFE={nfe}"] = summ
        plot_general_panel(summ, f"(all @NFE={nfe})", general_dir, f"nfe{nfe}_")

    write_general_report(panels, general_dir)

    # WHY analyses
    run_why(
        {"FlowPolicy": 8, "DP-CNN": 100, "DP-Transformer": 100},
        dp_root,
        OUT_BASE / "why",
        "operating point FP@8 vs DP@100",
    )
    run_why(
        {"FlowPolicy": 8, "DP-CNN": 8, "DP-Transformer": 8},
        dp_root,
        OUT_BASE / "why_nfe8",
        "equal NFE=8",
    )
    run_why(
        {"FlowPolicy": 100, "DP-CNN": 100, "DP-Transformer": 100},
        dp_root,
        OUT_BASE / "why_nfe100",
        "equal NFE=100",
    )

    # Also refresh NFE curves with correct DP path
    print("\nRefreshing NFE curve plots via analyze_kitchen_nfe100.py ...")
    import analyze_kitchen_nfe100 as nfe100

    runs = nfe100.discover(dp_root, FP_ROOT)
    agg = nfe100.aggregate(runs)
    nfe100.write_csv(OUT_BASE / "summary.csv", agg)
    nfe100.plot_all(agg, OUT_BASE)
    report = nfe100.write_report(runs, agg, OUT_BASE)
    print(f"  wrote {report} (runs={len(runs)})")
    print("\nDone. Outputs under", OUT_BASE)


if __name__ == "__main__":
    main()
