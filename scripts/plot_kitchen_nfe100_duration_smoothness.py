#!/usr/bin/env python3
"""Task duration & action smoothness plots from kitchen_eval_nfe100.

Outputs (under data/kitchen_eval_plots/nfe100/duration_smoothness/):
  01_task_duration.{png,pdf}       — 2x2 panels (NFE) of mean task duration
  02_action_smoothness.{png,pdf}   — 2x2 panels of within-chunk vs boundary L2
  duration_smoothness.csv

Usage:
  python scripts/plot_kitchen_nfe100_duration_smoothness.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/kitchen_eval_plots/nfe100/duration_smoothness"
FP_ROOT = ROOT / "kripsy12/FlowPolicy/data/kitchen_eval_nfe100/flowpolicy"
DP_CANDIDATES = [
    ROOT / "diffusion_policy/data/kitchen_eval_nfe100_diffusion",
    ROOT / "diffusion_policy/data/kitchen_eval_nfe100",
]

MODEL_ORDER = ["FlowPolicy", "DP-CNN", "DP-Transformer"]
COLORS = {
    "FlowPolicy": "#2ca02c",
    "DP-CNN": "#1f77b4",
    "DP-Transformer": "#ff7f0e",
}
LEGEND = {
    "FlowPolicy": "FlowPolicy (Ta=4)",
    "DP-CNN": "DP-CNN (Ta=8)",
    "DP-Transformer": "DP-Transformer (Ta=8)",
}
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
    return [dp_root / mid / f"seed_train{s}_nfe{nfe}_sseed0" for s in (0, 1, 2)]


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_task_duration(model_key: str, nfe: int, dp_root: Path) -> Dict[str, float]:
    """Mean±std of per-seed timing_ms.task_duration.overall.mean."""
    means: List[float] = []
    for d in seed_dirs(model_key, nfe, dp_root):
        p = d / "eval_metrics.json"
        if not p.is_file():
            raise FileNotFoundError(p)
        m = json.loads(p.read_text())
        means.append(float(m["timing_ms"]["task_duration"]["overall"]["mean"]))
    mu, sd = _mean_std(means)
    return {"mean": mu, "std": sd, "n_seeds": float(len(means))}


def episode_smoothness(npz_path: Path) -> Tuple[float, float]:
    """Return (within_chunk_mean_l2, boundary_mean_l2) for one episode."""
    z = np.load(npz_path)
    acts = np.asarray(z["executed_action"], dtype=np.float64)
    ctrl = np.asarray(z["control_step_per_env_step"], dtype=np.int32)
    if acts.ndim != 2 or len(acts) < 2:
        return float("nan"), float("nan")
    diffs = np.linalg.norm(acts[1:] - acts[:-1], axis=-1)
    boundary = ctrl[1:] != ctrl[:-1]
    within = diffs[~boundary]
    bound = diffs[boundary]
    w = float(np.mean(within)) if len(within) else float("nan")
    b = float(np.mean(bound)) if len(bound) else float("nan")
    return w, b


def load_smoothness(model_key: str, nfe: int, dp_root: Path) -> Dict[str, Any]:
    """Aggregate within/boundary L2: episode → seed mean → mean±std across seeds."""
    seed_within: List[float] = []
    seed_boundary: List[float] = []
    n_episodes_total = 0
    for d in seed_dirs(model_key, nfe, dp_root):
        traj = d / "trajectory_logs"
        if not traj.is_dir():
            raise FileNotFoundError(traj)
        ep_w: List[float] = []
        ep_b: List[float] = []
        for npz_path in sorted(traj.glob("ep_*.npz")):
            w, b = episode_smoothness(npz_path)
            if np.isfinite(w):
                ep_w.append(w)
            if np.isfinite(b):
                ep_b.append(b)
        n_episodes_total += len(ep_w)
        if ep_w:
            seed_within.append(float(np.mean(ep_w)))
        if ep_b:
            seed_boundary.append(float(np.mean(ep_b)))
    wu, ws = _mean_std(seed_within)
    bu, bs = _mean_std(seed_boundary)
    return {
        "within_mean": wu,
        "within_std": ws,
        "boundary_mean": bu,
        "boundary_std": bs,
        "n_seeds": len(seed_within),
        "n_episodes": n_episodes_total,
    }


def collect_all(dp_root: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for model in MODEL_ORDER:
        for nfe in NFES:
            dur = load_task_duration(model, nfe, dp_root)
            sm = load_smoothness(model, nfe, dp_root)
            rows[(model, nfe)] = {**dur, **sm}
            print(
                f"  {model:16} NFE={nfe:3}: "
                f"task_dur={dur['mean']:.1f}±{dur['std']:.1f} ms | "
                f"within={sm['within_mean']:.4f}±{sm['within_std']:.4f} "
                f"boundary={sm['boundary_mean']:.4f}±{sm['boundary_std']:.4f}"
            )
    return rows


def plot_task_duration(rows: Dict[Tuple[str, int], Dict[str, Any]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    x = np.arange(len(MODEL_ORDER))
    for ax, nfe in zip(axes.ravel(), NFES):
        means = [rows[(m, nfe)]["mean"] for m in MODEL_ORDER]
        stds = [rows[(m, nfe)]["std"] for m in MODEL_ORDER]
        colors = [COLORS[m] for m in MODEL_ORDER]
        ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            capsize=4,
            edgecolor="black",
            lw=0.4,
            width=0.7,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
        ax.set_title(f"NFE = {nfe}")
        ax.set_ylabel("Mean task duration (ms)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        _despine(ax)
    fig.suptitle("Task Duration by Model and NFE", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "01_task_duration")


def plot_action_smoothness(
    rows: Dict[Tuple[str, int], Dict[str, Any]], out_dir: Path
) -> None:
    categories = ["Within-chunk steps\n(mid trajectory)", "Boundary steps\n(chunk transition)"]
    x = np.arange(len(categories))
    width = 0.25
    # Per-panel y-scale: DP collapses at low NFE (L2 ~4–5) vs ~0.07 when healthy.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=False)
    for ax, nfe in zip(axes.ravel(), NFES):
        for i, model in enumerate(MODEL_ORDER):
            r = rows[(model, nfe)]
            vals = [r["within_mean"], r["boundary_mean"]]
            errs = [r["within_std"], r["boundary_std"]]
            offset = (i - 1) * width
            ax.bar(
                x + offset,
                vals,
                width,
                yerr=errs,
                color=COLORS[model],
                label=LEGEND[model],
                capsize=3,
                edgecolor="black",
                lw=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title(f"NFE = {nfe}")
        ax.set_ylabel("Mean L2 norm between consecutive actions")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        _despine(ax)
        if nfe == NFES[0]:
            ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Action Smoothness: Within-chunk vs Boundary Transitions", y=1.01
    )
    fig.tight_layout()
    _save(fig, out_dir / "02_action_smoothness")


def write_csv(rows: Dict[Tuple[str, int], Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "duration_smoothness.csv"
    fields = [
        "model",
        "nfe",
        "task_duration_mean_ms",
        "task_duration_std_ms",
        "within_chunk_l2_mean",
        "within_chunk_l2_std",
        "boundary_l2_mean",
        "boundary_l2_std",
        "n_seeds",
        "n_episodes_smoothness",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for nfe in NFES:
            for model in MODEL_ORDER:
                r = rows[(model, nfe)]
                w.writerow(
                    {
                        "model": model,
                        "nfe": nfe,
                        "task_duration_mean_ms": r["mean"],
                        "task_duration_std_ms": r["std"],
                        "within_chunk_l2_mean": r["within_mean"],
                        "within_chunk_l2_std": r["within_std"],
                        "boundary_l2_mean": r["boundary_mean"],
                        "boundary_l2_std": r["boundary_std"],
                        "n_seeds": int(r["n_seeds"]),
                        "n_episodes_smoothness": int(r["n_episodes"]),
                    }
                )
    print(f"Wrote {path}")


def main() -> None:
    dp_root = find_dp_root()
    print(f"DP root: {dp_root}")
    print(f"FP root: {FP_ROOT}")
    print("Collecting metrics...")
    rows = collect_all(dp_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Plotting task duration...")
    plot_task_duration(rows, OUT_DIR)
    print("Plotting action smoothness...")
    plot_action_smoothness(rows, OUT_DIR)
    write_csv(rows, OUT_DIR)
    print(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
