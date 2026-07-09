#!/usr/bin/env python3
"""
Analyze Kitchen eval trajectory logs (ep_XXXX.npz).

Examples:
  # Single episode
  python scripts/analyze_kitchen_trajectory.py \\
    --npz diffusion_policy/data/kitchen_eval/diffusion_policy_transformer/seed_train0/trajectory_logs/ep_0000.npz \\
    --output_dir /tmp/kitchen_analysis/ep_0000

  # All episodes in one seed
  python scripts/analyze_kitchen_trajectory.py \\
    --seed_dir diffusion_policy/data/kitchen_eval/diffusion_policy_transformer/seed_train0 \\
    --output_dir /tmp/kitchen_analysis/seed_train0

  # Aggregate summary only (no per-episode joint plots)
  python scripts/analyze_kitchen_trajectory.py \\
    --seed_dir kripsy12/FlowPolicy/data/kitchen_eval/flowpolicy/seed_baseline_42 \\
    --output_dir /tmp/fp_analysis --summary_only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _decode_names(arr) -> List[str]:
    if arr is None:
        return []
    return [str(x) for x in np.asarray(arr).tolist()]


def load_trajectory(npz_path: pathlib.Path) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=False)
    out: Dict[str, Any] = {k: data[k] for k in data.files}
    out["_path"] = str(npz_path)
    return out


def print_episode_summary(d: Dict[str, Any]) -> None:
    joint_names = _decode_names(d.get("robot_joint_names"))
    action_names = _decode_names(d.get("action_names"))
    print("=" * 72)
    print(f"File: {d.get('_path', '?')}")
    print(f"  episode_idx={int(d['episode_idx'])}  init_idx={int(d['init_idx'])}")
    print(f"  demo_valid_len={int(d['demo_valid_len'])}  n_env_steps={int(d['n_env_steps'])}")
    print(f"  n_control_steps={int(d['n_control_steps'])}")
    print(
        "  window/horizon: "
        f"n_obs_steps={int(d['n_obs_steps'])}  "
        f"n_action_steps={int(d['n_action_steps'])}  "
        f"horizon={int(d['horizon'])}  "
        f"slice=[{int(d['action_slice_start'])}:{int(d['action_slice_end'])}]  "
        f"has_obs_pred={int(d['has_obs_pred'])}"
    )
    if joint_names:
        print(f"  joints: {', '.join(joint_names)}")
    if action_names:
        print(f"  actions: {', '.join(action_names)}")
    if "action_error_l2" in d:
        valid = ~np.isnan(d["action_error_l2"])
        if valid.any():
            print(
                f"  action_error_l2: mean={d['action_error_l2'][valid].mean():.4f}  "
                f"max={d['action_error_l2'][valid].max():.4f}"
            )
    if "obs_error_l2" in d:
        valid = ~np.isnan(d["obs_error_l2"])
        if valid.any():
            print(
                f"  obs_error_l2:    mean={d['obs_error_l2'][valid].mean():.4f}  "
                f"max={d['obs_error_l2'][valid].max():.4f}"
            )
    print("=" * 72)


def _safe_demo_slice(demo_arr: np.ndarray, n: int) -> np.ndarray:
    if len(demo_arr) == 0:
        return np.full((n,) + demo_arr.shape[1:], np.nan, dtype=np.float32)
    out = np.full((n,) + demo_arr.shape[1:], np.nan, dtype=np.float32)
    m = min(n, len(demo_arr))
    out[:m] = demo_arr[:m]
    return out


def plot_joint_rollout_vs_demo(
    d: Dict[str, Any], out_path: pathlib.Path, max_joints: int = 9
) -> None:
    executed_qp = d.get("executed_qp")
    demo_qp = d.get("demo_qp")
    if executed_qp is None:
        return
    demo_qp = _safe_demo_slice(demo_qp, len(executed_qp))
    joint_names = _decode_names(d.get("robot_joint_names"))
    n_joints = min(max_joints, executed_qp.shape[1])
    n_cols = 3
    n_rows = int(np.ceil(n_joints / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)
    steps = np.arange(len(executed_qp))
    for j in range(n_joints):
        ax = axes[j // n_cols][j % n_cols]
        label = joint_names[j] if j < len(joint_names) else f"q{j}"
        ax.plot(steps, executed_qp[:, j], label="rollout", linewidth=1.5)
        ax.plot(steps, demo_qp[:, j], label="demo GT", linewidth=1.0, alpha=0.8, linestyle="--")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    for j in range(n_joints, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle(f"Joint positions — ep {int(d['episode_idx'])} (init {int(d['init_idx'])})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_errors(d: Dict[str, Any], out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    steps = np.arange(int(d["n_env_steps"]))
    if "action_error_l2" in d:
        axes[0].plot(steps, d["action_error_l2"], color="C0", linewidth=1.2)
        axes[0].set_ylabel("L2 action error vs demo")
        axes[0].grid(True, alpha=0.3)
    if "obs_error_l2" in d:
        axes[1].plot(steps, d["obs_error_l2"], color="C1", linewidth=1.2)
        axes[1].set_ylabel("L2 obs error vs demo")
        axes[1].set_xlabel("env step")
        axes[1].grid(True, alpha=0.3)
    if "control_step_per_env_step" in d:
        ctrl = d["control_step_per_env_step"]
        for k in np.unique(ctrl[1:]):
            idx = np.where(ctrl == k)[0]
            if len(idx) > 0:
                axes[1].axvline(idx[0], color="gray", alpha=0.15, linewidth=0.8)
    fig.suptitle(f"Demo GT errors — ep {int(d['episode_idx'])}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_action_executed_vs_demo(d: Dict[str, Any], out_path: pathlib.Path) -> None:
    executed = d.get("executed_action")
    demo = d.get("demo_action")
    if executed is None:
        return
    demo_at_step = d.get("demo_action_at_step")
    if demo_at_step is not None:
        demo_plot = demo_at_step
    else:
        demo_plot = _safe_demo_slice(demo, len(executed))
    action_names = _decode_names(d.get("action_names"))
    n_act = executed.shape[1]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), squeeze=False)
    steps = np.arange(len(executed))
    for a in range(min(n_act, 9)):
        ax = axes[a // 3][a % 3]
        name = action_names[a] if a < len(action_names) else f"a{a}"
        ax.plot(steps, executed[:, a], label="executed")
        ax.plot(steps, demo_plot[:, a], label="demo", linestyle="--", alpha=0.8)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        if a == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Actions executed vs demo — ep {int(d['episode_idx'])}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_action_pred_horizon(d: Dict[str, Any], out_path: pathlib.Path, control_step: int = 0) -> None:
    action_pred = d.get("action_pred")
    action_executed = d.get("action_executed")
    if action_pred is None:
        return
    ctrl = min(control_step, len(action_pred) - 1)
    start = int(d["action_slice_start"])
    end = int(d["action_slice_end"])
    action_names = _decode_names(d.get("action_names"))
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(action_pred[ctrl].T, aspect="auto", origin="lower", cmap="RdBu_r")
    ax.axhline(start - 0.5, color="lime", linewidth=1.5, label="slice start")
    ax.axhline(end - 0.5, color="red", linewidth=1.5, label="slice end")
    ax.set_xlabel("horizon step h")
    ax.set_ylabel("action dim")
    yticks = np.arange(action_pred.shape[2])
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [action_names[i] if i < len(action_names) else f"a{i}" for i in yticks],
        fontsize=8,
    )
    fig.colorbar(im, ax=ax, fraction=0.02)
    if action_executed is not None:
        ax.text(
            0.02,
            0.98,
            f"executed L2 vs slice: "
            f"{d.get('action_pred_executed_l2', np.array([np.nan]))[ctrl].mean():.4f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"action_pred heatmap — ep {int(d['episode_idx'])} control_step={ctrl}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_velocity(d: Dict[str, Any], out_path: pathlib.Path) -> None:
    qv = d.get("qv")
    if qv is None:
        return
    joint_names = _decode_names(d.get("robot_joint_names"))
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), squeeze=False)
    steps = np.arange(len(qv))
    for j in range(min(9, qv.shape[1])):
        ax = axes[j // 3][j % 3]
        name = joint_names[j] if j < len(joint_names) else f"q{j}"
        ax.plot(steps, qv[:, j], linewidth=1.2)
        ax.set_title(f"{name} velocity")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Joint velocities (rollout) — ep {int(d['episode_idx'])}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_episode(
    npz_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    summary_only: bool = False,
    control_step: int = 0,
) -> Dict[str, Any]:
    d = load_trajectory(npz_path)
    print_episode_summary(d)

    stats: Dict[str, Any] = {
        "npz": str(npz_path),
        "episode_idx": int(d["episode_idx"]),
        "init_idx": int(d["init_idx"]),
        "n_env_steps": int(d["n_env_steps"]),
    }
    for key in ["action_error_l2", "obs_error_l2"]:
        if key in d:
            valid = ~np.isnan(d[key])
            if valid.any():
                stats[f"{key}_mean"] = float(d[key][valid].mean())
                stats[f"{key}_max"] = float(d[key][valid].max())

    if summary_only:
        return stats

    ep_dir = output_dir / f"ep_{int(d['episode_idx']):04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    plot_joint_rollout_vs_demo(d, ep_dir / "joints_vs_demo.png")
    plot_errors(d, ep_dir / "errors_vs_demo.png")
    plot_action_executed_vs_demo(d, ep_dir / "actions_vs_demo.png")
    plot_action_pred_horizon(d, ep_dir / "action_pred_heatmap.png", control_step=control_step)
    plot_velocity(d, ep_dir / "joint_velocities.png")

    print(f"  Saved plots -> {ep_dir}")
    return stats


def plot_seed_aggregate(all_stats: List[Dict[str, Any]], out_path: pathlib.Path) -> None:
    if not all_stats:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    act_means = [s.get("action_error_l2_mean", np.nan) for s in all_stats]
    obs_means = [s.get("obs_error_l2_mean", np.nan) for s in all_stats]
    axes[0].hist(act_means, bins=20, color="C0", edgecolor="white")
    axes[0].set_xlabel("mean action L2 vs demo (per episode)")
    axes[0].set_ylabel("count")
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(obs_means, bins=20, color="C1", edgecolor="white")
    axes[1].set_xlabel("mean obs L2 vs demo (per episode)")
    axes[1].set_ylabel("count")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Aggregate demo GT error — {len(all_stats)} episodes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def find_npz_files(seed_dir: pathlib.Path) -> List[pathlib.Path]:
    traj_dir = seed_dir / "trajectory_logs"
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"No trajectory_logs/ in {seed_dir}")
    return sorted(traj_dir.glob("ep_*.npz"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Kitchen trajectory NPZ logs")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", type=pathlib.Path, help="Path to ep_XXXX.npz")
    src.add_argument(
        "--seed_dir",
        type=pathlib.Path,
        help="Seed output dir containing trajectory_logs/ and eval_metrics.json",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        required=True,
        help="Directory to write plots and summary.json",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Print stats / aggregate histogram only, skip per-episode plots",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit number of episodes when using --seed_dir",
    )
    parser.add_argument(
        "--control_step",
        type=int,
        default=0,
        help="Control step index for action_pred heatmap",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_stats: List[Dict[str, Any]] = []

    if args.npz is not None:
        if not args.npz.is_file():
            print(f"Not found: {args.npz}", file=sys.stderr)
            return 1
        stats = analyze_episode(
            args.npz,
            args.output_dir,
            summary_only=args.summary_only,
            control_step=args.control_step,
        )
        all_stats.append(stats)
    else:
        npz_files = find_npz_files(args.seed_dir)
        if args.max_episodes is not None:
            npz_files = npz_files[: args.max_episodes]
        if not npz_files:
            print(f"No ep_*.npz in {args.seed_dir / 'trajectory_logs'}", file=sys.stderr)
            return 1
        print(f"Analyzing {len(npz_files)} episodes from {args.seed_dir}")
        for npz_path in npz_files:
            stats = analyze_episode(
                npz_path,
                args.output_dir,
                summary_only=args.summary_only,
                control_step=args.control_step,
            )
            all_stats.append(stats)

    summary_path = args.output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Wrote {summary_path}")

    if len(all_stats) > 1:
        plot_seed_aggregate(all_stats, args.output_dir / "aggregate_error_histogram.png")
        print(f"Wrote {args.output_dir / 'aggregate_error_histogram.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
