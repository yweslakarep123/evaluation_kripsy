"""
Kitchen multi-seed evaluation with detailed metrics.

Usage:
  MUJOCO_GL=egl python eval_kitchen.py --output_root data/kitchen_eval_results --device cuda:0
  MUJOCO_GL=egl python eval_kitchen.py --checkpoints path/to/latest.ckpt --seeds 0 --n_episodes 2
"""

import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import glob
import inspect
import json
import os
import pathlib
from typing import Any, Dict, List, Optional

import click
import dill
import hydra
import numpy as np
import torch

from flow_policy_3d.common.multistage_metrics import compute_multistage_metrics
from flow_policy_3d.env_runner.kitchen_lowdim_eval_runner import (
    ALL_TASKS,
    KitchenLowdimEvalRunner,
)


def _compute_mean_std(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": None, "std": None, "n_samples": 0}
    if n == 1:
        return {"mean": float(arr[0]), "std": 0.0, "n_samples": 1}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "n_samples": n,
    }


def discover_checkpoints(checkpoints: Optional[tuple]) -> List[str]:
    if checkpoints:
        return list(checkpoints)
    patterns = [
        str(pathlib.Path("data") / "*" / "epoch=*.ckpt"),
        str(pathlib.Path("data") / "outputs" / "*" / "*" / "checkpoints" / "*.ckpt"),
        str(pathlib.Path("..") / "outputs" / "*" / "runs" / "*" / "checkpoints" / "latest.ckpt"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    return sorted(set(found))


def model_name_from_checkpoint(checkpoint_path: str) -> str:
    return pathlib.Path(checkpoint_path).parent.name


def load_policy(checkpoint_path: str, device: torch.device):
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    init_params = inspect.signature(cls.__init__).parameters
    if "output_dir" in init_params:
        workspace = cls(cfg, output_dir=None)
    else:
        workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    if (
        cfg.training.get("use_ema", False)
        and hasattr(workspace, "ema_model")
        and workspace.ema_model is not None
    ):
        policy = workspace.ema_model
    elif hasattr(workspace, "policy"):
        policy = workspace.policy
    else:
        policy = workspace.model

    policy.to(device)
    policy.eval()
    return policy, cfg


def build_runner(cfg, output_dir: str, eval_seed: int, n_episodes: int) -> KitchenLowdimEvalRunner:
    env_runner_cfg = cfg.task.get("eval_runner") or cfg.task.env_runner
    return KitchenLowdimEvalRunner(
        output_dir=output_dir,
        eval_seed=eval_seed,
        n_episodes=n_episodes,
        n_episodes_vis=n_episodes,
        max_steps=env_runner_cfg.get("max_steps", 280),
        n_obs_steps=env_runner_cfg.get("n_obs_steps", cfg.get("n_obs_steps", 2)),
        n_action_steps=env_runner_cfg.get(
            "n_action_steps", cfg.get("n_action_steps", 8)
        ),
        render_hw=tuple(env_runner_cfg.get("render_hw", [240, 360])),
        fps=env_runner_cfg.get("fps", 12.5),
        crf=env_runner_cfg.get("crf", 22),
        past_action=env_runner_cfg.get("past_action", cfg.get("past_action_visible", False)),
        abs_action=env_runner_cfg.get("abs_action", False),
        robot_noise_ratio=env_runner_cfg.get("robot_noise_ratio", 0.1),
        tqdm_interval_sec=env_runner_cfg.get("tqdm_interval_sec", 5.0),
    )


def format_eval_report(metrics: Dict[str, Any]) -> str:
    """Human-readable report: per-episode order + per-task success + p1..pN."""
    if not metrics.get("multistage_metrics") and metrics.get("episodes"):
        ms = compute_multistage_metrics(metrics["episodes"], sub_goals=ALL_TASKS)
        metrics.setdefault("multistage_metrics", {})["all_7_tasks"] = {
            "px": ms["px"],
            "cumulative_order_success_rate": ms["cumulative_order_success_rate"],
            "sub_goals": ms["sub_goals"],
        }

    lines = [
        "=" * 72,
        f"Kitchen Eval Report | seed={metrics.get('eval_seed')} | "
        f"{metrics.get('n_episodes')} episodes",
        "=" * 72,
        "",
        "Per-episode task completion order",
        "-" * 72,
        f"{'ep':>4}  {'seed':>6}  {'n/7':>4}  {'all7':>5}  completion_order",
        "-" * 72,
    ]
    for ep in metrics.get("episodes", []):
        order = ep.get("completion_order") or []
        order_str = " -> ".join(order) if order else "(none)"
        all7 = "yes" if ep.get("all_7_success") else "no"
        n_done = ep.get("num_tasks_completed", len(ep.get("completed_tasks", [])))
        lines.append(
            f"{ep.get('episode_idx', '?'):>4}  "
            f"{ep.get('env_seed', '?'):>6}  "
            f"{n_done:>4}  "
            f"{all7:>5}  "
            f"{order_str}"
        )

    lines.extend(["", "Per-task success rate (fraction of episodes)", "-" * 72])
    sr = metrics.get("success_rate", {})
    for task in ALL_TASKS:
        stat = sr.get(task, {})
        mean = stat.get("mean")
        std = stat.get("std")
        if mean is not None:
            lines.append(f"  {task:<16}  {mean:.3f} ± {std:.3f}")
        else:
            lines.append(f"  {task:<16}  n/a")

    cum = sr.get("all_7_tasks", {})
    if cum.get("mean") is not None:
        lines.extend([
            "",
            "Cumulative episode success (all 7 tasks completed in one episode)",
            "-" * 72,
            f"  success rate: {cum['mean']:.3f} ± {cum['std']:.3f}",
        ])

    ms7 = metrics.get("multistage_metrics", {}).get("all_7_tasks", {})
    px7 = ms7.get("px", {})
    if px7:
        lines.extend(["", "Multi-stage p_k (>= k of 7 tasks completed)", "-" * 72])
        px_line = "  ".join(
            f"p{k}={px7[f'p{k}']:.3f}"
            for k in range(1, len(ALL_TASKS) + 1)
            if f"p{k}" in px7
        )
        lines.append(f"  {px_line}")
        cum7 = ms7.get("cumulative_order_success_rate")
        if cum7 is not None:
            lines.append(
                f"  cumulative_order_success (all 7, any order): {cum7:.3f}"
            )

    lines.append("")
    return "\n".join(lines)


def write_eval_report(metrics: Dict[str, Any], output_dir: pathlib.Path) -> str:
    report = format_eval_report(metrics)
    report_path = output_dir / "eval_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    return str(report_path)


def aggregate_seed_metrics(seed_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "seeds": [m["eval_seed"] for m in seed_metrics],
        "tasks": ALL_TASKS,
        "success_rate": {},
        "timing_ms": {
            "inference_latency": {},
            "episode_duration": {},
            "task_duration": {"overall": {}},
        },
        "per_seed": {},
    }

    for metric in seed_metrics:
        seed = metric["eval_seed"]
        summary["per_seed"][str(seed)] = {
            "success_rate": metric["success_rate"],
            "timing_ms": metric["timing_ms"],
            "multistage_metrics": metric.get("multistage_metrics"),
            "eval_metrics_path": metric.get("_metrics_path"),
        }

    success_keys = ALL_TASKS + ["all_7_tasks"]
    for key in success_keys:
        means = [
            m["success_rate"][key]["mean"]
            for m in seed_metrics
            if m["success_rate"][key]["mean"] is not None
        ]
        summary["success_rate"][key] = _compute_mean_std(means)

    for timing_key in ["inference_latency", "episode_duration"]:
        means = [
            m["timing_ms"][timing_key]["mean"]
            for m in seed_metrics
            if m["timing_ms"][timing_key]["mean"] is not None
        ]
        summary["timing_ms"][timing_key] = _compute_mean_std(means)

    overall_means = [
        m["timing_ms"]["task_duration"]["overall"]["mean"]
        for m in seed_metrics
        if m["timing_ms"]["task_duration"]["overall"]["mean"] is not None
    ]
    summary["timing_ms"]["task_duration"]["overall"] = _compute_mean_std(overall_means)

    for task_name in ALL_TASKS:
        means = [
            m["timing_ms"]["task_duration"][task_name]["mean"]
            for m in seed_metrics
            if m["timing_ms"]["task_duration"][task_name]["mean"] is not None
        ]
        summary["timing_ms"]["task_duration"][task_name] = _compute_mean_std(means)

    # Aggregate multistage p_k across seeds (mean of per-seed p_k values)
    summary["multistage_metrics"] = {}
    for label in ["all_7_tasks", "paper_4_tasks"]:
        px_keys = set()
        for m in seed_metrics:
            ms = m.get("multistage_metrics", {}).get(label, {})
            px_keys.update(ms.get("px", {}).keys())
        agg_px = {}
        for pk in sorted(px_keys, key=lambda x: int(x[1:])):
            vals = [
                m["multistage_metrics"][label]["px"][pk]
                for m in seed_metrics
                if pk in m.get("multistage_metrics", {}).get(label, {}).get("px", {})
            ]
            agg_px[pk] = _compute_mean_std(vals)
        cum_vals = [
            m["multistage_metrics"][label]["cumulative_order_success_rate"]
            for m in seed_metrics
            if label in m.get("multistage_metrics", {})
            and m["multistage_metrics"][label].get("cumulative_order_success_rate")
            is not None
        ]
        sub_goals = (
            seed_metrics[0].get("multistage_metrics", {}).get(label, {}).get("sub_goals")
            if seed_metrics
            else []
        )
        summary["multistage_metrics"][label] = {
            "sub_goals": sub_goals,
            "px": agg_px,
            "cumulative_order_success_rate": _compute_mean_std(cum_vals),
        }

    return summary


@click.command()
@click.option(
    "--checkpoints",
    "-c",
    multiple=True,
    help="Checkpoint paths. Default: auto-scan data/*/epoch=*.ckpt",
)
@click.option("--output_root", "-o", default="data/kitchen_eval_results")
@click.option("--seeds", default="0,42,101", help="Comma-separated eval seeds")
@click.option("--n_episodes", default=100, type=int)
@click.option("--device", "-d", default="cuda:0")
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Overwrite existing seed output directories",
)
def main(checkpoints, output_root, seeds, n_episodes, device, overwrite):
    os.environ.setdefault("MUJOCO_GL", "egl")
    checkpoint_paths = discover_checkpoints(checkpoints if checkpoints else None)
    if len(checkpoint_paths) == 0:
        raise click.ClickException("No checkpoints found.")

    eval_seeds = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    device = torch.device(device)

    pathlib.Path(output_root).mkdir(parents=True, exist_ok=True)
    click.echo(f"Found {len(checkpoint_paths)} checkpoint(s)")
    click.echo(f"Seeds: {eval_seeds}, episodes per seed: {n_episodes}")

    for checkpoint_path in checkpoint_paths:
        model_name = model_name_from_checkpoint(checkpoint_path)
        click.echo(f"\n=== Evaluating {model_name} ===")
        click.echo(f"Checkpoint: {checkpoint_path}")

        policy, cfg = load_policy(checkpoint_path, device)
        model_output_root = pathlib.Path(output_root) / model_name
        model_output_root.mkdir(parents=True, exist_ok=True)

        seed_metrics: List[Dict[str, Any]] = []
        for eval_seed in eval_seeds:
            seed_output_dir = model_output_root / f"seed_{eval_seed}"
            if seed_output_dir.exists() and not overwrite:
                existing_metrics = seed_output_dir / "eval_metrics.json"
                if existing_metrics.is_file():
                    click.echo(
                        f"Skipping seed {eval_seed}: {existing_metrics} exists "
                        "(use --overwrite to rerun)"
                    )
                    with open(existing_metrics, "r") as f:
                        metrics = json.load(f)
                    metrics["_metrics_path"] = str(existing_metrics)
                    seed_metrics.append(metrics)
                    if not (seed_output_dir / "eval_report.txt").is_file():
                        write_eval_report(metrics, seed_output_dir)
                    continue
                click.confirm(
                    f"Output path {seed_output_dir} exists. Overwrite?", abort=True
                )
            seed_output_dir.mkdir(parents=True, exist_ok=True)

            runner = build_runner(
                cfg=cfg,
                output_dir=str(seed_output_dir),
                eval_seed=eval_seed,
                n_episodes=n_episodes,
            )
            metrics = runner.run(policy)
            metrics["checkpoint"] = str(pathlib.Path(checkpoint_path).resolve())
            metrics_path = seed_output_dir / "eval_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
            metrics["_metrics_path"] = str(metrics_path)
            seed_metrics.append(metrics)

            report_path = write_eval_report(metrics, seed_output_dir)
            click.echo(format_eval_report(metrics))
            click.echo(f"Wrote {metrics_path}")
            click.echo(f"Wrote {report_path}")

        summary = aggregate_seed_metrics(seed_metrics)
        summary["checkpoint"] = str(pathlib.Path(checkpoint_path).resolve())
        summary["model_name"] = model_name
        summary["n_episodes_per_seed"] = n_episodes
        summary_path = model_output_root / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        click.echo(f"Wrote summary: {summary_path}")

        summary_lines = [
            "=" * 72,
            f"Cross-seed summary | {model_name} | seeds={summary.get('seeds')}",
            "=" * 72,
            "",
            "Per-task success rate (mean ± std across seeds)",
        ]
        for task in ALL_TASKS + ["all_7_tasks"]:
            stat = summary.get("success_rate", {}).get(task, {})
            if stat.get("mean") is not None:
                label = task if task != "all_7_tasks" else "ALL 7 (cumulative episode)"
                summary_lines.append(f"  {label:<24}  {stat['mean']:.3f} ± {stat['std']:.3f}")
        ms7 = summary.get("multistage_metrics", {}).get("all_7_tasks", {})
        px7 = ms7.get("px", {})
        if px7:
            summary_lines.extend(["", "Multi-stage p_k (mean across seeds)"])
            for k in range(1, len(ALL_TASKS) + 1):
                pk = f"p{k}"
                if pk in px7 and px7[pk].get("mean") is not None:
                    summary_lines.append(f"  {pk}: {px7[pk]['mean']:.3f} ± {px7[pk]['std']:.3f}")
        summary_report = "\n".join(summary_lines) + "\n"
        summary_report_path = model_output_root / "summary_report.txt"
        with open(summary_report_path, "w") as f:
            f.write(summary_report)
        click.echo(summary_report)
        click.echo(f"Wrote {summary_report_path}")


if __name__ == "__main__":
    main()
