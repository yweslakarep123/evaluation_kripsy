"""
Kitchen flat evaluation (100 episodes per checkpoint).

Usage:
  MUJOCO_GL=egl python eval_kitchen.py --model diffusion_policy_transformer --device cuda:0
  MUJOCO_GL=egl python eval_kitchen.py --smoke --device cuda:0 -m diffusion_policy_transformer
"""

import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import glob
import inspect
import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import click
import dill
import hydra
import numpy as np
import torch

from diffusion_policy.common.multistage_metrics import compute_multistage_metrics
from diffusion_policy.env_runner.kitchen_lowdim_eval_runner import (
    ALL_TASKS,
    KitchenLowdimEvalRunner,
)

DEFAULT_CHECKPOINTS = {
    "diffusion_policy_transformer": [
        "data/diffusion_policy_transformer/train0/epoch=*.ckpt",
        "data/diffusion_policy_transformer/train1/epoch=*.ckpt",
        "data/diffusion_policy_transformer/train2/epoch=*.ckpt",
    ],
    "diffusion_policy_cnn": [
        "data/diffusion_policy_cnn/train0/epoch=*.ckpt",
        "data/diffusion_policy_cnn/train1/epoch=*.ckpt",
        "data/diffusion_policy_cnn/train2/epoch=*.ckpt",
    ],
}


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


def resolve_checkpoint(pattern: str) -> str:
    if pathlib.Path(pattern).is_file():
        return pattern
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    parent = pathlib.Path(pattern).parent
    if parent.exists():
        fallback = sorted(parent.glob("*.ckpt"))
        if fallback:
            return str(fallback[-1])
    raise click.ClickException(f"No checkpoint found for pattern: {pattern}")


def seed_name_from_checkpoint(checkpoint_path: str) -> str:
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


def build_runner(
    cfg,
    output_dir: str,
    dataset_dir: str,
    n_episodes: int,
    save_trajectory_logs: bool = True,
    n_episodes_vis: Optional[int] = None,
) -> KitchenLowdimEvalRunner:
    task_cfg = cfg.get("task", cfg)
    env_runner_cfg = task_cfg.get("env_runner", {})
    if n_episodes_vis is None:
        n_episodes_vis = n_episodes
    return KitchenLowdimEvalRunner(
        output_dir=output_dir,
        n_episodes=n_episodes,
        n_episodes_vis=n_episodes_vis,
        max_steps=env_runner_cfg.get("max_steps", 280),
        n_obs_steps=cfg.get("n_obs_steps", env_runner_cfg.get("n_obs_steps", 4)),
        n_action_steps=cfg.get(
            "n_action_steps", env_runner_cfg.get("n_action_steps", 8)
        ),
        render_hw=tuple(env_runner_cfg.get("render_hw", [240, 360])),
        fps=env_runner_cfg.get("fps", 12.5),
        crf=env_runner_cfg.get("crf", 22),
        past_action=cfg.get("past_action_visible", False),
        abs_action=task_cfg.get("abs_action", True),
        tqdm_interval_sec=env_runner_cfg.get("tqdm_interval_sec", 5.0),
        dataset_dir=dataset_dir,
        save_trajectory_logs=save_trajectory_logs,
    )


def format_eval_report(metrics: Dict[str, Any], seed_name: str) -> str:
    if not metrics.get("multistage_metrics") and metrics.get("episodes"):
        ms = compute_multistage_metrics(metrics["episodes"], sub_goals=ALL_TASKS)
        metrics.setdefault("multistage_metrics", {})["all_7_tasks"] = {
            "px": ms["px"],
            "cumulative_order_success_rate": ms["cumulative_order_success_rate"],
            "sub_goals": ms["sub_goals"],
        }

    lines = [
        "=" * 72,
        f"Kitchen Eval Report | seed={seed_name} | {metrics.get('n_episodes')} episodes",
        "=" * 72,
        "",
        "Per-task success rate (fraction of episodes)",
        "-" * 72,
    ]
    sr = metrics.get("success_rate", {})
    for task in ALL_TASKS:
        stat = sr.get(task, {})
        mean = stat.get("mean")
        std = stat.get("std")
        if mean is not None:
            lines.append(f"  {task:<16}  {mean:.3f} ± {std:.3f}")

    cum = sr.get("all_7_tasks", {})
    if cum.get("mean") is not None:
        lines.extend([
            "",
            "Cumulative episode success (all 7 tasks completed)",
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

    inf = metrics.get("timing_ms", {}).get("inference_latency", {})
    if inf.get("mean") is not None:
        lines.extend([
            "",
            "Timing",
            "-" * 72,
            f"  Inference latency (ms): {inf['mean']:.2f} ± {inf['std']:.2f}",
        ])

    lines.append("")
    return "\n".join(lines)


def write_eval_report(metrics: Dict[str, Any], output_dir: pathlib.Path, seed_name: str) -> str:
    report = format_eval_report(metrics, seed_name)
    report_path = output_dir / "eval_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    return str(report_path)


def aggregate_checkpoint_metrics(
    checkpoint_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "seeds": [m["model_seed"] for m in checkpoint_metrics],
        "tasks": ALL_TASKS,
        "success_rate": {},
        "timing_ms": {
            "inference_latency": {},
            "episode_duration": {},
            "task_duration": {"overall": {}},
        },
        "per_seed": {},
    }

    for metric in checkpoint_metrics:
        seed = metric["model_seed"]
        summary["per_seed"][seed] = {
            "success_rate": metric["success_rate"],
            "timing_ms": metric["timing_ms"],
            "multistage_metrics": metric.get("multistage_metrics"),
            "eval_metrics_path": metric.get("_metrics_path"),
        }

    success_keys = ALL_TASKS + ["all_7_tasks"]
    for key in success_keys:
        means = [
            m["success_rate"][key]["mean"]
            for m in checkpoint_metrics
            if m["success_rate"][key]["mean"] is not None
        ]
        summary["success_rate"][key] = _compute_mean_std(means)

    for timing_key in ["inference_latency", "episode_duration"]:
        means = [
            m["timing_ms"][timing_key]["mean"]
            for m in checkpoint_metrics
            if m["timing_ms"][timing_key]["mean"] is not None
        ]
        summary["timing_ms"][timing_key] = _compute_mean_std(means)

    overall_means = [
        m["timing_ms"]["task_duration"]["overall"]["mean"]
        for m in checkpoint_metrics
        if m["timing_ms"]["task_duration"]["overall"]["mean"] is not None
    ]
    summary["timing_ms"]["task_duration"]["overall"] = _compute_mean_std(overall_means)

    for task_name in ALL_TASKS:
        means = [
            m["timing_ms"]["task_duration"][task_name]["mean"]
            for m in checkpoint_metrics
            if m["timing_ms"]["task_duration"][task_name]["mean"] is not None
        ]
        summary["timing_ms"]["task_duration"][task_name] = _compute_mean_std(means)

    summary["multistage_metrics"] = {}
    for label in ["all_7_tasks", "paper_4_tasks"]:
        px_keys = set()
        for m in checkpoint_metrics:
            ms = m.get("multistage_metrics", {}).get(label, {})
            px_keys.update(ms.get("px", {}).keys())
        agg_px = {}
        for pk in sorted(px_keys, key=lambda x: int(x[1:])):
            vals = [
                m["multistage_metrics"][label]["px"][pk]
                for m in checkpoint_metrics
                if pk in m.get("multistage_metrics", {}).get(label, {}).get("px", {})
            ]
            agg_px[pk] = _compute_mean_std(vals)
        cum_vals = [
            m["multistage_metrics"][label]["cumulative_order_success_rate"]
            for m in checkpoint_metrics
            if label in m.get("multistage_metrics", {})
            and m["multistage_metrics"][label].get("cumulative_order_success_rate")
            is not None
        ]
        sub_goals = (
            checkpoint_metrics[0]
            .get("multistage_metrics", {})
            .get(label, {})
            .get("sub_goals", [])
            if checkpoint_metrics
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
    "--model",
    "-m",
    type=click.Choice(["diffusion_policy_transformer", "diffusion_policy_cnn"]),
    default="diffusion_policy_transformer",
)
@click.option(
    "--checkpoints",
    "-c",
    multiple=True,
    help="Checkpoint paths or globs",
)
@click.option("--output_root", "-o", default="data/kitchen_eval")
@click.option("--dataset_dir", default="data/kitchen")
@click.option("--n_episodes", default=100, type=int)
@click.option("--device", "-d", default="cuda:0")
@click.option("--smoke/--no-smoke", default=False, help="10 episodes, 1 checkpoint")
@click.option(
    "--save-trajectory-logs/--no-save-trajectory-logs",
    default=True,
    help="Save per-episode NPZ+TXT trajectory logs vs demo GT",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Overwrite existing checkpoint output directories",
)
@click.option(
    "--num_inference_steps",
    default=None,
    type=int,
    help="Override policy.num_inference_steps (NFE). Default: checkpoint config.",
)
@click.option(
    "--sampling_seed",
    default=None,
    type=int,
    help="Fix torch/cuda RNG for policy sampling. Default: unset.",
)
@click.option(
    "--no-video/--video",
    default=False,
    help="Skip MP4 rendering (n_episodes_vis=0). Faster for sweeps.",
)
def main(
    model: str,
    checkpoints: Tuple[str, ...],
    output_root: str,
    dataset_dir: str,
    n_episodes: int,
    device: str,
    smoke: bool,
    save_trajectory_logs: bool,
    overwrite: bool,
    num_inference_steps: Optional[int],
    sampling_seed: Optional[int],
    no_video: bool,
):
    os.environ.setdefault("MUJOCO_GL", "egl")

    if smoke:
        n_episodes = 10
        ckpt_patterns = [DEFAULT_CHECKPOINTS[model][0]]
    elif checkpoints:
        ckpt_patterns = list(checkpoints)
    else:
        ckpt_patterns = DEFAULT_CHECKPOINTS[model]

    ckpt_paths = [resolve_checkpoint(p) for p in ckpt_patterns]
    device_t = torch.device(device)
    output_model_dir = pathlib.Path(output_root) / model
    output_model_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Model: {model}")
    click.echo(f"Checkpoints: {ckpt_paths}")
    click.echo(f"Output: {output_model_dir}")
    click.echo(f"Episodes per checkpoint: {n_episodes}")
    if num_inference_steps is not None:
        if num_inference_steps < 1:
            raise click.ClickException("--num_inference_steps must be >= 1")
        click.echo(f"num_inference_steps override: {num_inference_steps}")
    if sampling_seed is not None:
        click.echo(f"sampling_seed: {sampling_seed}")
    if smoke:
        click.echo("SMOKE TEST mode")
    if no_video:
        click.echo("Video rendering disabled")

    checkpoint_metrics: List[Dict[str, Any]] = []

    for ckpt_path in ckpt_paths:
        seed_name = seed_name_from_checkpoint(ckpt_path)
        dir_name = f"seed_{seed_name}"
        if num_inference_steps is not None:
            dir_name += f"_nfe{num_inference_steps}"
        if sampling_seed is not None:
            dir_name += f"_sseed{sampling_seed}"
        seed_output_dir = output_model_dir / dir_name

        if seed_output_dir.exists() and not overwrite:
            existing_metrics = seed_output_dir / "eval_metrics.json"
            if existing_metrics.is_file():
                click.echo(f"Skipping {dir_name}: {existing_metrics} exists")
                with open(existing_metrics, "r") as f:
                    metrics = json.load(f)
                metrics["model_seed"] = seed_name
                metrics["_metrics_path"] = str(existing_metrics)
                checkpoint_metrics.append(metrics)
                continue

        click.echo(f"\n--- Evaluating {dir_name} ---")
        click.echo(f"Checkpoint: {ckpt_path}")

        if sampling_seed is not None:
            torch.manual_seed(sampling_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sampling_seed)
            np.random.seed(sampling_seed)

        policy, cfg = load_policy(ckpt_path, device_t)
        effective_nfe = getattr(policy, "num_inference_steps", None)
        if num_inference_steps is not None:
            if not hasattr(policy, "num_inference_steps"):
                raise click.ClickException(
                    "Policy has no num_inference_steps attribute to override"
                )
            policy.num_inference_steps = num_inference_steps
            effective_nfe = num_inference_steps
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        runner = build_runner(
            cfg=cfg,
            output_dir=str(seed_output_dir),
            dataset_dir=dataset_dir,
            n_episodes=n_episodes,
            save_trajectory_logs=save_trajectory_logs,
            n_episodes_vis=0 if no_video else None,
        )
        metrics = runner.run(policy)
        metrics["model_seed"] = seed_name
        metrics["checkpoint"] = str(pathlib.Path(ckpt_path).resolve())
        metrics["num_inference_steps"] = effective_nfe
        metrics["sampling_seed"] = sampling_seed

        metrics_path = seed_output_dir / "eval_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        metrics["_metrics_path"] = str(metrics_path)
        checkpoint_metrics.append(metrics)

        report_path = write_eval_report(metrics, seed_output_dir, seed_name)
        click.echo(format_eval_report(metrics, seed_name))
        click.echo(f"Wrote {metrics_path}")
        click.echo(f"Wrote {report_path}")

    if len(checkpoint_metrics) > 1:
        summary = aggregate_checkpoint_metrics(checkpoint_metrics)
        summary["model"] = model
        summary["n_episodes_per_checkpoint"] = n_episodes
        summary_path = output_model_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        click.echo(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
