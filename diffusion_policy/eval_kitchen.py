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
import shutil
from typing import Any, Dict, List, Optional, Tuple

import click
import dill
import hydra
import numpy as np
import torch

from diffusion_policy.common.multistage_metrics import K_MAX, compute_multistage_metrics
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
    "LSTM_GMM": [
        "data/LSTM_GMM/train0/epoch=*.ckpt",
        "data/LSTM_GMM/train1/epoch=*.ckpt",
        "data/LSTM_GMM/train2/epoch=*.ckpt",
    ],
    "implicit_behavior_cloning": [
        "data/implicit_behavior_cloning/train0/epoch=*.ckpt",
        "data/implicit_behavior_cloning/train1/epoch=*.ckpt",
        "data/implicit_behavior_cloning/train2/epoch=*.ckpt",
    ],
    "behavior_transformer": [
        "data/behavior_transformer/train0/epoch=*.ckpt",
        "data/behavior_transformer/train1/epoch=*.ckpt",
        "data/behavior_transformer/train2/epoch=*.ckpt",
    ],
}
MODEL_CHOICES = tuple(DEFAULT_CHECKPOINTS.keys())
NATIVE_ONLY_MODELS = {"LSTM_GMM", "behavior_transformer"}
IBC_MODEL = "implicit_behavior_cloning"


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


def _metric_mean(value: Any) -> Optional[float]:
    """Unwrap mean from float or {mean, std, n_samples}."""
    if value is None:
        return None
    if isinstance(value, dict):
        m = value.get("mean")
        return float(m) if m is not None else None
    return float(value)


def _format_rate(stat: Any, with_interval: bool = True) -> str:
    """Format mean ± std, plus [ci_low, ci_high] when Wilson fields exist."""
    if stat is None:
        return "n/a"
    if not isinstance(stat, dict):
        return f"{float(stat):.3f}"
    mean = stat.get("mean")
    std = stat.get("std")
    if mean is None:
        return "n/a"
    if std is None:
        return f"{float(mean):.3f}"
    line = f"{float(mean):.3f} ± {float(std):.3f}"
    lo = stat.get("ci_low")
    hi = stat.get("ci_high")
    if with_interval and lo is not None and hi is not None:
        line += f"  [{float(lo):.3f}, {float(hi):.3f}]"
    return line


def _format_latency(stat: Any, label: str) -> Optional[str]:
    if not isinstance(stat, dict):
        return None
    mean = stat.get("mean")
    if mean is None:
        return None
    p95 = stat.get("p95")
    n = stat.get("n_samples")
    if p95 is not None and n:
        return f"  {label}: {float(mean):.1f}  (P95={float(p95):.1f}, n={int(n)})"
    return f"  {label}: {float(mean):.1f}"


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


def load_complete_metrics(
    metrics_path: pathlib.Path, expected_n_episodes: int
) -> Optional[Dict[str, Any]]:
    if not metrics_path.is_file():
        return None
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    n = metrics.get("n_episodes")
    if n is None:
        n = len(metrics.get("episodes") or [])
    try:
        if int(n) != int(expected_n_episodes):
            return None
    except (TypeError, ValueError):
        return None
    return metrics


def prepare_seed_output_dir(
    seed_output_dir: pathlib.Path,
    overwrite: bool,
    n_episodes: int,
) -> Optional[Dict[str, Any]]:
    """Return existing metrics to skip, else None (run). Incomplete dirs are removed."""
    metrics_path = seed_output_dir / "eval_metrics.json"
    if overwrite:
        if seed_output_dir.exists():
            shutil.rmtree(seed_output_dir)
        return None
    if not seed_output_dir.exists():
        return None
    existing = load_complete_metrics(metrics_path, n_episodes)
    if existing is not None:
        return existing
    click.echo(f"Removing incomplete {seed_output_dir}")
    shutil.rmtree(seed_output_dir)
    return None


def apply_inference_override(
    policy, model: str, num_inference_steps: Optional[int]
) -> Any:
    """Apply optional NFE override; return the effective step count written to metrics."""
    if num_inference_steps is None:
        if hasattr(policy, "pred_n_iter"):
            return getattr(policy, "pred_n_iter")
        return getattr(policy, "num_inference_steps", None)

    if model in NATIVE_ONLY_MODELS:
        raise click.ClickException(
            f"{model} has no NFE override; omit --num_inference_steps"
        )

    if model == IBC_MODEL or hasattr(policy, "pred_n_iter"):
        policy.pred_n_iter = num_inference_steps
        click.echo(f"Mapped --num_inference_steps -> pred_n_iter={num_inference_steps}")
        return num_inference_steps

    if hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = num_inference_steps
        return num_inference_steps

    raise click.ClickException(
        f"Policy for {model} has no num_inference_steps or pred_n_iter to override"
    )


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
    inference_warmup_calls: int = 10,
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
        abs_action=task_cfg.get("abs_action", False),
        tqdm_interval_sec=env_runner_cfg.get("tqdm_interval_sec", 5.0),
        dataset_dir=dataset_dir,
        save_trajectory_logs=save_trajectory_logs,
        inference_warmup_calls=inference_warmup_calls,
    )


def format_eval_report(metrics: Dict[str, Any], seed_name: str) -> str:
    if not metrics.get("multistage_metrics") and metrics.get("episodes"):
        ms = compute_multistage_metrics(
            metrics["episodes"], sub_goals=ALL_TASKS, num_sub_goals=K_MAX
        )
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
        "Per-task success rate (p4 ceiling; n may be < episodes because "
        "leftover tasks after k>=4 are excluded; "
        "± is Wilson 95% half-width when CI is present)",
        "-" * 72,
    ]
    sr = metrics.get("success_rate", {})
    for task in ALL_TASKS:
        stat = sr.get(task, {})
        if stat.get("mean") is not None:
            n = stat.get("n_samples")
            extra = f"  n={int(n)}" if n is not None else ""
            lines.append(f"  {task:<16}  {_format_rate(stat)}{extra}")

    cum = sr.get("p4_success") or sr.get("all_7_tasks") or {}
    if cum.get("mean") is not None:
        lines.extend([
            "",
            "Episode success (p4: >= 4 of 7 tasks completed)",
            "-" * 72,
            f"  success rate: {_format_rate(cum)}",
        ])

    ms7 = metrics.get("multistage_metrics", {}).get("all_7_tasks", {})
    px7 = ms7.get("px", {})
    if px7:
        lines.extend(["", "Multi-stage p_k (>= k of 7 tasks; scored through p4)", "-" * 72])
        parts = []
        for k in range(1, K_MAX + 1):
            pk = f"p{k}"
            if pk not in px7:
                continue
            mean = _metric_mean(px7[pk])
            if mean is None:
                continue
            parts.append(f"p{k}={_format_rate(px7[pk], with_interval=False)}")
        lines.append(f"  {'  '.join(parts)}")

    inf = metrics.get("timing_ms", {}).get("inference_latency", {})
    per_act = metrics.get("timing_ms", {}).get("latency_per_executed_action", {})
    total_inf = metrics.get("timing_ms", {}).get("total_inference_compute", {})
    f_ctrl = metrics.get("timing_ms", {}).get("f_control_hz", {})
    method = metrics.get("timing_methodology") or {}
    latency_lines = [
        _format_latency(inf, "Inference latency per call (ms)"),
        _format_latency(per_act, "Latency per executed action (ms)"),
        _format_latency(total_inf, "Total inference compute / episode (ms)"),
    ]
    if any(
        d.get("mean") is not None for d in (inf, per_act, total_inf, f_ctrl)
    ) or method:
        lines.extend(["", "Timing", "-" * 72])
        if method:
            lines.append(
                "  methodology: "
                f"clock={method.get('clock', 'n/a')}; "
                f"cuda_sync={method.get('cuda_synchronize_before')} / "
                f"{method.get('cuda_synchronize_after')}; "
                f"warmup_discarded={method.get('warmup_calls_discarded')}; "
                f"batch_size={method.get('batch_size')}; "
                f"timed={method.get('timed_region')}; "
                f"H2D={method.get('h2d_transfer_in_timed_region')}; "
                f"D2H={method.get('d2h_transfer_in_timed_region')}; "
                f"env_step={method.get('env_step_in_timed_region')}; "
                f"n_timed_calls={method.get('n_timed_calls')}"
            )
        for line in latency_lines:
            if line is not None:
                lines.append(line)
        if f_ctrl.get("mean") is not None:
            std = f_ctrl.get("std")
            if std is not None:
                lines.append(
                    f"  f_control (Hz): {f_ctrl['mean']:.3f} ± {std:.3f}"
                )
            else:
                lines.append(f"  f_control (Hz): {f_ctrl['mean']:.3f}")

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
            "latency_per_executed_action": {},
            "total_inference_compute": {},
            "f_control_hz": {},
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

    success_keys = ALL_TASKS + ["p4_success"]
    for key in success_keys:
        means = [
            m["success_rate"][key]["mean"]
            for m in checkpoint_metrics
            if isinstance(m.get("success_rate", {}).get(key), dict)
            and m["success_rate"][key].get("mean") is not None
        ]
        summary["success_rate"][key] = _compute_mean_std(means)

    for timing_key in [
        "inference_latency",
        "latency_per_executed_action",
        "total_inference_compute",
    ]:
        means = [
            m["timing_ms"][timing_key]["mean"]
            for m in checkpoint_metrics
            if m.get("timing_ms", {}).get(timing_key, {}).get("mean") is not None
        ]
        p95s = [
            m["timing_ms"][timing_key]["p95"]
            for m in checkpoint_metrics
            if m.get("timing_ms", {}).get(timing_key, {}).get("p95") is not None
        ]
        n_calls = [
            m["timing_ms"][timing_key].get("n_samples")
            for m in checkpoint_metrics
            if m.get("timing_ms", {}).get(timing_key, {}).get("n_samples")
        ]
        timed_n = int(np.sum(n_calls)) if n_calls else 0
        agg: Dict[str, Any] = {
            "mean": float(np.mean(means)) if means else None,
            "n_samples": timed_n if timed_n else len(means),
        }
        if p95s:
            agg["p95"] = float(np.mean(p95s))
        if timed_n:
            agg["n_timed_calls"] = timed_n
        summary["timing_ms"][timing_key] = agg

    for timing_key in [
        "f_control_hz",
        "episode_duration",
    ]:
        means = [
            m["timing_ms"][timing_key]["mean"]
            for m in checkpoint_metrics
            if m.get("timing_ms", {}).get(timing_key, {}).get("mean") is not None
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
                _metric_mean(m["multistage_metrics"][label]["px"][pk])
                for m in checkpoint_metrics
                if pk in m.get("multistage_metrics", {}).get(label, {}).get("px", {})
                and _metric_mean(m["multistage_metrics"][label]["px"][pk]) is not None
            ]
            agg_px[pk] = _compute_mean_std(vals)
        cum_vals = [
            _metric_mean(m["multistage_metrics"][label]["cumulative_order_success_rate"])
            for m in checkpoint_metrics
            if label in m.get("multistage_metrics", {})
            and _metric_mean(
                m["multistage_metrics"][label].get("cumulative_order_success_rate")
            )
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

    if checkpoint_metrics:
        summary["timing_methodology"] = checkpoint_metrics[0].get(
            "timing_methodology"
        )

    return summary


@click.command()
@click.option(
    "--model",
    "-m",
    type=click.Choice(MODEL_CHOICES),
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
@click.option(
    "--inference_warmup_calls",
    default=10,
    type=int,
    help="Dummy predict_action calls discarded before episode 0 (GPU warmup).",
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
    inference_warmup_calls: int,
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
    click.echo(f"inference_warmup_calls: {inference_warmup_calls}")
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

        existing = prepare_seed_output_dir(
            seed_output_dir, overwrite=overwrite, n_episodes=n_episodes
        )
        if existing is not None:
            click.echo(f"Skipping {dir_name}: eval_metrics.json complete")
            existing["model_seed"] = seed_name
            existing["_metrics_path"] = str(seed_output_dir / "eval_metrics.json")
            checkpoint_metrics.append(existing)
            continue

        click.echo(f"\n--- Evaluating {dir_name} ---")
        click.echo(f"Checkpoint: {ckpt_path}")

        if sampling_seed is not None:
            torch.manual_seed(sampling_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sampling_seed)
            np.random.seed(sampling_seed)

        policy, cfg = load_policy(ckpt_path, device_t)
        effective_nfe = apply_inference_override(policy, model, num_inference_steps)
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        runner = build_runner(
            cfg=cfg,
            output_dir=str(seed_output_dir),
            dataset_dir=dataset_dir,
            n_episodes=n_episodes,
            save_trajectory_logs=save_trajectory_logs,
            n_episodes_vis=0 if no_video else None,
            inference_warmup_calls=inference_warmup_calls,
        )
        metrics = runner.run(policy)
        metrics["model_seed"] = seed_name
        metrics["checkpoint"] = str(pathlib.Path(ckpt_path).resolve())
        metrics["num_inference_steps"] = effective_nfe
        if hasattr(policy, "pred_n_iter"):
            metrics["pred_n_iter"] = int(policy.pred_n_iter)
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
