"""
Kitchen combinatorial 4-subtask evaluation for FlowPolicy.

Prerequisite: Kitchen env requires compatible dm_control/mujoco (see kripsy12/ReinFlow/docs/KnownIssues.md).
If you see "top-level default class 'main' cannot be renamed", pin mujoco/dm_control or use robodiff-compatible versions.

Usage:
  MUJOCO_GL=egl python eval_kitchen_combinations.py --device cuda:0
  MUJOCO_GL=egl python eval_kitchen_combinations.py --smoke --device cuda:0
"""

import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import glob
import os
import pathlib

_root = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
os.chdir(str(_root))
from typing import Any, Dict, List, Optional, Tuple

import click
import torch

from flow_policy_3d.common.kitchen_combo_protocol import (
    EPISODES_PER_COMBINATION,
    aggregate_model_metrics,
    format_model_report,
    save_json,
)
from flow_policy_3d.env_runner.kitchen_combo_eval_runner import KitchenComboEvalRunner
from train import TrainFlowPolicyWorkspace

DEFAULT_CHECKPOINTS = [
    "data/outputs/baseline_42/latest-001.ckpt",
    "data/outputs/baseline_43/latest-001.ckpt",
    "data/outputs/baseline_44/epoch=*.ckpt",
]


def resolve_checkpoint(pattern: str) -> str:
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
    workspace = TrainFlowPolicyWorkspace.create_from_checkpoint(checkpoint_path)
    cfg = workspace.cfg
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
    model_seed: str,
    n_episodes_per_combo: int,
    combination_ids: Optional[List[int]],
    resume: bool,
) -> KitchenComboEvalRunner:
    task_cfg = cfg.get("task", cfg)
    env_runner_cfg = task_cfg.get("eval_runner") or task_cfg.get("env_runner", {})
    return KitchenComboEvalRunner(
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        model_seed=model_seed,
        max_steps=env_runner_cfg.get("max_steps", 280),
        n_obs_steps=env_runner_cfg.get("n_obs_steps", cfg.get("n_obs_steps", 2)),
        n_action_steps=env_runner_cfg.get(
            "n_action_steps", cfg.get("n_action_steps", 4)
        ),
        render_hw=tuple(env_runner_cfg.get("render_hw", [240, 360])),
        fps=env_runner_cfg.get("fps", 12.5),
        crf=env_runner_cfg.get("crf", 22),
        past_action=env_runner_cfg.get(
            "past_action", cfg.get("past_action_visible", False)
        ),
        abs_action=env_runner_cfg.get("abs_action", task_cfg.get("abs_action", True)),
        robot_noise_ratio=env_runner_cfg.get(
            "robot_noise_ratio", task_cfg.get("robot_noise_ratio", 0.1)
        ),
        n_episodes_per_combo=n_episodes_per_combo,
        combination_ids=combination_ids,
        resume=resume,
    )


@click.command()
@click.option("--checkpoints", "-c", multiple=True, help="Checkpoint paths or globs")
@click.option("--dataset_dir", default="data/kitchen")
@click.option("--output_root", "-o", default="data/kitchen_combo_eval/flowpolicy")
@click.option("--device", "-d", default="cuda:0")
@click.option("--resume/--no-resume", default=True)
@click.option("--smoke/--no-smoke", default=False, help="1 combo, 2 episodes")
@click.option("--combination_id", type=int, default=None, help="Run single combination")
@click.option("--n_episodes_per_combo", type=int, default=EPISODES_PER_COMBINATION)
def main(
    checkpoints: Tuple[str, ...],
    dataset_dir: str,
    output_root: str,
    device: str,
    resume: bool,
    smoke: bool,
    combination_id: Optional[int],
    n_episodes_per_combo: int,
):
    os.environ.setdefault("MUJOCO_GL", "egl")

    if smoke:
        n_episodes_per_combo = 2
        combination_ids = [0]
    elif combination_id is not None:
        combination_ids = [combination_id]
    else:
        combination_ids = None

    ckpt_patterns = list(checkpoints) if checkpoints else DEFAULT_CHECKPOINTS
    ckpt_paths = [resolve_checkpoint(p) for p in ckpt_patterns]
    output_model_dir = pathlib.Path(output_root)
    output_model_dir.mkdir(parents=True, exist_ok=True)

    click.echo("Model: flowpolicy")
    click.echo(f"Checkpoints: {ckpt_paths}")
    click.echo(f"Output: {output_model_dir}")
    if smoke:
        click.echo("SMOKE TEST: combination 0, 2 episodes")

    device_t = torch.device(device)
    seed_summaries: List[Dict[str, Any]] = []

    for ckpt_path in ckpt_paths:
        seed_name = seed_name_from_checkpoint(ckpt_path)
        seed_dir = output_model_dir / f"seed_{seed_name}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"\n--- Evaluating seed={seed_name} ---")
        click.echo(f"Checkpoint: {ckpt_path}")

        policy, cfg = load_policy(ckpt_path, device_t)
        runner = build_runner(
            cfg,
            output_dir=str(seed_dir),
            dataset_dir=dataset_dir,
            model_seed=seed_name,
            n_episodes_per_combo=n_episodes_per_combo,
            combination_ids=combination_ids,
            resume=resume,
        )
        seed_summary = runner.run(policy)
        seed_summaries.append(seed_summary)
        click.echo(f"Seed {seed_name} done. Report: {seed_dir / 'seed_report.txt'}")

    if len(seed_summaries) > 1 and combination_ids is None and not smoke:
        model_summary = aggregate_model_metrics(seed_summaries)
        save_json(model_summary, str(output_model_dir / "model_summary.json"))
        report = format_model_report(model_summary, "flowpolicy")
        with open(output_model_dir / "model_summary.txt", "w") as f:
            f.write(report)
        click.echo(f"\nModel summary: {output_model_dir / 'model_summary.txt'}")


if __name__ == "__main__":
    main()
