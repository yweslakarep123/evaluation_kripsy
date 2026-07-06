"""
Kitchen lowdim evaluation for FlowPolicy checkpoints (7-task, p1–p7 protocol).

Uses KitchenLowdimEvalRunner + cross-seed aggregation per KITCHEN_EVAL_PORTING.md.

Example:
  MUJOCO_GL=egl python infer_kitchen_lowdim.py \\
    --checkpoint runs/foo/checkpoints/latest.ckpt \\
    --metrics-json runs/foo/metrics.json \\
    --n-infer-episodes 50 --eval-seeds 0,42,101
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import torch

if __name__ == "__main__":
    _root = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_root))
    os.chdir(str(_root))

from eval_kitchen import (  # noqa: E402
    aggregate_seed_metrics,
    build_runner,
)
from flow_policy_3d.env_runner.kitchen_lowdim_eval_runner import (  # noqa: E402
    ALL_TASKS,
    KITCHEN_4_SUBGOALS,
)
from train import TrainFlowPolicyWorkspace  # noqa: E402


def _task_key(task_name: str) -> str:
    return task_name.replace(" ", "_")


def _metrics_to_test_json(
    summary: dict,
    per_seed: list,
    n_episodes: int,
    eval_seeds: list,
) -> dict:
    """Flatten aggregated eval into test_* keys for results.csv parser."""
    out: dict = {
        "test_n_infer_episodes": n_episodes,
        "eval_seeds": eval_seeds,
        "n_eval_seeds": len(eval_seeds),
    }

    sr = summary.get("success_rate", {})
    all7 = sr.get("all_7_tasks", {})
    out["test_all_7_success"] = all7.get("mean")
    out["test_std_all_7_success"] = all7.get("std")
    out["success_rate_total"] = all7.get("mean")
    out["std_success_rate_total"] = all7.get("std")

    for task in ALL_TASKS:
        stat = sr.get(task, {})
        key = f"test_success_{_task_key(task)}"
        out[key] = stat.get("mean")
        out[f"{key}_std"] = stat.get("std")

    ms7 = summary.get("multistage_metrics", {}).get("all_7_tasks", {})
    px7 = ms7.get("px", {})
    for k in range(1, len(ALL_TASKS) + 1):
        pk = f"p{k}"
        if pk in px7:
            val = px7[pk]
            if isinstance(val, dict):
                out[f"test_{pk}"] = val.get("mean")
                out[f"test_std_{pk}"] = val.get("std")
            else:
                out[f"test_{pk}"] = val

    ms4 = summary.get("multistage_metrics", {}).get("paper_4_tasks", {})
    px4 = ms4.get("px", {})
    p4_val = px4.get("p4")
    if isinstance(p4_val, dict):
        out["test_p4_paper"] = p4_val.get("mean")
        out["test_std_p4_paper"] = p4_val.get("std")
    elif p4_val is not None:
        out["test_p4_paper"] = p4_val
    for k in range(1, 5):
        pk = f"p{k}"
        if pk in px4:
            val = px4[pk]
            if isinstance(val, dict):
                out[f"test_paper_{pk}"] = val.get("mean")
            else:
                out[f"test_paper_{pk}"] = val
    out["success_rate_k4"] = out.get("test_p4_paper")
    out["std_success_rate_k4"] = out.get("test_std_p4_paper")

    timing = summary.get("timing_ms", {})
    for tkey in ["inference_latency", "episode_duration"]:
        stat = timing.get(tkey, {})
        out[f"test_mean_{tkey}_ms"] = stat.get("mean")
        out[f"test_std_{tkey}_ms"] = stat.get("std")
    if tkey := timing.get("inference_latency"):
        out["mean_inference_latency_ms"] = tkey.get("mean")
        out["std_inference_latency_ms"] = tkey.get("std")

    td = timing.get("task_duration", {})
    overall = td.get("overall", {})
    out["test_mean_task_duration_ms"] = overall.get("mean")
    out["test_std_task_duration_ms"] = overall.get("std")

    out["multistage_metrics"] = summary.get("multistage_metrics")
    out["success_rate"] = summary.get("success_rate")
    out["timing_ms"] = summary.get("timing_ms")
    out["per_seed_metrics"] = {
        str(m["eval_seed"]): {
            "success_rate": m.get("success_rate"),
            "multistage_metrics": m.get("multistage_metrics"),
            "timing_ms": m.get("timing_ms"),
        }
        for m in per_seed
    }
    out["paper_4_subgoals"] = KITCHEN_4_SUBGOALS
    out["all_tasks"] = ALL_TASKS
    return out


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--metrics-json", type=str, required=True)
    p.add_argument("--n-infer-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="Training/random seed (logging only)")
    p.add_argument(
        "--eval-seeds",
        type=str,
        default="0,42,101",
        help="Comma-separated eval seeds (KitchenAllV0 protocol)",
    )
    p.add_argument(
        "--skip-inference-videos",
        action="store_true",
        help="Disable MP4 recording during eval",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ckpt = pathlib.Path(args.checkpoint).resolve()
    workspace = TrainFlowPolicyWorkspace.create_from_checkpoint(str(ckpt))
    cfg = workspace.cfg
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval()
    device = torch.device(cfg.training.device)
    policy.to(device)

    eval_seeds = [int(s.strip()) for s in args.eval_seeds.split(",") if s.strip()]
    run_dir = pathlib.Path(args.metrics_json).resolve().parent
    eval_root = run_dir / "kitchen_eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    n_vis = 0 if args.skip_inference_videos else args.n_infer_episodes
    seed_metrics = []

    for eval_seed in eval_seeds:
        seed_dir = eval_root / f"seed_{eval_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        runner = build_runner(
            cfg=cfg,
            output_dir=str(seed_dir),
            eval_seed=eval_seed,
            n_episodes=args.n_infer_episodes,
        )
        runner.n_episodes_vis = n_vis
        try:
            metrics = runner.run(policy)
        finally:
            pass
        metrics_path = seed_dir / "eval_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        seed_metrics.append(metrics)

    summary = aggregate_seed_metrics(seed_metrics)
    payload = _metrics_to_test_json(
        summary, seed_metrics, args.n_infer_episodes, eval_seeds
    )
    payload["checkpoint"] = str(ckpt)

    out_path = pathlib.Path(args.metrics_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    summary_path = eval_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
