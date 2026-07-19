#!/usr/bin/env python3
"""Backfill multistage mean/std into existing kitchen eval_metrics.json files.

Recomputes p_k and cumulative_order_success_rate from stored episodes
(without re-running evaluation) and regenerates eval_report.txt.

Usage:
  python scripts/backfill_multistage_mean_std.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALL_TASKS = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]
KITCHEN_4_SUBGOALS = ["microwave", "kettle", "bottom burner", "light switch"]

EVAL_ROOTS = [
    ROOT / "kripsy12/FlowPolicy/data/kitchen_eval_nfe100",
    ROOT / "diffusion_policy/data/kitchen_eval_nfe100_diffusion",
]


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


def compute_multistage_metrics(
    episodes: Sequence[Dict[str, Any]],
    sub_goals: Sequence[str],
    num_sub_goals: Optional[int] = None,
) -> Dict[str, Any]:
    goal_set = set(sub_goals)
    counts: List[int] = []
    for ep in episodes:
        completed = set(ep.get("completed_tasks", [])) & goal_set
        counts.append(len(completed))

    k = num_sub_goals if num_sub_goals is not None else len(sub_goals)
    n = len(episodes)
    empty = {"mean": 0.0, "std": 0.0, "n_samples": 0}
    if n == 0:
        return {
            "px": {f"p{i}": dict(empty) for i in range(1, k + 1)},
            "cumulative_order_success_rate": dict(empty),
            "sub_goals": list(sub_goals),
        }

    px = {
        f"p{i}": _compute_mean_std([1.0 if c >= i else 0.0 for c in counts])
        for i in range(1, k + 1)
    }
    all_success = _compute_mean_std(
        [
            1.0 if goal_set.issubset(set(ep.get("completed_tasks", []))) else 0.0
            for ep in episodes
        ]
    )
    return {
        "px": px,
        "cumulative_order_success_rate": all_success,
        "sub_goals": list(sub_goals),
    }


def _old_px_mean(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict):
        m = value.get("mean")
        return float(m) if m is not None else None
    return float(value)


def _metric_mean(value: Any) -> Optional[float]:
    return _old_px_mean(value)


def _metric_std(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        s = value.get("std")
        return float(s) if s is not None else None
    return None


def format_eval_report(metrics: Dict[str, Any], seed_name: str) -> str:
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
        lines.extend(
            [
                "",
                "Cumulative episode success (all 7 tasks completed)",
                "-" * 72,
                f"  success rate: {cum['mean']:.3f} ± {cum['std']:.3f}",
            ]
        )

    ms7 = metrics.get("multistage_metrics", {}).get("all_7_tasks", {})
    px7 = ms7.get("px", {})
    if px7:
        lines.extend(["", "Multi-stage p_k (>= k of 7 tasks completed)", "-" * 72])
        parts = []
        for k in range(1, len(ALL_TASKS) + 1):
            pk = f"p{k}"
            if pk not in px7:
                continue
            mean = _metric_mean(px7[pk])
            std = _metric_std(px7[pk])
            if mean is None:
                continue
            if std is not None:
                parts.append(f"p{k}={mean:.3f} ± {std:.3f}")
            else:
                parts.append(f"p{k}={mean:.3f}")
        lines.append(f"  {'  '.join(parts)}")

    inf = metrics.get("timing_ms", {}).get("inference_latency", {})
    if inf.get("mean") is not None:
        lines.extend(
            [
                "",
                "Timing",
                "-" * 72,
                f"  Inference latency (ms): {inf['mean']:.2f} ± {inf['std']:.2f}",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def backfill_one(path: Path) -> Dict[str, Any]:
    metrics = json.loads(path.read_text())
    episodes = metrics.get("episodes") or []
    if not episodes:
        raise ValueError(f"No episodes in {path}")

    old_ms = metrics.get("multistage_metrics", {})
    old_means = {
        label: {
            pk: _old_px_mean(v)
            for pk, v in old_ms.get(label, {}).get("px", {}).items()
        }
        for label in ("all_7_tasks", "paper_4_tasks")
        if label in old_ms
    }

    ms7 = compute_multistage_metrics(episodes, sub_goals=ALL_TASKS)
    ms4 = compute_multistage_metrics(
        episodes, sub_goals=KITCHEN_4_SUBGOALS, num_sub_goals=4
    )
    metrics["multistage_metrics"] = {
        "all_7_tasks": {
            "px": ms7["px"],
            "cumulative_order_success_rate": ms7["cumulative_order_success_rate"],
            "sub_goals": ms7["sub_goals"],
        },
        "paper_4_tasks": {
            "px": ms4["px"],
            "cumulative_order_success_rate": ms4["cumulative_order_success_rate"],
            "sub_goals": ms4["sub_goals"],
        },
    }

    # Verify means match prior float values (within float tolerance)
    mismatches = []
    for label, old_px in old_means.items():
        new_px = metrics["multistage_metrics"][label]["px"]
        for pk, old_m in old_px.items():
            new_m = new_px[pk]["mean"]
            if old_m is not None and abs(new_m - old_m) > 1e-9:
                mismatches.append((label, pk, old_m, new_m))

    path.write_text(json.dumps(metrics, indent=2) + "\n")

    seed_name = metrics.get("model_seed") or path.parent.name
    report_path = path.parent / "eval_report.txt"
    report_path.write_text(format_eval_report(metrics, str(seed_name)))

    return {
        "path": str(path),
        "n_episodes": len(episodes),
        "mismatches": mismatches,
        "p1": metrics["multistage_metrics"]["all_7_tasks"]["px"]["p1"],
        "p4": metrics["multistage_metrics"]["all_7_tasks"]["px"]["p4"],
    }


def main() -> int:
    paths: List[Path] = []
    for root in EVAL_ROOTS:
        if not root.is_dir():
            print(f"skip missing root: {root}")
            continue
        paths.extend(sorted(root.rglob("eval_metrics.json")))

    if not paths:
        print("No eval_metrics.json found", file=sys.stderr)
        return 1

    n_ok = 0
    n_mismatch = 0
    for path in paths:
        info = backfill_one(path)
        n_ok += 1
        if info["mismatches"]:
            n_mismatch += 1
            print(f"MEAN MISMATCH {path}: {info['mismatches']}")
        else:
            p1, p4 = info["p1"], info["p4"]
            print(
                f"ok {path.parent.name}: "
                f"p1={p1['mean']:.3f}±{p1['std']:.3f}  "
                f"p4={p4['mean']:.3f}±{p4['std']:.3f}"
            )

    print(f"\nBackfilled {n_ok} files; mean mismatches: {n_mismatch}")
    return 0 if n_mismatch == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
