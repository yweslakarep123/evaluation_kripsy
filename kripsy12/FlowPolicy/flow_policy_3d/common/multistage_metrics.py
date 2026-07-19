"""Multi-stage p_k metrics for Kitchen multitask evaluation."""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


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
    """Return px (p1..pK), cumulative_order_success_rate, and sub_goals.

    Each p_k and cumulative_order_success_rate is {mean, std, n_samples}
    over episode-level indicators (same convention as success_rate).
    """
    goal_set = set(sub_goals)
    counts: List[int] = []
    for ep in episodes:
        completed = set(ep.get("completed_tasks", [])) & goal_set
        counts.append(len(completed))

    k = num_sub_goals if num_sub_goals is not None else len(sub_goals)
    n = len(episodes)
    empty = {"mean": 0.0, "std": 0.0, "n_samples": 0}
    if n == 0:
        px = {f"p{i}": dict(empty) for i in range(1, k + 1)}
        return {
            "px": px,
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
