"""Multi-stage p_k metrics for Kitchen multitask evaluation."""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def compute_multistage_metrics(
    episodes: Sequence[Dict[str, Any]],
    sub_goals: Sequence[str],
    num_sub_goals: Optional[int] = None,
) -> Dict[str, Any]:
    """Return px (p1..pK), cumulative_order_success_rate, and sub_goals."""
    goal_set = set(sub_goals)
    counts: List[int] = []
    for ep in episodes:
        completed = set(ep.get("completed_tasks", [])) & goal_set
        counts.append(len(completed))

    k = num_sub_goals if num_sub_goals is not None else len(sub_goals)
    n = len(episodes)
    if n == 0:
        px = {f"p{i}": 0.0 for i in range(1, k + 1)}
        return {
            "px": px,
            "cumulative_order_success_rate": 0.0,
            "sub_goals": list(sub_goals),
        }

    px = {
        f"p{i}": float(np.mean([c >= i for c in counts]))
        for i in range(1, k + 1)
    }
    all_success = float(
        np.mean([goal_set.issubset(set(ep.get("completed_tasks", []))) for ep in episodes])
    )
    return {
        "px": px,
        "cumulative_order_success_rate": all_success,
        "sub_goals": list(sub_goals),
    }
