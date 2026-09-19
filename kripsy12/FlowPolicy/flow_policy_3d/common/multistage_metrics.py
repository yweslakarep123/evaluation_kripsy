"""Multi-stage p_k metrics for Kitchen multitask evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _import_kitchen_eval_stats():
    """Load the canonical stats helper; do not duplicate Wilson/timing formulas."""
    try:
        import kitchen_eval_stats as kes  # type: ignore
        return kes
    except ImportError:
        pass
    cur = Path(__file__).resolve().parent
    for _ in range(12):
        candidate = cur / "scripts" / "kitchen_eval_stats.py"
        if candidate.is_file():
            scripts_dir = str(candidate.parent)
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            import kitchen_eval_stats as kes  # type: ignore
            return kes
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    raise ImportError(
        "Could not locate scripts/kitchen_eval_stats.py for Wilson CI helpers"
    )


_kes = _import_kitchen_eval_stats()
proportion_stats = _kes.proportion_stats
timing_sample_stats = _kes.timing_sample_stats
save_rng_state = _kes.save_rng_state
restore_rng_state = _kes.restore_rng_state
build_timing_methodology = _kes.build_timing_methodology
clip_episode_to_p4 = _kes.clip_episode_to_p4
clip_episode_record = _kes.clip_episode_record
episode_k_capped = _kes.episode_k_capped
classify_p4_outcome = _kes.classify_p4_outcome
clipped_per_task_stats = _kes.clipped_per_task_stats
K_MAX = _kes.K_MAX


def compute_multistage_metrics(
    episodes: Sequence[Dict[str, Any]],
    sub_goals: Sequence[str],
    num_sub_goals: Optional[int] = None,
) -> Dict[str, Any]:
    """Return px (p1..pK), cumulative_order_success_rate, and sub_goals.

    Each p_k and cumulative_order_success_rate is a proportion_stats dict
    (mean = k/n, std = Wilson 95% half-width, plus ci_low/ci_high).
    """
    goal_set = set(sub_goals)
    counts: List[int] = []
    for ep in episodes:
        completed = set(ep.get("completed_tasks", [])) & goal_set
        counts.append(len(completed))

    k = num_sub_goals if num_sub_goals is not None else len(sub_goals)
    n = len(episodes)
    if n == 0:
        empty = proportion_stats([])
        px = {f"p{i}": dict(empty) for i in range(1, k + 1)}
        return {
            "px": px,
            "cumulative_order_success_rate": dict(empty),
            "sub_goals": list(sub_goals),
        }

    px = {
        f"p{i}": proportion_stats([1.0 if c >= i else 0.0 for c in counts])
        for i in range(1, k + 1)
    }
    all_success = proportion_stats(
        [1.0 if c >= k else 0.0 for c in counts]
    )
    return {
        "px": px,
        "cumulative_order_success_rate": all_success,
        "sub_goals": list(sub_goals),
    }
