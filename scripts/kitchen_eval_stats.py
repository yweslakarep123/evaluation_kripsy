#!/usr/bin/env python3
"""Canonical Kitchen eval statistics: Wilson CI, latency stats, and p4 clip.

This module is the single source of truth for Wilson score intervals,
inference-latency summaries, and the p4 scoring ceiling.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

BinaryLike = Union[int, float, bool]

K_MAX = 4

TIMING_METHODOLOGY_KEYS = (
    "clock",
    "cuda_synchronize_before",
    "cuda_synchronize_after",
    "cuda_async_guard",
    "warmup_calls_discarded",
    "batch_size",
    "timed_region",
    "cpu_preprocessing_in_timed_region",
    "h2d_transfer_in_timed_region",
    "d2h_transfer_in_timed_region",
    "env_step_in_timed_region",
    "model_normalizer_in_timed_region",
    "n_timed_calls",
    "percentiles",
)


def compute_wilson_ci(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval for a binomial proportion.

    Returns ``(lower_bound, upper_bound)``, clipped to ``[0, 1]``.
    Returns ``(None, None)`` when ``trials == 0``.
    """
    if trials < 0:
        raise ValueError(f"trials must be >= 0, got {trials}")
    if successes < 0 or successes > trials:
        raise ValueError(
            f"successes must be in [0, trials], got {successes} / {trials}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if trials == 0:
        return None, None

    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    n = float(trials)
    p = float(successes) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lo = min(1.0, max(0.0, center - margin))
    hi = min(1.0, max(0.0, center + margin))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def proportion_stats(
    binary_values: Iterable[BinaryLike],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Empirical success rate plus Wilson 95% CI for 0/1 indicators.

    ``mean`` is ``k / n``. ``std`` is the Wilson interval half-width so
    existing ``mean ± std`` printers keep working. The bounded interval is
    ``[ci_low, ci_high]``.
    """
    values = list(binary_values)
    n = len(values)
    k = sum(1 for v in values if float(v) >= 0.5)
    if n == 0:
        return {
            "mean": None,
            "std": None,
            "n_samples": 0,
            "n_success": 0,
            "ci_low": None,
            "ci_high": None,
            "ci_method": "wilson",
            "confidence": float(confidence),
        }

    lo, hi = compute_wilson_ci(k, n, confidence=confidence)
    halfwidth = None
    if lo is not None and hi is not None:
        halfwidth = (hi - lo) / 2.0
    return {
        "mean": float(k) / float(n),
        "std": halfwidth,
        "n_samples": n,
        "n_success": k,
        "ci_low": lo,
        "ci_high": hi,
        "ci_method": "wilson",
        "confidence": float(confidence),
    }


def timing_sample_stats(values: Sequence[float]) -> Dict[str, Any]:
    """Mean and percentiles for inference latency (no standard deviation)."""
    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "mean": None,
            "n_samples": 0,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    return {
        "mean": float(np.mean(arr)),
        "n_samples": n,
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def save_rng_state() -> Dict[str, Any]:
    """Snapshot CPU / CUDA / NumPy RNG so warmup can be rolled back."""
    import torch

    state: Dict[str, Any] = {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    import torch

    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def build_timing_methodology(
    *,
    warmup_calls_discarded: int,
    n_timed_calls: int,
    batch_size: int = 1,
) -> Dict[str, Any]:
    return {
        "clock": "time.perf_counter",
        "cuda_synchronize_before": True,
        "cuda_synchronize_after": True,
        "cuda_async_guard": "synchronize around predict_action",
        "warmup_calls_discarded": int(warmup_calls_discarded),
        "batch_size": int(batch_size),
        "timed_region": "policy.predict_action only",
        "cpu_preprocessing_in_timed_region": False,
        "h2d_transfer_in_timed_region": False,
        "d2h_transfer_in_timed_region": False,
        "env_step_in_timed_region": False,
        "model_normalizer_in_timed_region": True,
        "n_timed_calls": int(n_timed_calls),
        "percentiles": [50, 95, 99],
    }


def clip_episode_to_p4(
    completion_order: Optional[Sequence[str]] = None,
    all_tasks: Sequence[str] = (),
    k_max: int = K_MAX,
    completed_tasks: Optional[Iterable[str]] = None,
) -> Tuple[List[str], int, Dict[str, Optional[int]]]:
    """Score only the first ``k_max`` completions (any-order p4 ceiling).

    Returns ``(scored_order, k, flags)`` where ``flags[task]`` is 1 (success),
    0 (fail, only if ``k < k_max``), or ``None`` (exclude: leftover after p4).
    """
    order = [str(t) for t in (completion_order or [])]
    if order:
        scored = order[: int(k_max)]
        k = len(scored)
        scored_set = set(scored)
        flags: Dict[str, Optional[int]] = {}
        for t in all_tasks:
            name = str(t)
            if name in scored_set:
                flags[name] = 1
            elif k >= int(k_max):
                flags[name] = None
            else:
                flags[name] = 0
        return scored, min(k, int(k_max)), flags

    completed = [str(t) for t in (completed_tasks or [])]
    k_raw = len(completed)
    k = min(k_raw, int(k_max))
    completed_set = set(completed)
    flags = {}
    for t in all_tasks:
        name = str(t)
        if name in completed_set:
            flags[name] = 1
        elif k >= int(k_max):
            flags[name] = None
        else:
            flags[name] = 0
    return completed[: int(k_max)], k, flags


def clip_episode_record(
    episode: Dict[str, Any],
    all_tasks: Sequence[str],
    k_max: int = K_MAX,
) -> Tuple[List[str], int, Dict[str, Optional[int]]]:
    return clip_episode_to_p4(
        completion_order=episode.get("completion_order") or [],
        all_tasks=all_tasks,
        k_max=k_max,
        completed_tasks=episode.get("completed_tasks") or [],
    )


def episode_k_capped(episode: Dict[str, Any], k_max: int = K_MAX) -> int:
    """min(completions, k_max) from order, else from completed_tasks / count."""
    _, k, _ = clip_episode_record(episode, all_tasks=(), k_max=k_max)
    if k > 0:
        return k
    raw = episode.get("num_tasks_completed")
    if raw is None:
        return 0
    return min(int(raw), int(k_max))


def classify_p4_outcome(k: int, k_max: int = K_MAX) -> str:
    """Binary episode outcome under the p4 ceiling."""
    if int(k) < int(k_max):
        return "miss_subtask"
    return "success_p4"


def clipped_per_task_stats(
    episodes: Sequence[Dict[str, Any]],
    all_tasks: Sequence[str],
    k_max: int = K_MAX,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Per-task Wilson stats and p4_success under the scoring ceiling."""
    buckets: Dict[str, List[float]] = {str(t): [] for t in all_tasks}
    p4_flags: List[float] = []
    for ep in episodes:
        _, k, flags = clip_episode_record(ep, all_tasks, k_max=k_max)
        p4_flags.append(1.0 if k >= int(k_max) else 0.0)
        for t in all_tasks:
            name = str(t)
            flag = flags.get(name)
            if flag is not None:
                buckets[name].append(float(flag))
    per_task = {t: proportion_stats(buckets[t]) for t in buckets}
    return per_task, proportion_stats(p4_flags)
