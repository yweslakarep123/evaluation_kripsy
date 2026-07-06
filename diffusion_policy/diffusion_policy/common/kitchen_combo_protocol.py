"""Protocol utilities for Kitchen 4-subtask combinatorial evaluation."""

from __future__ import annotations

import itertools
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ALL_TASKS = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]

NUM_COMBINATIONS = 35
EPISODES_PER_COMBINATION = 50
NUM_SUBTASKS = 4


def compute_mean_std(values: Sequence[float]) -> Dict[str, Any]:
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


def generate_combinations() -> List[Tuple[str, ...]]:
    """Return 35 lexicographically sorted 4-task combinations."""
    return list(itertools.combinations(ALL_TASKS, NUM_SUBTASKS))


def generate_balanced_permutations(
    tasks: Sequence[str],
    n: int = EPISODES_PER_COMBINATION,
    seed: int = 0,
) -> List[List[str]]:
    """Generate n permutations with roughly equal task-position counts."""
    tasks = list(tasks)
    all_perms = list(itertools.permutations(tasks))
    rng = np.random.RandomState(seed)
    repeats = n // len(all_perms)
    remainder = n % len(all_perms)
    result: List[List[str]] = [list(p) for p in all_perms * repeats]
    if remainder > 0:
        extra_idx = rng.choice(len(all_perms), size=remainder, replace=False)
        result.extend([list(all_perms[i]) for i in extra_idx])
    rng.shuffle(result)
    return result[:n]


def build_episode_schedule(
    n_episodes_per_combo: int = EPISODES_PER_COMBINATION,
    n_inits: int = 566,
    combination_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Build flat episode schedule: 35 combos × n_episodes each."""
    combos = generate_combinations()
    if combination_ids is None:
        combination_ids = list(range(len(combos)))
    schedule: List[Dict[str, Any]] = []
    for combo_id in combination_ids:
        combo_tasks = list(combos[combo_id])
        permutations = generate_balanced_permutations(
            combo_tasks, n=n_episodes_per_combo, seed=combo_id
        )
        for ep_id, task_sequence in enumerate(permutations):
            init_idx = (combo_id * n_episodes_per_combo + ep_id) % n_inits
            schedule.append(
                {
                    "combination_id": combo_id,
                    "combination_tasks": combo_tasks,
                    "episode_id": ep_id,
                    "task_sequence": task_sequence,
                    "init_idx": init_idx,
                }
            )
    return schedule


def count_completed_in_sequence(
    completed_tasks: Sequence[str],
    task_sequence: Sequence[str],
) -> int:
    """Count prefix of task_sequence completed in order."""
    completed_set = set(completed_tasks)
    count = 0
    for task in task_sequence:
        if task in completed_set:
            count += 1
        else:
            break
    return count


def compute_sequential_pk(
    episodes: Sequence[Dict[str, Any]],
    task_sequence: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Compute p1..p4 for sequential prefix completion."""
    pk_values: Dict[str, List[float]] = {f"p{k}": [] for k in range(1, NUM_SUBTASKS + 1)}
    for ep in episodes:
        seq = task_sequence or ep.get("task_sequence", [])
        n_done = count_completed_in_sequence(ep.get("completed_tasks", []), seq)
        for k in range(1, NUM_SUBTASKS + 1):
            pk_values[f"p{k}"].append(1.0 if n_done >= k else 0.0)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in pk_values.items()}


def aggregate_combination_metrics(
    episodes: Sequence[Dict[str, Any]],
    combination_id: int,
    combination_tasks: Sequence[str],
) -> Dict[str, Any]:
    """Aggregate metrics over episodes for one combination."""
    pk_raw: Dict[str, List[float]] = {f"p{k}": [] for k in range(1, NUM_SUBTASKS + 1)}
    per_task_success: Dict[str, List[float]] = {t: [] for t in combination_tasks}
    per_task_duration: Dict[str, List[float]] = {t: [] for t in combination_tasks}
    inference_latencies: List[float] = []
    episode_durations: List[float] = []

    for ep in episodes:
        seq = ep.get("task_sequence", [])
        n_done = count_completed_in_sequence(ep.get("completed_tasks", []), seq)
        for k in range(1, NUM_SUBTASKS + 1):
            pk_raw[f"p{k}"].append(1.0 if n_done >= k else 0.0)

        completed_set = set(ep.get("completed_tasks", []))
        for task in combination_tasks:
            per_task_success[task].append(1.0 if task in completed_set else 0.0)

        for task, dur in ep.get("task_durations_ms", {}).items():
            if task in per_task_duration:
                per_task_duration[task].append(float(dur))

        inference_latencies.extend(ep.get("inference_latencies_ms", []))
        if ep.get("episode_duration_ms") is not None:
            episode_durations.append(float(ep["episode_duration_ms"]))

    sequential_pk = {k: compute_mean_std(v) for k, v in pk_raw.items()}
    return {
        "combination_id": combination_id,
        "tasks": list(combination_tasks),
        "n_episodes": len(episodes),
        "sequential_pk": sequential_pk,
        "per_task_success": {
            t: compute_mean_std(per_task_success[t]) for t in combination_tasks
        },
        "per_task_duration_ms": {
            t: compute_mean_std(per_task_duration[t]) for t in combination_tasks
        },
        "inference_latency_ms": compute_mean_std(inference_latencies),
        "episode_duration_ms": compute_mean_std(episode_durations),
    }


def aggregate_seed_metrics(
    combo_metrics: Sequence[Dict[str, Any]],
    model_seed: str,
) -> Dict[str, Any]:
    """Aggregate across 35 combinations for one checkpoint seed."""
    per_combo_pk: Dict[str, List[float]] = {f"p{k}": [] for k in range(1, NUM_SUBTASKS + 1)}
    per_task_duration_pooled: Dict[str, List[float]] = {t: [] for t in ALL_TASKS}
    inference_latencies: List[float] = []
    episode_durations: List[float] = []

    combination_summaries = []
    for cm in combo_metrics:
        combination_summaries.append(
            {
                "combination_id": cm["combination_id"],
                "tasks": cm["tasks"],
                "sequential_pk": {
                    k: v["mean"] for k, v in cm["sequential_pk"].items() if v["mean"] is not None
                },
            }
        )
        for k in range(1, NUM_SUBTASKS + 1):
            pk_stat = cm["sequential_pk"][f"p{k}"]
            if pk_stat["mean"] is not None:
                per_combo_pk[f"p{k}"].append(pk_stat["mean"])

        for task in ALL_TASKS:
            if task in cm.get("per_task_duration_ms", {}):
                dur = cm["per_task_duration_ms"][task]
                if dur["mean"] is not None and dur["n_samples"] > 0:
                    per_task_duration_pooled[task].append(dur["mean"])

        inf = cm["inference_latency_ms"]
        if inf["mean"] is not None:
            inference_latencies.append(inf["mean"])
        ep_dur = cm["episode_duration_ms"]
        if ep_dur["mean"] is not None:
            episode_durations.append(ep_dur["mean"])

    overall_pk = {k: compute_mean_std(v) for k, v in per_combo_pk.items()}
    overall_per_task_success = {}
    for task in ALL_TASKS:
        vals = []
        for cm in combo_metrics:
            if task in cm["tasks"]:
                sr = cm["per_task_success"][task]
                if sr["mean"] is not None:
                    vals.append(sr["mean"])
        overall_per_task_success[task] = compute_mean_std(vals)

    overall_per_task_duration = {
        t: compute_mean_std(per_task_duration_pooled[t]) for t in ALL_TASKS
    }

    return {
        "model_seed": model_seed,
        "n_combinations": len(combo_metrics),
        "n_episodes_total": sum(cm["n_episodes"] for cm in combo_metrics),
        "overall_sequential_pk": overall_pk,
        "overall_per_task_success": overall_per_task_success,
        "overall_per_task_duration_ms": overall_per_task_duration,
        "overall_inference_latency_ms": compute_mean_std(inference_latencies),
        "overall_episode_duration_ms": compute_mean_std(episode_durations),
        "combination_summaries": combination_summaries,
    }


def aggregate_model_metrics(seed_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-seed summary: mean-of-seed-means ± std."""
    result: Dict[str, Any] = {
        "seeds": [s["model_seed"] for s in seed_summaries],
        "n_seeds": len(seed_summaries),
    }

    pk_keys = [f"p{k}" for k in range(1, NUM_SUBTASKS + 1)]
    result["overall_sequential_pk"] = {}
    for pk in pk_keys:
        means = [
            s["overall_sequential_pk"][pk]["mean"]
            for s in seed_summaries
            if s["overall_sequential_pk"][pk]["mean"] is not None
        ]
        result["overall_sequential_pk"][pk] = compute_mean_std(means)

    result["overall_per_task_success"] = {}
    for task in ALL_TASKS:
        means = [
            s["overall_per_task_success"][task]["mean"]
            for s in seed_summaries
            if s["overall_per_task_success"][task]["mean"] is not None
        ]
        result["overall_per_task_success"][task] = compute_mean_std(means)

    result["overall_per_task_duration_ms"] = {}
    for task in ALL_TASKS:
        means = [
            s["overall_per_task_duration_ms"][task]["mean"]
            for s in seed_summaries
            if s["overall_per_task_duration_ms"][task]["mean"] is not None
        ]
        result["overall_per_task_duration_ms"][task] = compute_mean_std(means)

    for timing_key in ["overall_inference_latency_ms", "overall_episode_duration_ms"]:
        means = [
            s[timing_key]["mean"]
            for s in seed_summaries
            if s[timing_key]["mean"] is not None
        ]
        result[timing_key] = compute_mean_std(means)

    result["per_seed"] = {s["model_seed"]: s for s in seed_summaries}
    return result


def format_seed_report(seed_summary: Dict[str, Any]) -> str:
    """Human-readable report for one seed."""
    lines = [
        "=" * 80,
        f"Kitchen Combo Eval | seed={seed_summary['model_seed']} | "
        f"{seed_summary['n_episodes_total']} episodes | "
        f"{seed_summary['n_combinations']} combinations",
        "=" * 80,
        "",
        "Per-combination sequential p1-p4 (mean over 50 episodes)",
        "-" * 80,
        f"{'combo':>5}  {'tasks':<45}  p1      p2      p3      p4",
        "-" * 80,
    ]
    for cs in seed_summary["combination_summaries"]:
        tasks_str = ", ".join(t.split()[0] for t in cs["tasks"])
        pk = cs["sequential_pk"]
        lines.append(
            f"{cs['combination_id']:>5}  {tasks_str:<45}  "
            f"{pk.get('p1', 0):.3f}   {pk.get('p2', 0):.3f}   "
            f"{pk.get('p3', 0):.3f}   {pk.get('p4', 0):.3f}"
        )

    lines.extend(["", "Overall aggregate (mean ± std across 35 combinations)", "-" * 80])
    pk = seed_summary["overall_sequential_pk"]
    pk_line = "  ".join(
        f"p{k}={pk[f'p{k}']['mean']:.3f}±{pk[f'p{k}']['std']:.3f}"
        for k in range(1, NUM_SUBTASKS + 1)
        if pk[f"p{k}"]["mean"] is not None
    )
    lines.append(f"  Sequential: {pk_line}")

    lines.extend(["", "Per-task success rate (mean ± std across combinations)", "-" * 80])
    for task in ALL_TASKS:
        stat = seed_summary["overall_per_task_success"][task]
        if stat["mean"] is not None:
            lines.append(f"  {task:<16}  {stat['mean']:.3f} ± {stat['std']:.3f}")

    lines.extend(["", "Per-task duration ms (mean ± std across combinations)", "-" * 80])
    for task in ALL_TASKS:
        stat = seed_summary["overall_per_task_duration_ms"][task]
        if stat["mean"] is not None:
            lines.append(
                f"  {task:<16}  {stat['mean']:.1f} ± {stat['std']:.1f}  (n={stat['n_samples']})"
            )

    inf = seed_summary["overall_inference_latency_ms"]
    ep = seed_summary["overall_episode_duration_ms"]
    lines.extend([
        "",
        "Timing",
        "-" * 80,
        f"  Inference latency (ms): {inf['mean']:.2f} ± {inf['std']:.2f}",
        f"  Episode duration (ms):  {ep['mean']:.1f} ± {ep['std']:.1f}",
        "",
    ])
    return "\n".join(lines)


def format_model_report(model_summary: Dict[str, Any], model_name: str) -> str:
    """Human-readable cross-seed report."""
    lines = [
        "=" * 80,
        f"Kitchen Combo Eval Model Summary | {model_name} | seeds={model_summary['seeds']}",
        "=" * 80,
        "",
        "Overall sequential p_k (mean ± std across seeds)",
        "-" * 80,
    ]
    pk = model_summary["overall_sequential_pk"]
    for k in range(1, NUM_SUBTASKS + 1):
        stat = pk[f"p{k}"]
        if stat["mean"] is not None:
            lines.append(f"  p{k}: {stat['mean']:.3f} ± {stat['std']:.3f}")

    lines.extend(["", "Per-task success rate (mean ± std across seeds)", "-" * 80])
    for task in ALL_TASKS:
        stat = model_summary["overall_per_task_success"][task]
        if stat["mean"] is not None:
            lines.append(f"  {task:<16}  {stat['mean']:.3f} ± {stat['std']:.3f}")

    lines.extend(["", "Per-task duration ms (mean ± std across seeds)", "-" * 80])
    for task in ALL_TASKS:
        stat = model_summary["overall_per_task_duration_ms"][task]
        if stat["mean"] is not None:
            lines.append(f"  {task:<16}  {stat['mean']:.1f} ± {stat['std']:.1f}")

    lines.append("")
    return "\n".join(lines)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)
