# Kitchen Multi-Seed Evaluation — Implementation Guide

Portable spec for implementing the same evaluation protocol used in
`eval_kitchen.py` + `KitchenLowdimEvalRunner` (Diffusion Policy codebase).

Use this document to re-implement the evaluation in another codebase.

---

## 1. Goal

Evaluate a trained policy on the **Franka Kitchen** multitask environment with:

- Multiple eval seeds (default: `0`, `42`, `101`)
- N episodes per seed (default: `50` or `100`)
- One video per episode
- Per-episode task completion order
- Aggregated metrics: per-task success, p1..pN, cumulative all-task success, timing

---

## 2. Environment assumptions

### Task set (KitchenAllV0 — 7 sub-goals)

| # | Task name       |
|---|-----------------|
| 1 | bottom burner   |
| 2 | top burner      |
| 3 | light switch    |
| 4 | slide cabinet   |
| 5 | hinge cabinet   |
| 6 | microwave       |
| 7 | kettle          |

- Tasks may complete in **any order** (`COMPLETE_IN_ANY_ORDER=True`).
- Episode ends when **all 7 tasks are done** OR **max_steps** is reached (default `280`).
- Task completion: object state within distance threshold of goal (~0.3).

### Optional 4-task subset (paper / BET benchmark)

```python
KITCHEN_4_SUBGOALS = [
    "microwave",
    "kettle",
    "bottom burner",
    "light switch",
]
```

Use this for p1–p4 reporting aligned with Diffusion Policy paper Table 4.

---

## 3. Episode loop (core logic)

For each `eval_seed` in `[0, 42, 101]`:

For each `episode_idx` in `0 .. n_episodes-1`:

1. **Create fresh env** (do not reuse without `close()` — causes OOM).
2. Set `env_seed = eval_seed + episode_idx`.
3. Set video path: `media/seed{eval_seed}_ep{episode_idx:03d}.mp4`.
4. `obs = env.reset()`; `policy.reset()`.
5. Loop until `done`:
   - Measure inference latency around `policy.predict_action(obs)`.
   - `obs, reward, done, info = env.step(action)`.
   - Read `completed_tasks` from `info` (set of finished task names).
   - Diff with previous step → append newly completed tasks to `completion_order`.
   - Record timestamp for task duration (see §5).
6. **Always** `env.close()` in a `finally` block.
7. Append episode record to results list.

---

## 4. Metrics definitions

### 4.1 Per episode (store in JSON)

```python
{
  "episode_idx": 0,
  "env_seed": 0,
  "completion_order": ["kettle", "bottom burner", "microwave"],
  "completed_tasks": ["bottom burner", "kettle", "microwave"],
  "task_success": {
    "bottom burner": 1,
    "top burner": 0,
    "light switch": 0,
    "slide cabinet": 0,
    "hinge cabinet": 0,
    "microwave": 1,
    "kettle": 1
  },
  "num_tasks_completed": 3,
  "all_7_success": false,
  "video_path": "media/seed0_ep000.mp4",
  "episode_duration_ms": 34592.0,
  "task_durations_ms": {"kettle": 8011.0, "bottom burner": 8731.0},
  "mean_inference_latency_ms": 860.8
}
```

### 4.2 Per-task success rate (over N episodes)

For each task `t`:

```
success_rate[t].mean = (# episodes where task t completed) / N
success_rate[t].std  = std dev of N binary values {0,1}, ddof=1
```

### 4.3 Cumulative episode success (all 7 tasks)

```
all_7_success_rate = (# episodes where all 7 tasks completed) / N
```

Equivalent to **p7** when N_subgoals = 7.

### 4.4 Multi-stage p_k (Diffusion Policy protocol)

For sub-goal set of size `K`:

```
p_k = (# episodes where num_tasks_completed >= k) / N    for k = 1..K
```

Properties:

- `p1 >= p2 >= ... >= pK`
- Order of completion does **not** matter for p_k
- `pK` == cumulative all-task success rate

**4-task variant:** compute p1..p4 using only `KITCHEN_4_SUBGOALS`.

### 4.5 Timing (milliseconds)

| Metric | Definition |
|--------|------------|
| **Inference latency** | `perf_counter` before/after `predict_action`; sync GPU if CUDA |
| **Episode duration** | reset → episode done |
| **Task duration** | time from previous task completion (or episode start for 1st task) until current task marked complete |

Aggregate each with **mean ± std** (ddof=1) over all samples in N episodes.

**Note:** Policy may be called once per `n_action_steps` env steps, not every single timestep.

---

## 5. Reference implementation (p_k)

```python
import numpy as np

def compute_px(episodes, sub_goals):
    """Return {p1: float, p2: float, ..., pK: float}."""
    goal_set = set(sub_goals)
    counts = []
    for ep in episodes:
        completed = set(ep["completed_tasks"]) & goal_set
        counts.append(len(completed))
    K = len(sub_goals)
    return {f"p{k}": float(np.mean([c >= k for c in counts])) for k in range(1, K + 1)}


def cumulative_all_success(episodes, sub_goals):
    goal_set = set(sub_goals)
    flags = [goal_set.issubset(set(ep["completed_tasks"])) for ep in episodes]
    return float(np.mean(flags))
```

---

## 6. Cross-seed aggregation

After running seeds `[0, 42, 101]`:

For each metric (per-task success, p_k, timing means):

```
summary.mean = mean([seed_0_value, seed_42_value, seed_101_value])
summary.std  = std([seed_0_value, seed_42_value, seed_101_value], ddof=1)
```

**Important:** This is mean-of-seed-means, **not** pooling all 150 episodes into one p_k.

---

## 7. Output directory layout

```
eval_results/
  <model_name>/
    seed_0/
      eval_metrics.json
      eval_report.txt
      media/
        seed0_ep000.mp4
        ...
    seed_42/
      ...
    seed_101/
      ...
    summary.json
    summary_report.txt
```

---

## 8. Human-readable report templates

### 8.1 Per seed — `eval_report.txt`

```
========================================================================
Kitchen Eval Report | seed=0 | 50 episodes
========================================================================

Per-episode task completion order
------------------------------------------------------------------------
  ep    seed   n/7   all7  completion_order
------------------------------------------------------------------------
   0       0     4     no  kettle -> bottom burner -> slide cabinet -> microwave
   ...

Per-task success rate (fraction of episodes)
------------------------------------------------------------------------
  bottom burner     0.720 ± 0.452
  ...
  kettle            0.820 ± 0.387

Cumulative episode success (all 7 tasks completed in one episode)
------------------------------------------------------------------------
  success rate: 0.080 ± 0.274

Multi-stage p_k (>= k of 7 tasks completed)
------------------------------------------------------------------------
  p1=0.960  p2=0.880  p3=0.720  p4=0.560  p5=0.320  p6=0.160  p7=0.080
  cumulative_order_success (all 7, any order): 0.080
```

### 8.2 Cross-seed — `summary_report.txt`

```
========================================================================
Cross-seed summary | <model_name> | seeds=[0, 42, 101]
========================================================================

Per-task success rate (mean ± std across seeds)
  bottom burner             0.710 ± 0.035
  ...
  ALL 7 (cumulative episode)  0.075 ± 0.012

Multi-stage p_k (mean across seeds)
  p1: 0.950 ± 0.020
  p2: 0.870 ± 0.035
  ...
  p7: 0.075 ± 0.012
```

---

## 9. When reports appear during run

| When | Output |
|------|--------|
| During rollout | Progress bar only |
| After each seed | Full `eval_report.txt` + `eval_metrics.json` printed and saved |
| After all seeds for one model | `summary_report.txt` + `summary.json` |
| Multiple models | Repeat per model |

---

## 10. CLI interface (this repo)

```bash
MUJOCO_GL=egl python eval_kitchen.py \
  --output_root data/kitchen_eval_results \
  --seeds 0,42,101 \
  --n_episodes 50 \
  --device cuda:0 \
  --overwrite
```

Options:

- `--checkpoints`: list of `.ckpt` paths (or auto-scan `data/*/epoch=*.ckpt`)
- `--overwrite`: rerun even if `eval_metrics.json` exists
- `MUJOCO_GL=egl` (headless Linux)

---

## 11. Policy loading checklist

Different workspaces may store policy differently:

```python
import inspect

payload = torch.load(checkpoint_path, pickle_module=dill)
cfg = payload["cfg"]
cls = hydra.utils.get_class(cfg._target_)

init_params = inspect.signature(cls.__init__).parameters
if "output_dir" in init_params:
    workspace = cls(cfg, output_dir=None)
else:
    workspace = cls(cfg)
workspace.load_payload(payload)

if cfg.training.get("use_ema") and workspace.ema_model is not None:
    policy = workspace.ema_model
elif hasattr(workspace, "policy"):
    policy = workspace.policy      # e.g. BET
else:
    policy = workspace.model

policy.eval()
policy.to(device)
```

Normalizer weights must be restored from checkpoint (included in model state_dict).

---

## 12. Common pitfalls

| Issue | Fix |
|-------|-----|
| Process **Killed** (~episode 30–40) | Call `env.close()` every episode; MuJoCo leaks memory |
| `output_dir` TypeError on workspace init | Inspect `__init__` signature; omit unsupported kwargs |
| p7 != all_7 in theory | Should be identical; both = all tasks completed |
| completion_order wrong if 2 tasks finish same step | Order follows fixed task list iteration in runner |
| 280 steps != 280 inferences | Policy called every `n_action_steps` (often 8) |
| Episode ends at 280 steps with partial tasks | Expected; counts toward p1..p_k but not p7 |

---

## 13. JSON schema (top-level `eval_metrics.json`)

```json
{
  "eval_seed": 0,
  "n_episodes": 50,
  "tasks": ["bottom burner", "top burner", "..."],
  "success_rate": {
    "<task>": {"mean": 0.64, "std": 0.48, "n_samples": 50},
    "all_7_tasks": {"mean": 0.08, "std": 0.27, "n_samples": 50}
  },
  "multistage_metrics": {
    "all_7_tasks": {
      "px": {"p1": 0.96, "p2": 0.88, "p7": 0.08},
      "cumulative_order_success_rate": 0.08,
      "sub_goals": ["bottom burner", "..."]
    },
    "paper_4_tasks": {
      "px": {"p1": 0.98, "p2": 0.85, "p3": 0.72, "p4": 0.55},
      "sub_goals": ["microwave", "kettle", "bottom burner", "light switch"]
    }
  },
  "timing_ms": {
    "inference_latency": {"mean": 839.0, "std": 296.0, "n_samples": 3500},
    "episode_duration": {"mean": 33861.0, "std": 1034.0, "n_samples": 50},
    "task_duration": {
      "overall": {"mean": 8500.0, "std": 3200.0},
      "microwave": {"mean": 9200.0, "std": 4100.0, "n_samples": 32}
    }
  },
  "episodes": []
}
```

---

## 14. Source files in this repo

| File | Role |
|------|------|
| [`eval_kitchen.py`](../eval_kitchen.py) | CLI, checkpoint load, reports, cross-seed summary |
| [`diffusion_policy/env_runner/kitchen_lowdim_eval_runner.py`](../diffusion_policy/env_runner/kitchen_lowdim_eval_runner.py) | Episode loop, timing, task tracking |
| [`diffusion_policy/common/multistage_metrics.py`](../diffusion_policy/common/multistage_metrics.py) | p_k computation |
| [`compute_multistage_metrics.py`](../compute_multistage_metrics.py) | Optional standalone p_k tool |

---

## 15. Verification checklist

- [ ] Smoke test: 2 episodes, 1 seed → JSON + report + 2 videos
- [ ] Memory stable past episode 40 (no OOM)
- [ ] `p1 >= p2 >= ... >= p7`
- [ ] `p7 == success_rate.all_7_tasks.mean`
- [ ] `completion_order` length ≤ 7 per episode
- [ ] Cross-seed summary uses 3 seed values, not 150 pooled episodes

---

## References

- Diffusion Policy (Chi et al., 2023) — Section 5.3, Table 4, Appendix B.2
- Kitchen env: `diffusion_policy/env/kitchen/base.py` (`KitchenAllV0`, 7 tasks, any order)
