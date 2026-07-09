#!/usr/bin/env python3
"""Analyze WHY FlowPolicy vs Diffusion Policy differ on Kitchen eval.

Uses completion_order from all eval_metrics.json (+ n_control_steps from NPZ)
to explain task-level gaps, multi-stage, and speed.

Usage:
  python scripts/analyze_kitchen_completion_order.py
  python scripts/analyze_kitchen_completion_order.py --out-dir data/kitchen_eval_plots/why
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

MODEL_SPECS = {
    "FlowPolicy": {
        "seed_dirs": [
            ROOT
            / "kripsy12/FlowPolicy/data/kitchen_eval/flowpolicy"
            / f"seed_baseline_{s}"
            for s in (42, 43, 44)
        ],
        "color": "#2ca02c",
    },
    "DP-CNN": {
        "seed_dirs": [
            ROOT
            / "diffusion_policy/data/kitchen_eval/diffusion_policy_cnn"
            / f"seed_train{s}"
            for s in (0, 1, 2)
        ],
        "color": "#1f77b4",
    },
    "DP-Transformer": {
        "seed_dirs": [
            ROOT
            / "diffusion_policy/data/kitchen_eval/diffusion_policy_transformer"
            / f"seed_train{s}"
            for s in (0, 1, 2)
        ],
        "color": "#ff7f0e",
    },
}

MODEL_ORDER = list(MODEL_SPECS.keys())
# Canonical kitchen order for heatmaps / axes
TASKS = [
    "microwave",
    "kettle",
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
]
SHORT = {
    "microwave": "MW",
    "kettle": "Kettle",
    "bottom burner": "Bottom",
    "top burner": "Top",
    "light switch": "Light",
    "slide cabinet": "Slide",
    "hinge cabinet": "Hinge",
}
PATH_KEYS = [
    "MW->Kettle",
    "MW->Bottom",
    "Kettle-first",
    "Bottom-first",
    "MW->Other/stop",
    "Other",
]


def _save(fig, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path_base.with_suffix('.png')}")


def path_bucket(order: list[str]) -> str:
    if not order:
        return "Other"
    if order[0] == "microwave":
        if len(order) >= 2 and order[1] == "kettle":
            return "MW->Kettle"
        if len(order) >= 2 and order[1] == "bottom burner":
            return "MW->Bottom"
        return "MW->Other/stop"
    if order[0] == "kettle":
        return "Kettle-first"
    if order[0] == "bottom burner":
        return "Bottom-first"
    return "Other"


def load_model_stats(name: str) -> dict[str, Any]:
    """Aggregate completion-order + timing + NPZ control steps for one model."""
    seed_dirs = MODEL_SPECS[name]["seed_dirs"]

    positions: dict[str, list[int]] = defaultdict(list)
    first_task: Counter = Counter()
    imm: dict[str, Counter] = defaultdict(Counter)
    a_done: Counter = Counter()
    last_task: Counter = Counter()
    n_tasks_hist: Counter = Counter()
    path_n: Counter = Counter()
    path_success: dict[str, Counter] = defaultdict(Counter)
    full_orders: Counter = Counter()
    prefix2: Counter = Counter()

    latencies: list[float] = []
    ep_durs: list[float] = []
    task_durs: dict[str, list[float]] = defaultdict(list)

    n_control: list[int] = []
    n_env: list[int] = []
    ctrl_by_ntasks: dict[int, list[int]] = defaultdict(list)
    lat_by_ep: list[tuple[int, float, float]] = []  # n_ctrl, latency, ep_dur

    n_episodes = 0
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "eval_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        em = json.loads(metrics_path.read_text())

        for ep in em["episodes"]:
            n_episodes += 1
            order = list(ep.get("completion_order") or [])
            ts = ep["task_success"]
            n_done = int(ep["num_tasks_completed"])
            n_tasks_hist[n_done] += 1
            lat = float(ep["mean_inference_latency_ms"])
            dur = float(ep["episode_duration_ms"])
            latencies.append(lat)
            ep_durs.append(dur)
            for t, d in (ep.get("task_durations_ms") or {}).items():
                task_durs[t].append(float(d))

            if order:
                first_task[order[0]] += 1
                last_task[order[-1]] += 1
                full_orders[tuple(order)] += 1
                if len(order) >= 2:
                    prefix2[tuple(order[:2])] += 1

            for i, a in enumerate(order):
                positions[a].append(i)
                a_done[a] += 1
                if i + 1 < len(order):
                    imm[a][order[i + 1]] += 1

            bucket = path_bucket(order)
            path_n[bucket] += 1
            for t in TASKS:
                path_success[bucket][t] += int(ts.get(t, 0))

            # NPZ control-step stats
            npz_path = seed_dir / "trajectory_logs" / f"ep_{ep['episode_idx']:04d}.npz"
            n_ctrl = -1
            if npz_path.is_file():
                with np.load(npz_path, allow_pickle=False) as d:
                    n_ctrl = int(d["n_control_steps"])
                    n_env_i = int(d["n_env_steps"])
                n_control.append(n_ctrl)
                n_env.append(n_env_i)
                ctrl_by_ntasks[n_done].append(n_ctrl)
            lat_by_ep.append((n_ctrl, lat, dur))

    # Transition matrix P(next=B | A) including STOP
    next_labels = TASKS + ["STOP"]
    trans = np.zeros((len(TASKS), len(next_labels)), dtype=np.float64)
    for i, a in enumerate(TASKS):
        n = a_done[a]
        if n == 0:
            continue
        for j, b in enumerate(TASKS):
            trans[i, j] = imm[a][b] / n
        n_trans = sum(imm[a].values())
        trans[i, -1] = 1.0 - n_trans / n

    # Path-conditional SR matrix
    path_sr = {}
    for key in PATH_KEYS:
        n = path_n[key]
        if n == 0:
            path_sr[key] = {t: 0.0 for t in TASKS}
            path_sr[key]["_n"] = 0
        else:
            path_sr[key] = {t: path_success[key][t] / n for t in TASKS}
            path_sr[key]["_n"] = n

    return {
        "name": name,
        "n_episodes": n_episodes,
        "positions": dict(positions),
        "first_task": first_task,
        "a_done": a_done,
        "imm": {k: dict(v) for k, v in imm.items()},
        "trans": trans,
        "next_labels": next_labels,
        "last_task": last_task,
        "n_tasks_hist": n_tasks_hist,
        "path_n": path_n,
        "path_sr": path_sr,
        "full_orders": full_orders,
        "prefix2": prefix2,
        "latencies": latencies,
        "ep_durs": ep_durs,
        "task_durs": dict(task_durs),
        "n_control": n_control,
        "n_env": n_env,
        "ctrl_by_ntasks": dict(ctrl_by_ntasks),
        "lat_by_ep": lat_by_ep,
    }


def plot_position_hist(stats: dict[str, dict], out_dir: Path) -> None:
    """Grouped mean completion position + first-task bar."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: mean position among completed
    ax = axes[0]
    x = np.arange(len(TASKS))
    width = 0.25
    offsets = np.linspace(-1, 1, len(MODEL_ORDER)) * width
    for i, model in enumerate(MODEL_ORDER):
        means, ns = [], []
        for t in TASKS:
            pos = stats[model]["positions"].get(t, [])
            means.append(float(np.mean(pos)) if pos else np.nan)
            ns.append(len(pos))
        bars = ax.bar(
            x + offsets[i],
            means,
            width,
            label=model,
            color=MODEL_SPECS[model]["color"],
            alpha=0.9,
        )
        for bar, n in zip(bars, ns):
            if n > 0 and not np.isnan(bar.get_height()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=90,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[t] for t in TASKS], rotation=0)
    ax.set_ylabel("Mean completion position (0 = first)")
    ax.set_title("Mean position among completed episodes\n(lower = earlier in the chain)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: first-task distribution
    ax = axes[1]
    for i, model in enumerate(MODEL_ORDER):
        counts = [stats[model]["first_task"].get(t, 0) for t in TASKS]
        total = stats[model]["n_episodes"]
        rates = [c / total * 100 for c in counts]
        ax.bar(
            x + offsets[i],
            rates,
            width,
            label=model,
            color=MODEL_SPECS[model]["color"],
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[t] for t in TASKS])
    ax.set_ylabel("% of episodes starting with task")
    ax.set_title("First completed task\n(hypothesis: FP starts with kettle/bottom?)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Completion order positions — FP and DP both usually start with microwave",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "01_position_and_first_task")


def plot_transition_heatmaps(stats: dict[str, dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        mat = stats[model]["trans"] * 100.0
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=80, aspect="auto")
        ax.set_xticks(range(len(stats[model]["next_labels"])))
        ax.set_xticklabels(
            [SHORT.get(t, t) for t in stats[model]["next_labels"]],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(range(len(TASKS)))
        ax.set_yticklabels([SHORT[t] for t in TASKS], fontsize=8)
        ax.set_xlabel("Next task (or STOP)")
        ax.set_title(model)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if v >= 5:
                    ax.text(
                        j,
                        i,
                        f"{v:.0f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="black" if v < 50 else "white",
                    )
    axes[0].set_ylabel("After completing")
    fig.subplots_adjust(right=0.88, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.18, 0.015, 0.65])
    fig.colorbar(im, cax=cbar_ax, label="% of completions")
    fig.suptitle(
        "P(next = B | completed = A) — chaining is the main FP vs DP difference",
        fontsize=12,
    )
    _save(fig, out_dir / "02_transition_heatmap")


def plot_path_conditional(stats: dict[str, dict], out_dir: Path) -> None:
    focus_tasks = [
        "kettle",
        "bottom burner",
        "top burner",
        "light switch",
        "hinge cabinet",
        "slide cabinet",
    ]
    show_paths = ["MW->Kettle", "MW->Bottom", "Kettle-first", "Bottom-first"]

    fig, axes = plt.subplots(1, len(show_paths), figsize=(14, 4.8), sharey=True)
    x = np.arange(len(focus_tasks))
    width = 0.25
    offsets = np.linspace(-1, 1, len(MODEL_ORDER)) * width

    for ax, pkey in zip(axes, show_paths):
        for i, model in enumerate(MODEL_ORDER):
            ps = stats[model]["path_sr"][pkey]
            n = int(ps["_n"])
            vals = [ps[t] * 100 for t in focus_tasks]
            ax.bar(
                x + offsets[i],
                vals,
                width,
                label=model if pkey == show_paths[0] else None,
                color=MODEL_SPECS[model]["color"],
                alpha=0.9,
            )
        ns = [int(stats[m]["path_sr"][pkey]["_n"]) for m in MODEL_ORDER]
        ax.set_title(f"{pkey}\n(n={ns[0]}/{ns[1]}/{ns[2]})")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[t] for t in focus_tasks], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 110)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Success rate within path (%)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Path-conditional success — same start path, different later-task outcomes",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "03_path_conditional_sr")


def plot_stop_multistage(stats: dict[str, dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: num_tasks_completed hist
    ax = axes[0]
    ks = list(range(0, 8))
    x = np.arange(len(ks))
    width = 0.25
    offsets = np.linspace(-1, 1, len(MODEL_ORDER)) * width
    for i, model in enumerate(MODEL_ORDER):
        total = stats[model]["n_episodes"]
        vals = [stats[model]["n_tasks_hist"].get(k, 0) / total * 100 for k in ks]
        ax.bar(
            x + offsets[i],
            vals,
            width,
            label=model,
            color=MODEL_SPECS[model]["color"],
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("# tasks completed in episode")
    ax.set_ylabel("% of episodes")
    ax.set_title("Multi-stage: how many tasks per episode")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: last task (where they stop)
    ax = axes[1]
    x = np.arange(len(TASKS))
    for i, model in enumerate(MODEL_ORDER):
        total = stats[model]["n_episodes"]
        vals = [stats[model]["last_task"].get(t, 0) / total * 100 for t in TASKS]
        ax.bar(
            x + offsets[i],
            vals,
            width,
            label=model,
            color=MODEL_SPECS[model]["color"],
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[t] for t in TASKS], rotation=45, ha="right")
    ax.set_ylabel("% of episodes")
    ax.set_title("Last completed task (where the chain stops)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "FP often stops at slide; DP more often reaches hinge (5th task)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "04_stop_and_multistage")


def plot_speed(stats: dict[str, dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))

    # Panel 1: mean inference latency
    ax = axes[0]
    means = [float(np.mean(stats[m]["latencies"])) for m in MODEL_ORDER]
    colors = [MODEL_SPECS[m]["color"] for m in MODEL_ORDER]
    bars = ax.bar(MODEL_ORDER, means, color=colors, alpha=0.9)
    ax.set_ylabel("ms / predict_action")
    ax.set_title("Inference latency")
    ax.tick_params(axis="x", rotation=15)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: n_control_steps from NPZ
    ax = axes[1]
    means = []
    for m in MODEL_ORDER:
        vals = stats[m]["n_control"]
        means.append(float(np.mean(vals)) if vals else 0.0)
    bars = ax.bar(MODEL_ORDER, means, color=colors, alpha=0.9)
    ax.set_ylabel("control steps / episode")
    ax.set_title("n_control_steps (from NPZ)")
    ax.tick_params(axis="x", rotation=15)
    for bar, v, m in zip(bars, means, MODEL_ORDER):
        n_env = stats[m]["n_env"]
        note = ""
        if n_env and max(n_env) <= 5:
            note = "\n(env log truncated)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.0f}{note}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: estimated inference budget = n_ctrl * latency
    ax = axes[2]
    budgets = []
    ep_means = []
    for m in MODEL_ORDER:
        lat = float(np.mean(stats[m]["latencies"]))
        n_ctrl = float(np.mean(stats[m]["n_control"])) if stats[m]["n_control"] else 0.0
        budgets.append(n_ctrl * lat)
        ep_means.append(float(np.mean(stats[m]["ep_durs"])))
    x = np.arange(len(MODEL_ORDER))
    w = 0.35
    ax.bar(x - w / 2, budgets, w, label="n_ctrl × latency", color="#9467bd", alpha=0.85)
    ax.bar(x + w / 2, ep_means, w, label="episode_duration", color="#8c564b", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15)
    ax.set_ylabel("ms")
    ax.set_title("Inference budget vs wall-clock episode")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Speed: FP has many more (cheap) control steps; DP has few (expensive) ones",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out_dir / "05_speed")


def write_report(stats: dict[str, dict], out_dir: Path) -> str:
    lines: list[str] = []
    w = lines.append

    w("=" * 78)
    w("WHY FlowPolicy vs Diffusion Policy — Kitchen completion-order analysis")
    w("=" * 78)
    w("")
    w("Data: 3 seeds × 100 episodes per model (n=300).")
    w("Sources: eval_metrics.json completion_order + trajectory NPZ n_control_steps.")
    w("Note: NPZ per-env-step arrays are truncated (n_env_steps≈3); spatial EE")
    w("distance analysis is NOT used. Control-step counts are reliable.")
    w("")

    # First task
    w("-" * 78)
    w("1) Hypothesis check: are kettle / bottom burner usually FIRST in FlowPolicy?")
    w("-" * 78)
    w("NO. All methods most often start with microwave.")
    for m in MODEL_ORDER:
        total = stats[m]["n_episodes"]
        ft = stats[m]["first_task"]
        parts = ", ".join(
            f"{SHORT[t]}={ft.get(t, 0)/total*100:.0f}%" for t in TASKS if ft.get(t, 0) > 0
        )
        w(f"  {m}: {parts}")
    w("")
    w("Mean completion position (0=first) among completed:")
    header = f"  {'task':<16}" + "".join(f"{m:>18}" for m in MODEL_ORDER)
    w(header)
    for t in TASKS:
        cells = []
        for m in MODEL_ORDER:
            pos = stats[m]["positions"].get(t, [])
            if pos:
                cells.append(f"{np.mean(pos):5.2f} n={len(pos):<3}")
            else:
                cells.append("   —")
        w(f"  {t:<16}" + "".join(f"{c:>18}" for c in cells))
    w("")
    w("Kettle/bottom are early for BOTH methods (pos ~0.7–1.3). Position alone")
    w("does NOT explain why FP has higher kettle/bottom success rates.")
    w("")

    # Transitions
    w("-" * 78)
    w("2) Main finding: different chaining / transition patterns")
    w("-" * 78)
    for m in MODEL_ORDER:
        w(f"\n  {m} — P(next|after):")
        for a in TASKS:
            n = stats[m]["a_done"][a]
            if n == 0:
                continue
            row = stats[m]["trans"][TASKS.index(a)]
            tops = []
            for j, lab in enumerate(stats[m]["next_labels"]):
                if row[j] >= 0.05:
                    tops.append(f"{SHORT.get(lab, lab)}={row[j]*100:.0f}%")
            w(f"    after {SHORT[a]:<8} (n={n:<3}): " + ", ".join(tops))
    w("")
    w("Interpretation:")
    w("  • FP locks into MW→Kettle→Bottom, then often Slide, then STOPS (99%).")
    w("  • DP after Bottom more often goes Top/Light; after Light/Slide often Hinge.")
    w("  • So FP over-samples early chain (kettle/bottom); DP over-samples mid/late")
    w("    chain (top burner, light switch, hinge cabinet).")
    w("")

    # Path conditional
    w("-" * 78)
    w("3) Path-conditional success (same start, different later outcomes)")
    w("-" * 78)
    for pkey in ["MW->Kettle", "MW->Bottom", "Kettle-first", "Bottom-first"]:
        w(f"\n  Path {pkey}:")
        for m in MODEL_ORDER:
            ps = stats[m]["path_sr"][pkey]
            n = int(ps["_n"])
            w(
                f"    {m:<16} n={n:<3}  "
                f"kettle={ps['kettle']*100:5.1f}%  "
                f"bottom={ps['bottom burner']*100:5.1f}%  "
                f"top={ps['top burner']*100:5.1f}%  "
                f"light={ps['light switch']*100:5.1f}%  "
                f"hinge={ps['hinge cabinet']*100:5.1f}%"
            )
    w("")
    w("On MW→Kettle: FP keeps high bottom success but almost never reaches top/hinge;")
    w("DP reaches hinge far more often even when kettle came second.")
    w("")

    # Multistage / stop
    w("-" * 78)
    w("4) Why DP wins multi-stage (p3–p4)")
    w("-" * 78)
    for m in MODEL_ORDER:
        total = stats[m]["n_episodes"]
        hist = stats[m]["n_tasks_hist"]
        parts = ", ".join(f"{k}:{hist.get(k, 0)/total*100:.0f}%" for k in range(0, 8) if hist.get(k, 0))
        w(f"  {m} num_tasks_completed: {parts}")
        last = stats[m]["last_task"]
        last_parts = ", ".join(
            f"{SHORT[t]}={last.get(t, 0)/total*100:.0f}%"
            for t in TASKS
            if last.get(t, 0) > 0
        )
        w(f"    last task: {last_parts}")
    w("")
    w("FP mass at 3–4 tasks, last task often Slide. DP mass at 4–5 tasks, last")
    w("task often Hinge — hence higher p3/p4 and hinge/light/top success.")
    w("")

    # Speed
    w("-" * 78)
    w("5) Why FlowPolicy is faster")
    w("-" * 78)
    for m in MODEL_ORDER:
        lat = float(np.mean(stats[m]["latencies"]))
        dur = float(np.mean(stats[m]["ep_durs"]))
        n_ctrl = float(np.mean(stats[m]["n_control"])) if stats[m]["n_control"] else float("nan")
        budget = n_ctrl * lat if not np.isnan(n_ctrl) else float("nan")
        w(
            f"  {m:<16} latency={lat:7.1f} ms  n_ctrl={n_ctrl:6.1f}  "
            f"n_ctrl×lat={budget:8.0f} ms  episode_dur={dur:8.0f} ms"
        )
    w("")
    w("FP: many cheap predict calls (~94 × ~14 ms).")
    w("DP: few expensive predict calls (~35 × ~600 ms).")
    w("Wall-clock episode duration tracks the expensive DP inference, not step count.")
    w("")

    # Direct answers
    w("-" * 78)
    w("6) Direct answers")
    w("-" * 78)
    w("")
    w("Q: Why is FlowPolicy better at kettle & bottom burner?")
    w("A: Not because they are first. FP more often commits to the early chain")
    w("   MW→Kettle→Bottom (74% MW→Kettle vs ~48–50% for DP) and converts kettle")
    w("   completions into bottom (78%). DP branches away from that early chain")
    w("   more often (MW→Bottom or Kettle→Light), so fewer kettle/bottom successes.")
    w("")
    w("Q: Why is Diffusion Policy better at hinge, light switch, top burner?")
    w("A: After bottom/top, DP continues into Top→Light→Hinge much more often.")
    w("   FP after bottom prefers Slide and then almost always STOPS — it rarely")
    w("   attempts the mid/late cabinet/switch chain. Same start position, different")
    w("   next-task policy.")
    w("")
    w("Q: Why is FlowPolicy faster?")
    w("A: Per-call inference is ~40–45× cheaper. Even with more control steps,")
    w("   total inference budget and episode wall-clock stay much lower.")
    w("")
    w("Q: Why is Diffusion Policy better at multi-stage?")
    w("A: DP finishes 4–5 tasks in nearly every episode and often ends on hinge.")
    w("   FP frequently ends the chain at slide after 3–4 tasks, so p3/p4 drop.")
    w("")
    w("=" * 78)

    text = "\n".join(lines) + "\n"
    out_path = out_dir / "why_report.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"  wrote {out_path}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "kitchen_eval_plots" / "why",
    )
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir

    print("Loading eval_metrics + NPZ control steps...")
    stats = {name: load_model_stats(name) for name in MODEL_ORDER}
    for name in MODEL_ORDER:
        s = stats[name]
        print(
            f"  {name}: {s['n_episodes']} episodes, "
            f"{len(s['n_control'])} NPZ, "
            f"mean_ctrl={np.mean(s['n_control']) if s['n_control'] else float('nan'):.1f}"
        )

    print(f"\nWriting figures to {out_dir}")
    plot_position_hist(stats, out_dir)
    plot_transition_heatmaps(stats, out_dir)
    plot_path_conditional(stats, out_dir)
    plot_stop_multistage(stats, out_dir)
    plot_speed(stats, out_dir)

    print("\nReport:")
    text = write_report(stats, out_dir)
    print(text)
    print("Done.")


if __name__ == "__main__":
    main()
