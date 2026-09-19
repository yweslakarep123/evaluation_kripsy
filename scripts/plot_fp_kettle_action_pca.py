#!/usr/bin/env python3
"""PCA of FlowPolicy kettle-phase actions vs demonstration actions.

Hypothesis check: at higher NFE, predicted actions during the kettle sub-task
drift farther from the dominant demonstration action pattern.

Data (post-hoc, no re-eval):
  - Predictions: kitchen_eval_nfe100 FlowPolicy trajectory_logs/*.npz
    (executed_action in kettle window from task_durations_ms)
  - Demonstrations: data/kitchen/{actions,observations,existence}_seq.npy
    (kettle window from object-goal completions, same BONUS_THRESH as env)

Outputs:
  data/kitchen_eval_plots/nfe100/kettle_action_pca/
    01_kettle_action_pca.{png,pdf}
    kettle_action_pca_distances.csv
    README.txt

Usage:
  /home/daffa/miniforge3/envs/flowpolicy-kitchen/bin/python \\
      scripts/plot_fp_kettle_action_pca.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FP_ROOT = ROOT / "kripsy12/FlowPolicy/data/kitchen_eval_nfe100/flowpolicy"
DEMO_DIR = ROOT / "kripsy12/FlowPolicy/data/kitchen"
OUT_DIR = ROOT / "data/kitchen_eval_plots/nfe100/kettle_action_pca"

SEEDS = (42, 43, 44)
NFES = (1, 8, 32, 100)

# From flow_policy_3d/env/kitchen/base.py
QP_DIM = 9
OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3
TASK_ORDER = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]

# Scatter readability
MAX_POINTS_PLOT = {
    "demo": 8000,
    1: 6000,
    8: 6000,
    32: 6000,
    100: 6000,
}
RNG = np.random.default_rng(0)


def _save(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path_base.with_suffix('.png')}")
    print(f"  wrote {path_base.with_suffix('.pdf')}")


def _confidence_ellipse(
    ax: plt.Axes,
    pts: np.ndarray,
    n_std: float,
    **kwargs: Any,
) -> Optional[Ellipse]:
    """Draw covariance ellipse covering n_std mahalanobis radius."""
    if pts.ndim != 2 or pts.shape[0] < 3:
        return None
    cov = np.cov(pts, rowvar=False)
    if not np.isfinite(cov).all():
        return None
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.maximum(vals, 1e-12)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    mean = pts.mean(axis=0)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)
    return ell


def completion_steps_from_obs(obs: np.ndarray) -> dict[str, int]:
    """First env step at which each task crosses BONUS_THRESH (demo-style)."""
    obj = obs[:, QP_DIM : QP_DIM + 21]
    done: dict[str, int] = {}
    active = set(TASK_ORDER)
    for t in range(len(obs)):
        newly = []
        for name in list(active):
            idxs = OBS_ELEMENT_INDICES[name]
            dist = float(
                np.linalg.norm(obj[t, idxs - QP_DIM] - OBS_ELEMENT_GOALS[name])
            )
            if dist < BONUS_THRESH:
                newly.append(name)
        for name in newly:
            done[name] = t
            active.discard(name)
        if not active:
            break
    return done


def kettle_window_from_completions(completions: dict[str, int]) -> Optional[tuple[int, int]]:
    if "kettle" not in completions:
        return None
    end = int(completions["kettle"]) + 1  # inclusive end → slice end exclusive +1
    prior = [completions[t] for t in completions if t != "kettle" and completions[t] < completions["kettle"]]
    start = (max(prior) + 1) if prior else 0
    if end <= start:
        return None
    return start, end


def kettle_window_from_eval_durations(
    order: list[str], durations_ms: dict[str, float], n_env_steps: int, episode_ms: float
) -> Optional[tuple[int, int]]:
    """Map segment durations to env-step [start, end) for kettle."""
    if "kettle" not in order or episode_ms <= 0 or n_env_steps <= 0:
        return None
    ms_per = episode_ms / float(n_env_steps)
    cum = 0.0
    start = end = None
    for task in order:
        seg = float(durations_ms[task])
        seg_start = int(round(cum / ms_per))
        cum += seg
        seg_end = int(round(cum / ms_per))
        if task == "kettle":
            start, end = seg_start, seg_end
            break
    if start is None or end is None:
        return None
    start = max(0, min(start, n_env_steps))
    end = max(0, min(end, n_env_steps))
    if end <= start:
        return None
    return start, end


def load_demo_kettle_actions() -> np.ndarray:
    """Collect demo action vectors in kettle segments. Array layout (T, N, D)."""
    actions = np.load(DEMO_DIR / "actions_seq.npy")
    obs = np.load(DEMO_DIR / "observations_seq.npy")
    mask = np.load(DEMO_DIR / "existence_mask.npy")
    assert actions.ndim == 3 and actions.shape[0] == obs.shape[0]
    n_demos = actions.shape[1]
    chunks: list[np.ndarray] = []
    n_with_kettle = 0
    for i in range(n_demos):
        valid = mask[:, i] > 0
        if not np.any(valid):
            continue
        a = actions[valid, i, :].astype(np.float64)
        o = obs[valid, i, :].astype(np.float64)
        comps = completion_steps_from_obs(o)
        win = kettle_window_from_completions(comps)
        if win is None:
            continue
        n_with_kettle += 1
        s, e = win
        chunks.append(a[s:e])
    if not chunks:
        raise RuntimeError("No demo kettle segments found in actions_seq.npy")
    out = np.concatenate(chunks, axis=0)
    print(
        f"  demos with kettle completion: {n_with_kettle}/{n_demos}; "
        f"action vectors: {len(out)}"
    )
    return out


def load_policy_kettle_actions(nfe: int) -> np.ndarray:
    """Collect executed (predicted) actions in kettle windows for one NFE."""
    chunks: list[np.ndarray] = []
    n_eps = 0
    n_used = 0
    for seed in SEEDS:
        run = FP_ROOT / f"seed_baseline_{seed}_nfe{nfe}_sseed0"
        metrics = json.loads((run / "eval_metrics.json").read_text())
        for ep in metrics["episodes"]:
            n_eps += 1
            order = ep.get("completion_order") or []
            if "kettle" not in order:
                continue
            npz_path = run / "trajectory_logs" / f"ep_{int(ep['episode_idx']):04d}.npz"
            if not npz_path.is_file():
                continue
            z = np.load(npz_path)
            acts = np.asarray(z["executed_action"], dtype=np.float64)
            n_env = int(z["n_env_steps"])
            win = kettle_window_from_eval_durations(
                order,
                ep["task_durations_ms"],
                n_env,
                float(ep["episode_duration_ms"]),
            )
            if win is None:
                continue
            s, e = win
            seg = acts[s:e]
            if len(seg) == 0:
                continue
            chunks.append(seg)
            n_used += 1
    if not chunks:
        raise RuntimeError(f"No kettle action segments for FlowPolicy NFE={nfe}")
    out = np.concatenate(chunks, axis=0)
    print(
        f"  NFE={nfe}: kettle-success episodes used {n_used}/{n_eps}; "
        f"action vectors: {len(out)}"
    )
    return out


def subsample(arr: np.ndarray, k: int) -> np.ndarray:
    if len(arr) <= k:
        return arr
    idx = RNG.choice(len(arr), size=k, replace=False)
    return arr[idx]


def mean_dist_to_centroid(pts: np.ndarray, centroid: np.ndarray) -> float:
    d = np.linalg.norm(pts - centroid[None, :], axis=1)
    return float(np.mean(d))


def plot_pca(
    demo_xy: np.ndarray,
    nfe_xy: dict[int, np.ndarray],
    distances: dict[int, float],
    out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    # Demo background
    ax.scatter(
        demo_xy[:, 0],
        demo_xy[:, 1],
        s=8,
        c="#9e9e9e",
        alpha=0.18,
        linewidths=0,
        label="Demonstration",
        zorder=1,
    )
    _confidence_ellipse(
        ax,
        demo_xy,
        n_std=2.0,
        facecolor="none",
        edgecolor="#616161",
        linewidth=1.2,
        linestyle="--",
        alpha=0.9,
        zorder=2,
    )

    cmap = plt.get_cmap("viridis")
    colors = {nfe: cmap(i / max(len(NFES) - 1, 1)) for i, nfe in enumerate(NFES)}

    for nfe in NFES:
        pts = nfe_xy[nfe]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=10,
            c=[colors[nfe]],
            alpha=0.35,
            linewidths=0,
            label=f"NFE={nfe} (d̄={distances[nfe]:.3f})",
            zorder=3,
        )
        _confidence_ellipse(
            ax,
            pts,
            n_std=1.0,
            facecolor="none",
            edgecolor=colors[nfe],
            linewidth=1.8,
            alpha=0.95,
            zorder=4,
        )
        _confidence_ellipse(
            ax,
            pts,
            n_std=2.0,
            facecolor="none",
            edgecolor=colors[nfe],
            linewidth=1.0,
            linestyle=":",
            alpha=0.7,
            zorder=4,
        )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title(
        "FlowPolicy kettle-phase actions in demo-fitted PCA space\n"
        "Demonstration (gray) vs predicted actions by NFE",
        fontsize=12,
    )
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    _save(fig, out_base)


def write_csv(distances: dict[int, float], n_pts: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "n_action_vectors",
                "mean_euclidean_dist_to_demo_centroid",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "group": "demonstration",
                "n_action_vectors": n_pts["demo"],
                "mean_euclidean_dist_to_demo_centroid": 0.0,
            }
        )
        for nfe in NFES:
            w.writerow(
                {
                    "group": f"NFE={nfe}",
                    "n_action_vectors": n_pts[nfe],
                    "mean_euclidean_dist_to_demo_centroid": f"{distances[nfe]:.6f}",
                }
            )
    print(f"  wrote {path}")


def write_readme(
    path: Path,
    distances: dict[int, float],
    n_pts: dict[str, int],
    var_ratio: np.ndarray,
) -> None:
    lines = [
        "Kettle action PCA — FlowPolicy NFE hypothesis",
        "=============================================",
        "",
        "Figure: 01_kettle_action_pca.{png,pdf}",
        "Table:  kettle_action_pca_distances.csv",
        "",
        "Method",
        "------",
        "- Demo actions: kettle windows from data/kitchen/*_seq.npy,",
        "  segmented by object-goal completions (BONUS_THRESH=0.3).",
        "- Policy actions: executed_action in kettle windows from",
        "  kitchen_eval_nfe100 FlowPolicy trajectory_logs (seeds 42/43/44),",
        "  only episodes that completed kettle; window from task_durations_ms.",
        "- PCA(2) fit on demo kettle actions only, then transform all groups.",
        "- Ellipses: solid = 1σ, dotted = 2σ (per NFE); dashed gray = demo 2σ.",
        "",
        f"PCA variance explained: PC1={var_ratio[0]:.1%}, PC2={var_ratio[1]:.1%},",
        f"  cumulative={var_ratio.sum():.1%}",
        "",
        "Mean Euclidean distance to demo centroid (action space, 9-D)",
        "-----------------------------------------------------------",
    ]
    for nfe in NFES:
        lines.append(
            f"  NFE={nfe:<3}  n={n_pts[nfe]:<7}  mean_dist={distances[nfe]:.6f}"
        )
    trend = "increasing" if distances[8] > distances[1] else "NOT increasing"
    lines += [
        "",
        f"NFE1→NFE8 distance trend: {trend}",
        "  (increasing supports drift-from-demo hypothesis;",
        "   decreasing/stable weakens it)",
        "",
        "Limitation: only kettle-SUCCESS episodes contribute policy points;",
        "path-selection failures (never attempting kettle) are excluded by design.",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def main() -> None:
    print("Loading demonstration kettle-phase actions…")
    demo_actions = load_demo_kettle_actions()

    print("Loading FlowPolicy kettle-phase actions per NFE…")
    nfe_actions: dict[int, np.ndarray] = {}
    for nfe in NFES:
        nfe_actions[nfe] = load_policy_kettle_actions(nfe)

    print("Fitting PCA on demonstration actions…")
    pca = PCA(n_components=2, random_state=0)
    pca.fit(demo_actions)
    print(
        f"  explained variance: {pca.explained_variance_ratio_} "
        f"(sum={pca.explained_variance_ratio_.sum():.3f})"
    )

    demo_centroid = demo_actions.mean(axis=0)
    distances = {
        nfe: mean_dist_to_centroid(nfe_actions[nfe], demo_centroid) for nfe in NFES
    }
    n_pts: dict[str, int] = {"demo": len(demo_actions)}
    n_pts.update({nfe: len(nfe_actions[nfe]) for nfe in NFES})

    print("\n=== Mean Euclidean distance to demo centroid (9-D action) ===")
    for nfe in NFES:
        print(f"  NFE={nfe:<3}  n={n_pts[nfe]:<7}  mean_dist={distances[nfe]:.6f}")
    if distances[8] > distances[1]:
        print("  → NFE1→8 distance INCREASES (supports drift hypothesis)")
    else:
        print("  → NFE1→8 distance does NOT increase (weakens drift hypothesis)")

    demo_xy = pca.transform(subsample(demo_actions, MAX_POINTS_PLOT["demo"]))
    nfe_xy = {
        nfe: pca.transform(subsample(nfe_actions[nfe], MAX_POINTS_PLOT[nfe]))
        for nfe in NFES
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(distances, n_pts, OUT_DIR / "kettle_action_pca_distances.csv")
    write_readme(
        OUT_DIR / "README.txt",
        distances,
        n_pts,
        pca.explained_variance_ratio_,
    )
    print("Plotting…")
    plot_pca(demo_xy, nfe_xy, distances, OUT_DIR / "01_kettle_action_pca")
    print("Done.")


if __name__ == "__main__":
    main()
