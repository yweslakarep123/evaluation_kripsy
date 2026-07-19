#!/usr/bin/env python3
"""Analyze full Kitchen NFE100 eval (3 seeds × NFE grid × 3 models).

Reads:
  diffusion_policy/data/kitchen_eval_nfe100/<model>/seed_*_nfe*_sseed*/
  kripsy12/FlowPolicy/data/kitchen_eval_nfe100/flowpolicy/seed_*_nfe*_sseed*/

Writes under data/kitchen_eval_plots/nfe100/:
  report.txt, summary.csv, success_vs_nfe.png, latency_vs_nfe.png,
  pareto.png, pareto_per_seed.png, pk_degradation.png/.csv/.txt,
  utopia_distance.csv, utopia_distance.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIR_RE = re.compile(r"^seed_(?P<seed>.+)_nfe(?P<nfe>\d+)_sseed(?P<sseed>\d+)$")
TASKS = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]
LABEL = {
    "diffusion_policy_cnn": "DP-CNN",
    "diffusion_policy_transformer": "DP-Transformer",
    "flowpolicy": "FlowPolicy",
}
COLOR = {
    "diffusion_policy_cnn": "#1f77b4",
    "diffusion_policy_transformer": "#ff7f0e",
    "flowpolicy": "#2ca02c",
}
MARKER = {
    "diffusion_policy_cnn": "o",
    "diffusion_policy_transformer": "s",
    "flowpolicy": "D",
}
NFE_MARKER = {1: "o", 8: "s", 32: "^", 100: "D"}
MODEL_ORDER = (
    "flowpolicy",
    "diffusion_policy_cnn",
    "diffusion_policy_transformer",
)
# Operating points shown on focused Pareto / degradation plots
HIGHLIGHT_OPS = {
    ("flowpolicy", 1),
    ("flowpolicy", 8),
    ("diffusion_policy_cnn", 100),
    ("diffusion_policy_transformer", 100),
}
HIGHLIGHT_NFES = (1, 8, 100)


def _is_highlight(model: str, nfe: int) -> bool:
    return (model, int(nfe)) in HIGHLIGHT_OPS


def _filter_highlight_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in runs if _is_highlight(r["model"], r["nfe"])]


def _filter_highlight_agg(agg: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in agg if _is_highlight(r["model"], int(r["nfe"]))]


def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _px(m: Dict[str, Any], k: int) -> Optional[float]:
    v = m.get("multistage_metrics", {}).get("all_7_tasks", {}).get("px", {}).get(f"p{k}")
    return v.get("mean") if isinstance(v, dict) else v


def _px_std(m: Dict[str, Any], k: int) -> Optional[float]:
    v = m.get("multistage_metrics", {}).get("all_7_tasks", {}).get("px", {}).get(f"p{k}")
    return v.get("std") if isinstance(v, dict) else None


def _lat(m: Dict[str, Any]) -> Optional[float]:
    return m.get("timing_ms", {}).get("inference_latency", {}).get("mean")


def _lat_std(m: Dict[str, Any]) -> Optional[float]:
    return m.get("timing_ms", {}).get("inference_latency", {}).get("std")


def _mean_tasks(m: Dict[str, Any]) -> float:
    eps = m.get("episodes") or []
    if not eps:
        return float("nan")
    return float(np.mean([e.get("num_tasks_completed", 0) for e in eps]))


def _mean_tasks_std(m: Dict[str, Any]) -> float:
    eps = m.get("episodes") or []
    if len(eps) < 2:
        return 0.0
    return float(
        np.std(
            [e.get("num_tasks_completed", 0) for e in eps],
            ddof=1,
        )
    )


def _success_rate_p14(run: Dict[str, Any]) -> Optional[float]:
    """Mean of multistage p1..p4 (success-rate style quality metric)."""
    vals = [run.get(f"p{k}") for k in (1, 2, 3, 4)]
    if any(v is None for v in vals):
        return None
    return float(sum(vals) / 4.0)


def pareto_front(
    points: List[Dict[str, Any]],
    *,
    x_key: str = "latency_ms",
    y_key: str = "success_rate_p14",
) -> List[Dict[str, Any]]:
    """Non-dominated set: minimize x (latency), maximize y (success rate)."""
    valid = [
        p
        for p in points
        if p.get(x_key) is not None
        and p.get(y_key) is not None
        and np.isfinite(p[x_key])
        and np.isfinite(p[y_key])
    ]
    front: List[Dict[str, Any]] = []
    for a in valid:
        dominated = False
        for b in valid:
            if a is b:
                continue
            # b dominates a if b is no worse on both and better on at least one
            if (
                b[x_key] <= a[x_key]
                and b[y_key] >= a[y_key]
                and (b[x_key] < a[x_key] or b[y_key] > a[y_key])
            ):
                dominated = True
                break
        if not dominated:
            front.append(a)
    front.sort(key=lambda p: (p[x_key], -p[y_key]))
    return front


def discover(root_dp: Path, root_fp: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []

    def add(model: str, d: Path) -> None:
        m = DIR_RE.match(d.name)
        if not m:
            return
        path = d / "eval_metrics.json"
        if not path.is_file():
            return
        metrics = json.loads(path.read_text())
        runs.append(
            {
                "model": model,
                "train_seed": m.group("seed"),
                "nfe": int(m.group("nfe")),
                "sseed": int(m.group("sseed")),
                "mean_tasks": _mean_tasks(metrics),
                "mean_tasks_episode_std": _mean_tasks_std(metrics),
                "p1": _px(metrics, 1),
                "p1_episode_std": _px_std(metrics, 1),
                "p2": _px(metrics, 2),
                "p2_episode_std": _px_std(metrics, 2),
                "p3": _px(metrics, 3),
                "p3_episode_std": _px_std(metrics, 3),
                "p4": _px(metrics, 4),
                "p4_episode_std": _px_std(metrics, 4),
                "latency_ms": _lat(metrics),
                "latency_episode_std": _lat_std(metrics),
                "n_episodes": metrics.get("n_episodes"),
                "success_rate": {
                    t: metrics.get("success_rate", {}).get(t, {}).get("mean")
                    for t in TASKS
                },
                "path": str(path),
            }
        )

    if root_dp.exists():
        for model_dir in sorted(root_dp.iterdir()):
            if not model_dir.is_dir():
                continue
            for d in sorted(model_dir.iterdir()):
                if d.is_dir():
                    add(model_dir.name, d)

    if root_fp.exists():
        for d in sorted(root_fp.iterdir()):
            if d.is_dir():
                add("flowpolicy", d)

    return sorted(runs, key=lambda r: (r["model"], r["nfe"], r["train_seed"]))


def aggregate(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean±std across training seeds for each (model, nfe)."""
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped[(r["model"], r["nfe"])].append(r)

    rows = []
    for (model, nfe), group in sorted(grouped.items()):
        # unique by train_seed
        by_seed = {g["train_seed"]: g for g in group}
        group = list(by_seed.values())
        mt_m, mt_s = _mean_std([g["mean_tasks"] for g in group])
        p1_m, p1_s = _mean_std([g["p1"] for g in group if g["p1"] is not None])
        p2_m, p2_s = _mean_std([g["p2"] for g in group if g["p2"] is not None])
        p3_m, p3_s = _mean_std([g["p3"] for g in group if g["p3"] is not None])
        p4_m, p4_s = _mean_std([g["p4"] for g in group if g["p4"] is not None])
        lat_m, lat_s = _mean_std(
            [g["latency_ms"] for g in group if g["latency_ms"] is not None]
        )
        sr_vals = [
            v
            for g in group
            for v in [_success_rate_p14(g)]
            if v is not None
        ]
        sr_m, sr_s = _mean_std(sr_vals)
        row: Dict[str, Any] = {
            "model": model,
            "nfe": nfe,
            "n_seeds": len(group),
            "seeds": ",".join(sorted(by_seed.keys())),
            "mean_tasks": mt_m,
            "mean_tasks_std": mt_s,
            "p1": p1_m,
            "p1_std": p1_s,
            "p2": p2_m,
            "p2_std": p2_s,
            "p3": p3_m,
            "p3_std": p3_s,
            "p4": p4_m,
            "p4_std": p4_s,
            "success_rate_p14": sr_m,
            "success_rate_p14_std": sr_s,
            "latency_ms": lat_m,
            "latency_ms_std": lat_s,
        }
        for t in TASKS:
            vals = [
                g["success_rate"][t]
                for g in group
                if g["success_rate"].get(t) is not None
            ]
            m, s = _mean_std([float(v) for v in vals])
            row[f"sr_{t}"] = m
            row[f"sr_{t}_std"] = s
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def compute_utopia_distances(
    points: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    """Normalize latency by L_max; distance to ideal (0, 1) in normalized space.

    x_tilde = L / L_max,  y = success_rate_p14,
    d = sqrt(x_tilde^2 + (1 - y)^2)
    """
    lats = [float(p["latency_ms"]) for p in points]
    l_max = float(max(lats)) if lats else 1.0
    if l_max <= 0:
        l_max = 1.0
    rows: List[Dict[str, Any]] = []
    for p in points:
        lat = float(p["latency_ms"])
        y = float(p["success_rate_p14"])
        x_tilde = lat / l_max
        d = float(np.sqrt(x_tilde**2 + (1.0 - y) ** 2))
        rows.append(
            {
                "model": p["model"],
                "train_seed": p["train_seed"],
                "nfe": int(p["nfe"]),
                "sseed": p.get("sseed", 0),
                "latency_ms": lat,
                "success_rate_p14": y,
                "x_tilde": x_tilde,
                "y": y,
                "distance": d,
            }
        )
    rows.sort(key=lambda r: (r["distance"], r["model"], r["nfe"], str(r["train_seed"])))
    return l_max, rows


def write_utopia_logs(
    out_dir: Path, l_max: float, dist_rows: List[Dict[str, Any]], agg: List[Dict[str, Any]]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "utopia_distance.csv", dist_rows)

    # Mean distance per (model, NFE) using mean latency/SR (same formula)
    mean_rows: List[Dict[str, Any]] = []
    for r in agg:
        if r.get("latency_ms") is None or r.get("success_rate_p14") is None:
            continue
        lat = float(r["latency_ms"])
        y = float(r["success_rate_p14"])
        x_tilde = lat / l_max
        d = float(np.sqrt(x_tilde**2 + (1.0 - y) ** 2))
        mean_rows.append(
            {
                "model": r["model"],
                "nfe": int(r["nfe"]),
                "latency_ms": lat,
                "success_rate_p14": y,
                "x_tilde": x_tilde,
                "distance": d,
            }
        )
    mean_rows.sort(key=lambda r: r["distance"])

    lines = [
        "Utopia distance to Ideal (0, 1) in normalized (latency, success) space",
        "=" * 72,
        "",
        "Ideal point: (x*, y*) = (0, 1)",
        "  x_tilde_i = L_i / L_max",
        "  y_i       = (p1 + p2 + p3 + p4) / 4",
        "  d_i       = sqrt( x_tilde_i^2 + (1 - y_i)^2 )",
        "",
        f"L_max (ms) = {l_max:.6f}   (max latency among {len(dist_rows)} seed points)",
        "",
        "Mean distance per (model, NFE), sorted ascending:",
        f"{'model':<18} {'NFE':>4} {'lat_ms':>10} {'SR':>8} {'x_tilde':>8} {'d':>8}",
        "-" * 72,
    ]
    for r in mean_rows:
        lines.append(
            f"{LABEL.get(r['model'], r['model']):<18} {r['nfe']:>4} "
            f"{r['latency_ms']:10.2f} {r['success_rate_p14']:8.4f} "
            f"{r['x_tilde']:8.4f} {r['distance']:8.4f}"
        )
    lines.extend(
        [
            "",
            "Per-seed distances (see utopia_distance.csv for full table).",
            f"Closest seed point: "
            f"{LABEL.get(dist_rows[0]['model'], dist_rows[0]['model'])} "
            f"seed={dist_rows[0]['train_seed']} NFE={dist_rows[0]['nfe']} "
            f"d={dist_rows[0]['distance']:.4f}",
            f"Farthest seed point: "
            f"{LABEL.get(dist_rows[-1]['model'], dist_rows[-1]['model'])} "
            f"seed={dist_rows[-1]['train_seed']} NFE={dist_rows[-1]['nfe']} "
            f"d={dist_rows[-1]['distance']:.4f}",
            "",
        ]
    )
    (out_dir / "utopia_distance.txt").write_text("\n".join(lines) + "\n")


def plot_pareto_per_seed(
    runs: List[Dict[str, Any]], agg: List[Dict[str, Any]], out_dir: Path
) -> None:
    """Scatter FP@1/8 + DP@100 seeds + utopia Ideal(0,1) + Euclidean distances."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return
    if not runs:
        return

    # Full set for utopia log; focused set for the figure
    all_points: List[Dict[str, Any]] = []
    for r in runs:
        sr = _success_rate_p14(r)
        if sr is None or r.get("latency_ms") is None:
            continue
        all_points.append({**r, "success_rate_p14": sr})
    if not all_points:
        return

    l_max, dist_rows = compute_utopia_distances(all_points)
    write_utopia_logs(out_dir, l_max, dist_rows, agg)

    points = [p for p in all_points if _is_highlight(p["model"], p["nfe"])]
    agg_h = _filter_highlight_agg(agg)
    if not points or not agg_h:
        return

    # Mean-point distances (annotate on plot); same L_max as full log
    mean_dist: Dict[Tuple[str, int], float] = {}
    for r in agg_h:
        if r.get("latency_ms") is None or r.get("success_rate_p14") is None:
            continue
        x_tilde = float(r["latency_ms"]) / l_max
        y = float(r["success_rate_p14"])
        mean_dist[(r["model"], int(r["nfe"]))] = float(
            np.sqrt(x_tilde**2 + (1.0 - y) ** 2)
        )

    fig, ax = plt.subplots(figsize=(9.0, 5.8))

    lats = [float(p["latency_ms"]) for p in points]
    l_min = float(min(lats))
    # Visual utopia slightly left of min latency (log scale cannot show L=0)
    utopia_x = l_min / 1.6
    utopia_y = 1.0

    # Mean curves: FlowPolicy connects NFE 1→8 only
    fp_pts = sorted(
        [
            r
            for r in agg_h
            if r["model"] == "flowpolicy" and r.get("success_rate_p14") is not None
        ],
        key=lambda x: x["nfe"],
    )
    if len(fp_pts) >= 2:
        ax.plot(
            [p["latency_ms"] for p in fp_pts],
            [p["success_rate_p14"] for p in fp_pts],
            color=COLOR["flowpolicy"],
            lw=1.5,
            alpha=0.55,
            zorder=2,
        )

    # Thin dashed lines from utopia to each highlight mean
    for r in agg_h:
        if r.get("latency_ms") is None or r.get("success_rate_p14") is None:
            continue
        ax.plot(
            [utopia_x, r["latency_ms"]],
            [utopia_y, r["success_rate_p14"]],
            color=COLOR[r["model"]],
            linestyle=":",
            lw=0.8,
            alpha=0.35,
            zorder=1,
        )

    # Per-seed scatter
    for r in points:
        mk = NFE_MARKER.get(int(r["nfe"]), "o")
        ax.scatter(
            r["latency_ms"],
            r["success_rate_p14"],
            c=COLOR[r["model"]],
            marker=mk,
            s=48,
            alpha=0.55,
            edgecolors="none",
            zorder=3,
        )

    # Mean points + NFE label + distance annotation
    for r in agg_h:
        if r.get("latency_ms") is None or r.get("success_rate_p14") is None:
            continue
        model, nfe = r["model"], int(r["nfe"])
        ax.scatter(
            r["latency_ms"],
            r["success_rate_p14"],
            c=COLOR[model],
            marker=NFE_MARKER.get(nfe, "o"),
            s=90,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        d = mean_dist.get((model, nfe), float("nan"))
        ax.annotate(
            f"NFE={nfe}\nd={d:.3f}",
            (r["latency_ms"], r["success_rate_p14"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=6.5,
            color=COLOR[model],
            zorder=6,
        )

    # Ideal marker
    ax.scatter(
        [utopia_x],
        [utopia_y],
        c="black",
        marker="*",
        s=220,
        zorder=7,
        label="Ideal (0,1)",
    )
    ax.annotate(
        "Ideal (0,1)",
        (utopia_x, utopia_y),
        textcoords="offset points",
        xytext=(-8, -14),
        fontsize=8,
        fontweight="bold",
        zorder=7,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Inference latency (ms)")
    ax.set_ylabel("Success rate  (p1+p2+p3+p4) / 4")
    ax.set_title("Quality–latency with utopia distance (FP@1/8, DP@100)")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR[m],
            markersize=8,
            label=LABEL.get(m, m),
        )
        for m in MODEL_ORDER
    ]
    nfe_handles = [
        Line2D(
            [0],
            [0],
            marker=NFE_MARKER[n],
            color="w",
            markerfacecolor="#555555",
            markeredgecolor="#555555",
            markersize=8,
            label=f"NFE={n}",
        )
        for n in HIGHLIGHT_NFES
    ]
    ideal_handle = Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor="black",
        markersize=12,
        label="Ideal (0,1)",
    )
    leg1 = ax.legend(
        handles=model_handles + [ideal_handle],
        loc="lower right",
        fontsize=8,
        frameon=False,
        title="Model",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=nfe_handles,
        loc="center right",
        fontsize=8,
        frameon=False,
        title="NFE",
    )

    fig.tight_layout()
    fig.savefig(out_dir / "pareto_per_seed.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "pareto_per_seed.pdf", bbox_inches="tight")
    plt.close(fig)


def write_pk_degradation_logs(
    agg_h: List[Dict[str, Any]], out_dir: Path
) -> None:
    """Write mean±std multistage p1..p4 tables (aggregated across seeds)."""
    stages = ["p1", "p2", "p3", "p4"]
    csv_rows: List[Dict[str, Any]] = []
    lines = [
        "Multistage success rate p1→p4 (mean ± std across 3 training seeds)",
        "Operating points: FlowPolicy @ NFE={1,8}, DP-CNN/DP-Transformer @ NFE=100",
        "=" * 78,
        "",
        f"{'model':<18} {'NFE':>4} "
        + " ".join(f"{s + '_mean':>10} {s + '_std':>9}" for s in stages),
        "-" * 78,
    ]
    for model in MODEL_ORDER:
        for nfe in HIGHLIGHT_NFES:
            if not _is_highlight(model, nfe):
                continue
            row = next(
                (r for r in agg_h if r["model"] == model and int(r["nfe"]) == nfe),
                None,
            )
            if row is None:
                continue
            csv_row: Dict[str, Any] = {
                "model": model,
                "nfe": nfe,
                "n_seeds": row.get("n_seeds"),
            }
            parts = [f"{LABEL.get(model, model):<18} {nfe:>4}"]
            for s in stages:
                mu = row.get(s)
                sd = row.get(f"{s}_std")
                if mu is None:
                    parts.append(f"{'n/a':>10} {'n/a':>9}")
                    csv_row[f"{s}_mean"] = ""
                    csv_row[f"{s}_std"] = ""
                else:
                    sd_f = float(sd or 0.0)
                    parts.append(f"{float(mu):10.4f} {sd_f:9.4f}")
                    csv_row[f"{s}_mean"] = float(mu)
                    csv_row[f"{s}_std"] = sd_f
            lines.append(" ".join(parts))
            # compact mean±std line
            compact = []
            for s in stages:
                mu = row.get(s)
                sd = float(row.get(f"{s}_std") or 0.0) if mu is not None else None
                if mu is None:
                    compact.append(f"{s}=n/a")
                else:
                    compact.append(f"{s}={float(mu):.3f}±{sd:.3f}")
            lines.append("    " + "  ".join(compact))
            csv_rows.append(csv_row)
    lines.append("")
    (out_dir / "pk_degradation.txt").write_text("\n".join(lines) + "\n")
    write_csv(out_dir / "pk_degradation.csv", csv_rows)


def plot_pk_degradation(
    agg: List[Dict[str, Any]], out_dir: Path
) -> None:
    """One plot: mean±std p1→p4 for FP@1/8 and DP@100 only."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return
    if not agg:
        return

    agg_h = _filter_highlight_agg(agg)
    if not agg_h:
        return

    write_pk_degradation_logs(agg_h, out_dir)

    stages = ["p1", "p2", "p3", "p4"]
    xs = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    for model in MODEL_ORDER:
        for nfe in HIGHLIGHT_NFES:
            if not _is_highlight(model, nfe):
                continue
            row = next(
                (r for r in agg_h if r["model"] == model and int(r["nfe"]) == nfe),
                None,
            )
            if row is None:
                continue
            ys = [row.get(s) for s in stages]
            if any(v is None for v in ys):
                continue
            yerr = [float(row.get(f"{s}_std") or 0.0) for s in stages]
            ax.errorbar(
                xs,
                [float(v) for v in ys],
                yerr=yerr,
                color=COLOR[model],
                marker=NFE_MARKER[nfe],
                lw=1.8,
                markersize=8,
                capsize=4,
                elinewidth=1.2,
                markeredgecolor="black",
                markeredgewidth=0.4,
                alpha=0.95,
                label=f"{LABEL.get(model, model)} NFE={nfe}",
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Multistage success rate (mean ± std)")
    ax.set_xlabel("Stage")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(
        "Multistage degradation p1→p4 (FP@1/8, DP@100)\n"
        "mean ± std across 3 training seeds"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    model_handles = [
        Line2D(
            [0],
            [0],
            color=COLOR[m],
            lw=2,
            label=LABEL.get(m, m),
        )
        for m in MODEL_ORDER
    ]
    nfe_handles = [
        Line2D(
            [0],
            [0],
            marker=NFE_MARKER[n],
            color="w",
            markerfacecolor="#555555",
            markeredgecolor="#555555",
            markersize=8,
            label=f"NFE={n}",
        )
        for n in HIGHLIGHT_NFES
    ]
    leg1 = ax.legend(
        handles=model_handles,
        loc="lower left",
        fontsize=8,
        frameon=False,
        title="Model",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=nfe_handles,
        loc="center right",
        fontsize=8,
        frameon=False,
        title="NFE",
    )

    fig.tight_layout()
    fig.savefig(out_dir / "pk_degradation.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "pk_degradation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_all(
    agg: List[Dict[str, Any]], out_dir: Path, runs: Optional[List[Dict[str, Any]]] = None
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not agg:
        return

    models = sorted({r["model"] for r in agg})

    # 1) mean_tasks / p3 / p4 vs NFE
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for model in models:
        pts = sorted([r for r in agg if r["model"] == model], key=lambda x: x["nfe"])
        xs = [p["nfe"] for p in pts]
        c, mk, lb = COLOR[model], MARKER[model], LABEL.get(model, model)
        axes[0].errorbar(
            xs,
            [p["mean_tasks"] for p in pts],
            yerr=[p["mean_tasks_std"] or 0 for p in pts],
            marker=mk,
            color=c,
            label=lb,
            lw=2,
            capsize=3,
        )
        axes[1].errorbar(
            xs,
            [p["p3"] for p in pts],
            yerr=[p["p3_std"] or 0 for p in pts],
            marker=mk,
            color=c,
            label=lb,
            lw=2,
            capsize=3,
        )
        axes[2].errorbar(
            xs,
            [p["p4"] for p in pts],
            yerr=[p["p4_std"] or 0 for p in pts],
            marker=mk,
            color=c,
            label=lb,
            lw=2,
            capsize=3,
        )
    for ax, title, ylab, ylim in zip(
        axes,
        ["Mean tasks vs NFE", "p3 vs NFE", "p4 vs NFE"],
        ["Mean tasks", "p3", "p4"],
        [(-0.1, 4.3), (-0.05, 1.05), (-0.05, 1.05)],
    ):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("NFE")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Kitchen NFE100 (mean ± std across 3 training seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "success_vs_nfe.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "success_vs_nfe.pdf", bbox_inches="tight")
    plt.close(fig)

    # 2) latency
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model in models:
        pts = sorted([r for r in agg if r["model"] == model], key=lambda x: x["nfe"])
        ax.errorbar(
            [p["nfe"] for p in pts],
            [p["latency_ms"] for p in pts],
            yerr=[p["latency_ms_std"] or 0 for p in pts],
            marker=MARKER[model],
            color=COLOR[model],
            label=LABEL.get(model, model),
            lw=2,
            capsize=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("NFE")
    ax.set_ylabel("Inference latency (ms)")
    ax.set_title("Latency vs NFE")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_nfe.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "latency_vs_nfe.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3) pareto
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model in models:
        pts = sorted([r for r in agg if r["model"] == model], key=lambda x: x["nfe"])
        xs = [p["latency_ms"] for p in pts]
        ys = [p["mean_tasks"] for p in pts]
        ax.plot(
            xs,
            ys,
            marker=MARKER[model],
            color=COLOR[model],
            label=LABEL.get(model, model),
            lw=2,
        )
        for x, y, n in zip(xs, ys, [p["nfe"] for p in pts]):
            if x is not None and y is not None:
                ax.annotate(
                    str(n),
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=COLOR[model],
                )
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency (ms)")
    ax.set_ylabel("Mean tasks completed")
    ax.set_title("Quality–latency trade-off (labels = NFE)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "pareto.pdf", bbox_inches="tight")
    plt.close(fig)

    if runs:
        plot_pareto_per_seed(runs, agg, out_dir)
        plot_pk_degradation(agg, out_dir)


def write_report(
    runs: List[Dict[str, Any]], agg: List[Dict[str, Any]], out_dir: Path
) -> Path:
    lines = [
        "=" * 78,
        "Kitchen NFE100 full eval report",
        "DP-CNN / DP-Transformer / FlowPolicy × NFE {1,8,32,100} × 3 seeds × 100 ep",
        "=" * 78,
        "",
        f"Discovered runs: {len(runs)}",
        f"Aggregated (model,NFE) points: {len(agg)}",
        "",
        f"{'model':<32} {'NFE':>4} {'n_seeds':>7} {'mean_tasks':>14} "
        f"{'p1':>14} {'p2':>14} {'p3':>14} {'p4':>14} {'lat_ms':>12}",
        "-" * 140,
    ]
    for r in agg:
        lines.append(
            f"{r['model']:<32} {r['nfe']:>4} {r['n_seeds']:>7} "
            f"{(r['mean_tasks'] or 0):6.3f}±{(r['mean_tasks_std'] or 0):.3f}  "
            f"{(r['p1'] or 0):5.3f}±{(r['p1_std'] or 0):.3f}  "
            f"{(r['p2'] or 0):5.3f}±{(r['p2_std'] or 0):.3f}  "
            f"{(r['p3'] or 0):5.3f}±{(r['p3_std'] or 0):.3f}  "
            f"{(r['p4'] or 0):5.3f}±{(r['p4_std'] or 0):.3f}  "
            f"{(r['latency_ms'] or 0):7.1f}±{(r['latency_ms_std'] or 0):.1f}"
        )

    lines.extend(
        [
            "",
            "Per-seed details (mean ± std across episodes within each seed):",
            f"{'model':<18} {'seed':<12} {'NFE':>4} {'n_ep':>5} {'mean_tasks':>14} "
            f"{'p1':>14} {'p2':>14} {'p3':>14} {'p4':>14} {'lat_ms':>14}",
            "-" * 135,
        ]
    )
    for model in MODEL_ORDER:
        model_runs = [r for r in runs if r["model"] == model]
        for r in sorted(model_runs, key=lambda x: (x["train_seed"], x["nfe"])):
            lines.append(
                f"{LABEL.get(model, model):<18} {r['train_seed']:<12} "
                f"{r['nfe']:>4} {r['n_episodes']:>5} "
                f"{r['mean_tasks']:6.3f}±{r['mean_tasks_episode_std']:.3f}  "
                f"{(r['p1'] or 0):5.3f}±{(r['p1_episode_std'] or 0):.3f}  "
                f"{(r['p2'] or 0):5.3f}±{(r['p2_episode_std'] or 0):.3f}  "
                f"{(r['p3'] or 0):5.3f}±{(r['p3_episode_std'] or 0):.3f}  "
                f"{(r['p4'] or 0):5.3f}±{(r['p4_episode_std'] or 0):.3f}  "
                f"{(r['latency_ms'] or 0):7.1f}±"
                f"{(r['latency_episode_std'] or 0):.1f}"
            )

    # Highlight equal-NFE comparisons at 1,8,32,100
    lines.extend(["", "Equal-NFE snapshots (mean_tasks / p1–p4):"])
    by_mn: Dict[Tuple[str, int], Dict[str, Any]] = {
        (r["model"], r["nfe"]): r for r in agg
    }
    for nfe in (1, 8, 32, 100):
        lines.append(f"  NFE={nfe}:")
        for model in (
            "flowpolicy",
            "diffusion_policy_cnn",
            "diffusion_policy_transformer",
        ):
            r = by_mn.get((model, nfe))
            if not r:
                lines.append(f"    {LABEL.get(model, model)}: (missing)")
                continue
            lines.append(
                f"    {LABEL.get(model, model)}: "
                f"tasks={r['mean_tasks']:.3f}±{(r['mean_tasks_std'] or 0):.3f}  "
                f"p1={(r['p1'] or 0):.3f}±{(r['p1_std'] or 0):.3f}  "
                f"p2={(r['p2'] or 0):.3f}±{(r['p2_std'] or 0):.3f}  "
                f"p3={(r['p3'] or 0):.3f}±{(r['p3_std'] or 0):.3f}  "
                f"p4={(r['p4'] or 0):.3f}±{(r['p4_std'] or 0):.3f}  "
                f"lat={r['latency_ms']:.1f}ms"
            )

    expected = 3 * 3 * 4  # models × seeds × nfe
    lines.extend(
        [
            "",
            f"Completeness: {len(runs)}/{expected} runs "
            f"({'COMPLETE' if len(runs) >= expected else 'INCOMPLETE — resume orchestrator'})",
            "",
            "=" * 78,
        ]
    )
    path = out_dir / "report.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_root_dp",
        type=Path,
        default=None,
        help="Default: kitchen_eval_nfe100_diffusion if present, else kitchen_eval_nfe100",
    )
    ap.add_argument(
        "--input_root_fp",
        type=Path,
        default=ROOT / "kripsy12/FlowPolicy/data/kitchen_eval_nfe100/flowpolicy",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "data/kitchen_eval_plots/nfe100",
    )
    args = ap.parse_args()
    if args.input_root_dp is None:
        cand = [
            ROOT / "diffusion_policy/data/kitchen_eval_nfe100_diffusion",
            ROOT / "diffusion_policy/data/kitchen_eval_nfe100",
        ]
        args.input_root_dp = next(
            (p for p in cand if p.is_dir() and any(p.rglob("eval_metrics.json"))),
            cand[0],
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover(args.input_root_dp, args.input_root_fp)
    agg = aggregate(runs)
    write_csv(args.output_dir / "summary.csv", agg)
    # also per-run csv
    write_csv(
        args.output_dir / "runs.csv",
        [
            {
                "model": r["model"],
                "train_seed": r["train_seed"],
                "nfe": r["nfe"],
                "sseed": r["sseed"],
                "mean_tasks": r["mean_tasks"],
                "mean_tasks_episode_std": r["mean_tasks_episode_std"],
                "p1": r["p1"],
                "p1_episode_std": r["p1_episode_std"],
                "p2": r["p2"],
                "p2_episode_std": r["p2_episode_std"],
                "p3": r["p3"],
                "p3_episode_std": r["p3_episode_std"],
                "p4": r["p4"],
                "p4_episode_std": r["p4_episode_std"],
                "success_rate_p14": _success_rate_p14(r),
                "latency_ms": r["latency_ms"],
                "latency_episode_std": r["latency_episode_std"],
                "n_episodes": r["n_episodes"],
                "path": r["path"],
            }
            for r in runs
        ],
    )
    plot_all(agg, args.output_dir, runs=runs)
    report = write_report(runs, agg, args.output_dir)
    print(report.read_text())
    print(f"Wrote {report}")
    print(f"Runs={len(runs)} aggregated_points={len(agg)}")


if __name__ == "__main__":
    main()
