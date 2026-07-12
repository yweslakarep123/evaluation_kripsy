#!/usr/bin/env python3
"""Analyze full Kitchen NFE100 eval (3 seeds × NFE grid × 3 models).

Reads:
  diffusion_policy/data/kitchen_eval_nfe100/<model>/seed_*_nfe*_sseed*/
  kripsy12/FlowPolicy/data/kitchen_eval_nfe100/flowpolicy/seed_*_nfe*_sseed*/

Writes under data/kitchen_eval_plots/nfe100/:
  report.txt, summary.csv, success_vs_nfe.png, latency_vs_nfe.png, pareto.png
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


def _lat(m: Dict[str, Any]) -> Optional[float]:
    return m.get("timing_ms", {}).get("inference_latency", {}).get("mean")


def _mean_tasks(m: Dict[str, Any]) -> float:
    eps = m.get("episodes") or []
    if not eps:
        return float("nan")
    return float(np.mean([e.get("num_tasks_completed", 0) for e in eps]))


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
                "p1": _px(metrics, 1),
                "p2": _px(metrics, 2),
                "p3": _px(metrics, 3),
                "p4": _px(metrics, 4),
                "latency_ms": _lat(metrics),
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
        p3_m, p3_s = _mean_std([g["p3"] for g in group if g["p3"] is not None])
        p4_m, p4_s = _mean_std([g["p4"] for g in group if g["p4"] is not None])
        lat_m, lat_s = _mean_std(
            [g["latency_ms"] for g in group if g["latency_ms"] is not None]
        )
        row: Dict[str, Any] = {
            "model": model,
            "nfe": nfe,
            "n_seeds": len(group),
            "seeds": ",".join(sorted(by_seed.keys())),
            "mean_tasks": mt_m,
            "mean_tasks_std": mt_s,
            "p3": p3_m,
            "p3_std": p3_s,
            "p4": p4_m,
            "p4_std": p4_s,
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


def plot_all(agg: List[Dict[str, Any]], out_dir: Path) -> None:
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
        f"{'model':<32} {'NFE':>4} {'n_seeds':>7} {'mean_tasks':>14} {'p3':>14} {'p4':>14} {'lat_ms':>12}",
        "-" * 110,
    ]
    for r in agg:
        lines.append(
            f"{r['model']:<32} {r['nfe']:>4} {r['n_seeds']:>7} "
            f"{(r['mean_tasks'] or 0):6.3f}±{(r['mean_tasks_std'] or 0):.3f}  "
            f"{(r['p3'] or 0):5.3f}±{(r['p3_std'] or 0):.3f}  "
            f"{(r['p4'] or 0):5.3f}±{(r['p4_std'] or 0):.3f}  "
            f"{(r['latency_ms'] or 0):7.1f}±{(r['latency_ms_std'] or 0):.1f}"
        )

    # Highlight equal-NFE comparisons at 1,8,32,100
    lines.extend(["", "Equal-NFE snapshots (mean_tasks / p3 / p4):"])
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
                f"p3={r['p3']}  p4={r['p4']}  "
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
        default=ROOT / "diffusion_policy/data/kitchen_eval_nfe100",
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
                "p3": r["p3"],
                "p4": r["p4"],
                "latency_ms": r["latency_ms"],
                "n_episodes": r["n_episodes"],
                "path": r["path"],
            }
            for r in runs
        ],
    )
    plot_all(agg, args.output_dir)
    report = write_report(runs, agg, args.output_dir)
    print(report.read_text())
    print(f"Wrote {report}")
    print(f"Runs={len(runs)} aggregated_points={len(agg)}")


if __name__ == "__main__":
    main()
