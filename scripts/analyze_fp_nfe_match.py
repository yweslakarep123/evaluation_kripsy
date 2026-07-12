#!/usr/bin/env python3
"""Analyze FlowPolicy NFE sweep vs DP p3/p4 target (~1.0).

Reads existing FP NFE runs under kitchen_eval_nfe/flowpolicy/ plus DP@100
reference from the earlier sweep. Writes report + plot.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/kitchen_eval_plots/nfe_variance"
FP_ROOT = ROOT / "kripsy12/FlowPolicy/data/kitchen_eval_nfe/flowpolicy"
DP_ROOT = ROOT / "diffusion_policy/data/kitchen_eval_nfe"
DIR_RE = re.compile(r"^seed_(?P<seed>.+)_nfe(?P<nfe>\d+)_sseed(?P<sseed>\d+)$")
TARGET_P3 = 0.98
TARGET_P4 = 0.95


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


def load_fp_points() -> List[Dict[str, Any]]:
    rows = []
    if not FP_ROOT.exists():
        return rows
    for d in sorted(FP_ROOT.iterdir()):
        m = DIR_RE.match(d.name)
        if not m or int(m.group("sseed")) != 0:
            continue
        path = d / "eval_metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text())
        rows.append(
            {
                "model": "flowpolicy",
                "nfe": int(m.group("nfe")),
                "p3": _px(metrics, 3),
                "p4": _px(metrics, 4),
                "mean_tasks": _mean_tasks(metrics),
                "latency_ms": _lat(metrics),
                "path": str(path),
            }
        )
    return sorted(rows, key=lambda r: r["nfe"])


def load_dp_ref() -> List[Dict[str, Any]]:
    rows = []
    for model in ("diffusion_policy_cnn", "diffusion_policy_transformer"):
        path = DP_ROOT / model / "seed_train0_nfe100_sseed0" / "eval_metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text())
        rows.append(
            {
                "model": model,
                "nfe": 100,
                "p3": _px(metrics, 3),
                "p4": _px(metrics, 4),
                "mean_tasks": _mean_tasks(metrics),
                "latency_ms": _lat(metrics),
            }
        )
    return rows


def plot(fp_rows: List[Dict[str, Any]], dp_rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not fp_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    xs = [r["nfe"] for r in fp_rows]
    axes[0].plot(xs, [r["p3"] for r in fp_rows], "D-", color="#2ca02c", lw=2, label="FlowPolicy")
    axes[1].plot(xs, [r["p4"] for r in fp_rows], "D-", color="#2ca02c", lw=2, label="FlowPolicy")
    axes[2].plot(xs, [r["latency_ms"] for r in fp_rows], "D-", color="#2ca02c", lw=2, label="FlowPolicy")
    axes[0].axhline(TARGET_P3, color="gray", ls="--", lw=1, label=f"target p3={TARGET_P3}")
    axes[1].axhline(TARGET_P4, color="gray", ls="--", lw=1, label=f"target p4={TARGET_P4}")
    for dp in dp_rows:
        label = "DP-CNN@100" if "cnn" in dp["model"] else "DP-TF@100"
        color = "#1f77b4" if "cnn" in dp["model"] else "#ff7f0e"
        axes[0].scatter([dp["nfe"]], [dp["p3"]], color=color, marker="o", s=60, zorder=3, label=label)
        axes[1].scatter([dp["nfe"]], [dp["p4"]], color=color, marker="o", s=60, zorder=3, label=label)
        axes[2].scatter([dp["nfe"]], [dp["latency_ms"]], color=color, marker="o", s=60, zorder=3, label=label)
    for ax, title, ylab in zip(
        axes,
        ["p3 vs NFE", "p4 vs NFE", "Latency vs NFE"],
        ["p3", "p4", "ms"],
    ):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("NFE")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    fig.suptitle("FlowPolicy NFE to match DP p3/p4", y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fp_nfe_match_p3_p4.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "fp_nfe_match_p3_p4.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(fp_rows: List[Dict[str, Any]], dp_rows: List[Dict[str, Any]]) -> Path:
    hits = [
        r
        for r in fp_rows
        if r["p3"] is not None
        and r["p4"] is not None
        and r["p3"] >= TARGET_P3
        and r["p4"] >= TARGET_P4
    ]
    best = None
    if fp_rows:
        best = max(fp_rows, key=lambda r: (r["p4"] or -1, r["p3"] or -1))

    lines = [
        "=" * 78,
        "FlowPolicy NFE match report — target p3>=0.98 and p4>=0.95 (DP@100-like)",
        "=" * 78,
        "",
        "DP@100 reference:",
    ]
    for dp in dp_rows:
        lines.append(
            f"  {dp['model']}: p3={dp['p3']} p4={dp['p4']} "
            f"mean_tasks={dp['mean_tasks']:.3f} lat_ms={dp['latency_ms']:.1f}"
        )
    lines.extend(["", "FlowPolicy curve (sseed=0):", f"{'NFE':>6}  {'p3':>6}  {'p4':>6}  {'mean_tasks':>10}  {'lat_ms':>8}"])
    for r in fp_rows:
        lines.append(
            f"{r['nfe']:>6}  {r['p3'] if r['p3'] is not None else float('nan'):6.3f}  "
            f"{r['p4'] if r['p4'] is not None else float('nan'):6.3f}  "
            f"{r['mean_tasks']:10.3f}  {r['latency_ms'] if r['latency_ms'] is not None else float('nan'):8.1f}"
        )
    lines.append("")
    if hits:
        first = min(hits, key=lambda r: r["nfe"])
        lines.append(
            f"VERDICT: TARGET REACHED at minimal NFE={first['nfe']} "
            f"(p3={first['p3']:.3f}, p4={first['p4']:.3f}, lat={first['latency_ms']:.1f} ms)"
        )
        if dp_rows:
            dp_lat = np.mean([d["latency_ms"] for d in dp_rows if d["latency_ms"]])
            lines.append(
                f"  Latency vs DP@100: FP@{first['nfe']}={first['latency_ms']:.1f} ms "
                f"vs DP≈{dp_lat:.1f} ms "
                f"({dp_lat / first['latency_ms']:.1f}× slower DP)" if first["latency_ms"] else ""
            )
    else:
        lines.append("VERDICT: TARGET NOT REACHED in swept NFE grid.")
        if best:
            lines.append(
                f"  Best FP point: NFE={best['nfe']} p3={best['p3']} p4={best['p4']} "
                f"mean_tasks={best['mean_tasks']:.3f} lat={best['latency_ms']:.1f} ms"
            )
            lines.append(
                "  Likely ceiling: adding inference steps does not push p4 to DP levels "
                "(policy chaining / stop behavior), not just insufficient NFE."
            )
    lines.append("")
    lines.append("=" * 78)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fp_nfe_match_report.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    fp_rows = load_fp_points()
    dp_rows = load_dp_ref()
    plot(fp_rows, dp_rows)
    report = write_report(fp_rows, dp_rows)
    print(report.read_text())
    print(f"Wrote {report}")
    print(f"FP points: {len(fp_rows)}  DP refs: {len(dp_rows)}")


if __name__ == "__main__":
    main()
