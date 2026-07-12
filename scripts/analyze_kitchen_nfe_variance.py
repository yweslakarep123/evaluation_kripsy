#!/usr/bin/env python3
"""Analyze Kitchen NFE sweep + sampling-seed variance for FlowPolicy vs DP claims.

Reads eval_metrics.json under kitchen_eval_nfe/ directories named:
  seed_<name>_nfe<N>_sseed<S>/

Writes:
  report.txt, success_vs_nfe.csv, sampling_variance.csv,
  success_vs_nfe.png, sampling_variance.png
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
DEFAULT_NFE = {
    "flowpolicy": 1,
    "diffusion_policy_cnn": 100,
    "diffusion_policy_transformer": 100,
}


def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _mean_tasks(metrics: Dict[str, Any]) -> float:
    episodes = metrics.get("episodes") or []
    if episodes:
        return float(np.mean([float(e.get("num_tasks_completed", 0)) for e in episodes]))
    # fallback from success rates
    srs = []
    for t in TASKS:
        m = metrics.get("success_rate", {}).get(t, {}).get("mean")
        if m is not None:
            srs.append(float(m))
    return float(sum(srs)) if srs else 0.0


def _px(metrics: Dict[str, Any], k: int) -> Optional[float]:
    px = (
        metrics.get("multistage_metrics", {})
        .get("all_7_tasks", {})
        .get("px", {})
        .get(f"p{k}")
    )
    if isinstance(px, dict):
        return px.get("mean")
    return px


def _latency(metrics: Dict[str, Any]) -> Optional[float]:
    return metrics.get("timing_ms", {}).get("inference_latency", {}).get("mean")


def discover_runs(
    root: Path, model_label: str
) -> List[Dict[str, Any]]:
    runs = []
    if not root.exists():
        return runs
    # DP layout: root/model/seed_*_nfe*_sseed*/
    # FP layout: root/seed_*_nfe*_sseed*/  (root already ends with flowpolicy)
    candidates = []
    model_dirs = [p for p in root.iterdir() if p.is_dir()]
    for p in model_dirs:
        if DIR_RE.match(p.name):
            candidates.append((model_label, p))
        else:
            for child in p.iterdir():
                if child.is_dir() and DIR_RE.match(child.name):
                    candidates.append((p.name, child))

    for model, d in candidates:
        m = DIR_RE.match(d.name)
        assert m is not None
        metrics_path = d / "eval_metrics.json"
        if not metrics_path.is_file():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        nfe = int(m.group("nfe"))
        sseed = int(m.group("sseed"))
        # Prefer label from parent for DP
        label = model if model != model_label else model_label
        if "diffusion_policy" in str(d):
            # parent of seed dir is model name
            label = d.parent.name
        elif "flowpolicy" in str(d).lower() or model_label == "flowpolicy":
            label = "flowpolicy"
        runs.append(
            {
                "model": label,
                "train_seed": m.group("seed"),
                "nfe": nfe,
                "sseed": sseed,
                "path": str(metrics_path),
                "mean_tasks": _mean_tasks(metrics),
                "p1": _px(metrics, 1),
                "p2": _px(metrics, 2),
                "p3": _px(metrics, 3),
                "p4": _px(metrics, 4),
                "latency_ms": _latency(metrics),
                "success_rate": {
                    t: metrics.get("success_rate", {}).get(t, {}).get("mean")
                    for t in TASKS
                },
                "n_episodes": metrics.get("n_episodes"),
            }
        )
    return runs


def load_all(input_root_dp: Path, input_root_fp: Path) -> List[Dict[str, Any]]:
    runs = []
    runs.extend(discover_runs(input_root_dp, "diffusion_policy"))
    runs.extend(discover_runs(input_root_fp, "flowpolicy"))
    # Deduplicate by path
    by_path = {r["path"]: r for r in runs}
    return sorted(
        by_path.values(),
        key=lambda r: (r["model"], r["nfe"], r["sseed"]),
    )


def nfe_curve_rows(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate by (model, nfe) using sseed=0 preferentially; else mean across seeds."""
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped[(r["model"], r["nfe"])].append(r)

    rows = []
    for (model, nfe), group in sorted(grouped.items()):
        preferred = [g for g in group if g["sseed"] == 0]
        use = preferred if preferred else group
        mt = [g["mean_tasks"] for g in use]
        p3 = [g["p3"] for g in use if g["p3"] is not None]
        p4 = [g["p4"] for g in use if g["p4"] is not None]
        lat = [g["latency_ms"] for g in use if g["latency_ms"] is not None]
        mt_m, mt_s = _mean_std(mt)
        p3_m, p3_s = _mean_std(p3)
        p4_m, p4_s = _mean_std(p4)
        lat_m, lat_s = _mean_std(lat)
        rows.append(
            {
                "model": model,
                "nfe": nfe,
                "n_runs": len(use),
                "mean_tasks": mt_m,
                "mean_tasks_std": mt_s,
                "p3": p3_m,
                "p3_std": p3_s,
                "p4": p4_m,
                "p4_std": p4_s,
                "latency_ms": lat_m,
                "latency_ms_std": lat_s,
            }
        )
    return rows


def variance_rows(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Std across sampling seeds at each model's default NFE."""
    rows = []
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        default = DEFAULT_NFE.get(r["model"])
        if default is None:
            continue
        if r["nfe"] == default:
            by_model[r["model"]].append(r)

    for model, group in sorted(by_model.items()):
        # unique by sseed
        by_sseed = {g["sseed"]: g for g in group}
        group = list(by_sseed.values())
        if len(group) < 2:
            continue
        mt = [g["mean_tasks"] for g in group]
        p3 = [g["p3"] for g in group if g["p3"] is not None]
        p4 = [g["p4"] for g in group if g["p4"] is not None]
        mt_m, mt_s = _mean_std(mt)
        p3_m, p3_s = _mean_std(p3)
        p4_m, p4_s = _mean_std(p4)
        task_stds = []
        task_detail = {}
        for t in TASKS:
            vals = [
                g["success_rate"][t]
                for g in group
                if g["success_rate"].get(t) is not None
            ]
            m, s = _mean_std([float(v) for v in vals])
            task_detail[t] = {"mean": m, "std": s}
            if s is not None:
                task_stds.append(s)
        rows.append(
            {
                "model": model,
                "nfe": DEFAULT_NFE[model],
                "n_sampling_seeds": len(group),
                "sampling_seeds": sorted(by_sseed.keys()),
                "mean_tasks_mean": mt_m,
                "mean_tasks_std": mt_s,
                "p3_mean": p3_m,
                "p3_std": p3_s,
                "p4_mean": p4_m,
                "p4_std": p4_s,
                "mean_per_task_std": float(np.mean(task_stds)) if task_stds else None,
                "per_task": task_detail,
            }
        )
    return rows


def verdict_claim1(nfe_rows: List[Dict[str, Any]]) -> str:
    """FP better at low NFE (<=8) on mean_tasks / p3 / p4?"""
    by_model: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for r in nfe_rows:
        by_model[r["model"]][r["nfe"]] = r

    fp = by_model.get("flowpolicy", {})
    dp_models = [m for m in by_model if m.startswith("diffusion_policy")]
    if not fp or not dp_models:
        return "INCONCLUSIVE — missing FlowPolicy or DP NFE curve points."

    lines = []
    # Equal-NFE comparisons only (FP grid is typically 1/2/4)
    low_nfes = sorted(set(fp.keys()) & {1, 2, 4, 8})
    fp_wins = 0
    comparisons = 0
    for nfe in low_nfes:
        fp_mt = fp[nfe]["mean_tasks"]
        fp_p3 = fp[nfe]["p3"]
        for dm in dp_models:
            if nfe not in by_model[dm]:
                continue
            dp_mt = by_model[dm][nfe]["mean_tasks"]
            dp_p3 = by_model[dm][nfe]["p3"]
            if fp_mt is None or dp_mt is None:
                continue
            comparisons += 1
            # Primary metric: mean tasks completed at equal NFE
            better = fp_mt >= dp_mt
            if better:
                fp_wins += 1
            lines.append(
                f"  NFE={nfe}: FP mean_tasks={fp_mt:.3f} p3={fp_p3} vs "
                f"{dm} mean_tasks={dp_mt:.3f} p3={dp_p3} "
                f"-> {'FP>=DP' if better else 'DP>FP'}"
            )

    # Degradation slope: DP from high to low NFE
    for dm in dp_models:
        pts = by_model[dm]
        if 100 in pts and 1 in pts:
            d_hi = pts[100]["mean_tasks"]
            d_lo = pts[1]["mean_tasks"]
            if d_hi is not None and d_lo is not None:
                lines.append(
                    f"  {dm} degradation mean_tasks @100→@1: "
                    f"{d_hi:.3f} → {d_lo:.3f} (Δ={d_lo - d_hi:+.3f})"
                )
    if 1 in fp and 4 in fp:
        lines.append(
            f"  flowpolicy mean_tasks @1→@4: "
            f"{fp[1]['mean_tasks']:.3f} → {fp[4]['mean_tasks']:.3f}"
        )

    if comparisons == 0:
        return "INCONCLUSIVE — no overlapping low-NFE points.\n" + "\n".join(lines)

    if fp_wins >= max(1, comparisons // 2 + 1):
        verdict = "SUPPORTED"
    elif fp_wins == 0:
        verdict = "REJECTED"
    else:
        verdict = "MIXED / WEAK"
    return f"{verdict} ({fp_wins}/{comparisons} low-NFE comparisons FP>=DP)\n" + "\n".join(
        lines
    )


def verdict_claim2(var_rows: List[Dict[str, Any]]) -> str:
    fp = next((r for r in var_rows if r["model"] == "flowpolicy"), None)
    dps = [r for r in var_rows if r["model"].startswith("diffusion_policy")]
    if fp is None or not dps:
        return "INCONCLUSIVE — need >=2 sampling seeds for FP and at least one DP model."

    lines = []
    fp_std = fp["mean_tasks_std"]
    fp_task = fp["mean_per_task_std"]
    lines.append(
        f"  flowpolicy @NFE={fp['nfe']}: mean_tasks_std={fp_std:.4f}, "
        f"mean_per_task_std={fp_task:.4f} (n_seeds={fp['n_sampling_seeds']})"
    )
    tighter = 0
    for dp in dps:
        lines.append(
            f"  {dp['model']} @NFE={dp['nfe']}: mean_tasks_std={dp['mean_tasks_std']:.4f}, "
            f"mean_per_task_std={dp['mean_per_task_std']:.4f} "
            f"(n_seeds={dp['n_sampling_seeds']})"
        )
        if fp_std is not None and dp["mean_tasks_std"] is not None:
            if fp_std < dp["mean_tasks_std"]:
                tighter += 1
        if fp_task is not None and dp["mean_per_task_std"] is not None:
            if fp_task < dp["mean_per_task_std"]:
                tighter += 1

    # 2 metrics × n DP models
    total = 2 * len(dps)
    if tighter == total:
        verdict = "SUPPORTED"
    elif tighter == 0:
        verdict = "REJECTED"
    else:
        verdict = "MIXED"
    return f"{verdict} (FP tighter on {tighter}/{total} std comparisons)\n" + "\n".join(
        lines
    )


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_nfe(nfe_rows: List[Dict[str, Any]], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    models = sorted({r["model"] for r in nfe_rows})
    for model in models:
        pts = sorted([r for r in nfe_rows if r["model"] == model], key=lambda x: x["nfe"])
        xs = [p["nfe"] for p in pts]
        ys = [p["mean_tasks"] for p in pts]
        yerr = [p["mean_tasks_std"] or 0 for p in pts]
        axes[0].errorbar(xs, ys, yerr=yerr, marker="o", label=model)
        ys3 = [p["p3"] for p in pts]
        axes[1].plot(xs, ys3, marker="o", label=model)
    axes[0].set_xlabel("NFE")
    axes[0].set_ylabel("Mean tasks completed")
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("Success quality vs NFE")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("NFE")
    axes[1].set_ylabel("p3")
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("p3 vs NFE")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_variance(var_rows: List[Dict[str, Any]], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not var_rows:
        return
    labels = [r["model"].replace("diffusion_policy_", "DP-") for r in var_rows]
    mt_std = [r["mean_tasks_std"] * 100 if r["mean_tasks_std"] else 0 for r in var_rows]
    task_std = [
        (r["mean_per_task_std"] or 0) * 100 for r in var_rows
    ]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, mt_std, w, label="std(mean_tasks) ×100")
    ax.bar(x + w / 2, task_std, w, label="mean per-task SR std ×100")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Std across sampling seeds")
    ax.set_title("Claim #2: sampling reproducibility (lower = more deterministic)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def load_training_seed_baseline(experiment_root: Path) -> str:
    baseline = experiment_root / "data/kitchen_eval_plots/nfe_variance/baseline_report.txt"
    if baseline.is_file():
        return baseline.read_text()
    return "(baseline_report.txt not found)\n"


def write_report(
    out_dir: Path,
    runs: List[Dict[str, Any]],
    nfe_rows: List[Dict[str, Any]],
    var_rows: List[Dict[str, Any]],
    experiment_root: Path,
) -> Path:
    v1 = verdict_claim1(nfe_rows)
    v2 = verdict_claim2(var_rows)
    lines = [
        "=" * 78,
        "NFE / SAMPLING-VARIANCE VERIFICATION REPORT",
        "FlowPolicy vs Diffusion Policy — Kitchen lean sweep",
        "=" * 78,
        "",
        f"Discovered runs: {len(runs)}",
        f"Models: {sorted({r['model'] for r in runs})}",
        "",
        "-" * 78,
        "CLAIM #1 — Sample quality at low NFE (success vs NFE)",
        "-" * 78,
        v1,
        "",
        "NFE curve (aggregated):",
    ]
    for r in nfe_rows:
        lines.append(
            f"  {r['model']:<32} NFE={r['nfe']:<4} "
            f"mean_tasks={r['mean_tasks']:.3f}±{(r['mean_tasks_std'] or 0):.3f}  "
            f"p3={r['p3']}  p4={r['p4']}  "
            f"lat_ms={r['latency_ms']}"
        )
    lines.extend(
        [
            "",
            "-" * 78,
            "CLAIM #2 — Deterministic ODE vs stochastic SDE (sampling-seed std)",
            "-" * 78,
            v2,
            "",
            "Note: this is std across sampling RNG seeds on the SAME checkpoint,",
            "not training-seed std from the original 900-episode eval.",
            "",
            "-" * 78,
            "Appendix: why the original eval could not answer these claims",
            "-" * 78,
            "",
        ]
    )
    # Include a short pointer; full text is in baseline_report.txt
    lines.append("See baseline_report.txt in this directory for the full write-up.")
    lines.append("")
    lines.append("=" * 78)
    lines.append("VERDICT SUMMARY")
    lines.append("=" * 78)
    lines.append(f"Claim #1 (quality @ low NFE): {v1.splitlines()[0]}")
    lines.append(f"Claim #2 (lower sampling variance): {v2.splitlines()[0]}")
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.txt"
    path.write_text("\n".join(lines) + "\n")

    # Also dump machine-readable verdicts
    verdict_json = {
        "claim1": v1.splitlines()[0],
        "claim2": v2.splitlines()[0],
        "claim1_detail": v1,
        "claim2_detail": v2,
        "n_runs": len(runs),
        "nfe_rows": nfe_rows,
        "variance_rows": [
            {k: v for k, v in r.items() if k != "per_task"} for r in var_rows
        ],
    }
    with open(out_dir / "verdicts.json", "w") as f:
        json.dump(verdict_json, f, indent=2, sort_keys=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_root_dp",
        type=Path,
        default=Path("diffusion_policy/data/kitchen_eval_nfe"),
    )
    ap.add_argument(
        "--input_root_fp",
        type=Path,
        default=Path("kripsy12/FlowPolicy/data/kitchen_eval_nfe/flowpolicy"),
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/kitchen_eval_plots/nfe_variance"),
    )
    ap.add_argument(
        "--experiment_root",
        type=Path,
        default=None,
    )
    args = ap.parse_args()
    experiment_root = args.experiment_root or Path(__file__).resolve().parents[1]

    runs = load_all(args.input_root_dp, args.input_root_fp)
    nfe_rows = nfe_curve_rows(runs)
    var_rows = variance_rows(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "success_vs_nfe.csv",
        nfe_rows,
        [
            "model",
            "nfe",
            "n_runs",
            "mean_tasks",
            "mean_tasks_std",
            "p3",
            "p3_std",
            "p4",
            "p4_std",
            "latency_ms",
            "latency_ms_std",
        ],
    )
    write_csv(
        args.output_dir / "sampling_variance.csv",
        [
            {
                "model": r["model"],
                "nfe": r["nfe"],
                "n_sampling_seeds": r["n_sampling_seeds"],
                "mean_tasks_mean": r["mean_tasks_mean"],
                "mean_tasks_std": r["mean_tasks_std"],
                "p3_mean": r["p3_mean"],
                "p3_std": r["p3_std"],
                "p4_mean": r["p4_mean"],
                "p4_std": r["p4_std"],
                "mean_per_task_std": r["mean_per_task_std"],
            }
            for r in var_rows
        ],
        [
            "model",
            "nfe",
            "n_sampling_seeds",
            "mean_tasks_mean",
            "mean_tasks_std",
            "p3_mean",
            "p3_std",
            "p4_mean",
            "p4_std",
            "mean_per_task_std",
        ],
    )
    plot_nfe(nfe_rows, args.output_dir / "success_vs_nfe.png")
    plot_variance(var_rows, args.output_dir / "sampling_variance.png")
    report = write_report(
        args.output_dir, runs, nfe_rows, var_rows, experiment_root
    )
    print(f"Wrote {report}")
    print(f"Runs analyzed: {len(runs)}")
    if not runs:
        print("WARNING: no runs found — sweep has not produced eval_metrics.json yet.")


if __name__ == "__main__":
    main()
