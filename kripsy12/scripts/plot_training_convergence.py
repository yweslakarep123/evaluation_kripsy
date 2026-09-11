#!/usr/bin/env python3
"""Overlay kurva training 3 seed dan ringkasan apakah konvergensi comparable.

Contoh:
  python3 scripts/plot_training_convergence.py --output-dir outputs/experiment
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FLOWPOLICY_ROOT = Path(__file__).resolve().parents[1] / "FlowPolicy"
import sys

sys.path.insert(0, str(FLOWPOLICY_ROOT))

from flow_policy_3d.common.training_curves import (  # noqa: E402
    compute_convergence,
    load_history,
)

_RUN_RE = re.compile(
    r"^(?P<group>baseline|hb_best|cfg-?\d+)_seed(?P<seed>\d+)_(?P<profile>.+)$"
)
_CV_COMPARABLE = 0.15


def _save(fig, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _parse_run_dir(run_dir: Path) -> Optional[Tuple[str, int, str]]:
    m = _RUN_RE.match(run_dir.name)
    if not m:
        return None
    return m.group("group"), int(m.group("seed")), m.group("profile")


def _cv(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    if abs(mean) < 1e-12:
        return None
    return statistics.pstdev(values) / abs(mean)


def collect_runs(runs_root: Path) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    if not runs_root.is_dir():
        return collected
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = _parse_run_dir(run_dir)
        hist = load_history(str(run_dir))
        if not hist:
            continue
        summary = compute_convergence(hist)
        group, seed, profile = parsed if parsed else ("other", -1, run_dir.name)
        collected.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "group": group,
                "seed": seed,
                "profile": profile,
                "history": hist,
                "summary": summary,
            }
        )
    return collected


def plot_group_overlay(runs: List[Dict[str, Any]], out_dir: Path, title: str) -> None:
    if not runs:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for item in runs:
        hist = item["history"]
        epochs = [int(r["epoch"]) for r in hist]
        train = [r.get("train_loss") for r in hist]
        val = [r.get("val_loss") for r in hist]
        label = f"seed {item['seed']}"
        axes[0].plot(epochs, train, linewidth=1.3, label=label)
        if any(v is not None for v in val):
            axes[1].plot(epochs, val, linewidth=1.3, label=label)
        best_ep = item["summary"].get("best_val_epoch")
        best_loss = item["summary"].get("best_val_loss")
        if best_ep is not None and best_loss is not None:
            axes[1].scatter([best_ep], [best_loss], s=28, zorder=5)
    axes[0].set_ylabel("train_loss")
    axes[0].set_title(f"{title} — train loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("val_loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_title(f"{title} — val loss (dot = best val)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    _save(fig, out_dir / f"{slug}_overlay")


def summarize_comparability(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    best_losses = []
    final_vals = []
    best_epochs = []
    plateau_flags = []
    per_seed = []
    for item in runs:
        s = item["summary"]
        row = {
            "seed": item["seed"],
            "run_name": item["run_name"],
            "best_val_loss": s.get("best_val_loss"),
            "best_val_epoch": s.get("best_val_epoch"),
            "final_val_loss": s.get("final_val_loss"),
            "final_train_loss": s.get("final_train_loss"),
            "plateaued": bool(s.get("plateaued")),
            "relative_val_improvement": s.get("relative_val_improvement"),
            "n_epochs": s.get("n_epochs"),
        }
        per_seed.append(row)
        if s.get("best_val_loss") is not None:
            best_losses.append(float(s["best_val_loss"]))
        if s.get("final_val_loss") is not None:
            final_vals.append(float(s["final_val_loss"]))
        if s.get("best_val_epoch") is not None:
            best_epochs.append(int(s["best_val_epoch"]))
        plateau_flags.append(bool(s.get("plateaued")))

    cv_best = _cv(best_losses)
    cv_final = _cv(final_vals)
    all_plateaued = bool(plateau_flags) and all(plateau_flags)
    comparable = (
        len(best_losses) >= 2
        and cv_best is not None
        and cv_best <= _CV_COMPARABLE
    )
    return {
        "n_runs": len(runs),
        "seeds": [item["seed"] for item in runs],
        "per_seed": per_seed,
        "cv_best_val_loss": cv_best,
        "cv_final_val_loss": cv_final,
        "best_val_epoch_min": min(best_epochs) if best_epochs else None,
        "best_val_epoch_max": max(best_epochs) if best_epochs else None,
        "all_plateaued": all_plateaued,
        "comparable": comparable,
        "comparable_criterion": (
            f"CV(best_val_loss) <= {_CV_COMPARABLE} across seeds "
            "(lower CV = more comparable convergence)"
        ),
        "checkpoint_selection_criterion": "min_val_loss",
        "checkpoint_used_for_eval": "best_val.ckpt (fallback latest.ckpt)",
    }


def plot_experiment_convergence(output_dir: str) -> Dict[str, Any]:
    out_root = Path(output_dir)
    runs_root = out_root / "runs"
    plot_dir = out_root / "plots" / "seed_convergence"
    collected = collect_runs(runs_root)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for item in collected:
        key = (str(item["group"]), str(item["profile"]))
        groups.setdefault(key, []).append(item)

    report: Dict[str, Any] = {
        "output_dir": str(out_root),
        "n_runs_with_history": len(collected),
        "groups": {},
    }
    for (group, profile), runs in sorted(groups.items()):
        title = f"{group} / {profile}"
        plot_group_overlay(runs, plot_dir, title)
        report["groups"][f"{group}_{profile}"] = summarize_comparability(runs)

    report_path = out_root / "seed_convergence.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[plot_training_convergence] {len(collected)} run → {plot_dir}")
    print(f"[plot_training_convergence] ringkasan → {report_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay kurva loss antar-seed dan cek konvergensi comparable."
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Folder eksperimen (berisi runs/).",
    )
    args = ap.parse_args()
    plot_experiment_convergence(args.output_dir)


if __name__ == "__main__":
    main()
