#!/usr/bin/env python3
"""Miss vs p4 success head-to-head: FlowPolicy@NFE3 vs DP@NFE100.

Classification (p4 ceiling):
  miss_subtask : k < 4
  success_p4   : k >= 4

Usage:
    python scripts/plot_fail_vs_timeout_fp3_vs_dp100.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kitchen_eval_paths import (  # noqa: E402
    PLOT_ROOT,
    dp_seed_dirs,
    fp_seed_dirs,
    resolve_dp_root,
    resolve_fp_root,
)
from kitchen_eval_stats import classify_p4_outcome, episode_k_capped  # noqa: E402

OUT_DIR = PLOT_ROOT / "fail_vs_timeout"

HEAD2HEAD = [
    ("FlowPolicy", 3, "FP\nNFE=3"),
    ("DP-CNN", 100, "CNN\nNFE=100"),
    ("DP-Transformer", 100, "Trans\nNFE=100"),
]
STEM = "03_fail_vs_timeout_fp3_vs_dp100"


def find_dp_root() -> Path:
    return resolve_dp_root()


def seed_dirs(model_key: str, nfe: int, dp_root: Path) -> List[Path]:
    if model_key == "FlowPolicy":
        found = fp_seed_dirs(nfe, fp_root=resolve_fp_root())
        if found:
            return found
        root = resolve_fp_root()
        return [root / f"seed_seed{s}_nfe{nfe}_sseed0" for s in (42, 43, 44)]
    mid = (
        "diffusion_policy_cnn"
        if model_key == "DP-CNN"
        else "diffusion_policy_transformer"
    )
    return dp_seed_dirs(mid, nfe, dp_root=dp_root)


def classify(k: int) -> str:
    return classify_p4_outcome(k)


def load_counts(model: str, nfe: int, dp_root: Path) -> Tuple[Counter, int]:
    outcome: Counter = Counter()
    n = 0
    for d in seed_dirs(model, nfe, dp_root):
        mj = d / "eval_metrics.json"
        if not mj.exists():
            raise FileNotFoundError(mj)
        for ep in json.loads(mj.read_text())["episodes"]:
            k = episode_k_capped(ep)
            outcome[classify(k)] += 1
            n += 1
    return outcome, n


def plot_stacked(rows: List[Dict], out_dir: Path) -> None:
    categories = [
        ("miss_subtask_n", "Miss subtask (k<4)", "#d62728"),
        ("success_p4_n", "Success p4 (k≥4)", "#2ca02c"),
    ]
    lookup = {(r["model"], r["nfe"]): r for r in rows}
    x = np.arange(len(HEAD2HEAD))
    width = 0.62
    bottoms = np.zeros(len(HEAD2HEAD))

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for key, label, color in categories:
        vals = np.array(
            [
                lookup[(m, nfe)][key] / lookup[(m, nfe)]["n_episodes"]
                for m, nfe, _ in HEAD2HEAD
            ]
        )
        ax.bar(
            x,
            vals,
            width,
            bottom=bottoms,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.6,
        )
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            row = lookup[(HEAD2HEAD[i][0], HEAD2HEAD[i][1])]
            n = int(row[key])
            if v >= 0.04:
                ax.text(
                    i,
                    b + v / 2,
                    f"{100 * v:.0f}%\n(n={n})",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
            elif n > 0:
                ax.text(
                    i,
                    b + v + 0.02,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )
        bottoms = bottoms + vals

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, _, lab in HEAD2HEAD])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Episode fraction")
    ax.axhline(1.0, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.14))
    ax.set_title(
        "Miss vs p4 success: FlowPolicy@NFE3 vs DP@NFE100\n"
        "(300 eval episodes each; completions after k=4 are not scored)",
        pad=28,
    )
    fig.tight_layout()
    stem = out_dir / STEM
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.png')}")
    print(f"Wrote {stem.with_suffix('.pdf')}")


def main() -> None:
    dp_root = find_dp_root()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    print(f"{'model':16} {'nfe':>4} {'miss':>6} {'p4':>6} {'n':>5}")
    for model, nfe, _ in HEAD2HEAD:
        outcome, n = load_counts(model, nfe, dp_root)
        miss = outcome.get("miss_subtask", 0)
        ok = outcome.get("success_p4", 0)
        print(f"{model:16} {nfe:4} {miss:6} {ok:6} {n:5}")
        rows.append(
            {
                "model": model,
                "nfe": nfe,
                "n_episodes": n,
                "miss_subtask_n": miss,
                "miss_subtask_pct": round(100.0 * miss / n, 2) if n else 0.0,
                "success_p4_n": ok,
                "success_p4_pct": round(100.0 * ok / n, 2) if n else 0.0,
            }
        )

    csv_path = OUT_DIR / f"{STEM}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    plot_stacked(rows, OUT_DIR)


if __name__ == "__main__":
    main()
