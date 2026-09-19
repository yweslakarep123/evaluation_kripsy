#!/usr/bin/env python3
"""Multistage p1→p4 degradation: FlowPolicy@NFE3 vs DP@NFE100.

Style matches data/kitchen_eval_plots/nfe100/pk_degradation.png.

Usage:
  python scripts/plot_pk_degradation_fp3_vs_dp100.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/kitchen_eval_plots/nfe100"
OUT_DIR = DATA_DIR

KEEP: set[Tuple[str, int]] = {
    ("flowpolicy", 3),
    ("diffusion_policy_cnn", 100),
    ("diffusion_policy_transformer", 100),
}

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
# Match original NFE marker language: 3 → use triangle (new), 100 → diamond
NFE_MARKER = {3: "^", 100: "D"}
MODEL_ORDER = (
    "flowpolicy",
    "diffusion_policy_cnn",
    "diffusion_policy_transformer",
)
STAGES = ("p1", "p2", "p3", "p4")
PLOT_ORDER = (
    ("flowpolicy", 3),
    ("diffusion_policy_cnn", 100),
    ("diffusion_policy_transformer", 100),
)


def load_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with (DATA_DIR / "summary.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            key = (r["model"], int(r["nfe"]))
            if key not in KEEP:
                continue
            row: Dict[str, Any] = {
                "model": r["model"],
                "nfe": int(r["nfe"]),
                "n_seeds": int(float(r["n_seeds"])),
            }
            for s in STAGES:
                row[s] = float(r[s])
                row[f"{s}_std"] = float(r[f"{s}_std"] or 0.0)
            rows.append(row)
    missing = KEEP - {(r["model"], r["nfe"]) for r in rows}
    if missing:
        raise SystemExit(f"Missing summary rows for: {sorted(missing)}")
    return rows


def write_logs(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    by_key = {(r["model"], r["nfe"]): r for r in rows}
    csv_rows: List[Dict[str, Any]] = []
    lines = [
        "Multistage success rate p1→p4 (mean ± std across 3 training seeds)",
        "Operating points: FlowPolicy @ NFE=3, DP-CNN/DP-Transformer @ NFE=100",
        "=" * 78,
        "",
        f"{'model':<18} {'NFE':>4} "
        + " ".join(f"{s + '_mean':>10} {s + '_std':>9}" for s in STAGES),
        "-" * 78,
    ]
    for model, nfe in PLOT_ORDER:
        row = by_key[(model, nfe)]
        csv_row: Dict[str, Any] = {
            "model": model,
            "nfe": nfe,
            "n_seeds": row["n_seeds"],
        }
        parts = [f"{LABEL[model]:<18} {nfe:>4}"]
        compact = []
        for s in STAGES:
            mu = float(row[s])
            sd = float(row[f"{s}_std"])
            parts.append(f"{mu:10.4f} {sd:9.4f}")
            csv_row[f"{s}_mean"] = mu
            csv_row[f"{s}_std"] = sd
            compact.append(f"{s}={mu:.3f}±{sd:.3f}")
        lines.append(" ".join(parts))
        lines.append("    " + "  ".join(compact))
        csv_rows.append(csv_row)
    lines.append("")

    stem = "pk_degradation_fp3_vs_dp100"
    (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    with (out_dir / f"{stem}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)


def plot(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    by_key = {(r["model"], r["nfe"]): r for r in rows}
    xs = np.arange(len(STAGES))
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    for model, nfe in PLOT_ORDER:
        row = by_key[(model, nfe)]
        ys = [float(row[s]) for s in STAGES]
        yerr = [float(row[f"{s}_std"]) for s in STAGES]
        ax.errorbar(
            xs,
            ys,
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
            label=f"{LABEL[model]} NFE={nfe}",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(list(STAGES))
    ax.set_ylabel("Multistage success rate (mean ± std)")
    ax.set_xlabel("Stage")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(
        "Multistage degradation p1→p4 (FP@3, DP@100)\n"
        "mean ± std across 3 training seeds"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    model_handles = [
        Line2D([0], [0], color=COLOR[m], lw=2, label=LABEL[m]) for m in MODEL_ORDER
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
        for n in (3, 100)
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
    stem = out_dir / "pk_degradation_fp3_vs_dp100"
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem.with_suffix('.png')}")
    print(f"Wrote {stem.with_suffix('.pdf')}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    for r in rows:
        print(
            f"  {LABEL[r['model']]:16} NFE={r['nfe']:<3}  "
            + "  ".join(
                f"{s}={r[s]:.3f}±{r[f'{s}_std']:.3f}" for s in STAGES
            )
        )
    write_logs(rows, OUT_DIR)
    plot(rows, OUT_DIR)


if __name__ == "__main__":
    main()
