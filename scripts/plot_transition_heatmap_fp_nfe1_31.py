#!/usr/bin/env python3
"""Transition heatmaps: FlowPolicy NFE 1–31 (3 per image) + FP@3 vs DP@100.

Each image has 3 side-by-side heatmaps:
  - FlowPolicy NFE 1–3, 4–6, …, 28–30, then NFE 31 alone
  - FlowPolicy@3 vs DP-CNN@100 vs DP-Transformer@100

Outputs under:
  data/kitchen_eval_plots/nfe100/why_transition_by_nfe/

Usage:
  python scripts/plot_transition_heatmap_fp_nfe1_31.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_transition_heatmap_by_nfe as th  # noqa: E402

OUT_DIR = ROOT / "data/kitchen_eval_plots/nfe100/why_transition_by_nfe"


def fp_triplet_groups(nfe_min: int = 1, nfe_max: int = 31) -> list[list[int]]:
    """Group NFEs into consecutive triplets: [1,2,3], [4,5,6], …, [31]."""
    groups: list[list[int]] = []
    cur: list[int] = []
    for nfe in range(nfe_min, nfe_max + 1):
        cur.append(nfe)
        if len(cur) == 3:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def main() -> None:
    dp_root = th.find_dp_root()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"FP root: {th.FP_ROOT}")
    print(f"DP root: {dp_root}")
    print(f"Output: {OUT_DIR}")

    # ── FlowPolicy NFE 1–31, three heatmaps per image ───────────────────
    groups = fp_triplet_groups(1, 31)
    print(f"\nFlowPolicy NFE 1–31 → {len(groups)} images")
    for nfes in groups:
        label = (
            f"NFE{nfes[0]}-{nfes[-1]}"
            if len(nfes) > 1
            else f"NFE{nfes[0]}"
        )
        print(f"\n  Group {label}: {nfes}")
        panels = [(f"FP@NFE={n}", "FlowPolicy", n) for n in nfes]
        keys = [k for k, _, _ in panels]
        stats = th.load_stats_panels(panels, dp_root)
        th.plot_transition_heatmaps(
            stats,
            keys,
            OUT_DIR,
            out_stem=f"02_transition_heatmap_flowpolicy_{label.lower()}",
            title=(
                f"P(next = B | completed = A) — FlowPolicy "
                f"NFE {', '.join(str(n) for n in nfes)}"
            ),
            subplot_titles={k: k for k in keys},
        )

    # ── FlowPolicy@3 vs DP@100 ──────────────────────────────────────────
    print("\nOperating: FlowPolicy@3 vs DP@100")
    nfe_map = {"FlowPolicy": 3, "DP-CNN": 100, "DP-Transformer": 100}
    stats = th.load_stats_for_nfe_map(nfe_map, dp_root)
    th.plot_transition_heatmaps(
        stats,
        th.MODEL_ORDER,
        OUT_DIR,
        out_stem="02_transition_heatmap_fp3_vs_dp100",
        title="P(next = B | completed = A) — FlowPolicy@3 vs DP@100",
        subplot_titles={
            "FlowPolicy": "FlowPolicy @ NFE=3",
            "DP-CNN": "DP-CNN @ NFE=100",
            "DP-Transformer": "DP-Transformer @ NFE=100",
        },
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
