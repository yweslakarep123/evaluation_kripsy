#!/usr/bin/env python3
"""Transition heatmaps P(next=B | completed=A) per NFE / operating point.

1) Equal-NFE panels: NFE in {1, 8, 32, 100}, all three models at the same NFE.
2) Operating-point panels:
   - FlowPolicy@8 vs DP-CNN@100 vs DP-Transformer@100
   - FlowPolicy@1 vs DP-CNN@100 vs DP-Transformer@100

Outputs (PNG + PDF) under:
  data/kitchen_eval_plots/nfe100/why_transition_by_nfe/

Usage:
  python scripts/plot_transition_heatmap_by_nfe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_kitchen_completion_order as why  # noqa: E402

OUT_DIR = ROOT / "data/kitchen_eval_plots/nfe100/why_transition_by_nfe"
FP_ROOT = ROOT / "kripsy12/FlowPolicy/data/kitchen_eval_nfe100/flowpolicy"
DP_CANDIDATES = [
    ROOT / "diffusion_policy/data/kitchen_eval_nfe100_diffusion",
    ROOT / "diffusion_policy/data/kitchen_eval_nfe100",
]

MODEL_ORDER = ["FlowPolicy", "DP-CNN", "DP-Transformer"]
COLORS = {
    "FlowPolicy": "#2ca02c",
    "DP-CNN": "#1f77b4",
    "DP-Transformer": "#ff7f0e",
}
NFES = (1, 8, 32, 100)


def find_dp_root() -> Path:
    for p in DP_CANDIDATES:
        if p.is_dir() and any(p.rglob("eval_metrics.json")):
            return p
    raise FileNotFoundError(f"No DP nfe100 root in {DP_CANDIDATES}")


def seed_dirs(model_key: str, nfe: int, dp_root: Path) -> list[Path]:
    if model_key == "FlowPolicy":
        return [
            FP_ROOT / f"seed_baseline_{s}_nfe{nfe}_sseed0" for s in (42, 43, 44)
        ]
    mid = (
        "diffusion_policy_cnn"
        if model_key == "DP-CNN"
        else "diffusion_policy_transformer"
    )
    return [
        dp_root / mid / f"seed_train{s}_nfe{nfe}_sseed0" for s in (0, 1, 2)
    ]


def plot_transition_heatmaps(
    stats: dict[str, dict],
    panel_order: list[str],
    out_dir: Path,
    out_stem: str,
    title: str,
    subplot_titles: dict[str, str] | None = None,
) -> None:
    """Side-by-side heatmaps for an arbitrary panel order."""
    n = len(panel_order)
    fig_w = max(5.0 * n, 10.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.8), sharey=True)
    if n == 1:
        axes = [axes]
    im = None
    for ax, key in zip(axes, panel_order):
        mat = stats[key]["trans"] * 100.0
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=80, aspect="auto")
        ax.set_xticks(range(len(stats[key]["next_labels"])))
        ax.set_xticklabels(
            [why.SHORT.get(t, t) for t in stats[key]["next_labels"]],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(range(len(why.TASKS)))
        ax.set_yticklabels([why.SHORT[t] for t in why.TASKS], fontsize=8)
        ax.set_xlabel("Next task (or STOP)")
        ax.set_title(
            subplot_titles[key] if subplot_titles else key,
            fontsize=11,
        )
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
    fig.suptitle(title, fontsize=12)
    why._save(fig, out_dir / out_stem)


def load_stats_panels(
    panels: list[tuple[str, str, int]],
    dp_root: Path,
) -> dict[str, dict]:
    """panels: list of (panel_key, model_name, nfe)."""
    specs = {}
    for key, model, nfe in panels:
        specs[key] = {
            "seed_dirs": seed_dirs(model, nfe, dp_root),
            "color": COLORS[model],
        }
    why.MODEL_SPECS = specs
    why.MODEL_ORDER = [key for key, _, _ in panels]

    for key, model, nfe in panels:
        for d in specs[key]["seed_dirs"]:
            if not (d / "eval_metrics.json").is_file():
                raise FileNotFoundError(d / "eval_metrics.json")
        print(f"  {key}: {model} @NFE={nfe}")

    return {key: why.load_model_stats(key) for key, _, _ in panels}


def load_stats_for_nfe_map(
    nfe_map: dict[str, int],
    dp_root: Path,
) -> dict[str, dict]:
    panels = [(name, name, nfe_map[name]) for name in MODEL_ORDER]
    return load_stats_panels(panels, dp_root)


def main() -> None:
    dp_root = find_dp_root()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"FP root: {FP_ROOT}")
    print(f"DP root: {dp_root}")
    print(f"Output: {OUT_DIR}")

    # Equal-NFE panels (3 models)
    for nfe in NFES:
        print(f"\nEqual NFE={nfe}")
        nfe_map = {name: nfe for name in MODEL_ORDER}
        stats = load_stats_for_nfe_map(nfe_map, dp_root)
        plot_transition_heatmaps(
            stats,
            MODEL_ORDER,
            OUT_DIR,
            out_stem=f"02_transition_heatmap_nfe{nfe}",
            title=(
                f"P(next = B | completed = A) @ NFE={nfe} — "
                "chaining is the main FP vs DP difference"
            ),
        )

    # Operating-point panels (FP low-NFE vs DP@100)
    operating = [
        (
            {"FlowPolicy": 8, "DP-CNN": 100, "DP-Transformer": 100},
            "02_transition_heatmap_fp8_vs_dp100",
            "P(next = B | completed = A) — FlowPolicy@8 vs DP@100",
            {
                "FlowPolicy": "FlowPolicy @ NFE=8",
                "DP-CNN": "DP-CNN @ NFE=100",
                "DP-Transformer": "DP-Transformer @ NFE=100",
            },
        ),
        (
            {"FlowPolicy": 1, "DP-CNN": 100, "DP-Transformer": 100},
            "02_transition_heatmap_fp1_vs_dp100",
            "P(next = B | completed = A) — FlowPolicy@1 vs DP@100",
            {
                "FlowPolicy": "FlowPolicy @ NFE=1",
                "DP-CNN": "DP-CNN @ NFE=100",
                "DP-Transformer": "DP-Transformer @ NFE=100",
            },
        ),
    ]
    for nfe_map, stem, title, subplot_titles in operating:
        print(f"\nOperating: {stem}")
        stats = load_stats_for_nfe_map(nfe_map, dp_root)
        plot_transition_heatmaps(
            stats,
            MODEL_ORDER,
            OUT_DIR,
            out_stem=stem,
            title=title,
            subplot_titles=subplot_titles,
        )

    # FlowPolicy across all NFEs
    print("\nFlowPolicy all NFEs")
    fp_panels = [(f"FP@NFE={nfe}", "FlowPolicy", nfe) for nfe in NFES]
    fp_keys = [k for k, _, _ in fp_panels]
    fp_stats = load_stats_panels(fp_panels, dp_root)
    plot_transition_heatmaps(
        fp_stats,
        fp_keys,
        OUT_DIR,
        out_stem="02_transition_heatmap_flowpolicy_all_nfe",
        title="P(next = B | completed = A) — FlowPolicy across NFE",
        subplot_titles={k: k for k in fp_keys},
    )

    # DP-CNN + DP-Transformer at NFE=100
    print("\nDP-CNN & DP-Transformer @ NFE=100")
    dp_panels = [
        ("DP-CNN @ NFE=100", "DP-CNN", 100),
        ("DP-Transformer @ NFE=100", "DP-Transformer", 100),
    ]
    dp_keys = [k for k, _, _ in dp_panels]
    dp_stats = load_stats_panels(dp_panels, dp_root)
    plot_transition_heatmaps(
        dp_stats,
        dp_keys,
        OUT_DIR,
        out_stem="02_transition_heatmap_dp_nfe100",
        title="P(next = B | completed = A) — DP-CNN & DP-Transformer @ NFE=100",
        subplot_titles={k: k for k in dp_keys},
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
