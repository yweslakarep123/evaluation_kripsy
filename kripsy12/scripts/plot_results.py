#!/usr/bin/env python3
"""Plot perbandingan dari results.csv dan summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiment_constants import SEARCH_SPACE  # noqa: E402


def _pick_col(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    return preferred if preferred in df.columns else fallback


def _success_mean_cols(summary: pd.DataFrame) -> list[str]:
    cols = []
    for i in range(1, 5):
        tp = f"test_success_rate_k{i}_mean"
        lp = f"success_rate_k{i}_mean"
        cols.append(tp if tp in summary.columns else lp)
    return cols


def _save(fig, path_base: Path):
    fig.savefig(path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _rank_mean_col(summary: pd.DataFrame) -> str:
    for c in (
        "test_p7_mean",
        "test_all_7_success_mean",
        "success_rate_total_mean",
    ):
        if c in summary.columns:
            return c
    raise KeyError("Tidak ada kolom success rate mean di summary.csv")


def plot_success_bars(summary: pd.DataFrame, results_ok: pd.DataFrame, out_dir: Path):
    """Top-10 cfg_idx by success rate mean per profile — batang k1–k4."""
    metric_cols = _success_mean_cols(summary)
    rank_c = _rank_mean_col(summary)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, profile in zip(axes, ["standard", "minimal"]):
        sub = summary[summary["profile"] == profile]
        if sub.empty:
            ax.set_visible(False)
            continue
        top = sub.nlargest(10, rank_c)["cfg_idx"].tolist()
        x = np.arange(len(top))
        width = 0.2
        for i, mc in enumerate(metric_cols):
            vals = []
            for c in top:
                row = sub[sub["cfg_idx"] == c]
                if row.empty:
                    vals.append(0)
                else:
                    vals.append(float(row.iloc[0][mc]))
            ax.bar(x + (i - 1.5) * width, vals, width, label=mc.replace("_mean", ""))
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(c)) for c in top])
        ax.set_xlabel(f"cfg_idx (top-10 {rank_c})")
        ax.set_ylabel("success rate mean (%)")
        ax.set_title(f"{profile}")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("Success rate per sub-task (from summary.csv)")
    plt.tight_layout()
    _save(fig, out_dir / "success_rate_bar")


def plot_cv_box(summary: pd.DataFrame, results_ok: pd.DataFrame, out_dir: Path):
    """Top-5 cfg_idx: kotak distribusi success rate (varians antar seed)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    rank_c = _rank_mean_col(summary)
    top_cfg = summary.groupby("cfg_idx")[rank_c].mean().nlargest(5).index.tolist()
    positions = []
    data = []
    colors = []
    pos = 0
    cmap = {"standard": "#1f77b4", "minimal": "#ff7f0e"}
    k7_c = _pick_col(results_ok, "test_p7", "test_all_7_success")
    for cfg_idx in top_cfg:
        for profile in ["standard", "minimal"]:
            sub = results_ok[
                (results_ok["cfg_idx"] == cfg_idx)
                & (results_ok["profile"] == profile)
            ]
            if len(sub) < 2:
                continue
            data.append(sub[k7_c].astype(float).values)
            positions.append(pos)
            colors.append(cmap.get(profile, "gray"))
            pos += 1
        pos += 0.5
    if data:
        bp = ax.boxplot(data, positions=positions, widths=0.35, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
    ax.set_ylabel(k7_c)
    ax.set_title("Seed variance (top-5 cfg_idx by success rate)")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_dir / "cv_fold_variance")


def plot_hparam_sensitivity(results_ok: pd.DataFrame, out_dir: Path):
    hp_keys = list(SEARCH_SPACE.keys())
    profiles = ["standard", "minimal"]
    k7_c = _pick_col(results_ok, "test_p7", "test_all_7_success")
    n_hp = len(hp_keys)
    ncols = 3
    nrows = int(np.ceil(n_hp / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes).ravel()
    for ax_i, hp in enumerate(hp_keys):
        ax = axes[ax_i]
        for profile in profiles:
            sub = results_ok[results_ok["profile"] == profile].copy()
            if sub.empty:
                continue
            sub[hp] = pd.to_numeric(sub[hp], errors="coerce")
            sub[k7_c] = pd.to_numeric(sub[k7_c], errors="coerce")
            g = sub.groupby(hp, as_index=False)[k7_c].mean().sort_values(hp)
            ax.plot(
                g[hp].astype(str),
                g[k7_c].values,
                marker="o",
                label=profile,
                alpha=0.8,
            )
        ax.set_xlabel(hp)
        ax.set_ylabel(f"mean {k7_c}")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for j in range(len(hp_keys), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"Hyperparameter sensitivity (mean {k7_c})")
    plt.tight_layout()
    _save(fig, out_dir / "hyperparam_sensitivity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default="outputs/experiment")
    ap.add_argument(
        "--results-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Jalur results.csv (default: <output-dir>/results.csv). Relatif ke akar repo atau absolut.",
    )
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    out_root = (repo_root / args.output_dir).resolve()
    plots = out_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    if args.results_csv:
        p = Path(args.results_csv)
        res_path = p.resolve() if p.is_absolute() else (repo_root / p).resolve()
    else:
        res_path = out_root / "results.csv"
    sum_path = out_root / "summary.csv"
    if not res_path.is_file():
        print(f"Tidak ada {res_path}")
        return

    df = pd.read_csv(res_path)
    df_ok = df[df["status"] == "ok"].copy()
    if df_ok.empty:
        print("Tidak ada data status=ok untuk plot.")
        return

    k7_c = _pick_col(df_ok, "test_p7", "test_all_7_success")
    for c in [k7_c, "cfg_idx"]:
        df_ok[c] = pd.to_numeric(df_ok[c], errors="coerce")

    if sum_path.is_file():
        summary = pd.read_csv(sum_path)
        plot_success_bars(summary, df_ok, plots)
        plot_cv_box(summary, df_ok, plots)
    else:
        print(f"Lewati plot yang membutuhkan {sum_path}")

    plot_hparam_sensitivity(df_ok, plots)
    print(f"Plot disimpan di {plots}")


if __name__ == "__main__":
    main()
