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


def plot_tradeoff_scatter(df_ok: pd.DataFrame, out_dir: Path):
    """Satu titik per (profile, cfg_idx); error bar = std lintas seed."""
    lat_c = _pick_col(df_ok, "test_mean_inference_latency_ms", "mean_inference_latency_ms")
    k7_c = _pick_col(df_ok, "test_p7", "test_all_7_success")
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"standard": "o", "minimal": "s"}
    profiles = df_ok["profile"].unique()
    cfg_list = sorted(df_ok["cfg_idx"].unique())
    for profile in profiles:
        for cfg_idx in cfg_list:
            sub = df_ok[
                (df_ok["profile"] == profile) & (df_ok["cfg_idx"] == cfg_idx)
            ]
            if len(sub) < 1:
                continue
            per_seed_lat = sub.groupby("seed")[lat_c].mean()
            per_seed_sr = sub.groupby("seed")[k7_c].mean()
            mx = float(per_seed_lat.mean())
            my = float(per_seed_sr.mean())
            ex = float(per_seed_lat.std(ddof=0)) if len(per_seed_lat) > 1 else 0.0
            ey = float(per_seed_sr.std(ddof=0)) if len(per_seed_sr) > 1 else 0.0
            ax.errorbar(
                mx,
                my,
                xerr=ex,
                yerr=ey,
                fmt=markers.get(str(profile), "o"),
                color=plt.cm.tab10(int(cfg_idx) % 10),
                label=f"{profile} cfg{int(cfg_idx)}",
                alpha=0.8,
                capsize=2,
            )
    ax.set_xlabel(lat_c)
    ax.set_ylabel(f"{k7_c} (%)")
    ax.set_title("Trade-off scatter (titik = cfg×profile; batang = std seeds)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), fontsize=7)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir / "tradeoff_scatter")


def plot_success_bars(summary: pd.DataFrame, results_ok: pd.DataFrame, out_dir: Path):
    """Top-10 cfg_idx by trade_off_mean per profile — batang k1–k4."""
    metric_cols = _success_mean_cols(summary)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, profile in zip(axes, ["standard", "minimal"]):
        sub = summary[summary["profile"] == profile]
        if sub.empty:
            ax.set_visible(False)
            continue
        top = sub.nlargest(10, "trade_off_mean")["cfg_idx"].tolist()
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
        ax.set_xlabel("cfg_idx (top-10 trade_off)")
        ax.set_ylabel("success rate mean (%)")
        ax.set_title(f"{profile}")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("Success rate per sub-task (from summary.csv)")
    plt.tight_layout()
    _save(fig, out_dir / "success_rate_bar")


def plot_cv_box(summary: pd.DataFrame, results_ok: pd.DataFrame, out_dir: Path):
    """Top-5 cfg_idx: kotak distribusi success_rate_k4 (varians antar seed)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    top_cfg = summary.groupby("cfg_idx")["trade_off_mean"].mean().nlargest(5).index.tolist()
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
    ax.set_title("Seed variance (top-5 cfg_idx by trade_off)")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_dir / "cv_fold_variance")


def plot_hparam_sensitivity(results_ok: pd.DataFrame, out_dir: Path):
    hp_keys = list(SEARCH_SPACE.keys())
    profiles = ["standard", "minimal"]
    lat_c = _pick_col(results_ok, "test_mean_inference_latency_ms", "mean_inference_latency_ms")
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
            sub["to"] = np.where(
                sub[lat_c].astype(float) > 1e-9,
                sub[k7_c].astype(float) / sub[lat_c].astype(float),
                np.nan,
            )
            g = sub.groupby(hp, as_index=False)["to"].mean().sort_values(hp)
            ax.plot(
                g[hp].astype(str),
                g["to"].values,
                marker="o",
                label=profile,
                alpha=0.8,
            )
        ax.set_xlabel(hp)
        ax.set_ylabel("mean trade_off")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for j in range(len(hp_keys), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Hyperparameter sensitivity (mean trade_off)")
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

    lat_c = _pick_col(df_ok, "test_mean_inference_latency_ms", "mean_inference_latency_ms")
    k7_c = _pick_col(df_ok, "test_p7", "test_all_7_success")
    for c in [k7_c, lat_c, "cfg_idx"]:
        df_ok[c] = pd.to_numeric(df_ok[c], errors="coerce")

    plot_tradeoff_scatter(df_ok, plots)

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
