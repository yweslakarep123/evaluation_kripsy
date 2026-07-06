#!/usr/bin/env python3
"""Agregasi results.csv → summary.csv (mean±std lintas seed)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(output_dir: Path, *, results_csv: Path | None = None) -> None:
    results_path = results_csv if results_csv is not None else (output_dir / "results.csv")
    if not results_path.is_file():
        print(f"Tidak ada {results_path}")
        return

    df = pd.read_csv(results_path)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("Tidak ada baris status=ok untuk diagregasi.")
        return

    def _col(preferred: str, fallback: str) -> str:
        return preferred if preferred in df.columns else fallback

    k7 = _col("test_p7", "test_all_7_success")
    k4 = _col("test_p4_paper", "test_success_rate_k4")
    k_total = _col("test_all_7_success", "test_success_rate_total", "success_rate_total")
    lat = _col("test_mean_inference_latency_ms", "mean_inference_latency_ms")
    exec_ms = _col("test_mean_episode_duration_ms", "test_mean_execution_time_ms", "mean_execution_time_ms")
    to = _col("test_trade_off", "trade_off")

    metrics = [k_total, k7, k4, lat, exec_ms]
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    df["trade_off_computed"] = np.where(
        df[lat] > 1e-9,
        df[k_total] / df[lat],
        np.where(
            pd.to_numeric(df[to], errors="coerce").notna(),
            pd.to_numeric(df[to], errors="coerce"),
            np.nan,
        ),
    )

    gcols = ["cfg_idx", "profile", "fold"]
    agg_rows = []
    for keys, sub in df.groupby(gcols):
        if isinstance(keys, tuple):
            row = dict(zip(gcols, keys))
        else:
            row = {gcols[0]: keys}
        for m in metrics:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_std"] = float(sub[m].std(ddof=0))
        row["trade_off_mean"] = float(sub["trade_off_computed"].mean())
        row["trade_off_std"] = float(sub["trade_off_computed"].std(ddof=0))
        agg_rows.append(row)

    out = pd.DataFrame(agg_rows)
    out = out.sort_values("trade_off_mean", ascending=False)
    out_path = output_dir / "summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Ditulis {out_path} ({len(out)} baris)")


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
    repo_root = Path(__file__).resolve().parent.parent
    args = ap.parse_args()
    out_dir = (repo_root / args.output_dir).resolve()
    res: Path | None = None
    if args.results_csv:
        p = Path(args.results_csv)
        res = p.resolve() if p.is_absolute() else (repo_root / p).resolve()
    summarize(out_dir, results_csv=res)


if __name__ == "__main__":
    main()
