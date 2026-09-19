#!/usr/bin/env python3
"""Observational kettle-pattern analysis (no retrain, no causal claim).

Decomposes kettle scored success under the p4 ceiling into attempt vs
execution, stratifies by demo kettle rank of the eval init, and compares
transitions to the demonstration prior.

Outputs under data/kitchen_eval_plots/nfe100/kettle_pattern/.

Usage:
  python scripts/analyze_kitchen_kettle_pattern.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    normalize_fp_seed,
    resolve_dp_root,
    resolve_fp_root,
)
from kitchen_eval_stats import clip_episode_record, proportion_stats  # noqa: E402

DEMO_DIR = ROOT / "kripsy12/FlowPolicy/data/kitchen"
DEMO_POS_CSV = ROOT / "data/kitchen_eval_plots/demo_stats/demo_task_positions.csv"
DEMO_TRANS_CSV = ROOT / "data/kitchen_eval_plots/demo_stats/demo_transition_matrix.csv"
PCA_README = PLOT_ROOT / "kettle_action_pca/README.txt"
OUT_DIR = PLOT_ROOT / "kettle_pattern"

DIR_RE = re.compile(r"^seed_(?P<seed>.+)_nfe(?P<nfe>\d+)_sseed(?P<sseed>\d+)$")
DIR_RE_COMPACT = re.compile(r"^seed(?P<seed>\d+)_NFE(?P<nfe>\d+)$", re.IGNORECASE)

TASKS = [
    "microwave",
    "kettle",
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
]
SHORT = {
    "microwave": "MW",
    "kettle": "Kettle",
    "bottom burner": "Bottom",
    "top burner": "Top",
    "light switch": "Light",
    "slide cabinet": "Slide",
    "hinge cabinet": "Hinge",
    "STOP": "STOP",
}
NEXT_LABELS = TASKS + ["STOP"]
QP = 9
OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3
DISP_THR = 0.05
LAPLACE = 0.5
FP_NFES = (1, 8, 100)
CONFIGS = [
    ("FlowPolicy", 1, "FP NFE=1"),
    ("FlowPolicy", 8, "FP NFE=8"),
    ("FlowPolicy", 100, "FP NFE=100"),
    ("DP-CNN", 100, "CNN NFE=100"),
    ("DP-Transformer", 100, "Trans NFE=100"),
]
COLORS = {
    "FlowPolicy": "#2ca02c",
    "DP-CNN": "#1f77b4",
    "DP-Transformer": "#ff7f0e",
}


def shannon_entropy(probs: Sequence[float], eps: float = 1e-12) -> float:
    """Shannon entropy in nats; zero-mass bins are skipped."""
    h = 0.0
    for p in probs:
        q = float(p)
        if q > eps:
            h -= q * math.log(q)
    return h


def kl_divergence(
    p: Sequence[float],
    q: Sequence[float],
    eps: float = 1e-12,
) -> float:
    """KL(p || q) in nats; skip bins where p is ~0."""
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    kl = 0.0
    for pi, qi in zip(p, q):
        a, b = float(pi), float(qi)
        if a <= eps:
            continue
        if b <= eps:
            return float("inf")
        kl += a * math.log(a / b)
    return kl


def laplace_normalize(counts: Sequence[float], alpha: float = LAPLACE) -> List[float]:
    arr = np.asarray(counts, dtype=np.float64)
    sm = arr + float(alpha)
    tot = float(sm.sum())
    if tot <= 0:
        return [1.0 / len(arr)] * len(arr)
    return [float(x) / tot for x in sm]


def find_dp_root() -> Path:
    return resolve_dp_root()


def parse_run_dir(name: str) -> Optional[Tuple[str, int]]:
    m = DIR_RE.match(name)
    if m:
        return m.group("seed"), int(m.group("nfe"))
    m2 = DIR_RE_COMPACT.match(name)
    if m2:
        return m2.group("seed"), int(m2.group("nfe"))
    return None


def seed_dirs(model_key: str, nfe: int, dp_root: Path) -> List[Path]:
    if model_key == "FlowPolicy":
        fp_root = resolve_fp_root()
        found = fp_seed_dirs(nfe, fp_root=fp_root)
        if found:
            return found
        found = []
        for d in sorted(fp_root.iterdir()) if fp_root.is_dir() else []:
            parsed = parse_run_dir(d.name)
            if parsed is None:
                continue
            seed, n = parsed
            if int(n) != int(nfe):
                continue
            if normalize_fp_seed(seed) not in {"42", "43", "44"}:
                continue
            if (d / "eval_metrics.json").is_file():
                found.append(d)
        if found:
            return found
        return [fp_root / f"seed_seed{s}_nfe{nfe}_sseed0" for s in (42, 43, 44)]
    mid = (
        "diffusion_policy_cnn"
        if model_key == "DP-CNN"
        else "diffusion_policy_transformer"
    )
    return dp_seed_dirs(mid, nfe, dp_root=dp_root)


def npz_attempt_valid(z: Any) -> bool:
    if "policy_obs_obj_qp" not in z.files:
        return False
    arr = np.asarray(z["policy_obs_obj_qp"])
    return arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[-1] >= 21


def kettle_attempt_from_npz(z: Any, raw_success: bool) -> Tuple[bool, float]:
    """Physical kettle attempt from obj-qp displacement (NPZ must be valid)."""
    k = np.asarray(z["policy_obs_obj_qp"], dtype=np.float64)[:, -1, 14:21]
    disp = float(np.linalg.norm(k - k[0], axis=1).max()) if len(k) else 0.0
    return bool(raw_success or disp > DISP_THR), disp


def task_attempt_from_npz(z: Any, task: str, raw_success: bool) -> Optional[bool]:
    if not npz_attempt_valid(z):
        return None
    idxs = OBS_ELEMENT_INDICES[task] - QP
    x = np.asarray(z["policy_obs_obj_qp"], dtype=np.float64)[:, -1, idxs]
    if len(x) <= 1:
        return None
    if x.ndim == 1:
        disp = float(np.max(np.abs(x - x[0])))
    else:
        disp = float(np.linalg.norm(x - x[0], axis=1).max())
    return bool(raw_success or disp > DISP_THR)


def demo_completion_order(
    obs_seq: np.ndarray, mask_seq: np.ndarray, init_idx: int
) -> List[str]:
    valid = mask_seq[:, init_idx] > 0
    o = obs_seq[valid, init_idx, :].astype(np.float64)
    obj = o[:, QP : QP + 21]
    done: List[str] = []
    active = set(TASKS)
    for t in range(len(o)):
        newly = []
        for name in list(active):
            idxs = OBS_ELEMENT_INDICES[name]
            dist = float(np.linalg.norm(obj[t, idxs - QP] - OBS_ELEMENT_GOALS[name]))
            if dist < BONUS_THRESH:
                newly.append(name)
        for name in newly:
            done.append(name)
            active.discard(name)
        if not active:
            break
    return done


def load_demo_transition() -> Dict[str, List[float]]:
    """Row distributions over NEXT_LABELS (END mapped to STOP)."""
    rows: Dict[str, List[float]] = {}
    with DEMO_TRANS_CSV.open() as f:
        reader = csv.DictReader(f)
        col_map = {
            "microwave": "p_MW",
            "kettle": "p_Kettle",
            "bottom burner": "p_Bottom",
            "top burner": "p_Top",
            "light switch": "p_Light",
            "slide cabinet": "p_Slide",
            "hinge cabinet": "p_Hinge",
            "STOP": "p_END",
        }
        for rec in reader:
            after = rec["after"]
            rows[after] = [float(rec[col_map[lab]]) for lab in NEXT_LABELS]
    return rows


def load_demo_kettle_positions() -> Dict[int, int]:
    with DEMO_POS_CSV.open() as f:
        for rec in csv.DictReader(f):
            if rec["task"] != "kettle":
                continue
            return {
                1: int(rec["n_pos1"]),
                2: int(rec["n_pos2"]),
                3: int(rec["n_pos3"]),
                4: int(rec["n_pos4"]),
            }
    return {1: 0, 2: 0, 3: 0, 4: 0}


def _save(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path_base.with_suffix('.png')}")


def collect_config(
    model_key: str,
    nfe: int,
    dp_root: Path,
    obs_seq: np.ndarray,
    mask_seq: np.ndarray,
    demo_rank_cache: Dict[int, int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed_dir in seed_dirs(model_key, nfe, dp_root):
        metrics_path = seed_dir / "eval_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        em = json.loads(metrics_path.read_text())
        seed_name = seed_dir.name
        for ep in em.get("episodes") or []:
            scored, k, flags = clip_episode_record(ep, TASKS)
            kettle_flag = flags.get("kettle")
            init_idx = int(ep.get("init_idx", -1))
            if init_idx not in demo_rank_cache and init_idx >= 0:
                order = demo_completion_order(obs_seq, mask_seq, init_idx)
                demo_rank_cache[init_idx] = (
                    order.index("kettle") + 1 if "kettle" in order else 0
                )
            attempted: Optional[bool] = None
            disp = float("nan")
            hinge_att: Optional[bool] = None
            light_att: Optional[bool] = None
            npz_path = (
                seed_dir
                / "trajectory_logs"
                / f"ep_{int(ep.get('episode_idx', 0)):04d}.npz"
            )
            raw_k = bool((ep.get("task_success") or {}).get("kettle", 0))
            if npz_path.is_file():
                with np.load(npz_path, allow_pickle=False) as z:
                    if npz_attempt_valid(z):
                        attempted, disp = kettle_attempt_from_npz(z, raw_k)
                        hinge_att = task_attempt_from_npz(
                            z,
                            "hinge cabinet",
                            bool((ep.get("task_success") or {}).get("hinge cabinet", 0)),
                        )
                        light_att = task_attempt_from_npz(
                            z,
                            "light switch",
                            bool((ep.get("task_success") or {}).get("light switch", 0)),
                        )
            kettle_pos = None
            if "kettle" in scored:
                kettle_pos = scored.index("kettle") + 1
            rows.append(
                {
                    "model": model_key,
                    "nfe": nfe,
                    "seed": seed_name,
                    "init_idx": init_idx,
                    "scored_order": scored,
                    "k": k,
                    "kettle_flag": kettle_flag,
                    "raw_kettle": raw_k,
                    "attempted": attempted,
                    "disp": disp,
                    "demo_kettle_rank": demo_rank_cache.get(init_idx, 0),
                    "kettle_pos": kettle_pos,
                    "hinge_flag": flags.get("hinge cabinet"),
                    "light_flag": flags.get("light switch"),
                    "hinge_attempted": hinge_att,
                    "light_attempted": light_att,
                }
            )
    return rows


def _rate(vals: Iterable[Optional[float]]) -> Dict[str, Any]:
    clean = [float(v) for v in vals if v is not None]
    return proportion_stats(clean)


def transition_counts(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    mat = np.zeros((len(TASKS), len(NEXT_LABELS)), dtype=np.float64)
    idx = {t: i for i, t in enumerate(TASKS)}
    nxt = {t: i for i, t in enumerate(NEXT_LABELS)}
    for ep in rows:
        order: List[str] = list(ep["scored_order"])
        for i, a in enumerate(order):
            if a not in idx:
                continue
            if i + 1 < len(order):
                b = order[i + 1]
                if b in nxt:
                    mat[idx[a], nxt[b]] += 1.0
            else:
                mat[idx[a], nxt["STOP"]] += 1.0
    return mat


def p_next(mat: np.ndarray, after: str, nxt: str) -> Optional[float]:
    i = TASKS.index(after)
    j = NEXT_LABELS.index(nxt)
    tot = float(mat[i].sum())
    if tot <= 0:
        return None
    return float(mat[i, j] / tot)


def summarize(rows: List[Dict[str, Any]], demo_trans: Dict[str, List[float]]) -> Dict[str, Any]:
    scored_vals = [
        None if r["kettle_flag"] is None else float(r["kettle_flag"]) for r in rows
    ]
    episode_vals = [1.0 if r["kettle_flag"] == 1 else 0.0 for r in rows]
    att_vals = [None if r["attempted"] is None else float(r["attempted"]) for r in rows]
    cond_vals = [
        float(r["kettle_flag"] == 1)
        for r in rows
        if r["attempted"] is True
    ]
    n_attempt_valid = sum(r["attempted"] is not None for r in rows)
    rank_counts = Counter(int(r["demo_kettle_rank"]) for r in rows)
    by_rank: Dict[int, Dict[str, Any]] = {}
    for rank in sorted(rank_counts):
        sub = [r for r in rows if int(r["demo_kettle_rank"]) == rank]
        by_rank[rank] = {
            "n": len(sub),
            "episode": _rate(1.0 if r["kettle_flag"] == 1 else 0.0 for r in sub),
            "scored": _rate(
                None if r["kettle_flag"] is None else float(r["kettle_flag"]) for r in sub
            ),
            "attempt": _rate(
                None if r["attempted"] is None else float(r["attempted"]) for r in sub
            ),
        }
    pos_hist = Counter(int(r["kettle_pos"]) for r in rows if r["kettle_pos"] is not None)
    mat = transition_counts(rows)
    mw_i = TASKS.index("microwave")
    k_i = TASKS.index("kettle")
    mw_row = laplace_normalize(mat[mw_i]) if mat[mw_i].sum() > 0 else None
    k_row = laplace_normalize(mat[k_i]) if mat[k_i].sum() > 0 else None
    demo_mw = laplace_normalize(demo_trans.get("microwave", [0.0] * len(NEXT_LABELS)))
    demo_k = laplace_normalize(demo_trans.get("kettle", [0.0] * len(NEXT_LABELS)))
    hinge_cond = [
        float(r["hinge_flag"] == 1)
        for r in rows
        if r["hinge_attempted"] is True and r["hinge_flag"] is not None
    ]
    light_cond = [
        float(r["light_flag"] == 1)
        for r in rows
        if r["light_attempted"] is True and r["light_flag"] is not None
    ]
    return {
        "n": len(rows),
        "n_attempt_valid": n_attempt_valid,
        "episode": _rate(episode_vals),
        "scored": _rate(scored_vals),
        "attempt": _rate(att_vals),
        "cond_scored": _rate(cond_vals),
        "rank_counts": dict(rank_counts),
        "by_rank": by_rank,
        "pos_hist": dict(pos_hist),
        "p_kettle_given_mw": p_next(mat, "microwave", "kettle"),
        "entropy_after_mw": shannon_entropy(mw_row) if mw_row else None,
        "entropy_after_kettle": shannon_entropy(k_row) if k_row else None,
        "kl_mw_to_demo": kl_divergence(mw_row, demo_mw) if mw_row else None,
        "kl_kettle_to_demo": kl_divergence(k_row, demo_k) if k_row else None,
        "hinge_cond": _rate(hinge_cond),
        "light_cond": _rate(light_cond),
        "n_mw": int(mat[mw_i].sum()),
        "n_kettle_trans": int(mat[k_i].sum()),
    }


def _fmt_rate(st: Dict[str, Any]) -> str:
    mu = st.get("mean")
    if mu is None:
        return "n/a"
    n = st.get("n_samples")
    half = st.get("std")
    extra = f" n={int(n)}" if n is not None else ""
    if half is None:
        return f"{mu:.3f}{extra}"
    return f"{mu:.3f}±{half:.3f}{extra}"


def write_csvs(
    summaries: Dict[Tuple[str, int], Dict[str, Any]],
    all_rows: Dict[Tuple[str, int], List[Dict[str, Any]]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    decomp_path = out_dir / "kettle_decomposition.csv"
    with decomp_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "nfe",
                "n",
                "n_attempt_valid",
                "episode_mean",
                "episode_n",
                "scored_eligible_mean",
                "scored_eligible_n",
                "attempt_mean",
                "attempt_n",
                "cond_scored_mean",
                "cond_n",
                "p_kettle_given_mw",
                "entropy_after_mw",
                "entropy_after_kettle",
                "kl_mw_to_demo",
                "kl_kettle_to_demo",
            ]
        )
        for model, nfe, _ in CONFIGS:
            s = summaries[(model, nfe)]
            w.writerow(
                [
                    model,
                    nfe,
                    s["n"],
                    s["n_attempt_valid"],
                    s["episode"].get("mean"),
                    s["episode"].get("n_samples"),
                    s["scored"].get("mean"),
                    s["scored"].get("n_samples"),
                    s["attempt"].get("mean"),
                    s["attempt"].get("n_samples"),
                    s["cond_scored"].get("mean"),
                    s["cond_scored"].get("n_samples"),
                    s["p_kettle_given_mw"],
                    s["entropy_after_mw"],
                    s["entropy_after_kettle"],
                    s["kl_mw_to_demo"],
                    s["kl_kettle_to_demo"],
                ]
            )

    rank_path = out_dir / "kettle_by_demo_rank.csv"
    with rank_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "nfe",
                "demo_kettle_rank",
                "n",
                "episode_mean",
                "episode_n",
                "scored_eligible_mean",
                "scored_eligible_n",
                "attempt_mean",
                "attempt_n",
            ]
        )
        for model, nfe, _ in CONFIGS:
            s = summaries[(model, nfe)]
            for rank, st in sorted(s["by_rank"].items()):
                w.writerow(
                    [
                        model,
                        nfe,
                        rank,
                        st["n"],
                        st["episode"].get("mean"),
                        st["episode"].get("n_samples"),
                        st["scored"].get("mean"),
                        st["scored"].get("n_samples"),
                        st["attempt"].get("mean"),
                        st["attempt"].get("n_samples"),
                    ]
                )

    pos_path = out_dir / "kettle_eval_position_hist.csv"
    with pos_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "nfe", "position", "count"])
        for model, nfe, _ in CONFIGS:
            rows = all_rows[(model, nfe)]
            hist = Counter(r["kettle_pos"] for r in rows if r["kettle_pos"] is not None)
            for pos in range(1, 5):
                w.writerow([model, nfe, pos, hist.get(pos, 0)])


def plot_decomposition(summaries: Dict[Tuple[str, int], Dict[str, Any]], out_dir: Path) -> None:
    labels = [lab for _, _, lab in CONFIGS]
    x = np.arange(len(CONFIGS))
    scored = [summaries[(m, n)]["episode"].get("mean") or 0.0 for m, n, _ in CONFIGS]
    attempt = [summaries[(m, n)]["attempt"].get("mean") or 0.0 for m, n, _ in CONFIGS]
    cond = [summaries[(m, n)]["cond_scored"].get("mean") or 0.0 for m, n, _ in CONFIGS]
    colors = [COLORS[m] for m, _, _ in CONFIGS]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, vals, title in zip(
        axes,
        (scored, attempt, cond),
        ("P(kettle in first 4)", "P(attempt kettle)", "SR | attempt"),
    ):
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(
        "Kettle decomposition (observational; SR|attempt skipped if NPZ invalid)",
        y=1.03,
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out_dir / "01_kettle_decomposition")


def plot_rank_strata(summaries: Dict[Tuple[str, int], Dict[str, Any]], out_dir: Path) -> None:
    ranks = sorted({r for s in summaries.values() for r in s["rank_counts"]})
    if not ranks:
        return
    fig, axes = plt.subplots(1, len(ranks), figsize=(5.2 * len(ranks), 4.4), sharey=True)
    if len(ranks) == 1:
        axes = [axes]
    x = np.arange(len(CONFIGS))
    w = 0.36
    for ax, rank in zip(axes, ranks):
        scored, attempt = [], []
        for m, n, _ in CONFIGS:
            st = summaries[(m, n)]["by_rank"].get(rank)
            scored.append(st["episode"].get("mean") if st else 0.0)
            attempt.append(st["attempt"].get("mean") if st else 0.0)
        ax.bar(x - w / 2, scored, w, label="P(kettle in first 4)", color="#4c78a8", edgecolor="black", lw=0.4)
        ax.bar(x + w / 2, attempt, w, label="attempt", color="#f58518", edgecolor="black", lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, _, lab in CONFIGS], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.08)
        n0 = summaries[CONFIGS[0][0], CONFIGS[0][1]]["rank_counts"].get(rank, 0)
        ax.set_title(f"demo_kettle_rank={rank} (FP n={n0} at NFE1)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Within-rank kettle rates (same demo-rank stratum; not paired FP↔DP episodes)",
        y=1.03,
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out_dir / "02_kettle_by_demo_rank")


def plot_position_hist(
    all_rows: Dict[Tuple[str, int], List[Dict[str, Any]]],
    demo_pos: Dict[int, int],
    out_dir: Path,
) -> None:
    focus = [
        ("Demo (full dataset)", None, demo_pos, "#7f7f7f"),
        ("FlowPolicy", 1, None, COLORS["FlowPolicy"]),
        ("DP-CNN", 100, None, COLORS["DP-CNN"]),
        ("DP-Transformer", 100, None, COLORS["DP-Transformer"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8), sharey=True)
    xs = np.arange(1, 5)
    for ax, (title, nfe, preset, color) in zip(axes, focus):
        if preset is not None:
            counts = [preset.get(p, 0) for p in xs]
        else:
            rows = all_rows[(title, nfe)]
            hist = Counter(r["kettle_pos"] for r in rows if r["kettle_pos"] is not None)
            counts = [hist.get(p, 0) for p in xs]
        tot = sum(counts) or 1
        ax.bar(xs, [c / tot * 100 for c in counts], color=color, edgecolor="black", lw=0.4)
        ax.set_xticks(xs)
        ax.set_xlabel("Kettle position (1–4)")
        ax.set_title(title if nfe is None else f"{title} NFE={nfe}")
        ax.grid(True, axis="y", alpha=0.3)
        for x, c in zip(xs, counts):
            if c:
                ax.text(x, c / tot * 100 + 1.5, str(c), ha="center", fontsize=7)
    axes[0].set_ylabel("% of kettle completions")
    fig.suptitle(
        "Kettle position: full demo dataset vs scored eval order (p4 clip)",
        y=1.04,
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, out_dir / "03_kettle_position_hist")


def plot_p_mw_kettle(summaries: Dict[Tuple[str, int], Dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(CONFIGS))
    vals = [summaries[(m, n)]["p_kettle_given_mw"] or 0.0 for m, n, _ in CONFIGS]
    colors = [COLORS[m] for m, _, _ in CONFIGS]
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, _, lab in CONFIGS], rotation=25, ha="right")
    ax.set_ylabel("P(next=kettle | microwave)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Transition after microwave (scored order, STOP after 4th)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0.417, color="#7f7f7f", ls="--", lw=1, label="demo P(kettle|MW)=0.42")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "04_p_kettle_given_mw")


def _pca_blurb() -> str:
    if not PCA_README.is_file():
        return (
            "PCA aksi kettle (hanya episode sukses) tidak dihitung ulang di skrip ini."
        )
    return (
        "PCA aksi kettle yang sudah ada (hanya episode sukses; "
        "data/kitchen_eval_plots/nfe100/kettle_action_pca/README.txt): "
        "jarak mean ke centroid demo NFE1=4.23, NFE8=4.10, NFE100=4.08 — "
        "tren tidak naik. Ini melemahkan hipotesis drift ruang aksi pada episode "
        "yang berhasil kettle; kegagalan path-selection memang dikecualikan dari PCA."
    )


def write_report(
    summaries: Dict[Tuple[str, int], Dict[str, Any]],
    out_dir: Path,
) -> None:
    lines: List[str] = []
    w = lines.append
    w("=" * 78)
    w("Pola kettle — laporan observasional (bukan bukti kausal)")
    w("=" * 78)
    w("")
    w("Bahasa: temuan di bawah **consistent with** / **diduga berkaitan** dengan")
    w("seleksi jalur, frekuensi urutan, atau kesulitan eksekusi. Tidak ada")
    w("rebalancing dataset, tidak ada intervensi posisi kettle, tidak ada retrain.")
    w("Kettle scored memakai plafon p4 (completion_order[:4]; leftover = exclude).")
    w("P(kettle in first 4) = flag==1 atas semua episode (None dihitung 0, n=300):")
    w("ini yang dibanding untuk klaim tesis. Eligible SR membuang leftover dari")
    w("penyebut dan bisa menaikkan angka pada model yang sering selesai 4 task")
    w("tanpa kettle — jangan dipakai sebagai 'overall kettle SR'.")
    w("Attempt = displacement kettle di policy_obs_obj_qp > 0.05, atau raw success")
    w("jika NPZ valid. Tanpa NPZ valid, dekomposisi attempt dilewati (bukan")
    w("fallback ke completion_order).")
    w("")
    w("-" * 78)
    w("1) Ringkasan per konfigurasi")
    w("-" * 78)
    for model, nfe, lab in CONFIGS:
        s = summaries[(model, nfe)]
        w(f"  {lab}")
        w(f"    P(kettle in first 4): {_fmt_rate(s['episode'])}")
        w(f"    eligible SR (leftover excluded): {_fmt_rate(s['scored'])}")
        if s["n_attempt_valid"]:
            w(f"    P(attempt):    {_fmt_rate(s['attempt'])}")
            w(f"    SR | attempt:  {_fmt_rate(s['cond_scored'])}")
        else:
            w("    attempt:       NPZ tidak valid — dilewati")
        pkm = s["p_kettle_given_mw"]
        w(
            f"    P(kettle|MW):  {pkm:.3f} (n_after_MW={s['n_mw']})"
            if pkm is not None
            else "    P(kettle|MW):  n/a"
        )
        if s["entropy_after_mw"] is not None:
            w(
                f"    H(next|MW)={s['entropy_after_mw']:.3f}  "
                f"H(next|kettle)={s['entropy_after_kettle']:.3f}  "
                f"KL(MW||demo)={s['kl_mw_to_demo']:.3f}  "
                f"KL(kettle||demo)={s['kl_kettle_to_demo']:.3f}"
            )
        w(f"    demo_kettle_rank counts: {s['rank_counts']}")
        w("")

    fp1 = summaries[("FlowPolicy", 1)]
    fp100 = summaries[("FlowPolicy", 100)]
    cnn = summaries[("DP-CNN", 100)]
    trans = summaries[("DP-Transformer", 100)]

    w("-" * 78)
    w("2) NFE (hanya FlowPolicy, init_idx sama antar NFE dalam seed)")
    w("-" * 78)
    w(
        f"  P(kettle in first 4) FP NFE1={fp1['episode'].get('mean'):.3f} → "
        f"NFE8={summaries[('FlowPolicy', 8)]['episode'].get('mean'):.3f} → "
        f"NFE100={fp100['episode'].get('mean'):.3f}."
    )
    if fp1["n_attempt_valid"] and fp100["n_attempt_valid"]:
        w(
            f"  Attempt {fp1['attempt'].get('mean'):.3f} → {fp100['attempt'].get('mean'):.3f}; "
            f"SR|attempt {fp1['cond_scored'].get('mean'):.3f} → "
            f"{fp100['cond_scored'].get('mean'):.3f}."
        )
        w(
            "  **Consistent with** seleksi jalur (lebih jarang mencoba kettle), "
            "bukan kettle menjadi lebih sulit dieksekusi. Rank demo konstan antar "
            "NFE sehingga mix posisi demo **tidak** menjelaskan drop NFE."
        )
    w("")

    w("-" * 78)
    w("3) Frekuensi urutan (distribusi init, bukan pasangan episode FP↔DP)")
    w("-" * 78)
    w(
        f"  Rank yang muncul (FP NFE1): {fp1['rank_counts']}. "
        "Jika rank 1 tidak ada, strata hanya rank yang teramati."
    )
    w(f"  Distribusi rank DP-CNN@100: {cnn['rank_counts']}")
    w(f"  Distribusi rank DP-Trans@100: {trans['rank_counts']}")
    for rank in sorted(fp1["rank_counts"]):
        bits = []
        for model, nfe, lab in (
            ("FlowPolicy", 1, "FP1"),
            ("FlowPolicy", 100, "FP100"),
            ("DP-CNN", 100, "CNN"),
            ("DP-Transformer", 100, "Trans"),
        ):
            st = summaries[(model, nfe)]["by_rank"].get(rank)
            if not st:
                bits.append(f"{lab}=n/a")
                continue
            bits.append(
                f"{lab} p4={st['episode'].get('mean'):.3f} "
                f"att={st['attempt'].get('mean') if st['attempt'].get('mean') is not None else float('nan'):.3f} "
                f"n={st['n']}"
            )
        w(f"  rank {rank}: " + " | ".join(bits))
    w(
        "  Jika gap FP@1 vs DP@100 tetap di setiap strata, mix rank demo "
        "**diduga tidak** menjelaskan gap antarmodel (mix harus dibandingkan di atas)."
    )
    w("  Rank 0 = demo init tanpa kettle: tetap mengambil kettle = bukan menyalin urutan demo.")
    p_demo = 0.417
    def _p(x: Optional[float]) -> str:
        return f"{x:.3f}" if x is not None else "n/a"

    w(
        f"  P(kettle|MW) demo={p_demo:.3f}; "
        f"FP1={_p(fp1['p_kettle_given_mw'])}; FP100={_p(fp100['p_kettle_given_mw'])}; "
        f"CNN={_p(cnn['p_kettle_given_mw'])}; Trans={_p(trans['p_kettle_given_mw'])}."
    )
    w(
        "  Jangan memakai angka WHY FP@8 sebagai bukti klaim NFE1: transisi dihitung per NFE."
    )
    w("  KL(policy||demo) deskriptif (Laplace); bukan uji hipotesis.")
    w("")

    w("-" * 78)
    w("4) Kesulitan eksekusi (SR | attempt)")
    w("-" * 78)
    if fp1["cond_scored"].get("mean") is not None:
        w(
            f"  Kettle SR|attempt FP1={fp1['cond_scored'].get('mean'):.3f}, "
            f"FP100={fp100['cond_scored'].get('mean'):.3f}, "
            f"CNN={cnn['cond_scored'].get('mean'):.3f}, "
            f"Trans={trans['cond_scored'].get('mean'):.3f}."
        )
        w(
            "  Conditional tinggi di FP (dan DP jika NPZ valid) **consistent with** "
            "gap overall yang diduga attempt/path, bukan kettle intrinsik sulit disentuh."
        )
    hinge_ok = all(
        summaries[(m, n)]["hinge_cond"].get("mean") is not None
        for m, n in (("FlowPolicy", 1), ("DP-CNN", 100))
    )
    if hinge_ok:
        w(
            f"  Hinge SR|attempt (analog displacement) FP1="
            f"{fp1['hinge_cond'].get('mean'):.3f} CNN="
            f"{cnn['hinge_cond'].get('mean'):.3f}; light FP1="
            f"{fp1['light_cond'].get('mean'):.3f} CNN="
            f"{cnn['light_cond'].get('mean'):.3f}."
        )
    else:
        w("  Perbandingan hinge/light | attempt tidak dipaksakan (sinyal tidak lengkap).")
    w("")
    w(_pca_blurb())
    w("")

    w("-" * 78)
    w("5) Head-to-head")
    w("-" * 78)
    w(
        f"  Klaim tesis (NFE tidak tertanding): FP@1 P(kettle in first 4)="
        f"{fp1['episode'].get('mean'):.3f} vs CNN@100={cnn['episode'].get('mean'):.3f} "
        f"vs Trans@100={trans['episode'].get('mean'):.3f}."
    )
    w(
        f"  NFE tertanding: FP@100={fp100['episode'].get('mean'):.3f} vs "
        f"CNN@100={cnn['episode'].get('mean'):.3f} vs Trans@100={trans['episode'].get('mean'):.3f}."
    )
    w("  DP tidak dieval pada NFE rendah (protokol existing).")
    w("")

    w("-" * 78)
    w("6) Apa yang tidak bisa disimpulkan")
    w("-" * 78)
    w("  - Mekanisme kausal NFE → diversity → drop kettle tidak terbukti.")
    w("  - Tidak ada eksperimen rebalance posisi kettle atau equalize transisi.")
    w("  - Tidak mengontrol 'jika kettle dipaksa posisi 3/4'.")
    w("  - Sample eval ini mungkin tidak memuat demo kettle-rank=1; itu bukan")
    w("    klaim bahwa seluruh dataset demo tidak punya kettle di posisi 1.")
    w("  - FP vs DP bukan pasangan init_idx per episode.")
    w("  - LSTM-GMM / IBC / BeT tidak dianalisis di sini.")
    w("")
    w("=" * 78)
    path = out_dir / "kettle_pattern_report.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    dp_root = find_dp_root()
    print(f"DP root: {dp_root}")
    print(f"FP root: {resolve_fp_root()}")
    obs_seq = np.load(DEMO_DIR / "observations_seq.npy")
    mask_seq = np.load(DEMO_DIR / "existence_mask.npy")
    demo_trans = load_demo_transition()
    demo_pos = load_demo_kettle_positions()
    demo_rank_cache: Dict[int, int] = {}

    all_rows: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    summaries: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for model, nfe, lab in CONFIGS:
        print(f"Loading {lab} …")
        rows = collect_config(model, nfe, dp_root, obs_seq, mask_seq, demo_rank_cache)
        all_rows[(model, nfe)] = rows
        summaries[(model, nfe)] = summarize(rows, demo_trans)
        s = summaries[(model, nfe)]
        print(
            f"  n={s['n']} episode={s['episode'].get('mean')} "
            f"attempt={s['attempt'].get('mean')} rank={s['rank_counts']}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csvs(summaries, all_rows, out_dir)
    plot_decomposition(summaries, out_dir)
    plot_rank_strata(summaries, out_dir)
    plot_position_hist(all_rows, demo_pos, out_dir)
    plot_p_mw_kettle(summaries, out_dir)
    write_report(summaries, out_dir)
    print("Done.", out_dir)


if __name__ == "__main__":
    main()
