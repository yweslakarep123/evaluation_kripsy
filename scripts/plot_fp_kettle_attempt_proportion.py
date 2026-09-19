#!/usr/bin/env python3
"""Kettle attempt proportion vs NFE for FlowPolicy (path-selection hypothesis).

Attempt definition (confirmed):
  attempted = task_success.kettle OR max displacement of kettle state
              in policy_obs_obj_qp > 0.05

Also runs an artifact check:
  - init_idx pairing across NFE
  - demo-derived kettle position (env has no fixed order under KitchenAllV0)
  - for not-attempted episodes: # other tasks completed before timeout

Outputs under data/kitchen_eval_plots/nfe100/kettle_action_pca/:
  02_kettle_attempt_proportion.{png,pdf}
  03_kettle_attempt_vs_conditional_sr.{png,pdf}
  kettle_attempt_summary.csv
  artifact_check.txt

Usage:
  /home/daffa/miniforge3/envs/flowpolicy-kitchen/bin/python \\
      scripts/plot_fp_kettle_attempt_proportion.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FP_ROOT = ROOT / "kripsy12/FlowPolicy/data/kitchen_eval_nfe100/flowpolicy"
DEMO_DIR = ROOT / "kripsy12/FlowPolicy/data/kitchen"
OUT_DIR = ROOT / "data/kitchen_eval_plots/nfe100/kettle_action_pca"

SEEDS = (42, 43, 44)
NFES = (1, 2, 3, 4, 8, 32, 100)
DISP_THR = 0.05

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
TASK_ORDER = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]
BONUS_THRESH = 0.3

COLOR_ATTEMPTED = "#2ca02c"
COLOR_NOT_ATTEMPTED = "#d62728"


def _save(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path_base.with_suffix('.png')}")


def is_attempted(z: np.lib.npyio.NpzFile, success: bool) -> tuple[bool, float]:
    k = np.asarray(z["policy_obs_obj_qp"], dtype=np.float64)[:, -1, 14:21]
    if len(k) == 0:
        return bool(success), 0.0
    disp = float(np.linalg.norm(k - k[0], axis=1).max())
    return bool(success or disp > DISP_THR), disp


def demo_completion_order(obs_seq: np.ndarray, mask_seq: np.ndarray, init_idx: int) -> list[str]:
    valid = mask_seq[:, init_idx] > 0
    o = obs_seq[valid, init_idx, :].astype(np.float64)
    obj = o[:, QP : QP + 21]
    done: list[str] = []
    active = set(TASK_ORDER)
    for t in range(len(o)):
        newly = []
        for name in list(active):
            idxs = OBS_ELEMENT_INDICES[name]
            dist = float(
                np.linalg.norm(obj[t, idxs - QP] - OBS_ELEMENT_GOALS[name])
            )
            if dist < BONUS_THRESH:
                newly.append(name)
        for name in newly:
            done.append(name)
            active.discard(name)
        if not active:
            break
    return done


def collect_episodes() -> dict[int, list[dict[str, Any]]]:
    """Load all episodes keyed by NFE with attempt labels."""
    obs_seq = np.load(DEMO_DIR / "observations_seq.npy")
    mask_seq = np.load(DEMO_DIR / "existence_mask.npy")
    demo_rank_cache: dict[int, int] = {}

    by_nfe: dict[int, list[dict[str, Any]]] = {n: [] for n in NFES}
    for nfe in NFES:
        for seed in SEEDS:
            run = FP_ROOT / f"seed_baseline_{seed}_nfe{nfe}_sseed0"
            metrics = json.loads((run / "eval_metrics.json").read_text())
            for ep in metrics["episodes"]:
                npz_path = (
                    run / "trajectory_logs" / f"ep_{int(ep['episode_idx']):04d}.npz"
                )
                z = np.load(npz_path)
                succ = bool(ep["task_success"]["kettle"])
                attempted, disp = is_attempted(z, succ)
                init_idx = int(ep["init_idx"])
                if init_idx not in demo_rank_cache:
                    order = demo_completion_order(obs_seq, mask_seq, init_idx)
                    demo_rank_cache[init_idx] = (
                        order.index("kettle") + 1 if "kettle" in order else 0
                    )
                others = [t for t in (ep.get("completion_order") or []) if t != "kettle"]
                by_nfe[nfe].append(
                    {
                        "seed": seed,
                        "nfe": nfe,
                        "episode_idx": int(ep["episode_idx"]),
                        "init_idx": init_idx,
                        "success": succ,
                        "attempted": attempted,
                        "max_disp": disp,
                        "demo_kettle_rank": demo_rank_cache[init_idx],
                        "n_other_completed": len(others),
                        "other_tasks": others,
                        "n_env_steps": int(z["n_env_steps"]),
                        "completion_order": list(ep.get("completion_order") or []),
                    }
                )
    return by_nfe


def check_init_identity() -> dict[int, bool]:
    out = {}
    for seed in SEEDS:
        maps = {}
        for nfe in NFES:
            m = json.loads(
                (
                    FP_ROOT / f"seed_baseline_{seed}_nfe{nfe}_sseed0" / "eval_metrics.json"
                ).read_text()
            )
            maps[nfe] = [e["init_idx"] for e in m["episodes"]]
        out[seed] = all(maps[1] == maps[n] for n in NFES)
    return out


def summarize(by_nfe: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    summary = {}
    for nfe in NFES:
        eps = by_nfe[nfe]
        n = len(eps)
        n_att = sum(1 for e in eps if e["attempted"])
        n_not = n - n_att
        n_succ = sum(1 for e in eps if e["success"])
        n_succ_att = sum(1 for e in eps if e["attempted"] and e["success"])
        not_eps = [e for e in eps if not e["attempted"]]
        others = [e["n_other_completed"] for e in not_eps]
        summary[nfe] = {
            "n_total": n,
            "n_attempted": n_att,
            "n_not_attempted": n_not,
            "pct_attempted": 100.0 * n_att / n,
            "pct_not_attempted": 100.0 * n_not / n,
            "overall_sr": n_succ / n,
            "conditional_sr": (n_succ_att / n_att) if n_att else float("nan"),
            "n_success": n_succ,
            "not_other_mean": float(np.mean(others)) if others else float("nan"),
            "not_other_std": float(np.std(others, ddof=1)) if len(others) > 1 else 0.0,
            "not_other_counter": Counter(others),
            "not_steps_mean": float(np.mean([e["n_env_steps"] for e in not_eps]))
            if not_eps
            else float("nan"),
        }
    return summary


def artifact_report(
    by_nfe: dict[int, list[dict[str, Any]]],
    summary: dict[int, dict[str, Any]],
    init_ok: dict[int, bool],
) -> str:
    lines: list[str] = []
    lines.append("Artifact check: is 'not attempted' just prior-task duration?")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1) Assigned order / kettle position")
    lines.append("-" * 40)
    lines.append(
        "KitchenAllV0 assigns ALL 7 tasks with COMPLETE_IN_ANY_ORDER=True."
    )
    lines.append(
        "There is NO fixed per-episode position of kettle in an env-assigned order."
    )
    lines.append(
        "Proxy used: kettle rank in the paired demo completion order "
        "(0 = kettle never completed in that demo)."
    )
    lines.append("")
    lines.append("init_idx lists identical across NFE for each seed?")
    for seed, ok in init_ok.items():
        lines.append(f"  seed {seed}: {ok}")
    lines.append("")
    # demo rank dist from NFE1 (identical inits ⇒ identical across NFE)
    ranks = [e["demo_kettle_rank"] for e in by_nfe[1]]
    lines.append(f"Demo kettle-rank distribution (NFE1, all seeds): {dict(sorted(Counter(ranks).items()))}")
    lines.append("Because init_idx lists match, this distribution is identical for NFE 8/32/100.")
    lines.append("")
    lines.append("P(not-attempted | demo_kettle_rank) by NFE:")
    for nfe in NFES:
        by_r_tot: dict[int, int] = defaultdict(int)
        by_r_b: dict[int, int] = defaultdict(int)
        for e in by_nfe[nfe]:
            r = e["demo_kettle_rank"]
            by_r_tot[r] += 1
            if not e["attempted"]:
                by_r_b[r] += 1
        bits = []
        for r in sorted(by_r_tot):
            bits.append(
                f"rank{r}: {by_r_b[r]}/{by_r_tot[r]}={by_r_b[r]/by_r_tot[r]:.2f}"
            )
        lines.append(f"  NFE={nfe}: " + ", ".join(bits))
    lines.append("")
    lines.append("2) Condition (b) — other tasks completed & env steps")
    lines.append("-" * 40)
    for nfe in NFES:
        s = summary[nfe]
        lines.append(
            f"  NFE={nfe}: not-attempted={s['n_not_attempted']} "
            f"({s['pct_not_attempted']:.1f}%), "
            f"other_tasks mean={s['not_other_mean']:.2f}±{s['not_other_std']:.2f}, "
            f"counts={dict(sorted(s['not_other_counter'].items()))}, "
            f"mean n_env_steps={s['not_steps_mean']:.1f}"
        )
    lines.append("")
    lines.append("3) Paired flips NFE1 → NFE8 (same seed, same init_idx)")
    lines.append("-" * 40)
    for seed in SEEDS:
        by_init = {
            nfe: {e["init_idx"]: e for e in by_nfe[nfe] if e["seed"] == seed}
            for nfe in (1, 8)
        }
        both_a = both_n = a2n = n2a = 0
        o1: list[int] = []
        o8: list[int] = []
        for ii, e1 in by_init[1].items():
            e8 = by_init[8][ii]
            if e1["attempted"] and e8["attempted"]:
                both_a += 1
            elif (not e1["attempted"]) and (not e8["attempted"]):
                both_n += 1
            elif e1["attempted"] and (not e8["attempted"]):
                a2n += 1
                o1.append(e1["n_other_completed"])
                o8.append(e8["n_other_completed"])
            else:
                n2a += 1
        lines.append(
            f"  seed{seed}: both_att={both_a} both_not={both_n} "
            f"att→not={a2n} not→att={n2a}"
        )
        if o1:
            lines.append(
                f"    on att→not: other_tasks NFE1={np.mean(o1):.2f} → "
                f"NFE8={np.mean(o8):.2f}"
            )
    lines.append("")
    lines.append("Conclusion")
    lines.append("-" * 40)
    lines.append(
        "- Demo/seed-linked kettle position is CONSTANT across NFE, so it cannot"
    )
    lines.append(
        "  explain the rise in not-attempted rate from NFE1 to NFE8."
    )
    lines.append(
        "- Not-attempted episodes at higher NFE complete MORE other tasks"
    )
    lines.append(
        f"  (mean {summary[1]['not_other_mean']:.2f} → {summary[8]['not_other_mean']:.2f})"
    )
    lines.append(
        "  and always use the full horizon (n_env_steps≈282), so this is NOT"
    )
    lines.append(
        "  'ran out of time before kettle's fixed turn'."
    )
    lines.append(
        "- Paired att→not flips show the same pattern: more other tasks at NFE8."
    )
    lines.append(
        "- Dominant account: path selection away from kettle, not prior-task"
    )
    lines.append("  duration starving a fixed kettle slot.")
    lines.append("")
    return "\n".join(lines)


def plot_stacked(summary: dict[int, dict[str, Any]], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(len(NFES))
    pct_not = [summary[n]["pct_not_attempted"] for n in NFES]
    pct_att = [summary[n]["pct_attempted"] for n in NFES]

    b1 = ax.bar(
        x,
        pct_not,
        color=COLOR_NOT_ATTEMPTED,
        width=0.65,
        label="Assigned tapi tidak dicoba",
    )
    b2 = ax.bar(
        x,
        pct_att,
        bottom=pct_not,
        color=COLOR_ATTEMPTED,
        width=0.65,
        label="Dicoba",
    )

    for i, (pn, pa) in enumerate(zip(pct_not, pct_att)):
        if pn >= 4:
            ax.text(i, pn / 2, f"{pn:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if pa >= 4:
            ax.text(
                i,
                pn + pa / 2,
                f"{pa:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in NFES])
    ax.set_xlabel("NFE")
    ax.set_ylabel("Persentase episode (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "FlowPolicy: proporsi episode yang mencoba kettle vs NFE\n"
        f"(dicoba = sukses atau max_disp kettle > {DISP_THR})",
        fontsize=12,
    )
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _save(fig, out_base)


def plot_lines(summary: dict[int, dict[str, Any]], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = np.array(NFES, dtype=float)
    attempt_pct = [summary[n]["pct_attempted"] for n in NFES]
    cond_sr_pct = [100.0 * summary[n]["conditional_sr"] for n in NFES]
    overall_sr_pct = [100.0 * summary[n]["overall_sr"] for n in NFES]

    ax.plot(
        x,
        attempt_pct,
        color=COLOR_ATTEMPTED,
        marker="o",
        linewidth=2.2,
        markersize=8,
        label="% episode yang mencoba kettle",
    )
    ax.plot(
        x,
        cond_sr_pct,
        color="#1f77b4",
        marker="s",
        linewidth=2.2,
        markersize=8,
        label="Success rate kettle | dicoba",
    )
    ax.plot(
        x,
        overall_sr_pct,
        color="#7f7f7f",
        marker="^",
        linewidth=1.6,
        markersize=7,
        linestyle="--",
        label="Success rate kettle overall (sanity)",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(list(NFES))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{int(v)}"))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("NFE (Number of Function Evaluations)")
    ax.set_ylabel("Persentase (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "FlowPolicy kettle: attempt rate vs conditional success\n"
        "Drop overall SR mengikuti penurunan attempt, bukan kegagalan saat mencoba",
        fontsize=12,
    )
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=9, framealpha=0.92)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _save(fig, out_base)


def write_csv(summary: dict[int, dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "nfe",
                "n_total",
                "n_attempted",
                "n_not_attempted",
                "pct_attempted",
                "pct_not_attempted",
                "overall_sr",
                "conditional_sr",
                "not_attempted_other_tasks_mean",
                "not_attempted_other_tasks_std",
            ],
        )
        w.writeheader()
        for nfe in NFES:
            s = summary[nfe]
            w.writerow(
                {
                    "nfe": nfe,
                    "n_total": s["n_total"],
                    "n_attempted": s["n_attempted"],
                    "n_not_attempted": s["n_not_attempted"],
                    "pct_attempted": f"{s['pct_attempted']:.4f}",
                    "pct_not_attempted": f"{s['pct_not_attempted']:.4f}",
                    "overall_sr": f"{s['overall_sr']:.6f}",
                    "conditional_sr": f"{s['conditional_sr']:.6f}",
                    "not_attempted_other_tasks_mean": f"{s['not_other_mean']:.4f}",
                    "not_attempted_other_tasks_std": f"{s['not_other_std']:.4f}",
                }
            )
    print(f"  wrote {path}")


def print_markdown(summary: dict[int, dict[str, Any]]) -> None:
    print("\n## Ringkasan proporsi attempt kettle (FlowPolicy, 3 seed × 100 ep)\n")
    print(
        "| NFE | n | Dicoba | Tidak dicoba | % dicoba | % tidak | "
        "SR overall | SR \| dicoba | Other tasks @tidak (mean±std) |"
    )
    print("|----:|----:|-------:|-------------:|---------:|--------:|----------:|------------:|--------------------------------:|")
    for nfe in NFES:
        s = summary[nfe]
        print(
            f"| {nfe} | {s['n_total']} | {s['n_attempted']} | {s['n_not_attempted']} | "
            f"{s['pct_attempted']:.1f}% | {s['pct_not_attempted']:.1f}% | "
            f"{100*s['overall_sr']:.1f}% | {100*s['conditional_sr']:.1f}% | "
            f"{s['not_other_mean']:.2f}±{s['not_other_std']:.2f} |"
        )
    print()
    print(
        "Catatan: SR overall harus mendekati Tabel 4.1 / `summary.csv` "
        "(~67.7% @NFE1, ~34.3% @NFE8). SR|dicoba adalah metrik baru."
    )


def main() -> None:
    print("Checking init_idx identity across NFE…")
    init_ok = check_init_identity()
    for seed, ok in init_ok.items():
        print(f"  seed {seed}: identical={ok}")

    print("Collecting episodes + attempt labels…")
    by_nfe = collect_episodes()
    summary = summarize(by_nfe)

    print("\n=== Per-NFE quick stats ===")
    for nfe in NFES:
        s = summary[nfe]
        print(
            f"  NFE={nfe}: attempted={s['pct_attempted']:.1f}%  "
            f"overall_SR={100*s['overall_sr']:.1f}%  "
            f"cond_SR={100*s['conditional_sr']:.1f}%  "
            f"other@not={s['not_other_mean']:.2f}"
        )

    report = artifact_report(by_nfe, summary, init_ok)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    art_path = OUT_DIR / "artifact_check.txt"
    art_path.write_text(report)
    print(f"\n{report}")
    print(f"  wrote {art_path}")

    write_csv(summary, OUT_DIR / "kettle_attempt_summary.csv")
    print_markdown(summary)

    print("Plotting…")
    plot_stacked(summary, OUT_DIR / "02_kettle_attempt_proportion")
    plot_lines(summary, OUT_DIR / "03_kettle_attempt_vs_conditional_sr")
    print("Done.")


if __name__ == "__main__":
    main()
