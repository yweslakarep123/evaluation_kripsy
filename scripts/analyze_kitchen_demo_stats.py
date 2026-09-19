#!/usr/bin/env python3
"""Analyze Kitchen demonstration dataset: per-task counts + transition heatmap.

Reads MJL demos from kitchen_demos_multitask/ (folder name = 4-task sequence).
Writes CSV summaries and a transition heatmap (PNG + PDF).

By default analyzes FlowPolicy's kitchen demos. Diffusion Policy uses the same
data via symlink (diffusion_policy/data/kitchen -> FlowPolicy/data/kitchen).

Outputs under --out-dir (default data/kitchen_eval_plots/demo_stats/):
  demo_dataset_meta.csv
  demo_per_task_counts.csv
  demo_sequences.csv
  demo_first_last_task.csv
  demo_transition_matrix.csv          (probabilities)
  demo_transition_counts.csv          (raw counts)
  02_demo_transition_heatmap.png/.pdf

Usage:
  python scripts/analyze_kitchen_demo_stats.py
  python scripts/analyze_kitchen_demo_stats.py \\
    --demo-dir diffusion_policy/data/kitchen/kitchen_demos_multitask \\
    --out-dir data/kitchen_eval_plots/demo_stats_diffusion_policy \\
    --label "Diffusion Policy"
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_DIR = (
    ROOT / "kripsy12/FlowPolicy/data/kitchen/kitchen_demos_multitask"
)
DEFAULT_OUT_DIR = ROOT / "data/kitchen_eval_plots/demo_stats"
FP_DEMO_RESOLVED = (
    ROOT / "kripsy12/FlowPolicy/data/kitchen/kitchen_demos_multitask"
).resolve()

TOKEN_MAP = {
    "microwave": "microwave",
    "kettle": "kettle",
    "topknob": "top burner",
    "bottomknob": "bottom burner",
    "switch": "light switch",
    "hinge": "hinge cabinet",
    "slide": "slide cabinet",
}
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
}
NEXT_LABELS = TASKS + ["END"]


def parse_seq(folder_name: str) -> list[str] | None:
    for prefix in ("friday_", "postcorl_"):
        if folder_name.startswith(prefix):
            rest = folder_name[len(prefix) :]
            break
    else:
        return None
    tokens = rest.split("_")
    if len(tokens) != 4:
        return None
    try:
        return [TOKEN_MAP[t] for t in tokens]
    except KeyError:
        return None


def collect_demos(demo_dir: Path) -> dict[str, tuple[list[str], int]]:
    """folder -> (sequence, n_mjl). Only direct *.mjl (matches training loader)."""
    out: dict[str, tuple[list[str], int]] = {}
    for d in sorted(demo_dir.iterdir()):
        if not d.is_dir():
            continue
        n = len(list(d.glob("*.mjl")))
        if n == 0:
            continue
        seq = parse_seq(d.name)
        if seq is None:
            raise ValueError(f"Cannot parse sequence from folder: {d.name}")
        out[d.name] = (seq, n)
    return out


def analyze(demos: dict[str, tuple[list[str], int]]) -> dict:
    total = sum(n for _, n in demos.values())
    task_count: Counter = Counter()
    first_task: Counter = Counter()
    last_task: Counter = Counter()
    imm: dict[str, Counter] = defaultdict(Counter)
    a_done: Counter = Counter()
    source: Counter = Counter()
    sequences: list[dict] = []

    for folder, (seq, n) in demos.items():
        src = "friday" if folder.startswith("friday_") else "postcorl"
        source[src] += n
        for t in seq:
            task_count[t] += n
        first_task[seq[0]] += n
        last_task[seq[-1]] += n
        for i, a in enumerate(seq):
            a_done[a] += n
            if i + 1 < len(seq):
                imm[a][seq[i + 1]] += n
        sequences.append(
            {
                "folder": folder,
                "source": src,
                "n_demos": n,
                "task1": seq[0],
                "task2": seq[1],
                "task3": seq[2],
                "task4": seq[3],
                "sequence": " → ".join(SHORT[t] for t in seq),
            }
        )

    count_mat = np.zeros((len(TASKS), len(NEXT_LABELS)), dtype=np.int64)
    for i, a in enumerate(TASKS):
        for j, b in enumerate(TASKS):
            count_mat[i, j] = imm[a][b]
        count_mat[i, -1] = a_done[a] - sum(imm[a].values())

    prob_mat = np.zeros_like(count_mat, dtype=np.float64)
    for i, a in enumerate(TASKS):
        n = a_done[a]
        if n > 0:
            prob_mat[i] = count_mat[i] / n

    return {
        "total": total,
        "task_count": task_count,
        "first_task": first_task,
        "last_task": last_task,
        "a_done": a_done,
        "source": source,
        "sequences": sequences,
        "count_mat": count_mat,
        "prob_mat": prob_mat,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def save_csvs(stats: dict, out_dir: Path, meta: dict) -> None:
    total = stats["total"]

    write_csv(
        out_dir / "demo_dataset_meta.csv",
        ["key", "value"],
        [{"key": k, "value": v} for k, v in meta.items()],
    )

    write_csv(
        out_dir / "demo_per_task_counts.csv",
        ["task", "short", "n_demos_containing", "pct_of_demos", "n_completions"],
        [
            {
                "task": t,
                "short": SHORT[t],
                "n_demos_containing": stats["task_count"][t],
                "pct_of_demos": round(100.0 * stats["task_count"][t] / total, 2),
                "n_completions": stats["a_done"][t],
            }
            for t in TASKS
        ],
    )

    write_csv(
        out_dir / "demo_first_last_task.csv",
        ["role", "task", "short", "n_demos", "pct_of_demos"],
        [
            {
                "role": "first",
                "task": t,
                "short": SHORT[t],
                "n_demos": stats["first_task"][t],
                "pct_of_demos": round(100.0 * stats["first_task"][t] / total, 2),
            }
            for t in TASKS
            if stats["first_task"][t] > 0
        ]
        + [
            {
                "role": "last",
                "task": t,
                "short": SHORT[t],
                "n_demos": stats["last_task"][t],
                "pct_of_demos": round(100.0 * stats["last_task"][t] / total, 2),
            }
            for t in TASKS
            if stats["last_task"][t] > 0
        ]
        + [
            {
                "role": "source",
                "task": src,
                "short": src,
                "n_demos": n,
                "pct_of_demos": round(100.0 * n / total, 2),
            }
            for src, n in sorted(stats["source"].items())
        ],
    )

    seq_rows = sorted(stats["sequences"], key=lambda r: -r["n_demos"])
    write_csv(
        out_dir / "demo_sequences.csv",
        [
            "folder",
            "source",
            "n_demos",
            "sequence",
            "task1",
            "task2",
            "task3",
            "task4",
        ],
        seq_rows,
    )

    prob_rows = []
    for i, a in enumerate(TASKS):
        row = {"after": a, "after_short": SHORT[a], "n_completions": stats["a_done"][a]}
        for j, b in enumerate(NEXT_LABELS):
            row[f"p_{SHORT.get(b, b)}"] = round(float(stats["prob_mat"][i, j]), 6)
        prob_rows.append(row)
    write_csv(
        out_dir / "demo_transition_matrix.csv",
        ["after", "after_short", "n_completions"]
        + [f"p_{SHORT.get(b, b)}" for b in NEXT_LABELS],
        prob_rows,
    )

    count_rows = []
    for i, a in enumerate(TASKS):
        row = {"after": a, "after_short": SHORT[a], "n_completions": stats["a_done"][a]}
        for j, b in enumerate(NEXT_LABELS):
            row[f"n_{SHORT.get(b, b)}"] = int(stats["count_mat"][i, j])
        count_rows.append(row)
    write_csv(
        out_dir / "demo_transition_counts.csv",
        ["after", "after_short", "n_completions"]
        + [f"n_{SHORT.get(b, b)}" for b in NEXT_LABELS],
        count_rows,
    )


def plot_heatmap(stats: dict, out_dir: Path, label: str, demo_path: str) -> None:
    mat = stats["prob_mat"] * 100.0
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=80, aspect="auto")
    ax.set_xticks(range(len(NEXT_LABELS)))
    ax.set_xticklabels(
        [SHORT.get(t, t) for t in NEXT_LABELS],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([SHORT[t] for t in TASKS], fontsize=9)
    ax.set_xlabel("Next task (or END)")
    ax.set_ylabel("After completing")
    ax.set_title(
        f"Demo dataset ({label}) — P(next = B | completed = A)\n"
        f"n={stats['total']} MJL, 4-of-7 sequences\n{demo_path}",
        fontsize=10,
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
                    fontsize=8,
                    color="black" if v < 50 else "white",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of completions")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "02_demo_transition_heatmap"
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {base.with_suffix('.png')}")
    print(f"  wrote {base.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=DEFAULT_DEMO_DIR,
        help="Path to kitchen_demos_multitask/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for CSV + heatmap",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="FlowPolicy / shared kitchen demos",
        help="Label shown on the heatmap title",
    )
    args = parser.parse_args()

    demo_dir = args.demo_dir
    if not demo_dir.is_absolute():
        demo_dir = ROOT / demo_dir
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    if not demo_dir.is_dir():
        raise FileNotFoundError(demo_dir)

    resolved = demo_dir.resolve()
    print(f"Demo dir:  {demo_dir}")
    print(f"Resolved:  {resolved}")
    print(f"Output:    {out_dir}")
    print(f"Label:     {args.label}")

    demos = collect_demos(demo_dir)
    stats = analyze(demos)
    print(f"Total demos: {stats['total']} across {len(demos)} sequence folders")

    try:
        demo_rel = str(demo_dir.relative_to(ROOT))
    except ValueError:
        demo_rel = str(demo_dir)

    meta = {
        "label": args.label,
        "demo_dir": demo_rel,
        "demo_dir_resolved": str(resolved),
        "n_demos": stats["total"],
        "n_sequence_folders": len(demos),
        "same_as_flowpolicy_demos": str(resolved == FP_DEMO_RESOLVED),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_csvs(stats, out_dir, meta)
    plot_heatmap(stats, out_dir, args.label, demo_rel)
    print("\nDone.")


if __name__ == "__main__":
    main()
