"""Canonical paths for the best_val Kitchen re-eval tree.

Raw eval:  data/kitchen_eval_nfe100/<model>/seed_...
Plots:     data/kitchen_eval_plots/nfe100/
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "data" / "kitchen_eval_nfe100"
FP_ROOT = EVAL_ROOT / "flowpolicy"
DP_ROOT = EVAL_ROOT
PLOT_ROOT = ROOT / "data" / "kitchen_eval_plots" / "nfe100"

LEGACY_FP_ROOT = ROOT / "kripsy12/FlowPolicy/data/kitchen_eval_nfe100/flowpolicy"
LEGACY_DP_CANDIDATES = (
    ROOT / "diffusion_policy/data/kitchen_eval_nfe100_diffusion",
    ROOT / "diffusion_policy/data/kitchen_eval_nfe100",
)

FP_SEEDS = ("42", "43", "44")
DP_SEEDS = ("train0", "train1", "train2")
DP_MODEL_DIRS = ("diffusion_policy_cnn", "diffusion_policy_transformer")
BASELINE_MODEL_DIRS = (
    "LSTM_GMM",
    "implicit_behavior_cloning",
    "behavior_transformer",
)


def _has_metrics(root: Path) -> bool:
    return root.is_dir() and any(root.rglob("eval_metrics.json"))


def resolve_fp_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    if _has_metrics(FP_ROOT):
        return FP_ROOT
    if _has_metrics(LEGACY_FP_ROOT):
        return LEGACY_FP_ROOT
    return FP_ROOT


def resolve_dp_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    if _has_metrics(DP_ROOT):
        return DP_ROOT
    for cand in LEGACY_DP_CANDIDATES:
        if _has_metrics(cand):
            return cand
    return DP_ROOT


def normalize_fp_seed(seed: str) -> str:
    s = str(seed)
    if s.startswith("baseline_"):
        s = s[len("baseline_") :]
    if s.startswith("seed"):
        s = s[len("seed") :]
    return s


def _fp_dir_names(seed: str, nfe: int, sseed: int) -> List[str]:
    raw = str(seed)
    canon = normalize_fp_seed(raw)
    aliases = [raw, canon, f"seed{canon}", f"baseline_{canon}"]
    names: List[str] = []
    seen = set()
    for alias in aliases:
        name = f"seed_{alias}_nfe{nfe}_sseed{sseed}"
        if name not in seen:
            names.append(name)
            seen.add(name)
    compact = f"seed{canon}_NFE{nfe}"
    if compact not in seen:
        names.append(compact)
    return names


def fp_run_dir(
    fp_root: Path,
    seed: str,
    nfe: int,
    sseed: int = 0,
) -> Optional[Path]:
    for name in _fp_dir_names(seed, nfe, sseed):
        path = fp_root / name
        if (path / "eval_metrics.json").is_file():
            return path
    return None


def fp_seed_dirs(
    nfe: int,
    fp_root: Optional[Path] = None,
    sseed: int = 0,
) -> List[Path]:
    root = fp_root if fp_root is not None else resolve_fp_root()
    found: List[Path] = []
    for seed in FP_SEEDS:
        path = fp_run_dir(root, seed, nfe, sseed=sseed)
        if path is not None:
            found.append(path)
    return found


def dp_seed_dirs(
    model: str,
    nfe: int,
    dp_root: Optional[Path] = None,
    sseed: int = 0,
) -> List[Path]:
    root = dp_root if dp_root is not None else resolve_dp_root()
    return [
        root / model / f"seed_{seed}_nfe{nfe}_sseed{sseed}" for seed in DP_SEEDS
    ]


def native_seed_dirs(
    model: str,
    dp_root: Optional[Path] = None,
    sseed: int = 0,
) -> List[Path]:
    root = dp_root if dp_root is not None else resolve_dp_root()
    return [root / model / f"seed_{seed}_sseed{sseed}" for seed in DP_SEEDS]


def existing_dirs(paths: Iterable[Path]) -> List[Path]:
    return [p for p in paths if (p / "eval_metrics.json").is_file()]
