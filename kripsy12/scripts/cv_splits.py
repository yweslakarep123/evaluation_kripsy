"""
Partisi train/val/test pada level episode demo Kitchen (.mjl).

Default eksperimen: 70% train / 20% val / 10% test (~605 demo MJL).
Evaluasi policy utama dilakukan via rollout simulasi MuJoCo
(``infer_kitchen_lowdim.py``) — **bukan** replay episode test holdout.

Indeks episode = urutan sorted ``*/*.mjl`` di folder dataset (deterministik).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def count_kitchen_mjl_episodes(dataset_dir: Path) -> int:
    """Hitung jumlah file ``*/*.mjl`` (satu file = satu episode)."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset dir tidak ada: {dataset_dir}")
    n = len(sorted(dataset_dir.glob("*/*.mjl")))
    if n == 0:
        raise FileNotFoundError(
            f"Tidak ada file */*.mjl di {dataset_dir.resolve()}"
        )
    return n


def build_kitchen_demo_split(
    n_episodes: int,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    seed: int = 12345,
) -> Dict[str, Any]:
    """
    Satu partisi train/val/test untuk episode demo Kitchen MJL.

    1. Acak ``n_episodes`` indeks dengan ``seed``.
    2. Alokasi test → val → train (sisanya) agar total tepat ``n_episodes``.
    3. Minimal 1 episode per split jika ``n_episodes >= 3``.

    Contoh 605 episode, 70/20/10 → train≈424, val≈121, test≈60.
    Inferensi simulasi (50 episode × eval-seed) terpisah dari split demo ini.
    """
    if n_episodes < 3:
        raise ValueError(f"n_episodes minimal 3, dapat {n_episodes}")
    fracs = (float(train_frac), float(val_frac), float(test_frac))
    if any(f <= 0.0 for f in fracs):
        raise ValueError(f"Semua fraksi harus > 0, dapat {fracs}")
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"train+val+test fraksi harus = 1, dapat {sum(fracs)}")

    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n_episodes).tolist()

    n_test = max(1, int(round(n_episodes * test_frac)))
    n_val = max(1, int(round(n_episodes * val_frac)))
    n_train = n_episodes - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Split tidak valid: n_train={n_train} "
            f"(n={n_episodes}, fracs={fracs})"
        )

    test_episodes = sorted(perm[:n_test])
    val_episodes = sorted(perm[n_test : n_test + n_val])
    train_episodes = sorted(perm[n_test + n_val :])

    return {
        "fold": 0,
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "test_episodes": test_episodes,
        "n_episodes": int(n_episodes),
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "split_seed": int(seed),
        "n_train": len(train_episodes),
        "n_val": len(val_episodes),
        "n_test": len(test_episodes),
    }


def build_single_train_val_split(
    n_episodes: int = 19,
    held_out_test: int = 1,
    *,
    n_grid_partitions: int = 5,
    partition_index: int = 0,
    seed: int = 12345,
) -> Dict[str, Any]:
    """
    Legacy k-fold geometry (19 episode). Prefer ``build_kitchen_demo_split``.
    """
    folds = build_cv_splits(
        n_episodes=n_episodes,
        n_folds=n_grid_partitions,
        held_out_test=held_out_test,
        seed=seed,
    )
    if partition_index < 0 or partition_index >= len(folds):
        raise ValueError(
            f"partition_index {partition_index} tidak valid "
            f"(ada {len(folds)} partisi)."
        )
    return folds[partition_index]


def build_cv_splits(
    n_episodes: int = 19,
    n_folds: int = 5,
    held_out_test: int = 1,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Legacy k-fold splits."""
    if held_out_test < 1:
        raise ValueError("held_out_test minimal 1")
    if n_episodes < held_out_test + n_folds:
        raise ValueError(
            f"n_episodes ({n_episodes}) terlalu kecil untuk test={held_out_test} "
            f"dan {n_folds} fold."
        )

    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(np.arange(n_episodes)).tolist()
    test_episodes = sorted(perm[:held_out_test])
    rest = np.array(perm[held_out_test:], dtype=int)

    splits = np.array_split(rest, n_folds)
    folds: List[Dict[str, Any]] = []
    for k in range(n_folds):
        val_arr = splits[k]
        train_arr = np.concatenate([splits[i] for i in range(n_folds) if i != k])
        folds.append(
            {
                "fold": k,
                "train_episodes": sorted(train_arr.astype(int).tolist()),
                "val_episodes": sorted(val_arr.astype(int).tolist()),
                "test_episodes": list(test_episodes),
            }
        )
    return folds


def save_episode_split(path: str, split: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> None:
    payload = {"meta": meta or {}, "split": split}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_splits(path: str, folds: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    payload = {"meta": meta, "folds": folds}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
