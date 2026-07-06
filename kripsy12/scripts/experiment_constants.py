"""Konstanta ruang pencarian hyperparameter untuk eksperimen FlowPolicy Kitchen.

Pencarian hiperparameter memakai Hyperband (Li et al., 2018,
https://arxiv.org/pdf/1603.06560). Karena ``training.num_epochs`` adalah
resource Hyperband (R), kunci ini DIKELUARKAN dari ``SEARCH_SPACE`` agar tidak
disampling sebagai dimensi pencarian. Nilai ``training.num_epochs`` yang
benar-benar dilatih untuk tiap baris ``results.csv`` tetap dicatat sebagai
kolom hiperparameter (``CSV_HPARAM_KEYS``) — untuk baseline = 3000,
untuk pemenang Hyperband final = R.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Root FlowPolicy/ (parent of scripts/).
_FLOWPOLICY_ROOT = Path(__file__).resolve().parent.parent / "FlowPolicy"


def resolve_dataset_dir_for_train(dataset_dir: str) -> str:
    """Normalisasi path demo MJL untuk ``train.py`` (cwd = ``FlowPolicy/``).

    Menerima path relatif ke repo (``FlowPolicy/data/...``), relatif ke
    ``FlowPolicy/`` (``data/...``), atau absolut.

    Jika path kanonik tidak ada, coba lokasi legacy
    ``FlowPolicy/FlowPolicy/data/...`` (sering terjadi saat dataset di-clone
    dengan path repo-relative sementara ``train.py`` sudah ``chdir`` ke
    ``FlowPolicy/``).
    """
    p = Path(dataset_dir)
    if p.is_absolute():
        return str(p.resolve())

    ds = dataset_dir.replace("\\", "/").strip("/")
    if ds.startswith("FlowPolicy/"):
        rel = ds[len("FlowPolicy/") :]
    else:
        rel = ds

    canonical = _FLOWPOLICY_ROOT / rel
    if canonical.is_dir():
        # Keep canonical path (do not follow symlinks into FlowPolicy/FlowPolicy/...).
        return str(canonical)

    legacy = _FLOWPOLICY_ROOT / "FlowPolicy" / rel
    if legacy.is_dir():
        return str(legacy)

    return str(canonical)

# Selaras dengan `flowpolicy.yaml` + `kitchen_lowdim_all` (Kitchen lowdim 7-task).
DEFAULT_BASELINE_HPARAMS = {
    "training.num_epochs": 3000,
    "optimizer.lr": 1e-4,
    "dataloader.batch_size": 128,
    "policy.Conditional_ConsistencyFM.num_segments": 2,
    "policy.Conditional_ConsistencyFM.eps": 1e-2,
    "policy.Conditional_ConsistencyFM.delta": 1e-2,
    "horizon": 4,
    "n_action_steps": 4,
    "n_obs_steps": 2,
    "policy.diffusion_step_embed_dim": 128,
    "_state_mlp_hidden": 64,
}

BASELINE_CFG_IDX = -1
# Pemenang final Hyperband yang di-rerun pada 3 seeds × 2 profiles.
HYPERBAND_BEST_CFG_IDX = -3
# Cfg_idx untuk konfigurasi yang dievaluasi di dalam fase Hyperband
# (mulai dari basis ini agar tidak bentrok dengan baseline / pemenang final).
HYPERBAND_CFG_IDX_BASE = 1000

# Ruang pencarian Hyperband (tanpa ``training.num_epochs`` — itu resource R).
SEARCH_SPACE = {
    "optimizer.lr": [1e-3, 5e-4, 1e-4, 1e-5],
    "dataloader.batch_size": [64, 128, 256, 512],
    "policy.Conditional_ConsistencyFM.num_segments": [1, 2, 3, 4],
    "policy.Conditional_ConsistencyFM.eps": [1e-4, 1e-3, 1e-2, 0.5],
    "policy.Conditional_ConsistencyFM.delta": [1e-4, 1e-3, 1e-2, 1.0],
    "n_action_steps": [4, 6, 8],
    "n_obs_steps": [2, 4],
    "policy.diffusion_step_embed_dim": [128, 256, 512, 1024],
    "_state_mlp_hidden": [128, 256, 512, 1024],
}

# Kolom hiperparameter di CSV (tanpa prefix policy untuk CFM agar rapi).
# Kolom pertama: ``training.num_epochs`` — nilai aktual epoch yang dilatih
# (baseline = 3000; Hyperband final winner = R; baris intermediate HB = r_i terakhir).
CSV_HPARAM_KEYS: List[str] = ["training.num_epochs"] + list(SEARCH_SPACE.keys())


def compute_horizon(n_obs_steps: int, n_action_steps: int) -> int:
    return 4 * ((max(n_obs_steps + n_action_steps - 1, 4) + 3) // 4)


def baseline_config_dict() -> dict:
    """Salinan baseline dengan cfg_idx untuk CSV dan orchestrator."""
    out = dict(DEFAULT_BASELINE_HPARAMS)
    out["cfg_idx"] = BASELINE_CFG_IDX
    return out


def sample_configs_hyperband(
    rng: np.random.RandomState,
    n: int,
    *,
    base_cfg_idx: int = HYPERBAND_CFG_IDX_BASE,
) -> List[Dict[str, Any]]:
    """Sample ``n`` konfigurasi random dari ``SEARCH_SPACE`` untuk Hyperband.

    ``training.num_epochs`` TIDAK disampling (= resource Hyperband). Field itu
    diset ke ``0`` di sini lalu di-overwrite menjadi ``r_i`` aktual saat training
    rung berjalan, dan menjadi ``R`` final saat config menyelesaikan rung terakhir.
    Cfg_idx unik global mulai dari ``base_cfg_idx``.
    """
    out: List[Dict[str, Any]] = []
    keys = list(SEARCH_SPACE.keys())
    for i in range(int(n)):
        d: Dict[str, Any] = {
            "cfg_idx": int(base_cfg_idx) + i,
            "training.num_epochs": 0,
        }
        for k in keys:
            choices = SEARCH_SPACE[k]
            d[k] = choices[int(rng.randint(0, len(choices)))]
        out.append(d)
    return out


# Kolom tambahan results.csv (metrik infer dua fase + alias kompatibel).
RESULTS_CSV_METRIC_COLUMNS = [
    "training_sim_success_rate_total",
    "training_sim_success_rate_k1",
    "training_sim_success_rate_k2",
    "training_sim_success_rate_k3",
    "training_sim_success_rate_k4",
    "training_sim_mean_inference_latency_ms",
    "training_sim_std_inference_latency_ms",
    "training_sim_mean_episode_mean_inference_latency_ms",
    "training_sim_std_episode_mean_inference_latency_ms",
    "training_sim_trade_off",
    "training_sim_trade_off_episode_latency",
    "training_sim_n_infer_episodes",
    "train_val_success_rate_total",
    "train_val_success_rate_k1",
    "train_val_success_rate_k2",
    "train_val_success_rate_k3",
    "train_val_success_rate_k4",
    "train_val_mean_inference_latency_ms",
    "train_val_std_inference_latency_ms",
    "train_val_mean_episode_mean_inference_latency_ms",
    "train_val_std_episode_mean_inference_latency_ms",
    "train_val_trade_off",
    "train_val_trade_off_episode_latency",
    "train_val_n_infer_episodes",
    "test_success_rate_total",
    "test_std_success_rate_total",
    "test_success_rate_k1",
    "test_success_rate_k2",
    "test_success_rate_k3",
    "test_success_rate_k4",
    "test_std_success_rate_k1",
    "test_std_success_rate_k2",
    "test_std_success_rate_k3",
    "test_std_success_rate_k4",
    "test_mean_inference_latency_ms",
    "test_std_inference_latency_ms",
    "test_mean_episode_mean_inference_latency_ms",
    "test_std_episode_mean_inference_latency_ms",
    "test_mean_execution_time_ms",
    "test_std_execution_time_ms",
    "test_total_execution_time_ms",
    "test_mean_all_tasks_execution_time_ms",
    "test_mean_task_execution_time_ms_k1",
    "test_std_task_execution_time_ms_k1",
    "test_mean_task_execution_time_ms_k2",
    "test_std_task_execution_time_ms_k2",
    "test_mean_task_execution_time_ms_k3",
    "test_std_task_execution_time_ms_k3",
    "test_mean_task_execution_time_ms_k4",
    "test_std_task_execution_time_ms_k4",
    "test_trade_off",
    "test_trade_off_episode_latency",
    "test_n_infer_episodes",
    "test_all_7_success",
    "test_std_all_7_success",
    "test_p1",
    "test_p2",
    "test_p3",
    "test_p4",
    "test_p5",
    "test_p6",
    "test_p7",
    "test_std_p1",
    "test_std_p2",
    "test_std_p3",
    "test_std_p4",
    "test_std_p5",
    "test_std_p6",
    "test_std_p7",
    "test_p4_paper",
    "test_std_p4_paper",
    "test_mean_episode_duration_ms",
    "test_std_episode_duration_ms",
    "success_rate_total",
    "std_success_rate_total",
    "success_rate_k1",
    "success_rate_k2",
    "success_rate_k3",
    "success_rate_k4",
    "std_success_rate_k1",
    "std_success_rate_k2",
    "std_success_rate_k3",
    "std_success_rate_k4",
    "mean_inference_latency_ms",
    "std_inference_latency_ms",
    "mean_episode_mean_inference_latency_ms",
    "std_episode_mean_inference_latency_ms",
    "mean_execution_time_ms",
    "std_execution_time_ms",
    "total_execution_time_ms",
    "mean_all_tasks_execution_time_ms",
    "mean_task_execution_time_ms_k1",
    "std_task_execution_time_ms_k1",
    "mean_task_execution_time_ms_k2",
    "std_task_execution_time_ms_k2",
    "mean_task_execution_time_ms_k3",
    "std_task_execution_time_ms_k3",
    "mean_task_execution_time_ms_k4",
    "std_task_execution_time_ms_k4",
    "trade_off",
    "trade_off_episode_latency",
]


def metrics_row_from_infer_json(met: Dict[str, Any]) -> Dict[str, Any]:
    """Isi kolom metrik CSV dari metrics.json (format baru bertahap atau legacy)."""

    def pick(*names: str, default: Any = "") -> Any:
        for n in names:
            if n in met and met[n] is not None:
                return met[n]
        return default

    has_tv = "train_val_success_rate_k1" in met
    has_ts = "training_sim_success_rate_k1" in met

    row: Dict[str, Any] = {}

    if has_ts:
        row["training_sim_success_rate_total"] = pick(
            "training_sim_success_rate_total"
        )
        row["training_sim_success_rate_k1"] = pick("training_sim_success_rate_k1")
        row["training_sim_success_rate_k2"] = pick("training_sim_success_rate_k2")
        row["training_sim_success_rate_k3"] = pick("training_sim_success_rate_k3")
        row["training_sim_success_rate_k4"] = pick("training_sim_success_rate_k4")
        row["training_sim_mean_inference_latency_ms"] = pick(
            "training_sim_mean_inference_latency_ms"
        )
        row["training_sim_std_inference_latency_ms"] = pick(
            "training_sim_std_inference_latency_ms"
        )
        row["training_sim_mean_episode_mean_inference_latency_ms"] = pick(
            "training_sim_mean_episode_mean_inference_latency_ms"
        )
        row["training_sim_std_episode_mean_inference_latency_ms"] = pick(
            "training_sim_std_episode_mean_inference_latency_ms"
        )
        row["training_sim_trade_off"] = pick("training_sim_trade_off")
        row["training_sim_trade_off_episode_latency"] = pick(
            "training_sim_trade_off_episode_latency"
        )
        row["training_sim_n_infer_episodes"] = pick("training_sim_n_infer_episodes")
    else:
        for c in (
            "training_sim_success_rate_total",
            "training_sim_success_rate_k1",
            "training_sim_success_rate_k2",
            "training_sim_success_rate_k3",
            "training_sim_success_rate_k4",
            "training_sim_mean_inference_latency_ms",
            "training_sim_std_inference_latency_ms",
            "training_sim_mean_episode_mean_inference_latency_ms",
            "training_sim_std_episode_mean_inference_latency_ms",
            "training_sim_trade_off",
            "training_sim_trade_off_episode_latency",
            "training_sim_n_infer_episodes",
        ):
            row[c] = ""

    if has_tv:
        row["train_val_success_rate_total"] = pick("train_val_success_rate_total")
        row["train_val_success_rate_k1"] = pick("train_val_success_rate_k1")
        row["train_val_success_rate_k2"] = pick("train_val_success_rate_k2")
        row["train_val_success_rate_k3"] = pick("train_val_success_rate_k3")
        row["train_val_success_rate_k4"] = pick("train_val_success_rate_k4")
        row["train_val_mean_inference_latency_ms"] = pick(
            "train_val_mean_inference_latency_ms"
        )
        row["train_val_std_inference_latency_ms"] = pick(
            "train_val_std_inference_latency_ms"
        )
        row["train_val_mean_episode_mean_inference_latency_ms"] = pick(
            "train_val_mean_episode_mean_inference_latency_ms"
        )
        row["train_val_std_episode_mean_inference_latency_ms"] = pick(
            "train_val_std_episode_mean_inference_latency_ms"
        )
        row["train_val_trade_off"] = pick("train_val_trade_off")
        row["train_val_trade_off_episode_latency"] = pick(
            "train_val_trade_off_episode_latency"
        )
        row["train_val_n_infer_episodes"] = pick("train_val_n_infer_episodes")
    else:
        for c in (
            "train_val_success_rate_total",
            "train_val_success_rate_k1",
            "train_val_success_rate_k2",
            "train_val_success_rate_k3",
            "train_val_success_rate_k4",
            "train_val_mean_inference_latency_ms",
            "train_val_std_inference_latency_ms",
            "train_val_mean_episode_mean_inference_latency_ms",
            "train_val_std_episode_mean_inference_latency_ms",
            "train_val_trade_off",
            "train_val_trade_off_episode_latency",
            "train_val_n_infer_episodes",
        ):
            row[c] = ""

    row["test_success_rate_total"] = pick(
        "test_success_rate_total", "success_rate_total"
    )
    row["test_std_success_rate_total"] = pick(
        "test_std_success_rate_total", "std_success_rate_total"
    )
    row["test_success_rate_k1"] = pick("test_success_rate_k1", "success_rate_k1")
    row["test_success_rate_k2"] = pick("test_success_rate_k2", "success_rate_k2")
    row["test_success_rate_k3"] = pick("test_success_rate_k3", "success_rate_k3")
    row["test_success_rate_k4"] = pick("test_success_rate_k4", "success_rate_k4")
    row["test_std_success_rate_k1"] = pick(
        "test_std_success_rate_k1", "std_success_rate_k1"
    )
    row["test_std_success_rate_k2"] = pick(
        "test_std_success_rate_k2", "std_success_rate_k2"
    )
    row["test_std_success_rate_k3"] = pick(
        "test_std_success_rate_k3", "std_success_rate_k3"
    )
    row["test_std_success_rate_k4"] = pick(
        "test_std_success_rate_k4", "std_success_rate_k4"
    )
    row["test_mean_inference_latency_ms"] = pick(
        "test_mean_inference_latency_ms", "mean_inference_latency_ms"
    )
    row["test_std_inference_latency_ms"] = pick(
        "test_std_inference_latency_ms", "std_inference_latency_ms"
    )
    row["test_mean_episode_mean_inference_latency_ms"] = pick(
        "test_mean_episode_mean_inference_latency_ms",
        "mean_episode_mean_inference_latency_ms",
    )
    row["test_std_episode_mean_inference_latency_ms"] = pick(
        "test_std_episode_mean_inference_latency_ms",
        "std_episode_mean_inference_latency_ms",
    )
    row["test_mean_execution_time_ms"] = pick(
        "test_mean_execution_time_ms", "mean_execution_time_ms"
    )
    row["test_std_execution_time_ms"] = pick(
        "test_std_execution_time_ms", "std_execution_time_ms"
    )
    row["test_total_execution_time_ms"] = pick(
        "test_total_execution_time_ms", "total_execution_time_ms"
    )
    row["test_mean_all_tasks_execution_time_ms"] = pick(
        "test_mean_all_tasks_execution_time_ms", "mean_all_tasks_execution_time_ms"
    )
    for ki in range(1, 5):
        row[f"test_mean_task_execution_time_ms_k{ki}"] = pick(
            f"test_mean_task_execution_time_ms_k{ki}",
            f"mean_task_execution_time_ms_k{ki}",
        )
        row[f"test_std_task_execution_time_ms_k{ki}"] = pick(
            f"test_std_task_execution_time_ms_k{ki}",
            f"std_task_execution_time_ms_k{ki}",
        )
    row["test_trade_off"] = pick("test_trade_off", "trade_off")
    row["test_trade_off_episode_latency"] = pick(
        "test_trade_off_episode_latency", "trade_off_episode_latency"
    )
    row["test_n_infer_episodes"] = pick(
        "test_n_infer_episodes", "n_infer_episodes"
    )

    row["test_all_7_success"] = pick(
        "test_all_7_success", "success_rate_total", "test_success_rate_total"
    )
    row["test_std_all_7_success"] = pick(
        "test_std_all_7_success", "std_success_rate_total", "test_std_success_rate_total"
    )
    for k in range(1, 8):
        row[f"test_p{k}"] = pick(f"test_p{k}")
        row[f"test_std_p{k}"] = pick(f"test_std_p{k}")
    row["test_p4_paper"] = pick("test_p4_paper", "success_rate_k4", "test_success_rate_k4")
    row["test_std_p4_paper"] = pick(
        "test_std_p4_paper", "std_success_rate_k4", "test_std_success_rate_k4"
    )
    row["test_mean_episode_duration_ms"] = pick("test_mean_episode_duration_ms")
    row["test_std_episode_duration_ms"] = pick("test_std_episode_duration_ms")

    row["success_rate_total"] = pick(
        "success_rate_total", "test_success_rate_total"
    )
    row["std_success_rate_total"] = pick(
        "std_success_rate_total", "test_std_success_rate_total"
    )
    row["success_rate_k1"] = pick("success_rate_k1", "test_success_rate_k1")
    row["success_rate_k2"] = pick("success_rate_k2", "test_success_rate_k2")
    row["success_rate_k3"] = pick("success_rate_k3", "test_success_rate_k3")
    row["success_rate_k4"] = pick("success_rate_k4", "test_success_rate_k4")
    row["std_success_rate_k1"] = pick(
        "std_success_rate_k1", "test_std_success_rate_k1"
    )
    row["std_success_rate_k2"] = pick(
        "std_success_rate_k2", "test_std_success_rate_k2"
    )
    row["std_success_rate_k3"] = pick(
        "std_success_rate_k3", "test_std_success_rate_k3"
    )
    row["std_success_rate_k4"] = pick(
        "std_success_rate_k4", "test_std_success_rate_k4"
    )
    row["mean_inference_latency_ms"] = pick(
        "mean_inference_latency_ms", "test_mean_inference_latency_ms"
    )
    row["std_inference_latency_ms"] = pick(
        "std_inference_latency_ms", "test_std_inference_latency_ms"
    )
    row["mean_episode_mean_inference_latency_ms"] = pick(
        "mean_episode_mean_inference_latency_ms",
        "test_mean_episode_mean_inference_latency_ms",
    )
    row["std_episode_mean_inference_latency_ms"] = pick(
        "std_episode_mean_inference_latency_ms",
        "test_std_episode_mean_inference_latency_ms",
    )
    row["mean_execution_time_ms"] = pick(
        "mean_execution_time_ms", "test_mean_execution_time_ms"
    )
    row["std_execution_time_ms"] = pick(
        "std_execution_time_ms", "test_std_execution_time_ms"
    )
    row["total_execution_time_ms"] = pick(
        "total_execution_time_ms", "test_total_execution_time_ms"
    )
    row["mean_all_tasks_execution_time_ms"] = pick(
        "mean_all_tasks_execution_time_ms",
        "test_mean_all_tasks_execution_time_ms",
    )
    for ki in range(1, 5):
        row[f"mean_task_execution_time_ms_k{ki}"] = pick(
            f"mean_task_execution_time_ms_k{ki}",
            f"test_mean_task_execution_time_ms_k{ki}",
        )
        row[f"std_task_execution_time_ms_k{ki}"] = pick(
            f"std_task_execution_time_ms_k{ki}",
            f"test_std_task_execution_time_ms_k{ki}",
        )
    row["trade_off"] = pick("trade_off", "test_trade_off")
    row["trade_off_episode_latency"] = pick(
        "trade_off_episode_latency", "test_trade_off_episode_latency"
    )

    return row


def empty_metrics_row() -> Dict[str, Any]:
    """Nilai kosong untuk semua kolom metrik results.csv."""
    return {k: "" for k in RESULTS_CSV_METRIC_COLUMNS}
