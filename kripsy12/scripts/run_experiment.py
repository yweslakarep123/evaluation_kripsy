#!/usr/bin/env python3
"""
Orkestrator eksperimen (tanpa k-fold, satu partisi train/val/test):

  1) Baseline — hyperparameter default × len(seeds) × len(profiles)
  2) Pencarian — **Hyperband** (Li et al., 2018, https://arxiv.org/pdf/1603.06560).

Hyperband berjalan pada **satu seed × satu profile** (default: seed=0,
profile=standard) menggunakan ``val_loss`` sebagai sinyal early-stopping
antar-rung. Setelah Hyperband selesai, konfigurasi pemenang (val_loss
terkecil seenough across all evaluations sesuai paper) di-**rerun penuh**
pada semua ``seeds × profiles`` user (default 3 × 2 = 6 run) dengan training
+ inference + write ``results.csv`` ``status=ok`` — analog baseline.

Flag mutually exclusive:

- ``--baseline-only`` — hanya baseline; Hyperband dilewati.
- ``--hyperband-only`` — hanya Hyperband (skip baseline; butuh ``--dataset-dir``
  tetap valid).

Tanpa flag: jalankan baseline lalu Hyperband berurutan.

Metrik inferensi (Kitchen lowdim, 7 task): eval multi-seed ``0,42,101`` via
``infer_kitchen_lowdim.py`` — p1–p7, all-7 success, latensi, dan MP4 per episode.

Resume:

- Baseline & pemenang Hyperband (cfg_idx=-3): metrik lengkap (``metrics.json``)
  dilewati; juga dilewati jika baris ``results.csv`` yang sama sudah ``status=ok``.
- Training terputus dilanjutkan (resume Hydra) jika ada ``latest.ckpt`` tanpa
  ``training_final.json``; infer saja jika ``training_final.json`` + ckpt sudah
  ada tetapi belum ``metrics.json``.
- Hyperband: ``hyperband_state.json`` di ``--output-dir`` menyimpan state
  bracket + rung + ``val_loss`` per config — resume otomatis melewati rung yang
  sudah dievaluasi.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FLOWPOLICY_ROOT = REPO_ROOT / "FlowPolicy"

sys.path.insert(0, str(SCRIPT_DIR))
from cv_splits import (  # noqa: E402
    build_kitchen_demo_split,
    count_kitchen_mjl_episodes,
    save_episode_split,
)
from experiment_constants import (  # noqa: E402
    BASELINE_CFG_IDX,
    CSV_HPARAM_KEYS,
    HYPERBAND_BEST_CFG_IDX,
    RESULTS_CSV_METRIC_COLUMNS,
    baseline_config_dict,
    compute_horizon,
    empty_metrics_row,
    metrics_row_from_infer_json,
    resolve_dataset_dir_for_train,
)
from hyperband_search import run_hyperband  # noqa: E402


def _fmt_hydra_val(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return repr(float(v))
    return str(v)


def apply_vram_limits(cfg: Dict[str, Any], max_batch: int) -> Dict[str, Any]:
    c = dict(cfg)
    c["dataloader.batch_size"] = min(int(c["dataloader.batch_size"]), int(max_batch))
    return c


def load_or_create_config_bundle(
    configs_path: pathlib.Path,
    max_batch: int,
) -> Dict[str, Any]:
    """Muat / buat ``configs.json`` (``version: 5``) dengan baseline saja.

    Hyperband menyimpan state-nya di ``hyperband_state.json`` (lihat
    ``scripts/hyperband_search.py``). File ini hanya menyimpan baseline
    yang dipakai fase-1 dan re-run pemenang final.
    """
    baseline = apply_vram_limits(baseline_config_dict(), max_batch)

    raw: Any = None
    if configs_path.is_file():
        text = configs_path.read_text(encoding="utf-8").strip()
        if not text:
            print("[warn] configs.json kosong; akan dibuat ulang.")
        elif text:
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as e:
                print(f"[warn] configs.json bukan JSON valid ({e}); akan dibuat ulang.")

    if isinstance(raw, dict) and isinstance(raw.get("baseline"), dict):
        b = raw["baseline"]
        baseline = apply_vram_limits(
            {**baseline, **b, "cfg_idx": BASELINE_CFG_IDX}, max_batch
        )

    configs_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": 5,
        "search_mode": "hyperband",
        "baseline": baseline,
    }
    with open(configs_path, "w") as f:
        json.dump(bundle, f, indent=2)
    return baseline


def build_train_overrides(
    cfg: Dict[str, Any],
    *,
    seed: int,
    profile: str,
    train_eps: List[int],
    val_eps: List[int],
    run_dir: pathlib.Path,
    dataset_dir: str,
    episode_split_path: pathlib.Path,
    resume_training: bool,
    checkpoint_every: int,
    dataloader_num_workers: int,
) -> List[str]:
    n_obs = int(cfg["n_obs_steps"])
    n_act = int(cfg["n_action_steps"])
    hz = int(cfg["horizon"]) if "horizon" in cfg else compute_horizon(n_obs, n_act)
    bs = int(cfg["dataloader.batch_size"])
    robot_noise = 0.1 if profile == "standard" else 0.0
    dataset_dir_resolved = resolve_dataset_dir_for_train(dataset_dir)

    odl: List[str] = [
        "--config-name=flowpolicy_kitchen_lowdim",
        f"task.dataset.dataset_dir={dataset_dir_resolved}",
        f"task.dataset.episode_split_path={episode_split_path.resolve()}",
        "task.dataset.val_ratio=0.0",
        f"task.dataset.robot_noise_ratio={robot_noise}",
        f"task.robot_noise_ratio={robot_noise}",
        f"training.seed={seed}",
        f"task.dataset.seed={seed}",
        "training.compute_val_loss=true",
        "training.rollout_every=999999",
        f"training.resume={str(resume_training).lower()}",
        "checkpoint.save_ckpt=true",
        f"training.checkpoint_every={checkpoint_every}",
        "checkpoint.save_last_ckpt=true",
        "logging.mode=offline",
        f"hydra.run.dir={run_dir.resolve()}",
        "hydra.job.chdir=true",
        f"horizon={hz}",
        f"n_obs_steps={n_obs}",
        f"n_action_steps={n_act}",
        f"dataloader.batch_size={bs}",
        f"val_dataloader.batch_size={bs}",
        f"dataloader.num_workers={dataloader_num_workers}",
        f"val_dataloader.num_workers={dataloader_num_workers}",
    ]

    for k in CSV_HPARAM_KEYS:
        if k == "cfg_idx":
            continue
        if k == "_state_mlp_hidden":
            odl.append(f"policy.encoder_output_dim={_fmt_hydra_val(cfg[k])}")
            odl.append(
                f"policy.obs_mlp_hidden=[{_fmt_hydra_val(cfg[k])},{_fmt_hydra_val(cfg[k])}]"
            )
            continue
        odl.append(f"{k}={_fmt_hydra_val(cfg[k])}")
    return odl


def row_key_ok_exists(csv_path: pathlib.Path, key: Tuple[int, int, str, int]) -> bool:
    if not csv_path.is_file():
        return False
    df = pd.read_csv(csv_path)
    if df.empty:
        return False
    df["cfg_idx"] = df["cfg_idx"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["fold"] = df["fold"].astype(int)
    m = (
        (df["cfg_idx"] == int(key[0]))
        & (df["seed"] == int(key[1]))
        & (df["profile"].astype(str) == str(key[2]))
        & (df["fold"] == int(key[3]))
        & (df["status"].astype(str) == "ok")
    )
    return bool(m.any())


def append_results_csv(
    csv_path: pathlib.Path,
    row: Dict[str, Any],
    hp_cols: List[str],
) -> None:
    fieldnames = (
        ["cfg_idx", "seed", "profile", "fold"]
        + hp_cols
        + list(RESULTS_CSV_METRIC_COLUMNS)
        + [
            "train_loss_final",
            "val_loss_final",
            "n_infer_episodes",
            "checkpoint_path",
            "status",
        ]
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_training_final(run_dir: pathlib.Path) -> Tuple[Any, Any]:
    p = run_dir / "training_final.json"
    if not p.is_file():
        return None, None
    with open(p) as f:
        d = json.load(f)
    return d.get("train_loss_final"), d.get("val_loss_final")


def print_run_configuration(
    label: str,
    cfg: Dict[str, Any],
    overrides: List[str],
    run_dir: pathlib.Path,
) -> None:
    print("\n" + "=" * 72)
    print(label)
    print("Folder run:", run_dir.resolve())
    print("-" * 72)
    print("Hyperparameter (flat):")
    hp = {k: cfg[k] for k in CSV_HPARAM_KEYS if k in cfg}
    print(json.dumps(hp, indent=2, default=str))
    print("-" * 72)
    print("Override Hydra (train):")
    for line in sorted(overrides):
        print(" ", line)
    print("=" * 72 + "\n")


def sync_csv_from_metrics_if_needed(
    results_csv: pathlib.Path,
    hp_cols: List[str],
    cfg: Dict[str, Any],
    cfg_idx: int,
    seed: int,
    profile: str,
    fold_i: int,
    run_dir: pathlib.Path,
    ckpt_path: pathlib.Path,
    metrics_path: pathlib.Path,
) -> None:
    rk = (cfg_idx, seed, profile, fold_i)
    if row_key_ok_exists(results_csv, rk):
        return
    with open(metrics_path) as f:
        met = json.load(f)
    tr_l, va_l = load_training_final(run_dir)
    mrow = metrics_row_from_infer_json(met)
    append_results_csv(
        results_csv,
        {
            "cfg_idx": cfg_idx,
            "seed": seed,
            "profile": profile,
            "fold": fold_i,
            **{k: cfg[k] for k in hp_cols},
            **mrow,
            "train_loss_final": tr_l,
            "val_loss_final": va_l,
            "n_infer_episodes": met.get(
                "test_n_infer_episodes",
                met.get("n_infer_episodes"),
            ),
            "checkpoint_path": str(ckpt_path),
            "status": "skipped_resume",
        },
        hp_cols,
    )


def run_infer_subprocess(
    py: str,
    infer_py: pathlib.Path,
    cwd_train: str,
    env: dict,
    ckpt_path: pathlib.Path,
    metrics_path: pathlib.Path,
    n_infer_episodes: int,
    seed: int,
    *,
    eval_seeds: str = "0,42,101",
    skip_inference_videos: bool = False,
) -> int:
    env = dict(env)
    env.setdefault("MUJOCO_GL", "egl")
    cmd = [
        py,
        str(infer_py),
        "--checkpoint",
        str(ckpt_path),
        "--metrics-json",
        str(metrics_path),
        "--n-infer-episodes",
        str(n_infer_episodes),
        "--seed",
        str(seed),
        "--eval-seeds",
        eval_seeds,
    ]
    if skip_inference_videos:
        cmd.append("--skip-inference-videos")
    return subprocess.run(cmd, cwd=cwd_train, env=env).returncode


def execute_one_job(
    *,
    cfg: Dict[str, Any],
    cfg_idx: int,
    seed: int,
    profile: str,
    fold_i: int,
    fold_entry: Dict[str, Any],
    run_name: str,
    runs_root: pathlib.Path,
    results_csv: pathlib.Path,
    hp_cols: List[str],
    py: str,
    train_py: pathlib.Path,
    infer_py: pathlib.Path,
    cwd_train: str,
    dataset_dir: str,
    episode_split_path: pathlib.Path,
    n_infer_episodes: int,
    checkpoint_every: int,
    dataloader_num_workers: int,
    eval_seeds: str = "0,42,101",
    skip_inference_videos: bool = False,
    resume_from_results_csv: bool = True,
) -> None:
    run_dir = runs_root / run_name
    metrics_path = run_dir / "metrics.json"
    ckpt_path = run_dir / "checkpoints" / "latest.ckpt"
    training_final_path = run_dir / "training_final.json"
    rk = (cfg_idx, seed, profile, fold_i)

    if metrics_path.is_file():
        print(f"[skip] {run_name}: infer selesai (metrics.json ada)")
        sync_csv_from_metrics_if_needed(
            results_csv,
            hp_cols,
            cfg,
            cfg_idx,
            seed,
            profile,
            fold_i,
            run_dir,
            ckpt_path,
            metrics_path,
        )
        return

    if resume_from_results_csv and row_key_ok_exists(results_csv, rk):
        print(f"[skip] {run_name}: sudah tercatat status=ok di results.csv")
        return

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")

    infer_only = (
        ckpt_path.is_file()
        and training_final_path.is_file()
        and not metrics_path.is_file()
    )

    if infer_only:
        print(f"[infer-only] {run_name}: training_final.json + ckpt ada, lanjut inferensi")
        rc = run_infer_subprocess(
            py,
            infer_py,
            cwd_train,
            env,
            ckpt_path,
            metrics_path,
            n_infer_episodes,
            seed,
            eval_seeds=eval_seeds,
            skip_inference_videos=skip_inference_videos,
        )
        tr_l, va_l = load_training_final(run_dir)
        if rc != 0 or not metrics_path.is_file():
            append_results_csv(
                results_csv,
                {
                    "cfg_idx": cfg_idx,
                    "seed": seed,
                    "profile": profile,
                    "fold": fold_i,
                    **{k: cfg[k] for k in hp_cols},
                    **empty_metrics_row(),
                    "train_loss_final": tr_l,
                    "val_loss_final": va_l,
                    "n_infer_episodes": n_infer_episodes,
                    "checkpoint_path": str(ckpt_path),
                    "status": f"infer_failed_{rc}",
                },
                hp_cols,
            )
            return
        with open(metrics_path) as f:
            met = json.load(f)
        mrow = metrics_row_from_infer_json(met)
        append_results_csv(
            results_csv,
            {
                "cfg_idx": cfg_idx,
                "seed": seed,
                "profile": profile,
                "fold": fold_i,
                **{k: cfg[k] for k in hp_cols},
                **mrow,
                "train_loss_final": tr_l,
                "val_loss_final": va_l,
                "n_infer_episodes": met.get(
                    "test_n_infer_episodes",
                    met.get("n_infer_episodes", n_infer_episodes),
                ),
                "checkpoint_path": str(ckpt_path),
                "status": "ok",
            },
            hp_cols,
        )
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    resume_training = bool(
        ckpt_path.is_file() and not training_final_path.is_file()
    )
    if resume_training:
        print(f"[resume] {run_name}: melanjutkan training dari checkpoints/latest.ckpt")

    overrides = build_train_overrides(
        cfg,
        seed=seed,
        profile=profile,
        train_eps=fold_entry["train_episodes"],
        val_eps=fold_entry["val_episodes"],
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        episode_split_path=episode_split_path,
        resume_training=resume_training,
        checkpoint_every=checkpoint_every,
        dataloader_num_workers=dataloader_num_workers,
    )

    phase = (
        "BASELINE (default)"
        if cfg_idx == BASELINE_CFG_IDX
        else f"Pencarian hiperparameter cfg_idx={cfg_idx}"
    )
    print_run_configuration(
        f"[train] {run_name}  |  {phase}",
        cfg,
        overrides,
        run_dir,
    )

    r = subprocess.run([py, str(train_py)] + overrides, cwd=cwd_train, env=env)
    if r.returncode != 0:
        append_results_csv(
            results_csv,
            {
                "cfg_idx": cfg_idx,
                "seed": seed,
                "profile": profile,
                "fold": fold_i,
                **{k: cfg[k] for k in hp_cols},
                **empty_metrics_row(),
                "train_loss_final": "",
                "val_loss_final": "",
                "n_infer_episodes": "",
                "checkpoint_path": str(ckpt_path),
                "status": f"train_failed_{r.returncode}",
            },
            hp_cols,
        )
        return

    if not ckpt_path.is_file():
        append_results_csv(
            results_csv,
            {
                "cfg_idx": cfg_idx,
                "seed": seed,
                "profile": profile,
                "fold": fold_i,
                **{k: cfg[k] for k in hp_cols},
                **empty_metrics_row(),
                "train_loss_final": "",
                "val_loss_final": "",
                "n_infer_episodes": "",
                "checkpoint_path": str(ckpt_path),
                "status": "no_checkpoint",
            },
            hp_cols,
        )
        return

    print_run_configuration(
        f"[infer] {run_name}",
        cfg,
        [
            f"checkpoint={ckpt_path}",
            f"metrics_json={metrics_path}",
            f"n_infer_episodes={n_infer_episodes}",
            f"seed={seed}",
        ],
        run_dir,
    )
    r2 = run_infer_subprocess(
        py,
        infer_py,
        cwd_train,
        env,
        ckpt_path,
        metrics_path,
        n_infer_episodes,
        seed,
        eval_seeds=eval_seeds,
        skip_inference_videos=skip_inference_videos,
    )
    tr_l, va_l = load_training_final(run_dir)
    if r2 != 0 or not metrics_path.is_file():
        append_results_csv(
            results_csv,
            {
                "cfg_idx": cfg_idx,
                "seed": seed,
                "profile": profile,
                "fold": fold_i,
                **{k: cfg[k] for k in hp_cols},
                **empty_metrics_row(),
                "train_loss_final": tr_l,
                "val_loss_final": va_l,
                "n_infer_episodes": n_infer_episodes,
                "checkpoint_path": str(ckpt_path),
                "status": f"infer_failed_{r2}",
            },
            hp_cols,
        )
        return

    with open(metrics_path) as f:
        met = json.load(f)
    mrow = metrics_row_from_infer_json(met)
    append_results_csv(
        results_csv,
        {
            "cfg_idx": cfg_idx,
            "seed": seed,
            "profile": profile,
            "fold": fold_i,
            **{k: cfg[k] for k in hp_cols},
            **mrow,
            "train_loss_final": tr_l,
            "val_loss_final": va_l,
            "n_infer_episodes": met.get(
                "test_n_infer_episodes",
                met.get("n_infer_episodes", n_infer_episodes),
            ),
            "checkpoint_path": str(ckpt_path),
            "status": "ok",
        },
        hp_cols,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 101])
    ap.add_argument(
        "--profiles", type=str, nargs="+", default=["standard", "minimal"]
    )
    ap.add_argument(
        "--cv-seed",
        type=int,
        default=12345,
        help="Seed pembagian episode train/val (satu partisi, tanpa k-fold).",
    )
    ap.add_argument("--n-infer-episodes", type=int, default=50)
    ap.add_argument("--output-dir", type=str, default="outputs/experiment")
    ap.add_argument(
        "--results-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Jalur results.csv (relatif ke akar repo atau absolut). "
        "Default: <output-dir>/results.csv. Jika diisi, semua fase menulis ke file "
        "ini dan melewati job (cfg_idx, seed, profile, fold) yang sudah status=ok.",
    )
    ap.add_argument(
        "--dataset-dir",
        type=str,
        default="data/kitchen/kitchen_demos_multitask",
        help="Direktori demo MJL kitchen (relatif ke FlowPolicy/, atau absolut). "
        "Juga menerima prefix FlowPolicy/ dari akar repo.",
    )
    ap.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraksi train demo MJL. Default 0.7 (dengan val 0.2, test 0.1).",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraksi val demo MJL. Default 0.2.",
    )
    ap.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraksi test demo MJL (holdout; tidak dipakai infer MuJoCo). Default 0.1.",
    )
    ap.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Plafon batch size (training+val) untuk mengurangi risiko OOM pada VRAM ~16GB.",
    )
    ap.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Kurangi memori CPU/host; turunkan jika RAM habis.",
    )
    ap.add_argument(
        "--baseline-only",
        action="store_true",
        help="Hanya baseline (3 seed × 2 profil = 6 run default); tanpa Hyperband.",
    )
    ap.add_argument(
        "--hyperband-only",
        action="store_true",
        help="Hanya Hyperband + re-run pemenang top-1 (tanpa baseline).",
    )
    ap.add_argument(
        "--hyperband-max-epochs",
        type=int,
        default=3000,
        metavar="R",
        help="Hyperband: resource maksimum per konfigurasi (R, default 3000 = "
        "baseline default num_epochs).",
    )
    ap.add_argument(
        "--hyperband-eta",
        type=int,
        default=3,
        help="Hyperband: rasio downsampling antar-rung (eta, default 3 sesuai paper).",
    )
    ap.add_argument(
        "--hyperband-s-min",
        type=int,
        default=0,
        help="Hyperband: indeks bracket terkecil yang dijalankan (default 0 = semua "
        "bracket hingga s=0/random search). Naikkan ke 2 untuk hanya single-bracket "
        "SHA (lebih hemat waktu) — lihat README untuk anggaran waktu.",
    )
    ap.add_argument(
        "--hyperband-s-max",
        type=int,
        default=None,
        metavar="S",
        help="Hyperband: indeks bracket terbesar (default = floor(log_eta(R))). "
        "Cap di bawah nilai native untuk hindari bracket dengan banyak config kecil-r.",
    )
    ap.add_argument(
        "--hyperband-seed",
        type=int,
        default=99,
        help="Seed RNG sampling konfigurasi Hyperband (reproducible).",
    )
    ap.add_argument(
        "--hyperband-search-train-seed",
        type=int,
        default=0,
        help="Seed training yang dipakai SELAMA fase Hyperband (1 seed saja agar cepat).",
    )
    ap.add_argument(
        "--hyperband-search-profile",
        type=str,
        default="standard",
        help="Profil preprocessing yang dipakai SELAMA fase Hyperband (1 profil saja).",
    )
    ap.add_argument(
        "--n-train-val-episodes",
        type=int,
        default=0,
        help="Episode simulasi untuk metrik fase train/val; 0 = lewati (default).",
    )
    ap.add_argument(
        "--train-val-eval-seed-offset",
        type=int,
        default=31,
        help="Offset seed eval train/val vs test (infer_kitchen_lowdim).",
    )
    ap.add_argument(
        "--skip-inference-videos",
        action="store_true",
        help="Jangan simpan MP4 infer_ep_*.mp4 (hemat waktu/ruang).",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Simpan checkpoint berkala agar training bisa dilanjut setelah mesin mati.",
    )
    args = ap.parse_args()
    if args.baseline_only and args.hyperband_only:
        ap.error("--baseline-only dan --hyperband-only saling meniadakan.")

    out_root = (REPO_ROOT / args.output_dir).resolve()
    runs_root = out_root / "runs"
    plots_dir = out_root / "plots"
    runs_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs_path = out_root / "configs.json"
    cv_path = out_root / "cv_splits.json"
    if args.results_csv:
        _rcp = pathlib.Path(args.results_csv)
        results_csv = (
            _rcp.resolve() if _rcp.is_absolute() else (REPO_ROOT / _rcp).resolve()
        )
        results_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_csv = out_root / "results.csv"

    baseline_cfg = load_or_create_config_bundle(configs_path, args.max_batch_size)

    dataset_dir_resolved = pathlib.Path(
        resolve_dataset_dir_for_train(args.dataset_dir)
    )
    n_mjl_episodes = count_kitchen_mjl_episodes(dataset_dir_resolved)
    fold_entry = build_kitchen_demo_split(
        n_episodes=n_mjl_episodes,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.cv_seed),
    )
    episode_split_path = out_root / "episode_split.json"
    split_meta = {
        "dataset_dir": str(dataset_dir_resolved),
        "n_mjl_episodes": n_mjl_episodes,
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "cv_seed": int(args.cv_seed),
        "split_mode": "kitchen_demo_train_val_test",
        "note": "Split demo MJL train/val/test; eval policy via simulasi MuJoCo (infer_kitchen_lowdim.py)",
        "max_batch_size": args.max_batch_size,
        "hyperparam_search": "hyperband",
        "hyperband_max_epochs": int(args.hyperband_max_epochs),
        "hyperband_eta": int(args.hyperband_eta),
        "hyperband_s_min": int(args.hyperband_s_min),
        "hyperband_s_max": (
            None if args.hyperband_s_max is None else int(args.hyperband_s_max)
        ),
    }
    save_episode_split(str(episode_split_path), fold_entry, meta=split_meta)
    with open(cv_path, "w") as f:
        json.dump({"meta": split_meta, "folds": [fold_entry]}, f, indent=2)

    print(
        f"\n>>> Episode split ({n_mjl_episodes} demo MJL, seed={args.cv_seed}):\n"
        f"    train={fold_entry['n_train']}  val={fold_entry['n_val']}  "
        f"test={fold_entry['n_test']}  "
        f"({args.train_frac:.0%}/{args.val_frac:.0%}/{args.test_frac:.0%})\n"
        f"    infer policy = simulasi MuJoCo (infer_kitchen_lowdim.py), bukan replay demo test\n"
        f"    episode_split.json → {episode_split_path}\n"
    )

    py = sys.executable
    train_py = FLOWPOLICY_ROOT / "train.py"
    infer_py = FLOWPOLICY_ROOT / "infer_kitchen_lowdim.py"
    cwd_train = str(FLOWPOLICY_ROOT.resolve())
    hp_cols = list(CSV_HPARAM_KEYS)
    split_fold_idx = int(fold_entry["fold"])

    n_base = len(args.seeds) * len(args.profiles)
    n_final = len(args.seeds) * len(args.profiles)
    if args.baseline_only:
        print(
            "\n>>> Mode --baseline-only: hanya baseline "
            f"({n_base} run). Fase Hyperband dilewati.\n"
            "    Satu partisi train/val, tanpa k-fold.\n"
            f"    VRAM: max_batch_size={args.max_batch_size}, "
            f"num_workers={args.dataloader_num_workers}\n"
        )
    elif args.hyperband_only:
        print(
            "\n>>> Mode --hyperband-only: Hyperband (1 seed × 1 profile) "
            f"diikuti rerun top-1 pemenang di {n_final} run "
            f"({len(args.seeds)} seeds × {len(args.profiles)} profiles).\n"
            "    Baseline dilewati.\n"
            f"    R={args.hyperband_max_epochs}, eta={args.hyperband_eta}, "
            f"s_min={args.hyperband_s_min}, s_max={args.hyperband_s_max}\n"
            f"    VRAM: max_batch_size={args.max_batch_size}, "
            f"num_workers={args.dataloader_num_workers}\n"
        )
    else:
        print(
            "\n>>> Urutan: (1) Baseline "
            f"({n_base} run) → (2) Hyperband (1 seed × 1 profile) "
            f"→ (3) rerun top-1 pemenang ({n_final} run). "
            "Satu partisi train/val, tanpa k-fold.\n"
            f"    Hyperband: R={args.hyperband_max_epochs}, eta={args.hyperband_eta}, "
            f"s_min={args.hyperband_s_min}, s_max={args.hyperband_s_max}\n"
            f"    VRAM: max_batch_size={args.max_batch_size}, "
            f"num_workers={args.dataloader_num_workers}\n"
        )

    def run_grid_for_configs(
        cfgs: List[Dict[str, Any]],
    ) -> None:
        for cfg in cfgs:
            cfg_idx = int(cfg["cfg_idx"])
            for seed in args.seeds:
                for profile in args.profiles:
                    if cfg_idx == BASELINE_CFG_IDX:
                        run_name = f"baseline_seed{seed}_{profile}"
                    elif cfg_idx == HYPERBAND_BEST_CFG_IDX:
                        run_name = f"hb_best_seed{seed}_{profile}"
                    else:
                        run_name = f"cfg{cfg_idx}_seed{seed}_{profile}"
                    execute_one_job(
                        cfg=cfg,
                        cfg_idx=cfg_idx,
                        seed=seed,
                        profile=profile,
                        fold_i=split_fold_idx,
                        fold_entry=fold_entry,
                        run_name=run_name,
                        runs_root=runs_root,
                        results_csv=results_csv,
                        hp_cols=hp_cols,
                        py=py,
                        train_py=train_py,
                        infer_py=infer_py,
                        cwd_train=cwd_train,
                        dataset_dir=args.dataset_dir,
                        episode_split_path=episode_split_path,
                        n_infer_episodes=args.n_infer_episodes,
                        checkpoint_every=args.checkpoint_every,
                        dataloader_num_workers=args.dataloader_num_workers,
                        eval_seeds="0,42,101",
                        skip_inference_videos=args.skip_inference_videos,
                        resume_from_results_csv=True,
                    )

    def run_hyperband_phase() -> None:
        """Jalankan Hyperband (single seed × single profile), lalu rerun top-1
        pemenang pada full ``seeds × profiles`` dengan pipeline train + infer
        (cfg_idx=``HYPERBAND_BEST_CFG_IDX``)."""
        best = run_hyperband(
            out_root=out_root,
            runs_root=runs_root,
            R=int(args.hyperband_max_epochs),
            eta=int(args.hyperband_eta),
            s_min=int(args.hyperband_s_min),
            s_max=(None if args.hyperband_s_max is None else int(args.hyperband_s_max)),
            sampling_seed=int(args.hyperband_seed),
            search_train_seed=int(args.hyperband_search_train_seed),
            search_profile=str(args.hyperband_search_profile),
            train_eps=fold_entry["train_episodes"],
            val_eps=fold_entry["val_episodes"],
            dataset_dir=args.dataset_dir,
            episode_split_path=episode_split_path,
            checkpoint_every=args.checkpoint_every,
            dataloader_num_workers=args.dataloader_num_workers,
            py=py,
            train_py=train_py,
            cwd_train=cwd_train,
            apply_vram_limits_fn=apply_vram_limits,
            max_batch_size=args.max_batch_size,
        )
        if best is None:
            print(
                "[hyperband] WARNING: tidak ada pemenang; melewati fase rerun top-1."
            )
            return

        # Bangun config untuk rerun pemenang pada full ``seeds × profiles``.
        winner_cfg: Dict[str, Any] = dict(best["hparams"])
        winner_cfg["cfg_idx"] = HYPERBAND_BEST_CFG_IDX
        # Latih pemenang dengan resource MAKSIMUM (R epoch), bukan r_i intermediate.
        winner_cfg["training.num_epochs"] = int(args.hyperband_max_epochs)
        winner_cfg = apply_vram_limits(winner_cfg, args.max_batch_size)
        print(
            f"\n>>> Rerun pemenang Hyperband (cfg_idx={HYPERBAND_BEST_CFG_IDX}) "
            f"pada {len(args.seeds)} seeds × {len(args.profiles)} profiles "
            f"@ training.num_epochs={int(args.hyperband_max_epochs)}.\n"
        )
        run_grid_for_configs([winner_cfg])

    if args.baseline_only:
        run_grid_for_configs([baseline_cfg])
    elif args.hyperband_only:
        run_hyperband_phase()
    else:
        run_grid_for_configs([baseline_cfg])
        run_hyperband_phase()

    summarize_script = SCRIPT_DIR / "summarize.py"
    plot_script = SCRIPT_DIR / "plot_results.py"
    _csv_args: List[str] = (
        ["--results-csv", str(results_csv)] if args.results_csv else []
    )
    subprocess.run(
        [py, str(summarize_script), "--output-dir", str(out_root)] + _csv_args,
        check=False,
    )
    subprocess.run(
        [py, str(plot_script), "--output-dir", str(out_root)] + _csv_args,
        check=False,
    )


if __name__ == "__main__":
    main()
