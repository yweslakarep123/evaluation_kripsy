#!/usr/bin/env python3
"""Hyperband (Li et al., 2018; https://arxiv.org/pdf/1603.06560).

Implementasi mengikuti **Algoritma 1** pada paper:

::

    input: R, eta (default eta = 3)
    initialization: s_max = floor(log_eta(R)), B = (s_max + 1) * R

    for s in {s_max, s_max-1, ..., 0}:
        n = ceil((B / R) * eta^s / (s + 1))
        r = R * eta^(-s)
        # begin SuccessiveHalving with (n, r) inner loop
        T = get_hyperparameter_configuration(n)
        for i in {0, ..., s}:
            n_i = floor(n * eta^(-i))
            r_i = r * eta^i
            L = {run_then_return_val_loss(t, r_i) : t in T}
            T = top_k(T, L, floor(n_i / eta))
        end
    end
    return configuration with the smallest intermediate loss seen so far

Modul ini bersifat berkas-tunggal (single-file) — dipakai langsung oleh
``scripts/run_experiment.py``. Resource = jumlah epoch training FlowPolicy.
Sinyal "val_loss" antar-rung diambil dari ``training_final.json.val_loss_final``
yang ditulis oleh ``FlowPolicy/train.py`` di akhir setiap pelatihan.

Catatan deviasi praktis (didokumentasikan oleh paper, Section 5):
**reuse of trained weights between rungs**. Setelah rung ke-``i`` selesai,
folder run config yang lolos top-k dipertahankan; rung ke-``i+1`` melanjutkan
training dari ``latest.ckpt`` (``training.resume=true``) dengan
``training.num_epochs = r_{i+1} - r_i`` (delta) — bukan absolut, karena loop
training ``for local_epoch_idx in range(num_epochs)`` di ``train.py``
berjalan relatif terhadap kondisi saat resume. Ini menghemat banyak compute
dibanding melatih ulang dari awal pada setiap rung.

State persisten: ``<output-dir>/hyperband_state.json`` (lihat ``_load_state``
dan ``_save_state``). Eksekusi yang terputus dapat dilanjutkan: rung/config
yang sudah ada ``val_loss`` di state akan dilewati.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_constants import (  # noqa: E402
    CSV_HPARAM_KEYS,
    HYPERBAND_CFG_IDX_BASE,
    SEARCH_SPACE,
    compute_horizon,
    resolve_dataset_dir_for_train,
    sample_configs_hyperband,
)


# ---------------------------------------------------------------------------
# Penghitungan bracket sesuai Algoritma 1
# ---------------------------------------------------------------------------


@dataclass
class Rung:
    i: int
    n_i: int
    r_i: int  # dibulatkan ke integer epoch (ceil agar paling tidak melatih sebanyak r_i)


@dataclass
class Bracket:
    s: int
    n: int
    r: int  # dibulatkan integer
    rungs: List[Rung] = field(default_factory=list)


def compute_brackets(
    R: int,
    eta: int,
    *,
    s_min: int = 0,
    s_max: Optional[int] = None,
) -> List[Bracket]:
    """Bangun daftar bracket Hyperband untuk ``s in [s_max, ..., s_min]``.

    Mengikuti **kode referensi Jamieson** (coauthor paper,
    https://homes.cs.washington.edu/~jamieson/hyperband.html) yang
    mereproduksi Tabel 1 paper persis:

    - ``s_max_native = floor(log_eta(R))``
    - ``B = (s_max_native + 1) * R``
    - ``n = ceil( int(B / R / (s + 1)) * eta^s )`` — perhatikan ``int(...)``
      (integer division / floor) dilakukan SEBELUM perkalian ``eta^s``.
      Pseudo-code Algorithm 1 di paper menulis
      ``n = ceil((B/R) * eta^s / (s+1))`` tetapi Tabel 1 dan implementasi
      referensi memakai bentuk integer-divide-dulu ini.
    - ``r = R * eta^(-s)``
    - Rung ``i in {0, ..., s}``: ``n_i = floor(n * eta^(-i))``,
      ``r_i = r * eta^i``.

    ``s_max`` dapat di-cap di bawah ``s_max_native`` untuk membatasi waktu.
    ``B`` tetap memakai ``s_max_native`` (anggaran teoritis penuh) walaupun
    ``s_max`` dipotong — ini menjaga proporsi ``n_s`` per bracket konsisten
    dengan Hyperband asli yang dijalankan penuh.
    """
    if R <= 0:
        raise ValueError(f"R harus > 0, dapat {R}")
    if eta < 2:
        raise ValueError(f"eta harus >= 2, dapat {eta}")
    s_max_native = int(math.floor(math.log(R) / math.log(eta)))
    if s_max is None:
        s_max = s_max_native
    s_max = int(min(max(0, s_max), s_max_native))
    s_min = int(min(max(0, s_min), s_max))
    B = (s_max_native + 1) * R

    brackets: List[Bracket] = []
    for s in range(s_max, s_min - 1, -1):
        # Reference Jamieson:
        #   n = int(ceil(int(B / max_iter / (s+1)) * eta**s))
        # ``int(...)`` di Python untuk float positif sama dengan floor.
        n = int(math.ceil(int(B / R / (s + 1)) * (eta ** s)))
        r = R * (eta ** (-s))
        rungs: List[Rung] = []
        for i in range(s + 1):
            n_i = int(math.floor(n * (eta ** (-i))))
            r_i = r * (eta ** i)
            # Pembulatan epoch: rung terakhir = R persis agar config penyelamat
            # benar-benar dilatih sampai resource maksimum.
            if i == s:
                r_i_int = int(R)
            else:
                r_i_int = max(1, int(round(r_i)))
            n_i = max(1, n_i)
            rungs.append(Rung(i=i, n_i=n_i, r_i=r_i_int))
        brackets.append(Bracket(s=s, n=n, r=max(1, int(round(r))), rungs=rungs))
    return brackets


# ---------------------------------------------------------------------------
# Helper override Hydra (mirip ``run_experiment.build_train_overrides`` tapi
# disesuaikan untuk Hyperband: tanpa CSV row, fixed search seed+profile,
# resume training inkremental antar-rung).
# ---------------------------------------------------------------------------


def _fmt_hydra_val(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return repr(float(v))
    return str(v)


def _build_train_overrides_hb(
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
    delta_num_epochs: int,
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
        f"training.num_epochs={int(delta_num_epochs)}",
    ]
    for k in CSV_HPARAM_KEYS:
        if k in ("cfg_idx", "training.num_epochs"):
            continue
        if k == "_state_mlp_hidden":
            odl.append(f"policy.encoder_output_dim={_fmt_hydra_val(cfg[k])}")
            odl.append(
                f"policy.obs_mlp_hidden=[{_fmt_hydra_val(cfg[k])},{_fmt_hydra_val(cfg[k])}]"
            )
            continue
        odl.append(f"{k}={_fmt_hydra_val(cfg[k])}")
    return odl


def _read_val_loss_final(run_dir: pathlib.Path) -> Optional[float]:
    p = run_dir / "training_final.json"
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return None
    v = d.get("val_loss_final")
    if v is None:
        return None
    try:
        vf = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(vf) or math.isinf(vf):
        return None
    return vf


def _run_dir_for_cfg(runs_root: pathlib.Path, cfg_idx: int) -> pathlib.Path:
    return runs_root / f"hb_cfg{int(cfg_idx)}"


# ---------------------------------------------------------------------------
# State persisten (hyperband_state.json)
# ---------------------------------------------------------------------------


def _state_path(out_root: pathlib.Path) -> pathlib.Path:
    return out_root / "hyperband_state.json"


def _load_state(out_root: pathlib.Path) -> Optional[Dict[str, Any]]:
    p = _state_path(out_root)
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _save_state(out_root: pathlib.Path, state: Dict[str, Any]) -> None:
    p = _state_path(out_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, p)


def _build_initial_state(
    *,
    R: int,
    eta: int,
    s_min: int,
    s_max: int,
    sampling_seed: int,
    search_train_seed: int,
    search_profile: str,
    brackets: List[Bracket],
    sampled_per_bracket: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    bracket_states: List[Dict[str, Any]] = []
    for br, cfgs in zip(brackets, sampled_per_bracket):
        configs_state: List[Dict[str, Any]] = []
        for cfg in cfgs:
            configs_state.append(
                {
                    "cfg_idx": int(cfg["cfg_idx"]),
                    "hparams": {k: cfg[k] for k in CSV_HPARAM_KEYS if k in cfg},
                    "epoch_trained": 0,
                    "evaluations": [],  # list of {"i": int, "r_i": int, "val_loss": float|None}
                    "active": True,  # masih bertahan (belum di-cull)
                }
            )
        bracket_states.append(
            {
                "s": br.s,
                "n": br.n,
                "r": br.r,
                "rungs": [{"i": rg.i, "n_i": rg.n_i, "r_i": rg.r_i} for rg in br.rungs],
                "configs": configs_state,
                "completed_rungs": [],  # daftar rung_i yang sudah selesai dievaluasi
                "survivors_final": [],  # cfg_idx yang lolos rung terakhir
            }
        )
    return {
        "version": 5,
        "algorithm": "hyperband",
        "R": int(R),
        "eta": int(eta),
        "s_min": int(s_min),
        "s_max": int(s_max),
        "sampling_seed": int(sampling_seed),
        "search_train_seed": int(search_train_seed),
        "search_profile": str(search_profile),
        "brackets": bracket_states,
        "best": None,  # diisi ketika selesai: {"cfg_idx", "hparams", "val_loss", "r_i", "bracket_s"}
        "all_evaluations": [],  # untuk "smallest intermediate loss seen so far"
    }


# ---------------------------------------------------------------------------
# Core: jalankan satu evaluasi (train ke target epoch via subprocess train.py)
# ---------------------------------------------------------------------------


def _evaluate_config_at_rung(
    *,
    cfg: Dict[str, Any],
    target_epoch: int,
    already_trained: int,
    run_dir: pathlib.Path,
    py: str,
    train_py: pathlib.Path,
    cwd_train: str,
    seed: int,
    profile: str,
    train_eps: List[int],
    val_eps: List[int],
    dataset_dir: str,
    episode_split_path: pathlib.Path,
    checkpoint_every: int,
    dataloader_num_workers: int,
) -> Tuple[Optional[float], int, int]:
    """Latih ``cfg`` hingga ``target_epoch`` total (inkremental dari
    ``already_trained``). Kembalikan ``(val_loss, returncode, epoch_trained)``.

    Implementasi: hapus ``training_final.json`` lama agar train.py tidak
    mengira sudah selesai; set ``training.resume=true`` jika sudah ada ckpt;
    ``training.num_epochs = target_epoch - already_trained``.
    """
    delta = int(target_epoch) - int(already_trained)
    if delta <= 0:
        # Sudah lebih dari target — pakai val_loss yang ada.
        v = _read_val_loss_final(run_dir)
        return v, 0, int(already_trained)

    run_dir.mkdir(parents=True, exist_ok=True)

    # Hapus training_final.json lama agar train.py melanjutkan training.
    tf = run_dir / "training_final.json"
    if tf.is_file():
        try:
            tf.unlink()
        except Exception:
            pass

    ckpt = run_dir / "checkpoints" / "latest.ckpt"
    resume = bool(ckpt.is_file())

    overrides = _build_train_overrides_hb(
        cfg,
        seed=seed,
        profile=profile,
        train_eps=train_eps,
        val_eps=val_eps,
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        episode_split_path=episode_split_path,
        resume_training=resume,
        delta_num_epochs=delta,
        checkpoint_every=checkpoint_every,
        dataloader_num_workers=dataloader_num_workers,
    )
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")

    cmd = [py, str(train_py)] + overrides
    print("\n" + "-" * 72)
    print(
        f"[hyperband] train cfg_idx={cfg['cfg_idx']} → target r_i={target_epoch} "
        f"(delta={delta} epoch, resume={resume})"
    )
    print("-" * 72)
    rc = subprocess.run(cmd, cwd=cwd_train, env=env).returncode
    v = _read_val_loss_final(run_dir)
    return v, int(rc), int(target_epoch) if rc == 0 else int(already_trained)


# ---------------------------------------------------------------------------
# Successive Halving (inner loop) + Hyperband (outer loop)
# ---------------------------------------------------------------------------


def _top_k_by_val_loss(
    cfg_idxs: List[int],
    losses: Dict[int, Optional[float]],
    k: int,
) -> List[int]:
    """Sort cfg_idxs ascending by val_loss (None / NaN -> +inf), return first k."""
    def keyfn(c: int) -> float:
        v = losses.get(c)
        if v is None:
            return float("inf")
        try:
            vf = float(v)
        except (TypeError, ValueError):
            return float("inf")
        if math.isnan(vf) or math.isinf(vf):
            return float("inf")
        return vf

    ordered = sorted(cfg_idxs, key=keyfn)
    return ordered[: max(0, int(k))]


def _successive_halving_for_bracket(
    *,
    bracket_state: Dict[str, Any],
    out_root: pathlib.Path,
    state: Dict[str, Any],
    runs_root: pathlib.Path,
    py: str,
    train_py: pathlib.Path,
    cwd_train: str,
    seed: int,
    profile: str,
    train_eps: List[int],
    val_eps: List[int],
    dataset_dir: str,
    episode_split_path: pathlib.Path,
    checkpoint_every: int,
    dataloader_num_workers: int,
    eta: int,
) -> None:
    """Eksekusi SH untuk satu bracket (mengubah ``bracket_state`` in-place)."""
    s = int(bracket_state["s"])
    rungs = bracket_state["rungs"]

    # Daftar cfg_idx aktif saat ini (yang lolos sampai rung saat ini).
    active_cfg_idxs: List[int] = [
        c["cfg_idx"] for c in bracket_state["configs"] if c.get("active", True)
    ]

    cfg_by_idx: Dict[int, Dict[str, Any]] = {
        c["cfg_idx"]: c for c in bracket_state["configs"]
    }

    for rung in rungs:
        i = int(rung["i"])
        r_i = int(rung["r_i"])
        n_i_planned = int(rung["n_i"])

        # Lewati jika sudah pernah selesai sepenuhnya.
        completed = set(int(x) for x in bracket_state.get("completed_rungs", []))
        if i in completed:
            # Tetap update active_cfg_idxs dari evaluations.
            losses_i: Dict[int, Optional[float]] = {}
            for ci in active_cfg_idxs:
                cstate = cfg_by_idx[ci]
                ev = next(
                    (e for e in cstate["evaluations"] if int(e["i"]) == i), None
                )
                losses_i[ci] = ev.get("val_loss") if ev else None
            # Cull untuk lanjut ke rung berikut (jika ada).
            if i < s:
                k_keep = int(math.floor(n_i_planned / eta))
                survivors = _top_k_by_val_loss(active_cfg_idxs, losses_i, k_keep)
                for ci in active_cfg_idxs:
                    if ci not in survivors:
                        cfg_by_idx[ci]["active"] = False
                active_cfg_idxs = survivors
            continue

        print("\n" + "=" * 72)
        print(
            f"[hyperband] Bracket s={s}, rung i={i}/{s} | "
            f"n_i={len(active_cfg_idxs)} configs → r_i={r_i} epoch"
        )
        print("=" * 72)

        losses_i = {}
        for ci in active_cfg_idxs:
            cstate = cfg_by_idx[ci]
            # Apakah evaluasi rung-i sudah ada (resume mesin mati)?
            ev_existing = next(
                (e for e in cstate["evaluations"] if int(e["i"]) == i), None
            )
            if ev_existing is not None and ev_existing.get("val_loss") is not None:
                print(
                    f"[skip] cfg_idx={ci}: rung i={i} sudah dievaluasi "
                    f"(val_loss={ev_existing['val_loss']:.6f})"
                )
                losses_i[ci] = ev_existing["val_loss"]
                continue

            cfg = {k: cstate["hparams"][k] for k in cstate["hparams"]}
            cfg["cfg_idx"] = ci
            run_dir = _run_dir_for_cfg(runs_root, ci)
            already = int(cstate.get("epoch_trained", 0))

            val_loss, rc, ep_trained = _evaluate_config_at_rung(
                cfg=cfg,
                target_epoch=r_i,
                already_trained=already,
                run_dir=run_dir,
                py=py,
                train_py=train_py,
                cwd_train=cwd_train,
                seed=seed,
                profile=profile,
                train_eps=train_eps,
                val_eps=val_eps,
                dataset_dir=dataset_dir,
                episode_split_path=episode_split_path,
                checkpoint_every=checkpoint_every,
                dataloader_num_workers=dataloader_num_workers,
            )

            cstate["epoch_trained"] = ep_trained
            ev_new = {
                "i": i,
                "r_i": r_i,
                "val_loss": val_loss,
                "rc": int(rc),
            }
            # Hapus eval lama untuk rung-i yang masih NULL (gagal), lalu append.
            cstate["evaluations"] = [
                e for e in cstate["evaluations"] if int(e["i"]) != i
            ] + [ev_new]
            losses_i[ci] = val_loss

            # Catat ke all_evaluations (untuk pemilihan "smallest intermediate loss").
            state["all_evaluations"].append(
                {
                    "bracket_s": s,
                    "rung_i": i,
                    "r_i": r_i,
                    "cfg_idx": ci,
                    "val_loss": val_loss,
                }
            )
            _save_state(out_root, state)

        # Tandai rung selesai.
        bracket_state["completed_rungs"] = sorted(
            set(int(x) for x in bracket_state.get("completed_rungs", [])) | {i}
        )

        # Cull: top floor(n_i / eta).
        if i < s:
            k_keep = int(math.floor(n_i_planned / eta))
            survivors = _top_k_by_val_loss(active_cfg_idxs, losses_i, k_keep)
            for ci in active_cfg_idxs:
                if ci not in survivors:
                    cfg_by_idx[ci]["active"] = False
                    # Hemat disk: hapus folder run yang sudah tidak aktif.
                    run_dir = _run_dir_for_cfg(runs_root, ci)
                    if run_dir.is_dir():
                        try:
                            shutil.rmtree(run_dir)
                        except Exception:
                            pass
            active_cfg_idxs = survivors
            print(
                f"[hyperband] cull: top-{k_keep} survivor = {survivors}"
            )
        else:
            # Rung terakhir (i == s): tidak ada cull lagi.
            bracket_state["survivors_final"] = list(active_cfg_idxs)
            print(
                f"[hyperband] bracket s={s} selesai. survivors_final={active_cfg_idxs}"
            )

        _save_state(out_root, state)


def _pick_best_from_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pilih konfigurasi dengan ``val_loss`` TERKECIL dari seluruh
    ``all_evaluations`` lintas semua bracket dan rung — persis seperti
    paper Algorithm 1: "Configuration with the smallest intermediate loss
    seen so far".

    Catatan: pemenang BISA berasal dari rung dengan ``r_i < R``. Itu sesuai
    paper. Untuk evaluasi akhir yang adil, ``run_experiment.py`` melatih ulang
    pemenang dari scratch (atau lanjut dari ckpt-nya hingga R epoch penuh)
    pada 3 seeds × 2 profiles.
    """
    if not state["all_evaluations"]:
        return None

    def keyfn(e: Dict[str, Any]) -> float:
        v = e.get("val_loss")
        if v is None:
            return float("inf")
        try:
            vf = float(v)
        except (TypeError, ValueError):
            return float("inf")
        if math.isnan(vf) or math.isinf(vf):
            return float("inf")
        return vf

    ordered = sorted(state["all_evaluations"], key=keyfn)
    best_ev = ordered[0]
    if keyfn(best_ev) == float("inf"):
        return None

    cfg_idx = int(best_ev["cfg_idx"])
    for br in state["brackets"]:
        for cstate in br["configs"]:
            if int(cstate["cfg_idx"]) == cfg_idx:
                return {
                    "cfg_idx": cfg_idx,
                    "hparams": dict(cstate["hparams"]),
                    "val_loss": float(best_ev["val_loss"]),
                    "r_i": int(best_ev["r_i"]),
                    "bracket_s": int(best_ev["bracket_s"]),
                }
    return None


# ---------------------------------------------------------------------------
# Entry: run_hyperband
# ---------------------------------------------------------------------------


def run_hyperband(
    *,
    out_root: pathlib.Path,
    runs_root: pathlib.Path,
    R: int,
    eta: int,
    s_min: int,
    s_max: Optional[int],
    sampling_seed: int,
    search_train_seed: int,
    search_profile: str,
    train_eps: List[int],
    val_eps: List[int],
    dataset_dir: str,
    episode_split_path: pathlib.Path,
    checkpoint_every: int,
    dataloader_num_workers: int,
    py: str,
    train_py: pathlib.Path,
    cwd_train: str,
    apply_vram_limits_fn,
    max_batch_size: int,
) -> Optional[Dict[str, Any]]:
    """Jalankan Hyperband (Algorithm 1 paper) dan kembalikan pemenang.

    ``apply_vram_limits_fn(cfg, max_batch_size)`` di-passing dari
    ``run_experiment.py`` untuk membatasi ``dataloader.batch_size`` agar tidak
    OOM (knob VRAM yang sama dipakai baseline).

    Return: dict ``{cfg_idx, hparams, val_loss, r_i, bracket_s}`` atau ``None``
    jika tidak ada evaluasi sukses.
    """
    out_root = pathlib.Path(out_root).resolve()
    runs_root = pathlib.Path(runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    brackets = compute_brackets(R, eta, s_min=s_min, s_max=s_max)
    if not brackets:
        raise RuntimeError("compute_brackets mengembalikan daftar kosong.")

    # Resume jika state ada dan parameter cocok; jika tidak, buat ulang.
    state = _load_state(out_root)
    reuse_state = False
    if state is not None:
        same = (
            int(state.get("R", -1)) == int(R)
            and int(state.get("eta", -1)) == int(eta)
            and int(state.get("s_min", -1)) == int(s_min)
            and int(state.get("s_max", -1)) == int(brackets[0].s)
            and int(state.get("sampling_seed", -1)) == int(sampling_seed)
            and int(state.get("search_train_seed", -1)) == int(search_train_seed)
            and str(state.get("search_profile", "")) == str(search_profile)
        )
        if same:
            reuse_state = True
        else:
            print(
                "[hyperband] parameter berubah vs hyperband_state.json — "
                "membuat state baru (file lama tetap di disk)."
            )

    if not reuse_state:
        rng = np.random.RandomState(int(sampling_seed))
        sampled_per_bracket: List[List[Dict[str, Any]]] = []
        next_cfg_idx = int(HYPERBAND_CFG_IDX_BASE)
        for br in brackets:
            cfgs = sample_configs_hyperband(rng, br.n, base_cfg_idx=next_cfg_idx)
            cfgs = [apply_vram_limits_fn(c, max_batch_size) for c in cfgs]
            sampled_per_bracket.append(cfgs)
            next_cfg_idx += br.n
        state = _build_initial_state(
            R=R,
            eta=eta,
            s_min=s_min,
            s_max=int(brackets[0].s),
            sampling_seed=sampling_seed,
            search_train_seed=search_train_seed,
            search_profile=search_profile,
            brackets=brackets,
            sampled_per_bracket=sampled_per_bracket,
        )
        _save_state(out_root, state)

    # Eksekusi bracket secara berurutan (dari s_max ke s_min).
    for bracket_state in state["brackets"]:
        if bracket_state.get("survivors_final"):
            print(
                f"[skip] bracket s={bracket_state['s']} sudah selesai "
                f"(survivors_final={bracket_state['survivors_final']})"
            )
            continue
        _successive_halving_for_bracket(
            bracket_state=bracket_state,
            out_root=out_root,
            state=state,
            runs_root=runs_root,
            py=py,
            train_py=train_py,
            cwd_train=cwd_train,
            seed=search_train_seed,
            profile=search_profile,
            train_eps=train_eps,
            val_eps=val_eps,
            dataset_dir=dataset_dir,
            episode_split_path=episode_split_path,
            checkpoint_every=checkpoint_every,
            dataloader_num_workers=dataloader_num_workers,
            eta=eta,
        )

    best = _pick_best_from_state(state)
    state["best"] = best
    _save_state(out_root, state)

    if best is None:
        print(
            "[hyperband] WARNING: tidak ada evaluasi sukses; tidak ada pemenang."
        )
    else:
        print("\n" + "#" * 72)
        print(
            f"[hyperband] PEMENANG: cfg_idx={best['cfg_idx']} "
            f"(bracket s={best['bracket_s']}, r_i={best['r_i']}, "
            f"val_loss={best['val_loss']:.6f})"
        )
        print(json.dumps(best["hparams"], indent=2, default=str))
        print("#" * 72 + "\n")
    return best
