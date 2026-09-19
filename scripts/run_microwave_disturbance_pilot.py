#!/usr/bin/env python3
"""Microwave disturbance-recovery pilot — FlowPolicy Jalur A across NFE.

Protocol C seeding. Qpos kick on microjoint at replan boundary after attempt
trigger. GT q_mw logged every env-step. Metrics: R, T_wall, max_dev, SR.

Usage:
  MUJOCO_GL=egl /home/daffa/miniforge3/envs/flowpolicy-kitchen/bin/python \\
      scripts/run_microwave_disturbance_pilot.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FP_ROOT = ROOT / "kripsy12/FlowPolicy"
sys.path.insert(0, str(FP_ROOT))

from eval_kitchen import build_runner, load_policy, resolve_checkpoint  # noqa: E402
from flow_policy_3d.common.pytorch_util import dict_apply  # noqa: E402
from flow_policy_3d.gym_util.multistep_wrapper import aggregate, dict_take_last_n  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT = ROOT / "data/kitchen_eval_plots/nfe100/disturbance_recovery"
FP_EVAL = ROOT / "kripsy12/FlowPolicy/data/kitchen_eval_nfe100/flowpolicy"
SUMMARY_CSV = ROOT / "data/kitchen_eval_plots/nfe100/summary.csv"
DATASET_DIR = "data/kitchen"
CKPT_BY_SEED = {
    42: "data/outputs/baseline_42/latest-001.ckpt",
    43: "data/outputs/baseline_43/latest-001.ckpt",
    44: "data/outputs/baseline_44/latest-001(1).ckpt",
}

SEEDS = (42, 43, 44)
NFES = (1, 8, 32, 100)
SAMPLING_SEED = 0
N_PILOT_PAIRS = 20
TA = 4
DT_ENV_MS = 80.0
T_CHUNK_MS = TA * DT_ENV_MS  # 320

MW_QPOS = 22
MW_OBJ = 13
MW_JOINT_LO, MW_JOINT_HI = -2.094, 0.0
MW_JOINT_SPAN = MW_JOINT_HI - MW_JOINT_LO  # 2.094
DELTA_RAD = 0.10 * MW_JOINT_SPAN  # +0.2094 toward closed
ATTEMPT_THR = 0.05
EPS = 0.05
W_RECOVER = 3
BONUS_THRESH = 0.3
MW_GOAL = -0.75
MAX_STEPS = 280


def latency_ms_from_summary() -> dict[int, float]:
    import csv as _csv

    out: dict[int, float] = {}
    with open(SUMMARY_CSV) as f:
        for row in _csv.DictReader(f):
            if row["model"] != "flowpolicy":
                continue
            out[int(row["nfe"])] = float(row["latency_ms"])
    return out


def seed_protocol_c(seed: int = SAMPLING_SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def unwrap_kitchen(env):
    return env.env.env.env


def step_chunk_per_env_step(env, action_chunk, kitchen):
    sim_mw_steps: list[float] = []
    for act in action_chunk:
        if len(env.done) > 0 and env.done[-1]:
            break
        observation, reward, done, info = env.env.step(act)
        env.obs.append(observation)
        env.reward.append(reward)
        if (env.max_episode_steps is not None) and (
            len(env.reward) >= env.max_episode_steps
        ):
            done = True
        env.done.append(done)
        env._add_info(info)
        sim_mw_steps.append(float(kitchen.sim.data.qpos[MW_QPOS]))
    observation = env._get_obs(env.n_obs_steps)
    reward = aggregate(env.reward, env.reward_agg_method)
    done = bool(aggregate(env.done, "max"))
    info = dict_take_last_n(env.info, env.n_obs_steps)
    return observation, reward, done, info, sim_mw_steps


def apply_qpos_kick(kitchen, delta: float) -> float:
    qpos = kitchen.sim.data.qpos.ravel().copy()
    qvel = kitchen.sim.data.qvel.ravel().copy()
    before = float(qpos[MW_QPOS])
    after = float(np.clip(before + delta, MW_JOINT_LO, MW_JOINT_HI))
    qpos[MW_QPOS] = after
    kitchen.set_state(qpos, qvel)
    return after - before


def microwave_success(kitchen) -> bool:
    q = float(kitchen.sim.data.qpos[MW_QPOS])
    return abs(q - MW_GOAL) < BONUS_THRESH


def build_intersection_pairs(n_pairs: int = N_PILOT_PAIRS) -> list[dict]:
    att: dict[tuple[int, int], dict[int, bool]] = defaultdict(dict)
    eidx_map: dict[tuple[int, int], int] = {}
    for nfe in NFES:
        for seed in SEEDS:
            run = FP_EVAL / f"seed_baseline_{seed}_nfe{nfe}_sseed0"
            metrics = json.loads((run / "eval_metrics.json").read_text())
            for ep in metrics["episodes"]:
                eidx = int(ep["episode_idx"])
                iidx = int(ep["init_idx"])
                z = np.load(run / "trajectory_logs" / f"ep_{eidx:04d}.npz")
                mw = np.asarray(z["policy_obs_obj_qp"], dtype=np.float64)[:, -1, MW_OBJ]
                succ = bool(ep["task_success"]["microwave"])
                disp = float(np.max(np.abs(mw - mw[0]))) if len(mw) else 0.0
                att[(seed, iidx)][nfe] = bool(succ or disp > ATTEMPT_THR)
                eidx_map[(seed, iidx)] = eidx
    inter = []
    for (seed, iidx), d in att.items():
        if set(d.keys()) == set(NFES) and all(d[n] for n in NFES):
            inter.append(
                {
                    "seed": seed,
                    "init_idx": iidx,
                    "episode_idx": eidx_map[(seed, iidx)],
                }
            )
    inter.sort(key=lambda x: (x["seed"], x["init_idx"]))
    return inter[:n_pairs]


def run_rollout(
    policy,
    runner,
    episode_idx: int,
    *,
    disturb: bool,
    delta_rad: float,
    nominal_q_mw: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """One control or disturbed rollout. Returns GT trajectory + metrics hooks."""
    env = runner._create_env()
    env, _ = runner._configure_env(env, episode_idx, enable_render=False)
    env.seed(episode_idx)  # Protocol C

    kitchen = unwrap_kitchen(env)
    obs = env.reset()
    policy.reset()

    q_mw = [float(kitchen.sim.data.qpos[MW_QPOS])]
    q0 = q_mw[0]
    injected = False
    t_inj: Optional[int] = None
    actual_delta = 0.0
    pending_inject = False  # attempt seen; inject at next replan boundary

    done = False
    env_step = 0
    past_action = None
    n_ctrl = 0

    while not done and env_step < runner.max_steps:
        np_obs_dict = {"obs": obs.astype(np.float32)[None, ...]}
        if runner.past_action and past_action is not None:
            np_obs_dict["past_action"] = past_action[
                :, -(runner.n_obs_steps - 1) :
            ].astype(np.float32)
        obs_dict = dict_apply(
            np_obs_dict, lambda x: torch.from_numpy(x).to(device=policy.device)
        )
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)
        action = action_dict["action"][0].detach().cpu().numpy()
        n_ctrl += 1

        # Inject ONLY at replan boundary (after predict, before chunk exec)
        if disturb and pending_inject and not injected:
            actual_delta = apply_qpos_kick(kitchen, delta_rad)
            injected = True
            t_inj = env_step  # next logged steps are post-disturbance
            # overwrite last logged state? t_inj marks env-step index before new steps
            # q_mw currently has length env_step+1 (includes t=0). Inject changes state
            # at boundary; first post-inject GT comes from subsequent steps.

        obs, reward, done, info, sim_steps = step_chunk_per_env_step(
            env, action, kitchen
        )
        past_action = action_dict["action"].detach().cpu().numpy()
        q_mw.extend(sim_steps)
        env_step += len(sim_steps)

        # Attempt trigger: after chunk, if displacement exceeded thr, arm inject
        # for NEXT replan cycle (so current chunk already reflected interaction).
        if disturb and not injected and not pending_inject:
            cur = float(kitchen.sim.data.qpos[MW_QPOS])
            if abs(cur - q0) > ATTEMPT_THR:
                pending_inject = True

    q_arr = np.asarray(q_mw, dtype=np.float64)
    succ = microwave_success(kitchen)
    # Also check via goal distance on final q
    if not succ and len(q_arr):
        succ = abs(float(q_arr[-1]) - MW_GOAL) < BONUS_THRESH

    env.close()
    return {
        "q_mw": q_arr,
        "n_env_steps": env_step,
        "n_ctrl": n_ctrl,
        "success_microwave": bool(succ),
        "injected": injected,
        "t_inj": t_inj,
        "actual_delta": actual_delta,
        "pending_inject_never_fired": disturb and not injected,
    }


def compute_recovery(
    q_ctrl: np.ndarray, q_dist: np.ndarray, t_inj: int, eps: float, w: int
) -> dict[str, Any]:
    """Compare disturbed vs control GT from t_inj onward."""
    t_max = min(len(q_ctrl), len(q_dist))
    if t_inj is None or t_inj >= t_max - 1:
        return {
            "recovered": False,
            "R": None,
            "C": None,
            "T_wall_ms": None,
            "max_dev": None,
        }

    # Align: after inject, disturbed state diverges. Compare from t_inj+1
    # (first env-step executed after kick). Use abs diff vs control.
    start = t_inj  # index into arrays: q[0]=reset; after t_inj env-steps done pre-inject
    # At inject boundary, disturbed q at index `t_inj` still equals pre-kick if we
    # logged before kick. Our loop: kick then execute chunk → first new samples are
    # post-kick. Array length before chunk was t_inj+1 (indices 0..t_inj).
    # Post-kick samples append at indices t_inj+1, ...
    # So compare from index t_inj+1 if available, else t_inj.
    i0 = min(t_inj + 1, t_max - 1)
    dev = np.abs(q_dist[i0:t_max] - q_ctrl[i0:t_max])
    max_dev = float(dev.max()) if len(dev) else 0.0

    recovered = False
    t_recover = None
    run = 0
    for k, d in enumerate(dev):
        if d <= eps:
            run += 1
            if run >= w:
                # recovery at env-step index i0+k (0-based in full traj)
                t_recover = i0 + k
                recovered = True
                break
        else:
            run = 0

    if not recovered:
        return {
            "recovered": False,
            "R": None,
            "C": None,
            "T_wall_ms": None,
            "max_dev": max_dev,
        }

    R = int(t_recover - t_inj)
    C = int(math.ceil(R / TA)) if R > 0 else 0
    return {
        "recovered": True,
        "R": R,
        "C": C,
        "T_wall_ms": None,  # filled by caller with L
        "max_dev": max_dev,
        "t_recover": t_recover,
    }


def _save_fig(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path_base.with_suffix('.png')}")


def make_plots(
    rows: list[dict], lat: dict[int, float], nfe_list: tuple[int, ...] = NFES
) -> None:
    # Aggregate per NFE
    by_nfe: dict[int, list[dict]] = {n: [] for n in nfe_list}
    for r in rows:
        by_nfe[r["nfe"]].append(r)

    xs = [lat[n] for n in nfe_list]
    # Plot 1a: R vs latency
    mean_R, sem_R = [], []
    mean_Tw, sem_Tw = [], []
    for n in nfe_list:
        Rs = [r["R"] for r in by_nfe[n] if r["recovered"] and r["R"] is not None]
        Tws = [
            r["T_wall_ms"]
            for r in by_nfe[n]
            if r["recovered"] and r["T_wall_ms"] is not None
        ]
        mean_R.append(float(np.mean(Rs)) if Rs else float("nan"))
        sem_R.append(
            float(np.std(Rs, ddof=1) / np.sqrt(len(Rs))) if len(Rs) > 1 else 0.0
        )
        mean_Tw.append(float(np.mean(Tws)) if Tws else float("nan"))
        sem_Tw.append(
            float(np.std(Tws, ddof=1) / np.sqrt(len(Tws))) if len(Tws) > 1 else 0.0
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, ys, yerr, ylabel in [
        (axes[0], mean_R, sem_R, "Mean steps-to-recover R (env-steps)"),
        (axes[1], mean_Tw, sem_Tw, "Mean wall-clock recovery T_wall (ms)"),
    ]:
        ax.errorbar(
            xs, ys, yerr=yerr, fmt="-o", color="#1f77b4", capsize=3, linewidth=1.5
        )
        ax.set_xscale("log")
        ax.set_xlabel("Inference latency (ms)")
        ax.set_ylabel(ylabel)
        ax.set_facecolor("white")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        for i, (x, n) in enumerate(zip(xs, nfe_list)):
            if np.isfinite(ys[i]):
                ax.annotate(
                    f"NFE={n}",
                    (x, ys[i]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Jalur A (terkontrol, satu arsitektur): FlowPolicy recovery vs latency\n"
        f"microwave qpos kick Δ={DELTA_RAD:.4f} rad, ε={EPS}, W={W_RECOVER}, "
        f"n={len({(r['seed'], r['init_idx']) for r in rows})} pairs",
        fontsize=11,
    )
    fig.tight_layout()
    _save_fig(fig, OUT / "01_jalurA_recovery_vs_latency")

    # Plot 2: SR control vs disturbed
    sr_c, sr_d = [], []
    for n in nfe_list:
        eps = by_nfe[n]
        sr_c.append(float(np.mean([r["success_control"] for r in eps])))
        sr_d.append(float(np.mean([r["success_disturbed"] for r in eps])))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(nfe_list))
    w = 0.35
    ax.bar(x - w / 2, sr_c, w, label="Kontrol (tanpa gangguan)", color="#2ca02c")
    ax.bar(x + w / 2, sr_d, w, label="Pasca-gangguan", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([f"NFE={n}\n({lat[n]:.0f} ms)" for n in nfe_list])
    ax.set_ylabel("Microwave success rate")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_title(
        "Jalur A: microwave SR kontrol vs pasca-gangguan (FlowPolicy, Ta=4)"
    )
    fig.tight_layout()
    _save_fig(fig, OUT / "02_jalurA_sr_control_vs_disturbed")


def print_markdown_table(
    rows: list[dict], lat: dict[int, float], nfe_list: tuple[int, ...] = NFES
) -> str:
    by_nfe: dict[int, list[dict]] = {n: [] for n in nfe_list}
    for r in rows:
        by_nfe[r["nfe"]].append(r)

    n_pairs = len({(r["seed"], r["init_idx"]) for r in rows})
    lines = [
        "## Pilot ringkasan — FlowPolicy Jalur A (microwave disturbance)",
        "",
        f"- delta_rad = {DELTA_RAD:.4f} (10% joint span {MW_JOINT_SPAN})",
        f"- ε = {EPS}, W = {W_RECOVER}, attempt_thr = {ATTEMPT_THR}",
        f"- n_pairs = {n_pairs}, Protocol C, inject-at-replan",
        "",
        "| NFE | L (ms) | n | n_inj | n_rec | mean R | SEM R | mean T_wall (ms) | SEM T_wall | "
        "mean max_dev | SR_ctrl | SR_dist |",
        "|----:|-------:|--:|------:|------:|-------:|------:|-----------------:|-----------:|"
        "-------------:|--------:|--------:|",
    ]
    for n in nfe_list:
        eps = by_nfe[n]
        n_all = len(eps)
        n_inj = sum(1 for r in eps if r["injected"])
        rec = [r for r in eps if r["recovered"]]
        Rs = [r["R"] for r in rec if r["R"] is not None]
        Tws = [r["T_wall_ms"] for r in rec if r["T_wall_ms"] is not None]
        devs = [r["max_dev"] for r in eps if r["max_dev"] is not None]
        mean_R = float(np.mean(Rs)) if Rs else float("nan")
        sem_R = float(np.std(Rs, ddof=1) / np.sqrt(len(Rs))) if len(Rs) > 1 else 0.0
        mean_Tw = float(np.mean(Tws)) if Tws else float("nan")
        sem_Tw = (
            float(np.std(Tws, ddof=1) / np.sqrt(len(Tws))) if len(Tws) > 1 else 0.0
        )
        mean_dev = float(np.mean(devs)) if devs else float("nan")
        sr_c = float(np.mean([r["success_control"] for r in eps]))
        sr_d = float(np.mean([r["success_disturbed"] for r in eps]))
        lines.append(
            f"| {n} | {lat[n]:.1f} | {n_all} | {n_inj} | {len(rec)} | "
            f"{mean_R:.2f} | {sem_R:.2f} | {mean_Tw:.1f} | {sem_Tw:.1f} | "
            f"{mean_dev:.4f} | {sr_c:.3f} | {sr_d:.3f} |"
        )
    # Calibration notes
    lines += [
        "",
        "### Catatan kalibrasi",
    ]
    never = sum(1 for r in rows if r.get("pending_never"))
    trivial = sum(
        1
        for r in rows
        if r["recovered"] and r["R"] is not None and r["R"] <= TA
    )
    fatal = sum(1 for r in rows if r["injected"] and not r["success_disturbed"])
    lines.append(f"- inject gagal (attempt tak terpicu): {never}/{len(rows)}")
    lines.append(f"- recover dalam ≤1 chunk (R≤{TA}): {trivial}/{len(rows)} (risiko terlalu kecil)")
    lines.append(f"- inject tapi MW gagal: {fatal}/{len(rows)} (risiko terlalu besar jika ≈n_inj)")
    text = "\n".join(lines)
    print(text)
    return text


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=N_PILOT_PAIRS)
    parser.add_argument(
        "--nfes",
        type=str,
        default=",".join(str(n) for n in NFES),
        help="Comma-separated NFE list, e.g. 1,8,32,100",
    )
    args = parser.parse_args()
    n_pairs = int(args.n_pairs)
    nfe_list = tuple(int(x) for x in args.nfes.split(",") if x.strip())

    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(FP_ROOT)
    lat = latency_ms_from_summary()
    pairs = build_intersection_pairs(n_pairs)
    (OUT / "pilot_pairs.json").write_text(json.dumps(pairs, indent=2))
    print(f"Pilot pairs ({len(pairs)}):", flush=True)
    for p in pairs:
        print(
            f"  seed={p['seed']} init_idx={p['init_idx']} episode_idx={p['episode_idx']}",
            flush=True,
        )
    print(f"delta_rad={DELTA_RAD:.4f}  eps={EPS}  W={W_RECOVER}", flush=True)
    print(f"NFEs={nfe_list}  Latencies={lat}", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    csv_path = OUT / "pilot_jalurA_metrics.csv"

    by_seed: dict[int, list[dict]] = defaultdict(list)
    for p in pairs:
        by_seed[p["seed"]].append(p)

    for nfe in nfe_list:
        for seed, seed_pairs in by_seed.items():
            print(
                f"\n=== NFE={nfe} seed={seed} ({len(seed_pairs)} pairs) ===",
                flush=True,
            )
            seed_protocol_c(SAMPLING_SEED)
            ckpt = resolve_checkpoint(CKPT_BY_SEED[seed])
            policy, cfg = load_policy(ckpt, device)
            policy.num_inference_step = nfe
            runner = build_runner(
                cfg=cfg,
                output_dir=str(OUT / f"runs/seed{seed}_nfe{nfe}"),
                dataset_dir=DATASET_DIR,
                n_episodes=1,
                save_trajectory_logs=False,
                n_episodes_vis=0,
            )
            assert runner.init_qpos is not None

            for p in seed_pairs:
                eidx = p["episode_idx"]
                print(
                    f"  pair seed={seed} init={p['init_idx']} ep={eidx} ...",
                    flush=True,
                )

                seed_protocol_c(SAMPLING_SEED)
                ctrl = run_rollout(
                    policy, runner, eidx, disturb=False, delta_rad=DELTA_RAD
                )

                seed_protocol_c(SAMPLING_SEED)
                dist = run_rollout(
                    policy, runner, eidx, disturb=True, delta_rad=DELTA_RAD
                )

                rec = compute_recovery(
                    ctrl["q_mw"], dist["q_mw"], dist["t_inj"], EPS, W_RECOVER
                )
                L = lat[nfe]
                T_wall = None
                if rec["recovered"] and rec["C"] is not None:
                    T_wall = rec["C"] * (T_CHUNK_MS + L)
                    rec["T_wall_ms"] = T_wall

                row = {
                    "seed": seed,
                    "init_idx": p["init_idx"],
                    "episode_idx": eidx,
                    "nfe": nfe,
                    "latency_ms": L,
                    "injected": dist["injected"],
                    "t_inj": dist["t_inj"],
                    "actual_delta": dist["actual_delta"],
                    "pending_never": dist["pending_inject_never_fired"],
                    "recovered": rec["recovered"],
                    "R": rec["R"],
                    "C": rec["C"],
                    "T_wall_ms": T_wall,
                    "max_dev": rec["max_dev"],
                    "success_control": ctrl["success_microwave"],
                    "success_disturbed": dist["success_microwave"],
                    "n_env_ctrl": ctrl["n_env_steps"],
                    "n_env_dist": dist["n_env_steps"],
                }
                rows.append(row)
                _write_csv(csv_path, rows)
                print(
                    f"    inj={row['injected']} t_inj={row['t_inj']} "
                    f"R={row['R']} T_wall={row['T_wall_ms']} "
                    f"max_dev={row['max_dev']} "
                    f"SR {row['success_control']}→{row['success_disturbed']}",
                    flush=True,
                )

            del policy, runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nWrote {csv_path}", flush=True)
    make_plots(rows, lat, nfe_list=nfe_list)
    md = print_markdown_table(rows, lat, nfe_list=nfe_list)
    (OUT / "pilot_jalurA_summary.md").write_text(md + "\n")
    print(f"Wrote {OUT / 'pilot_jalurA_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
