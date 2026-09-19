#!/usr/bin/env python3
"""Verify microwave q_mw determinism across repeated control rollouts.

Protocol C (default sweep): env.seed(episode_idx) + torch/np sampling_seed;
no cudnn/deterministic_torch.

Ground-truth q_mw is logged at EVERY env-step (not end-of-chunk replication)
by stepping each action in the chunk individually through the inner env while
keeping MultiStep obs buffers in sync.

Usage:
  MUJOCO_GL=egl /home/daffa/miniforge3/envs/flowpolicy-kitchen/bin/python \\
      scripts/verify_microwave_nominal_determinism.py

  # full Protocol-C sweep across NFE (main gate):
  MUJOCO_GL=egl ... scripts/verify_microwave_nominal_determinism.py --all-nfe
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FP_ROOT = ROOT / "kripsy12/FlowPolicy"
sys.path.insert(0, str(FP_ROOT))

from eval_kitchen import build_runner, load_policy, resolve_checkpoint  # noqa: E402
from flow_policy_3d.common.pytorch_util import dict_apply  # noqa: E402
from flow_policy_3d.gym_util.multistep_wrapper import aggregate, dict_take_last_n  # noqa: E402

CKPT = "data/outputs/baseline_42/latest-001.ckpt"
DATASET_DIR = "data/kitchen"
SAMPLING_SEED = 0
EPISODE_IDX = 0
N_REPEATS = 5
NFES = (1, 8, 32, 100)
MW_QPOS = 22
MW_OBJ = 13
OUT = ROOT / "data/kitchen_eval_plots/nfe100/disturbance_recovery"
TA = 4
IDENTICAL_THR = 1e-9


def seed_all(seed: int, deterministic_torch: bool) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def unwrap_kitchen(env):
    return env.env.env.env


def step_chunk_per_env_step(env, action_chunk, kitchen):
    """Execute action chunk one env-step at a time; return GT q_mw per step.

    Mirrors MultiStepWrapper.step so obs/reward/done/info buffers stay valid,
    but records sim.data.qpos[MW_QPOS] after every single action.
    """
    sim_mw_steps: list[float] = []
    obs_mw_steps: list[float] = []
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
        obs_mw_steps.append(float(np.asarray(info["obs_dict"]["obj_qp"])[MW_OBJ]))

    observation = env._get_obs(env.n_obs_steps)
    reward = aggregate(env.reward, env.reward_agg_method)
    done = aggregate(env.done, "max")
    info = dict_take_last_n(env.info, env.n_obs_steps)
    return observation, reward, bool(done), info, sim_mw_steps, obs_mw_steps


def run_one(policy, runner, episode_idx: int, hard_seed_env: bool) -> dict:
    env = runner._create_env()
    env, _ = runner._configure_env(env, episode_idx, enable_render=False)
    if hard_seed_env:
        env.seed(episode_idx)

    kitchen = unwrap_kitchen(env)
    obs = env.reset()
    policy.reset()

    sim_mw: list[float] = [float(kitchen.sim.data.qpos[MW_QPOS])]
    obs_mw: list[float] = [float(kitchen.obs_dict["obj_qp"][MW_OBJ])]
    actions: list[np.ndarray] = []

    done = False
    env_step = 0
    past_action = None
    while not done:
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
        actions.append(action.copy())

        obs, reward, done, info, sim_steps, obs_steps = step_chunk_per_env_step(
            env, action, kitchen
        )
        past_action = action_dict["action"].detach().cpu().numpy()
        sim_mw.extend(sim_steps)
        obs_mw.extend(obs_steps)
        env_step += len(sim_steps)

        if env_step >= runner.max_steps:
            break

    env.close()
    return {
        "sim_mw": np.asarray(sim_mw, dtype=np.float64),
        "obs_mw": np.asarray(obs_mw, dtype=np.float64),
        "n_env_steps": env_step,
        "n_ctrl": len(actions),
        "n_gt_samples": len(sim_mw),
        "first_action": actions[0].copy() if actions else None,
    }


def compare_repeats(trajs: list[dict], key: str) -> dict:
    series = [t[key] for t in trajs]
    min_len = min(len(s) for s in series)
    stacked = np.stack([s[:min_len] for s in series], axis=0)
    diffs = np.abs(stacked - stacked[0:1])
    std_t = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros(min_len)
    return {
        "min_len": min_len,
        "max_abs_diff_vs_first": [float(x) for x in diffs.max(axis=1)],
        "global_max_abs_diff": float(diffs.max()),
        "mean_std_over_t": float(std_t.mean()),
        "max_std_over_t": float(std_t.max()),
        "p95_std_over_t": float(np.percentile(std_t, 95)),
        "identical": bool(diffs.max() < IDENTICAL_THR),
        "std_t": std_t,
        "stacked": stacked,
    }


def run_protocol_c_for_nfe(nfe: int, n_repeats: int = N_REPEATS) -> dict:
    """Protocol C: env.seed + torch/np seed; no deterministic_torch."""
    print(f"\n===== Protocol C | NFE={nfe} | repeats={n_repeats} =====")
    print("GT logging: per env-step (not end-of-chunk replication)")
    os.chdir(FP_ROOT)
    ckpt = resolve_checkpoint(CKPT)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trajs = []
    for r in range(n_repeats):
        seed_all(SAMPLING_SEED, deterministic_torch=False)
        policy, cfg = load_policy(ckpt, device)
        policy.num_inference_step = nfe
        out_dir = OUT / f"determinism_C_nfe{nfe}" / f"repeat_{r}"
        out_dir.mkdir(parents=True, exist_ok=True)
        runner = build_runner(
            cfg=cfg,
            output_dir=str(out_dir),
            dataset_dir=DATASET_DIR,
            n_episodes=1,
            save_trajectory_logs=False,
            n_episodes_vis=0,
        )
        assert runner.init_qpos is not None
        t = run_one(policy, runner, EPISODE_IDX, hard_seed_env=True)
        trajs.append(t)
        print(
            f"  repeat {r}: env_steps={t['n_env_steps']} ctrl={t['n_ctrl']} "
            f"gt_samples={t['n_gt_samples']} "
            f"sim_mw0={t['sim_mw'][0]:.6f} sim_mw_end={t['sim_mw'][-1]:.6f} "
            f"obs_mw_end={t['obs_mw'][-1]:.6f}"
        )
        del policy, runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sim_stats = compare_repeats(trajs, "sim_mw")
    obs_stats = compare_repeats(trajs, "obs_mw")
    a0 = [t["first_action"] for t in trajs if t["first_action"] is not None]
    a_diff = (
        float(np.abs(np.stack(a0) - np.stack(a0)[0:1]).max()) if a0 else float("nan")
    )
    # Confirm GT length == 1 + n_env_steps (t=0 plus every step)
    gt_ok = all(t["n_gt_samples"] == t["n_env_steps"] + 1 for t in trajs)

    print(
        f"  SIM identical? {sim_stats['identical']}  "
        f"max_abs={sim_stats['global_max_abs_diff']:.6g}  "
        f"mean_std={sim_stats['mean_std_over_t']:.6g}  "
        f"max_std={sim_stats['max_std_over_t']:.6g}"
    )
    print(
        f"  OBS identical? {obs_stats['identical']}  "
        f"max_abs={obs_stats['global_max_abs_diff']:.6g}  "
        f"mean_std={obs_stats['mean_std_over_t']:.6g}"
    )
    print(f"  first_action_max_abs_diff={a_diff:.6g}  per_env_step_gt_ok={gt_ok}")
    return {
        "nfe": nfe,
        "sim": sim_stats,
        "obs": obs_stats,
        "first_action_diff": a_diff,
        "per_env_step_gt_ok": gt_ok,
        "trajs": trajs,
    }


def write_all_nfe_report(results: dict[int, dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / "determinism_check_all_nfe.txt"
    lines = [
        "Microwave nominal determinism — Protocol C sweep across NFE",
        f"model=FlowPolicy baseline_42  sampling_seed={SAMPLING_SEED}  "
        f"episode_idx={EPISODE_IDX}  repeats={N_REPEATS}",
        "Protocol C: env.seed(episode_idx) + torch/np seed; deterministic_torch=False",
        "GT logging: sim.data.qpos[22] after EVERY env-step (resolusi 1)",
        "",
        f"{'NFE':>5}  {'SIM_ident':>10}  {'max_abs':>12}  {'mean_std':>12}  "
        f"{'max_std':>12}  {'act_diff':>12}  {'gt_ok':>6}",
        "-" * 80,
    ]
    all_ok = True
    failed = []
    nfe_list = sorted(results.keys())
    for nfe in nfe_list:
        res = results[nfe]
        sim = res["sim"]
        ident = sim["identical"]
        if not ident:
            all_ok = False
            failed.append(nfe)
        lines.append(
            f"{nfe:5d}  {str(ident):>10}  {sim['global_max_abs_diff']:12.6g}  "
            f"{sim['mean_std_over_t']:12.6g}  {sim['max_std_over_t']:12.6g}  "
            f"{res['first_action_diff']:12.6g}  {str(res['per_env_step_gt_ok']):>6}"
        )
        np.savez_compressed(
            OUT / f"determinism_sim_mw_protocol_C_nfe{nfe}.npz",
            stacked=sim["stacked"],
            std_t=sim["std_t"],
        )

    lines.append("")
    lines.append("VERDICT")
    if all_ok:
        lines.append(
            "  ALL NFE bit-identical under Protocol C. "
            "Use Protocol C for the full disturbance experiment."
        )
        decision = "Protocol C"
    else:
        lines.append(
            f"  FAILED NFE under Protocol C: {failed}. "
            "DEFAULT ALL experiment conditions to Protocol B "
            "(deterministic_torch=True) for cross-NFE consistency."
        )
        decision = "Protocol B"
    lines.append(f"  DECISION: {decision}")
    lines.append("")
    lines.append("NOTE on GT logging:")
    lines.append(
        "  Previous report note 'end-of-chunk GT replicated' applied ONLY to the "
        "old verification script. This sweep and the future disturbance experiment "
        "log q_mw at every env-step so R has full 1-step resolution."
    )
    report.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {report}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-nfe",
        action="store_true",
        help="Sweep Protocol C across NFE 1/8/32/100 (main gate)",
    )
    parser.add_argument("--nfe", type=int, default=1, help="Single-NFE mode")
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.all_nfe:
        results = {}
        for nfe in NFES:
            results[nfe] = run_protocol_c_for_nfe(nfe, n_repeats=args.repeats)
        write_all_nfe_report(results)
    else:
        res = run_protocol_c_for_nfe(args.nfe, n_repeats=args.repeats)
        write_all_nfe_report({args.nfe: res})


if __name__ == "__main__":
    main()
