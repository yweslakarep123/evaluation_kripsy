import gc
import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import tqdm

from diffusion_policy.common.multistage_metrics import compute_multistage_metrics
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.kitchen.base import KitchenBase
from diffusion_policy.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
from diffusion_policy.env.kitchen.v0 import KitchenAllV0
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

module_logger = logging.getLogger(__name__)

TRAJECTORY_SCHEMA = """Kitchen trajectory log schema (ep_XXXX.npz)
================================================
Scalars: episode_idx, init_idx, demo_valid_len, n_control_steps, n_env_steps,
         horizon, n_obs_steps, n_action_steps

Demo ground truth (column init_idx in observations_seq/actions_seq):
  demo_obs          (T_demo, 60)   qp(9) + obj_qp(21) + goal(30)
  demo_action       (T_demo, 9)

Per control step (one predict_action call):
  policy_obs        (T_ctrl, n_obs_steps, 60)
  action_executed   (T_ctrl, n_action_steps, 9)
  action_pred       (T_ctrl, horizon, 9)
  obs_pred          (T_ctrl, horizon, 60)   optional (DP inpainting)
  action_obs_pred   (T_ctrl, n_action_steps, 60)   optional

Per env step (each sub-step in MultiStepWrapper):
  executed_obs      (T_env, 60)
  executed_action   (T_env, 9)
  qp                (T_env, 9)    robot joint positions
  qv                (T_env, 9)    robot joint velocities
  obj_qp            (T_env, 21)   object positions
  obj_qv            (T_env, 21)   object velocities
  demo_obs_at_step  (T_env, 60)   NaN if env step exceeds demo length
  demo_action_at_step (T_env, 9)  NaN if env step exceeds demo length
  action_error_l2   (T_env,)      L2 vs demo action
  obs_error_l2      (T_env,)      L2 vs demo obs

Note: demo GT has no velocity; qv/obj_qv are rollout-only from env obs_dict.
"""


def _close_env(env: MultiStepWrapper):
    if isinstance(env.env, VideoRecordingWrapper):
        env.env.video_recoder.stop()
        env.env.file_path = None
    env.close()
    gc.collect()


ALL_TASKS = list(KitchenBase.ALL_TASKS)
KITCHEN_4_SUBGOALS = ["microwave", "kettle", "bottom burner", "light switch"]
TASK_NOTE = (
    "Tasks can complete in any order; all 7 must finish for full episode success. "
    "Task duration is measured from the previous task completion (or episode start "
    "for the first completed task) until that task is marked complete."
)


def _compute_mean_std(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": None, "std": None, "n_samples": 0}
    if n == 1:
        return {"mean": float(arr[0]), "std": 0.0, "n_samples": 1}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "n_samples": n,
    }


def _extract_completed_tasks(info_item: dict) -> Set[str]:
    completed = info_item.get("completed_tasks", set())
    if isinstance(completed, (list, tuple, np.ndarray)):
        if len(completed) == 0:
            return set()
        last = completed[-1]
        if isinstance(last, set):
            return set(last)
        return set(last)
    if isinstance(completed, set):
        return completed
    return set(completed)


def _fmt_array(arr: np.ndarray, precision: int = 6) -> str:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    return " ".join(f"{x:.{precision}f}" for x in flat)


def _obs_dict_to_arrays(obs_dict: dict) -> Dict[str, np.ndarray]:
    return {
        "qp": np.asarray(obs_dict["qp"], dtype=np.float32),
        "qv": np.asarray(obs_dict["qv"], dtype=np.float32),
        "obj_qp": np.asarray(obs_dict["obj_qp"], dtype=np.float32),
        "obj_qv": np.asarray(obs_dict["obj_qv"], dtype=np.float32),
    }


class KitchenLowdimEvalRunner(BaseLowdimRunner):
    def __init__(
        self,
        output_dir,
        n_episodes: int = 100,
        n_episodes_vis: int = 100,
        max_steps: int = 280,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        render_hw=(240, 360),
        fps: float = 12.5,
        crf: int = 22,
        past_action: bool = False,
        abs_action: bool = False,
        tqdm_interval_sec: float = 5.0,
        dataset_dir: Optional[str] = None,
        save_trajectory_logs: bool = True,
    ):
        super().__init__(output_dir)
        self.n_episodes = n_episodes
        self.n_episodes_vis = n_episodes_vis
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.abs_action = abs_action
        self.fps = fps
        self.crf = crf
        self.render_hw = render_hw
        self.save_trajectory_logs = save_trajectory_logs

        self.init_qpos = None
        self.init_qvel = None
        self.demo_observations_seq = None
        self.demo_actions_seq = None
        self.demo_existence_mask = None

        if dataset_dir is not None:
            dataset_path = pathlib.Path(dataset_dir)
            self.init_qpos = np.load(dataset_path / "all_init_qpos.npy")
            self.init_qvel = np.load(dataset_path / "all_init_qvel.npy")
            if save_trajectory_logs:
                self.demo_observations_seq = np.load(
                    dataset_path / "observations_seq.npy"
                )
                self.demo_actions_seq = np.load(dataset_path / "actions_seq.npy")
                self.demo_existence_mask = np.load(
                    dataset_path / "existence_mask.npy"
                )

        pathlib.Path(output_dir).joinpath("media").mkdir(parents=True, exist_ok=True)
        if save_trajectory_logs:
            traj_dir = pathlib.Path(output_dir).joinpath("trajectory_logs")
            traj_dir.mkdir(parents=True, exist_ok=True)
            schema_path = traj_dir / "SCHEMA.txt"
            if not schema_path.is_file():
                schema_path.write_text(TRAJECTORY_SCHEMA)

        task_fps = 12.5
        steps_per_render = int(max(task_fps // fps, 1))

        def env_fn():
            env = KitchenAllV0(use_abs_action=abs_action)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    KitchenLowdimWrapper(
                        env=env,
                        init_qpos=None,
                        init_qvel=None,
                        render_hw=tuple(render_hw),
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=int(round(fps)),
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        self._env_fn = env_fn

    def _get_demo_trajectory(
        self, init_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        if self.demo_observations_seq is None:
            return (
                np.zeros((0, 60), dtype=np.float32),
                np.zeros((0, 9), dtype=np.float32),
                0,
            )
        mask = self.demo_existence_mask[:, init_idx] > 0
        valid_len = int(mask.sum())
        demo_obs = self.demo_observations_seq[mask, init_idx, :].astype(np.float32)
        demo_action = self.demo_actions_seq[mask, init_idx, :].astype(np.float32)
        return demo_obs, demo_action, valid_len

    def _make_env(self, episode_idx: int, enable_render: bool):
        env = self._env_fn()
        assert isinstance(env, MultiStepWrapper)
        assert isinstance(env.env, VideoRecordingWrapper)

        video_path = None
        if enable_render:
            video_path = pathlib.Path(self.output_dir).joinpath(
                "media", f"ep_{episode_idx:04d}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            env.env.file_path = str(video_path)
        else:
            env.env.file_path = None

        assert isinstance(env.env.env, KitchenLowdimWrapper)
        if self.init_qpos is not None:
            init_idx = episode_idx % len(self.init_qpos)
            env.env.env.init_qpos = self.init_qpos[init_idx]
            env.env.env.init_qvel = self.init_qvel[init_idx]
        else:
            env.env.env.init_qpos = None
            env.env.env.init_qvel = None
            env.seed(episode_idx)

        return env, video_path

    def _sync_device(self, device: torch.device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _save_trajectory_log(
        self,
        episode_idx: int,
        init_idx: int,
        trajectory: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> str:
        traj_dir = pathlib.Path(self.output_dir) / "trajectory_logs"
        stem = f"ep_{episode_idx:04d}"
        npz_path = traj_dir / f"{stem}.npz"
        txt_path = traj_dir / f"{stem}_detail.txt"

        npz_data: Dict[str, Any] = {
            "episode_idx": np.int32(episode_idx),
            "init_idx": np.int32(init_idx if init_idx is not None else -1),
            "demo_valid_len": np.int32(trajectory["demo_valid_len"]),
            "n_control_steps": np.int32(trajectory["n_control_steps"]),
            "n_env_steps": np.int32(trajectory["n_env_steps"]),
            "horizon": np.int32(trajectory["horizon"]),
            "n_obs_steps": np.int32(self.n_obs_steps),
            "n_action_steps": np.int32(self.n_action_steps),
            "demo_obs": trajectory["demo_obs"],
            "demo_action": trajectory["demo_action"],
        }
        for key in [
            "policy_obs",
            "action_executed",
            "action_pred",
            "obs_pred",
            "action_obs_pred",
            "executed_obs",
            "executed_action",
            "qp",
            "qv",
            "obj_qp",
            "obj_qv",
            "demo_obs_at_step",
            "demo_action_at_step",
            "action_error_l2",
            "obs_error_l2",
        ]:
            if key in trajectory and trajectory[key] is not None:
                npz_data[key] = trajectory[key]

        np.savez_compressed(npz_path, **npz_data)
        self._write_detail_txt(txt_path, episode_idx, init_idx, trajectory, summary)
        return os.path.relpath(str(npz_path), self.output_dir)

    def _write_detail_txt(
        self,
        txt_path: pathlib.Path,
        episode_idx: int,
        init_idx: int,
        trajectory: Dict[str, Any],
        summary: Dict[str, Any],
    ):
        lines = [
            "=" * 72,
            f"Episode {episode_idx} | init_idx={init_idx} | "
            f"demo_valid_len={trajectory['demo_valid_len']}",
            f"all_7_success={summary['all_7_success']} | "
            f"completed_tasks={summary['completed_tasks']}",
            f"completion_order={summary['completion_order']}",
            "=" * 72,
            "",
            "Demo GT trajectory (first 3 steps):",
        ]
        demo_obs = trajectory["demo_obs"]
        demo_action = trajectory["demo_action"]
        for t in range(min(3, len(demo_obs))):
            lines.append(f"  demo t={t} obs={_fmt_array(demo_obs[t])}")
            lines.append(f"  demo t={t} action={_fmt_array(demo_action[t])}")
        lines.append("")

        policy_obs = trajectory.get("policy_obs")
        action_executed = trajectory.get("action_executed")
        action_pred = trajectory.get("action_pred")
        obs_pred = trajectory.get("obs_pred")
        executed_obs = trajectory.get("executed_obs")
        executed_action = trajectory.get("executed_action")
        qp = trajectory.get("qp")
        qv = trajectory.get("qv")
        demo_obs_at_step = trajectory.get("demo_obs_at_step")
        demo_action_at_step = trajectory.get("demo_action_at_step")
        action_error_l2 = trajectory.get("action_error_l2")
        obs_error_l2 = trajectory.get("obs_error_l2")

        n_ctrl = trajectory["n_control_steps"]
        env_step = 0
        for ctrl in range(n_ctrl):
            lines.append("-" * 72)
            lines.append(f"control_step={ctrl} | env_steps={env_step}..")
            if policy_obs is not None:
                lines.append(
                    f"  policy_obs[last]={_fmt_array(policy_obs[ctrl, -1])}"
                )
            if action_executed is not None:
                for a_idx in range(action_executed.shape[1]):
                    lines.append(
                        f"  action_executed[{a_idx}]={_fmt_array(action_executed[ctrl, a_idx])}"
                    )
            if action_pred is not None:
                for h in range(action_pred.shape[1]):
                    lines.append(
                        f"  action_pred[h={h}]={_fmt_array(action_pred[ctrl, h])}"
                    )
            if obs_pred is not None:
                for h in range(min(3, obs_pred.shape[1])):
                    lines.append(
                        f"  obs_pred[h={h}]={_fmt_array(obs_pred[ctrl, h])}"
                    )
                if obs_pred.shape[1] > 3:
                    lines.append(
                        f"  ... obs_pred has {obs_pred.shape[1]} horizon steps"
                    )

            n_sub = self.n_action_steps
            for sub in range(n_sub):
                if env_step >= trajectory["n_env_steps"]:
                    break
                lines.append(f"  env_step={env_step}:")
                if executed_obs is not None:
                    lines.append(
                        f"    executed_obs={_fmt_array(executed_obs[env_step])}"
                    )
                if executed_action is not None:
                    lines.append(
                        f"    executed_action={_fmt_array(executed_action[env_step])}"
                    )
                if qp is not None:
                    lines.append(f"    qp={_fmt_array(qp[env_step])}")
                if qv is not None:
                    lines.append(f"    qv={_fmt_array(qv[env_step])}")
                if demo_obs_at_step is not None:
                    lines.append(
                        f"    demo_obs_at_step={_fmt_array(demo_obs_at_step[env_step])}"
                    )
                if demo_action_at_step is not None:
                    lines.append(
                        f"    demo_action_at_step={_fmt_array(demo_action_at_step[env_step])}"
                    )
                if action_error_l2 is not None:
                    err = action_error_l2[env_step]
                    lines.append(
                        f"    action_error_l2={err:.6f}"
                        if not np.isnan(err)
                        else "    action_error_l2=NaN (no demo GT)"
                    )
                if obs_error_l2 is not None:
                    err = obs_error_l2[env_step]
                    lines.append(
                        f"    obs_error_l2={err:.6f}"
                        if not np.isnan(err)
                        else "    obs_error_l2=NaN (no demo GT)"
                    )
                env_step += 1
            lines.append("")

        txt_path.write_text("\n".join(lines))

    def _run_single_episode(
        self,
        policy: BaseLowdimPolicy,
        episode_idx: int,
        enable_render: bool,
    ) -> Dict[str, Any]:
        device = policy.device
        env, video_path = self._make_env(episode_idx, enable_render)
        init_idx = episode_idx % len(self.init_qpos) if self.init_qpos is not None else None

        demo_obs_gt = np.zeros((0, 60), dtype=np.float32)
        demo_action_gt = np.zeros((0, 9), dtype=np.float32)
        demo_valid_len = 0
        if init_idx is not None and self.save_trajectory_logs:
            demo_obs_gt, demo_action_gt, demo_valid_len = self._get_demo_trajectory(
                init_idx
            )

        traj: Optional[Dict[str, Any]] = None
        if self.save_trajectory_logs:
            traj = {
                "demo_obs": demo_obs_gt,
                "demo_action": demo_action_gt,
                "demo_valid_len": demo_valid_len,
                "policy_obs": [],
                "action_executed": [],
                "action_pred": [],
                "obs_pred": [],
                "action_obs_pred": [],
                "executed_obs": [],
                "executed_action": [],
                "qp": [],
                "qv": [],
                "obj_qp": [],
                "obj_qv": [],
                "demo_obs_at_step": [],
                "demo_action_at_step": [],
                "action_error_l2": [],
                "obs_error_l2": [],
                "horizon": None,
                "n_control_steps": 0,
                "n_env_steps": 0,
            }

        try:
            episode_start = time.perf_counter()
            obs = env.reset()
            policy.reset()

            inference_latencies_ms: List[float] = []
            completion_order: List[str] = []
            task_durations_ms: Dict[str, float] = {}
            prev_completed: Set[str] = set()
            last_completion_time = episode_start
            past_action = None
            env_step = 0

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Ep {episode_idx + 1}/{self.n_episodes}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )
            done = False
            while not done:
                np_obs_dict = {"obs": obs.astype(np.float32)[None, ...]}
                if self.past_action and (past_action is not None):
                    np_obs_dict["past_action"] = past_action[
                        :, -(self.n_obs_steps - 1) :
                    ].astype(np.float32)

                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )

                self._sync_device(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                self._sync_device(device)
                inference_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )
                action = np_action_dict["action"][0]

                if traj is not None:
                    traj["policy_obs"].append(np_obs_dict["obs"][0].copy())
                    traj["action_executed"].append(action.copy())
                    traj["action_pred"].append(np_action_dict["action_pred"][0].copy())
                    if traj["horizon"] is None:
                        traj["horizon"] = int(np_action_dict["action_pred"].shape[1])
                    if "obs_pred" in np_action_dict:
                        traj["obs_pred"].append(np_action_dict["obs_pred"][0].copy())
                    if "action_obs_pred" in np_action_dict:
                        traj["action_obs_pred"].append(
                            np_action_dict["action_obs_pred"][0].copy()
                        )

                prev_env_step = env_step
                obs, reward, done, info = env.step(action)
                done = bool(done)
                past_action = np_action_dict["action"]
                pbar.update(action.shape[0])

                if traj is not None:
                    step_infos = env.get_infos()
                    obs_dicts = step_infos.get("obs_dict", [])
                    n_new = len(obs_dicts) - prev_env_step
                    for i in range(n_new):
                        od = obs_dicts[prev_env_step + i]
                        od_arrays = _obs_dict_to_arrays(od)
                        step_obs = np.asarray(
                            env.obs[prev_env_step + i], dtype=np.float32
                        )
                        step_action = action[i] if i < len(action) else action[-1]
                        traj["executed_obs"].append(step_obs)
                        traj["executed_action"].append(
                            np.asarray(step_action, dtype=np.float32)
                        )
                        traj["qp"].append(od_arrays["qp"])
                        traj["qv"].append(od_arrays["qv"])
                        traj["obj_qp"].append(od_arrays["obj_qp"])
                        traj["obj_qv"].append(od_arrays["obj_qv"])

                        t = prev_env_step + i
                        if t < demo_valid_len:
                            d_obs = demo_obs_gt[t]
                            d_act = demo_action_gt[t]
                            traj["demo_obs_at_step"].append(d_obs.copy())
                            traj["demo_action_at_step"].append(d_act.copy())
                            traj["action_error_l2"].append(
                                float(np.linalg.norm(step_action - d_act))
                            )
                            traj["obs_error_l2"].append(
                                float(np.linalg.norm(step_obs - d_obs))
                            )
                        else:
                            nan_obs = np.full(60, np.nan, dtype=np.float32)
                            nan_act = np.full(9, np.nan, dtype=np.float32)
                            traj["demo_obs_at_step"].append(nan_obs)
                            traj["demo_action_at_step"].append(nan_act)
                            traj["action_error_l2"].append(np.nan)
                            traj["obs_error_l2"].append(np.nan)
                    env_step = len(obs_dicts)

                now = time.perf_counter()
                current_completed = _extract_completed_tasks(info)
                new_tasks = current_completed - prev_completed
                for task_name in ALL_TASKS:
                    if task_name in new_tasks:
                        duration_ms = (now - last_completion_time) * 1000.0
                        completion_order.append(task_name)
                        task_durations_ms[task_name] = duration_ms
                        last_completion_time = now
                prev_completed = current_completed

            pbar.close()
            episode_end = time.perf_counter()
            episode_duration_ms = (episode_end - episode_start) * 1000.0

            if enable_render and video_path is not None:
                env.render()

            completed_tasks = sorted(prev_completed)
            all_7_success = len(prev_completed) == len(ALL_TASKS)

            result: Dict[str, Any] = {
                "episode_idx": episode_idx,
                "init_idx": init_idx,
                "completed_tasks": completed_tasks,
                "completion_order": completion_order,
                "all_7_success": all_7_success,
                "video_path": os.path.relpath(str(video_path), self.output_dir)
                if video_path is not None
                else None,
                "inference_latencies_ms": inference_latencies_ms,
                "episode_duration_ms": episode_duration_ms,
                "task_durations_ms": task_durations_ms,
            }

            if traj is not None and len(traj["policy_obs"]) > 0:
                stacked = {
                    "demo_obs": traj["demo_obs"],
                    "demo_action": traj["demo_action"],
                    "demo_valid_len": traj["demo_valid_len"],
                    "policy_obs": np.stack(traj["policy_obs"], axis=0),
                    "action_executed": np.stack(traj["action_executed"], axis=0),
                    "action_pred": np.stack(traj["action_pred"], axis=0),
                    "executed_obs": np.stack(traj["executed_obs"], axis=0),
                    "executed_action": np.stack(traj["executed_action"], axis=0),
                    "qp": np.stack(traj["qp"], axis=0),
                    "qv": np.stack(traj["qv"], axis=0),
                    "obj_qp": np.stack(traj["obj_qp"], axis=0),
                    "obj_qv": np.stack(traj["obj_qv"], axis=0),
                    "demo_obs_at_step": np.stack(traj["demo_obs_at_step"], axis=0),
                    "demo_action_at_step": np.stack(
                        traj["demo_action_at_step"], axis=0
                    ),
                    "action_error_l2": np.asarray(
                        traj["action_error_l2"], dtype=np.float32
                    ),
                    "obs_error_l2": np.asarray(traj["obs_error_l2"], dtype=np.float32),
                    "horizon": traj["horizon"] or int(traj["action_pred"][0].shape[0]),
                    "n_control_steps": len(traj["policy_obs"]),
                    "n_env_steps": len(traj["executed_obs"]),
                }
                if traj["obs_pred"]:
                    stacked["obs_pred"] = np.stack(traj["obs_pred"], axis=0)
                if traj["action_obs_pred"]:
                    stacked["action_obs_pred"] = np.stack(
                        traj["action_obs_pred"], axis=0
                    )
                result["trajectory"] = stacked

            return result
        finally:
            try:
                _close_env(env)
            except Exception:
                pass

    def run(self, policy: BaseLowdimPolicy) -> Dict[str, Any]:
        episode_records: List[Dict[str, Any]] = []
        all_inference_latencies_ms: List[float] = []
        all_episode_durations_ms: List[float] = []
        all_task_durations_ms: List[float] = []
        task_duration_by_name: Dict[str, List[float]] = {t: [] for t in ALL_TASKS}
        per_task_success: Dict[str, List[int]] = {t: [] for t in ALL_TASKS}
        all_7_success_flags: List[int] = []
        completion_positions: Dict[str, List[int]] = {t: [] for t in ALL_TASKS}

        for episode_idx in range(self.n_episodes):
            enable_render = episode_idx < self.n_episodes_vis
            record = self._run_single_episode(policy, episode_idx, enable_render)
            completed_set = set(record["completed_tasks"])
            task_success = {
                task_name: (1 if task_name in completed_set else 0)
                for task_name in ALL_TASKS
            }

            trajectory_log_path = None
            if self.save_trajectory_logs and "trajectory" in record:
                trajectory_log_path = self._save_trajectory_log(
                    episode_idx=record["episode_idx"],
                    init_idx=record["init_idx"],
                    trajectory=record["trajectory"],
                    summary={
                        "all_7_success": record["all_7_success"],
                        "completed_tasks": record["completed_tasks"],
                        "completion_order": record["completion_order"],
                    },
                )

            episode_records.append(
                {
                    "episode_idx": record["episode_idx"],
                    "init_idx": record["init_idx"],
                    "completed_tasks": record["completed_tasks"],
                    "completion_order": record["completion_order"],
                    "task_success": task_success,
                    "num_tasks_completed": len(completed_set),
                    "all_7_success": record["all_7_success"],
                    "video_path": record["video_path"],
                    "trajectory_log_path": trajectory_log_path,
                    "episode_duration_ms": record["episode_duration_ms"],
                    "task_durations_ms": record["task_durations_ms"],
                    "mean_inference_latency_ms": float(
                        np.mean(record["inference_latencies_ms"])
                    )
                    if record["inference_latencies_ms"]
                    else None,
                }
            )

            all_inference_latencies_ms.extend(record["inference_latencies_ms"])
            all_episode_durations_ms.append(record["episode_duration_ms"])
            all_7_success_flags.append(1 if record["all_7_success"] else 0)

            for task_name in ALL_TASKS:
                per_task_success[task_name].append(task_success[task_name])

            for pos, task_name in enumerate(record["completion_order"]):
                completion_positions[task_name].append(pos)

            for task_name, duration_ms in record["task_durations_ms"].items():
                all_task_durations_ms.append(duration_ms)
                task_duration_by_name[task_name].append(duration_ms)

        success_rate = {}
        for task_name in ALL_TASKS:
            success_rate[task_name] = _compute_mean_std(
                [float(x) for x in per_task_success[task_name]]
            )
        success_rate["all_7_tasks"] = _compute_mean_std(
            [float(x) for x in all_7_success_flags]
        )

        task_duration_stats = {"overall": _compute_mean_std(all_task_durations_ms)}
        for task_name in ALL_TASKS:
            task_duration_stats[task_name] = _compute_mean_std(
                task_duration_by_name[task_name]
            )

        completion_order_stats = {}
        for task_name in ALL_TASKS:
            positions = completion_positions[task_name]
            completion_order_stats[task_name] = {
                "mean_completion_position": _compute_mean_std(positions)["mean"],
                "completion_count": len(positions),
            }

        multistage_all_7 = compute_multistage_metrics(
            episode_records, sub_goals=ALL_TASKS
        )
        multistage_4 = compute_multistage_metrics(
            episode_records, sub_goals=KITCHEN_4_SUBGOALS, num_sub_goals=4
        )

        return {
            "n_episodes": self.n_episodes,
            "tasks": ALL_TASKS,
            "task_note": TASK_NOTE,
            "success_rate": success_rate,
            "multistage_metrics": {
                "all_7_tasks": {
                    "px": multistage_all_7["px"],
                    "cumulative_order_success_rate": multistage_all_7[
                        "cumulative_order_success_rate"
                    ],
                    "sub_goals": multistage_all_7["sub_goals"],
                },
                "paper_4_tasks": {
                    "px": multistage_4["px"],
                    "cumulative_order_success_rate": multistage_4[
                        "cumulative_order_success_rate"
                    ],
                    "sub_goals": multistage_4["sub_goals"],
                },
            },
            "timing_ms": {
                "inference_latency": _compute_mean_std(all_inference_latencies_ms),
                "episode_duration": _compute_mean_std(all_episode_durations_ms),
                "task_duration": task_duration_stats,
            },
            "completion_order_stats": completion_order_stats,
            "episodes": episode_records,
        }
