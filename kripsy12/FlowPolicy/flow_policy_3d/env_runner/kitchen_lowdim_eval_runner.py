import gc
import time
import pathlib
import logging
import os
from typing import Any, Dict, List, Set
import numpy as np
import torch
import tqdm

from flow_policy_3d.env.kitchen.base import KitchenBase
from flow_policy_3d.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
from flow_policy_3d.env.kitchen.v0 import KitchenAllV0
from flow_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from flow_policy_3d.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from flow_policy_3d.policy.base_lowdim_policy import BaseLowdimPolicy
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.common.multistage_metrics import compute_multistage_metrics
from flow_policy_3d.env_runner.base_lowdim_runner import BaseLowdimRunner

module_logger = logging.getLogger(__name__)


def _close_env(env: MultiStepWrapper):
    if isinstance(env.env, VideoRecordingWrapper):
        env.env.video_recoder.stop()
        env.env.file_path = None
    env.close()
    gc.collect()


ALL_TASKS = list(KitchenBase.ALL_TASKS)
# 4-object Kitchen subset referenced in Diffusion Policy paper / BET benchmark
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


class KitchenLowdimEvalRunner(BaseLowdimRunner):
    def __init__(
        self,
        output_dir,
        eval_seed: int = 0,
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
        robot_noise_ratio: float = 0.1,
        tqdm_interval_sec: float = 5.0,
    ):
        super().__init__(output_dir)
        self.eval_seed = eval_seed
        self.n_episodes = n_episodes
        self.n_episodes_vis = n_episodes_vis
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.abs_action = abs_action
        self.robot_noise_ratio = robot_noise_ratio
        self.fps = fps
        self.crf = crf
        self.render_hw = render_hw

        pathlib.Path(output_dir).joinpath("media").mkdir(parents=True, exist_ok=True)

        task_fps = 12.5
        steps_per_render = int(max(task_fps // fps, 1))

        def env_fn():
            env = KitchenAllV0(use_abs_action=abs_action)
            env.robot_noise_ratio = robot_noise_ratio
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    KitchenLowdimWrapper(
                        env=env,
                        init_qpos=None,
                        init_qvel=None,
                        render_hw=tuple(render_hw),
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
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

    def _make_env(self, episode_idx: int, enable_render: bool) -> MultiStepWrapper:
        env = self._env_fn()
        assert isinstance(env, MultiStepWrapper)
        assert isinstance(env.env, VideoRecordingWrapper)

        video_path = None
        if enable_render:
            video_path = pathlib.Path(self.output_dir).joinpath(
                "media", f"seed{self.eval_seed}_ep{episode_idx:03d}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            env.env.file_path = str(video_path)
        else:
            env.env.file_path = None

        assert isinstance(env.env.env, KitchenLowdimWrapper)
        env.env.env.init_qpos = None
        env.env.env.init_qvel = None

        env_seed = self.eval_seed + episode_idx
        env.seed(env_seed)
        return env, video_path

    def _sync_device(self, device: torch.device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _run_single_episode(
        self,
        policy: BaseLowdimPolicy,
        episode_idx: int,
        enable_render: bool,
    ) -> Dict[str, Any]:
        device = policy.device
        env, video_path = self._make_env(episode_idx, enable_render)

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

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Ep {episode_idx + 1}/{self.n_episodes} seed={self.eval_seed}",
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

                obs, reward, done, info = env.step(action)
                done = bool(done)
                past_action = np_action_dict["action"]
                pbar.update(action.shape[0])

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

            return {
                "episode_idx": episode_idx,
                "env_seed": self.eval_seed + episode_idx,
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
            episode_records.append(
                {
                    "episode_idx": record["episode_idx"],
                    "env_seed": record["env_seed"],
                    "completed_tasks": record["completed_tasks"],
                    "completion_order": record["completion_order"],
                    "task_success": task_success,
                    "num_tasks_completed": len(completed_set),
                    "all_7_success": record["all_7_success"],
                    "video_path": record["video_path"],
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
            "eval_seed": self.eval_seed,
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
