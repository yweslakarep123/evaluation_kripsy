import csv
import gc
import json
import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import tqdm

from flow_policy_3d.common.kitchen_combo_protocol import (
    EPISODES_PER_COMBINATION,
    aggregate_combination_metrics,
    aggregate_seed_metrics,
    build_episode_schedule,
    format_seed_report,
    save_json,
)
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
from flow_policy_3d.env.kitchen.kitchen_sequential_v0 import KitchenSequential4V0
from flow_policy_3d.env_runner.base_lowdim_runner import BaseLowdimRunner
from flow_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from flow_policy_3d.gym_util.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from flow_policy_3d.policy.base_lowdim_policy import BaseLowdimPolicy

module_logger = logging.getLogger(__name__)

NUM_JOINTS = 9


def _close_env(env: MultiStepWrapper):
    if isinstance(env.env, VideoRecordingWrapper):
        env.env.video_recoder.stop()
        env.env.file_path = None
    env.close()
    gc.collect()


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


class KitchenComboEvalRunner(BaseLowdimRunner):
    def __init__(
        self,
        output_dir,
        dataset_dir,
        model_seed: str,
        max_steps: int = 280,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        render_hw=(240, 360),
        fps: float = 12.5,
        crf: int = 22,
        past_action: bool = False,
        abs_action: bool = True,
        robot_noise_ratio: float = 0.1,
        tqdm_interval_sec: float = 5.0,
        n_episodes_per_combo: int = EPISODES_PER_COMBINATION,
        combination_ids: Optional[Sequence[int]] = None,
        resume: bool = True,
    ):
        super().__init__(output_dir)
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.model_seed = model_seed
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
        self.n_episodes_per_combo = n_episodes_per_combo
        self.resume = resume

        self.init_qpos = np.load(self.dataset_dir / "all_init_qpos.npy")
        self.init_qvel = np.load(self.dataset_dir / "all_init_qvel.npy")
        n_inits = len(self.init_qpos)

        self.schedule = build_episode_schedule(
            n_episodes_per_combo=n_episodes_per_combo,
            n_inits=n_inits,
            combination_ids=combination_ids,
        )
        self._schedule_by_combo: Dict[int, List[Dict[str, Any]]] = {}
        for ep_cfg in self.schedule:
            self._schedule_by_combo.setdefault(ep_cfg["combination_id"], []).append(
                ep_cfg
            )

        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _build_env(self, task_sequence, init_qpos, init_qvel, video_path):
        task_fps = 12.5
        steps_per_render = int(max(task_fps // self.fps, 1))

        env = KitchenSequential4V0(task_sequence, use_abs_action=self.abs_action)
        env.robot_noise_ratio = self.robot_noise_ratio

        return MultiStepWrapper(
            VideoRecordingWrapper(
                KitchenLowdimWrapper(
                    env=env,
                    init_qpos=init_qpos,
                    init_qvel=init_qvel,
                    render_hw=tuple(self.render_hw),
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=self.fps,
                    codec="h264",
                    input_pix_fmt="rgb24",
                    crf=self.crf,
                    thread_type="FRAME",
                    thread_count=1,
                ),
                file_path=str(video_path) if video_path is not None else None,
                steps_per_render=steps_per_render,
            ),
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_episode_steps=self.max_steps,
        )

    def _sync_device(self, device: torch.device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _write_joint_csv(
        self,
        path: pathlib.Path,
        rows: List[Dict[str, Any]],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "seed",
            "combination_id",
            "episode_id",
            "timestep",
            "joint_idx",
            "actual_qpos",
            "predicted_qpos",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _run_single_episode(
        self,
        policy: BaseLowdimPolicy,
        ep_cfg: Dict[str, Any],
        ep_output_dir: pathlib.Path,
    ) -> Dict[str, Any]:
        device = policy.device
        combo_id = ep_cfg["combination_id"]
        ep_id = ep_cfg["episode_id"]
        task_sequence = ep_cfg["task_sequence"]
        init_idx = ep_cfg["init_idx"]

        video_path = ep_output_dir / f"ep_{ep_id:03d}.mp4"
        joint_path = ep_output_dir / f"ep_{ep_id:03d}_joints.csv"
        meta_path = ep_output_dir / f"ep_{ep_id:03d}.json"

        if self.resume and meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)

        init_qpos = self.init_qpos[init_idx]
        init_qvel = self.init_qvel[init_idx]
        env = self._build_env(task_sequence, init_qpos, init_qvel, video_path)

        joint_rows: List[Dict[str, Any]] = []
        env_timestep = 0

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

                prev_obs_len = len(env.obs)
                obs, reward, done, info = env.step(action)
                done = bool(done)
                past_action = np_action_dict["action"]

                new_obs_list = list(env.obs)[prev_obs_len:]
                n_steps = min(len(new_obs_list), action.shape[0])
                for step_i in range(n_steps):
                    actual = new_obs_list[step_i][:NUM_JOINTS]
                    predicted = action[step_i][:NUM_JOINTS]
                    for joint_idx in range(NUM_JOINTS):
                        joint_rows.append(
                            {
                                "seed": self.model_seed,
                                "combination_id": combo_id,
                                "episode_id": ep_id,
                                "timestep": env_timestep,
                                "joint_idx": joint_idx,
                                "actual_qpos": float(actual[joint_idx]),
                                "predicted_qpos": float(predicted[joint_idx]),
                            }
                        )
                    env_timestep += 1

                now = time.perf_counter()
                current_completed = _extract_completed_tasks(info)
                new_tasks = current_completed - prev_completed
                for task_name in task_sequence:
                    if task_name in new_tasks:
                        duration_ms = (now - last_completion_time) * 1000.0
                        completion_order.append(task_name)
                        task_durations_ms[task_name] = duration_ms
                        last_completion_time = now
                prev_completed = current_completed

            episode_end = time.perf_counter()
            episode_duration_ms = (episode_end - episode_start) * 1000.0

            env.render()
            completed_tasks = sorted(prev_completed)

            record = {
                "combination_id": combo_id,
                "combination_tasks": ep_cfg["combination_tasks"],
                "episode_id": ep_id,
                "task_sequence": task_sequence,
                "init_idx": init_idx,
                "completed_tasks": completed_tasks,
                "completion_order": completion_order,
                "task_durations_ms": task_durations_ms,
                "inference_latencies_ms": inference_latencies_ms,
                "episode_duration_ms": episode_duration_ms,
                "video_path": os.path.relpath(str(video_path), self.output_dir),
                "joints_path": os.path.relpath(str(joint_path), self.output_dir),
                "n_env_steps": env_timestep,
            }

            self._write_joint_csv(joint_path, joint_rows)
            with open(meta_path, "w") as f:
                json.dump(record, f, indent=2)

            return record
        finally:
            try:
                _close_env(env)
            except Exception:
                pass

    def run(self, policy: BaseLowdimPolicy) -> Dict[str, Any]:
        combo_metrics: List[Dict[str, Any]] = []

        combo_ids = sorted(self._schedule_by_combo.keys())
        combo_pbar = tqdm.tqdm(combo_ids, desc=f"Combos seed={self.model_seed}")

        for combo_id in combo_pbar:
            combo_dir = pathlib.Path(self.output_dir) / f"combination_{combo_id:02d}"
            ep_dir = combo_dir / "episodes"
            ep_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = combo_dir / "metrics.json"

            ep_cfgs = self._schedule_by_combo[combo_id]
            combo_pbar.set_postfix(combo=combo_id)

            if self.resume and metrics_path.exists():
                with open(metrics_path) as f:
                    combo_metrics.append(json.load(f))
                continue

            episode_records: List[Dict[str, Any]] = []
            for ep_cfg in tqdm.tqdm(
                ep_cfgs,
                desc=f"Combo {combo_id}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            ):
                record = self._run_single_episode(policy, ep_cfg, ep_dir)
                episode_records.append(record)

            cm = aggregate_combination_metrics(
                episode_records,
                combination_id=combo_id,
                combination_tasks=ep_cfgs[0]["combination_tasks"],
            )
            save_json(cm, str(metrics_path))
            combo_metrics.append(cm)

        seed_summary = aggregate_seed_metrics(combo_metrics, self.model_seed)
        report = format_seed_report(seed_summary)

        save_json(seed_summary, str(pathlib.Path(self.output_dir) / "seed_summary.json"))
        with open(pathlib.Path(self.output_dir) / "seed_report.txt", "w") as f:
            f.write(report)

        return seed_summary
