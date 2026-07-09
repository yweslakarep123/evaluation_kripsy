import gc
import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import tqdm

from flow_policy_3d.common.multistage_metrics import compute_multistage_metrics
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.env.kitchen.base import KitchenBase
from flow_policy_3d.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
from flow_policy_3d.env.kitchen.v0 import KitchenAllV0
from flow_policy_3d.env_runner.base_lowdim_runner import BaseLowdimRunner
from flow_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from flow_policy_3d.gym_util.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from flow_policy_3d.policy.base_lowdim_policy import BaseLowdimPolicy

module_logger = logging.getLogger(__name__)

TRAJECTORY_SCHEMA = """Kitchen trajectory log schema (ep_XXXX.npz)
================================================
Scalars: episode_idx, init_idx, demo_valid_len, n_control_steps, n_env_steps,
         horizon, n_obs_steps, n_action_steps, action_slice_start, action_slice_end,
         has_obs_pred

Window / horizon metadata:
  action_slice_start/end   indices into action_pred horizon for executed chunk
  control_step_per_env_step (T_env,)     env step i -> control step index
  env_step_at_control_start (T_ctrl,)    first env step of each control step
  env_step_at_control_end   (T_ctrl,)    last env step of each control step

Name labels (for plotting):
  robot_joint_names  (9,)   string
  action_names       (9,)   string
  obj_qp_names       (21,)  string
  goal_names         (30,)  string

Demo ground truth (column init_idx in observations_seq/actions_seq):
  demo_obs          (T_demo, 60)   qp(9) + obj_qp(21) + goal(30)
  demo_action       (T_demo, 9)
  demo_qp           (T_demo, 9)
  demo_obj_qp       (T_demo, 21)
  demo_goal         (T_demo, 30)

Per control step (one predict_action call):
  policy_obs        (T_ctrl, n_obs_steps, 60)
  policy_obs_qp     (T_ctrl, n_obs_steps, 9)
  policy_obs_obj_qp (T_ctrl, n_obs_steps, 21)
  policy_obs_goal   (T_ctrl, n_obs_steps, 30)
  action_executed   (T_ctrl, n_action_steps, 9)
  action_pred       (T_ctrl, horizon, 9)
  action_pred_executed_l2 (T_ctrl, n_action_steps)  L2 vs action_executed per sub-step
  obs_pred          (T_ctrl, horizon, 60)   optional (DP inpainting)
  obs_pred_qp/obj_qp/goal                   optional decompositions of obs_pred
  action_obs_pred   (T_ctrl, n_action_steps, 60)   optional

Per env step (each sub-step in MultiStepWrapper):
  executed_obs      (T_env, 60)
  executed_qp       (T_env, 9)
  executed_obj_qp   (T_env, 21)
  executed_goal     (T_env, 30)
  executed_action   (T_env, 9)
  qp                (T_env, 9)    robot joint positions (from obs_dict)
  qv                (T_env, 9)    robot joint velocities
  obj_qp            (T_env, 21)   object positions
  obj_qv            (T_env, 21)   object velocities
  demo_obs_at_step  (T_env, 60)   NaN if env step exceeds demo length
  demo_obs_at_step_qp/obj_qp/goal           decomposed demo obs at step
  demo_action_at_step (T_env, 9)  NaN if env step exceeds demo length
  action_error_l2   (T_env,)      L2 vs demo action
  obs_error_l2      (T_env,)      L2 vs demo obs

Note: demo GT has no velocity; qv/obj_qv are rollout-only from env obs_dict.

Human-readable detail file (ep_XXXX_detail.txt):
  - Index legend for joint/action/obs component names
  - Policy Window & Horizon (n_obs_steps, n_action_steps, horizon, slice indices)
  - Demo GT action & joint-position sequence tables
  - Per control step: policy obs window, predicted action horizon, executed chunk
  - Per env step: labeled joints (qp/qv), decomposed observation, action, demo GT, errors
"""


def _fmt_window_horizon_section(
    n_obs_steps: int,
    n_action_steps: int,
    horizon: int,
    has_obs_pred: bool,
) -> List[str]:
    action_start = n_obs_steps - 1
    action_end = action_start + n_action_steps
    lines = [
        "## Policy Window & Horizon",
        "  How the policy is called each control step (MultiStepWrapper + diffusion/flow policy):",
        "",
        f"  Observation sliding window  n_obs_steps = {n_obs_steps}",
        f"    • Env maintains a deque of the last {n_obs_steps} observations.",
        f"    • Policy input shape: ({n_obs_steps}, 60) — logged as policy_obs[control, frame_0..frame_{n_obs_steps - 1}].",
        f"    • frame_0 = oldest in window, frame_{n_obs_steps - 1} = most recent.",
        "",
        f"  Action prediction horizon     horizon = {horizon}",
        f"    • Policy outputs action_pred with shape ({horizon}, 9): a full future action sequence.",
    ]
    if has_obs_pred:
        lines.extend([
            f"    • Policy also outputs obs_pred with shape ({horizon}, 60) (inpainting / joint pred mode).",
            f"    • action_obs_pred = obs_pred[{action_start}:{action_end}] when present.",
        ])
    else:
        lines.append(
            "    • obs_pred not produced by this policy checkpoint (obs-as-global-cond mode)."
        )
    lines.extend([
        "",
        f"  Action execution chunk        n_action_steps = {n_action_steps}",
        f"    • Only a slice of action_pred is executed per control step:",
        f"      action_executed = action_pred[{action_start}:{action_end}]  (indices inclusive start, exclusive end).",
        f"    • Env applies {n_action_steps} sub-steps before the next predict_action call.",
        f"    • Remaining horizon steps [{action_end}:{horizon}] are not executed (re-planned next control step).",
        "",
        "  Sliding control timeline (one line per control step):",
        f"    control k covers env steps [k*{n_action_steps} .. (k+1)*{n_action_steps}-1]",
        f"    obs window at control k = last {n_obs_steps} executed obs ending at env step k*{n_action_steps}-1",
        "",
    ])
    return lines


ROBOT_JOINT_NAMES = [
    "q0", "q1", "q2", "q3", "q4", "q5", "q6", "finger_l", "finger_r",
]
ACTION_NAMES = [
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "grip_l", "grip_r",
]
OBJ_QP_NAMES = [f"obj_{i:02d}" for i in range(21)]
GOAL_NAMES = [f"goal_{i:02d}" for i in range(30)]


def _close_env(env: MultiStepWrapper):
    #region agent log
    import json as _json, os as _os, time as _time
    _fd_before = len(_os.listdir('/proc/self/fd'))
    #endregion
    if isinstance(env.env, VideoRecordingWrapper):
        env.env.video_recoder.stop()
        env.env.file_path = None
    env.close()
    gc.collect()
    #region agent log
    _fd_after = len(_os.listdir('/proc/self/fd'))
    open('/home/daffa/Documents/experiment/.cursor/debug-374ef1.log','a').write(_json.dumps({"sessionId":"374ef1","runId":"post-fix","hypothesisId":"B","location":"kitchen_lowdim_eval_runner.py:_close_env","message":"env closed","data":{"fd_before":_fd_before,"fd_after":_fd_after,"fd_delta":_fd_after-_fd_before},"timestamp":int(_time.time()*1000)})+'\n')
    #endregion


ALL_TASKS = list(KitchenBase.ALL_TASKS)
KITCHEN_4_SUBGOALS = ["microwave", "kettle", "bottom burner", "light switch"]
TASK_NOTE = (
    "Tasks can complete in any order; all 7 must finish for full episode success. "
    "Task/episode duration is simulation/video time: (env_steps / fps) * 1000 ms, "
    "from the previous task completion (or episode start) until that task completes. "
    "Inference latency remains wall-clock GPU time per predict_action."
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


def _split_obs_60(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    return obs[:9], obs[9:30], obs[30:60]


def _decompose_obs_array(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32)
    return obs[..., :9], obs[..., 9:30], obs[..., 30:60]


def _build_enriched_npz_data(
    trajectory: Dict[str, Any],
    *,
    episode_idx: int,
    init_idx: Optional[int],
    n_obs_steps: int,
    n_action_steps: int,
) -> Dict[str, Any]:
    action_slice_start = n_obs_steps - 1
    action_slice_end = action_slice_start + n_action_steps
    n_env_steps = int(trajectory["n_env_steps"])
    n_control_steps = int(trajectory["n_control_steps"])

    npz_data: Dict[str, Any] = {
        "episode_idx": np.int32(episode_idx),
        "init_idx": np.int32(init_idx if init_idx is not None else -1),
        "demo_valid_len": np.int32(trajectory["demo_valid_len"]),
        "n_control_steps": np.int32(n_control_steps),
        "n_env_steps": np.int32(n_env_steps),
        "horizon": np.int32(trajectory["horizon"]),
        "n_obs_steps": np.int32(n_obs_steps),
        "n_action_steps": np.int32(n_action_steps),
        "action_slice_start": np.int32(action_slice_start),
        "action_slice_end": np.int32(action_slice_end),
        "has_obs_pred": np.int32(1 if trajectory.get("obs_pred") is not None else 0),
        "robot_joint_names": np.asarray(ROBOT_JOINT_NAMES, dtype="U10"),
        "action_names": np.asarray(ACTION_NAMES, dtype="U10"),
        "obj_qp_names": np.asarray(OBJ_QP_NAMES, dtype="U10"),
        "goal_names": np.asarray(GOAL_NAMES, dtype="U10"),
    }

    if n_env_steps > 0:
        npz_data["control_step_per_env_step"] = (
            np.arange(n_env_steps, dtype=np.int32) // n_action_steps
        )
    else:
        npz_data["control_step_per_env_step"] = np.zeros((0,), dtype=np.int32)

    if n_control_steps > 0:
        ctrl_idx = np.arange(n_control_steps, dtype=np.int32)
        npz_data["env_step_at_control_start"] = ctrl_idx * n_action_steps
        npz_data["env_step_at_control_end"] = np.minimum(
            (ctrl_idx + 1) * n_action_steps - 1,
            max(n_env_steps - 1, 0),
        )
    else:
        npz_data["env_step_at_control_start"] = np.zeros((0,), dtype=np.int32)
        npz_data["env_step_at_control_end"] = np.zeros((0,), dtype=np.int32)

    demo_obs = trajectory["demo_obs"]
    demo_action = trajectory["demo_action"]
    npz_data["demo_obs"] = demo_obs
    npz_data["demo_action"] = demo_action
    if len(demo_obs) > 0:
        dqp, dobj, dgoal = _decompose_obs_array(demo_obs)
        npz_data["demo_qp"] = dqp
        npz_data["demo_obj_qp"] = dobj
        npz_data["demo_goal"] = dgoal

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

    if "policy_obs" in npz_data:
        pqp, pobj, pgoal = _decompose_obs_array(npz_data["policy_obs"])
        npz_data["policy_obs_qp"] = pqp
        npz_data["policy_obs_obj_qp"] = pobj
        npz_data["policy_obs_goal"] = pgoal

    if "obs_pred" in npz_data:
        oqp, oobj, ogoal = _decompose_obs_array(npz_data["obs_pred"])
        npz_data["obs_pred_qp"] = oqp
        npz_data["obs_pred_obj_qp"] = oobj
        npz_data["obs_pred_goal"] = ogoal

    if "executed_obs" in npz_data:
        eqp, eobj, egoal = _decompose_obs_array(npz_data["executed_obs"])
        npz_data["executed_qp"] = eqp
        npz_data["executed_obj_qp"] = eobj
        npz_data["executed_goal"] = egoal

    if "demo_obs_at_step" in npz_data:
        dasp, dasobj, dasgoal = _decompose_obs_array(npz_data["demo_obs_at_step"])
        npz_data["demo_obs_at_step_qp"] = dasp
        npz_data["demo_obs_at_step_obj_qp"] = dasobj
        npz_data["demo_obs_at_step_goal"] = dasgoal

    action_pred = npz_data.get("action_pred")
    action_executed = npz_data.get("action_executed")
    if action_pred is not None and action_executed is not None:
        pred_slice = action_pred[:, action_slice_start:action_slice_end]
        npz_data["action_pred_executed_l2"] = np.linalg.norm(
            pred_slice - action_executed, axis=-1
        ).astype(np.float32)

    return npz_data


def _fmt_labeled(
    values: np.ndarray,
    names: List[str],
    *,
    precision: int = 6,
    cols: int = 5,
    indent: str = "    ",
) -> List[str]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = min(len(values), len(names))
    cells = [f"{names[i]:>8}={values[i]:+.{precision}f}" for i in range(n)]
    lines: List[str] = []
    for i in range(0, len(cells), cols):
        lines.append(indent + "  ".join(cells[i : i + cols]))
    return lines


def _fmt_table_header(col_names: List[str], widths: Optional[List[int]] = None) -> str:
    if widths is None:
        widths = [10] * len(col_names)
    return "  ".join(f"{name:>{w}}" for name, w in zip(col_names, widths))


def _fmt_table_row(
    row_label: str,
    values: np.ndarray,
    col_names: List[str],
    *,
    label_width: int = 6,
    val_width: int = 10,
    precision: int = 4,
) -> str:
    cells = [
        f"{v:>{val_width}.{precision}f}" if not np.isnan(v) else f"{'NaN':>{val_width}}"
        for v in np.asarray(values, dtype=np.float64).reshape(-1)
    ]
    n = min(len(cells), len(col_names))
    return f"{row_label:>{label_width}}  " + "  ".join(cells[:n])


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

    def _create_env(self) -> MultiStepWrapper:
        return self._env_fn()

    def _configure_env(
        self, env: MultiStepWrapper, episode_idx: int, enable_render: bool
    ) -> Tuple[MultiStepWrapper, Optional[pathlib.Path]]:
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

    def _make_env(self, episode_idx: int, enable_render: bool):
        return self._configure_env(self._create_env(), episode_idx, enable_render)

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

        npz_data = _build_enriched_npz_data(
            trajectory,
            episode_idx=episode_idx,
            init_idx=init_idx,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
        )

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
        demo_obs = trajectory["demo_obs"]
        demo_action = trajectory["demo_action"]
        policy_obs = trajectory.get("policy_obs")
        action_executed = trajectory.get("action_executed")
        action_pred = trajectory.get("action_pred")
        obs_pred = trajectory.get("obs_pred")
        executed_obs = trajectory.get("executed_obs")
        executed_action = trajectory.get("executed_action")
        qp = trajectory.get("qp")
        qv = trajectory.get("qv")
        obj_qp = trajectory.get("obj_qp")
        obj_qv = trajectory.get("obj_qv")
        demo_obs_at_step = trajectory.get("demo_obs_at_step")
        demo_action_at_step = trajectory.get("demo_action_at_step")
        action_error_l2 = trajectory.get("action_error_l2")
        obs_error_l2 = trajectory.get("obs_error_l2")

        lines = [
            "#" * 78,
            f"# KITCHEN TRAJECTORY LOG — Episode {episode_idx:04d}",
            "#" * 78,
            "",
            "## Index / Legend",
            "  Observation (60-dim) = robot_qp(9) + object_qp(21) + goal(30)",
            "  Robot joints:",
            "    " + ", ".join(ROBOT_JOINT_NAMES),
            "  Action dims:",
            "    " + ", ".join(ACTION_NAMES),
            "",
            "## Episode Summary",
            f"  init_idx          : {init_idx}",
            f"  demo_valid_len    : {trajectory['demo_valid_len']}",
            f"  n_control_steps   : {trajectory['n_control_steps']}",
            f"  n_env_steps       : {trajectory['n_env_steps']}",
            f"  horizon           : {trajectory['horizon']}",
            f"  n_obs_steps       : {self.n_obs_steps}",
            f"  n_action_steps    : {self.n_action_steps}",
            f"  all_7_success     : {summary['all_7_success']}",
            f"  completed_tasks   : {summary['completed_tasks']}",
            f"  completion_order  : {summary['completion_order']}",
            "",
        ]
        lines.extend(
            _fmt_window_horizon_section(
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.n_action_steps,
                horizon=int(trajectory["horizon"]),
                has_obs_pred=obs_pred is not None,
            )
        )

        if len(demo_action) > 0:
            lines.extend([
                "## Demo GT — Action Sequence",
                "  " + _fmt_table_header(["step"] + ACTION_NAMES),
                "  " + "-" * (6 + 11 * len(ACTION_NAMES)),
            ])
            for t in range(len(demo_action)):
                lines.append(
                    "  "
                    + _fmt_table_row(str(t), demo_action[t], ACTION_NAMES, label_width=4)
                )
            lines.append("")

        if len(demo_obs) > 0:
            lines.extend([
                "## Demo GT — Robot Joint Positions (from demo obs[:9])",
                "  " + _fmt_table_header(["step"] + ROBOT_JOINT_NAMES),
                "  " + "-" * (6 + 11 * len(ROBOT_JOINT_NAMES)),
            ])
            for t in range(len(demo_obs)):
                demo_qp, _, _ = _split_obs_60(demo_obs[t])
                lines.append(
                    "  "
                    + _fmt_table_row(
                        str(t), demo_qp, ROBOT_JOINT_NAMES, label_width=4, precision=4
                    )
                )
            lines.append("")

        if executed_action is not None and len(executed_action) > 0:
            lines.extend([
                "## Rollout — Executed Action Sequence",
                "  " + _fmt_table_header(["step"] + ACTION_NAMES + ["act_err"]),
                "  " + "-" * (6 + 11 * (len(ACTION_NAMES) + 1)),
            ])
            for t in range(len(executed_action)):
                err = action_error_l2[t] if action_error_l2 is not None else np.nan
                row_vals = np.concatenate([executed_action[t], [err]])
                row_names = ACTION_NAMES + ["act_err"]
                lines.append(
                    "  "
                    + _fmt_table_row(str(t), row_vals, row_names, label_width=4)
                )
            lines.append("")

        lines.extend([
            "## Rollout Detail (per control step → env sub-steps)",
            "",
        ])

        n_ctrl = trajectory["n_control_steps"]
        env_step = 0
        for ctrl in range(n_ctrl):
            lines.extend([
                "=" * 78,
                f"CONTROL STEP {ctrl}  (env steps {env_step} .. "
                f"{min(env_step + self.n_action_steps - 1, trajectory['n_env_steps'] - 1)})",
                "=" * 78,
                "",
            ])

            if policy_obs is not None:
                lines.append(
                    f"  Policy input observation (sliding window, {self.n_obs_steps} frames):"
                )
                for obs_i in range(policy_obs.shape[1]):
                    obs_vec = policy_obs[ctrl, obs_i]
                    rqp, objp, goal = _split_obs_60(obs_vec)
                    lines.append(f"    frame[{obs_i}] robot qp:")
                    lines.extend(_fmt_labeled(rqp, ROBOT_JOINT_NAMES))
                    lines.append(f"    frame[{obs_i}] object qp:")
                    lines.extend(_fmt_labeled(objp, OBJ_QP_NAMES, cols=4))
                    lines.append(f"    frame[{obs_i}] goal:")
                    lines.extend(_fmt_labeled(goal, GOAL_NAMES, cols=5))
                lines.append("")

            if action_pred is not None:
                action_start = self.n_obs_steps - 1
                action_end = action_start + self.n_action_steps
                lines.extend([
                    "  Predicted action sequence (full horizon):",
                    f"    (executed slice: indices [{action_start}:{action_end}] of each row below)",
                    "    "
                    + _fmt_table_header(["h"] + ACTION_NAMES),
                    "    " + "-" * (4 + 11 * len(ACTION_NAMES)),
                ])
                for h in range(action_pred.shape[1]):
                    lines.append(
                        "    "
                        + _fmt_table_row(
                            str(h), action_pred[ctrl, h], ACTION_NAMES, label_width=3
                        )
                    )
                lines.append("")

            if action_executed is not None:
                action_start = self.n_obs_steps - 1
                action_end = action_start + self.n_action_steps
                lines.extend([
                    f"  Executed action chunk (action_pred[{action_start}:{action_end}]):",
                    "    "
                    + _fmt_table_header(["sub"] + ACTION_NAMES),
                    "    " + "-" * (5 + 11 * len(ACTION_NAMES)),
                ])
                for sub in range(action_executed.shape[1]):
                    lines.append(
                        "    "
                        + _fmt_table_row(
                            str(sub),
                            action_executed[ctrl, sub],
                            ACTION_NAMES,
                            label_width=4,
                        )
                    )
                lines.append("")

            if obs_pred is not None:
                lines.append("  Predicted observation sequence (horizon, robot qp only):")
                lines.append(
                    "    "
                    + _fmt_table_header(["h"] + ROBOT_JOINT_NAMES),
                )
                for h in range(obs_pred.shape[1]):
                    pred_qp, _, _ = _split_obs_60(obs_pred[ctrl, h])
                    lines.append(
                        "    "
                        + _fmt_table_row(
                            str(h), pred_qp, ROBOT_JOINT_NAMES, label_width=3, precision=4
                        )
                    )
                lines.append("")

            for sub in range(self.n_action_steps):
                if env_step >= trajectory["n_env_steps"]:
                    break
                lines.extend([
                    "-" * 78,
                    f"  ENV STEP {env_step}  (control={ctrl}, sub={sub})",
                    "-" * 78,
                ])

                if qp is not None and qv is not None:
                    lines.append("  Robot joints:")
                    lines.append("    positions (qp):")
                    lines.extend(_fmt_labeled(qp[env_step], ROBOT_JOINT_NAMES))
                    lines.append("    velocities (qv):")
                    lines.extend(_fmt_labeled(qv[env_step], ROBOT_JOINT_NAMES))
                    if obj_qp is not None:
                        lines.append("    object positions (obj_qp):")
                        lines.extend(
                            _fmt_labeled(obj_qp[env_step], OBJ_QP_NAMES, cols=4)
                        )
                    if obj_qv is not None:
                        lines.append("    object velocities (obj_qv):")
                        lines.extend(
                            _fmt_labeled(obj_qv[env_step], OBJ_QP_NAMES, cols=4)
                        )
                    lines.append("")

                if executed_obs is not None:
                    rqp, objp, goal = _split_obs_60(executed_obs[env_step])
                    lines.append("  Observation (executed, decomposed):")
                    lines.append("    robot qp:")
                    lines.extend(_fmt_labeled(rqp, ROBOT_JOINT_NAMES))
                    lines.append("    object qp:")
                    lines.extend(_fmt_labeled(objp, OBJ_QP_NAMES, cols=4))
                    lines.append("    goal:")
                    lines.extend(_fmt_labeled(goal, GOAL_NAMES, cols=5))
                    lines.append("")

                if executed_action is not None:
                    lines.append("  Action executed:")
                    lines.extend(
                        _fmt_labeled(executed_action[env_step], ACTION_NAMES, cols=5)
                    )
                    lines.append("")

                if demo_action_at_step is not None and demo_obs_at_step is not None:
                    d_act = demo_action_at_step[env_step]
                    d_obs = demo_obs_at_step[env_step]
                    demo_qp, _, _ = _split_obs_60(d_obs)
                    lines.append("  Demo ground truth (aligned timestep):")
                    if not np.isnan(d_act[0]):
                        lines.append("    demo action:")
                        lines.extend(_fmt_labeled(d_act, ACTION_NAMES, cols=5))
                        lines.append("    demo obs robot qp:")
                        lines.extend(_fmt_labeled(demo_qp, ROBOT_JOINT_NAMES))
                        if action_error_l2 is not None and obs_error_l2 is not None:
                            lines.append(
                                f"    errors: action_l2={action_error_l2[env_step]:.6f}  "
                                f"obs_l2={obs_error_l2[env_step]:.6f}"
                            )
                    else:
                        lines.append("    (no demo GT — rollout exceeded demo length)")
                    lines.append("")

                env_step += 1

            lines.append("")

        txt_path.write_text("\n".join(lines))

    def _run_single_episode(
        self,
        policy: BaseLowdimPolicy,
        episode_idx: int,
        enable_render: bool,
        env: MultiStepWrapper,
    ) -> Dict[str, Any]:
        device = policy.device
        env, video_path = self._configure_env(env, episode_idx, enable_render)
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

        obs = env.reset()
        policy.reset()

        inference_latencies_ms: List[float] = []
        completion_order: List[str] = []
        task_durations_ms: Dict[str, float] = {}
        prev_completed: Set[str] = set()
        last_completion_env_step = 0
        past_action = None
        env_step = 0
        # Match rendered mp4: VideoRecorder uses int(round(fps)) and 1 frame/env step.
        video_fps = float(max(int(round(self.fps)), 1))
        ms_per_env_step = 1000.0 / video_fps

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
            steps_advanced = int(action.shape[0])
            env_step = prev_env_step + steps_advanced
            pbar.update(steps_advanced)

            if traj is not None:
                step_infos = env.get_infos()
                obs_dicts = step_infos.get("obs_dict", [])
                # Prefer MultiStepWrapper info length when available; otherwise
                # fall back to action chunk size (always advances env_step above).
                n_logged = max(0, len(obs_dicts) - prev_env_step)
                n_new = n_logged if n_logged > 0 else steps_advanced
                for i in range(n_new):
                    if i < n_logged:
                        od = obs_dicts[prev_env_step + i]
                        od_arrays = _obs_dict_to_arrays(od)
                        step_obs = np.asarray(
                            env.obs[prev_env_step + i], dtype=np.float32
                        )
                    else:
                        od_arrays = {
                            "qp": np.zeros(9, dtype=np.float32),
                            "qv": np.zeros(9, dtype=np.float32),
                            "obj_qp": np.zeros(21, dtype=np.float32),
                            "obj_qv": np.zeros(21, dtype=np.float32),
                        }
                        step_obs = np.zeros(60, dtype=np.float32)
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

            current_completed = _extract_completed_tasks(info)
            new_tasks = current_completed - prev_completed
            for task_name in ALL_TASKS:
                if task_name in new_tasks:
                    # Simulation/video time, not wall-clock (matches rendered mp4).
                    duration_ms = (env_step - last_completion_env_step) * ms_per_env_step
                    completion_order.append(task_name)
                    task_durations_ms[task_name] = float(duration_ms)
                    last_completion_env_step = env_step
            prev_completed = current_completed

        pbar.close()
        episode_duration_ms = float(env_step * ms_per_env_step)

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

    def run(self, policy: BaseLowdimPolicy) -> Dict[str, Any]:
        episode_records: List[Dict[str, Any]] = []
        all_inference_latencies_ms: List[float] = []
        all_episode_durations_ms: List[float] = []
        all_task_durations_ms: List[float] = []
        task_duration_by_name: Dict[str, List[float]] = {t: [] for t in ALL_TASKS}
        per_task_success: Dict[str, List[int]] = {t: [] for t in ALL_TASKS}
        all_7_success_flags: List[int] = []
        completion_positions: Dict[str, List[int]] = {t: [] for t in ALL_TASKS}

        env = self._create_env()
        #region agent log
        import json as _json, os as _os, time as _time
        open('/home/daffa/Documents/experiment/.cursor/debug-374ef1.log','a').write(_json.dumps({"sessionId":"374ef1","runId":"post-fix","hypothesisId":"C","location":"kitchen_lowdim_eval_runner.py:run","message":"shared env created","data":{"fd_count":len(_os.listdir('/proc/self/fd'))},"timestamp":int(_time.time()*1000)})+'\n')
        #endregion
        try:
            for episode_idx in range(self.n_episodes):
                enable_render = episode_idx < self.n_episodes_vis
                record = self._run_single_episode(
                    policy, episode_idx, enable_render, env=env
                )
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
        finally:
            try:
                _close_env(env)
            except Exception:
                pass

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
