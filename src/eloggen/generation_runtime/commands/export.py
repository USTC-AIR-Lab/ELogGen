"""
Convert a ElogGen generation runtime / robomimic-style demo.hdf5 file into a LeRobot-like dataset folder.

Example:

source scripts/activate_env.sh
python -m eloggen.generation_runtime.commands.export \
    --input runs/openarm_real_exp_1/generated_hdf5/demo.hdf5 \
    --output /tmp/openarm_lerobot_demo \
    --repo-id openarm/openarm_real_exp_1 \
    --task openarm_real_exp_1 \
    --robot-type openarm
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import imageio.v2 as imageio
import numpy as np


def _ensure_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        return pa, pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is an ElogGen runtime dependency; reinstall from "
            "environments/requirements-runtime.txt"
        ) from exc


PA, PQ = _ensure_pyarrow()


DEFAULT_CAMERA_MAP = {
    "observation.images.front": "external::base_cam::rgb",
    "observation.images.left": "external::left_wrist_cam::rgb",
    "observation.images.right": "external::right_wrist_cam::rgb",
}

DEFAULT_ORDERED_TASK_PROMPTS = {
    ("object_1", "object_2"): (
        "First pick the apple with the right arm and put it into the paper bag, "
        "then pick the lemon with the left arm and put it into the paper bag."
    ),
    ("object_2", "object_1"): (
        "First pick the lemon with the left arm and put it into the paper bag, "
        "then pick the apple with the right arm and put it into the paper bag."
    ),
}

EXAMPLES_V21_OBJECT_TASK_PROMPTS = {
    "object_1": "Pick the apple with the right arm and put it into the paper bag.",
    "object_2": "Pick the lemon with the left arm and put it into the paper bag.",
}

ORDER_NAME_BY_GRASP_ORDER = {
    ("object_1", "object_2"): "apple_first",
    ("object_2", "object_1"): "lemon_first",
}

STATE_NAMES = [
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_joint_7.pos",
    "left_gripper.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_joint_7.pos",
    "right_gripper.pos",
]

GRIPPER_ACTION_INDICES = (7, 15)
SIM_GRIPPER_OPEN = 1.0
SIM_GRIPPER_CLOSE = -1.0
REAL_GRIPPER_OPEN = -1.0
REAL_GRIPPER_CLOSE = 0.02
SIM_GRIPPER_STATE_OPEN = 0.044
SIM_GRIPPER_STATE_CLOSE = 0.02


def _sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    def sort_key(name: str) -> Tuple[int, str]:
        if name.startswith("demo_"):
            try:
                return (int(name.split("_")[-1]), name)
            except ValueError:
                pass
        return (10**9, name)

    return sorted(list(data_group.keys()), key=sort_key)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=4, ensure_ascii=False)
        file_obj.write("\n")


def _jsonl_dump(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False))
            file_obj.write("\n")


def _jsonl_load(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_index_from_path(path: Path) -> Optional[int]:
    stem = path.stem
    if not stem.startswith("episode_"):
        return None
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return None


def _read_context_task(demo_group: h5py.Group) -> Optional[str]:
    if "context" not in demo_group or "frame_context_json" not in demo_group["context"]:
        return None
    raw = demo_group["context"]["frame_context_json"][0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(raw).get("task")
    except Exception:
        return None


def _read_frame_contexts(demo_group: h5py.Group) -> Tuple[List[str], List[str]]:
    num_frames = int(demo_group.attrs["num_samples"])
    if "context" not in demo_group or "frame_context_json" not in demo_group["context"]:
        return (["{}"] * num_frames, ["unknown"] * num_frames)

    raw_rows = demo_group["context"]["frame_context_json"][:]
    frame_context_json = []
    stage_types = []
    for raw in raw_rows:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        frame_context_json.append(str(raw))
        try:
            stage_types.append(json.loads(raw).get("stage_type", "unknown"))
        except Exception:
            stage_types.append("unknown")
    return frame_context_json, stage_types


def _read_state_vector(demo_group: h5py.Group) -> np.ndarray:
    state = _read_raw_state_vector(demo_group)
    return _convert_gripper_state_sim_to_real(state)


def _read_raw_state_vector(demo_group: h5py.Group) -> np.ndarray:
    left_arm = np.asarray(demo_group["obs"]["arm_left_qpos"][:], dtype=np.float32)
    left_gripper = np.asarray(demo_group["obs"]["gripper_left_qpos"][:], dtype=np.float32)
    right_arm = np.asarray(demo_group["obs"]["arm_right_qpos"][:], dtype=np.float32)
    right_gripper = np.asarray(demo_group["obs"]["gripper_right_qpos"][:], dtype=np.float32)
    return np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1).astype(np.float32)


def _read_actions(demo_group: h5py.Group) -> np.ndarray:
    actions = _read_raw_actions(demo_group)
    return _convert_gripper_actions_sim_to_real(actions)


def _read_raw_actions(demo_group: h5py.Group) -> np.ndarray:
    actions = np.asarray(demo_group["actions"][:], dtype=np.float32)
    if actions.shape[1] != len(STATE_NAMES):
        raise ValueError(f"Expected action dim {len(STATE_NAMES)}, got {actions.shape[1]}")
    return actions.astype(np.float32)


def _convert_gripper_actions_sim_to_real(actions: np.ndarray) -> np.ndarray:
    """Convert sim gripper action convention to real OpenArm convention.

    Older sim demos use +1 for open and -1 for close. Newer OpenArm demos may
    store physical gripper joint-position targets in meters, matching sim
    state widths. The real robot convention uses -1.0 for open and 0.02 for
    close.
    """

    converted = np.asarray(actions, dtype=np.float32).copy()
    gripper_actions = converted[:, GRIPPER_ACTION_INDICES]
    finite_gripper_actions = gripper_actions[np.isfinite(gripper_actions)]
    if (
        finite_gripper_actions.size > 0
        and float(np.min(finite_gripper_actions)) >= -0.05
        and float(np.max(finite_gripper_actions)) <= 0.08
    ):
        return _convert_gripper_widths_sim_to_real(converted)

    scale = (REAL_GRIPPER_OPEN - REAL_GRIPPER_CLOSE) / (SIM_GRIPPER_OPEN - SIM_GRIPPER_CLOSE)
    offset = REAL_GRIPPER_OPEN - scale * SIM_GRIPPER_OPEN
    converted[:, GRIPPER_ACTION_INDICES] = converted[:, GRIPPER_ACTION_INDICES] * scale + offset
    return converted.astype(np.float32)


def _convert_gripper_widths_sim_to_real(values: np.ndarray) -> np.ndarray:
    converted = np.asarray(values, dtype=np.float32).copy()
    gripper_widths = np.clip(
        converted[:, GRIPPER_ACTION_INDICES],
        SIM_GRIPPER_STATE_CLOSE,
        SIM_GRIPPER_STATE_OPEN,
    )
    alpha = (gripper_widths - SIM_GRIPPER_STATE_CLOSE) / (SIM_GRIPPER_STATE_OPEN - SIM_GRIPPER_STATE_CLOSE)
    converted[:, GRIPPER_ACTION_INDICES] = REAL_GRIPPER_CLOSE + alpha * (
        REAL_GRIPPER_OPEN - REAL_GRIPPER_CLOSE
    )
    return converted.astype(np.float32)


def _convert_gripper_state_sim_to_real(state: np.ndarray) -> np.ndarray:
    """Convert sim gripper width observations to real OpenArm CAN convention.

    Sim state stores gripper width: larger is open, smaller is closed. The real
    robot convention stores motor position: -1.0 is open and 0.02 is closed.
    """

    return _convert_gripper_widths_sim_to_real(state)


def _read_video_frames(demo_group: h5py.Group, obs_key: str) -> np.ndarray:
    frames = np.asarray(demo_group["obs"][obs_key][:], dtype=np.uint8)
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D video tensor at {obs_key}, got shape {frames.shape}")
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    return frames


def _stack_obs_values(observations: List[Dict[str, Any]], key: str) -> np.ndarray:
    if not observations:
        raise ValueError("Cannot stack observations from an empty episode")
    if key not in observations[0]:
        raise KeyError(f"Observation key '{key}' not found in generated trajectory")
    return np.asarray([obs[key] for obs in observations])


def _raw_state_vector_from_observations(observations: List[Dict[str, Any]]) -> np.ndarray:
    left_arm = np.asarray(_stack_obs_values(observations, "arm_left_qpos"), dtype=np.float32)
    left_gripper = np.asarray(_stack_obs_values(observations, "gripper_left_qpos"), dtype=np.float32)
    right_arm = np.asarray(_stack_obs_values(observations, "arm_right_qpos"), dtype=np.float32)
    right_gripper = np.asarray(_stack_obs_values(observations, "gripper_right_qpos"), dtype=np.float32)
    return np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1).astype(np.float32)


def _video_frames_from_observations(observations: List[Dict[str, Any]], obs_key: str) -> np.ndarray:
    frames = np.asarray(_stack_obs_values(observations, obs_key), dtype=np.uint8)
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D video tensor at {obs_key}, got shape {frames.shape}")
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    return frames


def _write_video(path: Path, frames: np.ndarray, fps: int) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    codec = None
    for candidate in ("libx264", "libsvtav1"):
        writer = imageio.get_writer(
            path,
            fps=fps,
            codec=candidate,
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_log_level="error",
        )
        try:
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            codec = "h264" if candidate == "libx264" else "av1"
            break
        except Exception:
            try:
                writer.close()
            except Exception:
                pass
            if path.exists():
                path.unlink()
            continue

    if codec is None:
        raise RuntimeError(f"Failed to encode video at {path} with supported codecs")

    return {
        "video.height": height,
        "video.width": width,
        "video.codec": codec,
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }


def _fixed_size_list_array(values: np.ndarray, list_size: int):
    flat = PA.array(values.reshape(-1), type=PA.float32())
    return PA.FixedSizeListArray.from_arrays(flat, list_size)


def _feature_spec(dtype: str, shape: List[int], names: Optional[List[str]] = None, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"dtype": dtype, "shape": shape, "names": names}
    if info is not None:
        payload["info"] = info
    return payload


def _make_hf_schema_metadata(action_dim: int, state_dim: int) -> Dict[bytes, bytes]:
    """Return HuggingFace parquet metadata compatible with LeRobot 0.3.x.

    Video features live in meta/info.json, not in the parquet files. Rich
    per-frame semantic context is stored under meta/semantic_context/ so the
    standard LeRobot loader can ignore it unless a model explicitly opts in.
    """

    metadata = {
        "info": {
            "features": {
                "action": {
                    "feature": {"dtype": "float32", "_type": "Value"},
                    "length": action_dim,
                    "_type": "Sequence",
                },
                "observation.state": {
                    "feature": {"dtype": "float32", "_type": "Value"},
                    "length": state_dim,
                    "_type": "Sequence",
                },
                "timestamp": {"dtype": "float32", "_type": "Value"},
                "frame_index": {"dtype": "int64", "_type": "Value"},
                "episode_index": {"dtype": "int64", "_type": "Value"},
                "index": {"dtype": "int64", "_type": "Value"},
                "task_index": {"dtype": "int64", "_type": "Value"},
            }
        }
    }
    return {b"huggingface": json.dumps(metadata, ensure_ascii=False).encode("utf-8")}


def _compute_numeric_stats(values: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min": arr.min(axis=0).astype(np.float64).tolist(),
        "max": arr.max(axis=0).astype(np.float64).tolist(),
        "mean": arr.mean(axis=0).astype(np.float64).tolist(),
        "std": arr.std(axis=0).astype(np.float64).tolist(),
        "count": [int(arr.shape[0])],
    }


def _compute_image_stats(frames: np.ndarray) -> Dict[str, Any]:
    rgb = frames.astype(np.float32) / 255.0
    channel_first = np.moveaxis(rgb, -1, 0)
    mins = channel_first.min(axis=(1, 2, 3)).reshape(3, 1, 1).astype(np.float64).tolist()
    maxs = channel_first.max(axis=(1, 2, 3)).reshape(3, 1, 1).astype(np.float64).tolist()
    means = channel_first.mean(axis=(1, 2, 3)).reshape(3, 1, 1).astype(np.float64).tolist()
    stds = channel_first.std(axis=(1, 2, 3)).reshape(3, 1, 1).astype(np.float64).tolist()
    return {
        "min": mins,
        "max": maxs,
        "mean": means,
        "std": stds,
        "count": [int(rgb.shape[0])],
    }


def _read_video_stats_and_info(path: Path, fps: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    reader = imageio.get_reader(path)
    mins = np.full(3, np.inf, dtype=np.float64)
    maxs = np.full(3, -np.inf, dtype=np.float64)
    sums = np.zeros(3, dtype=np.float64)
    sq_sums = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    frame_count = 0
    height = None
    width = None
    try:
        for frame in reader:
            rgb = np.asarray(frame, dtype=np.float32)
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[:, :, None], 3, axis=2)
            if rgb.shape[-1] > 3:
                rgb = rgb[..., :3]
            if height is None:
                height, width = int(rgb.shape[0]), int(rgb.shape[1])
            flat = (rgb.reshape(-1, 3) / 255.0).astype(np.float64)
            mins = np.minimum(mins, flat.min(axis=0))
            maxs = np.maximum(maxs, flat.max(axis=0))
            sums += flat.sum(axis=0)
            sq_sums += np.square(flat).sum(axis=0)
            pixel_count += int(flat.shape[0])
            frame_count += 1
    finally:
        reader.close()

    if frame_count == 0 or height is None or width is None:
        raise ValueError(f"Cannot read video frames from {path}")

    means = sums / float(pixel_count)
    variances = np.maximum((sq_sums / float(pixel_count)) - np.square(means), 0.0)
    stds = np.sqrt(variances)
    stats = {
        "min": mins.reshape(3, 1, 1).tolist(),
        "max": maxs.reshape(3, 1, 1).tolist(),
        "mean": means.reshape(3, 1, 1).tolist(),
        "std": stds.reshape(3, 1, 1).tolist(),
        "count": [int(frame_count)],
    }
    info = {
        "video.height": int(height),
        "video.width": int(width),
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": int(fps),
        "video.channels": 3,
        "has_audio": False,
    }
    return stats, info


def _semantic_context_rows(
    frame_context_json: List[str],
    stage_types: List[str],
    episode_index: int,
    timestamps: np.ndarray,
) -> List[Dict[str, Any]]:
    rows = []
    for frame_idx, raw in enumerate(frame_context_json):
        try:
            row = json.loads(raw)
            if not isinstance(row, dict):
                row = {"frame_context_json": raw}
        except Exception:
            row = {"frame_context_json": raw}
        row.setdefault("stage_type", stage_types[frame_idx] if frame_idx < len(stage_types) else "unknown")
        row["episode_index"] = int(episode_index)
        row["frame_index"] = int(frame_idx)
        row["timestamp"] = float(timestamps[frame_idx])
        rows.append(row)
    return rows


def _context_json_strings(frame_contexts: Optional[List[Any]], num_frames: int) -> Tuple[List[str], List[str]]:
    if frame_contexts is None:
        return (["{}"] * num_frames, ["unknown"] * num_frames)
    if len(frame_contexts) != num_frames:
        raise ValueError(f"Context length mismatch: frame_contexts={len(frame_contexts)}, num_frames={num_frames}")
    frame_context_json = []
    stage_types = []
    for context in frame_contexts:
        if isinstance(context, bytes):
            context = context.decode("utf-8")
        if isinstance(context, str):
            raw = context
            try:
                context_dict = json.loads(raw)
            except Exception:
                context_dict = {}
        elif isinstance(context, dict):
            context_dict = context
            raw = json.dumps(context, ensure_ascii=False)
        else:
            context_dict = {}
            raw = json.dumps(context, ensure_ascii=False)
        frame_context_json.append(raw)
        stage_types.append(context_dict.get("stage_type", "unknown"))
    return frame_context_json, stage_types


def _episode_grasp_order(
    frame_context_json: List[str],
    episode_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[str, ...]]:
    if episode_metadata and episode_metadata.get("grasp_order"):
        return tuple(str(item) for item in episode_metadata["grasp_order"])
    if not frame_context_json:
        return None
    try:
        first_context = json.loads(frame_context_json[0])
    except Exception:
        return None
    grasp_order = first_context.get("grasp_order")
    if grasp_order:
        return tuple(str(item) for item in grasp_order)
    metadata = first_context.get("metadata")
    if isinstance(metadata, dict) and metadata.get("grasp_order"):
        return tuple(str(item) for item in metadata["grasp_order"])
    return None


def _read_episode_metadata(demo_group: h5py.Group) -> Optional[Dict[str, Any]]:
    raw = demo_group.attrs.get("metadata_json")
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        metadata = json.loads(str(raw))
    except Exception:
        return None
    return metadata if isinstance(metadata, dict) else None


class LeRobotDatasetWriter:
    """Incrementally write a LeRobot v2.1-style dataset.

    The parquet table intentionally contains only fields consumed by the
    standard LeRobot 0.3.x loader. Rich per-frame semantics are written as an
    optional sidecar under meta/semantic_context/.
    """

    def __init__(
        self,
        output_path: str,
        repo_id: str,
        task_name: Optional[str],
        robot_type: str = "openarm",
        fps: int = 30,
        chunk_size: int = 1000,
        camera_map: Optional[Dict[str, str]] = None,
        ordered_tasks: bool = False,
        ordered_task_prompts: Optional[Dict[Tuple[str, ...], str]] = None,
        overwrite: bool = False,
        resume: bool = False,
    ) -> None:
        self.output_root = Path(os.path.abspath(os.path.expanduser(output_path)))
        if self.output_root.exists() and overwrite:
            shutil.rmtree(self.output_root)
        elif self.output_root.exists() and any(self.output_root.iterdir()) and not resume:
            raise FileExistsError(f"LeRobot output directory already exists: {self.output_root}")

        self.repo_id = repo_id
        self.default_task_name = task_name or Path(output_path).name
        self.robot_type = robot_type
        self.fps = fps
        self.chunk_size = chunk_size
        self.camera_map = dict(camera_map or DEFAULT_CAMERA_MAP)
        self.ordered_tasks = ordered_tasks
        self.ordered_task_prompts = {
            tuple(key): value
            for key, value in (ordered_task_prompts or DEFAULT_ORDERED_TASK_PROMPTS).items()
        }

        self.data_root = self.output_root / "data"
        self.video_root = self.output_root / "videos"
        self.meta_root = self.output_root / "meta"
        self.semantic_root = self.meta_root / "semantic_context"
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.video_root.mkdir(parents=True, exist_ok=True)
        self.meta_root.mkdir(parents=True, exist_ok=True)
        self.semantic_root.mkdir(parents=True, exist_ok=True)

        self.episodes_meta: List[Dict[str, Any]] = []
        self.episodes_stats: List[Dict[str, Any]] = []
        self.video_feature_info: Dict[str, Dict[str, Any]] = {}
        self.tasks_meta: List[Dict[str, Any]] = []
        self.task_to_index: Dict[str, int] = {}
        self.task_episode_counts: Dict[str, int] = {}
        self.total_frames = 0
        self.total_videos = 0
        self.global_index_offset = 0
        self.global_stage_types = set()
        self.resumed_episodes = 0

        if resume:
            self._load_existing_state()

    def _data_path(self, episode_index: int) -> Path:
        return self.data_root / f"chunk-{episode_index // self.chunk_size:03d}" / f"episode_{episode_index:06d}.parquet"

    def _video_path(self, episode_index: int, video_key: str) -> Path:
        return (
            self.video_root
            / f"chunk-{episode_index // self.chunk_size:03d}"
            / video_key
            / f"episode_{episode_index:06d}.mp4"
        )

    def _semantic_path(self, episode_index: int) -> Path:
        return self.semantic_root / f"episode_{episode_index:06d}.jsonl"

    def _task_name_for_episode(
        self,
        frame_context_json: List[str],
        episode_metadata: Optional[Dict[str, Any]],
    ) -> str:
        if self.ordered_tasks:
            grasp_order = _episode_grasp_order(frame_context_json, episode_metadata)
            if grasp_order is None:
                raise ValueError("Cannot assign ordered LeRobot task because grasp_order is missing")
            task = self.ordered_task_prompts.get(grasp_order)
            if task is None:
                task = "Follow grasp order: {}.".format(", ".join(grasp_order))
            return task
        return self.default_task_name

    def _ensure_task_index(self, task: str) -> int:
        if task not in self.task_to_index:
            task_index = len(self.tasks_meta)
            self.task_to_index[task] = task_index
            self.tasks_meta.append({"task_index": task_index, "task": task})
        return self.task_to_index[task]

    def _parse_frame_context_dicts(self, frame_context_json: List[str]) -> List[Dict[str, Any]]:
        contexts = []
        for raw in frame_context_json:
            try:
                row = json.loads(raw)
            except Exception:
                row = {}
            contexts.append(row if isinstance(row, dict) else {})
        return contexts

    def _object_task_prompt(self, object_name: str) -> str:
        return EXAMPLES_V21_OBJECT_TASK_PROMPTS.get(object_name, f"Pick {object_name}.")

    def _resolve_examples_v21_tasks(
        self,
        frame_context_json: List[str],
        episode_metadata: Optional[Dict[str, Any]],
        num_frames: int,
    ) -> Tuple[np.ndarray, List[str]]:
        grasp_order = _episode_grasp_order(frame_context_json, episode_metadata)
        if grasp_order is None:
            raise ValueError("Cannot assign examples_v2.1 LeRobot tasks because grasp_order is missing")
        episode_tasks = [self._object_task_prompt(object_name) for object_name in grasp_order]
        object_to_task_index = {
            object_name: self._ensure_task_index(task)
            for object_name, task in zip(grasp_order, episode_tasks)
        }

        contexts = self._parse_frame_context_dicts(frame_context_json)
        fallback_object = grasp_order[0] if grasp_order else None
        last_object = fallback_object
        task_index_values = []
        for context in contexts:
            active_object = context.get("active_object")
            if active_object not in object_to_task_index:
                active_object = last_object
            if active_object not in object_to_task_index:
                active_object = fallback_object
            last_object = active_object
            task_index_values.append(object_to_task_index[active_object])

        if len(task_index_values) < num_frames:
            fill = object_to_task_index[fallback_object]
            task_index_values.extend([fill] * (num_frames - len(task_index_values)))
        task_index_col = np.asarray(task_index_values[:num_frames], dtype=np.int64)

        order_name = ORDER_NAME_BY_GRASP_ORDER.get(tuple(grasp_order), ",".join(grasp_order))
        self.task_episode_counts[order_name] = self.task_episode_counts.get(order_name, 0) + 1
        return task_index_col, episode_tasks

    def _resolve_task(self, frame_context_json: List[str], episode_metadata: Optional[Dict[str, Any]]) -> Tuple[int, str]:
        task = self._task_name_for_episode(frame_context_json, episode_metadata)
        task_index = self._ensure_task_index(task)
        self.task_episode_counts[str(task_index)] = self.task_episode_counts.get(str(task_index), 0) + 1
        return task_index, task

    def _existing_episode_indices(self) -> List[int]:
        paths = []
        paths.extend((self.data_root).glob("chunk-*/episode_*.parquet"))
        paths.extend((self.semantic_root).glob("episode_*.jsonl"))
        paths.extend((self.video_root).glob("chunk-*/*/episode_*.mp4"))
        indices = {_episode_index_from_path(path) for path in paths}
        return sorted(index for index in indices if index is not None)

    def _complete_episode_count(self) -> int:
        count = 0
        while True:
            if not self._data_path(count).exists():
                break
            if not self._semantic_path(count).exists():
                break
            if any(not self._video_path(count, key).exists() for key in self.camera_map):
                break
            count += 1
        return count

    def _remove_trailing_episode_files(self, start_index: int) -> None:
        for episode_index in self._existing_episode_indices():
            if episode_index < start_index:
                continue
            data_path = self._data_path(episode_index)
            if data_path.exists():
                data_path.unlink()
            semantic_path = self._semantic_path(episode_index)
            if semantic_path.exists():
                semantic_path.unlink()
            for video_path in self.video_root.glob(f"chunk-*/*/episode_{episode_index:06d}.mp4"):
                video_path.unlink()

    def _load_existing_state(self) -> None:
        info_path = self.meta_root / "info.json"
        episodes_path = self.meta_root / "episodes.jsonl"
        stats_path = self.meta_root / "episodes_stats.jsonl"
        tasks_path = self.meta_root / "tasks.jsonl"
        if all(path.exists() for path in (info_path, episodes_path, stats_path, tasks_path)):
            self._load_finalized_state(info_path, episodes_path, stats_path, tasks_path)
        else:
            self._rebuild_state_from_files()
        self.resumed_episodes = len(self.episodes_meta)

    def _load_finalized_state(
        self,
        info_path: Path,
        episodes_path: Path,
        stats_path: Path,
        tasks_path: Path,
    ) -> None:
        with open(info_path, "r", encoding="utf-8") as file_obj:
            info = json.load(file_obj)
        self.episodes_meta = _jsonl_load(episodes_path)
        self.episodes_stats = _jsonl_load(stats_path)
        self.tasks_meta = _jsonl_load(tasks_path)
        self.task_to_index = {str(row["task"]): int(row["task_index"]) for row in self.tasks_meta}
        self.task_episode_counts = {
            str(key): int(value)
            for key, value in info.get("task_episode_counts", {}).items()
        }
        if not self.task_episode_counts:
            for episode in self.episodes_meta:
                task = str(episode["tasks"][0])
                task_index = self.task_to_index[task]
                self.task_episode_counts[str(task_index)] = self.task_episode_counts.get(str(task_index), 0) + 1

        self.total_frames = int(info.get("total_frames", 0))
        self.total_videos = int(info.get("total_videos", 0))
        self.global_index_offset = self.total_frames
        self.global_stage_types = set(info.get("stage_type_vocab", []))
        self.video_feature_info = {
            key: feature.get("info", {})
            for key, feature in info.get("features", {}).items()
            if isinstance(feature, dict) and feature.get("dtype") == "video"
        }

    def _read_first_semantic_context_json(self, episode_index: int) -> List[str]:
        semantic_path = self._semantic_path(episode_index)
        with open(semantic_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if line:
                    return [line]
        return ["{}"]

    def _read_semantic_context_json(self, episode_index: int) -> List[str]:
        semantic_path = self._semantic_path(episode_index)
        rows = []
        with open(semantic_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if line:
                    rows.append(line)
        return rows or ["{}"]

    def _rebuild_state_from_files(self) -> None:
        complete_count = self._complete_episode_count()
        self._remove_trailing_episode_files(complete_count)
        task_by_index: Dict[int, str] = {}
        max_global_index = -1

        for episode_index in range(complete_count):
            table = PQ.read_table(self._data_path(episode_index))
            num_frames = int(table.num_rows)
            task_values = table.column("task_index").to_pylist()
            if self.ordered_tasks:
                frame_context_json = self._read_semantic_context_json(episode_index)
                expected_task_index_col, episode_tasks = self._resolve_examples_v21_tasks(
                    frame_context_json,
                    None,
                    num_frames,
                )
                if len(task_values) == len(expected_task_index_col) and any(
                    int(value) != int(expected)
                    for value, expected in zip(task_values, expected_task_index_col)
                ):
                    raise ValueError(f"task_index does not match semantic_context in episode {episode_index:06d}")
            else:
                task_index = int(task_values[0]) if task_values else 0
                if any(int(value) != task_index for value in task_values):
                    raise ValueError(f"Mixed task_index values in episode {episode_index:06d}")
                frame_context_json = self._read_first_semantic_context_json(episode_index)
                task = self._task_name_for_episode(frame_context_json, None)
                if task_index in task_by_index and task_by_index[task_index] != task:
                    raise ValueError(
                        f"Conflicting task text for task_index={task_index}: "
                        f"{task_by_index[task_index]!r} vs {task!r}"
                    )
                task_by_index[task_index] = task
                self.task_episode_counts[str(task_index)] = self.task_episode_counts.get(str(task_index), 0) + 1
                episode_tasks = [task]
            self.episodes_meta.append(
                {
                    "episode_index": episode_index,
                    "tasks": episode_tasks,
                    "length": num_frames,
                }
            )

            actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
            state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
            timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float32)
            frame_index = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
            episode_index_col = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
            global_index = np.asarray(table.column("index").to_pylist(), dtype=np.int64)
            task_index_col = np.asarray(task_values, dtype=np.int64)
            if global_index.size:
                max_global_index = max(max_global_index, int(global_index.max()))

            episode_stats = {
                "episode_index": episode_index,
                "stats": {
                    "action": _compute_numeric_stats(actions),
                    "observation.state": _compute_numeric_stats(state),
                    "timestamp": _compute_numeric_stats(timestamps),
                    "frame_index": _compute_numeric_stats(frame_index),
                    "episode_index": _compute_numeric_stats(episode_index_col),
                    "index": _compute_numeric_stats(global_index),
                    "task_index": _compute_numeric_stats(task_index_col),
                },
            }
            for video_key in self.camera_map:
                stats, info = _read_video_stats_and_info(self._video_path(episode_index, video_key), self.fps)
                episode_stats["stats"][video_key] = stats
                self.video_feature_info.setdefault(video_key, info)
            self.episodes_stats.append(episode_stats)

            with open(self._semantic_path(episode_index), "r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    self.global_stage_types.add(str(row.get("stage_type", "unknown")))

            self.total_frames += num_frames
            self.total_videos += len(self.camera_map)

        if not self.ordered_tasks:
            self.tasks_meta = [
                {"task_index": task_index, "task": task_by_index[task_index]}
                for task_index in sorted(task_by_index)
            ]
            self.task_to_index = {row["task"]: int(row["task_index"]) for row in self.tasks_meta}
        self.global_index_offset = max_global_index + 1 if max_global_index >= 0 else self.total_frames

    def write_episode(
        self,
        actions: np.ndarray,
        raw_state: np.ndarray,
        video_frames_by_key: Dict[str, np.ndarray],
        frame_contexts: Optional[List[Any]] = None,
        episode_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape[1] != len(STATE_NAMES):
            raise ValueError(f"Expected action dim {len(STATE_NAMES)}, got {actions.shape[1]}")
        actions = _convert_gripper_actions_sim_to_real(actions)
        state = _convert_gripper_state_sim_to_real(np.asarray(raw_state, dtype=np.float32))
        if state.shape[1] != len(STATE_NAMES):
            raise ValueError(f"Expected state dim {len(STATE_NAMES)}, got {state.shape[1]}")

        episode_index = len(self.episodes_meta)
        num_frames = int(actions.shape[0])
        if state.shape[0] != num_frames:
            raise ValueError(f"State/action length mismatch: state={state.shape[0]}, actions={num_frames}")
        frame_context_json, stage_types = _context_json_strings(frame_contexts, num_frames)
        self.global_stage_types.update(stage_types)

        timestamps = (np.arange(num_frames, dtype=np.float32) / float(self.fps)).astype(np.float32)
        frame_index = np.arange(num_frames, dtype=np.int64)
        global_index = np.arange(self.global_index_offset, self.global_index_offset + num_frames, dtype=np.int64)
        episode_index_col = np.full(num_frames, episode_index, dtype=np.int64)
        if self.ordered_tasks:
            task_index_col, episode_tasks = self._resolve_examples_v21_tasks(
                frame_context_json,
                episode_metadata,
                num_frames,
            )
        else:
            task_index, task = self._resolve_task(frame_context_json, episode_metadata)
            task_index_col = np.full(num_frames, task_index, dtype=np.int64)
            episode_tasks = [task]

        for lerobot_key, frames in video_frames_by_key.items():
            frames = np.asarray(frames, dtype=np.uint8)
            if frames.ndim != 4:
                raise ValueError(f"Expected 4D video tensor for {lerobot_key}, got {frames.shape}")
            if frames.shape[-1] == 4:
                frames = frames[..., :3]
            if frames.shape[0] != num_frames:
                raise ValueError(f"Video/action length mismatch for {lerobot_key}: video={frames.shape[0]}, actions={num_frames}")
            chunk_dir = self.video_root / f"chunk-{episode_index // self.chunk_size:03d}" / lerobot_key
            video_path = chunk_dir / f"episode_{episode_index:06d}.mp4"
            info = _write_video(video_path, frames, fps=self.fps)
            self.video_feature_info[lerobot_key] = info
            self.total_videos += 1

        table = PA.table(
            {
                "action": _fixed_size_list_array(actions, actions.shape[1]),
                "observation.state": _fixed_size_list_array(state, state.shape[1]),
                "timestamp": PA.array(timestamps, type=PA.float32()),
                "frame_index": PA.array(frame_index, type=PA.int64()),
                "episode_index": PA.array(episode_index_col, type=PA.int64()),
                "index": PA.array(global_index, type=PA.int64()),
                "task_index": PA.array(task_index_col, type=PA.int64()),
            }
        )
        table = table.cast(
            PA.schema(
                [
                    ("action", PA.list_(PA.float32(), actions.shape[1])),
                    ("observation.state", PA.list_(PA.float32(), state.shape[1])),
                    ("timestamp", PA.float32()),
                    ("frame_index", PA.int64()),
                    ("episode_index", PA.int64()),
                    ("index", PA.int64()),
                    ("task_index", PA.int64()),
                ]
            )
        )
        table = table.replace_schema_metadata(
            _make_hf_schema_metadata(actions.shape[1], state.shape[1])
        )
        chunk_dir = self.data_root / f"chunk-{episode_index // self.chunk_size:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        PQ.write_table(table, chunk_dir / f"episode_{episode_index:06d}.parquet")
        _jsonl_dump(
            self.semantic_root / f"episode_{episode_index:06d}.jsonl",
            _semantic_context_rows(frame_context_json, stage_types, episode_index, timestamps),
        )

        self.episodes_meta.append(
            {
                "episode_index": episode_index,
                "tasks": episode_tasks,
                "length": num_frames,
            }
        )
        episode_stats = {
            "episode_index": episode_index,
            "stats": {
                "action": _compute_numeric_stats(actions),
                "observation.state": _compute_numeric_stats(state),
                "timestamp": _compute_numeric_stats(timestamps),
                "frame_index": _compute_numeric_stats(frame_index),
                "episode_index": _compute_numeric_stats(episode_index_col),
                "index": _compute_numeric_stats(global_index),
                "task_index": _compute_numeric_stats(task_index_col),
            },
        }
        for lerobot_key, frames in video_frames_by_key.items():
            episode_stats["stats"][lerobot_key] = _compute_image_stats(frames[..., :3])
        self.episodes_stats.append(episode_stats)
        self.total_frames += num_frames
        self.global_index_offset += num_frames

    def write_generated_episode(self, generated_traj: Dict[str, Any]) -> None:
        observations = generated_traj.get("observations")
        if observations is None:
            raise ValueError("LeRobot output requires generated observations")
        raw_state = _raw_state_vector_from_observations(observations)
        video_frames_by_key = {
            lerobot_key: _video_frames_from_observations(observations, obs_key)
            for lerobot_key, obs_key in self.camera_map.items()
        }
        self.write_episode(
            actions=generated_traj["actions"],
            raw_state=raw_state,
            video_frames_by_key=video_frames_by_key,
            frame_contexts=generated_traj.get("frame_contexts"),
            episode_metadata=generated_traj.get("episode_metadata"),
        )

    def finalize(self) -> Dict[str, Any]:
        self._remove_trailing_episode_files(len(self.episodes_meta))
        num_chunks = max(1, (len(self.episodes_meta) + self.chunk_size - 1) // self.chunk_size)
        info_json = {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "total_episodes": len(self.episodes_meta),
            "total_frames": self.total_frames,
            "total_tasks": len(self.tasks_meta),
            "total_videos": self.total_videos,
            "total_chunks": num_chunks,
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self.episodes_meta)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "action": _feature_spec("float32", [len(STATE_NAMES)], STATE_NAMES),
                "observation.state": _feature_spec("float32", [len(STATE_NAMES)], STATE_NAMES),
                **{
                    key: _feature_spec(
                        "video",
                        [
                            int(info["video.height"]),
                            int(info["video.width"]),
                            int(info["video.channels"]),
                        ],
                        ["height", "width", "channels"],
                        info,
                    )
                    for key, info in self.video_feature_info.items()
                },
                "timestamp": _feature_spec("float32", [1], None),
                "frame_index": _feature_spec("int64", [1], None),
                "episode_index": _feature_spec("int64", [1], None),
                "index": _feature_spec("int64", [1], None),
                "task_index": _feature_spec("int64", [1], None),
            },
            "stage_type_vocab": sorted(self.global_stage_types),
            "semantic_context_path": "meta/semantic_context/episode_{episode_index:06d}.jsonl",
            "semantic_context_fields": [
                "stage_type",
                "active_arm",
                "active_object",
                "action",
                "instruction_en",
                "instruction_zh",
                "grasp_order",
                "phase_execution_order",
            ],
            "repo_id": self.repo_id,
        }
        if self.ordered_tasks:
            info_json["task_prompt_strategy"] = "episode_level_grasp_order_from_frame_context_json"
            info_json["task_prompt_mapping"] = {
                ",".join(key): value
                for key, value in DEFAULT_ORDERED_TASK_PROMPTS.items()
            }
            info_json["task_episode_counts"] = dict(self.task_episode_counts)

        _json_dump(self.meta_root / "info.json", info_json)
        _jsonl_dump(self.meta_root / "episodes.jsonl", self.episodes_meta)
        _jsonl_dump(self.meta_root / "episodes_stats.jsonl", self.episodes_stats)
        _jsonl_dump(self.meta_root / "tasks.jsonl", self.tasks_meta)
        return info_json


def _jsonl_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as file_obj:
        return sum(1 for line in file_obj if line.strip())


def validate_lerobot_dataset(
    output_path: str,
    expected_episodes: Optional[int] = None,
    expected_cameras: Optional[int] = None,
) -> Dict[str, Any]:
    output_root = Path(os.path.abspath(os.path.expanduser(output_path)))
    meta_root = output_root / "meta"
    info_path = meta_root / "info.json"
    tasks_path = meta_root / "tasks.jsonl"
    episodes_path = meta_root / "episodes.jsonl"
    stats_path = meta_root / "episodes_stats.jsonl"
    for path in (info_path, tasks_path, episodes_path, stats_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing LeRobot metadata file: {path}")

    with open(info_path, "r", encoding="utf-8") as file_obj:
        info = json.load(file_obj)

    total_episodes = int(info["total_episodes"])
    if expected_episodes is not None and total_episodes != int(expected_episodes):
        raise ValueError(f"Episode count mismatch: info={total_episodes}, expected={expected_episodes}")
    if _jsonl_count(episodes_path) != total_episodes:
        raise ValueError("episodes.jsonl row count does not match info.total_episodes")
    if _jsonl_count(stats_path) != total_episodes:
        raise ValueError("episodes_stats.jsonl row count does not match info.total_episodes")

    parquet_paths = sorted((output_root / "data").glob("chunk-*/episode_*.parquet"))
    semantic_paths = sorted((meta_root / "semantic_context").glob("episode_*.jsonl"))
    if len(parquet_paths) != total_episodes:
        raise ValueError(f"Parquet count mismatch: {len(parquet_paths)} != {total_episodes}")
    if len(semantic_paths) != total_episodes:
        raise ValueError(f"Semantic context count mismatch: {len(semantic_paths)} != {total_episodes}")

    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    camera_count = expected_cameras if expected_cameras is not None else len(video_keys)
    video_paths = sorted((output_root / "videos").glob("chunk-*/*/episode_*.mp4"))
    if len(video_paths) != total_episodes * camera_count:
        raise ValueError(
            f"Video count mismatch: {len(video_paths)} != {total_episodes} * {camera_count}"
        )

    expected_columns = {
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    if parquet_paths:
        schema_names = set(PQ.read_schema(parquet_paths[0]).names)
        if schema_names != expected_columns:
            raise ValueError(f"Unexpected parquet columns: {sorted(schema_names)}")

    summary = {
        "output": str(output_root),
        "episodes": total_episodes,
        "frames": int(info["total_frames"]),
        "videos": len(video_paths),
        "tasks": int(info["total_tasks"]),
        "semantic_context": len(semantic_paths),
    }
    print("LeRobot dataset complete:")
    print(f"  output: {summary['output']}")
    print(f"  episodes: {summary['episodes']}")
    print(f"  frames: {summary['frames']}")
    print(f"  videos: {summary['videos']}")
    print(f"  tasks: {summary['tasks']}")
    print(f"  semantic_context: {summary['semantic_context']} files")
    return summary


def _parse_camera_args(items: List[str]) -> Dict[str, str]:
    mapping = dict(DEFAULT_CAMERA_MAP)
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --camera value '{item}'. Expected lerobot_key=hdf5_obs_key.")
        key, value = item.split("=", 1)
        mapping[key] = value
    return mapping


def convert_dataset(
    input_path: str,
    output_path: str,
    repo_id: str,
    task_name: Optional[str],
    robot_type: str,
    fps: int,
    chunk_size: int,
    camera_map: Dict[str, str],
    ordered_tasks: bool = False,
    overwrite: bool = False,
) -> None:
    input_path = os.path.abspath(os.path.expanduser(input_path))
    with h5py.File(input_path, "r") as dataset:
        demo_keys = _sorted_demo_keys(dataset["data"])
        if not demo_keys:
            raise ValueError(f"No demos found in {input_path}")

        resolved_task_name = task_name or _read_context_task(dataset["data"][demo_keys[0]]) or Path(input_path).stem
        writer = LeRobotDatasetWriter(
            output_path=output_path,
            repo_id=repo_id,
            task_name=resolved_task_name,
            robot_type=robot_type,
            fps=fps,
            chunk_size=chunk_size,
            camera_map=camera_map,
            ordered_tasks=ordered_tasks,
            overwrite=overwrite,
        )

        for demo_key in demo_keys:
            demo = dataset["data"][demo_key]
            actions = _read_raw_actions(demo)
            state = _read_raw_state_vector(demo)
            frame_context_json, stage_types = _read_frame_contexts(demo)
            video_frames_by_key: Dict[str, np.ndarray] = {}
            for lerobot_key, obs_key in camera_map.items():
                video_frames_by_key[lerobot_key] = _read_video_frames(demo, obs_key)
            writer.write_episode(
                actions=actions,
                raw_state=state,
                video_frames_by_key=video_frames_by_key,
                frame_contexts=frame_context_json,
                episode_metadata=_read_episode_metadata(demo),
            )
        info_json = writer.finalize()
    validate_lerobot_dataset(
        output_path=output_path,
        expected_episodes=info_json["total_episodes"],
        expected_cameras=len(camera_map),
    )

    print(f"Converted {info_json['total_episodes']} episodes from {input_path}")
    print(f"Output written to: {Path(os.path.abspath(os.path.expanduser(output_path)))}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ElogGen generation runtime demo.hdf5 into a LeRobot-like dataset folder.")
    parser.add_argument("--input", required=True, help="Path to demo.hdf5 produced by eloggen pipeline")
    parser.add_argument("--output", required=True, help="Output dataset directory in LeRobot-like format")
    parser.add_argument("--repo-id", required=True, help="Logical repo id, e.g. openarm/openarm_real_exp_1")
    parser.add_argument("--task", default=None, help="Task name stored in tasks.jsonl; defaults to context task or input stem")
    parser.add_argument("--robot-type", default="openarm", help="Robot type written to meta/info.json")
    parser.add_argument("--fps", type=int, default=30, help="FPS for timestamp generation and MP4 export")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Episodes per chunk directory")
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Override camera mapping with lerobot_key=hdf5_obs_key, e.g. observation.images.front=external::base_cam::rgb",
    )
    parser.add_argument(
        "--ordered-tasks",
        action="store_true",
        help="Assign task prompts from semantic grasp_order instead of using a single task.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the LeRobot output directory before writing.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    camera_map = _parse_camera_args(args.camera)
    convert_dataset(
        input_path=args.input,
        output_path=args.output,
        repo_id=args.repo_id,
        task_name=args.task,
        robot_type=args.robot_type,
        fps=args.fps,
        chunk_size=args.chunk_size,
        camera_map=camera_map,
        ordered_tasks=args.ordered_tasks,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
