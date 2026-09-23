"""Build per-frame context timelines for generated demonstrations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from eloggen.generation_runtime.context.actions import get_action_spec
from eloggen.generation_runtime.context.boundary import detect_action_boundaries


@dataclass
class FrameContext:
    task: str
    action: str
    stage_type: str
    substage_id: int
    segment: str
    generated_frame: int
    segment_frame: int
    phase_index: Optional[int] = None
    subtask_index: Optional[int] = None
    active_arm: Optional[str] = None
    active_object: Optional[str] = None
    passive_object: Optional[str] = None
    object_label_zh: Optional[str] = None
    object_label_en: Optional[str] = None
    target_object: Optional[str] = None
    target_label_zh: Optional[str] = None
    target_label_en: Optional[str] = None
    instruction_zh: Optional[str] = None
    instruction_en: Optional[str] = None
    source_demo_ind: Optional[int] = None
    boundary_source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def _as_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values)


def _stack_datagen_field(datagen_infos: Sequence[Any], field_name: str) -> Optional[np.ndarray]:
    values = [getattr(info, field_name, None) for info in datagen_infos]
    if not values or values[0] is None:
        return None
    return _as_array(values)


def _stack_object_pose(datagen_infos: Sequence[Any], object_name: Optional[str]) -> Optional[np.ndarray]:
    if object_name is None:
        return None
    poses = []
    for info in datagen_infos:
        object_poses = getattr(info, "object_poses", None)
        if object_poses is None or object_name not in object_poses:
            return None
        poses.append(object_poses[object_name])
    return _as_array(poses)


def _active_arm_from_object_ref(object_ref: Dict[str, Optional[str]]) -> str:
    if object_ref.get("arm_left") is not None:
        return "left"
    if object_ref.get("arm_right") is not None:
        return "right"
    return "left"


def _segment_name_for_stage(stage_type: str, action: str) -> str:
    if stage_type.startswith("approach"):
        return "arm_mp"
    if stage_type == "retract":
        return "retract"
    if action == "reset":
        return "reset"
    return "arm_replay"


def _mp_end_for_arm(mp_end_steps: Optional[Sequence[int]], arm: str, num_frames: int) -> int:
    if mp_end_steps is None:
        return 0
    arm_index = 0 if arm == "left" else 1
    if len(mp_end_steps) <= arm_index:
        return 0
    return max(0, min(int(mp_end_steps[arm_index]), max(0, num_frames - 1)))


def _build_signals(
    datagen_infos: Sequence[Any],
    actions: np.ndarray,
    arm: str,
    object_name: Optional[str],
    demo_key: str,
) -> Dict[str, Any]:
    eef_pose = _stack_datagen_field(datagen_infos, "eef_pose")
    gripper_action = _stack_datagen_field(datagen_infos, "gripper_action")
    object_pose = _stack_object_pose(datagen_infos, object_name)
    if eef_pose is None:
        raise ValueError("Cannot build context: generated datagen_infos do not contain eef_pose.")
    num_frames = int(eef_pose.shape[0])

    arm_index = 0 if arm == "left" else 1
    eef_slice = slice(0, 4) if arm == "left" else slice(4, 8)
    eef_pose_arm = eef_pose[:, eef_slice, :]
    eef_pos = eef_pose_arm[:, :3, 3]

    if object_pose is None:
        object_pos = np.zeros_like(eef_pos)
        eef_object_dist = np.full(num_frames, np.inf, dtype=np.float32)
        object_z_delta = np.zeros(num_frames, dtype=np.float32)
    else:
        object_pos = object_pose[:, :3, 3]
        eef_object_dist = np.linalg.norm(eef_pos - object_pos, axis=1)
        object_z_delta = object_pos[:, 2] - object_pos[0, 2]

    if gripper_action is None:
        gripper = np.zeros(num_frames, dtype=np.float32)
    else:
        gripper = gripper_action[:, arm_index]

    return {
        "demo_key": demo_key,
        "arm": arm,
        "object_name": object_name,
        "num_frames": num_frames,
        "eef_pose": eef_pose_arm,
        "eef_pos": eef_pos,
        "object_pose": object_pose,
        "object_pos": object_pos,
        "gripper_action": gripper,
        "actions": actions,
        "eef_object_dist": eef_object_dist,
        "object_z_delta": object_z_delta,
    }


def _stage_for_frame(stages: Sequence[Any], frame_index: int) -> Any:
    for stage in reversed(stages):
        if stage.start <= frame_index <= stage.end:
            return stage
    return stages[-1]


def build_generated_segment_contexts(
    task_name: str,
    datagen_infos: Sequence[Any],
    actions: np.ndarray,
    object_ref: Dict[str, Optional[str]],
    mp_end_steps: Optional[Sequence[int]],
    subtask_length: int,
    phase_index: Optional[int],
    subtask_index: Optional[int],
    task_spec: Optional[Dict[str, Any]] = None,
    source_demo_ind: Optional[int] = None,
    generated_start_frame: int = 0,
) -> List[Dict[str, Any]]:
    """Build context for one generated segment using generated signals and generated segment lengths."""
    task_spec = dict(task_spec or {})
    num_frames = int(actions.shape[0])
    if num_frames != int(subtask_length):
        raise ValueError(f"Context length mismatch: actions={num_frames}, subtask_length={subtask_length}")
    if len(datagen_infos) != num_frames:
        raise ValueError(f"Context length mismatch: datagen_infos={len(datagen_infos)}, actions={num_frames}")
    if num_frames == 0:
        return []

    action = task_spec.get("action") or "pick"
    get_action_spec(action)
    active_arm = task_spec.get("arm") or _active_arm_from_object_ref(object_ref)
    passive_object = task_spec.get("object_ref") or object_ref.get(f"arm_{active_arm}")
    active_object = task_spec.get("active_object")
    target_object = task_spec.get("target_ref") or task_spec.get("target_object")
    signal_object = active_object if action in {"place", "insert"} and active_object is not None else passive_object
    detector_config = {
        "manual_subtask": {
            "MP_end_step": _mp_end_for_arm(mp_end_steps, active_arm, num_frames),
            "subtask_term_step": max(0, num_frames - 1),
        }
    }
    if active_object is not None:
        detector_config["active_object"] = active_object
    if target_object is not None:
        detector_config["target_object"] = target_object
    if action in {"place", "insert"} and passive_object is not None:
        detector_config["reference_object"] = passive_object
    for key in (
        "object_label_zh",
        "object_label_en",
        "target_ref",
        "target_label_zh",
        "target_label_en",
    ):
        if task_spec.get(key) is not None:
            target_key = "target_object" if key == "target_ref" else key
            detector_config[target_key] = task_spec[key]

    signals = _build_signals(
        datagen_infos=datagen_infos,
        actions=actions,
        arm=active_arm,
        object_name=signal_object,
        demo_key="generated_segment",
    )
    boundaries = detect_action_boundaries(signals, action=action, config=detector_config)
    boundaries.task = task_name

    frame_contexts = []
    for local_frame in range(num_frames):
        stage = _stage_for_frame(boundaries.stages, local_frame)
        metadata = dict(stage.metadata or {})
        metadata.update(
            {
                "boundary_set_demo_key": boundaries.demo_key,
                "detector": boundaries.detector,
                "segment_num_frames": num_frames,
            }
        )
        frame_contexts.append(
            FrameContext(
                task=task_name,
                action=stage.action,
                stage_type=stage.stage_type,
                substage_id=stage.substage_id,
                segment=_segment_name_for_stage(stage.stage_type, stage.action),
                generated_frame=generated_start_frame + local_frame,
                segment_frame=local_frame,
                phase_index=phase_index,
                subtask_index=subtask_index,
                active_arm=stage.active_arm,
                active_object=stage.active_object,
                passive_object=stage.passive_object,
                object_label_zh=stage.object_label_zh,
                object_label_en=stage.object_label_en,
                target_object=stage.target_object,
                target_label_zh=stage.target_label_zh,
                target_label_en=stage.target_label_en,
                instruction_zh=stage.instruction_zh,
                instruction_en=stage.instruction_en,
                source_demo_ind=source_demo_ind,
                boundary_source=metadata.get("boundary_source", boundaries.detector),
                metadata=metadata,
            ).to_dict()
        )
    return frame_contexts


def frame_contexts_to_json(frame_contexts: Iterable[Dict[str, Any]]) -> List[str]:
    return [json.dumps(context, ensure_ascii=False, sort_keys=True) for context in frame_contexts]
