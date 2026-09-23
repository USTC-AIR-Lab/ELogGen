"""Replay-boundary validation and source-state extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .replay_annotation import ReplayAnnotation, ReplayBoundary


REPLAY_BOUNDARY_ROLES = {"replay_start", "replay_end"}


@dataclass
class BoundaryState:
    boundary_id: str
    frame_index: int
    demo_id: str
    object: Optional[str] = None
    target: Optional[str] = None
    tool: Optional[str] = None
    handle: Optional[str] = None
    effector: Optional[str] = None
    robot_joint_state: Optional[List[float]] = None
    active_effector_pose: Optional[List[float]] = None
    active_gripper_state: Optional[float] = None
    active_object_pose: Optional[List[float]] = None
    target_pose: Optional[List[float]] = None
    attached_object: Optional[str] = None
    source_action: Optional[List[float]] = None
    state_vector: Optional[List[float]] = None
    before_predicates: List[str] = field(default_factory=list)
    after_predicates: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_replay_annotation(annotation: ReplayAnnotation) -> List[str]:
    errors: List[str] = []
    grouped: Dict[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str], List[ReplayBoundary]] = {}
    for boundary in annotation.boundaries:
        if boundary.boundary not in REPLAY_BOUNDARY_ROLES and not boundary.boundary.endswith(("_start", "_end")):
            errors.append(f"{boundary.id}: unknown boundary role {boundary.boundary!r}")
        key = (boundary.skill, boundary.segment, boundary.object or boundary.tool or boundary.handle, boundary.effector, boundary.target or "")
        grouped.setdefault(key, []).append(boundary)

    for key, boundaries in grouped.items():
        starts = [item for item in boundaries if item.boundary == "replay_start"]
        ends = [item for item in boundaries if item.boundary == "replay_end"]
        if len(starts) != len(ends):
            errors.append(f"{key}: replay_start/replay_end count mismatch ({len(starts)} != {len(ends)})")
            continue
        for start, end in zip(sorted(starts, key=lambda item: item.frame), sorted(ends, key=lambda item: item.frame)):
            if start.frame >= end.frame:
                errors.append(f"{key}: replay_start frame {start.frame} must be before replay_end frame {end.frame}")

        ordered = sorted(boundaries, key=lambda item: item.frame)
        for previous, current in zip(ordered, ordered[1:]):
            if previous.frame == current.frame and previous.id != current.id:
                errors.append(f"{key}: duplicate boundary frame {current.frame}")
    return errors


def find_replay_pair(
    annotation: Optional[ReplayAnnotation],
    *,
    skill_family: str,
    skill_name: str,
    segment: str,
    object: Optional[str],
    target: Optional[str],
    effector: Optional[str],
) -> Tuple[Optional[ReplayBoundary], Optional[ReplayBoundary]]:
    if annotation is None:
        return None, None
    matches = list(
        annotation.iter_for(
            skill_family=skill_family,
            skill_name=skill_name,
            segment=segment,
            object=object,
            target=target,
            effector=effector,
        )
    )
    starts = sorted([item for item in matches if item.boundary == "replay_start"], key=lambda item: item.frame)
    ends = sorted([item for item in matches if item.boundary == "replay_end"], key=lambda item: item.frame)
    if not starts or not ends:
        return None, None
    for start in starts:
        for end in ends:
            if start.frame < end.frame:
                return start, end
    return starts[0], ends[0]


def extract_boundary_states_from_hdf5(source_demo_path: str, annotation: ReplayAnnotation) -> Dict[str, BoundaryState]:
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError("h5py is required to extract boundary states from ElogGen generation runtime HDF5 files") from exc

    states: Dict[str, BoundaryState] = {}
    with h5py.File(source_demo_path, "r") as h5_file:
        demo_path = f"data/{annotation.demo_id}"
        if demo_path not in h5_file:
            raise ValueError(f"Cannot find {demo_path!r} in {source_demo_path}")
        demo = h5_file[demo_path]
        action_ds = demo.get("action")
        state_ds = demo.get("state")
        length = _dataset_length(action_ds, state_ds)
        for boundary in annotation.boundaries:
            if boundary.frame < 0 or boundary.frame >= length:
                raise ValueError(f"{boundary.id}: frame {boundary.frame} outside demo length {length}")
            warnings: List[str] = []
            source_action = _row_to_list(action_ds, boundary.frame)
            state_vector = _row_to_list(state_ds, boundary.frame)
            robot_joint_state = state_vector[:16] if state_vector else None
            if robot_joint_state is not None:
                warnings.append("state layout is not annotated; robot_joint_state uses state[:16] fallback")
            else:
                warnings.append("state dataset unavailable; robot_joint_state is null")
            if source_action is None:
                warnings.append("action dataset unavailable; source_action is null")
            warnings.extend(
                [
                    "active_effector_pose requires simulator/datagen pose metadata",
                    "active_object_pose requires simulator/datagen pose metadata",
                    "target_pose requires simulator/datagen pose metadata",
                ]
            )
            states[boundary.id] = BoundaryState(
                boundary_id=boundary.id,
                frame_index=boundary.frame,
                demo_id=annotation.demo_id,
                object=boundary.object,
                target=boundary.target,
                tool=boundary.tool,
                handle=boundary.handle,
                effector=boundary.effector,
                robot_joint_state=robot_joint_state,
                source_action=source_action,
                state_vector=state_vector,
                warnings=warnings,
            )
    return states


def _dataset_length(*datasets: Any) -> int:
    for dataset in datasets:
        if dataset is not None:
            return int(dataset.shape[0])
    raise ValueError("Cannot infer demo length: no action/state datasets found")


def _row_to_list(dataset: Any, frame: int) -> Optional[List[float]]:
    if dataset is None:
        return None
    row = dataset[frame]
    return [float(value) for value in row.tolist()]
