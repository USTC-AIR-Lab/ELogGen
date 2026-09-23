"""Boundary selection for source demonstration interaction segments."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from ..replay_annotation import ReplayAnnotation
from eloggen.planning.model import SemanticUnit


def annotation_ranges(annotation: ReplayAnnotation) -> Dict[Tuple[str, str, str], Tuple[int, int]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    for item in annotation.boundaries:
        if item.object is None or item.effector is None:
            continue
        key = (item.object, item.segment, item.effector)
        grouped.setdefault(key, {})[item.boundary] = item.frame
    return {
        key: (values["replay_start"], values["replay_end"])
        for key, values in grouped.items()
        if "replay_start" in values and "replay_end" in values
    }


def resolve_unit_boundaries(
    units: Iterable[SemanticUnit], annotation: ReplayAnnotation
) -> Dict[str, Tuple[int, int, str]]:
    """Resolve ranges with manual annotations taking priority over config metadata."""
    annotated = annotation_ranges(annotation)
    result: Dict[str, Tuple[int, int, str]] = {}
    for unit in units:
        if unit.object is None or unit.effector is None:
            continue
        segment = "grasp" if unit.action == "pick" else "place"
        key = (unit.object, segment, unit.effector)
        if key in annotated:
            start, end = annotated[key]
            result[unit.id] = (start, end, "annotation")
            continue
        start = unit.metadata.get("MP_end_step")
        end = unit.metadata.get("subtask_term_step")
        if start is not None and end is not None:
            result[unit.id] = (int(start), int(end), "task_config")
    return result


def detect_gripper_transitions(gripper_action: np.ndarray, threshold: float = 0.5) -> Dict[int, list[int]]:
    """Return frame indices where each effector's command changes materially."""
    values = np.asarray(gripper_action)
    if values.ndim != 2 or values.shape[0] < 2:
        return {}
    return {
        column: [int(frame) for frame in np.flatnonzero(np.abs(np.diff(values[:, column])) > threshold) + 1]
        for column in range(values.shape[1])
    }
