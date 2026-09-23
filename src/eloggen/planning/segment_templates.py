"""Default MP/replay/retract segment templates for ElogGen skills."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from .model import SemanticUnit


SEGMENT_TYPES = {"mp", "replay", "retract", "hold", "verify"}


@dataclass(frozen=True)
class SegmentTemplate:
    id: str
    skill_family: str
    skill_name: str
    segment: str
    type: str
    start_boundary_role: Optional[str] = None
    end_boundary_role: Optional[str] = None
    metadata: Dict[str, object] | None = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        if data["metadata"] is None:
            data["metadata"] = {}
        return data


def templates_for_unit(unit: SemanticUnit) -> List[SegmentTemplate]:
    family = str(unit.metadata.get("skill_family") or "")
    action = unit.action

    if family == "pick_place":
        if action == "grasp":
            skill_name = "grasp_object"
            return _mp_replay_pair(family, skill_name, "grasp")
        if action in {"insert", "place"}:
            skill_name = "insert_into_container" if action == "insert" else "place_on_target"
            return _mp_replay_retract(family, skill_name, "place")
        return []

    skill_name = str(unit.metadata.get("skill_name") or unit.action)
    if family == "articulated":
        if action in {"pull_open", "push_close"}:
            return _mp_replay_retract(family, skill_name, action)
        return []

    if family == "tool_assembly":
        if action in {"grasp_tool", "pick_tool", "align_tool", "insert_tool", "turn_tool", "fasten", "release_tool"}:
            return _mp_replay_retract(family, skill_name, action)
        return []

    if family == "storage" and action.startswith("verify"):
        return [
            SegmentTemplate(
                id=f"{family}.{skill_name}.{action}.verify",
                skill_family=family,
                skill_name=skill_name,
                segment=action,
                type="verify",
            )
        ]
    return []


def required_replay_segments(units: Iterable[SemanticUnit]) -> List[Dict[str, Optional[str]]]:
    required: List[Dict[str, Optional[str]]] = []
    for unit in units:
        for template in templates_for_unit(unit):
            if template.type != "replay":
                continue
            required.append(
                {
                    "skill_family": template.skill_family,
                    "skill_name": template.skill_name,
                    "segment": template.segment,
                    "object": unit.object,
                    "target": unit.target,
                    "effector": unit.effector,
                }
            )
    return required


def _mp_replay_pair(family: str, skill_name: str, segment: str) -> List[SegmentTemplate]:
    return [
        SegmentTemplate(
            id=f"{family}.{skill_name}.{segment}.mp",
            skill_family=family,
            skill_name=skill_name,
            segment=segment,
            type="mp",
            end_boundary_role="replay_start",
        ),
        SegmentTemplate(
            id=f"{family}.{skill_name}.{segment}.replay",
            skill_family=family,
            skill_name=skill_name,
            segment=segment,
            type="replay",
            start_boundary_role="replay_start",
            end_boundary_role="replay_end",
        ),
    ]


def _mp_replay_retract(family: str, skill_name: str, segment: str) -> List[SegmentTemplate]:
    templates = _mp_replay_pair(family, skill_name, segment)
    templates.append(
        SegmentTemplate(
            id=f"{family}.{skill_name}.{segment}.retract",
            skill_family=family,
            skill_name=skill_name,
            segment=segment,
            type="retract",
            start_boundary_role="replay_end",
        )
    )
    return templates
