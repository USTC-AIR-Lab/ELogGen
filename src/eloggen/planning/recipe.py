"""Compile execution logic into MP/replay/retract trajectory recipes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from eloggen.pipeline.boundaries import BoundaryState, find_replay_pair
from eloggen.pipeline.replay_annotation import ReplayAnnotation, ReplayBoundary
from .model import ExecutionPlan, SemanticUnit, TaskGraph
from .segment_templates import templates_for_unit


@dataclass
class TrajectorySegment:
    segment_id: str
    skill_family: str
    skill_name: str
    type: str
    object: Optional[str] = None
    target: Optional[str] = None
    tool: Optional[str] = None
    handle: Optional[str] = None
    effector: Optional[str] = None
    start_boundary: Optional[str] = None
    end_boundary: Optional[str] = None
    source_frame_range: Optional[List[int]] = None
    start_pose: Optional[Any] = None
    goal_pose: Optional[Any] = None
    attached_object: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryRecipe:
    recipe_id: str
    logic_id: str
    ordered_segments: List[TrajectorySegment] = field(default_factory=list)
    global_constraints: Dict[str, bool] = field(default_factory=dict)
    expected_final_goals: List[Dict[str, Any]] = field(default_factory=list)
    object_order: List[str] = field(default_factory=list)
    role_assignment: Dict[str, str] = field(default_factory=dict)
    boundary_coverage_status: str = "not_checked"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "logic_id": self.logic_id,
            "ordered_segments": [segment.to_dict() for segment in self.ordered_segments],
            "global_constraints": dict(self.global_constraints),
            "expected_final_goals": list(self.expected_final_goals),
            "object_order": list(self.object_order),
            "role_assignment": dict(self.role_assignment),
            "boundary_coverage_status": self.boundary_coverage_status,
            "metadata": dict(self.metadata),
        }


def build_trajectory_recipe(
    task_graph: TaskGraph,
    plan: ExecutionPlan,
    annotation: Optional[ReplayAnnotation] = None,
    boundary_states: Optional[Dict[str, BoundaryState]] = None,
) -> TrajectoryRecipe:
    boundary_states = boundary_states or {}
    ordered_segments: List[TrajectorySegment] = []
    missing_pairs = 0

    for unit_index, unit in enumerate(plan.stage_sequence):
        replay_pair: Optional[Tuple[Optional[ReplayBoundary], Optional[ReplayBoundary]]] = None
        for template in templates_for_unit(unit):
            start_boundary = None
            end_boundary = None
            source_frame_range = None
            goal_pose = None
            metadata: Dict[str, Any] = {
                "semantic_unit_id": unit.id,
                "boundary_coverage": "not_required",
            }

            if template.type == "replay":
                replay_pair = _find_pair_for_unit(annotation, template.skill_family, template.skill_name, template.segment, unit)
                start, end = replay_pair
                if start is not None and end is not None:
                    start_boundary = start.id
                    end_boundary = end.id
                    source_frame_range = [start.frame, end.frame]
                    metadata["boundary_coverage"] = "covered"
                else:
                    start_boundary = _symbolic_boundary_id(unit, template.segment, "replay_start")
                    end_boundary = _symbolic_boundary_id(unit, template.segment, "replay_end")
                    metadata["boundary_coverage"] = "missing"
                    missing_pairs += 1
            elif template.type == "mp":
                pair = replay_pair or _find_pair_for_unit(annotation, template.skill_family, template.skill_name, template.segment, unit)
                start, _ = pair
                end_boundary = start.id if start is not None else _symbolic_boundary_id(unit, template.segment, "replay_start")
                metadata["boundary_coverage"] = "covered" if start is not None else "symbolic_goal"
                if start is not None and start.id in boundary_states:
                    goal_pose = boundary_states[start.id].active_effector_pose
                else:
                    goal_pose = {"symbolic_boundary": end_boundary}
            elif template.type in {"retract", "hold"}:
                pair = replay_pair or _find_pair_for_unit(annotation, template.skill_family, template.skill_name, template.segment, unit)
                _, end = pair
                start_boundary = end.id if end is not None else _symbolic_boundary_id(unit, template.segment, "replay_end")
                metadata["boundary_coverage"] = "covered" if end is not None else "symbolic_start"
                goal_pose = {"symbolic_target": "canonical_safe_pose"}
            else:
                goal_pose = None

            segment_id = f"{len(ordered_segments) + 1:04d}_{unit.id.replace(':', '_')}_{template.type}"
            ordered_segments.append(
                TrajectorySegment(
                    segment_id=segment_id,
                    skill_family=template.skill_family,
                    skill_name=template.skill_name,
                    type=template.type,
                    object=unit.object,
                    target=unit.target,
                    effector=unit.effector,
                    start_boundary=start_boundary,
                    end_boundary=end_boundary,
                    source_frame_range=source_frame_range,
                    goal_pose=goal_pose,
                    attached_object=unit.object if unit.action in {"transport", "insert", "place"} else None,
                    constraints=list(unit.preconditions),
                    metadata=metadata,
                )
            )
        if not templates_for_unit(unit) and unit.action.startswith("verify") and unit.metadata.get("skill_family"):
            ordered_segments.append(_verify_or_symbolic_segment(unit_index, unit))

    coverage = "covered" if missing_pairs == 0 and annotation is not None else "missing_boundaries" if missing_pairs else "symbolic"
    return TrajectoryRecipe(
        recipe_id=f"{plan.id}_recipe",
        logic_id=plan.id,
        ordered_segments=ordered_segments,
        global_constraints=dict(plan.constraint_results),
        expected_final_goals=[goal.to_dict() for goal in task_graph.goals],
        object_order=list(plan.object_order),
        role_assignment=dict(plan.effector_assignment),
        boundary_coverage_status=coverage,
        metadata={
            "source": "eloggen_trajectory_recipe",
            "task_name": task_graph.name,
            "annotation_demo_id": annotation.demo_id if annotation is not None else None,
            "execution_order_strategy": "eloggen_logic",
            "phase_execution_order": list(plan.phase_order),
            "phase_execution_order_indices": [_phase_index_from_key(key) for key in plan.phase_order],
        },
    )


def _find_pair_for_unit(
    annotation: Optional[ReplayAnnotation],
    skill_family: str,
    skill_name: str,
    segment: str,
    unit: SemanticUnit,
) -> Tuple[Optional[ReplayBoundary], Optional[ReplayBoundary]]:
    return find_replay_pair(
        annotation,
        skill_family=skill_family,
        skill_name=skill_name,
        segment=segment,
        object=unit.object,
        target=unit.target,
        effector=unit.effector,
    )


def _symbolic_boundary_id(unit: SemanticUnit, segment: str, role: str) -> str:
    entity = unit.object or unit.target or "entity"
    return f"symbolic:{unit.metadata.get('skill_family')}:{entity}:{segment}:{role}"


def _verify_or_symbolic_segment(unit_index: int, unit: SemanticUnit) -> TrajectorySegment:
    family = str(unit.metadata.get("skill_family"))
    skill_name = str(unit.metadata.get("skill_name") or unit.action)
    segment_type = "verify" if unit.action.startswith("verify") else "symbolic"
    return TrajectorySegment(
        segment_id=f"{unit_index + 1:04d}_{unit.id.replace(':', '_')}_{segment_type}",
        skill_family=family,
        skill_name=skill_name,
        type=segment_type,
        object=unit.object,
        target=unit.target,
        effector=unit.effector,
        constraints=list(unit.preconditions),
        metadata={"semantic_unit_id": unit.id, "boundary_coverage": "not_required"},
    )


def _phase_index_from_key(key: str) -> int:
    try:
        return int(str(key).rsplit("_", 1)[-1]) - 1
    except Exception:
        return -1
