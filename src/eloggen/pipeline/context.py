"""ElogGen semantic-context sidecar helpers.

The top-level fields intentionally mirror eloggen.generation_runtime.context
frame_context_json records. ElogGen-specific fields live under metadata.eloggen
so existing HDF5 and LeRobot conversion code can keep reading stage_type,
active_object, grasp_order, and phase_execution_order without schema drift.
"""

from __future__ import annotations

from typing import Dict, List

from eloggen.planning.model import ExecutionPlan
from eloggen.planning.recipe import TrajectoryRecipe


def plan_context_records(plan: ExecutionPlan) -> List[Dict[str, object]]:
    records = []
    for unit in plan.stage_sequence:
        records.append(
            {
                "eloggen_plan_id": plan.id,
                "skill_family": unit.metadata.get("skill_family"),
                "skill_name": unit.metadata.get("skill_name"),
                "semantic_unit_id": unit.id,
                "object_order": list(plan.object_order),
                "effector_assignment": dict(plan.effector_assignment),
                "action": unit.action,
                "object": unit.object,
                "target": unit.target,
                "state_before": list(unit.preconditions),
                "state_after": list(unit.effects),
                "constraint_results": dict(plan.constraint_results),
            }
        )
    return records


def recipe_context_records(recipe: TrajectoryRecipe) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    generated_frame = 0
    for segment_index, segment in enumerate(recipe.ordered_segments):
        action = generation_action(segment)
        stage_type = generation_stage_type(segment)
        segment_name = generation_segment_name(segment, stage_type)
        frame_count = _skeleton_frame_count(segment)
        for segment_frame in range(frame_count):
            records.append(
                {
                    "task": recipe.metadata.get("task_name") or "eloggen_task",
                    "action": action,
                    "stage_type": stage_type,
                    "substage_id": segment_index,
                    "segment": segment_name,
                    "generated_frame": generated_frame,
                    "segment_frame": segment_frame,
                    "phase_index": segment.metadata.get("phase_index"),
                    "subtask_index": segment.metadata.get("subtask_index"),
                    "active_arm": segment.effector,
                    "active_object": segment.object,
                    "passive_object": segment.target if action in {"place", "insert"} else segment.object,
                    "object_label_zh": segment.metadata.get("object_label_zh"),
                    "object_label_en": segment.metadata.get("object_label_en"),
                    "target_object": segment.target,
                    "target_label_zh": segment.metadata.get("target_label_zh"),
                    "target_label_en": segment.metadata.get("target_label_en"),
                    "instruction_zh": segment.metadata.get("instruction_zh"),
                    "instruction_en": segment.metadata.get("instruction_en"),
                    "source_demo_ind": segment.metadata.get("source_demo_ind"),
                    "boundary_source": "eloggen_trajectory_recipe",
                    "skill_type": stage_type,
                    "phase_action": action,
                    "phase_reference_object": segment.target if action in {"place", "insert"} else segment.object,
                    "phase_attached_object": segment.object if action in {"place", "insert"} else None,
                    "subtask_id": segment.metadata.get("subtask_id"),
                    "subtask_index_in_task": segment.metadata.get("subtask_id"),
                    "subtask_type": "transfer" if segment.object and segment.target else action,
                    "subtask_object": segment.object,
                    "subtask_target": segment.target,
                    "execution_order_index": segment.metadata.get("execution_order_index"),
                    "execution_order_strategy": recipe.metadata.get("execution_order_strategy", "eloggen_logic"),
                    "phase_execution_order": recipe.metadata.get("phase_execution_order"),
                    "phase_execution_order_indices": recipe.metadata.get("phase_execution_order_indices"),
                    "grasp_order": list(recipe.object_order) if recipe.object_order else None,
                    "metadata": {
                        "execution_order_index": segment.metadata.get("execution_order_index"),
                        "execution_order_strategy": recipe.metadata.get("execution_order_strategy", "eloggen_logic"),
                        "phase_execution_order": recipe.metadata.get("phase_execution_order"),
                        "phase_execution_order_indices": recipe.metadata.get("phase_execution_order_indices"),
                        "grasp_order": list(recipe.object_order) if recipe.object_order else None,
                        "eloggen": {
                            "logic_id": recipe.logic_id,
                            "trajectory_recipe_id": recipe.recipe_id,
                            "segment_id": segment.segment_id,
                            "segment_type": segment.type,
                            "skill_family": segment.skill_family,
                            "skill_name": segment.skill_name,
                            "source_demo_id": recipe.metadata.get("annotation_demo_id"),
                            "source_frame_range": list(segment.source_frame_range) if segment.source_frame_range else None,
                            "is_replay_frame": segment.type == "replay",
                            "is_mp_frame": segment.type == "mp",
                            "boundary_coverage": segment.metadata.get("boundary_coverage"),
                            "constraint_results": dict(recipe.global_constraints),
                        },
                    },
                }
            )
            generated_frame += 1
    return records


def generation_action(segment) -> str:
    if segment.skill_family == "pick_place" and segment.skill_name == "grasp_object":
        return "pick"
    if segment.skill_family == "pick_place" and segment.skill_name in {"insert_into_container", "place_on_target"}:
        return "place"
    if segment.type == "verify":
        return "verify"
    if segment.skill_family == "articulated":
        return "rotate"
    return segment.skill_name or segment.type


def generation_stage_type(segment) -> str:
    action = generation_action(segment)
    if segment.type == "mp":
        if action == "place":
            return "approach_place"
        if action == "insert":
            return "approach_insert"
        return "approach"
    if segment.type == "replay":
        if action == "pick":
            return "grasp"
        if action == "place":
            return "place"
        if action == "insert":
            return "insert"
        return action
    if segment.type == "retract":
        return "retract"
    if segment.type == "verify":
        return "verify"
    return segment.type


def generation_segment_name(segment, stage_type: str) -> str:
    if stage_type.startswith("approach"):
        return "arm_mp"
    if segment.type == "retract":
        return "retract"
    if segment.type == "verify":
        return "verify"
    return "arm_replay"


def _skeleton_frame_count(segment) -> int:
    if segment.source_frame_range:
        start, end = segment.source_frame_range
        return max(1, int(end) - int(start) + 1)
    return 1
