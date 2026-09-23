"""Adapter from ElogGen plans to ElogGen generation runtime task specs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from eloggen.datasets import resolve_dataset_path
from eloggen.pipeline.boundaries import extract_boundary_states_from_hdf5, validate_replay_annotation
from eloggen.pipeline.replay_annotation import ReplayAnnotation, load_replay_annotation, replay_annotation_from_task_graph
from eloggen.planning.model import ExecutionPlan, TaskGraph
from eloggen.planning.recipe import build_trajectory_recipe
from eloggen.pipeline.io import load_structured_file


def export_generation_config(
    base_config_path: str,
    task_graph: TaskGraph,
    plan: ExecutionPlan,
    source_demo: Optional[str] = None,
    processed_source_demo: Optional[str] = None,
    replay_annotation_path: Optional[str] = None,
) -> Dict[str, Any]:
    base_config_path = resolve_dataset_path(base_config_path) or base_config_path
    source_demo = resolve_dataset_path(source_demo)
    processed_source_demo = resolve_dataset_path(processed_source_demo)
    replay_annotation_path = resolve_dataset_path(replay_annotation_path)
    config = deepcopy(load_structured_file(base_config_path))
    task_spec = config.get("task", {}).get("task_spec")
    if not isinstance(task_spec, dict):
        raise ValueError(f"Cannot find task.task_spec in {base_config_path}")

    # IMPORTANT: task_spec order is the chronological SOURCE replay order.
    # MP_end_step / subtask_term_step are absolute indices into the processed
    # source demonstration, so physically reordering phase_* entries here makes
    # source segments non-monotonic (for example [1120, 580]) and eventually
    # triggers "got empty subtasks!" in DataGenerator.
    #
    # Execution order is a separate runtime concern and is carried by the plan /
    # subtask scheduler. Keep the source task spec exactly in its base-config
    # order and record the requested execution order only as ElogGen metadata.
    config["task"]["task_spec"] = deepcopy(task_spec)

    annotation: Optional[ReplayAnnotation] = None
    boundary_states = {}
    annotation_errors = []
    if replay_annotation_path:
        annotation = load_replay_annotation(replay_annotation_path)
        annotation_errors = validate_replay_annotation(annotation)
        if annotation_errors:
            raise ValueError("Invalid replay annotation: " + "; ".join(annotation_errors))
        demo_path = source_demo or _first_source_demo(task_graph)
        if demo_path:
            boundary_states = extract_boundary_states_from_hdf5(demo_path, annotation)
    else:
        inferred = replay_annotation_from_task_graph(task_graph)
        if inferred.boundaries:
            annotation = inferred
            demo_path = source_demo or _first_source_demo(task_graph)
            if demo_path:
                boundary_states = extract_boundary_states_from_hdf5(demo_path, annotation)

    recipe = build_trajectory_recipe(task_graph, plan, annotation=annotation, boundary_states=boundary_states)

    config.setdefault("eloggen", {})
    config["eloggen"].update(
        {
            "task_graph": task_graph.name,
            "logic_id": plan.id,
            "plan_id": plan.id,
            "object_order": list(plan.object_order),
            "role_assignment": dict(plan.effector_assignment),
            "effector_assignment": dict(plan.effector_assignment),
            "phase_order": list(plan.phase_order),
            "constraint_results": dict(plan.constraint_results),
            "trajectory_recipe": recipe.to_dict(),
        }
    )
    if annotation is not None:
        config["eloggen"]["replay_annotation"] = annotation.to_dict()
        config["eloggen"]["boundary_states"] = {
            boundary_id: state.to_dict() for boundary_id, state in boundary_states.items()
        }
    if source_demo:
        config["eloggen"]["source_demo"] = source_demo
    if processed_source_demo:
        config["eloggen"]["processed_source_demo"] = processed_source_demo
    generation_source = processed_source_demo or source_demo
    if generation_source:
        config.setdefault("experiment", {}).setdefault("source", {})["dataset_path"] = generation_source
    return config


def _first_source_demo(task_graph: TaskGraph) -> Optional[str]:
    if not task_graph.source_demos:
        return None
    return task_graph.source_demos[0].path
