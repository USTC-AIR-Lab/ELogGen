"""Build ElogGen task graphs from existing ElogGen generation runtime configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from eloggen.planning.model import (
    ConstraintSpec,
    EffectorSpec,
    GoalSpec,
    ObjectBinding,
    ObjectSpec,
    SemanticUnit,
    SourceDemo,
    TargetSpec,
    TaskGraph,
)
from eloggen.datasets import resolve_dataset_path
from .io import load_structured_file


def _phase_sort_key(item: Tuple[str, Any]) -> int:
    key = item[0]
    try:
        return int(str(key).rsplit("_", 1)[-1])
    except Exception:
        return 10**9


def _iter_arm_specs(phase_spec: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for arm_key, arm_name in (("arm_left", "left"), ("arm_right", "right")):
        arm_data = phase_spec.get(arm_key)
        if not arm_data:
            continue
        if isinstance(arm_data, list):
            for spec in arm_data:
                if isinstance(spec, dict):
                    yield arm_name, spec
            continue
        if isinstance(arm_data, dict):
            for _, spec in sorted(arm_data.items()):
                if isinstance(spec, dict):
                    yield arm_name, spec


def _infer_action(spec: Dict[str, Any]) -> str:
    if spec.get("phase_action"):
        return str(spec["phase_action"])
    if spec.get("attached_obj") is not None and spec.get("object_ref") is not None:
        return "place"
    if spec.get("object_ref") is not None:
        return "pick"
    return "idle"


def inspect_source_demo(path: str) -> Dict[str, Any]:
    path = resolve_dataset_path(path) or path
    result: Dict[str, Any] = {"path": path, "exists": Path(path).exists()}
    if not result["exists"]:
        return result
    result["size_bytes"] = Path(path).stat().st_size
    try:
        import h5py  # type: ignore
    except Exception:
        result["hdf5_inspection"] = "h5py_unavailable"
        return result
    try:
        with h5py.File(path, "r") as f:
            result["root_keys"] = list(f.keys())
            if "data" in f:
                result["num_demos"] = len(f["data"].keys())
    except Exception as exc:
        result["hdf5_inspection_error"] = str(exc)
    return result


def task_graph_from_generation_config(config_path: str, source_demo: Optional[str] = None) -> TaskGraph:
    config_path = resolve_dataset_path(config_path) or config_path
    source_demo = resolve_dataset_path(source_demo)
    config = load_structured_file(config_path)
    name = str(config.get("name") or config.get("experiment", {}).get("task", {}).get("name") or Path(config_path).stem)
    task_spec = config.get("task", {}).get("task_spec")
    if task_spec is None:
        task_spec = config.get("experiment", {}).get("task", {}).get("task_spec")
    if not isinstance(task_spec, dict):
        raise ValueError(f"Cannot find task.task_spec in {config_path}")

    objects: Dict[str, ObjectSpec] = {}
    targets: Dict[str, TargetSpec] = {}
    bindings: Dict[str, ObjectBinding] = {}
    pick_effectors: Dict[str, List[str]] = {}
    semantic_units: List[SemanticUnit] = []
    phase_keys_by_object: Dict[str, List[str]] = {}

    for phase_key, phase_spec in sorted(task_spec.items(), key=_phase_sort_key):
        if not isinstance(phase_spec, dict):
            continue
        phase_index = _phase_sort_key((phase_key, phase_spec))
        for arm_name, spec in _iter_arm_specs(phase_spec):
            action = _infer_action(spec)
            object_ref = spec.get("object_ref")
            attached_obj = spec.get("attached_obj")
            active_object = attached_obj if action == "place" else object_ref
            target = object_ref if action == "place" else None
            if active_object is None:
                continue

            label = spec.get("object_label_en") or spec.get("object_label_zh")
            if str(active_object) not in objects:
                objects[str(active_object)] = ObjectSpec(
                    name=str(active_object),
                    label=str(label) if label else None,
                    allowed_effectors=[],
                    metadata={},
                )
            if label and not objects[str(active_object)].label:
                objects[str(active_object)].label = str(label)

            if action == "pick":
                pick_effectors.setdefault(str(active_object), [])
                if arm_name not in pick_effectors[str(active_object)]:
                    pick_effectors[str(active_object)].append(arm_name)
            if action == "place" and target is not None:
                target_label = spec.get("target_label_en") or spec.get("target_label_zh")
                if str(target) not in targets:
                    targets[str(target)] = TargetSpec(
                        name=str(target),
                        label=str(target_label) if target_label else None,
                        entity_type="container",
                    )
                bindings[str(active_object)] = ObjectBinding(
                    object=str(active_object),
                    target=str(target),
                    metadata={"source": "generation_task_spec"},
                )

            phase_keys_by_object.setdefault(str(active_object), [])
            if phase_key not in phase_keys_by_object[str(active_object)]:
                phase_keys_by_object[str(active_object)].append(phase_key)

            semantic_units.append(
                SemanticUnit(
                    id=f"{phase_key}:{arm_name}:{action}:{active_object}",
                    action=action,
                    object=str(active_object),
                    target=str(target) if target is not None else None,
                    effector=arm_name,
                    preconditions=[],
                    effects=[],
                    metadata={
                        "source": "generation_task_spec",
                        "phase_key": phase_key,
                        "phase_index": phase_index,
                        "arm": arm_name,
                        "object_ref": object_ref,
                        "attached_obj": attached_obj,
                        "MP_end_step": spec.get("MP_end_step"),
                        "subtask_term_step": spec.get("subtask_term_step"),
                        "retract_type": spec.get("retract_type"),
                        "num_interpolation_steps": spec.get("num_interpolation_steps"),
                    },
                )
            )

    all_effectors = ["left", "right"]
    for object_name, obj in objects.items():
        source_effectors = pick_effectors.get(object_name, [])
        obj.allowed_effectors = list(all_effectors)
        if source_effectors:
            obj.metadata["source_effectors"] = list(source_effectors)
            obj.metadata["preferred_effector"] = source_effectors[0]
        if object_name in bindings:
            bindings[object_name].allowed_effectors = list(all_effectors)
            if source_effectors:
                bindings[object_name].metadata["source_effectors"] = list(source_effectors)
                bindings[object_name].metadata["preferred_effector"] = source_effectors[0]
    for object_name, phase_keys in phase_keys_by_object.items():
        objects[object_name].metadata["phase_keys"] = phase_keys

    source_demos = []
    if source_demo:
        source_demos.append(SourceDemo(path=source_demo, metadata=inspect_source_demo(source_demo)))

    goals = [
        GoalSpec(predicate="inside", args=[binding.object, str(binding.target)])
        for binding in bindings.values()
        if binding.target is not None
    ]

    constraints = [
        ConstraintSpec(type="goal_consistency"),
        ConstraintSpec(type="dependency", params={"before": "pick(object)", "after": "place(object,target)"}),
        ConstraintSpec(type="mutual_exclusion", params={"scope": "same_object"}),
        ConstraintSpec(type="effector_reachability"),
        ConstraintSpec(type="collision_heuristic"),
    ]

    plan_aliases = _default_plan_aliases(objects)

    return TaskGraph(
        name=name,
        source_demos=source_demos,
        objects=list(objects.values()),
        targets=list(targets.values()),
        effectors=[EffectorSpec(name="left"), EffectorSpec(name="right")],
        goals=goals,
        constraints=constraints,
        bindings=list(bindings.values()),
        semantic_units=semantic_units,
        skill_families=["pick_place"],
        plan_aliases=plan_aliases,
        metadata={
            "source_config": config_path,
            "source_format": "generation_task_spec",
            "phase_keys_by_object": phase_keys_by_object,
        },
    )


def _default_plan_aliases(objects: Dict[str, ObjectSpec]) -> Dict[str, Dict[str, Any]]:
    aliases: Dict[str, Dict[str, Any]] = {}
    if {"object_1", "object_2"}.issubset(objects):
        label_1 = (objects["object_1"].label or "object_1").lower().replace(" ", "_")
        label_2 = (objects["object_2"].label or "object_2").lower().replace(" ", "_")
        aliases[f"{label_1}_first"] = {"object_order": ["object_1", "object_2"]}
        aliases[f"{label_2}_first"] = {"object_order": ["object_2", "object_1"]}
    return aliases
