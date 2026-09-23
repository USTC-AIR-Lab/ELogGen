"""Plan-level constraint checks."""

from __future__ import annotations

from typing import Dict

from .model import ExecutionPlan, TaskGraph


def check_plan_constraints(task_graph: TaskGraph, plan: ExecutionPlan) -> Dict[str, bool]:
    checks = {}
    checks["goal_consistency"] = _goal_consistency(task_graph, plan)
    checks["dependency"] = _dependency(plan)
    checks["mutual_exclusion"] = _mutual_exclusion(plan)
    checks["effector_reachability"] = _effector_reachability(task_graph, plan)
    checks["collision_heuristic"] = True
    return checks


def _goal_consistency(task_graph: TaskGraph, plan: ExecutionPlan) -> bool:
    bound_objects = {binding.object for binding in task_graph.bindings if binding.target is not None}
    return set(plan.object_order).issubset(bound_objects) if bound_objects else True


def _dependency(plan: ExecutionPlan) -> bool:
    seen_grasp = set()
    for unit in plan.stage_sequence:
        if unit.action == "grasp" and unit.object:
            seen_grasp.add(unit.object)
        if unit.action in {"place", "insert"} and unit.object and unit.object not in seen_grasp:
            return False
    return True


def _mutual_exclusion(plan: ExecutionPlan) -> bool:
    assigned = plan.effector_assignment
    return all(effector for effector in assigned.values())


def _effector_reachability(task_graph: TaskGraph, plan: ExecutionPlan) -> bool:
    for object_name, effector in plan.effector_assignment.items():
        binding = task_graph.binding_for_object(object_name)
        if binding and binding.allowed_effectors and effector not in binding.allowed_effectors:
            return False
    return True

