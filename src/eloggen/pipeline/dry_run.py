"""Dry-run executor that prints a symbolic plan trace."""

from __future__ import annotations

from typing import Dict, List

from eloggen.planning.feasibility import evaluate_plan_feasibility
from eloggen.planning.model import ExecutionPlan, TaskGraph


def dry_run_plan(task_graph: TaskGraph, plan: ExecutionPlan) -> Dict[str, object]:
    feasibility = evaluate_plan_feasibility(plan)
    trace: List[Dict[str, object]] = []
    for index, unit in enumerate(plan.stage_sequence):
        trace.append(
            {
                "step": index,
                "unit_id": unit.id,
                "action": unit.action,
                "object": unit.object,
                "target": unit.target,
                "effector": unit.effector,
                "effects": unit.effects,
            }
        )
    return {
        "task": task_graph.name,
        "plan": plan.to_dict(),
        "feasibility": feasibility.to_dict(),
        "trace": trace,
    }

