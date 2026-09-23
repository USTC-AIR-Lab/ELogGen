"""Lightweight feasibility reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .model import ExecutionPlan


@dataclass
class FeasibilityResult:
    plan_id: str
    plan_feasible: bool
    failed_constraint: Optional[str] = None
    failed_skill: Optional[str] = None
    reason: Optional[str] = None
    suggested_repair: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data["metadata"] is None:
            data["metadata"] = {}
        return data


def evaluate_plan_feasibility(plan: ExecutionPlan) -> FeasibilityResult:
    for name, passed in plan.constraint_results.items():
        if not passed:
            return FeasibilityResult(
                plan_id=plan.id,
                plan_feasible=False,
                failed_constraint=name,
                reason=f"Plan failed {name}",
                suggested_repair="Try another object order or effector assignment.",
                metadata={},
            )
    return FeasibilityResult(plan_id=plan.id, plan_feasible=True, metadata={})

