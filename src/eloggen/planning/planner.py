"""Constraint-aware execution plan enumeration."""

from __future__ import annotations

from itertools import product, permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .constraints import check_plan_constraints
from .model import ExecutionPlan, ObjectBinding, SemanticUnit, TaskGraph
from .skills import SKILL_FAMILIES


class PlanGenerator:
    def __init__(self, task_graph: TaskGraph):
        self.task_graph = task_graph

    def enumerate_plans(self, max_plans: int | None = None) -> List[ExecutionPlan]:
        bindings = list(self.task_graph.bindings)
        if not bindings:
            plan = self._skill_only_plan()
            plan.constraint_results = check_plan_constraints(self.task_graph, plan)
            return [plan]

        orders = self._candidate_orders([binding.object for binding in bindings])
        plans = []
        for order in orders:
            for effector_assignment in self._candidate_effector_assignments(order):
                for opener_effector in self._candidate_opener_effectors():
                    plan = self._build_plan(order, effector_assignment, opener_effector)
                    plan.constraint_results = check_plan_constraints(self.task_graph, plan)
                    if all(plan.constraint_results.values()):
                        plans.append(plan)
                    if max_plans is not None and len(plans) >= max_plans:
                        return plans
        return plans

    def get_plan(self, plan_id: str) -> ExecutionPlan:
        for plan in self.enumerate_plans():
            if plan.id == plan_id:
                return plan
        available = ", ".join(plan.id for plan in self.enumerate_plans())
        raise ValueError(f"Unknown plan_id {plan_id!r}. Available plans: {available}")

    def _candidate_orders(self, object_names: Sequence[str]) -> List[Tuple[str, ...]]:
        orders: List[Tuple[str, ...]] = []
        for alias in self.task_graph.plan_aliases.values():
            order = tuple(str(item) for item in alias.get("object_order", []))
            if len(order) == len(object_names) and set(order) == set(object_names) and order not in orders:
                orders.append(order)
        for order in permutations(object_names):
            if order not in orders:
                orders.append(order)
        return orders

    def _build_plan(
        self,
        object_order: Sequence[str],
        effector_assignment: Optional[Dict[str, str]] = None,
        opener_effector: Optional[str] = None,
    ) -> ExecutionPlan:
        if effector_assignment is None:
            effector_assignment = self._preferred_effector_assignment(object_order)
        stage_sequence: List[SemanticUnit] = []
        stage_sequence.extend(self._open_container_units(effector_assignment, opener_effector))
        pick_place = SKILL_FAMILIES["pick_place"]
        skills = {skill.inputs[0]: skill for skill in pick_place.propose_skills(self.task_graph) if skill.inputs}

        for object_name in object_order:
            binding = self.task_graph.binding_for_object(object_name)
            if binding is None:
                continue
            skill = skills.get(object_name)
            if skill is None:
                continue
            stage_sequence.extend(pick_place.expand(skill, binding, effector_assignment.get(object_name)))

        stage_sequence.extend(self._storage_verify_units(effector_assignment))
        stage_sequence.extend(self._close_container_units(effector_assignment, opener_effector))
        phase_order = self._phase_order_for_objects(object_order)
        return ExecutionPlan(
            id=self._plan_id(object_order, effector_assignment, opener_effector),
            object_order=list(object_order),
            effector_assignment=effector_assignment,
            stage_sequence=stage_sequence,
            phase_order=phase_order,
            metadata={
                "source": "eloggen_plan_generator",
                "opener_effector": opener_effector,
            },
        )

    def _candidate_effector_assignments(self, object_order: Sequence[str]) -> List[Dict[str, str]]:
        choices: List[List[str]] = []
        for object_name in object_order:
            binding = self.task_graph.binding_for_object(object_name)
            allowed = binding.allowed_effectors if binding and binding.allowed_effectors else []
            choices.append(allowed or self.task_graph.effector_names() or ["left", "right"])

        preferred = self._preferred_effector_assignment(object_order)
        assignments = []
        if preferred:
            assignments.append(preferred)
        for combo in product(*choices):
            assignment = {object_name: effector for object_name, effector in zip(object_order, combo)}
            if assignment not in assignments:
                assignments.append(assignment)
        return assignments

    def _preferred_effector_assignment(self, object_order: Sequence[str]) -> Dict[str, str]:
        effectors = self.task_graph.effector_names() or ["left", "right"]
        assignments: Dict[str, str] = {}
        for index, object_name in enumerate(object_order):
            binding = self.task_graph.binding_for_object(object_name)
            allowed = binding.allowed_effectors if binding and binding.allowed_effectors else []
            preferred = None
            if binding is not None:
                preferred = binding.metadata.get("preferred_effector")
            assignments[object_name] = str(preferred or (allowed[0] if allowed else effectors[index % len(effectors)]))
        return assignments

    def _phase_order_for_objects(self, object_order: Sequence[str]) -> List[str]:
        phase_order = []
        phase_keys_by_object = self.task_graph.metadata.get("phase_keys_by_object") or {}
        for object_name in object_order:
            for phase_key in phase_keys_by_object.get(object_name, []):
                if phase_key not in phase_order:
                    phase_order.append(phase_key)
        return phase_order

    def _candidate_opener_effectors(self) -> List[Optional[str]]:
        if "articulated" not in self.task_graph.skill_families and "storage" not in self.task_graph.skill_families:
            return [None]
        targets = [
            self._target_by_name(target_name)
            for target_name in self._targets_for_bindings()
        ]
        needs_opener = any(
            target is not None and target.entity_type in {"drawer", "door", "articulated", "container"}
            for target in targets
        )
        if not needs_opener:
            return [None]
        return self.task_graph.effector_names() or ["left", "right"]

    def _plan_id(
        self,
        object_order: Sequence[str],
        effector_assignment: Optional[Dict[str, str]] = None,
        opener_effector: Optional[str] = None,
    ) -> str:
        effector_assignment = effector_assignment or self._preferred_effector_assignment(object_order)
        preferred_assignment = self._preferred_effector_assignment(object_order)
        preferred_opener = (self.task_graph.effector_names() or ["left", "right"])[0] if opener_effector else None
        is_default_assignment = effector_assignment == preferred_assignment
        is_default_opener = opener_effector == preferred_opener
        base_id = None
        for alias_name, alias in self.task_graph.plan_aliases.items():
            if list(object_order) == list(alias.get("object_order", [])):
                base_id = str(alias_name)
                break
        if base_id is None:
            base_id = "order_" + "_then_".join(object_order)
        if is_default_assignment and is_default_opener:
            return base_id

        parts = [base_id]
        if opener_effector:
            parts.append(f"open_{opener_effector}")
        hand_suffix = "_".join(f"{object_name}_{effector_assignment[object_name]}" for object_name in object_order)
        if not is_default_assignment:
            parts.append("hands_" + hand_suffix)
        return "__".join(parts)

    def _target_by_name(self, target_name: str):
        for target in self.task_graph.targets:
            if target.name == target_name:
                return target
        return None

    def _targets_for_bindings(self) -> List[str]:
        names: List[str] = []
        for binding in self.task_graph.bindings:
            if binding.target and binding.target not in names:
                names.append(binding.target)
        return names

    def _open_container_units(
        self,
        effector_assignment: Dict[str, str],
        opener_effector: Optional[str] = None,
    ) -> List[SemanticUnit]:
        if "articulated" not in self.task_graph.skill_families and "storage" not in self.task_graph.skill_families:
            return []
        units: List[SemanticUnit] = []
        family = SKILL_FAMILIES["articulated"]
        skills = {skill.inputs[0]: skill for skill in family.propose_skills(self.task_graph) if skill.name == "open_articulated"}
        for target_name in self._targets_for_bindings():
            target = self._target_by_name(target_name)
            if target is None or target.entity_type not in {"drawer", "door", "articulated", "container"}:
                continue
            skill = skills.get(target_name)
            if skill is not None:
                units.extend(family.expand(skill, effector=opener_effector or next(iter(effector_assignment.values()), None)))
        return units

    def _close_container_units(
        self,
        effector_assignment: Dict[str, str],
        opener_effector: Optional[str] = None,
    ) -> List[SemanticUnit]:
        if "articulated" not in self.task_graph.skill_families and "storage" not in self.task_graph.skill_families:
            return []
        closed_goals = {goal.args[0] for goal in self.task_graph.goals if goal.predicate == "closed" and goal.args}
        if not closed_goals:
            return []
        units: List[SemanticUnit] = []
        family = SKILL_FAMILIES["articulated"]
        skills = {skill.inputs[0]: skill for skill in family.propose_skills(self.task_graph) if skill.name == "close_articulated"}
        for target_name in self._targets_for_bindings():
            if target_name not in closed_goals:
                continue
            skill = skills.get(target_name)
            if skill is not None:
                units.extend(family.expand(skill, effector=opener_effector or next(iter(effector_assignment.values()), None)))
        return units

    def _storage_verify_units(self, effector_assignment: Dict[str, str]) -> List[SemanticUnit]:
        if "storage" not in self.task_graph.skill_families:
            return []
        family = SKILL_FAMILIES["storage"]
        units: List[SemanticUnit] = []
        skills = {skill.inputs[0]: skill for skill in family.propose_skills(self.task_graph) if skill.inputs}
        for target_name in self._targets_for_bindings():
            skill = skills.get(target_name)
            if skill is not None:
                units.extend(family.expand(skill, effector=next(iter(effector_assignment.values()), None)))
        return units

    def _skill_only_plan(self) -> ExecutionPlan:
        effectors = self.task_graph.effector_names() or ["left", "right"]
        stage_sequence: List[SemanticUnit] = []
        for family_name in self.task_graph.skill_families:
            family = SKILL_FAMILIES.get(family_name)
            if family is None:
                continue
            for skill in family.propose_skills(self.task_graph):
                stage_sequence.extend(family.expand(skill, effector=effectors[0]))
        return ExecutionPlan(
            id="skill_plan_0",
            object_order=[],
            effector_assignment={},
            stage_sequence=stage_sequence,
            phase_order=[],
            metadata={"source": "eloggen_plan_generator", "plan_type": "skill_only"},
        )

    def _empty_plan(self) -> ExecutionPlan:
        plan = ExecutionPlan(id="empty", metadata={"source": "eloggen_plan_generator"})
        plan.constraint_results = check_plan_constraints(self.task_graph, plan)
        return plan
