"""Pick/place atomic skill family."""

from __future__ import annotations

from typing import List

from .base import AtomicSkill, SkillFamily
from ..model import ObjectBinding, SemanticUnit


class PickPlaceSkillFamily(SkillFamily):
    name = "pick_place"

    def propose_skills(self, task_graph) -> List[AtomicSkill]:
        skills = []
        for binding in task_graph.bindings:
            target = binding.target or "target"
            skills.append(
                AtomicSkill(
                    name="insert_into_container" if binding.target else "place_on_target",
                    family=self.name,
                    inputs=[binding.object, target],
                    outputs=[f"inside({binding.object},{target})"],
                    preconditions=[f"reachable({binding.object})", f"reachable({target})"],
                    effects=[f"inside({binding.object},{target})"],
                    constraints=["goal_consistency", "dependency", "mutual_exclusion", "reachability"],
                    stage_template=[
                        "approach_object",
                        "grasp_object",
                        "transport_object",
                        "place_or_insert",
                        "retract",
                    ],
                )
            )
        return skills

    def expand(self, skill: AtomicSkill, binding: ObjectBinding | None = None, effector: str | None = None):
        if binding is None:
            return []
        target = binding.target
        obj = binding.object
        action = "insert" if target else "place"
        return [
            SemanticUnit(
                id=f"{obj}:approach",
                action="approach",
                object=obj,
                effector=effector,
                effects=[f"near({obj})"],
                metadata={"skill_family": self.name, "skill_name": skill.name},
            ),
            SemanticUnit(
                id=f"{obj}:grasp",
                action="grasp",
                object=obj,
                effector=effector,
                preconditions=[f"near({obj})"],
                effects=[f"held({obj})"],
                metadata={"skill_family": self.name, "skill_name": skill.name},
            ),
            SemanticUnit(
                id=f"{obj}:transport",
                action="transport",
                object=obj,
                target=target,
                effector=effector,
                preconditions=[f"held({obj})"],
                effects=[f"near({target})"] if target else [],
                metadata={"skill_family": self.name, "skill_name": skill.name},
            ),
            SemanticUnit(
                id=f"{obj}:{action}",
                action=action,
                object=obj,
                target=target,
                effector=effector,
                preconditions=[f"held({obj})"],
                effects=[f"inside({obj},{target})"] if target else [f"placed({obj})"],
                metadata={"skill_family": self.name, "skill_name": skill.name},
            ),
            SemanticUnit(
                id=f"{obj}:retract",
                action="retract",
                object=obj,
                effector=effector,
                metadata={"skill_family": self.name, "skill_name": skill.name},
            ),
        ]

