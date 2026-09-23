"""Articulated-object skill family."""

from __future__ import annotations

from typing import List

from .base import AtomicSkill, SkillFamily


class ArticulatedSkillFamily(SkillFamily):
    name = "articulated"

    def propose_skills(self, task_graph) -> List[AtomicSkill]:
        skills = []
        for target in task_graph.targets:
            if target.entity_type not in {"articulated", "drawer", "door", "container"}:
                continue
            skills.append(
                AtomicSkill(
                    name="open_articulated",
                    family=self.name,
                    inputs=[target.name],
                    outputs=[f"open({target.name})"],
                    preconditions=[f"closed({target.name})"],
                    effects=[f"open({target.name})"],
                    constraints=["dependency", "reachability"],
                    stage_template=[
                        "approach_handle",
                        "grasp_handle",
                        "pull_open",
                        "release_handle",
                        "retract",
                    ],
                )
            )
            skills.append(
                AtomicSkill(
                    name="close_articulated",
                    family=self.name,
                    inputs=[target.name],
                    outputs=[f"closed({target.name})"],
                    preconditions=[f"open({target.name})"],
                    effects=[f"closed({target.name})"],
                    constraints=["dependency", "reachability"],
                    stage_template=[
                        "approach_handle",
                        "push_close",
                        "release_handle",
                        "retract",
                    ],
                )
            )
        return skills

    def expand(self, skill, binding=None, effector=None):
        from ..model import SemanticUnit

        target = skill.inputs[0] if skill.inputs else None
        if skill.name == "open_articulated":
            return [
                SemanticUnit(
                    id=f"{target}:approach_handle",
                    action="approach_handle",
                    target=target,
                    effector=effector,
                    effects=[f"near_handle({target})"],
                    metadata={"skill_family": self.name, "skill_name": skill.name},
                ),
                SemanticUnit(
                    id=f"{target}:pull_open",
                    action="pull_open",
                    target=target,
                    effector=effector,
                    preconditions=[f"closed({target})"],
                    effects=[f"open({target})"],
                    metadata={"skill_family": self.name, "skill_name": skill.name},
                ),
            ]
        if skill.name == "close_articulated":
            return [
                SemanticUnit(
                    id=f"{target}:push_close",
                    action="push_close",
                    target=target,
                    effector=effector,
                    preconditions=[f"open({target})"],
                    effects=[f"closed({target})"],
                    metadata={"skill_family": self.name, "skill_name": skill.name},
                )
            ]
        return []
