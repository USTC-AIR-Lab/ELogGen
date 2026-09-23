"""Storage skill family for container-aware tasks."""

from __future__ import annotations

from typing import List

from .base import AtomicSkill, SkillFamily


class StorageSkillFamily(SkillFamily):
    name = "storage"

    def propose_skills(self, task_graph) -> List[AtomicSkill]:
        containers = [target.name for target in task_graph.targets if target.entity_type in {"container", "target"}]
        skills = []
        for container in containers:
            skills.append(
                AtomicSkill(
                    name="store_items",
                    family=self.name,
                    inputs=[container],
                    outputs=[f"stored_items({container})"],
                    preconditions=[f"open({container})"],
                    effects=[f"stored_items({container})"],
                    constraints=["dependency", "mutual_exclusion"],
                    stage_template=[
                        "open_container",
                        "pick_item",
                        "place_item",
                        "arrange_item",
                        "close_container",
                        "verify_storage",
                    ],
                )
            )
        return skills

    def expand(self, skill, binding=None, effector=None):
        from ..model import SemanticUnit

        container = skill.inputs[0] if skill.inputs else None
        return [
            SemanticUnit(
                id=f"{container}:verify_storage",
                action="verify_storage",
                target=container,
                effector=effector,
                preconditions=[f"stored_items({container})"],
                effects=[f"verified_storage({container})"],
                metadata={"skill_family": self.name, "skill_name": skill.name},
            )
        ]
