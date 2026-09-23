"""Tool assembly skill family."""

from __future__ import annotations

from typing import List

from .base import AtomicSkill, SkillFamily


class ToolAssemblySkillFamily(SkillFamily):
    name = "tool_assembly"

    def propose_skills(self, task_graph) -> List[AtomicSkill]:
        tools = [obj.name for obj in task_graph.objects if obj.entity_type == "tool"]
        parts = [target.name for target in task_graph.targets if target.entity_type in {"part", "assembly"}]
        skills = []
        for tool in tools:
            for part in parts or ["assembly_target"]:
                skills.append(
                    AtomicSkill(
                        name="assemble_with_tool",
                        family=self.name,
                        inputs=[tool, part],
                        outputs=[f"fastened({part})"],
                        preconditions=[f"held({tool})", f"aligned({tool},{part})"],
                        effects=[f"fastened({part})"],
                        constraints=["dependency", "reachability", "collision"],
                        stage_template=[
                            "pick_tool",
                            "align_tool",
                            "insert_tool",
                            "turn_tool",
                            "fasten",
                            "release_tool",
                            "verify_attachment",
                        ],
                    )
                )
        return skills

    def expand(self, skill, binding=None, effector=None):
        from ..model import SemanticUnit

        tool = skill.inputs[0] if skill.inputs else "tool"
        part = skill.inputs[1] if len(skill.inputs) > 1 else "assembly_target"
        return [
            SemanticUnit(
                id=f"{tool}:pick_tool",
                action="pick_tool",
                object=tool,
                effector=effector,
                effects=[f"held({tool})"],
                metadata={"skill_family": self.name, "skill_name": skill.name},
            ),
            SemanticUnit(
                id=f"{tool}:{part}:align_tool",
                action="align_tool",
                object=tool,
                target=part,
                effector=effector,
                preconditions=[f"held({tool})"],
                effects=[f"aligned({tool},{part})"],
                metadata={"skill_family": self.name, "skill_name": skill.name},
            ),
            SemanticUnit(
                id=f"{tool}:{part}:insert_tool",
                action="insert_tool",
                object=tool,
                target=part,
                effector=effector,
                preconditions=[f"aligned({tool},{part})"],
                effects=[f"inserted({tool},{part})"],
                metadata={"skill_family": self.name, "skill_name": skill.name},
            ),
            SemanticUnit(
                id=f"{tool}:{part}:fasten",
                action="fasten",
                object=tool,
                target=part,
                effector=effector,
                preconditions=[f"inserted({tool},{part})"],
                effects=[f"fastened({part})"],
                metadata={"skill_family": self.name, "skill_name": skill.name},
            ),
            SemanticUnit(
                id=f"{tool}:release_tool",
                action="release_tool",
                object=tool,
                effector=effector,
                metadata={"skill_family": self.name, "skill_name": skill.name},
            ),
        ]
