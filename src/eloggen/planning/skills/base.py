"""Base interfaces for ElogGen atomic skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AtomicSkill:
    name: str
    family: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    stage_template: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SkillFamily:
    name = "base"

    def propose_skills(self, task_graph):
        raise NotImplementedError

    def expand(self, skill: AtomicSkill, binding=None, effector: str | None = None):
        raise NotImplementedError

