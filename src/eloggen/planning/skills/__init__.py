"""Atomic skill families."""

from .articulated import ArticulatedSkillFamily
from .base import AtomicSkill, SkillFamily
from .pick_place import PickPlaceSkillFamily
from .storage import StorageSkillFamily
from .tool_assembly import ToolAssemblySkillFamily

SKILL_FAMILIES = {
    "pick_place": PickPlaceSkillFamily(),
    "storage": StorageSkillFamily(),
    "articulated": ArticulatedSkillFamily(),
    "tool_assembly": ToolAssemblySkillFamily(),
}

__all__ = [
    "AtomicSkill",
    "SkillFamily",
    "PickPlaceSkillFamily",
    "StorageSkillFamily",
    "ArticulatedSkillFamily",
    "ToolAssemblySkillFamily",
    "SKILL_FAMILIES",
]

