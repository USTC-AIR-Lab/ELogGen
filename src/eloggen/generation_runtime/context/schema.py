"""Data structures for context boundary annotation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageBoundary:
    action: str
    stage_type: str
    substage_id: int
    start: int
    end: int
    active_arm: str = "left"
    active_object: Optional[str] = None
    passive_object: Optional[str] = None
    object_label_zh: Optional[str] = None
    object_label_en: Optional[str] = None
    target_object: Optional[str] = None
    target_label_zh: Optional[str] = None
    target_label_en: Optional[str] = None
    instruction_zh: Optional[str] = None
    instruction_en: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoundarySet:
    demo_key: str
    task: str
    detector: str
    action: str
    num_frames: int
    active_arm: str
    active_object: str
    passive_object: str
    object_label_zh: str
    object_label_en: str
    stages: List[StageBoundary]
    events: Dict[str, int]
    metrics: Dict[str, Any] = field(default_factory=dict)
    target_object: Optional[str] = None
    target_label_zh: Optional[str] = None
    target_label_en: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["stages"] = [stage.to_dict() for stage in self.stages]
        return data


@dataclass
class BoundaryCheck:
    name: str
    manual: Optional[int]
    detected: Optional[int]
    diff: Optional[int]
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoundaryReport:
    demo_key: str
    task: str
    detector: str
    status: str
    checks: List[BoundaryCheck]
    manual_steps: Dict[str, Any]
    detected_events: Dict[str, int]
    max_diff: int

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["checks"] = [check.to_dict() for check in self.checks]
        return data
