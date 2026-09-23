"""Manual replay-boundary annotation loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .io import load_structured_file


@dataclass(frozen=True)
class ReplayBoundary:
    id: str
    skill: str
    segment: str
    boundary: str
    frame: int
    object: Optional[str] = None
    target: Optional[str] = None
    tool: Optional[str] = None
    handle: Optional[str] = None
    effector: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def skill_family(self) -> str:
        return self.skill.split(".", 1)[0] if "." in self.skill else self.skill

    @property
    def skill_name(self) -> str:
        return self.skill.split(".", 1)[1] if "." in self.skill else self.skill

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReplayBoundary":
        return cls(
            id=str(data["id"]),
            skill=str(data["skill"]),
            segment=str(data["segment"]),
            boundary=str(data["boundary"]),
            frame=int(data["frame"]),
            object=data.get("object"),
            target=data.get("target"),
            tool=data.get("tool"),
            handle=data.get("handle"),
            effector=data.get("effector"),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayAnnotation:
    mode: str = "manual_replay"
    demo_id: str = "demo_0"
    boundaries: List[ReplayBoundary] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReplayAnnotation":
        if "source_annotation" in data:
            data = dict(data["source_annotation"])
        return cls(
            mode=str(data.get("mode", "manual_replay")),
            demo_id=str(data.get("demo_id", "demo_0")),
            boundaries=[ReplayBoundary.from_dict(item) for item in data.get("boundaries", [])],
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "demo_id": self.demo_id,
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
            "metadata": dict(self.metadata),
        }

    def iter_for(
        self,
        *,
        skill_family: Optional[str] = None,
        skill_name: Optional[str] = None,
        segment: Optional[str] = None,
        object: Optional[str] = None,
        target: Optional[str] = None,
        effector: Optional[str] = None,
    ) -> Iterable[ReplayBoundary]:
        for boundary in self.boundaries:
            if skill_family is not None and boundary.skill_family != skill_family:
                continue
            if skill_name is not None and boundary.skill_name != skill_name:
                continue
            if segment is not None and boundary.segment != segment:
                continue
            if object is not None and boundary.object != object:
                continue
            if target is not None and boundary.target not in {target, None}:
                continue
            if effector is not None and boundary.effector != effector:
                continue
            yield boundary


def load_replay_annotation(path: str) -> ReplayAnnotation:
    return ReplayAnnotation.from_dict(load_structured_file(path))


def replay_annotation_from_task_graph(task_graph, demo_id: str = "demo_0") -> ReplayAnnotation:
    boundaries: List[ReplayBoundary] = []
    for unit in task_graph.semantic_units:
        start = unit.metadata.get("MP_end_step")
        end = unit.metadata.get("subtask_term_step")
        if start is None or end is None:
            continue
        if unit.action == "pick":
            skill = "pick_place.grasp_object"
            segment = "grasp"
            entity = unit.object
            target = None
        elif unit.action == "place":
            skill = "pick_place.insert_into_container" if unit.target else "pick_place.place_on_target"
            segment = "place"
            entity = unit.object
            target = unit.target
        else:
            continue
        if entity is None or unit.effector is None:
            continue
        base_id = f"{entity}_{segment}_{unit.effector}"
        boundaries.append(
            ReplayBoundary(
                id=f"{base_id}_start",
                skill=skill,
                segment=segment,
                boundary="replay_start",
                frame=int(start),
                object=entity,
                target=target,
                effector=unit.effector,
                metadata={"source": "generation_task_spec", "phase_key": unit.metadata.get("phase_key")},
            )
        )
        boundaries.append(
            ReplayBoundary(
                id=f"{base_id}_end",
                skill=skill,
                segment=segment,
                boundary="replay_end",
                frame=int(end),
                object=entity,
                target=target,
                effector=unit.effector,
                metadata={"source": "generation_task_spec", "phase_key": unit.metadata.get("phase_key")},
            )
        )
    return ReplayAnnotation(
        mode="task_spec_replay_boundaries",
        demo_id=demo_id,
        boundaries=boundaries,
        metadata={"source": "generation_task_spec", "task_graph": task_graph.name},
    )
