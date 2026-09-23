"""Core dataclasses for ElogGen task graphs and execution plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Predicate:
    name: str
    args: List[str] = field(default_factory=list)
    value: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Predicate":
        return cls(
            name=str(data["name"]),
            args=[str(item) for item in data.get("args", [])],
            value=bool(data.get("value", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticRole:
    role: str
    entity: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticRole":
        return cls(role=str(data["role"]), entity=str(data["entity"]))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalBoundary:
    start_frame: int
    end_frame: int
    source_demo_id: str = "demo_0"

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError(f"Invalid frame range [{self.start_frame}, {self.end_frame})")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalBoundary":
        return cls(
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            source_demo_id=str(data.get("source_demo_id", "demo_0")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceTrajectorySlice:
    hdf5_path: str
    demo_key: str
    start_frame: int
    end_frame: int
    dataset_keys: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceTrajectorySlice":
        return cls(
            hdf5_path=str(data["hdf5_path"]),
            demo_key=str(data.get("demo_key", "demo_0")),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            dataset_keys=[str(item) for item in data.get("dataset_keys", [])],
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InteractionSegment:
    id: str
    alpha: str
    roles: List[SemanticRole]
    effectors: List[str]
    boundary: TemporalBoundary
    source_slice: SourceTrajectorySlice
    preconditions: List[Predicate] = field(default_factory=list)
    effects: List[Predicate] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionSegment":
        return cls(
            id=str(data["id"]),
            alpha=str(data["alpha"]),
            roles=[SemanticRole.from_dict(item) for item in data.get("roles", [])],
            effectors=[str(item) for item in data.get("effectors", [])],
            boundary=TemporalBoundary.from_dict(data["boundary"]),
            source_slice=SourceTrajectorySlice.from_dict(data["source_slice"]),
            preconditions=[Predicate.from_dict(item) for item in data.get("preconditions", [])],
            effects=[Predicate.from_dict(item) for item in data.get("effects", [])],
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "alpha": self.alpha,
            "roles": [item.to_dict() for item in self.roles],
            "effectors": list(self.effectors),
            "boundary": self.boundary.to_dict(),
            "source_slice": self.source_slice.to_dict(),
            "preconditions": [item.to_dict() for item in self.preconditions],
            "effects": [item.to_dict() for item in self.effects],
            "metadata": dict(self.metadata),
        }


@dataclass
class SourceDemo:
    path: str
    kind: str = "source_hdf5"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceDemo":
        return cls(path=str(data["path"]), kind=str(data.get("kind", "source_hdf5")), metadata=dict(data.get("metadata") or {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectSpec:
    name: str
    label: Optional[str] = None
    entity_type: str = "object"
    allowed_effectors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectSpec":
        return cls(
            name=str(data["name"]),
            label=data.get("label"),
            entity_type=str(data.get("entity_type", "object")),
            allowed_effectors=[str(item) for item in data.get("allowed_effectors", [])],
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TargetSpec:
    name: str
    label: Optional[str] = None
    entity_type: str = "target"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetSpec":
        return cls(
            name=str(data["name"]),
            label=data.get("label"),
            entity_type=str(data.get("entity_type", "target")),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EffectorSpec:
    name: str
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EffectorSpec":
        return cls(name=str(data["name"]), label=data.get("label"), metadata=dict(data.get("metadata") or {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GoalSpec:
    predicate: str
    args: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalSpec":
        return cls(
            predicate=str(data["predicate"]),
            args=[str(item) for item in data.get("args", [])],
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintSpec:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintSpec":
        if "type" not in data:
            raise ValueError(f"Constraint is missing type: {data}")
        params = {key: value for key, value in data.items() if key != "type"}
        params.update(dict(data.get("params") or {}))
        return cls(type=str(data["type"]), params=params)

    def to_dict(self) -> Dict[str, Any]:
        data = {"type": self.type}
        data.update(self.params)
        return data


@dataclass
class ObjectBinding:
    object: str
    target: Optional[str] = None
    allowed_effectors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectBinding":
        return cls(
            object=str(data["object"]),
            target=data.get("target"),
            allowed_effectors=[str(item) for item in data.get("allowed_effectors", [])],
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticUnit:
    id: str
    action: str
    object: Optional[str] = None
    target: Optional[str] = None
    effector: Optional[str] = None
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    stage_template: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticUnit":
        return cls(
            id=str(data["id"]),
            action=str(data["action"]),
            object=data.get("object"),
            target=data.get("target"),
            effector=data.get("effector"),
            preconditions=[str(item) for item in data.get("preconditions", [])],
            effects=[str(item) for item in data.get("effects", [])],
            stage_template=[str(item) for item in data.get("stage_template", [])],
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskGraph:
    name: str
    source_demos: List[SourceDemo] = field(default_factory=list)
    objects: List[ObjectSpec] = field(default_factory=list)
    targets: List[TargetSpec] = field(default_factory=list)
    effectors: List[EffectorSpec] = field(default_factory=list)
    goals: List[GoalSpec] = field(default_factory=list)
    constraints: List[ConstraintSpec] = field(default_factory=list)
    bindings: List[ObjectBinding] = field(default_factory=list)
    semantic_units: List[SemanticUnit] = field(default_factory=list)
    skill_families: List[str] = field(default_factory=list)
    plan_aliases: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGraph":
        return cls(
            name=str(data["name"]),
            source_demos=[SourceDemo.from_dict(item) for item in data.get("source_demos", [])],
            objects=[ObjectSpec.from_dict(item) for item in data.get("objects", [])],
            targets=[TargetSpec.from_dict(item) for item in data.get("targets", [])],
            effectors=[EffectorSpec.from_dict(item) for item in data.get("effectors", [])],
            goals=[GoalSpec.from_dict(item) for item in data.get("goals", [])],
            constraints=[ConstraintSpec.from_dict(item) for item in data.get("constraints", [])],
            bindings=[ObjectBinding.from_dict(item) for item in data.get("bindings", [])],
            semantic_units=[SemanticUnit.from_dict(item) for item in data.get("semantic_units", [])],
            skill_families=[str(item) for item in data.get("skill_families", [])],
            plan_aliases=dict(data.get("plan_aliases") or {}),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_demos": [item.to_dict() for item in self.source_demos],
            "objects": [item.to_dict() for item in self.objects],
            "targets": [item.to_dict() for item in self.targets],
            "effectors": [item.to_dict() for item in self.effectors],
            "goals": [item.to_dict() for item in self.goals],
            "constraints": [item.to_dict() for item in self.constraints],
            "bindings": [item.to_dict() for item in self.bindings],
            "semantic_units": [item.to_dict() for item in self.semantic_units],
            "skill_families": list(self.skill_families),
            "plan_aliases": dict(self.plan_aliases),
            "metadata": dict(self.metadata),
        }

    def object_names(self) -> List[str]:
        return [item.name for item in self.objects]

    def target_names(self) -> List[str]:
        return [item.name for item in self.targets]

    def effector_names(self) -> List[str]:
        return [item.name for item in self.effectors]

    def binding_for_object(self, object_name: str) -> Optional[ObjectBinding]:
        for binding in self.bindings:
            if binding.object == object_name:
                return binding
        return None


@dataclass
class ExecutionPlan:
    id: str
    object_order: List[str] = field(default_factory=list)
    effector_assignment: Dict[str, str] = field(default_factory=dict)
    stage_sequence: List[SemanticUnit] = field(default_factory=list)
    phase_order: List[str] = field(default_factory=list)
    constraint_results: Dict[str, bool] = field(default_factory=dict)
    role_bindings: Dict[str, str] = field(default_factory=dict)
    coordination_mode: str = "sequential"
    dependency_edges: List[List[str]] = field(default_factory=list)
    resource_schedule: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPlan":
        return cls(
            id=str(data["id"]),
            object_order=[str(item) for item in data.get("object_order", [])],
            effector_assignment={str(key): str(value) for key, value in dict(data.get("effector_assignment") or {}).items()},
            stage_sequence=[SemanticUnit.from_dict(item) for item in data.get("stage_sequence", [])],
            phase_order=[str(item) for item in data.get("phase_order", [])],
            constraint_results={str(key): bool(value) for key, value in dict(data.get("constraint_results") or {}).items()},
            role_bindings={str(key): str(value) for key, value in dict(data.get("role_bindings") or {}).items()},
            coordination_mode=str(data.get("coordination_mode", "sequential")),
            dependency_edges=[[str(value) for value in edge] for edge in data.get("dependency_edges", [])],
            resource_schedule={
                str(key): [str(value) for value in values]
                for key, values in dict(data.get("resource_schedule") or {}).items()
            },
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object_order": list(self.object_order),
            "effector_assignment": dict(self.effector_assignment),
            "stage_sequence": [item.to_dict() for item in self.stage_sequence],
            "phase_order": list(self.phase_order),
            "constraint_results": dict(self.constraint_results),
            "role_bindings": dict(self.role_bindings),
            "coordination_mode": self.coordination_mode,
            "dependency_edges": [list(edge) for edge in self.dependency_edges],
            "resource_schedule": {key: list(values) for key, values in self.resource_schedule.items()},
            "metadata": dict(self.metadata),
        }
