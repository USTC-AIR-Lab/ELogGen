"""Pipeline configuration schema for ElogGen."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from eloggen.datasets import resolve_dataset_path
from .io import load_structured_file


@dataclass
class TaskPipelineConfig:
    name: str
    family: str
    base_config: str
    source_demo: Optional[str] = None
    processed_source_demo: Optional[str] = None
    scene_file: Optional[str] = None
    task_graph: Optional[str] = None


@dataclass
class CollectionConfig:
    enabled: bool = False
    backend: str = "eloggen.collection"
    task_name: Optional[str] = None
    output: Optional[str] = None


@dataclass
class AnnotationConfig:
    boundary_source: str = "generation_task_spec"
    manual_replay_required: bool = True
    replay_step_fields: Dict[str, str] = field(
        default_factory=lambda: {"mp_end": "MP_end_step", "replay_end": "subtask_term_step"}
    )
    annotation_file: Optional[str] = None


@dataclass
class LogicConfig:
    max_plans: int = 16
    plan_id: Optional[str] = None
    retarget_policy: str = "source_effector_first"
    nearest_principle: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "hard_constraint": True, "unresolved_policy": "mark_unresolved"}
    )


@dataclass
class GenerationConfig:
    backend: str = "eloggen.generation_runtime.commands.generate"
    output_format: str = "hdf5"
    num_demos: int = 1
    bimanual: bool = True
    robot_type: str = "OpenArm"
    folder: Optional[str] = None
    seed: Optional[int] = None
    difficulty: Optional[str] = None
    grasp_order: Optional[str] = None
    grasp_order_quota: Optional[str] = None
    subtask_order: Optional[str] = None
    baseline: Optional[str] = None
    print_stage_type: bool = False
    headless: bool = True
    auto_remove_exp: bool = False
    render: bool = False
    no_video_save: bool = False
    video_skip: int = 5
    video_fps: int = 30
    render_image_names: Optional[List[str]] = None
    pause_subtask: bool = False
    enable_marker_vis: bool = False
    ds_ratio: int = 1
    no_partial_tasks: bool = False


@dataclass
class LeRobotExportConfig:
    enabled: bool = False
    output: Optional[str] = None
    repo_id: Optional[str] = None
    task: Optional[str] = None
    robot_type: str = "openarm"
    fps: Optional[int] = None
    ordered_tasks: bool = False
    resume: bool = False
    overwrite: bool = False
    checkpoint_every: int = 10


@dataclass
class ExportConfig:
    generated_hdf5: Optional[str] = None
    lerobot: LeRobotExportConfig = field(default_factory=LeRobotExportConfig)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


@dataclass
class PipelineConfig:
    task: TaskPipelineConfig
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    logic: LogicConfig = field(default_factory=LogicConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        path = Path(resolve_dataset_path(path) or path).expanduser().resolve()
        data = load_structured_file(path)
        if not isinstance(data, dict):
            raise ValueError(f"Pipeline config must be a mapping: {path}")
        return cls.from_dict(data, base_dir=path.parent)

    @classmethod
    def from_task_pack(cls, task: str | Any) -> "PipelineConfig":
        """Build the runnable pipeline from a task pack without an example config.

        The task manifest owns task identity and file locations. ``pipeline`` inside
        task.yaml only contains optional experiment defaults; CLI flags can still
        override those defaults for an individual run.
        """
        from .task_pack import load_task_pack

        task_pack = load_task_pack(task) if isinstance(task, str) else task
        runtime = task_pack.runtime
        robot = dict(runtime.get("robot") or {})
        collection_spec = dict(task_pack.spec.get("collection") or {})

        defaults: Dict[str, Any] = {
            "task": {
                "name": task_pack.name,
                "family": task_pack.family,
                "base_config": str(task_pack.path("generation")),
                "source_demo": str(task_pack.path("source")),
                "processed_source_demo": str(task_pack.path("processed_source")),
                "scene_file": str(task_pack.path("scene")),
            },
            "collection": {
                "enabled": False,
                "backend": "plugin",
                "task_name": task_pack.name,
                "output": f"runs/{task_pack.name}/source/{task_pack.name}.hdf5",
            },
            "annotation": {
                "boundary_source": "generation_task_spec",
                "manual_replay_required": True,
                "replay_step_fields": {
                    "mp_end": "MP_end_step",
                    "replay_end": "subtask_term_step",
                },
            },
            "logic": {
                "max_plans": 16,
                "plan_id": None,
                "retarget_policy": "source_effector_first",
                "nearest_principle": {
                    "enabled": True,
                    "hard_constraint": True,
                    "unresolved_policy": "mark_unresolved",
                },
            },
            "generation": {
                "backend": "eloggen.generation_runtime.commands.generate",
                "output_format": "hdf5",
                "num_demos": 1,
                "bimanual": bool(robot.get("bimanual", True)),
                "robot_type": str(robot.get("type") or "OpenArm"),
                "folder": f"runs/{task_pack.name}/generated_hdf5",
                "seed": 1,
                "difficulty": task_pack.default_difficulty,
                "grasp_order": "task_spec",
                "print_stage_type": True,
            },
            "export": {
                "generated_hdf5": None,
                "lerobot": {
                    "enabled": False,
                    "output": f"runs/{task_pack.name}/lerobot",
                    "repo_id": f"openarm/{task_pack.name}",
                    "task": task_pack.name,
                    "robot_type": str(robot.get("lerobot_type") or "openarm"),
                    "fps": 30,
                    "ordered_tasks": True,
                    "resume": False,
                    "overwrite": False,
                },
            },
            "metadata": {
                "task_pack": task_pack.name,
                "task_pack_root": str(task_pack.root),
                "task_pack_driven": True,
            },
        }

        if collection_spec.get("backend"):
            defaults["collection"]["backend"] = collection_spec["backend"]

        data = _deep_merge(defaults, task_pack.pipeline)
        # File locations are task-owned and cannot be shadowed by experiment defaults.
        data["task"].update(defaults["task"])
        return cls.from_dict(data)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        base_dir: str | Path | None = None,
    ) -> "PipelineConfig":
        task_data = dict(data.get("task") or {})
        if not task_data:
            raise ValueError("Pipeline config requires a task section")
        for key in (
            "base_config",
            "source_demo",
            "processed_source_demo",
            "scene_file",
            "task_graph",
        ):
            if task_data.get(key):
                task_data[key] = resolve_dataset_path(
                    task_data[key], base_dir=base_dir
                )
        export_data = dict(data.get("export") or {})
        return cls(
            task=TaskPipelineConfig(**task_data),
            collection=CollectionConfig(**dict(data.get("collection") or {})),
            annotation=AnnotationConfig(**dict(data.get("annotation") or {})),
            logic=LogicConfig(**dict(data.get("logic") or {})),
            generation=GenerationConfig(**dict(data.get("generation") or {})),
            export=ExportConfig(
                generated_hdf5=export_data.get("generated_hdf5"),
                lerobot=LeRobotExportConfig(**dict(export_data.get("lerobot") or {})),
            ),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def default_stages(self) -> List[str]:
        stages = ["task_graph", "logic", "recipe", "export_config"]
        if self.collection.enabled:
            stages.insert(0, "collect")
        if self.generation.num_demos > 0:
            stages.append("generate")
        if self.export.lerobot.enabled:
            stages.append("export_lerobot")
        return stages
