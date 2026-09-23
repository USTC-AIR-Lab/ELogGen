"""Generation-stage adapter for ElogGen generation runtime-backed ElogGen pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
from eloggen.simulation.omnigibson import activate


@dataclass
class GenerationImportStatus:
    ok: bool
    module: str = "eloggen.generation_runtime.commands.generate"
    error_type: Optional[str] = None
    error: Optional[str] = None
    has_generate_dataset: bool = False
    has_config_factory: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_generation_imports() -> GenerationImportStatus:
    try:
        activate()
        from eloggen.generation_runtime.config.base import config_factory  # noqa: F401
        from eloggen.generation_runtime.commands import generate as generate_module
    except Exception as exc:
        return GenerationImportStatus(
            ok=False,
            error_type=type(exc).__name__,
            error=str(exc),
        )
    return GenerationImportStatus(
        ok=True,
        has_generate_dataset=hasattr(generate_module, "generate_dataset"),
        has_config_factory=True,
    )


def generation_call_plan(config, run_dir) -> Dict[str, Any]:
    output_folder = config.generation.folder or str(run_dir.path("generated_hdf5"))
    return {
        "adapter": "eloggen.pipeline.generation",
        "backend": config.generation.backend,
        "base_config": config.task.base_config,
        "source_demo": config.task.processed_source_demo or config.task.source_demo,
        "raw_source_demo": config.task.source_demo,
        "scene_file": config.task.scene_file,
        "output_folder": output_folder,
        "output_format": config.generation.output_format,
        "num_demos": config.generation.num_demos,
        "bimanual": config.generation.bimanual,
        "robot_type": config.generation.robot_type,
        "seed": config.generation.seed,
        "difficulty": config.generation.difficulty,
        "grasp_order": config.generation.grasp_order,
        "print_stage_type": config.generation.print_stage_type,
        "headless": config.generation.headless,
        "auto_remove_exp": config.generation.auto_remove_exp,
        "render": config.generation.render,
        "no_video_save": config.generation.no_video_save,
        "video_skip": config.generation.video_skip,
        "video_fps": config.generation.video_fps,
        "render_image_names": config.generation.render_image_names,
        "pause_subtask": config.generation.pause_subtask,
        "enable_marker_vis": config.generation.enable_marker_vis,
        "ds_ratio": config.generation.ds_ratio,
        "no_partial_tasks": config.generation.no_partial_tasks,
        "baseline": config.generation.baseline,
        "subtask_order": config.generation.subtask_order,
        "grasp_order_quota": config.generation.grasp_order_quota,
    }
