"""LeRobot/HDF5 export-stage adapters for ElogGen pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from eloggen.simulation.omnigibson import activate


@dataclass
class LeRobotExportStatus:
    ok: bool
    module: str = "eloggen.generation_runtime.commands.export"
    error_type: Optional[str] = None
    error: Optional[str] = None
    has_lerobot_dataset_writer: bool = False
    has_validate_lerobot_dataset: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_lerobot_export_imports() -> LeRobotExportStatus:
    """Check that the LeRobot export/conversion backend is importable."""
    try:
        activate()
        from eloggen.generation_runtime.commands.export import (
            LeRobotDatasetWriter,
            validate_lerobot_dataset,
        )  # noqa: F401
    except Exception as exc:
        return LeRobotExportStatus(
            ok=False,
            error_type=type(exc).__name__,
            error=str(exc),
        )
    return LeRobotExportStatus(
        ok=True,
        has_lerobot_dataset_writer=True,
        has_validate_lerobot_dataset=callable(validate_lerobot_dataset),
    )


def lerobot_export_call_plan(config, run_dir) -> Dict[str, Any]:
    """Build a lightweight call plan for the LeRobot export stage.

    Does NOT launch a real conversion — returns the plan
    that an associated executor or manual command would follow.
    """
    export_cfg = config.export.lerobot
    output = export_cfg.output or str(run_dir.path("lerobot"))
    return {
        "adapter": "eloggen.pipeline.export",
        "backend": "eloggen.generation_runtime.commands.export",
        "generated_hdf5": config.export.generated_hdf5
        or str(run_dir.path("generated_hdf5", "demo.hdf5")),
        "output": output,
        "repo_id": export_cfg.repo_id or config.task.name,
        "task": export_cfg.task or config.task.name,
        "robot_type": export_cfg.robot_type,
        "fps": export_cfg.fps,
        "ordered_tasks": export_cfg.ordered_tasks,
        "resume": export_cfg.resume,
        "overwrite": export_cfg.overwrite,
    }


def export_hdf5_call_plan(config, run_dir) -> Dict[str, Any]:
    """Build a lightweight call plan for a HDF5 merge/export stage."""
    return {
        "adapter": "eloggen.pipeline.export",
        "backend": "eloggen.generation_runtime.commands.merge",
        "input_folder": config.generation.folder or str(run_dir.path("generated_hdf5")),
        "output": str(run_dir.path("generated_hdf5", "demo.hdf5")),
    }
