"""Run-directory management for ElogGen pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from .io import write_json


RUN_SUBDIRS = (
    "source",
    "processed_source",
    "execution_logic",
    "recipes",
    "generation_configs",
    "generated_hdf5",
    "lerobot",
    "context_reports",
    "logs",
)


@dataclass
class PipelineRunDir:
    root: Path

    @classmethod
    def create(cls, task_name: str, run_dir: str | None = None, base_dir: str = "runs") -> "PipelineRunDir":
        if run_dir:
            root = Path(run_dir)
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            root = Path(base_dir) / task_name / stamp
        root.mkdir(parents=True, exist_ok=True)
        for subdir in RUN_SUBDIRS:
            (root / subdir).mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def write_config(self, config: Dict[str, Any]) -> Path:
        path = self.path("config_resolved.json")
        write_json(path, config)
        return path

    def write_status(self, status: Dict[str, Any]) -> Path:
        path = self.path("status.json")
        write_json(path, status)
        return path

    def stage_path(self, stage: str, filename: str) -> Path:
        mapping = {
            "task_graph": "",
            "logic": "execution_logic",
            "recipe": "recipes",
            "export_config": "generation_configs",
            "generate": "generated_hdf5",
            "export_hdf5": "generated_hdf5",
            "export_lerobot": "lerobot",
            "validate": "context_reports",
            "collect": "source",
            "prepare_source": "processed_source",
            "annotate": "context_reports",
        }
        subdir = mapping.get(stage, stage)
        return self.path(subdir, filename) if subdir else self.path(filename)

    def relative_outputs(self, paths: Iterable[Path]) -> list[str]:
        result = []
        for path in paths:
            try:
                result.append(str(path.relative_to(self.root)))
            except ValueError:
                result.append(str(path))
        return result
