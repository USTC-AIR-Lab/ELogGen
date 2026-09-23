"""Execute ElogGen generation plans with the OmniGibson runtime."""

from __future__ import annotations

import json

import h5py
from pathlib import Path
from typing import Any

from eloggen.simulation.omnigibson import activate
from eloggen.datasets import resolve_dataset_path
from eloggen.generation_runtime.simulation.camera_config import runtime_openarm_task_name


def load_runtime_config_payload(config_path: str | Path) -> dict[str, Any]:
    """Return keys accepted by the key-locked generation configuration.

    ElogGen provenance remains in the exported JSON on disk, but it is not part
    of the runtime schema and must not be passed to ``EG_Config.update``.
    """
    payload = json.loads(Path(config_path).resolve().read_text(encoding="utf-8"))
    payload.pop("meta", None)
    payload.pop("eloggen", None)
    return payload


def runtime_environment_name(task_name: str, difficulty: str | None = None) -> str:
    """Resolve a public task-pack name to its simulator runtime identity."""
    return runtime_openarm_task_name(task_name, default_difficulty=difficulty) or str(task_name)


def validate_generation_source(source_demo: str | Path) -> Path:
    """Validate generation metadata before starting the expensive simulator."""
    path = Path(resolve_dataset_path(source_demo) or source_demo).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Generation source HDF5 not found: {path}")
    with h5py.File(path, "r") as stream:
        data = stream.get("data")
        demos = [] if data is None else [key for key in data.keys() if str(key).startswith("demo_")]
        missing = [key for key in demos if "datagen_info" not in data[key]] if data is not None else []
        if not demos or missing:
            detail = f"; demos missing datagen_info: {', '.join(missing)}" if missing else ""
            raise ValueError(
                f"Generation requires a processed source HDF5 containing data/demo_*/datagen_info: {path}{detail}. "
                "Use task.processed_source_demo, not the raw collection HDF5."
            )
    return path


class GenerationExecutor:
    """Execute an exported ElogGen recipe."""

    def execute(
        self,
        config_path: str | Path,
        *,
        task_name: str,
        source_demo: str | Path,
        scene_file: str | Path | None,
        output_folder: str | Path,
        num_demos: int = 1,
        seed: int | None = None,
        difficulty: str | None = None,
        bimanual: bool = True,
        robot_type: str = "OpenArm",
        grasp_order: str = "task_spec",
        print_stage_type: bool = True,
        output_format: str = "hdf5",
        headless: bool = True,
        auto_remove_exp: bool = False,
        render: bool = False,
        no_video_save: bool = False,
        video_skip: int = 5,
        render_image_names: list[str] | None = None,
        pause_subtask: bool = False,
        enable_marker_vis: bool = False,
        ds_ratio: int = 1,
        no_partial_tasks: bool = False,
        baseline: str | None = None,
        subtask_order: str | None = None,
        video_fps: int = 30,
        grasp_order_quota: str | None = None,
        lerobot_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_demo = validate_generation_source(source_demo)
        if scene_file is not None:
            scene_file = Path(resolve_dataset_path(scene_file) or scene_file).resolve()
            if not scene_file.is_file():
                raise FileNotFoundError(f"Task scene file not found: {scene_file}")

        activate()
        # Register manifest-owned runtime metadata before the inherited wrapper
        # creates the environment. This keeps TASK_CONFIGS compatibility while
        # allowing new ordinary task packs to avoid Python registration code.
        from eloggen.generation_runtime.simulation.taskpack_runtime import ensure_task_runtime_registered
        ensure_task_runtime_registered(task_name)

        from eloggen.generation_runtime.config.base import config_factory
        from eloggen.generation_runtime.commands.generate import generate_dataset

        config_path = Path(config_path).resolve()
        ext_cfg = load_runtime_config_payload(config_path)
        generation_config = config_factory(ext_cfg["name"], config_type=ext_cfg["type"])
        with generation_config.values_unlocked():
            generation_config.update(ext_cfg)
            source_subtasks = set(generation_config.task.task_spec.keys())
            selected_subtasks = set(ext_cfg["task"]["task_spec"].keys())
            for subtask in source_subtasks - selected_subtasks:
                del generation_config.task.task_spec[subtask]
            generation_config.experiment.task.name = runtime_environment_name(task_name, difficulty=difficulty)
            generation_config.experiment.source.dataset_path = str(Path(source_demo).resolve())
            generation_config.experiment.generation.path = str(Path(output_folder).resolve())
            generation_config.experiment.generation.num_trials = int(num_demos)
            if seed is not None:
                generation_config.experiment.seed = int(seed)

        lerobot = dict(lerobot_options or {})
        return generate_dataset(
            generation_config=generation_config,
            bimanual=bimanual,
            headless=headless,
            auto_remove_exp=auto_remove_exp,
            render=render,
            no_save_video=no_video_save,
            video_skip=video_skip,
            render_image_names=render_image_names,
            pause_subtask=pause_subtask,
            enable_marker_vis=enable_marker_vis,
            ds_ratio=ds_ratio,
            no_partial_tasks=no_partial_tasks,
            baseline=baseline,
            subtask_order=subtask_order,
            video_fps=video_fps,
            grasp_order_quota=grasp_order_quota,
            robot_type=robot_type,
            print_stage_type=print_stage_type,
            grasp_order=grasp_order,
            output_format=output_format,
            lerobot_output=lerobot.get("output"),
            lerobot_repo_id=lerobot.get("repo_id"),
            lerobot_task=lerobot.get("task"),
            lerobot_robot_type=lerobot.get("robot_type", "openarm"),
            lerobot_fps=lerobot.get("fps"),
            lerobot_ordered_tasks=lerobot.get("ordered_tasks", False),
            lerobot_overwrite=lerobot.get("overwrite", False),
            lerobot_resume=lerobot.get("resume", False),
            lerobot_checkpoint_every=lerobot.get("checkpoint_every", 10),
            scene_file=str(scene_file) if scene_file is not None else None,
        )
