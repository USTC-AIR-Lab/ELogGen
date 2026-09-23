"""Shared OpenArm camera and task-runtime settings."""

from __future__ import annotations

import os
from pathlib import Path

from eloggen.datasets import taskpack_path
from eloggen.pipeline.task_pack import discover_task_packs, load_task_pack, task_pack_supports_difficulty


# Compatibility export for inherited wrapper code. Unlike the old hand-written
# registry, this is discovered from task.yaml manifests.
PUBLISHED_OPENARM_TASKS = frozenset(discover_task_packs())

OPENARM_DIFFICULTY_SUFFIXES = ("_D0", "_D1", "_D2")


REAL_EXP_1_EXTERNAL_CAMERA_RESOLUTIONS = {
    "left_wrist_cam": (240, 424),
    "right_wrist_cam": (240, 424),
    "base_cam": (480, 640),
}


def square_camera_sensor_kwargs(size: int) -> dict[str, int]:
    return {
        "image_height": int(size),
        "image_width": int(size),
    }


def get_real_exp_1_external_sensor_kwargs(camera_name: str) -> dict[str, int]:
    height, width = REAL_EXP_1_EXTERNAL_CAMERA_RESOLUTIONS.get(camera_name, (256, 256))
    return {
        "image_height": height,
        "image_width": width,
    }


def canonical_openarm_task_name(task_name: str | None) -> str | None:
    """Normalize runtime difficulty variants such as ``*_D1`` to a task-pack name."""
    if not task_name:
        return None
    name = str(task_name)
    if name.endswith(OPENARM_DIFFICULTY_SUFFIXES):
        name = name.rsplit("_D", 1)[0]
    return name


def runtime_openarm_task_name(task_name: str | None, default_difficulty: str | None = None) -> str | None:
    """Return the runtime environment identity declared by the task pack.

    Public task identities remain suffix-free. A task only receives ``_D0/_D1/_D2``
    when its manifest declares runtime.difficulty. This removes the old requirement
    to add every new task name to a Python set.
    """
    if task_name is None:
        return None
    name = str(task_name)
    canonical = canonical_openarm_task_name(name)
    if canonical is None or not task_pack_supports_difficulty(canonical):
        return name
    if name.endswith(OPENARM_DIFFICULTY_SUFFIXES):
        return name

    task = load_task_pack(canonical)
    variants = task.difficulty_variants
    difficulty = str(default_difficulty or task.default_difficulty or variants[0]).upper()
    if difficulty not in variants:
        raise ValueError(
            f"Unsupported difficulty {difficulty!r} for {canonical!r}; expected one of {list(variants)!r}"
        )
    return f"{canonical}_{difficulty}"


def openarm_runtime_difficulty(task_name: str | None, default: str = "D0") -> str:
    """Extract ``D0/D1/D2`` from a runtime name, otherwise return task/default difficulty."""
    if task_name:
        name = str(task_name)
        for difficulty in ("D0", "D1", "D2"):
            if name.endswith(f"_{difficulty}"):
                return difficulty
        canonical = canonical_openarm_task_name(name)
        if canonical and task_pack_supports_difficulty(canonical):
            task_default = load_task_pack(canonical).default_difficulty
            if task_default:
                return task_default
    difficulty = str(default).upper()
    if difficulty not in {"D0", "D1", "D2"}:
        raise ValueError(f"Unsupported OpenArm difficulty: {default!r}")
    return difficulty


def openarm_camera_mount_file_candidates(
    task_name: str | None,
    config_dir: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Return task-specific mount files from most- to least-specific.

    ``_D0/_D1/_D2`` task variants resolve back to the canonical task file. Generic
    fallback files remain the caller's responsibility.
    """
    canonical = canonical_openarm_task_name(task_name)
    if canonical is None:
        return []

    candidates = [taskpack_path(canonical) / "cameras.json"]

    runtime_config_dir = (
        Path(config_dir).expanduser().resolve()
        if config_dir is not None
        else Path(__file__).resolve().parents[1] / "configs"
    )
    candidates.append(runtime_config_dir / f"{canonical}_camera_mounts.json")
    return list(dict.fromkeys(str(path) for path in candidates))
