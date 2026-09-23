"""Stable paths for ElogGen packaged data."""

from __future__ import annotations

from pathlib import Path
from typing import Union


DATASETS_ROOT = Path(__file__).resolve().parent
TASKPACKS_ROOT = DATASETS_ROOT / "taskpacks"
EXAMPLES_ROOT = DATASETS_ROOT / "examples"
MANIFEST_PATH = DATASETS_ROOT / "manifest.json"
PathLike = Union[str, Path]


def taskpack_path(name: str) -> Path:
    """Return a bundled task-pack directory by name."""
    return TASKPACKS_ROOT / str(name)


def example_path(name: str) -> Path:
    """Return a bundled example file by relative name."""
    return EXAMPLES_ROOT / str(name)


def resolve_dataset_path(
    value: PathLike | None,
    *,
    base_dir: PathLike | None = None,
) -> str | None:
    """Resolve packaged ``taskpacks/`` and ``examples/`` paths."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    if base_dir is not None:
        candidate = (Path(base_dir).expanduser() / path).resolve()
        if candidate.exists():
            return str(candidate)

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    parts = path.parts
    if parts:
        aliases = {
            "taskpacks": TASKPACKS_ROOT,
            "examples": EXAMPLES_ROOT,
            "datasets": DATASETS_ROOT,
        }
        alias_root = aliases.get(parts[0])
        if alias_root is not None:
            return str(alias_root.joinpath(*parts[1:]).resolve())

    return str(path)
