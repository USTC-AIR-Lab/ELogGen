"""Task-pack discovery, runtime metadata, and validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml

from eloggen.datasets import TASKPACKS_ROOT


DIFFICULTY_SUFFIXES = ("_D0", "_D1", "_D2")


@dataclass(frozen=True)
class TaskPack:
    """Portable task definition rooted at one ``task.yaml`` manifest."""

    name: str
    root: Path
    spec: dict[str, Any]

    def path(self, key: str) -> Path:
        value = self.spec.get("files", {}).get(key)
        if not value:
            raise KeyError(f"Task {self.name!r} does not declare files.{key}")
        return (self.root / value).resolve()

    @property
    def family(self) -> str:
        return str(self.spec.get("family") or "generic")

    @property
    def backend(self) -> str:
        return str(self.spec.get("backend") or "omnigibson")

    @property
    def processing(self) -> dict[str, Any]:
        return dict(self.spec.get("processing") or {})

    @property
    def runtime(self) -> dict[str, Any]:
        return dict(self.spec.get("runtime") or {})

    @property
    def pipeline(self) -> dict[str, Any]:
        return dict(self.spec.get("pipeline") or {})

    @property
    def difficulty_variants(self) -> tuple[str, ...]:
        difficulty = dict(self.runtime.get("difficulty") or {})
        variants = difficulty.get("variants") or ()
        return tuple(str(item).upper() for item in variants)

    @property
    def default_difficulty(self) -> str | None:
        difficulty = dict(self.runtime.get("difficulty") or {})
        value = difficulty.get("default")
        return str(value).upper() if value is not None else None


def repository_root() -> Path:
    """Return the source checkout root."""
    return Path(__file__).resolve().parents[3]


def _taskpacks_root(root: Path | None = None) -> Path:
    if root is None:
        return TASKPACKS_ROOT
    candidate = Path(root).expanduser()
    packaged = candidate / "src" / "eloggen" / "datasets" / "taskpacks"
    return packaged if packaged.is_dir() else candidate


def _strip_difficulty_suffix(name: str) -> str:
    value = str(name)
    if value.endswith(DIFFICULTY_SUFFIXES):
        return value.rsplit("_D", 1)[0]
    return value


def discover_task_packs(root: Path | None = None) -> dict[str, Path]:
    """Return all task packs discovered from manifests, keyed by canonical name."""
    taskpacks_root = _taskpacks_root(root)
    if not taskpacks_root.is_dir():
        return {}
    discovered: dict[str, Path] = {}
    for task_root in sorted(path for path in taskpacks_root.iterdir() if path.is_dir()):
        manifest = task_root / "task.yaml"
        if not manifest.is_file():
            continue
        spec = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        name = str(spec.get("name") or task_root.name)
        discovered[name] = task_root.resolve()
    return discovered


def load_task_pack(name: str, root: Path | None = None) -> TaskPack:
    """Load a task pack by canonical name or runtime name such as ``*_D1``."""
    taskpacks_root = _taskpacks_root(root)
    requested = str(name)
    task_root = taskpacks_root / requested
    manifest = task_root / "task.yaml"

    if not manifest.is_file():
        canonical = _strip_difficulty_suffix(requested)
        task_root = taskpacks_root / canonical
        manifest = task_root / "task.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(f"Task pack not found: {manifest}")

    spec = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    canonical_name = str(spec.get("name") or task_root.name)
    return TaskPack(name=canonical_name, root=task_root.resolve(), spec=spec)


def task_pack_exists(name: str, root: Path | None = None) -> bool:
    try:
        load_task_pack(name, root=root)
    except FileNotFoundError:
        return False
    return True


def task_pack_supports_difficulty(name: str, root: Path | None = None) -> bool:
    try:
        task = load_task_pack(name, root=root)
    except FileNotFoundError:
        return False
    return bool(task.difficulty_variants)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_task_pack(task: TaskPack, verify_hashes: bool = True) -> list[str]:
    errors: list[str] = []
    manifest_name = task.spec.get("name")
    if manifest_name and str(manifest_name) != task.name:
        errors.append(f"Manifest name {manifest_name!r} does not match task name {task.name!r}")

    for key in ("generation", "scene", "bddl", "cameras", "source", "processed_source"):
        try:
            path = task.path(key)
        except KeyError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file():
            errors.append(f"Missing files.{key}: {path}")

    processing = task.processing
    interface_type = processing.get("interface_type")
    if not interface_type:
        errors.append(f"Task {task.name!r} does not declare processing.interface_type")

    variants = task.difficulty_variants
    default_difficulty = task.default_difficulty
    if default_difficulty is not None and default_difficulty not in variants:
        errors.append(
            f"runtime.difficulty.default {default_difficulty!r} is not in variants {list(variants)!r}"
        )

    if verify_hashes:
        hashes = task.spec.get("sha256", {})
        for key, expected in hashes.items():
            try:
                path = task.path(key)
            except KeyError:
                continue
            if path.is_file() and sha256(path) != expected:
                errors.append(f"SHA256 mismatch for files.{key}: {path}")
    return errors
