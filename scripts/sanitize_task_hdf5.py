#!/usr/bin/env python3
"""Replace legacy host paths in task HDF5 metadata with task-relative paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil

import h5py


ROOT = Path(__file__).resolve().parents[1]
TASKPACK_ROOT = ROOT / "src" / "eloggen" / "data" / "taskpacks"
MACHINE_PATH = re.compile(r"/(?:home|media)/openarm(?:/|$)")
LEGACY_PROJECT = "/" + "home" + "/openarm/Open_Genbench"
SCENE_PATH = re.compile(
    re.escape(LEGACY_PROJECT) + r"/\.momagen_scene_cache/[^\"\s,}]+\.json"
)
SOURCE_PATH = re.compile(
    re.escape(LEGACY_PROJECT) + r"/momagen/datasets/source_og/[^\"\s,}]+\.hdf5"
)


def replacement(value: str, task_name: str) -> str:
    value = SCENE_PATH.sub(
        f"src/eloggen/datasets/taskpacks/{task_name}/scene.json", value
    )
    return SOURCE_PATH.sub(
        f"src/eloggen/datasets/taskpacks/{task_name}/source/source.hdf5", value
    )


def iter_objects(handle: h5py.File):
    yield "/", handle
    objects = []
    handle.visititems(lambda name, obj: objects.append((name, obj)))
    yield from objects


def inspect(path: Path, task_name: str, *, apply: bool) -> tuple[int, list[str]]:
    changed = 0
    remaining: list[str] = []
    mode = "r+" if apply else "r"
    with h5py.File(path, mode) as handle:
        for object_name, obj in iter_objects(handle):
            for key in tuple(obj.attrs.keys()):
                value = obj.attrs[key]
                if not isinstance(value, str):
                    continue
                updated = replacement(value, task_name)
                if updated != value:
                    changed += 1
                    if apply:
                        obj.attrs.modify(key, updated)
                if MACHINE_PATH.search(updated):
                    remaining.append(f"{object_name}:{key}")
    return changed, remaining


def repack(path: Path) -> None:
    """Rewrite live HDF5 content so deleted metadata cannot leak host paths."""
    temporary = path.with_name(f".{path.name}.repack-{os.getpid()}")
    try:
        with h5py.File(path, "r") as source, h5py.File(temporary, "w") as target:
            for key, value in source.attrs.items():
                target.attrs[key] = value
            for name in source:
                source.copy(name, target)
            target.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--repack",
        action="store_true",
        help="rewrite live content to remove deleted/free-space remnants",
    )
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    if args.apply and args.backup_dir is None:
        parser.error("--apply requires --backup-dir")
    if args.repack and not args.apply:
        parser.error("--repack requires --apply")

    failed = False
    for path in sorted(TASKPACK_ROOT.glob("*/source/*.hdf5")):
        task_name = path.parents[1].name
        if args.apply:
            backup = args.backup_dir.expanduser().resolve() / task_name / path.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                raise FileExistsError(f"refusing to overwrite backup: {backup}")
            shutil.copy2(path, backup)
        changed, remaining = inspect(path, task_name, apply=args.apply)
        if args.repack:
            repack(path)
        status = "updated" if args.apply else "would-update"
        repack_status = " repacked=yes" if args.repack else ""
        print(
            f"[{status}] {path.relative_to(ROOT)} attributes={changed}{repack_status}"
        )
        if remaining:
            failed = True
            print(f"[ERROR] unresolved machine paths: {', '.join(remaining)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
