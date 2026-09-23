#!/usr/bin/env python3
"""Audit the prospective ElogGen public tree without modifying it."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import h5py


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {
    ".baseline",
    ".eloggen",
    ".generation_scene_cache",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "build",
    "cache",
    "deps",
    "dist",
    "outputs",
    "runs",
    "third_party",
    "tmp",
    # Legacy external-source locations from older ElogGen layouts.
    "BEHAVIOR-1K",
    "robomimic",
}
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
MACHINE_PATH = re.compile(r"/(?:home|media)/openarm(?:/|$)")


def hdf5_has_machine_path(path: Path) -> bool:
    with h5py.File(path, "r") as handle:
        objects = [handle]
        handle.visititems(lambda _name, obj: objects.append(obj))
        return any(
            isinstance(value, str) and MACHINE_PATH.search(value)
            for obj in objects
            for value in obj.attrs.values()
        )


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in IGNORED_PARTS for part in relative.parts):
        return False
    if relative in {
        Path(".eloggen.local"),
        Path("AGENTS.md"),
    }:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-doc-paths", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    for name in (
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/momagen-NVIDIA.txt",
        "LICENSES/robomimic-MIT.txt",
        ".eloggen.local.example",
    ):
        if not (ROOT / name).is_file():
            errors.append(f"missing required release file: {name}")
    if not (ROOT / "LICENSE").is_file():
        warnings.append("root LICENSE for ElogGen-original code is not selected")

    for path in ROOT.rglob("*"):
        if not path.is_file() or not included(path):
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in {".h5", ".hdf5"}:
            allowed = (
                len(relative.parts) == 7
                and relative.parts[:4]
                == ("src", "eloggen", "datasets", "taskpacks")
                and relative.parts[5] == "source"
            )
            if not allowed:
                errors.append(f"HDF5 outside task source boundary: {relative}")
            elif hdf5_has_machine_path(path):
                errors.append(f"machine-specific path embedded in HDF5: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if not args.strict_doc_paths and relative.parts[0] == "docs":
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(content.splitlines(), 1):
            if MACHINE_PATH.search(line):
                errors.append(f"machine-specific path: {relative}:{number}")

    for message in warnings:
        print(f"[WARN] {message}")
    for message in errors:
        print(f"[ERROR] {message}")
    if errors:
        print(f"Release audit failed with {len(errors)} error(s).")
        return 1
    print(f"Release audit passed with {len(warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
