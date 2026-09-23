#!/usr/bin/env python3
"""Capture machine-local environment evidence for debugging and reproduction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "ELOGGEN_ENV_SNAPSHOT_DIR",
        str(PROJECT_ROOT / ".eloggen" / "environment"),
    )
)


def output(*command: str) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def optional_output(*command: str) -> str | None:
    try:
        return output(*command)
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"[WROTE] {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-root", required=True, type=Path)
    parser.add_argument("--robomimic-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help=(
            "machine-local snapshot directory "
            "(default: .eloggen/environment or ELOGGEN_ENV_SNAPSHOT_DIR)"
        ),
    )
    args = parser.parse_args()

    behavior_root = args.behavior_root.expanduser().resolve()
    robomimic_root = args.robomimic_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    env_prefix = Path(sys.prefix).resolve()

    # These files are diagnostic snapshots of the active machine, not install
    # constraints and not repository lock files.
    pip_snapshot = output(sys.executable, "-m", "pip", "freeze", "--exclude-editable")
    filtered = [
        line
        for line in pip_snapshot.splitlines()
        if line.strip() and " @ " not in line and not line.startswith("-e ")
    ]
    write_text(output_dir / "pip-freeze.snapshot.txt", "\n".join(filtered))

    conda_snapshot = optional_output("conda", "list", "--prefix", str(env_prefix), "--explicit")
    if conda_snapshot:
        write_text(output_dir / "conda-explicit.snapshot.txt", conda_snapshot)

    datasets = behavior_root / "datasets"
    data_links = {}
    for name in (
        "behavior-1k-assets",
        "omnigibson-robot-assets",
        "custom_dataset",
        "omnigibson.key",
    ):
        path = datasets / name
        data_links[name] = {
            "path": str(path),
            "is_symlink": path.is_symlink(),
            "resolved": str(path.resolve()) if path.exists() else None,
        }

    manifest = {
        "format_version": 2,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "machine-local diagnostic snapshot; not an installation lock",
        "python": sys.version,
        "python_executable": sys.executable,
        "environment_prefix": str(env_prefix),
        "platform": platform.platform(),
        "behavior_root": str(behavior_root),
        "behavior_commit": optional_output("git", "-C", str(behavior_root), "rev-parse", "HEAD"),
        "robomimic_root": str(robomimic_root),
        "robomimic_commit": optional_output("git", "-C", str(robomimic_root), "rev-parse", "HEAD"),
        "omnigibson_commit": optional_output("git", "-C", str(behavior_root / "OmniGibson"), "rev-parse", "HEAD"),
        "bddl_commit": optional_output("git", "-C", str(behavior_root / "bddl"), "rev-parse", "HEAD"),
        "versions": {
            name: package_version(name)
            for name in (
                "eloggen",
                "omnigibson",
                "bddl",
                "torch",
                "torchvision",
                "isaacsim",
                "numpy",
                "h5py",
                "scipy",
                "opencv-python",
                "open3d",
                "cvxpy",
            )
        },
        "nvidia_driver": optional_output(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ),
        "dataset_links": data_links,
    }
    write_text(
        output_dir / "environment-manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
