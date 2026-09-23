#!/usr/bin/env python3
"""Apply or verify ElogGen adaptations on pinned external source checkouts."""

from __future__ import annotations

import argparse
import filecmp
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = PROJECT_ROOT / "src" / "eloggen" / "patches"
BEHAVIOR_PATCH_ROOT = PATCH_ROOT / "behavior1k"
ROBOMIMIC_PATCH_ROOT = PATCH_ROOT / "robomimic"
TASKPACK_ROOT = PROJECT_ROOT / "src" / "eloggen" / "datasets" / "taskpacks"
PATCH_PATHS = (
    BEHAVIOR_PATCH_ROOT / "patches" / "omnigibson-openarm.patch",
    BEHAVIOR_PATCH_ROOT / "patches" / "omnigibson-local-curobo.patch",
    BEHAVIOR_PATCH_ROOT / "patches" / "omnigibson-explicit-scene-file.patch",
    BEHAVIOR_PATCH_ROOT / "patches" / "behavior-setup-isaac-cache.patch",
)
ROBOT_SOURCE = BEHAVIOR_PATCH_ROOT / "openarm_bimanual.py"
ROBOMIMIC_ENV_SOURCE = ROBOMIMIC_PATCH_ROOT / "env_omnigibson.py"


def expected_commit() -> str:
    manifest = BEHAVIOR_PATCH_ROOT / "version.yaml"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.startswith("commit:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"Missing top-level commit in {manifest}")


def expected_robomimic_commit() -> str:
    manifest = ROBOMIMIC_PATCH_ROOT / "version.yaml"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.startswith("commit:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"Missing robomimic commit in {manifest}")


def run_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def patch_state(root: Path, patch_path: Path) -> str:
    apply_check = run_git(
        root,
        "apply",
        "--check",
        "--ignore-space-change",
        "--ignore-whitespace",
        str(patch_path),
        check=False,
    )
    if apply_check.returncode == 0:
        return "not-applied"
    reverse_check = run_git(
        root,
        "apply",
        "--reverse",
        "--check",
        "--ignore-space-change",
        "--ignore-whitespace",
        str(patch_path),
        check=False,
    )
    if reverse_check.returncode == 0:
        return "applied"
    detail = apply_check.stderr.strip() or reverse_check.stderr.strip()
    raise RuntimeError(f"Overlay patch is incompatible with this checkout: {detail}")


def verify_destination(source: Path, destination: Path) -> str:
    if not destination.exists():
        return "missing"
    if not destination.is_file() or not filecmp.cmp(source, destination, shallow=False):
        raise RuntimeError(f"Existing overlay destination differs: {destination}")
    return "installed"


def install_file(source: Path, destination: Path, apply: bool) -> None:
    state = verify_destination(source, destination)
    if state == "installed":
        print(f"[OK] {destination}")
        return
    if not apply:
        raise RuntimeError(f"Overlay file is missing: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"[INSTALLED] {destination}")


def install_tracked_overlay(
    checkout: Path,
    source: Path,
    relative_destination: Path,
    apply: bool,
) -> None:
    """Replace one tracked upstream file without hiding unrelated local edits."""
    destination = checkout / relative_destination
    if destination.is_file() and filecmp.cmp(source, destination, shallow=False):
        print(f"[OK] {destination}")
        return

    upstream = run_git(checkout, "show", f"HEAD:{relative_destination.as_posix()}")
    current = destination.read_text(encoding="utf-8") if destination.is_file() else None
    if current is not None and current != upstream.stdout:
        raise RuntimeError(f"Refusing to replace locally modified upstream file: {destination}")
    if not apply:
        raise RuntimeError(f"Overlay file has not been installed: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"[INSTALLED] {destination}")


def install_bddl(root: Path, apply: bool) -> None:
    definitions = root / "bddl" / "bddl" / "activity_definitions"
    definitions_new = root / "bddl" / "bddl" / "activity_definitions_new"
    for task_dir in sorted(TASKPACK_ROOT.iterdir()):
        source = task_dir / "problem.bddl"
        if not source.is_file():
            continue
        destination = definitions_new / task_dir.name / "problem0.bddl"
        install_file(source, destination, apply)

        public_path = definitions / task_dir.name
        expected_target = Path("..") / "activity_definitions_new" / task_dir.name
        if public_path.is_symlink():
            if Path(os.readlink(public_path)) != expected_target:
                raise RuntimeError(f"Unexpected BDDL symlink target: {public_path}")
            print(f"[OK] {public_path} -> {expected_target}")
        elif public_path.exists():
            fallback = public_path / "problem0.bddl"
            if not fallback.is_file() or not filecmp.cmp(source, fallback, shallow=False):
                raise RuntimeError(f"Existing BDDL activity differs: {public_path}")
            print(f"[OK] {public_path} (directory copy)")
        elif apply:
            public_path.parent.mkdir(parents=True, exist_ok=True)
            public_path.symlink_to(expected_target, target_is_directory=True)
            print(f"[INSTALLED] {public_path} -> {expected_target}")
        else:
            raise RuntimeError(f"BDDL activity link is missing: {public_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-root", required=True, type=Path)
    parser.add_argument("--robomimic-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply missing overlay files")
    args = parser.parse_args()

    root = args.behavior_root.expanduser().resolve()
    robomimic_root = args.robomimic_root.expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"Not a BEHAVIOR-1K git checkout: {root}")

    actual = run_git(root, "rev-parse", "HEAD").stdout.strip()
    expected = expected_commit()
    if actual != expected:
        raise RuntimeError(f"BEHAVIOR-1K commit mismatch: expected={expected} actual={actual}")
    print(f"[OK] BEHAVIOR-1K commit {actual}")

    for patch_path in PATCH_PATHS:
        state = patch_state(root, patch_path)
        if state == "not-applied":
            if not args.apply:
                raise RuntimeError(f"Overlay patch has not been applied: {patch_path}")
            run_git(
                root,
                "apply",
                "--ignore-space-change",
                "--ignore-whitespace",
                str(patch_path),
            )
            print(f"[APPLIED] {patch_path}")
        else:
            print(f"[OK] overlay patch already applied: {patch_path}")

    robot_destination = root / "OmniGibson" / "omnigibson" / "robots" / "openarm_bimanual.py"
    install_file(ROBOT_SOURCE, robot_destination, args.apply)
    install_bddl(root, args.apply)
    print("BEHAVIOR-1K adaptations are complete")

    if not (robomimic_root / ".git").exists():
        raise RuntimeError(f"Not a robomimic git checkout: {robomimic_root}")
    actual_robomimic = run_git(robomimic_root, "rev-parse", "HEAD").stdout.strip()
    expected_robomimic = expected_robomimic_commit()
    if actual_robomimic != expected_robomimic:
        raise RuntimeError(
            "robomimic commit mismatch: "
            f"expected={expected_robomimic} actual={actual_robomimic}"
        )
    print(f"[OK] robomimic commit {actual_robomimic}")
    install_tracked_overlay(
        robomimic_root,
        ROBOMIMIC_ENV_SOURCE,
        Path("robomimic/envs/env_omnigibson.py"),
        args.apply,
    )
    print("robomimic adaptations are complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
