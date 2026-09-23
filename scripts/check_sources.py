#!/usr/bin/env python3
"""Read-only integrity checks for ElogGen's external source checkouts."""

from __future__ import annotations

import argparse
import filecmp
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
META = yaml.safe_load(
    (ROOT / "src/eloggen/patches/behavior1k/version.yaml").read_text(encoding="utf-8")
)
ROBOMIMIC_META = yaml.safe_load(
    (ROOT / "src/eloggen/patches/robomimic/version.yaml").read_text(encoding="utf-8")
)
DATA_ROOT = Path(os.environ.get("ELOGGEN_DATA_ROOT", ROOT)).expanduser().resolve()
THIRD_PARTY_ROOT = Path(
    os.environ.get("ELOGGEN_THIRD_PARTY_ROOT", DATA_ROOT / "third_party")
).expanduser().resolve()
BEHAVIOR = Path(
    os.environ.get("ELOGGEN_BEHAVIOR_ROOT", THIRD_PARTY_ROOT / "BEHAVIOR-1K")
).expanduser().resolve()
ROBOMIMIC = Path(
    os.environ.get("ELOGGEN_ROBOMIMIC_ROOT", THIRD_PARTY_ROOT / "robomimic")
).expanduser().resolve()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def run_checker(script: str, *arguments: str) -> bool:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        text=True,
        check=False,
    )
    return completed.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-openarm-assets",
        action="store_true",
        help="also require the separately installed OpenArm runtime assets",
    )
    args = parser.parse_args(argv)
    results: list[bool] = []
    results.append(check("BEHAVIOR root", BEHAVIOR.is_dir(), str(BEHAVIOR)))
    expected = str(META["commit"])
    actual = git_head(BEHAVIOR)
    results.append(
        check("BEHAVIOR commit", actual == expected, f"expected={expected} actual={actual}")
    )

    omni = BEHAVIOR / "OmniGibson"
    results.append(check("OmniGibson checkout", omni.is_dir(), str(omni)))
    results.append(check("robomimic root", ROBOMIMIC.is_dir(), str(ROBOMIMIC)))
    expected_robomimic = str(ROBOMIMIC_META["commit"])
    actual_robomimic = git_head(ROBOMIMIC)
    results.append(
        check(
            "robomimic commit",
            actual_robomimic == expected_robomimic,
            f"expected={expected_robomimic} actual={actual_robomimic}",
        )
    )
    patch = ROOT / "src/eloggen/patches/behavior1k" / META["omnigibson_patch"]
    results.append(
        check("overlay patch", patch.is_file() and patch.stat().st_size > 0, str(patch))
    )

    if BEHAVIOR.is_dir() and actual == expected:
        overlay_ok = run_checker(
            "apply_patches.py",
            "--behavior-root",
            str(BEHAVIOR),
            "--robomimic-root",
            str(ROBOMIMIC),
        )
        results.append(check("overlay contents", overlay_ok, str(BEHAVIOR)))

        datasets = BEHAVIOR / "datasets"
        required_data = (
            datasets / "behavior-1k-assets",
            datasets / "omnigibson-robot-assets",
            datasets / "omnigibson.key",
        )
        missing_data = [str(path) for path in required_data if not path.exists()]
        results.append(
            check(
                "BEHAVIOR datasets",
                not missing_data,
                str(datasets) if not missing_data else f"missing: {missing_data}",
            )
        )

    robot_assets = (
        BEHAVIOR
        / "datasets/custom_dataset/objects/robot/openarmbimanual"
    )
    required_assets = [
        robot_assets / "usd/openarmbimanual.usda",
        robot_assets / "misc/metadata.json",
        robot_assets / "curobo/openarmbimanual_description_curobo_default.yaml",
        robot_assets / "curobo/openarmbimanual_description_curobo_arm.yaml",
        robot_assets / "curobo/openarmbimanual_description_curobo_arm_no_torso.yaml",
    ]
    if args.with_openarm_assets:
        missing_assets = [str(path) for path in required_assets if not path.is_file()]
        results.append(
            check(
                "OpenArm assets",
                not missing_assets,
                "all canonical files present"
                if not missing_assets
                else ", ".join(missing_assets),
            )
        )
    else:
        print("[SKIP] OpenArm assets: install separately before simulation")

    task_dirs = sorted(
        path.parent
        for path in (ROOT / "src/eloggen/datasets/taskpacks").glob("*/task.yaml")
    )
    mismatches: list[str] = []
    for task_dir in task_dirs:
        installed = (
            BEHAVIOR
            / "bddl/bddl/activity_definitions_new"
            / task_dir.name
            / "problem0.bddl"
        )
        source = task_dir / "problem.bddl"
        if not installed.is_file() or not filecmp.cmp(source, installed, shallow=False):
            mismatches.append(task_dir.name)
    results.append(
        check(
            "task BDDL overlays",
            bool(task_dirs) and not mismatches,
            f"{len(task_dirs)} task packs" if not mismatches else f"mismatch: {mismatches}",
        )
    )
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
