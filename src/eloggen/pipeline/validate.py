"""Validate-stage adapter for ElogGen pipelines.

Performs lightweight structural checks on LeRobot datasets or generated HDF5
outputs, without launching a full training or simulation runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import os
from typing import Any, Dict, List, Optional


@dataclass
class DatasetValidationResult:
    ok: bool
    dataset_path: str
    checks: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_lerobot_dataset(dataset_path: str) -> DatasetValidationResult:
    """Lightweight structural validation of a LeRobot dataset directory.

    Checks presence and internal consistency of meta/, data/, and videos/
    without importing heavy LeRobot or PyTorch dependencies.
    """
    root = Path(dataset_path)
    checks: List[Dict[str, Any]] = []
    errors: List[str] = []

    def _check(name: str, passed: bool, detail: str = "") -> None:
        status = "passed" if passed else "failed"
        checks.append({"name": name, "status": status, "detail": detail})
        if not passed:
            errors.append(name)

    # --- meta/ checks ---
    meta_dir = root / "meta"
    _check("meta_dir_exists", meta_dir.is_dir(), str(meta_dir))

    info_path = meta_dir / "info.json"
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    semantic_dir = meta_dir / "semantic_context"

    _check("meta/info.json exists", info_path.is_file(), str(info_path))
    _check("meta/tasks.jsonl exists", tasks_path.is_file(), str(tasks_path))
    _check("meta/episodes.jsonl exists", episodes_path.is_file(), str(episodes_path))
    _check(
        "meta/episodes_stats.jsonl exists",
        episodes_stats_path.is_file(),
        str(episodes_stats_path),
    )
    _check(
        "meta/semantic_context dir exists",
        semantic_dir.is_dir(),
        str(semantic_dir),
    )

    # Read info.json for expected counts
    expected_episodes = 0
    if info_path.is_file():
        try:
            with open(info_path, "r") as fh:
                info = json.load(fh)
            expected_episodes = int(info.get("total_episodes", 0))
            _check("info.json total_episodes parsed", True, f"{expected_episodes}")
            _check(
                "info.json has features",
                "features" in info,
                str(list(info.get("features", {}).keys())[:10]) if info.get("features") else "missing",
            )
        except Exception as exc:
            _check("info.json parseable", False, str(exc))

    # Count episodes from episodes.jsonl
    episode_count = 0
    if episodes_path.is_file():
        try:
            with open(episodes_path, "r") as fh:
                episode_count = sum(1 for line in fh if line.strip())
            _check("episodes.jsonl count", episode_count > 0, f"{episode_count} episodes")
        except Exception as exc:
            _check("episodes.jsonl parseable", False, str(exc))

    # Count parquet files in data/
    parquet_count = 0
    data_dir = root / "data"
    _check("data/ dir exists", data_dir.is_dir(), str(data_dir))
    if data_dir.is_dir():
        parquet_count = len(list(data_dir.rglob("episode_*.parquet")))
        _check(
            "parquet files found",
            parquet_count > 0,
            f"{parquet_count} parquet files",
        )
        if expected_episodes > 0:
            _check(
                "parquet count matches info.json",
                parquet_count == expected_episodes,
                f"parquet={parquet_count} vs info.json={expected_episodes}",
            )

    # Count MP4 files in videos/
    video_count = 0
    videos_dir = root / "videos"
    _check("videos/ dir exists", videos_dir.is_dir(), str(videos_dir))
    if videos_dir.is_dir():
        video_count = len(list(videos_dir.rglob("episode_*.mp4")))
        _check(
            "video files found",
            video_count > 0,
            f"{video_count} MP4 files",
        )
        if expected_episodes > 0:
            cameras_expected = 3  # front / left / right
            expected_videos = expected_episodes * cameras_expected
            _check(
                "video count matches info.json × 3 cameras",
                video_count == expected_videos,
                f"videos={video_count} vs expected={expected_videos}",
            )

    # Count semantic_context files
    semantic_file_count = 0
    if semantic_dir.is_dir():
        semantic_file_count = len(list(semantic_dir.rglob("episode_*.jsonl")))
        _check(
            "semantic_context files found",
            semantic_file_count > 0,
            f"{semantic_file_count} files",
        )
        if expected_episodes > 0:
            _check(
                "semantic_context count matches info.json",
                semantic_file_count == expected_episodes,
                f"semantic={semantic_file_count} vs info.json={expected_episodes}",
            )

    passed = sum(1 for c in checks if c["status"] == "passed")
    failed = sum(1 for c in checks if c["status"] == "failed")
    return DatasetValidationResult(
        ok=(failed == 0),
        dataset_path=str(root),
        checks=checks,
        summary={
            "total_checks": len(checks),
            "passed": passed,
            "failed": failed,
            "episodes": episode_count,
            "parquet_files": parquet_count,
            "video_files": video_count,
            "semantic_files": semantic_file_count,
        },
        errors=errors,
    )


def validate_call_plan(config, run_dir) -> Dict[str, Any]:
    """Build a lightweight call plan for the validate stage."""
    lerobot_output = (
        config.export.lerobot.output
        or str(run_dir.path("lerobot"))
    )
    return {
        "adapter": "eloggen.pipeline.validate",
        "checks": [
            "LeRobot dataset structural validation",
            "meta/info.json consistency",
            "parquet × video × semantic_context count alignment",
        ],
        "dataset_paths": [
            lerobot_output,
        ],
        "note": "Lightweight structural checks only. Does not load parquet data or run training.",
    }