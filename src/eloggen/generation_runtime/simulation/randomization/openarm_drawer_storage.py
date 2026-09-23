"""Reproducible initialization distributions for OpenArm drawer storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DrawerStorageDifficulty:
    name: str
    object_xy_magnitude: float
    object_yaw_deg: float
    cabinet_x_magnitude: float
    cabinet_y_magnitude: float
    cabinet_yaw_deg: float
    robot_arm_joint_magnitude_deg: float = 0.0
    drawer_open_distance_m: float = 0.184
    object_drawer_clearance_m: float = 0.05
    object_pair_clearance_m: float = 0.05
    drawer_open_fraction: tuple[float, float] = (0.0, 0.0)
    max_sampling_attempts: int = 50

    def to_dict(self) -> dict:
        result = asdict(self)
        result["difficulty"] = self.name
        result["drawer_open_fraction"] = list(self.drawer_open_fraction)
        return result


DRAWER_STORAGE_DIFFICULTIES = {
    "D0": DrawerStorageDifficulty("D0", 0.06, 7.5, 0.04, 0.025, 0.0, 3.0),
    "D1": DrawerStorageDifficulty("D1", 0.04, 5.0, 0.03, 0.02, 0.0),
    "D2": DrawerStorageDifficulty(
        "D2", 0.08, 10.0, 0.05, 0.03, 10.0, 3.0,
        drawer_open_distance_m=0.184,
        object_drawer_clearance_m=0.05,
    ),
}


@dataclass(frozen=True)
class PlanarPoseDelta:
    delta_xy: tuple[float, float]
    delta_yaw_deg: float

    def to_dict(self) -> dict:
        return {
            "delta_xy": [float(value) for value in self.delta_xy],
            "delta_yaw_deg": float(self.delta_yaw_deg),
        }


@dataclass(frozen=True)
class DrawerInitializationSample:
    difficulty: str
    seed: int
    attempt_index: int
    candidate_index: int
    object_1: PlanarPoseDelta
    object_2: PlanarPoseDelta
    cabinet: PlanarPoseDelta
    robot_arm_joint_delta_deg: dict[str, tuple[float, ...]]
    drawer_open_fraction: float
    object_order: tuple[str, str]
    reset_valid: bool = True
    rejection_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "seed": self.seed,
            "attempt_index": self.attempt_index,
            "candidate_index": self.candidate_index,
            "objects": {
                "object_1": self.object_1.to_dict(),
                "object_2": self.object_2.to_dict(),
            },
            "cabinet": self.cabinet.to_dict(),
            "robot_arm_joint_delta_deg": {
                arm: [float(value) for value in values]
                for arm, values in self.robot_arm_joint_delta_deg.items()
            },
            "drawer_open_fraction": float(self.drawer_open_fraction),
            "object_order": list(self.object_order),
            "reset_valid": bool(self.reset_valid),
            "rejection_reason": self.rejection_reason,
        }


def _resolve_difficulty_name(name: str, values: Mapping | None = None) -> str:
    """Resolve a valid D0/D1/D2 key without treating the base task suffix as one.

    Runtime env names such as ``openarm_drawer_storage`` end in ``storage``.  Older
    code blindly used that suffix as the difficulty key and raised ``KeyError``.
    Explicit runtime variants (``*_D0``/``*_D1``/``*_D2``) still take precedence;
    otherwise use the initialization-distribution metadata and finally D0.
    """
    requested = str(name).upper()
    if requested in DRAWER_STORAGE_DIFFICULTIES:
        return requested
    if values:
        for key in ("difficulty", "name"):
            configured = values.get(key)
            if configured is not None:
                configured = str(configured).upper()
                if configured in DRAWER_STORAGE_DIFFICULTIES:
                    return configured
    return "D0"


def difficulty_from_mapping(name: str, values: Mapping | None = None) -> DrawerStorageDifficulty:
    resolved_name = _resolve_difficulty_name(name, values)
    base = DRAWER_STORAGE_DIFFICULTIES[resolved_name]
    if not values:
        return base
    drawer_range = values.get("drawer_open_fraction", base.drawer_open_fraction)
    return DrawerStorageDifficulty(
        name=resolved_name,
        object_xy_magnitude=float(values.get("object_xy_magnitude", base.object_xy_magnitude)),
        object_yaw_deg=float(values.get("object_yaw_deg", base.object_yaw_deg)),
        cabinet_x_magnitude=float(values.get("cabinet_x_magnitude", base.cabinet_x_magnitude)),
        cabinet_y_magnitude=float(values.get("cabinet_y_magnitude", base.cabinet_y_magnitude)),
        cabinet_yaw_deg=float(values.get("cabinet_yaw_deg", base.cabinet_yaw_deg)),
        robot_arm_joint_magnitude_deg=float(
            values.get("robot_arm_joint_magnitude_deg", base.robot_arm_joint_magnitude_deg)
        ),
        drawer_open_distance_m=float(values.get("drawer_open_distance_m", base.drawer_open_distance_m)),
        object_drawer_clearance_m=float(
            values.get("object_drawer_clearance_m", base.object_drawer_clearance_m)
        ),
        object_pair_clearance_m=float(
            values.get("object_pair_clearance_m", base.object_pair_clearance_m)
        ),
        drawer_open_fraction=(float(drawer_range[0]), float(drawer_range[1])),
        max_sampling_attempts=int(values.get("max_sampling_attempts", base.max_sampling_attempts)),
    )


def _uniform_symmetric(rng: np.random.Generator, magnitude: float, size=None):
    if magnitude == 0.0:
        return np.zeros(size, dtype=np.float64) if size is not None else 0.0
    return rng.uniform(-magnitude, magnitude, size=size)


def validate_aabb_support(
    child_center,
    child_extent,
    support_center,
    support_extent,
    vertical_tolerance: float = 0.02,
    min_xy_coverage: float = 0.95,
) -> tuple[bool, str | None]:
    child_center = np.asarray(child_center, dtype=np.float64)
    child_extent = np.asarray(child_extent, dtype=np.float64)
    support_center = np.asarray(support_center, dtype=np.float64)
    support_extent = np.asarray(support_extent, dtype=np.float64)
    child_min, child_max = child_center - child_extent / 2.0, child_center + child_extent / 2.0
    support_min, support_max = support_center - support_extent / 2.0, support_center + support_extent / 2.0

    vertical_gap = float(child_min[2] - support_max[2])
    if abs(vertical_gap) > vertical_tolerance:
        return False, f"cabinet_support_gap:{vertical_gap:.6f}"

    overlap = np.maximum(0.0, np.minimum(child_max[:2], support_max[:2]) - np.maximum(child_min[:2], support_min[:2]))
    child_area = float(np.prod(child_extent[:2]))
    coverage = float(np.prod(overlap) / child_area) if child_area > 0.0 else 0.0
    if coverage < min_xy_coverage:
        return False, f"cabinet_support_xy_coverage:{coverage:.6f}"
    return True, None


def aabb_xy_distance_to_translated_sweep(
    object_center,
    object_extent,
    obstacle_center,
    obstacle_extent,
    obstacle_translation_xy,
) -> float:
    object_center = np.asarray(object_center, dtype=np.float64)
    object_extent = np.asarray(object_extent, dtype=np.float64)
    obstacle_center = np.asarray(obstacle_center, dtype=np.float64)
    obstacle_extent = np.asarray(obstacle_extent, dtype=np.float64)
    translation = np.asarray(obstacle_translation_xy, dtype=np.float64)

    object_min = object_center[:2] - object_extent[:2] / 2.0
    object_max = object_center[:2] + object_extent[:2] / 2.0
    closed_min = obstacle_center[:2] - obstacle_extent[:2] / 2.0
    closed_max = obstacle_center[:2] + obstacle_extent[:2] / 2.0
    open_min, open_max = closed_min + translation, closed_max + translation
    sweep_min = np.minimum(closed_min, open_min)
    sweep_max = np.maximum(closed_max, open_max)
    separation = np.maximum(0.0, np.maximum(sweep_min - object_max, object_min - sweep_max))
    return float(np.linalg.norm(separation))


def sample_drawer_initialization(
    difficulty: str,
    seed: int,
    attempt_index: int,
    candidate_index: int = 0,
    overrides: Mapping | None = None,
) -> DrawerInitializationSample:
    spec = difficulty_from_mapping(difficulty, overrides)
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(attempt_index), int(candidate_index), 0xD0A7]
    )
    rng = np.random.default_rng(seed_sequence)

    def sample_object() -> PlanarPoseDelta:
        xy = _uniform_symmetric(rng, spec.object_xy_magnitude, size=2)
        yaw = _uniform_symmetric(rng, spec.object_yaw_deg)
        return PlanarPoseDelta(tuple(float(value) for value in xy), float(yaw))

    cabinet_xy = (
        float(_uniform_symmetric(rng, spec.cabinet_x_magnitude)),
        float(_uniform_symmetric(rng, spec.cabinet_y_magnitude)),
    )
    drawer_fraction = float(rng.uniform(*spec.drawer_open_fraction)) if spec.drawer_open_fraction[1] > 0 else 0.0
    return DrawerInitializationSample(
        difficulty=spec.name,
        seed=int(seed),
        attempt_index=int(attempt_index),
        candidate_index=int(candidate_index),
        object_1=sample_object(),
        object_2=sample_object(),
        cabinet=PlanarPoseDelta(
            cabinet_xy,
            float(_uniform_symmetric(rng, spec.cabinet_yaw_deg)),
        ),
        robot_arm_joint_delta_deg={
            arm: tuple(
                float(value)
                for value in _uniform_symmetric(rng, spec.robot_arm_joint_magnitude_deg, size=7)
            )
            for arm in ("left", "right")
        },
        drawer_open_fraction=drawer_fraction,
        object_order=tuple(rng.permutation(["object_1", "object_2"])),
    )
