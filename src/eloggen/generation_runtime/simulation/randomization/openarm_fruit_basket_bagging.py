"""Initialization randomization for OpenArm fruit basket bagging."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FruitBaggingDifficulty:
    name: str
    object_xy_magnitude: float
    object_yaw_deg: float
    container_x_magnitude: float
    container_y_magnitude: float
    container_yaw_deg: float
    robot_arm_joint_magnitude_deg: float = 0.0
    object_container_clearance_m: float = 0.05
    object_pair_clearance_m: float = 0.05
    container_pair_clearance_m: float = 0.05
    object_container_repair_gap_m: float = 0.05
    preserve_left_to_right_order: bool = True
    allow_object_order_permutation: bool = False
    max_sampling_attempts: int = 50

    def to_dict(self) -> dict:
        result = asdict(self)
        result["difficulty"] = self.name
        return result


FRUIT_BAGGING_DIFFICULTIES = {
    # D0: only fruit positions move. Containers, fruit yaw, and robot joints
    # stay at the source state.
    "D0": FruitBaggingDifficulty("D0", 0.02, 0.0, 0.00, 0.00, 0.0),
    # D1: fruit and containers have larger pose perturbations, plus arm joint
    # initialization perturbations. Fruit left-to-right order is preserved.
    "D1": FruitBaggingDifficulty("D1", 0.035, 7.5, 0.025, 0.02, 10.0, 3.0),
    # D2: same magnitudes as D1, but fruit placement slots may be permuted
    # within the corresponding arm-reachable side before perturbation.
    "D2": FruitBaggingDifficulty(
        "D2",
        0.035,
        7.5,
        0.025,
        0.02,
        10.0,
        3.0,
        preserve_left_to_right_order=False,
        allow_object_order_permutation=True,
    ),
}


FRUIT_SOURCE_SLOTS = {
    # These match generate_openarm_fruit_basket_bagging_scene.py. D2 samples a
    # slot assignment first, then adds per-object perturbation.
    "object_1": (-1.18, 3.81),  # apple, right arm
    "object_2": (-1.04, 3.81),  # orange, right arm
    "object_3": (-0.86, 3.81),  # lemon, left arm
    "object_4": (-0.72, 3.81),  # pear, left arm
}
FRUIT_D2_ALLOWED_SLOTS = {
    # Allow broad layout variation while avoiding the obviously hard extreme
    # swap: the leftmost source fruit should not move to the far-right slot,
    # and the rightmost source fruit should not move to the far-left slot.
    "object_1": ("object_1", "object_2", "object_3"),
    "object_2": ("object_1", "object_2", "object_3"),
    "object_3": ("object_2", "object_3", "object_4"),
    "object_4": ("object_2", "object_3", "object_4"),
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
class FruitBaggingInitializationSample:
    difficulty: str
    seed: int
    attempt_index: int
    candidate_index: int
    objects: dict[str, PlanarPoseDelta]
    containers: dict[str, PlanarPoseDelta]
    robot_arm_joint_delta_deg: dict[str, tuple[float, ...]]
    preserve_left_to_right_order: bool
    allow_object_order_permutation: bool
    object_slot_assignment: dict[str, str]
    reset_valid: bool = True
    rejection_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "seed": int(self.seed),
            "attempt_index": int(self.attempt_index),
            "candidate_index": int(self.candidate_index),
            "objects": {
                name: delta.to_dict()
                for name, delta in self.objects.items()
            },
            "containers": {
                name: delta.to_dict()
                for name, delta in self.containers.items()
            },
            "robot_arm_joint_delta_deg": {
                arm: [float(value) for value in values]
                for arm, values in self.robot_arm_joint_delta_deg.items()
            },
            "preserve_left_to_right_order": bool(self.preserve_left_to_right_order),
            "allow_object_order_permutation": bool(self.allow_object_order_permutation),
            "object_slot_assignment": dict(self.object_slot_assignment),
            "reset_valid": bool(self.reset_valid),
            "rejection_reason": self.rejection_reason,
        }


def _uniform_symmetric(rng: np.random.Generator, magnitude: float, size=None):
    if magnitude == 0.0:
        return np.zeros(size, dtype=np.float64) if size is not None else 0.0
    return rng.uniform(-magnitude, magnitude, size=size)


def _resolve_difficulty_name(name: str, values: Mapping | None = None) -> str:
    """Resolve a valid D0/D1/D2 key for base and difficulty-suffixed task names."""
    requested = str(name).upper()
    if requested in FRUIT_BAGGING_DIFFICULTIES:
        return requested
    if values:
        for key in ("difficulty", "name"):
            configured = values.get(key)
            if configured is not None:
                configured = str(configured).upper()
                if configured in FRUIT_BAGGING_DIFFICULTIES:
                    return configured
    return "D0"


def difficulty_from_mapping(name: str, values: Mapping | None = None) -> FruitBaggingDifficulty:
    resolved_name = _resolve_difficulty_name(name, values)
    base = FRUIT_BAGGING_DIFFICULTIES[resolved_name]
    if not values:
        return base
    return FruitBaggingDifficulty(
        name=resolved_name,
        object_xy_magnitude=float(values.get("object_xy_magnitude", base.object_xy_magnitude)),
        object_yaw_deg=float(values.get("object_yaw_deg", base.object_yaw_deg)),
        container_x_magnitude=float(values.get("container_x_magnitude", base.container_x_magnitude)),
        container_y_magnitude=float(values.get("container_y_magnitude", base.container_y_magnitude)),
        container_yaw_deg=float(values.get("container_yaw_deg", base.container_yaw_deg)),
        robot_arm_joint_magnitude_deg=float(
            values.get("robot_arm_joint_magnitude_deg", base.robot_arm_joint_magnitude_deg)
        ),
        object_container_clearance_m=float(
            values.get("object_container_clearance_m", base.object_container_clearance_m)
        ),
        object_pair_clearance_m=float(values.get("object_pair_clearance_m", base.object_pair_clearance_m)),
        container_pair_clearance_m=float(
            values.get("container_pair_clearance_m", base.container_pair_clearance_m)
        ),
        object_container_repair_gap_m=float(
            values.get("object_container_repair_gap_m", base.object_container_repair_gap_m)
        ),
        preserve_left_to_right_order=bool(
            values.get("preserve_left_to_right_order", base.preserve_left_to_right_order)
        ),
        allow_object_order_permutation=bool(
            values.get("allow_object_order_permutation", base.allow_object_order_permutation)
        ),
        max_sampling_attempts=int(values.get("max_sampling_attempts", base.max_sampling_attempts)),
    )


def sample_fruit_bagging_initialization(
    difficulty: str,
    seed: int,
    attempt_index: int,
    candidate_index: int = 0,
    overrides: Mapping | None = None,
) -> FruitBaggingInitializationSample:
    spec = difficulty_from_mapping(difficulty, overrides)
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(attempt_index), int(candidate_index), 0xF8A7]
    )
    rng = np.random.default_rng(seed_sequence)

    def sample_delta(xy_magnitude: tuple[float, float] | float, yaw_magnitude: float) -> PlanarPoseDelta:
        if isinstance(xy_magnitude, tuple):
            xy = np.array(
                [
                    _uniform_symmetric(rng, xy_magnitude[0]),
                    _uniform_symmetric(rng, xy_magnitude[1]),
                ],
                dtype=np.float64,
            )
        else:
            xy = _uniform_symmetric(rng, xy_magnitude, size=2)
        yaw = _uniform_symmetric(rng, yaw_magnitude)
        return PlanarPoseDelta(tuple(float(value) for value in xy), float(yaw))

    slot_assignment = {name: name for name in FRUIT_SOURCE_SLOTS}
    if spec.allow_object_order_permutation:
        object_names = list(FRUIT_SOURCE_SLOTS)
        for _ in range(100):
            candidate_slots = list(FRUIT_SOURCE_SLOTS)
            rng.shuffle(candidate_slots)
            candidate = dict(zip(object_names, candidate_slots))
            if all(
                slot_name in FRUIT_D2_ALLOWED_SLOTS[object_name]
                for object_name, slot_name in candidate.items()
            ):
                slot_assignment = candidate
                break

    def sample_object_delta(object_name: str) -> PlanarPoseDelta:
        noise = sample_delta(spec.object_xy_magnitude, spec.object_yaw_deg)
        source_xy = np.array(FRUIT_SOURCE_SLOTS[object_name], dtype=np.float64)
        slot_xy = np.array(FRUIT_SOURCE_SLOTS[slot_assignment[object_name]], dtype=np.float64)
        xy = slot_xy - source_xy + np.asarray(noise.delta_xy, dtype=np.float64)
        return PlanarPoseDelta(tuple(float(value) for value in xy), noise.delta_yaw_deg)

    return FruitBaggingInitializationSample(
        difficulty=spec.name,
        seed=int(seed),
        attempt_index=int(attempt_index),
        candidate_index=int(candidate_index),
        objects={
            name: sample_object_delta(name)
            for name in ("object_1", "object_2", "object_3", "object_4")
        },
        containers={
            name: sample_delta((spec.container_x_magnitude, spec.container_y_magnitude), spec.container_yaw_deg)
            for name in ("paper_bag_1", "basket_1")
        },
        robot_arm_joint_delta_deg={
            arm: tuple(
                float(value)
                for value in _uniform_symmetric(rng, spec.robot_arm_joint_magnitude_deg, size=7)
            )
            for arm in ("left", "right")
        },
        preserve_left_to_right_order=spec.preserve_left_to_right_order,
        allow_object_order_permutation=spec.allow_object_order_permutation,
        object_slot_assignment=slot_assignment,
    )
