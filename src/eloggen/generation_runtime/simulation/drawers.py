"""Task-owned articulated drawer definitions and runtime state access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _quat_rotate_xyzw(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quat[:3]
    scalar = quat[3]
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + scalar * vector)


@dataclass(frozen=True)
class DrawerTarget:
    target_id: str
    object_name: str
    joint_name: str
    drawer_link_name: str
    fillable_link_name: str
    handle_link_name: str | None
    handle_local_position: tuple[float, float, float]
    handle_local_orientation_xyzw: tuple[float, float, float, float]
    lower_limit: float
    upper_limit: float
    closed_fraction: float = 0.08
    open_fraction: float = 0.65

    def resolve_object(self, env: Any) -> Any:
        obj = env.scene.object_registry("name", self.object_name)
        if obj is None:
            raise KeyError(f"Articulated object not found: {self.object_name}")
        return obj

    def resolve_joint(self, env: Any) -> Any:
        obj = self.resolve_object(env)
        if self.joint_name not in obj.joints:
            raise KeyError(f"Joint {self.joint_name!r} missing from {self.object_name}")
        return obj.joints[self.joint_name]

    def resolve_drawer_link(self, env: Any) -> Any:
        obj = self.resolve_object(env)
        if self.drawer_link_name not in obj.links:
            raise KeyError(f"Link {self.drawer_link_name!r} missing from {self.object_name}")
        return obj.links[self.drawer_link_name]

    def get_joint_position(self, env: Any) -> float:
        obj = self.resolve_object(env)
        index = list(obj.joints).index(self.joint_name)
        return float(_as_numpy(obj.get_joint_positions())[index])

    def get_joint_velocity(self, env: Any) -> float:
        obj = self.resolve_object(env)
        index = list(obj.joints).index(self.joint_name)
        return float(_as_numpy(obj.get_joint_velocities())[index])

    def get_open_fraction(self, env: Any) -> float:
        span = self.upper_limit - self.lower_limit
        if span <= 0.0:
            raise ValueError(f"Invalid joint limits for {self.target_id}: {self.lower_limit}, {self.upper_limit}")
        return float(np.clip((self.get_joint_position(env) - self.lower_limit) / span, 0.0, 1.0))

    def is_open(self, env: Any) -> bool:
        return self.get_open_fraction(env) >= self.open_fraction

    def is_closed(self, env: Any) -> bool:
        return self.get_open_fraction(env) <= self.closed_fraction

    def get_handle_world_pose(self, env: Any) -> tuple[np.ndarray, np.ndarray]:
        obj = self.resolve_object(env)
        link_name = self.handle_link_name or self.drawer_link_name
        if link_name not in obj.links:
            raise KeyError(f"Handle reference link {link_name!r} missing from {self.object_name}")
        link_position, link_orientation = obj.links[link_name].get_position_orientation()
        link_position = _as_numpy(link_position)
        link_orientation = _as_numpy(link_orientation)
        local_position = np.asarray(self.handle_local_position, dtype=np.float64)
        local_orientation = np.asarray(self.handle_local_orientation_xyzw, dtype=np.float64)
        return (
            link_position + _quat_rotate_xyzw(link_orientation, local_position),
            _quat_multiply_xyzw(link_orientation, local_orientation),
        )


# mbmbpa has no separate handle link. The grasp reference is attached to the
# integrated slot on the lower drawer face and moves with link_1.
OPENARM_DRAWER_TARGETS = {
    "lower_drawer": DrawerTarget(
        target_id="lower_drawer",
        object_name="drawer_cabinet_1",
        joint_name="j_link_1",
        drawer_link_name="link_1",
        fillable_link_name="meta__link_1_fillable_0_0_link",
        handle_link_name=None,
        handle_local_position=(0.18, 0.0, 0.0404),
        handle_local_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        lower_limit=0.0,
        upper_limit=0.35203781723976135,
        closed_fraction=0.08,
        open_fraction=0.20,
    )
}
