"""Shared viewer camera poses for OpenArm tasks."""

from __future__ import annotations

from typing import Optional


REAL_EXP_1_VIEWER_POS = [-0.1913, 3.80157, 1.75215]
REAL_EXP_1_VIEWER_EULER_XYZ_DEG = [-0.896, 43.627, 91.299]


def is_openarm_real_exp_1(identifier: Optional[str]) -> bool:
    return identifier is not None and "openarm_real_exp_1" in str(identifier).lower()


def uses_real_exp_1_viewer_camera(identifier: Optional[str]) -> bool:
    if identifier is None:
        return False
    value = "".join(character for character in str(identifier).lower() if character.isalnum())
    return any(
        "".join(character for character in task_name if character.isalnum()) in value
        for task_name in (
            "openarm_real_exp_1",
            "openarm_drawer_storage",
            "openarm_fruit_basket_bagging",
        )
    )


def real_exp_1_viewer_quat_xyzw() -> list[float]:
    from scipy.spatial.transform import Rotation as R

    # Omniverse Transform UI shows X/Y/Z Euler fields. Use intrinsic XYZ in the
    # same X, Y, Z order; SciPy returns xyzw quaternions, which OmniGibson uses.
    return R.from_euler("XYZ", REAL_EXP_1_VIEWER_EULER_XYZ_DEG, degrees=True).as_quat().astype(float).tolist()


def set_real_exp_1_viewer_camera(og_module, print_prefix: Optional[str] = None) -> tuple[list[float], list[float]]:
    import torch as th

    camera_pos = th.tensor(REAL_EXP_1_VIEWER_POS, dtype=th.float32)
    camera_quat = th.tensor(real_exp_1_viewer_quat_xyzw(), dtype=th.float32)
    og_module.sim.viewer_camera.horizontal_aperture = 35.0
    og_module.sim.viewer_camera.set_position_orientation(
        position=camera_pos,
        orientation=camera_quat,
    )
    if og_module.sim.viewer_camera.active_camera_path != og_module.sim.viewer_camera.prim_path:
        og_module.sim.viewer_camera.active_camera_path = og_module.sim.viewer_camera.prim_path
    if print_prefix:
        print(
            f"{print_prefix} viewer_camera: "
            f"pos={camera_pos.tolist()} "
            f"euler_XYZ_deg={REAL_EXP_1_VIEWER_EULER_XYZ_DEG} "
            f"quat_xyzw={camera_quat.tolist()}"
        )
    return camera_pos.tolist(), camera_quat.tolist()
