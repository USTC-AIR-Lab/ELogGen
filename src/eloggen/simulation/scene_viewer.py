#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path
import sys

from eloggen.simulation.omnigibson import activate, behavior_root, project_root

REPO_ROOT = project_root()
BEHAVIOR_ROOT = behavior_root()


def _ensure_local_omnigibson_importable() -> None:
    og_root = BEHAVIOR_ROOT / "OmniGibson"
    if str(og_root) not in sys.path:
        sys.path.insert(0, str(og_root))


activate()
import omnigibson as og
import torch as th
from scipy.spatial.transform import Rotation as R
from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.utils.python_utils import create_class_from_registry_and_config
from eloggen.generation_runtime.simulation.camera import (
    set_real_exp_1_viewer_camera,
    uses_real_exp_1_viewer_camera,
)


OPENARM_STARTUP_QS_14 = (
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 0.0],
    [0.3, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0],
)
DEFAULT_DELAYED_OBJECTS = ("object_1", "object_2")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_root_link_state(root_link_state: dict) -> dict:
    root_link_state.setdefault("pos", [0.0, 0.0, 0.0])
    root_link_state.setdefault("ori", [0.0, 0.0, 0.0, 1.0])
    root_link_state.setdefault("lin_vel", [0.0, 0.0, 0.0])
    root_link_state.setdefault("ang_vel", [0.0, 0.0, 0.0])
    return root_link_state


def _hydrate_scene_state(scene_cfg: dict) -> None:
    init_info = scene_cfg.get("objects_info", {}).get("init_info", {})
    object_registry = scene_cfg.setdefault("state", {}).setdefault("registry", {}).setdefault("object_registry", {})

    for obj_name in init_info:
        obj_state = object_registry.setdefault(obj_name, {})
        obj_state.setdefault("is_asleep", False)
        _ensure_root_link_state(obj_state.setdefault("root_link", {}))
        obj_state.setdefault("non_kin", {})


def _extract_openarm_robot(scene_cfg: dict) -> tuple[dict | None, dict]:
    robot_spawn = None
    robot_args = {}
    init_info = scene_cfg["objects_info"]["init_info"]
    object_registry = scene_cfg["state"]["registry"]["object_registry"]

    for obj_name, obj_info in list(init_info.items()):
        if obj_info.get("class_name") == "OpenArmBimanual":
            root_link = object_registry.get(obj_name, {}).get("root_link", {})
            robot_spawn = {
                "position": root_link.get("pos", [0.0, 0.0, 0.0]),
                "orientation": root_link.get("ori", [0.0, 0.0, 0.0, 1.0]),
            }
            robot_args = copy.deepcopy(obj_info.get("args", {}))
            init_info.pop(obj_name, None)
            object_registry.pop(obj_name, None)
            break

    return robot_spawn, robot_args


def _extract_delayed_objects(scene_cfg: dict, object_names: list[str]) -> list[dict]:
    delayed_objects = []
    init_info = scene_cfg["objects_info"]["init_info"]
    object_registry = scene_cfg["state"]["registry"]["object_registry"]

    for obj_name in object_names:
        obj_info = init_info.pop(obj_name, None)
        obj_state = object_registry.pop(obj_name, None)
        if obj_info is None:
            continue
        delayed_objects.append(
            {
                "name": obj_name,
                "class_name": obj_info["class_name"],
                "args": copy.deepcopy(obj_info.get("args", {})),
                "state": copy.deepcopy(obj_state or {}),
            }
        )

    return delayed_objects


def _prepare_direct_scene_file(scene_path: Path, delayed_object_names: list[str]) -> tuple[Path, str, dict | None, dict, list[dict]]:
    scene_cfg = copy.deepcopy(_load_json(scene_path))
    scene_model = scene_cfg.get("init_info", {}).get("args", {}).get("scene_model")
    if not scene_model:
        raise ValueError("scene json 缺少 init_info.args.scene_model")

    scene_cfg.setdefault("objects_info", {}).setdefault("init_info", {})
    scene_cfg.setdefault("state", {}).setdefault("registry", {})
    scene_cfg["state"]["registry"].setdefault("system_registry", {})
    scene_cfg["state"]["registry"].setdefault("object_registry", {})
    _hydrate_scene_state(scene_cfg)
    robot_spawn, robot_args = _extract_openarm_robot(scene_cfg)
    delayed_objects = _extract_delayed_objects(scene_cfg, delayed_object_names)

    cache_dir = REPO_ROOT / ".generation_scene_cache"
    cache_dir.mkdir(exist_ok=True)
    scene_file = cache_dir / f"{scene_path.stem}_direct_visualize.json"
    with open(scene_file, "w", encoding="utf-8") as f:
        json.dump(scene_cfg, f, indent=2)

    return scene_file, scene_model, robot_spawn, robot_args, delayed_objects


def _build_openarm_robot_cfg(args, robot_args: dict | None = None) -> dict:
    robot_args = robot_args or {}
    robot_cfg = {
        "type": "OpenArmBimanual",
        "name": robot_args.get("name", "robot0"),
        "obs_modalities": robot_args.get("obs_modalities") or ["rgb"],
        "action_type": robot_args.get("action_type", "continuous"),
        "action_normalize": robot_args.get("action_normalize", False),
        "fixed_base": robot_args.get("fixed_base", True),
        "grasping_mode": robot_args.get("grasping_mode", "assisted"),
        "scale": args.robot_scale if args.robot_scale is not None else robot_args.get("scale", 1.0),
    }
    for key in ("self_collisions", "reset_joint_pos"):
        if key in robot_args:
            robot_cfg[key] = robot_args[key]
    return robot_cfg


def _build_cfg(args) -> tuple[dict, dict | None, list[dict]]:
    robot_spawn = None
    delayed_objects = []
    if args.template:
        template_path = Path(args.template).resolve()
        scene_file, scene_model, robot_spawn, robot_args, delayed_objects = _prepare_direct_scene_file(
            template_path,
            args.delayed_objects,
        )
        robot_cfg = _build_openarm_robot_cfg(args, robot_args)
        if robot_spawn is not None:
            robot_cfg["position"] = robot_spawn["position"]
            robot_cfg["orientation"] = robot_spawn["orientation"]
        cfg = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene_model,
                "scene_file": str(scene_file),
            },
            "robots": [robot_cfg],
        }
    else:
        robot_cfg = _build_openarm_robot_cfg(args)
        cfg = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": args.scene,
            },
            "robots": [robot_cfg],
        }

    if args.ignore:
        cfg["scene"]["not_load_object_categories"] = args.ignore

    return cfg, robot_spawn, delayed_objects


def _load_delayed_objects(env, delayed_objects: list[dict]) -> None:
    for item in delayed_objects:
        obj_args = copy.deepcopy(item["args"])
        obj = create_class_from_registry_and_config(
            cls_name=item["class_name"],
            cls_registry=REGISTERED_OBJECTS,
            cfg=obj_args,
            cls_type_descriptor="object",
        )
        env.scene.add_object(obj)

        obj_state = copy.deepcopy(item.get("state") or {})
        root_link_state = _ensure_root_link_state(obj_state.setdefault("root_link", {}))
        obj.set_position_orientation(
            position=th.tensor(root_link_state["pos"], dtype=th.float32),
            orientation=th.tensor(root_link_state["ori"], dtype=th.float32),
        )
        if hasattr(obj, "set_linear_velocity"):
            obj.set_linear_velocity(th.tensor(root_link_state["lin_vel"], dtype=th.float32))
        if hasattr(obj, "set_angular_velocity"):
            obj.set_angular_velocity(th.tensor(root_link_state["ang_vel"], dtype=th.float32))
        print(f"[*] Delayed object loaded after robot start: {item['name']} pos={root_link_state['pos']}")


def _openarm_q14_to_sim_q(robot, q14: list[float], finger_pos: float = 0.0) -> th.Tensor:
    q14_tensor = th.tensor(q14, dtype=th.float32)
    sim_q = robot.get_joint_positions().to(dtype=th.float32)
    joint_names = list(robot.joints.keys())
    joint_to_idx = {name: idx for idx, name in enumerate(joint_names)}

    source_names = [f"openarm_left_joint{i}" for i in range(1, 8)] + [
        f"openarm_right_joint{i}" for i in range(1, 8)
    ]
    for source_idx, joint_name in enumerate(source_names):
        if joint_name in joint_to_idx:
            sim_q[joint_to_idx[joint_name]] = q14_tensor[source_idx]

    for joint_name in (
        "openarm_left_finger_joint1",
        "openarm_left_finger_joint2",
        "openarm_right_finger_joint1",
        "openarm_right_finger_joint2",
    ):
        if joint_name in joint_to_idx:
            sim_q[joint_to_idx[joint_name]] = finger_pos

    return sim_q


def _set_openarm_sim_q(robot, sim_q: th.Tensor, drive: bool = False) -> None:
    robot.set_joint_positions(sim_q, drive=drive)
    if not drive:
        robot.set_joint_velocities(th.zeros(robot.n_dof, dtype=th.float32), drive=False)


def _run_openarm_startup_sequence(robot, steps_per_segment: int) -> None:
    steps_per_segment = max(1, int(steps_per_segment))
    q14_waypoints = [th.tensor(q, dtype=th.float32) for q in OPENARM_STARTUP_QS_14]

    _set_openarm_sim_q(robot, _openarm_q14_to_sim_q(robot, q14_waypoints[0].tolist()), drive=False)
    og.sim.step()

    for start_q, end_q in zip(q14_waypoints[:-1], q14_waypoints[1:]):
        for step in range(1, steps_per_segment + 1):
            ratio = step / steps_per_segment
            q14 = (1.0 - ratio) * start_q + ratio * end_q
            _set_openarm_sim_q(robot, _openarm_q14_to_sim_q(robot, q14.tolist()), drive=False)
            og.sim.step()

    final_q = _openarm_q14_to_sim_q(robot, q14_waypoints[-1].tolist())
    _set_openarm_sim_q(robot, final_q, drive=False)
    _set_openarm_sim_q(robot, final_q, drive=True)


def main():
    parser = argparse.ArgumentParser(description="Visualize an OmniGibson scene or a scene json.")
    parser.add_argument(
        "--scene",
        "-s",
        type=str,
        default="grocery_store_convenience",
        help="Scene model to load when --template is not provided.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Path to a scene json. The file is visualized directly.",
    )
    parser.add_argument(
        "--base-scene-variant",
        choices=["auto", "best", "with_clutter"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ignore",
        "-i",
        type=str,
        nargs="*",
        default=["drink_dispenser"],
        help="Object categories to ignore / not load.",
    )
    parser.add_argument(
        "--robot-scale",
        type=float,
        default=None,
        help="Override OpenArmBimanual scale. Defaults to the scale in the scene json.",
    )
    parser.add_argument(
        "--openarm-startup",
        action="store_true",
        help="Run the OpenArm Q0->Q1->Q2->Q3 startup sequence before showing the viewer.",
    )
    parser.add_argument(
        "--skip-openarm-startup",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--openarm-startup-steps",
        type=int,
        default=90,
        help="Interpolation steps per OpenArm startup segment.",
    )
    parser.add_argument(
        "--delayed-objects",
        type=str,
        nargs="*",
        default=list(DEFAULT_DELAYED_OBJECTS),
        help="Object names to load after the OpenArm has been placed in its start state.",
    )
    parser.add_argument(
        "--preview-joint-object",
        type=str,
        default=None,
        help="Object name whose joints should be posed for visualization.",
    )
    parser.add_argument(
        "--preview-joint-positions",
        type=float,
        nargs="+",
        default=None,
        help="Joint positions applied to --preview-joint-object after scene loading.",
    )
    parser.add_argument("--steps", type=int, default=1000000, help="Simulation steps before closing.")
    args = parser.parse_args()

    cfg, robot_spawn, delayed_objects = _build_cfg(args)

    if args.template:
        print(f"[*] Loading scene json directly: {args.template}")
        if robot_spawn is not None:
            print(f"[*] Scene robot pose: pos={robot_spawn['position']} ori={robot_spawn['orientation']}")
    else:
        print(f"[*] Loading scene: {args.scene}")

    if args.ignore:
        print(f"[*] Ignoring object categories: {args.ignore}")
    print(f"[*] Robot scale: {cfg['robots'][0].get('scale')}")

    env = og.Environment(configs=cfg)
    if robot_spawn is not None:
        env.robots[0].set_position_orientation(
            position=th.tensor(robot_spawn["position"], dtype=th.float32),
            orientation=th.tensor(robot_spawn["orientation"], dtype=th.float32),
        )
        og.sim.step()
        scene_identifier = str(args.template or args.scene)
        if uses_real_exp_1_viewer_camera(scene_identifier):
            set_real_exp_1_viewer_camera(og, print_prefix="[*]")
        else:
            viewer_pos = th.tensor([0.84451, 4.1226, 2.57063], dtype=th.float32)
            viewer_ori = th.tensor(
                R.from_euler("zyx", [80.176, 50.592, 7.621], degrees=True).as_quat(),
                dtype=th.float32,
            )
            og.sim.viewer_camera.horizontal_aperture = 35.0
            og.sim.viewer_camera.set_position_orientation(
                position=viewer_pos,
                orientation=viewer_ori,
            )
            print(f"[*] Cashier viewer camera: pos={viewer_pos.tolist()} quat={viewer_ori.tolist()}")

    if robot_spawn is not None and args.openarm_startup and not args.skip_openarm_startup:
        print("[*] Running OpenArm startup sequence: Q0 -> Q1 -> Q2 -> Q3")
        _run_openarm_startup_sequence(env.robots[0], args.openarm_startup_steps)

    if delayed_objects:
        _load_delayed_objects(env, delayed_objects)

    if args.preview_joint_object:
        obj = env.scene.object_registry("name", args.preview_joint_object)
        if obj is None:
            raise ValueError(f"Preview joint object not found: {args.preview_joint_object}")
        positions = args.preview_joint_positions
        if positions is None or len(positions) != obj.n_dof:
            raise ValueError(
                f"{args.preview_joint_object} requires {obj.n_dof} joint positions, got {positions}"
            )
        obj.set_joint_positions(th.tensor(positions, dtype=th.float32), drive=False)
        obj.set_joint_velocities(th.zeros(obj.n_dof, dtype=th.float32), drive=False)
        og.sim.step()
        print(f"[*] Preview joints: {args.preview_joint_object}={positions}")

    print("\n[!] Viewer ready.")
    print("[!] Press right click to rotate the view.")
    print("[!] Hold right click + WASD / Q E to fly the camera around.\n")

    for _ in range(args.steps):
        og.sim.step()

    if args.steps < 1000000:
        import os
        print(f"[smoke] Scene stepped {args.steps} time(s); startup verified.", flush=True)
        os._exit(0)

    env.close()


if __name__ == "__main__":
    main()
