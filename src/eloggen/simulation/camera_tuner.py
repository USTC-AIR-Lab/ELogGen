#!/usr/bin/env python3
"""
Interactive OpenArm camera pose tuner.

Loads an OpenArm scene and three temporary external cameras without running
ElogGen generation runtime data generation. Move camera prims in the Isaac Sim viewport, then
press P (or type p + Enter in the terminal) to save their poses relative to
the configured OpenArm links.
"""

import argparse
import copy
import json
import os
import sys
import threading
import time
from pathlib import Path

from eloggen.simulation.omnigibson import activate, project_root
from eloggen.datasets import taskpack_path

import torch as th


def _add_og_path() -> None:
    og_root = Path(__file__).resolve().parents[2] / "BEHAVIOR-1K" / "OmniGibson"
    if str(og_root) not in sys.path:
        sys.path.insert(0, str(og_root))


os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "True")
activate()

import omnigibson as og  # noqa: E402
import omnigibson.lazy as lazy  # noqa: E402
from omnigibson.utils import transform_utils as T  # noqa: E402
from eloggen.generation_runtime.simulation.camera import (  # noqa: E402
    is_openarm_real_exp_1,
    set_real_exp_1_viewer_camera,
    uses_real_exp_1_viewer_camera,
)
from eloggen.generation_runtime.simulation.camera_config import (  # noqa: E402
    get_real_exp_1_external_sensor_kwargs,
    square_camera_sensor_kwargs,
)


DEFAULT_MOUNTS = {
    "left_wrist_cam": {
        "link": "openarm_left_link6",
        "pos": [0.05, 0.0, -0.05],
        "quat": [-0.0923, -0.7011, -0.7011, -0.0923],
    },
    "right_wrist_cam": {
        "link": "openarm_right_link6",
        "pos": [0.05, 0.0, -0.05],
        "quat": [-0.0923, -0.7011, -0.7011, -0.0923],
    },
    "base_cam": {
        "link": "openarm_body_link0",
        "pos": [1.37, 0.03, 1.49],
        "quat": [0.3403, 0.3626, 0.6326, 0.5937],
    },
}


def _task_mounts_path(task_name):
    return str(taskpack_path(task_name) / "cameras.json")


def _task_scene_path(task_name):
    return str(taskpack_path(task_name) / "scene.json")


SCENE_PRESETS = {
    "openarm_real_exp_1": {
        "scene_model": "grocery_store_convenience",
        "scene_file": _task_scene_path("openarm_real_exp_1"),
        "task_name": "openarm_real_exp_1",
        "mounts": _task_mounts_path("openarm_real_exp_1"),
    },
    "openarm_drawer_storage": {
        "scene_model": "grocery_store_convenience",
        "scene_file": _task_scene_path("openarm_drawer_storage"),
        "task_name": "openarm_drawer_storage",
        "mounts": _task_mounts_path("openarm_drawer_storage"),
    },
    "openarm_fruit_basket_bagging": {
        "scene_model": "grocery_store_convenience",
        "scene_file": _task_scene_path("openarm_fruit_basket_bagging"),
        "task_name": "openarm_fruit_basket_bagging",
        "mounts": _task_mounts_path("openarm_fruit_basket_bagging"),
    },
}


def _load_json(path, default):
    if not os.path.exists(path):
        return copy.deepcopy(default)
    with open(path, "r") as f:
        data = json.load(f)
    merged = copy.deepcopy(default)
    for name, cfg in data.items():
        merged.setdefault(name, {})
        merged[name].update(cfg)
    return merged


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def _build_env(args, mounts):
    repo_root = project_root()
    scene_file = Path(args.scene_file)
    if not scene_file.is_absolute():
        scene_file = repo_root / scene_file
    scene_file = os.path.normpath(str(scene_file))
    with open(scene_file, "r", encoding="utf-8") as f:
        scene_cfg = json.load(f)
    scene_cfg = copy.deepcopy(scene_cfg)
    scene_cfg["objects_info"]["init_info"]["robot0"]["args"]["grasping_mode"] = args.grasping_mode

    scene_cache_dir = repo_root / ".generation_scene_cache"
    scene_cache_dir.mkdir(exist_ok=True)
    tmp_scene_file = scene_cache_dir / f"{Path(scene_file).stem}_{args.grasping_mode}_camera_tune.json"
    with open(tmp_scene_file, "w", encoding="utf-8") as f:
        json.dump(scene_cfg, f, indent=4)

    external_sensors = []
    for name in mounts:
        sensor_kwargs = (
            get_real_exp_1_external_sensor_kwargs(name)
            if uses_real_exp_1_viewer_camera(args.task_name)
            else square_camera_sensor_kwargs(args.camera_size)
        )
        external_sensors.append(
            {
                "sensor_type": "VisionSensor",
                "name": name,
                "relative_prim_path": f"/camera_tuning/{name}",
                "modalities": ["rgb"],
                "sensor_kwargs": sensor_kwargs,
                "position": [0.0, 0.0, 1.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "include_in_obs": True,
            }
        )

    cfg = {
        "env": {
            "external_sensors": external_sensors,
            "flatten_obs_space": True,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": args.scene_model,
            "scene_file": str(tmp_scene_file),
        },
        "task": {
            "type": "BehaviorTask",
            "activity_name": args.task_name,
            "activity_definition_id": args.activity_definition_id,
            "activity_instance_id": args.activity_instance_id,
            "online_object_sampling": False,
            "use_presampled_robot_pose": False,
        },
        "robots": [],
    }
    return og.Environment(configs=cfg), scene_cfg


def _restore_scene_object_pose(env, scene_cfg, obj_name, preserve_current_z=False):
    object_registry = scene_cfg.get("state", {}).get("registry", {}).get("object_registry", {})
    state = object_registry.get(obj_name, {})
    root_link = state.get("root_link", {})
    if "pos" not in root_link or "ori" not in root_link:
        return
    obj = env.scene.object_registry("name", obj_name)
    if obj is None:
        return

    target_pos = th.tensor(root_link["pos"], dtype=th.float32)
    if preserve_current_z:
        current_pos, _ = obj.get_position_orientation()
        target_pos[2] = th.as_tensor(current_pos, dtype=th.float32)[2]
    obj.set_position_orientation(
        position=target_pos,
        orientation=th.tensor(root_link["ori"], dtype=th.float32),
    )
    if hasattr(obj, "set_linear_velocity"):
        obj.set_linear_velocity(th.zeros(3, dtype=th.float32))
    if hasattr(obj, "set_angular_velocity"):
        obj.set_angular_velocity(th.zeros(3, dtype=th.float32))


def _stabilize_real_exp_1_tuning_scene(env, scene_cfg):
    for _ in range(80):
        og.sim.step()

    _restore_scene_object_pose(env, scene_cfg, "paper_bag_1")
    for obj_name in ("object_1", "object_2"):
        _restore_scene_object_pose(env, scene_cfg, obj_name, preserve_current_z=True)


def _stabilize_fruit_basket_bagging_tuning_scene(env, scene_cfg):
    for _ in range(80):
        og.sim.step()

    for obj_name in ("paper_bag_1", "basket_1"):
        _restore_scene_object_pose(env, scene_cfg, obj_name)
    for obj_name in ("object_1", "object_2", "object_3", "object_4"):
        _restore_scene_object_pose(env, scene_cfg, obj_name, preserve_current_z=True)

    for _ in range(10):
        og.sim.step()


def _place_cameras_from_mounts(env, mounts):
    robot = env.robots[0]
    for name, mount in mounts.items():
        sensor = env.external_sensors.get(name)
        link = robot.links.get(mount["link"])
        if sensor is None or link is None:
            print(f"[WARN] skip {name}: sensor or link {mount['link']} not found")
            continue
        link_pos, link_quat = link.get_position_orientation()
        local_pos = th.tensor(mount["pos"], dtype=th.float32)
        local_quat = th.tensor(mount["quat"], dtype=th.float32)
        cam_pos, cam_quat = T.pose_transform(link_pos, link_quat, local_pos, local_quat)
        sensor.set_position_orientation(position=cam_pos, orientation=cam_quat, frame="world")


def _read_mounts_from_scene(env, mounts):
    robot = env.robots[0]
    updated = copy.deepcopy(mounts)
    for name, mount in mounts.items():
        sensor = env.external_sensors.get(name)
        link = robot.links.get(mount["link"])
        if sensor is None or link is None:
            print(f"[WARN] skip {name}: sensor or link {mount['link']} not found")
            continue
        cam_pos, cam_quat = sensor.get_position_orientation()
        link_pos, link_quat = link.get_position_orientation()
        rel_pos, rel_quat = T.relative_pose_transform(cam_pos, cam_quat, link_pos, link_quat)
        updated[name]["pos"] = [round(float(x), 6) for x in rel_pos.tolist()]
        updated[name]["quat"] = [round(float(x), 6) for x in rel_quat.tolist()]
    return updated


def _print_mounts(mounts):
    print(json.dumps(mounts, indent=4))


def _register_controls(flags):
    try:
        def _kb_handler(event, *_, **__):
            is_press = (
                event.type == lazy.carb.input.KeyboardEventType.KEY_PRESS
                or event.type == lazy.carb.input.KeyboardEventType.KEY_REPEAT
            )
            if is_press:
                if event.input == lazy.carb.input.KeyboardInput.P:
                    flags["save"] = True
                elif event.input == lazy.carb.input.KeyboardInput.L:
                    flags["print"] = True
                elif event.input == lazy.carb.input.KeyboardInput.R:
                    flags["reset"] = True
                elif event.input == lazy.carb.input.KeyboardInput.ESCAPE:
                    flags["stop"] = True
            return True

        appwindow = lazy.omni.appwindow.get_default_app_window()
        input_iface = lazy.carb.input.acquire_input_interface()
        input_iface.subscribe_to_keyboard_events(appwindow.get_keyboard(), _kb_handler)
        print("[controls] GUI keyboard: P save, L print, R reset from JSON, ESC quit")
    except Exception as e:
        print(f"[WARN] GUI keyboard registration failed: {e}")

    def _stdin_reader():
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "p":
                flags["save"] = True
            elif cmd == "l":
                flags["print"] = True
            elif cmd == "r":
                flags["reset"] = True
            elif cmd in ("q", "esc", "exit"):
                flags["stop"] = True
                break

    threading.Thread(target=_stdin_reader, daemon=True).start()
    print("[controls] terminal: p save, l print, r reset from JSON, q quit")


def main():
    parser = argparse.ArgumentParser(description="Tune OpenArm external camera poses interactively.")
    parser.add_argument(
        "--preset",
        default="pick",
        choices=list(SCENE_PRESETS.keys()) + ["custom"],
        help="Scene preset. Use custom with --scene-file, --scene-model, and --task-name.",
    )
    parser.add_argument("--scene-file", default=None, help="OpenArm scene JSON to load.")
    parser.add_argument("--scene-model", default=None, help="OmniGibson scene model name.")
    parser.add_argument(
        "--mounts",
        default=None,
        help="JSON file to load/save camera mounts.",
    )
    parser.add_argument("--task-name", default=None, help="BEHAVIOR task activity name.")
    parser.add_argument("--activity-definition-id", type=int, default=0)
    parser.add_argument("--activity-instance-id", type=int, default=0)
    parser.add_argument("--grasping-mode", default="sticky", choices=["physical", "assisted", "sticky"])
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=None, help="Finite simulation steps for a startup smoke test.")
    args = parser.parse_args()

    if args.preset != "custom":
        preset = SCENE_PRESETS[args.preset]
        args.scene_file = args.scene_file or preset["scene_file"]
        args.scene_model = args.scene_model or preset["scene_model"]
        args.task_name = args.task_name or preset["task_name"]
        args.mounts = args.mounts or preset.get("mounts") or _task_mounts_path(args.task_name)
    else:
        if args.mounts is None and args.task_name is not None:
            args.mounts = _task_mounts_path(args.task_name)
        args.mounts = args.mounts or "generation_runtime/configs/openarm_camera_mounts.json"
    missing = [name for name in ("scene_file", "scene_model", "task_name") if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments for custom scene: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))

    mounts_path = os.path.abspath(args.mounts)
    mounts = _load_json(mounts_path, DEFAULT_MOUNTS)
    env, source_scene_cfg = _build_env(args, mounts)
    for _ in range(20):
        og.sim.step()
    if args.task_name == "openarm_fruit_basket_bagging":
        _stabilize_fruit_basket_bagging_tuning_scene(env, source_scene_cfg)
    elif is_openarm_real_exp_1(args.task_name) or args.task_name == "openarm_drawer_storage":
        _stabilize_real_exp_1_tuning_scene(env, source_scene_cfg)
    if uses_real_exp_1_viewer_camera(args.task_name):
        set_real_exp_1_viewer_camera(og, print_prefix=f"[viewer_camera][{args.task_name}]")
    _place_cameras_from_mounts(env, mounts)

    print("\nCamera prims:")
    for name, sensor in env.external_sensors.items():
        print(f"  {name}: {sensor.prim_path}")
    print(f"\nScene preset: {args.preset}")
    print(f"Scene file: {args.scene_file}")
    print(f"Scene model: {args.scene_model}")
    print(f"Task name: {args.task_name}")
    print(f"\nSaving mounts to: {mounts_path}")
    print("Move camera prims in the viewport, then press P to save relative poses.\n")
    _print_mounts(mounts)

    flags = {"save": False, "print": False, "reset": False, "stop": False}
    _register_controls(flags)
    step_count = 0

    try:
        while not flags["stop"]:
            og.sim.step()
            step_count += 1
            if args.steps is not None and step_count >= args.steps:
                print(f"[smoke] Camera rig stepped {step_count} time(s); startup verified.", flush=True)
                os._exit(0)
            if flags["reset"]:
                mounts = _load_json(mounts_path, DEFAULT_MOUNTS)
                _place_cameras_from_mounts(env, mounts)
                print("[reset] cameras restored from JSON")
                flags["reset"] = False
            if flags["print"] or flags["save"]:
                mounts = _read_mounts_from_scene(env, mounts)
                _print_mounts(mounts)
                if flags["save"]:
                    _write_json(mounts_path, mounts)
                    print(f"[save] wrote {mounts_path}")
                flags["print"] = False
                flags["save"] = False
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()