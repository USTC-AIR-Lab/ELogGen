"""
Collection of utilities related to robomimic.
"""
import argparse
import json
import os
import traceback
from copy import deepcopy
from pathlib import Path

import robomimic
import robomimic.utils.env_utils as EnvUtils
from robomimic.scripts.playback_dataset import playback_dataset, DEFAULT_CAMERAS

from eloggen.generation_runtime.simulation.camera_config import (
    PUBLISHED_OPENARM_TASKS,
    canonical_openarm_task_name,
    runtime_openarm_task_name,
)


def _override_env_identity(env_meta, env_name):
    """Override both robomimic and OmniGibson task identities.

    Historical processed HDF5 files can carry an ``env_kwargs.task.activity_name``
    from the source task. Robomimic's OmniGibson construction path may use that
    nested activity name when creating the wrapper, so changing only top-level
    ``env_name`` is insufficient.
    """
    env_meta["env_name"] = env_name
    try:
        is_omnigibson = EnvUtils.get_env_type(env_meta=env_meta) == EnvUtils.EB.EnvType.OG_TYPE
    except Exception:
        is_omnigibson = False

    activity_name = canonical_openarm_task_name(env_name)
    if is_omnigibson and activity_name in PUBLISHED_OPENARM_TASKS:
        env_meta.setdefault("env_kwargs", {}).setdefault("task", {})["activity_name"] = activity_name
    return env_meta


def _repair_legacy_openarm_scene_file(env_meta):
    """Repair legacy OpenArm scene paths that were serialized to /tmp in source datasets."""
    if EnvUtils.get_env_type(env_meta=env_meta) != EnvUtils.EB.EnvType.OG_TYPE:
        return env_meta

    scene_cfg = env_meta.get("env_kwargs", {}).get("scene", {})
    scene_file = scene_cfg.get("scene_file")
    if not isinstance(scene_file, str) or os.path.exists(scene_file):
        return env_meta

    basename = os.path.basename(scene_file)
    if not basename.endswith(".json"):
        return env_meta

    grasp_modes = ("sticky", "assisted", "physical")
    missing_scene_stem = basename[:-5]
    grasping_mode = next((mode for mode in grasp_modes if missing_scene_stem.endswith(f"_{mode}")), None)
    if grasping_mode is None:
        return env_meta

    scene_model = scene_cfg.get("scene_model")
    if scene_model is None:
        return env_meta

    task_name = env_meta.get("env_kwargs", {}).get("task", {}).get("activity_name")
    if not isinstance(task_name, str):
        return env_meta

    repo_root = Path(__file__).resolve().parents[3]
    packaged_scene_file = repo_root / "src" / "eloggen" / "datasets" / "taskpacks" / task_name / "scene.json"
    if not packaged_scene_file.exists():
        return env_meta

    with open(packaged_scene_file, "r", encoding="utf-8") as file_obj:
        repaired_scene_cfg = json.load(file_obj)

    robot_args = repaired_scene_cfg.get("objects_info", {}).get("init_info", {}).get("robot0", {}).get("args")
    if isinstance(robot_args, dict):
        robot_args["grasping_mode"] = grasping_mode

    cache_dir = repo_root / ".generation_scene_cache"
    cache_dir.mkdir(exist_ok=True)
    repaired_scene_path = cache_dir / basename
    with open(repaired_scene_path, "w", encoding="utf-8") as file_obj:
        json.dump(repaired_scene_cfg, file_obj, indent=4)

    env_meta["env_kwargs"]["scene"]["scene_file"] = str(repaired_scene_path)
    print(f"[ElogGen generation runtime] Repaired missing scene file {scene_file} -> {repaired_scene_path}")
    return env_meta


def create_env(
    env_meta,
    env_name=None,
    env_class=None,
    robot=None,
    gripper=None,
    camera_names=None,
    camera_height=84,
    camera_width=84,
    render=None,
    render_offscreen=None,
    use_image_obs=None,
    use_depth_obs=None,
    init_curobo=True,
    policy_rollout=False,
    manipulation_only=False,
    real_robot_mode=False,
    baseline=None
):
    """
    Helper function to create the environment from dataset metadata and arguments.

    Args:
        env_meta (dict): environment metadata compatible with robomimic, see
            https://robomimic.github.io/docs/modules/environments.html
        env_name (str or None): if provided, override environment name
            in @env_meta
        env_class (class or None): if provided, use this class instead of the
            one inferred from @env_meta
        robot (str or None): if provided, override the robot argument in
            @env_meta. Currently only supported by robosuite environments.
        gripper (str or None): if provided, override the gripper argument in
            @env_meta. Currently only supported by robosuite environments.
        camera_names (list of str or None): list of camera names that correspond to image observations
        camera_height (int): camera height for all cameras
        camera_width (int): camera width for all cameras
        render (bool or None): optionally override rendering behavior
        render_offscreen (bool or None): optionally override rendering behavior
        use_image_obs (bool or None): optionally override rendering behavior
        use_depth_obs (bool or None): optionally override rendering behavior
    """
    env_meta = deepcopy(env_meta)

    # Normalize all bundled OpenArm base names to explicit runtime variants so
    # reset(), difficulty selection, and task-object lookup follow one convention
    # no matter which entry point created the environment.
    if env_name is not None:
        env_name = runtime_openarm_task_name(env_name)
        env_meta = _override_env_identity(env_meta, env_name)
    env_meta = _repair_legacy_openarm_scene_file(env_meta)

    # maybe override some settings in environment metadata
    if robot is not None:
        # for now, only support this argument for robosuite environments
        assert EnvUtils.is_robosuite_env(env_meta)
        assert robot in ["IIWA", "Sawyer", "UR5e", "Panda", "Jaco", "Kinova3"]
        env_meta["env_kwargs"]["robots"] = [robot]
    if gripper is not None:
        # for now, only support this argument for robosuite environments
        assert EnvUtils.is_robosuite_env(env_meta)
        assert gripper in ["PandaGripper", "RethinkGripper", "Robotiq85Gripper", "Robotiq140Gripper"]
        env_meta["env_kwargs"]["gripper_types"] = [gripper]

    if camera_names is None:
        camera_names = []

    # create environment
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        env_class=env_class,
        camera_names=camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        reward_shaping=False,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        use_depth_obs=use_depth_obs,
        init_curobo=init_curobo,
        policy_rollout=policy_rollout,
        manipulation_only=manipulation_only,
        real_robot_mode=real_robot_mode,
        baseline=baseline,
    )

    # Some legacy OmniGibson wrappers derive ``_env_name`` from nested source
    # metadata even when robomimic received the requested top-level env_name.
    # Keep the wrapper identity consistent with the generation task as a final
    # safeguard; the nested Behavior activity has already been fixed above.
    if env_name is not None and hasattr(env, "_env_name") and getattr(env, "name", None) != env_name:
        print(
            "[ElogGen generation runtime] overriding wrapper env name: {} -> {}".format(
                getattr(env, "name", None), env_name
            )
        )
        env._env_name = env_name

    return env


def make_dataset_video(
    dataset_path,
    video_path,
    num_render=None,
    render_image_names=None,
    use_obs=False,
    video_skip=5,
):
    """
    Helper function to set up args and call @playback_dataset from robomimic
    to get video of generated dataset.
    """
    print("\nmake_dataset_video(\n\tdataset_path={},\n\tvideo_path={},{}\n)".format(
        dataset_path,
        video_path,
        "\n\tnum_render={},".format(num_render) if num_render is not None else "",
    ))
    playback_args = argparse.Namespace()
    playback_args.dataset = dataset_path
    playback_args.filter_key = None
    playback_args.n = num_render
    playback_args.use_obs = use_obs
    playback_args.use_actions = False
    playback_args.absolute = False
    playback_args.intervention = False
    playback_args.render = False
    playback_args.video_path = video_path
    playback_args.video_skip = video_skip
    playback_args.render_image_names = render_image_names
    if (render_image_names is None):
        # default robosuite
        playback_args.render_image_names = ["agentview"]
    playback_args.render_depth_names = None
    playback_args.first = False

    try:
        playback_dataset(playback_args)
    except Exception as e:
        res_str = "playback failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        print(res_str)


def get_default_env_cameras(env_meta):
    """
    Get the default set of cameras for a particular robomimic environment type.

    Args:
        env_meta (dict): environment metadata compatible with robomimic, see
            https://robomimic.github.io/docs/modules/environments.html

    Returns:
        camera_names (list of str): list of camera names that correspond to image observations
    """
    return DEFAULT_CAMERAS[EnvUtils.get_env_type(env_meta=env_meta)]
