"""OmniGibson dataset-generation runtime used by ``GenerationExecutor``."""

import os
import sys
import shutil
import json
import time
import argparse
import traceback
import random
import re
import imageio
import numpy as np
import torch as th
import warnings
import logging
from copy import deepcopy

# Ensure local packages are used when running this script directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_IMPORT_PATHS = [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "BEHAVIOR-1K", "OmniGibson"),
]
for path in reversed(LOCAL_IMPORT_PATHS):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

# Configure logging and warnings
th.set_printoptions(precision=3, sci_mode=False, linewidth=1000)
warnings.filterwarnings('ignore', module='trimesh')
logging.getLogger('trimesh').setLevel(logging.ERROR)
logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

from robomimic.utils.file_utils import get_env_metadata_from_dataset

import robomimic.utils.env_utils as EnvUtils
import eloggen.generation_runtime.datasets as DatasetUtils
import eloggen.generation_runtime.playback as Playback

from eloggen.generation_runtime.config.base import config_factory
from eloggen.generation_runtime.config.task_spec import EG_TaskSpec
from eloggen.generation_runtime.context.hdf5 import write_frame_context_report
from eloggen.generation_runtime.generator.generator import DataGenerator
from eloggen.generation_runtime.simulation.base import make_interface

import omnigibson as og

from omnigibson.objects.primitive_object import PrimitiveObject

# Disable pyembree for trimesh
os.environ["TRIMESH_NO_PYEMBREE"] = "1"


DEFAULT_GENERATION_VIDEO_FPS = 30
DATASET_OUTPUT_FORMATS = ("hdf5", "lerobot", "both")
DEFAULT_LEROBOT_DATASET_ROOT = os.path.join(REPO_ROOT, "outputs", "lerobot_dataset")
OPENARM_REAL_EXP_1_GRASP_ORDER_BY_NAME = {
    "apple_first": "object_1,object_2",
    "object_1_first": "object_1,object_2",
    "obj1_first": "object_1,object_2",
    "lemon_first": "object_2,object_1",
    "lime_first": "object_2,object_1",
    "object_2_first": "object_2,object_1",
    "obj2_first": "object_2,object_1",
}


def _parse_grasp_order_quotas(spec):
    if spec is None:
        return {}
    spec = str(spec).strip()
    if not spec:
        return {}

    if spec.isdigit():
        per_order = int(spec)
        return {
            "apple_first": per_order,
            "lemon_first": per_order,
        }

    quotas = {}
    for item in re.split(r"[;,]", spec):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "--grasp-order-quota entries must be COUNT or name=count, "
                f"got {item!r}"
            )
        name, value = item.split("=", 1)
        name = name.strip().lower().replace("-", "_")
        if name not in OPENARM_REAL_EXP_1_GRASP_ORDER_BY_NAME:
            raise ValueError(
                "Unknown grasp-order quota name {!r}. Supported names include: {}".format(
                    name,
                    ", ".join(sorted(set(OPENARM_REAL_EXP_1_GRASP_ORDER_BY_NAME))),
                )
            )
        canonical_name = "apple_first" if OPENARM_REAL_EXP_1_GRASP_ORDER_BY_NAME[name] == "object_1,object_2" else "lemon_first"
        quotas[canonical_name] = int(value)
    return quotas


def _grasp_order_name_from_metadata(episode_metadata):
    if not episode_metadata:
        return None
    grasp_order = episode_metadata.get("grasp_order")
    if not grasp_order:
        return None
    grasp_order = tuple(str(item) for item in grasp_order)
    if grasp_order == ("object_1", "object_2"):
        return "apple_first"
    if grasp_order == ("object_2", "object_1"):
        return "lemon_first"
    return ",".join(grasp_order)


class CameraVideoWriter:
    """Thin wrapper that carries camera names alongside an imageio writer."""

    def __init__(self, path, camera_names, fps=DEFAULT_GENERATION_VIDEO_FPS):
        self.path = path
        self.camera_names = list(camera_names) if camera_names is not None else None
        self._writer = imageio.get_writer(path, fps=fps)

    def append_data(self, image):
        self._writer.append_data(image)

    def close(self):
        self._writer.close()

def visualize_base_poses(env):
    """Visualize base poses with colored markers (debug function)."""
    sampled_base_poses = env.sampled_base_poses

    # Create failure markers (red)
    _create_pose_markers(
        positions=sampled_base_poses["failure"],
        prefix="base_marker_failure",
        color=th.tensor([1, 0, 0, 1]),
        env=env
    )

    # Create success markers (green)
    _create_pose_markers(
        positions=sampled_base_poses["success"],
        prefix="base_marker_success",
        color=th.tensor([0, 1, 0, 1]),
        env=env
    )

def _create_pose_markers(positions, prefix, color, env):
    """Helper to create visualization markers."""
    base_marker_list = []
    for i in range(len(positions)):
        base_marker = PrimitiveObject(
            relative_prim_path=f"/{prefix}_{i}",
            primitive_type="Cube",
            name=f"{prefix}_{i}",
            size=th.tensor([0.03, 0.03, 0.03]),
            visual_only=True,
            rgba=color
        )
        base_marker_list.append(base_marker)

    if base_marker_list:
        og.sim.batch_add_objects(base_marker_list, [env.env.scene] * len(base_marker_list))
        for i, pos in enumerate(positions):
            base_marker_list[i].set_position_orientation(position=pos)

def get_important_stats(
    new_dataset_folder_path,
    num_success,
    num_failures,
    num_attempts,
    num_problematic,
    ep_lengths,
    start_time=None,
    ep_length_stats=None,
    all_episode_logs=None
):
    """
    Return a summary of important stats to write to json.

    Args:
        new_dataset_folder_path (str): path to folder that will contain generated dataset
        num_success (int): number of successful trajectories generated
        num_failures (int): number of failed trajectories
        num_attempts (int): number of total attempts
        num_problematic (int): number of problematic trajectories that failed due
            to a specific exception that was caught
        start_time (float or None): starting time for this run from time.time()
        ep_length_stats (dict or None): if provided, should have entries that summarize
            the episode length statistics over the successfully generated trajectories

    Returns:
        important_stats (dict): dictionary with useful summary of statistics
    """
    important_stats = dict(
        generation_path=new_dataset_folder_path,
        success_rate=((100. * num_success) / num_attempts),
        failure_rate=((100. * num_failures) / num_attempts),
        num_success=num_success,
        num_failures=num_failures,
        num_attempts=num_attempts,
        num_problematic=num_problematic,
        ep_lengths=ep_lengths,
        all_episode_logs=all_episode_logs

    )
    if (ep_length_stats is not None):
        important_stats.update(ep_length_stats)
    if start_time is not None:
        # add in time taken
        important_stats["time spent (hrs)"] = "{:.2f}".format((time.time() - start_time) / 3600.)
    return important_stats


def _base_task_name(task_name):
    return re.sub(r"_D\d+$", "", str(task_name))


def _infer_lerobot_output_path(new_dataset_folder_path, ordered_tasks=False):
    root = os.path.abspath(os.path.expanduser(new_dataset_folder_path))
    if os.path.basename(root).startswith("demo_src_"):
        root = os.path.dirname(root)
    repo_outputs_root = os.path.join(REPO_ROOT, "outputs")
    if os.path.commonpath([root, DEFAULT_LEROBOT_DATASET_ROOT]) == DEFAULT_LEROBOT_DATASET_ROOT:
        pass
    elif os.path.commonpath([root, repo_outputs_root]) == repo_outputs_root:
        rel_output = os.path.relpath(root, repo_outputs_root)
        root = os.path.join(DEFAULT_LEROBOT_DATASET_ROOT, rel_output)
    elif "outputs" in root.split(os.sep):
        parts = root.split(os.sep)
        idx = parts.index("outputs")
        inferred_parts = parts[: idx + 1] + ["lerobot_dataset"] + parts[idx + 1 :]
        root = os.sep.join(inferred_parts)
    else:
        root = f"{root}_lerobot"
    if ordered_tasks and not root.endswith("_ordered"):
        root = f"{root}_ordered"
    return root


def _infer_lerobot_repo_id(task_name, ordered_tasks=False):
    base_name = _base_task_name(task_name)
    if ordered_tasks and not base_name.endswith("_ordered"):
        base_name = f"{base_name}_ordered"
    return f"openarm/{base_name}"


def _configured_obs_key_candidates(camera_names):
    candidates = set()
    for camera_name in camera_names:
        camera_name = str(camera_name)
        candidates.add(camera_name)
        if "::" not in camera_name:
            candidates.add(f"external::{camera_name}::rgb")
    return candidates


def generate_dataset(
    generation_config,
    auto_remove_exp=False,
    render=False,
    no_save_video=False,
    video_skip=5,
    render_image_names=None,
    pause_subtask=False,
    bimanual=False,
    enable_marker_vis=False,
    ds_ratio=1,
    no_partial_tasks=False,
    headless=False,
    baseline=None,
    robot_type="OpenArm",
    print_stage_type=False,
    grasp_order="random",
    subtask_order=None,
    video_fps=DEFAULT_GENERATION_VIDEO_FPS,
    output_format="hdf5",
    lerobot_output=None,
    lerobot_repo_id=None,
    lerobot_task=None,
    lerobot_robot_type="openarm",
    lerobot_fps=None,
    lerobot_ordered_tasks=False,
    lerobot_overwrite=False,
    lerobot_resume=False,
    lerobot_checkpoint_every=10,
    grasp_order_quota=None,
    scene_file=None,
):
    """
    Main function to collect a new dataset with ElogGen generation runtime.

    Args:
        generation_config (EG_Config instance): ElogGen generation runtime config object

        auto_remove_exp (bool): if True, will remove generation folder if it exists, else
            user will be prompted to decide whether to keep existing folder or not

        render (bool): if True, render each data generation attempt on-screen

        no_save_video (bool): if True, don't save video of data generation attempts 

        video_skip (int): skip every nth frame when writing video

        render_image_names (list of str or None): if provided, specify camera names to 
            use during on-screen / off-screen rendering to override defaults

        pause_subtask (bool): if True, pause after every subtask during generation, for
            debugging.

        bimanual (bool): if True, use bimanual robot configuration.

        enable_marker_vis (bool): if True, enable marker visualization.

        ds_ratio (int): downsampling ratio for trajectory.

        no_partial_tasks (bool): if True, don't save partial trajectories.

        headless (bool): if True, run in headless mode.

        baseline (str or None): baseline method to use (e.g., "mimicgen", "skillgen").

        print_stage_type (bool): if True, print generated stage_type transitions to the terminal.

        grasp_order (str): "random", "task_spec", or a comma-separated object order.

        video_fps (int): FPS used when writing generation debug videos under videos/.

        output_format (str): "hdf5" keeps the old demo.hdf5 flow, "lerobot"
            writes a LeRobot dataset directly, and "both" writes both formats.
    """

    # time this run
    script_start_time = time.time()

    # check some args
    if output_format not in DATASET_OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {DATASET_OUTPUT_FORMATS}, got {output_format}")
    write_hdf5 = output_format in ("hdf5", "both")
    write_lerobot = output_format in ("lerobot", "both")
    write_video = not no_save_video
    assert not (render and write_video) # either on-screen or video but not both
    if pause_subtask:
        assert render, "should enable on-screen rendering for pausing to be useful"
    grasp_order_quota_targets = _parse_grasp_order_quotas(grasp_order_quota)

    if write_video:
        if render_image_names is None and robot_type == "OpenArm" and generation_config.experiment.task.name.startswith(("openarm_real_exp_1", "openarm_drawer_storage", "openarm_fruit_basket_bagging")):
            # Use the viewer camera and task observation cameras for OpenArm videos.
            # large viewer camera on top, observation cameras on the bottom row.
            render_image_names = ["viewer_camera"] + list(generation_config.obs.camera_names)
        # debug video - use same cameras as observations by default
        elif len(generation_config.obs.camera_names) > 0 and render_image_names is None:
            render_image_names = list(generation_config.obs.camera_names)

    # path to source dataset
    source_dataset_path = os.path.expandvars(os.path.expanduser(generation_config.experiment.source.dataset_path))

    # get environment metadata from dataset
    env_meta = get_env_metadata_from_dataset(dataset_path=source_dataset_path)
    env_meta.setdefault("env_kwargs", {})["_source_dataset_path_for_pose"] = source_dataset_path
    if scene_file is not None:
        resolved_scene_file = os.path.abspath(os.path.expanduser(scene_file))
        if not os.path.isfile(resolved_scene_file):
            raise FileNotFoundError("ElogGen task scene file not found: {}".format(resolved_scene_file))
        env_meta.setdefault("env_kwargs", {}).setdefault("scene", {})["scene_file"] = resolved_scene_file
        print("Using ElogGen task scene: {}".format(resolved_scene_file))
    
    # set seed for generation
    random.seed(generation_config.experiment.seed)
    np.random.seed(generation_config.experiment.seed)
    th.manual_seed(generation_config.experiment.seed)

    # create new folder for this data generation run
    base_folder = os.path.expandvars(os.path.expanduser(generation_config.experiment.generation.path))
    new_dataset_folder_name = generation_config.experiment.name
    new_dataset_folder_path = os.path.join(
        base_folder,
        new_dataset_folder_name,
    )
    print("\nData will be generated at: {}".format(new_dataset_folder_path))

    # ensure dataset folder does not exist, and make new folder
    exist_ok = False
    if os.path.exists(new_dataset_folder_path):
        if not auto_remove_exp:
            # ans = input("\nWARNING: dataset folder ({}) already exists! \noverwrite? (y/n)\n".format(new_dataset_folder_path))
            ans = "n"
        else:
            ans = "y"
        if ans == "y":
            print("Removed old results folder at {}".format(new_dataset_folder_path))
            shutil.rmtree(new_dataset_folder_path)
        else:
            print("Keeping old dataset folder. Note that individual files may still be overwritten.")
            exist_ok = True
    os.makedirs(new_dataset_folder_path, exist_ok=exist_ok)

    # log terminal output to text file

    # save config to disk
    DatasetUtils.write_json(
        json_dic=generation_config,
        json_path=os.path.join(new_dataset_folder_path, "generation_config.json"),
    )

    print("\n============= Config =============")
    print(generation_config)
    print("")

    # some paths that we will create inside our new dataset folder

    # new dataset that will be generated
    new_dataset_path = os.path.join(new_dataset_folder_path, "demo.hdf5")

    # tmp folder that will contain per-episode hdf5s that were successful (they will be merged later)
    tmp_dataset_folder_path = os.path.join(new_dataset_folder_path, "tmp")
    if write_hdf5:
        os.makedirs(tmp_dataset_folder_path, exist_ok=exist_ok)

    context_report_folder_path = os.path.join(new_dataset_folder_path, "context_annotations")
    os.makedirs(context_report_folder_path, exist_ok=exist_ok)

    # folder containing logs
    json_log_path = os.path.join(new_dataset_folder_path, "logs")
    os.makedirs(json_log_path, exist_ok=exist_ok)

    if generation_config.experiment.generation.keep_failed:
        # new dataset for failed trajectories, and tmp folder for per-episode hdf5s that failed
        new_failed_dataset_path = os.path.join(new_dataset_folder_path, "demo_failed.hdf5")
        tmp_dataset_failed_folder_path = os.path.join(new_dataset_folder_path, "tmp_failed")
        if write_hdf5:
            os.makedirs(tmp_dataset_failed_folder_path, exist_ok=exist_ok)
        context_report_failed_folder_path = os.path.join(new_dataset_folder_path, "context_annotations_failed")
        os.makedirs(context_report_failed_folder_path, exist_ok=exist_ok)

    # get list of source demonstration keys from source hdf5
    all_demos = DatasetUtils.get_all_demos_from_dataset(
        dataset_path=source_dataset_path,
        filter_key=generation_config.experiment.source.filter_key,
        start=generation_config.experiment.source.start,
        n=generation_config.experiment.source.n,
    )

    # prepare args for creating simulation environment

    # auto-fill camera rendering info if not specified
    if (write_video or render or write_lerobot) and (render_image_names is None):
        render_image_names = Playback.get_default_env_cameras(env_meta=env_meta)
    if render:
        # on-screen rendering can only support one camera
        assert len(render_image_names) == 1

    obs_camera_names = list(generation_config.obs.camera_names)

    # Env cameras should include both observation cameras for HDF5 and any extra
    # debug-video cameras requested on the command line.
    camera_names = obs_camera_names
    if write_video:
        camera_names = list(dict.fromkeys(obs_camera_names + list(render_image_names or [])))

    # HDF5 image obs collection is controlled only by the obs config, not by
    # whether we also save debug videos.
    use_image_obs = generation_config.obs.collect_obs and (len(obs_camera_names) > 0)
    use_depth_obs = False

    lerobot_writer = None
    lerobot_output_path = None
    if write_lerobot:
        if not use_image_obs:
            raise ValueError("LeRobot direct output requires image observations in generation_config.obs.")
        from eloggen.generation_runtime.commands.export import (
            DEFAULT_CAMERA_MAP,
            LeRobotDatasetWriter,
            validate_lerobot_dataset,
        )

        configured_obs_keys = _configured_obs_key_candidates(obs_camera_names)
        missing_obs = [obs_key for obs_key in DEFAULT_CAMERA_MAP.values() if obs_key not in configured_obs_keys]
        if missing_obs:
            raise ValueError(
                "LeRobot direct output requires configured observation cameras: {}".format(
                    ", ".join(missing_obs)
                )
            )
        lerobot_output_path = os.path.abspath(
            os.path.expanduser(
                lerobot_output or _infer_lerobot_output_path(new_dataset_folder_path, lerobot_ordered_tasks)
            )
        )
        lerobot_writer = LeRobotDatasetWriter(
            output_path=lerobot_output_path,
            repo_id=lerobot_repo_id or _infer_lerobot_repo_id(generation_config.experiment.task.name, lerobot_ordered_tasks),
            task_name=lerobot_task or _base_task_name(generation_config.experiment.task.name),
            robot_type=lerobot_robot_type,
            fps=lerobot_fps or video_fps,
            camera_map=DEFAULT_CAMERA_MAP,
            ordered_tasks=lerobot_ordered_tasks,
            overwrite=lerobot_overwrite,
            resume=lerobot_resume,
        )
        print("\nLeRobot dataset will be written directly at: {}".format(lerobot_output_path))
        if lerobot_writer.resumed_episodes:
            print("Resuming LeRobot dataset from episode {:06d}".format(lerobot_writer.resumed_episodes))


    # simulation environment
    env = Playback.create_env(
        env_meta=env_meta,
        env_class=None,
        env_name=generation_config.experiment.task.name,
        robot=generation_config.experiment.task.robot,
        gripper=generation_config.experiment.task.gripper,
        camera_names=camera_names,
        camera_height=generation_config.obs.camera_height,
        camera_width=generation_config.obs.camera_width,
        render=render,
        render_offscreen=(write_video or write_lerobot),
        use_image_obs=use_image_obs,
        use_depth_obs=use_depth_obs,
        manipulation_only=False,
        real_robot_mode=False,
        baseline=baseline,
    )
    print("\n==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    task_name_for_generation = str(generation_config.experiment.task.name)
    if task_name_for_generation.startswith("openarm_drawer_storage"):
        env.drawer_initialization_distribution = json.loads(
            generation_config.task.initialization_distribution.dump()
        )
        env.drawer_initialization_seed = int(generation_config.experiment.seed)
        env.drawer_initialization_attempt_index = 0
    elif task_name_for_generation.startswith("openarm_fruit_basket_bagging"):
        env.fruit_initialization_distribution = json.loads(
            generation_config.task.initialization_distribution.dump()
        )
        env.fruit_initialization_seed = int(generation_config.experiment.seed)
        env.fruit_initialization_attempt_index = 0
    env.generation_attempt_index = 0

    # get information necessary to create env interface
    env_interface_name, env_interface_type = DatasetUtils.get_env_interface_info_from_dataset(
        dataset_path=source_dataset_path,
        demo_keys=all_demos,
    )
    # possibly override from config
    if generation_config.experiment.task.interface is not None:
        env_interface_name = generation_config.experiment.task.interface
    if generation_config.experiment.task.interface_type is not None:
        env_interface_type = generation_config.experiment.task.interface_type

    # create environment interface to use during data generation
    env_interface = make_interface(
        name=env_interface_name,
        interface_type=env_interface_type,
        # NOTE: env_interface takes underlying simulation environment, not robomimic wrapper
        env=env.base_env,
    )
    print("Created environment interface: {}".format(env_interface))

    # self.arm_command_start_idx {'left': 5, 'right': 12}
    # self.arm_command_end_idx {'left': 11, 'right': 18}

    # make sure we except the same exceptions that we would normally except during policy rollouts
    exceptions_to_except = env.rollout_exceptions

    # get task spec object from config
    task_spec_json_string = generation_config.task.task_spec.dump()
    if bimanual:
        task_spec = EG_TaskSpec.from_json_bimanual(json_string=task_spec_json_string)
    else:
        task_spec = EG_TaskSpec.from_json(json_string=task_spec_json_string)
    
    D2_sign = True if "D2" in generation_config.experiment.task.name else False
    subtask_graph_config = None
    subtask_scheduling_config = None
    node_to_phase = None
    if task_name_for_generation.startswith("openarm_drawer_storage"):
        from eloggen.generation_runtime.config.storage_task_graph import compile_storage_task_graph

        raw_task_spec = json.loads(generation_config.task.task_spec.dump())
        compiled_graph = compile_storage_task_graph(raw_task_spec)
        configured_graph = json.loads(generation_config.subtask_graph.dump())
        subtask_scheduling_config = json.loads(generation_config.subtask_scheduling.dump())
        graph_source = str(subtask_scheduling_config.get("graph_source", "auto"))
        if graph_source not in {"auto", "configured"}:
            raise ValueError(f"unknown drawer subtask graph_source: {graph_source!r}")
        if graph_source == "configured":
            if not configured_graph.get("nodes"):
                raise ValueError("graph_source='configured' requires subtask_graph.nodes")
            subtask_graph_config = configured_graph
        else:
            subtask_graph_config = compiled_graph.graph
        subtask_scheduling_config["seed"] = int(generation_config.experiment.seed)
        node_to_phase = compiled_graph.node_to_phase
    elif "subtask_graph" in generation_config and "subtask_scheduling" in generation_config:
        configured_graph = json.loads(generation_config.subtask_graph.dump())
        if configured_graph.get("nodes"):
            subtask_graph_config = configured_graph
            subtask_scheduling_config = json.loads(generation_config.subtask_scheduling.dump())
            subtask_scheduling_config["seed"] = int(generation_config.experiment.seed)
            node_to_phase = {
                str(phase.get("node_id")): index
                for index, (_, phase) in enumerate(
                    sorted(
                        json.loads(generation_config.task.task_spec.dump()).items(),
                        key=lambda item: int(str(item[0]).rsplit("_", 1)[-1]),
                    )
                )
                if phase.get("node_id")
            }
    # make data generator object
    data_generator = DataGenerator(
        task_spec=task_spec,
        dataset_path=source_dataset_path,
        demo_keys=all_demos,
        bimanual=bimanual,
        D2_sign=D2_sign,
        print_stage_type=print_stage_type,
        grasp_order=grasp_order,
        subtask_graph=subtask_graph_config,
        subtask_scheduling=subtask_scheduling_config,
        node_to_phase=node_to_phase,
        subtask_order=subtask_order,
    )

    if write_video:
        os.makedirs(f"{new_dataset_folder_path}/videos", exist_ok=True) 

    print("\n==== Created Data Generator ====")
    print(data_generator)
    print("")

    existing_log_jsons = os.listdir(json_log_path)
    if len(existing_log_jsons) > 0:
        # find the last json file
        existing_log_jsons.sort()
        last_json = existing_log_jsons[-1]
        last_json_path = os.path.join(json_log_path, last_json)
        with open(last_json_path, "r") as f:
            last_json_dict = json.load(f)
        num_attempts = last_json_dict["num_attempts"]
        num_success = last_json_dict["num_success"]
        num_failures = last_json_dict["num_failures"]
        num_problematic = last_json_dict["num_problematic"]
        # backward compatibility
        ep_lengths = last_json_dict.get("ep_lengths", [])
        all_episode_logs = last_json_dict["all_episode_logs"]
    else:
        # data generation statistics
        num_attempts = 0
        num_success = 0
        num_failures = 0
        num_problematic = 0
        ep_lengths = [] # episode lengths for successfully generated data
        all_episode_logs = {
            "episode_number": [],
            "err_status": [],
            "time_taken": [],
            "task_success": [],
            "phases_completed": [],
            "phase_logs": [],
            "initialization_samples": [],
        }
    all_episode_logs.setdefault(
        "initialization_samples",
        [None] * len(all_episode_logs.get("episode_number", [])),
    )
    if task_name_for_generation.startswith("openarm_drawer_storage"):
        env.drawer_initialization_attempt_index = num_attempts + num_problematic
    elif task_name_for_generation.startswith("openarm_fruit_basket_bagging"):
        env.fruit_initialization_attempt_index = num_attempts + num_problematic

    if write_lerobot and lerobot_resume and lerobot_writer is not None and lerobot_writer.resumed_episodes:
        resumed_success = int(lerobot_writer.resumed_episodes)
        if num_success != resumed_success:
            print(
                "Aligning generation success count to resumed LeRobot episodes: "
                "{} -> {}".format(num_success, resumed_success)
            )
            num_success = resumed_success
            num_attempts = max(num_attempts, num_success)
        if len(ep_lengths) < resumed_success:
            resumed_lengths = [
                int(episode["length"])
                for episode in lerobot_writer.episodes_meta[len(ep_lengths):resumed_success]
            ]
            ep_lengths.extend(resumed_lengths)

    grasp_order_quota_counts = {}
    grasp_order_quota_total_target = None
    if grasp_order_quota_targets:
        if not generation_config.experiment.task.name.startswith("openarm_real_exp_1"):
            raise ValueError("--grasp-order-quota currently supports openarm_real_exp_1 two-order generation")
        for order_name, target_count in grasp_order_quota_targets.items():
            if target_count < 0:
                raise ValueError(f"Negative quota for {order_name}: {target_count}")
            grasp_order_quota_counts[order_name] = 0
        if lerobot_writer is not None and lerobot_writer.task_episode_counts:
            for order_name in grasp_order_quota_counts:
                grasp_order_quota_counts[order_name] = int(
                    lerobot_writer.task_episode_counts.get(order_name, 0)
                )
        grasp_order_quota_total_target = int(sum(grasp_order_quota_targets.values()))
        print(
            "Using grasp-order quotas: targets={}, existing_counts={}".format(
                grasp_order_quota_targets,
                grasp_order_quota_counts,
            )
        )

    def grasp_order_quota_complete():
        if not grasp_order_quota_targets:
            return False
        return all(
            grasp_order_quota_counts.get(order_name, 0) >= target_count
            for order_name, target_count in grasp_order_quota_targets.items()
        )

    def select_next_grasp_order_for_quota():
        remaining = {
            order_name: target_count - grasp_order_quota_counts.get(order_name, 0)
            for order_name, target_count in grasp_order_quota_targets.items()
            if target_count - grasp_order_quota_counts.get(order_name, 0) > 0
        }
        if not remaining:
            return None
        max_remaining = max(remaining.values())
        candidates = sorted(order_name for order_name, value in remaining.items() if value == max_remaining)
        order_name = str(np.random.choice(candidates))
        return order_name, OPENARM_REAL_EXP_1_GRASP_ORDER_BY_NAME[order_name]


    lerobot_info = None

    def finalize_lerobot_checkpoint(reason=None, validate=False):
        nonlocal lerobot_info
        if not write_lerobot or lerobot_writer is None:
            return None
        if reason:
            print(reason)
        lerobot_info = lerobot_writer.finalize()
        print(
            "Checkpointed LeRobot metadata at {} (episodes={})".format(
                lerobot_output_path,
                lerobot_info["total_episodes"],
            )
        )
        if validate:
            validate_lerobot_dataset(
                output_path=lerobot_output_path,
                expected_episodes=lerobot_info["total_episodes"],
                expected_cameras=len(lerobot_writer.camera_map),
            )
        return lerobot_info

    if lerobot_checkpoint_every is None:
        lerobot_checkpoint_every = 10
    lerobot_checkpoint_every = int(lerobot_checkpoint_every)
    if lerobot_checkpoint_every < 0:
        raise ValueError("--lerobot-checkpoint-every must be >= 0")


    # we will keep generating data until @num_trials successes (if @guarantee_success) else @num_trials attempts
    num_trials = generation_config.experiment.generation.num_trials
    if grasp_order_quota_total_target is not None:
        num_trials = max(num_trials, grasp_order_quota_total_target)
    guarantee_success = generation_config.experiment.generation.guarantee
    
    base_mp_failures, arm_mp_ik_failures, arm_mp_trajopt_failures, arm_mp_other_failures, base_sampling_failures, base_mp_ik_failures = 0, 0, 0, 0, 0, 0
    obj_visible_at_start_of_manip = 0

    video_writer = None
    video_path = None

    try:
        while True:
            if grasp_order_quota_complete():
                break

            selected_quota_order = None
            if grasp_order_quota_targets:
                selected_quota_order = select_next_grasp_order_for_quota()
                if selected_quota_order is None:
                    break
                order_name, order_setting = selected_quota_order
                data_generator.grasp_order_setting = order_setting
                print(
                    "Quota-selected grasp_order={} ({}) counts={}/{}".format(
                        order_setting,
                        order_name,
                        grasp_order_quota_counts,
                        grasp_order_quota_targets,
                    )
                )

            print(f"======================= ATTEMPT {num_attempts} ========================")
            env.generation_attempt_index = num_attempts + num_problematic
    
            # we might write a video to show the data generation attempts
            video_writer = None
            video_path = None
            if write_video:
                video_path = f"{new_dataset_folder_path}/videos/attempt_{num_attempts:04d}.mp4"
                video_writer = CameraVideoWriter(video_path, camera_names=render_image_names, fps=video_fps)
    
            # generate trajectory
            try:
                episode_start_time = time.time()
                generated_traj = data_generator.generate(
                    env=env,
                    env_interface=env_interface,
                    select_src_per_subtask=generation_config.experiment.generation.select_src_per_subtask,
                    transform_first_robot_pose=generation_config.experiment.generation.transform_first_robot_pose,
                    interpolate_from_last_target_pose=generation_config.experiment.generation.interpolate_from_last_target_pose,
                    render=render,
                    video_writer=video_writer,
                    video_skip=video_skip,
                    camera_names=render_image_names,
                    pause_subtask=pause_subtask,
                    enable_marker_vis=enable_marker_vis,
                    ds_ratio=ds_ratio,
                    grasp_init_views_video_writer=None,
                    no_partial_tasks=no_partial_tasks,
                    baseline=baseline,
                )
                episode_time_taken = time.time() - episode_start_time
                print("==============================")
                print("Time taken for generation: {:.2f} seconds".format(episode_time_taken))
                print("==============================")
    
                # save episode logs
                all_episode_logs["episode_number"].append(num_attempts+num_problematic)
                all_episode_logs["err_status"].append(env.err)
                all_episode_logs["time_taken"].append(episode_time_taken)
                all_episode_logs["task_success"].append(env.is_success()["task"])
                if generated_traj is not None:
                    all_episode_logs["phases_completed"].append(generated_traj["phases_completed"])
                    all_episode_logs["phase_logs"].append(generated_traj["phase_logs"])
                else:
                    all_episode_logs["phases_completed"].append(-1)
                    all_episode_logs["phase_logs"].append(dict())
                all_episode_logs["initialization_samples"].append(
                    deepcopy(
                        getattr(env, "drawer_initialization_sample", None)
                        or getattr(env, "fruit_initialization_sample", None)
                    )
                )

            except exceptions_to_except as e:
                if write_video and video_writer is not None:
                    video_writer.close()
                    if video_path is not None and os.path.exists(video_path):
                        os.remove(video_path)
                # problematic trajectory - do not have this count towards our total number of attempts, and re-try
                print("")
                print("*" * 50)
                print("WARNING: got rollout exception {}".format(e))
                print("*" * 50)
                print("")
                
                episode_time_taken = time.time() - episode_start_time
                # save episode logs
                all_episode_logs["episode_number"].append(num_attempts+num_problematic)
                all_episode_logs["err_status"].append("problematic")
                all_episode_logs["time_taken"].append(episode_time_taken)
                all_episode_logs["task_success"].append(False)
                all_episode_logs["phases_completed"].append(-1)
                all_episode_logs["phase_logs"].append(dict())
                all_episode_logs["initialization_samples"].append(
                    deepcopy(
                        getattr(env, "drawer_initialization_sample", None)
                        or getattr(env, "fruit_initialization_sample", None)
                    )
                )
                
                num_problematic += 1
                continue
            
            num_attempts += 1
            if write_video:
                video_writer.close()
            
            if env.err == "BaseMPFailed":
                base_mp_failures += 1
            elif env.err == "BaseMPIKFailed":
                base_mp_ik_failures += 1
            elif env.err == "ArmMPTrajOptFailed":
                arm_mp_trajopt_failures += 1
            elif env.err == "ArmMPIKFailed":
                arm_mp_ik_failures += 1
            elif env.err == "BaseSamplingFailed":   
                base_sampling_failures += 1
            elif env.err == "ArmMPOtherFailed":
                arm_mp_other_failures += 1
            
            if env.obj_visible_at_start_of_manip:
                obj_visible_at_start_of_manip += 1
    
            # generated_traj will be None if a) the 0th phase of the trajectory failed due to MP or b) no_partial_tasks is True meaning that any MP failure in any phase
            # is considered a failure and is not saved in either the success or failure hdf5 file.
            invalid_traj = generated_traj is None or len(generated_traj["states"]) == 0
            if invalid_traj:
                success = False
                num_failures += 1
            else:
                success = env.is_success()["task"]
                if success:
                    num_success += 1
                else:
                    num_failures += 1
    
            print("")
            print("*" * 50)
            print("trial {} success: {}".format(num_attempts, success))
            print("have {} successes out of {} trials so far".format(num_success, num_attempts))
            print("have {} failures out of {} trials so far".format(num_failures, num_attempts))
            print('have {} Base MP failures, {} Arm MP IK failures, {} Arm MP TrajOpt failures, {} Arm MP other failures, {} Base sampling failures, {} Base MP IK failures'.format(base_mp_failures, arm_mp_ik_failures, arm_mp_trajopt_failures, arm_mp_other_failures, base_sampling_failures, base_mp_ik_failures))
            print("*" * 50)
    
            if success:
                if grasp_order_quota_targets:
                    actual_order_name = _grasp_order_name_from_metadata(generated_traj.get("episode_metadata"))
                    if actual_order_name not in grasp_order_quota_targets:
                        raise ValueError(
                            "Generated episode has grasp_order={!r}, which is not in quota targets {}".format(
                                actual_order_name,
                                grasp_order_quota_targets,
                            )
                        )
                    if grasp_order_quota_counts.get(actual_order_name, 0) >= grasp_order_quota_targets[actual_order_name]:
                        raise RuntimeError(
                            "Generated extra episode for full grasp_order quota {}: counts={}, targets={}".format(
                                actual_order_name,
                                grasp_order_quota_counts,
                                grasp_order_quota_targets,
                            )
                        )
                    grasp_order_quota_counts[actual_order_name] = grasp_order_quota_counts.get(actual_order_name, 0) + 1
                    print(
                        "Updated grasp-order quota counts: {}/{}".format(
                            grasp_order_quota_counts,
                            grasp_order_quota_targets,
                        )
                    )
                if write_video and video_path is not None and os.path.exists(video_path):
                    demo_video_path = os.path.join(new_dataset_folder_path, "videos", f"demo_{num_success - 1:06d}.mp4")
                    os.replace(video_path, demo_video_path)
                # store successful demonstration
                ep_lengths.append(generated_traj["actions"].shape[0])
                if write_hdf5:
                    DatasetUtils.write_demo_to_hdf5(
                        folder=tmp_dataset_folder_path,
                        env=env,
                        initial_state=generated_traj["initial_state"],
                        states=generated_traj["states"],
                        observations=(generated_traj["observations"] if generation_config.obs.collect_obs else None),
                        observations_info=generated_traj["observations_info"],
                        datagen_info=generated_traj["datagen_infos"],
                        actions=generated_traj["actions"],
                        src_demo_inds=generated_traj["src_demo_inds"],
                        src_demo_labels=generated_traj["src_demo_labels"],
                        mp_end_steps=generated_traj["mp_end_steps"],
                        subtask_lengths=generated_traj["subtask_lengths"],
                        sensor_info=generated_traj["sensor_info"],
                        episode_time_taken=episode_time_taken,
                        partial=generated_traj["partial"],
                        left_mp_ranges=generated_traj["left_mp_ranges"],
                        right_mp_ranges=generated_traj["right_mp_ranges"],
                        frame_contexts=generated_traj.get("frame_contexts"),
                        episode_metadata=generated_traj.get("episode_metadata"),
                    )
                if write_lerobot:
                    lerobot_writer.write_generated_episode(generated_traj)
                    if lerobot_checkpoint_every > 0 and (len(lerobot_writer.episodes_meta) % lerobot_checkpoint_every) == 0:
                        finalize_lerobot_checkpoint()
                if generated_traj.get("frame_contexts") is not None:
                    write_frame_context_report(
                        output_path=os.path.join(context_report_folder_path, "demo_{}.json".format(str(num_success - 1).zfill(6))),
                        frame_contexts=generated_traj["frame_contexts"],
                        metadata={
                            "demo_key": "demo_{}".format(num_success - 1),
                            "episode_number": num_attempts,
                            "success": True,
                            "partial": generated_traj["partial"],
                            "num_frames": int(generated_traj["actions"].shape[0]),
                            "src_demo_inds": generated_traj.get("src_demo_inds"),
                            "mp_end_steps": generated_traj.get("mp_end_steps"),
                            "subtask_lengths": generated_traj.get("subtask_lengths"),
                            "episode_metadata": generated_traj.get("episode_metadata"),
                        },
                    )
            else:
                keep_failed = generation_config.experiment.generation.keep_failed
                less_than_max_failures = (generation_config.experiment.max_num_failures is None) or (num_failures <= generation_config.experiment.max_num_failures)
                if write_video and video_path is not None and os.path.exists(video_path):
                    if keep_failed and less_than_max_failures and not invalid_traj:
                        failed_video_path = os.path.join(new_dataset_folder_path, "videos", f"failed_demo_{num_failures - 1:06d}.mp4")
                        os.replace(video_path, failed_video_path)
                    else:
                        os.remove(video_path)
                # check if this failure should be kept
                if keep_failed and less_than_max_failures and not invalid_traj and write_hdf5:
                    # save failed trajectory in separate folder
                    DatasetUtils.write_demo_to_hdf5(
                        folder=tmp_dataset_failed_folder_path,
                        env=env,
                        initial_state=generated_traj["initial_state"],
                        states=generated_traj["states"],
                        observations=(generated_traj["observations"] if generation_config.obs.collect_obs else None),
                        observations_info=generated_traj["observations_info"],
                        datagen_info=generated_traj["datagen_infos"],
                        actions=generated_traj["actions"],
                        src_demo_inds=generated_traj["src_demo_inds"],
                        src_demo_labels=generated_traj["src_demo_labels"],
                        mp_end_steps=generated_traj["mp_end_steps"],
                        subtask_lengths=generated_traj["subtask_lengths"],
                        sensor_info=generated_traj["sensor_info"],
                        episode_time_taken=episode_time_taken,
                        partial=generated_traj["partial"],
                        left_mp_ranges=generated_traj["left_mp_ranges"],
                        right_mp_ranges=generated_traj["right_mp_ranges"],
                        frame_contexts=generated_traj.get("frame_contexts"),
                        episode_metadata=generated_traj.get("episode_metadata"),
                    )
                    if generated_traj.get("frame_contexts") is not None:
                        write_frame_context_report(
                            output_path=os.path.join(context_report_failed_folder_path, "demo_{}.json".format(str(num_failures - 1).zfill(6))),
                            frame_contexts=generated_traj["frame_contexts"],
                            metadata={
                                "demo_key": "demo_{}".format(num_failures - 1),
                                "episode_number": num_attempts,
                                "success": False,
                                "partial": generated_traj["partial"],
                                "num_frames": int(generated_traj["actions"].shape[0]),
                                "src_demo_inds": generated_traj.get("src_demo_inds"),
                                "mp_end_steps": generated_traj.get("mp_end_steps"),
                                "subtask_lengths": generated_traj.get("subtask_lengths"),
                                "episode_metadata": generated_traj.get("episode_metadata"),
                            },
                        )
    
            # regularly log progress to disk every so often
            if (num_attempts % generation_config.experiment.log_every_n_attempts) == 0:
                # get summary stats
                summary_stats = get_important_stats(
                    new_dataset_folder_path=new_dataset_folder_path,
                    num_success=num_success,
                    num_failures=num_failures,
                    num_attempts=num_attempts,
                    num_problematic=num_problematic,
                    ep_lengths=ep_lengths,
                    start_time=script_start_time,
                    ep_length_stats=None,
                    all_episode_logs=all_episode_logs,
                )
    
                # write stats to disk
                max_digits = len(str(num_trials * 1000)) + 1 # assume we will never have lower than 0.1% data generation SR
                json_file_path = os.path.join(json_log_path, "attempt_{}_succ_{}_rate_{}.json".format(
                    str(num_attempts).zfill(max_digits), # pad with leading zeros for ordered list of jsons in directory
                    num_success,
                    np.round((100. * num_success) / num_attempts, 2),
                ))
                DatasetUtils.write_json(json_dic=summary_stats, json_path=json_file_path)
    
            # termination condition is on enough successes if @guarantee_success or enough attempts otherwise
            if grasp_order_quota_targets:
                if grasp_order_quota_complete():
                    break
            else:
                check_val = num_success if guarantee_success else num_attempts
                if check_val >= num_trials:
                    break
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Finalizing generated outputs before exiting...")
        if write_video and video_writer is not None:
            try:
                video_writer.close()
            except Exception:
                pass
            if video_path is not None and os.path.exists(video_path):
                os.remove(video_path)
        if write_lerobot:
            finalize_lerobot_checkpoint(
                reason="Finalizing LeRobot metadata for completed episodes after Ctrl-C...",
                validate=False,
            )
        with open(os.path.join(new_dataset_folder_path, "episode_logs.json"), "w") as f:
            json.dump(all_episode_logs, f, indent=4)
        if env_meta["type"] == EnvUtils.EB.EnvType.OG_TYPE:
            og.shutdown()
        raise


    # save episode logs
    with open(os.path.join(new_dataset_folder_path, "episode_logs.json"), "w") as f:
        json.dump(all_episode_logs, f, indent=4)
    
    if write_hdf5:
        # merge all new created files
        print("\nFinished data generation. Merging per-episode hdf5s together...\n")
        DatasetUtils.merge_all_hdf5(
            folder=tmp_dataset_folder_path,
            new_hdf5_path=new_dataset_path,
            delete_folder=True,
        )
        if generation_config.experiment.generation.keep_failed:
            DatasetUtils.merge_all_hdf5(
                folder=tmp_dataset_failed_folder_path,
                new_hdf5_path=new_failed_dataset_path,
                delete_folder=True,
            )
    else:
        print("\nFinished data generation. Skipping HDF5 merge because output_format=lerobot.\n")

    if write_lerobot:
        lerobot_info = finalize_lerobot_checkpoint(validate=True)
        print("Finalized LeRobot dataset at: {}".format(lerobot_output_path))

    # get episode length statistics
    ep_length_stats = None
    if len(ep_lengths) > 0:
        ep_length_mean = float(np.mean(ep_lengths))
        ep_length_std = float(np.std(ep_lengths))
        ep_length_max = int(np.max(ep_lengths))
        ep_length_3std = int(np.ceil(ep_length_mean + 3. * ep_length_std))
        ep_length_stats = dict(
            ep_length_mean=ep_length_mean,
            ep_length_std=ep_length_std,
            ep_length_max=ep_length_max,
            ep_length_3std=ep_length_3std,
        )

    stats = get_important_stats(
        new_dataset_folder_path=new_dataset_folder_path,
        num_success=num_success,
        num_failures=num_failures,
        num_attempts=num_attempts,
        num_problematic=num_problematic,
        ep_lengths=ep_lengths,
        start_time=script_start_time,
        ep_length_stats=ep_length_stats,
    )
    stats["output_format"] = output_format
    if write_lerobot:
        stats["lerobot_output_path"] = lerobot_output_path
        stats["lerobot_total_episodes"] = lerobot_info["total_episodes"]
        stats["lerobot_total_frames"] = lerobot_info["total_frames"]
    print("\nStats Summary")
    print(json.dumps(stats, indent=4))

    # maybe render videos
    if generation_config.experiment.render_video and write_hdf5:
        if (num_success > 0):
            playback_video_path = os.path.join(new_dataset_folder_path, "playback_{}.mp4".format(new_dataset_folder_name))
            num_render = generation_config.experiment.num_demo_to_render
            print("Rendering successful trajectories...")
            Playback.make_dataset_video(
                dataset_path=new_dataset_path,
                video_path=playback_video_path,
                num_render=num_render,
            )
        else:
            print("\n" + "*" * 80)
            print("\nWARNING: skipping dataset video creation since no successes")
            print("\n" + "*" * 80 + "\n")
        if generation_config.experiment.generation.keep_failed:
            if (num_failures > 0):
                playback_video_path = os.path.join(new_dataset_folder_path, "playback_{}_failed.mp4".format(new_dataset_folder_name))
                num_render = generation_config.experiment.num_fail_demo_to_render
                print("Rendering failure trajectories...")
                Playback.make_dataset_video(
                    dataset_path=new_failed_dataset_path,
                    video_path=playback_video_path,
                    num_render=num_render,
                )
            else:
                print("\n" + "*" * 80)
                print("\nWARNING: skipping dataset video creation since no failures")
                print("\n" + "*" * 80 + "\n")
    elif generation_config.experiment.render_video and not write_hdf5:
        print("\nWARNING: skipping robomimic playback video creation because output_format=lerobot")

    # return some summary info
    final_important_stats = get_important_stats(
        new_dataset_folder_path=new_dataset_folder_path,
        num_success=num_success,
        num_failures=num_failures,
        num_attempts=num_attempts,
        num_problematic=num_problematic,
        ep_lengths=ep_lengths,
        start_time=script_start_time,
        ep_length_stats=ep_length_stats,
        all_episode_logs=all_episode_logs,
    )
    final_important_stats["output_format"] = output_format
    if write_lerobot:
        final_important_stats["lerobot_output_path"] = lerobot_output_path
        final_important_stats["lerobot_total_episodes"] = lerobot_info["total_episodes"]
        final_important_stats["lerobot_total_frames"] = lerobot_info["total_frames"]

    # write stats to disk
    json_file_path = os.path.join(new_dataset_folder_path, "important_stats.json")
    DatasetUtils.write_json(json_dic=final_important_stats, json_path=json_file_path)


    if env_meta["type"] == EnvUtils.EB.EnvType.OG_TYPE:
        og.shutdown()

    return final_important_stats


def main(args):

    # load config object
    with open(args.config, "r") as f:
        ext_cfg = json.load(f)
        # config generator from robomimic generates this part of config unused by ElogGen generation runtime
        if "meta" in ext_cfg:
            del ext_cfg["meta"]
    generation_config = config_factory(ext_cfg["name"], config_type=ext_cfg["type"])

    # update config with external json - this will throw errors if
    # the external config has keys not present in the base config
    with generation_config.values_unlocked():
        generation_config.update(ext_cfg)

        # We assume that the external config specifies all subtasks, so
        # delete any subtasks not in the external config.
        source_subtasks = set(generation_config.task.task_spec.keys())
        new_subtasks = set(ext_cfg["task"]["task_spec"].keys())
        for subtask in (source_subtasks - new_subtasks):
            print("deleting subtask {} in original config".format(subtask))
            del generation_config.task.task_spec[subtask]

        # maybe override some settings
        if args.task_name is not None:
            generation_config.experiment.task.name = args.task_name

        if args.source is not None:
            generation_config.experiment.source.dataset_path = args.source

        if args.folder is not None:
            generation_config.experiment.generation.path = args.folder

        if args.num_demos is not None:
            generation_config.experiment.generation.num_trials = args.num_demos

        if args.seed is not None:
            generation_config.experiment.seed = args.seed

        # maybe modify config for debugging purposes
        if args.debug:
            # shrink length of generation to test whether this run is likely to crash
            generation_config.experiment.source.n = 3
            generation_config.experiment.generation.guarantee = False
            generation_config.experiment.generation.num_trials = 2

            # send output to a temporary directory
            generation_config.experiment.generation.path = "/tmp/tmpgeneration_runtime"

    res_str = "finished run successfully!"
    important_stats = None
    try:
        important_stats = generate_dataset(
            generation_config=generation_config,
            auto_remove_exp=args.auto_remove_exp,
            render=args.render,
            no_save_video=args.no_video_save,
            video_skip=args.video_skip,
            render_image_names=args.render_image_names,
            pause_subtask=args.pause_subtask,
            bimanual=args.bimanual,
            enable_marker_vis=args.enable_marker_vis,
            ds_ratio=args.ds_ratio,
            no_partial_tasks=args.no_partial_tasks,
            headless=args.headless,
            baseline=args.baseline,
            robot_type=args.robot_type,
            print_stage_type=args.print_stage_type,
            grasp_order=args.grasp_order,
            subtask_order=args.subtask_order,
            video_fps=args.video_fps,
            output_format=args.output_format,
            lerobot_output=args.lerobot_output,
            lerobot_repo_id=args.lerobot_repo_id,
            lerobot_task=args.lerobot_task,
            lerobot_robot_type=args.lerobot_robot_type,
            lerobot_fps=args.lerobot_fps,
            lerobot_ordered_tasks=args.lerobot_ordered_tasks,
            lerobot_overwrite=args.lerobot_overwrite,
            lerobot_resume=args.lerobot_resume,
            lerobot_checkpoint_every=args.lerobot_checkpoint_every,
            grasp_order_quota=args.grasp_order_quota,
        )
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    except KeyboardInterrupt:
        print("run interrupted by Ctrl-C after finalizing completed outputs.")
        sys.exit(130)
    print(res_str)
    if important_stats is not None:
        important_stats = json.dumps(important_stats, indent=4)
        print("\nFinal Data Generation Stats")
        print(important_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to ElogGen generation runtime config json",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick generation run for debugging purposes",
    )
    parser.add_argument(
        "--auto-remove-exp",
        action='store_true',
        help="force delete the experiment folder if it exists"
    )
    parser.add_argument(
        "--bimanual",
        action='store_true',
        help="force the code to use bimanual setup"
    )
    parser.add_argument(
        "--render",
        action='store_true',
        help="render each data generation attempt on-screen",
    )
    parser.add_argument(
        "--no_video_save",
        action='store_true',
        help="if provided, don't save video of data generation attempts",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="skip every nth frame when writing video",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=DEFAULT_GENERATION_VIDEO_FPS,
        help="FPS for generation debug videos saved under videos/.",
    )
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )
    parser.add_argument(
        "--pause_subtask",
        action='store_true',
        help="pause after every subtask during generation for debugging - only useful with render flag",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="path to source dataset, to override the one in the config",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="environment name to use for data generation, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="folder that will be created with new data, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        help="number of demos to generate, or attempt to generate, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--headless",
        action='store_true',
        help="whether to generate data in headless mode",
    )
    parser.add_argument(
        "--enable_marker_vis",
        action='store_true',
        help="enable the marker visualization when generating data, the markers are mainly for vis the eef pose and target pose",
    )
    parser.add_argument(
        "--ds_ratio",
        type=int,
        help="downsample rate for the replay data",
        default=None,
    )
    parser.add_argument(
        "--no_partial_tasks",
        action='store_true',
        help="disable the marker visualization when generating data, the markers are mainly for vis the eef pose and target pose",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="baseline to run. Options: mimicgen or skillgen",
        default=None,
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        help="robot type to use for data generation",
        default="OpenArm",
    )
    parser.add_argument(
        "--print_stage_type",
        action="store_true",
        help="Print generated stage_type transitions when ELOGGEN_VERBOSE_DATAGEN=1.",
    )
    parser.add_argument(
        "--grasp-order",
        "--grasp_order",
        dest="grasp_order",
        type=str,
        default="random",
        help=(
            "OpenArm object execution order. Use 'random' (default), "
            "'task_spec' for config phase order, or a comma-separated object order, e.g. "
            "'canned_food_1,candy_1,orange_1,apple_1' or 'object_1,object_2'. "
            "Short aliases can/canned/candy/orange/apple/obj1/obj2/lime are accepted."
        ),
    )
    parser.add_argument(
        "--grasp-order-quota",
        "--grasp_order_quota",
        dest="grasp_order_quota",
        type=str,
        default=None,
        help=(
            "Force fixed final counts for openarm_real_exp_1 grasp orders. "
            "Use COUNT for both orders, e.g. '50', or explicit quotas such as "
            "'apple_first=50,lemon_first=50'. When used with --lerobot-resume, "
            "existing task_episode_counts are counted toward the quota."
        ),
    )
    parser.add_argument(
        "--subtask-order",
        "--subtask_order",
        dest="subtask_order",
        type=str,
        default=None,
        help=(
            "Drawer storage full subtask order. Use 'random' or a comma-separated list of "
            "graph node ids, for example "
            "'pick_object_1,open_drawer,place_object_1,pick_object_2,place_object_2'."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=DATASET_OUTPUT_FORMATS,
        default="hdf5",
        help="Final dataset format: hdf5 keeps old behavior, lerobot writes LeRobot directly, both writes both.",
    )
    parser.add_argument(
        "--lerobot-output",
        type=str,
        default=None,
        help=(
            "Output directory for direct LeRobot generation. Defaults under "
            f"{DEFAULT_LEROBOT_DATASET_ROOT}/... inferred from --folder/config."
        ),
    )
    parser.add_argument(
        "--lerobot-repo-id",
        type=str,
        default=None,
        help="Repo id stored in LeRobot meta/info.json, e.g. openarm/openarm_real_exp_1.",
    )
    parser.add_argument(
        "--lerobot-task",
        type=str,
        default=None,
        help="Single task prompt stored in tasks.jsonl when --lerobot-ordered-tasks is not used.",
    )
    parser.add_argument(
        "--lerobot-robot-type",
        type=str,
        default="openarm",
        help="Robot type stored in LeRobot meta/info.json.",
    )
    parser.add_argument(
        "--lerobot-fps",
        type=int,
        default=None,
        help="FPS for LeRobot timestamps and MP4 export. Defaults to --video_fps.",
    )
    parser.add_argument(
        "--lerobot-ordered-tasks",
        action="store_true",
        help="Use semantic grasp_order to assign ordered multi-task prompts.",
    )
    parser.add_argument(
        "--lerobot-overwrite",
        action="store_true",
        help="Remove the direct LeRobot output directory before writing.",
    )
    parser.add_argument(
        "--lerobot-resume",
        action="store_true",
        help=(
            "Resume direct LeRobot generation from an existing output directory. "
            "If finalize metadata is missing, rebuild it from complete parquet/video/semantic_context episodes."
        ),
    )
    parser.add_argument(
        "--lerobot-checkpoint-every",
        type=int,
        default=10,
        help=(
            "Checkpoint LeRobot metadata every N successful episodes during direct generation. "
            "Use 1 for every episode, 0 to write metadata only on Ctrl-C/final exit. Defaults to 10."
        ),
    )

    args = parser.parse_args()
    main(args)
