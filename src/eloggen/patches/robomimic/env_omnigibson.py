"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import cv2
import time
import json
import os
import numpy as np
from copy import deepcopy

import omnigibson as og
import omnigibson.lazy as lazy
import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB

import omnigibson.utils.transform_utils as T
from omnigibson import object_states
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
from omnigibson.controllers import ControlType
from omnigibson.systems.system_base import BaseSystem
from omnigibson.robots.r1 import R1
from omnigibson.robots.tiago import Tiago
from omnigibson.robots import OpenArmBimanual
from omnigibson.utils.constants import LightingMode

from eloggen.generation_runtime.simulation.omnigibson import TASK_CONFIGS
from eloggen.generation_runtime.simulation.drawers import OPENARM_DRAWER_TARGETS
from eloggen.generation_runtime.simulation.randomization.openarm_drawer_storage import (
    difficulty_from_mapping,
    sample_drawer_initialization,
    validate_aabb_support,
    aabb_xy_distance_to_translated_sweep,
)
from eloggen.generation_runtime.simulation.randomization.openarm_fruit_basket_bagging import (
    difficulty_from_mapping as fruit_difficulty_from_mapping,
    sample_fruit_bagging_initialization,
)
from eloggen.generation_runtime.simulation.camera import (
    is_openarm_real_exp_1,
    set_real_exp_1_viewer_camera,
    uses_real_exp_1_viewer_camera,
)
from eloggen.generation_runtime.simulation.camera_config import (
    PUBLISHED_OPENARM_TASKS,
    canonical_openarm_task_name,
    get_real_exp_1_external_sensor_kwargs,
    openarm_camera_mount_file_candidates,
    square_camera_sensor_kwargs,
)

# from mimicgen.train_scripts.train_prep_data import compute_point_cloud_from_rgbd
from scipy.spatial.transform import Rotation as R
import fpsample
import open3d as o3d

from enum import Enum
import torch as th
import numpy as np
import gym
import time
import copy


from omnigibson.macros import gm, macros as og_macros

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False
DEFAULT_FORCE_LIGHT_INTENSITY = gm.FORCE_LIGHT_INTENSITY

DEBUG = False
OPENARM_CASHIER_STYLE_PREFIXES = (
    "openarm_cashier",
    "openarm_bag_groceries",
    "openarm_real_exp_1",
    "openarm_drawer_storage",
    "openarm_fruit_basket_bagging",
)
OPENARM_TASK_PREFIXES = (
    "openarm_pick_cup",
    *OPENARM_CASHIER_STYLE_PREFIXES,
)
OPENARM_GRASPING_MODES = ("sticky", "assisted", "physical")


def _openarm_grasping_mode(default="sticky"):
    mode = os.environ.get("OPENARM_GRASPING_MODE", default).strip().lower()
    if mode not in OPENARM_GRASPING_MODES:
        raise ValueError(
            "OPENARM_GRASPING_MODE must be one of {}, got {!r}".format(
                OPENARM_GRASPING_MODES,
                mode,
            )
        )
    return mode


def _openarm_task_grasping_mode(task_name):
    return _openarm_grasping_mode(default="sticky")


def _openarm_grasp_window(default=0.0):
    return float(os.environ.get("OPENARM_GRASP_WINDOW", default))


def _optional_float_env(name):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


def _openarm_gripper_joint_controller_config(task_name=None):
    cfg = {
        "name": "JointController",
        "motor_type": "position",
        "command_input_limits": None,
        "command_output_limits": None,
        "use_impedances": False,
        "use_delta_commands": False,
    }
    is_drawer = str(task_name or "").startswith("openarm_drawer_storage")
    is_fruit_bagging = str(task_name or "").startswith("openarm_fruit_basket_bagging")
    if not is_openarm_real_exp_1(task_name) and not is_drawer and not is_fruit_bagging:
        return cfg
    isaac_kp = _optional_float_env("OPENARM_GRIPPER_ISAAC_KP")
    isaac_kd = _optional_float_env("OPENARM_GRIPPER_ISAAC_KD")
    if is_drawer or is_fruit_bagging:
        isaac_kp = 3000.0 if isaac_kp is None else isaac_kp
        isaac_kd = 500.0 if isaac_kd is None else isaac_kd
    if isaac_kp is not None:
        cfg["isaac_kp"] = isaac_kp
    if isaac_kd is not None:
        cfg["isaac_kd"] = isaac_kd
    return cfg


def _openarm_camera_mount_file_candidates(task_name):
    """Return task-specific camera mount files, most-specific first."""
    candidates = openarm_camera_mount_file_candidates(task_name)
    if canonical_openarm_task_name(task_name) in PUBLISHED_OPENARM_TASKS:
        return candidates[:1]

    runtime_config_dir = os.path.dirname(candidates[-1]) if candidates else os.path.abspath("generation_runtime/configs")
    if str(task_name).startswith(OPENARM_CASHIER_STYLE_PREFIXES):
        candidates.append(os.path.join(runtime_config_dir, "openarm_cashier_camera_mounts.json"))
    candidates.append(os.path.join(runtime_config_dir, "openarm_camera_mounts.json"))
    return candidates


class EnvErrTypes(str, Enum):
    ArmMPFailed = "ArmMPFailed"
    BaseMPFailed = "BaseMPFailed"
    BaseSamplingFailed = "BaseSamplingFailed"

def hori_concatenate_image(images):
    # Ensure the images have the same height
    image1 = images[0]
    concatenated_image = image1
    for i in range(1, len(images)):
        image_i = images[i]
        if image1.shape[0] != image_i.shape[0]:
            # print("Images do not have the same height. Resizing the second image.")
            height = image1.shape[0]
            image_i = cv2.resize(image_i, (int(image_i.shape[1] * (height / image_i.shape[0])), height))

        # Concatenate the images side by side
        concatenated_image = np.concatenate((concatenated_image, image_i), axis=1)

    return np.array(concatenated_image)


def _to_uint8_rgb(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    img = img[..., :3]
    if img.dtype != np.uint8:
        if img.max(initial=0) <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _resize_rgb(img, width, height):
    return cv2.resize(_to_uint8_rgb(img), (width, height), interpolation=cv2.INTER_AREA)

def fps_downsample(color_pcd, num_points_to_sample):
    if color_pcd.shape[0] > num_points_to_sample:
        pc = color_pcd[:, 3:]
        color_img = color_pcd[:, :3]
        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, num_points_to_sample, h=5)
        pc = pc[kdline_fps_samples_idx]
        color_img = color_img[kdline_fps_samples_idx]
        color_pcd = np.concatenate([color_img, pc], axis=-1)
    else:
        raise ValueError("color_pcd shape is smaller than num_points_to_sample")
    return color_pcd

def pcd_vis(pc):
    # visualize with open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3)) 
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])
    print('number points', pc.shape[0])

def color_pcd_vis(color_pcd):
    # visualize with open3D
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(color_pcd[:, :3])
    pcd.points = o3d.utility.Vector3dVector(color_pcd[:,3:]) 
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])
    print('number points', color_pcd.shape[0])

class EnvOmniGibson(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self,
        env_name,
        policy_rollout=False,
        manipulation_only=False,
        real_robot_mode=False,
        baseline=None,
        **kwargs,
    ):
        self._env_name = env_name
        self.source_dataset_path_for_pose = kwargs.pop("_source_dataset_path_for_pose", None)
        self.disable_scene_lights = kwargs.pop("disable_scene_lights", False)
        default_viewer_light_rig_mode = os.environ.get("OPENARM_VIEWER_LIGHT_RIG_MODE")
        if default_viewer_light_rig_mode is None:
            default_viewer_light_rig_mode = (
                LightingMode.STAGE.value
                if is_openarm_real_exp_1(env_name)
                else LightingMode.RIG_DEFAULT.value
            )
        self.viewer_light_rig_mode = kwargs.pop(
            "viewer_light_rig_mode",
            default_viewer_light_rig_mode,
        )
        with og_macros.unlocked():
            gm.FORCE_LIGHT_INTENSITY = 0.0 if self.disable_scene_lights else DEFAULT_FORCE_LIGHT_INTENSITY
        self._init_kwargs = deepcopy(kwargs)
        self._init_kwargs["disable_scene_lights"] = self.disable_scene_lights
        self._init_kwargs["viewer_light_rig_mode"] = self.viewer_light_rig_mode
        self.add_distractor_objects = False
        self.single_arm = "right"
        self.policy_rollout = policy_rollout
        self.with_color = True
        self.manipulation_only = manipulation_only
        self.real_robot_mode = real_robot_mode
        self.init_nav_manip = False
        self.debug_from_saved_state = False
        # self.retract_type = "retract_to_start_of_arm_mp"     # Options: ["no_retract", "retract_to_canonical_pose_maintain_orn", "retract_to_canonical_pose", "retract_to_start_of_arm_mp"]
        self.phases_completed_wo_mp_err = 0
        # execution_phase_ind keeps track of each phase that was tried to be executed (even if MP failed for that phase). This is useful for logging phase
        # specific information (which we want to do even if there is a MP failure)
        self.execution_phase_ind = 0
        self.retry_nav_on_arm_mp_failure = False
        self.num_nav_retry_on_arm_mp_failure = 0
        self.use_base_pose_hack = False
        self.baseline = baseline
        self.check_upright = ["pot_plant"]
        self.start_nav_step = 0

        # Visibility parameters
        self.soft_visibility_constraint = True
        self.hard_visibility_constraint = True

        # Default attributes
        self.robot_reset_pos = "untuck"
        if kwargs["robots"]:
            self.reset_base_pose = (kwargs["robots"][0]["position"], kwargs["robots"][0]["orientation"])
        else:
            self.reset_base_pose = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

        # Task specific updates to kwargs
        if self.name.startswith("r1_pick_cup"):
            self.update_params_r1_pick_cup(kwargs)
            # need to use untuck other wise real-robot joint limits make tucked version out of limit
            self.robot_reset_pos = "untuck"       # Options: ["tuck", "untuck"]
            if self.name.endswith("D0"):
                self.use_base_pose_hack=True
        elif self.name.startswith("r1_tidy_table"):
            self.update_params_r1_tidy_table(kwargs)
            self.robot_reset_pos = "tuck"       # Options: ["tuck", "untuck"]
        elif self.name.startswith("r1_dishes_away"):
            self.update_params_r1_dishes_away(kwargs)
            self.robot_reset_pos = "tuck"       # Options: ["tuck", "untuck"]
            # With tucked and without hard visibility constraint there are a lot of self-collion or bound violations because of the robot mostly manipulating the
            # object in weird poses, so we make it untucked
            if not self.hard_visibility_constraint:
                self.robot_reset_pos = "untuck"

            # The human teleporting is not doing anything for the first n steps
            if baseline in ["mimicgen", "skillgen"]:
                self.start_nav_step = 700

        elif self.name.startswith("r1_clean_pan"):
            self.update_params_r1_clean_pan(kwargs)
            self.robot_reset_pos = "untuck"       # Options: ["tuck", "untuck"]
        
        elif self.name.startswith("r1_bringing_water"):
            self.update_params_r1_bringing_water(kwargs)
            self.robot_reset_pos = "tuck"       # Options: ["tuck", "untuck"]

        elif self.name.startswith(OPENARM_TASK_PREFIXES):
            self.soft_visibility_constraint = False   # 新增：无头部，不需要可见性约束
            self.hard_visibility_constraint = False   # 新增
            if self.name.startswith(OPENARM_CASHIER_STYLE_PREFIXES):
                self.update_params_openarm_cashier(kwargs)
            else:
                self.update_params_openarm_pick_cup(kwargs)
            self.robot_reset_pos = "untuck"     # Options: ["tuck", "untuck"]

        # Since the interpolation in mimicgen does not work well with the robot tucked. This is due to subpar IK controller probably
        if baseline == "mimicgen":
            self.robot_reset_pos = "untuck"

        # Some general updates to kwargs. Always call this after the task specific updates in the previous lines
        self.update_kwargs(kwargs)

        if self.policy_rollout:
            # customize the environment for policy rollout
            # 1. rewrite the robot pos to the same one when teleoperation
            # kwargs["robots"][0]["position"] = robot_pos
            # kwargs["robots"][0]["orientation"] = robot_quat

            if self.name.startswith("r1_pick_cup"):
                # 2. set the bbox for ego-centric pcd range
                self.x_range = [0.0, 2.3]
                self.y_range = [-0.5, 0.5]
                self.z_range = [0.7, 2.0]

                # 3. speficy the table height, and cup heigth, and intrinsic matrix
                self.cup_mask_height = 0.95

                if self.manipulation_only:
                    self.intrinsic_matrix = np.array([
                        [174.08,   0.000, 128.000],
                        [  0.000, 174.08, 128.000],
                        [  0.000,   0.000,   1.000]])
                    self.table_mask_height = 0.755 # this is used when fps the pcd in two groups

                if self.init_nav_manip:
                    # the initial nav+manip 2 demos has differernt intrinsic matrix and resolution
                    self.intrinsic_matrix = np.array([
                        [87.04,   0.000, 64.000],
                        [  0.000, 87.04, 64.000],
                        [  0.000,   0.000,   1.000]])
                    self.table_mask_height = 0.77
                    
                    # TODO: hacky!!, this resolution only works for the first 2 demos got from mobile manipulation
                    kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_height"] = 128
                    kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_width"] = 128
                    

        if og.sim is not None:
            og.sim.stop()
            og.clear()

        self.env = og.Environment(configs=kwargs)
        self._apply_viewer_lighting()

        self.robot = self.env.robots[0]
        self.robot_name = self.env.robots[0].name
        
        # Custom env parameters
        self.valid_env = True
        self.err = "None"
        self.obj_visible_at_start_of_manip = False
        self.IL_obs_keys = ["rgb", "depth_linear", "seg_instance"]
        self.sampled_base_poses = {"failure": list(), "success": list()}

        # initializing dict for storing visibility stats
        self.num_frames_with_obj_visible = dict()
        for sensor_name, sensor in self.robot.sensors.items():
            if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                self.num_frames_with_obj_visible[sensor_name.split(":")[1]] = 0            
        
        base_controller_cfg = {"name": "HolonomicBaseJointController", "motor_type": "position", "command_input_limits": None, "use_impedances": False}
        # Since the data was collected in velocity mode, we need to set the controller to velocity mode for baselines which repeat the base motion
        if self.baseline:
            base_controller_cfg = {"name": "HolonomicBaseJointController", "motor_type": "velocity", "command_input_limits": (-1.0, 1.0), "command_output_limits": ((-0.75, -0.75, -1.0), (0.75, 0.75, 1.0)), "use_impedances": False}

        if self.name.startswith(OPENARM_TASK_PREFIXES):
            # OpenArmBimanual requires JointController for grippers (no trunk/camera controller)
            controller_config = {
                "arm_left": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
                "arm_right": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
                "gripper_left": _openarm_gripper_joint_controller_config(self.name),
                "gripper_right": _openarm_gripper_joint_controller_config(self.name),
            }
        else:
            controller_config = {
                "base": base_controller_cfg,
                "trunk": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
                "arm_left": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
                "arm_right": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
                "gripper_left": {"name": "MultiFingerGripperController", "mode": "binary", "command_input_limits": (0.0, 1.0),},
                "gripper_right": {"name": "MultiFingerGripperController", "mode": "binary", "command_input_limits": (0.0, 1.0),},
                "camera": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "command_input_limits": None, "use_impedances": False},
            }
        self.robot.reload_controllers(controller_config=controller_config)
        self._apply_openarm_gripper_joint_limits()
        if hasattr(self.robot, "grasping_mode"):
            requested_grasping_mode = _openarm_task_grasping_mode(self.name)
            if getattr(self.robot, "grasping_mode", None) != requested_grasping_mode:
                print(
                    "[OpenArm] overriding grasping_mode {} -> {}".format(
                        getattr(self.robot, "grasping_mode", None),
                        requested_grasping_mode,
                    )
                )
                self.robot._grasping_mode = requested_grasping_mode
            if requested_grasping_mode == "physical":
                self.robot._disable_grasp_handling = True
                for arm in getattr(self.robot, "arm_names", []):
                    if hasattr(self.robot, "_ag_obj_in_hand"):
                        self.robot._ag_obj_in_hand[arm] = None
                    if hasattr(self.robot, "_ag_grasp_counter"):
                        self.robot._ag_grasp_counter[arm] = None
                    if hasattr(self.robot, "_ag_release_counter"):
                        self.robot._ag_release_counter[arm] = None
            print(
                "[OpenArm] robot_class={} effective grasping_mode={} (OPENARM_GRASPING_MODE={!r})".format(
                    type(self.robot).__name__,
                    getattr(self.robot, "grasping_mode", None),
                    os.environ.get("OPENARM_GRASPING_MODE"),
                )
            )

        # add distractor objects if D2
        if self.name.endswith("D2"):
            self.distractor_objects = list()

        # Perform any post env creation setup
        if self.name.startswith("r1_pick_cup"):
            self.update_env_post_creation_r1_pick_cup()
        elif self.name.startswith("r1_tidy_table"):
            self.update_env_post_creation_r1_tidy_table()
        elif self.name.startswith("r1_dishes_away"):
            self.update_env_post_creation_r1_dishes_away()
        elif self.name.startswith("r1_clean_pan"):
            self.update_env_post_creation_r1_clean_pan()
        elif self.name.startswith(OPENARM_TASK_PREFIXES):
            pass  # No post-creation customization needed for OpenArm tasks

        # self.robot._grasping_mode = "sticky"
        from omnigibson.macros import macros
        grasp_window = _openarm_grasp_window()
        with macros.unlocked():
            macros.robots.manipulation_robot.GRASP_WINDOW = grasp_window
        print(f"[OpenArm] GRASP_WINDOW={grasp_window}")
        self.env.scene.update_initial_file()

        self._configure_openarm_task_physics()
        self.customize_physical_properties()
        self._apply_viewer_lighting()
        self.sensor_info = self.sensor_setup()

        # Hide the robot eef links' visual meshes if not in debug mode
        if not DEBUG:
            for eef_link_name in self.robot.eef_link_names.values():
                if "VisualSphere" in self.robot.links[eef_link_name].visual_meshes:
                    self.robot.links[eef_link_name].visual_meshes["VisualSphere"].visible = False

        # Debug visualization
        self.eef_current_marker = PrimitiveObject(
            relative_prim_path="/eef_current_marker",
            primitive_type="Cube",
            name="eef_current",
            size=th.tensor([0.03, 0.03, 0.1]),
            visual_only=True,
            rgba=th.tensor([1, 0, 0, 1]),
        ) if DEBUG else None
        self.eef_goal_marker = PrimitiveObject(
            relative_prim_path="/eef_goal_marker",
            primitive_type="Cube",
            name="eef_goal_marker",
            size=th.tensor([0.03, 0.03, 0.1]),
            visual_only=True,
            rgba=th.tensor([0, 1, 0, 1]),
        ) if DEBUG else None

        # Debug visualization for bimanual setup
        self.eef_current_marker_left = PrimitiveObject(
            relative_prim_path="/eef_current_marker_left",
            primitive_type="Cube",
            name="eef_current_left",
            size=th.tensor([0.03, 0.03, 0.1]),
            visual_only=True,
            rgba=th.tensor([1, 0, 0, 1]),
        ) if DEBUG else None
        self.eef_goal_marker_left = PrimitiveObject(
            relative_prim_path="/eef_goal_marker_left",
            primitive_type="Cube",
            name="eef_goal_marker_left",
            size=th.tensor([0.03, 0.03, 0.1]),
            visual_only=True,
            rgba=th.tensor([0, 1, 0, 1]),
        ) if DEBUG else None
        self.eef_current_marker_right = PrimitiveObject(
            relative_prim_path="/eef_current_marker_right",
            primitive_type="Cube",
            name="eef_current_right",
            size=th.tensor([0.03, 0.03, 0.1]),
            visual_only=True,
            rgba=th.tensor([1, 0, 0, 1]),
        ) if DEBUG else None
        self.eef_goal_marker_right = PrimitiveObject(
            relative_prim_path="/eef_goal_marker_right",
            primitive_type="Cube",
            name="eef_goal_marker_right",
            size=th.tensor([0.03, 0.03, 0.1]),
            visual_only=True,
            rgba=th.tensor([0, 0, 1, 1]),
        ) if DEBUG else None

        if DEBUG:
            # og.sim.batch_add_objects([self.eef_current_marker, self.eef_goal_marker], [self.env.scene] * 2)
            og.sim.batch_add_objects([self.eef_current_marker_left, self.eef_goal_marker_left, 
                                      self.eef_current_marker_right, self.eef_goal_marker_right], [self.env.scene] * 4)
            og.sim.step()

        # Call reset so that robot is set to its initial pose as curobo warmup (base) depends on that (due to the joint limits of the base)
        self.env.robots[0].set_position_orientation(position=th.tensor(self.reset_base_pose[0]), orientation=th.tensor(self.reset_base_pose[1]))
        for _ in range(5): og.sim.step()

        if self._init_kwargs['init_curobo']:
        # if not self.policy_rollout:
            # For OpenArmBimanual, disable self-collision checks to avoid segfault
            # caused by too many collision pairs (>16384) in CuRobo.
            _curobo_motion_cfg_kwargs = None
            if isinstance(self.env.robots[0], OpenArmBimanual):
                _curobo_motion_cfg_kwargs = {"self_collision_check": False, "self_collision_opt": False}
                # Lift robot 2cm before CuRobo warm-up to avoid floor-penetration collision
                # warnings (same technique as plan_openarm_curobo_jointspace.py).
                _pos, _quat = self.env.robots[0].get_position_orientation()
                _lift = th.tensor([0.0, 0.0, 0.02], dtype=_pos.dtype, device=_pos.device)
                self.env.robots[0].set_position_orientation(position=_pos + _lift, orientation=_quat)
                for _ in range(3): og.sim.step()
            # Head tracking with soft visibility constraint requires use_cuda_graph=False
            self.primitive = StarterSemanticActionPrimitives(
                self.env,
                self.env.robots[0],
                enable_head_tracking=self.soft_visibility_constraint or isinstance(self.env.robots[0], Tiago), # TODO: for now, Tiago should always have head tracking enabled
                curobo_batch_size=6,
                # curobo_use_cuda_graph=not self.soft_visibility_constraint,
                curobo_use_cuda_graph=False,
                use_base_pose_hack=self.use_base_pose_hack,
                real_robot_mode=self.real_robot_mode,
                curobo_motion_cfg_kwargs=_curobo_motion_cfg_kwargs,
            )
            if isinstance(self.env.robots[0], OpenArmBimanual):
                # Restore robot to its original position after warm-up
                self.env.robots[0].set_position_orientation(position=_pos, orientation=_quat)
                for _ in range(3): og.sim.step()

            # Create CuRobo instance
            self.cmg = self.primitive._motion_generator


        self.global_env_step = 0    

        # Obtain reset left and right eef pose and eyes pose
        self.left_eef_reset_pose_wrt_robot = self.robot.get_relative_eef_pose(arm="left") 
        self.right_eef_reset_pose_wrt_robot = self.robot.get_relative_eef_pose(arm="right") 
        if "eyes" in self.robot.links:
            eyes_reset_pose_wrt_world = self.robot.links["eyes"].get_position_orientation()
            robot_pose_wrt_world = self.robot.get_position_orientation()
            self.eyes_reset_pose_wrt_robot = T.mat2pose(th.linalg.inv(T.pose2mat(robot_pose_wrt_world)) @ T.pose2mat(eyes_reset_pose_wrt_world))
        else:
            self.eyes_reset_pose_wrt_robot = None


    def _apply_openarm_gripper_joint_limits(self):
        is_drawer = self.name.startswith("openarm_drawer_storage")
        is_fruit_bagging = self.name.startswith("openarm_fruit_basket_bagging")
        if not isinstance(self.robot, OpenArmBimanual) or not (
            is_openarm_real_exp_1(self.name) or is_drawer or is_fruit_bagging
        ):
            return
        max_velocity = _optional_float_env("OPENARM_GRIPPER_MAX_VELOCITY")
        max_effort = _optional_float_env("OPENARM_GRIPPER_MAX_EFFORT")
        if is_drawer or is_fruit_bagging:
            max_velocity = 0.10 if max_velocity is None else max_velocity
            max_effort = 150.0 if max_effort is None else max_effort
        if max_velocity is None and max_effort is None:
            return
        for arm in self.robot.arm_names:
            for joint in self.robot.finger_joints[arm]:
                if max_velocity is not None:
                    joint.max_velocity = max_velocity
                if max_effort is not None:
                    joint.max_effort = max_effort
        print(f"[OpenArm] Set gripper finger joint max_velocity={max_velocity}, max_effort={max_effort}")


    def step(self, action, video_writer=None):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, truncated, info = self.env.step(action)
        if video_writer:
            camera_names = getattr(video_writer, "camera_names", None)
            if camera_names:
                img = self._render_video_cameras(camera_names)
            # Use external_sensor2 if available (R1 scenes), otherwise fall back to viewer camera (e.g. OpenArm)
            elif self.env._external_sensors is not None and "external_sensor2" in self.env._external_sensors:
                img = self.env._external_sensors["external_sensor2"].get_obs()[0]["rgb"][:,:,:3].numpy()
            else:
                img = og.sim.viewer_camera._get_obs()[0]['rgb'].numpy()[:, :, :3]
            video_writer.append_data(img)
        #     for env_idx, single_env in enumerate(self.env.envs):
        #         external_obs = single_env.external_sensors["external_sensor0"].get_obs()[0]["rgb"][:,:,:3].numpy()
        #         video_writer[env_idx].append_data(external_obs)

        # replace the observation with newly added IL obs function 
        obs, obs_info = self.get_obs_IL()
        
        # return obs, r, done, info
        # changed to output with truncated
        return obs, r, done, truncated, info

    def _render_video_cameras(self, camera_names):
        frames = []
        obs = None
        for camera_name in camera_names:
            frame = None
            if self.env._external_sensors is not None and camera_name in self.env._external_sensors:
                frame = self.env._external_sensors[camera_name].get_obs()[0]["rgb"]
            elif camera_name == "viewer_camera":
                if isinstance(self.robot, OpenArmBimanual) and self.name.startswith(OPENARM_CASHIER_STYLE_PREFIXES):
                    self._set_cashier_viewer_camera_pose()
                frame = og.sim.viewer_camera._get_obs()[0]["rgb"]
            else:
                if obs is None:
                    obs, _ = self.env.get_obs()
                robot_name = self.env.robots[0].name
                candidate_keys = [
                    camera_name,
                    f"{camera_name}::rgb",
                    f"{robot_name}::{camera_name}::rgb",
                    f"{robot_name}::{robot_name}:{camera_name}:Camera:0::rgb",
                ]
                for key in candidate_keys:
                    if key in obs:
                        frame = obs[key]
                        break
            if frame is None:
                raise KeyError(f"Could not find camera '{camera_name}' for video rendering")
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            elif hasattr(frame, "numpy"):
                frame = frame.numpy()
            frames.append(_to_uint8_rgb(frame))
        if self._should_use_cashier_video_layout(camera_names):
            return self._make_cashier_video_layout(frames)
        return hori_concatenate_image(frames)

    def _should_use_cashier_video_layout(self, camera_names):
        return (
            isinstance(self.robot, OpenArmBimanual)
            and self.name.startswith(OPENARM_CASHIER_STYLE_PREFIXES)
            and list(camera_names) == ["viewer_camera", "left_wrist_cam", "right_wrist_cam", "base_cam"]
        )

    def _make_cashier_video_layout(self, frames):
        tile_size = int(os.environ.get("OPENARM_CASHIER_VIDEO_TILE_SIZE", "256"))
        viewer_height = int(os.environ.get("OPENARM_CASHIER_VIDEO_VIEWER_HEIGHT", str(tile_size * 2)))

        bottom_frames = [_resize_rgb(frame, tile_size, tile_size) for frame in frames[1:]]
        bottom_row = hori_concatenate_image(bottom_frames)

        viewer_frame = _resize_rgb(frames[0], bottom_row.shape[1], viewer_height)
        return np.concatenate([viewer_frame, bottom_row], axis=0)

    def _turn_off_scene_lights(self):
        world = lazy.isaacsim.core.utils.prims.get_prim_at_path("/World")
        if not world:
            return

        def recursive_light_update(prim):
            if "Light" in prim.GetPrimTypeInfo().GetTypeName():
                intensity_attr = prim.GetAttribute("inputs:intensity")
                if intensity_attr and intensity_attr.IsValid():
                    intensity_attr.Set(0.0)
            for child_prim in prim.GetChildren():
                recursive_light_update(child_prim)

        recursive_light_update(world)

    def _apply_viewer_lighting(self):
        if self.disable_scene_lights:
            self._turn_off_scene_lights()
        else:
            self._set_viewer_light_rig_mode()

    def _set_viewer_light_rig_mode(self):
        if og.sim is None:
            return

        mode = self.viewer_light_rig_mode
        aliases = {
            "default": LightingMode.RIG_DEFAULT,
            "rig_default": LightingMode.RIG_DEFAULT,
            "grey": LightingMode.RIG_GREY,
            "grey_studio": LightingMode.RIG_GREY,
            "gray": LightingMode.RIG_GREY,
            "gray_studio": LightingMode.RIG_GREY,
            "colored": LightingMode.RIG_COLORED,
            "colored_lights": LightingMode.RIG_COLORED,
            "stage": LightingMode.STAGE,
            "stage_light": LightingMode.STAGE,
            "camera": LightingMode.CAMERA,
        }
        if isinstance(mode, str):
            mode = aliases.get(mode.strip().lower().replace(" ", "_"), mode)
        if isinstance(mode, str):
            mode = LightingMode(mode)
        og.sim.set_lighting_mode(mode=mode)

    # Get task relevant objects based on the env name (BDDL activity name)
    def _get_task_relevant_objs(self):
        task_name = self.name.rsplit('_', 1)[0]
        if task_name in TASK_CONFIGS:
            if task_name.startswith("openarm_bag_groceries"):
                # Only randomize / motion-check the movable groceries. The two
                # paper bags are fixed targets and should stay at the source pose.
                task_relevant_obj_names = ["apple_1", "candy_1", "orange_1", "canned_food_1"]
            elif task_name.startswith("openarm_real_exp_1"):
                task_relevant_obj_names = ["object_1", "object_2"]
            elif task_name.startswith("openarm_drawer_storage"):
                # The cabinet and checkout counter are fixed task geometry. Moving
                # either during D0 sampling breaks the drawer frame and can make
                # the cabinet OnTop condition impossible to re-satisfy.
                task_relevant_obj_names = ["object_1", "object_2"]
            elif task_name.startswith("openarm_cashier"):
                task_relevant_obj_names = ["apple_1", "candy_1"]
            else:
                task_relevant_obj_names = list(TASK_CONFIGS[task_name].tracked_objects.keys())
            return [self.env.scene.object_registry("name", name) for name in task_relevant_obj_names]
        else:
            raise ValueError(f"Unknown environment name: {self.name}")

    def early_termination(self, env_step, ob_dict=None):
        """
        Check if the episode should be terminated early.
        """
        if env_step < 20:
            self.initial_positions = {}
            for obj in self._get_task_relevant_objs():
                self.initial_positions[obj.name] = obj.get_position_orientation()
        
        # check table movement
        cur_positions = {}
        for obj in self._get_task_relevant_objs():
            cur_positions[obj.name] = obj.get_position_orientation()
        
        for key in self.initial_positions.keys():
            if 'table' in key: # if table is moved, directly terminate the episode 
                if np.linalg.norm(self.initial_positions[key][0] - cur_positions[key][0]) > 0.1:
                    return True
        
        # early termination when the robot get stuck, NOT WORKING NOW
        # if env_step == 0:
        #     # note that the poses are in the robot frame
        #     self.old_eef_left_pos = copy.deepcopy(ob_dict['eef_left_pos'])
        #     self.old_eef_left_quat = copy.deepcopy(ob_dict['eef_left_quat'])
        #     self.old_eef_right_pos = copy.deepcopy(ob_dict['eef_right_pos'])
        #     self.old_eef_right_quat = copy.deepcopy(ob_dict['eef_right_quat'])
        # # if the robot is stuck for n steps, early terminate the episode
        # if env_step > 150 and env_step % 100 == 0:
        #     print('env_step', env_step)
        #     # update the robot eef position
        #     cur_eef_left_pos = ob_dict['eef_left_pos']
        #     cur_eef_left_quat = ob_dict['eef_left_quat']
        #     cur_eef_right_pos = ob_dict['eef_right_pos']
        #     cur_eef_right_quat = ob_dict['eef_right_quat']
        #     left_eef_pos_diff = np.linalg.norm(self.old_eef_left_pos - cur_eef_left_pos) 
        #     left_eef_pos_nomove = left_eef_pos_diff < 0.02 
        #     left_eef_quat_diff = np.linalg.norm(self.old_eef_left_quat - cur_eef_left_quat)
        #     left_eef_quat_nomove = left_eef_quat_diff < 0.012
        #     right_eef_pos_diff = np.linalg.norm(self.old_eef_right_pos - cur_eef_right_pos) 
        #     right_eef_pos_nomove = right_eef_pos_diff < 0.02 
        #     right_eef_quat_diff = np.linalg.norm(self.old_eef_right_quat - cur_eef_right_quat)
        #     right_eef_quat_nomove = right_eef_quat_diff < 0.012

        #     self.old_eef_left_pos = copy.deepcopy(cur_eef_left_pos)
        #     self.old_eef_left_quat = copy.deepcopy(cur_eef_left_quat)
        #     self.old_eef_right_pos = copy.deepcopy(cur_eef_right_pos)
        #     self.old_eef_right_quat = copy.deepcopy(cur_eef_right_quat)

        #     if left_eef_pos_nomove and left_eef_quat_nomove and right_eef_pos_nomove and right_eef_quat_nomove:
        #         print('enter no move breakpoint')
        #         print('')
        #         print('left_eef_pos_nomove', left_eef_pos_diff, left_eef_pos_nomove)
        #         print('left_eef_quat_nomove', left_eef_quat_diff, left_eef_quat_nomove)
        #         print('right_eef_pos_nomove', right_eef_pos_diff, right_eef_pos_nomove)
        #         print('right_eef_quat_nomove', right_eef_quat_diff, right_eef_quat_nomove)
        #         print('')
        #         breakpoint()
        #         # return True
        #         return False
                
        return False

    def set_object_pose(self, obj_poses):
        """
        Set the object pose for the task relevant objects
        """
        if 'states' in obj_poses.keys(): obj_poses = obj_poses['states']
        if self.name.startswith("test_r1_cup"):
            task_relevant_objs = self._get_task_relevant_objs()
            for obj in task_relevant_objs:
                if 'table' not in obj.name:
                    obj.set_position_orientation(obj_poses[obj.name][:3], obj_poses[obj.name][3:])
            print('finishe setting object pose for r1 robot')

        elif self.name.startswith("r1_pick_cup"):
            task_relevant_objs = self._get_task_relevant_objs()
            for obj in task_relevant_objs:
                if 'table' not in obj.name:
                    if obj.name == 'coffee_cup_7':
                        # print("")
                        # print('initial coffee cup position', obj.get_position_orientation())
                        obj.set_position_orientation(obj_poses["coffee_cup"][:3], obj_poses["coffee_cup"][3:])
                        # pose = [1.034, -0.1601,  0.7769]  
                        # 6.359e-03 -1.849e-04  9.996e-01 -2.692e-02
                        # print('after reset coffee cup position', obj.get_position_orientation())
                        # print("")
            for _ in range(5): og.sim.step()
            for _ in range(5): og.sim.render()
            print('finishe setting object pose for r1 robot')

    @staticmethod
    def _apply_planar_pose_delta(obj, delta):
        position, orientation = obj.get_position_orientation()
        new_position = position.clone() if hasattr(position, "clone") else np.array(position, copy=True)
        new_position[0] += float(delta.delta_xy[0])
        new_position[1] += float(delta.delta_xy[1])
        yaw_delta = th.tensor(
            [0.0, 0.0, np.deg2rad(float(delta.delta_yaw_deg))],
            dtype=orientation.dtype,
            device=orientation.device,
        )
        new_orientation = T.mat2quat(T.euler2mat(yaw_delta) @ T.quat2mat(orientation))
        obj.set_position_orientation(position=new_position, orientation=new_orientation)

    def _drawer_storage_open_sweep_clearance(self, spec):
        if spec.object_drawer_clearance_m <= 0.0:
            return True, None, {}
        target = OPENARM_DRAWER_TARGETS["lower_drawer"]
        cabinet = target.resolve_object(self.env)
        drawer_link = target.resolve_drawer_link(self.env)
        _, cabinet_orientation = cabinet.get_position_orientation()
        pull_axis = T.quat2mat(cabinet_orientation) @ th.tensor(
            [1.0, 0.0, 0.0],
            dtype=cabinet_orientation.dtype,
            device=cabinet_orientation.device,
        )
        translation_xy = (
            pull_axis[:2].detach().cpu().numpy() * float(spec.drawer_open_distance_m)
        )
        distances = {}
        for object_name in ("object_1", "object_2"):
            obj = self.env.scene.object_registry("name", object_name)
            distance = aabb_xy_distance_to_translated_sweep(
                obj.aabb_center,
                obj.aabb_extent,
                drawer_link.aabb_center,
                drawer_link.aabb_extent,
                translation_xy,
            )
            distances[object_name] = distance
            if distance < spec.object_drawer_clearance_m:
                return (
                    False,
                    f"open_drawer_clearance:{object_name}:{distance:.6f}",
                    distances,
                )
        return True, None, distances

    def _drawer_storage_object_pair_clearance(self, spec):
        object_1 = self.env.scene.object_registry("name", "object_1")
        object_2 = self.env.scene.object_registry("name", "object_2")
        distance = aabb_xy_distance_to_translated_sweep(
            object_1.aabb_center,
            object_1.aabb_extent,
            object_2.aabb_center,
            object_2.aabb_extent,
            [0.0, 0.0],
        )
        if distance < spec.object_pair_clearance_m:
            return False, f"object_pair_clearance:{distance:.6f}", distance
        return True, None, distance

    def _drawer_storage_reset_is_valid(self, spec):
        for object_name in ("object_1", "object_2"):
            obj = self.env.scene.object_registry("name", object_name)
            condition = self._get_relevant_initial_condition(obj)
            if condition is not None and not bool(condition.evaluate()):
                return False, f"initial_condition_failed:{object_name}"

        cabinet = self.env.scene.object_registry("name", "drawer_cabinet_1")
        support = self.env.scene.object_registry("name", "checkout_counter_umnoqh_0")
        supported, reason = validate_aabb_support(
            cabinet.aabb_center,
            cabinet.aabb_extent,
            support.aabb_center,
            support.aabb_extent,
        )
        if not supported:
            return False, reason
        drawer_fraction = OPENARM_DRAWER_TARGETS["lower_drawer"].get_open_fraction(self.env)
        if drawer_fraction > 0.02:
            return False, f"drawer_not_closed:{drawer_fraction:.6f}"
        sweep_valid, reason, _ = self._drawer_storage_open_sweep_clearance(spec)
        if not sweep_valid:
            return False, reason
        pair_valid, reason, _ = self._drawer_storage_object_pair_clearance(spec)
        if not pair_valid:
            return False, reason
        return True, None

    def _apply_drawer_robot_joint_delta(self, sample):
        q = self.robot.get_joint_positions().clone()
        lower = self.robot.joint_lower_limits
        upper = self.robot.joint_upper_limits
        margin = np.deg2rad(0.5)
        for arm in ("left", "right"):
            indices = self.robot.arm_control_idx[arm]
            delta = th.deg2rad(
                th.tensor(
                    sample.robot_arm_joint_delta_deg[arm],
                    dtype=q.dtype,
                    device=q.device,
                )
            )
            candidate = q[indices] + delta
            if th.any(candidate < lower[indices] + margin) or th.any(candidate > upper[indices] - margin):
                return False, f"robot_joint_limit:{arm}"
            q[indices] = candidate
        self.robot.set_joint_positions(q)
        return True, None

    def _apply_openarm_robot_joint_delta(self, sample):
        q = self.robot.get_joint_positions().clone()
        lower = self.robot.joint_lower_limits
        upper = self.robot.joint_upper_limits
        margin = np.deg2rad(0.5)
        for arm in ("left", "right"):
            indices = self.robot.arm_control_idx[arm]
            delta_values = sample.robot_arm_joint_delta_deg.get(arm, ())
            if not delta_values:
                continue
            delta = th.deg2rad(
                th.tensor(delta_values, dtype=q.dtype, device=q.device)
            )
            candidate = q[indices] + delta
            if th.any(candidate < lower[indices] + margin) or th.any(candidate > upper[indices] - margin):
                return False, f"robot_joint_limit:{arm}"
            q[indices] = candidate
        self.robot.set_joint_positions(q)
        return True, None

    def _aabb_xy_distance(self, first_name, second_name):
        first = self.env.scene.object_registry("name", first_name)
        second = self.env.scene.object_registry("name", second_name)
        return aabb_xy_distance_to_translated_sweep(
            first.aabb_center,
            first.aabb_extent,
            second.aabb_center,
            second.aabb_extent,
            [0.0, 0.0],
        )

    def _fruit_object_pair_clearance(self, spec):
        object_names = ("object_1", "object_2", "object_3", "object_4")
        distances = {}
        for i, first in enumerate(object_names):
            for second in object_names[i + 1:]:
                distance = self._aabb_xy_distance(first, second)
                distances[f"{first}:{second}"] = distance
                if distance < spec.object_pair_clearance_m:
                    return False, f"object_pair_clearance:{first}:{second}:{distance:.6f}", distances
        return True, None, distances

    def _fruit_object_container_clearance(self, spec):
        object_names = ("object_1", "object_2", "object_3", "object_4")
        container_names = ("paper_bag_1", "basket_1")
        distances = {}
        for object_name in object_names:
            for container_name in container_names:
                distance = self._aabb_xy_distance(object_name, container_name)
                distances[f"{object_name}:{container_name}"] = distance
                if distance < spec.object_container_clearance_m:
                    return False, f"object_container_clearance:{object_name}:{container_name}:{distance:.6f}", distances
        return True, None, distances

    def _fruit_container_pair_clearance(self, spec):
        distance = self._aabb_xy_distance("paper_bag_1", "basket_1")
        if distance < spec.container_pair_clearance_m:
            return False, f"container_pair_clearance:{distance:.6f}", distance
        return True, None, distance

    def _repair_fruit_object_container_clearance(self, spec):
        gap = float(getattr(spec, "object_container_repair_gap_m", 0.0))
        if gap <= 0.0:
            return {}
        object_names = ("object_1", "object_2", "object_3", "object_4")
        container_names = ("paper_bag_1", "basket_1")
        repair_offsets = {name: 0.0 for name in object_names}
        for object_name in object_names:
            obj = self.env.scene.object_registry("name", object_name)
            # Fruits start in front of both containers along +Y. Keep that
            # relation deterministic by pushing only forward if a container
            # AABB becomes too close after randomization.
            for _ in range(3):
                obj_min_y = float(obj.aabb_center[1] - obj.aabb_extent[1] / 2.0)
                required_push = 0.0
                for container_name in container_names:
                    container = self.env.scene.object_registry("name", container_name)
                    container_max_y = float(container.aabb_center[1] + container.aabb_extent[1] / 2.0)
                    current_gap = obj_min_y - container_max_y
                    if current_gap < gap:
                        required_push = max(required_push, gap - current_gap)
                if required_push <= 1e-6:
                    break
                pos, orn = obj.get_position_orientation()
                new_pos = pos.clone() if hasattr(pos, "clone") else np.array(pos, copy=True)
                new_pos[1] += required_push
                obj.set_position_orientation(position=new_pos, orientation=orn)
                repair_offsets[object_name] += float(required_push)
        return {name: value for name, value in repair_offsets.items() if abs(value) > 1e-9}

    def _fruit_table_support_valid(self):
        support = self.env.scene.object_registry("name", "checkout_counter_umnoqh_0")
        for object_name in (
            "object_1",
            "object_2",
            "object_3",
            "object_4",
            "paper_bag_1",
            "basket_1",
        ):
            obj = self.env.scene.object_registry("name", object_name)
            supported, reason = validate_aabb_support(
                obj.aabb_center,
                obj.aabb_extent,
                support.aabb_center,
                support.aabb_extent,
                vertical_tolerance=0.04,
                min_xy_coverage=0.50,
            )
            if not supported:
                return False, f"{object_name}:{reason}"
        return True, None

    def _fruit_left_to_right_order_valid(self):
        object_order = ("object_1", "object_2", "object_3", "object_4")
        centers = {
            name: float(self.env.scene.object_registry("name", name).aabb_center[0])
            for name in object_order
        }
        values = [centers[name] for name in object_order]
        if values != sorted(values):
            return False, "fruit_left_to_right_order_changed", centers
        return True, None, centers

    def _fruit_coarse_workspace_valid(self):
        # Broad table-side filter only. This is not a kinematic reachability
        # proof; CuRobo still performs the actual IK / motion-planning check
        # during generation.
        arm_bounds = {
            "left": {
                "x": (-1.22, -0.60),
                "y": (3.64, 3.94),
            },
            "right": {
                "x": (-1.30, -0.70),
                "y": (3.64, 3.94),
            },
        }
        object_arms = {
            "object_1": "right",
            "object_2": "right",
            "object_3": "left",
            "object_4": "left",
        }
        centers = {}
        for object_name, arm in object_arms.items():
            obj = self.env.scene.object_registry("name", object_name)
            center = obj.aabb_center
            x = float(center[0])
            y = float(center[1])
            centers[object_name] = {"arm": arm, "x": x, "y": y}
            bounds = arm_bounds[arm]
            if not (bounds["x"][0] <= x <= bounds["x"][1]):
                return False, f"coarse_workspace_x:{object_name}:{arm}:{x:.6f}", centers
            if not (bounds["y"][0] <= y <= bounds["y"][1]):
                return False, f"coarse_workspace_y:{object_name}:{arm}:{y:.6f}", centers
        return True, None, centers

    def _fruit_bagging_reset_is_valid(self, spec):
        supported, reason = self._fruit_table_support_valid()
        if not supported:
            return False, reason
        workspace_valid, reason, _ = self._fruit_coarse_workspace_valid()
        if not workspace_valid:
            return False, reason
        if spec.preserve_left_to_right_order:
            order_valid, reason, _ = self._fruit_left_to_right_order_valid()
            if not order_valid:
                return False, reason
        pair_valid, reason, _ = self._fruit_object_pair_clearance(spec)
        if not pair_valid:
            return False, reason
        container_valid, reason, _ = self._fruit_object_container_clearance(spec)
        if not container_valid:
            return False, reason
        container_pair_valid, reason, _ = self._fruit_container_pair_clearance(spec)
        if not container_pair_valid:
            return False, reason
        return True, None

    def _randomize_openarm_drawer_storage(self):
        difficulty = self.name.rsplit("_", 1)[-1].upper()
        overrides = getattr(self, "drawer_initialization_distribution", None)
        spec = difficulty_from_mapping(difficulty, overrides)
        seed = int(getattr(self, "drawer_initialization_seed", 0))
        attempt_index = int(getattr(self, "drawer_initialization_attempt_index", 0))
        initial_state = og.sim.dump_state()
        last_sample = None
        rejection_reason = None

        for candidate_index in range(spec.max_sampling_attempts):
            if candidate_index:
                og.sim.load_state(initial_state)
                for _ in range(2):
                    og.sim.step()
            sample = sample_drawer_initialization(
                difficulty=difficulty,
                seed=seed,
                attempt_index=attempt_index,
                candidate_index=candidate_index,
                overrides=overrides,
            )
            last_sample = sample
            for object_name, delta in (
                ("drawer_cabinet_1", sample.cabinet),
                ("object_1", sample.object_1),
                ("object_2", sample.object_2),
            ):
                self._apply_planar_pose_delta(
                    self.env.scene.object_registry("name", object_name), delta
                )
            joints_valid, rejection_reason = self._apply_drawer_robot_joint_delta(sample)
            if not joints_valid:
                continue
            for _ in range(8):
                og.sim.step()
            reset_valid, rejection_reason = self._drawer_storage_reset_is_valid(spec)
            if reset_valid:
                self.drawer_initialization_sample = sample.to_dict()
                _, _, sweep_distances = self._drawer_storage_open_sweep_clearance(spec)
                self.drawer_initialization_sample["open_drawer_sweep_distance_m"] = sweep_distances
                _, _, pair_distance = self._drawer_storage_object_pair_clearance(spec)
                self.drawer_initialization_sample["object_pair_distance_m"] = pair_distance
                break
        else:
            self.drawer_initialization_sample = last_sample.to_dict()
            self.drawer_initialization_sample.update(
                {
                    "reset_valid": False,
                    "forced_accept": True,
                    "rejection_reason": rejection_reason or "sampling_exhausted",
                }
            )
            print(
                "[drawer_init][WARN] candidate budget exhausted; "
                "executing the final sampled initialization"
            )

        self.drawer_initialization_attempt_index = attempt_index + 1
        print("[drawer_init] {}".format(json.dumps(self.drawer_initialization_sample, sort_keys=True)))

    def _randomize_openarm_fruit_basket_bagging(self):
        difficulty = self.name.rsplit("_", 1)[-1].upper()
        overrides = getattr(self, "fruit_initialization_distribution", None)
        spec = fruit_difficulty_from_mapping(difficulty, overrides)
        seed = int(getattr(self, "fruit_initialization_seed", 0))
        attempt_index = int(getattr(self, "fruit_initialization_attempt_index", 0))
        initial_state = og.sim.dump_state()
        last_sample = None
        rejection_reason = None

        for candidate_index in range(spec.max_sampling_attempts):
            if candidate_index:
                og.sim.load_state(initial_state)
                for _ in range(2):
                    og.sim.step()
            sample = sample_fruit_bagging_initialization(
                difficulty=difficulty,
                seed=seed,
                attempt_index=attempt_index,
                candidate_index=candidate_index,
                overrides=overrides,
            )
            last_sample = sample
            for object_name, delta in sample.objects.items():
                self._apply_planar_pose_delta(
                    self.env.scene.object_registry("name", object_name), delta
                )
            for container_name, delta in sample.containers.items():
                self._apply_planar_pose_delta(
                    self.env.scene.object_registry("name", container_name), delta
                )
            repair_offsets = self._repair_fruit_object_container_clearance(spec)
            joints_valid, rejection_reason = self._apply_openarm_robot_joint_delta(sample)
            if not joints_valid:
                continue
            for _ in range(8):
                og.sim.step()
            reset_valid, rejection_reason = self._fruit_bagging_reset_is_valid(spec)
            if reset_valid:
                self.fruit_initialization_sample = sample.to_dict()
                _, _, object_pair_distances = self._fruit_object_pair_clearance(spec)
                _, _, object_container_distances = self._fruit_object_container_clearance(spec)
                _, _, container_pair_distance = self._fruit_container_pair_clearance(spec)
                _, _, fruit_x_order = self._fruit_left_to_right_order_valid()
                _, _, fruit_coarse_workspace = self._fruit_coarse_workspace_valid()
                self.fruit_initialization_sample["object_pair_distance_m"] = object_pair_distances
                self.fruit_initialization_sample["object_container_distance_m"] = object_container_distances
                self.fruit_initialization_sample["container_pair_distance_m"] = container_pair_distance
                self.fruit_initialization_sample["fruit_x_order"] = fruit_x_order
                self.fruit_initialization_sample["coarse_workspace"] = fruit_coarse_workspace
                self.fruit_initialization_sample["object_container_repair_y_m"] = repair_offsets
                break
        else:
            self.fruit_initialization_sample = last_sample.to_dict()
            self.fruit_initialization_sample.update(
                {
                    "reset_valid": False,
                    "forced_accept": True,
                    "rejection_reason": rejection_reason or "sampling_exhausted",
                }
            )
            print(
                "[fruit_init][WARN] candidate budget exhausted; "
                "executing the final sampled initialization"
            )

        self.fruit_initialization_attempt_index = attempt_index + 1
        print("[fruit_init] {}".format(json.dumps(self.fruit_initialization_sample, sort_keys=True)))

    # TODO: make it more generalizable
    # randomize the pose of all the task relevant objects in xy-pos and z-rot
    def _randomize_object_pose_D0(self, objs):

        # default values
        pos_magnitude = [-0.1, 0.1] 
        rot_magnitude = np.pi / 12 # 15 degrees
        # Sampling random object poses using custom thresholds
        if self.name.startswith("r1_pick_cup"):
            pos_magnitude = [-0.15, 0.15] 
            rot_magnitude = np.pi / 12  # 15 degrees
        elif self.name.startswith("r1_dishes_away"):
            pos_magnitude = [-0.1, 0.1] 
            rot_magnitude = 0.01
        elif self.name.startswith("r1_tidy_table"):
            pos_magnitude = [-0.15, 0.15] 
            rot_magnitude = np.pi / 12  # 15 degrees
        elif self.name.startswith("r1_clean_pan"):
            pos_magnitude = [-0.15, 0.15] 
            rot_magnitude = np.pi / 12  # 15 degrees
        elif self.name.startswith("r1_bringing_water"):
            pos_magnitude = [-0.05, 0.05] 
            rot_magnitude = np.pi / 24 # 7.5 degrees
        elif self.name.startswith("openarm_bag_groceries"):
            pos_mag = float(os.environ.get("OPENARM_BAG_GROCERIES_D0_POS_MAG", "0.04"))
            rot_magnitude = float(os.environ.get("OPENARM_BAG_GROCERIES_D0_ROT_MAG", "0.0"))
            pos_magnitude = [-pos_mag, pos_mag]
        elif self.name.startswith("openarm_cashier"):
            # Env vars are left as a tuning hook for ablations or debugging.
            pos_mag = float(os.environ.get("OPENARM_CASHIER_D0_POS_MAG", "0.1"))
            rot_magnitude = float(os.environ.get("OPENARM_CASHIER_D0_ROT_MAG", str(np.pi / 12)))
            pos_magnitude = [-pos_mag, pos_mag]
        elif self.name.startswith("openarm_real_exp_1"):
            pos_mag = float(os.environ.get("OPENARM_REAL_EXP_1_D0_POS_MAG", "0.04"))
            rot_magnitude = float(os.environ.get("OPENARM_REAL_EXP_1_D0_ROT_MAG", "0.0"))
            pos_magnitude = [-pos_mag, pos_mag]
        elif self.name.startswith("openarm_drawer_storage"):
            pos_mag = float(os.environ.get("OPENARM_DRAWER_STORAGE_D0_POS_MAG", "0.02"))
            rot_magnitude = float(os.environ.get("OPENARM_DRAWER_STORAGE_D0_ROT_MAG", "0.0"))
            pos_magnitude = [-pos_mag, pos_mag]

        # pan = self.env.scene.object_registry("name", "frying_pan_602")
        # pan.set_position_orientation(th.tensor([5.2, -1.8, 0.908]), th.tensor([    -0.000,      0.000,     -0.499,      0.866]))
        # for _ in range(10): og.sim.step()
        # breakpoint()

        for obj in objs:
            if all(keyword not in obj.name for keyword in ["table", "shelf", "countertop", "sink", "fridge"]):
                pos, orn = obj.get_position_orientation()
                state = og.sim.dump_state()
                while True:
                    pos_diff_xy = np.random.uniform(pos_magnitude[0], pos_magnitude[1], size=2)
                    pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
                    new_pos = pos + pos_diff
                    orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
                    new_orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))
                    obj.set_position_orientation(new_pos, new_orn)
                    for _ in range(10):
                        og.sim.step()
                    cond = self._get_relevant_initial_condition(obj)
                    assert cond is not None, f"Condition not found for object {obj.name}"
                    # Don't do bddl check for clean pan baselines 
                    if self.name.startswith("r1_clean_pan") and self.baseline in ["mimicgen", "skillgen"]:
                        scrub = self.env.scene.object_registry("name", "scrub_brush_601")
                        sink = self.env.scene.object_registry("name", "drop_in_sink_awvzkn_0")
                        if scrub.states[object_states.OnTop].get_value(sink):
                            break
                    else:
                        if cond.evaluate():
                            break
                    og.sim.load_state(state)

    def _get_relevant_initial_condition(self, obj):
        for cond in self.env.task.activity_initial_conditions:
            try:
                if self.env.task.object_scope[cond.body[1]].unwrapped == obj:
                    # cond is a HEAD condition
                    # cond.children[0] is the actual binary predicate
                    return cond.children[0]
            except Exception as e:
                print(f"Error in _get_relevant_initial_condition: {e}")
        return None

    def _randomize_object_pose_D1(self, objs):
        for obj in objs:
            # if "table" not in obj.name:
            if all(keyword not in obj.name for keyword in ["table", "shelf", "bar", "sink"]):
                state = og.sim.dump_state()
                while True:
                    cond = self._get_relevant_initial_condition(obj)
                    assert cond is not None, f"Condition not found for object {obj.name}"
                    if cond.sample(True):
                        break
                    og.sim.load_state(state)
                    
                # for scrub, we need to ensure that the handle part is facing upwards
                if obj.name == "scrub_brush_601":
                    _, scrub_orn = obj.get_position_orientation()
                    orn_diff = th.from_numpy(np.array([np.pi, 0.0, 0.0]))
                    new_orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(scrub_orn))
                    obj.set_position_orientation(orientation=new_orn)
                    for _ in range(5): og.sim.step()

        # # remove later
        # pan = self.env.scene.object_registry("name", "frying_pan_602")
        # pan.set_position_orientation(th.tensor([5.2, -1.8, 0.908]), th.tensor([    -0.000,      0.000,     -0.499,      0.866]))
        # for _ in range(5): og.sim.step()
        
        # coffee_cup_7 = self.env.scene.object_registry("name", "coffee_cup_7")
        # y_range = np.random.uniform(-0.2, 0.2)
        # coffee_cup_7.set_position_orientation(position=th.tensor([ 1.523, -0.196 + y_range,  0.81]), orientation=th.tensor([     0.006,     -0.001,      0.997,     -0.079]))
        # print("coffee_cup pos: ", [ 1.573, -0.196 + y_range,  0.81])

    
    def _randomize_object_pose_D2(self, objs):
        success = False
        restart = False
        init_state = og.sim.dump_state()
        
        # We keep an outer loop because there is the following case: 
        # - randomizing task-relvant object works
        # - randomizing obstacle gets stuck and is unable to find a valid solution
        # So we give up, re-sample the task relevant objects and start again
        while True:
            print("success: ", success)
            if success:
                # D1 randomization for task relevant objects and D2 randomizatin for distractor objects successful
                print("D2 randomization success")
                return

            # D1 randomization for task relevant objects
            self._randomize_object_pose_D1(objs)
            
            # Sampling random object poses using custom thresholds
            if self.name.startswith("r1_pick_cup"):
                pos_max_nav = [-1.0, 1.0] 
                pos_min_nav = [-0.2, 0.2] 
                pos_manip = [-0.4, 0.4]
            elif self.name.startswith("r1_tidy_table"):
                pos_max_nav = [-0.7, 0.7] # To keep the nav object close to the kitchen island
                pos_min_nav = [-0.1, 0.1] 
                pos_manip = [-0.5, 0.5]
            elif self.name.startswith("r1_clean_pan"):
                pos_max_nav = [-0.7, 0.7] # To keep the nav object close to the kitchen island
                pos_min_nav = [-0.1, 0.1] 
                pos_manip = [-0.5, 0.5]
            elif self.name.startswith("r1_dishes_away"):
                pos_max_nav = [-0.7, 0.7] # To keep the nav object close to the kitchen island
                pos_min_nav = [-0.1, 0.1] 
                pos_manip = [-0.5, 0.5]
            
            for distractor_obj in self.distractor_objects:
                success = False
                state = og.sim.dump_state()
                associated_furniture_obj = self.env.scene.object_registry("name", distractor_obj["associated_furniture"])
                associated_task_obj = self.env.scene.object_registry("name", distractor_obj["associated_task_obj"])
                associated_task_obj_pos = associated_task_obj.get_position_orientation()[0]
                distractor_obj["obj"].states[object_states.OnTop].set_value(other=associated_furniture_obj, new_value=True)
                for _ in range(10): og.sim.step()
                sampled_pos, _ = distractor_obj["obj"].get_position_orientation()
                # if distractor_obj["obj"].name == "gift_box":
                #     breakpoint()
                start_time = time.time()
                while True:
                    
                    # If timeout occurs for randomizing any of the obstacle it's probably unlikely to succeed (I think this is an OG bug), 
                    # So we give up, re-sample the task relevant objects and start again
                    if time.time() - start_time > 10:
                        print("Timeout while sampling object poses for {}".format(distractor_obj["obj"].name))
                        restart = True
                        og.sim.load_state(state)
                        break
                    
                    close_by_pos = th.zeros(3)
                    if distractor_obj["obstacle_for"] == "navigation":
                        delta_x_pos = np.random.uniform(pos_max_nav[0], pos_min_nav[0]) if np.random.rand() < 0.5 else np.random.uniform(pos_min_nav[1], pos_max_nav[1])
                        delta_y_pos = np.random.uniform(pos_max_nav[0], pos_min_nav[0]) if np.random.rand() < 0.5 else np.random.uniform(pos_min_nav[1], pos_max_nav[1])
                    elif distractor_obj["obstacle_for"] == "manipulation":
                        delta_x_pos = np.random.uniform(pos_manip[0], pos_manip[1])
                        delta_y_pos = np.random.uniform(pos_manip[0], pos_manip[1])
                    close_by_pos[0] = associated_task_obj_pos[0] + delta_x_pos
                    close_by_pos[1] = associated_task_obj_pos[1] + delta_y_pos
                    close_by_pos[2] = sampled_pos[2] # Use the z position from the OG sampling
                    # print("close_by_pos", close_by_pos)

                    # sample random orientation along +z axis
                    sampled_orn_euler = np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])
                    sampled_orn = R.from_euler('xyz', sampled_orn_euler, degrees=False).as_quat()

                    close_by_pos_raised_z = close_by_pos.clone()
                    # lift up a bit so after 1 physics step, there is still no contact
                    close_by_pos_raised_z[2] += 0.03
                    distractor_obj["obj"].set_position_orientation(close_by_pos_raised_z, sampled_orn)
                    # this is faster than og.sim.step()
                    og.sim.step_physics()
                                    
                    no_contact = len(distractor_obj["obj"].states[object_states.ContactBodies].get_value()) == 0
                    # print("no_contact: ", no_contact)
                    
                    # # remove later
                    # for _ in range(50): og.sim.step()
                    
                    if not no_contact:
                        og.sim.load_state(state)
                        continue
                    for _ in range(5): og.sim.step()
                    # But after 5 env steps, there should be contacts, and OnTop should return True!
                    # print("OnTop: ", distractor_obj["obj"].states[object_states.OnTop].get_value(associated_furniture_obj))
                    if not distractor_obj["obj"].states[object_states.OnTop].get_value(associated_furniture_obj):
                        og.sim.load_state(state)
                        continue
                    
                    # distractor object sampled correctly
                    print("D2 randomization success for {}".format(distractor_obj["obj"].name))
                    success = True
                    break
            
                # This means we want to try resampling the task-relevant object (using D1) and then re-sampling the distractor object
                if restart:
                    og.sim.load_state(init_state)
                    for _ in range(5): og.sim.step()
                    restart = False
                    break

    def check_object_upright(self, obj):
        q = obj.get_position_orientation()[1]
        r = R.from_quat(q)

        # Rotate the up vector
        up_rotated = r.apply([0, 0, 1])
        z_alignment = up_rotated[2]  # should be close to 1 if not toppled

        threshold = 0.995  # cos(small angle) ~1
        upright = z_alignment > threshold
        
        return upright

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        obs, info = self.env.reset()
        self.global_env_step = 0
        if not self.policy_rollout:
            self.valid_env = True
            if hasattr(self, "primitive"):
                # self.primitive.valid_env = True
                self.primitive.mp_err = "None"
            self.err = "None"
            self.obj_visible_at_start_of_manip = False
            self.execution_phase_ind = 0
            self.phases_completed_wo_mp_err = 0

        if self.debug_from_saved_state:
            import pickle
            debug_state_path = os.environ.get("ELOGGEN_DEBUG_STATE_PATH")
            if not debug_state_path:
                raise ValueError("ELOGGEN_DEBUG_STATE_PATH must be set for saved-state debugging")
            with open(debug_state_path, "rb") as debug_state_file:
                state = pickle.load(debug_state_file)
            og.sim.load_state(state)

            for _ in range(5): og.sim.step()
            # fridge = self.env.scene.object_registry("name", "fridge_dszchb_0")
            # self.env.scene.remove_object(obj=fridge)
            # for _ in range(5): og.sim.step()

            self.cmg.update_obstacles()
        else:
            # Reset the robot to a specific pose (Note that this is different from the spawned pose because curobo requires robot to be spawned at origin)
            self.env.robots[0].set_position_orientation(position=th.tensor(self.reset_base_pose[0]), orientation=th.tensor(self.reset_base_pose[1]))

        # for static manipulation only
        if self.manipulation_only:
            if self.real_robot_mode:
                init_joint_pos = th.tensor([
                    0.5, -0.430, 0.004, 0.007, 0.007, 0.259, # Base
                    1.3, -2.3, -1.2, 0.0, # Torso
                    0.0, 0.0, 1.894, 1.894, -0.985, -0.985, 1.561, 1.562, 0.910, 0.910, -1.554, -1.554, # Arms
                    0.050, 0.050, 0.050, 0.050]) # Grippers
            else:
                init_joint_pos = th.tensor([
                    0.332, -0.430, 0.004, 0.007,  0.007, 0.259, # Base
                    1.427, -1.658, -0.543, 0.051, # Torso
                    -0.000, -0.000, 1.894, 1.894, -0.985, -0.985, 1.561, 1.562, 0.910, 0.910, -1.554, -1.554, # Arms
                    0.050, 0.050, 0.050, 0.050]) # Grippers
            self.robot.set_joint_positions(init_joint_pos)
            # self.env.robots[0].set_position_orientation(position=th.tensor([0.332, -0.430, 0.0]))
        # else:
        #     self.env.robots[0].set_position_orientation(position=th.tensor([-0.863, -0.26, 0.0]))

        if self.name.startswith(OPENARM_CASHIER_STYLE_PREFIXES):
            if not self._load_openarm_cashier_source_initial_state():
                self._align_openarm_cashier_robot_to_source_initial_action()
            self._align_openarm_cashier_objects_to_source_initial_poses()
            self._resize_openarm_bag_grocery_bags_for_generation()
            self._apply_viewer_lighting()

        # If loading a saved state, don't do randomization for all objects. Choose according to what you want
        if self.debug_from_saved_state:
            pass

        # Drawer storage uses bounded task-specific distributions. The generic
        # D1 / D2 samplers move objects over arbitrary supports and add
        # distractors, which changes the task instead of its initialization.
        elif self.name.startswith("openarm_drawer_storage"):
            self._randomize_openarm_drawer_storage()
            og.sim.step()
            for _ in range(5):
                og.sim.render()
            obs, info = self.env.get_obs()

        elif self.name.startswith("openarm_fruit_basket_bagging"):
            self._randomize_openarm_fruit_basket_bagging()
            og.sim.step()
            for _ in range(5):
                og.sim.render()
            obs, info = self.env.get_obs()

        # D0 is the distribution with randomization in xy-pos and z-rot
        elif self.name.endswith("D0"):
            task_relevant_objs = self._get_task_relevant_objs()
            self._randomize_object_pose_D0(task_relevant_objs)

            # Step one time to update the scene and render a few times as well
            og.sim.step()
            for _ in range(5):
                og.sim.render()

            # Update the observation
            obs, info = self.env.get_obs()

        # D1 is randomization all over the furniture
        elif self.name.endswith("D1"):
            task_relevant_objs = self._get_task_relevant_objs()
            self._randomize_object_pose_D1(task_relevant_objs)

            # Step one time to update the scene and render a few times as well
            og.sim.step()
            for _ in range(5):
                og.sim.render()

            # Update the observation
            obs, info = self.env.get_obs()

        # D2 has ranomization with obstacles
        elif self.name.endswith("D2"):
            task_relevant_objs = self._get_task_relevant_objs()
            init_state = og.sim.dump_state()
            start_time = time.time()
            
            retry = False
            while True:
                self._randomize_object_pose_D2(task_relevant_objs)

                # loop a few sim steps to make sure the objects are in a stable state
                for _ in range(30): og.sim.step()

                # Some objects have suboptimal COM (like pot_plan), we want to make sure they're upright
                all_object_names = [obj.name for obj in self.env.scene.objects]
                for obj_name in all_object_names:
                    if obj_name in self.check_upright:
                        obj = self.env.scene.object_registry("name", obj_name)
                        upright = self.check_object_upright(obj)
                        print("object, upright: ", obj.name, upright)
                        if not upright:
                            print(f"Object {obj.name} is not upright, randomizing again")
                            retry = True
                            break
                
                # # remove later
                # retry = True
                
                if retry:
                    retry = False
                    # breakpoint()
                    og.sim.load_state(init_state)
                    for _ in range(5): og.sim.step()
                    # breakpoint()
                    continue
                else:
                    break

            print("D2 Randomization time: ", time.time() - start_time)
        else:
            raise ValueError(f"Unknown environment name: {self.name}")

        self._apply_viewer_lighting()

        if self.policy_rollout:
            # customize the viewer camera for policy rollout
            ext_sensor = self.env._external_sensors['external_sensor2']
            ext_sensor.set_position_orientation(position=th.tensor([1.9230, -0.2432,  1.4854]), orientation=th.tensor([0.3403, 0.3626, 0.6326, 0.5937]),)
        
        for _ in range(50): og.sim.step()

        self._set_viewer_camera_pose(
            default_position=[7.040, -1.375, 2.365],
            default_orientation=[0.382, 0.188, 0.400, 0.812],
        )
                
        # change to the new observation
        obs, obs_info = self.get_obs_IL()

        return obs

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        # There is probably a bug in og.sim.load_state where sometimes the state is not loaded correctly.
        # Empirically, I see that this happens when the robot is in a collision state with the object that is not reset correctly
        # but I might be wrong. For some reason this fix works.
        table_obj = self.env.scene.object_registry("name", "breakfast_table") 
        if table_obj is None:
            table_obj = self.env.scene.object_registry("name", "breakfast_table_6")
        if table_obj is not None:
            table_obj.set_position_orientation(position=th.tensor([0.0, -2.0, 0.7]))
            for _ in range(20): og.sim.step()

        og.sim.load_state(th.from_numpy(state["states"]).to(th.float32), serialized=True)
        self._apply_viewer_lighting()

        for _ in range(20): og.sim.step()

        self._set_viewer_camera_pose(
            default_position=[-3.0856, 0.1110, 3.4114],
            default_orientation=[-0.3543, 0.3566, 0.6132, -0.6093],
        )

        return self.get_obs_IL()
        # return self.env.get_obs()[0]

    def _set_viewer_camera_pose(self, default_position, default_orientation):
        """
        Set the interactive viewer camera. This affects the GUI / debug video view,
        not the image observations saved from task cameras.
        """
        if isinstance(self.env.robots[0], OpenArmBimanual):
            if self.name.startswith(OPENARM_CASHIER_STYLE_PREFIXES):
                self._set_cashier_viewer_camera_pose()
            else:
                og.sim.viewer_camera.set_position_orientation(
                    position=th.tensor([1.45, -1.55, 2.45]),
                    orientation=th.tensor([0.26, 0.08, 0.17, 0.95]),
                )
        else:
            og.sim.viewer_camera.set_position_orientation(
                position=th.tensor(default_position),
                orientation=th.tensor(default_orientation),
            )

        if og.sim.viewer_camera.active_camera_path != og.sim.viewer_camera.prim_path:
            og.sim.viewer_camera.active_camera_path = og.sim.viewer_camera.prim_path

    def _set_cashier_viewer_camera_pose(self):
        self._update_openarm_external_camera_poses()
        if uses_real_exp_1_viewer_camera(self.name):
            set_real_exp_1_viewer_camera(og, print_prefix="[viewer_camera][openarm_real_exp_1]")
            return

        camera_pos = th.tensor([0.84451, 4.1226, 2.57063], dtype=th.float32)
        # Omni's GUI displays the transform rotation as X/Y/Z fields, but the
        # equivalent quaternion matches scipy's zyx convention with [Z, Y, X].
        camera_quat = th.tensor(
            R.from_euler("zyx", [80.176, 50.592, 7.621], degrees=True).as_quat(),
            dtype=th.float32,
        )
        og.sim.viewer_camera.set_position_orientation(
            position=camera_pos,
            orientation=camera_quat,
        )
        if og.sim.viewer_camera.active_camera_path != og.sim.viewer_camera.prim_path:
            og.sim.viewer_camera.active_camera_path = og.sim.viewer_camera.prim_path

    def _get_openarm_viewer_focus(self, robot_pos, robot_quat):
        # Use the robot root plus a fixed height so the viewer does not drift as
        # the two arms move. This gives a repeatable overview centered on OpenArm.
        focus_z = float(os.environ.get("OPENARM_CASHIER_VIEWER_FOCUS_Z", "1.15"))
        return robot_pos + th.tensor([0.0, 0.0, focus_z], dtype=th.float32, device=robot_pos.device)

    def _get_cashier_viewer_focus(self):
        focus_points = []
        if self.name.startswith("openarm_bag_groceries"):
            obj_names = ("apple_1", "candy_1", "orange_1", "canned_food_1")
        elif self.name.startswith("openarm_drawer_storage"):
            obj_names = ("object_1", "object_2", "drawer_cabinet_1")
        elif self.name.startswith("openarm_real_exp_1"):
            obj_names = ("object_1", "object_2")
        else:
            obj_names = ("apple_1", "candy_1")
        for obj_name in obj_names:
            obj = self.env.scene.object_registry("name", obj_name)
            if obj is not None:
                obj_pos, _ = obj.get_position_orientation()
                focus_points.append(th.as_tensor(obj_pos, dtype=th.float32))

        if focus_points:
            return th.stack(focus_points).mean(dim=0)

        return th.tensor([-1.0, 3.60, 0.95], dtype=th.float32)

    # TODO: implement the case of "rgb_array" mode correctly, e.g. return the rendered image as a numpy array
    def render(self, mode="human", height=None, width=None, camera_name="agentview"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        if mode == "human":
            og.sim.render()
        else:
            # return np.zeros((height if height else 128, width if width else 128, 3), dtype=np.uint8)
            robot_name = self.env.robots[0].name
            obs, info = self.env.get_obs()
            if isinstance(self.robot, OpenArmBimanual) and self.name.startswith(OPENARM_CASHIER_STYLE_PREFIXES):
                self._set_cashier_viewer_camera_pose()
            viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'].numpy()[:, :, :3]
            ego_key = f"{robot_name}::{robot_name}:eyes:Camera:0::rgb"
            if ego_key in obs:
                ego_img = obs[ego_key].numpy()[:, :, :3]
                return hori_concatenate_image([ego_img, viewer_img])
            else:
                return viewer_img
        
    def customize_physical_properties(self):
        """
        Setup the mass, friction specifically for each task
        """
        # # Change the color of the robot to be black.
        # if isinstance(self.robot, R1):
        #     for material in self.robot.materials:
        #         material.diffuse_color_constant = th.tensor([0.0, 0.0, 0.0])

        # Increase gripper friction
        state = og.sim.dump_state()
        og.sim.stop()
        target_friction = 4.0
        gripper_mat = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
            prim_path=f"{self.env.robots[0].prim_path}/gripper_mat",
            name="gripper_material",
            static_friction=target_friction,
            dynamic_friction=target_friction,
            restitution=None,
        )
        for links in self.env.robots[0].finger_links.values():
            for link in links:
                for msh in link.collision_meshes.values():
                    msh.apply_physics_material(gripper_mat)
        og.sim.play()
        og.sim.load_state(state)

        # Any other task specific customizations for the physical properties should be added here        
        if self.name.startswith("r1_pick_cup"):
            pass
        
        elif self.name.startswith("r1_dishes_away"):
            pass

        elif self.name.startswith("r1_tidy_table"):
            pass
            
        elif self.name.startswith("r1_clean_pan"):
            pass

        elif self.name.startswith("r1_bringing_water"):
            pass

    def _configure_openarm_task_physics(self):
        if self.name.startswith("openarm_fruit_basket_bagging"):
            # Keep normal PhysX contacts for fruit, containers, and the table.
            # CuRobo-only collision relaxation is handled at planning time in
            # waypoint.py, so the simulated contact model stays faithful.
            for object_name in (
                "object_1",
                "object_2",
                "object_3",
                "object_4",
                "paper_bag_1",
                "basket_1",
            ):
                obj = self.env.scene.object_registry("name", object_name)
                obj.solver_position_iteration_count = 32
                obj.solver_velocity_iteration_count = 8
            return

        if not self.name.startswith("openarm_drawer_storage"):
            return
        cabinet = self.env.scene.object_registry("name", "drawer_cabinet_1")
        support = self.env.scene.object_registry("name", "checkout_counter_umnoqh_0")
        if cabinet is None or support is None:
            raise KeyError("openarm_drawer_storage requires drawer_cabinet_1 and checkout_counter_umnoqh_0")

        lower_drawer_link = cabinet.links["link_1"]
        for support_link in support.links.values():
            lower_drawer_link.add_filtered_collision_pair(prim=support_link)
        for cabinet_link_name in ("base_link", "link_2"):
            lower_drawer_link.add_filtered_collision_pair(prim=cabinet.links[cabinet_link_name])

        for object_name in ("drawer_cabinet_1", "object_1", "object_2"):
            obj = self.env.scene.object_registry("name", object_name)
            obj.solver_position_iteration_count = 32
            obj.solver_velocity_iteration_count = 8
        
    def sensor_setup(self):
        """
        Setup the sensor position, orientation of the environment
        """
        # Sensors used for visualization (saving videos)
        # TODO: setup other external sensors as well in case we are using it
        if self.env._external_sensors is not None and "external_sensor2" in self.env._external_sensors:
            ext_sensor2 = self.env._external_sensors["external_sensor2"]
            ext_sensor2.add_modality("rgb")
            ext_sensor2.image_height = 720
            ext_sensor2.image_width = 720

        # Robot specific sensors, these are stored in hdf5 file by datagen and used for policy learning
        all_sensor_info = {}
        for sensor_name, sensor in self.env.robots[0].sensors.items():
            # sensor = self.env.robots[0].sensors[f"{self.robot_name}:eyes:Camera:0"]
            # TODO: These are used in normalization of the point cloud, take a look at these values again!
            self.pcd_offset = np.array([0.0, 0.0, 0.0])
            self.pcd_norm_range = np.array([1.0, 1.0, 1.0])
            self.clip_bbox_size = np.array([10, 10, 10])
            self.world_to_cam_tf = np.eye(4)
            self.sensor_max_depth = 2.0
            self.number_ponits_to_sample = 4096

            # self.pcd_offset = np.array([ -4.116, 0.002,  -3.069])
            # self.pcd_norm_range = np.array([0.9, 0.9, 0.9])
            # self.clip_bbox_size = np.array([3, 1.5, 2])

            # # TODO: maybe reduce the pcd range can be helpful
            # # self.pcd_norm_range = np.array([1.0, 1.0, 1.0])
            # # self.clip_bbox_size = np.array([2.5, 1.5, 1])
            
            # # change the viewport output to the viewer camera
            # viewer_prim_path = og.sim.viewer_camera.prim_path
            # og.sim.viewer_camera.active_camera_path = viewer_prim_path #'/World/viewer_camera'

            # # change external sensor 0 pose and resolution
            # ext_sensor = self.env._external_sensors['external_sensor0']
            # ext_sensor.set_position_orientation(
            #     position=th.tensor([ 1.7330, -0.0486,  1.5626]),
            #     orientation=th.tensor([0.3689, 0.3718, 0.6047, 0.5999]),
            # )
            # ext_sensor.add_modality("depth_linear")

            sensor_info = {
                "K": sensor.intrinsic_matrix,
                "world_to_cam_tf": self.world_to_cam_tf,
                "image_height": sensor.image_height,
                "image_width": sensor.image_width,
                'sensor_max_depth': self.sensor_max_depth,
                'number_points_to_sample': self.number_ponits_to_sample,
                'pcd_offset': self.pcd_offset,
                'pcd_norm_range': self.pcd_norm_range,
                'clip_bbox_size': self.clip_bbox_size,
            }
            all_sensor_info[sensor_name] = sensor_info

        return all_sensor_info

    def _update_openarm_external_camera_poses(self):
        """
        Keep temporary external cameras attached to OpenArm links.

        External sensors are scene sensors, so a scene-relative prim path that
        looks like it is under a robot link is not enough to guarantee that it
        follows robot articulation. For the temporary camera setup, refresh the
        world pose from the current link pose before collecting observations.
        """
        if not isinstance(self.robot, OpenArmBimanual):
            return
        if self.env._external_sensors is None:
            return

        default_mounts = {
            "left_wrist_cam": {
                "link": "openarm_left_link7",
                "pos": [0.024684, 0.02621, 0.092541],
                "quat": [0.831856, 0.498769, -0.02012, -0.242571],
            },
            "right_wrist_cam": {
                "link": "openarm_right_link7",
                "pos": [0.03681, 0.014538, 0.046899],
                "quat": [0.651713, 0.737733, -0.027466, -0.17397],
            },
            "base_cam": {
                "link": "openarm_body_link0",
                "pos": [0.065302, 0.005802, 0.707584],
                "quat": [-0.293232, -0.293232, 0.64344, 0.64344],
            },
        }
        mounts_file = next(
            (path for path in _openarm_camera_mount_file_candidates(self.name) if os.path.exists(path)),
            None,
        )
        canonical_task = canonical_openarm_task_name(self.name)
        if canonical_task in PUBLISHED_OPENARM_TASKS and mounts_file is None:
            expected = _openarm_camera_mount_file_candidates(self.name)[0]
            raise FileNotFoundError(
                f"Camera mounts for published task {canonical_task!r} are missing: {expected}"
            )
        if mounts_file is not None and os.path.exists(mounts_file):
            with open(mounts_file, "r") as f:
                loaded_mounts = json.load(f)
            for camera_name, mount in loaded_mounts.items():
                default_mounts.setdefault(camera_name, {})
                default_mounts[camera_name].update(mount)

        for camera_name, mount in default_mounts.items():
            sensor = self.env._external_sensors.get(camera_name)
            link = self.robot.links.get(mount["link"])
            if sensor is None or link is None:
                continue
            link_pos, link_quat = link.get_position_orientation()
            local_pos = th.tensor(mount["pos"], dtype=th.float32)
            local_quat = th.tensor(mount["quat"], dtype=th.float32)
            camera_pos, camera_quat = T.pose_transform(link_pos, link_quat, local_pos, local_quat)
            sensor.set_position_orientation(position=camera_pos, orientation=camera_quat, frame="world")
    
        # the following can be deleted in the future
        # elif self.name.startswith("test_tiago_cup"):

        #     self.K = np.array([
        #         [259.6039,   0.0000, 160.0000],
        #         [  0.0000, 280.2977,  90.0000],
        #         [  0.0000,   0.0000,   1.0000]
        #         ])
        #     self.camera_position = th.tensor([ 1.0304, -0.0309,  1.0272])
        #     self.camera_quat= th.tensor([0.2690, 0.2659, 0.6509, 0.6583])
        #     self.world_to_cam_tf = T.pose2mat((self.camera_position, self.camera_quat)).numpy()
        #     self.sensor_max_depth = 2.0
        #     self.number_ponits_to_sample = 2048

    def depth_to_pcd(
            self,
            depth,
            pose,
            base_link_pose,
            K,
            max_depth=2,
        ):

        # get the homogeneous transformation matrix from quaternion
        pos = pose[:3]
        quat = pose[3:]
        rot = R.from_quat(quat)  # scipy expects [x, y, z, w]
        rot_add = R.from_euler('x', np.pi).as_matrix() # handle the cam_to_img transformation
        rot_matrix = rot.as_matrix() @ rot_add   # 3x3 rotation matrix
        world_to_cam_tf = np.eye(4)
        world_to_cam_tf[:3, :3] = rot_matrix
        world_to_cam_tf[:3, 3] = pos

        # filter depth
        mask = depth > max_depth
        depth[mask] = 0
        h, w = depth.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij", sparse=False)
        assert depth.min() >= 0
        u = x
        v = y
        uv = np.dstack((u, v, np.ones_like(u))) # (img_width, img_height, 3)

        Kinv = np.linalg.inv(K)
        
        pc = depth.reshape(-1, 1) * (uv.reshape(-1, 3) @ Kinv.T)
        pc = pc.reshape(h, w, 3)
        pc = np.concatenate([pc.reshape(-1, 3), np.ones((h * w, 1))], axis=-1)  # shape (H*W, 4)

        world_to_robot_tf = T.pose2mat((th.from_numpy(base_link_pose[:3]), th.from_numpy(base_link_pose[3:]))).numpy()
        robot_to_world_tf = np.linalg.inv(world_to_robot_tf)
        pc = (pc @ world_to_cam_tf.T @ robot_to_world_tf.T)[:, :3].reshape(h, w, 3)

        return pc

    def process_fused_point_cloud(self, obs):
 
        base_link_pose = obs['base_link_pose'] # (7,)

        # TODO: now assuming the camera intrinsic matrix are the same for all the cameras!!

        eye_rgb = obs['robot_r1::robot_r1:eyes:Camera:0::rgb'][...,:3] # (resolution 0, resolution 1, 3)
        eye_depth = obs['robot_r1::robot_r1:eyes:Camera:0::depth_linear'] # (resolution 0, resolution 1, 1)
        eye_pose = obs['robot_r1:eyes:Camera:0_pose'] # (7,)
        eye_cam_pcd = self.depth_to_pcd(eye_depth, eye_pose, base_link_pose, self.intrinsic_matrix, max_depth=self.sensor_max_depth)
        eye_cam_rgbd = np.concatenate([eye_rgb/255.0, eye_cam_pcd], axis=-1).reshape(-1,6)

        left_cam_rgb = obs['robot_r1::robot_r1:left_eef_link:Camera:0::rgb'][...,:3]
        left_cam_depth = obs['robot_r1::robot_r1:left_eef_link:Camera:0::depth_linear']
        left_cam_pose = obs['robot_r1:left_eef_link:Camera:0_pose']
        left_cam_pcd = self.depth_to_pcd(left_cam_depth, left_cam_pose, base_link_pose, self.intrinsic_matrix, max_depth=self.sensor_max_depth)
        left_cam_rgbd = np.concatenate([left_cam_rgb/255.0, left_cam_pcd], axis=-1).reshape(-1,6)

        right_cam_rgb = obs['robot_r1::robot_r1:right_eef_link:Camera:0::rgb'][...,:3]
        right_cam_depth = obs['robot_r1::robot_r1:right_eef_link:Camera:0::depth_linear']
        right_cam_pose = obs['robot_r1:right_eef_link:Camera:0_pose']
        right_cam_pcd = self.depth_to_pcd(right_cam_depth, right_cam_pose, base_link_pose, self.intrinsic_matrix, max_depth=self.sensor_max_depth)
        right_cam_rgbd = np.concatenate([right_cam_rgb/255.0, right_cam_pcd], axis=-1).reshape(-1,6)
   
        color_pcd = np.concatenate([eye_cam_rgbd, left_cam_rgbd, right_cam_rgbd], axis=0)

        # clip point cloud with a bounding box
        mask = (color_pcd[:, 3] > self.x_range[0]) & (color_pcd[:, 3] < self.x_range[1]) & (
            color_pcd[:, 4] > self.y_range[0]) & (color_pcd[:, 4] < self.y_range[1]) & (
            color_pcd[:, 5] > self.z_range[0]) & (color_pcd[:, 5] < self.z_range[1])
        color_pcd = color_pcd[mask]
        
        # split the pcd based on table and not table and then do down sample differently
        table_mask = color_pcd[:, 5] < self.table_mask_height
        table_pcd = color_pcd[table_mask]
        not_table_pcd = color_pcd[~table_mask]
        table_ratio = 0.3
        table_samples = int(self.number_ponits_to_sample * table_ratio)
        not_table_samples = self.number_ponits_to_sample - table_samples

        if table_pcd.shape[0] < table_samples:
            raise RuntimeError(
                f"Not enough table points for downsampling: {table_pcd.shape[0]} < {table_samples}"
            )
        table_pcd_ds = fps_downsample(table_pcd, table_samples)
        not_table_pcd_ds = fps_downsample(not_table_pcd, not_table_samples)
        color_pcd = np.concatenate([table_pcd_ds, not_table_pcd_ds], axis=0)
         
        return color_pcd

    def process_point_cloud(self, obs):
        """
        Get point cloud from the environment
        """
        # compute_pcd_time = time.time()
        # breakpoint()

        if self.name.startswith("r1_pick_cup"):
            pointcloud = self.process_fused_point_cloud(obs)

        elif self.name.startswith("test_r1_cup"):
            for key in obs.keys():
                if 'eyes:Camera:0::depth_linear' in key:
                    # print('depth key', key)
                    depth = obs[key]
                elif 'eyes:Camera:0::rgb' in key:
                    # print('rgb key', key)
                    rgb = obs[key]
            rgbd = np.concatenate([rgb, depth[:,:,None]], axis=-1)
            pointcloud = compute_point_cloud_from_rgbd(
                rgbd=rgbd, 
                K=self.intrinsic_matrix, 
                pcd_offset=self.pcd_offset,
                pcd_norm_range=self.pcd_norm_range,
                clip_bbox_size=self.clip_bbox_size,
                cam_to_img_tf=None, 
                world_to_cam_tf=self.world_to_cam_tf, 
                pcd_step_vis=False, 
                max_depth=self.sensor_max_depth,
                sample_type='fps',
                num_points_to_sample=self.number_ponits_to_sample,
                clip_scene=True,
                with_color=self.with_color
                )
        
        return pointcloud
    
    def process_prop(self, obs):
        # base_qpos = obs['base_qpos'] #  3
        if isinstance(self.robot, OpenArmBimanual):
            base_qvel = np.zeros(3, dtype=np.float32)  # fixed-base: no base velocity
            trunk_qpos = np.zeros(4, dtype=np.float32)  # no trunk
        else:
            base_qvel = obs['base_qvel'] # 3
            trunk_qpos = obs['trunk_qpos'] # 4
        arm_left_qpos = obs['arm_left_qpos'] #  6
        arm_right_qpos = obs['arm_right_qpos'] #  6
        left_gripper_width = obs['gripper_left_qpos'].sum()[None] # 1
        right_gripper_width = obs['gripper_right_qpos'].sum()[None] # 1
        prop_state = np.concatenate((base_qvel, trunk_qpos, arm_left_qpos, arm_right_qpos, left_gripper_width, right_gripper_width)) # 21
        if isinstance(self.robot, R1): assert prop_state.shape[0] == 21
        return prop_state

    def process_eef(self, obs):
        eef_left_pos = obs['eef_left_pos'] # 3
        eef_right_pos = obs['eef_right_pos'] # 3
        eef_left_quat = obs['eef_left_quat'] # 4
        eef_right_quat = obs['eef_right_quat'] # 4
        eef_state = np.concatenate((eef_left_pos, eef_right_pos, eef_left_quat, eef_right_quat)) # 14
        if isinstance(self.robot, R1): assert eef_state.shape[0] == 14 # for r1 robot
        return eef_state
    
    def process_prop_eef(self, obs):
        # base_qpos = obs['base_qpos'] #  3
        if isinstance(self.robot, OpenArmBimanual):
            base_qvel = np.zeros(3, dtype=np.float32)
            trunk_qpos = np.zeros(4, dtype=np.float32)
        else:
            base_qvel = obs['base_qvel'] # 3
            trunk_qpos = obs['trunk_qpos'] # 4
        arm_left_qpos = obs['arm_left_qpos'] #  6
        eef_left_pos = obs['eef_left_pos'] # 3
        eef_left_quat = obs['eef_left_quat'] # 4
        left_gripper_width = obs['gripper_left_qpos'].sum()[None] # 1
        arm_right_qpos = obs['arm_right_qpos'] #  6
        eef_right_pos = obs['eef_right_pos'] # 3
        eef_right_quat = obs['eef_right_quat'] # 4
        right_gripper_width = obs['gripper_right_qpos'].sum()[None] # 1

        prop_eef_state = np.concatenate((base_qvel, trunk_qpos, 
                                     arm_left_qpos, eef_left_pos, eef_left_quat, left_gripper_width, 
                                     arm_right_qpos, eef_right_pos, eef_right_quat, right_gripper_width)) # 35
        if isinstance(self.robot, R1): assert prop_eef_state.shape[0] == 35 # for r1 robot
        return prop_eef_state

    def process_prop_eef_basepose(self, obs):
        if isinstance(self.robot, OpenArmBimanual):
            base_qpos = np.zeros(3, dtype=np.float32)
            base_qvel = np.zeros(3, dtype=np.float32)
            trunk_qpos = np.zeros(4, dtype=np.float32)
        else:
            base_qpos = obs['base_qpos'] #  3
            base_qvel = obs['base_qvel'] # 3
            trunk_qpos = obs['trunk_qpos'] # 4
        arm_left_qpos = obs['arm_left_qpos'] #  6
        eef_left_pos = obs['eef_left_pos'] # 3
        eef_left_quat = obs['eef_left_quat'] # 4
        left_gripper_width = obs['gripper_left_qpos'].sum()[None] # 1
        arm_right_qpos = obs['arm_right_qpos'] #  6
        eef_right_pos = obs['eef_right_pos'] # 3
        eef_right_quat = obs['eef_right_quat'] # 4
        right_gripper_width = obs['gripper_right_qpos'].sum()[None] # 1

        prop_eef_basepose_state = np.concatenate((base_qpos, base_qvel, trunk_qpos, 
                                     arm_left_qpos, eef_left_pos, eef_left_quat, left_gripper_width, 
                                     arm_right_qpos, eef_right_pos, eef_right_quat, right_gripper_width)) # 38
        if isinstance(self.robot, R1): assert prop_eef_basepose_state.shape[0] == 38 # for r1 robot
        return prop_eef_basepose_state

    def process_base_vel_robot_frame(self, robot_prop_states):
        base_vel = copy.deepcopy(robot_prop_states['base_qvel'])
        base_vel_xy = base_vel[:2]
        base_vel_z = base_vel[2] # rotation along z-axis should not be changed
        base_vel_vec = th.cat([base_vel_xy, th.zeros(1)]) # 3
        base_vel_ori = th.Tensor([0, 0, 0, 1]) # 4
        base_link_pose = self.env.robots[0].get_position_orientation()
        # TODO: construct the frame attached to the base velocity 
        base_vel_vec_local, base_vel_ori_local = T.relative_pose_transform(base_vel_vec + base_link_pose[0], base_vel_ori, *base_link_pose)
        base_vel_local = th.cat([base_vel_vec_local[:2], th.Tensor([base_vel_z])])
        robot_prop_states['base_vel'] = base_vel_local
        return robot_prop_states

    def process_obj_robot_frame(self):
        # process object states, tranform them into robot fixed frames

        base_link_pose = self.env.robots[0].get_position_orientation()
        if self.name.startswith("test_r1_cup"):
            # TODO: only work for test r1 cup, which is a dummy task
            obj_states = {}
            obj_list = []
            obj_list.append(self.env.scene.object_registry("name", "coffee_cup"))
            obj_list.append(self.env.scene.object_registry("name", "teacup"))
            # obj_list = [self.env.scene.object_registry("name", name) for name in ["coffee_cup", "teacup"]]
            for obj in obj_list:
                obj_name = "object::"+obj.name
                pos, ori = obj.get_position_orientation()
                local_pos, local_ori = T.relative_pose_transform(pos, ori, *base_link_pose)
                obj_states[obj_name] = np.concatenate([local_pos, local_ori])
        
        else:
            # get object states for tasks that are not dummy tasks
            obj_states = {}
            obj_bddl_names = [obj.bddl_inst for obj in self.env._task.object_scope.values()] # get object names
            for obj_name in obj_bddl_names:
                if isinstance(self.env.task.object_scope[obj_name].unwrapped, BaseSystem):
                    continue
                # TODO: here not checking whether the object exist in the scene, may need to handle this silimar to omnigibson/tasks/behavior_task.py
                pos, ori = self.env.task.object_scope[obj_name].get_position_orientation()
                local_pos, local_ori = T.relative_pose_transform(pos, ori, *base_link_pose)
                if 'agent' not in obj_name and 'robot' not in obj_name:
                    # remove the .n.01_1 suffix and only keep the object name
                    obj_name = "object::"+obj_name.split('.')[0]
                obj_states[obj_name] = np.concatenate([local_pos, local_ori])
        
        return obj_states

    def get_obs_IL(self, di=None):
        """
        Get observation for IL baselines
         - robot proprioceptive state
         - objects in the scene and their states
         - default observations
        """

        # customize observation for IL baselines
        obs_IL = {}

        obj_states = self.process_obj_robot_frame()
        obs_IL.update(obj_states)

        # temp_start_time = time.time()
        other_obs, info = self.get_observation(di) # get default observations

        # retain only the relevant obs keys for IL policy
        for k in other_obs.keys():
            if k.split("::")[-1] in self.IL_obs_keys:
                obs_value = other_obs[k]
                if "seg" in k:
                    obs_IL[k] = obs_value.cpu() if hasattr(obs_value, "cpu") else obs_value
                    # breakpoint()
                else:
                    obs_IL[k] = obs_value
        # obs_IL.update(other_obs)
        # obs_time = time.time() - temp_start_time 

        # add robot sensor poses
        for k in self.robot.sensors:
            sensor_pose = self.robot.sensors[k].get_position_orientation()
            obs_IL.update({f"{k}_pose": np.concatenate([sensor_pose[0], sensor_pose[1]])})

        
        # temp_start_time = time.time()
        robot_prop_states = self.env.robots[0]._get_proprioception_dict()
        # TODO: need to add the base velocity in the robot frame
        # robot_prop_states = self.process_base_vel_robot_frame(robot_prop_states)
        # print('check base vel')
        # breakpoint()

        obs_IL.update(robot_prop_states)
        # print("Time taken for getting obs and proprio: {:.2f} and {:.2f} seconds".format(obs_time, time.time() - temp_start_time))

        base_link_pose = self.env.robots[0].get_position_orientation()
        obs_IL.update({'base_link_pose': np.concatenate([base_link_pose[0], base_link_pose[1]])})

        prop_state = {'prop_state': self.process_prop(robot_prop_states)}
        obs_IL.update(prop_state)

        prop_eef_state = {'prop_eef_state': self.process_prop_eef(robot_prop_states)}
        obs_IL.update(prop_eef_state)

        prop_eef_basepose = {'prop_eef_basepose': self.process_prop_eef_basepose(robot_prop_states)}
        obs_IL.update(prop_eef_basepose)

        # eef_state = {'eef_state': self.process_eef(robot_prop_states)}
        # obs_IL.update(eef_state)

        base_link_pose = self.env.robots[0].get_position_orientation()
        obs_IL.update({'base_link_pose': np.concatenate([base_link_pose[0], base_link_pose[1]])})

        eyes_pose = self.robot.links["eyes"].get_position_orientation() if "eyes" in self.robot.links else (np.zeros(3), np.array([0., 0., 0., 1.]))
        obs_IL.update({'eyes_pose': np.concatenate([eyes_pose[0], eyes_pose[1]])})

        if self.policy_rollout:
            pcd = self.process_point_cloud(obs_IL)
            if self.with_color:
                obs_IL['combined::color_point_cloud'] = pcd
            else:
                obs_IL['combined::point_cloud'] = pcd
        
        return obs_IL, info

    def get_observation(self, di=None):
        if di:
            return di

        self._update_openarm_external_camera_poses()
        obs, info = self.env.get_obs()
        return obs, info

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        state = og.sim.dump_state(serialized=True)
        return dict(states=state)

    # def is_success(self):
    #     """
    #     Check if the task condition(s) is reached. Should return a dictionary
    #     { str: bool } with at least a "task" key for the overall task success,
    #     and additional optional keys corresponding to other task criteria.
    #     """
    #     return {"task": len(self.env.task._termination_conditions["predicate"].goal_status["unsatisfied"]) == 0}
    
    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        unsuccess_bddl = len(self.env.task._termination_conditions["predicate"].goal_status["unsatisfied"])
        success_bddl = unsuccess_bddl == 0
        result = {"task": success_bddl}

        # Additional success criteria
        if self.name.startswith("r1_pick_cup"):
            # # NOTE: Currently only using the final state to determine success. Verify satisfactory for all tasks.
            # teacup_obj = self.env.scene.object_registry("name", "teacup")
            coffee_cup_obj = self.env.scene.object_registry("name", "coffee_cup_7")
            # success = teacup_obj.states[object_states.Inside].get_value(coffee_cup_obj)

            # # if teacup is grasped
            # success = teacup_obj.states[object_states.Touching].get_value(other=self.env.robots[0])

            # # if coffee_cup is grasped
            success_touching = coffee_cup_obj.states[object_states.Touching].get_value(other=self.env.robots[0])
            # get coffee_cup object position
            coffee_cup_pos = coffee_cup_obj.get_position_orientation()[0][2]
            success_lift = coffee_cup_pos > 0.82
            result.update({
                "touching": success_touching,
                "bddl": success_bddl,
                "lift": success_lift,
            })

        elif self.name.startswith("openarm_cashier"):
            apple_obj = self.env.scene.object_registry("name", "apple_1")
            candy_obj = self.env.scene.object_registry("name", "candy_1")
            bag_obj = self.env.scene.object_registry("name", "paper_bag_1")

            apple_inside = False
            candy_inside = False
            apple_geo_inside = False
            candy_geo_inside = False
            if apple_obj is not None and candy_obj is not None and bag_obj is not None:
                try:
                    apple_inside = apple_obj.states[object_states.Inside].get_value(bag_obj)
                    candy_inside = candy_obj.states[object_states.Inside].get_value(bag_obj)
                except Exception as exc:
                    print(f"[openarm_cashier][WARN] Inside predicate check failed: {exc}")

                bag_pos = bag_obj.get_position_orientation()[0]
                apple_pos = apple_obj.get_position_orientation()[0]
                candy_pos = candy_obj.get_position_orientation()[0]

                def _cashier_geo_inside(obj_pos):
                    xy_dist = th.linalg.norm(obj_pos[:2] - bag_pos[:2]).item()
                    dz = (obj_pos[2] - bag_pos[2]).item()
                    return xy_dist < 0.22 and -0.22 < dz < 0.12

                apple_geo_inside = _cashier_geo_inside(apple_pos)
                candy_geo_inside = _cashier_geo_inside(candy_pos)

            success_cashier = (apple_inside or apple_geo_inside) and (candy_inside or candy_geo_inside)
            result.update({
                "bddl": success_bddl,
                "apple_inside": apple_inside,
                "candy_inside": candy_inside,
                "apple_geo_inside": apple_geo_inside,
                "candy_geo_inside": candy_geo_inside,
            })
            result["task"] = success_bddl or success_cashier

        elif self.name.startswith("openarm_fruit_basket_bagging"):
            objects = {
                "object_1": self.env.scene.object_registry("name", "object_1"),
                "object_2": self.env.scene.object_registry("name", "object_2"),
                "object_3": self.env.scene.object_registry("name", "object_3"),
                "object_4": self.env.scene.object_registry("name", "object_4"),
            }
            containers = {
                "paper_bag_1": self.env.scene.object_registry("name", "paper_bag_1"),
                "basket_1": self.env.scene.object_registry("name", "basket_1"),
            }
            xy_margin = float(os.environ.get("OPENARM_FRUIT_SUCCESS_XY_MARGIN", "0.005"))
            z_margin_low = float(os.environ.get("OPENARM_FRUIT_SUCCESS_Z_MARGIN_LOW", "0.04"))
            z_margin_high = float(os.environ.get("OPENARM_FRUIT_SUCCESS_Z_MARGIN_HIGH", "0.08"))

            def _fruit_inside_or_geo(obj_name, container_name):
                obj = objects.get(obj_name)
                container = containers.get(container_name)
                if obj is None or container is None:
                    return {
                        "ok": False,
                        "raw_inside": False,
                        "geo": False,
                        "xy_dist": float("inf"),
                        "dz": float("inf"),
                    }
                raw_inside = False
                try:
                    raw_inside = bool(obj.states[object_states.Inside].get_value(container))
                except Exception as exc:
                    print(
                        "[openarm_fruit_basket_bagging][WARN] Inside predicate "
                        f"failed for {obj_name}->{container_name}: {exc}"
                    )
                obj_pos = obj.get_position_orientation()[0]
                container_center = container.aabb_center
                container_extent = container.aabb_extent
                xy_delta = th.abs(obj_pos[:2] - container_center[:2])
                xy_limit = container_extent[:2] / 2.0 + xy_margin
                xy_dist = th.linalg.norm(obj_pos[:2] - container_center[:2]).item()
                z_min = container_center[2] - container_extent[2] / 2.0 - z_margin_low
                z_max = container_center[2] + container_extent[2] / 2.0 + z_margin_high
                dz = (obj_pos[2] - container_center[2]).item()
                geo = bool(
                    th.all(xy_delta <= xy_limit).item()
                    and z_min.item() < obj_pos[2].item() < z_max.item()
                )
                return {
                    "ok": bool(raw_inside or geo),
                    "raw_inside": bool(raw_inside),
                    "geo": geo,
                    "xy_dist": float(xy_dist),
                    "dz": float(dz),
                    "xy_delta": [float(value) for value in xy_delta.detach().cpu().tolist()],
                    "xy_limit": [float(value) for value in xy_limit.detach().cpu().tolist()],
                    "z_min": float(z_min.item()),
                    "z_max": float(z_max.item()),
                }

            checks = {
                "apple_in_bag": _fruit_inside_or_geo("object_1", "paper_bag_1"),
                "orange_in_bag": _fruit_inside_or_geo("object_2", "paper_bag_1"),
                "lemon_in_basket": _fruit_inside_or_geo("object_3", "basket_1"),
                "pear_in_basket": _fruit_inside_or_geo("object_4", "basket_1"),
            }
            success_fruit = all(check["ok"] for check in checks.values())
            result.update({"bddl": success_bddl})
            for name, check in checks.items():
                result[name] = check["ok"]
                result[f"{name}_raw_inside"] = check["raw_inside"]
                result[f"{name}_geo"] = check["geo"]
                result[f"{name}_xy_dist"] = check["xy_dist"]
                result[f"{name}_dz"] = check["dz"]
                result[f"{name}_xy_delta"] = check["xy_delta"]
                result[f"{name}_xy_limit"] = check["xy_limit"]
                result[f"{name}_z_min"] = check["z_min"]
                result[f"{name}_z_max"] = check["z_max"]
            result["task"] = success_bddl or success_fruit

        elif self.name.startswith("openarm_drawer_storage"):
            object_1 = self.env.scene.object_registry("name", "object_1")
            object_2 = self.env.scene.object_registry("name", "object_2")
            cabinet = self.env.scene.object_registry("name", "drawer_cabinet_1")
            drawer_target = OPENARM_DRAWER_TARGETS["lower_drawer"]

            drawer_open = False
            drawer_open_fraction = 0.0
            object_1_inside = False
            object_2_inside = False
            if cabinet is not None:
                try:
                    drawer_open_fraction = drawer_target.get_open_fraction(self.env)
                    drawer_open = drawer_target.is_open(self.env)
                except Exception as exc:
                    print(f"[openarm_drawer_storage][WARN] drawer state check failed: {exc}")
            if object_1 is not None and object_2 is not None and cabinet is not None:
                try:
                    object_1_inside = bool(object_1.states[object_states.Inside].get_value(cabinet))
                    object_2_inside = bool(object_2.states[object_states.Inside].get_value(cabinet))
                except Exception as exc:
                    print(f"[openarm_drawer_storage][WARN] Inside predicate check failed: {exc}")

            result.update({
                "bddl": success_bddl,
                "drawer_open": bool(drawer_open),
                "drawer_open_fraction": float(drawer_open_fraction),
                "object_1_inside": object_1_inside,
                "object_2_inside": object_2_inside,
            })
            result["task"] = bool(drawer_open and object_1_inside and object_2_inside)

        elif self.name.startswith("openarm_real_exp_1"):
            object_1 = self.env.scene.object_registry("name", "object_1")
            object_2 = self.env.scene.object_registry("name", "object_2")
            bag_obj = self.env.scene.object_registry("name", "paper_bag_1")

            object_1_inside = False
            object_2_inside = False
            object_1_geo_inside = False
            object_2_geo_inside = False
            if object_1 is not None and object_2 is not None and bag_obj is not None:
                try:
                    object_1_inside = object_1.states[object_states.Inside].get_value(bag_obj)
                    object_2_inside = object_2.states[object_states.Inside].get_value(bag_obj)
                except Exception as exc:
                    print(f"[openarm_real_exp_1][WARN] Inside predicate check failed: {exc}")

                bag_pos = bag_obj.get_position_orientation()[0]
                object_1_pos = object_1.get_position_orientation()[0]
                object_2_pos = object_2.get_position_orientation()[0]
                bag_xy_radius = float(os.environ.get("OPENARM_REAL_EXP_1_SUCCESS_XY_RADIUS", "0.25"))
                bag_dz_min = float(os.environ.get("OPENARM_REAL_EXP_1_SUCCESS_DZ_MIN", "-0.25"))
                bag_dz_max = float(os.environ.get("OPENARM_REAL_EXP_1_SUCCESS_DZ_MAX", "0.12"))

                def _real_exp_geo_inside(obj_pos):
                    xy_dist = th.linalg.norm(obj_pos[:2] - bag_pos[:2]).item()
                    dz = (obj_pos[2] - bag_pos[2]).item()
                    return xy_dist < bag_xy_radius and bag_dz_min < dz < bag_dz_max

                object_1_geo_inside = _real_exp_geo_inside(object_1_pos)
                object_2_geo_inside = _real_exp_geo_inside(object_2_pos)

            success_real_exp = (object_1_inside or object_1_geo_inside) and (object_2_inside or object_2_geo_inside)
            result.update({
                "bddl": success_bddl,
                "object_1_inside": object_1_inside,
                "object_2_inside": object_2_inside,
                "object_1_geo_inside": object_1_geo_inside,
                "object_2_geo_inside": object_2_geo_inside,
            })
            result["task"] = success_bddl or success_real_exp

        elif self.name.startswith("openarm_bag_groceries"):
            apple_obj = self.env.scene.object_registry("name", "apple_1")
            orange_obj = self.env.scene.object_registry("name", "orange_1")
            canned_food_obj = self.env.scene.object_registry("name", "canned_food_1")
            candy_obj = self.env.scene.object_registry("name", "candy_1")
            bag_1_obj = self.env.scene.object_registry("name", "paper_bag_1")
            bag_2_obj = self.env.scene.object_registry("name", "paper_bag_2")
            bag_xy_radius = float(os.environ.get("OPENARM_BAG_SUCCESS_XY_RADIUS", "0.25"))
            bag_dz_min = float(os.environ.get("OPENARM_BAG_SUCCESS_DZ_MIN", "-0.25"))
            bag_dz_max = float(os.environ.get("OPENARM_BAG_SUCCESS_DZ_MAX", "0.12"))
            candy_bag_dz_max = float(os.environ.get("OPENARM_BAG_SUCCESS_CANDY_DZ_MAX", str(bag_dz_max)))

            def _check_inside_geo_heuristic(obj_, bag_, dz_max=None):
                if obj_ is None or bag_ is None:
                    return {
                        "inside": False,
                        "geo": False,
                        "xy_dist": float("inf"),
                        "dz": float("inf"),
                    }
                raw_inside_pred = False
                try:
                    raw_inside_pred = obj_.states[object_states.Inside].get_value(bag_)
                except:
                    pass
                obj_pos = obj_.get_position_orientation()[0]
                bag_pos = bag_.get_position_orientation()[0]
                xy_dist = th.linalg.norm(obj_pos[:2] - bag_pos[:2]).item()
                dz = (obj_pos[2] - bag_pos[2]).item()
                dz_max = bag_dz_max if dz_max is None else dz_max
                geo_in = (xy_dist < bag_xy_radius and bag_dz_min < dz < dz_max)
                inside_pred = bool(raw_inside_pred and geo_in)
                return {
                    "inside": inside_pred,
                    "raw_inside": bool(raw_inside_pred),
                    "geo": bool(geo_in),
                    "xy_dist": float(xy_dist),
                    "dz": float(dz),
                }

            # Task goal follows the source demo: canned food + candy into paper_bag_2;
            # apple + orange into paper_bag_1.
            cf = _check_inside_geo_heuristic(canned_food_obj, bag_2_obj)
            cd = _check_inside_geo_heuristic(candy_obj, bag_2_obj, dz_max=candy_bag_dz_max)
            ap = _check_inside_geo_heuristic(apple_obj, bag_1_obj)
            orange = _check_inside_geo_heuristic(orange_obj, bag_1_obj)
            cf_bag1 = _check_inside_geo_heuristic(canned_food_obj, bag_1_obj)
            cd_bag1 = _check_inside_geo_heuristic(candy_obj, bag_1_obj, dz_max=candy_bag_dz_max)
            ap_bag2 = _check_inside_geo_heuristic(apple_obj, bag_2_obj)
            orange_bag2 = _check_inside_geo_heuristic(orange_obj, bag_2_obj)

            c1 = cf["inside"] or cf["geo"]
            c2 = cd["inside"] or cd["geo"]
            c3 = ap["inside"] or ap["geo"]
            c4 = orange["inside"] or orange["geo"]

            success_bagging = c1 and c2 and c3 and c4
            result.update({
                "bddl": success_bddl,
                "canned_in_bag2": c1,
                "candy_in_bag2": c2,
                "apple_in_bag1": c3,
                "orange_in_bag1": c4,
                "canned_bag2_inside_pred": cf["inside"],
                "candy_bag2_inside_pred": cd["inside"],
                "apple_bag1_inside_pred": ap["inside"],
                "orange_bag1_inside_pred": orange["inside"],
                "canned_bag2_raw_inside_pred": cf["raw_inside"],
                "candy_bag2_raw_inside_pred": cd["raw_inside"],
                "apple_bag1_raw_inside_pred": ap["raw_inside"],
                "orange_bag1_raw_inside_pred": orange["raw_inside"],
                "canned_bag2_geo": cf["geo"],
                "candy_bag2_geo": cd["geo"],
                "apple_bag1_geo": ap["geo"],
                "orange_bag1_geo": orange["geo"],
                "canned_bag2_xy_dist": cf["xy_dist"],
                "candy_bag2_xy_dist": cd["xy_dist"],
                "apple_bag1_xy_dist": ap["xy_dist"],
                "orange_bag1_xy_dist": orange["xy_dist"],
                "canned_bag2_dz": cf["dz"],
                "candy_bag2_dz": cd["dz"],
                "apple_bag1_dz": ap["dz"],
                "orange_bag1_dz": orange["dz"],
                "canned_bag1_geo": cf_bag1["geo"],
                "candy_bag1_geo": cd_bag1["geo"],
                "apple_bag2_geo": ap_bag2["geo"],
                "orange_bag2_geo": orange_bag2["geo"],
                "canned_bag1_raw_inside_pred": cf_bag1["raw_inside"],
                "candy_bag1_raw_inside_pred": cd_bag1["raw_inside"],
                "apple_bag2_raw_inside_pred": ap_bag2["raw_inside"],
                "orange_bag2_raw_inside_pred": orange_bag2["raw_inside"],
                "canned_bag1_xy_dist": cf_bag1["xy_dist"],
                "candy_bag1_xy_dist": cd_bag1["xy_dist"],
                "apple_bag2_xy_dist": ap_bag2["xy_dist"],
                "orange_bag2_xy_dist": orange_bag2["xy_dist"],
                "canned_bag1_dz": cf_bag1["dz"],
                "candy_bag1_dz": cd_bag1["dz"],
                "apple_bag2_dz": ap_bag2["dz"],
                "orange_bag2_dz": orange_bag2["dz"],
            })
            result["task"] = success_bddl or success_bagging

        return result

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.OG_TYPE

    @property
    def version(self):
        """
        Returns version of robosuite used for this environment, eg. 1.2.0
        """
        return og.__version__

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(
            env_name=self.name,
            env_version=self.version,
            type=self.type,
            env_kwargs=deepcopy(self._init_kwargs)
        )

    @classmethod
    def create_for_data_processing(
        cls,
        env_name,
        **kwargs,
    ):
        # Always flatten observation space for data processing
        kwargs["env"]["flatten_obs_space"] = True
        return cls(env_name=env_name, **kwargs)

    @property
    def rollout_exceptions(self):
        return tuple()

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        return self.env

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)

    # Nothing below this is implemented yet - not needed for data generation
    def get_real_depth_map(self, depth_map):
        raise NotImplementedError

    def get_camera_intrinsic_matrix(self, camera_name, camera_height, camera_width):
        raise NotImplementedError

    def get_camera_extrinsic_matrix(self, camera_name):
        raise NotImplementedError

    def get_camera_transform_matrix(self, camera_name, camera_height, camera_width):
        raise NotImplementedError

    def get_reward(self):
        """
        Get current reward.
        """
        raise NotImplementedError

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        raise NotImplementedError

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        if 'tiago' in self.name:
            return 22
        elif 'r1' in self.name:
            return 21
        else:
            raise NotImplementedError
        
    def update_kwargs(self, kwargs):
        # RESOLUTION = (128, 450)
        RESOLUTION = (256, 256)

        # Explicity add the depth_linear and rgb modalities
        kwargs["robots"][0]["obs_modalities"].append("depth_linear")
        kwargs["robots"][0]["obs_modalities"].append("rgb")
        kwargs["robots"][0]["obs_modalities"].append("seg_instance")
        
        # Setting the camera height and width here because setting it later causes issues
        kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_height"] = RESOLUTION[0]
        kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_width"] = RESOLUTION[1]
        kwargs["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"]["horizontal_aperture"] = 25.0

        # Untucked reset joint positions. The torso is different from the default R1 untucked position
        if kwargs["robots"][0]["type"] == "R1" and self.robot_reset_pos == "untuck":
            kwargs["robots"][0]["reset_joint_pos"] = [
                    0.0000,
                    0.0000,
                    0.000,
                    0.000,
                    0.000,
                    -0.0000, # 6 virtual base joint 
                    1.375 if self.real_robot_mode else 0.5,
                    -2.195 if self.real_robot_mode else -1.0,
                    -0.96 if self.real_robot_mode else -0.8,
                    -0.0000, # 4 torso joints
                    -0.000,
                    0.000,
                    1.8944,
                    1.8945,
                    -0.9848,
                    -0.9849,
                    1.5612,
                    1.5621,
                    0.9097,
                    0.9096,
                    -1.5544,
                    -1.5545,
                    0.0500,
                    0.0500,
                    0.0500,
                    0.0500,
                ]
        elif kwargs["robots"][0]["type"] == "R1" and self.robot_reset_pos == "tuck":
            # Tucked reset joint positions. The torso is different from the default R1 tucked position
            kwargs["robots"][0]["reset_joint_pos"] = [
                    0.0000,
                    0.0000,
                    0.000,
                    0.000,
                    0.000,
                    -0.0000, # 6 virtual base joint 
                    1.375 if self.real_robot_mode else 0.5,
                    -2.195 if self.real_robot_mode else -1.0,
                    -0.96 if self.real_robot_mode else -0.8,
                    -0.0000, # 4 torso joints
                    0.0, # left arm joint 1
                    0.0, # right arm joint 1
                    0.0,
                    0.0,
                    -0.15,
                    -0.15,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0500,
                    0.0500,
                    0.0500,
                    0.0500,
                ] 


        # Always spawn robot at the origin with no rotation (this is to be compatible with curobo)
        kwargs["robots"][0]["position"] = [0.0, 0.0, 0.0]
        kwargs["robots"][0]["orientation"] = [0.0, 0.0, 0.0, 1.0]

    def update_params_r1_pick_cup(self, kwargs):
        if self.baseline in ["mimicgen", "skillgen"]:
            self.reset_base_pose = (th.tensor(kwargs["robots"][0]["position"]) + th.tensor([1.0, 0.0, 0.0]), kwargs["robots"][0]["orientation"])
        else:
            # self.reset_base_pose = (th.tensor([-0.863, -0.26, 0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
            self.reset_base_pose = (th.tensor([0.0, 0.0, 0]), th.tensor([0.0, 0.0, 0.0, 1.0]))

        # if self.name.endswith("D2"):
        #     kwargs["scene"]["load_object_categories"].append("straight_chair")

    def update_params_r1_tidy_table(self, kwargs):
        if kwargs["robots"][0]["type"] == "R1":
            kwargs["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
        else:
            kwargs["scene"]["load_room_instances"] = ["kitchen_0"]
        kwargs["scene"]["not_load_object_categories"] = ["taboret"]
        # NOTE: in mimicgen/skillgen we are reaplying the exact same base pose. So, we need the init robot pose to be the same as that in source demo
        if self.baseline not in ["mimicgen", "skillgen"]:
            original_quat = kwargs["robots"][0]["orientation"]
            rot_z_45 = R.from_euler('z', 45, degrees=True)
            original_rot = R.from_quat(original_quat)
            kwargs["robots"][0]["orientation"]
            new_rot = rot_z_45 * original_rot
            rotated_quat = new_rot.as_quat()
            kwargs["robots"][0]["orientation"] = rotated_quat
        
        self.reset_base_pose = (kwargs["robots"][0]["position"], kwargs["robots"][0]["orientation"])

    
    def update_params_r1_dishes_away(self, kwargs):
        if kwargs["robots"][0]["type"] == "R1":
            kwargs["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
        else:
            kwargs["scene"]["load_room_instances"] = ["kitchen_0"]
        # For the task of dishes away, we don't load the fridge
        kwargs["scene"]["not_load_object_categories"] = ["fridge"]
        if self.baseline not in ["mimicgen", "skillgen"]:
            # kwargs["robots"][0]["position"] = [4.1, 1.7, kwargs["robots"][0]["position"][2]]
            # kwargs["robots"][0]["orientation"] = R.from_euler('z', -1.1, degrees=False).as_quat().tolist()
            kwargs["robots"][0]["position"] = [5.4, 1.7, kwargs["robots"][0]["position"][2]]
            kwargs["robots"][0]["orientation"] = R.from_euler('z', -2.3, degrees=False).as_quat().tolist()
        
        self.reset_base_pose = (kwargs["robots"][0]["position"], kwargs["robots"][0]["orientation"])

    def update_params_r1_clean_pan(self, kwargs):
        if kwargs["robots"][0]["type"] == "R1":
            kwargs["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
        else:
            kwargs["scene"]["load_room_instances"] = ["kitchen_0"]
        if self.baseline not in ["mimicgen", "skillgen"]:
            # kwargs["robots"][0]["position"] = [4.1, 1.7, kwargs["robots"][0]["position"][2]]
            # kwargs["robots"][0]["orientation"] = R.from_euler('z', -1.1, degrees=False).as_quat().tolist()
            kwargs["robots"][0]["position"] = [5.4, 1.7, kwargs["robots"][0]["position"][2]]
            kwargs["robots"][0]["orientation"] = R.from_euler('z', -2.3, degrees=False).as_quat().tolist()
        
        self.reset_base_pose = (kwargs["robots"][0]["position"], kwargs["robots"][0]["orientation"])

    def _openarm_external_sensor_kwargs(self, camera_name):
        if uses_real_exp_1_viewer_camera(self.name):
            return get_real_exp_1_external_sensor_kwargs(camera_name)
        return square_camera_sensor_kwargs(256)
    
    def update_params_openarm_pick_cup(self, kwargs):
        # The source demo was recorded with robots=[] (robot loaded via BDDL task).
        # We need to explicitly add the OpenArmBimanual robot config so that
        # update_kwargs and subsequent env creation can proceed correctly.
        robot_config = {
            "type": "OpenArmBimanual",
            "name": "robot0",
            "action_normalize": False,
            "controller_config": {
                "arm_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "pos_kp": 150,
                    "command_input_limits": None,
                    "command_output_limits": None,
                    "use_impedances": False,
                    "use_delta_commands": False,
                },
                "arm_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "pos_kp": 150,
                    "command_input_limits": None,
                    "command_output_limits": None,
                    "use_impedances": False,
                    "use_delta_commands": False,
                },
                "gripper_left": _openarm_gripper_joint_controller_config(self.name),
                "gripper_right": _openarm_gripper_joint_controller_config(self.name),
            },
            "self_collisions": True,
            "obs_modalities": [],
            "position": [0.55, -0.27, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "grasping_mode": _openarm_task_grasping_mode(self.name),
            "sensor_config": {
                "VisionSensor": {
                    "sensor_kwargs": {
                        "image_height": 1080,
                        "image_width": 1080,
                    }
                }
            },
        }
        # kwargs["robots"].append(robot_config)
        # self.reset_base_pose = ([0.55, -0.27, 0.0], [0.0, 0.0, 0.0, 1.0])
        kwargs["robots"].append(robot_config)

        kwargs.setdefault("env", {})
        kwargs["env"]["external_sensors"] = [
            {
                "sensor_type": "VisionSensor",
                "name": "left_wrist_cam",
                "relative_prim_path": "/robot0/openarm_left_link7/left_wrist_cam",
                "modalities": ["rgb"],
                "sensor_kwargs": self._openarm_external_sensor_kwargs("left_wrist_cam"),
                "position": [0.03, 0.0, 0.04],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose_frame": "parent",
                "include_in_obs": True,
            },
            {
                "sensor_type": "VisionSensor",
                "name": "right_wrist_cam",
                "relative_prim_path": "/robot0/openarm_right_link7/right_wrist_cam",
                "modalities": ["rgb"],
                "sensor_kwargs": self._openarm_external_sensor_kwargs("right_wrist_cam"),
                "position": [0.03, 0.0, 0.04],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose_frame": "parent",
                "include_in_obs": True,
            },
            {
                "sensor_type": "VisionSensor",
                "name": "base_cam",
                "relative_prim_path": "/robot0/openarm_body_link0/base_cam",
                "modalities": ["rgb"],
                "sensor_kwargs": self._openarm_external_sensor_kwargs("base_cam"),
                "position": [0.35, 0.0, 0.35],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose_frame": "parent",
                "include_in_obs": True,
            },
        ]

        self.reset_base_pose = ([0.55, -0.27, 0.0], [0.0, 0.0, 0.0, 1.0])

    def _get_first_source_demo(self, h5_file):
        data_grp = h5_file.get("data")
        if data_grp is None:
            return None
        demo_keys = sorted([key for key in data_grp.keys() if key.startswith("demo_")])
        if not demo_keys:
            return None
        return data_grp[demo_keys[0]]

    def _get_source_scene_file_config(self, h5_file):
        data_grp = h5_file.get("data")
        if data_grp is None:
            return None
        scene_file = data_grp.attrs.get("scene_file", None)
        if scene_file is None:
            return None
        if isinstance(scene_file, bytes):
            scene_file = scene_file.decode("utf-8")
        try:
            return json.loads(scene_file)
        except Exception:
            scene_path = os.path.expanduser(os.path.expandvars(scene_file))
            if not os.path.isabs(scene_path):
                scene_path = os.path.join(os.getcwd(), scene_path)
            if not os.path.exists(scene_path):
                return None
            try:
                with open(scene_path, "r") as f:
                    return json.load(f)
            except Exception:
                return None

    def _get_openarm_cashier_source_robot_pose(self, default_pose):
        dataset_path = self.source_dataset_path_for_pose
        if dataset_path is None:
            return default_pose

        try:
            import h5py
            with h5py.File(dataset_path, "r") as f:
                demo_grp = self._get_first_source_demo(f)
                if demo_grp is not None and "datagen_info" in demo_grp and "base_pose" in demo_grp["datagen_info"]:
                    base_pose = np.asarray(demo_grp["datagen_info"]["base_pose"][0])
                    pos = base_pose[:3, 3].astype(float).tolist()
                    quat = R.from_matrix(base_pose[:3, :3]).as_quat().astype(float).tolist()
                    return pos, quat

                scene_cfg = self._get_source_scene_file_config(f)
                root_link = (
                    scene_cfg.get("state", {})
                    .get("registry", {})
                    .get("object_registry", {})
                    .get("robot0", {})
                    .get("root_link", {})
                    if scene_cfg is not None
                    else {}
                )
                if "pos" in root_link and "ori" in root_link:
                    pos = [float(v) for v in root_link["pos"]]
                    quat = [float(v) for v in root_link["ori"]]
                    return pos, quat
        except Exception as exc:
            print(f"[openarm_cashier][WARN] failed to read source robot base_pose from {dataset_path}: {exc}")

        return default_pose

    def _get_openarm_cashier_source_object_initial_poses(self):
        dataset_path = self.source_dataset_path_for_pose
        if dataset_path is None:
            return {}

        if self.name.startswith("openarm_bag_groceries"):
            object_names = ["apple_1", "candy_1", "orange_1", "canned_food_1", "paper_bag_1", "paper_bag_2"]
        elif self.name.startswith("openarm_drawer_storage"):
            object_names = ["object_1", "object_2", "drawer_cabinet_1"]
        elif self.name.startswith("openarm_real_exp_1"):
            object_names = ["object_1", "object_2", "paper_bag_1"]
        else:
            object_names = ["apple_1", "candy_1"]
        poses = {}
        try:
            import h5py
            with h5py.File(dataset_path, "r") as f:
                demo_grp = self._get_first_source_demo(f)
                object_poses = None
                if demo_grp is not None and "datagen_info" in demo_grp and "object_poses" in demo_grp["datagen_info"]:
                    object_poses = demo_grp["datagen_info"]["object_poses"]
                if object_poses is not None:
                    for obj_name in object_names:
                        if obj_name not in object_poses:
                            continue
                        pose = np.asarray(object_poses[obj_name][0])
                        poses[obj_name] = (
                            pose[:3, 3].astype(float).tolist(),
                            R.from_matrix(pose[:3, :3]).as_quat().astype(float).tolist(),
                        )
                    if poses:
                        return poses

                scene_cfg = self._get_source_scene_file_config(f)
                object_registry = (
                    scene_cfg.get("state", {}).get("registry", {}).get("object_registry", {})
                    if scene_cfg is not None
                    else {}
                )
                for obj_name in object_names:
                    root_link = object_registry.get(obj_name, {}).get("root_link", {})
                    if "pos" in root_link and "ori" in root_link:
                        poses[obj_name] = (
                            [float(v) for v in root_link["pos"]],
                            [float(v) for v in root_link["ori"]],
                        )
        except Exception as exc:
            print(f"[openarm_cashier][WARN] failed to read source object poses from {dataset_path}: {exc}")
        return poses

    def _get_openarm_cashier_source_initial_state(self):
        dataset_path = self.source_dataset_path_for_pose
        if dataset_path is None:
            return None

        try:
            import h5py
            with h5py.File(dataset_path, "r") as f:
                demo_grp = self._get_first_source_demo(f)
                if demo_grp is None or "state" not in demo_grp:
                    return None
                state = np.asarray(demo_grp["state"][0], dtype=np.float32)
                if "state_size" in demo_grp:
                    state_size = int(np.asarray(demo_grp["state_size"][0]))
                    state = state[:state_size]
                return state
        except Exception as exc:
            print(f"[openarm_cashier][WARN] failed to read source initial state from {dataset_path}: {exc}")
            return None

    def _load_openarm_cashier_source_initial_state(self):
        state = self._get_openarm_cashier_source_initial_state()
        if state is None:
            return False

        try:
            og.sim.load_state(th.from_numpy(state).to(th.float32), serialized=True)
            for _ in range(5):
                og.sim.step()
            return True
        except Exception as exc:
            print(f"[openarm_cashier][WARN] failed to load source frame 0 sim state: {exc}")
            return False

    def _get_openarm_cashier_source_initial_action(self):
        dataset_path = self.source_dataset_path_for_pose
        if dataset_path is None:
            return None

        try:
            import h5py
            with h5py.File(dataset_path, "r") as f:
                demo_grp = self._get_first_source_demo(f)
                if demo_grp is None:
                    return None
                action_key = "action" if "action" in demo_grp else "actions"
                if action_key not in demo_grp:
                    return None
                return np.asarray(demo_grp[action_key][0], dtype=np.float32)
        except Exception as exc:
            print(f"[openarm_cashier][WARN] failed to read source initial action from {dataset_path}: {exc}")
            return None

    def _align_openarm_cashier_robot_to_source_initial_action(self):
        action = self._get_openarm_cashier_source_initial_action()
        if action is None or action.shape[0] < 15:
            return False

        try:
            q = self.robot.get_joint_positions().clone()
            q[self.robot.arm_control_idx["left"]] = th.tensor(action[:7], dtype=q.dtype, device=q.device)
            q[self.robot.arm_control_idx["right"]] = th.tensor(action[8:15], dtype=q.dtype, device=q.device)
            self.robot.set_joint_positions(q)
            for _ in range(10):
                og.sim.step()
            return True
        except Exception as exc:
            print(f"[openarm_cashier][WARN] failed to align robot joints from source action[0]: {exc}")
            return False

    def _get_openarm_cashier_source_initial_eef_poses(self):
        dataset_path = self.source_dataset_path_for_pose
        if dataset_path is None:
            return {}

        try:
            import h5py
            with h5py.File(dataset_path, "r") as f:
                demo_grp = self._get_first_source_demo(f)
                if demo_grp is None or "datagen_info" not in demo_grp or "eef_pose" not in demo_grp["datagen_info"]:
                    return {}
                eef_pose = np.asarray(demo_grp["datagen_info"]["eef_pose"][0])
                return {
                    "left": eef_pose[:4, :],
                    "right": eef_pose[4:, :],
                }
        except Exception as exc:
            print(f"[openarm_cashier][WARN] failed to read source initial eef pose from {dataset_path}: {exc}")
            return {}

    def _diagnose_openarm_cashier_initial_robot_pose(self):
        source_eef_poses = self._get_openarm_cashier_source_initial_eef_poses()
        if not source_eef_poses:
            return

        print("[openarm_cashier] robot eef poses before randomization vs source frame 0:")
        for arm_name, src_pose in source_eef_poses.items():
            cur_pos, cur_quat = self.robot.get_eef_pose(arm_name)
            cur_mat = T.pose2mat((cur_pos, cur_quat))
            cur_pos_np = th.as_tensor(cur_mat[:3, 3]).detach().cpu().numpy()
            cur_quat_np = th.as_tensor(T.mat2quat(cur_mat[:3, :3])).detach().cpu().numpy()
            src_pos_np = np.asarray(src_pose[:3, 3], dtype=float)
            src_quat_np = R.from_matrix(src_pose[:3, :3]).as_quat()
            pos_delta = float(np.linalg.norm(cur_pos_np - src_pos_np))
            rot_delta = float((R.from_quat(src_quat_np).inv() * R.from_quat(cur_quat_np)).magnitude())
            print(
                f"  {arm_name}: current_pos={cur_pos_np.tolist()}, source_pos={src_pos_np.tolist()}, "
                f"pos_delta={pos_delta:.4f}, rot_delta_rad={rot_delta:.4f}"
            )

    def _diagnose_openarm_cashier_initial_object_poses(self):
        source_poses = self._get_openarm_cashier_source_object_initial_poses()
        if not source_poses:
            return

        print("[openarm_cashier] object poses before randomization vs source frame 0:")
        for obj_name, (src_pos, src_quat) in source_poses.items():
            obj = self.env.scene.object_registry("name", obj_name)
            if obj is None:
                print(f"  {obj_name}: missing in current scene")
                continue
            cur_pos, cur_quat = obj.get_position_orientation()
            cur_pos_np = th.as_tensor(cur_pos).detach().cpu().numpy()
            cur_quat_np = th.as_tensor(cur_quat).detach().cpu().numpy()
            src_pos_np = np.asarray(src_pos, dtype=float)
            src_quat_np = np.asarray(src_quat, dtype=float)
            pos_delta = float(np.linalg.norm(cur_pos_np - src_pos_np))
            rot_delta = float((R.from_quat(src_quat_np).inv() * R.from_quat(cur_quat_np)).magnitude())
            print(
                f"  {obj_name}: current_pos={cur_pos_np.tolist()}, source_pos={src_pos_np.tolist()}, "
                f"pos_delta={pos_delta:.4f}, rot_delta_rad={rot_delta:.4f}"
            )

    def _align_openarm_cashier_objects_to_source_initial_poses(self):
        source_poses = self._get_openarm_cashier_source_object_initial_poses()
        if not source_poses:
            return

        for obj_name, (src_pos, src_quat) in source_poses.items():
            obj = self.env.scene.object_registry("name", obj_name)
            if obj is None:
                continue
            obj.set_position_orientation(
                position=th.tensor(src_pos, dtype=th.float32),
                orientation=th.tensor(src_quat, dtype=th.float32),
            )
            if hasattr(obj, "set_linear_velocity"):
                obj.set_linear_velocity(th.zeros(3, dtype=th.float32))
            if hasattr(obj, "set_angular_velocity"):
                obj.set_angular_velocity(th.zeros(3, dtype=th.float32))
            obj.wake()

        for _ in range(10):
            og.sim.step()
        for _ in range(3):
            og.sim.render()

    def _resize_openarm_bag_grocery_bags_for_generation(self):
        if not self.name.startswith("openarm_bag_groceries"):
            return

        target_xy_scale = float(os.environ.get("OPENARM_BAG_GENERATION_XY_SCALE", "3"))
        if target_xy_scale <= 0.0:
            return

        # Unscaled paper_bag/bzsxgw visual bbox from metadata.json. This lets us
        # keep the robot-side (+Y) edge fixed while expanding the bag in XY.
        base_bbox_xy = th.tensor([0.10134799194335938, 0.1500009994506836], dtype=th.float32)
        keep_near_edge = os.environ.get("OPENARM_BAG_KEEP_NEAR_EDGE", "1").lower() not in ("0", "false", "no")

        for bag_name in ("paper_bag_1", "paper_bag_2"):
            bag_obj = self.env.scene.object_registry("name", bag_name)
            if bag_obj is None:
                continue

            old_scale = th.as_tensor(bag_obj.scale, dtype=th.float32).clone()
            new_scale = old_scale.clone()
            new_scale[0] = target_xy_scale
            new_scale[1] = target_xy_scale
            # Keep z scale unchanged.

            if th.allclose(old_scale, new_scale, atol=1e-5):
                continue

            pos, quat = bag_obj.get_position_orientation()
            pos = th.as_tensor(pos, dtype=th.float32).clone()
            quat_t = th.as_tensor(quat, dtype=th.float32)

            if keep_near_edge:
                rot = T.quat2mat(quat_t)
                old_half_y = 0.5 * (
                    abs(float(rot[1, 0])) * base_bbox_xy[0].item() * old_scale[0].item()
                    + abs(float(rot[1, 1])) * base_bbox_xy[1].item() * old_scale[1].item()
                )
                new_half_y = 0.5 * (
                    abs(float(rot[1, 0])) * base_bbox_xy[0].item() * new_scale[0].item()
                    + abs(float(rot[1, 1])) * base_bbox_xy[1].item() * new_scale[1].item()
                )
                # Robot / table-front side is +Y. Move the center toward -Y
                # by the added half extent so the +Y edge remains fixed.
                pos[1] -= max(0.0, new_half_y - old_half_y)

            bag_obj.scale = new_scale
            bag_obj.set_position_orientation(position=pos, orientation=quat_t)

        for _ in range(5):
            og.sim.step()
        try:
            self.cmg.update_obstacles()
        except Exception as exc:
            print(f"[openarm_bag_groceries][WARN] failed to update obstacles after bag resize: {exc}")

    def update_params_openarm_cashier(self, kwargs):
        # The cashier source demo is also recorded with robots=[] in env metadata.
        # Add OpenArm explicitly, matching the grocery-store cashier scene setup.
        source_base_pos, source_base_quat = self._get_openarm_cashier_source_robot_pose(
            ([-0.95, 4.75, 0.0], [0.0, 0.0, -0.7071, 0.7071])
        )
        is_drawer = self.name.startswith("openarm_drawer_storage")
        robot_config = {
            "type": "OpenArmBimanual",
            "name": "robot0",
            "fixed_base": True,
            "self_collisions": True,
            "action_normalize": False,
            "scale": [1.0, 1.0, 1.0] if is_drawer else [2.0, 2.0, 2.0],
            # OpenArm DOF order is interleaved left/right by joint index, then fingers.
            "reset_joint_pos": [
                0.3 if is_drawer else 0.5235987755982988,
                -0.3 if is_drawer else -0.5235987755982988,
                0.0, 0.0,
                0.0, 0.0,
                1.8 if is_drawer else 1.9198621771937625,
                1.8 if is_drawer else 1.9198621771937625,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
            ],
            "controller_config": {
                "arm_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "pos_kp": 150,
                    "command_input_limits": None,
                    "command_output_limits": None,
                    "use_impedances": False,
                    "use_delta_commands": False,
                },
                "arm_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "pos_kp": 150,
                    "command_input_limits": None,
                    "command_output_limits": None,
                    "use_impedances": False,
                    "use_delta_commands": False,
                },
                "gripper_left": _openarm_gripper_joint_controller_config(self.name),
                "gripper_right": _openarm_gripper_joint_controller_config(self.name),
            },
            "obs_modalities": [],
            "position": source_base_pos,
            "orientation": source_base_quat,
            "grasping_mode": _openarm_task_grasping_mode(self.name),
            "sensor_config": {
                "VisionSensor": {
                    "sensor_kwargs": {
                        "image_height": 1080,
                        "image_width": 1080,
                    }
                }
            },
        }
        kwargs["robots"].append(robot_config)

        kwargs.setdefault("env", {})
        kwargs["env"]["external_sensors"] = [
            {
                "sensor_type": "VisionSensor",
                "name": "left_wrist_cam",
                "relative_prim_path": "/robot0/openarm_left_link7/left_wrist_cam",
                "modalities": ["rgb"],
                "sensor_kwargs": self._openarm_external_sensor_kwargs("left_wrist_cam"),
                "position": [0.03, 0.0, 0.04],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose_frame": "parent",
                "include_in_obs": True,
            },
            {
                "sensor_type": "VisionSensor",
                "name": "right_wrist_cam",
                "relative_prim_path": "/robot0/openarm_right_link7/right_wrist_cam",
                "modalities": ["rgb"],
                "sensor_kwargs": self._openarm_external_sensor_kwargs("right_wrist_cam"),
                "position": [0.03, 0.0, 0.04],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose_frame": "parent",
                "include_in_obs": True,
            },
            {
                "sensor_type": "VisionSensor",
                "name": "base_cam",
                "relative_prim_path": "/robot0/openarm_body_link0/base_cam",
                "modalities": ["rgb"],
                "sensor_kwargs": self._openarm_external_sensor_kwargs("base_cam"),
                "position": [0.35, 0.0, 0.35],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose_frame": "parent",
                "include_in_obs": True,
            },
        ]

        self.reset_base_pose = (source_base_pos, source_base_quat)

    def update_params_r1_bringing_water(self, kwargs):
        if kwargs["robots"][0]["type"] == "R1":
            kwargs["scene"]["load_room_instances"] = ["kitchen_0", "dining_room_0", "entryway_0", "living_room_0"]
        else:
            kwargs["scene"]["load_room_instances"] = ["kitchen_0"]
            kwargs["scene"]["load_object_categories"] = ["floors", "fridge", "beer_bottle"]
        if self.baseline not in ["mimicgen", "skillgen"]:
            kwargs["robots"][0]["position"] = [6.0, -0.8, kwargs["robots"][0]["position"][2]] # TODO: need to change to a reasonable robot init position
            kwargs["robots"][0]["orientation"] = R.from_euler('z', -2.3, degrees=False).as_quat().tolist()
        
        self.reset_base_pose = (kwargs["robots"][0]["position"], kwargs["robots"][0]["orientation"])
    
    def update_env_post_creation_r1_pick_cup(self):
        floor = self.env.scene.object_registry("name", "floors_ptwlei_0")
        # floor2 = self.env.scene.object_registry("name", "floors_ifmioj_0")
        # breakfast_table = self.env.scene.object_registry("name", "breakfast_table_6")
        temp_state = og.sim.dump_state(serialized=False)
        og.sim.stop()
        floor.scale = th.tensor([1.8, 1.0, 1.0])
        # floor2.scale = th.tensor([1.8, 1.0, 1.0])
        # breakfast_table.scale = th.tensor([1.668, 1.038, 0.994])
        og.sim.play()
        og.sim.load_state(temp_state)
        og.sim.step()

        if self.name.endswith("D2"):
            distractor_objects = []
            
            obj = DatasetObject(
                name="pot_plant",
                category="pot_plant",
                model="mqhlkf",
                # model="udqjui",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="floors_ptwlei_0",
                associated_task_obj="coffee_cup_7",
                obstacle_for="navigation"
            )
            self.distractor_objects.append(distractor_object)
            
            obj = DatasetObject(
                name="straight_chair_0",
                category="straight_chair",
                model="amgwaw",
                # For some reason, this pose does not work!
                position=th.tensor([5.0,  0.0028,  0.4485]), 
                orientation=th.tensor([ 0.0016,  0.0020, -0.1448,  0.9895])
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="floors_ptwlei_0",
                associated_task_obj="coffee_cup_7",
                obstacle_for="navigation",
            )
            self.distractor_objects.append(distractor_object)

            # obj = DatasetObject(
            #     name="floor_lamp",
            #     category="floor_lamp",
            #     model="vdxlda",
            # )
            obj = DatasetObject(
                name="straight_chair_1",
                category="straight_chair",
                model="amgwaw",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="floors_ptwlei_0",
                associated_task_obj="coffee_cup_7",
                obstacle_for="navigation"
            )
            self.distractor_objects.append(distractor_object)

            obj = DatasetObject(
                name="gift_box",
                category="gift_box",
                model="mfalrc",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="breakfast_table_6",
                associated_task_obj="coffee_cup_7",
                obstacle_for="manipulation"
            )
            self.distractor_objects.append(distractor_object)

            # Load the objects into the scene
            og.sim.batch_add_objects(distractor_objects, [self.env.scene] * len(distractor_objects))
            
            # Set object pose to ensure no collision at spawn time
            x_pos = 5.0
            for distractor_object in distractor_objects:
                x_pos += 1.0
                distractor_object.set_position_orientation(position=th.tensor([x_pos,  0.0,  0.0]))
            og.sim.step()


    def update_env_post_creation_r1_tidy_table(self):
        if self.name.endswith("D2"):        
            distractor_objects = []
    
            obj = DatasetObject(
                name="vacuum",
                category="vacuum",
                model="bdmsbr",
                scale=th.tensor([1.0, 1.0, 1.5]),
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="floors_kxcpgy_0",
                associated_task_obj="drop_in_sink_awvzkn_0",
                obstacle_for="navigation"
            )
            self.distractor_objects.append(distractor_object)
            
            obj = DatasetObject(
                name="trash_can",
                category="trash_can",
                model="vasiit",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="floors_kxcpgy_0",
                associated_task_obj="drop_in_sink_awvzkn_0",
                obstacle_for="navigation",
            )
            self.distractor_objects.append(distractor_object)

            # obj = DatasetObject(
            #     name="floor_lamp",
            #     category="floor_lamp",
            #     model="jqsuky",
            #     scale=th.tensor([1.0, 1.0, 1.5]),
            # )
            # distractor_objects.append(obj)
            # distractor_object = dict(
            #     obj=obj,
            #     associated_furniture="floors_kxcpgy_0",
            #     associated_task_obj="teacup_601",
            #     obstacle_for="navigation"
            # )
            # self.distractor_objects.append(distractor_object)

            obj = DatasetObject(
                name="pot_plant",
                category="pot_plant",
                model="cqqyzp",
                scale=th.tensor([1.3, 1.3, 1.3]),
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="bar_udatjt_0",
                associated_task_obj="teacup_601",
                obstacle_for="manipulation"
            )
            self.distractor_objects.append(distractor_object)

            # obj = DatasetObject(
            #     name="pot_plant_2",
            #     category="pot_plant",
            #     model="cqqyzp",
            #     scale=th.tensor([1.3, 1.3, 1.3]),
            # )
            # distractor_objects.append(obj)
            # distractor_object = dict(
            #     obj=obj,
            #     associated_furniture="bar_udatjt_0",
            #     associated_task_obj="teacup_601",
            #     obstacle_for="manipulation"
            # )
            # self.distractor_objects.append(distractor_object)

            obj = DatasetObject(
                name="wine_bottle",
                category="wine_bottle",
                model="inkqch",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="bar_udatjt_0",
                associated_task_obj="teacup_601",
                obstacle_for="manipulation"
            )
            self.distractor_objects.append(distractor_object)

            obj = DatasetObject(
                name="laptop",
                category="laptop",
                model="izydvb",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="bar_udatjt_0",
                associated_task_obj="teacup_601",
                obstacle_for="manipulation"
            )
            self.distractor_objects.append(distractor_object)

            obj = DatasetObject(
                name="loudspeaker",
                category="loudspeaker",
                model="fsyioq",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="bar_udatjt_0",
                associated_task_obj="teacup_601",
                obstacle_for="manipulation"
            )
            self.distractor_objects.append(distractor_object)

            state = og.sim.dump_state()
            og.sim.stop()
            # Load the objects into the scene
            og.sim.batch_add_objects(distractor_objects, [self.env.scene] * len(distractor_objects))
            og.sim.play()
            og.sim.load_state(state)
            
            # Set object pose to ensure no collision at spawn time
            x_pos = 5.0
            for distractor_object in distractor_objects:
                x_pos += 1.0
                distractor_object.set_position_orientation(position=th.tensor([x_pos,  0.0,  0.0]))
                # Open the laptop
                if distractor_object.name == "laptop":
                    distractor_object.joints["j_screen"].set_pos(1.0, normalized=True)
            og.sim.step()

    
    def update_env_post_creation_r1_dishes_away(self):
        shelf = self.env.scene.object_registry("name", "shelf_pfusrd_1")
        shelf.set_position_orientation(position=th.tensor([ 7.122, -2.029,  1.403]))
        for _ in range(5): og.sim.step()

        if self.name.endswith("D2"):        
            distractor_objects = []
    
            # obj = DatasetObject(
            #     name="vacuum",
            #     category="vacuum",
            #     model="bdmsbr",
            #     scale=th.tensor([1.0, 1.0, 1.5]),
            # )
            # distractor_objects.append(obj)
            # distractor_object = dict(
            #     obj=obj,
            #     associated_furniture="floors_kxcpgy_0",
            #     associated_task_obj="plate_602",
            #     obstacle_for="navigation"
            # )
            # self.distractor_objects.append(distractor_object)

            obj = DatasetObject(
                name="trash_can",
                category="trash_can",
                model="vasiit",
                scale=th.tensor([0.7, 0.7, 1.0]),
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="floors_kxcpgy_0",
                associated_task_obj="plate_602",
                obstacle_for="navigation",
            )
            self.distractor_objects.append(distractor_object)

            
            obj = DatasetObject(
                name="mop",
                category="mop",
                model="qclfvj",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="floors_kxcpgy_0",
                associated_task_obj="plate_601",
                obstacle_for="navigation"
            )
            self.distractor_objects.append(distractor_object)

            # obj = DatasetObject(
            #     name="pot_plant",
            #     category="pot_plant",
            #     model="cqqyzp",
            #     scale=th.tensor([1.3, 1.3, 1.3]),
            # )
            # distractor_objects.append(obj)
            # distractor_object = dict(
            #     obj=obj,
            #     associated_furniture="bar_rkgjer_0",
            #     associated_task_obj="frying_pan_602",
            #     obstacle_for="manipulation"
            # )
            # self.distractor_objects.append(distractor_object)
            
            # barbecue_sauce_bottle-gfxrnj
            # bottle_of_beer-mljzrl
            # bowl-wtepsx
            # bowl-tvtive
            # can_of_oatmeal-qyukhm

            # obj = DatasetObject(
            #     name="instant_pot",
            #     category="instant_pot",
            #     model="wengzf",
            #     scale=th.tensor([0.5, 0.5, 0.5]),
            # )
            # distractor_objects.append(obj)
            # distractor_object = dict(
            #     obj=obj,
            #     associated_furniture="countertop_kelker_0",
            #     associated_task_obj="frying_pan_602",
            #     obstacle_for="manipulation"
            # )
            # self.distractor_objects.append(distractor_object)

            obj = DatasetObject(
                name="can_of_oatmeal",
                category="can_of_oatmeal",
                model="qyukhm",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="countertop_kelker_0",
                associated_task_obj="plate_601",
                obstacle_for="manipulation"
            )
            self.distractor_objects.append(distractor_object)

            # obj = DatasetObject(
            #     name="wine_bottle",
            #     category="wine_bottle",
            #     model="inkqch",
            # )
            # distractor_objects.append(obj)
            # distractor_object = dict(
            #     obj=obj,
            #     associated_furniture="countertop_kelker_0",
            #     associated_task_obj="frying_pan_602",
            #     obstacle_for="manipulation"
            # )
            # self.distractor_objects.append(distractor_object)

            obj = DatasetObject(
                name="bowl_1",
                category="bowl",
                model="wtepsx",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="countertop_kelker_0",
                associated_task_obj="plate_602",
                obstacle_for="manipulation"
            )
            self.distractor_objects.append(distractor_object)

            # obj = DatasetObject(
            #     name="bowl_2",
            #     category="bowl",
            #     model="tvtive",
            # )
            # distractor_objects.append(obj)
            # distractor_object = dict(
            #     obj=obj,
            #     associated_furniture="bar_rkgjer_0",
            #     associated_task_obj="scrub_brush_601",
            #     obstacle_for="manipulation"
            # )
            # self.distractor_objects.append(distractor_object)

            state = og.sim.dump_state()
            og.sim.stop()
            # Load the objects into the scene
            og.sim.batch_add_objects(distractor_objects, [self.env.scene] * len(distractor_objects))
            og.sim.play()
            og.sim.load_state(state)
            
            # Set object pose to ensure no collision at spawn time
            x_pos = 5.0
            for distractor_object in distractor_objects:
                x_pos += 1.0
                distractor_object.set_position_orientation(position=th.tensor([x_pos,  0.0,  0.0]))
                # Open the laptop
                if distractor_object.name == "laptop":
                    distractor_object.joints["j_screen"].set_pos(1.0, normalized=True)
            og.sim.step()

    def update_env_post_creation_r1_clean_pan(self):
        # Moving the scrub away from the faucet
        if self.baseline not in ["mimicgen", "skillgen"]: 
            scrub_brush_601 = self.env.scene.object_registry("name", "scrub_brush_601")
            scrub_brush_601.set_position_orientation(position=th.tensor([6.5, -1.856, 0.905]), orientation=th.tensor([0.796, -0.606, -0.001, -0.007]))
            for _ in range(5): og.sim.step()

        # Set the default orn of pan (around which D0 will sample)
        frying_pan_602 = self.env.scene.object_registry("name", "frying_pan_602")
        orientation = frying_pan_602.get_position_orientation()[1]
        rot_z = R.from_euler('z', -45, degrees=True)
        original_rot = R.from_quat(orientation)
        new_rot = rot_z * original_rot
        rotated_quat = new_rot.as_quat()
        frying_pan_602.set_position_orientation(orientation=rotated_quat)

        if self.name.endswith("D2"):        
            distractor_objects = []
    
            # obj = DatasetObject(
            #     name="vacuum",
            #     category="vacuum",
            #     model="bdmsbr",
            #     scale=th.tensor([1.0, 1.0, 1.5]),
            # )
            # distractor_objects.append(obj)
            # distractor_object = dict(
            #     obj=obj,
            #     associated_furniture="floors_kxcpgy_0",
            #     associated_task_obj="frying_pan_602",
            #     obstacle_for="navigation"
            # )
            # self.distractor_objects.append(distractor_object)
            
            # obj = DatasetObject(
            #     name="trash_can",
            #     category="trash_can",
            #     model="vasiit",
            # )
            # distractor_objects.append(obj)
            # distractor_object = dict(
            #     obj=obj,
            #     associated_furniture="floors_kxcpgy_0",
            #     associated_task_obj="scrub_brush_601",
            #     obstacle_for="navigation",
            # )
            # self.distractor_objects.append(distractor_object)

            # obj = DatasetObject(
            #     name="floor_lamp",
            #     category="floor_lamp",
            #     model="jqsuky",
            #     scale=th.tensor([1.0, 1.0, 1.5]),
            # )
            # distractor_objects.append(obj)
            # distractor_object = dict(
            #     obj=obj,
            #     associated_furniture="floors_kxcpgy_0",
            #     associated_task_obj="teacup_601",
            #     obstacle_for="navigation"
            # )
            # self.distractor_objects.append(distractor_object)

            # obj = DatasetObject(
            #     name="mop",
            #     category="mop",
            #     model="qclfvj",
            # )
            # distractor_objects.append(obj)
            # distractor_object = dict(
            #     obj=obj,
            #     associated_furniture="floors_kxcpgy_0",
            #     associated_task_obj="frying_pan_602",
            #     obstacle_for="navigation"
            # )
            # self.distractor_objects.append(distractor_object)

            # obj = DatasetObject(
            #     name="pot_plant",
            #     category="pot_plant",
            #     model="cqqyzp",
            #     scale=th.tensor([1.3, 1.3, 1.3]),
            # )
            # distractor_objects.append(obj)
            # distractor_object = dict(
            #     obj=obj,
            #     associated_furniture="bar_rkgjer_0",
            #     associated_task_obj="frying_pan_602",
            #     obstacle_for="manipulation"
            # )
            # self.distractor_objects.append(distractor_object)
            
            # barbecue_sauce_bottle-gfxrnj
            # bottle_of_beer-mljzrl
            # bowl-wtepsx
            # bowl-tvtive
            # can_of_oatmeal-qyukhm

            obj = DatasetObject(
                name="instant_pot",
                category="instant_pot",
                model="wengzf",
                scale=th.tensor([0.5, 0.5, 0.5]),
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="countertop_kelker_0",
                associated_task_obj="frying_pan_602",
                obstacle_for="manipulation"
            )
            self.distractor_objects.append(distractor_object)

            obj = DatasetObject(
                name="can_of_oatmeal",
                category="can_of_oatmeal",
                model="qyukhm",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="bar_rkgjer_0",
                associated_task_obj="scrub_brush_601",
                obstacle_for="manipulation"
            )
            self.distractor_objects.append(distractor_object)

            obj = DatasetObject(
                name="wine_bottle",
                category="wine_bottle",
                model="inkqch",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="countertop_kelker_0",
                associated_task_obj="frying_pan_602",
                obstacle_for="manipulation"
            )
            self.distractor_objects.append(distractor_object)

            obj = DatasetObject(
                name="bowl_1",
                category="bowl",
                model="wtepsx",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="bar_rkgjer_0",
                associated_task_obj="scrub_brush_601",
                obstacle_for="manipulation"
            )
            self.distractor_objects.append(distractor_object)

            obj = DatasetObject(
                name="bowl_2",
                category="bowl",
                model="tvtive",
            )
            distractor_objects.append(obj)
            distractor_object = dict(
                obj=obj,
                associated_furniture="bar_rkgjer_0",
                associated_task_obj="scrub_brush_601",
                obstacle_for="manipulation"
            )
            self.distractor_objects.append(distractor_object)

            state = og.sim.dump_state()
            og.sim.stop()
            # Load the objects into the scene
            og.sim.batch_add_objects(distractor_objects, [self.env.scene] * len(distractor_objects))
            og.sim.play()
            og.sim.load_state(state)
            
            # Set object pose to ensure no collision at spawn time
            x_pos = 5.0
            for distractor_object in distractor_objects:
                x_pos += 1.0
                distractor_object.set_position_orientation(position=th.tensor([x_pos,  0.0,  0.0]))
                # Open the laptop
                if distractor_object.name == "laptop":
                    distractor_object.joints["j_screen"].set_pos(1.0, normalized=True)
            og.sim.step()
