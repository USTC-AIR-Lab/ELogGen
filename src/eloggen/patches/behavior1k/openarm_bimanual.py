import os
from functools import cached_property

import torch as th

from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.python_utils import classproperty


class OpenArmBimanual(ManipulationRobot):
    """
    Fixed-base OpenArm bimanual manipulator.
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=True,
        link_physics_materials=None,
        load_config=None,
        fixed_base=True,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=False,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities=("rgb", "proprio"),
        include_sensor_names=None,
        exclude_sensor_names=None,
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="assisted",
        finger_static_friction=None,
        finger_dynamic_friction=None,
        **kwargs,
    ):
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            include_sensor_names=include_sensor_names,
            exclude_sensor_names=exclude_sensor_names,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            finger_static_friction=finger_static_friction,
            finger_dynamic_friction=finger_dynamic_friction,
            **kwargs,
        )

    @property
    def model_name(self):
        return "openarmbimanual"

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("OpenArmBimanual does not support discrete actions!")

    @property
    def _raw_controller_order(self):
        controllers = []
        for arm in self.arm_names:
            controllers += [f"arm_{arm}", f"gripper_{arm}"]
        return controllers

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        for arm in self.arm_names:
            # Joint-space control is the most robust default for a new imported robot.
            controllers[f"arm_{arm}"] = "JointController"
            controllers[f"gripper_{arm}"] = "MultiFingerGripperController"
        return controllers

    @property
    def _default_controller_config(self):
        cfg = super()._default_controller_config
        for arm in self.arm_names:
            arm_key = f"arm_{arm}"
            grip_key = f"gripper_{arm}"
            # Switch arm JointController to absolute position mode.
            # The base-class default is use_delta_commands=True (incremental), which conflicts
            # with direct set_joint_positions calls and causes residual-delta jitter every step.
            if arm_key in cfg and "JointController" in cfg[arm_key]:
                cfg[arm_key]["JointController"]["motor_type"] = "position"
                cfg[arm_key]["JointController"]["use_delta_commands"] = False
                # command_input_limits defaults to (-1, 1) which clips absolute position targets
                # (e.g. joint at 1.2 rad → clip to 1.0 → controller pulls joint backwards).
                # Set to None so the absolute position value passes through unmodified.
                cfg[arm_key]["JointController"]["command_input_limits"] = None
                # command_output_limits=None means no input→output scaling either.
                cfg[arm_key]["JointController"]["command_output_limits"] = None
            # Gripper: binary mode expects a ±1 open/close signal, NOT raw joint positions.
            if grip_key in cfg and "MultiFingerGripperController" in cfg[grip_key]:
                cfg[grip_key]["MultiFingerGripperController"]["mode"] = "binary"
        return cfg

    @property
    def _default_joint_pos(self):
        return self.untucked_default_joint_pos

    @property
    def tucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        for arm in self.arm_names:
            pos[self.gripper_control_idx[arm]] = th.tensor([0.044])
        return pos

    @property
    def untucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        # Match the cashier task's startup pose.
        arm_defaults = {
            "left": th.tensor([0.5235987755982988, 0.0, 0.0, 1.9198621771937625, 0.0, 0.0, 0.0]),
            "right": th.tensor([-0.5235987755982988, 0.0, 0.0, 1.9198621771937625, 0.0, 0.0, 0.0]),
        }
        for arm in self.arm_names:
            pos[self.arm_control_idx[arm]] = arm_defaults[arm]
            pos[self.gripper_control_idx[arm]] = th.tensor([0.0])
        return pos

    @classproperty
    def n_arms(cls):
        return 2

    @classproperty
    def arm_names(cls):
        return ["left", "right"]

    @cached_property
    def arm_link_names(self):
        return {arm: [f"openarm_{arm}_link{i}" for i in range(1, 8)] for arm in self.arm_names}

    @cached_property
    def arm_joint_names(self):
        return {arm: [f"openarm_{arm}_joint{i}" for i in range(1, 8)] for arm in self.arm_names}

    @cached_property
    def eef_link_names(self):
        return {arm: f"openarm_{arm}_hand_tcp" for arm in self.arm_names}

    @cached_property
    def finger_link_names(self):
        return {
            "left": ["openarm_left_left_finger", "openarm_left_right_finger"],
            "right": ["openarm_right_left_finger", "openarm_right_right_finger"],
        }

    @cached_property
    def finger_joint_names(self):
        # Do not include mimic joints as independently controlled joints.
        return {"left": ["openarm_left_finger_joint1"], "right": ["openarm_right_finger_joint1"]}

    @cached_property
    def gripper_link_names(self):
        # The hand body link is the gripper "body" — exclusive of arm links and finger links.
        return {arm: [f"openarm_{arm}_hand"] for arm in self.arm_names}

    @property
    def teleop_rotation_offset(self):
        # Offset quaternion (x,y,z,w) aligning the robot EEF frame with the teleoperation device frame.
        # OpenArm EEF z-axis points out from fingertips — same convention as franka gripper.
        return {arm: th.tensor([-1.0, 0.0, 0.0, 0.0]) for arm in self.arm_names}

    @property
    def _assisted_grasp_start_points(self):
        return {
            arm: [
                GraspingPoint(
                    link_name=f"openarm_{arm}_right_finger",
                    position=th.tensor([0.0, 0.0, 0.03]),
                )
            ]
            for arm in self.arm_names
        }

    @property
    def _assisted_grasp_end_points(self):
        return {
            arm: [
                GraspingPoint(
                    link_name=f"openarm_{arm}_left_finger",
                    position=th.tensor([0.0, 0.0, 0.03]),
                )
            ]
            for arm in self.arm_names
        }

    @property
    def disabled_collision_pairs(self):
        pairs = []
        for arm in self.arm_names:
            # Adjacent arm links: link0↔link1 ... link6↔link7
            for i in range(7):
                pairs.append([f"openarm_{arm}_link{i}", f"openarm_{arm}_link{i + 1}"])
            # Skip-one pairs for compact wrist area (joints 5-7 offsets are very small,
            # non-adjacent links physically overlap during wrist rotation).
            # This is the primary cause of joint-6/7 jitter — mirrors franka's link5↔link7 fix.
            pairs.append([f"openarm_{arm}_link4", f"openarm_{arm}_link6"])
            pairs.append([f"openarm_{arm}_link5", f"openarm_{arm}_link7"])
            # All wrist-area links ↔ hand (link6 is especially close to hand during joint7 motion)
            pairs.append([f"openarm_{arm}_link5", f"openarm_{arm}_hand"])
            pairs.append([f"openarm_{arm}_link6", f"openarm_{arm}_hand"])
            pairs.append([f"openarm_{arm}_link7", f"openarm_{arm}_hand"])
            # Gripper body to fingers and tcp
            pairs.append([f"openarm_{arm}_hand", f"openarm_{arm}_left_finger"])
            pairs.append([f"openarm_{arm}_hand", f"openarm_{arm}_right_finger"])
            pairs.append([f"openarm_{arm}_hand", f"openarm_{arm}_hand_tcp"])
            # Left finger ↔ right finger
            pairs.append([f"openarm_{arm}_left_finger", f"openarm_{arm}_right_finger"])
        # Body base ↔ both arm bases
        pairs.append(["openarm_body_link0", "openarm_left_link0"])
        pairs.append(["openarm_body_link0", "openarm_right_link0"])
        # Cross-arm: keep body link0 out of link1 collisions
        pairs.append(["openarm_left_link0", "openarm_right_link0"])
        return pairs

    @property
    def usd_path(self):
        model = self.model_name.lower()
        return os.path.join(get_dataset_path("custom_dataset"), f"objects/robot/{model}/usd/{model}.usda")

    @property
    def urdf_path(self):
        model = self.model_name.lower()
        return os.path.join(get_dataset_path("custom_dataset"), f"objects/robot/{model}/urdf/{model}_with_meta_links.urdf")

    @property
    def curobo_path(self):
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

        model = self.model_name.lower()
        curobo_dir = os.path.join(get_dataset_path("custom_dataset"), f"objects/robot/{model}/curobo")
        default_cfg = os.path.join(curobo_dir, f"{model}_description_curobo_default.yaml")
        return {
            CuRoboEmbodimentSelection.DEFAULT: default_cfg,
            CuRoboEmbodimentSelection.ARM: os.path.join(curobo_dir, f"{model}_description_curobo_arm.yaml"),
            CuRoboEmbodimentSelection.ARM_NO_TORSO: os.path.join(
                curobo_dir, f"{model}_description_curobo_arm_no_torso.yaml"
            ),
        }
