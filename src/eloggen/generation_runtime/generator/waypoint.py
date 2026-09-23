"""
A collection of classes used to represent waypoints and trajectories.
"""
import json
import os
import time
import numpy as np
import copy
from copy import deepcopy

import eloggen.generation_runtime.geometry as PoseUtils

import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
import torch as th
import omnigibson as og
from omnigibson.robots.openarm_bimanual import OpenArmBimanual

from scipy.spatial.transform import Rotation as R

class Waypoint(object):
    """
    Represents a single desired 6-DoF waypoint, along with corresponding gripper actuation for this point.
    """
    def __init__(self, pose, gripper_action, noise=None):
        """
        Args:
            pose (np.array): 4x4 pose target for robot controller
            gripper_action (np.array): gripper action for robot controller
            noise (float or None): action noise amplitude to apply during execution at this timestep
                (for arm actions, not gripper actions)
        """
        self.pose = np.array(pose)
        self.gripper_action = np.array(gripper_action)
        self.noise = noise
        assert len(self.gripper_action.shape) == 1
    
    def merge_wp(self, other):
        """
        Merge another Waypoint object into this one.
        """
        self.pose = np.concatenate([self.pose, other.pose], axis=0)
        self.gripper_action = np.concatenate([self.gripper_action, other.gripper_action], axis=0)
        self.noise = min(self.noise, other.noise)
        # TODO: the noise here is set to 0, can be change to help reduce the sim to real gap due to the sensor observation noises
        self.noise = 0.0


class WaypointSequence(object):
    """
    Represents a sequence of Waypoint objects.
    """
    def __init__(self, sequence=None):
        """
        Args:
            sequence (list or None): if provided, should be an list of Waypoint objects
        """
        if sequence is None:
            self.sequence = []
        else:
            for waypoint in sequence:
                assert isinstance(waypoint, Waypoint)
            self.sequence = deepcopy(sequence)

    @classmethod
    def from_poses(cls, poses, gripper_actions, action_noise):
        """
        Instantiate a WaypointSequence object given a sequence of poses, 
        gripper actions, and action noise.

        Args:
            poses (np.array): sequence of pose matrices of shape (T, 4, 4)
            gripper_actions (np.array): sequence of gripper actions
                that should be applied at each timestep of shape (T, D).
            action_noise (float or np.array): sequence of action noise
                magnitudes that should be applied at each timestep. If a 
                single float is provided, the noise magnitude will be
                constant over the trajectory.
        """
        assert isinstance(action_noise, float) or isinstance(action_noise, np.ndarray)

        # handle scalar to numpy array conversion
        num_timesteps = poses.shape[0]
        if isinstance(action_noise, float):
            action_noise = action_noise * np.ones((num_timesteps, 1))
        action_noise = action_noise.reshape(-1, 1)

        # make WaypointSequence instance
        sequence = [
            Waypoint(
                pose=poses[t],
                gripper_action=gripper_actions[t],
                noise=action_noise[t, 0],
            )
            for t in range(num_timesteps)
        ]
        return cls(sequence=sequence)

    def __len__(self):
        # length of sequence
        return len(self.sequence)

    def __getitem__(self, ind):
        """
        Returns waypoint at index.

        Returns:
            waypoint (Waypoint instance)
        """
        return self.sequence[ind]

    def __add__(self, other):
        """
        Defines addition (concatenation) of sequences
        """
        return WaypointSequence(sequence=(self.sequence + other.sequence))

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        return deepcopy(self.sequence[-1])

    def split(self, ind):
        """
        Splits this sequence into 2 pieces, the part up to time index @ind, and the
        rest. Returns 2 WaypointSequence objects.
        """
        seq_1 = self.sequence[:ind]
        seq_2 = self.sequence[ind:]
        return WaypointSequence(sequence=seq_1), WaypointSequence(sequence=seq_2)

    def merge(self, other):
        """
        Merge another WaypointSequence object into this one.
        """
        self.sequence += other.sequence

class WaypointTrajectory(object):
    """
    A sequence of WaypointSequence objects that corresponds to a full 6-DoF trajectory.
    """
    def __init__(self):
        self.waypoint_sequences = []

    def _select_reference_object(self, object_ref):
        left_ref = object_ref.get("arm_left")
        right_ref = object_ref.get("arm_right")

        if right_ref is None:
            return left_ref
        if left_ref is None:
            return right_ref
        return right_ref

    def _resolve_tracking_object(self, env, env_interface, ref_object):
        if ref_object is None:
            return None

        if "torso" in ref_object:
            if not isinstance(env.robot, OpenArmBimanual):
                raise TypeError("ElogGen generation requires OpenArmBimanual")
            torso_link_name = "base_link"
            return env.env.robots[0].links[torso_link_name]

        if env_interface is not None and hasattr(env_interface, "_get_object_by_name"):
            obj = env_interface._get_object_by_name(ref_object)
            if obj is not None:
                return obj

        task = getattr(env.env, "task", None)
        object_scope = getattr(task, "object_scope", None)
        if object_scope is not None and ref_object in object_scope:
            return object_scope[ref_object]

        scene = getattr(env.env, "scene", None)
        if scene is not None:
            return scene.object_registry("name", ref_object)

        return None

    def _tensor_indices_to_numpy(self, indices):
        if hasattr(indices, "detach"):
            indices = indices.detach().cpu().numpy()
        return np.asarray(indices, dtype=int)

    def _openarm_continuous_gripper_enabled(self):
        return bool(getattr(self, "_openarm_continuous_gripper_active", False))

    def _openarm_gripper_physical_target_enabled(self):
        return (
            self._openarm_continuous_gripper_enabled()
            and os.environ.get("OPENARM_GRIPPER_PHYSICAL_TARGET", "1") != "0"
        )

    def _openarm_gripper_width_limits(self):
        # The imported OpenArm gripper's prismatic joint is physically closed at
        # 0.0. LeRobot export still maps/clips this to the real-robot close value
        # of 0.02, but driving the simulator to 0.0 avoids half-closed sticky grasps.
        close_width = float(os.environ.get("OPENARM_GRIPPER_SIM_CLOSE", "0.0"))
        open_width = float(os.environ.get("OPENARM_GRIPPER_SIM_OPEN", "0.044"))
        return close_width, open_width

    def _openarm_gripper_hold_action(self, robot, arm, physical_target=None):
        if physical_target is None:
            physical_target = self._openarm_gripper_physical_target_enabled()
        close_width, open_width = self._openarm_gripper_width_limits()
        try:
            if robot.is_grasping(arm=arm) == og.controllers.IsGraspingState.TRUE:
                return close_width if physical_target else -1.0
        except Exception:
            pass
        return open_width if physical_target else 1.0

    def _make_openarm_gripper_interp_state(self, env, robot):
        env_name = getattr(env, "name", None) or getattr(getattr(env, "env", None), "name", "")
        self._openarm_continuous_gripper_active = (
            isinstance(robot, OpenArmBimanual)
            and str(env_name).startswith("openarm_real_exp_1")
        )
        if not self._openarm_continuous_gripper_enabled():
            return None
        if os.environ.get("OPENARM_GRIPPER_TARGET_INTERP", "1") == "0":
            return None

        physical_target = self._openarm_gripper_physical_target_enabled()
        if physical_target:
            close_width, open_width = self._openarm_gripper_width_limits()
            fps = float(os.environ.get("OPENARM_GRIPPER_SIM_FPS", os.environ.get("ELOGGEN_FPS", "30")))
            vel_limit = float(os.environ.get(
                "OPENARM_GRIPPER_SIM_VEL_LIMIT",
                os.environ.get("OPENARM_GRIPPER_MAX_VELOCITY", "0.10"),
            ))
            if fps <= 0 or vel_limit <= 0:
                return None
            max_delta = vel_limit / fps
        else:
            steps = int(os.environ.get(
                "OPENARM_GRIPPER_TARGET_INTERP_STEPS",
                os.environ.get("ELOGGEN_GRIPPER_TARGET_INTERP_STEPS", "12"),
            ))
            if steps <= 1:
                return None
            close_width, open_width = None, None
            max_delta = 2.0 / float(steps)
        return {
            "physical_target": physical_target,
            "close_width": close_width,
            "open_width": open_width,
            "current": {
                "left": self._openarm_gripper_hold_action(robot, "left", physical_target=physical_target),
                "right": self._openarm_gripper_hold_action(robot, "right", physical_target=physical_target),
            },
            "max_delta": max_delta,
            "filter_enabled": os.environ.get("OPENARM_GRIPPER_TARGET_FILTER", "1") != "0",
            "filter_weights": np.asarray([0.5, 0.3, 0.2], dtype=np.float32),
            "history": {"left": [], "right": []},
        }

    def _openarm_gripper_target_to_physical(self, interp_state, target):
        close_width = float(interp_state["close_width"])
        open_width = float(interp_state["open_width"])
        lower = min(close_width, open_width)
        upper = max(close_width, open_width)
        if lower - 1e-6 <= target <= upper + 1e-6:
            return float(np.clip(target, lower, upper))

        semantic_target = float(np.clip(target, -1.0, 1.0))
        open_alpha = (semantic_target - (-1.0)) / (1.0 - (-1.0))
        return close_width + open_alpha * (open_width - close_width)

    def _filter_openarm_gripper_target(self, interp_state, arm, target):
        if not interp_state.get("filter_enabled", False):
            return target

        history = interp_state["history"].setdefault(arm, [])
        history.append(target)
        max_len = int(len(interp_state["filter_weights"]))
        if len(history) > max_len:
            del history[0:len(history) - max_len]

        weights = interp_state["filter_weights"][-len(history):].astype(np.float32)
        weights = weights / np.sum(weights)
        return float(np.sum(np.asarray(history, dtype=np.float32) * weights))

    def _smooth_openarm_gripper_target(self, interp_state, arm, target):
        target = float(np.asarray(target).reshape(-1)[0])
        if interp_state is None:
            return target
        if interp_state.get("physical_target", False):
            target = self._openarm_gripper_target_to_physical(interp_state, target)
            target = self._filter_openarm_gripper_target(interp_state, arm, target)

        current_by_arm = interp_state["current"]
        prev = current_by_arm.get(arm)
        if prev is None:
            current_by_arm[arm] = target
            return target

        delta = target - prev
        max_delta = float(interp_state["max_delta"])
        if abs(delta) <= max_delta:
            value = target
        else:
            value = prev + np.sign(delta) * max_delta
        current_by_arm[arm] = value
        return value

    def _set_openarm_gripper_target(self, action, index, arm, target, interp_state=None):
        action[index] = self._smooth_openarm_gripper_target(
            interp_state=interp_state,
            arm=arm,
            target=target,
        )
        return action

    def _smooth_openarm_action_grippers(self, action, robot, interp_state, arms=("left", "right")):
        if interp_state is None:
            return action
        for arm in arms:
            gripper_action_idx = self._tensor_indices_to_numpy(robot.gripper_action_idx[arm])
            self._set_openarm_gripper_target(
                action=action,
                index=gripper_action_idx,
                arm=arm,
                target=action[gripper_action_idx],
                interp_state=interp_state,
            )
        return action

    def _openarm_has_attached_object(self, attached_obj, arm):
        if not attached_obj or not isinstance(attached_obj, dict):
            return False
        if attached_obj.get(arm) is not None:
            return True
        if attached_obj.get(f"arm_{arm}") is not None:
            return True
        # obtain_attached_object() stores entries by eef link name for CuRobo,
        # e.g. "left_eef_link" / "right_eef_link", so recognize those too.
        for key, value in attached_obj.items():
            if value is not None and arm in str(key):
                return True
        return False

    def _openarm_gripper_target_requests_open(self, interp_state, target):
        target = float(np.asarray(target).reshape(-1)[0])
        if interp_state is not None and interp_state.get("physical_target", False):
            target = self._openarm_gripper_target_to_physical(interp_state, target)
            close_width = float(interp_state["close_width"])
            open_width = float(interp_state["open_width"])
            return abs(target - open_width) < abs(target - close_width)
        return target > 0.0

    def _openarm_should_hold_attached_grippers(self, phase_type, reference_action):
        phase_name = str(phase_type or "").strip().lower()
        hold_phases = {
            item.strip().lower()
            for item in os.environ.get(
                "OPENARM_GRIPPER_HOLD_ATTACHED_PHASES",
                "transport,to_bag,move_to_bag,place",
            ).split(",")
            if item.strip()
        }
        if phase_name not in hold_phases:
            return False
        if phase_name == "place" and reference_action is None:
            return False
        return True

    def _hold_openarm_attached_grippers(
        self,
        action,
        robot,
        attached_obj,
        interp_state,
        phase_type=None,
        reference_action=None,
    ):
        if not self._openarm_continuous_gripper_enabled():
            return action
        if os.environ.get("OPENARM_GRIPPER_HOLD_ATTACHED", "1") == "0":
            return action
        if not self._openarm_should_hold_attached_grippers(phase_type, reference_action):
            return action
        for arm in ("left", "right"):
            if self._openarm_has_attached_object(attached_obj, arm):
                gripper_action_idx = self._tensor_indices_to_numpy(robot.gripper_action_idx[arm])
                if reference_action is not None and self._openarm_gripper_target_requests_open(
                    interp_state,
                    np.asarray(reference_action)[gripper_action_idx],
                ):
                    continue
                self._set_openarm_gripper_target(
                    action=action,
                    index=gripper_action_idx,
                    arm=arm,
                    target=-1.0,
                    interp_state=interp_state,
                )
        return action

    def _close_openarm_pick_grippers(self, action, robot, object_ref, phase_type, interp_state):
        if not self._openarm_continuous_gripper_enabled():
            return action
        if os.environ.get("OPENARM_GRIPPER_CLOSE_FROM_PICK_START", "1") == "0":
            return action
        if str(phase_type or "").strip().lower() != "pick":
            return action
        if not object_ref or not isinstance(object_ref, dict):
            return action
        for arm in ("left", "right"):
            if object_ref.get(f"arm_{arm}") is not None:
                gripper_action_idx = self._tensor_indices_to_numpy(robot.gripper_action_idx[arm])
                self._set_openarm_gripper_target(
                    action=action,
                    index=gripper_action_idx,
                    arm=arm,
                    target=-1.0,
                    interp_state=interp_state,
                )
        return action

    def _lock_openarm_attached_gripper_state(self, robot, attached_obj, action, interp_state):
        if not self._openarm_continuous_gripper_enabled():
            return action
        if os.environ.get("OPENARM_GRIPPER_LOCK_ATTACHED_STATE", "1") == "0":
            return action
        if not attached_obj:
            return action
        close_width, _ = self._openarm_gripper_width_limits()
        locked_any = False
        for arm in ("left", "right"):
            if not self._openarm_has_attached_object(attached_obj, arm):
                continue
            gripper_action_idx = self._tensor_indices_to_numpy(robot.gripper_action_idx[arm])
            if action is not None and self._openarm_gripper_target_requests_open(
                interp_state,
                np.asarray(action)[gripper_action_idx],
            ):
                continue
            gripper_control_idx = robot.gripper_control_idx[arm]
            if action is not None:
                action[gripper_action_idx] = close_width if self._openarm_gripper_physical_target_enabled() else -1.0

            lock_pos = th.full(
                (len(gripper_control_idx),),
                float(close_width),
                dtype=robot.get_joint_positions().dtype,
                device=robot.get_joint_positions().device,
            )
            robot.set_joint_positions(lock_pos, indices=gripper_control_idx, drive=False)
            robot.set_joint_positions(lock_pos, indices=gripper_control_idx, drive=True)
            locked_any = True

        if (
            locked_any
            and os.environ.get("OPENARM_GRIPPER_LOCK_DEBUG", "0") != "0"
            and not getattr(self, "_openarm_gripper_lock_debug_printed", False)
        ):
            print("[openarm_gripper] attached gripper state lock active")
            self._openarm_gripper_lock_debug_printed = True
        return action

    def _openarm_gripper_target_to_width(self, target):
        close_width, open_width = self._openarm_gripper_width_limits()
        target = float(np.asarray(target).reshape(-1)[0])
        lower = min(close_width, open_width)
        upper = max(close_width, open_width)
        if lower - 1e-6 <= target <= upper + 1e-6:
            return float(np.clip(target, lower, upper))
        semantic_target = float(np.clip(target, -1.0, 1.0))
        open_alpha = (semantic_target + 1.0) / 2.0
        return float(close_width + open_alpha * (open_width - close_width))

    def _force_openarm_gripper_width(self, robot, action, arm, width, force_state=True):
        gripper_action_idx = self._tensor_indices_to_numpy(robot.gripper_action_idx[arm])
        if action is not None:
            action[gripper_action_idx] = width if self._openarm_gripper_physical_target_enabled() else -1.0
        gripper_control_idx = robot.gripper_control_idx[arm]
        joint_positions = robot.get_joint_positions()
        lock_pos = th.full(
            (len(gripper_control_idx),),
            float(width),
            dtype=joint_positions.dtype,
            device=joint_positions.device,
        )
        if force_state:
            robot.set_joint_positions(lock_pos, indices=gripper_control_idx, drive=False)
        robot.set_joint_positions(lock_pos, indices=gripper_control_idx, drive=True)
        return action

    def _openarm_replay_close_delay(self, object_ref, arm):
        if not object_ref or not isinstance(object_ref, dict):
            return 0
        obj = object_ref.get(f"arm_{arm}")
        if obj is None:
            return 0
        obj_name = str(obj)
        if obj_name == "object_1":
            return int(os.environ.get(
                "OPENARM_GRIPPER_REPLAY_OBJECT_1_CLOSE_DELAY",
                os.environ.get("OPENARM_GRIPPER_REPLAY_APPLE_CLOSE_DELAY", "8"),
            ))
        if obj_name == "object_2":
            return int(os.environ.get(
                "OPENARM_GRIPPER_REPLAY_OBJECT_2_CLOSE_DELAY",
                os.environ.get("OPENARM_GRIPPER_REPLAY_LEMON_CLOSE_DELAY", "0"),
            ))
        return 0

    def _make_openarm_replay_gripper_schedule(self, src_actions, robot, mp_end_steps, object_ref=None):
        if not self._openarm_continuous_gripper_enabled():
            return {}
        if src_actions is None or len(src_actions) == 0:
            return {}
        ramp_steps = int(os.environ.get("OPENARM_GRIPPER_REPLAY_RAMP_STEPS", "12"))
        if ramp_steps <= 0:
            return {}
        schedule = {}
        src_actions = np.asarray(src_actions)
        for arm_i, arm in enumerate(("left", "right")):
            gripper_action_idx = self._tensor_indices_to_numpy(robot.gripper_action_idx[arm])
            values = np.asarray(src_actions[:, gripper_action_idx]).reshape(len(src_actions), -1)[:, 0]
            changes = np.where(np.diff(values) != 0)[0] + 1
            mp_end = int(mp_end_steps[arm_i]) if mp_end_steps is not None else 0
            changes = [int(i) for i in changes if int(i) >= mp_end]
            if not changes:
                continue
            switch_i = changes[0]
            old_width = self._openarm_gripper_target_to_width(values[switch_i - 1])
            new_width = self._openarm_gripper_target_to_width(values[switch_i])
            is_closing = new_width < old_width
            delay = 0
            if is_closing:
                delay = self._openarm_replay_close_delay(object_ref, arm)
                switch_i += delay
            schedule[arm] = {
                "switch_i": switch_i,
                "rel_to_replay": switch_i - mp_end,
                "ramp_steps": ramp_steps,
                "old_width": old_width,
                "new_width": new_width,
                "is_closing": is_closing,
                "delay": delay,
            }
        if (
            schedule
            and os.environ.get("OPENARM_GRIPPER_REPLAY_SCHEDULE_DEBUG", "0") != "0"
            and not getattr(self, "_openarm_gripper_replay_schedule_debug_printed", False)
        ):
            print(f"[openarm_gripper] replay schedule: {schedule}")
            self._openarm_gripper_replay_schedule_debug_printed = True
        return schedule

    def _apply_openarm_replay_gripper_schedule(self, robot, action, replay_i, schedule, replay_relative=False):
        if not schedule or os.environ.get("OPENARM_GRIPPER_REPLAY_SCHEDULE", "1") == "0":
            return action
        for arm, spec in schedule.items():
            switch_i = int(spec["rel_to_replay"] if replay_relative else spec["switch_i"])
            if switch_i < 0:
                continue
            ramp_steps = int(spec["ramp_steps"])
            old_width = float(spec["old_width"])
            new_width = float(spec["new_width"])
            if replay_i < switch_i:
                width = old_width
                in_ramp = False
                before_switch = True
            elif replay_i >= switch_i + ramp_steps:
                width = new_width
                in_ramp = False
                before_switch = False
            else:
                alpha = float(replay_i - switch_i + 1) / float(ramp_steps)
                width = old_width + alpha * (new_width - old_width)
                in_ramp = True
                before_switch = False
            # Replay gripper scheduling should only shape the commanded target.
            # Do not teleport the actual joint state at the end of the ramp; if
            # the object is attached, the following MP/transport phase can lock
            # the gripper state explicitly via OPENARM_GRIPPER_LOCK_ATTACHED_STATE.
            force_state = os.environ.get("OPENARM_GRIPPER_REPLAY_FORCE_STATE", "0") != "0"
            if (
                before_switch
                and bool(spec.get("is_closing", False))
                and os.environ.get("OPENARM_GRIPPER_REPLAY_LOCK_BEFORE_CLOSE", "1") != "0"
            ):
                force_state = True
            action = self._force_openarm_gripper_width(
                robot=robot,
                action=action,
                arm=arm,
                width=width,
                force_state=force_state,
            )
        return action

    def _freeze_openarm_source_replay_action(self, action, robot, frozen_arms):
        if not frozen_arms:
            return action

        joint_positions = robot.get_joint_positions()
        for arm in frozen_arms:
            arm_action_idx = self._tensor_indices_to_numpy(robot.arm_action_idx[arm])
            arm_control_idx = robot.arm_control_idx[arm]
            action[arm_action_idx] = joint_positions[arm_control_idx].detach().cpu().numpy()

            gripper_action_idx = self._tensor_indices_to_numpy(robot.gripper_action_idx[arm])
            action[gripper_action_idx] = self._openarm_gripper_hold_action(robot, arm)

        return action

    def _append_joint_space_target(self, q_traj, q_target, max_inter_dist):
        if q_target is None:
            return q_traj

        q_traj = q_traj.cpu()
        q_target = q_target.to(dtype=q_traj.dtype)
        q_start = q_traj[-1]
        max_delta = th.max(th.abs(q_target - q_start)).item()
        if max_delta <= 1e-6:
            return q_traj

        num_steps = max(1, int(np.ceil(max_delta / max(max_inter_dist, 1e-6))))
        alphas = th.linspace(1.0 / num_steps, 1.0, num_steps, dtype=q_traj.dtype).view(-1, 1)
        bridge = q_start.view(1, -1) + alphas * (q_target - q_start).view(1, -1)
        return th.cat([q_traj, bridge], dim=0)

    def __len__(self):
        # sum up length of all waypoint sequences
        return sum(len(s) for s in self.waypoint_sequences)

    def __getitem__(self, ind):
        """
        Returns waypoint at time index.
        
        Returns:
            waypoint (Waypoint instance)
        """
        assert len(self.waypoint_sequences) > 0
        assert (ind >= 0) and (ind < len(self))

        # find correct waypoint sequence we should index
        end_ind = 0
        for seq_ind in range(len(self.waypoint_sequences)):
            start_ind = end_ind
            end_ind += len(self.waypoint_sequences[seq_ind])
            if (ind >= start_ind) and (ind < end_ind):
                break

        # index within waypoint sequence
        return self.waypoint_sequences[seq_ind][ind - start_ind]

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        return self.waypoint_sequences[-1].last_waypoint

    def add_waypoint_sequence(self, sequence):
        """
        Directly append sequence to list (no interpolation).

        Args:
            sequence (WaypointSequence instance): sequence to add
        """
        assert isinstance(sequence, WaypointSequence)
        self.waypoint_sequences.append(sequence)

    def add_waypoint_sequence_for_target_pose(
        self,
        pose,
        gripper_action,
        num_steps,
        skip_interpolation=False,
        action_noise=0.,
        bimanual=False,
    ):
        """
        Adds a new waypoint sequence corresponding to a desired target pose. A new WaypointSequence
        will be constructed consisting of @num_steps intermediate Waypoint objects. These can either
        be constructed with linear interpolation from the last waypoint (default) or be a
        constant set of target poses (set @skip_interpolation to True).

        Args:
            pose (np.array): 4x4 target pose

            gripper_action (np.array): value for gripper action

            num_steps (int): number of action steps when trying to reach this waypoint. Will
                add intermediate linearly interpolated points between the last pose on this trajectory
                and the target pose, so that the total number of steps is @num_steps.

            skip_interpolation (bool): if True, keep the target pose fixed and repeat it @num_steps
                times instead of using linearly interpolated targets.

            action_noise (float): scale of random gaussian noise to add during action execution (e.g.
                when @execute is called)
        """
        if (len(self.waypoint_sequences) == 0):
            assert skip_interpolation, "cannot interpolate since this is the first waypoint sequence"

        if skip_interpolation:
            # repeat the target @num_steps times
            assert num_steps is not None
            poses = np.array([pose for _ in range(num_steps)])
            gripper_actions = np.array([[gripper_action] for _ in range(num_steps)])
        else:
            # linearly interpolate between the last pose and the new waypoint
            last_waypoint = self.last_waypoint
            if last_waypoint.pose.shape[0] == 8:
                # here is when transforming the two arms altogher, should be corresponding to the bimanual-coordinated phase
                poses_left, num_steps_2_left = PoseUtils.interpolate_poses(
                    pose_1=last_waypoint.pose[0:4, :],
                    pose_2=pose[0:4, :],
                    num_steps=num_steps,
                )
                poses_right, num_steps_2_right = PoseUtils.interpolate_poses(
                    pose_1=last_waypoint.pose[4:, :],
                    pose_2=pose[4:, :],
                    num_steps=num_steps,
                )
                poses = np.concatenate([poses_left, poses_right], axis=1)
                assert num_steps_2_left == num_steps_2_right
                num_steps_2 = num_steps_2_left
            else:
                # suitable for single arm transformation
                poses, num_steps_2 = PoseUtils.interpolate_poses(
                    pose_1=last_waypoint.pose,
                    pose_2=pose,
                    num_steps=num_steps,
                )
            assert num_steps == num_steps_2
            gripper_actions = np.array([gripper_action for _ in range(num_steps + 2)])
            # make sure to skip the first element of the new path, which already exists on the current trajectory path
            poses = poses[1:]
            gripper_actions = gripper_actions[1:]

        # add waypoint sequence for this set of poses
        sequence = WaypointSequence.from_poses(
            poses=poses,
            gripper_actions=gripper_actions,
            action_noise=action_noise,
        )
        self.add_waypoint_sequence(sequence)

    def pop_first(self):
        """
        Removes first waypoint in first waypoint sequence and returns it. If the first waypoint
        sequence is now empty, it is also removed.

        Returns:
            waypoint (Waypoint instance)
        """
        first, rest = self.waypoint_sequences[0].split(1)
        if len(rest) == 0:
            # remove empty waypoint sequence
            self.waypoint_sequences = self.waypoint_sequences[1:]
        else:
            # update first waypoint sequence
            self.waypoint_sequences[0] = rest
        return first

    def merge(
        self,
        other,
        num_steps_interp=None,
        num_steps_fixed=None,
        action_noise=0.,
        bimanual=False,
    ):
        """
        Merge this trajectory with another (@other).

        Args:
            other (WaypointTrajectory object): the other trajectory to merge into this one

            num_steps_interp (int or None): if not None, add a waypoint sequence that interpolates
                between the end of the current trajectory and the start of @other

            num_steps_fixed (int or None): if not None, add a waypoint sequence that has constant 
                target poses corresponding to the first target pose in @other

            action_noise (float): noise to use during the interpolation segment
        """
        need_interp = (num_steps_interp is not None) and (num_steps_interp > 0)
        need_fixed = (num_steps_fixed is not None) and (num_steps_fixed > 0)
        use_interpolation_segment = (need_interp or need_fixed)

        if use_interpolation_segment:
            # pop first element of other trajectory
            other_first = other.pop_first()

            # Get first target pose of other trajectory.
            # The interpolated segment will include this first element as its last point.
            target_for_interpolation = other_first[0]

            if need_interp:
                # interpolation segment
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose, # 8x4
                    gripper_action=target_for_interpolation.gripper_action, #2,
                    num_steps=num_steps_interp,
                    action_noise=action_noise,
                    skip_interpolation=False,
                    bimanual=bimanual,
                )

            if need_fixed:
                # segment of constant target poses equal to @other's first target pose

                # account for the fact that we pop'd the first element of @other in anticipation of an interpolation segment
                num_steps_fixed_to_use = num_steps_fixed if need_interp else (num_steps_fixed + 1)
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_fixed_to_use,
                    action_noise=action_noise,
                    skip_interpolation=True,
                    bimanual=bimanual,
                )

            # make sure to preserve noise from first element of other trajectory
            self.waypoint_sequences[-1][-1].noise = target_for_interpolation.noise

        # concatenate the trajectories
        self.waypoint_sequences += other.waypoint_sequences

    def _pad_tensors(self, tensor1, tensor2):
        M, _ = tensor1.shape
        N, _ = tensor2.shape
        max_size = max(M, N)

        def pad_tensor(tensor, size):
            if tensor.shape[0] < size:
                last_row = tensor[-1].unsqueeze(0)  # Extract last row
                repeat_count = size - tensor.shape[0]
                padding = last_row.repeat(repeat_count, 1)  # Repeat last row
                tensor = th.cat([tensor, padding], dim=0)
            return tensor

        tensor1 = pad_tensor(tensor1, max_size)
        tensor2 = pad_tensor(tensor2, max_size)

        return tensor1, tensor2
 
    def _subsample_tensor(self, tensor, num_samples=8):
        N = tensor.shape[0]

        if N <= num_samples:
            return tensor  # If N is less than or equal to num_samples, return as is

        indices = th.linspace(0, N - 1, steps=num_samples).long()  # Evenly spaced indices
        return tensor[indices]
    
    def downsample_replay_traj(self, left_replay_waypoints, right_reaplay_waypoints, ds_ratio=1, asyn_ds_ratio=True):
        # downsample the replay waypoints to reduce the hesitation problem
        if ds_ratio == 1 or ds_ratio is None:
            return left_replay_waypoints, right_reaplay_waypoints

        ds_ratio = int(ds_ratio)

        def _keep_final_waypoint(original, downsampled):
            if not original:
                return downsampled
            if not downsampled:
                return [original[-1]]
            if downsampled[-1] is not original[-1]:
                downsampled.append(original[-1])
            return downsampled

        def _downsample_one_arm(waypoints, gripper_idx):
            if len(waypoints) <= 1:
                return waypoints
            if not asyn_ds_ratio:
                return _keep_final_waypoint(waypoints, waypoints[::ds_ratio])

            gripper_actions = np.array([waypoint.gripper_action[gripper_idx] for waypoint in waypoints])
            gripper_action_changes = np.where(np.diff(gripper_actions) != 0)[0]
            grasp_start_idx = int(gripper_action_changes[0] + 1) if gripper_action_changes.size > 0 else len(waypoints)

            before_grasp = waypoints[:grasp_start_idx:ds_ratio]
            after_grasp = waypoints[grasp_start_idx::2]
            return _keep_final_waypoint(waypoints, before_grasp + after_grasp)

        return (
            _downsample_one_arm(left_replay_waypoints, 0),
            _downsample_one_arm(right_reaplay_waypoints, 1),
        )

    def _is_place_phase(self, object_ref, attached_obj):
        if object_ref is None or attached_obj is None:
            return False

        target_refs = [object_ref.get("arm_left"), object_ref.get("arm_right")]
        held_objects = []
        if isinstance(attached_obj, dict):
            held_objects = [attached_obj.get("left"), attached_obj.get("right")]

        has_held_object = any(obj is not None for obj in held_objects)
        targets_container = any(
            target is not None and str(target).startswith("paper_bag_")
            for target in target_refs
        )
        return has_held_object and targets_container

    def _arm_mp_max_inter_dist(self, object_ref=None, attached_obj=None, retract=False, env=None):
        if retract:
            env_name = getattr(env, "name", None) or getattr(getattr(env, "env", None), "name", "")
            if str(env_name).startswith("openarm_real_exp_1"):
                return float(os.environ.get(
                    "OPENARM_REAL_EXP_1_RETRACT_MP_MAX_INTER_DIST",
                    os.environ.get("ELOGGEN_RETRACT_MP_MAX_INTER_DIST", "0.015"),
                ))

            return float(os.environ.get(
                "ELOGGEN_RETRACT_MP_MAX_INTER_DIST",
                os.environ.get("ELOGGEN_ARM_MP_MAX_INTER_DIST", "0.03"),
            ))

        if self._is_place_phase(object_ref=object_ref, attached_obj=attached_obj):
            return float(os.environ.get(
                "ELOGGEN_PLACE_MP_MAX_INTER_DIST",
                os.environ.get("ELOGGEN_ARM_MP_MAX_INTER_DIST", "0.02"),
            ))

        return float(os.environ.get("ELOGGEN_ARM_MP_MAX_INTER_DIST", "0.01"))
    
    def setup_phase_logs(self, phase_type, baseline=None):
        current_phase_logs = dict()
        current_phase_logs["phase_type"] = phase_type
        current_phase_logs["base_sampling_time"] = dict()
        current_phase_logs["base_mp_planning_time"] = dict()
        current_phase_logs["base_mp_execution_time"] = dict()

        current_phase_logs["arm_mp_planning_time"] = dict()
        current_phase_logs["arm_mp_execution_time"] = dict()
        current_phase_logs["arm_replay_execution_time"] = dict()
        if baseline == "mimicgen":
            current_phase_logs["arm_interp_execution_time"] = dict()

        current_phase_logs["full_retract_mp_planning_time"] = dict()
        current_phase_logs["full_retract_mp_execution_time"] = dict()
        current_phase_logs["torso_retract_mp_planning_time"] = dict()
        current_phase_logs["torso_retract_mp_execution_time"] = dict()
        current_phase_logs["full_retract_mp_err"] = dict()
        current_phase_logs["torso_retract_mp_err"] = dict()

        current_phase_logs["visibility_stats"] = dict()

        return current_phase_logs
    
    def obtain_attached_object(self, env, robot):
        grasp_action = {"left": 1.0, "right": 1.0}
        attached_obj = {}
        attached_obj_scale = {}
        for local_arm_side in ["left", "right"]:  
            eef_link_name = robot.eef_link_names.get(local_arm_side, f"{local_arm_side}_eef_link")
            is_grasping = robot.is_grasping(arm=local_arm_side)
            # print("local_arm_side is_grasping: ", local_arm_side, is_grasping)
            if is_grasping == og.controllers.IsGraspingState.TRUE: 
                grasp_action[local_arm_side] = -1.0
                # Find the object that the robot is grapsing in that arm
                task_relevant_objs = env._get_task_relevant_objs()
                for task_relevant_obj in task_relevant_objs:
                    # TODO: remove the stationay object hardcoding. Make it more general
                    if all(keyword not in task_relevant_obj.name for keyword in ["table", "shelf", "bar", "sink"]):
                        is_grasping_candidate_obj = robot.is_grasping(arm=local_arm_side, candidate_obj=task_relevant_obj)
                        # print("local_arm_side is_grasping_candidate_obj: ", local_arm_side, is_grasping_candidate_obj, task_relevant_obj.root_link.name) 
                        if is_grasping_candidate_obj == og.controllers.IsGraspingState.TRUE:
                            print(f"arm {local_arm_side} is_grasping {task_relevant_obj.root_link.name}") 
                            attached_obj[eef_link_name] = task_relevant_obj.root_link
                            attached_obj_scale[eef_link_name] = 0.9
                            # robot can only be holding one object at a time
                            break
        retval = dict(
            grasp_action=grasp_action,
            attached_obj=attached_obj,
            attached_obj_scale=attached_obj_scale,
        )
        return retval
    
    def reset_visibility_counter(self, env):
        """
        Reset the visibility counter for each sensor.
        """
        for sensor_name, sensor in env.robot.sensors.items():
            if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                shortened_sensor_name = sensor_name.split(":")[1]
                env.num_frames_with_obj_visible[shortened_sensor_name] = 0
        env.num_frames_with_obj_visible["any"] = 0

    def check_ref_obj_visibility(self, env, obs, obs_info, ref_obj):
        any_visible = False
        for sensor_name, sensor in env.robot.sensors.items():
            if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                shortened_sensor_name = sensor_name.split(":")[1]
                seg_instance = obs[f"{env.robot_name}::{sensor_name}::seg_instance"]
                seg_instance_info = obs_info[f"{env.robot_name}"][sensor_name]["seg_instance"]
                obj_key = next((key for key, value in seg_instance_info.items() if value == ref_obj.name), None)
                if obj_key is None:
                    count = 0
                    # if shortened_sensor_name == "eyes":
                    #     print("not found")
                else:
                    count = (seg_instance == obj_key).sum().item()
                    # if shortened_sensor_name == "eyes":
                    #     print("found")
                if count > 0:
                    env.num_frames_with_obj_visible[shortened_sensor_name] += 1
                    any_visible = True

        if any_visible:
            env.num_frames_with_obj_visible["any"] += 1

    def execute_baseline(
        self, 
        env,
        env_interface, 
        render=False, 
        video_writer=None, 
        video_skip=5, 
        camera_names=None,
        bimanual=False,
        cur_subtask_end_step_MP=None,
        attached_obj=None,
        phase_type=None,
        object_ref=None,
        grasp_init_views_video_writer=None,
        enable_marker_vis=False,
        ds_ratio=1,
        phase_logs=None,
        retract_type=None,
        src_curr_phase_actions=None,
        baseline=None,
    ):
        ref_object = self._select_reference_object(object_ref)
        ref_obj = self._resolve_tracking_object(env=env, env_interface=env_interface, ref_object=ref_object)
        if ref_obj is not None:
            print("ref_obj: ", ref_obj.name)
        robot = env.env.robots[0]
        gripper_interp_state = self._make_openarm_gripper_interp_state(env=env, robot=robot)
        
        # TODO: implement early stopping on 1. collision 2. attached object misatch
        if phase_type == "navigation":
            phase_logs[env.execution_phase_ind] = self.setup_phase_logs(phase_type=phase_type)
            init_state = og.sim.dump_state()
            local_env_step = 0
            states = []
            actions = []
            observations = []
            observations_info = []
            datagen_infos = []
            success = {"task": False}
            init_global_env_step = env.global_env_step
            nav_execution_start_time = time.time()
            init_arm_left_pos = robot.get_joint_positions()[robot.arm_control_idx["left"]]
            init_arm_right_pos = robot.get_joint_positions()[robot.arm_control_idx["right"]]
            for temp_idx, src_action in enumerate(src_curr_phase_actions):
                
                # To skip initial stationary actions during human data collection
                if env.execution_phase_ind == 0 and temp_idx < env.start_nav_step:
                    continue
                action = env.primitive._empty_action()
                action[robot.base_action_idx] = th.tensor(src_action[robot.base_action_idx], dtype=th.float32)
                action[robot.arm_action_idx["left"]] = init_arm_left_pos
                action[robot.arm_action_idx["right"]] = init_arm_right_pos
                if attached_obj["left"] is not None:
                    self._set_openarm_gripper_target(
                        action, robot.gripper_action_idx["left"], "left", -1, gripper_interp_state,
                    )
                if attached_obj["right"] is not None:
                    self._set_openarm_gripper_target(
                        action, robot.gripper_action_idx["right"], "right", -1, gripper_interp_state,
                    )
                state = env.get_state()["states"]
                obs, obs_info = env.get_obs_IL()
                datagen_info = env_interface.get_datagen_info(action=action)
                env.step(action, video_writer)
                local_env_step += 1
                env.global_env_step += 1
                states.append(state)
                actions.append(action)
                observations.append(obs)
                observations_info.append(json.dumps(obs_info))
                datagen_infos.append(datagen_info)
                # Check reference object visibility
                if ref_obj is not None:
                    self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)

            # apply a zero action
            action = env.primitive._empty_action()
            action[robot.base_action_idx] = th.tensor([0.0, 0.0, 0.0], dtype=th.float32)
            action[robot.arm_action_idx["left"]] = init_arm_left_pos
            action[robot.arm_action_idx["right"]] = init_arm_right_pos
            env.step(action, video_writer)

            nav_execution_finish_time = time.time()
            phase_logs[env.execution_phase_ind]["base_mp_execution_time"][0] = round(nav_execution_finish_time - nav_execution_start_time, 2)
            print("nav execution time: ", phase_logs[env.execution_phase_ind]["base_mp_execution_time"][0])

            num_phase_steps = env.global_env_step - init_global_env_step
            for sensor_name, sensor in env.robot.sensors.items():
                if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                    shortened_sensor_name = sensor_name.split(":")[1]
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_{shortened_sensor_name}"]= 0
                    print(f"Visibility stats for nav_repeat {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_{shortened_sensor_name}"])
            if num_phase_steps > 0:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
            else:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_any"]= 0
            phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_steps"] = num_phase_steps
            print(f"Visibility stats for nav_repeat any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_repeat_any"])

            MP_end_step_local_list = [cur_subtask_end_step_MP[0], cur_subtask_end_step_MP[1]]
            left_mp_ranges = [0, 0]
            right_mp_ranges = [0, 0]
            results = dict(
                states=states,
                observations=observations,
                datagen_infos=datagen_infos,
                actions=np.array(actions),
                success=bool(success["task"]),
                mp_end_steps=MP_end_step_local_list,
                subtask_lengths=local_env_step,
                left_mp_ranges=left_mp_ranges,
                right_mp_ranges=right_mp_ranges,
                retry_nav=False,
                observations_info=observations_info
            )
            # execution_phase_ind keeps track of each phase that was tried to be executed (even if MP failed for that phase). 
            # In this case MP succeeded and phase was actually executed
            env.execution_phase_ind += 1
            env.phases_completed_wo_mp_err += 1
            return results

        else:
            phase_logs[env.execution_phase_ind] = self.setup_phase_logs(phase_type=phase_type, baseline=baseline)
            local_env_step = 0
            states = []
            actions = []
            observations = []
            observations_info = []
            datagen_infos = []
            success = {"task": False}

            assert len(self.waypoint_sequences) == 1
            seq = self.waypoint_sequences[0]
            for end_step in cur_subtask_end_step_MP:
                assert 0 <= end_step <= len(seq)

            # Segment the waypoints into motion planner waypoints and replay waypoints
            left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
            left_replay_waypoints = seq[cur_subtask_end_step_MP[0]:]
            right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
            right_replay_waypoints = seq[cur_subtask_end_step_MP[1]:]
            if ds_ratio not in (None, 1):
                left_replay_waypoints, right_replay_waypoints = self.downsample_replay_traj(
                    left_replay_waypoints=left_replay_waypoints,
                    right_reaplay_waypoints=right_replay_waypoints,
                    ds_ratio=int(ds_ratio),
                )

            # print("left_mp_waypoints", len(left_mp_waypoints))
            # print("left_replay_waypoints", len(left_replay_waypoints))
            # print("right_mp_waypoints", len(right_mp_waypoints))
            # print("right_replay_waypoints", len(right_replay_waypoints))

            # Get the last waypoint for padding later
            last_waypoint = seq[-1]

            # 1. make sure the gripper actions are the same
            # 2. get the last waypoint's pose and orientation as the MP target
            # Otherwise, use the current eef pose as the MP target
            if len(left_mp_waypoints) > 0:
                gripper_actions = np.array([waypoint.gripper_action for waypoint in left_mp_waypoints])
                # This is not necessarily true since while teleopating as a non-optimal teleoperator, I inadvertently would toggle gripper on / off
                # Specially when trying to grasp. So removed this assertion
                # assert (gripper_actions[:, 0] == gripper_actions[0, 0]).all()
                left_waypoint = left_mp_waypoints[-1]
                left_gripper_action = left_waypoint.gripper_action
                left_waypoint_pos, left_waypoint_ori = th.tensor(left_waypoint.pose[0:3, 3]), T.mat2quat(th.tensor(left_waypoint.pose[0:3, 0:3]))
            else:
                left_gripper_action = None
                left_waypoint_pos, left_waypoint_ori = robot.get_eef_pose("left")

            if len(right_mp_waypoints) > 0:
                gripper_actions = np.array([waypoint.gripper_action for waypoint in right_mp_waypoints])
                # This is not necessarily true since while teleopating as a non-optimal teleoperator, I inadvertently would toggle gripper on / off
                # Specially when trying to grasp. So removed this assertion
                # assert (gripper_actions[:, 1] == gripper_actions[0, 1]).all()
                right_waypoint = right_mp_waypoints[-1]
                right_gripper_action = right_waypoint.gripper_action
                right_waypoint_pos, right_waypoint_ori = th.tensor(right_waypoint.pose[4:7, 3]), T.mat2quat(th.tensor(right_waypoint.pose[4:7, 0:3]))
            else:
                right_gripper_action = None
                right_waypoint_pos, right_waypoint_ori = robot.get_eef_pose("right")

            # If baseline is mimicgen, perform interpolation + replay
            if baseline == "mimicgen":
                # ========================================= ARM INTERPOLATION START =============================================
                step_size = 0.005
                current_left_eef_pose = robot.get_eef_pose("left")
                if object_ref["arm_left"] is None:
                    poses_left = th.tensor(T.pose2mat(current_left_eef_pose), dtype=th.float32).unsqueeze(0)
                else:
                    poses_left, _ = PoseUtils.interpolate_poses(
                        pose_1=T.pose2mat(current_left_eef_pose),
                        pose_2=th.tensor(left_waypoint.pose[:4], dtype=th.float32),
                        step_size=step_size,
                    )
                    poses_left = th.tensor(poses_left, dtype=th.float32)

                current_right_eef_pose = robot.get_eef_pose("right")
                if object_ref["arm_right"] is None:
                    poses_right = th.tensor(T.pose2mat(current_right_eef_pose), dtype=th.float32).unsqueeze(0)
                else:
                    poses_right, _ = PoseUtils.interpolate_poses(
                        pose_1=T.pose2mat(current_right_eef_pose),
                        pose_2=th.tensor(right_waypoint.pose[4:], dtype=th.float32),
                        step_size=step_size,
                    )
                    poses_right = th.tensor(poses_right, dtype=th.float32)
                
                if enable_marker_vis:
                    env.eef_current_marker_left.set_position_orientation(*current_left_eef_pose)
                    env.eef_current_marker_right.set_position_orientation(*current_right_eef_pose)
                    interp_target_left = T.mat2pose(poses_left[-1])
                    interp_target_right = T.mat2pose(poses_right[-1])
                    env.eef_goal_marker_left.set_position_orientation(*interp_target_left)
                    env.eef_goal_marker_right.set_position_orientation(*interp_target_right)

                
                print("len(poses_left): ", len(poses_left))
                print("len(poses_right): ", len(poses_right))
                
                # Perform padding
                if len(poses_left) < len(poses_right):
                    repeat_times = len(poses_right) - len(poses_left)
                    poses_left = th.cat((poses_left, poses_left[-1].repeat(repeat_times, 1, 1)))
                elif len(poses_right) < len(poses_left):
                    repeat_times = len(poses_left) - len(poses_right)
                    poses_right = th.cat((poses_right, poses_right[-1].repeat(repeat_times, 1, 1)))

                if len(poses_left) != len(poses_right):
                    assert len(poses_left) == len(poses_right)
                poses = np.concatenate([poses_left, poses_right], axis=1)

                init_global_env_step = env.global_env_step
                arm_interp_start_time = time.time()
                for pose in poses:
                    interp_action = env_interface.target_pose_to_action(target_pose=pose)

                    self._set_openarm_gripper_target(
                        interp_action, env_interface.gripper_action_dim[0], "left",
                        left_waypoint.gripper_action[0], gripper_interp_state,
                    )
                    self._set_openarm_gripper_target(
                        interp_action, env_interface.gripper_action_dim[1], "right",
                        right_waypoint.gripper_action[1], gripper_interp_state,
                    )

                    state = env.get_state()["states"]
                    obs, obs_info = env.get_obs_IL()
                    datagen_info = env_interface.get_datagen_info(action=interp_action)
                    env.step(interp_action, video_writer)
                    left_eef_pose = (pose[0:3, 3], T.mat2quat(th.tensor(pose[0:3, 0:3])))
                    right_eef_pose = (pose[4:7, 3], T.mat2quat(th.tensor(pose[4:7, 0:3])))
                    local_env_step += 1
                    env.global_env_step += 1
                    states.append(state)
                    actions.append(interp_action)
                    observations.append(obs)
                    observations_info.append(json.dumps(obs_info))
                    datagen_infos.append(datagen_info)
                    cur_success_metrics = env.is_success()
                    if ref_obj is not None:
                        self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                    for k in success:
                        success[k] = success[k] or cur_success_metrics[k]

                arm_interp_finish_time = time.time()
                phase_logs[env.execution_phase_ind]["arm_interp_execution_time"][0] = round(arm_interp_finish_time - arm_interp_start_time, 2)
                print("Time taken for arm interpolation: ", phase_logs[env.execution_phase_ind]["arm_interp_execution_time"][0])

                num_phase_steps = env.global_env_step - init_global_env_step
                for sensor_name, sensor in env.robot.sensors.items():
                    if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                        shortened_sensor_name = sensor_name.split(":")[1]
                        if num_phase_steps > 0:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                        else:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_{shortened_sensor_name}"]= 0
                        print(f"Visibility stats for arm_interp {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_{shortened_sensor_name}"])
                if num_phase_steps > 0:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                else:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_any"]= 0
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_steps"] = num_phase_steps
                print(f"Visibility stats for arm_interp any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_interp_any"])

                # Setting the interpolation ranges
                MP_end_step_local_list = [local_env_step, local_env_step]
                # Set the MP ranges to save to hdf5 file
                left_mp_ranges, right_mp_ranges = None, None
                if len(left_mp_waypoints) > 0:
                    left_mp_ranges = [init_global_env_step, env.global_env_step]
                if len(right_mp_waypoints) > 0:
                    right_mp_ranges = [init_global_env_step, env.global_env_step]
                # =============================================== ARM INTERPOLATION END ==================================================

            # If baseline is skillgen, perform mp + replay  
            elif baseline == "skillgen":
                # =============================================== Arm MP Planning =============================================
                
                # If at least one hand has motion planner waypoints, plan the motion
                if len(left_mp_waypoints) > 0 or len(right_mp_waypoints) > 0:
                    target_pos = {
                        robot.eef_link_names["left"]: left_waypoint_pos,
                        robot.eef_link_names["right"]: right_waypoint_pos,
                    }
                    target_quat = {
                        robot.eef_link_names["left"]: left_waypoint_ori,
                        robot.eef_link_names["right"]: right_waypoint_ori,
                    }
                    emb_sel = CuRoboEmbodimentSelection.ARM_NO_TORSO
                    arm_mp_max_inter_dist = self._arm_mp_max_inter_dist(
                        object_ref=object_ref,
                        attached_obj=attached_obj,
                    )
                    
                    # Use OG to know attached objects
                    retval = self.obtain_attached_object(env, robot)
                    attached_obj = retval["attached_obj"]
                    attached_obj_scale = retval["attached_obj_scale"]

                    # If one of the arm does not hav a ref object, remove it from the target pose of MP (will move this arm randomly in this case)
                    if object_ref["arm_right"] is None:
                        del target_pos["right_eef_link"]
                        del target_quat["right_eef_link"]
                    elif object_ref["arm_left"] is None:
                        del target_pos["left_eef_link"]
                        del target_quat["left_eef_link"]

                    print("ARM MP START")
                    eyes_target_pos, eyes_target_quat = None, None

                    if enable_marker_vis:
                        env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                        env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                        env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos, orientation=left_waypoint_ori)
                        env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos, orientation=right_waypoint_ori)

                    # For manipulation, doing multiple tries does not help much (observed empirically). So, we set num_tries to 1
                    num_tries = 3
                    arm_mp_trial = 0
                    new_target_pos = copy.deepcopy(target_pos)
                    while True:
                        
                        # Base condition 
                        if arm_mp_trial > 0:
                            
                            # If we are not retrying nav on ARM IK/TrajOpt failures, no need to run num_tries times as it most likely won't succeed. So, we can save time
                            if env.retry_nav_on_arm_mp_failure:
                                base_condition = arm_mp_trial == num_tries
                            else:
                                base_condition = arm_mp_trial == num_tries or ("IK Fail" in mp_results[0].status.value)
                            
                            if base_condition:
                                print("Arm MP failed after {} trials. Giving up.".format(num_tries))
                                if "TrajOpt Fail" in mp_results[0].status.value:
                                    env.err = "ArmMPTrajOptFailed"
                                elif "IK Fail" in mp_results[0].status.value:
                                    env.err = "ArmMPIKFailed"
                                else:
                                    env.err = "ArmMPOtherFailed"
                                env.valid_env = False 
                                env.execution_phase_ind += 1
                                return None
                                    
                        # Aggregate target_pos and target_quat to match batch_size
                        new_target_pos = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in new_target_pos.items()}
                        new_target_quat = {
                            k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_quat.items()
                        }
                        
                        arm_mp_planning_start_time = time.time()
                        # === DEBUG: 打印本次规划的目标 EEF 位置 ===
                        print(f"  [PRE-PLAN] arm_mp_trial={arm_mp_trial}, targets:")
                        for _k, _v in new_target_pos.items():
                            print(f"    target_pos[{_k}] = {_v.tolist() if hasattr(_v, 'tolist') else _v}")
                        cur_l_pos, _ = robot.get_eef_pose("left")
                        cur_r_pos, _ = robot.get_eef_pose("right")
                        print(f"    current left  EEF = {cur_l_pos.tolist()}")
                        print(f"    current right EEF = {cur_r_pos.tolist()}")
                        # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
                        mp_results, traj_paths = env.cmg.compute_trajectories(
                            target_pos=new_target_pos,
                            target_quat=new_target_quat,
                            is_local=False,
                            max_attempts=50,
                            timeout=60.0,
                            ik_fail_return=50,
                            enable_finetune_trajopt=True,
                            finetune_attempts=1,
                            return_full_result=True,
                            success_ratio=1.0 / env.primitive._motion_generator.batch_size,
                            attached_obj=attached_obj,
                            attached_obj_scale=attached_obj_scale,
                            emb_sel=emb_sel,
                            eyes_target_pos=eyes_target_pos,
                            eyes_target_quat=eyes_target_quat,
                        )
                        arm_mp_planning_finish_time = time.time()
                        phase_logs[env.execution_phase_ind]["arm_mp_planning_time"][arm_mp_trial] = round(arm_mp_planning_finish_time - arm_mp_planning_start_time, 2)

                        successes = mp_results[0].success 
                        print("Arm MP successes: ", successes)
                        success_idx = th.where(successes)[0].cpu()
                        
                        if len(success_idx) == 0:
                            print(f"Arm MP trial {arm_mp_trial} failed with status {mp_results[0].status}. Retrying...")
                            # === DEBUG ===
                            try:
                                r = mp_results[0]
                                print(f"  [DEBUG] valid_query={r.valid_query}, attempts={r.attempts}")
                                print(f"  [DEBUG] ik_time={r.ik_time}, total_time={r.total_time}")
                                print(f"  [DEBUG] pos_err={r.position_error}, rot_err={r.rotation_error}")
                            except Exception as _dbg_e:
                                print(f"  [DEBUG] read error: {_dbg_e}")
                            # === END DEBUG ===
                            arm_mp_trial += 1
                            # modify target_pos a bit
                            for k in target_pos.keys():
                                new_target_pos[k] = target_pos[k] + th.rand(3) * 0.01 - 0.005
                            continue
                        else:
                            traj_path = traj_paths[success_idx[0]]
                            break
                
                    print("Time taken for arm MP planning: ", phase_logs[env.execution_phase_ind]["arm_mp_planning_time"])
                    # ========================================================= End of Arm MP Planning ==========================================================

                    # ========================================================== Arm MP Execution ==========================================================
                    arm_mp_execution_start_time = time.time()

                    # Convert planned joint trajectory to actions
                    # Need to call q_to_action after every env.step if the base is moving; we cannot pre-compute all actions
                    q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                    q_traj = th.stack(
                        env.primitive._add_linearly_interpolated_waypoints(
                            plan=q_traj,
                            max_inter_dist=arm_mp_max_inter_dist,
                        )
                    )
                    q_traj = q_traj.cpu()
                    mp_actions = []
                    for j_pos in q_traj:

                        # If option 2 was chosen for handling arm with no ref object, we can make the action for that arm as 0
                        if object_ref["arm_left"] is None:
                            j_pos[robot.arm_control_idx["left"]] = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                        elif object_ref["arm_right"] is None:
                            j_pos[robot.arm_control_idx["right"]] = robot.get_joint_positions()[robot.arm_control_idx["right"]]

                        action = robot.q_to_action(j_pos).cpu().numpy()

                        # Add gripper actions from the original waypoints (we already checked that they are the same across MP trajectories)
                        if left_gripper_action is not None:
                            self._set_openarm_gripper_target(
                                action, env_interface.gripper_action_dim[0], "left",
                                left_gripper_action[0], gripper_interp_state,
                            )
                        if right_gripper_action is not None:
                            self._set_openarm_gripper_target(
                                action, env_interface.gripper_action_dim[1], "right",
                                right_gripper_action[1], gripper_interp_state,
                            )
                        
                        mp_actions.append(action)

                    left_eef_poses = [(left_waypoint_pos, left_waypoint_ori)] * len(mp_actions)
                    right_eef_poses = [(right_waypoint_pos, right_waypoint_ori)] * len(mp_actions)

                    # If the left hand has no motion planner waypoints, we start replaying the left hand waypoints while the right hand are following the MP trajectory.
                    if len(left_mp_waypoints) == 0:
                        # We need to pad the left hand waypoints to match the length of the MP trajectory
                        if len(left_replay_waypoints) < len(mp_actions):
                            for _ in range(len(mp_actions) - len(left_replay_waypoints)):
                                left_replay_waypoints.append(last_waypoint)

                        left_eef_poses = []
                        # We convert the target pose of the left hand to replay_action
                        # Then we *overwrite* the motion planner action with the replay action for the left arm and gripper
                        for i, action in enumerate(mp_actions):
                            replay_action = env_interface.target_pose_to_action(target_pose=left_replay_waypoints[i].pose)
                            left_eef_poses.append((left_replay_waypoints[i].pose[0:3, 3], T.mat2quat(th.tensor(left_replay_waypoints[i].pose[0:3, 0:3]))))
                            action_idx = robot.controller_action_idx["arm_left"]
                            action[action_idx] = replay_action[action_idx]
                            self._set_openarm_gripper_target(
                                action, env_interface.gripper_action_dim[0], "left",
                                left_replay_waypoints[i].gripper_action[0], gripper_interp_state,
                            )

                        # We remove the waypoints that have been replayed for the left arm
                        left_replay_waypoints = left_replay_waypoints[len(mp_actions):]

                    # Same logic as above but for the right hand
                    elif len(right_mp_waypoints) == 0:
                        if len(right_replay_waypoints) < len(mp_actions):
                            for _ in range(len(mp_actions) - len(right_replay_waypoints)):
                                right_replay_waypoints.append(last_waypoint)
                        right_eef_poses = []
                        for i, action in enumerate(mp_actions):
                            replay_action = env_interface.target_pose_to_action(target_pose=right_replay_waypoints[i].pose)
                            right_eef_poses.append((right_replay_waypoints[i].pose[4:7, 3], T.mat2quat(th.tensor(right_replay_waypoints[i].pose[4:7, 0:3]))))
                            action_idx = robot.controller_action_idx["arm_right"]
                            action[action_idx] = replay_action[action_idx]
                            self._set_openarm_gripper_target(
                                action, env_interface.gripper_action_dim[1], "right",
                                right_replay_waypoints[i].gripper_action[1], gripper_interp_state,
                            )

                        right_replay_waypoints = right_replay_waypoints[len(mp_actions):]

                    assert len(mp_actions) == len(left_eef_poses) == len(right_eef_poses)

                    init_global_env_step = env.global_env_step
                    num_repeat = 1
                    for i, mp_action in enumerate(mp_actions):
                        for _ in range(num_repeat):
                            mp_action = self._hold_openarm_attached_grippers(
                                action=mp_action,
                                robot=robot,
                                attached_obj=attached_obj,
                                interp_state=gripper_interp_state,
                                phase_type=phase_type,
                            )
                            mp_action = self._close_openarm_pick_grippers(
                                action=mp_action,
                                robot=robot,
                                object_ref=object_ref,
                                phase_type=phase_type,
                                interp_state=gripper_interp_state,
                            )
                            mp_action = self._lock_openarm_attached_gripper_state(
                                robot=robot,
                                attached_obj=attached_obj,
                                action=mp_action,
                                interp_state=gripper_interp_state,
                            )
                            state = env.get_state()["states"]
                            obs, obs_info = env.get_obs_IL()
                            datagen_info = env_interface.get_datagen_info(action=mp_action)
                            # TODO: Check if we can use primtiive stack execute action here. This will allow for checking convergence errors etc.
                            env.step(mp_action, video_writer)
                            if enable_marker_vis:
                                env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                                env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                                env.eef_goal_marker_left.set_position_orientation(*left_eef_poses[i])
                                env.eef_goal_marker_right.set_position_orientation(*right_eef_poses[i])
                            local_env_step += 1
                            env.global_env_step += 1
                            states.append(state)
                            actions.append(mp_action)
                            observations.append(obs)
                            observations_info.append(json.dumps(obs_info))
                            datagen_infos.append(datagen_info)
                            cur_success_metrics = env.is_success()
                            if ref_obj is not None:
                                self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                            for k in success:
                                success[k] = success[k] or cur_success_metrics[k]

                # Set the MP ranges to save to hdf5 file
                left_mp_ranges, right_mp_ranges = None, None
                if len(left_mp_waypoints) > 0:
                    left_mp_ranges = [init_global_env_step, env.global_env_step]
                if len(right_mp_waypoints) > 0:
                    right_mp_ranges = [init_global_env_step, env.global_env_step]
                
                
                MP_end_step_local = copy.deepcopy(local_env_step)
                # left MP points
                if len(left_mp_waypoints) == 0: 
                    left_MP_end_step_local = 0
                else: 
                    left_MP_end_step_local = MP_end_step_local
                if len(right_mp_waypoints) == 0: 
                    right_MP_end_step_local = 0
                else: 
                    right_MP_end_step_local = MP_end_step_local

                MP_end_step_local_list = [left_MP_end_step_local, right_MP_end_step_local]

                arm_mp_execution_finish_time = time.time()
                # Since there is only 1 trial for arm MP execution, we set the 0th index
                phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0] = round(arm_mp_execution_finish_time - arm_mp_execution_start_time, 2)
                print("Time taken for arm MP execution:", phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0])
                
                num_phase_steps = env.global_env_step - init_global_env_step
                for sensor_name, sensor in env.robot.sensors.items():
                    if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                        shortened_sensor_name = sensor_name.split(":")[1]
                        if num_phase_steps > 0:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                        else:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"]= 0
                        print(f"Visibility stats for arm_mp {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"])
                if num_phase_steps > 0:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                else:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"]= 0
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_steps"] = num_phase_steps
                print(f"Visibility stats for arm_mp any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"])

                # ============================================== End of Arm MP ==========================================================

            # ================================================== Arm Replay ==========================================================
            # reset the visibility counter for each sensor
            self.reset_visibility_counter(env)
            
            # We need to pad the waypoints for the left and right hands to match the length of the longest trajectory
            if len(left_replay_waypoints) < len(right_replay_waypoints):
                for _ in range(len(right_replay_waypoints) - len(left_replay_waypoints)):
                    left_replay_waypoints.append(last_waypoint)
            elif len(right_replay_waypoints) < len(left_replay_waypoints):
                for _ in range(len(left_replay_waypoints) - len(right_replay_waypoints)):
                    right_replay_waypoints.append(last_waypoint)

            assert len(left_replay_waypoints) == len(right_replay_waypoints)
            # print('length of replay actions:', len(left_replay_waypoints))
            print("ARM REPLAY START")
            arm_replay_start_time = time.time()
            replay_gripper_schedule = self._make_openarm_replay_gripper_schedule(
                src_actions=src_curr_phase_actions,
                robot=robot,
                mp_end_steps=cur_subtask_end_step_MP,
                object_ref=object_ref,
            )
            
            # If one of the arms has no ref object, we set its target pose as the current pose
            if object_ref["arm_right"] is None:
                current_right_ee_pose = robot.get_eef_pose("right")
                current_right_ee_pos = current_right_ee_pose[0]
                current_right_ee_quat = current_right_ee_pose[1]
                current_right_ee_matrix = T.quat2mat(current_right_ee_quat)
                current_right_ee_pose = th.eye(4)
                current_right_ee_pose[:3, :3] = current_right_ee_matrix
                current_right_ee_pose[:3, 3] = current_right_ee_pos
            elif object_ref["arm_left"] is None:
                current_left_ee_pose = robot.get_eef_pose("left")
                current_left_ee_pos = current_left_ee_pose[0]
                current_left_ee_quat = current_left_ee_pose[1]
                current_left_ee_matrix = T.quat2mat(current_left_ee_quat)
                current_left_ee_pose = th.eye(4)
                current_left_ee_pose[:3, :3] = current_left_ee_matrix
                current_left_ee_pose[:3, 3] = current_left_ee_pos
            
            # For each pair of waypoints, we extract the pose for each hand and then convert to action
            # We also overwrite the gripper actions with the ones from the waypoints
            init_global_env_step = env.global_env_step
            for replay_i, (left_waypoint, right_waypoint) in enumerate(zip(left_replay_waypoints, right_replay_waypoints)):
                pose = np.zeros((8, 4))
                pose[:4, :] = left_waypoint.pose[:4, :]
                pose[4:, :] = right_waypoint.pose[4:, :]
                # If one of the arms has no ref object, we set its target pose as the current pose
                if object_ref["arm_right"] is None:
                    pose[4:, :] = current_right_ee_pose
                elif object_ref["arm_left"] is None:
                    pose[:4, :] = current_left_ee_pose
                replay_action = env_interface.target_pose_to_action(target_pose=pose)

                self._set_openarm_gripper_target(
                    replay_action, env_interface.gripper_action_dim[0], "left",
                    left_waypoint.gripper_action[0], gripper_interp_state,
                )
                self._set_openarm_gripper_target(
                    replay_action, env_interface.gripper_action_dim[1], "right",
                    right_waypoint.gripper_action[1], gripper_interp_state,
                )
                replay_action = self._apply_openarm_replay_gripper_schedule(
                    robot=robot,
                    action=replay_action,
                    replay_i=replay_i,
                    schedule=replay_gripper_schedule,
                    replay_relative=True,
                )

                state = env.get_state()["states"]
                temp_start_time = time.time()
                obs, obs_info = env.get_obs_IL()
                datagen_info = env_interface.get_datagen_info(action=replay_action)
                env.step(replay_action, video_writer)
                left_eef_pose = (pose[0:3, 3], T.mat2quat(th.tensor(pose[0:3, 0:3])))
                right_eef_pose = (pose[4:7, 3], T.mat2quat(th.tensor(pose[4:7, 0:3])))
                if enable_marker_vis:
                    env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                    env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                    env.eef_goal_marker_left.set_position_orientation(*left_eef_pose)
                    env.eef_goal_marker_right.set_position_orientation(*right_eef_pose)
                local_env_step += 1
                env.global_env_step += 1
                states.append(state)
                actions.append(replay_action)
                observations.append(obs)
                observations_info.append(json.dumps(obs_info))
                datagen_infos.append(datagen_info)
                cur_success_metrics = env.is_success()
                if ref_obj is not None:
                    self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]

            arm_replay_finish_time = time.time()
            phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0] = round(arm_replay_finish_time - arm_replay_start_time, 2)
            print("Time taken for arm replay: ", phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0])

            num_phase_steps = env.global_env_step - init_global_env_step
            for sensor_name, sensor in env.robot.sensors.items():
                if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                    shortened_sensor_name = sensor_name.split(":")[1]
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"]= 0
                    print(f"Visibility stats for arm_replay {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"])
            if num_phase_steps > 0:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
            else:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"]= 0
            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_steps"] = num_phase_steps
            print(f"Visibility stats for arm_replay any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"])

            # =================================================== End of Arm Replay ==========================================================

            results = dict(
                states=states,
                observations=observations,
                datagen_infos=datagen_infos,
                actions=np.array(actions),
                success=bool(success["task"]),
                mp_end_steps=MP_end_step_local_list,
                subtask_lengths=local_env_step,
                left_mp_ranges=left_mp_ranges,
                right_mp_ranges=right_mp_ranges,
                retry_nav=False,
                observations_info=observations_info
            )
            env.execution_phase_ind += 1
            env.phases_completed_wo_mp_err += 1
            return results



    def execute(
        self, 
        env,
        env_interface, 
        render=False, 
        video_writer=None, 
        video_skip=5, 
        camera_names=None,
        bimanual=False,
        cur_subtask_end_step_MP=None,
        attached_obj=None,
        phase_type=None,
        object_ref=None,
        grasp_init_views_video_writer=None,
        enable_marker_vis=False,
        ds_ratio=1,
        phase_logs=None,
        retract_type=None,
        src_curr_phase_actions=None,
        frozen_arms=None,
    ):
        """
        Main function to execute the trajectory. Will use env_interface.target_pose_to_action to
        convert each target pose at each waypoint to an action command, and pass that along to
        env.step.

        Args:
            env (robomimic EnvBase instance): environment to use for executing trajectory
            env_interface (EG_EnvInterface instance): environment interface for executing trajectory
            render (bool): if True, render on-screen
            video_writer (imageio writer): video writer
            video_skip (int): determines rate at which environment frames are written to video
            camera_names (list): determines which camera(s) are used for rendering. Pass more than
                one to output a video with multiple camera views concatenated horizontally.
            cur_subtask_end_step_MP: list of size 2, the end point of motion planner for two arms

        Returns:
            results (dict): dictionary with the following items for the executed trajectory:
                states (list): simulator state at each timestep
                observations (list): observation dictionary at each timestep
                datagen_infos (list): datagen_info at each timestep
                actions (list): action executed at each timestep
                success (bool): whether the trajectory successfully solved the task or not
        """
   
        ref_object = self._select_reference_object(object_ref)
        ref_obj = self._resolve_tracking_object(env=env, env_interface=env_interface, ref_object=ref_object)
        env.primitive._tracking_object = ref_obj
        if ref_obj is not None:
            print("Will track object for this sub-step: ", ref_obj.name)
        robot = env.env.robots[0]
        frozen_arms = tuple(frozen_arms or ())
        gripper_interp_state = self._make_openarm_gripper_interp_state(env=env, robot=robot)
        
        # ================================= Base Navigation ==================================
        if phase_type == "navigation":
            phase_logs[env.execution_phase_ind] = self.setup_phase_logs(phase_type=phase_type)
            seq = self.waypoint_sequences[0]
            
            left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
            left_replay_waypoints = seq[cur_subtask_end_step_MP[0]:]
            left_mp_last_waypoint = left_mp_waypoints[-1]
            left_waypoints = [left_mp_last_waypoint] + left_replay_waypoints

            left_waypoint_pos = th.vstack([th.tensor(wp.pose[0:3, 3]) for wp in left_waypoints])
            left_waypoint_ori = th.vstack([T.mat2quat(th.tensor(wp.pose[0:3, 0:3])) for wp in left_waypoints])

            right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
            right_replay_waypoints = seq[cur_subtask_end_step_MP[1]:]
            right_mp_last_waypoint = right_mp_waypoints[-1]
            right_waypoints = [right_mp_last_waypoint] + right_replay_waypoints
 
            right_waypoint_pos = th.vstack([th.tensor(wp.pose[4:7, 3]) for wp in right_waypoints])
            right_waypoint_ori = th.vstack([T.mat2quat(th.tensor(wp.pose[4:7, 0:3])) for wp in right_waypoints])

            left_waypoint_pos, right_waypoint_pos = self._pad_tensors(left_waypoint_pos, right_waypoint_pos)
            left_waypoint_ori, right_waypoint_ori = self._pad_tensors(left_waypoint_ori, right_waypoint_ori)

            left_waypoint_pos = self._subsample_tensor(left_waypoint_pos)
            left_waypoint_ori = self._subsample_tensor(left_waypoint_ori)
            right_waypoint_pos = self._subsample_tensor(right_waypoint_pos)
            right_waypoint_ori = self._subsample_tensor(right_waypoint_ori)
            
            # left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
            # left_waypoint = left_mp_waypoints[-1]
            # left_waypoint_pos, left_waypoint_ori = th.tensor(left_waypoint.pose[0:3, 3]), T.mat2quat(th.tensor(left_waypoint.pose[0:3, 0:3]))
            # right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
            # right_waypoint = right_mp_waypoints[-1]
            # right_waypoint_pos, right_waypoint_ori = th.tensor(right_waypoint.pose[4:7, 3]), T.mat2quat(th.tensor(right_waypoint.pose[4:7, 0:3]))

            eef_pose = {
                "left": (left_waypoint_pos, left_waypoint_ori),
                "right": (right_waypoint_pos, right_waypoint_ori)
            }
            
            if enable_marker_vis:
                env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                # env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos, orientation=left_waypoint_ori)
                # env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos, orientation=right_waypoint_ori)
                env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos[0], orientation=left_waypoint_ori[0])
                env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos[0], orientation=right_waypoint_ori[0])
                for _ in range(10): og.sim.step()
            
            # TODO: Implement this
            check_torso_mode_first = False
            if check_torso_mode_first:
                pass
                # Given the ref obect and the eef poses, check if we can only move the torso to satisfy reachability and visibility
                # 0. attach object
                # 1. call _target_in_reach_of_robot_and_visible(self,
                #                                               eef_pose,
                #                                               initial_joint_pos=env.robot.get_joint_positions(),
                #                                               skip_obstacle_update=True,
                #                                               ik_world_collision_check=False,
                #                                               emb_sel=CuRoboEmbodimentSelection.ARM,
                #                                               attach_obj=False, 
                #                                               eyes_pose=None):
                # So, above will sample eyes pose, check IK solving for (eef_poses, eyes_pose) w/o collision check, for samples that succeed previous Ik check
                # check if setting (current base + IK torso + current arms) is collision-free
                # 2. If yes, use the aforementioned (eyes_pose + eef poses) and do arm mode MP (which is with collision) and overwrite the arm actions to not do anything
                # 3. If above, succeeds, execute it
                # 4. else, continue to base MP

            
            num_tries = 3
            base_mp_trial = 0
            nav_mp_success = False
            while True:
                # Base condition
                if base_mp_trial == num_tries:
                    print("Base MP failed after {} trials. Giving up.".format(num_tries))
                    env.err = env.primitive.mp_err
                    # execution_phase_ind keeps track of each phase that was tried to be executed (even if MP failed for that phase). 
                    # In this case MP failed and phase was not actually executed
                    env.execution_phase_ind += 1
                    # env.valid_env = env.primitive.valid_env
                    return None

                print("Base MP trial: ", base_mp_trial)
                
                enable_visibility_constraint = False
                
                # Pass only the eef that has a reference object associated with it (i.e. the arm that is relevant for this sub-step)
                if object_ref["arm_right"] is None:
                    action_generator = env.primitive._navigate_to_obj(obj=ref_obj, eef_pose={"left": eef_pose["left"]}, visibility_constraint=enable_visibility_constraint)
                elif object_ref["arm_left"] is None:
                    action_generator = env.primitive._navigate_to_obj(obj=ref_obj, eef_pose={"right": eef_pose["right"]}, visibility_constraint=enable_visibility_constraint)
                else:
                    action_generator = env.primitive._navigate_to_obj(obj=ref_obj, eef_pose=eef_pose, visibility_constraint=enable_visibility_constraint)
                # action_generator = env.primitive._navigate_to_obj(obj=ref_obj, visibility_constraint=env.hard_visibility_constraint)
                
                init_state = og.sim.dump_state()
                local_env_step = 0
                states = []
                actions = []
                observations = []
                observations_info = []
                datagen_infos = []
                success = {"task": False}
                init_global_env_step = env.global_env_step
                # success = {k: False for k in env.is_success()} # success metrics
                for temp_idx, mp_action in enumerate(action_generator):
                    
                    # This will happen if
                    # 1. base sampling fails
                    # 2. base MP fails.
                    # 3. base execution fails to converge
                    if mp_action is None:
                        print(f"Base MP trial {base_mp_trial} failed. Retrying...")
                        base_mp_trial += 1
                        nav_mp_success = False
                        # This is there to avoid error in nav execution time (which in this case will always be 0)
                        nav_execution_start_time = time.time()
                        break
                    else:
                        nav_mp_success = True
                
                    if temp_idx == 0:
                        print("Time taken for base sampling: ", env.primitive.base_sampling_time)
                        print("Time taken for base MP planning: ", env.primitive.base_mp_planning_time)
                        nav_execution_start_time = time.time()

                    mp_action = mp_action.cpu().numpy()
                    # NOTE: For the MultiFinger gripper controler in binary mode that we use for tiago, we need to ensure that the
                    # gripper actions are correctly set based on whether an object is grasped by that gripper or not 
                    if attached_obj["left"] is not None:
                        self._set_openarm_gripper_target(
                            mp_action, robot.gripper_action_idx["left"], "left", -1, gripper_interp_state,
                        )
                    if attached_obj["right"] is not None:
                        self._set_openarm_gripper_target(
                            mp_action, robot.gripper_action_idx["right"], "right", -1, gripper_interp_state,
                        )
                    mp_action = self._lock_openarm_attached_gripper_state(
                        robot=robot,
                        attached_obj=attached_obj,
                        action=mp_action,
                        interp_state=gripper_interp_state,
                    )
                    state = env.get_state()["states"]
                    obs, obs_info = env.get_obs_IL()
                    datagen_info = env_interface.get_datagen_info(action=mp_action)
                    # print("mp_action[robot.base_action_idx]: ", mp_action[robot.base_action_idx])
                    env.step(mp_action, video_writer)
                    local_env_step += 1
                    env.global_env_step += 1
                    states.append(state)
                    actions.append(mp_action)
                    observations.append(obs)
                    observations_info.append(json.dumps(obs_info))
                    datagen_infos.append(datagen_info)
                    self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)

                # Save timings to current_phase_logs
                nav_execution_finish_time = time.time()
                phase_logs[env.execution_phase_ind]["base_sampling_time"][base_mp_trial] = env.primitive.base_sampling_time
                phase_logs[env.execution_phase_ind]["base_mp_planning_time"][base_mp_trial] = env.primitive.base_mp_planning_time
                phase_logs[env.execution_phase_ind]["base_mp_execution_time"][base_mp_trial] = round(nav_execution_finish_time - nav_execution_start_time, 2)
                
                num_phase_steps = env.global_env_step - init_global_env_step
                for sensor_name, sensor in env.robot.sensors.items():
                    if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                        shortened_sensor_name = sensor_name.split(":")[1]
                        if num_phase_steps > 0:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                        else:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_{shortened_sensor_name}"]= 0
                        print(f"Visibility stats for nav_mp {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_{shortened_sensor_name}"])
                if num_phase_steps > 0:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                else:
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_any"] = 0
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_steps"] = num_phase_steps
                print(f"Visibility stats for nav_mp any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"nav_mp_any"])

                if not nav_mp_success:
                    # This will happen if
                    # 1. base sampling fails
                    # 2. base MP fails.
                    # 3. base execution fails to converge

                    # In case #3, we actually step physics in OG, so we need to reset the state
                    if env.primitive.mp_err in ["BaseExecutionBaseTargetNotReached", "BaseExecutionArmTorsoTargetNotReached"]:
                        og.sim.load_state(init_state)
                        for _ in range(5): og.sim.step()
                        
                        # Reset the visibility stats
                        self.reset_visibility_counter(env)

                    continue
                
                env.err = env.primitive.mp_err
                MP_end_step_local_list = [cur_subtask_end_step_MP[0], cur_subtask_end_step_MP[1]]
                left_mp_ranges = [init_global_env_step, env.global_env_step]
                right_mp_ranges = [init_global_env_step, env.global_env_step]
                results = dict(
                    states=states,
                    observations=observations,
                    datagen_infos=datagen_infos,
                    actions=np.array(actions),
                    success=bool(success["task"]),
                    mp_end_steps=MP_end_step_local_list,
                    subtask_lengths=local_env_step,
                    left_mp_ranges=left_mp_ranges,
                    right_mp_ranges=right_mp_ranges,
                    retry_nav=False,
                    observations_info=observations_info
                )
                # execution_phase_ind keeps track of each phase that was tried to be executed (even if MP failed for that phase). 
                # In this case MP succeeded and phase was actually executed
                env.execution_phase_ind += 1
                env.phases_completed_wo_mp_err += 1
                return results
        # ============================================== Base Navigation ==============================================

        if phase_type != "navigation":
            # =============================================== Arm MP Planning =============================================
            phase_logs[env.execution_phase_ind] = self.setup_phase_logs(phase_type=phase_type)
            local_env_step = 0
            states = []
            actions = []
            observations = []
            observations_info = []
            datagen_infos = []
            success = {"task": False}
            # success = {k: False for k in env.is_success()} # success metrics

            assert len(self.waypoint_sequences) == 1
            seq = self.waypoint_sequences[0]
            for end_step in cur_subtask_end_step_MP:
                assert 0 <= end_step <= len(seq)

            if (
                getattr(env, "name", "").startswith("openarm_real_exp_1")
                and os.environ.get("ELOGGEN_MANIPULATION_MODE", "curobo") == "source_replay"
                and src_curr_phase_actions is not None
            ):
                print(
                    "[OpenArm] manipulation_mode=source_replay: "
                    "replaying source actions for manipulation phase and skipping CuRobo arm MP"
                )
                frozen_source_replay_arms = [
                    arm
                    for arm in ("left", "right")
                    if object_ref is not None and object_ref.get(f"arm_{arm}") is None
                ]
                if frozen_source_replay_arms:
                    phase_logs[env.execution_phase_ind]["source_replay_frozen_arms"] = frozen_source_replay_arms
                    print(
                        "[OpenArm] source_replay freezing inactive arms: "
                        f"{frozen_source_replay_arms}"
                    )
                self.reset_visibility_counter(env)
                replay_start_time = time.time()
                init_global_env_step = env.global_env_step
                replay_gripper_schedule = self._make_openarm_replay_gripper_schedule(
                    src_actions=src_curr_phase_actions,
                    robot=robot,
                    mp_end_steps=cur_subtask_end_step_MP,
                    object_ref=object_ref,
                )
                for replay_i, src_action in enumerate(src_curr_phase_actions):
                    action = np.asarray(src_action, dtype=np.float32).copy()
                    action = self._freeze_openarm_source_replay_action(
                        action=action,
                        robot=robot,
                        frozen_arms=frozen_source_replay_arms,
                    )
                    action = self._smooth_openarm_action_grippers(
                        action=action,
                        robot=robot,
                        interp_state=gripper_interp_state,
                    )
                    action = self._hold_openarm_attached_grippers(
                        action=action,
                        robot=robot,
                        attached_obj=attached_obj,
                        interp_state=gripper_interp_state,
                        phase_type=phase_type,
                        reference_action=src_action,
                    )
                    action = self._close_openarm_pick_grippers(
                        action=action,
                        robot=robot,
                        object_ref=object_ref,
                        phase_type=phase_type,
                        interp_state=gripper_interp_state,
                    )
                    action = self._lock_openarm_attached_gripper_state(
                        robot=robot,
                        attached_obj=attached_obj,
                        action=action,
                        interp_state=gripper_interp_state,
                    )
                    action = self._apply_openarm_replay_gripper_schedule(
                        robot=robot,
                        action=action,
                        replay_i=replay_i,
                        schedule=replay_gripper_schedule,
                    )
                    state = env.get_state()["states"]
                    obs, obs_info = env.get_obs_IL()
                    datagen_info = env_interface.get_datagen_info(action=action)
                    env.step(action, video_writer)
                    local_env_step += 1
                    env.global_env_step += 1
                    states.append(state)
                    actions.append(action)
                    observations.append(obs)
                    observations_info.append(json.dumps(obs_info))
                    datagen_infos.append(datagen_info)
                    cur_success_metrics = env.is_success()
                    if ref_obj is not None:
                        self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                    for k in success:
                        success[k] = success[k] or cur_success_metrics[k]

                replay_finish_time = time.time()
                phase_logs[env.execution_phase_ind]["arm_mp_planning_time"][0] = 0.0
                phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0] = round(
                    replay_finish_time - replay_start_time, 2
                )
                phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0] = 0.0

                num_phase_steps = env.global_env_step - init_global_env_step
                for sensor_name, sensor in env.robot.sensors.items():
                    if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                        shortened_sensor_name = sensor_name.split(":")[1]
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"source_replay_{shortened_sensor_name}"] = (
                            env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                            if num_phase_steps > 0
                            else 0
                        )
                        print(
                            f"Visibility stats for source_replay {shortened_sensor_name}: ",
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"source_replay_{shortened_sensor_name}"],
                        )
                phase_logs[env.execution_phase_ind]["visibility_stats"]["source_replay_any"] = (
                    env.num_frames_with_obj_visible["any"] / num_phase_steps if num_phase_steps > 0 else 0
                )
                phase_logs[env.execution_phase_ind]["visibility_stats"]["source_replay_steps"] = num_phase_steps
                print(
                    "Visibility stats for source_replay any: ",
                    phase_logs[env.execution_phase_ind]["visibility_stats"]["source_replay_any"],
                )

                mp_end_step_local_list = [
                    min(cur_subtask_end_step_MP[0], local_env_step),
                    min(cur_subtask_end_step_MP[1], local_env_step),
                ]
                results = dict(
                    states=states,
                    observations=observations,
                    datagen_infos=datagen_infos,
                    actions=np.array(actions),
                    success=bool(success["task"]),
                    mp_end_steps=mp_end_step_local_list,
                    subtask_lengths=local_env_step,
                    left_mp_ranges=[init_global_env_step, init_global_env_step + mp_end_step_local_list[0]],
                    right_mp_ranges=[init_global_env_step, init_global_env_step + mp_end_step_local_list[1]],
                    retry_nav=False,
                    observations_info=observations_info,
                )
                env.execution_phase_ind += 1
                env.phases_completed_wo_mp_err += 1
                return results

            # Segment the waypoints into motion planner waypoints and replay waypoints
            left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
            left_replay_waypoints = seq[cur_subtask_end_step_MP[0]:]
            right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
            right_replay_waypoints = seq[cur_subtask_end_step_MP[1]:]
            if ds_ratio not in (None, 1):
                left_replay_waypoints, right_replay_waypoints = self.downsample_replay_traj(
                    left_replay_waypoints=left_replay_waypoints,
                    right_reaplay_waypoints=right_replay_waypoints,
                    ds_ratio=int(ds_ratio),
                )

            # print("left_mp_waypoints", len(left_mp_waypoints))
            # print("left_replay_waypoints", len(left_replay_waypoints))
            # print("right_mp_waypoints", len(right_mp_waypoints))
            # print("right_replay_waypoints", len(right_replay_waypoints))

            # Get the last waypoint for padding later
            last_waypoint = seq[-1]

            # # Temporary: This is just to capture the first image after navigating to the teacup, just for visualization
            # if object_ref["arm_left"] == "teacup" and grasp_init_views_video_writer is not None:
            #     robot_name = env.env.robots[0].name
            #     obs, obs_info = env.get_observation()
            #     ego_img = obs[f"{robot_name}::{robot_name}:eyes:Camera:0::rgb"]
            #     # eef_left_img = obs[f"{robot_name}::{robot_name}:left_eef_link:Camera:0::rgb"]
            #     # eef_right_img = obs[f"{robot_name}::{robot_name}:right_eef_link:Camera:0::rgb"]
            #     concatenated_img = hori_concatenate_image([ego_img])
            #     grasp_init_views_video_writer.append_data(concatenated_img)

            
            # 1. make sure the gripper actions are the same
            # 2. get the last waypoint's pose and orientation as the MP target
            # Otherwise, use the current eef pose as the MP target
            if len(left_mp_waypoints) > 0:
                gripper_actions = np.array([waypoint.gripper_action for waypoint in left_mp_waypoints])
                # This is not necessarily true since while teleopating as a non-optimal teleoperator, I inadvertently would toggle gripper on / off
                # Specially when trying to grasp. So removed this assertion
                # assert (gripper_actions[:, 0] == gripper_actions[0, 0]).all()
                left_waypoint = left_mp_waypoints[-1]
                left_gripper_action = left_waypoint.gripper_action
                left_waypoint_pos, left_waypoint_ori = th.tensor(left_waypoint.pose[0:3, 3]), T.mat2quat(th.tensor(left_waypoint.pose[0:3, 0:3]))
            else:
                left_gripper_action = None
                left_waypoint_pos, left_waypoint_ori = robot.get_eef_pose("left")

            if len(right_mp_waypoints) > 0:
                gripper_actions = np.array([waypoint.gripper_action for waypoint in right_mp_waypoints])
                # This is not necessarily true since while teleopating as a non-optimal teleoperator, I inadvertently would toggle gripper on / off
                # Specially when trying to grasp. So removed this assertion
                # assert (gripper_actions[:, 1] == gripper_actions[0, 1]).all()
                right_waypoint = right_mp_waypoints[-1]
                right_gripper_action = right_waypoint.gripper_action
                right_waypoint_pos, right_waypoint_ori = th.tensor(right_waypoint.pose[4:7, 3]), T.mat2quat(th.tensor(right_waypoint.pose[4:7, 0:3]))
            else:
                right_gripper_action = None
                right_waypoint_pos, right_waypoint_ori = robot.get_eef_pose("right")

            # # Option 1: If one of the arm does not hav a ref object, set its target pose as the current pose
            # if object_ref["arm_right"] is None:
            #     right_waypoint_pos, right_waypoint_ori = robot.get_eef_pose("right")
            # elif object_ref["arm_left"] is None:
            #     left_waypoint_pos, left_waypoint_ori = robot.get_eef_pose("left")

            
            # If at least one hand has motion planner waypoints, plan the motion
            if len(left_mp_waypoints) > 0 or len(right_mp_waypoints) > 0:
                target_pos = {
                    robot.eef_link_names["left"]: left_waypoint_pos,
                    robot.eef_link_names["right"]: right_waypoint_pos,
                }
                target_quat = {
                    robot.eef_link_names["left"]: left_waypoint_ori,
                    robot.eef_link_names["right"]: right_waypoint_ori,
                }
                # If both hands have motion planner waypoints, we use the arm + torso embodiment
                # If only one of the hands has motion planner waypoints, we use the arm embodiment only because
                # when we replay the waypoints for the other hand, we assume the torso is fixed.
                emb_sel = CuRoboEmbodimentSelection.ARM if len(left_mp_waypoints) > 0 and len(right_mp_waypoints) > 0 else CuRoboEmbodimentSelection.ARM_NO_TORSO
                
                # To test MP in arm_no_toso mode instead of arm mode, uncomment the line below
                emb_sel = CuRoboEmbodimentSelection.ARM_NO_TORSO
                
                # # Option 1: Use template to know attached objects
                # if attached_obj is None:
                #     attached_obj_scale = None
                # else:
                #     attached_obj_new = {}
                #     attached_obj_scale = {}
                #     for arm, obj_name in attached_obj.items():
                #         if obj_name is not None:
                #             attached_obj_new[robot.eef_link_names[arm]] = env.env.scene.object_registry("name", obj_name).root_link
                #             attached_obj_scale[robot.eef_link_names[arm]] = 0.9
                #     attached_obj = attached_obj_new

                # Option 2: Use OG to know attached objects
                arm_mp_max_inter_dist = self._arm_mp_max_inter_dist(
                    object_ref=object_ref,
                    attached_obj=attached_obj,
                )
                retval = self.obtain_attached_object(env, robot)
                attached_obj = retval["attached_obj"]
                attached_obj_scale = retval["attached_obj_scale"]

                # Option 2: If one of the arm does not hav a ref object, remove it from the target pose of MP (will move this arm randomly in this case)
                if object_ref["arm_right"] is None:
                    del target_pos[robot.eef_link_names["right"]]
                    del target_quat[robot.eef_link_names["right"]]
                elif object_ref["arm_left"] is None:
                    del target_pos[robot.eef_link_names["left"]]
                    del target_quat[robot.eef_link_names["left"]]

                # # Check object visibility at start-of-manip step
                # try:
                #     obs, obs_info = env.get_observation()
                #     seg_instance = obs[f"{env.robot_name}::{env.robot_name}:eyes:Camera:0::seg_instance"]
                #     seg_instance_info = obs_info[f"{env.robot_name}"][f"{env.robot_name}:eyes:Camera:0"]["seg_instance"]
                #     key_of_coffee_cup = next((key for key, value in seg_instance_info.items() if value == "coffee_cup"), None)
                #     if key_of_coffee_cup is None:
                #         count = 0
                #     else:
                #         count = (seg_instance == key_of_coffee_cup).sum().item()
                #     if count > 150:
                #         env.obj_visible_at_start_of_manip = True
                # except Exception as e:

                
                # This is for retract behavior. We are not using this as of now, but let it be 
                initial_left_eef_pose = robot.get_eef_pose("left")
                initial_right_eef_pose = robot.get_eef_pose("right")
                
                print("ARM MP START")
                eyes_target_pos, eyes_target_quat = None, None
                # NOTE: Keep this commented out. We won't be using soft visibility constraint with manipulation for now. As we are using ARM_NO_TORSO mode
                # if env.soft_visibility_constraint:
                #     obj_pose = ref_obj.get_position_orientation()
                #     eyes_target_pos = obj_pose[0]
                #     eyes_target_quat = obj_pose[1]

                if enable_marker_vis:
                    env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                    env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                    env.eef_goal_marker_left.set_position_orientation(position=left_waypoint_pos, orientation=left_waypoint_ori)
                    env.eef_goal_marker_right.set_position_orientation(position=right_waypoint_pos, orientation=right_waypoint_ori)

                
                # For manipulation, doing multiple tries does not help much (observed empirically). So, we set num_tries to 1
                num_tries = 3
                arm_mp_trial = 0
                new_target_pos = copy.deepcopy(target_pos)
                while True:
                    
                    # Base condition 
                    if arm_mp_trial > 0:
                        # # Trying a hacky way to reduce the IK failure. Basically moving the robot base a bit towards the object. 
                        # # This does not ensure collision-free motion
                        # if "IK Fail" in mp_results[0].status.value:
                        #     obj_pos = ref_obj.get_position_orientation()[0][:2]
                        #     robot_base_pose = env.robot.get_position_orientation()
                        #     robot_base_pos = robot_base_pose[0][:2]
                        #     vec = obj_pos - robot_base_pos
                        #     vec = vec / np.linalg.norm(vec)
                        #     for _ in range(10):
                        #         joint_pos = env.robot.get_joint_positions()
                        #         joint_pos[:2] = joint_pos[:2] + (vec * 0.01)
                        #         action = env.robot.q_to_action(joint_pos).cpu().numpy()
                        #         # Add gripper actions from the original waypoints (we already checked that they are the same across MP trajectories)
                        #         if left_gripper_action is not None:
                        #             action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                        #         if right_gripper_action is not None:
                        #             action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]
                                
                        #         state = env.get_state()["states"]
                        #         obs, obs_info = env.get_obs_IL()
                        #         datagen_info = env_interface.get_datagen_info(action=action)
                        #         env.step(action, video_writer)
                        #         local_env_step += 1
                        #         env.global_env_step += 1
                        #         states.append(state)
                        #         actions.append(action)
                        #         observations.append(obs)
                        #         datagen_infos.append(datagen_info)

                        if ("IK Fail" in mp_results[0].status.value or "TrajOpt Fail" in mp_results[0].status.value) and env.retry_nav_on_arm_mp_failure:
                            results = dict(
                                states=states,
                                observations=observations,
                                datagen_infos=datagen_infos,
                                actions=np.array(actions),
                                success=bool(success["task"]),
                                retry_nav=True,
                                observations_info=observations_info
                            )
                            return results
                        
                        # If we are not retrying nav on ARM IK/TrajOpt failures, no need to run num_tries times as it most likely won't succeed. So, we can save time
                        if env.retry_nav_on_arm_mp_failure:
                            base_condition = arm_mp_trial == num_tries
                        else:
                            base_condition = arm_mp_trial == num_tries or ("IK Fail" in mp_results[0].status.value)
                        
                        if base_condition:
                            print("Arm MP failed after {} trials. Giving up.".format(num_tries))
                            if "TrajOpt Fail" in mp_results[0].status.value:
                                env.err = "ArmMPTrajOptFailed"
                            elif "IK Fail" in mp_results[0].status.value:
                                env.err = "ArmMPIKFailed"
                            else:
                                env.err = "ArmMPOtherFailed"
                            env.valid_env = False 
                            env.execution_phase_ind += 1
                            return None
                                
                    # Aggregate target_pos and target_quat to match batch_size
                    new_target_pos = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in new_target_pos.items()}
                    new_target_quat = {
                        k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_quat.items()
                    }
                    
                    arm_mp_planning_start_time = time.time()
                    # ===== [IK诊断补丁①] 规划前打印关键状态 =====
                    print(f"  [IK_DIAG] arm_mp_trial={arm_mp_trial}, emb_sel={emb_sel}")
                    for _ik_k, _ik_v in new_target_pos.items():
                        _ik_v0 = _ik_v[0] if _ik_v.dim() > 1 else _ik_v
                        print(f"  [IK_DIAG] target_pos[{_ik_k}] = {_ik_v0.tolist()}")
                    for _ik_k, _ik_v in new_target_quat.items():
                        _ik_v0 = _ik_v[0] if _ik_v.dim() > 1 else _ik_v
                        print(f"  [IK_DIAG] target_quat[{_ik_k}] = {_ik_v0.tolist()}")
                    _cur_l_pos, _cur_l_ori = robot.get_eef_pose("left")
                    _cur_r_pos, _cur_r_ori = robot.get_eef_pose("right")
                    print(f"  [IK_DIAG] cur left  EEF pos={_cur_l_pos.tolist()}, ori={_cur_l_ori.tolist()}")
                    print(f"  [IK_DIAG] cur right EEF pos={_cur_r_pos.tolist()}, ori={_cur_r_ori.tolist()}")
                    _base_pos, _ = robot.get_position_orientation()
                    print(f"  [IK_DIAG] robot base pos={_base_pos[:3].tolist()}")
                    try:
                        _jpos = robot.get_joint_positions()
                        _jlim_lo = robot.joint_lower_limits
                        _jlim_hi = robot.joint_upper_limits
                        _near = th.where((_jpos < _jlim_lo + 0.05) | (_jpos > _jlim_hi - 0.05))[0]
                        if len(_near) > 0:
                            print(f"  [IK_DIAG] joints near limits: idx={_near.tolist()}, val={_jpos[_near].tolist()}")
                        else:
                            print(f"  [IK_DIAG] no joints near limits")
                    except Exception as _diag_e:
                        print(f"  [IK_DIAG] joint limit check error: {_diag_e}")
                    # ===== CuRobo 精细诊断：bound / self-coll / env-coll =====
                    try:
                        env.cmg.debug_start_state(emb_sel=emb_sel)
                    except Exception as _dbg_e:
                        print(f"  [DEBUG_START] error: {_dbg_e}")
                    # ===== [END IK诊断补丁①] =====
                    # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
                    ignore_obstacle_links = {}
                    is_drawer_storage = str(getattr(env, "name", "")).startswith(
                        "openarm_drawer_storage"
                    )
                    is_fruit_bagging = str(getattr(env, "name", "")).startswith(
                        "openarm_fruit_basket_bagging"
                    )
                    if is_drawer_storage and (
                            getattr(env, "generation_phase_action", None) == "open"
                            or (
                                getattr(env, "generation_phase_action", None) == "pick"
                                and getattr(env, "generation_previous_phase_action", None) == "insert"
                            )
                    ):
                        drawer = env.base_env.scene.object_registry("name", "drawer_cabinet_1")
                        ignore_obstacle_links["drawer_cabinet_1"] = {
                            link.name for link in drawer.links.values()
                        }
                        print(
                            "[openarm_drawer_storage] relaxed CuRobo cabinet contact "
                            f"for {env.generation_previous_phase_action} -> "
                            f"{env.generation_phase_action} MP: "
                            f"{sorted(ignore_obstacle_links['drawer_cabinet_1'])}"
                        )
                    if is_drawer_storage and getattr(env, "generation_phase_action", None) == "pick":
                        pick_object_name = self._select_reference_object(object_ref)
                        pick_object = env.base_env.scene.object_registry("name", pick_object_name)
                        ignore_obstacle_links[pick_object_name] = {
                            link.name for link in pick_object.links.values()
                        }
                        print(
                            "[openarm_drawer_storage] relaxed CuRobo pick-target contact: "
                            f"{pick_object_name}"
                        )
                    if is_fruit_bagging:
                        fruit_phase_action = getattr(env, "generation_phase_action", None)
                        if fruit_phase_action == "pick":
                            pick_object_name = self._select_reference_object(object_ref)
                            pick_object = env.base_env.scene.object_registry("name", pick_object_name)
                            ignore_obstacle_links[pick_object_name] = {
                                link.name for link in pick_object.links.values()
                            }
                            print(
                                "[openarm_fruit_basket_bagging] relaxed CuRobo pick-target contact: "
                                f"{pick_object_name}"
                            )
                    if not ignore_obstacle_links:
                        ignore_obstacle_links = None
                    mp_results, traj_paths = env.cmg.compute_trajectories(
                        target_pos=new_target_pos,
                        target_quat=new_target_quat,
                        is_local=False,
                        max_attempts=50,
                        timeout=60.0,
                        ik_fail_return=50,
                        enable_finetune_trajopt=True,
                        finetune_attempts=1,
                            return_full_result=True,
                        success_ratio=1.0 / env.primitive._motion_generator.batch_size,
                        attached_obj=attached_obj,
                        attached_obj_scale=attached_obj_scale,
                        emb_sel=emb_sel,
                        eyes_target_pos=eyes_target_pos,
                        eyes_target_quat=eyes_target_quat,
                        ignore_obstacle_links=ignore_obstacle_links,
                        )
                    arm_mp_planning_finish_time = time.time()
                    phase_logs[env.execution_phase_ind]["arm_mp_planning_time"][arm_mp_trial] = round(arm_mp_planning_finish_time - arm_mp_planning_start_time, 2)

                    successes = mp_results[0].success 
                    print("Arm MP successes: ", successes)
                    success_idx = th.where(successes)[0].cpu()
                    
                    if len(success_idx) == 0:
                        print(f"Arm MP trial {arm_mp_trial} failed with status {mp_results[0].status}. Retrying...")
                        # ===== [IK诊断补丁②] IK失败后打印详细错误 =====
                        try:
                            _r = mp_results[0]
                            print(f"  [IK_DIAG] valid_query={_r.valid_query}")
                            print(f"  [IK_DIAG] ik_time={_r.ik_time:.3f}s, total_time={_r.total_time:.3f}s")
                            if hasattr(_r, 'position_error') and _r.position_error is not None:
                                print(f"  [IK_DIAG] pos_err={_r.position_error}, rot_err={_r.rotation_error}")
                            if hasattr(_r, 'attempts') and _r.attempts is not None:
                                print(f"  [IK_DIAG] attempts={_r.attempts}")
                            if (
                                getattr(env, "name", "").startswith("openarm_real_exp_1")
                                and src_curr_phase_actions is not None
                                and arm_mp_trial == 0
                            ):
                                _left_link = robot.eef_link_names["left"]
                                _right_link = robot.eef_link_names["right"]
                                if _left_link in target_pos and _right_link not in target_pos:
                                    _src_target_idx = cur_subtask_end_step_MP[0] - 1
                                elif _right_link in target_pos and _left_link not in target_pos:
                                    _src_target_idx = cur_subtask_end_step_MP[1] - 1
                                else:
                                    _src_target_idx = max(cur_subtask_end_step_MP) - 1
                                _src_target_idx = max(0, min(len(src_curr_phase_actions) - 1, _src_target_idx))
                                env.cmg.debug_openarm_source_joint_target(
                                    src_action=src_curr_phase_actions[_src_target_idx],
                                    target_pos=target_pos,
                                    target_quat=target_quat,
                                    emb_sel=emb_sel,
                                    label=f"source_action[{_src_target_idx}]",
                                )
                        except Exception as _diag_e:
                            print(f"  [IK_DIAG] read error: {_diag_e}")
                        # ===== [END IK诊断补丁②] =====
                        arm_mp_trial += 1
                        # modify target_pos a bit
                        for k in target_pos.keys():
                            new_target_pos[k] = target_pos[k] + th.rand(3) * 0.01 - 0.005
                        continue
                    else:
                        traj_path = traj_paths[success_idx[0]]
                        break
            
                print("Time taken for arm MP planning: ", phase_logs[env.execution_phase_ind]["arm_mp_planning_time"])
                # ========================================================= End of Arm MP Planning ==========================================================
                
                # ========================================================== Arm MP Execution ==========================================================
                # reset the visibility counter for each sensor
                self.reset_visibility_counter(env)

                arm_mp_execution_start_time = time.time()

                # These lines are for debugging purposes.
                # successes, traj_paths = env.cmg.compute_trajectories(target_pos=target_pos, target_quat=target_quat, is_local=False, max_attempts=50, timeout=60.0, ik_fail_return=5, enable_finetune_trajopt=True, finetune_attempts=1, return_full_result=False, success_ratio=1.0, attached_obj=attached_obj, attached_obj_scale=attached_obj_scale, emb_sel=emb_sel)
                # full_result = env.cmg.compute_trajectories(target_pos=target_pos, target_quat=target_quat, is_local=False, max_attempts=50, timeout=60.0, ik_fail_return=5, enable_finetune_trajopt=True, finetune_attempts=1, return_full_result=True, success_ratio=1.0, attached_obj=attached_obj, attached_obj_scale=attached_obj_scale, emb_sel=emb_sel)

                # Convert planned joint trajectory to actions
                # Need to call q_to_action after every env.step if the base is moving; we cannot pre-compute all actions
                q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                # If we use curobo joint space planning instead of Cartesian space planning, we need to downsample the trajectory 
                # q_traj = q_traj[::50]
                q_traj = th.stack(
                    env.primitive._add_linearly_interpolated_waypoints(
                        plan=q_traj,
                        max_inter_dist=arm_mp_max_inter_dist,
                    )
                )
                q_traj = q_traj.cpu()
                mp_actions = []
                for j_pos in q_traj:

                    # If option 2 was chosen for handling arm with no ref object, we can make the action for that arm as 0
                    if object_ref["arm_left"] is None:
                        j_pos[robot.arm_control_idx["left"]] = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                    elif object_ref["arm_right"] is None:
                        j_pos[robot.arm_control_idx["right"]] = robot.get_joint_positions()[robot.arm_control_idx["right"]]

                    action = robot.q_to_action(j_pos).cpu().numpy()

                    # Add gripper actions from the original waypoints (we already checked that they are the same across MP trajectories)
                    if left_gripper_action is not None:
                        self._set_openarm_gripper_target(
                            action, env_interface.gripper_action_dim[0], "left",
                            left_gripper_action[0], gripper_interp_state,
                        )
                    if right_gripper_action is not None:
                        self._set_openarm_gripper_target(
                            action, env_interface.gripper_action_dim[1], "right",
                            right_gripper_action[1], gripper_interp_state,
                        )
                    action = self._freeze_openarm_source_replay_action(
                        action=action,
                        robot=robot,
                        frozen_arms=frozen_arms,
                    )
                    
                    mp_actions.append(action)

                left_eef_poses = [(left_waypoint_pos, left_waypoint_ori)] * len(mp_actions)
                right_eef_poses = [(right_waypoint_pos, right_waypoint_ori)] * len(mp_actions)

                # If the left hand has no motion planner waypoints, we start replaying the left hand waypoints while the right hand are following the MP trajectory.
                if len(left_mp_waypoints) == 0:
                    # We need to pad the left hand waypoints to match the length of the MP trajectory
                    if len(left_replay_waypoints) < len(mp_actions):
                        for _ in range(len(mp_actions) - len(left_replay_waypoints)):
                            left_replay_waypoints.append(last_waypoint)

                    left_eef_poses = []
                    # We convert the target pose of the left hand to replay_action
                    # Then we *overwrite* the motion planner action with the replay action for the left arm and gripper
                    for i, action in enumerate(mp_actions):
                        replay_action = env_interface.target_pose_to_action(target_pose=left_replay_waypoints[i].pose)
                        left_eef_poses.append((left_replay_waypoints[i].pose[0:3, 3], T.mat2quat(th.tensor(left_replay_waypoints[i].pose[0:3, 0:3]))))
                        action_idx = robot.controller_action_idx["arm_left"]
                        action[action_idx] = replay_action[action_idx]
                        self._set_openarm_gripper_target(
                            action, env_interface.gripper_action_dim[0], "left",
                            left_replay_waypoints[i].gripper_action[0], gripper_interp_state,
                        )

                    # We remove the waypoints that have been replayed for the left arm
                    left_replay_waypoints = left_replay_waypoints[len(mp_actions):]

                # Same logic as above but for the right hand
                elif len(right_mp_waypoints) == 0:
                    if len(right_replay_waypoints) < len(mp_actions):
                        for _ in range(len(mp_actions) - len(right_replay_waypoints)):
                            right_replay_waypoints.append(last_waypoint)
                    right_eef_poses = []
                    for i, action in enumerate(mp_actions):
                        replay_action = env_interface.target_pose_to_action(target_pose=right_replay_waypoints[i].pose)
                        right_eef_poses.append((right_replay_waypoints[i].pose[4:7, 3], T.mat2quat(th.tensor(right_replay_waypoints[i].pose[4:7, 0:3]))))
                        action_idx = robot.controller_action_idx["arm_right"]
                        action[action_idx] = replay_action[action_idx]
                        self._set_openarm_gripper_target(
                            action, env_interface.gripper_action_dim[1], "right",
                            right_replay_waypoints[i].gripper_action[1], gripper_interp_state,
                        )

                    right_replay_waypoints = right_replay_waypoints[len(mp_actions):]

                assert len(mp_actions) == len(left_eef_poses) == len(right_eef_poses)

                init_global_env_step = env.global_env_step
                num_repeat = 1
                for i, mp_action in enumerate(mp_actions):
                    for _ in range(num_repeat):
                        mp_action = self._hold_openarm_attached_grippers(
                            action=mp_action,
                            robot=robot,
                            attached_obj=attached_obj,
                            interp_state=gripper_interp_state,
                            phase_type=phase_type,
                        )
                        mp_action = self._close_openarm_pick_grippers(
                            action=mp_action,
                            robot=robot,
                            object_ref=object_ref,
                            phase_type=phase_type,
                            interp_state=gripper_interp_state,
                        )
                        mp_action = env.primitive._postprocess_action(mp_action)
                        mp_action = self._lock_openarm_attached_gripper_state(
                            robot=robot,
                            attached_obj=attached_obj,
                            action=mp_action,
                            interp_state=gripper_interp_state,
                        )
                        state = env.get_state()["states"]
                        obs, obs_info = env.get_obs_IL()
                        # TODO: Check if we can use primitive stack execute action here. This will allow for checking convergence errors etc.
                        datagen_info = env_interface.get_datagen_info(action=mp_action)
                        env.step(mp_action, video_writer)
                        if enable_marker_vis:
                            env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                            env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                            env.eef_goal_marker_left.set_position_orientation(*left_eef_poses[i])
                            env.eef_goal_marker_right.set_position_orientation(*right_eef_poses[i])
                        local_env_step += 1
                        env.global_env_step += 1
                        states.append(state)
                        actions.append(mp_action)
                        observations.append(obs)
                        observations_info.append(json.dumps(obs_info))
                        datagen_infos.append(datagen_info)
                        cur_success_metrics = env.is_success()
                        self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                        for k in success:
                            success[k] = success[k] or cur_success_metrics[k]

                # # If using MP in default mode. Will remove this code later but keeping it for now for debugging purposes  
                # q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                # q_traj = th.stack(env.primitive._add_linearly_interpolated_waypoints(plan=q_traj, max_inter_dist=0.01))
                # q_traj = q_traj.cpu()
                # left_eef_poses = [(left_waypoint_pos, left_waypoint_ori)] * len(q_traj)
                # right_eef_poses = [(right_waypoint_pos, right_waypoint_ori)] * len(q_traj)
                # num_repeat = 1
                # for i, j_pos in enumerate(q_traj):
                #     for _ in range(num_repeat):
                #         action = robot.q_to_action(j_pos).cpu().numpy()
                #         if left_gripper_action is not None:
                #             action[env_interface.gripper_action_dim[0]] = left_gripper_action[0]
                #         if right_gripper_action is not None:
                #             action[env_interface.gripper_action_dim[1]] = right_gripper_action[1]
                #         state = env.get_state()["states"]
                #         # obs, obs_info = env.get_obs_IL()
                #         datagen_info = env_interface.get_datagen_info(action=action)
                #         env.step(action)
                #         env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                #         env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                #         env.eef_goal_marker_left.set_position_orientation(*left_eef_poses[i])
                #         env.eef_goal_marker_right.set_position_orientation(*right_eef_poses[i])
                #         local_env_step += 1
                #         states.append(state)
                #         actions.append(action)
                #         observations.append(obs)
                #         datagen_infos.append(datagen_info)


            # Set the MP ranges to save to hdf5 file
            left_mp_ranges, right_mp_ranges = None, None
            if len(left_mp_waypoints) > 0:
                left_mp_ranges = [init_global_env_step, env.global_env_step]
            if len(right_mp_waypoints) > 0:
                right_mp_ranges = [init_global_env_step, env.global_env_step]
            
            
            MP_end_step_local = copy.deepcopy(local_env_step)
            # left MP points
            if len(left_mp_waypoints) == 0: 
                left_MP_end_step_local = 0
            else: 
                left_MP_end_step_local = MP_end_step_local
            if len(right_mp_waypoints) == 0: 
                right_MP_end_step_local = 0
            else: 
                right_MP_end_step_local = MP_end_step_local

            MP_end_step_local_list = [left_MP_end_step_local, right_MP_end_step_local]

            arm_mp_execution_finish_time = time.time()
            # Since there is only 1 trial for arm MP execution, we set the 0th index
            phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0] = round(arm_mp_execution_finish_time - arm_mp_execution_start_time, 2)
            print("Time taken for arm MP execution:", phase_logs[env.execution_phase_ind]["arm_mp_execution_time"][0])
            
            num_phase_steps = env.global_env_step - init_global_env_step
            for sensor_name, sensor in env.robot.sensors.items():
                if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                    shortened_sensor_name = sensor_name.split(":")[1]
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"]= 0
                    print(f"Visibility stats for arm_mp {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_{shortened_sensor_name}"])
            if num_phase_steps > 0:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
            else:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"]= 0
            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_steps"] = num_phase_steps
            print(f"Visibility stats for arm_mp any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_mp_any"])

            # ============================================== End of Arm MP Execution ==========================================================
            
            # ================================================== Arm Replay ==========================================================
            # reset the visibility counter for each sensor
            self.reset_visibility_counter(env)
            
            # We need to pad the waypoints for the left and right hands to match the length of the longest trajectory
            if len(left_replay_waypoints) < len(right_replay_waypoints):
                for _ in range(len(right_replay_waypoints) - len(left_replay_waypoints)):
                    left_replay_waypoints.append(last_waypoint)
            elif len(right_replay_waypoints) < len(left_replay_waypoints):
                for _ in range(len(left_replay_waypoints) - len(right_replay_waypoints)):
                    right_replay_waypoints.append(last_waypoint)

            assert len(left_replay_waypoints) == len(right_replay_waypoints)
            # print('length of replay actions:', len(left_replay_waypoints))
            print("ARM REPLAY START")
            arm_replay_start_time = time.time()
            replay_gripper_schedule = self._make_openarm_replay_gripper_schedule(
                src_actions=src_curr_phase_actions,
                robot=robot,
                mp_end_steps=cur_subtask_end_step_MP,
                object_ref=object_ref,
            )
            
            # If one of the arms has no ref object, we set its target pose as the current pose
            if object_ref["arm_right"] is None:
                current_right_ee_pose = robot.get_eef_pose("right")
                current_right_ee_pos = current_right_ee_pose[0]
                current_right_ee_quat = current_right_ee_pose[1]
                current_right_ee_matrix = T.quat2mat(current_right_ee_quat)
                current_right_ee_pose = th.eye(4)
                current_right_ee_pose[:3, :3] = current_right_ee_matrix
                current_right_ee_pose[:3, 3] = current_right_ee_pos
            elif object_ref["arm_left"] is None:
                current_left_ee_pose = robot.get_eef_pose("left")
                current_left_ee_pos = current_left_ee_pose[0]
                current_left_ee_quat = current_left_ee_pose[1]
                current_left_ee_matrix = T.quat2mat(current_left_ee_quat)
                current_left_ee_pose = th.eye(4)
                current_left_ee_pose[:3, :3] = current_left_ee_matrix
                current_left_ee_pose[:3, 3] = current_left_ee_pos
            
            init_global_env_step = env.global_env_step
            # For each pair of waypoints, we extract the pose for each hand and then convert to action
            # We also overwrite the gripper actions with the ones from the waypoints
            for replay_i, (left_waypoint, right_waypoint) in enumerate(zip(left_replay_waypoints, right_replay_waypoints)):
                pose = np.zeros((8, 4))
                pose[:4, :] = left_waypoint.pose[:4, :]
                pose[4:, :] = right_waypoint.pose[4:, :]
                # If one of the arms has no ref object, we set its target pose as the current pose
                if object_ref["arm_right"] is None:
                    pose[4:, :] = current_right_ee_pose
                elif object_ref["arm_left"] is None:
                    pose[:4, :] = current_left_ee_pose
                replay_action = env_interface.target_pose_to_action(target_pose=pose)

                self._set_openarm_gripper_target(
                    replay_action, env_interface.gripper_action_dim[0], "left",
                    left_waypoint.gripper_action[0], gripper_interp_state,
                )
                self._set_openarm_gripper_target(
                    replay_action, env_interface.gripper_action_dim[1], "right",
                    right_waypoint.gripper_action[1], gripper_interp_state,
                )
                replay_action = self._apply_openarm_replay_gripper_schedule(
                    robot=robot,
                    action=replay_action,
                    replay_i=replay_i,
                    schedule=replay_gripper_schedule,
                    replay_relative=True,
                )
                replay_action = self._freeze_openarm_source_replay_action(
                    action=replay_action,
                    robot=robot,
                    frozen_arms=frozen_arms,
                )

                state = env.get_state()["states"]
                temp_start_time = time.time()
                obs, obs_info = env.get_obs_IL()
                datagen_info = env_interface.get_datagen_info(action=replay_action)
                env.step(replay_action, video_writer)
                left_eef_pose = (pose[0:3, 3], T.mat2quat(th.tensor(pose[0:3, 0:3])))
                right_eef_pose = (pose[4:7, 3], T.mat2quat(th.tensor(pose[4:7, 0:3])))
                if enable_marker_vis:
                    env.eef_current_marker_left.set_position_orientation(*robot.get_eef_pose("left"))
                    env.eef_current_marker_right.set_position_orientation(*robot.get_eef_pose("right"))
                    env.eef_goal_marker_left.set_position_orientation(*left_eef_pose)
                    env.eef_goal_marker_right.set_position_orientation(*right_eef_pose)
                local_env_step += 1
                env.global_env_step += 1
                states.append(state)
                actions.append(replay_action)
                observations.append(obs)
                observations_info.append(json.dumps(obs_info))
                datagen_infos.append(datagen_info)
                cur_success_metrics = env.is_success()
                self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]

            arm_replay_finish_time = time.time()
            phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0] = round(arm_replay_finish_time - arm_replay_start_time, 2)
            print("Time taken for arm replay: ", phase_logs[env.execution_phase_ind]["arm_replay_execution_time"][0])
            
            num_phase_steps = env.global_env_step - init_global_env_step
            for sensor_name, sensor in env.robot.sensors.items():
                if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                    shortened_sensor_name = sensor_name.split(":")[1]
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"]= 0
                    print(f"Visibility stats for arm_replay {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_{shortened_sensor_name}"])
            if num_phase_steps > 0:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
            else:
                phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"]= 0
            phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_steps"] = num_phase_steps
            print(f"Visibility stats for arm_replay any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"arm_replay_any"])

            # =================================================== End of Arm Replay ==========================================================

            # =================================================== Arm/Torso Retract ==========================================================
            if retract_type != "no_retract":
                print("Starting Retract")
                
                # reset the visibility counter for each sensor
                self.reset_visibility_counter(env)
                
                retract_torso_only = False
                current_robot_base_pose_wrt_world = robot.get_position_orientation()
                left_eef_link_name = robot.eef_link_names["left"]
                right_eef_link_name = robot.eef_link_names["right"]
                has_eyes = hasattr(robot, "links") and ("eyes" in robot.links) and hasattr(env, "eyes_reset_pose_wrt_robot")
                # If we retract the left and right eef to the pose at the start of arm MP
                if retract_type == "retract_to_start_of_arm_mp":
                    if object_ref["arm_right"] is None:
                        arm_side = "left"
                        current_left_eef_pose = robot.get_eef_pose("left")
                        target_pos = {left_eef_link_name: initial_left_eef_pose[0]}
                        target_quat = {left_eef_link_name: initial_left_eef_pose[1]}
                    elif object_ref["arm_left"] is None:
                        arm_side = "right"
                        current_right_eef_pose = robot.get_eef_pose("right")
                        target_pos = {right_eef_link_name: initial_right_eef_pose[0]}
                        target_quat = {right_eef_link_name: initial_right_eef_pose[1]}
                    # TODO: implement this. Not too important for now as this would never happen. In this case it's a bimanual coordinated and we don't need to retract
                    else:
                        pass

                # If we retract the left and right eef and eyes to a canonical pose
                elif retract_type == "retract_to_canonical_pose":
                    if has_eyes:
                        eyes_reset_pose_wrt_world = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.eyes_reset_pose_wrt_robot)
                        eyes_reset_pose_wrt_world = T.mat2pose(eyes_reset_pose_wrt_world)

                    left_eef_reset_pose_wrt_robot = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.left_eef_reset_pose_wrt_robot)
                    left_eef_reset_pose_wrt_robot = T.mat2pose(left_eef_reset_pose_wrt_robot)

                    right_eef_reset_pose_wrt_robot = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.right_eef_reset_pose_wrt_robot)
                    right_eef_reset_pose_wrt_robot = T.mat2pose(right_eef_reset_pose_wrt_robot)

                    target_pos = {
                        left_eef_link_name: left_eef_reset_pose_wrt_robot[0],
                        right_eef_link_name: right_eef_reset_pose_wrt_robot[0],
                    }
                    target_quat = {
                        left_eef_link_name: left_eef_reset_pose_wrt_robot[1],
                        right_eef_link_name: right_eef_reset_pose_wrt_robot[1],
                    }
                    if has_eyes:
                        target_pos["eyes"] = eyes_reset_pose_wrt_world[0]
                        target_quat["eyes"] = eyes_reset_pose_wrt_world[1]

                elif retract_type == "retract_to_canonical_pose_maintain_orn":
                    if has_eyes:
                        eyes_reset_pose_wrt_world = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.eyes_reset_pose_wrt_robot)
                        eyes_reset_pose_wrt_world = T.mat2pose(eyes_reset_pose_wrt_world)

                    left_eef_reset_pose_wrt_robot = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.left_eef_reset_pose_wrt_robot)
                    left_eef_reset_pose_wrt_robot = T.mat2pose(left_eef_reset_pose_wrt_robot)
                    current_left_eef_pose = robot.get_eef_pose("left")

                    right_eef_reset_pose_wrt_robot = T.pose2mat(current_robot_base_pose_wrt_world) @ T.pose2mat(env.right_eef_reset_pose_wrt_robot)
                    right_eef_reset_pose_wrt_robot = T.mat2pose(right_eef_reset_pose_wrt_robot)
                    current_right_eef_pose = robot.get_eef_pose("right")

                    target_pos = {
                        left_eef_link_name: left_eef_reset_pose_wrt_robot[0],
                        right_eef_link_name: right_eef_reset_pose_wrt_robot[0],
                    }
                    target_quat = {
                        left_eef_link_name: current_left_eef_pose[1],
                        right_eef_link_name: current_right_eef_pose[1],
                    }
                    if has_eyes:
                        target_pos["eyes"] = eyes_reset_pose_wrt_world[0]
                        target_quat["eyes"] = eyes_reset_pose_wrt_world[1]

                else:
                    raise ValueError(f"Invalid retract type: {retract_type}")


                # Aggregate target_pos and target_quat to match batch_size
                new_target_pos = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_pos.items()}
                new_target_quat = {
                    k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_quat.items()
                }
                
                retval = self.obtain_attached_object(env, robot)
                grasp_action = retval["grasp_action"]
                attached_obj = retval["attached_obj"]
                attached_obj_scale = retval["attached_obj_scale"]

                # if enable_marker_vis:
                #     if arm_side == "left":
                #         env.eef_goal_marker_left.set_position_orientation(target_pos["left_eef_link"], target_quat["left_eef_link"])
                #     elif arm_side == "right":
                #         env.eef_goal_marker_right.set_position_orientation(target_pos["right_eef_link"], target_quat["right_eef_link"])
                
                if retract_type == "retract_to_start_of_arm_mp":
                    emb_sel = CuRoboEmbodimentSelection.ARM_NO_TORSO
                elif retract_type == "retract_to_canonical_pose":
                    emb_sel = CuRoboEmbodimentSelection.ARM
                elif retract_type == "retract_to_canonical_pose_maintain_orn":
                    emb_sel = CuRoboEmbodimentSelection.ARM

                full_retract_mp_planning_start_time = time.time()
                mp_results, traj_paths = env.cmg.compute_trajectories(
                    target_pos=new_target_pos,
                    target_quat=new_target_quat,
                    is_local=False,
                    max_attempts=50,
                    timeout=20.0,
                    ik_fail_return=50,
                    enable_finetune_trajopt=True,
                    finetune_attempts=1,
                    return_full_result=True,
                    success_ratio=1.0 / env.primitive._motion_generator.batch_size,
                    attached_obj=attached_obj,
                    attached_obj_scale=attached_obj_scale,
                    emb_sel=emb_sel,
                )
                full_retract_mp_planning_finish_time = time.time()
                phase_logs[env.execution_phase_ind]["full_retract_mp_planning_time"][0] = round(full_retract_mp_planning_finish_time - full_retract_mp_planning_start_time, 2)
                print("Time taken for full retract MP planning: ", phase_logs[env.execution_phase_ind]["full_retract_mp_planning_time"][0])

                successes = mp_results[0].success 
                print("Retract Arm MP successes: ", successes)
                success_idx = th.where(successes)[0].cpu()

                if len(success_idx) == 0:
                    print(f"Arm retract failed with status {mp_results[0].status}.")
                    phase_logs[env.execution_phase_ind]["full_retract_mp_err"][0] = mp_results[0].status.value
                    retract_torso_only = True
                else:
                    phase_logs[env.execution_phase_ind]["full_retract_mp_err"][0] = "None"
                    full_retract_mp_execution_start_time = time.time()
                    traj_path = traj_paths[success_idx[0]]

                    q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                    q_traj = th.stack(
                        env.primitive._add_linearly_interpolated_waypoints(
                            plan=q_traj,
                            max_inter_dist=self._arm_mp_max_inter_dist(retract=True, env=env),
                        )
                    )
                    q_traj = q_traj.cpu()
                    if (
                        isinstance(robot, OpenArmBimanual)
                        and os.environ.get(
                            "OPENARM_RETRACT_TO_INITIAL_JOINTS",
                            os.environ.get("OPENARM_RETRACT_TO_SOURCE_JOINTS", "1"),
                        ).lower() not in {"0", "false", "no"}
                    ):
                        initial_joint_target = getattr(env, "openarm_generation_initial_joint_positions", None)
                        q_traj = self._append_joint_space_target(
                            q_traj=q_traj,
                            q_target=initial_joint_target,
                            max_inter_dist=self._arm_mp_max_inter_dist(retract=True, env=env),
                        )
                        phase_logs[env.execution_phase_ind]["initial_joint_retract"] = bool(initial_joint_target is not None)
                        print("[OpenArm] retract joint_target=generation_initial_joints")

                    num_repeat = 1
                    init_left_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                    init_right_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["right"]]
                    init_global_env_step = env.global_env_step
                    for j_pos in q_traj:
                        if retract_type == "retract_to_start_of_arm_mp":
                            if arm_side == "left":
                                j_pos[robot.arm_control_idx["right"]] = init_right_arm_pos
                            elif arm_side == "right":
                                j_pos[robot.arm_control_idx["left"]] = init_left_arm_pos

                        mp_action = robot.q_to_action(j_pos).cpu().numpy()
                        self._set_openarm_gripper_target(
                            mp_action, robot.gripper_action_idx["left"], "left",
                            grasp_action["left"], gripper_interp_state,
                        )
                        self._set_openarm_gripper_target(
                            mp_action, robot.gripper_action_idx["right"], "right",
                            grasp_action["right"], gripper_interp_state,
                        )
                        mp_action = self._lock_openarm_attached_gripper_state(
                            robot=robot,
                            attached_obj=attached_obj,
                            action=mp_action,
                            interp_state=gripper_interp_state,
                        )

                        state = env.get_state()["states"]
                        obs, obs_info = env.get_obs_IL()
                        datagen_info = env_interface.get_datagen_info(action=mp_action)
                        env.step(mp_action, video_writer)
                        local_env_step += 1
                        env.global_env_step += 1
                        states.append(state)
                        actions.append(mp_action)
                        observations.append(obs)
                        observations_info.append(json.dumps(obs_info))
                        datagen_infos.append(datagen_info)
                        cur_success_metrics = env.is_success()
                        self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                        for k in success:
                            success[k] = success[k] or cur_success_metrics[k]

                    full_retract_mp_execution_finish_time = time.time()
                    phase_logs[env.execution_phase_ind]["full_retract_mp_execution_time"][0] = round(full_retract_mp_execution_finish_time - full_retract_mp_execution_start_time, 2)

                    num_phase_steps = env.global_env_step - init_global_env_step
                    for sensor_name, sensor in env.robot.sensors.items():
                        if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                            shortened_sensor_name = sensor_name.split(":")[1]
                            if num_phase_steps > 0:
                                phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                            else:
                                phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_{shortened_sensor_name}"]= 0
                            print(f"Visibility stats for full_retract {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_{shortened_sensor_name}"])
                    if num_phase_steps > 0:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                    else:
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_any"]= 0
                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_steps"] = num_phase_steps
                    print(f"Visibility stats for full_retract any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"full_retract_any"])

                # If full retract failed, try retracting only the torso / eyes.
                if retract_torso_only and retract_type != "retract_to_start_of_arm_mp" and not has_eyes:
                    print("Skipping torso-only retract because this robot has no eyes target.")
                    phase_logs[env.execution_phase_ind]["torso_retract_mp_err"][0] = "no_eyes_target"
                    retract_torso_only = False

                if retract_torso_only and retract_type != "retract_to_start_of_arm_mp":
                    print("Retracting torso only")
                    
                    # reset the visibility counter for each sensor
                    self.reset_visibility_counter(env)
                    
                    target_pos = {"eyes": eyes_reset_pose_wrt_world[0]}
                    target_quat = {"eyes": eyes_reset_pose_wrt_world[1]}

                    new_target_pos = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_pos.items()}
                    new_target_quat = {k: th.stack([v for _ in range(env.primitive._motion_generator.batch_size)]) for k, v in target_quat.items()}

                    torso_retract_mp_planning_start_time = time.time()
                    mp_results, traj_paths = env.cmg.compute_trajectories(
                        target_pos=new_target_pos,
                        target_quat=new_target_quat,
                        is_local=False,
                        max_attempts=50,
                        timeout=20.0,
                        ik_fail_return=50,
                        enable_finetune_trajopt=True,
                        finetune_attempts=1,
                        return_full_result=True,
                        success_ratio=1.0 / env.primitive._motion_generator.batch_size,
                        attached_obj=attached_obj,
                        attached_obj_scale=attached_obj_scale,
                        emb_sel=emb_sel,
                    )
                    torso_retract_mp_planning_finish_time = time.time()
                    phase_logs[env.execution_phase_ind]["torso_retract_mp_planning_time"][0] = round(torso_retract_mp_planning_finish_time - torso_retract_mp_planning_start_time, 2)

                    successes = mp_results[0].success 
                    print("Torso-only retract: Arm MP successes: ", successes)
                    success_idx = th.where(successes)[0].cpu()

                    if len(success_idx) == 0:
                        print(f"Torso retract failed with status {mp_results[0].status}.")
                        phase_logs[env.execution_phase_ind]["torso_retract_mp_err"][0] = mp_results[0].status.value
                    else:
                        phase_logs[env.execution_phase_ind]["torso_retract_mp_err"][0] = "None"
                        torso_retract_mp_execution_start_time = time.time()
                        traj_path = traj_paths[success_idx[0]]

                        q_traj = env.cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel)
                        q_traj = th.stack(
                            env.primitive._add_linearly_interpolated_waypoints(
                                plan=q_traj,
                                max_inter_dist=self._arm_mp_max_inter_dist(retract=True, env=env),
                            )
                        )
                        q_traj = q_traj.cpu()

                        num_repeat = 1
                        init_left_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["left"]]
                        init_right_arm_pos = robot.get_joint_positions()[robot.arm_control_idx["right"]]
                        init_global_env_step = env.global_env_step
                        for j_pos in q_traj:
                            mp_action = robot.q_to_action(j_pos).cpu().numpy()
                            self._set_openarm_gripper_target(
                                mp_action, robot.gripper_action_idx["left"], "left",
                                grasp_action["left"], gripper_interp_state,
                            )
                            self._set_openarm_gripper_target(
                                mp_action, robot.gripper_action_idx["right"], "right",
                                grasp_action["right"], gripper_interp_state,
                            )
                            # Don't want to move the arm relative to the torso
                            mp_action[robot.arm_action_idx["right"]] = init_right_arm_pos
                            mp_action[robot.arm_action_idx["left"]] = init_left_arm_pos

                            state = env.get_state()["states"]
                            obs, obs_info = env.get_obs_IL()
                            datagen_info = env_interface.get_datagen_info(action=mp_action)
                            env.step(mp_action, video_writer)
                            local_env_step += 1
                            env.global_env_step += 1
                            states.append(state)
                            actions.append(mp_action)
                            observations.append(obs)
                            observations_info.append(json.dumps(obs_info))
                            datagen_infos.append(datagen_info)
                            cur_success_metrics = env.is_success()
                            self.check_ref_obj_visibility(env, obs, obs_info, ref_obj)
                            for k in success:
                                success[k] = success[k] or cur_success_metrics[k]
                        
                        torso_retract_mp_execution_finish_time = time.time()
                        phase_logs[env.execution_phase_ind]["torso_retract_mp_execution_time"][0] = round(torso_retract_mp_execution_finish_time - torso_retract_mp_execution_start_time, 2)
                        
                        num_phase_steps = env.global_env_step - init_global_env_step
                        for sensor_name, sensor in env.robot.sensors.items():
                            if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                                shortened_sensor_name = sensor_name.split(":")[1]
                                if num_phase_steps > 0:
                                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_{shortened_sensor_name}"] = env.num_frames_with_obj_visible[shortened_sensor_name] / num_phase_steps
                                else:
                                    phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_{shortened_sensor_name}"]= 0
                                print(f"Visibility stats for torso_retract {shortened_sensor_name}: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_{shortened_sensor_name}"])
                        if num_phase_steps > 0:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_any"] = env.num_frames_with_obj_visible["any"] / num_phase_steps
                        else:
                            phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_any"]= 0
                        phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_steps"] = num_phase_steps
                        print(f"Visibility stats for torso_retract any: ", phase_logs[env.execution_phase_ind]["visibility_stats"][f"torso_retract_any"])

            # ================================================== End of Arm/Torso Retract ==========================================================
                    
            results = dict(
                states=states,
                observations=observations,
                datagen_infos=datagen_infos,
                actions=np.array(actions),
                success=bool(success["task"]),
                mp_end_steps=MP_end_step_local_list,
                subtask_lengths=local_env_step,
                left_mp_ranges=left_mp_ranges,
                right_mp_ranges=right_mp_ranges,
                retry_nav=False,
                observations_info=observations_info
            )
            env.execution_phase_ind += 1
            env.phases_completed_wo_mp_err += 1
            return results
