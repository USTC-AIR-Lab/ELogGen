"""
Base class for data generator.
"""
import contextlib
import os
from copy import deepcopy
import numpy as np
import torch as th

import eloggen.generation_runtime.geometry as PoseUtils
import eloggen.generation_runtime.datasets as DatasetUtils

from eloggen.generation_runtime.config.task_spec import EG_TaskSpec
from eloggen.generation_runtime.config.subtask_graph import SubtaskGraph
from eloggen.generation_runtime.generator.subtask_scheduler import SubtaskScheduler
from eloggen.generation_runtime.generator.metadata import DatagenInfo
from eloggen.generation_runtime.generator.selection_strategy import make_selection_strategy
from eloggen.generation_runtime.generator.waypoint import WaypointSequence, WaypointTrajectory
from eloggen.generation_runtime.context.frames import build_generated_segment_contexts
from eloggen.generation_runtime.context.semantic_plan import (
    apply_phase_semantic_to_task_spec,
    attach_semantics_to_frame_contexts,
    build_semantic_plan,
)

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
from omnigibson.robots.openarm_bimanual import OpenArmBimanual

OPENARM_TASK_PREFIXES = (
    "openarm_real_exp_1",
    "openarm_drawer_storage",
    "openarm_fruit_basket_bagging",
)


class DataGenerator(object):
    """
    The main data generator object that loads a source dataset, parses it, and 
    generates new trajectories.
    """
    def __init__(
        self,
        task_spec,
        dataset_path,
        demo_keys=None,
        bimanual=False,
        D2_sign=False,
        print_stage_type=False,
        grasp_order="random",
        subtask_graph=None,
        subtask_scheduling=None,
        node_to_phase=None,
        subtask_order=None,
    ):
        """
        Args:
            task_spec (EG_TaskSpec instance): task specification that will be
                used to generate data
            dataset_path (str): path to hdf5 dataset to use for generation
            demo_keys (list of str): list of demonstration keys to use
                in file. If not provided, all demonstration keys will be
                used.
        """
        assert isinstance(task_spec, EG_TaskSpec)
        self.task_spec = task_spec
        self.dataset_path = dataset_path
        self.bimanual = bimanual
        self.D2_sign = D2_sign
        self.print_stage_type = print_stage_type
        self.grasp_order_setting = (grasp_order or "random").strip()
        self.subtask_scheduling = dict(subtask_scheduling or {})
        self.node_to_phase = dict(node_to_phase or {})
        self.subtask_order_setting = (subtask_order or "").strip()
        self.subtask_scheduler = None
        if subtask_graph:
            self.subtask_scheduler = SubtaskScheduler(
                SubtaskGraph.from_dict(subtask_graph),
                seed=int(self.subtask_scheduling.get("seed", 0)),
            )
        self.semantic_plan = build_semantic_plan(self.task_spec) if self.bimanual else {}
        self.verbose_datagen = os.environ.get("ELOGGEN_VERBOSE_DATAGEN", "0").lower() in {"1", "true", "yes"}

        if self.bimanual:
            self.num_phases = len(self.task_spec)
            # sanity check on task spec offset ranges - final subtask should not have any offset randomization
            for phase_index in range(self.num_phases):
                phase_spec = self.task_spec[phase_index]
                # for left arm
                assert phase_spec[0][-1]["subtask_term_offset_range"][0] == 0
                assert phase_spec[0][-1]["subtask_term_offset_range"][1] == 0
                # for right arm
                assert phase_spec[1][-1]["subtask_term_offset_range"][0] == 0
                assert phase_spec[1][-1]["subtask_term_offset_range"][1] == 0

        else:
            # sanity check on task spec offset ranges - final subtask should not have any offset randomization
            assert self.task_spec[-1]["subtask_term_offset_range"][0] == 0
            assert self.task_spec[-1]["subtask_term_offset_range"][1] == 0

        # demonstration keys to use from hdf5 as source dataset
        if demo_keys is None:
            # get all demonstration keys from file
            demo_keys = DatasetUtils.get_all_demos_from_dataset(dataset_path=self.dataset)
        self.demo_keys = demo_keys

        # parse source dataset
        self._load_dataset(dataset_path=dataset_path, demo_keys=demo_keys)

    def _load_dataset(self, dataset_path, demo_keys):
        """
        Load important information from a dataset into internal memory.
        """
        print("\nDataGenerator: loading dataset at path {}...".format(dataset_path))
        if self.bimanual:
            self.src_dataset_infos, self.src_subtask_indices, self.subtask_names, _, self.src_actions = DatasetUtils.parse_source_dataset_bimanual(
                dataset_path=dataset_path,
                demo_keys=demo_keys,
                task_spec=self.task_spec,
            )
        else:
            self.src_dataset_infos, self.src_subtask_indices, self.subtask_names, _ = DatasetUtils.parse_source_dataset(
                dataset_path=dataset_path,
                demo_keys=demo_keys,
                task_spec=self.task_spec,
            )
        print("\nDataGenerator: done loading\n")

    def __repr__(self):
        """
        Pretty print this object.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tdataset_path={}\n\tdemo_keys={}\n)".format(
            self.dataset_path,
            self.demo_keys,
        )
        return msg

    def randomize_subtask_boundaries(self, src_subtask_indices, task_spec):
        """
        Apply random offsets to sample subtask boundaries according to the task spec.
        Recall that each demonstration is segmented into a set of subtask segments, and the
        end index of each subtask can have a random offset.
        """
        # TODO: will need to sample the subtasks boundaries with the two arm coordination within consideration

        # initial subtask start and end indices - shape (N, S, 2)
        src_subtask_indices = np.array(src_subtask_indices)

        # for each subtask (except last one), sample all end offsets at once for each demonstration
        # add them to subtask end indices, and then set them as the start indices of next subtask too
        for i in range(src_subtask_indices.shape[1] - 1):
            end_offsets = np.random.randint(
                low=task_spec[i]["subtask_term_offset_range"][0],
                high=task_spec[i]["subtask_term_offset_range"][1] + 1,
                size=src_subtask_indices.shape[0]
            )
            src_subtask_indices[:, i, 1] = src_subtask_indices[:, i, 1] + end_offsets
            # don't forget to set these as start indices for next subtask too
            src_subtask_indices[:, i + 1, 0] = src_subtask_indices[:, i, 1]

        # ensure non-empty subtasks
        assert np.all((src_subtask_indices[:, :, 1] - src_subtask_indices[:, :, 0]) > 0), "got empty subtasks!"

        # ensure subtask indices increase (both starts and ends)
        assert np.all((src_subtask_indices[:, 1:, :] - src_subtask_indices[:, :-1, :]) > 0), "subtask indices do not strictly increase"

        # ensure subtasks are in order
        subtask_inds_flat = src_subtask_indices.reshape(src_subtask_indices.shape[0], -1)
        assert np.all((subtask_inds_flat[:, 1:] - subtask_inds_flat[:, :-1]) >= 0), "subtask indices not in order"

        return src_subtask_indices

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        subtask_ind,
        src_subtask_inds,
        subtask_object_name,
        selection_strategy_name,
        selection_strategy_kwargs=None,
    ):
        """
        Helper method to run source subtask segment selection.

        Args:
            eef_pose (np.array): current end effector pose
            object_pose (np.array): current object pose for this subtask
            subtask_ind (int): index of subtask
            src_subtask_inds (np.array): start and end indices for subtask segment in source demonstrations of shape (N, 2)
            subtask_object_name (str): name of reference object for this subtask
            selection_strategy_name (str): name of selection strategy
            selection_strategy_kwargs (dict): extra kwargs for running selection strategy

        Returns:
            selected_src_demo_ind (int): selected source demo index
        """
        if subtask_object_name is None:
            # no reference object - only random selection is supported
            assert selection_strategy_name == "random"

        # We need to collect the datagen info objects over the timesteps for the subtask segment in each source 
        # demo, so that it can be used by the selection strategy.
        src_subtask_datagen_infos = []
        for i in range(len(self.demo_keys)):
            # datagen info over all timesteps of the src trajectory
            src_ep_datagen_info = self.src_dataset_infos[i]

            # time indices for subtask
            subtask_start_ind = src_subtask_inds[i][0]
            subtask_end_ind = src_subtask_inds[i][1]

            # get subtask segment using indices
            src_subtask_datagen_infos.append(DatagenInfo(
                eef_pose=src_ep_datagen_info.eef_pose[subtask_start_ind : subtask_end_ind],
                # only include object pose for relevant object in subtask
                object_poses={ subtask_object_name : src_ep_datagen_info.object_poses[subtask_object_name][subtask_start_ind : subtask_end_ind] } if (subtask_object_name is not None) else None,
                # subtask termination signal is unused
                subtask_term_signals=None,
                target_pose=src_ep_datagen_info.target_pose[subtask_start_ind : subtask_end_ind],
                gripper_action=src_ep_datagen_info.gripper_action[subtask_start_ind : subtask_end_ind],
            ))

        # make selection strategy object
        selection_strategy_obj = make_selection_strategy(selection_strategy_name)

        # run selection
        if selection_strategy_kwargs is None:
            selection_strategy_kwargs = dict()
        selected_src_demo_ind = selection_strategy_obj.select_source_demo(
            eef_pose=eef_pose,
            object_pose=object_pose,
            src_subtask_datagen_infos=src_subtask_datagen_infos,
            **selection_strategy_kwargs,
        )

        return selected_src_demo_ind

    def merge_trajs(self, traj_list_all):
        # merge the waypoints for each arm
        # print('#################### in merge trajectories ####################')
        
        waypoint_traj_list = []
        for i in range(2):
            traj_list = traj_list_all[i]
            waypoint_traj = WaypointTrajectory()
            for traj in traj_list:
                for seq in traj.waypoint_sequences:
                    if waypoint_traj.waypoint_sequences == []:
                        waypoint_traj.add_waypoint_sequence(seq)
                    else:
                        waypoint_traj.waypoint_sequences[-1].sequence += seq.sequence
                    # print('num waypoints:', len(waypoint_traj.waypoint_sequences[-1].sequence))
            waypoint_traj_list.append(waypoint_traj)
        
        
        # merge the left and right eef pose
        traj_left = waypoint_traj_list[0]
        traj_right = waypoint_traj_list[1]
        min_length = min(len(traj_left.waypoint_sequences[0].sequence), len(traj_right.waypoint_sequences[0].sequence))
        max_length = max(len(traj_left.waypoint_sequences[0].sequence), len(traj_right.waypoint_sequences[0].sequence))
        if max_length > min_length:
            if len(traj_left.waypoint_sequences[0].sequence) == min_length:
                for _ in range(max_length - min_length):
                    traj_left.waypoint_sequences[0].sequence.append(traj_left.waypoint_sequences[0].sequence[-1])
            else:
                for _ in range(max_length - min_length):
                    traj_right.waypoint_sequences[0].sequence.append(traj_right.waypoint_sequences[0].sequence[-1])
        for i in range(max_length):
            traj_left.waypoint_sequences[0].sequence[i].merge_wp(traj_right.waypoint_sequences[0].sequence[i])
        traj_to_execute = traj_left

        return traj_to_execute

    def change_arm_role_heuristic(self,
                                  env_interface,
                                  start_step,
                                  selected_src_demo_ind,
                                  cur_phase_task_spec
                                  ):
        change_role = False

        src_left_arm_start_pos = self.src_dataset_infos[selected_src_demo_ind].eef_pose[start_step:start_step+1][:,:4,:] # shape (1, 4, 4)
        src_right_arm_start_pose = self.src_dataset_infos[selected_src_demo_ind].eef_pose[start_step:start_step+1][:,4:,:] # shape (1, 4, 4)

        left_arm_object_name = cur_phase_task_spec[0][0]["object_ref"]
        right_arm_object_name = cur_phase_task_spec[1][0]["object_ref"]
        src_left_arm_object_pose = self.src_dataset_infos[selected_src_demo_ind].object_poses[left_arm_object_name][start_step]
        src_right_arm_object_pose = self.src_dataset_infos[selected_src_demo_ind].object_poses[right_arm_object_name][start_step]
        cur_left_arm_object_pose = env_interface.get_datagen_info().object_poses[left_arm_object_name] # shape (4, 4)
        cur_right_arm_object_pose = env_interface.get_datagen_info().object_poses[right_arm_object_name] # shape (4, 4)

        transformed_eef_poses_left_arm_object = PoseUtils.transform_source_data_segment_using_object_pose(
            obj_pose=cur_left_arm_object_pose, 
            src_eef_poses=src_left_arm_start_pos,
            src_obj_pose=src_left_arm_object_pose) # shape (1, 4, 4)
        transformed_eef_poses_right_arm_object = PoseUtils.transform_source_data_segment_using_object_pose(
            obj_pose=cur_right_arm_object_pose, 
            src_eef_poses=src_right_arm_start_pose,
            src_obj_pose=src_right_arm_object_pose) # shape (1, 4, 4)

        cur_left_arm_pose = env_interface.get_datagen_info().eef_pose[None][:,:4,:] # shape (1, 4, 4)
        cur_right_arm_pose = env_interface.get_datagen_info().eef_pose[None][:,4:,:] # shape (1, 4, 4)

        distance_left_arm_to_traj_left_arm_object = np.linalg.norm(cur_left_arm_pose[:,:,-1] - transformed_eef_poses_left_arm_object[:,:,-1])
        distance_right_arm_to_traj_left_arm_object = np.linalg.norm(cur_right_arm_pose[:,:,-1] - transformed_eef_poses_left_arm_object[:,:,-1])

        distance_left_arm_to_traj_right_arm_object = np.linalg.norm(cur_left_arm_pose[:,:,-1] - transformed_eef_poses_right_arm_object[:,:,-1])
        distance_right_arm_to_traj_right_arm_object = np.linalg.norm(cur_right_arm_pose[:,:,-1] - transformed_eef_poses_right_arm_object[:,:,-1])

        print('========================================== new phase ==========================================')
        print('distance_left_arm_to_traj_left_arm_object', distance_left_arm_to_traj_left_arm_object)
        print('distance_right_arm_to_traj_left_arm_object', distance_right_arm_to_traj_left_arm_object)
        print('distance_left_arm_to_traj_right_arm_object', distance_left_arm_to_traj_right_arm_object)
        print('distance_right_arm_to_traj_right_arm_object', distance_right_arm_to_traj_right_arm_object)

        # compare the distances 
        if distance_left_arm_to_traj_left_arm_object < distance_right_arm_to_traj_left_arm_object and distance_right_arm_to_traj_right_arm_object < distance_right_arm_to_traj_left_arm_object:
            change_role = False
            print('no change role')
        elif distance_left_arm_to_traj_left_arm_object > distance_right_arm_to_traj_left_arm_object and distance_right_arm_to_traj_right_arm_object > distance_right_arm_to_traj_left_arm_object:
            change_role = True
            print('change role')
        else:
            # TODO: if the change arm role constaints are not satisfied, will keep the original arm role
            print('distance comparison heuristic is not applicable, check corner cases')
            change_role = False
            # raise ValueError('The distance comparison heuristic is not applicable, check corner cases')

        return change_role

    def parse_MP_end_step_local(self):
        """
        parse the MP_end_step from the configuration file and get the local information
        """
        # example output
        # [
        #   [
        #       [160, -1], 
        #       [110, 0]
        #   ], 
        #   [
        #       [180], 
        #       [-1]
        #   ]
        # ]
        end_step_of_MP = []
        for phase_ind in range(self.num_phases):
            end_step_of_MP.append([])
            for arm_ind in range(2): # left and right arms
                num_subtasks_cur_phase = len(self.task_spec[phase_ind][arm_ind])
                end_step_of_MP[-1].append([])
                for i in range(num_subtasks_cur_phase):
                    if self.task_spec[phase_ind][arm_ind][i]["MP_end_step"] is not None:
                        end_step = self.task_spec[phase_ind][arm_ind][i]["MP_end_step"]
                    elif self.task_spec[phase_ind][arm_ind][i]['subtask_term_step'] is not None:
                        end_step = self.task_spec[phase_ind][arm_ind][i]['subtask_term_step']
                    else:
                        # We only have one demo right now, so we can use the length of the demo as the end step
                        end_step = self.src_dataset_infos[0].eef_pose.shape[0]

                    end_step_of_MP[-1][-1].append(end_step)
        return end_step_of_MP

    def parse_annotations(self, annotations):
        annotations = None
        return annotations

    def obtain_attached_object(self, env, robot, attached_obj_new={}, attached_obj_scale={}):
        attached_object_names = {}
        for local_arm_side in ["left", "right"]:  
            eef_link_name = robot.eef_link_names.get(local_arm_side, f"{local_arm_side}_eef_link")
            is_grasping = robot.is_grasping(arm=local_arm_side)
            if is_grasping == og.controllers.IsGraspingState.TRUE: 
                # Find the object that the robot is grapsing in that arm
                task_relevant_objs = env._get_task_relevant_objs()
                for task_relevant_obj in task_relevant_objs:
                    # TODO: remove the stationay object hardcoding. Make it more general
                    if all(keyword not in task_relevant_obj.name for keyword in ["table", "shelf", "bar", "sink"]):
                        is_grasping_candidate_obj = robot.is_grasping(arm=local_arm_side, candidate_obj=task_relevant_obj)
                        if is_grasping_candidate_obj == og.controllers.IsGraspingState.TRUE:
                            print(f"arm {local_arm_side} is_grasping {task_relevant_obj.root_link.name}") 
                            attached_obj_new[eef_link_name] = task_relevant_obj.root_link
                            attached_obj_scale[eef_link_name] = 0.9
                            attached_object_names[local_arm_side] = task_relevant_obj.name
                            # robot can only be holding one object at a time
                            break
        return attached_object_names
    
    def visualize_traj(self, env, src_eef_poses, transformed_eef_poses):
        while True:
            for i in range(len(src_eef_poses)):
                src_eef_pose = T.mat2pose(th.tensor(src_eef_poses[i]))
                transformed_eef_pose = T.mat2pose(th.tensor(transformed_eef_poses[i]))
                env.eef_goal_marker_left.set_position_orientation(*src_eef_pose)
                env.eef_goal_marker_right.set_position_orientation(*transformed_eef_pose)
                for _ in range(5): og.sim.step()
            inp = input("Press r to replay or anything else to continue")
            if inp == 'r':
                continue
            else:
                break

    def _subtask_spec_for_context(self, cur_phase_task_spec, object_ref, subtask_ind_reordered):
        active_arm = "left" if object_ref.get("arm_left") is not None else "right"
        arm_index = 0 if active_arm == "left" else 1
        local_specs = cur_phase_task_spec[arm_index]
        if len(local_specs) == 0:
            return {"arm": active_arm, "object_ref": object_ref.get(f"arm_{active_arm}")}
        local_index = min(max(int(subtask_ind_reordered), 0), len(local_specs) - 1)
        spec = dict(local_specs[local_index])
        spec["arm"] = active_arm
        spec["object_ref"] = object_ref.get(f"arm_{active_arm}")
        return spec

    def _build_exec_frame_contexts(
        self,
        env,
        exec_results,
        object_ref,
        cur_phase_task_spec,
        current_phase_ind,
        subtask_ind_reordered,
        selected_src_demo_ind,
        generated_start_frame,
        phase_type,
        execution_metadata=None,
        execution_order_index=None,
    ):
        if exec_results is None or len(exec_results.get("states", [])) == 0:
            return []
        task_name = getattr(env, "name", None) or getattr(getattr(env, "env", None), "name", "unknown_task")
        if phase_type == "navigation":
            frame_contexts = [
                {
                    "task": task_name,
                    "action": "navigation",
                    "stage_type": "navigation",
                    "substage_id": 0,
                    "segment": "navigation",
                    "generated_frame": generated_start_frame + frame_index,
                    "segment_frame": frame_index,
                    "phase_index": current_phase_ind,
                    "subtask_index": subtask_ind_reordered,
                    "active_arm": None,
                    "active_object": None,
                    "passive_object": self._select_reference_object(object_ref),
                    "object_label_zh": None,
                    "object_label_en": None,
                    "target_object": None,
                    "target_label_zh": None,
                    "target_label_en": None,
                    "instruction_zh": "移动到可操作位置",
                    "instruction_en": "Navigate to a reachable manipulation pose",
                    "source_demo_ind": selected_src_demo_ind,
                    "boundary_source": "generated_navigation_segment",
                    "metadata": {"phase_type": phase_type},
                }
                for frame_index in range(exec_results["actions"].shape[0])
            ]
            return self._attach_execution_metadata_to_frame_contexts(
                frame_contexts,
                execution_metadata=execution_metadata,
                execution_order_index=execution_order_index,
            )
        task_spec = self._subtask_spec_for_context(cur_phase_task_spec, object_ref, subtask_ind_reordered)
        phase_semantic = self.semantic_plan.get((current_phase_ind, subtask_ind_reordered))
        task_spec = apply_phase_semantic_to_task_spec(task_spec, phase_semantic)
        frame_contexts = build_generated_segment_contexts(
            task_name=task_name,
            datagen_infos=exec_results["datagen_infos"],
            actions=exec_results["actions"],
            object_ref=object_ref,
            mp_end_steps=exec_results.get("mp_end_steps"),
            subtask_length=exec_results.get("subtask_lengths"),
            phase_index=current_phase_ind,
            subtask_index=subtask_ind_reordered,
            task_spec=task_spec,
            source_demo_ind=selected_src_demo_ind,
            generated_start_frame=generated_start_frame,
        )
        frame_contexts = attach_semantics_to_frame_contexts(frame_contexts, phase_semantic)
        return self._attach_execution_metadata_to_frame_contexts(
            frame_contexts,
            execution_metadata=execution_metadata,
            execution_order_index=execution_order_index,
        )

    def _log_stage_contexts(self, frame_contexts):
        if not self.verbose_datagen or not self.print_stage_type or not frame_contexts:
            return

        first = frame_contexts[0]
        current_stage = first["stage_type"]
        current_action = first["action"]
        start_frame = first["generated_frame"]
        phase_index = first.get("phase_index")
        subtask_index = first.get("subtask_index")

        for context in frame_contexts[1:]:
            if context["stage_type"] == current_stage and context["action"] == current_action:
                continue
            end_frame = context["generated_frame"] - 1
            print(
                "[stage_type] phase={} subtask={} action={} stage_type={} frames={}..{}".format(
                    phase_index,
                    subtask_index,
                    current_action,
                    current_stage,
                    start_frame,
                    end_frame,
                )
            )
            current_stage = context["stage_type"]
            current_action = context["action"]
            start_frame = context["generated_frame"]

        end_frame = frame_contexts[-1]["generated_frame"]
        print(
            "[stage_type] phase={} subtask={} action={} stage_type={} frames={}..{}".format(
                phase_index,
                subtask_index,
                current_action,
                current_stage,
                start_frame,
                end_frame,
            )
        )

    def _resolve_env_object(self, env, env_interface, object_name):
        if object_name is None:
            return None

        if env_interface is not None and hasattr(env_interface, "_get_object_by_name"):
            obj = env_interface._get_object_by_name(object_name)
            if obj is not None:
                return obj

        task = getattr(env, "task", None)
        object_scope = getattr(task, "object_scope", None)
        if object_scope is not None and object_name in object_scope:
            return object_scope[object_name]

        scene = getattr(env, "scene", None)
        if scene is not None:
            return scene.object_registry("name", object_name)

        nested_env = getattr(env, "env", None)
        nested_scene = getattr(nested_env, "scene", None)
        if nested_scene is not None:
            return nested_scene.object_registry("name", object_name)

        return None

    def _select_reference_object(self, object_ref):
        left_ref = object_ref.get("arm_left")
        right_ref = object_ref.get("arm_right")

        if right_ref is None:
            return left_ref
        if left_ref is None:
            return right_ref
        return right_ref

    @contextlib.contextmanager
    def _quiet_generation_stdout(self):
        if self.verbose_datagen:
            yield
            return
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            yield

    def _is_openarm_task_env(self, env):
        env_name = getattr(env, "name", None) or getattr(getattr(env, "env", None), "name", "")
        return self.bimanual and str(env_name).startswith(OPENARM_TASK_PREFIXES)

    def _is_openarm_real_exp_env(self, env):
        env_name = getattr(env, "name", None) or getattr(getattr(env, "env", None), "name", "")
        return str(env_name).startswith("openarm_real_exp_1")

    def _summarize_openarm_success_metrics(self, env, metrics):
        env_name = getattr(env, "name", None) or getattr(getattr(env, "env", None), "name", "")
        if str(env_name).startswith("openarm_real_exp_1"):
            keys = [
                "task",
                "bddl",
            ]
        elif str(env_name).startswith("openarm_drawer_storage"):
            keys = ["task", "bddl", "drawer_open"] + sorted(
                key for key in metrics if key.endswith("_inside")
            )
        else:
            keys = ["task", "bddl"]

        summary = {key: metrics[key] for key in keys if key in metrics}
        return summary

    def _resolve_grasp_order(self, object_phase_pairs):
        setting = (self.grasp_order_setting or "random").strip()
        normalized = setting.lower().replace("-", "_")
        object_names = [name for name, _ in object_phase_pairs]

        if normalized in {"random", "rand"}:
            permutation = np.random.permutation(len(object_phase_pairs))
            strategy = "random_grasp_order"
        elif normalized in {"task_spec", "source", "fixed", "default"}:
            permutation = np.arange(len(object_phase_pairs))
            strategy = "task_spec_grasp_order"
        else:
            aliases = {
                "can": "canned_food_1",
                "canned": "canned_food_1",
                "canned_food": "canned_food_1",
                "candy": "candy_1",
                "orange": "orange_1",
                "apple": "apple_1",
                "obj1": "object_1",
                "object1": "object_1",
                "obj_1": "object_1",
                "object_1": "object_1",
                "obj2": "object_2",
                "object2": "object_2",
                "obj_2": "object_2",
                "object_2": "object_2",
                "lime": "object_2",
                "lemon": "object_2",
            }
            if "object_1" in object_names:
                aliases["apple"] = "object_1"
            if "object_2" in object_names:
                aliases["lime"] = "object_2"
                aliases["lemon"] = "object_2"
            requested = []
            for raw_name in setting.replace(";", ",").replace(" ", ",").split(","):
                raw_name = raw_name.strip()
                if not raw_name:
                    continue
                requested.append(aliases.get(raw_name.lower(), raw_name))

            if len(requested) != len(object_names) or set(requested) != set(object_names):
                raise ValueError(
                    "Invalid --grasp-order '{}'. Expected 'random', 'task_spec', or a permutation of: {}".format(
                        setting, ",".join(object_names)
                    )
                )
            permutation = np.array([object_names.index(name) for name in requested], dtype=int)
            strategy = "manual_grasp_order"

        grasp_order = [object_phase_pairs[idx][0] for idx in permutation]
        phase_order = [
            phase_ind
            for idx in permutation
            for phase_ind in object_phase_pairs[idx][1]
        ]
        return grasp_order, phase_order, strategy

    @staticmethod
    def _get_phase_retract_type(cur_phase_task_spec):
        for arm_task_spec in cur_phase_task_spec:
            for subtask_spec in arm_task_spec:
                if subtask_spec.get("object_ref") is not None or subtask_spec.get("attached_obj") is not None:
                    return subtask_spec.get("retract_type")
        return cur_phase_task_spec[0][0].get("retract_type")

    def _sample_phase_execution_metadata(self, env):
        env_name = getattr(env, "name", None) or getattr(getattr(env, "env", None), "name", "")
        phase_order = list(range(self.num_phases))
        metadata = {
            "phase_execution_order_indices": phase_order,
            "phase_execution_order": [f"phase_{phase_ind + 1}" for phase_ind in phase_order],
            "execution_order_strategy": "task_spec_order",
            "record_execution_metadata": False,
        }
        initialization_sample = (
            getattr(env, "drawer_initialization_sample", None)
            or getattr(env, "fruit_initialization_sample", None)
        )
        if initialization_sample is not None:
            metadata["initialization_sample"] = deepcopy(initialization_sample)

        if not self._is_openarm_task_env(env):
            return metadata

        metadata["record_execution_metadata"] = True
        if self._is_openarm_real_exp_env(env) and self.num_phases == 4:
            object_phase_pairs = [
                ("object_2", [0, 1]),
                ("object_1", [2, 3]),
            ]
            grasp_order, phase_order, order_strategy = self._resolve_grasp_order(object_phase_pairs)
        elif str(env_name).startswith("openarm_drawer_storage") or (
            str(env_name).startswith("openarm_fruit_basket_bagging")
            and self.subtask_scheduler is not None
        ):
            if self.subtask_scheduler is None:
                raise ValueError(f"{env_name} requires a configured subtask_graph")
            initialization_attempt = int(
                (initialization_sample or {}).get(
                    "attempt_index",
                    getattr(env, "generation_attempt_index", 0),
                )
            )
            configured_strategy = str(self.subtask_scheduling.get("strategy", "random_topological"))
            fixed_order = self.subtask_scheduling.get("fixed_order") or None
            if self.subtask_order_setting:
                if self.subtask_order_setting.lower() in {"random", "random_topological"}:
                    configured_strategy = "random_topological"
                    fixed_order = None
                else:
                    configured_strategy = "fixed"
                    fixed_order = [
                        node.strip()
                        for node in self.subtask_order_setting.split(",")
                        if node.strip()
                    ]
            node_order = self.subtask_scheduler.select_for_attempt(
                configured_strategy,
                initialization_attempt,
                fixed_order=fixed_order,
            )
            phase_order = [self.node_to_phase[node] for node in node_order]
            grasp_order = [
                event[2]
                for node in node_order
                for event in self.subtask_scheduler.graph.resource_events.get(node, ())
                if event[1] == "acquire"
            ]
            order_strategy = configured_strategy
            metadata.update(
                {
                    "phase_execution_order_indices": phase_order,
                    "phase_execution_order": list(node_order),
                    "subtask_order": list(node_order),
                    "subtask_order_name": self.subtask_scheduler.order_name(node_order),
                    "open_before_pick": not any(
                        event[1] == "acquire"
                        for event in self.subtask_scheduler.graph.resource_events.get(node_order[0], ())
                    ),
                    "grasp_order": grasp_order,
                    "execution_order_strategy": f"subtask_graph_{order_strategy}",
                    "record_execution_metadata": True,
                }
            )
            return metadata
        else:
            print(
                "[OpenArm][WARN] unrecognized phase layout ({}); using task-spec order".format(self.num_phases)
            )
            return metadata
            
        metadata.update(
            {
                "phase_execution_order_indices": phase_order,
                "phase_execution_order": [f"phase_{phase_ind + 1}" for phase_ind in phase_order],
                "grasp_order": grasp_order,
                "execution_order_strategy": "openarm_{}".format(order_strategy),
                "record_execution_metadata": True,
            }
        )
        return metadata

    def _attach_execution_metadata_to_frame_contexts(
        self,
        frame_contexts,
        execution_metadata=None,
        execution_order_index=None,
    ):
        if not execution_metadata or not execution_metadata.get("record_execution_metadata", False):
            return frame_contexts

        grasp_order = execution_metadata.get("grasp_order")
        phase_execution_order = execution_metadata.get("phase_execution_order")
        phase_execution_order_indices = execution_metadata.get("phase_execution_order_indices")
        execution_order_strategy = execution_metadata.get("execution_order_strategy")
        initialization_sample = execution_metadata.get("initialization_sample")
        subtask_order = execution_metadata.get("subtask_order")
        subtask_order_name = execution_metadata.get("subtask_order_name")
        held_objects_before_phase = execution_metadata.get("held_objects_before_phase") or {}
        right_gripper_resource_state = execution_metadata.get("right_gripper_resource_state")
        open_before_pick = execution_metadata.get("open_before_pick")

        for context in frame_contexts:
            context["execution_order_index"] = execution_order_index
            context["execution_order_strategy"] = execution_order_strategy
            context["phase_execution_order"] = phase_execution_order
            context["phase_execution_order_indices"] = phase_execution_order_indices
            if grasp_order is not None:
                context["grasp_order"] = grasp_order
            if subtask_order is not None:
                context["subtask_order"] = subtask_order
                context["subtask_order_name"] = subtask_order_name
                context["held_objects"] = held_objects_before_phase
                context["held_object"] = right_gripper_resource_state
                context["right_gripper_resource_state"] = right_gripper_resource_state
                context["open_before_pick"] = open_before_pick
            if initialization_sample is not None:
                context["initialization_sample"] = initialization_sample

            context_metadata = dict(context.get("metadata") or {})
            context_metadata.update(
                {
                    "execution_order_index": execution_order_index,
                    "execution_order_strategy": execution_order_strategy,
                    "phase_execution_order": phase_execution_order,
                    "phase_execution_order_indices": phase_execution_order_indices,
                }
            )
            if grasp_order is not None:
                context_metadata["grasp_order"] = grasp_order
            if subtask_order is not None:
                context_metadata["subtask_order"] = subtask_order
                context_metadata["subtask_order_name"] = subtask_order_name
                context_metadata["held_objects"] = held_objects_before_phase
                context_metadata["right_gripper_resource_state"] = right_gripper_resource_state
                context_metadata["open_before_pick"] = open_before_pick
            if initialization_sample is not None:
                context_metadata["initialization_sample"] = initialization_sample
            context["metadata"] = context_metadata

        return frame_contexts
    
    def generate(
        self,
        env,
        env_interface,
        select_src_per_subtask=False,
        transform_first_robot_pose=False,
        interpolate_from_last_target_pose=True,
        render=False,
        video_writer=None,
        video_skip=5,
        camera_names=None,
        pause_subtask=False,
        enable_marker_vis=False,
        ds_ratio=1,
        grasp_init_views_video_writer=None,
        no_partial_tasks=False,
        baseline=None,
    ):
        """
        Attempt to generate a new demonstration.

        Args:
            env (robomimic EnvBase instance): environment to use for data collection
            
            env_interface (EG_EnvInterface instance): environment interface for some data generation operations

            select_src_per_subtask (bool): if True, select a different source demonstration for each subtask 
                during data generation, else keep the same one for the entire episode

            transform_first_robot_pose (bool): if True, each subtask segment will consist of the first
                robot pose and the target poses instead of just the target poses. Can sometimes help
                improve data generation quality as the interpolation segment will interpolate to where 
                the robot started in the source segment instead of the first target pose. Note that the
                first subtask segment of each episode will always include the first robot pose, regardless
                of this argument.
                TODO: not sure about the meaning of this property

            interpolate_from_last_target_pose (bool): if True, each interpolation segment will start from
                the last target pose in the previous subtask segment, instead of the current robot pose. Can
                sometimes improve data generation quality.

            render (bool): if True, render on-screen

            video_writer (imageio writer): video writer

            video_skip (int): determines rate at which environment frames are written to video

            camera_names (list): determines which camera(s) are used for rendering. Pass more than
                one to output a video with multiple camera views concatenated horizontally.

            pause_subtask (bool): if True, pause after every subtask during generation, for
                debugging.

        Returns:
            results (dict): dictionary with the following items:
                initial_state (dict): initial simulator state for the executed trajectory
                states (list): simulator state at each timestep
                observations (list): observation dictionary at each timestep
                datagen_infos (list): datagen_info at each timestep
                actions (np.array): action executed at each timestep
                success (bool): whether the trajectory successfully solved the task or not
                src_demo_inds (list): list of selected source demonstration indices for each subtask
                src_demo_labels (np.array): same as @src_demo_inds, but repeated to have a label for each timestep of the trajectory
        """

        # sample new task instance
        # env.customize_physical_properties() # change physical properties of the objects and robot for each task
        env.reset()
        if isinstance(getattr(env, "robot", None), OpenArmBimanual):
            env.openarm_generation_initial_joint_positions = (
                env.robot.get_joint_positions().detach().clone().cpu()
            )
        new_initial_state = env.get_state()
        
        sensor_info = env.sensor_setup()
        for _ in range(5): og.sim.render()
        
        # parse MP_end_step from the configuration file
        end_step_of_MP_local = self.parse_MP_end_step_local()

        # sample new subtask boundaries
        all_subtask_inds_structure = []
        for phase_index in range(self.num_phases):
            all_subtask_inds_structure.append([])
            for arm_i in range(2): # arm_left, arm_right
                all_subtask_inds_arm = self.randomize_subtask_boundaries(self.src_subtask_indices[phase_index][arm_i], self.task_spec[phase_index][arm_i]) # shape (1,2,2)
                all_subtask_inds_structure[-1].append(all_subtask_inds_arm)

        # all_subtask_inds_structure is a list of length @num_phases
        # all_subtask_inds_structure[0] is a list of length 2, corresponding to left and right arms
        # all_subtask_inds_structure[0][0] is a numpy array of shape (@num_demos, @num_subtasks, 2)
        # where @num_demos is 1 right now, @num_subtasks can vary, 2 means start and end indices

        #(Pdb) all_subtask_inds_structure
        #[[array([[[  0, 730]]]), array([[[  0, 730]]])], [array([[[ 730, 1210]]]), array([[[ 730, 1210]]])]]

        # some state variables used during generation
        selected_src_demo_ind = None
        prev_executed_traj = None

        # save generated data in these variables
        generated_states = []
        generated_obs = []
        generated_obs_info = []
        generated_datagen_infos = []
        generated_actions = []
        generated_demo_mp_end_steps = []
        generated_demo_subtask_lengths = []
        generated_success = False
        generated_src_demo_inds = [] # store selected src demo ind for each subtask in each trajectory
        generated_src_demo_labels = [] # like @generated_src_demo_inds, but padded to align with size of @generated_actions
        generated_demo_left_mp_ranges = []
        generated_demo_right_mp_ranges = []
        generated_frame_contexts = []
        phase_logs = dict()
        execution_metadata = self._sample_phase_execution_metadata(env)
        episode_metadata = execution_metadata if execution_metadata.get("record_execution_metadata", False) else None
        phase_execution_order = execution_metadata["phase_execution_order_indices"]
        if execution_metadata.get("record_execution_metadata", False):
            print(
                "[OpenArm] grasp_order: {} ({})".format(
                    execution_metadata.get("grasp_order"),
                    execution_metadata.get("execution_order_strategy"),
                )
            )

        # Track semantic transitions in execution order so task-specific MP
        # policies remain valid when phases are reordered.
        last_phase_action = "reset"
        held_resources = {}
        subtask_order = execution_metadata.get("subtask_order") or []

        # for left arms first
        for execution_order_index, current_phase_ind in enumerate(phase_execution_order):
            # This is probably not being used anymore. Confirm and remove if not.
            if not env.valid_env:
                break 
            
            # # remove later
            # if current_phase_ind < 2:
            #     continue
                        
            phase_type = self.task_spec[current_phase_ind][0][0]["phase_type"]            
            cur_phase_task_spec = self.task_spec[current_phase_ind]
            current_node = (
                subtask_order[execution_order_index]
                if execution_order_index < len(subtask_order)
                else None
            )
            held_by_arm = {
                resource.removesuffix("_gripper"): token
                for resource, token in held_resources.items()
                if resource.endswith("_gripper")
            }
            frozen_arms = tuple(
                arm for arm in held_by_arm if all(
                    spec.get("arm") != arm
                    for arm_specs in cur_phase_task_spec
                    for spec in arm_specs
                    if spec.get("object_ref") is not None or spec.get("attached_obj") is not None
                )
            )
            execution_metadata["held_objects_before_phase"] = dict(held_by_arm)
            execution_metadata["right_gripper_resource_state"] = held_by_arm.get("right")
            if current_node is not None:
                execution_metadata.setdefault("phase_resource_states", {})[current_node] = {
                    "held_objects_before": dict(held_by_arm),
                    "right_gripper": held_by_arm.get("right"),
                }
            active_phase_specs = [
                spec
                for arm_specs in cur_phase_task_spec
                for spec in arm_specs
                if spec.get("object_ref") is not None or spec.get("attached_obj") is not None
            ]
            env.generation_previous_phase_action = last_phase_action
            env.generation_phase_action = (
                active_phase_specs[0].get("action") if active_phase_specs else "reset"
            )
            last_phase_action = env.generation_phase_action
            selected_src_demo_ind = 0 # TODO: since we only have one demo, will need to modify if more demos are available
            
            
            # Obtain the retract type from the active subtask in this phase.
            retract_type = self._get_phase_retract_type(cur_phase_task_spec)

            # restructure subtasks indexes and reference objects
            all_subtask_inds = all_subtask_inds_structure[current_phase_ind]
            subtask_ind_vals = np.sort(np.unique(np.concatenate((np.unique(all_subtask_inds[0]), np.unique(all_subtask_inds[1])))))
            num_subtasks = len(subtask_ind_vals) - 1
                        
            # ==================================== Arm role change heuristic ====================================
            change_role = False
            # # a distance based heuristic to change the role of the two arms
            # # calculate the start of the replay part
            # # currently assume that the start point is the first subtask of the current phase
            # # TODO: need to change this to other starting point when the motion planner is integrated
            # start_step = subtask_ind_vals[0]
            
            # # Uncomment later. 
            # # change_role = self.change_arm_role_heuristic(
            # #     env_interface,
            # #     start_step,
            # #     selected_src_demo_ind,
            # #     cur_phase_task_spec
            # #     )
            # change_role = False

            # if change_role:
            #     # change the information for two arms
            #     cur_phase_task_spec_new = []
            #     cur_phase_task_spec_new.append(cur_phase_task_spec[1])
            #     cur_phase_task_spec_new.append(cur_phase_task_spec[0])
            #     cur_phase_task_spec = cur_phase_task_spec_new
            #     all_subtask_inds_new = []
            #     all_subtask_inds_new.append(all_subtask_inds[1])
            #     all_subtask_inds_new.append(all_subtask_inds[0])
            #     all_subtask_inds = all_subtask_inds_new
            # ====================================================================================================

            for subtask_ind_reordered in range(num_subtasks):
                print("========== Phase {} Subtask {} ==========".format(current_phase_ind, subtask_ind_reordered))

                # # remove later
                # if current_phase_ind == 1 and subtask_ind_reordered == 1:
                #     break

                # Reset the ref object visibility stats as that is calculated for each phase/subtask
                for sensor_name, sensor in env.robot.sensors.items():
                    if isinstance(sensor, og.sensors.vision_sensor.VisionSensor):
                        env.num_frames_with_obj_visible[sensor_name.split(":")[1]] = 0
                env.num_frames_with_obj_visible["any"] = 0

                selected_src_subtask_inds = subtask_ind_vals[subtask_ind_reordered : subtask_ind_reordered + 2] # [start_step, end_step]
                traj_list_all = [[],[]]
                attached_obj_dict = {}
                object_ref = {}
                MP_end_steps = []

                for arm_i, arm_name in enumerate(['arm_left', 'arm_right']):

                    # need to recalculate the matched subtask_ind to retrieve the correct task spec
                    local_task_spec = cur_phase_task_spec[arm_i]
                    arm_spec_subtask_inds = all_subtask_inds[arm_i][0]
                    arm_unique_subtask_inds = np.sort(np.unique(arm_spec_subtask_inds))
                    subtask_ind = np.where(selected_src_subtask_inds[1] <= arm_unique_subtask_inds)[0][0] - 1

                    # print('==========================================')
                    # print('arm_name:', arm_name, 'subtask_ind_reordered', subtask_ind_reordered, 'subtask_ind:', subtask_ind)
                    # print('subtask start and end step', selected_src_subtask_inds)
                    # print('arm_spec_subtask_inds', arm_spec_subtask_inds)

                    is_first_subtask = (subtask_ind == 0) and (current_phase_ind == 0)
                    is_first_subtask_in_phase = (subtask_ind == 0)

                    cur_datagen_info = env_interface.get_datagen_info()
                    subtask_object_name = cur_phase_task_spec[arm_i][subtask_ind]["object_ref"]
                    object_ref[arm_name] = subtask_object_name
                    cur_object_pose = cur_datagen_info.object_poses[subtask_object_name] if (subtask_object_name is not None) else None # 4x4
                    key_name = arm_name.replace('arm_', '')
                    attached_obj_dict[key_name] = cur_phase_task_spec[arm_i][subtask_ind]["attached_obj"]
                    if key_name in held_by_arm:
                        attached_obj_dict[key_name] = held_by_arm[key_name]
                    MP_end_steps.append(end_step_of_MP_local[current_phase_ind][arm_i][subtask_ind])
                    
                    # get poses
                    src_ep_datagen_info = self.src_dataset_infos[selected_src_demo_ind]
                    src_subtask_eef_poses = src_ep_datagen_info.eef_pose[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]] # 106 x 8 x 4
                    # src_subtask_target_poses = src_ep_datagen_info.target_pose[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]] # 106 x 8 x 4
                    src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]] # 106 x 2

                    if (arm_name == 'arm_left' and not change_role) or (arm_name == 'arm_right' and change_role):
                        # print('select left arm demo pose')
                        src_subtask_eef_poses = src_subtask_eef_poses[:,:4,:]
                        # src_subtask_target_poses = src_subtask_target_poses[:,:4,:]
                        src_subtask_gripper_actions = src_subtask_gripper_actions[:,:1]
                    elif (arm_name == 'arm_right' and not change_role) or (arm_name == 'arm_left' and change_role):
                        # print('select right arm demo pose')
                        src_subtask_eef_poses = src_subtask_eef_poses[:,4:,:]
                        # src_subtask_target_poses = src_subtask_target_poses[:,4:,:]
                        src_subtask_gripper_actions = src_subtask_gripper_actions[:,1:]

                    if not isinstance(env.robot, OpenArmBimanual):
                        raise TypeError("ElogGen generation requires OpenArmBimanual")
                    torso_link_name = "base_link"
                    if subtask_object_name in ["robot_r1", torso_link_name]:
                        frame_to_use_for_src_object_pose = end_step_of_MP_local[current_phase_ind][arm_i][subtask_ind]
                    else:
                        frame_to_use_for_src_object_pose = selected_src_subtask_inds[0]
                    # get reference object pose from source demo
                    src_subtask_object_pose = src_ep_datagen_info.object_poses[subtask_object_name][frame_to_use_for_src_object_pose] if (subtask_object_name is not None) else None # 4 x 4

                    # src_eef_poses = np.array(src_subtask_eef_poses)
                    # if is_first_subtask or transform_first_robot_pose:
                    #     # Source segment consists of first robot eef pose and the target poses. This ensures that
                    #     # we will interpolate to the first robot eef pose in this source segment, instead of the
                    #     # first robot target pose.
                    #     # TODO: not sure about the meaning of this; need to check the first dimension is 1 more
                    #     src_eef_poses = np.concatenate([src_subtask_eef_poses[0:1], src_subtask_target_poses], axis=0) # 107 x 8 x 4
                    # else:
                    #     # Source segment consists of just the target poses.
                    #     src_eef_poses = np.array(src_subtask_target_poses)

                    # account for extra timestep added to @src_eef_poses
                    # src_subtask_gripper_actions = np.concatenate([src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], axis=0) # 107 x2

                    src_eef_poses = src_subtask_eef_poses
                    # Transform source demonstration segment using relevant object pose.
                    if subtask_object_name is not None:
                        # print('cur_object_pose', cur_object_pose.shape)
                        # print('src_eef_poses', src_eef_poses.shape)
                        # print('src_subtask_object_pose', src_subtask_object_pose.shape)

                        # If the object is symmetric, we don't need to transform the rotation part of the object pose
                        if local_task_spec[subtask_ind]["symmetric_object"]:
                            cur_object_pose[:3, :3] = src_subtask_object_pose[:3, :3]

                        transformed_eef_poses = PoseUtils.transform_source_data_segment_using_object_pose(
                            obj_pose=cur_object_pose, 
                            src_eef_poses=src_eef_poses,
                            src_obj_pose=src_subtask_object_pose)
                        # transformed_eef_poses = np.concatenate([transformed_eef_poses_left, transformed_eef_poses_right], axis=1)
                    else:
                        # skip transformation if no reference object is provided
                        transformed_eef_poses = src_eef_poses

                    # # visualize original and transformed eef poses
                    # self.visualize_traj(env, src_eef_poses, transformed_eef_poses)
                    
                    # We will construct a WaypointTrajectory instance to keep track of robot control targets 
                    # that will be executed and then execute it.
                    # traj_to_execute = WaypointTrajectory()

                    # TODO: change the interpolation to curobo motion planner

                    # if interpolate_from_last_target_pose and (not is_first_subtask_in_phase):
                    #     # Interpolation segment will start from last target pose (which may not have been achieved).

                    #     # TODO: since we did not execute the subtask within each phase, the assettion will fail -> remove the assertion
                    #     # assert prev_executed_traj is not None
                    #     # last_waypoint = prev_executed_traj.last_waypoint

                    #     # instead, we get the last waypoint from the last subtask
                    #     last_waypoint = traj_list_all[arm_i][-1].last_waypoint
                    #     init_sequence = WaypointSequence(sequence=[last_waypoint])
                    # else:
                    # if True:
                    # if arm_name == 'arm_left':
                    #     # Interpolation segment will start from current robot eef pose.
                    #     init_sequence = WaypointSequence.from_poses(
                    #         poses=cur_datagen_info.eef_pose[None][:,:4,:], # 1 x 8 x 4
                    #         gripper_actions=src_subtask_gripper_actions[0:1], # 1 x 1
                    #         action_noise=cur_phase_task_spec[0][subtask_ind]["action_noise"],
                    #     )
                    # elif arm_name == 'arm_right':
                    #     # Interpolation segment will start from current robot eef pose.
                    #     init_sequence = WaypointSequence.from_poses(
                    #         poses=cur_datagen_info.eef_pose[None][:,4:,:], # 1 x 4 x 4
                    #         gripper_actions=src_subtask_gripper_actions[0:1], # 1 x 1
                    #         action_noise=cur_phase_task_spec[1][subtask_ind]["action_noise"],
                    #     )

                    # print('init_sequence[0].pose.shape', init_sequence[0].pose.shape) # 4 x 4
                    # traj_to_execute.add_waypoint_sequence(init_sequence)

                    # Construct trajectory for the transformed segment.
                    transformed_seq = WaypointSequence.from_poses(
                        poses=transformed_eef_poses, # 107 x 4 x 4
                        gripper_actions=src_subtask_gripper_actions,
                        action_noise=local_task_spec[subtask_ind]["action_noise"],
                    )
                    transformed_traj = WaypointTrajectory()
                    transformed_traj.add_waypoint_sequence(transformed_seq)
                    # print('transformed_traj[10].pose.shape', transformed_traj[10].pose.shape) # 8 x 4

                    # Merge this trajectory into our trajectory using linear interpolation.
                    # Interpolation will happen from the initial pose (@init_sequence) to the first element of @transformed_seq.
                    # traj_to_execute.merge(
                    #     transformed_traj,
                    #     num_steps_interp=local_task_spec[subtask_ind]["num_interpolation_steps"],
                    #     num_steps_fixed=local_task_spec[subtask_ind]["num_fixed_steps"],
                    #     action_noise=(float(local_task_spec[subtask_ind]["apply_noise_during_interpolation"]) * local_task_spec[subtask_ind]["action_noise"]),
                    #     bimanual=self.bimanual
                    # )

                    # We initialized @traj_to_execute with a pose to allow @merge to handle linear interpolation
                    # for us. However, we can safely discard that first waypoint now, and just start by executing
                    # the rest of the trajectory (interpolation segment and transformed subtask segment).
                    # traj_to_execute.pop_first()

                    traj_to_execute = transformed_traj

                    # print('*****************************')
                    # print('finished processing one subtask for one arm')
                    # print('num sequences:', len(traj_to_execute.waypoint_sequences))
                    # for seq in traj_to_execute.waypoint_sequences:
                    #     print('num waypoints:', len(seq.sequence))
                
                    traj_list_all[arm_i].append(traj_to_execute)
                
                traj_to_execute = self.merge_trajs(traj_list_all)

                # reformat the local info with the current subtask start and end steps
                # TODO: the logic here can be problematic when other demonstration annotations, need to double check with other data demonstrations
                for i in range(2):
                    # Clip between selected_src_subtask_inds[0] and selected_src_subtask_inds[1]
                    MP_end_steps[i] = min(max(MP_end_steps[i], selected_src_subtask_inds[0]), selected_src_subtask_inds[1])
                    MP_end_steps[i] -= selected_src_subtask_inds[0]

                if change_role:
                    MP_end_steps = MP_end_steps[::-1]
                    # TODO: need to change the attached_obj_dict as well
                
                if not env.manipulation_only:
                    if not isinstance(env.robot, OpenArmBimanual):
                        raise TypeError("ElogGen generation requires OpenArmBimanual")
                    torso_link_name = "base_link"

                    ref_object = self._select_reference_object(object_ref)
                    if ref_object is None:
                        env.primitive._tracking_object = None
                        reachable_and_visible = True
                    elif object_ref["arm_left"] is not None and object_ref["arm_left"] in ["robot_r1", torso_link_name]:
                        reachable_and_visible = True
                    else:         
                        # ========== Check reachibility and visibility of the reference object ==============
                        check_only_last_mp_waypoint = True
                        reachable, visible = False, False

                        seq = traj_to_execute.waypoint_sequences[0]
                        cur_subtask_end_step_MP = MP_end_steps
                        
                        ref_obj = self._resolve_env_object(env=env.env, env_interface=env_interface, object_name=ref_object)
                        if ref_obj is None:
                            raise ValueError(
                                "Failed to resolve reference object '{}' for task '{}'. "
                                "The object may exist in task.object_scope but not in scene.object_registry, "
                                "or the config object_ref may not match the runtime object name.".format(
                                    ref_object,
                                    getattr(env, "name", getattr(env.env, "name", "unknown")),
                                )
                        )
                        env.primitive._tracking_object = ref_obj
                        print("Will track object for this sub-step: ", ref_obj.name)

                        # Inform primitive stack about attached object for this phase
                        robot = env.robot
                        attached_obj_new = {}
                        attached_obj_scale = {}
                        self.obtain_attached_object(env, robot, attached_obj_new, attached_obj_scale)
                        if attached_obj_new == {}:
                            attached_obj_new = None
                            attached_obj_scale = None
                        env.primitive.attached_obj_info = {"attached_obj": attached_obj_new, "attached_obj_scale": attached_obj_scale}
                        
                        # In case reachability test is done for all eef poses (last MP waypoint + replay waypoints)
                        if not check_only_last_mp_waypoint:
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

                        # In case reachability test is done for only the last MP waypoint
                        else:
                            left_mp_waypoints = seq[:cur_subtask_end_step_MP[0]]
                            left_waypoint = left_mp_waypoints[-1]
                            left_waypoint_pos, left_waypoint_ori = th.tensor(left_waypoint.pose[0:3, 3]), T.mat2quat(th.tensor(left_waypoint.pose[0:3, 0:3]))
                            right_mp_waypoints = seq[:cur_subtask_end_step_MP[1]]
                            right_waypoint = right_mp_waypoints[-1]
                            right_waypoint_pos, right_waypoint_ori = th.tensor(right_waypoint.pose[4:7, 3]), T.mat2quat(th.tensor(right_waypoint.pose[4:7, 0:3]))

                        eef_pose = {
                            "left": (left_waypoint_pos, left_waypoint_ori),
                            "right": (right_waypoint_pos, right_waypoint_ori)
                        }

                        if object_ref["arm_right"] is None:
                            eef_pose = {"left": (left_waypoint_pos, left_waypoint_ori)}
                        elif object_ref["arm_left"] is None:
                            eef_pose = {"right": (right_waypoint_pos, right_waypoint_ori)}
                        else:
                            eef_pose = {"left": (left_waypoint_pos, left_waypoint_ori), "right": (right_waypoint_pos, right_waypoint_ori)}

                        # Check reachability. Three options:
                        # 1. [USING THIS FOR NOW] Use IK check with collision and only use the last MP waypoint (not replay waypoints as those could have contacts/collisions with the world)
                        # pro: We care about a collision-free IK solution, which this computes. Alternative approach is not that efficient and accurate as you'll see
                        # con: Does not verify for replay waypoints. Which means reaply waypoitns could be unreacahble. This typically won't happen as replay is pretty small deltas
                        # 2. Use IK check without collision and use all (last MP waypoint + replay waypoints). Set the arm position from the returned IK solution for first target pose
                        # (last waypoint of MP) and check for collision.
                        # pro: Verifies for replay waypoints. 
                        # con: If the chosen IK solution is not collision-free, but there exists one that wasn't chosen, we unnecessarily fail this check.
                        # 3. Do IK check with collision for last MP wayoint and IK check without collision for replay waypoints. Might be overkill so only use this if needed.
                        # retval = env.primitive._ik_solver_cartesian_to_joint_space(target_pose=eef_pose,
                        #                                                         initial_joint_pos=env.robot.get_joint_positions(),
                        #                                                         skip_obstacle_update=False,
                        #                                                         ik_world_collision_check=True,
                        #                                                         emb_sel=CuRoboEmbodimentSelection.ARM_NO_TORSO)
                        
                        eyes_pose = env.robot.links["eyes"].get_position_orientation() if "eyes" in env.robot.links else None
                        if isinstance(env.robot, OpenArmBimanual):
                            # Fixed-base robot with no eyes/navigation: skip visibility check,
                            # assume the target is always reachable from the fixed base.
                            reachable_and_visible = True
                        else:
                            reachable_and_visible = env.primitive._target_in_reach_of_robot_and_visible(target_pose=eef_pose,
                                                                                    initial_joint_pos=env.robot.get_joint_positions(),
                                                                                    skip_obstacle_update=False,
                                                                                    ik_world_collision_check=True,
                                                                                    emb_sel=CuRoboEmbodimentSelection.ARM_NO_TORSO,
                                                                                    attach_obj=True,
                                                                                    eyes_pose=eyes_pose,)
                        print("object to be manipulated is reachable and visible: ", reachable_and_visible)
                        # ======================== End of reachibility and visibility check =========================
                # If we are in the debugging mode of "manipulation_only" for pick_cup task, don't check reachability and visibility
                else:
                    reachable_and_visible = True
                
                # NOTE: This is not being used right now. If manipulation MP fails, we retry nav and manipulation phases but only 1 extra time at max
                for nav_try in range(env.num_nav_retry_on_arm_mp_failure+1):
                    # 1. If object is not reachable or visible, add a navigation phase
                    if not reachable_and_visible or nav_try > 0:
                        print("=========== Navigation phase ===========")
                        # Execute the navigation trajectory and collect data.
                        with self._quiet_generation_stdout():
                            exec_results = traj_to_execute.execute(
                                env=env,
                                env_interface=env_interface,
                                render=render,
                                video_writer=video_writer,
                                video_skip=video_skip,
                                camera_names=camera_names,
                                bimanual=self.bimanual,
                                cur_subtask_end_step_MP=MP_end_steps,
                                # attached_obj=attached_obj[current_phase_ind][subtask_ind_reordered],
                                attached_obj=attached_obj_dict,
                                phase_type="navigation",
                                object_ref=object_ref,
                                enable_marker_vis=enable_marker_vis,
                                ds_ratio=ds_ratio,
                                grasp_init_views_video_writer=grasp_init_views_video_writer,
                                phase_logs=phase_logs,
                                frozen_arms=frozen_arms,
                            )
                        # To let any remaining simulation steps finish.
                        for _ in range(50): og.sim.step()
                    
                        # This means that the the current phase failed 
                        if exec_results is None:
                            # If we want to save partially completed tasks (that had atleast 1 phase executed successfully otherwise it's just an empty trajectory)
                            if not no_partial_tasks and env.phases_completed_wo_mp_err > 0:
                                if len(generated_actions) > 0:
                                    generated_actions = np.concatenate(generated_actions, axis=0)
                                    generated_src_demo_labels = np.concatenate(generated_src_demo_labels, axis=0)
                                results = dict(
                                    initial_state=new_initial_state,
                                    states=generated_states,
                                    observations=generated_obs,
                                    observations_info=generated_obs_info,
                                    datagen_infos=generated_datagen_infos,
                                    actions=generated_actions,
                                    success=generated_success,
                                    src_demo_inds=generated_src_demo_inds,
                                    src_demo_labels=generated_src_demo_labels,
                                    mp_end_steps=generated_demo_mp_end_steps,
                                    subtask_lengths=generated_demo_subtask_lengths,
                                    sensor_info=sensor_info,
                                    partial=True,
                                    phases_completed=env.phases_completed_wo_mp_err,
                                    left_mp_ranges=generated_demo_left_mp_ranges,
                                    right_mp_ranges=generated_demo_right_mp_ranges,
                                    phase_logs=phase_logs,
                                    frame_contexts=generated_frame_contexts,
                                    episode_metadata=episode_metadata,
                                )
                                return results
                            else:
                                return None

                        # check that trajectory is non-empty
                        if len(exec_results["states"]) > 0:
                            generated_states += exec_results["states"]
                            generated_obs += exec_results["observations"]
                            generated_obs_info += exec_results["observations_info"]
                            generated_datagen_infos += exec_results["datagen_infos"]
                            generated_actions.append(exec_results["actions"])
                            generated_demo_mp_end_steps.append(exec_results["mp_end_steps"])
                            if exec_results["left_mp_ranges"] is not None:
                                generated_demo_left_mp_ranges.append(exec_results["left_mp_ranges"])
                            if exec_results["right_mp_ranges"] is not None:
                                generated_demo_right_mp_ranges.append(exec_results["right_mp_ranges"])
                            generated_demo_subtask_lengths.append(exec_results["subtask_lengths"])
                            generated_success = generated_success or exec_results["success"]
                            generated_src_demo_inds.append(selected_src_demo_ind)
                            generated_src_demo_labels.append(selected_src_demo_ind * np.ones((exec_results["actions"].shape[0], 1), dtype=int))
                            new_frame_contexts = self._build_exec_frame_contexts(
                                env=env,
                                exec_results=exec_results,
                                object_ref=object_ref,
                                cur_phase_task_spec=cur_phase_task_spec,
                                current_phase_ind=current_phase_ind,
                                subtask_ind_reordered=subtask_ind_reordered,
                                selected_src_demo_ind=selected_src_demo_ind,
                                generated_start_frame=len(generated_frame_contexts),
                                phase_type="navigation",
                                execution_metadata=execution_metadata,
                                execution_order_index=execution_order_index,
                            )
                            self._log_stage_contexts(new_frame_contexts)
                            generated_frame_contexts.extend(new_frame_contexts)

                    
                    # 2. Now we can execute the manipulation segment
                    print("=========== Manipulation phase ===========")
                    src_curr_phase_actions = None
                    if self._is_openarm_task_env(env):
                        src_curr_phase_actions = self.src_actions[selected_src_demo_ind][
                            selected_src_subtask_inds[0] : selected_src_subtask_inds[1]
                        ]
                    # Execute the manipulation trajectory and collect data.
                    with self._quiet_generation_stdout():
                        exec_results = traj_to_execute.execute(
                            env=env,
                            env_interface=env_interface,
                            render=render,
                            video_writer=video_writer,
                            video_skip=video_skip,
                            camera_names=camera_names,
                            bimanual=self.bimanual,
                            cur_subtask_end_step_MP=MP_end_steps,
                            # attached_obj=attached_obj[current_phase_ind][subtask_ind_reordered],
                            attached_obj=attached_obj_dict,
                            phase_type=phase_type,
                            object_ref=object_ref,
                            enable_marker_vis=enable_marker_vis,
                            ds_ratio=ds_ratio,
                            grasp_init_views_video_writer=grasp_init_views_video_writer,
                            phase_logs=phase_logs,
                            retract_type=retract_type,
                            src_curr_phase_actions=src_curr_phase_actions,
                            frozen_arms=frozen_arms,
                        )
                    # To let any remaining simulation steps finish.
                    for _ in range(50): og.sim.step()
                    if exec_results is not None:
                        final_success_metrics = env.is_success()
                        exec_results["success"] = bool(exec_results["success"] or final_success_metrics.get("task", False))
                        if self._is_openarm_task_env(env):
                            success_summary = self._summarize_openarm_success_metrics(
                                env=env,
                                metrics=final_success_metrics,
                            )
                            print(f"[OpenArm] success after settle: {success_summary}")

                    if exec_results is not None and current_node is not None and self.subtask_scheduler is not None:
                        held_resources = self.subtask_scheduler.apply_resource_events(
                            held_resources, current_node
                        )
                
                    # Early terminate if the expecetd attached obj (according to the template) is not what is actually in the gripper
                    if execution_order_index < len(phase_execution_order) - 1:
                        if self.subtask_scheduler is not None:
                            left_expected_attached_obj = held_resources.get("left_gripper")
                            right_expected_attached_obj = held_resources.get("right_gripper")
                        else:
                            next_phase_ind = phase_execution_order[execution_order_index + 1]
                            next_phase_task_spec = self.task_spec[next_phase_ind]
                            left_expected_attached_obj = next_phase_task_spec[0][0]["attached_obj"]
                            right_expected_attached_obj = next_phase_task_spec[1][0]["attached_obj"]
                        attached_object_names = self.obtain_attached_object(env, env.robot)
                        attached_object_mismatch = False
                        # If left eef actually has an object 
                        if "left" in attached_object_names.keys():
                            if attached_object_names["left"] != left_expected_attached_obj:
                                attached_object_mismatch = True
                        # If left eef actually does not have an object
                        elif "left" not in attached_object_names.keys():
                            if left_expected_attached_obj is not None:
                                attached_object_mismatch = True
                        # If right eef actually has an object 
                        if "right" in attached_object_names.keys():
                            if attached_object_names["right"] != right_expected_attached_obj:
                                attached_object_mismatch = True
                        # If right eef actually does not have an object
                        elif "right" not in attached_object_names.keys():
                            if right_expected_attached_obj is not None:
                                attached_object_mismatch = True
                        
                        if attached_object_mismatch:
                            print("Attached object mismatch, terminating early")
                            exec_results = None
                    
                    # This means that the the current phase failed
                    if exec_results is None:
                        if str(getattr(env, "err", "")).startswith("OpenArmBagPhaseFailed"):
                            return None
                        # If we want to save partially completed tasks (that had atleast 1 phase executed successfully otherwise it's just an empty trajectory)
                        if not no_partial_tasks and env.phases_completed_wo_mp_err > 0:
                            if len(generated_actions) > 0:
                                generated_actions = np.concatenate(generated_actions, axis=0)
                                generated_src_demo_labels = np.concatenate(generated_src_demo_labels, axis=0)
                            results = dict(
                                initial_state=new_initial_state,
                                states=generated_states,
                                observations=generated_obs,
                                observations_info=generated_obs_info,
                                datagen_infos=generated_datagen_infos,
                                actions=generated_actions,
                                success=generated_success,
                                src_demo_inds=generated_src_demo_inds,
                                src_demo_labels=generated_src_demo_labels,
                                mp_end_steps=generated_demo_mp_end_steps,
                                subtask_lengths=generated_demo_subtask_lengths,
                                sensor_info=sensor_info,
                                partial=True,
                                phases_completed=env.phases_completed_wo_mp_err,
                                left_mp_ranges=generated_demo_left_mp_ranges,
                                right_mp_ranges=generated_demo_right_mp_ranges,
                                phase_logs=phase_logs,
                                frame_contexts=generated_frame_contexts,
                                episode_metadata=episode_metadata,
                            )
                            return results
                        else:
                            return None

                    # check that trajectory is non-empty
                    if len(exec_results["states"]) > 0:
                        generated_states += exec_results["states"]
                        generated_obs += exec_results["observations"]
                        generated_obs_info += exec_results["observations_info"]
                        generated_datagen_infos += exec_results["datagen_infos"]
                        generated_actions.append(exec_results["actions"])
                        generated_demo_mp_end_steps.append(exec_results["mp_end_steps"])
                        if exec_results["left_mp_ranges"] is not None:
                            generated_demo_left_mp_ranges.append(exec_results["left_mp_ranges"])
                        if exec_results["right_mp_ranges"] is not None:
                            generated_demo_right_mp_ranges.append(exec_results["right_mp_ranges"])
                        generated_demo_subtask_lengths.append(exec_results["subtask_lengths"])
                        generated_success = generated_success or exec_results["success"]
                        generated_src_demo_inds.append(selected_src_demo_ind)
                        generated_src_demo_labels.append(selected_src_demo_ind * np.ones((exec_results["actions"].shape[0], 1), dtype=int))
                        new_frame_contexts = self._build_exec_frame_contexts(
                            env=env,
                            exec_results=exec_results,
                            object_ref=object_ref,
                            cur_phase_task_spec=cur_phase_task_spec,
                            current_phase_ind=current_phase_ind,
                            subtask_ind_reordered=subtask_ind_reordered,
                            selected_src_demo_ind=selected_src_demo_ind,
                            generated_start_frame=len(generated_frame_contexts),
                            phase_type=phase_type,
                            execution_metadata=execution_metadata,
                            execution_order_index=execution_order_index,
                        )
                        self._log_stage_contexts(new_frame_contexts)
                        generated_frame_contexts.extend(new_frame_contexts)

                    # In most cases we don't need to retry nav. This is only trigered if manipulation MP (arm_no_torso mode) fails due to IK or TrajOpt failure 
                    if not exec_results["retry_nav"]:
                        break
                    
                    if pause_subtask:
                        input("Pausing after subtask {} execution. Press any key to continue...".format(subtask_ind))

        # TODO: why need to merge the generated actions
        # merge numpy arrays
        if len(generated_actions) > 0:
            generated_actions = np.concatenate(generated_actions, axis=0)
            generated_src_demo_labels = np.concatenate(generated_src_demo_labels, axis=0)

        results = dict(
            initial_state=new_initial_state,
            states=generated_states,
            observations=generated_obs,
            observations_info=generated_obs_info,
            datagen_infos=generated_datagen_infos,
            actions=generated_actions,
            success=generated_success,
            src_demo_inds=generated_src_demo_inds,
            src_demo_labels=generated_src_demo_labels,
            mp_end_steps=generated_demo_mp_end_steps,
            subtask_lengths=generated_demo_subtask_lengths,
            sensor_info=sensor_info,
            partial=False,
            phases_completed=env.phases_completed_wo_mp_err,
            left_mp_ranges=generated_demo_left_mp_ranges,
            right_mp_ranges=generated_demo_right_mp_ranges,
            phase_logs=phase_logs,
            frame_contexts=generated_frame_contexts,
            episode_metadata=episode_metadata,
        )
        return results
    
