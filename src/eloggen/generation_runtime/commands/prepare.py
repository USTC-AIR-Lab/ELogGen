"""Build the processed generation HDF5 from a source OpenArm demonstration."""
import os
import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase

import eloggen.generation_runtime.datasets as DatasetUtils
from eloggen.generation_runtime.interface_names import canonicalize_env_interface_name
from eloggen.generation_runtime.simulation.base import make_interface
from eloggen.generation_runtime.simulation.camera import (
    is_openarm_real_exp_1,
    set_real_exp_1_viewer_camera,
    uses_real_exp_1_viewer_camera,
)
from eloggen.generation_runtime.source_workspace import create_replay_dataset_workspace
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm


def set_openarm_viewer_camera(env_interface_name, dataset_path):
    """Restore the viewer pose used by the bundled OpenArm demonstrations."""
    # Use the complete task-pack path rather than only the file basename. New
    # task-pack interfaces are generic, so the parent directory now carries the
    # task identity (e.g. openarm_real_exp_1/source/source.hdf5).
    camera_key = f"{env_interface_name} {os.path.abspath(dataset_path)}".lower()
    if not any(name in camera_key for name in (
        "openarmrealexp1",
        "openarm_real_exp_1",
        "openarmdrawerstorage",
        "openarm_drawer_storage",
        "openarmfruitbasketbagging",
        "openarm_fruit_basket_bagging",
    )):
        return

    import omnigibson as og
    import torch as th
    from scipy.spatial.transform import Rotation as R

    if (
        "openarm_drawer_storage" in camera_key
        or "openarmdrawerstorage" in camera_key
        or "openarm_fruit_basket_bagging" in camera_key
        or "openarmfruitbasketbagging" in camera_key
        or uses_real_exp_1_viewer_camera(camera_key)
    ):
        set_real_exp_1_viewer_camera(og, print_prefix=f"[viewer_camera][{env_interface_name}]")
        return

    if is_openarm_real_exp_1(camera_key):
        set_real_exp_1_viewer_camera(og, print_prefix="[viewer_camera][openarm_real_exp_1]")
        return

    camera_pos = th.tensor([-0.855, 4.339, 1.813], dtype=th.float32)
    # Keep this in sync with collect_src_pico.py. The viewer UI displays
    # intrinsic X/Y/Z Euler angles, and OmniGibson expects xyzw quaternions.
    camera_euler_xyz_deg = [-14.483, -0.105, -179.591]
    camera_quat_xyzw = R.from_euler("XYZ", camera_euler_xyz_deg, degrees=True).as_quat()
    camera_quat = th.tensor(camera_quat_xyzw, dtype=th.float32)

    og.sim.viewer_camera.horizontal_aperture = 35.0
    og.sim.viewer_camera.set_position_orientation(
        position=camera_pos,
        orientation=camera_quat,
    )
    print(
        "[viewer_camera] using collection pose: "
        f"pos={camera_pos.tolist()} euler_xyz_deg={camera_euler_xyz_deg} quat_xyzw={camera_quat.tolist()}"
    )


def extract_datagen_info_from_trajectory(
    env,
    env_interface,
    initial_state,
    states,
    actions,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of robomimic EnvBase): environment

        env_interface (EG_EnvInterface instance): environment interface for some data generation operations

        initial_state (dict): initial simulation state to load

        states (np.array): array of simulation states to load to extract information

        actions (np.array): array of actions

    Returns:
        datagen_infos (dict): the datagen info objects across all timesteps represented as a dictionary of
            numpy arrays, for easy writes to an hdf5
    """
    assert isinstance(env, EnvBase)
    assert len(states) == actions.shape[0]

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    all_datagen_infos = []
    traj_len = len(states)
    for t in range(traj_len):
        # reset to state
        print('timestep:', t)
        env.reset_to({"states": states[t]})

        # extract datagen info as a dictionary
        # datagen_info is a dict with dict_keys(['eef_pose', 'object_poses', 'subtask_term_signals', 'target_pose', 'gripper_action'])
        datagen_info = env_interface.get_datagen_info(action=actions[t]).to_dict()
        all_datagen_infos.append(datagen_info)

    # convert list of dict to dict of list for datagen info dictionaries (for convenient writes to an hdf5 dataset)
    all_datagen_infos = TensorUtils.list_of_flat_dict_to_dict_of_list(all_datagen_infos)

    for k in all_datagen_infos:
        if k in ["object_poses", "subtask_term_signals"]:
            # convert list of dict to dict of list again
            all_datagen_infos[k] = TensorUtils.list_of_flat_dict_to_dict_of_list(all_datagen_infos[k])
            # list to numpy array
            for k2 in all_datagen_infos[k]:
                all_datagen_infos[k][k2] = np.array(all_datagen_infos[k][k2])
        else:
            # list to numpy array
            all_datagen_infos[k] = np.array(all_datagen_infos[k])

    return all_datagen_infos


def prepare_source_dataset(
    dataset_path,
    env_interface_name,
    env_interface_type,
    filter_key=None,
    n=None,
    generate_processed_hdf5=False,
    replay_for_annotation=False,
    steps=50,
    output_path=None,
):
    """
    Replay a raw source demonstration and optionally build its processed HDF5.

    The raw ``dataset_path`` is treated as immutable. OmniGibson preprocessing
    can modify an HDF5 in-place, so ``source prepare`` preprocesses the copied
    processed output and ``source inspect-boundaries`` preprocesses a temporary
    replay copy that is removed afterwards.

    Args:
        dataset_path (str): immutable raw input HDF5 dataset

        env_interface_name (str): name of environment interface class to use for this source dataset

        env_interface_type (str): type of environment interface to use for this source dataset

        filter_key (str or None): name of filter key

        n (int or None): if provided, stop after n trajectories are processed

        generate_processed_hdf5 (bool): if True, generate the processed hdf5 with datagen_info key

        replay_for_annotation (bool): if True, replay the dataset to break after X steps to note down the MP_end_step and subtask_term_step for each subtask
        steps (int): replay step interval for annotation breakpoints
    """
    # Legacy MG_* values are accepted as input only. From this point onward the
    # prepare path, runtime lookup, logging, and newly written HDF5 metadata all
    # use the canonical EG_* interface name.
    env_interface_name = canonicalize_env_interface_name(env_interface_name)

    source_dataset_path = os.path.abspath(dataset_path)

    # A processed file belongs beside its source (or at an explicit task-pack
    # path), never in a working-directory-dependent runtime folder.
    if output_path is None:
        output_path = os.path.join(os.path.dirname(source_dataset_path), "processed.hdf5")
    output_path = os.path.abspath(output_path)

    workspace = create_replay_dataset_workspace(
        source_dataset_path,
        output_path=output_path,
        persistent=generate_processed_hdf5,
    )
    replay_dataset_path = str(workspace.replay_path)

    try:
        # Important: preprocessing is intentionally applied to the replay copy,
        # never to the task-pack source HDF5.
        if env_interface_type == "omnigibson" or env_interface_type == "omnigibson_bimanual":
            FileUtils.preprocess_omnigibson_dataset(replay_dataset_path)

        # create environment that was to collect source demonstrations
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=replay_dataset_path)

        print("==== Using environment with the following metadata ====")
        print(env_meta)

        gm.ENABLE_TRANSITION_RULES = False
        env = DataPlaybackWrapper.create_from_hdf5(
            input_path=replay_dataset_path,
            output_path=None,
            robot_obs_modalities=(),
            robot_sensor_config=None,
            external_sensors_config=None,
            n_render_iterations=1,
            only_successes=False
        )
        # Keep task identification tied to the original task-pack path even when
        # replaying a temporary copy.
        set_openarm_viewer_camera(
            env_interface_name=env_interface_name,
            dataset_path=source_dataset_path,
        )

        # create environment interface for us to grab relevant information from simulation at each timestep
        env_interface = make_interface(
            name=env_interface_name,
            interface_type=env_interface_type,
            # NOTE: env_interface takes underlying simulation environment, not robomimic wrapper
            env=env,
        )
        print("Created environment interface: {}".format(env_interface))

        # get list of source demonstration keys from the replay-safe hdf5
        demos = DatasetUtils.get_all_demos_from_dataset(
            dataset_path=replay_dataset_path,
            filter_key=filter_key,
            start=None,
            n=n,
        )

        demo_ids = None
        # Only playback the demos filtered by the filter_key
        if filter_key is not None:
            demo_ids = [int(demo.split("_")[-1]) for demo in demos]

        if generate_processed_hdf5:
            print(f"Processed dataset output: {output_path}")
        else:
            print(
                "Replaying temporary copy; raw source remains unchanged: "
                f"{source_dataset_path}"
            )

        all_datagen_info = env.playback_dataset(record_data=False,
                                                callback=env_interface.get_datagen_info,
                                                demo_ids=demo_ids,
                                                replay_for_annotation=replay_for_annotation,
                                                steps=steps)

        env.input_hdf5.close()

        if not generate_processed_hdf5:
            print("Not generating the processed hdf5. Raw source was not modified.")
            if env_interface_type in {"omnigibson", "omnigibson_bimanual"}:
                import omnigibson as og
                og.shutdown()
            return

        # Modify only the copied processed output.
        f = h5py.File(replay_dataset_path, "a")

        for ind in tqdm(range(len(demos))):
            ep = demos[ind]
            ep_grp = f["data/{}".format(ep)]

            datagen_info = all_datagen_info[ind]
            datagen_info = [info.to_dict() for info in datagen_info]

            # convert list of dict to dict of list for datagen info dictionaries (for convenient writes to hdf5 dataset)
            datagen_info = TensorUtils.list_of_flat_dict_to_dict_of_list(datagen_info)

            for k in datagen_info:
                if k in ["object_poses", "subtask_term_signals", "articulated_states"]:
                    # convert list of dict to dict of list again
                    datagen_info[k] = TensorUtils.list_of_flat_dict_to_dict_of_list(datagen_info[k])
                    # list to numpy array
                    for k2 in datagen_info[k]:
                        datagen_info[k][k2] = np.array(datagen_info[k][k2])
                else:
                    # list to numpy array
                    datagen_info[k] = np.array(datagen_info[k])

            # delete old dategen info if it already exists
            if "datagen_info" in ep_grp:
                del ep_grp["datagen_info"]

            for k in datagen_info:
                if k in ["object_poses", "subtask_term_signals", "articulated_states"]:
                    # handle dict
                    for k2 in datagen_info[k]:
                        ep_grp.create_dataset("datagen_info/{}/{}".format(k, k2), data=np.array(datagen_info[k][k2]))
                else:
                    ep_grp.create_dataset("datagen_info/{}".format(k), data=np.array(datagen_info[k]))

            # remember the canonical env interface used too
            ep_grp["datagen_info"].attrs["env_interface_name"] = env_interface_name
            ep_grp["datagen_info"].attrs["env_interface_type"] = env_interface_type

        print("Modified {} trajectories to include datagen info.".format(len(demos)))
        f.close()

        # Properly shutdown omnigibson if needed
        if env_interface_type == "omnigibson" or env_interface_type == "omnigibson_bimanual":
            import omnigibson as og
            og.shutdown()
    finally:
        # For inspect-boundaries this removes the hidden replay copy. For
        # source prepare the workspace is persistent and cleanup is a no-op.
        workspace.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to immutable raw input hdf5 dataset",
    )
    parser.add_argument(
        "--env_interface",
        type=str,
        required=True,
        help="name of environment interface class to use for this source dataset",
    )
    parser.add_argument(
        "--env_interface_type",
        type=str,
        required=True,
        help="type of environment interface to use for this source dataset",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) name of filter key, to select a subset of demo keys",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="processed HDF5 path (default: processed.hdf5 beside --dataset)",
    )
    parser.add_argument(
        "--generate_processed_hdf5",
        action='store_true',
        help="if passed, copy the raw source to the processed output and add datagen_info there",
    )
    parser.add_argument(
        "--replay_for_annotation",
        action='store_true',
        help="if passed, replay a temporary dataset copy and break after X steps for boundary annotation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of replay steps between annotation breakpoints when --replay_for_annotation is passed",
    )

    args = parser.parse_args()
    prepare_source_dataset(
        dataset_path=args.dataset,
        env_interface_name=args.env_interface,
        env_interface_type=args.env_interface_type,
        filter_key=args.filter_key,
        n=args.n,
        generate_processed_hdf5=args.generate_processed_hdf5,
        replay_for_annotation=args.replay_for_annotation,
        steps=args.steps,
        output_path=args.output,
    )
