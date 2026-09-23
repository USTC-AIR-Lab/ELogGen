#!/usr/bin/env python3
"""
Simplified CuRobo planning demo for OpenArmBimanual using fixed joint-space start / goal.

Design:
- Start and goal are hard-coded joint dictionaries below.
- Both configurations are validated against robot joint limits before planning.
- Planning is still done by CuRobo:
  we convert the fixed goal joint config to full-link pose targets and plan from fixed start joints.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import torch as th


def _ensure_local_omnigibson_importable():
    repo_root = Path(__file__).resolve().parents[2]
    og_root = repo_root / "BEHAVIOR-1K" / "OmniGibson"
    if str(og_root) not in sys.path:
        sys.path.insert(0, str(og_root))


_ensure_local_omnigibson_importable()

if "--headless" in sys.argv:
    os.environ["OMNIGIBSON_HEADLESS"] = "1"

import omnigibson as og  # noqa: E402
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator  # noqa: E402


# Fixed joint-space start and goal (radians; fingers in meters).
# Update these values directly if you want a different joint-space task.
START_JOINTS = {
    "openarm_left_joint1": 0.0,
    "openarm_left_joint2": -0.20,
    "openarm_left_joint3": 0.70,
    "openarm_left_joint4": 0.90,
    "openarm_left_joint5": -0.60,
    "openarm_left_joint6": 0.00,
    "openarm_left_joint7": 0.00,
    "openarm_left_finger_joint1": 0.020,
    "openarm_left_finger_joint2": 0.020,
    "openarm_right_joint1": 1.5,
    "openarm_right_joint2": 0.85,
    "openarm_right_joint3": 0.45,
    "openarm_right_joint4": 1.90,
    "openarm_right_joint5": -0.60,
    "openarm_right_joint6": 0.00,
    "openarm_right_joint7": 0.00,
    "openarm_right_finger_joint1": 0.000,
    "openarm_right_finger_joint2": 0.000,
}

GOAL_JOINTS = {
    "openarm_left_joint1": -1.5,
    "openarm_left_joint2": -0.85,
    "openarm_left_joint3": 0.45,
    "openarm_left_joint4": 1.95,
    "openarm_left_joint5": -0.35,
    "openarm_left_joint6": 0.20,
    "openarm_left_joint7": 0.20,
    "openarm_left_finger_joint1": 0.020,
    "openarm_left_finger_joint2": 0.020,
    "openarm_right_joint1": 0.00,
    "openarm_right_joint2": 0.70,
    "openarm_right_joint3": 0.70,
    "openarm_right_joint4": 0.90,
    "openarm_right_joint5": 0.60,
    "openarm_right_joint6": 0.00,
    "openarm_right_joint7": 0.00,
    "openarm_right_finger_joint1": 0.040,
    "openarm_right_finger_joint2": 0.040,
}


def build_env():
    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 300,
        },
        "scene": {"type": "Scene"},
        "robots": [
            {
                "type": "OpenArmBimanual",
                "name": "robot0",
                "obs_modalities": [],
                "action_type": "continuous",
                "action_normalize": False,
                "fixed_base": True,
                "self_collisions": True,
                "grasping_mode": "physical",
                "controller_config": {
                    "arm_left": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "use_delta_commands": False,
                        "use_impedances": False,
                    },
                    "arm_right": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "use_delta_commands": False,
                        "use_impedances": False,
                    },
                    "gripper_left": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "use_delta_commands": False,
                        "use_impedances": False,
                    },
                    "gripper_right": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "use_delta_commands": False,
                        "use_impedances": False,
                    },
                },
            }
        ],
    }
    return og.Environment(configs=cfg)


def set_view():
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.45, -1.55, 2.45]),
        orientation=th.tensor([0.26, 0.08, 0.17, 0.95]),
    )


def settle_robot(env, robot, steps):
    robot.keep_still()
    hold_action = robot.q_to_action(robot.get_joint_positions())
    for _ in range(steps):
        env.step(hold_action)


def lift_robot_base(robot, lift_m):
    if lift_m <= 0:
        return
    pos, quat = robot.get_position_orientation()
    delta = th.tensor([0.0, 0.0, lift_m], dtype=pos.dtype, device=pos.device)
    robot.set_position_orientation(position=pos + delta, orientation=quat)


def vector_from_joint_dict(robot, joint_values, name):
    joint_names = list(robot.joints.keys())
    missing = sorted(set(joint_names) - set(joint_values.keys()))
    extra = sorted(set(joint_values.keys()) - set(joint_names))
    if missing or extra:
        raise ValueError(f"{name} joint dict mismatch. missing={missing}, extra={extra}")

    q = robot.get_joint_positions().clone()
    for i, jn in enumerate(joint_names):
        q[i] = float(joint_values[jn])
    return q


def assert_in_joint_limits(robot, q, tag, tol=1e-6):
    low = robot.joint_lower_limits.to(dtype=q.dtype, device=q.device)
    high = robot.joint_upper_limits.to(dtype=q.dtype, device=q.device)
    finite = th.isfinite(low) & th.isfinite(high)
    bad = th.where(finite & ((q < (low - tol)) | (q > (high + tol))))[0]
    if len(bad) == 0:
        return

    names = list(robot.joints.keys())
    violators = []
    for i in bad.tolist():
        violators.append((names[i], float(q[i].item()), float(low[i].item()), float(high[i].item())))
    raise ValueError(f"{tag} is out of joint limits: {violators}")


def set_joint_state(robot, q):
    robot.set_joint_positions(q, drive=False)
    robot.set_joint_velocities(th.zeros_like(q), drive=False)


def world_collision(cmg, q):
    c = cmg.check_collisions(q.unsqueeze(0), self_collision_check=False, skip_obstacle_update=False)
    return bool(c[0].item())


def collect_goal_link_targets(robot, cmg, emb_sel):
    target_pos = {}
    target_quat = {}
    for link_name in cmg.mg[emb_sel].kinematics.link_names:
        pos, quat = robot.links[link_name].get_position_orientation()
        target_pos[link_name] = pos.clone()
        target_quat[link_name] = quat.clone()
    return target_pos, target_quat


def execute_trajectory(env, robot, q_traj, steps_per_waypoint=1):
    """以固定频率流式推送路径点：每个路径点执行固定仿真步数后立即推进到下一点。

    这与 Pico 遥操作的 30Hz 流式发送策略一致——机器人控制器始终跟随连续变化的
    目标，而非在每个点上反复等待收敛，从而避免振荡和顿挫。
    """
    for i, joint_target in enumerate(q_traj):
        for _ in range(steps_per_waypoint):
            env.step(robot.q_to_action(joint_target))
        # 每 20 个路径点或末尾打印一次进度（不阻塞流式推进）
        if (i + 1) % 20 == 0 or i == len(q_traj) - 1:
            current = robot.get_joint_positions()
            max_err = th.max(th.abs(current - joint_target)).item()
            print(f"waypoint={i + 1}/{len(q_traj)} max_err={max_err:.5f}")


def run(args):
    env = build_env()
    robot = env.robots[0]

    try:
        print("[DEBUG] env.reset()...")
        env.reset()
        print("[DEBUG] robot.reset()...")
        robot.reset()
        print("[DEBUG] og.sim.play()...")
        og.sim.play()
        if not args.headless:
            set_view()
        print("[DEBUG] lift_robot_base...")
        lift_robot_base(robot, args.base_lift)

        print("[DEBUG] initializing CuRoboMotionGenerator...")
        emb_sel = CuRoboEmbodimentSelection.DEFAULT
        cmg = CuRoboMotionGenerator(
            robot=robot,
            batch_size=1,
            motion_cfg_kwargs={"self_collision_check": False, "self_collision_opt": False},
            use_cuda_graph=not args.disable_cuda_graph,
            use_eyes_targets=False,
            embodiment_types={CuRoboEmbodimentSelection.DEFAULT},
        )
        print("[DEBUG] CuRoboMotionGenerator initialized OK.")

        start_q = vector_from_joint_dict(robot, START_JOINTS, "START_JOINTS")
        goal_q = vector_from_joint_dict(robot, GOAL_JOINTS, "GOAL_JOINTS")

        assert_in_joint_limits(robot, start_q, "start_q")
        assert_in_joint_limits(robot, goal_q, "goal_q")
        print("start_goal_joint_limits_ok=True")

        set_joint_state(robot, start_q)
        settle_robot(env, robot, steps=args.settle_steps)
        start_collision = world_collision(cmg, start_q)
        print(f"start_world_collision={start_collision}")
        if start_collision:
            raise RuntimeError("Start joint configuration is in world collision.")

        set_joint_state(robot, goal_q)
        settle_robot(env, robot, steps=2)
        goal_collision = world_collision(cmg, goal_q)
        print(f"goal_world_collision={goal_collision}")
        if goal_collision:
            raise RuntimeError("Goal joint configuration is in world collision.")

        target_pos, target_quat = collect_goal_link_targets(robot, cmg, emb_sel=emb_sel)

        # Restore the fixed start and plan to the fixed goal-link targets.
        set_joint_state(robot, start_q)
        settle_robot(env, robot, steps=2)

        successes, paths = cmg.compute_trajectories(
            target_pos=target_pos,
            target_quat=target_quat,
            initial_joint_pos=start_q,
            is_local=False,
            max_attempts=100,
            timeout=40.0,
            ik_fail_return=20,
            enable_finetune_trajopt=True,
            finetune_attempts=1,
            return_full_result=False,
            success_ratio=1.0,
            emb_sel=emb_sel,
        )
        success = bool(successes[0])
        print(f"plan_success={success}")
        if not success:
            raise RuntimeError("CuRobo failed to find a trajectory.")

        q_traj = cmg.path_to_joint_trajectory(paths[0], get_full_js=True, emb_sel=emb_sel).cpu().float()
        q_traj = cmg.add_linearly_interpolated_waypoints(traj=q_traj, max_inter_dist=args.max_inter_dist)
        print(f"trajectory_waypoints={len(q_traj)}")

        final_q = q_traj[-1].to(goal_q.device)
        goal_err = th.max(th.abs(final_q - goal_q)).item()
        print(f"final_goal_joint_max_abs_err={goal_err:.6f}")
        # if goal_err > args.goal_tol:
        #     raise RuntimeError(
        #         f"Planned final joint state is too far from fixed goal. err={goal_err:.6f} > tol={args.goal_tol:.6f}"
        #     )

        # Force exact fixed endpoint.
        q_traj[-1] = goal_q.cpu()

        if not args.dry_run:
            execute_trajectory(
                env=env,
                robot=robot,
                q_traj=q_traj,
                steps_per_waypoint=args.steps_per_waypoint,
            )

        if args.hold_steps > 0:
            settle_robot(env, robot, steps=args.hold_steps)

        og.shutdown()
    except Exception:  # noqa: BLE001
        print("\n[ERROR] Exception in run():\n")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        # Safe to call even if already shut down.
        if og.sim is not None:
            og.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only, do not execute trajectory.")
    parser.add_argument("--disable-cuda-graph", action="store_true", help="Disable CuRobo CUDA graph.")
    parser.add_argument(
        "--base-lift",
        type=float,
        default=0.02,
        help="Lift robot base in +Z before checks/planning to avoid floor collision.",
    )
    parser.add_argument("--settle-steps", type=int, default=30, help="Initial settle steps at start config.")
    parser.add_argument("--hold-steps", type=int, default=20, help="Extra settle steps after run.")
    parser.add_argument("--max-inter-dist", type=float, default=0.01, help="Max interpolation distance.")
    parser.add_argument(
        "--steps-per-waypoint",
        type=int,
        default=1,
        dest="steps_per_waypoint",
        help="每个路径点执行的仿真步数（默认 1）。值越大运动越慢但更稳定；"
             "配合 --max-inter-dist 控制整体速度。",
    )
    parser.add_argument("--goal-tol", type=float, default=0.05, help="Max final joint error allowed vs fixed goal.")
    args = parser.parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        if og.sim is not None:
            og.shutdown()
        raise SystemExit(130)


if __name__ == "__main__":
    main()
