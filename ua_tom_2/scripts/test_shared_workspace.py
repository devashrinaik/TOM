#!/usr/bin/env python3
"""
Test script: Custom shared-workspace TwoRobotPickCube environment.

Subclasses ManiSkill's TwoRobotPickCube with closer robot positions
so both arms can reach a shared central zone. Then verifies both arms
can reach the cube.
"""

import numpy as np
import sapien
import torch
import gymnasium as gym

from mani_skill.envs.tasks.tabletop.two_robot_pick_cube import TwoRobotPickCube
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.utils import randomization
from transforms3d.euler import euler2quat


# ── Custom environment with shared workspace ──────────────────────────
@register_env("SharedWorkspacePickCube-v1", max_episode_steps=200)
class SharedWorkspacePickCube(TwoRobotPickCube):
    """
    TwoRobotPickCube with robots moved closer together so both arms
    can reach a shared central workspace.

    Original: left at [0, -1, 0], right at [0, 1, 0]  (2m apart)
    Modified: left at [0, -0.35, 0], right at [0, 0.35, 0]  (0.7m apart)

    Cube spawns in the central zone (y ~ 0) reachable by both arms.
    """

    def _load_agent(self, options: dict):
        # Move robots closer: 0.35m from center instead of 1.0m
        # Skip TwoRobotPickCube's _load_agent, call BaseEnv's directly
        super(TwoRobotPickCube, self)._load_agent(
            options,
            [sapien.Pose(p=[0, -0.35, 0]), sapien.Pose(p=[0, 0.35, 0])]
        )

    # Robot separation: each arm 0.75m from center (1.5m total)
    # This matches TableSceneBuilder's default for dual panda_wristcam.
    # Both arms can reach the center (y≈0) within 0.033m — verified.
    # Shared zone is narrow (~0.1m wide at y=0) but sufficient.
    ROBOT_Y = 0.75

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Spawn cube in shared central zone, reachable by both arms."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Override robot positions AFTER table_scene.initialize()
            # (TableSceneBuilder hardcodes y=±0.75; we want y=±ROBOT_Y)
            self.agent.agents[0].robot.set_pose(
                sapien.Pose([0, -self.ROBOT_Y, 0], q=euler2quat(0, 0, np.pi / 2))
            )
            self.agent.agents[1].robot.set_pose(
                sapien.Pose([0, self.ROBOT_Y, 0], q=euler2quat(0, 0, -np.pi / 2))
            )

            self.left_init_qpos = self.left_agent.robot.get_qpos()

            # Cube in central zone: y in [-0.05, 0.05]
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05       # x: [-0.05, 0.05]
            xyz[:, 1] = torch.rand((b,)) * 0.1 - 0.05       # y: [-0.05, 0.05] (CENTER)
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # Goal also in shared zone but elevated
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            goal_xyz[:, 1] = torch.rand((b,)) * 0.1 - 0.05  # same central zone
            goal_xyz[:, 2] = torch.rand((b,)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))


# ── Frame transforms (discovered in previous testing) ─────────────────
def world_to_left(w):
    """World direction → left arm action frame."""
    return np.array([w[1], -w[0], w[2]])

def world_to_right(w):
    """World direction → right arm action frame."""
    return np.array([-w[1], w[0], w[2]])


# ── Test functions ────────────────────────────────────────────────────
def test_1_env_creation():
    """Test that the custom environment creates successfully."""
    print("\n" + "="*60)
    print("TEST 1: Environment Creation")
    print("="*60)

    env = gym.make(
        "SharedWorkspacePickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
    )
    obs, info = env.reset()

    # Check action space
    print(f"Action space: {env.action_space}")
    for key, space in env.action_space.spaces.items():
        print(f"  {key}: shape={space.shape}, range=[{space.low[0]:.1f}, {space.high[0]:.1f}]")

    # Check observations
    print(f"\nObservation keys: {list(obs.keys())}")
    extra = obs['extra']
    print(f"Extra keys: {list(extra.keys())}")

    cube_pos = extra['cube_pose'][0, :3].cpu().numpy()
    left_tcp = extra['left_arm_tcp'][0, :3].cpu().numpy()
    right_tcp = extra['right_arm_tcp'][0, :3].cpu().numpy()

    print(f"\nPositions:")
    print(f"  Cube:      ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
    print(f"  Left TCP:  ({left_tcp[0]:.3f}, {left_tcp[1]:.3f}, {left_tcp[2]:.3f})")
    print(f"  Right TCP: ({right_tcp[0]:.3f}, {right_tcp[1]:.3f}, {right_tcp[2]:.3f})")

    left_to_cube = np.linalg.norm(cube_pos - left_tcp)
    right_to_cube = np.linalg.norm(cube_pos - right_tcp)
    print(f"\nDistances to cube:")
    print(f"  Left arm:  {left_to_cube:.3f} m")
    print(f"  Right arm: {right_to_cube:.3f} m")

    env.close()
    return cube_pos, left_tcp, right_tcp


def test_2_both_arms_reach_cube(max_steps=80):
    """Test that BOTH arms can reach the cube position."""
    print("\n" + "="*60)
    print("TEST 2: Both Arms Reach Cube")
    print("="*60)

    env = gym.make(
        "SharedWorkspacePickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
    )
    obs, info = env.reset()

    extra = obs['extra']
    cube_pos = extra['cube_pose'][0, :3].cpu().numpy()
    print(f"Cube at: ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")

    # Phase 1: Move LEFT arm toward cube (30 steps)
    print("\n--- Phase 1: Left arm → cube ---")
    left_min_dist = float('inf')
    for step in range(max_steps // 2):
        extra = obs['extra']
        left_tcp = extra['left_arm_tcp'][0, :3].cpu().numpy()
        cube_pos = extra['cube_pose'][0, :3].cpu().numpy()

        direction = cube_pos - left_tcp
        direction[:2] *= 3.0  # stronger XY correction
        left_action = world_to_left(direction)
        left_action = np.clip(left_action[:3], -1, 1)
        # 4D: [dx, dy, dz, gripper] — open gripper
        left_cmd = np.concatenate([left_action, [1.0]])
        # Right arm stays still
        right_cmd = np.array([0.0, 0.0, 0.0, 1.0])

        action = {
            'panda_wristcam-0': torch.tensor([left_cmd], dtype=torch.float32),
            'panda_wristcam-1': torch.tensor([right_cmd], dtype=torch.float32),
        }
        obs, reward, terminated, truncated, info = env.step(action)

        dist = np.linalg.norm(cube_pos - left_tcp)
        left_min_dist = min(left_min_dist, dist)
        if step % 10 == 0:
            print(f"  Step {step}: left_tcp=({left_tcp[0]:.3f}, {left_tcp[1]:.3f}, {left_tcp[2]:.3f}), dist={dist:.3f}")

    print(f"  Left arm min distance to cube: {left_min_dist:.4f} m")

    # Reset for right arm test
    obs, info = env.reset()
    extra = obs['extra']
    cube_pos = extra['cube_pose'][0, :3].cpu().numpy()
    print(f"\n--- Phase 2: Right arm → cube (fresh reset) ---")
    print(f"  Cube at: ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")

    right_min_dist = float('inf')
    for step in range(max_steps // 2):
        extra = obs['extra']
        right_tcp = extra['right_arm_tcp'][0, :3].cpu().numpy()
        cube_pos = extra['cube_pose'][0, :3].cpu().numpy()

        direction = cube_pos - right_tcp
        direction[:2] *= 3.0
        right_action = world_to_right(direction)
        right_action = np.clip(right_action[:3], -1, 1)
        right_cmd = np.concatenate([right_action, [1.0]])
        left_cmd = np.array([0.0, 0.0, 0.0, 1.0])

        action = {
            'panda_wristcam-0': torch.tensor([left_cmd], dtype=torch.float32),
            'panda_wristcam-1': torch.tensor([right_cmd], dtype=torch.float32),
        }
        obs, reward, terminated, truncated, info = env.step(action)

        dist = np.linalg.norm(cube_pos - right_tcp)
        right_min_dist = min(right_min_dist, dist)
        if step % 10 == 0:
            print(f"  Step {step}: right_tcp=({right_tcp[0]:.3f}, {right_tcp[1]:.3f}, {right_tcp[2]:.3f}), dist={dist:.3f}")

    print(f"  Right arm min distance to cube: {right_min_dist:.4f} m")

    env.close()

    # Verdict
    print(f"\n{'='*40}")
    REACH_THRESHOLD = 0.05
    left_ok = left_min_dist < REACH_THRESHOLD
    right_ok = right_min_dist < REACH_THRESHOLD
    print(f"Left arm reaches cube:  {'YES' if left_ok else 'NO'} (min dist: {left_min_dist:.4f})")
    print(f"Right arm reaches cube: {'YES' if right_ok else 'NO'} (min dist: {right_min_dist:.4f})")
    print(f"SHARED WORKSPACE:       {'SUCCESS' if (left_ok and right_ok) else 'FAILED'}")

    return left_min_dist, right_min_dist


def test_3_left_arm_grasp():
    """Test that left arm can pick up the cube in shared workspace."""
    print("\n" + "="*60)
    print("TEST 3: Left Arm Grasp & Lift")
    print("="*60)

    env = gym.make(
        "SharedWorkspacePickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
    )
    obs, info = env.reset()

    extra = obs['extra']
    cube_pos_init = extra['cube_pose'][0, :3].cpu().numpy()
    print(f"Cube at: ({cube_pos_init[0]:.3f}, {cube_pos_init[1]:.3f}, {cube_pos_init[2]:.3f})")

    # Phased controller: above → descend → grasp → lift
    phases = [
        ("above",   15, lambda cp: np.array([cp[0], cp[1], cp[2] + 0.1])),
        ("descend", 15, lambda cp: np.array([cp[0], cp[1], cp[2] + 0.01])),
        ("grasp",   10, None),  # close gripper
        ("lift",    15, lambda cp: np.array([cp[0], cp[1], cp[2] + 0.3])),
    ]

    for phase_name, n_steps, target_fn in phases:
        print(f"\n  Phase: {phase_name} ({n_steps} steps)")
        for step in range(n_steps):
            extra = obs['extra']
            left_tcp = extra['left_arm_tcp'][0, :3].cpu().numpy()
            cube_pos = extra['cube_pose'][0, :3].cpu().numpy()

            if phase_name == "grasp":
                left_cmd = np.array([0.0, 0.0, 0.0, -1.0])  # close gripper
            else:
                target = target_fn(cube_pos)
                direction = target - left_tcp
                direction *= 3.0
                left_action = world_to_left(direction)
                left_action = np.clip(left_action[:3], -1, 1)
                gripper = 1.0 if phase_name != "lift" else -1.0
                left_cmd = np.concatenate([left_action, [gripper]])

            right_cmd = np.array([0.0, 0.0, 0.0, 1.0])
            action = {
                'panda_wristcam-0': torch.tensor([left_cmd], dtype=torch.float32),
                'panda_wristcam-1': torch.tensor([right_cmd], dtype=torch.float32),
            }
            obs, reward, terminated, truncated, info = env.step(action)

        extra = obs['extra']
        cp = extra['cube_pose'][0, :3].cpu().numpy()
        lt = extra['left_arm_tcp'][0, :3].cpu().numpy()
        print(f"    cube=({cp[0]:.3f}, {cp[1]:.3f}, {cp[2]:.3f}), tcp=({lt[0]:.3f}, {lt[1]:.3f}, {lt[2]:.3f})")

    final_cube_z = extra['cube_pose'][0, 2].item()
    lifted = final_cube_z > 0.1
    print(f"\n  Final cube Z: {final_cube_z:.3f}")
    print(f"  GRASP+LIFT: {'SUCCESS' if lifted else 'FAILED'}")

    env.close()
    return lifted


def test_4_right_arm_grasp():
    """Test that RIGHT arm can also pick up the cube in shared workspace."""
    print("\n" + "="*60)
    print("TEST 4: Right Arm Grasp & Lift")
    print("="*60)

    env = gym.make(
        "SharedWorkspacePickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
    )
    obs, info = env.reset()

    extra = obs['extra']
    cube_pos_init = extra['cube_pose'][0, :3].cpu().numpy()
    print(f"Cube at: ({cube_pos_init[0]:.3f}, {cube_pos_init[1]:.3f}, {cube_pos_init[2]:.3f})")

    phases = [
        ("above",   15, lambda cp: np.array([cp[0], cp[1], cp[2] + 0.1])),
        ("descend", 15, lambda cp: np.array([cp[0], cp[1], cp[2] + 0.01])),
        ("grasp",   10, None),
        ("lift",    15, lambda cp: np.array([cp[0], cp[1], cp[2] + 0.3])),
    ]

    for phase_name, n_steps, target_fn in phases:
        print(f"\n  Phase: {phase_name} ({n_steps} steps)")
        for step in range(n_steps):
            extra = obs['extra']
            right_tcp = extra['right_arm_tcp'][0, :3].cpu().numpy()
            cube_pos = extra['cube_pose'][0, :3].cpu().numpy()

            if phase_name == "grasp":
                right_cmd = np.array([0.0, 0.0, 0.0, -1.0])
            else:
                target = target_fn(cube_pos)
                direction = target - right_tcp
                direction *= 3.0
                right_action = world_to_right(direction)
                right_action = np.clip(right_action[:3], -1, 1)
                gripper = 1.0 if phase_name != "lift" else -1.0
                right_cmd = np.concatenate([right_action, [gripper]])

            left_cmd = np.array([0.0, 0.0, 0.0, 1.0])
            action = {
                'panda_wristcam-0': torch.tensor([left_cmd], dtype=torch.float32),
                'panda_wristcam-1': torch.tensor([right_cmd], dtype=torch.float32),
            }
            obs, reward, terminated, truncated, info = env.step(action)

        extra = obs['extra']
        cp = extra['cube_pose'][0, :3].cpu().numpy()
        rt = extra['right_arm_tcp'][0, :3].cpu().numpy()
        print(f"    cube=({cp[0]:.3f}, {cp[1]:.3f}, {cp[2]:.3f}), tcp=({rt[0]:.3f}, {rt[1]:.3f}, {rt[2]:.3f})")

    final_cube_z = extra['cube_pose'][0, 2].item()
    lifted = final_cube_z > 0.1
    print(f"\n  Final cube Z: {final_cube_z:.3f}")
    print(f"  GRASP+LIFT: {'SUCCESS' if lifted else 'FAILED'}")

    env.close()
    return lifted


if __name__ == "__main__":
    print("SharedWorkspacePickCube-v1 — Shared Workspace Validation")
    print("Robots at y=-0.5 and y=+0.5 (original: y=-1 and y=+1)")

    test_1_env_creation()
    test_2_both_arms_reach_cube()
    test_3_left_arm_grasp()
    test_4_right_arm_grasp()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
