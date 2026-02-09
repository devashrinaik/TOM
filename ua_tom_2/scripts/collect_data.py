#!/usr/bin/env python3
"""
ManiSkill3 Data Collection for UA-ToM
======================================

Collects .npz datasets from ManiSkill3 tasks for partner modeling experiments.

Outputs are compatible with ua_tom's PartnerDataset:
    - observations: [N, T, obs_dim] state vectors
    - images: [N, T, H, W, 3] RGB images (optional)
    - partner_actions: [N, T] discrete action indices
    - partner_types: [N, T] partner type labels
    - switch_times: [N] timestep of partner switch (-1 if none)

Usage:
    # Synthetic data (no GPU required)
    python scripts/collect_data.py --synthetic --task PickCube-v1 --episodes 200 --output data/
    python scripts/collect_data.py --synthetic --tasks PickCube-v1 StackCube-v1 --episodes 200 --output data/

    # ManiSkill (requires GPU)
    python scripts/collect_data.py --task PickCube-v1 --episodes 200 --output data/

    # Test / list
    python scripts/collect_data.py --test
    python scripts/collect_data.py --list
"""

import os
os.environ['MANI_SKILL_NO_DISPLAY'] = '1'

import numpy as np
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Partner Strategies
# =============================================================================

class PartnerStrategy(Enum):
    FAST = 0
    SLOW = 1
    LEFT_PREFER = 2
    RIGHT_PREFER = 3
    AGGRESSIVE = 4
    CAUTIOUS = 5


# Strategy-specific parameters: (speed_mult, noise_scale, spatial_bias)
STRATEGY_PARAMS = {
    PartnerStrategy.FAST: (1.5, 0.03, np.zeros(3)),
    PartnerStrategy.SLOW: (0.4, 0.01, np.zeros(3)),
    PartnerStrategy.LEFT_PREFER: (1.0, 0.02, np.array([-0.02, 0.0, 0.0])),
    PartnerStrategy.RIGHT_PREFER: (1.0, 0.02, np.array([0.02, 0.0, 0.0])),
    PartnerStrategy.AGGRESSIVE: (1.3, 0.04, np.zeros(3)),
    PartnerStrategy.CAUTIOUS: (0.5, 0.01, np.zeros(3)),
}

NUM_STRATEGIES = len(PartnerStrategy)


# =============================================================================
# Task Registry
# =============================================================================

# Tasks grouped by category with their control modes
TASK_REGISTRY = {
    # Tabletop manipulation (pd_ee_delta_pose)
    'PickCube-v1': 'pd_ee_delta_pose',
    'StackCube-v1': 'pd_ee_delta_pose',
    'PegInsertionSide-v1': 'pd_ee_delta_pose',
    'PushCube-v1': 'pd_ee_delta_pose',
    'PullCube-v1': 'pd_ee_delta_pose',
    'LiftPegUpright-v1': 'pd_ee_delta_pose',
    'PickSingleYCB-v1': 'pd_ee_delta_pose',
    'PlugCharger-v1': 'pd_ee_delta_pose',
    'PokeCube-v1': 'pd_ee_delta_pose',
    'PlaceSphere-v1': 'pd_ee_delta_pose',
    'TurnFaucet-v1': 'pd_ee_delta_pose',
    'PushT-v1': 'pd_ee_delta_pose',
    # Multi-robot
    'TwoRobotPickCube-v1': 'pd_ee_delta_pose',
    'TwoRobotStackCube-v1': 'pd_ee_delta_pose',
    # Dexterous (TriFinger uses pd_joint_delta_pos)
    'TriFingerRotateCubeLevel0-v1': 'pd_joint_delta_pos',
    'TriFingerRotateCubeLevel1-v1': 'pd_joint_delta_pos',
    'TriFingerRotateCubeLevel2-v1': 'pd_joint_delta_pos',
    'TriFingerRotateCubeLevel3-v1': 'pd_joint_delta_pos',
    'TriFingerRotateCubeLevel4-v1': 'pd_joint_delta_pos',
}

MULTI_ROBOT_TASKS = {'TwoRobotPickCube-v1', 'TwoRobotStackCube-v1'}

TASK_CATEGORIES = {
    'tabletop': [
        'PickCube-v1', 'StackCube-v1', 'PegInsertionSide-v1',
        'PushCube-v1', 'PullCube-v1', 'LiftPegUpright-v1',
        'PickSingleYCB-v1', 'PlugCharger-v1', 'PokeCube-v1',
        'PlaceSphere-v1', 'TurnFaucet-v1', 'PushT-v1',
    ],
    'multi_robot': [
        'TwoRobotPickCube-v1', 'TwoRobotStackCube-v1',
    ],
    'dexterous': [
        f'TriFingerRotateCubeLevel{i}-v1' for i in range(5)
    ],
}


# =============================================================================
# State Extraction
# =============================================================================

@dataclass
class TaskState:
    """Extracted task state for partner action computation and storage."""
    tcp_pos: np.ndarray        # [3] tool center point XYZ
    object_pos: np.ndarray     # [3] primary object XYZ
    goal_pos: np.ndarray       # [3] goal position XYZ
    gripper_state: float       # 0=closed, 1=open (normalized)
    gripper_vel: float         # gripper velocity
    dist_to_object: float      # hand-object distance
    dist_object_to_goal: float # object-goal distance
    object_height: float       # object Z position
    is_grasped: bool           # whether object is grasped

    def to_array(self) -> np.ndarray:
        """Flatten to 15D float32 vector."""
        return np.concatenate([
            self.tcp_pos,                      # 0-2
            self.object_pos,                   # 3-5
            self.goal_pos,                     # 6-8
            [self.gripper_state],              # 9
            [self.gripper_vel],                # 10
            [self.dist_to_object],             # 11
            [self.dist_object_to_goal],        # 12
            [self.object_height],              # 13
            [float(self.is_grasped)],          # 14
        ]).astype(np.float32)

    @staticmethod
    def dim() -> int:
        return 15


def _to_numpy(x):
    """Convert tensor/array to flat numpy."""
    if x is None:
        return None
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    x = np.asarray(x).flatten()
    return x


def extract_state(obs: Dict, env) -> TaskState:
    """Extract TaskState from ManiSkill observation + environment internals."""
    tcp_pos = np.zeros(3)
    object_pos = np.zeros(3)
    goal_pos = np.array([0.0, 0.0, 0.15])
    gripper_state = 1.0
    gripper_vel = 0.0
    is_grasped = False

    if not isinstance(obs, dict):
        return _default_state()

    # TCP position and goal from obs['extra']
    extra = obs.get('extra', {})
    if isinstance(extra, dict):
        # Single-robot: 'tcp_pose', Multi-robot: 'left_arm_tcp' or 'right_arm_tcp'
        for tcp_key in ('tcp_pose', 'left_arm_tcp', 'right_arm_tcp'):
            if tcp_key in extra:
                tcp = _to_numpy(extra[tcp_key])
                if tcp is not None and len(tcp) >= 3:
                    tcp_pos = tcp[:3].astype(float)
                    break
        if 'goal_pos' in extra:
            gp = _to_numpy(extra['goal_pos'])
            if gp is not None and len(gp) >= 3:
                goal_pos = gp[:3].astype(float)
        if 'is_grasped' in extra:
            val = _to_numpy(extra['is_grasped'])
            if val is not None:
                is_grasped = bool(val)

    # Gripper from agent qpos/qvel
    # Multi-robot obs nests agent data per-robot; use first robot found
    agent = obs.get('agent', {})
    if isinstance(agent, dict):
        # Check for nested per-robot dicts (e.g. agent['panda_wristcam-0'])
        first_val = next(iter(agent.values()), None) if agent else None
        if isinstance(first_val, dict):
            agent = first_val  # use first robot's data
        if 'qpos' in agent:
            qpos = _to_numpy(agent['qpos'])
            if qpos is not None and len(qpos) >= 2:
                gripper_state = float(np.clip(np.mean(qpos[-2:]) / 0.04, 0, 1))
        if 'qvel' in agent:
            qvel = _to_numpy(agent['qvel'])
            if qvel is not None and len(qvel) >= 2:
                gripper_vel = float(np.mean(qvel[-2:]))

    # Object position from environment internals
    if env is not None:
        unwrapped = getattr(env, 'unwrapped', env)
        for attr in ('cube', 'obj', 'cubeA', 'object'):
            obj = getattr(unwrapped, attr, None)
            if obj is not None and hasattr(obj, 'pose') and hasattr(obj.pose, 'p'):
                p = _to_numpy(obj.pose.p)
                if p is not None and len(p) >= 3:
                    object_pos = p[:3].astype(float)
                    break
        # Goal position from env if not in obs (multi-robot uses goal_site)
        if np.allclose(goal_pos, [0.0, 0.0, 0.15]):
            for gattr in ('goal_site', 'goal'):
                gs = getattr(unwrapped, gattr, None)
                if gs is not None and hasattr(gs, 'pose') and hasattr(gs.pose, 'p'):
                    gp = _to_numpy(gs.pose.p)
                    if gp is not None and len(gp) >= 3:
                        goal_pos = gp[:3].astype(float)
                        break

    dist_to_object = float(np.linalg.norm(tcp_pos - object_pos))
    dist_object_to_goal = float(np.linalg.norm(object_pos - goal_pos))

    return TaskState(
        tcp_pos=tcp_pos,
        object_pos=object_pos,
        goal_pos=goal_pos,
        gripper_state=gripper_state,
        gripper_vel=gripper_vel,
        dist_to_object=dist_to_object,
        dist_object_to_goal=dist_object_to_goal,
        object_height=float(object_pos[2]),
        is_grasped=is_grasped,
    )


def _default_state() -> TaskState:
    return TaskState(
        tcp_pos=np.zeros(3), object_pos=np.zeros(3),
        goal_pos=np.array([0.0, 0.0, 0.15]),
        gripper_state=1.0, gripper_vel=0.0,
        dist_to_object=0.2, dist_object_to_goal=0.2,
        object_height=0.02, is_grasped=False,
    )


# =============================================================================
# Partner Action Computation
# =============================================================================

def compute_partner_action(
    strategy: PartnerStrategy,
    state: TaskState,
    action_dim: int,
) -> np.ndarray:
    """
    Compute a strategy-dependent action in end-effector delta-pose space.

    Action layout (pd_ee_delta_pose, typically 7D):
        [0:3] XYZ delta position
        [3:6] rotation delta (zeroed for stability)
        [6]   gripper (-1=close, 1=open)
    """
    speed, noise, bias = STRATEGY_PARAMS[strategy]
    action = np.zeros(action_dim, dtype=np.float32)
    gripper_dim = min(6, action_dim - 1)

    dir_to_obj = state.object_pos - state.tcp_pos
    dist = np.linalg.norm(dir_to_obj)
    dir_norm = dir_to_obj / max(dist, 1e-4)

    dir_to_goal = state.goal_pos - state.object_pos
    dist_goal = np.linalg.norm(dir_to_goal)
    dir_goal_norm = dir_to_goal / max(dist_goal, 1e-4)

    gripper_closed = state.gripper_state < 0.3
    grasped = state.is_grasped and gripper_closed
    lifted = state.object_height > 0.05
    height_diff = state.tcp_pos[2] - state.object_pos[2]
    xy_dist = np.linalg.norm(state.tcp_pos[:2] - state.object_pos[:2])

    if grasped and lifted:
        # TRANSPORT: move object toward goal
        action[:3] = dir_goal_norm[:3] * speed * 0.12 + bias
        action[gripper_dim] = -1.0
        if dist_goal < 0.05:
            action[gripper_dim] = 1.0  # release at goal
    elif grasped or (gripper_closed and height_diff < 0.04 and xy_dist < 0.04):
        # LIFT
        action[2] = 0.3 * speed
        action[gripper_dim] = -1.0
    elif height_diff < 0.04 and xy_dist < 0.04:
        # GRASP: at object, close gripper
        action[gripper_dim] = -1.0
        action[2] = 0.1
        action[:2] = dir_norm[:2] * 0.01
    elif state.gripper_state < 0.8 and height_diff < 0.06:
        # Keep closing once started
        action[gripper_dim] = -1.0
        action[2] = -0.05
        action[:2] = dir_norm[:2] * 0.05
    elif xy_dist < 0.04 and height_diff > 0.04:
        # DESCEND over object
        action[2] = -0.2 * speed
        action[gripper_dim] = 1.0
        action[:2] = dir_norm[:2] * 0.02
    elif dist < 0.12:
        # REACH: fine approach
        action[:2] = dir_norm[:2] * speed * 0.12
        action[2] = -0.1 * speed
        action[gripper_dim] = 1.0
    else:
        # APPROACH: move toward object
        action[:3] = dir_norm[:3] * speed * 0.25 + bias
        action[gripper_dim] = 1.0 if dist > 0.1 else -1.0

    # Add noise and clip
    action[:3] += np.random.randn(3) * noise
    if action_dim > 4:
        action[3:gripper_dim] = 0.0  # zero rotation
    return np.clip(action, -1.0, 1.0)


def discretize_action(action: np.ndarray) -> int:
    """
    Map continuous EE action to discrete index for PartnerDataset compatibility.

    6 discrete actions based on dominant movement direction:
        0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z (lift), 5: -Z (lower)
    """
    xyz = action[:3]
    axis = int(np.argmax(np.abs(xyz)))
    sign = 1 if xyz[axis] >= 0 else 0
    return axis * 2 + (1 - sign)


# =============================================================================
# Image Extraction (ManiSkill mode)
# =============================================================================

def extract_image(obs: Dict, env, image_size: int = 128) -> Optional[np.ndarray]:
    """Extract RGB image [H, W, 3] uint8 from observation."""
    if isinstance(obs, dict):
        if 'sensor_data' in obs:
            for cam_data in obs['sensor_data'].values():
                if isinstance(cam_data, dict) and 'rgb' in cam_data:
                    rgb = cam_data['rgb']
                    if hasattr(rgb, 'cpu'):
                        rgb = rgb.cpu().numpy()
                    rgb = np.asarray(rgb)
                    if rgb.ndim == 4:
                        rgb = rgb[0]
                    return rgb.astype(np.uint8)
        if 'image' in obs:
            img = obs['image']
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            return np.asarray(img).astype(np.uint8)
    try:
        img = env.render()
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()
        if img is not None:
            return np.asarray(img).astype(np.uint8)
    except Exception:
        pass
    return None


# =============================================================================
# Synthetic Task Simulation (no GPU required)
# =============================================================================

# Per-task configs: (object_init_pos, goal_pos, task_type)
# task_type: 'pick' = pick-and-place, 'push' = push, 'insert' = peg insertion
SYNTHETIC_TASK_CONFIGS = {
    'PickCube-v1':       (np.array([0.0, 0.3, 0.02]), np.array([0.15, 0.0, 0.15]), 'pick'),
    'StackCube-v1':      (np.array([0.0, 0.3, 0.02]), np.array([0.1, 0.3, 0.06]), 'pick'),
    'PegInsertionSide-v1': (np.array([0.0, 0.2, 0.05]), np.array([0.2, 0.2, 0.05]), 'insert'),
    'PushCube-v1':       (np.array([0.0, 0.3, 0.02]), np.array([0.2, 0.3, 0.02]), 'push'),
    'PullCube-v1':       (np.array([0.2, 0.3, 0.02]), np.array([0.0, 0.3, 0.02]), 'push'),
    'LiftPegUpright-v1': (np.array([0.0, 0.3, 0.02]), np.array([0.0, 0.3, 0.15]), 'pick'),
    'PickSingleYCB-v1':  (np.array([0.05, 0.25, 0.03]), np.array([0.2, 0.0, 0.15]), 'pick'),
    'PlugCharger-v1':    (np.array([0.0, 0.2, 0.03]), np.array([0.15, 0.2, 0.03]), 'insert'),
    'PokeCube-v1':       (np.array([0.0, 0.3, 0.02]), np.array([0.15, 0.3, 0.02]), 'push'),
    'PlaceSphere-v1':    (np.array([0.0, 0.3, 0.02]), np.array([0.15, 0.15, 0.02]), 'pick'),
    'TurnFaucet-v1':     (np.array([0.0, 0.3, 0.1]), np.array([0.0, 0.3, 0.1]), 'insert'),
    'PushT-v1':          (np.array([0.0, 0.3, 0.02]), np.array([0.2, 0.1, 0.02]), 'push'),
    'TwoRobotPickCube-v1':  (np.array([0.0, 0.3, 0.02]), np.array([0.15, 0.0, 0.15]), 'pick'),
    'TwoRobotStackCube-v1': (np.array([0.0, 0.3, 0.02]), np.array([0.1, 0.3, 0.06]), 'pick'),
}
# TriFinger tasks
for i in range(5):
    SYNTHETIC_TASK_CONFIGS[f'TriFingerRotateCubeLevel{i}-v1'] = (
        np.array([0.0, 0.0, 0.04]), np.array([0.0, 0.0, 0.04]), 'insert'
    )


class SyntheticTaskSim:
    """
    Lightweight physics simulation of pick/push/insert manipulation tasks.
    Produces the same 15D TaskState as ManiSkill extraction.
    """

    def __init__(self, task: str, episode_length: int = 100):
        cfg = SYNTHETIC_TASK_CONFIGS.get(task)
        if cfg is None:
            # Fallback: generic pick task
            cfg = (np.array([0.0, 0.3, 0.02]), np.array([0.15, 0.0, 0.15]), 'pick')
        self.obj_init, self.goal_pos, self.task_type = cfg
        self.episode_length = episode_length
        self.reset()

    def reset(self):
        # Randomize initial positions slightly
        self.tcp_pos = np.array([0.0, 0.0, 0.25]) + np.random.randn(3) * 0.02
        self.obj_pos = self.obj_init.copy() + np.random.randn(3) * 0.01
        self.obj_pos[2] = max(self.obj_pos[2], 0.02)  # keep on table
        self.gripper_state = 1.0   # open
        self.gripper_vel = 0.0
        self.is_grasped = False
        self.t = 0

    def get_state(self) -> TaskState:
        dist_to_obj = float(np.linalg.norm(self.tcp_pos - self.obj_pos))
        dist_obj_goal = float(np.linalg.norm(self.obj_pos - self.goal_pos))
        return TaskState(
            tcp_pos=self.tcp_pos.copy(),
            object_pos=self.obj_pos.copy(),
            goal_pos=self.goal_pos.copy(),
            gripper_state=self.gripper_state,
            gripper_vel=self.gripper_vel,
            dist_to_object=dist_to_obj,
            dist_object_to_goal=dist_obj_goal,
            object_height=float(self.obj_pos[2]),
            is_grasped=self.is_grasped,
        )

    def step(self, action: np.ndarray):
        """Apply action, update state with simple physics."""
        dt = 0.05
        # TCP moves according to action XYZ
        delta = action[:3] * dt * 2.0
        self.tcp_pos = self.tcp_pos + delta
        # Clamp workspace
        self.tcp_pos = np.clip(self.tcp_pos, [-0.5, -0.5, 0.0], [0.5, 0.5, 0.5])

        # Gripper
        gripper_cmd = action[min(6, len(action) - 1)] if len(action) > 3 else 0.0
        old_grip = self.gripper_state
        if gripper_cmd < 0:
            self.gripper_state = max(0.0, self.gripper_state - 0.15)
        else:
            self.gripper_state = min(1.0, self.gripper_state + 0.15)
        self.gripper_vel = self.gripper_state - old_grip

        # Grasp logic
        dist = np.linalg.norm(self.tcp_pos - self.obj_pos)
        if dist < 0.04 and self.gripper_state < 0.3:
            self.is_grasped = True
        if self.gripper_state > 0.6:
            self.is_grasped = False

        # Object follows TCP if grasped
        if self.is_grasped:
            self.obj_pos = self.tcp_pos.copy()
            self.obj_pos[2] = max(self.obj_pos[2], 0.02)
        elif self.task_type == 'push' and dist < 0.05:
            # Push: object slides when TCP is close
            push_dir = self.obj_pos - self.tcp_pos
            push_dir[2] = 0  # keep on table
            push_norm = np.linalg.norm(push_dir)
            if push_norm > 1e-4:
                self.obj_pos += (push_dir / push_norm) * 0.02
            self.obj_pos[2] = max(self.obj_pos[2], 0.02)
        else:
            # Gravity: object falls if not grasped and above table
            if self.obj_pos[2] > 0.02:
                self.obj_pos[2] = max(0.02, self.obj_pos[2] - 0.02)

        # Add small process noise for realism
        self.obj_pos += np.random.randn(3) * 0.001
        self.obj_pos[2] = max(self.obj_pos[2], 0.02)

        self.t += 1
        terminated = False
        # Success check
        if np.linalg.norm(self.obj_pos - self.goal_pos) < 0.03:
            terminated = True
        return terminated


def collect_synthetic_episode(
    sim: SyntheticTaskSim,
    strategy: PartnerStrategy,
    episode_length: int,
    action_dim: int = 7,
    enable_switch: bool = True,
    switch_range: Tuple[int, int] = (20, 60),
    collect_images: bool = False,
    image_size: int = 128,
) -> Dict[str, np.ndarray]:
    """Collect one episode from synthetic sim."""
    switch_time = np.random.randint(*switch_range) if enable_switch else -1
    current_strategy = strategy
    sim.reset()

    states_buf = []
    images_buf = []
    actions_buf = []
    types_buf = []

    for t in range(episode_length):
        if enable_switch and t == switch_time:
            others = [s for s in PartnerStrategy if s != current_strategy]
            current_strategy = PartnerStrategy(np.random.choice([s.value for s in others]))

        state = sim.get_state()
        cont_action = compute_partner_action(current_strategy, state, action_dim)

        states_buf.append(state.to_array())
        actions_buf.append(discretize_action(cont_action))
        types_buf.append(current_strategy.value)

        if collect_images:
            images_buf.append(_render_synthetic(state, current_strategy, image_size))

        terminated = sim.step(cont_action)
        if terminated:
            for _ in range(t + 1, episode_length):
                states_buf.append(states_buf[-1].copy())
                actions_buf.append(actions_buf[-1])
                types_buf.append(types_buf[-1])
                if collect_images:
                    images_buf.append(images_buf[-1].copy())
            break

    result = {
        'states': np.stack(states_buf),
        'partner_actions': np.array(actions_buf),
        'partner_types': np.array(types_buf),
        'switch_time': switch_time,
    }
    if collect_images and images_buf:
        result['images'] = np.stack(images_buf)
    return result


def _render_synthetic(state: TaskState, strategy: PartnerStrategy, size: int = 128) -> np.ndarray:
    """Render a simple top-down visualization of the task state."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 200  # grey background

    def _world_to_px(pos, center=0.0, scale=200):
        px = int(size / 2 + (pos[0] - center) * scale)
        py = int(size / 2 - (pos[1] - 0.15) * scale)
        return np.clip(px, 2, size - 3), np.clip(py, 2, size - 3)

    # Draw goal (green square)
    gx, gy = _world_to_px(state.goal_pos)
    img[max(0,gy-4):gy+5, max(0,gx-4):gx+5] = [0, 180, 0]

    # Draw object (red square, brighter if lifted)
    ox, oy = _world_to_px(state.object_pos)
    red = 255 if state.object_height > 0.05 else 180
    img[max(0,oy-3):oy+4, max(0,ox-3):ox+4] = [red, 50, 50]

    # Draw TCP (blue dot)
    tx, ty = _world_to_px(state.tcp_pos)
    img[max(0,ty-2):ty+3, max(0,tx-2):tx+3] = [50, 50, 220]

    # Strategy indicator (colored bar at top)
    colors = {
        PartnerStrategy.FAST: [255, 100, 100],
        PartnerStrategy.SLOW: [100, 100, 255],
        PartnerStrategy.LEFT_PREFER: [100, 255, 100],
        PartnerStrategy.RIGHT_PREFER: [255, 255, 100],
        PartnerStrategy.AGGRESSIVE: [255, 100, 255],
        PartnerStrategy.CAUTIOUS: [100, 255, 255],
    }
    img[:4, :] = colors.get(strategy, [150, 150, 150])

    # Add sensor-like noise
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# =============================================================================
# ManiSkill Episode Collection
# =============================================================================

def collect_maniskill_episode(
    env,
    strategy: PartnerStrategy,
    episode_length: int,
    action_dim: int,
    enable_switch: bool = True,
    switch_range: Tuple[int, int] = (20, 60),
    collect_images: bool = True,
    image_size: int = 128,
    is_multi_robot: bool = False,
) -> Dict[str, np.ndarray]:
    """Collect one episode from ManiSkill env."""
    switch_time = np.random.randint(*switch_range) if enable_switch else -1
    current_strategy = strategy
    obs, _ = env.reset()
    # Cache action space keys for multi-robot envs
    if is_multi_robot:
        _robot_keys = list(env.action_space.spaces.keys())

    states_buf = []
    images_buf = []
    actions_buf = []
    types_buf = []

    for t in range(episode_length):
        if enable_switch and t == switch_time:
            others = [s for s in PartnerStrategy if s != current_strategy]
            current_strategy = PartnerStrategy(np.random.choice([s.value for s in others]))

        state = extract_state(obs, env)
        cont_action = compute_partner_action(current_strategy, state, action_dim)

        states_buf.append(state.to_array())
        actions_buf.append(discretize_action(cont_action))
        types_buf.append(current_strategy.value)

        if collect_images:
            img = extract_image(obs, env, image_size)
            if img is not None:
                images_buf.append(img)
            else:
                images_buf.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))

        if is_multi_robot:
            # First robot executes partner action, second robot idles
            step_action = {_robot_keys[0]: cont_action,
                           _robot_keys[1]: np.zeros(action_dim, dtype=np.float32)}
        else:
            step_action = cont_action
        obs, reward, terminated, truncated, info = env.step(step_action)
        if terminated or truncated:
            for _ in range(t + 1, episode_length):
                states_buf.append(states_buf[-1].copy())
                actions_buf.append(actions_buf[-1])
                types_buf.append(types_buf[-1])
                if collect_images:
                    images_buf.append(images_buf[-1].copy())
            break

    result = {
        'states': np.stack(states_buf),
        'partner_actions': np.array(actions_buf),
        'partner_types': np.array(types_buf),
        'switch_time': switch_time,
    }
    if collect_images and images_buf:
        result['images'] = np.stack(images_buf)
    return result


# =============================================================================
# Main Collection Loop
# =============================================================================

def collect_dataset(
    task: str,
    episodes_per_strategy: int = 200,
    episode_length: int = 100,
    output_dir: str = './data',
    collect_images: bool = True,
    image_size: int = 128,
    enable_switching: bool = True,
    synthetic: bool = False,
) -> Path:
    """Collect full dataset for one task and save as .npz."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    action_dim = 7  # default EE delta pose

    is_multi_robot = task in MULTI_ROBOT_TASKS

    if synthetic:
        print(f"\nSynthetic mode: {task}")
        sim = SyntheticTaskSim(task, episode_length)
    else:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        control_mode = TASK_REGISTRY.get(task, 'pd_ee_delta_pose')
        obs_mode = 'rgbd' if collect_images else 'state'
        print(f"\nInitializing {task} (control={control_mode}, obs={obs_mode})")
        env = gym.make(
            task, obs_mode=obs_mode, render_mode='rgb_array',
            control_mode=control_mode,
        )
        # Multi-robot envs have Dict action space; extract single-robot dim
        if is_multi_robot:
            first_key = list(env.action_space.spaces.keys())[0]
            action_dim = env.action_space[first_key].shape[0]
        else:
            action_dim = env.action_space.shape[0]
        print(f"  Action dim: {action_dim}{' (multi-robot)' if is_multi_robot else ''}")

    total = episodes_per_strategy * NUM_STRATEGIES
    print(f"  Collecting {total} episodes ({episodes_per_strategy} x {NUM_STRATEGIES} strategies)")

    all_states, all_images, all_actions, all_types, all_switches = [], [], [], [], []
    start = time.time()
    count = 0

    for strategy in PartnerStrategy:
        print(f"  Strategy: {strategy.name}")
        for ep in range(episodes_per_strategy):
            if synthetic:
                data = collect_synthetic_episode(
                    sim, strategy, episode_length, action_dim,
                    enable_switch=enable_switching,
                    collect_images=collect_images, image_size=image_size,
                )
            else:
                data = collect_maniskill_episode(
                    env, strategy, episode_length, action_dim,
                    enable_switch=enable_switching,
                    collect_images=collect_images, image_size=image_size,
                    is_multi_robot=is_multi_robot,
                )
            all_states.append(data['states'])
            all_actions.append(data['partner_actions'])
            all_types.append(data['partner_types'])
            all_switches.append(data['switch_time'])
            if collect_images and 'images' in data:
                all_images.append(data['images'])

            count += 1
            if (ep + 1) % 50 == 0:
                elapsed = time.time() - start
                rate = count / elapsed
                remaining = (total - count) / max(rate, 1e-6)
                print(f"    {ep+1}/{episodes_per_strategy} "
                      f"({rate:.1f} ep/s, ~{remaining/60:.1f} min left)")

    if not synthetic:
        env.close()

    # Assemble arrays
    result = {
        'observations': np.stack(all_states),        # [N, T, 15]
        'partner_actions': np.stack(all_actions),     # [N, T]
        'partner_types': np.stack(all_types),         # [N, T]
        'switch_times': np.array(all_switches),       # [N]
    }
    if collect_images and all_images:
        result['images'] = np.stack(all_images)       # [N, T, H, W, 3]

    elapsed = time.time() - start
    task_slug = task.replace('-', '_').lower()
    filepath = output_path / f'{task_slug}_data.npz'

    print(f"\nSaving to {filepath}")
    for k, v in result.items():
        print(f"  {k}: {v.shape} ({v.dtype})")
    np.savez_compressed(filepath, **result)

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Time: {elapsed/60:.1f} min ({count/elapsed:.1f} ep/s)")

    return filepath


# =============================================================================
# CLI
# =============================================================================

def list_tasks():
    """Print available tasks."""
    print("\nRegistered tasks (usable with --synthetic or ManiSkill):")
    for category, tasks in TASK_CATEGORIES.items():
        print(f"\n  {category}:")
        for t in tasks:
            ctrl = TASK_REGISTRY.get(t, '?')
            syn = 'Y' if t in SYNTHETIC_TASK_CONFIGS else ' '
            print(f"    [{syn}] {t:<40s} ({ctrl})")
    print("\n  [Y] = has synthetic physics config")

    # Try to discover ManiSkill envs
    try:
        import mani_skill.envs
        from gymnasium.envs.registration import registry
        ms_envs = sorted(k for k in registry if 'v1' in k and
                         any(pkg in str(registry[k].entry_point)
                             for pkg in ['mani_skill']))
        if ms_envs:
            print(f"\n  All ManiSkill envs ({len(ms_envs)}):")
            for e in ms_envs:
                marker = '*' if e in TASK_REGISTRY else ' '
                print(f"    {marker} {e}")
    except Exception as e:
        print(f"\n  (Could not enumerate ManiSkill envs: {e})")


def test_setup(synthetic: bool = False):
    """Quick test: run 10 steps, collect a short episode, verify shapes."""
    task = 'PickCube-v1'

    if synthetic:
        print(f"Testing synthetic sim for {task}...")
        sim = SyntheticTaskSim(task, episode_length=50)
        action_dim = 7

        for t in range(10):
            state = sim.get_state()
            action = compute_partner_action(PartnerStrategy.FAST, state, action_dim)
            terminated = sim.step(action)
            print(f"  t={t}: tcp=[{state.tcp_pos[0]:.3f},{state.tcp_pos[1]:.3f},{state.tcp_pos[2]:.3f}], "
                  f"obj=[{state.object_pos[0]:.3f},{state.object_pos[1]:.3f},{state.object_pos[2]:.3f}], "
                  f"dist={state.dist_to_object:.3f}, grip={state.gripper_state:.2f}, "
                  f"grasped={state.is_grasped}")
            if terminated:
                print("  -> task completed!")
                break

        print("\nCollecting test episode...")
        sim2 = SyntheticTaskSim(task, episode_length=50)
        ep = collect_synthetic_episode(
            sim2, PartnerStrategy.FAST, episode_length=50,
            action_dim=7, collect_images=False,
        )
    else:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        print(f"Testing ManiSkill {task}...")
        env = gym.make(task, obs_mode='state', render_mode='rgb_array',
                       control_mode='pd_ee_delta_pose')
        action_dim = env.action_space.shape[0]
        print(f"  Action space: {env.action_space}")

        obs, info = env.reset()
        for t in range(10):
            state = extract_state(obs, env)
            action = compute_partner_action(PartnerStrategy.FAST, state, action_dim)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  t={t}: tcp={state.tcp_pos}, obj={state.object_pos}, "
                  f"dist={state.dist_to_object:.3f}, grip={state.gripper_state:.2f}, "
                  f"grasped={state.is_grasped}")
        env.close()

        print("\nCollecting test episode...")
        env = gym.make(task, obs_mode='state', render_mode='rgb_array',
                       control_mode='pd_ee_delta_pose')
        ep = collect_maniskill_episode(
            env, PartnerStrategy.FAST, episode_length=50,
            action_dim=env.action_space.shape[0], collect_images=False,
        )
        env.close()

    print(f"  states: {ep['states'].shape}")
    print(f"  partner_actions: {ep['partner_actions'].shape}, "
          f"unique={np.unique(ep['partner_actions'])}")
    print(f"  partner_types: {ep['partner_types'].shape}, "
          f"unique={np.unique(ep['partner_types'])}")
    print(f"  switch_time: {ep['switch_time']}")
    print("\nTest passed.")


def main():
    parser = argparse.ArgumentParser(
        description='Collect ManiSkill3 datasets for UA-ToM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--test', action='store_true',
                        help='Test setup (combine with --synthetic for no-GPU test)')
    parser.add_argument('--list', action='store_true',
                        help='List available tasks')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic physics sim (no GPU/ManiSkill required)')

    # Task selection
    parser.add_argument('--task', type=str, default=None,
                        help='Single task to collect (e.g. PickCube-v1)')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='Multiple tasks to collect')
    parser.add_argument('--category', type=str, default=None,
                        choices=list(TASK_CATEGORIES.keys()),
                        help='Collect all tasks in a category')

    # Collection settings
    parser.add_argument('--episodes', type=int, default=200,
                        help='Episodes per strategy (default: 200)')
    parser.add_argument('--length', type=int, default=100,
                        help='Episode length in timesteps (default: 100)')
    parser.add_argument('--no-images', action='store_true',
                        help='Skip image collection (state-only)')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Image resolution (default: 128)')
    parser.add_argument('--no-switch', action='store_true',
                        help='Disable mid-episode partner switching')

    # Output
    parser.add_argument('--output', type=str, default='./data',
                        help='Output directory (default: ./data)')

    args = parser.parse_args()

    if args.list:
        list_tasks()
        return

    if args.test:
        test_setup(synthetic=args.synthetic)
        return

    # Determine tasks to collect
    tasks = []
    if args.task:
        tasks = [args.task]
    elif args.tasks:
        tasks = args.tasks
    elif args.category:
        tasks = TASK_CATEGORIES[args.category]
    else:
        parser.print_help()
        print("\nSpecify --task, --tasks, --category, --test, or --list.")
        return

    print("=" * 60)
    print("UA-ToM ManiSkill Data Collection")
    print("=" * 60)
    print(f"Tasks: {tasks}")
    print(f"Episodes/strategy: {args.episodes}")
    print(f"Episode length: {args.length}")
    print(f"Images: {not args.no_images}")
    print(f"Switching: {not args.no_switch}")
    print(f"Synthetic: {args.synthetic}")
    print(f"Output: {args.output}")
    total_episodes = args.episodes * NUM_STRATEGIES * len(tasks)
    print(f"Total episodes: {total_episodes}")

    saved = []
    for task in tasks:
        filepath = collect_dataset(
            task=task,
            episodes_per_strategy=args.episodes,
            episode_length=args.length,
            output_dir=args.output,
            collect_images=not args.no_images,
            image_size=args.image_size,
            enable_switching=not args.no_switch,
            synthetic=args.synthetic,
        )
        saved.append(filepath)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    for p in saved:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
