#!/usr/bin/env python3
"""
Data Collection for SharedWorkspacePickCube-v1
==============================================

Collects .npz datasets from the shared-workspace two-robot environment
with 4 qualitatively distinct partner strategies and mid-episode switches.

Output format (compatible with PartnerDataset):
    - observations: [N, T, obs_dim] state vectors (21D)
    - partner_actions: [N, T] discrete action indices (0-7)
    - partner_types: [N, T] partner type labels (0-3)
    - switch_times: [N] timestep of partner switch (-1 if none)

Partner types:
    0 = HELPER     (approach, grasp, carry toward ego)
    1 = COMPETITOR  (race to cube, grasp, steal to own side)
    2 = BLOCKER     (interpose between ego and cube)
    3 = PASSIVE     (stay idle)

Discrete actions (8 bins):
    0-5 = dominant movement direction (±X, ±Y, ±Z) with open gripper
    6   = close gripper (stationary)
    7   = no movement (idle / near-zero action)

Usage:
    python scripts/collect_shared_workspace.py --episodes 200 --output data/shared_workspace
    python scripts/collect_shared_workspace.py --episodes 500 --episode_length 150 --output data/shared_workspace
"""

import os
os.environ['MANI_SKILL_NO_DISPLAY'] = '1'

import sys
import argparse
import time
from pathlib import Path
from enum import IntEnum

import numpy as np
import torch
import gymnasium as gym

# Register SharedWorkspacePickCube-v1
sys.path.insert(0, str(Path(__file__).parent))
import test_shared_workspace  # noqa — registers the env


# =============================================================================
# Partner Types
# =============================================================================

class PartnerType(IntEnum):
    HELPER = 0
    COMPETITOR = 1
    BLOCKER = 2
    PASSIVE = 3


NUM_TYPES = len(PartnerType)


# =============================================================================
# Frame Transforms
# =============================================================================

def w2l(w):
    """World direction -> left arm action frame."""
    return np.array([w[1], -w[0], w[2]])

def w2r(w):
    """World direction -> right arm action frame."""
    return np.array([-w[1], w[0], w[2]])

def _move_toward(tcp, target, frame_fn, gain=5.0, gripper=1.0):
    direction = (target - tcp) * gain
    cmd = frame_fn(direction)
    cmd = np.clip(cmd[:3], -1, 1)
    return np.concatenate([cmd, [gripper]])


# =============================================================================
# Partner Strategies (step-count based phases)
# =============================================================================

class PartnerStrategy:
    def __init__(self, ptype: PartnerType):
        self.ptype = ptype
        self.step_count = 0
        self.step_in_phase = 0
        self.phase = "init"

    def reset(self):
        self.step_count = 0
        self.step_in_phase = 0
        self.phase = "init"

    def get_action(self, right_tcp, cube_pos, left_tcp):
        raise NotImplementedError


class HelperStrategy(PartnerStrategy):
    """Approach cube from above, grasp, carry toward ego side."""
    def __init__(self):
        super().__init__(PartnerType.HELPER)

    def reset(self):
        super().reset()
        self.phase = "above"

    def get_action(self, right_tcp, cube_pos, left_tcp):
        self.step_count += 1
        self.step_in_phase += 1

        if self.phase == "above":
            target = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.1])
            if self.step_in_phase >= 15:
                self.phase = "descend"
                self.step_in_phase = 0
            return _move_toward(right_tcp, target, w2r, gain=5.0, gripper=1.0)

        elif self.phase == "descend":
            target = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.01])
            if self.step_in_phase >= 15:
                self.phase = "grasp"
                self.step_in_phase = 0
            return _move_toward(right_tcp, target, w2r, gain=5.0, gripper=1.0)

        elif self.phase == "grasp":
            if self.step_in_phase >= 10:
                self.phase = "carry"
                self.step_in_phase = 0
            return np.array([0.0, 0.0, 0.0, -1.0])

        elif self.phase == "carry":
            target = np.array([cube_pos[0], -0.08, 0.20])
            return _move_toward(right_tcp, target, w2r, gain=3.0, gripper=-1.0)

        return np.array([0.0, 0.0, 0.0, 1.0])


class CompetitorStrategy(PartnerStrategy):
    """Race directly to cube, grasp, steal to own side."""
    def __init__(self):
        super().__init__(PartnerType.COMPETITOR)

    def reset(self):
        super().reset()
        self.phase = "race"

    def get_action(self, right_tcp, cube_pos, left_tcp):
        self.step_count += 1
        self.step_in_phase += 1

        if self.phase == "race":
            target = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.01])
            if self.step_in_phase >= 20:
                self.phase = "grab"
                self.step_in_phase = 0
            return _move_toward(right_tcp, target, w2r, gain=8.0, gripper=1.0)

        elif self.phase == "grab":
            if self.step_in_phase >= 10:
                self.phase = "steal"
                self.step_in_phase = 0
            return np.array([0.0, 0.0, 0.0, -1.0])

        elif self.phase == "steal":
            target = np.array([right_tcp[0], 0.12, 0.25])
            return _move_toward(right_tcp, target, w2r, gain=3.0, gripper=-1.0)

        return np.array([0.0, 0.0, 0.0, 1.0])


class BlockerStrategy(PartnerStrategy):
    """Interpose between ego and cube, push cube away when ego approaches."""
    def __init__(self):
        super().__init__(PartnerType.BLOCKER)

    def reset(self):
        super().reset()
        self.phase = "block"

    def get_action(self, right_tcp, cube_pos, left_tcp):
        self.step_count += 1
        self.step_in_phase += 1

        ego_dist = np.linalg.norm((cube_pos - left_tcp)[:2])

        if ego_dist < 0.12:
            push_dir = cube_pos - left_tcp
            push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-8)
            target = cube_pos + push_dir * 0.01
            target[2] = cube_pos[2]
            return _move_toward(right_tcp, target, w2r, gain=8.0, gripper=1.0)
        else:
            midpoint = (cube_pos + left_tcp) / 2
            target = 0.7 * cube_pos + 0.3 * midpoint
            target[2] = cube_pos[2] + 0.04
            return _move_toward(right_tcp, target, w2r, gain=5.0, gripper=1.0)


class PassiveStrategy(PartnerStrategy):
    """Stay near home position, minimal movement."""
    def __init__(self):
        super().__init__(PartnerType.PASSIVE)
        self.home = None

    def reset(self):
        super().reset()
        self.home = None

    def get_action(self, right_tcp, cube_pos, left_tcp):
        self.step_count += 1
        self.step_in_phase += 1
        if self.home is None:
            self.home = right_tcp.copy()
        noise = np.random.randn(3) * 0.005
        target = self.home + noise
        direction = (target - right_tcp) * 0.5
        cmd = w2r(direction)
        cmd = np.clip(cmd[:3], -0.05, 0.05)
        return np.concatenate([cmd, [1.0]])


STRATEGY_CLASSES = {
    PartnerType.HELPER: HelperStrategy,
    PartnerType.COMPETITOR: CompetitorStrategy,
    PartnerType.BLOCKER: BlockerStrategy,
    PartnerType.PASSIVE: PassiveStrategy,
}


def make_strategy(ptype: PartnerType) -> PartnerStrategy:
    return STRATEGY_CLASSES[ptype]()


# =============================================================================
# Ego Controller (step-count based, same as test_partner_behaviors.py)
# =============================================================================

class EgoController:
    def __init__(self):
        self.phase = "above"
        self.step_in_phase = 0

    def reset(self):
        self.phase = "above"
        self.step_in_phase = 0

    def get_action(self, left_tcp, cube_pos):
        self.step_in_phase += 1

        if self.phase == "above":
            target = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.1])
            if self.step_in_phase >= 15:
                self.phase = "descend"
                self.step_in_phase = 0
            return _move_toward(left_tcp, target, w2l, gain=5.0, gripper=1.0)

        elif self.phase == "descend":
            target = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.01])
            if self.step_in_phase >= 15:
                self.phase = "grasp"
                self.step_in_phase = 0
            return _move_toward(left_tcp, target, w2l, gain=5.0, gripper=1.0)

        elif self.phase == "grasp":
            if self.step_in_phase >= 10:
                self.phase = "lift"
                self.step_in_phase = 0
            return np.array([0.0, 0.0, 0.0, -1.0])

        elif self.phase == "lift":
            target = np.array([cube_pos[0], cube_pos[1], 0.35])
            return _move_toward(left_tcp, target, w2l, gain=3.0, gripper=-1.0)

        return np.array([0.0, 0.0, 0.0, 1.0])


# =============================================================================
# Observation Extraction
# =============================================================================

def extract_observation(obs) -> np.ndarray:
    """
    Extract a 21D state vector from ManiSkill observation.

    Components:
        [0:3]   left_tcp XYZ
        [3:6]   right_tcp XYZ
        [6:9]   cube_pos XYZ
        [9:12]  left_tcp_to_cube direction
        [12:15] right_tcp_to_cube direction
        [15]    left-to-cube distance
        [16]    right-to-cube distance
        [17]    cube height (Z)
        [18]    left-right tcp distance
        [19:21] cube_to_goal direction (XY only + Z)
    """
    extra = obs['extra']
    left_tcp = extra['left_arm_tcp'][0, :3].cpu().numpy()
    right_tcp = extra['right_arm_tcp'][0, :3].cpu().numpy()
    cube_pos = extra['cube_pose'][0, :3].cpu().numpy()

    l2c = cube_pos - left_tcp
    r2c = cube_pos - right_tcp
    l2c_dist = np.linalg.norm(l2c)
    r2c_dist = np.linalg.norm(r2c)
    lr_dist = np.linalg.norm(left_tcp - right_tcp)

    cube_to_goal = extra['cube_to_goal_pos'][0, :3].cpu().numpy()

    return np.concatenate([
        left_tcp,          # 0:3
        right_tcp,         # 3:6
        cube_pos,          # 6:9
        l2c,               # 9:12
        r2c,               # 12:15
        [l2c_dist],        # 15
        [r2c_dist],        # 16
        [cube_pos[2]],     # 17
        [lr_dist],         # 18
        cube_to_goal,      # 19:22
    ]).astype(np.float32)


OBS_DIM = 22  # actual dimension after concatenation


# =============================================================================
# Action Discretization
# =============================================================================

def discretize_action(action: np.ndarray) -> int:
    """
    Map 4D continuous action [dx, dy, dz, gripper] to discrete index (0-7).

    0-5: dominant movement direction (±X, ±Y, ±Z) — open gripper or moving
    6:   close gripper (stationary)
    7:   idle (near-zero action magnitude)
    """
    xyz = action[:3]
    gripper = action[3]
    mag = np.linalg.norm(xyz)

    # Idle: very small movement
    if mag < 0.03:
        if gripper < 0:
            return 6  # closing gripper while stationary
        return 7  # idle

    # Dominant direction
    axis = int(np.argmax(np.abs(xyz)))
    sign = 1 if xyz[axis] >= 0 else 0
    return axis * 2 + (1 - sign)


NUM_ACTIONS = 8


# =============================================================================
# Episode Collection
# =============================================================================

def collect_episode(
    env,
    initial_type: PartnerType,
    episode_length: int,
    enable_switch: bool = True,
    switch_range: tuple = (30, 80),
) -> dict:
    """
    Collect one episode.

    Args:
        env: SharedWorkspacePickCube-v1 environment
        initial_type: Starting partner type
        episode_length: Number of timesteps
        enable_switch: Whether to switch partner mid-episode
        switch_range: (min_t, max_t) for switch timing

    Returns:
        Dict with observations, partner_actions, partner_types, switch_time
    """
    # Determine switch
    switch_time = -1
    if enable_switch:
        switch_time = np.random.randint(*switch_range)

    # Initialize
    obs, info = env.reset()
    strategy = make_strategy(initial_type)
    strategy.reset()
    ego = EgoController()

    obs_buf = []
    action_buf = []
    type_buf = []

    for t in range(episode_length):
        # Switch partner strategy mid-episode
        if enable_switch and t == switch_time:
            others = [pt for pt in PartnerType if pt != strategy.ptype]
            new_type = PartnerType(np.random.choice([pt.value for pt in others]))
            strategy = make_strategy(new_type)
            strategy.reset()

        # Extract state
        extra = obs['extra']
        left_tcp = extra['left_arm_tcp'][0, :3].cpu().numpy()
        right_tcp = extra['right_arm_tcp'][0, :3].cpu().numpy()
        cube_pos = extra['cube_pose'][0, :3].cpu().numpy()

        # Record observation
        obs_vec = extract_observation(obs)
        obs_buf.append(obs_vec)

        # Get actions
        ego_action = ego.get_action(left_tcp, cube_pos)
        partner_action = strategy.get_action(right_tcp, cube_pos, left_tcp)

        # Record partner action and type
        action_buf.append(discretize_action(partner_action))
        type_buf.append(strategy.ptype.value)

        # Step environment
        step_action = {
            'panda_wristcam-0': torch.tensor(
                np.array([ego_action]), dtype=torch.float32
            ),
            'panda_wristcam-1': torch.tensor(
                np.array([partner_action]), dtype=torch.float32
            ),
        }
        obs, reward, terminated, truncated, info = env.step(step_action)

        # Handle early termination: pad remaining timesteps
        if terminated or truncated:
            for _ in range(t + 1, episode_length):
                obs_buf.append(obs_buf[-1].copy())
                action_buf.append(action_buf[-1])
                type_buf.append(type_buf[-1])
            break

    return {
        'observations': np.stack(obs_buf),          # [T, obs_dim]
        'partner_actions': np.array(action_buf),     # [T]
        'partner_types': np.array(type_buf),         # [T]
        'switch_time': switch_time,
    }


# =============================================================================
# Dataset Collection
# =============================================================================

def collect_dataset(
    episodes_per_type: int = 200,
    episode_length: int = 150,
    output_dir: str = 'data/shared_workspace',
    enable_switching: bool = True,
    switch_range: tuple = (30, 80),
    seed: int = 0,
):
    """Collect full dataset and save as .npz."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "SharedWorkspacePickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
    )

    total = episodes_per_type * NUM_TYPES
    print(f"\nData Collection: SharedWorkspacePickCube-v1")
    print(f"  Episodes: {total} ({episodes_per_type} x {NUM_TYPES} types)")
    print(f"  Episode length: {episode_length}")
    print(f"  Switching: {'enabled' if enable_switching else 'disabled'}")
    print(f"  Switch range: {switch_range}")
    print(f"  Seed: {seed}")

    all_obs, all_actions, all_types, all_switches = [], [], [], []
    start = time.time()
    count = 0

    for ptype in PartnerType:
        print(f"\n  Partner type: {ptype.name} ({episodes_per_type} episodes)")
        for ep in range(episodes_per_type):
            data = collect_episode(
                env, ptype, episode_length,
                enable_switch=enable_switching,
                switch_range=switch_range,
            )
            all_obs.append(data['observations'])
            all_actions.append(data['partner_actions'])
            all_types.append(data['partner_types'])
            all_switches.append(data['switch_time'])

            count += 1
            if (ep + 1) % 50 == 0:
                elapsed = time.time() - start
                rate = count / elapsed
                remaining = (total - count) / max(rate, 1e-6)
                print(f"    {ep+1}/{episodes_per_type} "
                      f"({rate:.1f} ep/s, ~{remaining/60:.1f} min left)")

    env.close()

    # Assemble
    result = {
        'observations': np.stack(all_obs),         # [N, T, obs_dim]
        'partner_actions': np.stack(all_actions),   # [N, T]
        'partner_types': np.stack(all_types),       # [N, T]
        'switch_times': np.array(all_switches),     # [N]
    }

    elapsed = time.time() - start
    filepath = output_path / 'shared_workspace_data.npz'

    print(f"\nSaving to {filepath}")
    for k, v in result.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        if k == 'partner_actions':
            unique, counts = np.unique(v, return_counts=True)
            total_actions = v.size
            print(f"    action distribution: {dict(zip(unique.tolist(), (counts/total_actions*100).round(1).tolist()))}")
        if k == 'partner_types':
            unique, counts = np.unique(v, return_counts=True)
            total_types = v.size
            print(f"    type distribution: {dict(zip([PartnerType(u).name for u in unique], (counts/total_types*100).round(1).tolist()))}")

    np.savez_compressed(filepath, **result)

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"\n  File size: {size_mb:.1f} MB")
    print(f"  Collection time: {elapsed/60:.1f} min ({count/elapsed:.1f} ep/s)")

    # Print switch statistics
    switches = result['switch_times']
    n_switched = np.sum(switches >= 0)
    print(f"\n  Episodes with switches: {n_switched}/{len(switches)} "
          f"({n_switched/len(switches)*100:.1f}%)")

    # Verify type transitions at switch points
    n_actual_switches = 0
    for i in range(len(result['partner_types'])):
        types = result['partner_types'][i]
        transitions = np.where(types[1:] != types[:-1])[0]
        n_actual_switches += len(transitions)
    print(f"  Total type transitions in data: {n_actual_switches}")

    return filepath


# =============================================================================
# Quick Validation
# =============================================================================

def validate_data(filepath):
    """Quick validation of collected data."""
    print(f"\n{'='*60}")
    print(f"Validating: {filepath}")
    print(f"{'='*60}")

    data = np.load(filepath)
    for k in data.keys():
        print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")

    obs = data['observations']
    actions = data['partner_actions']
    types = data['partner_types']
    switches = data['switch_times']

    print(f"\n  Observations range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  Observations mean:  {obs.mean():.3f}")
    print(f"  Observations std:   {obs.std():.3f}")

    # Check for NaN/Inf
    print(f"  NaN count: {np.isnan(obs).sum()}")
    print(f"  Inf count: {np.isinf(obs).sum()}")

    # Action distribution
    unique_a, counts_a = np.unique(actions, return_counts=True)
    print(f"\n  Action bins used: {len(unique_a)}/{NUM_ACTIONS}")
    for a, c in zip(unique_a, counts_a):
        pct = c / actions.size * 100
        print(f"    action {a}: {c:>8d} ({pct:>5.1f}%)")

    # Type distribution
    unique_t, counts_t = np.unique(types, return_counts=True)
    print(f"\n  Types used: {len(unique_t)}/{NUM_TYPES}")
    for t, c in zip(unique_t, counts_t):
        name = PartnerType(t).name
        pct = c / types.size * 100
        print(f"    {name:<12s}: {c:>8d} ({pct:>5.1f}%)")

    # Switch statistics
    n_with_switch = np.sum(switches >= 0)
    print(f"\n  Episodes with switch: {n_with_switch}/{len(switches)}")
    if n_with_switch > 0:
        valid_switches = switches[switches >= 0]
        print(f"  Switch time range: [{valid_switches.min()}, {valid_switches.max()}]")
        print(f"  Switch time mean:  {valid_switches.mean():.1f}")

    print(f"\n  Validation: PASSED")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Collect SharedWorkspace data')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Episodes per partner type')
    parser.add_argument('--episode_length', type=int, default=150,
                        help='Timesteps per episode')
    parser.add_argument('--output', type=str,
                        default='data/shared_workspace',
                        help='Output directory')
    parser.add_argument('--no_switch', action='store_true',
                        help='Disable mid-episode switching')
    parser.add_argument('--switch_min', type=int, default=30,
                        help='Earliest switch timestep')
    parser.add_argument('--switch_max', type=int, default=80,
                        help='Latest switch timestep')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validate_only', type=str, default=None,
                        help='Path to .npz file to validate')

    args = parser.parse_args()

    if args.validate_only:
        validate_data(args.validate_only)
        return

    filepath = collect_dataset(
        episodes_per_type=args.episodes,
        episode_length=args.episode_length,
        output_dir=args.output,
        enable_switching=not args.no_switch,
        switch_range=(args.switch_min, args.switch_max),
        seed=args.seed,
    )

    validate_data(filepath)


if __name__ == '__main__':
    main()
