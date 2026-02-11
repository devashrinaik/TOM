#!/usr/bin/env python3
"""
Test script: Distinct partner behaviors in shared workspace.

4 qualitatively different partner strategies:
  HELPER     - picks cube, carries toward ego
  COMPETITOR - races to grab cube, steals to own side
  BLOCKER    - positions near cube, pushes it away from ego
  PASSIVE    - stays idle at home position

Ego (left arm) tries to pick up the cube.
Partner (right arm) follows one of the strategies above.
"""

import numpy as np
import torch
import gymnasium as gym
import sys

sys.path.insert(0, '/mnt/ssd1/devashri/TOM_revision/ua_tom_2/scripts')
import test_shared_workspace  # registers SharedWorkspacePickCube-v1


# ── Frame transforms ─────────────────────────────────────────────────
def w2l(w):
    return np.array([w[1], -w[0], w[2]])

def w2r(w):
    return np.array([-w[1], w[0], w[2]])


def _move_toward(tcp, target, frame_fn, gain=5.0, gripper=1.0):
    """Standard proportional control toward a target."""
    direction = (target - tcp) * gain
    cmd = frame_fn(direction)
    cmd = np.clip(cmd[:3], -1, 1)
    return np.concatenate([cmd, [gripper]])


# ── Partner Strategies ────────────────────────────────────────────────
class PartnerStrategy:
    def __init__(self, name):
        self.name = name
        self.step_count = 0
        self.phase = "init"

    def get_action(self, right_tcp, cube_pos, left_tcp):
        raise NotImplementedError

    def reset(self):
        self.step_count = 0
        self.phase = "init"


class HelperStrategy(PartnerStrategy):
    """Step-count based: approaches cube, grasps, carries toward ego side."""
    def __init__(self):
        super().__init__("HELPER")
        self.step_in_phase = 0

    def reset(self):
        super().reset()
        self.phase = "above"
        self.step_in_phase = 0

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
            # Carry toward ego side (y < 0)
            target = np.array([cube_pos[0], -0.08, 0.20])
            return _move_toward(right_tcp, target, w2r, gain=3.0, gripper=-1.0)

        return np.array([0.0, 0.0, 0.0, 1.0])


class CompetitorStrategy(PartnerStrategy):
    """Step-count based: races directly to cube, grabs, steals to own side."""
    def __init__(self):
        super().__init__("COMPETITOR")
        self.step_in_phase = 0

    def reset(self):
        super().reset()
        self.phase = "race"
        self.step_in_phase = 0

    def get_action(self, right_tcp, cube_pos, left_tcp):
        self.step_count += 1
        self.step_in_phase += 1

        if self.phase == "race":
            # Skip "above" — go directly toward cube (aggressive)
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
    """Moves to interpose between ego and cube. Pushes cube away when ego approaches."""
    def __init__(self):
        super().__init__("BLOCKER")

    def reset(self):
        super().reset()
        self.phase = "block"

    def get_action(self, right_tcp, cube_pos, left_tcp):
        self.step_count += 1

        ego_dist = np.linalg.norm((cube_pos - left_tcp)[:2])

        if ego_dist < 0.12:
            # Push: go to cube and shove it away from ego
            push_dir = cube_pos - left_tcp
            push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-8)
            target = cube_pos + push_dir * 0.01
            target[2] = cube_pos[2]  # stay at cube height
            return _move_toward(right_tcp, target, w2r, gain=8.0, gripper=1.0)
        else:
            # Hover between ego and cube
            midpoint = (cube_pos + left_tcp) / 2
            # Bias toward cube side
            target = 0.7 * cube_pos + 0.3 * midpoint
            target[2] = cube_pos[2] + 0.04
            return _move_toward(right_tcp, target, w2r, gain=5.0, gripper=1.0)


class PassiveStrategy(PartnerStrategy):
    """Stays near home, barely moves."""
    def __init__(self):
        super().__init__("PASSIVE")
        self.home = None

    def reset(self):
        super().reset()
        self.home = None

    def get_action(self, right_tcp, cube_pos, left_tcp):
        self.step_count += 1
        if self.home is None:
            self.home = right_tcp.copy()
        noise = np.random.randn(3) * 0.005
        target = self.home + noise
        direction = (target - right_tcp) * 0.5
        cmd = w2r(direction)
        cmd = np.clip(cmd[:3], -0.05, 0.05)
        return np.concatenate([cmd, [1.0]])


# ── Ego controller (left arm) ────────────────────────────────────────
class EgoController:
    """Step-count based controller (like test_shared_workspace tests 3/4 that work)."""
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


# ── Run episode ──────────────────────────────────────────────────────
def run_episode(strategy, max_steps=150):
    env = gym.make(
        "SharedWorkspacePickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
    )
    obs, info = env.reset()
    ego = EgoController()
    strategy.reset()

    right_positions, left_positions, cube_positions = [], [], []
    right_actions, left_actions = [], []

    for step in range(max_steps):
        extra = obs['extra']
        left_tcp = extra['left_arm_tcp'][0, :3].cpu().numpy()
        right_tcp = extra['right_arm_tcp'][0, :3].cpu().numpy()
        cube_pos = extra['cube_pose'][0, :3].cpu().numpy()

        left_positions.append(left_tcp.copy())
        right_positions.append(right_tcp.copy())
        cube_positions.append(cube_pos.copy())

        left_cmd = ego.get_action(left_tcp, cube_pos)
        right_cmd = strategy.get_action(right_tcp, cube_pos, left_tcp)

        left_actions.append(left_cmd.copy())
        right_actions.append(right_cmd.copy())

        # Debug at key timesteps
        if step % 30 == 0 or step < 3:
            ego_phase = ego.phase
            partner_phase = getattr(strategy, 'phase', '?')
            r2c = np.linalg.norm(cube_pos - right_tcp)
            l2c = np.linalg.norm(cube_pos - left_tcp)
            print(f"  t={step:3d} ego={ego_phase:<10s} partner={partner_phase:<12s} "
                  f"r2c={r2c:.3f} l2c={l2c:.3f} cube=({cube_pos[0]:+.3f},{cube_pos[1]:+.3f},{cube_pos[2]:.3f})")

        action = {
            'panda_wristcam-0': torch.tensor(np.array([left_cmd]), dtype=torch.float32),
            'panda_wristcam-1': torch.tensor(np.array([right_cmd]), dtype=torch.float32),
        }
        obs, reward, terminated, truncated, info = env.step(action)

    env.close()
    return {
        'strategy': strategy.name,
        'right_positions': np.array(right_positions),
        'left_positions': np.array(left_positions),
        'cube_positions': np.array(cube_positions),
        'right_actions': np.array(right_actions),
        'left_actions': np.array(left_actions),
    }


def analyze(results):
    """Compare trajectories across strategies."""
    print("\n" + "="*70)
    print(f"{'Strategy':<14} {'R-displ':>8} {'R-path':>8} {'C-displ':>8} "
          f"{'C-fin-Y':>8} {'C-fin-Z':>8} {'R-grip':>7}")
    print("-"*70)
    for res in results:
        rp = res['right_positions']
        cp = res['cube_positions']
        ra = res['right_actions']
        r_displ = np.linalg.norm(rp[-1] - rp[0])
        r_path = np.sum(np.linalg.norm(np.diff(rp, axis=0), axis=1))
        c_displ = np.linalg.norm(cp[-1] - cp[0])
        c_fin_y = cp[-1, 1]
        c_fin_z = cp[-1, 2]
        mean_grip = np.mean(ra[:, 3])
        print(f"{res['strategy']:<14} {r_displ:>8.3f} {r_path:>8.3f} {c_displ:>8.3f} "
              f"{c_fin_y:>+8.3f} {c_fin_z:>8.3f} {mean_grip:>+7.3f}")

    # Action sequence distinguishability: pairwise DTW-like distance
    print("\n--- Action Sequence Pairwise L2 Distance (right arm) ---")
    names = [r['strategy'] for r in results]
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            a_i = results[i]['right_actions'][:, :3]
            a_j = results[j]['right_actions'][:, :3]
            min_len = min(len(a_i), len(a_j))
            dist = np.mean(np.linalg.norm(a_i[:min_len] - a_j[:min_len], axis=1))
            print(f"  {names[i]:<12} vs {names[j]:<12}: mean_L2 = {dist:.4f}")


if __name__ == "__main__":
    strategies = [
        HelperStrategy(),
        CompetitorStrategy(),
        BlockerStrategy(),
        PassiveStrategy(),
    ]

    all_results = []
    for strat in strategies:
        print(f"\n{'#'*60}")
        print(f"  {strat.name}")
        print(f"{'#'*60}")
        result = run_episode(strat)
        all_results.append(result)

    analyze(all_results)
