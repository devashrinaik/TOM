#!/usr/bin/env python3
"""
Ego counter-strategies for closed-loop evaluation.

Each strategy exploits a specific partner type's weakness:
  EgoStandard  — vs PASSIVE:     normal approach, no interference
  EgoYield     — vs HELPER:      stay home, let helper carry cube
  EgoFlank     — vs COMPETITOR:   side approach avoids push direction
  EgoFeint     — vs BLOCKER:     draw blocker to one side, approach from other

All strategies that grasp use the same proven sequence:
  above(15) → descend(15) → grasp(15) → lift
Only the PRE-APPROACH differs between strategies.

Interface:
  reset()
  get_action(left_tcp, cube_pos, right_tcp=None) → np.ndarray [4]
"""

import numpy as np


# ── Frame transform ──────────────────────────────────────────────────
def w2l(w):
    """World direction → left arm action frame."""
    return np.array([w[1], -w[0], w[2]])


def _move_toward(tcp, target, gain=3.0, gripper=1.0):
    """Proportional control: tcp → target in left-arm action frame.

    Uses moderate gain (3.0) matching the proven test_3 approach from
    test_shared_workspace.py. Higher gains cause oscillation at workspace limits.
    """
    direction = (target - tcp) * gain
    cmd = w2l(direction)
    cmd = np.clip(cmd[:3], -1, 1)
    return np.concatenate([cmd, [gripper]])


# ── Standard grasp phases (shared by all grasping strategies) ─────────
def _grasp_action(phase, step_in_phase, left_tcp, cube_pos):
    """
    Standard grasp sequence: above → descend → grasp → lift.
    Returns (action, next_phase, reset_step_flag).

    Uses fixed step counts (proven reliable in test_shared_workspace.py).
    The moderate gain (3.0) ensures smooth convergence without oscillation.
    """
    if phase == "above":
        target = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.1])
        done = step_in_phase >= 20
        return _move_toward(left_tcp, target, gain=3.0, gripper=1.0), "descend" if done else phase, done

    elif phase == "descend":
        target = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.01])
        done = step_in_phase >= 20
        return _move_toward(left_tcp, target, gain=3.0, gripper=1.0), "grasp" if done else phase, done

    elif phase == "grasp":
        done = step_in_phase >= 15
        return np.array([0.0, 0.0, 0.0, -1.0]), "lift" if done else phase, done

    elif phase == "lift":
        target = np.array([cube_pos[0], cube_pos[1], 0.35])
        return _move_toward(left_tcp, target, gain=3.0, gripper=-1.0), phase, False

    return np.array([0.0, 0.0, 0.0, 1.0]), phase, False


# ── EgoStandard (vs PASSIVE) ────────────────────────────────────────
class EgoStandard:
    """Normal approach → descend → grasp → lift. No partner to worry about."""

    name = "standard"

    def __init__(self):
        self.phase = "above"
        self.step_in_phase = 0

    def reset(self):
        self.phase = "above"
        self.step_in_phase = 0

    def get_action(self, left_tcp, cube_pos, right_tcp=None):
        self.step_in_phase += 1
        action, next_phase, reset = _grasp_action(self.phase, self.step_in_phase, left_tcp, cube_pos)
        if reset:
            self.phase = next_phase
            self.step_in_phase = 0
        return action


# ── EgoYield (vs HELPER) ─────────────────────────────────────────────
class EgoYield:
    """
    Stay near home position and don't interfere with the helper.

    Keeps the arm at its initial home position (small jitter only).
    This preserves the joint configuration so that switching to standard
    approach works identically to starting from scratch.

    The helper carries the cube toward the ego side; the ego should not
    reach for it and risk collision.
    """

    name = "yield"

    def __init__(self):
        self.home = None

    def reset(self):
        self.home = None

    def get_action(self, left_tcp, cube_pos, right_tcp=None):
        if self.home is None:
            self.home = left_tcp.copy()
        # Stay at home with minimal movement
        noise = np.random.randn(3) * 0.002
        target = self.home + noise
        return _move_toward(left_tcp, target, gain=1.0, gripper=1.0)


# ── EgoFlank (vs COMPETITOR) ────────────────────────────────────────
class EgoFlank:
    """
    Approach cube from x-offset to avoid the competitor's push direction.

    The competitor pushes cube in +y (away from ego). A standard y-axis
    approach loses the cube. The flank approaches from the side (x-offset),
    so the +y push is perpendicular and the ego can still reach the cube.

    Pre-approach: 15 steps to move to (cube_x + offset, cube_y, cube_z + 0.1)
    Then standard grasp sequence from that position.
    """

    name = "flank"
    FLANK_OFFSET_X = 0.06  # offset to avoid competitor gripper interference

    def __init__(self):
        self.phase = "preposition"
        self.step_in_phase = 0
        self.flank_side = 1.0

    def reset(self):
        self.phase = "preposition"
        self.step_in_phase = 0
        self.flank_side = np.random.choice([-1.0, 1.0])

    def get_action(self, left_tcp, cube_pos, right_tcp=None):
        self.step_in_phase += 1

        # Choose side opposite to partner on first step
        if right_tcp is not None and self.phase == "preposition" and self.step_in_phase == 1:
            partner_side_x = right_tcp[0] - cube_pos[0]
            self.flank_side = -1.0 if partner_side_x >= 0 else 1.0

        if self.phase == "preposition":
            # Move to x-offset position above cube
            target = np.array([
                cube_pos[0] + self.flank_side * self.FLANK_OFFSET_X,
                cube_pos[1],
                cube_pos[2] + 0.1
            ])
            if self.step_in_phase >= 15:
                self.phase = "above"  # converge to cube_pos at altitude before descend
                self.step_in_phase = 0
            return _move_toward(left_tcp, target, gain=5.0, gripper=1.0)

        # Standard grasp from the flanking position
        action, next_phase, reset = _grasp_action(self.phase, self.step_in_phase, left_tcp, cube_pos)
        if reset:
            self.phase = next_phase
            self.step_in_phase = 0
        return action


# ── EgoFeint (vs BLOCKER) ───────────────────────────────────────────
class EgoFeint:
    """
    Draw the blocker to one side, then approach from the other.

    The blocker tracks ego position but only updates every 12 steps
    and is biased toward where the ego IS. The feint:
    1. Approaches from one side (draws blocker there)
    2. Quickly repositions to the other side (blocker still tracking stale position)
    3. Standard grasp sequence from the unblocked side

    The blocker's delayed update means it stays committed to the feint
    direction while the ego has already moved to the opposite side.
    """

    name = "feint"
    FEINT_OFFSET_X = 0.08

    def __init__(self):
        self.phase = "draw"
        self.step_in_phase = 0
        self.feint_side = 1.0

    def reset(self):
        self.phase = "draw"
        self.step_in_phase = 0
        self.feint_side = np.random.choice([-1.0, 1.0])

    def get_action(self, left_tcp, cube_pos, right_tcp=None):
        self.step_in_phase += 1

        # Pick side opposite to blocker on first step
        if right_tcp is not None and self.phase == "draw" and self.step_in_phase == 1:
            blocker_side = right_tcp[0] - cube_pos[0]
            self.feint_side = -1.0 if blocker_side >= 0 else 1.0

        if self.phase == "draw":
            # Move toward cube from one side to draw blocker (18 steps to commit)
            target = np.array([
                cube_pos[0] + self.feint_side * self.FEINT_OFFSET_X,
                cube_pos[1],
                cube_pos[2] + 0.1
            ])
            if self.step_in_phase >= 18:
                self.phase = "above"  # go to standard grasp sequence, converge to cube_pos
                self.step_in_phase = 0
            return _move_toward(left_tcp, target, gain=5.0, gripper=1.0)

        # Standard grasp from the repositioned side
        action, next_phase, reset = _grasp_action(self.phase, self.step_in_phase, left_tcp, cube_pos)
        if reset:
            self.phase = next_phase
            self.step_in_phase = 0
        return action


# ── Strategy registry ────────────────────────────────────────────────

ALL_EGO_STRATEGIES = {
    'standard': EgoStandard,
    'yield':    EgoYield,
    'flank':    EgoFlank,
    'feint':    EgoFeint,
}

# Optimal counter-strategy for each partner type (by index)
COUNTER_STRATEGY = {
    0: 'yield',     # HELPER  → stay out of helper's way
    1: 'flank',     # COMPETITOR → side approach avoids push
    2: 'feint',     # BLOCKER → exploit tracking delay
    3: 'standard',  # PASSIVE → normal pickup
}
