#!/usr/bin/env python3
"""
Partner strategies for closed-loop evaluation.

4 qualitatively distinct partner types:
  HELPER:     grasps cube, lifts and carries toward ego side — ego should yield
  COMPETITOR: rushes to cube, pushes it away from ego — ego must flank
  BLOCKER:    interposes between ego and cube — ego must feint
  PASSIVE:    idle — ego proceeds normally

Key design: each partner has a clear exploitable weakness that only
the correct ego counter-strategy addresses.
"""

import numpy as np
from enum import IntEnum


class PartnerType(IntEnum):
    HELPER = 0
    COMPETITOR = 1
    BLOCKER = 2
    PASSIVE = 3


NUM_TYPES = len(PartnerType)

PARTNER_NAMES = {0: 'HELPER', 1: 'COMPETITOR', 2: 'BLOCKER', 3: 'PASSIVE'}


# ── Frame transform ──────────────────────────────────────────────────
def w2r(w):
    """World direction → right arm action frame."""
    return np.array([-w[1], w[0], w[2]])


def _move_toward(tcp, target, gain=5.0, gripper=1.0):
    """Proportional control: tcp → target in right-arm action frame."""
    direction = (target - tcp) * gain
    cmd = w2r(direction)
    cmd = np.clip(cmd[:3], -1, 1)
    return np.concatenate([cmd, [gripper]])


# ── Base ─────────────────────────────────────────────────────────────
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


# ── HELPER: grasp, lift, carry to ego side ────────────────────────────
class HelperStrategy(PartnerStrategy):
    """
    Approaches cube, grasps firmly, lifts straight up, then carries toward ego side.
    Uses the same proven grasp sequence as EgoStandard.

    Ego counter: YIELD (stay home, don't interfere).
    Wrong strategy (standard/flank/feint) = ego collides with helper,
    helper drops cube, task fails.
    """
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
            if self.step_in_phase >= 20:
                self.phase = "descend"
                self.step_in_phase = 0
            return _move_toward(right_tcp, target, gain=5.0, gripper=1.0)

        elif self.phase == "descend":
            target = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.01])
            if self.step_in_phase >= 20:
                self.phase = "grasp"
                self.step_in_phase = 0
            return _move_toward(right_tcp, target, gain=5.0, gripper=1.0)

        elif self.phase == "grasp":
            if self.step_in_phase >= 20:
                self.phase = "lift"
                self.step_in_phase = 0
            return np.array([0.0, 0.0, 0.0, -1.0])

        elif self.phase == "lift":
            # Lift straight up first to secure grip
            target = np.array([right_tcp[0], right_tcp[1], 0.25])
            if self.step_in_phase >= 25:
                self.phase = "carry"
                self.step_in_phase = 0
            return _move_toward(right_tcp, target, gain=5.0, gripper=-1.0)

        elif self.phase == "carry":
            # Carry toward ego side while maintaining height
            target = np.array([right_tcp[0], -0.08, 0.25])
            return _move_toward(right_tcp, target, gain=3.0, gripper=-1.0)

        return np.array([0.0, 0.0, 0.0, 1.0])


# ── COMPETITOR: grasp and steal ───────────────────────────────────────
class CompetitorStrategy(PartnerStrategy):
    """
    Races to grasp cube and steal it to partner's side.
    Uses the proven above→descend→grasp→steal sequence.

    When standard ego and competitor both approach the cube from opposite
    y-sides simultaneously, their grippers interfere and neither gets a
    clean grasp. The flank approaches from x-offset, avoiding the deadlock.

    Ego counter: FLANK (x-offset avoids gripper interference).
    Wrong strategies: standard (deadlock), yield (cube stolen).
    """
    def __init__(self):
        super().__init__(PartnerType.COMPETITOR)

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
            return _move_toward(right_tcp, target, gain=5.0, gripper=1.0)

        elif self.phase == "descend":
            target = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.01])
            if self.step_in_phase >= 15:
                self.phase = "grasp"
                self.step_in_phase = 0
            return _move_toward(right_tcp, target, gain=5.0, gripper=1.0)

        elif self.phase == "grasp":
            if self.step_in_phase >= 15:
                self.phase = "steal"
                self.step_in_phase = 0
            return np.array([0.0, 0.0, 0.0, -1.0])

        elif self.phase == "steal":
            # Lift and carry to partner's side
            target = np.array([right_tcp[0], 0.15, 0.25])
            return _move_toward(right_tcp, target, gain=5.0, gripper=-1.0)

        return np.array([0.0, 0.0, 0.0, 1.0])


# ── BLOCKER: interpose with delayed tracking ─────────────────────────
class BlockerStrategy(PartnerStrategy):
    """
    Interposes between ego and cube by physical presence (open gripper).
    Blocks by occupying the space the ego needs to descend through.

    Key weakness: updates tracking target only every UPDATE_INTERVAL steps.
    Between updates, the blocker commits to a stale target position.
    Tracks ego's direction aggressively, so feints draw it far from cube.

    Ego counter: FEINT (draw blocker to one side, approach from other).
    Wrong strategies: standard (blocked), yield (cube never comes).
    """
    UPDATE_INTERVAL = 20  # steps between tracking updates

    def __init__(self):
        super().__init__(PartnerType.BLOCKER)
        self._cached_target = None

    def reset(self):
        super().reset()
        self.phase = "block"
        self._cached_target = None

    def get_action(self, right_tcp, cube_pos, left_tcp):
        self.step_count += 1
        self.step_in_phase += 1

        # Track ego and interpose — delayed updates only
        if self._cached_target is None or self.step_count % self.UPDATE_INTERVAL == 0:
            # Position between ego and cube, at a distance that makes
            # the blocker actually move away from cube when ego feints
            ego_dir = left_tcp - cube_pos
            ego_dir_xy = ego_dir[:2]
            ego_dist_xy = np.linalg.norm(ego_dir_xy) + 1e-8
            # Position along ego→cube line, 0.05m from cube center
            block_offset = (ego_dir_xy / ego_dist_xy) * 0.05
            self._cached_target = cube_pos.copy()
            self._cached_target[0] += block_offset[0]
            self._cached_target[1] += block_offset[1]
            self._cached_target[2] = cube_pos[2] + 0.03  # just above cube

        return _move_toward(right_tcp, self._cached_target, gain=3.0, gripper=1.0)


# ── PASSIVE: idle ────────────────────────────────────────────────────
class PassiveStrategy(PartnerStrategy):
    """Stay near home, minimal movement."""
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


# ── Registry ─────────────────────────────────────────────────────────

STRATEGY_CLASSES = {
    PartnerType.HELPER: HelperStrategy,
    PartnerType.COMPETITOR: CompetitorStrategy,
    PartnerType.BLOCKER: BlockerStrategy,
    PartnerType.PASSIVE: PassiveStrategy,
}


def make_strategy(ptype: PartnerType) -> PartnerStrategy:
    return STRATEGY_CLASSES[ptype]()
