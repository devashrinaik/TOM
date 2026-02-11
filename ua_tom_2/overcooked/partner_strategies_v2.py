"""
V2 partner strategies: Delayed-revelation types for Overcooked.

All types behave identically (GHM) until diverge_time ~ U[30,50],
then diverge into distinct behavioral patterns. This makes types
confusable during the early observation window, forcing models to
detect differences from post-divergence behavioral signals.

Types:
  RELIABLE  (0): Pure GHM throughout (no divergence)
  LAZY      (1): GHM until diverge, then increasing STAY probability
  SABOTEUR  (2): GHM until diverge, then soup-theft state machine
  ERRATIC   (3): GHM until diverge, then noisy GHM (40% random movement)

Common interface (same as v1):
  reset(state, mdp, mlam)
  get_action(state) -> action_tuple, action_index
"""

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.agents.agent import GreedyHumanModel


# ── Base class ──────────────────────────────────────────────────────

class DelayedRevealPartner:
    """Base class for delayed-revelation partners.

    All types use GHM before diverge_time. After diverge_time,
    subclasses implement type-specific behavior via _diverged_action().
    """

    name: str = "base"
    type_index: int = -1

    def __init__(self, diverge_range=(30, 50)):
        self.diverge_range = diverge_range
        self.diverge_time = None
        self._ghm = None
        self._mdp = None
        self._mlam = None
        self._step = 0

    def reset(self, state, mdp, mlam=None):
        self._step = 0
        self._mdp = mdp
        self._mlam = mlam
        self.diverge_time = np.random.randint(
            self.diverge_range[0], self.diverge_range[1] + 1)
        if mlam is not None:
            self._ghm = GreedyHumanModel(mlam)
            self._ghm.set_agent_index(1)

    @property
    def has_diverged(self):
        return self._step >= self.diverge_time and self._should_diverge()

    def _should_diverge(self):
        """Override to False for types that never diverge (RELIABLE)."""
        return True

    def get_action(self, state):
        self._step += 1
        if not self.has_diverged:
            action, _ = self._ghm.action(state)
            return action, Action.ACTION_TO_INDEX[action]
        return self._diverged_action(state)

    def _diverged_action(self, state):
        raise NotImplementedError


# ── RELIABLE (0): Pure GHM, never diverges ─────────────────────────

class ReliablePartner(DelayedRevealPartner):
    """Pure GHM throughout — the partner that never diverges.

    Acts as the "cooperative baseline" type. Counter-strategy: trusting
    (ego can rely on this partner to do its share).
    """
    name = "reliable"
    type_index = 0

    def _should_diverge(self):
        return False

    def _diverged_action(self, state):
        # Unreachable, but implement for safety
        action, _ = self._ghm.action(state)
        return action, Action.ACTION_TO_INDEX[action]


# ── LAZY (1): Ramps up STAY probability after diverge ──────────────

class LazyPartner(DelayedRevealPartner):
    """After diverge_time, increasingly inserts STAY actions.

    p_stay ramps linearly from 0.1 at diverge_time to 0.8 at
    diverge_time + 50 steps. Ego must compensate by becoming
    self-sufficient rather than relying on partner contributions.
    """
    name = "lazy"
    type_index = 1

    def _diverged_action(self, state):
        steps_since = self._step - self.diverge_time
        # Ramp from 0.1 to 0.8 over 50 steps
        p_stay = min(0.8, 0.1 + steps_since * 0.014)

        if np.random.random() < p_stay:
            return Action.ALL_ACTIONS[4], 4  # STAY
        action, _ = self._ghm.action(state)
        return action, Action.ACTION_TO_INDEX[action]


# ── SABOTEUR (2): Soup-theft state machine after diverge ───────────

class SaboteurPartner(DelayedRevealPartner):
    """After diverge_time, switches to adversarial soup-theft.

    Reuses the proven adversarial state machine from v1:
      LOITER → GET_DISH → STEAL_SOUP → HOLD → dump → LOITER

    Counter-strategy: protective (ego camps pot to beat saboteur).
    """
    name = "saboteur"
    type_index = 2

    # State constants
    LOITER = 0
    GET_DISH = 1
    STEAL_SOUP = 2
    HOLD = 3

    def __init__(self, diverge_range=(30, 50)):
        super().__init__(diverge_range)
        self._adv_state = self.LOITER
        self._hold_timer = 0
        self._pot_locs = []
        self._dish_locs = []
        self._pot_adjacent = []
        self._loiter_positions = []

    def reset(self, state, mdp, mlam=None):
        super().reset(state, mdp, mlam)
        self._adv_state = self.LOITER
        self._hold_timer = 0
        self._init_layout_cache(mdp)

    def _init_layout_cache(self, mdp):
        """Cache layout-specific positions for the state machine."""
        self._pot_locs = mdp.get_pot_locations()
        self._dish_locs = mdp.get_dish_dispenser_locations()

        # Pot-adjacent walkable cells
        self._pot_adjacent = []
        for pot in self._pot_locs:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                pos = (pot[0] + dx, pot[1] + dy)
                if (0 <= pos[0] < mdp.width and 0 <= pos[1] < mdp.height
                        and mdp.get_terrain_type_at_pos(pos) == " "):
                    self._pot_adjacent.append(pos)

        # Loiter positions: far from pot, dish, serving
        all_empty = []
        for x in range(mdp.width):
            for y in range(mdp.height):
                if mdp.get_terrain_type_at_pos((x, y)) == " ":
                    all_empty.append((x, y))

        serve_locs = mdp.get_serving_locations()
        key_locs = (list(self._pot_locs) + list(self._dish_locs)
                    + list(serve_locs))
        if key_locs and all_empty:
            def min_dist(p):
                return min(abs(p[0] - k[0]) + abs(p[1] - k[1])
                           for k in key_locs)
            all_empty.sort(key=lambda p: -min_dist(p))
            self._loiter_positions = all_empty[:3]
        else:
            self._loiter_positions = all_empty[:1] if all_empty else []

    def _move_toward(self, pos, target):
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        if dx == 0 and dy == 0:
            return Action.ALL_ACTIONS[4], 4
        if abs(dx) >= abs(dy):
            return (Action.ALL_ACTIONS[2], 2) if dx > 0 else (Action.ALL_ACTIONS[3], 3)
        return (Action.ALL_ACTIONS[1], 1) if dy > 0 else (Action.ALL_ACTIONS[0], 0)

    def _is_facing(self, player, target):
        pos = player.position
        orient = player.orientation
        return (pos[0] + orient[0], pos[1] + orient[1]) == target

    def _adjacent_to(self, pos, target):
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1]) == 1

    def _face_and_interact(self, player, pos, target):
        if self._adjacent_to(pos, target):
            if self._is_facing(player, target):
                return Action.ALL_ACTIONS[5], 5
            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            direction = {(0, -1): 0, (0, 1): 1,
                         (1, 0): 2, (-1, 0): 3}[(dx, dy)]
            return Action.ALL_ACTIONS[direction], direction
        return self._move_toward(pos, target)

    def _diverged_action(self, state):
        player = state.players[1]
        pos = player.position
        held = player.held_object

        pot_states = self._mdp.get_pot_states(state)
        has_ready = len(pot_states.get("ready", [])) > 0
        has_cooking = len(pot_states.get("cooking", [])) > 0

        # If holding soup → HOLD state
        if held is not None and held.name == "soup":
            self._adv_state = self.HOLD
            self._hold_timer = 0

        if self._adv_state == self.HOLD:
            self._hold_timer += 1
            # Move away from pot first
            if self._loiter_positions:
                target = self._loiter_positions[0]
                if pos != target:
                    return self._move_toward(pos, target)
            # After hold, dump soup on counter
            if self._hold_timer > 10:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    adj = (pos[0] + dx, pos[1] + dy)
                    if (0 <= adj[0] < self._mdp.width
                            and 0 <= adj[1] < self._mdp.height):
                        terrain = self._mdp.get_terrain_type_at_pos(adj)
                        if terrain == "X" and not state.has_object(adj):
                            if player.orientation == (dx, dy):
                                self._adv_state = self.LOITER
                                self._hold_timer = 0
                                return Action.ALL_ACTIONS[5], 5
                            direction = {(0, -1): 0, (0, 1): 1,
                                         (1, 0): 2, (-1, 0): 3}[(dx, dy)]
                            return Action.ALL_ACTIONS[direction], direction
                if self._hold_timer > 20 and self._loiter_positions:
                    others = [p for p in self._loiter_positions if p != pos]
                    if others:
                        return self._move_toward(
                            pos, others[self._hold_timer % len(others)])
            return Action.ALL_ACTIONS[4], 4

        if self._adv_state == self.STEAL_SOUP:
            if held is None or held.name != "dish":
                self._adv_state = self.LOITER
                return Action.ALL_ACTIONS[4], 4
            if has_ready:
                return self._face_and_interact(
                    player, pos, pot_states["ready"][0])
            if has_cooking:
                if self._pot_adjacent:
                    target = self._pot_adjacent[0]
                    if pos != target:
                        return self._move_toward(pos, target)
                return Action.ALL_ACTIONS[4], 4
            self._adv_state = self.LOITER
            return Action.ALL_ACTIONS[4], 4

        if self._adv_state == self.GET_DISH:
            if held is not None and held.name == "dish":
                self._adv_state = self.STEAL_SOUP
                return Action.ALL_ACTIONS[4], 4
            if held is not None:
                return Action.ALL_ACTIONS[5], 5
            if self._dish_locs:
                return self._face_and_interact(
                    player, pos, self._dish_locs[0])
            return Action.ALL_ACTIONS[4], 4

        # LOITER state
        if has_cooking or has_ready:
            self._adv_state = self.GET_DISH
            return self._diverged_action(state)
        if held is not None and held.name == "dish":
            self._adv_state = self.STEAL_SOUP
            return self._diverged_action(state)

        # Wander between loiter positions
        if self._loiter_positions:
            target = self._loiter_positions[
                (self._step // 5) % len(self._loiter_positions)]
            if pos != target:
                return self._move_toward(pos, target)
        return Action.ALL_ACTIONS[4], 4


# ── ERRATIC (3): Noisy GHM after diverge ──────────────────────────

class ErraticPartner(DelayedRevealPartner):
    """After diverge_time, mixes GHM with random movement actions.

    40% of the time, takes a random MOVEMENT action (not STAY or INTERACT),
    creating maximal physical interference — blocking corridors, colliding
    with ego, and disrupting planned paths.

    Counter-strategy: robust (ego avoids partner's position to minimize
    collision-related deadlocks).
    """
    name = "erratic"
    type_index = 3

    NOISE_PROB = 0.4

    def _diverged_action(self, state):
        if np.random.random() < self.NOISE_PROB:
            # Random movement (UP/DOWN/LEFT/RIGHT only — no STAY/INTERACT)
            # Movement-only noise is more disruptive than uniform random
            idx = np.random.randint(0, 4)
            return Action.ALL_ACTIONS[idx], idx
        action, _ = self._ghm.action(state)
        return action, Action.ACTION_TO_INDEX[action]


# ── Registry ────────────────────────────────────────────────────────

PARTNER_TYPES_V2 = {
    0: ReliablePartner,
    1: LazyPartner,
    2: SaboteurPartner,
    3: ErraticPartner,
}

PARTNER_NAMES_V2 = {
    0: "reliable",
    1: "lazy",
    2: "saboteur",
    3: "erratic",
}

NUM_PARTNER_TYPES_V2 = len(PARTNER_TYPES_V2)
