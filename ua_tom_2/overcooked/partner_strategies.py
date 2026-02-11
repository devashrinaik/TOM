"""
Partner strategies for Overcooked closed-loop evaluation.

Four qualitatively distinct partner types:
  COOPERATIVE (0): GreedyHumanModel — optimal collaboration
  GREEDY      (1): Only picks up dishes + serves ready soups, never places onions
  ADVERSARIAL (2): Actively steals soups from pot — sabotages ego's cooking work
  RANDOM      (3): Uniform random actions

Common interface:
  reset(state, mdp)
  get_action(state) -> action_tuple, action_index
"""

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import Action, OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.agents.agent import GreedyHumanModel


# ── Base class ──────────────────────────────────────────────────────

class BasePartner:
    """Common interface for all partner strategies."""

    name: str = "base"
    type_index: int = -1

    def reset(self, state, mdp, mlam=None):
        """Reset internal state for a new episode."""
        pass

    def get_action(self, state):
        """
        Return (action_tuple, action_index) for the current state.

        action_tuple: element from Action.ALL_ACTIONS
        action_index: int 0-5
        """
        raise NotImplementedError


# ── COOPERATIVE (0): GreedyHumanModel wrapper ───────────────────────

class CooperativePartner(BasePartner):
    """
    Wraps GreedyHumanModel as player 1.

    Optimal collaboration: alternates between placing onions in pot
    and picking up dishes to serve completed soups.
    """
    name = "cooperative"
    type_index = 0

    def __init__(self):
        self._ghm = None

    def reset(self, state, mdp, mlam=None):
        if mlam is None:
            raise ValueError("CooperativePartner requires mlam")
        self._ghm = GreedyHumanModel(mlam)
        self._ghm.set_agent_index(1)

    def get_action(self, state):
        action, info = self._ghm.action(state)
        idx = Action.ACTION_TO_INDEX[action]
        return action, idx


# ── GREEDY (1): Only serves ready soups, never places onions ───────

class GreedyPartner(BasePartner):
    """
    Only picks up dishes and serves ready soups. Never places onions.

    Behavior: scan state for soups that are ready → go pick dish → go serve.
    If no soup ready, STAY. This partner is selfish — it reaps rewards from
    ego's onion work but never contributes to cooking.
    """
    name = "greedy"
    type_index = 1

    def __init__(self):
        self._ghm = None
        self._agent_index = 1

    def reset(self, state, mdp, mlam=None):
        if mlam is None:
            raise ValueError("GreedyPartner requires mlam")
        self._mdp = mdp
        self._mlam = mlam
        # Use GHM for pathfinding but filter actions
        self._ghm = GreedyHumanModel(mlam)
        self._ghm.set_agent_index(self._agent_index)

    def get_action(self, state):
        player = state.players[self._agent_index]
        held = player.held_object

        # Check if any pot has a ready soup
        pot_states = self._mdp.get_pot_states(state)
        has_ready = len(pot_states.get("ready", [])) > 0
        has_cooking = len(pot_states.get("cooking", [])) > 0
        soup_available = has_ready or has_cooking

        # If holding a dish and soup is ready, use GHM to go serve
        if held is not None and held.name == "dish" and soup_available:
            action, _ = self._ghm.action(state)
            return action, Action.ACTION_TO_INDEX[action]

        # If holding a soup, use GHM to go to serving area
        if held is not None and held.name == "soup":
            action, _ = self._ghm.action(state)
            return action, Action.ACTION_TO_INDEX[action]

        # If not holding anything and soup available, go get a dish
        if held is None and soup_available:
            action, _ = self._ghm.action(state)
            idx = Action.ACTION_TO_INDEX[action]
            # Filter: don't interact with onion dispensers
            if action == "interact":
                # Check what we're facing
                pos = player.position
                orient = player.orientation
                facing = (pos[0] + orient[0], pos[1] + orient[1])
                terrain = self._mdp.get_terrain_type_at_pos(facing)
                if terrain == "O":  # onion dispenser
                    return Action.ALL_ACTIONS[4], 4  # STAY instead
            return action, idx

        # If holding onion (picked up by mistake from GHM), drop it
        if held is not None and held.name == "onion":
            # Try to put it on a counter
            action, _ = self._ghm.action(state)
            return action, Action.ACTION_TO_INDEX[action]

        # Default: STAY
        return Action.ALL_ACTIONS[4], 4


# ── ADVERSARIAL (2): Actively steals soups ───────────────────────

class AdversarialPartner(BasePartner):
    """
    Actively sabotages ego by stealing finished soups from the pot.

    State machine:
      LOITER     → wander the far side of the kitchen, away from pot, until
                   soup starts cooking (letting ego do the work)
      GET_DISH   → go to dish dispenser, pick up a dish
      STEAL_SOUP → go to pot-adjacent, wait for soup to be ready, pick it up
      HOLD       → hold the stolen soup and STAY (wasting it)
      DUMP       → drop soup on counter to restart the cycle

    Key: during LOITER, the adversarial partner stays AWAY from the pot
    so the ego can cook. It only moves to steal once cooking begins.
    This creates a real penalty: ego invests in cooking, adversarial steals
    the output.
    """
    name = "adversarial"
    type_index = 2

    # State constants
    LOITER = 0
    GET_DISH = 1
    STEAL_SOUP = 2
    HOLD = 3

    def __init__(self):
        self._mdp = None
        self._mlam = None
        self._state = self.LOITER
        self._step = 0
        self._hold_timer = 0
        self._ghm = None

    def _is_valid_empty(self, mdp, pos):
        """Check if position is within bounds and is an empty cell."""
        x, y = pos
        if x < 0 or x >= mdp.width or y < 0 or y >= mdp.height:
            return False
        return mdp.get_terrain_type_at_pos(pos) == " "

    def reset(self, state, mdp, mlam=None):
        self._mdp = mdp
        self._mlam = mlam
        self._state = self.LOITER
        self._step = 0
        self._hold_timer = 0

        # GHM for pathfinding when needed (agent index 1 = partner)
        if mlam is not None:
            self._ghm = GreedyHumanModel(mlam)
            self._ghm.set_agent_index(1)

        # Cache key positions
        self._pot_locs = mdp.get_pot_locations()
        self._dish_locs = mdp.get_dish_dispenser_locations()

        # Find pot-adjacent positions (for stealing)
        self._pot_adjacent = []
        for pot in self._pot_locs:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                pos = (pot[0] + dx, pot[1] + dy)
                if self._is_valid_empty(mdp, pos):
                    self._pot_adjacent.append(pos)

        # Find loiter positions: far from BOTH pot and dish dispenser
        # so the adversarial doesn't accidentally block either.
        # In cramped_room: walkable cells are (1,1)(2,1)(3,1)(1,2)(2,2)(3,2)
        # Pot-adjacent is (2,1), dish-adjacent is (1,2)/(2,2).
        # Best loiter: (3,2) or (3,1) — far corner.
        all_empty = []
        for x in range(mdp.width):
            for y in range(mdp.height):
                if self._is_valid_empty(mdp, (x, y)):
                    all_empty.append((x, y))

        # Score each cell by distance from pot, dish dispenser, AND serving
        # Prefer cells far from all critical resources
        serve_locs = mdp.get_serving_locations()
        key_locs = (list(self._pot_locs) + list(self._dish_locs)
                    + list(serve_locs))
        if key_locs and all_empty:
            def min_dist(p):
                return min(abs(p[0]-k[0]) + abs(p[1]-k[1])
                           for k in key_locs)
            all_empty.sort(key=lambda p: -min_dist(p))
            self._loiter_positions = all_empty[:3]
        else:
            self._loiter_positions = all_empty[:1] if all_empty else []

    def _move_toward(self, pos, target):
        """Simple greedy movement toward target position."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]

        if dx == 0 and dy == 0:
            return Action.ALL_ACTIONS[4], 4  # STAY

        if abs(dx) >= abs(dy):
            if dx > 0:
                return Action.ALL_ACTIONS[2], 2  # RIGHT
            else:
                return Action.ALL_ACTIONS[3], 3  # LEFT
        else:
            if dy > 0:
                return Action.ALL_ACTIONS[1], 1  # DOWN
            else:
                return Action.ALL_ACTIONS[0], 0  # UP

    def _is_facing(self, player, target):
        """Check if player is adjacent to and facing target."""
        pos = player.position
        orient = player.orientation
        facing = (pos[0] + orient[0], pos[1] + orient[1])
        return facing == target

    def _adjacent_to(self, pos, target):
        """Check if pos is adjacent (manhattan dist 1) to target."""
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1]) == 1

    def _face_and_interact(self, player, pos, target):
        """Return action to face target and interact, or move toward it."""
        if self._adjacent_to(pos, target):
            if self._is_facing(player, target):
                return Action.ALL_ACTIONS[5], 5  # INTERACT
            else:
                dx = target[0] - pos[0]
                dy = target[1] - pos[1]
                direction = {(0, -1): 0, (0, 1): 1,
                             (1, 0): 2, (-1, 0): 3}[(dx, dy)]
                return Action.ALL_ACTIONS[direction], direction
        else:
            return self._move_toward(pos, target)

    def get_action(self, state):
        self._step += 1
        player = state.players[1]
        pos = player.position
        held = player.held_object

        pot_states = self._mdp.get_pot_states(state)
        has_ready = len(pot_states.get("ready", [])) > 0
        has_cooking = len(pot_states.get("cooking", [])) > 0

        # ── State machine ──

        # If holding a soup, go to HOLD
        if held is not None and held.name == "soup":
            self._state = self.HOLD
            self._hold_timer = 0

        if self._state == self.HOLD:
            self._hold_timer += 1

            # First: move AWAY from pot to a loiter position so ego can cook
            if self._loiter_positions:
                target = self._loiter_positions[0]
                if pos != target:
                    return self._move_toward(pos, target)

            # After short hold, try to dump soup on any reachable counter
            if self._hold_timer > 10:
                # Try adjacent counters first
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    adj = (pos[0] + dx, pos[1] + dy)
                    if (0 <= adj[0] < self._mdp.width and
                            0 <= adj[1] < self._mdp.height):
                        terrain = self._mdp.get_terrain_type_at_pos(adj)
                        if terrain == "X" and not state.has_object(adj):
                            if player.orientation == (dx, dy):
                                self._state = self.LOITER
                                self._hold_timer = 0
                                return Action.ALL_ACTIONS[5], 5  # drop
                            else:
                                direction = {(0, -1): 0, (0, 1): 1,
                                             (1, 0): 2, (-1, 0): 3}[(dx, dy)]
                                return Action.ALL_ACTIONS[direction], direction

                # If no adjacent counter is free, wander to find one
                if self._hold_timer > 20:
                    # Move to a different position to find empty counters
                    other_positions = [p for p in self._loiter_positions
                                       if p != pos]
                    if other_positions:
                        return self._move_toward(
                            pos, other_positions[
                                self._hold_timer % len(other_positions)])

            return Action.ALL_ACTIONS[4], 4  # STAY

        if self._state == self.STEAL_SOUP:
            if held is None or held.name != "dish":
                self._state = self.LOITER
                return Action.ALL_ACTIONS[4], 4

            if has_ready:
                # Soup ready — go get it!
                ready_pot = pot_states["ready"][0]
                return self._face_and_interact(player, pos, ready_pot)

            if has_cooking:
                # Wait near pot for soup to finish
                if self._pot_adjacent:
                    target = self._pot_adjacent[0]
                    if pos != target:
                        return self._move_toward(pos, target)
                return Action.ALL_ACTIONS[4], 4
            else:
                # Nothing cooking or ready — ego hasn't started yet
                # Go back to loitering
                self._state = self.LOITER
                return Action.ALL_ACTIONS[4], 4

        if self._state == self.GET_DISH:
            if held is not None and held.name == "dish":
                self._state = self.STEAL_SOUP
                return Action.ALL_ACTIONS[4], 4

            if held is not None:
                # Holding something else, try to drop it
                return Action.ALL_ACTIONS[5], 5  # INTERACT

            # Go to dish dispenser
            if self._dish_locs:
                return self._face_and_interact(
                    player, pos, self._dish_locs[0])

            return Action.ALL_ACTIONS[4], 4

        # ── LOITER state ──
        # Stay far from pot, let ego cook. Watch for cooking to start.
        if has_cooking or has_ready:
            # Soup is cooking — time to go steal!
            self._state = self.GET_DISH
            return self.get_action(state)

        if held is not None and held.name == "dish":
            # Already have a dish from previous cycle
            self._state = self.STEAL_SOUP
            return self.get_action(state)

        # Wander between loiter positions (far from pot)
        if self._loiter_positions:
            target = self._loiter_positions[
                (self._step // 5) % len(self._loiter_positions)]
            if pos != target:
                return self._move_toward(pos, target)

        return Action.ALL_ACTIONS[4], 4  # STAY


# ── RANDOM (3): Uniform random actions ─────────────────────────────

class RandomPartner(BasePartner):
    """Uniform random over all 6 actions each step."""
    name = "random"
    type_index = 3

    def get_action(self, state):
        idx = np.random.randint(0, 6)
        return Action.ALL_ACTIONS[idx], idx


# ── Registry ────────────────────────────────────────────────────────

PARTNER_TYPES = {
    0: CooperativePartner,
    1: GreedyPartner,
    2: AdversarialPartner,
    3: RandomPartner,
}

PARTNER_NAMES = {
    0: "cooperative",
    1: "greedy",
    2: "adversarial",
    3: "random",
}

NUM_PARTNER_TYPES = len(PARTNER_TYPES)
