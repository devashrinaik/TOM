"""
Adaptive ego policy for Overcooked closed-loop evaluation.

Maps model predictions (type_logits, switch_probs) to ego strategy selection.
Adapted from closed_loop/ego_policy.py for Overcooked's discrete action space.

Ego strategies:
  coordinated: GHM + wait N steps after INTERACT (yields time for cooperative partner)
  independent: Pure GHM (self-sufficient, no coordination overhead)
  preemptive:  GHM but races to pick up soup before adversarial partner can steal it
  passive:     STAY only (baseline)

Conditions:
  oracle:          Ground truth type + switch time
  ua_tom:          Use pretrained UA-ToM predictions
  gru:             Use pretrained GRU baseline predictions
  no_detect:       Fixed initial strategy, never adapt
  random:          Random strategy switches
  always_coordinated: Fixed coordinated strategy
"""

import numpy as np
import torch

from overcooked_ai_py.mdp.overcooked_mdp import Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.agents.agent import GreedyHumanModel


# ── Ego strategies ──────────────────────────────────────────────────

class CoordinatedEgo:
    """
    GHM + short yield after INTERACT to let cooperative partner complement.

    After each INTERACT action, inserts `wait_steps` STAY actions. When a
    cooperative partner is present, this small pause lets the partner fill the
    complementary role (e.g., ego places onion → waits → partner places next
    onion or picks up dish). Against non-cooperative partners, these waits
    waste precious time.

    Also avoids moving to partner's position to reduce corridor collisions.
    """
    name = "coordinated"

    def __init__(self, mlam, wait_steps=3):
        self._ghm = GreedyHumanModel(mlam)
        self._ghm.set_agent_index(0)
        self._wait_steps = wait_steps
        self._wait_counter = 0

    def get_action(self, state):
        # If in a wait period, STAY
        if self._wait_counter > 0:
            self._wait_counter -= 1
            return Action.ALL_ACTIONS[4], 4  # STAY

        action, _ = self._ghm.action(state)
        idx = Action.ACTION_TO_INDEX[action]

        # After INTERACT, start a wait period
        if action == "interact":
            self._wait_counter = self._wait_steps

        # Avoid colliding with partner in narrow corridor
        if action != "interact" and action != (0, 0):
            ego_pos = state.players[0].position
            partner_pos = state.players[1].position
            next_pos = (ego_pos[0] + action[0], ego_pos[1] + action[1])
            if next_pos == partner_pos:
                return Action.ALL_ACTIONS[4], 4  # STAY instead of bumping

        return action, idx


class IndependentEgo:
    """
    Pure GreedyHumanModel — self-sufficient, no coordination overhead.

    Optimal against unreliable partners (GREEDY, RANDOM) where coordination
    has no benefit. Also the default fallback strategy.
    """
    name = "independent"

    def __init__(self, mlam):
        self._ghm = GreedyHumanModel(mlam)
        self._ghm.set_agent_index(0)

    def get_action(self, state):
        action, _ = self._ghm.action(state)
        return action, Action.ACTION_TO_INDEX[action]


class PreemptiveEgo:
    """
    GHM that prioritizes securing soups before the adversarial partner steals.

    Mostly uses GHM, but with one critical override: when holding a dish and
    soup is cooking, camps at the pot-adjacent cell to pick up soup the instant
    it's ready — before the adversarial partner can arrive.

    Against adversarial: ego is already at pot when soup finishes, beating the
    adversarial's steal attempt. Saves ~30-40 reward per episode.
    Against cooperative: slight throughput loss since ego camps at pot instead
    of starting next onion batch. Loses ~0-10 reward vs independent.
    """
    name = "preemptive"

    def __init__(self, mlam):
        self._mlam = mlam
        self._ghm = GreedyHumanModel(mlam)
        self._ghm.set_agent_index(0)
        self._mdp = None

    def _init_mdp(self, state):
        if self._mdp is None:
            self._mdp = self._mlam.mdp

    def get_action(self, state):
        self._init_mdp(state)

        player = state.players[0]
        pos = player.position
        held = player.held_object

        pot_states = self._mdp.get_pot_states(state)
        has_ready = len(pot_states.get("ready", [])) > 0
        has_cooking = len(pot_states.get("cooking", [])) > 0

        # Override: holding dish + soup cooking → camp at pot-adjacent cell
        # GHM might wander; we stay put to beat adversarial to the soup
        if held is not None and held.name == "dish" and has_cooking:
            pot_loc = pot_states["cooking"][0]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                adj = (pot_loc[0] + dx, pot_loc[1] + dy)
                if (0 <= adj[0] < self._mdp.width and
                        0 <= adj[1] < self._mdp.height and
                        self._mdp.get_terrain_type_at_pos(adj) == " "):
                    if pos == adj:
                        # At pot — face it and wait
                        face_dx = pot_loc[0] - pos[0]
                        face_dy = pot_loc[1] - pos[1]
                        if player.orientation != (face_dx, face_dy):
                            direction = {(0, -1): 0, (0, 1): 1,
                                         (1, 0): 2, (-1, 0): 3}[
                                (face_dx, face_dy)]
                            return Action.ALL_ACTIONS[direction], direction
                        return Action.ALL_ACTIONS[4], 4  # camp and wait
                    else:
                        # Move toward pot-adjacent
                        diff_x = adj[0] - pos[0]
                        diff_y = adj[1] - pos[1]
                        if diff_x == 0 and diff_y == 0:
                            return Action.ALL_ACTIONS[4], 4
                        if abs(diff_x) >= abs(diff_y):
                            idx = 2 if diff_x > 0 else 3
                        else:
                            idx = 1 if diff_y > 0 else 0
                        return Action.ALL_ACTIONS[idx], idx

        # Everything else: let GHM handle (placing onions, getting dish,
        # picking up ready soup, serving)
        action, _ = self._ghm.action(state)
        return action, Action.ACTION_TO_INDEX[action]


class PassiveEgo:
    """Always STAY — baseline for reward comparison."""
    name = "passive"

    def __init__(self, mlam):
        pass

    def get_action(self, state):
        return Action.ALL_ACTIONS[4], 4  # STAY


# Strategy registry
EGO_STRATEGIES = {
    "coordinated": CoordinatedEgo,
    "independent": IndependentEgo,
    "preemptive": PreemptiveEgo,
    "passive": PassiveEgo,
}

# Optimal counter-strategy for each partner type
# Note: 3/4 types map to "independent" — use hard_detection_rate metric
# (in closed_loop_eval.py) to get honest detection metrics on the 6/12
# transitions that actually require a strategy change.
COUNTER_STRATEGY = {
    0: "independent",   # COOPERATIVE → GHM is self-sufficient
    1: "independent",   # GREEDY → self-sufficient
    2: "preemptive",    # ADVERSARIAL → camp pot to beat soup-stealer
    3: "independent",   # RANDOM → self-sufficient
}


# ── V2 Ego Strategies ──────────────────────────────────────────────
# Designed for bijective optimality against v2 delayed-revelation partners.

class TrustingEgo:
    """Onion-only: ego handles onion placement, relies on partner for
    the dish→soup→serve pipeline.

    Never picks up dishes. When pot is cooking/ready and ego has nothing
    to carry, waits for partner to complete the serving pipeline.

    Trade-off: +efficiency with reliable partner (clean division of labor,
    no resource contention), -throughput when partner doesn't serve
    (lazy/erratic/saboteur).
    """
    name = "trusting"

    def __init__(self, mlam):
        self._ghm = GreedyHumanModel(mlam)
        self._ghm.set_agent_index(0)
        self._mdp = mlam.mdp

    def get_action(self, state):
        player = state.players[0]
        pos = player.position
        held = player.held_object

        # If accidentally holding dish: drop on nearest empty counter
        if held is not None and held.name == "dish":
            return self._drop_on_counter(player, pos)

        # If holding soup (rare — from accidental pot interaction): serve it
        if held is not None and held.name == "soup":
            action, _ = self._ghm.action(state)
            return action, Action.ACTION_TO_INDEX[action]

        # If holding onion: GHM will route to pot and place it
        if held is not None and held.name == "onion":
            action, _ = self._ghm.action(state)
            return action, Action.ACTION_TO_INDEX[action]

        # Holding nothing — check pot state
        pot_states = self._mdp.get_pot_states(state)
        has_cooking = len(pot_states.get("cooking", [])) > 0
        has_ready = len(pot_states.get("ready", [])) > 0

        # If pot cooking/ready and we can't contribute: wait for partner
        if has_cooking or has_ready:
            return Action.ALL_ACTIONS[4], 4  # STAY

        # Use GHM but block dish/serving interactions
        action, _ = self._ghm.action(state)
        idx = Action.ACTION_TO_INDEX[action]

        if action == "interact":
            orient = player.orientation
            facing = (pos[0] + orient[0], pos[1] + orient[1])
            if (0 <= facing[0] < self._mdp.width
                    and 0 <= facing[1] < self._mdp.height):
                terrain = self._mdp.get_terrain_type_at_pos(facing)
                if terrain == "D":  # dish dispenser — block
                    return Action.ALL_ACTIONS[4], 4
                if terrain == "S":  # serving location — block
                    return Action.ALL_ACTIONS[4], 4

        return action, idx

    def _drop_on_counter(self, player, pos):
        """Drop held item on nearest empty counter."""
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj = (pos[0] + dx, pos[1] + dy)
            if (0 <= adj[0] < self._mdp.width
                    and 0 <= adj[1] < self._mdp.height):
                terrain = self._mdp.get_terrain_type_at_pos(adj)
                if terrain == "X":
                    if player.orientation == (dx, dy):
                        return Action.ALL_ACTIONS[5], 5  # drop
                    direction = {(0, -1): 0, (0, 1): 1,
                                 (1, 0): 2, (-1, 0): 3}[(dx, dy)]
                    return Action.ALL_ACTIONS[direction], direction
        return Action.ALL_ACTIONS[4], 4


class SelfSufficientEgo:
    """Pure GHM — self-sufficient, no coordination overhead.

    Optimal when partner is unreliable (LAZY, ERRATIC) and ego must
    handle the full cooking pipeline alone.
    """
    name = "self_sufficient"

    def __init__(self, mlam):
        self._ghm = GreedyHumanModel(mlam)
        self._ghm.set_agent_index(0)

    def get_action(self, state):
        action, _ = self._ghm.action(state)
        return action, Action.ACTION_TO_INDEX[action]


class ProtectiveEgo:
    """GHM + camp pot when soup is cooking/ready while holding dish.

    Uses standard GHM for all tasks but with one override: when holding
    a dish and soup is cooking, camps at pot-adjacent cell to pick up
    soup the instant it's ready — before saboteur can arrive.

    This is the proven preemptive approach from v1. Against saboteur
    it prevents soup theft (+30 reward). Against other types, camping
    has near-zero cost since ego needs to be at pot anyway.
    """
    name = "protective"

    def __init__(self, mlam):
        self._mlam = mlam
        self._ghm = GreedyHumanModel(mlam)
        self._ghm.set_agent_index(0)
        self._mdp = mlam.mdp

    def get_action(self, state):
        player = state.players[0]
        pos = player.position
        held = player.held_object

        pot_states = self._mdp.get_pot_states(state)
        has_cooking = len(pot_states.get("cooking", [])) > 0
        has_ready = len(pot_states.get("ready", [])) > 0

        # Override: holding dish + soup cooking/ready → camp at pot
        if held is not None and held.name == "dish" and (has_cooking or has_ready):
            pot_loc = None
            if has_ready:
                pot_loc = pot_states["ready"][0]
            elif has_cooking:
                pot_loc = pot_states["cooking"][0]

            if pot_loc is not None:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    adj = (pot_loc[0] + dx, pot_loc[1] + dy)
                    if (0 <= adj[0] < self._mdp.width
                            and 0 <= adj[1] < self._mdp.height
                            and self._mdp.get_terrain_type_at_pos(adj) == " "):
                        if pos == adj:
                            face = (pot_loc[0] - pos[0], pot_loc[1] - pos[1])
                            if player.orientation != face:
                                direction = {(0, -1): 0, (0, 1): 1,
                                             (1, 0): 2, (-1, 0): 3}[face]
                                return Action.ALL_ACTIONS[direction], direction
                            if has_ready:
                                return Action.ALL_ACTIONS[5], 5
                            return Action.ALL_ACTIONS[4], 4
                        else:
                            diff_x = adj[0] - pos[0]
                            diff_y = adj[1] - pos[1]
                            if diff_x == 0 and diff_y == 0:
                                return Action.ALL_ACTIONS[4], 4
                            if abs(diff_x) >= abs(diff_y):
                                idx = 2 if diff_x > 0 else 3
                            else:
                                idx = 1 if diff_y > 0 else 0
                            return Action.ALL_ACTIONS[idx], idx

        # Everything else: standard GHM
        action, _ = self._ghm.action(state)
        return action, Action.ACTION_TO_INDEX[action]


class RobustEgo:
    """GHM + aggressive partner avoidance for erratic partners.

    Avoids stepping into partner's current position AND adjacent cells.
    When partner is within distance 1, ego STAYs to let partner clear
    before continuing. This wastes time but prevents the repeated
    collision deadlocks that erratic partners cause.

    Trade-off: +resilience against erratic (avoids collision chains),
    -efficiency against reliable/lazy (unnecessary waiting near partner).
    """
    name = "robust"

    def __init__(self, mlam):
        self._ghm = GreedyHumanModel(mlam)
        self._ghm.set_agent_index(0)

    def get_action(self, state):
        ego_pos = state.players[0].position
        partner_pos = state.players[1].position

        action, _ = self._ghm.action(state)
        idx = Action.ACTION_TO_INDEX[action]

        # INTERACT always executes (we need to pick up/place items)
        if action == "interact":
            return action, idx

        # If moving, check if we'd collide or get too close to partner
        if action != (0, 0):
            next_pos = (ego_pos[0] + action[0], ego_pos[1] + action[1])

            # Block: would land on partner
            if next_pos == partner_pos:
                # Try perpendicular directions (farther from partner)
                best_alt = None
                best_dist = -1
                for alt_idx in range(4):
                    alt = Action.ALL_ACTIONS[alt_idx]
                    if isinstance(alt, tuple) and alt_idx != idx:
                        alt_next = (ego_pos[0] + alt[0], ego_pos[1] + alt[1])
                        d = abs(alt_next[0] - partner_pos[0]) + abs(
                            alt_next[1] - partner_pos[1])
                        if d > best_dist:
                            best_dist = d
                            best_alt = alt_idx
                if best_alt is not None and best_dist > 0:
                    return Action.ALL_ACTIONS[best_alt], best_alt
                return Action.ALL_ACTIONS[4], 4  # STAY

            # Caution: would move adjacent to partner — STAY instead
            dist_after = (abs(next_pos[0] - partner_pos[0])
                          + abs(next_pos[1] - partner_pos[1]))
            dist_now = (abs(ego_pos[0] - partner_pos[0])
                        + abs(ego_pos[1] - partner_pos[1]))
            if dist_after <= 1 and dist_now > 1:
                return Action.ALL_ACTIONS[4], 4  # wait for partner to move

        return action, idx


# V2 strategy registry
V2_EGO_STRATEGIES = {
    "trusting": TrustingEgo,
    "self_sufficient": SelfSufficientEgo,
    "protective": ProtectiveEgo,
    "robust": RobustEgo,
}

# V2 counter-strategy mapping
# Non-bijective: 3/4 types map to self_sufficient, saboteur maps to protective.
# Bijective mapping is infeasible in standard Overcooked because GHM is
# near-optimal — no specialization beats it except pot-camping vs saboteur.
# Use hard_detection_rate metric for meaningful evaluation.
V2_COUNTER_STRATEGY = {
    0: "self_sufficient",  # RELIABLE → GHM handles efficiently
    1: "self_sufficient",  # LAZY → ego handles everything alone
    2: "protective",       # SABOTEUR → camp pot to prevent theft
    3: "self_sufficient",  # ERRATIC → GHM handles collisions OK
}


# ── Model inference wrapper ─────────────────────────────────────────

class OvercookedModelWrapper:
    """Wraps a pretrained model for step-by-step inference in closed-loop.

    Uses fixed-window inference (W=50) with left-padding to ensure the model
    always sees sequences of consistent length, matching training distribution.
    This prevents instability from growing-sequence inference where early steps
    (T=1..20) produce out-of-distribution inputs.
    """

    WINDOW_SIZE = 50  # Fixed inference window

    def __init__(self, model, device, max_steps=200):
        self.model = model
        self.device = device
        self.max_steps = max_steps
        self.obs_dim = model.config.obs_dim

        self.obs_buffer = np.zeros((max_steps, self.obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros(max_steps, dtype=np.int64)
        self.step = 0

    def reset(self):
        self.obs_buffer[:] = 0
        self.action_buffer[:] = 0
        self.step = 0

    def update(self, obs_vec, partner_action_idx):
        """
        Add observation + action, run model inference with sliding window.

        Uses a fixed window of W=50 steps. At step t<W, left-pads with zeros
        so the model always processes length-W sequences.

        Args:
            obs_vec: np.ndarray [obs_dim] (192 for Overcooked)
            partner_action_idx: int (0-5)

        Returns:
            dict with pred_type (int), switch_prob (float), type_probs (ndarray)
        """
        if self.step >= self.max_steps:
            n = self.model.config.num_types
            probs = np.zeros(n)
            probs[-1] = 1.0
            return {'pred_type': n - 1, 'switch_prob': 0.0, 'type_probs': probs}

        self.obs_buffer[self.step] = obs_vec
        self.action_buffer[self.step] = partner_action_idx
        self.step += 1

        W = self.WINDOW_SIZE
        t = self.step

        # Build fixed-length window with left-padding
        obs_window = np.zeros((W, self.obs_dim), dtype=np.float32)
        act_window = np.zeros(W, dtype=np.int64)

        if t >= W:
            # Enough history: use last W steps
            obs_window[:] = self.obs_buffer[t - W:t]
            act_window[:] = self.action_buffer[t - W:t]
        else:
            # Not enough: left-pad with zeros, place data at end
            obs_window[W - t:] = self.obs_buffer[:t]
            act_window[W - t:] = self.action_buffer[:t]

        obs_t = torch.from_numpy(obs_window).unsqueeze(0).float().to(self.device)
        act_t = torch.from_numpy(act_window).unsqueeze(0).long().to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(obs_t, act_t)

        # Always read last position (rightmost = most recent data)
        type_logits = outputs.type_logits[0, -1].cpu()
        type_probs = torch.softmax(type_logits, dim=0).numpy()
        switch_prob = outputs.switch_probs[0, -1].item()

        return {
            'pred_type': int(type_logits.argmax().item()),
            'switch_prob': switch_prob,
            'type_probs': type_probs,
        }


# ── Adaptive ego policy ────────────────────────────────────────────

class OvercookedAdaptiveEgo:
    """
    Selects ego strategy based on model predictions or experimental condition.

    Conditions:
        oracle:          Use ground truth type + switch
        ua_tom:          Use pretrained UA-ToM
        gru:             Use pretrained GRU baseline
        no_detect:       Fixed initial strategy, never adapt
        random:          Random strategy switch (one per episode)
        always_coordinated: Fixed coordinated strategy
    """

    def __init__(
        self,
        condition='oracle',
        model=None,
        device=None,
        mlam=None,
        switch_threshold=0.3,
        cooldown=10,
        type_confidence_gate=0.6,
        type_change_confirm=3,
        max_steps=200,
        v2=False,
    ):
        self.condition = condition
        self.switch_threshold = switch_threshold
        self.cooldown = cooldown
        self.type_confidence_gate = type_confidence_gate
        self.type_change_confirm = type_change_confirm  # consecutive steps to confirm type change
        self.max_steps = max_steps
        self.mlam = mlam
        self.v2 = v2

        # Select strategy/counter registries based on version
        if v2:
            self._ego_strategies = V2_EGO_STRATEGIES
            self._counter_strategy = V2_COUNTER_STRATEGY
        else:
            self._ego_strategies = EGO_STRATEGIES
            self._counter_strategy = COUNTER_STRATEGY

        # Model wrapper
        self.model_wrapper = None
        if model is not None and condition not in ('oracle', 'no_detect', 'random', 'always_coordinated'):
            self.model_wrapper = OvercookedModelWrapper(model, device, max_steps)

        # State
        self.current_strategy_name = 'coordinated' if not v2 else 'self_sufficient'
        self.current_strategy = None
        self.steps_since_switch = 0
        self.step_count = 0

        # Type-change tracking: count consecutive steps predicting a different type
        self._last_pred_type = None
        self._type_change_counter = 0

        # Random condition state
        self._random_switch_time = -1
        self._random_switch_type = 3

    def reset(self, initial_type=None):
        """Reset for new episode."""
        self.step_count = 0
        self.steps_since_switch = 0
        self._last_pred_type = None
        self._type_change_counter = 0

        if self.model_wrapper is not None:
            self.model_wrapper.reset()

        # Set initial strategy
        default_strategy = 'self_sufficient' if self.v2 else 'independent'
        if self.condition == 'always_coordinated':
            self._set_strategy('coordinated' if not self.v2 else 'self_sufficient')
        elif initial_type is not None:
            counter = self._counter_strategy.get(initial_type, default_strategy)
            self._set_strategy(counter)
        else:
            self._set_strategy(default_strategy)

        # Random condition setup
        if self.condition == 'random':
            self._random_switch_time = np.random.randint(0, self.max_steps)
            self._random_switch_type = np.random.choice([0, 1, 2, 3])

    def _set_strategy(self, name):
        if name != self.current_strategy_name or self.current_strategy is None:
            self.current_strategy_name = name
            self.current_strategy = self._ego_strategies[name](self.mlam)
            self.steps_since_switch = 0

    def _type_for_strategy(self, strategy_name):
        """Reverse lookup: which partner type does this counter-strategy target?

        Returns the 'canonical' type for each strategy. For strategies that
        are optimal against multiple types (e.g. 'independent' for 0/1/3),
        returns the lowest-numbered type so that a model currently on that
        strategy will only trigger a switch when it predicts a type whose
        counter differs.
        """
        if self.v2:
            _STRATEGY_TO_TYPE = {
                'self_sufficient': 0,  # reliable / lazy / erratic (canonical: 0)
                'protective': 2,       # saboteur
                'trusting': 0,         # (unused in current mapping)
                'robust': 3,           # (unused in current mapping)
            }
        else:
            _STRATEGY_TO_TYPE = {
                'independent': 0,  # cooperative / greedy / random (canonical: 0)
                'preemptive': 2,   # adversarial
                'coordinated': 0,  # (unused in current mapping)
                'passive': 3,      # (unused in current mapping)
            }
        return _STRATEGY_TO_TYPE.get(strategy_name, -1)

    def update(self, obs_vec=None, partner_action_idx=None,
               ground_truth_type=None, ground_truth_switch=False):
        """
        Update belief after each step.

        Args:
            obs_vec: np.ndarray [192] (for model conditions)
            partner_action_idx: int 0-5 (for model conditions)
            ground_truth_type: int (for oracle)
            ground_truth_switch: bool (for oracle)
        """
        self.step_count += 1
        self.steps_since_switch += 1

        default_strategy = 'self_sufficient' if self.v2 else 'independent'

        if self.condition == 'oracle':
            if ground_truth_switch and ground_truth_type is not None:
                counter = self._counter_strategy.get(ground_truth_type, default_strategy)
                self._set_strategy(counter)

        elif self.model_wrapper is not None:
            if obs_vec is not None:
                pred = self.model_wrapper.update(obs_vec, partner_action_idx)
                pred_type = pred['pred_type']
                type_conf = pred['type_probs'].max()

                # Track consecutive type predictions that differ from current
                current_expected_type = self._type_for_strategy(
                    self.current_strategy_name)
                if pred_type != current_expected_type and type_conf > self.type_confidence_gate:
                    self._type_change_counter += 1
                else:
                    self._type_change_counter = 0
                self._last_pred_type = pred_type

                # Switch on EITHER trigger (with cooldown):
                # 1) switch_prob spike + confidence gate
                # 2) sustained type change for N consecutive steps
                should_switch = False
                if self.steps_since_switch >= self.cooldown:
                    if (pred['switch_prob'] > self.switch_threshold
                            and type_conf > self.type_confidence_gate):
                        should_switch = True
                    elif self._type_change_counter >= self.type_change_confirm:
                        should_switch = True

                if should_switch:
                    counter = self._counter_strategy.get(pred_type, default_strategy)
                    self._set_strategy(counter)
                    self._type_change_counter = 0

        elif self.condition == 'no_detect':
            pass

        elif self.condition == 'random':
            if self.step_count == self._random_switch_time:
                counter = self._counter_strategy.get(
                    self._random_switch_type, default_strategy)
                self._set_strategy(counter)

        elif self.condition == 'always_coordinated':
            pass

    def get_action(self, state):
        """Get ego action from current strategy."""
        return self.current_strategy.get_action(state)
