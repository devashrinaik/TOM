"""
Data collection for Overcooked partner modeling experiments.

Collects episodes with partner type switches in the same .npz format
as ManiSkill data, directly compatible with PartnerDataset.

Output format:
    observations:    [N, T, 192]  float32  (ego_feats ++ partner_feats)
    partner_actions: [N, T]       int64    (0-5)
    partner_types:   [N, T]       int64    (0-3)
    switch_times:    [N]          int64
"""

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.agents.agent import GreedyHumanModel

from .partner_strategies import PARTNER_TYPES, NUM_PARTNER_TYPES


DEFAULT_MLAM_PARAMS = {
    "counter_goals": [],   # filled in at runtime
    "counter_drop": [],
    "counter_pickup": [],
    "wait_allowed": True,
    "start_orientations": False,
    "same_motion_goals": True,
}


class OvercookedEpisodeCollector:
    """
    Collects episodes with partner type switches for training UA-ToM.

    Ego uses GreedyHumanModel (agent_index=0).
    Partner type switches mid-episode at a random time in switch_range.
    """

    def __init__(self, layout="cramped_room", horizon=200):
        self.layout = layout
        self.horizon = horizon

        # Create MDP and env
        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)

        # MLAM for featurization and GreedyHumanModel
        params = DEFAULT_MLAM_PARAMS.copy()
        params["counter_goals"] = self.mdp.get_counter_locations()
        params["counter_drop"] = self.mdp.get_counter_locations()
        params["counter_pickup"] = self.mdp.get_counter_locations()
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(
            self.mdp, params, force_compute=False,
        )

        # Ego policy
        self.ego = GreedyHumanModel(self.mlam)
        self.ego.set_agent_index(0)

        # Observation dim: 96 (ego) + 96 (partner) = 192
        self.obs_dim = 192

    def _featurize(self, state):
        """
        Featurize state from ego (player 0) perspective.

        Returns: np.ndarray [192] = concat(ego_feats, partner_feats)
        """
        feats = self.mdp.featurize_state(state, self.mlam)
        # feats is a list of 2 arrays, each (96,)
        ego_feats = np.array(feats[0], dtype=np.float32)
        partner_feats = np.array(feats[1], dtype=np.float32)
        return np.concatenate([ego_feats, partner_feats])

    def collect_episode(
        self,
        initial_type,
        switch_to_type=None,
        switch_range=(40, 120),
        episode_length=None,
    ):
        """
        Collect a single episode.

        Args:
            initial_type: int, initial partner type index (0-3)
            switch_to_type: int or None, type to switch to (None = no switch)
            switch_range: (min, max) range for random switch time
            episode_length: override horizon (default: self.horizon)

        Returns:
            dict with observations, partner_actions, partner_types, switch_time
        """
        T = episode_length or self.horizon

        observations = np.zeros((T, self.obs_dim), dtype=np.float32)
        partner_actions = np.zeros(T, dtype=np.int64)
        partner_types = np.zeros(T, dtype=np.int64)

        # Determine switch time
        if switch_to_type is not None:
            switch_time = np.random.randint(switch_range[0], switch_range[1])
        else:
            switch_time = T  # no switch

        # Create initial partner
        partner = PARTNER_TYPES[initial_type]()

        # Reset env (OvercookedEnv.reset() returns None; use env.state)
        self.env.reset()
        state = self.env.state
        partner.reset(state, self.mdp, self.mlam)

        # Re-create ego for fresh episode
        self.ego = GreedyHumanModel(self.mlam)
        self.ego.set_agent_index(0)

        for t in range(T):
            # Check for partner switch
            if t == switch_time and switch_to_type is not None:
                partner = PARTNER_TYPES[switch_to_type]()
                partner.reset(state, self.mdp, self.mlam)

            current_type = initial_type if t < switch_time else switch_to_type
            if current_type is None:
                current_type = initial_type

            # Record observation
            observations[t] = self._featurize(state)
            partner_types[t] = current_type

            # Get actions
            ego_action, _ = self.ego.action(state)
            partner_action, partner_action_idx = partner.get_action(state)
            partner_actions[t] = partner_action_idx

            # Step environment
            joint_action = (ego_action, partner_action)
            try:
                state, reward, done, info = self.env.step(joint_action)
            except Exception:
                # If env errors (e.g., collision), fill remaining with zeros
                break

            if done:
                # Fill remaining timesteps with last values
                for t2 in range(t + 1, T):
                    observations[t2] = observations[t]
                    partner_actions[t2] = partner_actions[t]
                    partner_types[t2] = current_type
                break

        return {
            "observations": observations,
            "partner_actions": partner_actions,
            "partner_types": partner_types,
            "switch_time": switch_time if switch_to_type is not None else -1,
        }

    def collect_dataset(
        self,
        episodes_per_pair=25,
        episode_length=None,
        switch_range=(40, 120),
        include_no_switch=True,
        seed=42,
        verbose=True,
    ):
        """
        Collect a full dataset with all partner type transitions.

        Args:
            episodes_per_pair: episodes per (initial_type, switch_type) pair
            episode_length: episode length (default: self.horizon)
            switch_range: (min, max) for switch time
            include_no_switch: also collect episodes without switches
            seed: random seed
            verbose: print progress

        Returns:
            dict with all episode data stacked as arrays
        """
        np.random.seed(seed)

        all_obs = []
        all_actions = []
        all_types = []
        all_switch_times = []

        T = episode_length or self.horizon
        type_indices = list(range(NUM_PARTNER_TYPES))

        # Episodes WITH switches (all initial->switch pairs where initial != switch)
        pairs = [(i, j) for i in type_indices for j in type_indices if i != j]
        total_switch = len(pairs) * episodes_per_pair

        # Episodes WITHOUT switches
        total_no_switch = len(type_indices) * episodes_per_pair if include_no_switch else 0

        total = total_switch + total_no_switch
        if verbose:
            print(f"Collecting {total} episodes "
                  f"({total_switch} with switch, {total_no_switch} without)")

        count = 0

        # No-switch episodes
        if include_no_switch:
            for type_idx in type_indices:
                for ep in range(episodes_per_pair):
                    data = self.collect_episode(
                        initial_type=type_idx,
                        switch_to_type=None,
                        episode_length=T,
                    )
                    all_obs.append(data["observations"])
                    all_actions.append(data["partner_actions"])
                    all_types.append(data["partner_types"])
                    all_switch_times.append(data["switch_time"])
                    count += 1
                    if verbose and count % 50 == 0:
                        print(f"  {count}/{total} episodes collected")

        # Switch episodes
        for (init_type, switch_type) in pairs:
            for ep in range(episodes_per_pair):
                data = self.collect_episode(
                    initial_type=init_type,
                    switch_to_type=switch_type,
                    switch_range=switch_range,
                    episode_length=T,
                )
                all_obs.append(data["observations"])
                all_actions.append(data["partner_actions"])
                all_types.append(data["partner_types"])
                all_switch_times.append(data["switch_time"])
                count += 1
                if verbose and count % 50 == 0:
                    print(f"  {count}/{total} episodes collected")

        # Stack into arrays
        result = {
            "observations": np.stack(all_obs),           # [N, T, 192]
            "partner_actions": np.stack(all_actions),     # [N, T]
            "partner_types": np.stack(all_types),         # [N, T]
            "switch_times": np.array(all_switch_times),   # [N]
        }

        if verbose:
            print(f"\nDataset shapes:")
            for k, v in result.items():
                print(f"  {k}: {v.shape} ({v.dtype})")
            print(f"  Action range: [{result['partner_actions'].min()}, "
                  f"{result['partner_actions'].max()}]")
            print(f"  Type range: [{result['partner_types'].min()}, "
                  f"{result['partner_types'].max()}]")

        return result
