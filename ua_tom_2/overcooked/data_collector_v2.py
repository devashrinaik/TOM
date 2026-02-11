"""
V2 data collection using delayed-revelation partners.

Same output format as v1 (.npz with observations, partner_actions,
partner_types, switch_times), but uses v2 partners that all start
as GHM and diverge after diverge_time.

Also records diverge_times for post-hoc analysis.
"""

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.agents.agent import GreedyHumanModel

from .partner_strategies_v2 import PARTNER_TYPES_V2, NUM_PARTNER_TYPES_V2
from .data_collector import DEFAULT_MLAM_PARAMS


class OvercookedEpisodeCollectorV2:
    """Collects episodes with v2 delayed-revelation partner types."""

    def __init__(self, layout="cramped_room", horizon=200):
        self.layout = layout
        self.horizon = horizon

        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)

        params = DEFAULT_MLAM_PARAMS.copy()
        params["counter_goals"] = self.mdp.get_counter_locations()
        params["counter_drop"] = self.mdp.get_counter_locations()
        params["counter_pickup"] = self.mdp.get_counter_locations()
        self.mlam = MediumLevelActionManager.from_pickle_or_compute(
            self.mdp, params, force_compute=False,
        )

        self.ego = GreedyHumanModel(self.mlam)
        self.ego.set_agent_index(0)
        self.obs_dim = 192

    def _featurize(self, state):
        feats = self.mdp.featurize_state(state, self.mlam)
        ego_feats = np.array(feats[0], dtype=np.float32)
        partner_feats = np.array(feats[1], dtype=np.float32)
        return np.concatenate([ego_feats, partner_feats])

    def collect_episode(
        self, initial_type, switch_to_type=None,
        switch_range=(40, 120), episode_length=None,
    ):
        T = episode_length or self.horizon

        observations = np.zeros((T, self.obs_dim), dtype=np.float32)
        partner_actions = np.zeros(T, dtype=np.int64)
        partner_types = np.zeros(T, dtype=np.int64)

        if switch_to_type is not None:
            switch_time = np.random.randint(switch_range[0], switch_range[1])
        else:
            switch_time = T

        partner = PARTNER_TYPES_V2[initial_type]()

        self.env.reset()
        state = self.env.state
        partner.reset(state, self.mdp, self.mlam)

        self.ego = GreedyHumanModel(self.mlam)
        self.ego.set_agent_index(0)

        # Track diverge times for both partners
        initial_diverge = partner.diverge_time if hasattr(partner, 'diverge_time') else -1
        switch_diverge = -1

        for t in range(T):
            if t == switch_time and switch_to_type is not None:
                partner = PARTNER_TYPES_V2[switch_to_type]()
                partner.reset(state, self.mdp, self.mlam)
                switch_diverge = partner.diverge_time if hasattr(partner, 'diverge_time') else -1

            current_type = initial_type if t < switch_time else (switch_to_type or initial_type)

            observations[t] = self._featurize(state)
            partner_types[t] = current_type

            ego_action, _ = self.ego.action(state)
            partner_action, partner_action_idx = partner.get_action(state)
            partner_actions[t] = partner_action_idx

            try:
                state, reward, done, info = self.env.step(
                    (ego_action, partner_action))
            except Exception:
                break

            if done:
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
            "initial_diverge_time": initial_diverge,
            "switch_diverge_time": switch_diverge,
        }

    def collect_dataset(
        self, episodes_per_pair=25, episode_length=None,
        switch_range=(40, 120), include_no_switch=True,
        seed=42, verbose=True,
    ):
        np.random.seed(seed)

        all_obs = []
        all_actions = []
        all_types = []
        all_switch_times = []
        all_initial_diverge = []
        all_switch_diverge = []

        T = episode_length or self.horizon
        type_indices = list(range(NUM_PARTNER_TYPES_V2))

        pairs = [(i, j) for i in type_indices for j in type_indices if i != j]
        total_switch = len(pairs) * episodes_per_pair
        total_no_switch = len(type_indices) * episodes_per_pair if include_no_switch else 0
        total = total_switch + total_no_switch

        if verbose:
            print(f"Collecting {total} episodes "
                  f"({total_switch} with switch, {total_no_switch} without)")

        count = 0

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
                    all_initial_diverge.append(data["initial_diverge_time"])
                    all_switch_diverge.append(data["switch_diverge_time"])
                    count += 1
                    if verbose and count % 50 == 0:
                        print(f"  {count}/{total} episodes collected")

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
                all_initial_diverge.append(data["initial_diverge_time"])
                all_switch_diverge.append(data["switch_diverge_time"])
                count += 1
                if verbose and count % 50 == 0:
                    print(f"  {count}/{total} episodes collected")

        result = {
            "observations": np.stack(all_obs),
            "partner_actions": np.stack(all_actions),
            "partner_types": np.stack(all_types),
            "switch_times": np.array(all_switch_times),
            "initial_diverge_times": np.array(all_initial_diverge),
            "switch_diverge_times": np.array(all_switch_diverge),
        }

        if verbose:
            print(f"\nDataset shapes:")
            for k, v in result.items():
                print(f"  {k}: {v.shape} ({v.dtype})")

        return result
