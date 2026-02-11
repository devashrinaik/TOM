"""
Closed-loop evaluation for Overcooked.

Runs episodes with adaptive ego policy, measuring:
- Cumulative reward (soups served)
- UA-ToM detection metrics (type accuracy, switch detection latency)
- Strategy-specific metrics

Conditions:
  oracle:          Ground truth type + switch → upper bound
  ua_tom:          UA-ToM model predictions
  gru:             GRU baseline predictions
  no_detect:       Fixed initial strategy → lower bound
  random:          Random strategy switch → stochastic baseline
  always_coordinated: Fixed coordinated strategy
"""

import numpy as np
from collections import defaultdict

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager

from .partner_strategies import PARTNER_TYPES, PARTNER_NAMES
from .partner_strategies_v2 import PARTNER_TYPES_V2, PARTNER_NAMES_V2
from .ego_policy import OvercookedAdaptiveEgo
from .data_collector import DEFAULT_MLAM_PARAMS


def run_closed_loop_episode(
    env,
    mdp,
    mlam,
    ego_policy,
    initial_type,
    switch_to_type=None,
    switch_time=80,
    max_steps=200,
    v2=False,
):
    """
    Run a single closed-loop episode.

    Args:
        env: OvercookedEnv
        mdp: OvercookedGridworld
        mlam: MediumLevelActionManager
        ego_policy: OvercookedAdaptiveEgo
        initial_type: int, starting partner type
        switch_to_type: int or None
        switch_time: int, when to switch partner
        max_steps: int
        v2: bool, use v2 delayed-revelation partners

    Returns:
        dict with per-step data and episode metrics
    """
    # Create partner
    _partner_types = PARTNER_TYPES_V2 if v2 else PARTNER_TYPES
    partner = _partner_types[initial_type]()

    # Reset (OvercookedEnv.reset() returns None; use env.state)
    env.reset()
    state = env.state
    partner.reset(state, mdp, mlam)
    ego_policy.reset(initial_type=initial_type)

    # Tracking
    rewards = []
    partner_types_actual = []
    ego_strategies = []
    partner_actions_log = []
    ego_actions_log = []
    cumulative_reward = 0

    for t in range(max_steps):
        # Check for partner switch
        is_switch = (t == switch_time and switch_to_type is not None)
        if is_switch:
            partner = _partner_types[switch_to_type]()
            partner.reset(state, mdp, mlam)

        current_type = initial_type if (switch_to_type is None or t < switch_time) else switch_to_type

        # Featurize for model
        feats = mdp.featurize_state(state, mlam)
        obs_vec = np.concatenate([
            np.array(feats[0], dtype=np.float32),
            np.array(feats[1], dtype=np.float32),
        ])

        # Get actions
        ego_action, ego_idx = ego_policy.get_action(state)
        partner_action, partner_idx = partner.get_action(state)

        # Log
        partner_types_actual.append(current_type)
        ego_strategies.append(ego_policy.current_strategy_name)
        partner_actions_log.append(partner_idx)
        ego_actions_log.append(ego_idx)

        # Step
        joint_action = (ego_action, partner_action)
        try:
            state, reward, done, info = env.step(joint_action)
        except Exception:
            reward = 0
            done = True

        rewards.append(reward)
        cumulative_reward += reward

        # Update ego belief
        ego_policy.update(
            obs_vec=obs_vec,
            partner_action_idx=partner_idx,
            ground_truth_type=current_type,
            ground_truth_switch=is_switch,
        )

        if done:
            break

    # Compute detection metrics
    detection_latency = None
    if switch_to_type is not None:
        # Find when ego strategy actually changed after switch
        for t_detect in range(switch_time, len(ego_strategies)):
            expected = _counter_for_type(switch_to_type, v2=v2)
            if ego_strategies[t_detect] == expected:
                detection_latency = t_detect - switch_time
                break
        if detection_latency is None:
            detection_latency = max_steps  # never detected

    return {
        "rewards": rewards,
        "cumulative_reward": cumulative_reward,
        "partner_types": partner_types_actual,
        "ego_strategies": ego_strategies,
        "partner_actions": partner_actions_log,
        "ego_actions": ego_actions_log,
        "initial_type": initial_type,
        "switch_to_type": switch_to_type,
        "switch_time": switch_time,
        "detection_latency": detection_latency,
        "episode_length": len(rewards),
    }


def _counter_for_type(type_idx, v2=False):
    """Get expected counter-strategy name for a partner type."""
    if v2:
        from .ego_policy import V2_COUNTER_STRATEGY
        return V2_COUNTER_STRATEGY.get(type_idx, "self_sufficient")
    else:
        from .ego_policy import COUNTER_STRATEGY
        return COUNTER_STRATEGY.get(type_idx, "independent")


def run_evaluation(
    conditions,
    layout="cramped_room",
    n_episodes=20,
    switch_range=(40, 120),
    max_steps=200,
    seed=42,
    verbose=True,
    v2=False,
):
    """
    Run full closed-loop evaluation across conditions.

    Args:
        conditions: dict mapping condition_name → (model_or_None, device_or_None)
            e.g., {"oracle": (None, None), "ua_tom": (model, device), ...}
        layout: Overcooked layout
        n_episodes: episodes per (initial_type, switch_to_type) pair
        switch_range: (min, max) for random switch time
        max_steps: episode length
        seed: random seed
        verbose: print progress

    Returns:
        dict mapping condition → aggregated metrics
    """
    np.random.seed(seed)

    # Create env
    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=max_steps)

    params = DEFAULT_MLAM_PARAMS.copy()
    params["counter_goals"] = mdp.get_counter_locations()
    params["counter_drop"] = mdp.get_counter_locations()
    params["counter_pickup"] = mdp.get_counter_locations()
    mlam = MediumLevelActionManager.from_pickle_or_compute(
        mdp, params, force_compute=False,
    )

    # Select partner names for logging
    _pnames = PARTNER_NAMES_V2 if v2 else PARTNER_NAMES

    # Define transition pairs
    type_indices = [0, 1, 2, 3]
    switch_pairs = [(i, j) for i in type_indices for j in type_indices if i != j]

    results = {}

    for cond_name, (model, device) in conditions.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Condition: {cond_name}")
            print(f"{'='*50}")

        ego_policy = OvercookedAdaptiveEgo(
            condition=cond_name,
            model=model,
            device=device,
            mlam=mlam,
            max_steps=max_steps,
            v2=v2,
        )

        all_episodes = []

        for (init_type, switch_type) in switch_pairs:
            for ep in range(n_episodes):
                switch_time = np.random.randint(switch_range[0], switch_range[1])

                episode_data = run_closed_loop_episode(
                    env, mdp, mlam, ego_policy,
                    initial_type=init_type,
                    switch_to_type=switch_type,
                    switch_time=switch_time,
                    max_steps=max_steps,
                    v2=v2,
                )
                all_episodes.append(episode_data)

        # Aggregate metrics
        cum_rewards = [e["cumulative_reward"] for e in all_episodes]
        latencies = [e["detection_latency"] for e in all_episodes
                     if e["detection_latency"] is not None]

        # Hard detection: only transitions where counter-strategy actually changes
        hard_latencies = [
            e["detection_latency"] for e in all_episodes
            if e["detection_latency"] is not None
            and _counter_for_type(e["initial_type"], v2=v2) != _counter_for_type(e["switch_to_type"], v2=v2)
        ]

        # Pre/post switch reward comparison
        pre_switch_rewards = []
        post_switch_rewards = []
        for e in all_episodes:
            st = e["switch_time"]
            r = e["rewards"]
            if st < len(r):
                pre_switch_rewards.append(sum(r[:st]))
                post_switch_rewards.append(sum(r[st:]))

        # Per-transition breakdown
        transition_rewards = defaultdict(list)
        for e in all_episodes:
            key = (e["initial_type"], e["switch_to_type"])
            transition_rewards[key].append(e["cumulative_reward"])

        metrics = {
            "mean_reward": np.mean(cum_rewards),
            "std_reward": np.std(cum_rewards),
            "median_reward": np.median(cum_rewards),
            "mean_detection_latency": np.mean(latencies) if latencies else float('inf'),
            "median_detection_latency": np.median(latencies) if latencies else float('inf'),
            "detection_rate": sum(1 for l in latencies if l < max_steps) / max(len(latencies), 1),
            "hard_detection_rate": (
                sum(1 for l in hard_latencies if l < max_steps) / max(len(hard_latencies), 1)
                if hard_latencies else float('nan')
            ),
            "hard_detection_n": len(hard_latencies),
            "mean_pre_switch_reward": np.mean(pre_switch_rewards) if pre_switch_rewards else 0,
            "mean_post_switch_reward": np.mean(post_switch_rewards) if post_switch_rewards else 0,
            "n_episodes": len(all_episodes),
            "transition_rewards": {
                f"{_pnames[k[0]]}→{_pnames[k[1]]}": {
                    "mean": np.mean(v),
                    "std": np.std(v),
                }
                for k, v in transition_rewards.items()
            },
        }

        results[cond_name] = metrics

        if verbose:
            print(f"  Mean reward: {metrics['mean_reward']:.1f} "
                  f"(±{metrics['std_reward']:.1f})")
            print(f"  Detection rate: {metrics['detection_rate']*100:.0f}%")
            hard_dr = metrics['hard_detection_rate']
            hard_str = f"{hard_dr*100:.0f}%" if not np.isnan(hard_dr) else "N/A"
            print(f"  Hard detection rate: {hard_str} "
                  f"(n={metrics['hard_detection_n']})")
            print(f"  Mean detection latency: {metrics['mean_detection_latency']:.1f}")
            print(f"  Pre-switch reward: {metrics['mean_pre_switch_reward']:.1f}")
            print(f"  Post-switch reward: {metrics['mean_post_switch_reward']:.1f}")

    return results


def print_results_table(results):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print("CLOSED-LOOP EVALUATION RESULTS")
    print(f"{'='*90}")
    print(f"{'Condition':<18s} {'Reward':>10s} {'Det.Rate':>10s} "
          f"{'HardDet':>10s} {'Latency':>10s} {'PreSwR':>10s} {'PostSwR':>10s}")
    print("-" * 80)

    for cond, m in results.items():
        det_rate = f"{m['detection_rate']*100:.0f}%"
        hard_dr = m.get('hard_detection_rate', float('nan'))
        hard_str = f"{hard_dr*100:.0f}%" if not np.isnan(hard_dr) else "N/A"
        latency = f"{m['mean_detection_latency']:.1f}" \
            if m['mean_detection_latency'] < float('inf') else "N/A"
        print(f"{cond:<18s} {m['mean_reward']:>9.1f} "
              f"{det_rate:>10s} {hard_str:>10s} {latency:>10s} "
              f"{m['mean_pre_switch_reward']:>9.1f} "
              f"{m['mean_post_switch_reward']:>9.1f}")
