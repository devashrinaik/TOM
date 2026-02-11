#!/usr/bin/env python3
"""
Validate Overcooked ego strategies × partner types reward matrix.

Runs each of the 4 ego strategies (coordinated, independent, preemptive, passive)
against each of the 4 partner types (cooperative, greedy, adversarial, random)
and prints a reward matrix.

Goal: correct counter-strategy should outperform alternatives for each partner.
Key metric: adaptation gap = oracle_reward - no_detect_reward on cooperative→adversarial.

Usage:
    python scripts/validate_overcooked_strategies.py
    python scripts/validate_overcooked_strategies.py --episodes 20 --layout cramped_room
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Package setup
_pkg_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_pkg_dir.parent))

import importlib, importlib.util
_spec = importlib.util.spec_from_file_location(
    'ua_tom', _pkg_dir / '__init__.py',
    submodule_search_locations=[str(_pkg_dir)],
)
_pkg = importlib.util.module_from_spec(_spec)
sys.modules['ua_tom'] = _pkg
_spec.loader.exec_module(_pkg)

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.agents.agent import GreedyHumanModel

from ua_tom.overcooked.partner_strategies import PARTNER_TYPES, PARTNER_NAMES
from ua_tom.overcooked.ego_policy import EGO_STRATEGIES, COUNTER_STRATEGY
from ua_tom.overcooked.data_collector import DEFAULT_MLAM_PARAMS


def run_episode(env, mdp, mlam, ego_strategy, partner, max_steps=200):
    """Run a single episode and return cumulative reward."""
    env.reset()
    state = env.state
    partner.reset(state, mdp, mlam)

    cumulative_reward = 0
    for t in range(max_steps):
        ego_action, _ = ego_strategy.get_action(state)
        partner_action, _ = partner.get_action(state)

        joint_action = (ego_action, partner_action)
        try:
            state, reward, done, info = env.step(joint_action)
        except Exception:
            reward = 0
            done = True

        cumulative_reward += reward
        if done:
            break

    return cumulative_reward


def run_matrix(layout="cramped_room", n_episodes=10, max_steps=200, seed=42):
    """Run the full ego-strategy × partner-type reward matrix."""
    np.random.seed(seed)

    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=max_steps)

    params = DEFAULT_MLAM_PARAMS.copy()
    params["counter_goals"] = mdp.get_counter_locations()
    params["counter_drop"] = mdp.get_counter_locations()
    params["counter_pickup"] = mdp.get_counter_locations()
    mlam = MediumLevelActionManager.from_pickle_or_compute(
        mdp, params, force_compute=False,
    )

    ego_names = list(EGO_STRATEGIES.keys())
    partner_ids = sorted(PARTNER_TYPES.keys())

    # results[ego_name][partner_id] = list of rewards
    results = {e: {p: [] for p in partner_ids} for e in ego_names}

    total = len(ego_names) * len(partner_ids) * n_episodes
    count = 0

    for ego_name in ego_names:
        for pid in partner_ids:
            for ep in range(n_episodes):
                ego = EGO_STRATEGIES[ego_name](mlam)
                partner = PARTNER_TYPES[pid]()

                reward = run_episode(env, mdp, mlam, ego, partner, max_steps)
                results[ego_name][pid].append(reward)

                count += 1

            mean_r = np.mean(results[ego_name][pid])
            is_optimal = COUNTER_STRATEGY.get(pid) == ego_name
            marker = " <<<" if is_optimal else ""
            print(f"  {ego_name:<14s} vs {PARTNER_NAMES[pid]:<14s}: "
                  f"{mean_r:6.1f} reward (n={n_episodes}){marker}")

    return results


def run_switch_test(layout="cramped_room", n_episodes=10, max_steps=200,
                    switch_time=80, seed=42):
    """
    Test cooperative→adversarial transition with different strategies.

    Simulates the key transition where adaptation matters most.
    """
    np.random.seed(seed)

    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=max_steps)

    params = DEFAULT_MLAM_PARAMS.copy()
    params["counter_goals"] = mdp.get_counter_locations()
    params["counter_drop"] = mdp.get_counter_locations()
    params["counter_pickup"] = mdp.get_counter_locations()
    mlam = MediumLevelActionManager.from_pickle_or_compute(
        mdp, params, force_compute=False,
    )

    # Test scenarios:
    # 1. oracle: independent pre-switch, preemptive post-switch (adapts)
    # 2. no_detect: independent throughout (initial strategy, never adapts)
    # 3. always_preemptive: preemptive throughout
    # 4. always_coordinated: coordinated throughout
    scenarios = {
        "oracle (indep→preempt)":  ("independent", "preemptive"),
        "no_detect (indep→indep)": ("independent", "independent"),
        "always_preemptive":       ("preemptive", "preemptive"),
        "always_coordinated":      ("coordinated", "coordinated"),
    }

    results = {}
    for name, (pre_strat, post_strat) in scenarios.items():
        rewards = []
        for ep in range(n_episodes):
            env.reset()
            state = env.state

            # Pre-switch: cooperative partner
            partner = PARTNER_TYPES[0]()  # cooperative
            partner.reset(state, mdp, mlam)
            ego = EGO_STRATEGIES[pre_strat](mlam)

            cumulative_reward = 0
            for t in range(max_steps):
                # Switch at switch_time
                if t == switch_time:
                    partner = PARTNER_TYPES[2]()  # adversarial
                    partner.reset(state, mdp, mlam)
                    ego = EGO_STRATEGIES[post_strat](mlam)

                ego_action, _ = ego.get_action(state)
                partner_action, _ = partner.get_action(state)

                try:
                    state, reward, done, info = env.step(
                        (ego_action, partner_action))
                except Exception:
                    reward = 0
                    done = True

                cumulative_reward += reward
                if done:
                    break

            rewards.append(cumulative_reward)

        results[name] = rewards

    return results


def print_results(results):
    """Print the reward matrix and analysis."""
    ego_names = list(EGO_STRATEGIES.keys())
    partner_ids = sorted(PARTNER_TYPES.keys())

    # ── Reward matrix ──
    print("\n" + "=" * 75)
    print("REWARD MATRIX — rows=ego strategy, cols=partner type")
    print("=" * 75)

    col_label = "Ego \\ Partner"
    header = f"{col_label:<16s}"
    for pid in partner_ids:
        header += f"  {PARTNER_NAMES[pid]:>12s}"
    header += "       Avg"
    print(header)
    print("-" * 75)

    for ego_name in ego_names:
        row = f"{ego_name:<16s}"
        vals = []
        for pid in partner_ids:
            mean_r = np.mean(results[ego_name][pid])
            vals.append(mean_r)
            is_optimal = COUNTER_STRATEGY.get(pid) == ego_name
            marker = "*" if is_optimal else " "
            row += f"  {mean_r:>11.1f}{marker}"
        avg = np.mean(vals)
        row += f"  {avg:>8.1f}"
        print(row)

    print("-" * 75)
    print("  * = optimal counter-strategy for that partner type\n")

    # ── Diagonal vs off-diagonal ──
    print("=" * 75)
    print("DIAGONAL (correct counter) vs OFF-DIAGONAL (wrong) SUMMARY")
    print("=" * 75)

    for pid in partner_ids:
        optimal_ego = COUNTER_STRATEGY.get(pid)
        opt_reward = np.mean(results[optimal_ego][pid])
        others = []
        for ego_name in ego_names:
            if ego_name != optimal_ego and ego_name != "passive":
                others.append(np.mean(results[ego_name][pid]))
        best_other = max(others) if others else 0
        gap = opt_reward - best_other
        print(f"  {PARTNER_NAMES[pid]:>14s}: optimal={opt_reward:.1f} "
              f"({optimal_ego}), best_other={best_other:.1f}, gap={gap:+.1f}")

    # Overall
    diag_rewards = []
    off_diag_rewards = []
    for pid in partner_ids:
        optimal_ego = COUNTER_STRATEGY.get(pid)
        for ego_name in ego_names:
            if ego_name == "passive":
                continue
            mean_r = np.mean(results[ego_name][pid])
            if ego_name == optimal_ego:
                diag_rewards.append(mean_r)
            else:
                off_diag_rewards.append(mean_r)

    diag_mean = np.mean(diag_rewards)
    offdiag_mean = np.mean(off_diag_rewards)
    gap = diag_mean - offdiag_mean

    print(f"\n  Overall diagonal (correct):   {diag_mean:.1f}")
    print(f"  Overall off-diagonal (wrong): {offdiag_mean:.1f}")
    print(f"  Performance gap:              {gap:+.1f}")

    if gap >= 5:
        print("\n  >>> ADAPTATION GAP EXISTS — strategy detection matters!")
    else:
        print("\n  >>> WEAK/NO GAP — strategies need further differentiation")


def print_switch_results(results):
    """Print cooperative→adversarial transition results."""
    print("\n" + "=" * 75)
    print("COOPERATIVE → ADVERSARIAL TRANSITION TEST")
    print("=" * 75)

    for name, rewards in results.items():
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        print(f"  {name:<30s}: {mean_r:6.1f} +/- {std_r:4.1f}")

    # Check adaptation gap
    oracle_key = [k for k in results if "oracle" in k][0]
    nodetect_key = [k for k in results if "no_detect" in k][0]
    oracle_mean = np.mean(results[oracle_key])
    nodetect_mean = np.mean(results[nodetect_key])
    gap = oracle_mean - nodetect_mean

    print(f"\n  Adaptation gap (oracle - no_detect): {gap:+.1f}")
    if gap >= 10:
        print("  >>> STRONG GAP (>= 10) — closed-loop eval will show clear benefit")
    elif gap >= 5:
        print("  >>> MODERATE GAP (>= 5) — may need more episodes for significance")
    else:
        print("  >>> WEAK GAP (< 5) — need stronger strategy differentiation")


def main():
    parser = argparse.ArgumentParser(
        description='Validate Overcooked ego strategies')
    parser.add_argument('--layout', type=str, default='cramped_room')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Episodes per ego-partner combination')
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--switch_time', type=int, default=80)
    args = parser.parse_args()

    print("=" * 75)
    print("Overcooked Strategy Validation")
    print(f"  Layout: {args.layout}")
    print(f"  Episodes per combination: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print("=" * 75)

    # Part 1: Full reward matrix
    print("\n--- Part 1: Full 4x4 Reward Matrix ---\n")
    results = run_matrix(
        layout=args.layout,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print_results(results)

    # Part 2: Cooperative → Adversarial transition test
    print("\n--- Part 2: Transition Test (cooperative → adversarial) ---\n")
    switch_results = run_switch_test(
        layout=args.layout,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        switch_time=args.switch_time,
        seed=args.seed,
    )
    print_switch_results(switch_results)


if __name__ == '__main__':
    main()
