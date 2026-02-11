#!/usr/bin/env python3
"""
Validate V2 Overcooked strategies: delayed-revelation partners × v2 ego strategies.

Runs a 4×4 reward matrix (v2 ego strategies × v2 partner types) and checks
whether the COUNTER_STRATEGY mapping is bijective-optimal (each partner type
has a unique best counter-strategy on the diagonal).

Usage:
    python scripts/validate_v2_strategies.py
    python scripts/validate_v2_strategies.py --episodes 30 --layout cramped_room
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

from ua_tom.overcooked.partner_strategies_v2 import (
    PARTNER_TYPES_V2, PARTNER_NAMES_V2,
)
from ua_tom.overcooked.ego_policy import V2_EGO_STRATEGIES, V2_COUNTER_STRATEGY
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

        try:
            state, reward, done, info = env.step((ego_action, partner_action))
        except Exception:
            reward = 0
            done = True

        cumulative_reward += reward
        if done:
            break

    return cumulative_reward


def run_matrix(layout="cramped_room", n_episodes=10, max_steps=200, seed=42):
    """Run the full v2 ego-strategy × v2 partner-type reward matrix."""
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

    ego_names = list(V2_EGO_STRATEGIES.keys())
    partner_ids = sorted(PARTNER_TYPES_V2.keys())

    results = {e: {p: [] for p in partner_ids} for e in ego_names}

    for ego_name in ego_names:
        for pid in partner_ids:
            for ep in range(n_episodes):
                ego = V2_EGO_STRATEGIES[ego_name](mlam)
                partner = PARTNER_TYPES_V2[pid]()

                reward = run_episode(env, mdp, mlam, ego, partner, max_steps)
                results[ego_name][pid].append(reward)

            mean_r = np.mean(results[ego_name][pid])
            std_r = np.std(results[ego_name][pid])
            is_counter = V2_COUNTER_STRATEGY.get(pid) == ego_name
            marker = " <<<" if is_counter else ""
            print(f"  {ego_name:<16s} vs {PARTNER_NAMES_V2[pid]:<10s}: "
                  f"{mean_r:6.1f} ±{std_r:4.1f} (n={n_episodes}){marker}")

    return results


def print_results(results):
    """Print the reward matrix and bijective analysis."""
    ego_names = list(V2_EGO_STRATEGIES.keys())
    partner_ids = sorted(PARTNER_TYPES_V2.keys())

    # ── Reward matrix ──
    print("\n" + "=" * 80)
    print("V2 REWARD MATRIX — rows=ego strategy, cols=partner type")
    print("=" * 80)

    col_label = "Ego \\ Partner"
    header = f"{col_label:<18s}"
    for pid in partner_ids:
        header += f"  {PARTNER_NAMES_V2[pid]:>12s}"
    header += "       Avg"
    print(header)
    print("-" * 80)

    # Track column maxima for bijective check
    col_max = {}
    col_max_ego = {}
    for pid in partner_ids:
        best_val = -float('inf')
        best_ego = None
        for ego_name in ego_names:
            mean_r = np.mean(results[ego_name][pid])
            if mean_r > best_val:
                best_val = mean_r
                best_ego = ego_name
        col_max[pid] = best_val
        col_max_ego[pid] = best_ego

    for ego_name in ego_names:
        row = f"{ego_name:<18s}"
        vals = []
        for pid in partner_ids:
            mean_r = np.mean(results[ego_name][pid])
            vals.append(mean_r)
            is_counter = V2_COUNTER_STRATEGY.get(pid) == ego_name
            is_col_max = col_max_ego[pid] == ego_name
            if is_counter and is_col_max:
                marker = "*"  # correct and is max
            elif is_counter:
                marker = "!"  # designated counter but NOT max
            elif is_col_max:
                marker = "^"  # actual max but not designated
            else:
                marker = " "
            row += f"  {mean_r:>11.1f}{marker}"
        avg = np.mean(vals)
        row += f"  {avg:>8.1f}"
        print(row)

    print("-" * 80)
    print("  * = designated counter AND column max (correct)")
    print("  ! = designated counter but NOT column max (PROBLEM)")
    print("  ^ = column max but not designated counter (PROBLEM)")

    # ── Bijective check ──
    print("\n" + "=" * 80)
    print("BIJECTIVE OPTIMALITY CHECK")
    print("=" * 80)

    bijective = True
    for pid in partner_ids:
        expected = V2_COUNTER_STRATEGY[pid]
        actual_best = col_max_ego[pid]
        expected_val = np.mean(results[expected][pid])
        actual_val = col_max[pid]
        gap = expected_val - actual_val if expected != actual_best else 0
        second_best_val = -float('inf')
        for ego_name in ego_names:
            if ego_name != actual_best:
                v = np.mean(results[ego_name][pid])
                if v > second_best_val:
                    second_best_val = v

        margin = actual_val - second_best_val

        status = "OK" if expected == actual_best else "MISMATCH"
        if status == "MISMATCH":
            bijective = False

        print(f"  {PARTNER_NAMES_V2[pid]:>10s}: expected={expected:<16s} "
              f"actual_best={actual_best:<16s} "
              f"margin={margin:+.1f}  [{status}]")

    # Check uniqueness (each strategy is optimal for exactly one type)
    used = set(col_max_ego.values())
    if len(used) < len(partner_ids):
        bijective = False
        print(f"\n  WARNING: Some strategies are optimal for multiple types!")
        for ego_name in ego_names:
            types = [pid for pid in partner_ids if col_max_ego[pid] == ego_name]
            if len(types) > 1:
                type_names = [PARTNER_NAMES_V2[p] for p in types]
                print(f"    {ego_name} dominates: {', '.join(type_names)}")

    if bijective:
        print(f"\n  >>> BIJECTIVE OPTIMALITY ACHIEVED")
    else:
        print(f"\n  >>> NOT BIJECTIVE — strategies need adjustment")

    # ── Per-type gap analysis ──
    print("\n" + "=" * 80)
    print("PER-TYPE REWARD GAPS (column max - 2nd best)")
    print("=" * 80)

    for pid in partner_ids:
        vals = [(np.mean(results[e][pid]), e) for e in ego_names]
        vals.sort(reverse=True)
        gap = vals[0][0] - vals[1][0]
        se = np.std(results[vals[0][1]][pid]) / np.sqrt(len(results[vals[0][1]][pid]))
        sig = gap / (2 * se) if se > 0 else float('inf')
        print(f"  {PARTNER_NAMES_V2[pid]:>10s}: best={vals[0][1]:<16s} "
              f"({vals[0][0]:.1f}), 2nd={vals[1][1]:<16s} ({vals[1][0]:.1f}), "
              f"gap={gap:.1f}, ~{sig:.1f}σ")

    # ── Standard errors ──
    print("\n" + "=" * 80)
    print("STANDARD ERRORS (for significance assessment)")
    print("=" * 80)
    for ego_name in ego_names:
        row = f"{ego_name:<18s}"
        for pid in partner_ids:
            se = np.std(results[ego_name][pid]) / np.sqrt(len(results[ego_name][pid]))
            row += f"  {se:>12.1f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description='Validate V2 Overcooked strategies')
    parser.add_argument('--layout', type=str, default='cramped_room')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Episodes per ego-partner combination')
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 80)
    print("V2 Overcooked Strategy Validation (Delayed-Revelation Partners)")
    print(f"  Layout: {args.layout}")
    print(f"  Episodes per combination: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print("=" * 80)

    print("\n--- 4x4 Reward Matrix ---\n")
    results = run_matrix(
        layout=args.layout,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print_results(results)


if __name__ == '__main__':
    main()
