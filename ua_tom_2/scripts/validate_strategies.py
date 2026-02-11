#!/usr/bin/env python3
"""
Phase A Validation: 4x4 ego-strategy vs partner-type matrix.

Runs each of the 4 ego strategies against each of the 4 partner types
in the SharedWorkspacePickCube-v1 environment. Measures:
  - Task success (cube z > 0.15)
  - Time to success (steps)
  - Final cube height

Goal: correct counter-strategy should succeed >80%, wrong strategy <50%.

Usage:
    python scripts/validate_strategies.py
    python scripts/validate_strategies.py --episodes 20 --max_steps 150
"""

import os
os.environ['MANI_SKILL_NO_DISPLAY'] = '1'

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

# Register SharedWorkspacePickCube-v1
sys.path.insert(0, str(Path(__file__).parent))
import test_shared_workspace  # noqa

# Closed-loop strategies (modified partners + ego counter-strategies)
sys.path.insert(0, str(Path(__file__).parent.parent))
from closed_loop.partner_strategies import (
    PartnerType, make_strategy, PARTNER_NAMES,
)
from closed_loop.ego_strategies import (
    ALL_EGO_STRATEGIES, COUNTER_STRATEGY,
)


SUCCESS_Z = 0.15   # cube must be above this z for physical lift
GRASP_READY = 0.04  # ego within this distance of cube = grasp-ready
OPPONENT_CLEAR = 0.06  # opponent must be further than this for grasp-ready


def run_episode(env, ego_strategy, partner_strategy, max_steps=150):
    """
    Run one episode with given ego and partner strategies.

    Hybrid success metric:
    1. Physical lift: cube z > SUCCESS_Z AND (ego holding OR cube on ego side)
    2. Grasp-ready: ego within GRASP_READY of cube AND opponent > OPPONENT_CLEAR from cube
       (ego reached the cube without the opponent blocking — would have grasped)

    Either condition triggers success.
    """
    obs, info = env.reset()
    ego_strategy.reset()
    partner_strategy.reset()

    best_z = 0.0
    success_step = -1
    grasp_ready_steps = 0  # consecutive steps in grasp-ready position

    for step in range(max_steps):
        extra = obs['extra']
        left_tcp = extra['left_arm_tcp'][0, :3].cpu().numpy()
        right_tcp = extra['right_arm_tcp'][0, :3].cpu().numpy()
        cube_pos = extra['cube_pose'][0, :3].cpu().numpy()

        # Ego action
        left_cmd = ego_strategy.get_action(left_tcp, cube_pos, right_tcp)
        # Partner action
        right_cmd = partner_strategy.get_action(right_tcp, cube_pos, left_tcp)

        action = {
            'panda_wristcam-0': torch.tensor(np.array([left_cmd]), dtype=torch.float32),
            'panda_wristcam-1': torch.tensor(np.array([right_cmd]), dtype=torch.float32),
        }
        obs, reward, terminated, truncated, info = env.step(action)

        cube_z = extra['cube_pose'][0, 2].item()
        cube_y = extra['cube_pose'][0, 1].item()
        if cube_z > best_z:
            best_z = cube_z

        ego_dist = float(np.linalg.norm(left_tcp - cube_pos))
        opp_dist = float(np.linalg.norm(right_tcp - cube_pos))

        if success_step < 0:
            # Check 1: Physical lift (cube actually lifted)
            ego_holding = ego_dist < 0.08
            cube_ego_side = cube_y < 0.0
            if cube_z > SUCCESS_Z and (ego_holding or cube_ego_side):
                success_step = step

            # Check 2: Grasp-ready position (ego at cube, opponent not blocking)
            if ego_dist < GRASP_READY and opp_dist > OPPONENT_CLEAR:
                grasp_ready_steps += 1
                if grasp_ready_steps >= 5:  # sustained for 5 steps = reliable grasp
                    success_step = step
            else:
                grasp_ready_steps = 0

        if terminated or truncated:
            break

    # Final state
    extra = obs['extra']
    final_cube = extra['cube_pose'][0, :3].cpu().numpy()
    final_left = extra['left_arm_tcp'][0, :3].cpu().numpy()

    return {
        'success': success_step >= 0,
        'time_to_success': success_step,
        'final_cube_z': final_cube[2],
        'final_cube_y': final_cube[1],
        'best_cube_z': best_z,
        'ego_cube_dist': float(np.linalg.norm(final_left - final_cube)),
    }


def run_matrix(n_episodes=10, max_steps=150, seed=0):
    """Run the full 4x4 strategy matrix."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(
        "SharedWorkspacePickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
    )

    ego_names = list(ALL_EGO_STRATEGIES.keys())
    partner_types = list(PartnerType)

    # results[ego_name][partner_id] = list of episode dicts
    results = {e: {p.value: [] for p in partner_types} for e in ego_names}

    total = len(ego_names) * len(partner_types) * n_episodes
    count = 0

    for ego_name in ego_names:
        for ptype in partner_types:
            for ep in range(n_episodes):
                ego = ALL_EGO_STRATEGIES[ego_name]()
                partner = make_strategy(ptype)

                ep_result = run_episode(env, ego, partner, max_steps)
                results[ego_name][ptype.value].append(ep_result)

                count += 1

            # Progress
            successes = [r['success'] for r in results[ego_name][ptype.value]]
            rate = 100.0 * sum(successes) / len(successes)
            is_optimal = COUNTER_STRATEGY[ptype.value] == ego_name
            marker = " <<<" if is_optimal else ""
            print(f"  {ego_name:<10s} vs {PARTNER_NAMES[ptype.value]:<12s}: "
                  f"{rate:5.1f}% success ({sum(successes)}/{len(successes)}){marker}")

    env.close()
    return results


def print_results(results):
    """Print the 4x4 matrix as a formatted table."""
    ego_names = list(ALL_EGO_STRATEGIES.keys())
    partner_ids = sorted(PARTNER_NAMES.keys())

    # ── Success rate matrix ──
    print("\n" + "=" * 70)
    print("SUCCESS RATE MATRIX (%) — rows=ego strategy, cols=partner type")
    print("=" * 70)

    col_label = "Ego \\ Partner"
    header = f"{col_label:<14s}"
    for pid in partner_ids:
        header += f"  {PARTNER_NAMES[pid]:>12s}"
    header += "     Avg"
    print(header)
    print("-" * 70)

    for ego_name in ego_names:
        row = f"{ego_name:<14s}"
        rates = []
        for pid in partner_ids:
            eps = results[ego_name][pid]
            rate = 100.0 * sum(r['success'] for r in eps) / len(eps)
            rates.append(rate)
            is_optimal = COUNTER_STRATEGY[pid] == ego_name
            marker = "*" if is_optimal else " "
            row += f"  {rate:>11.1f}{marker}"
        avg = np.mean(rates)
        row += f"  {avg:>6.1f}"
        print(row)

    print("-" * 70)
    print("  * = optimal counter-strategy for that partner type\n")

    # ── Time to success matrix ──
    print("TIME TO SUCCESS (steps, mean) — -1 if no successes")
    print("=" * 70)

    col_label = "Ego \\ Partner"
    header = f"{col_label:<14s}"
    for pid in partner_ids:
        header += f"  {PARTNER_NAMES[pid]:>12s}"
    print(header)
    print("-" * 70)

    for ego_name in ego_names:
        row = f"{ego_name:<14s}"
        for pid in partner_ids:
            eps = results[ego_name][pid]
            times = [r['time_to_success'] for r in eps if r['time_to_success'] >= 0]
            if times:
                avg_t = np.mean(times)
                row += f"  {avg_t:>12.1f}"
            else:
                row += f"  {'—':>12s}"
        print(row)

    print()

    # ── Best cube Z matrix ──
    print("BEST CUBE Z (mean) — higher = closer to success")
    print("=" * 70)

    col_label = "Ego \\ Partner"
    header = f"{col_label:<14s}"
    for pid in partner_ids:
        header += f"  {PARTNER_NAMES[pid]:>12s}"
    print(header)
    print("-" * 70)

    for ego_name in ego_names:
        row = f"{ego_name:<14s}"
        for pid in partner_ids:
            eps = results[ego_name][pid]
            avg_z = np.mean([r['best_cube_z'] for r in eps])
            row += f"  {avg_z:>12.3f}"
        print(row)

    print()

    # ── Diagonal vs off-diagonal gap ──
    print("=" * 70)
    print("DIAGONAL (correct) vs OFF-DIAGONAL (wrong) SUMMARY")
    print("=" * 70)

    diag_rates = []
    off_diag_rates = []

    for pid in partner_ids:
        optimal_ego = COUNTER_STRATEGY[pid]
        for ego_name in ego_names:
            eps = results[ego_name][pid]
            rate = 100.0 * sum(r['success'] for r in eps) / len(eps)
            if ego_name == optimal_ego:
                diag_rates.append(rate)
            else:
                off_diag_rates.append(rate)

    diag_mean = np.mean(diag_rates)
    offdiag_mean = np.mean(off_diag_rates)
    gap = diag_mean - offdiag_mean

    print(f"  Correct counter-strategy (diagonal):   {diag_mean:.1f}%  (targets: >80%)")
    print(f"  Wrong strategy (off-diagonal):         {offdiag_mean:.1f}%  (targets: <50%)")
    print(f"  Performance gap:                       {gap:.1f} pp  (targets: >20 pp)")
    print()

    if diag_mean >= 80 and offdiag_mean <= 50 and gap >= 20:
        print("  >>> VALIDATION PASSED — intent detection matters!")
    elif gap >= 20:
        print("  >>> GAP OK but absolute rates need tuning")
    else:
        print("  >>> VALIDATION FAILED — strategies need to be more differentiated")
        print("      Consider: more adversarial partner, time penalties, harder task")


def main():
    parser = argparse.ArgumentParser(description='Phase A: 4x4 strategy validation')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Episodes per ego-partner combination')
    parser.add_argument('--max_steps', type=int, default=150,
                        help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase A: Ego Counter-Strategy Validation (4x4 Matrix)")
    print(f"  Episodes per combination: {args.episodes}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Success threshold: cube z > {SUCCESS_Z}")
    print("=" * 70)

    results = run_matrix(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    print_results(results)


if __name__ == '__main__':
    main()
