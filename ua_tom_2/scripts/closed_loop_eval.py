#!/usr/bin/env python3
"""
Closed-Loop Evaluation: 2-Type (HELPER + PASSIVE) Experiment.

Runs each experimental condition and measures task success when the ego
must adapt to mid-episode partner switches.

Conditions:
    oracle:      Ground truth type + switch detection
    ua_tom:      UA-ToM model predictions
    gru:         GRU baseline predictions
    no_detect:   Fixed initial strategy, never adapt
    random:      Random strategy switch (1 per episode, random timing + type)
    always_std:  Fixed standard strategy

Usage:
    # Oracle-only (no model needed)
    python scripts/closed_loop_eval.py --episodes 50

    # Full evaluation with models
    python scripts/closed_loop_eval.py \
        --ua_tom_ckpt results/checkpoints/ua_tom_seed0.pt \
        --gru_ckpt results/checkpoints/gru_seed0.pt \
        --episodes 50 --max_steps 150
"""

import os
os.environ['MANI_SKILL_NO_DISPLAY'] = '1'

import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import gymnasium as gym

# ── Package setup ───────────────────────────────────────────────────
_script_dir = Path(__file__).resolve().parent
_pkg_dir = _script_dir.parent  # ua_tom_2/
sys.path.insert(0, str(_pkg_dir.parent))

import importlib, importlib.util
_spec = importlib.util.spec_from_file_location(
    'ua_tom', _pkg_dir / '__init__.py',
    submodule_search_locations=[str(_pkg_dir)],
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules['ua_tom'] = _mod
_spec.loader.exec_module(_mod)

from ua_tom.models import get_model, ModelConfig

# Register SharedWorkspacePickCube-v1
sys.path.insert(0, str(_script_dir))
import test_shared_workspace  # noqa

# Closed-loop components
sys.path.insert(0, str(_pkg_dir))
from closed_loop.partner_strategies import (
    PartnerType, make_strategy, PARTNER_NAMES,
)
from closed_loop.ego_strategies import ALL_EGO_STRATEGIES, COUNTER_STRATEGY
from closed_loop.ego_policy import AdaptiveEgoPolicy

# Observation + action utilities
from collect_shared_workspace import extract_observation, discretize_action


# ── Constants ───────────────────────────────────────────────────────
SUCCESS_Z = 0.15
PARTNER_TYPES_2 = [PartnerType.HELPER, PartnerType.PASSIVE]


# ── Model loading ──────────────────────────────────────────────────
def load_model(ckpt_path, device):
    """Load a pretrained model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model_name = ckpt['model_name']

    model = get_model(model_name, config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    m = ckpt.get('metrics', {})
    print(f"  Loaded {model_name} from {ckpt_path}")
    print(f"    action_acc={m.get('action_acc', 0):.1f}%, "
          f"type_acc={m.get('type_acc', 0):.1f}%, "
          f"switch_f1={m.get('switch_f1', 0):.1f}%")
    return model


# ── Episode runner ──────────────────────────────────────────────────
def run_episode(env, ego_policy, initial_type, switch_time, switch_to_type,
                max_steps=150):
    """
    Run one closed-loop episode.

    Timeline per step:
        1. Extract state (obs_vec, positions) from current env obs
        2. Check if partner should switch at this step
        3. Ego computes action from current strategy (based on prior belief)
        4. Partner computes action
        5. Update ego policy with (obs_vec, partner_disc, gt_type, switch_flag)
           → belief updated for NEXT step
        6. Step environment with both actions
        7. Check success metrics
    """
    obs, env_info = env.reset()

    # Create partner
    partner_type = initial_type
    partner = make_strategy(partner_type)
    partner.reset()

    # Reset ego policy
    ego_policy.reset(initial_type=initial_type.value)

    # Tracking
    success_step = -1
    best_z = 0.0
    strategy_correct_steps = 0
    ego_switch_step = -1
    prev_strategy = ego_policy.current_strategy_name

    for step in range(max_steps):
        # 1. Extract state
        extra = obs['extra']
        left_tcp = extra['left_arm_tcp'][0, :3].cpu().numpy()
        right_tcp = extra['right_arm_tcp'][0, :3].cpu().numpy()
        cube_pos = extra['cube_pose'][0, :3].cpu().numpy()
        obs_vec = extract_observation(obs)

        # 2. Partner switch?
        switch_happened = False
        if step == switch_time and switch_to_type is not None:
            partner_type = switch_to_type
            partner = make_strategy(partner_type)
            partner.reset()
            switch_happened = True

        # 3. Ego acts (current belief)
        ego_action = ego_policy.get_action(left_tcp, cube_pos, right_tcp)

        # 4. Partner acts
        partner_action = partner.get_action(right_tcp, cube_pos, left_tcp)
        partner_disc = discretize_action(partner_action)

        # 5. Update ego belief for next step
        ego_policy.update(
            obs_vec=obs_vec,
            partner_action_discrete=partner_disc,
            ground_truth_type=partner_type.value,
            ground_truth_switch=switch_happened,
        )

        # Track strategy correctness
        optimal = COUNTER_STRATEGY.get(partner_type.value, 'standard')
        if ego_policy.current_strategy_name == optimal:
            strategy_correct_steps += 1

        # Track first ego switch
        if (ego_policy.current_strategy_name != prev_strategy
                and ego_switch_step < 0):
            ego_switch_step = step
        prev_strategy = ego_policy.current_strategy_name

        # 6. Step environment
        action_dict = {
            'panda_wristcam-0': torch.tensor(
                np.array([ego_action]), dtype=torch.float32),
            'panda_wristcam-1': torch.tensor(
                np.array([partner_action]), dtype=torch.float32),
        }
        obs, reward, terminated, truncated, info = env.step(action_dict)

        # 7. Check success — ego must be holding the cube (not helper)
        new_extra = obs['extra']
        cube_z = new_extra['cube_pose'][0, 2].item()
        cube_y = new_extra['cube_pose'][0, 1].item()
        new_left = new_extra['left_arm_tcp'][0, :3].cpu().numpy()
        new_cube = new_extra['cube_pose'][0, :3].cpu().numpy()
        ego_dist = float(np.linalg.norm(new_left - new_cube))

        if cube_z > best_z:
            best_z = cube_z

        if success_step < 0:
            # Success = cube lifted AND ego is holding it (within 10cm)
            ego_holding = ego_dist < 0.10
            if cube_z > SUCCESS_Z and ego_holding:
                success_step = step

        if terminated or truncated:
            break

    # Reaction time: steps between partner switch and ego strategy switch
    reaction_time = -1
    if switch_time >= 0 and ego_switch_step >= 0:
        reaction_time = ego_switch_step - switch_time

    return {
        'success': success_step >= 0,
        'time_to_success': success_step,
        'best_cube_z': best_z,
        'strategy_accuracy': strategy_correct_steps / max(step + 1, 1) * 100,
        'reaction_time': reaction_time,
        'ego_switched': ego_switch_step >= 0,
        'initial_type': initial_type.value,
        'switch_to': switch_to_type.value if switch_to_type else -1,
        'switch_time': switch_time,
    }


# ── Condition runner ────────────────────────────────────────────────
def run_condition(condition_name, model, device, n_episodes, max_steps,
                  switch_range, seed):
    """Run all episodes for one experimental condition."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(
        "SharedWorkspacePickCube-v1",
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
    )

    # Map condition name to ego_policy condition
    if condition_name in ('ua_tom', 'gru'):
        policy_condition = 'model'
    else:
        policy_condition = condition_name

    ego_policy = AdaptiveEgoPolicy(
        condition=policy_condition,
        model=model,
        device=device,
        switch_threshold=0.3,
        cooldown=5,
        max_steps=max_steps,
    )

    results = []

    for ep in range(n_episodes):
        # Alternate initial types (HELPER, PASSIVE, HELPER, ...)
        initial_type = PARTNER_TYPES_2[ep % 2]

        # Plan switch
        switch_time = np.random.randint(*switch_range)
        others = [t for t in PARTNER_TYPES_2 if t != initial_type]
        switch_to = others[0]

        ep_result = run_episode(
            env, ego_policy, initial_type, switch_time, switch_to, max_steps,
        )
        results.append(ep_result)

        if (ep + 1) % 10 == 0:
            sr = 100 * sum(r['success'] for r in results) / len(results)
            print(f"    [{condition_name}] ep {ep+1}/{n_episodes}: "
                  f"{sr:.0f}% success so far")

    env.close()
    return results


# ── Results printing ────────────────────────────────────────────────
def print_results(all_results, condition_order):
    """Print comprehensive results table."""
    print("\n" + "=" * 80)
    print("CLOSED-LOOP RESULTS (2-Type: HELPER + PASSIVE)")
    print("=" * 80)

    # Main table
    print(f"\n{'Condition':<16s} {'Success':>9s} {'Time':>8s} "
          f"{'BestZ':>8s} {'StratAcc':>9s} {'Reaction':>9s}")
    print("-" * 65)

    for cond in condition_order:
        if cond not in all_results:
            continue
        results = all_results[cond]
        n = len(results)

        sr = 100 * sum(r['success'] for r in results) / n
        times = [r['time_to_success'] for r in results
                 if r['time_to_success'] >= 0]
        mean_time = np.mean(times) if times else -1
        mean_z = np.mean([r['best_cube_z'] for r in results])
        strat_acc = np.mean([r['strategy_accuracy'] for r in results])
        reactions = [r['reaction_time'] for r in results
                     if r['reaction_time'] >= 0]
        mean_react = np.mean(reactions) if reactions else -1

        t_str = f"{mean_time:.1f}" if mean_time >= 0 else "  --"
        r_str = f"{mean_react:.1f}" if mean_react >= 0 else "  --"

        print(f"{cond:<16s} {sr:>8.1f}% {t_str:>8s} "
              f"{mean_z:>8.3f} {strat_acc:>8.1f}% {r_str:>9s}")

    # Adaptation gap
    if 'oracle' in all_results and 'no_detect' in all_results:
        oracle_sr = sum(r['success'] for r in all_results['oracle']) / \
            len(all_results['oracle'])
        nodet_sr = sum(r['success'] for r in all_results['no_detect']) / \
            len(all_results['no_detect'])
        gap = oracle_sr - nodet_sr

        print(f"\n{'─'*65}")
        print("ADAPTATION GAP (% of oracle–nodetect gap closed):")
        for cond in condition_order:
            if cond in ('oracle', 'no_detect') or cond not in all_results:
                continue
            csr = sum(r['success'] for r in all_results[cond]) / \
                len(all_results[cond])
            if gap > 0.01:
                pct = (csr - nodet_sr) / gap * 100
            else:
                pct = 0.0
            print(f"  {cond:<16s}: {pct:>6.1f}%")

    # Per-type breakdown
    print(f"\n{'='*80}")
    print("PER INITIAL TYPE BREAKDOWN")
    print(f"{'='*80}")
    print(f"{'Condition':<16s} {'Start HELPER':>14s} {'Start PASSIVE':>14s}")
    print("-" * 50)

    for cond in condition_order:
        if cond not in all_results:
            continue
        results = all_results[cond]

        helper_eps = [r for r in results if r['initial_type'] == 0]
        passive_eps = [r for r in results if r['initial_type'] == 3]

        h_sr = (100 * sum(r['success'] for r in helper_eps) /
                max(len(helper_eps), 1))
        p_sr = (100 * sum(r['success'] for r in passive_eps) /
                max(len(passive_eps), 1))

        print(f"{cond:<16s} {h_sr:>13.1f}% {p_sr:>13.1f}%")

    print()


def save_results(all_results, output_path):
    """Save results to JSON."""

    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    serializable = {}
    for cond, results in all_results.items():
        serializable[cond] = [
            {k: to_serializable(v) for k, v in r.items()} for r in results
        ]

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Closed-Loop 2-Type Evaluation')
    parser.add_argument('--ua_tom_ckpt', type=str, default=None,
                        help='Path to UA-ToM checkpoint')
    parser.add_argument('--gru_ckpt', type=str, default=None,
                        help='Path to GRU baseline checkpoint')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Episodes per condition')
    parser.add_argument('--max_steps', type=int, default=150,
                        help='Max steps per episode')
    parser.add_argument('--switch_min', type=int, default=30,
                        help='Earliest partner switch step')
    parser.add_argument('--switch_max', type=int, default=80,
                        help='Latest partner switch step')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output', type=str,
                        default='results/closed_loop_results.json')
    parser.add_argument('--conditions', nargs='+', default=None,
                        help='Specific conditions to run')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    switch_range = (args.switch_min, args.switch_max)

    print("=" * 80)
    print("Closed-Loop Evaluation: 2-Type (HELPER + PASSIVE)")
    print(f"  Episodes/condition: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Switch range: {switch_range}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {device}")
    print("=" * 80)

    # Load models
    ua_tom_model = None
    gru_model = None

    if args.ua_tom_ckpt:
        print("\nLoading UA-ToM...")
        ua_tom_model = load_model(args.ua_tom_ckpt, device)

    if args.gru_ckpt:
        print("\nLoading GRU baseline...")
        gru_model = load_model(args.gru_ckpt, device)

    # Determine conditions to run
    if args.conditions:
        condition_order = args.conditions
    else:
        condition_order = ['oracle', 'no_detect', 'always_std', 'random']
        if ua_tom_model is not None:
            condition_order.insert(1, 'ua_tom')
        if gru_model is not None:
            idx = condition_order.index('no_detect')
            condition_order.insert(idx, 'gru')

    condition_models = {
        'oracle': None,
        'ua_tom': ua_tom_model,
        'gru': gru_model,
        'no_detect': None,
        'random': None,
        'always_std': None,
    }

    # Run each condition
    all_results = {}

    for cond in condition_order:
        if cond not in condition_models:
            print(f"\n  Skipping unknown condition: {cond}")
            continue

        model = condition_models[cond]
        if cond in ('ua_tom', 'gru') and model is None:
            print(f"\n  Skipping {cond}: no checkpoint provided")
            continue

        print(f"\n  Running: {cond} ({args.episodes} episodes)...")
        results = run_condition(
            cond, model, device,
            args.episodes, args.max_steps, switch_range, args.seed,
        )
        all_results[cond] = results

        sr = 100 * sum(r['success'] for r in results) / len(results)
        print(f"    {cond} → {sr:.1f}% success")

    # Print and save results
    print_results(all_results, condition_order)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(all_results, output_path)


if __name__ == '__main__':
    main()
