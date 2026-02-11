#!/usr/bin/env python3
"""
Diagnostic script for closed-loop inference issues.

Runs cooperative→adversarial episodes and prints per-step model outputs
to diagnose normalization spikes, detection failures, and strategy mismatches.

Usage:
    python scripts/diagnose_cl_inference.py \
        --checkpoint_dir results/overcooked/checkpoints_v2 \
        --layout cramped_room --n_episodes 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

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

from ua_tom.models import get_model, ModelConfig
from ua_tom.overcooked.partner_strategies import PARTNER_TYPES, PARTNER_NAMES
from ua_tom.overcooked.ego_policy import OvercookedModelWrapper, COUNTER_STRATEGY
from ua_tom.overcooked.data_collector import DEFAULT_MLAM_PARAMS

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.agents.agent import GreedyHumanModel


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    model_name = ckpt['model_name']
    model = get_model(model_name, config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    return model, model_name


def run_diagnostic_episode(
    env, mdp, mlam, wrappers, initial_type, switch_to_type,
    switch_time, max_steps, device,
):
    """Run one episode and collect per-step diagnostics from all model wrappers."""
    partner = PARTNER_TYPES[initial_type]()
    env.reset()
    state = env.state
    partner.reset(state, mdp, mlam)

    # Ego uses GHM (we don't adapt — just observe model outputs)
    ego_ghm = GreedyHumanModel(mlam)
    ego_ghm.set_agent_index(0)

    # Reset wrappers
    for w in wrappers.values():
        w.reset()

    diagnostics = []

    for t in range(max_steps):
        is_switch = (t == switch_time)
        if is_switch:
            partner = PARTNER_TYPES[switch_to_type]()
            partner.reset(state, mdp, mlam)

        current_type = initial_type if t < switch_time else switch_to_type

        # Featurize
        feats = mdp.featurize_state(state, mlam)
        obs_vec = np.concatenate([
            np.array(feats[0], dtype=np.float32),
            np.array(feats[1], dtype=np.float32),
        ])

        # Get actions
        ego_action, ego_idx = ego_ghm.action(state)
        partner_action, partner_idx = partner.get_action(state)

        # Query each model
        step_diag = {
            't': t,
            'gt_type': current_type,
            'gt_type_name': PARTNER_NAMES[current_type],
            'is_switch': is_switch,
            'partner_action': partner_idx,
        }

        for name, wrapper in wrappers.items():
            pred = wrapper.update(obs_vec, partner_idx)
            step_diag[f'{name}_switch_prob'] = pred['switch_prob']
            step_diag[f'{name}_pred_type'] = pred['pred_type']
            step_diag[f'{name}_type_conf'] = pred['type_probs'].max()
            step_diag[f'{name}_type_probs'] = pred['type_probs'].copy()
            counter = COUNTER_STRATEGY.get(pred['pred_type'], 'independent')
            step_diag[f'{name}_would_switch_to'] = counter

        diagnostics.append(step_diag)

        # Step env
        joint_action = (ego_action, partner_action)
        try:
            state, reward, done, info = env.step(joint_action)
        except Exception:
            break

        if done:
            break

    return diagnostics


def print_diagnostics(diagnostics, model_names, switch_time):
    """Print per-step table focused on key steps."""
    # Print header
    cols = ['t', 'GT_type', 'switch']
    for m in model_names:
        cols += [f'{m}_sp', f'{m}_type', f'{m}_conf']
    header = ' | '.join(f'{c:>10s}' for c in cols)
    print(header)
    print('-' * len(header))

    # Print key steps: first 10, around switch, last 5
    key_steps = set(range(min(10, len(diagnostics))))
    key_steps |= set(range(
        max(0, switch_time - 5),
        min(len(diagnostics), switch_time + 20),
    ))
    key_steps |= set(range(max(0, len(diagnostics) - 5), len(diagnostics)))

    prev_was_printed = True
    for i, d in enumerate(diagnostics):
        if i not in key_steps:
            if prev_was_printed:
                print('       ...')
            prev_was_printed = False
            continue
        prev_was_printed = True

        row = [
            f"{d['t']:>10d}",
            f"{d['gt_type_name']:>10s}",
            f"{'***YES***' if d['is_switch'] else '':>10s}",
        ]
        for m in model_names:
            sp = d[f'{m}_switch_prob']
            pt = d[f'{m}_pred_type']
            conf = d[f'{m}_type_conf']
            type_name = PARTNER_NAMES[pt] if pt < len(PARTNER_NAMES) else '?'

            # Flag suspicious values
            sp_str = f"{sp:.3f}"
            if sp > 0.5 and not d['is_switch']:
                sp_str = f"!{sp:.3f}"  # false positive
            if d['is_switch'] and sp < 0.3:
                sp_str = f"?{sp:.3f}"  # missed

            row += [f"{sp_str:>10s}", f"{type_name:>10s}", f"{conf:.3f}".rjust(10)]

        print(' | '.join(row))


def summarize_early_spikes(diagnostics, model_names):
    """Check for normalization spikes in first 20 steps."""
    print("\n=== Early-step spike analysis (t=0..19) ===")
    for m in model_names:
        early_probs = [d[f'{m}_switch_prob'] for d in diagnostics[:20]]
        if early_probs:
            mean_sp = np.mean(early_probs)
            max_sp = np.max(early_probs)
            n_above_03 = sum(1 for p in early_probs if p > 0.3)
            n_above_05 = sum(1 for p in early_probs if p > 0.5)
            print(f"  {m}: mean={mean_sp:.3f}, max={max_sp:.3f}, "
                  f"n>0.3={n_above_03}, n>0.5={n_above_05}")
            if max_sp > 0.5:
                print(f"    !! NORMALIZATION SPIKE DETECTED — max switch_prob={max_sp:.3f} in first 20 steps")


def main():
    parser = argparse.ArgumentParser(description='Diagnose CL inference issues')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='results/overcooked/checkpoints_v2')
    parser.add_argument('--layout', type=str, default='cramped_room')
    parser.add_argument('--n_episodes', type=int, default=5)
    parser.add_argument('--switch_time', type=int, default=80)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--initial_type', type=int, default=0,
                        help='Initial partner type (0=coop)')
    parser.add_argument('--switch_to_type', type=int, default=2,
                        help='Switch-to partner type (2=adversarial)')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Resolve checkpoint dir relative to package
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = _pkg_dir / ckpt_dir

    print(f"Diagnostic: {PARTNER_NAMES[args.initial_type]} → "
          f"{PARTNER_NAMES[args.switch_to_type]} at t={args.switch_time}")
    print(f"Layout: {args.layout}, Episodes: {args.n_episodes}")
    print(f"Checkpoints: {ckpt_dir}")
    print()

    # Create env
    mdp = OvercookedGridworld.from_layout_name(args.layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=args.max_steps)

    params = DEFAULT_MLAM_PARAMS.copy()
    params["counter_goals"] = mdp.get_counter_locations()
    params["counter_drop"] = mdp.get_counter_locations()
    params["counter_pickup"] = mdp.get_counter_locations()
    mlam = MediumLevelActionManager.from_pickle_or_compute(
        mdp, params, force_compute=False,
    )

    # Load models
    wrappers = {}
    model_names = []
    for name in ['ua_tom', 'gru']:
        ckpt_path = ckpt_dir / f'{name}_overcooked_seed0.pt'
        if not ckpt_path.exists():
            print(f"  Skipping {name}: {ckpt_path} not found")
            continue
        model, _ = load_checkpoint(ckpt_path, device)
        wrappers[name] = OvercookedModelWrapper(model, device, args.max_steps)
        model_names.append(name)
        print(f"  Loaded {name}")

    if not wrappers:
        print("No models found!")
        return

    # Run episodes
    all_early_spikes = {m: [] for m in model_names}

    for ep in range(args.n_episodes):
        print(f"\n{'='*80}")
        print(f"Episode {ep+1}/{args.n_episodes}")
        print(f"{'='*80}")

        np.random.seed(42 + ep)
        diagnostics = run_diagnostic_episode(
            env, mdp, mlam, wrappers,
            initial_type=args.initial_type,
            switch_to_type=args.switch_to_type,
            switch_time=args.switch_time,
            max_steps=args.max_steps,
            device=device,
        )

        print_diagnostics(diagnostics, model_names, args.switch_time)
        summarize_early_spikes(diagnostics, model_names)

        # Collect spike stats
        for m in model_names:
            early = [d[f'{m}_switch_prob'] for d in diagnostics[:20]]
            all_early_spikes[m].extend(early)

    # Summary across all episodes
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL EPISODES")
    print(f"{'='*80}")
    for m in model_names:
        spikes = all_early_spikes[m]
        if spikes:
            print(f"\n{m} early-step (t<20) switch_prob stats:")
            print(f"  mean={np.mean(spikes):.4f}, std={np.std(spikes):.4f}")
            print(f"  max={np.max(spikes):.4f}, min={np.min(spikes):.4f}")
            print(f"  >0.3: {sum(1 for s in spikes if s > 0.3)}/{len(spikes)} "
                  f"({sum(1 for s in spikes if s > 0.3)/len(spikes)*100:.1f}%)")
            print(f"  >0.5: {sum(1 for s in spikes if s > 0.5)}/{len(spikes)} "
                  f"({sum(1 for s in spikes if s > 0.5)/len(spikes)*100:.1f}%)")


if __name__ == '__main__':
    main()
