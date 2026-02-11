#!/usr/bin/env python3
"""
Closed-loop evaluation for Overcooked.

Loads trained model checkpoints and runs all experimental conditions.

Usage:
    python scripts/eval_overcooked_cl.py \
        --checkpoint_dir results/overcooked/checkpoints \
        --layout cramped_room

    python scripts/eval_overcooked_cl.py \
        --checkpoint_dir results/overcooked/checkpoints \
        --models ua_tom gru --n_episodes 10
"""

import argparse
import json
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
from ua_tom.overcooked.closed_loop_eval import run_evaluation, print_results_table


def load_checkpoint(ckpt_path, device):
    """Load a model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    model_name = ckpt['model_name']

    model = get_model(model_name, config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    print(f"  Loaded {model_name} from {ckpt_path}")
    print(f"    Training metrics: action={ckpt['metrics'].get('action_acc', 0):.1f}%, "
          f"type={ckpt['metrics'].get('type_acc', 0):.1f}%, "
          f"switch_f1={ckpt['metrics'].get('switch_f1', 0):.1f}%")

    return model, model_name


def main():
    parser = argparse.ArgumentParser(
        description='Closed-loop evaluation on Overcooked')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='results/overcooked/checkpoints',
                        help='Directory with .pt checkpoints')
    parser.add_argument('--layout', type=str, default='cramped_room')
    parser.add_argument('--models', nargs='+',
                        default=['ua_tom', 'gru', 'transformer', 'btom',
                                 'mamba', 'bocpd', 'context_cond', 'liam'],
                        help='Model names to evaluate (must have matching checkpoints)')
    parser.add_argument('--n_episodes', type=int, default=30,
                        help='Episodes per (init, switch) pair')
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--switch_min', type=int, default=40)
    parser.add_argument('--switch_max', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help='Save results JSON to this path')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"Overcooked Closed-Loop Evaluation")
    print(f"  Layout: {args.layout}")
    print(f"  Episodes per pair: {args.n_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Switch range: [{args.switch_min}, {args.switch_max}]")
    print(f"  Device: {device}")

    # Build conditions dict
    conditions = {
        "oracle": (None, None),
        "no_detect": (None, None),
        "random": (None, None),
        "always_coordinated": (None, None),
    }

    # Load model checkpoints
    ckpt_dir = Path(args.checkpoint_dir)
    for model_name in args.models:
        # Try different naming patterns
        candidates = [
            ckpt_dir / f'{model_name}_overcooked_seed0.pt',
            ckpt_dir / f'{model_name}_seed0.pt',
        ]
        ckpt_path = None
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break

        if ckpt_path is None:
            print(f"  Warning: No checkpoint found for {model_name}, skipping")
            continue

        model, _ = load_checkpoint(ckpt_path, device)
        conditions[model_name] = (model, device)

    # Run evaluation
    results = run_evaluation(
        conditions=conditions,
        layout=args.layout,
        n_episodes=args.n_episodes,
        switch_range=(args.switch_min, args.switch_max),
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # Print results
    print_results_table(results)

    # Save JSON
    if args.output is None:
        output_dir = _pkg_dir / "results" / "overcooked"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"closed_loop_{args.layout}_v5.json")

    # Convert results to JSON-serializable
    json_results = {}
    for cond, m in results.items():
        json_m = {}
        for k, v in m.items():
            if isinstance(v, (np.integer, np.floating)):
                json_m[k] = float(v)
            elif isinstance(v, dict):
                json_m[k] = {
                    kk: {kkk: float(vvv) for kkk, vvv in vv.items()}
                    if isinstance(vv, dict) else float(vv)
                    for kk, vv in v.items()
                }
            else:
                json_m[k] = v
        json_results[cond] = json_m

    with open(args.output, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
