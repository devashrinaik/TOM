#!/usr/bin/env python3
"""
Collect Overcooked partner modeling data.

Generates .npz datasets compatible with PartnerDataset for training UA-ToM.

Usage:
    python scripts/collect_overcooked_data.py --layout cramped_room --output data/overcooked/cramped_room.npz
    python scripts/collect_overcooked_data.py --layout forced_coordination --episodes_per_pair 50
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

from ua_tom.overcooked.data_collector import OvercookedEpisodeCollector


def validate_data(data):
    """Check data integrity."""
    obs = data["observations"]
    actions = data["partner_actions"]
    types = data["partner_types"]
    switch_times = data["switch_times"]

    N, T, D = obs.shape
    assert actions.shape == (N, T), f"Actions shape mismatch: {actions.shape}"
    assert types.shape == (N, T), f"Types shape mismatch: {types.shape}"
    assert switch_times.shape == (N,), f"Switch times shape mismatch: {switch_times.shape}"

    # Check for NaN/Inf
    assert not np.isnan(obs).any(), "NaN in observations"
    assert not np.isinf(obs).any(), "Inf in observations"

    # Check action/type ranges
    assert actions.min() >= 0 and actions.max() <= 5, \
        f"Action range: [{actions.min()}, {actions.max()}]"
    assert types.min() >= 0 and types.max() <= 3, \
        f"Type range: [{types.min()}, {types.max()}]"

    # Check distributions
    print("\nValidation passed!")
    print(f"  Shape: {N} episodes x {T} steps x {D} obs_dim")

    print("\n  Action distribution:")
    action_names = ["UP", "DOWN", "RIGHT", "LEFT", "STAY", "INTERACT"]
    for a in range(6):
        count = (actions == a).sum()
        pct = count / actions.size * 100
        print(f"    {action_names[a]:>8s}: {count:>8d} ({pct:.1f}%)")

    print("\n  Type distribution:")
    type_names = ["COOPERATIVE", "GREEDY", "ADVERSARIAL", "RANDOM"]
    for t in range(4):
        count = (types == t).sum()
        pct = count / types.size * 100
        print(f"    {type_names[t]:>12s}: {count:>8d} ({pct:.1f}%)")

    # Switch statistics
    n_switch = (switch_times >= 0).sum()
    n_no_switch = (switch_times < 0).sum()
    print(f"\n  Episodes with switch: {n_switch}")
    print(f"  Episodes without switch: {n_no_switch}")
    if n_switch > 0:
        valid_times = switch_times[switch_times >= 0]
        print(f"  Switch time range: [{valid_times.min()}, {valid_times.max()}]")
        print(f"  Switch time mean: {valid_times.mean():.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect Overcooked data for UA-ToM training")
    parser.add_argument("--layout", type=str, default="cramped_room",
                        help="Overcooked layout name")
    parser.add_argument("--episodes_per_pair", type=int, default=25,
                        help="Episodes per (initial, switch) type pair")
    parser.add_argument("--episode_length", type=int, default=200,
                        help="Steps per episode")
    parser.add_argument("--switch_min", type=int, default=40,
                        help="Earliest switch time")
    parser.add_argument("--switch_max", type=int, default=120,
                        help="Latest switch time")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz path (default: data/overcooked/<layout>.npz)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_switch", action="store_true",
                        help="Also include no-switch episodes (default: True)")
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        output_dir = _pkg_dir / "data" / "overcooked"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"{args.layout}.npz")
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Overcooked Data Collection")
    print(f"  Layout: {args.layout}")
    print(f"  Episodes per pair: {args.episodes_per_pair}")
    print(f"  Episode length: {args.episode_length}")
    print(f"  Switch range: [{args.switch_min}, {args.switch_max}]")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    print()

    # Collect
    collector = OvercookedEpisodeCollector(
        layout=args.layout,
        horizon=args.episode_length,
    )

    data = collector.collect_dataset(
        episodes_per_pair=args.episodes_per_pair,
        episode_length=args.episode_length,
        switch_range=(args.switch_min, args.switch_max),
        include_no_switch=True,
        seed=args.seed,
    )

    # Validate
    validate_data(data)

    # Save
    np.savez_compressed(args.output, **data)
    print(f"\nSaved to {args.output}")
    print(f"  File size: {Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
