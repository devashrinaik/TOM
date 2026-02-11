#!/usr/bin/env python3
"""
Collect Overcooked data with V2 delayed-revelation partners.

Same output format as v1 (.npz), but uses v2 partners that all start
as GHM and diverge after diverge_time ~ U[30,50]. Also saves diverge_times.

Usage:
    python scripts/collect_overcooked_data_v2.py --layout cramped_room
    python scripts/collect_overcooked_data_v2.py --layout asymmetric_advantages --episodes_per_pair 25
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

from ua_tom.overcooked.data_collector_v2 import OvercookedEpisodeCollectorV2
from ua_tom.overcooked.partner_strategies_v2 import PARTNER_NAMES_V2


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

    assert not np.isnan(obs).any(), "NaN in observations"
    assert not np.isinf(obs).any(), "Inf in observations"

    assert actions.min() >= 0 and actions.max() <= 5, \
        f"Action range: [{actions.min()}, {actions.max()}]"
    assert types.min() >= 0 and types.max() <= 3, \
        f"Type range: [{types.min()}, {types.max()}]"

    print("\nValidation passed!")
    print(f"  Shape: {N} episodes x {T} steps x {D} obs_dim")

    print("\n  Action distribution:")
    action_names = ["UP", "DOWN", "RIGHT", "LEFT", "STAY", "INTERACT"]
    for a in range(6):
        count = (actions == a).sum()
        pct = count / actions.size * 100
        print(f"    {action_names[a]:>8s}: {count:>8d} ({pct:.1f}%)")

    print("\n  Type distribution (v2 delayed-revelation):")
    for t in range(4):
        name = PARTNER_NAMES_V2.get(t, f"type_{t}")
        count = (types == t).sum()
        pct = count / types.size * 100
        print(f"    {name:>12s}: {count:>8d} ({pct:.1f}%)")

    n_switch = (switch_times >= 0).sum()
    n_no_switch = (switch_times < 0).sum()
    print(f"\n  Episodes with switch: {n_switch}")
    print(f"  Episodes without switch: {n_no_switch}")
    if n_switch > 0:
        valid_times = switch_times[switch_times >= 0]
        print(f"  Switch time range: [{valid_times.min()}, {valid_times.max()}]")
        print(f"  Switch time mean: {valid_times.mean():.1f}")

    # V2-specific: diverge times
    if "initial_diverge_times" in data:
        dt = data["initial_diverge_times"]
        valid_dt = dt[dt >= 0]
        if len(valid_dt) > 0:
            print(f"\n  Diverge times (initial partner):")
            print(f"    Range: [{valid_dt.min()}, {valid_dt.max()}]")
            print(f"    Mean: {valid_dt.mean():.1f}")
            n_no_diverge = (dt < 0).sum()
            print(f"    No diverge (reliable): {n_no_diverge}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect Overcooked V2 data (delayed-revelation partners)")
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
                        help="Output .npz path (default: data/overcooked/<layout>_v2.npz)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        output_dir = _pkg_dir / "data" / "overcooked"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"{args.layout}_v2.npz")
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Overcooked V2 Data Collection (Delayed-Revelation Partners)")
    print(f"  Layout: {args.layout}")
    print(f"  Episodes per pair: {args.episodes_per_pair}")
    print(f"  Episode length: {args.episode_length}")
    print(f"  Switch range: [{args.switch_min}, {args.switch_max}]")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    print()

    collector = OvercookedEpisodeCollectorV2(
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

    validate_data(data)

    np.savez_compressed(args.output, **data)
    print(f"\nSaved to {args.output}")
    print(f"  File size: {Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
