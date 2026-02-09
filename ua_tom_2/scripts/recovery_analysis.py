#!/usr/bin/env python3
"""
Recovery Curve Analysis for UA-ToM
===================================

Computes fine-grained post-switch recovery metrics:
1. Per-timestep recovery curves (action acc at each offset after switch)
2. Type re-identification speed (timesteps to correct type prediction)
3. Detected vs missed switch comparison (causal value of detection)
4. Generates publication-quality plots

Usage:
    python recovery_analysis.py --data_path data.npz --output_dir results/recovery
    python recovery_analysis.py --data_path data.npz --models ua_tom mamba gru --output_dir results/recovery
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Package setup (same as train.py)
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

from ua_tom.models import get_model, ModelConfig, MODELS
from ua_tom.data.dataset import PartnerDataset, _split_dataset, create_switch_labels
from ua_tom.training.trainer import TrainingConfig, train_model
from ua_tom.evaluation.metrics import evaluate_model


def compute_recovery_curves(
    model,
    dataloader,
    device,
    max_offset: int = 20,
    switch_threshold: float = 0.3,
):
    """
    Compute per-timestep action accuracy relative to switch events.

    Returns accuracy at each offset t-K..t+K around every ground truth switch,
    plus separate curves for detected vs missed switches.

    Args:
        model: Trained model
        dataloader: Validation data
        device: Device
        max_offset: Max timesteps before/after switch to track
        switch_threshold: Threshold for declaring a detected switch

    Returns:
        Dictionary with:
            - action_curve: [2*max_offset+1] accuracy at each offset
            - type_curve: [2*max_offset+1] type accuracy at each offset
            - detected_curve: [2*max_offset+1] accuracy for detected switches
            - missed_curve: [2*max_offset+1] accuracy for missed switches
            - type_reid_time: mean timesteps to correct type after switch
            - n_detected: number of detected switches
            - n_missed: number of missed switches
    """
    model.eval()

    # Accumulators: offset → list of (correct, total) per timestep
    action_by_offset = defaultdict(list)  # offset → [0/1, ...]
    type_by_offset = defaultdict(list)
    detected_action_by_offset = defaultdict(list)
    missed_action_by_offset = defaultdict(list)
    detected_type_by_offset = defaultdict(list)
    missed_type_by_offset = defaultdict(list)

    type_reid_times = []
    n_detected = 0
    n_missed = 0

    with torch.no_grad():
        for batch in dataloader:
            observations = batch['observations'].to(device)
            actions = batch['actions'].to(device)
            types = batch['types'].to(device)
            images = batch.get('images')
            if images is not None:
                images = images.to(device)

            outputs = model(observations, actions, types, images)
            switch_signal = model.get_switch_signal(outputs)

            # Action predictions (shifted by 1: predict t+1 from t)
            action_preds = outputs.action_logits[:, :-1].argmax(dim=-1)  # [B, T-1]
            action_targets = actions[:, 1:]  # [B, T-1]
            action_correct = (action_preds == action_targets)  # [B, T-1]

            # Type predictions
            type_preds = outputs.type_logits.argmax(dim=-1)  # [B, T]
            type_correct = (type_preds == types)  # [B, T]

            # Switch detection predictions
            switch_preds = (switch_signal > switch_threshold).long()
            switch_labels = create_switch_labels(types)

            B, T = types.shape

            for b in range(B):
                # Find ground truth switch points
                gt_switches = (switch_labels[b] == 1).nonzero(as_tuple=True)[0].tolist()

                for st in gt_switches:
                    # Skip if too close to boundaries
                    if st < max_offset or st + max_offset >= T:
                        continue

                    # Check if this switch was detected (within ±1 tolerance)
                    w_start = max(0, st - 1)
                    w_end = min(T, st + 2)
                    detected = (switch_preds[b, w_start:w_end] == 1).any().item()

                    if detected:
                        n_detected += 1
                    else:
                        n_missed += 1

                    # Collect per-offset metrics
                    for offset in range(-max_offset, max_offset + 1):
                        t_abs = st + offset

                        # Type accuracy at this offset
                        if 0 <= t_abs < T:
                            tc = type_correct[b, t_abs].item()
                            type_by_offset[offset].append(tc)
                            if detected:
                                detected_type_by_offset[offset].append(tc)
                            else:
                                missed_type_by_offset[offset].append(tc)

                        # Action accuracy (indices shifted by 1)
                        t_act = t_abs - 1  # action_correct is [B, T-1], index 0 = prediction for t=1
                        if 0 <= t_act < T - 1:
                            ac = action_correct[b, t_act].item()
                            action_by_offset[offset].append(ac)
                            if detected:
                                detected_action_by_offset[offset].append(ac)
                            else:
                                missed_action_by_offset[offset].append(ac)

                    # Type re-identification time
                    reid_time = max_offset  # default: never recovered
                    for dt in range(0, min(max_offset, T - st)):
                        if type_correct[b, st + dt].item():
                            reid_time = dt
                            break
                    type_reid_times.append(reid_time)

    # Aggregate curves
    offsets = list(range(-max_offset, max_offset + 1))

    def aggregate(d):
        return [np.mean(d[o]) * 100 if d[o] else 0.0 for o in offsets]

    def aggregate_se(d):
        results = []
        for o in offsets:
            if d[o] and len(d[o]) > 1:
                arr = np.array(d[o])
                results.append(arr.std() / np.sqrt(len(arr)) * 100)
            else:
                results.append(0.0)
        return results

    return {
        'offsets': offsets,
        'action_curve': aggregate(action_by_offset),
        'action_curve_se': aggregate_se(action_by_offset),
        'type_curve': aggregate(type_by_offset),
        'type_curve_se': aggregate_se(type_by_offset),
        'detected_action_curve': aggregate(detected_action_by_offset),
        'missed_action_curve': aggregate(missed_action_by_offset),
        'detected_type_curve': aggregate(detected_type_by_offset),
        'missed_type_curve': aggregate(missed_type_by_offset),
        'type_reid_time': float(np.mean(type_reid_times)) if type_reid_times else 0.0,
        'type_reid_time_se': float(np.std(type_reid_times) / np.sqrt(len(type_reid_times))) if len(type_reid_times) > 1 else 0.0,
        'n_detected': n_detected,
        'n_missed': n_missed,
        'n_switches': n_detected + n_missed,
    }


def plot_recovery_curves(all_results, output_dir, task_name):
    """Generate publication-quality recovery curve plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir = Path(output_dir)

    # Color scheme
    colors = {
        'ua_tom': '#1f77b4',
        'ua_tom_frozen': '#1f77b4',
        'ua_tom_lora': '#aec7e8',
        'ua_tom_full': '#6baed6',
        'mamba': '#ff7f0e',
        'transformer': '#2ca02c',
        'gru': '#d62728',
        'tomnet': '#9467bd',
        'bocpd': '#8c564b',
        'btom': '#e377c2',
        'context_cond': '#7f7f7f',
        'liam': '#bcbd22',
    }

    # ---- Plot 1: Action accuracy recovery curves ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for model_name, res in all_results.items():
        if model_name == 'ua_tom_frozen':
            continue  # identical to ua_tom
        offsets = res['offsets']
        curve = res['action_curve']
        color = colors.get(model_name, '#333333')
        lw = 2.5 if 'ua_tom' in model_name else 1.2
        ls = '-' if 'ua_tom' in model_name else '--'
        ax.plot(offsets, curve, label=model_name, color=color, linewidth=lw, linestyle=ls)

    ax.axvline(x=0, color='red', linestyle=':', alpha=0.7, label='switch point')
    ax.set_xlabel('Timestep offset from switch', fontsize=12)
    ax.set_ylabel('Action Prediction Accuracy (%)', fontsize=12)
    ax.set_title(f'Post-Switch Recovery Curves — {task_name}', fontsize=14)
    ax.legend(fontsize=9, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'recovery_curves_{task_name}.png', dpi=150)
    fig.savefig(output_dir / f'recovery_curves_{task_name}.pdf')
    plt.close(fig)
    print(f"  Saved recovery_curves_{task_name}.png/pdf")

    # ---- Plot 2: Type re-identification curves ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for model_name, res in all_results.items():
        if model_name == 'ua_tom_frozen':
            continue
        offsets = res['offsets']
        curve = res['type_curve']
        color = colors.get(model_name, '#333333')
        lw = 2.5 if 'ua_tom' in model_name else 1.2
        ls = '-' if 'ua_tom' in model_name else '--'
        ax.plot(offsets, curve, label=model_name, color=color, linewidth=lw, linestyle=ls)

    ax.axvline(x=0, color='red', linestyle=':', alpha=0.7, label='switch point')
    ax.set_xlabel('Timestep offset from switch', fontsize=12)
    ax.set_ylabel('Type Classification Accuracy (%)', fontsize=12)
    ax.set_title(f'Type Re-Identification Around Switches — {task_name}', fontsize=14)
    ax.legend(fontsize=9, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'type_reidentification_{task_name}.png', dpi=150)
    fig.savefig(output_dir / f'type_reidentification_{task_name}.pdf')
    plt.close(fig)
    print(f"  Saved type_reidentification_{task_name}.png/pdf")

    # ---- Plot 3: Detected vs missed switches (UA-ToM only) ----
    if 'ua_tom' in all_results:
        res = all_results['ua_tom']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        offsets = res['offsets']
        ax1.plot(offsets, res['detected_action_curve'], label=f"Detected (n={res['n_detected']})",
                 color='#2ca02c', linewidth=2)
        ax1.plot(offsets, res['missed_action_curve'], label=f"Missed (n={res['n_missed']})",
                 color='#d62728', linewidth=2)
        ax1.axvline(x=0, color='red', linestyle=':', alpha=0.7)
        ax1.set_xlabel('Timestep offset from switch', fontsize=12)
        ax1.set_ylabel('Action Accuracy (%)', fontsize=12)
        ax1.set_title('Action Recovery: Detected vs Missed', fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.plot(offsets, res['detected_type_curve'], label=f"Detected (n={res['n_detected']})",
                 color='#2ca02c', linewidth=2)
        ax2.plot(offsets, res['missed_type_curve'], label=f"Missed (n={res['n_missed']})",
                 color='#d62728', linewidth=2)
        ax2.axvline(x=0, color='red', linestyle=':', alpha=0.7)
        ax2.set_xlabel('Timestep offset from switch', fontsize=12)
        ax2.set_ylabel('Type Accuracy (%)', fontsize=12)
        ax2.set_title('Type Re-ID: Detected vs Missed', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'UA-ToM: Causal Value of Switch Detection — {task_name}', fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / f'detected_vs_missed_{task_name}.png', dpi=150)
        fig.savefig(output_dir / f'detected_vs_missed_{task_name}.pdf')
        plt.close(fig)
        print(f"  Saved detected_vs_missed_{task_name}.png/pdf")

    # ---- Plot 4: Type re-identification speed bar chart ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    model_names = [m for m in all_results if m != 'ua_tom_frozen']
    reid_times = [all_results[m]['type_reid_time'] for m in model_names]
    reid_ses = [all_results[m]['type_reid_time_se'] for m in model_names]

    bars_colors = [colors.get(m, '#333333') for m in model_names]
    x = np.arange(len(model_names))
    ax.bar(x, reid_times, yerr=reid_ses, color=bars_colors, alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Timesteps to Correct Type', fontsize=12)
    ax.set_title(f'Type Re-Identification Speed — {task_name}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(output_dir / f'type_reid_speed_{task_name}.png', dpi=150)
    fig.savefig(output_dir / f'type_reid_speed_{task_name}.pdf')
    plt.close(fig)
    print(f"  Saved type_reid_speed_{task_name}.png/pdf")


def run_recovery_analysis(
    data_path: str,
    models_to_run: list,
    output_dir: str,
    device: torch.device,
    seed: int = 0,
    epochs: int = 30,
    batch_size: int = 16,
    max_offset: int = 20,
):
    """Run full recovery analysis for specified models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_name = Path(data_path).stem.replace('_v1_data', '')
    print(f"\n{'='*60}")
    print(f"Recovery Analysis: {task_name}")
    print(f"{'='*60}")

    # Load dataset once
    dataset = PartnerDataset(data_path)

    model_config = ModelConfig(hidden_dim=128, training_mode='frozen')
    model_config.obs_dim = dataset.obs_dim
    model_config.action_dim = dataset.action_dim
    model_config.num_types = dataset.num_types

    training_config = TrainingConfig(epochs=epochs, lr=1e-3)
    training_config.batch_size = batch_size

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, val_loader = _split_dataset(dataset, batch_size=batch_size, seed=seed)

    all_results = {}

    for model_name in models_to_run:
        print(f"\n--- {model_name} ---")

        # Train
        torch.manual_seed(seed)
        model = get_model(model_name, model_config)

        print(f"  Training ({epochs} epochs)...")
        train_model(
            model, train_loader, val_loader,
            training_config, device,
            evaluator=evaluate_model, verbose=True,
        )

        # Recovery analysis
        print(f"  Computing recovery curves...")
        recovery = compute_recovery_curves(
            model, val_loader, device,
            max_offset=max_offset,
        )

        all_results[model_name] = recovery
        print(f"  Type re-ID time: {recovery['type_reid_time']:.2f} ± {recovery['type_reid_time_se']:.2f} steps")
        print(f"  Switches: {recovery['n_detected']} detected, {recovery['n_missed']} missed "
              f"({recovery['n_detected']/(recovery['n_switches']+1e-8)*100:.1f}% detection rate)")

        del model
        torch.cuda.empty_cache()

    # Save results
    results_path = output_dir / f'recovery_{task_name}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_recovery_curves(all_results, output_dir, task_name)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Recovery Summary: {task_name}")
    print(f"{'='*60}")
    print(f"{'Model':<18} {'Type ReID':>10} {'Detected':>10} {'Missed':>8} {'Det Rate':>10}")
    print("-" * 60)
    for model_name, res in all_results.items():
        print(f"{model_name:<18} "
              f"{res['type_reid_time']:>9.2f}s "
              f"{res['n_detected']:>10d} "
              f"{res['n_missed']:>8d} "
              f"{res['n_detected']/(res['n_switches']+1e-8)*100:>9.1f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Recovery curve analysis')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/recovery')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to analyze (default: ua_tom + key baselines)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_offset', type=int, default=20)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = args.models or ['ua_tom', 'mamba', 'transformer', 'gru', 'tomnet', 'bocpd']

    run_recovery_analysis(
        data_path=args.data_path,
        models_to_run=models,
        output_dir=args.output_dir,
        device=device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_offset=args.max_offset,
    )


if __name__ == '__main__':
    main()
