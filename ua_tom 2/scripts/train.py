#!/usr/bin/env python3
"""
UA-ToM Training Script
======================

Train and evaluate UA-ToM and baseline models.

Usage:
    # Train UA-ToM
    python train.py --data_path data.npz --model ua_tom --output_dir results
    
    # Train all baselines
    python train.py --data_path data.npz --all_baselines --output_dir results
    
    # Quick test
    python train.py --data_path data.npz --model gru --epochs 5 --seeds 1

Author: For IROS 2026
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model, ModelConfig, MODELS
from data.dataset import get_dataloaders
from training.trainer import TrainingConfig, train_model
from evaluation.metrics import evaluate_model


def run_experiment(
    model_name: str,
    data_path: str,
    config: ModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
    num_seeds: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Run experiment with multiple seeds.
    
    Returns:
        Dictionary with per-seed results and summary statistics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
    
    all_results = []
    
    for seed in range(num_seeds):
        if verbose:
            print(f"\n  Seed {seed + 1}/{num_seeds}")
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create dataloaders
        train_loader, val_loader, dataset = get_dataloaders(
            data_path,
            batch_size=training_config.epochs,  # Use config batch size
            seed=seed,
        )
        
        # Update config with dataset dimensions
        config.obs_dim = dataset.obs_dim
        config.action_dim = dataset.action_dim
        config.num_types = dataset.num_types
        
        # Create model
        model = get_model(model_name, config)
        
        # Train
        metrics = train_model(
            model,
            train_loader,
            val_loader,
            training_config,
            device,
            evaluator=evaluate_model,
            verbose=verbose,
        )
        
        all_results.append(metrics)
        
        if verbose:
            print(f"    → action={metrics.get('action_acc', 0):.1f}%, "
                  f"switch_f1={metrics.get('switch_f1', 0):.1f}%")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Compute summary statistics
    summary = {}
    for key in all_results[0].keys():
        values = [r[key] for r in all_results]
        summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'se': np.std(values) / np.sqrt(len(values)),
        }
    
    if verbose:
        print(f"\n  Summary ({num_seeds} seeds):")
        print(f"    Action Acc: {summary['action_acc']['mean']:.1f} ± {summary['action_acc']['se']:.1f}%")
        print(f"    Switch F1:  {summary['switch_f1']['mean']:.1f} ± {summary['switch_f1']['se']:.1f}%")
        if 'degradation' in summary:
            print(f"    Degradation: {summary['degradation']['mean']:.1f} ± {summary['degradation']['se']:.1f}%")
    
    return {
        'per_seed': all_results,
        'summary': summary,
    }


def main():
    parser = argparse.ArgumentParser(description='Train UA-ToM models')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to .npz data file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    
    # Model selection
    parser.add_argument('--model', type=str, default='ua_tom',
                        choices=list(MODELS.keys()),
                        help='Model to train')
    parser.add_argument('--all_baselines', action='store_true',
                        help='Train all baseline models')
    parser.add_argument('--strong_baselines', action='store_true',
                        help='Train only strong baselines (IROS)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seeds', type=int, default=5)
    
    # Model config
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--training_mode', type=str, default='frozen',
                        choices=['frozen', 'lora', 'full'])
    
    # Hardware
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--no_amp', action='store_true')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("UA-ToM Training")
    print("=" * 60)
    print(f"Data: {args.data_path}")
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    
    # Configs
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        training_mode=args.training_mode,
    )
    
    training_config = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        use_amp=not args.no_amp,
    )
    # Fix: batch_size should be in training config
    training_config.batch_size = args.batch_size
    
    # Determine which models to train
    if args.all_baselines:
        models = list(MODELS.keys())
    elif args.strong_baselines:
        models = ['ua_tom', 'mamba', 'bocpd', 'context_cond', 'liam']
    else:
        models = [args.model]
    
    # Run experiments
    all_results = {}
    
    for model_name in models:
        results = run_experiment(
            model_name,
            args.data_path,
            model_config,
            training_config,
            device,
            num_seeds=args.seeds,
            verbose=True,
        )
        all_results[model_name] = results
    
    # Save results
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Action':>10} {'Switch F1':>12} {'FP Rate':>10}")
    print("-" * 60)
    
    for model_name, results in all_results.items():
        s = results['summary']
        print(f"{model_name:<20} "
              f"{s['action_acc']['mean']:>9.1f}% "
              f"{s['switch_f1']['mean']:>11.1f}% "
              f"{s.get('fp_rate', {}).get('mean', 0):>9.1f}%")
    
    # Generate LaTeX table
    latex = generate_latex_table(all_results)
    latex_path = output_dir / 'results_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {latex_path}")


def generate_latex_table(results: Dict) -> str:
    """Generate LaTeX table from results."""
    
    latex = r"""\begin{table}[t]
\caption{Comparison of partner modeling approaches. Mean $\pm$ SE over 5 seeds.}
\label{tab:results}
\centering
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Action} & \textbf{Type} & \textbf{Switch F1} & \textbf{Precision} & \textbf{FP Rate} & \textbf{Degrad.} \\
\midrule
"""
    
    for model_name, data in results.items():
        s = data['summary']
        
        def fmt(key):
            if key in s:
                return f"{s[key]['mean']:.1f}$\\pm${s[key]['se']:.1f}"
            return "-"
        
        latex += f"{model_name} & {fmt('action_acc')} & {fmt('type_acc')} & "
        latex += f"{fmt('switch_f1')} & {fmt('switch_precision')} & "
        latex += f"{fmt('fp_rate')} & {fmt('degradation')} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


if __name__ == '__main__':
    main()
