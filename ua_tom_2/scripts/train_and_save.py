#!/usr/bin/env python3
"""
Train UA-ToM and baseline models, save checkpoints for closed-loop eval.

Saves model state_dict + config to results/checkpoints/<model>_seed<N>.pt

Usage:
    python scripts/train_and_save.py \
        --data_path data/shared_workspace/shared_workspace_data.npz

    python scripts/train_and_save.py \
        --data_path data/shared_workspace/shared_workspace_data.npz \
        --models ua_tom gru --epochs 30 --seed 0
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

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
from ua_tom.data.dataset import get_dataloaders
from ua_tom.training.trainer import TrainingConfig, Trainer
from ua_tom.evaluation.metrics import evaluate_model


def train_and_save(model_name, data_path, output_dir, epochs=30, seed=0, device=None):
    """Train a single model and save checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    train_loader, val_loader, dataset = get_dataloaders(
        data_path, batch_size=16, seed=seed,
    )

    # Config
    config = ModelConfig(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        num_types=dataset.num_types,
        hidden_dim=128,
    )

    training_config = TrainingConfig(
        epochs=epochs,
        lr=1e-3,
        use_amp=True,
    )
    training_config.batch_size = 16

    # Create model
    model = get_model(model_name, config)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"  Params: {params:,} (trainable: {trainable:,})")
    print(f"  Obs dim: {config.obs_dim}, Action dim: {config.action_dim}, "
          f"Types: {config.num_types}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Train with checkpoint saving
    trainer = Trainer(
        model, train_loader, val_loader,
        training_config, device, evaluator=evaluate_model,
    )

    best_state_dict = None
    best_score = 0
    best_metrics = {}

    for epoch in range(epochs):
        train_loss = trainer.train_epoch()
        trainer.scheduler.step()

        # Evaluate every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            metrics = evaluate_model(model, val_loader, device)
            score = metrics.get('action_acc', 0) + metrics.get('switch_f1', 0)

            if score > best_score:
                best_score = score
                best_metrics = metrics.copy()
                best_state_dict = {k: v.cpu().clone()
                                   for k, v in model.state_dict().items()}

            print(f"  Epoch {epoch+1:3d}: loss={train_loss:.4f}, "
                  f"action={metrics.get('action_acc', 0):.1f}%, "
                  f"type={metrics.get('type_acc', 0):.1f}%, "
                  f"switch_f1={metrics.get('switch_f1', 0):.1f}%"
                  f"{' *best*' if score >= best_score else ''}")

    # Save checkpoint
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / f'{model_name}_seed{seed}.pt'
    torch.save({
        'model_state_dict': best_state_dict or model.state_dict(),
        'config': config,
        'metrics': best_metrics,
        'model_name': model_name,
        'seed': seed,
        'epochs': epochs,
    }, ckpt_path)

    print(f"\n  Saved: {ckpt_path}")
    print(f"  Best action acc: {best_metrics.get('action_acc', 0):.1f}%")
    print(f"  Best switch F1:  {best_metrics.get('switch_f1', 0):.1f}%")
    print(f"  Best type acc:   {best_metrics.get('type_acc', 0):.1f}%")

    return best_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train models and save checkpoints')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str,
                        default='results/checkpoints')
    parser.add_argument('--models', nargs='+', default=['ua_tom', 'gru'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    all_metrics = {}
    for model_name in args.models:
        metrics = train_and_save(
            model_name, args.data_path, args.output_dir,
            epochs=args.epochs, seed=args.seed, device=device,
        )
        all_metrics[model_name] = metrics

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15s} {'Action%':>10s} {'Type%':>10s} {'SwitchF1%':>10s}")
    print("-" * 50)
    for name, m in all_metrics.items():
        print(f"{name:<15s} {m.get('action_acc', 0):>9.1f}% "
              f"{m.get('type_acc', 0):>9.1f}% "
              f"{m.get('switch_f1', 0):>9.1f}%")


if __name__ == '__main__':
    main()
