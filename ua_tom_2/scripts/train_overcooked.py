#!/usr/bin/env python3
"""
Train UA-ToM and baselines on Overcooked data.

Thin wrapper around existing training infrastructure with Overcooked defaults.
Saves checkpoints for closed-loop evaluation.

Usage:
    python scripts/train_overcooked.py --data_path data/overcooked/cramped_room.npz
    python scripts/train_overcooked.py --data_path data/overcooked/cramped_room.npz \
        --models ua_tom gru transformer --epochs 30
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

from ua_tom.models import get_model, ModelConfig, MODELS
from ua_tom.data.dataset import get_dataloaders
from ua_tom.training.trainer import TrainingConfig, Trainer
from ua_tom.evaluation.metrics import evaluate_model


def train_and_save(model_name, data_path, output_dir, epochs=30, seed=0, device=None):
    """Train a single model on Overcooked data and save checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    train_loader, val_loader, dataset = get_dataloaders(
        data_path, batch_size=16, seed=seed,
    )

    # Overcooked-specific config
    config = ModelConfig(
        obs_dim=dataset.obs_dim,     # 192 for Overcooked
        action_dim=dataset.action_dim,  # 6
        num_types=dataset.num_types,    # 4
        hidden_dim=128,
    )

    training_config = TrainingConfig(
        epochs=epochs,
        lr=1e-3,
        use_amp=True,
    )

    # Create model
    model = get_model(model_name, config)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"Training: {model_name} (Overcooked)")
    print(f"  Params: {params:,} (trainable: {trainable:,})")
    print(f"  Obs dim: {config.obs_dim}, Action dim: {config.action_dim}, "
          f"Types: {config.num_types}")
    print(f"  Data: {len(dataset)} episodes, seq_len={dataset.seq_len}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Train with best checkpoint tracking
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

    ckpt_path = output_dir / f'{model_name}_overcooked_seed{seed}.pt'
    torch.save({
        'model_state_dict': best_state_dict or model.state_dict(),
        'config': config,
        'metrics': best_metrics,
        'model_name': model_name,
        'seed': seed,
        'epochs': epochs,
        'domain': 'overcooked',
    }, ckpt_path)

    print(f"\n  Saved: {ckpt_path}")
    print(f"  Best action acc: {best_metrics.get('action_acc', 0):.1f}%")
    print(f"  Best type acc:   {best_metrics.get('type_acc', 0):.1f}%")
    print(f"  Best switch F1:  {best_metrics.get('switch_f1', 0):.1f}%")

    return best_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train models on Overcooked data')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to .npz data file')
    parser.add_argument('--output_dir', type=str,
                        default='results/overcooked/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--models', nargs='+',
                        default=['ua_tom', 'gru', 'transformer', 'btom',
                                 'mamba', 'bocpd', 'context_cond', 'liam'],
                        help='Models to train')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    all_metrics = {}
    for model_name in args.models:
        if model_name not in MODELS:
            print(f"Warning: {model_name} not in registry, skipping")
            continue
        metrics = train_and_save(
            model_name, args.data_path, args.output_dir,
            epochs=args.epochs, seed=args.seed, device=device,
        )
        all_metrics[model_name] = metrics

    # Summary table
    print(f"\n{'='*70}")
    print("OVERCOOKED TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<18s} {'Action%':>10s} {'Type%':>10s} {'SwitchF1%':>10s} "
          f"{'Recovery':>10s}")
    print("-" * 60)
    for name, m in all_metrics.items():
        print(f"{name:<18s} {m.get('action_acc', 0):>9.1f}% "
              f"{m.get('type_acc', 0):>9.1f}% "
              f"{m.get('switch_f1', 0):>9.1f}% "
              f"{m.get('recovery_time', 0):>9.1f}")


if __name__ == '__main__':
    main()
