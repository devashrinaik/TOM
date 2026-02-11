"""
Training utilities for UA-ToM experiments.

Provides:
- Loss functions
- Trainer class with logging
- Training utilities
"""

from typing import Dict, Optional, Callable
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from ..models.base import BaseModel, ModelOutput
from ..data.dataset import create_switch_labels


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_amp: bool = True
    
    # Loss weights
    action_weight: float = 1.0
    type_weight: float = 1.0
    switch_weight: float = 2.0
    contrastive_weight: float = 0.5
    consistency_weight: float = 0.1
    stability_weight: float = 0.3

    # Switch loss settings
    switch_pos_weight_max: float = 50.0
    
    # Logging
    log_interval: int = 5  # epochs


def compute_loss(
    outputs: ModelOutput,
    actions: torch.Tensor,
    types: torch.Tensor,
    action_dim: int,
    num_types: int,
    config: TrainingConfig,
) -> Dict[str, torch.Tensor]:
    """
    Compute training losses.
    
    Args:
        outputs: Model outputs
        actions: [B, T] partner actions
        types: [B, T] partner types
        action_dim: Number of action classes
        num_types: Number of partner types
        config: Training configuration
    
    Returns:
        Dictionary with 'total' and individual losses
    """
    B, T = actions.shape
    device = actions.device
    losses = {}
    
    # Action prediction loss (predict next action)
    action_logits = outputs.action_logits[:, :-1]  # [B, T-1, A]
    action_targets = actions[:, 1:]  # [B, T-1]
    
    losses['action'] = F.cross_entropy(
        action_logits.reshape(-1, action_dim),
        action_targets.reshape(-1),
    )
    
    # Type classification loss
    type_logits = outputs.type_logits  # [B, T, K]
    
    losses['type'] = F.cross_entropy(
        type_logits.reshape(-1, num_types),
        types.reshape(-1),
    )
    
    # Switch detection loss (class-balanced BCE)
    switch_probs = outputs.switch_probs  # [B, T]
    switch_labels = create_switch_labels(types).float()  # [B, T]
    
    # Compute class weights for imbalanced data
    num_pos = switch_labels.sum() + 1e-6
    num_neg = (1 - switch_labels).sum() + 1e-6
    pos_weight = (num_neg / num_pos).clamp(max=config.switch_pos_weight_max)
    
    # Compute BCE with proper numerical stability
    with autocast(enabled=False):
        switch_probs_f = switch_probs.float().clamp(1e-7, 1 - 1e-7)
        switch_labels_f = switch_labels.float()
        
        bce = F.binary_cross_entropy(
            switch_probs_f,
            switch_labels_f,
            reduction='none',
        )
        
        # Apply class weights
        weights = torch.where(
            switch_labels_f == 1,
            pos_weight,
            torch.ones_like(switch_labels_f),
        )
        losses['switch'] = (weights * bce).mean()

    # Contrastive loss (from model output, if available)
    contrastive_loss = getattr(outputs, 'contrastive_loss', None)
    if contrastive_loss is not None and torch.is_tensor(contrastive_loss):
        losses['contrastive'] = contrastive_loss

    # Stability loss (from hierarchical prediction error, if available)
    stability_logits = getattr(outputs, 'pred_stability_logits', None)
    if stability_logits is not None:
        type_changed = torch.zeros(B, T, dtype=torch.long, device=device)
        type_changed[:, 1:] = (types[:, 1:] != types[:, :-1]).long()
        losses['stability'] = F.cross_entropy(
            stability_logits.reshape(-1, 2),
            type_changed.reshape(-1),
        )

    # Consistency loss (partner embeddings should be stable for same type)
    partner_embeds = getattr(outputs, 'partner_embeds', None)
    if partner_embeds is not None:
        same_type = (types[:, 1:] == types[:, :-1]).float()
        embed_diff = (partner_embeds[:, 1:] - partner_embeds[:, :-1]).pow(2).sum(dim=-1)
        losses['consistency'] = (embed_diff * same_type).sum() / (same_type.sum() + 1e-6)

    # Total loss
    losses['total'] = (
        config.action_weight * losses['action'] +
        config.type_weight * losses['type'] +
        config.switch_weight * losses['switch'] +
        config.contrastive_weight * losses.get('contrastive', 0) +
        config.stability_weight * losses.get('stability', 0) +
        config.consistency_weight * losses.get('consistency', 0)
    )

    return losses


class Trainer:
    """
    Trainer for UA-ToM and baseline models.
    
    Handles:
    - Training loop with AMP
    - Gradient clipping
    - Learning rate scheduling
    - Logging
    """
    
    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
        evaluator: Optional[Callable] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.evaluator = evaluator
        
        # Optimizer (only trainable parameters)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
        )
        
        # AMP scaler
        self.scaler = GradScaler() if config.use_amp and device.type == 'cuda' else None
        
        # Best metrics tracking
        self.best_metrics = None
        self.best_score = 0
    
    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            # Move to device
            observations = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            types = batch['types'].to(self.device)
            images = batch.get('images')
            if images is not None:
                images = images.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward with AMP
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(observations, actions, types, images)
                    losses = compute_loss(
                        outputs, actions, types,
                        self.model.action_dim,
                        self.model.num_types,
                        self.config,
                    )
                
                self.scaler.scale(losses['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(observations, actions, types, images)
                losses = compute_loss(
                    outputs, actions, types,
                    self.model.action_dim,
                    self.model.num_types,
                    self.config,
                )
                
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
                self.optimizer.step()
            
            total_loss += losses['total'].item()
        
        return total_loss / len(self.train_loader)
    
    def train(self, verbose: bool = True) -> Dict:
        """
        Full training loop.
        
        Returns:
            Best validation metrics
        """
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.scheduler.step()
            
            # Evaluate
            if (epoch + 1) % self.config.log_interval == 0 or epoch == self.config.epochs - 1:
                if self.evaluator is not None:
                    metrics = self.evaluator(self.model, self.val_loader, self.device)
                    
                    # Track best
                    score = metrics.get('action_acc', 0) + metrics.get('switch_f1', 0)
                    if score > self.best_score:
                        self.best_score = score
                        self.best_metrics = metrics.copy()
                    
                    if verbose:
                        elapsed = time.time() - start_time
                        print(
                            f"  Epoch {epoch+1:3d}: "
                            f"loss={train_loss:.4f}, "
                            f"action={metrics.get('action_acc', 0):.1f}%, "
                            f"switch_f1={metrics.get('switch_f1', 0):.1f}%, "
                            f"({elapsed:.1f}s)"
                        )
        
        return self.best_metrics or {}


def train_model(
    model: BaseModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    evaluator: Optional[Callable] = None,
    verbose: bool = True,
) -> Dict:
    """
    Convenience function to train a model.
    
    Args:
        model: Model to train
        train_loader: Training data
        val_loader: Validation data
        config: Training configuration
        device: Device to train on
        evaluator: Optional evaluation function
        verbose: Print progress
    
    Returns:
        Best validation metrics
    """
    trainer = Trainer(
        model, train_loader, val_loader,
        config, device, evaluator,
    )
    return trainer.train(verbose=verbose)
