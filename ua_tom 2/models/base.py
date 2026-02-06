"""
Base model class for all partner modeling approaches.

All models must inherit from this class and implement the forward method
with a consistent interface for fair comparison.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration for all models."""
    obs_dim: int = 256
    action_dim: int = 8
    num_types: int = 6
    hidden_dim: int = 128
    ssm_dim: int = 64
    ssm_conv_width: int = 4
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # Hierarchical prediction
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 5, 20])

    # Contrastive memory
    memory_dim: int = 128
    memory_size: int = 256
    temperature: float = 0.07

    # Training mode
    training_mode: str = 'frozen'  # 'frozen', 'lora', 'full'
    lora_rank: int = 16


class ModelOutput:
    """Standardized output format for all models."""
    
    def __init__(
        self,
        action_logits: torch.Tensor,
        type_logits: torch.Tensor,
        switch_probs: torch.Tensor,
        belief_changes: Optional[torch.Tensor] = None,
        **kwargs
    ):
        self.action_logits = action_logits      # [B, T, action_dim]
        self.type_logits = type_logits          # [B, T, num_types]
        self.switch_probs = switch_probs        # [B, T]
        self.belief_changes = belief_changes    # [B, T] or None
        
        # Store any additional outputs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getitem__(self, key: str):
        return getattr(self, key)
    
    def get(self, key: str, default=None):
        return getattr(self, key, default)
    
    def __contains__(self, key: str):
        return hasattr(self, key)
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all partner modeling approaches.
    
    All models must:
    1. Accept (observations, actions, types) as input
    2. Return ModelOutput with action_logits, type_logits, switch_probs
    3. Implement get_switch_signal() for fair switch detection comparison
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.num_types = config.num_types
        self.hidden_dim = config.hidden_dim
    
    @abstractmethod
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Forward pass.
        
        Args:
            observations: [B, T, obs_dim] state observations or VLA features
            actions: [B, T] partner actions (discrete)
            types: [B, T] partner types (optional, for training)
            images: [B, T, C, H, W] raw images (optional)
        
        Returns:
            ModelOutput with predictions
        """
        pass
    
    def get_switch_signal(self, outputs: ModelOutput) -> torch.Tensor:
        """
        Extract switch detection signal from model outputs.
        
        For fair comparison, all models use this method to produce
        their switch detection signal for evaluation.
        
        Default implementation uses switch_probs directly.
        Subclasses can override to combine multiple signals.
        
        Returns:
            [B, T] tensor of switch probabilities in [0, 1]
        """
        signal = outputs.switch_probs
        
        # Optionally combine with belief changes if available
        if outputs.belief_changes is not None:
            bc = outputs.belief_changes
            bc_norm = bc / (bc.max() + 1e-6)
            signal = torch.max(signal, bc_norm)
        
        return signal
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight
