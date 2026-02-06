"""
UA-ToM: Unified Adaptive Theory of Mind

A lightweight module for adaptive belief tracking in VLA models
to enable coordination with non-stationary partners.
"""

__version__ = '0.1.0'
__author__ = 'IROS 2026'

from .models import (
    UAToM,
    ModelConfig,
    get_model,
    MODELS,
)

from .data import (
    PartnerDataset,
    get_dataloaders,
)

from .training import (
    TrainingConfig,
    train_model,
)

from .evaluation import (
    evaluate_model,
)

__all__ = [
    # Models
    'UAToM',
    'ModelConfig',
    'get_model',
    'MODELS',
    
    # Data
    'PartnerDataset',
    'get_dataloaders',
    
    # Training
    'TrainingConfig',
    'train_model',
    
    # Evaluation
    'evaluate_model',
]
