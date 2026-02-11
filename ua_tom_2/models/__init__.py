"""
UA-ToM Models Package.

Provides all models for partner modeling experiments.
"""

from .base import BaseModel, ModelConfig, ModelOutput, RMSNorm
from .ua_tom import UAToM, UAToMFrozen, UAToMLoRA, UAToMFull

from .baselines.temporal import GRUBaseline, TransformerBaseline
from .baselines.tom import BToM, ToMnet
from .baselines.mamba_baseline import MambaNoBeliefBaseline
from .baselines.bocpd_baseline import BOCPDBaseline
from .baselines.context_conditional import ContextConditionalBaseline
from .baselines.opponent_modeling_extended import LIAMBaseline

# Model registry for easy access
MODELS = {
    # Main model
    'ua_tom': UAToM,
    'ua_tom_frozen': UAToMFrozen,
    'ua_tom_lora': UAToMLoRA,
    'ua_tom_full': UAToMFull,
    
    # Weak baselines
    'gru': GRUBaseline,
    'transformer': TransformerBaseline,
    'btom': BToM,
    'tomnet': ToMnet,
    
    # Strong baselines (new for IROS)
    'mamba': MambaNoBeliefBaseline,
    'bocpd': BOCPDBaseline,
    'context_cond': ContextConditionalBaseline,
    'liam': LIAMBaseline,
}


def get_model(name: str, config: ModelConfig, **kwargs) -> BaseModel:
    """
    Factory function to create models.
    
    Args:
        name: Model name (see MODELS dict)
        config: Model configuration
        **kwargs: Additional model-specific arguments
    
    Returns:
        Instantiated model
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    
    return MODELS[name](config, **kwargs)


__all__ = [
    # Base
    'BaseModel',
    'ModelConfig', 
    'ModelOutput',
    'RMSNorm',
    
    # Main model
    'UAToM',
    'UAToMFrozen',
    'UAToMLoRA',
    'UAToMFull',
    
    # Baselines
    'GRUBaseline',
    'TransformerBaseline',
    'BToM',
    'ToMnet',
    'MambaNoBeliefBaseline',
    'BOCPDBaseline',
    'ContextConditionalBaseline',
    'LIAMBaseline',
    
    # Registry
    'MODELS',
    'get_model',
]
