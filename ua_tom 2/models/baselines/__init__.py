"""
Baselines Package
=================

Proper implementations of baseline models for UA-ToM comparison.

All baselines follow published methods faithfully:

SSM/Architecture Baselines:
    - MambaNoBeliefBaseline: Official Mamba (Gu & Dao 2023) without belief heads
    
Classical Methods:
    - BOCPDBaseline: Bayesian Online Changepoint Detection (Adams & MacKay 2007)
    
Theory of Mind:
    - ToMnetBaseline: Machine Theory of Mind (Rabinowitz et al. 2018)
    - ToMnetCharacterOnly: ToMnet ablation (character network only)
    - ToMnetMentalOnly: ToMnet ablation (mental state only)
    
Opponent Modeling:
    - DRONBaseline: Deep Reinforcement Opponent Network (He et al. 2016)
    - LatentOpponentVAE: VAE-based latent opponent embedding
    - OIAMBaseline: Online Implicit Agent Modeling
    - LIAMBaseline: Local Information Agent Modeling (Papoudakis et al. 2021)
    - SOMBaseline: Self Other-Modeling (Raileanu et al. 2018)
    - MBOMLite: Simplified Model-Based Opponent Modeling (Yu et al. 2022)
    
Context-Conditional:
    - ContextConditionalBaseline: FiLM conditioning (Perez et al. 2018)
    - ContextConditionalGRU: Simplified GRU + FiLM variant
    - ContextConditionalPolicy: VAE-based context inference

Simple/Weak Baselines:
    - GRUBaseline: Standard GRU (weak baseline)
    - TransformerBaseline: Standard Transformer (weak baseline)

Usage:
    from ua_tom.models.baselines import MambaNoBeliefBaseline, BOCPDBaseline
    
    model = MambaNoBeliefBaseline(config)
"""

# Mamba baseline (official package or faithful reimplementation)
from .mamba_baseline import (
    MambaNoBeliefBaseline,
    MambaBlockPure,
    MAMBA_AVAILABLE,
)

# BOCPD baseline (Adams & MacKay 2007)
from .bocpd_baseline import (
    BOCPDBaseline,
    BOCPDDetector,
    BOCPDConfig,
    StudentTSufficientStats,
    BAYESIAN_CPD_AVAILABLE,
)

# ToMnet baseline (Rabinowitz et al. 2018)
from .tomnet_baseline import (
    ToMnetBaseline,
    ToMnetConfig,
    ToMnetCharacterOnly,
    ToMnetMentalOnly,
    CharacterNetwork,
    MentalStateNetwork,
    PredictionNetwork,
)

# Context-Conditional with FiLM (Perez et al. 2018)
from .context_conditional import (
    ContextConditionalBaseline,
    ContextConditionalGRU,
    FiLMLayer,
    CausalWindowAttention,
)

# Opponent modeling baselines
from .opponent_modeling import (
    DRONBaseline,
    LatentOpponentVAE,
    OIAMBaseline,
)

# Extended opponent modeling (LIAM, SOM, Context-Conditional)
from .opponent_modeling_extended import (
    LIAMBaseline,
    LIAMConfig,
    SOMBaseline,
    SOMConfig,
    ContextConditionalPolicy,
    ContextConditionalConfig,
    MBOMLite,
    create_extended_opponent_model,
)

# Simple/weak baselines
from .temporal import (
    GRUBaseline,
    TransformerBaseline,
)

# Legacy imports for backwards compatibility
from .tom import BToM  # Keep old BToM


# =============================================================================
# Model Registry
# =============================================================================

BASELINE_REGISTRY = {
    # Strong baselines (for ICML paper)
    'mamba': MambaNoBeliefBaseline,
    'bocpd': BOCPDBaseline,
    'tomnet': ToMnetBaseline,
    'context_cond': ContextConditionalBaseline,
    'dron': DRONBaseline,
    'vae_opponent': LatentOpponentVAE,
    
    # Extended opponent modeling (new)
    'liam': LIAMBaseline,
    'som': SOMBaseline,
    'context_conditional_policy': ContextConditionalPolicy,
    'mbom_lite': MBOMLite,
    
    # Weak baselines
    'gru': GRUBaseline,
    'transformer': TransformerBaseline,
    
    # Ablations
    'tomnet_char_only': ToMnetCharacterOnly,
    'tomnet_mental_only': ToMnetMentalOnly,
    'context_cond_gru': ContextConditionalGRU,
    'oiam': OIAMBaseline,
    
    # Legacy
    'btom': BToM,
}


def get_baseline(name: str, config):
    """
    Get a baseline model by name.
    
    Args:
        name: Baseline name (see BASELINE_REGISTRY)
        config: ModelConfig instance
    
    Returns:
        Instantiated baseline model
    """
    if name not in BASELINE_REGISTRY:
        available = ', '.join(BASELINE_REGISTRY.keys())
        raise ValueError(f"Unknown baseline: {name}. Available: {available}")
    
    return BASELINE_REGISTRY[name](config)


def list_baselines() -> list:
    """List all available baseline names."""
    return list(BASELINE_REGISTRY.keys())


# =============================================================================
# Package availability info
# =============================================================================

def check_packages():
    """Print availability of optional packages."""
    print("Package availability:")
    print(f"  mamba-ssm: {'✓' if MAMBA_AVAILABLE else '✗ (using pure PyTorch)'}")
    print(f"  bayesian_changepoint_detection: {'✓' if BAYESIAN_CPD_AVAILABLE else '✗ (using custom impl)'}")


__all__ = [
    # Mamba
    'MambaNoBeliefBaseline',
    'MambaBlockPure',
    'MAMBA_AVAILABLE',
    
    # BOCPD
    'BOCPDBaseline',
    'BOCPDDetector',
    'BOCPDConfig',
    'StudentTSufficientStats',
    'BAYESIAN_CPD_AVAILABLE',
    
    # ToMnet
    'ToMnetBaseline',
    'ToMnetConfig',
    'ToMnetCharacterOnly',
    'ToMnetMentalOnly',
    'CharacterNetwork',
    'MentalStateNetwork',
    'PredictionNetwork',
    
    # Context-Conditional (FiLM-based)
    'ContextConditionalBaseline',
    'ContextConditionalGRU',
    'FiLMLayer',
    'CausalWindowAttention',
    
    # Opponent Modeling (original)
    'DRONBaseline',
    'LatentOpponentVAE',
    'OIAMBaseline',
    
    # Extended Opponent Modeling (new)
    'LIAMBaseline',
    'LIAMConfig',
    'SOMBaseline', 
    'SOMConfig',
    'ContextConditionalPolicy',
    'ContextConditionalConfig',
    'MBOMLite',
    'create_extended_opponent_model',
    
    # Simple baselines
    'GRUBaseline',
    'TransformerBaseline',
    
    # Legacy
    'BToM',
    
    # Registry
    'BASELINE_REGISTRY',
    'get_baseline',
    'list_baselines',
    'check_packages',
]
