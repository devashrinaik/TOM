# UA-ToM Baselines

Comprehensive collection of baseline models for comparison with UA-ToM. All baselines follow published methods faithfully.

## Quick Start

```python
from ua_tom.models.baselines import get_baseline, list_baselines

# List all 16+ available baselines
print(list_baselines())

# Get any baseline by name
model = get_baseline('tomnet', config)
```

## Installation

```bash
# Core requirements
pip install torch numpy einops

# Optional packages for official implementations
pip install mamba-ssm causal-conv1d>=1.4.0  # Requires CUDA
pip install bayesian-changepoint-detection
```

## Baseline Overview

### By Category

| Category | Baselines | Focus |
|----------|-----------|-------|
| **SSM Architecture** | Mamba | Tests architecture contribution |
| **Classical Bayesian** | BOCPD | Gold-standard changepoint detection |
| **Theory of Mind** | ToMnet, ToMnet-CharOnly, ToMnet-MentalOnly | Mental state modeling |
| **Opponent Modeling** | DRON, DRON-MoE, VAE-Opponent, OIAM | Partner representation learning |
| **Self-Modeling** | SOM, LIAM | Model partners using own policy |
| **Context-Conditional** | FiLM, Context-VAE | Latent context inference |
| **Model-Based** | MBOM-Lite | Dynamics modeling |
| **Simple/Weak** | GRU, Transformer | Lower bounds |

### Summary Table

| Baseline | Paper | Year | Switch Detection Method |
|----------|-------|------|------------------------|
| **Mamba** | Gu & Dao | 2023 | State change magnitude |
| **BOCPD** | Adams & MacKay | 2007 | Run-length posterior |
| **ToMnet** | Rabinowitz et al. | 2018 | Mental state change |
| **DRON** | He et al. | 2016 | Embedding change |
| **SOM** | Raileanu et al. | 2018 | Goal drift |
| **LIAM** | Papoudakis et al. | 2021 | Embedding drift |
| **Context-Cond** | Perez et al. | 2018 | Context change |
| **MBOM-Lite** | Yu et al. | 2022 | Prediction error |

---

## Detailed Descriptions

### 1. Mamba (SSM Architecture)

**Paper:** Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)

Uses the official `mamba-ssm` package when available. Tests whether UA-ToM's gains come from the SSM architecture or from belief mechanisms.

```python
from ua_tom.models.baselines import MambaNoBeliefBaseline, MAMBA_AVAILABLE

model = MambaNoBeliefBaseline(config)
print(f"Using official package: {MAMBA_AVAILABLE}")
```

### 2. BOCPD (Bayesian Online Changepoint Detection)

**Paper:** Adams & MacKay, "Bayesian Online Changepoint Detection" (2007)

Exact Algorithm 1 with Student-t conjugate prior. Gold-standard for changepoint detection.

```python
from ua_tom.models.baselines import BOCPDBaseline, BOCPDDetector

model = BOCPDBaseline(config, hazard_lambda=50.0)

# Standalone detector
detector = BOCPDDetector()
cp_prob = detector.update(observation)
```

### 3. ToMnet (Machine Theory of Mind)

**Paper:** Rabinowitz et al., "Machine Theory of Mind", ICML 2018

Three-network architecture separating stable traits from dynamic mental state:
- **Character Network**: Past episodes → stable agent embedding
- **Mental State Network**: Current trajectory → dynamic beliefs
- **Prediction Network**: Combined prediction

```python
from ua_tom.models.baselines import ToMnetBaseline, ToMnetConfig

config = ToMnetConfig(
    char_embed_dim=64,
    mental_embed_dim=64,
    num_past_episodes=5,
)
model = ToMnetBaseline(model_config, tomnet_config=config)
```

### 4. DRON (Deep Reinforcement Opponent Network)

**Paper:** He et al., "Opponent Modeling in Deep Reinforcement Learning", ICML 2016

Encodes opponent observations into DQN. Two variants:
- **DRON-Concat**: Concatenate state and opponent embedding
- **DRON-MoE**: Mixture of Experts for different opponent types

```python
from ua_tom.models.baselines import DRONBaseline

model = DRONBaseline(
    config,
    opponent_embed_dim=64,
    use_mixture_of_experts=True,
    num_experts=4,
)
```

### 5. SOM (Self Other-Modeling)

**Paper:** Raileanu et al., "Modeling Others using Oneself in Multi-Agent RL", ICML 2018

Uses own policy to infer partner's hidden goal via maximum likelihood.

```python
from ua_tom.models.baselines import SOMBaseline, SOMConfig

som_config = SOMConfig(goal_dim=16, use_variational=True)
model = SOMBaseline(config, som_config=som_config)
```

### 6. LIAM (Local Information Agent Modeling)

**Paper:** Papoudakis et al., "Agent Modelling under Partial Observability", NeurIPS 2021

Encoder-decoder that learns to reconstruct partner information from local trajectory only.

```python
from ua_tom.models.baselines import LIAMBaseline, LIAMConfig

liam_config = LIAMConfig(embed_dim=64, reconstruction_weight=1.0)
model = LIAMBaseline(config, liam_config=liam_config)
```

### 7. Context-Conditional (FiLM)

**Paper:** Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018

Sliding window attention for context + FiLM modulation layers.

```python
from ua_tom.models.baselines import ContextConditionalBaseline

model = ContextConditionalBaseline(
    config,
    window_size=20,
    num_heads=4,
)
```

### 8. Context-Conditional Policy (VAE)

VAE-based context inference, similar to PEARL/VariBAD approaches.

```python
from ua_tom.models.baselines import ContextConditionalPolicy, ContextConditionalConfig

cc_config = ContextConditionalConfig(context_dim=32, use_variational=True)
model = ContextConditionalPolicy(config, cc_config=cc_config)
```

### 9. MBOM-Lite (Model-Based Opponent Modeling)

**Paper:** Yu et al., "Model-Based Opponent Modeling", NeurIPS 2022

Simplified version: learns opponent dynamics, detects switches via prediction error.

```python
from ua_tom.models.baselines import MBOMLite

model = MBOMLite(config, dynamics_hidden=128)
```

### 10. Latent Opponent VAE

VAE for structured latent opponent representation with KL regularization.

```python
from ua_tom.models.baselines import LatentOpponentVAE

model = LatentOpponentVAE(config, latent_dim=32, kl_weight=0.01)
```

---

## Full Registry

```python
BASELINE_REGISTRY = {
    # Strong baselines
    'mamba': MambaNoBeliefBaseline,
    'bocpd': BOCPDBaseline,
    'tomnet': ToMnetBaseline,
    'context_cond': ContextConditionalBaseline,
    'dron': DRONBaseline,
    'vae_opponent': LatentOpponentVAE,
    
    # Extended opponent modeling
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
}
```

---

## Expected Results

| Baseline | Action Acc | Switch F1 | Hypothesis |
|----------|------------|-----------|------------|
| **UA-ToM** | Best | Best | Full system |
| Mamba | ≈UA-ToM | Lower | Architecture alone insufficient |
| BOCPD | Lower | Variable | Classical, no learned features |
| ToMnet | Good | Lower | No explicit switch detection |
| DRON/SOM | Good | Lower | Implicit opponent modeling |
| LIAM | Good | Medium | Local trajectory limited |
| GRU | Lower | Lower | Weak baseline |

---

## References

1. Gu & Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces
2. Adams & MacKay (2007). Bayesian Online Changepoint Detection
3. Rabinowitz et al. (2018). Machine Theory of Mind. ICML
4. He et al. (2016). Opponent Modeling in Deep Reinforcement Learning. ICML
5. Raileanu et al. (2018). Modeling Others using Oneself in Multi-Agent RL. ICML
6. Papoudakis et al. (2021). Agent Modelling under Partial Observability. NeurIPS
7. Perez et al. (2018). FiLM: Visual Reasoning with Conditioning. AAAI
8. Yu et al. (2022). Model-Based Opponent Modeling. NeurIPS
