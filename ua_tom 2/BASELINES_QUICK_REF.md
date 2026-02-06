# UA-ToM Baselines Quick Reference

## All Available Baselines (17 total)

```python
from ua_tom.models.baselines import get_baseline, list_baselines

# List all
print(list_baselines())

# Get any by name
model = get_baseline('tomnet', config)
```

## By Category

### Theory of Mind
| Name | Class | Paper |
|------|-------|-------|
| `tomnet` | ToMnetBaseline | Rabinowitz 2018 |
| `tomnet_char_only` | ToMnetCharacterOnly | Ablation |
| `tomnet_mental_only` | ToMnetMentalOnly | Ablation |

### Opponent Modeling
| Name | Class | Paper |
|------|-------|-------|
| `dron` | DRONBaseline | He 2016 |
| `vae_opponent` | LatentOpponentVAE | Various |
| `oiam` | OIAMBaseline | Implicit modeling |
| `liam` | LIAMBaseline | Papoudakis 2021 |
| `som` | SOMBaseline | Raileanu 2018 |
| `mbom_lite` | MBOMLite | Yu 2022 |

### SSM / Architecture
| Name | Class | Paper |
|------|-------|-------|
| `mamba` | MambaNoBeliefBaseline | Gu & Dao 2023 |

### Classical
| Name | Class | Paper |
|------|-------|-------|
| `bocpd` | BOCPDBaseline | Adams & MacKay 2007 |

### Context-Conditional
| Name | Class | Paper |
|------|-------|-------|
| `context_cond` | ContextConditionalBaseline | Perez 2018 (FiLM) |
| `context_cond_gru` | ContextConditionalGRU | Simplified variant |
| `context_conditional_policy` | ContextConditionalPolicy | VAE-style |

### Weak Baselines
| Name | Class | Notes |
|------|-------|-------|
| `gru` | GRUBaseline | Simple GRU |
| `transformer` | TransformerBaseline | Causal attention |

### Legacy
| Name | Class | Notes |
|------|-------|-------|
| `btom` | BToM | Old implementation |

## Example Usage

```python
from ua_tom.models.base import ModelConfig
from ua_tom.models.baselines import (
    get_baseline,
    ToMnetBaseline, ToMnetConfig,
    LIAMBaseline, LIAMConfig,
    SOMBaseline, SOMConfig,
)

# Basic config
config = ModelConfig(
    obs_dim=64,
    action_dim=6,
    hidden_dim=128,
    num_types=4,
)

# Method 1: Factory function
model = get_baseline('tomnet', config)

# Method 2: Direct instantiation with custom config
tomnet_cfg = ToMnetConfig(char_embed_dim=64, num_past_episodes=5)
model = ToMnetBaseline(config, tomnet_config=tomnet_cfg)

# Method 3: Extended baselines
liam_cfg = LIAMConfig(embed_dim=64, reconstruction_weight=1.0)
model = LIAMBaseline(config, liam_config=liam_cfg)
```

## Switch Detection Methods

| Baseline | Detection Signal |
|----------|------------------|
| UA-ToM | Learned switch head |
| Mamba | SSM state magnitude |
| BOCPD | Run-length posterior |
| ToMnet | Mental state change |
| DRON/OIAM | Embedding change |
| LIAM | Embedding drift |
| SOM | Goal drift |
| Context-Cond | Context change |
| MBOM-Lite | Prediction error |

## File Locations

```
models/baselines/
├── mamba_baseline.py          # Mamba SSM
├── bocpd_baseline.py          # BOCPD classical
├── tomnet_baseline.py         # ToMnet + ablations
├── opponent_modeling.py       # DRON, VAE, OIAM
├── opponent_modeling_extended.py  # LIAM, SOM, Context-VAE, MBOM
├── context_conditional.py     # FiLM conditioning
├── temporal.py                # GRU, Transformer
└── tom.py                     # Legacy BToM
```
