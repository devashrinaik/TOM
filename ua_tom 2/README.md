# UA-ToM: Unified Adaptive Theory of Mind

> Lightweight belief tracking module for Vision-Language-Action models to enable coordination with non-stationary partners.

**Paper:** *UA-ToM: Uncertainty-Aware Theory of Mind for Adaptive Partner Modeling*
**Venue:** IROS 2026

## Overview

UA-ToM is a 650K parameter add-on module (0.16% of backbone) that augments frozen VLA backbones with adaptive belief tracking. When partners change behavior mid-episode, standard VLA policies degrade 7-15% with slow recovery. UA-ToM detects switches with 72-91% F1 (vs 3-12% baselines) and reduces post-switch degradation by 2×.

### Key Features

- **Dual-path temporal processing**: Selective SSM + Multi-layer Causal Attention
- **Hierarchical prediction errors**: Multi-scale signals (action, pattern, stability) for robust switch detection
- **Contrastive partner memory**: MoCo-style momentum encoder for zero-shot transfer
- **Plug-and-play**: Works with frozen VLA backbones (no fine-tuning needed)

## Installation

```bash
git clone https://github.com/xxx/ua_tom.git
cd ua_tom
pip install -r requirements.txt
```

## Quick Start

### Train UA-ToM

```bash
python scripts/train.py \
    --data_path data/pickcube.npz \
    --model ua_tom \
    --epochs 30 \
    --seeds 5 \
    --output_dir results/ua_tom
```

### Train All Baselines

```bash
python scripts/train.py \
    --data_path data/pickcube.npz \
    --all_baselines \
    --output_dir results/comparison
```

### Train Specific Baseline

```bash
python scripts/train.py \
    --data_path data/pickcube.npz \
    --model tomnet \
    --output_dir results/tomnet
```

## Project Structure

```
ua_tom/
├── models/
│   ├── base.py                  # Base class, ModelConfig, ModelOutput
│   ├── ua_tom.py                # Main UA-ToM model
│   ├── backbones/
│   │   └── vla_encoder.py       # VLA backbone interface
│   └── baselines/
│       ├── README.md            # Detailed baseline documentation
│       ├── mamba_baseline.py    # Mamba SSM (Gu & Dao 2023)
│       ├── bocpd_baseline.py    # BOCPD (Adams & MacKay 2007)
│       ├── tomnet_baseline.py   # ToMnet (Rabinowitz et al. 2018)
│       ├── opponent_modeling.py # DRON, VAE, OIAM
│       ├── opponent_modeling_extended.py  # LIAM, SOM, Context-VAE, MBOM
│       ├── context_conditional.py  # FiLM conditioning
│       ├── temporal.py          # GRU, Transformer (weak baselines)
│       └── tom.py               # Legacy BToM
├── data/
│   └── dataset.py               # Dataset classes
├── training/
│   └── trainer.py               # Training loop
├── evaluation/
│   └── metrics.py               # Evaluation metrics
└── scripts/
    └── train.py                 # Main training script
```

## Models

### Main Model

| Model | Description | Params |
|-------|-------------|--------|
| `ua_tom` | Full UA-ToM with dual-path + prediction error | 650K |

### Strong Baselines (16+)

| Model | Paper | Category |
|-------|-------|----------|
| `mamba` | Gu & Dao 2023 | SSM Architecture |
| `bocpd` | Adams & MacKay 2007 | Classical Bayesian |
| `tomnet` | Rabinowitz et al. 2018 | Theory of Mind |
| `dron` | He et al. 2016 | Opponent Modeling |
| `som` | Raileanu et al. 2018 | Self Other-Modeling |
| `liam` | Papoudakis et al. 2021 | Local Information |
| `context_cond` | Perez et al. 2018 | FiLM Conditioning |
| `context_conditional_policy` | PEARL/VariBAD style | VAE Context |
| `mbom_lite` | Yu et al. 2022 | Model-Based |
| `vae_opponent` | Various | Latent VAE |

### Weak Baselines

| Model | Description |
|-------|-------------|
| `gru` | Standard GRU encoder |
| `transformer` | Causal transformer |

### Ablations

| Model | Description |
|-------|-------------|
| `tomnet_char_only` | ToMnet character network only |
| `tomnet_mental_only` | ToMnet mental state only |
| `oiam` | Online Implicit Agent Modeling |

See `models/baselines/README.md` for detailed documentation.

## Data Format

Expected `.npz` format:

```python
{
    'images': np.array,          # [N, T, H, W, C] RGB images (optional)
    'observations': np.array,    # [N, T, obs_dim] state observations (optional)
    'partner_actions': np.array, # [N, T] or [N, T, action_dim]
    'partner_types': np.array,   # [N, T] type labels
}
```

At least one of `images` or `observations` must be present.

## Metrics

| Metric | Description |
|--------|-------------|
| `action_acc` | Action prediction accuracy |
| `type_acc` | Partner type classification accuracy |
| `switch_f1` | Event-level switch detection F1 (±1 tolerance) |
| `fp_rate` | False positive rate for switch detection |
| `degradation` | Pre-switch - post-switch accuracy |
| `recovery_time` | Steps to recover post-switch |

## Citation

```bibtex
@inproceedings{uatom2026,
  title={UA-ToM: Uncertainty-Aware Theory of Mind for Adaptive Partner Modeling},
  author={...},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```

## License

MIT License
