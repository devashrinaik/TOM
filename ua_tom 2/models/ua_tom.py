"""
UA-ToM: Unified Adaptive Theory of Mind

Core architecture combining:
1. Selective State Space Model (Mamba-inspired) for belief tracking
2. Multi-layer Causal Attention for pattern recognition
3. Hierarchical prediction error (action, pattern, stability)
4. Contrastive partner memory (MoCo-style) for zero-shot transfer

This is the main model for the IROS paper.
"""

from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, ModelConfig, ModelOutput, RMSNorm
from .backbones.vla_encoder import get_encoder, LoRALayer


# =============================================================================
# SELECTIVE STATE SPACE MODEL (Mamba-Inspired)
# =============================================================================

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model for belief tracking.

    Mamba-inspired architecture with:
    - conv1d for local context aggregation
    - Input-dependent A, B, C parameters via learned projections
    - ZOH (Zero-Order Hold) discretization
    - Gated skip connection

    The selective gating learns WHEN to update state:
    - High surprise → large dt → big state update (belief revision)
    - Low surprise → small dt → incremental update (belief refinement)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # Input projection (expand to 2x for gated path)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal depthwise convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # Input-dependent step size (controls update magnitude)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # A: state transition matrix (log-parameterized for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # B, C: input/output projections (input-dependent)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # D: skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_state_changes: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, T, D] input sequence
            return_state_changes: Whether to return per-timestep state change magnitudes
        Returns:
            y: [B, T, D] output sequence
            state_changes: [B, T] magnitude of state updates (if requested)
        """
        B, T, D = x.shape

        # Input projection → two paths
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # Local convolution (causal: trim future)
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Compute input-dependent SSM parameters
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        dt = F.softplus(self.dt_proj(x_conv))  # [B, T, d_inner]
        B_val = self.B_proj(x_conv)  # [B, T, d_state]
        C_val = self.C_proj(x_conv)  # [B, T, d_state]

        # ZOH discretization: dA = exp(dt * A), dB = dt * B
        dA = torch.exp(torch.einsum('btd,dn->btdn', dt, A))  # [B, T, d_inner, d_state]
        dB = torch.einsum('btd,btn->btdn', dt, B_val)  # [B, T, d_inner, d_state]

        # Selective scan (sequential recurrence)
        y, state_changes = self._selective_scan(x_conv, dA, dB, C_val, return_state_changes)

        # Gated skip connection
        y = y * F.silu(z) + x_conv * self.D

        # Output projection + residual
        y = self.norm(self.dropout(self.out_proj(y)) + x)

        return y, state_changes

    def _selective_scan(
        self,
        x: torch.Tensor,
        dA: torch.Tensor,
        dB: torch.Tensor,
        C: torch.Tensor,
        return_state_changes: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        State equation: h_t = dA_t * h_{t-1} + dB_t * x_t
        Output equation: y_t = C_t * h_t
        """
        B, T, d_inner = x.shape
        d_state = dA.shape[-1]
        device = x.device

        h = torch.zeros(B, d_inner, d_state, device=device)
        outputs = []
        state_changes = [] if return_state_changes else None

        for t in range(T):
            if return_state_changes:
                h_prev = h.clone()

            # State update
            h = dA[:, t] * h + dB[:, t] * x[:, t, :, None]

            # Output
            y_t = torch.einsum('bdn,bn->bd', h, C[:, t])
            outputs.append(y_t)

            if return_state_changes:
                change = (h - h_prev).abs().mean(dim=(1, 2))  # [B]
                state_changes.append(change)

        y = torch.stack(outputs, dim=1)  # [B, T, d_inner]

        if return_state_changes:
            state_changes = torch.stack(state_changes, dim=1)  # [B, T]

        return y, state_changes


# =============================================================================
# HIERARCHICAL PREDICTION ERROR
# =============================================================================

class HierarchicalPredictionError(nn.Module):
    """
    Multi-scale prediction error for switch detection.

    Three levels of prediction:
    1. Action (horizon=1): Predict next action → action surprise
    2. Pattern (horizon=H): Predict next H actions → pattern surprise
    3. Stability: Predict type stability → type surprise

    Different error profiles signal different events:
    - High action only → unusual action, same partner
    - High pattern → partner behavior changing
    - High stability → partner has switched
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        horizons: List[int] = [1, 5, 20],
    ):
        super().__init__()

        self.action_dim = action_dim
        self.horizons = horizons
        self.pattern_horizon = horizons[1] if len(horizons) > 1 else 5

        # Level 1: Immediate action predictor
        self.action_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Level 2: Pattern predictor (next H actions simultaneously)
        self.pattern_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * self.pattern_horizon),
        )

        # Level 3: Stability predictor (stable vs switched)
        self.stability_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        # Error integration (3 error signals → switch signal)
        self.error_gate = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        state_changes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden: [B, T, hidden_dim]
            actions: [B, T] observed actions
            types: [B, T] partner types (for stability supervision)
            state_changes: [B, T] from SSM (unused, kept for API compat)
        Returns:
            Dictionary with predictions and switch signal
        """
        B, T, D = hidden.shape
        device = hidden.device
        H = self.pattern_horizon

        # Level 1: Action prediction error
        action_logits = self.action_pred(hidden)
        action_error = torch.zeros(B, T, device=device)
        if T > 1:
            pred_error = F.cross_entropy(
                action_logits[:, :-1].reshape(-1, self.action_dim),
                actions[:, 1:].reshape(-1),
                reduction='none',
            ).reshape(B, T - 1)
            action_error[:, 1:] = pred_error

        # Level 2: Pattern prediction error (multi-horizon)
        pattern_raw = self.pattern_pred(hidden)  # [B, T, H*A]
        pattern_logits = pattern_raw.reshape(B, T, H, self.action_dim)

        pattern_error = torch.zeros(B, T, device=device)
        for h in range(min(H, T - 1)):
            valid_t = T - 1 - h
            if valid_t > 0:
                pred = pattern_logits[:, :valid_t, h, :]
                target = actions[:, 1 + h:1 + h + valid_t]
                err = F.cross_entropy(
                    pred.reshape(-1, self.action_dim),
                    target.reshape(-1),
                    reduction='none',
                ).reshape(B, valid_t)
                pattern_error[:, :valid_t] += err / H

        # Level 3: Stability error
        stability_logits = self.stability_pred(hidden)
        stability_error = torch.zeros(B, T, device=device)

        if types is not None:
            type_changed = torch.zeros(B, T, dtype=torch.long, device=device)
            type_changed[:, 1:] = (types[:, 1:] != types[:, :-1]).long()
            stability_error = F.cross_entropy(
                stability_logits.reshape(-1, 2),
                type_changed.reshape(-1),
                reduction='none',
            ).reshape(B, T)

        # Normalize and fuse errors into switch signal
        def safe_normalize(x):
            return x / (x.mean() + 1e-6)

        errors = torch.stack([
            safe_normalize(action_error),
            safe_normalize(pattern_error),
            safe_normalize(stability_error),
        ], dim=-1)  # [B, T, 3]

        switch_signal = self.error_gate(errors).squeeze(-1)

        return {
            'action_logits': action_logits,
            'pattern_logits': pattern_logits,
            'stability_logits': stability_logits,
            'switch_signal': switch_signal,
            'action_error': action_error,
            'pattern_error': pattern_error,
            'stability_error': stability_error,
        }


# =============================================================================
# CONTRASTIVE PARTNER MEMORY (MoCo-Style)
# =============================================================================

class ContrastivePartnerMemory(nn.Module):
    """
    Contrastive learning module with external memory bank.

    Uses MoCo-style momentum encoder for stable partner prototypes.
    Enables zero-shot transfer to novel partners via:
    - Discriminative embeddings (InfoNCE loss)
    - Memory bank queue of past partner representations
    - Learnable prototype centroids
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int = 128,
        memory_size: int = 256,
        temperature: float = 0.07,
        momentum: float = 0.999,
        num_prototypes: int = 16,
    ):
        super().__init__()

        self.memory_dim = memory_dim
        self.memory_size = memory_size
        self.temperature = temperature
        self.momentum = momentum

        # Query encoder (main, receives gradients)
        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, memory_dim),
        )

        # Key encoder (momentum-updated copy, no gradients)
        self.key_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, memory_dim),
        )

        # Initialize key encoder from query encoder
        for param_q, param_k in zip(
            self.query_encoder.parameters(),
            self.key_encoder.parameters(),
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Memory bank (FIFO queue of partner embeddings)
        self.register_buffer(
            'memory_bank',
            F.normalize(torch.randn(memory_size, memory_dim), dim=-1),
        )
        self.register_buffer('memory_labels', torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # Learnable prototypes (centroids for partner types)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, memory_dim))
        nn.init.xavier_uniform_(self.prototypes)

    @torch.no_grad()
    def _momentum_update(self):
        """Update key encoder parameters with exponential moving average."""
        for param_q, param_k in zip(
            self.query_encoder.parameters(),
            self.key_encoder.parameters(),
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor, labels: torch.Tensor):
        """Update memory bank FIFO queue with new keys."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.memory_size:
            self.memory_bank[ptr:ptr + batch_size] = keys
            self.memory_labels[ptr:ptr + batch_size] = labels
        else:
            remaining = self.memory_size - ptr
            self.memory_bank[ptr:] = keys[:remaining]
            self.memory_labels[ptr:] = labels[:remaining]
            self.memory_bank[:batch_size - remaining] = keys[remaining:]
            self.memory_labels[:batch_size - remaining] = labels[remaining:]

        self.queue_ptr[0] = (ptr + batch_size) % self.memory_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [B, T, D]
            types: [B, T] partner type labels (needed for contrastive loss)
            update_memory: Whether to update memory bank
        Returns:
            query_embeds: [B, T, memory_dim]
            contrastive_loss: scalar InfoNCE loss
            prototype_sim: [B, T, num_prototypes]
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device

        # Query embeddings
        query_embeds = F.normalize(self.query_encoder(hidden_states), dim=-1)

        # Key embeddings (momentum encoder, no grad)
        with torch.no_grad():
            self._momentum_update()
            key_embeds = F.normalize(self.key_encoder(hidden_states), dim=-1)

        # Compute contrastive loss
        contrastive_loss = torch.tensor(0.0, device=device)
        if types is not None:
            queries_flat = query_embeds.reshape(B * T, -1)
            keys_flat = key_embeds.reshape(B * T, -1)
            types_flat = types.reshape(B * T)

            # Similarity to memory bank + batch keys
            memory_sim = torch.mm(queries_flat, self.memory_bank.t()) / self.temperature
            batch_sim = torch.mm(queries_flat, keys_flat.t()) / self.temperature

            # Positive mask (same type, exclude self)
            type_match = types_flat.unsqueeze(1) == types_flat.unsqueeze(0)
            self_mask = torch.eye(B * T, dtype=torch.bool, device=device)
            type_match = type_match & ~self_mask

            has_positive = type_match.any(dim=1)
            if has_positive.any():
                valid_batch_sim = batch_sim[has_positive]
                valid_memory_sim = memory_sim[has_positive]
                valid_type_match = type_match[has_positive]

                memory_type_match = (
                    types_flat[has_positive].unsqueeze(1) == self.memory_labels.unsqueeze(0)
                )

                all_sim = torch.cat([valid_batch_sim, valid_memory_sim], dim=1)
                all_pos_mask = torch.cat([valid_type_match, memory_type_match], dim=1)

                exp_sim = torch.exp(all_sim)
                pos_sum = (exp_sim * all_pos_mask.float()).sum(dim=1)
                all_sum = exp_sim.sum(dim=1)
                contrastive_loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8).mean()

        # Update memory bank
        if update_memory and self.training and types is not None:
            sample_idx = torch.randint(0, T, (B,), device=device)
            sampled_keys = key_embeds[torch.arange(B, device=device), sample_idx]
            sampled_labels = types[torch.arange(B, device=device), sample_idx]
            self._dequeue_and_enqueue(sampled_keys, sampled_labels)

        # Prototype similarities
        prototypes_norm = F.normalize(self.prototypes, dim=-1)
        prototype_sim = torch.mm(
            query_embeds.reshape(-1, self.memory_dim),
            prototypes_norm.t(),
        ).reshape(B, T, -1)

        return query_embeds, contrastive_loss, prototype_sim


# =============================================================================
# IN-CONTEXT AGGREGATOR (Multi-Layer Causal Attention)
# =============================================================================

class InContextAggregator(nn.Module):
    """
    Multi-layer causal attention with relative positional encodings.

    Learns to attend to the most informative past timesteps adaptively,
    with no fixed window limitation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        # Learnable relative position embeddings
        self.rel_pos_embed = nn.Parameter(
            torch.randn(max_seq_len, hidden_dim) * 0.02
        )

        # Multi-layer attention blocks
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn_norm': RMSNorm(hidden_dim),
                'attn': nn.MultiheadAttention(
                    hidden_dim, num_heads,
                    dropout=dropout, batch_first=True,
                ),
                'ffn_norm': RMSNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input sequence
        Returns:
            [B, T, D] contextualized representations
        """
        B, T, D = x.shape
        device = x.device

        # Add relative position information
        x = x + self.rel_pos_embed[:T].unsqueeze(0)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=device), diagonal=1
        ).bool()

        for layer in self.layers:
            # Pre-norm self-attention
            normed = layer['attn_norm'](x)
            attn_out, _ = layer['attn'](
                normed, normed, normed,
                attn_mask=causal_mask,
            )
            x = x + attn_out

            # Pre-norm feed-forward
            x = x + layer['ffn'](layer['ffn_norm'](x))

        return self.norm(x)


# =============================================================================
# UA-ToM: MAIN MODEL
# =============================================================================

class UAToM(BaseModel):
    """
    Unified Adaptive Theory of Mind.

    Architecture:
        Input → [Encoder] → z_t
                              ↓
        Path 1: [Selective SSM] → h_ssm    (belief tracking)
        Path 2: [InContext Agg] → h_attn    (pattern recognition)
                              ↓
                    [Fusion] → fused
                              ↓
        [Contrastive Memory] → partner embeddings + InfoNCE loss
        [Hierarchical Pred Error] → multi-scale switch signal
                              ↓
        [Heads] → action_logits, type_logits, switch_probs

    The dual-path design allows:
    - SSM: O(T) belief state tracking with selective updates
    - Attention: Capturing longer-range patterns in partner behavior

    The contrastive memory enables zero-shot transfer to novel partners.
    """

    def __init__(
        self,
        config: ModelConfig,
        encoder_type: str = 'cnn',
        use_vision: bool = True,
    ):
        super().__init__(config)

        self.use_vision = use_vision

        # Visual encoder (optional)
        if use_vision:
            self.encoder = get_encoder(
                encoder_type,
                output_dim=config.obs_dim,
                training_mode=config.training_mode,
            )

            if config.training_mode == 'lora':
                self.lora = LoRALayer(config.obs_dim, config.obs_dim, config.lora_rank)
            else:
                self.lora = None

            if config.training_mode == 'frozen':
                for p in self.encoder.parameters():
                    p.requires_grad = False
        else:
            self.encoder = None
            self.lora = None

        # Input projection
        input_dim = config.obs_dim + config.action_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)

        # Path 1: Selective SSM for belief tracking
        self.ssm = SelectiveSSM(
            d_model=config.hidden_dim,
            d_state=config.ssm_dim,
            d_conv=config.ssm_conv_width,
            dropout=config.dropout,
        )

        # Path 2: In-Context Aggregator (multi-layer attention)
        self.context_aggregator = InContextAggregator(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        # Fusion of both paths
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            RMSNorm(config.hidden_dim),
        )

        # Hierarchical prediction error (with pattern prediction)
        self.pred_error = HierarchicalPredictionError(
            config.hidden_dim,
            config.action_dim,
            horizons=config.prediction_horizons,
        )

        # Contrastive partner memory
        self.partner_memory = ContrastivePartnerMemory(
            hidden_dim=config.hidden_dim,
            memory_dim=config.memory_dim,
            memory_size=config.memory_size,
            temperature=config.temperature,
        )

        # Output heads
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.type_head = nn.Linear(config.hidden_dim, config.num_types)

        # Switch detection head (fused repr + 3 normalized signals)
        self.switch_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 3, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        B, T = actions.shape
        device = actions.device

        # Encode observations
        if images is not None and self.encoder is not None:
            if self.config.training_mode == 'frozen':
                with torch.no_grad():
                    obs = self.encoder(images)
            else:
                obs = self.encoder(images)

            if self.lora is not None:
                obs = obs + self.lora(obs)
        else:
            obs = observations

        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)

        # Embed inputs
        act_oh = F.one_hot(actions, self.action_dim).float()
        hidden = self.input_proj(torch.cat([obs, act_oh], dim=-1))

        # Path 1: Selective SSM
        ssm_out, state_changes = self.ssm(hidden, return_state_changes=True)

        # Path 2: In-Context Aggregator
        attn_out = self.context_aggregator(hidden)

        # Fuse both paths
        fused = self.fusion(torch.cat([ssm_out, attn_out], dim=-1))

        # Hierarchical prediction error
        pred_out = self.pred_error(fused, actions, types, state_changes)
        error_signal = pred_out['switch_signal']

        # Contrastive partner memory
        partner_embeds, contrastive_loss, prototype_sim = self.partner_memory(
            fused, types,
        )

        # Action and type predictions
        action_logits = self.action_head(fused)
        type_logits = self.type_head(fused)

        # Belief changes (from type distribution shift)
        type_probs = F.softmax(type_logits, dim=-1)
        belief_changes = torch.zeros(B, T, device=device)
        belief_changes[:, 1:] = (type_probs[:, 1:] - type_probs[:, :-1]).abs().sum(dim=-1)

        # Switch detection (fused features + 3 normalized signals)
        state_changes_norm = state_changes / (state_changes.mean() + 1e-6)
        belief_changes_norm = belief_changes / (belief_changes.mean() + 1e-6)

        switch_features = torch.cat([
            fused,
            state_changes_norm.unsqueeze(-1),
            error_signal.unsqueeze(-1),
            belief_changes_norm.unsqueeze(-1),
        ], dim=-1)
        switch_probs = self.switch_head(switch_features).squeeze(-1)

        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
            state_changes=state_changes,
            error_signal=error_signal,
            pred_action_logits=pred_out['action_logits'],
            pred_pattern_logits=pred_out['pattern_logits'],
            pred_stability_logits=pred_out['stability_logits'],
            contrastive_loss=contrastive_loss,
            partner_embeds=partner_embeds,
        )


# =============================================================================
# CONVENIENCE VARIANTS
# =============================================================================

class UAToMFrozen(UAToM):
    """UA-ToM with frozen VLA backbone (default configuration)."""

    def __init__(self, config: ModelConfig, **kwargs):
        config.training_mode = 'frozen'
        super().__init__(config, **kwargs)


class UAToMLoRA(UAToM):
    """UA-ToM with LoRA fine-tuning of VLA backbone."""

    def __init__(self, config: ModelConfig, **kwargs):
        config.training_mode = 'lora'
        super().__init__(config, **kwargs)


class UAToMFull(UAToM):
    """UA-ToM with full fine-tuning of VLA backbone."""

    def __init__(self, config: ModelConfig, **kwargs):
        config.training_mode = 'full'
        super().__init__(config, **kwargs)
