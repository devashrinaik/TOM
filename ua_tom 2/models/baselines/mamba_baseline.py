"""
Mamba Baseline (No Belief Heads)
================================

Uses the OFFICIAL Mamba implementation from Tri Dao's mamba-ssm package.

Paper: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective 
       State Spaces", arXiv:2312.00752 (2023)
Code:  https://github.com/state-spaces/mamba

This baseline uses the same powerful SSM architecture but WITHOUT any 
belief tracking mechanism. Switch detection is post-hoc via hidden 
state change magnitude.

This is the CRITICAL architectural control - it tests whether UA-ToM's
gains come from the SSM architecture or from the belief mechanism.

Installation:
    pip install mamba-ssm causal-conv1d>=1.4.0

Requirements:
    - Linux
    - NVIDIA GPU
    - PyTorch 1.12+
    - CUDA 11.6+

If mamba-ssm is unavailable, falls back to a faithful pure-PyTorch 
reimplementation following the paper exactly (Algorithm 2, Section 3.4).

Key architectural details (from paper):
    - d_state (N): SSM state expansion, default 16
    - d_conv: Local convolution width, default 4  
    - expand (E): Block expansion factor, default 2
    - dt_rank: Rank for Δ projection, ceil(d_model/16)
    - A: Diagonal, initialized as S4D-Real: A[n] = -(n+1)
    - B, C, Δ: Input-dependent (selective) - key innovation
    - D: Skip connection (residual)
"""

import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel, ModelConfig, ModelOutput, RMSNorm


# =============================================================================
# Try to import official Mamba
# =============================================================================

MAMBA_AVAILABLE = False
MAMBA_VERSION = None
try:
    from mamba_ssm import Mamba
    import mamba_ssm
    MAMBA_AVAILABLE = True
    MAMBA_VERSION = getattr(mamba_ssm, '__version__', 'unknown')
except ImportError:
    pass


# =============================================================================
# Faithful Mamba Block Reimplementation (when mamba-ssm unavailable)
# =============================================================================

class MambaBlockPure(nn.Module):
    """
    Pure PyTorch reimplementation of Mamba block from Gu & Dao 2023.
    
    This follows the paper EXACTLY (Section 3.4, Algorithm 2):
    
    Architecture:
        x → in_proj → [x_branch, z]
        x_branch → conv1d → SiLU → SSM → y
        output = out_proj(y * SiLU(z))
    
    SSM with selective scan:
        x → x_proj → [Δ, B, C]  (input-dependent!)
        Δ → dt_proj → softplus → discretize A, B
        h_t = Ā·h_{t-1} + B̄·x_t
        y_t = C_t·h_t + D·x_t
    
    Key differences from S4/S5:
        - B, C, Δ are input-dependent (selective)
        - A, D are input-independent but learned
        - Uses ZOH discretization: Ā = exp(ΔA), B̄ ≈ ΔB
    
    Reference:
        Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling 
        with Selective State Spaces. arXiv:2312.00752
        
        Algorithm 2, Section 3.4
        Code: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        """
        Args:
            d_model: Input/output dimension (D in paper)
            d_state: SSM state expansion factor (N in paper), default 16
            d_conv: Local convolution width, default 4
            expand: Block expansion factor (E in paper), default 2
            dt_rank: Rank for Δ projection, "auto" = ceil(d_model/16)
            dt_min, dt_max: Range for Δ initialization
            dt_init: Initialization type for dt_proj ("random" or "constant")
            dt_scale: Scale for dt_proj initialization
            dt_init_floor: Minimum value for Δ initialization
            bias: Use bias in linear projections
            conv_bias: Use bias in conv1d
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # dt_rank from paper: ceil(D / 16)
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = int(dt_rank)
        
        # === Projections (following mamba_simple.py exactly) ===
        
        # Input projection: d_model → 2 * d_inner (for x and z branches)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # Depthwise causal conv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # Causal: pad left only
            groups=self.d_inner,  # Depthwise
            bias=conv_bias,
        )
        
        # === SSM Parameters ===
        
        # A: Diagonal matrix, NOT input-dependent
        # Initialize as S4D-Real: A[n] = -(n+1) for n = 0, 1, ..., N-1
        # Store as log for numerical stability (ensures A < 0)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)  # [d_inner, d_state]
        self.A_log = nn.Parameter(torch.log(A))
        
        # D: Skip connection (residual), one per channel
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # x_proj: Input-dependent B, C, Δ
        # Projects: d_inner → dt_rank + d_state + d_state
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # dt_proj: Projects low-rank Δ to full dimension
        # dt_rank → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # === Initialization (following paper exactly) ===
        
        # Initialize dt_proj.weight
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt_proj.bias so softplus(bias) ∈ [dt_min, dt_max]
        # This is crucial for stability (see paper Section 3.5)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) 
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: x = log(exp(y) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Prevent reinitialization by other frameworks
        self.dt_proj.bias._no_reinit = True
        
        # Output projection: d_inner → d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        
        # For tracking SSM states (used in switch detection)
        self._last_ssm_states = None
    
    def forward(
        self, 
        x: torch.Tensor,
        return_ssm_states: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through Mamba block.
        
        Args:
            x: Input tensor [B, T, d_model]
            return_ssm_states: If True, store SSM states for analysis
        
        Returns:
            y: Output tensor [B, T, d_model]
        """
        B, T, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # [B, T, 2 * d_inner]
        x_branch, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]
        
        # Causal convolution
        x_conv = x_branch.transpose(1, 2)  # [B, d_inner, T]
        x_conv = self.conv1d(x_conv)[:, :, :T]  # Truncate for causality
        x_conv = x_conv.transpose(1, 2)  # [B, T, d_inner]
        x_conv = F.silu(x_conv)
        
        # Selective SSM
        y, ssm_states = self.selective_scan(x_conv, return_states=return_ssm_states)
        
        if return_ssm_states:
            self._last_ssm_states = ssm_states
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        
        return y
    
    def selective_scan(
        self, 
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Selective Scan (SSM) - Algorithm 2 from the paper.
        
        This implements the core Mamba operation:
            h_t = Ā_t · h_{t-1} + B̄_t · x_t
            y_t = C_t · h_t + D · x_t
        
        Where Ā_t, B̄_t, C_t are input-dependent (selective).
        
        Discretization (Zero-Order Hold):
            Ā = exp(Δ · A)
            B̄ = (Δ · A)^{-1} · (exp(Δ · A) - I) · Δ · B ≈ Δ · B
        
        Args:
            x: Input after conv [B, T, d_inner]
            return_states: Whether to return all SSM states
        
        Returns:
            y: Output [B, T, d_inner]
            states: SSM states [B, T, d_inner, d_state] if return_states else None
        """
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Input-dependent projections
        x_dbl = self.x_proj(x)  # [B, T, dt_rank + 2*d_state]
        dt_rank_out, B_param, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # Project Δ and apply softplus for positivity
        dt = self.dt_proj(dt_rank_out)  # [B, T, d_inner]
        dt = F.softplus(dt)
        
        # Sequential scan
        h = torch.zeros(B, self.d_inner, self.d_state, device=device, dtype=torch.float32)
        outputs = []
        states = [] if return_states else None
        
        for t in range(T):
            # Current timestep values
            dt_t = dt[:, t, :, None]  # [B, d_inner, 1]
            B_t = B_param[:, t, None, :]  # [B, 1, d_state]
            C_t = C[:, t, None, :]  # [B, 1, d_state]
            x_t = x[:, t, :, None]  # [B, d_inner, 1]
            
            # Discretize
            # Ā = exp(Δ · A), where A is [d_inner, d_state]
            dA = dt_t * A.unsqueeze(0)  # [B, d_inner, d_state]
            A_bar = torch.exp(dA)
            
            # B̄ ≈ Δ · B (first-order approximation, standard in SSMs)
            B_bar = dt_t * B_t  # [B, d_inner, d_state]
            
            # State update
            h = A_bar * h + B_bar * x_t
            
            # Output
            y_t = (C_t * h).sum(dim=-1) + self.D * x[:, t]  # [B, d_inner]
            outputs.append(y_t)
            
            if return_states:
                states.append(h.clone())
        
        y = torch.stack(outputs, dim=1)  # [B, T, d_inner]
        
        if return_states:
            states = torch.stack(states, dim=1)  # [B, T, d_inner, d_state]
        
        return y.to(dtype), states


class MambaBlockWithStateTracking(nn.Module):
    """
    Wrapper around official Mamba that tracks hidden states for switch detection.
    
    Since the official Mamba doesn't expose internal states by default,
    we wrap it and compute approximate state changes from output differences.
    """
    
    def __init__(self, mamba_block):
        super().__init__()
        self.mamba = mamba_block
        self._last_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mamba(x)
        self._last_output = y.detach()
        return y


# =============================================================================
# Main Baseline Class
# =============================================================================

class MambaNoBeliefBaseline(BaseModel):
    """
    Mamba baseline WITHOUT belief tracking.
    
    This is the CRITICAL architectural control for UA-ToM experiments.
    
    Architecture:
        - Uses official Mamba blocks (or faithful pure-PyTorch reimplementation)
        - Multiple stacked layers with RMSNorm and residual connections
        - Action and type prediction heads
        - NO explicit belief mechanism
        - NO hierarchical prediction errors
        - NO attention path
        - NO learned switch detector
    
    Switch Detection:
        Post-hoc via hidden state change magnitude:
        switch_signal = ||h_t - h_{t-1}||_2
        
        This tests whether explicit belief tracking is necessary
        or if implicit state dynamics suffice.
    
    Expected Results:
        ✓ Similar or better action accuracy than GRU/Transformer
        ✓ Comparable to UA-ToM on static partner prediction
        ✗ Worse switch detection (especially precision)
        ✗ Higher false positive rate (no selective gating)
        ✗ Slower adaptation after switches
    
    This validates that UA-ToM's gains come from the belief mechanism,
    not just the SSM architecture.
    
    Reference:
        Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling 
        with Selective State Spaces. arXiv:2312.00752
    """
    
    def __init__(
        self,
        config: ModelConfig,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
        track_ssm_states: bool = True,
    ):
        """
        Args:
            config: Model configuration
            d_state: SSM state expansion factor (N in paper), default 16
            d_conv: Local convolution width, default 4
            expand: Block expansion factor (E in paper), default 2
            n_layers: Number of stacked Mamba blocks
            track_ssm_states: Whether to track SSM states for switch detection
        """
        super().__init__(config)
        
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_layers = n_layers
        self.track_ssm_states = track_ssm_states
        
        input_dim = config.obs_dim + config.action_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        
        # Mamba layers
        if MAMBA_AVAILABLE:
            print(f"[MambaBaseline] Using official mamba-ssm v{MAMBA_VERSION}")
            self.mamba_layers = nn.ModuleList([
                MambaBlockWithStateTracking(
                    Mamba(
                        d_model=config.hidden_dim,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                )
                for _ in range(n_layers)
            ])
            self.use_official = True
        else:
            print(f"[MambaBaseline] Using pure PyTorch reimplementation "
                  f"(install mamba-ssm for faster training)")
            self.mamba_layers = nn.ModuleList([
                MambaBlockPure(
                    d_model=config.hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ])
            self.use_official = False
        
        # Layer norms (RMSNorm as in original Mamba)
        self.norms = nn.ModuleList([
            RMSNorm(config.hidden_dim) for _ in range(n_layers)
        ])
        
        # Output heads
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.type_head = nn.Linear(config.hidden_dim, config.num_types)
        
        # For switch detection analysis
        self._layer_outputs = None
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Forward pass through stacked Mamba layers.
        
        Args:
            observations: [B, T, obs_dim] or [B, obs_dim]
            actions: [B, T] action indices
            types: [B, T] partner type indices (unused)
            images: [B, T, C, H, W] images (unused)
        
        Returns:
            ModelOutput with predictions and state change signals
        """
        B, T = actions.shape
        device = actions.device
        
        # Handle observation dimensions
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Embed inputs
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = self.input_proj(torch.cat([obs, act_oh], dim=-1))
        
        # Track outputs for switch detection
        layer_outputs = [x]
        
        # Forward through Mamba layers
        for mamba, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = mamba(x)
            x = norm(x + residual)
            layer_outputs.append(x)
        
        self._layer_outputs = layer_outputs
        
        # Compute state changes for switch detection
        state_changes = self._compute_state_changes(layer_outputs)
        
        # Predictions
        action_logits = self.action_head(x)
        type_logits = self.type_head(x)
        
        # Normalize state changes to [0, 1] for switch probabilities
        # Use per-sequence normalization
        sc_min = state_changes.min(dim=-1, keepdim=True)[0]
        sc_max = state_changes.max(dim=-1, keepdim=True)[0]
        switch_probs = (state_changes - sc_min) / (sc_max - sc_min + 1e-8)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=state_changes,
            state_changes=state_changes,
        )
    
    def _compute_state_changes(
        self, 
        layer_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute state change magnitude for post-hoc switch detection.
        
        Combines two signals:
        1. Temporal changes: ||h_t - h_{t-1}||_2 (frame-to-frame)
        2. Layer changes: ||h^{l}_t - h^{l-1}_t||_2 (residual magnitude)
        
        Args:
            layer_outputs: List of [B, T, D] tensors from each layer
        
        Returns:
            state_changes: [B, T] combined state change magnitude
        """
        B, T, D = layer_outputs[0].shape
        device = layer_outputs[0].device
        
        # Temporal changes in final layer output
        final_output = layer_outputs[-1]
        temporal_changes = torch.zeros(B, T, device=device)
        if T > 1:
            diff = final_output[:, 1:] - final_output[:, :-1]  # [B, T-1, D]
            temporal_changes[:, 1:] = torch.norm(diff, dim=-1)
        
        # Layer-wise residual changes (how much each layer modifies)
        layer_changes = torch.zeros(B, T, device=device)
        for i in range(1, len(layer_outputs)):
            diff = layer_outputs[i] - layer_outputs[i-1]
            layer_changes = layer_changes + torch.norm(diff, dim=-1)
        layer_changes = layer_changes / (len(layer_outputs) - 1)
        
        # Combine signals (equal weighting)
        combined = temporal_changes + layer_changes
        
        return combined
    
    def get_switch_signal(self, outputs: ModelOutput) -> torch.Tensor:
        """
        Get switch detection signal for evaluation.
        
        For the Mamba baseline, this is purely based on state change magnitude.
        No learned thresholding or gating.
        """
        # Z-score normalization for comparability
        sc = outputs.state_changes
        mean = sc.mean()
        std = sc.std() + 1e-8
        return (sc - mean) / std
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_param_breakdown(self) -> dict:
        """Get parameter count breakdown by component."""
        breakdown = {
            'input_proj': sum(p.numel() for p in self.input_proj.parameters()),
            'mamba_layers': sum(
                sum(p.numel() for p in layer.parameters()) 
                for layer in self.mamba_layers
            ),
            'norms': sum(
                sum(p.numel() for p in norm.parameters()) 
                for norm in self.norms
            ),
            'action_head': sum(p.numel() for p in self.action_head.parameters()),
            'type_head': sum(p.numel() for p in self.type_head.parameters()),
        }
        breakdown['total'] = sum(breakdown.values())
        return breakdown
