"""
Context-Conditional Baseline with FiLM
======================================

Uses Feature-wise Linear Modulation (FiLM) from Perez et al. (2018):
"FiLM: Visual Reasoning with a General Conditioning Layer"
https://arxiv.org/abs/1709.07871

This baseline tests whether rich context representations can substitute
for explicit belief tracking. It uses:
    1. A sliding window attention over recent (obs, action) pairs
    2. FiLM conditioning to modulate hidden representations
    3. Post-hoc switch detection via context change magnitude

FiLM Mechanism:
    Given context c and hidden state h:
        γ, β = MLP(c)
        output = γ ⊙ h + β
    
    This allows context to scale and shift features, enabling
    flexible modulation of the processing based on partner history.

Reference:
    Perez et al. (2018). FiLM: Visual Reasoning with a General 
    Conditioning Layer. AAAI.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel, ModelConfig, ModelOutput


# =============================================================================
# FiLM Layer (Perez et al. 2018)
# =============================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Applies affine transformation conditioned on context:
        output = γ ⊙ input + β
    
    Where γ (scale) and β (shift) are predicted from context.
    
    Reference:
        Perez et al. (2018). FiLM: Visual Reasoning with a General
        Conditioning Layer. AAAI 2018.
        https://arxiv.org/abs/1709.07871
    """
    
    def __init__(
        self,
        feature_dim: int,
        context_dim: int,
        hidden_dim: int = None,
    ):
        """
        Args:
            feature_dim: Dimension of input features to modulate
            context_dim: Dimension of conditioning context
            hidden_dim: Hidden dimension for γ, β predictor MLP
        """
        super().__init__()
        
        hidden_dim = hidden_dim or feature_dim
        
        # MLP to predict γ and β from context
        self.film_generator = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2),  # γ and β concatenated
        )
        
        # Initialize to identity transform (γ=1, β=0)
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        with torch.no_grad():
            # Set γ portion to 1
            self.film_generator[-1].bias[:feature_dim].fill_(1.0)
    
    def forward(
        self,
        features: torch.Tensor,  # [B, T, feature_dim]
        context: torch.Tensor,   # [B, T, context_dim]
    ) -> torch.Tensor:
        """
        Apply FiLM modulation.
        
        Args:
            features: Input features to modulate
            context: Conditioning context
        
        Returns:
            Modulated features [B, T, feature_dim]
        """
        # Predict γ and β
        film_params = self.film_generator(context)  # [B, T, 2*feature_dim]
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # Apply FiLM: γ ⊙ features + β
        output = gamma * features + beta
        
        return output


# =============================================================================
# Causal Sliding Window Attention
# =============================================================================

class CausalWindowAttention(nn.Module):
    """
    Causal attention over a sliding window of past observations.
    
    Unlike full self-attention, this only attends to the last W timesteps,
    providing bounded context with O(W²) complexity per timestep.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        window_size: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply causal sliding window attention.
        
        Args:
            x: Input tensor [B, T, hidden_dim]
        
        Returns:
            output: Attended output [B, T, hidden_dim]
            context: Context vectors for FiLM [B, T, hidden_dim]
        """
        B, T, D = x.shape
        H, W = self.num_heads, self.window_size
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, T, H, self.head_dim)
        K = self.k_proj(x).view(B, T, H, self.head_dim)
        V = self.v_proj(x).view(B, T, H, self.head_dim)
        
        # Compute attention for each timestep over its window
        outputs = []
        contexts = []
        
        for t in range(T):
            # Window start (causal: only past, including current)
            start = max(0, t - W + 1)
            window_len = t - start + 1
            
            # Get Q for current timestep, K/V for window
            q_t = Q[:, t:t+1, :, :]  # [B, 1, H, head_dim]
            k_window = K[:, start:t+1, :, :]  # [B, window_len, H, head_dim]
            v_window = V[:, start:t+1, :, :]  # [B, window_len, H, head_dim]
            
            # Attention scores
            # q_t: [B, 1, H, head_dim], k_window: [B, window_len, H, head_dim]
            # We want: [B, H, 1, window_len]
            q_t = q_t.transpose(1, 2)  # [B, H, 1, head_dim]
            k_window = k_window.transpose(1, 2)  # [B, H, window_len, head_dim]
            v_window = v_window.transpose(1, 2)  # [B, H, window_len, head_dim]
            
            attn = torch.matmul(q_t, k_window.transpose(-2, -1)) / self.scale  # [B, H, 1, window_len]
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention
            out_t = torch.matmul(attn, v_window)  # [B, H, 1, head_dim]
            out_t = out_t.transpose(1, 2).reshape(B, 1, D)  # [B, 1, D]
            
            outputs.append(out_t)
            contexts.append(out_t.clone())  # Context = attention output
        
        output = torch.cat(outputs, dim=1)  # [B, T, D]
        context = torch.cat(contexts, dim=1)  # [B, T, D]
        
        output = self.out_proj(output)
        
        return output, context


# =============================================================================
# Context-Conditional Baseline
# =============================================================================

class ContextConditionalBaseline(BaseModel):
    """
    Context-Conditional baseline using FiLM modulation.
    
    Architecture:
        1. Embed observations and actions
        2. Causal sliding window attention (W=20) extracts context
        3. FiLM layers modulate processing based on context
        4. Prediction heads for actions and types
        5. Post-hoc switch detection via context change magnitude
    
    This tests whether rich context representations (via attention + FiLM)
    can substitute for explicit belief tracking. The hypothesis is that
    FiLM conditioning on windowed context provides sufficient information
    for partner adaptation without discrete belief states.
    
    Expected characteristics:
        - Good action prediction (rich context)
        - Potentially delayed switch detection (window smoothing)
        - Context changes continuously (not discrete)
    
    References:
        Perez et al. (2018). FiLM: Visual Reasoning with a General
        Conditioning Layer. AAAI 2018.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        window_size: int = 20,
        num_heads: int = 4,
        num_film_layers: int = 2,
        context_threshold: float = 0.3,
    ):
        """
        Args:
            config: Model configuration
            window_size: Attention window size (W)
            num_heads: Number of attention heads
            num_film_layers: Number of FiLM-modulated layers
            context_threshold: Threshold for context-change switch detection
        """
        super().__init__(config)
        
        self.window_size = window_size
        self.context_threshold = context_threshold
        
        input_dim = config.obs_dim + config.action_dim
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Sliding window attention for context extraction
        self.context_attention = CausalWindowAttention(
            hidden_dim=config.hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
        )
        
        # FiLM-modulated processing layers
        self.film_layers = nn.ModuleList()
        self.processing_layers = nn.ModuleList()
        
        for _ in range(num_film_layers):
            self.processing_layers.append(nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
            ))
            self.film_layers.append(FiLMLayer(
                feature_dim=config.hidden_dim,
                context_dim=config.hidden_dim,
            ))
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(num_film_layers)
        ])
        
        # Prediction heads
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        self.type_head = nn.Linear(config.hidden_dim, config.num_types)
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Forward pass with FiLM conditioning.
        
        Args:
            observations: [B, T, obs_dim] or [B, obs_dim]
            actions: [B, T] action indices
        
        Returns:
            ModelOutput with predictions and context changes
        """
        B, T = actions.shape
        device = actions.device
        
        # Handle observations
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Embed inputs
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = self.input_embed(torch.cat([obs, act_oh], dim=-1))
        
        # Extract context via windowed attention
        attn_out, context = self.context_attention(x)
        x = x + attn_out  # Residual
        
        # Store context for switch detection
        context_vectors = [context.clone()]
        
        # FiLM-modulated processing
        for i, (process, film, norm) in enumerate(
            zip(self.processing_layers, self.film_layers, self.layer_norms)
        ):
            residual = x
            x = process(x)
            x = film(x, context)  # FiLM modulation
            x = norm(x + residual)
            
            # Update context based on processed features
            if i < len(self.film_layers) - 1:
                _, context = self.context_attention(x)
                context_vectors.append(context.clone())
        
        # Predictions
        action_logits = self.action_head(x)
        type_logits = self.type_head(x)
        
        # Compute context changes (average across layers)
        context_changes = torch.zeros(B, T, device=device)
        for ctx in context_vectors:
            if T > 1:
                ctx_diff = (ctx[:, 1:] - ctx[:, :-1]).abs().mean(dim=-1)
                temp = torch.zeros(B, T, device=device)
                temp[:, 1:] = ctx_diff
                context_changes = context_changes + temp
        context_changes = context_changes / len(context_vectors)
        
        # Switch detection via context change magnitude
        # Threshold or normalize
        switch_probs = context_changes / (context_changes.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=context_changes,
            context=context_vectors[-1],
        )
    
    def get_switch_signal(self, outputs: ModelOutput) -> torch.Tensor:
        """Context changes as switch signal."""
        return outputs.belief_changes / (outputs.belief_changes.max() + 1e-8)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Simpler Context-Conditional Variants
# =============================================================================

class ContextConditionalGRU(BaseModel):
    """
    Simpler context-conditional baseline using GRU + FiLM.
    
    Uses GRU for sequential processing with FiLM conditioning
    from a sliding window summary (mean pooling, no attention).
    """
    
    def __init__(self, config: ModelConfig, window_size: int = 10):
        super().__init__(config)
        
        self.window_size = window_size
        input_dim = config.obs_dim + config.action_dim
        
        self.input_embed = nn.Linear(input_dim, config.hidden_dim)
        
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        
        self.film = FiLMLayer(config.hidden_dim, config.hidden_dim)
        
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.type_head = nn.Linear(config.hidden_dim, config.num_types)
    
    def forward(self, observations, actions, types=None, images=None):
        B, T = actions.shape
        device = actions.device
        
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = self.input_embed(torch.cat([obs, act_oh], dim=-1))
        
        # GRU processing
        gru_out, _ = self.gru(x)
        
        # Sliding window context (mean pool)
        context = torch.zeros_like(gru_out)
        for t in range(T):
            start = max(0, t - self.window_size + 1)
            context[:, t] = gru_out[:, start:t+1].mean(dim=1)
        
        # FiLM modulation
        features = self.film(gru_out, context)
        
        action_logits = self.action_head(features)
        type_logits = self.type_head(features)
        
        # Context changes
        belief_changes = torch.zeros(B, T, device=device)
        if T > 1:
            belief_changes[:, 1:] = (context[:, 1:] - context[:, :-1]).abs().mean(dim=-1)
        
        switch_probs = belief_changes / (belief_changes.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
        )
