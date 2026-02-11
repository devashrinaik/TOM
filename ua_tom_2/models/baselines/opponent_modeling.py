"""
Opponent Modeling Baselines
===========================

Implements proper opponent modeling approaches from the multi-agent RL literature:

1. DRON (Deep Reinforcement Opponent Network) - He et al. (2016)
   "Opponent Modeling in Deep Reinforcement Learning", ICML 2016
   https://arxiv.org/abs/1609.05559

2. Latent Opponent Embedding (VAE-based)
   Inspired by work on learning latent agent representations.

These baselines model opponent behavior through learned embeddings,
either via direct observation encoding (DRON) or variational inference
(VAE). Switch detection uses embedding drift/changes.

Reference:
    He et al. (2016). Opponent Modeling in Deep Reinforcement Learning.
    ICML 2016. https://arxiv.org/abs/1609.05559
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel, ModelConfig, ModelOutput


# =============================================================================
# DRON: Deep Reinforcement Opponent Network (He et al. 2016)
# =============================================================================

class DRONBaseline(BaseModel):
    """
    Deep Reinforcement Opponent Network following He et al. (2016).
    
    Architecture from the paper:
        1. Observation encoder processes environment state
        2. Opponent encoder processes opponent observation (their actions)
        3. Combined features go to Q-network / policy
    
    Key insight: Rather than explicitly predicting opponent actions,
    encode observations of opponents into the value function directly.
    This lets the model learn implicit opponent features.
    
    For our task (predicting partner actions), we adapt this to:
        1. Observation encoder for environment state
        2. Partner action encoder (LSTM over their action history)
        3. Combined features predict their next action
    
    Switch detection: Use partner embedding changes.
    
    Reference:
        He et al. (2016). Opponent Modeling in Deep Reinforcement Learning.
        ICML 2016. https://arxiv.org/abs/1609.05559
    """
    
    def __init__(
        self,
        config: ModelConfig,
        opponent_embed_dim: int = 64,
        use_mixture_of_experts: bool = False,
        num_experts: int = 4,
    ):
        """
        Args:
            config: Model configuration
            opponent_embed_dim: Dimension of opponent embedding
            use_mixture_of_experts: Use MoE to discover opponent types
            num_experts: Number of expert networks (if using MoE)
        """
        super().__init__(config)
        
        self.opponent_embed_dim = opponent_embed_dim
        self.use_moe = use_mixture_of_experts
        self.num_experts = num_experts
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        # Opponent/Partner action encoder (LSTM)
        self.opponent_encoder = nn.LSTM(
            input_size=config.action_dim,
            hidden_size=opponent_embed_dim,
            num_layers=2,
            batch_first=True,
        )
        
        # Combine observation and opponent features
        combined_dim = config.hidden_dim + opponent_embed_dim
        
        if use_mixture_of_experts:
            # Mixture of Experts for discovering opponent types
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(combined_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                )
                for _ in range(num_experts)
            ])
            # Gating network
            self.gate = nn.Sequential(
                nn.Linear(opponent_embed_dim, num_experts),
                nn.Softmax(dim=-1),
            )
        else:
            # Standard MLP
            self.combiner = nn.Sequential(
                nn.Linear(combined_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
            )
        
        # Prediction heads
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.type_head = nn.Linear(opponent_embed_dim, config.num_types)
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Forward pass with opponent modeling.
        
        Args:
            observations: [B, T, obs_dim] or [B, obs_dim]
            actions: [B, T] partner action indices (opponent actions)
        
        Returns:
            ModelOutput with predictions
        """
        B, T = actions.shape
        device = actions.device
        
        # Handle observations
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Encode observations
        obs_features = self.obs_encoder(obs)  # [B, T, hidden_dim]
        
        # Encode opponent actions
        act_oh = F.one_hot(actions, self.action_dim).float()
        opponent_embed, _ = self.opponent_encoder(act_oh)  # [B, T, opponent_embed_dim]
        
        # Combine features
        combined = torch.cat([obs_features, opponent_embed], dim=-1)
        
        if self.use_moe:
            # Mixture of Experts
            gate_weights = self.gate(opponent_embed)  # [B, T, num_experts]
            
            # Compute expert outputs
            expert_outputs = torch.stack([
                expert(combined) for expert in self.experts
            ], dim=-1)  # [B, T, hidden_dim, num_experts]
            
            # Weighted combination
            features = (expert_outputs * gate_weights.unsqueeze(-2)).sum(dim=-1)
        else:
            features = self.combiner(combined)
        
        # Predictions
        action_logits = self.action_head(features)
        type_logits = self.type_head(opponent_embed)
        
        # Belief changes from opponent embedding changes
        belief_changes = torch.zeros(B, T, device=device)
        if T > 1:
            embed_diff = (opponent_embed[:, 1:] - opponent_embed[:, :-1]).abs().mean(dim=-1)
            belief_changes[:, 1:] = embed_diff
        
        # Switch detection
        switch_probs = belief_changes / (belief_changes.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
            opponent_embed=opponent_embed,
        )
    
    def get_switch_signal(self, outputs: ModelOutput) -> torch.Tensor:
        """Opponent embedding changes as switch signal."""
        return outputs.belief_changes / (outputs.belief_changes.max() + 1e-8)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Latent Opponent VAE
# =============================================================================

class LatentOpponentVAE(BaseModel):
    """
    VAE-based latent opponent modeling.
    
    Uses variational inference to learn a compact latent representation
    of the opponent/partner from their action trajectory.
    
    Architecture:
        Encoder: BiGRU over action history → μ, log σ²
        Latent: z ~ N(μ, σ²) via reparameterization
        Policy: MLP([obs; z]) → action prediction
    
    The VAE encourages learning a structured latent space where:
        - Different partner types occupy different regions
        - Continuous interpolation represents behavioral blending
    
    Switch detection: Large jumps in latent space indicate partner changes.
    
    KL regularization encourages a smooth, structured latent space,
    but may blur distinctions between similar partner types.
    
    Expected characteristics:
        - Good action prediction (rich latent representation)
        - Continuous latent changes (may miss sharp switches)
        - Interpretable latent space (can visualize partner types)
    """
    
    def __init__(
        self,
        config: ModelConfig,
        latent_dim: int = 32,
        kl_weight: float = 0.01,
        window_size: int = 10,
    ):
        """
        Args:
            config: Model configuration
            latent_dim: Dimension of latent space z
            kl_weight: Weight for KL divergence loss (β in β-VAE)
            window_size: Window of past actions for encoding
        """
        super().__init__(config)
        
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.window_size = window_size
        
        # Encoder: BiGRU over partner actions
        self.encoder_gru = nn.GRU(
            input_size=config.action_dim,
            hidden_size=config.hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        
        # VAE parameters (μ and log σ²)
        self.fc_mu = nn.Linear(config.hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, latent_dim)
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(config.obs_dim + latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.type_head = nn.Linear(latent_dim, config.num_types)
        
        # Track KL loss for training
        self.kl_loss = 0.0
    
    def encode(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode action trajectory to latent distribution.
        
        Args:
            actions: [B, T] action indices
        
        Returns:
            z: [B, T, latent_dim] sampled latent
            mu: [B, T, latent_dim] mean
            logvar: [B, T, latent_dim] log variance
        """
        B, T = actions.shape
        device = actions.device
        
        # One-hot encode actions
        act_oh = F.one_hot(actions, self.action_dim).float()
        
        # Process with BiGRU
        enc_out, _ = self.encoder_gru(act_oh)  # [B, T, hidden_dim]
        
        # Compute windowed encoding for each timestep
        # This allows the latent to be computed from recent history only
        mu_list = []
        logvar_list = []
        
        for t in range(T):
            start = max(0, t - self.window_size + 1)
            window = enc_out[:, start:t+1]  # [B, window_len, hidden_dim]
            window_pooled = window.mean(dim=1)  # [B, hidden_dim]
            
            mu_t = self.fc_mu(window_pooled)
            logvar_t = self.fc_logvar(window_pooled)
            
            mu_list.append(mu_t)
            logvar_list.append(logvar_t)
        
        mu = torch.stack(mu_list, dim=1)  # [B, T, latent_dim]
        logvar = torch.stack(logvar_list, dim=1)  # [B, T, latent_dim]
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # Use mean at test time
        
        return z, mu, logvar
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Forward pass with VAE encoding.
        
        Args:
            observations: [B, T, obs_dim] or [B, obs_dim]
            actions: [B, T] partner action indices
        
        Returns:
            ModelOutput with predictions
        """
        B, T = actions.shape
        device = actions.device
        
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Encode partner actions to latent space
        z, mu, logvar = self.encode(actions)
        
        # Compute KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        self.kl_loss = self.kl_weight * kl_loss
        
        # Policy forward pass
        policy_input = torch.cat([obs, z], dim=-1)
        features = self.policy(policy_input)
        
        # Predictions
        action_logits = self.action_head(features)
        type_logits = self.type_head(z)
        
        # Belief changes from latent space drift
        belief_changes = torch.zeros(B, T, device=device)
        if T > 1:
            latent_diff = (z[:, 1:] - z[:, :-1]).norm(dim=-1)
            belief_changes[:, 1:] = latent_diff
        
        # Switch detection via latent drift
        switch_probs = belief_changes / (belief_changes.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
            latent=z,
            latent_mu=mu,
            latent_logvar=logvar,
            kl_loss=self.kl_loss,
        )
    
    def get_loss(self, outputs: ModelOutput, targets: Dict) -> torch.Tensor:
        """
        Compute total loss including KL term.
        
        Should be called in training loop to add VAE regularization.
        """
        return outputs.kl_loss
    
    def get_switch_signal(self, outputs: ModelOutput) -> torch.Tensor:
        """Latent drift as switch signal."""
        return outputs.belief_changes / (outputs.belief_changes.max() + 1e-8)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Online Implicit Agent Modeling (OIAM)
# =============================================================================

class OIAMBaseline(BaseModel):
    """
    Online Implicit Agent Modeling baseline.
    
    Inspired by Bard et al. (2013) and implicit opponent modeling approaches.
    
    Rather than explicit opponent representation, this model:
        1. Maintains a recurrent state that implicitly captures opponent info
        2. Updates this state online as new observations arrive
        3. Doesn't require separate opponent encoder
    
    This is essentially a strong GRU baseline that's optimized for
    implicit opponent modeling through its hidden state.
    
    Switch detection: Hidden state change magnitude.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(config)
        
        input_dim = config.obs_dim + config.action_dim
        
        # Deep GRU for implicit modeling
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=config.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        
        # Prediction heads with residual connections
        self.action_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim + config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        
        self.type_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
        )
        self.type_head = nn.Linear(config.hidden_dim // 2, config.num_types)
    
    def forward(self, observations, actions, types=None, images=None):
        B, T = actions.shape
        device = actions.device
        
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = torch.cat([obs, act_oh], dim=-1)
        
        # GRU processing
        hidden_states, _ = self.gru(x)
        
        # Action prediction with obs residual
        action_features = self.action_mlp(torch.cat([hidden_states, obs], dim=-1))
        action_logits = self.action_head(action_features)
        
        # Type prediction
        type_features = self.type_mlp(hidden_states)
        type_logits = self.type_head(type_features)
        
        # Hidden state changes for switch detection
        belief_changes = torch.zeros(B, T, device=device)
        if T > 1:
            hidden_diff = (hidden_states[:, 1:] - hidden_states[:, :-1]).abs().mean(dim=-1)
            belief_changes[:, 1:] = hidden_diff
        
        switch_probs = belief_changes / (belief_changes.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
        )
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
