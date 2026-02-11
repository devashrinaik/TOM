"""
Simple temporal baselines: GRU and Transformer.

These are the standard sequence models without any explicit
belief tracking or Theory of Mind components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel, ModelConfig, ModelOutput


class GRUBaseline(BaseModel):
    """
    GRU baseline for partner modeling.
    
    Simple recurrent architecture without explicit belief tracking.
    Uses learned switch detection head.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        input_dim = config.obs_dim + config.action_dim
        
        self.gru = nn.GRU(
            input_dim,
            config.hidden_dim,
            batch_first=True,
        )
        
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.type_head = nn.Linear(config.hidden_dim, config.num_types)
        self.switch_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, observations, actions, types=None, images=None):
        B, T = actions.shape
        device = actions.device
        
        # Use images if provided, otherwise observations
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Embed
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = torch.cat([obs, act_oh], dim=-1)
        
        # GRU forward
        h, _ = self.gru(x)
        
        # Predictions
        action_logits = self.action_head(h)
        type_logits = self.type_head(h)
        switch_probs = self.switch_head(h).squeeze(-1)
        
        # Belief changes from type distribution
        type_probs = F.softmax(type_logits, dim=-1)
        belief_changes = torch.zeros(B, T, device=device)
        belief_changes[:, 1:] = (type_probs[:, 1:] - type_probs[:, :-1]).abs().sum(dim=-1)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
        )


class TransformerBaseline(BaseModel):
    """
    Transformer baseline for partner modeling.
    
    Standard causal transformer without explicit belief tracking.
    Uses learned switch detection head.
    """
    
    def __init__(self, config: ModelConfig, num_layers: int = 2):
        super().__init__(config)
        
        input_dim = config.obs_dim + config.action_dim
        
        self.embedding = nn.Linear(input_dim, config.hidden_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 500, config.hidden_dim) * 0.02
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.type_head = nn.Linear(config.hidden_dim, config.num_types)
        self.switch_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, observations, actions, types=None, images=None):
        B, T = actions.shape
        device = actions.device
        
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Embed
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = torch.cat([obs, act_oh], dim=-1)
        x = self.embedding(x) + self.pos_encoding[:, :T]
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        
        # Transformer forward
        h = self.transformer(x, mask=causal_mask)
        
        # Predictions
        action_logits = self.action_head(h)
        type_logits = self.type_head(h)
        switch_probs = self.switch_head(h).squeeze(-1)
        
        # Belief changes
        type_probs = F.softmax(type_logits, dim=-1)
        belief_changes = torch.zeros(B, T, device=device)
        belief_changes[:, 1:] = (type_probs[:, 1:] - type_probs[:, :-1]).abs().sum(dim=-1)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
        )
