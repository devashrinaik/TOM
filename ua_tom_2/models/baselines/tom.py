"""
Theory of Mind baselines: B-ToM and ToMnet.

These models have explicit belief tracking mechanisms but
use simpler architectures than UA-ToM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel, ModelConfig, ModelOutput


class BToM(BaseModel):
    """
    Bayesian Theory of Mind (B-ToM).
    
    Maintains explicit belief distribution over partner types
    with Bayesian updates based on action likelihoods.
    
    Key difference from UA-ToM:
    - No selective gating (updates beliefs every timestep)
    - No hierarchical prediction errors
    - Simpler hypothesis bank
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        input_dim = config.obs_dim + config.action_dim
        
        # Per-type action predictors (hypothesis bank)
        self.type_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.obs_dim + config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.action_dim),
            )
            for _ in range(config.num_types)
        ])
        
        # Temporal encoder
        self.temporal = nn.GRU(input_dim, config.hidden_dim, batch_first=True)
        
        # Type classifier
        self.type_head = nn.Linear(config.hidden_dim, config.num_types)
    
    def forward(self, observations, actions, types=None, images=None):
        B, T = actions.shape
        device = actions.device
        
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Initialize beliefs uniformly
        beliefs = torch.ones(B, self.num_types, device=device) / self.num_types
        h = None
        
        all_action_logits = []
        all_type_logits = []
        all_belief_changes = []
        
        for t in range(T):
            obs_t = obs[:, t]
            act_oh = F.one_hot(actions[:, t], self.action_dim).float()
            prev_beliefs = beliefs.clone()
            
            # Temporal update
            gru_input = torch.cat([obs_t, act_oh], dim=-1).unsqueeze(1)
            _, h = self.temporal(gru_input, h)
            h_t = h.squeeze(0)
            
            # Per-type predictions
            state = torch.cat([obs_t, h_t], dim=-1)
            per_type_logits = torch.stack([
                pred(state) for pred in self.type_predictors
            ], dim=1)  # [B, K, A]
            
            # Belief-weighted action prediction
            action_logits = (beliefs.unsqueeze(-1) * per_type_logits).sum(dim=1)
            all_action_logits.append(action_logits)
            
            # Type classification
            type_logits = self.type_head(h_t)
            all_type_logits.append(type_logits)
            
            # Bayesian belief update
            with torch.no_grad():
                action_probs = F.softmax(per_type_logits, dim=-1)
                obs_action = actions[:, t]
                
                # Likelihood of observed action under each type
                likelihoods = action_probs[
                    torch.arange(B, device=device).unsqueeze(1),
                    torch.arange(self.num_types, device=device).unsqueeze(0),
                    obs_action.unsqueeze(1).expand(-1, self.num_types),
                ].clamp(1e-8, 1.0)
                
                # Posterior = prior * likelihood
                new_beliefs = beliefs * likelihoods
                new_beliefs = new_beliefs / (new_beliefs.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Belief change
            belief_change = (new_beliefs - prev_beliefs).abs().sum(dim=-1)
            all_belief_changes.append(belief_change)
            
            # Update beliefs
            beliefs = new_beliefs
        
        all_belief_changes = torch.stack(all_belief_changes, dim=1)
        
        # Switch detected where belief change > threshold
        switch_probs = (all_belief_changes > 0.1).float()
        
        return ModelOutput(
            action_logits=torch.stack(all_action_logits, dim=1),
            type_logits=torch.stack(all_type_logits, dim=1),
            switch_probs=switch_probs,
            belief_changes=all_belief_changes,
        )


class ToMnet(BaseModel):
    """
    Theory of Mind Network (ToMnet).
    
    Based on Rabinowitz et al. 2018.
    Uses separate character and mental state networks.
    
    Key features:
    - Character network: Encodes past behavior into character embedding
    - Mental state network: Predicts current mental state
    - No explicit belief distribution
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        input_dim = config.obs_dim + config.action_dim
        
        # Character network (encodes trajectory into embedding)
        self.char_encoder = nn.GRU(
            input_dim,
            config.hidden_dim,
            batch_first=True,
        )
        self.char_embed = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        
        # Mental state network
        self.mental_encoder = nn.GRU(
            input_dim + config.hidden_dim // 2,
            config.hidden_dim,
            batch_first=True,
        )
        
        # Prediction heads
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
        
        # Embed actions
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = torch.cat([obs, act_oh], dim=-1)
        
        # Character encoding (from full trajectory)
        char_out, _ = self.char_encoder(x)
        char_embed = self.char_embed(char_out)  # [B, T, hidden//2]
        
        # Mental state encoding (conditioned on character)
        mental_input = torch.cat([x, char_embed], dim=-1)
        mental_out, _ = self.mental_encoder(mental_input)
        
        # Predictions
        action_logits = self.action_head(mental_out)
        type_logits = self.type_head(mental_out)
        switch_probs = self.switch_head(mental_out).squeeze(-1)
        
        # Belief changes from embeddings
        belief_changes = torch.zeros(B, T, device=device)
        belief_changes[:, 1:] = (char_embed[:, 1:] - char_embed[:, :-1]).abs().mean(dim=-1)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
        )
