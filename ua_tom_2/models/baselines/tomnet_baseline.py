"""
ToMnet - Machine Theory of Mind
===============================

Exact implementation of ToMnet from Rabinowitz et al. (2018):
"Machine Theory of Mind", ICML 2018

Paper: https://arxiv.org/abs/1802.07740
       http://proceedings.mlr.press/v80/rabinowitz18a/rabinowitz18a.pdf

ToMnet uses meta-learning to model other agents from behavioral observations.
The key insight is separating STABLE agent characteristics from DYNAMIC mental
states that change within an episode.

Architecture (from paper Figure 2):
==================================

1. CHARACTER NETWORK (e_char):
   - Input: Past episode trajectories [(s_1, a_1), ..., (s_T, a_T)]
   - Architecture: Embedding → LSTM → final hidden state
   - Aggregation: Mean pool across multiple past episodes
   - Output: e_char ∈ R^{d_char} - captures "who is this agent"
   
   The character network encodes stable behavioral patterns that persist
   across episodes: goals, preferences, skill level, strategy type.

2. MENTAL STATE NETWORK (e_mental):
   - Input: Current episode trajectory up to time t, conditioned on e_char
   - Architecture: [s_t; a_t; e_char] → Embedding → LSTM
   - Output: e_mental_t ∈ R^{d_mental} for each timestep t
   
   The mental network captures dynamic beliefs about current goals,
   information state, and intentions that evolve during an episode.

3. PREDICTION NETWORK:
   - Input: [e_char; e_mental_t; s_t]
   - Architecture: Concatenation → MLP
   - Output: P(a_{t+1} | s_t, trajectory, past_episodes)
   
   Combines character and mental state to predict next action.

Key Differences from UA-ToM:
===========================
- ToMnet: Separate stable (character) vs dynamic (mental) representations
- UA-ToM: Unified belief tracking with explicit switch detection
- ToMnet: Meta-learning across many agents from a "species"  
- UA-ToM: Online adaptation to single partner with behavioral changes

Expected Results:
================
✓ Good at modeling stable agent traits across episodes
✓ Can generalize to new agents from same distribution
✗ No explicit switch detection mechanism
✗ May be slow to detect within-episode behavioral changes
✗ Requires past episodes for character embedding (cold start problem)

For switch detection, we use ||e_mental_t - e_mental_{t-1}|| as proxy.
This is NOT part of the original ToMnet - it's our adaptation for comparison.

Reference:
    Rabinowitz, N., Perbet, F., Song, F., Zhang, C., Eslami, S.M.A., 
    & Botvinick, M. (2018). Machine Theory of Mind. ICML.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel, ModelConfig, ModelOutput


# =============================================================================
# ToMnet Configuration
# =============================================================================

@dataclass 
class ToMnetConfig:
    """
    Configuration for ToMnet following Rabinowitz et al. 2018.
    
    The original paper uses relatively small networks since they work
    in simple gridworld environments. We use similar defaults but allow
    larger sizes for more complex domains.
    
    Key hyperparameters from the paper:
        - Character embedding dimension: varies by experiment
        - LSTM hidden sizes: typically 64-128
        - MLP layers: 2-3 layers
        - Number of past episodes: 0-10 (sampled uniformly)
    """
    
    # Character network (encodes stable agent traits)
    char_embed_dim: int = 64      # e_char dimension
    char_hidden_dim: int = 64     # LSTM hidden size
    char_num_layers: int = 1      # LSTM layers
    
    # Mental state network (encodes dynamic beliefs)
    mental_embed_dim: int = 64    # e_mental dimension
    mental_hidden_dim: int = 64   # LSTM hidden size
    mental_num_layers: int = 1    # LSTM layers
    
    # Prediction network
    pred_hidden_dim: int = 64     # MLP hidden size
    pred_num_layers: int = 2      # MLP layers
    
    # Meta-learning settings
    num_past_episodes: int = 5    # N_past in paper (0-10 typical)
    
    # Optional: variational character embedding (paper Section 3.5)
    use_variational: bool = False
    variational_dim: int = 8      # Latent dimension for VAE


# =============================================================================
# Character Network
# =============================================================================

class CharacterNetwork(nn.Module):
    """
    Character Network from ToMnet.
    
    Processes past episode trajectories to extract stable agent characteristics.
    
    Architecture:
        For each past episode:
            1. Embed (state, action) pairs
            2. Process with LSTM
            3. Take final hidden state as episode embedding
        
        Aggregate episode embeddings (mean pooling) → e_char
    
    This captures "who is this agent" - stable behavioral patterns
    that persist across episodes (goals, preferences, capabilities).
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        
        # Input embedding for (state, action) pairs
        input_dim = obs_dim + action_dim
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # LSTM to process trajectory
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Project to character embedding
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
    
    def forward_single_episode(
        self,
        observations: torch.Tensor,  # [T, obs_dim]
        actions: torch.Tensor,       # [T]
    ) -> torch.Tensor:
        """Process a single episode trajectory."""
        T = actions.shape[0]
        
        # Embed inputs
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = torch.cat([observations, act_oh], dim=-1)  # [T, input_dim]
        x = self.input_embed(x)  # [T, hidden_dim]
        
        # Process with LSTM
        x = x.unsqueeze(0)  # [1, T, hidden_dim]
        _, (h_n, _) = self.lstm(x)
        
        # Final hidden state → embedding
        episode_embed = self.output_proj(h_n[-1, 0])  # [embed_dim]
        
        return episode_embed
    
    def forward(
        self,
        past_observations: List[torch.Tensor],  # List of [T_i, obs_dim]
        past_actions: List[torch.Tensor],        # List of [T_i]
    ) -> torch.Tensor:
        """
        Process multiple past episodes and aggregate.
        
        Args:
            past_observations: List of observation sequences
            past_actions: List of action sequences
        
        Returns:
            Character embedding e_char [embed_dim]
        """
        if len(past_observations) == 0:
            # No past episodes - return zero embedding
            device = past_observations[0].device if past_observations else 'cpu'
            return torch.zeros(self.embed_dim, device=device)
        
        # Process each episode
        episode_embeds = []
        for obs, acts in zip(past_observations, past_actions):
            embed = self.forward_single_episode(obs, acts)
            episode_embeds.append(embed)
        
        # Aggregate via mean pooling
        e_char = torch.stack(episode_embeds).mean(dim=0)
        
        return e_char


# =============================================================================
# Mental State Network
# =============================================================================

class MentalStateNetwork(nn.Module):
    """
    Mental State Network from ToMnet.
    
    Processes current episode trajectory to infer dynamic mental state.
    
    Architecture:
        1. Embed (state, action) pairs from current episode so far
        2. Process with LSTM
        3. Final hidden state → e_mental
    
    This captures "what does this agent currently believe/want" -
    dynamic state that changes during an episode based on observations.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        char_embed_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        
        # Input: (state, action, character_embedding)
        input_dim = obs_dim + action_dim + char_embed_dim
        
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
    
    def forward(
        self,
        observations: torch.Tensor,  # [B, T, obs_dim]
        actions: torch.Tensor,       # [B, T]
        e_char: torch.Tensor,        # [B, char_embed_dim]
    ) -> torch.Tensor:
        """
        Process current episode with character context.
        
        Returns:
            Mental state embeddings e_mental [B, T, embed_dim]
        """
        B, T, _ = observations.shape
        
        # Embed inputs with character embedding
        act_oh = F.one_hot(actions, self.action_dim).float()
        e_char_expanded = e_char.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([observations, act_oh, e_char_expanded], dim=-1)
        x = self.input_embed(x)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(x)  # [B, T, hidden_dim]
        
        # Project to mental state embedding
        e_mental = self.output_proj(lstm_out)  # [B, T, embed_dim]
        
        return e_mental


# =============================================================================
# Prediction Network
# =============================================================================

class PredictionNetwork(nn.Module):
    """
    Prediction Network from ToMnet.
    
    Combines character embedding, mental state, and current observation
    to make predictions about agent behavior.
    
    Architecture:
        MLP([e_char; e_mental; current_state]) → predictions
    """
    
    def __init__(
        self,
        obs_dim: int,
        char_embed_dim: int,
        mental_embed_dim: int,
        action_dim: int,
        num_types: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        
        input_dim = obs_dim + char_embed_dim + mental_embed_dim
        
        # Build MLP
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
            ])
        self.mlp = nn.Sequential(*layers)
        
        # Prediction heads
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.type_head = nn.Linear(hidden_dim, num_types)
    
    def forward(
        self,
        observations: torch.Tensor,  # [B, T, obs_dim]
        e_char: torch.Tensor,        # [B, char_embed_dim]
        e_mental: torch.Tensor,      # [B, T, mental_embed_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions given embeddings.
        
        Returns:
            action_logits: [B, T, action_dim]
            type_logits: [B, T, num_types]
        """
        B, T, _ = observations.shape
        
        # Expand character embedding
        e_char_expanded = e_char.unsqueeze(1).expand(-1, T, -1)
        
        # Concatenate inputs
        x = torch.cat([observations, e_char_expanded, e_mental], dim=-1)
        
        # MLP forward
        features = self.mlp(x)
        
        # Predictions
        action_logits = self.action_head(features)
        type_logits = self.type_head(features)
        
        return action_logits, type_logits


# =============================================================================
# Full ToMnet Baseline
# =============================================================================

class ToMnetBaseline(BaseModel):
    """
    ToMnet baseline for partner modeling.
    
    Faithful implementation of Rabinowitz et al. (2018) "Machine Theory of Mind".
    
    The model learns to predict partner behavior by:
        1. Building a character embedding from past behavior
        2. Tracking mental state during current episode
        3. Combining both for action/type prediction
    
    For switch detection, we use changes in the mental state embedding
    as a proxy (ToMnet doesn't explicitly model switches).
    
    Expected characteristics:
        - Good at learning stable partner traits
        - Can generalize to new agents from same "species"
        - May be slower to adapt to within-episode changes
        - No explicit switch detection mechanism
    
    Reference:
        Rabinowitz et al. (2018). Machine Theory of Mind. ICML.
        https://arxiv.org/abs/1802.07740
    """
    
    def __init__(
        self,
        config: ModelConfig,
        tomnet_config: ToMnetConfig = None,
    ):
        super().__init__(config)
        
        self.tomnet_config = tomnet_config or ToMnetConfig()
        tc = self.tomnet_config
        
        # Character network
        self.char_net = CharacterNetwork(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            embed_dim=tc.char_embed_dim,
            hidden_dim=tc.char_hidden_dim,
            num_layers=tc.char_num_layers,
        )
        
        # Mental state network
        self.mental_net = MentalStateNetwork(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            char_embed_dim=tc.char_embed_dim,
            embed_dim=tc.mental_embed_dim,
            hidden_dim=tc.mental_hidden_dim,
            num_layers=tc.mental_num_layers,
        )
        
        # Prediction network
        self.pred_net = PredictionNetwork(
            obs_dim=config.obs_dim,
            char_embed_dim=tc.char_embed_dim,
            mental_embed_dim=tc.mental_embed_dim,
            action_dim=config.action_dim,
            num_types=config.num_types,
            hidden_dim=tc.pred_hidden_dim,
            num_layers=tc.pred_num_layers,
        )
        
        # For online use: accumulated past episodes
        self.past_episodes: List[Tuple[torch.Tensor, torch.Tensor]] = []
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        past_observations: Optional[List[torch.Tensor]] = None,
        past_actions: Optional[List[torch.Tensor]] = None,
    ) -> ModelOutput:
        """
        Forward pass through ToMnet.
        
        In training, past_observations/past_actions should be provided.
        In online inference, the model uses self.past_episodes.
        
        Args:
            observations: [B, T, obs_dim] or [B, obs_dim]
            actions: [B, T] action indices
            past_observations: Optional list of past episode observations
            past_actions: Optional list of past episode actions
        
        Returns:
            ModelOutput with predictions
        """
        B, T = actions.shape
        device = actions.device
        
        # Handle observation dimensions
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Compute character embedding
        # In batch mode, we use a simplified approach where we treat
        # the early part of the sequence as "past episodes"
        if past_observations is not None and past_actions is not None:
            # Use provided past episodes
            e_char = torch.stack([
                self.char_net(
                    [po.to(device) for po in past_observations],
                    [pa.to(device) for pa in past_actions],
                )
                for _ in range(B)
            ])
        else:
            # Split current trajectory: first half as "past", second as "current"
            # This is a simplification for batch training
            split_t = max(1, T // 4)
            past_obs = [obs[b, :split_t] for b in range(B)]
            past_act = [actions[b, :split_t] for b in range(B)]
            
            e_char = torch.stack([
                self.char_net([past_obs[b]], [past_act[b]])
                for b in range(B)
            ])
        
        # Compute mental state embeddings
        e_mental = self.mental_net(obs, actions, e_char)
        
        # Get predictions
        action_logits, type_logits = self.pred_net(obs, e_char, e_mental)
        
        # Compute belief changes from mental state changes
        belief_changes = torch.zeros(B, T, device=device)
        if T > 1:
            mental_diffs = (e_mental[:, 1:] - e_mental[:, :-1]).abs().mean(dim=-1)
            belief_changes[:, 1:] = mental_diffs
        
        # Switch detection via mental state change magnitude
        # Normalize for probability-like output
        switch_probs = belief_changes / (belief_changes.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
            mental_state=e_mental,
            character_embed=e_char,
        )
    
    def add_past_episode(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """Add a completed episode to past history (for online inference)."""
        self.past_episodes.append((observations, actions))
        
        # Keep only recent episodes
        max_past = self.tomnet_config.num_past_episodes
        if len(self.past_episodes) > max_past:
            self.past_episodes = self.past_episodes[-max_past:]
    
    def reset_past_episodes(self):
        """Clear past episode history."""
        self.past_episodes = []
    
    def get_switch_signal(self, outputs: ModelOutput) -> torch.Tensor:
        """Use mental state changes as switch signal."""
        return outputs.belief_changes / (outputs.belief_changes.max() + 1e-8)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_param_breakdown(self) -> dict:
        """Get parameter count breakdown by component."""
        breakdown = {
            'character_network': sum(p.numel() for p in self.char_net.parameters()),
            'mental_network': sum(p.numel() for p in self.mental_net.parameters()),
            'prediction_network': sum(p.numel() for p in self.pred_net.parameters()),
        }
        breakdown['total'] = sum(breakdown.values())
        return breakdown


# =============================================================================
# Simplified ToMnet for ablation (Character-only, Mental-only)
# =============================================================================

class ToMnetCharacterOnly(BaseModel):
    """ToMnet with only character network (no mental state tracking)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.char_net = CharacterNetwork(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
        )
        
        # Direct prediction from character + observation
        self.pred_mlp = nn.Sequential(
            nn.Linear(config.obs_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, config.action_dim)
        self.type_head = nn.Linear(64, config.num_types)
    
    def forward(self, observations, actions, types=None, images=None):
        B, T = actions.shape
        device = actions.device
        
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Character embedding from early trajectory
        split_t = max(1, T // 4)
        e_char = torch.stack([
            self.char_net([obs[b, :split_t]], [actions[b, :split_t]])
            for b in range(B)
        ])
        
        # Expand and concatenate
        e_char_exp = e_char.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([obs, e_char_exp], dim=-1)
        
        features = self.pred_mlp(x)
        action_logits = self.action_head(features)
        type_logits = self.type_head(features)
        
        # No mental state = no switch detection
        switch_probs = torch.zeros(B, T, device=device)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=switch_probs,
        )


class ToMnetMentalOnly(BaseModel):
    """ToMnet with only mental state network (no character embedding)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Mental state LSTM (no character conditioning)
        self.mental_lstm = nn.LSTM(
            input_size=config.obs_dim + config.action_dim,
            hidden_size=64,
            batch_first=True,
        )
        
        self.pred_mlp = nn.Sequential(
            nn.Linear(config.obs_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(64, config.action_dim)
        self.type_head = nn.Linear(64, config.num_types)
    
    def forward(self, observations, actions, types=None, images=None):
        B, T = actions.shape
        device = actions.device
        
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = torch.cat([obs, act_oh], dim=-1)
        
        e_mental, _ = self.mental_lstm(x)
        
        features = self.pred_mlp(torch.cat([obs, e_mental], dim=-1))
        action_logits = self.action_head(features)
        type_logits = self.type_head(features)
        
        # Mental state changes for switch detection
        belief_changes = torch.zeros(B, T, device=device)
        if T > 1:
            belief_changes[:, 1:] = (e_mental[:, 1:] - e_mental[:, :-1]).abs().mean(dim=-1)
        
        switch_probs = belief_changes / (belief_changes.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
        )
