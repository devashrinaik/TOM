"""
Extended Opponent Modeling Baselines
=====================================

Additional opponent/partner modeling approaches from the literature.

Implemented Methods:
-------------------

1. LIAM (Local Information Agent Modeling) - Papoudakis et al. 2021
   "Agent Modelling under Partial Observability for Deep Reinforcement Learning", NeurIPS 2021
   Paper: https://proceedings.neurips.cc/paper/2021/file/a03caec56cd82478bf197475b48c05f9-Paper.pdf
   
   Key idea: Encoder-decoder architecture that learns to reconstruct other agents'
   observations and actions from only the controlled agent's local trajectory.
   During inference, only the encoder is used to produce embeddings.

2. SOM (Self Other-Modeling) - Raileanu et al. 2018
   "Modeling Others using Oneself in Multi-Agent Reinforcement Learning", ICML 2018
   Paper: https://arxiv.org/abs/1802.09640
   
   Key idea: Use own policy to predict other agent's actions and infer their
   hidden goal via maximum likelihood optimization.

3. Context-Conditional Policy
   General approach used in meta-RL (e.g., PEARL, VariBAD)
   
   Key idea: Infer latent context from behavior history, condition policy on context.

4. MBOM (Model-Based Opponent Modeling) - Yu et al. 2022
   "Model-Based Opponent Modeling", NeurIPS 2022
   
   Key idea: Learn world model that includes opponent behavior, use for planning.

These complement the baselines in opponent_modeling.py (DRON, LatentOpponentVAE, OIAM).
"""

from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel, ModelConfig, ModelOutput


# =============================================================================
# LIAM: Local Information Agent Modeling (Papoudakis et al. 2021)
# =============================================================================

@dataclass
class LIAMConfig:
    """
    Configuration for LIAM.
    
    From Papoudakis et al. 2021:
    - embed_dim: Dimension of agent embedding
    - encoder_hidden: Hidden size for encoder GRU
    - decoder_hidden: Hidden size for decoder
    - reconstruction_weight: Weight for reconstruction loss
    """
    embed_dim: int = 64
    encoder_hidden: int = 128
    decoder_hidden: int = 128
    reconstruction_weight: float = 1.0
    use_obs_reconstruction: bool = True
    use_action_reconstruction: bool = True


class LIAMEncoder(nn.Module):
    """
    LIAM Encoder: Maps local trajectory to agent embedding.
    
    The encoder processes the controlled agent's local information:
    - Own observations
    - Own actions
    
    And produces an embedding that captures information about OTHER agents.
    This is the key insight: local trajectory contains implicit information
    about partners through their effects on the environment.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        
        # Input embedding
        input_dim = obs_dim + action_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Recurrent encoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        
        # Output to embedding
        self.embed_proj = nn.Linear(hidden_dim, embed_dim)
    
    def forward(
        self,
        observations: torch.Tensor,  # [B, T, obs_dim]
        actions: torch.Tensor,       # [B, T]
    ) -> torch.Tensor:
        """
        Encode local trajectory to embedding.
        
        Returns:
            embed: [B, T, embed_dim] - embedding at each timestep
        """
        B, T, _ = observations.shape
        
        # Create input
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = torch.cat([observations, act_oh], dim=-1)
        x = self.input_proj(x)
        
        # Process with GRU
        h, _ = self.gru(x)  # [B, T, hidden]
        
        # Project to embedding
        embed = self.embed_proj(h)  # [B, T, embed_dim]
        
        return embed


class LIAMDecoder(nn.Module):
    """
    LIAM Decoder: Reconstructs other agents' information from embedding.
    
    The decoder takes the embedding and tries to reconstruct:
    - Other agents' observations
    - Other agents' actions
    
    This reconstruction loss encourages the encoder to capture
    information about other agents in the embedding.
    """
    
    def __init__(
        self,
        embed_dim: int,
        other_obs_dim: int,
        other_action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        
        self.other_obs_dim = other_obs_dim
        self.other_action_dim = other_action_dim
        
        # Observation reconstruction head
        self.obs_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, other_obs_dim),
        )
        
        # Action reconstruction head
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, other_action_dim),
        )
    
    def forward(
        self,
        embed: torch.Tensor,  # [B, T, embed_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct other agent's information.
        
        Returns:
            obs_pred: [B, T, other_obs_dim]
            action_logits: [B, T, other_action_dim]
        """
        obs_pred = self.obs_decoder(embed)
        action_logits = self.action_decoder(embed)
        return obs_pred, action_logits


class LIAMBaseline(BaseModel):
    """
    Local Information Agent Modeling (LIAM) from Papoudakis et al. (2021).
    
    "Agent Modelling under Partial Observability for Deep Reinforcement Learning"
    NeurIPS 2021
    
    Architecture:
        Training:
            1. Encoder: local trajectory → embedding z
            2. Decoder: z → reconstruct other agents' obs/actions
            3. Policy: (obs, z) → action
        
        Inference:
            - Only encoder is used
            - Embedding captures learned representation of other agents
    
    Key insight: By training to reconstruct other agents' information from
    only local observations, the encoder learns to extract implicit information
    about partners from how they affect the environment.
    
    For our task, we adapt this to:
        - Encoder learns embedding from partner's observed behavior
        - Decoder reconstructs partner's future behavior (for training)
        - Policy conditions on embedding for prediction
    
    Reference:
        Papoudakis, G., Christianos, F., & Albrecht, S. V. (2021).
        Agent Modelling under Partial Observability for Deep Reinforcement Learning.
        NeurIPS 2021.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        liam_config: LIAMConfig = None,
    ):
        super().__init__(config)
        
        self.liam_config = liam_config or LIAMConfig()
        lc = self.liam_config
        
        # Encoder
        self.encoder = LIAMEncoder(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            embed_dim=lc.embed_dim,
            hidden_dim=lc.encoder_hidden,
        )
        
        # Decoder (for training)
        self.decoder = LIAMDecoder(
            embed_dim=lc.embed_dim,
            other_obs_dim=config.obs_dim,  # Reconstruct partner obs
            other_action_dim=config.action_dim,  # Reconstruct partner actions
            hidden_dim=lc.decoder_hidden,
        )
        
        # Policy network
        policy_input_dim = config.obs_dim + lc.embed_dim
        self.policy = nn.Sequential(
            nn.Linear(policy_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        # Output heads
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.type_head = nn.Linear(lc.embed_dim, config.num_types)
        
        # Store reconstruction loss for training
        self.reconstruction_loss = torch.tensor(0.0)
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        partner_observations: Optional[torch.Tensor] = None,
        partner_actions: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Forward pass through LIAM.
        
        Args:
            observations: [B, T, obs_dim] - own observations
            actions: [B, T] - partner actions we're predicting
            partner_observations: Optional partner obs for reconstruction loss
            partner_actions: Optional partner actions for reconstruction loss
        """
        B, T = actions.shape
        device = actions.device
        lc = self.liam_config
        
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Encode trajectory to embedding
        embed = self.encoder(obs, actions)  # [B, T, embed_dim]
        
        # Compute reconstruction loss if training data provided
        if self.training and partner_observations is not None:
            obs_pred, action_logits = self.decoder(embed)
            
            # Observation reconstruction loss (MSE)
            obs_loss = F.mse_loss(obs_pred, partner_observations)
            
            # Action reconstruction loss (cross-entropy)
            if partner_actions is not None:
                action_loss = F.cross_entropy(
                    action_logits.reshape(-1, self.action_dim),
                    partner_actions.reshape(-1),
                )
            else:
                action_loss = torch.tensor(0.0, device=device)
            
            self.reconstruction_loss = lc.reconstruction_weight * (obs_loss + action_loss)
        else:
            self.reconstruction_loss = torch.tensor(0.0, device=device)
        
        # Policy forward
        policy_input = torch.cat([obs, embed], dim=-1)
        features = self.policy(policy_input)
        
        # Predictions
        action_logits = self.action_head(features)
        type_logits = self.type_head(embed)
        
        # Embedding changes for switch detection
        belief_changes = torch.zeros(B, T, device=device)
        if T > 1:
            embed_diff = (embed[:, 1:] - embed[:, :-1]).norm(dim=-1)
            belief_changes[:, 1:] = embed_diff
        
        switch_probs = belief_changes / (belief_changes.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
            agent_embed=embed,
            reconstruction_loss=self.reconstruction_loss,
        )
    
    def get_reconstruction_loss(self) -> torch.Tensor:
        """Get reconstruction loss for training."""
        return self.reconstruction_loss
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_param_breakdown(self) -> dict:
        return {
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters()),
            'policy': sum(p.numel() for p in self.policy.parameters()),
            'heads': sum(p.numel() for p in self.action_head.parameters()) + 
                    sum(p.numel() for p in self.type_head.parameters()),
            'total': self.count_parameters(),
        }


# =============================================================================
# SOM: Self Other-Modeling (Raileanu et al. 2018)
# =============================================================================

@dataclass
class SOMConfig:
    """
    Configuration for Self Other-Modeling.
    
    From Raileanu et al. 2018:
    - goal_dim: Dimension of inferred goal/hidden state
    - policy_hidden: Hidden size for policy LSTM
    """
    goal_dim: int = 16
    policy_hidden: int = 128
    use_variational: bool = True


class SOMBaseline(BaseModel):
    """
    Self Other-Modeling (SOM) from Raileanu et al. (2018).
    
    "Modeling Others using Oneself in Multi-Agent Reinforcement Learning"
    ICML 2018
    Paper: https://arxiv.org/abs/1802.09640
    
    Key insight: Use your OWN policy to model what others would do,
    then infer their hidden goal by finding which goal best explains
    their observed actions via maximum likelihood.
    
    Architecture:
        1. Goal inference network: trajectory → goal distribution
        2. Policy LSTM: (obs, action, goal) → action distribution
    
    The "self-other modeling" means we use OUR policy network to
    simulate what the other agent would do with different goals,
    rather than training a separate opponent model.
    
    For our implementation, we use variational inference to infer
    the latent goal from observed behavior, then condition predictions
    on this inferred goal.
    
    Reference:
        Raileanu, R., Denton, E., Szlam, A., & Fergus, R. (2018).
        Modeling Others using Oneself in Multi-Agent Reinforcement Learning.
        ICML 2018.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        som_config: SOMConfig = None,
    ):
        super().__init__(config)
        
        self.som_config = som_config or SOMConfig()
        sc = self.som_config
        
        # Goal inference network
        input_dim = config.obs_dim + config.action_dim
        self.goal_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=sc.policy_hidden,
            batch_first=True,
        )
        
        # Goal distribution parameters
        output_dim = sc.goal_dim * 2 if sc.use_variational else sc.goal_dim
        self.goal_proj = nn.Linear(sc.policy_hidden, output_dim)
        
        # Goal embedding
        self.goal_embed = nn.Linear(sc.goal_dim, sc.policy_hidden)
        
        # Policy LSTM
        policy_input = config.obs_dim + config.action_dim + sc.policy_hidden
        self.policy_lstm = nn.LSTM(
            input_size=policy_input,
            hidden_size=sc.policy_hidden,
            batch_first=True,
        )
        
        # Output heads
        self.action_head = nn.Linear(sc.policy_hidden, config.action_dim)
        self.type_head = nn.Linear(sc.policy_hidden, config.num_types)
        
        self.kl_loss = torch.tensor(0.0)
    
    def infer_goal(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Infer partner's hidden goal from behavior.
        
        Returns:
            goal: [B, T, goal_dim] - sampled or mean goal
            goal_mean: [B, T, goal_dim] - mean (if variational)
            goal_logvar: [B, T, goal_dim] - log variance (if variational)
        """
        sc = self.som_config
        
        # Create trajectory features
        act_oh = F.one_hot(actions, self.action_dim).float()
        traj = torch.cat([observations, act_oh], dim=-1)
        
        # Encode
        h, _ = self.goal_lstm(traj)
        goal_params = self.goal_proj(h)
        
        if sc.use_variational:
            goal_mean, goal_logvar = goal_params.chunk(2, dim=-1)
            
            # Reparameterization
            if self.training:
                std = torch.exp(0.5 * goal_logvar)
                eps = torch.randn_like(std)
                goal = goal_mean + eps * std
            else:
                goal = goal_mean
            
            return goal, goal_mean, goal_logvar
        else:
            return goal_params, None, None
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Forward pass through SOM."""
        B, T = actions.shape
        device = actions.device
        sc = self.som_config
        
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Infer goal
        goal, goal_mean, goal_logvar = self.infer_goal(obs, actions)
        
        # Compute KL loss
        if sc.use_variational and goal_mean is not None:
            self.kl_loss = -0.5 * torch.mean(
                1 + goal_logvar - goal_mean.pow(2) - goal_logvar.exp()
            )
        else:
            self.kl_loss = torch.tensor(0.0, device=device)
        
        # Embed goal
        goal_embedded = self.goal_embed(goal)
        
        # Policy forward
        act_oh = F.one_hot(actions, self.action_dim).float()
        policy_input = torch.cat([obs, act_oh, goal_embedded], dim=-1)
        h, _ = self.policy_lstm(policy_input)
        
        # Predictions
        action_logits = self.action_head(h)
        type_logits = self.type_head(h)
        
        # Goal changes for switch detection
        belief_changes = torch.zeros(B, T, device=device)
        if T > 1 and goal_mean is not None:
            goal_diff = (goal_mean[:, 1:] - goal_mean[:, :-1]).norm(dim=-1)
            belief_changes[:, 1:] = goal_diff
        
        switch_probs = belief_changes / (belief_changes.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
            goal=goal,
            goal_mean=goal_mean,
            goal_logvar=goal_logvar,
            kl_loss=self.kl_loss,
        )
    
    def get_kl_loss(self) -> torch.Tensor:
        """Get KL divergence loss for VAE training."""
        return self.kl_loss
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Context-Conditional Policy
# =============================================================================

@dataclass
class ContextConditionalConfig:
    """
    Configuration for Context-Conditional Policy.
    
    Based on approaches like PEARL (Rakelly et al. 2019) and VariBAD (Zintgraf et al. 2020).
    """
    context_dim: int = 32
    encoder_hidden: int = 64
    policy_hidden: int = 128
    use_variational: bool = True
    use_attention: bool = False  # Attention over trajectory


class ContextConditionalPolicy(BaseModel):
    """
    Context-Conditional Policy Baseline.
    
    A general approach for adapting to different partners/tasks by
    inferring a latent context from behavior history.
    
    Architecture:
        1. Context encoder: trajectory → latent context z
        2. Policy: (obs, z) → action distribution
    
    The context z captures partner-specific information:
        - Strategy type (aggressive, defensive, cooperative)
        - Skill level
        - Behavioral preferences
    
    Related work:
        - PEARL (Rakelly et al. 2019) for meta-RL
        - VariBAD (Zintgraf et al. 2020) for belief-based RL
        - Hausman et al. 2018 for skill embeddings
    
    Comparison to UA-ToM:
        - Both infer latent state from behavior
        - Context-conditional uses single continuous vector
        - UA-ToM uses discrete beliefs + explicit tracking
    """
    
    def __init__(
        self,
        config: ModelConfig,
        cc_config: ContextConditionalConfig = None,
    ):
        super().__init__(config)
        
        self.cc_config = cc_config or ContextConditionalConfig()
        cc = self.cc_config
        
        # Context encoder
        input_dim = config.obs_dim + config.action_dim
        
        if cc.use_attention:
            # Attention-based encoder
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=cc.encoder_hidden,
                    nhead=4,
                    batch_first=True,
                ),
                num_layers=2,
            )
            self.input_proj = nn.Linear(input_dim, cc.encoder_hidden)
        else:
            # GRU-based encoder
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=cc.encoder_hidden,
                batch_first=True,
            )
            self.input_proj = None
        
        # Context projection
        output_dim = cc.context_dim * 2 if cc.use_variational else cc.context_dim
        self.context_proj = nn.Linear(cc.encoder_hidden, output_dim)
        
        # Policy network
        policy_input = config.obs_dim + cc.context_dim
        self.policy = nn.Sequential(
            nn.Linear(policy_input, cc.policy_hidden),
            nn.ReLU(),
            nn.Linear(cc.policy_hidden, cc.policy_hidden),
            nn.ReLU(),
        )
        
        # Output heads
        self.action_head = nn.Linear(cc.policy_hidden, config.action_dim)
        self.type_head = nn.Linear(cc.policy_hidden, config.num_types)
        
        self.kl_loss = torch.tensor(0.0)
    
    def encode_context(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode trajectory to context.
        
        Returns:
            context: [B, T, context_dim]
            context_mean: if variational
            context_logvar: if variational
        """
        cc = self.cc_config
        
        # Create trajectory
        act_oh = F.one_hot(actions, self.action_dim).float()
        traj = torch.cat([observations, act_oh], dim=-1)
        
        # Encode
        if cc.use_attention:
            x = self.input_proj(traj)
            h = self.encoder(x)
        else:
            h, _ = self.encoder(traj)
        
        # Project to context
        context_params = self.context_proj(h)
        
        if cc.use_variational:
            context_mean, context_logvar = context_params.chunk(2, dim=-1)
            
            if self.training:
                std = torch.exp(0.5 * context_logvar)
                eps = torch.randn_like(std)
                context = context_mean + eps * std
            else:
                context = context_mean
            
            return context, context_mean, context_logvar
        else:
            return context_params, None, None
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Forward pass."""
        B, T = actions.shape
        device = actions.device
        cc = self.cc_config
        
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Encode context
        context, context_mean, context_logvar = self.encode_context(obs, actions)
        
        # KL loss
        if cc.use_variational and context_mean is not None:
            self.kl_loss = -0.5 * torch.mean(
                1 + context_logvar - context_mean.pow(2) - context_logvar.exp()
            )
        else:
            self.kl_loss = torch.tensor(0.0, device=device)
        
        # Policy forward
        policy_input = torch.cat([obs, context], dim=-1)
        features = self.policy(policy_input)
        
        # Predictions
        action_logits = self.action_head(features)
        type_logits = self.type_head(features)
        
        # Context changes for switch detection
        belief_changes = torch.zeros(B, T, device=device)
        if T > 1 and context_mean is not None:
            context_diff = (context_mean[:, 1:] - context_mean[:, :-1]).norm(dim=-1)
            belief_changes[:, 1:] = context_diff
        
        switch_probs = belief_changes / (belief_changes.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
            context=context,
            context_mean=context_mean,
            context_logvar=context_logvar,
            kl_loss=self.kl_loss,
        )
    
    def get_kl_loss(self) -> torch.Tensor:
        return self.kl_loss
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# MBOM-Lite: Simplified Model-Based Opponent Modeling
# =============================================================================

class MBOMLite(BaseModel):
    """
    Simplified Model-Based Opponent Modeling.
    
    Inspired by Yu et al. (2022) "Model-Based Opponent Modeling", NeurIPS 2022.
    
    The full MBOM learns a world model including opponent dynamics and uses
    it for planning. This simplified version:
        1. Learns opponent dynamics model: (state, our_action) → opponent_action
        2. Uses model for multi-step prediction
        3. Detects switches via prediction error
    
    Architecture:
        - Dynamics model: predicts partner's next action given state
        - Prediction errors indicate model mismatch → potential switch
    """
    
    def __init__(
        self,
        config: ModelConfig,
        dynamics_hidden: int = 128,
        prediction_horizon: int = 3,
    ):
        super().__init__(config)
        
        self.prediction_horizon = prediction_horizon
        
        # Dynamics model: (obs, own_action_embed) → partner_action
        # We simplify by just using obs history
        input_dim = config.obs_dim + config.action_dim
        
        self.dynamics_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=dynamics_hidden,
            batch_first=True,
        )
        
        self.dynamics_head = nn.Sequential(
            nn.Linear(dynamics_hidden, dynamics_hidden),
            nn.ReLU(),
            nn.Linear(dynamics_hidden, config.action_dim),
        )
        
        # Type prediction from dynamics representation
        self.type_head = nn.Linear(dynamics_hidden, config.num_types)
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        types: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Forward pass."""
        B, T = actions.shape
        device = actions.device
        
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Encode trajectory
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = torch.cat([obs, act_oh], dim=-1)
        h, _ = self.dynamics_encoder(x)  # [B, T, hidden]
        
        # Predict actions
        action_logits = self.dynamics_head(h)  # [B, T, action_dim]
        type_logits = self.type_head(h)
        
        # Compute prediction errors as switch signal
        # Compare predicted action with actual action
        pred_probs = F.softmax(action_logits, dim=-1)
        actual_probs = F.one_hot(actions, self.action_dim).float()
        
        # Cross-entropy style error
        prediction_errors = -(actual_probs * torch.log(pred_probs + 1e-8)).sum(dim=-1)
        
        # Normalize
        belief_changes = prediction_errors
        switch_probs = belief_changes / (belief_changes.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
            prediction_errors=prediction_errors,
        )
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Utility Functions
# =============================================================================

def create_extended_opponent_model(
    model_type: str,
    config: ModelConfig,
    **kwargs,
) -> BaseModel:
    """
    Factory function for extended opponent modeling baselines.
    
    Args:
        model_type: One of 'liam', 'som', 'context_conditional', 'mbom_lite'
        config: Model configuration
        **kwargs: Model-specific config
    
    Returns:
        Instantiated model
    """
    models = {
        'liam': LIAMBaseline,
        'som': SOMBaseline,
        'context_conditional': ContextConditionalPolicy,
        'mbom_lite': MBOMLite,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model: {model_type}. Choose from: {list(models.keys())}")
    
    return models[model_type](config, **kwargs)


def get_all_extended_models() -> List[str]:
    """Return list of available extended opponent models."""
    return ['liam', 'som', 'context_conditional', 'mbom_lite']
