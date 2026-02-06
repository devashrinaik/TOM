"""
Bayesian Online Change-Point Detection (BOCPD)
==============================================

Exact implementation following Adams & MacKay (2007):
"Bayesian Online Changepoint Detection", arXiv:0710.3742

Paper: https://arxiv.org/abs/0710.3742
Code reference: http://www.inference.phy.cam.ac.uk/rpa23/changepoint.php

This is a CLASSICAL statistical method - the change-point detection itself
requires NO neural network training. Uses exact Bayesian inference with 
conjugate priors (Normal-Inverse-Gamma → Student-t predictive).

Algorithm Overview (from paper):
    The key insight is that we can recursively compute P(r_t, x_{1:t}), 
    the joint distribution over the run length r_t and data x_{1:t}.
    
    At each timestep:
    1. Compute predictive P(x_t | r, x_{r:t-1}) using conjugate prior
    2. Growth: P(r_t=r+1) ∝ P(r_{t-1}=r) × P(x_t|r) × (1-H(r))
    3. Changepoint: P(r_t=0) ∝ Σ_r P(r_{t-1}=r) × P(x_t|r) × H(r)
    4. Normalize and return P(r_t=0) as changepoint probability

    Where H(r) is the hazard function (probability of changepoint given
    run length r). We use constant hazard H(r) = 1/λ (geometric prior).

For our baseline:
    - A GRU predicts partner actions → generates prediction error stream
    - BOCPD detects changepoints in the error stream (no learning!)
    - Tests: Is UA-ToM's learned detection better than classical Bayes?

Expected Results:
    ✓ Good precision on clean, distinct behavioral switches
    ✓ Calibrated uncertainty (proper Bayesian inference)
    ✗ May struggle with noisy or gradual transitions
    ✗ No task-specific adaptation (hyperparameters are fixed)
    ✗ Sensitive to hazard rate hyperparameter

Alternative packages:
    - bayesian-changepoint-detection: pip install bayesian-changepoint-detection
    - ruptures: pip install ruptures (offline only)
"""

import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from functools import lru_cache
from scipy.special import logsumexp as scipy_logsumexp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..base import BaseModel, ModelConfig, ModelOutput


def logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    if len(x) == 0:
        return -np.inf
    max_x = np.max(x)
    if max_x == -np.inf:
        return -np.inf
    return max_x + np.log(np.sum(np.exp(x - max_x)))


# =============================================================================
# BOCPD Configuration
# =============================================================================

@dataclass
class BOCPDConfig:
    """
    Configuration for BOCPD algorithm following Adams & MacKay 2007.
    
    Key hyperparameters:
        hazard_lambda: Expected run length between changepoints.
                      Larger = fewer expected changepoints.
                      Set to ~episode_length/expected_switches.
                      
        prior_alpha, prior_beta: Normal-Inverse-Gamma prior parameters.
                                Control how quickly we adapt to new data.
                                Smaller = faster adaptation, less stable.
                                
        prior_kappa: Pseudo-observations for prior mean.
                    Controls confidence in prior_mu.
                    
        prior_mu: Prior mean for observation model.
                 For prediction errors, set to expected error (e.g., 0).
    """
    
    # Hazard rate λ: Expected run length between changepoints
    # For constant hazard: H(τ) = 1/λ (geometric prior)
    # λ = 50 means we expect a changepoint every ~50 timesteps
    hazard_lambda: float = 50.0
    
    # Normal-Inverse-Gamma (NIG) prior parameters
    # These define the Student-t predictive distribution
    # Smaller values = more diffuse prior, faster adaptation
    prior_alpha: float = 0.1   # Shape (ν/2 where ν is degrees of freedom)
    prior_beta: float = 0.1    # Rate (controls scale)
    prior_kappa: float = 1.0   # Pseudo-count for mean
    prior_mu: float = 0.0      # Prior mean (0 for prediction errors)
    
    # Computational settings
    max_run_length: int = 300      # Max run length to track
    prune_threshold: float = 1e-10  # Prune run lengths below this prob
    use_log_space: bool = True      # Use log-space for numerical stability


# =============================================================================
# Student-t Sufficient Statistics (Conjugate Prior)
# =============================================================================

class StudentTSufficientStats:
    """
    Sufficient statistics for Student-t conjugate prior.
    
    This implements the Normal-Inverse-Gamma (NIG) conjugate prior
    which leads to a Student-t predictive distribution.
    
    Prior: μ ~ N(μ_0, σ²/κ_0), σ² ~ IG(α_0, β_0)
    Posterior: Same form with updated parameters
    Predictive: Student-t distribution
    
    Following Murphy (2007) "Conjugate Bayesian analysis of the Gaussian distribution"
    """
    
    def __init__(self, alpha: float, beta: float, kappa: float, mu: float):
        """
        Initialize with NIG hyperparameters.
        
        Args:
            alpha: Shape parameter (df/2)
            beta: Rate parameter  
            kappa: Pseudo-observations
            mu: Prior mean
        """
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.mu = mu
        
        # Sufficient statistics
        self.n = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
    
    def update(self, x: float) -> 'StudentTSufficientStats':
        """
        Return NEW stats object with observation incorporated.
        
        Posterior update rules from Murphy (2007):
            κ_n = κ_0 + n
            μ_n = (κ_0 * μ_0 + Σx_i) / κ_n
            α_n = α_0 + n/2
            β_n = β_0 + 0.5 * (Σx_i² - κ_n * μ_n²)
        """
        new_stats = StudentTSufficientStats(
            self.alpha, self.beta, self.kappa, self.mu
        )
        new_stats.n = self.n + 1
        new_stats.sum_x = self.sum_x + x
        new_stats.sum_x2 = self.sum_x2 + x * x
        return new_stats
    
    def predictive_log_prob(self, x: float) -> float:
        """
        Compute log predictive probability p(x | past data).
        
        The predictive distribution is Student-t:
            p(x_new | D) = t_{2α_n}(μ_n, β_n(κ_n+1)/(α_n * κ_n))
        
        Using the standard Student-t log pdf.
        """
        # Posterior parameters
        kappa_n = self.kappa + self.n
        mu_n = (self.kappa * self.mu + self.sum_x) / kappa_n if kappa_n > 0 else self.mu
        alpha_n = self.alpha + self.n / 2.0
        
        # For beta_n, we need the sum of squared deviations
        if self.n > 0:
            # β_n = β_0 + 0.5 * (Σ(x-μ_n)² + κ_0*n*(μ_0-x̄)²/(κ_0+n))
            sample_mean = self.sum_x / self.n if self.n > 0 else 0
            ss = self.sum_x2 - self.n * sample_mean ** 2  # Sum of squared deviations
            beta_n = self.beta + 0.5 * ss + \
                     0.5 * self.kappa * self.n * (self.mu - sample_mean) ** 2 / kappa_n
        else:
            beta_n = self.beta
        
        # Predictive Student-t parameters
        df = 2 * alpha_n
        pred_mu = mu_n
        pred_var = beta_n * (kappa_n + 1) / (alpha_n * kappa_n)
        pred_scale = math.sqrt(pred_var)
        
        # Student-t log pdf
        z = (x - pred_mu) / pred_scale
        log_prob = (
            math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
            - 0.5 * math.log(df * math.pi)
            - math.log(pred_scale)
            - (df + 1) / 2 * math.log(1 + z ** 2 / df)
        )
        
        return log_prob


# =============================================================================
# BOCPD Detector (Standalone)
# =============================================================================

class BOCPDDetector:
    """
    Bayesian Online Change-Point Detection following Adams & MacKay (2007).
    
    Maintains P(r_t | x_{1:t}) - the posterior distribution over run lengths
    given all observations. The run length r_t is the time since the last
    changepoint.
    
    Key insight: We can compute the JOINT P(r_t, x_{1:t}) recursively:
    
        P(r_t, x_{1:t}) = Σ_{r_{t-1}} P(r_t | r_{t-1}) × P(x_t | r_{t-1}, x^{(r)}) 
                                      × P(r_{t-1}, x_{1:t-1})
    
    Where:
        - P(r_t | r_{t-1}) = H(r_{t-1}) if r_t=0, (1-H(r_{t-1})) if r_t=r_{t-1}+1
        - P(x_t | r_{t-1}, x^{(r)}) is the Student-t predictive
        - H(τ) = 1/λ is the constant hazard function
    
    Algorithm (Algorithm 1 from paper):
        1. Observe x_t
        2. For each run length r, compute predictive P(x_t | r, x^{(r)})
        3. Growth probs: P(r_t = r+1) ∝ P(r_{t-1}=r) × pred × (1-H)
        4. Changepoint: P(r_t = 0) ∝ Σ_r P(r_{t-1}=r) × pred × H
        5. Normalize to get posterior P(r_t | x_{1:t})
        6. Update sufficient statistics for each hypothesis
        
    Uses log-space computation for numerical stability on long sequences.
    """
    
    def __init__(self, config: BOCPDConfig = None):
        """
        Initialize BOCPD detector.
        
        Args:
            config: BOCPD configuration (uses defaults if None)
        """
        self.config = config or BOCPDConfig()
        
        # Log run length distribution: log P(r_t = r | x_{1:t})
        # We work in log space for numerical stability
        self.log_run_length_dist = np.full(self.config.max_run_length, -np.inf)
        self.log_run_length_dist[0] = 0.0  # P(r_0 = 0) = 1, so log = 0
        
        # Sufficient statistics for each possible run length
        # stats[r] contains statistics for hypothesis "run length = r"
        self.stats: List[StudentTSufficientStats] = [
            StudentTSufficientStats(
                self.config.prior_alpha,
                self.config.prior_beta, 
                self.config.prior_kappa,
                self.config.prior_mu,
            )
        ]
        
        # Precompute log hazard terms
        self.log_H = np.log(1.0 / self.config.hazard_lambda)
        self.log_1mH = np.log(1.0 - 1.0 / self.config.hazard_lambda)
    
    def hazard(self, r: int) -> float:
        """
        Hazard function H(r) = P(changepoint | run_length = r).
        
        Using constant hazard (geometric prior on run lengths):
            H(r) = 1 / λ for all r
        
        This corresponds to a memoryless changepoint process where
        the probability of a changepoint doesn't depend on how long
        the current regime has lasted.
        
        Could be extended to time-varying hazard for more complex priors.
        """
        return 1.0 / self.config.hazard_lambda
    
    def update(self, x: float) -> float:
        """
        Process new observation and return changepoint probability.
        
        Implements Algorithm 1 from Adams & MacKay (2007).
        
        Args:
            x: New observation (scalar, typically prediction error)
        
        Returns:
            P(changepoint at current timestep) = P(r_t = 0 | x_{1:t})
        """
        n_hyp = len(self.stats)
        
        # Step 1: Compute log predictive probabilities for each run length
        log_pred = np.zeros(n_hyp)
        for r in range(n_hyp):
            log_pred[r] = self.stats[r].predictive_log_prob(x)
        
        # Step 2: Compute growth probabilities in log space
        # log P(r_t = r+1, x_{1:t}) = log P(r_{t-1}=r) + log pred + log(1-H)
        log_growth = (
            self.log_run_length_dist[:n_hyp] + 
            log_pred + 
            self.log_1mH
        )
        
        # Step 3: Compute changepoint probability in log space
        # log P(r_t = 0, x_{1:t}) = logsumexp(log P(r_{t-1}=r) + log pred + log H)
        log_cp_terms = self.log_run_length_dist[:n_hyp] + log_pred + self.log_H
        log_cp_mass = logsumexp(log_cp_terms)
        
        # Step 4: Build new log run length distribution
        new_max_len = min(n_hyp + 1, self.config.max_run_length)
        new_log_dist = np.full(new_max_len, -np.inf)
        new_log_dist[0] = log_cp_mass  # Mass at r=0 (changepoint)
        new_log_dist[1:min(n_hyp+1, new_max_len)] = log_growth[:new_max_len-1]
        
        # Normalize: subtract logsumexp to get proper log probabilities
        log_evidence = logsumexp(new_log_dist)
        if log_evidence > -np.inf:
            new_log_dist -= log_evidence
        else:
            # Numerical issue - reset to changepoint
            new_log_dist = np.full(new_max_len, -np.inf)
            new_log_dist[0] = 0.0
        
        # Prune very low probability run lengths (optional, for efficiency)
        log_threshold = np.log(self.config.prune_threshold)
        new_log_dist[new_log_dist < log_threshold] = -np.inf
        
        # Re-normalize after pruning
        log_evidence = logsumexp(new_log_dist)
        if log_evidence > -np.inf:
            new_log_dist -= log_evidence
        
        self.log_run_length_dist = new_log_dist
        
        # Step 5: Update sufficient statistics
        # r=0 gets fresh prior, others get updated with new observation
        new_stats = [
            StudentTSufficientStats(
                self.config.prior_alpha,
                self.config.prior_beta,
                self.config.prior_kappa, 
                self.config.prior_mu,
            )
        ]
        for stats in self.stats[:self.config.max_run_length - 1]:
            new_stats.append(stats.update(x))
        self.stats = new_stats
        
        # Return changepoint probability
        return float(np.exp(new_log_dist[0]))
    
    def reset(self):
        """Reset detector to initial state."""
        self.log_run_length_dist = np.full(self.config.max_run_length, -np.inf)
        self.log_run_length_dist[0] = 0.0
        self.stats = [
            StudentTSufficientStats(
                self.config.prior_alpha,
                self.config.prior_beta,
                self.config.prior_kappa,
                self.config.prior_mu,
            )
        ]
    
    def get_run_length_distribution(self) -> np.ndarray:
        """Get current posterior over run lengths (probabilities, not log)."""
        return np.exp(self.log_run_length_dist)
    
    def get_map_run_length(self) -> int:
        """Get maximum a posteriori run length estimate."""
        return int(np.argmax(self.log_run_length_dist))


# =============================================================================
# BOCPD Baseline Model
# =============================================================================

class BOCPDBaseline(BaseModel):
    """
    BOCPD baseline for partner switch detection.
    
    This is a HYBRID approach:
        - Prediction: Learned neural network (GRU)
        - Change-point detection: Classical BOCPD (no neural learning)
    
    The model uses a simple GRU to predict partner actions and compute
    prediction errors. BOCPD is then applied to the error stream to
    detect change-points (partner switches).
    
    This tests whether UA-ToM's LEARNED switch detection outperforms
    classical Bayesian change-point detection on the same error signal.
    
    Expected results:
        - Good precision on clean, distinct switches
        - May struggle with noisy observations
        - No learned adaptation to task-specific switch patterns
        - Provides calibrated uncertainty (proper Bayesian inference)
    
    Reference:
        Adams & MacKay (2007). Bayesian Online Changepoint Detection.
        arXiv:0710.3742
    """
    
    def __init__(
        self,
        config: ModelConfig,
        hazard_lambda: float = 50.0,
        prior_alpha: float = 0.1,
        prior_beta: float = 0.1,
    ):
        """
        Args:
            config: Model configuration
            hazard_lambda: Expected run length between changepoints
            prior_alpha: Student-t prior shape parameter
            prior_beta: Student-t prior rate parameter
        """
        super().__init__(config)
        
        self.bocpd_config = BOCPDConfig(
            hazard_lambda=hazard_lambda,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )
        
        input_dim = config.obs_dim + config.action_dim
        
        # Simple GRU predictor for generating error stream
        self.predictor = nn.GRU(
            input_size=input_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        
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
        Forward pass with BOCPD for switch detection.
        
        Args:
            observations: [B, T, obs_dim] or [B, obs_dim]
            actions: [B, T] action indices
            types: Partner type indices (unused)
            images: Images (unused)
        
        Returns:
            ModelOutput with BOCPD switch probabilities
        """
        B, T = actions.shape
        device = actions.device
        
        # Handle observations
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        # Embed inputs
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = torch.cat([obs, act_oh], dim=-1)
        
        # Get predictions from GRU
        hidden, _ = self.predictor(x)
        action_logits = self.action_head(hidden)
        type_logits = self.type_head(hidden)
        
        # Compute prediction errors (negative log probability)
        errors = torch.zeros(B, T, device=device)
        if T > 1:
            with torch.no_grad():
                pred_probs = F.softmax(action_logits[:, :-1], dim=-1)
                target_actions = actions[:, 1:]
                
                for b in range(B):
                    for t in range(T - 1):
                        prob = pred_probs[b, t, target_actions[b, t]].clamp(min=1e-8)
                        errors[b, t + 1] = -torch.log(prob)
        
        # Apply BOCPD to error stream (per batch element)
        switch_probs = torch.zeros(B, T, device=device)
        
        for b in range(B):
            detector = BOCPDDetector(self.bocpd_config)
            
            for t in range(T):
                error_t = errors[b, t].item()
                cp_prob = detector.update(error_t)
                switch_probs[b, t] = cp_prob
        
        # Belief changes from type predictions
        type_probs = F.softmax(type_logits, dim=-1)
        belief_changes = torch.zeros(B, T, device=device)
        if T > 1:
            belief_changes[:, 1:] = (type_probs[:, 1:] - type_probs[:, :-1]).abs().sum(dim=-1)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
            prediction_errors=errors,
        )
    
    def get_switch_signal(self, outputs: ModelOutput) -> torch.Tensor:
        """BOCPD provides direct change-point probabilities."""
        return outputs.switch_probs
    
    def count_parameters(self) -> int:
        """Count trainable parameters (only the predictor, not BOCPD)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_param_breakdown(self) -> dict:
        """Get parameter count breakdown by component."""
        breakdown = {
            'predictor_gru': sum(p.numel() for p in self.predictor.parameters()),
            'action_head': sum(p.numel() for p in self.action_head.parameters()),
            'type_head': sum(p.numel() for p in self.type_head.parameters()),
            'bocpd': 0,  # BOCPD has no learned parameters!
        }
        breakdown['total'] = sum(breakdown.values())
        return breakdown


# =============================================================================
# Try to use bayesian_changepoint_detection package if available
# =============================================================================

BAYESIAN_CPD_AVAILABLE = False
try:
    from bayesian_changepoint_detection import online_changepoint_detection as oncd
    from functools import partial
    BAYESIAN_CPD_AVAILABLE = True
except ImportError:
    pass


class BOCPDBaselinePackage(BaseModel):
    """
    Alternative BOCPD baseline using bayesian_changepoint_detection package.
    
    Installation: pip install bayesian-changepoint-detection
    
    This provides a reference implementation to validate our custom BOCPD.
    """
    
    def __init__(self, config: ModelConfig, hazard_lambda: float = 100.0):
        super().__init__(config)
        
        if not BAYESIAN_CPD_AVAILABLE:
            raise ImportError(
                "bayesian_changepoint_detection not available. "
                "Install with: pip install bayesian-changepoint-detection"
            )
        
        self.hazard_lambda = hazard_lambda
        
        input_dim = config.obs_dim + config.action_dim
        self.predictor = nn.GRU(input_dim, config.hidden_dim, batch_first=True)
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.type_head = nn.Linear(config.hidden_dim, config.num_types)
    
    def forward(self, observations, actions, types=None, images=None):
        B, T = actions.shape
        device = actions.device
        
        obs = observations
        if obs.dim() == 2:
            obs = obs.unsqueeze(1).expand(-1, T, -1)
        
        act_oh = F.one_hot(actions, self.action_dim).float()
        x = torch.cat([obs, act_oh], dim=-1)
        
        hidden, _ = self.predictor(x)
        action_logits = self.action_head(hidden)
        type_logits = self.type_head(hidden)
        
        # Compute errors
        errors = torch.zeros(B, T, device=device)
        if T > 1:
            with torch.no_grad():
                pred_probs = F.softmax(action_logits[:, :-1], dim=-1)
                for b in range(B):
                    for t in range(T - 1):
                        errors[b, t + 1] = -torch.log(
                            pred_probs[b, t, actions[b, t + 1]] + 1e-8
                        )
        
        # Use package for BOCPD
        switch_probs = torch.zeros(B, T, device=device)
        
        for b in range(B):
            error_seq = errors[b].cpu().numpy()
            
            R, maxes = oncd.online_changepoint_detection(
                error_seq,
                partial(oncd.constant_hazard, self.hazard_lambda),
                oncd.StudentT(0.1, 0.01, 1, 0)  # alpha, beta, kappa, mu
            )
            
            # R[0, 1:] contains P(r_t = 0) for each timestep
            cp_probs = R[0, 1:T+1] if R.shape[1] > T else R[0, 1:]
            switch_probs[b, :len(cp_probs)] = torch.from_numpy(cp_probs).to(device)
        
        belief_changes = torch.zeros(B, T, device=device)
        type_probs = F.softmax(type_logits, dim=-1)
        if T > 1:
            belief_changes[:, 1:] = (type_probs[:, 1:] - type_probs[:, :-1]).abs().sum(dim=-1)
        
        return ModelOutput(
            action_logits=action_logits,
            type_logits=type_logits,
            switch_probs=switch_probs,
            belief_changes=belief_changes,
        )
