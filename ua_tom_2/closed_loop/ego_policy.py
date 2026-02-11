#!/usr/bin/env python3
"""
Adaptive ego policy for closed-loop evaluation.

Maps model predictions (type_logits, switch_probs) to ego strategy selection.
Supports multiple experimental conditions (oracle, model-based, etc.).

Conditions:
    oracle:          Ground truth type + switch time
    model:           Use pretrained model predictions
    no_detect:       Fixed initial strategy, never adapt
    random:          Random strategy switches (one per episode, random timing)
    always_standard: Fixed standard strategy
"""

import numpy as np
import torch

from .ego_strategies import ALL_EGO_STRATEGIES, COUNTER_STRATEGY


class ModelInferenceWrapper:
    """Wraps a pretrained model for step-by-step inference in closed-loop."""

    def __init__(self, model, device, max_steps=150):
        self.model = model
        self.device = device
        self.max_steps = max_steps
        self.obs_dim = model.config.obs_dim

        self.obs_buffer = np.zeros((max_steps, self.obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros(max_steps, dtype=np.int64)
        self.step = 0

    def reset(self):
        self.obs_buffer[:] = 0
        self.action_buffer[:] = 0
        self.step = 0

    def update(self, obs_vec, partner_action_discrete):
        """
        Add observation and action for this timestep, run model inference.

        Args:
            obs_vec: np.ndarray [obs_dim] current state observation
            partner_action_discrete: int (0-7) discretized partner action

        Returns:
            dict with pred_type (int), switch_prob (float), type_probs (ndarray)
        """
        if self.step >= self.max_steps:
            return {'pred_type': 3, 'switch_prob': 0.0,
                    'type_probs': np.array([0, 0, 0, 1.0])}

        self.obs_buffer[self.step] = obs_vec
        self.action_buffer[self.step] = partner_action_discrete
        self.step += 1

        t = self.step
        obs_t = torch.from_numpy(
            self.obs_buffer[:t].copy()
        ).unsqueeze(0).float().to(self.device)
        act_t = torch.from_numpy(
            self.action_buffer[:t].copy()
        ).unsqueeze(0).long().to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(obs_t, act_t)

        type_logits = outputs.type_logits[0, -1].cpu()
        type_probs = torch.softmax(type_logits, dim=0).numpy()
        switch_prob = outputs.switch_probs[0, -1].item()

        return {
            'pred_type': int(type_logits.argmax().item()),
            'switch_prob': switch_prob,
            'type_probs': type_probs,
        }


class AdaptiveEgoPolicy:
    """
    Selects ego strategy based on model predictions or experimental condition.

    The policy maintains a current strategy (from ego_strategies) and updates
    it when a switch is detected.
    """

    def __init__(self, condition='oracle', model=None, device=None,
                 switch_threshold=0.3, cooldown=5, max_steps=150):
        """
        Args:
            condition: 'oracle', 'model', 'no_detect', 'random', 'always_standard'
            model: pretrained BaseModel (required for condition='model')
            device: torch device (required for condition='model')
            switch_threshold: switch_probs threshold for model condition
            cooldown: min steps between strategy switches
            max_steps: max episode length
        """
        self.condition = condition
        self.switch_threshold = switch_threshold
        self.cooldown = cooldown
        self.max_steps = max_steps

        # Model wrapper (for model-based conditions)
        self.model_wrapper = None
        if model is not None and condition == 'model':
            self.model_wrapper = ModelInferenceWrapper(model, device, max_steps)

        # Strategy state
        self.current_strategy_name = 'standard'
        self.current_strategy = ALL_EGO_STRATEGIES['standard']()
        self.steps_since_switch = 0
        self.step_count = 0

        # Random condition state
        self._random_switch_time = -1
        self._random_switch_type = 3

    def reset(self, initial_type=None):
        """Reset for new episode."""
        self.step_count = 0
        self.steps_since_switch = 0

        if self.model_wrapper is not None:
            self.model_wrapper.reset()

        # Set initial strategy
        if self.condition == 'always_standard':
            self._set_strategy('standard')
        elif initial_type is not None:
            counter = COUNTER_STRATEGY.get(initial_type, 'standard')
            self._set_strategy(counter)
        else:
            self._set_strategy('standard')

        # Random condition: pick one random switch time + type
        if self.condition == 'random':
            self._random_switch_time = np.random.randint(0, self.max_steps)
            self._random_switch_type = np.random.choice([0, 3])

    def _set_strategy(self, name):
        """Switch to a named strategy, reset its internal state."""
        if name != self.current_strategy_name or self.step_count == 0:
            self.current_strategy_name = name
            self.current_strategy = ALL_EGO_STRATEGIES[name]()
            self.current_strategy.reset()
            self.steps_since_switch = 0

    def _is_committed(self):
        """
        Check if current strategy is committed to a grasp sequence.

        Don't abandon a descend/grasp/lift in progress to switch to yield.
        Switching to a more active strategy (yield → standard) is always OK.
        """
        strat = self.current_strategy
        if hasattr(strat, 'phase') and strat.phase in ('descend', 'grasp', 'lift'):
            return True
        return False

    def _try_switch(self, new_strategy_name):
        """Switch strategy if not committed to an irreversible action."""
        # Switching to a more active strategy is always OK
        if new_strategy_name == 'standard' and self.current_strategy_name == 'yield':
            self._set_strategy(new_strategy_name)
            return
        # Switching to yield from an active grasp: only if still in approach
        if new_strategy_name == 'yield' and self._is_committed():
            return  # Don't abandon grasp
        self._set_strategy(new_strategy_name)

    def update(self, obs_vec=None, partner_action_discrete=None,
               ground_truth_type=None, ground_truth_switch=False):
        """
        Update policy based on new observation.

        Called AFTER the ego has acted at this step. Updates belief for
        the NEXT step's action.

        Args:
            obs_vec: np.ndarray [obs_dim] (for model conditions)
            partner_action_discrete: int 0-7 (for model conditions)
            ground_truth_type: int (for oracle condition)
            ground_truth_switch: bool (for oracle condition)
        """
        self.step_count += 1
        self.steps_since_switch += 1

        if self.condition == 'oracle':
            if ground_truth_switch and ground_truth_type is not None:
                counter = COUNTER_STRATEGY.get(ground_truth_type, 'standard')
                self._try_switch(counter)

        elif self.condition == 'model':
            if self.model_wrapper is not None and obs_vec is not None:
                pred = self.model_wrapper.update(obs_vec, partner_action_discrete)
                if (pred['switch_prob'] > self.switch_threshold
                        and self.steps_since_switch >= self.cooldown):
                    counter = COUNTER_STRATEGY.get(
                        pred['pred_type'], 'standard')
                    self._try_switch(counter)

        elif self.condition == 'no_detect':
            pass  # Never change strategy

        elif self.condition == 'random':
            if self.step_count == self._random_switch_time:
                counter = COUNTER_STRATEGY.get(
                    self._random_switch_type, 'standard')
                self._try_switch(counter)

        elif self.condition == 'always_standard':
            pass  # Fixed standard

    def get_action(self, left_tcp, cube_pos, right_tcp=None):
        """Get ego action from current strategy."""
        return self.current_strategy.get_action(left_tcp, cube_pos, right_tcp)
