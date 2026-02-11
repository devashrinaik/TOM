"""
Overcooked pipeline for UA-ToM experiments.

Provides data collection, partner strategies, ego policies,
and closed-loop evaluation for the Overcooked-AI environment.
"""

from .partner_strategies import (
    PARTNER_TYPES,
    PARTNER_NAMES,
    NUM_PARTNER_TYPES,
    CooperativePartner,
    GreedyPartner,
    AdversarialPartner,
    RandomPartner,
)

from .data_collector import OvercookedEpisodeCollector

from .ego_policy import (
    OvercookedAdaptiveEgo,
    OvercookedModelWrapper,
    EGO_STRATEGIES,
    COUNTER_STRATEGY,
)

from .closed_loop_eval import (
    run_closed_loop_episode,
    run_evaluation,
    print_results_table,
)

__all__ = [
    # Partner strategies
    'PARTNER_TYPES',
    'PARTNER_NAMES',
    'NUM_PARTNER_TYPES',
    'CooperativePartner',
    'GreedyPartner',
    'AdversarialPartner',
    'RandomPartner',

    # Data collection
    'OvercookedEpisodeCollector',

    # Ego policy
    'OvercookedAdaptiveEgo',
    'OvercookedModelWrapper',
    'EGO_STRATEGIES',
    'COUNTER_STRATEGY',

    # Closed-loop evaluation
    'run_closed_loop_episode',
    'run_evaluation',
    'print_results_table',
]
