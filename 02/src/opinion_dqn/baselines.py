"""论文中各类固定启发式基线。"""

from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .dynamics import run_opinion_dynamics
from .graph import SocialTrustNetwork
from .heuristics import detect_communities, rank_by_action, select_potential_seeds


def run_single_strategy(
    network: SocialTrustNetwork,
    initial_opinions: np.ndarray,
    initial_strategies: np.ndarray,
    config: ExperimentConfig,
    action_idx: int,
) -> tuple[list[int], float]:
    # 固定基线：整个选种过程始终使用同一种启发式动作策略。
    rng = np.random.default_rng(config.random_seed)
    candidates = select_potential_seeds(
        network,
        initial_opinions,
        config.seed_budget,
        config.candidate_multiplier,
    )
    communities = detect_communities(network)
    selected: list[int] = []
    selected_set: set[int] = set()
    for _ in range(config.seed_budget):
        ranked = rank_by_action(network, candidates, selected_set, action_idx, communities, config.mix_lambda)
        if not ranked:
            break
        node = ranked[0]
        selected.append(node)
        selected_set.add(node)
    result = run_opinion_dynamics(
        network,
        initial_opinions,
        initial_strategies,
        selected,
        config.dynamics,
        rng,
    )
    return selected, result.overall_opinion
