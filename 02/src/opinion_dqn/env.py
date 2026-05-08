"""种子选择环境：负责奖励评估与状态提取。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DynamicsConfig
from .dynamics import SimulationResult, run_opinion_dynamics
from .graph import SocialTrustNetwork
from .state import build_state_vector


@dataclass
class SeedSelectionEnv:
    """把论文中的“选种子”过程包装成一个强化学习环境。"""

    network: SocialTrustNetwork
    initial_opinions: np.ndarray
    initial_strategies: np.ndarray
    dynamics_config: DynamicsConfig
    seed_budget: int
    rng: np.random.Generator
    device_name: str = "auto"

    def __post_init__(self) -> None:
        # 缓存相同种子集合的传播结果，避免 reward / next_state / evaluate 重复仿真。
        self._simulation_cache: dict[tuple[int, ...], SimulationResult] = {}

    @staticmethod
    def _seed_key(seeds: list[int]) -> tuple[int, ...]:
        return tuple(sorted(seeds))

    def simulate(self, seeds: list[int]) -> SimulationResult:
        key = self._seed_key(seeds)
        if key not in self._simulation_cache:
            self._simulation_cache[key] = run_opinion_dynamics(
                self.network,
                self.initial_opinions,
                self.initial_strategies,
                seeds,
                self.dynamics_config,
                self.rng,
                device_name=self.device_name,
            )
        return self._simulation_cache[key]

    def evaluate(self, seeds: list[int]) -> float:
        # 目标函数就是传播 T 步后的最终总体观点。
        return self.simulate(seeds).overall_opinion

    def reward(self, current_seeds: list[int], new_seed: int) -> float:
        # 奖励定义为加入一个新种子后的最终总体观点增量。
        base = self.evaluate(current_seeds)
        improved = self.evaluate(current_seeds + [new_seed])
        return improved - base

    def next_state(self, selected_seeds: list[int]) -> np.ndarray:
        # 当前已选种子集合确定后，先跑一次传播，再从传播后的系统中抽取状态特征。
        result = self.simulate(selected_seeds)
        return build_state_vector(
            self.network,
            result.final_opinions,
            result.final_strategies,
            selected_seeds,
        )
