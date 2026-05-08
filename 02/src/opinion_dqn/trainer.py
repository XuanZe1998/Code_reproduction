"""T-DQN 训练与选种主流程。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig
from .dqn import QNetwork
from .env import SeedSelectionEnv
from .graph import SocialTrustNetwork
from .heuristics import ACTION_NAMES, detect_communities, rank_by_action, select_potential_seeds
from .replay import PrioritizedReplayBuffer, Transition
from .state import build_state_vector


@dataclass
class TDQNResult:
    """T-DQN 训练输出。"""

    candidates: list[int]
    seeds: list[int]
    best_score: float
    episode_scores: list[float]


class TDQNTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.device = self._resolve_device(config.dqn.device)
        # policy_net 用来学习当前动作价值；
        # target_net 用来提供更稳定的 TD 目标值。
        self.policy_net = QNetwork(config.dqn.state_dim, config.dqn.action_dim, config.dqn.hidden_dim).to(self.device)
        self.target_net = QNetwork(config.dqn.state_dim, config.dqn.action_dim, config.dqn.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.dqn.learning_rate)
        self.loss_fn = nn.MSELoss(reduction="none")
        self.replay = PrioritizedReplayBuffer(
            config.dqn.replay_size,
            config.dqn.priority_alpha,
            config.dqn.priority_eps,
        )

    def fit(
        self,
        network: SocialTrustNetwork,
        initial_opinions: np.ndarray,
        initial_strategies: np.ndarray,
    ) -> TDQNResult:
        # 一次 fit 会重复若干轮“从空集合开始选 k 个种子”的 episode，
        # 最后返回训练过程中找到的最好种子集合。
        env = SeedSelectionEnv(
            network=network,
            initial_opinions=initial_opinions,
            initial_strategies=initial_strategies,
            dynamics_config=self.config.dynamics,
            seed_budget=self.config.seed_budget,
            rng=self.rng,
            device_name=str(self.device),
        )
        candidates = select_potential_seeds(
            network,
            initial_opinions,
            self.config.seed_budget,
            self.config.candidate_multiplier,
        )
        community_map = detect_communities(network)

        best_score = float("-inf")
        best_seeds: list[int] = []
        episode_scores: list[float] = []

        for _ in range(self.config.episodes):
            epsilon = self.config.dqn.epsilon_start
            importance_beta = self.config.dqn.importance_beta_start
            seeds: list[int] = []
            selected: set[int] = set()
            current_state = build_state_vector(network, initial_opinions, initial_strategies, seeds)

            for step in range(self.config.seed_budget):
                # 先决定用哪一种动作策略，再按该策略从候选集中选出具体节点。
                action = self._select_action(current_state, epsilon)
                epsilon = max(self.config.dqn.epsilon_min, epsilon * self.config.dqn.epsilon_decay)
                ranked = rank_by_action(network, candidates, selected, action, community_map, self.config.mix_lambda)
                if not ranked:
                    break
                node = ranked[0]
                reward = env.reward(seeds, node)
                next_seeds = seeds + [node]
                next_state = env.next_state(next_seeds)
                done = len(next_seeds) >= self.config.seed_budget
                self.replay.add(
                    Transition(
                        state=current_state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                    )
                )
                self._optimize(importance_beta)
                if (step + 1) % self.config.dqn.target_update_interval == 0:
                    self._soft_update()
                importance_beta = min(1.0, importance_beta + self.config.dqn.importance_beta_increment)
                seeds = next_seeds
                selected.add(node)
                current_state = next_state

            score = env.evaluate(seeds)
            episode_scores.append(score)
            if score > best_score:
                best_score = score
                best_seeds = seeds.copy()

        return TDQNResult(
            candidates=candidates,
            seeds=best_seeds,
            best_score=best_score,
            episode_scores=episode_scores,
        )

    def _select_action(self, state: np.ndarray, epsilon: float) -> int:
        # 论文采用 epsilon-greedy：一部分概率随机探索，其余概率贪心利用。
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.config.dqn.action_dim))
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def _optimize(self, importance_beta: float) -> None:
        if len(self.replay) < self.config.dqn.batch_size:
            return
        # 标准 DQN 更新流程，并加入优先经验回放与重要性采样修正。
        transitions, indices, weights = self.replay.sample(
            self.config.dqn.batch_size,
            importance_beta,
            self.rng,
        )
        states = torch.tensor(np.stack([t.state for t in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([t.next_state for t in transitions]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=self.device)
        sample_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.config.dqn.gamma * next_q * (1.0 - dones)

        td_errors = (targets - q_values).detach().cpu().numpy()
        losses = self.loss_fn(q_values, targets)
        loss = (sample_weights * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.replay.update_priorities(indices, td_errors)

    def _soft_update(self) -> None:
        # 软更新：
        # theta_target <- tau * theta_policy + (1 - tau) * theta_target
        tau = self.config.dqn.tau
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def action_name(action_idx: int) -> str:
        return ACTION_NAMES[action_idx]

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        # 自动选择设备：优先使用 CUDA，其次使用 Intel XPU，最后退回 CPU。
        if device_name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return torch.device("xpu")
            return torch.device("cpu")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前 PyTorch 不支持 CUDA，无法按要求使用 GPU。")
        if device_name == "xpu":
            if not hasattr(torch, "xpu") or not torch.xpu.is_available():
                raise RuntimeError("当前 PyTorch 不支持 XPU，无法按要求使用 Intel GPU。")
        return torch.device(device_name)
