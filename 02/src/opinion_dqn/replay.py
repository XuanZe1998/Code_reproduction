"""优先经验回放缓冲区。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    """一条强化学习经验样本。"""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float, eps: float) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer: list[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        # 新加入的样本先给较高优先级，确保它能尽快参与训练。
        max_priority = float(self.priorities[: len(self.buffer)].max()) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities[self.position] = max(max_priority, 1.0)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float, rng: np.random.Generator) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        # 重要性采样权重用于修正优先采样带来的分布偏差。
        size = len(self.buffer)
        scaled = self.priorities[:size]
        probs = scaled / scaled.sum()
        indices = rng.choice(size, size=min(batch_size, size), replace=False, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        return samples, indices.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        # TD 误差越大，说明样本越“难学”，其后续被抽中的概率也应更高。
        for idx, td_error in zip(indices, td_errors):
            self.priorities[int(idx)] = float((abs(td_error) + self.eps) ** self.alpha)
