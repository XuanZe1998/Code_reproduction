"""实验与模型配置定义。"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class DynamicsConfig:
    """观点动力学参数。"""

    time_steps: int = 30
    selection_strength: float = 10.0
    synchronous: bool = True


@dataclass(slots=True)
class DQNConfig:
    """DQN 与优先经验回放相关参数。"""

    state_dim: int = 25
    action_dim: int = 5
    hidden_dim: int = 256
    learning_rate: float = 0.01
    gamma: float = 0.95
    epsilon_start: float = 0.4
    epsilon_decay: float = 0.9
    epsilon_min: float = 0.1
    tau: float = 0.05
    batch_size: int = 32
    replay_size: int = 10_000
    priority_alpha: float = 0.6
    priority_eps: float = 1e-6
    importance_beta_start: float = 0.4
    importance_beta_increment: float = 1e-4
    target_update_interval: int = 10
    # 可选值示例：
    # - "auto"：按 cuda -> xpu -> cpu 的顺序自动选择
    # - "cuda"：强制使用 CUDA
    # - "xpu"：强制使用 Intel XPU
    # - "cpu"：强制使用 CPU
    device: str = "auto"


@dataclass(slots=True)
class ExperimentConfig:
    """整套实验流程的高层配置。"""

    seed_budget: int = 10
    candidate_multiplier: int = 2
    episodes: int = 50
    random_seed: int = 7
    mix_lambda: float = 0.5
    community_method: str = "greedy_modularity"
    dqn: DQNConfig = field(default_factory=DQNConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
