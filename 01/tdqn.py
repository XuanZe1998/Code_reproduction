"""
两阶段深度 Q 网络（Two-Stage DQN, T-DQN）观点最大化算法。

算法概述：
  第一阶段（Stage 1）：基于节点影响力的启发式规则，筛选 λ×k 个候选种子
  第二阶段（Stage 2）：DQN 从候选种子池中逐步选出最终的 k 个种子节点

核心特性：
  - 5 维动作空间：MaxDegree、Blocking、MixStrategy、CbC、CI
  - 优先经验回放（Prioritized Experience Replay, PER）
  - 目标网络软更新（Soft Update）
  - 基于候选种子特征的状态表示
  - 热身训练（Warmup）：在正式选种前先预训练 DQN

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from config import (
    DQN_BATCH_SIZE, DQN_BUFFER_SIZE, DQN_GAMMA, DQN_LR, DQN_TAU,
    DQN_EPSILON_INIT, DQN_EPSILON_DECAY, DQN_EPSILON_MIN,
    DQN_BETA_INIT, DQN_BETA_INCREMENT, DQN_SOFT_UPDATE_FREQ,
    HIDDEN_DIMS, ACTION_DIM, LAMBDA_POTENTIAL,
    STUBBORNNESS_COOPERATION_PROB, DEFAULT_T
)
from opinion_dynamics import OpinionDynamics, compute_node_influence
from baseline_algorithms import (
    select_by_action, ACTION_NAMES, ACTION_MAXDEGREE,
    ACTION_BLOCKING, ACTION_MIXSTRATEGY, ACTION_CBC, ACTION_CI
)

# 热身训练轮数：在正式选种前用随机观点实例预训练 DQN
WARMUP_EPISODES = 20


# ===================== 优先经验回放缓冲区 =====================

class PrioritizedReplayBuffer:
    """
    优先经验回放（Prioritized Experience Replay, PER）缓冲区。
    
    原理：根据 TD-error 的大小分配采样优先级，TD-error 越大的经验
    被采样的概率越高，因为这些经验对学习更有价值。
    
    使用重要性采样（Importance Sampling）权重修正优先采样带来的偏差。
    """
    
    def __init__(self, capacity, alpha=0.6):
        """
        Args:
            capacity: 缓冲区最大容量
            alpha: 优先级指数，alpha=0 为均匀采样，alpha=1 为完全优先采样
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []       # 经验存储列表
        self.priorities = []   # 每条经验的优先级
        self.pos = 0           # 当前写入位置（循环缓冲区）
    
    def push(self, state, action, reward, next_state, done):
        """
        添加一条新经验，初始优先级设为当前最大值。
        
        Args:
            state: 当前状态向量
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态向量
            done: 是否终止
        """
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            # 缓冲区未满：直接追加
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            # 缓冲区已满：循环覆盖最旧的记录
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
        
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """
        根据优先级采样一批经验，并计算重要性采样权重。
        
        Args:
            batch_size: 采样批量大小
            beta: 重要性采样修正系数，0=无修正，1=完全修正
        
        Returns:
            tuple: (batch 经验列表, 采样索引, IS 权重张量)
        """
        priorities = np.array(self.priorities[:len(self.buffer)])
        # 将优先级转换为采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 按概率采样（不放回）
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # 计算重要性采样权重
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化，防止权重过大
        weights = torch.FloatTensor(weights)
        
        batch = [self.buffer[i] for i in indices]
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        根据新的 TD-error 更新经验的优先级。
        
        TD-error 越大，说明该经验越"出乎意料"，越值得学习。
        
        Args:
            indices: 需要更新的经验索引
            td_errors: 对应的 TD-error 值
        """
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6  # 加小常数防止优先级为零
    
    def __len__(self):
        """返回当前缓冲区中的经验数量。"""
        return len(self.buffer)


# ===================== DQN 神经网络 =====================

class DQNNetwork(nn.Module):
    """
    全连接前馈神经网络，用于 Q 值估计。
    
    输入：状态向量（state_dim 维）
    输出：各动作的 Q 值（ACTION_DIM = 5 维）
    
    网络结构：state_dim → 128 → 64 → 32 → 5
    激活函数：ReLU
    """
    
    def __init__(self, state_dim, hidden_dims=None, action_dim=None):
        """
        Args:
            state_dim: 输入状态向量维度
            hidden_dims: 隐藏层尺寸列表，默认 [128, 64, 32]
            action_dim: 输出动作维度，默认 5
        """
        super().__init__()
        hidden_dims = hidden_dims or HIDDEN_DIMS
        action_dim = action_dim or ACTION_DIM
        
        # 构建全连接层序列
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())  # ReLU 激活
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))  # 输出层
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播：输入状态，输出各动作的 Q 值。"""
        return self.network(x)


# ===================== T-DQN 算法核心 =====================

class TDQN:
    """
    两阶段 DQN 观点最大化算法（论文算法 2）。
    
    第一阶段（Stage 1）：使用基于影响力的启发式规则筛选 λ×k 个候选种子（算法 1）
    第二阶段（Stage 2）：DQN 逐步从候选池中选出 k 个最终种子
    """
    
    def __init__(self, G, W, state_dim, device=None):
        """
        Args:
            G: networkx.DiGraph 有向图
            W: 权重矩阵（N x N）
            state_dim: 状态向量维度
            device: 计算设备（GPU/CPU），自动检测
        """
        self.G = G
        self.W = W
        self.N = W.shape[0]
        self.state_dim = state_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化策略网络和目标网络
        self.policy_net = DQNNetwork(state_dim).to(self.device)    # 策略网络
        self.target_net = DQNNetwork(state_dim).to(self.device)    # 目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 初始时两网络相同
        
        # Adam 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=DQN_LR)
        # 优先经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(DQN_BUFFER_SIZE)
    
    def _get_potential_seeds(self, initial_opinions, k, lam=LAMBDA_POTENTIAL):
        """
        第一阶段（Stage 1）：基于节点影响力筛选候选种子。
        
        算法流程：
        1. 计算每个节点的影响力得分 p_i
        2. 按影响力降序排列，取前 λ×k 个节点作为候选
        
        Args:
            initial_opinions: 初始观点向量
            k: 目标种子数量
            lam: 比例参数，默认 2（即候选数 = 2×k）
        
        Returns:
            list: 候选种子节点索引列表
        """
        influence = compute_node_influence(self.W, initial_opinions)
        top_indices = np.argsort(-influence)[:lam * k]  # 按影响力降序取前 λ×k 个
        return list(top_indices)
    
    def _extract_state(self, candidate_set, selected_seeds, initial_opinions):
        """
        提取 DQN 输入的状态向量。
        
        状态向量由候选种子池的聚合特征构成：
        
        每个候选节点的特征（4维）：
          - 归一化出度
          - 归一化入度
          - 信任出权重之和
          - 信任入权重之和
        
        聚合方式：对每种特征计算均值、标准差、最大值、最小值 → 16维
        额外特征：
          - 已选种子占候选池的比例 → 1维
          - 候选池的平均观点 → 1维
        总计：18维（填充/截断至 state_dim）
        
        Args:
            candidate_set: 当前候选种子集合
            selected_seeds: 已选种子列表
            initial_opinions: 初始观点向量
        
        Returns:
            np.ndarray: 状态向量
        """
        features = []
        
        # 提取每个候选节点的 4 维特征
        for v in candidate_set:
            out_deg = self.G.out_degree(v)
            in_deg = self.G.in_degree(v)
            
            # 信任出权重之和（仅正权重边）
            w_out = sum(self.W[v][j] for j in range(self.N) if self.W[v][j] > 0)
            # 信任入权重之和（仅正权重边）
            w_in = sum(self.W[j][v] for j in range(self.N) if self.W[j][v] > 0)
            
            # 归一化（除以最大可能度数 N-1）
            max_deg = max(self.N - 1, 1)
            features.append([
                out_deg / max_deg,
                in_deg / max_deg,
                w_out / max_deg,
                w_in / max_deg,
            ])
        
        if not features:
            return np.zeros(self.state_dim)
        
        features = np.array(features)
        
        # 聚合：每种特征计算 均值、标准差、最大值、最小值
        agg = []
        for col in range(features.shape[1]):
            agg.append(np.mean(features[:, col]))
            agg.append(np.std(features[:, col]) + 1e-8)  # 加小常数防止为零
            agg.append(np.max(features[:, col]))
            agg.append(np.min(features[:, col]))
        
        # 已选种子占候选池的比例
        agg.append(len(selected_seeds) / max(len(candidate_set) + len(selected_seeds), 1))
        
        # 候选池的平均观点值
        candidate_opinions = [initial_opinions[v] for v in candidate_set if v < len(initial_opinions)]
        agg.append(np.mean(candidate_opinions) if candidate_opinions else 0.0)
        
        state = np.array(agg, dtype=np.float32)
        
        # 填充或截断到目标维度
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]
        
        return state
    
    def _select_action(self, state, epsilon):
        """
        使用 ε-贪心策略选择动作。
        
        以概率 ε 随机探索，以概率 1-ε 选择 Q 值最大的动作。
        
        Args:
            state: 当前状态向量
            epsilon: 探索率
        
        Returns:
            int: 选择的动作编号（0-4）
        """
        if np.random.random() < epsilon:
            # 随机探索
            return np.random.randint(ACTION_DIM)
        else:
            # 利用：选择 Q 值最大的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def _run_dqn_episode(self, k, T, initial_opinions, communities, epsilon, beta_dqn):
        """
        运行一个完整的 DQN 种子选择回合（共 k 步）。
        
        每一步：
        1. 获取候选种子池（排除已选种子）
        2. 提取状态向量
        3. DQN 选择动作
        4. 用对应启发式策略选出节点
        5. 模拟观点动力学计算奖励
        6. 将经验存入回放缓冲区
        7. 训练网络并更新目标网络
        
        Args:
            k: 种子数量
            T: 模拟时间步数
            initial_opinions: 初始观点
            communities: 社区映射
            epsilon: 探索率
            beta_dqn: 重要性采样系数
        
        Returns:
            tuple: (种子列表, 总奖励)
        """
        potential_seeds = self._get_potential_seeds(initial_opinions, k)
        seed_set = []              # 已选种子列表
        total_reward = 0.0         # 累计奖励
        prev_total = 0.0           # 上一步的观点总和（用于计算边际增益）
        
        for m in range(k):
            # 候选池：排除已选种子
            candidates = [v for v in potential_seeds if v not in seed_set]
            if not candidates:
                break
            
            # 提取当前状态
            state = self._extract_state(candidates, seed_set, initial_opinions)
            
            # DQN 选择动作
            action = self._select_action(state, epsilon)
            
            # 用对应启发式策略从候选中选出最优节点
            v_star = select_by_action(candidates, action, self.G, communities)
            new_seed_set = seed_set + [v_star]
            
            # 模拟观点动力学，计算总观点
            total = OpinionDynamics(self.W, initial_opinions=initial_opinions).run(T, seed_indices=new_seed_set)
            reward = total - prev_total   # 奖励 = 边际观点增益
            total_reward += reward
            
            # 构建下一状态
            next_candidates = [v for v in potential_seeds if v not in new_seed_set]
            done = (m == k - 1)  # 最后一步为终止状态
            next_state = self._extract_state(next_candidates, new_seed_set, initial_opinions) if next_candidates else np.zeros(self.state_dim)
            
            # 存入经验回放缓冲区
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # 缓冲区足够大时进行网络训练
            if len(self.replay_buffer) >= DQN_BATCH_SIZE:
                self._update_network(beta_dqn)
            
            # 定期软更新目标网络
            if (m + 1) % DQN_SOFT_UPDATE_FREQ == 0:
                self._soft_update_target()
            
            seed_set.append(v_star)
            prev_total = total
        
        return seed_set, total_reward
    
    def _warmup(self, k, T, communities, n_episodes=WARMUP_EPISODES):
        """
        热身训练：在正式选种前，使用随机观点实例预训练 DQN。
        
        目的：让 DQN 在面对真实问题时已有一定的策略基础，
        避免初始阶段完全随机选择导致的低效。
        
        训练过程中 ε 从 1.0 逐渐衰减，探索率随训练逐步降低。
        
        Args:
            k: 种子数量
            T: 模拟时间步数
            communities: 社区映射
            n_episodes: 热身轮数，默认 20
        """
        print(f"    [热身] 预训练 DQN {n_episodes} 轮...", end=" ", flush=True)
        
        beta_dqn = DQN_BETA_INIT  # 重要性采样系数初始值
        
        for ep in range(n_episodes):
            # 每轮使用随机初始观点
            initial_opinions = np.random.uniform(-1, 1, self.N)
            
            # ε 从 1.0 线性衰减到 DQN_EPSILON_INIT
            epsilon = max(1.0 - ep / n_episodes * 0.5, DQN_EPSILON_INIT)
            
            # 运行一完整回合
            self._run_dqn_episode(k, T, initial_opinions, communities, epsilon, beta_dqn)
            
            # 逐步增大重要性采样系数（趋近 1.0 时偏差修正更完全）
            beta_dqn = min(1.0, beta_dqn + DQN_BETA_INCREMENT * k)
        
        print("完成。")
    
    def select_seeds(self, k, T, initial_opinions=None, communities=None):
        """
        T-DQN 主函数：两阶段种子选择（论文算法 2）。
        
        完整流程：
        1. 热身训练：在随机观点实例上预训练 DQN
        2. 正式选种：使用训练好的 DQN + 低探索率选择最终种子
        
        Args:
            k: 种子节点数量
            T: 模拟时间步数
            initial_opinions: 初始观点向量（可选）
            communities: 社区映射（可选）
        
        Returns:
            list: 最终选中的 k 个种子节点索引
        """
        if initial_opinions is None:
            initial_opinions = np.random.uniform(-1, 1, self.N)
        
        # 第一步：热身训练
        self._warmup(k, T, communities, WARMUP_EPISODES)
        
        # 第二步：正式选种（使用低探索率）
        potential_seeds = self._get_potential_seeds(initial_opinions, k)
        seed_set = []
        epsilon = DQN_EPSILON_MIN  # 低探索率，更倾向于利用已学策略
        beta_dqn = min(1.0, DQN_BETA_INIT + DQN_BETA_INCREMENT * WARMUP_EPISODES * k)
        
        for m in range(k):
            candidates = [v for v in potential_seeds if v not in seed_set]
            if not candidates:
                break
            
            state = self._extract_state(candidates, seed_set, initial_opinions)
            action = self._select_action(state, epsilon)
            
            v_star = select_by_action(candidates, action, self.G, communities)
            new_seed_set = seed_set + [v_star]
            
            # 计算添加新种子后的总观点
            total = OpinionDynamics(self.W, initial_opinions=initial_opinions).run(T, seed_indices=new_seed_set)
            
            # 计算奖励 = 边际观点增益
            next_candidates = [v for v in potential_seeds if v not in new_seed_set]
            done = (m == k - 1)
            reward = total - (OpinionDynamics(self.W, initial_opinions=initial_opinions).run(T, seed_indices=seed_set) if seed_set else 0)
            next_state = self._extract_state(next_candidates, new_seed_set, initial_opinions) if next_candidates else np.zeros(self.state_dim)
            
            # 存入经验并继续训练
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            if len(self.replay_buffer) >= DQN_BATCH_SIZE:
                self._update_network(beta_dqn)
            
            if (m + 1) % DQN_SOFT_UPDATE_FREQ == 0:
                self._soft_update_target()
            
            seed_set.append(v_star)
        
        return seed_set
    
    def _update_network(self, beta_dqn):
        """
        使用优先经验回放更新策略网络。
        
        训练流程：
        1. 从缓冲区中按优先级采样一批经验
        2. 计算当前 Q 值（策略网络）和目标 Q 值（目标网络）
        3. 使用 Double DQN：策略网络选动作，目标网络评估 Q 值
        4. 计算带 IS 权重的 MSE 损失
        5. 反向传播更新网络参数
        6. 根据新 TD-error 更新优先级
        
        Args:
            beta_dqn: 重要性采样修正系数
        """
        # 从缓冲区采样
        batch, indices, weights = self.replay_buffer.sample(DQN_BATCH_SIZE, beta_dqn)
        
        # 解包并转换为张量
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions = torch.LongTensor(np.array([b[1] for b in batch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1).to(self.device)
        
        # 计算当前 Q 值：Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            # Double DQN：用策略网络选择动作，用目标网络评估
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            # 目标 Q 值：r + γ × Q_target(s', a*)（非终止状态才加未来值）
            target_q = rewards + (1 - dones) * DQN_GAMMA * next_q_values
        
        # 计算 TD-error 用于更新优先级
        td_errors = (q_values - target_q).detach().cpu().numpy().flatten()
        
        # 计算带 IS 权重的损失
        loss = (weights.to(self.device) * (q_values - target_q) ** 2).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新缓冲区中经验的优先级
        self.replay_buffer.update_priorities(indices, td_errors)
    
    def _soft_update_target(self):
        """
        目标网络软更新：θ_target = τ × θ_policy + (1-τ) × θ_target
        
        软更新使目标网络平滑地跟踪策略网络，提高训练稳定性。
        τ 越大，目标网络更新越快。
        """
        for target_param, policy_param in zip(self.target_net.parameters(),
                                               self.policy_net.parameters()):
            target_param.data.copy_(
                DQN_TAU * policy_param.data + (1 - DQN_TAU) * target_param.data
            )


# ===================== Q-Learning 基线 =====================

class QLearning:
    """
    表格型 Q-Learning 基线算法。
    
    与 DQN 的区别：
    - 使用离散状态索引而非连续状态向量
    - 使用表格存储 Q 值而非神经网络
    - 状态空间被离散化为固定数量的区间
    
    用于对比 DQN 的优势：展示深度学习方法比传统表格方法的优越性。
    """
    
    def __init__(self, G, W, n_states=100):
        """
        Args:
            G: 有向图
            W: 权重矩阵
            n_states: 离散状态数，默认 100
        """
        self.G = G
        self.W = W
        self.N = W.shape[0]
        self.n_states = n_states
        self.q_table = np.zeros((n_states, ACTION_DIM))  # Q 表
        self.lr = 0.1          # 学习率
        self.gamma = DQN_GAMMA # 折扣因子
        self.epsilon = 0.3     # 探索率
    
    def _state_to_idx(self, n_selected, k):
        """将连续状态（已选种子数/k）映射到离散索引。"""
        idx = int(min(n_selected / max(k, 1) * (self.n_states - 1), self.n_states - 1))
        return idx
    
    def select_seeds(self, k, T, initial_opinions=None, communities=None):
        """
        使用 Q-Learning 选择 k 个种子节点。
        
        Args:
            k: 种子数量
            T: 模拟时间步数
            initial_opinions: 初始观点（可选）
            communities: 社区映射（可选）
        
        Returns:
            list: 选中的种子节点索引
        """
        if initial_opinions is None:
            initial_opinions = np.random.uniform(-1, 1, self.N)
        
        # 使用影响力筛选候选种子
        potential_seeds = compute_potential_seeds(self.W, initial_opinions, k)
        seed_set = []
        prev_total = 0.0
        
        for m in range(k):
            candidates = [v for v in potential_seeds if v not in seed_set]
            if not candidates:
                break
            
            # 将当前进度映射为离散状态
            state_idx = self._state_to_idx(m, k)
            
            # ε-贪心选择动作
            if np.random.random() < self.epsilon:
                action = np.random.randint(ACTION_DIM)
            else:
                action = np.argmax(self.q_table[state_idx])
            
            # 用对应策略选出节点
            v_star = select_by_action(candidates, action, self.G, communities)
            
            # 模拟观点动力学
            new_seed_set = seed_set + [v_star]
            total = OpinionDynamics(self.W, initial_opinions=initial_opinions).run(T, seed_indices=new_seed_set)
            
            # Q-Learning 更新
            if m > 0:
                reward = total - prev_total
                next_idx = self._state_to_idx(m + 1, k)
                max_next_q = np.max(self.q_table[next_idx])
                self.q_table[state_idx, action] += self.lr * (
                    reward + self.gamma * max_next_q - self.q_table[state_idx, action]
                )
            
            prev_total = total
            seed_set.append(v_star)
        
        return seed_set


def compute_potential_seeds(W, initial_opinions, k, lam=LAMBDA_POTENTIAL):
    """
    辅助函数：基于节点影响力计算候选种子列表。
    
    Args:
        W: 权重矩阵
        initial_opinions: 初始观点向量
        k: 目标种子数量
        lam: 比例参数，候选数 = lam × k
    
    Returns:
        list: 候选种子节点索引列表
    """
    influence = compute_node_influence(W, initial_opinions)
    top_indices = np.argsort(-influence)[:lam * k]  # 按影响力降序
    return list(top_indices)
