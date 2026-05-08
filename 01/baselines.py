"""
基线种子选择算法模块（独立版本）。

实现了五种启发式策略和 Q-Learning 基线：
1. MaxDegree: 选择出度最高的节点（最大传播范围）
2. Blocking: 选择入度最高的节点（最大阻断效果）
3. MixStrategy: 平衡出度和入度的混合策略
4. CbC: 基于社区的中心性（Community-based Centrality）
5. CI: 集体影响力（Collective Influence，半径=2）
6. Q-Learning: 表格型 Q-Learning 基线（用于与 DQN 对比）

该模块与 baseline_algorithms.py 的区别：
- baseline_algorithms.py 侧重于为 DQN 动作空间提供单步选择接口
- 本模块侧重于完整的独立基线算法（批量选种 + 结果评估）
"""

import numpy as np
from opinion_dynamics import OpinionDynamics
from config import CI_RADIUS, MIX_STRATEGY_SIGMA


def select_by_maxdegree(G, candidate_indices, k):
    """
    按出度从候选节点中选择 k 个节点。
    
    出度高的节点能直接影响更多邻居，传播范围广。
    
    Args:
        G: 有向图
        candidate_indices: 候选节点索引列表
        k: 选择数量
    
    Returns:
        list: 按出度降序排列的前 k 个节点
    """
    scores = {}
    for i in candidate_indices:
        scores[i] = G.out_degree(i)
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:k]]


def select_by_blocking(G, candidate_indices, k):
    """
    按入度从候选节点中选择 k 个节点。
    
    入度高的节点被更多节点关注，如果将其设为种子，
    可以"阻断"负面观点的传播路径。
    
    Args:
        G: 有向图
        candidate_indices: 候选节点索引列表
        k: 选择数量
    
    Returns:
        list: 按入度降序排列的前 k 个节点
    """
    scores = {}
    for i in candidate_indices:
        scores[i] = G.in_degree(i)
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:k]]


def select_by_mixstrategy(G, candidate_indices, k, sigma=MIX_STRATEGY_SIGMA):
    """
    按混合度得分从候选节点中选择 k 个节点。
    
    混合度得分 = σ × 出度 + (1-σ) × 入度
    同时兼顾传播能力和抗阻断能力。
    
    Args:
        G: 有向图
        candidate_indices: 候选节点索引列表
        k: 选择数量
        sigma: 出度权重系数，默认 0.5
    
    Returns:
        list: 按混合度降序排列的前 k 个节点
    """
    scores = {}
    for i in candidate_indices:
        od = G.out_degree(i)
        id_ = G.in_degree(i)
        scores[i] = sigma * od + (1 - sigma) * id_
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:k]]


def select_by_cbc(G, candidate_indices, k, communities):
    """
    按社区中心性（CbC）从候选节点中选择 k 个节点。
    
    CbC(v) = Σ_{u∈N(v), C_u=C_v} |C_u| + Σ_{u∈N(v), C_u≠C_v} |C_u|
    
    核心思想：邻居所在社区规模越大，该节点的影响力辐射范围越广。
    能桥接多个社区的节点尤其有价值。
    
    Args:
        G: 有向图
        candidate_indices: 候选节点索引列表
        k: 选择数量
        communities: 社区映射字典 {节点: 社区ID}
    
    Returns:
        list: 按 CbC 得分降序排列的前 k 个节点
    """

    # 预计算各社区的节点数
    community_sizes = {}
    for c in set(communities.values()):
        community_sizes[c] = sum(1 for v in communities if communities[v] == c)
    
    scores = {}
    for v in candidate_indices:
        # 获取所有邻居（前驱 + 后继）
        neighbors = set(G.predecessors(v)) | set(G.successors(v))
        cv = communities.get(v, -1)
        cbc_score = 0.0
        for u in neighbors:
            cu = communities.get(u, -1)
            cbc_score += community_sizes.get(cu, 1)
        scores[v] = cbc_score
    
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:k]]


def select_by_ci(G, candidate_indices, k, radius=CI_RADIUS):
    """
    按集体影响力（CI）从候选节点中选择 k 个节点。
    
    CI(v) = (degree(v) - 1) × Σ_{u: dist(v,u)≤radius} (degree(u) - 1)
    
    基于级联失效模型：移除该节点后，通过球域内的级联效应
    能影响的节点越多，该节点越"关键"。
    
    Args:
        G: 有向图
        candidate_indices: 候选节点索引列表
        k: 选择数量
        radius: 影响半径，默认 2
    
    Returns:
        list: 按 CI 得分降序排列的前 k 个节点
    """
    try:
        # 使用无向版本计算距离
        UG = G.to_undirected()
    except:
        UG = G
    
    node_list = list(G.nodes())
    
    # 预计算每个候选节点的半径内邻居
    hop2_neighbors = {}
    for v in candidate_indices:
        if radius <= 1:
            hop2_neighbors[v] = set(UG.neighbors(v))
        else:
            # BFS 搜索半径内的节点
            visited = {v}
            frontier = {v}
            for _ in range(radius):
                next_frontier = set()
                for node in frontier:
                    for nb in UG.neighbors(node):
                        if nb not in visited:
                            visited.add(nb)
                            next_frontier.add(nb)
                frontier = next_frontier
            hop2_neighbors[v] = visited
    
    # 计算 CI 得分
    scores = {}
    for v in candidate_indices:
        deg_v = G.out_degree(v) + G.in_degree(v)
        ci_val = max(deg_v - 1, 0)
        if radius > 0:
            second_order_sum = 0
            for u in hop2_neighbors[v]:
                if u != v:
                    deg_u = G.out_degree(u) + G.in_degree(u)
                    second_order_sum += max(deg_u - 1, 0)
            ci_val *= second_order_sum
        scores[v] = ci_val
    
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:k]]


class QLearningBaseline:
    """
    表格型 Q-Learning 基线算法，用于种子选择。
    
    简化版本（参考 He 等人的方法）：
    - 状态：已选种子集合（用最近选中节点的索引表示）
    - 动作：5种启发式策略之一
    - 奖励：添加所选节点带来的边际观点增量
    
    与 DQN 的区别：
    - Q-Learning 使用离散状态和表格存储 Q 值
    - DQN 使用连续状态向量和神经网络近似 Q 值
    """
    
    def __init__(self, n_actions=5, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Args:
            n_actions: 动作数量（默认 5）
            alpha: 学习率，默认 0.1
            gamma: 折扣因子，默认 0.9
            epsilon: 探索率，默认 0.1
        """
        self.n_actions = n_actions
        self.alpha = alpha          # 学习率
        self.gamma = gamma          # 折扣因子
        self.epsilon = epsilon      # 探索率
        self.q_table = {}           # Q表：(状态,) -> [各动作的 Q 值]
        self.action_names = ['MaxDegree', 'Blocking', 'MixStrategy', 'CbC', 'CI']
    
    def get_q_values(self, state_key):
        """获取某个状态的 Q 值，若未见过则初始化为全零。"""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]
    
    def select_action(self, state_key):
        """使用 ε-贪心策略选择动作。"""
        if np.random.random() < self.epsilon:
            # 随机探索
            return np.random.randint(self.n_actions)
        else:
            # 选择 Q 值最大的动作
            q_values = self.get_q_values(state_key)
            return np.argmax(q_values)
    
    def update(self, state_key, action, reward, next_state_key, done=False):
        """
        使用 Q-Learning 更新规则更新 Q 值。
        
        Q(s,a) ← Q(s,a) + α × [r + γ × max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state_key: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state_key: 下一状态
            done: 是否为终止状态
        """
        current_q = self.get_q_values(state_key)
        
        if done:
            # 终止状态：无未来奖励
            target = reward
        else:
            # 非终止状态：加上未来最大 Q 值的折扣
            next_q = self.get_q_values(next_state_key)
            target = reward + self.gamma * np.max(next_q)
        
        # Q-Learning 更新
        current_q[action] += self.alpha * (target - current_q[action])


def run_qlearning_baseline(G, W, candidate_indices, k, T, communities=None, 
                           n_episodes=50):
    """
    运行 Q-Learning 训练并选择 k 个种子节点。
    
    训练流程：
    1. 进行 n_episodes 轮训练
    2. 每轮中逐步选择 k 个种子
    3. 记录每轮的最终观点总和
    4. 返回表现最好的一轮种子集合
    
    Args:
        G: 有向图
        W: 权重矩阵
        candidate_indices: 候选节点索引列表
        k: 种子节点数量
        T: 模拟时间步数
        communities: 社区映射（可选）
        n_episodes: 训练轮数，默认 50
    
    Returns:
        tuple: (最佳种子列表, 对应的观点总和)
    """
    from opinion_dynamics import simulate_opinion_dynamics
    
    n_actions = 5
    agent = QLearningBaseline(n_actions=n_actions, alpha=0.1, gamma=0.9, epsilon=0.3)
    
    best_seeds = None
    best_total_opinion = -float('inf')
    
    for episode in range(n_episodes):
        seeds = []                     # 当前轮的种子集合
        remaining = set(candidate_indices)  # 剩余候选节点
        
        for step in range(k):
            # 状态：用最近选中节点的索引表示（第一步为 -1）
            state_key = (seeds[-1] if seeds else -1,)
            
            # 选择动作
            action = agent.select_action(state_key)
            
            # 根据动作在剩余候选中选出最优节点
            remaining_list = list(remaining)
            if action == 0:
                # MaxDegree：按出度排序
                node_scores = [(n, G.out_degree(n)) for n in remaining_list]
            elif action == 1:
                # Blocking：按入度排序
                node_scores = [(n, G.in_degree(n)) for n in remaining_list]
            elif action == 2:
                # MixStrategy：混合度排序
                node_scores = [(n, MIX_STRATEGY_SIGMA * G.out_degree(n) + 
                               (1 - MIX_STRATEGY_SIGMA) * G.in_degree(n)) for n in remaining_list]
            elif action == 3:
                # CbC：社区中心性排序
                if communities:
                    community_sizes = {}
                    for c in set(communities.values()):
                        community_sizes[c] = sum(1 for v in communities if communities[v] == c)
                    node_scores = []
                    for n in remaining_list:
                        neighbors = set(G.predecessors(n)) | set(G.successors(n))
                        cv = communities.get(n, -1)
                        score = sum(community_sizes.get(communities.get(u, -1), 1) for u in neighbors)
                        node_scores.append((n, score))
                else:
                    # 无社区信息时退化到出度
                    node_scores = [(n, G.out_degree(n)) for n in remaining_list]
            else:
                # CI：使用总度数作为近似
                node_scores = [(n, G.out_degree(n) + G.in_degree(n)) for n in remaining_list]
            
            node_scores.sort(key=lambda x: x[1], reverse=True)
            selected_node = node_scores[0][0]  # 得分最高的节点
            
            # 计算奖励：边际观点增益（添加该节点前后的观点差）
            old_opinion = simulate_opinion_dynamics(W, seeds, T) if seeds else 0
            new_seeds = seeds + [selected_node]
            new_opinion = simulate_opinion_dynamics(W, new_seeds, T)
            reward = new_opinion - old_opinion
            
            # 更新 Q 值
            next_state_key = (selected_node,)
            done = (step == k - 1)  # 最后一步为终止状态
            agent.update(state_key, action, reward, next_state_key, done)
            
            seeds.append(selected_node)
            remaining.remove(selected_node)
        
        # 评估当前轮的最终效果
        total_opinion = simulate_opinion_dynamics(W, seeds, T)
        if total_opinion > best_total_opinion:
            best_total_opinion = total_opinion
            best_seeds = seeds.copy()
    
    return best_seeds, best_total_opinion


def run_heuristic_baseline(G, W, strategy_name, candidate_indices, k, T, 
                           communities=None):
    """
    运行启发式基线策略并返回结果。
    
    Args:
        G: 有向图
        W: 权重矩阵
        strategy_name: 策略名称，可选 'MaxDegree'/'Blocking'/'MixStrategy'/'CbC'/'CI'
        candidate_indices: 候选节点索引列表
        k: 种子节点数量
        T: 模拟时间步数
        communities: 社区映射（CbC 策略需要）
    
    Returns:
        tuple: (种子列表, 最终观点总和)
    
    Raises:
        ValueError: 当策略名称未知时
    """
    if strategy_name == 'MaxDegree':
        seeds = select_by_maxdegree(G, candidate_indices, k)
    elif strategy_name == 'Blocking':
        seeds = select_by_blocking(G, candidate_indices, k)
    elif strategy_name == 'MixStrategy':
        seeds = select_by_mixstrategy(G, candidate_indices, k)
    elif strategy_name == 'CbC':
        seeds = select_by_cbc(G, candidate_indices, k, communities)
    elif strategy_name == 'CI':
        seeds = select_by_ci(G, candidate_indices, k)
    else:
        raise ValueError(f"未知策略: {strategy_name}")
    
    # 运行观点动力学模拟获取最终观点总和
    total_opinion = simulate_opinion_dynamics(W, seeds, T)
    return seeds, total_opinion
