"""
基于博弈论的连续观点动力学模型。

核心机制：
  - 节点拥有 [-1, 1] 范围内的连续观点值
  - 节点选择策略：合作（更新观点）或背叛（保持观点不变）
  - 策略更新遵循 Fermi 规则（模仿收益更高的邻居）
  - 观点更新使用基于信任邻居的加权平均

参考文献：Yu et al., "Opinion maximization on social trust networks 
based on game theory and DQN method", Information Sciences 729 (2026) 122879
"""

import numpy as np
from config import BETA_SELECTION, SEED_OPINION, STUBBORNNESS_COOPERATION_PROB


class OpinionDynamics:
    """
    基于博弈论的社会信任网络连续观点动力学模拟器。
    
    该模型结合了演化博弈论与观点传播：
    - 每个节点可以选择"合作"（接受邻居影响）或"背叛"（固执己见）
    - 策略选择受邻居收益差异驱动（Fermi规则）
    - 信任关系（正权重）促进观点趋同，不信任关系（负权重）促进观点分化
    """
    
    def __init__(self, W, initial_opinions=None, initial_cooperation_prob=STUBBORNNESS_COOPERATION_PROB):
        """
        初始化观点动力学模型。
        
        Args:
            W: 权重矩阵（N x N），W[i][j] 表示节点 i 对节点 j 的信任/不信任权重
            initial_opinions: 初始观点向量（长度 N）。若为 None，则在 [-1, 1] 内随机生成
            initial_cooperation_prob: 每个节点初始选择合作策略的概率，默认 0.5
        """
        self.N = W.shape[0]           # 网络节点数
        self.W = W                     # 权重矩阵
        self.beta = BETA_SELECTION     # Fermi 规则的选择强度参数
        
        # 初始化观点：随机分布在 [-1, 1] 区间
        if initial_opinions is not None:
            self.opinions = np.array(initial_opinions, dtype=np.float64)
        else:
            self.opinions = np.random.uniform(-1, 1, self.N)
        
        # 初始化合作策略：z_i = True 表示合作（会更新观点），False 表示背叛（保持观点）
        self.cooperate = np.random.random(self.N) < initial_cooperation_prob
        
        # 预计算邻居信息以提高效率
        self._precompute_neighbors()
    
    def _precompute_neighbors(self):
        """预计算每个节点的邻居集合和权重信息，避免重复计算。"""
        self.out_neighbors = []          # 出邻居列表：i 指向的节点
        self.in_neighbors = []           # 入邻居列表：指向 i 的节点
        self.trusted_out_neighbors = []  # 信任出邻居：权重为正的出邻居
        self.out_weights = []            # 出邻居权重
        self.in_weights = []             # 入邻居权重
        
        for i in range(self.N):
            # 出邻居（i -> j）：权重矩阵第 i 行非零元素
            out_mask = self.W[i] != 0
            out_j = np.where(out_mask)[0]
            self.out_neighbors.append(out_j)
            self.out_weights.append(self.W[i][out_j])
            
            # 入邻居（j -> i）：权重矩阵第 i 列非零元素
            in_mask = self.W[:, i] != 0
            in_j = np.where(in_mask)[0]
            self.in_neighbors.append(in_j)
            self.in_weights.append(self.W[in_j, i])
            
            # 信任出邻居：仅保留正权重的出邻居（用于策略更新中的邻居选择）
            trusted_mask = self.W[i] > 0
            trusted_j = np.where(trusted_mask)[0]
            self.trusted_out_neighbors.append(trusted_j)
    
    def set_seed_nodes(self, seed_indices):
        """
        设置种子节点：将其观点固定为 +1 并设为背叛策略（顽固不化）。
        
        种子节点代表观点传播的源头，它们不会受邻居影响而改变观点。
        
        Args:
            seed_indices: 种子节点的索引列表（0-based）
        """
        for idx in seed_indices:
            self.opinions[idx] = SEED_OPINION    # 观点固定为 +1
            self.cooperate[idx] = False           # 种子节点为背叛策略（顽固）
    
    def compute_local_benefit(self, i, j):
        """
        计算节点 i 面对出邻居 j 时的局部收益 l_ij。
        
        对应论文公式 (3)：
        - 若 omega_ij > 0（信任）：l_ij = (1/2)(1 - |o_i - o_j|)
          → 观点越相似，收益越高（信任促进共识）
        - 若 omega_ij < 0（不信任）：l_ij = (1/2)|o_j - o_i| - 1/2
          → 观点越不同，收益越高（不信任促进对立）
        
        Args:
            i: 节点 i 的索引
            j: 节点 j 的索引（i 的出邻居）
        
        Returns:
            float: 局部收益值
        """
        o_i = self.opinions[i]       # 节点 i 的观点
        o_j = self.opinions[j]       # 节点 j 的观点
        omega_ij = self.W[i][j]      # i 到 j 的权重
        
        if omega_ij > 0:
            # 信任关系：观点越接近收益越高
            return 0.5 * (1 - abs(o_i - o_j))
        else:
            # 不信任关系：观点差异越大收益越高
            return 0.5 * abs(o_j - o_i) - 0.5
    
    def compute_weighted_average_benefit(self):
        """
        计算所有节点的加权平均收益 B_i。
        
        对应论文公式 (4)：B_i = Σ_j(|ω_ij| * l_ij) / Σ_j(|ω_ij|)
        对节点 i 的所有出邻居 j 求和。
        
        收益 B_i 反映了节点 i 在当前观点分布下的整体"满意度"，
        是策略更新的关键指标。
        
        Returns:
            np.ndarray: 长度为 N 的数组，每个元素为对应节点的加权平均收益
        """
        benefits = np.zeros(self.N)
        
        for i in range(self.N):
            out_j = self.out_neighbors[i]
            
            if len(out_j) == 0:
                benefits[i] = 0.0
                continue
            
            total_weight = 0.0    # 权重绝对值之和
            total_benefit = 0.0  # 加权收益之和
            
            for j in out_j:
                omega_ij = self.W[i][j]
                l_ij = self.compute_local_benefit(i, j)
                abs_w = abs(omega_ij)
                total_benefit += abs_w * l_ij
                total_weight += abs_w
            
            if total_weight > 0:
                benefits[i] = total_benefit / total_weight
            else:
                benefits[i] = 0.0
        
        return benefits
    
    def update_strategies(self):
        """
        使用 Fermi 规则更新节点策略（对应论文公式 5-7）。
        
        对每个节点 i：
        1. 以概率 p_ij 从信任出邻居中随机选择一个邻居 j
        2. 以概率 W(z_i ← z_j) = 1 / (1 + exp(-β * (B_j - B_i))) 模仿 j 的策略
           → 若 B_j > B_i，模仿概率 > 0.5（倾向模仿收益更高的邻居）
           → 若 B_j < B_i，模仿概率 < 0.5（倾向保持自身策略）
        3. 种子节点始终为背叛策略（顽固不变）
        
        注意：实际的模仿概率还需乘以选择概率 p_ij，即 R[i←j] = p_ij * W(z_i ← z_j)
        """
        # 先计算所有节点的收益
        benefits = self.compute_weighted_average_benefit()
        
        for i in range(self.N):
            trusted_j = self.trusted_out_neighbors[i]
            
            if len(trusted_j) == 0:
                # 没有信任邻居：随机决定合作或背叛
                self.cooperate[i] = np.random.random() < 0.5
                continue
            
            # 计算每个信任邻居的选择概率 p_ij（正比于信任权重）
            weights_j = self.W[i][trusted_j]
            weights_j = np.maximum(weights_j, 0)  # 仅保留正权重
            total_w = weights_j.sum()
            
            if total_w == 0:
                self.cooperate[i] = np.random.random() < 0.5
                continue
            
            p_j = weights_j / total_w  # 归一化得到概率分布
            
            # 以概率 p_j 随机选择一个邻居 j
            selected_idx = np.random.choice(len(trusted_j), p=p_j)
            j = trusted_j[selected_idx]
            
            # Fermi 规则：模仿邻居 j 策略的概率
            delta_b = benefits[j] - benefits[i]  # 收益差
            W_imitate = 1.0 / (1.0 + np.exp(-self.beta * delta_b))
            
            # 综合模仿概率 R[i←j] = p_ij × W(z_i ← z_j)
            R_imitate = p_j[selected_idx] * W_imitate
            
            if np.random.random() < R_imitate:
                # 模仿邻居 j 的策略
                self.cooperate[i] = self.cooperate[j]
            # 否则保持自身策略（概率 = 1 - R_imitate）
    
    def update_opinions(self):
        """
        更新合作节点的观点（对应论文公式 2）。
        
        对于合作节点 i：
            o_i^{t+1} = Σ_j(|ω_ij| * o_j^t) / Σ_j(|ω_ij|)
        
        背叛节点保持观点不变。
        种子节点始终保持观点 = +1。
        
        观点值会被裁剪到 [-1, 1] 范围内。
        """
        new_opinions = self.opinions.copy()
        
        for i in range(self.N):
            if not self.cooperate[i]:
                # 背叛策略：保持观点不变
                continue
            
            out_j = self.out_neighbors[i]
            
            if len(out_j) == 0:
                continue
            
            total_weight = 0.0       # 权重绝对值之和
            weighted_opinion = 0.0   # 加权观点之和
            
            for j in out_j:
                omega_ij = self.W[i][j]
                abs_w = abs(omega_ij)
                weighted_opinion += abs_w * self.opinions[j]
                total_weight += abs_w
            
            if total_weight > 0:
                # 加权平均更新观点
                new_opinions[i] = weighted_opinion / total_weight
                # 裁剪到合法范围 [-1, 1]
                new_opinions[i] = np.clip(new_opinions[i], -1.0, 1.0)
        
        self.opinions = new_opinions
    
    def step(self):
        """执行一个完整的时间步：先更新策略，再更新观点。"""
        self.update_strategies()   # 第一步：策略更新（Fermi 规则）
        self.update_opinions()     # 第二步：观点更新（加权平均）
    
    def run(self, T, seed_indices=None):
        """
        运行观点动力学模拟 T 个时间步。
        
        Args:
            T: 模拟的时间步数
            seed_indices: 种子节点索引列表。若提供，则在运行前设置种子节点
        
        Returns:
            float: 最终所有节点观点之和（用于评估观点最大化效果）
        """
        if seed_indices is not None:
            self.set_seed_nodes(seed_indices)
        
        for _ in range(T):
            self.step()
        
        return np.sum(self.opinions)
    
    def get_total_opinion(self):
        """返回所有节点观点之和。"""
        return np.sum(self.opinions)


def simulate_opinion_dynamics(W, seed_indices, T, initial_opinions=None):
    """
    便捷函数：模拟观点动力学并返回最终观点总和。
    
    Args:
        W: 权重矩阵（N x N）
        seed_indices: 种子节点索引列表
        T: 模拟时间步数
        initial_opinions: 可选的初始观点向量
    
    Returns:
        float: 最终所有节点观点之和
    """
    model = OpinionDynamics(W, initial_opinions=initial_opinions)
    return model.run(T, seed_indices=seed_indices)


def compute_node_influence(W, opinions):
    """
    计算每个节点的影响力得分 p_i（对应论文算法 1 的公式 8）。
    
    p_i = B_i^0 + Σ_{v∈N_out(i)} B_v^0 * ω_iv + Σ_{u∈N_in(i)} B_i^0 * ω_ui
    
    其中：
    - B_i^0 是节点 i 的初始加权平均收益
    - 第一项是自身收益
    - 第二项是出邻居的收益加权贡献（反映 i 的向外影响力）
    - 第三项是入邻居对 i 的加权影响（反映 i 的被影响力）
    
    该得分用于 T-DQN 第一阶段筛选候选种子节点。
    
    Args:
        W: 权重矩阵（N x N）
        opinions: 初始观点向量
    
    Returns:
        np.ndarray: 长度为 N 的数组，每个元素为对应节点的影响力得分
    """
    N = W.shape[0]
    
    # 计算初始收益
    model = OpinionDynamics(W, initial_opinions=opinions)
    benefits = model.compute_weighted_average_benefit()
    
    influence = np.zeros(N)
    
    for i in range(N):
        # 自身收益
        influence[i] = benefits[i]
        
        # 向外分量：遍历出邻居
        for v in model.out_neighbors[i]:
            influence[i] += benefits[v] * W[i][v]
        
        # 向内分量：遍历入邻居
        for u in model.in_neighbors[i]:
            influence[i] += benefits[i] * W[u][i]
    
    return influence
