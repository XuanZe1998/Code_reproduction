"""
基线种子选择算法模块。

实现了 DQN 动作空间中使用的五种启发式策略：
1. MaxDegree    - 选择出度最高的节点（最大传播能力）
2. Blocking     - 选择入度最高的节点（最大阻断能力）
3. MixStrategy  - 平衡出度和入度（sigma=0.5）
4. CbC          - 基于社区的中心性（Community-based Centrality）
5. CI           - 集体影响力（Collective Influence，半径=2）

另外还包含贪心算法（Greedy）作为参考基线。
"""

import numpy as np
import networkx as nx
from config import CI_RADIUS, MIX_STRATEGY_SIGMA

# ===================== 动作名称常量 =====================
ACTION_MAXDEGREE = 0       # 最大出度策略
ACTION_BLOCKING = 1        # 最大入度策略
ACTION_MIXSTRATEGY = 2     # 混合度策略
ACTION_CBC = 3             # 社区中心性策略
ACTION_CI = 4              # 集体影响力策略

ACTION_NAMES = ['MaxDegree', 'Blocking', 'MixStrategy', 'CbC', 'CI']


def select_by_action(candidates, action, G, communities):
    """
    根据给定的动作/策略，从候选节点中选出最佳节点。
    
    这是 DQN 选择动作后的执行接口：DQN 输出一个动作编号，
    该函数根据动作编号使用对应的启发式策略选择节点。
    
    Args:
        candidates: 候选节点索引列表/集合
        action: 动作编号（0-4）
        G: networkx.DiGraph 有向图
        communities: 社区映射字典 {节点: 社区ID}
    
    Returns:
        int: 选中的节点索引
    
    Raises:
        ValueError: 当动作编号不在 0-4 范围内时
    """
    candidates = list(candidates)
    if not candidates:
        return None
    
    if action == ACTION_MAXDEGREE:
        return _maxdegree(candidates, G)
    elif action == ACTION_BLOCKING:
        return _blocking(candidates, G)
    elif action == ACTION_MIXSTRATEGY:
        return _mixstrategy(candidates, G)
    elif action == ACTION_CBC:
        return _cbc(candidates, G, communities)
    elif action == ACTION_CI:
        return _ci(candidates, G)
    else:
        raise ValueError(f"未知动作: {action}")


def _maxdegree(candidates, G):
    """选择出度最大的候选节点。出度越高，传播范围越广。"""
    return max(candidates, key=lambda v: G.out_degree(v))


def _blocking(candidates, G):
    """选择入度最大的候选节点。入度越高，越容易被他人影响。"""
    return max(candidates, key=lambda v: G.in_degree(v))


def _mixstrategy(candidates, G):
    """
    选择混合度得分最高的候选节点。
    
    混合度得分 = σ × 出度 + (1-σ) × 入度
    其中 σ = MIX_STRATEGY_SIGMA（默认 0.5）
    
    该策略同时考虑传播能力和被影响程度。
    """
    sigma = MIX_STRATEGY_SIGMA
    return max(candidates, key=lambda v: sigma * G.out_degree(v) + (1 - sigma) * G.in_degree(v))


def _cbc(candidates, G, communities):
    """
    基于社区的中心性（Community-based Centrality, CbC）。
    
    CbC(v) = Σ_{u∈N(v), C_u=C_v} |C_u| + Σ_{u∈N(v), C_u≠C_v} |C_u|
    
    原理：邻居所在社区越大，说明该节点的影响力越能辐射到大规模群体。
    优先选择能够桥接不同社区的节点。
    
    Args:
        candidates: 候选节点列表
        G: 有向图
        communities: 社区映射字典
    """
    if communities is None:
        # 无社区信息时退化到度数策略
        return _maxdegree(candidates, G)
    
    # 预计算各社区的规模
    comm_sizes = {}
    for c in set(communities.values()):
        comm_sizes[c] = sum(1 for v in communities if communities[v] == c)
    
    def cbc_score(v):
        """计算节点 v 的 CbC 得分"""
        score = 0.0
        # 获取所有邻居（前驱 + 后继）
        neighbors = set(G.predecessors(v)) | set(G.successors(v))
        for u in neighbors:
            score += comm_sizes.get(communities.get(u, -1), 0)
        return score
    
    return max(candidates, key=cbc_score)


def _ci(candidates, G, radius=CI_RADIUS):
    """
    集体影响力（Collective Influence, CI）。
    
    CI(v) = (deg(v) - 1) × Σ_{u: dist(v,u)≤radius} (deg(u) - 1)
    
    原理：移除该节点后能影响的节点越多（通过球域内的级联效应），
    其影响力越大。该指标在影响力最大化领域被广泛使用。
    
    Args:
        candidates: 候选节点列表
        G: 有向图
        radius: 计算半径，默认 2
    """
    # 使用无向版本进行距离计算
    UG = G.to_undirected()
    
    def ci_score(v):
        """计算节点 v 的 CI 得分"""
        deg_v = G.degree(v)  # 总度数（入度 + 出度）
        if radius == 0:
            return (deg_v - 1)
        
        # 使用 BFS 找到半径内的所有节点
        visited = {v}
        current_frontier = {v}
        
        for _ in range(radius):
            next_frontier = set()
            for node in current_frontier:
                for neighbor in UG.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            current_frontier = next_frontier
        
        # 对半径内所有节点（不含自身）求 Σ(deg(u) - 1)
        score = 0
        for u in visited:
            if u != v:
                deg_u = G.degree(u)
                score += (deg_u - 1)
        
        return (deg_v - 1) * score
    
    return max(candidates, key=ci_score)


# ===================== 完整基线算法 =====================

def greedy_opinion_maximization(W, G, k, T, communities, initial_opinions=None):
    """
    贪心算法进行观点最大化。
    
    迭代地选择使总观点最大的节点加入种子集合。
    每一步都尝试所有候选节点，选出边际增益最大的。
    
    注意：该算法复杂度极高（O(N × k × T × N)），仅适用于小规模网络。
    
    Args:
        W: 权重矩阵
        G: 有向图
        k: 种子节点数量
        T: 模拟时间步数
        communities: 社区映射
        initial_opinions: 可选初始观点
    
    Returns:
        list: 选中的种子节点索引列表
    """
    N = W.shape[0]
    seed_set = []           # 已选种子集合
    remaining = set(range(N))  # 候选节点集合
    
    from opinion_dynamics import OpinionDynamics
    
    for _ in range(k):
        best_node = None
        best_opinion = -np.inf
        
        # 遍历所有候选节点，找到使总观点最大的那个
        for v in remaining:
            trial_seeds = seed_set + [v]
            total = OpinionDynamics(W, initial_opinions=initial_opinions).run(T, seed_indices=trial_seeds)
            if total > best_opinion:
                best_opinion = total
                best_node = v
        
        seed_set.append(best_node)
        remaining.remove(best_node)
    
    return seed_set


def heuristic_seed_selection(action, W, G, k, communities, initial_opinions=None,
                             potential_seeds=None, lambda_k=None):
    """
    使用纯启发式策略选择 k 个种子节点（作为基线）。
    
    逐次选择：每轮从候选池中用指定策略选出最优节点，
    然后从候选池中移除该节点，重复 k 次。
    
    Args:
        action: 动作编号（0-4）或 'greedy'
        W: 权重矩阵
        G: 有向图
        k: 种子节点数量
        communities: 社区映射
        initial_opinions: 可选初始观点
        potential_seeds: 若提供，仅从此候选集中选择
        lambda_k: 若提供且候选数超过此值，仅使用前 lambda_k 个
    
    Returns:
        list: 选中的种子节点索引列表
    """
    from opinion_dynamics import compute_node_influence
    
    N = W.shape[0]
    
    # 确定候选池
    if potential_seeds is not None:
        candidates = set(potential_seeds)
    else:
        # 默认使用所有节点作为候选
        candidates = set(range(N))
    
    seed_set = []
    
    if action == 'greedy':
        # 贪心策略：大网络上太慢，仅用于小规模测试
        return greedy_opinion_maximization(W, G, k, T=30, communities=communities,
                                           initial_opinions=initial_opinions)
    
    # 启发式策略：逐次选择最优节点
    remaining = set(candidates)
    for _ in range(k):
        if not remaining:
            break
        node = select_by_action(remaining, action, G, communities)
        seed_set.append(node)
        remaining.remove(node)
    
    return seed_set
