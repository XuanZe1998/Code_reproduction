"""
社会信任网络（Social Trust Network, STN）生成模块。

生成带符号权重的有向图，其中：
  - 正权重边表示"信任"关系
  - 负权重边表示"不信任"关系

支持以下合成网络类型：
  - BBV（Barabási-Albert-Vázquez）：无标度网络
  - SBM（Stochastic Block Model）：随机块模型（社区结构）
  - WS（Watts-Strogatz）：小世界网络
  - 真实网络：从边列表文件加载
"""

import numpy as np
import networkx as nx
from config import POSITIVE_WEIGHT_RATIO


def generate_bbv_network(n=5000, m=3, p_rewire=0.1, positive_ratio=POSITIVE_WEIGHT_RATIO):
    """
    生成 BBV（Barabási-Albert-Vázquez）有向带符号权重网络。
    
    BBV 扩展了经典的 BA 模型，增加了边移除机制：
    - 每个时间步，新节点以概率正比于度的方式连接到 m 个已有节点
    - 现有边以概率 p_rewire 被移除
    - 最终转换为有向图并分配带符号权重
    
    Args:
        n: 节点数量，默认 5000
        m: 每步新增边数，默认 3
        p_rewire: 边移除概率，默认 0.1
        positive_ratio: 正权重边的比例，默认 0.8
    
    Returns:
        networkx.DiGraph: 生成的有向带符号权重网络
    """
    # 第一步：先生成无向 BA 图作为基础拓扑
    G = nx.barabasi_albert_graph(n, m, seed=42)
    
    # 第二步：BBV 边移除机制 —— 随机移除部分边
    edges_to_remove = []
    for u, v in list(G.edges()):
        if np.random.random() < p_rewire:
            edges_to_remove.append((u, v))
    for u, v in edges_to_remove:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
    
    # 第三步：转换为有向图（双向随机化处理）
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    for u, v in G.edges():
        # 添加有向边 u -> v
        DG.add_edge(u, v)
        # 以 50% 概率同时添加反向边 v -> u
        if np.random.random() < 0.5:
            DG.add_edge(v, u)
    
    # 第四步：为所有边分配带符号权重
    DG = _assign_signed_weights(DG, positive_ratio)
    
    return DG


def generate_sbm_network(n=5000, n_communities=5, p_intra=0.02, p_inter=0.002,
                         positive_ratio=POSITIVE_WEIGHT_RATIO):
    """
    生成 SBM（Stochastic Block Model，随机块模型）有向带符号权重网络。
    
    SBM 能生成具有明显社区结构的网络：
    - 社区内部连边概率高（p_intra）
    - 社区之间连边概率低（p_inter）
    
    Args:
        n: 节点数量，默认 5000
        n_communities: 社区数量，默认 5
        p_intra: 社区内部连边概率，默认 0.02
        p_inter: 社区之间连边概率，默认 0.002
        positive_ratio: 正权重边的比例，默认 0.8
    
    Returns:
        networkx.DiGraph: 生成的有向带符号权重网络
    """
    # 创建各社区的节点数量（尽量均匀分配）
    sizes = [n // n_communities] * n_communities
    sizes[-1] += n - sum(sizes)  # 将余数分配给最后一个社区
    
    # 构建概率矩阵：对角线为社区内概率，非对角线为社区间概率
    probs = np.full((n_communities, n_communities), p_inter)
    np.fill_diagonal(probs, p_intra)
    
    # 生成有向 SBM 图
    G = nx.stochastic_block_model(sizes, probs.tolist(), directed=True, seed=42)
    DG = nx.DiGraph(G)
    
    # 分配带符号权重
    DG = _assign_signed_weights(DG, positive_ratio)
    return DG


def generate_ws_network(n=5000, k=6, p_rewire=0.1, positive_ratio=POSITIVE_WEIGHT_RATIO):
    """
    生成 WS（Watts-Strogatz）小世界网络（有向带符号权重）。
    
    WS 网络具有高聚类系数和短平均路径长度的特征：
    - 先生成环形最近邻连接
    - 以 p_rewire 概率随机重连每条边
    - 最后转换为有向图并分配权重
    
    Args:
        n: 节点数量，默认 5000
        k: 每个节点连接的最近邻居数（必须为偶数），默认 6
        p_rewire: 边重连概率，默认 0.1
        positive_ratio: 正权重边的比例，默认 0.8
    
    Returns:
        networkx.DiGraph: 生成的有向带符号权重网络
    """
    # 生成无向 WS 小世界图
    G = nx.watts_strogatz_graph(n, k, p_rewire, seed=42)
    
    # 转换为有向图：每条无向边随机保留一个方向或双向
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    for u, v in G.edges():
        r = np.random.random()
        if r < 0.6:
            # 60% 概率：仅保留 u -> v
            DG.add_edge(u, v)
        elif r < 0.8:
            # 20% 概率：仅保留 v -> u
            DG.add_edge(v, u)
        else:
            # 20% 概率：双向保留 u <-> v
            DG.add_edge(u, v)
            DG.add_edge(v, u)
    
    # 分配带符号权重
    DG = _assign_signed_weights(DG, positive_ratio)
    return DG


def load_real_network(filepath, positive_ratio=POSITIVE_WEIGHT_RATIO):
    """
    从边列表文件加载真实世界网络。
    
    文件格式：每行 '源节点 目标节点' 或 '源节点 目标节点 权重'
    
    Args:
        filepath: 边列表文件路径
        positive_ratio: 正权重边的比例（仅当文件无权重时使用）
    
    Returns:
        networkx.DiGraph: 加载的有向带符号权重网络
    """
    # 读取边列表文件
    G = nx.read_edgelist(filepath, create_using=nx.DiGraph(), nodetype=int,
                         data=(('weight', float),))
    
    # 如果边没有权重信息，则随机分配带符号权重
    if not all('weight' in G[u][v] for u, v in G.edges()):
        G = _assign_signed_weights(G, positive_ratio)
    else:
        # 如果有权重信息，确保权重在 [-1, 1] 范围内并带符号
        for u, v in G.edges():
            w = G[u][v]['weight']
            sign = 1 if np.random.random() < positive_ratio else -1
            G[u][v]['weight'] = sign * min(abs(w), 1.0)
    
    return G


def _assign_signed_weights(G, positive_ratio):
    """
    内部函数：为有向图的所有边分配带符号权重。
    
    - positive_ratio 比例的边分配正权重（信任边），取值 (0.1, 1.0]
    - 剩余的边分配负权重（不信任边），取值 [-1.0, -0.1)
    
    Args:
        G: networkx.DiGraph 待处理的有向图
        positive_ratio: 正权重边的比例
    
    Returns:
        networkx.DiGraph: 分配权重后的有向图（原地修改）
    """
    for u, v in G.edges():
        if np.random.random() < positive_ratio:
            # 信任边：权重在 (0.1, 1.0] 之间
            G[u][v]['weight'] = np.random.uniform(0.1, 1.0)
        else:
            # 不信任边：权重在 [-1.0, -0.1) 之间
            G[u][v]['weight'] = np.random.uniform(-1.0, -0.1)
    return G


def get_weight_matrix(G, node_list):
    """
    根据网络图构建权重矩阵 W。
    
    W[i][j] 表示从节点 node_list[i] 到 node_list[j] 的边权重。
    无边连接的节点对权重为 0。
    
    Args:
        G: networkx.DiGraph 有向图
        node_list: 节点索引列表
    
    Returns:
        np.ndarray: N x N 的权重矩阵
    """
    n = len(node_list)
    idx = {node: i for i, node in enumerate(node_list)}  # 节点到矩阵索引的映射
    W = np.zeros((n, n))
    
    # 遍历所有边，填充权重矩阵
    for u, v, data in G.edges(data=True):
        if u in idx and v in idx:
            W[idx[u]][idx[v]] = data.get('weight', 0.0)
    
    return W


def get_network_info(G):
    """
    打印网络的基本统计信息。
    
    包括：节点数、边数、平均度、正权重比例。
    
    Args:
        G: networkx.DiGraph 有向图
    
    Returns:
        dict: 包含网络统计信息的字典
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    # 计算平均度（入度和出度的平均值）
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    avg_deg = (np.mean(in_degrees) + np.mean(out_degrees)) / 2
    
    # 统计正权重边的数量
    pos_count = sum(1 for _, _, d in G.edges(data=True) if d.get('weight', 0) > 0)
    pos_ratio = pos_count / m if m > 0 else 0
    
    print(f"节点数: {n}, 边数: {m}, 平均度: {avg_deg:.2f}, "
          f"正权重比例: {pos_ratio:.2%}")
    
    return {
        'n': n, 'm': m, 'avg_degree': avg_deg,
        'positive_ratio': pos_ratio
    }


if __name__ == '__main__':
    from config import BBV_PARAMS, SBM_PARAMS, WS_PARAMS
    
    # 演示：生成三种类型的网络并输出统计信息
    print("=== BBV 无标度网络 ===")
    G_bbv = generate_bbv_network(**BBV_PARAMS)
    get_network_info(G_bbv)
    
    print("\n=== SBM 社区网络 ===")
    G_sbm = generate_sbm_network(**SBM_PARAMS)
    get_network_info(G_sbm)
    
    print("\n=== WS 小世界网络 ===")
    G_ws = generate_ws_network(**WS_PARAMS)
    get_network_info(G_ws)
