"""
T-DQN 观点最大化实验框架。

支持多维度对比实验：
- 多种网络拓扑：BBV（无标度）、SBM（社区结构）、WS（小世界）、真实网络
- 多种种子数量（k 值）
- 多种时间步数（T 值）
- 多种初始观点设置（随机 / 回音室效应）

输出结果表格和可视化图表。
"""

import numpy as np
import time
import os
import json

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免弹窗
import matplotlib.pyplot as plt

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

from network_generator import (
    generate_bbv_network, generate_sbm_network, generate_ws_network,
    get_weight_matrix, get_network_info
)
from opinion_dynamics import OpinionDynamics, simulate_opinion_dynamics
from baseline_algorithms import (
    heuristic_seed_selection, ACTION_MAXDEGREE, ACTION_BLOCKING,
    ACTION_MIXSTRATEGY, ACTION_CBC, ACTION_CI, greedy_opinion_maximization,
    ACTION_NAMES
)
from tdqn import TDQN, QLearning, compute_potential_seeds
from config import (
    BBV_PARAMS, SBM_PARAMS, WS_PARAMS,
    DEFAULT_K, DEFAULT_T, NUM_TRIALS, K_VALUES, T_VALUES,
    LAMBDA_POTENTIAL, STUBBORNNESS_COOPERATION_PROB,
    HIDDEN_DIMS, ACTION_DIM
)

# 尝试导入社区检测库
try:
    import community.community_louvain as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False


def detect_communities(G):
    """
    使用 Louvain 算法检测网络社区结构。
    
    注意事项：
    - Louvain 仅支持无向图
    - Louvain 不支持负权重边，需要先移除
    
    Args:
        G: networkx.DiGraph 有向图
    
    Returns:
        dict: 社区划分结果 {节点ID: 社区ID}，若库不可用则返回 None
    """
    if not HAS_COMMUNITY:
        return None
    
    # 转换为无向图
    UG = G.to_undirected()
    # 移除负权重边（Louvain 不支持负权重）
    neg_edges = [(u, v) for u, v, d in UG.edges(data=True) if d.get('weight', 1) < 0]
    UG.remove_edges_from(neg_edges)
    partition = community_louvain.best_partition(UG)
    return partition


def compute_state_dim(G, W, k, lam=LAMBDA_POTENTIAL):
    """
    计算 DQN 的状态向量维度。
    
    状态构成：
    - 4 个特征/节点 × 4 种聚合统计（均值、标准差、最大值、最小值）= 16
    - 1 个种子选择比例 = 1
    总计：17 维
    
    Args:
        G: 有向图
        W: 权重矩阵
        k: 种子数量
        lam: 候选比例参数
    
    Returns:
        int: 状态维度
    """
    return 4 * 4 + 1  # 17 维


def run_single_algorithm(algo_name, G, W, k, T, initial_opinions, communities, **kwargs):
    """
    运行单个算法，返回结果。
    
    Args:
        algo_name: 算法名称（'T-DQN'/'Q-Learning'/五种启发式策略名）
        G: 有向图
        W: 权重矩阵
        k: 种子数量
        T: 模拟时间步数
        initial_opinions: 初始观点向量
        communities: 社区映射
    
    Returns:
        tuple: (种子列表, 最终观点总和, 运行时间)
    """
    start = time.time()
    
    if algo_name == 'T-DQN':
        # T-DQN 算法
        state_dim = compute_state_dim(G, W, k)
        agent = TDQN(G, W, state_dim)
        seeds = agent.select_seeds(k, T, initial_opinions, communities)
    elif algo_name == 'Q-Learning':
        # Q-Learning 基线
        agent = QLearning(G, W)
        seeds = agent.select_seeds(k, T, initial_opinions, communities)
    elif algo_name in ACTION_NAMES:
        # 五种启发式策略
        action = ACTION_NAMES.index(algo_name)
        potential_seeds = compute_potential_seeds(W, initial_opinions, k)
        seeds = heuristic_seed_selection(
            action, W, G, k, communities, initial_opinions,
            potential_seeds=potential_seeds
        )
    else:
        raise ValueError(f"未知算法: {algo_name}")
    
    # 模拟观点动力学，获取最终观点总和
    total = OpinionDynamics(W, initial_opinions=initial_opinions).run(T, seed_indices=seeds)
    elapsed = time.time() - start
    
    return seeds, total, elapsed


def run_experiment(network_name, G, k_values=None, T=None, num_trials=NUM_TRIALS,
                   algo_names=None):
    """
    在给定网络上运行完整对比实验。
    
    对每个 k 值，运行多次独立实验（随机不同初始观点），
    记录各算法的平均观点总和和标准差。
    
    Args:
        network_name: 网络名称（用于显示）
        G: networkx.DiGraph 有向图
        k_values: 测试的 k 值列表
        T: 固定的时间步数
        num_trials: 独立实验次数
        algo_names: 参与对比的算法名称列表
    
    Returns:
        dict: 结果字典 {算法名: {k: {'mean': 平均值, 'std': 标准差, 'time': 平均时间, 'values': 原始值列表}}}
    """
    if k_values is None:
        k_values = K_VALUES
    if T is None:
        T = DEFAULT_T
    if algo_names is None:
        algo_names = ACTION_NAMES + ['Q-Learning', 'T-DQN']  # 全部 7 种算法
    
    print(f"\n{'='*60}")
    print(f"网络: {network_name} (N={G.number_of_nodes()}, M={G.number_of_edges()})")
    print(f"k 值: {k_values}, T={T}, 实验次数={num_trials}")
    print(f"算法: {algo_names}")
    print(f"{'='*60}")
    
    # 预计算：权重矩阵和社区划分
    node_list = list(G.nodes())
    W = get_weight_matrix(G, node_list)
    communities = detect_communities(G)
    
    # 初始化结果存储
    results = {algo: {k: {'mean': 0, 'std': 0, 'time': 0, 'values': []} for k in k_values}
               for algo in algo_names}
    
    for k in k_values:
        print(f"\n  k = {k}")
        
        for trial in range(num_trials):
            print(f"    实验 {trial + 1}/{num_trials}...", end=" ", flush=True)
            
            # 生成随机初始观点（使用固定种子保证可复现性）
            np.random.seed(42 + trial)
            initial_opinions = np.random.uniform(-1, 1, W.shape[0])
            
            for algo_name in algo_names:
                np.random.seed(42 + trial)  # 重置随机种子，保证公平对比
                seeds, total, elapsed = run_single_algorithm(
                    algo_name, G, W, k, T, initial_opinions, communities
                )
                results[algo_name][k]['values'].append(total)
                results[algo_name][k]['time'] += elapsed
                
                print(f"{algo_name}:{total:.1f}", end=" ", flush=True)
            
            print()
        
        # 统计：计算均值和标准差
        for algo_name in algo_names:
            vals = results[algo_name][k]['values']
            results[algo_name][k]['mean'] = np.mean(vals)
            results[algo_name][k]['std'] = np.std(vals)
            results[algo_name][k]['time'] /= num_trials
    
    # 打印汇总表格
    print(f"\n  结果汇总（平均观点总和 ± 标准差）:")
    print(f"  {'算法':<15}", end="")
    for k in k_values:
        print(f"  {'k='+str(k):<14}", end="")
    print()
    print(f"  {'-'*15}", end="")
    for _ in k_values:
        print(f"  {'-'*14}", end="")
    print()
    
    for algo_name in algo_names:
        print(f"  {algo_name:<15}", end="")
        for k in k_values:
            m = results[algo_name][k]['mean']
            s = results[algo_name][k]['std']
            print(f"  {m:>7.1f}±{s:<5.1f}", end="")
        print()
    
    return results


def run_time_step_experiment(network_name, G, k, T_values=None, num_trials=5,
                             algo_names=None):
    """
    时间步变化实验：固定 k，改变 T。
    
    对应论文图 4：观察不同算法在不同演化时间下的观点传播效果。
    
    Args:
        network_name: 网络名称
        G: 有向图
        k: 固定的种子数量
        T_values: 测试的 T 值列表
        num_trials: 实验次数
        algo_names: 算法名称列表
    
    Returns:
        dict: 结果字典 {算法名: {T: {'mean': 平均值, 'std': 标准差, 'values': 列表}}}
    """
    if T_values is None:
        T_values = T_VALUES
    if algo_names is None:
        algo_names = ACTION_NAMES + ['Q-Learning', 'T-DQN']
    
    print(f"\n{'='*60}")
    print(f"时间步实验: {network_name}, k={k}")
    print(f"T 值: {T_values}")
    print(f"{'='*60}")
    
    # 预计算
    node_list = list(G.nodes())
    W = get_weight_matrix(G, node_list)
    communities = detect_communities(G)
    
    results = {algo: {T: {'mean': 0, 'std': 0, 'values': []} for T in T_values}
               for algo in algo_names}
    
    for T in T_values:
        print(f"\n  T = {T}")
        
        for trial in range(num_trials):
            np.random.seed(42 + trial)
            initial_opinions = np.random.uniform(-1, 1, W.shape[0])
            
            for algo_name in algo_names:
                np.random.seed(42 + trial)
                seeds, total, elapsed = run_single_algorithm(
                    algo_name, G, W, k, T, initial_opinions, communities
                )
                results[algo_name][T]['values'].append(total)
        
        # 计算统计量
        for algo_name in algo_names:
            vals = results[algo_name][T]['values']
            results[algo_name][T]['mean'] = np.mean(vals)
            results[algo_name][T]['std'] = np.std(vals)
        
        print(f"    ", end="")
        for algo_name in algo_names:
            print(f"{algo_name}:{results[algo_name][T]['mean']:.1f}", end=" ")
        print()
    
    # 打印汇总
    print(f"\n  结果（平均观点总和）:")
    print(f"  {'算法':<15}", end="")
    for T in T_values:
        print(f"  {'T='+str(T):<14}", end="")
    print()
    
    for algo_name in algo_names:
        print(f"  {algo_name:<15}", end="")
        for T in T_values:
            m = results[algo_name][T]['mean']
            print(f"  {m:>12.1f}", end="")
        print()
    
    return results


def run_small_scale_test():
    """
    小规模快速测试：验证所有组件是否正常工作。
    
    使用 200 节点的小型网络和简化参数，快速运行一次完整的对比实验。
    适用于开发调试和功能验证。
    
    Returns:
        dict: 各算法的测试结果
    """
    import networkx as nx
    from config import POSITIVE_WEIGHT_RATIO
    
    print("=" * 60)
    print("小规模测试（200 个节点）")
    print("=" * 60)
    
    # 生成一个小型 WS 网络
    np.random.seed(42)
    G = nx.watts_strogatz_graph(200, 6, 0.1, seed=42)
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    for u, v in G.edges():
        r = np.random.random()
        if r < 0.5:
            DG.add_edge(u, v)
        else:
            DG.add_edge(v, u)
    
    # 分配带符号权重
    for u, v in DG.edges():
        if np.random.random() < POSITIVE_WEIGHT_RATIO:
            DG[u][v]['weight'] = np.random.uniform(0.1, 1.0)
        else:
            DG[u][v]['weight'] = np.random.uniform(-1.0, -0.1)
    
    # 构建权重矩阵和社区划分
    node_list = list(DG.nodes())
    W = get_weight_matrix(DG, node_list)
    communities = detect_communities(DG)
    
    # 测试参数
    k = 5     # 少量种子
    T = 10    # 少量时间步
    np.random.seed(42)
    initial_opinions = np.random.uniform(-1, 1, 200)
    
    results = {}
    
    # 运行所有 7 种算法
    for algo_name in ACTION_NAMES + ['Q-Learning', 'T-DQN']:
        np.random.seed(42)
        print(f"\n  运行 {algo_name}...", end=" ", flush=True)
        start = time.time()
        seeds, total, elapsed = run_single_algorithm(
            algo_name, DG, W, k, T, initial_opinions, communities
        )
        results[algo_name] = {'total': total, 'time': elapsed, 'seeds': seeds}
        print(f"观点总和 = {total:.2f}, 耗时 = {elapsed:.2f}s")
    
    # 按观点总和排名
    print("\n" + "=" * 60)
    print("排名（按观点总和降序）:")
    print("=" * 60)
    ranking = sorted(results.items(), key=lambda x: -x[1]['total'])
    for rank, (name, res) in enumerate(ranking, 1):
        print(f"  {rank}. {name:<15}: {res['total']:.2f}  ({res['time']:.2f}s)")
    
    # ---- 绘制柱状图：横坐标=各算法（种子集合），纵坐标=意见总和 ----
    _plot_seed_opinion_bar(results)
    
    # ---- 绘制折线图：横坐标=种子节点数 k (0~100 步长20)，纵坐标=意见总和 ----
    _plot_k_opinion_line(DG, W, communities, initial_opinions, T)
    
    return results


def _plot_seed_opinion_bar(results, save_path=None):
    """
    绘制柱状图：横坐标为各算法对应的种子集合，纵坐标为意见总和。
    
    按意见总和从高到低排序，每个柱子注释具体数值。
    
    Args:
        results: dict，格式为 {算法名: {'total': float, ...}}
        save_path: 图片保存路径，默认保存到 results/seed_opinion_bar.png
    """
    # 按意见总和降序排序
    sorted_items = sorted(results.items(), key=lambda x: x[1]['total'], reverse=True)
    algo_names = [item[0] for item in sorted_items]
    totals = [item[1]['total'] for item in sorted_items]

    # 颜色：T-DQN 用深蓝突出，Q-Learning 用橙色，其余启发式用灰色系
    colors = []
    for name in algo_names:
        if name == 'T-DQN':
            colors.append('#1f77b4')
        elif name == 'Q-Learning':
            colors.append('#ff7f0e')
        else:
            colors.append('#aec7e8')

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(algo_names))
    bars = ax.bar(x_pos, totals, color=colors, width=0.6, edgecolor='white', linewidth=1.2)

    # 在每个柱子顶部标注数值
    for bar, val in zip(bars, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01 * (max(totals) - min(totals)),
            f'{val:.2f}',
            ha='center', va='bottom', fontsize=10
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(algo_names, fontsize=11, rotation=15, ha='right')
    ax.set_xlabel('种子集合（算法）', fontsize=13)
    ax.set_ylabel('最终意见总和', fontsize=13)
    ax.set_title('各算法种子集合对应的意见总和对比', fontsize=14)
    ax.grid(axis='y', alpha=0.35, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 图例说明
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='T-DQN'),
        Patch(facecolor='#ff7f0e', label='Q-Learning'),
        Patch(facecolor='#aec7e8', label='启发式基线'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')

    plt.tight_layout()

    # 保存图片
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = os.path.join('results', 'seed_opinion_bar.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [图表] 已保存: {save_path}")


def _plot_k_opinion_line(G, W, communities, initial_opinions, T,
                         k_values=None, save_path=None):
    """
    折线图：横坐标为种子节点数 k（0~100，步长20），纵坐标为意见总和。
    
    对比 7 种算法：MaxDegree / Blocking / MixStrategy / CbC / CI / Q-Learning / T-DQN。
    k=0 时意见总和为无任何种子干预的基准值。
    
    Args:
        G: 有向图
        W: 权重矩阵
        communities: 社区映射
        initial_opinions: 初始观点向量
        T: 时间步数
        k_values: 测试的 k 列表，默认 [0, 20, 40, 60, 80, 100]
        save_path: 图片保存路径，默认 results/k_opinion_line.png
    """
    if k_values is None:
        k_values = list(range(0, 101, 20))  # [0, 20, 40, 60, 80, 100]

    algo_names = ACTION_NAMES + ['Q-Learning', 'T-DQN']

    # 每种算法的颜色与标记
    style_map = {
        'MaxDegree':   {'color': '#e377c2', 'marker': 'o',  'ls': '-'},
        'Blocking':    {'color': '#8c564b', 'marker': 's',  'ls': '--'},
        'MixStrategy': {'color': '#2ca02c', 'marker': '^',  'ls': '-.'},
        'CbC':         {'color': '#9467bd', 'marker': 'D',  'ls': ':'},
        'CI':          {'color': '#17becf', 'marker': 'v',  'ls': '-'},
        'Q-Learning':  {'color': '#ff7f0e', 'marker': 'p',  'ls': '--'},
        'T-DQN':       {'color': '#1f77b4', 'marker': '*',  'ls': '-'},
    }

    # ---- 计算各算法在不同 k 值下的意见总和 ----
    print(f"\n{'='*60}")
    print(f"折线图实验：k 从 0 到 100（步长 20），T={T}")
    print(f"{'='*60}")

    # k=0 时的基准（无种子干预）
    baseline_total = OpinionDynamics(W, initial_opinions=initial_opinions).run(T, seed_indices=[])

    data = {algo: [] for algo in algo_names}
    # k 不能超过节点数-1，防止候选池为空
    max_k = G.number_of_nodes() - 1

    for k in k_values:
        print(f"  k = {k} ...", end=" ", flush=True)
        if k == 0:
            # k=0：所有算法都等于基准
            for algo in algo_names:
                data[algo].append(baseline_total)
        else:
            actual_k = min(k, max_k)
            for algo in algo_names:
                np.random.seed(42)
                seeds, total, _ = run_single_algorithm(
                    algo, G, W, actual_k, T, initial_opinions, communities
                )
                data[algo].append(total)
                print(f"{algo}:{total:.1f}", end=" ", flush=True)
        print()

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(11, 6))

    for algo in algo_names:
        st = style_map[algo]
        ax.plot(
            k_values, data[algo],
            color=st['color'],
            marker=st['marker'],
            linestyle=st['ls'],
            linewidth=2,
            markersize=7,
            label=algo
        )

    ax.set_xticks(k_values)
    ax.set_xlabel('种子节点数量 (k)', fontsize=13)
    ax.set_ylabel('最终意见总和', fontsize=13)
    ax.set_title('各对比算法意见总和随种子节点数变化折线图', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = os.path.join('results', 'k_opinion_line.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [图表] 折线图已保存: {save_path}")


if __name__ == '__main__':
    # 默认运行小规模快速测试
    run_small_scale_test()
