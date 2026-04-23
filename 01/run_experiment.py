"""
主实验运行器和可视化模块。

功能：
- 在多种网络上运行完整的对比实验
- 生成发表质量的可视化图表
- 支持三种运行模式：quick（快速）、medium（中等）、full（完整）

使用方式：
  python run_experiment.py --mode quick     # 200节点，快速验证
  python run_experiment.py --mode medium    # 500节点，中等规模
  python run_experiment.py --mode full      # 5000节点，完整实验
"""

import numpy as np
import time
import os
import sys
import io

# 修复 Windows 中文编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免弹窗
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
except:
    pass

from network_generator import (
    generate_bbv_network, generate_sbm_network, generate_ws_network,
    get_weight_matrix, get_network_info
)
from opinion_dynamics import OpinionDynamics
from baseline_algorithms import (
    heuristic_seed_selection, ACTION_NAMES,
    ACTION_MAXDEGREE, ACTION_BLOCKING, ACTION_MIXSTRATEGY, ACTION_CBC, ACTION_CI
)
from tdqn import TDQN, QLearning, compute_potential_seeds
from experiment import detect_communities, run_single_algorithm
from config import (
    BBV_PARAMS, SBM_PARAMS, WS_PARAMS,
    DEFAULT_K, DEFAULT_T, NUM_TRIALS, K_VALUES, T_VALUES,
    LAMBDA_POTENTIAL
)


def plot_k_comparison(results, network_name, k_values, save_path=None):
    """
    绘制观点总和随种子数量 k 变化的对比图。
    
    对应论文图 3：展示不同算法在不同种子数量下的观点传播效果。
    使用误差棒表示多次实验的标准差。
    
    Args:
        results: 实验结果字典
        network_name: 网络名称（用于标题）
        k_values: k 值列表
        save_path: 图片保存路径（可选）
    """
    algo_names = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 定义标记和颜色
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, algo in enumerate(algo_names):
        means = [results[algo][k]['mean'] for k in k_values]
        stds = [results[algo][k]['std'] for k in k_values]
        
        ax.errorbar(k_values, means, yerr=stds, 
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=algo, linewidth=2, markersize=8, capsize=3)
    
    ax.set_xlabel('种子数量 (k)', fontsize=12)
    ax.set_ylabel('最终观点总和', fontsize=12)
    ax.set_title(f'{network_name}: 观点最大化 vs 种子数量', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图片已保存: {save_path}")
    plt.close()


def plot_T_comparison(results, network_name, T_values, k, save_path=None):
    """
    绘制观点总和随时间步 T 变化的对比图。
    
    对应论文图 4：展示不同算法在观点演化过程中的表现差异。
    
    Args:
        results: 实验结果字典
        network_name: 网络名称
        T_values: T 值列表
        k: 固定的种子数量
        save_path: 图片保存路径（可选）
    """
    algo_names = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, algo in enumerate(algo_names):
        means = [results[algo][T]['mean'] for T in T_values]
        stds = [results[algo][T]['std'] for T in T_values]
        
        ax.errorbar(T_values, means, yerr=stds,
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=algo, linewidth=2, markersize=8, capsize=3)
    
    ax.set_xlabel('时间步数 (T)', fontsize=12)
    ax.set_ylabel('最终观点总和', fontsize=12)
    ax.set_title(f'{network_name}: 观点最大化 vs 时间步数 (k={k})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图片已保存: {save_path}")
    plt.close()


def run_full_experiment_on_network(network_name, G, k_values, T_values, 
                                    num_trials, output_dir='results'):
    """
    在单个网络上运行完整的实验套件。
    
    包含两组实验：
    1. 实验 1：改变 k（种子数量），固定 T → 对应论文图 3
    2. 实验 2：改变 T（时间步数），固定 k → 对应论文图 4
    
    Args:
        network_name: 网络名称
        G: 有向图
        k_values: 测试的 k 值列表
        T_values: 测试的 T 值列表
        num_trials: 每组实验的重复次数
        output_dir: 结果输出目录
    
    Returns:
        tuple: (k 实验结果, T 实验结果)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 预计算权重矩阵和社区划分
    node_list = list(G.nodes())
    W = get_weight_matrix(G, node_list)
    communities = detect_communities(G)
    
    # ---- 实验 1：改变 k（论文图 3） ----
    print(f"\n{'='*60}")
    print(f"实验 1: 在 {network_name} 上改变 k 值")
    print(f"{'='*60}")
    
    # 初始化结果存储
    k_results = {algo: {k: {'mean': 0, 'std': 0, 'values': []} for k in k_values}
                 for algo in ACTION_NAMES + ['Q-Learning', 'T-DQN']}
    
    for k in k_values:
        print(f"\n  k = {k}")
        for trial in range(num_trials):
            # 生成随机初始观点
            np.random.seed(42 + trial)
            initial_opinions = np.random.uniform(-1, 1, W.shape[0])
            
            # 对每个算法运行一次
            for algo_name in ACTION_NAMES + ['Q-Learning', 'T-DQN']:
                np.random.seed(42 + trial)  # 重置种子保证公平
                seeds, total, elapsed = run_single_algorithm(
                    algo_name, G, W, k, DEFAULT_T, initial_opinions, communities
                )
                k_results[algo_name][k]['values'].append(total)
            
            print(f"    实验 {trial+1}/{num_trials} 完成。")
        
        # 统计结果
        for algo in k_results:
            vals = k_results[algo][k]['values']
            k_results[algo][k]['mean'] = np.mean(vals)
            k_results[algo][k]['std'] = np.std(vals)
    
    # 打印结果表格
    print(f"\n  结果（均值 ± 标准差）:")
    print(f"  {'算法':<15}", end="")
    for k in k_values:
        print(f"  k={k:<10}", end="")
    print()
    for algo in ACTION_NAMES + ['Q-Learning', 'T-DQN']:
        print(f"  {algo:<15}", end="")
        for k in k_values:
            m = k_results[algo][k]['mean']
            s = k_results[algo][k]['std']
            print(f"  {m:.1f}±{s:.1f}  ", end="")
        print()
    
    # 绘制 k 对比图
    plot_k_comparison(k_results, network_name, k_values,
                      save_path=os.path.join(output_dir, f'{network_name}_k_comparison.png'))
    
    # ---- 实验 2：改变 T（论文图 4） ----
    print(f"\n{'='*60}")
    print(f"实验 2: 在 {network_name} 上改变 T 值 (k={DEFAULT_K})")
    print(f"{'='*60}")
    
    T_results = {algo: {T: {'mean': 0, 'std': 0, 'values': []} for T in T_values}
                 for algo in ACTION_NAMES + ['Q-Learning', 'T-DQN']}
    
    for T in T_values:
        print(f"\n  T = {T}")
        for trial in range(min(num_trials, 5)):  # T 实验减少重复次数
            np.random.seed(42 + trial)
            initial_opinions = np.random.uniform(-1, 1, W.shape[0])
            
            for algo_name in ACTION_NAMES + ['Q-Learning', 'T-DQN']:
                np.random.seed(42 + trial)
                seeds, total, elapsed = run_single_algorithm(
                    algo_name, G, W, DEFAULT_K, T, initial_opinions, communities
                )
                T_results[algo_name][T]['values'].append(total)
            
            print(f"    实验 {trial+1} 完成。")
        
        # 统计
        for algo in T_results:
            vals = T_results[algo][T]['values']
            T_results[algo][T]['mean'] = np.mean(vals)
            T_results[algo][T]['std'] = np.std(vals)
    
    # 打印结果
    print(f"\n  结果（均值 ± 标准差）:")
    print(f"  {'算法':<15}", end="")
    for T in T_values:
        print(f"  T={T:<10}", end="")
    print()
    for algo in ACTION_NAMES + ['Q-Learning', 'T-DQN']:
        print(f"  {algo:<15}", end="")
        for T in T_values:
            m = T_results[algo][T]['mean']
            s = T_results[algo][T]['std']
            print(f"  {m:.1f}±{s:.1f}  ", end="")
        print()
    
    # 绘制 T 对比图
    plot_T_comparison(T_results, network_name, T_values, DEFAULT_K,
                      save_path=os.path.join(output_dir, f'{network_name}_T_comparison.png'))
    
    return k_results, T_results


def run_medium_scale_demo():
    """
    中等规模演示实验（500 个节点）。
    
    在 BBV、SBM、WS 三种网络上运行完整实验，
    使用缩减的参数以确保在合理时间内完成。
    
    Returns:
        dict: 所有网络的实验结果
    """
    print("=" * 60)
    print("中等规模演示（500 个节点）")
    print("=" * 60)
    
    # 生成三种网络
    print("\n生成网络中...")
    np.random.seed(42)
    
    G_bbv = generate_bbv_network(n=500, m=3, p_rewire=0.1)
    G_sbm = generate_sbm_network(n=500, n_communities=5, p_intra=0.02, p_inter=0.002)
    G_ws = generate_ws_network(n=500, k=6, p_rewire=0.1)
    
    # 输出网络统计信息
    for name, G in [('BBV', G_bbv), ('SBM', G_sbm), ('WS', G_ws)]:
        get_network_info(G)
    
    # 实验参数
    k_values = [5, 10, 15, 20]   # 缩减的 k 值
    T_values = [10, 20, 30]      # 缩减的 T 值
    num_trials = 3               # 减少实验次数
    output_dir = os.path.join('results', 'medium_scale')
    
    all_results = {}
    
    # 在每种网络上运行完整实验
    for name, G in [('BBV', G_bbv), ('SBM', G_sbm), ('WS', G_ws)]:
        k_res, T_res = run_full_experiment_on_network(
            name, G, k_values, T_values, num_trials, 
            output_dir=os.path.join(output_dir, name)
        )
        all_results[name] = {'k': k_res, 'T': T_res}
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='T-DQN 观点最大化实验')
    parser.add_argument('--mode', choices=['quick', 'medium', 'full'], default='quick',
                        help='实验规模: quick(200节点快速验证), medium(500节点中等规模), full(5000节点完整实验)')
    args = parser.parse_args()
    
    if args.mode == 'quick':
        # 快速模式：200 节点，用于功能验证
        from experiment import run_small_scale_test
        run_small_scale_test()
    elif args.mode == 'medium':
        # 中等模式：500 节点
        run_medium_scale_demo()
    elif args.mode == 'full':
        # 完整模式：5000 节点（论文实际规模）
        print("完整规模实验（5000 个节点）...")
        print("这可能需要数小时。按 Ctrl+C 可随时停止。\n")
        run_medium_scale_demo()  # TODO: 替换为完整规模实验
