"""
独立运行折线图实验：横坐标 k=0~100（步长20），纵坐标意见总和。
包含全部 7 种算法（含 T-DQN）。
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import networkx as nx

from network_generator import get_weight_matrix
from experiment import detect_communities, run_single_algorithm, _plot_k_opinion_line
from baseline_algorithms import ACTION_NAMES
from config import POSITIVE_WEIGHT_RATIO

# ---- 构建 200 节点 WS 网络（与 quick 模式一致）----
np.random.seed(42)
G = nx.watts_strogatz_graph(200, 6, 0.1, seed=42)
DG = nx.DiGraph()
DG.add_nodes_from(G.nodes())
for u, v in G.edges():
    if np.random.random() < 0.5:
        DG.add_edge(u, v)
    else:
        DG.add_edge(v, u)
for u, v in DG.edges():
    if np.random.random() < POSITIVE_WEIGHT_RATIO:
        DG[u][v]['weight'] = np.random.uniform(0.1, 1.0)
    else:
        DG[u][v]['weight'] = np.random.uniform(-1.0, -0.1)

node_list = list(DG.nodes())
W = get_weight_matrix(DG, node_list)
communities = detect_communities(DG)
np.random.seed(42)
initial_opinions = np.random.uniform(-1, 1, 200)
T = 10

# ---- 调用折线图函数（含全部 7 种算法）----
_plot_k_opinion_line(
    DG, W, communities, initial_opinions, T,
    k_values=[0, 20, 40, 60, 80, 100],
    save_path='results/k_opinion_line.png'
)
