"""候选种子筛选与 5 种动作策略实现。"""

from __future__ import annotations

from collections.abc import Callable

import networkx as nx
import numpy as np

from .dynamics import weighted_average_benefits
from .graph import SocialTrustNetwork

ACTION_NAMES = ["MaxDegree", "Blocking", "MixStrategy", "CbC", "CI"]


def node_influence_scores(network: SocialTrustNetwork, initial_opinions: np.ndarray) -> dict[int, float]:
    # 论文公式（8）：
    # 计算第一阶段候选种子筛选所需的节点影响力分数。
    benefits = weighted_average_benefits(network, initial_opinions)
    bundle = network.get_numpy_bundle()
    edge_src = bundle["edge_src"]
    edge_dst = bundle["edge_dst"]
    edge_weight = bundle["edge_weight"]
    num_nodes = network.num_nodes

    out_term = np.zeros(num_nodes, dtype=np.float32)
    in_term = np.zeros(num_nodes, dtype=np.float32)
    if edge_src.size > 0:
        np.add.at(out_term, edge_src, benefits[edge_dst] * edge_weight)
        np.add.at(in_term, edge_dst, benefits[edge_src] * edge_weight)

    total = benefits + out_term + in_term
    scores: dict[int, float] = {node: float(total[node]) for node in range(num_nodes)}
    return scores


def select_potential_seeds(
    network: SocialTrustNetwork,
    initial_opinions: np.ndarray,
    seed_budget: int,
    multiplier: int,
) -> list[int]:
    # T-DQN 第一阶段：
    # 仅保留 xi * k 个高分节点作为候选种子，缩小动作搜索空间。
    scores = node_influence_scores(network, initial_opinions)
    size = min(network.num_nodes, seed_budget * multiplier)
    ranked = sorted(scores, key=lambda node: scores[node], reverse=True)
    return ranked[:size]


def detect_communities(network: SocialTrustNetwork) -> dict[int, int]:
    # CbC 需要社区划分，这里使用 networkx 的贪心模块度社区检测。
    cache_key = "community_map"
    if cache_key in network._analysis_cache:
        return network._analysis_cache[cache_key]  # type: ignore[return-value]

    undirected = network.get_undirected_graph()
    communities = list(nx.algorithms.community.greedy_modularity_communities(undirected))
    mapping: dict[int, int] = {}
    for cid, community in enumerate(communities):
        for node in community:
            mapping[int(node)] = cid
    network._analysis_cache[cache_key] = mapping
    return mapping


def community_sizes(community_map: dict[int, int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for cid in community_map.values():
        counts[cid] = counts.get(cid, 0) + 1
    return counts


def cbc_score(network: SocialTrustNetwork, node: int, community_map: dict[int, int]) -> float:
    # Community-based Centrality：综合节点连接到的社区数量和社区规模。
    sizes = community_sizes(community_map)
    current = community_map[node]
    score = 0.0
    neighbors = network.get_numpy_bundle()["undirected_neighbors"][node]
    for nbr in neighbors:
        cid = community_map[nbr]
        score += sizes[cid] / network.num_nodes
    return score + (sizes[current] / network.num_nodes)


def ci_score(network: SocialTrustNetwork, node: int) -> float:
    # Collective Influence：按照论文设定，半径固定为 2。
    bundle = network.get_numpy_bundle()
    neighbors = bundle["undirected_neighbors"]
    degree = bundle["degree"]

    first_ring = neighbors[node]
    second_ring: set[int] = set()
    for nbr in first_ring:
        second_ring.update(neighbors[nbr])
    second_ring.discard(node)
    second_ring.difference_update(first_ring)

    total = sum(max(int(degree[nbr]) - 1, 0) for nbr in second_ring)
    return float(max(int(degree[node]) - 1, 0) * total)


def get_static_action_scores(network: SocialTrustNetwork, community_map: dict[int, int]) -> dict[str, np.ndarray]:
    # 这些分数只依赖图结构，与训练过程中的状态无关，适合网络级缓存。
    cache_key = "static_action_scores"
    cached = network._analysis_cache.get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    bundle = network.get_numpy_bundle()
    out_degree = bundle["out_degree"].astype(np.float32)
    in_degree = bundle["in_degree"].astype(np.float32)
    degree = bundle["degree"].astype(np.float32)
    neighbors = bundle["undirected_neighbors"]

    sizes = community_sizes(community_map)
    cbc = np.zeros(network.num_nodes, dtype=np.float32)
    ci = np.zeros(network.num_nodes, dtype=np.float32)

    for node in range(network.num_nodes):
        current = community_map[node]
        score = sizes[current] / network.num_nodes
        for nbr in neighbors[node]:
            score += sizes[community_map[nbr]] / network.num_nodes
        cbc[node] = score

        first_ring = neighbors[node]
        second_ring: set[int] = set()
        for nbr in first_ring:
            second_ring.update(neighbors[nbr])
        second_ring.discard(node)
        second_ring.difference_update(first_ring)
        total = sum(max(int(degree[nbr]) - 1, 0) for nbr in second_ring)
        ci[node] = float(max(int(degree[node]) - 1, 0) * total)

    scores = {
        "out_degree": out_degree,
        "in_degree": in_degree,
        "mix_degree": 0.5 * out_degree + 0.5 * in_degree,
        "cbc": cbc,
        "ci": ci,
    }
    network._analysis_cache[cache_key] = scores
    return scores


def rank_by_action(
    network: SocialTrustNetwork,
    candidates: list[int],
    selected: set[int],
    action_idx: int,
    community_map: dict[int, int],
    mix_lambda: float = 0.5,
) -> list[int]:
    # T-DQN 第二阶段：
    # 先选一个“动作策略族”，再在当前未使用候选节点中选该策略的最高分节点。
    bundle = network.get_numpy_bundle()
    out_degree = bundle["out_degree"]
    in_degree = bundle["in_degree"]
    static_scores = get_static_action_scores(network, community_map)
    pool = [node for node in candidates if node not in selected]
    if action_idx == 0:
        key: Callable[[int], float] = lambda node: float(static_scores["out_degree"][node])
    elif action_idx == 1:
        key = lambda node: float(static_scores["in_degree"][node])
    elif action_idx == 2:
        if abs(mix_lambda - 0.5) < 1e-12:
            key = lambda node: float(static_scores["mix_degree"][node])
        else:
            key = lambda node: mix_lambda * int(out_degree[node]) + (1.0 - mix_lambda) * int(in_degree[node])
    elif action_idx == 3:
        key = lambda node: float(static_scores["cbc"][node])
    elif action_idx == 4:
        key = lambda node: float(static_scores["ci"][node])
    else:
        raise ValueError(f"unknown action index: {action_idx}")
    return sorted(pool, key=key, reverse=True)
