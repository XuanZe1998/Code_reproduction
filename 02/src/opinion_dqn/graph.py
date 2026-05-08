"""社会信任网络数据结构与论文合成网络生成器。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import torch


@dataclass
class SocialTrustNetwork:
    # 有向带符号图。边权先保留正负号，随后再按出度归一化，
    # 对应论文里“一个节点把注意力平均分配给所有出邻居”的设定。
    graph: nx.DiGraph

    def __post_init__(self) -> None:
        # 不同设备上的张量缓存。构建一次后复用，避免每次仿真都从 networkx 重建矩阵。
        self._tensor_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._numpy_cache: dict[str, np.ndarray | list[set[int]]] = {}
        self._analysis_cache: dict[str, object] = {}

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def nodes(self) -> list[int]:
        return list(self.graph.nodes())

    def out_neighbors(self, node: int) -> list[int]:
        return list(self.graph.successors(node))

    def in_neighbors(self, node: int) -> list[int]:
        return list(self.graph.predecessors(node))

    def weight(self, src: int, dst: int) -> float:
        return float(self.graph[src][dst]["weight"])

    def out_degree(self, node: int) -> int:
        return int(self.get_numpy_bundle()["out_degree"][node])

    def in_degree(self, node: int) -> int:
        return int(self.get_numpy_bundle()["in_degree"][node])

    def degree(self, node: int) -> int:
        return int(self.get_numpy_bundle()["degree"][node])

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[int, int, float]],
        num_nodes: int | None = None,
    ) -> "SocialTrustNetwork":
        graph = nx.DiGraph()
        if num_nodes is not None:
            graph.add_nodes_from(range(num_nodes))
        for src, dst, weight in edges:
            graph.add_edge(int(src), int(dst), weight=float(weight))
        return cls(graph)

    @classmethod
    def from_snap_signed_csv(cls, csv_path: str | Path) -> "SocialTrustNetwork":
        # 读取 SNAP 风格带符号边表：
        # SOURCE, TARGET, RATING, TIME
        path = Path(csv_path)
        raw_edges: list[tuple[int, int, float, float]] = []
        node_ids: set[int] = set()

        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or len(row) < 4:
                    continue
                src = int(row[0])
                dst = int(row[1])
                rating = float(row[2])
                timestamp = float(row[3])
                raw_edges.append((src, dst, rating, timestamp))
                node_ids.add(src)
                node_ids.add(dst)

        sorted_nodes = sorted(node_ids)
        node_to_idx = {node_id: idx for idx, node_id in enumerate(sorted_nodes)}

        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(sorted_nodes)))
        for src, dst, rating, timestamp in raw_edges:
            graph.add_edge(
                node_to_idx[src],
                node_to_idx[dst],
                weight=rating,
                rating=rating,
                time=timestamp,
                raw_src=src,
                raw_dst=dst,
            )

        network = cls(graph)
        network.normalize_outgoing_weights()
        return network

    @classmethod
    def random_signed_digraph(
        cls,
        num_nodes: int,
        edge_prob: float,
        positive_ratio: float = 0.8,
        seed: int = 7,
    ) -> "SocialTrustNetwork":
        rng = np.random.default_rng(seed)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(num_nodes))
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src == dst or rng.random() >= edge_prob:
                    continue
                sign = 1.0 if rng.random() < positive_ratio else -1.0
                graph.add_edge(src, dst, weight=sign)
        cls(graph).normalize_outgoing_weights()
        return cls(graph)

    @classmethod
    def from_undirected_generator(
        cls,
        graph: nx.Graph,
        positive_ratio: float = 0.8,
        seed: int = 7,
    ) -> "SocialTrustNetwork":
        rng = np.random.default_rng(seed)
        digraph = nx.DiGraph()
        digraph.add_nodes_from(graph.nodes())
        for u, v in graph.edges():
            # 将无向拓扑转成有向拓扑。
            # 一部分边只保留单向，一部分边转成双向，用来模拟不对称信任关系。
            if rng.random() < 0.5:
                pairs = [(u, v)]
            else:
                pairs = [(u, v), (v, u)]
            for src, dst in pairs:
                sign = 1.0 if rng.random() < positive_ratio else -1.0
                digraph.add_edge(src, dst, weight=sign)
        stn = cls(digraph)
        stn.ensure_no_isolates(seed)
        stn.normalize_outgoing_weights()
        return stn

    @classmethod
    def generate_sbm(
        cls,
        sizes: list[int],
        probs: list[list[float]],
        positive_ratio: float = 0.8,
        seed: int = 7,
    ) -> "SocialTrustNetwork":
        graph = nx.stochastic_block_model(sizes, probs, seed=seed)
        return cls.from_undirected_generator(graph, positive_ratio=positive_ratio, seed=seed)

    @classmethod
    def generate_ws(
        cls,
        num_nodes: int,
        k: int,
        p: float,
        positive_ratio: float = 0.8,
        seed: int = 7,
    ) -> "SocialTrustNetwork":
        graph = nx.watts_strogatz_graph(num_nodes, k, p, seed=seed)
        return cls.from_undirected_generator(graph, positive_ratio=positive_ratio, seed=seed)

    @classmethod
    def generate_scale_free(
        cls,
        num_nodes: int,
        m: int,
        positive_ratio: float = 0.8,
        seed: int = 7,
    ) -> "SocialTrustNetwork":
        graph = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
        return cls.from_undirected_generator(graph, positive_ratio=positive_ratio, seed=seed)

    @classmethod
    def generate_bbv_like(
        cls,
        num_nodes: int,
        m: int,
        delete_prob: float = 0.08,
        rewire_prob: float = 0.12,
        positive_ratio: float = 0.8,
        seed: int = 7,
    ) -> "SocialTrustNetwork":
        # 论文只说明 BBV 是在 BA 基础上加入“边删除/边变化”的网络，
        # 没给出完整生成代码。这里实现一个工程复现版本：
        # 1）先按优先连接增长
        # 2）增长过程中以一定概率删除边
        # 3）增长过程中以一定概率重连边
        rng = np.random.default_rng(seed)
        undirected = nx.complete_graph(m + 1)
        targets = list(undirected.nodes())

        for new_node in range(m + 1, num_nodes):
            undirected.add_node(new_node)

            degrees = np.array([undirected.degree(node) for node in targets], dtype=np.float64)
            probs = degrees / degrees.sum()
            chosen = rng.choice(targets, size=min(m, len(targets)), replace=False, p=probs)
            for target in chosen:
                undirected.add_edge(new_node, int(target))

            existing_edges = list(undirected.edges())
            if existing_edges and rng.random() < delete_prob:
                edge = existing_edges[int(rng.integers(0, len(existing_edges)))]
                if undirected.number_of_edges() > m:
                    undirected.remove_edge(*edge)

            existing_edges = list(undirected.edges())
            if existing_edges and rng.random() < rewire_prob:
                src, dst = existing_edges[int(rng.integers(0, len(existing_edges)))]
                undirected.remove_edge(src, dst)
                candidates = [node for node in undirected.nodes() if node not in {src} and not undirected.has_edge(src, node)]
                if candidates:
                    cand_deg = np.array([undirected.degree(node) + 1 for node in candidates], dtype=np.float64)
                    cand_probs = cand_deg / cand_deg.sum()
                    new_dst = int(rng.choice(candidates, p=cand_probs))
                    undirected.add_edge(src, new_dst)
                else:
                    undirected.add_edge(src, dst)

            targets.append(new_node)

        return cls.from_undirected_generator(undirected, positive_ratio=positive_ratio, seed=seed)

    @classmethod
    def generate_paper_synthetic_network(
        cls,
        name: str,
        seed: int = 7,
        positive_ratio: float = 0.8,
    ) -> "SocialTrustNetwork":
        key = name.strip().lower()
        if key == "bbv":
            # 论文表 1 给出的目标规模约为 5000 节点、30000 边、平均度 12。
            return cls.generate_bbv_like(
                num_nodes=5000,
                m=4,
                delete_prob=0.03,
                rewire_prob=0.12,
                positive_ratio=positive_ratio,
                seed=seed,
            )
        if key == "sbm":
            sizes = [1000, 1000, 1000, 1000, 1000]
            probs = [
                [0.0027, 0.0004, 0.0004, 0.0004, 0.0004],
                [0.0004, 0.0027, 0.0004, 0.0004, 0.0004],
                [0.0004, 0.0004, 0.0027, 0.0004, 0.0004],
                [0.0004, 0.0004, 0.0004, 0.0027, 0.0004],
                [0.0004, 0.0004, 0.0004, 0.0004, 0.0027],
            ]
            return cls.generate_sbm(sizes=sizes, probs=probs, positive_ratio=positive_ratio, seed=seed)
        if key == "ws":
            return cls.generate_ws(
                num_nodes=5000,
                k=8,
                p=0.2,
                positive_ratio=positive_ratio,
                seed=seed,
            )
        raise ValueError(f"unsupported synthetic network: {name}")

    def summary(self) -> dict[str, float]:
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        avg_degree = (2.0 * num_edges / num_nodes) if num_nodes else 0.0
        positive = sum(1 for _, _, data in self.graph.edges(data=True) if data["weight"] > 0)
        positive_ratio = (positive / num_edges) if num_edges else 0.0
        return {
            "nodes": float(num_nodes),
            "edges": float(num_edges),
            "avg_degree": avg_degree,
            "positive_ratio": positive_ratio,
        }

    def normalize_outgoing_weights(self) -> None:
        # 将每个节点的所有出边权重缩放为 ±1/out_degree，
        # 保留信任/不信任符号，同时满足论文的归一化设定。
        for node in self.graph.nodes():
            neighbors = list(self.graph.successors(node))
            if not neighbors:
                continue
            norm = float(len(neighbors))
            for nbr in neighbors:
                sign = 1.0 if self.graph[node][nbr]["weight"] >= 0 else -1.0
                self.graph[node][nbr]["weight"] = sign / norm

    def ensure_no_isolates(self, seed: int = 7) -> None:
        # 若某个节点没有任何出边，则为其补一条正向边，避免后续动力学中失效。
        rng = np.random.default_rng(seed)
        nodes = list(self.graph.nodes())
        for node in nodes:
            if self.graph.out_degree(node) == 0 and len(nodes) > 1:
                other = int(rng.choice([v for v in nodes if v != node]))
                self.graph.add_edge(node, other, weight=1.0)

    def to_undirected_unsigned(self) -> nx.Graph:
        # 某些启发式策略只用拓扑结构，因此这里忽略方向和符号。
        graph = nx.Graph()
        graph.add_nodes_from(self.graph.nodes())
        for u, v in self.graph.edges():
            graph.add_edge(u, v)
        return graph

    def get_undirected_graph(self) -> nx.Graph:
        # 无向图结构只构建一次，供社区检测等静态分析复用。
        cache_key = "undirected_graph"
        if cache_key not in self._analysis_cache:
            self._analysis_cache[cache_key] = self.to_undirected_unsigned()
        return self._analysis_cache[cache_key]  # type: ignore[return-value]

    def to_edge_list(self) -> list[tuple[int, int, float]]:
        # 导出为简单边列表，便于写入 CSV 或后续复现实验。
        return [
            (int(src), int(dst), float(data["weight"]))
            for src, dst, data in self.graph.edges(data=True)
        ]

    def export_snap_signed_csv(
        self,
        csv_path: str | Path,
        include_header: bool = False,
        synthetic_start_time: float = 1_600_000_000.0,
        synthetic_step: float = 1.0,
    ) -> None:
        # 统一导出为 SNAP 风格：
        # SOURCE, TARGET, RATING, TIME
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[tuple[int, int, int, float]] = []
        current_time = synthetic_start_time
        for src, dst, data in sorted(self.graph.edges(data=True), key=lambda item: (int(item[0]), int(item[1]))):
            export_src = int(data.get("raw_src", int(src) + 1))
            export_dst = int(data.get("raw_dst", int(dst) + 1))
            if "rating" in data:
                rating = int(round(float(data["rating"])))
            else:
                # 人工合成网络没有原始打分，统一映射为 ±10。
                rating = 10 if float(data["weight"]) > 0 else -10
            timestamp = float(data.get("time", current_time))
            rows.append((export_src, export_dst, rating, timestamp))
            current_time += synthetic_step

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if include_header:
                writer.writerow(["SOURCE", "TARGET", "RATING", "TIME"])
            writer.writerows(rows)

    def get_numpy_bundle(self) -> dict[str, np.ndarray | list[set[int]]]:
        # numpy 侧缓存：供启发式筛选、状态提取等 CPU 逻辑复用。
        if self._numpy_cache:
            return self._numpy_cache

        num_nodes = self.num_nodes
        out_degree = np.zeros(num_nodes, dtype=np.int32)
        in_degree = np.zeros(num_nodes, dtype=np.int32)
        out_neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
        in_neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
        undirected_neighbors: list[set[int]] = [set() for _ in range(num_nodes)]

        edge_src: list[int] = []
        edge_dst: list[int] = []
        edge_weight: list[float] = []

        for src, dst, data in self.graph.edges(data=True):
            i = int(src)
            j = int(dst)
            w = float(data["weight"])
            edge_src.append(i)
            edge_dst.append(j)
            edge_weight.append(w)
            out_degree[i] += 1
            in_degree[j] += 1
            out_neighbors[i].append(j)
            in_neighbors[j].append(i)
            undirected_neighbors[i].add(j)
            undirected_neighbors[j].add(i)

        self._numpy_cache = {
            "edge_src": np.asarray(edge_src, dtype=np.int64),
            "edge_dst": np.asarray(edge_dst, dtype=np.int64),
            "edge_weight": np.asarray(edge_weight, dtype=np.float32),
            "out_degree": out_degree,
            "in_degree": in_degree,
            "degree": out_degree + in_degree,
            "out_neighbors": out_neighbors,
            "in_neighbors": in_neighbors,
            "undirected_neighbors": undirected_neighbors,
        }
        return self._numpy_cache

    def get_torch_bundle(self, device: torch.device) -> dict[str, torch.Tensor]:
        # 将 networkx 图转换为边列表 + 分段索引，供大图传播阶段重复使用。
        key = str(device)
        if key in self._tensor_cache:
            return self._tensor_cache[key]

        num_nodes = self.num_nodes
        edge_records = sorted(
            ((int(src), int(dst), float(data["weight"])) for src, dst, data in self.graph.edges(data=True)),
            key=lambda item: (item[0], item[1]),
        )

        if edge_records:
            edge_src = torch.tensor([src for src, _, _ in edge_records], dtype=torch.int64, device=device)
            edge_dst = torch.tensor([dst for _, dst, _ in edge_records], dtype=torch.int64, device=device)
            edge_weight = torch.tensor([weight for _, _, weight in edge_records], dtype=torch.float32, device=device)
        else:
            edge_src = torch.empty(0, dtype=torch.int64, device=device)
            edge_dst = torch.empty(0, dtype=torch.int64, device=device)
            edge_weight = torch.empty(0, dtype=torch.float32, device=device)

        edge_abs_weight = edge_weight.abs()
        edge_positive_mask = edge_weight > 0

        out_weight_sum = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        if edge_src.numel() > 0:
            out_weight_sum.index_add_(0, edge_src, edge_abs_weight)

        out_degree_counts = torch.bincount(edge_src, minlength=num_nodes) if edge_src.numel() > 0 else torch.zeros(num_nodes, dtype=torch.int64, device=device)
        edge_row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=device)
        edge_row_ptr[1:] = torch.cumsum(out_degree_counts, dim=0)

        trusted_src = edge_src[edge_positive_mask]
        trusted_dst = edge_dst[edge_positive_mask]
        trusted_weight = edge_weight[edge_positive_mask]
        trusted_row_sum = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        if trusted_src.numel() > 0:
            trusted_row_sum.index_add_(0, trusted_src, trusted_weight)

        trusted_counts = (
            torch.bincount(trusted_src, minlength=num_nodes)
            if trusted_src.numel() > 0
            else torch.zeros(num_nodes, dtype=torch.int64, device=device)
        )
        trusted_row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=device)
        trusted_row_ptr[1:] = torch.cumsum(trusted_counts, dim=0)

        valid_trust_rows = trusted_counts > 0
        trusted_prob = (
            trusted_weight / trusted_row_sum[trusted_src].clamp_min(1e-12)
            if trusted_src.numel() > 0
            else torch.empty(0, dtype=torch.float32, device=device)
        )
        trusted_log_prob = trusted_prob.clamp_min(1e-12).log() if trusted_prob.numel() > 0 else trusted_prob

        bundle = {
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_weight": edge_weight,
            "edge_abs_weight": edge_abs_weight,
            "edge_positive_mask": edge_positive_mask,
            "edge_row_ptr": edge_row_ptr,
            "out_weight_sum": out_weight_sum,
            "trusted_src": trusted_src,
            "trusted_dst": trusted_dst,
            "trusted_weight": trusted_weight,
            "trusted_prob": trusted_prob,
            "trusted_log_prob": trusted_log_prob,
            "trusted_row_ptr": trusted_row_ptr,
            "valid_trust_rows": valid_trust_rows,
        }
        self._tensor_cache[key] = bundle
        return bundle
