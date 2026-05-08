"""将当前网络状态压缩为论文中的 25 维特征向量。"""

from __future__ import annotations

import numpy as np

from .graph import SocialTrustNetwork


def _degree_hist(values: np.ndarray, bins: int = 10) -> np.ndarray:
    # 将度分布压缩成固定长度直方图，避免状态维度依赖网络规模。
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)
    max_value = max(int(values.max()), 1)
    hist, _ = np.histogram(values, bins=bins, range=(0, max_value))
    return (hist / max(hist.sum(), 1)).astype(np.float32)


def build_state_vector(
    network: SocialTrustNetwork,
    opinions: np.ndarray,
    strategies: np.ndarray,
    selected_seeds: list[int],
) -> np.ndarray:
    # 25 维状态向量组成：
    # 2 个全局特征 + 3 个种子集合统计量 + 10 个出度分箱 + 10 个入度分箱。
    bundle = network.get_numpy_bundle()
    degree = bundle["degree"]
    out_degree = bundle["out_degree"]
    in_degree = bundle["in_degree"]
    coop_ratio = float(np.mean(strategies))
    avg_opinion = float(np.mean(opinions))

    if selected_seeds:
        degrees = degree[np.asarray(selected_seeds, dtype=np.int64)].astype(np.float32)
        seed_stats = np.array(
            [len(selected_seeds), float(np.mean(degrees)), float(np.max(degrees))],
            dtype=np.float32,
        )
    else:
        seed_stats = np.zeros(3, dtype=np.float32)

    out_degrees = out_degree.astype(np.float32)
    in_degrees = in_degree.astype(np.float32)

    state = np.concatenate(
        [
            np.array([coop_ratio, avg_opinion], dtype=np.float32),
            seed_stats,
            _degree_hist(out_degrees, bins=10),
            _degree_hist(in_degrees, bins=10),
        ]
    )
    return state.astype(np.float32)
