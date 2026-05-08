"""基于博弈论的连续观点动力学实现。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import DynamicsConfig
from .graph import SocialTrustNetwork


@dataclass
class SimulationResult:
    """一次观点传播仿真的输出结果。"""

    final_opinions: np.ndarray
    final_strategies: np.ndarray
    overall_opinion: float


def _resolve_tensor_device(device_name: str) -> torch.device:
    """按 cuda -> xpu -> cpu 的顺序自动选择张量设备。"""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 不支持 CUDA，无法在 GPU 上执行观点传播。")
    if device_name == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("当前 PyTorch 不支持 XPU，无法在 Intel GPU 上执行观点传播。")
    return torch.device(device_name)


def local_benefit(weight: float, opinion_i: float, opinion_j: float) -> float:
    """论文公式（3）的标量版本。"""
    diff = abs(opinion_j - opinion_i)
    if weight > 0:
        return 1.0 - 0.5 * diff
    return 0.5 * diff


def weighted_average_benefits(network: SocialTrustNetwork, opinions: np.ndarray) -> np.ndarray:
    """论文公式（4）的 numpy 包装。"""
    device = _resolve_tensor_device("cpu")
    bundle = network.get_torch_bundle(device)
    opinions_tensor = torch.as_tensor(opinions, dtype=torch.float32, device=device)
    benefits_tensor = weighted_average_benefits_tensor(bundle, opinions_tensor)
    return benefits_tensor.cpu().numpy()


def weighted_average_benefits_tensor(bundle: dict[str, torch.Tensor], opinions: torch.Tensor) -> torch.Tensor:
    """论文公式（4）的边列表张量化实现。"""
    num_nodes = opinions.shape[0]
    edge_src = bundle["edge_src"]
    edge_dst = bundle["edge_dst"]
    edge_weight = bundle["edge_weight"]
    edge_abs_weight = bundle["edge_abs_weight"]
    out_weight_sum = bundle["out_weight_sum"]

    if edge_src.numel() == 0:
        return torch.zeros_like(opinions)

    src_opinions = opinions[edge_src]
    dst_opinions = opinions[edge_dst]
    diff = (dst_opinions - src_opinions).abs()
    edge_local_benefit = torch.where(edge_weight > 0, 1.0 - 0.5 * diff, 0.5 * diff)

    numerators = torch.zeros(num_nodes, dtype=opinions.dtype, device=opinions.device)
    numerators.index_add_(0, edge_src, edge_abs_weight * edge_local_benefit)

    benefits = torch.zeros_like(opinions)
    valid_rows = out_weight_sum > 0
    benefits[valid_rows] = numerators[valid_rows] / out_weight_sum[valid_rows].clamp_min(1e-12)
    return benefits


def update_opinions(
    network: SocialTrustNetwork,
    opinions: np.ndarray,
    strategies: np.ndarray,
    benefits: np.ndarray,
    seed_mask: np.ndarray,
) -> np.ndarray:
    """论文公式（2）的 numpy 包装。"""
    device = _resolve_tensor_device("cpu")
    bundle = network.get_torch_bundle(device)
    updated = update_opinions_tensor(
        bundle,
        torch.as_tensor(opinions, dtype=torch.float32, device=device),
        torch.as_tensor(strategies, dtype=torch.int64, device=device),
        torch.as_tensor(benefits, dtype=torch.float32, device=device),
        torch.as_tensor(seed_mask, dtype=torch.bool, device=device),
    )
    return updated.cpu().numpy()


def update_opinions_tensor(
    bundle: dict[str, torch.Tensor],
    opinions: torch.Tensor,
    strategies: torch.Tensor,
    benefits: torch.Tensor,
    seed_mask: torch.Tensor,
) -> torch.Tensor:
    """论文公式（2）的边列表张量化实现。"""
    num_nodes = opinions.shape[0]
    edge_src = bundle["edge_src"]
    edge_dst = bundle["edge_dst"]
    edge_weight = bundle["edge_weight"]
    edge_abs_weight = bundle["edge_abs_weight"]

    if edge_src.numel() == 0:
        updated = opinions.clone()
        updated[seed_mask] = 1.0
        return updated

    neighbor_signal = edge_weight * benefits[edge_dst] * opinions[edge_dst]
    neighbor_norm = edge_abs_weight * benefits[edge_dst]

    weighted_neighbor_signal = torch.zeros(num_nodes, dtype=opinions.dtype, device=opinions.device)
    weighted_neighbor_norm = torch.zeros(num_nodes, dtype=opinions.dtype, device=opinions.device)
    weighted_neighbor_signal.index_add_(0, edge_src, neighbor_signal)
    weighted_neighbor_norm.index_add_(0, edge_src, neighbor_norm)

    numerator = opinions + weighted_neighbor_signal
    denominator = 1.0 + weighted_neighbor_norm
    cooperative_update = torch.clamp(numerator / denominator.clamp_min(1e-12), min=-1.0, max=1.0)

    updated = torch.where(strategies == 1, cooperative_update, opinions)
    updated = torch.where(seed_mask, torch.ones_like(updated), updated)
    return updated


def update_strategies(
    network: SocialTrustNetwork,
    strategies: np.ndarray,
    benefits: np.ndarray,
    selection_strength: float,
    seed_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """论文公式（5）到（7）的 numpy 包装。"""
    device = _resolve_tensor_device("cpu")
    bundle = network.get_torch_bundle(device)
    updated = update_strategies_tensor(
        bundle,
        torch.as_tensor(strategies, dtype=torch.int64, device=device),
        torch.as_tensor(benefits, dtype=torch.float32, device=device),
        selection_strength,
        torch.as_tensor(seed_mask, dtype=torch.bool, device=device),
        torch.Generator(device=device.type).manual_seed(int(rng.integers(0, 2**31 - 1))),
    )
    return updated.cpu().numpy()


def update_strategies_tensor(
    bundle: dict[str, torch.Tensor],
    strategies: torch.Tensor,
    benefits: torch.Tensor,
    selection_strength: float,
    seed_mask: torch.Tensor,
    torch_rng: torch.Generator | None = None,
) -> torch.Tensor:
    """论文公式（5）到（7）的边列表张量化实现。"""
    num_nodes = strategies.shape[0]
    trusted_src = bundle["trusted_src"]
    trusted_dst = bundle["trusted_dst"]
    trusted_log_prob = bundle["trusted_log_prob"]
    valid_trust_rows = bundle["valid_trust_rows"]

    updated = strategies.clone()
    updated[seed_mask] = 1

    if trusted_src.numel() == 0:
        return updated

    # 用 Gumbel-Max 技巧在每个节点的可信出邻居中完成一次离散采样。
    uniform = torch.rand(trusted_src.shape[0], device=strategies.device, generator=torch_rng).clamp_min(1e-12)
    gumbel = -torch.log(-torch.log(uniform))
    sample_scores = trusted_log_prob + gumbel

    max_scores = torch.full((num_nodes,), float("-inf"), dtype=sample_scores.dtype, device=sample_scores.device)
    max_scores.scatter_reduce_(0, trusted_src, sample_scores, reduce="amax", include_self=True)
    chosen_mask = sample_scores == max_scores[trusted_src]

    chosen_dst_plus = torch.zeros(num_nodes, dtype=torch.int64, device=strategies.device)
    chosen_dst_plus.scatter_reduce_(
        0,
        trusted_src,
        torch.where(chosen_mask, trusted_dst + 1, torch.zeros_like(trusted_dst)),
        reduce="amax",
        include_self=True,
    )
    chosen_dst = chosen_dst_plus - 1

    valid_choice = valid_trust_rows & (~seed_mask) & (chosen_dst >= 0)
    if valid_choice.any():
        chosen_benefits = torch.zeros_like(benefits)
        chosen_benefits[valid_choice] = benefits[chosen_dst[valid_choice]]
        imitation_prob = torch.sigmoid(selection_strength * (chosen_benefits - benefits))
        random_values = torch.rand(num_nodes, device=strategies.device, generator=torch_rng)
        adopt_mask = valid_choice & (random_values < imitation_prob)
        updated[adopt_mask] = strategies[chosen_dst[adopt_mask]]

    updated[seed_mask] = 1
    return updated


def run_opinion_dynamics(
    network: SocialTrustNetwork,
    initial_opinions: np.ndarray,
    initial_strategies: np.ndarray,
    seeds: list[int],
    config: DynamicsConfig,
    rng: np.random.Generator,
    device_name: str = "auto",
) -> SimulationResult:
    """执行一次完整的观点传播仿真。"""
    device = _resolve_tensor_device(device_name)
    bundle = network.get_torch_bundle(device)

    opinions = torch.as_tensor(initial_opinions, dtype=torch.float32, device=device).clone()
    strategies = torch.as_tensor(initial_strategies, dtype=torch.int64, device=device).clone()
    seed_mask = torch.zeros(network.num_nodes, dtype=torch.bool, device=device)

    if seeds:
        seed_indices = torch.as_tensor(seeds, dtype=torch.int64, device=device)
        seed_mask[seed_indices] = True
        opinions[seed_mask] = 1.0
        strategies[seed_mask] = 1

    seed_value = int(rng.integers(0, 2**31 - 1))
    torch_rng = torch.Generator(device=device.type).manual_seed(seed_value)

    with torch.no_grad():
        for _ in range(config.time_steps):
            benefits = weighted_average_benefits_tensor(bundle, opinions)
            next_opinions = update_opinions_tensor(bundle, opinions, strategies, benefits, seed_mask)
            next_benefits = weighted_average_benefits_tensor(bundle, next_opinions)
            next_strategies = update_strategies_tensor(
                bundle,
                strategies,
                next_benefits,
                config.selection_strength,
                seed_mask,
                torch_rng=torch_rng,
            )
            opinions = next_opinions
            strategies = next_strategies

    final_opinions = opinions.detach().cpu().numpy()
    final_strategies = strategies.detach().cpu().numpy()
    return SimulationResult(
        final_opinions=final_opinions,
        final_strategies=final_strategies,
        overall_opinion=float(final_opinions.sum()),
    )
