# T-DQN: Opinion Maximization on Social Trust Networks

**论文复现**: Yu et al., "Opinion maximization on social trust networks based on game theory and DQN method", *Information Sciences* 729 (2026) 122879

## 项目结构

```
01/
├── config.py              # 所有超参数配置
├── network_generator.py   # 社交信任网络生成（BBV, SBM, WS）
├── opinion_dynamics.py    # 博弈论观点动态模型（Fermi策略更新）
├── baseline_algorithms.py # 基线算法（5种启发式 + Greedy）
├── tdqn.py                # T-DQN 核心算法（两阶段DQN + Q-Learning）
├── experiment.py          # 实验框架（多种子数/时间步对比）
├── run_experiment.py      # 主入口 + 可视化绘图
├── README.md              # 本文件
└── results/               # 实验结果输出（自动生成）
```

## 核心方法

### 1. 博弈论观点动态模型
- 节点观点 ∈ [-1, 1]，种子节点观点固定为 +1
- 节点选择策略：合作（更新观点）或背叛（保持观点）
- Fermi规则更新策略：模仿收益更高的邻居

### 2. T-DQN 两阶段种子选择
- **Stage 1**: 基于节点影响力筛选 λ·k 个潜在种子
- **Stage 2**: DQN 从潜在种子中选出 k 个最终种子
  - 5个动作：MaxDegree, Blocking, MixStrategy, CbC, CI
  - 优先经验回放（PER）+ 软更新目标网络
  - Double DQN + 梯度裁剪

## 快速开始

```bash
# 安装依赖
pip install torch numpy networkx matplotlib python-louvain

# 快速测试（200节点网络）
python run_experiment.py --mode quick

# 中等规模实验（500节点，3个网络）
python run_experiment.py --mode medium
```

## 论文对应实验

| 实验 | 论文图号 | 说明 |
|------|----------|------|
| 不同种子数对比 | Fig. 3 | k = [10, 20, 30, 40, 50] |
| 不同时间步对比 | Fig. 4 | T = [10, 30, 50, 80, 100] |
| 初始合作率影响 | Fig. 5 | 变化初始合作概率 |
| 网络结构分析 | Fig. 6 | 不同网络拓扑对比 |

## 关键参数（与论文一致）

| 参数 | 值 | 说明 |
|------|-----|------|
| β (Fermi选择强度) | 10 | 中等强度选择 |
| λ (潜在种子比例) | 2 | 选择 2k 个潜在种子 |
| τ (软更新系数) | 0.05 | 目标网络更新速率 |
| ε (初始探索率) | 0.4 | ε-greedy 初始值 |
| γ (折扣因子) | 0.99 | DQN 折扣因子 |
| 正权重比例 | 80% | 信任边占比 |

## 注意事项

1. **观点动态随机性**: 由于策略更新使用Fermi随机规则，每次运行结果会有波动，需多次trial取平均
2. **大规模网络**: 5000节点网络的完整实验（7个算法×5个k×10个trial）需要数小时
3. **GPU加速**: T-DQN 自动检测CUDA，如有GPU可显著加速训练
4. **真实网络**: WikiElec、Gnutella、Reddit 数据集需从公开源下载
