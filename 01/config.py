"""
T-DQN 观点最大化复现项目的全局配置参数。
所有默认值均遵循论文的设定。
"""

# ===================== 网络生成参数 =====================

# BBV (Barabási-Albert-Vázquez) 无标度网络参数
BBV_PARAMS = dict(
    n=5000,          # 网络节点数
    m=3,             # 每步新增边数（平均度约6，有向后翻倍）
    p_rewire=0.1,    # 边移除概率（BBV扩展机制）
)

# SBM (Stochastic Block Model) 随机块模型参数
SBM_PARAMS = dict(
    n=5000,          # 网络节点数
    n_communities=5, # 社区数量
    p_intra=0.02,    # 社区内部连边概率
    p_inter=0.002,   # 社区之间连边概率
)

# WS (Watts-Strogatz) 小世界网络参数
WS_PARAMS = dict(
    n=5000,          # 网络节点数
    k=6,             # 每个节点连接的最近邻居数
    p_rewire=0.1,    # 边重连概率
)

# ===================== 信任网络参数 =====================
POSITIVE_WEIGHT_RATIO = 0.80   # 正权重（信任边）占比 80%，负权重（不信任边）占比 20%

# ===================== 观点动力学参数 =====================
BETA_SELECTION = 10.0           # Fermi规则选择强度（中等强度）
OPINION_RANGE = (-1.0, 1.0)    # 观点取值范围 [-1, 1]
SEED_OPINION = 1.0              # 种子节点的固定观点值（+1，即完全正向）
STUBBORNNESS_COOPERATION_PROB = 0.5  # 初始合作策略选择概率

# ===================== T-DQN 算法参数 =====================
LAMBDA_POTENTIAL = 2            # 比例参数：第一阶段选取 λ*k 个候选种子
DQN_BATCH_SIZE = 32             # 每次训练的批量大小
DQN_BUFFER_SIZE = 10000         # 经验回放缓冲区容量
DQN_GAMMA = 0.99                # 折扣因子（越大越重视长期回报）
DQN_LR = 0.001                  # 学习率
DQN_TAU = 0.05                  # 目标网络软更新系数
DQN_EPSILON_INIT = 0.4          # ε-贪心策略初始探索率
DQN_EPSILON_DECAY = 0.9         # ε衰减系数
DQN_EPSILON_MIN = 0.1           # ε最小值
DQN_BETA_INIT = 0.4             # 优先经验回放重要性采样修正系数初始值
DQN_BETA_INCREMENT = 1e-4       # 重要性采样系数增量
DQN_SOFT_UPDATE_FREQ = 10       # 每10次选择后进行一次目标网络软更新

# ===================== DQN 网络架构参数 =====================
STATE_DIM = None                # 状态维度（根据网络动态设置，通常为 2 * 候选种子特征维度）
HIDDEN_DIMS = [128, 64, 32]     # 隐藏层尺寸
ACTION_DIM = 5                  # 5种动作策略：MaxDegree、Blocking、MixStrategy、CbC、CI

# ===================== 实验设置 =====================
K_VALUES = [10, 20, 30, 40, 50]  # 测试的种子节点数量列表
T_VALUES = [10, 30, 50, 80, 100] # 测试的观点动力学时间步列表
NUM_TRIALS = 10                  # 独立实验次数（用于求平均）
DEFAULT_K = 40                   # 默认种子节点数
DEFAULT_T = 30                   # 默认时间步数

# ===================== CbC 社区检测与CI参数 =====================
CI_RADIUS = 2                    # Collective Influence 计算半径
MIX_STRATEGY_SIGMA = 0.5         # MixStrategy 中出度与入度的平衡系数
