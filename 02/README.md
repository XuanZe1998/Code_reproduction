# 论文复现说明

本目录包含两部分内容：

1. [paper_translation_zh.md](D:/Code/OutSource/3w/代码复现/02/paper_translation_zh.md)：按章节整理的中文翻译。
2. `src/opinion_dqn`：对论文 *Opinion maximization on social trust networks based on game theory and DQN method* 的模块化算法复现。

## 目录结构

```text
src/opinion_dqn/
  config.py                  # 参数定义
  graph.py                   # 社会信任网络数据结构与合成网络生成
  dynamics.py                # 博弈观点动力学
  heuristics.py              # 候选种子筛选与 5 种动作策略
  state.py                   # 25 维状态向量
  replay.py                  # 优先经验回放
  dqn.py                     # 策略网络/目标网络
  env.py                     # 选种环境与奖励计算
  trainer.py                 # T-DQN 主流程
  baselines.py               # 基线策略
examples/
  generate_synthetic_networks.py
                             # 生成论文中的 BBV / SBM / WS 合成网络
  plot_paper_style_results.py
                             # 跑实验并按论文风格生成图片
  run_full_pipeline.py
                             # 统一入口：生成网络 + 跑实验 + 画图
```

## 实现范围

已复现论文核心方法：

- 社会信任网络上的连续观点传播
- 合作/背叛博弈式策略更新
- 候选种子两阶段筛选
- 5 种动作策略
- 25 维压缩状态
- DQN + Prioritized Replay + Soft Update
- 论文中的三类合成网络：`BBV / SBM / WS`

## 依赖

- Python 3.11+
- `numpy`
- `networkx`
- `torch`

## 运行示例

```powershell
python examples/run_full_pipeline.py
```

如果只想单独导出合成网络，可以运行：

```powershell
python examples/generate_synthetic_networks.py
```

它会在 `generated_networks/` 下导出：

- `bbv_edges.csv`
- `sbm_edges.csv`
- `ws_edges.csv`
- `network_summaries.json`

## 关于 BBV 生成器

论文只给了 BBV 网络的文字描述，没有给出完整生成代码。当前仓库实现的是一个 `BBV-like` 生成器：

1. 以 BA 式优先连接为基础；
2. 在增长过程中引入边删除；
3. 在增长过程中引入边重连。

这和论文对 “BA 扩展并允许边丢失/变化” 的描述一致，适合作为工程复现版本。如果你后面拿到作者原始代码或更精确的 BBV 定义，可以直接替换 `graph.py` 里的 `generate_bbv_like()`。

## 说明

论文中的真实数据集与图表原始文件当前不在目录里，所以现在是“算法级复现 + 合成网络复现”，还不是“逐图逐表复现”。如果你后续补齐真实数据，我可以继续把实验脚本补成论文同款。
