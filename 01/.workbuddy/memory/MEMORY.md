# MEMORY.md - Long-term Memory

## 项目: 代码复现

### 2026-04-21: T-DQN 论文复现
- **论文**: Yu et al., "Opinion maximization on social trust networks based on game theory and DQN method", Information Sciences 729 (2026)
- **目录**: `d:\Code\OutSource\3w\代码复现\01\`
- **文件**: config.py, network_generator.py, opinion_dynamics.py, baseline_algorithms.py, tdqn.py, experiment.py, run_experiment.py, README.md
- **依赖**: torch, numpy, networkx, matplotlib, python-louvain
- **运行方式**: `python run_experiment.py --mode quick/medium`
- **状态**: 核心代码完成，小规模测试通过

### 技术踩坑记录
- `community` 包的正确导入: `import community.community_louvain as community_louvain`，不能用 `import community`
- Louvain社区检测不支持负权重，需先移除负权重边
- Windows PowerShell 中 Python 的 stderr 输出会被误报为 Error（CLIXML），实际不影响执行
