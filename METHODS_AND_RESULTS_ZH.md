# 当前方法与结果汇总

更新时间：2026-07-07

## 1. 总体目标

当前工作不是证明某个统一大理论，而是建立一套可复现实验框架，用来区分以下神经群体动力学机制：

- H1：历史相关与局部耦合。
- H2：近临界/critical initialization 与幂律或长尾协方差谱。
- H3：振荡同步码，需要窄带谱峰、相位锁定、复 DMD 模式和 phase reset 等额外证据。
- H4：神经雪崩/临界传播，不能只依赖 log-log 直线。
- H5：能量、稀疏性、布线成本和稳定性约束。

## 2. 已实现方法

### 2.1 `neural_multiscale_tests`

该目录实现了 6 类自包含合成模拟和统一指标输出：

- Baseline Bernoulli/Poisson-like negative control。
- GLM/Hawkes：history、local coupling、global latent/stimulus covariates 的嵌套预测增益。
- Linear random dynamics：不同谱半径、对称性、DMD/VAR、Lyapunov 协方差一致性。
- Branching/avalanche：branching ratio、tail model comparison、dynamic range。
- E/I spiking network：PSD、PLV、DMD、phase-reset proxy。
- Energy efficiency：information/activity/wiring-cost proxy sweep。

核心入口：

- `neural_multiscale_tests/run_simulations.py`
- `neural_multiscale_tests/analyses/metrics.py`
- `neural_multiscale_tests/analyses/reporting.py`
- `neural_multiscale_tests/fit_public_data.py`

主要输出：

- `neural_multiscale_tests/reports/summary.json`
- `neural_multiscale_tests/reports/decision_matrix.json`
- `neural_multiscale_tests/reports/integrated_goal_status_zh.md`
- `neural_multiscale_tests/figures/integrated_goal_status.png`

### 2.2 `minimal_computation_python`

该目录将 `ChrisWLynn/Minimal_computation` 的核心思想转成 Python：

- 读取 MATLAB v5 `.mat` 二值活动矩阵。
- 计算二值 pairwise mutual information。
- 拟合 maximum-entropy/logistic output neuron。
- 贪心选择输入，报告 model entropy 与 residual pairwise-correlation error。
- 生成每个数据集的 JSON、Markdown 报告和 PNG/PDF 曲线。

核心入口：

- `minimal_computation_python/run_reproduction.py`
- `minimal_computation_python/minimal_computation/core.py`
- `minimal_computation_python/minimal_computation/matv5.py`

发布说明：本仓库不重分发上游 `Minimal_computation` 的 MATLAB 原代码和 `.mat` 数据。若要重新运行 sweep，需要从 `https://github.com/ChrisWLynn/Minimal_computation` 获取上游仓库，并放在本仓库根目录下，命名为 `minimal_computation_original/`。

## 3. 当前结果

### 3.1 H1-H5 合成框架结果

`python -B run_simulations.py --quick --seed 7` 后，当前 decision matrix 为：

| 假设 | 等级 | 关键结果 |
|---|---|---|
| H1 history + local coupling | strong | history delta = 0.0021 bits/bin；local delta = 0.0077 bits/bin |
| H2 near-critical / power-law spectrum | strong | alpha = 0.9136；Lyapunov log-eig corr = 0.9893 |
| H3 oscillatory synchrony code | weak | PSD peak 和 PLV 存在，但缺 near-unit complex DMD 与 positive phase reset |
| H4 avalanche criticality | strong | m=1 附近 branching ratio = 0.9761；dynamic range 最优点在 m=1 |
| H5 energy constraint | strong | best sparsity = 0.09；long-range fraction = 0.035；rho = 0.9 |

解释：

- H1/H2/H4/H5 在合成设置中与原始预期一致。
- H3 被降级为 weak 是预期内结果，因为当前证据不足以支持“同步码”强结论。

### 3.2 Minimal_computation Python 复现结果

| 数据集 | shape | max sweep entropy drop | final residual / complete threshold | complete 判据 |
|---|---:|---:|---:|---|
| C. elegans | 128 x 1600 | 90.08% | 0.098 | 达到：21 inputs |
| Hippocampus | 1485 x 70338 | 49.46% | 7.671 | 未在 30 inputs 内达到 |
| Visual spontaneous | 11445 x 4696 | 40.62% | 2.204 | 未在 30 inputs 内达到 |
| Visual responding | 11445 x 5166 | 27.48% | 2.442 | 未在 30 inputs 内达到 |

解释：

- C. elegans 单神经元结果支持论文中“少数直接依赖解释相关结构”的方向。
- 鼠 hippocampus/visual 当前只得到熵下降的部分证据，未在轻量 sweep 中达到 complete，不应解释为对论文的完整反驳。
- Python 版尚未完全复刻 MATLAB 的解析 entropy-drop Hessian 选择和二分搜索。

## 4. 当前完成边界

已完成：

- 合成模拟框架、统一指标、H1-H5 decision matrix。
- 两套核心可视化：H1-H5 总览图、integrated goal status 图。
- Minimal_computation Python 复现和四个数据集的单神经元轻量 sweep 结果汇总。
- smoke tests。

尚未完成：

- Allen/IBL/Steinmetz/Stringer/Buzsaki/CRCNS 等真实公开数据的实际下载、清洗、分层和拟合。
- running、pupil、lick、wheel、reward、sleep/wake、theta phase 等行为状态控制。
- optogenetic/MEA/Neuropixels/ECoG/MEG/fMRI 等真实因果扰动数据分析。
- 论文级全神经元、全区域、全任务统计复现。

当前最准确结论：框架主体已经可运行，合成结果与原始预期基本一致；真实数据与因果实验层仍需要下一阶段推进。
