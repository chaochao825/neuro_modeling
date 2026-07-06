# 原始 Goal 当前完成情况与 Minimal_computation 子复现整合报告

本报告整合两个工作区的当前结果：

- `neural_multiscale_tests`：原始 H1-H5 神经群体动力学框架的合成模拟与分析 pipeline。
- `minimal_computation_python`：对 `ChrisWLynn/Minimal_computation` 的 Python 转写和单神经元轻量复现。

## 当前状态

`neural_multiscale_tests` 已重新运行：

```powershell
python -B run_simulations.py --quick --seed 7
```

当前 `decision_matrix.json` 的结论为：

| 假设 | 当前等级 | 关键指标 |
|---|---|---|
| H1 history + local coupling | strong | history delta = 0.0021 bits/bin；local delta = 0.0077 bits/bin |
| H2 near-critical / power-law spectrum | strong | alpha = 0.9136；Lyapunov log-eig corr = 0.9893 |
| H3 oscillatory synchrony code | weak | PSD peak 和 PLV 存在，但缺 near-unit complex DMD 与 positive phase reset |
| H4 avalanche criticality | strong | m=1 附近 branching ratio = 0.9761；dynamic range 最优点在 m=1 |
| H5 energy constraint | strong | best sparsity = 0.09；long-range fraction = 0.035；rho = 0.9 |

`Minimal_computation` 子复现给出的真实/公开数据侧补充为：

| 数据集 | 结论 |
|---|---|
| C. elegans | 21 个输入达到 complete 判据，支持“少数直接依赖解释相关结构”的方向 |
| Hippocampus | 当前 30-input sweep 未达到 complete，但 entropy 降低 49.46% |
| Visual spontaneous | 当前 30-input sweep 未达到 complete，但 entropy 降低 40.62% |
| Visual responding | 当前 30-input sweep 未达到 complete，但 entropy 降低 27.48% |

## 与原始预期是否一致

总体一致，但只在“合成模拟层 + 一个局部公开数据子任务”上成立。

一致的部分：

- 原始 goal 要求不要证明统一大理论，而是区分竞争机制；当前 `decision_matrix.json` 正是按 H1-H5 分别给 strong/weak。
- H1、H2、H4、H5 在合成设置中得到与预期一致的强证据。
- H3 被降级为 weak，符合原始 goal 中“振荡同步码需要额外条件，不能只靠相关性/PSD/PLV”的判据。
- `Minimal_computation` 结果补充了 H1 的一个局部直接依赖案例，尤其 C. elegans 单神经元达到 complete。

不完整的部分：

- 真实公开数据拟合仍不足：`fit_public_data.py` 和 registry 已有，但 Allen/IBL/Steinmetz/Buzsaki/Stringer 等真实数据尚未实际下载、分层和拟合。
- 行为状态控制还没有进入真实数据分析：running、pupil、lick、wheel、trial event、reward、sleep/wake、theta phase 等目前只是接口/设计要求。
- 因果扰动和真实实验设计尚未形成可执行数据分析：opto/MEA/Neuropixels/ECoG/MEG/fMRI 当前没有真实扰动数据。
- `Minimal_computation` 只是 H1 的直接依赖子模块，不覆盖 H2-H5。

## 新增可视化

- `figures/integrated_goal_status.png`
- `figures/integrated_goal_status.pdf`
- `figures/integrated_goal_status_plot.py`

图中四个面板含义：

- A：H1-H5 合成 pipeline 证据等级。
- B：`Minimal_computation` 四个数据集的熵降低比例。
- C：`Minimal_computation` 四个数据集最终残余误差与 complete 阈值的比值。
- D：原始大 goal 覆盖度审计，区分合成模拟、公开数据接口、Minimal 子复现和真实因果实验层。

## 当前最准确结论

当前工作已经完成了原始大 goal 的“可复现机制区分框架”主体：6 类模拟、统一指标、H1-H5 决策矩阵、可视化和测试均可运行。结果与预期基本一致，尤其是 H3 的弱证据结论避免了过度声称。

尚未完成的是原始 goal 中最重的真实数据与真实实验层：公开数据下载/拟合、行为状态控制、区域/任务分层、真实扰动因果检验。下一阶段应优先选择一个真实公开数据源，先把 H1 + H2 的真实数据路径跑通，再逐步扩展到 avalanche、oscillation 和 energy proxy。
