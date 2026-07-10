# 神经群体动力学框架完成情况对比汇总与分析

> `synthetic_calibration / single_seed` 辅助报告；不是生物统计推断。H1 数值已按训练前缀 scaler 的当前代码重新生成。

本报告对照用户粘贴文本中的目标，核对当前 `neural_multiscale_tests` 仓库的实际交付物、模拟结果、证据等级和仍未覆盖的范围。结论只基于当前工作树中的文件与生成结果，不把模拟验证外推为真实脑数据结论。

## 1. 总体完成度

| 目标项 | 当前完成情况 | 证据文件 | 评价 |
|---|---|---|---|
| 可复现 Python repo | 已完成 | `README.md`, `run_simulations.py`, `tests/test_smoke.py` | 已形成可运行的自包含模拟与分析仓库。 |
| 6 类模拟模型 | 已完成 | `simulations/*.py`, `run_simulations.py` | Baseline、GLM/Hawkes、线性随机动力学、分支/雪崩、E/I LIF、能量约束模型均有实现。 |
| 统一指标输出 | 已完成 | `analyses/metrics.py`, `reports/summary.json` | 输出自相关、互相关、协方差谱、PSD、DMD、雪崩、GLM、Lyapunov、能量 proxy 等指标。 |
| H1-H5 证据矩阵 | 已完成 | `reports/decision_matrix.json`, `reports/report.md` | 自动映射为 strong / moderate / weak / refuted；当前 H3 被严格降级为 weak。 |
| 公开数据拟合接口 | 部分完成 | `fit_public_data.py`, `data_loaders/public_registry.py` | 已有统一入口和数据集 registry，但未实际下载/拟合 Allen、IBL、Steinmetz、Buzsáki/CRCNS 真实数据。 |
| 真实实验设计 | 部分完成 | 本报告第 5 节、原目标文本判据 | 当前代码主要实现模拟和公开数据接口，尚未形成独立的真实实验方案文档或实验数据分析。 |
| SSH 可复现执行 | 已完成过一次远端验证 | `scripts/run_remote.sh`, `environment.yml`, `requirements.txt` | 远端脚本支持显式 `NEURAL_TESTS_CONDA_ENV` / `NEURAL_TESTS_PYTHON`，并打印 Python/NumPy 版本。 |

## 2. H1-H5 结果对比

| 假设 | 目标判据 | 当前结果 | 证据等级 | 分析 |
|---|---|---:|---|---|
| H1 历史相关 + 局部耦合 | history-only 提升 test likelihood；local-coupled 继续提升；global/latent 可用于检验共同状态混淆 | history delta = 0.0022 bits/bin；local delta = 0.0081 bits/bin；Hawkes 平均互相关 0.0328，高于 baseline 0.0290 | strong | 合成 GLM/Hawkes 数据中，自历史项和局部耦合均提供增量预测信息。该结论只证明 pipeline 能检出这类机制，不等价于真实皮层数据中一定存在同等强度局部耦合。 |
| H2 近临界/幂律谱 | 近对称、近临界线性动力学产生幂律协方差谱；Lyapunov 方程可预测经验协方差 | critical symmetric case 的谱指数 alpha = 0.9136；Lyapunov log-eigenspectrum corr = 0.9893 | strong | 当前合成设置能复现“近临界归一化 + 近对称动力学 -> 长尾协方差谱”的核心现象，并且不是只看相关性，而是加入了 Lyapunov 预测一致性。 |
| H3 振荡同步码 | 需同时出现窄带峰、phase locking、接近单位圆复 DMD 模态、phase reset | PSD peak ratio = 13.17；PLV = 0.722；near-unit complex DMD = 0；phase reset proxy = -0.012 | weak | 虽有窄带功率峰和相位锁定，但缺少接近单位圆的复模态和正向 phase reset。因此只能说出现部分振荡/相位统计特征，不能支持“振荡同步码”。 |
| H4 近临界/雪崩 | m≈1、幂律优于替代分布、finite-size scaling / dynamic range 峰值等共同支持，不能只靠 log-log | m=1 case 的 estimated branching ratio = 0.9761；dynamic range 最优 m = 1.0；size/duration tail 当前最佳为 power-law | strong | 在合成分支过程里，临界点附近的 branching ratio 和 dynamic range 判据同时成立。当前 finite-size scaling 仍是简化 proxy，真实数据上必须继续做 subsampling、bin-size sweep 和状态控制。 |
| H5 能量约束 | 中等稀疏、少量长程连接、rho(A)≤1 时信息/能耗比最高 | best sparsity = 0.09；long-range fraction = 0.035；rho = 0.9；information/cost = 2.0819 | strong | 能量 proxy sweep 支持“中等稀疏 + 少量长程连接 + 近临界但稳定”的低功耗塑形路线。该结果基于 proxy，不是代谢测量。 |

## 3. 六类模拟实验完成情况

| 模拟模块 | 粘贴文本要求 | 当前实现 | 当前结论 |
|---|---|---|---|
| Baseline 独立 Poisson/Bernoulli | 无历史、无耦合时不应自动产生幂律谱或同步码 | `simulations/baseline.py` | 作为 negative control；平均互相关 0.0290，低于 Hawkes 合成数据。 |
| GLM/Hawkes | 历史项、局部耦合、共同输入、刺激项；比较 nested likelihood | `simulations/hawkes.py`, `models/glm.py`, `fit_glm_hawkes.py` | history/local 均产生正增益，支持 H1 的可检出性。 |
| 线性随机动力学 | 比较对称、部分对称、非对称和不同谱半径；验证 Lyapunov 协方差 | `simulations/linear_dynamics.py`, `fit_dmd_var.py` | 近临界对称 case 产生长尾谱，Lyapunov 预测一致性高。 |
| 分支过程/雪崩 | 扫 m<1, m=1, m>1；输出 branching ratio、tail model、dynamic range | `simulations/branching.py`, `avalanche.py` | m≈1 case 最符合当前合成临界判据。 |
| E/I spiking network | LIF/AdEx E/I、AMPA/GABA 延迟；扫 E/I、延迟、噪声；检测同步 | `simulations/ei_spiking.py`, `oscillation.py` | 当前只得到 PSD/PLV 支持，未达到同步码强证据。 |
| 能量约束模型 | 最大化信息收益并惩罚 spike、布线、失稳；比较稀疏/长程/小世界 | `simulations/energy.py`, `energy_efficiency.py` | 最优点落在中等稀疏、少量长程、rho=0.9。 |

## 4. 交付物逐项核对

| 编号 | 要求文件 | 当前状态 | 备注 |
|---:|---|---|---|
| 1 | `run_simulations.py` | 已完成 | 运行 6 类模型并写出 `summary.json/report.md/decision_matrix.json`。 |
| 2 | `fit_public_data.py` | 已完成接口 | 支持本地 spike matrix 或 synthetic demo；未内置大数据下载。 |
| 3 | `fit_glm_hawkes.py` | 已完成 | 比较 bias/history/local/global/full。 |
| 4 | `fit_dmd_var.py` | 已完成 | 估计有效线性动力学和 Lyapunov 协方差一致性。 |
| 5 | `eigenspectrum.py` | 已完成 | 拟合协方差谱幂律并做 time shuffle control。 |
| 6 | `avalanche.py` | 已完成 | 提取 avalanche，比较 power-law / exponential / lognormal / stretched exponential。 |
| 7 | `oscillation.py` | 已完成但指标为近似实现 | 包含 PSD、PLV、DMD、spike-phase locking；coherence/phase reset 目前是轻量 proxy，不是完整实验仪式。 |
| 8 | `energy_efficiency.py` | 已完成 | 输出信息、稀疏度、布线和能量 proxy。 |
| 9 | `report.md` | 已完成 | 自动报告 H1-H5 当前证据等级。 |
| 10 | `decision_matrix.json` | 已完成 | 结构化证据等级。 |

## 5. 与公开数据/真实实验要求的差距

| 范围 | 原要求 | 当前状态 | 后续需要 |
|---|---|---|---|
| Allen / IBL / Steinmetz / Buzsáki/CRCNS | 真实公开数据拟合与行为变量控制 | 只完成 registry 和统一本地矩阵入口 | 需要准备实际数据导出、bin-size sweep、行为协变量、区域标签和任务状态。 |
| Stringer 2019 / calcium | CV-PCA/SVCA、deconvolved vs raw 对比 | 当前只有通用 eigenspectrum 与 shuffle control | 需要接入 calcium traces、deconvolution 版本和 trial/stimulus 元数据。 |
| 行为/状态控制 | running、pupil、lick、wheel、reward、sleep/wake、theta phase | 在 registry 和 public-data 输出中列为 required controls | 尚未实现具体回归/分层控制模型。 |
| 因果扰动 | opto/脉冲、top eigenmode、phase reset | E/I 模拟中有 pulse/reset proxy | 需要真实 opto 或扰动数据，当前不支持因果强结论。 |
| 有限尺寸缩放 | avalanche finite-size scaling | 当前尾部分布和 dynamic range 已有，finite-size scaling 仍简化 | 需要多网络尺寸或多 subsampling 条件系统扫描。 |

## 6. 解释边界

当前结果证明的是：这套 pipeline 能区分若干竞争机制，并在合成数据上给出与目标判据一致的证据矩阵。当前结果没有证明“大脑统一理论”，也没有证明真实脑区在所有状态下都符合 H1-H5。

最可靠的已完成部分是模拟侧的可复现验证：H1、H2、H4、H5 在 synthetic setting 下获得 strong；H3 因缺少 near-unit complex DMD 和 positive phase reset，被判为 weak。最主要的未完成部分是真实公开数据拟合和行为/状态控制，这需要实际数据文件、区域标签和实验元数据。

