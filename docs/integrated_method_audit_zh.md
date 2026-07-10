# 方法整合审计：从局部依赖到门控低维动力学

审计日期：2026-07-11。本文区分三类证据：旧仓库的合成校准、`local_plasticity_gated_dynamics` 的预注册仿真，以及公开神经数据的 held-out 检验。三者不得互相替代。

## 核心假说与可证伪对象

目标假说是：局部资格迹、低维第三因子和 E/I 稳态共同产生可复用的低维任务动力学，并由低维门控实现快速上下文切换。必须区分以下对象：

- 任务可塑更新 `Delta W` 的秩；
- 掩码、Dale 投影和归一化后的实际连接变化秩；
- 线性化 Jacobian 的谱；
- 神经活动协方差或流形维度；
- Minimal-computation 的最小输入数 `n*`。

低活动维度、稀疏输入数和低秩连接不是同一个命题。

## 当前实现审计

### 可保留的证据

- Phase 1 的构造性任务在 20 seeds 上支持：feedback dimension 为 4 时，更新 effective rank 约为 4；aligned feedback 的 latent R2 与 full feedback 非劣；orthogonal control 明显更差。
- 训练集拟合 normalization/PCA/subspace、按 trial/block 切分、seed/session/animal 统计单位和失败结果留存已经由单元测试保护。
- BPTT 位于隔离 baseline 路径，没有进入 local-learning 主模型。

### 必须降级或重做的证据

1. Phase 2 的低秩闭包没有保留。反馈维度为 4 时，正式 E/I 条件的 raw update effective rank 约为 9.6--10.1；经过稀疏 mask、Dale 投影和 fan-in 归一化后约为 312--316/512。当前结果只能说明低维活动可能由高秩受约束连接产生，不能说明最终连接更新仍为 rank 4。
2. local、full-feedback 和 shuffled-feedback 的正式任务准确率在所有可配对 seeds 上完全相同，因此行为结果无法归因于反馈对齐。
3. inhibitory-homeostasis 的累计 L1 更新约为任务三因子更新的 660--690 倍。`no-homeostasis` 的不稳定性方向还与预期相反。后续必须增加 homeostasis-only，并按每突触、每次更新匹配预算。
4. learned gate 仍接受显式 context 监督：MD 拟合和第三因子特征含真实 context 信息。它应命名为 supervised Hebbian context classifier，而不是无监督上下文发现。
5. B1 的 `support` 来自“达到配对 BPTT 的 90%”分支；绝对 accuracy 0.85 分支失败。当前 BPTT 也未作为充分调参的性能上界。
6. 相位门控在严格 rate/source matching 后差值精确为零，因此 C1 为 `oppose`，不能用旧 H3 合成振荡结果补强。

## 旧 H1--H5 的证据边界

`neural_multiscale_tests` 当前输出来自 quick seed=7、N=36、T=900 的单次合成校准。`strong/moderate/weak` 是手工阈值评分，不是独立重复上的统计推断。

- H1 原先在全时间序列标准化后再切分，存在泄漏；本分支已改为训练前缀拟合 scaler。没有空间坐标的真实矩阵不再把列索引环当作“局部连接”。
- H2 使用按目标幂律构造的生成矩阵，只能验证分析代码是否恢复已知结构。
- H3 的 population phase reference 含被测神经元自身，存在循环相关。
- H4 是已知临界 branching process 的校准，缺少离散尾部分布的 goodness-of-fit bootstrap。
- H5 的目标函数显式奖励预设 long-range 值，不能作为独立能量最优证据。

因此旧结论统一标为 `synthetic_calibration / single_seed`，不能与 20-seed 或真实数据结论并列。

## Minimal-computation 方法边界

公开论文方法拟合同时刻条件依赖 `P(y_t | x_t)`；其 `n*` 是落入 Poisson 相关误差包络所需的最少输入神经元数，不是 latent rank，也不是未来动力学或因果连接。

旧 Python 版本在首输入后用归一化残余相关排序，并仅报告粗 sweep 首个通过点；原 MATLAB 使用 Hessian/Schur 二阶熵下降并继续二分最小输入数。因此旧版 C. elegans neuron 13 的 `n*=21` 不能作为严格复现。本分支的 MATLAB-parity block-Schur 实跑在 7 inputs 达到 residual 判据，MATLAB 1-based 输入顺序为 `[74,34,3,1,123,54,120]`；旧算法保留为显式 approximation baseline。分块 Schur 与 full Hessian 有 golden parity 测试，MATLAB residual-only completion 与更保守的 optimizer+residual 模式也分别命名。

## 真实数据首轮设计

现有四个 MAT 文件只有二值 `X[neuron,time]`。上游说明 visual responding 与 visual spontaneous 来自同一批 11,445 个神经元；MAT 文件没有 unit ID，因此逐行对齐仍是不可独立复核的上游假设。在该边界下可比较：

- common LDS：共享 observation basis 和共同 transition；
- shared-basis switching LDS：共享 basis、context-specific transition；
- separate-basis low-rank LDS：各 context 独立 basis、均值、噪声和 transition；它是受限低秩对照，不是 full observation-space LDS；
- random、orthogonal 和 neuron-alignment-shuffled basis controls。

所有 fold 使用连续时间块，禁止随机拆时间点或跨块构造 lag；unit selection、normalization、PCA、noise 和 transition 只在训练块拟合。primary endpoint 是 held-out marginal Gaussian LDS NLL，secondary endpoints 包括 one-step R2、rollout NRMSE、参数量、effective rank 和 context subspace angle。

正式面板已完成 2,160/2,160 个计划 cell（20 个固定 neuron-subset seeds、3 个连续 block folds、6 个 latent dimensions、6 个 models），失败数为 0。`d=4` 的关键结果为：

- shared-minus-common NLL 的 seed 中位数为 `+0.000203`，即 context-specific transition 没有改善 common transition；
- shared 的 mean one-step R2 为 `-0.001567`，mean rollout NRMSE 为 `1.013948`；另按 seed 中位数计算的联合绝对预测 margin 为 `-0.013727`；
- random、orthogonal、shuffled 三类 control 相对 aligned shared 的逐-seed 最小 NLL gap 中位数为 `+0.001340`，所以对齐基底在该维度有 recording 内描述性优势；
- shared 参数量中位数为 934，separate low-rank 为 1,692；但 separate NLL 更差，导致“保留 separate 相对 common 的 switching gain”分母为负，R1 不可计算；
- latent-transition effective rank 为 `3.03`。该量天然不超过预设 latent dimension，不能单独证明观察空间或连接更新低秩。

在 `d=4`，common 同时具有更低 NLL（`-0.239844` 对 `-0.239648`）、更高 one-step R2（`-0.000743` 对 `-0.001567`）、更低 rollout NRMSE（`1.013199` 对 `1.013948`）和更少参数（910 对 934），因此首轮真实数据没有展示 switching/shared 模型优势。R2 只说明训练集 PCA aligned basis 优于三种基底对照，不能抵消 common baseline，也不能验证局部可塑性机制。

这两个 recording 没有 trial、刺激 ID、行为、E/I 类型或自然切换时刻；fold、seed 和 neuron 不是生物重复。因此上述支持/反对只是单 recording pair 内的描述方向，所有真实数据总体结论仍为“无法判定”。真实快速切换、行为预测和 E/I 稳态仍需多个 session/animal 的 IBL 或同类数据。

## 当前三分类结论

| 命题 | 结论 | 原因 |
|---|---|---|
| 构造性低维反馈可形成 rank-matched 更新 | 支持 | Phase 1，20 seeds，aligned controls 完整 |
| 当前 E/I 实现保持低秩连接更新 | 反对 | mask/Dale/normalization 后更新接近高秩 |
| 当前任务行为由对齐的低维第三因子驱动 | 无法判定 | local/full/shuffled 行为不可辨识 |
| 当前 inhibitory homeostasis 提高稳定性 | 反对 | ablation 方向与预期相反且预算严重失配 |
| 当前 learned MD gate 是无监督上下文发现 | 反对 | 使用真实 context 监督信息 |
| 严格 rate-matched phase gate 有独立优势 | 反对 | 20 seeds 差值为零 |
| shared switching 在真实数据上优于 common dynamics | 无法判定（单记录内反对） | d=4 shared-minus-common NLL 为正，且仅一个 recording pair |
| aligned shared basis 优于 basis controls | 无法判定（单记录内支持） | d=4 三类 control gap 均为正，但没有独立 session/animal；不等于 shared 优于 common |
| d=4 shared NLL 接近最高测试维度 d=32 | 无法判定（单记录内支持） | 仅比较两个预设维度，不是 intrinsic rank=4 的估计 |
| shared 模型有绝对 held-out 预测信号 | 无法判定（单记录内反对） | one-step R2 < 0 且 rollout NRMSE > 1 |
| shared-basis 保留 separate low-rank 对照的 switching gain | 无法判定 | separate 比 common 更差，retained-gain 分母无效；也未实现 full observation-space LDS |

任何后续“优势”必须同时满足 held-out 改善、参数/能量代价、合法统计单位和预注册对照，而不能仅由 effective rank 或单神经元 entropy 下降推断。
