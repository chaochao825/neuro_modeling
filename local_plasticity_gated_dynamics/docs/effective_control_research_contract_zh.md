# 高秩 E/I 基底上的低维 credit assignment：执行合同

## 修正后的核心命题

本项目不再把“低维反馈必然产生低秩物理连接”作为主命题。稀疏 mask、
Dale 投影、状态依赖导数和 fan-in normalization 都可以把瞬时 rank-1
外积变成高秩物理更新。新的可证伪命题是：

> 局部资格迹与低维任务相关 learning signal 在高秩稀疏 E/I 网络中塑造
> 少数可控、可观测模式；低维 belief/gain gate 改变这些模式的有效算子；
> 只有改善 held-out 行为或预测的低维结构才计为支持。

需要区分并分别报告：credit-signal span、导数后的 post-factor span、
plastic tangent dimension、物理更新秩、Jacobian outlier 数、activity PR、
empirical Hankel dimension 和 decoder dimension。这些量没有必须相等的理论
理由。

## 阶段门禁

| 阶段 | 必须完成的证据 | 当前代码状态 | 下游门禁 |
|---|---|---|---|
| P0 机制可识别性 | 30 seeds；真实数组指纹配对；L1/L2 分面预算；aligned 对 frozen/shuffled；独立绝对/相对 baseline 结论 | `exp07` 已实现，正式结果待运行 | P0 未过不得把真实数据相关性解释为局部学习机制 |
| P1 理论秩与有效维度 | masked-rank 恒等式；逐阶段 rank；tangent/outlier/PR/Hankel；三种参数化；dimension/angle sweep | `exp08` 驱动、smoke/formal 配置和测试已实现，正式 sweep 待运行 | 低 matrix rank 单独不能 support |
| P2 hidden context | HMM context；learned gate 不可访问真 context；校准和因果干预 | 待实现 | 监督 gate 仅作上界 |
| P3 on/off manifold | 共享 basis 预训练；三类新任务；四种允许更新范围 | 待实现 | 必须在未见组合上评估 |
| P4 E/I/homeostasis | inhibitory fraction、匹配电流/谱余量、timescale 与预算 | 待实现 | homeostasis 不得掩盖 feedback 差异 |
| P5 phase boundary | near-threshold conductance receiver、16 phases、delay、charge match | 待实现 | rate-matched 零效应则正式 oppose 当前 phase gate |
| P6 真实数据 | CompositionalTasks、>=5 animals/>=20 IBL sessions、NLB；MICrONS 延后 | loader/legacy LDS 已有，层级 Poisson/NB 模型待实现 | P0 与 P6A/B 未过不启动 MICrONS 主分析 |

## P0 严格配对定义

每个 seed 只建立一次 branch point，所有反馈条件共享：

- `W_bulk`、sparse mask、E/I signs 和 input weights 的内容哈希；
- oracle gate 数组；
- 只用训练 blocks 拟合并冻结的 readout；
- trial order、outer/inner block split 和逐 trial noise stream；
- frozen-reference 产生的未投影 feedback error 与 homeostatic activity signal。

`aligned/random/orthogonal/full` 只改变同一未投影神经信号的 projector；
`shuffled` 使用 aligned projector，并在同一 block/context 内对完整的预计算
third-factor 向量作精确置换；它保持经验信号边际，只破坏 trial correspondence。
组件面板显式区分 task-only、homeostasis-only、normalization-only、
task+homeostasis、task+normalization、homeostasis+normalization、
task+homeostasis+normalization 和 frozen recurrent。由此 normalization 不再是
隐藏在其他机制名称中的副作用。P0 的 homeostasis 信号是只读 replay 的单向
抑制增强 control；它只用于消除混杂，不能作为 E/I 稳态或闭环 homeostasis
证据，后者必须由 P4 单独验证。

L1 与 L2 是两个独立正式面板。单个标量缩放通常不可能同时精确匹配两种
范数，因此每个面板只以选定范数作为门禁，另一范数作为诊断量。任何预算
shortfall 都使对应 seed/condition 不能进入 support 统计。L1/L2 不得先平均：
每个面板独立计算区间与方向检验，联合原始 p 值取两面板 p 值的最大值，随后
进入全声明族 Holm 校正；只有两面板均通过才可 support。正式 seed 集严格为
`0..29`，额外校准 seed 不得替代任何缺失或失败 seed。
配置中的 budget 是每个机制的目标：task-only 与 homeostasis-only 各使用一份
相同预算；task+homeostasis 保持同一份 homeostasis 预算并额外加入一份等额
task 预算，因此严格 on/off 对照不会改变 homeostasis 本身。

## Baseline 与统计合同

- rate-RNN BPTT 与 GRU 使用完整 block 的 inner train/validation 划分；
- 相同 cell type 与 hidden size 的超参数候选共享参数初始化和 minibatch 顺序，
  避免把幸运初始化混入超参数选择；
- test 数据不在 tuning/refit API 参数中；最佳超参在完整 development blocks
  上从头 refit，epoch 数由 inner best epoch 预先确定；
- simulation 的独立单位只能是 seed；真实数据只能是 session/animal；
- 所有预处理只在训练折拟合，时间点不能随机拆分；
- support/oppose 需要预注册 bootstrap 方向条件与 Holm 校正同时通过；
- P0 的绝对 accuracy、相对 tuned-BPTT、相对 tuned-GRU 是三条独立结论；
- 所有失败、无效条件和候选调参失败都写入不可变 artifact。

## P0 支持标准

P0 只有在 30 个完整 seeds、两个预算面板均有效时才可能 support：

1. aligned task plasticity 相对 bitwise frozen recurrent 改善 held-out
   prediction/behavior；
2. aligned 相对 shuffled 改善同一 held-out endpoint；
3. 在 normalization 明确关闭时，加入 aligned task plasticity 相对相同预算的
   homeostasis-only control 有增益；其他 normalization 配对作为独立组件审计；
4. 绝对性能与两个相对 baseline 判据分开报告；
5. 低秩或低 tangent dimension 若不伴随 held-out 改善，不计为支持。
