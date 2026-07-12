# 结构化推理与神经数据验证契约

## 1. 研究问题

本扩展不把脉冲作为先验要求。主模型首先采用连续速率状态，检验如下可证伪命题：

> 在相同任务输入、候选/动作生成器、训练任务、随机性和计算预算下，慢状态经低维控制瓶颈调制快状态，是否比无层级或无门控控制器更好地泛化到未见任务；这种收益是否伴随低维有效控制，而不是仅表现为低矩阵秩。

HRM 只启发“慢—快分层更新和截断的一步梯度”，CTM 只启发“连续内部时钟和折扣双线性时序痕迹”。本项目中的实现不是 HRM 或 CTM 的复现。官方 CTM/GRU 等通过时间反向传播的模型只能列为 baseline。

## 2. 三条互不替代的证据轨道

| 轨道 | 输入到输出 | 可支持的结论 | 不可支持的结论 |
|---|---|---|---|
| `exp12` 冻结候选路由 | 外部候选 tape 到候选索引 | 静态、匹配候选上的路由审计 | 端到端求解、真实数据优势、神经机制 |
| `exp13` 可复算结构化推理 | 公开任务到可执行预测，再由隐藏目标评估 | ARC/迷宫/数独上的功能泛化和计算控制 | 神经活动共享动力学、生物可塑性 |
| 多 session 神经轨道 | trial/block 划分的多 session 活动到 held-out likelihood/behavior | 共享潜在基底和隐状态门控的预测价值 | 若无干预，不能单独识别生物门控机制 |

三条轨道分别报告，禁止用迷宫、数独或 ARC 准确率填补神经证据缺口。

## 3. 能力隔离

公开任务对象只包含 support 输入/输出与 query 输入。query 目标由独立 `TargetStore` 保存：

- 训练能力只能为训练或验证任务生成监督视图；
- 测试阶段求解器只接收无目标的公开任务或候选集；
- 测试目标只能交给不可训练的 family evaluator；
- 改写测试目标不得改变预处理、模型参数、停止阈值或预测；
- 所有 augmentation 必须继承 base-task group，先分组再增强。

## 4. 匹配对照

主比较至少包括：

1. 分层低维控制器；
2. 无慢状态的 flat 控制器；
3. 分层控制器加折扣双线性 trace；
4. readout-only 或 frozen recurrent；
5. 隔离在 baseline 路径中的 GRU/BPTT；
6. family-specific symbolic/搜索基线和 oracle candidate ceiling。

同一 seed 内，各条件共享任务划分、基础特征、候选/动作集合、初始化中可共享的随机张量和硬计算上限。失败、超时和空预测都保留在分母中。模型选择只用训练/验证 group；测试 group 不用于选维度、停止阈值或超参数。

## 5. 主要终点

- ARC：每个 task 的全部 query 输出完全正确；
- 迷宫：路径合法率和最短路最优率，允许多条等长正确路径；
- 数独：保留线索、满足行/列/宫约束及整盘完全正确；
- 跨 family：task-level exact accuracy、失败率、计算步数、参数量、低维控制能量和状态/trace 维度；
- 统计单位：base task 或独立 seed，不能把 cell、step 或候选当独立重复。

## 6. 结论规则

核心命题只有在以下条件同时满足时才可记为 `support`：

- 分层低维控制器在 held-out base tasks 上优于匹配 flat/frozen 对照；
- 收益不是 candidate coverage 或预算不匹配造成；
- 绝对 exact accuracy 和相对差异分别报告；
- 至少一个 OOD split 上成立，并通过预注册的多重比较校正；
- 低维控制诊断与行为/预测改善同时存在。

若置信区间明确落在反方向，记为 `oppose`；样本不足、候选覆盖不足、数据来源或划分无法验证时记为 `inconclusive`。任何只呈现低秩矩阵、但不改善 held-out 任务或神经预测的结果不得计为支持。

## 7. Maze/Sudoku 公开数据冻结

`scripts/prepare_exp13_public_benchmarks.py` 是正式 Maze/Sudoku 数据的唯一准备入口：

- Sudoku 固定到 `wichtounet/sudoku_dataset@0db6de0`，核验 README 和官方 160/40 描述文件，只下载描述文件列出的 `image*.dat`；每题确定性求解并要求唯一解。正式派生集按模型可见 puzzle 内容寻址，跨官方 split 的重复组采用 test precedence，并且每个内容只保留一个确定性代表：200 条来源记录得到 76 个唯一 puzzle（48 train、28 non-OOD test），124 条重复排除和全部来源 receipt 均保留在 manifest。训练集与测试集的内容哈希交集必须为零。
- Maze 固定到 `albertoRodriguez97/MazeBench@a71a2d1`，核验数据卡和 annotations；对 110 张 PNG 使用固定 tile pipeline。reachable 样本只有在每一条公开 accepted path 都能在派生网格上合法到达同一 goal，且其长度等于独立 BFS 距离时才进入任务 JSONL；upstream-unreachable 和解析失败均保留在 manifest。
- MazeBench 上游仅提供 evaluation set，因此项目划分是派生的：优先把最大 grid size 留作 OOD test；若尺寸分组不可行，才使用固定 revision + task ID 的 SHA-256 划分。manifest 明确记录该事实，不把它表述为官方 split。
- 只有预注册且经 manifest 核验的 OOD test 可以进入核心 `support` 判定。Sudoku 的官方来源 split 不构成 OOD，因此其正式结果仅作 exploratory functional evidence；即使统计差异显著，也不得升级为核心支持。
- `data/structured/` 是可重建、git-ignored 的本地数据。正式配置必须把人工复核后的 manifest SHA-256 写入配置；实验再次核验 manifest、派生 JSONL、revision、license 和路径，并把完整 preparation receipt 嵌入 run provenance。占位哈希保持 fail-closed。

这些任务只提供结构化功能与 hybrid proposal-selection 证据；当前 fast/slow/trace 控制器只是受 HRM/CTM 启发的窄化抽象，不是两者的复现，也不能替代多 session 神经活动证据。
