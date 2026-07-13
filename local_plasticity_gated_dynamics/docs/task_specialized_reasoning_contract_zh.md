# Exp15 任务专用推理验证契约

## 定位

Exp15 参考现有类脑推理工作“按任务结构设计模型”的原则，但不复刻其模型和训练方法：

- [Pathway BDH](https://github.com/pathwaycom/bdh) 的公开基线展示稀疏正激活、乘性交互和状态空间式计算；其 README 同时明确，公开仓库不能直接复现内部报告的 Sudoku Extreme 97.4% 结果。因此本项目不引用该数字作为 baseline，也不声称实现 BDH。
- [Sapient HRM](https://github.com/sapientinc/HRM) 使用快/慢层级状态、重复计算周期和 adaptive halting，并为 ARC 使用固定网格编码及数据增强。本项目只借用“慢速任务假设 + 快速局部执行”的接口思想，不复制 HRM block、ACT/Q-learning、puzzle embedding 或训练流程。

## Sudoku adapter

`SudokuConstraintDynamics` 将每个 cell-digit 对表示为稀疏非负候选活动，并仅通过同行、同列和同宫的局部抑制进行更新：

1. clue clamp；
2. peer elimination；
3. naked single 与 hidden single；
4. 固定点或预算耗尽时停止。

默认 `branch_budget=0`，因此纯局部动力学和显式离散搜索不会混淆。`sudoku_local_bounded_branch` 是单独条件；每次分支、传播步数和局部 assignment 均写入 receipt。对可能多解的 smoke puzzle，`valid_constraint_solution` 与数据集给定答案的 `exact` 分开报告。

## ARC adapter

`ARCSlowFastProgramReasoner` 只接收 demonstrations 和 query inputs。现有 target-free proposal library 生成候选程序后：

1. 慢状态在 geometry、rescale、extract、identity 算子族上累积 belief；
2. 每个算子族用有界 top-k log-sum-exp 聚合多个 demonstration-grounded
   程序的证据，快状态再结合族 belief 与程序局部证据进行选择；
3. belief margin 达阈值时可提前停止。

测试 query target 不进入 proposal、belief、halting 或选择。该方法仍依赖有限 proposal library，因此不是端到端 ARC solver；proposal coverage 必须与 selector accuracy 分开解释。

## 统计与结论门

- ARC/Sudoku 的统计单位是独立 task/source group，不把 cell、token 或内部 step 当独立重复；
- Sudoku 正式数据复用 Exp13 的 revision、license、manifest、split 与 duplicate fail-closed 校验；ARC 正式配置逐文件验证 800 个 JSON、LICENSE、split count、固定 revision、获取清单和独立 validation receipt，任一不一致即拒绝运行；
- 保存逐 task 原始 score、public fingerprint、内部计算 receipt、所有失败条件；
- ARC 的 `arc_slow_fast_program` 与 `arc_flat_program_matched` 在每个 condition
  内确定性重建等价候选批次，并逐 task 强制匹配候选 fingerprint、coverage 与
  固定 charged budget；它们不是同一个 Python 对象。charged budget 是审计用
  抽象操作代理，不代表 FLOPs、墙钟时间或能耗；
- 注册比较要求 OOD、数据来源、候选配对、charged budget 配对和至少 90% candidate coverage 全部通过，才允许根据配对 bootstrap CI 与 Holm 校正结果给出 support/oppose；否则为 `inconclusive`；
- Exp15 不能替代 IBL 多 session 神经活动、局部可塑性或共享潜在动力学证据。

## 当前正式 ARC 结果

干净提交 `cbec277503d02844729d8fea5648a9e34e2ce44b` 在排除已注册的跨 split
重复项后评估 399 个 ARC-AGI-1 evaluation tasks。slow/fast 与 flat matched
各精确解出 1/399（0.2506%，95% source-group CI 0–0.7519%）；配对差为 0
个百分点（95% CI [0, 0]，Holm `p=1`）。候选库只覆盖 5/399（1.2531%），
未通过 90% 门，因此 `core_claim_eligible=false`，结论为 `inconclusive`。
该结果修复了来源可追溯性，但不支持层级选择优势，也不是有竞争力的 ARC solver。
