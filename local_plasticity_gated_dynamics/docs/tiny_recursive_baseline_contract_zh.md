# Exp16 微型递归推理基线契约

## 证据边界

Exp16 是隔离在 `src/baselines/` 下的全局 autograd/BPTT 计算基线，不是局部三因子学习模型，也不能向局部模型传递 checkpoint。它参考但不复制：

- Sapient HRM 官方代码快照 [`ac15626`](https://github.com/sapientinc/HRM/tree/ac15626f8db096a63c775b84c9dc868776a6feda) 与论文 [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)；
- Samsung SAIL TRM 官方归档快照 [`c011037`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/tree/c01103738605ba39d1430519b1ee0c62f4c707f8) 与论文 [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)。

本仓库实现为独立重写，不导入上游源文件。HRM 为 Apache-2.0，TRM 为 MIT；上游仓库和论文仅作为算法行为与命名边界的一手依据。

## 已复现的最小机制

`micro_trm_bptt` 维护答案状态 `y` 和推理状态 `z`，使用同一个共享非因果 attention/MLP core，按以下次序递归：

```text
z <- core(z, y + x)
y <- core(y, z)
```

每个 supervision segment 的前 `H_cycles - 1` 个 outer cycles 在 `no_grad` 下运行，最后一个 outer cycle 内的全部 `L_cycles` 进入 BPTT；segment 之间的 `y/z` carry 显式 detach。评估固定运行配置的全部 supervision steps，不实现论文式或自定义 early halt。

对照 `single_state_core_call_matched` 使用相同模块、初始 state dict、训练/验证数组、epoch permutation、优化步数和名义 core-call 次数，只让一个答案状态动态更新。这里的 matched 不代表 backward FLOPs、显存、GPU 时间、墙钟时间或物理能耗相等。

## 有意保留的差异

该 micro 实现没有声称复现官方规模或分数，且明确省略或改变：

- ACT/Q-learning、StableMax、puzzle identifier、EMA；
- 官方 RMSNorm、SwiGLU、RoPE 和 512 hidden / 2-layer 规模；
- Sudoku-Extreme 1K 与每题 1000 增强；
- 官方全 token loss；本实现只对 blank cells 计算 loss；
- 官方 ARC 的 transductive demonstrations、puzzle embedding、1000 增强投票；
- 原始预测可能改写 clue；正式 exact/valid 评分前会把公开 clue clamp 回输入，同时另报 unclamped clue accuracy。

因此可检验的命题仅是：在同一小型 Sudoku 接口上，交替双状态 schedule 是否优于指定的单动态状态 recurrent 对照。它不是“HRM/TRM 已复现”，也不是推理、能效或生物机制的一般优势。

## 数据与统计

- smoke 只使用可审计的合成 Sudoku fixture，结论固定为 `inconclusive`；
- public formal config 指向经清洗的 48-train / 28-test Sudoku V2，test 为 `non_ood`，不是 Sudoku-Extreme；
- inner train/validation 按 source、augmentation、public-content 三类标识的连通分量拆分；
- 所有 Sudoku symmetry augmentation 仅作用于 inner train；
- test target 不进入训练 API，只在冻结预测后通过 `TargetStore.score` 使用；
- 跨训练重复的统计单位是 seed，task/source-group bootstrap 仅作单 seed 内描述。

当前 publisher 是 pilot-only：它保留显式传入的全部尝试、按 seed 选择最新尝试作描述、拒绝覆盖，并绑定 raw/manifest 哈希；但在 canonical all-attempt inventory 与 raw task-level 重算完成前，`formal_claim_eligible` 永远为 false，Exp16 不进入全局 `results/summary.csv`。
