# Exp18 ARC 递归基线：协议与证据边界

## 为什么新增 Exp18

Exp13/15 是候选程序生成与候选选择审计。它们的正式 ARC-AGI-1
结果显示，正确答案只出现在 5/399 个候选集中，因此 1.253% 是 proposal
coverage 的硬上限。Exp18 保留这些否证性结果，另行建立一个直接生成
variable-shape grid 的 BPTT baseline；它不会替换、初始化或训练局部学习主模型。

## 三类 ARC 协议不得混报

1. `inductive`：evaluation task 在训练期间完全不可见。
2. `demo_tta`：共享模型只在 training tasks 预训练；evaluation task 仅使用公开
   demonstrations 做测试时适配，query output 始终由独立 scorer 保管。Exp18
   当前属于这一类。
3. `transductive_trm_parity`：像官方 TRM 一样，把 evaluation task 的
   demonstrations 和 task/augmentation ID 纳入训练。该协议可复现公开 TRM
   行为，但不能称为 unseen-task generalization。

任何报告都必须把这三类结果分表。ARC-AGI-2 training 与 ARC-AGI-1
evaluation 有已知重叠，使用 ARC-AGI-2 训练的模型不得把 ARC-AGI-1
evaluation 标记为独立 OOD 测试。

## Exp18 已实现的机制

- 30×30 以内的 variable-shape packing、二维 row/column position embedding、
  独立 height/width decoding；padding 不进入 cell loss 或评分。
- 共享 Transformer core 严格按照 `z_L -> z_H` 顺序更新；前
  `H_cycles - 1` 个 high cycles 无梯度，最后一个完整 high cycle 使用 BPTT；
  outer supervision 之间 carry detach。
- D4 与非背景颜色置换均可精确逆变换；测试时把不同增强和不同 refinement
  step 的输出逆变换后投票，最多保留两个候选。
- `demo_tta` 只接收 `PublicTask`；adaptation loss 只能读取 demonstrations，
  不能取得 query target capability。每个失败 task 仍保留在分母中。
- 递归模型与 single-state 对照共用 state dict、训练数据、batch order、
  optimizer budget、参数量和 nominal shared-core calls。
- task-level pass@1、pass@2、shape exact、pixel accuracy diagnostics、wall time、
  adaptation optimizer steps 与参数量全部逐任务保存。

## 与官方 TRM 的差异

当前实现是独立重写的 `micro-TRM-like ARC baseline`，不是 Samsung SAIL
发布的约 7M 参数模型或其 checkpoint。尚未实现或严格匹配：12-token
PAD/EOS crop、RoPE、StableMax、EMA、halt/Q head、随机平移、每 task 最多
1000 个 augmentation IDs，以及官方 100k-update/4×H100 训练规模。

仓库中的 `exp18_arc_recursive_arc1_canary.json` 是 20-task 真实数据 canary，
只检验端到端协议和量级；`exp18_arc_recursive_arc1_full.json` 登记了更接近
公开机制的 3-seed 运行，但在真正完成之前不得写成结果。官方 TRM 报告的
ARC-AGI-1/2 public pass@2 约 44.6%/7.8% 也属于强 task-conditioned 协议，
不能直接用来支持本项目的共享低维神经机制。

ARC-AGI-2 另有独立的 1120-file byte manifest 和 canary 配置；它不复用
ARC-AGI-1 manifest。两个数据集的分数、训练来源和 OOD 标签必须分别报告，
ARC-AGI-2-trained 模型不得反向报告 ARC-AGI-1 evaluation 为独立 OOD。

## 已完成的真实数据 canary

seed 3000 的冻结 canary 已分别运行 20 个 ARC-AGI-1 和 20 个 ARC-AGI-2
evaluation tasks，所有任务均完成且无 invalid/failure。三个条件在两个数据集
上的 pass@1/pass@2 都为 0；注册的递归减单状态 pass@2 差值为 0。ARC-AGI-1
三个条件的 shape exact 均为 5%；ARC-AGI-2 中递归与 no-TTA 为 5%，单状态
为 0%。因此结论是 `inconclusive`，不能支持递归状态优势。递归 TTA 与 no-TTA
的 endpoint 完全相同，也说明当前单 epoch TTA 预算没有形成可测行为增益。

## 核心判据

Exp18 只给主项目提供强 BPTT 对照。只有后续低维 belief-gated/local-learning
模型在 matched compute/update budget 下提高 held-out task pass@2，并且优于
single-state、frozen refinement 与 shuffled gate，才可标记为 `support`。
只有低 matrix rank、低 activity dimension 或训练 demo fit，不能算支持。

## 一手参考

- TRM（MIT）：https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- ARC-AGI-2 数据与评分协议（Apache-2.0）：https://github.com/arcprize/ARC-AGI-2
- ARC Prize 2025 结果与方法分析：https://arcprize.org/blog/arc-prize-2025-results-analysis
- ARC Prize 对 HRM/TRM 增益来源的复核：https://arcprize.org/blog/hrm-analysis
- SR²（Apache-2.0）：https://github.com/dengyl20/SR2
- CompressARC（MIT）：https://github.com/iliao2345/CompressARC
- SOAR（MIT）：https://github.com/flowersteam/SOAR
