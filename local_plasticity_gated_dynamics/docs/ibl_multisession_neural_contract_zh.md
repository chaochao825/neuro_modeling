# IBL 多 session 神经活动验证契约（exp14）

## 目标与证据边界

`exp14` 用至少 20 个 session、5 个 animal 的公开 IBL spike counts 检验：在 session-specific observation matrices 之上，共享的低维 belief-gated dynamics 是否比 common dynamics 提高 held-out count likelihood，并以更少参数保留 session-full model 至少 90% 的 held-out gain。

该实验是观察性预测验证。即使结果为正，也不能单独识别 MD/PFC 生物门控、局部三因子可塑性或因果机制。ARC、迷宫和数独结果不得替代本轨道。

## 对 legacy exp06 的修正

`exp06` 保留为单 session 历史 pilot，不原地覆盖。`exp14` 必须修正：

1. stimulus-pre 主分析不得使用当前 trial 的 choice、reward、reaction time 或 stimulus-to-response wheel；只允许 pre-event camera ROI motion-energy proxy 和滞后一 trial 的行为量；该代理不得称为 DLC pose；
2. full-trial current-covariate residualization 单列为时间解释不合格的 sensitivity panel；
3. HMM belief 只由 stimulus-side 历史推断，fit 不接收 `probabilityLeft`，trial `t` belief 不使用 trial `t` stimulus；
4. filter、LDS/conditional dynamics 不在真实 context switch 处 reset；
5. 真实 `probabilityLeft` 仅用于预测冻结后的 context score/lead-lag 描述，不进入 learned gate；
6. split 使用连续 trial chunk/whole analysis block，任何时间点和 unit 均不得作为独立拆分单位；
7. latent dimension 在 outer-train 内做 nested chronological validation；test 不用于选维度、ridge、停止阈值或 nuisance 列；
8. 固定 sorting revision 和 good-unit QC，记录每个 PID、unit exclusion、dataset revision/hash 和失败 session。

## 模型族

三个模型共享完全相同的 train-only preprocessing 和 session observation terms：

- `common`：所有 session/context 共用一个 latent operator；
- `shared`：两个 operator 由 past-only belief 软混合，并跨 session 共享；
- `full`：每个 session 有各自的两个 operator。

为适配不同 session 的 unit 集合，先用共同 broad-region anchors 拟合 train-only shared latent basis，再拟合每个 session 的 latent/nuisance-to-log-rate observation matrix。主要 endpoint 是 next-bin conditional Poisson/NB likelihood；若实现仍是 Gaussian 或对 `log1p` 状态的 MSE，必须在名称和结论中明确，不能称为 count likelihood 或 Poisson LDS。

当前实现选择更窄但可审计的第一步：在 trial 内以当前 bin 的 train-only latent 投影预测下一 bin，再经 session-specific observation map 得到 Poisson rate。它报告包含 `gammaln(y+1)` 的 exact one-step conditional Poisson likelihood。它没有过程噪声分布、跨 trial latent filtering/smoothing 或 marginal sequence likelihood，因此代码、指标和报告均明确标记为“不是完整 latent Poisson LDS”。

`common/shared/full` 三族必须复用同一个 scaler、PCA basis 和 session observation matrix 指纹；只有 latent operator 的共享方式不同。nuisance 不先从 spike counts 中做连续残差化，因为那会破坏非负整数计数语义；primary past-safe nuisance 只作为 log-rate observation controls。连续残差只允许作为另列 Gaussian sensitivity，不能进入 Poisson 主结果。

## 数据和划分

- cohort 绑定 BWM repository commit、原 exp11 cohort manifest SHA-256、EID、animal、PID 和 ONE dataset revision；
- primary cohort 至少 20 sessions、5 animals，预先冻结，不按模型成功与否筛选；
- stimulus-pre 和 movement-pre 分开报告，stimulus-pre 是主要时间因果面板；
- 每个 session 的失败、unit 不足、context coverage 不足、下载失败和模型失败都保留；
- test trial 的 neural counts、behavior truth 或 context truth 变更不得改变 train preprocessing、latent dimension、belief checkpoint 或 model parameters。

当前冻结 acquisition panel 为 20 sessions/20 animals、35 PIDs，BWM commit `118fc36cb3602934466ad2c6087c2b3b441f9f1f`，spike sorting revision `2024-05-06`，good-unit threshold `>=1.0`。下载清单以 exact dataset UUID/revision/MD5/size 绑定；模型 formal loader 只接受离线 compact manifest、逐文件 SHA-256 和完整 20-row disposition，不允许 synthetic fallback。ROI motion energy 是 pre-event movement proxy，不得在正文中改称 DLC pose。

## 统计与结论

session 指标先在 animal 内聚合，再做 animal→session hierarchical bootstrap；fold、trial、time bin、unit 均为嵌套观测而非独立重复。主要注册比较：

1. shared vs common held-out NLL；
2. shared 保留至少 90% 的 full-vs-common gain；
3. shared 参数量少于 full；
4. stimulus-pre belief switch 相对 behavior bias switch 的 lead/lag，仅作描述；
5. movement-pre 和 full-trial covariate panel 为 sensitivity。

`support` 要求完整 cohort、train-only/nested-CV/hidden-gate gate 全部通过，并同时满足 likelihood、90% gain 与参数优势；反方向且置信区间排除零时为 `oppose`；下载/样本/模型/门控不可识别或区间跨零时为 `inconclusive`。
