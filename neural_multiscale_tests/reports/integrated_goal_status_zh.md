# 历史整合快照：single-seed synthetic calibration

本文件及配套 `integrated_goal_status` 图是 2026-07-07 的历史产物。它们读取
seed 7 的缩小合成 H1--H5 输出和旧 fixed-name Minimal-computation JSON，因此不
代表当前 Schur 复现或正式真实数据结果。

解释边界：

- H1--H5 的 `strong/weak` 是 synthetic pipeline calibration，不是生物统计推断；
- 图中的 C. elegans 21-input 是旧 `residual_approximation`，当前 block-Schur
  residual 判据在 7 个 MATLAB 1-based 输入
  `[74,34,3,1,123,54,120]` 达到；
- Minimal-computation 拟合 equal-time `P(y_t | x_t)`。它不检验 temporal
  history，也没有 anatomical coordinates，不能补强 H1 的 history/locality；
- 当前 visual shared-dynamics formal panel 已单独完成，但只有一个 recording
  pair，不能把 neuron、fold 或 computational seed 当生物重复。

当前综合结论请见仓库根目录的 `docs/integrated_method_audit_zh.md`，当前真实
数据报告请见 `shared_dynamics_real_data/results/report.md`。本文件保留仅用于
provenance。
