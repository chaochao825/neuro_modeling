# 历史方法与结果快照（已归档）

本文原先记录 2026-07-07 的 single-seed H1--H5 quick calibration，以及旧
`residual_approximation` Minimal-computation 结果。它已被 2026-07-11 的方法审计
和正式实验取代，不再作为当前结果来源。

历史数字的边界：

- H1--H5 的 `strong/weak` 来自 seed 7、缩小规模的合成校准，不是统计推断；
- C. elegans 的 21-input 数字来自 residual-ranking 近似，不是 MATLAB
  Hessian/Schur 选择器；
- equal-time direct dependency 不能补强 temporal history、anatomical locality、
  latent rank 或因果连接；
- 旧报告声称“真实数据尚未运行”已经过时：当前 visual formal panel 已完成
  2,160/2,160 个计划 cell，但只有一个 recording pair，正式结论仍无法判定。

请使用以下当前证据：

- `docs/integrated_method_audit_zh.md`：方法整合、泄漏/统计单位审计与三分类结论；
- `local_plasticity_gated_dynamics/results/report.md`：局部可塑性核心项目正式结果；
- `minimal_computation_python/results/c_elegans_matlab_schur_neuron13_max32_report.md`：
  C. elegans neuron 13 的 block-Schur 7-input 结果；
- `shared_dynamics_real_data/results/report.md`：20-seed、contiguous-block 真实数据面板；
- `REPRODUCE.md`：Python 3.11 环境和精确复现命令。

旧的详细 JSON、Markdown 和图仍保留在 `neural_multiscale_tests/` 与
`minimal_computation_python/results/` 中，仅用于 provenance 和方法比较。
