# 历史 residual-approximation 对比（已归档）

本报告及 `paper_comparison_summary` 图只读取四个旧 fixed-name JSON。它们在
block-Schur 实现之前生成，使用 residual-ranking 近似和稀疏 coarse sweep；不
是当前 MATLAB-parity 结果。

历史结果包括：C. elegans neuron 13 在 coarse sweep 的 21 inputs 首次通过，
三个鼠数据集在 30 inputs 内未通过。该 21-input 数字已经被当前
block-Schur + MATLAB residual-control-flow 结果取代：同一输出神经元在 7 个
MATLAB 1-based inputs `[74,34,3,1,123,54,120]` 达到判据。

方法边界：

- 这是 equal-time conditional dependency `P(y_t | x_t)`，不是 temporal
  history、anatomical locality、latent rank、future dynamics 或因果连接；
- 单个输出神经元结果不能代表论文的全神经元/全数据集统计；
- 旧鼠数据结果仍保留作 approximation baseline，不能标成 Schur 复现。

当前结果见
`c_elegans_matlab_schur_neuron13_max32_report.md`，完整方法审计见仓库根目录
`docs/integrated_method_audit_zh.md`。
