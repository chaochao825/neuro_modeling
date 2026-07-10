# Minimal_computation Python 复现结果

- 数据集: `c_elegans`
- 输出神经元: MATLAB index `13`
- 数据形状: 128 neurons x 1600 time bins
- independent entropy: 0.444760 bits
- complete model: 当前 sweep 在 7 个输入时首次达到 complete-model 判据，约占所有其他神经元的 5.512%。
- 达到判据的输入（MATLAB 1-based IDs）: `[74, 34, 3, 1, 123, 54, 120]`
- selector: `schur_entropy_drop`
- initialization: `matlab_reset`
- completion mode: `matlab_residual`
- run command: `python -B run_reproduction.py --dataset c_elegans --neuron 13 --max-inputs 32 --selector schur_entropy_drop --completion-mode matlab_residual --failure-selection matlab_last`
- sweep inputs: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]`
- publication snapshot: machine paths were replaced by `${REPO_ROOT}`; original config fingerprint `f6cfd38f...`, published path-sanitized payload fingerprint `67fc6570...`

## 与论文结论的对应

论文主张是：神经元活动中的大量可预测结构可以由输出神经元与少数直接输入之间的依赖解释；不需要显式建模所有高阶输入交互即可得到高解释度。当前 Python 复现保留了这个核心结构：用最大熵/logistic neuron 模型匹配输出均值和已选输入的 pairwise correlation，并贪心加入最能降低残余相关误差的输入。

当前复现是单神经元分析，不等同于论文的全数据集全神经元统计。MATLAB-parity 模式按未选正相关输入的归一化残差阈值判定 n*，并单独报告优化器是否收敛；更保守的 Python strict 模式才同时要求二者。

## 数值曲线

| inputs | entropy_bits | residual_corr_error | optimizer_complete | criterion_complete | phase | iterations |
|---:|---:|---:|---:|---:|:---|---:|
| 1 | 0.307231 | 6.245467 | True | False | coarse | 868 |
| 2 | 0.231804 | 6.981986 | True | False | coarse | 1229 |
| 3 | 0.165309 | 5.972544 | True | False | coarse | 1176 |
| 4 | 0.128362 | 5.372195 | True | False | coarse | 1207 |
| 5 | 0.106071 | 3.229213 | True | False | coarse | 1270 |
| 6 | 0.094879 | 3.083513 | True | False | coarse | 1185 |
| 7 | 0.079960 | 1.504544 | True | True | coarse | 1249 |
| 8 | 0.073347 | 1.394338 | True | True | coarse | 1115 |
| 9 | 0.069852 | 1.395073 | True | True | coarse | 1171 |
| 10 | 0.065996 | 1.201628 | True | True | coarse | 1120 |
| 11 | 0.063874 | 1.028041 | True | True | coarse | 1033 |
| 12 | 0.060674 | 1.288149 | True | True | coarse | 1112 |
| 13 | 0.058326 | 1.237568 | True | True | coarse | 1149 |
| 14 | 0.056945 | 1.159331 | True | True | coarse | 1102 |
| 15 | 0.055878 | 1.041312 | True | True | coarse | 1149 |
| 16 | 0.054815 | 1.412307 | True | True | coarse | 1139 |
| 17 | 0.052716 | 0.596010 | True | True | coarse | 1126 |
| 18 | 0.051324 | 0.578523 | True | True | coarse | 1153 |
| 19 | 0.049770 | 0.393607 | True | True | coarse | 1182 |
| 20 | 0.049033 | 0.474177 | True | True | coarse | 1145 |
| 21 | 0.048247 | 0.693569 | True | True | coarse | 1139 |
| 22 | 0.047414 | 0.715835 | True | True | coarse | 1073 |
| 23 | 0.046114 | 0.542424 | True | True | coarse | 1084 |
| 24 | 0.045419 | 0.565885 | True | True | coarse | 1076 |
| 25 | 0.044559 | 0.503721 | True | True | coarse | 1092 |
| 26 | 0.044086 | 0.499359 | True | True | coarse | 1130 |
| 27 | 0.043634 | 0.594870 | True | True | coarse | 1050 |
| 28 | 0.042817 | 0.605173 | True | True | coarse | 1070 |
| 29 | 0.042094 | 0.279691 | True | True | coarse | 1098 |
| 30 | 0.041425 | 0.067118 | True | True | coarse | 1134 |
| 31 | 0.041252 | 0.072852 | True | True | coarse | 1085 |
| 32 | 0.040973 | 0.099430 | True | True | coarse | 1101 |

## 实现差异

- 默认 `schur_entropy_drop` 在候选块上计算与 MATLAB 全 Hessian 相同的 Schur entropy-drop；不会构造 N x N Hessian。
- `residual_approximation` 显式保留旧 Python 归一化残差选择器，仅作为 baseline。
- 默认每次拟合均采用 MATLAB 的 independent-bias/zero-weight 重置初始化，并在首次 coarse complete 后二分细化最小输入数。
- 当前目标仍是方法等价的单神经元复现，不代表论文的全神经元群体统计已完成。
