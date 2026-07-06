# Minimal_computation Python 复现结果

- 数据集: `c_elegans`
- 输出神经元: MATLAB index `13`
- 数据形状: 128 neurons x 1600 time bins
- independent entropy: 0.444760 bits
- complete model: 当前 sweep 在 21 个输入时首次达到 complete-model 判据，约占所有其他神经元的 16.535%。
- run command: `python -B run_reproduction.py --dataset c_elegans --neuron 13 --max-inputs 32 --sweep 1,2,3,5,8,13,21,32`
- sweep inputs: `[1, 2, 3, 5, 8, 13, 21, 32]`

## 与论文结论的对应

论文主张是：神经元活动中的大量可预测结构可以由输出神经元与少数直接输入之间的依赖解释；不需要显式建模所有高阶输入交互即可得到高解释度。当前 Python 复现保留了这个核心结构：用最大熵/logistic neuron 模型匹配输出均值和已选输入的 pairwise correlation，并贪心加入最能降低残余相关误差的输入。

当前复现是单神经元、轻量 sweep，不等同于论文的全数据集全神经元统计。若 complete fraction 很小且 entropy 随输入数快速下降，则方向上支持论文的 minimal direct-dependence 结论；若没有达到 complete 判据，则说明当前 sweep/近似选择还不足以复现完整结论。

## 数值曲线

| inputs | entropy_bits | residual_corr_error | optimizer_complete | iterations |
|---:|---:|---:|---:|---:|
| 1 | 0.307302 | 6.245188 | True | 1453 |
| 2 | 0.218137 | 9.318200 | True | 1474 |
| 3 | 0.199728 | 7.220196 | True | 1336 |
| 5 | 0.188643 | 3.436755 | True | 1238 |
| 8 | 0.169967 | 2.443954 | True | 1341 |
| 13 | 0.086221 | 2.016619 | True | 1533 |
| 21 | 0.057658 | 0.618084 | True | 1345 |
| 32 | 0.044138 | 0.196429 | True | 1217 |

## 实现差异

- MATLAB 代码使用解析近似的 entropy-drop 二阶公式选择新输入；Python 版本第一步使用 pairwise MI，后续使用当前模型的归一化残余相关误差作为快速近似。
- MATLAB 原始脚本会继续二分搜索最小 complete input set；Python 版本当前报告 sweep 网格中的首次 complete 点。
- 当前目标是验证转换后的核心机制和单神经元复现，不是完全复制论文所有图和全数据集批处理。