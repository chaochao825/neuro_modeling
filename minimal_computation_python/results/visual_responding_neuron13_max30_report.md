# Minimal_computation Python 复现结果

- 数据集: `visual_responding`
- 输出神经元: MATLAB index `13`
- 数据形状: 11445 neurons x 5166 time bins
- independent entropy: 0.196349 bits
- complete model: 当前 sweep 未达到 residual error < 2 的 complete-model 判据。
- run command: `python -B run_reproduction.py --dataset visual_responding --neuron 13 --max-inputs 30 --sweep 1,2,3,5,8,13,21,30`
- sweep inputs: `[1, 2, 3, 5, 8, 13, 21, 30]`

## 与论文结论的对应

论文主张是：神经元活动中的大量可预测结构可以由输出神经元与少数直接输入之间的依赖解释；不需要显式建模所有高阶输入交互即可得到高解释度。当前 Python 复现保留了这个核心结构：用最大熵/logistic neuron 模型匹配输出均值和已选输入的 pairwise correlation，并贪心加入最能降低残余相关误差的输入。

当前复现是单神经元、轻量 sweep，不等同于论文的全数据集全神经元统计。若 complete fraction 很小且 entropy 随输入数快速下降，则方向上支持论文的 minimal direct-dependence 结论；若没有达到 complete 判据，则说明当前 sweep/近似选择还不足以复现完整结论。

## 数值曲线

| inputs | entropy_bits | residual_corr_error | optimizer_complete | iterations |
|---:|---:|---:|---:|---:|
| 1 | 0.185269 | 8.163307 | True | 1085 |
| 2 | 0.183473 | 7.774903 | True | 532 |
| 3 | 0.181757 | 7.862701 | True | 676 |
| 5 | 0.178999 | 7.390928 | True | 857 |
| 8 | 0.174289 | 6.160427 | True | 1092 |
| 13 | 0.167878 | 5.557890 | True | 1038 |
| 21 | 0.151700 | 5.495058 | True | 1322 |
| 30 | 0.142396 | 4.883593 | True | 1100 |

## 实现差异

- MATLAB 代码使用解析近似的 entropy-drop 二阶公式选择新输入；Python 版本第一步使用 pairwise MI，后续使用当前模型的归一化残余相关误差作为快速近似。
- MATLAB 原始脚本会继续二分搜索最小 complete input set；Python 版本当前报告 sweep 网格中的首次 complete 点。
- 当前目标是验证转换后的核心机制和单神经元复现，不是完全复制论文所有图和全数据集批处理。