# Minimal_computation Python 复现与论文结论对比

本报告基于当前工作区的 `minimal_computation_python` 结果，检查对 Chris W. Lynn 的 `Minimal_computation` MATLAB 代码的 Python 转写与轻量复现情况。它也对照原始大 goal 中“历史相关 + 局部耦合 + 近临界/幂律谱 + 振荡同步 + 能量约束”的框架边界：本子任务只覆盖“少数直接依赖/局部耦合可以解释输出神经元相关结构”的一部分，不覆盖时间历史 GLM、近临界谱、振荡同步和能量约束。

## 完成情况

已完成：

- 读取原始 MATLAB v5 `.mat` 二值活动矩阵，不依赖 SciPy。
- 转写二值 mutual information、最大熵/logistic 输出神经元拟合、贪心输入选择、模型熵和残余相关误差指标。
- 对 4 个原始数据集的 MATLAB index 13 神经元运行单神经元 sweep。
- 生成每组 JSON、Markdown 报告、PNG/PDF 曲线图，并新增跨数据集总览图。
- 主 JSON 均写入 `run_config`，记录实际 sweep、阈值、max_inputs 和生成命令。

主要限制：

- MATLAB 原实现使用解析的 entropy-drop Hessian 近似选择新增输入；当前 Python 版第一步使用 pairwise MI，后续使用归一化残余相关误差近似，因此输入顺序和完整模型点不保证逐项一致。
- MATLAB 脚本会在首次达到 complete 后二分搜索最小输入数；当前 Python 报告的是给定 sweep 网格中的首次 complete。
- 当前只验证每个数据集的一个输出神经元，不是论文中的全神经元、全数据集统计。

## 数值结果

| 数据集 | shape | independent entropy | max sweep entropy | entropy reduction | final residual error | complete 判据 |
|---|---:|---:|---:|---:|---:|---|
| C. elegans | 128 x 1600 | 0.444760 | 0.044138 | 90.08% | 0.196 | 达到：21 inputs，16.535% of other neurons |
| Hippocampus | 1485 x 70338 | 0.100668 | 0.050879 | 49.46% | 15.343 | 未在 30 inputs 内达到 |
| Visual spontaneous | 11445 x 4696 | 0.254997 | 0.151417 | 40.62% | 4.407 | 未在 30 inputs 内达到 |
| Visual responding | 11445 x 5166 | 0.196349 | 0.142396 | 27.48% | 4.884 | 未在 30 inputs 内达到 |

## 复现命令

```powershell
python -B run_reproduction.py --dataset c_elegans --neuron 13 --max-inputs 32 --sweep 1,2,3,5,8,13,21,32
python -B run_reproduction.py --dataset hippocampus --neuron 13 --max-inputs 30 --sweep 1,2,3,5,8,13,21,30
python -B run_reproduction.py --dataset visual_spontaneous --neuron 13 --max-inputs 30 --sweep 1,2,3,5,8,13,21,30
python -B run_reproduction.py --dataset visual_responding --neuron 13 --max-inputs 30 --sweep 1,2,3,5,8,13,21,30
python -B figures\plot_paper_comparison.py
```

C. elegans 使用显式 sweep `1,2,3,5,8,13,21,32`，不是空 `--sweep` 时的默认网格；这一点已记录在 `results/c_elegans_neuron13_max32.json` 的 `run_config.actual_sweep` 中。

可视化：

- `figures/paper_comparison_summary.png`
- `figures/paper_comparison_summary.pdf`
- `figures/plot_paper_comparison.py`

## 是否符合论文预期

部分一致。

C. elegans 单神经元结果最符合论文方向：随着输入增加，条件熵从 0.444760 bits 降至 0.044138 bits，并在 21 个输入时残余相关误差降到 2 以下。这支持“少数直接依赖可以解释输出神经元与群体的相关结构”的 minimal direct-dependence 结论，至少在该单神经元轻量复现中成立。

三个鼠数据集结果只显示部分一致：熵随输入数增加持续下降，说明已选输入确实解释了一部分输出不确定性；但在当前 30 输入 sweep 和近似选择器下，残余相关误差仍高于 complete 阈值。这个结果不能直接反驳论文，因为当前 Python 版尚未复刻 MATLAB 的解析 entropy-drop 选择和二分搜索，也没有运行到原 MATLAB 对视觉数据建议的更大 sweep 范围。

## 与原始大 goal 的关系

当前 `Minimal_computation` 复现可以作为大框架中的一个局部耦合/直接依赖模块，但不能替代整套神经群体动力学框架验证：

- 对 H1 有弱到中等相关性：它检验少数输入与输出神经元的直接统计依赖，但不是带时间历史项的 GLM/Hawkes。
- 对 H2 没有覆盖：没有 PCA/SVCA、DMD/VAR、Lyapunov 协方差或近临界谱检验。
- 对 H3 没有覆盖：没有 PSD、coherence、PLV、phase reset 或 complex DMD 的同步码判据。
- 对 H4 没有覆盖：没有 avalanche、branching ratio、finite-size scaling 或 dynamic range 检验。
- 对 H5 没有覆盖：没有信息/能耗/布线成本优化。

因此，当前最准确的结论是：`Minimal_computation` 子任务已经形成可运行 Python 复现，并在 C. elegans 单神经元上得到与论文方向一致的 complete-model 结果；在鼠 hippocampus/visual 的当前轻量配置下只得到熵下降的部分证据，尚未达到完整复现判据。大 goal 的完整验证仍应以 `neural_multiscale_tests` 的 H1-H5 pipeline 继续推进，并补真实公开数据拟合与行为状态控制。
