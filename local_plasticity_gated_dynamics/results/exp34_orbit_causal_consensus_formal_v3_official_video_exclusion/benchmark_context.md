# Exp34 external benchmark context

The paired Exp34 mechanism analysis uses user as the independent unit. For
descriptive benchmark context only, the same result also reports the
official-style mean over task/video frame accuracies: **67.43%**.

| Published ORBIT condition | Encoder | Published test frame accuracy | Exp34 minus reference |
|---|---|---:|---:|
| FineTuner | EfficientNet-B0 | 64.57% | +2.86 pp |
| FineTuner + FiLM | EfficientNet-B0 | 66.63% | +0.80 pp |
| cosine ProtoNet | EfficientNet-B0 | 67.48% | -0.05 pp |
| CNAPs | EfficientNet-B0 | 67.68% | -0.25 pp |
| ProtoNet | EfficientNet-B0 | 67.91% | -0.48 pp |
| cosine ProtoNet | ViT-B/32 | 75.38% | -7.95 pp |

The reference values are copied from the
[official ORBIT repository](https://github.com/microsoft/ORBIT-Dataset), whose
protocol accompanies the
[ORBIT ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Massiceti_ORBIT_A_Real-World_Few-Shot_Dataset_for_Teachable_Object_Recognition_ICCV_2021_paper.html).
They are independent published runs, not rerun on Exp34's frozen feature tape.
Encoder initialization, episodic representation training, and optimization
budgets differ. The table therefore does **not** establish superiority,
non-inferiority, or state of the art. It shows that the current frozen
EfficientNet-B0 mechanism audit is competitive with older same-backbone
reference points while remaining far below the stronger published ViT result.

The next competitive test must rerun the controller and baselines on one
shared official task tape with the same stronger backbone and training budget.
Until then, Exp34 supports only its paired causal actuator-selection contrast.
