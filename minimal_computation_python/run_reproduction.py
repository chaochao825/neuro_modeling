"""Run a lightweight Python reproduction of Minimal_computation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from minimal_computation.core import greedy_minimax_entropy
from minimal_computation.matv5 import load_activity


ROOT = Path(__file__).resolve().parent
ORIG = ROOT.parent / "minimal_computation_original"
DATASETS = {
    "hippocampus": ORIG / "data_mouse_hippocampus.mat",
    "visual_responding": ORIG / "data_mouse_visual_responding.mat",
    "visual_spontaneous": ORIG / "data_mouse_visual_spontaneous.mat",
    "c_elegans": ORIG / "data_C_elegans.mat",
}


def parse_sweep(text: str, max_inputs: int) -> list[int]:
    if text:
        vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    else:
        vals = [1, 2, 3, 5, 8, 13, 21, 30]
    vals = sorted({v for v in vals if 0 < v <= max_inputs})
    if max_inputs not in vals:
        vals.append(max_inputs)
    return vals


def plot_result(result: dict, out_path: Path) -> None:
    nums = np.asarray(result["nums_in"], dtype=float)
    ent = np.asarray(result["entropies"], dtype=float)
    err = np.asarray(result["residual_errors"], dtype=float)
    s0 = float(result["independent_entropy"])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].plot(nums, ent, marker="o", color="#4c78a8", linewidth=2, label="model entropy")
    axes[0].axhline(s0, color="#333333", linestyle="--", linewidth=1, label="independent entropy")
    axes[0].set_xlabel("Number of selected inputs")
    axes[0].set_ylabel("Conditional entropy (bits)")
    axes[0].grid(color="#dddddd", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(nums, err, marker="s", color="#e15759", linewidth=2)
    axes[1].axhline(2.0, color="#333333", linestyle="--", linewidth=1, label="complete threshold")
    axes[1].set_xlabel("Number of selected inputs")
    axes[1].set_ylabel("Max normalized corr. error")
    axes[1].set_yscale("log")
    axes[1].grid(color="#dddddd", linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8)

    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def write_report(result: dict, report_path: Path, dataset: str) -> None:
    complete_num = result["complete_num_inputs"]
    complete_fraction = result["complete_fraction"]
    if complete_num is None:
        complete_text = "当前 sweep 未达到 residual error < 2 的 complete-model 判据。"
    else:
        complete_text = (
            f"当前 sweep 在 {complete_num} 个输入时首次达到 complete-model 判据，"
            f"约占所有其他神经元的 {complete_fraction:.3%}。"
        )
    lines = [
        "# Minimal_computation Python 复现结果",
        "",
        f"- 数据集: `{dataset}`",
        f"- 输出神经元: MATLAB index `{result['neuron']}`",
        f"- 数据形状: {result['n_neurons']} neurons x {result['n_time']} time bins",
        f"- independent entropy: {result['independent_entropy']:.6f} bits",
        f"- complete model: {complete_text}",
        f"- run command: `python -B {' '.join(sys.argv)}`",
        f"- sweep inputs: `{result.get('run_config', {}).get('actual_sweep', result['nums_in'])}`",
        "",
        "## 与论文结论的对应",
        "",
        "论文主张是：神经元活动中的大量可预测结构可以由输出神经元与少数直接输入之间的依赖解释；不需要显式建模所有高阶输入交互即可得到高解释度。当前 Python 复现保留了这个核心结构：用最大熵/logistic neuron 模型匹配输出均值和已选输入的 pairwise correlation，并贪心加入最能降低残余相关误差的输入。",
        "",
        "当前复现是单神经元、轻量 sweep，不等同于论文的全数据集全神经元统计。若 complete fraction 很小且 entropy 随输入数快速下降，则方向上支持论文的 minimal direct-dependence 结论；若没有达到 complete 判据，则说明当前 sweep/近似选择还不足以复现完整结论。",
        "",
        "## 数值曲线",
        "",
        "| inputs | entropy_bits | residual_corr_error | optimizer_complete | iterations |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n, s, e, ok, it in zip(
        result["nums_in"],
        result["entropies"],
        result["residual_errors"],
        result["complete_flags"],
        result["iterations"],
    ):
        lines.append(f"| {n} | {s:.6f} | {e:.6f} | {ok} | {it} |")
    lines.extend(
        [
            "",
            "## 实现差异",
            "",
            "- MATLAB 代码使用解析近似的 entropy-drop 二阶公式选择新输入；Python 版本第一步使用 pairwise MI，后续使用当前模型的归一化残余相关误差作为快速近似。",
            "- MATLAB 原始脚本会继续二分搜索最小 complete input set；Python 版本当前报告 sweep 网格中的首次 complete 点。",
            "- 当前目标是验证转换后的核心机制和单神经元复现，不是完全复制论文所有图和全数据集批处理。",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="hippocampus")
    parser.add_argument("--neuron", type=int, default=13, help="MATLAB-style 1-based neuron index.")
    parser.add_argument("--max-inputs", type=int, default=13)
    parser.add_argument("--sweep", default="")
    parser.add_argument("--threshold", type=float, default=1e-6)
    args = parser.parse_args()

    if not DATASETS[args.dataset].exists():
        raise FileNotFoundError(
            f"Missing source data: {DATASETS[args.dataset]}. "
            "Clone https://github.com/ChrisWLynn/Minimal_computation as "
            "'minimal_computation_original' next to this project before rerunning sweeps."
        )
    activity = load_activity(DATASETS[args.dataset])
    sweep = parse_sweep(args.sweep, args.max_inputs)
    result = greedy_minimax_entropy(activity, args.neuron, sweep, threshold=args.threshold)
    result_dict = asdict(result)
    result_dict["dataset"] = args.dataset
    result_dict["source_mat"] = str(DATASETS[args.dataset])
    result_dict["run_config"] = {
        "dataset": args.dataset,
        "neuron": args.neuron,
        "max_inputs": args.max_inputs,
        "sweep_arg": args.sweep,
        "actual_sweep": sweep,
        "threshold": args.threshold,
        "corr_error_threshold": 2.0,
        "command": f"python -B {' '.join(sys.argv)}",
    }

    out_dir = ROOT / "results"
    fig_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    stem = f"{args.dataset}_neuron{args.neuron}_max{args.max_inputs}"
    json_path = out_dir / f"{stem}.json"
    report_path = out_dir / f"{stem}_report.md"
    fig_base = fig_dir / stem
    json_path.write_text(json.dumps(result_dict, indent=2, sort_keys=True), encoding="utf-8")
    write_report(result_dict, report_path, args.dataset)
    plot_result(result_dict, fig_base)
    print(json_path)
    print(report_path)
    print(fig_base.with_suffix(".png"))


if __name__ == "__main__":
    main()
