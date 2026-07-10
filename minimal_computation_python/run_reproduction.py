"""Run a lightweight Python reproduction of Minimal_computation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
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

# These values follow the comments and sweep grids in the upstream MATLAB
# release.  ``max_inputs`` still caps every grid for lightweight runs.
DATASET_CONFIGS = {
    "hippocampus": {
        "optimizer_threshold": 1e-6,
        "corr_error_threshold": 2.0,
        "sweep_family": "hippocampus_matlab",
    },
    "visual_responding": {
        "optimizer_threshold": 1e-5,
        "corr_error_threshold": 2.0,
        "sweep_family": "visual_matlab",
    },
    "visual_spontaneous": {
        "optimizer_threshold": 1e-5,
        "corr_error_threshold": 2.0,
        "sweep_family": "visual_matlab",
    },
    "c_elegans": {
        "optimizer_threshold": 1e-5,
        "corr_error_threshold": 2.0,
        "sweep_family": "c_elegans_matlab",
    },
}


def file_sha256(path: Path, *, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def run_config_fingerprint(config: dict) -> str:
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def default_sweep(family: str, max_inputs: int) -> list[int]:
    if family == "hippocampus_matlab":
        values = list(range(1, 11)) + list(range(15, 51, 5)) + list(range(60, 101, 10))
        values += list(range(150, 1401, 50))
    elif family == "visual_matlab":
        values = list(range(1, 11)) + list(range(15, 51, 5)) + list(range(60, 101, 10))
        values += list(range(150, 501, 50)) + list(range(600, 2001, 200))
        values += list(range(3000, 11001, 1000))
    elif family == "c_elegans_matlab":
        values = list(range(1, max_inputs + 1))
    else:
        raise ValueError(f"Unknown sweep family: {family}")
    return sorted({value for value in values if 0 < value <= max_inputs})


def parse_sweep(text: str, max_inputs: int, family: str = "hippocampus_matlab") -> list[int]:
    if text:
        vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    else:
        vals = default_sweep(family, max_inputs)
    vals = sorted({v for v in vals if 0 < v <= max_inputs})
    if max_inputs not in vals:
        vals.append(max_inputs)
    return vals


def plot_result(result: dict, out_path: Path) -> None:
    nums = np.asarray(result["nums_in"], dtype=float)
    ent = np.asarray(result["entropies"], dtype=float)
    err = np.asarray(result["residual_errors"], dtype=float)
    s0 = float(result["independent_entropy"])
    corr_threshold = float(result.get("corr_error_threshold", 2.0))

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
    axes[1].axhline(
        corr_threshold,
        color="#333333",
        linestyle="--",
        linewidth=1,
        label="complete threshold",
    )
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
    completion_mode = result.get("completion_mode", "legacy_unspecified")
    criterion_text = (
        "residual error 严格低于阈值（MATLAB parity；优化收敛单独报告）"
        if completion_mode == "matlab_residual"
        else "优化收敛且 residual error 严格低于阈值"
    )
    if complete_num is None:
        complete_text = f"当前 sweep 未达到 {criterion_text} 的 complete-model 判据。"
        selected_text = "未达到判据，因此没有 complete-model 输入集合。"
    else:
        complete_text = (
            f"当前 sweep 在 {complete_num} 个输入时首次达到 complete-model 判据，"
            f"约占所有其他神经元的 {complete_fraction:.3%}。"
        )
        selected_text = (
            "达到判据的输入（MATLAB 1-based IDs）: "
            f"`{result['selected_inputs_complete']}`"
        )
    lines = [
        "# Minimal_computation Python 复现结果",
        "",
        f"- 数据集: `{dataset}`",
        f"- 输出神经元: MATLAB index `{result['neuron']}`",
        f"- 数据形状: {result['n_neurons']} neurons x {result['n_time']} time bins",
        f"- independent entropy: {result['independent_entropy']:.6f} bits",
        f"- complete model: {complete_text}",
        f"- {selected_text}",
        f"- selector: `{result.get('selector', 'legacy_unspecified')}`",
        f"- initialization: `{result.get('initialization', 'legacy_unspecified')}`",
        f"- completion mode: `{result.get('completion_mode', 'legacy_unspecified')}`",
        f"- run command: `python -B {' '.join(sys.argv)}`",
        f"- sweep inputs: `{result.get('run_config', {}).get('actual_sweep', result['nums_in'])}`",
        "",
        "## 与论文结论的对应",
        "",
        "论文主张是：神经元活动中的大量可预测结构可以由输出神经元与少数直接输入之间的依赖解释；不需要显式建模所有高阶输入交互即可得到高解释度。当前 Python 复现保留了这个核心结构：用最大熵/logistic neuron 模型匹配输出均值和已选输入的 pairwise correlation，并贪心加入最能降低残余相关误差的输入。",
        "",
        "当前复现是单神经元分析，不等同于论文的全数据集全神经元统计。MATLAB-parity 模式按未选正相关输入的归一化残差阈值判定 n*，并单独报告优化器是否收敛；更保守的 Python strict 模式才同时要求二者。",
        "",
        "## 数值曲线",
        "",
        "| inputs | entropy_bits | residual_corr_error | optimizer_complete | criterion_complete | phase | iterations |",
        "|---:|---:|---:|---:|---:|:---|---:|",
    ]
    for n, s, e, ok, strict_ok, phase, it in zip(
        result["nums_in"],
        result["entropies"],
        result["residual_errors"],
        result["complete_flags"],
        result.get("criterion_complete_flags", result["complete_flags"]),
        result.get("evaluation_phases", ["coarse"] * len(result["nums_in"])),
        result["iterations"],
    ):
        lines.append(f"| {n} | {s:.6f} | {e:.6f} | {ok} | {strict_ok} | {phase} | {it} |")
    lines.extend(
        [
            "",
            "## 实现差异",
            "",
            "- 默认 `schur_entropy_drop` 在候选块上计算与 MATLAB 全 Hessian 相同的 Schur entropy-drop；不会构造 N x N Hessian。",
            "- `residual_approximation` 显式保留旧 Python 归一化残差选择器，仅作为 baseline。",
            "- 默认每次拟合均采用 MATLAB 的 independent-bias/zero-weight 重置初始化，并在首次 coarse complete 后二分细化最小输入数。",
            "- 当前目标仍是方法等价的单神经元复现，不代表论文的全神经元群体统计已完成。",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="hippocampus")
    parser.add_argument("--neuron", type=int, default=13, help="MATLAB-style 1-based neuron index.")
    parser.add_argument("--max-inputs", type=int, default=13)
    parser.add_argument("--sweep", default="")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optimizer threshold; defaults to the dataset-specific MATLAB value.",
    )
    parser.add_argument("--corr-error-threshold", type=float, default=None)
    parser.add_argument(
        "--selector",
        choices=("schur_entropy_drop", "residual_approximation"),
        default="schur_entropy_drop",
    )
    parser.add_argument(
        "--initialization",
        choices=("matlab_reset", "warm_start"),
        default="matlab_reset",
    )
    parser.add_argument("--candidate-block-size", type=int, default=256)
    parser.add_argument(
        "--completion-mode",
        choices=("matlab_residual", "strict_optimizer_and_residual"),
        default="matlab_residual",
    )
    parser.add_argument(
        "--failure-selection",
        choices=("matlab_last", "best_error"),
        default="matlab_last",
    )
    parser.add_argument(
        "--no-binary-search",
        action="store_true",
        help="Disable coarse-bracket refinement (diagnostic only).",
    )
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    dataset_config = DATASET_CONFIGS[args.dataset]
    optimizer_threshold = (
        float(dataset_config["optimizer_threshold"])
        if args.threshold is None
        else float(args.threshold)
    )
    corr_error_threshold = (
        float(dataset_config["corr_error_threshold"])
        if args.corr_error_threshold is None
        else float(args.corr_error_threshold)
    )
    sweep = parse_sweep(args.sweep, args.max_inputs, str(dataset_config["sweep_family"]))
    source_path = DATASETS[args.dataset]
    run_config = {
        "dataset": args.dataset,
        "neuron": args.neuron,
        "max_inputs": args.max_inputs,
        "sweep_arg": args.sweep,
        "actual_sweep": sweep,
        "sweep_family": dataset_config["sweep_family"],
        "optimizer_threshold": optimizer_threshold,
        "corr_error_threshold": corr_error_threshold,
        "selector": args.selector,
        "initialization": args.initialization,
        "candidate_block_size": args.candidate_block_size,
        "binary_search": not args.no_binary_search,
        "completion_mode": args.completion_mode,
        "failure_selection": args.failure_selection,
        "algorithm_version": "matlab-schur-block-v1",
        "dataset_defaults": dataset_config,
        "seed": None,
        "deterministic": True,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "source_path": str(source_path.resolve()),
        "source_exists": source_path.is_file(),
        "source_sha256": file_sha256(source_path) if source_path.is_file() else None,
        "command": f"python -B {' '.join(sys.argv)}",
    }
    fingerprint = run_config_fingerprint(run_config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = args.results_root / "runs" / f"{stamp}_{fingerprint[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config["config_fingerprint"] = fingerprint
    run_config["run_id"] = run_dir.name
    write_json(run_dir / "config.json", run_config)
    started = time.perf_counter()
    status = {
        "status": "running",
        "run_id": run_dir.name,
        "config_fingerprint": fingerprint,
    }
    write_json(run_dir / "status.json", status)
    try:
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing source data: {source_path}. Clone "
                "https://github.com/ChrisWLynn/Minimal_computation as "
                "'minimal_computation_original' next to this project."
            )
        activity = load_activity(source_path)
        result = greedy_minimax_entropy(
            activity,
            args.neuron,
            sweep,
            threshold=optimizer_threshold,
            corr_error_threshold=corr_error_threshold,
            selector=args.selector,
            initialization=args.initialization,
            candidate_block_size=args.candidate_block_size,
            refine_complete=not args.no_binary_search,
            completion_mode=args.completion_mode,
            failure_selection=args.failure_selection,
        )
        result_dict = asdict(result)
        result_dict["status"] = "complete"
        result_dict["dataset"] = args.dataset
        result_dict["source_mat"] = str(source_path.resolve())
        result_dict["run_config"] = run_config
        write_json(run_dir / "metrics.json", result_dict)
        write_report(result_dict, run_dir / "report.md", args.dataset)
        plot_result(result_dict, run_dir / "figure")
        status.update(
            {
                "status": "complete",
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": "",
                "error_message": "",
            }
        )
        (run_dir / "run.log").write_text(
            "complete\n", encoding="utf-8"
        )
    except BaseException as error:
        failure = {
            "status": "failed",
            "dataset": args.dataset,
            "run_config": run_config,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        write_json(run_dir / "metrics.json", failure)
        status.update(
            {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        (run_dir / "run.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        write_json(run_dir / "status.json", status)
    print(run_dir)


if __name__ == "__main__":
    main()
