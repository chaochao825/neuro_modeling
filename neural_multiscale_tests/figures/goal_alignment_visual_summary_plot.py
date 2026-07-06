"""Plot goal-alignment summary panels from generated simulation reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_BASE = Path(__file__).with_name("goal_alignment_visual_summary")


def load_json(name: str) -> dict:
    with (REPORTS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def level_color(level: str) -> str:
    return {
        "strong": "#1b9e77",
        "moderate": "#7570b3",
        "weak": "#d95f02",
        "refuted": "#d73027",
    }.get(level, "#666666")


def panel_decision_scores(ax, matrix: dict) -> None:
    keys = list(matrix)
    labels = [k.split("_")[0] for k in keys]
    scores = [matrix[k]["score"] for k in keys]
    colors = [level_color(matrix[k]["level"]) for k in keys]
    ax.bar(labels, scores, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_ylabel("Evidence score")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.55, 0.8, 1.0])
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    for i, k in enumerate(keys):
        ax.text(i, scores[i] + 0.035, matrix[k]["level"], ha="center", va="bottom", fontsize=8)
    ax.set_title("A. H1-H5 evidence levels", loc="left", fontweight="bold")


def panel_h1(ax, summary: dict) -> None:
    glm = summary["hawkes"]["glm_comparison"]
    deltas = [glm["history_delta_bits"], glm["local_delta_bits"], glm["global_delta_bits"]]
    labels = ["history", "local", "global"]
    ax.bar(labels, deltas, color=["#4c78a8", "#59a14f", "#f28e2b"], edgecolor="#333333", linewidth=0.6)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Delta bits/bin")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    baseline_xcorr = summary["baseline"]["metrics"]["cross_correlation"]["mean_abs_offdiag"]
    hawkes_xcorr = summary["hawkes"]["metrics"]["cross_correlation"]["mean_abs_offdiag"]
    ax.text(
        0.02,
        0.95,
        f"mean abs xcorr: baseline={baseline_xcorr:.4f}, Hawkes={hawkes_xcorr:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    ax.set_title("B. H1 GLM gains and correlations", loc="left", fontweight="bold")


def panel_h2(ax, summary: dict) -> None:
    cases = summary["linear"]["cases"]
    labels = [c["name"].replace("nearcritical_", "near-").replace("_", "\n") for c in cases]
    alpha = [c["eigenspectrum_power_law"]["alpha"] for c in cases]
    lyap = [c["lyapunov_agreement"]["log_eigenspectrum_corr"] for c in cases]
    x = np.arange(len(cases))
    width = 0.36
    ax.bar(x - width / 2, alpha, width, label="alpha", color="#4c78a8", edgecolor="#333333", linewidth=0.5)
    ax.bar(x + width / 2, lyap, width, label="Lyap corr", color="#59a14f", edgecolor="#333333", linewidth=0.5)
    ax.axhspan(0.7, 0.85, color="#f2cf5b", alpha=0.25, label="target alpha band")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Value")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("C. H2 spectrum and Lyapunov check", loc="left", fontweight="bold")


def panel_h3(ax, matrix: dict) -> None:
    ev = matrix["H3_oscillatory_synchrony"]["evidence"]
    criteria = ev["required_criteria"]
    labels = ["PSD", "PLV", "complex\nDMD", "phase\nreset"]
    passed = [
        criteria["narrowband_psd"],
        criteria["phase_locking"],
        criteria["near_unit_complex_dmd"],
        criteria["positive_phase_reset"],
    ]
    vals = [1 if p else 0 for p in passed]
    colors = ["#1b9e77" if p else "#d95f02" for p in passed]
    ax.bar(labels, vals, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["missing", "present"])
    ax.text(0.03, 1.08, f"peak={ev['psd']['peak_ratio']:.2f}, PLV={ev['phase_locking']['mean_plv']:.2f}", transform=ax.transAxes, fontsize=8)
    ax.text(0.03, 0.98, f"near-unit complex={ev['dmd']['near_unit_complex']}, reset={ev['phase_reset_proxy']:.3f}", transform=ax.transAxes, fontsize=8)
    ax.set_title("D. H3 synchrony-code criteria", loc="left", fontweight="bold")


def panel_h4(ax, summary: dict) -> None:
    cases = summary["branching"]["cases"]
    m = [c["m"] for c in cases]
    br = [c["estimated_branching_ratio"] for c in cases]
    dyn = [c["dynamic_range"]["dynamic_range_db"] for c in cases]
    ax.plot(m, br, color="#4c78a8", marker="o", linewidth=2, label="branching ratio")
    ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("m")
    ax.set_ylabel("Estimated branching ratio")
    ax.grid(color="#dddddd", linewidth=0.6)
    ax2 = ax.twinx()
    ax2.plot(m, dyn, color="#e15759", marker="s", linewidth=2, label="dynamic range")
    ax2.set_ylabel("Dynamic range (dB)")
    lines = [line for line in ax.get_lines() + ax2.get_lines() if not line.get_label().startswith("_")]
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8, loc="upper left")
    ax.set_title("E. H4 critical branching check", loc="left", fontweight="bold")


def panel_h5(ax, summary: dict) -> None:
    grid = summary["energy"]["grid"]
    best = summary["energy"]["best"]
    xs = np.array([g["target_sparsity"] for g in grid])
    ys = np.array([g["rho"] for g in grid])
    cs = np.array([g["information_per_cost"] for g in grid])
    sizes = np.array([60 + 900 * g["long_range_fraction"] for g in grid])
    sc = ax.scatter(xs, ys, c=cs, s=sizes, cmap="viridis", edgecolor="#333333", linewidth=0.5)
    ax.scatter([best["target_sparsity"]], [best["rho"]], marker="*", s=220, color="#d73027", edgecolor="#333333", linewidth=0.7, label="best")
    ax.set_xlabel("Target sparsity")
    ax.set_ylabel("Spectral radius rho")
    ax.grid(color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Information / cost")
    ax.set_title("F. H5 energy-efficiency sweep", loc="left", fontweight="bold")


def main() -> None:
    summary = load_json("summary.json")
    matrix = load_json("decision_matrix.json")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)
    panel_decision_scores(axes[0, 0], matrix)
    panel_h1(axes[0, 1], summary)
    panel_h2(axes[0, 2], summary)
    panel_h3(axes[1, 0], matrix)
    panel_h4(axes[1, 1], summary)
    panel_h5(axes[1, 2], summary)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_BASE.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(OUT_BASE.with_suffix(".png"))
    print(OUT_BASE.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
