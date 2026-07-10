"""Report and decision-matrix generation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List


def _level(score: float) -> str:
    if score >= 0.8:
        return "strong"
    if score >= 0.55:
        return "moderate"
    if score >= 0.25:
        return "weak"
    return "refuted"


def build_decision_matrix(summary: Dict[str, object]) -> Dict[str, object]:
    hawkes = summary["hawkes"]["glm_comparison"]
    h1_score = 0.0
    if hawkes["history_delta_bits"] > 0.001:
        h1_score += 0.45
    if hawkes["local_delta_bits"] > 0.001:
        h1_score += 0.35
    if summary["hawkes"]["metrics"]["cross_correlation"]["mean_abs_offdiag"] > summary["baseline"]["metrics"]["cross_correlation"]["mean_abs_offdiag"]:
        h1_score += 0.2

    linear_cases = summary["linear"]["cases"]
    critical = next(c for c in linear_cases if c["name"] == "critical_symmetric_powerlaw")
    alpha = critical["eigenspectrum_power_law"]["alpha"]
    lyap = critical["lyapunov_agreement"]["log_eigenspectrum_corr"]
    h2_score = 0.0
    if 0.45 <= alpha <= 0.95:
        h2_score += 0.45
    if lyap > 0.75:
        h2_score += 0.35
    if critical["target_rho"] > 0.95:
        h2_score += 0.2

    ei_cases = summary["ei"]["cases"]
    gamma = next(c for c in ei_cases if c["mode"] == "gamma_sync")
    h3_flags = {
        "narrowband_psd": gamma["metrics"]["psd"]["peak_ratio"] > 5.0,
        "phase_locking": gamma["metrics"]["phase_locking"]["mean_plv"] > 0.12,
        "near_unit_complex_dmd": gamma["metrics"]["dmd"]["near_unit_complex"] > 0,
        "positive_phase_reset": gamma["metrics"]["phase_reset_proxy"] > 0.02,
    }
    if all(h3_flags.values()):
        h3_score = 1.0
    elif h3_flags["narrowband_psd"] and h3_flags["phase_locking"]:
        h3_score = 0.35 if (h3_flags["near_unit_complex_dmd"] or h3_flags["positive_phase_reset"]) else 0.25
    elif any(h3_flags.values()):
        h3_score = 0.15
    else:
        h3_score = 0.0

    branch_cases = summary["branching"]["cases"]
    near = min(branch_cases, key=lambda c: abs(c["m"] - 1.0))
    best_dynamic = max(branch_cases, key=lambda c: c["dynamic_range"]["dynamic_range_db"])
    h4_score = 0.0
    if abs(near["estimated_branching_ratio"] - 1.0) < 0.2:
        h4_score += 0.3
    if near["size_tail"]["best_model"] == "power_law":
        h4_score += 0.25
    if abs(best_dynamic["m"] - 1.0) <= 0.12:
        h4_score += 0.25
    if near["n_avalanches"] >= 20:
        h4_score += 0.2

    best_energy = summary["energy"]["best"]
    h5_score = 0.0
    if 0.05 <= best_energy["target_sparsity"] <= 0.3:
        h5_score += 0.3
    if best_energy["rho"] < 1.0 and best_energy["rho"] >= 0.85:
        h5_score += 0.25
    if 0.0 < best_energy["long_range_fraction"] <= 0.12:
        h5_score += 0.2
    if best_energy["information_per_cost"] > 0:
        h5_score += 0.25

    return {
        "H1_history_local_coupling": {
            "level": _level(h1_score),
            "score": round(h1_score, 3),
            "evidence": {
                "history_delta_bits": hawkes["history_delta_bits"],
                "local_delta_bits": hawkes["local_delta_bits"],
            },
            "interpretation": "Nested GLM gains test whether history and local coupling add predictive information.",
        },
        "H2_nearcritical_powerlaw_spectrum": {
            "level": _level(h2_score),
            "score": round(h2_score, 3),
            "evidence": {"alpha": alpha, "lyapunov_log_eig_corr": lyap},
            "interpretation": "Power-law-like eigenspectrum is counted only with Lyapunov covariance agreement.",
        },
        "H3_oscillatory_synchrony": {
            "level": _level(h3_score),
            "score": round(h3_score, 3),
            "evidence": {**gamma["metrics"], "required_criteria": h3_flags},
            "interpretation": "Narrowband power and PLV alone are not sufficient for a synchrony-code claim; near-unit complex DMD and positive phase reset are required for strong support.",
        },
        "H4_avalanche_criticality": {
            "level": _level(h4_score),
            "score": round(h4_score, 3),
            "evidence": {
                "near_m_case": near,
                "dynamic_range_best_m": best_dynamic["m"],
            },
            "interpretation": "A log-log-looking tail alone is not counted as criticality.",
        },
        "H5_energy_constraint": {
            "level": _level(h5_score),
            "score": round(h5_score, 3),
            "evidence": best_energy,
            "interpretation": "Energy support is based on information per activity and wiring cost proxy.",
        },
    }


def write_outputs(summary: Dict[str, object], root: Path) -> Dict[str, Path]:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    matrix = build_decision_matrix(summary)
    summary_path = reports / "summary.json"
    matrix_path = reports / "decision_matrix.json"
    report_path = reports / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(render_markdown(summary, matrix), encoding="utf-8")
    shutil.copyfile(matrix_path, root / "decision_matrix.json")
    shutil.copyfile(report_path, root / "report.md")
    return {"summary": summary_path, "matrix": matrix_path, "report": report_path}


def render_markdown(summary: Dict[str, object], matrix: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Neural Multiscale Validation Report")
    lines.append("")
    lines.append(
        "> Scope: `synthetic_calibration / single_seed`. Levels test whether "
        "the pipeline recovers configured synthetic mechanisms; they are not "
        "biological inference."
    )
    lines.append("")
    lines.append("This report separates mechanistic evidence instead of claiming a unified theory is proven.")
    lines.append("")
    lines.append("## Decision Matrix")
    lines.append("")
    lines.append("| Hypothesis | Level | Key evidence |")
    lines.append("|---|---:|---|")
    for key, row in matrix.items():
        evidence = row["evidence"]
        if key.startswith("H1"):
            detail = f"history delta={evidence['history_delta_bits']:.4f} bits/bin, local delta={evidence['local_delta_bits']:.4f}"
        elif key.startswith("H2"):
            detail = f"alpha={evidence['alpha']:.3f}, Lyapunov log-eig corr={evidence['lyapunov_log_eig_corr']:.3f}"
        elif key.startswith("H3"):
            detail = f"PSD peak ratio={evidence['psd']['peak_ratio']:.2f}, PLV={evidence['phase_locking']['mean_plv']:.3f}, complex modes={evidence['dmd']['near_unit_complex']}, reset={evidence['phase_reset_proxy']:.3f}"
        elif key.startswith("H4"):
            detail = f"near m branching={evidence['near_m_case']['estimated_branching_ratio']:.3f}, best dynamic-range m={evidence['dynamic_range_best_m']}"
        else:
            detail = f"best sparsity={evidence['target_sparsity']:.2f}, long-range={evidence['long_range_fraction']:.3f}, rho={evidence['rho']:.3f}"
        lines.append(f"| {key} | {row['level']} | {detail} |")
    lines.append("")
    lines.append("## Controls and Caveats")
    lines.append("")
    lines.append("- Baseline independent Bernoulli activity is used as a negative control for correlations, spectrum, DMD modes, and avalanches.")
    lines.append("- Public-data scripts do not download large datasets by default; they analyze local exported matrices with the same metrics.")
    lines.append("- Avalanche evidence requires model comparison and branching/dynamic-range checks. A log-log tail alone is not accepted.")
    lines.append("- Oscillation evidence is downgraded unless phase locking, complex modes, narrowband PSD, and reset proxy align.")
    lines.append("")
    lines.append("## Selected Raw Metrics")
    lines.append("")
    lines.append(f"- Baseline mean abs cross-correlation: {summary['baseline']['metrics']['cross_correlation']['mean_abs_offdiag']:.4f}")
    lines.append(f"- Hawkes mean abs cross-correlation: {summary['hawkes']['metrics']['cross_correlation']['mean_abs_offdiag']:.4f}")
    lines.append(f"- Critical linear eigenspectrum alpha: {matrix['H2_nearcritical_powerlaw_spectrum']['evidence']['alpha']:.4f}")
    lines.append(f"- Gamma-sync PSD peak ratio: {matrix['H3_oscillatory_synchrony']['evidence']['psd']['peak_ratio']:.4f}")
    lines.append("")
    return "\n".join(lines)
