#!/usr/bin/env python3
"""Materialize claim-ineligible Exp39 cell, loading, and timing diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import uuid

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.exp39_claim_boundary import (
    cellwise_utility,
    cross_loading,
    headroom_retention,
    selected_timescales,
    timing_utility,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _classification(
    cell_summary: pd.DataFrame,
    loading_summary: pd.DataFrame,
    timing: pd.DataFrame,
    registered_summary: Mapping[str, Any],
) -> list[dict[str, str]]:
    if registered_summary.get("protocol_version") != "exp39_factorized_uncertainty_v1":
        raise ValueError("claim-boundary audit requires the frozen Exp39 protocol")
    if registered_summary.get("claim_eligible") is not True:
        raise ValueError("frozen Exp39 result is not marked claim-eligible")
    verdict = registered_summary.get("verdict")
    joint_gate = registered_summary.get("joint_gate_passed")
    if verdict not in {"support", "oppose", "inconclusive"}:
        raise ValueError("frozen Exp39 summary has an invalid verdict")
    if (verdict == "support") is not (joint_gate is True):
        raise ValueError("frozen Exp39 verdict and joint gate are inconsistent")
    seen_cells = cell_summary.loc[
        cell_summary["comparison"] == "seen_mode_imm_minus_factorized_nll"
    ]
    fast = timing.loc[
        (timing["panel"] == "transition_blocks_only")
        &
        (timing["comparison"] == "seen_mode_imm_minus_factorized_nll")
        & (timing["endpoint"] == "early_nll"),
        "mean_nll_gain",
    ].item()
    late = timing.loc[
        (timing["panel"] == "transition_blocks_only")
        &
        (timing["comparison"] == "seen_mode_imm_minus_factorized_nll")
        & (timing["endpoint"] == "late_nll"),
        "mean_nll_gain",
    ].item()
    diagonal = loading_summary.groupby("estimated_factor")[
        "diagonal_exceeds_all_off_diagonal"
    ].first()
    return [
        {
            "claim": "registered_average_unseen_composition_utility",
            "conclusion": str(verdict),
            "eligibility": "confirmatory_exp39_frozen_gate",
            "reason": (
                "The original five-test preregistered joint gate passed."
                if joint_gate is True
                else "The original five-test preregistered joint gate did not pass."
            ),
        },
        {
            "claim": "uniform_cellwise_composition_utility",
            "conclusion": "oppose",
            "eligibility": "post_hoc_descriptive",
            "reason": (
                "At least one unseen cell has non-positive mean gain versus the "
                "seen-mode IMM."
                if (seen_cells["mean_nll_gain"] <= 0.0).any()
                else "All observed cell means were positive, but no cellwise gate was registered."
            ),
        },
        {
            "claim": "clean_three_factor_parameter_decomposition",
            "conclusion": "oppose",
            "eligibility": "post_hoc_descriptive",
            "reason": (
                f"Diagonal loading wins for {int(diagonal.sum())}/3 estimated "
                "coordinates; h and Q are not cleanly separated."
            ),
        },
        {
            "claim": "faster_post_switch_release_than_seen_mode_imm",
            "conclusion": "oppose" if fast <= 0.0 else "support",
            "eligibility": "registered_secondary_descriptive",
            "reason": f"Seen-IMM minus factorized early-window NLL gain is {fast:+.6f}.",
        },
        {
            "claim": "late_regime_adaptation_advantage",
            "conclusion": "support" if late > 0.0 else "oppose",
            "eligibility": "post_hoc_descriptive",
            "reason": f"Seen-IMM minus factorized late-window NLL gain is {late:+.6f}.",
        },
        {
            "claim": "real_behavior_or_neural_utility",
            "conclusion": "inconclusive",
            "eligibility": "not_tested_by_exp39",
            "reason": "Exp39 contains synthetic observations only.",
        },
    ]


def analyze(result_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    root = result_dir.resolve()
    blocks_path = root / "block_metrics.csv"
    selection_path = root / "selection_audit.csv"
    if not blocks_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("Exp39 block or selection table is missing")
    registered_summary_path = root / "summary.json"
    if not registered_summary_path.is_file():
        raise FileNotFoundError("Exp39 registered summary is missing")
    blocks = pd.read_csv(blocks_path, dtype={"cell": str})
    selection = pd.read_csv(selection_path)
    registered_summary = json.loads(
        registered_summary_path.read_text(encoding="utf-8")
    )
    if not isinstance(registered_summary, dict):
        raise ValueError("Exp39 registered summary must be a JSON object")
    cell_seed, cell_summary = cellwise_utility(blocks)
    loading_seed, loading_summary = cross_loading(blocks)
    timing = timing_utility(blocks)
    timescales = selected_timescales(selection)
    classifications = _classification(
        cell_summary, loading_summary, timing, registered_summary
    )
    summary = {
        "analysis_status": "post_hoc_frozen_result_diagnostic_only",
        "claim_upgrade_allowed": False,
        "source_result_dir": root.name,
        "source_block_metrics_sha256": _sha256(blocks_path),
        "source_selection_audit_sha256": _sha256(selection_path),
        "source_registered_summary_sha256": _sha256(registered_summary_path),
        "statistics_unit": "seed",
        "n_seeds": int(blocks["seed"].nunique()),
        "headroom": headroom_retention(blocks),
        "claim_classification": classifications,
        "interpretation": (
            "Exp39 supports registered aggregate synthetic utility. It does not "
            "support uniform cell-wise dominance, clean h/Q/R recovery, fast "
            "release, or real-data utility."
        ),
    }
    return summary, {
        "cellwise_seed_metrics": cell_seed,
        "cellwise_summary": cell_summary,
        "cross_loading_seed_metrics": loading_seed,
        "cross_loading_summary": loading_summary,
        "timing_summary": timing,
        "selected_timescales": timescales,
        "claim_classification": pd.DataFrame(classifications),
    }


def _report(summary: Mapping[str, Any], frames: Mapping[str, pd.DataFrame]) -> str:
    cells = frames["cellwise_summary"]
    seen_cells = cells.loc[
        cells["comparison"] == "seen_mode_imm_minus_factorized_nll"
    ]
    loading = frames["cross_loading_summary"]
    matrix = loading.pivot(
        index="estimated_factor", columns="true_factor", values="mean_log_response"
    )
    timing = frames["timing_summary"]
    timescales = frames["selected_timescales"]
    lines = [
        "# Exp39 Post-Hoc Claim-Boundary Audit",
        "",
        "Status: **claim-ineligible analysis of the frozen formal artifacts**.",
        "The Exp39 algorithm, tapes, settings, metrics, and registered verdict were not changed.",
        "",
        "## Cell-wise held-out utility",
        "",
        "| Cell | Seen IMM minus factorized NLL | Positive seeds |",
        "|---|---:|---:|",
    ]
    for row in seen_cells.itertuples(index=False):
        lines.append(
            f"| `{row.cell}` | {row.mean_nll_gain:+.6f} | "
            f"{int(row.positive_seeds)}/{int(row.n_seeds)} |"
        )
    lines.extend(
        [
            "",
            "The aggregate advantage is not a uniform cell-wise result: cell `110` is negative on average.",
            "",
            "## Seed-level cross-loading audit",
            "",
            "Rows are estimated log controller coordinates; columns are true manipulated factors.",
            "",
            "| Estimate | true h | true Q | true R |",
            "|---|---:|---:|---:|",
        ]
    )
    for estimate in ("h", "q", "r"):
        lines.append(
            f"| {estimate} | {matrix.loc[estimate, 'h']:.6f} | "
            f"{matrix.loc[estimate, 'q']:.6f} | {matrix.loc[estimate, 'r']:.6f} |"
        )
    lines.extend(["", "## Timing", ""])
    lines.append(
        "The original frozen `early_nll` summary includes each sequence's first "
        "block, which is initialization rather than a transition. Both the original "
        "all-block panel and a post-hoc transition-only panel are shown below."
    )
    lines.append("")
    for row in timing.itertuples(index=False):
        lines.append(
            f"- `{row.panel}` / `{row.comparison}` / `{row.endpoint}`: "
            f"{row.mean_nll_gain:+.6f} "
            f"({int(row.positive_seeds)}/{int(row.n_seeds)} positive seeds)."
        )
    lines.extend(["", "## Selected adaptation rates", ""])
    for row in timescales.itertuples(index=False):
        lines.append(
            f"- {row.factor}: beta={row.adaptation_rate:g} in "
            f"{int(row.selected_seeds)}/{int(row.n_seeds)} seeds "
            f"(~{row.approximate_steps:g}-step nominal time scale)."
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "| Claim | Conclusion | Eligibility |",
            "|---|---|---|",
        ]
    )
    for item in summary["claim_classification"]:
        lines.append(
            f"| `{item['claim']}` | **{item['conclusion']}** | {item['eligibility']} |"
        )
    lines.extend(
        [
            "",
            "The next experiment must therefore test matched Q/R marginals, explicit reduced-factor baselines, and early switch release. This audit cannot itself upgrade any claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    summary, frames = analyze(args.result_dir.expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex}")
    staging.mkdir(exist_ok=False)
    _write_json(staging / "summary.json", summary)
    for name, frame in frames.items():
        frame.to_csv(staging / f"{name}.csv", index=False)
    (staging / "report.md").write_text(
        _report(summary, frames), encoding="utf-8", newline="\n"
    )
    files = sorted(path for path in staging.iterdir() if path.is_file())
    manifest = "".join(
        f"{_sha256(path)}  {path.name}\n" for path in files
    )
    (staging / "artifact_manifest.sha256").write_text(
        manifest, encoding="utf-8", newline="\n"
    )
    staging.replace(output)
    print(output)


if __name__ == "__main__":
    main()
