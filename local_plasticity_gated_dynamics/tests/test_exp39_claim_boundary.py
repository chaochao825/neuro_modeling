from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from scripts.audit_exp39_claim_boundaries import analyze
from src.analysis.exp39_claim_boundary import (
    cellwise_utility,
    cross_loading,
    headroom_retention,
    selected_timescales,
    timing_utility,
    validate_block_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def _formal_blocks() -> pd.DataFrame:
    return pd.read_csv(
        ROOT
        / "results/exp39_factorized_uncertainty_prospective_v1/block_metrics.csv",
        dtype={"cell": str},
    )


def test_frozen_exp39_claim_boundary_reproduces_disclosed_diagnostics() -> None:
    blocks = _formal_blocks()
    seed_cells, cells = cellwise_utility(blocks)
    assert seed_cells["seed"].nunique() == 30
    seen = cells.set_index(["comparison", "cell"])
    assert seen.loc[
        ("seen_mode_imm_minus_factorized_nll", "110"), "mean_nll_gain"
    ] == pytest.approx(-0.0043339, abs=1e-6)
    assert seen.loc[
        ("selected_fixed_minus_factorized_nll", "110"), "positive_seeds"
    ] == 4

    _, loading = cross_loading(blocks)
    matrix = loading.pivot(
        index="estimated_factor", columns="true_factor", values="mean_log_response"
    )
    assert matrix.loc["h", "h"] == pytest.approx(0.088327, abs=1e-6)
    assert matrix.loc["q", "r"] == pytest.approx(1.491769, abs=1e-6)
    diagonal = loading.groupby("estimated_factor")[
        "diagonal_exceeds_all_off_diagonal"
    ].first()
    assert diagonal.to_dict() == {"h": False, "q": False, "r": True}

    timing = timing_utility(blocks).set_index(
        ["panel", "comparison", "endpoint"]
    )
    assert timing.loc[
        (
            "all_blocks_including_sequence_initialization",
            "seen_mode_imm_minus_factorized_nll",
            "early_nll",
        ),
        "mean_nll_gain",
    ] == pytest.approx(-0.0866983, abs=1e-6)
    assert timing.loc[
        (
            "transition_blocks_only",
            "seen_mode_imm_minus_factorized_nll",
            "early_nll",
        ),
        "mean_nll_gain",
    ] == pytest.approx(-0.0875231, abs=1e-6)
    assert timing.loc[
        (
            "transition_blocks_only",
            "seen_mode_imm_minus_factorized_nll",
            "late_nll",
        ),
        "mean_nll_gain",
    ] == pytest.approx(0.0831526, abs=1e-6)

    headroom = headroom_retention(blocks)
    assert headroom[
        "fixed_to_oracle_factorial_imm_headroom_retained"
    ] == pytest.approx(0.790827, abs=1e-6)
    assert headroom[
        "seen_imm_to_oracle_dynamic_gap_closed"
    ] == pytest.approx(0.291699, abs=1e-6)


def test_selected_timescales_match_frozen_selection_audit() -> None:
    selection = pd.read_csv(
        ROOT
        / "results/exp39_factorized_uncertainty_prospective_v1/selection_audit.csv"
    )
    counts = selected_timescales(selection).set_index(
        ["factor", "adaptation_rate"]
    )
    assert counts.loc[("h", 0.002), "selected_seeds"] == 26
    assert counts.loc[("q", 0.5), "selected_seeds"] == 30
    assert counts.loc[("r", 0.2), "selected_seeds"] == 30


def test_analysis_is_explicitly_claim_ineligible() -> None:
    result = ROOT / "results/exp39_factorized_uncertainty_prospective_v1"
    summary, frames = analyze(result)
    assert summary["claim_upgrade_allowed"] is False
    claims = {
        item["claim"]: item["conclusion"]
        for item in summary["claim_classification"]
    }
    assert claims["registered_average_unseen_composition_utility"] == "support"
    assert claims["uniform_cellwise_composition_utility"] == "oppose"
    assert claims["clean_three_factor_parameter_decomposition"] == "oppose"
    assert claims["faster_post_switch_release_than_seen_mode_imm"] == "oppose"
    assert claims["real_behavior_or_neural_utility"] == "inconclusive"
    assert not frames["claim_classification"].empty


def test_materialized_claim_boundary_artifacts_are_hash_complete() -> None:
    output = ROOT / "results/exp39_posthoc_claim_boundary_20260727"
    entries = {}
    for line in (output / "artifact_manifest.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        entries[filename] = digest
    assert set(entries) == {
        "cellwise_seed_metrics.csv",
        "cellwise_summary.csv",
        "claim_classification.csv",
        "cross_loading_seed_metrics.csv",
        "cross_loading_summary.csv",
        "report.md",
        "selected_timescales.csv",
        "summary.json",
        "timing_summary.csv",
    }
    for filename, expected in entries.items():
        observed = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        assert observed == expected


def test_claim_audit_fails_closed_on_label_or_pairing_corruption() -> None:
    blocks = _formal_blocks().iloc[:16].copy()
    blocks.loc[blocks.index[0], "cell"] = "bad"
    with pytest.raises(ValueError, match="three-bit"):
        validate_block_metrics(blocks)

    blocks = _formal_blocks()
    blocks.loc[blocks.index[0], "test_tape_digest"] = "forged"
    with pytest.raises(ValueError, match="one test tape"):
        validate_block_metrics(blocks)


def test_cross_loading_rejects_nonpositive_estimates() -> None:
    blocks = _formal_blocks()
    mask = blocks["method"].eq("factorized")
    blocks.loc[mask.idxmax(), "mean_q_estimate"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        cross_loading(blocks)


def test_claim_audit_never_hardcodes_registered_support(tmp_path: Path) -> None:
    source = ROOT / "results/exp39_factorized_uncertainty_prospective_v1"
    for filename in ("block_metrics.csv", "selection_audit.csv", "summary.json"):
        shutil.copy2(source / filename, tmp_path / filename)
    registered = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    registered["joint_gate_passed"] = False
    (tmp_path / "summary.json").write_text(
        json.dumps(registered), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="verdict and joint gate"):
        analyze(tmp_path)
