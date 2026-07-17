"""Tests for seed-level actuator phase-diagram inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.actuator_phase_statistics import (
    fit_chi_threshold,
    holm_adjust,
    seed_phase_endpoints,
    summarize_phase_diagram,
)


def _records(n_seeds: int = 8) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    modes = ("frozen", "routing", "gain", "low_rank")
    for seed in range(n_seeds):
        for split in ("discovery", "heldout"):
            for index, chi in enumerate(np.linspace(0.05, 0.95, 10)):
                generator = f"{split}-{index}"
                alpha = np.linspace(0.05, 0.95, 10)[(index * 3) % 10]
                low_rank = 0.55 + 0.35 * chi
                routing = 0.90 - 0.35 * chi
                gain = routing - 0.02
                values = {
                    "frozen": 0.5,
                    "routing": routing,
                    "gain": gain,
                    "low_rank": low_rank,
                }
                for mode in modes:
                    rows.append(
                        {
                            "seed": seed,
                            "generator_id": generator,
                            "generator_split": split,
                            "actuator_mode": mode,
                            "chi": chi,
                            "alpha": alpha,
                            "validation_balanced_accuracy": values[mode],
                            "test_balanced_accuracy": values[mode],
                            "status": "complete",
                            "functional_budget_valid": True,
                        }
                    )
    return pd.DataFrame(rows)


def test_holm_adjustment_and_threshold_tie_break_are_deterministic() -> None:
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    chi = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert fit_chi_threshold(chi, labels) == pytest.approx(0.5)


def test_seed_endpoints_use_other_seed_discovery_and_heldout_test() -> None:
    frame = _records()
    endpoints = seed_phase_endpoints(frame)
    assert len(endpoints) == 8
    assert all(item.heldout_generators == 10 for item in endpoints)
    assert all(item.heldout_ties == 0 for item in endpoints)
    assert min(item.spearman_rho for item in endpoints) > 0.99
    assert min(item.classifier_balanced_accuracy for item in endpoints) == 1.0
    assert min(item.classifier_auroc for item in endpoints) == 1.0
    assert min(item.chi_minus_alpha_auroc for item in endpoints) > 0.25
    # Changing one seed's discovery data cannot tune that same seed's LOSO threshold.
    baseline = endpoints[0].discovery_threshold
    mask = (frame["seed"] == 0) & (frame["generator_split"] == "discovery")
    frame.loc[mask & (frame["actuator_mode"] == "low_rank"), "validation_balanced_accuracy"] = 0.0
    changed = seed_phase_endpoints(frame)[0].discovery_threshold
    assert changed == baseline


def test_summary_infers_only_over_seeds_and_supports_clear_relation() -> None:
    conclusion = summarize_phase_diagram(
        _records(),
        expected_seeds=tuple(range(8)),
        bootstrap_samples=1_000,
        permutation_samples=2_000,
    )
    assert conclusion.statistics_unit == "seed"
    assert conclusion.n_seeds == 8
    assert conclusion.complete_primary_coverage
    assert conclusion.conclusion == "support"
    assert conclusion.gramian_predictor_beats_alpha
    assert all(item.p_value_holm < 0.05 for item in conclusion.endpoint_summaries)


def test_missing_or_failed_primary_cell_forces_inconclusive() -> None:
    frame = _records()
    frame.loc[0, "status"] = "failed"
    conclusion = summarize_phase_diagram(
        frame,
        expected_seeds=tuple(range(8)),
        bootstrap_samples=500,
        permutation_samples=500,
    )
    assert not conclusion.complete_primary_coverage
    assert conclusion.conclusion == "inconclusive"


def test_statistics_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="both task families"):
        fit_chi_threshold([0.1, 0.2], [0, 0])
    frame = _records()
    frame.loc[0, "chi"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        seed_phase_endpoints(frame)
