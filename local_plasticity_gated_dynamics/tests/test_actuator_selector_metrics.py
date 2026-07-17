from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.actuator_selector_metrics import (
    SELECTOR_MODES,
    summarize_actuator_selector,
    validate_selector_records,
)


def _records(seeds: tuple[int, ...] = tuple(range(8))) -> pd.DataFrame:
    candidates = (
        (0.60, 0.55, 0.70),
        (0.50, 0.65, 0.60),
        (0.60, 0.55, 0.70),
    )
    modes = ("routing", "gain", "low_rank")
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for generator, utilities in enumerate(candidates):
            oracle_index = max(range(3), key=utilities.__getitem__)
            oracle_mode = modes[oracle_index]
            for selector in SELECTOR_MODES:
                if selector in {"oracle", "local_three_factor"}:
                    selected = oracle_mode
                elif selector == "fixed_best":
                    selected = "low_rank"
                else:
                    selected = "routing"
                selected_index = modes.index(selected)
                rows.append(
                    {
                        "seed": seed,
                        "outer_seed": seed,
                        "source_seed": seed,
                        "generator_id": f"g{generator}",
                        "generator_split": "heldout",
                        "strict_unseen_composition": True,
                        "primary_endpoint_eligible": True,
                        "composition_overlap_secondary": False,
                        "selector": selector,
                        "mode_selected": selected,
                        "oracle_mode": oracle_mode,
                        "fixed_best_mode": "low_rank",
                        "utility": utilities[selected_index],
                        "candidate_routing_utility": utilities[0],
                        "candidate_gain_utility": utilities[1],
                        "candidate_low_rank_utility": utilities[2],
                        "train_mean_candidate_routing_utility": 0.55,
                        "train_mean_candidate_gain_utility": 0.60,
                        "train_mean_candidate_low_rank_utility": 0.70,
                        "training_source_seed_count": len(seeds) - 1,
                        "outer_seed_excluded_from_training": True,
                        "train_split": "discovery",
                        "train_endpoint": "validation_balanced_accuracy",
                        "test_endpoint": "test_balanced_accuracy",
                        "plasticity_l1": 1.0
                        if selector == "local_three_factor"
                        else 0.0,
                        "plasticity_l2": 0.5
                        if selector == "local_three_factor"
                        else 0.0,
                        "status": "complete",
                    }
                )
    return pd.DataFrame(rows)


def test_selector_summary_supports_paired_strict_unseen_gain() -> None:
    records = _records()
    conclusion = summarize_actuator_selector(
        records,
        expected_seeds=tuple(range(8)),
        expected_primary_generators_per_seed=3,
        bootstrap_samples=500,
        permutation_samples=500,
        random_seed=7,
    )

    assert conclusion.conclusion == "support"
    assert conclusion.statistics_unit == "outer_seed"
    assert conclusion.complete_primary_coverage
    assert len(conclusion.seed_endpoints) == 8
    assert all(item.lower_confidence > 0.0 for item in conclusion.primary_contrasts)
    assert all(
        item.p_value_holm is not None and item.p_value_holm < 0.05
        for item in conclusion.primary_contrasts
    )
    assert conclusion.seed_endpoints[0].local_selection_accuracy == 1.0
    assert conclusion.seed_endpoints[0].local_recovered_oracle_gain == pytest.approx(
        1.0
    )


def test_smoke_is_forced_inconclusive_without_discarding_endpoints() -> None:
    records = _records((9000, 9001))
    conclusion = summarize_actuator_selector(
        records,
        expected_seeds=(9000, 9001),
        expected_primary_generators_per_seed=3,
        bootstrap_samples=50,
        permutation_samples=50,
        force_inconclusive=True,
    )

    assert conclusion.conclusion == "inconclusive"
    assert conclusion.complete_primary_coverage
    assert conclusion.n_seeds == 2
    assert "forced inconclusive" in conclusion.reason


def test_selector_summary_requires_holm_controlled_negative_evidence_to_oppose() -> (
    None
):
    records = _records()
    local = records["selector"] == "local_three_factor"
    records.loc[local, "mode_selected"] = "routing"
    records.loc[local, "utility"] = records.loc[local, "candidate_routing_utility"]

    conclusion = summarize_actuator_selector(
        records,
        expected_seeds=tuple(range(8)),
        expected_primary_generators_per_seed=3,
        bootstrap_samples=500,
        permutation_samples=500,
        random_seed=11,
    )

    assert conclusion.conclusion == "oppose"
    contradicted = [
        item for item in conclusion.primary_contrasts if item.upper_confidence <= 0.0
    ]
    assert contradicted
    assert all(
        item.opposition_p_value_holm is not None and item.opposition_p_value_holm < 0.05
        for item in contradicted
    )


def test_validation_fails_closed_on_selected_utility_mismatch() -> None:
    records = _records()
    records.loc[0, "utility"] = 0.01

    with pytest.raises(ValueError, match="selected frozen actuator"):
        validate_selector_records(records, expected_seeds=tuple(range(8)))


def test_validation_rejects_missing_policy_and_string_boolean() -> None:
    records = _records()
    missing = records.drop(index=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="exactly four selector rows"):
        validate_selector_records(missing, expected_seeds=tuple(range(8)))

    wrong_boolean = records.copy()
    wrong_boolean["strict_unseen_composition"] = "true"
    with pytest.raises(TypeError, match="literal booleans"):
        validate_selector_records(wrong_boolean, expected_seeds=tuple(range(8)))


def test_validation_rejects_non_outer_source_seed() -> None:
    records = _records()
    records.loc[0, "source_seed"] = 99

    with pytest.raises(ValueError, match="source_seed"):
        validate_selector_records(records, expected_seeds=tuple(range(8)))


def test_validation_recomputes_train_only_fixed_best_and_oracle_tie_break() -> None:
    records = _records()
    wrong_fixed = records.copy()
    wrong_fixed["fixed_best_mode"] = "gain"
    fixed_rows = wrong_fixed["selector"] == "fixed_best"
    wrong_fixed.loc[fixed_rows, "mode_selected"] = "gain"
    wrong_fixed.loc[fixed_rows, "utility"] = wrong_fixed.loc[
        fixed_rows, "candidate_gain_utility"
    ]
    with pytest.raises(ValueError, match="train-only registered argmax"):
        validate_selector_records(wrong_fixed, expected_seeds=tuple(range(8)))

    tied = _records()
    first_cell = (tied["outer_seed"] == 0) & (tied["generator_id"] == "g0")
    tied.loc[first_cell, "candidate_routing_utility"] = 0.70
    tied.loc[first_cell, "oracle_mode"] = "low_rank"
    tied.loc[first_cell & tied["mode_selected"].eq("routing"), "utility"] = 0.70
    with pytest.raises(ValueError, match="tie break"):
        validate_selector_records(tied, expected_seeds=tuple(range(8)))
