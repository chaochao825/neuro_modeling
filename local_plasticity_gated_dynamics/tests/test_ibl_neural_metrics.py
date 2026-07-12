from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.analysis.ibl_neural_metrics import compare_count_families
from src.models.hierarchical_count_dynamics import (
    HierarchicalCountScore,
    SessionCountMetrics,
)


def _score(
    family: str, nlls: tuple[float, ...], parameters: int
) -> HierarchicalCountScore:
    sessions = tuple(
        SessionCountMetrics(
            session_id=f"s{index}",
            animal_id=f"a{index // 2}",
            n_transitions=10,
            n_count_observations=100,
            log_likelihood=-100.0 * nll,
            null_log_likelihood=-150.0,
            saturated_log_likelihood=-50.0,
            nll_per_count=nll,
            pseudo_r2=0.2,
            closure_mse=0.1,
        )
        for index, nll in enumerate(nlls)
    )
    return HierarchicalCountScore(
        family=family,
        per_session=sessions,
        parameter_count=parameters,
        nll_per_count=sum(nlls) / len(nlls),
        pseudo_r2=0.2,
        closure_mse=0.1,
    )


def test_animal_nested_comparison_supports_positive_compact_shared_model() -> None:
    scores = {
        "common": _score("common", (1.2, 1.3, 1.1, 1.25), 100),
        "shared": _score("shared", (1.0, 1.1, 0.9, 1.05), 120),
        "full": _score("full", (0.99, 1.09, 0.89, 1.04), 300),
    }
    comparison = compare_count_families(
        scores,
        planned_sessions=4,
        planned_animals=2,
        n_bootstrap=500,
        seed=9,
    )
    assert comparison.shared_vs_common.ci_low > 0.0
    assert comparison.full_vs_common.ci_low > 0.0
    assert comparison.retention_margin.ci_low >= 0.0
    assert comparison.retention_margin.null_value == 0.0
    assert comparison.retention_margin.holm_adjusted_p < 0.05
    assert comparison.retained_full_gain_ratio >= 0.9
    assert comparison.shared_has_fewer_parameters
    assert comparison.conclusion == "support"
    assert comparison.inference_unit == "animal_with_session_nested"


def test_missing_cohort_or_undefined_full_gain_is_inconclusive() -> None:
    common = _score("common", (1.0, 1.0), 100)
    scores = {
        "common": common,
        "shared": _score("shared", (0.9, 0.9), 120),
        "full": _score("full", (1.1, 1.1), 300),
    }
    result = compare_count_families(
        scores,
        planned_sessions=20,
        planned_animals=5,
        n_bootstrap=200,
    )
    assert result.retention_margin.n_sessions == 2
    assert np.isnan(result.retained_full_gain_ratio)
    assert result.conclusion == "inconclusive"

    mismatched = replace(scores["shared"], per_session=scores["shared"].per_session[:1])
    try:
        compare_count_families(
            {**scores, "shared": mismatched},
            planned_sessions=2,
            planned_animals=1,
            n_bootstrap=200,
        )
    except ValueError as error:
        assert "same sessions" in str(error)
    else:
        raise AssertionError("mismatched paired sessions must fail closed")


def test_retention_margin_uses_all_sessions_without_positive_gain_filter() -> None:
    scores = {
        "common": _score("common", (1.0, 1.0, 1.0, 1.0), 100),
        "shared": _score("shared", (0.6, 1.4, 0.6, 1.4), 120),
        "full": _score("full", (0.5, 1.5, 0.5, 1.5), 300),
    }
    result = compare_count_families(
        scores,
        planned_sessions=4,
        planned_animals=2,
        n_bootstrap=500,
        seed=4,
    )
    assert result.retention_margin.n_sessions == 4
    assert np.isclose(result.retention_margin.estimate, 0.0)
    assert np.isnan(result.retained_full_gain_ratio)
    assert result.conclusion == "inconclusive"


def test_negative_full_gain_makes_retention_undefined_not_opposed() -> None:
    scores = {
        "common": _score("common", (1.0, 1.0, 1.0, 1.0), 100),
        "shared": _score("shared", (0.8, 0.8, 0.8, 0.8), 120),
        "full": _score("full", (1.2, 1.2, 1.2, 1.2), 300),
    }
    result = compare_count_families(
        scores,
        planned_sessions=4,
        planned_animals=2,
        n_bootstrap=500,
        seed=8,
    )
    assert result.shared_vs_common.ci_low > 0.0
    assert result.full_vs_common.ci_high < 0.0
    assert not result.retention_defined
    assert result.conclusion == "inconclusive"


def test_retention_margin_cannot_oppose_when_full_gain_ci_crosses_zero() -> None:
    full_gain = np.tile(np.asarray([1.0, 1.0, -0.8, -0.8]), 5)
    shared_gain = 0.9 * full_gain - 0.05
    common_nll = np.ones(len(full_gain))
    scores = {
        "common": _score("common", tuple(common_nll), 100),
        "shared": _score("shared", tuple(common_nll - shared_gain), 120),
        "full": _score("full", tuple(common_nll - full_gain), 300),
    }
    result = compare_count_families(
        scores,
        planned_sessions=20,
        planned_animals=10,
        n_bootstrap=1000,
        seed=18,
    )
    assert result.full_vs_common.ci_low < 0.0 < result.full_vs_common.ci_high
    assert result.retention_margin.ci_high < 0.0
    assert not result.retention_defined
    assert result.conclusion == "inconclusive"
