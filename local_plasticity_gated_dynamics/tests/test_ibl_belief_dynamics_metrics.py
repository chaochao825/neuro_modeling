from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.analysis.ibl_belief_dynamics_metrics import (
    compare_belief_dynamics_conditions,
)
from src.models.hierarchical_count_dynamics import (
    HierarchicalCountScore,
    SessionCountMetrics,
)


def _score(condition: str, shifts: list[float]) -> HierarchicalCountScore:
    rows = tuple(
        SessionCountMetrics(
            session_id=f"s{index}",
            animal_id=f"a{index}",
            n_transitions=10,
            n_count_observations=100,
            log_likelihood=-10.0,
            null_log_likelihood=-12.0,
            saturated_log_likelihood=-8.0,
            nll_per_count=1.0 + shift,
            pseudo_r2=0.1,
            closure_mse=0.2,
        )
        for index, shift in enumerate(shifts)
    )
    return HierarchicalCountScore(
        family="shared",
        per_session=rows,
        parameter_count=5,
        nll_per_count=float(np.mean([item.nll_per_count for item in rows])),
        pseudo_r2=0.1,
        closure_mse=0.2,
    )


def test_animal_primary_contrasts_apply_one_holm_family() -> None:
    intact = _score("intact", [0.0] * 8)
    result = compare_belief_dynamics_conditions(
        {
            "md_shared": intact,
            "common": _score("common", [0.2] * 8),
            "clamp": _score("clamp", [0.1] * 8),
        },
        intact_condition="md_shared",
        comparator_conditions=("common", "clamp"),
        planned_sessions=8,
        planned_animals=8,
        n_bootstrap=200,
        seed=3,
    )
    assert len(result) == 2
    assert all(item.conclusion == "support" for item in result)
    assert all(item.ci_low > 0.0 for item in result)
    assert all(item.holm_adjusted_p < 0.05 for item in result)
    assert all(item.inference_unit == "animal_with_session_nested" for item in result)


def test_incomplete_or_negative_contrast_is_not_support() -> None:
    intact = _score("intact", [0.0] * 4)
    result = compare_belief_dynamics_conditions(
        {"md_shared": intact, "worse": _score("worse", [-0.2] * 4)},
        intact_condition="md_shared",
        comparator_conditions=("worse",),
        planned_sessions=4,
        planned_animals=4,
        n_bootstrap=100,
    )
    assert result[0].conclusion == "oppose"
    incomplete = replace(result[0], complete_cohort=False, conclusion="inconclusive")
    assert incomplete.conclusion == "inconclusive"


def test_comparison_fails_closed_on_missing_or_misaligned_conditions() -> None:
    intact = _score("intact", [0.0, 0.0])
    with pytest.raises(ValueError, match="exactly"):
        compare_belief_dynamics_conditions(
            {"md_shared": intact, "extra": intact},
            intact_condition="md_shared",
            comparator_conditions=("common",),
            planned_sessions=2,
            planned_animals=2,
            n_bootstrap=100,
        )
    changed_rows = list(intact.per_session)
    changed_rows[0] = replace(changed_rows[0], session_id="different")
    changed = replace(intact, per_session=tuple(changed_rows))
    with pytest.raises(ValueError, match="same sessions"):
        compare_belief_dynamics_conditions(
            {"md_shared": intact, "common": changed},
            intact_condition="md_shared",
            comparator_conditions=("common",),
            planned_sessions=2,
            planned_animals=2,
            n_bootstrap=100,
        )
