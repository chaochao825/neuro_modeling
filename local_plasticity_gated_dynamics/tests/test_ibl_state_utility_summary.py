import numpy as np
import pandas as pd
import pytest

from figures.exp40_ibl_state_utility_plot import make_figure
from src.analysis.ibl_state_utility_summary import (
    CONDITIONS,
    exp40_animal_effects,
    exp40_claims,
    exp40_condition_summary,
)


def _frame(*, profile: str, failed_last: bool = True) -> pd.DataFrame:
    rows = []
    for animal_index in range(13):
        failed = failed_last and animal_index == 12
        for condition_index, condition in enumerate(CONDITIONS):
            baseline = 0.50 + 0.001 * animal_index
            low_nll = {
                "history_only": baseline,
                "learned_hmm_mean": baseline - 0.002,
                "semimarkov_mean": baseline - 0.003,
                "semimarkov_release": baseline - 0.006,
                "semimarkov_concentration": baseline - 0.007,
                "factorized_state": baseline - 0.013,
                "oracle_context_mean": baseline - 0.020,
            }[condition]
            context_nll = {
                "history_only": 0.69,
                "learned_hmm_mean": 0.32,
                "semimarkov_mean": 0.22,
                "semimarkov_release": 0.22,
                "semimarkov_concentration": 0.22,
                "factorized_state": 0.22,
                "oracle_context_mean": 0.0,
            }[condition]
            row = {
                "eid": f"eid-{animal_index}",
                "animal_id": f"animal-{animal_index}",
                "condition": condition,
                "profile": profile,
                "status": "failed" if failed else "complete",
                "test_low_contrast_choice_nll": np.nan if failed else low_nll,
                "context_nll": np.nan if failed else context_nll,
                "behavior_parameter_count": 7 + condition_index,
            }
            if failed:
                row.update(
                    error_type="IBLBehaviorDataError",
                    error="too few low-contrast choices in test split",
                )
            if condition == "factorized_state" and not failed:
                row.update(
                    factorized_nll_gain_vs_primary_baseline=0.010,
                    clamp_release_low_contrast_nll_harm=0.006,
                    clamp_concentration_low_contrast_nll_harm=-0.004,
                    selected_primary_baseline="semimarkov_mean",
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_summary_keeps_failed_animals_and_uses_animal_level_claims(tmp_path) -> None:
    primary = _frame(profile="development_posthoc_exposed_cohort")
    probe = _frame(profile="development_posthoc_assay_probe")
    effects = exp40_animal_effects(primary, probe)
    claims = exp40_claims(effects, n_resamples=500)
    summary = exp40_condition_summary(primary, probe)

    assert len(effects) == 13
    assert effects["endpoint_status"].eq("failed").sum() == 1
    assert (
        effects["primary_gain_selected_baseline_minus_factorized"].notna().sum() == 12
    )
    assert set(claims["evidence_tier"]) == {"posthoc_development"}
    assert not claims["confirmatory_eligible"].any()
    conclusions = claims.set_index("claim")["conclusion"].to_dict()
    assert conclusions["context_decoding_gain"] == "support"
    assert conclusions["meaningful_behavioral_utility"] == "support"
    assert conclusions["release_actuator_contribution"] == "support"
    assert conclusions["precision_actuator_contribution"] == "oppose"
    assert set(summary["complete_animals"]) == {12}
    assert set(summary["failed_animals"]) == {1}

    make_figure(effects, tmp_path)
    assert (tmp_path / "exp40_ibl_state_utility.pdf").stat().st_size > 0
    assert (tmp_path / "exp40_ibl_state_utility.png").stat().st_size > 0


def test_summary_rejects_condition_selective_failures_and_cohort_changes() -> None:
    primary = _frame(profile="development_posthoc_exposed_cohort")
    probe = _frame(profile="development_posthoc_assay_probe")
    row = primary.index[
        (primary["eid"] == "eid-0") & (primary["condition"] == "history_only")
    ][0]
    primary.loc[row, "status"] = "failed"
    primary.loc[row, "error"] = "selective failure"
    with pytest.raises(ValueError, match="asymmetric"):
        exp40_animal_effects(primary, probe)

    primary = _frame(profile="development_posthoc_exposed_cohort")
    probe = _frame(profile="development_posthoc_assay_probe")
    probe.loc[probe["eid"] == "eid-0", "animal_id"] = "different-animal"
    with pytest.raises(ValueError, match="cohorts differ"):
        exp40_animal_effects(primary, probe)


def test_claim_bootstrap_rejects_too_few_resamples() -> None:
    effects = exp40_animal_effects(
        _frame(profile="development_posthoc_exposed_cohort"),
        _frame(profile="development_posthoc_assay_probe"),
    )
    with pytest.raises(ValueError, match="at least 100"):
        exp40_claims(effects, n_resamples=99)
