from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_report import collect_runs, write_report
from src.analysis.claims import evaluate_core_claims
from src.utils.artifacts import ExperimentRun


def _phase1_formal(n_seeds: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in range(n_seeds):
        jitter = 0.03 * np.sin(seed)
        common = {
            "profile": "formal",
            "experiment": "exp01_feedback_dimension_sweep",
            "status": "complete",
            "grid": "core",
            "seed": seed,
        }
        rows.extend(
            [
                {
                    **common,
                    "feedback_mode": "aligned",
                    "feedback_dim": 4,
                    "effective_rank": 4.0 + jitter,
                    "latent_r2": 0.91 + jitter / 10,
                },
                {
                    **common,
                    "feedback_mode": "aligned",
                    "feedback_dim": 128,
                    "effective_rank": 14.0,
                    "latent_r2": 0.915 + jitter / 10,
                },
                {
                    **common,
                    "feedback_mode": "orthogonal",
                    "feedback_dim": 4,
                    "effective_rank": 4.0,
                    "latent_r2": 0.65 + jitter / 10,
                },
            ]
        )
    return pd.DataFrame(rows)


def _phase4_formal(n_seeds: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in range(n_seeds):
        for condition, accuracy in (("in_phase", 0.82), ("no_oscillation", 0.70)):
            rows.append(
                {
                    "profile": "formal",
                    "experiment": "exp04_phase_gating",
                    "status": "complete",
                    "seed": seed,
                    "phase_condition": condition,
                    "decoding_accuracy": accuracy + 0.002 * np.sin(seed),
                    "mean_rate_match_exact": True,
                    "per_trial_spike_count_match_exact": True,
                    "mean_coupling_match_exact": True,
                    "shared_source_fingerprint": f"source-{seed}",
                }
            )
    return pd.DataFrame(rows)


def _phase2_formal(
    n_seeds: int = 20, architecture: str = "ei_n512_fi20_gain1"
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in range(n_seeds):
        for experiment in (
            "exp02_context_ei_oracle_gate",
            "exp03_context_ei_learned_gate",
        ):
            common = {
                "profile": "formal",
                "experiment": experiment,
                "status": "complete",
                "seed": seed,
                "architecture": architecture,
                "model_kind": "ei" if architecture.startswith("ei_") else "non_dale",
            }
            rows.extend(
                [
                    {
                        **common,
                        "condition": "local",
                        "accuracy": 0.90,
                        "switch_cost": -0.10,
                        "jacobian_max_real_part": -0.20,
                        "raw_update_effective_rank": 4.0,
                    },
                    {
                        **common,
                        "condition": "bptt",
                        "accuracy": 0.95,
                    },
                    {
                        **common,
                        "condition": "no-gate",
                        "switch_cost": 0.10,
                    },
                    {
                        **common,
                        "condition": "full-feedback",
                        "raw_update_effective_rank": 8.0,
                    },
                ]
            )
            if experiment == "exp02_context_ei_oracle_gate":
                rows.append(
                    {
                        **common,
                        "condition": "no-homeostasis",
                        "jacobian_max_real_part": -0.10,
                    }
                )
    return pd.DataFrame(rows)


def _p0_formal(n_seeds: int = 30) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in range(n_seeds):
        common = {
            "profile": "formal",
            "experiment": "exp07_mechanism_identifiability",
            "status": "complete",
            "seed": seed,
            "architecture": "ei_n128_fi20_p0",
            "model_kind": "ei",
        }
        for norm in ("l1", "l2"):
            rows.extend(
                [
                    {
                        **common,
                        "condition": f"task-only__aligned__{norm}",
                        "mechanism": "task-only",
                        "feedback_mode": "aligned",
                        "task_plasticity_enabled": True,
                        "homeostasis_enabled": False,
                        "normalization_enabled": False,
                        "budget_norm": norm,
                        "budget_match_valid": True,
                        "heldout_masked_mse": 0.80,
                        "accuracy": 0.90,
                    },
                    {
                        **common,
                        "condition": f"task-only__shuffled__{norm}",
                        "mechanism": "task-only",
                        "feedback_mode": "shuffled",
                        "task_plasticity_enabled": True,
                        "homeostasis_enabled": False,
                        "normalization_enabled": False,
                        "budget_norm": norm,
                        "budget_match_valid": True,
                        "heldout_masked_mse": 0.95,
                        "accuracy": 0.85,
                    },
                    {
                        **common,
                        "condition": f"frozen-recurrent__aligned__{norm}",
                        "mechanism": "frozen-recurrent",
                        "feedback_mode": "aligned",
                        "task_plasticity_enabled": False,
                        "homeostasis_enabled": False,
                        "normalization_enabled": False,
                        "budget_norm": norm,
                        "budget_match_valid": True,
                        "heldout_masked_mse": 1.00,
                        "accuracy": 0.80,
                    },
                    {
                        **common,
                        "condition": f"homeostasis-only__aligned__{norm}",
                        "mechanism": "homeostasis-only",
                        "feedback_mode": "aligned",
                        "task_plasticity_enabled": False,
                        "homeostasis_enabled": True,
                        "normalization_enabled": False,
                        "budget_norm": norm,
                        "budget_match_valid": True,
                        "heldout_masked_mse": 1.10,
                        "accuracy": 0.80,
                    },
                    {
                        **common,
                        "condition": f"task-homeostasis__aligned__{norm}",
                        "mechanism": "task+homeostasis",
                        "feedback_mode": "aligned",
                        "task_plasticity_enabled": True,
                        "homeostasis_enabled": True,
                        "normalization_enabled": False,
                        "budget_norm": norm,
                        "budget_match_valid": True,
                        "heldout_masked_mse": 0.85,
                        "accuracy": 0.90,
                    },
                    {
                        **common,
                        "condition": (
                            f"task-homeostasis-normalization__aligned__{norm}"
                        ),
                        "mechanism": "task+homeostasis+normalization",
                        "feedback_mode": "aligned",
                        "task_plasticity_enabled": True,
                        "homeostasis_enabled": True,
                        "normalization_enabled": True,
                        "budget_norm": norm,
                        "budget_match_valid": True,
                        "heldout_masked_mse": 0.85,
                        "accuracy": 0.90,
                    },
                ]
            )
        rows.extend(
            [
                {**common, "condition": "tuned-bptt", "accuracy": 0.95},
                {**common, "condition": "tuned-gru", "accuracy": 0.93},
            ]
        )
    return pd.DataFrame(rows)


def _p1_formal(n_seeds: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "profile": "formal",
                "experiment": "exp08_rank_stage_validation",
                "status": "complete",
                "seed": seed,
                "condition": "direct__feedback-4__angle-0",
                "parameterization": "direct",
                "requested_feedback_dim": 4,
                "feedback_angle_degrees": 0.0,
                "geometry_valid": True,
                "masked_identity_max_abs_residual": 0.0,
                "lowdim_credit_tangent_dimension": 4,
                "masked_numerical_rank": 100,
            }
            for seed in range(n_seeds)
        ]
    )


def test_missing_formal_evidence_is_explicitly_inconclusive() -> None:
    raw = pd.DataFrame(
        [
            {
                "profile": "smoke",
                "experiment": "exp01_feedback_dimension_sweep",
                "status": "complete",
            }
        ]
    )
    claims = evaluate_core_claims(raw)
    assert len(claims) == 23
    assert {claim.conclusion for claim in claims} == {"inconclusive"}
    assert {
        "A1_rank_matches_feedback",
        "E2_latent_precedes_behavior_bias",
        "P0_overall",
    } <= {claim.claim_id for claim in claims}


def test_report_exposes_attempt_categories_and_claim_evidence_notes(
    tmp_path: Path,
) -> None:
    runs = pd.DataFrame(
        [
            {"experiment": "exp", "profile": "formal", "status": status, "n_planned": 1}
            for status in ("complete", "complete_with_failures", "failed")
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "claim_id": "B1",
                "criterion": "absolute threshold reported separately",
                "n_complete": 19,
                "n_planned": 20,
                "n_failed": 1,
                "estimate": 0.1,
                "ci_low": -0.1,
                "ci_high": 0.2,
                "conclusion": "inconclusive",
                "note": "absolute accuracy-minus-0.85 CI [-0.2, -0.1]",
            }
        ]
    )

    write_report(tmp_path, pd.DataFrame(), runs, summary)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| exp | formal | 3 | 1 | 1 | 1 | 3 |" in report
    assert "### Evidence details" in report
    assert "`B1` (failed=1)" in report
    assert "absolute accuracy-minus-0.85 CI [-0.2, -0.1]" in report


def test_twenty_seed_phase1_support_and_missing_seed_is_inconclusive() -> None:
    claims = {claim.claim_id: claim for claim in evaluate_core_claims(_phase1_formal())}
    assert claims["A1_rank_matches_feedback"].conclusion == "support"
    assert claims["A2_d4_r2_noninferior_full"].conclusion == "support"
    assert claims["A3_alignment_is_necessary"].conclusion == "support"
    assert claims["A1_rank_matches_feedback"].n_complete == 20

    incomplete = {
        claim.claim_id: claim for claim in evaluate_core_claims(_phase1_formal(19))
    }
    assert incomplete["A1_rank_matches_feedback"].conclusion == "inconclusive"
    assert incomplete["A1_rank_matches_feedback"].n_complete == 19
    assert "19/20" in incomplete["A1_rank_matches_feedback"].note

    missing_full = _phase1_formal().loc[
        lambda frame: (
            ~(frame["feedback_mode"].eq("aligned") & frame["feedback_dim"].eq(128))
        )
    ]
    missing_claims = {
        claim.claim_id: claim for claim in evaluate_core_claims(missing_full)
    }
    assert missing_claims["A2_d4_r2_noninferior_full"].conclusion == "inconclusive"
    assert missing_claims["A2_d4_r2_noninferior_full"].n_complete == 0


def test_unrelated_phase1_failures_do_not_contaminate_required_panels() -> None:
    raw = _phase1_formal()
    failures = pd.DataFrame(
        [
            {
                "profile": "formal",
                "experiment": "exp01_feedback_dimension_sweep",
                "status": "failed",
                "grid": "ablation",
                "seed": seed,
                "feedback_mode": "shuffled",
                "feedback_dim": 32,
            }
            for seed in range(20)
        ]
    )
    claims = {
        claim.claim_id: claim
        for claim in evaluate_core_claims(pd.concat([raw, failures], ignore_index=True))
    }
    assert claims["A1_rank_matches_feedback"].conclusion == "support"
    assert claims["A2_d4_r2_noninferior_full"].conclusion == "support"
    assert claims["A3_alignment_is_necessary"].conclusion == "support"
    assert claims["A1_rank_matches_feedback"].n_failed == 0

    required_failure = raw.loc[
        ~(
            (raw["seed"] == 0)
            & (raw["feedback_mode"] == "aligned")
            & (raw["feedback_dim"] == 4)
        )
    ].copy()
    required_failure = pd.concat(
        [
            required_failure,
            pd.DataFrame(
                [
                    {
                        "profile": "formal",
                        "experiment": "exp01_feedback_dimension_sweep",
                        "status": "failed",
                        "grid": "core",
                        "seed": 0,
                        "feedback_mode": "aligned",
                        "feedback_dim": 4,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    claims = {claim.claim_id: claim for claim in evaluate_core_claims(required_failure)}
    assert claims["A1_rank_matches_feedback"].conclusion == "inconclusive"
    assert claims["A1_rank_matches_feedback"].n_failed == 1


def test_holm_is_applied_across_full_registered_family() -> None:
    claims = evaluate_core_claims(_phase1_formal())
    statistical_claims = [item for item in claims if item.claim_id != "P0_overall"]
    adjusted_pairs = []
    for claim in statistical_claims:
        if claim.p_value is None:
            continue
        match = re.search(r"raw Wilcoxon p=([0-9.eE+-]+)", claim.note)
        assert match is not None
        raw = float(match.group(1))
        assert claim.p_value >= raw - 1e-15
        assert f"all {len(statistical_claims)} registered claims" in claim.note
        adjusted_pairs.append((raw, claim.p_value))
    assert adjusted_pairs
    assert any(adjusted > raw for raw, adjusted in adjusted_pairs if raw > 0)
    overall = next(item for item in claims if item.claim_id == "P0_overall")
    assert overall.p_value is None
    assert overall.multiplicity_method == "derived_after_holm(no_additional_test)"


def test_twenty_seed_primary_ei_phase2_claims_are_evaluated() -> None:
    claims = {claim.claim_id: claim for claim in evaluate_core_claims(_phase2_formal())}
    for claim_id in (
        "B1a_local_absolute_accuracy",
        "B1b_local_relative_noninferiority",
        "B2_gate_reduces_switch_cost",
        "B3_homeostasis_stabilizes",
        "B4_local_rank_below_full_feedback",
    ):
        assert claims[claim_id].conclusion == "support"
        assert claims[claim_id].n_complete == 20


def test_thirty_seed_p0_claims_require_both_budget_panels() -> None:
    claims = {item.claim_id: item for item in evaluate_core_claims(_p0_formal())}
    assert not any(claim_id.endswith("_panel") for claim_id in claims)
    p0_ids = {
        "P0a_aligned_task_improves_prediction_vs_frozen",
        "P0b_aligned_task_beats_shuffled",
        "P0c_aligned_adds_value_over_matched_homeostasis",
        "P0d_local_absolute_accuracy",
        "P0e_local_noninferior_tuned_bptt",
        "P0f_local_noninferior_tuned_gru",
    }
    assert all(claims[item].conclusion == "support" for item in p0_ids)
    assert all(claims[item].n_complete == 30 for item in p0_ids)
    joint = claims["P0b_aligned_task_beats_shuffled"]
    panel_p = [float(value) for value in re.findall(r"raw_p=([0-9.eE+-]+)", joint.note)]
    registered = re.search(r"raw Wilcoxon p=([0-9.eE+-]+)", joint.note)
    assert len(panel_p) == 2 and registered is not None
    assert float(registered.group(1)) == pytest.approx(max(panel_p))
    assert claims["P0_overall"].conclusion == "support"
    assert (
        "P0a_aligned_task_improves_prediction_vs_frozen=support"
        in claims["P0_overall"].note
    )

    incomplete = _p0_formal()
    incomplete.loc[
        (incomplete["seed"] == 29)
        & incomplete["condition"].isin(
            [
                "task-homeostasis__aligned__l2",
                "task-homeostasis-normalization__aligned__l2",
            ]
        ),
        "budget_match_valid",
    ] = False
    claims = {item.claim_id: item for item in evaluate_core_claims(incomplete)}
    for claim_id in (
        "P0c_aligned_adds_value_over_matched_homeostasis",
        "P0d_local_absolute_accuracy",
        "P0e_local_noninferior_tuned_bptt",
        "P0f_local_noninferior_tuned_gru",
    ):
        assert claims[claim_id].conclusion == "inconclusive"
        assert claims[claim_id].n_complete == 29
        assert claims[claim_id].n_failed == 1
    assert claims["P0_overall"].conclusion == "inconclusive"


def test_p0_opposite_l1_l2_directions_cannot_average_to_support() -> None:
    opposed = _p0_formal()
    opposed.loc[
        opposed["condition"].eq("task-only__aligned__l1"),
        "heldout_masked_mse",
    ] = 0.0
    opposed.loc[
        opposed["condition"].eq("task-only__aligned__l2"),
        "heldout_masked_mse",
    ] = 1.6

    claims = {item.claim_id: item for item in evaluate_core_claims(opposed)}
    for claim_id in (
        "P0a_aligned_task_improves_prediction_vs_frozen",
        "P0b_aligned_task_beats_shuffled",
    ):
        assert claims[claim_id].conclusion == "inconclusive"
        assert claims[claim_id].p_value is None
        assert "panel conclusions are not unanimous" in claims[claim_id].note
        assert "l1: conclusion=support" in claims[claim_id].note
        assert "l2: conclusion=oppose" in claims[claim_id].note
    assert claims["P0_overall"].conclusion == "inconclusive"


def test_p0_extra_seed_cannot_replace_a_missing_preregistered_seed() -> None:
    raw = _p0_formal()
    missing = raw["seed"].eq(29) & raw["condition"].eq("task-only__aligned__l2")
    raw = raw.loc[~missing].copy()
    extra = _p0_formal(1).copy()
    extra["seed"] = 30
    claims = {
        item.claim_id: item
        for item in evaluate_core_claims(pd.concat([raw, extra], ignore_index=True))
    }

    claim = claims["P0b_aligned_task_beats_shuffled"]
    assert claim.conclusion == "inconclusive"
    assert claim.n_complete == 29
    assert claim.n_failed == 0
    assert "29/30 planned seeds" in claim.note
    assert claims["P0_overall"].conclusion == "inconclusive"


def test_p0_joint_complete_count_uses_seed_intersection_across_panels() -> None:
    raw = _p0_formal()
    missing = (raw["seed"].eq(28) & raw["condition"].eq("task-only__aligned__l1")) | (
        raw["seed"].eq(29) & raw["condition"].eq("task-only__aligned__l2")
    )
    claims = {
        item.claim_id: item for item in evaluate_core_claims(raw.loc[~missing].copy())
    }

    claim = claims["P0b_aligned_task_beats_shuffled"]
    assert claim.conclusion == "inconclusive"
    assert claim.n_complete == 28
    assert claim.n_failed == 0
    assert "28/30 planned seeds" in claim.note


def test_p0_setup_failure_is_counted_as_seed_wide() -> None:
    raw = _p0_formal()
    raw = raw.loc[~raw["seed"].eq(29)].copy()
    setup_failure = pd.DataFrame(
        [
            {
                "profile": "formal",
                "experiment": "exp07_mechanism_identifiability",
                "status": "failed",
                "seed": 29,
                "condition": "setup",
                "error": "failed before the scientific grid was materialized",
            }
        ]
    )
    claims = {
        item.claim_id: item
        for item in evaluate_core_claims(
            pd.concat([raw, setup_failure], ignore_index=True)
        )
    }

    for claim_id in (
        "P0a_aligned_task_improves_prediction_vs_frozen",
        "P0b_aligned_task_beats_shuffled",
        "P0c_aligned_adds_value_over_matched_homeostasis",
        "P0d_local_absolute_accuracy",
        "P0e_local_noninferior_tuned_bptt",
        "P0f_local_noninferior_tuned_gru",
    ):
        assert claims[claim_id].conclusion == "inconclusive"
        assert claims[claim_id].n_complete == 29
        assert claims[claim_id].n_failed == 1
    assert claims["P0_overall"].n_complete == 29
    assert claims["P0_overall"].n_failed == 1


def test_p0_overall_counts_joint_seed_coverage_across_constituents() -> None:
    raw = _p0_formal()
    missing = (
        raw["seed"].eq(28)
        & raw["condition"].isin(["task-only__aligned__l1", "task-only__aligned__l2"])
    ) | (
        raw["seed"].eq(29)
        & raw["condition"].isin(
            [
                "task-homeostasis-normalization__aligned__l1",
                "task-homeostasis-normalization__aligned__l2",
            ]
        )
    )
    claims = {
        item.claim_id: item for item in evaluate_core_claims(raw.loc[~missing].copy())
    }

    assert claims["P0a_aligned_task_improves_prediction_vs_frozen"].n_complete == 29
    assert claims["P0d_local_absolute_accuracy"].n_complete == 29
    assert claims["P0_overall"].n_complete == 28
    assert claims["P0_overall"].n_failed == 0


@pytest.mark.parametrize(
    ("condition", "claim_id"),
    [
        ("task-only__aligned__l2", "P0b_aligned_task_beats_shuffled"),
        (
            "homeostasis-only__aligned__l2",
            "P0c_aligned_adds_value_over_matched_homeostasis",
        ),
        (
            "task-homeostasis__aligned__l2",
            "P0c_aligned_adds_value_over_matched_homeostasis",
        ),
        (
            "task-homeostasis-normalization__aligned__l2",
            "P0d_local_absolute_accuracy",
        ),
    ],
)
def test_p0_sparse_failure_dimensions_are_recovered_from_condition_name(
    condition: str,
    claim_id: str,
) -> None:
    raw = _p0_formal()
    missing = raw["seed"].eq(29) & raw["condition"].eq(condition)
    raw = raw.loc[~missing].copy()
    sparse_failure = pd.DataFrame(
        [
            {
                "profile": "formal",
                "experiment": "exp07_mechanism_identifiability",
                "status": "failed",
                "seed": 29,
                "condition": condition,
                "error": "condition failed before metrics existed",
            }
        ]
    )
    claims = {
        item.claim_id: item
        for item in evaluate_core_claims(
            pd.concat([raw, sparse_failure], ignore_index=True)
        )
    }

    claim = claims[claim_id]
    assert claim.conclusion == "inconclusive"
    assert claim.n_complete == 29
    assert claim.n_failed == 1
    assert "failed/invalid planned seeds=29" in claim.note


def test_p0_overall_opposes_when_a_holm_adjusted_constituent_opposes() -> None:
    opposed = _p0_formal()
    opposed.loc[
        opposed["condition"].isin(["task-only__aligned__l1", "task-only__aligned__l2"]),
        "heldout_masked_mse",
    ] = 1.3
    claims = {item.claim_id: item for item in evaluate_core_claims(opposed)}

    assert (
        claims["P0a_aligned_task_improves_prediction_vs_frozen"].conclusion == "oppose"
    )
    assert claims["P0b_aligned_task_beats_shuffled"].conclusion == "oppose"
    assert claims["P0_overall"].conclusion == "oppose"
    assert claims["P0_overall"].p_value is None


def test_p0_overall_propagates_holm_downgrade_of_raw_support() -> None:
    weak = _p0_formal()
    shuffled = weak["condition"].isin(
        ["task-only__shuffled__l1", "task-only__shuffled__l2"]
    )
    weak.loc[shuffled, "heldout_masked_mse"] = 0.80
    weak.loc[shuffled & weak["seed"].lt(7), "heldout_masked_mse"] = 0.95

    claims = {item.claim_id: item for item in evaluate_core_claims(weak)}
    constituent = claims["P0b_aligned_task_beats_shuffled"]
    raw = re.search(r"raw Wilcoxon p=([0-9.eE+-]+)", constituent.note)

    assert raw is not None and float(raw.group(1)) <= 0.05
    assert constituent.p_value is not None and constituent.p_value > 0.05
    assert constituent.conclusion == "inconclusive"
    assert claims["P0_overall"].conclusion == "inconclusive"


def test_thirty_seed_p1_theorem_claims_are_separate_from_behavior() -> None:
    claims = {item.claim_id: item for item in evaluate_core_claims(_p1_formal())}
    p1_ids = {
        "P1a_masked_outer_product_identity",
        "P1b_credit_tangent_respects_feedback_bound",
        "P1c_highrank_physical_update_coexists_with_lowdim_credit",
    }
    assert all(claims[item].conclusion == "support" for item in p1_ids)
    assert all(claims[item].n_complete == 30 for item in p1_ids)
    assert (
        "does not imply held-out task support"
        in claims["P1c_highrank_physical_update_coexists_with_lowdim_credit"].criterion
    )

    incomplete = {item.claim_id: item for item in evaluate_core_claims(_p1_formal(29))}
    assert all(incomplete[item].conclusion == "inconclusive" for item in p1_ids)


def test_phase2_claims_do_not_fallback_to_nonprimary_architecture() -> None:
    claims = {
        claim.claim_id: claim
        for claim in evaluate_core_claims(_phase2_formal(architecture="non_dale_n256"))
    }
    assert all(
        claims[claim_id].conclusion == "inconclusive"
        for claim_id in (
            "B1a_local_absolute_accuracy",
            "B1b_local_relative_noninferiority",
            "B2_gate_reduces_switch_cost",
            "B3_homeostasis_stabilizes",
            "B4_local_rank_below_full_feedback",
        )
    )


def test_phase_match_requires_all_exact_flags_and_twenty_complete_seeds() -> None:
    raw = _phase4_formal()
    claims = {claim.claim_id: claim for claim in evaluate_core_claims(raw)}
    assert claims["C1_phase_effect_survives_rate_match"].conclusion == "support"

    raw.loc[raw.index[0], "mean_coupling_match_exact"] = False
    claims = {claim.claim_id: claim for claim in evaluate_core_claims(raw)}
    assert claims["C1_phase_effect_survives_rate_match"].conclusion == "inconclusive"
    assert "flags are false" in claims["C1_phase_effect_survives_rate_match"].note


def test_only_failed_required_phase_cells_prevent_support() -> None:
    raw = _phase4_formal()
    unrelated = pd.concat(
        [
            raw,
            pd.DataFrame(
                [
                    {
                        "profile": "formal",
                        "experiment": "exp04_phase_gating",
                        "status": "failed",
                        "seed": 99,
                        "phase_condition": "anti_phase",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    claim = next(
        item
        for item in evaluate_core_claims(unrelated)
        if item.claim_id == "C1_phase_effect_survives_rate_match"
    )
    assert claim.conclusion == "support"
    assert claim.n_failed == 0

    relevant = raw.loc[
        ~((raw["seed"] == 0) & (raw["phase_condition"] == "no_oscillation"))
    ].copy()
    relevant = pd.concat(
        [
            relevant,
            pd.DataFrame(
                [
                    {
                        "profile": "formal",
                        "experiment": "exp04_phase_gating",
                        "status": "failed",
                        "seed": 0,
                        "phase_condition": "no_oscillation",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    claim = next(
        item
        for item in evaluate_core_claims(relevant)
        if item.claim_id == "C1_phase_effect_survives_rate_match"
    )
    assert claim.conclusion == "inconclusive"
    assert claim.n_failed == 1


def test_latest_immutable_run_attempt_supersedes_old_failure() -> None:
    current = _phase4_formal()
    current["run_id"] = current["seed"].map(lambda seed: f"new-{seed}")
    current["recorded_at"] = "2026-07-10T12:00:00Z"
    old = pd.DataFrame(
        [
            {
                "profile": "formal",
                "experiment": "exp04_phase_gating",
                "status": "failed",
                "seed": 0,
                "phase_condition": "in_phase",
                "run_id": "old-0",
                "recorded_at": "2026-07-10T11:00:00Z",
            }
        ]
    )
    claim = next(
        item
        for item in evaluate_core_claims(pd.concat([old, current], ignore_index=True))
        if item.claim_id == "C1_phase_effect_survives_rate_match"
    )
    assert claim.conclusion == "support"
    assert claim.n_failed == 0


def test_latest_attempt_uses_start_time_not_last_metric_time() -> None:
    current = _phase4_formal()
    current["run_id"] = current["seed"].map(lambda seed: f"retry-{seed}")
    current["run_started_at"] = "2026-07-10T12:00:00Z"
    current["recorded_at"] = "2026-07-10T12:30:00Z"
    old = _phase4_formal(1)
    old["run_id"] = "old-0"
    # Legacy attempt-directory timestamps use compact ISO form.
    old["run_started_at"] = "20260710T110000.000000Z"
    old["recorded_at"] = "2026-07-10T13:00:00Z"
    old["decoding_accuracy"] = old["phase_condition"].map(
        {"in_phase": 0.40, "no_oscillation": 0.80}
    )

    claim = next(
        item
        for item in evaluate_core_claims(pd.concat([old, current], ignore_index=True))
        if item.claim_id == "C1_phase_effect_survives_rate_match"
    )
    assert claim.conclusion == "support"
    assert claim.n_complete == 20


def test_latest_nonterminal_attempt_invalidates_streamed_complete_cells() -> None:
    old = _phase4_formal()
    old["run_id"] = old["seed"].map(lambda seed: f"old-{seed}")
    old["recorded_at"] = "2026-07-10T11:00:00Z"
    old["run_status"] = "complete"
    partial = pd.DataFrame(
        [
            {
                "profile": "formal",
                "experiment": "exp04_phase_gating",
                "status": "complete",
                "seed": 0,
                "phase_condition": "in_phase",
                "decoding_accuracy": 0.99,
                "run_id": "partial-0",
                "recorded_at": "2026-07-10T12:00:01Z",
                "run_status": "running",
            },
            {
                "profile": "formal",
                "experiment": "exp04_phase_gating",
                "status": "failed",
                "seed": 0,
                "run_id": "partial-0",
                "recorded_at": "2026-07-10T12:00:00Z",
                "run_status": "running",
                "run_level_failure": True,
            },
        ]
    )
    claim = next(
        item
        for item in evaluate_core_claims(pd.concat([old, partial], ignore_index=True))
        if item.claim_id == "C1_phase_effect_survives_rate_match"
    )
    assert claim.conclusion == "inconclusive"
    assert claim.n_complete == 19
    assert claim.n_failed == 1


def test_empty_phase2_run_failure_is_counted_for_primary_claims() -> None:
    complete = _phase2_formal(19)
    non_primary = pd.DataFrame(
        [
            {
                "profile": "formal",
                "experiment": experiment,
                "status": "failed",
                "seed": 19,
                "architecture": "non_dale_n256",
                "model_kind": "non_dale",
                "condition": "local",
            }
            for experiment in (
                "exp02_context_ei_oracle_gate",
                "exp03_context_ei_learned_gate",
            )
        ]
    )
    failure = pd.DataFrame(
        [
            {
                "profile": "formal",
                "experiment": experiment,
                "status": "failed",
                "seed": 19,
                "run_id": f"failed-{experiment}",
                "run_started_at": "2026-07-10T12:00:00Z",
                "recorded_at": "2026-07-10T12:00:01Z",
                "run_status": "failed",
                "run_level_failure": True,
            }
            for experiment in (
                "exp02_context_ei_oracle_gate",
                "exp03_context_ei_learned_gate",
            )
        ]
    )

    claims = {
        item.claim_id: item
        for item in evaluate_core_claims(
            pd.concat([complete, non_primary, failure], ignore_index=True)
        )
    }
    for claim_id in (
        "B1a_local_absolute_accuracy",
        "B1b_local_relative_noninferiority",
        "B2_gate_reduces_switch_cost",
        "B3_homeostasis_stabilizes",
        "B4_local_rank_below_full_feedback",
    ):
        assert claims[claim_id].conclusion == "inconclusive"
        assert claims[claim_id].n_complete == 19
        assert claims[claim_id].n_failed == 1


def test_explicit_non_primary_phase2_failure_does_not_contaminate_claims() -> None:
    complete = _phase2_formal()
    non_primary = pd.DataFrame(
        [
            {
                "profile": "formal",
                "experiment": experiment,
                "status": "failed",
                "seed": 0,
                "architecture": "non_dale_n256",
                "model_kind": "non_dale",
                "condition": "local",
            }
            for experiment in (
                "exp02_context_ei_oracle_gate",
                "exp03_context_ei_learned_gate",
            )
        ]
    )

    claims = {
        item.claim_id: item
        for item in evaluate_core_claims(
            pd.concat([complete, non_primary], ignore_index=True)
        )
    }
    for claim_id in (
        "B1a_local_absolute_accuracy",
        "B1b_local_relative_noninferiority",
        "B2_gate_reduces_switch_cost",
        "B3_homeostasis_stabilizes",
        "B4_local_rank_below_full_feedback",
    ):
        assert claims[claim_id].conclusion == "support"
        assert claims[claim_id].n_complete == 20
        assert claims[claim_id].n_failed == 0


def test_real_data_folds_aggregate_to_animal_and_do_not_promote_two_animals() -> None:
    rows: list[dict[str, object]] = []
    for animal in ("a0", "a1"):
        for session_index in (0, 1):
            session = f"{animal}-s{session_index}"
            for fold in range(5):
                base = {
                    "profile": "formal",
                    "experiment": "exp05_sequence_real_data",
                    "status": "complete",
                    "animal_id": animal,
                    "session_id": session,
                    "fold": fold,
                }
                for model, nll, parameters in (
                    ("common", 2.0, 10),
                    ("shared", 1.04, 20),
                    ("full", 1.0, 50),
                ):
                    rows.append(
                        {
                            **base,
                            "model_family": model,
                            "heldout_nll_per_scalar": nll,
                            "parameter_count": parameters,
                        }
                    )
            for model, nll, parameters in (
                ("common", 2.0, 10),
                ("shared", 1.0, 20),
                ("full", 1.2, 50),
            ):
                rows.append(
                    {
                        "profile": "formal",
                        "experiment": "exp05_sequence_real_data",
                        "status": "complete",
                        "animal_id": animal,
                        "session_id": session,
                        "fold": "unseen_combination",
                        "model_family": model,
                        "heldout_nll_per_scalar": nll,
                        "parameter_count": parameters,
                    }
                )
    claims = {
        claim.claim_id: claim for claim in evaluate_core_claims(pd.DataFrame(rows))
    }
    assert claims["D1_shared_basis_near_full"].conclusion == "inconclusive"
    assert claims["D2_unseen_sequence_generalization"].conclusion == "inconclusive"
    assert claims["D1_shared_basis_near_full"].stats_unit == "animal"
    assert claims["D1_shared_basis_near_full"].n_complete == 2


def test_streamed_sequence_session_failure_invalidates_earlier_complete_folds() -> None:
    rows: list[dict[str, object]] = []
    for session in ("s0", "s1"):
        for model, nll, parameters in (
            ("common", 2.0, 10),
            ("shared", 1.04, 20),
            ("full", 1.0, 50),
        ):
            rows.append(
                {
                    "profile": "formal",
                    "experiment": "exp05_sequence_real_data",
                    "status": "complete",
                    "session_id": session,
                    "fold": 0,
                    "model_family": model,
                    "heldout_nll_per_scalar": nll,
                    "parameter_count": parameters,
                }
            )
    rows.append(
        {
            "profile": "formal",
            "experiment": "exp05_sequence_real_data",
            "status": "failed",
            "session_id": "s0",
            "error": "later fold failed",
        }
    )
    claim = next(
        item
        for item in evaluate_core_claims(pd.DataFrame(rows))
        if item.claim_id == "D1_shared_basis_near_full"
    )
    assert claim.conclusion == "inconclusive"
    assert claim.n_failed == 1


def test_ibl_lead_requires_both_views_and_never_cancels_a_failed_view() -> None:
    rows: list[dict[str, object]] = []
    for animal in ("a0", "a1"):
        for view in ("stimulus_pre", "movement_pre"):
            rows.append(
                {
                    "profile": "formal",
                    "experiment": "exp06_ibl_context_switch",
                    "status": "complete",
                    "animal_id": animal,
                    "session_id": f"{animal}-s0",
                    "view": view,
                    "model_family": "lead_lag",
                    "latent_lead_trials": 2.0,
                    "condition_schedule_observed": False,
                    "lead_lag_is_causal_claim": False,
                }
            )
    rows.append(
        {
            "profile": "formal",
            "experiment": "exp06_ibl_context_switch",
            "status": "failed",
            "animal_id": "a0",
            "session_id": "a0-s0",
            "view": "movement_pre",
        }
    )
    claim = next(
        item
        for item in evaluate_core_claims(pd.DataFrame(rows))
        if item.claim_id == "E2_latent_precedes_behavior_bias"
    )
    assert claim.conclusion == "inconclusive"
    assert claim.n_complete == 1
    assert claim.n_failed == 1


def test_collect_runs_handles_empty_results(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    raw, runs = collect_runs(tmp_path)
    assert raw.empty and runs.empty


def test_collect_runs_materializes_empty_top_level_failure(tmp_path: Path) -> None:
    try:
        with ExperimentRun(
            "failed_experiment",
            0,
            {"profile": "formal"},
            results_root=tmp_path,
        ):
            raise RuntimeError("setup exploded")
    except RuntimeError:
        pass
    raw, runs = collect_runs(tmp_path)
    assert len(runs) == 1
    assert len(raw) == 1
    assert raw.iloc[0]["status"] == "failed"
    assert bool(raw.iloc[0]["run_level_failure"])
    assert raw.iloc[0]["error"] == "setup exploded"


def test_collect_runs_invalidates_partial_metrics_after_top_level_failure(
    tmp_path: Path,
) -> None:
    try:
        with ExperimentRun(
            "exp04_phase_gating",
            0,
            {"profile": "formal"},
            results_root=tmp_path,
        ) as run:
            run.record(
                {
                    "status": "complete",
                    "decoding_accuracy": 0.99,
                    "mean_rate_match_exact": True,
                    "per_trial_spike_count_match_exact": True,
                    "mean_coupling_match_exact": True,
                    "shared_source_fingerprint": "partial-source",
                },
                phase_condition="in_phase",
            )
            raise RuntimeError("failed after one streamed cell")
    except RuntimeError:
        pass
    partial, _ = collect_runs(tmp_path)
    assert set(partial["run_status"]) == {"failed"}
    remaining = _phase4_formal().loc[lambda frame: frame["seed"].ne(0)]
    claim = next(
        item
        for item in evaluate_core_claims(
            pd.concat([remaining, partial], ignore_index=True)
        )
        if item.claim_id == "C1_phase_effect_survives_rate_match"
    )
    assert claim.conclusion == "inconclusive"
    assert claim.n_complete == 19
    assert claim.n_failed == 1
