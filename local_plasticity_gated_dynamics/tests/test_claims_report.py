from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_report import (
    PORTABLE_RUNS_ROOT,
    REDACTED_HOST_TEXT,
    _assert_no_host_paths,
    _portable_run_path,
    collect_runs,
    merge_compact_snapshot,
    write_compact_raw,
    write_compact_runs,
    write_report,
)
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
                "hidden_context_task": True,
                "cue_encodes_observation_not_state": True,
                "gate_test_accessed_true_context": False,
                "gate_fit_accessed_true_context": False,
                "third_factor_accessed_true_context": False,
                "oracle_warm_start_used": False,
                "md_fit_used_context_bias": False,
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
    assert len(claims) == 36
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
    assert "## P2 formal diagnostics" not in report


def test_report_adds_formal_p2_macro_and_fit_diagnostics(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    gates = (
        "oracle_bayes",
        "supervised_upper_bound",
        "learned_hmm",
        "md_recurrent_belief",
        "no_gate",
    )
    gate_nll = {
        "oracle_bayes": 0.2,
        "supervised_upper_bound": 0.22,
        "learned_hmm": 0.25,
        "md_recurrent_belief": 0.3,
        "no_gate": 0.69,
    }
    for seed in range(2):
        for q in (0.55, 0.70, 0.85, 1.0):
            for hazard in (0.01, 0.05, 0.10, 0.20):
                cell_offset = 0.02 * (1.0 - q) + 0.01 * hazard
                for gate in gates:
                    row: dict[str, object] = {
                        "experiment": "exp09_hidden_context_gate",
                        "profile": "formal",
                        "seed": seed,
                        "status": "complete",
                        "gate_model": gate,
                        "intervention": "none",
                        "cue_reliability": q,
                        "context_hazard": hazard,
                        "context_nll": gate_nll[gate] + cell_offset,
                        "context_brier": 0.1 + cell_offset,
                        "context_ece": 0.03 + cell_offset,
                        "switch_latency_trials": 1.0 + 2.0 * hazard,
                        "false_switch_rate": 0.01 + 0.01 * hazard,
                        "behavior_balanced_accuracy": 0.8 + 0.1 * q,
                        "energy_proxy_per_trial": 0.9 + 0.05 * q,
                    }
                    if gate == "learned_hmm":
                        row["hmm_fit_converged"] = not (
                            seed == 1 and q == 0.55 and hazard == 0.20
                        )
                        row["hmm_fit_iterations"] = 10 + seed
                    if gate == "md_recurrent_belief":
                        identifiable = q >= 0.70
                        row["md_moment_anchor_identifiable"] = identifiable
                        row["estimated_context_hazard"] = (
                            hazard if identifiable else 0.499999
                        )
                        row["estimated_cue_reliability"] = (
                            q if identifiable else 0.500001
                        )
                    rows.append(row)
    summary = pd.DataFrame(
        [
            {
                "claim_id": "P2i_md_energy",
                "criterion": "MD energy upper ratio CI <=1.10",
                "n_complete": 30,
                "n_planned": 30,
                "n_failed": 0,
                "estimate": np.log(0.90),
                "ci_low": np.log(0.85),
                "ci_high": np.log(0.95),
                "conclusion": "support",
                "note": "registered log energy ratio",
            }
        ]
    )

    write_report(tmp_path, pd.DataFrame(rows), pd.DataFrame(), summary)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert "## P2 formal diagnostics" in report
    assert (
        "macro average does not assert that the result holds in every q/h cell"
        in report
    )
    assert "| MD recurrent belief | 2 |" in report
    assert "Learned-HMM convergence: 31/32 reported fits converged" in report
    assert "non-converged fits are retained as a sensitivity caveat" in report
    assert "| q = 0.55 (weak cue) | 0/8 | 0 | 8/8 |" in report
    assert "| q >= 0.70 | 24/24 | 1 | unavailable |" in report
    assert "returns neutral parameter estimates (q̂≈0.5, ĥ≈0.5)" in report
    assert "### MD q/h-cell range" in report
    assert "### P2i energy-ratio interpretation" in report
    assert "energy ratio of 0.9 [0.85, 0.95]" in report


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
    derived = {"P0_overall", "P2_overall"}
    statistical_claims = [item for item in claims if item.claim_id not in derived]
    assert len(statistical_claims) == 34
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


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_provenance",
        "hidden_context_false",
        "cue_encodes_state",
        "gate_test_context",
        "gate_fit_context",
        "true_context_third_factor",
        "oracle_warm_start",
        "supervised_context_bias",
        "string_false_pollution",
    ],
)
def test_legacy_or_leaky_phase2_gate_cannot_support_hidden_context_claim(
    mutation: str,
) -> None:
    raw = _phase2_formal()
    if mutation == "missing_provenance":
        raw = raw.drop(
            columns=[
                "hidden_context_task",
                "cue_encodes_observation_not_state",
                "gate_test_accessed_true_context",
                "third_factor_accessed_true_context",
            ]
        )
    else:
        field, value = {
            "hidden_context_false": ("hidden_context_task", False),
            "cue_encodes_state": ("cue_encodes_observation_not_state", False),
            "gate_test_context": ("gate_test_accessed_true_context", True),
            "gate_fit_context": ("gate_fit_accessed_true_context", True),
            "true_context_third_factor": (
                "third_factor_accessed_true_context",
                True,
            ),
            "oracle_warm_start": ("oracle_warm_start_used", True),
            "supervised_context_bias": ("md_fit_used_context_bias", True),
            "string_false_pollution": ("hidden_context_task", "False"),
        }[mutation]
        raw[field] = value
    claim = next(
        item
        for item in evaluate_core_claims(raw)
        if item.claim_id == "B2_gate_reduces_switch_cost"
    )
    assert claim.conclusion == "inconclusive"
    assert claim.n_complete == 0
    assert "supervised/oracle-warm-start" in claim.note


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


def test_compact_csv_boolean_tokens_preserve_p0_and_p1_claims() -> None:
    p0 = _p0_formal()
    p0["budget_match_valid"] = p0["budget_match_valid"].map(
        lambda value: "True" if value is True else value
    )
    p1 = _p1_formal()
    p1["requested_feedback_dim"] = "4"
    p1["geometry_valid"] = "True"
    claims = {
        item.claim_id: item
        for item in evaluate_core_claims(pd.concat([p0, p1], ignore_index=True))
    }

    assert claims["P0_overall"].conclusion == "support"
    assert claims["P1a_masked_outer_product_identity"].conclusion == "support"
    assert claims["P1b_credit_tangent_respects_feedback_bound"].conclusion == "support"
    assert (
        claims["P1c_highrank_physical_update_coexists_with_lowdim_credit"].conclusion
        == "support"
    )


def test_invalid_compact_boolean_tokens_fail_closed() -> None:
    p0 = _p0_formal()
    p0.loc[p0["seed"].eq(29), "budget_match_valid"] = "garbage"
    p1 = _p1_formal()
    p1["geometry_valid"] = p1["geometry_valid"].astype(object)
    p1.loc[p1["seed"].eq(29), "geometry_valid"] = "garbage"
    claims = {
        item.claim_id: item
        for item in evaluate_core_claims(pd.concat([p0, p1], ignore_index=True))
    }

    p0_claim = claims["P0b_aligned_task_beats_shuffled"]
    assert p0_claim.conclusion == "inconclusive"
    assert p0_claim.n_complete == 29
    assert p0_claim.n_failed == 1
    assert claims["P0_overall"].conclusion == "inconclusive"
    for claim_id in (
        "P1a_masked_outer_product_identity",
        "P1b_credit_tangent_respects_feedback_bound",
        "P1c_highrank_physical_update_coexists_with_lowdim_credit",
    ):
        assert claims[claim_id].conclusion == "inconclusive"
        assert claims[claim_id].n_complete == 29


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


def test_ibl_stimulus_pre_lead_does_not_average_or_inherit_movement_failure() -> None:
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
                    "behavior_bias_used_true_block_boundaries": False,
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
    assert claim.n_complete == 2
    assert claim.n_failed == 0
    assert "strict E1" in claim.note


def _valid_multianimal_ibl_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for animal_index in range(10):
        animal = f"animal-{animal_index}"
        for session_index in range(2):
            session = f"{animal}-session-{session_index}"
            for fold in range(2):
                for model, nll, parameters in (
                    ("common", 2.0, 10),
                    ("shared", 1.05, 20),
                    ("full", 1.0, 50),
                ):
                    rows.append(
                        {
                            "profile": "formal",
                            "experiment": "exp06_ibl_context_switch",
                            "status": "complete",
                            "animal_id": animal,
                            "session_id": session,
                            "view": "stimulus_pre",
                            "fold": fold,
                            "model_family": model,
                            "heldout_nll_per_scalar": nll,
                            "parameter_count": parameters,
                            "hierarchical_observation_model": True,
                            "nested_cv_latent_dimension": True,
                            "unit_qc_applied": True,
                            "context_coverage_valid": True,
                            "parameter_count_includes_preprocessing": True,
                            "hidden_context_inference": True,
                            "test_context_observed": False,
                            "belief_filter_used_true_block_boundaries": False,
                            "condition_schedule_observed": False,
                        }
                    )
    return pd.DataFrame(rows)


def test_ibl_primary_claim_requires_full_cohort_and_method_provenance() -> None:
    valid = _valid_multianimal_ibl_panel()
    claim = next(
        item
        for item in evaluate_core_claims(valid)
        if item.claim_id == "E1_ibl_shared_switching"
    )
    assert claim.conclusion == "support"
    assert claim.stats_unit == "animal"
    assert claim.n_complete == 10
    assert "full-minus-shared parameter CI" in claim.note

    invalid = valid.copy()
    invalid["context_coverage_valid"] = "False"
    invalid_claim = next(
        item
        for item in evaluate_core_claims(invalid)
        if item.claim_id == "E1_ibl_shared_switching"
    )
    assert invalid_claim.conclusion == "inconclusive"
    assert "context_coverage_valid" in invalid_claim.note


def test_ibl_primary_claim_rejects_any_invalid_retained_gain_denominator() -> None:
    invalid = _valid_multianimal_ibl_panel()
    affected = invalid["animal_id"].isin(
        [f"animal-{index}" for index in range(5)]
    ) & invalid["model_family"].eq("full")
    invalid.loc[affected, "heldout_nll_per_scalar"] = 2.1

    claim = next(
        item
        for item in evaluate_core_claims(invalid)
        if item.claim_id == "E1_ibl_shared_switching"
    )
    assert claim.conclusion == "inconclusive"
    assert "positive full-vs-common gain denominator" in claim.note


@pytest.mark.parametrize("bad_parameter", [np.nan, 20.5])
def test_ibl_primary_claim_rejects_partial_or_noninteger_parameter_count(
    bad_parameter: float,
) -> None:
    invalid = _valid_multianimal_ibl_panel()
    invalid["parameter_count"] = invalid["parameter_count"].astype(float)
    target = (
        invalid["animal_id"].eq("animal-0")
        & invalid["fold"].eq(1)
        & invalid["model_family"].eq("shared")
    )
    invalid.loc[target, "parameter_count"] = bad_parameter

    claim = next(
        item
        for item in evaluate_core_claims(invalid)
        if item.claim_id == "E1_ibl_shared_switching"
    )
    assert claim.conclusion == "inconclusive"
    assert "parameter_count" in claim.note


def test_ibl_primary_claim_rejects_duplicate_model_cell() -> None:
    valid = _valid_multianimal_ibl_panel()
    duplicate = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)

    claim = next(
        item
        for item in evaluate_core_claims(duplicate)
        if item.claim_id == "E1_ibl_shared_switching"
    )
    assert claim.conclusion == "inconclusive"
    assert "duplicate" in claim.note


def test_ibl_primary_claim_requires_hidden_context_provenance() -> None:
    invalid = _valid_multianimal_ibl_panel().drop(columns=["test_context_observed"])

    claim = next(
        item
        for item in evaluate_core_claims(invalid)
        if item.claim_id == "E1_ibl_shared_switching"
    )
    assert claim.conclusion == "inconclusive"
    assert "test_context_observed" in claim.note


def _ibl_panel_with_lead_records(*, lead_sessions: int = 20) -> pd.DataFrame:
    model_rows = _valid_multianimal_ibl_panel()
    sessions = (
        model_rows[["animal_id", "session_id"]]
        .drop_duplicates()
        .sort_values(["animal_id", "session_id"])
        .head(lead_sessions)
    )
    lead_rows: list[dict[str, object]] = []
    for row in sessions.itertuples(index=False):
        lead_rows.append(
            {
                "profile": "formal",
                "experiment": "exp06_ibl_context_switch",
                "status": "complete",
                "animal_id": row.animal_id,
                "session_id": row.session_id,
                "view": "stimulus_pre",
                "model_family": "lead_lag",
                "latent_lead_trials": 2.0,
                "hierarchical_observation_model": True,
                "nested_cv_latent_dimension": True,
                "unit_qc_applied": True,
                "context_coverage_valid": True,
                "parameter_count_includes_preprocessing": True,
                "hidden_context_inference": True,
                "test_context_observed": False,
                "belief_filter_used_true_block_boundaries": False,
                "condition_schedule_observed": False,
                "lead_lag_is_causal_claim": False,
                "behavior_bias_used_true_block_boundaries": False,
            }
        )
    return pd.concat([model_rows, pd.DataFrame(lead_rows)], ignore_index=True)


def test_ibl_lead_claim_requires_exact_same_twenty_session_cohort() -> None:
    complete_claim = next(
        item
        for item in evaluate_core_claims(_ibl_panel_with_lead_records())
        if item.claim_id == "E2_latent_precedes_behavior_bias"
    )
    assert complete_claim.conclusion == "support"
    assert complete_claim.n_complete == 10
    assert "10 animals/20 sessions" in complete_claim.note

    incomplete_claim = next(
        item
        for item in evaluate_core_claims(_ibl_panel_with_lead_records(lead_sessions=10))
        if item.claim_id == "E2_latent_precedes_behavior_bias"
    )
    assert incomplete_claim.conclusion == "inconclusive"
    assert "exactly match" in incomplete_claim.note


def test_ibl_lead_claim_rejects_swapped_session_animal_mapping() -> None:
    invalid = _ibl_panel_with_lead_records()
    lead_mask = invalid["model_family"].eq("lead_lag")
    lead_animals = invalid.loc[lead_mask, "animal_id"].to_numpy(copy=True)
    invalid.loc[lead_mask, "animal_id"] = np.roll(lead_animals, 2)

    claim = next(
        item
        for item in evaluate_core_claims(invalid)
        if item.claim_id == "E2_latent_precedes_behavior_bias"
    )
    assert claim.conclusion == "inconclusive"
    assert "exactly match" in claim.note


def test_collect_runs_handles_empty_results(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    raw, runs = collect_runs(tmp_path)
    assert raw.empty and runs.empty


def test_collect_runs_publishes_portable_paths_and_keeps_raw_runs_ignored(
    tmp_path: Path,
) -> None:
    with ExperimentRun(
        "portable_collection",
        3,
        {"profile": "smoke"},
        results_root=tmp_path,
    ) as run:
        run.record({"metric": 1.0})
        expected_run_id = run.run_id

    raw, runs = collect_runs(tmp_path)
    assert raw["run_id"].tolist() == [expected_run_id]
    assert runs["run_id"].tolist() == [expected_run_id]
    assert raw.iloc[0]["run_path"].startswith(
        f"{PORTABLE_RUNS_ROOT}/portable_collection/seed_0003/"
    )
    assert runs.iloc[0]["path"] == raw.iloc[0]["run_path"]
    assert str(tmp_path.resolve()) not in raw.iloc[0]["run_path"]

    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(
        encoding="utf-8"
    )
    assert "results/runs/" in gitignore.splitlines()


@pytest.mark.parametrize(
    ("source", "relative"),
    [
        (
            "/home/user/project/results/runs/exp/seed_0000/attempt",
            "exp/seed_0000/attempt",
        ),
        (
            "/mnt/work/project/results/runs/exp/seed_0000/attempt",
            "exp/seed_0000/attempt",
        ),
        (
            r"C:\Users\user\project\results\runs\exp\seed_0000\attempt",
            "exp/seed_0000/attempt",
        ),
        (
            "D:/work/project/results/runs/exp/seed_0000/attempt",
            "exp/seed_0000/attempt",
        ),
        (
            r"\\server\share\project\results\runs\exp\seed_0000\attempt",
            "exp/seed_0000/attempt",
        ),
        (
            "//server/share/project/results/runs/exp/seed_0000/attempt",
            "exp/seed_0000/attempt",
        ),
        ("results/runs/exp/seed_0000/attempt", "exp/seed_0000/attempt"),
        ("runs/exp/seed_0000/attempt", "exp/seed_0000/attempt"),
        (
            f"{PORTABLE_RUNS_ROOT}/exp/seed_0000/attempt",
            "exp/seed_0000/attempt",
        ),
        (
            rf"{PORTABLE_RUNS_ROOT}\exp\seed_0000\attempt",
            "exp/seed_0000/attempt",
        ),
        ("/srv/private/runs/patient-007/session", None),
        (r"\\server\share\patient-007\session", None),
        ("../results/runs/exp/seed_0000/attempt", None),
        ("runs/../private", None),
        ("other/results/runs/exp/seed_0000/attempt", None),
    ],
)
def test_portable_run_path_accepts_only_project_run_locations(
    source: str, relative: str | None
) -> None:
    portable = _portable_run_path(source)
    if relative is None:
        assert str(portable).startswith(f"{PORTABLE_RUNS_ROOT}/_sanitized/")
    else:
        assert portable == f"{PORTABLE_RUNS_ROOT}/{relative}"


def test_portable_run_path_preserves_missing_and_empty_values() -> None:
    assert _portable_run_path(None) is None
    assert _portable_run_path("") == ""
    assert pd.isna(_portable_run_path(np.nan))


@pytest.mark.parametrize(
    "text",
    [
        json.dumps({"path": r"\\server\share name\secret,old.txt"}),
        json.dumps({"path": r"C:\Users\John Doe\secret,old.txt"}),
        json.dumps({"path": "/home/John Doe/secret,old.txt"}),
    ],
)
def test_final_path_audit_independently_rejects_json_escaped_paths(
    text: str,
) -> None:
    with pytest.raises(ValueError, match="column 'details', row 0"):
        _assert_no_host_paths(pd.DataFrame([{"details": text}]))


def test_collect_runs_redacts_paths_inside_failed_condition_text(
    tmp_path: Path,
) -> None:
    normal_text = (
        "mirror https://example.org/data; ratio 1/2; support/oppose; "
        "relative ./file ../other ~/cache relative/path"
    )
    with ExperimentRun(
        "portable_failure",
        4,
        {"profile": "smoke"},
        results_root=tmp_path,
    ) as run:
        run.mark_condition_failure(
            FileNotFoundError(
                "missing /mnt/John Doe/trials,private.csv; "
                "cache C:/Users/John Doe/cache; "
                rf"UNC \\server\share\private; {normal_text}"
            ),
            condition="runtime",
        )
        run.record_failed_condition(
            {
                "failure_reason": r"invalid cache D:\private model\model,old.bin",
                "note": normal_text,
                "details": {
                    "unc": r"\\server\share name\secret,old.txt",
                    "relative": "relative/file.txt",
                },
            },
            condition="scientific",
        )

    raw, _ = collect_runs(tmp_path)
    error = raw.set_index("condition").loc["runtime", "error"]
    reason = raw.set_index("condition").loc["scientific", "failure_reason"]
    redacted_pattern = rf"{re.escape(REDACTED_HOST_TEXT)}/[0-9a-f]{{24}}"
    assert re.fullmatch(redacted_pattern, error)
    assert re.fullmatch(redacted_pattern, reason)
    assert raw.set_index("condition").loc["scientific", "note"] == normal_text
    details = json.loads(raw.set_index("condition").loc["scientific", "details"])
    assert re.fullmatch(redacted_pattern, details["unc"])
    assert details["relative"] == "relative/file.txt"
    assert "/mnt/" not in error
    assert "C:/" not in error
    assert "\\\\server\\" not in error


def test_merge_and_write_sanitize_existing_and_discovered_host_paths(
    tmp_path: Path,
) -> None:
    existing_raw = pd.DataFrame(
        [
            {
                "run_id": "posix-run",
                "metric": 1.0,
                "run_path": (
                    "/home/researcher/project/results/runs/exp_posix/"
                    "seed_0000/attempt_a"
                ),
                "details": json.dumps(
                    {
                        "unc": r"\\server\share name\secret,old.txt",
                        "relative": "relative/file.txt",
                    }
                ),
            }
        ]
    )
    existing_runs = pd.DataFrame(
        [
            {
                "run_id": "posix-run",
                "status": "complete",
                "path": (
                    "/home/researcher/project/results/runs/exp_posix/"
                    "seed_0000/attempt_a"
                ),
            }
        ]
    )
    existing_raw.to_csv(tmp_path / "raw_metrics.csv", index=False)
    existing_runs.to_csv(tmp_path / "runs.csv", index=False)
    discovered_raw = pd.DataFrame(
        [
            {
                "run_id": "windows-run",
                "metric": 2.0,
                "run_path": (
                    r"C:\Users\Researcher\project\results\runs\exp_windows"
                    r"\seed_0001\attempt_b"
                ),
                "error": "missing /mnt/John Doe/data/trials,private.csv",
            },
            {
                "run_id": "opaque-run",
                "metric": 3.0,
                "run_path": r"E:\private\legacy-artifact",
                "failure_reason": r"cache D:\private model\model,old.bin unavailable",
                "note": (
                    "mirror https://example.org/a; ratio 1/2; support/oppose; "
                    "relative ./file ../other ~/cache relative/path"
                ),
                "details": {
                    "unc": r"\\server\share name\secret,old.txt",
                    "windows": r"C:\Users\John Doe\secret,old.txt",
                    "posix": "/home/John Doe/secret,old.txt",
                    "relative": "relative/path.txt",
                    "url": "https://example.org/data/file.txt",
                },
            },
        ]
    )
    discovered_runs = pd.DataFrame(
        [
            {
                "run_id": "windows-run",
                "status": "complete",
                "path": (
                    r"E:\checkout\results\runs\exp_windows"
                    r"\seed_0001\attempt_b"
                ),
            },
            {
                "run_id": "opaque-run",
                "status": "failed",
                "path": r"C:\Users\Researcher\opaque-artifact",
            },
        ]
    )

    raw, runs = merge_compact_snapshot(tmp_path, discovered_raw, discovered_runs)
    assert set(raw["run_id"]) == {"posix-run", "windows-run", "opaque-run"}
    assert set(runs["run_id"]) == {"posix-run", "windows-run", "opaque-run"}
    assert raw.set_index("run_id").loc["posix-run", "run_path"] == (
        f"{PORTABLE_RUNS_ROOT}/exp_posix/seed_0000/attempt_a"
    )
    assert raw.set_index("run_id").loc["windows-run", "run_path"] == (
        f"{PORTABLE_RUNS_ROOT}/exp_windows/seed_0001/attempt_b"
    )
    assert (
        raw.set_index("run_id")
        .loc["opaque-run", "run_path"]
        .startswith(f"{PORTABLE_RUNS_ROOT}/_sanitized/")
    )
    redacted_pattern = rf"{re.escape(REDACTED_HOST_TEXT)}/[0-9a-f]{{24}}"
    assert re.fullmatch(
        redacted_pattern,
        raw.set_index("run_id").loc["windows-run", "error"],
    )
    assert re.fullmatch(
        redacted_pattern,
        raw.set_index("run_id").loc["opaque-run", "failure_reason"],
    )
    assert raw.set_index("run_id").loc["opaque-run", "note"] == (
        "mirror https://example.org/a; ratio 1/2; support/oppose; "
        "relative ./file ../other ~/cache relative/path"
    )
    details = json.loads(raw.set_index("run_id").loc["opaque-run", "details"])
    for key in ("unc", "windows", "posix"):
        assert re.fullmatch(redacted_pattern, details[key])
    assert details["relative"] == "relative/path.txt"
    assert details["url"] == "https://example.org/data/file.txt"
    historical_details = json.loads(raw.set_index("run_id").loc["posix-run", "details"])
    assert re.fullmatch(redacted_pattern, historical_details["unc"])
    assert historical_details["relative"] == "relative/file.txt"

    write_compact_raw(tmp_path, raw)
    write_compact_runs(tmp_path, runs)
    raw_text = (tmp_path / "raw_metrics.csv").read_text(encoding="utf-8")
    runs_text = (tmp_path / "runs.csv").read_text(encoding="utf-8")
    compressed_text = gzip.decompress(
        (tmp_path / "raw_metrics.csv.gz").read_bytes()
    ).decode("utf-8")
    for published in (raw_text, runs_text, compressed_text):
        assert "/home/" not in published
        assert "/mnt/" not in published
        assert "C:\\Users\\" not in published
        assert "C:/Users/" not in published
        assert "D:\\" not in published
        assert "E:\\" not in published
        assert "\\\\server\\" not in published
        assert published.count(PORTABLE_RUNS_ROOT) >= 3


def test_compact_raw_snapshot_is_lossless_deterministic_and_preferred(
    tmp_path: Path,
) -> None:
    authoritative = pd.DataFrame(
        [
            {
                "run_id": "new",
                "metric": 3.7940759646432964e-16,
                "status": "complete",
            },
            {"run_id": "failed", "metric": np.nan, "status": "failed"},
        ]
    )
    write_compact_raw(tmp_path, authoritative)
    compressed = tmp_path / "raw_metrics.csv.gz"
    plain = tmp_path / "raw_metrics.csv"
    first_bytes = compressed.read_bytes()
    for _ in range(3):
        round_tripped, _ = merge_compact_snapshot(
            tmp_path, pd.DataFrame(), pd.DataFrame()
        )
        write_compact_raw(tmp_path, round_tripped)

    assert compressed.read_bytes() == first_bytes
    assert not (tmp_path / "raw_metrics.csv.gz.tmp").exists()
    pd.testing.assert_frame_equal(pd.read_csv(compressed), pd.read_csv(plain))
    with gzip.open(compressed, "rb") as handle:
        assert b"\r\n" not in handle.read()

    second_root = tmp_path / "second-root"
    second_root.mkdir()
    write_compact_raw(second_root, authoritative)
    assert (second_root / "raw_metrics.csv.gz").read_bytes() == first_bytes

    pd.DataFrame([{"run_id": "stale", "metric": -1.0}]).to_csv(plain, index=False)
    merged, _ = merge_compact_snapshot(tmp_path, pd.DataFrame(), pd.DataFrame())
    assert set(merged["run_id"]) == {"new", "failed"}


def test_empty_authoritative_raw_snapshot_fails_closed(tmp_path: Path) -> None:
    pd.DataFrame([{"run_id": "legacy"}]).to_csv(
        tmp_path / "raw_metrics.csv", index=False
    )
    (tmp_path / "raw_metrics.csv.gz").touch()

    with pytest.raises(ValueError, match="authoritative raw_metrics.csv.gz is empty"):
        merge_compact_snapshot(tmp_path, pd.DataFrame(), pd.DataFrame())


def test_oversized_raw_snapshot_is_not_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.build_report as build_report

    monkeypatch.setattr(build_report, "MAX_PUBLISHED_RAW_BYTES", 1)
    with pytest.raises(ValueError, match="95 MiB publication safety limit"):
        build_report.write_compact_raw(
            tmp_path, pd.DataFrame([{"run_id": "too-large"}])
        )
    assert not (tmp_path / "raw_metrics.csv.gz").exists()
    assert (tmp_path / "raw_metrics.csv.gz.tmp").exists()


def test_compact_snapshot_merge_preserves_history_replaces_runs_and_is_idempotent(
    tmp_path: Path,
) -> None:
    historical_raw = pd.DataFrame(
        [
            {
                "run_id": "history-only",
                "recorded_at": "2026-01-01T00:00:00Z",
                "condition": "failed-control",
                "status": "failed",
            },
            {
                "run_id": "rediscovered",
                "recorded_at": "2026-01-02T00:00:00Z",
                "condition": "stale-partial",
                "status": "failed",
            },
        ]
    )
    historical_runs = pd.DataFrame(
        [
            {"run_id": "history-only", "status": "failed"},
            {"run_id": "rediscovered", "status": "running"},
        ]
    )
    historical_raw.to_csv(tmp_path / "raw_metrics.csv", index=False)
    historical_runs.to_csv(tmp_path / "runs.csv", index=False)
    # Two legitimate rows deliberately share a timestamp.  A timestamp key
    # would erase one of them, whereas run-level replacement preserves both.
    discovered_raw = pd.DataFrame(
        [
            {
                "run_id": "rediscovered",
                "recorded_at": "2026-01-02T01:00:00Z",
                "condition": "aligned",
                "status": "complete",
            },
            {
                "run_id": "rediscovered",
                "recorded_at": "2026-01-02T01:00:00Z",
                "condition": "shuffled",
                "status": "complete",
            },
        ]
    )
    discovered_runs = pd.DataFrame([{"run_id": "rediscovered", "status": "complete"}])

    raw, runs = merge_compact_snapshot(tmp_path, discovered_raw, discovered_runs)
    assert set(raw["run_id"]) == {"history-only", "rediscovered"}
    assert set(raw.loc[raw["run_id"].eq("rediscovered"), "condition"]) == {
        "aligned",
        "shuffled",
    }
    assert "stale-partial" not in set(raw["condition"])
    assert runs.set_index("run_id").loc["history-only", "status"] == "failed"
    assert runs.set_index("run_id").loc["rediscovered", "status"] == "complete"

    raw.to_csv(tmp_path / "raw_metrics.csv", index=False)
    runs.to_csv(tmp_path / "runs.csv", index=False)
    repeated_raw, repeated_runs = merge_compact_snapshot(
        tmp_path, discovered_raw, discovered_runs
    )
    pd.testing.assert_frame_equal(repeated_raw, raw)
    pd.testing.assert_frame_equal(repeated_runs, runs)


def test_compact_snapshot_normalizes_numeric_ids_and_legacy_fallbacks(
    tmp_path: Path,
) -> None:
    historical_raw = pd.DataFrame(
        [
            {
                "run_id": 1,
                "experiment": "numeric",
                "seed": 0,
                "run_started_at": "2026-01-01T00:00:00Z",
                "condition": "stale-numeric",
            },
            {
                "run_id": None,
                "experiment": "legacy",
                "seed": 2,
                "run_started_at": "2026-01-03T00:00:00Z",
                "condition": "stale-legacy",
            },
        ]
    )
    historical_runs = pd.DataFrame(
        [
            {
                "run_id": 1,
                "experiment": "numeric",
                "seed": 0,
                "started_at": "2026-01-01T00:00:00Z",
                "status": "failed",
            },
            {
                "run_id": None,
                "experiment": "legacy",
                "seed": 2,
                "started_at": "2026-01-03T00:00:00Z",
                "status": "failed",
            },
        ]
    )
    historical_raw.to_csv(tmp_path / "raw_metrics.csv", index=False)
    historical_runs.to_csv(tmp_path / "runs.csv", index=False)
    discovered_raw = pd.DataFrame(
        [
            {
                "run_id": 1,
                "experiment": "numeric",
                "seed": 0,
                "run_started_at": "2026-01-01T00:00:00Z",
                "condition": "fresh-numeric",
            },
            {
                "run_id": None,
                "experiment": "legacy",
                "seed": 2,
                "run_started_at": "2026-01-03T00:00:00Z",
                "condition": "fresh-legacy",
            },
        ]
    )
    discovered_runs = pd.DataFrame(
        [
            {
                "run_id": 1,
                "experiment": "numeric",
                "seed": 0,
                "started_at": "2026-01-01T00:00:00Z",
                "status": "complete",
            },
            {
                "run_id": None,
                "experiment": "legacy",
                "seed": 2,
                "started_at": "2026-01-03T00:00:00Z",
                "status": "complete",
            },
        ]
    )

    raw, runs = merge_compact_snapshot(tmp_path, discovered_raw, discovered_runs)
    assert set(raw["condition"]) == {"fresh-numeric", "fresh-legacy"}
    assert set(runs["status"]) == {"complete"}


def test_compact_snapshot_rejects_duplicate_or_unidentified_discovered_runs(
    tmp_path: Path,
) -> None:
    duplicate_runs = pd.DataFrame(
        [
            {"run_id": "duplicate", "status": "complete"},
            {"run_id": "duplicate", "status": "failed"},
        ]
    )
    with pytest.raises(ValueError, match="share one run identity"):
        merge_compact_snapshot(tmp_path, pd.DataFrame(), duplicate_runs)

    unidentified = pd.DataFrame([{"status": "complete"}])
    with pytest.raises(ValueError, match="stable.*provenance"):
        merge_compact_snapshot(tmp_path, pd.DataFrame(), unidentified)


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
