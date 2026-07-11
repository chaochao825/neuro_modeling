from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.claims import evaluate_core_claims
from src.analysis.p2_protocol import FORMAL_P2_PROTOCOL_ID


Q_VALUES = (0.55, 0.70, 0.85, 1.0)
HAZARDS = (0.01, 0.05, 0.10, 0.20)
GATES = (
    "oracle_bayes",
    "supervised_upper_bound",
    "learned_hmm",
    "md_recurrent_belief",
    "no_gate",
)
INTERVENTIONS = ("clamp", "delay", "shuffle")
P2_CLAIMS = (
    "P2a_hmm_context_nll",
    "P2b_md_context_nll",
    "P2c_md_context_brier",
    "P2d_md_calibration",
    "P2e_md_switch_latency",
    "P2f_md_false_switch",
    "P2g_md_behavior",
    "P2h_md_retains_oracle_gain",
    "P2i_md_energy",
    "P2j_clamp_causal",
    "P2k_delay_causal",
    "P2l_shuffle_causal",
)


def _gate_metrics(gate: str) -> dict[str, float]:
    return {
        "oracle_bayes": {
            "context_nll": 0.10,
            "context_brier": 0.04,
            "context_ece": 0.01,
            "switch_latency_trials": 0.0,
            "false_switch_rate": 0.001,
            "behavior_balanced_accuracy": 0.95,
            "energy_proxy_per_trial": 1.00,
        },
        "supervised_upper_bound": {
            "context_nll": 0.12,
            "context_brier": 0.05,
            "context_ece": 0.02,
            "switch_latency_trials": 0.2,
            "false_switch_rate": 0.002,
            "behavior_balanced_accuracy": 0.94,
            "energy_proxy_per_trial": 1.03,
        },
        "learned_hmm": {
            "context_nll": 0.20,
            "context_brier": 0.07,
            "context_ece": 0.03,
            "switch_latency_trials": 0.4,
            "false_switch_rate": 0.004,
            "behavior_balanced_accuracy": 0.92,
            "energy_proxy_per_trial": 1.04,
        },
        "md_recurrent_belief": {
            "context_nll": 0.22,
            "context_brier": 0.08,
            "context_ece": 0.03,
            "switch_latency_trials": 0.5,
            "false_switch_rate": 0.005,
            "behavior_balanced_accuracy": 0.935,
            "energy_proxy_per_trial": 1.05,
        },
        "no_gate": {
            "context_nll": 0.69,
            "context_brier": 0.25,
            "context_ece": 0.25,
            "switch_latency_trials": 6.0,
            "false_switch_rate": 0.0,
            "behavior_balanced_accuracy": 0.75,
            "energy_proxy_per_trial": 1.00,
        },
    }[gate]


def _condition_row(
    seed: int,
    reliability: float,
    hazard: float,
    gate: str,
    intervention: str,
) -> dict[str, object]:
    supervised = gate == "supervised_upper_bound"
    oracle = gate == "oracle_bayes"
    metrics = dict(_gate_metrics(gate))
    if intervention != "none":
        metrics.update(
            context_nll=0.40,
            context_brier=0.15,
            context_ece=0.10,
            switch_latency_trials=2.0,
            false_switch_rate=0.02,
            energy_proxy_per_trial=1.04,
        )
        metrics["behavior_balanced_accuracy"] = {
            "clamp": 0.85,
            "delay": 0.90,
            "shuffle": 0.86,
        }[intervention]
    pair = f"seed={seed}:q={reliability}:h={hazard}"
    md_checkpoint = f"md-checkpoint:{pair}"
    md_readout = f"md-readout:{pair}"
    return {
        "experiment": "exp09_hidden_context_gate",
        "profile": "formal",
        "seed": seed,
        "status": "complete",
        "condition": f"{gate}__{intervention}",
        "cue_reliability": reliability,
        "context_hazard": hazard,
        "gate_model": gate,
        "intervention": intervention,
        "eligible_switch_count": 50,
        "p2_protocol_id": FORMAL_P2_PROTOCOL_ID,
        "train_trial_count": 6000,
        "dev_trial_count": 2000,
        "test_trial_count": 4000,
        "latency_limit_trials": 5,
        "latency_sustain_trials": 2,
        "posterior_threshold": 0.8,
        "minimum_state_duration": 5,
        "switch_tolerance_trials": 1,
        "minimum_eligible_switches": 20,
        "delay_trials": 1,
        **metrics,
        "hidden_context_task": True,
        "cue_encodes_observation_not_state": True,
        "gate_test_accessed_true_context": False,
        "gate_fit_accessed_true_context": supervised,
        "third_factor_accessed_true_context": False,
        "oracle_warm_start_used": False,
        "md_fit_used_context_bias": False,
        "gate_fit_accessed_task_target": False,
        "gate_test_accessed_task_target": False,
        "gate_test_future_observations_accessed": False,
        "gate_fit_used_batch_smoothing": gate == "learned_hmm",
        "state_label_alignment_accessed_true_context": False,
        "test_switch_boundaries_accessed_by_model": False,
        "preprocessing_fit_train_only": True,
        "hyperparameters_preregistered": True,
        "dev_used_for_selection": False,
        "train_dev_test_episode_disjoint": True,
        "belief_online_causal": True,
        "predictions_frozen_before_truth_scoring": True,
        "eligible_for_p2_support": not supervised,
        "gate_received_true_q_h": oracle,
        "gate_fit_supervision": (
            "known_generative_params"
            if oracle
            else ("train_context_labels" if supervised else "none")
        ),
        "true_context_access_scope": (
            "train_gate_fit_and_evaluation" if supervised else "evaluation_only"
        ),
        "intervention_postfit": intervention != "none",
        "intervention_reuses_intact_checkpoint": intervention != "none",
        "intervention_reuses_intact_readout": intervention != "none",
        "intervention_permutation_accessed_true_context": False,
        "random_tape_id": f"random:{pair}",
        "hidden_state_tape_id": f"state:{pair}",
        "observation_tape_id": f"observation:{pair}",
        "task_tape_id": f"task:{pair}",
        "noise_tape_id": f"noise:{pair}",
        "network_initialization_id": f"network:{pair}",
        "split_id": f"split:{pair}",
        "readout_fit_data_id": f"readout-data:{pair}",
        "readout_protocol_id": "fixed-readout-protocol",
        "checkpoint_id": (
            md_checkpoint if gate == "md_recurrent_belief" else f"{gate}:{pair}"
        ),
        "readout_id": (
            md_readout if gate == "md_recurrent_belief" else f"{gate}-readout:{pair}"
        ),
        "belief_trajectory_id": (
            f"md-belief:{pair}:none"
            if gate == "md_recurrent_belief" and intervention == "none"
            else f"{gate}-belief:{pair}:{intervention}"
        ),
        "intact_belief_trajectory_id": (
            f"md-belief:{pair}:none"
            if gate == "md_recurrent_belief"
            else f"{gate}-belief:{pair}:none"
        ),
    }


def _p2_formal(seeds=range(30)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for reliability in Q_VALUES:
            for hazard in HAZARDS:
                for gate in GATES:
                    rows.append(_condition_row(seed, reliability, hazard, gate, "none"))
                for intervention in INTERVENTIONS:
                    rows.append(
                        _condition_row(
                            seed,
                            reliability,
                            hazard,
                            "md_recurrent_belief",
                            intervention,
                        )
                    )
    return pd.DataFrame(rows)


def _claims(raw: pd.DataFrame) -> dict[str, object]:
    return {claim.claim_id: claim for claim in evaluate_core_claims(raw)}


def test_p2_full_grid_supports_with_fixed_full_holm_family() -> None:
    claims = _claims(_p2_formal())

    assert len(claims) == 36
    assert len(set(claims) - {"P0_overall", "P2_overall"}) == 34
    for claim_id in P2_CLAIMS:
        claim = claims[claim_id]
        assert claim.conclusion == "support"
        assert claim.n_complete == 30
        assert claim.n_failed == 0
        assert "all 34 registered claims" in claim.note
    assert claims["P2_overall"].conclusion == "support"
    assert claims["P2_overall"].n_complete == 30
    assert (
        claims["P2_overall"].multiplicity_method
        == "derived_after_holm(no_additional_test)"
    )


def test_p2_missing_cell_and_extra_seed_cannot_replace_planned_seed() -> None:
    raw = _p2_formal(range(31))
    missing = (
        raw["seed"].eq(29)
        & raw["cue_reliability"].eq(0.55)
        & raw["context_hazard"].eq(0.01)
        & raw["gate_model"].eq("no_gate")
        & raw["intervention"].eq("none")
    )
    claims = _claims(raw.loc[~missing].copy())

    for claim_id in P2_CLAIMS:
        assert claims[claim_id].conclusion == "inconclusive"
        assert claims[claim_id].n_complete == 29
        assert claims[claim_id].n_failed == 0
    assert claims["P2_overall"].n_complete == 29
    assert claims["P2_overall"].conclusion == "inconclusive"


@pytest.mark.parametrize("corruption", ["invalid", "duplicate"])
def test_p2_invalid_or_duplicate_cell_fails_closed(corruption: str) -> None:
    raw = _p2_formal()
    target = (
        raw["seed"].eq(7)
        & raw["cue_reliability"].eq(0.70)
        & raw["context_hazard"].eq(0.05)
        & raw["gate_model"].eq("learned_hmm")
        & raw["intervention"].eq("none")
    )
    if corruption == "invalid":
        raw.loc[target, "status"] = "invalid"
    else:
        raw = pd.concat([raw, raw.loc[target].copy()], ignore_index=True)

    claims = _claims(raw)
    claim = claims["P2b_md_context_nll"]
    assert claim.conclusion == "inconclusive"
    assert claim.n_complete == 29
    assert claim.n_failed == 1
    assert corruption in claim.note
    assert claims["P2_overall"].conclusion == "inconclusive"
    assert claims["P2_overall"].n_failed == 1


def test_p2_candidate_true_context_leakage_is_not_hidden_by_supervised_bound() -> None:
    raw = _p2_formal()
    leaked = (
        raw["seed"].eq(11)
        & raw["gate_model"].eq("md_recurrent_belief")
        & raw["intervention"].eq("none")
    )
    raw.loc[leaked, "gate_fit_accessed_true_context"] = True

    claims = _claims(raw)
    claim = claims["P2g_md_behavior"]
    assert claim.conclusion == "inconclusive"
    assert claim.n_complete == 29
    assert claim.n_failed == 1
    assert "gate_fit_accessed_true_context" in claim.note
    assert claims["P2_overall"].conclusion == "inconclusive"


def test_p2_pairing_mismatch_fails_the_whole_affected_seed() -> None:
    raw = _p2_formal()
    mismatched = (
        raw["seed"].eq(13)
        & raw["cue_reliability"].eq(0.85)
        & raw["context_hazard"].eq(0.10)
        & raw["gate_model"].eq("md_recurrent_belief")
        & raw["intervention"].eq("none")
    )
    raw.loc[mismatched, "noise_tape_id"] = "different-noise-tape"

    claims = _claims(raw)
    claim = claims["P2e_md_switch_latency"]
    assert claim.conclusion == "inconclusive"
    assert claim.n_complete == 29
    assert claim.n_failed == 1
    assert "noise_tape_id" in claim.note


def test_p2_rejects_intervention_that_repeats_the_intact_trajectory() -> None:
    raw = _p2_formal()
    target = (
        raw["seed"].eq(13)
        & raw["cue_reliability"].eq(0.85)
        & raw["context_hazard"].eq(0.10)
        & raw["gate_model"].eq("md_recurrent_belief")
    )
    intact = raw.loc[
        target & raw["intervention"].eq("none"), "belief_trajectory_id"
    ].iloc[0]
    raw.loc[target & raw["intervention"].eq("delay"), "belief_trajectory_id"] = intact

    claim = _claims(raw)["P2k_delay_causal"]
    assert claim.conclusion == "inconclusive"
    assert claim.n_complete == 29
    assert claim.n_failed == 1
    assert "belief trajectories" in claim.note


def test_p2_rejects_nonregistered_protocol_id() -> None:
    raw = _p2_formal()
    raw.loc[raw["seed"].eq(13), "p2_protocol_id"] = "tampered-protocol"

    claim = _claims(raw)["P2b_md_context_nll"]
    assert claim.conclusion == "inconclusive"
    assert claim.n_complete == 29
    assert claim.n_failed == 1
    assert "p2_protocol_id" in claim.note


def test_p2_overall_opposes_when_holm_adjusted_behavior_claim_opposes() -> None:
    raw = _p2_formal()
    md_intact = raw["gate_model"].eq("md_recurrent_belief") & raw["intervention"].eq(
        "none"
    )
    raw.loc[md_intact, "behavior_balanced_accuracy"] = 0.70

    claims = _claims(raw)
    assert claims["P2g_md_behavior"].conclusion == "oppose"
    assert claims["P2h_md_retains_oracle_gain"].conclusion == "oppose"
    assert claims["P2_overall"].conclusion == "oppose"


def test_missing_exp09_keeps_twelve_p2_placeholders_in_fixed_family() -> None:
    claims = _claims(pd.DataFrame())

    assert len(claims) == 36
    assert all(claims[claim_id].conclusion == "inconclusive" for claim_id in P2_CLAIMS)
    assert claims["P2_overall"].conclusion == "inconclusive"
    assert claims["P2_overall"].n_complete == 0
