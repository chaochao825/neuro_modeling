import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from figures.core_results_plot import _latest_attempt, plot_core_results
from figures.exp10_bridge_pilot_plot import (
    _comparison_summary,
    _latest_complete_rows,
    make_figure,
)
from figures.exp11_ibl_behavior_plot import (
    load_real_rows,
    make_figure as make_ibl_behavior_figure,
    real_data_comparison_summary,
)
from figures.hidden_context_plot import plot_hidden_context
from figures.phase_models_plot import _complete_profile, plot_phase_models
from figures.plot_style import save_figure


def test_saved_figure_bytes_are_deterministic(tmp_path: Path) -> None:
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [1.0, 0.0])
    first = tmp_path / "first"
    second = tmp_path / "second"
    save_figure(figure, "bound", first)
    save_figure(figure, "bound", second)
    plt.close(figure)

    pdf_bytes = (first / "bound.pdf").read_bytes()
    assert pdf_bytes == (second / "bound.pdf").read_bytes()
    assert b"CreationDate" not in pdf_bytes
    assert b"ModDate" not in pdf_bytes
    assert (first / "bound.png").read_bytes() == (second / "bound.png").read_bytes()


def test_plot_functions_accept_empty_and_minimal_bound_data(tmp_path: Path) -> None:
    empty_core = plot_core_results(pd.DataFrame())
    empty_phase = plot_phase_models(pd.DataFrame())
    empty_hidden = plot_hidden_context(pd.DataFrame())
    assert len(empty_core.axes) == 4
    assert len(empty_phase.axes) == 4
    assert len(empty_hidden.axes) == 4
    minimal = pd.DataFrame(
        [
            {
                "experiment": "exp01_feedback_dimension_sweep",
                "profile": "formal",
                "seed": 0,
                "status": "complete",
                "grid": "core",
                "feedback_mode": "aligned",
                "feedback_dim": 4,
                "effective_rank": 4.0,
                "latent_r2": 0.9,
                "rollout_normalized_rmse": 0.2,
                "plasticity_cost": 1.0,
            }
        ]
    )
    figure = plot_core_results(minimal)
    output = tmp_path / "test.pdf"
    figure.savefig(output)
    assert output.stat().st_size > 0


def test_plot_filters_share_start_time_based_latest_attempt_selection() -> None:
    attempts = pd.DataFrame(
        [
            {
                "experiment": "exp01_feedback_dimension_sweep",
                "profile": "formal",
                "seed": 0,
                "run_id": "old",
                "run_started_at": "20260710T110000.000000Z",
                "recorded_at": "2026-07-10T13:00:00Z",
                "run_status": "complete",
                "status": "complete",
            },
            {
                "experiment": "exp01_feedback_dimension_sweep",
                "profile": "formal",
                "seed": 0,
                "run_id": "retry",
                "run_started_at": "2026-07-10T12:00:00Z",
                "recorded_at": "2026-07-10T12:30:00Z",
                "run_status": "complete",
                "status": "complete",
            },
        ]
    )

    assert _latest_attempt(attempts)["run_id"].tolist() == ["retry"]
    assert _complete_profile(attempts)["run_id"].tolist() == ["retry"]


def test_phase_plot_falls_back_to_complete_ibl_when_sequence_only_failed() -> None:
    raw = pd.DataFrame(
        [
            {
                "experiment": "exp05_sequence_real_data",
                "profile": "formal",
                "status": "failed",
                "session_id": "restricted-sequence",
            },
            {
                "experiment": "exp06_ibl_context_switch",
                "profile": "formal",
                "status": "complete",
                "session_id": "ibl-session",
                "fold": 0,
                "model_family": "shared",
                "heldout_nll_per_scalar": 1.25,
            },
        ]
    )

    figure = plot_phase_models(raw)
    assert figure.axes[3].get_title() == "IBL LDS; folds nested in session"
    assert len(figure.axes[3].patches) == 1


def test_hidden_context_delay_bar_matches_preregistered_high_hazard_scope() -> None:
    rows = []
    for hazard in (0.01, 0.05, 0.10, 0.20):
        common = {
            "experiment": "exp09_hidden_context_gate",
            "profile": "formal",
            "seed": 0,
            "status": "complete",
            "cue_reliability": 0.70,
            "context_hazard": hazard,
        }
        rows += [
            {
                **common,
                "gate_model": "no_gate",
                "intervention": "none",
                "behavior_balanced_accuracy": 0.5,
                "context_nll": 0.7,
            },
            {
                **common,
                "gate_model": "md_recurrent_belief",
                "intervention": "none",
                "behavior_balanced_accuracy": 1.0,
                "context_nll": 0.4,
            },
            {
                **common,
                "gate_model": "md_recurrent_belief",
                "intervention": "clamp",
                "behavior_balanced_accuracy": 0.5,
                "context_nll": 0.4,
            },
            {
                **common,
                "gate_model": "md_recurrent_belief",
                "intervention": "delay",
                "behavior_balanced_accuracy": 0.8 if hazard >= 0.10 else 0.99,
                "context_nll": 0.4,
            },
            {
                **common,
                "gate_model": "md_recurrent_belief",
                "intervention": "shuffle",
                "behavior_balanced_accuracy": 0.7,
                "context_nll": 0.4,
            },
        ]

    figure = plot_hidden_context(pd.DataFrame(rows))
    axis = next(
        item
        for item in figure.axes
        if item.get_ylabel() == "Intact minus intervention accuracy"
    )
    assert np.allclose([patch.get_height() for patch in axis.patches], [0.5, 0.2, 0.3])
    assert [label.get_text() for label in axis.get_xticklabels()] == [
        "clamp",
        "delay\n(h=.10/.20)",
        "shuffle",
    ]
    plt.close(figure)


def test_exp10_pilot_plot_uses_seed_paired_differences(tmp_path: Path) -> None:
    rows = []
    conditions = [
        ("no_gate", "none", 0.60),
        ("learned_hmm", "none", 0.65),
        ("md_recurrent_belief", "none", 0.61),
        ("oracle_bayes", "none", 0.68),
        ("md_recurrent_belief", "clamp", 0.60),
        ("md_recurrent_belief", "delay", 0.605),
        ("md_recurrent_belief", "shuffle", 0.602),
    ]
    for seed in range(30):
        for gate, intervention, accuracy in conditions:
            rows.append(
                {
                    "seed": seed,
                    "gate_model": gate,
                    "intervention": intervention,
                    "behavior_balanced_accuracy": accuracy + seed * 1e-4,
                    "status": "complete",
                    "profile": "smoke",
                    "cue_reliability": 0.70,
                    "context_hazard": 0.10,
                    "network_n_units": 32,
                    "bridge_protocol_id": "a" * 64,
                    "recurrent_learning": False,
                    "base_conditions_share_readout": False,
                    "base_comparison_scope": (
                        "separately_train_optimized_pipeline_comparison"
                    ),
                    "efficiency_claim_eligible": False,
                    "three_factor_plasticity_claim_eligible": False,
                }
            )
    frame = pd.DataFrame(rows)
    summary = _comparison_summary(frame)
    hmm = summary.loc[summary["comparison"] == "hmm_vs_no_gate"].iloc[0]
    assert hmm["mean_balanced_accuracy_difference"] == pytest.approx(0.05)
    assert hmm["statistics_unit"] == "seed"
    assert hmm["conclusion"] == "functional_pipeline_support_pilot"
    assert not hmm["biological_mechanism_claim_eligible"]
    make_figure(frame, tmp_path)
    assert (tmp_path / "exp10_bridge_pilot.pdf").stat().st_size > 0
    assert (tmp_path / "exp10_bridge_pilot.png").stat().st_size > 0
    frame.to_csv(
        tmp_path / "exp10_bridge_pilot_raw.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    reloaded = _latest_complete_rows(tmp_path)
    assert len(reloaded) == 210
    assert reloaded["bridge_protocol_id"].nunique() == 1

    newer_formal = (
        tmp_path
        / "runs"
        / "exp10_hidden_context_ei_bridge"
        / "seed_0000"
        / "20990101T000000.000000Z"
    )
    newer_formal.mkdir(parents=True)
    (newer_formal / "status.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    formal_rows = []
    for row in rows[:7]:
        formal_rows.append({**row, "profile": "formal", "network_n_units": 256})
    (newer_formal / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in formal_rows), encoding="utf-8"
    )
    still_pilot = _latest_complete_rows(tmp_path)
    assert len(still_pilot) == 210
    assert set(still_pilot["network_n_units"]) == {32}


def test_exp11_summary_uses_animal_primary_paired_inference(tmp_path: Path) -> None:
    rows = []
    condition_values = {
        "no_memory": (0.69, 0.60),
        "exponential_history": (0.50, 0.58),
        "learned_categorical_hmm": (0.30, 0.65),
        "oracle_ceiling": (0.0, 0.55),
    }
    for animal in range(10):
        for session in range(2):
            for condition, (context_nll, behavior_loss) in condition_values.items():
                rows.append(
                    {
                        "eid": f"eid-{animal}-{session}",
                        "animal_id": f"mouse-{animal}",
                        "condition": condition,
                        "status": "complete",
                        "context_nll": context_nll,
                        "behavior_log_loss": behavior_loss,
                        "official_bwm_mask_present": True,
                        "eligible_for_context_inference_support": condition
                        == "learned_categorical_hmm",
                        "eligible_for_behavior_pipeline_evaluation": condition
                        == "learned_categorical_hmm",
                    }
                )
    frame = pd.DataFrame(rows)
    summary, paired = real_data_comparison_summary(frame, n_bootstrap=1000)
    context = summary.loc[summary["claim"] == "hmm_context_nll_gain"].iloc[0]
    behavior = summary.loc[summary["claim"] == "hmm_behavior_log_loss_gain"].iloc[0]
    assert context["n_animals"] == 10
    assert context["n_sessions"] == 20
    assert context["conclusion"] == "support"
    assert behavior["conclusion"] == "oppose"
    make_ibl_behavior_figure(frame, summary, paired, tmp_path)
    assert (tmp_path / "exp11_ibl_behavior_real.pdf").stat().st_size > 0


def test_exp11_summary_does_not_select_only_valid_or_successful_hmm_sessions() -> None:
    rows = []
    for session in range(30):
        for condition in (
            "no_memory",
            "exponential_history",
            "learned_categorical_hmm",
            "oracle_ceiling",
        ):
            rows.append(
                {
                    "eid": f"eid-{session:02d}",
                    "animal_id": f"mouse-{session // 3:02d}",
                    "condition": condition,
                    "status": "complete",
                    "context_nll": (
                        0.2 if condition == "learned_categorical_hmm" else 0.7
                    ),
                    "behavior_log_loss": 0.5,
                    "official_bwm_mask_present": True,
                    "eligible_for_context_inference_support": condition
                    == "learned_categorical_hmm"
                    and session >= 10,
                }
            )
    frame = pd.DataFrame(rows)
    summary, paired = real_data_comparison_summary(frame, n_bootstrap=200)
    context = summary.loc[summary["claim"] == "hmm_context_nll_gain"].iloc[0]
    assert len(paired["hmm_context_nll_gain"]) == 30
    assert context["n_invalid_gate_sessions"] == 10
    assert context["conclusion"] == "inconclusive_invalid_gate_fit"

    frame.loc[
        frame["condition"].eq("learned_categorical_hmm"),
        "eligible_for_context_inference_support",
    ] = True
    frame.loc[
        frame["condition"].eq("learned_categorical_hmm") & frame["eid"].eq("eid-00"),
        "status",
    ] = "failed"
    summary, _ = real_data_comparison_summary(frame, n_bootstrap=200)
    context = summary.loc[summary["claim"] == "hmm_context_nll_gain"].iloc[0]
    assert context["n_failed_or_missing_sessions"] == 1
    assert context["conclusion"] == "inconclusive_incomplete_cohort"


def test_exp11_loader_uses_latest_attempt_even_when_it_failed(tmp_path: Path) -> None:
    seed_root = tmp_path / "runs" / "exp11_ibl_behavior_belief" / "seed_0000"
    older = seed_root / "20260101T000000.000000Z"
    latest = seed_root / "20260102T000000.000000Z"
    for run_dir, status in ((older, "complete"), (latest, "failed")):
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps({"profile": "formal"}), encoding="utf-8"
        )
        (run_dir / "status.json").write_text(
            json.dumps({"status": status}), encoding="utf-8"
        )
    (older / "metrics.jsonl").write_text(
        json.dumps({"condition": "no_memory", "status": "complete"}) + "\n",
        encoding="utf-8",
    )
    (latest / "metrics.jsonl").write_text(
        json.dumps(
            {
                "condition": "setup",
                "status": "failed",
                "error": "manifest unavailable",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_real_rows(tmp_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["condition"] == "setup"
    assert loaded.iloc[0]["source_run_status"] == "failed"


def test_exp11_partial_failed_attempt_cannot_support_from_written_subset(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "runs"
        / "exp11_ibl_behavior_belief"
        / "seed_0000"
        / "20260103T000000.000000Z"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"profile": "formal"}), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    conditions = (
        "no_memory",
        "exponential_history",
        "learned_categorical_hmm",
        "oracle_ceiling",
    )
    planned = []
    for session in range(30):
        for condition in conditions:
            planned.append(
                {
                    "condition_index": len(planned),
                    "eid": f"eid-{session:02d}",
                    "animal_id": f"mouse-{session // 3:02d}",
                    "condition": condition,
                    "cohort_manifest_sha256": "c" * 64,
                }
            )
    (run_dir / "planned_conditions.json").write_text(
        json.dumps(planned), encoding="utf-8"
    )
    rows = []
    for session in range(20):
        for condition in conditions:
            rows.append(
                {
                    "eid": f"eid-{session:02d}",
                    "animal_id": f"mouse-{session // 3:02d}",
                    "condition": condition,
                    "status": "complete",
                    "profile": "formal",
                    "context_nll": (
                        0.2 if condition == "learned_categorical_hmm" else 0.7
                    ),
                    "behavior_log_loss": (
                        0.4 if condition == "learned_categorical_hmm" else 0.6
                    ),
                    "official_bwm_mask_present": True,
                    "cohort_manifest_sha256": "c" * 64,
                    "algorithmic_seed_is_statistical_unit": False,
                    "behavior_only_benchmark": True,
                    "neural_activity_analyzed": False,
                    "eligible_for_context_inference_support": condition
                    == "learned_categorical_hmm",
                }
            )
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    loaded = load_real_rows(tmp_path)
    assert len(loaded) == 120
    assert (loaded["status"] == "missing_from_metrics").sum() == 40
    summary, _ = real_data_comparison_summary(loaded, n_bootstrap=200)
    context = summary.loc[summary["claim"] == "hmm_context_nll_gain"].iloc[0]
    assert context["n_planned_sessions"] == 30
    assert context["n_paired_complete_sessions"] == 20
    assert context["conclusion"] == "inconclusive_failed_or_unfinalized_attempt"
