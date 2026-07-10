from pathlib import Path

import pandas as pd

from figures.core_results_plot import _latest_attempt, plot_core_results
from figures.phase_models_plot import _complete_profile, plot_phase_models


def test_plot_functions_accept_empty_and_minimal_bound_data(tmp_path: Path) -> None:
    empty_core = plot_core_results(pd.DataFrame())
    empty_phase = plot_phase_models(pd.DataFrame())
    assert len(empty_core.axes) == 4
    assert len(empty_phase.axes) == 4
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
