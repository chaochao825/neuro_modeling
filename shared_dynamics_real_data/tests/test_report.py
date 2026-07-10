import json
from pathlib import Path

import pandas as pd
import pytest

from shared_dynamics_real_data.build_report import (
    _claim_rows,
    _comparisons,
    _latest_panel,
    _model_summary,
    _seed_level,
    collect_runs,
    write_report,
)
from shared_dynamics_real_data.figures.plot_visual_context import plot
from shared_dynamics_real_data.run_visual_context import _source_fingerprint


def _rows() -> pd.DataFrame:
    records = []
    model_nll = {
        "common": 1.2,
        "shared": 1.1,
        "separate": 1.0,
        "random": 1.3,
        "orthogonal": 1.4,
        "shuffled": 1.35,
    }
    for seed in (0, 1):
        for fold in (0, 1):
            for latent_dim in (4, 8):
                for model, nll in model_nll.items():
                    records.append(
                        {
                            "run_id": "20260711",
                            "analysis_fingerprint": "analysis-a",
                            "data_fingerprint": "data-a",
                            "configured_seed_universe": "[0,1]",
                            "profile": "formal",
                            "dataset_pair": "pair",
                            "computational_seed": seed,
                            "fold": fold,
                            "latent_dim": latent_dim,
                            "model": model,
                            "status": "complete",
                            "nll_per_scalar": nll - 0.01 * (latent_dim == 8),
                            "standardized_nll_per_scalar": nll,
                            "one_step_r2": 0.2,
                            "rollout_nrmse": 1.0,
                            "parameter_count": 100 if model == "shared" else 200,
                            "effective_rank": 3.5,
                            "top_k_singular_energy": 0.9,
                            "subspace_angle_degrees": 0.0,
                        }
                    )
    return pd.DataFrame(records)


def test_committed_formal_panel_matches_current_reporting_source() -> None:
    panel_path = Path(__file__).resolve().parents[1] / "results" / "panel.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    assert panel["report_source_fingerprint"] == _source_fingerprint()


def test_folds_are_aggregated_inside_seed_and_claims_stay_inconclusive() -> None:
    latest = _latest_panel(_rows(), "formal")
    seed_level = _seed_level(latest)
    assert len(seed_level) == 2 * 2 * 6
    comparisons = _comparisons(seed_level)
    assert (comparisons["random_minus_shared_nll"] > 0).all()
    claims = _claim_rows(comparisons, seed_level, latest)
    assert set(claims["conclusion"]) == {"inconclusive"}
    assert set(claims["n_biological_units"]) == {1}
    directions = claims.set_index("claim_id")["descriptive_direction"]
    assert directions["R0_switching_improves_common"] == "support"
    # The toy panel has positive one-step R2 but rollout NRMSE exactly at the
    # unit baseline, so the conjunctive absolute-signal criterion is opposed.
    assert directions["R4_shared_has_absolute_predictive_signal"] == "oppose"
    margin = claims.set_index("claim_id").loc[
        "R4_shared_has_absolute_predictive_signal", "descriptive_estimate"
    ]
    assert margin == pytest.approx(0.0)


def test_mixed_analysis_fingerprints_are_rejected() -> None:
    rows = _rows()
    mixed = rows.iloc[[0]].copy()
    mixed["analysis_fingerprint"] = "analysis-b"
    with pytest.raises(ValueError, match="multiple analysis fingerprints"):
        _latest_panel(pd.concat([rows, mixed], ignore_index=True), "formal")


def test_failed_fold_invalidates_whole_seed_dimension_panel() -> None:
    rows = _rows()
    mask = (
        (rows["computational_seed"] == 0)
        & (rows["latent_dim"] == 4)
        & (rows["fold"] == 1)
        & (rows["model"] == "random")
    )
    rows.loc[mask, "status"] = "failed"
    seed_level = _seed_level(_latest_panel(rows, "formal"))
    invalid = seed_level.loc[
        (seed_level["computational_seed"] == 0)
        & (seed_level["latent_dim"] == 4)
    ]
    assert invalid.empty
    valid = seed_level.loc[
        (seed_level["computational_seed"] == 1)
        & (seed_level["latent_dim"] == 4)
    ]
    assert set(valid["model"]) == {
        "common",
        "shared",
        "separate",
        "random",
        "orthogonal",
        "shuffled",
    }


def test_missing_configured_seed_is_synthesized_as_failure() -> None:
    rows = _rows().loc[lambda frame: frame["computational_seed"] == 0].copy()
    rows["configured_seed_universe"] = "[0,1]"
    latest = _latest_panel(rows, "formal")
    missing = latest.loc[latest["computational_seed"] == 1]
    assert not missing.empty
    assert set(missing["status"]) == {"missing"}
    assert _seed_level(latest).loc[
        lambda frame: frame["computational_seed"] == 1
    ].empty


def test_all_failed_and_missing_control_remain_reportable(tmp_path: Path) -> None:
    failed = _rows()
    failed["status"] = "failed"
    latest_failed = _latest_panel(failed, "formal")
    seed_failed = _seed_level(latest_failed)
    model_failed = _model_summary(seed_failed)
    comparisons_failed = _comparisons(seed_failed)
    claims_failed = _claim_rows(comparisons_failed, seed_failed, latest_failed)
    assert set(claims_failed["descriptive_direction"]) == {"unavailable"}
    report = tmp_path / "report.md"
    write_report(
        report,
        profile="formal",
        raw=failed,
        latest=latest_failed,
        model_summary=model_failed,
        comparisons=comparisons_failed,
        claims=claims_failed,
    )
    assert report.is_file()
    assert "failed:" in report.read_text(encoding="utf-8")

    missing = _rows().loc[lambda frame: frame["model"] != "random"]
    latest_missing = _latest_panel(missing, "formal")
    seed_missing = _seed_level(latest_missing)
    claims_missing = _claim_rows(
        _comparisons(seed_missing), seed_missing, latest_missing
    )
    direction = claims_missing.set_index("claim_id").loc[
        "R2_aligned_beats_basis_controls", "descriptive_direction"
    ]
    assert direction == "unavailable"


def test_rank4_claims_are_unavailable_when_dimension_four_was_not_tested() -> None:
    rows = _rows()
    rows["latent_dim"] = rows["latent_dim"].replace({4: 2})
    latest = _latest_panel(rows, "formal")
    seed_level = _seed_level(latest)
    claims = _claim_rows(_comparisons(seed_level), seed_level, latest).set_index(
        "claim_id"
    )
    for claim_id in (
        "R0_switching_improves_common",
        "R1_shared_retains_separate_gain",
        "R2_aligned_beats_basis_controls",
        "R3_d4_nll_vs_highest_tested_dimension",
        "R4_shared_has_absolute_predictive_signal",
    ):
        assert claims.loc[claim_id, "descriptive_direction"] == "unavailable"
        assert "no complete paired" in claims.loc[claim_id, "reason"]


def test_r0_opposes_when_switching_nll_is_higher() -> None:
    rows = _rows()
    rows.loc[rows["model"] == "shared", "nll_per_scalar"] = 1.25
    latest = _latest_panel(rows, "formal")
    seed_level = _seed_level(latest)
    claims = _claim_rows(_comparisons(seed_level), seed_level, latest).set_index(
        "claim_id"
    )
    assert claims.loc[
        "R0_switching_improves_common", "descriptive_direction"
    ] == "oppose"
    assert claims.loc[
        "R0_switching_improves_common", "descriptive_estimate"
    ] > 0


def test_r1_margin_and_invalid_denominator_are_reported_truthfully(
    tmp_path: Path,
) -> None:
    rows = _rows()
    latest = _latest_panel(rows, "formal")
    seed_level = _seed_level(latest)
    claims = _claim_rows(_comparisons(seed_level), seed_level, latest).set_index(
        "claim_id"
    )
    assert claims.loc[
        "R1_shared_retains_separate_gain", "descriptive_direction"
    ] == "oppose"

    # Make separate worse than common: all pairs remain complete, but the
    # positive separate-over-common gain denominator no longer exists.
    rows.loc[rows["model"] == "separate", "nll_per_scalar"] = 1.3
    latest = _latest_panel(rows, "formal")
    seed_level = _seed_level(latest)
    comparisons = _comparisons(seed_level)
    claim_table = _claim_rows(comparisons, seed_level, latest)
    claims = claim_table.set_index("claim_id")
    r1 = claims.loc["R1_shared_retains_separate_gain"]
    assert r1["descriptive_direction"] == "unavailable"
    assert r1["n_computational_seeds"] == 2
    assert "separate does not improve common" in r1["reason"]
    report = tmp_path / "report.md"
    write_report(
        report,
        profile="formal",
        raw=rows,
        latest=latest,
        model_summary=_model_summary(seed_level),
        comparisons=comparisons,
        claims=claim_table,
    )
    assert "separate does not improve common" in report.read_text(encoding="utf-8")

@pytest.mark.parametrize(
    ("one_step_r2", "rollout_nrmse", "expected"),
    [(0.2, 0.8, "support"), (-0.1, 0.8, "oppose"), (0.2, 1.1, "oppose")],
)
def test_r4_requires_both_absolute_prediction_components(
    one_step_r2: float, rollout_nrmse: float, expected: str
) -> None:
    rows = _rows()
    rows["one_step_r2"] = one_step_r2
    rows["rollout_nrmse"] = rollout_nrmse
    latest = _latest_panel(rows, "formal")
    seed_level = _seed_level(latest)
    claims = _claim_rows(_comparisons(seed_level), seed_level, latest).set_index(
        "claim_id"
    )
    assert claims.loc[
        "R4_shared_has_absolute_predictive_signal", "descriptive_direction"
    ] == expected


def test_data_bound_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    latest = _latest_panel(_rows(), "formal")
    seed_level = _seed_level(latest)
    _model_summary(seed_level).to_csv(tmp_path / "model_summary.csv", index=False)
    _comparisons(seed_level).to_csv(tmp_path / "comparisons.csv", index=False)
    output = tmp_path / "figure"
    plot(tmp_path, output)
    assert output.with_suffix(".png").stat().st_size > 1000
    assert output.with_suffix(".pdf").stat().st_size > 1000


def test_collector_backfills_planned_cell_missing_from_metrics(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "20260711"
    run.mkdir(parents=True)
    config = {
        "profile": "formal",
        "analysis_fingerprint": "analysis-a",
        "configured_seeds": [0],
        "seeds": [0],
        "n_units": 8,
        "purge": 1,
    }
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    planned = [
        {
            "computational_seed": 0,
            "fold": 0,
            "latent_dim": 2,
            "model": model,
            "family": model,
            "basis_control": "aligned",
        }
        for model in ("common", "shared")
    ]
    (run / "planned_conditions.json").write_text(
        json.dumps(planned), encoding="utf-8"
    )
    (run / "data_manifest.json").write_text(
        json.dumps(
            {
                "dataset_pair": "pair",
                "data_fingerprint": "data-a",
            }
        ),
        encoding="utf-8",
    )
    (run / "status.json").write_text(
        json.dumps({"status": "complete_with_failures"}), encoding="utf-8"
    )
    recorded = _rows().iloc[[0]].copy()
    recorded["model"] = "common"
    recorded["latent_dim"] = 2
    recorded["fold"] = 0
    recorded.to_csv(run / "metrics.csv", index=False)

    collected = collect_runs(tmp_path)
    assert len(collected) == 2
    missing = collected.loc[collected["model"] == "shared"].iloc[0]
    assert missing["status"] == "missing"
    assert missing["error_type"] == "MissingCell"
