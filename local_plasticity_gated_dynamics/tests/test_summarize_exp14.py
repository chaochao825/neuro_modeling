from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.summarize_exp14 as exp14_summary
from figures.exp14_ibl_multisession_neural_plot import plot_exp14
from scripts.build_report import _exp14_claim_statistics, append_exp14_neural_claim
from scripts.summarize_exp14 import (
    DEFAULT_PREFIX,
    _expected_attempt_config,
    _portable_formal_config_sha256,
    _recompute_comparison,
    _assert_nested_close,
    build_snapshot,
    collect_formal_run,
    load_validated_exp14_snapshot,
)
from src.data.ibl_neural_panel import default_allen_macro_region_mapping


def _config() -> dict[str, object]:
    return {
        "profile": "formal",
        "seeds": [0],
        "data_mode": "frozen_compact_cache",
        "views": ["stimulus_pre", "movement_pre"],
        "panels": ["primary_past_safe", "full_trial_sensitivity"],
        "planned_sessions": 20,
        "planned_animals": 5,
        "n_bootstrap": 100,
        "minimum_region_sessions": 5,
        "latent_dims": [1, 2],
        "ridges": [0.1],
        "learned_hmm": {
            "n_restarts": 5,
            "restart_selection_policy": (
                "eligible_converged_identifiable_then_likelihood"
            ),
            "require_converged": True,
            "require_identifiable": True,
        },
        "expected_source_manifest_sha256": "a" * 64,
        "expected_acquisition_bundle_sha256": "b" * 64,
        "expected_compact_manifest_sha256": "a5acb134ae4b34f47db150948a7f7ab58e8eb85e204fb981e0ca744eba328a09",
        "expected_compact_bundle_sha256": "f" * 64,
        "expected_bwm_repository_commit": "d" * 40,
        "macro_region_mapping_path": "configs/exp14_allen_macro_region_mapping_v1.json",
        "expected_macro_region_mapping_sha256": "3bac702ed6b3ee5c21acbbfd929b077baa63226369ca8e1bef0b6faeb487fc23",
        "expected_macro_region_mapping_schema": "exp14_allen_macro_region_mapping_v1",
        "expected_macro_region_source_ontology_sha256": "63654b8d35c7c1b5665636b645da774776ee8263658192f5dca1e815095e9147",
        "expected_macro_region_source_provenance_sha256": "a01b7fa535e6de437ac46e8cf9de68a87d6a9b5587d055a3935476d956109fdc",
        "macro_region_mapping_formal_compact_manifest_sha256": "a5acb134ae4b34f47db150948a7f7ab58e8eb85e204fb981e0ca744eba328a09",
    }


def test_nested_comparison_accepts_only_the_canonical_serialized_nan() -> None:
    _assert_nested_close("nan", float("nan"))
    _assert_nested_close(float("nan"), float("nan"))
    with pytest.raises(ValueError, match="differs from recomputed"):
        _assert_nested_close("NaN", float("nan"))
    with pytest.raises(ValueError, match="differs from recomputed"):
        _assert_nested_close("nan", 0.0)


def _interval(estimate: float, low: float, high: float) -> dict[str, object]:
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_p_two_sided": 0.01,
        "null_value": 0.0,
        "holm_adjusted_p": 0.03,
        "n_sessions": 20,
        "n_animals": 5,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _register(monkeypatch: pytest.MonkeyPatch, config: dict[str, object]) -> None:
    monkeypatch.setattr(
        exp14_summary, "_registered_formal_config", lambda: dict(config)
    )


def _fake_run(
    results_root: Path, config: dict[str, object], *, name: str = "20260712T000000Z"
) -> Path:
    attempt = (
        results_root / "runs" / "exp14_ibl_multisession_neural" / "seed_0000" / name
    )
    attempt.mkdir(parents=True)
    _write_json(attempt / "config.json", _expected_attempt_config(config))
    _write_json(
        attempt / "environment.json", {"git": {"commit": "e" * 40, "dirty": False}}
    )
    _write_json(attempt / "status.json", {"status": "complete"})
    _write_json(attempt / "manifest.json", {"run_id": "run-14", "status": "complete"})
    planned = []
    metrics = []
    families = ("common", "shared", "full")
    all_session_ids = tuple(f"session-{index:02d}" for index in range(20))
    common_regions = ("cortex", "thalamus")
    coverage = tuple(
        {
            "region": region,
            "n_sessions_present": 20,
            "session_fraction_present": 1.0,
            "n_sessions_missing": 0,
            "missing_session_ids": (),
        }
        for region in common_regions
    )
    anchor_metrics = {
        "region_anchor_policy": "fixed_region_order_union",
        "region_imputation_strategy": "pooled_training_fold_region_mean",
        "minimum_region_sessions": 5,
        "region_session_coverage": coverage,
        "n_complete_sessions_input": 20,
        "n_sessions_retained": 20,
        "complete_session_ids": all_session_ids,
        "retained_session_ids": all_session_ids,
        "all_complete_sessions_retained": True,
        **dict(default_allen_macro_region_mapping().receipt()),
    }
    for session_index in range(20):
        session = f"session-{session_index:02d}"
        animal = f"animal-{session_index // 4}"
        for view in config["views"]:
            for panel in config["panels"]:
                scope = (
                    "registered_primary"
                    if (view, panel) == ("stimulus_pre", "primary_past_safe")
                    else "sensitivity_only"
                )
                for family_index, family in enumerate(families):
                    nll = (1.2, 1.105, 1.1)[family_index]
                    planned.append(
                        {
                            "condition_index": len(planned),
                            "session_id": session,
                            "view": view,
                            "panel": panel,
                            "model_family": family,
                        }
                    )
                    metrics.append(
                        {
                            "run_id": "run-14",
                            "stage": "outer_test",
                            "status": "complete",
                            "session_id": session,
                            "animal_id": animal,
                            "view": view,
                            "panel": panel,
                            "model_family": family,
                            "aggregation_level": "session",
                            "statistics_unit": "session_nested_within_animal",
                            "n_transitions": 10,
                            "n_count_observations": 100,
                            "log_likelihood": -100.0 * nll,
                            "null_log_likelihood": -130.0,
                            "saturated_log_likelihood": -50.0,
                            "nll_per_count": nll,
                            "pseudo_r2": 0.1 + 0.02 * family_index,
                            "closure_mse": 0.3 - 0.02 * family_index,
                            "parameter_count": (80, 120, 300)[family_index],
                            "preprocessing_fit_train_only": True,
                            "hidden_context_inference": True,
                            "test_context_observed": False,
                            "condition_schedule_used_for_split_only": True,
                            "nuisance_as_log_rate_controls": True,
                            "counts_residualized_before_poisson": False,
                            "likelihood_kind": "one_step_conditional_poisson",
                            "full_latent_lds": False,
                            "claim_scope": scope,
                            "causal_timing_eligible": True,
                            "comparison_preprocessing_sha256": "9" * 64,
                            "selected_latent_dim": 1,
                            "selected_ridge": 0.1,
                            "common_regions": common_regions,
                            **anchor_metrics,
                            "anchor_regions_present": common_regions,
                            "anchor_regions_missing": (),
                            "n_anchor_regions_present": len(common_regions),
                            "n_anchor_regions_missing": 0,
                            "hmm_fit_converged": True,
                            "hmm_state_identifiable": True,
                            "hmm_restart_selection_policy": config["learned_hmm"][
                                "restart_selection_policy"
                            ],
                            "hmm_selected_restart": session_index % 5,
                            "hmm_eligible_restart_count": 4,
                            "hmm_eligible_restart_fallback": False,
                            "belief_checkpoint_sha256": "1" * 64,
                            "belief_trajectory_sha256": "2" * 64,
                            "source_manifest_sha256": config[
                                "expected_source_manifest_sha256"
                            ],
                            "acquisition_bundle_sha256": config[
                                "expected_acquisition_bundle_sha256"
                            ],
                            "compact_manifest_sha256": config[
                                "expected_compact_manifest_sha256"
                            ],
                            "compact_bundle_sha256": config[
                                "expected_compact_bundle_sha256"
                            ],
                            "bwm_repository_commit": config[
                                "expected_bwm_repository_commit"
                            ],
                        }
                    )
    for view in config["views"]:
        for panel in config["panels"]:
            for latent_dim in config["latent_dims"]:
                for ridge in config["ridges"]:
                    metrics.append(
                        {
                            "run_id": "run-14",
                            "stage": "nested_selection",
                            "view": view,
                            "panel": panel,
                            "latent_dim": latent_dim,
                            "ridge": ridge,
                            "status": "complete",
                            "animal_mean_validation_nll": 1.0
                            if latent_dim == 1
                            else 1.1,
                        }
                    )
    for view in config["views"]:
        for panel in config["panels"]:
            scope = (
                "registered_primary"
                if (view, panel) == ("stimulus_pre", "primary_past_safe")
                else "sensitivity_only"
            )
            metrics.append(
                {
                    "run_id": "run-14",
                    "stage": "animal_session_comparison",
                    "status": "complete",
                    "view": view,
                    "panel": panel,
                    "claim_scope": scope,
                    "core_claim_eligible": scope == "registered_primary",
                    "aggregation_level": "animal_with_session_nested",
                    "nested_selection_objective": (
                        "mean_animal_validation_nll_across_common_shared_full"
                    ),
                    "selected_latent_dim": 1,
                    "selected_ridge": 0.1,
                    "common_regions": common_regions,
                    **anchor_metrics,
                    "comparison": _recompute_comparison(
                        pd.DataFrame(
                            [row for row in metrics if row.get("stage") == "outer_test"]
                        ),
                        view=str(view),
                        panel=str(panel),
                        config=config,
                    ),
                    "likelihood_kind": "one_step_conditional_poisson",
                    "full_latent_lds": False,
                    "causal_timing_eligible": view == "stimulus_pre",
                }
            )
    _write_json(attempt / "planned_conditions.json", planned)
    (attempt / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in metrics) + "\n",
        encoding="utf-8",
    )
    (attempt / "run.log").write_text("complete\n", encoding="utf-8")
    return attempt


def _records(attempt: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (attempt / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _replace_records(attempt: Path, records: list[dict[str, object]]) -> None:
    (attempt / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records), encoding="utf-8"
    )


def test_builds_hash_bound_snapshot_and_never_promotes_sensitivity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    _register(monkeypatch, config)
    _fake_run(tmp_path, config)
    build_snapshot(tmp_path, config)
    conditions, comparisons, raw, manifest = load_validated_exp14_snapshot(tmp_path)
    assert len(conditions) == 12
    assert len(raw.loc[raw["stage"] == "outer_test"]) == 240
    assert len(manifest) == 1
    primary = comparisons.loc[comparisons["claim_scope"] == "registered_primary"].iloc[
        0
    ]
    assert primary["core_conclusion"] == "support"
    assert set(
        comparisons.loc[
            comparisons["claim_scope"] == "sensitivity_only", "core_conclusion"
        ]
    ) == {"inconclusive"}


def test_raw_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    _register(monkeypatch, config)
    _fake_run(tmp_path, config)
    build_snapshot(tmp_path, config)
    raw_path = tmp_path / f"{DEFAULT_PREFIX}_raw.csv.gz"
    raw = pd.read_csv(raw_path)
    raw.loc[0, "nll_per_count"] = 999
    raw.to_csv(raw_path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="does not bind"):
        load_validated_exp14_snapshot(tmp_path)


def test_mixed_config_attempt_is_not_selected(tmp_path: Path) -> None:
    config = _config()
    mixed = dict(config, planned_animals=6)
    attempt = _fake_run(tmp_path, config)
    _write_json(attempt / "config.json", _expected_attempt_config(mixed))
    with pytest.raises(FileNotFoundError, match="exactly matches"):
        collect_formal_run(tmp_path, config)


def test_noncomplete_run_is_ineligible(tmp_path: Path) -> None:
    config = _config()
    attempt = _fake_run(tmp_path, config)
    _write_json(attempt / "status.json", {"status": "complete_with_failures"})
    with pytest.raises(ValueError, match="status complete"):
        collect_formal_run(tmp_path, config)


def test_failed_planned_condition_is_retained_but_ineligible(tmp_path: Path) -> None:
    config = _config()
    attempt = _fake_run(tmp_path, config)
    records = [
        json.loads(line)
        for line in (attempt / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    records[0]["status"] = "failed"
    (attempt / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="failed, missing, or duplicate"):
        collect_formal_run(tmp_path, config)


def test_compact_bundle_lineage_is_required_and_exact(tmp_path: Path) -> None:
    missing = _config()
    del missing["expected_compact_bundle_sha256"]
    with pytest.raises(ValueError, match="expected_compact_bundle_sha256"):
        collect_formal_run(tmp_path, missing)

    config = _config()
    attempt = _fake_run(tmp_path, config)
    records = [
        json.loads(line)
        for line in (attempt / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    records[0]["compact_bundle_sha256"] = "0" * 64
    (attempt / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="provenance"):
        collect_formal_run(tmp_path, config)


@pytest.mark.parametrize(
    "mutation",
    ["comparison", "animal_count", "session_map", "parameter_count"],
)
def test_outer_pairing_and_stored_comparison_tampering_are_rejected(
    tmp_path: Path, mutation: str
) -> None:
    config = _config()
    attempt = _fake_run(tmp_path, config)
    records = _records(attempt)
    outer = [row for row in records if row.get("stage") == "outer_test"]
    if mutation == "comparison":
        comparison = next(
            row for row in records if row.get("stage") == "animal_session_comparison"
        )
        comparison["comparison"]["shared_vs_common"]["estimate"] = 99.0
    elif mutation == "animal_count":
        for row in outer:
            row["animal_id"] = "one-animal"
    elif mutation == "session_map":
        common = [row for row in outer if row["model_family"] == "common"]
        first = common[0]
        second = next(row for row in common if row["animal_id"] != first["animal_id"])
        first["session_id"], second["session_id"] = (
            second["session_id"],
            first["session_id"],
        )
    else:
        outer[0]["parameter_count"] = int(outer[0]["parameter_count"]) + 1
    _replace_records(attempt, records)
    with pytest.raises(ValueError):
        collect_formal_run(tmp_path, config)


def test_outer_mechanism_and_nested_selection_gates_are_fail_closed(
    tmp_path: Path,
) -> None:
    config = _config()
    attempt = _fake_run(tmp_path, config)
    records = _records(attempt)
    outer = next(row for row in records if row.get("stage") == "outer_test")
    outer["hidden_context_inference"] = False
    _replace_records(attempt, records)
    with pytest.raises(ValueError, match="inference contract"):
        collect_formal_run(tmp_path, config)

    other = tmp_path / "other"
    attempt = _fake_run(other, config)
    records = _records(attempt)
    records = [
        row
        for index, row in enumerate(records)
        if not (row.get("stage") == "nested_selection" and index >= 240)
    ]
    _replace_records(attempt, records)
    _, _, comparisons, _ = collect_formal_run(other, config)
    assert set(comparisons["core_conclusion"]) == {"inconclusive"}
    assert not comparisons["nested_selection_valid"].any()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hmm_restart_selection_policy", "likelihood_only"),
        ("hmm_selected_restart", 5),
        ("hmm_eligible_restart_count", 0),
        ("hmm_eligible_restart_count", 6),
        ("hmm_eligible_restart_fallback", True),
        ("hmm_fit_converged", False),
        ("hmm_state_identifiable", False),
    ],
)
def test_outer_hmm_restart_receipts_are_fail_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    config = _config()
    attempt = _fake_run(tmp_path, config)
    records = _records(attempt)
    outer = next(row for row in records if row.get("stage") == "outer_test")
    outer[field] = value
    _replace_records(attempt, records)
    with pytest.raises(ValueError, match="eligible-restart contract"):
        collect_formal_run(tmp_path, config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hmm_selected_restart", 1),
        ("hmm_eligible_restart_count", 3),
        ("belief_checkpoint_sha256", "3" * 64),
        ("belief_trajectory_sha256", "4" * 64),
    ],
)
def test_paired_model_families_must_share_the_hmm_receipt(
    tmp_path: Path, field: str, value: object
) -> None:
    config = _config()
    attempt = _fake_run(tmp_path, config)
    records = _records(attempt)
    outer = next(row for row in records if row.get("stage") == "outer_test")
    outer[field] = value
    _replace_records(attempt, records)
    with pytest.raises(ValueError, match="families disagree"):
        collect_formal_run(tmp_path, config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("restart_selection_policy", "likelihood_only"),
        ("n_restarts", 0),
        ("require_converged", False),
        ("require_identifiable", False),
    ],
)
def test_formal_hmm_restart_config_is_fail_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    config = _config()
    config["learned_hmm"] = {**config["learned_hmm"], field: value}
    with pytest.raises(ValueError, match="eligible-first HMM restart selection"):
        collect_formal_run(tmp_path, config)


@pytest.mark.parametrize(
    "mutation",
    ["fingerprint", "primary_causal", "common_regions", "selected", "comparison"],
)
def test_reporting_scope_and_nested_receipts_are_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    config = _config()
    attempt = _fake_run(tmp_path, config)
    records = _records(attempt)
    outer = [row for row in records if row.get("stage") == "outer_test"]
    comparisons = [
        row for row in records if row.get("stage") == "animal_session_comparison"
    ]
    if mutation == "fingerprint":
        outer[0]["comparison_preprocessing_sha256"] = "8" * 64
    elif mutation == "primary_causal":
        next(row for row in outer if row["claim_scope"] == "registered_primary")[
            "causal_timing_eligible"
        ] = False
    elif mutation == "common_regions":
        outer[0]["common_regions"] = ["MD"]
    elif mutation == "selected":
        for row in outer:
            if row["view"] == "stimulus_pre" and row["panel"] == "primary_past_safe":
                row["selected_latent_dim"] = 2
    else:
        comparisons[0]["nested_selection_objective"] = "unregistered_objective"
    _replace_records(attempt, records)
    with pytest.raises(ValueError):
        collect_formal_run(tmp_path, config)


@pytest.mark.parametrize(
    "mutation",
    [
        "policy",
        "retained_ids",
        "coverage",
        "per_session_missing",
        "mapping_hash",
        "mapping_schema",
        "ontology_hash",
        "ancestor_ids",
        "selection_policy",
        "behavior_selected",
        "unknown_policy",
        "acronym_hash",
    ],
)
def test_missing_region_receipts_are_fail_closed(tmp_path: Path, mutation: str) -> None:
    config = _config()
    attempt = _fake_run(tmp_path, config)
    records = _records(attempt)
    outer = [row for row in records if row.get("stage") == "outer_test"]
    comparison = next(
        row for row in records if row.get("stage") == "animal_session_comparison"
    )
    if mutation == "policy":
        outer[0]["region_anchor_policy"] = "intersection_drop_sessions"
    elif mutation == "retained_ids":
        comparison["retained_session_ids"] = comparison["retained_session_ids"][:-1]
    elif mutation == "coverage":
        comparison["region_session_coverage"][0]["n_sessions_missing"] = 1
    elif mutation == "per_session_missing":
        outer[0]["anchor_regions_missing"] = ["thalamus"]
    elif mutation == "mapping_hash":
        outer[0]["macro_region_mapping_sha256"] = "0" * 64
    elif mutation == "mapping_schema":
        comparison["macro_region_mapping_schema"] = "wrong_schema"
    elif mutation == "ontology_hash":
        outer[0]["macro_region_source_ontology_sha256"] = "0" * 64
    elif mutation == "ancestor_ids":
        comparison["macro_region_ancestor_ids"]["cortex"] = 0
    elif mutation == "selection_policy":
        outer[0]["macro_region_selection_policy"] = "selected_by_behavior"
    elif mutation == "behavior_selected":
        comparison["macro_region_behavior_or_model_selected"] = True
    elif mutation == "unknown_policy":
        outer[0]["macro_region_unknown_policy"] = "cortex"
    else:
        comparison["macro_region_formal_acronyms_sha256"] = "0" * 64
    _replace_records(attempt, records)
    with pytest.raises(ValueError):
        collect_formal_run(tmp_path, config)


def test_default_snapshot_rejects_an_internally_consistent_alternate_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = _config()
    alternate = dict(registered, expected_compact_bundle_sha256="0" * 64)
    _register(monkeypatch, registered)
    _fake_run(tmp_path, alternate)
    with pytest.raises(ValueError, match="frozen registered config"):
        build_snapshot(tmp_path, alternate)
    build_snapshot(tmp_path, alternate, prefix="exp14_exploratory")
    _, comparisons, _, _ = load_validated_exp14_snapshot(
        tmp_path, prefix="exp14_exploratory"
    )
    assert set(comparisons["core_conclusion"]) == {"inconclusive"}
    assert not comparisons["core_claim_eligible"].any()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("macro_region_mapping_path", "configs/unregistered_mapping.json"),
        ("expected_macro_region_mapping_schema", "wrong_schema"),
        ("expected_macro_region_source_ontology_sha256", "0" * 64),
    ],
)
def test_formal_macro_mapping_config_is_fail_closed(
    tmp_path: Path, field: str, value: str
) -> None:
    config = dict(_config(), **{field: value})
    with pytest.raises(ValueError, match="macro-region mapping"):
        collect_formal_run(tmp_path, config)


def test_registered_config_binding_is_portable_but_science_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows = dict(_config(), config_path=r"E:\checkout\configs\formal\exp14.json")
    linux = dict(_config(), config_path="/home/lab/configs/formal/exp14.json")
    assert _portable_formal_config_sha256(windows) == _portable_formal_config_sha256(
        linux
    )
    _register(monkeypatch, windows)
    _fake_run(tmp_path, linux)
    build_snapshot(tmp_path, linux)
    _, comparisons, _, _ = load_validated_exp14_snapshot(tmp_path)
    assert set(comparisons["formal_config_sha256"]) == {
        _portable_formal_config_sha256(windows)
    }

    changed_science = dict(linux, latent_dims=[1])
    with pytest.raises(ValueError, match="frozen registered config"):
        build_snapshot(tmp_path, changed_science)


def test_global_report_claim_is_absent_until_valid_snapshot_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = pd.DataFrame()
    assert append_exp14_neural_claim(empty, tmp_path).empty
    config = _config()
    _register(monkeypatch, config)
    _fake_run(tmp_path, config)
    build_snapshot(tmp_path, config)
    appended = append_exp14_neural_claim(empty, tmp_path)
    assert appended.iloc[0]["claim_id"] == "U1_ibl_shared_neural_dynamics"
    assert appended.iloc[0]["conclusion"] == "support"
    primary = (
        load_validated_exp14_snapshot(tmp_path)[1]
        .loc[lambda frame: frame["claim_scope"] == "registered_primary"]
        .iloc[0]
    )
    assert appended.iloc[0]["p_value"] == max(
        primary["shared_vs_common_holm_adjusted_p"],
        primary["full_vs_common_holm_adjusted_p"],
        primary["retention_margin_holm_adjusted_p"],
    )


def test_oppose_claim_uses_the_actual_triggering_estimand() -> None:
    row = pd.Series(
        {
            "core_conclusion": "oppose",
            "shared_vs_common_estimate": 0.04,
            "shared_vs_common_ci_low": 0.01,
            "shared_vs_common_ci_high": 0.07,
            "shared_vs_common_holm_adjusted_p": 0.02,
            "full_vs_common_holm_adjusted_p": 0.01,
            "retention_defined": True,
            "retention_margin_estimate": -0.03,
            "retention_margin_ci_low": -0.05,
            "retention_margin_ci_high": -0.01,
            "retention_margin_holm_adjusted_p": 0.04,
        }
    )
    result = _exp14_claim_statistics(row)
    assert result["metric"] == "shared_90pct_full_gain_retention_margin"
    assert result["estimate"] == -0.03
    assert result["ci_high"] == -0.01
    assert result["p_value"] == 0.04


def test_plot_reads_only_validated_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    _register(monkeypatch, config)
    _fake_run(tmp_path, config)
    build_snapshot(tmp_path, config)
    figure = plot_exp14(tmp_path)
    assert len(figure.axes) == 4
    assert figure.axes[1].get_ylabel() == (
        "Common - shared NLL/count (positive favors shared)"
    )
    labels = [tick.get_text() for tick in figure.axes[1].get_xticklabels()]
    assert "registered core" in labels[0]
    assert all("sensitivity" in label for label in labels[1:])


def test_plot_does_not_color_panel_support_as_core_when_nested_gate_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    _register(monkeypatch, config)
    attempt = _fake_run(tmp_path, config)
    records = [
        row for row in _records(attempt) if row.get("stage") != "nested_selection"
    ]
    _replace_records(attempt, records)
    build_snapshot(tmp_path, config)
    figure = plot_exp14(tmp_path)
    primary_rgb = figure.axes[1].collections[-1].get_facecolors()[0, :3]
    assert tuple(primary_rgb.round(3)) == (0.467, 0.467, 0.467)
