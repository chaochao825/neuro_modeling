from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from matplotlib.axes import Axes

import scripts.summarize_exp19 as exp19_summary_module
import scripts.summarize_exp20 as exp20_summary_module
from experiments.exp19_belief_ei_effective_dynamics import ALL_CONDITION_SPECS
from experiments.exp20_ibl_md_belief_dynamics import MODEL_CONDITIONS
from scripts.summarize_exp19 import (
    _expected_run_config as exp19_expected_config,
)
from scripts.summarize_exp19 import _environment_sha256 as exp19_environment_sha256
from scripts.summarize_exp19 import _markdown_table as exp19_markdown_table
from scripts.summarize_exp19 import _plot as plot_exp19
from scripts.summarize_exp19 import _validate_cross_seed_receipts
from scripts.summarize_exp19 import summarize_formal_runs as summarize_exp19
from scripts.summarize_exp20 import _environment_sha256 as exp20_environment_sha256
from scripts.summarize_exp20 import _validate_outer_contract
from scripts.summarize_exp20 import _markdown_table as exp20_markdown_table
from scripts.summarize_exp20 import _plot as plot_exp20
from scripts.summarize_exp20 import summarize_formal_run as summarize_exp20
from scripts.integrate_exp19_exp20 import SUMMARY_COLUMNS, integrate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("renderer", [exp19_markdown_table, exp20_markdown_table])
def test_scoped_reports_do_not_require_optional_tabulate(renderer) -> None:
    rendered = renderer(pd.DataFrame([{"claim": "a|b", "note": "line1\nline2"}]))
    assert "a\\|b" in rendered
    assert "line1 line2" in rendered


def _config(experiment: int) -> dict[str, object]:
    path = (
        PROJECT_ROOT
        / "configs"
        / "formal"
        / (
            "exp19_belief_ei_effective_dynamics.json"
            if experiment == 19
            else "exp20_ibl_md_belief_dynamics.json"
        )
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    result["config_path"] = str(path)
    return result


def test_exp19_summary_uses_filtered_timing_and_seed_holm_family() -> None:
    config = _config(19)
    expected = exp19_expected_config(config, 0)
    assert expected["training_algorithm"] == "md_filtered_belief_frozen_high_rank_dale_ei"
    assert expected["used_bptt"] is False

    rows = []
    for seed in range(30):
        for spec in ALL_CONDITION_SPECS:
            balanced = 0.8 if spec.condition == "md_combined_intact" else 0.7
            if spec.condition == "direct_evidence_mix":
                balanced = 0.85
            rows.append(
                {
                    "seed": seed,
                    "condition": spec.condition,
                    "status": "complete",
                    "behavior_balanced_accuracy": balanced,
                    "physical_rank_fraction": 1.0,
                    "physical_dale_valid": True,
                    "moment_anchor_identifiable": True,
                    "declared_scalar_control_dimension": 1,
                    "combined_effective_control_dimension": 1,
                    "empirical_combined_control_trajectory_rank": 1,
                    "operator_control_dimension": 1,
                    "operator_delta_frobenius_norm": 0.1,
                    "heldout_normalized_closure_mse": 0.5,
                    "heldout_basis_residual_fraction": 0.2,
                    "normal_stability_eligible": True,
                    "normal_local_decay_ratio": 0.8,
                    "normal_local_max_real_part": -0.01,
                }
            )
    summary = summarize_exp19(pd.DataFrame(rows), config, n_bootstrap=200)
    clamp = summary.loc[
        summary["comparison"].eq("md_combined_intact_vs_md_combined_clamp")
    ].iloc[0]
    direct = summary.loc[
        summary["comparison"].eq("md_combined_intact_vs_direct_evidence_mix")
    ].iloc[0]
    population_only = summary.loc[
        summary["comparison"].eq("md_combined_intact_vs_md_population_only")
    ].iloc[0]
    assert clamp["conclusion"] == "support"
    assert direct["conclusion"] == "oppose"
    assert direct["proposition"] == "separate_train_only_baseline"
    assert "not input-charge-matched" in population_only["claim_scope"]
    assert (
        "filtered_gate_moment_anchor_identifiability"
        in set(summary["proposition"])
    )
    threshold = summary.loc[
        summary["multiplicity_family"].eq("none_registered_threshold")
    ]
    assert set(threshold["conclusion"]) == {"support"}


def test_exp19_publication_rejects_mixed_clean_commits() -> None:
    _validate_cross_seed_receipts(
        pd.DataFrame(
            [
                {
                    "seed": 0,
                    "git_commit": "a" * 40,
                    "environment_sha256": "c" * 64,
                },
                {
                    "seed": 1,
                    "git_commit": "a" * 40,
                    "environment_sha256": "c" * 64,
                },
            ]
        )
    )
    with pytest.raises(ValueError, match="mixed Git commits"):
        _validate_cross_seed_receipts(
            pd.DataFrame(
                [
                    {
                        "seed": 0,
                        "git_commit": "a" * 40,
                        "environment_sha256": "c" * 64,
                    },
                    {
                        "seed": 1,
                        "git_commit": "b" * 40,
                        "environment_sha256": "c" * 64,
                    },
                ]
            )
        )
    with pytest.raises(ValueError, match="mixed software environments"):
        _validate_cross_seed_receipts(
            pd.DataFrame(
                [
                    {
                        "seed": 0,
                        "git_commit": "a" * 40,
                        "environment_sha256": "c" * 64,
                    },
                    {
                        "seed": 1,
                        "git_commit": "a" * 40,
                        "environment_sha256": "d" * 64,
                    },
                ]
            )
        )


def test_exp19_scalar_control_requires_nonzero_empirical_operator() -> None:
    config = _config(19)
    rows = []
    for seed in range(30):
        for spec in ALL_CONDITION_SPECS:
            rows.append(
                {
                    "seed": seed,
                    "condition": spec.condition,
                    "status": "complete",
                    "behavior_balanced_accuracy": 0.7,
                    "physical_rank_fraction": 1.0,
                    "physical_dale_valid": True,
                    "moment_anchor_identifiable": True,
                    "declared_scalar_control_dimension": 1,
                    "combined_effective_control_dimension": 1,
                    "empirical_combined_control_trajectory_rank": 0,
                    "operator_control_dimension": 0,
                    "operator_delta_frobenius_norm": 0.0,
                    "heldout_normalized_closure_mse": 0.5,
                    "heldout_basis_residual_fraction": 0.2,
                    "normal_stability_eligible": True,
                    "normal_local_decay_ratio": 0.8,
                    "normal_local_max_real_part": -0.01,
                }
            )
    summary = summarize_exp19(pd.DataFrame(rows), config, n_bootstrap=200)
    control = summary.loc[summary["proposition"].eq("scalar_effective_control")]
    assert control.iloc[0]["conclusion"] == "oppose"


@pytest.mark.parametrize(
    "validator", [exp19_environment_sha256, exp20_environment_sha256]
)
def test_publication_environment_requires_python311_and_bound_packages(validator) -> None:
    packages = {
        "matplotlib": "3.10",
        "numpy": "2.0",
        "pandas": "2.0",
        "scikit-learn": "1.0",
        "scipy": "1.0",
        "statsmodels": "0.14",
        "torch": "2.0",
    }
    assert len(validator({"python": "3.11.15 build", "packages": packages})) == 64
    with pytest.raises(ValueError, match="Python 3.11"):
        validator({"python": "3.10.16 build", "packages": packages})


@pytest.mark.parametrize(
    "module", [exp19_summary_module, exp20_summary_module]
)
def test_snapshot_provenance_requires_clean_analysis_commit(monkeypatch, module) -> None:
    outputs = iter(["a" * 40 + "\n", ""])

    def clean_run(*args, **kwargs):
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(module.subprocess, "run", clean_run)
    receipt = module._analysis_provenance()
    assert receipt["analysis_git_commit"] == "a" * 40
    assert len(receipt["analysis_script_sha256"]) == 64

    outputs = iter(["a" * 40 + "\n", " M results/report.md\n"])
    with pytest.raises(ValueError, match="clean Git commit"):
        module._analysis_provenance()


def test_publication_plots_generate_raster_and_vector_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    exp19_summary = pd.DataFrame(
        [
            {
                "proposition": "heldout_behavior_fixed_checkpoint",
                "multiplicity_family": "exp19_paired_behavior_wilcoxon",
                "comparison": "md_combined_intact_vs_md_combined_clamp",
                "estimate": 0.1,
                "ci_low": 0.04,
                "ci_high": 0.13,
                "conclusion": "support",
                "threshold": 0.0,
            },
            {
                "proposition": "separate_train_only_baseline",
                "multiplicity_family": "exp19_paired_behavior_wilcoxon",
                "comparison": "md_combined_intact_vs_direct_evidence_mix",
                "estimate": -0.03,
                "ci_low": -0.05,
                "ci_high": -0.01,
                "conclusion": "oppose",
                "threshold": 0.0,
            },
            {
                "proposition": "local_normal_perturbation_decay",
                "multiplicity_family": "none_registered_threshold",
                "comparison": "registered_threshold_audit",
                "estimate": 0.5,
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "conclusion": "oppose",
                "threshold": "eligible_fraction>=0.9; decay<1.0; max_real_part<0.0",
            },
        ]
    )
    calls = []
    original_errorbar = Axes.errorbar

    def capture_errorbar(axis, *args, **kwargs):
        calls.append((args, kwargs))
        return original_errorbar(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture_errorbar)
    plot_exp19(
        exp19_summary,
        pd.DataFrame([{"status": "complete"}]),
        tmp_path / "exp19.png",
        tmp_path / "exp19.pdf",
    )
    assert calls[0][1]["xerr"][0, 0] == pytest.approx(0.06)
    assert calls[0][1]["xerr"][1, 0] == pytest.approx(0.03)
    assert calls[1][1]["fmt"] == "s"
    assert calls[1][1]["color"] == "#b2182b"

    outer = pd.DataFrame(
        [
            {
                "stage": "outer_test",
                "status": "complete",
                "condition": condition,
                "nll_per_count": 1.0 + 0.01 * index,
                "context_nll": 0.5 + 0.01 * index,
                "parameter_count": 100 + index,
            }
            for index, condition in enumerate(MODEL_CONDITIONS)
        ]
    )
    exp20_summary = pd.DataFrame(
        [
            {
                "proposition": "belief_condition_neural_prediction",
                "comparison": "md_shared_vs_common",
                "estimate": 0.02,
                "ci_low": -0.01,
                "ci_high": 0.025,
            }
        ]
    )
    exp20_call_index = len(calls)
    plot_exp20(
        outer,
        exp20_summary,
        png=tmp_path / "exp20.png",
        pdf=tmp_path / "exp20.pdf",
    )
    assert calls[exp20_call_index][1]["xerr"][0, 0] == pytest.approx(0.03)
    assert calls[exp20_call_index][1]["xerr"][1, 0] == pytest.approx(0.005)
    for name in ("exp19.png", "exp19.pdf", "exp20.png", "exp20.pdf"):
        assert (tmp_path / name).stat().st_size > 1_000

    plot_exp19(
        exp19_summary,
        pd.DataFrame([{"status": "complete"}]),
        tmp_path / "exp19_repeat.png",
        tmp_path / "exp19_repeat.pdf",
    )
    plot_exp20(
        outer,
        exp20_summary,
        png=tmp_path / "exp20_repeat.png",
        pdf=tmp_path / "exp20_repeat.pdf",
    )
    for prefix in ("exp19", "exp20"):
        for suffix in ("png", "pdf"):
            assert (tmp_path / f"{prefix}.{suffix}").read_bytes() == (
                tmp_path / f"{prefix}_repeat.{suffix}"
            ).read_bytes()


def test_exp20_snapshot_separates_results_input_from_output(
    tmp_path: Path, monkeypatch
) -> None:
    results_root = tmp_path / "raw_results"
    output_dir = tmp_path / "snapshot"
    observed = {}

    def fake_collect(root, config):
        observed["root"] = Path(root)
        return (
            pd.DataFrame([{"stage": "outer_test", "git_commit": "a" * 40}]),
            pd.DataFrame([{"git_commit": "a" * 40}]),
        )

    summary = pd.DataFrame(
        [
            {
                "proposition": "p",
                "comparison": "c",
                "estimate": 0.0,
                "ci_low": -0.1,
                "ci_high": 0.1,
                "conclusion": "inconclusive",
                "claim_scope": "test",
            }
        ]
    )

    def fake_plot(raw, scoped, *, png, pdf):
        png.write_bytes(b"png")
        pdf.write_bytes(b"pdf")

    analysis = {
        "analysis_git_commit": "b" * 40,
        "analysis_script_sha256": "c" * 64,
        "analysis_python": "3.11 test",
    }
    monkeypatch.setattr(exp20_summary_module, "_analysis_provenance", lambda: analysis)
    monkeypatch.setattr(exp20_summary_module, "collect_formal_run", fake_collect)
    monkeypatch.setattr(
        exp20_summary_module, "summarize_formal_run", lambda raw, config: summary
    )
    monkeypatch.setattr(exp20_summary_module, "_plot", fake_plot)
    paths = exp20_summary_module.publish_snapshot(
        results_root,
        {},
        output_dir=output_dir,
        prefix="scoped",
    )
    assert observed["root"] == results_root
    assert not results_root.exists()
    assert all(path.parent == output_dir for path in paths.values())
    published = pd.read_csv(paths["summary"])
    assert published["raw_run_git_commit"].iloc[0] == "a" * 40
    assert published["analysis_git_commit"].iloc[0] == "b" * 40
    report = paths["report"].read_text(encoding="utf-8")
    assert "a" * 40 in report and "b" * 40 in report


def _outer_rows() -> list[dict[str, object]]:
    provenance = {
        "access_scope": "whole_block_split_and_postfit_evaluation_only",
        "eligible_for_whole_block_split": True,
        "eligible_for_postfit_evaluation": True,
        "eligible_for_gate_input": False,
        "eligible_for_model_input": False,
        "cohort_manifest_sha256": "a" * 64,
    }
    rows = []
    for session in range(20):
        for condition in MODEL_CONDITIONS:
            fixed = condition in {
                "md_shared",
                "md_clamp",
                "md_delay_1",
                "md_delay_5",
                "md_shuffle",
            }
            rows.append(
                {
                    "stage": "outer_test",
                    "status": "complete",
                    "session_id": f"s{session}",
                    "animal_id": f"a{session}",
                    "condition": condition,
                    "statistics_unit": "animal_with_session_nested",
                    "preprocessing_fit_train_only": True,
                    "split_unit": "contiguous_true_probabilityLeft_block",
                    "probability_left_access_scope": (
                        "whole_block_split_and_postfit_evaluation_only"
                    ),
                    "likelihood_kind": "one_step_conditional_poisson",
                    "gate_received_probability_left": False,
                    "belief_uses_current_trial_stimulus": False,
                    "belief_uses_future_trials": False,
                    "belief_accessed_true_context": False,
                    "full_latent_lds": False,
                    "truth_sidecar_provenance": provenance,
                    "fit_fingerprint": "a" * 64 if fixed else "e" * 64,
                    "belief_checkpoint_sha256": "b" * 64 if fixed else None,
                    "source_belief_trajectory_sha256": "c" * 64 if fixed else None,
                    "evaluated_heldout_belief_sha256": "d" * 64 if fixed else None,
                    "belief_intervention_postfit": fixed and condition != "md_shared",
                    "all_model_parameters_frozen_for_intervention": (
                        fixed and condition != "md_shared"
                    ),
                }
            )
    return rows


def test_exp20_outer_contract_rejects_truth_leakage() -> None:
    config = _config(20)
    rows = _outer_rows()
    _validate_outer_contract(rows, config=config)
    rows[0]["gate_received_probability_left"] = True
    with pytest.raises(ValueError, match="capability"):
        _validate_outer_contract(rows, config=config)
    rows[0]["gate_received_probability_left"] = False
    rows[0]["truth_sidecar_provenance"] = {
        **rows[0]["truth_sidecar_provenance"],
        "eligible_for_whole_block_split": False,
    }
    with pytest.raises(ValueError, match="sidecar"):
        _validate_outer_contract(rows, config=config)


def test_exp20_outer_contract_rejects_missing_or_mixed_intervention_receipts() -> None:
    config = _config(20)
    rows = _outer_rows()
    _validate_outer_contract(rows, config=config)
    target = next(row for row in rows if row["condition"] == "md_shared")
    target["fit_fingerprint"] = None
    with pytest.raises(ValueError, match="provenance"):
        _validate_outer_contract(rows, config=config)

    rows = _outer_rows()
    target = next(
        row
        for row in rows
        if row["session_id"] == "s0" and row["condition"] == "md_delay_1"
    )
    target["source_belief_trajectory_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="source trajectory"):
        _validate_outer_contract(rows, config=config)


def test_exp20_summary_reuses_stored_animal_nested_inference() -> None:
    config = _config(20)
    contrasts = []
    comparators = (
        "common",
        "hmm_shared",
        "md_clamp",
        "md_delay_1",
        "md_delay_5",
        "md_shuffle",
    )
    for comparator in comparators:
        contrasts.append(
            {
                "stage": "animal_session_belief_contrast",
                "comparison": f"md_shared_vs_{comparator}",
                "effect_definition": (
                    "comparator_nll_per_count_minus_md_shared_nll_per_count"
                ),
                "inference_unit": "animal_with_session_nested",
                "estimate": 0.01,
                "ci_low": 0.001,
                "ci_high": 0.02,
                "bootstrap_p_two_sided": 0.01,
                "holm_adjusted_p": 0.03,
                "n_sessions": 20,
                "n_animals": 20,
                "conclusion": "support",
            }
        )
    interval = {
        "estimate": 0.01,
        "ci_low": 0.001,
        "ci_high": 0.02,
        "bootstrap_p_two_sided": 0.01,
        "holm_adjusted_p": 0.03,
        "n_sessions": 20,
        "n_animals": 20,
    }
    cohort = {
        "stage": "cohort_summary",
        "comparison": {"shared_vs_common": interval},
        "core_conclusion": "support",
    }
    outer = []
    for index in range(20):
        outer.append(
            {
                "stage": "outer_test",
                "status": "complete",
                "condition": "md_shared",
                "session_id": f"s{index}",
                "animal_id": f"a{index}",
                "belief_minus_behavior_switch_latency_trials": float(index % 3 - 1),
                "gate_received_probability_left": False,
                "belief_uses_current_trial_stimulus": False,
                "belief_uses_future_trials": False,
                "belief_accessed_true_context": False,
            }
        )
    # Contract completeness is computed against all outer cells, while timing
    # uses the intact rows.  Add compact placeholders for the remaining grid.
    for index in range(20):
        for condition in MODEL_CONDITIONS:
            if condition == "md_shared":
                continue
            outer.append(
                {
                    "stage": "outer_test",
                    "status": "complete",
                    "condition": condition,
                    "session_id": f"s{index}",
                    "animal_id": f"a{index}",
                }
            )
    summary = summarize_exp20(pd.DataFrame([*contrasts, cohort, *outer]), config)
    stored = summary.loc[
        summary["proposition"].eq("belief_condition_neural_prediction")
    ]
    assert len(stored) == 6
    assert set(stored["conclusion"]) == {"support"}
    joint = summary.loc[
        summary["proposition"].eq("shared_basis_joint_registered_claim")
    ].iloc[0]
    assert joint["conclusion"] == "support"
    timing = summary.loc[
        summary["proposition"].eq(
            "belief_vs_behavior_bias_switch_timing_descriptive"
        )
    ].iloc[0]
    assert timing["conclusion"] == "inconclusive"


def test_scoped_claims_integrate_idempotently(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    existing = pd.DataFrame(
        [
            {
                "claim_id": "old",
                "experiment": "exp00",
                "metric": "metric",
                "comparison": "comparison",
                "stats_unit": "seed",
                "n_planned": 1,
                "n_complete": 1,
                "n_failed": 0,
                "estimate": 0.0,
                "ci_low": -1.0,
                "ci_high": 1.0,
                "effect_size": 0.0,
                "p_value": 1.0,
                "multiplicity_method": "none",
                "conclusion": "inconclusive",
                "criterion": "test",
                "note": "existing",
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    existing.to_csv(root / "summary.csv", index=False)
    (root / "report.md").write_text("# Report\n", encoding="utf-8")
    template = pd.DataFrame(
        [
            {
                "experiment": "exp19_belief_ei_effective_dynamics",
                "proposition": "p19",
                "comparison": "c19",
                "inference_unit": "seed",
                "n_planned": 30,
                "n_complete": 30,
                "estimate": 0.1,
                "ci_low": 0.01,
                "ci_high": 0.2,
                "p_value": 0.01,
                "holm_adjusted_p": 0.02,
                "multiplicity_family": "Holm",
                "conclusion": "support",
                "threshold": ">0",
                "claim_scope": "synthetic",
            }
        ]
    )
    template.to_csv(
        root / "exp19_belief_ei_effective_dynamics_formal_summary.csv", index=False
    )
    template.assign(
        experiment="exp20_ibl_md_belief_dynamics",
        proposition="p20",
        comparison="c20",
        inference_unit="animal_with_session_nested",
        n_planned=20,
        n_complete=20,
        claim_scope="real observational",
    ).to_csv(root / "exp20_ibl_md_belief_dynamics_formal_summary.csv", index=False)
    integrate(root)
    integrate(root)
    combined = pd.read_csv(root / "summary.csv")
    assert len(combined) == 3
    assert set(combined["experiment"]) == {
        "exp00",
        "exp19_belief_ei_effective_dynamics",
        "exp20_ibl_md_belief_dynamics",
    }
    report = (root / "report.md").read_text(encoding="utf-8")
    assert report.count("<!-- exp19-exp20:start -->") == 1
