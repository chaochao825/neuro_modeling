from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments import exp28_independent_actuator_selector as exp28_runner
from experiments.exp28_independent_actuator_selector import (
    CANDIDATE_MODES,
    CONFIRMATORY_ELIGIBLE,
    EXPECTED_EVALUATION_SEEDS,
    EXPECTED_META_SEEDS,
    EXPERIMENT,
    INFERENCE_SCOPE,
    INFERENCE_STATUS,
    REQUIRED_RUN_LABEL,
    SOURCE_RECEIPT_SCHEMA,
    _file_sha256,
    _fit_frozen_selectors,
    _planned_conditions,
    _semantic_records,
    _self_hash,
    _validate_config,
    _write_receipt,
)
from scripts import summarize_exp28 as exp28_summary
from scripts.summarize_exp28 import (
    collect_exp28,
    directional_sensitivity_classification,
    summarize_independent_selector,
    validate_independent_selector_records,
    write_exp28_summary,
)
from src.data.actuator_selector_dataset import (
    Exp26SelectorSource,
    build_frozen_selector_meta_training,
    build_independent_selector_test_folds,
)
from src.models.actuator_selector import GRUSelectorBaseline
from src.plasticity.selector_three_factor import LocalThreeFactorSelector
from src.utils import artifacts
from src.utils.artifacts import ExperimentRun


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "formal" / "exp28_independent_actuator_selector.json"
)


def _generator_schema() -> tuple[tuple[object, ...], ...]:
    compositions = [
        (alpha / 10.0, rank_a, rank_b)
        for alpha in range(11)
        for rank_a in (1, 2, 4, 8)
        for rank_b in (1, 2, 4)
    ]
    discovery = compositions[:44]
    heldout = [discovery[0], *compositions[44:87]]
    rows: list[tuple[object, ...]] = []
    for split, values in (("discovery", discovery), ("heldout", heldout)):
        prefix = "d" if split == "discovery" else "h"
        for index, (alpha, rank_a, rank_b) in enumerate(values):
            rows.append(
                (
                    f"{prefix}{index:03d}",
                    split,
                    alpha,
                    rank_a,
                    rank_b,
                    float((index % 4) * 4),
                    (0.1, 0.3, 0.6, 1.0)[index % 4],
                )
            )
    return tuple(rows)


def _source(
    seeds: tuple[int, ...], *, profile: str, conclusion: str, hash_character: str
) -> Exp26SelectorSource:
    seed_values: list[int] = []
    generator_ids: list[str] = []
    generator_splits: list[str] = []
    alpha_values: list[float] = []
    transition_ranks: list[int] = []
    input_ranks: list[int] = []
    delays: list[float] = []
    noise_values: list[float] = []
    features: list[list[float]] = []
    validation: list[list[float]] = []
    test: list[list[float]] = []
    schema = _generator_schema()
    for seed in seeds:
        for index, (
            generator_id,
            split,
            alpha,
            rank_a,
            rank_b,
            delay,
            noise,
        ) in enumerate(schema):
            seed_values.append(seed)
            generator_ids.append(str(generator_id))
            generator_splits.append(str(split))
            alpha_values.append(float(alpha))
            transition_ranks.append(int(rank_a))
            input_ranks.append(int(rank_b))
            delays.append(float(delay))
            noise_values.append(float(noise))
            chi = float(alpha) * 0.8 + 0.1
            features.append(
                [
                    chi,
                    -2.0 + 0.01 * seed + 0.001 * index,
                    -1.0 - 0.005 * seed + 0.002 * index,
                    np.log2(float(rank_a)),
                    np.log2(float(rank_b)),
                    float(delay) / 4.0,
                    np.log(float(noise)),
                ]
            )
            best = index % len(CANDIDATE_MODES)
            utility = np.asarray([0.55, 0.55, 0.55], dtype=np.float64)
            utility[best] = 0.85
            validation.append(utility.tolist())
            test.append((utility + 0.001 * (seed % 5)).clip(0.0, 1.0).tolist())
    return Exp26SelectorSource(
        profile=profile,
        conclusion=conclusion,
        raw_metrics_sha256=hash_character * 64,
        conclusion_sha256="b" * 64,
        config_sha256="c" * 64,
        manifest_sha256="d" * 64,
        candidate_modes=CANDIDATE_MODES,
        seeds=np.asarray(seed_values),
        generator_ids=tuple(generator_ids),
        generator_splits=tuple(generator_splits),
        alpha=np.asarray(alpha_values),
        transition_rank=np.asarray(transition_ranks),
        input_rank=np.asarray(input_ranks),
        delay=np.asarray(delays),
        noise_std=np.asarray(noise_values),
        raw_features=np.asarray(features),
        validation_utilities=np.asarray(validation),
        test_utilities=np.asarray(test),
    )


def _fast_config(meta: object | None = None) -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["selector_fit_seed"] = 2801
    config["local_selector"]["epochs"] = 2
    config["gru_selector"]["epochs"] = 2
    if meta is not None:
        config["registered_heldout_generators"] = [
            {
                "generator_id": item.generator_id,
                "alpha": item.alpha,
                "transition_rank": item.transition_rank,
                "input_rank": item.input_rank,
                "delay": item.delay,
                "noise_std": item.noise_std,
            }
            for item in meta.generator_schema
            if item.generator_split == "heldout"
        ]
    return config


def test_bound_sensitivity_config_and_fail_closed_placeholders() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _validate_config(config)
    assert config["sensitivity_readiness"] == "ready"
    assert config["independent_source_package"]["package_dir"] == (
        "results/exp28_source_amend1_v1_28b6c76/package"
    )

    config["sensitivity_readiness"] = "INSERT_AMENDED_PACKAGE_HASHES_AND_SET_TO_READY"
    with pytest.raises(ValueError, match="not runnable"):
        _validate_config(config)

    config["sensitivity_readiness"] = "ready"
    config["independent_source_package"]["package_dir"] = (
        "INSERT_PROJECT_RELATIVE_OR_ABSOLUTE_PACKAGE_DIR"
    )
    with pytest.raises(ValueError, match="non-placeholder path"):
        _validate_config(config)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["selector_fit_seed"] = 7
    with pytest.raises(ValueError, match="must equal 2801"):
        _validate_config(config)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["analysis"]["force_inconclusive"] = False
    with pytest.raises(ValueError, match="analysis contract"):
        _validate_config(config)


def test_one_frozen_fit_drives_all_amended_seeds_and_directional_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta_source = _source(
        EXPECTED_META_SEEDS,
        profile="formal",
        conclusion="support",
        hash_character="a",
    )
    independent_source = _source(
        EXPECTED_EVALUATION_SEEDS,
        profile="independent_test",
        conclusion="inconclusive",
        hash_character="1",
    )
    meta = build_frozen_selector_meta_training(meta_source)
    folds = build_independent_selector_test_folds(meta, independent_source)
    config = _fast_config(meta)

    counts = {"local": 0, "gru": 0}
    original_local_fit = LocalThreeFactorSelector.fit
    original_gru_fit = GRUSelectorBaseline.fit
    expected_train_utilities = np.asarray(meta.train_validation_utilities)
    independent_test_utilities = np.vstack([fold.test_utilities for fold in folds])

    def local_fit(self: LocalThreeFactorSelector, cues: object, utilities: object):
        counts["local"] += 1
        np.testing.assert_array_equal(utilities, expected_train_utilities)
        assert np.asarray(utilities).shape != independent_test_utilities.shape or not (
            np.array_equal(utilities, independent_test_utilities)
        )
        return original_local_fit(self, cues, utilities)

    def gru_fit(self: GRUSelectorBaseline, cues: object, utilities: object):
        counts["gru"] += 1
        np.testing.assert_array_equal(utilities, expected_train_utilities)
        assert np.asarray(utilities).shape != independent_test_utilities.shape or not (
            np.array_equal(utilities, independent_test_utilities)
        )
        return original_gru_fit(self, cues, utilities)

    monkeypatch.setattr(LocalThreeFactorSelector, "fit", local_fit)
    monkeypatch.setattr(GRUSelectorBaseline, "fit", gru_fit)
    fit = _fit_frozen_selectors(meta, folds, config)

    assert counts == {"local": 1, "gru": 1}
    assert fit.fit_receipt["fit_count"] == 1
    assert fit.fit_receipt["independent_rows_used_for_fit"] == 0
    assert fit.fit_receipt["normalizer"]["fit_count"] == 1
    assert fit.fit_receipt["fixed_best"]["fit_count"] == 1
    assert fit.decision_receipt["evaluation_seed_order"] == list(
        EXPECTED_EVALUATION_SEEDS
    )
    assert fit.probabilities["local_three_factor"].shape == (30 * 44, 3)

    source_receipt = _self_hash(
        {"schema_version": "test", "inference_scope": INFERENCE_SCOPE}
    )
    records = _semantic_records(
        meta,
        folds,
        fit,
        config,
        run_label="test",
        config_sha256="e" * 64,
        run_git={"commit": "f" * 40, "tree": "0" * 40, "dirty": False},
        source_receipt=source_receipt,
        source_receipt_file_sha256="1" * 64,
        fit_receipt_file_sha256="2" * 64,
        decision_receipt_file_sha256="3" * 64,
    )
    frame = pd.DataFrame(
        {
            "seed": 2801,
            **dimensions,
            **metrics,
        }
        for metrics, dimensions in records
    )
    primary = validate_independent_selector_records(
        frame,
        selector_fit_seed=2801,
        expected_primary_generators_per_seed=43,
    )
    assert len(frame) == 30 * 44 * 4
    assert len(primary) == 30 * 43 * 4
    assert set(frame["training_source_seed_count"]) == {30}
    assert set(frame["independent_rows_used_for_fit"]) == {0}
    assert set(frame["selector_fit_count"]) == {1}
    assert len(_planned_conditions(config)) == len(frame)

    conclusion = summarize_independent_selector(
        frame,
        selector_fit_seed=2801,
        expected_primary_generators_per_seed=43,
        noninferiority_fraction=0.8,
        bootstrap_samples=50,
        permutation_samples=100,
        confidence=0.95,
        random_seed=2801,
    )
    assert conclusion.n_seeds == 30
    assert conclusion.conclusion == "inconclusive"
    assert "forced inconclusive" in conclusion.reason
    directional = directional_sensitivity_classification(conclusion)
    assert directional["classification"] in {"support", "oppose", "inconclusive"}
    assert directional["non_confirmatory"] is True
    assert directional["inference_status"] == INFERENCE_STATUS
    assert conclusion.statistics_unit == "post_hoc_amended_evaluation_seed"
    assert conclusion.complete_primary_coverage is True
    assert {item.seed for item in conclusion.seed_endpoints} == set(
        EXPECTED_EVALUATION_SEEDS
    )


def test_independent_validator_fails_closed_on_incomplete_schema() -> None:
    rows = pd.DataFrame(
        {
            "seed": [2801],
            "selector_fit_seed": [2801],
            "evaluation_seed": [30],
        }
    )
    with pytest.raises(ValueError, match="lack columns"):
        validate_independent_selector_records(
            rows,
            selector_fit_seed=2801,
            expected_primary_generators_per_seed=43,
        )


def test_source_failure_occurs_after_plan_and_retains_every_condition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fast_config()
    calls = {"load_sources": 0}

    def fail_source(value: object) -> object:
        calls["load_sources"] += 1
        raise ValueError("injected meta source failure")

    clean_git = {"commit": "a" * 40, "tree": "b" * 40, "dirty": False}
    monkeypatch.setattr(exp28_runner, "_validate_config", lambda value: None)
    monkeypatch.setattr(exp28_runner, "_load_sources", fail_source)
    monkeypatch.setattr(exp28_runner, "git_identity", lambda: clean_git)
    with pytest.raises(ValueError, match="injected meta source failure"):
        exp28_runner.run_experiment(
            config,
            tmp_path,
            run_label=REQUIRED_RUN_LABEL,
        )

    assert calls["load_sources"] == 1
    attempts = list((tmp_path / "runs" / EXPERIMENT / "seed_2801").iterdir())
    assert len(attempts) == 1
    attempt = attempts[0]
    plan = json.loads((attempt / "planned_conditions.json").read_text())
    metrics = [
        json.loads(line)
        for line in (attempt / "metrics.jsonl").read_text().splitlines()
    ]
    status = json.loads((attempt / "status.json").read_text())
    assert len(plan) == 30 * 44 * 4
    assert len(metrics) == len(plan)
    assert {row["status"] for row in metrics} == {"failed"}
    assert {row["failure_reason"] for row in metrics} == {
        "injected meta source failure"
    }
    assert status["status"] == "failed"


@pytest.mark.parametrize(
    "tampered_field,pattern",
    [
        ("expected_source_panel_receipt_file_sha256", "receipt-file"),
        ("expected_conclusion_file_sha256", "conclusion-file"),
        ("expected_raw_metrics_sha256", "raw_metrics_sha256"),
        ("expected_receipt_payload_sha256", "receipt_payload_sha256"),
        ("expected_registered_config_sha256", "registered_config_sha256"),
        ("expected_registered_config_file_sha256", "registered config file"),
        ("expected_protocol_amendment_sha256", "protocol_amendment_sha256"),
        ("receipt_inference_status", "metadata binding"),
        ("receipt_confirmatory_independence", "metadata binding"),
        ("receipt_amended_max_scale", "metadata binding"),
        ("receipt_performance_metrics_inspected", "metadata binding"),
    ],
)
def test_independent_package_identity_fields_fail_closed_on_one_bit_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_field: str,
    pattern: str,
) -> None:
    meta = build_frozen_selector_meta_training(
        _source(
            EXPECTED_META_SEEDS,
            profile="formal",
            conclusion="support",
            hash_character="a",
        )
    )
    independent = _source(
        EXPECTED_EVALUATION_SEEDS,
        profile="independent_test",
        conclusion="inconclusive",
        hash_character="1",
    )
    receipt_path = tmp_path / "source_panel_receipt.json"
    conclusion_path = tmp_path / "conclusion.json"
    raw_path = tmp_path / "raw_metrics.jsonl"
    registered_config_path = tmp_path / "amended_source_config.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    conclusion_path.write_text("{}\n", encoding="utf-8")
    raw_path.write_text('{"row": 1}\n', encoding="utf-8")
    registered_config_path.write_text('{"amended": true}\n', encoding="utf-8")
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    registered_file_sha = _file_sha256(registered_config_path)
    amendment = {
        "amendment_id": "exp28_ceiling_reachability_amendment_1",
        "previous_max_scale": 128.0,
        "amended_max_scale": 256.0,
        "performance_metrics_inspected": True,
        "performance_metrics_used_for_amendment": False,
    }
    package_receipt = {
        "schema_version": ("exp28_independent_source_package_v1_ceiling_amendment_1"),
        "protocol_version": ("exp28_exp26_independent_source_v1_ceiling_amendment_1"),
        "protocol_amendment": amendment,
        "protocol_amendment_sha256": "5" * 64,
        "inference_status": INFERENCE_STATUS,
        "confirmatory_independence_restored": False,
        "raw_metrics_file": raw_path.name,
        "raw_metrics_sha256": raw_sha,
        "receipt_payload_sha256": "2" * 64,
        "registered_config_sha256": "3" * 64,
        "registered_config_file_sha256": registered_file_sha,
        "source_contract_sha256": "4" * 64,
    }
    package = SimpleNamespace(receipt=package_receipt, rows=())
    config = _fast_config(meta)
    config["independent_source_package"] = {
        "package_dir": str(tmp_path),
        "registered_source_config_path": str(registered_config_path),
        "expected_source_panel_receipt_file_sha256": _file_sha256(receipt_path),
        "expected_conclusion_file_sha256": _file_sha256(conclusion_path),
        "expected_raw_metrics_sha256": raw_sha,
        "expected_receipt_payload_sha256": "2" * 64,
        "expected_registered_config_sha256": "3" * 64,
        "expected_registered_config_file_sha256": registered_file_sha,
        "expected_protocol_amendment_sha256": "5" * 64,
        "expected_package_schema_version": (
            "exp28_independent_source_package_v1_ceiling_amendment_1"
        ),
        "expected_source_protocol_version": (
            "exp28_exp26_independent_source_v1_ceiling_amendment_1"
        ),
        "expected_amendment_id": "exp28_ceiling_reachability_amendment_1",
        "expected_inference_status": INFERENCE_STATUS,
        "expected_confirmatory_independence_restored": False,
        "expected_previous_max_scale": 128.0,
        "expected_amended_max_scale": 256.0,
        "expected_performance_metrics_inspected": True,
        "expected_performance_metrics_used_for_amendment": False,
    }
    if tampered_field.startswith("expected_"):
        config["independent_source_package"][tampered_field] = "f" * 64
    elif tampered_field == "receipt_inference_status":
        package_receipt["inference_status"] = "confirmatory"
    elif tampered_field == "receipt_confirmatory_independence":
        package_receipt["confirmatory_independence_restored"] = True
    elif tampered_field == "receipt_amended_max_scale":
        amendment["amended_max_scale"] = 512.0
    else:
        amendment["performance_metrics_inspected"] = False
    monkeypatch.setattr(exp28_runner, "_load_meta_source", lambda value: meta)
    monkeypatch.setattr(
        exp28_runner, "load_source_panel_package", lambda *args, **kwargs: package
    )
    monkeypatch.setattr(
        exp28_runner,
        "exp26_selector_source_from_independent_package",
        lambda value: independent,
    )
    with pytest.raises(ValueError, match=pattern):
        exp28_runner._load_sources(config)


def test_collector_replays_exact_rows_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = build_frozen_selector_meta_training(
        _source(
            EXPECTED_META_SEEDS,
            profile="formal",
            conclusion="support",
            hash_character="a",
        )
    )
    folds = build_independent_selector_test_folds(
        meta,
        _source(
            EXPECTED_EVALUATION_SEEDS,
            profile="independent_test",
            conclusion="inconclusive",
            hash_character="1",
        ),
    )
    config = _fast_config(meta)
    config["analysis"]["bootstrap_samples"] = 50
    config["analysis"]["permutation_samples"] = 100
    fit = _fit_frozen_selectors(meta, folds, config)
    source_receipt = _self_hash(
        {
            "schema_version": SOURCE_RECEIPT_SCHEMA,
            "inference_scope": INFERENCE_SCOPE,
            "inference_status": INFERENCE_STATUS,
            "confirmatory_eligible": CONFIRMATORY_ELIGIBLE,
            "independent_package": {
                "protocol_amendment_sha256": "9" * 64,
            },
        }
    )
    run_git = {"commit": "a" * 40, "tree": "b" * 40, "dirty": False}
    monkeypatch.setattr(
        artifacts,
        "_software_provenance",
        lambda: {"packages": {}, "git": run_git},
    )
    monkeypatch.setattr(exp28_summary, "_validate_config", lambda value: None)
    replay_counts = {"load_sources": 0}

    def replay_sources(value: object):
        replay_counts["load_sources"] += 1
        return meta, folds, source_receipt

    monkeypatch.setattr(
        exp28_summary,
        "_load_sources",
        replay_sources,
    )
    monkeypatch.setattr(
        exp28_summary,
        "_fit_frozen_selectors",
        lambda meta_value, fold_value, config_value: fit,
    )
    monkeypatch.setattr(exp28_summary, "git_identity", lambda: run_git)
    config_sha = exp28_summary.canonical_config_sha256(config)
    run_label = REQUIRED_RUN_LABEL
    run_config = {
        **config,
        "evidence_provenance": {
            "schema_version": "exp28_amended_sensitivity_selector_evidence_v1",
            "canonical_config_sha256": config_sha,
            "git": run_git,
            "run_label": run_label,
            "inference_scope": INFERENCE_SCOPE,
            "inference_status": INFERENCE_STATUS,
            "confirmatory_eligible": CONFIRMATORY_ELIGIBLE,
        },
    }
    plan = _planned_conditions(config)
    with ExperimentRun(
        EXPERIMENT,
        2801,
        run_config,
        results_root=tmp_path,
        run_label=run_label,
    ) as run:
        run.register_conditions(plan)
        source_file = _write_receipt(
            run.path / "source_package_receipt.json", source_receipt
        )
        fit_file = _write_receipt(
            run.path / "selector_fit_receipt.json", fit.fit_receipt
        )
        decision_file = _write_receipt(
            run.path / "decision_receipt.json", fit.decision_receipt
        )
        semantic = _semantic_records(
            meta,
            folds,
            fit,
            config,
            run_label=run_label,
            config_sha256=config_sha,
            run_git=run_git,
            source_receipt=source_receipt,
            source_receipt_file_sha256=source_file,
            fit_receipt_file_sha256=fit_file,
            decision_receipt_file_sha256=decision_file,
        )
        for metrics, dimensions in semantic:
            run.record(metrics, **dimensions)
        attempt = run.path

    collection = collect_exp28(tmp_path, config=config, run_label=run_label)
    assert replay_counts["load_sources"] == 1
    assert len(collection.raw) == 30 * 44 * 4
    assert collection.fit_receipt["fit_count"] == 1
    assert collection.source_receipt_file_sha256 == _file_sha256(
        attempt / "source_package_receipt.json"
    )
    summary_dir = tmp_path / "summary"
    summary_conclusion = write_exp28_summary(collection, summary_dir, make_figure=False)
    summary_payload = json.loads((summary_dir / "conclusion.json").read_text())
    report = (summary_dir / "report.md").read_text(encoding="utf-8")
    assert summary_conclusion.conclusion == "inconclusive"
    assert summary_payload["confirmatory_eligible"] is False
    assert summary_payload["overall_classification_forced"] is True
    assert summary_payload["force_inconclusive"] is True
    assert summary_payload["inference_status"] == INFERENCE_STATUS
    assert summary_payload["directional_sensitivity"]["classification"] in {
        "support",
        "oppose",
        "inconclusive",
    }
    assert summary_payload["historical_frozen_source_classification"] == "oppose"
    assert summary_payload["historical_selector_formal_classification"] == (
        "inconclusive"
    )
    assert "seeds `60--89`" in report
    assert "overall `INCONCLUSIVE`" in report

    metrics_path = attempt / "metrics.jsonl"
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["utility"] = float(first["utility"]) - 0.01
    lines[0] = json.dumps(first, sort_keys=True)
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="semantic-row replay"):
        collect_exp28(tmp_path, config=config, run_label=run_label)
