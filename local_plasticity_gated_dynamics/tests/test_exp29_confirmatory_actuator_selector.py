from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import exp29_confirmatory_actuator_selector as runner
from experiments.exp29_confirmatory_actuator_selector import (
    DECISION_RECEIPT_SCHEMA,
    EXPECTED_EVALUATION_SEEDS,
    EXPECTED_META_SEEDS,
    EXPERIMENT,
    FIT_RECEIPT_SCHEMA,
    INFERENCE_SCOPE,
    INFERENCE_STATUS,
    REQUIRED_RUN_LABEL,
    REGISTERED_ANALYSIS_CONTRACT_SHA256,
    SOURCE_RECEIPT_SCHEMA,
    _fit_frozen_selectors,
    _planned_conditions,
    _self_hash,
    _semantic_records,
    _validate_config,
    _validate_implementation_binding,
    _write_receipt,
    analysis_contract_sha256,
    normalized_source_sha256,
)
from scripts import summarize_exp29_selector as summary
from scripts.package_exp29_confirmatory_source_panel import SourcePanelPackage
from scripts.summarize_exp29_selector import (
    collect_exp29,
    plot_exp29_selector,
    summarize_confirmatory_selector,
    validate_confirmatory_selector_records,
    write_exp29_summary,
)
from src.data.actuator_selector_dataset import (
    CANDIDATE_MODES,
    Exp26SelectorSource,
    build_frozen_selector_meta_training,
)
from src.data.exp29_feasibility_selector_dataset import (
    ConfirmatorySelectorSource,
    build_confirmatory_selector_folds,
)
from src.models.actuator_selector import GRUSelectorBaseline
from src.plasticity.selector_three_factor import LocalThreeFactorSelector
from src.utils import artifacts
from src.utils.artifacts import ExperimentRun


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "formal" / "exp29_confirmatory_actuator_selector.json"
)


def _generator_schema() -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for split, prefix in (("discovery", "d"), ("heldout", "h")):
        for index in range(44):
            rows.append(
                (
                    f"{prefix}{index:03d}",
                    split,
                    (index % 11) / 10.0,
                    (1, 2, 4, 8)[index % 4],
                    (1, 2, 4)[index % 3],
                    float((index % 4) * 4),
                    (0.1, 0.3, 0.6, 1.0)[index % 4],
                )
            )
    return tuple(rows)


def _feature(
    *,
    seed: int,
    index: int,
    alpha: float,
    transition_rank: int,
    input_rank: int,
    delay: float,
    noise: float,
) -> list[float]:
    return [
        alpha * 0.8 + 0.1,
        -2.0 + 0.001 * seed + 0.002 * index,
        -1.0 - 0.001 * seed + 0.001 * index,
        np.log2(float(transition_rank)),
        np.log2(float(input_rank)),
        delay / 4.0,
        np.log(noise),
    ]


def _meta_training():
    seed_values: list[int] = []
    ids: list[str] = []
    splits: list[str] = []
    alphas: list[float] = []
    transition_ranks: list[int] = []
    input_ranks: list[int] = []
    delays: list[float] = []
    noise_values: list[float] = []
    features: list[list[float]] = []
    validation: list[list[float]] = []
    test: list[list[float]] = []
    schema = _generator_schema()
    for seed in EXPECTED_META_SEEDS:
        for index, (
            generator_id,
            split,
            alpha,
            transition_rank,
            input_rank,
            delay,
            noise,
        ) in enumerate(schema):
            seed_values.append(seed)
            ids.append(str(generator_id))
            splits.append(str(split))
            alphas.append(float(alpha))
            transition_ranks.append(int(transition_rank))
            input_ranks.append(int(input_rank))
            delays.append(float(delay))
            noise_values.append(float(noise))
            features.append(
                _feature(
                    seed=seed,
                    index=index,
                    alpha=float(alpha),
                    transition_rank=int(transition_rank),
                    input_rank=int(input_rank),
                    delay=float(delay),
                    noise=float(noise),
                )
            )
            best = index % 3
            utility = np.asarray([0.64, 0.62, 0.60])
            utility[best] += 0.20
            validation.append(utility.tolist())
            test.append(utility.tolist())
    source = Exp26SelectorSource(
        profile="formal",
        conclusion="support",
        raw_metrics_sha256="a" * 64,
        conclusion_sha256="b" * 64,
        config_sha256="c" * 64,
        manifest_sha256="d" * 64,
        candidate_modes=CANDIDATE_MODES,
        seeds=np.asarray(seed_values),
        generator_ids=tuple(ids),
        generator_splits=tuple(splits),
        alpha=np.asarray(alphas),
        transition_rank=np.asarray(transition_ranks),
        input_rank=np.asarray(input_ranks),
        delay=np.asarray(delays),
        noise_std=np.asarray(noise_values),
        raw_features=np.asarray(features),
        validation_utilities=np.asarray(validation),
        test_utilities=np.asarray(test),
    )
    return build_frozen_selector_meta_training(source)


def _confirmatory_source(meta: object) -> ConfirmatorySelectorSource:
    seeds: list[int] = []
    ids: list[str] = []
    splits: list[str] = []
    features: list[list[float]] = []
    feasible: list[list[bool]] = []
    validation: list[list[float]] = []
    test: list[list[float]] = []
    frozen_validation: list[float] = []
    frozen_test: list[float] = []
    for seed in EXPECTED_EVALUATION_SEEDS:
        for index, spec in enumerate(meta.generator_schema):
            seeds.append(seed)
            ids.append(spec.generator_id)
            splits.append(spec.generator_split)
            features.append(
                _feature(
                    seed=seed,
                    index=index,
                    alpha=spec.alpha,
                    transition_rank=spec.transition_rank,
                    input_rank=spec.input_rank,
                    delay=spec.delay,
                    noise=spec.noise_std,
                )
            )
            flags = [index % 7 != 0, index % 11 != 0, index % 13 != 0]
            frozen = 0.76 if index % 17 == 0 else 0.54 + 0.001 * (seed % 5)
            active = np.asarray([0.65, 0.69, 0.67]) + 0.002 * (index % 3)
            active[~np.asarray(flags)] = frozen
            feasible.append(flags)
            validation.append(active.tolist())
            test.append(active.tolist())
            frozen_validation.append(frozen)
            frozen_test.append(frozen)
    return ConfirmatorySelectorSource(
        package_receipt_sha256="1" * 64,
        package_receipt_file_sha256="2" * 64,
        package_conclusion_file_sha256="3" * 64,
        raw_metrics_sha256="4" * 64,
        source_config_sha256="5" * 64,
        source_manifest_sha256=meta.source_manifest_sha256,
        implementation_contract_sha256="6" * 64,
        statistics_unit="seed",
        seeds=np.asarray(seeds),
        generator_ids=tuple(ids),
        generator_splits=tuple(splits),
        generator_schema=meta.generator_schema,
        raw_features=np.asarray(features),
        candidate_feasible=np.asarray(feasible),
        validation_deployment_utilities=np.asarray(validation),
        test_deployment_utilities=np.asarray(test),
        frozen_validation_utilities=np.asarray(frozen_validation),
        frozen_test_utilities=np.asarray(frozen_test),
    )


def _ready_config(*, fast: bool, meta: object | None = None) -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["confirmatory_readiness"] = "ready"
    package = config["confirmatory_source_package"]
    package["package_dir"] = "results/synthetic-exp29-package"
    for key in (
        "expected_source_panel_receipt_file_sha256",
        "expected_conclusion_file_sha256",
        "expected_raw_metrics_sha256",
        "expected_receipt_payload_sha256",
    ):
        package[key] = "e" * 64
    if fast:
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


def _records(meta: object, folds: object, config: dict[str, object]):
    fit = _fit_frozen_selectors(meta, folds, config)
    source_receipt = _self_hash(
        {
            "schema_version": SOURCE_RECEIPT_SCHEMA,
            "inference_scope": INFERENCE_SCOPE,
            "inference_status": INFERENCE_STATUS,
        }
    )
    run_git = {"commit": "a" * 40, "tree": "b" * 40, "dirty": False}
    semantic = _semantic_records(
        meta,
        folds,
        fit,
        config,
        run_label=REQUIRED_RUN_LABEL,
        config_sha256="c" * 64,
        run_git=run_git,
        source_receipt=source_receipt,
        source_receipt_file_sha256="6" * 64,
        fit_receipt_file_sha256="7" * 64,
        decision_receipt_file_sha256="8" * 64,
    )
    frame = pd.DataFrame(
        {"seed": 2801, **dimensions, **metrics} for metrics, dimensions in semantic
    )
    return fit, source_receipt, run_git, semantic, frame


def test_registration_fails_closed_and_has_no_force_inconclusive() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="not runnable"):
        _validate_config(config)
    ready = _ready_config(fast=False)
    _validate_config(ready)
    assert ready["evaluation_seeds"] == list(range(60, 90))
    assert ready["analysis"].get("force_inconclusive") is None
    assert ready["evaluation"]["primary_scope"] == (
        "unconditional_all_registered_heldout_cells"
    )
    ready["analysis"]["force_inconclusive"] = True
    with pytest.raises(ValueError, match="analysis.contract"):
        _validate_config(ready)


def test_two_phase_contract_allows_only_package_materialization_and_binds_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    registered = analysis_contract_sha256(template)
    assert registered == REGISTERED_ANALYSIS_CONTRACT_SHA256
    materialized = _ready_config(fast=False)
    assert analysis_contract_sha256(materialized) == registered
    materialized["config_path"] = "runtime-only.json"
    assert analysis_contract_sha256(materialized) == registered

    for mutate in (
        lambda value: value["local_selector"].__setitem__("epochs", 201),
        lambda value: value["analysis"].__setitem__("alpha", 0.01),
        lambda value: value["evaluation"].__setitem__(
            "primary_scope", "strict_unseen_only"
        ),
        lambda value: value["confirmatory_source_package"].__setitem__(
            "expected_registered_config_sha256", "f" * 64
        ),
    ):
        changed = json.loads(json.dumps(materialized))
        mutate(changed)
        assert analysis_contract_sha256(changed) != registered

    package = template["confirmatory_source_package"]
    assert package["expected_registered_config_sha256"] == (
        "70db02c9e578ace8a1719e4ff6c71c07048a1836f2dafc9fcd845ad5b0bd9e14"
    )
    assert package["expected_registered_config_file_sha256"] == (
        "a058e175d2634cbcb0dde68e1709904c897f51ec8055fe9de666b0e1cffd7500"
    )
    assert package["expected_source_contract_sha256"] == (
        "f548724759f2b95a862feca564dfc9897a7356ee81ac949b20f0106077e689c3"
    )

    binding = template["implementation_binding"]
    for path_key, hash_key in (
        ("runner_path", "runner_normalized_sha256"),
        ("summarizer_path", "summarizer_normalized_sha256"),
    ):
        source_path = PROJECT_ROOT / binding[path_key]
        source_text = source_path.read_text(encoding="utf-8")
        assert source_text.count(f'"{REGISTERED_ANALYSIS_CONTRACT_SHA256}"') == 1
        assert normalized_source_sha256(source_path) == binding[hash_key]
        literal_only = tmp_path / f"literal-{source_path.name}"
        literal_only.write_text(
            source_text.replace(
                REGISTERED_ANALYSIS_CONTRACT_SHA256,
                "f" * 64,
                1,
            ),
            encoding="utf-8",
        )
        assert normalized_source_sha256(literal_only) == binding[hash_key]
        tampered = tmp_path / f"tampered-{source_path.name}"
        tampered.write_text(
            source_text + "\n# implementation tamper\n", encoding="utf-8"
        )
        assert normalized_source_sha256(tampered) != binding[hash_key]

    for relative in (binding["runner_path"], binding["summarizer_path"]):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            (PROJECT_ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8"
        )
    runner_copy = tmp_path / binding["runner_path"]
    runner_copy.write_text(
        runner_copy.read_text(encoding="utf-8") + "\n# implementation tamper\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="implementation hash mismatch"):
        _validate_implementation_binding(template)


def test_one_meta_fit_scores_all_cells_with_frozen_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    meta = _meta_training()
    source = _confirmatory_source(meta)
    folds = build_confirmatory_selector_folds(meta, source)
    config = _ready_config(fast=True, meta=meta)
    counts = {"local": 0, "gru": 0}
    expected_train = np.asarray(meta.train_validation_utilities)
    original_local = LocalThreeFactorSelector.fit
    original_gru = GRUSelectorBaseline.fit

    def local_fit(self: LocalThreeFactorSelector, cues: object, utilities: object):
        counts["local"] += 1
        np.testing.assert_array_equal(utilities, expected_train)
        return original_local(self, cues, utilities)

    def gru_fit(self: GRUSelectorBaseline, cues: object, utilities: object):
        counts["gru"] += 1
        np.testing.assert_array_equal(utilities, expected_train)
        return original_gru(self, cues, utilities)

    monkeypatch.setattr(LocalThreeFactorSelector, "fit", local_fit)
    monkeypatch.setattr(GRUSelectorBaseline, "fit", gru_fit)
    fit, _, _, _, frame = _records(meta, folds, config)
    validated = validate_confirmatory_selector_records(frame)
    assert counts == {"local": 1, "gru": 1}
    assert fit.fit_receipt["confirmatory_rows_used_for_fit"] == 0
    assert len(validated) == 30 * 44 * 4
    assert validated["primary_endpoint_eligible"].all()
    assert validated["unconditional_cell_retained"].all()
    oracle = validated[validated["selector"] == "oracle"]
    for mode in CANDIDATE_MODES:
        infeasible = ~oracle[f"candidate_{mode}_feasible"].astype(bool)
        np.testing.assert_array_equal(
            oracle.loc[infeasible, f"candidate_{mode}_utility"],
            oracle.loc[infeasible, "candidate_frozen_utility"],
        )
    conclusion = summarize_confirmatory_selector(
        frame,
        noninferiority_fraction=0.8,
        bootstrap_samples=100,
        permutation_samples=200,
        confidence=0.95,
        random_seed=2901,
        alpha=0.05,
    )
    assert conclusion.conclusion in {"support", "oppose", "inconclusive"}
    assert conclusion.conclusion != "invalid"
    assert conclusion.n_seeds == 30
    assert conclusion.unconditional_all_registered_cells is True
    assert all(item.registered_generators == 44 for item in conclusion.seed_endpoints)
    png, pdf = plot_exp29_selector(conclusion, tmp_path / "exp29")
    assert png.is_file() and pdf.is_file()


def test_unexpected_failure_is_retained_invalid_and_cannot_be_replaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    meta = _meta_training()
    config = _ready_config(fast=True, meta=meta)
    clean_git = {"commit": "a" * 40, "tree": "b" * 40, "dirty": False}
    monkeypatch.setattr(runner, "_validate_config", lambda value: None)
    monkeypatch.setattr(
        runner,
        "_load_sources",
        lambda value: (_ for _ in ()).throw(RuntimeError("injected source failure")),
    )
    monkeypatch.setattr(runner, "git_identity", lambda: clean_git)
    monkeypatch.setattr(
        artifacts,
        "_software_provenance",
        lambda: {"packages": {}, "git": clean_git},
    )
    with pytest.raises(RuntimeError, match="injected source failure"):
        runner.run_experiment(config, tmp_path, run_label=REQUIRED_RUN_LABEL)
    attempt_root = tmp_path / "runs" / EXPERIMENT / "seed_2801"
    attempts = list(attempt_root.iterdir())
    assert len(attempts) == 1
    rows = [
        json.loads(line)
        for line in (attempts[0] / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 30 * 44 * 4
    assert {row["status"] for row in rows} == {"failed"}
    with pytest.raises(FileExistsError, match="rerun/replacement"):
        runner.run_experiment(config, tmp_path, run_label=REQUIRED_RUN_LABEL)

    monkeypatch.setattr(summary, "_validate_config", lambda value: None)
    monkeypatch.setattr(
        summary,
        "analysis_contract_sha256",
        lambda value: summary.REGISTERED_ANALYSIS_CONTRACT_SHA256,
    )
    monkeypatch.setattr(summary, "_validate_implementation_binding", lambda value: None)
    collection = collect_exp29(tmp_path, config=config, run_label=REQUIRED_RUN_LABEL)
    assert collection.evidence_valid is False
    output = tmp_path / "invalid-summary"
    conclusion = write_exp29_summary(collection, output, make_figure=True)
    assert conclusion.conclusion == "invalid"
    assert conclusion.confirmatory_eligible is False
    assert (output / "exp29_selector_evidence.png").is_file()


def test_collector_replays_valid_one_shot_and_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    meta = _meta_training()
    source = _confirmatory_source(meta)
    folds = build_confirmatory_selector_folds(meta, source)
    config = _ready_config(fast=True, meta=meta)
    fit, source_receipt, run_git, semantic, _ = _records(meta, folds, config)
    monkeypatch.setattr(
        artifacts,
        "_software_provenance",
        lambda: {"packages": {}, "git": run_git},
    )
    monkeypatch.setattr(summary, "_validate_config", lambda value: None)
    monkeypatch.setattr(
        summary,
        "analysis_contract_sha256",
        lambda value: summary.REGISTERED_ANALYSIS_CONTRACT_SHA256,
    )
    monkeypatch.setattr(summary, "_validate_implementation_binding", lambda value: None)
    monkeypatch.setattr(summary, "git_identity", lambda: run_git)
    monkeypatch.setattr(
        summary, "_load_sources", lambda value: (meta, folds, source_receipt)
    )
    monkeypatch.setattr(
        summary,
        "_fit_frozen_selectors",
        lambda meta_value, fold_value, config_value: fit,
    )
    config_sha = summary.canonical_config_sha256(config)
    run_config = {
        **config,
        "evidence_provenance": {
            "schema_version": "exp29_confirmatory_selector_evidence_v1",
            "canonical_config_sha256": config_sha,
            "git": run_git,
            "run_label": REQUIRED_RUN_LABEL,
            "inference_scope": INFERENCE_SCOPE,
            "inference_status": INFERENCE_STATUS,
            "confirmatory_eligible": True,
            "one_shot_attempt": True,
        },
    }
    with ExperimentRun(
        EXPERIMENT,
        2801,
        run_config,
        results_root=tmp_path,
        run_label=REQUIRED_RUN_LABEL,
    ) as run:
        run.register_conditions(_planned_conditions(config))
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
            run_label=REQUIRED_RUN_LABEL,
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
    collection = collect_exp29(tmp_path, config=config, run_label=REQUIRED_RUN_LABEL)
    assert collection.evidence_valid is True
    assert len(collection.raw) == 30 * 44 * 4
    output = tmp_path / "valid-summary"
    conclusion = write_exp29_summary(collection, output, make_figure=False)
    assert conclusion.conclusion in {"support", "oppose", "inconclusive"}
    payload = json.loads((output / "conclusion.json").read_text())
    assert payload["confirmatory_eligible"] is True
    assert payload["unconditional_all_registered_cells"] is True

    metrics_path = attempt / "metrics.jsonl"
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["utility"] = float(first["utility"]) - 0.01
    lines[0] = json.dumps(first, sort_keys=True)
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="semantic-row replay"):
        collect_exp29(tmp_path, config=config, run_label=REQUIRED_RUN_LABEL)


@pytest.mark.parametrize(
    "tampered_field,pattern",
    [
        ("expected_source_panel_receipt_file_sha256", "receipt-file"),
        ("expected_conclusion_file_sha256", "conclusion-file"),
        ("expected_raw_metrics_sha256", "binding mismatch"),
        ("expected_receipt_payload_sha256", "binding mismatch"),
        ("expected_registered_config_sha256", "binding mismatch"),
        ("expected_registered_config_file_sha256", "registered source config"),
        ("expected_source_contract_sha256", "binding mismatch"),
        ("receipt_metadata", "metadata"),
    ],
)
def test_package_binding_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tampered_field: str,
    pattern: str,
) -> None:
    meta = _meta_training()
    source = _confirmatory_source(meta)
    config = _ready_config(fast=True, meta=meta)
    package_config = config["confirmatory_source_package"]
    package_config["package_dir"] = str(tmp_path)
    registered = tmp_path / "source.json"
    registered.write_text("{}\n", encoding="utf-8")
    receipt_path = tmp_path / "source_panel_receipt.json"
    conclusion_path = tmp_path / "conclusion.json"
    raw_path = tmp_path / "raw_metrics.jsonl"
    receipt_path.write_text("{}\n", encoding="utf-8")
    conclusion_path.write_text("{}\n", encoding="utf-8")
    raw_path.write_text("{}\n", encoding="utf-8")
    receipt = {
        "schema_version": "exp29_confirmatory_source_package_v1",
        "protocol_version": "exp29_confirmatory_source_v1",
        "evidence_schema_version": "exp29_confirmatory_source_evidence_v1",
        "experiment": "exp29_confirmatory_source_panel",
        "profile": "confirmatory_test",
        "evidence_role": "confirmatory_test_source_only",
        "conclusion": "inconclusive",
        "standalone_inference_performed": False,
        "standalone_inference_permitted": False,
        "registered_config_sha256": "5" * 64,
        "registered_config_file_sha256": runner._file_sha256(registered),
        "source_contract_sha256": "6" * 64,
        "raw_metrics_file": raw_path.name,
        "run_provenance": {},
        "coverage": {"source_panel_valid": True},
    }
    package = SourcePanelPackage(
        receipt=receipt,
        rows=(),
        receipt_payload_sha256="4" * 64,
        receipt_file_sha256=runner._file_sha256(receipt_path),
        conclusion_file_sha256=runner._file_sha256(conclusion_path),
        raw_metrics_sha256=runner._file_sha256(raw_path),
    )
    package_config.update(
        {
            "registered_source_config_path": str(registered),
            "expected_source_panel_receipt_file_sha256": package.receipt_file_sha256,
            "expected_conclusion_file_sha256": package.conclusion_file_sha256,
            "expected_raw_metrics_sha256": package.raw_metrics_sha256,
            "expected_receipt_payload_sha256": package.receipt_payload_sha256,
            "expected_registered_config_sha256": "5" * 64,
            "expected_registered_config_file_sha256": runner._file_sha256(registered),
            "expected_source_contract_sha256": "6" * 64,
        }
    )
    monkeypatch.setattr(runner, "_load_meta_source", lambda value: meta)
    monkeypatch.setattr(runner, "load_source_panel_package", lambda *a, **k: package)
    monkeypatch.setattr(
        runner, "confirmatory_source_from_package", lambda value: source
    )
    if tampered_field == "receipt_metadata":
        receipt["profile"] = "post_hoc_sensitivity"
    else:
        package_config[tampered_field] = "f" * 64
    with pytest.raises(ValueError, match=pattern):
        runner._load_sources(config)


def test_registered_schemas_are_not_accidentally_reused_from_exp28() -> None:
    assert SOURCE_RECEIPT_SCHEMA.startswith("exp29_confirmatory")
    assert FIT_RECEIPT_SCHEMA.startswith("exp29_confirmatory")
    assert DECISION_RECEIPT_SCHEMA.startswith("exp29_confirmatory")
    assert INFERENCE_STATUS == "preregistered_confirmatory"
