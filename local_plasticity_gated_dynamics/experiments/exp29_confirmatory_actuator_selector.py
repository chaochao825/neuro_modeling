"""One-shot fixed-meta selector test on the untouched Exp29 source package.

The normalizer, fixed-best policy, local three-factor selector, and GRU-BPTT
baseline are fitted exactly once on Exp26 seeds 0--29 discovery/validation
rows.  They are then frozen and evaluated on every registered heldout cell in
Exp29 seeds 60--89.  Infeasible active choices receive the preregistered
same-cell frozen utility.  No Exp29 row enters preprocessing or fitting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import initialize_seed, load_json_config
from experiments.exp26_actuator_phase_diagram import (
    canonical_config_sha256,
    git_identity,
)
from experiments.exp29_confirmatory_source_panel import EXPECTED_SEEDS
from scripts.package_exp29_confirmatory_source_panel import (
    SourcePanelPackage,
    load_source_panel_package,
)
from src.data.actuator_selector_dataset import (
    CANDIDATE_MODES,
    LOCKED_EXP26_META_SEEDS,
    FrozenSelectorMetaTrainingSet,
    SelectorGeneratorSpec,
    build_frozen_selector_meta_training,
    build_three_step_cues,
    load_exp26_selector_source,
)
from src.data.exp29_feasibility_selector_dataset import (
    ConfirmatorySelectorFold,
    ORACLE_MODES,
    build_confirmatory_selector_folds,
    confirmatory_source_from_package,
)
from src.models.actuator_selector import GRUSelectorBaseline
from src.plasticity.selector_three_factor import LocalThreeFactorSelector
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp29_confirmatory_actuator_selector"
PROFILE = "confirmatory"
PROTOCOL_VERSION = "exp29_confirmatory_selector_v1_fixed_meta"
REQUIRED_RUN_LABEL = "exp29-confirmatory-selector-v1"
INFERENCE_STATUS = "preregistered_confirmatory"
CONFIRMATORY_ELIGIBLE = True
SELECTORS = ("oracle", "gru_bptt", "local_three_factor", "fixed_best")
DECISION_MODES = ORACLE_MODES
EXPECTED_META_SEEDS = LOCKED_EXP26_META_SEEDS
EXPECTED_EVALUATION_SEEDS = EXPECTED_SEEDS
TRAINING_SCOPE = "exp26_meta_seeds_0_29_discovery_validation_only"
TEST_SCOPE = "exp29_seeds_60_89_all_registered_heldout_cells"
INFERENCE_SCOPE = "one_shot_fixed_meta_confirmatory_test"
SOURCE_RECEIPT_SCHEMA = "exp29_confirmatory_selector_sources_v1"
FIT_RECEIPT_SCHEMA = "exp29_confirmatory_frozen_selector_fit_v1"
DECISION_RECEIPT_SCHEMA = "exp29_confirmatory_selector_decisions_v1"
REGISTERED_ANALYSIS_CONTRACT_SHA256 = (
    "c680985c2d23c0b230be185b7f28ceb8a41e0377429f6fbd085c24a87e379e69"
)
ANALYSIS_CONTRACT_SCHEMA = "exp29_confirmatory_selector_analysis_contract_v1"
MATERIALIZATION_SENTINEL = "__EXP29_IMMUTABLE_PACKAGE_MATERIALIZATION__"

_META_SOURCE_REGISTRY = {
    "raw_metrics_path": (
        "results/exp26_actuator_matching_formal_v2_e08beaf/formal/raw_metrics.csv.gz"
    ),
    "conclusion_path": (
        "results/exp26_actuator_matching_formal_v2_e08beaf/formal/conclusion.json"
    ),
    "expected_raw_metrics_sha256": (
        "b3ef5e22c241f832b1fd50254f87e3890ec45057bfeda3a784cbd218623a1193"
    ),
    "expected_conclusion_sha256": (
        "2038127ac875f9faae94b305343415b8fb3a794f9ea032f017401e432fa9d40f"
    ),
    "expected_config_sha256": (
        "07ad3f16d9de6b5906155d95f215e9434e478ca992fd023adfabcd21a0005ecf"
    ),
    "expected_manifest_sha256": (
        "a1c17a1e88c731f6678760865cf51d7236ae771bf839645c401e5cff8798ebfa"
    ),
    "required_profile": "formal",
    "required_conclusion": "support",
}

_PACKAGE_HASH_FIELDS = (
    "expected_source_panel_receipt_file_sha256",
    "expected_conclusion_file_sha256",
    "expected_raw_metrics_sha256",
    "expected_receipt_payload_sha256",
    "expected_registered_config_sha256",
    "expected_registered_config_file_sha256",
    "expected_source_contract_sha256",
)

_PACKAGE_METADATA = {
    "expected_package_schema_version": "exp29_confirmatory_source_package_v1",
    "expected_source_protocol_version": "exp29_confirmatory_source_v1",
    "expected_evidence_schema_version": "exp29_confirmatory_source_evidence_v1",
    "expected_experiment": "exp29_confirmatory_source_panel",
    "expected_profile": "confirmatory_test",
    "expected_evidence_role": "confirmatory_test_source_only",
    "expected_source_conclusion": "inconclusive",
    "expected_standalone_inference_performed": False,
    "expected_standalone_inference_permitted": False,
    "expected_source_panel_valid": True,
}

_MATERIALIZATION_PACKAGE_FIELDS = (
    "package_dir",
    "expected_source_panel_receipt_file_sha256",
    "expected_conclusion_file_sha256",
    "expected_raw_metrics_sha256",
    "expected_receipt_payload_sha256",
)

_IMPLEMENTATION_BINDING = {
    "schema_version": "exp29_selector_implementation_binding_v1",
    "runner_path": "experiments/exp29_confirmatory_actuator_selector.py",
    "summarizer_path": "scripts/summarize_exp29_selector.py",
}

_CONTRACT_LITERAL = re.compile(
    r"(?ms)^(REGISTERED_ANALYSIS_CONTRACT_SHA256\s*=\s*\(\s*)"
    r'"[0-9a-f]{64}"(\s*\))$'
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or value.startswith("INSERT_"):
        raise ValueError(f"{name} must be a non-placeholder path")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def _payload_sha256(value: object) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def analysis_contract_payload(config: Mapping[str, Any]) -> dict[str, object]:
    """Normalize only the future immutable-package materialization fields."""

    normalized = _jsonable(config)
    if not isinstance(normalized, dict):
        raise TypeError("Exp29 analysis config must normalize to a mapping")
    for key in ("config_path", "seed", "run_label", "evidence_provenance"):
        normalized.pop(key, None)
    normalized["confirmatory_readiness"] = MATERIALIZATION_SENTINEL
    package = normalized.get("confirmatory_source_package")
    if not isinstance(package, dict):
        raise ValueError("Exp29 analysis contract lacks source package binding")
    for key in _MATERIALIZATION_PACKAGE_FIELDS:
        if key not in package:
            raise ValueError(f"Exp29 materialization field is missing: {key}")
        package[key] = MATERIALIZATION_SENTINEL
    return {
        "schema_version": ANALYSIS_CONTRACT_SCHEMA,
        "config": normalized,
    }


def analysis_contract_sha256(config: Mapping[str, Any]) -> str:
    return _payload_sha256(analysis_contract_payload(config))


def normalized_source_sha256(path: str | Path) -> str:
    """Hash UTF-8/LF source while ignoring only the contract digest literal."""

    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(
            f"cannot read registered implementation {source_path}"
        ) from error
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_CONTRACT_LITERAL.finditer(normalized))
    if len(matches) != 1:
        raise ValueError(
            f"registered implementation must contain one contract literal: {source_path}"
        )
    normalized = _CONTRACT_LITERAL.sub(rf'\1"{MATERIALIZATION_SENTINEL}"\2', normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validate_implementation_binding(config: Mapping[str, Any]) -> None:
    binding = config.get("implementation_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("Exp29 analysis contract lacks implementation binding")
    for key, expected in _IMPLEMENTATION_BINDING.items():
        if binding.get(key) != expected:
            raise ValueError(f"Exp29 implementation binding mismatch: {key}")
    for path_key, hash_key in (
        ("runner_path", "runner_normalized_sha256"),
        ("summarizer_path", "summarizer_normalized_sha256"),
    ):
        expected_hash = _sha256(binding.get(hash_key), name=hash_key)
        path = _resolve_path(binding.get(path_key), name=path_key)
        if normalized_source_sha256(path) != expected_hash:
            raise ValueError(
                f"Exp29 registered implementation hash mismatch: {path_key}"
            )


def _self_hash(payload: Mapping[str, object]) -> dict[str, object]:
    normalized = _jsonable(payload)
    if not isinstance(normalized, dict):
        raise TypeError("receipt payload must normalize to a mapping")
    return {**normalized, "receipt_sha256": _payload_sha256(normalized)}


def _array_fingerprint(label: str, *arrays: object) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _identity_fingerprint(
    label: str, seeds: object, generator_ids: Sequence[str], *arrays: object
) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    seed_array = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))
    digest.update(seed_array.tobytes())
    for identifier in generator_ids:
        encoded = identifier.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _registered_heldout_generators(
    config: Mapping[str, Any],
) -> tuple[SelectorGeneratorSpec, ...]:
    values = config.get("registered_heldout_generators")
    if not isinstance(values, list):
        raise ValueError("Exp29 requires a registered heldout generator schema")
    result: list[SelectorGeneratorSpec] = []
    required = {
        "generator_id",
        "alpha",
        "transition_rank",
        "input_rank",
        "delay",
        "noise_std",
    }
    for value in values:
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("Exp29 heldout generator entry is malformed")
        result.append(
            SelectorGeneratorSpec(
                generator_id=str(value["generator_id"]),
                generator_split="heldout",
                alpha=float(value["alpha"]),
                transition_rank=int(value["transition_rank"]),
                input_rank=int(value["input_rank"]),
                delay=float(value["delay"]),
                noise_std=float(value["noise_std"]),
            )
        )
    identifiers = tuple(item.generator_id for item in result)
    if len(result) != 44 or len(set(identifiers)) != 44:
        raise ValueError("Exp29 must register exactly 44 unique heldout generators")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError("Exp29 heldout registry must be identifier sorted")
    return tuple(result)


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("profile") != PROFILE or config.get("dev_only") is not False:
        raise ValueError("Exp29 selector must use the confirmatory profile")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp29 selector protocol mismatch")
    if config.get("confirmatory_readiness") != "ready":
        raise ValueError(
            "Exp29 selector registration is not runnable until the immutable "
            "package hashes are bound and confirmatory_readiness is 'ready'"
        )
    if analysis_contract_sha256(config) != REGISTERED_ANALYSIS_CONTRACT_SHA256:
        raise ValueError("Exp29 registered analysis-contract SHA-256 mismatch")
    _validate_implementation_binding(config)
    if (
        config.get("required_run_label") != REQUIRED_RUN_LABEL
        or config.get("inference_status") != INFERENCE_STATUS
        or config.get("confirmatory_eligible") is not True
    ):
        raise ValueError("Exp29 selector inference identity mismatch")
    if config.get("training_algorithm") != (
        "single_fixed_meta_selector_fit_then_one_shot_confirmatory_test"
    ):
        raise ValueError("Exp29 selector training algorithm mismatch")
    if config.get("used_autograd") is not False or config.get("used_bptt") is not False:
        raise ValueError("Exp29 local main model must prohibit autograd and BPTT")
    if tuple(config.get("meta_training_seeds", ())) != EXPECTED_META_SEEDS:
        raise ValueError("Exp29 meta-training seeds must be exactly 0--29")
    if tuple(config.get("evaluation_seeds", ())) != EXPECTED_EVALUATION_SEEDS:
        raise ValueError("Exp29 evaluation seeds must be exactly 60--89")
    if set(config["meta_training_seeds"]) & set(config["evaluation_seeds"]):
        raise ValueError("Exp29 meta and evaluation seeds overlap")
    if tuple(config.get("dictionary", ())) != CANDIDATE_MODES:
        raise ValueError("Exp29 actuator dictionary mismatch")
    if tuple(config.get("selectors", ())) != SELECTORS:
        raise ValueError("Exp29 selector registry mismatch")
    if tuple(config.get("decision_modes", ())) != DECISION_MODES:
        raise ValueError("Exp29 decision-mode registry mismatch")
    if (
        _exact_nonnegative_int(
            config.get("selector_fit_seed"), name="selector_fit_seed"
        )
        != 2801
    ):
        raise ValueError("Exp29 selector_fit_seed must equal 2801")
    _registered_heldout_generators(config)
    if tuple(config.get("feature_columns", ())) != (
        "chi",
        "state_demand",
        "input_demand",
        "transition_rank",
        "input_rank",
        "delay",
        "noise_std",
    ):
        raise ValueError("Exp29 selector features mismatch")
    if config.get("feature_transform") != {
        "cues": ["demand", "ranks", "timing_noise"],
        "raw_to_transformed": {
            "chi": "identity",
            "state_demand": "log",
            "input_demand": "log",
            "transition_rank": "log2",
            "input_rank": "log2",
            "delay": "scaled",
            "noise_std": "log",
        },
        "include_bias": True,
        "fit_scope": TRAINING_SCOPE,
    }:
        raise ValueError("Exp29 feature transform is not preregistered")
    if config.get("local_selector") != {
        "epochs": 200,
        "learning_rate": 0.05,
        "temperature": 1.0,
        "teacher_temperature": 0.05,
        "l2": 0.0001,
        "eligibility_decay": 0.8,
        "belief_retention": 0.8,
    }:
        raise ValueError("Exp29 local-selector hyperparameters mismatch")
    if config.get("gru_selector") != {
        "hidden_dim": 8,
        "epochs": 200,
        "learning_rate": 0.02,
        "weight_decay": 0.0001,
        "teacher_temperature": 0.05,
        "device": "cpu",
        "deterministic": True,
    }:
        raise ValueError("Exp29 GRU hyperparameters mismatch")
    if config.get("evaluation") != {
        "train_generator_split": "discovery",
        "train_endpoint": "validation_balanced_accuracy",
        "test_generator_split": "heldout",
        "test_endpoint": "test_balanced_accuracy_with_frozen_fallback",
        "statistics_unit": "evaluation_seed",
        "primary_scope": "unconditional_all_registered_heldout_cells",
        "tie_break_order": list(DECISION_MODES),
        "local_oracle_gain_fraction_threshold": 0.8,
        "expected_heldout_generators_per_seed": 44,
        "matched_budget_support_requires_feasible_active_row": True,
        "unexpected_failure_conclusion": "invalid",
    }:
        raise ValueError("Exp29 evaluation contract mismatch")
    analysis = config.get("analysis")
    if analysis != {
        "bootstrap_samples": 20000,
        "permutation_samples": 100000,
        "confidence": 0.95,
        "statistics_seed": 2901,
        "multiple_comparison_correction": "holm",
        "alpha": 0.05,
    }:
        raise ValueError("Exp29 analysis contract mismatch")
    if isinstance(analysis, Mapping) and "force_inconclusive" in analysis:
        raise ValueError("Exp29 confirmatory analysis cannot force inconclusive")
    meta = config.get("source_exp26_meta")
    if not isinstance(meta, Mapping):
        raise ValueError("Exp29 requires the frozen Exp26 meta source")
    for key, expected in _META_SOURCE_REGISTRY.items():
        if meta.get(key) != expected:
            raise ValueError(f"Exp29 meta source registry mismatch: {key}")
    _resolve_path(meta.get("raw_metrics_path"), name="meta raw_metrics_path")
    _resolve_path(meta.get("conclusion_path"), name="meta conclusion_path")
    package = config.get("confirmatory_source_package")
    if not isinstance(package, Mapping):
        raise ValueError("Exp29 requires an immutable source package")
    _resolve_path(package.get("package_dir"), name="Exp29 package_dir")
    _resolve_path(
        package.get("registered_source_config_path"),
        name="Exp29 registered source config path",
    )
    for key in _PACKAGE_HASH_FIELDS:
        _sha256(package.get(key), name=f"Exp29 package {key}")
    for key, expected in _PACKAGE_METADATA.items():
        if package.get(key) != expected:
            raise ValueError(f"Exp29 package metadata mismatch: {key}")


def _construct_registered(cls: type, options: Mapping[str, object]) -> object:
    signature = inspect.signature(cls)
    unknown = set(options) - set(signature.parameters)
    if unknown:
        raise ValueError(f"unregistered {cls.__name__} options: {sorted(unknown)}")
    return cls(**dict(options))


def _receipt_value(receipt: object, name: str) -> object:
    if isinstance(receipt, Mapping):
        value = receipt.get(name)
    else:
        value = getattr(receipt, name, None)
    if value is None:
        raise ValueError(f"selector fit receipt lacks {name}")
    return value


def _receipt_costs(receipt: object) -> tuple[float, float]:
    l1 = float(_receipt_value(receipt, "cumulative_update_l1"))
    l2 = float(_receipt_value(receipt, "cumulative_update_l2"))
    if not np.isfinite([l1, l2]).all() or l1 < 0.0 or l2 < 0.0:
        raise ValueError("selector update costs are invalid")
    return l1, l2


def _load_meta_source(config: Mapping[str, Any]) -> FrozenSelectorMetaTrainingSet:
    source_config = config["source_exp26_meta"]
    raw_path = _resolve_path(source_config["raw_metrics_path"], name="meta raw path")
    conclusion_path = _resolve_path(
        source_config["conclusion_path"], name="meta conclusion path"
    )
    if _file_sha256(conclusion_path) != source_config["expected_conclusion_sha256"]:
        raise ValueError("Exp29 meta conclusion file SHA-256 mismatch")
    source = load_exp26_selector_source(
        raw_path,
        conclusion_path,
        expected_profile="formal",
        expected_raw_sha256=source_config["expected_raw_metrics_sha256"],
        require_support=True,
    )
    if (
        source.config_sha256 != source_config["expected_config_sha256"]
        or source.manifest_sha256 != source_config["expected_manifest_sha256"]
        or source.unique_seeds != EXPECTED_META_SEEDS
    ):
        raise ValueError("Exp29 meta source identity mismatch")
    return build_frozen_selector_meta_training(source)


def _package_metadata(receipt: Mapping[str, Any]) -> dict[str, object]:
    coverage = receipt.get("coverage")
    return {
        "expected_package_schema_version": receipt.get("schema_version"),
        "expected_source_protocol_version": receipt.get("protocol_version"),
        "expected_evidence_schema_version": receipt.get("evidence_schema_version"),
        "expected_experiment": receipt.get("experiment"),
        "expected_profile": receipt.get("profile"),
        "expected_evidence_role": receipt.get("evidence_role"),
        "expected_source_conclusion": receipt.get("conclusion"),
        "expected_standalone_inference_performed": receipt.get(
            "standalone_inference_performed"
        ),
        "expected_standalone_inference_permitted": receipt.get(
            "standalone_inference_permitted"
        ),
        "expected_source_panel_valid": (
            coverage.get("source_panel_valid")
            if isinstance(coverage, Mapping)
            else None
        ),
    }


def _load_sources(
    config: Mapping[str, Any],
) -> tuple[
    FrozenSelectorMetaTrainingSet,
    tuple[ConfirmatorySelectorFold, ...],
    dict[str, object],
]:
    meta_training = _load_meta_source(config)
    package_config = config["confirmatory_source_package"]
    package_dir = _resolve_path(package_config["package_dir"], name="Exp29 package_dir")
    receipt_path = package_dir / "source_panel_receipt.json"
    conclusion_path = package_dir / "conclusion.json"
    if (
        _file_sha256(receipt_path)
        != package_config["expected_source_panel_receipt_file_sha256"]
    ):
        raise ValueError("Exp29 package receipt-file SHA-256 mismatch")
    if (
        _file_sha256(conclusion_path)
        != package_config["expected_conclusion_file_sha256"]
    ):
        raise ValueError("Exp29 package conclusion-file SHA-256 mismatch")
    registered_config = _resolve_path(
        package_config["registered_source_config_path"],
        name="Exp29 registered source config",
    )
    if (
        _file_sha256(registered_config)
        != package_config["expected_registered_config_file_sha256"]
    ):
        raise ValueError("Exp29 registered source config file SHA-256 mismatch")
    package = load_source_panel_package(
        package_dir,
        require_complete=True,
        config_path=registered_config,
    )
    if not isinstance(package, SourcePanelPackage):
        raise TypeError("Exp29 loader did not return SourcePanelPackage")
    receipt = package.receipt
    bound_values = {
        "expected_source_panel_receipt_file_sha256": package.receipt_file_sha256,
        "expected_conclusion_file_sha256": package.conclusion_file_sha256,
        "expected_raw_metrics_sha256": package.raw_metrics_sha256,
        "expected_receipt_payload_sha256": package.receipt_payload_sha256,
        "expected_registered_config_sha256": receipt.get("registered_config_sha256"),
        "expected_registered_config_file_sha256": receipt.get(
            "registered_config_file_sha256"
        ),
        "expected_source_contract_sha256": receipt.get("source_contract_sha256"),
    }
    for key, observed in bound_values.items():
        if observed != package_config[key]:
            raise ValueError(f"Exp29 immutable package binding mismatch: {key}")
    if _package_metadata(receipt) != _PACKAGE_METADATA:
        raise ValueError("Exp29 immutable package metadata mismatch")
    raw_name = receipt.get("raw_metrics_file")
    if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
        raise ValueError("Exp29 package raw filename is invalid")
    if (
        _file_sha256(package_dir / raw_name)
        != package_config["expected_raw_metrics_sha256"]
    ):
        raise ValueError("Exp29 package raw-file SHA-256 mismatch")
    source = confirmatory_source_from_package(package)
    folds = build_confirmatory_selector_folds(meta_training, source)
    if tuple(fold.test_seed for fold in folds) != EXPECTED_EVALUATION_SEEDS:
        raise ValueError("Exp29 fold seed order mismatch")
    heldout = tuple(
        item
        for item in meta_training.generator_schema
        if item.generator_split == "heldout"
    )
    if heldout != _registered_heldout_generators(config):
        raise ValueError("Exp29 heldout schema differs from registration")
    expected_cells = int(config["evaluation"]["expected_heldout_generators_per_seed"])
    if meta_training.train_validation_utilities.shape[0] != (
        len(EXPECTED_META_SEEDS) * expected_cells
    ):
        raise ValueError("Exp29 meta discovery coverage mismatch")
    if any(len(fold.test_generator_ids) != expected_cells for fold in folds):
        raise ValueError("Exp29 heldout fold coverage mismatch")
    if source.source_manifest_sha256 != meta_training.source_manifest_sha256:
        raise ValueError("Exp29 source and meta manifests differ")
    source_receipt = _self_hash(
        {
            "schema_version": SOURCE_RECEIPT_SCHEMA,
            "meta": {
                "raw_metrics_path": str(
                    _resolve_path(
                        config["source_exp26_meta"]["raw_metrics_path"],
                        name="meta raw path",
                    )
                ),
                "conclusion_path": str(
                    _resolve_path(
                        config["source_exp26_meta"]["conclusion_path"],
                        name="meta conclusion path",
                    )
                ),
                "raw_metrics_sha256": meta_training.source_raw_metrics_sha256,
                "conclusion_sha256": meta_training.source_conclusion_sha256,
                "config_sha256": meta_training.source_config_sha256,
                "manifest_sha256": meta_training.source_manifest_sha256,
                "seeds": list(EXPECTED_META_SEEDS),
            },
            "confirmatory_package": {
                "package_dir": str(package_dir),
                "source_panel_receipt_file_sha256": package.receipt_file_sha256,
                "conclusion_file_sha256": package.conclusion_file_sha256,
                "receipt_payload_sha256": package.receipt_payload_sha256,
                "raw_metrics_sha256": package.raw_metrics_sha256,
                "registered_config_sha256": receipt["registered_config_sha256"],
                "registered_config_file_sha256": receipt[
                    "registered_config_file_sha256"
                ],
                "source_contract_sha256": receipt["source_contract_sha256"],
                "package_schema_version": receipt["schema_version"],
                "source_protocol_version": receipt["protocol_version"],
                "evidence_schema_version": receipt["evidence_schema_version"],
                "run_provenance": receipt["run_provenance"],
                "coverage": receipt["coverage"],
                "evaluation_seeds": list(EXPECTED_EVALUATION_SEEDS),
            },
            "training_scope": TRAINING_SCOPE,
            "test_scope": TEST_SCOPE,
            "inference_scope": INFERENCE_SCOPE,
            "inference_status": INFERENCE_STATUS,
            "confirmatory_eligible": CONFIRMATORY_ELIGIBLE,
        }
    )
    return meta_training, folds, source_receipt


def _flatten_test_folds(
    folds: Sequence[ConfirmatorySelectorFold],
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    seeds = np.concatenate(
        [np.full(len(fold.test_generator_ids), fold.test_seed) for fold in folds]
    )
    identifiers = tuple(
        identifier for fold in folds for identifier in fold.test_generator_ids
    )
    return (
        seeds,
        identifiers,
        np.vstack([fold.test_normalized_features for fold in folds]),
        np.vstack([fold.candidate_feasible for fold in folds]),
        np.vstack([fold.test_deployment_utilities for fold in folds]),
        np.concatenate([fold.frozen_test_utilities for fold in folds]),
        np.concatenate([fold.test_unseen_composition for fold in folds]),
        np.concatenate([fold.oracle(split="test")[0] for fold in folds]),
    )


@dataclass(frozen=True)
class FrozenSelectorFit:
    decision_probabilities: Mapping[str, np.ndarray]
    costs: Mapping[str, tuple[float, float]]
    fit_receipt: Mapping[str, object]
    decision_receipt: Mapping[str, object]


def _model_fit_receipt(
    selector: str, *, options: Mapping[str, object], fit_receipt: object
) -> dict[str, object]:
    fields = asdict(fit_receipt) if is_dataclass(fit_receipt) else vars(fit_receipt)
    payload = {
        "selector": selector,
        "fit_count": 1,
        "options": dict(options),
        "fit_receipt": fields,
        "fit_receipt_fingerprint": _payload_sha256(fields),
    }
    return _self_hash(payload)


def _decision_fingerprint(
    seeds: object, generator_ids: Sequence[str], probabilities: object
) -> str:
    return _identity_fingerprint(
        "exp29-confirmatory-selector-decisions-v1",
        seeds,
        generator_ids,
        probabilities,
    )


def _fit_frozen_selectors(
    meta_training: FrozenSelectorMetaTrainingSet,
    folds: Sequence[ConfirmatorySelectorFold],
    config: Mapping[str, Any],
) -> FrozenSelectorFit:
    fit_seed = _exact_nonnegative_int(
        config["selector_fit_seed"], name="selector_fit_seed"
    )
    train_features = np.asarray(
        meta_training.train_normalized_features, dtype=np.float64
    )
    train_utilities = np.asarray(
        meta_training.train_validation_utilities, dtype=np.float64
    )
    train_cues = build_three_step_cues(train_features)
    (
        evaluation_seeds,
        generator_ids,
        test_features,
        _,
        _,
        _,
        _,
        oracle_indices,
    ) = _flatten_test_folds(folds)
    test_cues = build_three_step_cues(test_features)
    input_dim = int(train_cues.shape[-1])

    local_options = dict(config["local_selector"])
    local_options.update(
        input_dim=input_dim,
        shuffle_seed=derive_seed(fit_seed, "exp29", "local_selector"),
    )
    local = _construct_registered(LocalThreeFactorSelector, local_options)
    local_fit = local.fit(train_cues, train_utilities)
    if (
        _receipt_value(local_fit, "used_autograd") is not False
        or _receipt_value(local_fit, "used_bptt") is not False
    ):
        raise RuntimeError("Exp29 local selector must prohibit autograd and BPTT")
    local_active = np.asarray(local.predict_proba(test_cues), dtype=np.float64)

    gru_options = dict(config["gru_selector"])
    gru_options.update(
        input_dim=input_dim,
        seed=derive_seed(fit_seed, "exp29", "gru_selector"),
    )
    gru = _construct_registered(GRUSelectorBaseline, gru_options)
    gru_fit = gru.fit(train_cues, train_utilities)
    if (
        _receipt_value(gru_fit, "used_autograd") is not True
        or _receipt_value(gru_fit, "used_bptt") is not True
    ):
        raise RuntimeError("Exp29 GRU baseline must disclose autograd and BPTT")
    gru_active = np.asarray(gru.predict_proba(test_cues), dtype=np.float64)

    expected_shape = (len(evaluation_seeds), len(CANDIDATE_MODES))
    for selector, values in (
        ("local_three_factor", local_active),
        ("gru_bptt", gru_active),
    ):
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"Exp29 {selector} probability shape/value mismatch")
        if not np.allclose(np.sum(values, axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"Exp29 {selector} probabilities are invalid")

    fixed_indices = np.full(
        len(evaluation_seeds), meta_training.fixed_best_index + 1, dtype=np.int64
    )
    decision_probabilities: dict[str, np.ndarray] = {}
    for selector, active in (
        ("gru_bptt", gru_active),
        ("local_three_factor", local_active),
    ):
        decision_probabilities[selector] = np.column_stack(
            [np.zeros(len(active), dtype=np.float64), active]
        )
    oracle_probabilities = np.zeros((len(evaluation_seeds), len(DECISION_MODES)))
    oracle_probabilities[np.arange(len(evaluation_seeds)), oracle_indices] = 1.0
    fixed_probabilities = np.zeros_like(oracle_probabilities)
    fixed_probabilities[np.arange(len(evaluation_seeds)), fixed_indices] = 1.0
    decision_probabilities = {
        "oracle": oracle_probabilities,
        "gru_bptt": decision_probabilities["gru_bptt"],
        "local_three_factor": decision_probabilities["local_three_factor"],
        "fixed_best": fixed_probabilities,
    }
    costs = {
        "oracle": (0.0, 0.0),
        "gru_bptt": _receipt_costs(gru_fit),
        "local_three_factor": _receipt_costs(local_fit),
        "fixed_best": (0.0, 0.0),
    }
    fit_receipt = _self_hash(
        {
            "schema_version": FIT_RECEIPT_SCHEMA,
            "selector_fit_seed": fit_seed,
            "fit_count": 1,
            "fit_scope": TRAINING_SCOPE,
            "meta_training_seeds": list(EXPECTED_META_SEEDS),
            "n_training_examples": int(train_utilities.shape[0]),
            "training_identity_fingerprint": _identity_fingerprint(
                "exp29-confirmatory-meta-training-v1",
                meta_training.train_seeds,
                meta_training.train_generator_ids,
                meta_training.train_raw_features,
                train_features,
                train_utilities,
            ),
            "normalizer": {
                "algorithm": "meta_discovery_only_standardization",
                "fit_count": 1,
                "fit_scope": TRAINING_SCOPE,
                "n_fit_samples": meta_training.normalizer.n_fit_samples,
                "mean": meta_training.normalizer.mean,
                "scale": meta_training.normalizer.scale,
                "fit_fingerprint": meta_training.normalizer.fit_fingerprint,
            },
            "fixed_best": {
                "fit_count": 1,
                "mode": meta_training.fixed_best_mode,
                "index": meta_training.fixed_best_index,
                "train_mean_candidate_utilities": np.mean(train_utilities, axis=0),
            },
            "local_three_factor": _model_fit_receipt(
                "local_three_factor", options=local_options, fit_receipt=local_fit
            ),
            "gru_bptt": _model_fit_receipt(
                "gru_bptt", options=gru_options, fit_receipt=gru_fit
            ),
            "confirmatory_rows_used_for_fit": 0,
            "evaluation_seed_count": len(EXPECTED_EVALUATION_SEEDS),
        }
    )
    decisions: dict[str, object] = {}
    for selector in SELECTORS:
        values = decision_probabilities[selector]
        decisions[selector] = _self_hash(
            {
                "selector": selector,
                "evaluation_seeds": evaluation_seeds,
                "generator_ids": list(generator_ids),
                "decision_modes": list(DECISION_MODES),
                "probabilities": values,
                "decision_fingerprint": _decision_fingerprint(
                    evaluation_seeds, generator_ids, values
                ),
            }
        )
    decision_receipt = _self_hash(
        {
            "schema_version": DECISION_RECEIPT_SCHEMA,
            "selector_fit_receipt_sha256": fit_receipt["receipt_sha256"],
            "evaluation_seed_order": list(EXPECTED_EVALUATION_SEEDS),
            "n_evaluation_examples": len(generator_ids),
            "decision_modes": list(DECISION_MODES),
            "selectors": decisions,
        }
    )
    return FrozenSelectorFit(
        decision_probabilities=decision_probabilities,
        costs=costs,
        fit_receipt=fit_receipt,
        decision_receipt=decision_receipt,
    )


def _planned_conditions(config: Mapping[str, Any]) -> list[dict[str, object]]:
    conditions: list[dict[str, object]] = []
    fit_seed = int(config["selector_fit_seed"])
    for seed in EXPECTED_EVALUATION_SEEDS:
        for spec in _registered_heldout_generators(config):
            for selector in SELECTORS:
                conditions.append(
                    {
                        "selector_fit_seed": fit_seed,
                        "evaluation_seed": seed,
                        "source_seed": seed,
                        "generator_split": "heldout",
                        "generator_id": spec.generator_id,
                        "alpha": spec.alpha,
                        "transition_rank": spec.transition_rank,
                        "input_rank": spec.input_rank,
                        "delay": spec.delay,
                        "noise": spec.noise_std,
                        "selector": selector,
                    }
                )
    return conditions


def _selector_flags(selector: str) -> dict[str, bool]:
    return {
        "selector_used_autograd": selector == "gru_bptt",
        "selector_used_bptt": selector == "gru_bptt",
        "local_learning_main_model": selector == "local_three_factor",
        "bptt_baseline_isolated": selector == "gru_bptt",
        "selector_uses_confirmatory_test_for_selection": selector == "oracle",
    }


def _semantic_records(
    meta_training: FrozenSelectorMetaTrainingSet,
    folds: Sequence[ConfirmatorySelectorFold],
    fit: FrozenSelectorFit,
    config: Mapping[str, Any],
    *,
    run_label: str,
    config_sha256: str,
    run_git: Mapping[str, object],
    source_receipt: Mapping[str, object],
    source_receipt_file_sha256: str,
    fit_receipt_file_sha256: str,
    decision_receipt_file_sha256: str,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    (
        evaluation_seeds,
        generator_ids,
        _,
        feasible,
        candidate_utilities,
        frozen_utilities,
        unseen,
        oracle_indices,
    ) = _flatten_test_folds(folds)
    train_means = np.mean(meta_training.train_validation_utilities, axis=0)
    schema = {item.generator_id: item for item in meta_training.generator_schema}
    records: list[tuple[dict[str, object], dict[str, object]]] = []
    fit_seed = int(config["selector_fit_seed"])
    for index, (evaluation_seed, generator_id) in enumerate(
        zip(evaluation_seeds, generator_ids, strict=True)
    ):
        spec = schema[generator_id]
        active_feasible = feasible[index]
        active_utilities = candidate_utilities[index]
        utility_choice_set = np.concatenate(
            [[frozen_utilities[index]], active_utilities]
        )
        oracle_index = int(oracle_indices[index])
        oracle_utility = float(utility_choice_set[oracle_index])
        fixed_index = meta_training.fixed_best_index + 1
        fixed_utility = float(utility_choice_set[fixed_index])
        denominator = oracle_utility - fixed_utility
        for selector in SELECTORS:
            probabilities = np.asarray(fit.decision_probabilities[selector][index])
            selected_index = int(np.argmax(probabilities))
            requested_mode = DECISION_MODES[selected_index]
            if selected_index == 0:
                selected_active_feasible: bool | None = None
                deployed_mode = "frozen"
                fallback = False
                matched_budget_eligible = False
            else:
                selected_active_feasible = bool(active_feasible[selected_index - 1])
                deployed_mode = requested_mode if selected_active_feasible else "frozen"
                fallback = not selected_active_feasible
                matched_budget_eligible = selected_active_feasible
            selected_utility = float(utility_choice_set[selected_index])
            dimensions = {
                "protocol": PROTOCOL_VERSION,
                "run_label": run_label,
                "selector_fit_seed": fit_seed,
                "evaluation_seed": int(evaluation_seed),
                "outer_seed": int(evaluation_seed),
                "source_seed": int(evaluation_seed),
                "generator_split": "heldout",
                "generator_id": generator_id,
                "alpha": spec.alpha,
                "transition_rank": spec.transition_rank,
                "input_rank": spec.input_rank,
                "delay": spec.delay,
                "noise": spec.noise_std,
                "noise_std": spec.noise_std,
                "selector": selector,
                "training_scope": TRAINING_SCOPE,
                "test_scope": TEST_SCOPE,
                "inference_scope": INFERENCE_SCOPE,
                "statistics_unit": "evaluation_seed",
                "exp29_config_sha256": config_sha256,
                "run_git_commit": run_git.get("commit"),
                "run_git_tree": run_git.get("tree"),
                "run_git_dirty": run_git.get("dirty"),
                "source_receipt_sha256": source_receipt["receipt_sha256"],
                "source_receipt_file_sha256": source_receipt_file_sha256,
                "selector_fit_receipt_sha256": fit.fit_receipt["receipt_sha256"],
                "selector_fit_receipt_file_sha256": fit_receipt_file_sha256,
                "decision_receipt_sha256": fit.decision_receipt["receipt_sha256"],
                "decision_receipt_file_sha256": decision_receipt_file_sha256,
            }
            metrics = {
                "status": "complete",
                "profile": PROFILE,
                "dev_only": False,
                "inference_status": INFERENCE_STATUS,
                "confirmatory_eligible": True,
                "requested_mode": requested_mode,
                "mode_selected": requested_mode,
                "deployed_mode": deployed_mode,
                "target_mode": DECISION_MODES[oracle_index],
                "oracle_mode": DECISION_MODES[oracle_index],
                "fixed_best_mode": meta_training.fixed_best_mode,
                "utility": selected_utility,
                "candidate_frozen_utility": float(frozen_utilities[index]),
                "candidate_routing_utility": float(active_utilities[0]),
                "candidate_gain_utility": float(active_utilities[1]),
                "candidate_low_rank_utility": float(active_utilities[2]),
                "candidate_routing_feasible": bool(active_feasible[0]),
                "candidate_gain_feasible": bool(active_feasible[1]),
                "candidate_low_rank_feasible": bool(active_feasible[2]),
                "selected_active_actuator_feasible": selected_active_feasible,
                "deployment_fallback_applied": fallback,
                "matched_budget_support_eligible": matched_budget_eligible,
                "oracle_utility": oracle_utility,
                "fixed_best_utility": fixed_utility,
                "selection_correct": selected_index == oracle_index,
                "regret": oracle_utility - selected_utility,
                "oracle_gain_over_fixed": denominator,
                "oracle_gain_fraction_applicable": denominator > 1e-12,
                "oracle_gain_recovered_fraction": (
                    (selected_utility - fixed_utility) / denominator
                    if denominator > 1e-12
                    else None
                ),
                "beats_fixed": selected_utility > fixed_utility,
                "plasticity_l1": fit.costs[selector][0],
                "plasticity_l2": fit.costs[selector][1],
                "selection_probability_frozen": float(probabilities[0]),
                "selection_probability_routing": float(probabilities[1]),
                "selection_probability_gain": float(probabilities[2]),
                "selection_probability_low_rank": float(probabilities[3]),
                "meta_training_source_seeds": list(EXPECTED_META_SEEDS),
                "training_source_seed_count": len(EXPECTED_META_SEEDS),
                "n_training_examples": int(
                    meta_training.train_validation_utilities.shape[0]
                ),
                "train_mean_candidate_routing_utility": float(train_means[0]),
                "train_mean_candidate_gain_utility": float(train_means[1]),
                "train_mean_candidate_low_rank_utility": float(train_means[2]),
                "evaluation_seed_excluded_from_training": True,
                "outer_seed_excluded_from_training": True,
                "confirmatory_rows_used_for_fit": 0,
                "selector_fit_count": 1,
                "normalization_fit_scope": TRAINING_SCOPE,
                "train_split": "discovery",
                "train_endpoint": "validation_balanced_accuracy",
                "test_endpoint": "test_balanced_accuracy_with_frozen_fallback",
                "strict_unseen_composition": bool(unseen[index]),
                "unconditional_cell_retained": True,
                "primary_endpoint_eligible": True,
                "primary_scope": "unconditional_all_registered_heldout_cells",
                "carrier_frozen": True,
                "actuator_dictionary_frozen": True,
                "readout_trained": False,
                "task_descriptor_cues_privileged": True,
                "selector_teacher_signal": (
                    "full_three_candidate_validation_utility_vector"
                    if selector in {"local_three_factor", "gru_bptt"}
                    else "not_applicable"
                ),
                "selector_teacher_counterfactual": selector
                in {"local_three_factor", "gru_bptt"},
                "selector_teacher_is_scalar_reward": False,
                "update_budget_matched_across_selectors": False,
                **_selector_flags(selector),
            }
            records.append((metrics, dimensions))
    return records


def _failure_metrics(selector: str, reason: str) -> dict[str, object]:
    return {
        "failure_reason": reason,
        "inference_status": INFERENCE_STATUS,
        "confirmatory_eligible": False,
        "unexpected_failure_invalidates_inference": True,
        "requested_mode": None,
        "mode_selected": None,
        "deployed_mode": None,
        "utility": None,
        **_selector_flags(selector),
    }


def _write_receipt(path: Path, receipt: Mapping[str, object]) -> str:
    path.write_text(
        json.dumps(_jsonable(receipt), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return _file_sha256(path)


def _assert_one_shot_unused(results_root: str | Path, fit_seed: int) -> None:
    seed_root = Path(results_root) / "runs" / EXPERIMENT / f"seed_{fit_seed:04d}"
    if seed_root.exists() and any(seed_root.iterdir()):
        raise FileExistsError(
            "Exp29 confirmatory selector already has an attempt; rerun/replacement "
            "is forbidden"
        )


def run_experiment(
    config: dict[str, Any],
    results_root: str | Path,
    *,
    run_label: str,
) -> Path:
    _validate_config(config)
    if run_label != REQUIRED_RUN_LABEL:
        raise ValueError("Exp29 selector requires the registered run_label")
    fit_seed = int(config["selector_fit_seed"])
    _assert_one_shot_unused(results_root, fit_seed)
    plan = _planned_conditions(config)
    run_git = git_identity()
    config_sha = canonical_config_sha256(config)
    initialize_seed(fit_seed)
    provenance = {
        "schema_version": "exp29_confirmatory_selector_evidence_v1",
        "canonical_config_sha256": config_sha,
        "git": run_git,
        "run_label": run_label,
        "inference_scope": INFERENCE_SCOPE,
        "inference_status": INFERENCE_STATUS,
        "confirmatory_eligible": True,
        "one_shot_attempt": True,
    }
    run_config = {**config, "evidence_provenance": provenance}
    with ExperimentRun(
        EXPERIMENT,
        fit_seed,
        run_config,
        results_root=results_root,
        run_label=run_label,
    ) as run:
        run.register_conditions(plan)
        if run_git.get("dirty") is not False:
            error = RuntimeError(
                "Exp29 confirmatory selector requires a clean Git tree"
            )
            for condition in plan:
                run.record_failed_condition(
                    _failure_metrics(str(condition["selector"]), str(error)),
                    **condition,
                )
            raise error
        try:
            meta, folds, source_receipt = _load_sources(config)
            fit = _fit_frozen_selectors(meta, folds, config)
            source_file_sha = _write_receipt(
                run.path / "source_package_receipt.json", source_receipt
            )
            fit_file_sha = _write_receipt(
                run.path / "selector_fit_receipt.json", fit.fit_receipt
            )
            decision_file_sha = _write_receipt(
                run.path / "decision_receipt.json", fit.decision_receipt
            )
            records = _semantic_records(
                meta,
                folds,
                fit,
                config,
                run_label=run_label,
                config_sha256=config_sha,
                run_git=run_git,
                source_receipt=source_receipt,
                source_receipt_file_sha256=source_file_sha,
                fit_receipt_file_sha256=fit_file_sha,
                decision_receipt_file_sha256=decision_file_sha,
            )
            if len(records) != len(plan):
                raise RuntimeError("Exp29 semantic row count differs from plan")
            for metrics, dimensions in records:
                run.record(metrics, **dimensions)
        except Exception as error:
            observed: set[str] = set()
            for line in run.metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                observed.add(
                    _payload_sha256(
                        {
                            key: value.get(key)
                            for key in (
                                "selector_fit_seed",
                                "evaluation_seed",
                                "source_seed",
                                "generator_split",
                                "generator_id",
                                "alpha",
                                "transition_rank",
                                "input_rank",
                                "delay",
                                "noise",
                                "selector",
                            )
                        }
                    )
                )
            for condition in plan:
                if _payload_sha256(condition) in observed:
                    continue
                run.record_failed_condition(
                    _failure_metrics(str(condition["selector"]), str(error)),
                    **condition,
                )
            raise
        end_git = git_identity()
        if any(
            end_git.get(key) != run_git.get(key) for key in ("commit", "tree", "dirty")
        ):
            raise RuntimeError("Exp29 selector Git identity changed during the run")
        return run.path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/formal/exp29_confirmatory_actuator_selector.json",
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args(argv)
    config = load_json_config(args.config)
    print(run_experiment(config, args.results_root, run_label=args.run_label))


if __name__ == "__main__":
    main()
