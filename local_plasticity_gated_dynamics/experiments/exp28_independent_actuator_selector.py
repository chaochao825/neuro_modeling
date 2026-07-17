"""Fixed-meta selector sensitivity on a post-hoc amended Exp26 panel.

The selector normalizer, fixed-best policy, local three-factor model, and GRU
baseline are fitted exactly once on Exp26 seeds 0--29 discovery/validation
rows.  Those frozen objects are then evaluated without refitting on seeds
30--59 heldout/test rows from the separately packaged, ceiling-amended Exp28
source panel.  Because the amendment followed inspection of the original
panel, this analysis is permanently non-confirmatory.

This experiment uses prospective task-demand descriptors.  It is not a
hidden-context or online-belief inference experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import initialize_seed, load_json_config
from experiments.exp26_actuator_phase_diagram import (
    canonical_config_sha256,
    git_identity,
)
from scripts.package_exp28_independent_source_panel import (
    load_source_panel_package,
)
from src.data.actuator_selector_dataset import (
    CANDIDATE_MODES,
    LOCKED_EXP26_INDEPENDENT_TEST_SEEDS,
    LOCKED_EXP26_META_SEEDS,
    FrozenSelectorMetaTrainingSet,
    IndependentSelectorTestFold,
    SelectorGeneratorSpec,
    build_frozen_selector_meta_training,
    build_independent_selector_test_folds,
    build_three_step_cues,
    exp26_selector_source_from_independent_package,
    load_exp26_selector_source,
)
from src.models.actuator_selector import GRUSelectorBaseline
from src.plasticity.selector_three_factor import LocalThreeFactorSelector
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "exp28_post_hoc_amended_actuator_selector_sensitivity"
PROTOCOL_VERSION = "exp28_post_hoc_amended_sensitivity_v1_fixed_meta"
REQUIRED_RUN_LABEL = "exp28-amended-selector-sensitivity-v1"
INFERENCE_STATUS = "post_hoc_amended_sensitivity_non_confirmatory"
CONFIRMATORY_ELIGIBLE = False
SELECTORS = ("oracle", "gru_bptt", "local_three_factor", "fixed_best")
TRAINING_SCOPE = "exp26_meta_seeds_0_29_discovery_validation_only"
TEST_SCOPE = "exp26_independent_seeds_30_59_heldout_test_only"
INFERENCE_SCOPE = "fixed_meta_train_30_post_hoc_amended_test_seeds"
SOURCE_RECEIPT_SCHEMA = "exp28_amended_sensitivity_selector_sources_v1"
FIT_RECEIPT_SCHEMA = "exp28_amended_sensitivity_frozen_selector_fit_v1"
DECISION_RECEIPT_SCHEMA = "exp28_amended_sensitivity_selector_decisions_v1"
EXPECTED_META_SEEDS = LOCKED_EXP26_META_SEEDS
EXPECTED_EVALUATION_SEEDS = LOCKED_EXP26_INDEPENDENT_TEST_SEEDS

_META_SOURCE_REGISTRY = {
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
_HASH_FIELDS = (
    "expected_source_panel_receipt_file_sha256",
    "expected_conclusion_file_sha256",
    "expected_raw_metrics_sha256",
    "expected_receipt_payload_sha256",
    "expected_registered_config_sha256",
    "expected_registered_config_file_sha256",
    "expected_protocol_amendment_sha256",
)
_AMENDED_PACKAGE_METADATA = {
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or value.startswith("INSERT_"):
        raise ValueError(f"{name} must be a registered non-placeholder path")
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
        raise ValueError(f"{name} must be a registered lowercase SHA-256")
    return value


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _jsonable(value: object) -> object:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Exp28 receipts cannot contain non-finite floats")
    return value


def _payload_sha256(value: object) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _self_hash(payload: Mapping[str, object]) -> dict[str, object]:
    value = dict(payload)
    if "receipt_sha256" in value:
        raise ValueError("receipt payload must not predefine receipt_sha256")
    value["receipt_sha256"] = _payload_sha256(value)
    return value


def _array_fingerprint(label: str, *arrays: object) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _identity_fingerprint(
    label: str,
    seeds: Sequence[object],
    generator_ids: Sequence[object],
    *arrays: object,
) -> str:
    identity = json.dumps(
        {
            "seeds": [int(value) for value in seeds],
            "generator_ids": [str(value) for value in generator_ids],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(label.encode("utf-8"))
    digest.update(len(identity).to_bytes(8, "little"))
    digest.update(identity)
    digest.update(_array_fingerprint(label, *arrays).encode("ascii"))
    return digest.hexdigest()


def _registered_heldout_generators(
    config: Mapping[str, Any],
) -> tuple[SelectorGeneratorSpec, ...]:
    value = config.get("registered_heldout_generators")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("Exp28 registered heldout generators must be a sequence")
    required = {
        "generator_id",
        "alpha",
        "transition_rank",
        "input_rank",
        "delay",
        "noise_std",
    }
    result: list[SelectorGeneratorSpec] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError(
                f"Exp28 registered heldout generator {index} has an invalid schema"
            )
        result.append(
            SelectorGeneratorSpec(
                generator_id=item["generator_id"],
                generator_split="heldout",
                alpha=item["alpha"],
                transition_rank=item["transition_rank"],
                input_rank=item["input_rank"],
                delay=item["delay"],
                noise_std=item["noise_std"],
            )
        )
    identifiers = tuple(item.generator_id for item in result)
    if len(result) != 44 or len(set(identifiers)) != len(identifiers):
        raise ValueError("Exp28 must register exactly 44 unique heldout generators")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError(
            "Exp28 heldout generator registry must be sorted by identifier"
        )
    return tuple(result)


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("profile") != "post_hoc_sensitivity"
        or config.get("dev_only") is not False
    ):
        raise ValueError("Exp28 amended selector must be post-hoc sensitivity only")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Exp28 independent selector protocol mismatch")
    if config.get("sensitivity_readiness") != "ready":
        raise ValueError(
            "Exp28 amended-sensitivity template is not runnable until package "
            "hashes are inserted and sensitivity_readiness is set to 'ready'"
        )
    if config.get("inference_status") != INFERENCE_STATUS:
        raise ValueError("Exp28 amended-sensitivity inference status mismatch")
    if config.get("confirmatory_eligible") is not CONFIRMATORY_ELIGIBLE:
        raise ValueError("Exp28 amended sensitivity can never be confirmatory")
    if config.get("required_run_label") != REQUIRED_RUN_LABEL:
        raise ValueError("Exp28 independent selector run-label mismatch")
    if tuple(config.get("dictionary", ())) != CANDIDATE_MODES:
        raise ValueError("Exp28 actuator dictionary/order mismatch")
    if tuple(config.get("selectors", ())) != SELECTORS:
        raise ValueError("Exp28 selector registry/order mismatch")
    if (
        config.get("training_algorithm")
        != "single_fixed_meta_selector_fit_then_post_hoc_amended_test"
    ):
        raise ValueError("Exp28 training algorithm registry mismatch")
    if config.get("used_autograd") is not False or config.get("used_bptt") is not False:
        raise ValueError("Exp28 local main model must prohibit autograd and BPTT")
    if tuple(config.get("meta_training_seeds", ())) != EXPECTED_META_SEEDS:
        raise ValueError("Exp28 meta-training seed registry must be exactly 0--29")
    if tuple(config.get("evaluation_seeds", ())) != EXPECTED_EVALUATION_SEEDS:
        raise ValueError("Exp28 independent evaluation seeds must be exactly 30--59")
    future = config.get("future_confirmatory_reservation")
    if future != {
        "protocol_namespace": ("exp28_fresh_independent_confirmatory_v2_seeds60_89"),
        "status": "reserved_not_executed",
        "evaluation_seeds": list(range(60, 90)),
        "current_analysis_may_use_reserved_seeds": False,
    }:
        raise ValueError("Exp28 future seeds 60--89 reservation mismatch")
    if set(config["evaluation_seeds"]) & set(future["evaluation_seeds"]):
        raise ValueError("current amended and future confirmatory seeds overlap")
    _registered_heldout_generators(config)
    if (
        _exact_nonnegative_int(
            config.get("selector_fit_seed"), name="selector_fit_seed"
        )
        != 2801
    ):
        raise ValueError("Exp28 selector_fit_seed must equal 2801")
    if tuple(config.get("feature_columns", ())) != (
        "chi",
        "state_demand",
        "input_demand",
        "transition_rank",
        "input_rank",
        "delay",
        "noise_std",
    ):
        raise ValueError("Exp28 selector features mismatch")
    transform = config.get("feature_transform")
    expected_transform = {
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
    }
    if transform != expected_transform:
        raise ValueError(
            "Exp28 amended-sensitivity feature transform is not registered"
        )
    expected_local = {
        "epochs": 200,
        "learning_rate": 0.05,
        "temperature": 1.0,
        "teacher_temperature": 0.05,
        "l2": 0.0001,
        "eligibility_decay": 0.8,
        "belief_retention": 0.8,
    }
    if config.get("local_selector") != expected_local:
        raise ValueError("Exp28 local-selector hyperparameters mismatch")
    expected_gru = {
        "hidden_dim": 8,
        "epochs": 200,
        "learning_rate": 0.02,
        "weight_decay": 0.0001,
        "teacher_temperature": 0.05,
        "device": "cpu",
        "deterministic": True,
    }
    if config.get("gru_selector") != expected_gru:
        raise ValueError("Exp28 GRU hyperparameters mismatch")
    evaluation = config.get("evaluation")
    expected_evaluation = {
        "train_generator_split": "discovery",
        "train_endpoint": "validation_balanced_accuracy",
        "test_generator_split": "heldout",
        "test_endpoint": "test_balanced_accuracy",
        "statistics_unit": "post_hoc_amended_evaluation_seed",
        "tie_break_order": list(CANDIDATE_MODES),
        "local_oracle_gain_fraction_threshold": 0.8,
        "expected_heldout_generators_per_seed": 44,
        "expected_strict_unseen_per_seed": 43,
    }
    if evaluation != expected_evaluation:
        raise ValueError("Exp28 evaluation contract mismatch")
    analysis = config.get("analysis")
    if analysis != {
        "bootstrap_samples": 20000,
        "permutation_samples": 100000,
        "confidence": 0.95,
        "statistics_seed": 2801,
        "force_inconclusive": True,
    }:
        raise ValueError("Exp28 analysis contract mismatch")
    meta = config.get("source_exp26_meta")
    if not isinstance(meta, Mapping):
        raise ValueError("Exp28 requires the frozen Exp26 meta source")
    for key, expected in _META_SOURCE_REGISTRY.items():
        if meta.get(key) != expected:
            raise ValueError(f"Exp28 meta source registry mismatch: {key}")
    _resolve_path(meta.get("raw_metrics_path"), name="meta raw_metrics_path")
    _resolve_path(meta.get("conclusion_path"), name="meta conclusion_path")
    package = config.get("independent_source_package")
    if not isinstance(package, Mapping):
        raise ValueError("Exp28 requires an independent source package")
    _resolve_path(package.get("package_dir"), name="independent package_dir")
    _resolve_path(
        package.get("registered_source_config_path"),
        name="amended registered source config path",
    )
    for key in _HASH_FIELDS:
        _sha256(package.get(key), name=f"independent package {key}")
    for key, expected in _AMENDED_PACKAGE_METADATA.items():
        if package.get(key) != expected:
            raise ValueError(f"Exp28 amended package metadata mismatch: {key}")


def _construct_registered(cls: type, options: Mapping[str, object]) -> object:
    signature = inspect.signature(cls)
    unsupported = set(options) - set(signature.parameters)
    if unsupported:
        raise TypeError(
            f"{cls.__name__} lacks registered parameters: {sorted(unsupported)}"
        )
    return cls(**options)


def _receipt_value(receipt: object, name: str) -> object:
    if isinstance(receipt, Mapping):
        return receipt.get(name)
    return getattr(receipt, name, None)


def _receipt_costs(receipt: object) -> tuple[float, float]:
    values = (
        float(_receipt_value(receipt, "cumulative_update_l1")),
        float(_receipt_value(receipt, "cumulative_update_l2")),
    )
    if not np.isfinite(values).all() or min(values) < 0.0:
        raise ValueError("selector update costs must be finite and non-negative")
    return values


def _load_meta_source(config: Mapping[str, Any]) -> FrozenSelectorMetaTrainingSet:
    source_config = config["source_exp26_meta"]
    raw_path = _resolve_path(
        source_config["raw_metrics_path"], name="meta raw_metrics_path"
    )
    conclusion_path = _resolve_path(
        source_config["conclusion_path"], name="meta conclusion_path"
    )
    expected_raw = _sha256(
        source_config["expected_raw_metrics_sha256"], name="meta raw SHA-256"
    )
    expected_conclusion = _sha256(
        source_config["expected_conclusion_sha256"],
        name="meta conclusion SHA-256",
    )
    if _file_sha256(raw_path) != expected_raw:
        raise ValueError("Exp28 meta raw-metrics SHA-256 mismatch")
    if _file_sha256(conclusion_path) != expected_conclusion:
        raise ValueError("Exp28 meta conclusion SHA-256 mismatch")
    source = load_exp26_selector_source(
        raw_path,
        conclusion_path,
        expected_profile="formal",
        expected_raw_sha256=expected_raw,
        require_support=True,
    )
    if (
        source.config_sha256 != source_config["expected_config_sha256"]
        or source.manifest_sha256 != source_config["expected_manifest_sha256"]
        or source.conclusion_sha256 != expected_conclusion
        or source.unique_seeds != EXPECTED_META_SEEDS
    ):
        raise ValueError("Exp28 meta source identity mismatch")
    return build_frozen_selector_meta_training(source)


def _load_sources(
    config: Mapping[str, Any],
) -> tuple[
    FrozenSelectorMetaTrainingSet,
    tuple[IndependentSelectorTestFold, ...],
    dict[str, object],
]:
    meta_training = _load_meta_source(config)
    package_config = config["independent_source_package"]
    package_dir = _resolve_path(
        package_config["package_dir"], name="independent package_dir"
    )
    receipt_path = package_dir / "source_panel_receipt.json"
    conclusion_path = package_dir / "conclusion.json"
    expected_receipt_file = _sha256(
        package_config["expected_source_panel_receipt_file_sha256"],
        name="independent package receipt-file SHA-256",
    )
    expected_conclusion_file = _sha256(
        package_config["expected_conclusion_file_sha256"],
        name="independent package conclusion-file SHA-256",
    )
    if _file_sha256(receipt_path) != expected_receipt_file:
        raise ValueError("Exp28 independent package receipt-file SHA-256 mismatch")
    if _file_sha256(conclusion_path) != expected_conclusion_file:
        raise ValueError("Exp28 independent package conclusion-file SHA-256 mismatch")
    registered_source_config = _resolve_path(
        package_config["registered_source_config_path"],
        name="amended registered source config path",
    )
    if (
        _file_sha256(registered_source_config)
        != package_config["expected_registered_config_file_sha256"]
    ):
        raise ValueError("Exp28 amended registered config file SHA-256 mismatch")
    package = load_source_panel_package(
        package_dir,
        require_complete=True,
        config_path=registered_source_config,
    )
    package_receipt = package.receipt
    for key, expected_key in (
        ("raw_metrics_sha256", "expected_raw_metrics_sha256"),
        ("receipt_payload_sha256", "expected_receipt_payload_sha256"),
        ("registered_config_sha256", "expected_registered_config_sha256"),
        (
            "registered_config_file_sha256",
            "expected_registered_config_file_sha256",
        ),
        ("protocol_amendment_sha256", "expected_protocol_amendment_sha256"),
    ):
        expected = _sha256(package_config[expected_key], name=expected_key)
        if package_receipt.get(key) != expected:
            raise ValueError(f"Exp28 independent package {key} mismatch")
    amendment = package_receipt.get("protocol_amendment")
    if not isinstance(amendment, Mapping):
        raise ValueError("Exp28 amended package lacks protocol amendment metadata")
    amended_metadata = {
        "expected_package_schema_version": package_receipt.get("schema_version"),
        "expected_source_protocol_version": package_receipt.get("protocol_version"),
        "expected_amendment_id": amendment.get("amendment_id"),
        "expected_inference_status": package_receipt.get("inference_status"),
        "expected_confirmatory_independence_restored": package_receipt.get(
            "confirmatory_independence_restored"
        ),
        "expected_previous_max_scale": amendment.get("previous_max_scale"),
        "expected_amended_max_scale": amendment.get("amended_max_scale"),
        "expected_performance_metrics_inspected": amendment.get(
            "performance_metrics_inspected"
        ),
        "expected_performance_metrics_used_for_amendment": amendment.get(
            "performance_metrics_used_for_amendment"
        ),
    }
    if amended_metadata != _AMENDED_PACKAGE_METADATA:
        raise ValueError("Exp28 amended package metadata binding mismatch")
    raw_name = package_receipt.get("raw_metrics_file")
    if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
        raise ValueError("Exp28 independent package raw filename is invalid")
    raw_path = package_dir / raw_name
    if _file_sha256(raw_path) != package_config["expected_raw_metrics_sha256"]:
        raise ValueError("Exp28 independent package raw-file SHA-256 mismatch")
    independent_source = exp26_selector_source_from_independent_package(package)
    folds = build_independent_selector_test_folds(meta_training, independent_source)
    if tuple(fold.test_seed for fold in folds) != EXPECTED_EVALUATION_SEEDS:
        raise ValueError("Exp28 independent fold ordering mismatch")
    expected_heldout = int(config["evaluation"]["expected_heldout_generators_per_seed"])
    expected_unseen = int(config["evaluation"]["expected_strict_unseen_per_seed"])
    observed_heldout = tuple(
        item
        for item in meta_training.generator_schema
        if item.generator_split == "heldout"
    )
    if observed_heldout != _registered_heldout_generators(config):
        raise ValueError("Exp28 source heldout schema differs from preregistration")
    if meta_training.train_validation_utilities.shape[0] != (
        len(EXPECTED_META_SEEDS) * expected_heldout
    ):
        raise ValueError("Exp28 meta discovery training coverage mismatch")
    for fold in folds:
        if (
            len(fold.test_generator_ids) != expected_heldout
            or int(np.sum(fold.test_unseen_composition)) != expected_unseen
        ):
            raise ValueError(
                "Exp28 independent heldout/strict-unseen coverage mismatch"
            )
    source_receipt = _self_hash(
        {
            "schema_version": SOURCE_RECEIPT_SCHEMA,
            "meta": {
                "raw_metrics_path": str(
                    _resolve_path(
                        config["source_exp26_meta"]["raw_metrics_path"],
                        name="meta raw_metrics_path",
                    )
                ),
                "conclusion_path": str(
                    _resolve_path(
                        config["source_exp26_meta"]["conclusion_path"],
                        name="meta conclusion_path",
                    )
                ),
                "raw_metrics_sha256": meta_training.source_raw_metrics_sha256,
                "conclusion_sha256": meta_training.source_conclusion_sha256,
                "config_sha256": meta_training.source_config_sha256,
                "manifest_sha256": meta_training.source_manifest_sha256,
                "seeds": list(EXPECTED_META_SEEDS),
            },
            "independent_package": {
                "package_dir": str(package_dir),
                "source_panel_receipt_file_sha256": expected_receipt_file,
                "conclusion_file_sha256": expected_conclusion_file,
                "receipt_payload_sha256": package_receipt["receipt_payload_sha256"],
                "raw_metrics_sha256": package_receipt["raw_metrics_sha256"],
                "registered_config_sha256": package_receipt["registered_config_sha256"],
                "registered_config_file_sha256": package_receipt[
                    "registered_config_file_sha256"
                ],
                "source_contract_sha256": package_receipt["source_contract_sha256"],
                "package_schema_version": package_receipt["schema_version"],
                "source_protocol_version": package_receipt["protocol_version"],
                "protocol_amendment": amendment,
                "protocol_amendment_sha256": package_receipt[
                    "protocol_amendment_sha256"
                ],
                "inference_status": package_receipt["inference_status"],
                "confirmatory_independence_restored": package_receipt[
                    "confirmatory_independence_restored"
                ],
                "evaluation_seeds": list(EXPECTED_EVALUATION_SEEDS),
            },
            "shared_manifest_sha256": meta_training.source_manifest_sha256,
            "training_scope": TRAINING_SCOPE,
            "test_scope": TEST_SCOPE,
            "inference_status": INFERENCE_STATUS,
            "confirmatory_eligible": CONFIRMATORY_ELIGIBLE,
        }
    )
    return meta_training, folds, source_receipt


def _flatten_test_folds(
    folds: Sequence[IndependentSelectorTestFold],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if tuple(fold.test_seed for fold in folds) != EXPECTED_EVALUATION_SEEDS:
        raise ValueError("Exp28 requires all independent folds in registered order")
    seeds = np.concatenate([fold.test_seeds for fold in folds])
    identifiers = tuple(
        generator_id for fold in folds for generator_id in fold.test_generator_ids
    )
    features = np.vstack([fold.test_normalized_features for fold in folds])
    utilities = np.vstack([fold.test_utilities for fold in folds])
    unseen = np.concatenate([fold.test_unseen_composition for fold in folds])
    overlap = np.concatenate([fold.test_composition_overlap for fold in folds])
    return seeds, identifiers, features, utilities, unseen, overlap


@dataclass(frozen=True)
class FrozenSelectorFit:
    probabilities: Mapping[str, np.ndarray]
    costs: Mapping[str, tuple[float, float]]
    fit_receipt: Mapping[str, object]
    decision_receipt: Mapping[str, object]


def _model_fit_receipt(
    selector: str,
    *,
    options: Mapping[str, object],
    fit_receipt: object,
) -> dict[str, object]:
    return _self_hash(
        {
            "selector": selector,
            "algorithm": (
                "local_three_factor_eligibility_selector"
                if selector == "local_three_factor"
                else "deterministic_cpu_gru_bptt_baseline"
            ),
            "hyperparameters": dict(options),
            "fit_receipt": _jsonable(fit_receipt),
            "used_autograd": selector == "gru_bptt",
            "used_bptt": selector == "gru_bptt",
            "status": "complete",
        }
    )


def _decision_fingerprint(
    evaluation_seeds: Sequence[object],
    generator_ids: Sequence[object],
    probabilities: object,
) -> str:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(generator_ids), len(CANDIDATE_MODES)):
        raise ValueError("Exp28 decision probability shape mismatch")
    return _identity_fingerprint(
        "exp28-amended-sensitivity-selector-decisions-v1",
        evaluation_seeds,
        generator_ids,
        values,
    )


def _fit_frozen_selectors(
    meta_training: FrozenSelectorMetaTrainingSet,
    folds: Sequence[IndependentSelectorTestFold],
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
        test_utilities,
        _,
        _,
    ) = _flatten_test_folds(folds)
    test_cues = build_three_step_cues(test_features)
    input_dim = int(train_cues.shape[-1])

    local_options = dict(config["local_selector"])
    local_options.update(
        input_dim=input_dim,
        shuffle_seed=derive_seed(fit_seed, "exp28", "local_selector"),
    )
    local = _construct_registered(LocalThreeFactorSelector, local_options)
    local_fit = local.fit(train_cues, train_utilities)
    if (
        _receipt_value(local_fit, "used_autograd") is not False
        or _receipt_value(local_fit, "used_bptt") is not False
    ):
        raise RuntimeError("Exp28 local selector must prohibit autograd and BPTT")
    local_probabilities = np.asarray(local.predict_proba(test_cues), dtype=np.float64)

    gru_options = dict(config["gru_selector"])
    gru_options.update(
        input_dim=input_dim,
        seed=derive_seed(fit_seed, "exp28", "gru_selector"),
    )
    gru = _construct_registered(GRUSelectorBaseline, gru_options)
    gru_fit = gru.fit(train_cues, train_utilities)
    if (
        _receipt_value(gru_fit, "used_autograd") is not True
        or _receipt_value(gru_fit, "used_bptt") is not True
    ):
        raise RuntimeError("Exp28 GRU baseline must disclose autograd and BPTT")
    gru_probabilities = np.asarray(gru.predict_proba(test_cues), dtype=np.float64)

    expected_shape = test_utilities.shape
    for selector, values in (
        ("local_three_factor", local_probabilities),
        ("gru_bptt", gru_probabilities),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"Exp28 {selector} probability shape mismatch")
        if not np.isfinite(values).all() or not np.allclose(
            np.sum(values, axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"Exp28 {selector} probabilities are invalid")

    oracle_indices = np.argmax(test_utilities, axis=1)
    fixed_indices = np.full(
        len(test_utilities), meta_training.fixed_best_index, dtype=np.int64
    )
    one_hot_oracle = np.zeros(expected_shape, dtype=np.float64)
    one_hot_oracle[np.arange(len(oracle_indices)), oracle_indices] = 1.0
    one_hot_fixed = np.zeros(expected_shape, dtype=np.float64)
    one_hot_fixed[np.arange(len(fixed_indices)), fixed_indices] = 1.0
    probabilities = {
        "oracle": one_hot_oracle,
        "gru_bptt": gru_probabilities,
        "local_three_factor": local_probabilities,
        "fixed_best": one_hot_fixed,
    }
    costs = {
        "oracle": (0.0, 0.0),
        "gru_bptt": _receipt_costs(gru_fit),
        "local_three_factor": _receipt_costs(local_fit),
        "fixed_best": (0.0, 0.0),
    }
    local_receipt = _model_fit_receipt(
        "local_three_factor", options=local_options, fit_receipt=local_fit
    )
    gru_receipt = _model_fit_receipt(
        "gru_bptt", options=gru_options, fit_receipt=gru_fit
    )
    fit_receipt = _self_hash(
        {
            "schema_version": FIT_RECEIPT_SCHEMA,
            "selector_fit_seed": fit_seed,
            "fit_count": 1,
            "fit_scope": TRAINING_SCOPE,
            "meta_training_seeds": list(EXPECTED_META_SEEDS),
            "n_training_examples": int(train_utilities.shape[0]),
            "training_identity_fingerprint": _identity_fingerprint(
                "exp28-amended-sensitivity-meta-training-v1",
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
            "local_three_factor": local_receipt,
            "gru_bptt": gru_receipt,
            "independent_rows_used_for_fit": 0,
            "evaluation_seed_count": len(EXPECTED_EVALUATION_SEEDS),
        }
    )
    decision_entries: dict[str, object] = {}
    for selector in SELECTORS:
        values = probabilities[selector]
        decision_entries[selector] = _self_hash(
            {
                "selector": selector,
                "evaluation_seeds": evaluation_seeds,
                "generator_ids": list(generator_ids),
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
            "candidate_modes": list(CANDIDATE_MODES),
            "selectors": decision_entries,
        }
    )
    return FrozenSelectorFit(
        probabilities=probabilities,
        costs=costs,
        fit_receipt=fit_receipt,
        decision_receipt=decision_receipt,
    )


def _planned_conditions(config: Mapping[str, Any]) -> list[dict[str, object]]:
    """Build the complete plan from registered schema without source I/O."""

    heldout = _registered_heldout_generators(config)
    fit_seed = int(config["selector_fit_seed"])
    conditions: list[dict[str, object]] = []
    for evaluation_seed in EXPECTED_EVALUATION_SEEDS:
        for item in heldout:
            for selector in SELECTORS:
                conditions.append(
                    {
                        "selector_fit_seed": fit_seed,
                        "evaluation_seed": evaluation_seed,
                        "source_seed": evaluation_seed,
                        "generator_split": "heldout",
                        "generator_id": item.generator_id,
                        "alpha": item.alpha,
                        "transition_rank": item.transition_rank,
                        "input_rank": item.input_rank,
                        "delay": item.delay,
                        "noise": item.noise_std,
                        "selector": selector,
                    }
                )
    return conditions


def _selector_flags(selector: str) -> dict[str, bool]:
    return {
        "used_autograd": selector == "gru_bptt",
        "used_bptt": selector == "gru_bptt",
        "local_learning_main_model": selector == "local_three_factor",
        "bptt_baseline_isolated": selector == "gru_bptt",
        "selector_uses_independent_test_for_selection": selector == "oracle",
    }


def _semantic_records(
    meta_training: FrozenSelectorMetaTrainingSet,
    folds: Sequence[IndependentSelectorTestFold],
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
        test_utilities,
        unseen,
        overlap,
    ) = _flatten_test_folds(folds)
    train_means = np.mean(meta_training.train_validation_utilities, axis=0)
    schema = {item.generator_id: item for item in meta_training.generator_schema}
    oracle_indices = np.argmax(test_utilities, axis=1)
    records: list[tuple[dict[str, object], dict[str, object]]] = []
    fit_seed = int(config["selector_fit_seed"])
    for index, (evaluation_seed, generator_id) in enumerate(
        zip(evaluation_seeds, generator_ids, strict=True)
    ):
        spec = schema[generator_id]
        utilities = test_utilities[index]
        oracle_index = int(oracle_indices[index])
        fixed_index = meta_training.fixed_best_index
        for selector in SELECTORS:
            probabilities = np.asarray(fit.probabilities[selector][index])
            selected_index = int(np.argmax(probabilities))
            selected_utility = float(utilities[selected_index])
            oracle_utility = float(utilities[oracle_index])
            fixed_utility = float(utilities[fixed_index])
            denominator = oracle_utility - fixed_utility
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
                "strict_unseen_composition": bool(unseen[index]),
                "selector": selector,
                "training_scope": TRAINING_SCOPE,
                "test_scope": TEST_SCOPE,
                "inference_scope": INFERENCE_SCOPE,
                "statistics_unit": "post_hoc_amended_evaluation_seed",
                "exp28_config_sha256": config_sha256,
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
                "profile": "post_hoc_sensitivity",
                "dev_only": False,
                "inference_status": INFERENCE_STATUS,
                "confirmatory_eligible": CONFIRMATORY_ELIGIBLE,
                "force_inconclusive": True,
                "mode_selected": CANDIDATE_MODES[selected_index],
                "target_mode": CANDIDATE_MODES[oracle_index],
                "oracle_mode": CANDIDATE_MODES[oracle_index],
                "fixed_best_mode": meta_training.fixed_best_mode,
                "utility": selected_utility,
                "candidate_routing_utility": float(utilities[0]),
                "candidate_gain_utility": float(utilities[1]),
                "candidate_low_rank_utility": float(utilities[2]),
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
                "selection_probability_routing": float(probabilities[0]),
                "selection_probability_gain": float(probabilities[1]),
                "selection_probability_low_rank": float(probabilities[2]),
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
                "independent_rows_used_for_fit": 0,
                "selector_fit_count": 1,
                "normalization_fit_scope": TRAINING_SCOPE,
                "train_split": "discovery",
                "train_endpoint": "validation_balanced_accuracy",
                "test_endpoint": "test_balanced_accuracy",
                "composition_overlap_secondary": bool(overlap[index]),
                "primary_endpoint_eligible": bool(unseen[index]),
                "directional_sensitivity_eligible": bool(unseen[index]),
                "confirmatory_endpoint_eligible": False,
                "carrier_frozen": True,
                "actuator_dictionary_frozen": True,
                "actuator_basis_trained": False,
                "readout_trained": False,
                "hidden_belief_inference_enabled": False,
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
        "mode_selected": None,
        "target_mode": None,
        "oracle_mode": None,
        "fixed_best_mode": None,
        "utility": None,
        "candidate_routing_utility": None,
        "candidate_gain_utility": None,
        "candidate_low_rank_utility": None,
        "oracle_utility": None,
        "fixed_best_utility": None,
        "selection_correct": None,
        "plasticity_l1": None,
        "plasticity_l2": None,
        **_selector_flags(selector),
    }


def _write_receipt(path: Path, receipt: Mapping[str, object]) -> str:
    path.write_text(
        json.dumps(_jsonable(receipt), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return _file_sha256(path)


def run_experiment(
    config: dict[str, Any],
    results_root: str | Path,
    *,
    run_label: str,
) -> Path:
    _validate_config(config)
    if run_label != REQUIRED_RUN_LABEL:
        raise ValueError("Exp28 amended sensitivity requires its registered run_label")
    fit_seed = int(config["selector_fit_seed"])
    plan = _planned_conditions(config)
    run_git = git_identity()
    config_sha = canonical_config_sha256(config)
    initialize_seed(fit_seed)
    provenance = {
        "schema_version": "exp28_amended_sensitivity_selector_evidence_v1",
        "canonical_config_sha256": config_sha,
        "git": run_git,
        "run_label": run_label,
        "inference_scope": INFERENCE_SCOPE,
        "inference_status": INFERENCE_STATUS,
        "confirmatory_eligible": CONFIRMATORY_ELIGIBLE,
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
                "Exp28 amended sensitivity requires a clean Git worktree"
            )
            for condition in plan:
                selector = str(condition["selector"])
                run.record_failed_condition(
                    _failure_metrics(selector, str(error)), **condition
                )
            raise error
        try:
            loaded_meta, folds, source_receipt = _load_sources(config)
            fit = _fit_frozen_selectors(loaded_meta, folds, config)
            source_path = run.path / "source_package_receipt.json"
            fit_path = run.path / "selector_fit_receipt.json"
            decision_path = run.path / "decision_receipt.json"
            source_file_sha = _write_receipt(source_path, source_receipt)
            fit_file_sha = _write_receipt(fit_path, fit.fit_receipt)
            decision_file_sha = _write_receipt(decision_path, fit.decision_receipt)
            records = _semantic_records(
                loaded_meta,
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
                raise RuntimeError("Exp28 semantic row count differs from plan")
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
                selector = str(condition["selector"])
                run.record_failed_condition(
                    _failure_metrics(selector, str(error)), **condition
                )
            raise
        end_git = git_identity()
        if any(
            end_git.get(key) != run_git.get(key) for key in ("commit", "tree", "dirty")
        ):
            raise RuntimeError(
                "Exp28 amended-sensitivity Git identity changed during run"
            )
        return run.path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=("configs/formal/exp28_independent_actuator_selector.json"),
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args(argv)
    config = load_json_config(args.config)
    print(
        run_experiment(
            config,
            args.results_root,
            run_label=args.run_label,
        )
    )


if __name__ == "__main__":
    main()
