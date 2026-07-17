"""Exploratory associative-memory axis of the Actuator Matching Principle.

Exp30 is a trend-first mechanism panel, not a strong-baseline benchmark.  It
holds one high-rank Dale-compatible carrier, task tape, scalar readout, and
train-fitted functional RMS budget fixed across conditions.  The registered
memory-demand coordinate mixes a query-time direct cue with a trial-local
key--value retrieval target.  Routing should dominate at low memory demand and
the associative outer-product actuator should dominate at high demand.

The matched condition is an oracle demand rule.  It is not an observation-only
or locally learned selector; a four-family learned selector remains future
work.  No autograd or BPTT is used.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import basic_parser, initialize_seed, load_json_config, seed_list
from src.analysis.associative_actuator_metrics import (
    matched_rms_scale,
    sign_accuracy,
    train_normalized_score,
    train_reference_variance,
)
from src.models.associative_memory_actuator import (
    AssociativeMemoryActuator,
    FrozenCarrierBridge,
)
from src.tasks.actuator_matching import CarrierConfig, make_carrier
from src.tasks.associative_actuator import (
    AssociativeActuatorSplit,
    AssociativeActuatorTaskConfig,
    make_associative_actuator_dataset,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


EXPERIMENT = "exp30_associative_actuator_trend"
PROTOCOL_VERSION = "exp30_trend_v1"
EVIDENCE_SCHEMA_VERSION = "exp30_evidence_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRITICAL_CODE_FILES = (
    "experiments/exp30_associative_actuator_trend.py",
    "src/tasks/associative_actuator.py",
    "src/models/associative_memory_actuator.py",
    "src/analysis/associative_actuator_metrics.py",
    "scripts/summarize_exp30.py",
    "figures/exp30_associative_actuator_trend_plot.py",
)
SINGLE_MODES = ("routing", "low_rank", "associative")
MODES = (
    "frozen",
    "routing",
    "low_rank",
    "associative",
    "associative_shuffled",
    "fixed_best",
    "matched",
    "combined",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    files = {
        name: _file_sha256(PROJECT_ROOT / name) for name in CRITICAL_CODE_FILES
    }
    config_path = Path(str(config.get("config_path", "")))
    config_hash = _file_sha256(config_path) if config_path.is_file() else None
    git: dict[str, object] = {"commit": None, "tree": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        git = {"commit": commit, "tree": tree, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "critical_file_sha256": files,
        "source_config_sha256": config_hash,
        "git": git,
    }


def _task_config(config: Mapping[str, Any]) -> AssociativeActuatorTaskConfig:
    return AssociativeActuatorTaskConfig(**dict(config["task"]))


def _carrier_config(config: Mapping[str, Any]) -> CarrierConfig:
    return CarrierConfig(**dict(config["carrier"]))


def _mu_values(config: Mapping[str, Any]) -> tuple[float, ...]:
    values = tuple(float(value) for value in config["memory_demand_values"])
    if len(values) < 4 or len(set(values)) != len(values):
        raise ValueError("memory_demand_values must contain at least four unique cells")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("memory_demand_values must lie in [0, 1]")
    if values != tuple(sorted(values)):
        raise ValueError("memory_demand_values must be strictly increasing")
    if min(values) != 0.0 or max(values) != 1.0:
        raise ValueError("memory_demand_values must include both endpoints")
    return values


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("config protocol_version does not match Exp30")
    _task_config(config)
    carrier = _carrier_config(config)
    if carrier.n_outputs != 1:
        raise ValueError("Exp30 requires a one-dimensional carrier readout")
    _mu_values(config)
    seeds = seed_list(config["seeds"])
    if not seeds:
        raise ValueError("Exp30 requires at least one seed")
    if bool(config.get("used_autograd", False)) or bool(config.get("used_bptt", False)):
        raise ValueError("Exp30 cannot use autograd or BPTT")


def _planned_conditions(config: Mapping[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "memory_demand": mu,
            "condition": mode,
            "actuator_mode": mode,
            "protocol_version": PROTOCOL_VERSION,
        }
        for mu in _mu_values(config)
        for mode in MODES
    ]


def _raw_controls(
    split: AssociativeActuatorSplit,
    actuator: AssociativeMemoryActuator,
) -> dict[str, np.ndarray]:
    return {
        "routing": np.asarray(split.direct_cues, dtype=np.float64),
        "low_rank": np.asarray(
            actuator.compressive_retrieval(split), dtype=np.float64
        ),
        "associative": np.asarray(actuator.retrieve(split), dtype=np.float64),
        "associative_shuffled": np.asarray(
            actuator.retrieve_shuffled(split), dtype=np.float64
        ),
    }


def _scaled_single_predictions(
    train: AssociativeActuatorSplit,
    test: AssociativeActuatorSplit,
    *,
    mu: float,
    train_raw: Mapping[str, np.ndarray],
    test_raw: Mapping[str, np.ndarray],
    bridge: FrozenCarrierBridge,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    reference = train.noiseless_target(mu)
    train_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for mode in ("routing", "low_rank", "associative", "associative_shuffled"):
        scale = matched_rms_scale(reference, train_raw[mode])
        scales[mode] = scale
        train_predictions[mode] = bridge.transmit(scale * train_raw[mode])
        test_predictions[mode] = bridge.transmit(scale * test_raw[mode])
    train_predictions["combined"] = bridge.transmit(
        np.sqrt(1.0 - mu) * train_raw["routing"]
        + np.sqrt(mu) * train_raw["associative"]
    )
    test_predictions["combined"] = bridge.transmit(
        np.sqrt(1.0 - mu) * test_raw["routing"]
        + np.sqrt(mu) * test_raw["associative"]
    )
    return train_predictions, test_predictions, scales


def _dominant_actuator(mu: float, *, tie_tolerance: float) -> str | None:
    routing_weight = np.sqrt(1.0 - mu)
    memory_weight = np.sqrt(mu)
    if abs(routing_weight - memory_weight) <= tie_tolerance:
        return None
    return "routing" if routing_weight > memory_weight else "associative"


def _prediction_for_mode(
    mode: str,
    *,
    split: str,
    mu: float,
    fixed_best: str,
    matched: str,
    train_predictions: Mapping[str, np.ndarray],
    test_predictions: Mapping[str, np.ndarray],
    train_split: AssociativeActuatorSplit,
    test_split: AssociativeActuatorSplit,
    bridge: FrozenCarrierBridge,
) -> np.ndarray:
    predictions = train_predictions if split == "train" else test_predictions
    selected_split = train_split if split == "train" else test_split
    if mode == "frozen":
        return bridge.transmit(np.zeros(selected_split.n_trials, dtype=np.float64))
    if mode in predictions:
        return np.asarray(predictions[mode], dtype=np.float64)
    if mode == "fixed_best":
        return np.asarray(predictions[fixed_best], dtype=np.float64)
    if mode == "matched":
        return np.asarray(predictions[matched], dtype=np.float64)
    raise ValueError(f"unsupported Exp30 mode: {mode}")


def run_seed(
    config: dict[str, Any],
    seed: int,
    results_root: str | Path,
    *,
    run_label: str | None = None,
) -> Path:
    """Run one fully paired exploratory seed and retain every condition."""

    _validate_config(config)
    registered_seeds = seed_list(config["seeds"])
    if seed not in registered_seeds:
        raise ValueError("seed is not registered by the selected Exp30 config")
    initialize_seed(seed)
    evidence_provenance = _source_provenance(config)
    if config.get("profile") == "formal" and evidence_provenance["git"].get(
        "dirty"
    ) is not False:
        raise RuntimeError("formal Exp30 requires a clean, Git-bound worktree")
    dataset = make_associative_actuator_dataset(_task_config(config), seed)
    carrier = make_carrier(
        _carrier_config(config), derive_seed(seed, EXPERIMENT, "carrier")
    )
    bridge = FrozenCarrierBridge.from_carrier(carrier)
    model_options = dict(config.get("model", {}))
    actuator = AssociativeMemoryActuator.random(
        key_dim=dataset.config.key_dim,
        seed=derive_seed(seed, EXPERIMENT, "actuator-dictionary"),
        compression_decay=float(model_options.get("compression_decay", 0.98)),
        distractor_gain=float(model_options.get("distractor_gain", 0.08)),
    )
    train_raw = _raw_controls(dataset.train, actuator)
    test_raw = _raw_controls(dataset.test, actuator)
    max_retrieval_error = float(
        np.max(np.abs(train_raw["associative"] - dataset.train.retrieval_targets))
    )
    if max_retrieval_error > 1e-12:
        raise RuntimeError("associative memory failed its exact train retrieval audit")

    cells: dict[float, dict[str, Any]] = {}
    for mu in _mu_values(config):
        train_predictions, test_predictions, scales = _scaled_single_predictions(
            dataset.train,
            dataset.test,
            mu=mu,
            train_raw=train_raw,
            test_raw=test_raw,
            bridge=bridge,
        )
        train_target = dataset.train.target(mu)
        test_target = dataset.test.target(mu)
        train_variance = train_reference_variance(train_target)
        train_scores = {
            mode: train_normalized_score(
                train_target,
                train_predictions[mode],
                train_variance=train_variance,
            )
            for mode in SINGLE_MODES
        }
        test_scores = {
            mode: train_normalized_score(
                test_target,
                test_predictions[mode],
                train_variance=train_variance,
            )
            for mode in SINGLE_MODES
        }
        cells[mu] = {
            "train_predictions": train_predictions,
            "test_predictions": test_predictions,
            "scales": scales,
            "train_target": train_target,
            "test_target": test_target,
            "train_variance": train_variance,
            "train_scores": train_scores,
            "test_scores": test_scores,
        }
    macro_train_scores = {
        mode: float(np.mean([cells[mu]["train_scores"][mode] for mu in cells]))
        for mode in SINGLE_MODES
    }
    fixed_best = max(SINGLE_MODES, key=lambda mode: (macro_train_scores[mode], mode))
    tie_tolerance = float(config.get("analysis", {}).get("tie_tolerance", 1e-12))
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("analysis.tie_tolerance must be finite and non-negative")

    associative_budget = actuator.write_budget(dataset.train)
    shuffled_budget = actuator.write_budget(dataset.train, shuffled=True)
    write_budget_equal = bool(
        associative_budget.mean_l1 == shuffled_budget.mean_l1
        and associative_budget.mean_l2 == shuffled_budget.mean_l2
    )
    if not write_budget_equal:
        raise RuntimeError("associative and shuffled write budgets differ")

    run_config = {
        key: value for key, value in config.items() if key != "config_path"
    }
    run_config["evidence_provenance"] = evidence_provenance
    with ExperimentRun(
        EXPERIMENT,
        seed,
        run_config,
        results_root=results_root,
        run_label=run_label,
    ) as run:
        run.register_conditions(_planned_conditions(config))
        for mu in _mu_values(config):
            cell = cells[mu]
            dominant = _dominant_actuator(mu, tie_tolerance=tie_tolerance)
            matched = dominant if dominant is not None else fixed_best
            best_test_single = max(
                SINGLE_MODES,
                key=lambda mode: (cell["test_scores"][mode], mode),
            )
            matched_test_prediction = _prediction_for_mode(
                "matched",
                split="test",
                mu=mu,
                fixed_best=fixed_best,
                matched=matched,
                train_predictions=cell["train_predictions"],
                test_predictions=cell["test_predictions"],
                train_split=dataset.train,
                test_split=dataset.test,
                bridge=bridge,
            )
            fixed_test_prediction = _prediction_for_mode(
                "fixed_best",
                split="test",
                mu=mu,
                fixed_best=fixed_best,
                matched=matched,
                train_predictions=cell["train_predictions"],
                test_predictions=cell["test_predictions"],
                train_split=dataset.train,
                test_split=dataset.test,
                bridge=bridge,
            )
            paired_matched_minus_fixed = train_normalized_score(
                cell["test_target"],
                matched_test_prediction,
                train_variance=cell["train_variance"],
            ) - train_normalized_score(
                cell["test_target"],
                fixed_test_prediction,
                train_variance=cell["train_variance"],
            )
            for mode in MODES:
                dimensions = {
                    "memory_demand": mu,
                    "condition": mode,
                    "actuator_mode": mode,
                    "protocol_version": PROTOCOL_VERSION,
                }
                try:
                    train_prediction = _prediction_for_mode(
                        mode,
                        split="train",
                        mu=mu,
                        fixed_best=fixed_best,
                        matched=matched,
                        train_predictions=cell["train_predictions"],
                        test_predictions=cell["test_predictions"],
                        train_split=dataset.train,
                        test_split=dataset.test,
                        bridge=bridge,
                    )
                    test_prediction = _prediction_for_mode(
                        mode,
                        split="test",
                        mu=mu,
                        fixed_best=fixed_best,
                        matched=matched,
                        train_predictions=cell["train_predictions"],
                        test_predictions=cell["test_predictions"],
                        train_split=dataset.train,
                        test_split=dataset.test,
                        bridge=bridge,
                    )
                    test_score = train_normalized_score(
                        cell["test_target"],
                        test_prediction,
                        train_variance=cell["train_variance"],
                    )
                    target_rms = float(
                        np.sqrt(np.mean(dataset.train.noiseless_target(mu) ** 2))
                    )
                    prediction_rms = float(np.sqrt(np.mean(train_prediction**2)))
                    budget_applicable = mode != "frozen"
                    budget_error = (
                        abs(prediction_rms - target_rms) / target_rms
                        if budget_applicable
                        else 0.0
                    )
                    metrics = {
                        "status": "complete",
                        "profile": config["profile"],
                        "statistics_unit": "seed",
                        "split_unit": "block",
                        "time_points_randomly_split": False,
                        "training_algorithm": config["training_algorithm"],
                        "used_autograd": False,
                        "used_bptt": False,
                        "fixed_high_rank_carrier": True,
                        "carrier_shared_across_modes_and_demands": True,
                        "motif_dictionary_shared_across_demands": True,
                        "readout_shared_across_modes": True,
                        "control_gain_train_fitted_per_mode_and_demand": True,
                        "functional_budget_scope": "query_output_rms_only",
                        "write_energy_budget_matched_across_all_modes": False,
                        "target_noise_paired_across_modes": True,
                        "trial_order_paired_across_modes": True,
                        "train_split_fingerprint": dataset.train.fingerprint,
                        "test_split_fingerprint": dataset.test.fingerprint,
                        "carrier_fingerprint": carrier.fingerprint,
                        "carrier_rank": int(np.linalg.matrix_rank(carrier.a0)),
                        "carrier_n_neurons": carrier.config.n_neurons,
                        "carrier_spectral_radius": carrier.spectral_radius,
                        "carrier_bridge_reconstruction_error": bridge.reconstruction_error,
                        "carrier_bridge_is_identity_calibrated": True,
                        "carrier_dynamics_contribute_to_task_solution": False,
                        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                        "run_git_commit": evidence_provenance["git"]["commit"],
                        "run_git_tree": evidence_provenance["git"]["tree"],
                        "run_git_dirty": evidence_provenance["git"]["dirty"],
                        "memory_key_dim": dataset.config.key_dim,
                        "memory_pairs": dataset.config.n_pairs,
                        "delay": dataset.config.delay,
                        "target_noise_std": dataset.config.target_noise_std,
                        "task_routing_weight": float(np.sqrt(1.0 - mu)),
                        "task_memory_weight": float(np.sqrt(mu)),
                        "task_dominant_actuator": dominant,
                        "matched_selected_actuator": matched,
                        "fixed_best_actuator": fixed_best,
                        "best_test_single_actuator": best_test_single,
                        "matched_selector_correct": matched == best_test_single,
                        "selector_uses_explicit_demand_descriptor": True,
                        "selector_is_observation_only": False,
                        "selector_is_learned": False,
                        "combined_composes_actuator_outputs": True,
                        "combined_oracle_target_access": False,
                        "train_normalized_score": train_normalized_score(
                            cell["train_target"],
                            train_prediction,
                            train_variance=cell["train_variance"],
                        ),
                        "test_normalized_score": test_score,
                        "test_sign_accuracy": sign_accuracy(
                            cell["test_target"], test_prediction
                        ),
                        "test_mse": float(
                            np.mean((cell["test_target"] - test_prediction) ** 2)
                        ),
                        "train_reference_variance": cell["train_variance"],
                        "train_functional_rms": prediction_rms,
                        "target_functional_rms": target_rms,
                        "functional_budget_applicable": budget_applicable,
                        "functional_budget_relative_error": budget_error,
                        "functional_budget_valid": (not budget_applicable)
                        or budget_error <= 1e-10,
                        "paired_matched_minus_fixed_score": paired_matched_minus_fixed,
                        "associative_write_l1": associative_budget.mean_l1,
                        "associative_write_l2": associative_budget.mean_l2,
                        "shuffled_write_l1": shuffled_budget.mean_l1,
                        "shuffled_write_l2": shuffled_budget.mean_l2,
                        "associative_shuffled_write_budget_equal": write_budget_equal,
                        "associative_update_rank": associative_budget.max_update_rank,
                        "exact_associative_retrieval_error": max_retrieval_error,
                        "parameter_updates_to_carrier": 0,
                    }
                    if not metrics["functional_budget_valid"]:
                        metrics["failure_reason"] = "functional RMS budget mismatch"
                        run.record_failed_condition(metrics, **dimensions)
                    else:
                        run.record(metrics, **dimensions)
                except Exception as error:
                    run.mark_condition_failure(error, **dimensions)
        return run.path


def _selected_seeds(config: dict[str, Any], override: str | None) -> Iterable[int]:
    return seed_list(override if override is not None else config["seeds"])


def main() -> None:
    parser = basic_parser(
        "Exp30 exploratory associative-actuator matching trend",
        "configs/smoke/exp30_associative_actuator_trend.json",
    )
    parser.add_argument(
        "--run-label",
        help="path-safe label shared by every seed in one exploratory panel",
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    for seed in _selected_seeds(config, args.seeds):
        path = run_seed(
            config,
            seed,
            args.results_root,
            run_label=args.run_label,
        )
        print(path)


if __name__ == "__main__":
    main()
