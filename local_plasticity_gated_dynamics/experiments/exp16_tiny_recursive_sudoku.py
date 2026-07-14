"""Micro-TRM-like Sudoku baseline with a matched single-state comparator.

This additive experiment never modifies or initializes the local-learning
models.  It tests a much narrower computational question: whether alternating
updates of answer and latent states help when parameters, initialization,
training arrays/order, optimizer budget, and nominal shared-core calls match.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.stats import wilcoxon

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (  # noqa: E402
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from experiments.exp13_structured_reasoning import _load_dataset  # noqa: E402
from src.baselines.tiny_recursive import (  # noqa: E402
    TinyRecursiveBaseline,
    TinyRecursiveConfig,
    TinyRecursiveTrainingConfig,
    fit_tiny_recursive,
    parameter_count,
    predict_tiny_recursive,
    state_dict_sha256,
)
from src.data.tiny_reasoning_data import (  # noqa: E402
    augment_sudoku_training,
    public_sudoku_test_inputs,
    split_sudoku_training_tasks,
)
from src.utils.artifacts import ExperimentRun  # noqa: E402
from src.utils.reproducibility import derive_seed  # noqa: E402


CONDITIONS = {
    "micro_trm_bptt": "trm_like",
    "single_state_core_call_matched": "single_state_core_call_matched",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "torch",
    "matplotlib",
    "statsmodels",
)
FREEZE_REQUIRED_EVIDENCE_STAGES = frozenset({"frozen_confirmation", "retry_pilot"})


def _group_bootstrap(
    values: np.ndarray,
    groups: tuple[str, ...],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(groups) or not len(values):
        raise ValueError("group bootstrap inputs must align and be non-empty")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    unique_groups = tuple(sorted(set(groups)))
    macro = np.asarray(
        [float(np.mean(values[np.asarray(groups) == group])) for group in unique_groups]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(macro), size=(n_bootstrap, len(macro)), endpoint=False
    )
    draws = np.mean(macro[indices], axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(np.mean(macro)), float(low), float(high)


def _paired_comparison(
    candidate: list[dict[str, object]],
    reference: list[dict[str, object]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    candidate_by_task = {str(row["task_id"]): row for row in candidate}
    reference_by_task = {str(row["task_id"]): row for row in reference}
    if not candidate_by_task or set(candidate_by_task) != set(reference_by_task):
        raise ValueError("paired conditions must contain the identical test tasks")
    group_differences: dict[str, list[float]] = {}
    public_fingerprints_matched = True
    for task_id in sorted(candidate_by_task):
        left = candidate_by_task[task_id]
        right = reference_by_task[task_id]
        group = str(left["source_group"])
        if group != str(right["source_group"]):
            raise ValueError("source groups differ across paired conditions")
        public_fingerprints_matched &= str(left["public_fingerprint"]) == str(
            right["public_fingerprint"]
        )
        group_differences.setdefault(group, []).append(
            float(bool(left["exact"])) - float(bool(right["exact"]))
        )
    groups = tuple(sorted(group_differences))
    differences = np.asarray(
        [float(np.mean(group_differences[group])) for group in groups]
    )
    estimate, low, high = _group_bootstrap(
        differences,
        groups,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    nonzero = int(np.count_nonzero(differences))
    p_value = (
        1.0
        if nonzero == 0
        else float(
            wilcoxon(
                differences,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    )
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "wilcoxon_p": p_value,
        "wilcoxon_p_holm": p_value,
        "n_independent_source_groups": len(groups),
        "n_nonzero_source_groups": nonzero,
        "test_panel_fingerprints_matched": public_fingerprints_matched,
    }


def _device(value: object) -> str:
    requested = str(value)
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def _configure_strict_torch_determinism(device: str) -> dict[str, object]:
    """Use a deterministic attention backend instead of warning and continuing."""

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    attention_backend = "cpu_default"
    if device.startswith("cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        attention_backend = "cuda_math_sdp"
    return {
        "strict_deterministic_algorithms": True,
        "attention_backend": attention_backend,
    }


def _architecture(config: Mapping[str, Any], *, mode: str) -> TinyRecursiveConfig:
    model = dict(config.get("model", {}))
    return TinyRecursiveConfig(
        seq_len=81,
        vocab_size=10,
        hidden_size=int(model.get("hidden_size", 64)),
        num_heads=int(model.get("num_heads", 4)),
        layers=int(model.get("layers", 1)),
        expansion=float(model.get("expansion", 2.0)),
        high_cycles=int(model.get("high_cycles", 2)),
        low_cycles=int(model.get("low_cycles", 2)),
        supervision_steps=int(model.get("supervision_steps", 2)),
        mode=mode,  # type: ignore[arg-type]
    )


def _training_config(
    config: Mapping[str, Any], *, resolve_device: bool = True
) -> TinyRecursiveTrainingConfig:
    training = dict(config.get("training", {}))
    requested_device = str(training.get("device", "auto"))
    return TinyRecursiveTrainingConfig(
        epochs=int(training.get("epochs", 20)),
        batch_size=int(training.get("batch_size", 16)),
        learning_rate=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        grad_clip=float(training.get("grad_clip", 1.0)),
        device=_device(requested_device) if resolve_device else requested_device,
        loss_scope=str(training.get("loss_scope", "blank_only")),  # type: ignore[arg-type]
        checkpoint_metric=str(  # type: ignore[arg-type]
            training.get("checkpoint_metric", "validation_loss")
        ),
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registered_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the semantic input config while excluding its machine-local path."""

    payload = {
        key: value for key, value in dict(config).items() if key != "config_path"
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def calibration_candidate_sha256(config: Mapping[str, Any]) -> str:
    """Hash the effective candidate and its train/validation/data contract."""

    data = dict(config.get("data", {}))
    if config.get("synthetic_fixture") is not None:
        dataset_contract: dict[str, object] = {
            "profile": str(config.get("profile", "unspecified")),
            "synthetic_fixture": dict(config["synthetic_fixture"]),
        }
    else:
        dataset_contract = {
            "profile": str(config.get("profile", "unspecified")),
        }
        dataset_contract.update(
            {
                key: data.get(key)
                for key in (
                    "dataset_name",
                    "revision",
                    "manifest_sha256",
                    "license",
                    "license_status",
                    "test_split_role",
                )
            }
        )
    payload = {
        "architecture": asdict(_architecture(config, mode="trm_like")),
        "training": asdict(_training_config(config, resolve_device=False)),
        "augmentations_per_task": int(config.get("augmentations_per_task", 0)),
        "validation_fraction": float(config.get("validation_fraction", 0.2)),
        "dataset_contract": dataset_contract,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def calibration_code_sha256() -> str:
    """Bind calibration, fitting, data-capability, and freeze implementations."""

    relative_paths = (
        "experiments/common.py",
        "experiments/exp13_structured_reasoning.py",
        "experiments/exp16_tiny_recursive_sudoku.py",
        "experiments/exp17_tiny_recursive_calibration.py",
        "scripts/summarize_exp17_tiny_recursive_calibration.py",
        "scripts/prepare_exp13_public_benchmarks.py",
        "src/baselines/tiny_recursive.py",
        "src/data/structured_protocol.py",
        "src/data/sudoku_tasks.py",
        "src/data/tiny_reasoning_data.py",
        "src/utils/artifacts.py",
        "src/utils/reproducibility.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = PROJECT_ROOT / relative
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def calibration_environment_sha256() -> str:
    """Bind the software environment used for calibration and confirmation."""

    packages: dict[str, str | None] = {}
    for distribution in CALIBRATION_DISTRIBUTIONS:
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = None
    payload = {"python": sys.version, "packages": packages}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    repository = PROJECT_ROOT.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit or None, bool(status.strip())


def _validate_calibration_freeze(
    config: Mapping[str, Any], *, confirmation_seed: int
) -> dict[str, object]:
    evidence_stage = str(config.get("evidence_stage", ""))
    if (
        evidence_stage in FREEZE_REQUIRED_EVIDENCE_STAGES
        and config.get("require_calibration_freeze") is not True
    ):
        raise ValueError(f"{evidence_stage} requires a calibration freeze")
    if config.get("require_calibration_freeze") is not True:
        return {"required": False, "validated": False}
    real_dataset = config.get("synthetic_fixture") is None
    if real_dataset and str(config.get("profile")) != "formal":
        raise ValueError("real-data confirmation requires formal data validation")
    freeze = config.get("calibration_freeze")
    if not isinstance(freeze, Mapping):
        raise ValueError("confirmation requires a calibration_freeze mapping")
    raw_path = freeze.get("freeze_decision_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("calibration freeze_decision_path is required")
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_decision_sha256 = _file_sha256(path)
    if actual_decision_sha256 != freeze.get("freeze_decision_sha256"):
        raise ValueError("calibration freeze decision SHA-256 mismatch")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decision, Mapping):
        raise ValueError("calibration freeze decision must be a JSON object")
    required_flags = {
        "status": "frozen_validation_only",
        "all_freeze_gates_passed": True,
        "enough_seeds": True,
        "all_runs_clean": True,
        "all_candidates_complete": True,
        "test_data_used_for_fit_or_selection": False,
        "test_prediction_array_requested": False,
        "public_test_prediction_adapter_called": False,
        "hidden_target_scorer_called": False,
        "confirmation_test_still_required": True,
    }
    if any(decision.get(key) != value for key, value in required_flags.items()):
        raise ValueError("calibration freeze decision gates are not satisfied")
    if real_dataset and decision.get("formal_data_validation_required") is not True:
        raise ValueError("calibration did not require formal data validation")
    if (
        decision.get("require_clean_git") is True
        and decision.get("all_git_clean") is not True
    ):
        raise ValueError("calibration freeze decision is not git-clean")
    selected_candidate = freeze.get("selected_candidate")
    if selected_candidate != decision.get("selected_candidate"):
        raise ValueError("calibration selected candidate mismatch")
    effective_sha256 = calibration_candidate_sha256(config)
    if (
        effective_sha256 != freeze.get("selected_candidate_config_sha256")
        or effective_sha256 != decision.get("selected_candidate_config_sha256")
    ):
        raise ValueError("confirmation config does not match the frozen candidate")
    selection_seeds = tuple(int(value) for value in freeze.get("selection_seeds", ()))
    if list(selection_seeds) != decision.get("submitted_seeds"):
        raise ValueError("calibration selection seed receipt mismatch")
    if confirmation_seed in selection_seeds:
        raise ValueError("confirmation seed must be disjoint from calibration seeds")
    current_commit, current_dirty = _git_state()
    current_code_sha256 = calibration_code_sha256()
    current_environment_sha256 = calibration_environment_sha256()
    if current_code_sha256 != decision.get("calibration_code_sha256"):
        raise ValueError("confirmation code differs from frozen calibration code")
    if current_environment_sha256 != decision.get(
        "calibration_environment_sha256"
    ):
        raise ValueError(
            "confirmation software environment differs from frozen calibration"
        )
    if decision.get("require_clean_git") is True and current_dirty is not False:
        raise ValueError("confirmation requires a clean git worktree")
    return {
        "required": True,
        "validated": True,
        "freeze_decision_path": str(path),
        "freeze_decision_sha256": actual_decision_sha256,
        "selected_candidate": selected_candidate,
        "selected_candidate_config_sha256": effective_sha256,
        "selection_seeds": list(selection_seeds),
        "confirmation_seed": confirmation_seed,
        "git_commit": current_commit,
        "git_dirty": current_dirty,
        "calibration_git_commit": decision.get("git_commit"),
        "calibration_code_sha256": current_code_sha256,
        "calibration_environment_sha256": current_environment_sha256,
        "test_prediction_array_requested_during_selection": False,
        "hidden_target_scorer_called_during_selection": False,
    }


def run_seed(config: Mapping[str, Any], seed: int, results_root: str | Path) -> Path:
    initialize_seed(seed)
    if str(config.get("family", "sudoku")).lower() != "sudoku":
        raise ValueError("Exp16 currently supports only Sudoku")
    conditions = tuple(config.get("conditions", CONDITIONS))
    if not conditions or not set(conditions).issubset(CONDITIONS):
        raise ValueError("unknown Exp16 condition")
    registered = config.get("registered_comparison")
    if (
        not isinstance(registered, Mapping)
        or registered.get("name") != "micro_trm_minus_single_state_core_call_matched"
        or registered.get("candidate") != "micro_trm_bptt"
        or registered.get("reference") != "single_state_core_call_matched"
        or not set(CONDITIONS).issubset(conditions)
    ):
        raise ValueError("Exp16 requires its frozen paired comparison contract")
    run_config = {
        **dict(config),
        "training_algorithm": "bptt_tiny_recursive_baseline",
        "used_autograd": True,
        "used_bptt": True,
        "eligible_for_local_initialization": False,
        "claim_scope": "computational_baseline_only",
        "official_hrm_reproduction": False,
        "official_trm_reproduction": False,
        "strict_deterministic_algorithms_requested": True,
        "registered_config_sha256": registered_config_sha256(config),
    }
    with ExperimentRun(
        "exp16_tiny_recursive_sudoku", seed, run_config, results_root=results_root
    ) as run:
        (run.path / "fit_receipts.json").write_text("{}\n", encoding="utf-8")
        run.register_conditions(
            [
                {
                    "condition": condition,
                    "task_family": "sudoku",
                    "reasoning_mode": CONDITIONS[condition],
                }
                for condition in conditions
            ]
        )
        try:
            freeze_receipt = _validate_calibration_freeze(
                config, confirmation_seed=seed
            )
            (run.path / "calibration_freeze_receipt.json").write_text(
                json.dumps(freeze_receipt, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as error:
            for condition in conditions:
                run.mark_condition_failure(
                    error,
                    condition=condition,
                    task_family="sudoku",
                    stage="calibration_freeze",
                )
            return run.path
        try:
            dataset, fixture_only, provenance = _load_dataset(dict(config), run.path)
            split_counts = {
                split: len(dataset.for_split(split))
                for split in ("train", "validation", "test")
            }
            if split_counts["train"] < 2 or split_counts["test"] < 1:
                raise ValueError("Exp16 requires train and held-out test Sudoku tasks")
            split_seed = derive_seed(seed, "exp16", "inner_group_split")
            training, validation = split_sudoku_training_tasks(
                dataset,
                validation_fraction=float(config.get("validation_fraction", 0.2)),
                seed=split_seed,
            )
            augmentation_seed = derive_seed(seed, "exp16", "train_augmentation")
            augmented_training = augment_sudoku_training(
                training,
                augmentations_per_task=int(config.get("augmentations_per_task", 0)),
                seed=augmentation_seed,
            )
            test_inputs, test_tasks = public_sudoku_test_inputs(dataset)
            provenance_payload = {
                **provenance,
                "calibration_freeze": freeze_receipt,
                "fixture_only": fixture_only,
                "split_counts": split_counts,
                "inner_training_task_ids": list(training.task_ids),
                "inner_validation_task_ids": list(validation.task_ids),
                "inner_training_source_groups": list(training.source_groups),
                "inner_validation_source_groups": list(validation.source_groups),
                "inner_training_augmentation_groups": list(
                    training.augmentation_groups
                ),
                "inner_validation_augmentation_groups": list(
                    validation.augmentation_groups
                ),
                "inner_training_content_groups": list(training.content_groups),
                "inner_validation_content_groups": list(validation.content_groups),
                "inner_groups_disjoint": all(
                    set(getattr(training, field)).isdisjoint(getattr(validation, field))
                    for field in (
                        "source_groups",
                        "augmentation_groups",
                        "content_groups",
                    )
                ),
                "augmentation_seed": augmentation_seed,
                "augmentations_per_task": int(config.get("augmentations_per_task", 0)),
                "test_tasks": [
                    {
                        "task_id": task.task_id,
                        "source_group": task.source_group,
                        "public_fingerprint": task.fingerprint,
                    }
                    for task in test_tasks
                ],
                "test_targets_exposed_to_fit": False,
            }
            (run.path / "source_provenance.json").write_text(
                json.dumps(provenance_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as error:
            for condition in conditions:
                run.mark_condition_failure(
                    error,
                    condition=condition,
                    task_family="sudoku",
                    stage="dataset",
                )
            return run.path

        source_manifest_verified = provenance.get("preparation_manifest_status") in {
            "complete",
            "complete_with_exclusions",
        }
        formal_data_eligible = bool(
            str(config.get("profile")) == "formal"
            and not fixture_only
            and provenance.get("license_status") == "verified"
            and source_manifest_verified
            and len(test_tasks) >= int(config.get("minimum_test_tasks", 1))
        )
        model_seed = derive_seed(seed, "exp16", "shared_model_initialization")
        optimizer_seed = derive_seed(seed, "exp16", "shared_optimizer_order")
        task_rows_by_condition: dict[str, list[dict[str, object]]] = {}
        aggregate_by_condition: dict[str, dict[str, object]] = {}
        fit_receipts: dict[str, object] = {}
        try:
            training_options = _training_config(config)
            determinism_receipt = _configure_strict_torch_determinism(
                training_options.device
            )
            checkpoint_root = run.path / "checkpoints"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            for condition in conditions:
                run.mark_condition_failure(
                    error,
                    condition=condition,
                    task_family="sudoku",
                    stage="training_setup",
                )
            return run.path

        for condition in conditions:
            try:
                torch.manual_seed(model_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(model_seed)
                architecture = _architecture(config, mode=CONDITIONS[condition])
                model = TinyRecursiveBaseline(architecture)
                initial_fingerprint = state_dict_sha256(model)
                count = parameter_count(model)
                receipt = fit_tiny_recursive(
                    model,
                    augmented_training.inputs,
                    augmented_training.targets,
                    validation.inputs,
                    validation.targets,
                    training_options,
                    seed=optimizer_seed,
                )
                checkpoint_path = checkpoint_root / f"{condition}.pt"
                torch.save(
                    {
                        "state_dict": {
                            name: value.detach().cpu()
                            for name, value in model.state_dict().items()
                        },
                        "metadata": model.checkpoint_metadata(),
                        "fit_receipt": receipt.to_dict(),
                    },
                    checkpoint_path,
                )
                raw_predictions = predict_tiny_recursive(
                    model,
                    test_inputs,
                    batch_size=training_options.batch_size,
                    device=training_options.device,
                    clamp_visible_tokens=False,
                )
                predictions = raw_predictions.copy()
                visible_tokens = test_inputs > 0
                predictions[visible_tokens] = test_inputs[visible_tokens]
                task_rows: list[dict[str, object]] = []
                for task, prediction, raw_prediction, puzzle in zip(
                    test_tasks,
                    predictions,
                    raw_predictions,
                    test_inputs,
                    strict=True,
                ):
                    score = dict(
                        dataset.target_store.score(task, prediction.reshape(9, 9))
                    )
                    row = {
                        "task_id": task.task_id,
                        "source_group": task.source_group,
                        "public_fingerprint": task.fingerprint,
                        "exact": bool(score.get("exact", False)),
                        "valid_solution": bool(score.get("valid_solution", False)),
                        "blank_cell_accuracy": float(
                            score.get("blank_cell_accuracy", 0.0)
                        ),
                        "full_cell_accuracy": float(
                            score.get("full_cell_accuracy", 0.0)
                        ),
                        "clues_preserved": bool(score.get("clues_preserved", False)),
                        "prediction_provided": bool(
                            score.get("prediction_provided", True)
                        ),
                        "raw_unclamped_clue_accuracy": float(
                            np.mean(raw_prediction[puzzle > 0] == puzzle[puzzle > 0])
                        ),
                    }
                    task_rows.append(row)
                    run.record(
                        row,
                        stage="task_test",
                        condition=condition,
                        task_family="sudoku",
                        statistics_unit="source_group",
                    )
                task_rows_by_condition[condition] = task_rows
                exact = np.asarray([row["exact"] for row in task_rows], dtype=float)
                valid = np.asarray(
                    [row["valid_solution"] for row in task_rows], dtype=float
                )
                blank_accuracy = np.asarray(
                    [row["blank_cell_accuracy"] for row in task_rows], dtype=float
                )
                groups = tuple(str(row["source_group"]) for row in task_rows)
                n_bootstrap = int(config.get("n_bootstrap", 1000))
                accuracy_seed = derive_seed(seed, "exp16", condition, "bootstrap")
                exact_estimate, exact_low, exact_high = _group_bootstrap(
                    exact,
                    groups,
                    n_bootstrap=n_bootstrap,
                    seed=accuracy_seed,
                )
                valid_estimate, valid_low, valid_high = _group_bootstrap(
                    valid,
                    groups,
                    n_bootstrap=n_bootstrap,
                    seed=derive_seed(seed, "exp16", condition, "valid_bootstrap"),
                )
                blank_estimate, blank_low, blank_high = _group_bootstrap(
                    blank_accuracy,
                    groups,
                    n_bootstrap=n_bootstrap,
                    seed=derive_seed(seed, "exp16", condition, "blank_bootstrap"),
                )
                aggregate = {
                    "status": "complete",
                    "n_test_tasks": len(task_rows),
                    "n_independent_source_groups": len(set(groups)),
                    "exact_accuracy": exact_estimate,
                    "exact_accuracy_ci_low": exact_low,
                    "exact_accuracy_ci_high": exact_high,
                    "valid_solution_rate": valid_estimate,
                    "valid_solution_ci_low": valid_low,
                    "valid_solution_ci_high": valid_high,
                    "blank_cell_accuracy": blank_estimate,
                    "blank_cell_accuracy_ci_low": blank_low,
                    "blank_cell_accuracy_ci_high": blank_high,
                    "parameter_count": count,
                    "nominal_core_calls_per_evaluation": architecture.core_calls,
                    "core_calls_per_segment": architecture.core_calls_per_segment,
                    "supervision_steps": architecture.supervision_steps,
                    "optimizer_steps": receipt.optimizer_steps,
                    "fixed_training_budget": receipt.fixed_training_budget,
                    "best_epoch": receipt.best_epoch,
                    "best_validation_loss": receipt.best_validation_loss,
                    "selected_validation_loss": receipt.selected_validation_loss,
                    "best_validation_exact_accuracy": max(
                        receipt.validation_exact_accuracy
                    ),
                    "best_validation_blank_cell_accuracy": max(
                        receipt.validation_blank_cell_accuracy
                    ),
                    "selected_train_blank_cell_accuracy": (
                        receipt.selected_train_blank_cell_accuracy
                    ),
                    "selected_validation_blank_cell_accuracy": (
                        receipt.selected_validation_blank_cell_accuracy
                    ),
                    "loss_scope": receipt.loss_scope,
                    "checkpoint_metric": receipt.checkpoint_metric,
                    "mean_raw_unclamped_clue_accuracy": float(
                        np.mean(
                            [row["raw_unclamped_clue_accuracy"] for row in task_rows]
                        )
                    ),
                    "initialization_sha256": initial_fingerprint,
                    "checkpoint_sha256": receipt.checkpoint_sha256,
                    "checkpoint_file_sha256": _file_sha256(checkpoint_path),
                    "formal_data_eligible": formal_data_eligible,
                    "fixture_only": fixture_only,
                    "claim_scope": "computational_baseline_only",
                    "claim_conclusion": "inconclusive",
                    "claim_reason": "requires_preregistered_multi_seed_formal_aggregate",
                    "training_algorithm": model.training_algorithm,
                    "used_autograd": True,
                    "used_bptt": True,
                    "eligible_for_local_initialization": False,
                    "official_hrm_reproduction": False,
                    "official_trm_reproduction": False,
                    **determinism_receipt,
                }
                aggregate_by_condition[condition] = aggregate
                fit_receipts[condition] = {
                    "architecture": model.checkpoint_metadata(),
                    "initialization_sha256": initial_fingerprint,
                    "checkpoint_path": str(checkpoint_path.relative_to(run.path)),
                    "checkpoint_file_sha256": _file_sha256(checkpoint_path),
                    "determinism": determinism_receipt,
                    **receipt.to_dict(),
                }
                run.record(
                    aggregate,
                    stage="aggregate",
                    condition=condition,
                    task_family="sudoku",
                    statistics_unit="seed",
                )
            except Exception as error:
                run.mark_condition_failure(
                    error,
                    condition=condition,
                    task_family="sudoku",
                    stage="fit_or_test",
                )

        comparison_name = "micro_trm_minus_single_state_core_call_matched"
        if set(CONDITIONS).issubset(task_rows_by_condition):
            comparison_seed = derive_seed(seed, "exp16", comparison_name, "bootstrap")
            comparison = _paired_comparison(
                task_rows_by_condition["micro_trm_bptt"],
                task_rows_by_condition["single_state_core_call_matched"],
                n_bootstrap=int(config.get("n_bootstrap", 1000)),
                seed=comparison_seed,
            )
            left = aggregate_by_condition["micro_trm_bptt"]
            right = aggregate_by_condition["single_state_core_call_matched"]
            matching = {
                "parameter_count_matched": left["parameter_count"]
                == right["parameter_count"],
                "initialization_matched": left["initialization_sha256"]
                == right["initialization_sha256"],
                "optimizer_steps_matched": left["optimizer_steps"]
                == right["optimizer_steps"],
                "nominal_core_calls_matched": left["nominal_core_calls_per_evaluation"]
                == right["nominal_core_calls_per_evaluation"],
                "training_data_matched": fit_receipts["micro_trm_bptt"][
                    "training_data_sha256"
                ]
                == fit_receipts["single_state_core_call_matched"][
                    "training_data_sha256"
                ],
                "validation_data_matched": fit_receipts["micro_trm_bptt"][
                    "validation_data_sha256"
                ]
                == fit_receipts["single_state_core_call_matched"][
                    "validation_data_sha256"
                ],
                "epoch_permutations_matched": fit_receipts["micro_trm_bptt"][
                    "epoch_permutation_sha256"
                ]
                == fit_receipts["single_state_core_call_matched"][
                    "epoch_permutation_sha256"
                ],
                "test_panel_fingerprints_matched": comparison[
                    "test_panel_fingerprints_matched"
                ],
            }
            run.record(
                {
                    "status": "complete",
                    "candidate": "micro_trm_bptt",
                    "reference": "single_state_core_call_matched",
                    "estimand": "paired_exact_accuracy_difference",
                    **comparison,
                    **matching,
                    "all_matching_gates_passed": all(matching.values()),
                    "formal_data_eligible": formal_data_eligible,
                    "claim_scope": "computational_baseline_only",
                    "claim_conclusion": "inconclusive",
                    "claim_reason": "single_seed_run_not_an_independent_seed_aggregate",
                    "used_bptt": True,
                    "eligible_for_local_initialization": False,
                },
                stage="comparison",
                condition=comparison_name,
                task_family="sudoku",
                statistics_unit="seed",
            )
        (run.path / "fit_receipts.json").write_text(
            json.dumps(fit_receipts, indent=2, sort_keys=True), encoding="utf-8"
        )
        return run.path


def main() -> None:
    args = basic_parser(
        "Micro-TRM-like Sudoku baseline audit",
        "configs/smoke/exp16_tiny_recursive_sudoku.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        print(run_seed(config, seed, args.results_root))


if __name__ == "__main__":
    main()
