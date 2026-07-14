"""Test-score-blind calibration for the micro tiny-recursive Sudoku baseline.

This experiment may inspect supervised train and inner-validation targets. The
dataset adapter constructs its normal opaque capability store, including test
records, but this runner never requests a test prediction array, invokes the
hidden-target scorer, or emits a test metric. Its only purpose is to freeze a
candidate before a separate confirmation experiment is run.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (  # noqa: E402
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from experiments.exp13_structured_reasoning import _load_dataset  # noqa: E402
from experiments.exp16_tiny_recursive_sudoku import (  # noqa: E402
    _architecture,
    _configure_strict_torch_determinism,
    _training_config,
    calibration_candidate_sha256,
    calibration_code_sha256,
    calibration_environment_sha256,
)
from src.baselines.tiny_recursive import (  # noqa: E402
    TinyRecursiveBaseline,
    fit_tiny_recursive,
    parameter_count,
    state_dict_sha256,
)
from src.data.tiny_reasoning_data import (  # noqa: E402
    SupervisedSudokuArrays,
    augment_sudoku_training,
    split_sudoku_training_tasks,
)
from src.utils.artifacts import ExperimentRun  # noqa: E402
from src.utils.reproducibility import derive_seed  # noqa: E402


NO_TEST_ACCESS_EVIDENCE = {
    "test_data_used_for_fit_or_selection": False,
    "test_prediction_array_requested": False,
    "public_test_prediction_adapter_called": False,
    "hidden_target_scorer_called": False,
}


def _semantic_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_configs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = config.get("candidates")
    if not isinstance(raw, Mapping) or len(raw) < 2:
        raise ValueError("Exp17 requires at least two named calibration candidates")
    candidates: dict[str, dict[str, Any]] = {}
    casefold_names: set[str] = set()
    windows_reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    for name, value in raw.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name) is None
            or name in {".", ".."}
            or name.split(".", 1)[0].upper() in windows_reserved
            or name.casefold() in casefold_names
            or not isinstance(value, Mapping)
        ):
            raise ValueError(
                "candidate names must be path-safe identifiers and payloads mappings"
            )
        candidate = dict(value)
        candidate.setdefault("model", dict(config.get("model", {})))
        candidate.setdefault("training", dict(config.get("training", {})))
        candidates[name] = candidate
        casefold_names.add(name.casefold())
    return candidates


def _candidate_payload(
    config: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    model = {**dict(config.get("model", {})), **dict(candidate.get("model", {}))}
    training = {
        **dict(config.get("training", {})),
        **dict(candidate.get("training", {})),
    }
    return {"model": model, "training": training}


def _selection_key(row: Mapping[str, object]) -> tuple[float, float, int, str]:
    return (
        -float(row["selected_validation_blank_cell_accuracy"]),
        -float(row["selected_validation_exact_accuracy"]),
        int(row["parameter_count"]),
        str(row["candidate"]),
    )


def run_seed(config: Mapping[str, Any], seed: int, results_root: str | Path) -> Path:
    """Run candidate calibration using only train and inner-validation data."""

    initialize_seed(seed)
    if str(config.get("family", "sudoku")).lower() != "sudoku":
        raise ValueError("Exp17 currently supports only Sudoku")
    selection_contract = config.get("selection_contract")
    if not isinstance(selection_contract, Mapping) or (
        selection_contract.get("primary_metric")
        != "selected_validation_blank_cell_accuracy"
        or selection_contract.get("test_access_forbidden") is not True
    ):
        raise ValueError("Exp17 requires its test-free validation selection contract")
    candidates = _candidate_configs(config)
    run_config = {
        **dict(config),
        "claim_scope": "calibration_only",
        "test_access_forbidden": True,
        **NO_TEST_ACCESS_EVIDENCE,
        "used_bptt": True,
        "eligible_for_local_initialization": False,
        "semantic_config_sha256": _semantic_sha256(
            {key: value for key, value in dict(config).items() if key != "config_path"}
        ),
        "calibration_code_sha256": calibration_code_sha256(),
        "calibration_environment_sha256": calibration_environment_sha256(),
    }
    with ExperimentRun(
        "exp17_tiny_recursive_calibration", seed, run_config, results_root=results_root
    ) as run:
        (run.path / "fit_receipts.json").write_text("{}\n", encoding="utf-8")
        run.register_conditions(
            [
                {
                    "condition": name,
                    "task_family": "sudoku",
                    "stage": "calibration_candidate",
                }
                for name in candidates
            ]
        )
        try:
            dataset, fixture_only, provenance = _load_dataset(dict(config), run.path)
            split_counts = {
                split: len(dataset.for_split(split))
                for split in ("train", "validation")
            }
            if split_counts["train"] < 2:
                raise ValueError("Exp17 requires at least two training Sudoku tasks")
            split_seed = derive_seed(seed, "exp17", "inner_group_split")
            training, validation = split_sudoku_training_tasks(
                dataset,
                validation_fraction=float(config.get("validation_fraction", 0.2)),
                seed=split_seed,
            )
            provenance_payload = {
                **provenance,
                "fixture_only": fixture_only,
                "split_counts": split_counts,
                "inner_training_task_ids": list(training.task_ids),
                "inner_validation_task_ids": list(validation.task_ids),
                "inner_groups_disjoint": all(
                    set(getattr(training, field)).isdisjoint(
                        getattr(validation, field)
                    )
                    for field in (
                        "source_groups",
                        "augmentation_groups",
                        "content_groups",
                    )
                ),
                "dataset_adapter_loaded_test_records": any(
                    task.split == "test" for task in dataset.tasks
                ),
                **NO_TEST_ACCESS_EVIDENCE,
                "test_targets_remained_opaque_in_target_store": True,
            }
            (run.path / "source_provenance.json").write_text(
                json.dumps(provenance_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            checkpoint_root = run.path / "checkpoints"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            (run.path / "source_provenance.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "failure_type": type(error).__name__,
                        "dataset_adapter_loaded_test_records": None,
                        **NO_TEST_ACCESS_EVIDENCE,
                        "test_targets_remained_opaque_in_target_store": True,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            for name in candidates:
                run.mark_condition_failure(
                    error,
                    condition=name,
                    task_family="sudoku",
                    stage="calibration_dataset",
                    **NO_TEST_ACCESS_EVIDENCE,
                )
            return run.path

        model_seed = derive_seed(seed, "exp17", "shared_model_initialization")
        optimizer_seed = derive_seed(seed, "exp17", "shared_optimizer_order")
        augmented_cache: dict[int, SupervisedSudokuArrays] = {}
        successful_rows: list[dict[str, object]] = []
        receipts: dict[str, object] = {}
        for name, candidate in candidates.items():
            try:
                payload = _candidate_payload(config, candidate)
                augmentations = int(
                    candidate.get(
                        "augmentations_per_task",
                        config.get("augmentations_per_task", 0),
                    )
                )
                if augmentations not in augmented_cache:
                    augmentation_seed = derive_seed(
                        seed, "exp17", "train_augmentation", augmentations
                    )
                    augmented_cache[augmentations] = augment_sudoku_training(
                        training,
                        augmentations_per_task=augmentations,
                        seed=augmentation_seed,
                    )
                augmented_training = augmented_cache[augmentations]
                training_options = _training_config(payload)
                determinism = _configure_strict_torch_determinism(
                    training_options.device
                )
                architecture = _architecture(payload, mode="trm_like")
                candidate_config_sha256 = calibration_candidate_sha256(
                    {
                        **dict(config),
                        "model": payload["model"],
                        "training": payload["training"],
                        "augmentations_per_task": augmentations,
                    }
                )
                torch.manual_seed(model_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(model_seed)
                model = TinyRecursiveBaseline(architecture)
                initialization_sha256 = state_dict_sha256(model)
                receipt = fit_tiny_recursive(
                    model,
                    augmented_training.inputs,
                    augmented_training.targets,
                    validation.inputs,
                    validation.targets,
                    training_options,
                    seed=optimizer_seed,
                )
                checkpoint_path = checkpoint_root / (
                    f"{name}-{candidate_config_sha256[:12]}.pt"
                )
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "metadata": model.checkpoint_metadata(),
                        "fit_receipt": receipt.to_dict(),
                    },
                    checkpoint_path,
                )
                row: dict[str, object] = {
                    "status": "complete",
                    "candidate": name,
                    "candidate_config_sha256": candidate_config_sha256,
                    "augmentation_count": augmentations,
                    "train_source_tasks": len(training.inputs),
                    "train_augmented_examples": len(augmented_training.inputs),
                    "validation_source_tasks": len(validation.inputs),
                    "parameter_count": parameter_count(model),
                    "nominal_core_calls_per_evaluation": architecture.core_calls,
                    "optimizer_steps": receipt.optimizer_steps,
                    "loss_scope": receipt.loss_scope,
                    "checkpoint_metric": receipt.checkpoint_metric,
                    "selected_train_blank_cell_accuracy": (
                        receipt.selected_train_blank_cell_accuracy
                    ),
                    "selected_validation_blank_cell_accuracy": (
                        receipt.selected_validation_blank_cell_accuracy
                    ),
                    "selected_validation_exact_accuracy": (
                        receipt.selected_validation_exact_accuracy
                    ),
                    "best_validation_blank_cell_accuracy": (
                        receipt.best_validation_blank_cell_accuracy
                    ),
                    "best_validation_loss": receipt.best_validation_loss,
                    "selected_validation_loss": receipt.selected_validation_loss,
                    "blank_accuracy_generalization_gap": (
                        receipt.selected_train_blank_cell_accuracy
                        - receipt.selected_validation_blank_cell_accuracy
                    ),
                    "best_epoch": receipt.best_epoch,
                    "initialization_sha256": initialization_sha256,
                    "training_data_sha256": receipt.training_data_sha256,
                    "validation_data_sha256": receipt.validation_data_sha256,
                    "epoch_permutation_sha256": receipt.epoch_permutation_sha256,
                    "checkpoint_sha256": receipt.checkpoint_sha256,
                    "checkpoint_path": str(checkpoint_path.relative_to(run.path)),
                    **NO_TEST_ACCESS_EVIDENCE,
                    "claim_scope": "calibration_only",
                    "claim_conclusion": "inconclusive",
                    "used_bptt": True,
                    "eligible_for_local_initialization": False,
                    **determinism,
                }
                successful_rows.append(row)
                receipts[name] = {
                    "candidate_config": candidate,
                    "resolved_candidate_config": {
                        "model": payload["model"],
                        "training": payload["training"],
                        "augmentations_per_task": augmentations,
                    },
                    "architecture": model.checkpoint_metadata(),
                    **receipt.to_dict(),
                }
                run.record(
                    row,
                    stage="calibration_candidate",
                    condition=name,
                    task_family="sudoku",
                    statistics_unit="seed",
                )
            except Exception as error:
                run.mark_condition_failure(
                    error,
                    condition=name,
                    task_family="sudoku",
                    stage="calibration_fit",
                    **NO_TEST_ACCESS_EVIDENCE,
                )

        if successful_rows:
            selected = min(successful_rows, key=_selection_key)
            run.record(
                {
                    "status": "complete",
                    "selected_candidate": selected["candidate"],
                    "primary_metric": "selected_validation_blank_cell_accuracy",
                    "primary_metric_value": selected[
                        "selected_validation_blank_cell_accuracy"
                    ],
                    "tie_breakers": [
                        "selected_validation_exact_accuracy",
                        "parameter_count",
                        "candidate_name",
                    ],
                    "n_candidates_planned": len(candidates),
                    "n_candidates_complete": len(successful_rows),
                    "seed_local_selection_only": True,
                    "requires_cross_seed_freeze": True,
                    **NO_TEST_ACCESS_EVIDENCE,
                    "claim_scope": "calibration_only",
                    "claim_conclusion": "inconclusive",
                },
                stage="calibration_selection",
                condition="validation_only_selection",
                task_family="sudoku",
                statistics_unit="seed",
            )
        (run.path / "fit_receipts.json").write_text(
            json.dumps(receipts, indent=2, sort_keys=True), encoding="utf-8"
        )
        return run.path


def main() -> None:
    args = basic_parser(
        "Test-free tiny-recursive Sudoku calibration",
        "configs/smoke/exp17_tiny_recursive_calibration.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        print(run_seed(config, seed, args.results_root))


if __name__ == "__main__":
    main()
