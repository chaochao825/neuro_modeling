"""P0 strict paired recurrent-plasticity and tuned-baseline experiment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.analysis.rank_metrics import participation_ratio
from src.baselines.tuned_recurrent import (
    AllCandidatesFailedError,
    RecurrentSequenceData,
    RefitFailedError,
    build_candidate_grid,
    evaluate_masked_mse,
    parameter_count,
    predict_recurrent,
    refit_selected_recurrent_baseline,
    tune_recurrent_baseline,
)
from src.tasks.context_integration import (
    ContextIntegrationBatch,
    ContextIntegrationConfig,
    generate_context_integration,
)
from src.training.context_local import (
    _activity_representation_metrics,
    _trial_behavior,
    balanced_block_split,
)
from src.training.mechanism_identifiability import (
    build_mechanism_conditions,
    prepare_paired_resources,
    run_mechanism_condition,
)
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


def _sequence_data(batch: ContextIntegrationBatch, name: str) -> RecurrentSequenceData:
    return RecurrentSequenceData(
        batch.inputs,
        batch.targets,
        batch.loss_mask,
        batch.block_ids,
        name,
    )


def _candidate_grid(options: Mapping[str, Any], cell_type: str):
    return build_candidate_grid(
        cell_types=(cell_type,),
        hidden_sizes=tuple(int(value) for value in options["hidden_sizes"]),
        learning_rates=tuple(float(value) for value in options["learning_rates"]),
        weight_decays=tuple(
            float(value) for value in options.get("weight_decays", [0.0])
        ),
        rate_leaks=tuple(float(value) for value in options.get("rate_leaks", [1.0])),
        max_epochs=int(options.get("max_epochs", 100)),
        batch_size=int(options.get("batch_size", 32)),
        grad_clip=float(options.get("grad_clip", 1.0)),
        patience=int(options.get("patience", 10)),
        min_delta=float(options.get("min_delta", 0.0)),
    )


def _baseline_metrics(
    development: ContextIntegrationBatch,
    test: ContextIntegrationBatch,
    inner_train: ContextIntegrationBatch,
    validation: ContextIntegrationBatch,
    *,
    cell_type: str,
    baseline_options: Mapping[str, Any],
    training_options: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    device = str(baseline_options.get("device", "cpu"))
    tuned = tune_recurrent_baseline(
        _sequence_data(inner_train, "inner_train"),
        _sequence_data(validation, "inner_validation"),
        _candidate_grid(baseline_options, cell_type),
        seed=derive_seed(seed, "p0-baseline-tuning", cell_type),
        device=device,
    )
    refit = refit_selected_recurrent_baseline(
        _sequence_data(development, "full_development"),
        tuned,
        device=device,
    )
    model = refit.model
    train_prediction, train_activity = predict_recurrent(
        model, development.inputs, device=device
    )
    test_prediction, test_activity = predict_recurrent(
        model, test.inputs, device=device
    )
    predictions = test_prediction[..., 0]
    behavior, _ = _trial_behavior(
        test,
        predictions,
        switch_window=int(training_options.get("switch_window", 1)),
    )
    representation = _activity_representation_metrics(
        train_activity,
        test_activity,
        development.contexts,
        test.contexts,
        reduced_dim=int(training_options.get("reduced_dim", 4)),
        ridge=float(training_options.get("reduced_ridge", 1e-3)),
    )
    metadata = tuned.audit_metadata()
    return {
        **behavior,
        **representation,
        "status": "complete",
        "training_algorithm": (
            "tuned_bptt_rate_rnn" if cell_type == "rate_rnn" else "tuned_bptt_gru"
        ),
        "used_autograd": True,
        "baseline_cell_type": cell_type,
        "parameter_count": parameter_count(model),
        "test_masked_mse": evaluate_masked_mse(
            model, _sequence_data(test, "outer_test"), device=device
        ),
        "activity_participation_ratio_direct": participation_ratio(
            test_activity.reshape(-1, test_activity.shape[-1])
        ),
        "tuning_audit": metadata,
        "refit_audit": refit.audit_metadata(),
        "selected_candidate_id": tuned.selected_candidate_id,
        "selected_config": tuned.selected_config.to_dict(),
        "test_data_used_for_selection": False,
        "inner_train_block_ids": np.unique(inner_train.block_ids).tolist(),
        "validation_block_ids": np.unique(validation.block_ids).tolist(),
        "outer_test_block_ids": np.unique(test.block_ids).tolist(),
        "development_prediction_shape": list(train_prediction.shape),
        "checkpoint_eligible_for_local_initialization": False,
    }


def run_seed(config: dict[str, Any], seed: int, results_root: str) -> Path:
    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": "strict_paired_local_plus_tuned_baselines",
        "used_autograd": True,
        "autograd_scope": "tuned_bptt_and_gru_baselines_only",
        "parent_checkpoint": None,
    }
    with ExperimentRun(
        "exp07_mechanism_identifiability",
        seed,
        run_config,
        results_root=results_root,
    ) as run:
        try:
            architecture = dict(config["architecture"])
            if architecture.get("kind") != "ei":
                raise ValueError("exp07 requires an E/I architecture")
            training = dict(config["training"])
            conditions = build_mechanism_conditions(
                tuple(str(value) for value in config.get("budget_norms", ["l1", "l2"]))
            )
            architecture_dimensions = {
                "architecture": str(architecture["name"]),
                "model_kind": "ei",
                "n_units": int(architecture["n_units"]),
            }
            baseline_dimensions = {
                name: {
                    "condition": name,
                    "mechanism": f"{cell_type}-baseline",
                    "feedback_mode": "not-applicable",
                    "task_plasticity_enabled": False,
                    "homeostasis_enabled": False,
                    "normalization_enabled": False,
                    "budget_norm": "not-applicable",
                    **architecture_dimensions,
                }
                for cell_type, name in (
                    ("rate-rnn-bptt", "tuned-bptt"),
                    ("gru-bptt", "tuned-gru"),
                )
            }
            planned = [
                {
                    "condition": item.name,
                    **{
                        key: value
                        for key, value in item.as_dict().items()
                        if key != "name"
                    },
                    **architecture_dimensions,
                }
                for item in conditions
            ]
            planned.extend(
                baseline_dimensions[name] for name in ("tuned-bptt", "tuned-gru")
            )
            run.register_conditions(planned)
        except Exception as error:
            run.register_conditions([{"condition": "setup"}])
            run.mark_condition_failure(error, condition="setup")
            return run.path

        try:
            batch = generate_context_integration(
                ContextIntegrationConfig(**dict(config["task"])), seed=seed
            )
            development, test = balanced_block_split(
                batch,
                test_fraction=float(config.get("test_fraction", 0.25)),
                seed=derive_seed(seed, "p0-outer-split"),
                switch_window=int(training.get("switch_window", 1)),
            )
            inner_train, validation = balanced_block_split(
                development,
                test_fraction=float(config.get("validation_fraction", 0.25)),
                seed=derive_seed(seed, "p0-inner-split"),
                switch_window=int(training.get("switch_window", 1)),
            )
            resources = prepare_paired_resources(
                development,
                test,
                architecture,
                training,
                seed=derive_seed(seed, "p0-paired-resources"),
            )
            split_metrics = {
                "split_unit": "paired_adjacent_scheduling_blocks",
                "requested_outer_test_fraction": float(
                    config.get("test_fraction", 0.25)
                ),
                "actual_outer_test_fraction": float(
                    test.inputs.shape[0] / batch.inputs.shape[0]
                ),
                "requested_inner_validation_fraction": float(
                    config.get("validation_fraction", 0.25)
                ),
                "actual_inner_validation_fraction": float(
                    validation.inputs.shape[0] / development.inputs.shape[0]
                ),
                "development_block_ids": np.unique(development.block_ids).tolist(),
                "outer_test_block_ids": np.unique(test.block_ids).tolist(),
                "inner_train_block_ids": np.unique(inner_train.block_ids).tolist(),
                "inner_validation_block_ids": np.unique(validation.block_ids).tolist(),
            }
        except Exception as error:
            for dimensions in planned:
                run.mark_condition_failure(error, **dimensions)
            return run.path

        for condition in conditions:
            dimensions = {
                "condition": condition.name,
                **{
                    key: value
                    for key, value in condition.as_dict().items()
                    if key != "name"
                },
                **architecture_dimensions,
            }
            try:
                result = run_mechanism_condition(
                    development,
                    test,
                    resources,
                    condition,
                    training,
                )
                payload = dict(result.metrics)
                for key in dimensions:
                    payload.pop(key, None)
                record = {
                    **payload,
                    **split_metrics,
                    "profile": config.get("profile", "unspecified"),
                    "train_trial_count": int(development.inputs.shape[0]),
                    "validation_trial_count": int(validation.inputs.shape[0]),
                    "test_trial_count": int(test.inputs.shape[0]),
                    "statistics_unit": "seed",
                }
                if bool(result.metrics.get("budget_match_valid", False)):
                    run.record(record, **dimensions)
                else:
                    run.record_failed_condition(record, **dimensions)
            except Exception as error:
                run.mark_condition_failure(error, **dimensions)

        baseline_options = dict(config["baseline"])
        for cell_type, name in (("rate_rnn", "tuned-bptt"), ("gru", "tuned-gru")):
            dimensions = baseline_dimensions[name]
            try:
                metrics = _baseline_metrics(
                    development,
                    test,
                    inner_train,
                    validation,
                    cell_type=cell_type,
                    baseline_options=baseline_options,
                    training_options=training,
                    seed=seed,
                )
                run.record(
                    {
                        **metrics,
                        **split_metrics,
                        "profile": config.get("profile", "unspecified"),
                        "statistics_unit": "seed",
                    },
                    **dimensions,
                )
            except (AllCandidatesFailedError, RefitFailedError) as error:
                run.record_failed_condition(
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "baseline_audit": error.audit_metadata(),
                        "test_data_used_for_selection": False,
                    },
                    **dimensions,
                )
            except Exception as error:
                run.mark_condition_failure(error, **dimensions)
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "P0 mechanism identifiability",
        "configs/formal/exp07_mechanism_identifiability.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
