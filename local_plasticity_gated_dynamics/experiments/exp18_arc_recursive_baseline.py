"""Direct ARC grid generation with recursive refinement and demo-only TTA.

Exp18 is additive: Exp13/15 remain proposal-selection audits.  This experiment
uses BPTT only as an explicitly labelled baseline and never initializes a
local-learning model.  Public-evaluation query targets remain behind the
registered scorer capability for the complete run.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
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
from src.baselines.arc_recursive import (  # noqa: E402
    ARCRecursiveBaseline,
    ARCRecursiveConfig,
    ARCRecursiveTrainingConfig,
    ARCTestTimeConfig,
    fit_arc_recursive,
    parameter_count,
    solve_arc_task,
    state_dict_sha256,
)
from src.data.arc_recursive_dataset import (  # noqa: E402
    build_arc_examples,
    split_arc_training_tasks,
)
from src.data.structured_protocol import PublicTask  # noqa: E402
from src.utils.artifacts import ExperimentRun  # noqa: E402
from src.utils.reproducibility import derive_seed  # noqa: E402


CONDITION_SPECS: Mapping[str, Mapping[str, object]] = {
    "trm_demo_tta_aug_vote": {
        "mode": "trm_like",
        "tta": True,
        "augmentation": True,
    },
    "trm_no_tta_aug_vote": {
        "mode": "trm_like",
        "tta": False,
        "augmentation": True,
    },
    "trm_demo_tta_no_aug": {
        "mode": "trm_like",
        "tta": True,
        "augmentation": False,
    },
    "single_state_demo_tta_aug_vote": {
        "mode": "single_state_core_call_matched",
        "tta": True,
        "augmentation": True,
    },
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _device(value: object) -> str:
    requested = str(value)
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def _strict_determinism(device: str) -> Mapping[str, object]:
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    backend = "cpu_default"
    if device.startswith("cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        backend = "cuda_math_sdp"
    return {
        "strict_deterministic_algorithms": True,
        "attention_backend": backend,
    }


def _model_config(
    config: Mapping[str, Any], *, mode: str, n_puzzles: int
) -> ARCRecursiveConfig:
    model = dict(config.get("model", {}))
    return ARCRecursiveConfig(
        max_grid_size=int(model.get("max_grid_size", 30)),
        hidden_size=int(model.get("hidden_size", 128)),
        num_heads=int(model.get("num_heads", 4)),
        layers=int(model.get("layers", 2)),
        expansion=float(model.get("expansion", 4.0)),
        high_cycles=int(model.get("high_cycles", 3)),
        low_cycles=int(model.get("low_cycles", 4)),
        supervision_steps=int(model.get("supervision_steps", 4)),
        num_puzzle_embeddings=n_puzzles,
        mode=mode,  # type: ignore[arg-type]
    )


def _training_config(config: Mapping[str, Any]) -> ARCRecursiveTrainingConfig:
    training = dict(config.get("training", {}))
    return ARCRecursiveTrainingConfig(
        epochs=int(training.get("epochs", 10)),
        batch_size=int(training.get("batch_size", 8)),
        learning_rate=float(training.get("learning_rate", 1e-4)),
        puzzle_learning_rate=float(training.get("puzzle_learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.1)),
        grad_clip=float(training.get("grad_clip", 1.0)),
        device=_device(training.get("device", "auto")),
    )


def _test_time_config(
    config: Mapping[str, Any], condition: str
) -> ARCTestTimeConfig:
    options = dict(config.get("test_time", {}))
    base = ARCTestTimeConfig(
        adaptation_epochs=int(options.get("adaptation_epochs", 8)),
        learning_rate=float(options.get("learning_rate", 1e-4)),
        weight_decay=float(options.get("weight_decay", 0.0)),
        grad_clip=float(options.get("grad_clip", 1.0)),
        batch_size=int(options.get("batch_size", 8)),
        support_augmentations=int(options.get("support_augmentations", 3)),
        inference_augmentations=int(options.get("inference_augmentations", 8)),
        scope=str(options.get("scope", "full")),  # type: ignore[arg-type]
    )
    specification = CONDITION_SPECS[condition]
    if not bool(specification["tta"]):
        base = replace(base, adaptation_epochs=0)
    if not bool(specification["augmentation"]):
        base = replace(
            base, support_augmentations=0, inference_augmentations=1
        )
    return base


def _task_panel(
    tasks: Sequence[PublicTask], *, limit: int | None, seed: int
) -> tuple[PublicTask, ...]:
    tasks = tuple(tasks)
    if not tasks:
        raise ValueError("task panel is empty")
    if limit is None:
        return tuple(sorted(tasks, key=lambda task: task.task_id))
    if isinstance(limit, bool) or int(limit) < 1:
        raise ValueError("task limits must be positive")
    limit = min(int(limit), len(tasks))

    def key(task: PublicTask) -> str:
        return hashlib.sha256(f"{seed}:{task.task_id}".encode()).hexdigest()

    return tuple(sorted(tasks, key=key)[:limit])


def _bootstrap_mean(
    values: Sequence[float], *, n_bootstrap: int, seed: int
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError("bootstrap values must be a non-empty vector")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
    draws = np.mean(array[indices], axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(np.mean(array)), float(low), float(high)


def _attempt_fingerprints(prediction: Mapping[str, object]) -> tuple[str, ...]:
    fingerprints: list[str] = []
    for attempt in tuple(prediction.get("attempts", ())):
        digest = hashlib.sha256()
        for grid in tuple(attempt):
            array = np.ascontiguousarray(grid, dtype=np.int64)
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        fingerprints.append(digest.hexdigest())
    return tuple(fingerprints)


def _checkpoint_payload(model: ARCRecursiveBaseline) -> Mapping[str, object]:
    return {
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "metadata": model.checkpoint_metadata(),
        "checkpoint_sha256": state_dict_sha256(model),
    }


def run_seed(config: dict[str, Any], seed: int, results_root: str) -> Path:
    initialize_seed(seed)
    if str(config.get("family", "arc")).lower() != "arc":
        raise ValueError("Exp18 supports ARC only")
    conditions = tuple(config.get("conditions", CONDITION_SPECS))
    unknown = set(conditions) - set(CONDITION_SPECS)
    if unknown:
        raise ValueError(f"unknown Exp18 conditions: {sorted(unknown)}")
    if not conditions:
        raise ValueError("Exp18 needs at least one condition")
    data_options = dict(config.get("data", {}))
    if not bool(data_options.get("attempt_aware_scoring", False)):
        raise ValueError("Exp18 requires attempt_aware_scoring=true")
    training_config = _training_config(config)
    deterministic = _strict_determinism(training_config.device)
    run_config = {
        **config,
        "protocol": "demo_tta",
        "training_algorithm": "bptt_faithful_small_trm_arc_baseline",
        "used_autograd": True,
        "parent_checkpoint": None,
        "eligible_for_local_initialization": False,
        "official_trm_reproduction": False,
        "official_arc_attempt_limit": 2,
        "test_target_access": "registered_scorer_only",
        **deterministic,
    }
    with ExperimentRun(
        "exp18_arc_recursive_baseline",
        seed,
        run_config,
        results_root=results_root,
    ) as run:
        dataset, synthetic, provenance = _load_dataset(config, run.path)
        if dataset.families != frozenset({"arc"}):
            raise ValueError("Exp18 loaded a non-ARC dataset")
        split_seed = int(config.get("inner_split_seed", 1729))
        inner_train_tasks, inner_validation_tasks = split_arc_training_tasks(
            dataset,
            validation_fraction=float(config.get("validation_fraction", 0.2)),
            seed=split_seed,
        )
        inner_train_tasks = _task_panel(
            inner_train_tasks,
            limit=config.get("max_train_tasks"),
            seed=int(config.get("train_panel_seed", 2718)),
        )
        inner_validation_tasks = _task_panel(
            inner_validation_tasks,
            limit=config.get("max_validation_tasks"),
            seed=int(config.get("validation_panel_seed", 3141)),
        )
        test_tasks = _task_panel(
            dataset.for_split("test"),
            limit=config.get("test_task_limit"),
            seed=int(config.get("test_panel_seed", 1618)),
        )
        model_options = dict(config.get("model", {}))
        max_grid_size = int(model_options.get("max_grid_size", 30))
        training_examples = build_arc_examples(
            dataset,
            inner_train_tasks,
            max_grid_size=max_grid_size,
            augmentations_per_pair=int(config.get("train_augmentations_per_pair", 3)),
            seed=int(config.get("augmentation_seed", 5772)),
            include_query_targets=True,
            name="inner_train",
        )
        # Checkpoint selection sees only public demonstrations from held-out
        # tasks, never their query outputs.
        validation_examples = build_arc_examples(
            None,
            inner_validation_tasks,
            max_grid_size=max_grid_size,
            augmentations_per_pair=int(
                config.get("validation_augmentations_per_pair", 0)
            ),
            seed=int(config.get("augmentation_seed", 5772)) + 1,
            include_query_targets=False,
            name="inner_validation_public_demos",
        )
        planned = [
            {"condition": condition, "task_id": task.task_id}
            for condition in conditions
            for task in test_tasks
        ]
        run.register_conditions(planned)
        provenance_payload = {
            **dict(provenance),
            "synthetic_not_scientific": synthetic,
            "inner_train_task_ids": tuple(task.task_id for task in inner_train_tasks),
            "inner_validation_task_ids": tuple(
                task.task_id for task in inner_validation_tasks
            ),
            "test_task_ids": tuple(task.task_id for task in test_tasks),
            "inner_validation_query_targets_used_for_selection": False,
            "test_query_targets_used_for_fit_or_tta": False,
        }
        (run.path / "source_provenance.json").write_text(
            json.dumps(provenance_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        needed_modes = {str(CONDITION_SPECS[item]["mode"]) for item in conditions}
        model_seed = derive_seed(seed, "exp18", "shared_initialization")
        torch.manual_seed(model_seed)
        initial_model = ARCRecursiveBaseline(
            _model_config(
                config,
                mode="trm_like",
                n_puzzles=training_examples.n_puzzles,
            )
        )
        initial_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in initial_model.state_dict().items()
        }
        initial_hash = state_dict_sha256(initial_model)
        models: dict[str, ARCRecursiveBaseline] = {}
        fit_receipts: dict[str, object] = {}
        for mode in sorted(needed_modes):
            model = ARCRecursiveBaseline(
                _model_config(
                    config, mode=mode, n_puzzles=training_examples.n_puzzles
                )
            )
            model.load_state_dict(initial_state)
            if state_dict_sha256(model) != initial_hash:
                raise AssertionError("matched model initializations diverged")
            receipt = fit_arc_recursive(
                model,
                training_examples,
                validation_examples,
                training_config,
                seed=derive_seed(seed, "exp18", "matched_training_order"),
            )
            models[mode] = model
            fit_receipts[mode] = {
                **receipt.to_dict(),
                "initial_state_sha256": initial_hash,
                "parameter_count": parameter_count(model),
                "checkpoint_metadata": model.checkpoint_metadata(),
            }
            if bool(config.get("save_checkpoints", True)):
                torch.save(
                    _checkpoint_payload(model),
                    run.path / f"{mode}_checkpoint.pt",
                )
        (run.path / "fit_receipts.json").write_text(
            json.dumps(fit_receipts, indent=2, sort_keys=True), encoding="utf-8"
        )

        rows_by_condition: dict[str, list[dict[str, object]]] = {
            condition: [] for condition in conditions
        }
        n_bootstrap = int(config.get("n_bootstrap", 1000))
        for condition in conditions:
            specification = CONDITION_SPECS[condition]
            mode = str(specification["mode"])
            model = models[mode]
            tta = _test_time_config(config, condition)
            for task_index, task in enumerate(test_tasks):
                dimensions = {
                    "level": "task",
                    "condition": condition,
                    "task_id": task.task_id,
                    "source_group": task.source_group,
                    "task_index": task_index,
                }
                started = time.perf_counter()
                try:
                    prediction, diagnostics = solve_arc_task(
                        model,
                        task,
                        tta,
                        seed=derive_seed(
                            seed, "exp18", "test_task", task.fingerprint
                        ),
                    )
                    score = dict(dataset.target_store.score(task, prediction))
                    attempts = tuple(prediction.get("attempts", ()))
                    first_prediction = {
                        "attempts": attempts[:1]
                    } if attempts else {"attempts": ()}
                    pass1_score = dict(
                        dataset.target_store.score(task, first_prediction)
                    )
                    elapsed = time.perf_counter() - started
                    adaptation = dict(diagnostics["adaptation"])
                    row: dict[str, object] = {
                        "status": "complete",
                        "pass_at_1": bool(pass1_score["exact"]),
                        "pass_at_2": bool(score["exact"]),
                        "shape_exact_fraction": float(score["shape_exact_fraction"]),
                        "best_cell_accuracy": float(score["best_cell_accuracy"]),
                        "query_exact_fraction": float(score["query_exact_fraction"]),
                        "n_queries": int(score["n_queries"]),
                        "n_attempts": int(score["n_attempts_received"]),
                        "attempt_fingerprints": _attempt_fingerprints(prediction),
                        "wall_seconds": elapsed,
                        "adaptation_optimizer_steps": int(
                            adaptation["optimizer_steps"]
                        ),
                        "adaptation_support_examples": int(
                            adaptation["support_examples"]
                        ),
                        "query_targets_used": bool(
                            diagnostics["query_targets_used"]
                        ),
                        "core_calls_per_candidate": int(
                            diagnostics["core_calls_per_candidate"]
                        ),
                        "vote_counts_by_query": diagnostics[
                            "vote_counts_by_query"
                        ],
                        "public_fingerprint": task.fingerprint,
                        "parameter_count": parameter_count(model),
                    }
                    rows_by_condition[condition].append(
                        {**dimensions, **row}
                    )
                    run.record(row, **dimensions)
                except Exception as error:  # keep every failed task in denominator
                    failed = {
                        "failure_reason": str(error),
                        "pass_at_1": False,
                        "pass_at_2": False,
                        "shape_exact_fraction": 0.0,
                        "best_cell_accuracy": 0.0,
                        "query_exact_fraction": 0.0,
                        "wall_seconds": time.perf_counter() - started,
                        "query_targets_used": False,
                    }
                    rows_by_condition[condition].append(
                        {**dimensions, **failed, "status": "failed"}
                    )
                    run.record_failed_condition(failed, **dimensions)

            rows = rows_by_condition[condition]
            summary: dict[str, object] = {"n_tasks": len(rows)}
            for metric in (
                "pass_at_1",
                "pass_at_2",
                "shape_exact_fraction",
                "best_cell_accuracy",
            ):
                estimate, low, high = _bootstrap_mean(
                    [float(row[metric]) for row in rows],
                    n_bootstrap=n_bootstrap,
                    seed=derive_seed(seed, "exp18", condition, metric),
                )
                summary[f"{metric}_mean"] = estimate
                summary[f"{metric}_ci_low"] = low
                summary[f"{metric}_ci_high"] = high
            summary.update(
                failures=sum(row["status"] != "complete" for row in rows),
                query_targets_used=False,
                official_private_score=False,
                formal_claim_promotion_enabled=bool(
                    config.get("formal_claim_promotion_enabled", False)
                ),
            )
            run.record(summary, level="condition_summary", condition=condition)

        registered = dict(config.get("registered_comparison", {}))
        candidate = str(registered.get("candidate", ""))
        reference = str(registered.get("reference", ""))
        if candidate in rows_by_condition and reference in rows_by_condition:
            left = {
                str(row["task_id"]): row for row in rows_by_condition[candidate]
            }
            right = {
                str(row["task_id"]): row for row in rows_by_condition[reference]
            }
            if set(left) != set(right):
                raise AssertionError("registered comparison task panels differ")
            differences = [
                float(left[task_id]["pass_at_2"])
                - float(right[task_id]["pass_at_2"])
                for task_id in sorted(left)
            ]
            estimate, low, high = _bootstrap_mean(
                differences,
                n_bootstrap=n_bootstrap,
                seed=derive_seed(seed, "exp18", "registered_comparison"),
            )
            conclusion = "support" if low > 0 else "oppose" if high < 0 else "inconclusive"
            run.record(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "pass_at_2_difference": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "conclusion": conclusion,
                    "n_independent_tasks": len(differences),
                    "matched_initialization": True,
                    "matched_training_order": True,
                    "claim_scope": "recursive_baseline_not_local_learning",
                },
                level="registered_comparison",
                comparison=str(registered.get("name", "unnamed")),
            )
        return run.path


def main() -> None:
    parser = basic_parser(
        "Run the leakage-safe ARC recursive baseline",
        "configs/smoke/exp18_arc_recursive_arc.json",
    )
    args = parser.parse_args()
    config = load_json_config(args.config)
    seeds = seed_list(args.seeds) if args.seeds else seed_list(config["seeds"])
    paths = [run_seed(config, seed, args.results_root) for seed in seeds]
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
