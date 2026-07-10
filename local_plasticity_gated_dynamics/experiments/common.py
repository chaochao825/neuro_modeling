"""Shared experiment CLI and Phase-1 evaluation helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.analysis.manifold_metrics import fit_train_pca, latent_r2, rollout_metrics
from src.analysis.rank_metrics import effective_rank, participation_ratio, singular_values, top_k_singular_energy
from src.models.local_predictive import LocalPredictiveConfig, LocalPredictiveModel
from src.tasks.latent_dynamics import (
    LatentDynamicsConfig,
    SyntheticLatentDataset,
    generate_latent_dynamics,
    make_blockwise_feedback_permutation,
    make_feedback_subspace,
)
from src.utils.reproducibility import derive_seed, set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON object")
    payload["config_path"] = str(config_path.resolve())
    return payload


def seed_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("seed list is empty")
        seeds = [int(part) for part in parts]
    else:
        seeds = [int(seed) for seed in value]
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique non-negative integers")
    return seeds


def basic_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--seeds", default=None, help="comma-separated override")
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    return parser


@dataclass(frozen=True)
class Phase1Condition:
    grid: str
    feedback_dim: int
    feedback_mode: str
    activity_noise_std: float
    weight_decay: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_phase1_conditions(config: Mapping[str, Any]) -> list[Phase1Condition]:
    dims = [int(value) for value in config["feedback_dims"]]
    modes = [str(value) for value in config["feedback_modes"]]
    core_noise = float(config["core_activity_noise_std"])
    core_decay = float(config["core_weight_decay"])
    conditions = {
        Phase1Condition("core", dim, mode, core_noise, core_decay)
        for dim in dims
        for mode in modes
    }
    if config.get("include_ablations", False):
        ablation_dims = [int(value) for value in config.get("ablation_dims", [4, 128])]
        ablation_modes = [str(value) for value in config.get("ablation_modes", ["aligned", "random"])]
        for noise in config.get("activity_noise_sweep", [core_noise]):
            for dim in ablation_dims:
                for mode in ablation_modes:
                    conditions.add(
                        Phase1Condition("noise", dim, mode, float(noise), core_decay)
                    )
        for decay in config.get("weight_decay_sweep", [core_decay]):
            for dim in ablation_dims:
                for mode in ablation_modes:
                    conditions.add(
                        Phase1Condition("decay", dim, mode, core_noise, float(decay))
                    )
    return sorted(
        conditions,
        key=lambda item: (
            item.grid,
            item.activity_noise_std,
            item.weight_decay,
            item.feedback_mode,
            item.feedback_dim,
        ),
    )


def make_phase1_dataset(config: Mapping[str, Any], condition: Phase1Condition, seed: int) -> SyntheticLatentDataset:
    task_options = dict(config["task"])
    task_options.update(seed=seed, activity_noise_std=condition.activity_noise_std)
    return generate_latent_dynamics(LatentDynamicsConfig(**task_options))


def _fit_train_decoder(dataset: SyntheticLatentDataset, alpha: float) -> Ridge:
    transitions = dataset.train.transitions()
    return Ridge(alpha=alpha).fit(transitions.activity_tp1, transitions.latent_tp1)


def _rollout_latents(
    dataset: SyntheticLatentDataset,
    model: LocalPredictiveModel,
    decoder: Ridge,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for block, context in zip(
        dataset.test.activities, dataset.test.block_contexts, strict=True
    ):
        steps = min(horizon, block.shape[0] - 1)
        predicted_activity = model.rollout(block[0], steps, int(context))
        predicted_latent = decoder.predict(predicted_activity)
        block_index = len(truths)
        # Match the corresponding latent block by the shared block ordering.
        true_latent = dataset.test.latent_states[block_index, : steps + 1]
        truths.append(true_latent[1:])
        predictions.append(predicted_latent[1:])
    return np.stack(truths), np.stack(predictions)


def evaluate_phase1_condition(
    dataset: SyntheticLatentDataset,
    condition: Phase1Condition,
    experiment_config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], LocalPredictiveModel]:
    """Train one local-feedback cell and evaluate it only on held-out blocks."""

    train = dataset.train.transitions()
    test = dataset.test.transitions()
    feedback = make_feedback_subspace(
        dataset.embedding,
        condition.feedback_dim,
        condition.feedback_mode,
        seed=derive_seed(seed, "feedback", condition.feedback_mode, condition.feedback_dim),
    )
    model_options = dict(experiment_config["model"])
    model_options.update(
        seed=derive_seed(seed, "local_model", condition.feedback_mode, condition.feedback_dim),
        weight_decay=condition.weight_decay,
    )
    model = LocalPredictiveModel(
        feedback,
        n_contexts=dataset.config.n_contexts,
        config=LocalPredictiveConfig(**model_options),
    )
    permutation = None
    if condition.feedback_mode == "shuffled":
        permutation = make_blockwise_feedback_permutation(
            train.block_ids,
            seed=derive_seed(seed, "temporal_shuffle", condition.feedback_dim),
        )
    model.fit_fixed_point(
        train.activity_t,
        train.activity_tp1,
        train.contexts,
        feedback_permutation=permutation,
        block_ids=train.block_ids if permutation is not None else None,
    )

    predicted_activity = model.predict(test.activity_t, test.contexts)
    decoder = _fit_train_decoder(dataset, float(experiment_config.get("decoder_ridge", 1e-6)))
    predicted_latent = decoder.predict(predicted_activity)
    one_step_latent_r2 = float(latent_r2(test.latent_tp1, predicted_latent))
    one_step_activity_r2 = float(
        r2_score(test.activity_tp1, predicted_activity, multioutput="variance_weighted")
    )

    rollout_truth, rollout_prediction = _rollout_latents(
        dataset,
        model,
        decoder,
        int(experiment_config.get("rollout_horizon", 100)),
    )
    rollout = rollout_metrics(
        rollout_truth,
        rollout_prediction,
        train_reference=train.latent_tp1,
    )

    task_pca = fit_train_pca(
        train.activity_tp1,
        dataset.config.latent_dim,
        normalize=False,
        sample_ids=np.arange(train.activity_tp1.shape[0]),
    )
    centered_prediction = predicted_activity - task_pca.mean_
    in_task = (centered_prediction @ task_pca.basis_) @ task_pca.basis_.T
    outside = centered_prediction - in_task
    total_prediction_energy = float(np.sum(centered_prediction**2))
    noise_energy_fraction = (
        float(np.sum(outside**2)) / total_prediction_energy
        if total_prediction_energy > 0.0
        else 0.0
    )

    component = model.plastic_component
    ranks = [effective_rank(matrix) for matrix in component]
    top_energy = [
        top_k_singular_energy(matrix, dataset.config.latent_dim) for matrix in component
    ]
    spectra = [singular_values(matrix).tolist() for matrix in component]
    converged = bool(model.converged_)
    metrics: dict[str, Any] = {
        "status": "complete" if converged else "failed",
        "failure_reason": None if converged else "local fixed-point iteration did not converge",
        "requested_feedback_dim": condition.feedback_dim,
        "actual_feedback_dim": feedback.dimension,
        "feedback_alignment_fraction": feedback.alignment_fraction,
        "effective_rank": float(np.mean(ranks)),
        "effective_rank_by_context": ranks,
        "top_k_singular_energy": float(np.mean(top_energy)),
        "top_k_singular_energy_by_context": top_energy,
        "singular_values_by_context": spectra,
        "latent_r2": one_step_latent_r2,
        "activity_r2": one_step_activity_r2,
        "rollout_rmse": rollout.rmse,
        "rollout_normalized_rmse": rollout.normalized_rmse,
        "rollout_per_horizon_rmse": rollout.per_horizon_rmse.tolist(),
        "plasticity_cost": model.plasticity_cost,
        "raw_plasticity_cost": model.raw_plasticity_cost,
        "noise_prediction_energy_fraction": noise_energy_fraction,
        "activity_dimension": participation_ratio(predicted_activity),
        "n_epochs": model.n_epochs_,
        "converged": converged,
        "training_algorithm": "local_predictive_fixed_point_iteration",
        "used_autograd": False,
    }
    return metrics, model


def initialize_seed(seed: int) -> None:
    set_global_seed(seed)
