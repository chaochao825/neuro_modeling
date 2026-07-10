"""Verify the local predictive fixed point and low-rank analytic invariant."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import basic_parser, initialize_seed, load_json_config, seed_list
from src.analysis.rank_metrics import effective_rank, top_k_singular_energy
from src.models.local_predictive import LocalPredictiveConfig, LocalPredictiveModel
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import make_rng


def run_seed(config: dict, seed: int, results_root: str) -> Path:
    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": "local_predictive_fixed_point_iteration",
        "used_autograd": False,
        "parent_checkpoint": None,
    }
    with ExperimentRun("exp00_fixed_point", seed, run_config, results_root=results_root) as run:
        run.register_conditions([{"condition": "linear_fixed_point"}])
        try:
            n = int(config["n_neurons"])
            d = int(config["latent_dim"])
            samples = int(config["n_samples"])
            rng = make_rng(seed, "exp00")
            basis, _ = np.linalg.qr(rng.normal(size=(n, d)), mode="reduced")
            latent_matrix = 0.85 * np.eye(d)
            for index in range(0, d - 1, 2):
                angle = 0.15 if index == 0 else -0.1
                rotation = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                latent_matrix[index : index + 2, index : index + 2] = 0.95 * rotation
            latent = rng.normal(size=(samples, d))
            x = latent @ basis.T
            y = (latent @ latent_matrix.T) @ basis.T
            c00 = x.T @ x / samples
            c10 = y.T @ x / samples
            analytic = basis @ latent_matrix @ basis.T
            local = LocalPredictiveModel(
                basis,
                config=LocalPredictiveConfig(**config["model"], seed=seed),
            ).fit_fixed_point(x, y)
            learned = local.plastic_component[0]
            metrics = {
                "status": "complete" if local.converged_ else "failed",
                "failure_reason": None if local.converged_ else "fixed-point iteration did not converge",
                "analytic_effective_rank": effective_rank(analytic),
                "analytic_top_k_energy": top_k_singular_energy(analytic, d),
                "analytic_fixed_point_residual": float(np.linalg.norm(analytic @ c00 - c10)),
                "learned_effective_rank": effective_rank(learned),
                "learned_top_k_energy": top_k_singular_energy(learned, d),
                "learned_fixed_point_residual": float(np.linalg.norm(learned @ c00 - c10)),
                "learned_vs_analytic_fro": float(np.linalg.norm(learned - analytic)),
                "rank_bound": d,
                "n_epochs": local.n_epochs_,
                "converged": bool(local.converged_),
                "training_algorithm": "local_predictive_fixed_point_iteration",
                "used_autograd": False,
            }
            if local.converged_:
                run.record(metrics, condition="linear_fixed_point")
            else:
                run.record_failed_condition(metrics, condition="linear_fixed_point")
        except Exception as error:
            run.mark_condition_failure(error, condition="linear_fixed_point")
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "fixed point", "configs/formal/exp00_fixed_point.json"
    ).parse_args()
    config = load_json_config(args.config)
    seeds = seed_list(args.seeds or config["seeds"])
    for seed in seeds:
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
