"""Test phase-dependent transfer after exact firing-rate/spike-count matching."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import basic_parser, initialize_seed, load_json_config, seed_list
from src.models.phase_gating import (
    PhaseGatingConfig,
    information_transfer_latency,
    simulate_phase_gating,
)
from src.utils.artifacts import ExperimentRun
from src.utils.splits import grouped_train_test_split


def _transfer_prediction(simulation, seed: int) -> float:
    train_trials, test_trials = grouped_train_test_split(
        simulation.block_ids, test_fraction=0.25, seed=seed
    )

    def rows(trials):
        current = simulation.rates_hz[trials, :-1].reshape(-1, 1)
        source = simulation.source[trials, :-1].reshape(-1, 1)
        target = simulation.rates_hz[trials, 1:].reshape(-1)
        return current, source, target

    current_train, source_train, target_train = rows(train_trials)
    current_test, source_test, target_test = rows(test_trials)
    reduced = Ridge(alpha=1e-3).fit(current_train, target_train)
    augmented = Ridge(alpha=1e-3).fit(
        np.column_stack([current_train, source_train]), target_train
    )
    reduced_r2 = r2_score(target_test, reduced.predict(current_test))
    augmented_r2 = r2_score(
        target_test, augmented.predict(np.column_stack([current_test, source_test]))
    )
    return float(augmented_r2 - reduced_r2)


def _decode_accuracy(simulation, seed: int) -> float:
    train, test = grouped_train_test_split(
        simulation.block_ids, test_fraction=0.25, seed=seed
    )
    features = simulation.rates_hz[:, simulation.rates_hz.shape[1] // 2 :]
    decoder = LogisticRegression(C=1.0, random_state=seed).fit(
        features[train], simulation.labels[train]
    )
    return float(accuracy_score(simulation.labels[test], decoder.predict(features[test])))


def run_seed(config: dict, seed: int, results_root: str) -> Path:
    initialize_seed(seed)
    conditions = list(config["conditions"])
    run_config = {
        **config,
        "training_algorithm": "fixed_rate_phase_communication",
        "used_autograd": False,
        "parent_checkpoint": None,
    }
    with ExperimentRun("exp04_phase_gating", seed, run_config, results_root=results_root) as run:
        run.register_conditions([{"phase_condition": condition} for condition in conditions])
        for condition in conditions:
            try:
                options = dict(config["simulation"])
                options["seed"] = seed
                simulation = simulate_phase_gating(condition, PhaseGatingConfig(**options))
                per_trial_spikes = np.sum(simulation.spikes, axis=1)
                source_fingerprint = hashlib.sha256(
                    np.ascontiguousarray(simulation.source).view(np.uint8)
                ).hexdigest()
                metrics = {
                    "status": "complete",
                    "decoding_accuracy": _decode_accuracy(simulation, seed),
                    "information_transfer_latency_bins": information_transfer_latency(simulation),
                    "low_dimensional_mode_gain": float(
                        np.std(simulation.rates_hz) / np.std(simulation.source)
                    ),
                    "cross_validated_transfer_r2_gain": _transfer_prediction(simulation, seed),
                    "mean_firing_rate_hz": float(np.mean(simulation.rates_hz)),
                    "spike_count_per_trial_mean": float(
                        np.mean(np.sum(simulation.spikes, axis=1))
                    ),
                    "mean_effective_coupling": float(np.mean(simulation.coupling_trace)),
                    "mean_rate_match_exact": bool(
                        np.isclose(
                            np.mean(simulation.rates_hz),
                            float(options["target_rate_hz"]),
                            rtol=0.0,
                            atol=1e-12,
                        )
                    ),
                    "per_trial_spike_count_match_exact": bool(
                        np.all(per_trial_spikes == int(options["spikes_per_trial"]))
                    ),
                    "mean_coupling_match_exact": bool(
                        np.isclose(
                            np.mean(simulation.coupling_trace),
                            1.0,
                            rtol=0.0,
                            atol=1e-12,
                        )
                    ),
                    "shared_source_fingerprint": source_fingerprint,
                    "matching_scope": (
                        "exact_global_mean_rate_per_trial_spike_total_"
                        "mean_coupling_and_shared_source"
                    ),
                    "analysis_signal": "rate_matched_firing_rate",
                }
                run.record(metrics, phase_condition=condition)
            except Exception as error:
                run.mark_condition_failure(error, phase_condition=condition)
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "phase gating", "configs/formal/exp04_phase_gating.json"
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
