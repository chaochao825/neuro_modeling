import json
from pathlib import Path

import numpy as np

from experiments.common import load_json_config
from experiments.exp04_phase_gating import run_seed
from src.models.phase_gating import PhaseGatingConfig, simulate_phase_gating


def test_phase_conditions_are_exactly_rate_spike_and_coupling_matched() -> None:
    config = PhaseGatingConfig(n_trials=20, n_steps=20, spikes_per_trial=25, block_size=5)
    simulations = [
        simulate_phase_gating(condition, config)
        for condition in ("in_phase", "anti_phase", "random_phase", "no_oscillation")
    ]
    assert np.allclose([simulation.rates_hz.mean() for simulation in simulations], 10.0)
    assert all(np.all(simulation.spikes.sum(axis=1) == 25) for simulation in simulations)
    assert np.allclose([simulation.coupling_trace.mean() for simulation in simulations], 1.0)
    assert all(np.array_equal(simulations[0].source, simulation.source) for simulation in simulations[1:])


def test_phase_experiment_smoke_records_all_conditions(tmp_path: Path) -> None:
    config = load_json_config("configs/smoke/exp04_phase_gating.json")
    path = run_seed(config, 0, str(tmp_path))
    records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert all(record["status"] == "complete" for record in records)
    assert len({round(record["mean_firing_rate_hz"], 8) for record in records}) == 1
    assert len({record["spike_count_per_trial_mean"] for record in records}) == 1
    assert all(record["mean_rate_match_exact"] for record in records)
    assert all(record["per_trial_spike_count_match_exact"] for record in records)
    assert all(record["mean_coupling_match_exact"] for record in records)
    assert len({record["shared_source_fingerprint"] for record in records}) == 1
