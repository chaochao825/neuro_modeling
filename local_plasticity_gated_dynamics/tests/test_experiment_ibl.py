import json
from pathlib import Path

import numpy as np

from experiments.common import load_json_config
from experiments.exp06_ibl_context_switch import _block_sequences, run_seed
from src.data.ibl_loader import ProbeSpikes


class RichFakeIBLSource:
    def __init__(self) -> None:
        self.n_trials = 30
        self.stim = 1.0 + np.arange(self.n_trials) * 2.0
        self.move = self.stim + 0.35
        self.probability = np.repeat([0.8, 0.2, 0.8, 0.2, 0.8, 0.2], 5)
        self.choice = np.where(np.arange(self.n_trials) % 3 == 0, -1, 1)

    def search_sessions(self, *, limit):
        return ["fake-eid"][:limit]

    def load_trials(self, eid):
        return {
            "stimOn_times": self.stim,
            "firstMovement_times": self.move,
            "contrastLeft": np.where(np.arange(self.n_trials) % 2 == 0, 0.5, np.nan),
            "contrastRight": np.where(np.arange(self.n_trials) % 2 == 1, 0.5, np.nan),
            "choice": self.choice,
            "feedbackType": np.where(np.arange(self.n_trials) % 4 == 0, -1, 1),
            "probabilityLeft": self.probability,
        }

    def load_probe_spikes(self, eid):
        times = []
        clusters = []
        for trial, event in enumerate(self.stim):
            for unit in range(6):
                count = 1 + ((trial + unit) % 3)
                offsets = np.linspace(-0.18, -0.02, count)
                times.extend(event + offsets + unit * 1e-4)
                clusters.extend([unit] * count)
                # A second packet precedes movement but lies after stimulus.
                times.extend(self.move[trial] + offsets)
                clusters.extend([unit] * count)
        order = np.argsort(times)
        return [
            ProbeSpikes(
                "probe00",
                np.asarray(times)[order],
                np.asarray(clusters)[order],
                np.arange(6),
                np.array(["MD", "CP", "MOs", "VISp", "ACA", "LP"]),
            )
        ]

    def load_wheel(self, eid):
        timestamps = np.linspace(0, self.move[-1] + 1, 3000)
        return {"timestamps": timestamps, "position": np.sin(timestamps)}

    def load_pose_summary(self, eid, events, *, window_s=(-0.5, 0.0)):
        return 0.5 + 0.1 * np.sin(events)

    def session_details(self, eid):
        return {"subject": "mouse-fake"}


def test_ibl_experiment_runs_with_fake_source_and_session_level_records(tmp_path: Path) -> None:
    config = load_json_config("configs/smoke/exp06_ibl_context_switch.json")
    path = run_seed(config, 0, str(tmp_path), source=RichFakeIBLSource())
    records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    assert any(record.get("model_family") == "shared" for record in records)
    assert {record.get("view") for record in records} == {"stimulus_pre", "movement_pre"}
    assert all(record["aggregation_level"] == "session" for record in records)
    assert any(record.get("lead_lag_is_causal_claim") is False for record in records)
    lead_record = next(record for record in records if record.get("model_family") == "lead_lag")
    assert lead_record["condition_schedule_observed"] is False
    assert lead_record["latent_source"] == "shared_switching_lds_hidden_context_imm_posterior"


def test_hidden_context_sequences_do_not_reset_at_true_context_boundary() -> None:
    latent = np.arange(12, dtype=float).reshape(4, 3)
    conditions = np.array([0.2, 0.2, 0.8, 0.8])
    dataset, positions = _block_sequences(
        latent,
        conditions,
        np.array([0, 0, 1, 1]),
        np.arange(4),
    )
    assert len(dataset.observations) == 1
    np.testing.assert_array_equal(dataset.conditions[0], conditions)
    np.testing.assert_array_equal(positions[0], np.arange(4))
