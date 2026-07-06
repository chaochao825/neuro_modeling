"""Unified local interface for public neural dataset exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyses.metrics import summarize_activity
from data_loaders.public_registry import describe_registry
from eigenspectrum import analyze_eigenspectrum
from models.glm import compare_nested_glms
from simulations.hawkes import simulate_hawkes


def _load_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        key = "spikes" if "spikes" in data else data.files[0]
        return data[key]
    return np.loadtxt(path, delimiter=",")


def analyze_public_matrix(spikes: np.ndarray, dataset: str, bin_ms: float, seed: int) -> dict:
    return {
        "dataset": dataset,
        "bin_ms": bin_ms,
        "activity": summarize_activity(spikes),
        "glm_hawkes": compare_nested_glms(spikes, seed=seed),
        "eigenspectrum_controls": analyze_eigenspectrum(spikes, seed=seed),
        "state_controls_required": [
            "running",
            "pupil",
            "lick",
            "wheel",
            "trial_event",
            "reward",
            "sleep_wake",
            "theta_phase",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--spike-matrix", type=Path)
    parser.add_argument("--dataset", default="local_export")
    parser.add_argument("--bin-ms", type=float, default=10.0)
    parser.add_argument("--demo-synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.list_datasets:
        print(json.dumps(describe_registry(), indent=2, sort_keys=True))
        return
    if args.demo_synthetic:
        spikes = simulate_hawkes(seed=args.seed)["spikes"]
    elif args.spike_matrix:
        spikes = _load_matrix(args.spike_matrix)
    else:
        raise SystemExit("Pass --list-datasets, --spike-matrix, or --demo-synthetic.")
    print(json.dumps(analyze_public_matrix(spikes, args.dataset, args.bin_ms, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

