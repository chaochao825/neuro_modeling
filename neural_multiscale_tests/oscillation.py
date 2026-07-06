"""Oscillation metrics for population rate and spike matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyses.metrics import dmd_summary, psd_summary, spike_phase_locking
from simulations.ei_spiking import run_ei_suite


def analyze_oscillation(spikes: np.ndarray, dt: float = 0.001) -> dict:
    pop = spikes.sum(axis=1)
    return {"psd": psd_summary(pop, dt=dt), "dmd": dmd_summary(spikes), "phase_locking": spike_phase_locking(spikes, pop)}


def _load(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return data["spikes"] if "spikes" in data else data[data.files[0]]
    return np.loadtxt(path, delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spike-matrix", type=Path)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--demo-synthetic", action="store_true")
    args = parser.parse_args()
    if args.demo_synthetic:
        result = run_ei_suite()
    elif args.spike_matrix:
        result = analyze_oscillation(_load(args.spike_matrix), dt=args.dt)
    else:
        raise SystemExit("Pass --spike-matrix or --demo-synthetic.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

