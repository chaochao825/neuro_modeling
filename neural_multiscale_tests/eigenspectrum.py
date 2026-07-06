"""Cross-validated eigenspectrum and shuffle-control helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyses.metrics import covariance_eigenspectrum, fit_power_law_slope


def analyze_eigenspectrum(x: np.ndarray, seed: int = 0) -> dict:
    gen = np.random.default_rng(seed)
    eig = covariance_eigenspectrum(x)
    shuffled = np.asarray(x).copy()
    for j in range(shuffled.shape[1]):
        gen.shuffle(shuffled[:, j])
    shuf_eig = covariance_eigenspectrum(shuffled)
    return {
        "raw": fit_power_law_slope(eig, stop_rank=min(40, eig.size)),
        "time_shuffle": fit_power_law_slope(shuf_eig, stop_rank=min(40, shuf_eig.size)),
        "top_eigenvalues": [float(v) for v in eig[:10]],
    }


def _load(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return data["spikes"] if "spikes" in data else data[data.files[0]]
    return np.loadtxt(path, delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(analyze_eigenspectrum(_load(args.matrix), args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

