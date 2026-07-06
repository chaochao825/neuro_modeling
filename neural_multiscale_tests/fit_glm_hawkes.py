"""Fit nested history/local/global/stimulus GLMs to a local spike matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spike-matrix", type=Path)
    parser.add_argument("--demo-synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.demo_synthetic:
        out = simulate_hawkes(seed=args.seed)
        result = out["glm_comparison"]
    elif args.spike_matrix:
        result = compare_nested_glms(_load_matrix(args.spike_matrix), seed=args.seed)
    else:
        raise SystemExit("Pass --spike-matrix or --demo-synthetic.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

