"""Estimate effective linear dynamics, spectral radius, and Lyapunov covariance agreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyses.metrics import covariance_agreement, dmd_summary, fit_linear_dynamics, lyapunov_covariance
from simulations.linear_dynamics import run_linear_suite


def _load_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return data["spikes"] if "spikes" in data else data[data.files[0]]
    return np.loadtxt(path, delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--demo-synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.demo_synthetic:
        result = run_linear_suite(seed=args.seed)
    elif args.matrix:
        x = _load_matrix(args.matrix)
        a = fit_linear_dynamics(x)
        q = np.eye(a.shape[0])
        pred = lyapunov_covariance(a, q)
        result = {"dmd": dmd_summary(x), "lyapunov_agreement": covariance_agreement(np.cov(x, rowvar=False), pred)}
    else:
        raise SystemExit("Pass --matrix or --demo-synthetic.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

