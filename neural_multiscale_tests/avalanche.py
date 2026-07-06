"""Avalanche extraction and tail-model comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyses.metrics import branching_ratio_counts, compare_tail_models, extract_avalanches
from simulations.branching import run_branching_suite


def analyze_counts(counts: np.ndarray) -> dict:
    sizes, durations = extract_avalanches(counts)
    return {
        "branching_ratio": branching_ratio_counts(counts),
        "n_avalanches": int(sizes.size),
        "size_tail": compare_tail_models(sizes),
        "duration_tail": compare_tail_models(durations),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=Path)
    parser.add_argument("--demo-synthetic", action="store_true")
    args = parser.parse_args()
    if args.demo_synthetic:
        result = run_branching_suite()
    elif args.counts:
        result = analyze_counts(np.loadtxt(args.counts, delimiter=","))
    else:
        raise SystemExit("Pass --counts or --demo-synthetic.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

