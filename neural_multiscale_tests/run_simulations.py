"""Run all synthetic validation models and write reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from analyses.reporting import write_outputs
from simulations.baseline import run_baseline
from simulations.branching import run_branching_suite
from simulations.ei_spiking import run_ei_suite
from simulations.energy import run_energy_sweep
from simulations.hawkes import simulate_hawkes
from simulations.linear_dynamics import run_linear_suite


def run_all(seed: int, quick: bool) -> Dict[str, object]:
    n_units = 36 if quick else 72
    t_steps = 900 if quick else 2200
    return {
        "metadata": {"seed": seed, "quick": quick, "n_units": n_units, "t_steps": t_steps},
        "baseline": _strip_arrays(run_baseline(n_units=n_units, t_steps=t_steps, seed=seed)),
        "hawkes": _strip_arrays(simulate_hawkes(n_units=n_units, t_steps=t_steps, seed=seed + 1)),
        "linear": run_linear_suite(n_units=n_units, t_steps=t_steps + 400, seed=seed + 2),
        "branching": run_branching_suite(t_steps=3000 if quick else 6500, seed=seed + 3),
        "ei": run_ei_suite(seed=seed + 4),
        "energy": run_energy_sweep(n_units=max(48, n_units), t_steps=700 if quick else 1200, seed=seed + 5),
    }


def _strip_arrays(obj: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in obj.items() if k not in {"spikes", "stimulus", "population_rate"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true", help="Use small dimensions for CI/SSH smoke runs.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    summary = run_all(seed=args.seed, quick=args.quick)
    paths = write_outputs(summary, args.root)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()

