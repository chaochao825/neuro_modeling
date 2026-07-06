"""Run energy/sparsity/wiring efficiency sweeps."""

from __future__ import annotations

import argparse
import json

from simulations.energy import run_energy_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = run_energy_sweep(n_units=48 if args.quick else 64, t_steps=700 if args.quick else 1200, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

