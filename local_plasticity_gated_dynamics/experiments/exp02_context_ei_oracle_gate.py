"""Phase-2 context integration with oracle gain gating and required controls."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import basic_parser, initialize_seed, load_json_config, seed_list
from src.training.context_local import run_phase2_experiment


def run_seed(config: dict, seed: int, results_root: str) -> Path:
    initialize_seed(seed)
    return run_phase2_experiment(
        config,
        seed=seed,
        results_root=results_root,
        experiment_name="exp02_context_ei_oracle_gate",
        base_gate="oracle",
    )


def main() -> None:
    args = basic_parser(
        __doc__ or "oracle context gate",
        "configs/formal/exp02_context_ei_oracle_gate.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
