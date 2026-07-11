"""Bridge hidden-context beliefs into a frozen Dale E/I gain receiver."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (
    basic_parser,
    initialize_seed,
    load_json_config,
    seed_list,
)
from src.models.belief_gain import balanced_gain_axis
from src.models.ei_rate_network import EIRateNetwork
from src.tasks.hidden_context import (
    HiddenContextConfig,
    generate_hidden_context,
    make_hidden_context_random_tape,
)
from src.training.hidden_context_ei import (
    BRIDGE_BASE_GATES,
    BRIDGE_MD_INTERVENTIONS,
    evaluate_receiver_condition,
    fit_gate_split_predictions,
    fit_receiver_readout,
    intervene_on_test_prediction,
    simulate_receiver,
)
from src.training.hidden_context_gate import split_hidden_context_dataset
from src.utils.artifacts import ExperimentRun
from src.utils.reproducibility import derive_seed


def _array_id(label: str, *arrays: np.ndarray) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _protocol_id(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "config_path"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(b"exp10-bridge-protocol-v1\0" + encoded).hexdigest()


def _condition_name(q: float, h: float, gate: str, intervention: str) -> str:
    return (
        f"q{q:.2f}".replace(".", "p")
        + f"__h{h:.2f}".replace(".", "p")
        + f"__{gate}__{intervention}"
    )


def _planned_conditions(config: dict[str, Any]) -> list[dict[str, object]]:
    conditions: list[dict[str, object]] = []
    for q in config["cue_reliabilities"]:
        for h in config["context_hazards"]:
            for gate in BRIDGE_BASE_GATES:
                conditions.append(
                    {
                        "condition": _condition_name(float(q), float(h), gate, "none"),
                        "cue_reliability": float(q),
                        "context_hazard": float(h),
                        "gate_model": gate,
                        "intervention": "none",
                    }
                )
            for intervention in BRIDGE_MD_INTERVENTIONS:
                conditions.append(
                    {
                        "condition": _condition_name(
                            float(q),
                            float(h),
                            "md_recurrent_belief",
                            intervention,
                        ),
                        "cue_reliability": float(q),
                        "context_hazard": float(h),
                        "gate_model": "md_recurrent_belief",
                        "intervention": intervention,
                    }
                )
    if not conditions or len(conditions) != len(
        {str(item["condition"]) for item in conditions}
    ):
        raise ValueError("bridge condition grid must be non-empty and unique")
    return conditions


def _network(
    config: dict[str, Any], task: HiddenContextConfig, seed: int
) -> EIRateNetwork:
    options = dict(config["network"])
    substeps = int(config["integration_substeps"])
    expected_dt = float(task.dt_ms) / substeps
    configured_dt = float(options.get("dt", expected_dt))
    if not np.isclose(configured_dt, expected_dt, atol=0.0, rtol=1e-12):
        raise ValueError(
            "network.dt * integration_substeps must equal the task time step"
        )
    options["dt"] = configured_dt
    return EIRateNetwork(
        n_inputs=2,
        seed=derive_seed(seed, "p2-ei", "network-init"),
        **options,
    )


def run_seed(config: dict[str, Any], seed: int, results_root: str) -> Path:
    """Run one paired seed; every condition shares receiver and task tapes."""

    initialize_seed(seed)
    run_config = {
        **config,
        "training_algorithm": "hidden_belief_rank1_gain_frozen_ei",
        "used_autograd": False,
        "parent_checkpoint": None,
        "recurrent_learning": False,
    }
    with ExperimentRun(
        "exp10_hidden_context_ei_bridge",
        seed,
        run_config,
        results_root=results_root,
    ) as run:
        try:
            planned = _planned_conditions(config)
            run.register_conditions(planned)
            q0 = float(config["cue_reliabilities"][0])
            h0 = float(config["context_hazards"][0])
            task_template = HiddenContextConfig(
                **dict(config["task"]),
                cue_reliability=q0,
                context_hazard=h0,
            )
            tape = make_hidden_context_random_tape(task_template, seed=seed)
            network = _network(config, task_template, seed)
            initial_weights = network.recurrent_weights
            gain_axis = balanced_gain_axis(
                network.excitatory_mask,
                seed=derive_seed(seed, "p2-ei", "gain-axis"),
            )
            network_id = _array_id(
                "frozen-dale-ei-network-v1",
                network.recurrent_weights,
                network.input_weights,
            )
            gain_axis_id = _array_id("balanced-ei-rank1-gain-axis-v1", gain_axis)
            protocol_id = _protocol_id(config)
        except Exception as error:
            if "planned" not in locals():
                planned = [{"condition": "setup"}]
                run.register_conditions(planned)
            for dimensions in planned:
                run.mark_condition_failure(error, **dimensions)
            return run.path

        emitted: set[str] = set()

        def emit_failure(dimensions: dict[str, object], error: BaseException) -> None:
            name = str(dimensions["condition"])
            if name in emitted:
                raise RuntimeError(f"condition emitted twice: {name}")
            run.mark_condition_failure(error, **dimensions)
            emitted.add(name)

        def emit_success(
            dimensions: dict[str, object], metrics: dict[str, object]
        ) -> None:
            name = str(dimensions["condition"])
            if name in emitted:
                raise RuntimeError(f"condition emitted twice: {name}")
            run.record(metrics, **dimensions)
            emitted.add(name)

        by_cell: dict[tuple[float, float], list[dict[str, object]]] = {}
        for item in planned:
            by_cell.setdefault(
                (float(item["cue_reliability"]), float(item["context_hazard"])), []
            ).append(item)

        for (q, h), cell_conditions in by_cell.items():
            condition_map = {
                (str(item["gate_model"]), str(item["intervention"])): item
                for item in cell_conditions
            }
            try:
                dataset = generate_hidden_context(
                    replace(task_template, cue_reliability=q, context_hazard=h),
                    seed=seed,
                    random_tape=tape,
                )
                splits = split_hidden_context_dataset(
                    dataset,
                    outer_test_fraction=float(config["outer_test_fraction"]),
                    validation_fraction=float(config["validation_fraction"]),
                    seed=seed,
                )
            except Exception as error:
                for dimensions in cell_conditions:
                    emit_failure(dimensions, error)
                continue

            md_bundle: tuple[object, object] | None = None
            for gate_model in BRIDGE_BASE_GATES:
                dimensions = condition_map[(gate_model, "none")]
                try:
                    fitted = fit_gate_split_predictions(
                        gate_model,
                        splits,
                        context_hazard=h,
                        cue_reliability=q,
                        config=config,
                        seed=seed,
                    )
                    train_simulation = simulate_receiver(
                        network,
                        splits.train,
                        fitted.train.context_probability,
                        gain_axis,
                        gain_strength=float(config["gain_strength"]),
                        integration_substeps=int(config["integration_substeps"]),
                        trial_batch_size=int(config["trial_batch_size"]),
                    )
                    test_simulation = simulate_receiver(
                        network,
                        splits.test,
                        fitted.test.context_probability,
                        gain_axis,
                        gain_strength=float(config["gain_strength"]),
                        integration_substeps=int(config["integration_substeps"]),
                        trial_batch_size=int(config["trial_batch_size"]),
                    )
                    readout = fit_receiver_readout(
                        train_simulation,
                        splits.train.task,
                        alpha=float(config["readout_alpha"]),
                    )
                    gate_cost = dict(config["gate_compute"])[gate_model]
                    metrics = evaluate_receiver_condition(
                        network=network,
                        simulation=test_simulation,
                        readout=readout,
                        prediction=fitted.test,
                        dataset=splits.test,
                        gate_model=gate_model,
                        intervention="none",
                        gate_checkpoint_id=fitted.checkpoint_id,
                        gain_axis_id=gain_axis_id,
                        split_id=splits.fingerprint,
                        network_init_id=network_id,
                        gate_operations_per_trial=float(gate_cost["operations"]),
                        gate_state_updates_per_trial=float(gate_cost["states"]),
                    )
                    metrics.update(
                        profile=str(config.get("profile", "unspecified")),
                        training_algorithm="hidden_belief_rank1_gain_frozen_ei",
                        gate_fit_supervision=fitted.fit_metadata.get(
                            "gate_fit_supervision"
                        ),
                        gate_received_true_q_h=fitted.fit_metadata.get(
                            "gate_received_true_q_h"
                        ),
                        random_tape_id=tape.fingerprint,
                        bridge_protocol_id=protocol_id,
                        network_n_units=network.n_units,
                        network_excitatory_fraction=network.excitatory_fraction,
                    )
                    emit_success(dimensions, metrics)
                    if gate_model == "md_recurrent_belief":
                        md_bundle = (fitted, readout)
                except Exception as error:
                    emit_failure(dimensions, error)
                    if gate_model == "md_recurrent_belief":
                        for intervention in BRIDGE_MD_INTERVENTIONS:
                            emit_failure(
                                condition_map[(gate_model, intervention)], error
                            )

            if md_bundle is not None:
                fitted, readout = md_bundle
                for intervention in BRIDGE_MD_INTERVENTIONS:
                    dimensions = condition_map[("md_recurrent_belief", intervention)]
                    try:
                        altered = intervene_on_test_prediction(
                            fitted,
                            intervention,
                            delay_trials=int(config["interventions"]["delay_trials"]),
                            seed=seed,
                        )
                        simulation = simulate_receiver(
                            network,
                            splits.test,
                            altered.context_probability,
                            gain_axis,
                            gain_strength=float(config["gain_strength"]),
                            integration_substeps=int(config["integration_substeps"]),
                            trial_batch_size=int(config["trial_batch_size"]),
                        )
                        gate_cost = dict(config["gate_compute"])["md_recurrent_belief"]
                        metrics = evaluate_receiver_condition(
                            network=network,
                            simulation=simulation,
                            readout=readout,
                            prediction=altered,
                            dataset=splits.test,
                            gate_model="md_recurrent_belief",
                            intervention=intervention,
                            gate_checkpoint_id=fitted.checkpoint_id,
                            gain_axis_id=gain_axis_id,
                            split_id=splits.fingerprint,
                            network_init_id=network_id,
                            gate_operations_per_trial=float(gate_cost["operations"]),
                            gate_state_updates_per_trial=float(gate_cost["states"]),
                        )
                        metrics.update(
                            profile=str(config.get("profile", "unspecified")),
                            training_algorithm="hidden_belief_rank1_gain_frozen_ei",
                            gate_fit_supervision=fitted.fit_metadata.get(
                                "gate_fit_supervision"
                            ),
                            gate_received_true_q_h=False,
                            random_tape_id=tape.fingerprint,
                            bridge_protocol_id=protocol_id,
                            network_n_units=network.n_units,
                            network_excitatory_fraction=network.excitatory_fraction,
                        )
                        emit_success(dimensions, metrics)
                    except Exception as error:
                        emit_failure(dimensions, error)

            if not np.array_equal(network.recurrent_weights, initial_weights):
                raise RuntimeError("frozen E/I receiver changed during bridge run")

        expected = {str(item["condition"]) for item in planned}
        if emitted != expected:
            raise RuntimeError(
                f"planned/emitted mismatch: missing={sorted(expected - emitted)}; "
                f"extra={sorted(emitted - expected)}"
            )
        return run.path


def main() -> None:
    args = basic_parser(
        __doc__ or "Hidden belief to E/I gain bridge",
        "configs/formal/exp10_hidden_context_ei_bridge.json",
    ).parse_args()
    config = load_json_config(args.config)
    for seed in seed_list(args.seeds or config["seeds"]):
        run_seed(config, seed, args.results_root)


if __name__ == "__main__":
    main()
