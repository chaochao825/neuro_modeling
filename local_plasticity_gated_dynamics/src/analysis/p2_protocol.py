"""Canonical preregistration contract for the formal P2 gate-only audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


FORMAL_P2_PROTOCOL: dict[str, Any] = {
    "profile": "formal",
    "seeds": list(range(30)),
    "cue_reliabilities": [0.55, 0.70, 0.85, 1.0],
    "context_hazards": [0.01, 0.05, 0.10, 0.20],
    "task": {
        "n_episodes": 60,
        "trials_per_episode": 200,
        "dt_ms": 100,
        "cue_ms": 100,
        "sensory_ms": 400,
        "delay_ms": 100,
        "response_ms": 100,
        "coherence_values": [-0.5, -0.25, 0.25, 0.5],
        "sensory_noise_std": 1.0,
        "input_scale": 1.0,
        "response_target": "choice",
    },
    "outer_test_fraction": 1.0 / 3.0,
    "validation_fraction": 0.25,
    "learned_hmm": {
        "max_iterations": 100,
        "tolerance": 1e-8,
        "min_probability": 1e-6,
    },
    "md_gate": {
        "learning_rate": 0.03,
        "inverse_temperature": 1.2,
        "pseudocount": 0.05,
        "n_passes": 2,
    },
    "supervised_gate": {"ridge": 0.001},
    "switch_metrics": {
        "max_latency": 5,
        "sustain_trials": 2,
        "posterior_threshold": 0.8,
        "minimum_state_duration": 5,
        "match_tolerance": 1,
        "minimum_eligible_switches": 20,
    },
    "interventions": {"delay_trials": 1},
    "gate_grid": {
        "base": [
            "oracle_bayes",
            "supervised_upper_bound",
            "learned_hmm",
            "md_recurrent_belief",
            "no_gate",
        ],
        "md_interventions": ["clamp", "delay", "shuffle"],
    },
    "md_algorithm": "causal_two_slice_with_hebbian_moment_shrinkage_v1",
}


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _critical_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("P2 config must be a mapping")
    payload = {
        key: config[key]
        for key in (
            "profile",
            "seeds",
            "cue_reliabilities",
            "context_hazards",
            "task",
            "outer_test_fraction",
            "validation_fraction",
            "learned_hmm",
            "md_gate",
            "supervised_gate",
            "switch_metrics",
            "interventions",
        )
    }
    payload["gate_grid"] = FORMAL_P2_PROTOCOL["gate_grid"]
    payload["md_algorithm"] = FORMAL_P2_PROTOCOL["md_algorithm"]
    # JSON normalization removes tuple/list and NumPy scalar ambiguity while
    # failing on non-serializable or non-finite protocol values.
    return json.loads(_canonical_bytes(payload).decode("ascii"))


def p2_protocol_id(config: Mapping[str, Any]) -> str:
    """Return the content ID of every experiment-critical P2 setting."""

    return hashlib.sha256(_canonical_bytes(_critical_payload(config))).hexdigest()


FORMAL_P2_PROTOCOL_ID = hashlib.sha256(_canonical_bytes(FORMAL_P2_PROTOCOL)).hexdigest()


def validate_formal_p2_protocol(config: Mapping[str, Any]) -> None:
    """Fail closed if a formal run differs from the fixed preregistration."""

    actual = _critical_payload(config)
    if actual == FORMAL_P2_PROTOCOL:
        return
    differing = sorted(
        key for key in FORMAL_P2_PROTOCOL if actual.get(key) != FORMAL_P2_PROTOCOL[key]
    )
    raise ValueError(
        "formal P2 config differs from the preregistered protocol: "
        + ", ".join(differing)
    )


__all__ = [
    "FORMAL_P2_PROTOCOL",
    "FORMAL_P2_PROTOCOL_ID",
    "p2_protocol_id",
    "validate_formal_p2_protocol",
]
