"""Leakage-safe linear tasks for actuator-matching experiments.

The carrier is a frozen, strictly stable high-rank matrix obeying Dale column
signs.  A task adds independently scaled, rank-controlled state and input
demands.  Random trial tapes are keyed by generator and split rather than by a
grid position, so extending an experiment grid cannot alter existing data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from src.utils.reproducibility import derive_seed


Array = np.ndarray
SplitName = Literal["train", "validation", "test"]
_ALLOWED_STATE_RANKS = frozenset((1, 2, 4, 8))
_ALLOWED_INPUT_RANKS = frozenset((1, 2, 4))
_SPLIT_OFFSETS: dict[str, int] = {
    "train": 0,
    "validation": 1_000_000,
    "test": 2_000_000,
}


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
    strict_maximum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        if invalid:
            relation = "greater than" if strict_minimum else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None:
        invalid = result >= maximum if strict_maximum else result > maximum
        if invalid:
            relation = "less than" if strict_maximum else "at most"
            raise ValueError(f"{name} must be {relation} {maximum}")
    return result


def _bounds(value: object, *, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain two real numbers")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must contain two real numbers") from error
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    low = _real(values[0], name=f"{name}[0]")
    high = _real(values[1], name=f"{name}[1]")
    if low >= high:
        raise ValueError(f"{name} must be strictly increasing")
    return low, high


def _array(
    value: object,
    *,
    name: str,
    dtype: type[float] | type[int],
    shape: tuple[int, ...] | None = None,
) -> Array:
    raw = np.asarray(value)
    if dtype is int and (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise TypeError(f"{name} must contain integers")
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _fingerprint(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        elif hasattr(value, "__dataclass_fields__"):
            digest.update(
                json.dumps(
                    asdict(value), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _spectral_radius(matrix: Array) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def _orthogonal_basis(rows: int, columns: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(rows, columns))
    q, r = np.linalg.qr(raw, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs[None, :]


@dataclass(frozen=True, slots=True)
class CarrierConfig:
    """Configuration of the frozen Dale-compatible carrier."""

    n_neurons: int = 32
    n_inputs: int = 4
    n_outputs: int = 1
    inhibitory_fraction: float = 0.2
    spectral_radius: float = 0.75
    input_scale: float = 0.4

    def __post_init__(self) -> None:
        for name, minimum in (
            ("n_neurons", 8),
            ("n_inputs", 1),
            ("n_outputs", 1),
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name=name, minimum=minimum)
            )
        if self.n_outputs > self.n_neurons:
            raise ValueError("n_outputs cannot exceed n_neurons")
        inhibitory_fraction = _real(
            self.inhibitory_fraction,
            name="inhibitory_fraction",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
            strict_maximum=True,
        )
        n_inhibitory = int(round(self.n_neurons * inhibitory_fraction))
        if not 1 <= n_inhibitory < self.n_neurons:
            raise ValueError(
                "inhibitory_fraction must allocate at least one E and one I column"
            )
        object.__setattr__(self, "inhibitory_fraction", inhibitory_fraction)
        object.__setattr__(
            self,
            "spectral_radius",
            _real(
                self.spectral_radius,
                name="spectral_radius",
                minimum=0.0,
                maximum=1.0,
                strict_minimum=True,
                strict_maximum=True,
            ),
        )
        object.__setattr__(
            self,
            "input_scale",
            _real(
                self.input_scale,
                name="input_scale",
                minimum=0.0,
                strict_minimum=True,
            ),
        )

    @property
    def n_inhibitory(self) -> int:
        return int(round(self.n_neurons * self.inhibitory_fraction))

    @property
    def n_excitatory(self) -> int:
        return self.n_neurons - self.n_inhibitory


ActuatorCarrierConfig = CarrierConfig


@dataclass(frozen=True, slots=True, eq=False)
class ActuatorCarrier:
    """Frozen high-rank physical carrier and observation map."""

    config: CarrierConfig
    seed: int
    a0: Array
    b0: Array
    c: Array
    dale_signs: Array
    spectral_radius: float = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, CarrierConfig):
            raise TypeError("config must be a CarrierConfig")
        seed = _integer(self.seed, name="seed")
        n = self.config.n_neurons
        m = self.config.n_inputs
        p = self.config.n_outputs
        a0 = _array(self.a0, name="a0", dtype=float, shape=(n, n))
        b0 = _array(self.b0, name="b0", dtype=float, shape=(n, m))
        c = _array(self.c, name="c", dtype=float, shape=(p, n))
        signs = _array(self.dale_signs, name="dale_signs", dtype=int, shape=(n,))
        expected = np.ones(n, dtype=int)
        expected[self.config.n_excitatory :] = -1
        if not np.array_equal(signs, expected):
            raise ValueError("dale_signs do not match the configured E/I partition")
        signed = a0 * signs[None, :]
        if np.any(signed < 0.0) or np.any(np.max(signed, axis=0) <= 0.0):
            raise ValueError("a0 must obey non-degenerate Dale column signs")
        if np.linalg.matrix_rank(a0) != n:
            raise ValueError("a0 must be high rank (full numerical rank)")
        if np.linalg.matrix_rank(b0) != min(n, m):
            raise ValueError("b0 must have full attainable rank")
        if np.linalg.matrix_rank(c) != p:
            raise ValueError("c must have full row rank")
        radius = _spectral_radius(a0)
        if not radius < 1.0:
            raise ValueError("a0 must be strictly stable")
        for name, value in (
            ("seed", seed),
            ("a0", a0),
            ("b0", b0),
            ("c", c),
            ("dale_signs", signs),
            ("spectral_radius", radius),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint(self.config, seed, a0, b0, c, signs),
        )


def make_carrier(config: CarrierConfig, seed: int) -> ActuatorCarrier:
    """Create one deterministic, frozen Dale carrier."""

    if not isinstance(config, CarrierConfig):
        raise TypeError("config must be a CarrierConfig")
    seed = _integer(seed, name="seed")
    n = config.n_neurons
    signs = np.ones(n, dtype=int)
    signs[config.n_excitatory :] = -1

    a0: Array | None = None
    for attempt in range(16):
        rng = np.random.default_rng(
            derive_seed(seed, "actuator-matching", "carrier", "a0", attempt)
        )
        candidate = np.abs(rng.normal(size=(n, n))) * signs[None, :]
        if np.linalg.matrix_rank(candidate) == n:
            radius = _spectral_radius(candidate)
            if radius > 0.0:
                a0 = candidate * (config.spectral_radius / radius)
                break
    if a0 is None:  # pragma: no cover - full-rank Gaussian failure is measure zero
        raise RuntimeError("failed to construct a full-rank Dale carrier")

    b_rng = np.random.default_rng(
        derive_seed(seed, "actuator-matching", "carrier", "b0")
    )
    b0 = (
        config.input_scale
        * b_rng.normal(size=(n, config.n_inputs))
        / np.sqrt(config.n_inputs)
    )
    c_basis = _orthogonal_basis(
        n,
        config.n_outputs,
        derive_seed(seed, "actuator-matching", "carrier", "c"),
    )
    return ActuatorCarrier(config, seed, a0, b0, c_basis.T, signs)


@dataclass(frozen=True, slots=True, eq=False)
class ActuatorTaskSpec:
    """One rank-controlled task family on a frozen carrier.

    Context index 0 is ``s=-1`` and index 1 is ``s=+1``.
    ``delta_a`` is the stability-adjusted demand; the unadjusted draw and the
    common adjustment factor remain available for audit.
    """

    carrier: ActuatorCarrier
    alpha: float
    rA: int
    rB: int
    delay: int
    noise: float
    rotation_seed: int
    generator_id: str
    stability_limit: float
    delta_a_amplitude: float
    delta_b_amplitude: float
    stability_shrink: float
    delta_a_unshrunk: Array
    delta_a: Array
    delta_b: Array
    a_context: Array
    b_context: Array
    unshrunk_spectral_radii: Array
    spectral_radii: Array
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.carrier, ActuatorCarrier):
            raise TypeError("carrier must be an ActuatorCarrier")
        alpha = _real(self.alpha, name="alpha", minimum=0.0, maximum=1.0)
        r_a = _integer(self.rA, name="rA", minimum=1)
        r_b = _integer(self.rB, name="rB", minimum=1)
        if r_a not in _ALLOWED_STATE_RANKS:
            raise ValueError(f"rA must be one of {sorted(_ALLOWED_STATE_RANKS)}")
        if r_b not in _ALLOWED_INPUT_RANKS:
            raise ValueError(f"rB must be one of {sorted(_ALLOWED_INPUT_RANKS)}")
        n = self.carrier.config.n_neurons
        m = self.carrier.config.n_inputs
        if r_a > n:
            raise ValueError("rA cannot exceed n_neurons")
        if r_b > min(n, m):
            raise ValueError("rB cannot exceed the attainable input-map rank")
        delay = _integer(self.delay, name="delay", minimum=0)
        noise = _real(self.noise, name="noise", minimum=0.0)
        rotation_seed = _integer(self.rotation_seed, name="rotation_seed")
        if not isinstance(self.generator_id, str):
            raise TypeError("generator_id must be a string")
        generator_id = self.generator_id.strip()
        if not generator_id:
            raise ValueError("generator_id must be non-empty")
        stability_limit = _real(
            self.stability_limit,
            name="stability_limit",
            minimum=self.carrier.spectral_radius,
            maximum=1.0,
            strict_minimum=True,
            strict_maximum=True,
        )
        amplitude_a = _real(
            self.delta_a_amplitude,
            name="delta_a_amplitude",
            minimum=0.0,
            strict_minimum=True,
        )
        amplitude_b = _real(
            self.delta_b_amplitude,
            name="delta_b_amplitude",
            minimum=0.0,
            strict_minimum=True,
        )
        shrink = _real(
            self.stability_shrink,
            name="stability_shrink",
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
        )
        raw_a = _array(
            self.delta_a_unshrunk,
            name="delta_a_unshrunk",
            dtype=float,
            shape=(n, n),
        )
        delta_a = _array(self.delta_a, name="delta_a", dtype=float, shape=(n, n))
        delta_b = _array(self.delta_b, name="delta_b", dtype=float, shape=(n, m))
        a_context = _array(
            self.a_context, name="a_context", dtype=float, shape=(2, n, n)
        )
        b_context = _array(
            self.b_context, name="b_context", dtype=float, shape=(2, n, m)
        )
        raw_radii = _array(
            self.unshrunk_spectral_radii,
            name="unshrunk_spectral_radii",
            dtype=float,
            shape=(2,),
        )
        radii = _array(
            self.spectral_radii,
            name="spectral_radii",
            dtype=float,
            shape=(2,),
        )
        if np.linalg.matrix_rank(raw_a) != r_a or np.linalg.matrix_rank(delta_a) != r_a:
            raise ValueError("delta_a matrices must have exact numerical rank rA")
        if np.linalg.matrix_rank(delta_b) != r_b:
            raise ValueError("delta_b must have exact numerical rank rB")
        if not np.allclose(delta_a, shrink * raw_a, rtol=1e-11, atol=1e-13):
            raise ValueError("delta_a must equal stability_shrink * delta_a_unshrunk")
        contexts = np.array([-1.0, 1.0])
        expected_a = np.stack(
            [self.carrier.a0 + 0.5 * s * alpha * delta_a for s in contexts]
        )
        expected_b = np.stack(
            [self.carrier.b0 + 0.5 * s * (1.0 - alpha) * delta_b for s in contexts]
        )
        if not np.allclose(a_context, expected_a, rtol=1e-11, atol=1e-13):
            raise ValueError("a_context does not implement the centered task equation")
        if not np.allclose(b_context, expected_b, rtol=1e-11, atol=1e-13):
            raise ValueError("b_context does not implement the centered task equation")
        observed_radii = np.array([_spectral_radius(value) for value in a_context])
        raw_expected = np.stack(
            [self.carrier.a0 + 0.5 * s * alpha * raw_a for s in contexts]
        )
        observed_raw = np.array([_spectral_radius(value) for value in raw_expected])
        if not np.allclose(radii, observed_radii, rtol=1e-10, atol=1e-12):
            raise ValueError("spectral_radii do not match a_context")
        if not np.allclose(raw_radii, observed_raw, rtol=1e-10, atol=1e-12):
            raise ValueError("unshrunk_spectral_radii do not match the raw demand")
        if np.any(radii > stability_limit + 1e-10) or np.any(radii >= 1.0):
            raise ValueError("both context dynamics must be strictly stable")
        replacements = {
            "alpha": alpha,
            "rA": r_a,
            "rB": r_b,
            "delay": delay,
            "noise": noise,
            "rotation_seed": rotation_seed,
            "generator_id": generator_id,
            "stability_limit": stability_limit,
            "delta_a_amplitude": amplitude_a,
            "delta_b_amplitude": amplitude_b,
            "stability_shrink": shrink,
            "delta_a_unshrunk": raw_a,
            "delta_a": delta_a,
            "delta_b": delta_b,
            "a_context": a_context,
            "b_context": b_context,
            "unshrunk_spectral_radii": raw_radii,
            "spectral_radii": radii,
        }
        for name, value in replacements.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint(
                self.carrier.fingerprint,
                alpha,
                r_a,
                r_b,
                delay,
                noise,
                rotation_seed,
                generator_id,
                stability_limit,
                amplitude_a,
                amplitude_b,
                shrink,
                raw_a,
                delta_a,
                delta_b,
                a_context,
                b_context,
            ),
        )

    def context_index(self, context: int) -> int:
        value = _integer(abs(context), name="abs(context)", minimum=1)
        if value != 1 or isinstance(context, (bool, np.bool_)):
            raise ValueError("context must be -1 or +1")
        return 0 if int(context) == -1 else 1

    def matrices(self, context: int) -> tuple[Array, Array]:
        index = self.context_index(context)
        return self.a_context[index], self.b_context[index]


def _rank_demand(
    *,
    rows: int,
    columns: int,
    rank: int,
    amplitude: float,
    root_seed: int,
    generator_id: str,
    rotation_seed: int,
    label: str,
) -> Array:
    left = _orthogonal_basis(
        rows,
        rank,
        derive_seed(
            root_seed, generator_id, "rotation", rotation_seed, label, "left"
        ),
    )
    right = _orthogonal_basis(
        columns,
        rank,
        derive_seed(
            root_seed, generator_id, "rotation", rotation_seed, label, "right"
        ),
    )
    singular_values = amplitude * np.geomspace(1.0, 0.55, num=rank)
    return (left * singular_values[None, :]) @ right.T


def _common_stability_shrink(
    carrier: ActuatorCarrier,
    delta_a: Array,
    *,
    alpha: float,
    limit: float,
) -> float:
    def maximum_radius(scale: float) -> float:
        return max(
            _spectral_radius(carrier.a0 - 0.5 * alpha * scale * delta_a),
            _spectral_radius(carrier.a0 + 0.5 * alpha * scale * delta_a),
        )

    if maximum_radius(1.0) <= limit:
        return 1.0
    low = 0.0
    high = 1.0
    for _ in range(48):
        middle = 0.5 * (low + high)
        if maximum_radius(middle) <= limit:
            low = middle
        else:
            high = middle
    shrink = low * (1.0 - 1e-12)
    if shrink <= 0.0:  # protected by carrier radius < limit
        raise RuntimeError("failed to find a positive common stability shrink")
    return shrink


def make_task_spec(
    carrier: ActuatorCarrier,
    *,
    alpha: float,
    rA: int,
    rB: int,
    delay: int,
    noise: float,
    rotation_seed: int,
    generator_id: str = "default",
    delta_a_log10_range: tuple[float, float] = (-1.0, 0.15),
    delta_b_log10_range: tuple[float, float] = (-1.0, 0.15),
    stability_limit: float = 0.98,
) -> ActuatorTaskSpec:
    """Create independently scaled rank-controlled task perturbations."""

    if not isinstance(carrier, ActuatorCarrier):
        raise TypeError("carrier must be an ActuatorCarrier")
    alpha = _real(alpha, name="alpha", minimum=0.0, maximum=1.0)
    r_a = _integer(rA, name="rA", minimum=1)
    r_b = _integer(rB, name="rB", minimum=1)
    if r_a not in _ALLOWED_STATE_RANKS or r_a > carrier.config.n_neurons:
        raise ValueError(
            f"rA must be in {sorted(_ALLOWED_STATE_RANKS)} and not exceed n_neurons"
        )
    if r_b not in _ALLOWED_INPUT_RANKS or r_b > min(
        carrier.config.n_neurons, carrier.config.n_inputs
    ):
        raise ValueError(
            f"rB must be in {sorted(_ALLOWED_INPUT_RANKS)} and attainable by b0"
        )
    delay = _integer(delay, name="delay", minimum=0)
    noise = _real(noise, name="noise", minimum=0.0)
    rotation_seed = _integer(rotation_seed, name="rotation_seed")
    if not isinstance(generator_id, str):
        raise TypeError("generator_id must be a string")
    generator_id = generator_id.strip()
    if not generator_id:
        raise ValueError("generator_id must be non-empty")
    a_bounds = _bounds(delta_a_log10_range, name="delta_a_log10_range")
    b_bounds = _bounds(delta_b_log10_range, name="delta_b_log10_range")
    stability_limit = _real(
        stability_limit,
        name="stability_limit",
        minimum=carrier.spectral_radius,
        maximum=1.0,
        strict_minimum=True,
        strict_maximum=True,
    )
    a_rng = np.random.default_rng(
        derive_seed(
            carrier.seed, generator_id, "rotation", rotation_seed, "amplitude-a"
        )
    )
    b_rng = np.random.default_rng(
        derive_seed(
            carrier.seed, generator_id, "rotation", rotation_seed, "amplitude-b"
        )
    )
    amplitude_a = float(10.0 ** a_rng.uniform(*a_bounds))
    amplitude_b = float(10.0 ** b_rng.uniform(*b_bounds))
    raw_a = _rank_demand(
        rows=carrier.config.n_neurons,
        columns=carrier.config.n_neurons,
        rank=r_a,
        amplitude=amplitude_a,
        root_seed=carrier.seed,
        generator_id=generator_id,
        rotation_seed=rotation_seed,
        label="delta-a",
    )
    delta_b = _rank_demand(
        rows=carrier.config.n_neurons,
        columns=carrier.config.n_inputs,
        rank=r_b,
        amplitude=amplitude_b,
        root_seed=carrier.seed,
        generator_id=generator_id,
        rotation_seed=rotation_seed,
        label="delta-b",
    )
    shrink = _common_stability_shrink(
        carrier, raw_a, alpha=alpha, limit=stability_limit
    )
    delta_a = shrink * raw_a
    context_values = (-1.0, 1.0)
    raw_context = np.stack(
        [carrier.a0 + 0.5 * s * alpha * raw_a for s in context_values]
    )
    a_context = np.stack(
        [carrier.a0 + 0.5 * s * alpha * delta_a for s in context_values]
    )
    b_context = np.stack(
        [
            carrier.b0 + 0.5 * s * (1.0 - alpha) * delta_b
            for s in context_values
        ]
    )
    raw_radii = np.array([_spectral_radius(value) for value in raw_context])
    radii = np.array([_spectral_radius(value) for value in a_context])
    return ActuatorTaskSpec(
        carrier=carrier,
        alpha=alpha,
        rA=r_a,
        rB=r_b,
        delay=delay,
        noise=noise,
        rotation_seed=rotation_seed,
        generator_id=generator_id,
        stability_limit=stability_limit,
        delta_a_amplitude=amplitude_a,
        delta_b_amplitude=amplitude_b,
        stability_shrink=shrink,
        delta_a_unshrunk=raw_a,
        delta_a=delta_a,
        delta_b=delta_b,
        a_context=a_context,
        b_context=b_context,
        unshrunk_spectral_radii=raw_radii,
        spectral_radii=radii,
    )


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Whole-block split sizes and input epoch configuration."""

    n_train_blocks: int = 6
    n_validation_blocks: int = 2
    n_test_blocks: int = 4
    trials_per_block: int = 8
    input_steps: int = 4
    input_std: float = 1.0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("n_train_blocks", 1),
            ("n_validation_blocks", 1),
            ("n_test_blocks", 1),
            ("trials_per_block", 2),
            ("input_steps", 1),
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name=name, minimum=minimum)
            )
        if self.trials_per_block % 2:
            raise ValueError("trials_per_block must be even for antithetic pairs")
        object.__setattr__(
            self,
            "input_std",
            _real(
                self.input_std,
                name="input_std",
                minimum=0.0,
                strict_minimum=True,
            ),
        )


ActuatorDatasetConfig = DatasetConfig


@dataclass(frozen=True, slots=True, eq=False)
class ActuatorTaskSplit:
    """Immutable complete trials from one whole-block split."""

    split_name: SplitName
    spec_fingerprint: str
    input_steps: int
    delay: int
    target_states: Array
    inputs: Array
    contexts: Array
    labels: Array
    block_ids: Array
    trial_ids: Array
    noise: Array
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.split_name not in _SPLIT_OFFSETS:
            raise ValueError("split_name must be train, validation, or test")
        if not isinstance(self.spec_fingerprint, str) or not self.spec_fingerprint:
            raise ValueError("spec_fingerprint must be non-empty")
        input_steps = _integer(self.input_steps, name="input_steps", minimum=1)
        delay = _integer(self.delay, name="delay", minimum=0)
        raw_inputs = np.asarray(self.inputs)
        if raw_inputs.ndim != 3:
            raise ValueError("inputs must have shape [trial, step, input]")
        n_trials, n_steps, n_inputs = raw_inputs.shape
        if n_trials < 2 or n_inputs < 1 or n_steps != input_steps + delay:
            raise ValueError("inputs do not match input_steps + delay")
        inputs = _array(
            self.inputs,
            name="inputs",
            dtype=float,
            shape=(n_trials, n_steps, n_inputs),
        )
        raw_states = np.asarray(self.target_states)
        if raw_states.ndim != 3 or raw_states.shape[:2] != (
            n_trials,
            n_steps + 1,
        ):
            raise ValueError(
                "target_states must have shape [trial, step + 1, neuron]"
            )
        n_neurons = raw_states.shape[2]
        states = _array(
            self.target_states,
            name="target_states",
            dtype=float,
            shape=(n_trials, n_steps + 1, n_neurons),
        )
        process_noise = _array(
            self.noise,
            name="noise",
            dtype=float,
            shape=(n_trials, n_steps, n_neurons),
        )
        contexts = _array(
            self.contexts, name="contexts", dtype=int, shape=(n_trials,)
        )
        labels = _array(self.labels, name="labels", dtype=int, shape=(n_trials,))
        blocks = _array(
            self.block_ids, name="block_ids", dtype=int, shape=(n_trials,)
        )
        trials = _array(
            self.trial_ids, name="trial_ids", dtype=int, shape=(n_trials,)
        )
        if not np.isin(contexts, [-1, 1]).all():
            raise ValueError("contexts must contain only -1 and +1")
        if not np.isin(labels, [-1, 1]).all():
            raise ValueError("labels must contain only -1 and +1")
        if np.any(blocks < 0) or np.any(trials < 0):
            raise ValueError("block_ids and trial_ids must be non-negative")
        if np.unique(trials).size != n_trials:
            raise ValueError("trial_ids must be unique within a split")
        if not np.array_equal(states[:, 0], np.zeros_like(states[:, 0])):
            raise ValueError("target_states must start from the zero state")
        if np.any(inputs[:, input_steps:] != 0.0):
            raise ValueError("all inputs must be zero throughout the delay")
        for block_id in np.unique(blocks):
            selected = blocks == block_id
            if np.unique(contexts[selected]).size != 1:
                raise ValueError("each block must contain exactly one context")
            block_labels = labels[selected]
            if np.sum(block_labels == -1) != np.sum(block_labels == 1):
                raise ValueError("each block must have antithetically balanced labels")
        replacements = {
            "input_steps": input_steps,
            "delay": delay,
            "target_states": states,
            "inputs": inputs,
            "contexts": contexts,
            "labels": labels,
            "block_ids": blocks,
            "trial_ids": trials,
            "noise": process_noise,
        }
        for name, value in replacements.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint(
                self.split_name,
                self.spec_fingerprint,
                input_steps,
                delay,
                states,
                inputs,
                contexts,
                labels,
                blocks,
                trials,
                process_noise,
            ),
        )

    @property
    def n_steps(self) -> int:
        return self.inputs.shape[1]


@dataclass(frozen=True, slots=True, eq=False)
class ActuatorMatchingDataset:
    """Train/validation/test splits with disjoint whole blocks."""

    spec: ActuatorTaskSpec
    config: DatasetConfig
    seed: int
    train: ActuatorTaskSplit
    validation: ActuatorTaskSplit
    test: ActuatorTaskSplit
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ActuatorTaskSpec):
            raise TypeError("spec must be an ActuatorTaskSpec")
        if not isinstance(self.config, DatasetConfig):
            raise TypeError("config must be a DatasetConfig")
        seed = _integer(self.seed, name="seed")
        expected = (
            ("train", self.train, self.config.n_train_blocks),
            ("validation", self.validation, self.config.n_validation_blocks),
            ("test", self.test, self.config.n_test_blocks),
        )
        seen_blocks: set[int] = set()
        seen_trials: set[int] = set()
        for name, split, n_blocks in expected:
            if not isinstance(split, ActuatorTaskSplit):
                raise TypeError(f"{name} must be an ActuatorTaskSplit")
            if split.split_name != name or split.spec_fingerprint != self.spec.fingerprint:
                raise ValueError(f"{name} does not belong to this task specification")
            if np.unique(split.block_ids).size != n_blocks:
                raise ValueError(f"{name} has the wrong number of whole blocks")
            if split.inputs.shape[0] != n_blocks * self.config.trials_per_block:
                raise ValueError(f"{name} has the wrong number of trials")
            if split.input_steps != self.config.input_steps or split.delay != self.spec.delay:
                raise ValueError(f"{name} has inconsistent epoch lengths")
            blocks = set(split.block_ids.tolist())
            trials = set(split.trial_ids.tolist())
            if seen_blocks.intersection(blocks) or seen_trials.intersection(trials):
                raise ValueError("splits must have disjoint block and trial identifiers")
            seen_blocks.update(blocks)
            seen_trials.update(trials)
            self._validate_rollout(split)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(
            self,
            "fingerprint",
            _fingerprint(
                self.spec.fingerprint,
                self.config,
                seed,
                self.train.fingerprint,
                self.validation.fingerprint,
                self.test.fingerprint,
            ),
        )

    def _validate_rollout(self, split: ActuatorTaskSplit) -> None:
        states = split.target_states
        for context in (-1, 1):
            selected = np.flatnonzero(split.contexts == context)
            if not selected.size:
                continue
            a_matrix, b_matrix = self.spec.matrices(context)
            for step in range(split.n_steps):
                expected = (
                    states[selected, step] @ a_matrix.T
                    + split.inputs[selected, step] @ b_matrix.T
                    + split.noise[selected, step]
                )
                if not np.allclose(
                    states[selected, step + 1], expected, rtol=1e-11, atol=1e-12
                ):
                    raise ValueError("target_states do not match the registered rollout")
        scores = states[:, -1] @ self.spec.carrier.c[0]
        if np.any(scores == 0.0) or not np.array_equal(
            np.sign(scores).astype(int), split.labels
        ):
            raise ValueError("labels must be the final linear readout signs")


ActuatorDataset = ActuatorMatchingDataset


def _block_contexts(seed: int, generator_id: str, split: str, n_blocks: int) -> Array:
    base = np.resize(np.array([-1, 1], dtype=int), n_blocks)
    rng = np.random.default_rng(
        derive_seed(seed, generator_id, "dataset", split, "block-contexts")
    )
    return base[rng.permutation(n_blocks)]


def _make_split(
    spec: ActuatorTaskSpec,
    config: DatasetConfig,
    *,
    split_name: SplitName,
    n_blocks: int,
    seed: int,
) -> ActuatorTaskSplit:
    n_trials = n_blocks * config.trials_per_block
    n_steps = config.input_steps + spec.delay
    n = spec.carrier.config.n_neurons
    m = spec.carrier.config.n_inputs
    inputs = np.zeros((n_trials, n_steps, m), dtype=float)
    process_noise = np.zeros((n_trials, n_steps, n), dtype=float)
    contexts = np.empty(n_trials, dtype=int)
    blocks = np.empty(n_trials, dtype=int)
    context_by_block = _block_contexts(
        seed, spec.generator_id, split_name, n_blocks
    )
    pairs_per_block = config.trials_per_block // 2
    for local_block in range(n_blocks):
        block_start = local_block * config.trials_per_block
        contexts[block_start : block_start + config.trials_per_block] = (
            context_by_block[local_block]
        )
        blocks[block_start : block_start + config.trials_per_block] = (
            _SPLIT_OFFSETS[split_name] + local_block
        )
        for pair in range(pairs_per_block):
            first = block_start + 2 * pair
            input_rng = np.random.default_rng(
                derive_seed(
                    seed,
                    spec.generator_id,
                    "dataset",
                    split_name,
                    "block",
                    local_block,
                    "pair",
                    pair,
                    "input",
                )
            )
            noise_rng = np.random.default_rng(
                derive_seed(
                    seed,
                    spec.generator_id,
                    "dataset",
                    split_name,
                    "block",
                    local_block,
                    "pair",
                    pair,
                    "process-noise",
                )
            )
            input_tape = config.input_std * input_rng.normal(
                size=(config.input_steps, m)
            )
            noise_tape = spec.noise * noise_rng.normal(size=(n_steps, n))
            inputs[first, : config.input_steps] = input_tape
            inputs[first + 1, : config.input_steps] = -input_tape
            process_noise[first] = noise_tape
            process_noise[first + 1] = -noise_tape

    states = np.zeros((n_trials, n_steps + 1, n), dtype=float)
    for context in (-1, 1):
        selected = np.flatnonzero(contexts == context)
        if not selected.size:
            continue
        a_matrix, b_matrix = spec.matrices(context)
        for step in range(n_steps):
            states[selected, step + 1] = (
                states[selected, step] @ a_matrix.T
                + inputs[selected, step] @ b_matrix.T
                + process_noise[selected, step]
            )
    scores = states[:, -1] @ spec.carrier.c[0]
    if np.any(scores == 0.0):
        raise RuntimeError("a generated trial has an undefined zero linear label")
    labels = np.sign(scores).astype(int)
    for local_block in range(n_blocks):
        selected = blocks == _SPLIT_OFFSETS[split_name] + local_block
        if np.sum(labels[selected] == -1) != pairs_per_block:
            raise RuntimeError("antithetic generation failed to balance a block")
    trial_ids = _SPLIT_OFFSETS[split_name] + np.arange(n_trials, dtype=int)
    return ActuatorTaskSplit(
        split_name=split_name,
        spec_fingerprint=spec.fingerprint,
        input_steps=config.input_steps,
        delay=spec.delay,
        target_states=states,
        inputs=inputs,
        contexts=contexts,
        labels=labels,
        block_ids=blocks,
        trial_ids=trial_ids,
        noise=process_noise,
    )


def make_dataset(
    spec: ActuatorTaskSpec,
    config: DatasetConfig | None = None,
    *,
    seed: int | None = None,
) -> ActuatorMatchingDataset:
    """Generate deterministic whole-block train/validation/test data.

    Inputs and process-noise tapes depend on ``seed``, ``generator_id`` and
    named split/block/pair labels only.  They deliberately do not depend on
    alpha, demand ranks, or grid enumeration order.
    """

    if not isinstance(spec, ActuatorTaskSpec):
        raise TypeError("spec must be an ActuatorTaskSpec")
    if config is None:
        config = DatasetConfig()
    if not isinstance(config, DatasetConfig):
        raise TypeError("config must be a DatasetConfig")
    dataset_seed = spec.carrier.seed if seed is None else _integer(seed, name="seed")
    train = _make_split(
        spec,
        config,
        split_name="train",
        n_blocks=config.n_train_blocks,
        seed=dataset_seed,
    )
    validation = _make_split(
        spec,
        config,
        split_name="validation",
        n_blocks=config.n_validation_blocks,
        seed=dataset_seed,
    )
    test = _make_split(
        spec,
        config,
        split_name="test",
        n_blocks=config.n_test_blocks,
        seed=dataset_seed,
    )
    return ActuatorMatchingDataset(
        spec=spec,
        config=config,
        seed=dataset_seed,
        train=train,
        validation=validation,
        test=test,
    )


def make_actuator_matching_train_split(
    spec: ActuatorTaskSpec,
    config: DatasetConfig | None = None,
    *,
    seed: int | None = None,
) -> ActuatorTaskSplit:
    """Generate only the registered training blocks for strict preflights.

    This is not a view over :func:`make_dataset`: validation and test split
    factories are never invoked.  It shares the exact labelled ``_make_split``
    path used by the full dataset, so its training fingerprint is identical.
    """

    if not isinstance(spec, ActuatorTaskSpec):
        raise TypeError("spec must be an ActuatorTaskSpec")
    if config is None:
        config = DatasetConfig()
    if not isinstance(config, DatasetConfig):
        raise TypeError("config must be a DatasetConfig")
    dataset_seed = spec.carrier.seed if seed is None else _integer(seed, name="seed")
    return _make_split(
        spec,
        config,
        split_name="train",
        n_blocks=config.n_train_blocks,
        seed=dataset_seed,
    )


__all__ = [
    "ActuatorCarrier",
    "ActuatorCarrierConfig",
    "ActuatorDataset",
    "ActuatorDatasetConfig",
    "ActuatorMatchingDataset",
    "ActuatorTaskSpec",
    "ActuatorTaskSplit",
    "CarrierConfig",
    "DatasetConfig",
    "make_carrier",
    "make_actuator_matching_train_split",
    "make_dataset",
    "make_task_spec",
]
