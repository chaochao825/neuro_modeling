"""Tests for the continuous actuator-matching task generator."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.tasks import actuator_matching as actuator_task
from src.tasks.actuator_matching import (
    ActuatorCarrier,
    ActuatorMatchingDataset,
    CarrierConfig,
    DatasetConfig,
    make_actuator_matching_train_split,
    make_carrier,
    make_dataset,
    make_task_spec,
)


def _carrier(seed: int = 17) -> ActuatorCarrier:
    return make_carrier(
        CarrierConfig(
            n_neurons=20,
            n_inputs=4,
            n_outputs=2,
            inhibitory_fraction=0.2,
            spectral_radius=0.72,
        ),
        seed,
    )


def _spec(
    carrier: ActuatorCarrier | None = None,
    *,
    alpha: float = 0.5,
    rA: int = 4,
    rB: int = 2,
    delay: int = 4,
    noise: float = 0.02,
    rotation_seed: int = 13,
    generator_id: str = "generator-3",
):
    carrier = _carrier() if carrier is None else carrier
    return make_task_spec(
        carrier,
        alpha=alpha,
        rA=rA,
        rB=rB,
        delay=delay,
        noise=noise,
        rotation_seed=rotation_seed,
        generator_id=generator_id,
    )


def _dataset_config(**overrides: object) -> DatasetConfig:
    values: dict[str, object] = {
        "n_train_blocks": 4,
        "n_validation_blocks": 2,
        "n_test_blocks": 2,
        "trials_per_block": 6,
        "input_steps": 3,
        "input_std": 0.7,
    }
    values.update(overrides)
    return DatasetConfig(**values)  # type: ignore[arg-type]


def test_carrier_is_full_rank_dale_compatible_and_strictly_stable() -> None:
    carrier = _carrier()
    n_excitatory = carrier.config.n_excitatory
    assert carrier.config.n_inhibitory == 4
    assert np.all(carrier.a0[:, :n_excitatory] >= 0.0)
    assert np.all(carrier.a0[:, n_excitatory:] <= 0.0)
    assert np.linalg.matrix_rank(carrier.a0) == carrier.config.n_neurons
    assert np.linalg.matrix_rank(carrier.b0) == carrier.config.n_inputs
    assert np.linalg.matrix_rank(carrier.c) == carrier.config.n_outputs
    assert carrier.spectral_radius == pytest.approx(0.72, abs=1e-12)
    assert carrier.spectral_radius < 1.0
    for array in (carrier.a0, carrier.b0, carrier.c, carrier.dale_signs):
        assert not array.flags.writeable


@pytest.mark.parametrize("rank_a", (1, 2, 4, 8))
@pytest.mark.parametrize("rank_b", (1, 2, 4))
def test_task_demands_have_exact_registered_ranks(
    rank_a: int, rank_b: int
) -> None:
    spec = _spec(rA=rank_a, rB=rank_b)
    assert np.linalg.matrix_rank(spec.delta_a_unshrunk) == rank_a
    assert np.linalg.matrix_rank(spec.delta_a) == rank_a
    assert np.linalg.matrix_rank(spec.delta_b) == rank_b
    assert spec.delta_a_amplitude > 0.0
    assert spec.delta_b_amplitude > 0.0
    # The A and B amplitudes are independent log-uniform draws, not jointly
    # normalized to make their weighted demands equal.
    assert spec.delta_a_amplitude != spec.delta_b_amplitude


@pytest.mark.parametrize("alpha", (0.0, 0.25, 0.5, 1.0))
def test_centered_context_equations_and_stability(alpha: float) -> None:
    spec = _spec(alpha=alpha)
    np.testing.assert_allclose(
        spec.a_context[1] - spec.a_context[0],
        alpha * spec.delta_a,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        0.5 * (spec.a_context[1] + spec.a_context[0]),
        spec.carrier.a0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        spec.b_context[1] - spec.b_context[0],
        (1.0 - alpha) * spec.delta_b,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        0.5 * (spec.b_context[1] + spec.b_context[0]),
        spec.carrier.b0,
        atol=1e-12,
    )
    assert np.all(spec.spectral_radii < 1.0)
    assert np.all(spec.spectral_radii <= spec.stability_limit + 1e-10)
    for array in (
        spec.delta_a_unshrunk,
        spec.delta_a,
        spec.delta_b,
        spec.a_context,
        spec.b_context,
        spec.spectral_radii,
    ):
        assert not array.flags.writeable


def test_unstable_raw_contexts_receive_one_deterministic_common_shrink() -> None:
    carrier = _carrier()
    kwargs = {
        "alpha": 1.0,
        "rA": 8,
        "rB": 4,
        "delay": 3,
        "noise": 0.0,
        "rotation_seed": 91,
        "generator_id": "forced-shrink",
        "delta_a_log10_range": (1.5, 1.5001),
        "stability_limit": 0.9,
    }
    first = make_task_spec(carrier, **kwargs)
    second = make_task_spec(carrier, **kwargs)
    assert np.max(first.unshrunk_spectral_radii) > first.stability_limit
    assert 0.0 < first.stability_shrink < 1.0
    assert first.stability_shrink == second.stability_shrink
    np.testing.assert_array_equal(first.delta_a, second.delta_a)
    np.testing.assert_allclose(
        first.delta_a,
        first.stability_shrink * first.delta_a_unshrunk,
        atol=1e-12,
    )
    assert np.all(first.spectral_radii <= first.stability_limit + 1e-10)
    assert np.linalg.matrix_rank(first.delta_a) == first.rA


def test_block_splits_and_label_pairs_are_deterministic_and_balanced() -> None:
    spec = _spec()
    config = _dataset_config()
    first = make_dataset(spec, config, seed=101)
    second = make_dataset(spec, config, seed=101)
    assert first.fingerprint == second.fingerprint
    assert first.spec.generator_id == "generator-3"
    assert {first.train.fingerprint, first.validation.fingerprint, first.test.fingerprint}
    for name in ("train", "validation", "test"):
        left = getattr(first, name)
        right = getattr(second, name)
        np.testing.assert_array_equal(left.inputs, right.inputs)
        np.testing.assert_array_equal(left.noise, right.noise)
        np.testing.assert_array_equal(left.target_states, right.target_states)
        for block in np.unique(left.block_ids):
            selected = np.flatnonzero(left.block_ids == block)
            assert np.unique(left.contexts[selected]).size == 1
            assert np.sum(left.labels[selected] == -1) == np.sum(
                left.labels[selected] == 1
            )
            for pair_start in selected[::2]:
                np.testing.assert_array_equal(
                    left.inputs[pair_start], -left.inputs[pair_start + 1]
                )
                np.testing.assert_array_equal(
                    left.noise[pair_start], -left.noise[pair_start + 1]
                )
                assert left.labels[pair_start] == -left.labels[pair_start + 1]

    train_blocks = set(first.train.block_ids.tolist())
    validation_blocks = set(first.validation.block_ids.tolist())
    test_blocks = set(first.test.block_ids.tolist())
    assert train_blocks.isdisjoint(validation_blocks)
    assert train_blocks.isdisjoint(test_blocks)
    assert validation_blocks.isdisjoint(test_blocks)


def test_delay_is_zero_input_and_process_noise_tape_remains_explicit() -> None:
    dataset = make_dataset(_spec(noise=0.05, delay=5), _dataset_config(), seed=27)
    split = dataset.train
    assert split.inputs.shape[1] == dataset.config.input_steps + dataset.spec.delay
    assert np.any(split.inputs[:, : dataset.config.input_steps] != 0.0)
    assert np.all(split.inputs[:, dataset.config.input_steps :] == 0.0)
    assert np.any(split.noise[:, dataset.config.input_steps :] != 0.0)
    assert split.target_states.shape[1] == split.inputs.shape[1] + 1


def test_zero_delay_uses_only_the_input_epoch() -> None:
    dataset = make_dataset(_spec(noise=0.0, delay=0), _dataset_config(), seed=28)
    for split in (dataset.train, dataset.validation, dataset.test):
        assert split.delay == 0
        assert split.n_steps == dataset.config.input_steps
        assert split.inputs.shape[1] == dataset.config.input_steps
        assert split.target_states.shape[1] == dataset.config.input_steps + 1
        assert np.any(split.inputs != 0.0)


def test_grid_extension_does_not_change_registered_random_tapes() -> None:
    carrier = _carrier()
    input_task = _spec(carrier, alpha=0.0, rA=1, rB=1, rotation_seed=3)
    state_task = _spec(carrier, alpha=1.0, rA=8, rB=4, rotation_seed=3)
    short = _dataset_config(n_test_blocks=2)
    extended = _dataset_config(n_test_blocks=6)
    input_data = make_dataset(input_task, short, seed=404)
    state_data = make_dataset(state_task, short, seed=404)
    extended_data = make_dataset(input_task, extended, seed=404)
    for name in ("train", "validation"):
        input_split = getattr(input_data, name)
        state_split = getattr(state_data, name)
        extension_split = getattr(extended_data, name)
        for field in ("inputs", "noise", "contexts", "block_ids", "trial_ids"):
            np.testing.assert_array_equal(
                getattr(input_split, field), getattr(state_split, field)
            )
            np.testing.assert_array_equal(
                getattr(input_split, field), getattr(extension_split, field)
            )
    # The common tape is fixed, but task-dependent target states need not be.
    assert not np.array_equal(input_data.train.target_states, state_data.train.target_states)


def test_public_train_only_factory_never_builds_heldout_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    config = _dataset_config()
    expected = make_dataset(spec, config, seed=606).train
    original = actuator_task._make_split
    calls: list[str] = []

    def spy(*args: object, **kwargs: object):
        calls.append(str(kwargs["split_name"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(actuator_task, "_make_split", spy)
    observed = make_actuator_matching_train_split(spec, config, seed=606)
    assert calls == ["train"]
    assert observed.fingerprint == expected.fingerprint
    np.testing.assert_array_equal(observed.target_states, expected.target_states)
    np.testing.assert_array_equal(observed.inputs, expected.inputs)
    np.testing.assert_array_equal(observed.noise, expected.noise)


def test_dataset_arrays_are_immutable_copies_without_cross_split_aliasing() -> None:
    dataset = make_dataset(_spec(), _dataset_config(), seed=5)
    arrays = (
        dataset.train.inputs,
        dataset.train.target_states,
        dataset.train.contexts,
        dataset.train.labels,
        dataset.train.block_ids,
        dataset.train.noise,
    )
    assert all(not value.flags.writeable for value in arrays)
    assert not np.shares_memory(dataset.train.inputs, dataset.validation.inputs)
    assert not np.shares_memory(dataset.train.noise, dataset.test.noise)
    assert not np.shares_memory(dataset.train.inputs, dataset.train.noise)
    copied = np.array(dataset.train.inputs, copy=True)
    copied[0, 0, 0] += 1.0
    assert copied[0, 0, 0] != dataset.train.inputs[0, 0, 0]


def test_different_seeds_and_generators_have_distinct_fingerprints() -> None:
    first_carrier = _carrier(1)
    second_carrier = _carrier(2)
    assert first_carrier.fingerprint != second_carrier.fingerprint
    first = _spec(first_carrier, generator_id="g-a")
    second = _spec(first_carrier, generator_id="g-b")
    assert first.fingerprint != second.fingerprint
    assert make_dataset(first, _dataset_config(), seed=1).fingerprint != make_dataset(
        first, _dataset_config(), seed=2
    ).fingerprint


def test_validation_fails_closed_for_invalid_configs_and_tampered_rollouts() -> None:
    with pytest.raises(TypeError, match="n_neurons must be an integer"):
        CarrierConfig(n_neurons=True)
    with pytest.raises(ValueError, match="inhibitory_fraction"):
        CarrierConfig(inhibitory_fraction=0.0)
    with pytest.raises(ValueError, match="even"):
        DatasetConfig(trials_per_block=3)
    carrier = _carrier()
    with pytest.raises(ValueError, match="rA"):
        _spec(carrier, rA=3)
    narrow_input_carrier = make_carrier(CarrierConfig(n_neurons=20, n_inputs=2), 1)
    with pytest.raises(ValueError, match="attainable"):
        make_task_spec(
            narrow_input_carrier,
            alpha=0.5,
            rA=2,
            rB=4,
            delay=2,
            noise=0.0,
            rotation_seed=1,
        )
    with pytest.raises(ValueError, match="alpha"):
        _spec(carrier, alpha=1.1)
    with pytest.raises(ValueError, match="non-empty"):
        _spec(carrier, generator_id=" ")

    dataset = make_dataset(_spec(carrier), _dataset_config(), seed=9)
    tampered_states = np.array(dataset.train.target_states, copy=True)
    tampered_states[0, -1, 0] += 1.0
    tampered_split = replace(dataset.train, target_states=tampered_states)
    with pytest.raises(ValueError, match="registered rollout"):
        ActuatorMatchingDataset(
            spec=dataset.spec,
            config=dataset.config,
            seed=dataset.seed,
            train=tampered_split,
            validation=dataset.validation,
            test=dataset.test,
        )
