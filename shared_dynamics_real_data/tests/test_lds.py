from __future__ import annotations

import numpy as np

from shared_dynamics_real_data.lds import (
    SharedBasisLDS,
    _fit_affine_dynamics,
    effective_rank,
    woodbury_gaussian_logpdf,
)
from shared_dynamics_real_data.pipeline import SharedDynamicsPipeline
from shared_dynamics_real_data.splits import TimeSegment, purged_contiguous_folds


def _orthonormal(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(rows, columns)), mode="reduced")[0]


def test_woodbury_logpdf_matches_literal_dense_innovation() -> None:
    rng = np.random.default_rng(3)
    n, d = 7, 3
    residual = rng.normal(size=n)
    observation = rng.normal(size=(n, d))
    root = rng.normal(size=(d, d))
    latent_covariance = root @ root.T + 0.3 * np.eye(d)
    observation_variance = rng.uniform(0.2, 1.1, size=n)

    scalable = woodbury_gaussian_logpdf(
        residual, observation, latent_covariance, observation_variance
    )
    information = observation.T @ (
        observation / observation_variance[:, None]
    )
    weighted_transpose = observation.T / observation_variance[None, :]
    cached = woodbury_gaussian_logpdf(
        residual,
        observation,
        latent_covariance,
        observation_variance,
        observation_information=information,
        weighted_observation_transpose=weighted_transpose,
    )
    dense = np.diag(observation_variance) + observation @ latent_covariance @ observation.T
    sign, logdet = np.linalg.slogdet(dense)
    literal = -0.5 * (
        n * np.log(2 * np.pi)
        + logdet
        + residual @ np.linalg.solve(dense, residual)
    )

    assert sign == 1
    np.testing.assert_allclose(scalable, literal, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(cached, literal, rtol=1e-11, atol=1e-11)


def test_multistep_kalman_likelihood_matches_joint_dense_gaussian() -> None:
    observation = np.array([[1.0], [0.35]])
    transition = np.array([[0.62]])
    process_variance = np.array([0.18])
    observation_variance = np.array([0.3, 0.5])
    values = np.array([[0.2, -0.4], [0.5, 0.1], [-0.3, 0.25]])
    model = SharedBasisLDS("common", 1)
    model.contexts_ = ("c",)
    model.n_features_ = 2
    model.bases_ = {"c": observation}
    model.transitions_ = {"c": transition}
    model.offsets_ = {"c": np.zeros(1)}
    model.process_variances_ = {"c": process_variance}
    model.observation_means_ = {"c": np.zeros(2)}
    model.observation_variances_ = {"c": observation_variance}
    model.observation_information_ = {
        "c": observation.T @ (observation / observation_variance[:, None])
    }
    model.weighted_observation_transposes_ = {
        "c": observation.T / observation_variance[None, :]
    }
    model.rollout_reference_scale_ = 1.0
    model._fitted = True

    score = model.score((TimeSegment("c", values, np.arange(3)),))

    latent_covariance = np.empty((3, 3))
    latent_covariance[0, 0] = 1.0
    latent_covariance[1, 0] = latent_covariance[0, 1] = 0.62
    latent_covariance[1, 1] = 0.62**2 + 0.18
    latent_covariance[2, 0] = latent_covariance[0, 2] = 0.62**2
    latent_covariance[2, 1] = latent_covariance[1, 2] = (
        0.62 * latent_covariance[1, 1]
    )
    latent_covariance[2, 2] = 0.62**2 * latent_covariance[1, 1] + 0.18
    joint = np.kron(latent_covariance, observation @ observation.T)
    joint += np.kron(np.eye(3), np.diag(observation_variance))
    vector = values.reshape(-1)
    sign, logdet = np.linalg.slogdet(joint)
    dense_logpdf = -0.5 * (
        vector.size * np.log(2 * np.pi)
        + logdet
        + vector @ np.linalg.solve(joint, vector)
    )
    assert sign == 1
    np.testing.assert_allclose(
        score.standardized_marginal_log_likelihood,
        dense_logpdf,
        rtol=1e-11,
        atol=1e-11,
    )


def test_rank_and_separate_subspace_metrics_have_numeric_goldens() -> None:
    probabilities = np.array([2.0 / 3.0, 1.0 / 3.0])
    expected_rank = np.exp(-np.sum(probabilities * np.log(probabilities)))
    np.testing.assert_allclose(effective_rank(np.diag([2.0, 1.0])), expected_rank)

    time = np.linspace(-2, 2, 40)
    first = np.column_stack([time, time**2, np.zeros((40, 2))])
    second = np.column_stack([np.zeros((40, 2)), time, time**2])
    model = SharedBasisLDS("separate", 2).fit(
        (
            TimeSegment("a", first, np.arange(40)),
            TimeSegment("b", second, np.arange(40)),
        )
    )
    np.testing.assert_allclose(model.subspace_angle_degrees(), 90.0, atol=1e-10)


def _switching_recordings(seed: int = 12) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_features, latent_dim, n_time = 10, 2, 360
    basis = _orthonormal(rng, n_features, latent_dim)
    transitions = {
        "integrate": np.array([[0.92, 0.24], [-0.12, 0.78]]),
        "rotate": np.array([[0.35, -0.72], [0.75, 0.32]]),
    }
    recordings: dict[str, np.ndarray] = {}
    for context, transition in transitions.items():
        latent = np.empty((n_time, latent_dim))
        latent[0] = rng.normal(scale=0.3, size=latent_dim)
        for time in range(n_time - 1):
            latent[time + 1] = transition @ latent[time] + rng.normal(
                scale=0.045, size=latent_dim
            )
        observations = latent @ basis.T + rng.normal(
            scale=0.025, size=(n_time, n_features)
        )
        recordings[context] = observations
    return recordings


def test_shared_switching_beats_common_and_is_smaller_than_separate() -> None:
    fold = purged_contiguous_folds(
        _switching_recordings(), n_splits=4, purge=2
    )[2]
    scores = {}
    pipelines = {}
    for family in ("common", "shared", "separate"):
        pipeline = SharedDynamicsPipeline(
            family,
            latent_dim=2,
            random_state=9,
            ridge=1e-5,
            variance_floor=1e-5,
        ).fit(fold.train)
        pipelines[family] = pipeline
        scores[family] = pipeline.score(fold.test)

    assert scores["shared"].nll_per_scalar < scores["common"].nll_per_scalar
    assert scores["shared"].parameter_count < scores["separate"].parameter_count
    assert scores["shared"].one_step_r2 > scores["common"].one_step_r2
    assert 1.0 <= scores["shared"].effective_rank <= 2.0
    assert scores["shared"].subspace_angle_degrees == 0.0
    assert scores["separate"].subspace_angle_degrees >= 0.0
    assert scores["shared"].likelihood_coordinate == "original_selected_units"
    assert (
        scores["shared"].prediction_metric_coordinate
        == "train_standardized_selected_units"
    )
    np.testing.assert_allclose(
        scores["shared"].nll_per_scalar,
        scores["shared"].standardized_nll_per_scalar
        + np.mean(np.log(pipelines["shared"].preprocessor_.scale_)),
    )


def test_model_fit_and_scores_are_seed_deterministic() -> None:
    fold = purged_contiguous_folds(
        _switching_recordings(seed=22), n_splits=3, purge=1
    )[0]
    outputs = []
    bases = []
    for _ in range(2):
        pipeline = SharedDynamicsPipeline(
            "shared",
            latent_dim=2,
            basis_control="random",
            random_state=123,
        ).fit(fold.train)
        outputs.append(pipeline.score(fold.test))
        bases.append(pipeline.model_.bases_["integrate"].copy())
    assert outputs[0] == outputs[1]
    np.testing.assert_array_equal(bases[0], bases[1])


def test_separate_parameter_count_includes_context_specific_bases_and_noise() -> None:
    rng = np.random.default_rng(7)
    segments = tuple(
        TimeSegment(context, rng.normal(size=(30, 6)), np.arange(30))
        for context in ("a", "b")
    )
    model = SharedBasisLDS("separate", 2, random_state=0).fit(segments)
    # Shared preprocessing 2N + K*(Stiefel basis + A + b + Q + diagonal R)
    # plus (K-1)*N identifiable context observation-mean degrees.
    assert model.parameter_count() == (
        2 * 6 + 2 * ((6 * 2 - 3) + 4 + 2 + 2 + 6) + (2 - 1) * 6
    )
    assert 0.0 < model.mean_topk_singular_energy(1) <= 1.0
    assert model.mean_topk_singular_energy(2) == 1.0


def test_separate_observation_model_centers_each_context_before_noise_fit() -> None:
    rng = np.random.default_rng(91)
    base_a = rng.normal(scale=0.2, size=(80, 3)) + np.array([0.0, 0.0, 0.0])
    base_b = rng.normal(scale=0.2, size=(80, 3)) + np.array([0.0, 4.0, -2.0])
    segments = (
        TimeSegment("a", base_a, np.arange(80)),
        TimeSegment("b", base_b, np.arange(80)),
    )
    model = SharedBasisLDS("separate", 1, random_state=3).fit(segments)

    np.testing.assert_allclose(model.observation_means_["a"], base_a.mean(axis=0))
    np.testing.assert_allclose(model.observation_means_["b"], base_b.mean(axis=0))
    for context, matrix in (("a", base_a), ("b", base_b)):
        centered = matrix - model.observation_means_[context]
        basis = model.bases_[context]
        residual = centered - (centered @ basis) @ basis.T
        expected = np.maximum(np.mean(residual**2, axis=0), model.variance_floor)
        np.testing.assert_allclose(model.observation_variances_[context], expected)


def test_stabilizing_transition_refits_offset_and_noise_mse() -> None:
    rng = np.random.default_rng(32)
    current = rng.normal(size=(200, 2)) + np.array([2.0, -1.0])
    unstable = np.array([[1.4, 0.2], [0.0, 1.2]])
    target = current @ unstable.T + np.array([3.0, -2.0])
    transition, offset, process_variance = _fit_affine_dynamics(
        current,
        target,
        ridge=0.0,
        variance_floor=1e-9,
        max_radius=0.8,
    )
    residual = target - (current @ transition.T + offset)
    np.testing.assert_allclose(residual.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        process_variance,
        np.maximum(np.mean(residual**2, axis=0), 1e-9),
    )


def test_seeded_random_basis_is_not_counted_as_fitted_dof() -> None:
    rng = np.random.default_rng(8)
    segments = tuple(
        TimeSegment(context, rng.normal(size=(40, 8)), np.arange(40))
        for context in ("a", "b")
    )
    aligned = SharedBasisLDS("shared", 2, basis_control="aligned").fit(segments)
    random = SharedBasisLDS("shared", 2, basis_control="random").fit(segments)
    basis_dof = 8 * 2 - 2 * 3 // 2
    assert aligned.parameter_count() - random.parameter_count() == basis_dof
