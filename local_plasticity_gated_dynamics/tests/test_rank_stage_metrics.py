from __future__ import annotations

import numpy as np
import pytest

from src.analysis.rank_stage_metrics import (
    credit_tangent_summary,
    masked_outer_product_identity,
    matrix_rank_summary,
    update_stage_rank_summary,
)


def test_masked_outer_product_identity_preserves_mask_rank_for_nonzero_factors() -> (
    None
):
    mask = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1],
            [1, 1, 0, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )
    left = np.array([0.5, -1.0, 2.0, 0.25])
    right = np.array([1.0, -0.5, 3.0, 2.0, -2.0])

    summary = masked_outer_product_identity(mask, left, right)

    assert summary.equal
    assert summary.max_abs_residual == 0.0
    assert summary.raw_outer.numerical_rank == 1
    assert summary.all_factors_nonzero
    assert summary.exact_rank_preservation_expected
    assert summary.numerically_preserves_mask_rank is True
    assert (
        summary.masked_outer.numerical_rank
        == summary.diagonal_form.numerical_rank
        == summary.mask.numerical_rank
    )


def test_zero_factors_restrict_the_mask_without_breaking_the_identity() -> None:
    mask = np.array([[1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 1]], dtype=bool)
    left = np.array([1.0, 0.0, -2.0])
    right = np.array([0.0, 3.0, 1.0, 0.0])

    summary = masked_outer_product_identity(mask, left, right)
    active = mask[np.ix_(left != 0.0, right != 0.0)].astype(float)

    assert summary.equal
    assert not summary.all_factors_nonzero
    assert not summary.exact_rank_preservation_expected
    assert summary.numerically_preserves_mask_rank is None
    assert summary.left_diagonal_condition_number is None
    assert summary.right_diagonal_condition_number is None
    assert summary.masked_outer.numerical_rank == np.linalg.matrix_rank(active)


def test_exact_mask_rank_theorem_is_separate_from_numerical_detectability() -> None:
    summary = masked_outer_product_identity(
        np.eye(3, dtype=bool),
        np.array([1.0, 1e-14, 1.0]),
        np.ones(3),
    )

    assert summary.equal
    assert summary.exact_rank_preservation_expected
    assert summary.mask.numerical_rank == 3
    assert summary.masked_outer.numerical_rank == 2
    assert summary.numerically_preserves_mask_rank is False
    assert summary.left_diagonal_condition_number == pytest.approx(1e14)
    assert summary.right_diagonal_condition_number == pytest.approx(1.0)


def test_stage_summary_separates_hebbian_decay_dale_and_normalization() -> None:
    hebbian = np.outer([1.0, 2.0, 3.0], [0.5, -1.0, 2.0])
    decay = np.diag([0.2, -0.1, 0.0])
    raw = hebbian + decay
    mask = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
    masked = mask * raw
    dale = masked.copy()
    normalization = np.diag([0.05, 0.0, -0.02])
    total = dale + normalization

    summary = update_stage_rank_summary(
        hebbian_update=hebbian,
        decay_update=decay,
        raw_update=raw,
        masked_update=masked,
        dale_applied_update=dale,
        normalization_correction=normalization,
        total_update=total,
    )

    assert summary.hebbian.numerical_rank == 1
    assert summary.decay.numerical_rank == 2
    for name, matrix in (
        ("raw", raw),
        ("masked", masked),
        ("dale_applied", dale),
        ("normalization_correction", normalization),
        ("total", total),
    ):
        assert getattr(summary, name).numerical_rank == np.linalg.matrix_rank(
            matrix, tol=1e-12
        )
        assert np.isfinite(getattr(summary, name).effective_rank)

    with pytest.raises(ValueError, match="hebbian_update"):
        update_stage_rank_summary(
            hebbian_update=hebbian,
            decay_update=decay,
            raw_update=raw + 1.0,
            masked_update=masked,
            dale_applied_update=dale,
            normalization_correction=normalization,
            total_update=total,
        )


def test_credit_tangent_chunked_gram_matches_explicit_weight_directions() -> None:
    rng = np.random.default_rng(17)
    basis, _ = np.linalg.qr(rng.normal(size=(6, 3)), mode="reduced")
    derivative = np.linspace(0.4, 1.0, 6)
    eligibility = np.linspace(-0.8, 1.2, 5)
    mask = rng.random((6, 5)) < 0.55
    scale = rng.uniform(0.5, 1.5, size=(6, 5))

    summary = credit_tangent_summary(
        basis,
        eligibility,
        post_derivative=derivative,
        connectivity_mask=mask,
        synaptic_scale=scale,
        stage="dale_active",
        edge_chunk_size=2,
    )
    explicit = np.column_stack(
        [
            (
                scale * mask * np.outer(derivative * basis[:, channel], eligibility)
            ).reshape(-1)
            for channel in range(basis.shape[1])
        ]
    )

    np.testing.assert_allclose(summary.gram, explicit.T @ explicit)
    assert summary.numerical_dimension == np.linalg.matrix_rank(explicit, tol=1e-12)
    assert summary.numerical_dimension <= summary.feedback_dim == 3
    assert summary.n_active_synapses == np.count_nonzero(mask)
    assert summary.stage == "dale_active"


def test_credit_tangent_detects_redundant_or_silenced_channels() -> None:
    basis = np.column_stack([np.array([1.0, 0.0, 1.0]), np.array([2.0, 0.0, 2.0])])
    summary = credit_tangent_summary(
        basis,
        np.array([1.0, -1.0]),
        post_derivative=np.array([1.0, 0.0, 1.0]),
        connectivity_mask=np.ones((3, 2), dtype=bool),
    )

    assert summary.feedback_dim == 2
    assert summary.numerical_dimension == 1
    assert summary.effective_dimension == pytest.approx(1.0)


@pytest.mark.parametrize("feedback_dim", [8, 32])
def test_credit_tangent_gram_noise_floor_rejects_false_rank_directions(
    feedback_dim: int,
) -> None:
    rng = np.random.default_rng(991)
    shared = rng.normal(size=512)
    # Exactly collinear feedback channels produce a true tangent rank of one.
    basis = shared[:, None] * np.arange(1.0, feedback_dim + 1.0)[None, :]
    summary = credit_tangent_summary(
        basis,
        rng.normal(size=512),
        connectivity_mask=rng.random((512, 512)) < 0.1,
        edge_chunk_size=4096,
    )

    assert summary.feedback_dim == feedback_dim
    assert summary.numerical_dimension == 1


def test_rank_stage_metrics_reject_implicit_or_invalid_numerics() -> None:
    zero = matrix_rank_summary(np.zeros((3, 2)))
    assert zero.numerical_rank == 0
    assert zero.effective_rank == 0.0
    thresholded = matrix_rank_summary(np.diag([1.0, 1e-11]))
    assert thresholded.numerical_rank == 1
    assert thresholded.threshold == pytest.approx(1e-10)

    with pytest.raises(TypeError):
        matrix_rank_summary(np.array([[True]]))
    with pytest.raises(ValueError, match="binary"):
        masked_outer_product_identity(np.array([[2]]), [1.0], [1.0])
    with pytest.raises(ValueError, match="positive integer"):
        credit_tangent_summary(np.eye(2), [1.0, 1.0], edge_chunk_size=0)
