import pathlib
import unittest
from unittest import mock

import numpy as np

from minimal_computation.core import (
    EPS,
    MaxEntFit,
    binary_entropy,
    candidate_input_scores,
    choose_best_fit,
    greedy_minimax_entropy,
    schur_entropy_drop_scores,
    sigmoid,
    unique_columns_distribution,
)
from minimal_computation.matv5 import load_activity


class MinimalComputationSmokeTests(unittest.TestCase):
    def test_mat_reader_c_elegans(self):
        path = pathlib.Path(__file__).resolve().parents[2] / "minimal_computation_original" / "data_C_elegans.mat"
        if not path.exists():
            self.skipTest("minimal_computation_original data is not vendored; clone ChrisWLynn/Minimal_computation next to minimal_computation_python to run this check")
        x = load_activity(path)
        self.assertEqual(x.shape, (128, 1600))
        self.assertEqual(x.dtype, np.uint8)
        self.assertGreater(int(x.sum()), 0)

    def test_single_input_maxent_matches_constraints(self):
        x = np.array([[0, 0, 1, 1]], dtype=float)
        p = np.ones(4) / 4
        y_obs = 0.5
        corr_obs = np.array([0.4])
        fit = choose_best_fit(y_obs, corr_obs, x, p, b0=0.0, w0=np.zeros(1), threshold=1e-5)
        self.assertTrue(fit.complete)
        py = 1 / (1 + np.exp(-fit.bias - x.T @ fit.weights))
        self.assertAlmostEqual(float(p @ py), y_obs, places=4)
        self.assertAlmostEqual(float((x @ (p * py))[0]), corr_obs[0], places=4)

    def test_matlab_reset_is_default_and_ignores_warm_start_values(self):
        x = np.array([[0, 0, 1, 1]], dtype=float)
        p = np.ones(4) / 4
        reset = choose_best_fit(
            0.5,
            np.array([0.4]),
            x,
            p,
            b0=100.0,
            w0=np.array([-100.0]),
            threshold=1e-5,
        )
        warm = choose_best_fit(
            0.5,
            np.array([0.4]),
            x,
            p,
            b0=100.0,
            w0=np.array([-100.0]),
            threshold=1e-5,
            initialization="warm_start",
            max_steps=1,
        )
        self.assertTrue(reset.complete)
        self.assertFalse(warm.complete)

    def test_entropy_bounds(self):
        self.assertAlmostEqual(binary_entropy(0.5), 1.0, places=8)
        states, probs = unique_columns_distribution(np.array([[0, 1, 1], [1, 0, 0]]))
        self.assertEqual(states.shape[1], 2)
        self.assertAlmostEqual(float(probs.sum()), 1.0)

    def test_block_schur_matches_full_hessian_golden_value(self):
        rng = np.random.default_rng(17)
        activity = (rng.random((6, 96)) < 0.45).astype(float)
        y = 0
        selected = np.array([1, 2])
        candidates = np.array([3, 4, 5])
        fit = MaxEntFit(
            bias=-0.35,
            weights=np.array([0.8, -0.4]),
            complete=True,
            iterations=1,
            error=0.0,
        )

        blocked = schur_entropy_drop_scores(
            activity,
            y,
            selected,
            candidates,
            fit,
            candidate_block_size=1,
        )

        py = sigmoid(fit.bias + fit.weights @ activity[selected, :])
        q = py * (1.0 - py) / activity.shape[1]
        x_bias = np.vstack((np.ones(activity.shape[1]), activity))
        full_hessian = (x_bias * q) @ x_bias.T
        selected_bias = np.r_[0, selected + 1]
        candidate_bias = candidates + 1
        a_ss = full_hessian[np.ix_(selected_bias, selected_bias)]
        a_sr = full_hessian[np.ix_(selected_bias, candidate_bias)]
        schur_diag = np.diag(full_hessian[np.ix_(candidate_bias, candidate_bias)]) - np.sum(
            a_sr * np.linalg.solve(a_ss, a_sr), axis=0
        )
        residual = (
            activity[candidates, :] @ activity[y, :] / activity.shape[1]
            - activity[candidates, :] @ py / activity.shape[1]
        )
        expected = 0.5 * residual**2 / np.maximum(schur_diag, EPS)
        np.testing.assert_allclose(blocked, expected, rtol=1e-11, atol=1e-12)

    def test_optimizer_nonconvergence_cannot_be_complete(self):
        activity = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=float,
        )
        result = greedy_minimax_entropy(
            activity,
            neuron_id_matlab=2,
            nums_in=[1],
            threshold=1e-15,
            corr_error_threshold=1e9,
            optimizer_max_steps=1,
            optimizer_steps_check=1,
            completion_mode="strict_optimizer_and_residual",
        )
        self.assertEqual(result.residual_errors, [0.0])
        self.assertEqual(result.complete_flags, [False])
        self.assertEqual(result.criterion_complete_flags, [False])
        self.assertIsNone(result.complete_num_inputs)

        matlab_result = greedy_minimax_entropy(
            activity,
            neuron_id_matlab=2,
            nums_in=[1],
            threshold=1e-15,
            corr_error_threshold=1e9,
            optimizer_max_steps=1,
            optimizer_steps_check=1,
            completion_mode="matlab_residual",
        )
        self.assertEqual(matlab_result.complete_flags, [False])
        self.assertEqual(matlab_result.criterion_complete_flags, [True])
        self.assertEqual(matlab_result.complete_num_inputs, 1)

    def test_matlab_failure_mode_keeps_last_exponent(self):
        x = np.array([[0, 1]], dtype=float)
        p = np.array([0.5, 0.5])
        fits = [
            MaxEntFit(0.0, np.zeros(1), False, 1, 0.1),
            MaxEntFit(0.0, np.zeros(1), False, 1, 0.2),
        ]
        with mock.patch(
            "minimal_computation.core.fit_maxent_neuron", side_effect=fits
        ):
            matlab = choose_best_fit(
                0.5,
                np.array([0.25]),
                x,
                p,
                exponents=(1.0, 0.0),
                failure_selection="matlab_last",
            )
        with mock.patch(
            "minimal_computation.core.fit_maxent_neuron", side_effect=fits
        ):
            robust = choose_best_fit(
                0.5,
                np.array([0.25]),
                x,
                p,
                exponents=(1.0, 0.0),
                failure_selection="best_error",
            )
        self.assertEqual(matlab.error, 0.2)
        self.assertEqual(robust.error, 0.1)

    def test_binary_search_refines_first_complete_coarse_bracket(self):
        # Every input is exactly independent of the output, so the optimizer
        # converges at its reset initialization.  Mocked residuals isolate the
        # coarse-to-binary-search control flow from optimizer numerics.
        states = np.indices((2, 2, 2, 2, 2)).reshape(5, -1).astype(float)

        def residual_by_size(_activity, _y, inputs, _fit):
            return 3.0 if len(inputs) < 3 else 1.0

        with mock.patch(
            "minimal_computation.core.residual_correlation_error",
            side_effect=residual_by_size,
        ):
            result = greedy_minimax_entropy(
                states,
                neuron_id_matlab=5,
                nums_in=[1, 4],
                corr_error_threshold=2.0,
            )

        self.assertTrue(result.binary_search_performed)
        self.assertEqual(result.coarse_nums_in, [1, 4])
        self.assertEqual(result.nums_in, [1, 2, 3, 4])
        self.assertEqual(result.complete_num_inputs, 3)
        self.assertEqual(result.evaluation_phases, ["coarse", "binary_search", "binary_search", "coarse"])

    def test_exact_and_residual_selectors_are_distinct(self):
        rng = np.random.default_rng(1)
        rates = np.array([[0.5], [0.2], [0.5], [0.8], [0.35]])
        activity = (rng.random((5, 80)) < rates).astype(float)
        fit = MaxEntFit(
            bias=-0.4,
            weights=np.array([1.7]),
            complete=True,
            iterations=1,
            error=0.0,
        )
        candidates = np.array([2, 3, 4])
        exact = candidate_input_scores(
            activity,
            0,
            [1],
            candidates,
            fit,
            selector="schur_entropy_drop",
            candidate_block_size=2,
        )
        baseline = candidate_input_scores(
            activity,
            0,
            [1],
            candidates,
            fit,
            selector="residual_approximation",
        )
        self.assertEqual(int(np.argmax(exact)), 1)
        self.assertEqual(int(np.argmax(baseline)), 0)


if __name__ == "__main__":
    unittest.main()
