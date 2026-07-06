import pathlib
import unittest

import numpy as np

from minimal_computation.core import binary_entropy, choose_best_fit, unique_columns_distribution
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

    def test_entropy_bounds(self):
        self.assertAlmostEqual(binary_entropy(0.5), 1.0, places=8)
        states, probs = unique_columns_distribution(np.array([[0, 1, 1], [1, 0, 0]]))
        self.assertEqual(states.shape[1], 2)
        self.assertAlmostEqual(float(probs.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
