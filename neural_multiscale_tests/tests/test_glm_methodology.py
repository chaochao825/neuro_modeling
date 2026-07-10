import unittest

import numpy as np

from models.glm import _split_standardize_features, compare_nested_glms


class GlmMethodologyTests(unittest.TestCase):
    def test_scaler_is_fit_on_training_prefix_only(self):
        base = np.column_stack(
            [np.ones(6), np.array([0.0, 1.0, 2.0, 10.0, 20.0, 30.0])]
        )
        shifted_test = base.copy()
        shifted_test[3:, 1] += 10000.0

        train_a, test_a = _split_standardize_features(base, split=3)
        train_b, test_b = _split_standardize_features(shifted_test, split=3)

        np.testing.assert_allclose(train_a, train_b)
        self.assertAlmostEqual(float(train_a[:, 1].mean()), 0.0, places=12)
        self.assertFalse(np.allclose(test_a, test_b))

    def test_public_mode_does_not_invent_index_locality(self):
        rng = np.random.default_rng(19)
        spikes = rng.binomial(1, 0.2, size=(48, 6))
        result = compare_nested_glms(
            spikes,
            max_units=3,
            seed=4,
            use_index_ring=False,
        )

        self.assertEqual(
            result["local_structure"], "not_evaluated_missing_coordinates"
        )
        self.assertIsNone(result["local_delta_bits"])
        self.assertNotIn("history_local", result["test_loglik_per_bin"])


if __name__ == "__main__":
    unittest.main()
