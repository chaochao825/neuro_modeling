import unittest

from analyses.reporting import build_decision_matrix
from run_simulations import run_all
from simulations.baseline import run_baseline
from simulations.hawkes import simulate_hawkes


class SmokeTests(unittest.TestCase):
    def test_baseline_shape_and_metrics(self):
        out = run_baseline(n_units=12, t_steps=120, seed=11)
        self.assertEqual(out["spikes"].shape, (120, 12))
        self.assertIn("eigenspectrum", out["metrics"])

    def test_hawkes_glm_comparison(self):
        out = simulate_hawkes(n_units=14, t_steps=180, seed=12)
        self.assertIn("history_delta_bits", out["glm_comparison"])
        self.assertGreaterEqual(out["glm_comparison"]["units_fit"], 1)

    def test_full_quick_summary_decision_matrix(self):
        summary = run_all(seed=13, quick=True)
        matrix = build_decision_matrix(summary)
        self.assertEqual(set(matrix.keys()), {
            "H1_history_local_coupling",
            "H2_nearcritical_powerlaw_spectrum",
            "H3_oscillatory_synchrony",
            "H4_avalanche_criticality",
            "H5_energy_constraint",
        })


if __name__ == "__main__":
    unittest.main()

