import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from run_reproduction import (
    DATASETS,
    DATASET_CONFIGS,
    file_sha256,
    main,
    parse_sweep,
    run_config_fingerprint,
    write_report,
)


class ReproductionConfigTests(unittest.TestCase):
    def test_report_uses_refined_complete_input_set(self):
        result = {
            "complete_num_inputs": 2,
            "complete_fraction": 0.2,
            "completion_mode": "matlab_residual",
            "neuron": 3,
            "n_neurons": 11,
            "n_time": 20,
            "independent_entropy": 0.5,
            "selector": "schur_entropy_drop",
            "initialization": "matlab_reset",
            "run_config": {"actual_sweep": [1, 2, 4]},
            "nums_in": [1, 2, 4],
            "entropies": [0.4, 0.3, 0.2],
            "residual_errors": [3.0, 1.0, 0.5],
            "complete_flags": [True, True, True],
            "criterion_complete_flags": [False, True, True],
            "evaluation_phases": ["coarse", "binary_search", "coarse"],
            "iterations": [1, 1, 1],
            "inputs_order": [10, 20, 30, 40],
            "selected_inputs_complete": [20, 30],
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            sys, "argv", ["run_reproduction.py"]
        ):
            report = Path(directory) / "report.md"
            write_report(result, report, "synthetic")
            text = report.read_text(encoding="utf-8")
        self.assertIn("`[20, 30]`", text)
        self.assertNotIn("`[10, 20]`", text)

    def test_published_sanitized_snapshot_has_separate_verifiable_fingerprint(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "c_elegans_matlab_schur_neuron13_max32_config.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertTrue(config["snapshot_sanitized"])
        self.assertEqual(
            config["config_fingerprint"], config["original_config_fingerprint"]
        )
        payload = dict(config)
        for key in (
            "run_id",
            "config_fingerprint",
            "config_fingerprint_scope",
            "original_config_fingerprint",
            "published_config_fingerprint",
            "published_config_fingerprint_scope",
            "snapshot_sanitized",
        ):
            payload.pop(key, None)
        self.assertEqual(
            run_config_fingerprint(payload), config["published_config_fingerprint"]
        )

    def test_dataset_specific_matlab_thresholds_are_explicit(self):
        self.assertEqual(DATASET_CONFIGS["hippocampus"]["optimizer_threshold"], 1e-6)
        self.assertEqual(DATASET_CONFIGS["visual_responding"]["optimizer_threshold"], 1e-5)
        self.assertEqual(DATASET_CONFIGS["visual_spontaneous"]["optimizer_threshold"], 1e-5)
        self.assertEqual(DATASET_CONFIGS["c_elegans"]["optimizer_threshold"], 1e-5)
        self.assertTrue(all(config["corr_error_threshold"] == 2.0 for config in DATASET_CONFIGS.values()))

    def test_default_sweep_is_dataset_family_specific_and_capped(self):
        hippocampus = parse_sweep("", 16, "hippocampus_matlab")
        c_elegans = parse_sweep("", 16, "c_elegans_matlab")
        self.assertEqual(hippocampus, list(range(1, 11)) + [15, 16])
        self.assertEqual(c_elegans, list(range(1, 17)))

    def test_file_hash_is_recordable_without_loading_dataset(self):
        payload = b"provenance-sentinel"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.mat"
            path.write_bytes(payload)
            self.assertEqual(file_sha256(path), hashlib.sha256(payload).hexdigest())

    def test_config_fingerprint_changes_with_method(self):
        base = {"selector": "schur", "threshold": 1e-5}
        changed = {"selector": "residual", "threshold": 1e-5}
        self.assertNotEqual(
            run_config_fingerprint(base), run_config_fingerprint(changed)
        )

    def test_missing_data_failure_retains_immutable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "absent.mat"
            argv = [
                "run_reproduction.py",
                "--dataset",
                "c_elegans",
                "--max-inputs",
                "2",
                "--results-root",
                str(root / "results"),
            ]
            with mock.patch.dict(DATASETS, {"c_elegans": missing}), mock.patch.object(
                sys, "argv", argv
            ):
                with self.assertRaises(FileNotFoundError):
                    main()
            runs = list((root / "results" / "runs").iterdir())
            self.assertEqual(len(runs), 1)
            run = runs[0]
            for name in ("config.json", "metrics.json", "status.json", "run.log"):
                self.assertTrue((run / name).is_file())
            status = json.loads((run / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")


if __name__ == "__main__":
    unittest.main()
