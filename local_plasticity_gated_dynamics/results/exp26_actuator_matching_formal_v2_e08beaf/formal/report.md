# Exp26 actuator phase-diagram summary

**Conclusion: support**

Formal scope requires exactly seeds 0--29 and complete paired primary coverage. Missing or failed cells cannot be dropped.

all three held-out seed-level confirmatory endpoints passed

## Confirmatory endpoints

The three co-primary endpoints are seed-level held-out Spearman rho, threshold-classifier balanced accuracy, and AUROC. Their one-sided tests form one Holm-corrected family and the joint claim uses an intersection-union AND gate. Generator, neuron, and time point are not replicates.

| name | null_value | mean | lower_confidence | upper_confidence | p_value | p_value_holm |
| --- | --- | --- | --- | --- | --- | --- |
| spearman_rho | 0 | 0.760467 | 0.744335 | 0.775696 | 9.9999e-06 | 2.99997e-05 |
| classifier_balanced_accuracy | 0.5 | 0.85715 | 0.837534 | 0.875804 | 9.9999e-06 | 2.99997e-05 |
| classifier_auroc | 0.5 | 0.946663 | 0.93652 | 0.955992 | 9.9999e-06 | 2.99997e-05 |

## Gramian χ versus raw α incremental gate

Support additionally requires the held-out χ AUROC to exceed the raw-α AUROC with a positive seed-level confidence bound and one-sided p < 0.05. This gate is reported separately from the three-member Holm family.

| name | null_value | mean | lower_confidence | upper_confidence | p_value | p_value_holm |
| --- | --- | --- | --- | --- | --- | --- |
| chi_minus_alpha_auroc | 0 | 0.11477 | 0.100875 | 0.128637 | 9.9999e-06 | 9.9999e-06 |

## Frozen provenance and analysis contract

The canonical config hash excludes only runtime path, seed, run-label, and embedded evidence fields. Every selected run must match this config and manifest exactly, share one clean Git commit/tree and scientific runtime, and carry the registered analysis values in every metrics row.

| config_sha256 | source_config_file_sha256_by_seed | manifest_sha256 | git_commit | git_tree | git_dirty | run_label | budget_preflight_required | budget_preflight_passed | budget_preflight_receipt_sha256 | numpy_version | pandas_version | python_version | scikit_learn_version | scipy_version | statsmodels_version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 07ad3f16d9de6b5906155d95f215e9434e478ca992fd023adfabcd21a0005ecf | {"0": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "1": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "10": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "11": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "12": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "13": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "14": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "15": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "16": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "17": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "18": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "19": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "2": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "20": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "21": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "22": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "23": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "24": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "25": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "26": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "27": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "28": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "29": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "3": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "4": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "5": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "6": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "7": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "8": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63", "9": "50add2a0b3ab18a449f5a2090d52b31cf3d38b78ea209e0967acb9c48a366b63"} | a1c17a1e88c731f6678760865cf51d7236ae771bf839645c401e5cff8798ebfa | e08beaf9f51aacaaa80d42b2755c60d4080364bb | c91f11863a654c7588b0d7bde5eb1e1f43e5c774 | False | exp26-formal-v2 | True | True | bad665691233c9611fcdcce897c642d517a938b78adbabadee783c5e8cb1a671 | 2.3.5 | 3.0.1 | 3.11.15 | 1.8.0 | 1.17.1 | 0.14.6 |

| tie_margin | bootstrap_samples | permutation_samples | statistics_seed |
| --- | --- | --- | --- |
| 0.01 | 20000 | 100000 | 2601 |

## Coverage and retained failures

- Expected seeds: 30
- Observed expected seeds: 30
- Missing seeds: 0
- Unexpected seeds: 0
- Failed or non-terminal attempts: 0
- Failed rows retained: 0
- Invalid rows retained: 0
- Complete rows failing the functional-budget gate: 0
- Attempts with incomplete/malformed planned-cell coverage: 0
- Duplicate primary cell rows (automatically non-confirmatory): 0

## Seed-level descriptive metrics

| generator_split | actuator_mode | metric | mean | sd | n_seed |
| --- | --- | --- | --- | --- | --- |
| discovery | frozen | validation_balanced_accuracy | 0.5 | 0 | 30 |
| discovery | gain | validation_balanced_accuracy | 0.531439 | 0.0132673 | 30 |
| discovery | low_rank | validation_balanced_accuracy | 0.767566 | 0.0159893 | 30 |
| discovery | rgl | validation_balanced_accuracy | 0.955919 | 0.00729541 | 30 |
| discovery | routing | validation_balanced_accuracy | 0.616698 | 0.0148504 | 30 |
| heldout | frozen | validation_balanced_accuracy | 0.5 | 0 | 30 |
| heldout | gain | validation_balanced_accuracy | 0.536127 | 0.0156105 | 30 |
| heldout | low_rank | validation_balanced_accuracy | 0.753441 | 0.0185167 | 30 |
| heldout | rgl | validation_balanced_accuracy | 0.945802 | 0.00780766 | 30 |
| heldout | routing | validation_balanced_accuracy | 0.63065 | 0.0207347 | 30 |
| discovery | frozen | test_balanced_accuracy | 0.5 | 0 | 30 |
| discovery | gain | test_balanced_accuracy | 0.531234 | 0.00947478 | 30 |
| discovery | low_rank | test_balanced_accuracy | 0.768718 | 0.0153161 | 30 |
| discovery | rgl | test_balanced_accuracy | 0.956937 | 0.00616106 | 30 |
| discovery | routing | test_balanced_accuracy | 0.615633 | 0.014364 | 30 |
| heldout | frozen | test_balanced_accuracy | 0.5 | 0 | 30 |
| heldout | gain | test_balanced_accuracy | 0.537145 | 0.0153065 | 30 |
| heldout | low_rank | test_balanced_accuracy | 0.753788 | 0.0174112 | 30 |
| heldout | rgl | test_balanced_accuracy | 0.948595 | 0.00656955 | 30 |
| heldout | routing | test_balanced_accuracy | 0.628575 | 0.0169692 | 30 |

## RGL interpretation boundary

RGL is a descriptive composite ceiling. It is not an additional primary actuator family, is excluded from χ threshold fitting and all three co-primary tests, and cannot rescue failed routing, gain, or low-rank cells.

Plot status: written.

All raw rows, including scientific failures, remain in `raw_metrics.csv`.
