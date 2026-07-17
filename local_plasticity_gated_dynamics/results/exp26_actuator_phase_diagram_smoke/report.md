# Exp26 actuator phase-diagram summary

**Conclusion: inconclusive**

**DEVELOPMENT ONLY:** smoke seeds 9000 and 9001 are permanently scoped to pipeline validation. Their numerical outcome is forced to `inconclusive`.

smoke profile is development-only and cannot support or oppose the registered claim

## Confirmatory endpoints

The three co-primary endpoints are seed-level held-out Spearman rho, threshold-classifier balanced accuracy, and AUROC. Their one-sided tests form one Holm-corrected family and the joint claim uses an intersection-union AND gate. Generator, neuron, and time point are not replicates.

| name | null_value | mean | lower_confidence | upper_confidence | p_value | p_value_holm |
| --- | --- | --- | --- | --- | --- | --- |
| spearman_rho | 0 | 0.790285 | 0.724977 | 0.855593 | 0.4 | 1 |
| classifier_balanced_accuracy | 0.5 | 0.866667 | 0.833333 | 0.9 | 0.4 | 1 |
| classifier_auroc | 0.5 | 0.969444 | 0.966667 | 0.972222 | 0.4 | 1 |

## Gramian χ versus raw α incremental gate

Support additionally requires the held-out χ AUROC to exceed the raw-α AUROC with a positive seed-level confidence bound and one-sided p < 0.05. This gate is reported separately from the three-member Holm family.

| name | null_value | mean | lower_confidence | upper_confidence | p_value | p_value_holm |
| --- | --- | --- | --- | --- | --- | --- |
| chi_minus_alpha_auroc | 0 | 0.0138889 | 0 | 0.0277778 | 0.6 | 0.6 |

## Coverage and retained failures

- Expected seeds: 2
- Observed expected seeds: 2
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
| discovery | frozen | validation_balanced_accuracy | 0.5 | 0 | 2 |
| discovery | gain | validation_balanced_accuracy | 0.541667 | 0.0441942 | 2 |
| discovery | low_rank | validation_balanced_accuracy | 0.760417 | 0.0294628 | 2 |
| discovery | rgl | validation_balanced_accuracy | 0.979167 | 0.0147314 | 2 |
| discovery | routing | validation_balanced_accuracy | 0.786458 | 0.0073657 | 2 |
| heldout | frozen | validation_balanced_accuracy | 0.5 | 0 | 2 |
| heldout | gain | validation_balanced_accuracy | 0.557292 | 0.0073657 | 2 |
| heldout | low_rank | validation_balanced_accuracy | 0.723958 | 0.0368285 | 2 |
| heldout | rgl | validation_balanced_accuracy | 0.96875 | 0.0441942 | 2 |
| heldout | routing | validation_balanced_accuracy | 0.723958 | 0.0368285 | 2 |
| discovery | frozen | test_balanced_accuracy | 0.5 | 0 | 2 |
| discovery | gain | test_balanced_accuracy | 0.515625 | 0.0368285 | 2 |
| discovery | low_rank | test_balanced_accuracy | 0.71875 | 0.0294628 | 2 |
| discovery | rgl | test_balanced_accuracy | 0.96875 | 0 | 2 |
| discovery | routing | test_balanced_accuracy | 0.71875 | 0.0441942 | 2 |
| heldout | frozen | test_balanced_accuracy | 0.5 | 0 | 2 |
| heldout | gain | test_balanced_accuracy | 0.583333 | 0.0589256 | 2 |
| heldout | low_rank | test_balanced_accuracy | 0.734375 | 0.0073657 | 2 |
| heldout | rgl | test_balanced_accuracy | 0.979167 | 0 | 2 |
| heldout | routing | test_balanced_accuracy | 0.713542 | 0.0368285 | 2 |

## RGL interpretation boundary

RGL is a descriptive composite ceiling. It is not an additional primary actuator family, is excluded from χ threshold fitting and all three co-primary tests, and cannot rescue failed routing, gain, or low-rank cells.

Plot status: written.

All raw rows, including scientific failures, remain in `raw_metrics.csv`.
