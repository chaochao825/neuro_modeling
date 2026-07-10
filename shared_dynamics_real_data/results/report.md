# Shared-basis real-data report

Profile: `formal`. Latest planned panel: 2160 cells; complete: 2160; failed: 0.

The analysis uses the upstream-described same-neuron visual responding/spontaneous pair; row alignment cannot be independently verified because the MAT files contain no unit identifiers. Contiguous blocks are held out with a purge gap. Unit selection, scaling, PCA bases, transition/noise parameters, and rollout normalization are fit on training blocks only.

Computational seeds sample matched neuron subsets. Folds, seeds, time bins, and neurons are not biological replicates. Consequently every population-level claim remains `inconclusive` until independent sessions/animals are available.

The direction column is a deterministic audit of this single recording pair, not an inferential result. In particular, a positive shared-minus-common NLL opposes a switching advantage, while a non-positive absolute-signal margin opposes positive one-step R2 together with rollout error below the training-dispersion scale.

## Three-way claim audit

| Claim | Criterion | Descriptive estimate | Direction within this recording | Formal conclusion |
|---|---|---:|---|---|
| R0_switching_improves_common | shared context transitions have lower NLL than common transition | 0.00020319 | oppose | **inconclusive** |
| R1_shared_retains_separate_gain | min(median retained gain-0.95, 1-median parameter fraction) >=0 | NA | unavailable | **inconclusive** |
| R2_aligned_beats_basis_controls | median per-seed minimum control-minus-shared NLL >0 | 0.0013404 | support | **inconclusive** |
| R3_d4_nll_vs_highest_tested_dimension | d=4 shared NLL no more than 0.01 above highest tested dimension (d=32); not an intrinsic-rank estimate | -0.04244 | support | **inconclusive** |
| R4_shared_has_absolute_predictive_signal | min(median one-step R2, 1-median rollout NRMSE) >0 | -0.013727 | oppose | **inconclusive** |

Unavailable descriptive criteria:
- `R1_shared_retains_separate_gain`: paired models are complete, but separate does not improve common (common-minus-separate NLL <=0), so there is no positive switching gain to retain; only one aligned recording pair; seeds/folds/neurons are not biological replicates.

## Model summary

Values aggregate folds inside each computational seed and then summarize seed robustness.

| d | Model | NLL/scalar mean | one-step R2 mean | rollout NRMSE mean | parameters median | effective rank mean | top-k energy mean | seeds |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | common | -0.24178 | -0.001465 | 1.0132 | 514 | 1 | 1 | 20 |
| 1 | orthogonal | -0.24034 | -0.0027023 | 1.0133 | 517 | 1 | 1 | 20 |
| 1 | random | -0.24 | -0.0029985 | 1.0132 | 390 | 1 | 1 | 20 |
| 1 | separate | -0.22606 | -0.0020677 | 1.0138 | 900 | 1 | 1 | 20 |
| 1 | shared | -0.24161 | -0.0022191 | 1.0138 | 517 | 1 | 1 | 20 |
| 1 | shuffled | -0.24078 | -0.002951 | 1.0135 | 517 | 1 | 1 | 20 |
| 2 | common | -0.24131 | -0.0010539 | 1.0132 | 645 | 1.8781 | 1 | 20 |
| 2 | orthogonal | -0.23941 | -0.0028207 | 1.0133 | 653 | 1.8206 | 1 | 20 |
| 2 | random | -0.23914 | -0.0029968 | 1.0132 | 400 | 1.7331 | 1 | 20 |
| 2 | separate | -0.2258 | -0.0017671 | 1.0138 | 1162 | 1.6917 | 1 | 20 |
| 2 | shared | -0.24111 | -0.0019306 | 1.0139 | 653 | 1.7754 | 1 | 20 |
| 2 | shuffled | -0.23996 | -0.0029083 | 1.0135 | 653 | 1.4744 | 1 | 20 |
| 4 | common | -0.23984 | -0.00074294 | 1.0132 | 910 | 3.4115 | 1 | 20 |
| 4 | orthogonal | -0.23728 | -0.0027432 | 1.0132 | 934 | 3.3128 | 1 | 20 |
| 4 | random | -0.23737 | -0.0030002 | 1.0132 | 432 | 3.2045 | 1 | 20 |
| 4 | separate | -0.22433 | -0.0013984 | 1.0138 | 1692 | 2.839 | 1 | 20 |
| 4 | shared | -0.23965 | -0.0015668 | 1.0139 | 934 | 3.0297 | 1 | 20 |
| 4 | shuffled | -0.23821 | -0.0028596 | 1.0135 | 934 | 2.6385 | 1 | 20 |
| 8 | common | -0.23589 | -0.00039776 | 1.0132 | 1452 | 6.171 | 0.92093 | 20 |
| 8 | orthogonal | -0.23226 | -0.0026703 | 1.0131 | 1532 | 6.351 | 0.90018 | 20 |
| 8 | random | -0.23375 | -0.0029648 | 1.0132 | 544 | 6.2117 | 0.90774 | 20 |
| 8 | separate | -0.21975 | -0.0010906 | 1.0138 | 2776 | 5.1323 | 0.96973 | 20 |
| 8 | shared | -0.23572 | -0.0010673 | 1.014 | 1532 | 5.442 | 0.95265 | 20 |
| 8 | shuffled | -0.23449 | -0.0027685 | 1.0136 | 1532 | 5.4309 | 0.95226 | 20 |
| 16 | common | -0.22544 | -9.6519e-05 | 1.0132 | 2584 | 11.293 | 0.78963 | 20 |
| 16 | orthogonal | -0.22181 | -0.0027768 | 1.0131 | 2872 | 12.563 | 0.64897 | 20 |
| 16 | random | -0.22549 | -0.0029089 | 1.0133 | 960 | 12.446 | 0.66682 | 20 |
| 16 | separate | -0.20788 | -0.00091281 | 1.0138 | 5040 | 10.688 | 0.83272 | 20 |
| 16 | shared | -0.22529 | -0.00052663 | 1.0139 | 2872 | 10.9 | 0.8132 | 20 |
| 16 | shuffled | -0.22631 | -0.0026535 | 1.0136 | 2872 | 11.667 | 0.75081 | 20 |
| 32 | common | -0.19751 | 5.5751e-05 | 1.0132 | 5040 | 22.988 | 0.58567 | 20 |
| 32 | orthogonal | -0.1999 | -0.0033177 | 1.0132 | 6128 | 25.341 | 0.39549 | 20 |
| 32 | random | -0.20478 | -0.0030931 | 1.0134 | 2560 | 24.875 | 0.43781 | 20 |
| 32 | separate | -0.17732 | -0.0015344 | 1.0138 | 9952 | 23.336 | 0.57524 | 20 |
| 32 | shared | -0.19722 | -0.00038505 | 1.0139 | 6128 | 23.352 | 0.56698 | 20 |
| 32 | shuffled | -0.20601 | -0.0026765 | 1.0136 | 6128 | 24.458 | 0.48185 | 20 |

## Interpretation boundary

- The Gaussian LDS likelihood is a predictive population score for binarized calcium-event vectors; it is not a Bernoulli spike likelihood or a causal recurrent-connectivity estimate.
- Responding and spontaneous are separate recordings without within-recording switch timestamps. This tests cross-context parameter sharing, not natural fast switching.
- The source files contain no trial, behavior, animal, E/I, or anatomical-coordinate metadata. Those claims are not tested here.
- `minimal_computation_python` estimates equal-time direct dependencies and minimal input count; it is reported separately from latent rank.

## Artifacts

- `results/raw_metrics.csv`: all attempts, including failures.
- `results/latest_metrics.csv`: latest complete/failed planned cells for the selected profile.
- `results/model_summary.csv`: fold-within-seed aggregation.
- `results/comparisons.csv`: paired computational robustness contrasts.
- `results/summary.csv`: formal three-category claim table.
