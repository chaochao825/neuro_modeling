# Neuro Modeling: Multiscale Neural Population Tests

This repository collects four related but evidence-separated workstreams:

1. `local_plasticity_gated_dynamics/`: the tested core project for local three-factor plasticity, E/I constraints, gates, and shared-subspace models.
2. `shared_dynamics_real_data/`: leakage-safe contiguous-block validation of common, shared-basis switching, and separate-basis LDS models on public recordings.
3. `minimal_computation_python/`: the direct-dependency workflow, with the original residual approximation retained as a baseline and an exact block-Schur selector under test.
4. `neural_multiscale_tests/`: legacy single-seed synthetic H1-H5 calibration and auxiliary diagnostics.

The repository does not vendor the upstream `Minimal_computation` MATLAB code or `.mat` data because redistribution terms were not stated in the working copy used for this release. To rerun the Minimal_computation sweeps, clone the upstream repository inside this repository root as `minimal_computation_original/`.

## Current Result Snapshot

As of 2026-07-11:

- The clean synthetic latent task supports rank-matched aligned feedback at dimension 4 over 20 seeds.
- The current E/I implementation does **not** preserve that rank after derivative modulation, sparse masks, Dale projection, and normalization; its feedback and homeostasis mechanisms are not behaviorally identified.
- Rate-matched phase gating is opposed by the completed formal test; sequence-memory and multi-session IBL claims remain inconclusive.
- The legacy H1-H5 labels are `synthetic_calibration / single_seed`, not inferential evidence.
- For C. elegans neuron 13 only, the previously reported `21 inputs` value came from the approximate selector. The MATLAB-parity block-Schur rerun reaches the residual criterion at 7 inputs (MATLAB 1-based IDs `[74,34,3,1,123,54,120]`) and supersedes that lightweight number.
- The formal public visual panel completed all 2,160 planned cells (20 deterministic neuron-subset seeds, 3 contiguous folds, 6 dimensions, 6 models). At `d=4`, shared switching was slightly worse than common dynamics (median shared-minus-common NLL `+2.03e-4`), and its median-based absolute prediction margin was negative (`-0.0137`; the corresponding seed means were one-step R2 `-0.00157` and rollout NRMSE `1.01395`). Aligned shared dynamics did beat all three basis controls descriptively at `d=4` (median minimum NLL gap `+0.00134`).
- At `d=4`, common dynamics also dominates shared on the reported NLL, one-step R2, rollout error, and fitted-parameter count. The aligned-vs-basis-control result therefore does not establish a switching-model advantage.
- That visual comparison is one aligned recording pair, not 20 biological replicates. Every population-level real-data claim therefore remains `inconclusive`; the within-recording directions above are robustness diagnostics only.

Current reports:

- `docs/integrated_method_audit_zh.md`
- `local_plasticity_gated_dynamics/results/report.md`
- `minimal_computation_python/results/c_elegans_matlab_schur_neuron13_max32_report.md`
- `shared_dynamics_real_data/results/report.md`
- `shared_dynamics_real_data/figures/visual_context_shared_dynamics.png`

Historical calibration artifacts (kept for provenance, not current evidence):

- `neural_multiscale_tests/reports/integrated_goal_status_zh.md`
- `neural_multiscale_tests/figures/integrated_goal_status.png`
- `minimal_computation_python/results/paper_comparison_report_zh.md`
- `minimal_computation_python/figures/paper_comparison_summary.png`

## Quick Reproduction

Run the synthetic H1-H5 framework:

```bash
cd /path/to/neuro_modeling/neural_multiscale_tests
python -B run_simulations.py --quick --seed 7
python -B figures/integrated_goal_status_plot.py
python -B -m unittest discover -s tests
```

Run the Minimal_computation Python smoke tests:

```bash
cd /path/to/neuro_modeling/minimal_computation_python
python -B -m unittest discover -s tests
```

Run the current MATLAB-parity C. elegans selector:

```bash
cd /path/to/neuro_modeling
git clone https://github.com/ChrisWLynn/Minimal_computation minimal_computation_original
cd /path/to/neuro_modeling/minimal_computation_python
python -B run_reproduction.py --dataset c_elegans --neuron 13 --max-inputs 32 \
  --selector schur_entropy_drop --completion-mode matlab_residual \
  --failure-selection matlab_last
```

The other three committed single-neuron JSON files are historical
`residual_approximation` sweeps, not Schur-selector results. Exact commands for
both modes and the formal shared-dynamics panel are in `REPRODUCE.md`.

## Boundary of Claims

The synthetic framework demonstrates that the pipeline can distinguish some competing mechanisms under controlled simulations. It does not prove a unified brain theory, and the legacy decision scores are not statistical tests.

The bundled public MAT files lack trials, behavior, E/I identity, and natural switch annotations. They can test held-out shared low-dimensional structure across recordings, but not rapid behavioral switching or E/I homeostasis. Those claims require multiple sessions/animals with explicit covariates.
