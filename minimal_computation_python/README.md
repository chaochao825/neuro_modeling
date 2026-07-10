# Minimal Computation Python Reproduction

Python translation and lightweight reproduction for
[ChrisWLynn/Minimal_computation](https://github.com/ChrisWLynn/Minimal_computation),
the code accompanying "Simple input-output dependencies explain neuronal
activity" (Nature Physics, 2026).

The default implementation now follows the core MATLAB logic:

- binary activity matrix `X` shaped neurons x time
- mutual information from pairwise binary statistics
- maximum-entropy/logistic neuron model matching output mean and selected pairwise correlations
- exact pairwise-MI first selection followed by the analytic Schur
  entropy-drop criterion
- candidate-block computation of the Schur diagonal (no full `N x N`
  Hessian allocation)
- MATLAB-style reset initialization for each optimizer exponent
- MATLAB-parity completion based on normalized residual correlation error;
  optimizer convergence is recorded separately, as in the upstream control flow
- MATLAB-parity use of the final learning-rate exponent when none converge
- coarse sweep followed by binary refinement of the smallest complete set
- model entropy and residual pairwise-correlation error curves

The former normalized-residual selector and warm-start behavior remain
available explicitly as baselines (`--selector residual_approximation` and
`--initialization warm_start`). Dataset-specific optimizer thresholds and
MATLAB sweep families are recorded in every result's `run_config`.

For the more conservative Python criterion that also requires optimizer
convergence, pass `--completion-mode strict_optimizer_and_residual`.  For the
former best-error fallback across failed exponents, pass
`--failure-selection best_error`.  These modes are named separately and are
not labeled MATLAB parity.

The upstream MATLAB code and `.mat` datasets are not vendored in this release.
To rerun sweeps, clone the upstream repository inside the `neuro_modeling/` root:

```powershell
cd /path/to/neuro_modeling
git clone https://github.com/ChrisWLynn/Minimal_computation minimal_computation_original
```

## Run

```powershell
cd /path/to/neuro_modeling/minimal_computation_python
python -B run_reproduction.py --dataset c_elegans --neuron 13 --max-inputs 32 \
  --selector schur_entropy_drop --completion-mode matlab_residual \
  --failure-selection matlab_last
```

This exact configuration reached the residual criterion at 7 inputs for C.
elegans neuron 13 (all neuron IDs are MATLAB-style 1-based). The committed hippocampus and visual JSON files predate the
Schur implementation and remain historical residual-ranking baselines; they
must not be presented as current MATLAB-parity outputs.

To run the historical Python selector as a named baseline:

```powershell
python -B run_reproduction.py --dataset hippocampus --neuron 13 --max-inputs 30 \
  --sweep 1,2,3,5,8,13,21,30 --selector residual_approximation \
  --initialization warm_start --completion-mode strict_optimizer_and_residual \
  --failure-selection best_error --no-binary-search
```

Each execution writes an immutable
`results/runs/<timestamp>_<config-fingerprint>/` directory containing config,
status, metrics, report, log, and figure artifacts. The selector is part of the
saved config and fingerprint, so different methods cannot silently overwrite
one another. The fixed-name C. elegans files in `results/` are a compact copy
of the verified Schur run for this release.

The compact result replaces the server's absolute source path with
`${REPO_ROOT}`. It records both the original run fingerprint and a separate
fingerprint for the published path-sanitized payload; numeric metrics and the
source-data SHA-256 are unchanged.

This analysis models equal-time conditional dependence `P(y_t | x_t)`.  Its
minimal input count is neither a causal connectivity estimate nor the latent
rank of a future-dynamics model.  It also uses the full recording for the
paper-parity descriptive fit; held-out temporal prediction is implemented
separately in `../shared_dynamics_real_data/`.
