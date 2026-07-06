# Neuro Modeling: Multiscale Neural Population Tests

This repository collects the current reproducible methods and results for two related workstreams:

1. `neural_multiscale_tests/`: a synthetic, self-contained framework for separating H1-H5 mechanistic claims in neural population dynamics.
2. `minimal_computation_python/`: a Python reproduction of Chris W. Lynn's `Minimal_computation` workflow for direct dependencies between neurons.

The repository does not vendor the upstream `Minimal_computation` MATLAB code or `.mat` data because redistribution terms were not stated in the working copy used for this release. To rerun the Minimal_computation sweeps, clone the upstream repository inside this repository root as `minimal_computation_original/`.

## Current Result Snapshot

As of 2026-07-07:

- H1 history + local coupling: `strong` in the synthetic decision matrix.
- H2 near-critical / power-law eigenspectrum: `strong` in the synthetic decision matrix.
- H3 oscillatory synchrony code: `weak`; PSD and PLV are not treated as sufficient evidence without complex DMD and phase-reset support.
- H4 avalanche criticality: `strong` in the synthetic branching-process checks.
- H5 energy constraint: `strong` in the synthetic information/cost proxy sweep.
- Minimal computation reproduction: C. elegans reaches the complete-model criterion at 21 inputs; the current mouse hippocampus and visual sweeps show entropy reduction but do not reach complete within the lightweight input sweep.

Key integrated report:

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

Re-run the four reported Minimal_computation sweeps:

```bash
cd /path/to/neuro_modeling
git clone https://github.com/ChrisWLynn/Minimal_computation minimal_computation_original
cd /path/to/neuro_modeling/minimal_computation_python
python -B run_reproduction.py --dataset c_elegans --neuron 13 --max-inputs 32 --sweep 1,2,3,5,8,13,21,32
python -B run_reproduction.py --dataset hippocampus --neuron 13 --max-inputs 30 --sweep 1,2,3,5,8,13,21,30
python -B run_reproduction.py --dataset visual_spontaneous --neuron 13 --max-inputs 30 --sweep 1,2,3,5,8,13,21,30
python -B run_reproduction.py --dataset visual_responding --neuron 13 --max-inputs 30 --sweep 1,2,3,5,8,13,21,30
python -B figures/plot_paper_comparison.py
```

## Boundary of Claims

The synthetic framework demonstrates that the pipeline can distinguish competing mechanisms under controlled simulations. It does not prove a unified brain theory.

The public-data/real-experiment layer is incomplete: Allen, IBL, Steinmetz, Stringer, Buzsaki/CRCNS data loaders are represented as interfaces and registry entries, but full public data downloads, behavioral-state control, region/task stratification, and causal perturbation analyses remain future work.
