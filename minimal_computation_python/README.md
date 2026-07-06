# Minimal Computation Python Reproduction

Python translation and lightweight reproduction for
[ChrisWLynn/Minimal_computation](https://github.com/ChrisWLynn/Minimal_computation),
the code accompanying "Direct dependencies between neurons explain activity".

The implementation keeps the core MATLAB logic:

- binary activity matrix `X` shaped neurons x time
- mutual information from pairwise binary statistics
- maximum-entropy/logistic neuron model matching output mean and selected pairwise correlations
- greedy minimax input selection by approximate entropy drop
- model entropy and residual pairwise-correlation error curves

The default reproduction intentionally uses a small single-neuron hippocampus
example so it runs locally without MATLAB or SciPy.

The upstream MATLAB code and `.mat` datasets are not vendored in this release.
To rerun sweeps, clone the upstream repository inside the `neuro_modeling/` root:

```powershell
cd /path/to/neuro_modeling
git clone https://github.com/ChrisWLynn/Minimal_computation minimal_computation_original
```

## Run

```powershell
cd /path/to/neuro_modeling/minimal_computation_python
python -B run_reproduction.py --dataset hippocampus --neuron 13 --max-inputs 30 --sweep 1,2,3,5,8,13,21,30
```

Outputs are written under `results/` and `figures/`.
