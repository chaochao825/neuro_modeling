# Neural Multiscale Validation Report

This report separates mechanistic evidence instead of claiming a unified theory is proven.

## Decision Matrix

| Hypothesis | Level | Key evidence |
|---|---:|---|
| H1_history_local_coupling | strong | history delta=0.0021 bits/bin, local delta=0.0077 |
| H2_nearcritical_powerlaw_spectrum | strong | alpha=0.914, Lyapunov log-eig corr=0.989 |
| H3_oscillatory_synchrony | weak | PSD peak ratio=13.17, PLV=0.722, complex modes=0, reset=-0.012 |
| H4_avalanche_criticality | strong | near m branching=0.976, best dynamic-range m=1.0 |
| H5_energy_constraint | strong | best sparsity=0.09, long-range=0.035, rho=0.900 |

## Controls and Caveats

- Baseline independent Bernoulli activity is used as a negative control for correlations, spectrum, DMD modes, and avalanches.
- Public-data scripts do not download large datasets by default; they analyze local exported matrices with the same metrics.
- Avalanche evidence requires model comparison and branching/dynamic-range checks. A log-log tail alone is not accepted.
- Oscillation evidence is downgraded unless phase locking, complex modes, narrowband PSD, and reset proxy align.

## Selected Raw Metrics

- Baseline mean abs cross-correlation: 0.0290
- Hawkes mean abs cross-correlation: 0.0328
- Critical linear eigenspectrum alpha: 0.9136
- Gamma-sync PSD peak ratio: 13.1750
