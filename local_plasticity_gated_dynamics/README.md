# Local Plasticity to Gated Low-Dimensional Dynamics

This repository tests whether local three-factor plasticity constrained by
low-dimensional feedback and E/I homeostasis produces reusable, rapidly
switchable low-rank recurrent dynamics. It also provides leakage-safe shared
subspace models for public sequence-memory and IBL data.

The scientific protocol is intentionally falsifiable. Every core claim is
classified as `support`, `oppose`, or `inconclusive`; failed seeds and missing
external-data runs are retained as first-class results.

Read [LIMITATIONS.md](LIMITATIONS.md) before interpreting the committed formal
summary. In particular, the current Phase 2 behavior does not identify aligned
feedback as the causal mechanism, and mask/Dale/normalization operations do not
preserve the raw low-rank update bound.

## Reproducibility contract

- Python 3.11 only.
- All stochastic entry points receive and record an explicit seed.
- The local-learning models use NumPy local updates and never autograd or BPTT.
- BPTT exists only in an isolated baseline module/experiment path.
- Trials or blocks, never individual time points, are split across folds.
- Scaling, PCA, subspaces, and nuisance regressions are fit on training data.
- Inference treats seed, session, or animal as the replicate unit.

## Layout

The requested modules live under `src/`; `experiments/exp00_*.py` through
`exp06_*.py` are executable entry points. Formal runs write immutable run
folders under `results/runs/`. `scripts/build_report.py` aggregates all statuses
(including failures) into `results/summary.csv` and `results/report.md`.

## Reproduce

On Windows, the bootstrap script installs the project-local Python 3.11 runtime,
creates `.venv`, and installs the locked project requirements:

```powershell
./scripts/bootstrap_windows.ps1
.\.venv\Scripts\python.exe -m pytest -q
```

Each experiment accepts a JSON config, an optional comma-separated seed
override, and a results root. For example:

```powershell
.\.venv\Scripts\python.exe experiments\exp01_feedback_dimension_sweep.py `
  --config configs\formal\exp01_feedback_dimension_sweep.json --results-root results
.\.venv\Scripts\python.exe experiments\exp04_phase_gating.py `
  --config configs\formal\exp04_phase_gating.json --results-root results
.\.venv\Scripts\python.exe experiments\exp06_ibl_context_switch.py `
  --config configs\formal\exp06_ibl_context_switch.json --results-root results
.\.venv\Scripts\python.exe scripts\build_report.py --results-root results --plots
```

The sequence-memory loader expects one directory per session beneath
`data/raw/sequence_memory/`, each containing `trials.csv`, `units.csv`, and
`spikes.mat`. Missing or access-restricted data is written as a failed
session-level artifact. The IBL experiment lazily imports ONE-api, selects 1–5
public sessions, caches downloads under `data/ibl_cache/`, and analyzes only
stimulus-pre and movement-pre windows.

`results/raw_metrics.csv` retains every complete, invalid, and failed condition.
`results/summary.csv` uses only the latest formal run attempt per experiment and
seed for inference while preserving all earlier attempts in the raw table and
run-coverage report. Figures compute uncertainty across seeds or sessions, never
across neurons or folds.
