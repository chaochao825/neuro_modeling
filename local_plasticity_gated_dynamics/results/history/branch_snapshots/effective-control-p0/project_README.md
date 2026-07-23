# Local Plasticity to Gated Low-Dimensional Dynamics

This repository tests a revised, more precise hypothesis: local eligibility
traces combined with a low-dimensional credit signal can shape a small number
of controllable/observable modes on top of a high-rank sparse E/I substrate.
Low-dimensional gates may then modify effective dynamics without requiring the
physical recurrent matrix or its masked update to be low rank. The repository
also provides leakage-safe shared-subspace models for public sequence-memory
and IBL data.

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
- Tuned BPTT rate-RNN and GRU models exist only in isolated baseline paths;
  candidate selection uses validation blocks and the outer test set is never
  accepted by the tuning/refit API.
- Trials or blocks, never individual time points, are split across folds.
- Scaling, PCA, subspaces, and nuisance regressions are fit on training data.
- Inference treats seed, session, or animal as the replicate unit.

## Layout

The requested modules live under `src/`; `experiments/exp00_*.py` through
`exp08_*.py` are executable entry points as implementation advances. `exp07`
is the strict P0 pairing/budget experiment and `exp08` audits rank stages and
effective dimensions. Formal runs write immutable run
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
.\.venv\Scripts\python.exe experiments\exp07_mechanism_identifiability.py `
  --config configs\formal\exp07_mechanism_identifiability.json --results-root results
.\.venv\Scripts\python.exe experiments\exp08_rank_stage_validation.py `
  --config configs\formal\exp08_rank_stage_validation.json --results-root results
.\.venv\Scripts\python.exe scripts\build_report.py --results-root results --plots
```

For `exp07`, normalization is an explicit causal axis rather than a hidden
side effect. Each L1/L2 panel contains task-only feedback geometries plus
task/homeostasis/normalization combinations, including matched cells with and
without normalization. The so-called homeostasis tape in P0 is transparently
recorded as a *yoked inhibitory-strengthening control*; it is not evidence for
closed-loop E/I stability, which remains a P4 question. Frozen-reference rates
and unprojected feedback signals are materialized once, made read-only, and
content-fingerprinted before any branch runs. Identity feedback reports both
its projector rank and the empirical span of the signal it actually receives.
The shuffled control applies an exact within-block/context permutation to the
complete precomputed third-factor vector, preserving its empirical marginal
while breaking trial correspondence.

The sequence-memory loader expects one directory per session beneath
`data/raw/sequence_memory/`, each containing `trials.csv`, `units.csv`, and
`spikes.mat`. Missing or access-restricted data is written as a failed
session-level artifact. The IBL experiment lazily imports ONE-api, selects 1–5
public sessions, caches downloads under `data/ibl_cache/`, and analyzes only
stimulus-pre and movement-pre windows.

`results/raw_metrics.csv.gz` losslessly retains every complete, invalid, and
failed condition. The ignored uncompressed CSV is regenerated as a local
plotting cache.
`results/summary.csv` uses only the latest formal run attempt per experiment and
seed for inference while preserving all earlier attempts in the raw table and
run-coverage report. Figures compute uncertainty across seeds or sessions, never
across neurons or folds.
