# Local Plasticity to Gated Low-Dimensional Dynamics

This repository tests a revised, more precise hypothesis: local eligibility
traces combined with a low-dimensional credit signal can shape a small number
of controllable/observable modes on top of a high-rank sparse E/I substrate.
Low-dimensional gates may then modify effective dynamics without requiring the
physical recurrent matrix or its masked update to be low rank. The repository
also contains shared-subspace prototypes for public sequence-memory and IBL
data. Its only neural IBL result is the legacy one-session `exp06` pilot and
remains inconclusive; `exp11` is a distinct trials-only behavior benchmark.

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
`exp12_*.py` are executable entry points as implementation advances. `exp07`
is the strict P0 pairing/budget experiment and `exp08` audits rank stages and
effective dimensions. `exp09` is the leakage-safe hidden-HMM gate audit.
`exp10` connects frozen hidden beliefs to a shared Dale E/I receiver through a
rank-one gain axis, `exp11` evaluates past-only online hidden-block inference
on IBL behavioral trial tables (without spikes or neural activity), and `exp12`
provides a secondary frozen-candidate ARC-style routing contract. Formal runs write immutable run
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
.\.venv\Scripts\python.exe experiments\exp09_hidden_context_gate.py `
  --config configs\formal\exp09_hidden_context_gate.json --results-root results
.\.venv\Scripts\python.exe experiments\exp10_hidden_context_ei_bridge.py `
  --config configs\formal\exp10_hidden_context_ei_bridge.json --results-root results
.\.venv\Scripts\python.exe scripts\build_report.py --results-root results --plots
```

The incremental validation system has three deliberately separated evidence
layers:

1. `exp10` is a functional bridge pipeline. Both sensory streams enter one
   frozen E/I checkpoint unchanged; only population gain is belief dependent.
   Base gates receive separately train-fitted readouts, whereas MD-like
   post-fit interventions reuse its intact readout. Thus the base comparison
   tests functional sufficiency, while only the intervention panel is a fixed-
   checkpoint within-model counterfactual test. The recurrent matrix is never
   trained.
2. `exp11` is a real-data behavior test. Trial `t` beliefs use stimulus-side
   history only through `t-1`, never reset at true block boundaries, and are
   scored with `probabilityLeft` only after predictions are frozen. The
   no-belief comparator retains a strong past-only choice/history readout.
3. `exp12` is a secondary functional interface. It consumes a frozen candidate
   tape and splits complete tasks. Tape hashes and train-example source IDs are
   schema attestations, not independently reproduced feature provenance, so the
   current adapter is fail-closed for scientific claims even on a formal tape.
   ARC/maze/Sudoku results cannot replace neural evidence.

### Freeze and run the IBL behavior cohort

The cohort freezer requires the official
[`paper-brain-wide-map`](https://github.com/int-brain-lab/paper-brain-wide-map)
repository and ONE-api. It downloads only `_ibl_trials.table.pqt`, balances
selection across animals, and writes every attempted exclusion/failure to an
immutable manifest:

```bash
python scripts/freeze_ibl_behavior_cohort.py \
  --bwm-repo /path/to/paper-brain-wide-map \
  --output-root data/ibl_behavior \
  --cache-dir data/ibl_cache \
  --cohort-id bwm_behavior_v1
python experiments/exp11_ibl_behavior_belief.py \
  --config configs/formal/exp11_ibl_behavior_belief.json \
  --results-root results
python figures/exp11_ibl_behavior_plot.py \
  --results-root results --n-bootstrap 100000
python scripts/build_report.py --results-root results --plots
```

The formal real-data config fixes one algorithmic seed; replication is across
session/animal, never seed x session. At least 20 eligible sessions and 5
animals are enforced before exp11 starts.

### Current incremental pilot

The committed `exp10_bridge_pilot` artifact is a small N=32, 30-seed bridge
pilot, not the N=256 formal run. The learned-HMM pipeline differed from a
separately optimized no-gate pipeline by +2.52 held-out balanced-accuracy points
(seed-bootstrap 95% CI 0.75 to 4.38; one-sided paired Wilcoxon Holm-adjusted
`p=0.032`). The current MD-like posterior-to-gain bridge improved by only 0.33
points (CI 0.00 to 0.93; Holm `p=0.315`) and its clamp/delay/shuffle contrasts
were inconclusive. These null/inconclusive MD results are retained rather than replaced
by a stronger input-routing mechanism.

`exp09` separates cue observations, task-facing inputs, and evaluation truth
into immutable capabilities. The learned HMM and MD-like recurrent belief gate
fit cue episodes only; the supervised gate is an ineligible upper bound, and
the oracle knows the generative q/h values but never the realized state.
The MD candidate is a named hybrid—past-only two-slice local soft counts with
Hebbian lag-1--5 moment shrinkage—and its component parameter estimates are
saved separately to avoid attributing results to a pure soft-count rule.
Clamp, one-trial delay, and trajectory derangement are applied after fitting to
the frozen MD belief trajectory. This is deliberately a gate-only behavioral
benchmark: it tests hidden-context inference and effective control, not yet a
coupled N=256/512 recurrent PFC E/I implementation.

For `exp07`, normalization is an explicit intervention axis rather than a hidden
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
session-level artifact. Legacy neural `exp06` lazily imports ONE-api, selects
1--5 public sessions, caches downloads under `data/ibl_cache/`, and analyzes
stimulus-pre and movement-pre windows; its current one-session result is not
leakage-safe enough for support. Behavior-only `exp11` instead consumes a
frozen cohort of at least 20 sessions/5 animals and never loads neural activity.

`results/raw_metrics.csv.gz` losslessly retains every complete, invalid, and
failed condition. The ignored uncompressed CSV is regenerated as a local
plotting cache.
`results/summary.csv` uses only the latest formal run attempt per experiment and
seed for inference while preserving all earlier attempts in the raw table and
run-coverage report. Figures compute uncertainty across seeds or sessions, never
across neurons or folds.
