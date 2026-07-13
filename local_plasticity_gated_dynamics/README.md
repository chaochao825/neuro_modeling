# Local Plasticity to Gated Low-Dimensional Dynamics

This repository tests a revised, more precise hypothesis: local eligibility
traces combined with a low-dimensional credit signal can shape a small number
of controllable/observable modes on top of a high-rank sparse E/I substrate.
Low-dimensional gates may then modify effective dynamics without requiring the
physical recurrent matrix or its masked update to be low rank. The repository
also contains shared-subspace prototypes for public sequence-memory and IBL
data. The legacy one-session `exp06` pilot remains inconclusive; `exp11` is a
distinct trials-only behavior benchmark. `exp14` now provides a fail-closed
multi-session count-dynamics pipeline, a reviewed hash-bound 20-session/
20-animal compact cache, and a completed registered comparison. The registered
primary result is inconclusive and does not support the shared-dynamics claim.

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
`exp15_*.py` are executable entry points as implementation advances. `exp07`
is the strict P0 pairing/budget experiment and `exp08` audits rank stages and
effective dimensions. `exp09` is the leakage-safe hidden-HMM gate audit.
`exp10` connects frozen hidden beliefs to a shared Dale E/I receiver through a
rank-one gain axis, `exp11` evaluates past-only online hidden-block inference
on IBL behavioral trial tables (without spikes or neural activity), and `exp12`
provides a secondary frozen-candidate ARC-style routing contract. `exp13` adds
a target-isolated hybrid ARC/maze/Sudoku benchmark with continuous fast/slow
states, an explicit low-dimensional control bottleneck, and an optional
discounted bilinear rate trace; spikes are not required. Formal runs write immutable run
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
.\.venv\Scripts\python.exe scripts\prepare_exp13_public_benchmarks.py `
  --family all --output-root data\structured
.\.venv\Scripts\python.exe experiments\exp13_structured_reasoning.py `
  --config configs\formal\exp13_structured_reasoning_arc.json --results-root results
.\.venv\Scripts\python.exe scripts\summarize_exp13.py `
  --config configs\formal\exp13_structured_reasoning_arc.json --results-root results
.\.venv\Scripts\python.exe figures\exp13_structured_reasoning_plot.py `
  --results-root results
# Replace `arc` with `maze` or `sudoku` in both the formal config and prefix:
.\.venv\Scripts\python.exe scripts\summarize_exp13.py `
  --config configs\formal\exp13_structured_reasoning_maze.json `
  --results-root results --output-prefix exp13_maze_formal
.\.venv\Scripts\python.exe figures\exp13_structured_reasoning_plot.py `
  --results-root results --prefix exp13_maze_formal
.\.venv\Scripts\python.exe experiments\exp14_ibl_multisession_neural.py `
  --config configs\formal\exp14_ibl_multisession_neural.json --results-root results
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
4. `exp13` is a stronger but still separate functional track. Public tasks
   expose demonstrations and query inputs but never query targets. One
   recomputed proposal panel is shared by the heuristic, flat local,
   hierarchical local, trace local, GRU/BPTT, and candidate-oracle conditions.
   This is a hybrid proposal selector, not an official HRM/CTM reproduction or
   a proposal-free neural solver. ARC source/augmentation dependency components
   are the statistical unit; seed replicates are averaged within task before
   component-level inference. The known exact
   ARC-AGI-1 train/evaluation duplicate is explicitly excluded and recorded.
   The former unlicensed Maze-Hard/Sudoku-Extreme placeholders are replaced by
   pinned licensed sources. The auditable builder retains 79/79 reachable
   MazeBench images (61 train, 18 grid-size OOD test; 31 upstream-unreachable
   images remain in the manifest). Sudoku's official 160/40 lists contain
   extensive exact-puzzle duplication: all 200 source receipts are retained,
   while content addressing and test precedence yield 76 unique puzzles (48
   train, 28 non-OOD test) and audit 124 duplicate exclusions. Both formal
   30-seed runs are complete. Maze supports only the registered selector-level
   90%-of-GRU retention endpoint; its hierarchical-over-flat endpoint is
   inconclusive. Sudoku is at a 100% ceiling for every condition, leaving all
   registered mechanism comparisons inconclusive.
5. `exp14` is the distinct neural track. A past-only learned HMM produces soft
   beliefs without a `probabilityLeft` capability; raw non-negative counts are
   adjusted through past-safe nuisance terms in the session log-rate map,
   rather than residualized into invalid non-count targets. Whole chronological
   blocks, unit selection, normalization/PCA, and the observation maps are
   train-only. Common, shared-belief, and session-full models share exactly the
   same preprocessing and observations. The current endpoint is explicitly a
   teacher-forced within-trial one-step Poisson likelihood, not a full latent
   Poisson LDS. Registered inference is animal-primary with sessions nested.
6. `exp15` is an additive task-specialization track. Sudoku uses sparse positive
   candidate activity with local row/column/box interactions; ARC uses a slow
   operator-family belief and fast demonstration-grounded program selection.
   These are task adapters inspired by design principles, not BDH or HRM
   reproductions. Spikes and BPTT are absent. The verified-source ARC panel now
   compares slow/fast belief with a flat selector using deterministically
   regenerated, fingerprint-identical candidate panels and an exactly matched
   charged abstract-operation budget. Its claim
   remains fail-closed on source, OOD, pairing, compute, and candidate-coverage
   gates. See `docs/task_specialized_reasoning_contract_zh.md`.

### Current exp13 public ARC result

The clean 30-seed ARC-AGI-1 audit evaluated 399 de-duplicated public evaluation
tasks. The deterministic target-free proposal library covered only 5 tasks
(1.253%), far below the registered 90% coverage gate. Exact-task accuracy was
0.301% for flat local, 0.301% for hierarchical local, 0.343% for trace local,
0.501% for GRU/BPTT, and 1.253% for the candidate oracle. All six registered
contrasts are therefore `inconclusive`; this is a leakage-safe negative audit,
not evidence for hierarchical advantage or a competitive ARC solver.

### Current exp13 public Maze/Sudoku results

On the 18 grid-size-OOD MazeBench test tasks, exact accuracy was 88.89% for the
support heuristic, 99.44% for flat local, 99.07% for hierarchical local,
98.89% for trace local, and 100% for GRU/BPTT. Hierarchical versus flat was
-0.37 percentage points (95% CI [-2.04, 0.74]) and is `inconclusive`.
Hierarchical local did satisfy the separately registered 90%-of-GRU retention
endpoint (margin 9.07 points, 95% CI [7.41, 10.00], Holm-adjusted
`p=0.000352`), which supports selector-level accuracy retention rather than
superiority or end-to-end efficiency. On the 28 de-duplicated Sudoku V2 test puzzles, every selector,
heuristic, and oracle scored 100%; all mechanism comparisons are
`inconclusive` because the benchmark is saturated. Both tasks use the same
deterministic target-free proposal/search library for every learned selector,
so neither result is an end-to-end neural reasoning claim.

### Current exp15 task-specialized results

The clean Exp15 ARC run verified all 800 ARC-AGI-1 JSON files and the license,
then evaluated 399 de-duplicated evaluation tasks. Slow/fast belief and the flat
matched selector each solved 1/399 tasks: 0.2506% exact (95% source-group CI
0–0.7519%). Their paired difference is 0 percentage points (95% CI [0, 0],
Holm `p=1`). Candidate fingerprints and charged compute match exactly, but the
finite proposal library covers only 5/399 tasks (1.2531%), far below the
registered 90% gate. The conclusion is therefore `inconclusive`, not evidence
for hierarchical advantage. The compute quantity is an audited abstract proxy,
not FLOPs, wall-clock time, energy, or end-to-end efficiency. See the
[scoped report](results/exp15_arc_matched_formal_report.md).

On 28 de-duplicated Sudoku V2 puzzles, pure local constraint dynamics achieved
75.0% exact accuracy (95% CI 57.14–89.29%); the separately reported
bounded-search condition (up to 256 branches) achieved 100%. Because branch
search is a distinct mechanism and the Sudoku split is non-OOD, the Sudoku
mechanism conclusion remains `inconclusive`.

### exp14 multi-session neural status

The exp14 synthetic smoke path is complete and tests nested latent-dimension
selection, past-only belief receipts, train-only anatomical unit selection,
paired common/shared/full count models, exact Poisson likelihood, parameter
counting, and animal-with-session bootstrap. The formal profile cannot fall
back to synthetic data: it accepts only the reviewed compact cache derived from
the frozen 20-session/20-animal BWM panel (35 probes, 3,183 units, sorting
revision `2024-05-06`, good-unit threshold `>=1`). Acquisition and offline
count binning are complete and their manifests, failures, producer snapshot,
   and compact bundle are hash-bound. The repaired formal run retained all 340
   planned records with zero failures. The registered primary common-minus-
   shared contrast was -0.000995 NLL/count (95% CI [-0.003061, 0.000157],
   Holm-adjusted `p=1`) and is `inconclusive`; all three sensitivity panels are
   also inconclusive. The first pre-repair attempt remains preserved as
   `complete_with_failures` rather than being overwritten. The real-data
   preflight selected 1,347 of 3,183 units through training-fold-only
anatomical/variance criteria; formal likelihoods therefore apply to those
selected anchor units, not every recorded unit. Here `full` means
session-specific gated operators on the same six-region basis, not a full
latent LDS or an unrestricted unit-space model.

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
`p=0.032`). The current MD-like posterior-to-gain bridge point estimate was +0.33
points (CI 0.00 to 0.93; Holm `p=0.315`) and its clamp/delay/shuffle contrasts
were inconclusive. These null/inconclusive MD results are retained rather than
replaced by a stronger input-routing mechanism.

### Current exp10 N=256 formal result

The registered N=256 bridge grid completed all 30 independent seeds, four
cue-reliability/hazard cells, and seven conditions (840/840 rows; zero failed
conditions). After equal macro-averaging across the four cells, the learned-HMM
and MD-like functional pipelines improved held-out balanced accuracy over their
separately optimized no-gate pipelines by 10.00 points (seed-bootstrap 95% CI
9.62 to 10.38) and 9.64 points (CI 9.20 to 10.09), respectively. Both remain
significant after Holm correction across the nine formal exp10 comparisons
(`p=1.68e-8`). Because each base condition has its own fitted readout, these are
whole-pipeline comparisons rather than isolated fixed-readout gate effects.

At the frozen MD-like checkpoint and readout, clamp, one-trial delay, and
within-seed belief shuffling reduced balanced accuracy by 8.99, 2.23, and 9.92
points, respectively (all Holm `p=1.68e-8`). The MD-like pipeline retained at
least 90% of the oracle gain only under the registered equal-cell macro average:
the non-inferiority margin was +0.57 points (CI 0.34 to 0.78; Holm
`p=4.41e-5`), but one of the four q/h cell means was negative (-0.61 points).
This does not establish cell-wise non-inferiority. Recurrent weights are frozen,
so the result supports simulated hidden-context inference, a functional
belief-to-E/I control pipeline, and within-model gate counterfactuals—not
three-factor recurrent plasticity, a biological mechanism, or efficiency.

This formal snapshot was rerun from clean Git commit
`52fdcaa1e55ae0e0510ecca553c5acf6a4358072` with `dirty=false`. The scoped raw
SHA-256 is `5c2f37e12864a73e5d2202dbceb077f3caae147fa94c6ce94b3309f2656c9749`;
the clean 30-run manifest SHA-256 is
`b0e29f5053a37593a197832ee12adc93ccb80fb55bd65003f20f90fff67aba94`.
The manifest publishes each seed's run ID and hashes of its config, planned
conditions, status, manifest, environment, metrics, and log artifacts. Earlier
interrupted and dirty retries remain in run coverage but are ineligible for
inference.

### Current exp11 real-data result

The frozen BWM behavior cohort attempted 44 sessions and retained 30 sessions
from 30 different animals; 14 pre-specified QC exclusions and zero download
failures remain in the manifest. All 120 session-condition rows completed, and
all 30 learned HMM fits converged and passed the emission-separation gate.
The cohort manifest SHA-256 is
`112c84ad93eee49186ab117343ebebb4921d2f1bcea57a9c9326ca38d337a0e6`.

At the animal-primary level, the task-informed HMM (initialized from the known
0.2/0.8 task rates but fit without `probabilityLeft` labels) improved context
NLL over uniform belief by 0.3768 (hierarchical 95% CI 0.3313 to 0.4178;
Holm-adjusted `p=1.49e-8`), which supports hidden block-state inference. A simple exponential
history rule was significantly worse than uniform belief and is classified as
`oppose`. Neither learned-HMM nor exponential-history beliefs improved the
strong past-only held-out choice model (`inconclusive` for both). Thus the real
data support is limited to task-variable inference; it is not evidence for
neural activity, shared neural dynamics, or a biological gating mechanism.

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
