# Known limitations of the current formal results

The committed summary is an immutable snapshot of the completed runs, not a claim that every mechanism is identified.

This monorepo carries the compact evidence snapshot (`raw_metrics.csv.gz`,
`runs.csv`, `summary.csv`, the generated report, and PNG/PDF figures). The
timestamped `results/runs/` directories and per-attempt logs remain in the
source experiment workspace and are intentionally not duplicated here; the
committed configs and scripts regenerate that layout.

- Phase 1 is the clean constructive test: the aligned rank-4 update is rank matched and non-inferior to full feedback on the synthetic latent task.
- In Phase 2, derivative modulation, sparse masks, Dale projection, and fan-in normalization do not preserve the algebraic rank bound of the raw outer-product rule. The final E/I update is therefore not low rank even when activity remains low dimensional.
- In the legacy `exp02`/`exp03` conditions, local, full-feedback, and shuffled-feedback task accuracies are indistinguishable. The later strictly paired `exp07` audit finds a small aligned advantage under its matched controls, but still does not establish the complete recurrent E/I mechanism.
- Homeostatic updates dominate task-plastic updates in cumulative L1 budget. The existing homeostasis ablation is not budget matched and its stability result opposes the preregistered direction.
- The learned MD gate is a supervised Hebbian context classifier because its fit and modulatory features use true context information.
- The historical B1 snapshot passed only a relative branch against an untuned
  BPTT model and failed the absolute threshold. The current evaluator reports
  absolute performance, BPTT non-inferiority, and GRU non-inferiority as three
  separate claims.
- `exp07` completed all 30 formal seeds with exact selected-norm budget
  attainment. Its six Holm-adjusted P0 claims support mechanism
  identifiability, absolute seed-level mean accuracy, and the preregistered 90%
  baseline-retention margins. This is not parity or outperformance: mean local
  accuracy remains below tuned BPTT and GRU, five seeds fall below 0.85, and
  the aligned behavioral-accuracy advantage over frozen/shuffled is small.
- P0's matched homeostatic component is a yoked, one-sided inhibitory-
  strengthening nuisance control. It guarantees exact replay across feedback
  branches but does not establish firing-rate homeostasis, E/I balance, or
  closed-loop stability; those require the bidirectional P4 experiments.
- `exp08` distinguishes control-coordinate, physical-weight, mask, Dale, and
  normalization ranks. Its full-per-synapse parameterization currently samples
  a low-dimensional auxiliary slice, and cross-parameterization update budgets
  are not matched; Jacobian/PR/Hankel differences across those
  parameterizations are therefore descriptive rather than causal evidence.
- The P0 primary Jacobian maximum real part is positive in all 30 seeds. Normal
  Lyapunov, normal-perturbation decay, and formal closure error are not yet
  available, so the complete stable E/I low-dimensional-dynamics chain remains
  inconclusive.
- The legacy learned MD gate uses true-context information and cannot support
  the hidden-context claim. The sequence-memory dataset was unavailable. The
  legacy `exp06` IBL result contains one session/animal and is descriptive
  only; strict P6 support requires at least 5 animals and 20 sessions on a
  leakage-free shared hidden-context panel.
- `exp09` repairs the hidden-context leakage at the gate-only level and adds
  cue-only HMM/MD inference plus frozen post-fit interventions. The MD
  candidate combines past-only two-slice soft counts with 80% Hebbian multi-lag
  moment shrinkage when the cue process is identifiable; it is not a pure
  recurrent soft-count mechanism. It does not yet
  couple that belief gate to the local-plasticity N=256/N=512 recurrent PFC/E/I
  models, so even positive P2 gate results cannot by themselves establish the
  complete credit-assignment-to-recurrent-dynamics mechanism.
- `exp10` is the repository's first belief-to-Dale-E/I bridge, but recurrent
  weights remain frozen and the control axis is rank one by construction. The
  N=32 pilot remains a separate null/inconclusive MD intervention result. The
  registered N=256 grid completed all 30 seeds and supports separately refit
  HMM/MD-like functional pipelines plus fixed-checkpoint clamp, delay, and
  shuffle counterfactuals. The 90%-of-oracle margin is supported only after
  equal macro-averaging across four q/h cells; one cell has a negative margin,
  so cell-wise retention is not established. Because base pipelines use
  separately fitted readouts, and recurrence is frozen, these results cannot
  identify an isolated gate effect, three-factor recurrent learning, a
  biological mechanism, or an efficiency advantage. Formal inference uses the
  latest rerun from clean Git commit `52fdcaa1e55ae0e0510ecca553c5acf6a4358072`,
  bound to clean-run manifest SHA-256
  `b0e29f5053a37593a197832ee12adc93ccb80fb55bd65003f20f90fff67aba94`;
  earlier interrupted and dirty attempts remain visible but ineligible.
- `exp11` now has a frozen 30-session/30-animal BWM behavior cohort, with all
  120 session-condition rows complete and all learned HMM fits identifiable.
  The task-informed HMM is initialized from known 0.2/0.8 task rates, although
  fitting never sees trial-level `probabilityLeft` labels. Its hidden-block NLL
  supports task-variable context inference, while
  the exponential-history context model is opposed. Neither belief model
  improves held-out choice log loss over the strong past-only history readout,
  so behavioral prediction remains inconclusive. This trials-only result does
  not analyze neural activity and cannot support shared neural dynamics or a
  biological gating mechanism. The distinct MFD_09 `exp06` neural pilot is not
  exp11 behavior evidence.
- `exp12` validates only the task-safe candidate-routing interface on a
  synthetic smoke tape. No public ARC candidate tape has yet passed the frozen
  coverage contract, and the external feature/candidate extractor is not yet
  independently reproducible. Hash-shaped provenance fields are therefore
  treated only as schema attestations and cannot make a formal tape eligible
  for scientific claims. Maze, Sudoku, and ARC remain secondary functional
  tests and are not evidence for a biological mechanism.
- `exp13` removes the frozen external-tape dependency: public ARC/maze/Sudoku
  tasks, target-free program/search proposals, predictions, and family
  evaluators are all recomputed in this repository. Query targets remain in an
  identity-bound evaluator capability. This repairs provenance and label
  leakage but does not make the controller a proposal-free solver: every
  selector receives the same deterministic candidate library. Its
  fast/slow/trace mechanisms are narrow HRM/CTM-inspired abstractions, not
  reproductions of either model. Structured-task performance cannot replace
  neural activity evidence. The pinned ARC-AGI-1 snapshot contains one exact
  cross-split duplicate, which the formal config excludes explicitly. The
  unlicensed Maze-Hard/Sudoku-Extreme placeholders have been replaced by
  pinned MIT MazeBench and CC-BY-4.0 Sudoku V2 sources. Preparation retains
  79/79 reachable MazeBench tasks while preserving 31 upstream-unreachable
  mazes in the manifest. The 200 Sudoku source records collapse to 76 exact
  puzzle contents (48 train, 28 non-OOD test) under test-precedence dedup;
  all 124 duplicate exclusions remain auditable. The formal 30-seed runs are
  complete. Maze's hierarchical-over-flat contrast is inconclusive (-0.37
  percentage points, 95% CI [-2.04, 0.74]); only the distinct registered
  selector-level 90%-of-GRU retention endpoint is supported; shared proposal/
  search costs are not counted as an end-to-end efficiency advantage. Sudoku gives every condition
  100% exact accuracy, so its mechanism contrasts are ceiling-limited and
  inconclusive. Neither benchmark isolates an end-to-end neural solver because
  all selectors receive the same deterministic proposal/search library.
- The formal ARC-AGI-1 exp13 run is now complete, but candidate coverage is
  only 5/399 tasks (1.253%) versus the registered 90% gate. Hierarchical local
  accuracy equals flat local at 0.301%; trace local reaches 0.343%, GRU/BPTT
  0.501%, and the candidate oracle 1.253%. All registered advantage claims are
  `inconclusive`. Low coverage prevents either support or a clean mechanism
  rejection, and the result is not a competitive ARC solution.
- The additive Exp15 ARC run repairs the former source-manifest and comparator
  gaps: all 800 JSON files plus the license are hash-verified, and slow/fast and
  flat selectors receive identical candidates with exactly matched charged
  abstract-operation budgets. Both solve 1/399 tasks (0.2506%); their paired
  difference is exactly zero. Candidate coverage remains only 5/399 (1.2531%)
  versus the registered 90% gate, so the scoped conclusion is `inconclusive`.
  The charged budget is not FLOPs, wall time, energy, or an end-to-end
  efficiency measurement, and this task adapter is not a BDH/HRM reproduction.
- `exp14` repairs the main exp06 interface flaws in code: event-specific
  nuisance tables, frozen complete-case masks, integer-count validation,
  whole-block split fingerprints, past-only belief receipts, train-only unit
  selection/PCA, paired observation maps, exact conditional Poisson scoring,
  and animal-primary bootstrap. Its synthetic path is only a systems smoke
  test. The formal loader is fail-closed on a reviewed 20-session compact cache.
  Acquisition and offline good-unit binning are complete for 20 sessions,
  20 animals, 35 probes, and 3,183 units. The registered real-data outer
  comparison has also completed, but its common-minus-shared primary result is
  `inconclusive`; it therefore provides no support for shared neural dynamics.
  Moreover, the
  implemented model remains a
  teacher-forced within-trial one-step conditional model—not a full Poisson/NB
  switching LDS with filtering, smoothing, normal-stability, or causal gate
  perturbations. The preflight models 1,347 train-selected anchor units out of
  3,183 recorded units. Its `full` comparator has session-specific gated
  operators but retains the same six-region shared basis; it is not a full
  unit-space or full latent-LDS comparator.
- `exp16` is an isolated global-autograd/BPTT computational baseline. Its
  micro-TRM-like and single-state conditions can test an alternating-state
  recursion schedule under matched parameters and nominal shared-core calls.
  The TRM-like path follows the official no-gradient outer-cycle prefix and
  detached segment carry, but omits ACT, StableMax, puzzle identifiers, EMA,
  the official block details and model scale. It uses modest train-only Sudoku
  symmetry augmentation rather than the official Extreme-1K/1000-augmentation
  protocol, blank-only loss, and clue-clamped exact evaluation on a 28-task
  non-OOD panel. The pilot publisher cannot promote a formal claim. Neither a
  smoke result nor a future Sudoku advantage can initialize a local-learning
  model or count as evidence for a biological hierarchy, three-factor
  plasticity, or shared neural dynamics.
- The legacy `exp06` neural result remains a one-session Gaussian-LDS pilot.
  Shared dynamics retains only about 25--29% of the full-model likelihood gain,
  far below the 90% criterion; pre-stim future-covariate and true-boundary-reset
  issues must be removed before it can be joined to the exp11 belief trajectory.

See `../docs/integrated_method_audit_zh.md` for the cross-workstream audit and the next falsification tests.
