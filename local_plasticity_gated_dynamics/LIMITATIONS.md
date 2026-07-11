# Known limitations of the current formal results

The committed summary is an immutable snapshot of the completed runs, not a claim that every mechanism is identified.

This monorepo carries the compact evidence snapshot (`raw_metrics.csv.gz`,
`runs.csv`, `summary.csv`, the generated report, and PNG/PDF figures). The
timestamped `results/runs/` directories and per-attempt logs remain in the
source experiment workspace and are intentionally not duplicated here; the
committed configs and scripts regenerate that layout.

- Phase 1 is the clean constructive test: the aligned rank-4 update is rank matched and non-inferior to full feedback on the synthetic latent task.
- In Phase 2, derivative modulation, sparse masks, Dale projection, and fan-in normalization do not preserve the algebraic rank bound of the raw outer-product rule. The final E/I update is therefore not low rank even when activity remains low dimensional.
- Local, full-feedback, and shuffled-feedback task accuracies are indistinguishable in the completed conditions. These runs do not identify feedback alignment as the cause of behavior.
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
- The legacy `exp06` neural result remains a one-session Gaussian-LDS pilot.
  Shared dynamics retains only about 25--29% of the full-model likelihood gain,
  far below the 90% criterion; pre-stim future-covariate and true-boundary-reset
  issues must be removed before it can be joined to the exp11 belief trajectory.

See `../docs/integrated_method_audit_zh.md` for the cross-workstream audit and the next falsification tests.
