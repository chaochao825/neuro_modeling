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
  IBL result contains one session/animal and is descriptive only; strict P6
  support requires at least 5 animals and 20 sessions on a leakage-free shared
  hidden-context panel.
- `exp09` repairs the hidden-context leakage at the gate-only level and adds
  cue-only HMM/MD inference plus frozen post-fit interventions. The MD
  candidate combines past-only two-slice soft counts with 80% Hebbian multi-lag
  moment shrinkage when the cue process is identifiable; it is not a pure
  recurrent soft-count mechanism. It does not yet
  couple that belief gate to the local-plasticity N=256/N=512 recurrent PFC/E/I
  models, so even positive P2 gate results cannot by themselves establish the
  complete credit-assignment-to-recurrent-dynamics mechanism.
- `exp10` is the repository's first belief-to-Dale-E/I bridge, but recurrent weights remain
  frozen and the control axis is rank one by construction. Its N=32 pilot
  shows a small learned-HMM *pipeline* gain over a separately refit no-gate
  readout; it does not support the current fixed-readout MD-like intervention
  panel, and it cannot establish three-factor recurrent learning. The
  registered N=256/30-seed run remains outstanding.
- `exp11` now implements a leakage-safe trials-only IBL benchmark plus a cohort
  freezer and immutable-manifest contract; no formal multi-animal artifact is
  yet committed. A one-session developmental check was descriptively worse on
  held-out choice log loss than the strong history readout, but the inferential
  conclusion is `inconclusive_insufficient_cohort`, not `oppose`.
  Multi-session/animal inference must be completed before either a support or
  oppose conclusion. The distinct MFD_09 `exp06` neural pilot is not exp11
  behavior evidence.
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
