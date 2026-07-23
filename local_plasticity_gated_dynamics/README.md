# Actuator Matching Principle

Flexible computation need not retrain a full network for every task. A system
can reuse a fixed high-dimensional carrier and a small dictionary of control
motifs, then select the actuator that best matches the task's demand on input
mapping, internal dynamics, or associative memory.

The active hypothesis is deliberately narrower than the project's original
physical-low-rank proposal:

> Low-dimensional credit and belief signals can control useful low-dimensional
> effective dynamics on a high-rank substrate; task performance depends on
> matching the controller's actuator family to the required computation.

Low matrix rank alone is never evidence. A result must improve held-out
behavior or prediction, and its conclusion is always one of `support`,
`oppose`, or `inconclusive` at the registered seed/session/animal level.

## Evidence status

The repository has two exhaustive and mutually exclusive result views:

- [Current evidence](results/current/README.md) contains only active
  foundations, core results, and open endpoints.
- [Historical evidence](results/history/README.md) contains every superseded,
  rejected, abandoned, or exploratory proposal, including its original
  positive, negative, failed, and inconclusive rows.

The [experiment registry](provenance/experiment_registry.csv) classifies every
entry point from Exp00 through Exp32. The [branch audit](docs/branch_consolidation.md)
shows that all other remote branches were already ancestors of the audited
base, so no implementation commit was missing. Hash-bound snapshots preserve
their prior README/report/summary surfaces.

The current evidence chain is:

1. Exp08: low-dimensional credit can coexist with high-rank physical E/I
   updates after mask, Dale, and normalization operations.
2. Exp09/10/21: leakage-safe hidden belief can modulate bounded effective
   dynamics on a frozen high-rank Dale E/I receiver.
3. Exp11: real IBL behavior gives mixed but useful evidence for past-only
   hidden-block inference; it is not neural validation.
4. Exp24/26: synthetic task demand reverses which actuator family is useful.
5. Exp29: an independently evaluated descriptor selector improves over one
   globally fixed family.
6. Exp31: executed scalar reward selects between two fixed motifs in reset
   blocks.
7. Exp32: persistent sparse-feedback control supports at the slow-switch main
   endpoint, while the stronger registered timescale phase claim remains
   inconclusive.
8. Exp25: the real compositional neural endpoint remains active but correctly
   fails closed because an eligible canonical neural bundle is unavailable.

See the [formal principle ledger](docs/actuator_matching_principle.md) and
[current critical audit](docs/current_evidence_critical_audit.md) for effect
sizes, boundaries, and scale priorities.

## What is not currently claimed

- Low-dimensional feedback does not imply a low-rank physical recurrent
  matrix.
- The rate-matched independent phase-gate proposal is rejected in this model.
- The tested Exp23 local gain-axis rule/budget does not improve held-out
  behavior and is historical only.
- The ARC, maze, Sudoku, tiny-HRM, and recursive baseline experiments are
  historical capability probes, not evidence for the active neural theory.
- Exp31/32 do not yet contain a participating high-rank E/I carrier.
- No shared gated model has yet beaten common dynamics on the required
  multi-animal/session neural dataset.
- The project does not claim general SOTA, a biological MD/ACC identity, or a
  replacement for arbitrary history or KV cache.

## Reproducibility contract

- Python 3.11 only.
- NumPy, SciPy, pandas, scikit-learn, PyTorch, matplotlib, and statsmodels;
  ONE-api/ibllib are optional and isolated to IBL acquisition.
- Every stochastic entry point receives and records an explicit seed.
- Local-learning candidates do not use autograd or BPTT. BPTT/GRU appear only
  as isolated baselines.
- Trials or blocks, never individual time points, define train/test folds.
- Scaling, PCA, subspaces, nuisance regression, and latent-dimension selection
  are fit inside training folds.
- Seed, session, or animal is the independent statistical unit; neurons and
  time bins are never treated as independent replicates.
- Failed, invalid, infeasible, and missing conditions remain first-class
  output rows; joint claims use AND gates rather than success-selecting OR
  rules.

## Repository layout

- `src/`: tasks, models, plasticity rules, analysis, and data adapters.
- `experiments/`: all current and historical executable entry points. See
  [their status index](experiments/README.md) before running one.
- `configs/`: smoke and formal frozen configurations.
- `results/current/`: active evidence indexes and current-only claim rows.
- `results/history/`: historical experiment index, branch snapshots, failed
  rows, and immutable-object provenance.
- `provenance/`: authoritative experiment/branch/object registries.
- `scripts/build_evidence_views.py`: validates provenance and deterministically
  regenerates the two evidence views.

## Reproduce the consolidated state

On Windows, bootstrap the project-local Python 3.11 environment and run all
tests:

```powershell
./scripts/bootstrap_windows.ps1
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\build_evidence_views.py
git diff --exit-code -- results\current results\history\README.md `
  results\history\experiments.csv results\history\branches.csv `
  results\history\claims.csv results\history\git_objects.csv `
  results\history\snapshot_manifest.csv
```

Formal experiment commands and immutable package-specific receipts are kept in
the corresponding current evidence report or protocol. Historical commands
remain in the archived
[pre-consolidation README](results/history/project_README_pre_consolidation.md),
so reorganizing the active narrative does not erase earlier workflows.
