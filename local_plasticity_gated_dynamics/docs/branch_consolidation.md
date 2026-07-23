# Branch and evidence consolidation audit

## Outcome

The 2026-07-23 audit found no unmerged implementation commits. All nine other
remote branches are ancestors of `agent/exp26-actuator-matching`; therefore an
octopus merge or a sequence of no-op merge commits would add no code and would
misrepresent the repository history.

The consolidation instead fixes the evidence layer:

1. one registry classifies every `exp00`--`exp32` entry point;
2. generated current and historical views are exhaustive and disjoint;
3. every ancestor branch has an exact README/report/summary snapshot;
4. snapshot hashes make later accidental rewriting detectable;
5. the sole ancestor-tip result file deleted later is materialized as a
   compressed archive and verified against its original Git blob SHA;
6. tests prevent abandoned or refuted methods from re-entering the current
   presentation surface.

## Current theory boundary

The active statement is deliberately narrower than the original proposal:

> A fixed high-rank carrier can expose several low-dimensional control motifs.
> A belief or reward controller can select the motif whose functional tangent
> best matches the task demand.

Current evidence comprises the credit/rank-stage audit (Exp08), hidden belief
inference (Exp09), bounded E/I bridges (Exp10 and Exp21), real behavioral belief
inference (Exp11), actuator specialization and geometry (Exp24 and Exp26), and
independent selector tests (Exp29, Exp31, and the bounded main claim of Exp32).
The real compositional neural endpoint (Exp25) remains active but fail-closed.

The current evidence does **not** establish low physical matrix rank, a general
phase gate, the failed Exp23 gain-axis rule, a general-purpose HRM/ARC/Sudoku
solver, a participating E/I carrier in Exp31/32, or multi-animal neural
superiority of the shared model.

## Historical-only boundary

`results/history/experiments.csv` contains every superseded, rejected,
abandoned, or exploratory experiment. This includes positive results whose
interpretation was superseded (Exp00/01/07/27), explicit negative results
(Exp04/23), incomplete real-data routes (Exp05/06/14/20), reasoning prototypes
(Exp12--18), and development panels replaced by independent formal tests
(Exp28/30).

Historical classification never changes an original numeric result. It changes
only whether that result may support the current theory.

## Audited branch topology

The exact branch tips, dates, scopes, and distances to the audited base are in
`provenance/branch_history.csv`. Their preserved presentation artifacts are in
`results/history/branch_snapshots/`; the generated
`results/history/snapshot_manifest.csv` records SHA-256 and byte count. The one
later-deleted artifact, `results/raw_metrics.csv` from
`agent/real-lowdim-validation`, is preserved as
`results/history/branch_snapshots/real-lowdim-validation/raw_metrics.csv.tar.gz`
and indexed by commit/blob SHA in `results/history/git_objects.csv`.

After the consolidation commit was fast-forwarded to `main`, all ten redundant
remote `agent/*` references were deleted. `provenance/consolidation_receipt.json`
records that operation. Branch deletion removed names only: the audit in
`results/history/branch_reachability.csv` shows zero commits outside the
consolidated main history for every recorded tip.
