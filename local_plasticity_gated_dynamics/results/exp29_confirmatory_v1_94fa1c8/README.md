# Exp29 one-shot confirmatory evidence

This directory preserves the first and only registered Exp29 source and
selector attempts run from Git commit
`94fa1c86e210e5bddd4e0ac7332577c07923cfca` (tree
`f95d2ba7180ec93c57f27dc257987ae8df13ae8c`).  The frozen analysis-contract
SHA-256 is
`c680985c2d23c0b230be185b7f28ceb8a41e0377429f6fbd085c24a87e379e69`.

## Confirmatory conclusion

**SUPPORT.** Both preregistered, unconditional seed-level endpoints passed
over 30 independent evaluation seeds and all 44 registered held-out task
generators per seed:

- local three-factor minus fixed-best utility: `0.100260`, 95% bootstrap CI
  `[0.095376, 0.104656]`, one-sided Holm-adjusted `p=1.99998e-05`;
- 80%-of-oracle-gain non-inferiority contrast: `0.009964`, 95% bootstrap CI
  `[0.007113, 0.012734]`, one-sided Holm-adjusted `p=1.99998e-05`.

Mean held-out utilities were oracle `0.863707`, local three-factor
`0.851097`, GRU-BPTT `0.848406`, and fixed-best `0.750836`.  The local-minus-GRU
difference (`+0.002691`) is descriptive only; it was not a preregistered
primary contrast.

## Source audit

The untouched seeds 60--89 produced exactly 13,200 registered source rows
(`30 x 88 x 5`).  The source package replay found 13,200 complete rows, zero
failed, invalid, or infeasible rows, complete Cartesian coverage, valid
functional-budget semantics, and `statistics_unit=seed`.  No failed seed was
replaced and no second attempt was made.

The selector was fitted exactly once on Exp26 seeds 0--29
discovery/validation data.  Confirmatory rows were used only for evaluation.
The one selector attempt produced exactly 5,280 decision rows
(`30 x 44 x 4`).  The local selector used neither autograd nor BPTT; GRU-BPTT
is retained only as a baseline.

## Contents

- `summary/`: conclusion, report, seed endpoints, raw summarized metrics,
  provenance, and publication PNG/PDF;
- `source/`: source-only conclusion, receipt, and report (the raw JSONL is in
  the full archive);
- `materialization/`: the mechanically bound runtime config; only the
  preregistered readiness/package fields differ from the committed template;
- `launch/`: the exact one-shot source launcher;
- `exp29_confirmatory_full_94fa1c8.tar.gz`: complete source attempts, per-seed
  logs, package, materialized config, selector attempt, and summary.

Full archive SHA-256:
`e62e4aef71d21355c71b7d8ffbe4e230de3e95cdd2934166052925d24ad47bc8`.

This result supports task-dependent selection from a frozen actuator
dictionary.  It does not by itself establish hidden-context inference,
online scalar-reward learning, or biological realization of the dictionary.
