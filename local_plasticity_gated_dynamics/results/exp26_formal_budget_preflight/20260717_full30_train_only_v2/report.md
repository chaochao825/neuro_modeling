# Exp26 train-only budget preflight: development no-go receipt

This receipt stopped the first formal launch before any validation/test
behavior or rollout was evaluated. The audit used a dedicated training-split
factory and the same actuator fitting entry point as Exp26.

- Coverage: 30 seeds × 88 generators × 4 active modes = 10,560 fits.
- Finite required scales: 10,560/10,560; other fit/degeneracy failures: 0.
- Old registered ceiling: 25.
- Blocked fits: 144 (116 routing, 28 gain).
- Required-scale quantiles: median 1.449, q90 7.764, q95 13.428,
  q99 27.545; maximum 90.092.
- Worst cell: seed 7, held-out generator `6c8b9987b26988e5530f`, routing,
  alpha 1, transition/input rank 1/1, delay 4, noise 0.6.
- Validation/test splits constructed: no.
- Validation/test behavior or rollout used: no.

Conclusion: **NO-GO for the old configuration**. The ceiling of 25 was an
arbitrary numerical admission gate, not the matched functional-current budget
and not a stability gate. The revised, outcome-independent rule is
`next_power_of_two(1.25 × train-only maximum)`, giving 128. This does not alter
the raw actuator direction or unique matching scale; poorly aligned fits remain
in the formal panel and may fail behavior or stability.

The receipt records `git_dirty=true` because the preflight/provenance code was
being hardened during this development audit. It is retained as a failed design
result, not promoted as formal evidence. A clean-commit pass receipt is required
before the revised formal run.

Publication note: the runtime-only absolute `config_path` was removed from the
archived config and its archive hash recomputed; no scientific config field or
raw preflight cell was changed.
