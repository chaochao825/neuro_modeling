# Exp28 independent-source ceiling amendment 1

## Status

This is a transparent post-hoc, reachability-only amendment to
`exp28_exp26_independent_source_v1`. It is an amended sensitivity analysis,
not a restoration of confirmatory independence.

The frozen ceiling-128 run remains immutable and is preserved under
`results/exp28_independent_source_ceiling128_invalid_v3_e650dd1/`. Its
13,200-row Cartesian panel contained 13,199 complete rows and one failed row:
seed 52, generator `d65464a1a0917550a226`, routing mode. The recorded error was
`functional-budget scale is non-finite or exceeds max_scale`.

## Sole change

The independent-test functional-budget ceiling changes from 128 to 256. The
rule is the next power of two strictly above the previous ceiling, so this is a
single deterministic doubling. The full 30-seed x 88-generator x 5-mode panel
must be rerun; selective replacement of the failed cell is prohibited.
No further ceiling amendment is permitted under this protocol.

The meta source, generator manifest, task, carrier, noise, trial order,
actuator modes, performance endpoints, and performance acceptance criteria are
unchanged. No independent-panel refit is permitted.

Performance metrics were inspected by a reviewer after the deterministic
amendment decision draft. They were not used to choose 256. This fact is
recorded explicitly in the config and every amended evidence row. Any
downstream selector analysis that consumes this panel must remain
sensitivity-only and force an inconclusive confirmatory conclusion.

## Audit binding

The amended config binds the preserved ceiling-128 receipt, conclusion, raw
metrics digest, deterministic package archive, run commit/tree, coverage, and
unique failed-cell identity. The runner recomputes those checks before any
amended attempt starts. Amended rows carry a distinct protocol version,
evidence schema, run label, amendment hash, and canonical config hash. The
frozen-v1 contract remains separately replayable and test-covered.
