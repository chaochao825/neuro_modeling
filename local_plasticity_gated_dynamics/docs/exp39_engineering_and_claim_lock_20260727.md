# Exp39 engineering and claim lock

Status: **additive audit of a frozen formal result**. This document and the
associated audit utilities do not change the Exp39 algorithm, configuration,
tapes, update order, numerical behavior, checkpoint schema, registered tests,
or formal result artifacts.

## Why this lock exists

Exp39 is a valid positive result at one deliberately narrow level: across the
four registered held-out cells, the factorized method improves mean predictive
NLL over selected fixed and seen-mode IMM in all 30 seeds. Later diagnostics
show that this aggregate result cannot support stronger wording about every
cell, clean recovery of three uncertainty parameters, fast release, or real
data. The engineering lock protects the original result while making those
claim boundaries mechanically auditable.

## Immutable scientific surface

The following surfaces remain authoritative and unmodified:

- frozen Exp39 implementation and configuration;
- fit/test tape split, method pairing, trial order, and random streams;
- formal per-seed shards and root CSV/JSON outputs;
- registered summary, tests, verdict, and publication amendment;
- all failed, invalid, and opposing rows.

No diagnostic produced after the formal run may overwrite a frozen artifact,
upgrade a registered claim, or be represented as preregistered evidence.

## Additive semantic lock

`provenance/exp39_semantic_lock_20260727.json` binds the audit to the formal
artifact and publication manifests, the implementation/execution receipts,
the transferred formal archive, and the commit that introduced the result.
The additive checker `scripts/audit_exp39_semantic_lock.py` fails closed unless
all of the following remain true:

1. both frozen manifests parse without unsafe, duplicate, or missing paths and
   every listed SHA-256 digest matches;
2. implementation and execution receipt hashes match their recorded files;
3. each per-seed shard exactly reproduces the corresponding root table rows;
4. selected candidates are the minimum fit-tape selection NLL within their
   registered seed/family comparison;
5. all fit and test tapes regenerate from the frozen configuration and seeds
   with matching portable semantic fingerprints; exact tape digests are
   reported separately because cross-platform normal transforms can differ in
   floating-point tail bits;
6. the registered summary and its three derived CSV tables replay, including
   numerical values and the original joint verdict;
7. frozen numerical fingerprints for the cell, loading, timing, and oracle
   headroom panels remain unchanged.

This is a semantic regression guard, not a new scientific result. A replay
environment recorded now describes the replay only; it must never be
retroactively labelled as the environment of the original formal execution.
The checked 210-server replay is retained as
`provenance/exp39_replay_receipt_210_20260727.json`; its environment scope is
explicitly `current_replay_only_not_original_formal_execution`.

## Additive claim-boundary audit

`scripts/audit_exp39_claim_boundaries.py` materializes
`results/exp39_posthoc_claim_boundary_20260727/` from the frozen tables. The
output is explicitly `claim-ineligible` and uses seed, not row or time bin, as
the statistical unit.

### What remains supported

- Registered average held-out-composition utility: selected fixed minus
  factorized NLL is +0.290580 nats and seen-mode IMM minus factorized is
  +0.048008 nats, each positive in 30/30 seeds after the frozen joint gate.
- The wording applies to the average over cells `011`, `101`, `110`, and `111`.
- Functional clamp selectivity passed the frozen Holm family. It does not, by
  itself, prove recovery of the generating parameters.

### What is opposed or unresolved

| Question | Frozen/post-hoc diagnostic | Boundary |
|---|---:|---|
| Uniform cell-wise utility vs seen IMM | `011` +0.042613; `101` +0.020958; `110` -0.004333; `111` +0.132794 nats | **Oppose** |
| Clean three-factor decomposition | only one of three estimate rows has a clearly diagonal loading; Q and R cross-load | **Oppose** |
| Fast release vs seen IMM | transition-only early gain -0.087523 nats | **Oppose** |
| Late-regime adaptation vs seen IMM | transition-only late gain +0.083153 nats | Post-hoc **support** only |
| Real behavior or neural utility | no real data in Exp39 | **Inconclusive / not tested** |

The seed-level loading matrix (rows: estimated coordinates; columns: true
manipulated factors) is:

| Coordinate | true h | true Q | true R |
|---|---:|---:|---:|
| `z_h` | 0.088327 | 0.028228 | 0.091796 |
| `z_Q` | 0.067182 | 0.539656 | 1.491769 |
| `z_R` | 0.075404 | 0.549543 | 1.518126 |

Consequently, manuscript prose must call these quantities **factor-indexed
adaptive coordinates** unless a new matched-identifiability experiment
supports a stronger interpretation.

## Timing terminology lock

The formal field name `early_nll` is retained for artifact compatibility. In
prose it must be called **early-in-block NLL**, because the registered summary
includes the initial block of each sequence, where no preceding transition
exists. The transition-only recomputation excludes those initialization
blocks, is post-hoc, and does not rescue the fast-release claim.

## Permitted maintenance

Maintenance may add independent validators, schema checks, replay receipts,
documentation, and tests that fail before scientific code is invoked. It may
not silently change defaults, floating-point order, RNG consumption, private
selection rules, artifact paths, or frozen result contents. Any future
algorithmic change requires a new experiment identifier, configuration,
result directory, and evidence label.
