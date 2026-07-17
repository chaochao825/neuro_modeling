# Exp27 low-dimensional actuator selector

## Conclusion

**SUPPORT** — both strict-unseen seed-level confirmatory endpoints passed.

The confirmatory unit is the outer network seed (`n=30`),
and only strict-unseen `(alpha, transition_rank, input_rank)` compositions enter
the primary endpoints. Generator cells are paired prediction targets, not
independent replicates.

## Registered primary endpoints

| Endpoint | Seed mean | 95% seed-bootstrap CI | Positive p | Positive Holm p | Negative Holm p |
|---|---:|---:|---:|---:|---:|
| local_noninferiority_contrast | 0.014511 | [0.012111, 0.016844] | 9.9999e-06 | 1.99998e-05 | 1 |
| local_minus_fixed_best | 0.108083 | [0.101914, 0.114382] | 9.9999e-06 | 1.99998e-05 | 1 |

Support requires both the local-vs-fixed gain and the 0.8-oracle
non-inferiority contrast to have a positive lower confidence bound and
Holm-adjusted `p < 0.05`.

## Held-out utility

| Policy | Seed mean | Seed SD |
|---|---:|---:|
| Fixed best | 0.749919 | 0.018468 |
| Local three-factor | 0.858002 | 0.014225 |
| GRU-BPTT | 0.858196 | 0.013189 |
| Oracle ceiling | 0.866885 | 0.010915 |

## Provenance and coverage

- Profile: `formal`; run label: `exp27-formal-v1`.
- Git commit/tree: `c44519135e48521ef3742fd63081ee34846936e2` / `f10986b5ad25cf721659bc3f2013b165a728a8b3`.
- Python runtime: `3.11.15`.
- Canonical Exp27 config SHA-256: `c5572273ac0800896235aaec2cc76f11d7b32ea360985abfb6770fb9dfac6791`.
- Frozen Exp26 raw SHA-256: `b3ef5e22c241f832b1fd50254f87e3890ec45057bfeda3a784cbd218623a1193`.
- Frozen Exp26 conclusion SHA-256: `2038127ac875f9faae94b305343415b8fb3a794f9ea032f017401e432fa9d40f`.
- Collected Exp27 raw-metrics SHA-256: `ca6ce10620c9988fc10c5b5e687592f79e958938cac9ce0d37b743c36fb57bee`.
- Complete outer seeds: 30; every planned selector row was retained.
- Local main method: no autograd and no BPTT. GRU-BPTT is an isolated baseline.

## Interpretation boundary

Exp27 selects among frozen, task-matched **actuator-family policies**; it does
not learn recurrent weights or prove a single global biophysical motif
dictionary. Its prospective demand cues are generator-provided task
descriptors, not an online hidden-state belief inferred from observations.
The local third factor is the full three-candidate validation-utility vector,
not a scalar reward from only the selected action, and its update-cost proxy is
not budget-matched to the differently parameterized GRU. Accordingly, a
support result establishes supervised unseen-composition family selection,
not hidden-context inference, de-novo motif formation, or an independent
temporal-credit mechanism.

The 30 LOSO endpoints reuse highly overlapping meta-training seed sets, so
their seed bootstrap and sign-flip inference is a cross-fitted, conditional
analysis rather than 30 independently trained selectors. A fixed meta-train
versus independent test-seed split remains the appropriate sensitivity check.
Also, "strict unseen" is registered only for the `(alpha, transition rank,
input rank)` triple, not every delay/noise/rotation coordinate.
