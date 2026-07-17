# Exp27 low-dimensional actuator selector

## Conclusion

**INCONCLUSIVE** — development/smoke profile is forced inconclusive.

The confirmatory unit is the outer network seed (`n=2`),
and only strict-unseen `(alpha, transition_rank, input_rank)` compositions enter
the primary endpoints. Generator cells are paired prediction targets, not
independent replicates.

## Registered primary endpoints

| Endpoint | Seed mean | 95% seed-bootstrap CI | Positive p | Positive Holm p | Negative Holm p |
|---|---:|---:|---:|---:|---:|
| local_noninferiority_contrast | 0.090625 | [0.075000, 0.106250] | 0.4 | 0.8 | 1 |
| local_minus_fixed_best | 0.453125 | [0.375000, 0.531250] | 0.4 | 0.8 | 1 |

Support requires both the local-vs-fixed gain and the 0.8-oracle
non-inferiority contrast to have a positive lower confidence bound and
Holm-adjusted `p < 0.05`.

## Held-out utility

| Policy | Seed mean | Seed SD |
|---|---:|---:|
| Fixed best | 0.546875 | 0.110485 |
| Local three-factor | 1.000000 | 0.000000 |
| GRU-BPTT | 1.000000 | 0.000000 |
| Oracle ceiling | 1.000000 | 0.000000 |

## Provenance and coverage

- Profile: `smoke`; run label: `exp27-smoke-v1`.
- Git commit/tree: `c44519135e48521ef3742fd63081ee34846936e2` / `f10986b5ad25cf721659bc3f2013b165a728a8b3`.
- Python runtime: `3.11.15`.
- Canonical Exp27 config SHA-256: `7760df73788dd8ee908465e51fc30caf6862d02ff0ff34fda0d05b827dcb77c0`.
- Frozen Exp26 raw SHA-256: `e5dfd3ba9ea26b7b4319de910a0724b40a631f0f591f585f3b0c09033250700c`.
- Frozen Exp26 conclusion SHA-256: `7a0f1dd04fb7d8ac88e05ea9ab4eff614f0b6bfc68873c4c7dd78a1058c96d1b`.
- Collected Exp27 raw-metrics SHA-256: `adead48dae43a5763afa4a6cde21fa7338ad4efc27fe4bd29881fdb3c4e9c0a8`.
- Complete outer seeds: 2; every planned selector row was retained.
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
