# Exp29 untouched confirmatory source protocol

## Status and purpose

Exp29 is a new one-shot confirmatory source panel for actuator selection.  It
must be committed and pushed before any performance, reachability, or
feasibility output from seeds 60--89 is read.  The source runner itself fits no
selector and performs no hypothesis test, so every source package has the
standalone conclusion `inconclusive`.

The prior panels have distinct roles:

- Exp26 seeds 0--29 are the only selector meta-training data.
- Exp28 seeds 30--59 at cap 128 are the preserved fail-closed independent run.
- The Exp28 cap-256 rerun is a transparent post-hoc sensitivity analysis.
- Exp29 seeds 60--89 are the untouched confirmatory evaluation panel.

No result from seeds 30--59 may be relabelled as Exp29 evidence.

## Frozen registration

The exact registration is
`configs/formal/exp29_confirmatory_source_panel.json`.  The runner additionally
locks its canonical SHA-256 in code.  It fixes:

- evaluation seeds 60--89 and meta-training seeds 0--29;
- all 88 Exp26 generators and all five modes (`frozen`, `routing`, `gain`,
  `low_rank`, and `rgl`);
- the Exp26 carrier, task, manifest, trial order, noise construction, readout,
  train-only fitting rule, and critical-code hashes;
- the Exp29 runner through a normalized source hash that replaces only the
  embedded registered-config digest with a fixed sentinel, plus raw SHA-256
  hashes of the packager and feasibility-aware adapter;
- `max_scale = 256` with no later ceiling amendment;
- seed as the statistical unit and all registered heldout cells as the
  unconditional inference population.

After the registration commit, ceiling retuning, selective reruns, replacement
of failed seeds, or inspection-driven changes are forbidden.  A second attempt
for the same seed is rejected by the packager rather than selected by success
or recency.

## Feasibility and deployment policy

Each seed saves the complete `88 x 5` Cartesian panel.  An active actuator is
feasible only when its preregistered train-only fit satisfies the functional
L2 budget and its effective dynamics are strictly stable.

An actuator that exceeds cap 256, has a degenerate fit, misses the functional
budget, or is unstable is retained with terminal status `infeasible`.  It is
not deleted, retried, or statistically imputed.  Its deployable utility is the
exact validation/test utility of the `frozen` row with the same seed,
generator, and split.  The raw row records the frozen correction fingerprint
and copied utilities so the packager can replay this equality.

The downstream choice rules are frozen as follows:

- selecting an infeasible actuator receives same-cell frozen utility;
- the selector oracle chooses only among feasible `routing`, `gain`, and
  `low_rank` actuators plus `frozen`; `rgl` remains a non-selector combined
  ceiling/control condition;
- all summaries and tests remain unconditional over registered heldout cells;
- matched-budget support can use only feasible active rows;
- infeasibility is reported per seed and actuator family.

Unexpected execution failures remain `failed`, are preserved, and make the
source package invalid.  They are not converted to scientific infeasibility.

## Leakage and evidence boundaries

The adapter reuses one normalizer and one selector fit obtained exclusively
from Exp26 seeds 0--29 discovery/validation rows.  It builds 30 evaluation
folds, one per seed 60--89, without fitting preprocessing or a selector on the
confirmatory panel.  The panel's discovery rows remain preserved source data
but do not enter the primary heldout evaluation.

Source packaging verifies the registered config, source manifest, critical
code, the three registered Exp29 implementation hashes, clean git commit/tree,
Python 3.11 scientific runtime, every attempt
artifact hash, exact Cartesian coverage, terminal status, frozen fallback
identity, and canonical JSONL hash.  The package loader independently replays
the same invariants, requires `statistics_unit = seed` in every raw row and in
both package metadata files, and fails closed on receipt, raw-row, coverage,
fallback, implementation-hash, or statistics-unit tampering.

## Execution boundary

Only after the registration files and tests are committed on a clean tree may
the 30 seed jobs be launched with the exact label
`exp29-confirmatory-source-v1`.  A new empty results root must be used.  The
packager refuses to overwrite an existing package.

```bash
python experiments/exp29_confirmatory_source_panel.py \
  --config configs/formal/exp29_confirmatory_source_panel.json \
  --seeds 60 \
  --run-label exp29-confirmatory-source-v1 \
  --results-root /new/immutable/results/root

python scripts/package_exp29_confirmatory_source_panel.py \
  --config configs/formal/exp29_confirmatory_source_panel.json \
  --run-label exp29-confirmatory-source-v1 \
  --results-root /new/immutable/results/root \
  --output-dir /new/immutable/results/root/package
```

Parallel launchers may partition seeds 60--89, but each seed may be attempted
exactly once.  Missing or failed seeds are reported as missing/failed evidence;
they are not replaced.
