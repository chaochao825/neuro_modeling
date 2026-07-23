# Evidence provenance

`experiment_registry.csv` is the authoritative classification of every
implemented experiment ID. `branch_history.csv` records the remote branch tips
audited on 2026-07-23 against `agent/exp26-actuator-matching`.
`historical_git_objects.csv` binds the only tracked result file present at an
ancestor tip but absent from the current tip. Its 29 MB content is materialized
as a compressed historical archive; validation recomputes the Git blob SHA from
the decompressed bytes.

`consolidation_receipt.json` records the main fast-forward and deleted remote
branch names. `scripts/audit_branch_reachability.py` rechecks every recorded tip
against that consolidation commit and rebuilds
`results/history/branch_reachability.csv`.

The disposition is intentionally independent of the numerical conclusion:

- `current_core`: direct evidence for the active Actuator Matching Principle.
- `current_foundation`: a still-required mechanism or real-behavior result.
- `current_open`: an active endpoint that has not yet produced eligible data.
- `historical_only`: superseded, rejected, abandoned, or exploratory work.

A historical `support` result is not silently converted to `oppose`. It keeps
its original statistical conclusion, while its disposition prevents it from
being cited as current evidence. Conversely, failed and negative attempts are
retained rather than deleted.

Run `python scripts/build_evidence_views.py` to validate the registry and
rebuild the disjoint views under `results/current/` and `results/history/`.
