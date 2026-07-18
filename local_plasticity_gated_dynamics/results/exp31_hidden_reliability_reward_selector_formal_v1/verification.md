# Exp31 verification record

- Formal coverage: 30/30 seeds and 22,680/22,680 planned condition rows
  complete; no failed or invalid condition.
- Python environment: project Python 3.11 environment on server 210.
- Full regression suite: `1124 passed in 727.20s`.
- Exp31-focused suite after the seed-level figure correction: `11 passed`;
  static checks passed.
- Top-level evidence files: SHA-256 verified against
  `BUNDLE_SHA256SUMS` after final regeneration.
- Archived per-seed receipts: 210/210 file hashes verified against
  `receipt_manifest.csv`; the archive contains every config, environment,
  planned-condition registry, raw JSONL metric file, status, manifest, and log.
- Independent review recomputed the primary contrast, bootstrap interval,
  sign-flip tests, Holm adjustment, probe/deploy weighting, feedback-access
  path, block split, and associative/query-shuffled budget equality.

The independent review requested a figure-only correction because the first
panel-b error bars did not use seed-level dispersion. The corrected figure was
regenerated from the unchanged raw metrics. `figure_provenance.json` binds the
old run-time plotting script, amended plotting script, source tables, and final
PNG/PDF hashes. No registered metric or inference changed.
