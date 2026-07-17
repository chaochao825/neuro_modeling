# Exp28 ceiling-amendment source package

This is the source-only package for the post-hoc ceiling-256 sensitivity on
seeds 30--59. It contains 13,200 complete rows, zero failed/invalid rows,
valid functional budgets, and stable effective dynamics. Its inference status
is `post_hoc_amended_sensitivity_non_confirmatory`; it cannot restore the
confirmatory status rejected by the original ceiling-128 run.

The full canonical package is stored as
`exp28_source_amend1_v1_28b6c76_valid_package.tar.gz` (SHA-256
`5f3b19b557e542a3bfc378e676aae1d81302eca025644c88bd5a2c6076ca6e95`).
It expands to a `package/` directory. That materialized directory is ignored
because its canonical raw JSONL is 109,738,821 bytes; the tracked archive is
6,590,459 bytes and contains the same receipt, conclusion, report, and raw
metrics.

Before replaying the selector from a fresh clone, extract the archive in this
directory so that `package/source_panel_receipt.json` exists. Extraction must
refuse to overwrite an existing package unless its hashes have first been
verified.

Registered package identities:

- Receipt file: `363797923540f309d1eafd13511bb7eaa18678f35b87d074afea25331a304b20`
- Conclusion file: `51144c4c5dedc58ad6862bcd799db9942352fdea8a3dce2474011195c9a8cb6d`
- Canonical raw JSONL: `736e060609072ad6bec2773d9bcd369b6e129c1b7f0a2911f0929b508ecc200a`
- Receipt payload: `fdd2d01282e3ac5f2c27c7392cb87e68eb7f055c6cd2fea7f74cb3da9e2a3827`
- Registered source config: `0ec0ff7c1079df793063b0bb37f52b9c823b8fff96a8572e021f51438788c03b`
- Ceiling amendment: `d91f01144078f1b6af2198f50cbc59381f393165e2bbf1d9795c9365c304089f`
