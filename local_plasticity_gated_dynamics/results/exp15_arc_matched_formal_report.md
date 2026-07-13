# Exp15 ARC verified-source matched-compute audit

This additive run used clean commit `cbec277503d02844729d8fea5648a9e34e2ce44b` and verified all 800 ARC-AGI-1 JSON files plus the Apache-2.0 license against the reviewed per-file source manifest `76e2360f6673093730676345fd3db8bf289be3f58179c002980a4e91ae0d9cda`. The separate acquisition and validation receipts were also verified. Query targets were used only by the held-out scorer and candidate-coverage diagnostic.

## Absolute task performance

- Slow/fast selector: 0.2506% exact (95% source-group CI 0.0000%–0.7519%).
- Flat matched selector: 0.2506% exact (95% source-group CI 0.0000%–0.7519%).

## Registered paired comparison

The slow/fast-minus-flat exact-accuracy difference is 0.0000 percentage points (95% paired source-group bootstrap CI 0.0000 to 0.0000; Holm p=1). Candidate fingerprints and charged compute are matched. Compute is an audited abstract operation proxy, not FLOPs, wall-clock time, or energy. Candidate coverage is only 1.2531% versus the preregistered 90.0% gate, so `core_claim_eligible=false` and the conclusion is **inconclusive**.

## Interpretation

The source-provenance defect is repaired, but source eligibility alone does not create evidence for hierarchical advantage. The finite proposal library remains the limiting factor; low matrix/state dimension and task-specific architecture are not counted as support without held-out behavioral gain.

Published raw SHA-256: `2353bf280aa5ded0fd3c1d202fe5eedf796c0de07091afffa4db0178548c9fcf`. Run-manifest SHA-256: `fc4c6a177d2c08a81dcac6e0075f6bff8fa60f22937f63b29c715c83d8d6a066`.
