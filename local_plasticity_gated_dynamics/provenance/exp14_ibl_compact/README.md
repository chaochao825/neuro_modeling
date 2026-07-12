# exp14 IBL compact audit snapshot

This directory preserves the byte-exact code and review receipts used once on
server profile `210` to produce the reviewed exp14 compact neural cache.  It is
an audit snapshot, not a supported data-generation command.

## Frozen files

| File | SHA-256 | Purpose |
|---|---|---|
| `postprocess_compact.py` | `951a516ae7eb2aa024f37e1890d6afd34dd19378905b90744dfadb32e7a76a17` | Reviewed one-shot producer |
| `compact_schema.json` | `f29c5506be93393499b90535c401bec1c82e0754737733897716cd4e2fade39d` | Producer input/output contract |
| `launch_postprocess.sh` | `fe8a30ab705d7e0c8474e574fcae36ce68f510d2d53de81003edd3a3f05f2837` | Reviewed fail-closed launcher |
| `bwm_loading.py` | `c2e570c62cd0e047303c97d7999711b659a1c37eaa56dd2740ffff2c81f85321` | Pinned BWM mask/loader provenance |
| `POSTPROCESS_READY_FOR_REVIEW.json` | `e338ea6058782b486198bb12dc649f8416e864507221163862f0af1f8702ec2c` | Pre-execution candidate receipt |
| `POSTPROCESS_REVIEW_APPROVED.json` | `9e8b7b0ec1c28029c82da19211f69ad8ed0c912c55f3b6c4eb519e8d15e28ff1` | Independent approval binding |
| `compact_contract_audit_v2_20260712T0320Z.json` | `5690e0611fa4931ba2f2f11735df5b5f5886bd2cd8638714bcb88a452827a7eb` | Read-only output contract audit |
| `run_receipts.json` | `55df22b11d0544ae328d6a2fa179d76a9a2c5e7db42e668536c5c1d6937dd22c` | Acquisition, execution, artifact, and retained-failure index |
| `BWM_LICENSE.txt` | `150ac8fd0875e151e7d44ef9bb16deb632e0f56903d0cb8ffb8051f3978baf04` | Byte-exact upstream MIT license |

The pinned BWM file came from commit
`118fc36cb3602934466ad2c6087c2b3b441f9f1f` of the International Brain
Laboratory brain-wide-map repository.  It is redistributed under the MIT
license in `BWM_LICENSE.txt`.  The frozen atlas mapping is not duplicated in
Git: it remains hash-bound inside the ignored compact cache under
`compact_v1/provenance/`.

`run_receipts.json` indexes the acquisition, approval, execution, compact
artifact, retained auxiliary failure, and non-authoritative prefetch hashes.
The first validation attempt is deliberately preserved: it exited `126`
because the launcher lacked its reviewed executable bit.  After the mode was
corrected, the pre-review and post-approval validation receipts both reported
`inputs_valid`; the approved producer then exited `0` exactly once.

## Why these files must not be run from the repository

The producer and launcher deliberately bind all of the following:

- staging root `/home/spco/sow_linear/ibl_neural_exp14_staging`;
- exact source, inventory, approval, launcher, producer, schema, BWM, and atlas
  hashes;
- original acquisition status receipts and download log;
- a one-shot output rule that rejects an existing `compact_v1` directory.

Changing paths or making the producer generally configurable changes its hash
and breaks the reviewed approval chain.  Running the snapshot from this
directory is therefore unsupported.  Do not edit these frozen files.  Any new
ONE acquisition is a **non-identical generation workflow**: it must use a new
cohort/version, preserve every failed condition, receive a new independent
review, and publish new hashes.  It must not claim bitwise reproduction of the
2026-07-12 compact artifact.

## Offline validation of a local copy

Place the complete ignored cache at the path registered by the formal config,
including `sessions/`, `provenance/`, `compact_manifest.csv`, and
`compact_bundle.json`.  Validate without network access or filesystem writes:

```bash
python scripts/validate_exp14_ibl_compact.py \
  --config configs/formal/exp14_ibl_multisession_neural.json
```

The validator delegates artifact-level checks to the strict exp14 consumer,
prints a JSON report to standard output, and never downloads, installs, copies,
repairs, or rewrites data.
