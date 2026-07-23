# Invalid Exp34 v2 formal coverage attempt

This package records the first formal Exp34 execution and is not eligible for
claim inference. All five seeds completed with failures: 15 of 17 test users
were evaluated, while every task for P204 and P901 failed because the loader
raised on one clutter video with fewer than 50 valid frames. The official
ORBIT protocol requires excluding each such video rather than excluding the
user.

Each seed retained 91,000 complete condition rows and 1,000 failed condition
rows. The failures were exactly 500 rows for P204 and 500 for P901: ten
conditions across 50 tasks. The triggering videos had 40 and 46 valid frames;
P204 also had a second 49-frame video that the fail-fast loader did not reach.

The pre-fix aggregator incorrectly reported `support` from the 15 complete
users. That output is retained as `original_ineligible_summary.json` but is
explicitly invalid: the registered endpoint required all 17 users and no
failed conditions. Protocol v3 adds both the official per-video exclusion and
a hard coverage audit; it was rerun transparently after this test-data failure
was observed, so the corrected run is not described as an untouched first
look.

The full immutable run artifacts remain on the 210 server at
`/home/spco/sow_linear/exp34_runs/formal_v1_annotation_safe_majority` (478 MiB).
`run_artifact_hashes.csv` binds every seed's raw metrics, failure log, and
status without placing files above GitHub's size limit in the repository.

