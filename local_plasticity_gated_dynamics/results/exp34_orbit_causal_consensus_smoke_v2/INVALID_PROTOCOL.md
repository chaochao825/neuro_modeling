# Invalidated development package

This package is retained for audit but cannot authorize or support Exp34.

The `v1` feature pipeline used `object_not_present_issue` to filter both clean
support and clutter query frames. The official ORBIT protocol permits this
extra annotation for query sampling but forbids extra clean-video annotations
during personalization. No test user was evaluated, and the formal launch was
stopped before feature extraction completed.

The active successor is protocol
`exp34_orbit_causal_consensus_v2_support_annotation_safe`, which uses every
clean support frame and reserves object-presence filtering for clutter query
videos. The original metrics and authorization receipt remain unchanged here
so the invalidation is auditable.
