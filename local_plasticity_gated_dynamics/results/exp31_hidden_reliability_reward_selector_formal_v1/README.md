# Exp31 formal evidence bundle

This bundle contains the complete 30-seed formal panel for
`exp31_hidden_reward_v1`. All 22,680 registered seed/block/condition rows are
present and complete. The selector's primary full-block utility includes its
64-trial forced-exploration prefix.

The run is bound to clean source commit
`9b068056a0c73806665d1c4792f59f4c6f59bbc9` and tree
`cfca8323fa41de2b1bb259d4d6b4cdcfc5d4db27`. The experiment implementation was
frozen in parent commit `4dde5c625e06f16e24db823dba4bc7537020528c` before any
registered smoke or formal result was observed.

`run_receipts.tar.gz` contains every seed's config, environment, planned
condition registry, raw JSONL metrics, status, manifest, and log. The archive
has 30 seed directories and 210 receipt files. `receipt_manifest.csv` records
the SHA-256 of every file inside that archive. `BUNDLE_SHA256SUMS` binds the
top-level bundle files.

The registered formal conclusion is **support** for the narrow claim that an
executed-reward-only local controller can outperform one train-selected fixed
actuator across hidden cue-reliability blocks under dense-memory capacity and
distractor pressure. It does not support high-rank E/I carrier computation,
neural-data validity, arbitrary local plasticity rules, or a state-of-the-art
task-performance claim.

The controller receives labels on 64 of 128 trials, resets at every block, and
chooses between only two fixed motifs. These are material limitations, not
implementation details. The figure was regenerated after independent review
to compute panel-b uncertainty from seed-level aggregates; the registered raw
metrics and all numerical summaries were unchanged. `figure_provenance.json`
binds this presentation-only correction to both plotting-script hashes and the
unchanged raw inputs.
