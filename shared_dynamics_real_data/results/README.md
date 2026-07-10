# Published run snapshot

This directory contains the complete aggregate tables plus 21 immutable run
directories: one smoke run and 20 formal seed shards. Each run retains its
resolved config, environment, data manifest, planned conditions, unit subset,
JSONL/CSV metrics, status, and log. Failed or missing cells would remain in the
same schema; the current formal panel completed all 2,160 cells.

Only machine-specific path strings were sanitized for publication:

- `${REPO_ROOT}` denotes the repository root used on the experiment server.
- `${PYTHON_3_11}` and `${VENV_3_11}` denote the server interpreter/runtime.
- `${CORE_PROJECT_ROOT}` appears only in the migrated core-project aggregate
  tables and denotes its original Windows workspace.

The numeric metrics, seeds, hashes, timestamps, statuses, and scientific
configuration values were not changed. The original unsanitized run folders
remain on the 210 experiment server.
