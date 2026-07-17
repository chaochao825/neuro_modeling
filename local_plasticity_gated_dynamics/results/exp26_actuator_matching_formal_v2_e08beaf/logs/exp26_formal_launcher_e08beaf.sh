#!/usr/bin/env bash
set -euo pipefail

WORKTREE=/home/spco/sow_linear/neuro_modeling_exp26_publish_20260717_130100
PROJECT="$WORKTREE/local_plasticity_gated_dynamics"
PYTHON=/home/spco/sow_linear/.venvs/neuro_modeling_311/bin/python
RECEIPT=/home/spco/sow_linear/exp26_clean_preflight_20260717_v3_e08beaf
LABEL=exp26-formal-v2
EXPECTED_COMMIT=e08beaf9f51aacaaa80d42b2755c60d4080364bb
EXPECTED_TREE=c91f11863a654c7588b0d7bde5eb1e1f43e5c774
LOG_ROOT=/home/spco/sow_linear/exp26_formal_v2_e08beaf_logs
RUN_ROOT="$PROJECT/results/runs/exp26_actuator_phase_diagram"

if [[ -e "$LOG_ROOT" ]]; then
  echo "refusing to reuse existing log root: $LOG_ROOT" >&2
  exit 73
fi
mkdir -p "$LOG_ROOT"

finish() {
  local rc=$?
  trap - EXIT
  date -u +%Y-%m-%dT%H:%M:%SZ >"$LOG_ROOT/finished_at_utc.txt"
  printf '%s\n' "$rc" >"$LOG_ROOT/exit_code.txt"
  exit "$rc"
}
trap finish EXIT

exec 9>"$LOG_ROOT/launch.lock"
flock -n 9

cd "$PROJECT"
if git status --porcelain=v1 | grep -q .; then
  echo "formal worktree is dirty" >&2
  exit 2
fi
git rev-parse HEAD | grep -Fxq "$EXPECTED_COMMIT"
git rev-parse 'HEAD^{tree}' | grep -Fxq "$EXPECTED_TREE"
test -d "$RECEIPT"

if [[ -d "$RUN_ROOT" ]] && find "$RUN_ROOT" -mindepth 2 -maxdepth 2 \
  -type d -name "*_$LABEL" -print -quit | grep -q .; then
  echo "formal label already has run attempts: $LABEL" >&2
  exit 73
fi

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

run_seed() {
  local seed=$1
  local log
  log=$(printf '%s/seed_%04d.log' "$LOG_ROOT" "$seed")
  printf 'seed %02d started %s\n' "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if "$PYTHON" experiments/exp26_actuator_phase_diagram.py \
    --config configs/formal/exp26_actuator_phase_diagram.json \
    --results-root results \
    --seeds "$seed" \
    --run-label "$LABEL" \
    --preflight-receipt "$RECEIPT" >"$log" 2>&1; then
    printf 'seed %02d finished %s\n' "$seed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    local rc=$?
    printf 'seed %02d failed rc=%d %s\n' \
      "$seed" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
    return "$rc"
  fi
}
export -f run_seed
export PYTHON RECEIPT LABEL LOG_ROOT

printf '%s\n' {0..29} | xargs -n 1 -P 30 bash -c 'run_seed "$1"' _

if git status --porcelain=v1 | grep -q .; then
  echo "formal worktree changed during the panel" >&2
  exit 2
fi

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("results/runs/exp26_actuator_phase_diagram")
label = "exp26-formal-v2"
observed = {}
for seed in range(30):
    parent = root / f"seed_{seed:04d}"
    attempts = sorted(parent.glob(f"*_{label}"))
    if len(attempts) != 1:
        raise SystemExit(f"seed {seed} has {len(attempts)} labelled attempts")
    status = json.loads((attempts[0] / "status.json").read_text())
    if status["status"] not in {"complete", "complete_with_failures"}:
        raise SystemExit(f"seed {seed} is non-terminal: {status['status']}")
    observed[seed] = {
        "status": status["status"],
        "condition_failures": status["condition_failures"],
        "condition_invalid": status["condition_invalid"],
    }
print(json.dumps(observed, sort_keys=True))
PY
