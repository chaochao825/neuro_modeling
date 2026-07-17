#!/usr/bin/env bash
set -uo pipefail

PROJECT=/home/spco/sow_linear/neuro_modeling_exp26_publish_20260717_130100/local_plasticity_gated_dynamics
PYTHON=/home/spco/sow_linear/.venvs/neuro_modeling_311/bin/python
RESULTS=/home/spco/sow_linear/exp27_formal_v1_c445191
CONFIG=configs/formal/exp27_low_dimensional_actuator_selector.json
RUN_LABEL=exp27-formal-v1

export PROJECT PYTHON RESULTS CONFIG RUN_LABEL
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$RESULTS/logs"

run_seed() {
  local seed="$1"
  local padded
  local code
  printf -v padded '%04d' "$seed"
  "$PYTHON" "$PROJECT/experiments/exp27_low_dimensional_actuator_selector.py" \
    --config "$PROJECT/$CONFIG" \
    --results-root "$RESULTS" \
    --run-label "$RUN_LABEL" \
    --seeds "$seed" \
    >"$RESULTS/logs/seed_${padded}.log" 2>&1
  code=$?
  printf '%s\n' "$code" >"$RESULTS/logs/seed_${padded}.exit_code.txt"
  return "$code"
}

export -f run_seed
seq 0 29 | xargs -P10 -n1 bash -c 'run_seed "$1"' _
panel_code=$?
printf '%s\n' "$panel_code" >"$RESULTS/logs/panel.exit_code.txt"
exit "$panel_code"
