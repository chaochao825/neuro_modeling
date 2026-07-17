#!/usr/bin/env bash
set -uo pipefail

worktree=/home/spco/sow_linear/neuro_modeling_exp26_publish_20260717_130100
project="$worktree/local_plasticity_gated_dynamics"
python=/home/spco/sow_linear/.venvs/neuro_modeling_311/bin/python
expected_commit=94fa1c86e210e5bddd4e0ac7332577c07923cfca
results_root=/home/spco/sow_linear/exp29_confirmatory_source_v1_94fa1c8
logs_root=/home/spco/sow_linear/exp29_confirmatory_source_v1_94fa1c8_logs
run_label=exp29-confirmatory-source-v1

actual_commit="$(git -C "$worktree" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  printf 'commit mismatch: expected=%s actual=%s\n' "$expected_commit" "$actual_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$worktree" status --porcelain)" ]]; then
  printf 'refusing to run from a dirty worktree\n' >&2
  exit 2
fi
if [[ -e "$results_root" || -e "$logs_root" ]]; then
  printf 'refusing to reuse Exp29 result or log root\n' >&2
  exit 2
fi

mkdir -p "$results_root" "$logs_root"
printf 'EXP29_SOURCE_LAUNCH commit=%s seeds=60-89 results=%s\n' \
  "$actual_commit" "$results_root"

declare -a pids=()
declare -a seeds=()
for seed in $(seq 60 89); do
  (
    set +e
    /usr/bin/env \
      -C "$project" \
      PYTHONHASHSEED=0 \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      "$python" experiments/exp29_confirmatory_source_panel.py \
      --config configs/formal/exp29_confirmatory_source_panel.json \
      --seeds "$seed" \
      --run-label "$run_label" \
      --results-root "$results_root"
    code=$?
    printf '\nEXP29_SEED_EXIT seed=%s code=%s\n' "$seed" "$code"
    exit "$code"
  ) >"$logs_root/seed_${seed}.log" 2>&1 &
  pids+=("$!")
  seeds+=("$seed")
done

overall=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf 'EXP29_SEED_DONE seed=%s code=0\n' "${seeds[$index]}"
  else
    code=$?
    printf 'EXP29_SEED_DONE seed=%s code=%s\n' "${seeds[$index]}" "$code"
    overall=1
  fi
done

printf 'EXP29_SOURCE_COMPLETE overall=%s\n' "$overall"
exit "$overall"
