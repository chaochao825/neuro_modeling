#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/spco/sow_linear/ibl_neural_exp14_staging
PY=/home/spco/sow_linear/.venvs/neuro_modeling_311/bin/python

if [[ "$#" -ne 1 ]]
then
  printf '%s\n' 'usage: launch_postprocess.sh --validate-inputs-only|--execute' >&2
  exit 64
fi
case "$1" in
  --validate-inputs-only|--execute) ;;
  *)
    printf '%s\n' 'unsupported postprocess mode' >&2
    exit 64
    ;;
esac

cd "$ROOT"
umask 077
exec 9>"$ROOT/status/postprocess.lock"
if ! flock -n 9
then
  printf '%s\n' 'another postprocess instance holds the lock' >&2
  exit 75
fi

exec env -i \
  HOME=/nonexistent \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PATH=/usr/bin:/bin \
  PYTHONHASHSEED=0 \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "$PY" "$ROOT/scripts/postprocess_compact.py" "$1"
