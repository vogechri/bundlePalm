#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_stage_c_scaling_pilot.sh"
ANALYZER="$SCRIPT_DIR/analyze_stage_c_scaling_repeats.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/stage_c_scaling_repeats_k4_16_i30"}
REPEATS=${REPEATS:-3}
ITERATIONS=${ITERATIONS:-30}
OVERWRITE=${OVERWRITE:-0}
REQUEST_PORT=${REQUEST_PORT:-39220}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-14400}

if [[ ! "$REPEATS" =~ ^[0-9]+$ ]] || (( REPEATS < 3 )); then
  echo "REPEATS must be at least 3" >&2
  exit 2
fi

run_repeat() {
  local label=$1 offset=$2
  OUTPUT_ROOT="$OUTPUT_ROOT/$label" \
  K_VALUES="4 16" \
  ITERATIONS="$ITERATIONS" \
  ONE_D_SFM_DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_roman_trafalgar_datasets.txt" \
  ONE_D_SFM_PROBLEM_FILTER="roman_forum trafalgar" \
  BAL_PROBLEM_FILTER="52 3068" \
  REQUEST_PORT="$((REQUEST_PORT + offset))" \
  OVERWRITE="$OVERWRITE" \
  CASE_TIMEOUT_SECONDS="$CASE_TIMEOUT_SECONDS" \
  "$RUNNER"
}

run_repeat warmup 0
for repeat in $(seq 1 "$REPEATS"); do
  run_repeat "repeat${repeat}" "$((20 * repeat))"
done

"$PYTHON" "$ANALYZER" \
  --root "$OUTPUT_ROOT" \
  --repeats "$REPEATS" \
  --iterations "$ITERATIONS"

echo "Stage-C K4/K16 scaling repeats finished: $OUTPUT_ROOT"