#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_stage_c_final_benchmarks.sh"
ANALYZER="$SCRIPT_DIR/analyze_stage_c_tuned_repeats.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/stage_c_tuned_repeats_k24_i30"}
REPEATS=${REPEATS:-3}
OVERWRITE=${OVERWRITE:-0}
REQUEST_PORT=${REQUEST_PORT:-34220}

if (( REPEATS < 3 )); then
  echo "REPEATS must be at least 3" >&2
  exit 2
fi

run_repeat() {
  local label=$1 offset=$2
  DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_roman_trafalgar_datasets.txt" \
  PROBLEM_FILTER="roman_forum trafalgar" \
  FAMILIES=1dsfm MODES=c1_c5 \
  OUTPUT_ROOT="$OUTPUT_ROOT/$label" \
  REQUEST_PORT="$((REQUEST_PORT + offset))" \
  OVERWRITE="$OVERWRITE" \
  "$RUNNER"

  PROBLEM_FILTER="52 3068" \
  FAMILIES=bal MODES=c1_c5 \
  OUTPUT_ROOT="$OUTPUT_ROOT/$label" \
  REQUEST_PORT="$((REQUEST_PORT + offset + 2))" \
  OVERWRITE="$OVERWRITE" \
  "$RUNNER"
}

run_repeat warmup 0
for repeat in $(seq 1 "$REPEATS"); do
  run_repeat "repeat${repeat}" "$((4 * repeat))"
done

"$PYTHON" "$ANALYZER" \
  --root "$OUTPUT_ROOT" \
  --repeats "$REPEATS" \
  --require-complete

echo "Stage-C tuned repeats finished: $OUTPUT_ROOT"