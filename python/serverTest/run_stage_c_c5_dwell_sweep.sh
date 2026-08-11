#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_stage_c_final_benchmarks.sh"
ANALYZER="$SCRIPT_DIR/analyze_stage_c_c5_high_sweep.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/stage_c_c5_dwell_development_k24_i30"}
REFERENCE_ROOT=${REFERENCE_ROOT:-"$WORKSPACE/benchmark_results/stage_c_final_all15_all29_k24_i30"}
ITERATIONS=${ITERATIONS:-30}
HIGH_THRESHOLD=${HIGH_THRESHOLD:-0.35}
LOW_THRESHOLD=${LOW_THRESHOLD:-0.20}
WINDOW=${WINDOW:-3}
VALUES=${VALUES:-"3 5"}
OVERWRITE=${OVERWRITE:-0}
REQUEST_PORT=${REQUEST_PORT:-32820}

one_d_sfm_filter=$(cut -d'|' -f1 "$SCRIPT_DIR/1dsfm_six_datasets.txt" | tr '\n' ' ')
bal_filter=$(cut -d'|' -f1 "$SCRIPT_DIR/bal_tuning_sentinel_five_datasets.txt" | tr '\n' ' ')

index=0
for value in $VALUES; do
  tag="d${value}p00"
  DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_six_datasets.txt" \
  PROBLEM_FILTER="$one_d_sfm_filter" \
  FAMILIES=1dsfm MODES=c1_c5 ITERATIONS="$ITERATIONS" \
  C5_HIGH_THRESHOLD="$HIGH_THRESHOLD" C5_LOW_THRESHOLD="$LOW_THRESHOLD" \
  C5_WINDOW="$WINDOW" C5_DWELL="$value" \
  OUTPUT_ROOT="$OUTPUT_ROOT/$tag" OVERWRITE="$OVERWRITE" \
  REQUEST_PORT="$((REQUEST_PORT + 4 * index))" \
  "$RUNNER"

  PROBLEM_FILTER="$bal_filter" \
  FAMILIES=bal MODES=c1_c5 ITERATIONS="$ITERATIONS" \
  C5_HIGH_THRESHOLD="$HIGH_THRESHOLD" C5_LOW_THRESHOLD="$LOW_THRESHOLD" \
  C5_WINDOW="$WINDOW" C5_DWELL="$value" \
  OUTPUT_ROOT="$OUTPUT_ROOT/$tag" OVERWRITE="$OVERWRITE" \
  REQUEST_PORT="$((REQUEST_PORT + 4 * index + 2))" \
  "$RUNNER"
  index=$((index + 1))
done

"$PYTHON" "$ANALYZER" \
  --root "$OUTPUT_ROOT" \
  --reference-root "$REFERENCE_ROOT" \
  --thresholds "$VALUES" \
  --parameter dwell \
  --tag-prefix d \
  --iterations "$ITERATIONS" \
  --require-complete

echo "Stage-C C5 dwell sweep finished: $OUTPUT_ROOT"