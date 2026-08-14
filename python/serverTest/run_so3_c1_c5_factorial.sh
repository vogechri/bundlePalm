#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_stage_c_final_benchmarks.sh"
ANALYZER="$SCRIPT_DIR/analyze_so3_c1_c5_factorial.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/so3_c1_c5_development_k24_i30"}
LEFT_REFERENCE_ROOT=${LEFT_REFERENCE_ROOT:-"$WORKSPACE/benchmark_results/camera_parameterization_so3_development_k24_i30/se3_left"}
ITERATIONS=${ITERATIONS:-30}
REQUEST_PORT=${REQUEST_PORT:-62420}
OVERWRITE=${OVERWRITE:-0}
MODES=${MODES:-"plain c1 c5 c1_c5"}

one_d_sfm_filter=$(cut -d'|' -f1 "$SCRIPT_DIR/1dsfm_six_datasets.txt" | tr '\n' ' ')
bal_filter=$(cut -d'|' -f1 "$SCRIPT_DIR/bal_tuning_sentinel_five_datasets.txt" | tr '\n' ' ')

DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_six_datasets.txt" \
PROBLEM_FILTER="$one_d_sfm_filter" \
FAMILIES=1dsfm MODES="$MODES" ITERATIONS="$ITERATIONS" \
CAMERA_UPDATE=so3_left SCHUR_PCG_MAX_ITERATIONS=1000 \
OUTPUT_ROOT="$OUTPUT_ROOT" REQUEST_PORT="$REQUEST_PORT" \
OVERWRITE="$OVERWRITE" "$RUNNER"

PROBLEM_FILTER="$bal_filter" \
FAMILIES=bal MODES="$MODES" ITERATIONS="$ITERATIONS" \
CAMERA_UPDATE=so3_left SCHUR_PCG_MAX_ITERATIONS=1000 \
OUTPUT_ROOT="$OUTPUT_ROOT" REQUEST_PORT="$((REQUEST_PORT + 8))" \
OVERWRITE="$OVERWRITE" "$RUNNER"

"$PYTHON" "$ANALYZER" \
  --root "$OUTPUT_ROOT" \
  --left-reference-root "$LEFT_REFERENCE_ROOT" \
  --iterations "$ITERATIONS" --require-complete

echo "SO3 C1 x C5 factorial finished: $OUTPUT_ROOT"
