#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_stage_c_final_benchmarks.sh"
ANALYZER="$SCRIPT_DIR/analyze_so3_maximum_trust_sweep.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/so3_maximum_trust_development_k24_i30"}
LEFT_REFERENCE_ROOT=${LEFT_REFERENCE_ROOT:-"$WORKSPACE/benchmark_results/camera_parameterization_so3_development_k24_i30/se3_left"}
VALUES=${VALUES:-"1000 10000 100000 1000000"}
ITERATIONS=${ITERATIONS:-30}
REQUEST_PORT=${REQUEST_PORT:-64220}
OVERWRITE=${OVERWRITE:-0}

one_d_sfm_filter=$(cut -d'|' -f1 "$SCRIPT_DIR/1dsfm_six_datasets.txt" | tr '\n' ' ')
bal_filter=$(cut -d'|' -f1 "$SCRIPT_DIR/bal_tuning_sentinel_five_datasets.txt" | tr '\n' ' ')

index=0
for value in $VALUES; do
  tag="max${value}"
  DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_six_datasets.txt" \
  PROBLEM_FILTER="$one_d_sfm_filter" \
  FAMILIES=1dsfm MODES=c1_c5 ITERATIONS="$ITERATIONS" \
  CAMERA_UPDATE=so3_left SCHUR_PCG_MAX_ITERATIONS=1000 \
  CAMERA_DIAGONAL_METRIC_SCALE=25 SO3_TRANSLATION_METRIC_RATIO=1 \
  C5_START_ITERATION=5 INITIAL_TRUST_REGION_RADIUS=10 \
  DABA_INITIAL_TRUST_REGION_CAP=100 MAXIMUM_TRUST_REGION_RADIUS="$value" \
  OUTPUT_ROOT="$OUTPUT_ROOT/$tag" REQUEST_PORT="$((REQUEST_PORT + 4 * index))" \
  OVERWRITE="$OVERWRITE" "$RUNNER"

  PROBLEM_FILTER="$bal_filter" \
  FAMILIES=bal MODES=c1_c5 ITERATIONS="$ITERATIONS" \
  CAMERA_UPDATE=so3_left SCHUR_PCG_MAX_ITERATIONS=1000 \
  CAMERA_DIAGONAL_METRIC_SCALE=25 SO3_TRANSLATION_METRIC_RATIO=1 \
  C5_START_ITERATION=5 INITIAL_TRUST_REGION_RADIUS=10 \
  DABA_INITIAL_TRUST_REGION_CAP=100 MAXIMUM_TRUST_REGION_RADIUS="$value" \
  OUTPUT_ROOT="$OUTPUT_ROOT/$tag" REQUEST_PORT="$((REQUEST_PORT + 4 * index + 2))" \
  OVERWRITE="$OVERWRITE" "$RUNNER"
  index=$((index + 1))
done

"$PYTHON" "$ANALYZER" \
  --root "$OUTPUT_ROOT" --left-reference-root "$LEFT_REFERENCE_ROOT" \
  --values $VALUES --iterations "$ITERATIONS" --require-complete

echo "SO3 maximum-trust sweep finished: $OUTPUT_ROOT"
