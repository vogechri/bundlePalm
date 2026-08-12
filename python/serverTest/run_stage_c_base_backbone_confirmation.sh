#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_stage_c_base_backbone_factorial.sh"
ANALYZER="$SCRIPT_DIR/analyze_stage_c_base_backbone_confirmation.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/stage_c_base_backbone_confirmation_k24_i30"}
OVERWRITE=${OVERWRITE:-0}
REQUEST_PORT=${REQUEST_PORT:-42220}

one_d_sfm_filter=$(cut -d'|' -f1 "$SCRIPT_DIR/1dsfm_all_fifteen_datasets.txt" | tr '\n' ' ')
bal_filter=$(
  find "$WORKSPACE" -maxdepth 1 -type f -name 'problem-*-pre.txt' -printf '%f\n' \
    | sort -V | sed -E 's/^problem-([0-9]+)-.*/\1/' | tr '\n' ' '
)

OUTPUT_ROOT="$OUTPUT_ROOT" \
FAMILIES="1dsfm bal" MODES="c1 c1_c5" RUN_ANALYZER=0 \
ONE_D_SFM_DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_all_fifteen_datasets.txt" \
ONE_D_SFM_PROBLEM_FILTER="$one_d_sfm_filter" \
BAL_PROBLEM_FILTER="$bal_filter" \
OVERWRITE="$OVERWRITE" REQUEST_PORT="$REQUEST_PORT" \
"$RUNNER"

"$PYTHON" "$ANALYZER" --root "$OUTPUT_ROOT" --iterations 30
echo "Stage-C base-backbone all-scene confirmation finished: $OUTPUT_ROOT"