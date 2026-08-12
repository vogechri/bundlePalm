#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_stage_c_base_backbone_factorial.sh"
ANALYZER="$SCRIPT_DIR/analyze_stage_c_base_structure_factorial.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/stage_c_base_structure_factorial_k24_i30"}
SHARED_ROOT=${SHARED_ROOT:-"$WORKSPACE/benchmark_results/stage_c_base_backbone_confirmation_k24_i30"}
OVERWRITE=${OVERWRITE:-0}
REQUEST_PORT=${REQUEST_PORT:-43220}

one_d_sfm_filter=$(cut -d'|' -f1 "$SCRIPT_DIR/1dsfm_all_fifteen_datasets.txt" | tr '\n' ' ')
bal_filter=$(
  find "$WORKSPACE" -maxdepth 1 -type f -name 'problem-*-pre.txt' -printf '%f\n' \
    | sort -V | sed -E 's/^problem-([0-9]+)-.*/\1/' | tr '\n' ' '
)

run_arm() {
  local arm=$1 proposal=$2 port=$3
  OUTPUT_ROOT="$OUTPUT_ROOT/$arm" \
  FAMILIES="1dsfm bal" MODES=c1 RUN_ANALYZER=0 \
  ONE_D_SFM_DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_all_fifteen_datasets.txt" \
  ONE_D_SFM_PROBLEM_FILTER="$one_d_sfm_filter" \
  BAL_PROBLEM_FILTER="$bal_filter" \
  BASE_SHARED_ONLY_CAMERA_PROXIMAL=0 \
  BASE_METRIC_PROPOSAL_DISAGREEMENT_SCALE="$proposal" \
  BASE_WORKER_OWNED_CAMERAS=0 BASE_VARIANT_PREFIX="$arm" \
  OVERWRITE="$OVERWRITE" REQUEST_PORT="$port" \
  "$RUNNER"
}

run_arm legacy_proposal 0.5 "$REQUEST_PORT"
run_arm legacy_no_proposal 1.0 "$((REQUEST_PORT + 20))"

OUTPUT_ROOT="$OUTPUT_ROOT/shared_proposal" \
FAMILIES="1dsfm bal" MODES=c1 RUN_ANALYZER=0 \
ONE_D_SFM_DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_all_fifteen_datasets.txt" \
ONE_D_SFM_PROBLEM_FILTER="$one_d_sfm_filter" \
BAL_PROBLEM_FILTER="$bal_filter" \
BASE_SHARED_ONLY_CAMERA_PROXIMAL=1 \
BASE_METRIC_PROPOSAL_DISAGREEMENT_SCALE=0.5 \
BASE_WORKER_OWNED_CAMERAS=0 BASE_VARIANT_PREFIX=shared_proposal \
OVERWRITE="$OVERWRITE" REQUEST_PORT="$((REQUEST_PORT + 40))" \
"$RUNNER"

"$PYTHON" "$ANALYZER" \
  --legacy-proposal-root "$OUTPUT_ROOT/legacy_proposal" \
  --legacy-no-proposal-root "$OUTPUT_ROOT/legacy_no_proposal" \
  --shared-root "$SHARED_ROOT" \
  --shared-proposal-root "$OUTPUT_ROOT/shared_proposal" \
  --output-root "$OUTPUT_ROOT"

echo "Stage-C base structural factorial finished: $OUTPUT_ROOT"