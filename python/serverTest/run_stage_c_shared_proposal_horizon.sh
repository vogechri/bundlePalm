#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_stage_c_base_backbone_factorial.sh"
ANALYZER="$SCRIPT_DIR/analyze_stage_c_shared_proposal_horizon.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
ITERATIONS=${ITERATIONS:-90}
PROPOSAL_UNTIL=${PROPOSAL_UNTIL:-30}
ACCELERATION_UNTIL=${ACCELERATION_UNTIL:-30}
REBASE_ITERATION=${REBASE_ITERATION:-0}
MODE=${MODE:-c1}
PROPOSAL_SCALE=${PROPOSAL_SCALE:-0.5}
SHARED_ONLY=${SHARED_ONLY:-1}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/stage_c_shared_${MODE}_proposal${PROPOSAL_SCALE}_rebase${REBASE_ITERATION}_k24_i${ITERATIONS}"}
OVERWRITE=${OVERWRITE:-0}
REQUEST_PORT=${REQUEST_PORT:-44220}

one_d_sfm_filter=$(cut -d'|' -f1 "$SCRIPT_DIR/1dsfm_all_fifteen_datasets.txt" | tr '\n' ' ')
bal_filter=$(
  find "$WORKSPACE" -maxdepth 1 -type f -name 'problem-*-pre.txt' -printf '%f\n' \
    | sort -V | sed -E 's/^problem-([0-9]+)-.*/\1/' | tr '\n' ' '
)

OUTPUT_ROOT="$OUTPUT_ROOT" \
ITERATIONS="$ITERATIONS" FAMILIES="1dsfm bal" MODES="$MODE" RUN_ANALYZER=0 \
ONE_D_SFM_DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_all_fifteen_datasets.txt" \
ONE_D_SFM_PROBLEM_FILTER="$one_d_sfm_filter" BAL_PROBLEM_FILTER="$bal_filter" \
BASE_SHARED_ONLY_CAMERA_PROXIMAL="$SHARED_ONLY" \
BASE_METRIC_PROPOSAL_DISAGREEMENT_SCALE="$PROPOSAL_SCALE" \
BASE_METRIC_PROPOSAL_DISAGREEMENT_UNTIL="$PROPOSAL_UNTIL" \
BASE_OUTER_ACCELERATION_UNTIL="$ACCELERATION_UNTIL" \
BASE_LOCAL_STATE_REBASE_ITERATION="$REBASE_ITERATION" \
BASE_WORKER_OWNED_CAMERAS=0 BASE_VARIANT_PREFIX=shared_proposal_horizon \
OVERWRITE="$OVERWRITE" REQUEST_PORT="$REQUEST_PORT" \
"$RUNNER"

"$PYTHON" "$ANALYZER" \
  --root "$OUTPUT_ROOT" --iterations "$ITERATIONS" \
  --proposal-until "$PROPOSAL_UNTIL" \
  --acceleration-until "$ACCELERATION_UNTIL" \
  --rebase-iteration "$REBASE_ITERATION" \
  --mode "$MODE" --proposal-scale "$PROPOSAL_SCALE" \
  --shared-only "$SHARED_ONLY"
echo "Stage-C shared-proposal horizon finished: $OUTPUT_ROOT"