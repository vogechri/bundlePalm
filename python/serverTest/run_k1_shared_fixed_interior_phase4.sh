#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
DETACHED_ROOT=${DETACHED_ROOT:-/home/chvogel/bundlePalm_k1_long/python}
RUNNER=${RUNNER:-"$DETACHED_ROOT/serverTest/run_drs_failure_top3_live.sh"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/k1_carryover_late_correction/phase4_interior_development"}
DATASET_LIST_FILE="$OUTPUT_ROOT/datasets.txt"
MODES=${MODES:-"control trial analyze"}
REQUEST_PORT=${REQUEST_PORT:-30620}
OVERWRITE=${OVERWRITE:-0}

mkdir -p "$OUTPUT_ROOT"
one_d_sfm_manifest="$DETACHED_ROOT/serverTest/1dsfm_all_fifteen_datasets.txt"
{
  for scene in gendarmenmarkt piccadilly roman_forum trafalgar union_square vienna_cathedral; do
    awk -F'|' -v scene="$scene" '$1 == scene { print }' "$one_d_sfm_manifest"
  done
  for problem_id in 52 245 1490 1778 3068; do
    dataset=$(find "$WORKSPACE" -maxdepth 1 -type f \
      -name "problem-${problem_id}-*-pre.txt" -print -quit)
    if [[ -z "$dataset" ]]; then
      echo "Missing BAL dataset $problem_id" >&2
      exit 2
    fi
    printf '%s|%s\n' "$problem_id" "$dataset"
  done
} > "$DATASET_LIST_FILE"

run_arm() {
  local arm=$1 trial=$2 port=$3
  cd "$DETACHED_ROOT/serverTest"
  env \
    OUTPUT_DIR="$OUTPUT_ROOT/$arm" DATASET_LIST_FILE="$DATASET_LIST_FILE" \
    PROBLEM_FILTER="gendarmenmarkt piccadilly roman_forum trafalgar union_square vienna_cathedral 52 245 1490 1778 3068" \
    CLUSTERS_LIST=24 ITERATIONS=30 LOCAL_STEPS=1 THREADS_PER_CLUSTER=1 \
    CAMERA_UPDATE=se3_left LOCAL_SOLVER=nesterov \
    NESTEROV_MAX_ITERATIONS=300 ENHANCED_INNER_MAX_ITERATIONS=300 \
    NESTEROV_MIN_ITERATIONS=1 NESTEROV_STOP_TOLERANCE=1e-2 \
    ENHANCED_INNER_UNTIL=30 TRUST_REGION_POLICY=drs PERSISTENT_TRUST_REGION=1 \
    TRUST_REGION_RECOVERY_RATIO=0.5 SCENE_NORMALIZATION=points_p95 \
    CAMERA_SCALING=jacobi_initial PROXIMAL_METRIC=block CONSENSUS_METRIC=full \
    CONSENSUS_EXECUTION=coordinator SHARED_ONLY_CAMERA_PROXIMAL=1 \
    BLOCK_REGULARIZATION=1e-4 BLOCK_CURVATURE_MULTIPLIER=0.4 \
    BLOCK_RECOVERY_MODE=curvature MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=64 \
    CURVATURE_DECAY_AFTER=5 CURVATURE_DECAY_RATIO=0.5 \
    METRIC_PROPOSAL_DISAGREEMENT_SCALE=1 OUTER_ACCELERATION=themelis_nesterov \
    LINE_SEARCH_GRID=0,1 ACCELERATION_RESTART_AFTER=3 ADAPTIVE_LOCAL_DEPTH=0 \
    WORKER_OWNED_LANDMARKS=1 WORKER_OWNED_CAMERAS=0 PACKED_REQUEST_BUFFERS=1 \
    BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS=1 \
    BUNDLE_PALM_DISABLE_LANDMARK_PRECONDITIONING=0 \
    BUNDLE_PALM_DIAGONAL_TRUST_DAMPING=0 BUNDLE_PALM_BAE_TRUST_SCHEDULE=0 \
    BUNDLE_PALM_CUMULATIVE_DIAGONAL_DAMPING=0 \
    BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=75 \
    BUNDLE_PALM_CAMERA_TRUST_DIAGONAL_SCALE=1e-4 \
    BUNDLE_PALM_INITIAL_TRUST_REGION_RADIUS=10 \
    BUNDLE_PALM_MAXIMUM_TRUST_REGION_RADIUS=10000 \
    BUNDLE_PALM_DABA_INITIAL_TRUST_REGION_CAP=100 \
    BUNDLE_PALM_SHARED_FIXED_INTERIOR_TRIAL="$trial" \
    BUNDLE_PALM_SHARED_FIXED_INTERIOR_TRIAL_MAX_BACKTRACKS=8 \
    BUNDLE_PALM_LOCAL_SOLVE_METRICS=1 \
    BUNDLE_PALM_REQUEST_PORT="$port" BUNDLE_PALM_RESULT_PORT="$((port + 1))" \
    DEBUG_OUTPUT=1 LIVE_OUTPUT=0 WORKER_LIVE_OUTPUT=0 OVERWRITE="$OVERWRITE" \
    CASE_TIMEOUT_SECONDS=14400 VARIANT_TAG="p4_${arm}" \
    "$RUNNER"
}

if [[ " $MODES " == *" control "* ]]; then
  run_arm control 0 "$REQUEST_PORT"
fi
if [[ " $MODES " == *" trial "* ]]; then
  run_arm trial 1 "$((REQUEST_PORT + 2))"
fi
if [[ " $MODES " == *" analyze "* ]]; then
  "$WORKSPACE/serverTest/.venv/bin/python" \
    "$WORKSPACE/serverTest/analyze_k1_shared_fixed_interior_phase4.py" \
    --root "$OUTPUT_ROOT"
fi
