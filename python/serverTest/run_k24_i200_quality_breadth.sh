#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
DETACHED_ROOT=${DETACHED_ROOT:-/home/chvogel/bundlePalm_k1_long/python}
RUNNER=${RUNNER:-"$DETACHED_ROOT/serverTest/run_drs_failure_top3_live.sh"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/k24_i200_terminal_correction_breadth"}
REQUEST_PORT=${REQUEST_PORT:-30820}
OVERWRITE=${OVERWRITE:-0}
FAMILIES=${FAMILIES:-"1dsfm bal"}
ARMS=${ARMS:-"i90 i200"}
ANALYZE=${ANALYZE:-1}

mkdir -p "$OUTPUT_ROOT"
one_d_sfm_manifest="$DETACHED_ROOT/serverTest/1dsfm_all_fifteen_datasets.txt"
one_d_sfm_filter=$(cut -d'|' -f1 "$one_d_sfm_manifest" | tr '\n' ' ')
bal_manifest="$OUTPUT_ROOT/bal_datasets.txt"
for dataset in $(find "$WORKSPACE" -maxdepth 1 -type f -name 'problem-*-pre.txt' -printf '%p\n' | sort -V); do
  problem_id=$(basename "$dataset" | sed -E 's/^problem-([0-9]+)-.*/\1/')
  printf '%s|%s\n' "$problem_id" "$dataset" >> "$bal_manifest"
done
bal_filter=$(cut -d'|' -f1 "$bal_manifest" | tr '\n' ' ')

run_arm() {
  local family=$1 arm=$2 stop_after=$3 port=$4 tag=$5
  local manifest filter
  if [[ "$family" == "1dsfm" ]]; then
    manifest="$one_d_sfm_manifest"
    filter="$one_d_sfm_filter"
  else
    manifest="$bal_manifest"
    filter="$bal_filter"
  fi
  cd "$DETACHED_ROOT/serverTest"
  env \
    OUTPUT_DIR="$OUTPUT_ROOT/$family/$arm" DATASET_LIST_FILE="$manifest" \
    PROBLEM_FILTER="$filter" CLUSTERS_LIST=24 ITERATIONS=200 \
    STOP_AFTER_ITERATION="$stop_after" LOCAL_STEPS=1 THREADS_PER_CLUSTER=1 \
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
    SAFEGUARD_ANNEALING_ITERATIONS=200 \
    WORKER_OWNED_LANDMARKS=1 WORKER_OWNED_CAMERAS=0 PACKED_REQUEST_BUFFERS=1 \
    INITIAL_SHARED_SCHUR_CORRECTION=0 FINAL_SHARED_SCHUR_CORRECTION=1 \
    SHARED_SCHUR_MAXIMUM_CORRECTIONS=1 \
    SHARED_SCHUR_CAMERA_DAMPING=0.005859375 \
    SHARED_SCHUR_LANDMARK_DAMPING=0.005859375 \
    SHARED_SCHUR_MINIMUM_RELATIVE_DECREASE=1e-3 \
    SHARED_SCHUR_RELATIVE_TOLERANCE=1e-6 \
    SHARED_SCHUR_OPERATOR=bsr_low_memory SHARED_SCHUR_PRECONDITIONER=jacobi \
    BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS=1 \
    BUNDLE_PALM_DISABLE_LANDMARK_PRECONDITIONING=0 \
    BUNDLE_PALM_DIAGONAL_TRUST_DAMPING=0 BUNDLE_PALM_BAE_TRUST_SCHEDULE=0 \
    BUNDLE_PALM_CUMULATIVE_DIAGONAL_DAMPING=0 \
    BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=75 \
    BUNDLE_PALM_CAMERA_TRUST_DIAGONAL_SCALE=1e-4 \
    BUNDLE_PALM_INITIAL_TRUST_REGION_RADIUS=10 \
    BUNDLE_PALM_MAXIMUM_TRUST_REGION_RADIUS=10000 \
    BUNDLE_PALM_DABA_INITIAL_TRUST_REGION_CAP=100 \
    BUNDLE_PALM_REQUEST_PORT="$port" BUNDLE_PALM_RESULT_PORT="$((port + 1))" \
    DEBUG_OUTPUT=1 LIVE_OUTPUT=0 WORKER_LIVE_OUTPUT=0 OVERWRITE="$OVERWRITE" \
    CASE_TIMEOUT_SECONDS=28800 VARIANT_TAG="$tag" \
    "$RUNNER"
}

port=$REQUEST_PORT
for family in $FAMILIES; do
  if [[ " $ARMS " == *" i90 "* ]]; then
    run_arm "$family" i90_horizon200 90 "$port" "b_${family}_h90"
    port=$((port + 2))
  fi
  if [[ " $ARMS " == *" i200 "* ]]; then
    run_arm "$family" i200 0 "$port" "b_${family}_i200"
    port=$((port + 2))
  fi
done

if [[ "$ANALYZE" == "1" ]]; then
  "$WORKSPACE/serverTest/.venv/bin/python" \
    "$WORKSPACE/serverTest/analyze_k24_i200_quality_breadth.py" \
    --root "$OUTPUT_ROOT"
fi
