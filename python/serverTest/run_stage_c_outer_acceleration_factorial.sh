#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_drs_failure_top3_live.sh"
ANALYZER="$SCRIPT_DIR/analyze_stage_c_outer_acceleration_factorial.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/stage_c_outer_acceleration_factorial_k24_i30"}
REQUEST_PORT=${REQUEST_PORT:-57220}
OVERWRITE=${OVERWRITE:-0}
MODES=${MODES:-"none nesterov themelis_nesterov lbfgs anderson"}

run_mode() {
  local family=$1 mode=$2 port_offset=$3
  local -a family_environment
  if [[ "$family" == "1dsfm" ]]; then
    family_environment=(
      ALL_PROBLEMS=0
      DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_roman_trafalgar_datasets.txt"
      PROBLEM_FILTER="roman_forum trafalgar"
    )
  else
    family_environment=(ALL_PROBLEMS=1 DATASET_LIST_FILE= PROBLEM_FILTER="52 3068")
  fi

  env \
    "${family_environment[@]}" \
    OUTPUT_DIR="$OUTPUT_ROOT/$family/$mode" \
    VARIANT_TAG="outer_accel_${mode}" \
    BUNDLE_PALM_REQUEST_PORT="$((REQUEST_PORT + port_offset))" \
    BUNDLE_PALM_RESULT_PORT="$((REQUEST_PORT + port_offset + 1))" \
    BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS=1 \
    BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=75 \
    BUNDLE_PALM_CAMERA_TRUST_DIAGONAL_SCALE=1e-4 \
    BUNDLE_PALM_MAXIMUM_TRUST_REGION_RADIUS=1e4 \
    CLUSTERS_LIST=24 ITERATIONS=30 THREADS_PER_CLUSTER=1 \
    LOCAL_STEPS=1 LOCAL_SOLVER=nesterov \
    NESTEROV_MAX_ITERATIONS=300 ENHANCED_INNER_MAX_ITERATIONS=300 \
    NESTEROV_MIN_ITERATIONS=1 NESTEROV_STOP_TOLERANCE=1e-2 \
    ENHANCED_INNER_UNTIL=30 \
    TRUST_REGION_POLICY=drs PERSISTENT_TRUST_REGION=1 \
    TRUST_REGION_RECOVERY_RATIO=0.5 \
    SCENE_NORMALIZATION=points_p95 CAMERA_SCALING=jacobi_initial \
    CAMERA_UPDATE=se3_left CLUSTERING=landmark_scalable \
    PROXIMAL_METRIC=block CONSENSUS_METRIC=full CONSENSUS_EXECUTION=coordinator \
    SHARED_ONLY_CAMERA_PROXIMAL=1 \
    BLOCK_REGULARIZATION=1e-4 BLOCK_CURVATURE_MULTIPLIER=0.4 \
    BLOCK_RECOVERY_MODE=curvature MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=64 \
    CURVATURE_DECAY_AFTER=5 CURVATURE_DECAY_RATIO=0.5 \
    METRIC_PROPOSAL_DISAGREEMENT_SCALE=0.5 \
    OUTER_ACCELERATION="$mode" LINE_SEARCH_GRID=0,1 \
    ACCELERATION_RESTART_AFTER=3 \
    SAFEGUARD_MODE=relative DRE_RELATIVE_INCREASE=0.01 \
    MINIMUM_PRIMAL_RATIO=1.001 \
    WORKER_OWNED_LANDMARKS=1 WORKER_OWNED_CAMERAS=0 \
    PACKED_REQUEST_BUFFERS=1 DEBUG_OUTPUT=0 LIVE_OUTPUT=0 \
    OVERWRITE="$OVERWRITE" CASE_TIMEOUT_SECONDS=14400 \
    "$RUNNER"
}

port_offset=0
for family in 1dsfm bal; do
  for mode in $MODES; do
    run_mode "$family" "$mode" "$port_offset"
    port_offset=$((port_offset + 2))
  done
done

"$PYTHON" "$ANALYZER" --root "$OUTPUT_ROOT" --iterations 30
echo "Stage-C outer-acceleration factorial finished: $OUTPUT_ROOT"