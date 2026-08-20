#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
DETACHED_ROOT=${DETACHED_ROOT:-/home/chvogel/bundlePalm_k1_long/python}
RUNNER=${RUNNER:-"$DETACHED_ROOT/serverTest/run_drs_failure_top3_live.sh"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/k1_carryover_late_correction/phase2_restart_oracle_i90_stop60"}
CONTROL="$OUTPUT_ROOT/control_i90_correction"
STAGE_A="$OUTPUT_ROOT/stage_a_i60_correction"
STAGE_B="$OUTPUT_ROOT/stage_b_restart_i30"
DATASET_LIST_FILE="$OUTPUT_ROOT/datasets.txt"
STATE_DIR="$OUTPUT_ROOT/corrected_states"
REQUEST_PORT=${REQUEST_PORT:-30590}
OVERWRITE=${OVERWRITE:-0}
MODES=${MODES:-"control stage_a stage_b analyze"}

mkdir -p "$OUTPUT_ROOT" "$STATE_DIR"

one_d_sfm_manifest="$DETACHED_ROOT/serverTest/1dsfm_all_fifteen_datasets.txt"
{
  for scene in roman_forum trafalgar; do
    awk -F'|' -v scene="$scene" '$1 == scene { print }' "$one_d_sfm_manifest"
  done
  for problem_id in 52 3068; do
    dataset=$(find "$WORKSPACE" -maxdepth 1 -type f \
      -name "problem-${problem_id}-*-pre.txt" -print -quit)
    if [[ -z "$dataset" ]]; then
      echo "Missing BAL dataset $problem_id" >&2
      exit 2
    fi
    printf '%s|%s\n' "$problem_id" "$dataset"
  done
} > "$DATASET_LIST_FILE"

run_drs() {
  local output_dir=$1 iterations=$2 initial_state_directory=$3 final_correction=$4
  local stop_after_iteration=$5 port=$6 tag=$7
  cd "$DETACHED_ROOT/serverTest"
  env \
    OUTPUT_DIR="$output_dir" DATASET_LIST_FILE="$DATASET_LIST_FILE" \
    PROBLEM_FILTER="roman_forum trafalgar 52 3068" \
    INITIAL_STATE_DIRECTORY="$initial_state_directory" INITIAL_STATE_FRAME=raw \
    CLUSTERS_LIST=24 ITERATIONS="$iterations" LOCAL_STEPS=1 THREADS_PER_CLUSTER=1 \
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
    SAFEGUARD_ANNEALING_ITERATIONS=90 \
    WORKER_OWNED_LANDMARKS=1 WORKER_OWNED_CAMERAS=0 PACKED_REQUEST_BUFFERS=1 \
    INITIAL_SHARED_SCHUR_CORRECTION=0 FINAL_SHARED_SCHUR_CORRECTION="$final_correction" \
    STOP_AFTER_ITERATION="$stop_after_iteration" \
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
    CASE_TIMEOUT_SECONDS=14400 VARIANT_TAG="$tag" \
    "$RUNNER"
}

if [[ " $MODES " == *" control "* ]]; then
  run_drs "$CONTROL" 90 "" 1 0 "$((REQUEST_PORT + 4))" p2c
fi

if [[ " $MODES " == *" stage_a "* ]]; then
  run_drs "$STAGE_A" 90 "" 1 60 "$REQUEST_PORT" p2a
fi

if [[ " $MODES " == *" stage_b "* ]]; then
  for scene in roman_forum trafalgar; do
    state=$(find "$STAGE_A/states" -maxdepth 1 -name "*_${scene}_k24_i90_l1_t1.npz" -print -quit)
    if [[ -z "$state" || ! -f "$state" ]]; then
      echo "Missing corrected Stage A state for $scene" >&2
      exit 2
    fi
    ln -sfn "$state" "$STATE_DIR/$scene.npz"
  done
  for problem_id in 52 3068; do
    state=$(find "$STAGE_A/states" -maxdepth 1 -name "*_${problem_id}_k24_i90_l1_t1.npz" -print -quit)
    if [[ -z "$state" || ! -f "$state" ]]; then
      echo "Missing corrected Stage A state for BAL$problem_id" >&2
      exit 2
    fi
    ln -sfn "$state" "$STATE_DIR/$problem_id.npz"
  done
  run_drs "$STAGE_B" 30 "$STATE_DIR" 0 0 "$((REQUEST_PORT + 2))" p2b
fi

if [[ " $MODES " == *" analyze "* ]]; then
  "$WORKSPACE/serverTest/.venv/bin/python" \
    "$WORKSPACE/serverTest/analyze_k1_late_correction_phase2.py" \
    --root "$OUTPUT_ROOT"
fi
