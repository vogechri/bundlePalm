#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
DETACHED_ROOT=${DETACHED_ROOT:-/home/chvogel/bundlePalm_k1_long/python}
RUNNER=${RUNNER:-"$DETACHED_ROOT/serverTest/run_drs_failure_top3_live.sh"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/terminal_correction_scaling_sentinels"}
REQUEST_PORT=${REQUEST_PORT:-30640}
OVERWRITE=${OVERWRITE:-0}
MODES=${MODES:-"reload inprocess analyze"}

mkdir -p "$OUTPUT_ROOT"
one_d_sfm_manifest="$DETACHED_ROOT/serverTest/1dsfm_all_fifteen_datasets.txt"
reload_manifest="$OUTPUT_ROOT/reload_datasets.txt"
inprocess_manifest="$OUTPUT_ROOT/inprocess_datasets.txt"
{
  for scene in roman_forum trafalgar; do
    awk -F'|' -v scene="$scene" '$1 == scene { print }' "$one_d_sfm_manifest"
  done
  dataset=$(find "$WORKSPACE" -maxdepth 1 -type f -name 'problem-52-*-pre.txt' -print -quit)
  printf '52|%s\n' "$dataset"
} > "$reload_manifest"
dataset=$(find "$WORKSPACE" -maxdepth 1 -type f -name 'problem-3068-*-pre.txt' -print -quit)
printf '3068|%s\n' "$dataset" > "$inprocess_manifest"

common_environment=(
  LOCAL_STEPS=1 THREADS_PER_CLUSTER=1 CAMERA_UPDATE=se3_left
  SCENE_NORMALIZATION=points_p95 CAMERA_SCALING=jacobi_initial
  PROXIMAL_METRIC=block CONSENSUS_METRIC=full SHARED_ONLY_CAMERA_PROXIMAL=1
  FINAL_SHARED_SCHUR_CORRECTION=1 SHARED_SCHUR_MAXIMUM_CORRECTIONS=1
  SHARED_SCHUR_CAMERA_DAMPING=0.005859375
  SHARED_SCHUR_LANDMARK_DAMPING=0.005859375
  SHARED_SCHUR_MINIMUM_RELATIVE_DECREASE=1e-3
  SHARED_SCHUR_RELATIVE_TOLERANCE=1e-6
  SHARED_SCHUR_OPERATOR=bsr_low_memory SHARED_SCHUR_PRECONDITIONER=jacobi
  BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS=1
  BUNDLE_PALM_DISABLE_LANDMARK_PRECONDITIONING=0
  BUNDLE_PALM_DIAGONAL_TRUST_DAMPING=0 BUNDLE_PALM_BAE_TRUST_SCHEDULE=0
  BUNDLE_PALM_CUMULATIVE_DIAGONAL_DAMPING=0
  BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=25
  BUNDLE_PALM_INITIAL_TRUST_REGION_RADIUS=10
  BUNDLE_PALM_MAXIMUM_TRUST_REGION_RADIUS=1000000
  BUNDLE_PALM_DABA_INITIAL_TRUST_REGION_CAP=100
  DEBUG_OUTPUT=1 LIVE_OUTPUT=0 WORKER_LIVE_OUTPUT=0 OVERWRITE="$OVERWRITE"
  CASE_TIMEOUT_SECONDS=14400
)

cd "$DETACHED_ROOT/serverTest"
for clusters in 4 16; do
  if [[ " $MODES " == *" reload "* ]]; then
    state_dir="$OUTPUT_ROOT/reload_k${clusters}_states"
    mkdir -p "$state_dir"
    for scene in roman_forum trafalgar; do
      state=$(find "$WORKSPACE/benchmark_results/stage_c_scaling_confirmation_k4_16_i30/1dsfm/c1_c5/states" \
        -maxdepth 1 -name "*_${scene}_k${clusters}_i30_l1_t1.npz" -print -quit)
      ln -sfn "$state" "$state_dir/$scene.npz"
    done
    state=$(find "$WORKSPACE/benchmark_results/stage_c_scaling_confirmation_k4_16_i30/bal/c1_c5/states" \
      -maxdepth 1 -name "*_52_k${clusters}_i30_l1_t1.npz" -print -quit)
    ln -sfn "$state" "$state_dir/52.npz"
    env "${common_environment[@]}" \
      OUTPUT_DIR="$OUTPUT_ROOT/reload_k${clusters}" \
      DATASET_LIST_FILE="$reload_manifest" PROBLEM_FILTER="roman_forum trafalgar 52" \
      INITIAL_STATE_DIRECTORY="$state_dir" INITIAL_STATE_FRAME=raw \
      CLUSTERS_LIST="$clusters" ITERATIONS=0 LOCAL_SOLVER=nesterov \
      TRUST_REGION_POLICY=drs PERSISTENT_TRUST_REGION=1 \
      OUTER_ACCELERATION=none ADAPTIVE_LOCAL_DEPTH=0 \
      BUNDLE_PALM_REQUEST_PORT="$((REQUEST_PORT + clusters))" \
      BUNDLE_PALM_RESULT_PORT="$((REQUEST_PORT + clusters + 1))" \
      VARIANT_TAG="scale_reload_k${clusters}" "$RUNNER"
  fi

  if [[ " $MODES " == *" inprocess "* ]]; then
    env "${common_environment[@]}" \
      OUTPUT_DIR="$OUTPUT_ROOT/inprocess_k${clusters}" \
      DATASET_LIST_FILE="$inprocess_manifest" PROBLEM_FILTER=3068 \
      CLUSTERS_LIST="$clusters" ITERATIONS=30 LOCAL_SOLVER=schur_pcg \
      TRUST_REGION_POLICY=daba PERSISTENT_TRUST_REGION=1 \
      BLOCK_REGULARIZATION=5e-5 BLOCK_CURVATURE_MULTIPLIER=0 \
      BLOCK_RECOVERY_MODE=regularization MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=16 \
      OUTER_ACCELERATION=themelis_nesterov LINE_SEARCH_GRID=0,1 \
      ACCELERATION_RESTART_AFTER=3 ADAPTIVE_LOCAL_DEPTH=1 \
      ADAPTIVE_LOCAL_DEPTH_START=5 ADAPTIVE_LOCAL_DEPTH_MAXIMUM=2 \
      ADAPTIVE_LOCAL_DEPTH_HIGH=0.35 ADAPTIVE_LOCAL_DEPTH_LOW=0.20 \
      ADAPTIVE_LOCAL_DEPTH_WINDOW=3 ADAPTIVE_LOCAL_DEPTH_DWELL=3 \
      SAFEGUARD_MODE=relative DRE_RELATIVE_INCREASE=0.01 MINIMUM_PRIMAL_RATIO=1.001 \
      WORKER_OWNED_LANDMARKS=1 WORKER_OWNED_CAMERAS=1 PACKED_REQUEST_BUFFERS=1 \
      BUNDLE_PALM_SCHUR_PCG_RELATIVE_TOLERANCE=0.01 \
      BUNDLE_PALM_SCHUR_PCG_Q_TOLERANCE=0 \
      BUNDLE_PALM_SCHUR_PCG_JACOBI_PRECONDITIONER=0 \
      BUNDLE_PALM_SCHUR_PCG_MAX_ITERATIONS=400 \
      BUNDLE_PALM_REQUEST_PORT="$((REQUEST_PORT + 40 + clusters))" \
      BUNDLE_PALM_RESULT_PORT="$((REQUEST_PORT + 41 + clusters))" \
      VARIANT_TAG="scale_inproc_k${clusters}" "$RUNNER"
  fi
done

if [[ " $MODES " == *" analyze "* ]]; then
  "$WORKSPACE/serverTest/.venv/bin/python" \
    "$WORKSPACE/serverTest/analyze_terminal_correction_scaling_sentinels.py" \
    --root "$OUTPUT_ROOT"
fi
