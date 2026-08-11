#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_drs_failure_top3_live.sh"
ANALYZER="$SCRIPT_DIR/analyze_stage_c_final_benchmarks.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/stage_c_tuned_final_all15_all29_k24_i30"}
ITERATIONS=${ITERATIONS:-30}
CLUSTERS=${CLUSTERS:-24}
THREADS_PER_CLUSTER=${THREADS_PER_CLUSTER:-1}
REQUEST_PORT=${REQUEST_PORT:-30220}
OVERWRITE=${OVERWRITE:-0}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-14400}
FAMILIES=${FAMILIES:-"1dsfm bal"}
MODES=${MODES:-"plain c1 c5 c1_c5"}
C5_MAXIMUM_DEPTH=${C5_MAXIMUM_DEPTH:-2}
C5_START_ITERATION=${C5_START_ITERATION:-5}
C5_HIGH_THRESHOLD=${C5_HIGH_THRESHOLD:-0.35}
C5_LOW_THRESHOLD=${C5_LOW_THRESHOLD:-0.20}
C5_WINDOW=${C5_WINDOW:-3}
C5_DWELL=${C5_DWELL:-3}
HUBER_DELTA=${HUBER_DELTA:-0}
ONE_D_SFM_DATASET_LIST_FILE=${DATASET_LIST_FILE:-"$SCRIPT_DIR/1dsfm_all_fifteen_datasets.txt"}
ONE_D_SFM_PROBLEM_FILTER=${PROBLEM_FILTER:-"$(cut -d'|' -f1 "$ONE_D_SFM_DATASET_LIST_FILE" | tr '\n' ' ')"}
BAL_PROBLEM_FILTER=${PROBLEM_FILTER:-"$(
  find "$WORKSPACE" -maxdepth 1 -type f -name 'problem-*-pre.txt' \
    -printf '%f\n' | sort -V \
    | sed -E 's/^problem-([0-9]+)-.*/\1/' | tr '\n' ' '
)"}

run_case() {
  local family=$1 mode=$2 port_offset=$3
  local outer_acceleration=none
  local adaptive_local_depth=0
  local -a family_environment

  if [[ "$mode" == "c1" ]]; then
    outer_acceleration=themelis_nesterov
  elif [[ "$mode" == "c5" ]]; then
    adaptive_local_depth=1
  elif [[ "$mode" == "c1_c5" ]]; then
    outer_acceleration=themelis_nesterov
    adaptive_local_depth=1
  elif [[ "$mode" != "plain" ]]; then
    echo "Unknown mode: $mode" >&2
    return 2
  fi

  if [[ "$family" == "1dsfm" ]]; then
    family_environment=(
      ALL_PROBLEMS=0
      DATASET_LIST_FILE="$ONE_D_SFM_DATASET_LIST_FILE"
      PROBLEM_FILTER="$ONE_D_SFM_PROBLEM_FILTER"
    )
  elif [[ "$family" == "bal" ]]; then
    family_environment=(
      ALL_PROBLEMS=1
      DATASET_LIST_FILE=
      PROBLEM_FILTER="$BAL_PROBLEM_FILTER"
    )
  else
    echo "Unknown family: $family" >&2
    return 2
  fi

  env \
    "${family_environment[@]}" \
    OUTPUT_DIR="$OUTPUT_ROOT/$family/$mode" \
    VARIANT_TAG="stage_c_final_${mode}" \
    BUNDLE_PALM_REQUEST_PORT="$((REQUEST_PORT + port_offset))" \
    BUNDLE_PALM_RESULT_PORT="$((REQUEST_PORT + port_offset + 1))" \
    BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS=1 \
    BUNDLE_PALM_DISABLE_LANDMARK_PRECONDITIONING=0 \
    BUNDLE_PALM_DIAGONAL_TRUST_DAMPING=0 \
    BUNDLE_PALM_BAE_TRUST_SCHEDULE=0 \
    BUNDLE_PALM_CUMULATIVE_DIAGONAL_DAMPING=0 \
    BUNDLE_PALM_SCHUR_PCG_RELATIVE_TOLERANCE=0.01 \
    BUNDLE_PALM_SCHUR_PCG_Q_TOLERANCE=0 \
    BUNDLE_PALM_SCHUR_PCG_JACOBI_PRECONDITIONER=0 \
    BUNDLE_PALM_SCHUR_PCG_MAX_ITERATIONS=400 \
    CLUSTERS_LIST="$CLUSTERS" \
    ITERATIONS="$ITERATIONS" \
    THREADS_PER_CLUSTER="$THREADS_PER_CLUSTER" \
    LOCAL_STEPS=1 \
    LOCAL_SOLVER=schur_pcg \
    TRUST_REGION_POLICY=daba \
    PERSISTENT_TRUST_REGION=1 \
    SCENE_NORMALIZATION=points_p95 \
    CAMERA_SCALING=jacobi_initial \
    CAMERA_UPDATE=se3_left \
    CLUSTERING=landmark_scalable \
    OUTER_ACCELERATION="$outer_acceleration" \
    LINE_SEARCH_GRID=0,1 \
    ACCELERATION_RESTART_AFTER=3 \
    ADAPTIVE_LOCAL_DEPTH="$adaptive_local_depth" \
    ADAPTIVE_LOCAL_DEPTH_START="$C5_START_ITERATION" \
    ADAPTIVE_LOCAL_DEPTH_MAXIMUM="$C5_MAXIMUM_DEPTH" \
    ADAPTIVE_LOCAL_DEPTH_HIGH="$C5_HIGH_THRESHOLD" \
    ADAPTIVE_LOCAL_DEPTH_LOW="$C5_LOW_THRESHOLD" \
    ADAPTIVE_LOCAL_DEPTH_WINDOW="$C5_WINDOW" \
    ADAPTIVE_LOCAL_DEPTH_DWELL="$C5_DWELL" \
    PROXIMAL_METRIC=block \
    CONSENSUS_METRIC=full \
    SHARED_ONLY_CAMERA_PROXIMAL=1 \
    BLOCK_REGULARIZATION=5e-5 \
    BLOCK_CURVATURE_MULTIPLIER=0 \
    MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=16 \
    BLOCK_RECOVERY_MODE=regularization \
    HUBER_DELTA="$HUBER_DELTA" \
    SAFEGUARD_MODE=relative \
    DRE_RELATIVE_INCREASE=0.01 \
    MINIMUM_PRIMAL_RATIO=1.001 \
    WORKER_OWNED_LANDMARKS=1 \
    WORKER_OWNED_CAMERAS=1 \
    PACKED_REQUEST_BUFFERS=1 \
    DEBUG_OUTPUT=1 \
    LIVE_OUTPUT=0 \
    OVERWRITE="$OVERWRITE" \
    CASE_TIMEOUT_SECONDS="$CASE_TIMEOUT_SECONDS" \
    "$RUNNER"
}

port_offset=0
for family in $FAMILIES; do
  for mode in $MODES; do
    run_case "$family" "$mode" "$port_offset"
    port_offset=$((port_offset + 2))
  done
done

if [[ " $FAMILIES " == *" 1dsfm "* && " $FAMILIES " == *" bal "* && \
  " $MODES " == *" plain "* && " $MODES " == *" c1 "* && \
  " $MODES " == *" c5 "* && " $MODES " == *" c1_c5 "* ]]; then
  "$PYTHON" "$ANALYZER" --root "$OUTPUT_ROOT" \
    --drs-iterations "$ITERATIONS" --require-complete
fi

echo "Stage-C final benchmarks finished: $OUTPUT_ROOT"