#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_drs_failure_top3_live.sh"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
ANALYZER="$SCRIPT_DIR/analyze_mid_schur_transport_gate.py"
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/mid_schur_transport_i10_gate"}
REQUEST_PORT=${REQUEST_PORT:-61620}
OVERWRITE=${OVERWRITE:-0}
ARMS=${ARMS:-"control reset transport"}

run_arm() {
  local arm=$1 family=$2 port_offset=$3
  local correction_iteration=0
  local transport=0
  local -a family_environment

  case "$arm" in
    control) ;;
    reset) correction_iteration=10 ;;
    transport) correction_iteration=10; transport=1 ;;
    *) echo "Unknown arm: $arm" >&2; return 2 ;;
  esac

  if [[ "$family" == "1dsfm" ]]; then
    family_environment=(
      ALL_PROBLEMS=0
      DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_roman_trafalgar_datasets.txt"
      PROBLEM_FILTER="roman_forum trafalgar"
    )
  else
    family_environment=(ALL_PROBLEMS=1 DATASET_LIST_FILE= PROBLEM_FILTER="1490 3068")
  fi

  env \
    "${family_environment[@]}" \
    OUTPUT_DIR="$OUTPUT_ROOT/$arm/$family" \
    VARIANT_TAG="mid_transport_${arm}" \
    BUNDLE_PALM_REQUEST_PORT="$((REQUEST_PORT + port_offset))" \
    BUNDLE_PALM_RESULT_PORT="$((REQUEST_PORT + port_offset + 1))" \
    BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS=1 \
    BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=75 \
    BUNDLE_PALM_SCHUR_PCG_JACOBI_PRECONDITIONER=1 \
    CLUSTERS_LIST=24 ITERATIONS=30 THREADS_PER_CLUSTER=1 \
    LOCAL_STEPS=1 LOCAL_SOLVER=schur_pcg \
    TRUST_REGION_POLICY=daba PERSISTENT_TRUST_REGION=1 \
    SCENE_NORMALIZATION=points_p95 CAMERA_SCALING=jacobi_initial \
    CAMERA_UPDATE=se3_left CLUSTERING=landmark_scalable \
    PROXIMAL_METRIC=block CONSENSUS_METRIC=full \
    SHARED_ONLY_CAMERA_PROXIMAL=1 \
    BLOCK_REGULARIZATION=5e-5 BLOCK_CURVATURE_MULTIPLIER=0 \
    BLOCK_RECOVERY_MODE=regularization \
    WORKER_OWNED_LANDMARKS=1 WORKER_OWNED_CAMERAS=0 \
    PACKED_REQUEST_BUFFERS=1 OUTER_ACCELERATION=none \
    MID_SHARED_SCHUR_CORRECTION_ITERATION="$correction_iteration" \
    MID_SHARED_SCHUR_TRANSPORT_PRODUCT_STATE="$transport" \
    SHARED_SCHUR_OPERATOR=bsr_low_memory \
    SHARED_SCHUR_PRECONDITIONER=jacobi \
    SHARED_SCHUR_RELATIVE_TOLERANCE=1e-6 \
    DEBUG_OUTPUT=1 LIVE_OUTPUT=0 OVERWRITE="$OVERWRITE" \
    CASE_TIMEOUT_SECONDS=14400 \
    "$RUNNER"
}

port_offset=0
for arm in $ARMS; do
  for family in 1dsfm bal; do
    run_arm "$arm" "$family" "$port_offset"
    port_offset=$((port_offset + 2))
  done
done

"$PYTHON" "$ANALYZER" --root "$OUTPUT_ROOT" --arms $ARMS --require-complete

echo "Mid-Schur transport gate finished: $OUTPUT_ROOT"
