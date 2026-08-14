#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_drs_failure_top3_live.sh"
ANALYZER="$SCRIPT_DIR/analyze_drs_parameter_sensitivity.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/drs_parameter_sensitivity_k24_i30_gate"}
ITERATIONS=${ITERATIONS:-30}
REQUEST_PORT=${REQUEST_PORT:-61220}
OVERWRITE=${OVERWRITE:-0}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-14400}
ARMS=${ARMS:-"control block_2p5em5 block_1em4 restart_1 restart_5 cap_3 exponent_2 exponent_8 dre_0p005 dre_0p02"}
COHORT=${COHORT:-sentinel}

run_arm() {
  local arm=$1 family=$2 port_offset=$3
  local block_regularization=5e-5
  local restart_after=3
  local maximum_step_ratio=10
  local annealing_exponent=4
  local dre_relative_increase=0.01
  local -a family_environment

  case "$arm" in
    control) ;;
    block_2p5em5) block_regularization=2.5e-5 ;;
    block_1em4) block_regularization=1e-4 ;;
    restart_1) restart_after=1 ;;
    restart_5) restart_after=5 ;;
    cap_3) maximum_step_ratio=3 ;;
    exponent_2) annealing_exponent=2 ;;
    exponent_8) annealing_exponent=8 ;;
    dre_0p005) dre_relative_increase=0.005 ;;
    dre_0p02) dre_relative_increase=0.02 ;;
    *) echo "Unknown parameter arm: $arm" >&2; return 2 ;;
  esac

  if [[ "$family" == "1dsfm" ]]; then
    if [[ "$COHORT" == "sentinel" ]]; then
      family_environment=(
        ALL_PROBLEMS=0
        DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_roman_trafalgar_datasets.txt"
        PROBLEM_FILTER="roman_forum trafalgar"
      )
    elif [[ "$COHORT" == "development" ]]; then
      family_environment=(
        ALL_PROBLEMS=0
        DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_six_datasets.txt"
        PROBLEM_FILTER="$(cut -d'|' -f1 "$SCRIPT_DIR/1dsfm_six_datasets.txt" | tr '\n' ' ')"
      )
    else
      echo "Unknown cohort: $COHORT" >&2
      return 2
    fi
  elif [[ "$family" == "bal" ]]; then
    if [[ "$COHORT" == "sentinel" ]]; then
      family_environment=(ALL_PROBLEMS=1 DATASET_LIST_FILE= PROBLEM_FILTER="1490 3068")
    elif [[ "$COHORT" == "development" ]]; then
      family_environment=(
        ALL_PROBLEMS=1
        DATASET_LIST_FILE=
        PROBLEM_FILTER="$(cut -d'|' -f1 "$SCRIPT_DIR/bal_tuning_sentinel_five_datasets.txt" | tr '\n' ' ')"
      )
    else
      echo "Unknown cohort: $COHORT" >&2
      return 2
    fi
  else
    echo "Unknown family: $family" >&2
    return 2
  fi

  env \
    "${family_environment[@]}" \
    OUTPUT_DIR="$OUTPUT_ROOT/$arm/$family" \
    VARIANT_TAG="parameter_${arm}" \
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
    BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=25 \
    BUNDLE_PALM_ACCEL_MAX_STEP_RATIO="$maximum_step_ratio" \
    CLUSTERS_LIST=24 ITERATIONS="$ITERATIONS" THREADS_PER_CLUSTER=1 \
    LOCAL_STEPS=1 LOCAL_SOLVER=schur_pcg \
    TRUST_REGION_POLICY=daba PERSISTENT_TRUST_REGION=1 \
    SCENE_NORMALIZATION=points_p95 CAMERA_SCALING=jacobi_initial \
    CAMERA_UPDATE=se3_left CLUSTERING=landmark_scalable \
    OUTER_ACCELERATION=themelis_nesterov LINE_SEARCH_GRID=0,1 \
    ACCELERATION_RESTART_AFTER="$restart_after" \
    ADAPTIVE_LOCAL_DEPTH=1 ADAPTIVE_LOCAL_DEPTH_START=5 \
    ADAPTIVE_LOCAL_DEPTH_MAXIMUM=2 ADAPTIVE_LOCAL_DEPTH_HIGH=0.35 \
    ADAPTIVE_LOCAL_DEPTH_LOW=0.20 ADAPTIVE_LOCAL_DEPTH_WINDOW=3 \
    ADAPTIVE_LOCAL_DEPTH_DWELL=3 \
    PROXIMAL_METRIC=block CONSENSUS_METRIC=full \
    SHARED_ONLY_CAMERA_PROXIMAL=1 \
    BLOCK_REGULARIZATION="$block_regularization" \
    BLOCK_CURVATURE_MULTIPLIER=0 MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=16 \
    BLOCK_RECOVERY_MODE=regularization \
    SAFEGUARD_MODE=relative \
    DRE_RELATIVE_INCREASE="$dre_relative_increase" \
    MINIMUM_PRIMAL_RATIO=1.001 \
    SAFEGUARD_ANNEALING_ITERATIONS="$ITERATIONS" \
    SAFEGUARD_REFERENCE_ITERATION=5 \
    SAFEGUARD_ANNEALING_EXPONENT="$annealing_exponent" \
    WORKER_OWNED_LANDMARKS=1 WORKER_OWNED_CAMERAS=1 \
    PACKED_REQUEST_BUFFERS=1 DEBUG_OUTPUT=1 LIVE_OUTPUT=0 \
    OVERWRITE="$OVERWRITE" CASE_TIMEOUT_SECONDS="$CASE_TIMEOUT_SECONDS" \
    "$RUNNER"
}

port_offset=0
for arm in $ARMS; do
  for family in 1dsfm bal; do
    run_arm "$arm" "$family" "$port_offset"
    port_offset=$((port_offset + 2))
  done
done

"$PYTHON" "$ANALYZER" \
  --root "$OUTPUT_ROOT" --arms $ARMS --iterations "$ITERATIONS" \
  --cohort "$COHORT" --require-complete

echo "DRS parameter sensitivity sweep finished: $OUTPUT_ROOT"
