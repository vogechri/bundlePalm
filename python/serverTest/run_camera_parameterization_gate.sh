#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_drs_failure_top3_live.sh"
ANALYZER="$SCRIPT_DIR/analyze_camera_parameterization_gate.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/camera_parameterization_direct_tangent_k24_i30_gate"}
ITERATIONS=${ITERATIONS:-30}
REQUEST_PORT=${REQUEST_PORT:-61820}
OVERWRITE=${OVERWRITE:-0}
MODES=${MODES:-"se3_left so3_left se3_right"}
COHORT=${COHORT:-sentinel}
SCHUR_PCG_MAX_ITERATIONS=${SCHUR_PCG_MAX_ITERATIONS:-400}

run_mode() {
  local mode=$1 family=$2 port_offset=$3
  local -a family_environment
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
    elif [[ "$COHORT" == "confirmation" ]]; then
      family_environment=(
        ALL_PROBLEMS=0
        DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_heldout_nine_datasets.txt"
        PROBLEM_FILTER="$(cut -d'|' -f1 "$SCRIPT_DIR/1dsfm_heldout_nine_datasets.txt" | tr '\n' ' ')"
      )
    else
      echo "Unknown cohort: $COHORT" >&2
      return 2
    fi
  else
    if [[ "$COHORT" == "sentinel" ]]; then
      family_environment=(ALL_PROBLEMS=1 DATASET_LIST_FILE= PROBLEM_FILTER="1490 3068")
    elif [[ "$COHORT" == "development" ]]; then
      family_environment=(
        ALL_PROBLEMS=1
        DATASET_LIST_FILE=
        PROBLEM_FILTER="$(cut -d'|' -f1 "$SCRIPT_DIR/bal_tuning_sentinel_five_datasets.txt" | tr '\n' ' ')"
      )
    elif [[ "$COHORT" == "confirmation" ]]; then
      family_environment=(
        ALL_PROBLEMS=1
        DATASET_LIST_FILE=
        PROBLEM_FILTER="$(${PYTHON} - "$WORKSPACE" <<'PY'
import re
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
identifiers = []
for path in workspace.glob("problem-*-pre.txt"):
    match = re.match(r"problem-(\d+)-", path.name)
    if match:
        identifiers.append(int(match.group(1)))
print(" ".join(str(identifier) for identifier in sorted(identifiers)))
PY
)"
      )
    else
      echo "Unknown cohort: $COHORT" >&2
      return 2
    fi
  fi

  env \
    "${family_environment[@]}" \
    OUTPUT_DIR="$OUTPUT_ROOT/$mode/$family" \
    VARIANT_TAG="camera_parameterization_${mode}" \
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
    BUNDLE_PALM_SCHUR_PCG_MAX_ITERATIONS="$SCHUR_PCG_MAX_ITERATIONS" \
    BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=25 \
    CLUSTERS_LIST=24 ITERATIONS="$ITERATIONS" THREADS_PER_CLUSTER=1 \
    LOCAL_STEPS=1 LOCAL_SOLVER=schur_pcg \
    TRUST_REGION_POLICY=daba PERSISTENT_TRUST_REGION=1 \
    SCENE_NORMALIZATION=points_p95 CAMERA_SCALING=jacobi_initial \
    CAMERA_UPDATE="$mode" CLUSTERING=landmark_scalable \
    OUTER_ACCELERATION=themelis_nesterov LINE_SEARCH_GRID=0,1 \
    ACCELERATION_RESTART_AFTER=3 \
    ADAPTIVE_LOCAL_DEPTH=1 ADAPTIVE_LOCAL_DEPTH_START=5 \
    ADAPTIVE_LOCAL_DEPTH_MAXIMUM=2 ADAPTIVE_LOCAL_DEPTH_HIGH=0.35 \
    ADAPTIVE_LOCAL_DEPTH_LOW=0.20 ADAPTIVE_LOCAL_DEPTH_WINDOW=3 \
    ADAPTIVE_LOCAL_DEPTH_DWELL=3 \
    PROXIMAL_METRIC=block CONSENSUS_METRIC=full \
    SHARED_ONLY_CAMERA_PROXIMAL=1 \
    BLOCK_REGULARIZATION=5e-5 BLOCK_CURVATURE_MULTIPLIER=0 \
    MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=16 \
    BLOCK_RECOVERY_MODE=regularization \
    SAFEGUARD_MODE=relative DRE_RELATIVE_INCREASE=0.01 \
    MINIMUM_PRIMAL_RATIO=1.001 \
    WORKER_OWNED_LANDMARKS=1 WORKER_OWNED_CAMERAS=1 \
    PACKED_REQUEST_BUFFERS=1 DEBUG_OUTPUT=1 LIVE_OUTPUT=0 \
    OVERWRITE="$OVERWRITE" CASE_TIMEOUT_SECONDS=14400 \
    "$RUNNER"
}

port_offset=0
for mode in $MODES; do
  for family in 1dsfm bal; do
    run_mode "$mode" "$family" "$port_offset"
    port_offset=$((port_offset + 2))
  done
done

"$PYTHON" "$ANALYZER" \
  --root "$OUTPUT_ROOT" --modes $MODES --cohort "$COHORT" \
  --iterations "$ITERATIONS" --require-complete

echo "Camera parameterization gate finished: $OUTPUT_ROOT"
