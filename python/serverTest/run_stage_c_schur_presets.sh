#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_drs_failure_top3_live.sh"
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/stage_c_schur_presets"}
REQUEST_PORT=${REQUEST_PORT:-56600}
OVERWRITE=${OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}
PRESETS=${PRESETS:-"fast balanced quality"}

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
fi

run_preset() {
  local preset=$1 port=$2 iterations corrections
  case "$preset" in
    fast)
      iterations=30
      corrections=10
      ;;
    balanced)
      iterations=60
      corrections=10
      ;;
    quality)
      iterations=30
      corrections=20
      ;;
    *)
      echo "Unknown preset: $preset" >&2
      exit 2
      ;;
  esac

  local -a command=(
    env
    ALL_PROBLEMS=0
    DATASET_LIST_FILE="$SCRIPT_DIR/1dsfm_all_fifteen_datasets.txt"
    PROBLEM_FILTER="alamo ellis_island gendarmenmarkt madrid_metropolis montreal_notre_dame notre_dame nyc_library piazza_del_popolo piccadilly roman_forum tower_of_london trafalgar union_square vienna_cathedral yorkminster"
    OUTPUT_DIR="$OUTPUT_ROOT/$preset"
    VARIANT_TAG="schur_${preset}"
    BUNDLE_PALM_REQUEST_PORT="$port"
    BUNDLE_PALM_RESULT_PORT="$((port + 1))"
    BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS=1
    BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=75
    BUNDLE_PALM_CAMERA_TRUST_DIAGONAL_SCALE=1e-4
    BUNDLE_PALM_MAXIMUM_TRUST_REGION_RADIUS=1e4
    CLUSTERS_LIST=24
    ITERATIONS="$iterations"
    THREADS_PER_CLUSTER=1
    LOCAL_STEPS=1
    LOCAL_SOLVER=nesterov
    NESTEROV_MAX_ITERATIONS=300
    ENHANCED_INNER_MAX_ITERATIONS=300
    NESTEROV_MIN_ITERATIONS=1
    NESTEROV_STOP_TOLERANCE=1e-2
    ENHANCED_INNER_UNTIL=30
    TRUST_REGION_POLICY=drs
    PERSISTENT_TRUST_REGION=1
    TRUST_REGION_RECOVERY_RATIO=0.5
    SCENE_NORMALIZATION=points_p95
    CAMERA_SCALING=jacobi_initial
    CAMERA_UPDATE=se3_left
    CLUSTERING=landmark_scalable
    PROXIMAL_METRIC=block
    CONSENSUS_METRIC=full
    CONSENSUS_EXECUTION=coordinator
    SHARED_ONLY_CAMERA_PROXIMAL=1
    BLOCK_REGULARIZATION=1e-4
    BLOCK_CURVATURE_MULTIPLIER=0.4
    BLOCK_RECOVERY_MODE=curvature
    MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=64
    CURVATURE_DECAY_AFTER=5
    CURVATURE_DECAY_RATIO=0.5
    METRIC_PROPOSAL_DISAGREEMENT_SCALE=0.5
    OUTER_ACCELERATION=themelis_nesterov
    LINE_SEARCH_GRID=0,1
    ACCELERATION_RESTART_AFTER=3
    SAFEGUARD_MODE=relative
    DRE_RELATIVE_INCREASE=0.01
    MINIMUM_PRIMAL_RATIO=1.001
    SAFEGUARD_ANNEALING_ITERATIONS=30
    WORKER_OWNED_LANDMARKS=1
    WORKER_OWNED_CAMERAS=0
    PACKED_REQUEST_BUFFERS=1
    FINAL_SHARED_SCHUR_CORRECTION=1
    SHARED_SCHUR_MAXIMUM_CORRECTIONS="$corrections"
    SHARED_SCHUR_OPERATOR=bsr_low_memory
    SHARED_SCHUR_PRECONDITIONER=jacobi
    SHARED_SCHUR_MINIMUM_RELATIVE_DECREASE=1e-3
    DEBUG_OUTPUT=0
    LIVE_OUTPUT=0
    OVERWRITE="$OVERWRITE"
    CASE_TIMEOUT_SECONDS=1800
    "$RUNNER"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

port=$REQUEST_PORT
for preset in $PRESETS; do
  run_preset "$preset" "$port"
  port=$((port + 2))
done

echo "Stage-C Schur presets finished: $OUTPUT_ROOT"
