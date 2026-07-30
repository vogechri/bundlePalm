#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_drs_failure_top3_live.sh"
ANALYZER="$SCRIPT_DIR/analyze_drs_29_scene_phase1.py"
PHASE0_ROOT=${PHASE0_ROOT:-"$WORKSPACE/benchmark_results/drs_29_scene_phase0_k30_i90"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/drs_29_scene_phase1_k30_i90"}
ITERATIONS=${ITERATIONS:-90}
CLUSTERS=${CLUSTERS:-30}
OVERWRITE=${OVERWRITE:-0}
LIVE_OUTPUT=${LIVE_OUTPUT:-0}
DEBUG_OUTPUT=${DEBUG_OUTPUT:-0}
REQUEST_PORT=${BUNDLE_PALM_REQUEST_PORT:-18256}
RESULT_PORT=${BUNDLE_PALM_RESULT_PORT:-18257}
SCENE_FILTER=${SCENE_FILTER:-}
VARIANT_FILTER=${VARIANT_FILTER:-}

mapfile -t SCENES < <(
  find "$WORKSPACE" -maxdepth 1 -type f -name 'problem-*-pre.txt' \
    -printf '%f\n' | sort -V | sed -E 's/^problem-([0-9]+)-.*/\1/'
)
if [[ ${#SCENES[@]} -ne 29 ]]; then
  echo "Expected 29 BAL scenes, found ${#SCENES[@]}" >&2
  exit 2
fi
if [[ -z "$SCENE_FILTER" ]]; then
  SCENE_FILTER="${SCENES[*]}"
fi

VARIANTS=(
  consensus_diagonal
  consensus_scalar
  consensus_arithmetic
  scaling_none
  recovery_regularization
  curvature_005
  curvature_02
  trust_persistent_daba
  trust_drs
  local_lmref1
  partition_stable
  solver_schur_pcg
  nesterov_ternary
  final_polish1
)

run_variant() {
  local variant=$1
  local -a overrides=()
  case "$variant" in
    consensus_diagonal) overrides+=(CONSENSUS_METRIC=diagonal) ;;
    consensus_scalar) overrides+=(CONSENSUS_METRIC=scalar) ;;
    consensus_arithmetic) overrides+=(CONSENSUS_METRIC=arithmetic) ;;
    scaling_none) overrides+=(CAMERA_SCALING=none) ;;
    recovery_regularization)
      overrides+=(BLOCK_RECOVERY_MODE=regularization)
      ;;
    curvature_005) overrides+=(BLOCK_CURVATURE_MULTIPLIER=0.05) ;;
    curvature_02) overrides+=(BLOCK_CURVATURE_MULTIPLIER=0.2) ;;
    trust_persistent_daba) overrides+=(PERSISTENT_TRUST_REGION=1) ;;
    trust_drs) overrides+=(TRUST_REGION_POLICY=drs) ;;
    local_lmref1) overrides+=(LANDMARK_REFINEMENT_STEPS=1) ;;
    partition_stable) overrides+=(CLUSTERING=landmark_scalable_stable) ;;
    solver_schur_pcg) overrides+=(LOCAL_SOLVER=schur_pcg) ;;
    nesterov_ternary)
      overrides+=(OUTER_ACCELERATION=nesterov LINE_SEARCH_GRID=0,0.5,1)
      ;;
    final_polish1)
      overrides+=(
        CONSENSUS_LANDMARK_REFINEMENT_STEPS=1
        CONSENSUS_LANDMARK_REFINEMENT_POLICY=final
      )
      ;;
    *)
      echo "Unknown Phase-1 variant: $variant" >&2
      return 2
      ;;
  esac

  echo "===== Phase 1 variant: $variant ====="
  env \
    ALL_PROBLEMS=1 \
    BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
    BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
    BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1 \
    BUNDLE_PALM_MAX_REFINEMENT_PASSES=3 \
    BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32 \
    BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2 \
    BUNDLE_PALM_CLUSTERING=landmark_scalable \
    BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01 \
    BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20 \
    DEBUG_OUTPUT="$DEBUG_OUTPUT" \
    OVERWRITE="$OVERWRITE" \
    LIVE_OUTPUT="$LIVE_OUTPUT" \
    PROBLEM_FILTER="$SCENE_FILTER" \
    CLUSTERS_LIST="$CLUSTERS" \
    ITERATIONS="$ITERATIONS" \
    LOCAL_STEPS=1 \
    LOCAL_SOLVER=nesterov \
    TRUST_REGION_POLICY=daba \
    PERSISTENT_TRUST_REGION=0 \
    TRUST_REGION_RECOVERY_RATIO=0.5 \
    PROXIMAL_METRIC=block \
    CONSENSUS_METRIC=full \
    BLOCK_CURVATURE_MULTIPLIER=0.1 \
    BLOCK_RECOVERY_MODE=curvature \
    MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=64 \
    CURVATURE_DECAY_AFTER=0 \
    BLOCK_REGULARIZATION=5e-5 \
    SAFEGUARD_MODE=relative \
    MINIMUM_PRIMAL_RATIO=1.001 \
    CAMERA_SCALING=jacobi_initial \
    RELAXATION=1.0 \
    METRIC_DIAGNOSTIC_ITERATIONS=0 \
    WORKER_SSE_SHADOW=0 \
    SUPPRESS_ACCELERATED_LANDMARK_REPLIES=0 \
    WORKER_OWNED_LANDMARKS=0 \
    LANDMARK_REFINEMENT_STEPS=0 \
    CONSENSUS_LANDMARK_REFINEMENT_STEPS=0 \
    CONSENSUS_LANDMARK_REFINEMENT_POLICY=safeguard \
    OUTER_ACCELERATION=none \
    LINE_SEARCH_GRID=0,1 \
    ACCELERATION_RESTART_AFTER=3 \
    OUTPUT_DIR="$OUTPUT_ROOT/$variant" \
    "${overrides[@]}" \
    "$RUNNER"
}

mkdir -p "$OUTPUT_ROOT"
for variant in "${VARIANTS[@]}"; do
  if [[ -n "$VARIANT_FILTER" && " $VARIANT_FILTER " != *" $variant "* ]]; then
    continue
  fi
  run_variant "$variant"
  "$SCRIPT_DIR/.venv/bin/python" "$ANALYZER" \
    "$PHASE0_ROOT" "$OUTPUT_ROOT" > /dev/null || true
done

echo "Phase-1 benchmark complete: $OUTPUT_ROOT"
"$SCRIPT_DIR/.venv/bin/python" "$ANALYZER" \
  "$PHASE0_ROOT" "$OUTPUT_ROOT" --report "$OUTPUT_ROOT/report.md"