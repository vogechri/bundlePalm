#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
PHASE1_RUNNER="$SCRIPT_DIR/run_drs_29_scene_phase1.sh"
OUTPUT_ROOT="$WORKSPACE/benchmark_results/drs_29_scene_phase1_k30_i90"
REQUEST_PORT=${BUNDLE_PALM_REQUEST_PORT:-18856}
RESULT_PORT=${BUNDLE_PALM_RESULT_PORT:-18857}

repair_variant() {
  local variant=$1
  local scenes=$2
  echo "===== Repairing $variant: $scenes ====="
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  SCENE_FILTER="$scenes" \
  VARIANT_FILTER="$variant" \
  BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
  BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
  OVERWRITE=1 \
  LIVE_OUTPUT=0 \
  DEBUG_OUTPUT=0 \
  "$PHASE1_RUNNER"
}

repair_variant consensus_arithmetic "52 88 135 253 257 427 744"
repair_variant consensus_scalar "1778"
repair_variant curvature_005 "744 871 1723 1778"
repair_variant curvature_02 "1490"
repair_variant final_polish1 "253 287 308 394 427 744 783 871"
repair_variant local_lmref1 "287 308 394 427 951 1723 3068"
repair_variant nesterov_ternary "427"
repair_variant partition_stable "142 427 744 1064"
repair_variant recovery_regularization "245 257 287 308 1723"
repair_variant scaling_none "951 961 1064"
repair_variant solver_schur_pcg "88 89 253 308 356 427 783 951"
repair_variant trust_drs "427 951 961 1064"
repair_variant trust_persistent_daba "142 173 245 287 783"

echo "Phase-1 state repair complete"
