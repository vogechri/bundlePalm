#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/k24_landmark_response_parameter_sensitivity"}
REQUEST_PORT=${REQUEST_PORT:-31380}
OVERWRITE=${OVERWRITE:-0}
MEMORY_LIMIT_KB=${MEMORY_LIMIT_KB:-14680064}
PROBLEM_FILTER=${PROBLEM_FILTER:-"madrid_metropolis tower_of_london yorkminster"}

run_arm() {
  local arm=$1 damping=$2 steps=$3 port=$4
  OUTPUT_ROOT="$OUTPUT_ROOT/$arm" \
  FAMILIES=1dsfm \
  ONE_D_SFM_PROBLEM_FILTER="$PROBLEM_FILTER" \
  ANALYZE=0 OVERWRITE="$OVERWRITE" MEMORY_LIMIT_KB="$MEMORY_LIMIT_KB" \
  REQUEST_PORT="$port" VARIANT_NAME_OVERRIDE="$arm" \
  PROPOSAL_ITERATIONS=60 \
  PROPOSAL_CAMERA_DAMPING="$damping" \
  PROPOSAL_LANDMARK_DAMPING="$damping" \
  PROPOSAL_LANDMARK_REFINEMENT_STEPS="$steps" \
  ONE_STEP_SCHUR_RESIDUAL_PROPOSAL_REBASE_TRUST_STATE=1 \
  SCHUR_PROPOSAL_LANDMARK_RESPONSE_ORACLE=1 \
  APPLY_SCHUR_PROPOSAL_LANDMARK_RESPONSE=1 \
  "$SCRIPT_DIR/run_k24_one_step_schur_proposal_breadth.sh"
}

run_arm damping_half 0.0029296875 3 "$REQUEST_PORT"
run_arm base 0.005859375 3 "$((REQUEST_PORT + 10))"
run_arm damping_double 0.01171875 3 "$((REQUEST_PORT + 20))"
run_arm landmarks_1 0.005859375 1 "$((REQUEST_PORT + 30))"
run_arm landmarks_5 0.005859375 5 "$((REQUEST_PORT + 40))"

run_split_arm() {
  local arm=$1 camera_damping=$2 landmark_damping=$3 port=$4
  OUTPUT_ROOT="$OUTPUT_ROOT/$arm" \
  FAMILIES=1dsfm \
  ONE_D_SFM_PROBLEM_FILTER="$PROBLEM_FILTER" \
  ANALYZE=0 OVERWRITE="$OVERWRITE" MEMORY_LIMIT_KB="$MEMORY_LIMIT_KB" \
  REQUEST_PORT="$port" VARIANT_NAME_OVERRIDE="$arm" \
  PROPOSAL_ITERATIONS=60 \
  PROPOSAL_CAMERA_DAMPING="$camera_damping" \
  PROPOSAL_LANDMARK_DAMPING="$landmark_damping" \
  PROPOSAL_LANDMARK_REFINEMENT_STEPS=3 \
  ONE_STEP_SCHUR_RESIDUAL_PROPOSAL_REBASE_TRUST_STATE=1 \
  SCHUR_PROPOSAL_LANDMARK_RESPONSE_ORACLE=1 \
  APPLY_SCHUR_PROPOSAL_LANDMARK_RESPONSE=1 \
  "$SCRIPT_DIR/run_k24_one_step_schur_proposal_breadth.sh"
}

run_split_arm camera_half 0.0029296875 0.005859375 "$((REQUEST_PORT + 50))"
run_split_arm landmark_half 0.005859375 0.0029296875 "$((REQUEST_PORT + 60))"
run_split_arm damping_quarter 0.00146484375 0.00146484375 "$((REQUEST_PORT + 70))"

echo "K24 landmark-response sensitivity finished: $OUTPUT_ROOT"