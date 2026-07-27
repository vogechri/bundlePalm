#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-"$WORKSPACE/benchmark_results/drs1723_recovered"}
ITERATIONS=${ITERATIONS:-90}
CLUSTERS=${CLUSTERS:-30}

mkdir -p "$OUTPUT_DIR"

export BUNDLE_PALM_RESULTS_FILE="$OUTPUT_DIR/results.jsonl"
export BUNDLE_PALM_STATE_FILE="$OUTPUT_DIR/state.npz"
export BUNDLE_PALM_EXPLICIT_COST_LANDMARKS=0
export BUNDLE_PALM_STRICT_TRIAL_SAFEGUARD=0
export BUNDLE_PALM_REQUIRE_COMMON_COST_MATCH=1
export BUNDLE_PALM_DRS_SCALING=jacobi_gmean
export BUNDLE_PALM_CLUSTERING=landmark_scalable_stable
export BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01
export BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20
export BUNDLE_PALM_MAX_REFINEMENT_PASSES=3
export BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32
export BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2
export BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1
export BUNDLE_PALM_DRS_MAX_FAILED_NESTEROV_ACCELERATION=3

cd "$SCRIPT_DIR"
exec .venv/bin/python -u client_acc.py \
  http://grail.cs.washington.edu/projects/bal/data/ladybug/ \
  problem-1723-156502-pre.txt.bz2 \
  "$ITERATIONS" "$CLUSTERS"
