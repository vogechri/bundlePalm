#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

label=${1:?usage: run_optimization_subset.sh LABEL}
python_bin="$SCRIPT_DIR/.venv/bin/python"

export BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1
export BUNDLE_PALM_MAX_REFINEMENT_PASSES=3
export BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32
export BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2
export BUNDLE_PALM_CLUSTERING=landmark_scalable
export BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01
export BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20
export BUNDLE_PALM_DRS_MAX_FAILED_NESTEROV_ACCELERATION=3
export BUNDLE_PALM_DRS_SCALING=jacobi_gmean

problems=(
  "venice|problem-52-64053-pre.txt.bz2|52"
  "dubrovnik|problem-142-93602-pre.txt.bz2|142"
  "venice|problem-245-198739-pre.txt.bz2|245"
  "dubrovnik|problem-287-182023-pre.txt.bz2|287"
  "final|problem-394-100368-pre.txt.bz2|394"
  "ladybug|problem-646-73584-pre.txt.bz2|646"
  "venice|problem-744-543562-pre.txt.bz2|744"
  "ladybug|problem-931-102699-pre.txt.bz2|931"
  "ladybug|problem-1064-113655-pre.txt.bz2|1064"
  "ladybug|problem-1266-132593-pre.txt.bz2|1266"
)

for entry in "${problems[@]}"; do
  IFS='|' read -r family problem short_name <<< "$entry"
  echo "[$label] $problem"
  "$python_bin" -u client_acc.py \
    "http://grail.cs.washington.edu/projects/bal/data/$family/" \
    "$problem" 90 30 \
    > "/tmp/${label}_${short_name}.out" \
    2> "/tmp/${label}_${short_name}.err"
done