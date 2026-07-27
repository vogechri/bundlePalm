#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

label=${1:?usage: run_optimization_full.sh LABEL}
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
  "final|problem-394-100368-pre.txt.bz2|394"
  "ladybug|problem-1064-113655-pre.txt.bz2|1064"
  "venice|problem-245-198739-pre.txt.bz2|245"
  "ladybug|problem-1723-156502-pre.txt.bz2|1723"
  "ladybug|problem-1266-132593-pre.txt.bz2|1266"
  "ladybug|problem-931-102699-pre.txt.bz2|931"
  "ladybug|problem-783-84444-pre.txt.bz2|783"
  "venice|problem-89-110973-pre.txt.bz2|89"
  "dubrovnik|problem-287-182023-pre.txt.bz2|287"
  "dubrovnik|problem-142-93602-pre.txt.bz2|142"
  "ladybug|problem-646-73584-pre.txt.bz2|646"
  "dubrovnik|problem-135-90642-pre.txt.bz2|135"
  "venice|problem-52-64053-pre.txt.bz2|52"
  "dubrovnik|problem-173-111908-pre.txt.bz2|173"
  "dubrovnik|problem-356-226730-pre.txt.bz2|356"
  "dubrovnik|problem-88-64298-pre.txt.bz2|88"
  "ladybug|problem-49-7776-pre.txt.bz2|49"
  "trafalgar|problem-126-40037-pre.txt.bz2|126"
  "venice|problem-427-310384-pre.txt.bz2|427"
  "dubrovnik|problem-253-163691-pre.txt.bz2|253"
  "final|problem-961-187103-pre.txt.bz2|961"
  "trafalgar|problem-257-65132-pre.txt.bz2|257"
  "venice|problem-744-543562-pre.txt.bz2|744"
  "venice|problem-951-708276-pre.txt.bz2|951"
  "dubrovnik|problem-308-195089-pre.txt.bz2|308"
  "final|problem-871-527480-pre.txt.bz2|871"
  "venice|problem-1778-993923-pre.txt.bz2|1778"
  "venice|problem-1490-935273-pre.txt.bz2|1490"
  "final|problem-3068-310854-pre.txt.bz2|3068"
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
