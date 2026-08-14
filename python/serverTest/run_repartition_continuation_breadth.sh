#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
CLIENT="$SCRIPT_DIR/client_drs.py"
WORKER=${WORKER:-"$SCRIPT_DIR/build_admm/zeromq_cpp_server_ex"}
PROTO_BUILD=${PROTO_BUILD:-"$SCRIPT_DIR/build_admm"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/repartition_continuation_i15_i15_breadth"}
DATASET_LIST=${DATASET_LIST:-"$SCRIPT_DIR/1dsfm_all_fifteen_datasets.txt"}
REQUEST_PORT=${BUNDLE_PALM_REQUEST_PORT:-6866}
RESULT_PORT=${BUNDLE_PALM_RESULT_PORT:-6867}

mkdir -p "$OUTPUT_ROOT/states" "$OUTPUT_ROOT/logs"
datasets=()
while IFS='|' read -r scene relative_path || [[ -n "$scene" ]]; do
  [[ -z "$scene" ]] && continue
  datasets+=("1dsfm_${scene}|$WORKSPACE/$relative_path")
done < "$DATASET_LIST"
while IFS= read -r filename; do
  scene=$(sed -E 's/^problem-([0-9]+)-.*/\1/' <<<"$filename")
  datasets+=("bal_${scene}|$WORKSPACE/$filename")
done < <(
  find "$WORKSPACE" -maxdepth 1 -type f -name 'problem-*-pre.txt' \
    -printf '%f\n' | sort -V
)

export BUNDLE_PALM_PROTO_BUILD="$PROTO_BUILD"
export BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT"
export BUNDLE_PALM_RESULT_PORT="$RESULT_PORT"
export BUNDLE_PALM_CAMERA_UPDATE=se3_left
export BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS=1
export BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=75
export BUNDLE_PALM_SCHUR_PCG_JACOBI_PRECONDITIONER=1
export PYTHONPATH="$PROTO_BUILD/generated:$PROTO_BUILD/generated/proto:${PYTHONPATH:-}"

"$WORKER" >"$OUTPUT_ROOT/logs/worker.log" 2>&1 &
worker_pid=$!
trap 'kill "$worker_pid" 2>/dev/null || true' EXIT

common=(
  --local-steps 1
  --local-solver schur_pcg
  --trust-region-policy daba
  --persistent-trust-region
  --camera-scaling jacobi_initial
  --scene-normalization points_p95
  --proximal-metric block
  --consensus-metric full
  --shared-only-camera-proximal
  --block-curvature-multiplier 0
  --block-recovery-mode regularization
  --worker-owned-landmarks
  --packed-request-buffers
  --partition-cache auto
  --outer-acceleration none
)

run_case() {
  local scene=$1 dataset=$2 arm=$3
  shift 3
  local result="$OUTPUT_ROOT/${scene}_${arm}.jsonl"
  if [[ -s "$result" ]]; then
    echo "skip $scene $arm"
    return
  fi
  echo "run $scene $arm"
  "$PYTHON" "$CLIENT" "$dataset" "${common[@]}" "$@" \
    --results "$result" --variant-name "$arm" \
    >"$OUTPUT_ROOT/logs/${scene}_${arm}.log" 2>&1
}

for entry in "${datasets[@]}"; do
  scene=${entry%%|*}
  dataset=${entry#*|}
  state="$OUTPUT_ROOT/states/${scene}_default_i15.npz"

  run_case "$scene" "$dataset" stage_a \
    --clusters 24 --iterations 15 --clustering landmark_scalable \
    --state "$state"
  run_case "$scene" "$dataset" restart \
    --clusters 24 --iterations 15 --clustering landmark_scalable \
    --initial-state "$state" --initial-state-frame canonical
  run_case "$scene" "$dataset" repartition \
    --clusters 24 --iterations 15 --clustering landmark_scalable_stable \
    --initial-state "$state" --initial-state-frame canonical
done
