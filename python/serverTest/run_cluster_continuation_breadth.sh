#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
CLIENT="$SCRIPT_DIR/client_drs.py"
WORKER=${WORKER:-"$SCRIPT_DIR/build_admm/zeromq_cpp_server_ex"}
PROTO_BUILD=${PROTO_BUILD:-"$SCRIPT_DIR/build_admm"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/cluster_continuation_breadth_i30"}
DATASET_LIST=${DATASET_LIST:-"$SCRIPT_DIR/1dsfm_all_fifteen_datasets.txt"}
REQUEST_PORT=${BUNDLE_PALM_REQUEST_PORT:-6796}
RESULT_PORT=${BUNDLE_PALM_RESULT_PORT:-6797}

mkdir -p "$OUTPUT_ROOT/states" "$OUTPUT_ROOT/logs"

datasets=()
while IFS='|' read -r scene relative_path || [[ -n "$scene" ]]; do
  [[ -z "$scene" ]] && continue
  datasets+=("$scene|$WORKSPACE/$relative_path")
done < "$DATASET_LIST"
datasets+=(
  "bal1490|$WORKSPACE/problem-1490-935273-pre.txt"
  "bal1778|$WORKSPACE/problem-1778-993923-pre.txt"
  "bal3068|$WORKSPACE/problem-3068-310854-pre.txt"
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
  local result="$OUTPUT_ROOT/${scene}_${arm}.jsonl"
  shift 3
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
  state="$OUTPUT_ROOT/states/${scene}_k4_i3.npz"

  run_case "$scene" "$dataset" k4_i3 \
    --clusters 4 --iterations 3 --state "$state"
  run_case "$scene" "$dataset" k4_i3_k24_i27 \
    --clusters 24 --iterations 27 \
    --initial-state "$state" --initial-state-frame canonical
  run_case "$scene" "$dataset" k24_i30 \
    --clusters 24 --iterations 30
done