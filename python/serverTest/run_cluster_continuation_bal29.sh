#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
CLIENT="$SCRIPT_DIR/client_drs.py"
WORKER=${WORKER:-"$SCRIPT_DIR/build_admm/zeromq_cpp_server_ex"}
PROTO_BUILD=${PROTO_BUILD:-"$SCRIPT_DIR/build_admm"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/cluster_continuation_bal29_i30_i40"}
REQUEST_PORT=${BUNDLE_PALM_REQUEST_PORT:-6816}
RESULT_PORT=${BUNDLE_PALM_RESULT_PORT:-6817}

mapfile -t datasets < <(
  find "$WORKSPACE" -maxdepth 1 -type f -name 'problem-*-pre.txt' \
    -printf '%f\n' | sort -V
)
if [[ ${#datasets[@]} -ne 29 ]]; then
  echo "Expected 29 BAL scenes, found ${#datasets[@]}" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/states" "$OUTPUT_ROOT/logs"
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

for filename in "${datasets[@]}"; do
  scene=$(sed -E 's/^problem-([0-9]+)-.*/\1/' <<<"$filename")
  dataset="$WORKSPACE/$filename"
  state="$OUTPUT_ROOT/states/${scene}_k4_i3.npz"

  run_case "$scene" "$dataset" k4_i3 \
    --clusters 4 --iterations 3 --state "$state"
  run_case "$scene" "$dataset" k4_i3_k24_i27 \
    --clusters 24 --iterations 27 \
    --initial-state "$state" --initial-state-frame canonical
  run_case "$scene" "$dataset" k24_i30 \
    --clusters 24 --iterations 30
  run_case "$scene" "$dataset" k24_i40 \
    --clusters 24 --iterations 40
  run_case "$scene" "$dataset" k4_i3_k24_i87 \
    --clusters 24 --iterations 87 \
    --initial-state "$state" --initial-state-frame canonical
  run_case "$scene" "$dataset" k24_i90 \
    --clusters 24 --iterations 90
  run_case "$scene" "$dataset" k4_i3_k24_i87_s16 \
    --clusters 24 --iterations 87 \
    --initial-state "$state" --initial-state-frame canonical \
    --final-shared-schur-correction \
    --shared-schur-maximum-corrections 16 \
    --shared-schur-operator bsr_low_memory \
    --shared-schur-preconditioner jacobi \
    --shared-schur-linear-solver cg \
    --shared-schur-relative-tolerance 1e-6 \
    --shared-schur-maximum-iterations 500 \
    --shared-schur-minimum-relative-decrease 0.001 \
    --shared-schur-landmark-refinement-steps 3
  run_case "$scene" "$dataset" k24_i90_s16 \
    --clusters 24 --iterations 90 \
    --final-shared-schur-correction \
    --shared-schur-maximum-corrections 16 \
    --shared-schur-operator bsr_low_memory \
    --shared-schur-preconditioner jacobi \
    --shared-schur-linear-solver cg \
    --shared-schur-relative-tolerance 1e-6 \
    --shared-schur-maximum-iterations 500 \
    --shared-schur-minimum-relative-decrease 0.001 \
    --shared-schur-landmark-refinement-steps 3
  run_case "$scene" "$dataset" k4_i3_k24_i197_s16 \
    --clusters 24 --iterations 197 \
    --initial-state "$state" --initial-state-frame canonical \
    --final-shared-schur-correction \
    --shared-schur-maximum-corrections 16 \
    --shared-schur-operator bsr_low_memory \
    --shared-schur-preconditioner jacobi \
    --shared-schur-linear-solver cg \
    --shared-schur-relative-tolerance 1e-6 \
    --shared-schur-maximum-iterations 500 \
    --shared-schur-minimum-relative-decrease 0.001 \
    --shared-schur-landmark-refinement-steps 3
  run_case "$scene" "$dataset" k24_i200_s16 \
    --clusters 24 --iterations 200 \
    --final-shared-schur-correction \
    --shared-schur-maximum-corrections 16 \
    --shared-schur-operator bsr_low_memory \
    --shared-schur-preconditioner jacobi \
    --shared-schur-linear-solver cg \
    --shared-schur-relative-tolerance 1e-6 \
    --shared-schur-maximum-iterations 500 \
    --shared-schur-minimum-relative-decrease 0.001 \
    --shared-schur-landmark-refinement-steps 3
done
