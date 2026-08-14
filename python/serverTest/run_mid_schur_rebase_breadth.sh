#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
CLIENT="$SCRIPT_DIR/client_drs.py"
WORKER=${WORKER:-"$SCRIPT_DIR/build_admm/zeromq_cpp_server_ex"}
PROTO_BUILD=${PROTO_BUILD:-"$SCRIPT_DIR/build_admm"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/mid_schur_rebase_i10_breadth"}
DATASET_LIST=${DATASET_LIST:-"$SCRIPT_DIR/1dsfm_all_fifteen_datasets.txt"}
REQUEST_PORT=${BUNDLE_PALM_REQUEST_PORT:-6836}
RESULT_PORT=${BUNDLE_PALM_RESULT_PORT:-6837}

mkdir -p "$OUTPUT_ROOT/logs"
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
  --iterations 30
  --clusters 24
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
  --mid-shared-schur-correction-iteration 10
  --shared-schur-camera-damping 3
  --shared-schur-landmark-damping 3
  --shared-schur-linear-solver cg
  --shared-schur-relative-tolerance 1e-6
  --shared-schur-maximum-iterations 500
  --shared-schur-operator bsr_low_memory
  --shared-schur-preconditioner jacobi
  --shared-schur-landmark-refinement-steps 3
)

for entry in "${datasets[@]}"; do
  scene=${entry%%|*}
  dataset=${entry#*|}
  result="$OUTPUT_ROOT/${scene}_mid_i10.jsonl"
  if [[ -s "$result" ]]; then
    echo "skip $scene"
    continue
  fi
  echo "run $scene"
  "$PYTHON" "$CLIENT" "$dataset" "${common[@]}" \
    --results "$result" --variant-name mid_i10 \
    >"$OUTPUT_ROOT/logs/${scene}.log" 2>&1
done
