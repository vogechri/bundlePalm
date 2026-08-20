#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
BRIDGE_CLIENT=${BRIDGE_CLIENT:-"$SCRIPT_DIR/client_drs_k1_bridge.py"}
BASE_CLIENT=${BASE_CLIENT:-"$SCRIPT_DIR/client_drs.py"}
PROTO_BUILD=${PROTO_BUILD:-"$SCRIPT_DIR/build_admm"}
WORKER=${WORKER:-"$SCRIPT_DIR/build_admm/zeromq_cpp_server_ex"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/k1_inner_on_mature_base_k24_i30"}
ITERATIONS=${ITERATIONS:-30}
ARMS=${ARMS:-"legacy direct direct_shared direct_shared_diag direct_shared_l2"}
SCENES=${SCENES:-"roman_forum trafalgar"}
REQUEST_PORT=${REQUEST_PORT:-30420}
RESULT_PORT=${RESULT_PORT:-30421}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-7200}

declare -A DATASETS=(
  [gendarmenmarkt]="$WORKSPACE/benchmark_results/sfm_init_1dsfm/gendarmenmarkt/gendarmenmarkt_sfm_init_derived.txt"
  [piccadilly]="$WORKSPACE/benchmark_results/sfm_init_1dsfm/piccadilly/piccadilly_sfm_init_derived.txt"
  [roman_forum]="$WORKSPACE/benchmark_results/sfm_init_1dsfm/roman_forum/roman_forum_sfm_init_derived.txt"
  [trafalgar]="$WORKSPACE/benchmark_results/sfm_init_1dsfm/trafalgar/trafalgar_sfm_init_derived.txt"
  [union_square]="$WORKSPACE/benchmark_results/sfm_init_1dsfm/union_square/union_square_sfm_init_derived.txt"
  [vienna_cathedral]="$WORKSPACE/benchmark_results/sfm_init_1dsfm/vienna_cathedral/vienna_cathedral_sfm_init_derived.txt"
  [bal1490]="$WORKSPACE/problem-1490-935273-pre.txt"
  [bal3068]="$WORKSPACE/problem-3068-310854-pre.txt"
)

WORKER_PID=""
cleanup_worker() {
  if [[ -n "$WORKER_PID" ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
    kill -TERM "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi
  WORKER_PID=""
}
trap cleanup_worker EXIT INT TERM

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/states"

for arm in $ARMS; do
  case "$arm" in
    legacy) direct_tangent=0 ;;
    direct|direct_shared|direct_shared_diag|direct_shared_l2) direct_tangent=1 ;;
    *) echo "Unknown arm: $arm" >&2; exit 2 ;;
  esac
  result_file="$OUTPUT_ROOT/${arm}.jsonl"
  for scene in $SCENES; do
    dataset=${DATASETS[$scene]:-}
    if [[ -z "$dataset" || ! -f "$dataset" ]]; then
      echo "Dataset unavailable for $scene: $dataset" >&2
      exit 2
    fi
    if "$PYTHON" - "$result_file" "$dataset" "$ITERATIONS" <<'PY'
import json, pathlib, sys
path, dataset, iterations = sys.argv[1:]
try:
    rows = [json.loads(line) for line in open(path) if line.strip()]
except FileNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if any(
    pathlib.Path(row.get("dataset", "")).resolve() == pathlib.Path(dataset).resolve()
    and row.get("completedIterations") == int(iterations)
    for row in rows
) else 1)
PY
    then
      echo "Skipping completed $arm/$scene"
      continue
    fi

    cleanup_worker
    worker_log="$OUTPUT_ROOT/logs/${arm}_${scene}_worker.log"
    env \
      BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
      BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
      BUNDLE_PALM_THREADS_PER_CLUSTER=1 \
      BUNDLE_PALM_CAMERA_UPDATE=se3_left \
      BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS="$direct_tangent" \
      BUNDLE_PALM_DIAGONAL_TRUST_DAMPING=0 \
      BUNDLE_PALM_DISABLE_LANDMARK_PRECONDITIONING=0 \
      BUNDLE_PALM_BAE_TRUST_SCHEDULE=0 \
      BUNDLE_PALM_CUMULATIVE_DIAGONAL_DAMPING=0 \
      BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=75 \
      "$WORKER" >"$worker_log" 2>&1 &
    WORKER_PID=$!
    for _ in {1..600}; do
      grep -q "Starting the server" "$worker_log" 2>/dev/null && break
      kill -0 "$WORKER_PID" 2>/dev/null || break
      read -r -t 0.1 _ || true
    done
    kill -0 "$WORKER_PID" 2>/dev/null || {
      echo "Worker failed for $arm/$scene" >&2
      exit 1
    }

    echo "Running $arm/$scene at K24/I${ITERATIONS}"
    (cd "$SCRIPT_DIR" && env \
        BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
        BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
        BUNDLE_PALM_PROTO_BUILD="$PROTO_BUILD" \
        BUNDLE_PALM_BASE_CLIENT_DRS="$BASE_CLIENT" \
        timeout --foreground --signal=TERM --kill-after=30 "$CASE_TIMEOUT_SECONDS" \
        "$PYTHON" "$BRIDGE_CLIENT" --bridge-arm "$arm" "$dataset" \
          --iterations "$ITERATIONS" --clusters 24 \
          --threads-per-cluster 1 --clustering landmark_scalable \
          --partition-cache auto --worker-owned-landmarks --packed-request-buffers \
          --results "$result_file" \
          --state "$OUTPUT_ROOT/states/${arm}_${scene}.npz" \
          --variant-name "k1_bridge_${arm}" --debug-output) \
      >"$OUTPUT_ROOT/logs/${arm}_${scene}.log" 2>&1
    cleanup_worker
  done
done

echo "K1 inner bridge gate finished: $OUTPUT_ROOT"