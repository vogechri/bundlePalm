#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON="$SCRIPT_DIR/.venv/bin/python"
CLIENT="$SCRIPT_DIR/client_acc.py"
SERVER="$SCRIPT_DIR/build/zeromq_cpp_server_ex"
ITERATIONS=${ITERATIONS:-1000}
CLUSTERS_LIST=${CLUSTERS_LIST:-"4 8 16 32"}
PROBLEM_FILTER=${PROBLEM_FILTER:-}
CLUSTERING_MODE=${CLUSTERING_MODE:-landmark_scalable}
OUTPUT_DIR=${OUTPUT_DIR:-"$WORKSPACE/benchmark_results/daba_protocol_l2"}
RESULTS_FILE="$OUTPUT_DIR/results.jsonl"
STATUS_FILE="$OUTPUT_DIR/status.tsv"
SERVER_PID=""

PROBLEMS=(
  "ladybug|problem-1723-156502-pre.txt.bz2|1723"
  "venice|problem-1778-993923-pre.txt.bz2|1778"
  "final|problem-13682-4456117-pre.txt.bz2|13682"
)

COMMON_ENV=(
  "BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1"
  "BUNDLE_PALM_MAX_REFINEMENT_PASSES=3"
  "BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32"
  "BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2"
  "BUNDLE_PALM_CLUSTERING=$CLUSTERING_MODE"
  "BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01"
  "BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20"
  "BUNDLE_PALM_DRS_SCALING=jacobi_gmean"
  "BUNDLE_PALM_PARTITION_TRACE=0"
  "BUNDLE_PALM_PRINT_SELECTED_CAMERAS=0"
)

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/states" "$OUTPUT_DIR/memory"
if [[ ! -f "$STATUS_FILE" ]]; then
  printf 'dataset\tclusters\titerations\tstatus\texit_code\telapsed_seconds\tcoordinator_max_rss_kb\tworker_max_rss_kb\n' > "$STATUS_FILE"
fi

cleanup_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    mapfile -t worker_children < <(pgrep -P "$SERVER_PID" || true)
    if [[ ${#worker_children[@]} -gt 0 ]]; then
      kill "${worker_children[@]}" 2>/dev/null || true
    else
      kill "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

completed() {
  "$PYTHON" - "$RESULTS_FILE" "$1" "$2" "$ITERATIONS" <<'PY'
import json
import sys

path, file_name, clusters, iterations = sys.argv[1:]
try:
    rows = open(path, encoding="utf-8")
except FileNotFoundError:
    raise SystemExit(1)
for line in rows:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if (row.get("file_name", "").endswith(file_name)
            and row.get("kClusters") == int(clusters)
            and row.get("iterations") == int(iterations)
            and row.get("status") == "completed"):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

read_max_rss() {
  awk -F: '/Maximum resident set size/ {gsub(/^[[:space:]]+/, "", $2); print $2}' "$1"
}

run_case() {
  local family=$1 file_name=$2 short_name=$3 clusters=$4
  local label="${short_name}_k${clusters}_i${ITERATIONS}"
  local log_file="$OUTPUT_DIR/logs/${label}.log"
  local server_log="$OUTPUT_DIR/logs/${label}_worker.log"
  local coordinator_time="$OUTPUT_DIR/memory/${label}_coordinator.time"
  local worker_time="$OUTPUT_DIR/memory/${label}_worker.time"
  local state_file="$OUTPUT_DIR/states/${label}.npz"
  local start_seconds=$SECONDS exit_code status coordinator_rss worker_rss

  cleanup_server
  (cd "$WORKSPACE" && exec /usr/bin/time -v -o "$worker_time" "$SERVER") \
    > "$server_log" 2>&1 &
  SERVER_PID=$!
  for _ in {1..50}; do
    grep -q "Starting the server" "$server_log" 2>/dev/null && break
    kill -0 "$SERVER_PID" 2>/dev/null || break
    read -r -t 0.1 _ || true
  done
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf '%s\t%s\t%s\tserver_failed\t1\t0\t\t\n' \
      "$file_name" "$clusters" "$ITERATIONS" >> "$STATUS_FILE"
    return 1
  fi

  set +e
  (cd "$SCRIPT_DIR" && /usr/bin/time -v -o "$coordinator_time" \
    env "${COMMON_ENV[@]}" \
      "BUNDLE_PALM_RESULTS_FILE=$RESULTS_FILE" \
      "BUNDLE_PALM_STATE_FILE=$state_file" \
      "$PYTHON" -u "$CLIENT" \
      "http://grail.cs.washington.edu/projects/bal/data/$family/" \
      "../$file_name" "$ITERATIONS" "$clusters") > "$log_file" 2>&1
  exit_code=$?
  cleanup_server

  status=failed
  [[ $exit_code -eq 0 ]] && status=completed
  coordinator_rss=$(read_max_rss "$coordinator_time")
  worker_rss=$(read_max_rss "$worker_time")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$file_name" "$clusters" "$ITERATIONS" "$status" "$exit_code" \
    "$((SECONDS - start_seconds))" "$coordinator_rss" "$worker_rss" \
    >> "$STATUS_FILE"
}

for problem_spec in "${PROBLEMS[@]}"; do
  IFS='|' read -r family file_name short_name <<< "$problem_spec"
  if [[ -n "$PROBLEM_FILTER" && "$short_name" != "$PROBLEM_FILTER" ]]; then
    continue
  fi
  if [[ ! -f "$WORKSPACE/$file_name" ]]; then
    echo "Missing dataset $WORKSPACE/$file_name" >&2
    exit 1
  fi
  for clusters in $CLUSTERS_LIST; do
    if completed "$file_name" "$clusters"; then
      echo "Skipping completed $file_name at K=$clusters"
      continue
    fi
    echo "Running L2 $file_name at K=$clusters for $ITERATIONS iterations"
    run_case "$family" "$file_name" "$short_name" "$clusters" || true
  done
done

echo "DABA-protocol L2 matrix finished: $STATUS_FILE"
