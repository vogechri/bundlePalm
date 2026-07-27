#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON="$SCRIPT_DIR/.venv/bin/python"
CLIENT="$SCRIPT_DIR/client_admm.py"
WORKER=${WORKER:-"$SCRIPT_DIR/build_admm/zeromq_cpp_server_ex"}
PROTO_BUILD=${PROTO_BUILD:-"$SCRIPT_DIR/build_admm"}
MANIFEST=${MANIFEST:-"$WORKSPACE/benchmark_results/admm_innovation_manifest.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"$WORKSPACE/benchmark_results/admm_innovation_matrix"}
ITERATIONS=${ITERATIONS:-100}
LOCAL_STEPS=${LOCAL_STEPS:-20}
THREADS_PER_CLUSTER=${THREADS_PER_CLUSTER:-1}
CLUSTERS_LIST=${CLUSTERS_LIST:-"5 15 30"}
MODE_LIST=${MODE_LIST:-"baseline"}
REQUEST_PORT=${BUNDLE_PALM_REQUEST_PORT:-6556}
RESULT_PORT=${BUNDLE_PALM_RESULT_PORT:-6557}
PROBLEM_FILTER=${PROBLEM_FILTER:-}
ALL_PROBLEMS=${ALL_PROBLEMS:-0}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-3600}
LIVE_OUTPUT=${LIVE_OUTPUT:-0}
DEBUG_OUTPUT=${DEBUG_OUTPUT:-0}
OVERWRITE=${OVERWRITE:-0}
STATUS_FILE="$OUTPUT_DIR/status.tsv"
WORKER_PID=""

if ! [[ "$THREADS_PER_CLUSTER" =~ ^[1-9][0-9]*$ ]]; then
  echo "THREADS_PER_CLUSTER must be a positive integer" >&2
  exit 2
fi
if ! [[ "$CASE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CASE_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ "$LIVE_OUTPUT" != "0" && "$LIVE_OUTPUT" != "1" ]]; then
  echo "LIVE_OUTPUT must be 0 or 1" >&2
  exit 2
fi
if [[ "$DEBUG_OUTPUT" != "0" && "$DEBUG_OUTPUT" != "1" ]]; then
  echo "DEBUG_OUTPUT must be 0 or 1" >&2
  exit 2
fi
if [[ "$OVERWRITE" != "0" && "$OVERWRITE" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1" >&2
  exit 2
fi

PROBLEMS=(
  "problem-1723-156502-pre.txt|1723"
  "problem-52-64053-pre.txt|52"
  "problem-245-198739-pre.txt|245"
  "problem-394-100368-pre.txt|394"
  "problem-871-527480-pre.txt|871"
  "problem-3068-310854-pre.txt|3068"
  "problem-744-543562-pre.txt|744"
  "problem-1266-132593-pre.txt|1266"
  "problem-1064-113655-pre.txt|1064"
  "problem-931-102699-pre.txt|931"
)

if [[ "$ALL_PROBLEMS" == "1" ]]; then
  PROBLEMS=()
  while IFS= read -r dataset; do
    short_name=${dataset#problem-}
    short_name=${short_name%%-*}
    PROBLEMS+=("$dataset|$short_name")
  done < <(
    find "$WORKSPACE" -maxdepth 1 -type f -name 'problem-*-pre.txt' \
      -printf '%f\n' | sort -V
  )
  if [[ ${#PROBLEMS[@]} -ne 29 ]]; then
    echo "Expected 29 top-level BAL problems, found ${#PROBLEMS[@]}" >&2
    exit 2
  fi
fi

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/states" "$OUTPUT_DIR/memory"
if [[ ! -f "$STATUS_FILE" ]]; then
  printf 'variant\tdataset\tclusters\titerations\tlocal_steps\tthreads_per_cluster\tstatus\texit_code\telapsed_seconds\tcoordinator_max_rss_kb\tworker_max_rss_kb\n' > "$STATUS_FILE"
fi

cleanup_worker() {
  if [[ -n "$WORKER_PID" ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
    mapfile -t worker_children < <(pgrep -P "$WORKER_PID" || true)
    if [[ ${#worker_children[@]} -gt 0 ]]; then
      kill "${worker_children[@]}" 2>/dev/null || true
    else
      kill "$WORKER_PID" 2>/dev/null || true
    fi
    wait "$WORKER_PID" 2>/dev/null || true
  fi
  WORKER_PID=""
}
trap cleanup_worker EXIT INT TERM

manifest_status() {
  "$PYTHON" - "$MANIFEST" "$1" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
mode = manifest.get(sys.argv[2])
if mode is None:
    raise SystemExit(2)
print(mode["status"])
PY
}

mapfile -t mode_arguments < /dev/null
load_mode_arguments() {
  mapfile -t mode_arguments < <("$PYTHON" - "$MANIFEST" "$1" <<'PY'
import json, sys
mode = json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]
for argument in mode.get("arguments", []):
    print(argument)
PY
  )
}

completed() {
  local result_file=$1 variant=$2 dataset=$3 clusters=$4
  "$PYTHON" - "$result_file" "$variant" "$dataset" "$clusters" "$ITERATIONS" "$LOCAL_STEPS" "$THREADS_PER_CLUSTER" <<'PY'
import json, pathlib, sys
path, variant, dataset, clusters, iterations, local_steps, threads = sys.argv[1:]
try:
    lines = open(path, encoding="utf-8")
except FileNotFoundError:
    raise SystemExit(1)
for line in lines:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if (row.get("variant") == variant
            and pathlib.Path(row.get("dataset", "")).name == dataset
            and row.get("clusters") == int(clusters)
            and row.get("iterations") == int(iterations)
            and row.get("localSteps") == int(local_steps)
            and row.get("threadsPerCluster") == int(threads)):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

clear_completed() {
  local result_file=$1 variant=$2 dataset=$3 clusters=$4
  "$PYTHON" - "$result_file" "$STATUS_FILE" "$variant" "$dataset" \
  "$clusters" "$ITERATIONS" "$LOCAL_STEPS" "$THREADS_PER_CLUSTER" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

result_path, status_path, variant, dataset, clusters, iterations, local_steps, threads = sys.argv[1:]
clusters = int(clusters)
iterations = int(iterations)
local_steps = int(local_steps)
threads = int(threads)

def replace_lines(path, keep):
  path = pathlib.Path(path)
  if not path.exists():
    return
  with path.open(encoding="utf-8") as source:
    lines = source.readlines()
  retained = [line for index, line in enumerate(lines) if keep(index, line)]
  descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
      output.writelines(retained)
    os.replace(temporary, path)
  finally:
    if os.path.exists(temporary):
      os.remove(temporary)

def keep_result(_index, line):
  try:
    row = json.loads(line)
  except json.JSONDecodeError:
    return True
  return not (
    row.get("variant") == variant
    and pathlib.Path(row.get("dataset", "")).name == dataset
    and row.get("clusters") == clusters
    and row.get("iterations") == iterations
    and row.get("localSteps") == local_steps
    and row.get("threadsPerCluster") == threads
  )

def keep_status(index, line):
  if index == 0:
    return True
  fields = line.rstrip("\n").split("\t")
  if len(fields) < 6:
    return True
  return not (
    fields[0] == variant
    and fields[1] == dataset
    and fields[2] == str(clusters)
    and fields[3] == str(iterations)
    and fields[4] == str(local_steps)
    and fields[5] == str(threads)
  )

replace_lines(result_path, keep_result)
replace_lines(status_path, keep_status)
PY
}

read_max_rss() {
  awk -F: '/Maximum resident set size/ {gsub(/^[[:space:]]+/, "", $2); print $2}' "$1"
}

run_case() {
  local variant=$1 dataset=$2 short_name=$3 clusters=$4
  local label="${variant}_${short_name}_k${clusters}_i${ITERATIONS}_l${LOCAL_STEPS}_t${THREADS_PER_CLUSTER}"
  local result_file="$OUTPUT_DIR/${variant}.jsonl"
  local state_file="$OUTPUT_DIR/states/${label}.npz"
  local log_file="$OUTPUT_DIR/logs/${label}.log"
  local worker_log="$OUTPUT_DIR/logs/${label}_worker.log"
  local coordinator_time="$OUTPUT_DIR/memory/${label}_coordinator.time"
  local worker_time="$OUTPUT_DIR/memory/${label}_worker.time"
  local start_seconds=$SECONDS exit_code status coordinator_rss worker_rss
  local -a debug_arguments=()
  if [[ "$DEBUG_OUTPUT" == "1" ]]; then
    debug_arguments+=(--debug-output)
  fi

  echo "Coordinator log: $log_file"
  echo "Worker log:      $worker_log"

  cleanup_worker
  (cd "$SCRIPT_DIR" && exec /usr/bin/time -v -o "$worker_time" \
    env BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
        BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
        BUNDLE_PALM_THREADS_PER_CLUSTER="$THREADS_PER_CLUSTER" "$WORKER") \
    > "$worker_log" 2>&1 &
  WORKER_PID=$!
  for _ in {1..100}; do
    grep -q "Starting the server" "$worker_log" 2>/dev/null && break
    kill -0 "$WORKER_PID" 2>/dev/null || break
    read -r -t 0.1 _ || true
  done
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tserver_failed\t1\t0\t\t\n' \
      "$variant" "$dataset" "$clusters" "$ITERATIONS" "$LOCAL_STEPS" \
      "$THREADS_PER_CLUSTER" \
      >> "$STATUS_FILE"
    return 1
  fi

  set +e
  if [[ "$LIVE_OUTPUT" == "1" ]]; then
    (cd "$SCRIPT_DIR" && /usr/bin/time -v -o "$coordinator_time" \
      timeout --signal=TERM --kill-after=30 "$CASE_TIMEOUT_SECONDS" \
      env BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
          BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
          BUNDLE_PALM_PROTO_BUILD="$PROTO_BUILD" \
          "$PYTHON" "$CLIENT" "$WORKSPACE/$dataset" \
          --variant-name "$variant" \
          --iterations "$ITERATIONS" \
          --clusters "$clusters" \
          --local-steps "$LOCAL_STEPS" \
          --threads-per-cluster "$THREADS_PER_CLUSTER" \
          --results "$result_file" \
          --state "$state_file" \
          "${mode_arguments[@]}" \
          "${debug_arguments[@]}") 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
  else
    (cd "$SCRIPT_DIR" && /usr/bin/time -v -o "$coordinator_time" \
      timeout --signal=TERM --kill-after=30 "$CASE_TIMEOUT_SECONDS" \
      env BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
          BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
          BUNDLE_PALM_PROTO_BUILD="$PROTO_BUILD" \
          "$PYTHON" "$CLIENT" "$WORKSPACE/$dataset" \
          --variant-name "$variant" \
          --iterations "$ITERATIONS" \
          --clusters "$clusters" \
          --local-steps "$LOCAL_STEPS" \
          --threads-per-cluster "$THREADS_PER_CLUSTER" \
          --results "$result_file" \
          --state "$state_file" \
          "${mode_arguments[@]}" \
          "${debug_arguments[@]}") > "$log_file" 2>&1
    exit_code=$?
  fi
  set -e
  cleanup_worker

  status=failed
  [[ $exit_code -eq 0 ]] && status=completed
  coordinator_rss=$(read_max_rss "$coordinator_time")
  worker_rss=$(read_max_rss "$worker_time")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$dataset" "$clusters" "$ITERATIONS" "$LOCAL_STEPS" \
    "$THREADS_PER_CLUSTER" "$status" "$exit_code" \
    "$((SECONDS - start_seconds))" \
    "$coordinator_rss" "$worker_rss" >> "$STATUS_FILE"
}

for variant in $MODE_LIST; do
  status=$(manifest_status "$variant") || {
    echo "Unknown variant: $variant" >&2
    exit 2
  }
  if [[ "$status" == "pending" ]]; then
    echo "Skipping pending variant $variant" >&2
    continue
  fi
  load_mode_arguments "$variant"
  for problem_spec in "${PROBLEMS[@]}"; do
    IFS='|' read -r dataset short_name <<< "$problem_spec"
    if [[ -n "$PROBLEM_FILTER" \
      && " $PROBLEM_FILTER " != *" $short_name "* ]]; then
      continue
    fi
    if [[ ! -f "$WORKSPACE/$dataset" ]]; then
      echo "Missing dataset $WORKSPACE/$dataset" >&2
      exit 1
    fi
    for clusters in $CLUSTERS_LIST; do
      result_file="$OUTPUT_DIR/${variant}.jsonl"
      if [[ "$OVERWRITE" == "1" ]]; then
        echo "Overwriting $variant $dataset K=$clusters"
        clear_completed "$result_file" "$variant" "$dataset" "$clusters"
      elif completed "$result_file" "$variant" "$dataset" "$clusters"; then
        echo "Skipping completed $variant $dataset K=$clusters"
        continue
      fi
      echo "Running $variant $dataset K=$clusters"
      run_case "$variant" "$dataset" "$short_name" "$clusters" || true
    done
  done
done

echo "ADMM innovation matrix finished: $STATUS_FILE"
