#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON="$SCRIPT_DIR/.venv/bin/python"
CLIENT="$SCRIPT_DIR/client_drs.py"
WORKER=${WORKER:-"$SCRIPT_DIR/build_admm/zeromq_cpp_server_ex"}
PROTO_BUILD=${PROTO_BUILD:-"$SCRIPT_DIR/build_admm"}
OUTPUT_DIR=${OUTPUT_DIR:-"$WORKSPACE/benchmark_results/drs_failure_top3_i30_k10_k20_k30"}
PROBLEM_FILTER=${PROBLEM_FILTER:-"646 931 1266"}
CLUSTERS_LIST=${CLUSTERS_LIST:-"10 20 30"}
ITERATIONS=${ITERATIONS:-30}
LOCAL_STEPS=${LOCAL_STEPS:-1}
THREADS_PER_CLUSTER=${THREADS_PER_CLUSTER:-1}
LOCAL_SOLVER=${LOCAL_SOLVER:-nesterov}
TRUST_REGION_POLICY=${TRUST_REGION_POLICY:-daba}
CAMERA_SCALING=${CAMERA_SCALING:-jacobi_initial}
PROXIMAL_METRIC=${PROXIMAL_METRIC:-block}
CONSENSUS_METRIC=${CONSENSUS_METRIC:-${BUNDLE_PALM_DRS_CONSENSUS_METRIC:-full}}
BLOCK_REGULARIZATION=${BLOCK_REGULARIZATION:-5e-5}
MAXIMUM_BLOCK_REGULARIZATION=${MAXIMUM_BLOCK_REGULARIZATION:-0.5}
RELAXATION=${RELAXATION:-1.0}
PENALTY_MULTIPLIER=${PENALTY_MULTIPLIER:-1.0}
SAFEGUARD_MODE=${SAFEGUARD_MODE:-relative}
DRE_RELATIVE_INCREASE=${DRE_RELATIVE_INCREASE:-0.01}
MINIMUM_PRIMAL_RATIO=${MINIMUM_PRIMAL_RATIO:-1.001}
CATASTROPHIC_RATIO=${CATASTROPHIC_RATIO:-${SAFEGUARD_RATIO:-1000000}}
RECOVERY_PENALTY_RATIO=${RECOVERY_PENALTY_RATIO:-2.0}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-3600}
LIVE_OUTPUT=${LIVE_OUTPUT:-1}
DEBUG_OUTPUT=${DEBUG_OUTPUT:-1}
OVERWRITE=${OVERWRITE:-0}
REQUEST_PORT=${BUNDLE_PALM_REQUEST_PORT:-6656}
RESULT_PORT=${BUNDLE_PALM_RESULT_PORT:-6657}
VARIANT_NAME="plain_drs_${PROXIMAL_METRIC}_${CONSENSUS_METRIC}"
RESULT_FILE="$OUTPUT_DIR/${VARIANT_NAME}.jsonl"
STATUS_FILE="$OUTPUT_DIR/status.tsv"
WORKER_PID=""

for flag in LIVE_OUTPUT DEBUG_OUTPUT OVERWRITE; do
  value=${!flag}
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "$flag must be 0 or 1" >&2
    exit 2
  fi
done

PROBLEMS=(
  "52|problem-52-64053-pre.txt"
  "646|problem-646-73584-pre.txt"
  "931|problem-931-102699-pre.txt"
  "1266|problem-1266-132593-pre.txt"
)

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

case_exists() {
  local dataset=$1 clusters=$2
  "$PYTHON" - "$RESULT_FILE" "$dataset" "$clusters" "$ITERATIONS" "$LOCAL_STEPS" "$THREADS_PER_CLUSTER" "$VARIANT_NAME" <<'PY'
import json, pathlib, sys
path, dataset, clusters, iterations, local_steps, threads, variant = sys.argv[1:]
try:
    lines = open(path, encoding="utf-8")
except FileNotFoundError:
    raise SystemExit(1)
for line in lines:
    row = json.loads(line)
    if (pathlib.Path(row.get("dataset", "")).name == dataset
            and row.get("clusters") == int(clusters)
            and row.get("iterations") == int(iterations)
            and row.get("localSteps") == int(local_steps)
            and row.get("threadsPerCluster") == int(threads)
            and row.get("variant") == variant):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

clear_case() {
  local dataset=$1 clusters=$2
  "$PYTHON" - "$RESULT_FILE" "$STATUS_FILE" "$dataset" "$clusters" "$ITERATIONS" "$LOCAL_STEPS" "$THREADS_PER_CLUSTER" "$VARIANT_NAME" <<'PY'
import json, os, pathlib, sys, tempfile
result_path, status_path, dataset, clusters, iterations, local_steps, threads, variant = sys.argv[1:]
key = variant, dataset, int(clusters), int(iterations), int(local_steps), int(threads)

def atomic_filter(path, keep):
    path = pathlib.Path(path)
    if not path.exists(): return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.writelines(line for index, line in enumerate(lines) if keep(index, line))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.remove(temporary)

def keep_result(_, line):
    row = json.loads(line)
    found = (row.get("variant"), pathlib.Path(row.get("dataset", "")).name, row.get("clusters"), row.get("iterations"), row.get("localSteps"), row.get("threadsPerCluster"))
    return found != key

def keep_status(index, line):
    if index == 0: return True
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 6: return True
    found = fields[0], fields[1], int(fields[2]), int(fields[3]), int(fields[4]), int(fields[5])
    return found != key

atomic_filter(result_path, keep_result)
atomic_filter(status_path, keep_status)
PY
}

read_max_rss() {
  awk -F: '/Maximum resident set size/ {gsub(/^[[:space:]]+/, "", $2); print $2}' "$1"
}

for problem in "${PROBLEMS[@]}"; do
  IFS='|' read -r scene dataset <<< "$problem"
  if [[ " $PROBLEM_FILTER " != *" $scene "* ]]; then
    continue
  fi
  for clusters in $CLUSTERS_LIST; do
    label="${VARIANT_NAME}_${scene}_k${clusters}_i${ITERATIONS}_l${LOCAL_STEPS}_t${THREADS_PER_CLUSTER}"
    log_file="$OUTPUT_DIR/logs/${label}.log"
    worker_log="$OUTPUT_DIR/logs/${label}_worker.log"
    state_file="$OUTPUT_DIR/states/${label}.npz"
    coordinator_time="$OUTPUT_DIR/memory/${label}_coordinator.time"
    worker_time="$OUTPUT_DIR/memory/${label}_worker.time"
    if [[ "$OVERWRITE" == "1" ]]; then
      echo "Overwriting $VARIANT_NAME $dataset K=$clusters"
      clear_case "$dataset" "$clusters"
    elif case_exists "$dataset" "$clusters"; then
      echo "Skipping completed $VARIANT_NAME $dataset K=$clusters"
      continue
    fi

    echo "Running $VARIANT_NAME $dataset K=$clusters"
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
      echo "DRS worker failed to start" >&2
      exit 1
    fi

    debug_args=()
    [[ "$DEBUG_OUTPUT" == "1" ]] && debug_args+=(--debug-output)
    start_seconds=$SECONDS
    set +e
    if [[ "$LIVE_OUTPUT" == "1" ]]; then
      (cd "$SCRIPT_DIR" && /usr/bin/time -v -o "$coordinator_time" \
        timeout --signal=TERM --kill-after=30 "$CASE_TIMEOUT_SECONDS" \
        env BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
            BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
            BUNDLE_PALM_PROTO_BUILD="$PROTO_BUILD" \
            "$PYTHON" "$CLIENT" "$WORKSPACE/$dataset" \
            --variant-name "$VARIANT_NAME" \
            --iterations "$ITERATIONS" --clusters "$clusters" \
            --local-steps "$LOCAL_STEPS" \
            --threads-per-cluster "$THREADS_PER_CLUSTER" \
            --local-solver "$LOCAL_SOLVER" \
            --trust-region-policy "$TRUST_REGION_POLICY" \
            --camera-scaling "$CAMERA_SCALING" \
            --proximal-metric "$PROXIMAL_METRIC" \
            --consensus-metric "$CONSENSUS_METRIC" \
            --block-regularization "$BLOCK_REGULARIZATION" \
            --maximum-block-regularization "$MAXIMUM_BLOCK_REGULARIZATION" \
            --relaxation "$RELAXATION" \
            --penalty-multiplier "$PENALTY_MULTIPLIER" \
            --safeguard-mode "$SAFEGUARD_MODE" \
            --dre-relative-increase "$DRE_RELATIVE_INCREASE" \
            --minimum-primal-ratio "$MINIMUM_PRIMAL_RATIO" \
            --catastrophic-ratio "$CATASTROPHIC_RATIO" \
            --recovery-penalty-ratio "$RECOVERY_PENALTY_RATIO" \
            --results "$RESULT_FILE" --state "$state_file" \
            "${debug_args[@]}") 2>&1 | tee "$log_file"
      exit_code=${PIPESTATUS[0]}
    else
      (cd "$SCRIPT_DIR" && /usr/bin/time -v -o "$coordinator_time" \
        timeout --signal=TERM --kill-after=30 "$CASE_TIMEOUT_SECONDS" \
        env BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
            BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
            BUNDLE_PALM_PROTO_BUILD="$PROTO_BUILD" \
            "$PYTHON" "$CLIENT" "$WORKSPACE/$dataset" \
            --variant-name "$VARIANT_NAME" \
            --iterations "$ITERATIONS" --clusters "$clusters" \
            --local-steps "$LOCAL_STEPS" \
            --threads-per-cluster "$THREADS_PER_CLUSTER" \
            --local-solver "$LOCAL_SOLVER" \
            --trust-region-policy "$TRUST_REGION_POLICY" \
            --camera-scaling "$CAMERA_SCALING" \
            --proximal-metric "$PROXIMAL_METRIC" \
            --consensus-metric "$CONSENSUS_METRIC" \
            --block-regularization "$BLOCK_REGULARIZATION" \
            --maximum-block-regularization "$MAXIMUM_BLOCK_REGULARIZATION" \
            --relaxation "$RELAXATION" \
            --penalty-multiplier "$PENALTY_MULTIPLIER" \
            --safeguard-mode "$SAFEGUARD_MODE" \
            --dre-relative-increase "$DRE_RELATIVE_INCREASE" \
            --minimum-primal-ratio "$MINIMUM_PRIMAL_RATIO" \
            --catastrophic-ratio "$CATASTROPHIC_RATIO" \
            --recovery-penalty-ratio "$RECOVERY_PENALTY_RATIO" \
            --results "$RESULT_FILE" --state "$state_file" \
            "${debug_args[@]}") > "$log_file" 2>&1
      exit_code=$?
    fi
    set -e
    cleanup_worker
    status=failed
    [[ $exit_code -eq 0 ]] && status=completed
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$VARIANT_NAME" \
      "$dataset" "$clusters" "$ITERATIONS" "$LOCAL_STEPS" \
      "$THREADS_PER_CLUSTER" "$status" "$exit_code" \
      "$((SECONDS - start_seconds))" "$(read_max_rss "$coordinator_time")" \
      "$(read_max_rss "$worker_time")" >> "$STATUS_FILE"
  done
done

echo "DRS failure matrix finished: $STATUS_FILE"
