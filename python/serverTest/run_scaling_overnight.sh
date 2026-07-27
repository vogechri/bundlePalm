#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON="$SCRIPT_DIR/.venv/bin/python"
CLIENT="$SCRIPT_DIR/client_acc.py"
SERVER="$SCRIPT_DIR/build/zeromq_cpp_server_ex"
RESULTS_FILE="$SCRIPT_DIR/results_server.json"
WAIT_PID=${WAIT_PID:-}
ITERATIONS=${ITERATIONS:-90}
CLUSTERS=${CLUSTERS:-30}
MODES=${MODES:-"jacobi_q025_gmean jacobi_q0375_gmean jacobi_q0625_gmean block_jacobi_gmean"}
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/scaling_results/$RUN_ID"}
STATUS_FILE="$OUTPUT_DIR/status.tsv"
SERVER_PID=""

PROBLEMS=(
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

COMMON_ENV=(
    "BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1"
    "BUNDLE_PALM_MAX_REFINEMENT_PASSES=3"
    "BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32"
    "BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2"
    "BUNDLE_PALM_CLUSTERING=landmark_scalable"
    "BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01"
    "BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20"
)

cleanup() {
    export BUNDLE_PALM_DRS_SCALING=jacobi_gmean
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

mkdir -p "$OUTPUT_DIR"
printf 'problem\tscaling\tstatus\texit_code\telapsed_seconds\tlog\n' > "$STATUS_FILE"

if [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
    echo "Waiting for active scaling runner PID $WAIT_PID"
    tail --pid="$WAIT_PID" -f /dev/null
fi

mapfile -t OLD_SERVER_PIDS < <(pgrep -f '[/]zeromq_cpp_server_ex([[:space:]]|$)' || true)
for pid in "${OLD_SERVER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
done
for _ in {1..50}; do
    servers_stopped=1
    for pid in "${OLD_SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            servers_stopped=0
            break
        fi
    done
    [[ "$servers_stopped" == "1" ]] && break
    sleep 0.1
done
if [[ "$servers_stopped" != "1" ]]; then
    echo "Existing server did not stop cleanly" >&2
    exit 1
fi

(cd "$SCRIPT_DIR" && exec "$SERVER") > "$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
sleep 1
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server failed to start; see $OUTPUT_DIR/server.log" >&2
    exit 1
fi

run_case() {
    local collection=$1 file_name=$2 short_name=$3 mode=$4 iterations=$5 label=$6
    local log_file="$OUTPUT_DIR/${short_name}_${mode}_${label}.log"
    local start_seconds=$SECONDS
    local exit_code status
    set +e
    (cd "$SCRIPT_DIR" && env "${COMMON_ENV[@]}" "BUNDLE_PALM_DRS_SCALING=$mode" \
        "$PYTHON" -u "$CLIENT" \
        "http://grail.cs.washington.edu/projects/bal/data/$collection/" \
        "$file_name" "$iterations" "$CLUSTERS") > "$log_file" 2>&1
    exit_code=$?
    set -e
    status=failed
    [[ $exit_code -eq 0 ]] && status=completed
    printf '%s\t%s\t%s\t%d\t%d\t%s\n' "$file_name" "$mode" "$status" \
        "$exit_code" "$((SECONDS - start_seconds))" "$log_file" >> "$STATUS_FILE"
    return "$exit_code"
}

completed() {
    "$PYTHON" - "$RESULTS_FILE" "$1" "$2" "$ITERATIONS" <<'PY'
import json
import sys

path, method, file_name, iterations = sys.argv[1:]
try:
    lines = open(path, encoding="utf-8")
except FileNotFoundError:
    raise SystemExit(1)
for line in lines:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if (row.get("drsScaling") == method
        and row.get("file_name") == file_name
        and row.get("iterations") == int(iterations)):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "Smoke-testing diagonal and block protocol paths"
run_case ladybug problem-49-7776-pre.txt.bz2 49 jacobi_gmean 2 smoke || exit 1
BLOCK_ENABLED=1
run_case ladybug problem-49-7776-pre.txt.bz2 49 block_jacobi_gmean 2 smoke || BLOCK_ENABLED=0

read -r -a MODES_TO_RUN <<< "$MODES"
for mode in "${MODES_TO_RUN[@]}"; do
    if [[ "$mode" == "block_jacobi_gmean" && "$BLOCK_ENABLED" != "1" ]]; then
        echo "Skipping block cohort because its smoke test failed."
        continue
    fi
    for problem_spec in "${PROBLEMS[@]}"; do
        IFS='|' read -r collection file_name short_name <<< "$problem_spec"
        if completed "$mode" "$file_name"; then
            echo "Skipping completed $file_name with $mode"
            continue
        fi
        echo "Running $file_name with $mode"
        run_case "$collection" "$file_name" "$short_name" "$mode" "$ITERATIONS" full || true
    done
done

echo "Scaling benchmark finished. Status: $STATUS_FILE"