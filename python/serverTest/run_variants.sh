#!/usr/bin/env bash

# Compare outer DRS accelerators on every active problem in run.sh order.
# Override defaults, for example:
#   PER_RUN_TIMEOUT=10m ITERATIONS=60 MODES="anderson lbfgs themelis_nesterov" ./run_variants.sh

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON="$SCRIPT_DIR/.venv/bin/python"
CLIENT="$SCRIPT_DIR/client_acc_variants.py"
SERVER="$SCRIPT_DIR/build/zeromq_cpp_server_ex"
PROTO_DIR="$SCRIPT_DIR/build/generated/proto"
RESULTS_FILE="$SCRIPT_DIR/results_server.json"

ITERATIONS=${ITERATIONS:-90}
CLUSTERS=${CLUSTERS:-30}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-}
ACCEL_BACKTRACKS=${ACCEL_BACKTRACKS:-0}
DRY_RUN=${DRY_RUN:-0}

read -r -a MODES_TO_RUN <<< "${MODES:-anderson lbfgs nesterov themelis_nesterov fista adaptive_nesterov}"
DEFAULT_PROBLEMS=(
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
if [[ -n ${PROBLEMS:-} ]]; then
    read -r -a PROBLEMS_TO_RUN <<< "$PROBLEMS"
else
    PROBLEMS_TO_RUN=("${DEFAULT_PROBLEMS[@]}")
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python environment: $PYTHON" >&2
    exit 1
fi
if [[ ! -x "$SERVER" ]]; then
    echo "Missing server executable: $SERVER" >&2
    exit 1
fi
if [[ ! -f "$PROTO_DIR/test_pb2.py" ]]; then
    echo "Missing generated protobuf module: $PROTO_DIR/test_pb2.py" >&2
    exit 1
fi
if [[ -n "$PER_RUN_TIMEOUT" ]] && ! command -v timeout >/dev/null 2>&1; then
    echo "The GNU timeout command is required." >&2
    exit 1
fi

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/variant_results/$RUN_ID"}
mkdir -p "$OUTPUT_DIR"
STATUS_FILE="$OUTPUT_DIR/status.tsv"
printf 'problem\taccelerator\tstatus\texit_code\telapsed_seconds\tlog\n' > "$STATUS_FILE"

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ "$DRY_RUN" != "1" ]]; then
    if pgrep -f '[/]zeromq_cpp_server_ex([[:space:]]|$)' >/dev/null 2>&1; then
        echo "Using the already running ZeroMQ server."
    else
        echo "Starting $SERVER"
        (
            cd "$SCRIPT_DIR/build"
            exec ./zeromq_cpp_server_ex
        ) > "$OUTPUT_DIR/server.log" 2>&1 &
        SERVER_PID=$!
        sleep 2
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server failed to start; see $OUTPUT_DIR/server.log" >&2
            exit 1
        fi
    fi
fi

COMMON_ENV=(
    "PYTHONPATH=$PROTO_DIR${PYTHONPATH:+:$PYTHONPATH}"
    "BUNDLE_PALM_ACCEL_BACKTRACKS=$ACCEL_BACKTRACKS"
    "BUNDLE_PALM_LBFGS_MEMORY=5"
    "BUNDLE_PALM_ANDERSON_MEMORY=5"
    "BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS=1"
    "BUNDLE_PALM_MAX_REFINEMENT_PASSES=3"
    "BUNDLE_PALM_REPAIR_RESTART_INTERVAL=32"
    "BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS=2"
    "BUNDLE_PALM_CLUSTERING=landmark_scalable"
    "BUNDLE_PALM_RESIDUAL_BALANCE_SLACK=0.01"
    "BUNDLE_PALM_MIN_CAMERA_LANDMARKS=20"
)

TOTAL_RUNS=$(( ${#PROBLEMS_TO_RUN[@]} * ${#MODES_TO_RUN[@]} ))
RUN_NUMBER=0

for problem_spec in "${PROBLEMS_TO_RUN[@]}"; do
    IFS='|' read -r collection file_name short_name <<< "$problem_spec"
    base_url="http://grail.cs.washington.edu/projects/bal/data/$collection/"

    for accelerator in "${MODES_TO_RUN[@]}"; do
        RUN_NUMBER=$((RUN_NUMBER + 1))
        log_file="$OUTPUT_DIR/${short_name}_${accelerator}.log"
        result_file="$OUTPUT_DIR/${short_name}_${accelerator}.json"
        echo "[$RUN_NUMBER/$TOTAL_RUNS] $file_name with $accelerator"

        command=(
            env "${COMMON_ENV[@]}" "BUNDLE_PALM_ACCELERATOR=$accelerator"
            "$PYTHON" -u "$CLIENT" "$base_url" "$file_name"
            "$ITERATIONS" "$CLUSTERS"
        )
        if [[ -n "$PER_RUN_TIMEOUT" ]]; then
            command=(
                timeout --foreground --kill-after=2m "$PER_RUN_TIMEOUT"
                "${command[@]}"
            )
        fi

        if [[ "$DRY_RUN" == "1" ]]; then
            printf '  '
            printf '%q ' "${command[@]}"
            printf '\n'
            continue
        fi

        result_lines_before=0
        if [[ -f "$RESULTS_FILE" ]]; then
            result_lines_before=$(wc -l < "$RESULTS_FILE")
        fi
        start_seconds=$SECONDS

        set +e
        (
            cd "$SCRIPT_DIR"
            "${command[@]}"
        ) > "$log_file" 2>&1
        exit_code=$?
        set -e

        elapsed_seconds=$((SECONDS - start_seconds))
        status=failed
        if [[ $exit_code -eq 0 ]]; then
            status=completed
        elif [[ $exit_code -eq 124 ]]; then
            status=timed_out
        fi

        if [[ -f "$RESULTS_FILE" ]] && [[ $(wc -l < "$RESULTS_FILE") -gt $result_lines_before ]]; then
            tail -n 1 "$RESULTS_FILE" > "$result_file"
        fi

        printf '%s\t%s\t%s\t%d\t%d\t%s\n' \
            "$file_name" "$accelerator" "$status" "$exit_code" \
            "$elapsed_seconds" "$log_file" >> "$STATUS_FILE"
        echo "  $status after ${elapsed_seconds}s; log: $log_file"

        if [[ $exit_code -eq 124 ]]; then
            echo "  Cooling down after timeout so the server can discard stale work."
            sleep 30
        fi
    done
done

echo "Variant benchmark finished."
echo "Status: $STATUS_FILE"
echo "Per-run results and logs: $OUTPUT_DIR"
