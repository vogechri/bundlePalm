#!/usr/bin/env bash

# Compare PALM execution and acceleration variants on representative BAL data.
# All defaults can be overridden, for example:
#   EPOCHS=3 PARTITIONS=4 PROBLEMS="ladybug|problem-49-7776-pre.txt.bz2|49" ./run_palm_variants.sh

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON="$SCRIPT_DIR/serverTest/.venv/bin/python"
CLIENT="$SCRIPT_DIR/palm_ba.py"

EPOCHS=${EPOCHS:-5}
PARTITIONS=${PARTITIONS:-6}
LOCAL_NFEV=${LOCAL_NFEV:-2}
MEMORY=${MEMORY:-5}
MOMENTUM=${MOMENTUM:-0.8}
MOMENTUM_SCHEDULE=${MOMENTUM_SCHEDULE:-constant}
MAX_ACCEL_INCREASE=${MAX_ACCEL_INCREASE:-0.02}
AGE_WEIGHT=${AGE_WEIGHT:-1.0}
RUNTIME_WEIGHT=${RUNTIME_WEIGHT:-0.0}
WORKERS=${WORKERS:-$PARTITIONS}
GLOBAL_JACOBI=${GLOBAL_JACOBI:-none}
GLOBAL_JACOBI_BATCH_SIZE=${GLOBAL_JACOBI_BATCH_SIZE:-50000}
GLOBAL_JACOBI_FLOOR=${GLOBAL_JACOBI_FLOOR:-1e-12}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-}
DRY_RUN=${DRY_RUN:-0}

read -r -a EXECUTIONS_TO_RUN <<< "${EXECUTIONS:-sequential parallel}"
read -r -a ACCELERATORS_TO_RUN <<< "${ACCELERATORS:-none heavy_ball nesterov bfgs anderson}"
DEFAULT_PROBLEMS=(
    "ladybug|problem-49-7776-pre.txt.bz2|49"
    "dubrovnik|problem-142-93602-pre.txt.bz2|142"
    "venice|problem-52-64053-pre.txt.bz2|52"
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
if [[ ! -f "$CLIENT" ]]; then
    echo "Missing PALM client: $CLIENT" >&2
    exit 1
fi
if [[ -n "$PER_RUN_TIMEOUT" ]] && ! command -v timeout >/dev/null 2>&1; then
    echo "The GNU timeout command is required when PER_RUN_TIMEOUT is set." >&2
    exit 1
fi

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/palm_variant_results/$RUN_ID"}
mkdir -p "$OUTPUT_DIR"
STATUS_FILE="$OUTPUT_DIR/status.tsv"
LIVE_SUMMARY_FILE="$OUTPUT_DIR/live_results.json"
SUMMARY_FILE=${SUMMARY_FILE:-"$SCRIPT_DIR/results_palm_variants.json"}
printf 'problem\texecution\taccelerator\tstatus\texit_code\telapsed_seconds\tbest_cost\tlog\n' > "$STATUS_FILE"

TOTAL_RUNS=$(( ${#PROBLEMS_TO_RUN[@]} * ${#EXECUTIONS_TO_RUN[@]} * ${#ACCELERATORS_TO_RUN[@]} ))
RUN_NUMBER=0

trap 'exit 130' INT TERM

for problem_spec in "${PROBLEMS_TO_RUN[@]}"; do
    IFS='|' read -r collection file_name short_name <<< "$problem_spec"
    base_url="http://grail.cs.washington.edu/projects/bal/data/$collection/"

    for execution in "${EXECUTIONS_TO_RUN[@]}"; do
        for accelerator in "${ACCELERATORS_TO_RUN[@]}"; do
            RUN_NUMBER=$((RUN_NUMBER + 1))
            stem="${short_name}_${execution}_${accelerator}"
            result_file="$OUTPUT_DIR/$stem.jsonl"
            log_file="$OUTPUT_DIR/$stem.log"
            echo "[$RUN_NUMBER/$TOTAL_RUNS] $file_name, $execution, $accelerator"

            command=(
                "$PYTHON" -u "$CLIENT" "$base_url" "$file_name"
                "$EPOCHS" "$PARTITIONS"
                --execution "$execution"
                --workers "$WORKERS"
                --accelerator "$accelerator"
                --momentum "$MOMENTUM"
                --momentum-schedule "$MOMENTUM_SCHEDULE"
                --max-accel-increase "$MAX_ACCEL_INCREASE"
                --memory "$MEMORY"
                --local-nfev "$LOCAL_NFEV"
                --age-weight "$AGE_WEIGHT"
                --runtime-weight "$RUNTIME_WEIGHT"
                --global-jacobi "$GLOBAL_JACOBI"
                --global-jacobi-batch-size "$GLOBAL_JACOBI_BATCH_SIZE"
                --global-jacobi-floor "$GLOBAL_JACOBI_FLOOR"
                --output "$result_file"
                --live-summary "$LIVE_SUMMARY_FILE"
            )
            if [[ -n "$PER_RUN_TIMEOUT" ]]; then
                command=(timeout --foreground --kill-after=2m "$PER_RUN_TIMEOUT" "${command[@]}")
            fi

            if [[ "$DRY_RUN" == "1" ]]; then
                printf '  '
                printf '%q ' "${command[@]}"
                printf '\n'
                continue
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
            best_cost=""
            if [[ -s "$result_file" ]]; then
                best_cost=$(
                    "$PYTHON" - "$result_file" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]
print(min(record["best_cost"] for record in records))
PY
                )
            fi
            printf '%s\t%s\t%s\t%s\t%d\t%d\t%s\t%s\n' \
                "$file_name" "$execution" "$accelerator" "$status" \
                "$exit_code" "$elapsed_seconds" "$best_cost" "$log_file" \
                >> "$STATUS_FILE"
            if [[ -s "$result_file" ]]; then
                "$PYTHON" - "$result_file" "$SUMMARY_FILE" "$RUN_ID" \
                    "$base_url" "$file_name" "$EPOCHS" "$PARTITIONS" \
                    "$LOCAL_NFEV" "$execution" "$accelerator" "$status" \
                    "$elapsed_seconds" "$WORKERS" "$MOMENTUM" \
                    "$MOMENTUM_SCHEDULE" "$MAX_ACCEL_INCREASE" \
                    "$GLOBAL_JACOBI" "$GLOBAL_JACOBI_FLOOR" <<'PY'
import json
import sys

(trajectory_path, summary_path, run_id, base_url, file_name, epochs,
 partitions, local_nfev, execution, accelerator, status,
 elapsed_seconds, workers, momentum, momentum_schedule,
 max_accel_increase, global_jacobi, global_jacobi_floor) = sys.argv[1:]
with open(trajectory_path, encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]
best_record = min(records, key=lambda record: record["best_cost"])
summary = {
    "algorithm": "palm_ba",
    "base_url": base_url,
    "file_name": file_name,
    "epochs": int(epochs),
    "iterations": None,
    "bestCost": best_record["best_cost"],
    "bestIt": best_record["epoch"],
    "partitions": int(partitions),
    "kClusters": None,
    "localNfev": int(local_nfev),
    "execution": execution,
    "workers": int(workers),
    "accelerator": accelerator,
    "momentum": float(momentum),
    "momentumSchedule": momentum_schedule,
    "maxAccelerationIncrease": float(max_accel_increase),
    "globalJacobi": global_jacobi,
    "globalJacobiFloor": float(global_jacobi_floor),
    "bestCost30": records[29]["best_cost"] if len(records) >= 30 else None,
    "bestCost60": records[59]["best_cost"] if len(records) >= 60 else None,
    "status": status,
    "elapsedSeconds": int(elapsed_seconds),
    "runId": run_id,
    "trajectory": trajectory_path,
}
with open(summary_path, "a", encoding="utf-8") as handle:
    json.dump(summary, handle, separators=(",", ":"))
    handle.write("\n")
PY
            fi
            echo "  $status after ${elapsed_seconds}s; best=$best_cost"
        done
    done
done

echo "PALM variant benchmark finished."
echo "Status: $STATUS_FILE"
echo "Summary: $SUMMARY_FILE"
echo "Logs, trajectories, and states: $OUTPUT_DIR"
