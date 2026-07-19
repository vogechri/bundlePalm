#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

PYTHON=/home/vogechri/bae/.venv/bin/python
RUN_NAME=${RUN_NAME:-}
if [[ -n "$RUN_NAME" && ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_NAME may contain only letters, numbers, dots, underscores, and hyphens" >&2
    exit 2
fi
if [[ -n "$RUN_NAME" ]]; then
    RUN_DIR="$SCRIPT_DIR/palm_runs/$RUN_NAME"
    OUTPUT_DIR=${OUTPUT_DIR:-"$RUN_DIR/trajectories"}
    RESULTS_FILE=${RESULTS_FILE:-"$RUN_DIR/results_palm.json"}
else
    OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/palm_results"}
    RESULTS_FILE=${RESULTS_FILE:-"$SCRIPT_DIR/results_palm.json"}
fi
EPOCHS=${EPOCHS:-90}
PARTITIONS=${PARTITIONS:-10}
RECOMPUTE=${RECOMPUTE:-}
mkdir -p "$OUTPUT_DIR" "$(dirname -- "$RESULTS_FILE")"
: > "$RESULTS_FILE"

echo "Run name: ${RUN_NAME:-default}"
echo "Trajectories: $OUTPUT_DIR"
echo "Summary: $RESULTS_FILE"

COMMON_ARGS=(
    --execution sequential
    --workers 1
    --local-solver bae
    --accelerator nesterov
    --momentum-schedule palm
    --local-nfev 2
    --runtime-weight 0
    --partitioner overlap
    --block-overrelaxation 1.0488088481701516
    --no-block-safeguard
)

should_recompute() {
    local file_name=$1
    local short_name=$2
    local selector
    local selectors=()

    IFS=',' read -ra selectors <<< "$RECOMPUTE"
    for selector in "${selectors[@]}"; do
        if [[ "$selector" == "$short_name" || "$selector" == "$file_name" ]]; then
            return 0
        fi
    done
    return 1
}

run_problem() {
    local base_url=$1
    local file_name=$2
    local short_name=$3
    local result_file="$OUTPUT_DIR/palm${short_name}.jsonl"
    local state_file="$OUTPUT_DIR/palm${short_name}.npz"
    local log_file="$OUTPUT_DIR/palm${short_name}.txt"
    local exit_code=0
    local status=completed
    local recompute=false

    if should_recompute "$file_name" "$short_name"; then
        recompute=true
        echo "Recomputing $file_name"
    fi

    if [[ "$recompute" == false && -s "$result_file" ]] && "$PYTHON" - "$result_file" "$EPOCHS" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]
expected_epochs = int(sys.argv[2])
raise SystemExit(not (len(records) == expected_epochs
                      and records[-1]["epoch"] == expected_epochs - 1))
PY
    then
        echo "Skipping completed $file_name"
    else
        rm -f "$result_file" "$state_file"
        "$PYTHON" -u palm_ba.py "$base_url" "$file_name" \
            "$EPOCHS" "$PARTITIONS" "${COMMON_ARGS[@]}" \
            --output "$result_file" > "$log_file" 2>&1
        exit_code=$?
        if [[ $exit_code -eq 132 ]]; then
            echo "Retrying $file_name after SIGILL"
            rm -f "$result_file" "$state_file"
            "$PYTHON" -u palm_ba.py "$base_url" "$file_name" \
                "$EPOCHS" "$PARTITIONS" "${COMMON_ARGS[@]}" \
                --output "$result_file" > "$log_file" 2>&1
            exit_code=$?
        fi
        if [[ $exit_code -ne 0 ]]; then
            status=failed
            echo "$file_name failed with exit code $exit_code; see $log_file" >&2
        fi
    fi

    if [[ -s "$result_file" ]]; then
        "$PYTHON" - "$result_file" "$RESULTS_FILE" "$base_url" \
            "$file_name" "$EPOCHS" "$PARTITIONS" "$status" <<'PY'
import json
import sys

(trajectory_path, summary_path, base_url, file_name,
 epochs, partitions, status) = sys.argv[1:]
with open(trajectory_path, encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]
best_record = min(records, key=lambda record: record["best_cost"])
summary = {
    "algorithm": "palm_ba",
    "base_url": base_url,
    "file_name": file_name,
    "iterations": int(epochs),
    "bestCost": int(round(best_record["best_cost"])),
    "bestIt": best_record["epoch"],
    "kClusters": int(partitions),
    "bestCost60": (int(round(records[59]["best_cost"]))
                   if len(records) >= 60 else None),
    "bestCost30": (int(round(records[29]["best_cost"]))
                   if len(records) >= 30 else None),
    "status": status,
    "epochsCompleted": len(records),
    "execution": best_record["execution"],
    "accelerator": best_record["accelerator"],
    "localSolver": best_record["local_solver"],
    "partitioner": best_record["partitioner"],
    "trajectory": trajectory_path,
}
with open(summary_path, "a", encoding="utf-8") as handle:
    json.dump(summary, handle, separators=(",", ":"))
    handle.write("\n")
PY
    fi
}

run_problem http://grail.cs.washington.edu/projects/bal/data/final/ problem-394-100368-pre.txt.bz2 394
run_problem http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1064-113655-pre.txt.bz2 1064
run_problem http://grail.cs.washington.edu/projects/bal/data/venice/ problem-245-198739-pre.txt.bz2 245
run_problem http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1723-156502-pre.txt.bz2 1723
run_problem http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-1266-132593-pre.txt.bz2 1266
run_problem http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-931-102699-pre.txt.bz2 931
run_problem http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-783-84444-pre.txt.bz2 783
run_problem http://grail.cs.washington.edu/projects/bal/data/venice/ problem-89-110973-pre.txt.bz2 89
run_problem http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-287-182023-pre.txt.bz2 287
run_problem http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-142-93602-pre.txt.bz2 142
run_problem http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-646-73584-pre.txt.bz2 646
run_problem http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-135-90642-pre.txt.bz2 135
run_problem http://grail.cs.washington.edu/projects/bal/data/venice/ problem-52-64053-pre.txt.bz2 52
run_problem http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-173-111908-pre.txt.bz2 173
run_problem http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-356-226730-pre.txt.bz2 356
run_problem http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-88-64298-pre.txt.bz2 88
run_problem http://grail.cs.washington.edu/projects/bal/data/ladybug/ problem-49-7776-pre.txt.bz2 49
run_problem http://grail.cs.washington.edu/projects/bal/data/trafalgar/ problem-126-40037-pre.txt.bz2 126
run_problem http://grail.cs.washington.edu/projects/bal/data/venice/ problem-427-310384-pre.txt.bz2 427
run_problem http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-253-163691-pre.txt.bz2 253
run_problem http://grail.cs.washington.edu/projects/bal/data/final/ problem-961-187103-pre.txt.bz2 961
run_problem http://grail.cs.washington.edu/projects/bal/data/trafalgar/ problem-257-65132-pre.txt.bz2 257
run_problem http://grail.cs.washington.edu/projects/bal/data/venice/ problem-744-543562-pre.txt.bz2 744
run_problem http://grail.cs.washington.edu/projects/bal/data/venice/ problem-951-708276-pre.txt.bz2 951
run_problem http://grail.cs.washington.edu/projects/bal/data/dubrovnik/ problem-308-195089-pre.txt.bz2 308
run_problem http://grail.cs.washington.edu/projects/bal/data/final/ problem-871-527480-pre.txt.bz2 871
run_problem http://grail.cs.washington.edu/projects/bal/data/venice/ problem-1778-993923-pre.txt.bz2 1778
run_problem http://grail.cs.washington.edu/projects/bal/data/venice/ problem-1490-935273-pre.txt.bz2 1490
run_problem http://grail.cs.washington.edu/projects/bal/data/final/ problem-3068-310854-pre.txt.bz2 3068
