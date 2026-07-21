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
BLOCK_OVERRELAXATION=${BLOCK_OVERRELAXATION:-1.0488088481701516}
ACCELERATOR=${ACCELERATOR:-nesterov}
GLOBAL_JACOBI=${GLOBAL_JACOBI:-full}
GPU_STATE_CACHE=${GPU_STATE_CACHE:-0}
REPARTITION_EVERY=${REPARTITION_EVERY:-0}
REPARTITION_CANDIDATES=${REPARTITION_CANDIDATES:-4}
PARTITION_SEED=${PARTITION_SEED:-0}
RECOMPUTE=${RECOMPUTE:-}
DRY_RUN=${DRY_RUN:-0}
case "$ACCELERATOR" in
    none|heavy_ball|nesterov|anderson|lbfgs|bfgs) ;;
    *)
        echo "Unsupported ACCELERATOR: $ACCELERATOR" >&2
        exit 2
        ;;
esac
    case "$GLOBAL_JACOBI" in
        none|camera|full) ;;
        *)
        echo "Unsupported GLOBAL_JACOBI: $GLOBAL_JACOBI" >&2
        exit 2
        ;;
    esac
if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$OUTPUT_DIR" "$(dirname -- "$RESULTS_FILE")"
    : > "$RESULTS_FILE"
fi

echo "Run name: ${RUN_NAME:-default}"
echo "Accelerator: $ACCELERATOR"
echo "Global Jacobi: $GLOBAL_JACOBI"
echo "GPU state cache: $GPU_STATE_CACHE"
echo "Block over-relaxation: $BLOCK_OVERRELAXATION"
echo "Repartition every: ${REPARTITION_EVERY:-never}"
echo "Trajectories: $OUTPUT_DIR"
echo "Summary: $RESULTS_FILE"

COMMON_ARGS=(
    --execution sequential
    --workers 1
    --local-solver bae
    --accelerator "$ACCELERATOR"
    --momentum-schedule palm
    --local-nfev 2
    --runtime-weight 0
    --partitioner overlap
    --repartition-every "$REPARTITION_EVERY"
    --repartition-candidates "$REPARTITION_CANDIDATES"
    --partition-seed "$PARTITION_SEED"
    --block-overrelaxation "$BLOCK_OVERRELAXATION"
    --no-block-safeguard
    --global-jacobi "$GLOBAL_JACOBI"
)
if [[ "$GPU_STATE_CACHE" == "1" ]]; then
    COMMON_ARGS+=(--gpu-state-cache)
else
    COMMON_ARGS+=(--no-gpu-state-cache)
fi

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

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  '
        printf '%q ' "$PYTHON" -u palm_ba.py "$base_url" "$file_name" \
            "$EPOCHS" "$PARTITIONS" "${COMMON_ARGS[@]}" \
            --output "$result_file"
        printf '\n'
        return
    fi

    if should_recompute "$file_name" "$short_name"; then
        recompute=true
        echo "Recomputing $file_name"
    fi

    if [[ "$recompute" == false && -s "$result_file" ]] && "$PYTHON" - \
        "$result_file" "$EPOCHS" "$PARTITIONS" "$BLOCK_OVERRELAXATION" \
        "$ACCELERATOR" "$GLOBAL_JACOBI" "$REPARTITION_EVERY" "$REPARTITION_CANDIDATES" \
        "$PARTITION_SEED" "$GPU_STATE_CACHE" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]
expected_epochs = int(sys.argv[2])
expected_partitions = int(sys.argv[3])
expected_overrelaxation = float(sys.argv[4])
expected_accelerator = sys.argv[5]
expected_global_jacobi = sys.argv[6]
expected_repartition_every = int(sys.argv[7])
expected_repartition_candidates = int(sys.argv[8])
expected_partition_seed = int(sys.argv[9])
expected_gpu_state_cache = sys.argv[10] == "1"
configuration_matches = records and all(
    record.get("partitions") == expected_partitions
    and record.get("execution") == "sequential"
    and record.get("local_solver") == "bae"
    and record.get("accelerator") == expected_accelerator
    and record.get("momentum_schedule") == "palm"
    and record.get("global_jacobi") == expected_global_jacobi
    and record.get("partitioner") == "overlap"
    and record.get("repartition_every", 0) == expected_repartition_every
    and record.get("repartition_candidates", 4) == expected_repartition_candidates
    and record.get("partition_seed", 0) == expected_partition_seed
    and record.get("gpu_state_cache", False) is expected_gpu_state_cache
    and record.get("block_backtracks") == 3
    and record.get("block_safeguard") is False
    and math.isclose(record.get("block_overrelaxation", 1.0),
                     expected_overrelaxation, rel_tol=0.0, abs_tol=1e-14)
    for record in (records[0], records[-1]))
raise SystemExit(not (len(records) == expected_epochs
                      and records[-1]["epoch"] == expected_epochs - 1
                      and configuration_matches))
PY
    then
        echo "Skipping completed $file_name"
    else
        rm -f "$result_file" "$state_file"
        "$PYTHON" -u palm_ba.py "$base_url" "$file_name" \
            "$EPOCHS" "$PARTITIONS" "${COMMON_ARGS[@]}" \
            --output "$result_file" > "$log_file" 2>&1
        exit_code=$?
        if [[ $exit_code -eq 132 || $exit_code -eq 139 ]]; then
            echo "Retrying $file_name after native crash (exit $exit_code)"
            rm -f "$result_file" "$state_file"
            "$PYTHON" -u palm_ba.py "$base_url" "$file_name" \
                "$EPOCHS" "$PARTITIONS" "${COMMON_ARGS[@]}" \
                --output "$result_file" > "$log_file" 2>&1
            exit_code=$?
        fi
        if [[ $exit_code -ne 0 ]]; then
            status=failed
            ((FAILURES += 1))
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
    "globalJacobi": best_record.get("global_jacobi", "none"),
    "gpuStateCache": best_record.get("gpu_state_cache", False),
    "localSolver": best_record["local_solver"],
    "partitioner": best_record["partitioner"],
    "repartitionEvery": best_record.get("repartition_every", 0),
    "repartitionCandidates": best_record.get("repartition_candidates", 4),
    "partitionSeed": best_record.get("partition_seed", 0),
    "blockOverrelaxation": best_record.get("block_overrelaxation", 1.0),
    "blockSafeguard": best_record.get("block_safeguard", True),
    "trajectory": trajectory_path,
}

FAILURES=0
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

if ((FAILURES > 0)); then
    echo "$FAILURES PALM problem(s) failed" >&2
    exit 1
fi
