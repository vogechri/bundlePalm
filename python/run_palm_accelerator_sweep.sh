#!/usr/bin/env bash

set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

read -r -a ACCELERATORS_TO_RUN <<< \
    "${ACCELERATORS:-none heavy_ball nesterov anderson lbfgs}"
read -r -a OVERRELAXATIONS_TO_RUN <<< \
    "${OVERRELAXATIONS:-1.0488088481701516 1.2}"

run_name_for() {
    local accelerator=$1
    local factor=$2

    case "$factor:$accelerator" in
        1.0488088481701516:nesterov) echo overrelax105 ;;
        1.2:nesterov) echo overrelax12 ;;
        1.2:none) echo overrelax120_none ;;
        1.2:heavy_ball) echo overrelax120_hb ;;
        1.0488088481701516:*) echo "accel_${accelerator}_overrelax105" ;;
        1.2:*) echo "accel_${accelerator}_overrelax120" ;;
        *)
            local factor_label=${factor//./_}
            echo "accel_${accelerator}_overrelax${factor_label}"
            ;;
    esac
}

failures=0
for factor in "${OVERRELAXATIONS_TO_RUN[@]}"; do
    for accelerator in "${ACCELERATORS_TO_RUN[@]}"; do
        run_name=$(run_name_for "$accelerator" "$factor")
        echo "===== $run_name: $accelerator, over-relaxation $factor ====="
        if ! RUN_NAME="$run_name" ACCELERATOR="$accelerator" \
            BLOCK_OVERRELAXATION="$factor" DRY_RUN="${DRY_RUN:-0}" \
            ./run_palm.sh; then
            echo "$run_name failed; continuing with the next configuration" >&2
            ((failures += 1))
        fi
    done
done

if ((failures > 0)); then
    echo "$failures PALM accelerator configuration(s) failed" >&2
    exit 1
fi