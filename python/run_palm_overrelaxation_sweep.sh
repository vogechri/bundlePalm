#!/usr/bin/env bash

set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

run_factor() {
    local run_name=$1
    local factor=$2

    echo "===== $run_name: block over-relaxation $factor ====="
    if ! RUN_NAME="$run_name" BLOCK_OVERRELAXATION="$factor" ./run_palm.sh; then
        echo "$run_name failed; continuing with the next factor" >&2
        return 1
    fi
}

failures=0
run_factor overrelax105 1.0488088481701516 || ((failures += 1))
run_factor overrelax11 1.10 || ((failures += 1))
run_factor overrelax115 1.15 || ((failures += 1))
run_factor overrelax12 1.20 || ((failures += 1))
run_factor overrelax125 1.25 || ((failures += 1))

if ((failures > 0)); then
    echo "$failures sweep run(s) failed" >&2
    exit 1
fi