#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)

export LIVE_OUTPUT=1
export DEBUG_OUTPUT=1
export PROBLEM_FILTER="646 931 1266"
export CLUSTERS_LIST="10 20 30"
export ITERATIONS=30
export LOCAL_STEPS=1
export THREADS_PER_CLUSTER=1
export MODE_LIST="nesterov_daba_tr_linesearch_guard nesterov_daba_tr_catastrophic_guard"
export OUTPUT_DIR="$WORKSPACE/benchmark_results/admm_failure_top3_i30_k10_k20_k30"

exec "$SCRIPT_DIR/run_admm_linesearch_overnight.sh"
