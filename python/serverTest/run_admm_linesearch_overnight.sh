#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)

export ALL_PROBLEMS=1
export ITERATIONS=${ITERATIONS:-30}
export LOCAL_STEPS=${LOCAL_STEPS:-1}
export THREADS_PER_CLUSTER=${THREADS_PER_CLUSTER:-1}
export CLUSTERS_LIST=${CLUSTERS_LIST:-"10 20 30"}
export MODE_LIST=${MODE_LIST:-"nesterov_daba_tr_linesearch_guard nesterov_daba_tr_catastrophic_guard"}
export CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-3600}
export OUTPUT_DIR=${OUTPUT_DIR:-"$WORKSPACE/benchmark_results/admm_linesearch_overnight_i30_k10_k20_k30"}

exec "$SCRIPT_DIR/run_admm_innovation_matrix.sh"