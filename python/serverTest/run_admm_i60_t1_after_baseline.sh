#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
OUTPUT_DIR="$WORKSPACE/benchmark_results/admm_innovation_matrix_i60_t1"

if pgrep -f '[c]lient_admm.py' >/dev/null \
    || pgrep -f '[b]uild_admm/zeromq_cpp_server_ex' >/dev/null; then
  echo "An isolated ADMM run is still active; continuation not started." >&2
  exit 3
fi

cmake --build "$SCRIPT_DIR/build_admm" \
  --target zeromq_cpp_server_ex --parallel 2

ITERATIONS=60 \
LOCAL_STEPS=20 \
THREADS_PER_CLUSTER=1 \
CLUSTERS_LIST='10 30' \
MODE_LIST='baseline' \
OUTPUT_DIR="$OUTPUT_DIR" \
  "$SCRIPT_DIR/run_admm_innovation_matrix.sh"

"$SCRIPT_DIR/.venv/bin/python" \
  "$WORKSPACE/analyze_admm_innovation_matrix.py" \
  "$OUTPUT_DIR" --clusters 10 30 --output "$OUTPUT_DIR/report.md"

echo "60-iteration one-core matrix finished: $OUTPUT_DIR/report.md"