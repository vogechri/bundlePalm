#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-"$WORKSPACE/benchmark_results/admm_five_scene_i30_k20"}
MODE_LIST=${MODE_LIST:-baseline}

ITERATIONS=30 \
LOCAL_STEPS=1 \
THREADS_PER_CLUSTER=1 \
CLUSTERS_LIST=20 \
PROBLEM_FILTER='1723 52 245 394 871' \
MODE_LIST="$MODE_LIST" \
OUTPUT_DIR="$OUTPUT_DIR" \
  "$SCRIPT_DIR/run_admm_innovation_matrix.sh"

"$SCRIPT_DIR/.venv/bin/python" \
  "$WORKSPACE/analyze_admm_innovation_matrix.py" \
  "$OUTPUT_DIR" \
  --clusters 20 \
  --datasets 1723 52 245 394 871 \
  --checkpoint 30 \
  --output "$OUTPUT_DIR/report.md"

echo "Five-scene K20 ablation finished: $OUTPUT_DIR/report.md"
