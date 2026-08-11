#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNNER="$SCRIPT_DIR/run_stage_c_final_benchmarks.sh"
ANALYZER="$SCRIPT_DIR/analyze_stage_c_scaling_pilot.py"
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
K_VALUES=${K_VALUES:-"2 4 8 16 24"}
ITERATIONS=${ITERATIONS:-30}
C5_START_ITERATION=${C5_START_ITERATION:-5}
THREADS_PER_CLUSTER=${THREADS_PER_CLUSTER:-1}
ONE_D_SFM_DATASET_LIST_FILE=${ONE_D_SFM_DATASET_LIST_FILE:-"$SCRIPT_DIR/1dsfm_roman_trafalgar_datasets.txt"}
ONE_D_SFM_PROBLEM_FILTER=${ONE_D_SFM_PROBLEM_FILTER:-"roman_forum trafalgar"}
BAL_PROBLEM_FILTER=${BAL_PROBLEM_FILTER:-"52 3068"}
REQUEST_PORT=${REQUEST_PORT:-35220}
OVERWRITE=${OVERWRITE:-0}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-14400}

if [[ ! "$ITERATIONS" =~ ^[0-9]+$ ]] || (( ITERATIONS <= C5_START_ITERATION )); then
  echo "ITERATIONS must be greater than C5_START_ITERATION=$C5_START_ITERATION" >&2
  exit 2
fi

read -r -a clusters <<< "$K_VALUES"
if (( ${#clusters[@]} < 2 )); then
  echo "K_VALUES must contain at least two cluster counts" >&2
  exit 2
fi
declare -A seen_clusters=()
for clusters_value in "${clusters[@]}"; do
  if [[ ! "$clusters_value" =~ ^[0-9]+$ ]] || (( clusters_value <= 1 )); then
    echo "K_VALUES must contain only integers greater than one" >&2
    exit 2
  fi
  if [[ -n "${seen_clusters[$clusters_value]:-}" ]]; then
    echo "K_VALUES contains duplicate K=$clusters_value" >&2
    exit 2
  fi
  seen_clusters[$clusters_value]=1
done

cluster_tag=$(IFS=_; echo "${clusters[*]}")
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/stage_c_scaling_pilot_k${cluster_tag}_i${ITERATIONS}"}

env \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  FAMILIES=1dsfm MODES=c1_c5 \
  DATASET_LIST_FILE="$ONE_D_SFM_DATASET_LIST_FILE" \
  PROBLEM_FILTER="$ONE_D_SFM_PROBLEM_FILTER" \
  CLUSTERS="$K_VALUES" ITERATIONS="$ITERATIONS" \
  C5_START_ITERATION="$C5_START_ITERATION" \
  THREADS_PER_CLUSTER="$THREADS_PER_CLUSTER" \
  HUBER_DELTA=0 REQUEST_PORT="$REQUEST_PORT" \
  OVERWRITE="$OVERWRITE" CASE_TIMEOUT_SECONDS="$CASE_TIMEOUT_SECONDS" \
  "$RUNNER"

env \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  FAMILIES=bal MODES=c1_c5 \
  DATASET_LIST_FILE= PROBLEM_FILTER="$BAL_PROBLEM_FILTER" \
  CLUSTERS="$K_VALUES" ITERATIONS="$ITERATIONS" \
  C5_START_ITERATION="$C5_START_ITERATION" \
  THREADS_PER_CLUSTER="$THREADS_PER_CLUSTER" \
  HUBER_DELTA=0 REQUEST_PORT="$((REQUEST_PORT + 10))" \
  OVERWRITE="$OVERWRITE" CASE_TIMEOUT_SECONDS="$CASE_TIMEOUT_SECONDS" \
  "$RUNNER"

one_d_sfm_expected=
for scene in $ONE_D_SFM_PROBLEM_FILTER; do
  one_d_sfm_expected="${one_d_sfm_expected:+$one_d_sfm_expected,}$scene"
done
bal_expected=
for scene in $BAL_PROBLEM_FILTER; do
  bal_expected="${bal_expected:+$bal_expected,}bal$scene"
done

"$PYTHON" "$ANALYZER" \
  --root "$OUTPUT_ROOT" \
  --clusters "${clusters[@]}" \
  --iterations "$ITERATIONS" \
  --expected-1dsfm "$one_d_sfm_expected" \
  --expected-bal "$bal_expected"

echo "Stage-C scaling pilot finished: $OUTPUT_ROOT"