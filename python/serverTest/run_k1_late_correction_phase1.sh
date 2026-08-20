#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
DETACHED_ROOT=${DETACHED_ROOT:-/home/chvogel/bundlePalm_k1_long/python}
RUNNER=${RUNNER:-"$DETACHED_ROOT/serverTest/run_drs_failure_top3_live.sh"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/k1_carryover_late_correction"}
OUTPUT_DIR=${OUTPUT_DIR:-"$OUTPUT_ROOT/phase1_controls"}
STATE_DIR="$OUTPUT_DIR/initial_states"
DATASET_LIST_FILE="$OUTPUT_DIR/datasets.txt"
REQUEST_PORT=${REQUEST_PORT:-30580}
OVERWRITE=${OVERWRITE:-0}

if [[ ! -x "$RUNNER" ]]; then
  echo "Detached runner is unavailable: $RUNNER" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$STATE_DIR"

one_d_sfm_manifest="$DETACHED_ROOT/serverTest/1dsfm_all_fifteen_datasets.txt"
for scene in roman_forum trafalgar; do
  dataset=$(awk -F'|' -v scene="$scene" '$1 == scene { print $2 }' "$one_d_sfm_manifest")
  state=$(find "$WORKSPACE/benchmark_results/stage_c_shared_c1_no_proposal_k24_i90/1dsfm/c1/states" \
    -maxdepth 1 -name "*_${scene}_k24_i90_l1_t1.npz" -print -quit)
  if [[ -z "$dataset" || ! -f "$dataset" || -z "$state" || ! -f "$state" ]]; then
    echo "Missing dataset or mature state for $scene" >&2
    exit 2
  fi
  ln -sfn "$state" "$STATE_DIR/$scene.npz"
done

for problem_id in 52 3068; do
  dataset=$(find "$WORKSPACE" -maxdepth 1 -type f \
    -name "problem-${problem_id}-*-pre.txt" -print -quit)
  state=$(find "$WORKSPACE/benchmark_results/stage_c_shared_c1_no_proposal_k24_i90/bal/c1/states" \
    -maxdepth 1 -name "*_${problem_id}_k24_i90_l1_t1.npz" -print -quit)
  if [[ -z "$dataset" || ! -f "$dataset" || -z "$state" || ! -f "$state" ]]; then
    echo "Missing dataset or mature state for BAL$problem_id" >&2
    exit 2
  fi
  ln -sfn "$state" "$STATE_DIR/$problem_id.npz"
done

{
  for scene in roman_forum trafalgar; do
    awk -F'|' -v scene="$scene" '$1 == scene { print }' "$one_d_sfm_manifest"
  done
  for problem_id in 52 3068; do
    dataset=$(find "$WORKSPACE" -maxdepth 1 -type f \
      -name "problem-${problem_id}-*-pre.txt" -print -quit)
    printf '%s|%s\n' "$problem_id" "$dataset"
  done
} > "$DATASET_LIST_FILE"

cd "$DETACHED_ROOT/serverTest"
env \
  OUTPUT_DIR="$OUTPUT_DIR" \
  DATASET_LIST_FILE="$DATASET_LIST_FILE" \
  PROBLEM_FILTER="roman_forum trafalgar 52 3068" \
  INITIAL_STATE_DIRECTORY="$STATE_DIR" INITIAL_STATE_FRAME=raw \
  CLUSTERS_LIST=24 ITERATIONS=0 LOCAL_STEPS=1 THREADS_PER_CLUSTER=1 \
  CAMERA_UPDATE=se3_left LOCAL_SOLVER=nesterov \
  NESTEROV_MAX_ITERATIONS=300 NESTEROV_STOP_TOLERANCE=1e-2 \
  TRUST_REGION_POLICY=drs PERSISTENT_TRUST_REGION=1 \
  SCENE_NORMALIZATION=points_p95 CAMERA_SCALING=jacobi_initial \
  PROXIMAL_METRIC=block CONSENSUS_METRIC=full SHARED_ONLY_CAMERA_PROXIMAL=1 \
  BLOCK_REGULARIZATION=1e-4 BLOCK_CURVATURE_MULTIPLIER=0.4 \
  BLOCK_RECOVERY_MODE=curvature MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=64 \
  METRIC_PROPOSAL_DISAGREEMENT_SCALE=1 OUTER_ACCELERATION=none \
  ADAPTIVE_LOCAL_DEPTH=0 INITIAL_SHARED_SCHUR_CORRECTION=0 \
  FINAL_SHARED_SCHUR_CORRECTION=1 SHARED_SCHUR_MAXIMUM_CORRECTIONS=1 \
  SHARED_SCHUR_CAMERA_DAMPING=0.005859375 \
  SHARED_SCHUR_LANDMARK_DAMPING=0.005859375 \
  SHARED_SCHUR_MINIMUM_RELATIVE_DECREASE=1e-3 \
  SHARED_SCHUR_RELATIVE_TOLERANCE=1e-6 \
  SHARED_SCHUR_OPERATOR=bsr_low_memory SHARED_SCHUR_PRECONDITIONER=jacobi \
  BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS=1 \
  BUNDLE_PALM_DISABLE_LANDMARK_PRECONDITIONING=0 \
  BUNDLE_PALM_DIAGONAL_TRUST_DAMPING=0 BUNDLE_PALM_BAE_TRUST_SCHEDULE=0 \
  BUNDLE_PALM_CUMULATIVE_DIAGONAL_DAMPING=0 \
  BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=75 \
  BUNDLE_PALM_INITIAL_TRUST_REGION_RADIUS=10 \
  BUNDLE_PALM_MAXIMUM_TRUST_REGION_RADIUS=1000000 \
  BUNDLE_PALM_DABA_INITIAL_TRUST_REGION_CAP=100 \
  BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
  BUNDLE_PALM_RESULT_PORT="$((REQUEST_PORT + 1))" \
  DEBUG_OUTPUT=1 LIVE_OUTPUT=0 WORKER_LIVE_OUTPUT=0 \
  OVERWRITE="$OVERWRITE" CASE_TIMEOUT_SECONDS=7200 \
  VARIANT_TAG=late_correction_phase1 \
  "$RUNNER"

"$WORKSPACE/serverTest/.venv/bin/python" \
  "$WORKSPACE/serverTest/analyze_k1_late_correction_phase1.py" \
  --root "$OUTPUT_DIR"
