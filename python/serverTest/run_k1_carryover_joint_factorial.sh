#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
DETACHED_ROOT=${DETACHED_ROOT:-/home/chvogel/bundlePalm_k1_long/python}
RUNNER=${RUNNER:-"$SCRIPT_DIR/run_drs_failure_top3_live.sh"}
CLIENT=${CLIENT:-"$SCRIPT_DIR/client_drs_k1_bridge.py"}
BASE_CLIENT=${BASE_CLIENT:-"$DETACHED_ROOT/serverTest/client_drs.py"}
PYTHON=${PYTHON:-"$DETACHED_ROOT/serverTest/.venv/bin/python"}
PROTO_BUILD=${PROTO_BUILD:-"$DETACHED_ROOT/serverTest/build"}
WORKER=${WORKER:-"$PROTO_BUILD/zeromq_cpp_server_ex"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$WORKSPACE/benchmark_results/k1_carryover_joint_factorial"}
REQUEST_PORT=${REQUEST_PORT:-31420}
OVERWRITE=${OVERWRITE:-0}
MEMORY_LIMIT_KB=${MEMORY_LIMIT_KB:-14680064}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-14400}
ARMS=${ARMS:-"direct direct_shared direct_proposal direct_shared_proposal"}
ANALYZE=${ANALYZE:-1}
COHORT=${COHORT:-development}

mkdir -p "$OUTPUT_ROOT"
if [[ "$MEMORY_LIMIT_KB" != "0" ]]; then
  ulimit -v "$MEMORY_LIMIT_KB"
fi

manifest="$OUTPUT_ROOT/development_datasets.txt"
one_d_sfm_manifest="$DETACHED_ROOT/serverTest/1dsfm_all_fifteen_datasets.txt"
: > "$manifest"
case "$COHORT" in
  development)
    one_d_sfm_scenes="gendarmenmarkt piccadilly roman_forum trafalgar union_square vienna_cathedral"
    bal_problem_ids="52 245 1490 1778 3068"
    ;;
  all)
    one_d_sfm_scenes=$(cut -d'|' -f1 "$one_d_sfm_manifest" | tr '\n' ' ')
    bal_problem_ids=$(find "$WORKSPACE" -maxdepth 1 -type f \
      -name 'problem-*-pre.txt' -printf '%f\n' \
      | sed -E 's/^problem-([0-9]+)-.*/\1/' | sort -n | tr '\n' ' ')
    ;;
  *)
    echo "Unknown cohort: $COHORT" >&2
    exit 2
    ;;
esac
for scene in $one_d_sfm_scenes; do
  awk -F'|' -v scene="$scene" '$1 == scene { print }' \
    "$one_d_sfm_manifest" >> "$manifest"
done
for problem_id in $bal_problem_ids; do
  dataset=$(find "$WORKSPACE" -maxdepth 1 -type f \
    -name "problem-${problem_id}-*-pre.txt" -print -quit)
  if [[ -z "$dataset" ]]; then
    echo "Missing BAL dataset $problem_id" >&2
    exit 2
  fi
  printf '%s|%s\n' "$problem_id" "$dataset" >> "$manifest"
done
problem_filter=$(cut -d'|' -f1 "$manifest" | tr '\n' ' ')

port=$REQUEST_PORT
for arm in $ARMS; do
  case "$arm" in
    direct)
      shared_only=0
      proposal=0
      ;;
    direct_shared)
      shared_only=1
      proposal=0
      ;;
    direct_proposal)
      shared_only=0
      proposal=1
      ;;
    direct_shared_proposal)
      shared_only=1
      proposal=1
      ;;
    *)
      echo "Unknown factorial arm: $arm" >&2
      exit 2
      ;;
  esac

  if [[ "$proposal" == "1" ]]; then
    proposal_iterations=60,90
    proposal_direction=krylov2
    proposal_rebase=1
    allow_two=1
    landmark_oracle=1
    apply_landmark=1
    repeat_margin=0.01
  else
    proposal_iterations=
    proposal_direction=jacobi
    proposal_rebase=0
    allow_two=0
    landmark_oracle=0
    apply_landmark=0
    repeat_margin=-1
  fi

  env \
    MALLOC_ARENA_MAX=2 \
    CLIENT="$CLIENT" PYTHON="$PYTHON" PROTO_BUILD="$PROTO_BUILD" WORKER="$WORKER" \
    BUNDLE_PALM_BASE_CLIENT_DRS="$BASE_CLIENT" \
    BUNDLE_PALM_K1_BRIDGE_ARM=direct \
    BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS=1 \
    BUNDLE_PALM_DIAGONAL_TRUST_DAMPING=0 \
    BUNDLE_PALM_DISABLE_LANDMARK_PRECONDITIONING=0 \
    BUNDLE_PALM_BAE_TRUST_SCHEDULE=0 \
    BUNDLE_PALM_CUMULATIVE_DIAGONAL_DAMPING=0 \
    BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE=75 \
    BUNDLE_PALM_CAMERA_TRUST_DIAGONAL_SCALE=1e-4 \
    BUNDLE_PALM_INITIAL_TRUST_REGION_RADIUS=10 \
    BUNDLE_PALM_MAXIMUM_TRUST_REGION_RADIUS=10000 \
    BUNDLE_PALM_DABA_INITIAL_TRUST_REGION_CAP=100 \
    OUTPUT_DIR="$OUTPUT_ROOT/$arm" DATASET_LIST_FILE="$manifest" \
    PROBLEM_FILTER="$problem_filter" CLUSTERS_LIST=24 ITERATIONS=120 \
    LOCAL_STEPS=1 THREADS_PER_CLUSTER=1 \
    CAMERA_UPDATE=se3_left LOCAL_SOLVER=nesterov \
    NESTEROV_MAX_ITERATIONS=300 ENHANCED_INNER_MAX_ITERATIONS=300 \
    NESTEROV_MIN_ITERATIONS=1 NESTEROV_STOP_TOLERANCE=1e-2 \
    ENHANCED_INNER_UNTIL=30 TRUST_REGION_POLICY=drs PERSISTENT_TRUST_REGION=1 \
    TRUST_REGION_RECOVERY_RATIO=0.5 SCENE_NORMALIZATION=points_p95 \
    CAMERA_SCALING=jacobi_initial PROXIMAL_METRIC=block CONSENSUS_METRIC=full \
    CONSENSUS_EXECUTION=coordinator SHARED_ONLY_CAMERA_PROXIMAL="$shared_only" \
    BLOCK_REGULARIZATION=1e-4 BLOCK_CURVATURE_MULTIPLIER=0.4 \
    BLOCK_RECOVERY_MODE=curvature MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=64 \
    CURVATURE_DECAY_AFTER=5 CURVATURE_DECAY_RATIO=0.5 \
    METRIC_PROPOSAL_DISAGREEMENT_SCALE=0.5 \
    OUTER_ACCELERATION=nesterov LINE_SEARCH_GRID=0,1 ACCELERATION_RESTART_AFTER=3 \
    ADAPTIVE_LOCAL_DEPTH=0 SAFEGUARD_ANNEALING_ITERATIONS=200 \
    WORKER_OWNED_LANDMARKS=1 WORKER_OWNED_CAMERAS=0 PACKED_REQUEST_BUFFERS=1 \
    ONE_STEP_SCHUR_RESIDUAL_PROPOSAL_ITERATIONS="$proposal_iterations" \
    SCHUR_RESIDUAL_PROPOSAL_DIRECTION="$proposal_direction" \
    ONE_STEP_SCHUR_RESIDUAL_PROPOSAL_REBASE_TRUST_STATE="$proposal_rebase" \
    ALLOW_TWO_SCHUR_RESIDUAL_PROPOSALS="$allow_two" \
    SCHUR_PROPOSAL_LANDMARK_RESPONSE_ORACLE="$landmark_oracle" \
    APPLY_SCHUR_PROPOSAL_LANDMARK_RESPONSE="$apply_landmark" \
    SCHUR_REPEAT_MINIMUM_RELATIVE_DECREASE="$repeat_margin" \
    SCHUR_ALIGNMENT_CAMERA_DAMPING=0.00146484375 \
    SCHUR_ALIGNMENT_LANDMARK_DAMPING=0.00146484375 \
    SHARED_SCHUR_LANDMARK_REFINEMENT_STEPS=3 \
    SHARED_SCHUR_MINIMUM_RELATIVE_DECREASE=1e-3 \
    SHARED_SCHUR_RELATIVE_TOLERANCE=1e-6 \
    SCHUR_ALIGNMENT_MAXIMUM_ITERATIONS=5000 \
    SHARED_SCHUR_OPERATOR=bsr_low_memory SHARED_SCHUR_PRECONDITIONER=jacobi \
    DEBUG_OUTPUT=1 LIVE_OUTPUT=0 WORKER_LIVE_OUTPUT=0 \
    OVERWRITE="$OVERWRITE" CASE_TIMEOUT_SECONDS="$CASE_TIMEOUT_SECONDS" \
    VARIANT_NAME_OVERRIDE="$arm" \
    BUNDLE_PALM_REQUEST_PORT="$port" BUNDLE_PALM_RESULT_PORT="$((port + 1))" \
    "$RUNNER"
  port=$((port + 2))
done

if [[ "$ANALYZE" == "1" ]]; then
  "$SCRIPT_DIR/.venv/bin/python" \
    "$SCRIPT_DIR/analyze_k1_carryover_joint_factorial.py" \
    --root "$OUTPUT_ROOT" --cohort "$COHORT"
fi

echo "K1 carryover joint factorial finished: $OUTPUT_ROOT"
