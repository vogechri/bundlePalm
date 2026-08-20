#!/usr/bin/env bash

# Usage and parameter reference: DRS_RUNNER_USAGE.md

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-"$SCRIPT_DIR/.venv/bin/python"}
CLIENT="$SCRIPT_DIR/client_drs.py"
WORKER=${WORKER:-"$SCRIPT_DIR/build_admm/zeromq_cpp_server_ex"}
PROTO_BUILD=${PROTO_BUILD:-"$SCRIPT_DIR/build_admm"}
OUTPUT_DIR=${OUTPUT_DIR:-"$WORKSPACE/benchmark_results/drs_failure_top3_i30_k10_k20_k30"}
PROBLEM_FILTER=${PROBLEM_FILTER:-"646 931 1266"}
ALL_PROBLEMS=${ALL_PROBLEMS:-0}
DATASET_LIST_FILE=${DATASET_LIST_FILE:-}
INITIAL_STATE_DIRECTORY=${INITIAL_STATE_DIRECTORY:-}
INITIAL_STATE_FRAME=${INITIAL_STATE_FRAME:-raw}
CLUSTERS_LIST=${CLUSTERS_LIST:-"10 20 30"}
SINGLE_CLUSTER_PROXIMAL=${SINGLE_CLUSTER_PROXIMAL:-0}
ITERATIONS=${ITERATIONS:-30}
LOCAL_STEPS=${LOCAL_STEPS:-1}
LOCAL_CAMERA_STEP_SCALE=${LOCAL_CAMERA_STEP_SCALE:-1.0}
LOCAL_CAMERA_STEP_GRID=${LOCAL_CAMERA_STEP_GRID:-1}
SHARED_CAMERA_STEP_GRID=${SHARED_CAMERA_STEP_GRID:-1}
SHARED_CAMERA_STEP_SCALE=${SHARED_CAMERA_STEP_SCALE:-1.0}
SHARED_CAMERA_DISAGREEMENT_SCALE=${SHARED_CAMERA_DISAGREEMENT_SCALE:-1.0}
METRIC_PROPOSAL_DISAGREEMENT_SCALE=${METRIC_PROPOSAL_DISAGREEMENT_SCALE:-1.0}
METRIC_PROPOSAL_DISAGREEMENT_UNTIL=${METRIC_PROPOSAL_DISAGREEMENT_UNTIL:-0}
METRIC_PROPOSAL_DISAGREEMENT_GRID=${METRIC_PROPOSAL_DISAGREEMENT_GRID:-1}
METRIC_PROPOSAL_SUBSPACE_SCALES=${METRIC_PROPOSAL_SUBSPACE_SCALES:-1,1,1}
METRIC_PROPOSAL_DISAGREEMENT_THRESHOLD=${METRIC_PROPOSAL_DISAGREEMENT_THRESHOLD:--1.0}
METRIC_PROPOSAL_DISAGREEMENT_HYSTERESIS=${METRIC_PROPOSAL_DISAGREEMENT_HYSTERESIS:-}
CAMERA_DIAGONAL_QUANTILE_ITERATIONS=${CAMERA_DIAGONAL_QUANTILE_ITERATIONS:-}
CAMERA_DISAGREEMENT_DIAGNOSTIC_IDS=${CAMERA_DISAGREEMENT_DIAGNOSTIC_IDS:-}
CAMERA_DISAGREEMENT_DIAGNOSTIC_ITERATIONS=${CAMERA_DISAGREEMENT_DIAGNOSTIC_ITERATIONS:-}
SHARED_CAMERA_METRIC_BETA=${SHARED_CAMERA_METRIC_BETA:-0.0}
ADAPTIVE_LOCAL_DEPTH=${ADAPTIVE_LOCAL_DEPTH:-0}
ADAPTIVE_LOCAL_DEPTH_START=${ADAPTIVE_LOCAL_DEPTH_START:-0}
ADAPTIVE_LOCAL_DEPTH_MAXIMUM=${ADAPTIVE_LOCAL_DEPTH_MAXIMUM:-2}
ADAPTIVE_LOCAL_DEPTH_HIGH=${ADAPTIVE_LOCAL_DEPTH_HIGH:-20}
ADAPTIVE_LOCAL_DEPTH_LOW=${ADAPTIVE_LOCAL_DEPTH_LOW:-5}
ADAPTIVE_LOCAL_DEPTH_WINDOW=${ADAPTIVE_LOCAL_DEPTH_WINDOW:-3}
ADAPTIVE_LOCAL_DEPTH_DWELL=${ADAPTIVE_LOCAL_DEPTH_DWELL:-3}
THREADS_PER_CLUSTER=${THREADS_PER_CLUSTER:-1}
NESTEROV_MAX_ITERATIONS=${NESTEROV_MAX_ITERATIONS:-100}
ENHANCED_INNER_MAX_ITERATIONS=${ENHANCED_INNER_MAX_ITERATIONS:-300}
NESTEROV_MIN_ITERATIONS=${NESTEROV_MIN_ITERATIONS:-1}
NESTEROV_STOP_TOLERANCE=${NESTEROV_STOP_TOLERANCE:-1e-2}
ENHANCED_INNER_UNTIL=${ENHANCED_INNER_UNTIL:-0}
DIAGONAL_TRUST_UNTIL=${DIAGONAL_TRUST_UNTIL:-0}
RELATIVE_RESIDUAL_UNTIL=${RELATIVE_RESIDUAL_UNTIL:-0}
PARTITION_CACHE=${PARTITION_CACHE:-auto}
PARTITION_CACHE_DIRECTORY=${PARTITION_CACHE_DIRECTORY:-$HOME/.cache/bundle_palm/partitions}
LOCAL_SOLVER=${LOCAL_SOLVER:-nesterov}
TRUST_REGION_POLICY=${TRUST_REGION_POLICY:-daba}
PERSISTENT_TRUST_REGION=${PERSISTENT_TRUST_REGION:-0}
TRUST_REGION_RECOVERY_RATIO=${TRUST_REGION_RECOVERY_RATIO:-0.5}
SHARED_TRUST_REGION_UNTIL=${SHARED_TRUST_REGION_UNTIL:-0}
COLLECTIVE_TRUST_TRIAL_UNTIL=${COLLECTIVE_TRUST_TRIAL_UNTIL:-0}
SHARED_TRUST_REGION_INITIAL_RADIUS=${SHARED_TRUST_REGION_INITIAL_RADIUS:-1000000}
LOCAL_STATE_REBASE_ITERATION=${LOCAL_STATE_REBASE_ITERATION:-0}
SCENE_NORMALIZATION=${SCENE_NORMALIZATION:-points_p95}
CAMERA_SCALING=${CAMERA_SCALING:-jacobi_initial}
CAMERA_SCALING_MAXIMUM_RATIO=${CAMERA_SCALING_MAXIMUM_RATIO:-}
CAMERA_SCALING_CLIPPING_PERCENTILE=${CAMERA_SCALING_CLIPPING_PERCENTILE:-}
CAMERA_DIAGONAL_RELATIVE_FLOOR=${BUNDLE_PALM_CAMERA_DIAGONAL_FLOOR:-1e-48}
CONSENSUS_UNFLOORED_CAMERA_DIAGONAL=${BUNDLE_PALM_CONSENSUS_UNFLOORED_CAMERA_DIAGONAL:-0}
CAMERA_TRUST_DIAGONAL_SCALE=${BUNDLE_PALM_CAMERA_TRUST_DIAGONAL_SCALE:-1e-4}
CAMERA_DIAGONAL_METRIC_SCALE=${BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE:-25}
CLUSTERING=${CLUSTERING:-${BUNDLE_PALM_CLUSTERING:-landmark_scalable}}
RESIDUAL_BALANCE_SLACK=${RESIDUAL_BALANCE_SLACK:-0.01}
MINIMUM_CAMERA_LANDMARKS=${MINIMUM_CAMERA_LANDMARKS:-20}
MAX_REFINEMENT_PASSES=${MAX_REFINEMENT_PASSES:-3}
PROXIMAL_METRIC=${PROXIMAL_METRIC:-block}
CONSENSUS_METRIC=${CONSENSUS_METRIC:-${BUNDLE_PALM_DRS_CONSENSUS_METRIC:-full}}
CONSENSUS_EXECUTION=${CONSENSUS_EXECUTION:-coordinator}
CONSENSUS_SHARED_FLOOR_PRIOR_SCALE=${CONSENSUS_SHARED_FLOOR_PRIOR_SCALE:-0}
BLOCK_REGULARIZATION=${BLOCK_REGULARIZATION:-5e-5}
BLOCK_CURVATURE_MULTIPLIER=${BLOCK_CURVATURE_MULTIPLIER:-0}
BLOCK_RECOVERY_MODE=${BLOCK_RECOVERY_MODE:-regularization}
MAXIMUM_BLOCK_CURVATURE_MULTIPLIER=${MAXIMUM_BLOCK_CURVATURE_MULTIPLIER:-16.0}
RECOVERY_EXHAUSTION_POLICY=${RECOVERY_EXHAUSTION_POLICY:-stop}
RECOVERY_RELAXED_ITERATIONS=${RECOVERY_RELAXED_ITERATIONS:-1}
CURVATURE_DECAY_AFTER=${CURVATURE_DECAY_AFTER:-0}
CURVATURE_DECAY_RATIO=${CURVATURE_DECAY_RATIO:-0.5}
METRIC_DIAGNOSTIC_ITERATIONS=${METRIC_DIAGNOSTIC_ITERATIONS:-0}
WORKER_SSE_SHADOW=${WORKER_SSE_SHADOW:-0}
SUPPRESS_ACCELERATED_LANDMARK_REPLIES=${SUPPRESS_ACCELERATED_LANDMARK_REPLIES:-0}
WORKER_OWNED_LANDMARKS=${WORKER_OWNED_LANDMARKS:-1}
WORKER_OWNED_CAMERAS=${WORKER_OWNED_CAMERAS:-1}
WORKER_CONSENSUS_SHADOW=${WORKER_CONSENSUS_SHADOW:-0}
PACKED_REQUEST_BUFFERS=${PACKED_REQUEST_BUFFERS:-1}
SHARED_ONLY_CAMERA_PROXIMAL=${SHARED_ONLY_CAMERA_PROXIMAL:-0}
INITIAL_SHARED_SCHUR_CORRECTION=${INITIAL_SHARED_SCHUR_CORRECTION:-0}
INITIAL_SHARED_SCHUR_MAXIMUM_CORRECTIONS=${INITIAL_SHARED_SCHUR_MAXIMUM_CORRECTIONS:-1}
INITIAL_SHARED_SCHUR_MAXIMUM_ITERATIONS=${INITIAL_SHARED_SCHUR_MAXIMUM_ITERATIONS:-0}
INITIAL_SHARED_SCHUR_OPERATOR=${INITIAL_SHARED_SCHUR_OPERATOR:-inherit}
INITIAL_SHARED_SCHUR_DAMPING_POLICY=${INITIAL_SHARED_SCHUR_DAMPING_POLICY:-geometric}
INITIAL_SHARED_SCHUR_MODEL_RATIO_MINIMUM_FACTOR=${INITIAL_SHARED_SCHUR_MODEL_RATIO_MINIMUM_FACTOR:-0.3333333333333333}
MID_SHARED_SCHUR_CORRECTION_ITERATION=${MID_SHARED_SCHUR_CORRECTION_ITERATION:-0}
MID_SHARED_SCHUR_TRANSPORT_PRODUCT_STATE=${MID_SHARED_SCHUR_TRANSPORT_PRODUCT_STATE:-0}
FINAL_SHARED_SCHUR_CORRECTION=${FINAL_SHARED_SCHUR_CORRECTION:-0}
STOP_AFTER_ITERATION=${STOP_AFTER_ITERATION:-0}
SHARED_SCHUR_MAXIMUM_CORRECTIONS=${SHARED_SCHUR_MAXIMUM_CORRECTIONS:-1}
SHARED_SCHUR_CAMERA_DAMPING=${SHARED_SCHUR_CAMERA_DAMPING:-3}
SHARED_SCHUR_LANDMARK_DAMPING=${SHARED_SCHUR_LANDMARK_DAMPING:-3}
SHARED_SCHUR_MINIMUM_RELATIVE_DECREASE=${SHARED_SCHUR_MINIMUM_RELATIVE_DECREASE:-1e-4}
SHARED_SCHUR_RELATIVE_TOLERANCE=${SHARED_SCHUR_RELATIVE_TOLERANCE:-1e-6}
SHARED_SCHUR_OPERATOR=${SHARED_SCHUR_OPERATOR:-python}
SHARED_SCHUR_PRECONDITIONER=${SHARED_SCHUR_PRECONDITIONER:-jacobi}
VARIANT_TAG=${VARIANT_TAG:-}
LANDMARK_REFINEMENT_STEPS=${LANDMARK_REFINEMENT_STEPS:-0}
CONSENSUS_LANDMARK_REFINEMENT_STEPS=${CONSENSUS_LANDMARK_REFINEMENT_STEPS:-0}
CONSENSUS_LANDMARK_REFINEMENT_POLICY=${CONSENSUS_LANDMARK_REFINEMENT_POLICY:-safeguard}
TARGET_TRANSFORMED_LIPSCHITZ=${TARGET_TRANSFORMED_LIPSCHITZ:-0.475}
MAXIMUM_BLOCK_REGULARIZATION=${MAXIMUM_BLOCK_REGULARIZATION:-0.5}
RELAXATION=${RELAXATION:-1.0}
OUTER_ACCELERATION=${OUTER_ACCELERATION:-none}
OUTER_ACCELERATION_UNTIL=${OUTER_ACCELERATION_UNTIL:-0}
OUTER_ACCELERATION_RESTART_ITERATION=${OUTER_ACCELERATION_RESTART_ITERATION:-0}
LINE_SEARCH_GRID=${LINE_SEARCH_GRID:-0,1}
ACCELERATION_RESTART_AFTER=${ACCELERATION_RESTART_AFTER:-3}
PENALTY_MULTIPLIER=${PENALTY_MULTIPLIER:-1.0}
HUBER_DELTA=${HUBER_DELTA:-0}
SAFEGUARD_MODE=${SAFEGUARD_MODE:-relative}
DRE_RELATIVE_INCREASE=${DRE_RELATIVE_INCREASE:-0.01}
MINIMUM_PRIMAL_RATIO=${MINIMUM_PRIMAL_RATIO:-1.001}
SAFEGUARD_ANNEALING_ITERATIONS=${SAFEGUARD_ANNEALING_ITERATIONS:-0}
SAFEGUARD_REFERENCE_ITERATION=${SAFEGUARD_REFERENCE_ITERATION:-5}
SAFEGUARD_ANNEALING_EXPONENT=${SAFEGUARD_ANNEALING_EXPONENT:-4}
SAFEGUARD_RELATIVE_DEADBAND=${SAFEGUARD_RELATIVE_DEADBAND:-0}
CATASTROPHIC_RATIO=${CATASTROPHIC_RATIO:-${SAFEGUARD_RATIO:-1000000}}
RECOVERY_PENALTY_RATIO=${RECOVERY_PENALTY_RATIO:-2.0}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-3600}
LIVE_OUTPUT=${LIVE_OUTPUT:-1}
WORKER_LIVE_OUTPUT=${WORKER_LIVE_OUTPUT:-0}
DEBUG_OUTPUT=${DEBUG_OUTPUT:-1}
OVERWRITE=${OVERWRITE:-0}
REQUEST_PORT=${BUNDLE_PALM_REQUEST_PORT:-6656}
RESULT_PORT=${BUNDLE_PALM_RESULT_PORT:-6657}
CAMERA_UPDATE=${CAMERA_UPDATE:-${BUNDLE_PALM_CAMERA_UPDATE:-additive}}
if [[ "$SINGLE_CLUSTER_PROXIMAL" == "1" ]]; then
  CLUSTERS_LIST=1
  OUTER_ACCELERATION=none
  RELAXATION=1.0
  SAFEGUARD_MODE=none
  CONSENSUS_EXECUTION=coordinator
  CONSENSUS_LANDMARK_REFINEMENT_STEPS=0
  WORKER_OWNED_LANDMARKS=0
  WORKER_OWNED_CAMERAS=0
fi
VARIANT_NAME="plain_drs_${PROXIMAL_METRIC}_${CONSENSUS_METRIC}"
if [[ "$SINGLE_CLUSTER_PROXIMAL" == "1" ]]; then
  VARIANT_NAME="proximal_point_${PROXIMAL_METRIC}"
fi
if [[ "$LOCAL_SOLVER" == "ceres_se3" ]]; then
  VARIANT_NAME="centralized_ceres_se3_inner"
elif [[ "$LOCAL_SOLVER" == "ceres_prox_se3" ]]; then
  VARIANT_NAME="ceres_prox_se3_${PROXIMAL_METRIC}_${CONSENSUS_METRIC}"
fi
if [[ "$OUTER_ACCELERATION" != "none" ]]; then
  grid_name=${LINE_SEARCH_GRID//,/}
  grid_name=${grid_name//./p}
  VARIANT_NAME="${OUTER_ACCELERATION}_ls${grid_name}_${PROXIMAL_METRIC}_${CONSENSUS_METRIC}"
fi
if [[ "$OUTER_ACCELERATION_UNTIL" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_accel_until${OUTER_ACCELERATION_UNTIL}"
fi
if [[ "$OUTER_ACCELERATION_RESTART_ITERATION" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_accel_restart${OUTER_ACCELERATION_RESTART_ITERATION}"
fi
if [[ "$RELAXATION" != "1" && "$RELAXATION" != "1.0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_relax${RELAXATION}"
fi
if [[ "$CONSENSUS_EXECUTION" != "coordinator" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_${CONSENSUS_EXECUTION}"
fi
if [[ "$CLUSTERING" != "landmark_scalable" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_${CLUSTERING}"
fi
if [[ "$CAMERA_UPDATE" != "additive" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_${CAMERA_UPDATE}"
fi
if [[ "$SCENE_NORMALIZATION" == "none" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_scene_raw"
elif [[ "$SCENE_NORMALIZATION" != "points_p95" ]]; then
  echo "SCENE_NORMALIZATION must be points_p95 or none" >&2
  exit 2
fi
if [[ "$CAMERA_SCALING" != "jacobi_initial" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_scale_${CAMERA_SCALING}"
fi
if [[ -n "$CAMERA_SCALING_MAXIMUM_RATIO" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_scale_cap${CAMERA_SCALING_MAXIMUM_RATIO}"
fi
if [[ -n "$CAMERA_SCALING_CLIPPING_PERCENTILE" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_scale_clip${CAMERA_SCALING_CLIPPING_PERCENTILE}"
fi
if [[ "$CAMERA_DIAGONAL_RELATIVE_FLOOR" != "1e-48" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_camera_floor${CAMERA_DIAGONAL_RELATIVE_FLOOR}"
fi
if [[ "$CAMERA_TRUST_DIAGONAL_SCALE" != "1e-4" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_trust_diag${CAMERA_TRUST_DIAGONAL_SCALE}"
fi
if [[ "$CAMERA_DIAGONAL_METRIC_SCALE" != "25" && "$CAMERA_DIAGONAL_METRIC_SCALE" != "25.0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_diag_metric${CAMERA_DIAGONAL_METRIC_SCALE}"
fi
if [[ "$SAFEGUARD_RELATIVE_DEADBAND" != "0" && "$SAFEGUARD_RELATIVE_DEADBAND" != "0.0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_guard_db${SAFEGUARD_RELATIVE_DEADBAND}"
fi
if [[ "$BLOCK_CURVATURE_MULTIPLIER" != "0" && "$BLOCK_CURVATURE_MULTIPLIER" != "0.0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_lip${BLOCK_CURVATURE_MULTIPLIER}"
fi
if [[ "$SHARED_CAMERA_METRIC_BETA" != "0" && "$SHARED_CAMERA_METRIC_BETA" != "0.0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_shared_metric_beta${SHARED_CAMERA_METRIC_BETA}"
fi
if [[ "$SHARED_CAMERA_DISAGREEMENT_SCALE" != "1" && "$SHARED_CAMERA_DISAGREEMENT_SCALE" != "1.0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_shared_disagreement${SHARED_CAMERA_DISAGREEMENT_SCALE}"
fi
if [[ "$METRIC_PROPOSAL_DISAGREEMENT_SCALE" != "1" && "$METRIC_PROPOSAL_DISAGREEMENT_SCALE" != "1.0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_metric_proposal${METRIC_PROPOSAL_DISAGREEMENT_SCALE}"
fi
if [[ "$METRIC_PROPOSAL_DISAGREEMENT_UNTIL" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_until${METRIC_PROPOSAL_DISAGREEMENT_UNTIL}"
fi
if [[ "$METRIC_PROPOSAL_DISAGREEMENT_GRID" != "1" ]]; then
  grid_name=${METRIC_PROPOSAL_DISAGREEMENT_GRID//,/}
  grid_name=${grid_name//./p}
  VARIANT_NAME="${VARIANT_NAME}_metric_proposal_grid${grid_name}"
fi
if [[ "$METRIC_PROPOSAL_SUBSPACE_SCALES" != "1,1,1" ]]; then
  subspace_name=${METRIC_PROPOSAL_SUBSPACE_SCALES//,/}
  subspace_name=${subspace_name//./p}
  VARIANT_NAME="${VARIANT_NAME}_metric_subspace${subspace_name}"
fi
if [[ "$METRIC_PROPOSAL_DISAGREEMENT_THRESHOLD" != "-1" && "$METRIC_PROPOSAL_DISAGREEMENT_THRESHOLD" != "-1.0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_threshold${METRIC_PROPOSAL_DISAGREEMENT_THRESHOLD}"
fi
if [[ -n "$METRIC_PROPOSAL_DISAGREEMENT_HYSTERESIS" ]]; then
  hysteresis_name=${METRIC_PROPOSAL_DISAGREEMENT_HYSTERESIS//,/}
  hysteresis_name=${hysteresis_name//./p}
  VARIANT_NAME="${VARIANT_NAME}_hysteresis${hysteresis_name}"
fi
if [[ "$BLOCK_RECOVERY_MODE" != "regularization" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_${BLOCK_RECOVERY_MODE}"
fi
if [[ "$PERSISTENT_TRUST_REGION" == "1" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_persistent_tr"
fi
if [[ "$TRUST_REGION_POLICY" != "daba" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_trust_${TRUST_REGION_POLICY}"
fi
if [[ "$LOCAL_STATE_REBASE_ITERATION" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_rebase${LOCAL_STATE_REBASE_ITERATION}"
fi
if [[ "$SHARED_TRUST_REGION_UNTIL" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_shared_tr${SHARED_TRUST_REGION_UNTIL}"
fi
if [[ "$COLLECTIVE_TRUST_TRIAL_UNTIL" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_collective_tr${COLLECTIVE_TRUST_TRIAL_UNTIL}"
fi
if [[ "$ENHANCED_INNER_UNTIL" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_enhanced${ENHANCED_INNER_UNTIL}"
else
  if [[ "$DIAGONAL_TRUST_UNTIL" != "0" ]]; then
    VARIANT_NAME="${VARIANT_NAME}_diag_until${DIAGONAL_TRUST_UNTIL}"
  fi
  if [[ "$RELATIVE_RESIDUAL_UNTIL" != "0" ]]; then
    VARIANT_NAME="${VARIANT_NAME}_relres_until${RELATIVE_RESIDUAL_UNTIL}"
  fi
fi
if [[ "$CURVATURE_DECAY_AFTER" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_decay${CURVATURE_DECAY_AFTER}"
fi
if [[ "$METRIC_DIAGNOSTIC_ITERATIONS" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_metricdiag${METRIC_DIAGNOSTIC_ITERATIONS}"
fi
if [[ "$ADAPTIVE_LOCAL_DEPTH" == "1" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_adaptl${LOCAL_STEPS}-${ADAPTIVE_LOCAL_DEPTH_MAXIMUM}_h${ADAPTIVE_LOCAL_DEPTH_HIGH}_l${ADAPTIVE_LOCAL_DEPTH_LOW}_w${ADAPTIVE_LOCAL_DEPTH_WINDOW}"
  if [[ "$ADAPTIVE_LOCAL_DEPTH_DWELL" != "3" ]]; then
    VARIANT_NAME="${VARIANT_NAME}_d${ADAPTIVE_LOCAL_DEPTH_DWELL}"
  fi
  if [[ "$ADAPTIVE_LOCAL_DEPTH_START" != "0" ]]; then
    VARIANT_NAME="${VARIANT_NAME}_start${ADAPTIVE_LOCAL_DEPTH_START}"
  fi
fi
if [[ "$LANDMARK_REFINEMENT_STEPS" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_lmref${LANDMARK_REFINEMENT_STEPS}"
fi
if [[ "$CONSENSUS_LANDMARK_REFINEMENT_STEPS" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_vlmref${CONSENSUS_LANDMARK_REFINEMENT_STEPS}"
fi
if [[ "$CONSENSUS_LANDMARK_REFINEMENT_POLICY" != "safeguard" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_${CONSENSUS_LANDMARK_REFINEMENT_POLICY}"
fi
if [[ "$HUBER_DELTA" != "0" && "$HUBER_DELTA" != "0.0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_huber${HUBER_DELTA}"
fi
if [[ "$INITIAL_SHARED_SCHUR_CORRECTION" == "1" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_initial_schur${INITIAL_SHARED_SCHUR_MAXIMUM_CORRECTIONS}"
  if [[ "$INITIAL_SHARED_SCHUR_DAMPING_POLICY" != "geometric" ]]; then
    VARIANT_NAME="${VARIANT_NAME}_${INITIAL_SHARED_SCHUR_DAMPING_POLICY}"
    if [[ "$INITIAL_SHARED_SCHUR_MODEL_RATIO_MINIMUM_FACTOR" != "0.3333333333333333" ]]; then
      VARIANT_NAME="${VARIANT_NAME}_floor${INITIAL_SHARED_SCHUR_MODEL_RATIO_MINIMUM_FACTOR}"
    fi
  fi
  if [[ "$INITIAL_SHARED_SCHUR_OPERATOR" != "inherit" ]]; then
    VARIANT_NAME="${VARIANT_NAME}_${INITIAL_SHARED_SCHUR_OPERATOR}"
  fi
fi
if [[ "$MID_SHARED_SCHUR_CORRECTION_ITERATION" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_mid_schur${MID_SHARED_SCHUR_CORRECTION_ITERATION}"
  if [[ "$MID_SHARED_SCHUR_TRANSPORT_PRODUCT_STATE" == "1" ]]; then
    VARIANT_NAME="${VARIANT_NAME}_transport"
  fi
fi
if [[ "$FINAL_SHARED_SCHUR_CORRECTION" == "1" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_final_schur${SHARED_SCHUR_MAXIMUM_CORRECTIONS}"
  if [[ "$SHARED_SCHUR_CAMERA_DAMPING" != "3" || "$SHARED_SCHUR_LANDMARK_DAMPING" != "3" ]]; then
    VARIANT_NAME="${VARIANT_NAME}_damping${SHARED_SCHUR_CAMERA_DAMPING}_${SHARED_SCHUR_LANDMARK_DAMPING}"
  fi
  if [[ "$SHARED_SCHUR_OPERATOR" != "python" ]]; then
    VARIANT_NAME="${VARIANT_NAME}_${SHARED_SCHUR_OPERATOR}"
  fi
  if [[ "$SHARED_SCHUR_PRECONDITIONER" != "jacobi" ]]; then
    VARIANT_NAME="${VARIANT_NAME}_${SHARED_SCHUR_PRECONDITIONER}"
  fi
  if [[ "$SHARED_SCHUR_MINIMUM_RELATIVE_DECREASE" != "1e-4" ]]; then
    VARIANT_NAME="${VARIANT_NAME}_stop${SHARED_SCHUR_MINIMUM_RELATIVE_DECREASE}"
  fi
  if [[ "$SHARED_SCHUR_RELATIVE_TOLERANCE" != "1e-6" ]]; then
    VARIANT_NAME="${VARIANT_NAME}_rtol${SHARED_SCHUR_RELATIVE_TOLERANCE}"
  fi
fi
if [[ "$STOP_AFTER_ITERATION" != "0" ]]; then
  VARIANT_NAME="${VARIANT_NAME}_stop${STOP_AFTER_ITERATION}"
fi
if [[ -n "$VARIANT_TAG" ]]; then
  if [[ ! "$VARIANT_TAG" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "VARIANT_TAG may contain only letters, digits, dots, underscores, and hyphens" >&2
    exit 2
  fi
  VARIANT_NAME="${VARIANT_NAME}_${VARIANT_TAG}"
fi
RESULT_FILE="$OUTPUT_DIR/${VARIANT_NAME}.jsonl"
STATUS_FILE="$OUTPUT_DIR/status.tsv"
WORKER_PID=""

if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter is not executable: $PYTHON" >&2
  echo "Set PYTHON to a working environment with numpy, pyzmq, and torch." >&2
  exit 2
fi
if [[ ! -x "$WORKER" ]]; then
  echo "DRS worker is not executable: $WORKER" >&2
  echo "Run ./build_local.sh or set WORKER to a built server." >&2
  exit 2
fi

for flag in LIVE_OUTPUT DEBUG_OUTPUT OVERWRITE PERSISTENT_TRUST_REGION ALL_PROBLEMS WORKER_SSE_SHADOW SUPPRESS_ACCELERATED_LANDMARK_REPLIES WORKER_OWNED_LANDMARKS WORKER_OWNED_CAMERAS WORKER_CONSENSUS_SHADOW PACKED_REQUEST_BUFFERS SHARED_ONLY_CAMERA_PROXIMAL INITIAL_SHARED_SCHUR_CORRECTION FINAL_SHARED_SCHUR_CORRECTION ADAPTIVE_LOCAL_DEPTH; do
  value=${!flag}
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "$flag must be 0 or 1" >&2
    exit 2
  fi
done
if [[ "$CONSENSUS_EXECUTION" != "coordinator" && "$CONSENSUS_EXECUTION" != "single-node" ]]; then
  echo "CONSENSUS_EXECUTION must be coordinator or single-node" >&2
  exit 2
fi
if [[ "$CAMERA_UPDATE" != "additive" && "$CAMERA_UPDATE" != "angle_axis_left" && "$CAMERA_UPDATE" != "so3_left" && "$CAMERA_UPDATE" != "so3_center_left" && "$CAMERA_UPDATE" != "se3_left" && "$CAMERA_UPDATE" != "se3_right" ]]; then
  echo "CAMERA_UPDATE must be additive, angle_axis_left, so3_left, so3_center_left, se3_left, or se3_right" >&2
  exit 2
fi

PROBLEMS=(
  "52|problem-52-64053-pre.txt"
  "89|problem-89-110973-pre.txt"
  "142|problem-142-93602-pre.txt"
  "245|problem-245-198739-pre.txt"
  "287|problem-287-182023-pre.txt"
  "394|problem-394-100368-pre.txt"
  "646|problem-646-73584-pre.txt"
  "783|problem-783-84444-pre.txt"
  "931|problem-931-102699-pre.txt"
  "1064|problem-1064-113655-pre.txt"
  "1266|problem-1266-132593-pre.txt"
  "1723|problem-1723-156502-pre.txt"
)

if [[ "$ALL_PROBLEMS" == "1" ]]; then
  PROBLEMS=()
  while IFS= read -r dataset; do
    scene=${dataset#problem-}
    scene=${scene%%-*}
    PROBLEMS+=("$scene|$dataset")
  done < <(
    find "$WORKSPACE" -maxdepth 1 -type f -name 'problem-*-pre.txt' \
      -printf '%f\n' | sort -V
  )
  if [[ ${#PROBLEMS[@]} -ne 29 ]]; then
    echo "Expected 29 top-level BAL problems, found ${#PROBLEMS[@]}" >&2
    exit 2
  fi
fi

if [[ -n "$DATASET_LIST_FILE" ]]; then
  if [[ ! -f "$DATASET_LIST_FILE" ]]; then
    echo "DATASET_LIST_FILE does not exist: $DATASET_LIST_FILE" >&2
    exit 2
  fi
  PROBLEMS=()
  while IFS='|' read -r scene dataset extra || [[ -n "${scene}${dataset}${extra}" ]]; do
    [[ -z "$scene" && -z "$dataset" ]] && continue
    if [[ -z "$scene" || -z "$dataset" || -n "$extra" ]]; then
      echo "Invalid DATASET_LIST_FILE row; expected scene|path" >&2
      exit 2
    fi
    PROBLEMS+=("$scene|$dataset")
  done < "$DATASET_LIST_FILE"
  if [[ ${#PROBLEMS[@]} -eq 0 ]]; then
    echo "DATASET_LIST_FILE contains no datasets" >&2
    exit 2
  fi
fi

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/states" "$OUTPUT_DIR/memory"
if [[ ! -f "$STATUS_FILE" ]]; then
  printf 'variant\tdataset\tclusters\titerations\tlocal_steps\tthreads_per_cluster\tstatus\texit_code\telapsed_seconds\tcoordinator_max_rss_kb\tworker_max_rss_kb\n' > "$STATUS_FILE"
fi

cleanup_worker() {
  if [[ -n "$WORKER_PID" ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
    worker_children=$(pgrep -P "$WORKER_PID" || true)
    if [[ -n "$worker_children" ]]; then
      kill -TERM $worker_children 2>/dev/null || true
    else
      kill -TERM "$WORKER_PID" 2>/dev/null || true
    fi
    for _ in {1..50}; do
      kill -0 "$WORKER_PID" 2>/dev/null || break
      read -r -t 0.1 _ || true
    done
    if kill -0 "$WORKER_PID" 2>/dev/null; then
      worker_children=$(pgrep -P "$WORKER_PID" || true)
      if [[ -n "$worker_children" ]]; then
        kill -KILL $worker_children 2>/dev/null || true
      fi
      kill -TERM "$WORKER_PID" 2>/dev/null || true
    fi
    wait "$WORKER_PID" 2>/dev/null || true
  fi
  WORKER_PID=""
}

trap cleanup_worker EXIT
trap 'cleanup_worker; exit 130' INT
trap 'cleanup_worker; exit 143' TERM

case_exists() {
  local dataset=$1 clusters=$2
  "$PYTHON" - "$RESULT_FILE" "$dataset" "$clusters" "$ITERATIONS" "$LOCAL_STEPS" "$THREADS_PER_CLUSTER" "$VARIANT_NAME" <<'PY'
import json, pathlib, sys
path, dataset, clusters, iterations, local_steps, threads, variant = sys.argv[1:]
try:
    lines = open(path, encoding="utf-8")
except FileNotFoundError:
    raise SystemExit(1)
for line in lines:
    row = json.loads(line)
    if (pathlib.Path(row.get("dataset", "")).name == dataset
            and row.get("clusters") == int(clusters)
            and row.get("iterations") == int(iterations)
          and row.get("completedIterations") == int(iterations)
            and row.get("localSteps") == int(local_steps)
            and row.get("threadsPerCluster") == int(threads)
            and row.get("variant") == variant):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

clear_case() {
  local dataset=$1 clusters=$2
  "$PYTHON" - "$RESULT_FILE" "$STATUS_FILE" "$dataset" "$clusters" "$ITERATIONS" "$LOCAL_STEPS" "$THREADS_PER_CLUSTER" "$VARIANT_NAME" <<'PY'
import json, os, pathlib, sys, tempfile
result_path, status_path, dataset, clusters, iterations, local_steps, threads, variant = sys.argv[1:]
key = variant, dataset, int(clusters), int(iterations), int(local_steps), int(threads)

def atomic_filter(path, keep):
    path = pathlib.Path(path)
    if not path.exists(): return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.writelines(line for index, line in enumerate(lines) if keep(index, line))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.remove(temporary)

def keep_result(_, line):
    row = json.loads(line)
    found = (row.get("variant"), pathlib.Path(row.get("dataset", "")).name, row.get("clusters"), row.get("iterations"), row.get("localSteps"), row.get("threadsPerCluster"))
    return found != key

def keep_status(index, line):
    if index == 0: return True
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 6: return True
    found = fields[0], fields[1], int(fields[2]), int(fields[3]), int(fields[4]), int(fields[5])
    return found != key

atomic_filter(result_path, keep_result)
atomic_filter(status_path, keep_status)
PY
}

read_max_rss() {
  awk -F: '/Maximum resident set size/ {gsub(/^[[:space:]]+/, "", $2); print $2}' "$1"
}

for problem in "${PROBLEMS[@]}"; do
  IFS='|' read -r scene dataset <<< "$problem"
  if [[ -n "$PROBLEM_FILTER" && " $PROBLEM_FILTER " != *" $scene "* ]]; then
    continue
  fi
  if [[ "$dataset" = /* ]]; then
    dataset_path=$dataset
  else
    dataset_path="$WORKSPACE/$dataset"
  fi
  if [[ ! -f "$dataset_path" ]]; then
    echo "Dataset does not exist: $dataset_path" >&2
    exit 2
  fi
  dataset_key=$(basename -- "$dataset_path")
  for clusters in $CLUSTERS_LIST; do
    label="${VARIANT_NAME}_${scene}_k${clusters}_i${ITERATIONS}_l${LOCAL_STEPS}_t${THREADS_PER_CLUSTER}"
    log_file="$OUTPUT_DIR/logs/${label}.log"
    worker_log="$OUTPUT_DIR/logs/${label}_worker.log"
    state_file="$OUTPUT_DIR/states/${label}.npz"
    coordinator_time="$OUTPUT_DIR/memory/${label}_coordinator.time"
    worker_time="$OUTPUT_DIR/memory/${label}_worker.time"
    if [[ "$OVERWRITE" == "1" ]]; then
      echo "Overwriting $VARIANT_NAME $dataset K=$clusters"
      clear_case "$dataset_key" "$clusters"
    elif case_exists "$dataset_key" "$clusters"; then
      echo "Skipping completed $VARIANT_NAME $dataset K=$clusters"
      continue
    fi

    echo "Running $VARIANT_NAME $dataset K=$clusters"
    echo "Coordinator log: $log_file"
    echo "Worker log:      $worker_log"
    cleanup_worker
    if [[ "$WORKER_LIVE_OUTPUT" == "1" ]]; then
      (cd "$SCRIPT_DIR" && exec setsid /usr/bin/time -v -o "$worker_time" \
        env BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
            BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
            BUNDLE_PALM_THREADS_PER_CLUSTER="$THREADS_PER_CLUSTER" \
            BUNDLE_PALM_CAMERA_DIAGONAL_FLOOR="$CAMERA_DIAGONAL_RELATIVE_FLOOR" \
            BUNDLE_PALM_CONSENSUS_UNFLOORED_CAMERA_DIAGONAL="$CONSENSUS_UNFLOORED_CAMERA_DIAGONAL" \
            BUNDLE_PALM_CAMERA_TRUST_DIAGONAL_SCALE="$CAMERA_TRUST_DIAGONAL_SCALE" \
            BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE="$CAMERA_DIAGONAL_METRIC_SCALE" \
            BUNDLE_PALM_CAMERA_UPDATE="$CAMERA_UPDATE" \
            "$WORKER") \
        > >(tee "$worker_log") 2>&1 &
    else
      (cd "$SCRIPT_DIR" && exec setsid /usr/bin/time -v -o "$worker_time" \
        env BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
            BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
            BUNDLE_PALM_THREADS_PER_CLUSTER="$THREADS_PER_CLUSTER" \
            BUNDLE_PALM_CAMERA_DIAGONAL_FLOOR="$CAMERA_DIAGONAL_RELATIVE_FLOOR" \
            BUNDLE_PALM_CONSENSUS_UNFLOORED_CAMERA_DIAGONAL="$CONSENSUS_UNFLOORED_CAMERA_DIAGONAL" \
            BUNDLE_PALM_CAMERA_TRUST_DIAGONAL_SCALE="$CAMERA_TRUST_DIAGONAL_SCALE" \
            BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE="$CAMERA_DIAGONAL_METRIC_SCALE" \
            BUNDLE_PALM_CAMERA_UPDATE="$CAMERA_UPDATE" \
            "$WORKER") \
        > "$worker_log" 2>&1 &
    fi
    WORKER_PID=$!
    for _ in {1..600}; do
      grep -q "Starting the server" "$worker_log" 2>/dev/null && break
      kill -0 "$WORKER_PID" 2>/dev/null || break
      read -r -t 0.1 _ || true
    done
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
      echo "DRS worker failed to start" >&2
      exit 1
    fi

    debug_args=()
    [[ "$DEBUG_OUTPUT" == "1" ]] && debug_args+=(--debug-output)
    trust_args=()
    [[ "$PERSISTENT_TRUST_REGION" == "1" ]] && trust_args+=(--persistent-trust-region)
    scaling_args=()
    [[ -n "$CAMERA_SCALING_MAXIMUM_RATIO" ]] && scaling_args+=(--camera-scaling-maximum-ratio "$CAMERA_SCALING_MAXIMUM_RATIO")
    [[ -n "$CAMERA_SCALING_CLIPPING_PERCENTILE" ]] && scaling_args+=(--camera-scaling-clipping-percentile "$CAMERA_SCALING_CLIPPING_PERCENTILE")
    worker_sse_args=()
    [[ "$WORKER_SSE_SHADOW" == "1" ]] && worker_sse_args+=(--worker-sse-shadow)
    [[ "$SUPPRESS_ACCELERATED_LANDMARK_REPLIES" == "1" ]] && worker_sse_args+=(--suppress-accelerated-landmark-replies)
    [[ "$WORKER_OWNED_LANDMARKS" == "1" ]] && worker_sse_args+=(--worker-owned-landmarks)
    [[ "$WORKER_OWNED_CAMERAS" == "1" ]] && worker_sse_args+=(--worker-owned-cameras)
    [[ "$WORKER_CONSENSUS_SHADOW" == "1" ]] && worker_sse_args+=(--worker-consensus-shadow)
    [[ "$PACKED_REQUEST_BUFFERS" == "1" ]] && worker_sse_args+=(--packed-request-buffers)
    adaptive_depth_args=()
    if [[ "$ADAPTIVE_LOCAL_DEPTH" == "1" ]]; then
      adaptive_depth_args+=(
        --adaptive-local-depth
        --adaptive-local-depth-start "$ADAPTIVE_LOCAL_DEPTH_START"
        --adaptive-local-depth-maximum "$ADAPTIVE_LOCAL_DEPTH_MAXIMUM"
        --adaptive-local-depth-high "$ADAPTIVE_LOCAL_DEPTH_HIGH"
        --adaptive-local-depth-low "$ADAPTIVE_LOCAL_DEPTH_LOW"
        --adaptive-local-depth-window "$ADAPTIVE_LOCAL_DEPTH_WINDOW"
        --adaptive-local-depth-dwell "$ADAPTIVE_LOCAL_DEPTH_DWELL"
      )
    fi
    single_cluster_args=()
    [[ "$SINGLE_CLUSTER_PROXIMAL" == "1" ]] && single_cluster_args+=(--single-cluster-proximal)
    shared_only_args=()
    [[ "$SHARED_ONLY_CAMERA_PROXIMAL" == "1" ]] && shared_only_args+=(--shared-only-camera-proximal)
    initial_shared_schur_args=()
    if [[ "$INITIAL_SHARED_SCHUR_CORRECTION" == "1" ]]; then
      initial_shared_schur_args+=(
        --initial-shared-schur-correction
        --initial-shared-schur-maximum-corrections "$INITIAL_SHARED_SCHUR_MAXIMUM_CORRECTIONS"
        --initial-shared-schur-maximum-iterations "$INITIAL_SHARED_SCHUR_MAXIMUM_ITERATIONS"
        --initial-shared-schur-operator "$INITIAL_SHARED_SCHUR_OPERATOR"
        --initial-shared-schur-damping-policy "$INITIAL_SHARED_SCHUR_DAMPING_POLICY"
        --initial-shared-schur-model-ratio-minimum-factor "$INITIAL_SHARED_SCHUR_MODEL_RATIO_MINIMUM_FACTOR"
        --shared-schur-minimum-relative-decrease "$SHARED_SCHUR_MINIMUM_RELATIVE_DECREASE"
        --shared-schur-relative-tolerance "$SHARED_SCHUR_RELATIVE_TOLERANCE"
        --shared-schur-operator "$SHARED_SCHUR_OPERATOR"
        --shared-schur-preconditioner "$SHARED_SCHUR_PRECONDITIONER"
      )
    fi
    mid_shared_schur_args=()
    if [[ "$MID_SHARED_SCHUR_CORRECTION_ITERATION" != "0" ]]; then
      mid_shared_schur_args+=(
        --mid-shared-schur-correction-iteration "$MID_SHARED_SCHUR_CORRECTION_ITERATION"
        --shared-schur-relative-tolerance "$SHARED_SCHUR_RELATIVE_TOLERANCE"
        --shared-schur-operator "$SHARED_SCHUR_OPERATOR"
        --shared-schur-preconditioner "$SHARED_SCHUR_PRECONDITIONER"
      )
      if [[ "$MID_SHARED_SCHUR_TRANSPORT_PRODUCT_STATE" == "1" ]]; then
        mid_shared_schur_args+=(--mid-shared-schur-transport-product-state)
      fi
    fi
    final_shared_schur_args=()
    if [[ "$FINAL_SHARED_SCHUR_CORRECTION" == "1" ]]; then
      final_shared_schur_args+=(
        --final-shared-schur-correction
        --shared-schur-maximum-corrections "$SHARED_SCHUR_MAXIMUM_CORRECTIONS"
        --shared-schur-camera-damping "$SHARED_SCHUR_CAMERA_DAMPING"
        --shared-schur-landmark-damping "$SHARED_SCHUR_LANDMARK_DAMPING"
        --shared-schur-minimum-relative-decrease "$SHARED_SCHUR_MINIMUM_RELATIVE_DECREASE"
        --shared-schur-relative-tolerance "$SHARED_SCHUR_RELATIVE_TOLERANCE"
        --shared-schur-operator "$SHARED_SCHUR_OPERATOR"
        --shared-schur-preconditioner "$SHARED_SCHUR_PRECONDITIONER"
      )
    fi
    initial_state_args=()
    if [[ -n "$INITIAL_STATE_DIRECTORY" ]]; then
      initial_state_path="$INITIAL_STATE_DIRECTORY/${scene}.npz"
      if [[ ! -f "$initial_state_path" ]]; then
        echo "Initial state does not exist: $initial_state_path" >&2
        exit 2
      fi
      initial_state_args+=(
        --initial-state "$initial_state_path"
        --initial-state-frame "$INITIAL_STATE_FRAME"
      )
    fi
    start_seconds=$SECONDS
    set +e
    if [[ "$LIVE_OUTPUT" == "1" ]]; then
      (cd "$SCRIPT_DIR" && /usr/bin/time -v -o "$coordinator_time" \
        timeout --foreground --signal=TERM --kill-after=30 "$CASE_TIMEOUT_SECONDS" \
        env BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
            BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
            BUNDLE_PALM_PROTO_BUILD="$PROTO_BUILD" \
            BUNDLE_PALM_CAMERA_UPDATE="$CAMERA_UPDATE" \
            "$PYTHON" "$CLIENT" "$dataset_path" \
            --variant-name "$VARIANT_NAME" \
            --iterations "$ITERATIONS" --clusters "$clusters" \
            --local-steps "$LOCAL_STEPS" \
            --local-camera-step-scale "$LOCAL_CAMERA_STEP_SCALE" \
            --local-camera-step-grid "$LOCAL_CAMERA_STEP_GRID" \
            --shared-camera-step-grid "$SHARED_CAMERA_STEP_GRID" \
            --shared-camera-step-scale "$SHARED_CAMERA_STEP_SCALE" \
            --shared-camera-disagreement-scale "$SHARED_CAMERA_DISAGREEMENT_SCALE" \
            --metric-proposal-disagreement-scale "$METRIC_PROPOSAL_DISAGREEMENT_SCALE" \
            --metric-proposal-disagreement-until "$METRIC_PROPOSAL_DISAGREEMENT_UNTIL" \
            --metric-proposal-disagreement-grid "$METRIC_PROPOSAL_DISAGREEMENT_GRID" \
            --metric-proposal-subspace-scales "$METRIC_PROPOSAL_SUBSPACE_SCALES" \
            --metric-proposal-disagreement-threshold "$METRIC_PROPOSAL_DISAGREEMENT_THRESHOLD" \
            --metric-proposal-disagreement-hysteresis "$METRIC_PROPOSAL_DISAGREEMENT_HYSTERESIS" \
            --camera-diagonal-quantile-iterations "$CAMERA_DIAGONAL_QUANTILE_ITERATIONS" \
            --camera-disagreement-diagnostic-ids "$CAMERA_DISAGREEMENT_DIAGNOSTIC_IDS" \
            --camera-disagreement-diagnostic-iterations "$CAMERA_DISAGREEMENT_DIAGNOSTIC_ITERATIONS" \
            --shared-camera-metric-beta "$SHARED_CAMERA_METRIC_BETA" \
            --threads-per-cluster "$THREADS_PER_CLUSTER" \
            --nesterov-max-iterations "$NESTEROV_MAX_ITERATIONS" \
            --enhanced-inner-max-iterations "$ENHANCED_INNER_MAX_ITERATIONS" \
            --nesterov-min-iterations "$NESTEROV_MIN_ITERATIONS" \
            --nesterov-stop-tolerance "$NESTEROV_STOP_TOLERANCE" \
            --enhanced-inner-until "$ENHANCED_INNER_UNTIL" \
            --diagonal-trust-until "$DIAGONAL_TRUST_UNTIL" \
            --relative-residual-until "$RELATIVE_RESIDUAL_UNTIL" \
            --stop-after-iteration "$STOP_AFTER_ITERATION" \
            --partition-cache "$PARTITION_CACHE" \
            --partition-cache-directory "$PARTITION_CACHE_DIRECTORY" \
            --local-solver "$LOCAL_SOLVER" \
            --trust-region-policy "$TRUST_REGION_POLICY" \
            --trust-region-recovery-ratio "$TRUST_REGION_RECOVERY_RATIO" \
            --shared-trust-region-until "$SHARED_TRUST_REGION_UNTIL" \
            --collective-trust-trial-until "$COLLECTIVE_TRUST_TRIAL_UNTIL" \
            --shared-trust-region-initial-radius "$SHARED_TRUST_REGION_INITIAL_RADIUS" \
            --local-state-rebase-iteration "$LOCAL_STATE_REBASE_ITERATION" \
            --scene-normalization "$SCENE_NORMALIZATION" \
            --camera-scaling "$CAMERA_SCALING" \
            --camera-diagonal-relative-floor "$CAMERA_DIAGONAL_RELATIVE_FLOOR" \
            --camera-trust-diagonal-scale "$CAMERA_TRUST_DIAGONAL_SCALE" \
            --camera-diagonal-metric-scale "$CAMERA_DIAGONAL_METRIC_SCALE" \
            --clustering "$CLUSTERING" \
            --residual-balance-slack "$RESIDUAL_BALANCE_SLACK" \
            --minimum-camera-landmarks "$MINIMUM_CAMERA_LANDMARKS" \
            --max-refinement-passes "$MAX_REFINEMENT_PASSES" \
            --outer-acceleration "$OUTER_ACCELERATION" \
            --outer-acceleration-until "$OUTER_ACCELERATION_UNTIL" \
            --outer-acceleration-restart-iteration "$OUTER_ACCELERATION_RESTART_ITERATION" \
            --line-search-grid "$LINE_SEARCH_GRID" \
            --acceleration-restart-after "$ACCELERATION_RESTART_AFTER" \
            --proximal-metric "$PROXIMAL_METRIC" \
            --consensus-metric "$CONSENSUS_METRIC" \
            --consensus-execution "$CONSENSUS_EXECUTION" \
            --consensus-shared-floor-prior-scale "$CONSENSUS_SHARED_FLOOR_PRIOR_SCALE" \
            --block-regularization "$BLOCK_REGULARIZATION" \
            --block-curvature-multiplier "$BLOCK_CURVATURE_MULTIPLIER" \
            --block-recovery-mode "$BLOCK_RECOVERY_MODE" \
            --maximum-block-curvature-multiplier "$MAXIMUM_BLOCK_CURVATURE_MULTIPLIER" \
            --recovery-exhaustion-policy "$RECOVERY_EXHAUSTION_POLICY" \
            --recovery-relaxed-iterations "$RECOVERY_RELAXED_ITERATIONS" \
            --curvature-decay-after "$CURVATURE_DECAY_AFTER" \
            --curvature-decay-ratio "$CURVATURE_DECAY_RATIO" \
            --metric-diagnostic-iterations "$METRIC_DIAGNOSTIC_ITERATIONS" \
            --landmark-refinement-steps "$LANDMARK_REFINEMENT_STEPS" \
            --consensus-landmark-refinement-steps "$CONSENSUS_LANDMARK_REFINEMENT_STEPS" \
            --consensus-landmark-refinement-policy "$CONSENSUS_LANDMARK_REFINEMENT_POLICY" \
            --target-transformed-lipschitz "$TARGET_TRANSFORMED_LIPSCHITZ" \
            --maximum-block-regularization "$MAXIMUM_BLOCK_REGULARIZATION" \
            --relaxation "$RELAXATION" \
            --penalty-multiplier "$PENALTY_MULTIPLIER" \
            --huber-delta "$HUBER_DELTA" \
            --safeguard-mode "$SAFEGUARD_MODE" \
            --dre-relative-increase "$DRE_RELATIVE_INCREASE" \
            --minimum-primal-ratio "$MINIMUM_PRIMAL_RATIO" \
            --safeguard-annealing-iterations "$SAFEGUARD_ANNEALING_ITERATIONS" \
            --safeguard-reference-iteration "$SAFEGUARD_REFERENCE_ITERATION" \
            --safeguard-annealing-exponent "$SAFEGUARD_ANNEALING_EXPONENT" \
            --safeguard-relative-deadband "$SAFEGUARD_RELATIVE_DEADBAND" \
            --catastrophic-ratio "$CATASTROPHIC_RATIO" \
            --recovery-penalty-ratio "$RECOVERY_PENALTY_RATIO" \
            --results "$RESULT_FILE" --state "$state_file" \
            "${debug_args[@]}" "${trust_args[@]}" "${scaling_args[@]}" "${worker_sse_args[@]}" "${adaptive_depth_args[@]}" "${single_cluster_args[@]}" "${shared_only_args[@]}" "${initial_shared_schur_args[@]}" "${mid_shared_schur_args[@]}" "${final_shared_schur_args[@]}" "${initial_state_args[@]}") 2>&1 | tee "$log_file"
      exit_code=${PIPESTATUS[0]}
    else
      (cd "$SCRIPT_DIR" && /usr/bin/time -v -o "$coordinator_time" \
        timeout --foreground --signal=TERM --kill-after=30 "$CASE_TIMEOUT_SECONDS" \
        env BUNDLE_PALM_REQUEST_PORT="$REQUEST_PORT" \
            BUNDLE_PALM_RESULT_PORT="$RESULT_PORT" \
            BUNDLE_PALM_PROTO_BUILD="$PROTO_BUILD" \
            BUNDLE_PALM_CAMERA_UPDATE="$CAMERA_UPDATE" \
            "$PYTHON" "$CLIENT" "$dataset_path" \
            --variant-name "$VARIANT_NAME" \
            --iterations "$ITERATIONS" --clusters "$clusters" \
            --local-steps "$LOCAL_STEPS" \
            --local-camera-step-scale "$LOCAL_CAMERA_STEP_SCALE" \
            --local-camera-step-grid "$LOCAL_CAMERA_STEP_GRID" \
            --shared-camera-step-grid "$SHARED_CAMERA_STEP_GRID" \
            --shared-camera-step-scale "$SHARED_CAMERA_STEP_SCALE" \
            --shared-camera-disagreement-scale "$SHARED_CAMERA_DISAGREEMENT_SCALE" \
            --metric-proposal-disagreement-scale "$METRIC_PROPOSAL_DISAGREEMENT_SCALE" \
            --metric-proposal-disagreement-until "$METRIC_PROPOSAL_DISAGREEMENT_UNTIL" \
            --metric-proposal-disagreement-grid "$METRIC_PROPOSAL_DISAGREEMENT_GRID" \
            --metric-proposal-subspace-scales "$METRIC_PROPOSAL_SUBSPACE_SCALES" \
            --metric-proposal-disagreement-threshold "$METRIC_PROPOSAL_DISAGREEMENT_THRESHOLD" \
            --metric-proposal-disagreement-hysteresis "$METRIC_PROPOSAL_DISAGREEMENT_HYSTERESIS" \
            --camera-diagonal-quantile-iterations "$CAMERA_DIAGONAL_QUANTILE_ITERATIONS" \
            --camera-disagreement-diagnostic-ids "$CAMERA_DISAGREEMENT_DIAGNOSTIC_IDS" \
            --camera-disagreement-diagnostic-iterations "$CAMERA_DISAGREEMENT_DIAGNOSTIC_ITERATIONS" \
            --shared-camera-metric-beta "$SHARED_CAMERA_METRIC_BETA" \
            --threads-per-cluster "$THREADS_PER_CLUSTER" \
            --nesterov-max-iterations "$NESTEROV_MAX_ITERATIONS" \
            --enhanced-inner-max-iterations "$ENHANCED_INNER_MAX_ITERATIONS" \
            --nesterov-min-iterations "$NESTEROV_MIN_ITERATIONS" \
            --nesterov-stop-tolerance "$NESTEROV_STOP_TOLERANCE" \
            --enhanced-inner-until "$ENHANCED_INNER_UNTIL" \
            --diagonal-trust-until "$DIAGONAL_TRUST_UNTIL" \
            --relative-residual-until "$RELATIVE_RESIDUAL_UNTIL" \
            --stop-after-iteration "$STOP_AFTER_ITERATION" \
            --partition-cache "$PARTITION_CACHE" \
            --partition-cache-directory "$PARTITION_CACHE_DIRECTORY" \
            --local-solver "$LOCAL_SOLVER" \
            --trust-region-policy "$TRUST_REGION_POLICY" \
            --trust-region-recovery-ratio "$TRUST_REGION_RECOVERY_RATIO" \
            --shared-trust-region-until "$SHARED_TRUST_REGION_UNTIL" \
            --collective-trust-trial-until "$COLLECTIVE_TRUST_TRIAL_UNTIL" \
            --shared-trust-region-initial-radius "$SHARED_TRUST_REGION_INITIAL_RADIUS" \
            --local-state-rebase-iteration "$LOCAL_STATE_REBASE_ITERATION" \
            --scene-normalization "$SCENE_NORMALIZATION" \
            --camera-scaling "$CAMERA_SCALING" \
            --camera-diagonal-relative-floor "$CAMERA_DIAGONAL_RELATIVE_FLOOR" \
            --camera-trust-diagonal-scale "$CAMERA_TRUST_DIAGONAL_SCALE" \
            --camera-diagonal-metric-scale "$CAMERA_DIAGONAL_METRIC_SCALE" \
            --clustering "$CLUSTERING" \
            --residual-balance-slack "$RESIDUAL_BALANCE_SLACK" \
            --minimum-camera-landmarks "$MINIMUM_CAMERA_LANDMARKS" \
            --max-refinement-passes "$MAX_REFINEMENT_PASSES" \
            --outer-acceleration "$OUTER_ACCELERATION" \
            --outer-acceleration-until "$OUTER_ACCELERATION_UNTIL" \
            --outer-acceleration-restart-iteration "$OUTER_ACCELERATION_RESTART_ITERATION" \
            --line-search-grid "$LINE_SEARCH_GRID" \
            --acceleration-restart-after "$ACCELERATION_RESTART_AFTER" \
            --proximal-metric "$PROXIMAL_METRIC" \
            --consensus-metric "$CONSENSUS_METRIC" \
            --consensus-execution "$CONSENSUS_EXECUTION" \
            --consensus-shared-floor-prior-scale "$CONSENSUS_SHARED_FLOOR_PRIOR_SCALE" \
            --block-regularization "$BLOCK_REGULARIZATION" \
            --block-curvature-multiplier "$BLOCK_CURVATURE_MULTIPLIER" \
            --block-recovery-mode "$BLOCK_RECOVERY_MODE" \
            --maximum-block-curvature-multiplier "$MAXIMUM_BLOCK_CURVATURE_MULTIPLIER" \
            --recovery-exhaustion-policy "$RECOVERY_EXHAUSTION_POLICY" \
            --recovery-relaxed-iterations "$RECOVERY_RELAXED_ITERATIONS" \
            --curvature-decay-after "$CURVATURE_DECAY_AFTER" \
            --curvature-decay-ratio "$CURVATURE_DECAY_RATIO" \
            --metric-diagnostic-iterations "$METRIC_DIAGNOSTIC_ITERATIONS" \
            --landmark-refinement-steps "$LANDMARK_REFINEMENT_STEPS" \
            --consensus-landmark-refinement-steps "$CONSENSUS_LANDMARK_REFINEMENT_STEPS" \
            --consensus-landmark-refinement-policy "$CONSENSUS_LANDMARK_REFINEMENT_POLICY" \
            --target-transformed-lipschitz "$TARGET_TRANSFORMED_LIPSCHITZ" \
            --maximum-block-regularization "$MAXIMUM_BLOCK_REGULARIZATION" \
            --relaxation "$RELAXATION" \
            --penalty-multiplier "$PENALTY_MULTIPLIER" \
            --huber-delta "$HUBER_DELTA" \
            --safeguard-mode "$SAFEGUARD_MODE" \
            --dre-relative-increase "$DRE_RELATIVE_INCREASE" \
            --minimum-primal-ratio "$MINIMUM_PRIMAL_RATIO" \
            --safeguard-annealing-iterations "$SAFEGUARD_ANNEALING_ITERATIONS" \
            --safeguard-reference-iteration "$SAFEGUARD_REFERENCE_ITERATION" \
            --safeguard-annealing-exponent "$SAFEGUARD_ANNEALING_EXPONENT" \
            --safeguard-relative-deadband "$SAFEGUARD_RELATIVE_DEADBAND" \
            --catastrophic-ratio "$CATASTROPHIC_RATIO" \
            --recovery-penalty-ratio "$RECOVERY_PENALTY_RATIO" \
            --results "$RESULT_FILE" --state "$state_file" \
            "${debug_args[@]}" "${trust_args[@]}" "${scaling_args[@]}" "${worker_sse_args[@]}" "${adaptive_depth_args[@]}" "${single_cluster_args[@]}" "${shared_only_args[@]}" "${initial_shared_schur_args[@]}" "${mid_shared_schur_args[@]}" "${final_shared_schur_args[@]}" "${initial_state_args[@]}") > "$log_file" 2>&1
      exit_code=$?
    fi
    set -e
    cleanup_worker
    status=failed
    [[ $exit_code -eq 0 ]] && status=completed
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$VARIANT_NAME" \
      "$dataset" "$clusters" "$ITERATIONS" "$LOCAL_STEPS" \
      "$THREADS_PER_CLUSTER" "$status" "$exit_code" \
      "$((SECONDS - start_seconds))" "$(read_max_rss "$coordinator_time")" \
      "$(read_max_rss "$worker_time")" >> "$STATUS_FILE"
  done
done

echo "DRS failure matrix finished: $STATUS_FILE"
