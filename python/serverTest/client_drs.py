"""Plain product-space Douglas-Rachford splitting for standard BAL pixels."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from bal_evaluator import (
    canonicalize_bal_problem,
    evaluate_bal_state,
    read_bal_problem,
    save_bal_state,
)
from client_admm import AdmmWorkerClient as DrsWorkerClient
from clustering import (
    cluster_by_landmark_scalable,
    cluster_by_landmark_scalable_stable,
)
from drs_consensus import (
    DrsResiduals,
    complete_douglas_rachford_envelope,
    dre_splitting_term,
    drs_step,
    metric_quadratic_sum,
    recover_local_data_objective,
    reset_to_consensus,
)
from drs_consensus_metrics import CONSENSUS_METRIC_MODES
from drs_safeguards import (
    increase_recovery_parameter,
    relative_safeguard_ratios,
    should_reject_trial,
)
from outer_acceleration import create_accelerator, interpolate_line_search_center
from partition_cache import PARTITION_CACHE_MODES, partition_with_cache
from admm_scaling import (
    compute_initial_jacobi_scaling,
    to_physical_cameras,
    to_scaled_cameras,
)


WORKER_SSE_RELATIVE_TOLERANCE = 1e-9
WORKER_CONSENSUS_RELATIVE_TOLERANCE = 1e-11


def apply_single_node_consensus(
    local_cameras, centers, camera_masks, summary, relaxation
):
    """Apply a worker-reduced consensus result to coordinator-owned state."""
    consensus = np.asarray(summary.consensus, dtype=np.float64)
    if consensus.shape != local_cameras.shape[1:]:
        raise ValueError("single-node consensus has an invalid shape")
    active = np.asarray(camera_masks, dtype=bool)[:, :, None]
    reflected = 2.0 * local_cameras - centers
    next_centers = centers + active * relaxation * (
        consensus[None, :, :] - local_cameras
    )
    residuals = DrsResiduals(
        fixed_point_squared=summary.fixed_point_squared,
        proximal_displacement_squared=(
            summary.proximal_displacement_squared
        ),
        reflection_projection_squared=(
            summary.reflection_projection_squared
        ),
        center_step_squared=summary.center_step_squared,
    )
    return consensus, next_centers, reflected, residuals


def validate_worker_consensus_rhs(
    local_cameras,
    centers,
    metric_blocks,
    worker_rhs,
    reference_consensus,
):
    """Validate worker M(2u-s) contributions and their ordered reduction."""
    active_reflection = (
        2.0 * local_cameras[
            metric_blocks.cluster_indices, metric_blocks.camera_indices
        ]
        - centers[
            metric_blocks.cluster_indices, metric_blocks.camera_indices
        ]
    )
    expected_rhs = np.einsum(
        "bij,bj->bi", metric_blocks.blocks, active_reflection
    )
    rhs_scale = max(float(np.max(np.abs(expected_rhs))), 1.0)
    rhs_relative_error = float(
        np.max(np.abs(worker_rhs - expected_rhs)) / rhs_scale
    )
    metric_sum = np.zeros(
        (metric_blocks.camera_count, 9, 9), dtype=np.float64
    )
    rhs_sum = np.zeros((metric_blocks.camera_count, 9), dtype=np.float64)
    for cluster in range(metric_blocks.cluster_count):
        start = metric_blocks._cluster_starts[cluster]
        stop = metric_blocks._cluster_starts[cluster + 1]
        camera_indices = metric_blocks.camera_indices[start:stop]
        metric_sum[camera_indices] += metric_blocks.blocks[start:stop]
        rhs_sum[camera_indices] += worker_rhs[start:stop]
    shadow_consensus = np.linalg.solve(
        metric_sum, rhs_sum[..., None]
    )[..., 0]
    consensus_scale = max(float(np.max(np.abs(reference_consensus))), 1.0)
    consensus_relative_error = float(
        np.max(np.abs(shadow_consensus - reference_consensus))
        / consensus_scale
    )
    if max(rhs_relative_error, consensus_relative_error) > (
        WORKER_CONSENSUS_RELATIVE_TOLERANCE
    ):
        raise RuntimeError(
            "worker consensus shadow mismatch: "
            f"rhs_relative_error={rhs_relative_error:.3g} "
            f"consensus_relative_error={consensus_relative_error:.3g}"
        )
    return rhs_relative_error, consensus_relative_error


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--threads-per-cluster", type=int, default=1)
    parser.add_argument("--nesterov-max-iterations", type=int, default=100)
    parser.add_argument("--nesterov-min-iterations", type=int, default=1)
    parser.add_argument("--nesterov-stop-tolerance", type=float, default=1e-2)
    parser.add_argument(
        "--local-solver",
        choices=("ceres_pcg", "schur_pcg", "nesterov"),
        default="nesterov",
    )
    parser.add_argument(
        "--trust-region-policy", choices=("ceres", "drs", "daba"), default="daba"
    )
    parser.add_argument("--persistent-trust-region", action="store_true")
    parser.add_argument("--trust-region-recovery-ratio", type=float, default=0.5)
    parser.add_argument(
        "--camera-scaling", choices=("none", "jacobi_initial"), default="jacobi_initial"
    )
    parser.add_argument("--camera-scaling-maximum-ratio", type=float)
    parser.add_argument("--camera-scaling-clipping-percentile", type=float)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument(
        "--outer-acceleration",
        choices=("none", "nesterov", "lbfgs", "anderson"),
        default="none",
    )
    parser.add_argument(
        "--line-search-grid", choices=("0,1", "0,0.5,1"), default="0,1"
    )
    parser.add_argument("--acceleration-restart-after", type=int, default=3)
    parser.add_argument("--penalty-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--proximal-metric", choices=("scalar", "block"), default="scalar"
    )
    parser.add_argument(
        "--consensus-metric",
        choices=CONSENSUS_METRIC_MODES,
        default="arithmetic",
    )
    parser.add_argument("--block-regularization", type=float, default=5e-5)
    parser.add_argument("--block-curvature-multiplier", type=float, default=0.0)
    parser.add_argument(
        "--block-recovery-mode",
        choices=("regularization", "curvature", "measured_curvature"),
        default="regularization",
    )
    parser.add_argument(
        "--maximum-block-curvature-multiplier", type=float, default=16.0
    )
    parser.add_argument("--curvature-decay-after", type=int, default=0)
    parser.add_argument("--curvature-decay-ratio", type=float, default=0.5)
    parser.add_argument("--metric-diagnostic-iterations", type=int, default=0)
    parser.add_argument("--worker-sse-shadow", action="store_true")
    parser.add_argument(
        "--suppress-accelerated-landmark-replies", action="store_true"
    )
    parser.add_argument("--worker-owned-landmarks", action="store_true")
    parser.add_argument("--worker-owned-cameras", action="store_true")
    parser.add_argument("--worker-consensus-shadow", action="store_true")
    parser.add_argument(
        "--consensus-execution",
        choices=("coordinator", "single-node"),
        default="coordinator",
    )
    parser.add_argument("--packed-request-buffers", action="store_true")
    parser.add_argument("--landmark-refinement-steps", type=int, default=0)
    parser.add_argument(
        "--consensus-landmark-refinement-steps", type=int, default=0
    )
    parser.add_argument(
        "--consensus-landmark-refinement-policy",
        choices=("safeguard", "reporting", "final"),
        default="safeguard",
    )
    parser.add_argument(
        "--target-transformed-lipschitz", type=float, default=0.475
    )
    parser.add_argument(
        "--maximum-block-regularization", type=float, default=0.5
    )
    parser.add_argument(
        "--safeguard-mode",
        choices=("relative", "catastrophic", "none"),
        default="relative",
    )
    parser.add_argument("--dre-relative-increase", type=float, default=0.01)
    parser.add_argument("--minimum-primal-ratio", type=float, default=1.001)
    parser.add_argument("--catastrophic-ratio", type=float, default=1e6)
    parser.add_argument("--recovery-penalty-ratio", type=float, default=2.0)
    parser.add_argument("--maximum-penalty", type=float, default=1e12)
    parser.add_argument("--results", default="results_drs.jsonl")
    parser.add_argument("--state")
    parser.add_argument("--variant-name", default="plain_drs")
    parser.add_argument("--residual-balance-slack", type=float, default=0.01)
    parser.add_argument("--minimum-camera-landmarks", type=int, default=20)
    parser.add_argument("--max-refinement-passes", type=int, default=3)
    parser.add_argument(
        "--clustering",
        choices=("landmark_scalable", "landmark_scalable_stable"),
        default="landmark_scalable",
    )
    parser.add_argument(
        "--partition-cache", choices=PARTITION_CACHE_MODES, default="auto"
    )
    parser.add_argument(
        "--partition-cache-directory",
        default="~/.cache/bundle_palm/partitions",
    )
    parser.add_argument("--debug-output", action="store_true")
    return parser.parse_args()


def validate_arguments(arguments):
    if arguments.iterations <= 0 or arguments.clusters <= 0:
        raise ValueError("iterations and clusters must be positive")
    if arguments.local_steps <= 0 or arguments.threads_per_cluster <= 0:
        raise ValueError("local steps and threads must be positive")
    if arguments.nesterov_max_iterations <= 0:
        raise ValueError("Nesterov maximum iterations must be positive")
    if not 1 <= arguments.nesterov_min_iterations <= arguments.nesterov_max_iterations:
        raise ValueError(
            "Nesterov minimum iterations must be between 1 and the maximum"
        )
    if not 0.0 < arguments.nesterov_stop_tolerance < 1.0:
        raise ValueError("Nesterov stop tolerance must be in (0, 1)")
    if not 0.0 < arguments.relaxation < 2.0:
        raise ValueError("relaxation must be in (0, 2)")
    if arguments.acceleration_restart_after <= 0:
        raise ValueError("acceleration restart count must be positive")
    if (
        arguments.outer_acceleration != "none"
        and arguments.consensus_landmark_refinement_steps > 0
        and arguments.consensus_landmark_refinement_policy != "final"
    ):
        raise ValueError(
            "accelerated trials support only final consensus landmark refinement"
        )
    if (
        arguments.suppress_accelerated_landmark_replies
        and arguments.outer_acceleration == "none"
    ):
        raise ValueError(
            "accelerated landmark suppression requires outer acceleration"
        )
    if (
        arguments.worker_owned_landmarks
        and arguments.consensus_landmark_refinement_steps > 0
        and arguments.consensus_landmark_refinement_policy != "final"
    ):
        raise ValueError(
            "worker-owned landmarks support only final consensus refinement"
        )
    if arguments.worker_owned_cameras and not arguments.worker_owned_landmarks:
        raise ValueError(
            "worker-owned cameras require worker-owned landmarks"
        )
    if arguments.worker_consensus_shadow and not (
        arguments.proximal_metric == "block"
        and arguments.consensus_metric == "full"
        and arguments.outer_acceleration == "none"
    ):
        raise ValueError(
            "worker consensus shadow requires block/full metrics without acceleration"
        )
    if arguments.consensus_execution == "single-node" and not (
        arguments.proximal_metric == "block"
        and arguments.consensus_metric == "full"
    ):
        raise ValueError(
            "single-node consensus currently requires block/full metrics"
        )
    if (
        arguments.consensus_execution == "single-node"
        and arguments.worker_consensus_shadow
    ):
        raise ValueError(
            "single-node consensus and worker consensus shadow are separate modes"
        )
    if arguments.penalty_multiplier <= 0.0:
        raise ValueError("penalty multiplier must be positive")
    if arguments.block_regularization <= 0.0:
        raise ValueError("block regularization must be positive")
    if (
        not np.isfinite(arguments.block_curvature_multiplier)
        or arguments.block_curvature_multiplier < 0.0
    ):
        raise ValueError("block curvature multiplier must be nonnegative")
    if (
        not np.isfinite(arguments.maximum_block_curvature_multiplier)
        or arguments.maximum_block_curvature_multiplier <= 0.0
    ):
        raise ValueError("maximum block curvature multiplier must be positive")
    if arguments.block_recovery_mode in ("curvature", "measured_curvature"):
        if arguments.proximal_metric != "block":
            raise ValueError("curvature recovery requires block proximal mode")
        if arguments.block_curvature_multiplier <= 0.0:
            raise ValueError(
                "curvature recovery requires a positive block curvature multiplier"
            )
        if (
            arguments.maximum_block_curvature_multiplier
            < arguments.block_curvature_multiplier
        ):
            raise ValueError(
                "maximum block curvature multiplier must not be below its initial value"
            )
    if arguments.curvature_decay_after < 0:
        raise ValueError("curvature decay wait must be nonnegative")
    if not 0 <= arguments.metric_diagnostic_iterations <= 100:
        raise ValueError("metric diagnostic iterations must be in [0, 100]")
    if not 0 <= arguments.landmark_refinement_steps <= 20:
        raise ValueError("landmark refinement steps must be in [0, 20]")
    if not 0 <= arguments.consensus_landmark_refinement_steps <= 20:
        raise ValueError(
            "consensus landmark refinement steps must be in [0, 20]"
        )
    if (
        arguments.worker_sse_shadow
        and arguments.consensus_landmark_refinement_steps > 0
    ):
        raise ValueError(
            "worker SSE shadow currently requires zero consensus landmark refinement"
        )
    if not np.isfinite(arguments.target_transformed_lipschitz) or not (
        0.0 < arguments.target_transformed_lipschitz < 1.0
    ):
        raise ValueError("target transformed Lipschitz must be in (0, 1)")
    if (
        arguments.block_recovery_mode == "measured_curvature"
        and arguments.metric_diagnostic_iterations <= 0
    ):
        raise ValueError(
            "measured curvature recovery requires metric diagnostics"
        )
    if not 0.0 < arguments.curvature_decay_ratio < 1.0:
        raise ValueError("curvature decay ratio must be in (0, 1)")
    if arguments.curvature_decay_after > 0 and (
        arguments.block_recovery_mode not in ("curvature", "measured_curvature")
    ):
        raise ValueError("curvature decay requires curvature recovery mode")
    if arguments.maximum_block_regularization < arguments.block_regularization:
        raise ValueError(
            "maximum block regularization must not be below its initial value"
        )
    if (
        arguments.proximal_metric == "scalar"
        and arguments.consensus_metric != "arithmetic"
    ):
        raise ValueError(
            "scalar proximal mode only supports arithmetic consensus; "
            "use --proximal-metric block for weighted projection"
        )
    if arguments.dre_relative_increase < 0.0:
        raise ValueError("DRE relative increase must be nonnegative")
    if arguments.minimum_primal_ratio < 1.0:
        raise ValueError("minimum primal ratio must be at least one")
    if arguments.catastrophic_ratio < 1.0:
        raise ValueError("catastrophic ratio must be at least one")
    if arguments.recovery_penalty_ratio <= 1.0:
        raise ValueError("recovery penalty ratio must exceed one")
    if arguments.maximum_penalty <= 0.0:
        raise ValueError("maximum penalty must be positive")
    if not 0.0 < arguments.trust_region_recovery_ratio <= 1.0:
        raise ValueError("trust region recovery ratio must be in (0, 1]")
    if arguments.local_solver == "ceres_pcg":
        if arguments.trust_region_policy != "ceres":
            raise ValueError("ceres_pcg requires the ceres trust policy")
        if arguments.proximal_metric == "block":
            raise ValueError("block proximal metrics require a custom local solver")
    elif arguments.trust_region_policy == "ceres":
        raise ValueError("custom local solvers require drs or daba trust policy")


def print_setup(arguments, camera_count, point_count, observation_count, metrics):
    print(
        "======== DRS DEBUG SETUP ========\n"
        f"dataset={Path(arguments.dataset).name} cameras={camera_count} "
        f"points={point_count} observations={observation_count}\n"
        f"iteration: u=prox_F(s), v=P_C(2u-s), "
        f"s_next=s+lambda(v-u)\n"
        f"lambda={arguments.relaxation:g} local_solver={arguments.local_solver} "
        f"trust={arguments.trust_region_policy} "
        f"persistent_trust={arguments.persistent_trust_region} "
        f"local_steps={arguments.local_steps}\n"
        f"proximal_metric={arguments.proximal_metric} "
        f"consensus_metric={arguments.consensus_metric} "
        f"consensus_execution={arguments.consensus_execution} "
        f"block_regularization={arguments.block_regularization:g} "
        f"block_curvature_multiplier="
        f"{arguments.block_curvature_multiplier:g} "
        f"block_recovery={arguments.block_recovery_mode} "
        f"metric_diagnostic_iterations="
        f"{arguments.metric_diagnostic_iterations} "
        f"landmark_refinement_steps="
        f"{arguments.landmark_refinement_steps} "
        f"consensus_landmark_refinement_steps="
        f"{arguments.consensus_landmark_refinement_steps} "
        f"consensus_landmark_refinement_policy="
        f"{arguments.consensus_landmark_refinement_policy}\n"
        f"safeguard={arguments.safeguard_mode} "
        f"dre_increase_at_k5={arguments.dre_relative_increase:g} "
        f"minimum_primal_ratio={arguments.minimum_primal_ratio:g}\n"
        f"initial_sse={metrics['sumSquaredError']:.6g} "
        f"initial_mean_px={metrics['meanReprojectionError']:.6g}",
        file=sys.stderr,
        flush=True,
    )


def print_iteration(row, best_sse, best_iteration, prox_costs):
    action = "REJECT" if row["rejected"] else "ACCEPT"
    print(
        f"{row['iteration']:03d} / 0 ======== DRS {action} ======== "
        f"accepted_f(v)={row['sumSquaredError']:.6g} "
        f"trial_f(v)={row['candidateSumSquaredError']:.6g} "
        f"best={best_sse:.6g}@{best_iteration} "
        f"gain={row['referenceSumSquaredError'] - row['candidateSumSquaredError']:.6g} "
        f"accel_w={row.get('acceptedAccelerationWeight', 0):g} "
        f"trials={row.get('acceleratedTrials', 0)} "
        f"oracle={row.get('oracleCallsThisIteration', 1)}/"
        f"{row.get('proximalOracleCalls', row['iteration'] + 1)} "
        f"elapsed={row['overallSeconds']:.3f}s",
        file=sys.stderr,
        flush=True,
    )
    print(
        "    splitting: "
        f"|u-v|^2={row['fixedPointResidualSquared']:.6g} "
        f"|u-s|^2={row['proximalDisplacementSquared']:.6g} "
        f"|2u-s-v|^2={row['reflectionProjectionSquared']:.6g} "
        f"|s_next-s|^2={row['centerStepSquared']:.6g}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "    merit: "
        f"F(u)={row['localDataObjective']:.6g} "
        f"DRE_model_trial={row['dreModelEnvelope']:.6g} "
        f"DRE_trial={row['douglasRachfordEnvelope']:.6g} "
        f"DRE_ref={row['referenceDRE']:.6g} "
        f"DRE_ref_next={row['acceptedDRE']:.6g} "
        f"DRE_gain={row['referenceDRE'] - row['douglasRachfordEnvelope']:.6g} "
        f"dre_split={row['dreSplittingTerm']:.6g} "
        f"prox_obj sum/min/max={np.sum(prox_costs):.6g}/"
        f"{np.min(prox_costs):.6g}/{np.max(prox_costs):.6g} "
        f"rho={row['proximalPenalty']:.6g} "
        f"gamma={row['gamma']:.6g} "
        f"rho_next={row['nextPenalty']:.6g} "
        f"be={row['blockRegularization']:.6g}->"
        f"{row['nextBlockRegularization']:.6g} "
        f"lip={row['blockCurvatureMultiplier']:.6g}->"
        f"{row['nextBlockCurvatureMultiplier']:.6g} "
        f"lambda={row['relaxation']:.6g} "
        f"rejections={row['rejections']} "
        f"thresholds=(DRE {row['dreRatio']:.6g}x, "
        f"primal {row['primalRatio']:.6g}x) "
        f"failed=(DRE {row['dreThresholdExceeded']}, "
        f"primal {row['primalThresholdExceeded']}) "
        f"recovery={row['recoveryAction']} "
        f"recovery_exhausted={row['recoveryExhausted']} "
        f"L_M(max/median)={row['transformedLipschitzMaximum']:.6g}/"
        f"{row['transformedLipschitzMedian']:.6g} "
        f"L_M_residual_max={row['transformedLipschitzResidualMaximum']:.3g} "
        f"themelis=(metric {row['themelisMetricAdmissible']}, "
        f"decrease {row['themelisDecreasePassed']}) "
        f"prox_defect/fp={row['proximalDefectToFixedPointRatio']:.6g} "
        f"transport={row['transportBytesSent'] + row['transportBytesReceived']}B",
        file=sys.stderr,
        flush=True,
    )


def main():
    arguments = parse_arguments()
    validate_arguments(arguments)
    started_at = time.perf_counter()

    raw_cameras, raw_points, camera_indices, point_indices, raw_observations = (
        read_bal_problem(arguments.dataset)
    )
    cameras, points, observations = canonicalize_bal_problem(
        raw_cameras, raw_points, camera_indices, raw_observations
    )
    camera_count = len(cameras)
    point_count = len(points)
    initial_metrics = evaluate_bal_state(
        cameras, points, camera_indices, point_indices, observations
    )
    if arguments.debug_output:
        print_setup(
            arguments, camera_count, point_count, len(observations), initial_metrics
        )

    partition_started = time.perf_counter()
    partitioner = (
        cluster_by_landmark_scalable
        if arguments.clustering == "landmark_scalable"
        else cluster_by_landmark_scalable_stable
    )
    partition, partition_cache_status, partition_cache_path = partition_with_cache(
        partitioner,
        arguments.clustering,
        camera_indices,
        observations,
        point_indices,
        arguments.clusters,
        camera_count,
        point_count,
        arguments.residual_balance_slack,
        arguments.minimum_camera_landmarks,
        arguments.max_refinement_passes,
        arguments.partition_cache,
        arguments.partition_cache_directory,
    )
    (
        camera_indices_in_cluster,
        point_indices_in_cluster,
        points_2d_in_cluster,
        cluster_count,
    ) = partition
    partition_seconds = time.perf_counter() - partition_started
    if cluster_count != arguments.clusters:
        raise RuntimeError("DRS partitioner changed the requested cluster count")

    local_camera_indices = [
        np.unique(indices, return_inverse=True)[1]
        for indices in camera_indices_in_cluster
    ]
    local_point_indices = [
        np.unique(indices, return_inverse=True)[1]
        for indices in point_indices_in_cluster
    ]
    camera_masks = np.zeros((cluster_count, camera_count), dtype=bool)
    for cluster_id, indices in enumerate(camera_indices_in_cluster):
        camera_masks[cluster_id, np.unique(indices)] = True
    if arguments.debug_output:
        observations_per_cluster = np.asarray(
            [len(indices) for indices in camera_indices_in_cluster]
        )
        cameras_per_cluster = np.sum(camera_masks, axis=1)
        print(
            "======== DRS DEBUG PARTITION ========\n"
            f"clusters={cluster_count} partition={partition_seconds:.3f}s "
            f"obs min/mean/max={observations_per_cluster.min()}/"
            f"{observations_per_cluster.mean():.1f}/"
            f"{observations_per_cluster.max()} "
            f"cameras min/mean/max={cameras_per_cluster.min()}/"
            f"{cameras_per_cluster.mean():.1f}/{cameras_per_cluster.max()} "
            f"additional_camera_copies={int(np.sum(cameras_per_cluster) - camera_count)}",
            file=sys.stderr,
            flush=True,
        )

    scaling_started = time.perf_counter()
    if arguments.camera_scaling == "jacobi_initial":
        camera_scaling = compute_initial_jacobi_scaling(
            cameras,
            points,
            camera_indices,
            point_indices,
            maximum_ratio=arguments.camera_scaling_maximum_ratio,
            clipping_percentile=arguments.camera_scaling_clipping_percentile,
        )
        camera_scaling *= 2.0 / np.sqrt(cluster_count)
    else:
        camera_scaling = np.ones_like(cameras)
    scaling_seconds = time.perf_counter() - scaling_started

    scaled_cameras = to_scaled_cameras(cameras, camera_scaling)
    local_cameras = np.repeat(scaled_cameras[None, :, :], cluster_count, axis=0)
    centers = local_cameras.copy()
    consensus = scaled_cameras.copy()
    landmarks = points.copy()
    penalty = (
        arguments.penalty_multiplier * 2.5 * len(observations) / camera_count
    )
    block_regularization = arguments.block_regularization
    block_curvature_multiplier = arguments.block_curvature_multiplier
    best_sse = initial_metrics["sumSquaredError"]
    best_iteration = -1
    best_cameras = cameras.copy()
    best_points = points.copy()
    accepted_metrics = initial_metrics
    accepted_dre = initial_metrics["sumSquaredError"]
    accepted_model_dre = initial_metrics["sumSquaredError"]
    accepted_fixed_point_squared = 0.0
    accepted_consensus = consensus.copy()
    accepted_landmarks = landmarks.copy()
    rejected_count = 0
    accepted_since_curvature_increase = 0
    revert_landmark_mode = 0
    trust_region_recovery_ratio = 1.0
    trajectory = []
    termination_reason = "iteration_limit"
    final_polishing_applied = False
    final_polishing_initial_sse = float("nan")
    final_polishing_refined_sse = float("nan")
    initialization_seconds = float("nan")
    optimization_seconds = float("nan")
    accelerator = create_accelerator(arguments.outer_acceleration)
    line_search_weights = tuple(
        sorted((float(value) for value in arguments.line_search_grid.split(",")),
               reverse=True)
    )
    acceleration_failures = 0
    accelerated_acceptances = 0
    nominal_fallbacks = 0
    proximal_oracle_calls = 0
    override_landmarks = False
    maximum_worker_sse_relative_error = 0.0
    maximum_worker_consensus_rhs_relative_error = 0.0
    maximum_worker_consensus_relative_error = 0.0
    suppressed_accelerated_landmark_replies = 0
    materialized_accelerated_landmark_states = 0
    suppressed_routine_landmark_replies = 0
    consensus_projection_seconds = 0.0

    worker = DrsWorkerClient()
    try:
        bootstrap_cameras = np.repeat(
            cameras[None, :, :], cluster_count, axis=0
        )
        worker.solve_batch(
            camera_indices_in_cluster,
            point_indices_in_cluster,
            points_2d_in_cluster,
            local_camera_indices,
            local_point_indices,
            bootstrap_cameras,
            landmarks,
            bootstrap_cameras.copy(),
            penalty,
            penalty,
            False,
            initialize=True,
            cluster_count=cluster_count,
            local_steps=arguments.local_steps,
            local_solver=arguments.local_solver,
            trust_region_policy=arguments.trust_region_policy,
            camera_scaling=np.ones_like(cameras),
            revert_landmarks=0,
            persistent_trust_region=arguments.persistent_trust_region,
            trust_region_recovery_ratio=1.0,
            scalar_proximal_prior=(arguments.proximal_metric == "scalar"),
            block_regularization=block_regularization,
            block_curvature_multiplier=block_curvature_multiplier,
            return_metric_blocks=(arguments.proximal_metric == "block"),
            nesterov_max_iterations=arguments.nesterov_max_iterations,
            nesterov_min_iterations=arguments.nesterov_min_iterations,
            nesterov_stop_tolerance=arguments.nesterov_stop_tolerance,
        )
        worker.update_preconditioning(
            camera_indices_in_cluster,
            point_indices_in_cluster,
            camera_scaling,
            cluster_count,
            camera_state=(
                local_cameras if arguments.worker_owned_cameras else None
            ),
        )
        accepted_landmarks = landmarks.copy()
        if arguments.worker_owned_landmarks:
            worker.control_nominal_landmark_state(
                cluster_count, 1, "save_accepted"
            )
            worker.control_nominal_landmark_state(
                cluster_count, 1, "save_best"
            )
        initialization_seconds = time.perf_counter() - started_at
        optimization_operation_start = worker.operation_seconds.copy()
        optimization_operation_call_start = worker.operation_calls.copy()
        optimization_transport_phase_start = (
            worker.transport_phase_seconds.copy()
        )
        optimization_started_at = time.perf_counter()
        for iteration in range(arguments.iterations):
            reference_sse = accepted_metrics["sumSquaredError"]
            reference_dre = accepted_dre
            proximal_penalty = penalty
            proximal_block_regularization = block_regularization
            proximal_block_curvature_multiplier = block_curvature_multiplier
            recovery_exhausted = False
            curvature_decay_applied = False
            oracle_initial_local_cameras = local_cameras.copy()
            oracle_initial_landmarks = landmarks.copy()
            oracle_input_centers = centers.copy()
            oracle_input_consensus = consensus.copy()
            oracle_calls_this_iteration = 1
            accelerated_trials = 0
            accepted_acceleration_weight = 0.0

            prox_result = worker.solve_batch(
                camera_indices_in_cluster,
                point_indices_in_cluster,
                points_2d_in_cluster,
                local_camera_indices,
                local_point_indices,
                local_cameras,
                landmarks,
                centers,
                proximal_penalty,
                proximal_penalty,
                False,
                initialize=False,
                cluster_count=cluster_count,
                local_steps=arguments.local_steps,
                local_solver=arguments.local_solver,
                trust_region_policy=arguments.trust_region_policy,
                camera_scaling=camera_scaling,
                revert_landmarks=revert_landmark_mode,
                persistent_trust_region=arguments.persistent_trust_region,
                trust_region_recovery_ratio=trust_region_recovery_ratio,
                scalar_proximal_prior=(arguments.proximal_metric == "scalar"),
                block_regularization=proximal_block_regularization,
                block_curvature_multiplier=(
                    proximal_block_curvature_multiplier
                ),
                metric_diagnostic_iterations=(
                    arguments.metric_diagnostic_iterations
                ),
                landmark_refinement_steps=(
                    arguments.landmark_refinement_steps
                ),
                override_landmarks=override_landmarks,
                return_metric_blocks=(arguments.proximal_metric == "block"),
                return_metric_diagnostics=(
                    arguments.metric_diagnostic_iterations > 0
                ),
                return_landmarks=not arguments.worker_owned_landmarks,
                worker_owned_cameras=arguments.worker_owned_cameras,
                return_consensus_rhs=arguments.worker_consensus_shadow,
                packed_request_buffers=arguments.packed_request_buffers,
                single_node_consensus=(
                    arguments.consensus_execution == "single-node"
                ),
                consensus_relaxation=arguments.relaxation,
                nesterov_max_iterations=arguments.nesterov_max_iterations,
                nesterov_min_iterations=arguments.nesterov_min_iterations,
                nesterov_stop_tolerance=arguments.nesterov_stop_tolerance,
            )
            if arguments.worker_owned_landmarks:
                suppressed_routine_landmark_replies += cluster_count
            proximal_oracle_calls += 1
            override_landmarks = False
            worker_consensus_rhs = None
            single_node_summary = None
            if (
                arguments.proximal_metric == "block"
                and arguments.metric_diagnostic_iterations > 0
            ):
                if arguments.consensus_execution == "single-node":
                    prox_costs, single_node_summary, metric_diagnostics = (
                        prox_result
                    )
                    raw_metric_blocks = None
                elif arguments.worker_consensus_shadow:
                    (
                        prox_costs,
                        raw_metric_blocks,
                        metric_diagnostics,
                        worker_consensus_rhs,
                    ) = prox_result
                else:
                    prox_costs, raw_metric_blocks, metric_diagnostics = prox_result
            elif arguments.proximal_metric == "block":
                if arguments.consensus_execution == "single-node":
                    prox_costs, single_node_summary = prox_result
                    raw_metric_blocks = None
                elif arguments.worker_consensus_shadow:
                    prox_costs, raw_metric_blocks, worker_consensus_rhs = prox_result
                else:
                    prox_costs, raw_metric_blocks = prox_result
                metric_diagnostics = None
            else:
                prox_costs = prox_result
                raw_metric_blocks = None
                metric_diagnostics = None
            revert_landmark_mode = 0
            trust_region_recovery_ratio = 1.0

            consensus_started_at = time.perf_counter()
            if single_node_summary is not None:
                (
                    candidate_consensus,
                    candidate_centers,
                    _,
                    residuals,
                ) = apply_single_node_consensus(
                    local_cameras,
                    centers,
                    camera_masks,
                    single_node_summary,
                    arguments.relaxation,
                )
                selected_metric_blocks = None
            else:
                (
                    candidate_consensus,
                    candidate_centers,
                    _,
                    residuals,
                    selected_metric_blocks,
                ) = drs_step(
                    local_cameras,
                    centers,
                    camera_masks,
                    consensus,
                    relaxation=arguments.relaxation,
                    metric_blocks=raw_metric_blocks,
                    metric_mode=arguments.consensus_metric,
                )
            consensus_projection_seconds += (
                time.perf_counter() - consensus_started_at
            )
            if arguments.worker_consensus_shadow:
                rhs_error, consensus_error = validate_worker_consensus_rhs(
                    local_cameras,
                    centers,
                    raw_metric_blocks,
                    worker_consensus_rhs,
                    candidate_consensus,
                )
                maximum_worker_consensus_rhs_relative_error = max(
                    maximum_worker_consensus_rhs_relative_error, rhs_error
                )
                maximum_worker_consensus_relative_error = max(
                    maximum_worker_consensus_relative_error, consensus_error
                )
            physical_candidate = to_physical_cameras(
                candidate_consensus, camera_scaling
            )
            worker_sse = None
            if arguments.worker_owned_landmarks:
                worker_sse = worker.evaluate_consensus_sse(
                    camera_indices_in_cluster,
                    candidate_consensus,
                    cluster_count,
                    preserve_cameras=arguments.worker_owned_cameras,
                    packed_request_buffers=arguments.packed_request_buffers,
                )
                unrefined_candidate_metrics = {
                    "sumSquaredError": worker_sse,
                    "meanReprojectionError": float("nan"),
                }
            else:
                unrefined_candidate_metrics = evaluate_bal_state(
                    physical_candidate,
                    landmarks,
                    camera_indices,
                    point_indices,
                    observations,
                )
            if arguments.worker_sse_shadow and worker_sse is None:
                worker_sse = worker.evaluate_consensus_sse(
                    camera_indices_in_cluster,
                    candidate_consensus,
                    cluster_count,
                    preserve_cameras=arguments.worker_owned_cameras,
                    packed_request_buffers=arguments.packed_request_buffers,
                )
                worker_sse_relative_error = abs(
                    worker_sse
                    - unrefined_candidate_metrics["sumSquaredError"]
                ) / max(
                    abs(unrefined_candidate_metrics["sumSquaredError"]),
                    np.finfo(np.float64).tiny,
                )
                maximum_worker_sse_relative_error = max(
                    maximum_worker_sse_relative_error,
                    worker_sse_relative_error,
                )
                if worker_sse_relative_error > WORKER_SSE_RELATIVE_TOLERANCE:
                    raise RuntimeError(
                        "worker/Python candidate SSE mismatch: "
                        f"worker={worker_sse:.17g} "
                        f"python={unrefined_candidate_metrics['sumSquaredError']:.17g} "
                        f"relative_error={worker_sse_relative_error:.3g}"
                    )
            candidate_landmarks = landmarks
            refined_candidate_metrics = None
            if (
                arguments.consensus_landmark_refinement_steps > 0
                and arguments.consensus_landmark_refinement_policy != "final"
            ):
                _, candidate_landmarks = worker.refine_landmarks_at_consensus(
                    camera_indices_in_cluster,
                    point_indices_in_cluster,
                    candidate_consensus,
                    landmarks,
                    cluster_count,
                    arguments.consensus_landmark_refinement_steps,
                    preserve_cameras=arguments.worker_owned_cameras,
                    packed_request_buffers=arguments.packed_request_buffers,
                )
                refined_candidate_metrics = evaluate_bal_state(
                    physical_candidate,
                    candidate_landmarks,
                    camera_indices,
                    point_indices,
                    observations,
                )
            candidate_metrics = (
                refined_candidate_metrics
                if refined_candidate_metrics is not None
                and arguments.consensus_landmark_refinement_policy
                == "safeguard"
                else unrefined_candidate_metrics
            )
            reporting_candidate_metrics = (
                refined_candidate_metrics
                if refined_candidate_metrics is not None
                else candidate_metrics
            )
            candidate_sse = candidate_metrics["sumSquaredError"]
            if arguments.proximal_metric == "scalar":
                splitting_scale = proximal_penalty
                proximal_displacement_cost = (
                    proximal_penalty * residuals.proximal_displacement_squared
                )
                local_data_objective = recover_local_data_objective(
                    float(np.sum(prox_costs)),
                    proximal_penalty,
                    residuals.proximal_displacement_squared,
                )
            else:
                splitting_scale = 1.0
                proximal_displacement_cost = (
                    single_node_summary.proximal_displacement_squared
                    if single_node_summary is not None
                    else metric_quadratic_sum(
                        local_cameras - centers,
                        raw_metric_blocks,
                        camera_masks,
                    )
                )
                local_data_objective = float(np.sum(prox_costs))
            splitting_term = (
                single_node_summary.splitting_term
                if single_node_summary is not None
                else dre_splitting_term(
                    local_cameras,
                    candidate_consensus,
                    centers,
                    camera_masks,
                    splitting_scale,
                    metric_blocks=selected_metric_blocks,
                )
            )
            dre_model_envelope, douglas_rachford_envelope = (
                complete_douglas_rachford_envelope(
                    local_data_objective,
                    splitting_term,
                    candidate_sse,
                )
            )
            transformed_lipschitz_maximum = (
                float(np.nanmax(metric_diagnostics["transformedLipschitz"]))
                if metric_diagnostics is not None
                and np.any(np.isfinite(
                    metric_diagnostics["transformedLipschitz"]
                ))
                else float("nan")
            )
            themelis_margin = (2.0 - arguments.relaxation) / 2.0
            themelis_metric_admissible = (
                np.isfinite(transformed_lipschitz_maximum)
                and transformed_lipschitz_maximum < themelis_margin
            )
            if themelis_metric_admissible:
                themelis_decrease_constant = 0.5 * (
                    arguments.relaxation
                    / (1.0 + transformed_lipschitz_maximum) ** 2
                    * (themelis_margin - transformed_lipschitz_maximum)
                )
                themelis_model_threshold = (
                    accepted_model_dre
                    - themelis_decrease_constant
                    * accepted_fixed_point_squared
                )
                themelis_decrease_passed = (
                    dre_model_envelope <= themelis_model_threshold
                )
            else:
                themelis_decrease_constant = float("nan")
                themelis_model_threshold = float("nan")
                themelis_decrease_passed = False
            dre_ratio, primal_ratio = relative_safeguard_ratios(
                iteration,
                arguments.iterations,
                dre_increase_at_reference=arguments.dre_relative_increase,
                minimum_primal_ratio=arguments.minimum_primal_ratio,
            )
            dre_threshold_exceeded = (
                not np.isfinite(douglas_rachford_envelope)
                or douglas_rachford_envelope > dre_ratio * reference_dre
            )
            primal_threshold_exceeded = (
                not np.isfinite(candidate_sse)
                or candidate_sse > primal_ratio * reference_sse
            )
            if arguments.safeguard_mode == "relative":
                rejected = should_reject_trial(
                    0,
                    1,
                    douglas_rachford_envelope,
                    candidate_sse,
                    reference_dre,
                    reference_sse,
                    dre_ratio,
                    primal_ratio,
                )
            elif arguments.safeguard_mode == "catastrophic":
                rejected = (
                    not np.isfinite(candidate_sse)
                    or candidate_sse
                    > arguments.catastrophic_ratio * reference_sse
                )
            else:
                rejected = (
                    not np.isfinite(douglas_rachford_envelope)
                    or not np.isfinite(candidate_sse)
                )

            nominal_trial = {
                "local_cameras": local_cameras.copy(),
                "landmarks": landmarks.copy(),
                "prox_costs": prox_costs.copy(),
                "raw_metric_blocks": (
                    raw_metric_blocks.copy()
                    if raw_metric_blocks is not None else None
                ),
                "metric_diagnostics": metric_diagnostics,
                "candidate_consensus": candidate_consensus.copy(),
                "candidate_centers": candidate_centers.copy(),
                "residuals": residuals,
                "selected_metric_blocks": (
                    selected_metric_blocks.copy()
                    if selected_metric_blocks is not None else None
                ),
                "candidate_metrics": candidate_metrics,
                "candidate_sse": candidate_sse,
                "splitting_scale": splitting_scale,
                "proximal_displacement_cost": proximal_displacement_cost,
                "local_data_objective": local_data_objective,
                "splitting_term": splitting_term,
                "dre_model_envelope": dre_model_envelope,
                "douglas_rachford_envelope": douglas_rachford_envelope,
                "rejected": rejected,
            }
            acceleration_proposal, proposal_is_accelerated = accelerator.propose(
                oracle_input_centers,
                candidate_centers,
                iteration,
            )
            selected_trial = nominal_trial
            evaluated_accelerated_trial = False
            if (
                proposal_is_accelerated
                or accelerator.requires_first_trial_observation
            ):
                nominal_landmark_state_id = iteration + 1
                if (
                    arguments.worker_sse_shadow
                    or arguments.suppress_accelerated_landmark_replies
                    or arguments.worker_owned_landmarks
                ):
                    worker.control_nominal_landmark_state(
                        cluster_count,
                        nominal_landmark_state_id,
                        "save",
                    )
                for acceleration_weight in line_search_weights:
                    if acceleration_weight == 0.0:
                        continue
                    evaluated_accelerated_trial = True
                    accelerated_trials += 1
                    oracle_calls_this_iteration += 1
                    proximal_oracle_calls += 1
                    trial_centers = interpolate_line_search_center(
                        candidate_centers,
                        acceleration_proposal,
                        acceleration_weight,
                    )
                    trial_local_cameras = oracle_initial_local_cameras.copy()
                    trial_landmarks = oracle_initial_landmarks.copy()
                    trial_result = worker.solve_batch(
                        camera_indices_in_cluster,
                        point_indices_in_cluster,
                        points_2d_in_cluster,
                        local_camera_indices,
                        local_point_indices,
                        trial_local_cameras,
                        trial_landmarks,
                        trial_centers,
                        proximal_penalty,
                        proximal_penalty,
                        False,
                        initialize=False,
                        cluster_count=cluster_count,
                        local_steps=arguments.local_steps,
                        local_solver=arguments.local_solver,
                        trust_region_policy=arguments.trust_region_policy,
                        camera_scaling=camera_scaling,
                        revert_landmarks=1,
                        persistent_trust_region=arguments.persistent_trust_region,
                        trust_region_recovery_ratio=1.0,
                        scalar_proximal_prior=(arguments.proximal_metric == "scalar"),
                        block_regularization=proximal_block_regularization,
                        block_curvature_multiplier=proximal_block_curvature_multiplier,
                        metric_diagnostic_iterations=arguments.metric_diagnostic_iterations,
                        landmark_refinement_steps=arguments.landmark_refinement_steps,
                        return_metric_blocks=(arguments.proximal_metric == "block"),
                        return_metric_diagnostics=(arguments.metric_diagnostic_iterations > 0),
                        return_landmarks=(
                            not arguments.suppress_accelerated_landmark_replies
                            and not arguments.worker_owned_landmarks
                        ),
                        worker_owned_cameras=arguments.worker_owned_cameras,
                        packed_request_buffers=arguments.packed_request_buffers,
                        single_node_consensus=(
                            arguments.consensus_execution == "single-node"
                        ),
                        consensus_relaxation=arguments.relaxation,
                        nesterov_max_iterations=(
                            arguments.nesterov_max_iterations
                        ),
                        nesterov_min_iterations=(
                            arguments.nesterov_min_iterations
                        ),
                        nesterov_stop_tolerance=(
                            arguments.nesterov_stop_tolerance
                        ),
                    )
                    if (
                        arguments.suppress_accelerated_landmark_replies
                        or arguments.worker_owned_landmarks
                    ):
                        suppressed_accelerated_landmark_replies += cluster_count
                    if (
                        arguments.proximal_metric == "block"
                        and arguments.metric_diagnostic_iterations > 0
                    ):
                        if arguments.consensus_execution == "single-node":
                            (
                                trial_prox_costs,
                                trial_single_node_summary,
                                trial_diagnostics,
                            ) = trial_result
                            trial_raw_blocks = None
                        else:
                            (
                                trial_prox_costs,
                                trial_raw_blocks,
                                trial_diagnostics,
                            ) = trial_result
                    elif arguments.proximal_metric == "block":
                        if arguments.consensus_execution == "single-node":
                            trial_prox_costs, trial_single_node_summary = (
                                trial_result
                            )
                            trial_raw_blocks = None
                        else:
                            trial_prox_costs, trial_raw_blocks = trial_result
                        trial_diagnostics = None
                    else:
                        trial_prox_costs = trial_result
                        trial_raw_blocks = None
                        trial_diagnostics = None
                        trial_single_node_summary = None
                    consensus_started_at = time.perf_counter()
                    if arguments.consensus_execution == "single-node":
                        (
                            trial_consensus,
                            trial_next_centers,
                            _,
                            trial_residuals,
                        ) = apply_single_node_consensus(
                            trial_local_cameras,
                            trial_centers,
                            camera_masks,
                            trial_single_node_summary,
                            arguments.relaxation,
                        )
                        trial_selected_blocks = None
                    else:
                        (
                            trial_consensus,
                            trial_next_centers,
                            _,
                            trial_residuals,
                            trial_selected_blocks,
                        ) = drs_step(
                            trial_local_cameras,
                            trial_centers,
                            camera_masks,
                            oracle_input_consensus,
                            relaxation=arguments.relaxation,
                            metric_blocks=trial_raw_blocks,
                            metric_mode=arguments.consensus_metric,
                        )
                    consensus_projection_seconds += (
                        time.perf_counter() - consensus_started_at
                    )
                    if acceleration_weight == 1.0:
                        accelerator.observe_first_trial(
                            trial_centers - trial_next_centers
                        )
                    trial_metrics = None
                    worker_sse = None
                    if (
                        arguments.suppress_accelerated_landmark_replies
                        or arguments.worker_owned_landmarks
                    ):
                        worker_sse = worker.evaluate_consensus_sse(
                            camera_indices_in_cluster,
                            trial_consensus,
                            cluster_count,
                            preserve_cameras=arguments.worker_owned_cameras,
                            packed_request_buffers=arguments.packed_request_buffers,
                        )
                        trial_sse = worker_sse
                    else:
                        trial_metrics = evaluate_bal_state(
                            to_physical_cameras(trial_consensus, camera_scaling),
                            trial_landmarks,
                            camera_indices,
                            point_indices,
                            observations,
                        )
                        trial_sse = trial_metrics["sumSquaredError"]
                    if arguments.worker_sse_shadow and worker_sse is None:
                        worker_sse = worker.evaluate_consensus_sse(
                            camera_indices_in_cluster,
                            trial_consensus,
                            cluster_count,
                            preserve_cameras=arguments.worker_owned_cameras,
                            packed_request_buffers=arguments.packed_request_buffers,
                        )
                        worker_sse_relative_error = abs(
                            worker_sse - trial_metrics["sumSquaredError"]
                        ) / max(
                            abs(trial_metrics["sumSquaredError"]),
                            np.finfo(np.float64).tiny,
                        )
                        maximum_worker_sse_relative_error = max(
                            maximum_worker_sse_relative_error,
                            worker_sse_relative_error,
                        )
                        if (
                            worker_sse_relative_error
                            > WORKER_SSE_RELATIVE_TOLERANCE
                        ):
                            raise RuntimeError(
                                "worker/Python trial SSE mismatch: "
                                f"worker={worker_sse:.17g} "
                                f"python={trial_metrics['sumSquaredError']:.17g} "
                                f"relative_error={worker_sse_relative_error:.3g}"
                            )
                    if arguments.proximal_metric == "scalar":
                        trial_splitting_scale = proximal_penalty
                        trial_displacement_cost = (
                            proximal_penalty
                            * trial_residuals.proximal_displacement_squared
                        )
                        trial_local_objective = recover_local_data_objective(
                            float(np.sum(trial_prox_costs)),
                            proximal_penalty,
                            trial_residuals.proximal_displacement_squared,
                        )
                    else:
                        trial_splitting_scale = 1.0
                        trial_displacement_cost = (
                            trial_single_node_summary.proximal_displacement_squared
                            if arguments.consensus_execution == "single-node"
                            else metric_quadratic_sum(
                                trial_local_cameras - trial_centers,
                                trial_raw_blocks,
                                camera_masks,
                            )
                        )
                        trial_local_objective = float(np.sum(trial_prox_costs))
                    trial_splitting_term = (
                        trial_single_node_summary.splitting_term
                        if arguments.consensus_execution == "single-node"
                        else dre_splitting_term(
                            trial_local_cameras,
                            trial_consensus,
                            trial_centers,
                            camera_masks,
                            trial_splitting_scale,
                            metric_blocks=trial_selected_blocks,
                        )
                    )
                    trial_model_dre, trial_dre = complete_douglas_rachford_envelope(
                        trial_local_objective,
                        trial_splitting_term,
                        trial_sse,
                    )
                    if arguments.safeguard_mode == "relative":
                        trial_rejected = should_reject_trial(
                            0, 1, trial_dre, trial_sse,
                            reference_dre, reference_sse,
                            dre_ratio, primal_ratio,
                        )
                    elif arguments.safeguard_mode == "catastrophic":
                        trial_rejected = (
                            not np.isfinite(trial_sse)
                            or trial_sse > arguments.catastrophic_ratio * reference_sse
                        )
                    else:
                        trial_rejected = (
                            not np.isfinite(trial_dre)
                            or not np.isfinite(trial_sse)
                        )
                    if (
                        arguments.suppress_accelerated_landmark_replies
                        and not arguments.worker_owned_landmarks
                    ):
                        error_bound = (
                            WORKER_SSE_RELATIVE_TOLERANCE
                            * max(abs(trial_sse), np.finfo(np.float64).tiny)
                        )
                        if arguments.safeguard_mode == "relative":
                            decisive_rejection = (
                                trial_rejected
                                and trial_dre
                                > dre_ratio * reference_dre + error_bound
                                and trial_sse
                                > primal_ratio * reference_sse + error_bound
                            )
                        elif arguments.safeguard_mode == "catastrophic":
                            decisive_rejection = (
                                trial_rejected
                                and trial_sse
                                > arguments.catastrophic_ratio * reference_sse
                                + error_bound
                            )
                        else:
                            decisive_rejection = trial_rejected
                        if not decisive_rejection:
                            trial_landmarks = (
                                worker.materialize_current_landmarks(
                                    point_indices_in_cluster,
                                    trial_landmarks,
                                    cluster_count,
                                    nominal_landmark_state_id,
                                )
                            )
                            materialized_accelerated_landmark_states += 1
                            trial_metrics = evaluate_bal_state(
                                to_physical_cameras(
                                    trial_consensus, camera_scaling
                                ),
                                trial_landmarks,
                                camera_indices,
                                point_indices,
                                observations,
                            )
                            trial_sse = trial_metrics["sumSquaredError"]
                            worker_sse_relative_error = abs(
                                worker_sse - trial_sse
                            ) / max(
                                abs(trial_sse), np.finfo(np.float64).tiny
                            )
                            maximum_worker_sse_relative_error = max(
                                maximum_worker_sse_relative_error,
                                worker_sse_relative_error,
                            )
                            if (
                                worker_sse_relative_error
                                > WORKER_SSE_RELATIVE_TOLERANCE
                            ):
                                raise RuntimeError(
                                    "worker/Python materialized trial SSE "
                                    "mismatch: "
                                    f"worker={worker_sse:.17g} "
                                    f"python={trial_sse:.17g} "
                                    f"relative_error="
                                    f"{worker_sse_relative_error:.3g}"
                                )
                            trial_model_dre, trial_dre = (
                                complete_douglas_rachford_envelope(
                                    trial_local_objective,
                                    trial_splitting_term,
                                    trial_sse,
                                )
                            )
                            if arguments.safeguard_mode == "relative":
                                trial_rejected = should_reject_trial(
                                    0, 1, trial_dre, trial_sse,
                                    reference_dre, reference_sse,
                                    dre_ratio, primal_ratio,
                                )
                            elif arguments.safeguard_mode == "catastrophic":
                                trial_rejected = (
                                    not np.isfinite(trial_sse)
                                    or trial_sse
                                    > arguments.catastrophic_ratio
                                    * reference_sse
                                )
                            else:
                                trial_rejected = (
                                    not np.isfinite(trial_dre)
                                    or not np.isfinite(trial_sse)
                                )
                    if trial_metrics is None:
                        trial_metrics = {
                            "sumSquaredError": trial_sse,
                            "meanReprojectionError": float("nan"),
                        }
                    if not trial_rejected:
                        selected_trial = {
                            "local_cameras": trial_local_cameras,
                            "landmarks": trial_landmarks,
                            "prox_costs": trial_prox_costs,
                            "raw_metric_blocks": trial_raw_blocks,
                            "metric_diagnostics": trial_diagnostics,
                            "candidate_consensus": trial_consensus,
                            "candidate_centers": trial_next_centers,
                            "residuals": trial_residuals,
                            "selected_metric_blocks": trial_selected_blocks,
                            "candidate_metrics": trial_metrics,
                            "candidate_sse": trial_sse,
                            "splitting_scale": trial_splitting_scale,
                            "proximal_displacement_cost": trial_displacement_cost,
                            "local_data_objective": trial_local_objective,
                            "splitting_term": trial_splitting_term,
                            "dre_model_envelope": trial_model_dre,
                            "douglas_rachford_envelope": trial_dre,
                            "rejected": False,
                        }
                        accepted_acceleration_weight = acceleration_weight
                        break

            if selected_trial is not nominal_trial:
                local_cameras = selected_trial["local_cameras"]
                landmarks = selected_trial["landmarks"]
                prox_costs = selected_trial["prox_costs"]
                raw_metric_blocks = selected_trial["raw_metric_blocks"]
                metric_diagnostics = selected_trial["metric_diagnostics"]
                candidate_consensus = selected_trial["candidate_consensus"]
                candidate_centers = selected_trial["candidate_centers"]
                residuals = selected_trial["residuals"]
                selected_metric_blocks = selected_trial["selected_metric_blocks"]
                candidate_metrics = selected_trial["candidate_metrics"]
                reporting_candidate_metrics = candidate_metrics
                candidate_landmarks = landmarks
                candidate_sse = selected_trial["candidate_sse"]
                splitting_scale = selected_trial["splitting_scale"]
                proximal_displacement_cost = selected_trial["proximal_displacement_cost"]
                local_data_objective = selected_trial["local_data_objective"]
                splitting_term = selected_trial["splitting_term"]
                dre_model_envelope = selected_trial["dre_model_envelope"]
                douglas_rachford_envelope = selected_trial["douglas_rachford_envelope"]
                rejected = False
                accelerated_acceptances += 1
                acceleration_failures = 0
                accelerator.accepted(True)
            else:
                local_cameras = nominal_trial["local_cameras"]
                landmarks = nominal_trial["landmarks"]
                if evaluated_accelerated_trial and not rejected:
                    nominal_fallbacks += 1
                    acceleration_failures += 1
                    override_landmarks = not arguments.worker_owned_landmarks
                    accelerator.accepted(False)
                    if acceleration_failures >= arguments.acceleration_restart_after:
                        accelerator.reset()
                        acceleration_failures = 0
                elif rejected:
                    accelerator.reset()
                    acceleration_failures = 0

            if (
                evaluated_accelerated_trial
                and (
                    arguments.worker_sse_shadow
                    or arguments.suppress_accelerated_landmark_replies
                    or arguments.worker_owned_landmarks
                )
            ):
                if selected_trial is nominal_trial:
                    worker.control_nominal_landmark_state(
                        cluster_count,
                        nominal_landmark_state_id,
                        (
                            "restore_roundtrip"
                            if arguments.worker_owned_landmarks
                            else "restore"
                        ),
                    )
                    restored_worker_sse = worker.evaluate_consensus_sse(
                        camera_indices_in_cluster,
                        nominal_trial["candidate_consensus"],
                        cluster_count,
                        preserve_cameras=arguments.worker_owned_cameras,
                        packed_request_buffers=arguments.packed_request_buffers,
                    )
                    restored_relative_error = abs(
                        restored_worker_sse - nominal_trial["candidate_sse"]
                    ) / max(
                        abs(nominal_trial["candidate_sse"]),
                        np.finfo(np.float64).tiny,
                    )
                    maximum_worker_sse_relative_error = max(
                        maximum_worker_sse_relative_error,
                        restored_relative_error,
                    )
                    if restored_relative_error > WORKER_SSE_RELATIVE_TOLERANCE:
                        raise RuntimeError(
                            "restored nominal landmark SSE mismatch: "
                            f"worker={restored_worker_sse:.17g} "
                            f"python={nominal_trial['candidate_sse']:.17g} "
                            f"relative_error={restored_relative_error:.3g}"
                        )
                worker.control_nominal_landmark_state(
                    cluster_count,
                    nominal_landmark_state_id,
                    "discard",
                )

            transformed_lipschitz_maximum = (
                float(np.nanmax(metric_diagnostics["transformedLipschitz"]))
                if metric_diagnostics is not None
                and np.any(np.isfinite(metric_diagnostics["transformedLipschitz"]))
                else float("nan")
            )
            themelis_metric_admissible = (
                np.isfinite(transformed_lipschitz_maximum)
                and transformed_lipschitz_maximum < themelis_margin
            )
            if themelis_metric_admissible:
                themelis_decrease_constant = 0.5 * (
                    arguments.relaxation
                    / (1.0 + transformed_lipschitz_maximum) ** 2
                    * (themelis_margin - transformed_lipschitz_maximum)
                )
                themelis_model_threshold = (
                    accepted_model_dre
                    - themelis_decrease_constant * accepted_fixed_point_squared
                )
                themelis_decrease_passed = (
                    dre_model_envelope <= themelis_model_threshold
                )
            else:
                themelis_decrease_constant = float("nan")
                themelis_model_threshold = float("nan")
                themelis_decrease_passed = False
            dre_threshold_exceeded = (
                not np.isfinite(douglas_rachford_envelope)
                or douglas_rachford_envelope > dre_ratio * reference_dre
            )
            primal_threshold_exceeded = (
                not np.isfinite(candidate_sse)
                or candidate_sse > primal_ratio * reference_sse
            )
            if rejected:
                rejected_count += 1
                if arguments.worker_owned_landmarks:
                    worker.control_nominal_landmark_state(
                        cluster_count,
                        iteration + 1,
                        "restore_accepted",
                    )
                (
                    local_cameras,
                    centers,
                    consensus,
                ) = reset_to_consensus(accepted_consensus, cluster_count)
                landmarks = accepted_landmarks.copy()
                if arguments.proximal_metric == "scalar":
                    penalty, proximal_metric_changed = increase_recovery_parameter(
                        proximal_penalty,
                        arguments.maximum_penalty,
                        arguments.recovery_penalty_ratio,
                    )
                else:
                    if arguments.block_recovery_mode in (
                        "curvature", "measured_curvature"
                    ):
                        minimum_next_multiplier = (
                            proximal_block_curvature_multiplier
                            * arguments.recovery_penalty_ratio
                        )
                        if (
                            arguments.block_recovery_mode
                            == "measured_curvature"
                            and np.isfinite(transformed_lipschitz_maximum)
                        ):
                            minimum_next_multiplier = max(
                                minimum_next_multiplier,
                                proximal_block_curvature_multiplier
                                * transformed_lipschitz_maximum
                                / arguments.target_transformed_lipschitz,
                            )
                        block_curvature_multiplier = min(
                            arguments.maximum_block_curvature_multiplier,
                            minimum_next_multiplier,
                        )
                        proximal_metric_changed = (
                            block_curvature_multiplier
                            > proximal_block_curvature_multiplier
                        )
                        if proximal_metric_changed:
                            accepted_since_curvature_increase = 0
                    else:
                        (
                            block_regularization,
                            proximal_metric_changed,
                        ) = increase_recovery_parameter(
                            proximal_block_regularization,
                            arguments.maximum_block_regularization,
                            arguments.recovery_penalty_ratio,
                        )
                recovery_exhausted = not proximal_metric_changed
                metrics = accepted_metrics
                revert_landmark_mode = (
                    3 if arguments.worker_owned_landmarks else 2
                )
                if arguments.persistent_trust_region:
                    trust_region_recovery_ratio = (
                        arguments.trust_region_recovery_ratio
                    )
                recovery_action = "accepted_consensus_reset"
            else:
                centers = candidate_centers
                consensus = candidate_consensus
                accepted_metrics = candidate_metrics
                accepted_dre = douglas_rachford_envelope
                accepted_model_dre = dre_model_envelope
                accepted_fixed_point_squared = residuals.fixed_point_squared
                accepted_consensus = consensus.copy()
                accepted_landmarks = landmarks.copy()
                if arguments.worker_owned_landmarks:
                    worker.control_nominal_landmark_state(
                        cluster_count,
                        iteration + 1,
                        "save_accepted",
                    )
                metrics = candidate_metrics
                recovery_action = "none"
                accepted_since_curvature_increase += 1
                if (
                    arguments.curvature_decay_after > 0
                    and accepted_since_curvature_increase
                    >= arguments.curvature_decay_after
                    and block_curvature_multiplier
                    > arguments.block_curvature_multiplier
                ):
                    block_curvature_multiplier = max(
                        arguments.block_curvature_multiplier,
                        block_curvature_multiplier
                        * arguments.curvature_decay_ratio,
                    )
                    accepted_since_curvature_increase = 0
                    curvature_decay_applied = True

            if metric_diagnostics is not None and np.any(np.isfinite(
                metric_diagnostics["proximalDefectSquared"]
            )):
                camera_defect_squared = float(np.nansum(
                    metric_diagnostics["cameraProximalDefectSquared"]
                ))
                landmark_defect_squared = float(np.nansum(
                    metric_diagnostics["landmarkProximalDefectSquared"]
                ))
                proximal_defect_squared = float(np.nansum(
                    metric_diagnostics["proximalDefectSquared"]
                ))
                proximal_defect_maximum_squared = float(np.nanmax(
                    metric_diagnostics["proximalDefectSquared"]
                ))
                proximal_defect_to_fixed_point_ratio = float(np.sqrt(
                    proximal_defect_squared
                    / max(
                        residuals.fixed_point_squared,
                        np.finfo(np.float64).tiny,
                    )
                ))
            else:
                camera_defect_squared = float("nan")
                landmark_defect_squared = float("nan")
                proximal_defect_squared = float("nan")
                proximal_defect_maximum_squared = float("nan")
                proximal_defect_to_fixed_point_ratio = float("nan")

            row = {
                "iteration": iteration,
                "overallSeconds": time.perf_counter() - started_at,
                "optimizationSeconds": (
                    time.perf_counter() - optimization_started_at
                ),
                "sumSquaredError": metrics["sumSquaredError"],
                "candidateSumSquaredError": candidate_sse,
                "refinedCandidateSumSquaredError": (
                    reporting_candidate_metrics["sumSquaredError"]
                ),
                "referenceSumSquaredError": reference_sse,
                "meanReprojectionError": metrics["meanReprojectionError"],
                "rejected": rejected,
                "rejections": rejected_count,
                "fixedPointResidualSquared": residuals.fixed_point_squared,
                "proximalDisplacementSquared": (
                    residuals.proximal_displacement_squared
                ),
                "reflectionProjectionSquared": (
                    residuals.reflection_projection_squared
                ),
                "centerStepSquared": residuals.center_step_squared,
                "dreSplittingTerm": splitting_term,
                "localDataObjective": local_data_objective,
                "proximalDisplacementCost": proximal_displacement_cost,
                "douglasRachfordEnvelope": douglas_rachford_envelope,
                "dreModelEnvelope": dre_model_envelope,
                "referenceDRE": reference_dre,
                "acceptedDRE": accepted_dre,
                "safeguardMode": arguments.safeguard_mode,
                "dreRatio": dre_ratio,
                "primalRatio": primal_ratio,
                "dreThreshold": dre_ratio * reference_dre,
                "primalThreshold": primal_ratio * reference_sse,
                "dreThresholdExceeded": bool(dre_threshold_exceeded),
                "primalThresholdExceeded": bool(primal_threshold_exceeded),
                "recoveryAction": recovery_action,
                "outerAcceleration": arguments.outer_acceleration,
                "lineSearchGrid": arguments.line_search_grid,
                "acceptedAccelerationWeight": accepted_acceleration_weight,
                "acceleratedTrials": accelerated_trials,
                "oracleCallsThisIteration": oracle_calls_this_iteration,
                "proximalOracleCalls": proximal_oracle_calls,
                "acceleratedAcceptances": accelerated_acceptances,
                "nominalFallbacks": nominal_fallbacks,
                "localProximalObjectiveSum": float(np.sum(prox_costs)),
                "proximalPenalty": proximal_penalty,
                "nextPenalty": penalty,
                "penalty": penalty,
                "relaxation": arguments.relaxation,
                "recoveryExhausted": recovery_exhausted,
                "gamma": (
                    1.0 / proximal_penalty
                    if arguments.proximal_metric == "scalar"
                    else 1.0
                ),
                "proximalMetric": arguments.proximal_metric,
                "consensusMetric": arguments.consensus_metric,
                "consensusExecution": arguments.consensus_execution,
                "blockRegularization": proximal_block_regularization,
                "blockCurvatureMultiplier": (
                    proximal_block_curvature_multiplier
                ),
                "nextBlockCurvatureMultiplier": block_curvature_multiplier,
                "curvatureDecayApplied": curvature_decay_applied,
                "transformedLipschitzEstimates": (
                    metric_diagnostics["transformedLipschitz"].tolist()
                    if metric_diagnostics is not None
                    else []
                ),
                "transformedLipschitzResiduals": (
                    metric_diagnostics["relativeResidual"].tolist()
                    if metric_diagnostics is not None
                    else []
                ),
                "transformedLipschitzMaximum": (
                    transformed_lipschitz_maximum
                ),
                "transformedLipschitzMedian": (
                    float(np.nanmedian(metric_diagnostics["transformedLipschitz"]))
                    if metric_diagnostics is not None
                    and np.any(np.isfinite(
                        metric_diagnostics["transformedLipschitz"]
                    ))
                    else float("nan")
                ),
                "transformedLipschitzResidualMaximum": (
                    float(np.nanmax(metric_diagnostics["relativeResidual"]))
                    if metric_diagnostics is not None
                    and np.any(np.isfinite(metric_diagnostics["relativeResidual"]))
                    else float("nan")
                ),
                "themelisMetricAdmissible": bool(
                    themelis_metric_admissible
                ),
                "themelisDecreaseConstant": themelis_decrease_constant,
                "themelisModelThreshold": themelis_model_threshold,
                "themelisDecreasePassed": bool(themelis_decrease_passed),
                "cameraProximalDefectSquared": camera_defect_squared,
                "landmarkProximalDefectSquared": landmark_defect_squared,
                "proximalDefectSquared": proximal_defect_squared,
                "proximalDefectMaximumSquared": (
                    proximal_defect_maximum_squared
                ),
                "proximalDefectToFixedPointRatio": (
                    proximal_defect_to_fixed_point_ratio
                ),
                "nextBlockRegularization": block_regularization,
                "transportBytesSent": worker.sent_bytes,
                "transportBytesReceived": worker.received_bytes,
            }
            trajectory.append(row)
            if (
                not rejected
                and reporting_candidate_metrics["sumSquaredError"] < best_sse
            ):
                best_sse = reporting_candidate_metrics["sumSquaredError"]
                best_iteration = iteration
                best_cameras = to_physical_cameras(consensus, camera_scaling).copy()
                if arguments.worker_owned_landmarks:
                    worker.control_nominal_landmark_state(
                        cluster_count,
                        iteration + 1,
                        "save_best",
                    )
                else:
                    best_points = candidate_landmarks.copy()
            if arguments.debug_output:
                print_iteration(row, best_sse, best_iteration, prox_costs)
            if recovery_exhausted:
                termination_reason = "recovery_exhausted"
                break
        optimization_seconds = time.perf_counter() - optimization_started_at
        optimization_worker_operation_seconds = {
            name: seconds - optimization_operation_start[name]
            for name, seconds in worker.operation_seconds.items()
        }
        optimization_worker_operation_calls = {
            name: calls - optimization_operation_call_start[name]
            for name, calls in worker.operation_calls.items()
        }
        optimization_transport_phase_seconds = {
            name: seconds - optimization_transport_phase_start[name]
            for name, seconds in worker.transport_phase_seconds.items()
        }
        if arguments.worker_owned_landmarks:
            best_points = worker.materialize_current_landmarks(
                point_indices_in_cluster,
                best_points,
                cluster_count,
                max(1, best_iteration + 1),
                source="best",
            )
        if (
            arguments.consensus_landmark_refinement_steps > 0
            and arguments.consensus_landmark_refinement_policy == "final"
        ):
            final_polishing_applied = True
            final_polishing_initial_sse = best_sse
            _, refined_points = worker.refine_landmarks_at_consensus(
                camera_indices_in_cluster,
                point_indices_in_cluster,
                to_scaled_cameras(best_cameras, camera_scaling),
                best_points,
                cluster_count,
                arguments.consensus_landmark_refinement_steps,
                use_landmark_state=True,
                preserve_cameras=arguments.worker_owned_cameras,
                packed_request_buffers=arguments.packed_request_buffers,
            )
            refined_metrics = evaluate_bal_state(
                best_cameras,
                refined_points,
                camera_indices,
                point_indices,
                observations,
            )
            final_polishing_refined_sse = refined_metrics["sumSquaredError"]
            if final_polishing_refined_sse < best_sse:
                best_sse = final_polishing_refined_sse
                best_points = refined_points
    finally:
        sent_bytes = worker.sent_bytes
        received_bytes = worker.received_bytes
        worker_operation_seconds = worker.operation_seconds.copy()
        worker_operation_calls = worker.operation_calls.copy()
        transport_phase_seconds = worker.transport_phase_seconds.copy()
        worker.close()

    final_metrics = evaluate_bal_state(
        best_cameras, best_points, camera_indices, point_indices, observations
    )
    result = {
        "solver": "cpu-product-space-drs",
        "variant": arguments.variant_name,
        "dataset": str(Path(arguments.dataset).resolve()),
        "iterations": arguments.iterations,
        "clusters": cluster_count,
        "clustering": arguments.clustering,
        "partitionCacheMode": arguments.partition_cache,
        "partitionCacheStatus": partition_cache_status,
        "partitionCachePath": str(partition_cache_path),
        "localSteps": arguments.local_steps,
        "nesterovMaxIterations": arguments.nesterov_max_iterations,
        "nesterovMinIterations": arguments.nesterov_min_iterations,
        "nesterovStopTolerance": arguments.nesterov_stop_tolerance,
        "threadsPerCluster": arguments.threads_per_cluster,
        "localSolver": arguments.local_solver,
        "trustRegionPolicy": arguments.trust_region_policy,
        "persistentTrustRegion": arguments.persistent_trust_region,
        "trustRegionRecoveryRatio": arguments.trust_region_recovery_ratio,
        "cameraScaling": arguments.camera_scaling,
        "scalingSeconds": scaling_seconds,
        "initializationSeconds": initialization_seconds,
        "optimizationSeconds": optimization_seconds,
        "relaxation": arguments.relaxation,
        "outerAcceleration": arguments.outer_acceleration,
        "lineSearchGrid": arguments.line_search_grid,
        "accelerationRestartAfter": arguments.acceleration_restart_after,
        "acceleratedAcceptances": accelerated_acceptances,
        "nominalFallbacks": nominal_fallbacks,
        "proximalOracleCalls": proximal_oracle_calls,
        "proximalMetric": arguments.proximal_metric,
        "consensusMetric": arguments.consensus_metric,
        "consensusExecution": arguments.consensus_execution,
        "consensusProjectionSeconds": consensus_projection_seconds,
        "workerOperationSeconds": worker_operation_seconds,
        "optimizationWorkerOperationSeconds": (
            optimization_worker_operation_seconds
        ),
        "optimizationWorkerOperationCalls": (
            optimization_worker_operation_calls
        ),
        "workerOperationCalls": worker_operation_calls,
        "transportPhaseSeconds": transport_phase_seconds,
        "optimizationTransportPhaseSeconds": (
            optimization_transport_phase_seconds
        ),
        "initialBlockRegularization": arguments.block_regularization,
        "finalBlockRegularization": block_regularization,
        "initialBlockCurvatureMultiplier": (
            arguments.block_curvature_multiplier
        ),
        "finalBlockCurvatureMultiplier": block_curvature_multiplier,
        "maximumBlockCurvatureMultiplier": (
            arguments.maximum_block_curvature_multiplier
        ),
        "blockRecoveryMode": arguments.block_recovery_mode,
        "curvatureDecayAfter": arguments.curvature_decay_after,
        "curvatureDecayRatio": arguments.curvature_decay_ratio,
        "metricDiagnosticIterations": arguments.metric_diagnostic_iterations,
        "workerSSEShadow": arguments.worker_sse_shadow,
        "workerOwnedLandmarks": arguments.worker_owned_landmarks,
        "workerOwnedCameras": arguments.worker_owned_cameras,
        "workerConsensusShadow": arguments.worker_consensus_shadow,
        "packedRequestBuffers": arguments.packed_request_buffers,
        "suppressedRoutineLandmarkReplies": (
            suppressed_routine_landmark_replies
        ),
        "suppressedAcceleratedLandmarkReplies": (
            suppressed_accelerated_landmark_replies
        ),
        "materializedAcceleratedLandmarkStates": (
            materialized_accelerated_landmark_states
        ),
        "acceleratedLandmarkReplySuppression": (
            arguments.suppress_accelerated_landmark_replies
        ),
        "maximumWorkerSSERelativeError": maximum_worker_sse_relative_error,
        "workerSSERelativeTolerance": WORKER_SSE_RELATIVE_TOLERANCE,
        "maximumWorkerConsensusRHSRelativeError": (
            maximum_worker_consensus_rhs_relative_error
        ),
        "maximumWorkerConsensusRelativeError": (
            maximum_worker_consensus_relative_error
        ),
        "workerConsensusRelativeTolerance": (
            WORKER_CONSENSUS_RELATIVE_TOLERANCE
        ),
        "landmarkRefinementSteps": arguments.landmark_refinement_steps,
        "consensusLandmarkRefinementSteps": (
            arguments.consensus_landmark_refinement_steps
        ),
        "consensusLandmarkRefinementPolicy": (
            arguments.consensus_landmark_refinement_policy
        ),
        "finalLandmarkPolishingApplied": final_polishing_applied,
        "finalLandmarkPolishingInitialSSE": final_polishing_initial_sse,
        "finalLandmarkPolishingRefinedSSE": final_polishing_refined_sse,
        "targetTransformedLipschitz": (
            arguments.target_transformed_lipschitz
        ),
        "initialPenalty": (
            arguments.penalty_multiplier * 2.5 * len(observations) / camera_count
        ),
        "finalPenalty": penalty,
        "gamma": 1.0 / penalty if arguments.proximal_metric == "scalar" else 1.0,
        "safeguardMode": arguments.safeguard_mode,
        "dreRelativeIncrease": arguments.dre_relative_increase,
        "minimumPrimalRatio": arguments.minimum_primal_ratio,
        "catastrophicRatio": arguments.catastrophic_ratio,
        "recoveryPenaltyRatio": arguments.recovery_penalty_ratio,
        "recoveryState": "last_accepted_consensus",
        "rejections": rejected_count,
        "completedIterations": len(trajectory),
        "terminationReason": termination_reason,
        "bestIteration": best_iteration,
        "partitionSeconds": partition_seconds,
        "overallSeconds": time.perf_counter() - started_at,
        "initialQualityMetrics": initial_metrics,
        "qualityMetrics": final_metrics,
        "transportBytesSent": sent_bytes,
        "transportBytesReceived": received_bytes,
        "trajectory": trajectory,
    }
    if arguments.state:
        save_bal_state(arguments.state, best_cameras, best_points, result)
        result["stateFile"] = str(Path(arguments.state).resolve())
    output_path = Path(arguments.results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(result) + "\n")
    if arguments.debug_output:
        print(
            "======== DRS DEBUG FINAL ========\n"
            f"best_sse={best_sse:.6g} best_iteration={best_iteration} "
            f"mean_px={final_metrics['meanReprojectionError']:.6g} "
            f"rmse_px={final_metrics['rmsePerObservation']:.6g}\n"
            f"rejections={rejected_count} rho={penalty:.6g} "
            f"gamma={result['gamma']:.6g} "
            f"be={block_regularization:.6g} "
            f"lip={block_curvature_multiplier:.6g} "
            f"termination={termination_reason} "
            f"optimization={result['optimizationSeconds']:.3f}s "
            f"overall={result['overallSeconds']:.3f}s\n"
            f"result={output_path} state={arguments.state or '-'}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
