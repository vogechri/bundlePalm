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
from clustering import cluster_by_landmark_scalable_stable
from drs_consensus import (
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
from admm_scaling import (
    compute_initial_jacobi_scaling,
    to_physical_cameras,
    to_scaled_cameras,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--threads-per-cluster", type=int, default=1)
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
    parser.add_argument("--debug-output", action="store_true")
    return parser.parse_args()


def validate_arguments(arguments):
    if arguments.iterations <= 0 or arguments.clusters <= 0:
        raise ValueError("iterations and clusters must be positive")
    if arguments.local_steps <= 0 or arguments.threads_per_cluster <= 0:
        raise ValueError("local steps and threads must be positive")
    if not 0.0 < arguments.relaxation < 2.0:
        raise ValueError("relaxation must be in (0, 2)")
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
    (
        camera_indices_in_cluster,
        point_indices_in_cluster,
        points_2d_in_cluster,
        cluster_count,
    ) = cluster_by_landmark_scalable_stable(
        camera_indices,
        observations,
        point_indices,
        arguments.clusters,
        camera_count,
        point_count,
        arguments.residual_balance_slack,
        arguments.minimum_camera_landmarks,
        arguments.max_refinement_passes,
    )
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
    accepted_since_curvature_change = 0
    revert_landmark_mode = 0
    trust_region_recovery_ratio = 1.0
    trajectory = []
    termination_reason = "iteration_limit"
    final_polishing_applied = False
    final_polishing_initial_sse = float("nan")
    final_polishing_refined_sse = float("nan")

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
        )
        worker.update_preconditioning(
            camera_indices_in_cluster,
            point_indices_in_cluster,
            camera_scaling,
            cluster_count,
        )
        accepted_landmarks = landmarks.copy()
        for iteration in range(arguments.iterations):
            reference_sse = accepted_metrics["sumSquaredError"]
            reference_dre = accepted_dre
            proximal_penalty = penalty
            proximal_block_regularization = block_regularization
            proximal_block_curvature_multiplier = block_curvature_multiplier
            recovery_exhausted = False
            curvature_decay_applied = False

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
                return_metric_blocks=(arguments.proximal_metric == "block"),
                return_metric_diagnostics=(
                    arguments.metric_diagnostic_iterations > 0
                ),
            )
            if (
                arguments.proximal_metric == "block"
                and arguments.metric_diagnostic_iterations > 0
            ):
                prox_costs, raw_metric_blocks, metric_diagnostics = prox_result
            elif arguments.proximal_metric == "block":
                prox_costs, raw_metric_blocks = prox_result
                metric_diagnostics = None
            else:
                prox_costs = prox_result
                raw_metric_blocks = None
                metric_diagnostics = None
            revert_landmark_mode = 0
            trust_region_recovery_ratio = 1.0

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
            physical_candidate = to_physical_cameras(
                candidate_consensus, camera_scaling
            )
            unrefined_candidate_metrics = evaluate_bal_state(
                physical_candidate,
                landmarks,
                camera_indices,
                point_indices,
                observations,
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
                proximal_displacement_cost = metric_quadratic_sum(
                    local_cameras - centers,
                    raw_metric_blocks,
                    camera_masks,
                )
                local_data_objective = float(np.sum(prox_costs))
            splitting_term = dre_splitting_term(
                local_cameras,
                candidate_consensus,
                centers,
                camera_masks,
                splitting_scale,
                metric_blocks=selected_metric_blocks,
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
            if rejected:
                rejected_count += 1
                accepted_since_curvature_change = 0
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
                revert_landmark_mode = 2
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
                metrics = candidate_metrics
                recovery_action = "none"
                accepted_since_curvature_change += 1
                if (
                    arguments.curvature_decay_after > 0
                    and accepted_since_curvature_change
                    >= arguments.curvature_decay_after
                    and block_curvature_multiplier
                    > arguments.block_curvature_multiplier
                ):
                    block_curvature_multiplier = max(
                        arguments.block_curvature_multiplier,
                        block_curvature_multiplier
                        * arguments.curvature_decay_ratio,
                    )
                    accepted_since_curvature_change = 0
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
                best_points = candidate_landmarks.copy()
            if arguments.debug_output:
                print_iteration(row, best_sse, best_iteration, prox_costs)
            if recovery_exhausted:
                termination_reason = "recovery_exhausted"
                break
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
        "localSteps": arguments.local_steps,
        "threadsPerCluster": arguments.threads_per_cluster,
        "localSolver": arguments.local_solver,
        "trustRegionPolicy": arguments.trust_region_policy,
        "persistentTrustRegion": arguments.persistent_trust_region,
        "trustRegionRecoveryRatio": arguments.trust_region_recovery_ratio,
        "cameraScaling": arguments.camera_scaling,
        "scalingSeconds": scaling_seconds,
        "relaxation": arguments.relaxation,
        "proximalMetric": arguments.proximal_metric,
        "consensusMetric": arguments.consensus_metric,
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
            f"overall={result['overallSeconds']:.3f}s\n"
            f"result={output_path} state={arguments.state or '-'}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
