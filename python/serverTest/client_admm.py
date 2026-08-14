"""Parallel CPU consensus ADMM for the standard Snavely BAL objective."""

import argparse
import importlib
import json
import os
import secrets
import sys
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np
import zmq

from admm_consensus import (
    adapt_penalty,
    admm_residuals,
    combine_proximal_terms,
    consensus_update,
    dual_update,
    rescale_scaled_duals,
    plain_drs_step,
)
from admm_acceleration import (
    augmented_consensus_merit,
    consensus_disagreement_squared,
    extrapolate_centers,
    nesterov_coefficient,
    should_fallback_acceleration,
)
from admm_scaling import (
    compute_initial_jacobi_scaling,
    to_physical_cameras,
    to_scaled_cameras,
)
from bal_evaluator import (
    canonicalize_bal_problem,
    evaluate_bal_state,
    read_bal_problem,
    save_bal_state,
)
from clustering import cluster_by_landmark_scalable_stable
from drs_consensus import ActiveCameraMetricBlocks
from drs_consensus_metrics import unpack_symmetric_camera_metric_blocks
from drs_factorized_metrics import FactorizedCameraMetric


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROTO_BUILD = Path(os.environ.get(
    "BUNDLE_PALM_PROTO_BUILD", SCRIPT_DIRECTORY / "build_admm"))
sys.path.insert(0, str(PROTO_BUILD / "generated" / "proto"))
sys.path.insert(0, str(PROTO_BUILD / "generated"))
test_pb2 = importlib.import_module("test_pb2")

_REQUIRED_PROTO_FIELDS = {
    "program_proto": {
        "block_curvature_multiplier",
        "camera_proximal_multiplier",
        "collect_camera_diagonal_metrics",
        "oracle_kind",
        "outer_iteration",
    },
    "prox_cluster_proto": {
        "block_curvature_multiplier",
        "camera_proximal_multiplier",
        "collect_camera_diagonal_metrics",
        "oracle_kind",
        "outer_iteration",
    },
    "return_cluster_proto": {
        "cameras_f64",
        "landmarks_f64",
        "interior_defect_squared",
        "landmark_interior_defect_squared",
        "linear_iterations",
        "linear_relative_residual",
        "has_factorized_metric",
        "factorized_metric_camera_ids_u32",
        "factorized_metric_diagonal_blocks_f64",
        "factorized_metric_inverse_blocks_f64",
        "factorized_metric_edge_factors_u32",
        "factorized_metric_edge_cameras_u32",
        "factorized_metric_edge_blocks_f64",
        "step_size_upper_f32",
        "unique_camera_interior_defect_squared",
    },
    "return_schur_system_proto": {
        "block_columns_u32",
        "block_rows_u32",
        "blocks_f64",
        "camera_diagonal_f64",
        "camera_ids_u32",
        "landmark_model_reduction",
        "reduced_gradient_f64",
    },
}
_missing_proto_fields = {
    message_name: sorted(
        required_fields
        - set(getattr(test_pb2, message_name).DESCRIPTOR.fields_by_name)
    )
    for message_name, required_fields in _REQUIRED_PROTO_FIELDS.items()
}
_missing_proto_fields = {
    message_name: fields
    for message_name, fields in _missing_proto_fields.items()
    if fields
}
if _missing_proto_fields:
    raise RuntimeError(
        f"stale generated protobuf module {test_pb2.__file__}: missing "
        f"{_missing_proto_fields}; rebuild {PROTO_BUILD} from "
        f"{SCRIPT_DIRECTORY / 'proto' / 'test.proto'} or set "
        "BUNDLE_PALM_PROTO_BUILD to a matching build directory"
    )


@dataclass(frozen=True)
class SingleNodeConsensusSummary:
    consensus: np.ndarray
    fixed_point_squared: float
    proximal_displacement_squared: float
    reflection_projection_squared: float
    center_step_squared: float
    splitting_term: float


@dataclass
class ClusterSchurSystem:
    camera_ids: np.ndarray
    block_rows: np.ndarray
    block_columns: np.ndarray
    blocks: np.ndarray
    reduced_gradient: np.ndarray
    camera_diagonal: np.ndarray
    landmark_model_reduction: float = 0.0


def _timed_operation(name):
    def decorate(function):
        @wraps(function)
        def timed(self, *args, **kwargs):
            started_at = time.perf_counter()
            try:
                return function(self, *args, **kwargs)
            finally:
                self.operation_seconds[name] += time.perf_counter() - started_at
                self.operation_calls[name] += 1

        return timed

    return decorate


def _debug_number(value):
    if value is None:
        return "-"
    return f"{float(value):.6g}"


def print_iteration_debug(row, best_sse, best_iteration, local_costs):
    """Print one compact, human-readable ADMM outer-iteration diagnostic."""
    iteration = row["iteration"]
    decision = "REJECT" if row["outerRejected"] else "ACCEPT"
    acceleration = "none"
    if row["accelerationCoefficient"] > 0.0:
        if row["accelerationFallback"]:
            acceleration = "fallback"
        elif row["accelerationAccepted"]:
            acceleration = "accepted"
        else:
            acceleration = "trial"
    print(
        f"{iteration:03d} ======== ADMM {decision} ======== "
        f"current={row['sumSquaredError']:.6g} "
        f"candidate={row['candidateSumSquaredError']:.6g} "
        f"best={best_sse:.6g}@{best_iteration} "
        f"mean_px={row['meanReprojectionError']:.6g} "
        f"elapsed={row['overallSeconds']:.3f}s",
        file=sys.stderr,
        flush=True,
    )
    print(
        "    residuals: "
        f"primal^2={row['primalResidualSquared']:.6g} "
        f"dual^2={row['dualResidualSquared']:.6g} | "
        f"rho_ext={row['extrinsicsPenalty']:.6g} "
        f"rho_int={row['intrinsicsPenalty']:.6g} "
        f"worker_rho=({row['workerExtrinsicsPenalty']:.6g},"
        f"{row['workerIntrinsicsPenalty']:.6g}) | "
        f"local_obj sum/min/max={np.sum(local_costs):.6g}/"
        f"{np.min(local_costs):.6g}/{np.max(local_costs):.6g}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "    control: "
        f"accel={acceleration} beta={row['accelerationCoefficient']:.6g} "
        f"trial_sse={_debug_number(row['accelerationTrialSSE'])} "
        f"trial_disagreement^2="
        f"{_debug_number(row['accelerationTrialDisagreementSquared'])} | "
        f"outer_rejections={row['outerRejections']} "
        f"accel_fallbacks={row['accelerationFallbacks']} "
        f"accel_resets={row['accelerationResets']} "
        f"recovery_damping={row['recoveryDamping']:.6g} "
        f"transport={row['transportBytesSent'] + row['transportBytesReceived']}B",
        file=sys.stderr,
        flush=True,
    )


class AdmmWorkerClient:
    def __init__(self, timeout_ms=600000):
        self.context = zmq.Context()
        self.push_socket = self.context.socket(zmq.PUSH)
        self.pull_socket = self.context.socket(zmq.PULL)
        for socket in (self.push_socket, self.pull_socket):
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            socket.setsockopt(zmq.LINGER, 0)
        request_port = os.environ.get("BUNDLE_PALM_REQUEST_PORT", "5556")
        result_port = os.environ.get("BUNDLE_PALM_RESULT_PORT", "5557")
        self.push_socket.connect(f"tcp://localhost:{request_port}")
        self.pull_socket.connect(f"tcp://localhost:{result_port}")
        self.run_id = secrets.randbits(63) or 1
        self.phase_id = 0
        self.sent_bytes = 0
        self.received_bytes = 0
        self.maximum_metric_asymmetry = 0.0
        self.last_trust_region_radii = np.empty(0, dtype=np.float64)
        self.last_linear_iterations = np.empty(0, dtype=np.int32)
        self.last_linear_relative_residuals = np.empty(0, dtype=np.float64)
        self.last_consensus_l2_sse = float("nan")
        self.last_consensus_metric_blocks = None
        self.operation_seconds = {
            "solveBatch": 0.0,
            "updatePreconditioning": 0.0,
            "buildSchurSystems": 0.0,
            "applyCameraStep": 0.0,
            "refineLandmarksAtConsensus": 0.0,
            "evaluateConsensusSSE": 0.0,
            "controlLandmarkState": 0.0,
            "materializeLandmarks": 0.0,
        }
        self.operation_calls = {name: 0 for name in self.operation_seconds}
        self.transport_phase_seconds = {
            "batchSetup": 0.0,
            "requestBuild": 0.0,
            "requestSerialize": 0.0,
            "requestSend": 0.0,
            "replySetup": 0.0,
            "replyReceiveWait": 0.0,
            "replyParse": 0.0,
            "replyDecode": 0.0,
            "batchFinalize": 0.0,
        }

    def close(self):
        if os.environ.get("BUNDLE_PALM_VALIDATE_METRIC_SYMMETRY") == "1":
            print(
                f"maximum_metric_asymmetry={self.maximum_metric_asymmetry:.17g}",
                file=sys.stderr,
                flush=True,
            )
        self.push_socket.close()
        self.pull_socket.close()
        self.context.term()

    def _send(self, request):
        started_at = time.perf_counter()
        payload = request.SerializeToString()
        self.transport_phase_seconds["requestSerialize"] += (
            time.perf_counter() - started_at
        )
        started_at = time.perf_counter()
        self.push_socket.send(payload)
        self.transport_phase_seconds["requestSend"] += (
            time.perf_counter() - started_at
        )
        self.sent_bytes += len(payload)

    @_timed_operation("solveBatch")
    def solve_batch(
        self,
        camera_indices_in_cluster,
        point_indices_in_cluster,
        points_2d_in_cluster,
        local_camera_indices,
        local_point_indices,
        local_cameras,
        landmarks,
        centers,
        extrinsics_penalty,
        intrinsics_penalty,
        split_camera_penalty,
        initialize,
        cluster_count,
        local_steps,
        local_solver,
        trust_region_policy,
        camera_scaling,
        revert_landmarks,
        persistent_trust_region,
        trust_region_recovery_ratio,
        scalar_proximal_prior=True,
        block_regularization=5e-5,
        block_curvature_multiplier=0.0,
        metric_diagnostic_iterations=0,
        proximal_defect_diagnostic=False,
        landmark_refinement_steps=0,
        override_landmarks=False,
        return_metric_blocks=False,
        return_metric_diagnostics=False,
        return_landmarks=True,
        worker_owned_cameras=False,
        return_consensus_rhs=False,
        packed_request_buffers=False,
        single_node_consensus=False,
        consensus_relaxation=1.0,
        nesterov_max_iterations=100,
        nesterov_min_iterations=1,
        nesterov_stop_tolerance=1e-2,
        diagonal_trust_damping=False,
        nesterov_relative_residual=False,
        camera_proximal_multipliers=None,
        forced_trust_region_radius=None,
        outer_iteration=-1,
        oracle_kind=0,
        collect_camera_diagonal_metrics=False,
        schur_observability_diagnostic=False,
        schur_offdiagonal_majorizer=False,
        factorized_coupled_schur_metric=False,
        huber_delta=0.0,
    ):
        batch_setup_started_at = time.perf_counter()
        self.last_consensus_metric_blocks = None
        if schur_observability_diagnostic or schur_offdiagonal_majorizer:
            raise NotImplementedError(
                "Schur observability diagnostics and off-diagonal "
                "majorization are unavailable in the recovered worker backend"
            )
        if return_consensus_rhs and not return_metric_blocks:
            raise ValueError(
                "consensus RHS requires returned camera metric blocks"
            )
        if single_node_consensus and not return_metric_blocks:
            raise ValueError(
                "single-node consensus requires block camera metrics"
            )
        if single_node_consensus and not 0.0 < consensus_relaxation < 2.0:
            raise ValueError("consensus relaxation must be in (0, 2)")
        if nesterov_max_iterations <= 0:
            raise ValueError("Nesterov maximum iterations must be positive")
        if not 1 <= nesterov_min_iterations <= nesterov_max_iterations:
            raise ValueError(
                "Nesterov minimum iterations must be between 1 and the maximum"
            )
        if not 0.0 < nesterov_stop_tolerance < 1.0:
            raise ValueError("Nesterov stop tolerance must be in (0, 1)")
        local_steps_by_cluster = np.asarray(local_steps, dtype=np.int32)
        if local_steps_by_cluster.ndim == 0:
            local_steps_by_cluster = np.full(
                cluster_count, int(local_steps_by_cluster), dtype=np.int32
            )
        elif local_steps_by_cluster.shape != (cluster_count,):
            raise ValueError(
                "local steps must be a scalar or one value per cluster"
            )
        if np.any(local_steps_by_cluster <= 0):
            raise ValueError("local steps must be positive")
        forced_trust_region_radii = None
        if forced_trust_region_radius is not None:
            forced_trust_region_radii = np.asarray(
                forced_trust_region_radius, dtype=np.float64
            )
            if forced_trust_region_radii.ndim == 0:
                forced_trust_region_radii = np.full(
                    cluster_count, float(forced_trust_region_radii)
                )
            elif forced_trust_region_radii.shape != (cluster_count,):
                raise ValueError(
                    "forced trust radius must be scalar or one per cluster"
                )
            if (
                np.any(~np.isfinite(forced_trust_region_radii))
                or np.any(forced_trust_region_radii <= 0.0)
            ):
                raise ValueError("forced trust radii must be finite and positive")
        self.phase_id += 1
        phase_id = self.phase_id
        unique_cameras_by_cluster = [
            np.unique(indices) for indices in camera_indices_in_cluster
        ]
        unique_points_by_cluster = [
            np.unique(indices) for indices in point_indices_in_cluster
        ]
        self.transport_phase_seconds["batchSetup"] += (
            time.perf_counter() - batch_setup_started_at
        )
        for cluster_id in range(cluster_count):
            request_build_started_at = time.perf_counter()
            cluster_local_steps = int(local_steps_by_cluster[cluster_id])
            unique_cameras = unique_cameras_by_cluster[cluster_id]
            unique_points = unique_points_by_cluster[cluster_id]
            request = test_pb2.request_proto()
            if initialize:
                program = request.program
                program.cameras[:] = local_cameras[cluster_id][
                    unique_cameras].ravel()
                program.landmarks[:] = landmarks[unique_points].ravel()
                program.observations[:] = points_2d_in_cluster[cluster_id].ravel()
                program.cam_id[:] = local_camera_indices[cluster_id]
                program.lm_id[:] = local_point_indices[cluster_id]
                program.iterations = cluster_local_steps
                program.be = block_regularization
                program.block_curvature_multiplier = block_curvature_multiplier
                program.metric_diagnostic_iterations = metric_diagnostic_iterations
                program.proximal_defect_diagnostic = proximal_defect_diagnostic
                program.diagonal_trust_damping = diagonal_trust_damping
                program.nesterov_relative_residual = nesterov_relative_residual
                program.landmark_refinement_steps = landmark_refinement_steps
                program.global_camera_id[:] = unique_cameras
                if camera_proximal_multipliers is not None:
                    program.camera_proximal_multiplier[:] = (
                        camera_proximal_multipliers[unique_cameras]
                    )
                program.nesterov_max_iterations = nesterov_max_iterations
                program.nesterov_min_iterations = nesterov_min_iterations
                program.nesterov_stop_tolerance = nesterov_stop_tolerance
                if forced_trust_region_radii is not None:
                    program.forced_trust_region_radius = (
                        forced_trust_region_radii[cluster_id]
                    )
                program.outer_iteration = outer_iteration
                program.oracle_kind = oracle_kind
                program.collect_camera_diagonal_metrics = (
                    collect_camera_diagonal_metrics
                )
                program.factorized_coupled_schur_metric = (
                    factorized_coupled_schur_metric
                )
                program.huber_delta = huber_delta
                program.cluster_id = cluster_id
                program.num_clusters = cluster_count
                program.run_id = self.run_id
                program.phase_id = phase_id
                if np.asarray(camera_scaling).ndim == 2:
                    program.unorm[:] = (1.0 / camera_scaling[
                        unique_cameras]).ravel()
                    program.camera_transform[:] = np.tile(
                        np.eye(9), (unique_cameras.size, 1, 1)).ravel()
                else:
                    program.unorm[:] = np.ones(9 * unique_cameras.size)
                    program.camera_transform[:] = camera_scaling[
                        unique_cameras
                    ].ravel()
                program.vnorm[:] = np.ones(3 * unique_points.size)
                program.scalar_proximal_prior = scalar_proximal_prior
                program.proximal_rho = extrinsics_penalty
                program.split_camera_penalty = split_camera_penalty
                program.proximal_rho_intrinsics = intrinsics_penalty
                program.ceres_local_solver = local_solver == "ceres_pcg"
                program.local_linear_solver = (
                    2 if local_solver == "ceres_se3"
                    else 3 if local_solver == "ceres_prox_se3"
                    else 5 if local_solver == "poba_power"
                    else 1 if local_solver == "schur_pcg"
                    else 0)
                program.trust_region_policy = (
                    1 if trust_region_policy == "daba" else 0)
                program.persistent_trust_region = persistent_trust_region
            else:
                update = request.update
                retain_worker_cameras = (
                    worker_owned_cameras and int(revert_landmarks) != 3
                )
                if not retain_worker_cameras:
                    request_cameras = np.ascontiguousarray(
                        local_cameras[cluster_id][unique_cameras],
                        dtype="<f8",
                    )
                    if packed_request_buffers:
                        update.cameras_f64 = request_cameras.tobytes()
                    else:
                        update.cameras[:] = request_cameras.ravel()
                request_centers = np.ascontiguousarray(
                    centers[cluster_id][unique_cameras], dtype="<f8"
                )
                if packed_request_buffers:
                    update.cameras_s_f64 = request_centers.tobytes()
                else:
                    update.cameras_s[:] = request_centers.ravel()
                update.retain_cameras = retain_worker_cameras
                update.revert_cameras = (
                    worker_owned_cameras and int(revert_landmarks) == 1
                )
                update.return_consensus_rhs = return_consensus_rhs
                update.single_node_consensus = single_node_consensus
                update.consensus_relaxation = consensus_relaxation
                update.nesterov_max_iterations = nesterov_max_iterations
                update.nesterov_min_iterations = nesterov_min_iterations
                update.nesterov_stop_tolerance = nesterov_stop_tolerance
                if forced_trust_region_radii is not None:
                    update.forced_trust_region_radius = (
                        forced_trust_region_radii[cluster_id]
                    )
                update.outer_iteration = outer_iteration
                update.oracle_kind = oracle_kind
                update.collect_camera_diagonal_metrics = (
                    collect_camera_diagonal_metrics
                )
                update.factorized_coupled_schur_metric = (
                    factorized_coupled_schur_metric
                )
                if collect_camera_diagonal_metrics and cluster_id == 0:
                    print(
                        "CAMERA_DIAGONAL_REQUEST_SENT "
                        f"outer_iteration={outer_iteration} "
                        f"oracle_kind={oracle_kind}",
                        file=sys.stderr,
                        flush=True,
                    )
                update.cluster_id = cluster_id
                update.run_id = self.run_id
                update.phase_id = phase_id
                update.be = block_regularization
                update.block_curvature_multiplier = block_curvature_multiplier
                update.metric_diagnostic_iterations = metric_diagnostic_iterations
                update.proximal_defect_diagnostic = proximal_defect_diagnostic
                update.diagonal_trust_damping = diagonal_trust_damping
                update.nesterov_relative_residual = nesterov_relative_residual
                if camera_proximal_multipliers is not None:
                    update.camera_proximal_multiplier[:] = (
                        camera_proximal_multipliers[unique_cameras]
                    )
                update.local_iterations = cluster_local_steps
                update.landmark_refinement_steps = landmark_refinement_steps
                update.omit_landmarks = not return_landmarks
                update.scalar_proximal_prior = scalar_proximal_prior
                update.proximal_rho = extrinsics_penalty
                update.split_camera_penalty = split_camera_penalty
                update.proximal_rho_intrinsics = intrinsics_penalty
                update.ceres_local_solver = local_solver == "ceres_pcg"
                update.local_linear_solver = (
                    2 if local_solver == "ceres_se3"
                    else 3 if local_solver == "ceres_prox_se3"
                    else 5 if local_solver == "poba_power"
                    else 1 if local_solver == "schur_pcg"
                    else 0)
                update.trust_region_policy = (
                    1 if trust_region_policy == "daba" else 0)
                update.revert_lm = int(revert_landmarks)
                update.persistent_trust_region = persistent_trust_region
                update.trust_region_recovery_ratio = (
                    trust_region_recovery_ratio)
                if int(revert_landmarks) == 2 or override_landmarks:
                    update.landmarks[:] = landmarks[unique_points].ravel()
            self.transport_phase_seconds["requestBuild"] += (
                time.perf_counter() - request_build_started_at
            )
            self._send(request)

        reply_setup_started_at = time.perf_counter()
        pending = set(range(cluster_count))
        costs = np.zeros(cluster_count)
        metric_blocks = None
        consensus_metric_blocks = None
        factorized_diagonal_clusters = []
        factorized_diagonal_cameras = []
        factorized_diagonal_blocks = []
        factorized_factor_clusters = []
        factorized_inverse_blocks = []
        factorized_edge_factors = []
        factorized_edge_cameras = []
        factorized_edge_blocks = []
        factorized_metric_reply_count = 0
        factorized_factor_offset = 0
        metric_offsets = None
        collect_metric_blocks = return_metric_blocks and not single_node_consensus
        if return_metric_blocks:
            metric_counts = np.fromiter(
                (indices.size for indices in unique_cameras_by_cluster),
                dtype=np.int64,
                count=cluster_count,
            )
            metric_offsets = np.empty(cluster_count + 1, dtype=np.int64)
            metric_offsets[0] = 0
            np.cumsum(metric_counts, out=metric_offsets[1:])
            camera_dtype = (
                np.uint16
                if local_cameras.shape[1] <= np.iinfo(np.uint16).max
                else np.uint32
            )
            cluster_dtype = (
                np.uint16
                if cluster_count <= np.iinfo(np.uint16).max
                else np.uint32
            )
            if collect_metric_blocks:
                metric_blocks = ActiveCameraMetricBlocks(
                    np.repeat(
                        np.arange(cluster_count, dtype=cluster_dtype),
                        metric_counts,
                    ),
                    np.concatenate(unique_cameras_by_cluster).astype(
                        camera_dtype, copy=False
                    ),
                    np.empty((metric_offsets[-1], 9, 9), dtype=np.float64),
                    cluster_count,
                    local_cameras.shape[1],
                )
                consensus_metric_blocks = ActiveCameraMetricBlocks(
                    metric_blocks.cluster_indices.copy(),
                    metric_blocks.camera_indices.copy(),
                    np.empty_like(metric_blocks.blocks),
                    metric_blocks.cluster_count,
                    metric_blocks.camera_count,
                )
        transformed_lipschitz = np.full(cluster_count, np.nan)
        transformed_lipschitz_residual = np.full(cluster_count, np.nan)
        metric_iterations = np.zeros(cluster_count, dtype=np.int32)
        camera_proximal_defect_squared = np.full(cluster_count, np.nan)
        landmark_proximal_defect_squared = np.full(cluster_count, np.nan)
        proximal_defect_squared = np.full(cluster_count, np.nan)
        unique_camera_interior_defect_squared = np.full(cluster_count, np.nan)
        landmark_interior_defect_squared = np.full(cluster_count, np.nan)
        interior_defect_squared = np.full(cluster_count, np.nan)
        trust_region_radii = np.full(cluster_count, np.nan)
        linear_iterations = np.zeros(cluster_count, dtype=np.int32)
        linear_relative_residuals = np.full(cluster_count, np.nan)
        consensus_rhs = (
            np.empty((metric_offsets[-1], 9), dtype=np.float64)
            if return_consensus_rhs else None
        )
        single_node_summary = None
        self.transport_phase_seconds["replySetup"] += (
            time.perf_counter() - reply_setup_started_at
        )
        while pending:
            started_at = time.perf_counter()
            payload = self.pull_socket.recv()
            self.transport_phase_seconds["replyReceiveWait"] += (
                time.perf_counter() - started_at
            )
            self.received_bytes += len(payload)
            reply = test_pb2.return_cluster_proto()
            started_at = time.perf_counter()
            reply.ParseFromString(payload)
            self.transport_phase_seconds["replyParse"] += (
                time.perf_counter() - started_at
            )
            if reply.run_id != self.run_id or reply.phase_id != phase_id:
                continue
            decode_started_at = time.perf_counter()
            cluster_id = reply.cluster_id
            if cluster_id not in pending:
                raise RuntimeError(f"duplicate ADMM reply for cluster {cluster_id}")
            pending.remove(cluster_id)
            unique_cameras = unique_cameras_by_cluster[cluster_id]
            unique_points = unique_points_by_cluster[cluster_id]
            expected_cameras = 9 * unique_cameras.size
            expected_points = 3 * unique_points.size
            reply_cameras = (
                np.frombuffer(reply.cameras_f64, dtype="<f8")
                if reply.cameras_f64
                else np.asarray(reply.cameras, dtype=np.float64)
            )
            reply_landmarks = (
                np.frombuffer(reply.landmarks_f64, dtype="<f8")
                if reply.landmarks_f64
                else np.asarray(reply.landmarks, dtype=np.float64)
            )
            if reply_cameras.size != expected_cameras:
                raise RuntimeError("ADMM worker returned an invalid camera state")
            if return_landmarks and reply_landmarks.size != expected_points:
                raise RuntimeError("ADMM worker returned an invalid landmark state")
            if not return_landmarks and reply_landmarks.size != 0:
                raise RuntimeError("ADMM worker unexpectedly returned landmarks")
            if collect_metric_blocks:
                if reply.step_size_upper_f64:
                    packed_metric_blocks = np.frombuffer(
                        reply.step_size_upper_f64, dtype="<f8"
                    )
                    if packed_metric_blocks.size != 45 * unique_cameras.size:
                        raise RuntimeError(
                            "worker returned an invalid packed camera metric state"
                        )
                    reply_metric_blocks = (
                        unpack_symmetric_camera_metric_blocks(
                            packed_metric_blocks
                        )
                    )
                elif reply.step_size_upper_f32:
                    packed_metric_blocks = np.frombuffer(
                        reply.step_size_upper_f32, dtype="<f4"
                    )
                    if packed_metric_blocks.size != 45 * unique_cameras.size:
                        raise RuntimeError(
                            "worker returned an invalid packed camera metric state"
                        )
                    reply_metric_blocks = (
                        unpack_symmetric_camera_metric_blocks(
                            packed_metric_blocks
                        )
                    )
                else:
                    expected_step_values = 81 * unique_cameras.size
                    reply_metric_blocks = (
                        np.frombuffer(reply.step_size_f32, dtype="<f4")
                        if reply.step_size_f32
                        else np.asarray(reply.step_size, dtype=np.float32)
                    )
                    if reply_metric_blocks.size != expected_step_values:
                        raise RuntimeError(
                            "worker returned an invalid camera metric block state"
                        )
                    reply_metric_blocks = reply_metric_blocks.reshape((-1, 9, 9))
                if os.environ.get("BUNDLE_PALM_VALIDATE_METRIC_SYMMETRY") == "1":
                    asymmetry = float(np.max(np.abs(
                        reply_metric_blocks
                        - np.swapaxes(reply_metric_blocks, 1, 2)
                    )))
                    self.maximum_metric_asymmetry = max(
                        self.maximum_metric_asymmetry, asymmetry
                    )
                if reply.has_factorized_metric:
                    reply_camera_ids = np.frombuffer(
                        reply.factorized_metric_camera_ids_u32, dtype="<u4"
                    ).astype(np.int64)
                    reply_diagonal_blocks = np.frombuffer(
                        reply.factorized_metric_diagonal_blocks_f64,
                        dtype="<f8",
                    )
                    reply_inverse_blocks = np.frombuffer(
                        reply.factorized_metric_inverse_blocks_f64,
                        dtype="<f8",
                    )
                    reply_edge_factors = np.frombuffer(
                        reply.factorized_metric_edge_factors_u32, dtype="<u4"
                    ).astype(np.int64)
                    reply_edge_cameras = np.frombuffer(
                        reply.factorized_metric_edge_cameras_u32, dtype="<u4"
                    ).astype(np.int64)
                    reply_edge_blocks = np.frombuffer(
                        reply.factorized_metric_edge_blocks_f64, dtype="<f8"
                    )
                    if not (
                        reply_diagonal_blocks.size == 81 * reply_camera_ids.size
                        and reply_inverse_blocks.size % 9 == 0
                        and reply_edge_factors.size == reply_edge_cameras.size
                        and reply_edge_blocks.size == 27 * reply_edge_factors.size
                    ):
                        raise RuntimeError(
                            "worker returned an invalid factorized camera metric"
                        )
                    factor_count = reply_inverse_blocks.size // 9
                    if np.any(reply_edge_factors >= factor_count):
                        raise RuntimeError(
                            "worker returned an invalid factorized edge index"
                        )
                    factorized_diagonal_clusters.append(np.full(
                        reply_camera_ids.size, cluster_id, dtype=np.int64
                    ))
                    factorized_diagonal_cameras.append(reply_camera_ids)
                    factorized_diagonal_blocks.append(
                        reply_diagonal_blocks.reshape((-1, 9, 9))
                    )
                    factorized_factor_clusters.append(np.full(
                        factor_count, cluster_id, dtype=np.int64
                    ))
                    factorized_inverse_blocks.append(
                        reply_inverse_blocks.reshape((-1, 3, 3))
                    )
                    factorized_edge_factors.append(
                        reply_edge_factors + factorized_factor_offset
                    )
                    factorized_edge_cameras.append(reply_edge_cameras)
                    factorized_edge_blocks.append(
                        reply_edge_blocks.reshape((-1, 9, 3))
                    )
                    factorized_factor_offset += factor_count
                    factorized_metric_reply_count += 1
            local_cameras[cluster_id][unique_cameras] = reply_cameras.reshape(
                -1, 9
            )
            if return_landmarks:
                landmarks[unique_points] = reply_landmarks.reshape(-1, 3)
            if collect_metric_blocks:
                metric_blocks.blocks[
                    metric_offsets[cluster_id]:metric_offsets[cluster_id + 1]
                ] = reply_metric_blocks
                if reply.consensus_step_size_upper_f64:
                    packed_consensus_blocks = np.frombuffer(
                        reply.consensus_step_size_upper_f64, dtype="<f8"
                    )
                    if packed_consensus_blocks.size != 45 * unique_cameras.size:
                        raise RuntimeError(
                            "worker returned an invalid consensus metric state"
                        )
                    reply_consensus_blocks = (
                        unpack_symmetric_camera_metric_blocks(
                            packed_consensus_blocks
                        )
                    )
                else:
                    reply_consensus_blocks = reply_metric_blocks
                consensus_metric_blocks.blocks[
                    metric_offsets[cluster_id]:metric_offsets[cluster_id + 1]
                ] = reply_consensus_blocks
            if return_consensus_rhs:
                reply_rhs = np.frombuffer(
                    reply.consensus_rhs_f64, dtype="<f8"
                )
                if reply_rhs.size != 9 * unique_cameras.size:
                    raise RuntimeError(
                        "worker returned an invalid consensus RHS"
                    )
                consensus_rhs[
                    metric_offsets[cluster_id]:metric_offsets[cluster_id + 1]
                ] = reply_rhs.reshape((-1, 9))
            if reply.has_single_node_consensus:
                if single_node_summary is not None:
                    raise RuntimeError("duplicate single-node consensus reply")
                reply_consensus = np.frombuffer(
                    reply.consensus_f64, dtype="<f8"
                )
                if reply_consensus.size != 9 * local_cameras.shape[1]:
                    raise RuntimeError(
                        "worker returned an invalid single-node consensus"
                    )
                single_node_summary = SingleNodeConsensusSummary(
                    consensus=reply_consensus.reshape((-1, 9)).copy(),
                    fixed_point_squared=reply.consensus_fixed_point_squared,
                    proximal_displacement_squared=(
                        reply.consensus_proximal_displacement_squared
                    ),
                    reflection_projection_squared=(
                        reply.consensus_reflection_projection_squared
                    ),
                    center_step_squared=reply.consensus_center_step_squared,
                    splitting_term=reply.consensus_splitting_term,
                )
            costs[cluster_id] = reply.cost
            transformed_lipschitz[cluster_id] = (
                reply.transformed_lipschitz_estimate
            )
            transformed_lipschitz_residual[cluster_id] = (
                reply.transformed_lipschitz_residual
            )
            metric_iterations[cluster_id] = reply.metric_diagnostic_iterations
            camera_proximal_defect_squared[cluster_id] = (
                reply.camera_proximal_defect_squared
            )
            landmark_proximal_defect_squared[cluster_id] = (
                reply.landmark_proximal_defect_squared
            )
            proximal_defect_squared[cluster_id] = (
                reply.proximal_defect_squared
            )
            unique_camera_interior_defect_squared[cluster_id] = (
                reply.unique_camera_interior_defect_squared
            )
            landmark_interior_defect_squared[cluster_id] = (
                reply.landmark_interior_defect_squared
            )
            interior_defect_squared[cluster_id] = (
                reply.interior_defect_squared
            )
            trust_region_radii[cluster_id] = reply.trust_region_radius
            linear_iterations[cluster_id] = reply.linear_iterations
            linear_relative_residuals[cluster_id] = (
                reply.linear_relative_residual
            )
            self.transport_phase_seconds["replyDecode"] += (
                time.perf_counter() - decode_started_at
            )
        batch_finalize_started_at = time.perf_counter()
        if factorized_metric_reply_count:
            if factorized_metric_reply_count != cluster_count:
                raise RuntimeError(
                    "workers returned a partial factorized camera metric"
                )
            metric_blocks = FactorizedCameraMetric(
                np.concatenate(factorized_diagonal_clusters),
                np.concatenate(factorized_diagonal_cameras),
                np.concatenate(factorized_diagonal_blocks),
                np.concatenate(factorized_factor_clusters),
                np.concatenate(factorized_inverse_blocks),
                np.concatenate(factorized_edge_factors),
                np.concatenate(factorized_edge_cameras),
                np.concatenate(factorized_edge_blocks),
                cluster_count,
                local_cameras.shape[1],
            )
            consensus_metric_blocks = metric_blocks
        self.last_trust_region_radii = trust_region_radii
        self.last_linear_iterations = linear_iterations
        self.last_linear_relative_residuals = linear_relative_residuals
        diagnostics = None
        if return_metric_diagnostics:
            diagnostics = {
                "transformedLipschitz": transformed_lipschitz,
                "relativeResidual": transformed_lipschitz_residual,
                "iterations": metric_iterations,
                "cameraProximalDefectSquared": (
                    camera_proximal_defect_squared
                ),
                "landmarkProximalDefectSquared": (
                    landmark_proximal_defect_squared
                ),
                "proximalDefectSquared": proximal_defect_squared,
                "uniqueCameraInteriorDefectSquared": (
                    unique_camera_interior_defect_squared
                ),
                "landmarkInteriorDefectSquared": (
                    landmark_interior_defect_squared
                ),
                "interiorDefectSquared": interior_defect_squared,
                "schurObservabilityFractions": np.full(
                    cluster_count, np.nan
                ),
            }
        self.transport_phase_seconds["batchFinalize"] += (
            time.perf_counter() - batch_finalize_started_at
        )
        self.last_consensus_metric_blocks = consensus_metric_blocks
        if single_node_consensus:
            if single_node_summary is None:
                raise RuntimeError("worker omitted single-node consensus reply")
            if return_metric_diagnostics:
                return costs, single_node_summary, diagnostics
            return costs, single_node_summary
        if return_metric_blocks and return_metric_diagnostics:
            if return_consensus_rhs:
                return costs, metric_blocks, diagnostics, consensus_rhs
            return costs, metric_blocks, diagnostics
        if return_metric_blocks:
            if return_consensus_rhs:
                return costs, metric_blocks, consensus_rhs
            return costs, metric_blocks
        if return_metric_diagnostics:
            return costs, {
                "transformedLipschitz": transformed_lipschitz,
                "relativeResidual": transformed_lipschitz_residual,
                "iterations": metric_iterations,
                "cameraProximalDefectSquared": (
                    camera_proximal_defect_squared
                ),
                "landmarkProximalDefectSquared": (
                    landmark_proximal_defect_squared
                ),
                "proximalDefectSquared": proximal_defect_squared,
                "uniqueCameraInteriorDefectSquared": (
                    unique_camera_interior_defect_squared
                ),
                "landmarkInteriorDefectSquared": (
                    landmark_interior_defect_squared
                ),
                "interiorDefectSquared": interior_defect_squared,
                "schurObservabilityFractions": np.full(
                    cluster_count, np.nan
                ),
            }
        return costs

    @_timed_operation("updatePreconditioning")
    def update_preconditioning(
        self,
        camera_indices_in_cluster,
        point_indices_in_cluster,
        camera_scaling,
        cluster_count,
        camera_state=None,
    ):
        for cluster_id in range(cluster_count):
            unique_cameras = np.unique(camera_indices_in_cluster[cluster_id])
            unique_points = np.unique(point_indices_in_cluster[cluster_id])
            request = test_pb2.request_proto()
            update = request.preconditioning_update
            if np.asarray(camera_scaling).ndim == 2:
                update.unorm[:] = (
                    1.0 / camera_scaling[unique_cameras]
                ).ravel()
            else:
                update.unorm[:] = np.ones(9 * unique_cameras.size)
                update.camera_transform[:] = camera_scaling[
                    unique_cameras
                ].ravel()
            update.vnorm[:] = np.ones(3 * unique_points.size)
            if camera_state is not None:
                update.cameras[:] = camera_state[cluster_id][
                    unique_cameras
                ].ravel()
            update.cluster_id = cluster_id
            self._send(request)

    @_timed_operation("buildSchurSystems")
    def build_schur_systems(
        self,
        camera_indices_in_cluster,
        point_indices_in_cluster,
        consensus,
        landmarks,
        cluster_count,
        landmark_damping,
    ):
        if landmark_damping < 0.0:
            raise ValueError("landmark damping must be nonnegative")
        self.phase_id += 1
        phase_id = self.phase_id
        for cluster_id in range(cluster_count):
            unique_cameras = np.unique(camera_indices_in_cluster[cluster_id])
            unique_points = np.unique(point_indices_in_cluster[cluster_id])
            request = test_pb2.request_proto()
            update = request.schur_system
            update.cluster_id = cluster_id
            update.run_id = self.run_id
            update.phase_id = phase_id
            update.cameras_f64 = np.ascontiguousarray(
                consensus[unique_cameras], dtype="<f8"
            ).tobytes()
            update.landmarks_f64 = np.ascontiguousarray(
                landmarks[unique_points], dtype="<f8"
            ).tobytes()
            update.landmark_damping = landmark_damping
            self._send(request)

        systems = [None] * cluster_count
        pending = set(range(cluster_count))
        while pending:
            payload = self.pull_socket.recv()
            self.received_bytes += len(payload)
            reply = test_pb2.return_schur_system_proto()
            reply.ParseFromString(payload)
            if reply.run_id != self.run_id or reply.phase_id != phase_id:
                continue
            cluster_id = reply.cluster_id
            if cluster_id not in pending:
                raise RuntimeError(
                    f"duplicate Schur reply for cluster {cluster_id}"
                )
            pending.remove(cluster_id)
            camera_ids = np.frombuffer(
                reply.camera_ids_u32, dtype="<u4"
            ).astype(np.int64)
            block_rows = np.frombuffer(
                reply.block_rows_u32, dtype="<u4"
            ).astype(np.int64)
            block_columns = np.frombuffer(
                reply.block_columns_u32, dtype="<u4"
            ).astype(np.int64)
            blocks = np.frombuffer(reply.blocks_f64, dtype="<f8")
            reduced_gradient = np.frombuffer(
                reply.reduced_gradient_f64, dtype="<f8"
            )
            camera_diagonal = np.frombuffer(
                reply.camera_diagonal_f64, dtype="<f8"
            )
            if block_rows.size != block_columns.size or (
                blocks.size != 81 * block_rows.size
            ):
                raise RuntimeError("worker returned invalid Schur blocks")
            if reduced_gradient.size != 9 * camera_ids.size:
                raise RuntimeError("worker returned invalid Schur gradient")
            if camera_diagonal.size != 81 * camera_ids.size:
                raise RuntimeError(
                    "worker returned invalid Schur camera diagonal"
                )
            finite_fields = {
                "blocks": np.all(np.isfinite(blocks)),
                "reduced_gradient": np.all(np.isfinite(reduced_gradient)),
                "camera_diagonal": np.all(np.isfinite(camera_diagonal)),
                "landmark_model_reduction": np.isfinite(
                    reply.landmark_model_reduction
                ),
            }
            if not all(finite_fields.values()):
                invalid = sorted(
                    name for name, finite in finite_fields.items() if not finite
                )
                raise RuntimeError(
                    f"worker cluster {cluster_id} returned non-finite Schur "
                    f"fields: {invalid}"
                )
            systems[cluster_id] = ClusterSchurSystem(
                camera_ids,
                block_rows,
                block_columns,
                blocks.reshape((-1, 9, 9)),
                reduced_gradient.reshape((-1, 9)),
                camera_diagonal.reshape((-1, 9, 9)),
                reply.landmark_model_reduction,
            )
        return systems

    @_timed_operation("applyCameraStep")
    def apply_camera_step(
        self,
        camera_indices_in_cluster,
        point_indices_in_cluster,
        consensus,
        landmarks,
        tangent_step,
        cluster_count,
        landmark_refinement_steps,
        rebase_trust_state=False,
    ):
        tangent_step = np.asarray(tangent_step, dtype=np.float64)
        if tangent_step.shape != consensus.shape:
            raise ValueError("tangent step shape must match consensus")
        if not 0 <= landmark_refinement_steps <= 20:
            raise ValueError("landmark refinement steps must be in [0, 20]")
        self.phase_id += 1
        phase_id = self.phase_id
        for cluster_id in range(cluster_count):
            unique_cameras = np.unique(camera_indices_in_cluster[cluster_id])
            unique_points = np.unique(point_indices_in_cluster[cluster_id])
            request = test_pb2.request_proto()
            update = request.camera_step
            update.cluster_id = cluster_id
            update.run_id = self.run_id
            update.phase_id = phase_id
            update.cameras_f64 = np.ascontiguousarray(
                consensus[unique_cameras], dtype="<f8"
            ).tobytes()
            update.landmarks_f64 = np.ascontiguousarray(
                landmarks[unique_points], dtype="<f8"
            ).tobytes()
            update.tangent_step_f64 = np.ascontiguousarray(
                tangent_step[unique_cameras], dtype="<f8"
            ).tobytes()
            update.landmark_refinement_steps = landmark_refinement_steps
            update.rebase_trust_state = rebase_trust_state
            self._send(request)

        corrected_cameras = np.full_like(consensus, np.nan)
        corrected_landmarks = landmarks.copy()
        costs = np.empty(cluster_count, dtype=np.float64)
        trust_region_radii = np.empty(cluster_count, dtype=np.float64)
        pending = set(range(cluster_count))
        while pending:
            payload = self.pull_socket.recv()
            self.received_bytes += len(payload)
            reply = test_pb2.return_cluster_proto()
            reply.ParseFromString(payload)
            if reply.run_id != self.run_id or reply.phase_id != phase_id:
                continue
            cluster_id = reply.cluster_id
            if cluster_id not in pending:
                raise RuntimeError(
                    f"duplicate camera-step reply for cluster {cluster_id}"
                )
            pending.remove(cluster_id)
            unique_cameras = np.unique(camera_indices_in_cluster[cluster_id])
            unique_points = np.unique(point_indices_in_cluster[cluster_id])
            local_cameras = np.frombuffer(
                reply.cameras_f64, dtype="<f8"
            ).reshape((-1, 9))
            local_landmarks = np.frombuffer(
                reply.landmarks_f64, dtype="<f8"
            ).reshape((-1, 3))
            if local_cameras.shape[0] != unique_cameras.size:
                raise RuntimeError("worker returned invalid corrected cameras")
            if local_landmarks.shape[0] != unique_points.size:
                raise RuntimeError("worker returned invalid corrected landmarks")
            existing = corrected_cameras[unique_cameras]
            overlap = np.all(np.isfinite(existing), axis=1)
            if np.any(overlap) and not np.allclose(
                existing[overlap], local_cameras[overlap], rtol=1e-10, atol=1e-12
            ):
                raise RuntimeError("workers disagree on corrected camera state")
            corrected_cameras[unique_cameras] = local_cameras
            corrected_landmarks[unique_points] = local_landmarks
            costs[cluster_id] = reply.cost
            trust_region_radii[cluster_id] = reply.trust_region_radius
        if not np.all(np.isfinite(corrected_cameras)):
            raise RuntimeError("corrected camera state is incomplete")
        self.last_trust_region_radii = trust_region_radii
        return costs, corrected_cameras, corrected_landmarks

    @_timed_operation("refineLandmarksAtConsensus")
    def refine_landmarks_at_consensus(
        self,
        camera_indices_in_cluster,
        point_indices_in_cluster,
        consensus,
        landmarks,
        cluster_count,
        refinement_steps,
        use_landmark_state=False,
        preserve_cameras=False,
        packed_request_buffers=False,
    ):
        self.phase_id += 1
        phase_id = self.phase_id
        for cluster_id in range(cluster_count):
            unique_cameras = np.unique(camera_indices_in_cluster[cluster_id])
            unique_points = np.unique(point_indices_in_cluster[cluster_id])
            request = test_pb2.request_proto()
            update = request.cost_update
            request_cameras = np.ascontiguousarray(
                consensus[unique_cameras], dtype="<f8"
            )
            if packed_request_buffers:
                update.cameras_f64 = request_cameras.tobytes()
            else:
                update.cameras[:] = request_cameras.ravel()
            if use_landmark_state:
                update.landmarks[:] = landmarks[unique_points].ravel()
            update.cluster_id = cluster_id
            update.run_id = self.run_id
            update.phase_id = phase_id
            update.landmark_refinement_steps = refinement_steps
            update.preserve_cameras = preserve_cameras
            self._send(request)

        pending = set(range(cluster_count))
        costs = np.zeros(cluster_count)
        refined_landmarks = landmarks.copy()
        while pending:
            payload = self.pull_socket.recv()
            self.received_bytes += len(payload)
            reply = test_pb2.return_cost_proto()
            reply.ParseFromString(payload)
            if reply.run_id != self.run_id or reply.phase_id != phase_id:
                continue
            cluster_id = reply.cluster_id
            if cluster_id not in pending:
                raise RuntimeError(
                    f"duplicate refinement reply for cluster {cluster_id}"
                )
            pending.remove(cluster_id)
            unique_points = np.unique(point_indices_in_cluster[cluster_id])
            if len(reply.landmarks) != 3 * unique_points.size:
                raise RuntimeError(
                    "worker returned an invalid refined landmark state"
                )
            refined_landmarks[unique_points] = np.asarray(
                reply.landmarks, dtype=np.float64
            ).reshape((-1, 3))
            costs[cluster_id] = reply.cost
        return costs, refined_landmarks

    @_timed_operation("evaluateConsensusSSE")
    def evaluate_consensus_sse(
        self,
        camera_indices_in_cluster,
        camera_state,
        cluster_count,
        preserve_cameras=False,
        packed_request_buffers=False,
    ):
        """Evaluate global or per-cluster cameras against worker landmark states."""
        camera_state = np.asarray(camera_state, dtype=np.float64)
        if camera_state.ndim not in (2, 3):
            raise ValueError("camera state must be global or cluster-local")
        if camera_state.ndim == 3 and camera_state.shape[0] != cluster_count:
            raise ValueError("cluster-local camera state has an invalid shape")
        self.phase_id += 1
        phase_id = self.phase_id
        for cluster_id in range(cluster_count):
            unique_cameras = np.unique(camera_indices_in_cluster[cluster_id])
            request = test_pb2.request_proto()
            update = request.cost_update
            cluster_cameras = (
                camera_state
                if camera_state.ndim == 2
                else camera_state[cluster_id]
            )
            request_cameras = np.ascontiguousarray(
                cluster_cameras[unique_cameras], dtype="<f8"
            )
            if packed_request_buffers:
                update.cameras_f64 = request_cameras.tobytes()
            else:
                update.cameras[:] = request_cameras.ravel()
            update.cluster_id = cluster_id
            update.run_id = self.run_id
            update.phase_id = phase_id
            update.omit_landmarks = True
            update.preserve_cameras = preserve_cameras
            self._send(request)

        pending = set(range(cluster_count))
        costs = np.zeros(cluster_count, dtype=np.float64)
        l2_costs = np.zeros(cluster_count, dtype=np.float64)
        while pending:
            payload = self.pull_socket.recv()
            self.received_bytes += len(payload)
            reply = test_pb2.return_cost_proto()
            reply.ParseFromString(payload)
            if reply.run_id != self.run_id or reply.phase_id != phase_id:
                continue
            cluster_id = reply.cluster_id
            if cluster_id not in pending:
                raise RuntimeError(
                    f"duplicate consensus-cost reply for cluster {cluster_id}"
                )
            pending.remove(cluster_id)
            if reply.landmarks:
                raise RuntimeError("worker unexpectedly returned landmarks")
            costs[cluster_id] = reply.precise_cost
            l2_costs[cluster_id] = reply.precise_l2_cost
        self.last_consensus_l2_sse = float(np.sum(l2_costs))
        return float(np.sum(costs))

    @_timed_operation("controlLandmarkState")
    def control_nominal_landmark_state(
        self,
        cluster_count,
        state_id,
        operation,
    ):
        """Save, restore, or discard one versioned nominal landmark snapshot."""
        operations = {
            "save": test_pb2.landmark_state_proto.SAVE_NOMINAL,
            "restore": test_pb2.landmark_state_proto.RESTORE_NOMINAL,
            "restore_roundtrip": (
                test_pb2.landmark_state_proto.RESTORE_NOMINAL_ROUNDTRIP
            ),
            "discard": test_pb2.landmark_state_proto.DISCARD_NOMINAL,
            "save_accepted": test_pb2.landmark_state_proto.SAVE_ACCEPTED,
            "restore_accepted": test_pb2.landmark_state_proto.RESTORE_ACCEPTED,
            "save_best": test_pb2.landmark_state_proto.SAVE_BEST,
            "restore_best": test_pb2.landmark_state_proto.RESTORE_BEST,
        }
        if operation not in operations:
            raise ValueError(f"unknown landmark state operation: {operation}")
        if state_id <= 0:
            raise ValueError("landmark state ID must be positive")
        self.phase_id += 1
        phase_id = self.phase_id
        for cluster_id in range(cluster_count):
            request = test_pb2.request_proto()
            state = request.landmark_state
            state.cluster_id = cluster_id
            state.run_id = self.run_id
            state.phase_id = phase_id
            state.state_id = state_id
            state.operation = operations[operation]
            self._send(request)

        pending = set(range(cluster_count))
        while pending:
            payload = self.pull_socket.recv()
            self.received_bytes += len(payload)
            reply = test_pb2.landmark_state_reply_proto()
            reply.ParseFromString(payload)
            if reply.run_id != self.run_id or reply.phase_id != phase_id:
                continue
            cluster_id = reply.cluster_id
            if cluster_id not in pending:
                raise RuntimeError(
                    f"duplicate landmark-state reply for cluster {cluster_id}"
                )
            if reply.state_id != state_id:
                raise RuntimeError("worker acknowledged the wrong landmark state")
            pending.remove(cluster_id)

    @_timed_operation("materializeLandmarks")
    def materialize_current_landmarks(
        self,
        point_indices_in_cluster,
        landmarks,
        cluster_count,
        state_id,
        source="current",
    ):
        """Fetch a packed worker landmark state through a versioned snapshot."""
        if state_id <= 0:
            raise ValueError("landmark state ID must be positive")
        operations = {
            "current": test_pb2.landmark_state_proto.MATERIALIZE_CURRENT,
            "best": test_pb2.landmark_state_proto.MATERIALIZE_BEST,
        }
        if source not in operations:
            raise ValueError(f"unknown landmark materialization source: {source}")
        self.phase_id += 1
        phase_id = self.phase_id
        for cluster_id in range(cluster_count):
            request = test_pb2.request_proto()
            state = request.landmark_state
            state.cluster_id = cluster_id
            state.run_id = self.run_id
            state.phase_id = phase_id
            state.state_id = state_id
            state.operation = operations[source]
            self._send(request)

        materialized = landmarks.copy()
        pending = set(range(cluster_count))
        while pending:
            payload = self.pull_socket.recv()
            self.received_bytes += len(payload)
            reply = test_pb2.landmark_state_reply_proto()
            reply.ParseFromString(payload)
            if reply.run_id != self.run_id or reply.phase_id != phase_id:
                continue
            cluster_id = reply.cluster_id
            if cluster_id not in pending:
                raise RuntimeError(
                    f"duplicate landmark materialization for cluster {cluster_id}"
                )
            if reply.state_id != state_id:
                raise RuntimeError("worker materialized the wrong landmark state")
            unique_points = np.unique(point_indices_in_cluster[cluster_id])
            reply_landmarks = np.frombuffer(reply.landmarks_f64, dtype="<f8")
            if reply_landmarks.size != 3 * unique_points.size:
                raise RuntimeError(
                    "worker returned an invalid materialized landmark state"
                )
            materialized[unique_points] = reply_landmarks.reshape((-1, 3))
            pending.remove(cluster_id)
        return materialized


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument(
        "--outer-method",
        choices=("admm", "plain_drs"),
        default="admm",
    )
    parser.add_argument(
        "--outer-acceleration",
        choices=("none", "nesterov"),
        default="none",
    )
    parser.add_argument("--acceleration-cost-ratio", type=float, default=1.01)
    parser.add_argument(
        "--acceleration-disagreement-ratio", type=float, default=1.01)
    parser.add_argument(
        "--acceleration-best-cost-ratio", type=float, default=1e4)
    parser.add_argument("--local-steps", type=int, default=15)
    parser.add_argument("--threads-per-cluster", type=int)
    parser.add_argument(
        "--camera-scaling",
        choices=("none", "jacobi_initial"),
        default="none",
    )
    parser.add_argument("--camera-scaling-maximum-ratio", type=float)
    parser.add_argument("--camera-scaling-clipping-percentile", type=float)
    parser.add_argument(
        "--local-solver",
        choices=("ceres_pcg", "ceres_prox_se3", "schur_pcg", "nesterov"),
        default="ceres_pcg",
    )
    parser.add_argument(
        "--trust-region-policy",
        choices=("ceres", "drs", "daba"),
        default="ceres",
    )
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument(
        "--penalty-mode",
        choices=("adaptive", "fixed", "adaptive_capped"),
        default="adaptive",
    )
    parser.add_argument("--maximum-penalty", type=float, default=1e8)
    parser.add_argument("--initial-penalty-multiplier", type=float, default=1.0)
    parser.add_argument("--outer-safeguard-ratio", type=float)
    parser.add_argument("--outer-recovery-penalty-ratio", type=float, default=2.0)
    parser.add_argument(
        "--outer-recovery-mode",
        choices=("penalty", "temporary_proximal", "trust_region"),
        default="penalty",
    )
    parser.add_argument("--outer-recovery-stable-steps", type=int, default=2)
    parser.add_argument("--outer-recovery-decay", type=float, default=0.5)
    parser.add_argument("--outer-recovery-cutoff-ratio", type=float, default=0.05)
    parser.add_argument(
        "--outer-recovery-trust-region-ratio", type=float, default=0.5)
    parser.add_argument(
        "--camera-penalty-groups",
        choices=("combined", "extrinsics_intrinsics"),
        default="combined",
    )
    parser.add_argument(
        "--penalty-adaptation-groups",
        choices=("combined", "independent"),
        default="combined",
    )
    parser.add_argument("--results", default="results_admm.jsonl")
    parser.add_argument("--state")
    parser.add_argument("--variant-name", default="baseline")
    parser.add_argument("--residual-balance-slack", type=float, default=0.01)
    parser.add_argument("--minimum-camera-landmarks", type=int, default=20)
    parser.add_argument("--max-refinement-passes", type=int, default=3)
    parser.add_argument(
        "--debug-output",
        action="store_true",
        help="print human-readable setup and per-iteration diagnostics to stderr",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.iterations <= 0 or arguments.clusters <= 0:
        raise ValueError("iterations and clusters must be positive")
    if arguments.local_steps <= 0:
        raise ValueError("local_steps must be positive")
    if arguments.outer_acceleration != "none" and arguments.outer_method != "admm":
        raise ValueError("outer acceleration currently requires ADMM")
    if (
        not np.isfinite(arguments.acceleration_cost_ratio)
        or arguments.acceleration_cost_ratio < 1.0
    ):
        raise ValueError("acceleration_cost_ratio must be finite and at least one")
    if (
        not np.isfinite(arguments.acceleration_disagreement_ratio)
        or arguments.acceleration_disagreement_ratio < 1.0
    ):
        raise ValueError(
            "acceleration_disagreement_ratio must be finite and at least one")
    if (
        not np.isfinite(arguments.acceleration_best_cost_ratio)
        or arguments.acceleration_best_cost_ratio < 1.0
    ):
        raise ValueError(
            "acceleration_best_cost_ratio must be finite and at least one")
    if arguments.threads_per_cluster is not None and (
        arguments.threads_per_cluster <= 0
    ):
        raise ValueError("threads_per_cluster must be positive")
    if arguments.maximum_penalty <= 0:
        raise ValueError("maximum_penalty must be positive")
    if arguments.outer_safeguard_ratio is not None and (
        not np.isfinite(arguments.outer_safeguard_ratio)
        or arguments.outer_safeguard_ratio < 1.0
    ):
        raise ValueError("outer_safeguard_ratio must be finite and at least one")
    if (
        not np.isfinite(arguments.outer_recovery_penalty_ratio)
        or arguments.outer_recovery_penalty_ratio <= 1.0
    ):
        raise ValueError(
            "outer_recovery_penalty_ratio must be finite and greater than one")
    if arguments.outer_recovery_stable_steps <= 0:
        raise ValueError("outer_recovery_stable_steps must be positive")
    if (
        not np.isfinite(arguments.outer_recovery_decay)
        or not 0.0 < arguments.outer_recovery_decay < 1.0
    ):
        raise ValueError("outer_recovery_decay must be finite and in (0, 1)")
    if (
        not np.isfinite(arguments.outer_recovery_cutoff_ratio)
        or arguments.outer_recovery_cutoff_ratio < 0.0
    ):
        raise ValueError(
            "outer_recovery_cutoff_ratio must be finite and nonnegative")
    if (
        arguments.outer_recovery_mode == "temporary_proximal"
        and arguments.outer_method != "admm"
    ):
        raise ValueError("temporary proximal recovery requires ADMM")
    if (
        not np.isfinite(arguments.outer_recovery_trust_region_ratio)
        or not 0.0 < arguments.outer_recovery_trust_region_ratio < 1.0
    ):
        raise ValueError(
            "outer_recovery_trust_region_ratio must be finite and in (0, 1)")
    if (
        arguments.outer_recovery_mode == "trust_region"
        and arguments.trust_region_policy == "ceres"
    ):
        raise ValueError("trust-region recovery requires a custom local solver")
    if arguments.local_solver == "ceres_pcg":
        if arguments.trust_region_policy != "ceres":
            raise ValueError(
                "ceres_pcg requires Ceres's internal trust-region policy")
    elif arguments.trust_region_policy == "ceres":
        raise ValueError(
            "custom Schur solvers require drs or daba trust-region policy")
    if (
        not np.isfinite(arguments.initial_penalty_multiplier)
        or arguments.initial_penalty_multiplier <= 0.0
    ):
        raise ValueError("initial_penalty_multiplier must be finite and positive")
    if arguments.camera_scaling_maximum_ratio is not None and (
        not np.isfinite(arguments.camera_scaling_maximum_ratio)
        or arguments.camera_scaling_maximum_ratio < 1.0
    ):
        raise ValueError(
            "camera_scaling_maximum_ratio must be finite and at least one")
    if arguments.camera_scaling_clipping_percentile is not None and not (
        0.0 <= arguments.camera_scaling_clipping_percentile < 50.0
    ):
        raise ValueError(
            "camera_scaling_clipping_percentile must be in [0, 50)")

    started_at = time.perf_counter()
    raw_cameras, raw_points, camera_indices, point_indices, raw_observations = (
        read_bal_problem(arguments.dataset))
    cameras, points, observations = canonicalize_bal_problem(
        raw_cameras, raw_points, camera_indices, raw_observations)
    camera_count = cameras.shape[0]
    point_count = points.shape[0]
    initial_metrics = evaluate_bal_state(
        cameras, points, camera_indices, point_indices, observations)
    if arguments.debug_output:
        print(
            "======== ADMM DEBUG SETUP ========\n"
            f"dataset={Path(arguments.dataset).name} "
            f"cameras={camera_count} points={point_count} "
            f"observations={len(observations)}\n"
            f"method={arguments.outer_method} "
            f"acceleration={arguments.outer_acceleration} "
            f"local_solver={arguments.local_solver} "
            f"trust={arguments.trust_region_policy} "
            f"local_steps={arguments.local_steps}\n"
            f"initial_sse={initial_metrics['sumSquaredError']:.6g} "
            f"initial_mean_px={initial_metrics['meanReprojectionError']:.6g}",
            file=sys.stderr,
            flush=True,
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
        raise RuntimeError("ADMM partitioner changed the requested cluster count")

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
    else:
        camera_scaling = np.ones_like(cameras)
    scaling_seconds = time.perf_counter() - scaling_started

    local_camera_indices = [
        np.unique(indices, return_inverse=True)[1]
        for indices in camera_indices_in_cluster]
    local_point_indices = [
        np.unique(indices, return_inverse=True)[1]
        for indices in point_indices_in_cluster]
    camera_masks = np.zeros((cluster_count, camera_count), dtype=bool)
    for cluster_id, indices in enumerate(camera_indices_in_cluster):
        camera_masks[cluster_id, np.unique(indices)] = True
    if arguments.debug_output:
        observations_per_cluster = np.asarray([
            len(indices) for indices in camera_indices_in_cluster
        ])
        cameras_per_cluster = np.sum(camera_masks, axis=1)
        print(
            "======== ADMM DEBUG PARTITION ========\n"
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

    scaled_cameras = to_scaled_cameras(cameras, camera_scaling)
    local_cameras = np.repeat(
        scaled_cameras[None, :, :], cluster_count, axis=0)
    scaled_duals = np.zeros_like(local_cameras)
    consensus = scaled_cameras.copy()
    drs_centers = np.repeat(
        scaled_cameras[None, :, :], cluster_count, axis=0)
    landmarks = points.copy()
    initial_penalty = (
        arguments.initial_penalty_multiplier
        * 2.5 * observations.shape[0] / camera_count
    )
    extrinsics_penalty = initial_penalty
    intrinsics_penalty = initial_penalty
    split_camera_penalty = (
        arguments.camera_penalty_groups == "extrinsics_intrinsics")
    independent_penalty_adaptation = (
        arguments.penalty_adaptation_groups == "independent")
    if independent_penalty_adaptation and not split_camera_penalty:
        raise ValueError(
            "independent penalty adaptation requires split camera penalties")
    trajectory = []
    best_sse = initial_metrics["sumSquaredError"]
    best_iteration = -1
    best_cameras = cameras.copy()
    best_points = landmarks.copy()
    accepted_metrics = initial_metrics
    revert_landmarks = False
    outer_rejections = 0
    recovery_damping = 0.0
    recovery_stable_steps = 0
    trust_region_recovery_ratio = 1.0
    previous_proximal_centers = (
        consensus[None, :, :] - scaled_duals).copy()
    acceleration_reset_iteration = 0
    acceleration_trials = 0
    acceleration_fallbacks = 0
    acceleration_resets = 0
    accepted_disagreement = 0.0
    accepted_acceleration_merit = initial_metrics["sumSquaredError"]

    worker = AdmmWorkerClient()
    try:
        initialized = False
        for iteration in range(arguments.iterations):
            previous_local_cameras = local_cameras.copy()
            previous_landmarks = landmarks.copy()
            previous_scaled_duals = scaled_duals.copy()
            previous_extrinsics_penalty = extrinsics_penalty
            previous_intrinsics_penalty = intrinsics_penalty
            if arguments.outer_method == "admm":
                centers = consensus[None, :, :] - scaled_duals
            else:
                centers = drs_centers
            acceleration_coefficient = 0.0
            acceleration_trial_sse = None
            acceleration_trial_disagreement = None
            acceleration_trial_merit = None
            acceleration_fallback = False
            acceleration_accepted = False
            trial_centers = centers
            if arguments.outer_acceleration == "nesterov":
                acceleration_coefficient = nesterov_coefficient(
                    iteration, acceleration_reset_iteration)
                if acceleration_coefficient > 0.0:
                    trial_centers = extrapolate_centers(
                        centers,
                        previous_proximal_centers,
                        acceleration_coefficient,
                    )
                    acceleration_trials += 1
            worker_centers = trial_centers
            worker_extrinsics_penalty = extrinsics_penalty
            worker_intrinsics_penalty = intrinsics_penalty
            if recovery_damping > 0.0:
                worker_centers = trial_centers.copy()
                worker_centers[:, :, :6], worker_extrinsics_penalty = (
                    combine_proximal_terms(
                        trial_centers[:, :, :6],
                        previous_local_cameras[:, :, :6],
                        extrinsics_penalty, recovery_damping))
                worker_centers[:, :, 6:], worker_intrinsics_penalty = (
                    combine_proximal_terms(
                        trial_centers[:, :, 6:],
                        previous_local_cameras[:, :, 6:],
                        intrinsics_penalty, recovery_damping))
            for _ in range(1):
                costs = worker.solve_batch(
                    camera_indices_in_cluster,
                    point_indices_in_cluster,
                    points_2d_in_cluster,
                    local_camera_indices,
                    local_point_indices,
                    local_cameras,
                    landmarks,
                    worker_centers,
                    worker_extrinsics_penalty,
                    worker_intrinsics_penalty,
                    split_camera_penalty,
                    initialize=not initialized,
                    cluster_count=cluster_count,
                    local_steps=arguments.local_steps,
                    local_solver=arguments.local_solver,
                    trust_region_policy=arguments.trust_region_policy,
                    camera_scaling=camera_scaling,
                    revert_landmarks=revert_landmarks,
                    persistent_trust_region=(
                        arguments.outer_recovery_mode == "trust_region"),
                    trust_region_recovery_ratio=trust_region_recovery_ratio,
                )
                initialized = True
                revert_landmarks = False
                trust_region_recovery_ratio = 1.0

            if acceleration_coefficient > 0.0:
                accelerated_consensus = consensus_update(
                    local_cameras,
                    scaled_duals,
                    camera_masks,
                    consensus,
                    penalties=np.full(cluster_count, extrinsics_penalty),
                )
                acceleration_trial_disagreement = (
                    consensus_disagreement_squared(
                        local_cameras, accelerated_consensus, camera_masks))
                accelerated_metrics = evaluate_bal_state(
                    to_physical_cameras(accelerated_consensus, camera_scaling),
                    landmarks,
                    camera_indices,
                    point_indices,
                    observations,
                )
                acceleration_trial_sse = accelerated_metrics["sumSquaredError"]
                acceleration_trial_merit = augmented_consensus_merit(
                    acceleration_trial_sse,
                    acceleration_trial_disagreement,
                    extrinsics_penalty,
                )
                acceleration_fallback = should_fallback_acceleration(
                    acceleration_trial_sse,
                    accepted_metrics["sumSquaredError"],
                    best_sse,
                    acceleration_trial_merit,
                    accepted_acceleration_merit,
                    arguments.acceleration_cost_ratio,
                    arguments.acceleration_disagreement_ratio,
                    arguments.acceleration_best_cost_ratio,
                )
                if acceleration_fallback:
                    acceleration_fallbacks += 1
                    acceleration_resets += 1
                    acceleration_reset_iteration = iteration
                    local_cameras = previous_local_cameras.copy()
                    landmarks = previous_landmarks.copy()
                    worker_centers = centers
                    worker_extrinsics_penalty = extrinsics_penalty
                    worker_intrinsics_penalty = intrinsics_penalty
                    if recovery_damping > 0.0:
                        worker_centers = centers.copy()
                        worker_centers[:, :, :6], worker_extrinsics_penalty = (
                            combine_proximal_terms(
                                centers[:, :, :6],
                                previous_local_cameras[:, :, :6],
                                extrinsics_penalty,
                                recovery_damping,
                            ))
                        worker_centers[:, :, 6:], worker_intrinsics_penalty = (
                            combine_proximal_terms(
                                centers[:, :, 6:],
                                previous_local_cameras[:, :, 6:],
                                intrinsics_penalty,
                                recovery_damping,
                            ))
                    costs = worker.solve_batch(
                        camera_indices_in_cluster,
                        point_indices_in_cluster,
                        points_2d_in_cluster,
                        local_camera_indices,
                        local_point_indices,
                        local_cameras,
                        landmarks,
                        worker_centers,
                        worker_extrinsics_penalty,
                        worker_intrinsics_penalty,
                        split_camera_penalty,
                        initialize=False,
                        cluster_count=cluster_count,
                        local_steps=arguments.local_steps,
                        local_solver=arguments.local_solver,
                        trust_region_policy=arguments.trust_region_policy,
                        camera_scaling=camera_scaling,
                        revert_landmarks=True,
                        persistent_trust_region=(
                            arguments.outer_recovery_mode == "trust_region"),
                        trust_region_recovery_ratio=1.0,
                    )
                else:
                    acceleration_accepted = True

            previous_consensus = consensus.copy()
            if arguments.outer_method == "admm":
                consensus = consensus_update(
                    local_cameras,
                    scaled_duals,
                    camera_masks,
                    previous_consensus,
                    penalties=np.full(cluster_count, extrinsics_penalty),
                )
                scaled_duals = dual_update(
                    scaled_duals,
                    local_cameras,
                    consensus,
                    camera_masks,
                    alpha=arguments.alpha,
                )
                extrinsics_residuals = admm_residuals(
                    previous_local_cameras[:, :, :6],
                    consensus[:, :6],
                    previous_consensus[:, :6],
                    camera_masks,
                )
                intrinsics_residuals = admm_residuals(
                    previous_local_cameras[:, :, 6:],
                    consensus[:, 6:],
                    previous_consensus[:, 6:],
                    camera_masks,
                )
                residuals = admm_residuals(
                    previous_local_cameras,
                    consensus,
                    previous_consensus,
                    camera_masks,
                )
                if arguments.penalty_mode == "fixed":
                    extrinsics_penalty_ratio = 1.0
                    intrinsics_penalty_ratio = 1.0
                else:
                    next_extrinsics_penalty, extrinsics_penalty_ratio = (
                        adapt_penalty(
                            extrinsics_penalty,
                            extrinsics_residuals
                            if independent_penalty_adaptation
                            else residuals,
                            initial_penalty,
                        ))
                    if independent_penalty_adaptation:
                        next_intrinsics_penalty, intrinsics_penalty_ratio = (
                            adapt_penalty(
                                intrinsics_penalty,
                                intrinsics_residuals,
                                initial_penalty,
                            ))
                    else:
                        next_intrinsics_penalty = next_extrinsics_penalty
                        intrinsics_penalty_ratio = extrinsics_penalty_ratio
                    if arguments.penalty_mode == "adaptive_capped":
                        next_extrinsics_penalty = min(
                            next_extrinsics_penalty, arguments.maximum_penalty)
                        next_intrinsics_penalty = min(
                            next_intrinsics_penalty, arguments.maximum_penalty)
                        extrinsics_penalty_ratio = (
                            next_extrinsics_penalty / extrinsics_penalty)
                        intrinsics_penalty_ratio = (
                            next_intrinsics_penalty / intrinsics_penalty)
                    extrinsics_penalty = next_extrinsics_penalty
                    intrinsics_penalty = next_intrinsics_penalty
                scaled_duals[:, :, :6] = rescale_scaled_duals(
                    scaled_duals[:, :, :6], extrinsics_penalty_ratio)
                scaled_duals[:, :, 6:] = rescale_scaled_duals(
                    scaled_duals[:, :, 6:], intrinsics_penalty_ratio)
            else:
                consensus, drs_centers, residuals = plain_drs_step(
                    local_cameras,
                    drs_centers,
                    camera_masks,
                    previous_consensus,
                )

            physical_consensus = to_physical_cameras(
                consensus, camera_scaling)
            metrics = evaluate_bal_state(
                physical_consensus,
                landmarks,
                camera_indices,
                point_indices,
                observations,
            )
            candidate_sse = metrics["sumSquaredError"]
            rejected_outer = False
            if arguments.outer_safeguard_ratio is not None:
                rejected_outer = (
                    not np.isfinite(candidate_sse)
                    or candidate_sse > (
                        arguments.outer_safeguard_ratio
                        * accepted_metrics["sumSquaredError"])
                )
            if rejected_outer:
                outer_rejections += 1
                local_cameras = previous_local_cameras
                landmarks = previous_landmarks
                consensus = previous_consensus
                scaled_duals = previous_scaled_duals
                if arguments.outer_recovery_mode == "penalty":
                    recovery_ratio = arguments.outer_recovery_penalty_ratio
                    extrinsics_penalty = (
                        previous_extrinsics_penalty * recovery_ratio)
                    intrinsics_penalty = (
                        previous_intrinsics_penalty * recovery_ratio)
                    scaled_duals[:, :, :6] = rescale_scaled_duals(
                        scaled_duals[:, :, :6], recovery_ratio)
                    scaled_duals[:, :, 6:] = rescale_scaled_duals(
                        scaled_duals[:, :, 6:], recovery_ratio)
                elif arguments.outer_recovery_mode == "temporary_proximal":
                    extrinsics_penalty = previous_extrinsics_penalty
                    intrinsics_penalty = previous_intrinsics_penalty
                    recovery_damping = (
                        previous_extrinsics_penalty
                        if recovery_damping == 0.0
                        else recovery_damping
                        * arguments.outer_recovery_penalty_ratio)
                    recovery_stable_steps = 0
                else:
                    extrinsics_penalty = previous_extrinsics_penalty
                    intrinsics_penalty = previous_intrinsics_penalty
                    trust_region_recovery_ratio = (
                        arguments.outer_recovery_trust_region_ratio)
                metrics = accepted_metrics
                physical_consensus = to_physical_cameras(
                    consensus, camera_scaling)
                revert_landmarks = True
                if arguments.outer_acceleration == "nesterov":
                    acceleration_resets += 1
                    acceleration_reset_iteration = iteration
            else:
                accepted_metrics = metrics
                accepted_disagreement = consensus_disagreement_squared(
                    local_cameras, consensus, camera_masks)
                accepted_acceleration_merit = augmented_consensus_merit(
                    metrics["sumSquaredError"],
                    accepted_disagreement,
                    extrinsics_penalty,
                )
                if recovery_damping > 0.0:
                    recovery_stable_steps += 1
                    if recovery_stable_steps >= (
                        arguments.outer_recovery_stable_steps
                    ):
                        recovery_damping *= arguments.outer_recovery_decay
                        recovery_stable_steps = 0
                        if recovery_damping < (
                            arguments.outer_recovery_cutoff_ratio
                            * extrinsics_penalty
                        ):
                            recovery_damping = 0.0
            previous_proximal_centers = centers.copy()
            elapsed = time.perf_counter() - started_at
            iteration_row = {
                "iteration": iteration,
                "overallSeconds": elapsed,
                "sumSquaredError": metrics["sumSquaredError"],
                "meanReprojectionError": metrics["meanReprojectionError"],
                "candidateSumSquaredError": candidate_sse,
                "outerRejected": rejected_outer,
                "outerRejections": outer_rejections,
                "accelerationCoefficient": acceleration_coefficient,
                "accelerationTrialSSE": acceleration_trial_sse,
                "accelerationTrialDisagreementSquared": (
                    acceleration_trial_disagreement),
                "accelerationTrialMerit": acceleration_trial_merit,
                "acceptedAccelerationMerit": accepted_acceleration_merit,
                "accelerationFallback": acceleration_fallback,
                "accelerationAccepted": acceleration_accepted,
                "accelerationTrials": acceleration_trials,
                "accelerationFallbacks": acceleration_fallbacks,
                "accelerationResets": acceleration_resets,
                "primalResidualSquared": residuals.primal_squared,
                "dualResidualSquared": residuals.dual_squared,
                "penalty": extrinsics_penalty,
                "extrinsicsPenalty": extrinsics_penalty,
                "intrinsicsPenalty": intrinsics_penalty,
                "workerExtrinsicsPenalty": worker_extrinsics_penalty,
                "workerIntrinsicsPenalty": worker_intrinsics_penalty,
                "recoveryDamping": recovery_damping,
                "recoveryDampingRatio": (
                    recovery_damping / extrinsics_penalty),
                "recoveryStableSteps": recovery_stable_steps,
                "extrinsicsPrimalResidualSquared": (
                    extrinsics_residuals.primal_squared
                    if arguments.outer_method == "admm" else None),
                "extrinsicsDualResidualSquared": (
                    extrinsics_residuals.dual_squared
                    if arguments.outer_method == "admm" else None),
                "intrinsicsPrimalResidualSquared": (
                    intrinsics_residuals.primal_squared
                    if arguments.outer_method == "admm" else None),
                "intrinsicsDualResidualSquared": (
                    intrinsics_residuals.dual_squared
                    if arguments.outer_method == "admm" else None),
                "localObjectiveSum": float(np.sum(costs)),
                "transportBytesSent": worker.sent_bytes,
                "transportBytesReceived": worker.received_bytes,
            }
            trajectory.append(iteration_row)
            if metrics["sumSquaredError"] < best_sse:
                best_sse = metrics["sumSquaredError"]
                best_iteration = iteration
                best_cameras = physical_consensus.copy()
                best_points = landmarks.copy()
            if arguments.debug_output:
                print_iteration_debug(
                    iteration_row, best_sse, best_iteration, costs)
    finally:
        sent_bytes = worker.sent_bytes
        received_bytes = worker.received_bytes
        worker.close()

    final_metrics = evaluate_bal_state(
        best_cameras,
        best_points,
        camera_indices,
        point_indices,
        observations,
    )
    result = {
        "solver": f"cpu-consensus-{arguments.outer_method}",
        "outerMethod": arguments.outer_method,
        "outerAcceleration": arguments.outer_acceleration,
        "accelerationCostRatio": arguments.acceleration_cost_ratio,
        "accelerationDisagreementRatio": (
            arguments.acceleration_disagreement_ratio),
        "accelerationBestCostRatio": arguments.acceleration_best_cost_ratio,
        "accelerationTrials": acceleration_trials,
        "accelerationFallbacks": acceleration_fallbacks,
        "accelerationResets": acceleration_resets,
        "variant": arguments.variant_name,
        "dataset": str(Path(arguments.dataset).resolve()),
        "iterations": arguments.iterations,
        "clusters": cluster_count,
        "localSteps": arguments.local_steps,
        "threadsPerCluster": arguments.threads_per_cluster,
        "cameraScaling": arguments.camera_scaling,
        "cameraScalingMaximumRatio": arguments.camera_scaling_maximum_ratio,
        "cameraScalingClippingPercentile": (
            arguments.camera_scaling_clipping_percentile),
        "cameraScalingSeconds": scaling_seconds,
        "cameraScalingMinimum": float(np.min(camera_scaling)),
        "cameraScalingMaximum": float(np.max(camera_scaling)),
        "localSolver": arguments.local_solver,
        "trustRegionPolicy": arguments.trust_region_policy,
        "alpha": arguments.alpha,
        "penaltyMode": arguments.penalty_mode,
        "maximumPenalty": (
            arguments.maximum_penalty
            if arguments.penalty_mode == "adaptive_capped"
            else None
        ),
        "initialPenalty": initial_penalty,
        "initialPenaltyMultiplier": arguments.initial_penalty_multiplier,
        "outerSafeguardRatio": arguments.outer_safeguard_ratio,
        "outerRecoveryPenaltyRatio": arguments.outer_recovery_penalty_ratio,
        "outerRecoveryMode": arguments.outer_recovery_mode,
        "outerRecoveryStableSteps": arguments.outer_recovery_stable_steps,
        "outerRecoveryDecay": arguments.outer_recovery_decay,
        "outerRecoveryCutoffRatio": arguments.outer_recovery_cutoff_ratio,
        "outerRecoveryTrustRegionRatio": (
            arguments.outer_recovery_trust_region_ratio),
        "outerRejections": outer_rejections,
        "cameraPenaltyGroups": arguments.camera_penalty_groups,
        "penaltyAdaptationGroups": arguments.penalty_adaptation_groups,
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
            "======== ADMM DEBUG FINAL ========\n"
            f"best_sse={best_sse:.6g} best_iteration={best_iteration} "
            f"mean_px={final_metrics['meanReprojectionError']:.6g} "
            f"rmse_px={final_metrics['rmsePerObservation']:.6g}\n"
            f"outer_rejections={outer_rejections} "
            f"acceleration_trials={acceleration_trials} "
            f"fallbacks={acceleration_fallbacks} "
            f"resets={acceleration_resets} "
            f"overall={result['overallSeconds']:.3f}s\n"
            f"result={output_path} state={arguments.state or '-'}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
