"""Plain product-space Douglas-Rachford splitting for standard BAL pixels."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial.transform import Rotation

from bal_evaluator import (
    angle_axis_rotation_matrices,
    canonicalize_bal_problem,
    evaluate_bal_state,
    read_bal_problem,
    save_bal_state,
)
from camera_tangent_diagnostics import (
    consensus_vote_coherence,
    diagonal_weighted_copy_alignment,
    diagonal_weighted_tangent_alignment,
    left_se3_camera_minus,
    left_se3_camera_plus,
    project_tangent_orthogonal_to_basis,
    tangent_alignment,
)
from client_admm import AdmmWorkerClient as DrsWorkerClient
from clustering import (
    cluster_by_daba_louvain,
    cluster_by_landmark_scalable,
    cluster_by_landmark_scalable_stable,
)
from drs_consensus import (
    ActiveCameraMetricBlocks,
    DrsResiduals,
    complete_douglas_rachford_envelope,
    dre_splitting_term,
    drs_step,
    drs_state_for_consensus,
    metric_quadratic_sum,
    proximal_point_step,
    project_consensus,
    recover_local_data_objective,
    reduce_metric_tensor,
    reset_to_consensus,
    shared_floor_prior_blocks,
)
from drs_consensus_metrics import CONSENSUS_METRIC_MODES
from drs_factorized_metrics import FactorizedCameraMetric
from drs_safeguards import (
    bootstrap_basin_guard_decision,
    exceeds_with_relative_deadband,
    increase_recovery_parameter,
    relative_safeguard_ratios,
    should_reject_trial,
)
from outer_acceleration import create_accelerator, interpolate_line_search_center
from partition_cache import PARTITION_CACHE_MODES, partition_with_cache
from admm_scaling import (
    aggregate_camera_metric_blocks,
    block_jacobi_coordinate_maps,
    camera_block_correlation,
    camera_coordinate_scale_values,
    compute_initial_block_jacobi_maps,
    compute_initial_jacobi_scaling,
    compute_initial_ruiz_scaling,
    diagonal_jacobi_scaling_from_blocks,
    restricted_block_jacobi_coordinate_maps,
    to_physical_cameras,
    to_scaled_cameras,
)


WORKER_SSE_RELATIVE_TOLERANCE = 1e-9
WORKER_CONSENSUS_RELATIVE_TOLERANCE = 1e-11


class SchurBSRSymbolicCache:
    def __init__(self):
        self.camera_count = None
        self.pattern_fingerprint = None
        self.source_inverse = None
        self.source_is_unique = None
        self.operator = None
        self.builds = 0
        self.hits = 0


def schur_bsr_pattern_fingerprint(systems, camera_count):
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray([camera_count, len(systems)], dtype=np.int64).tobytes())
    for system in systems:
        for values in (system.block_rows, system.block_columns):
            contiguous = np.ascontiguousarray(values, dtype=np.int64)
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
    return digest.digest()


def transform_cameras_by_similarity(
    cameras, scale, world_rotation, world_translation
):
    cameras = np.asarray(cameras, dtype=np.float64)
    camera_rotations = angle_axis_rotation_matrices(cameras[:, :3])
    centers = -np.einsum(
        "nji,nj->ni", camera_rotations, cameras[:, 3:6]
    )
    transformed_centers = (
        scale * centers @ world_rotation + world_translation
    )
    transformed_rotations = camera_rotations @ world_rotation
    transformed = cameras.copy()
    transformed[:, :3] = Rotation.from_matrix(
        transformed_rotations
    ).as_rotvec()
    transformed[:, 3:6] = -np.einsum(
        "nij,nj->ni", transformed_rotations, transformed_centers
    )
    return transformed


def similarity_gauge_tangent_basis(cameras, epsilon=1e-6):
    cameras = np.asarray(cameras, dtype=np.float64)
    if cameras.ndim != 2 or cameras.shape[1] != 9:
        raise ValueError("gauge basis cameras must have shape (camera_count, 9)")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("gauge basis epsilon must be positive and finite")
    identity = np.eye(3)
    zero = np.zeros(3)
    tangents = []
    for coordinate in range(3):
        translation = np.zeros(3)
        translation[coordinate] = epsilon
        transformed = transform_cameras_by_similarity(
            cameras, 1.0, identity, translation
        )
        tangents.append(
            left_se3_camera_minus(transformed, cameras).ravel() / epsilon
        )
    for coordinate in range(3):
        rotation = np.zeros(3)
        rotation[coordinate] = epsilon
        transformed = transform_cameras_by_similarity(
            cameras,
            1.0,
            Rotation.from_rotvec(rotation).as_matrix(),
            zero,
        )
        tangents.append(
            left_se3_camera_minus(transformed, cameras).ravel() / epsilon
        )
    transformed = transform_cameras_by_similarity(
        cameras, 1.0 + epsilon, identity, zero
    )
    tangents.append(
        left_se3_camera_minus(transformed, cameras).ravel() / epsilon
    )
    basis, triangular = np.linalg.qr(np.column_stack(tangents), mode="reduced")
    retained = np.abs(np.diag(triangular)) > 1e-10
    if not np.any(retained):
        raise RuntimeError("similarity gauge basis has zero rank")
    return basis[:, retained]


def snapshot_metric_blocks(metric_blocks):
    """Snapshot mutable metrics while sharing immutable factorized storage."""
    if metric_blocks is None or isinstance(
        metric_blocks, FactorizedCameraMetric
    ):
        return metric_blocks
    return metric_blocks.copy()


def model_ratio_damping_factor(gain_ratio, minimum_factor=1.0 / 3.0):
    if not 0.0 < minimum_factor < 1.0:
        raise ValueError("model-ratio minimum factor must be in (0, 1)")
    return max(minimum_factor, 1.0 - (2.0 * gain_ratio - 1.0) ** 3)


def rejected_schur_damping(
    camera_damping,
    landmark_damping,
    attempt,
    damping_factor,
    fallback_camera_damping=-1.0,
    fallback_landmark_damping=-1.0,
):
    if (
        attempt == 0
        and fallback_camera_damping > 0.0
        and (
            camera_damping < fallback_camera_damping
            or landmark_damping < fallback_landmark_damping
        )
    ):
        return fallback_camera_damping, fallback_landmark_damping
    return (
        camera_damping * damping_factor,
        landmark_damping * damping_factor,
    )


def assess_schur_trial(
    initial_sse,
    candidate_sse,
    diagnostics,
    damping_policy,
    minimum_gain_ratio,
):
    actual_reduction = 0.5 * (initial_sse - candidate_sse)
    damped_predicted_reduction = diagnostics["dampedPredictedReduction"]
    undamped_predicted_reduction = diagnostics["undampedPredictedReduction"]
    damped_gain_ratio = (
        actual_reduction / damped_predicted_reduction
        if damped_predicted_reduction > 0.0 else float("-inf")
    )
    undamped_gain_ratio = (
        actual_reduction / undamped_predicted_reduction
        if undamped_predicted_reduction > 0.0 else float("-inf")
    )
    accepted = (
        np.isfinite(candidate_sse)
        and candidate_sse < initial_sse
        and diagnostics["linearTermination"] == 0
        and (
            damping_policy != "model_ratio"
            or damped_gain_ratio > minimum_gain_ratio
        )
    )
    return {
        "actualReduction": actual_reduction,
        "dampedGainRatio": damped_gain_ratio,
        "undampedGainRatio": undamped_gain_ratio,
        "accepted": accepted,
    }


def evaluate_global_schur_direction(
    systems, camera_count, camera_damping, tangent_step
):
    tangent_step = np.asarray(tangent_step, dtype=np.float64)
    if tangent_step.shape != (camera_count, 9):
        raise ValueError("Schur diagnostic tangent has an invalid shape")
    gradient = np.zeros((camera_count, 9), dtype=np.float64)
    camera_diagonal = np.zeros((camera_count, 9, 9), dtype=np.float64)
    landmark_model_reduction = 0.0
    undamped_action = np.zeros_like(tangent_step)
    for system in systems:
        np.add.at(gradient, system.camera_ids, system.reduced_gradient)
        np.add.at(camera_diagonal, system.camera_ids, system.camera_diagonal)
        landmark_model_reduction += system.landmark_model_reduction
        action = np.einsum(
            "bij,bj->bi",
            system.blocks,
            tangent_step[system.block_columns],
        )
        np.add.at(undamped_action, system.block_rows, action)
        off_diagonal = system.block_rows != system.block_columns
        if np.any(off_diagonal):
            transpose_action = np.einsum(
                "bji,bj->bi",
                system.blocks[off_diagonal],
                tangent_step[system.block_rows[off_diagonal]],
            )
            np.add.at(
                undamped_action,
                system.block_columns[off_diagonal],
                transpose_action,
            )
    damping_diagonal = np.diagonal(
        camera_diagonal, axis1=1, axis2=2
    )
    positive_diagonal = damping_diagonal[damping_diagonal > 0.0]
    diagonal_floor = (
        np.median(positive_diagonal) * 1e-12
        if positive_diagonal.size else 1e-12
    )
    damping_diagonal = np.maximum(damping_diagonal, diagonal_floor)
    damping_action = camera_damping * damping_diagonal * tangent_step
    gradient_action = float(np.sum(gradient * tangent_step))
    undamped_quadratic = float(np.sum(tangent_step * undamped_action))
    damped_quadratic = undamped_quadratic + float(np.sum(
        tangent_step * damping_action
    ))
    return {
        "gradientAction": gradient_action,
        "undampedQuadratic": undamped_quadratic,
        "dampedQuadratic": damped_quadratic,
        "landmarkModelReduction": float(landmark_model_reduction),
        "undampedPredictedReduction": float(
            landmark_model_reduction
            - gradient_action
            - 0.5 * undamped_quadratic
        ),
        "dampedPredictedReduction": float(
            landmark_model_reduction
            - gradient_action
            - 0.5 * damped_quadratic
        ),
    }


def solve_global_schur_system(
    systems,
    camera_count,
    camera_damping,
    step_scale,
    linear_solver="direct",
    relative_tolerance=1e-6,
    maximum_iterations=500,
    initial_step=None,
    operator_mode="python",
    bsr_symbolic_cache=None,
    preconditioner_mode="jacobi",
    coarse_basis=None,
):
    solve_started_at = time.perf_counter()
    if camera_damping <= 0.0:
        raise ValueError("camera damping must be positive")
    if not 0.0 < step_scale <= 1.0:
        raise ValueError("Schur step scale must be in (0, 1]")
    if linear_solver not in ("direct", "cg"):
        raise ValueError("Schur linear solver must be direct or cg")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("Schur relative tolerance must be in (0, 1)")
    if maximum_iterations <= 0:
        raise ValueError("Schur maximum iterations must be positive")
    if operator_mode not in ("python", "bsr", "bsr_low_memory"):
        raise ValueError(
            "Schur operator must be python, bsr, or bsr_low_memory"
        )
    if preconditioner_mode not in ("jacobi", "gauge_deflated"):
        raise ValueError(
            "Schur preconditioner must be jacobi or gauge_deflated"
        )
    if preconditioner_mode == "gauge_deflated" and coarse_basis is None:
        raise ValueError("gauge-deflated Schur solve requires a coarse basis")
    aggregate_started_at = time.perf_counter()
    gradient = np.zeros((camera_count, 9), dtype=np.float64)
    camera_diagonal = np.zeros((camera_count, 9, 9), dtype=np.float64)
    landmark_model_reduction = 0.0
    block_count = 0
    for system in systems:
        np.add.at(gradient, system.camera_ids, system.reduced_gradient)
        np.add.at(camera_diagonal, system.camera_ids, system.camera_diagonal)
        landmark_model_reduction += system.landmark_model_reduction
        block_count += system.blocks.shape[0]
    aggregate_seconds = time.perf_counter() - aggregate_started_at
    preconditioner_seconds = 0.0
    symbolic_fingerprint_seconds = 0.0
    symbolic_assembly_seconds = 0.0
    numeric_assembly_seconds = 0.0
    linear_solve_seconds = 0.0
    coarse_setup_seconds = 0.0
    coarse_basis_rank = 0
    symbolic_cache_hit = False
    dimension = 9 * camera_count
    initial_vector = None
    if initial_step is not None:
        initial_step = np.asarray(initial_step, dtype=np.float64)
        if initial_step.shape != (camera_count, 9):
            raise ValueError("initial Schur step has an invalid shape")
        if not np.all(np.isfinite(initial_step)):
            raise ValueError("initial Schur step must be finite")
        initial_vector = initial_step.ravel()
    damping_diagonal = np.diagonal(
        camera_diagonal, axis1=1, axis2=2
    )
    positive_diagonal = damping_diagonal[damping_diagonal > 0.0]
    diagonal_floor = (
        np.median(positive_diagonal) * 1e-12
        if positive_diagonal.size else 1e-12
    )
    damping_diagonal = np.maximum(damping_diagonal, diagonal_floor)

    if linear_solver == "direct":
        numeric_assembly_started_at = time.perf_counter()
        scalar_rows = []
        scalar_columns = []
        scalar_values = []
        parameter_offsets = np.arange(9, dtype=np.int64)
        for system in systems:
            block_rows = (
                9 * system.block_rows[:, None, None]
                + parameter_offsets[None, :, None]
            )
            block_columns = (
                9 * system.block_columns[:, None, None]
                + parameter_offsets[None, None, :]
            )
            scalar_rows.append(np.broadcast_to(
                block_rows, system.blocks.shape
            ).ravel())
            scalar_columns.append(np.broadcast_to(
                block_columns, system.blocks.shape
            ).ravel())
            scalar_values.append(system.blocks.ravel())
            off_diagonal = system.block_rows != system.block_columns
            if np.any(off_diagonal):
                scalar_rows.append(np.broadcast_to(
                    block_columns[off_diagonal],
                    system.blocks[off_diagonal].shape,
                ).ravel())
                scalar_columns.append(np.broadcast_to(
                    block_rows[off_diagonal],
                    system.blocks[off_diagonal].shape,
                ).ravel())
                scalar_values.append(
                    np.swapaxes(
                        system.blocks[off_diagonal], 1, 2
                    ).ravel()
                )
        schur = sparse.coo_matrix(
            (
                np.concatenate(scalar_values),
                (
                    np.concatenate(scalar_rows),
                    np.concatenate(scalar_columns),
                ),
            ),
            shape=(dimension, dimension),
        ).tocsr()
        schur = 0.5 * (schur + schur.T)
        damped_schur = schur + camera_damping * sparse.diags(
            damping_diagonal.ravel()
        )
        numeric_assembly_seconds = (
            time.perf_counter() - numeric_assembly_started_at
        )
        linear_solve_started_at = time.perf_counter()
        tangent_step = -sparse_linalg.spsolve(
            damped_schur.tocsc(), gradient.ravel()
        )
        linear_solve_seconds = time.perf_counter() - linear_solve_started_at
        residual = damped_schur @ tangent_step + gradient.ravel()
        def damped_matrix_vector_product(flat_vector):
            return damped_schur @ flat_vector
        iterations = 1
        termination = 0
        scalar_nonzeros = int(schur.nnz)
    else:
        preconditioner_started_at = time.perf_counter()
        preconditioner_blocks = np.zeros(
            (camera_count, 9, 9), dtype=np.float64
        )
        scalar_nonzeros = 0
        for system in systems:
            diagonal = system.block_rows == system.block_columns
            np.add.at(
                preconditioner_blocks,
                system.block_rows[diagonal],
                system.blocks[diagonal],
            )
            scalar_nonzeros += 81 * int(
                np.sum(diagonal) + 2 * np.sum(~diagonal)
            )
        diagonal_indices = np.arange(9)
        preconditioner_blocks[
            :, diagonal_indices, diagonal_indices
        ] += camera_damping * damping_diagonal
        preconditioner_blocks = 0.5 * (
            preconditioner_blocks
            + np.swapaxes(preconditioner_blocks, 1, 2)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(preconditioner_blocks)
        eigenvalue_floor = max(
            float(np.median(eigenvalues[eigenvalues > 0.0])) * 1e-12,
            np.finfo(np.float64).tiny,
        )
        inverse_blocks = np.einsum(
            "bij,bj,bkj->bik",
            eigenvectors,
            1.0 / np.maximum(eigenvalues, eigenvalue_floor),
            eigenvectors,
        )
        preconditioner_seconds = time.perf_counter() - preconditioner_started_at

        if operator_mode == "python":
            def matrix_vector_product(flat_vector):
                vector = flat_vector.reshape((camera_count, 9))
                output = camera_damping * damping_diagonal * vector
                for system in systems:
                    action = np.einsum(
                        "bij,bj->bi",
                        system.blocks,
                        vector[system.block_columns],
                    )
                    np.add.at(output, system.block_rows, action)
                    off_diagonal = system.block_rows != system.block_columns
                    if np.any(off_diagonal):
                        transpose_action = np.einsum(
                            "bji,bj->bi",
                            system.blocks[off_diagonal],
                            vector[system.block_rows[off_diagonal]],
                        )
                        np.add.at(
                            output,
                            system.block_columns[off_diagonal],
                            transpose_action,
                        )
                return output.ravel()
        else:
            block_rows = []
            block_columns = []
            block_sources = []
            for system in systems:
                block_rows.append(system.block_rows)
                block_columns.append(system.block_columns)
                block_sources.append(system.blocks)
                off_diagonal = system.block_rows != system.block_columns
                if np.any(off_diagonal):
                    block_rows.append(system.block_columns[off_diagonal])
                    block_columns.append(system.block_rows[off_diagonal])
                    block_sources.append(np.swapaxes(
                        system.blocks[off_diagonal], 1, 2
                    ))
            block_rows.append(np.arange(camera_count, dtype=np.int64))
            block_columns.append(np.arange(camera_count, dtype=np.int64))
            damping_blocks = np.zeros(
                (camera_count, 9, 9), dtype=np.float64
            )
            damping_blocks[:, np.arange(9), np.arange(9)] = (
                camera_damping * damping_diagonal
            )
            block_sources.append(damping_blocks)
            if operator_mode == "bsr":
                numeric_assembly_started_at = time.perf_counter()
                block_rows = np.concatenate(block_rows)
                block_columns = np.concatenate(block_columns)
                block_keys = block_rows * camera_count + block_columns
                block_values = np.concatenate(block_sources)
                order = np.argsort(block_keys, kind="stable")
                block_keys = block_keys[order]
                block_values = block_values[order]
                unique = np.concatenate((
                    np.array([True]), block_keys[1:] != block_keys[:-1]
                ))
                starts = np.flatnonzero(unique)
                block_values = np.add.reduceat(
                    block_values, starts, axis=0
                )
                block_keys = block_keys[starts]
                block_rows = block_keys // camera_count
                block_columns = block_keys % camera_count
                row_counts = np.bincount(
                    block_rows, minlength=camera_count
                )
                indptr = np.empty(camera_count + 1, dtype=np.int64)
                indptr[0] = 0
                np.cumsum(row_counts, out=indptr[1:])
                bsr_operator = sparse.bsr_matrix(
                    (block_values, block_columns, indptr),
                    shape=(dimension, dimension),
                )
                numeric_assembly_seconds = (
                    time.perf_counter() - numeric_assembly_started_at
                )
            else:
                fingerprint_started_at = time.perf_counter()
                pattern_fingerprint = schur_bsr_pattern_fingerprint(
                    systems, camera_count
                )
                symbolic_fingerprint_seconds = (
                    time.perf_counter() - fingerprint_started_at
                )
                cache = (
                    bsr_symbolic_cache
                    if bsr_symbolic_cache is not None
                    else SchurBSRSymbolicCache()
                )
                symbolic_cache_hit = (
                    cache.camera_count == camera_count
                    and cache.pattern_fingerprint == pattern_fingerprint
                )
                if symbolic_cache_hit:
                    cache.hits += 1
                else:
                    symbolic_assembly_started_at = time.perf_counter()
                    concatenated_rows = np.concatenate(block_rows)
                    concatenated_columns = np.concatenate(block_columns)
                    block_keys = (
                        concatenated_rows * camera_count
                        + concatenated_columns
                    )
                    block_keys, inverse = np.unique(
                        block_keys, return_inverse=True
                    )
                    source_inverse = []
                    source_is_unique = []
                    source_offset = 0
                    for source in block_sources:
                        source_end = source_offset + source.shape[0]
                        source_mapping = inverse[
                            source_offset:source_end
                        ].copy()
                        source_inverse.append(source_mapping)
                        source_is_unique.append(
                            np.unique(source_mapping).size
                            == source_mapping.size
                        )
                        source_offset = source_end
                    unique_rows = block_keys // camera_count
                    indices = block_keys % camera_count
                    row_counts = np.bincount(
                        unique_rows, minlength=camera_count
                    )
                    indptr = np.empty(camera_count + 1, dtype=np.int64)
                    indptr[0] = 0
                    np.cumsum(row_counts, out=indptr[1:])
                    block_values = np.zeros(
                        (block_keys.size, 9, 9), dtype=np.float64
                    )
                    cache.camera_count = camera_count
                    cache.pattern_fingerprint = pattern_fingerprint
                    cache.source_inverse = tuple(source_inverse)
                    cache.source_is_unique = tuple(source_is_unique)
                    cache.operator = sparse.bsr_matrix(
                        (block_values, indices, indptr),
                        shape=(dimension, dimension),
                        copy=False,
                    )
                    cache.builds += 1
                    symbolic_assembly_seconds = (
                        time.perf_counter() - symbolic_assembly_started_at
                    )
                numeric_assembly_started_at = time.perf_counter()
                bsr_operator = cache.operator
                bsr_operator.data.fill(0.0)
                for source, inverse, source_is_unique in zip(
                    block_sources,
                    cache.source_inverse,
                    cache.source_is_unique,
                ):
                    if source_is_unique:
                        bsr_operator.data[inverse] += source
                    else:
                        np.add.at(bsr_operator.data, inverse, source)
                numeric_assembly_seconds = (
                    time.perf_counter() - numeric_assembly_started_at
                )
            scalar_nonzeros = int(bsr_operator.nnz)

            def matrix_vector_product(flat_vector):
                return bsr_operator @ flat_vector

        operator = sparse_linalg.LinearOperator(
            (dimension, dimension), matvec=matrix_vector_product
        )

        def apply_block_jacobi(flat_vector):
            vector = flat_vector.reshape((camera_count, 9))
            return np.einsum(
                "bij,bj->bi",
                inverse_blocks,
                vector,
            ).ravel()

        if preconditioner_mode == "gauge_deflated":
            coarse_setup_started_at = time.perf_counter()
            coarse_basis = np.asarray(coarse_basis, dtype=np.float64)
            if (
                coarse_basis.ndim != 2
                or coarse_basis.shape[0] != dimension
                or not np.all(np.isfinite(coarse_basis))
            ):
                raise ValueError("Schur coarse basis has an invalid shape")
            coarse_basis, triangular = np.linalg.qr(
                coarse_basis, mode="reduced"
            )
            retained = np.abs(np.diag(triangular)) > 1e-10
            coarse_basis = coarse_basis[:, retained]
            if coarse_basis.shape[1] == 0:
                raise ValueError("Schur coarse basis has zero rank")
            coarse_action = np.column_stack([
                matrix_vector_product(coarse_basis[:, column])
                for column in range(coarse_basis.shape[1])
            ])
            coarse_matrix = 0.5 * (
                coarse_basis.T @ coarse_action
                + coarse_action.T @ coarse_basis
            )
            eigenvalues, eigenvectors = np.linalg.eigh(coarse_matrix)
            positive = eigenvalues[eigenvalues > 0.0]
            coarse_floor = max(
                (
                    float(np.median(positive)) * 1e-12
                    if positive.size else 1e-12
                ),
                np.finfo(np.float64).tiny,
            )
            coarse_inverse = np.einsum(
                "ij,j,kj->ik",
                eigenvectors,
                1.0 / np.maximum(eigenvalues, coarse_floor),
                eigenvectors,
            )
            coarse_basis_rank = coarse_basis.shape[1]
            coarse_setup_seconds = (
                time.perf_counter() - coarse_setup_started_at
            )

            def apply_preconditioner(flat_vector):
                coarse_coefficients = coarse_inverse @ (
                    coarse_basis.T @ flat_vector
                )
                residual = flat_vector - coarse_action @ coarse_coefficients
                fine = apply_block_jacobi(residual)
                fine_coarse_coefficients = coarse_inverse @ (
                    coarse_action.T @ fine
                )
                return (
                    fine
                    - coarse_basis @ fine_coarse_coefficients
                    + coarse_basis @ coarse_coefficients
                )
        else:
            apply_preconditioner = apply_block_jacobi

        preconditioner = sparse_linalg.LinearOperator(
            (dimension, dimension),
            matvec=apply_preconditioner,
        )
        iterations = 0

        def count_iteration(_):
            nonlocal iterations
            iterations += 1

        linear_solve_started_at = time.perf_counter()
        tangent_step, termination = sparse_linalg.cg(
            operator,
            -gradient.ravel(),
            x0=initial_vector,
            M=preconditioner,
            rtol=relative_tolerance,
            atol=0.0,
            maxiter=maximum_iterations,
            callback=count_iteration,
        )
        linear_solve_seconds = time.perf_counter() - linear_solve_started_at
        residual = operator @ tangent_step + gradient.ravel()
    if not np.all(np.isfinite(tangent_step)):
        raise RuntimeError("global Schur solve produced a non-finite step")
    scaled_step = step_scale * tangent_step
    damped_action = (
        damped_matrix_vector_product(scaled_step)
        if linear_solver == "direct"
        else matrix_vector_product(scaled_step)
    )
    damping_action = (
        camera_damping * damping_diagonal.ravel() * scaled_step
    )
    gradient_action = float(np.dot(gradient.ravel(), scaled_step))
    damped_predicted_reduction = float(
        landmark_model_reduction
        - gradient_action
        - 0.5 * np.dot(scaled_step, damped_action)
    )
    undamped_predicted_reduction = float(
        landmark_model_reduction
        - gradient_action
        - 0.5 * np.dot(scaled_step, damped_action - damping_action)
    )
    return scaled_step.reshape((-1, 9)), {
        "blockCount": block_count,
        "nonzeros": scalar_nonzeros,
        "gradientNorm": float(np.linalg.norm(gradient)),
        "stepNorm": float(np.linalg.norm(tangent_step)),
        "linearSolver": linear_solver,
        "preconditioner": preconditioner_mode,
        "operator": operator_mode,
        "linearIterations": iterations,
        "linearTermination": int(termination),
        "relativeResidual": float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(gradient), np.finfo(np.float64).tiny)
        ),
        "landmarkModelReduction": float(landmark_model_reduction),
        "dampedPredictedReduction": damped_predicted_reduction,
        "undampedPredictedReduction": undamped_predicted_reduction,
        "aggregateSeconds": aggregate_seconds,
        "preconditionerSeconds": preconditioner_seconds,
        "symbolicFingerprintSeconds": symbolic_fingerprint_seconds,
        "symbolicAssemblySeconds": symbolic_assembly_seconds,
        "numericAssemblySeconds": numeric_assembly_seconds,
        "linearSolveSeconds": linear_solve_seconds,
        "coarseSetupSeconds": coarse_setup_seconds,
        "coarseBasisRank": coarse_basis_rank,
        "totalLinearSystemSeconds": time.perf_counter() - solve_started_at,
        "symbolicCacheHit": symbolic_cache_hit,
    }


def camera_metric_multipliers(
    camera_copy_count, shared_camera_metric_beta, unique_camera_metric_scale
):
    copy_count = np.asarray(camera_copy_count)
    return np.where(
        copy_count == 1,
        unique_camera_metric_scale,
        1.0 + shared_camera_metric_beta * (copy_count - 1.0),
    )


def select_global_schur_majorizer(observability_fractions, threshold):
    fractions = np.asarray(observability_fractions, dtype=np.float64)
    if fractions.ndim != 2 or fractions.shape[1] != 3:
        raise ValueError("Schur observability fractions must have shape (K, 3)")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Schur observability threshold must be in [0, 1]")
    valid = np.all(np.isfinite(fractions), axis=1)
    if not np.any(valid):
        return False, float("nan"), 0
    cluster_means = np.mean(fractions[valid], axis=1)
    statistic = float(np.median(cluster_means))
    return statistic > threshold, statistic, int(np.sum(valid))


def schur_system_observability_fractions(schur_systems):
    fractions = np.full((len(schur_systems), 3), np.nan, dtype=np.float64)
    for cluster, system in enumerate(schur_systems):
        diagonal_mask = system.block_rows == system.block_columns
        diagonal_rows = system.block_rows[diagonal_mask]
        diagonal_blocks = system.blocks[diagonal_mask]
        schur_by_camera = {
            int(camera): block
            for camera, block in zip(diagonal_rows, diagonal_blocks)
        }
        schur_blocks = np.asarray([
            schur_by_camera[int(camera)] for camera in system.camera_ids
        ])
        for group in range(3):
            parameters = np.arange(3 * group, 3 * group + 3)
            raw_trace = float(np.sum(
                system.camera_diagonal[:, parameters, parameters]
            ))
            schur_trace = float(np.sum(
                schur_blocks[:, parameters, parameters]
            ))
            if raw_trace > 0.0 and np.isfinite(raw_trace + schur_trace):
                fractions[cluster, group] = np.clip(
                    (raw_trace - schur_trace) / raw_trace, 0.0, 1.0
                )
    return fractions


def damp_shared_camera_disagreement(
    local_cameras, centers, camera_masks, scale
):
    """Preserve mean shared-camera motion while damping copy disagreement."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError("shared camera disagreement scale must be in [0, 1]")
    active = np.asarray(camera_masks, dtype=bool)
    copy_count = np.sum(active, axis=0)
    shared = copy_count > 1
    if not np.any(shared) or scale == 1.0:
        return
    displacement = local_cameras[:, shared] - centers[:, shared]
    shared_active = active[:, shared, None]
    mean_displacement = np.sum(
        np.where(shared_active, displacement, 0.0), axis=0
    ) / copy_count[shared, None]
    corrected = mean_displacement[None, :, :] + scale * (
        displacement - mean_displacement[None, :, :]
    )
    local_cameras[:, shared] = np.where(
        shared_active,
        centers[:, shared] + corrected,
        local_cameras[:, shared],
    )


def damp_metric_projected_camera_proposals(
    local_cameras,
    centers,
    camera_masks,
    previous_consensus,
    metric_blocks,
    metric_mode,
    scale,
    disagreement_threshold=None,
    prior_blocks=None,
):
    """Damp shared proposal residuals while preserving their metric projection."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError("metric proposal disagreement scale must be in [0, 1]")
    active = np.asarray(camera_masks, dtype=bool)
    shared = np.sum(active, axis=0) > 1
    if not np.any(shared) or scale == 1.0:
        return float("nan"), 1.0
    disagreement_ratio, projected = shared_camera_compatibility_ratio(
        local_cameras,
        centers,
        active,
        previous_consensus,
        metric_blocks,
        metric_mode,
        prior_blocks=prior_blocks,
    )
    if not np.isfinite(disagreement_ratio):
        return disagreement_ratio, 1.0
    shared = np.sum(active, axis=0) > 1
    applied_scale = (
        scale
        if disagreement_threshold is None
        or disagreement_ratio >= disagreement_threshold
        else 1.0
    )
    shared_active = active[:, shared, None]
    corrected = projected[None, shared] + applied_scale * (
        local_cameras[:, shared] - projected[None, shared]
    )
    local_cameras[:, shared] = np.where(
        shared_active, corrected, local_cameras[:, shared]
    )
    return disagreement_ratio, applied_scale


def scheduled_metric_proposal_scale(scale, until, iteration):
    """Return the fixed proposal scale for this outer iteration."""
    return scale if until == 0 or iteration < until else 1.0


def scheduled_outer_acceleration_active(until, iteration):
    """Return whether outer acceleration is active this iteration."""
    return until == 0 or iteration < until


def collective_trust_trial_radius(radii, recovery_ratio):
    """Return one shrunken shared radius from completed local trials."""
    radii = np.asarray(radii, dtype=np.float64)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("collective trust radii must be a nonempty vector")
    if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("collective trust radii must be finite and positive")
    if not 0.0 < recovery_ratio < 1.0:
        raise ValueError("collective trust recovery ratio must be in (0, 1)")
    return float(recovery_ratio * np.exp(np.mean(np.log(radii))))


def prefer_collective_trust_trial(
    nominal_rejected, nominal_sse, trial_rejected, trial_sse
):
    """Prefer only a safely accepted, materially lower physical SSE trial."""
    if trial_rejected or not np.isfinite(trial_sse):
        return False
    if nominal_rejected or not np.isfinite(nominal_sse):
        return True
    tolerance = 1e-12 * max(abs(nominal_sse), 1.0)
    return trial_sse < nominal_sse - tolerance


def outer_acceleration_restart_is_active(restart_iteration, iteration):
    """Return whether outer acceleration state resets this iteration."""
    return restart_iteration > 0 and iteration == restart_iteration


def local_state_rebase_is_active(rebase_iteration, iteration):
    """Return whether local trust/curvature state rebases this iteration."""
    return rebase_iteration > 0 and iteration == rebase_iteration


def shared_camera_compatibility_ratio(
    local_cameras,
    centers,
    camera_masks,
    previous_consensus,
    metric_blocks,
    metric_mode,
    prior_blocks=None,
):
    """Return shared-copy disagreement energy relative to proposal motion."""
    active = np.asarray(camera_masks, dtype=bool)
    shared = np.sum(active, axis=0) > 1
    if not np.any(shared):
        return float("nan"), np.asarray(previous_consensus).copy()
    selected_metrics = reduce_metric_tensor(
        metric_blocks, active, metric_mode, local_cameras.shape[2]
    )
    projected = project_consensus(
        local_cameras,
        active,
        previous_consensus,
        selected_metrics,
        prior_blocks=prior_blocks,
        prior_center=previous_consensus,
        direct_singletons=local_cameras,
    )
    projected[~shared] = previous_consensus[~shared]
    shared_active = active & shared[None, :]
    proposal_residual = np.where(
        shared_active[:, :, None],
        local_cameras - projected[None, :, :],
        0.0,
    )
    proposal_displacement = np.where(
        shared_active[:, :, None], local_cameras - centers, 0.0
    )
    residual_squared = metric_quadratic_sum(
        proposal_residual, selected_metrics, active
    )
    displacement_squared = metric_quadratic_sum(
        proposal_displacement, selected_metrics, active
    )
    disagreement_ratio = residual_squared / max(
        displacement_squared, np.finfo(np.float64).tiny
    )
    return disagreement_ratio, projected


def prefer_metric_selector_trial(
    nominal_rejected,
    nominal_dre,
    nominal_sse,
    trial_rejected,
    trial_dre,
    trial_sse,
):
    """Prefer an admissible reduced-metric trial with lower DRE then SSE."""
    return not trial_rejected and (
        nominal_rejected
        or (trial_dre, trial_sse) < (nominal_dre, nominal_sse)
    )


def camera_copy_disagreement_diagnostics(
    local_cameras,
    camera_masks,
    previous_consensus,
    metric_blocks,
    metric_mode,
    camera_scaling,
    camera_ids,
):
    """Summarize selected camera copies before proposal damping."""
    active = np.asarray(camera_masks, dtype=bool)
    selected_metrics = reduce_metric_tensor(
        metric_blocks, active, metric_mode, local_cameras.shape[2]
    )
    projected = project_consensus(
        local_cameras, active, previous_consensus, selected_metrics
    )
    diagnostics = []
    for camera_id in camera_ids:
        if camera_id < 0 or camera_id >= local_cameras.shape[1]:
            continue
        clusters = np.flatnonzero(active[:, camera_id])
        if clusters.size <= 1:
            continue
        scaled_copies = local_cameras[clusters, camera_id]
        if np.asarray(camera_scaling).ndim == 3:
            camera_coordinate_map = camera_scaling[camera_id:camera_id + 1]
            physical_copies = to_physical_cameras(
                scaled_copies[:, None, :], camera_coordinate_map
            )[:, 0, :]
            physical_projection = to_physical_cameras(
                projected[camera_id][None, :], camera_coordinate_map
            )[0]
        else:
            physical_copies = to_physical_cameras(
                scaled_copies, camera_scaling[camera_id]
            )
            physical_projection = to_physical_cameras(
                projected[camera_id], camera_scaling[camera_id]
            )
        physical_residual = physical_copies - physical_projection
        if isinstance(selected_metrics, ActiveCameraMetricBlocks):
            selected = selected_metrics.camera_indices == camera_id
            blocks = selected_metrics.blocks[selected]
            metric_clusters = selected_metrics.cluster_indices[selected]
            order = np.argsort(metric_clusters)
            blocks = blocks[order]
            metric_clusters = metric_clusters[order]
            if not np.array_equal(metric_clusters, clusters):
                raise RuntimeError("camera diagnostic metric copies are misordered")
        else:
            blocks = selected_metrics[clusters, camera_id]
        scaled_residual = scaled_copies - projected[camera_id]
        copy_energies = np.einsum(
            "bi,bij,bj->b", scaled_residual, blocks, scaled_residual
        )
        if np.asarray(camera_scaling).ndim == 3:
            inverse_map = np.linalg.inv(camera_scaling[camera_id])
            physical_metrics = np.einsum(
                "ji,bjk,kl->bil", inverse_map, blocks, inverse_map
            )
        else:
            physical_metrics = (
                camera_scaling[camera_id][None, :, None]
                * blocks
                * camera_scaling[camera_id][None, None, :]
            )
        metric_sum = np.sum(blocks, axis=0)
        metric_sum_eigenvalues = np.linalg.eigvalsh(
            0.5 * (metric_sum + metric_sum.T)
        )
        positive_eigenvalues = metric_sum_eigenvalues[
            metric_sum_eigenvalues > 0.0
        ]
        metric_sum_condition = (
            float(np.max(positive_eigenvalues) / np.min(positive_eigenvalues))
            if positive_eigenvalues.size == metric_sum_eigenvalues.size
            else float("inf")
        )
        group_ranges = []
        group_rms = []
        group_metric_diagonal = []
        for start in (0, 3, 6):
            group = physical_copies[:, start:start + 3]
            residual_group = physical_residual[:, start:start + 3]
            group_ranges.append(float(np.max(np.ptp(group, axis=0))))
            group_rms.append(float(np.sqrt(np.mean(residual_group**2))))
            diagonals = np.diagonal(
                physical_metrics[:, start:start + 3, start:start + 3],
                axis1=1,
                axis2=2,
            )
            group_metric_diagonal.append([
                float(np.min(diagonals)), float(np.max(diagonals))
            ])
        diagnostics.append({
            "camera": int(camera_id),
            "clusters": clusters.tolist(),
            "copies": int(clusters.size),
            "translationRange": group_ranges[0],
            "rotationRange": group_ranges[1],
            "intrinsicsRange": group_ranges[2],
            "translationRms": group_rms[0],
            "rotationRms": group_rms[1],
            "intrinsicsRms": group_rms[2],
            "copyMetricEnergies": copy_energies.tolist(),
            "metricEnergy": float(np.sum(copy_energies)),
            "metricSumEigenvalueMinimum": float(
                np.min(metric_sum_eigenvalues)
            ),
            "metricSumEigenvalueMaximum": float(
                np.max(metric_sum_eigenvalues)
            ),
            "metricSumCondition": metric_sum_condition,
            "physicalMetricDiagonalRanges": group_metric_diagonal,
        })
    return diagnostics


def select_metric_proposal_hysteresis_scale(
    disagreement_ratio,
    current_scale,
    strong_scale,
    normal_scale,
    low_threshold,
    high_threshold,
):
    """Select proposal damping with two-threshold hysteresis."""
    if current_scale == normal_scale and disagreement_ratio >= high_threshold:
        return strong_scale
    if current_scale == strong_scale and disagreement_ratio <= low_threshold:
        return normal_scale
    return current_scale


def damp_metric_projected_camera_proposals_hysteresis(
    local_cameras,
    centers,
    camera_masks,
    previous_consensus,
    metric_blocks,
    metric_mode,
    current_scale,
    strong_scale,
    normal_scale,
    low_threshold,
    high_threshold,
):
    """Apply proposal damping selected from the pre-damping disagreement."""
    original = local_cameras.copy()
    disagreement_ratio, _ = damp_metric_projected_camera_proposals(
        local_cameras,
        centers,
        camera_masks,
        previous_consensus,
        metric_blocks,
        metric_mode,
        scale=current_scale,
    )
    selected_scale = select_metric_proposal_hysteresis_scale(
        disagreement_ratio,
        current_scale,
        strong_scale,
        normal_scale,
        low_threshold,
        high_threshold,
    )
    if selected_scale != current_scale:
        local_cameras[:] = original
        disagreement_ratio, _ = damp_metric_projected_camera_proposals(
            local_cameras,
            centers,
            camera_masks,
            previous_consensus,
            metric_blocks,
            metric_mode,
            scale=selected_scale,
        )
    return disagreement_ratio, selected_scale


def damp_metric_projected_camera_subspaces(
    local_cameras,
    camera_masks,
    previous_consensus,
    metric_blocks,
    metric_mode,
    scales,
):
    """Damp coordinate groups and recenter to preserve metric projection."""
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (3,) or np.any(scales < 0.0) or np.any(scales > 1.0):
        raise ValueError("metric proposal subspace scales must be three values in [0, 1]")
    active = np.asarray(camera_masks, dtype=bool)
    selected_metrics = reduce_metric_tensor(
        metric_blocks, active, metric_mode, local_cameras.shape[2]
    )
    projected = project_consensus(
        local_cameras, active, previous_consensus, selected_metrics
    )
    residual = np.where(
        active[:, :, None], local_cameras - projected[None, :, :], 0.0
    )
    group_energies = np.empty(3, dtype=np.float64)
    for group, start in enumerate((0, 3, 6)):
        group_residual = residual[..., start:start + 3]
        if hasattr(selected_metrics, "blocks"):
            blocks = selected_metrics.blocks[:, start:start + 3, start:start + 3]
            vectors = group_residual[
                selected_metrics.cluster_indices,
                selected_metrics.camera_indices,
            ]
        else:
            blocks = selected_metrics[..., start:start + 3, start:start + 3][active]
            vectors = group_residual[active]
        group_energies[group] = np.einsum(
            "bi,bij,bj->", vectors, blocks, vectors
        )
    energy_sum = max(float(np.sum(group_energies)), np.finfo(np.float64).tiny)
    energy_fractions = group_energies / energy_sum

    parameter_scales = np.repeat(scales, 3)
    corrected = projected[None, :, :] + residual * parameter_scales
    corrected_projection = project_consensus(
        corrected, active, previous_consensus, selected_metrics
    )
    corrected += projected[None, :, :] - corrected_projection[None, :, :]
    local_cameras[:] = np.where(active[:, :, None], corrected, local_cameras)
    return energy_fractions


def select_metric_proposal_scale(
    local_cameras,
    centers,
    camera_masks,
    previous_consensus,
    metric_blocks,
    metric_mode,
    scales,
    undamped_local_objective,
    evaluate_local_objective,
):
    """Select a projection-preserving proposal scale by corrected model DRE."""
    selected_metrics = reduce_metric_tensor(
        metric_blocks, camera_masks, metric_mode, local_cameras.shape[2]
    )
    projected_reflection = project_consensus(
        2.0 * local_cameras - centers,
        camera_masks,
        previous_consensus,
        selected_metrics,
    )
    best = None
    for scale in scales:
        candidate = local_cameras.copy()
        disagreement_ratio = float("nan")
        if scale != 1.0:
            disagreement_ratio, _ = damp_metric_projected_camera_proposals(
                candidate,
                centers,
                camera_masks,
                previous_consensus,
                metric_blocks,
                metric_mode,
                scale,
            )
            local_objective = evaluate_local_objective(candidate)
        else:
            local_objective = undamped_local_objective
        splitting_term = dre_splitting_term(
            candidate,
            projected_reflection,
            centers,
            camera_masks,
            1.0,
            metric_blocks=selected_metrics,
        )
        score = local_objective + splitting_term
        if best is None or score < best[0]:
            best = (
                score,
                candidate,
                scale,
                disagreement_ratio,
                local_objective,
            )
    return best[1:]


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
    parser.add_argument("--single-cluster-proximal", action="store_true")
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--local-camera-step-scale", type=float, default=1.0)
    parser.add_argument(
        "--local-camera-step-grid",
        choices=("1", "0.25,0.5,1"),
        default="1",
    )
    parser.add_argument(
        "--shared-camera-step-grid",
        choices=("1", "0.25,0.5,1"),
        default="1",
    )
    parser.add_argument("--shared-camera-step-scale", type=float, default=1.0)
    parser.add_argument(
        "--shared-camera-disagreement-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--metric-proposal-disagreement-scale", type=float, default=1.0
    )
    parser.add_argument("--metric-proposal-disagreement-until", type=int, default=0)
    parser.add_argument(
        "--metric-proposal-disagreement-grid",
        default="1",
    )
    parser.add_argument(
        "--metric-proposal-subspace-scales",
        choices=(
            "1,1,1",
            "0.6,1,1",
            "1,0.6,1",
            "1,1,0.6",
            "0.6,0.6,1",
        ),
        default="1,1,1",
    )
    parser.add_argument(
        "--metric-proposal-disagreement-threshold", type=float, default=-1.0
    )
    parser.add_argument(
        "--metric-proposal-disagreement-hysteresis",
        default="",
        help="strong,normal,low,high proposal damping hysteresis",
    )
    parser.add_argument(
        "--camera-diagonal-quantile-iterations",
        default="",
        help="comma-separated one-based nominal outer iterations",
    )
    parser.add_argument(
        "--camera-disagreement-diagnostic-ids",
        default="",
        help="comma-separated global camera IDs for copy diagnostics",
    )
    parser.add_argument(
        "--camera-disagreement-diagnostic-iterations",
        default="",
        help="comma-separated one-based nominal outer iterations",
    )
    parser.add_argument(
        "--schur-alignment-diagnostic-iterations",
        default="",
        help="comma-separated one-based nominal outer iterations",
    )
    parser.add_argument(
        "--schur-alignment-camera-damping", type=float, default=3.0
    )
    parser.add_argument(
        "--schur-alignment-landmark-damping", type=float, default=3.0
    )
    parser.add_argument(
        "--schur-alignment-maximum-iterations", type=int, default=500
    )
    parser.add_argument(
        "--schur-model-consensus-clipping", action="store_true"
    )
    parser.add_argument(
        "--schur-model-consensus-clipping-minimum-scale",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--schur-model-consensus-clipping-maximum-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--shared-camera-metric-beta", type=float, default=0.0)
    parser.add_argument("--adaptive-local-depth", action="store_true")
    parser.add_argument("--interior-defect-diagnostic", action="store_true")
    parser.add_argument("--adaptive-local-depth-start", type=int, default=0)
    parser.add_argument("--adaptive-local-depth-maximum", type=int, default=2)
    parser.add_argument("--adaptive-local-depth-high", type=float, default=0.3)
    parser.add_argument("--adaptive-local-depth-low", type=float, default=0.15)
    parser.add_argument("--adaptive-local-depth-window", type=int, default=3)
    parser.add_argument("--adaptive-local-depth-dwell", type=int, default=3)
    parser.add_argument("--initial-shared-schur-correction", action="store_true")
    parser.add_argument("--initial-shared-schur-basin-guard", action="store_true")
    parser.add_argument(
        "--initial-shared-schur-rebase-trust-state", action="store_true"
    )
    parser.add_argument(
        "--initial-shared-schur-operator",
        choices=("inherit", "python", "bsr", "bsr_low_memory"),
        default="inherit",
    )
    parser.add_argument(
        "--initial-shared-schur-maximum-iterations", type=int, default=0
    )
    parser.add_argument(
        "--initial-shared-schur-maximum-corrections", type=int, default=1
    )
    parser.add_argument(
        "--initial-shared-schur-damping-policy",
        choices=("geometric", "model_ratio"),
        default="geometric",
    )
    parser.add_argument(
        "--initial-shared-schur-model-ratio-minimum-factor",
        type=float,
        default=1.0 / 3.0,
    )
    parser.add_argument(
        "--mid-shared-schur-correction-iteration", type=int, default=0
    )
    parser.add_argument(
        "--mid-shared-schur-transport-product-state", action="store_true"
    )
    parser.add_argument("--final-shared-schur-correction", action="store_true")
    parser.add_argument("--stop-after-iteration", type=int, default=0)
    parser.add_argument(
        "--shared-schur-landmark-damping", type=float, default=3.0
    )
    parser.add_argument(
        "--shared-schur-camera-damping", type=float, default=3.0
    )
    parser.add_argument("--shared-schur-step-scale", type=float, default=1.0)
    parser.add_argument(
        "--shared-schur-landmark-refinement-steps", type=int, default=3
    )
    parser.add_argument(
        "--shared-schur-linear-solver", choices=("direct", "cg"), default="cg"
    )
    parser.add_argument(
        "--shared-schur-relative-tolerance", type=float, default=1e-6
    )
    parser.add_argument(
        "--shared-schur-maximum-iterations", type=int, default=500
    )
    parser.add_argument(
        "--shared-schur-maximum-corrections", type=int, default=1
    )
    parser.add_argument(
        "--shared-schur-python-confirmation-corrections", type=int, default=0
    )
    parser.add_argument(
        "--shared-schur-confirmation-camera-damping", type=float, default=-1.0
    )
    parser.add_argument(
        "--shared-schur-confirmation-landmark-damping", type=float, default=-1.0
    )
    parser.add_argument(
        "--shared-schur-confirmation-after-screening-budget",
        action="store_true",
    )
    parser.add_argument(
        "--shared-schur-maximum-attempts", type=int, default=6
    )
    parser.add_argument(
        "--shared-schur-fallback-camera-damping", type=float, default=-1.0
    )
    parser.add_argument(
        "--shared-schur-fallback-landmark-damping", type=float, default=-1.0
    )
    parser.add_argument(
        "--shared-schur-damping-increase", type=float, default=2.0
    )
    parser.add_argument(
        "--shared-schur-damping-decrease", type=float, default=0.5
    )
    parser.add_argument(
        "--shared-schur-damping-policy",
        choices=("geometric", "model_ratio"),
        default="geometric",
    )
    parser.add_argument(
        "--shared-schur-minimum-gain-ratio", type=float, default=1e-4
    )
    parser.add_argument(
        "--shared-schur-minimum-relative-decrease", type=float, default=1e-4
    )
    parser.add_argument("--shared-schur-warm-start", action="store_true")
    parser.add_argument(
        "--shared-schur-preconditioner",
        choices=("jacobi", "gauge_deflated"),
        default="jacobi",
    )
    parser.add_argument(
        "--shared-schur-operator",
        choices=("python", "bsr", "bsr_low_memory"),
        default="python",
    )
    parser.add_argument("--threads-per-cluster", type=int, default=1)
    parser.add_argument("--nesterov-max-iterations", type=int, default=100)
    parser.add_argument("--enhanced-inner-max-iterations", type=int, default=300)
    parser.add_argument("--nesterov-min-iterations", type=int, default=1)
    parser.add_argument("--nesterov-stop-tolerance", type=float, default=1e-2)
    parser.add_argument("--enhanced-inner-until", type=int, default=0)
    parser.add_argument("--diagonal-trust-until", type=int, default=0)
    parser.add_argument("--relative-residual-until", type=int, default=0)
    parser.add_argument(
        "--local-solver",
        choices=(
            "ceres_pcg", "ceres_se3", "ceres_prox_se3", "poba_power",
            "schur_pcg", "nesterov",
        ),
        default="nesterov",
    )
    parser.add_argument(
        "--local-solver-switch-iteration", type=int, default=0
    )
    parser.add_argument(
        "--local-solver-after-switch",
        choices=("nesterov", "schur_pcg"),
    )
    parser.add_argument(
        "--trust-region-policy", choices=("ceres", "drs", "daba"), default="daba"
    )
    parser.add_argument("--persistent-trust-region", action="store_true")
    parser.add_argument("--trust-region-recovery-ratio", type=float, default=0.5)
    parser.add_argument("--shared-trust-region-until", type=int, default=0)
    parser.add_argument("--collective-trust-trial-until", type=int, default=0)
    parser.add_argument("--local-state-rebase-iteration", type=int, default=0)
    parser.add_argument(
        "--shared-trust-region-initial-radius", type=float, default=1e6
    )
    parser.add_argument(
        "--camera-scaling",
        choices=(
            "none",
            "jacobi_initial",
            "ruiz_initial",
            "block_jacobi_initial",
            "worker_block_jacobi_initial",
            "worker_diagonal_jacobi_initial",
            "worker_z_f_block_jacobi_initial",
        ),
        default="jacobi_initial",
    )
    parser.add_argument(
        "--scene-normalization",
        choices=("points_p95", "none"),
        default="points_p95",
    )
    parser.add_argument("--camera-scaling-maximum-ratio", type=float)
    parser.add_argument("--camera-scaling-clipping-percentile", type=float)
    parser.add_argument("--camera-diagonal-relative-floor", type=float, default=1e-48)
    parser.add_argument("--camera-trust-diagonal-scale", type=float, default=1e-4)
    parser.add_argument("--camera-diagonal-metric-scale", type=float, default=25.0)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument(
        "--outer-acceleration",
        choices=("none", "nesterov", "themelis_nesterov", "lbfgs", "anderson"),
        default="none",
    )
    parser.add_argument("--outer-acceleration-until", type=int, default=0)
    parser.add_argument(
        "--outer-acceleration-restart-iteration", type=int, default=0
    )
    parser.add_argument(
        "--line-search-grid", choices=("0,1", "0,0.5,1"), default="0,1"
    )
    parser.add_argument("--acceleration-restart-after", type=int, default=3)
    parser.add_argument("--penalty-multiplier", type=float, default=1.0)
    parser.add_argument("--huber-delta", type=float, default=0.0)
    parser.add_argument(
        "--proximal-metric", choices=("scalar", "block"), default="scalar"
    )
    parser.add_argument(
        "--consensus-metric",
        choices=CONSENSUS_METRIC_MODES,
        default="arithmetic",
    )
    parser.add_argument("--block-regularization", type=float, default=5e-5)
    parser.add_argument("--shared-only-camera-proximal", action="store_true")
    parser.add_argument(
        "--factorized-coupled-schur-proximal-metric", action="store_true"
    )
    parser.add_argument("--unique-camera-metric-scale", type=float, default=1.0)
    parser.add_argument(
        "--unique-camera-metric-selector",
        default="",
        help="reduced_scale,compatibility_threshold",
    )
    parser.add_argument("--block-curvature-multiplier", type=float, default=0.0)
    parser.add_argument(
        "--global-schur-majorizer-observability-threshold",
        type=float,
        default=-1.0,
    )
    parser.add_argument("--global-schur-majorizer-until", type=int, default=0)
    parser.add_argument(
        "--block-recovery-mode",
        choices=("regularization", "curvature", "measured_curvature"),
        default="regularization",
    )
    parser.add_argument(
        "--maximum-block-curvature-multiplier", type=float, default=16.0
    )
    parser.add_argument(
        "--recovery-exhaustion-policy",
        choices=("stop", "restart_best_relaxed"),
        default="stop",
    )
    parser.add_argument("--recovery-relaxed-iterations", type=int, default=1)
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
    parser.add_argument(
        "--consensus-shared-floor-prior-scale", type=float, default=0.0
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
    parser.add_argument("--safeguard-annealing-iterations", type=int, default=0)
    parser.add_argument("--safeguard-reference-iteration", type=int, default=5)
    parser.add_argument("--safeguard-annealing-exponent", type=float, default=4.0)
    parser.add_argument("--safeguard-relative-deadband", type=float, default=0.0)
    parser.add_argument("--catastrophic-ratio", type=float, default=1e6)
    parser.add_argument("--recovery-penalty-ratio", type=float, default=2.0)
    parser.add_argument("--maximum-penalty", type=float, default=1e12)
    parser.add_argument("--results", default="results_drs.jsonl")
    parser.add_argument("--state")
    parser.add_argument(
        "--initial-state",
        help="NPZ cameras/points in the raw dataset BAL coordinate frame",
    )
    parser.add_argument(
        "--initial-state-frame",
        choices=("raw", "canonical"),
        default="raw",
        help="coordinate frame of --initial-state cameras/points",
    )
    parser.add_argument("--variant-name", default="plain_drs")
    parser.add_argument("--residual-balance-slack", type=float, default=0.01)
    parser.add_argument("--minimum-camera-landmarks", type=int, default=20)
    parser.add_argument("--max-refinement-passes", type=int, default=3)
    parser.add_argument(
        "--clustering",
        choices=(
            "landmark_scalable",
            "landmark_scalable_stable",
            "daba_louvain",
        ),
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
    if arguments.iterations < 0 or arguments.clusters <= 0:
        raise ValueError("iterations must be nonnegative and clusters positive")
    if arguments.iterations == 0 and not arguments.final_shared_schur_correction:
        raise ValueError("zero iterations require final shared Schur correction")
    if arguments.local_steps <= 0 or arguments.threads_per_cluster <= 0:
        raise ValueError("local steps and threads must be positive")
    if not 0.0 < arguments.local_camera_step_scale <= 1.0:
        raise ValueError("local camera step scale must be in (0, 1]")
    if arguments.local_camera_step_scale != 1.0:
        if arguments.worker_owned_cameras:
            raise ValueError(
                "scaled local camera steps require coordinator-owned cameras"
            )
        if arguments.outer_acceleration != "none":
            raise ValueError(
                "scaled local camera steps currently forbid outer acceleration"
            )
        if arguments.safeguard_mode != "none":
            raise ValueError(
                "scaled local camera steps currently require safeguard mode none"
            )
    if arguments.local_camera_step_grid != "1":
        if arguments.proximal_metric != "block":
            raise ValueError("local camera step grid requires block metrics")
        if arguments.local_camera_step_scale != 1.0:
            raise ValueError("local camera step scale and grid are mutually exclusive")
        if arguments.worker_owned_cameras or arguments.worker_owned_landmarks:
            raise ValueError(
                "local camera step grid requires coordinator-owned state"
            )
        if arguments.outer_acceleration != "none":
            raise ValueError("local camera step grid forbids outer acceleration")
        if arguments.safeguard_mode != "none":
            raise ValueError("local camera step grid requires safeguard mode none")
        if arguments.consensus_execution != "coordinator":
            raise ValueError("local camera step grid requires coordinator consensus")
        if arguments.consensus_landmark_refinement_steps != 0:
            raise ValueError("local camera step grid forbids consensus refinement")
    if arguments.shared_camera_step_grid != "1":
        if arguments.local_camera_step_grid != "1":
            raise ValueError("local and shared camera step grids are mutually exclusive")
        if arguments.proximal_metric != "block":
            raise ValueError("shared camera step grid requires block metrics")
        if arguments.worker_owned_cameras or arguments.worker_owned_landmarks:
            raise ValueError("shared camera step grid requires coordinator-owned state")
        if arguments.outer_acceleration != "none":
            raise ValueError("shared camera step grid forbids outer acceleration")
        if arguments.safeguard_mode != "none":
            raise ValueError("shared camera step grid requires safeguard mode none")
        if arguments.consensus_execution != "coordinator":
            raise ValueError("shared camera step grid requires coordinator consensus")
        if arguments.consensus_landmark_refinement_steps != 0:
            raise ValueError("shared camera step grid forbids consensus refinement")
    if not 0.0 < arguments.shared_camera_step_scale <= 1.0:
        raise ValueError("shared camera step scale must be in (0, 1]")
    if arguments.shared_camera_step_scale != 1.0:
        if arguments.shared_camera_step_grid != "1":
            raise ValueError("shared camera step scale and grid are mutually exclusive")
        if arguments.local_camera_step_grid != "1" or arguments.local_camera_step_scale != 1.0:
            raise ValueError("shared and global local step scaling are mutually exclusive")
        if arguments.worker_owned_cameras:
            raise ValueError("shared camera step scale requires coordinator-owned cameras")
    if not 0.0 <= arguments.shared_camera_disagreement_scale <= 1.0:
        raise ValueError("shared camera disagreement scale must be in [0, 1]")
    if arguments.shared_camera_disagreement_scale != 1.0:
        if (
            arguments.shared_camera_step_grid != "1"
            or arguments.shared_camera_step_scale != 1.0
            or arguments.local_camera_step_grid != "1"
            or arguments.local_camera_step_scale != 1.0
        ):
            raise ValueError(
                "shared camera disagreement damping and step scaling are mutually exclusive"
            )
        if arguments.worker_owned_cameras:
            raise ValueError(
                "shared camera disagreement damping requires coordinator-owned cameras"
            )
        if arguments.consensus_execution != "coordinator":
            raise ValueError(
                "shared camera disagreement damping requires coordinator consensus"
            )
        if arguments.worker_consensus_shadow:
            raise ValueError(
                "shared camera disagreement damping forbids worker consensus shadow"
            )
    if not 0.0 <= arguments.metric_proposal_disagreement_scale <= 1.0:
        raise ValueError("metric proposal disagreement scale must be in [0, 1]")
    if not 0 <= arguments.metric_proposal_disagreement_until <= arguments.iterations:
        raise ValueError("metric proposal disagreement cutoff must be in [0, iterations]")
    if arguments.metric_proposal_disagreement_until > 0:
        if arguments.metric_proposal_disagreement_scale == 1.0:
            raise ValueError("metric proposal disagreement cutoff requires fixed damping")
        if (
            arguments.metric_proposal_disagreement_grid != "1"
            or arguments.metric_proposal_disagreement_hysteresis
            or arguments.metric_proposal_subspace_scales != "1,1,1"
        ):
            raise ValueError("metric proposal disagreement cutoff supports fixed damping only")
    if arguments.metric_proposal_disagreement_scale != 1.0:
        if arguments.proximal_metric != "block":
            raise ValueError(
                "metric proposal disagreement damping requires block metrics"
            )
        if arguments.consensus_execution != "coordinator":
            raise ValueError(
                "metric proposal disagreement damping requires coordinator consensus"
            )
        if arguments.worker_owned_cameras:
            raise ValueError(
                "metric proposal disagreement damping requires coordinator-owned cameras"
            )
        if arguments.worker_consensus_shadow:
            raise ValueError(
                "metric proposal disagreement damping forbids worker consensus shadow"
            )
        if (
            arguments.shared_camera_disagreement_scale != 1.0
            or arguments.shared_camera_step_grid != "1"
            or arguments.shared_camera_step_scale != 1.0
            or arguments.local_camera_step_grid != "1"
            or arguments.local_camera_step_scale != 1.0
        ):
            raise ValueError(
                "metric proposal disagreement damping and step scaling are mutually exclusive"
            )
    if arguments.metric_proposal_disagreement_grid != "1":
        try:
            proposal_grid = tuple(map(
                float,
                arguments.metric_proposal_disagreement_grid.split(","),
            ))
        except ValueError as error:
            raise ValueError(
                "metric proposal disagreement grid must contain numbers"
            ) from error
        if (
            not proposal_grid
            or any(not 0.0 <= value <= 1.0 for value in proposal_grid)
        ):
            raise ValueError(
                "metric proposal disagreement grid values must be in [0, 1]"
            )
        if arguments.metric_proposal_disagreement_scale != 1.0:
            raise ValueError(
                "metric proposal disagreement scale and grid are mutually exclusive"
            )
        if arguments.metric_proposal_disagreement_threshold >= 0.0:
            raise ValueError(
                "metric proposal disagreement grid forbids adaptive thresholding"
            )
        if arguments.proximal_metric != "block":
            raise ValueError("metric proposal disagreement grid requires block metrics")
        if arguments.consensus_execution != "coordinator":
            raise ValueError(
                "metric proposal disagreement grid requires coordinator consensus"
            )
        if arguments.worker_owned_cameras:
            raise ValueError(
                "metric proposal disagreement grid requires coordinator-owned cameras"
            )
    if arguments.metric_proposal_subspace_scales != "1,1,1":
        if (
            arguments.metric_proposal_disagreement_scale != 1.0
            or arguments.metric_proposal_disagreement_grid != "1"
        ):
            raise ValueError(
                "metric proposal subspace and global damping are mutually exclusive"
            )
        if arguments.proximal_metric != "block":
            raise ValueError("metric proposal subspace damping requires block metrics")
        if arguments.consensus_execution != "coordinator":
            raise ValueError(
                "metric proposal subspace damping requires coordinator consensus"
            )
        if arguments.worker_owned_cameras:
            raise ValueError(
                "metric proposal subspace damping requires coordinator-owned cameras"
            )
    if arguments.metric_proposal_disagreement_threshold >= 0.0:
        if arguments.metric_proposal_disagreement_scale == 1.0:
            raise ValueError(
                "adaptive metric proposal damping requires a nonunit scale"
            )
    proposal_hysteresis = None
    if arguments.metric_proposal_disagreement_hysteresis:
        try:
            proposal_hysteresis = tuple(map(
                float,
                arguments.metric_proposal_disagreement_hysteresis.split(","),
            ))
        except ValueError as error:
            raise ValueError(
                "metric proposal hysteresis must contain four numbers"
            ) from error
        if len(proposal_hysteresis) != 4:
            raise ValueError(
                "metric proposal hysteresis must be strong,normal,low,high"
            )
        strong_scale, normal_scale, low_threshold, high_threshold = (
            proposal_hysteresis
        )
        if not 0.0 <= strong_scale <= normal_scale <= 1.0:
            raise ValueError(
                "metric proposal hysteresis scales must satisfy "
                "0 <= strong <= normal <= 1"
            )
        if not 0.0 <= low_threshold < high_threshold:
            raise ValueError(
                "metric proposal hysteresis thresholds must satisfy "
                "0 <= low < high"
            )
        if (
            arguments.metric_proposal_disagreement_scale != 1.0
            or arguments.metric_proposal_disagreement_grid != "1"
            or arguments.metric_proposal_disagreement_threshold >= 0.0
            or arguments.metric_proposal_subspace_scales != "1,1,1"
        ):
            raise ValueError(
                "metric proposal hysteresis is mutually exclusive with "
                "other proposal damping modes"
            )
        if arguments.proximal_metric != "block":
            raise ValueError("metric proposal hysteresis requires block metrics")
        if arguments.consensus_execution != "coordinator":
            raise ValueError(
                "metric proposal hysteresis requires coordinator consensus"
            )
        if arguments.worker_owned_cameras:
            raise ValueError(
                "metric proposal hysteresis requires coordinator-owned cameras"
            )
        if arguments.worker_consensus_shadow:
            raise ValueError(
                "metric proposal hysteresis forbids worker consensus shadow"
            )
    if arguments.shared_camera_metric_beta < 0.0:
        raise ValueError("shared camera metric beta must be nonnegative")
    if not (
        arguments.global_schur_majorizer_observability_threshold == -1.0
        or 0.0
        <= arguments.global_schur_majorizer_observability_threshold
        <= 1.0
    ):
        raise ValueError(
            "global Schur majorizer observability threshold must be -1 "
            "or in [0, 1]"
        )
    if arguments.global_schur_majorizer_until < 0:
        raise ValueError("global Schur majorizer until must be nonnegative")
    if (
        arguments.global_schur_majorizer_until > 0
        and arguments.global_schur_majorizer_observability_threshold < 0.0
    ):
        raise ValueError(
            "global Schur majorizer until requires observability selection"
        )
    if arguments.global_schur_majorizer_observability_threshold >= 0.0:
        if not arguments.shared_only_camera_proximal:
            raise ValueError(
                "global Schur majorizer selection requires shared-only "
                "camera proximal"
            )
        if os.environ.get("BUNDLE_PALM_SCHUR_PROXIMAL_METRIC", "0") == "1":
            raise ValueError(
                "global Schur majorizer selection forbids a pre-enabled "
                "Schur proximal metric"
            )
    if arguments.factorized_coupled_schur_proximal_metric:
        if arguments.global_schur_majorizer_observability_threshold < 0.0:
            raise ValueError(
                "factorized Schur metrics require the global observability selector"
            )
        if arguments.local_solver != "schur_pcg":
            raise ValueError("factorized Schur metrics require Schur PCG")
        if arguments.proximal_metric != "block":
            raise ValueError("factorized Schur metrics require block metrics")
        if arguments.consensus_metric != "full":
            raise ValueError("factorized Schur metrics require full consensus")
        if not arguments.shared_only_camera_proximal:
            raise ValueError(
                "factorized Schur metrics require shared-only camera proximal"
            )
    if not np.isfinite(arguments.unique_camera_metric_scale) or not (
        0.0 < arguments.unique_camera_metric_scale <= 1.0
    ):
        raise ValueError("unique camera metric scale must be in (0, 1]")
    if arguments.shared_only_camera_proximal:
        if arguments.proximal_metric != "block":
            raise ValueError("shared-only camera proximal requires block metrics")
        if arguments.consensus_metric != "full":
            raise ValueError("shared-only camera proximal requires full consensus")
        if arguments.consensus_execution != "coordinator":
            raise ValueError(
                "shared-only camera proximal requires coordinator consensus"
            )
        if arguments.unique_camera_metric_scale != 1.0:
            raise ValueError(
                "shared-only and fixed unique camera metrics are mutually exclusive"
            )
        if arguments.unique_camera_metric_selector:
            raise ValueError(
                "shared-only camera proximal forbids the unique metric selector"
            )
        if arguments.worker_consensus_shadow:
            raise ValueError(
                "shared-only camera proximal forbids worker consensus shadow"
            )
        if arguments.consensus_shared_floor_prior_scale != 0.0:
            raise ValueError(
                "shared-only camera proximal forbids consensus floor priors"
            )
        if (
            arguments.local_camera_step_scale != 1.0
            or arguments.local_camera_step_grid != "1"
            or arguments.shared_camera_step_grid != "1"
            or arguments.shared_camera_step_scale != 1.0
            or arguments.shared_camera_disagreement_scale != 1.0
            or arguments.metric_proposal_disagreement_grid != "1"
            or arguments.metric_proposal_disagreement_hysteresis
            or arguments.metric_proposal_subspace_scales != "1,1,1"
        ):
            raise ValueError(
                "shared-only camera proximal forbids camera proposal controls"
            )
    unique_metric_selector = None
    if arguments.unique_camera_metric_selector:
        try:
            unique_metric_selector = tuple(map(
                float, arguments.unique_camera_metric_selector.split(",")
            ))
        except ValueError as error:
            raise ValueError(
                "unique camera metric selector must contain two numbers"
            ) from error
        if len(unique_metric_selector) != 2:
            raise ValueError(
                "unique camera metric selector must be reduced,threshold"
            )
        reduced_scale, threshold = unique_metric_selector
        if not 0.0 < reduced_scale < 1.0 or threshold < 0.0:
            raise ValueError(
                "unique camera metric selector requires reduced in (0, 1) "
                "and threshold >= 0"
            )
        if (
            arguments.unique_camera_metric_scale != 1.0
        ):
            raise ValueError(
                "unique camera metric selector is mutually exclusive with "
                "other unique-camera metric controls"
            )
        if (
            arguments.worker_owned_cameras
            or arguments.worker_owned_landmarks
            or arguments.persistent_trust_region
        ):
            raise ValueError(
                "unique camera metric selector requires coordinator-owned "
                "state and non-persistent trust"
            )
        if arguments.outer_acceleration != "none":
            raise ValueError(
                "unique camera metric selector forbids outer acceleration"
            )
    if arguments.shared_camera_metric_beta != 0.0 and arguments.proximal_metric != "block":
        raise ValueError("shared camera metric beta requires block metrics")
    if arguments.single_cluster_proximal:
        if arguments.clusters != 1:
            raise ValueError("single-cluster proximal mode requires clusters=1")
        if arguments.outer_acceleration != "none":
            raise ValueError("single-cluster proximal mode forbids outer acceleration")
        if arguments.relaxation != 1.0:
            raise ValueError("single-cluster proximal mode requires unit relaxation")
        if arguments.safeguard_mode != "none":
            raise ValueError("single-cluster proximal mode requires safeguard mode none")
        if arguments.consensus_execution != "coordinator":
            raise ValueError("single-cluster proximal mode bypasses consensus execution")
        if arguments.worker_owned_cameras or arguments.worker_owned_landmarks:
            raise ValueError("single-cluster proximal mode requires materialized state")
        if arguments.consensus_landmark_refinement_steps != 0:
            raise ValueError("single-cluster proximal mode forbids consensus refinement")
    if arguments.local_solver == "ceres_se3" and not arguments.single_cluster_proximal:
        raise ValueError("ceres_se3 is restricted to single-cluster proximal mode")
    if arguments.local_solver == "ceres_se3":
        if arguments.iterations != 1 or arguments.local_steps != 1:
            raise ValueError("ceres_se3 requires one outer iteration and one local step")
        if os.environ.get("BUNDLE_PALM_CAMERA_UPDATE", "additive") != "se3_left":
            raise ValueError("ceres_se3 requires the se3_left camera update")
    if arguments.local_solver == "ceres_prox_se3":
        if os.environ.get("BUNDLE_PALM_CAMERA_UPDATE", "additive") != "se3_left":
            raise ValueError("ceres_prox_se3 requires the se3_left camera update")
        if arguments.proximal_metric != "block":
            raise ValueError("ceres_prox_se3 requires the block proximal metric")
    if not (
        arguments.local_solver_switch_iteration == 0
        or 0 < arguments.local_solver_switch_iteration < arguments.iterations
    ):
        raise ValueError("local solver switch must be in [0, iterations)")
    if arguments.local_solver_switch_iteration > 0:
        if arguments.local_solver not in ("nesterov", "schur_pcg"):
            raise ValueError(
                "local solver switching supports Nesterov and Schur-PCG only"
            )
        if arguments.local_solver_after_switch is None:
            raise ValueError(
                "local solver switch requires --local-solver-after-switch"
            )
        if arguments.local_solver_after_switch == arguments.local_solver:
            raise ValueError("local solver switch must change the solver")
    elif arguments.local_solver_after_switch is not None:
        raise ValueError(
            "local solver after-switch requires a positive switch iteration"
        )
    if arguments.adaptive_local_depth:
        if arguments.proximal_metric != "block":
            raise ValueError("adaptive local depth requires block proximal metric")
        if not arguments.local_steps < arguments.adaptive_local_depth_maximum <= 20:
            raise ValueError(
                "adaptive maximum depth must exceed local steps and be at most 20"
            )
        if not 0.0 < arguments.adaptive_local_depth_low < arguments.adaptive_local_depth_high:
            raise ValueError("adaptive depth thresholds must satisfy 0 < low < high")
        if arguments.adaptive_local_depth_window <= 0:
            raise ValueError("adaptive depth window must be positive")
        if arguments.adaptive_local_depth_dwell < 0:
            raise ValueError("adaptive depth dwell must be nonnegative")
        if not 0 <= arguments.adaptive_local_depth_start < arguments.iterations:
            raise ValueError(
                "adaptive depth start must be inside the outer iteration range"
            )
    if (
        arguments.initial_shared_schur_correction
        or arguments.mid_shared_schur_correction_iteration > 0
        or arguments.final_shared_schur_correction
    ):
        if not arguments.shared_only_camera_proximal:
            raise ValueError(
                "shared Schur correction requires shared-only DRS"
            )
        if arguments.consensus_execution != "coordinator":
            raise ValueError(
                "shared Schur correction requires coordinator consensus"
            )
        if os.environ.get("BUNDLE_PALM_CAMERA_UPDATE", "additive") != "se3_left":
            raise ValueError("shared Schur correction requires se3_left")
        if os.environ.get(
            "BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS", "0"
        ) != "1":
            raise ValueError(
                "shared Schur correction requires direct tangent assembly"
            )
    if (
        arguments.mid_shared_schur_correction_iteration < 0
        or (
            arguments.mid_shared_schur_correction_iteration > 0
            and arguments.mid_shared_schur_correction_iteration
            >= arguments.iterations
        )
    ):
        raise ValueError(
            "mid shared Schur correction must be in [0, iterations)"
        )
    if (
        arguments.mid_shared_schur_transport_product_state
        and arguments.mid_shared_schur_correction_iteration <= 0
    ):
        raise ValueError(
            "mid shared Schur product transport requires a correction iteration"
        )
    if (
        arguments.initial_shared_schur_basin_guard
        and not arguments.initial_shared_schur_correction
    ):
        raise ValueError(
            "initial shared Schur basin guard requires the initial correction"
        )
    if (
        arguments.initial_shared_schur_rebase_trust_state
        and not arguments.initial_shared_schur_correction
    ):
        raise ValueError(
            "initial shared Schur trust rebase requires the initial correction"
        )
    if arguments.initial_shared_schur_maximum_corrections <= 0:
        raise ValueError(
            "initial shared Schur maximum corrections must be positive"
        )
    if not (
        0.0
        < arguments.initial_shared_schur_model_ratio_minimum_factor
        < 1.0
    ):
        raise ValueError(
            "initial shared Schur model-ratio minimum factor must be in (0, 1)"
        )
    if (
        arguments.initial_shared_schur_correction
        or arguments.final_shared_schur_correction
    ):
        if arguments.shared_schur_landmark_damping < 0.0:
            raise ValueError("shared Schur landmark damping must be nonnegative")
        if arguments.shared_schur_camera_damping <= 0.0:
            raise ValueError("shared Schur camera damping must be positive")
        if not 0.0 < arguments.shared_schur_step_scale <= 1.0:
            raise ValueError("shared Schur step scale must be in (0, 1]")
        if not 0 <= arguments.shared_schur_landmark_refinement_steps <= 20:
            raise ValueError(
                "shared Schur landmark refinement steps must be in [0, 20]"
            )
        if not 0.0 < arguments.shared_schur_relative_tolerance < 1.0:
            raise ValueError(
                "shared Schur relative tolerance must be in (0, 1)"
            )
        if arguments.shared_schur_maximum_iterations <= 0:
            raise ValueError(
                "shared Schur maximum iterations must be positive"
            )
        if arguments.initial_shared_schur_maximum_iterations < 0:
            raise ValueError(
                "initial shared Schur maximum iterations must be nonnegative"
            )
        if arguments.shared_schur_maximum_corrections <= 0:
            raise ValueError(
                "shared Schur maximum corrections must be positive"
            )
        if arguments.shared_schur_python_confirmation_corrections < 0:
            raise ValueError(
                "shared Schur Python confirmation corrections must be "
                "nonnegative"
            )
        if (
            arguments.shared_schur_confirmation_camera_damping == 0.0
            or arguments.shared_schur_confirmation_landmark_damping == 0.0
        ):
            raise ValueError(
                "shared Schur confirmation damping must be positive or "
                "negative to inherit"
            )
        if arguments.shared_schur_maximum_attempts <= 0:
            raise ValueError("shared Schur maximum attempts must be positive")
        fallback_damping_enabled = (
            arguments.shared_schur_fallback_camera_damping > 0.0
            or arguments.shared_schur_fallback_landmark_damping > 0.0
        )
        if fallback_damping_enabled and not (
            arguments.shared_schur_fallback_camera_damping > 0.0
            and arguments.shared_schur_fallback_landmark_damping > 0.0
        ):
            raise ValueError(
                "shared Schur fallback camera and landmark damping must both "
                "be positive or both be disabled"
            )
        if arguments.shared_schur_damping_increase <= 1.0:
            raise ValueError("shared Schur damping increase must exceed one")
        if not 0.0 < arguments.shared_schur_damping_decrease < 1.0:
            raise ValueError("shared Schur damping decrease must be in (0, 1)")
        if not 0.0 <= arguments.shared_schur_minimum_gain_ratio < 1.0:
            raise ValueError(
                "shared Schur minimum gain ratio must be in [0, 1)"
            )
        if not 0.0 <= arguments.shared_schur_minimum_relative_decrease < 1.0:
            raise ValueError(
                "shared Schur minimum relative decrease must be in [0, 1)"
            )
    if arguments.nesterov_max_iterations <= 0:
        raise ValueError("Nesterov maximum iterations must be positive")
    if arguments.enhanced_inner_max_iterations <= 0:
        raise ValueError("enhanced inner maximum iterations must be positive")
    if not 1 <= arguments.nesterov_min_iterations <= arguments.nesterov_max_iterations:
        raise ValueError(
            "Nesterov minimum iterations must be between 1 and the maximum"
        )
    if not 0.0 < arguments.nesterov_stop_tolerance < 1.0:
        raise ValueError("Nesterov stop tolerance must be in (0, 1)")
    if not 0 <= arguments.enhanced_inner_until <= arguments.iterations:
        raise ValueError("enhanced inner cutoff must be in [0, iterations]")
    if not 0 <= arguments.diagonal_trust_until <= arguments.iterations:
        raise ValueError("diagonal trust cutoff must be in [0, iterations]")
    if not 0 <= arguments.relative_residual_until <= arguments.iterations:
        raise ValueError("relative residual cutoff must be in [0, iterations]")
    if not 0 <= arguments.stop_after_iteration <= arguments.iterations:
        raise ValueError("iteration stop must be in [0, iterations]")
    if not 0 <= arguments.collective_trust_trial_until <= arguments.iterations:
        raise ValueError(
            "collective trust trial cutoff must be in [0, iterations]"
        )
    if (
        arguments.collective_trust_trial_until > 0
        and not arguments.persistent_trust_region
    ):
        raise ValueError("collective trust trials require persistent trust")
    if (
        arguments.collective_trust_trial_until > 0
        and arguments.worker_owned_cameras
    ):
        raise ValueError(
            "collective trust trials require coordinator-owned cameras"
        )
    if not (
        arguments.local_state_rebase_iteration == 0
        or 0 < arguments.local_state_rebase_iteration < arguments.iterations
    ):
        raise ValueError("local state rebase must be in [0, iterations)")
    if not 0.0 < arguments.relaxation < 2.0:
        raise ValueError("relaxation must be in (0, 2)")
    if arguments.acceleration_restart_after <= 0:
        raise ValueError("acceleration restart count must be positive")
    if not 0 <= arguments.outer_acceleration_until <= arguments.iterations:
        raise ValueError("outer acceleration cutoff must be in [0, iterations]")
    if not (
        arguments.outer_acceleration_restart_iteration == 0
        or 0 < arguments.outer_acceleration_restart_iteration < arguments.iterations
    ):
        raise ValueError(
            "outer acceleration restart iteration must be in [0, iterations)"
        )
    if (
        arguments.outer_acceleration_until > 0
        and arguments.outer_acceleration == "none"
    ):
        raise ValueError("outer acceleration cutoff requires acceleration")
    if (
        arguments.outer_acceleration_restart_iteration > 0
        and arguments.outer_acceleration == "none"
    ):
        raise ValueError("outer acceleration restart requires acceleration")
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
        not np.isfinite(arguments.consensus_shared_floor_prior_scale)
        or arguments.consensus_shared_floor_prior_scale < 0.0
    ):
        raise ValueError(
            "consensus shared floor prior scale must be finite and nonnegative"
        )
    if arguments.consensus_shared_floor_prior_scale > 0.0 and not (
        arguments.proximal_metric == "block"
        and arguments.consensus_metric == "full"
        and arguments.consensus_execution == "coordinator"
        and arguments.metric_proposal_disagreement_grid == "1"
        and arguments.metric_proposal_disagreement_hysteresis == ""
        and arguments.metric_proposal_subspace_scales == "1,1,1"
    ):
        raise ValueError(
            "shared floor prior requires fixed-scale coordinator block/full consensus"
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
    if arguments.recovery_relaxed_iterations <= 0:
        raise ValueError("recovery relaxed iterations must be positive")
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
    if not np.isfinite(arguments.huber_delta) or arguments.huber_delta < 0.0:
        raise ValueError("Huber delta must be finite and nonnegative")
    if arguments.safeguard_annealing_iterations < 0:
        raise ValueError("safeguard annealing iterations must be nonnegative")
    if arguments.safeguard_reference_iteration < 0:
        raise ValueError("safeguard reference iteration must be nonnegative")
    if (
        not np.isfinite(arguments.safeguard_annealing_exponent)
        or arguments.safeguard_annealing_exponent <= 0.0
    ):
        raise ValueError("safeguard annealing exponent must be positive and finite")
    if (
        not np.isfinite(arguments.safeguard_relative_deadband)
        or arguments.safeguard_relative_deadband < 0.0
    ):
        raise ValueError("safeguard relative deadband must be finite and nonnegative")
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


def diagonal_trust_is_active(iteration, cutoff):
    return (
        os.environ.get("BUNDLE_PALM_DIAGONAL_TRUST_DAMPING", "0") == "1"
        or iteration < cutoff
    )


def transport_product_camera_state(
    local_cameras,
    centers,
    tangent_step,
    camera_scaling,
):
    local_cameras = np.asarray(local_cameras, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    tangent_step = np.asarray(tangent_step, dtype=np.float64)
    if local_cameras.shape != centers.shape or local_cameras.ndim != 3:
        raise ValueError("local cameras and centers must have equal 3D shapes")
    if tangent_step.shape != local_cameras.shape[1:]:
        raise ValueError("tangent step must match one global camera state")

    physical_cameras = to_physical_cameras(local_cameras, camera_scaling)
    repeated_step = np.broadcast_to(tangent_step, local_cameras.shape)
    transported_physical = left_se3_camera_plus(
        physical_cameras,
        repeated_step,
    )
    transported_cameras = to_scaled_cameras(
        transported_physical,
        camera_scaling,
    )
    transported_centers = transported_cameras - (local_cameras - centers)
    return transported_cameras, transported_centers


def run_mid_shared_schur_rebase(
    worker,
    arguments,
    camera_indices_in_cluster,
    point_indices_in_cluster,
    accepted_consensus,
    accepted_landmarks,
    camera_scaling,
    camera_count,
    cluster_count,
    evaluate_state,
    camera_indices,
    point_indices,
    observations,
    materialization_state_id,
    local_cameras=None,
    centers=None,
):
    started_at = time.perf_counter()
    current_landmarks = accepted_landmarks.copy()
    if arguments.worker_owned_landmarks:
        current_landmarks = worker.materialize_current_landmarks(
            point_indices_in_cluster,
            current_landmarks,
            cluster_count,
            materialization_state_id,
            source="accepted",
        )
    initial_metrics = evaluate_state(
        to_physical_cameras(accepted_consensus, camera_scaling),
        current_landmarks,
        camera_indices,
        point_indices,
        observations,
    )
    systems = worker.build_schur_systems(
        camera_indices_in_cluster,
        point_indices_in_cluster,
        accepted_consensus,
        current_landmarks,
        cluster_count,
        arguments.shared_schur_landmark_damping,
    )
    tangent_step, diagnostics = solve_global_schur_system(
        systems,
        camera_count,
        arguments.shared_schur_camera_damping,
        arguments.shared_schur_step_scale,
        arguments.shared_schur_linear_solver,
        arguments.shared_schur_relative_tolerance,
        arguments.shared_schur_maximum_iterations,
        operator_mode=arguments.shared_schur_operator,
        preconditioner_mode=arguments.shared_schur_preconditioner,
        coarse_basis=(
            similarity_gauge_tangent_basis(
                to_physical_cameras(accepted_consensus, camera_scaling)
            )
            if arguments.shared_schur_preconditioner == "gauge_deflated"
            else None
        ),
    )
    corrected_costs, corrected_consensus, corrected_landmarks = (
        worker.apply_camera_step(
            camera_indices_in_cluster,
            point_indices_in_cluster,
            accepted_consensus,
            current_landmarks,
            tangent_step,
            cluster_count,
            arguments.shared_schur_landmark_refinement_steps,
        )
    )
    candidate_metrics = evaluate_state(
        to_physical_cameras(corrected_consensus, camera_scaling),
        corrected_landmarks,
        camera_indices,
        point_indices,
        observations,
    )
    initial_sse = initial_metrics["objectiveValue"]
    candidate_sse = candidate_metrics["objectiveValue"]
    accepted = (
        diagnostics["linearTermination"] == 0
        and np.isfinite(candidate_sse)
        and candidate_sse < initial_sse
    )
    relative_decrease = (
        (initial_sse - candidate_sse)
        / max(initial_sse, np.finfo(np.float64).tiny)
        if np.isfinite(candidate_sse)
        else float("-inf")
    )
    diagnostics = {
        **diagnostics,
        "initialSSE": initial_sse,
        "candidateSSE": candidate_sse,
        "workerSSE": float(np.sum(corrected_costs)),
        "relativeDecrease": relative_decrease,
        "accepted": accepted,
    }
    pre_rebase_radii = worker.last_trust_region_radii.tolist()
    post_rebase_radii = []
    trust_rebased = False
    product_state_transported = False
    transported_local_cameras = None
    transported_centers = None
    transport_worker_sse = float("nan")
    transport_offset_error = float("nan")
    if accepted:
        if getattr(
            arguments,
            "mid_shared_schur_transport_product_state",
            False,
        ):
            if local_cameras is None or centers is None:
                raise ValueError(
                    "product-state transport requires local cameras and centers"
                )
            (
                expected_local_cameras,
                expected_centers,
            ) = transport_product_camera_state(
                local_cameras,
                centers,
                tangent_step,
                camera_scaling,
            )
            active_cameras = np.zeros(
                local_cameras.shape[:2], dtype=bool
            )
            for cluster_id, indices in enumerate(camera_indices_in_cluster):
                active_cameras[cluster_id, np.unique(indices)] = True
            expected_local_cameras = np.where(
                active_cameras[..., None],
                expected_local_cameras,
                local_cameras,
            )
            expected_centers = np.where(
                active_cameras[..., None],
                expected_centers,
                centers,
            )
            (
                transport_costs,
                transported_local_cameras,
                transported_landmarks,
            ) = worker.transport_product_state(
                camera_indices_in_cluster,
                point_indices_in_cluster,
                local_cameras,
                centers,
                corrected_landmarks,
                tangent_step,
                cluster_count,
            )
            if not (
                np.allclose(
                    transported_landmarks,
                    corrected_landmarks,
                    rtol=1e-12,
                    atol=1e-14,
                )
                and np.isfinite(np.sum(transport_costs))
            ):
                raise RuntimeError(
                    "worker product-state transport changed landmarks or cost"
                )
            original_offsets = np.asarray(local_cameras) - np.asarray(centers)
            transported_centers = transported_local_cameras - original_offsets
            transported_centers = np.where(
                active_cameras[..., None],
                transported_centers,
                centers,
            )
            transported_offsets = (
                transported_local_cameras - transported_centers
            )
            transport_offset_error = float(
                np.max(
                    np.abs(transported_offsets - original_offsets)[
                        active_cameras
                    ]
                )
            )
            transport_worker_sse = float(np.sum(transport_costs))
            diagnostics["transportExpectedCameraCoordinateError"] = float(
                np.max(
                    np.abs(
                        transported_local_cameras - expected_local_cameras
                    )[active_cameras]
                )
            )
            diagnostics["transportExpectedCenterCoordinateError"] = float(
                np.max(
                    np.abs(transported_centers - expected_centers)[
                        active_cameras
                    ]
                )
            )
            post_rebase_radii = worker.last_trust_region_radii.tolist()
            if not np.array_equal(
                np.asarray(post_rebase_radii),
                np.asarray(pre_rebase_radii),
            ):
                raise RuntimeError(
                    "product-state transport changed worker trust radii"
                )
            product_state_transported = True
        else:
            rebase_costs, rebased_consensus, rebased_landmarks = (
                worker.apply_camera_step(
                    camera_indices_in_cluster,
                    point_indices_in_cluster,
                    corrected_consensus,
                    corrected_landmarks,
                    np.zeros_like(tangent_step),
                    cluster_count,
                    0,
                    rebase_trust_state=True,
                )
            )
            if not (
                np.allclose(
                    rebased_consensus,
                    corrected_consensus,
                    rtol=1e-12,
                    atol=1e-14,
                )
                and np.allclose(
                    rebased_landmarks,
                    corrected_landmarks,
                    rtol=1e-12,
                    atol=1e-14,
                )
                and np.isfinite(np.sum(rebase_costs))
            ):
                raise RuntimeError(
                    "mid-run trust rebase changed accepted correction geometry"
                )
            corrected_consensus = rebased_consensus
            corrected_landmarks = rebased_landmarks
            post_rebase_radii = worker.last_trust_region_radii.tolist()
            trust_rebased = True
    else:
        worker.apply_camera_step(
            camera_indices_in_cluster,
            point_indices_in_cluster,
            accepted_consensus,
            current_landmarks,
            np.zeros_like(tangent_step),
            cluster_count,
            0,
        )
        corrected_consensus = accepted_consensus.copy()
        corrected_landmarks = current_landmarks
    return {
        "accepted": accepted,
        "initialMetrics": initial_metrics,
        "candidateMetrics": candidate_metrics,
        "workerSSE": float(np.sum(corrected_costs)),
        "consensus": corrected_consensus,
        "landmarks": corrected_landmarks,
        "diagnostics": diagnostics,
        "trustRebased": trust_rebased,
        "productStateTransported": product_state_transported,
        "localCameras": transported_local_cameras,
        "centers": transported_centers,
        "transportWorkerSSE": transport_worker_sse,
        "transportOffsetError": transport_offset_error,
        "preRebaseTrustRadii": pre_rebase_radii,
        "postRebaseTrustRadii": post_rebase_radii,
        "seconds": time.perf_counter() - started_at,
    }


def main():
    arguments = parse_arguments()
    validate_arguments(arguments)
    camera_diagonal_quantile_iterations = {
        int(value) - 1
        for value in arguments.camera_diagonal_quantile_iterations.split(",")
        if value
    }
    if any(value < 0 for value in camera_diagonal_quantile_iterations):
        raise ValueError(
            "camera diagonal quantile iterations must be positive"
        )
    camera_disagreement_diagnostic_ids = tuple(
        int(value)
        for value in arguments.camera_disagreement_diagnostic_ids.split(",")
        if value
    )
    camera_disagreement_diagnostic_iterations = {
        int(value) - 1
        for value in arguments.camera_disagreement_diagnostic_iterations.split(",")
        if value
    }
    schur_alignment_diagnostic_iterations = {
        int(value) - 1
        for value in arguments.schur_alignment_diagnostic_iterations.split(",")
        if value
    }
    if any(value < 0 for value in camera_disagreement_diagnostic_ids):
        raise ValueError("camera disagreement diagnostic IDs must be nonnegative")
    if any(value < 0 for value in camera_disagreement_diagnostic_iterations):
        raise ValueError(
            "camera disagreement diagnostic iterations must be positive"
        )
    if any(value < 0 for value in schur_alignment_diagnostic_iterations):
        raise ValueError("Schur alignment diagnostic iterations must be positive")
    if schur_alignment_diagnostic_iterations and not (
        arguments.schur_alignment_camera_damping > 0.0
        and arguments.schur_alignment_landmark_damping >= 0.0
        and arguments.schur_alignment_maximum_iterations > 0
    ):
        raise ValueError(
            "Schur alignment camera damping must be positive and landmark "
            "damping nonnegative, with positive maximum iterations"
        )
    if (
        arguments.schur_model_consensus_clipping
        and not schur_alignment_diagnostic_iterations
    ):
        raise ValueError(
            "Schur-model consensus clipping requires alignment iterations"
        )
    if not 0.0 <= arguments.schur_model_consensus_clipping_minimum_scale <= 1.0:
        raise ValueError(
            "Schur-model consensus clipping minimum scale must be in [0, 1]"
        )
    if not (
        arguments.schur_model_consensus_clipping_minimum_scale
        <= arguments.schur_model_consensus_clipping_maximum_scale
        <= 1.0
    ):
        raise ValueError(
            "Schur-model consensus clipping maximum scale must be in "
            "[minimum scale, 1]"
        )
    proposal_hysteresis = (
        tuple(map(
            float,
            arguments.metric_proposal_disagreement_hysteresis.split(","),
        ))
        if arguments.metric_proposal_disagreement_hysteresis
        else None
    )
    unique_metric_selector = (
        tuple(map(float, arguments.unique_camera_metric_selector.split(",")))
        if arguments.unique_camera_metric_selector
        else None
    )
    started_at = time.perf_counter()

    raw_cameras, raw_points, camera_indices, point_indices, raw_observations = (
        read_bal_problem(arguments.dataset)
    )
    initial_state_cameras = None
    initial_state_points = None
    if arguments.initial_state:
        expected_camera_shape = raw_cameras.shape
        expected_point_shape = raw_points.shape
        with np.load(arguments.initial_state) as initial_state:
            initial_state_cameras = np.asarray(
                initial_state["cameras"], dtype=np.float64
            )
            initial_state_points = np.asarray(
                initial_state["points"], dtype=np.float64
            )
        if initial_state_cameras.shape != expected_camera_shape:
            raise ValueError("initial-state camera shape does not match dataset")
        if initial_state_points.shape != expected_point_shape:
            raise ValueError("initial-state point shape does not match dataset")
        if arguments.initial_state_frame == "raw":
            raw_cameras = initial_state_cameras
            raw_points = initial_state_points
    cameras, points, observations = canonicalize_bal_problem(
        raw_cameras,
        raw_points,
        camera_indices,
        raw_observations,
        normalize_scene=arguments.scene_normalization == "points_p95",
    )
    if initial_state_cameras is not None and (
        arguments.initial_state_frame == "canonical"
    ):
        cameras = initial_state_cameras.copy()
        points = initial_state_points.copy()
    camera_count = len(cameras)
    point_count = len(points)
    huber_delta = arguments.huber_delta if arguments.huber_delta > 0.0 else None
    def evaluate_state(state_cameras, state_points, *_):
        return evaluate_bal_state(
            state_cameras,
            state_points,
            camera_indices,
            point_indices,
            observations,
            huber_delta=huber_delta,
        )

    initial_metrics = evaluate_state(
        cameras, points, camera_indices, point_indices, observations
    )
    if arguments.debug_output:
        print_setup(
            arguments, camera_count, point_count, len(observations), initial_metrics
        )

    partition_started = time.perf_counter()
    partitioners = {
        "landmark_scalable": cluster_by_landmark_scalable,
        "landmark_scalable_stable": cluster_by_landmark_scalable_stable,
        "daba_louvain": cluster_by_daba_louvain,
    }
    partitioner = partitioners[arguments.clustering]
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
    camera_copy_count = np.sum(camera_masks, axis=0)
    current_unique_camera_metric_scale = arguments.unique_camera_metric_scale
    camera_proximal_multipliers = camera_metric_multipliers(
        camera_copy_count,
        arguments.shared_camera_metric_beta,
        current_unique_camera_metric_scale,
    )
    if arguments.shared_only_camera_proximal:
        camera_proximal_multipliers[camera_copy_count == 1] = 0.0
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
    worker_block_scaling = (
        arguments.camera_scaling == "worker_block_jacobi_initial"
    )
    worker_diagonal_scaling = (
        arguments.camera_scaling == "worker_diagonal_jacobi_initial"
    )
    worker_z_f_scaling = (
        arguments.camera_scaling == "worker_z_f_block_jacobi_initial"
    )
    if arguments.camera_scaling == "block_jacobi_initial":
        camera_scaling = compute_initial_block_jacobi_maps(
            cameras, points, camera_indices, point_indices
        )
        camera_scaling *= np.sqrt(cluster_count) / 2.0
    elif arguments.camera_scaling in {"jacobi_initial", "ruiz_initial"}:
        scaling_function = (
            compute_initial_ruiz_scaling
            if arguments.camera_scaling == "ruiz_initial"
            else compute_initial_jacobi_scaling
        )
        camera_scaling = scaling_function(
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
    best_sse = initial_metrics["objectiveValue"]
    best_metrics = initial_metrics.copy()
    best_iteration = -1
    best_cameras = cameras.copy()
    best_points = points.copy()
    accepted_metrics = initial_metrics
    accepted_dre = initial_metrics["objectiveValue"]
    accepted_model_dre = initial_metrics["objectiveValue"]
    accepted_fixed_point_squared = 0.0
    accepted_consensus = consensus.copy()
    accepted_landmarks = landmarks.copy()
    rejected_count = 0
    best_checkpoint_restarts = 0
    relaxed_exploration_remaining = 0
    accepted_since_curvature_increase = 0
    revert_landmark_mode = 0
    trust_region_recovery_ratio = 1.0
    shared_trust_region_radius = arguments.shared_trust_region_initial_radius
    collective_trust_region_radii = None
    trajectory = []
    termination_reason = "iteration_limit"
    final_polishing_applied = False
    final_polishing_initial_sse = float("nan")
    final_polishing_refined_sse = float("nan")
    final_shared_schur_attempted = False
    final_shared_schur_accepted = False
    final_shared_schur_initial_sse = float("nan")
    final_shared_schur_corrected_sse = float("nan")
    final_shared_schur_worker_sse = float("nan")
    final_shared_schur_seconds = 0.0
    final_shared_schur_diagnostics = {}
    final_shared_schur_attempts = []
    final_shared_schur_accepted_corrections = 0
    final_shared_schur_screening_corrections = 0
    final_shared_schur_confirmation_corrections = 0
    final_shared_schur_termination = "disabled"
    final_shared_schur_final_camera_damping = float("nan")
    final_shared_schur_final_landmark_damping = float("nan")
    initial_shared_schur_attempted = False
    initial_shared_schur_accepted = False
    initial_shared_schur_initial_sse = float("nan")
    initial_shared_schur_candidate_sse = float("nan")
    initial_shared_schur_worker_sse = float("nan")
    initial_shared_schur_seconds = 0.0
    initial_shared_schur_diagnostics = {}
    initial_shared_schur_attempts = []
    initial_shared_schur_accepted_corrections = 0
    initial_shared_schur_termination = "disabled"
    initial_shared_schur_trust_rebased = False
    initial_shared_schur_pre_rebase_trust_radii = []
    initial_shared_schur_post_rebase_trust_radii = []
    mid_shared_schur_attempted = False
    mid_shared_schur_accepted = False
    mid_shared_schur_initial_sse = float("nan")
    mid_shared_schur_candidate_sse = float("nan")
    mid_shared_schur_worker_sse = float("nan")
    mid_shared_schur_seconds = 0.0
    mid_shared_schur_diagnostics = {}
    mid_shared_schur_trust_rebased = False
    mid_shared_schur_product_state_transported = False
    mid_shared_schur_transport_worker_sse = float("nan")
    mid_shared_schur_transport_offset_error = float("nan")
    mid_shared_schur_pre_rebase_trust_radii = []
    mid_shared_schur_post_rebase_trust_radii = []
    bootstrap_basin_guard_active = False
    bootstrap_basin_guard_ceiling = float("nan")
    bootstrap_basin_guard_rejections = 0
    bootstrap_basin_guard_release_iteration = -1
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
    adaptive_local_steps = np.full(
        cluster_count, arguments.local_steps, dtype=np.int64
    )
    adaptive_defect_history = []
    adaptive_depth_dwell_remaining = np.zeros(cluster_count, dtype=np.int64)
    metric_proposal_hysteresis_scale = (
        proposal_hysteresis[1] if proposal_hysteresis is not None else 1.0
    )
    return_proximal_diagnostics = (
        arguments.metric_diagnostic_iterations > 0
        or arguments.adaptive_local_depth
        or arguments.interior_defect_diagnostic
        or arguments.global_schur_majorizer_observability_threshold >= 0.0
        or os.environ.get(
            "BUNDLE_PALM_SCHUR_OBSERVABILITY_DIAGNOSTIC", "0"
        ) == "1"
    )
    global_schur_majorizer_decided = False
    global_schur_majorizer_selected = False
    global_schur_observability_statistic = float("nan")
    global_schur_observability_valid_clusters = 0
    worker_translation_z_focal_correlation = np.empty(0, dtype=np.float64)

    worker = DrsWorkerClient()
    try:
        bootstrap_cameras = np.repeat(
            cameras[None, :, :], cluster_count, axis=0
        )
        bootstrap_result = worker.solve_batch(
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
            diagonal_trust_damping=False,
            nesterov_relative_residual=False,
            camera_proximal_multipliers=camera_proximal_multipliers,
            huber_delta=arguments.huber_delta,
        )
        if worker_block_scaling or worker_diagonal_scaling or worker_z_f_scaling:
            if arguments.proximal_metric != "block":
                raise ValueError(
                    "worker block scaling requires block proximal metrics"
                )
            _, bootstrap_metric_blocks = bootstrap_result
            aggregate_blocks = aggregate_camera_metric_blocks(
                bootstrap_metric_blocks, camera_count
            )
            worker_translation_z_focal_correlation = (
                camera_block_correlation(aggregate_blocks, 5, 6)
            )
            if worker_diagonal_scaling:
                camera_scaling = diagonal_jacobi_scaling_from_blocks(
                    aggregate_blocks,
                    relative_floor=1e-6,
                    maximum_ratio=arguments.camera_scaling_maximum_ratio,
                )
                camera_scaling *= 2.0 / np.sqrt(cluster_count)
            elif worker_z_f_scaling:
                camera_scaling = restricted_block_jacobi_coordinate_maps(
                    aggregate_blocks,
                    coupled_coordinates=(5, 6),
                    relative_floor=1e-6,
                )
                camera_scaling *= np.sqrt(cluster_count) / 2.0
            else:
                camera_scaling = block_jacobi_coordinate_maps(
                    aggregate_blocks, relative_floor=1e-6
                )
                camera_scaling *= np.sqrt(cluster_count) / 2.0
            local_cameras = to_scaled_cameras(
                bootstrap_cameras, camera_scaling
            )
            centers = to_scaled_cameras(centers, camera_scaling)
            consensus = to_scaled_cameras(consensus, camera_scaling)
            accepted_consensus = consensus.copy()
        worker.update_preconditioning(
            camera_indices_in_cluster,
            point_indices_in_cluster,
            camera_scaling,
            cluster_count,
            camera_state=(
                local_cameras
                if (
                    arguments.worker_owned_cameras
                    or worker_block_scaling
                    or worker_diagonal_scaling
                    or worker_z_f_scaling
                )
                else None
            ),
        )
        if arguments.factorized_coupled_schur_proximal_metric:
            startup_schur_systems = worker.build_schur_systems(
                camera_indices_in_cluster,
                point_indices_in_cluster,
                consensus,
                landmarks,
                cluster_count,
                0.0,
            )
            startup_observability = schur_system_observability_fractions(
                startup_schur_systems
            )
            (
                global_schur_majorizer_selected,
                global_schur_observability_statistic,
                global_schur_observability_valid_clusters,
            ) = select_global_schur_majorizer(
                startup_observability,
                arguments.global_schur_majorizer_observability_threshold,
            )
            global_schur_majorizer_decided = True
        if arguments.initial_shared_schur_correction:
            initial_shared_schur_attempted = True
            initial_shared_schur_initial_sse = best_sse
            initial_shared_schur_termination = "maximum_corrections"
            initial_schur_started = time.perf_counter()
            schur_systems = worker.build_schur_systems(
                camera_indices_in_cluster,
                point_indices_in_cluster,
                consensus,
                landmarks,
                cluster_count,
                arguments.shared_schur_landmark_damping,
            )
            tangent_step, initial_shared_schur_diagnostics = (
                solve_global_schur_system(
                    schur_systems,
                    camera_count,
                    arguments.shared_schur_camera_damping,
                    arguments.shared_schur_step_scale,
                    arguments.shared_schur_linear_solver,
                    arguments.shared_schur_relative_tolerance,
                    (
                        arguments.initial_shared_schur_maximum_iterations
                        or arguments.shared_schur_maximum_iterations
                    ),
                    operator_mode=(
                        arguments.shared_schur_operator
                        if arguments.initial_shared_schur_operator == "inherit"
                        else arguments.initial_shared_schur_operator
                    ),
                    preconditioner_mode=(
                        arguments.shared_schur_preconditioner
                    ),
                    coarse_basis=(
                        similarity_gauge_tangent_basis(
                            to_physical_cameras(consensus, camera_scaling)
                        )
                        if arguments.shared_schur_preconditioner
                        == "gauge_deflated"
                        else None
                    ),
                )
            )
            (
                corrected_costs,
                corrected_scaled_cameras,
                corrected_points,
            ) = worker.apply_camera_step(
                camera_indices_in_cluster,
                point_indices_in_cluster,
                consensus,
                landmarks,
                tangent_step,
                cluster_count,
                arguments.shared_schur_landmark_refinement_steps,
            )
            initial_shared_schur_worker_sse = float(np.sum(corrected_costs))
            corrected_cameras = to_physical_cameras(
                corrected_scaled_cameras, camera_scaling
            )
            corrected_metrics = evaluate_state(
                corrected_cameras,
                corrected_points,
                camera_indices,
                point_indices,
                observations,
            )
            initial_shared_schur_candidate_sse = corrected_metrics[
                "sumSquaredError"
            ]
            initial_shared_schur_accepted = (
                initial_shared_schur_diagnostics["linearTermination"] == 0
                and np.isfinite(initial_shared_schur_candidate_sse)
                and initial_shared_schur_candidate_sse < best_sse
            )
            initial_assessment = assess_schur_trial(
                best_sse,
                initial_shared_schur_candidate_sse,
                initial_shared_schur_diagnostics,
                "geometric",
                arguments.shared_schur_minimum_gain_ratio,
            )
            initial_relative_decrease = (
                (best_sse - initial_shared_schur_candidate_sse)
                / max(best_sse, np.finfo(np.float64).tiny)
                if np.isfinite(initial_shared_schur_candidate_sse)
                else float("-inf")
            )
            initial_shared_schur_attempts.append({
                "correction": 0,
                "attempt": 0,
                "cameraDamping": arguments.shared_schur_camera_damping,
                "landmarkDamping": arguments.shared_schur_landmark_damping,
                "initialSSE": best_sse,
                "candidateSSE": initial_shared_schur_candidate_sse,
                "workerSSE": initial_shared_schur_worker_sse,
                "relativeDecrease": initial_relative_decrease,
                "accepted": initial_shared_schur_accepted,
                "actualReduction": initial_assessment["actualReduction"],
                "dampedGainRatio": initial_assessment["dampedGainRatio"],
                "undampedGainRatio": initial_assessment[
                    "undampedGainRatio"
                ],
                "diagnostics": initial_shared_schur_diagnostics,
            })
            if initial_shared_schur_accepted:
                initial_shared_schur_accepted_corrections = 1
                scaled_cameras = corrected_scaled_cameras.copy()
                local_cameras = np.repeat(
                    scaled_cameras[None, :, :], cluster_count, axis=0
                )
                centers = local_cameras.copy()
                consensus = scaled_cameras.copy()
                landmarks = corrected_points.copy()
                best_sse = initial_shared_schur_candidate_sse
                best_metrics = corrected_metrics.copy()
                best_cameras = corrected_cameras.copy()
                best_points = corrected_points.copy()
                accepted_metrics = corrected_metrics.copy()
                accepted_dre = initial_shared_schur_candidate_sse
                accepted_model_dre = initial_shared_schur_candidate_sse
                accepted_fixed_point_squared = 0.0
                accepted_consensus = consensus.copy()
                accepted_landmarks = landmarks.copy()
                initial_damping_factor = (
                    model_ratio_damping_factor(
                        initial_assessment["dampedGainRatio"],
                        arguments.
                        initial_shared_schur_model_ratio_minimum_factor,
                    )
                    if arguments.initial_shared_schur_damping_policy
                    == "model_ratio"
                    else arguments.shared_schur_damping_decrease
                )
                camera_damping = (
                    arguments.shared_schur_camera_damping
                    * initial_damping_factor
                )
                landmark_damping = (
                    arguments.shared_schur_landmark_damping
                    * initial_damping_factor
                )
                if initial_relative_decrease < (
                    arguments.shared_schur_minimum_relative_decrease
                ):
                    initial_shared_schur_termination = (
                        "minimum_relative_decrease"
                    )
                else:
                    correction = 1
                    attempt = 0
                    rejection_damping_factor = 2.0
                    while correction < (
                        arguments.initial_shared_schur_maximum_corrections
                    ):
                        correction_initial_sse = best_sse
                        schur_systems = worker.build_schur_systems(
                            camera_indices_in_cluster,
                            point_indices_in_cluster,
                            consensus,
                            landmarks,
                            cluster_count,
                            landmark_damping,
                        )
                        tangent_step, initial_shared_schur_diagnostics = (
                            solve_global_schur_system(
                                schur_systems,
                                camera_count,
                                camera_damping,
                                arguments.shared_schur_step_scale,
                                arguments.shared_schur_linear_solver,
                                arguments.shared_schur_relative_tolerance,
                                (
                                    arguments.
                                    initial_shared_schur_maximum_iterations
                                    or arguments.shared_schur_maximum_iterations
                                ),
                                operator_mode=(
                                    arguments.shared_schur_operator
                                    if arguments.initial_shared_schur_operator
                                    == "inherit"
                                    else arguments.initial_shared_schur_operator
                                ),
                                preconditioner_mode=(
                                    arguments.shared_schur_preconditioner
                                ),
                                coarse_basis=(
                                    similarity_gauge_tangent_basis(
                                        to_physical_cameras(
                                            consensus, camera_scaling
                                        )
                                    )
                                    if arguments.shared_schur_preconditioner
                                    == "gauge_deflated"
                                    else None
                                ),
                            )
                        )
                        (
                            corrected_costs,
                            corrected_scaled_cameras,
                            corrected_points,
                        ) = worker.apply_camera_step(
                            camera_indices_in_cluster,
                            point_indices_in_cluster,
                            consensus,
                            landmarks,
                            tangent_step,
                            cluster_count,
                            arguments.shared_schur_landmark_refinement_steps,
                        )
                        initial_shared_schur_worker_sse = float(
                            np.sum(corrected_costs)
                        )
                        corrected_cameras = to_physical_cameras(
                            corrected_scaled_cameras, camera_scaling
                        )
                        corrected_metrics = evaluate_state(
                            corrected_cameras,
                            corrected_points,
                            camera_indices,
                            point_indices,
                            observations,
                        )
                        initial_shared_schur_candidate_sse = (
                            corrected_metrics["sumSquaredError"]
                        )
                        relative_decrease = (
                            (
                                correction_initial_sse
                                - initial_shared_schur_candidate_sse
                            )
                            / max(
                                correction_initial_sse,
                                np.finfo(np.float64).tiny,
                            )
                            if np.isfinite(initial_shared_schur_candidate_sse)
                            else float("-inf")
                        )
                        assessment = assess_schur_trial(
                            correction_initial_sse,
                            initial_shared_schur_candidate_sse,
                            initial_shared_schur_diagnostics,
                            arguments.initial_shared_schur_damping_policy,
                            arguments.shared_schur_minimum_gain_ratio,
                        )
                        initial_shared_schur_attempts.append({
                            "correction": correction,
                            "attempt": attempt,
                            "cameraDamping": camera_damping,
                            "landmarkDamping": landmark_damping,
                            "initialSSE": correction_initial_sse,
                            "candidateSSE": initial_shared_schur_candidate_sse,
                            "workerSSE": initial_shared_schur_worker_sse,
                            "relativeDecrease": relative_decrease,
                            **assessment,
                            "diagnostics": initial_shared_schur_diagnostics,
                        })
                        if not assessment["accepted"]:
                            worker.apply_camera_step(
                                camera_indices_in_cluster,
                                point_indices_in_cluster,
                                consensus,
                                landmarks,
                                np.zeros_like(tangent_step),
                                cluster_count,
                                0,
                            )
                            if (
                                arguments.initial_shared_schur_damping_policy
                                == "model_ratio"
                                and attempt + 1
                                < arguments.shared_schur_maximum_attempts
                            ):
                                camera_damping, landmark_damping = (
                                    rejected_schur_damping(
                                        camera_damping,
                                        landmark_damping,
                                        attempt,
                                        rejection_damping_factor,
                                        arguments.
                                        shared_schur_fallback_camera_damping,
                                        arguments.
                                        shared_schur_fallback_landmark_damping,
                                    )
                                )
                                rejection_damping_factor *= 2.0
                                attempt += 1
                                continue
                            initial_shared_schur_termination = (
                                "attempts_exhausted"
                                if arguments.initial_shared_schur_damping_policy
                                == "model_ratio"
                                else "rejected"
                            )
                            break
                        initial_shared_schur_accepted_corrections += 1
                        scaled_cameras = corrected_scaled_cameras.copy()
                        local_cameras = np.repeat(
                            scaled_cameras[None, :, :],
                            cluster_count,
                            axis=0,
                        )
                        centers = local_cameras.copy()
                        consensus = scaled_cameras.copy()
                        landmarks = corrected_points.copy()
                        best_sse = initial_shared_schur_candidate_sse
                        best_metrics = corrected_metrics.copy()
                        best_cameras = corrected_cameras.copy()
                        best_points = corrected_points.copy()
                        accepted_metrics = corrected_metrics.copy()
                        accepted_dre = initial_shared_schur_candidate_sse
                        accepted_model_dre = initial_shared_schur_candidate_sse
                        accepted_fixed_point_squared = 0.0
                        accepted_consensus = consensus.copy()
                        accepted_landmarks = landmarks.copy()
                        damping_factor = (
                            model_ratio_damping_factor(
                                assessment["dampedGainRatio"],
                                arguments.
                                initial_shared_schur_model_ratio_minimum_factor,
                            )
                            if arguments.initial_shared_schur_damping_policy
                            == "model_ratio"
                            else arguments.shared_schur_damping_decrease
                        )
                        camera_damping *= damping_factor
                        landmark_damping *= damping_factor
                        correction += 1
                        attempt = 0
                        rejection_damping_factor = 2.0
                        if relative_decrease < (
                            arguments.shared_schur_minimum_relative_decrease
                        ):
                            initial_shared_schur_termination = (
                                "minimum_relative_decrease"
                            )
                            break
                bootstrap_basin_guard_active = (
                    arguments.initial_shared_schur_basin_guard
                )
                bootstrap_basin_guard_ceiling = best_sse
                if arguments.initial_shared_schur_rebase_trust_state:
                    initial_shared_schur_pre_rebase_trust_radii = (
                        worker.last_trust_region_radii.tolist()
                    )
                    (
                        rebase_costs,
                        rebased_scaled_cameras,
                        rebased_points,
                    ) = worker.apply_camera_step(
                        camera_indices_in_cluster,
                        point_indices_in_cluster,
                        consensus,
                        landmarks,
                        np.zeros_like(tangent_step),
                        cluster_count,
                        0,
                        rebase_trust_state=True,
                    )
                    if not (
                        np.allclose(
                            rebased_scaled_cameras,
                            consensus,
                            rtol=1e-12,
                            atol=1e-14,
                        )
                        and np.allclose(
                            rebased_points,
                            landmarks,
                            rtol=1e-12,
                            atol=1e-14,
                        )
                        and np.isfinite(np.sum(rebase_costs))
                    ):
                        raise RuntimeError(
                            "trust-state rebase changed accepted bootstrap geometry"
                        )
                    initial_shared_schur_post_rebase_trust_radii = (
                        worker.last_trust_region_radii.tolist()
                    )
                    initial_shared_schur_trust_rebased = True
            else:
                initial_shared_schur_termination = "rejected"
                worker.apply_camera_step(
                    camera_indices_in_cluster,
                    point_indices_in_cluster,
                    consensus,
                    landmarks,
                    np.zeros_like(tangent_step),
                    cluster_count,
                    0,
                )
            initial_shared_schur_seconds = (
                time.perf_counter() - initial_schur_started
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
        initial_local_trust_region_radius = float(os.environ.get(
            "BUNDLE_PALM_INITIAL_TRUST_REGION_RADIUS", "10"
        ))
        if not np.isfinite(initial_local_trust_region_radius) or initial_local_trust_region_radius <= 0.0:
            raise ValueError("initial local trust-region radius must be positive and finite")
        safeguard_annealing_iterations = (
            arguments.safeguard_annealing_iterations or arguments.iterations
        )
        for iteration in range(arguments.iterations):
            mid_shared_schur_triggered = (
                arguments.mid_shared_schur_correction_iteration > 0
                and iteration
                == arguments.mid_shared_schur_correction_iteration
            )
            mid_shared_schur_accepted_this_iteration = False
            outer_acceleration_restart_applied = (
                outer_acceleration_restart_is_active(
                    arguments.outer_acceleration_restart_iteration,
                    iteration,
                )
            )
            if outer_acceleration_restart_applied:
                accelerator.reset()
                acceleration_failures = 0
            local_state_rebase_applied = local_state_rebase_is_active(
                arguments.local_state_rebase_iteration,
                iteration,
            )
            if local_state_rebase_applied:
                block_curvature_multiplier = arguments.block_curvature_multiplier
                block_regularization = arguments.block_regularization
                accepted_since_curvature_increase = 0
                accelerator.reset()
                acceleration_failures = 0
            if mid_shared_schur_triggered:
                mid_shared_schur_attempted = True
                mid_result = run_mid_shared_schur_rebase(
                    worker,
                    arguments,
                    camera_indices_in_cluster,
                    point_indices_in_cluster,
                    accepted_consensus,
                    accepted_landmarks,
                    camera_scaling,
                    camera_count,
                    cluster_count,
                    evaluate_state,
                    camera_indices,
                    point_indices,
                    observations,
                    3 * arguments.iterations + iteration + 2,
                    local_cameras=local_cameras,
                    centers=centers,
                )
                mid_shared_schur_accepted = mid_result["accepted"]
                mid_shared_schur_accepted_this_iteration = mid_result[
                    "accepted"
                ]
                mid_shared_schur_initial_sse = mid_result[
                    "initialMetrics"
                ]["objectiveValue"]
                mid_shared_schur_candidate_sse = mid_result[
                    "candidateMetrics"
                ]["objectiveValue"]
                mid_shared_schur_worker_sse = mid_result["workerSSE"]
                mid_shared_schur_seconds = mid_result["seconds"]
                mid_shared_schur_diagnostics = mid_result["diagnostics"]
                mid_shared_schur_trust_rebased = mid_result[
                    "trustRebased"
                ]
                mid_shared_schur_product_state_transported = mid_result[
                    "productStateTransported"
                ]
                mid_shared_schur_transport_worker_sse = mid_result[
                    "transportWorkerSSE"
                ]
                mid_shared_schur_transport_offset_error = mid_result[
                    "transportOffsetError"
                ]
                mid_shared_schur_pre_rebase_trust_radii = mid_result[
                    "preRebaseTrustRadii"
                ]
                mid_shared_schur_post_rebase_trust_radii = mid_result[
                    "postRebaseTrustRadii"
                ]
                if mid_shared_schur_accepted:
                    consensus = mid_result["consensus"].copy()
                    landmarks = mid_result["landmarks"].copy()
                    if mid_shared_schur_product_state_transported:
                        local_cameras = mid_result["localCameras"].copy()
                        centers = mid_result["centers"].copy()
                    else:
                        local_cameras = np.repeat(
                            consensus[None, :, :], cluster_count, axis=0
                        )
                        centers = local_cameras.copy()
                    accepted_metrics = mid_result["candidateMetrics"].copy()
                    if mid_shared_schur_product_state_transported:
                        objective_change = (
                            mid_shared_schur_candidate_sse
                            - mid_shared_schur_initial_sse
                        )
                        accepted_dre += objective_change
                        accepted_model_dre += objective_change
                    else:
                        accepted_dre = mid_shared_schur_candidate_sse
                        accepted_model_dre = mid_shared_schur_candidate_sse
                        accepted_fixed_point_squared = 0.0
                    accepted_consensus = consensus.copy()
                    accepted_landmarks = landmarks.copy()
                    mid_state_id = 3 * arguments.iterations + iteration + 3
                    if arguments.worker_owned_landmarks:
                        worker.control_nominal_landmark_state(
                            cluster_count,
                            mid_state_id,
                            "save_accepted",
                        )
                    if mid_shared_schur_candidate_sse < best_sse:
                        best_sse = mid_shared_schur_candidate_sse
                        best_metrics = mid_result["candidateMetrics"].copy()
                        best_iteration = iteration
                        best_cameras = to_physical_cameras(
                            consensus, camera_scaling
                        ).copy()
                        best_points = landmarks.copy()
                        if arguments.worker_owned_landmarks:
                            worker.control_nominal_landmark_state(
                                cluster_count,
                                mid_state_id,
                                "save_best",
                            )
                    if not mid_shared_schur_product_state_transported:
                        block_curvature_multiplier = (
                            arguments.block_curvature_multiplier
                        )
                        block_regularization = arguments.block_regularization
                        accepted_since_curvature_increase = 0
                        revert_landmark_mode = 0
                        trust_region_recovery_ratio = 1.0
                        shared_trust_region_radius = (
                            arguments.shared_trust_region_initial_radius
                        )
                        collective_trust_region_radii = None
                        adaptive_local_steps.fill(arguments.local_steps)
                        adaptive_defect_history.clear()
                        adaptive_depth_dwell_remaining.fill(0)
                        accelerator.reset()
                        acceleration_failures = 0
            schur_alignment_tangent = None
            schur_alignment_base_cameras = None
            schur_alignment_diagnostics = None
            if iteration in schur_alignment_diagnostic_iterations:
                diagnostic_landmarks = accepted_landmarks.copy()
                if arguments.worker_owned_landmarks:
                    diagnostic_landmarks = worker.materialize_current_landmarks(
                        point_indices_in_cluster,
                        diagnostic_landmarks,
                        cluster_count,
                        arguments.iterations + iteration + 2,
                        source="accepted",
                    )
                schur_alignment_base_cameras = to_physical_cameras(
                    accepted_consensus, camera_scaling
                )
                schur_alignment_systems = worker.build_schur_systems(
                    camera_indices_in_cluster,
                    point_indices_in_cluster,
                    accepted_consensus,
                    diagnostic_landmarks,
                    cluster_count,
                    arguments.schur_alignment_landmark_damping,
                )
                (
                    schur_alignment_tangent,
                    schur_alignment_solve_diagnostics,
                ) = solve_global_schur_system(
                    schur_alignment_systems,
                    camera_count,
                    arguments.schur_alignment_camera_damping,
                    1.0,
                    arguments.shared_schur_linear_solver,
                    arguments.shared_schur_relative_tolerance,
                    arguments.schur_alignment_maximum_iterations,
                    operator_mode=arguments.shared_schur_operator,
                    preconditioner_mode=(
                        arguments.shared_schur_preconditioner
                    ),
                    coarse_basis=(
                        similarity_gauge_tangent_basis(
                            schur_alignment_base_cameras
                        )
                        if arguments.shared_schur_preconditioner
                        == "gauge_deflated"
                        else None
                    ),
                )
            shared_trust_region_active = (
                iteration < arguments.shared_trust_region_until
            )
            collective_trust_trial_active = (
                iteration < arguments.collective_trust_trial_until
            )
            iteration_shared_trust_region_radius = (
                shared_trust_region_radius
                if shared_trust_region_active else None
            )
            iteration_forced_trust_region_radius = (
                initial_local_trust_region_radius
                if local_state_rebase_applied
                else (
                    collective_trust_region_radii
                    if collective_trust_trial_active
                    and collective_trust_region_radii is not None
                    else iteration_shared_trust_region_radius
                )
            )
            shared_trust_region_log_spread = float("nan")
            diagonal_trust_until = (
                arguments.diagonal_trust_until
                or arguments.enhanced_inner_until
            )
            relative_residual_until = (
                arguments.relative_residual_until
                or arguments.enhanced_inner_until
            )
            diagonal_trust_active = diagonal_trust_is_active(
                iteration, diagonal_trust_until
            )
            relative_residual_active = iteration < relative_residual_until
            enhanced_inner_active = (
                diagonal_trust_active or relative_residual_active
            )
            iteration_metric_proposal_disagreement_scale = (
                scheduled_metric_proposal_scale(
                    arguments.metric_proposal_disagreement_scale,
                    arguments.metric_proposal_disagreement_until,
                    iteration,
                )
            )
            iteration_nesterov_maximum = (
                arguments.enhanced_inner_max_iterations
                if relative_residual_active
                else arguments.nesterov_max_iterations
            )
            iteration_local_steps = (
                adaptive_local_steps.copy()
                if arguments.adaptive_local_depth
                else arguments.local_steps
            )
            iteration_local_solver = (
                arguments.local_solver_after_switch
                if arguments.local_solver_switch_iteration > 0
                and iteration >= arguments.local_solver_switch_iteration
                else arguments.local_solver
            )
            reference_sse = accepted_metrics["objectiveValue"]
            reference_dre = accepted_dre
            proximal_penalty = penalty
            proximal_block_regularization = block_regularization
            proximal_block_curvature_multiplier = block_curvature_multiplier
            recovery_exhausted = False
            relaxed_acceptance_applied = False
            curvature_decay_applied = False
            oracle_initial_local_cameras = local_cameras.copy()
            oracle_initial_landmarks = landmarks.copy()
            oracle_input_centers = centers.copy()
            oracle_input_consensus = consensus.copy()
            oracle_calls_this_iteration = 1
            accelerated_trials = 0
            accepted_acceleration_weight = 0.0
            camera_copy_diagnostics = []
            applied_unique_camera_metric_scale = (
                current_unique_camera_metric_scale
            )
            next_unique_camera_metric_scale = (
                current_unique_camera_metric_scale
            )
            shared_camera_compatibility = float("nan")
            local_linear_iterations = np.full(cluster_count, np.nan)
            local_linear_relative_residuals = np.full(cluster_count, np.nan)
            unique_metric_selector_attempted = False
            unique_metric_selector_selected = False
            unique_metric_selector_sse = float("nan")
            unique_metric_selector_dre = float("nan")
            unique_metric_selector_rejected = False
            collective_trust_trial_radius_value = float("nan")
            collective_trust_trial_evaluated = False
            collective_trust_trial_selected = False
            collective_trust_trial_sse = float("nan")
            collective_trust_trial_dre = float("nan")
            collective_trust_trial_rejected = False
            iteration_schur_observability_diagnostic = (
                arguments.global_schur_majorizer_observability_threshold >= 0.0
                and not global_schur_majorizer_decided
            )
            iteration_schur_majorizer_active = (
                not arguments.factorized_coupled_schur_proximal_metric
                and
                global_schur_majorizer_decided
                and global_schur_majorizer_selected
                and (
                    arguments.global_schur_majorizer_until == 0
                    or iteration < arguments.global_schur_majorizer_until
                )
            )
            iteration_factorized_schur_active = (
                arguments.factorized_coupled_schur_proximal_metric
                and global_schur_majorizer_decided
                and global_schur_majorizer_selected
            )

            camera_proximal_multipliers = camera_metric_multipliers(
                camera_copy_count,
                arguments.shared_camera_metric_beta,
                applied_unique_camera_metric_scale,
            )
            if arguments.shared_only_camera_proximal:
                camera_proximal_multipliers[camera_copy_count == 1] = 0.0

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
                local_steps=iteration_local_steps,
                local_solver=iteration_local_solver,
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
                proximal_defect_diagnostic=(
                    arguments.adaptive_local_depth
                    or arguments.interior_defect_diagnostic
                ),
                landmark_refinement_steps=(
                    arguments.landmark_refinement_steps
                ),
                override_landmarks=override_landmarks,
                return_metric_blocks=(arguments.proximal_metric == "block"),
                return_metric_diagnostics=return_proximal_diagnostics,
                return_landmarks=not arguments.worker_owned_landmarks,
                worker_owned_cameras=arguments.worker_owned_cameras,
                return_consensus_rhs=arguments.worker_consensus_shadow,
                packed_request_buffers=arguments.packed_request_buffers,
                single_node_consensus=(
                    arguments.consensus_execution == "single-node"
                ),
                consensus_relaxation=arguments.relaxation,
                nesterov_max_iterations=iteration_nesterov_maximum,
                nesterov_min_iterations=arguments.nesterov_min_iterations,
                nesterov_stop_tolerance=arguments.nesterov_stop_tolerance,
                diagonal_trust_damping=diagonal_trust_active,
                nesterov_relative_residual=relative_residual_active,
                camera_proximal_multipliers=camera_proximal_multipliers,
                forced_trust_region_radius=(
                    iteration_forced_trust_region_radius
                ),
                outer_iteration=iteration,
                oracle_kind=1,
                collect_camera_diagonal_metrics=(
                    iteration in camera_diagonal_quantile_iterations
                ),
                schur_observability_diagnostic=(
                    iteration_schur_observability_diagnostic
                ),
                schur_offdiagonal_majorizer=(
                    iteration_schur_majorizer_active
                ),
                factorized_coupled_schur_metric=(
                    iteration_factorized_schur_active
                ),
            )
            local_linear_iterations = worker.last_linear_iterations.astype(
                np.float64, copy=True
            )
            local_linear_relative_residuals = (
                worker.last_linear_relative_residuals.copy()
            )
            nominal_trust_region_radii = (
                worker.last_trust_region_radii.copy()
            )
            if collective_trust_trial_active:
                collective_trust_trial_radius_value = (
                    collective_trust_trial_radius(
                        worker.last_trust_region_radii,
                        arguments.trust_region_recovery_ratio,
                    )
                )
            if shared_trust_region_active:
                returned_radii = worker.last_trust_region_radii
                valid_radii = returned_radii[
                    np.isfinite(returned_radii) & (returned_radii > 0.0)
                ]
                if valid_radii.size != cluster_count:
                    raise RuntimeError(
                        "worker omitted a synchronized trust-region radius"
                    )
                logarithms = np.log(valid_radii)
                shared_trust_region_radius = float(np.exp(np.mean(logarithms)))
                shared_trust_region_log_spread = float(
                    np.max(logarithms) - np.min(logarithms)
                )
            if arguments.worker_owned_landmarks:
                suppressed_routine_landmark_replies += cluster_count
            proximal_oracle_calls += 1
            override_landmarks = False
            worker_consensus_rhs = None
            single_node_summary = None
            if (
                arguments.proximal_metric == "block"
                and return_proximal_diagnostics
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
            if iteration_schur_observability_diagnostic:
                (
                    global_schur_majorizer_selected,
                    global_schur_observability_statistic,
                    global_schur_observability_valid_clusters,
                ) = select_global_schur_majorizer(
                    metric_diagnostics["schurObservabilityFractions"],
                    arguments.
                    global_schur_majorizer_observability_threshold,
                )
                global_schur_majorizer_decided = True
            projection_metric_blocks = (
                worker.last_consensus_metric_blocks
                if worker.last_consensus_metric_blocks is not None
                else raw_metric_blocks
            )
            compatibility_diagnostic_requested = (
                bool(camera_disagreement_diagnostic_ids)
                and iteration in camera_disagreement_diagnostic_iterations
            )
            if (
                unique_metric_selector is not None
                or compatibility_diagnostic_requested
            ):
                shared_camera_compatibility, _ = (
                    shared_camera_compatibility_ratio(
                        local_cameras,
                        centers,
                        camera_masks,
                        consensus,
                        projection_metric_blocks,
                        arguments.consensus_metric,
                    )
                )
            consensus_prior_blocks = shared_floor_prior_blocks(
                raw_metric_blocks,
                projection_metric_blocks,
                camera_masks,
                arguments.consensus_shared_floor_prior_scale,
            ) if arguments.consensus_shared_floor_prior_scale > 0.0 else None
            revert_landmark_mode = 0
            trust_region_recovery_ratio = 1.0

            selected_local_camera_step_scale = arguments.local_camera_step_scale
            selected_local_landmark_step_scale = arguments.local_camera_step_scale
            corrected_local_data_objective = None
            metric_proposal_disagreement_ratio = float("nan")
            applied_metric_proposal_disagreement_scale = 1.0
            metric_proposal_subspace_energy_fractions = np.full(3, np.nan)
            if arguments.local_camera_step_scale != 1.0:
                local_cameras[:] = centers + arguments.local_camera_step_scale * (
                    local_cameras - centers
                )
                landmarks[:] = oracle_initial_landmarks + (
                    arguments.local_camera_step_scale
                    * (landmarks - oracle_initial_landmarks)
                )
            elif arguments.local_camera_step_grid != "1":
                oracle_local_cameras = local_cameras.copy()
                oracle_landmarks = landmarks.copy()
                best_scale_sse = float("inf")
                scales = tuple(map(float, arguments.local_camera_step_grid.split(",")))
                for camera_scale in scales:
                  for landmark_scale in scales:
                    scaled_local_cameras = centers + camera_scale * (
                        oracle_local_cameras - centers
                    )
                    scaled_landmarks = oracle_initial_landmarks + landmark_scale * (
                        oracle_landmarks - oracle_initial_landmarks
                    )
                    scaled_consensus, _, _, _, _ = drs_step(
                        scaled_local_cameras,
                        centers,
                        camera_masks,
                        consensus,
                        relaxation=arguments.relaxation,
                        metric_blocks=projection_metric_blocks,
                        metric_mode=arguments.consensus_metric,
                    )
                    scaled_metrics = evaluate_state(
                        to_physical_cameras(scaled_consensus, camera_scaling),
                        scaled_landmarks,
                        camera_indices,
                        point_indices,
                        observations,
                    )
                    if scaled_metrics["objectiveValue"] < best_scale_sse:
                        best_scale_sse = scaled_metrics["objectiveValue"]
                        selected_local_camera_step_scale = camera_scale
                        selected_local_landmark_step_scale = landmark_scale
                local_cameras[:] = centers + selected_local_camera_step_scale * (
                    oracle_local_cameras - centers
                )
                landmarks[:] = oracle_initial_landmarks + (
                    selected_local_landmark_step_scale
                    * (oracle_landmarks - oracle_initial_landmarks)
                )
            elif arguments.shared_camera_step_grid != "1":
                oracle_local_cameras = local_cameras.copy()
                shared_cameras = np.sum(camera_masks, axis=0) > 1
                best_scale_sse = float("inf")
                for scale in map(float, arguments.shared_camera_step_grid.split(",")):
                    scaled_local_cameras = oracle_local_cameras.copy()
                    scaled_local_cameras[:, shared_cameras] = (
                        centers[:, shared_cameras]
                        + scale * (
                            oracle_local_cameras[:, shared_cameras]
                            - centers[:, shared_cameras]
                        )
                    )
                    scaled_consensus, _, _, _, _ = drs_step(
                        scaled_local_cameras,
                        centers,
                        camera_masks,
                        consensus,
                        relaxation=arguments.relaxation,
                        metric_blocks=projection_metric_blocks,
                        metric_mode=arguments.consensus_metric,
                    )
                    scaled_metrics = evaluate_state(
                        to_physical_cameras(scaled_consensus, camera_scaling),
                        landmarks,
                        camera_indices,
                        point_indices,
                        observations,
                    )
                    if scaled_metrics["objectiveValue"] < best_scale_sse:
                        best_scale_sse = scaled_metrics["objectiveValue"]
                        selected_local_camera_step_scale = scale
                local_cameras[:, shared_cameras] = (
                    centers[:, shared_cameras]
                    + selected_local_camera_step_scale * (
                        oracle_local_cameras[:, shared_cameras]
                        - centers[:, shared_cameras]
                    )
                )
            elif arguments.shared_camera_step_scale != 1.0:
                shared_cameras = np.sum(camera_masks, axis=0) > 1
                selected_local_camera_step_scale = arguments.shared_camera_step_scale
                local_cameras[:, shared_cameras] = (
                    centers[:, shared_cameras]
                    + selected_local_camera_step_scale * (
                        local_cameras[:, shared_cameras]
                        - centers[:, shared_cameras]
                    )
                )
            elif arguments.shared_camera_disagreement_scale != 1.0:
                damp_shared_camera_disagreement(
                    local_cameras,
                    centers,
                    camera_masks,
                    arguments.shared_camera_disagreement_scale,
                )
            elif arguments.metric_proposal_subspace_scales != "1,1,1":
                metric_proposal_subspace_energy_fractions = (
                    damp_metric_projected_camera_subspaces(
                        local_cameras,
                        camera_masks,
                        consensus,
                        projection_metric_blocks,
                        arguments.consensus_metric,
                        tuple(map(
                            float,
                            arguments.metric_proposal_subspace_scales.split(","),
                        )),
                    )
                )
                corrected_local_data_objective = worker.evaluate_consensus_sse(
                    camera_indices_in_cluster,
                    local_cameras,
                    cluster_count,
                    preserve_cameras=True,
                    packed_request_buffers=arguments.packed_request_buffers,
                )
            elif proposal_hysteresis is not None:
                (
                    metric_proposal_disagreement_ratio,
                    applied_metric_proposal_disagreement_scale,
                ) = damp_metric_projected_camera_proposals_hysteresis(
                    local_cameras,
                    centers,
                    camera_masks,
                    consensus,
                    projection_metric_blocks,
                    arguments.consensus_metric,
                    metric_proposal_hysteresis_scale,
                    *proposal_hysteresis,
                )
                corrected_local_data_objective = worker.evaluate_consensus_sse(
                    camera_indices_in_cluster,
                    local_cameras,
                    cluster_count,
                    preserve_cameras=True,
                    packed_request_buffers=arguments.packed_request_buffers,
                )
            elif arguments.metric_proposal_disagreement_grid != "1":
                (
                    local_cameras,
                    applied_metric_proposal_disagreement_scale,
                    metric_proposal_disagreement_ratio,
                    corrected_local_data_objective,
                ) = select_metric_proposal_scale(
                    local_cameras,
                    centers,
                    camera_masks,
                    consensus,
                    projection_metric_blocks,
                    arguments.consensus_metric,
                    tuple(map(
                        float,
                        arguments.metric_proposal_disagreement_grid.split(","),
                    )),
                    float(np.sum(prox_costs)),
                    lambda candidate: worker.evaluate_consensus_sse(
                        camera_indices_in_cluster,
                        candidate,
                        cluster_count,
                        preserve_cameras=True,
                        packed_request_buffers=arguments.packed_request_buffers,
                    ),
                )
            elif iteration_metric_proposal_disagreement_scale != 1.0:
                if (
                    camera_disagreement_diagnostic_ids
                    and iteration in camera_disagreement_diagnostic_iterations
                ):
                    camera_copy_diagnostics = camera_copy_disagreement_diagnostics(
                        local_cameras,
                        camera_masks,
                        consensus,
                        projection_metric_blocks,
                        arguments.consensus_metric,
                        camera_scaling,
                        camera_disagreement_diagnostic_ids,
                    )
                (
                    metric_proposal_disagreement_ratio,
                    applied_metric_proposal_disagreement_scale,
                ) = (
                    damp_metric_projected_camera_proposals(
                    local_cameras,
                    centers,
                    camera_masks,
                    consensus,
                    projection_metric_blocks,
                    arguments.consensus_metric,
                    iteration_metric_proposal_disagreement_scale,
                    (
                        arguments.metric_proposal_disagreement_threshold
                        if arguments.metric_proposal_disagreement_threshold >= 0.0
                        else None
                    ),
                    prior_blocks=consensus_prior_blocks,
                    )
                )
                corrected_local_data_objective = worker.evaluate_consensus_sse(
                    camera_indices_in_cluster,
                    local_cameras,
                    cluster_count,
                    preserve_cameras=True,
                    packed_request_buffers=arguments.packed_request_buffers,
                )

            consensus_started_at = time.perf_counter()
            reflected = None
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
                if (
                    arguments.single_cluster_proximal
                    and not arguments.shared_only_camera_proximal
                ):
                    (
                        candidate_consensus,
                        candidate_centers,
                        residuals,
                        selected_metric_blocks,
                    ) = proximal_point_step(
                        local_cameras,
                        centers,
                        camera_masks,
                        metric_blocks=projection_metric_blocks,
                    )
                else:
                    (
                        candidate_consensus,
                        candidate_centers,
                        reflected,
                        residuals,
                        selected_metric_blocks,
                    ) = drs_step(
                        local_cameras,
                        centers,
                        camera_masks,
                        consensus,
                        relaxation=arguments.relaxation,
                        metric_blocks=projection_metric_blocks,
                        metric_mode=arguments.consensus_metric,
                        prior_blocks=consensus_prior_blocks,
                        shared_only=arguments.shared_only_camera_proximal,
                    )
            consensus_projection_seconds += (
                time.perf_counter() - consensus_started_at
            )
            if arguments.worker_consensus_shadow:
                rhs_error, consensus_error = validate_worker_consensus_rhs(
                    local_cameras,
                    centers,
                    projection_metric_blocks,
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
            if schur_alignment_tangent is not None:
                consensus_tangent = left_se3_camera_minus(
                    physical_candidate, schur_alignment_base_cameras
                )
                schur_alignment_gradient = np.zeros(
                    (camera_count, 9), dtype=np.float64
                )
                schur_alignment_diagonal = np.zeros(
                    (camera_count, 9), dtype=np.float64
                )
                for system in schur_alignment_systems:
                    np.add.at(
                        schur_alignment_gradient,
                        system.camera_ids,
                        system.reduced_gradient,
                    )
                    np.add.at(
                        schur_alignment_diagonal,
                        system.camera_ids,
                        np.diagonal(
                            system.camera_diagonal, axis1=1, axis2=2
                        ),
                    )
                shared_cameras = camera_copy_count > 1
                shared_consensus_tangent = np.zeros_like(consensus_tangent)
                shared_consensus_tangent[shared_cameras] = (
                    consensus_tangent[shared_cameras]
                )
                unique_consensus_tangent = np.zeros_like(consensus_tangent)
                unique_consensus_tangent[~shared_cameras] = (
                    consensus_tangent[~shared_cameras]
                )
                shared_schur_tangent = np.zeros_like(schur_alignment_tangent)
                shared_schur_tangent[shared_cameras] = (
                    schur_alignment_tangent[shared_cameras]
                )
                if not isinstance(
                    selected_metric_blocks, ActiveCameraMetricBlocks
                ) or reflected is None or consensus_prior_blocks is not None:
                    raise RuntimeError(
                        "Schur alignment vote coherence requires unregularized "
                        "active block-metric consensus"
                    )
                vote_coherence = consensus_vote_coherence(
                    reflected,
                    accepted_consensus,
                    selected_metric_blocks.cluster_indices,
                    selected_metric_blocks.camera_indices,
                    selected_metric_blocks.blocks,
                    camera_copy_count,
                )
                active_shared_copies = shared_cameras[
                    selected_metric_blocks.camera_indices
                ]
                copy_cluster_indices = selected_metric_blocks.cluster_indices[
                    active_shared_copies
                ]
                copy_camera_indices = selected_metric_blocks.camera_indices[
                    active_shared_copies
                ]
                scaled_reflected_copies = reflected[
                    copy_cluster_indices, copy_camera_indices
                ]
                scaling = np.asarray(camera_scaling)
                copy_scaling = (
                    scaling[copy_camera_indices]
                    if scaling.ndim in (2, 3)
                    else scaling
                )
                physical_reflected_copies = to_physical_cameras(
                    scaled_reflected_copies, copy_scaling
                )
                reflected_copy_tangents = left_se3_camera_minus(
                    physical_reflected_copies,
                    schur_alignment_base_cameras[copy_camera_indices],
                )
                copy_schur_alignment = diagonal_weighted_copy_alignment(
                    schur_alignment_tangent,
                    reflected_copy_tangents,
                    schur_alignment_diagonal,
                    copy_camera_indices,
                )
                similarity_gauge_basis = similarity_gauge_tangent_basis(
                    schur_alignment_base_cameras
                )
                (
                    quotient_schur_tangent,
                    schur_gauge_diagnostics,
                ) = project_tangent_orthogonal_to_basis(
                    schur_alignment_tangent,
                    similarity_gauge_basis,
                    schur_alignment_diagonal,
                )
                (
                    quotient_consensus_tangent,
                    consensus_gauge_diagnostics,
                ) = project_tangent_orthogonal_to_basis(
                    consensus_tangent,
                    similarity_gauge_basis,
                    schur_alignment_diagonal,
                )
                schur_alignment_diagnostics = {
                    "cameraDamping": arguments.schur_alignment_camera_damping,
                    "landmarkDamping": (
                        arguments.schur_alignment_landmark_damping
                    ),
                    "allCameras": tangent_alignment(
                        schur_alignment_tangent, consensus_tangent
                    ),
                    "allCamerasDiagonalWeighted": (
                        diagonal_weighted_tangent_alignment(
                            schur_alignment_tangent,
                            consensus_tangent,
                            schur_alignment_diagonal,
                        )
                    ),
                    "sharedCameras": tangent_alignment(
                        schur_alignment_tangent[shared_cameras],
                        consensus_tangent[shared_cameras],
                    ),
                    "sharedCamerasDiagonalWeighted": (
                        diagonal_weighted_tangent_alignment(
                            schur_alignment_tangent[shared_cameras],
                            consensus_tangent[shared_cameras],
                            schur_alignment_diagonal[shared_cameras],
                        )
                    ),
                    "similarityGauge": {
                        "rank": int(similarity_gauge_basis.shape[1]),
                        "schur": schur_gauge_diagnostics,
                        "consensus": consensus_gauge_diagnostics,
                    },
                    "consensusVoteCoherence": vote_coherence,
                    "reflectedCopySchurAlignment": copy_schur_alignment,
                    "quotientAllCamerasDiagonalWeighted": (
                        diagonal_weighted_tangent_alignment(
                            quotient_schur_tangent,
                            quotient_consensus_tangent,
                            schur_alignment_diagonal,
                        )
                    ),
                    "quotientSharedCamerasDiagonalWeighted": (
                        diagonal_weighted_tangent_alignment(
                            quotient_schur_tangent[shared_cameras],
                            quotient_consensus_tangent[shared_cameras],
                            schur_alignment_diagonal[shared_cameras],
                        )
                    ),
                    "schurGradientAction": float(np.sum(
                        schur_alignment_gradient * schur_alignment_tangent
                    )),
                    "consensusGradientAction": float(np.sum(
                        schur_alignment_gradient * consensus_tangent
                    )),
                    "consensusModel": evaluate_global_schur_direction(
                        schur_alignment_systems,
                        camera_count,
                        arguments.schur_alignment_camera_damping,
                        consensus_tangent,
                    ),
                    "sharedConsensusModel": evaluate_global_schur_direction(
                        schur_alignment_systems,
                        camera_count,
                        arguments.schur_alignment_camera_damping,
                        shared_consensus_tangent,
                    ),
                    "uniqueConsensusModel": evaluate_global_schur_direction(
                        schur_alignment_systems,
                        camera_count,
                        arguments.schur_alignment_camera_damping,
                        unique_consensus_tangent,
                    ),
                    "sharedSchurModel": evaluate_global_schur_direction(
                        schur_alignment_systems,
                        camera_count,
                        arguments.schur_alignment_camera_damping,
                        shared_schur_tangent,
                    ),
                    "quotientConsensusModel": evaluate_global_schur_direction(
                        schur_alignment_systems,
                        camera_count,
                        arguments.schur_alignment_camera_damping,
                        quotient_consensus_tangent,
                    ),
                    "quotientSchurModel": evaluate_global_schur_direction(
                        schur_alignment_systems,
                        camera_count,
                        arguments.schur_alignment_camera_damping,
                        quotient_schur_tangent,
                    ),
                    "schur": schur_alignment_solve_diagnostics,
                }
                if arguments.schur_model_consensus_clipping:
                    consensus_model = schur_alignment_diagnostics[
                        "consensusModel"
                    ]
                    model_optimal_scale = float(np.clip(
                        -consensus_model["gradientAction"]
                        / max(
                            consensus_model["undampedQuadratic"],
                            np.finfo(np.float64).tiny,
                        ),
                        0.0,
                        1.0,
                    ))
                    model_scale = min(
                        model_optimal_scale,
                        arguments.schur_model_consensus_clipping_maximum_scale,
                    )
                    ordinary_worker_sse = worker.evaluate_consensus_sse(
                        camera_indices_in_cluster,
                        candidate_consensus,
                        cluster_count,
                        preserve_cameras=True,
                        packed_request_buffers=arguments.packed_request_buffers,
                    )
                    clipping_eligible = model_optimal_scale >= (
                        arguments.
                        schur_model_consensus_clipping_minimum_scale
                    )
                    clipped_worker_sse = float("nan")
                    clipping_selected = False
                    if clipping_eligible:
                        clipped_physical_candidate = left_se3_camera_plus(
                            schur_alignment_base_cameras,
                            model_scale * consensus_tangent,
                        )
                        clipped_consensus = to_scaled_cameras(
                            clipped_physical_candidate, camera_scaling
                        )
                        clipped_worker_sse = worker.evaluate_consensus_sse(
                            camera_indices_in_cluster,
                            clipped_consensus,
                            cluster_count,
                            preserve_cameras=True,
                            packed_request_buffers=(
                                arguments.packed_request_buffers
                            ),
                        )
                        clipping_selected = (
                            clipped_worker_sse < ordinary_worker_sse
                        )
                    schur_alignment_diagnostics["modelOptimalScale"] = (
                        model_optimal_scale
                    )
                    schur_alignment_diagnostics["modelScale"] = model_scale
                    schur_alignment_diagnostics["clippingEligible"] = (
                        clipping_eligible
                    )
                    schur_alignment_diagnostics["ordinaryWorkerSSE"] = (
                        ordinary_worker_sse
                    )
                    schur_alignment_diagnostics["clippedWorkerSSE"] = (
                        clipped_worker_sse
                    )
                    schur_alignment_diagnostics["clippingSelected"] = (
                        clipping_selected
                    )
                    if clipping_selected:
                        candidate_consensus = clipped_consensus
                        physical_candidate = clipped_physical_candidate
                        candidate_centers, residuals = drs_state_for_consensus(
                            local_cameras,
                            centers,
                            camera_masks,
                            candidate_consensus,
                            arguments.relaxation,
                            selected_metric_blocks,
                            shared_only=arguments.shared_only_camera_proximal,
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
                    "sumSquaredError": worker.last_consensus_l2_sse,
                    "objectiveValue": worker_sse,
                    "meanReprojectionError": float("nan"),
                }
                if huber_delta is not None:
                    unrefined_candidate_metrics["huberDelta"] = huber_delta
                    unrefined_candidate_metrics["huberCeresCost"] = 0.5 * worker_sse
            else:
                unrefined_candidate_metrics = evaluate_state(
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
                    - unrefined_candidate_metrics["objectiveValue"]
                ) / max(
                    abs(unrefined_candidate_metrics["objectiveValue"]),
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
                refined_candidate_metrics = evaluate_state(
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
            candidate_sse = candidate_metrics["objectiveValue"]
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
                local_data_objective = (
                    corrected_local_data_objective
                    if corrected_local_data_objective is not None
                    else float(np.sum(prox_costs))
                )
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
                    shared_only=arguments.shared_only_camera_proximal,
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
                min(iteration, safeguard_annealing_iterations - 1),
                safeguard_annealing_iterations,
                dre_increase_at_reference=arguments.dre_relative_increase,
                reference_iteration=arguments.safeguard_reference_iteration,
                annealing_exponent=arguments.safeguard_annealing_exponent,
                minimum_primal_ratio=arguments.minimum_primal_ratio,
            )
            dre_threshold_exceeded = (
                not np.isfinite(douglas_rachford_envelope)
                or exceeds_with_relative_deadband(
                    douglas_rachford_envelope,
                    dre_ratio * reference_dre,
                    arguments.safeguard_relative_deadband,
                )
            )
            primal_threshold_exceeded = (
                not np.isfinite(candidate_sse)
                or exceeds_with_relative_deadband(
                    candidate_sse,
                    primal_ratio * reference_sse,
                    arguments.safeguard_relative_deadband,
                )
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
                    arguments.safeguard_relative_deadband,
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
            if (
                rejected
                and relaxed_exploration_remaining > 0
                and np.isfinite(douglas_rachford_envelope)
                and np.isfinite(candidate_sse)
            ):
                rejected = False
                relaxed_exploration_remaining -= 1
                relaxed_acceptance_applied = True

            nominal_trial = {
                "trial_kind": "nominal",
                "local_cameras": local_cameras.copy(),
                "landmarks": landmarks.copy(),
                "prox_costs": prox_costs.copy(),
                "raw_metric_blocks": (
                    raw_metric_blocks.copy()
                    if raw_metric_blocks is not None else None
                ),
                "projection_metric_blocks": (
                    projection_metric_blocks.copy()
                    if projection_metric_blocks is not None else None
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
                "metric_proposal_disagreement_ratio": (
                    metric_proposal_disagreement_ratio
                ),
                "applied_metric_proposal_disagreement_scale": (
                    applied_metric_proposal_disagreement_scale
                ),
                "metric_proposal_subspace_energy_fractions": (
                    metric_proposal_subspace_energy_fractions.copy()
                ),
                "trust_region_radii": nominal_trust_region_radii,
            }
            iteration_outer_acceleration_active = (
                scheduled_outer_acceleration_active(
                    arguments.outer_acceleration_until,
                    iteration,
                )
            )
            if iteration_outer_acceleration_active:
                acceleration_proposal, proposal_is_accelerated = accelerator.propose(
                    oracle_input_centers,
                    candidate_centers,
                    iteration,
                )
            else:
                acceleration_proposal = candidate_centers.copy()
                proposal_is_accelerated = False
            selected_trial = nominal_trial
            evaluated_accelerated_trial = False
            evaluated_collective_trust_trial = False
            selector_triggered = (
                unique_metric_selector is not None
                and np.isfinite(shared_camera_compatibility)
                and shared_camera_compatibility <= unique_metric_selector[1]
            )
            if (
                proposal_is_accelerated
                or (
                    iteration_outer_acceleration_active
                    and accelerator.requires_first_trial_observation
                )
                or selector_triggered
                or collective_trust_trial_active
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
                trial_specs = []
                if collective_trust_trial_active:
                    trial_specs.append((
                        "collective_trust",
                        0.0,
                        oracle_input_centers.copy(),
                        camera_proximal_multipliers,
                        collective_trust_trial_radius_value,
                    ))
                if selector_triggered:
                    trial_specs.append((
                        "unique_metric",
                        0.0,
                        oracle_input_centers.copy(),
                        camera_metric_multipliers(
                            camera_copy_count,
                            arguments.shared_camera_metric_beta,
                            unique_metric_selector[0],
                        ),
                        iteration_forced_trust_region_radius,
                    ))
                if (
                    proposal_is_accelerated
                    or (
                        iteration_outer_acceleration_active
                        and accelerator.requires_first_trial_observation
                    )
                ):
                    for acceleration_weight in line_search_weights:
                        if acceleration_weight != 0.0:
                            trial_specs.append((
                                "acceleration",
                                acceleration_weight,
                                interpolate_line_search_center(
                                    candidate_centers,
                                    acceleration_proposal,
                                    acceleration_weight,
                                ),
                                camera_proximal_multipliers,
                                iteration_forced_trust_region_radius,
                            ))
                for (
                    trial_kind,
                    acceleration_weight,
                    trial_centers,
                    trial_camera_multipliers,
                    trial_forced_trust_region_radius,
                ) in trial_specs:
                    evaluated_accelerated_trial |= (
                        trial_kind == "acceleration"
                    )
                    unique_metric_selector_attempted |= (
                        trial_kind == "unique_metric"
                    )
                    evaluated_collective_trust_trial |= (
                        trial_kind == "collective_trust"
                    )
                    collective_trust_trial_evaluated |= (
                        trial_kind == "collective_trust"
                    )
                    accelerated_trials += trial_kind == "acceleration"
                    oracle_calls_this_iteration += 1
                    proximal_oracle_calls += 1
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
                        local_steps=iteration_local_steps,
                        local_solver=iteration_local_solver,
                        trust_region_policy=arguments.trust_region_policy,
                        camera_scaling=camera_scaling,
                        revert_landmarks=(
                            2 if trial_kind == "unique_metric" else 1
                        ),
                        persistent_trust_region=arguments.persistent_trust_region,
                        trust_region_recovery_ratio=1.0,
                        scalar_proximal_prior=(arguments.proximal_metric == "scalar"),
                        block_regularization=proximal_block_regularization,
                        block_curvature_multiplier=proximal_block_curvature_multiplier,
                        metric_diagnostic_iterations=arguments.metric_diagnostic_iterations,
                        proximal_defect_diagnostic=(
                            arguments.adaptive_local_depth
                            or arguments.interior_defect_diagnostic
                        ),
                        landmark_refinement_steps=arguments.landmark_refinement_steps,
                        return_metric_blocks=(arguments.proximal_metric == "block"),
                        return_metric_diagnostics=return_proximal_diagnostics,
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
                            iteration_nesterov_maximum
                        ),
                        nesterov_min_iterations=(
                            arguments.nesterov_min_iterations
                        ),
                        nesterov_stop_tolerance=(
                            arguments.nesterov_stop_tolerance
                        ),
                        diagonal_trust_damping=diagonal_trust_active,
                        nesterov_relative_residual=relative_residual_active,
                        camera_proximal_multipliers=trial_camera_multipliers,
                        forced_trust_region_radius=(
                            trial_forced_trust_region_radius
                        ),
                        outer_iteration=iteration,
                        oracle_kind=2,
                        collect_camera_diagonal_metrics=False,
                        schur_observability_diagnostic=(
                            iteration_schur_observability_diagnostic
                        ),
                        schur_offdiagonal_majorizer=(
                            iteration_schur_majorizer_active
                        ),
                        factorized_coupled_schur_metric=(
                            iteration_factorized_schur_active
                        ),
                    )
                    trial_trust_region_radii = (
                        worker.last_trust_region_radii.copy()
                    )
                    trial_projection_blocks = (
                        worker.last_consensus_metric_blocks
                        if worker.last_consensus_metric_blocks is not None
                        else None
                    )
                    if (
                        arguments.suppress_accelerated_landmark_replies
                        or arguments.worker_owned_landmarks
                    ):
                        suppressed_accelerated_landmark_replies += cluster_count
                    if (
                        arguments.proximal_metric == "block"
                        and return_proximal_diagnostics
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
                    if trial_projection_blocks is None:
                        trial_projection_blocks = trial_raw_blocks
                    trial_prior_blocks = shared_floor_prior_blocks(
                        trial_raw_blocks,
                        trial_projection_blocks,
                        camera_masks,
                        arguments.consensus_shared_floor_prior_scale,
                    ) if arguments.consensus_shared_floor_prior_scale > 0.0 else None
                    trial_corrected_local_data_objective = None
                    trial_metric_proposal_disagreement_ratio = float("nan")
                    trial_applied_metric_proposal_disagreement_scale = 1.0
                    trial_metric_proposal_subspace_energy_fractions = np.full(
                        3, np.nan
                    )
                    if arguments.shared_camera_step_scale != 1.0:
                        shared_cameras = np.sum(camera_masks, axis=0) > 1
                        trial_local_cameras[:, shared_cameras] = (
                            trial_centers[:, shared_cameras]
                            + arguments.shared_camera_step_scale * (
                                trial_local_cameras[:, shared_cameras]
                                - trial_centers[:, shared_cameras]
                            )
                        )
                    elif arguments.shared_camera_disagreement_scale != 1.0:
                        damp_shared_camera_disagreement(
                            trial_local_cameras,
                            trial_centers,
                            camera_masks,
                            arguments.shared_camera_disagreement_scale,
                        )
                    elif arguments.metric_proposal_subspace_scales != "1,1,1":
                        trial_metric_proposal_subspace_energy_fractions = (
                            damp_metric_projected_camera_subspaces(
                                trial_local_cameras,
                                camera_masks,
                                consensus,
                                trial_raw_blocks,
                                arguments.consensus_metric,
                                tuple(map(
                                    float,
                                    arguments.metric_proposal_subspace_scales.split(","),
                                )),
                            )
                        )
                        trial_corrected_local_data_objective = (
                            worker.evaluate_consensus_sse(
                                camera_indices_in_cluster,
                                trial_local_cameras,
                                cluster_count,
                                preserve_cameras=True,
                                packed_request_buffers=(
                                    arguments.packed_request_buffers
                                ),
                            )
                        )
                    elif proposal_hysteresis is not None:
                        (
                            trial_metric_proposal_disagreement_ratio,
                            trial_applied_metric_proposal_disagreement_scale,
                        ) = damp_metric_projected_camera_proposals_hysteresis(
                            trial_local_cameras,
                            trial_centers,
                            camera_masks,
                            consensus,
                            trial_projection_blocks,
                            arguments.consensus_metric,
                            metric_proposal_hysteresis_scale,
                            *proposal_hysteresis,
                        )
                        trial_corrected_local_data_objective = (
                            worker.evaluate_consensus_sse(
                                camera_indices_in_cluster,
                                trial_local_cameras,
                                cluster_count,
                                preserve_cameras=True,
                                packed_request_buffers=(
                                    arguments.packed_request_buffers
                                ),
                            )
                        )
                    elif arguments.metric_proposal_disagreement_grid != "1":
                        (
                            trial_local_cameras,
                            trial_applied_metric_proposal_disagreement_scale,
                            trial_metric_proposal_disagreement_ratio,
                            trial_corrected_local_data_objective,
                        ) = select_metric_proposal_scale(
                            trial_local_cameras,
                            trial_centers,
                            camera_masks,
                            consensus,
                            trial_projection_blocks,
                            arguments.consensus_metric,
                            tuple(map(
                                float,
                                arguments.metric_proposal_disagreement_grid.split(","),
                            )),
                            float(np.sum(trial_prox_costs)),
                            lambda candidate: worker.evaluate_consensus_sse(
                                camera_indices_in_cluster,
                                candidate,
                                cluster_count,
                                preserve_cameras=True,
                                packed_request_buffers=(
                                    arguments.packed_request_buffers
                                ),
                            ),
                        )
                    elif iteration_metric_proposal_disagreement_scale != 1.0:
                        (
                            trial_metric_proposal_disagreement_ratio,
                            trial_applied_metric_proposal_disagreement_scale,
                        ) = (
                            damp_metric_projected_camera_proposals(
                            trial_local_cameras,
                            trial_centers,
                            camera_masks,
                            consensus,
                            trial_projection_blocks,
                            arguments.consensus_metric,
                            iteration_metric_proposal_disagreement_scale,
                            (
                                arguments.metric_proposal_disagreement_threshold
                                if arguments.metric_proposal_disagreement_threshold >= 0.0
                                else None
                            ),
                            prior_blocks=trial_prior_blocks,
                            )
                        )
                        trial_corrected_local_data_objective = (
                            worker.evaluate_consensus_sse(
                                camera_indices_in_cluster,
                                trial_local_cameras,
                                cluster_count,
                                preserve_cameras=True,
                                packed_request_buffers=(
                                    arguments.packed_request_buffers
                                ),
                            )
                        )
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
                            metric_blocks=trial_projection_blocks,
                            metric_mode=arguments.consensus_metric,
                            prior_blocks=trial_prior_blocks,
                            shared_only=arguments.shared_only_camera_proximal,
                        )
                    consensus_projection_seconds += (
                        time.perf_counter() - consensus_started_at
                    )
                    if trial_kind == "acceleration" and acceleration_weight == 1.0:
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
                        trial_metrics = {
                            "sumSquaredError": worker.last_consensus_l2_sse,
                            "objectiveValue": worker_sse,
                            "meanReprojectionError": float("nan"),
                        }
                        if huber_delta is not None:
                            trial_metrics["huberDelta"] = huber_delta
                            trial_metrics["huberCeresCost"] = 0.5 * worker_sse
                    else:
                        trial_metrics = evaluate_state(
                            to_physical_cameras(trial_consensus, camera_scaling),
                            trial_landmarks,
                            camera_indices,
                            point_indices,
                            observations,
                        )
                        trial_sse = trial_metrics["objectiveValue"]
                    if arguments.worker_sse_shadow and worker_sse is None:
                        worker_sse = worker.evaluate_consensus_sse(
                            camera_indices_in_cluster,
                            trial_consensus,
                            cluster_count,
                            preserve_cameras=arguments.worker_owned_cameras,
                            packed_request_buffers=arguments.packed_request_buffers,
                        )
                        worker_sse_relative_error = abs(
                            worker_sse - trial_metrics["objectiveValue"]
                        ) / max(
                            abs(trial_metrics["objectiveValue"]),
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
                        trial_local_objective = (
                            trial_corrected_local_data_objective
                            if trial_corrected_local_data_objective is not None
                            else float(np.sum(trial_prox_costs))
                        )
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
                            shared_only=arguments.shared_only_camera_proximal,
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
                            arguments.safeguard_relative_deadband,
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
                                and exceeds_with_relative_deadband(
                                    trial_dre - error_bound,
                                    dre_ratio * reference_dre,
                                    arguments.safeguard_relative_deadband,
                                )
                                and exceeds_with_relative_deadband(
                                    trial_sse - error_bound,
                                    primal_ratio * reference_sse,
                                    arguments.safeguard_relative_deadband,
                                )
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
                            trial_metrics = evaluate_state(
                                to_physical_cameras(
                                    trial_consensus, camera_scaling
                                ),
                                trial_landmarks,
                                camera_indices,
                                point_indices,
                                observations,
                            )
                            trial_sse = trial_metrics["objectiveValue"]
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
                                    arguments.safeguard_relative_deadband,
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
                    if trial_kind == "unique_metric":
                        unique_metric_selector_sse = trial_sse
                        unique_metric_selector_dre = trial_dre
                        unique_metric_selector_rejected = trial_rejected
                    elif trial_kind == "collective_trust":
                        collective_trust_trial_sse = trial_sse
                        collective_trust_trial_dre = trial_dre
                        collective_trust_trial_rejected = trial_rejected
                    trial_is_better = prefer_metric_selector_trial(
                        nominal_trial["rejected"],
                        nominal_trial["douglas_rachford_envelope"],
                        nominal_trial["candidate_sse"],
                        trial_rejected,
                        trial_dre,
                        trial_sse,
                    )
                    if trial_kind == "collective_trust":
                        trial_is_better = prefer_collective_trust_trial(
                            nominal_trial["rejected"],
                            nominal_trial["candidate_sse"],
                            trial_rejected,
                            trial_sse,
                        )
                    if not trial_rejected and (
                        trial_kind == "acceleration" or trial_is_better
                    ):
                        selected_trial = {
                            "trial_kind": trial_kind,
                            "local_cameras": trial_local_cameras,
                            "landmarks": trial_landmarks,
                            "prox_costs": trial_prox_costs,
                            "raw_metric_blocks": trial_raw_blocks,
                            "projection_metric_blocks": trial_projection_blocks,
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
                            "metric_proposal_disagreement_ratio": (
                                trial_metric_proposal_disagreement_ratio
                            ),
                            "applied_metric_proposal_disagreement_scale": (
                                trial_applied_metric_proposal_disagreement_scale
                            ),
                            "metric_proposal_subspace_energy_fractions": (
                                trial_metric_proposal_subspace_energy_fractions
                            ),
                            "trust_region_radii": trial_trust_region_radii,
                        }
                        if trial_kind == "unique_metric":
                            unique_metric_selector_selected = True
                        elif trial_kind == "acceleration":
                            accepted_acceleration_weight = acceleration_weight
                        else:
                            collective_trust_trial_selected = True
                        break

            if selected_trial is not nominal_trial:
                local_cameras = selected_trial["local_cameras"]
                landmarks = selected_trial["landmarks"]
                prox_costs = selected_trial["prox_costs"]
                raw_metric_blocks = selected_trial["raw_metric_blocks"]
                projection_metric_blocks = selected_trial[
                    "projection_metric_blocks"
                ]
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
                metric_proposal_disagreement_ratio = selected_trial[
                    "metric_proposal_disagreement_ratio"
                ]
                applied_metric_proposal_disagreement_scale = selected_trial[
                    "applied_metric_proposal_disagreement_scale"
                ]
                metric_proposal_subspace_energy_fractions = selected_trial[
                    "metric_proposal_subspace_energy_fractions"
                ]
                rejected = False
                if selected_trial["trial_kind"] == "acceleration":
                    accelerated_acceptances += 1
                    acceleration_failures = 0
                    accelerator.accepted(True)
            else:
                local_cameras = nominal_trial["local_cameras"]
                landmarks = nominal_trial["landmarks"]
                if (
                    evaluated_accelerated_trial
                    or unique_metric_selector_attempted
                    or evaluated_collective_trust_trial
                ) and not rejected:
                    override_landmarks = not arguments.worker_owned_landmarks
                    if evaluated_accelerated_trial:
                        nominal_fallbacks += 1
                        acceleration_failures += 1
                        accelerator.accepted(False)
                        if acceleration_failures >= arguments.acceleration_restart_after:
                            accelerator.reset()
                            acceleration_failures = 0
                elif rejected:
                    accelerator.reset()
                    acceleration_failures = 0

            if collective_trust_trial_active:
                collective_trust_region_radii = selected_trial[
                    "trust_region_radii"
                ].copy()

            proximal_point_sse = float("nan")
            if (
                (cluster_count == 1 or arguments.single_cluster_proximal)
                and not arguments.worker_owned_cameras
                and not arguments.worker_owned_landmarks
            ):
                proximal_point_sse = evaluate_state(
                    to_physical_cameras(local_cameras[0], camera_scaling),
                    landmarks,
                    camera_indices,
                    point_indices,
                    observations,
                )["objectiveValue"]

            if (
                (
                    evaluated_accelerated_trial
                    or evaluated_collective_trust_trial
                )
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
            bootstrap_basin_guard_triggered = False
            bootstrap_basin_guard_released = False
            if bootstrap_basin_guard_active:
                guard_rejected, guard_released = (
                    bootstrap_basin_guard_decision(
                        candidate_sse,
                        bootstrap_basin_guard_ceiling,
                        True,
                    )
                )
                if guard_rejected:
                    rejected = True
                    bootstrap_basin_guard_triggered = True
                    bootstrap_basin_guard_rejections += 1
                elif guard_released and not rejected:
                    bootstrap_basin_guard_active = False
                    bootstrap_basin_guard_released = True
                    bootstrap_basin_guard_release_iteration = iteration
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
                if (
                    recovery_exhausted
                    and arguments.recovery_exhaustion_policy
                    == "restart_best_relaxed"
                ):
                    best_scaled_cameras = to_scaled_cameras(
                        best_cameras, camera_scaling
                    )
                    local_cameras, centers, consensus = reset_to_consensus(
                        best_scaled_cameras, cluster_count
                    )
                    landmarks = best_points.copy()
                    accepted_consensus = consensus.copy()
                    accepted_landmarks = landmarks.copy()
                    accepted_metrics = best_metrics.copy()
                    metrics = accepted_metrics
                    accepted_dre = (
                        douglas_rachford_envelope
                        if np.isfinite(douglas_rachford_envelope)
                        else best_sse
                    )
                    accepted_model_dre = accepted_dre
                    accepted_fixed_point_squared = 0.0
                    if arguments.worker_owned_landmarks:
                        worker.control_nominal_landmark_state(
                            cluster_count, iteration + 1, "restore_best"
                        )
                        worker.control_nominal_landmark_state(
                            cluster_count, iteration + 1, "save_accepted"
                        )
                    revert_landmark_mode = (
                        3 if arguments.worker_owned_landmarks else 2
                    )
                    accelerator.reset()
                    acceleration_failures = 0
                    accepted_since_curvature_increase = 0
                    relaxed_exploration_remaining = (
                        arguments.recovery_relaxed_iterations
                    )
                    best_checkpoint_restarts += 1
                    recovery_exhausted = False
                    recovery_action = "best_checkpoint_restart"
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
                if proposal_hysteresis is not None:
                    metric_proposal_hysteresis_scale = (
                        applied_metric_proposal_disagreement_scale
                    )
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
            if metric_diagnostics is not None and np.any(np.isfinite(
                metric_diagnostics["interiorDefectSquared"]
            )):
                unique_camera_interior_defect_squared = float(np.nansum(
                    metric_diagnostics["uniqueCameraInteriorDefectSquared"]
                ))
                landmark_interior_defect_squared = float(np.nansum(
                    metric_diagnostics["landmarkInteriorDefectSquared"]
                ))
                interior_defect_squared = float(np.nansum(
                    metric_diagnostics["interiorDefectSquared"]
                ))
                interior_defect_maximum_squared = float(np.nanmax(
                    metric_diagnostics["interiorDefectSquared"]
                ))
            else:
                unique_camera_interior_defect_squared = float("nan")
                landmark_interior_defect_squared = float("nan")
                interior_defect_squared = float("nan")
                interior_defect_maximum_squared = float("nan")
            if metric_diagnostics is not None:
                interior_defect_objective_ratio_by_cluster = np.divide(
                    metric_diagnostics["interiorDefectSquared"],
                    np.maximum(prox_costs, np.finfo(np.float64).tiny),
                )
            else:
                interior_defect_objective_ratio_by_cluster = np.full(
                    cluster_count, np.nan
                )

            adaptive_rolling_defect_ratio = float("nan")
            adaptive_rolling_defect_ratio_by_cluster = np.full(
                cluster_count, np.nan
            )
            next_local_steps = (
                iteration_local_steps.copy()
                if arguments.adaptive_local_depth
                else iteration_local_steps
            )
            adaptive_depth_changed = False
            if (
                arguments.adaptive_local_depth
                and iteration >= arguments.adaptive_local_depth_start
                and not rejected
                and np.any(np.isfinite(
                    interior_defect_objective_ratio_by_cluster
                ))
            ):
                adaptive_defect_history.append(
                    interior_defect_objective_ratio_by_cluster.copy()
                )
                adaptive_defect_history = adaptive_defect_history[
                    -arguments.adaptive_local_depth_window:
                ]
                adaptive_depth_dwell_remaining = np.maximum(
                    adaptive_depth_dwell_remaining - 1, 0
                )
                if len(adaptive_defect_history) == (
                    arguments.adaptive_local_depth_window
                ):
                    adaptive_rolling_defect_ratio_by_cluster = np.nanmedian(
                        np.stack(adaptive_defect_history), axis=0
                    )
                    finite_rolling = np.isfinite(
                        adaptive_rolling_defect_ratio_by_cluster
                    )
                    increase_depth = (
                        (iteration_local_steps == arguments.local_steps)
                        & finite_rolling
                        & (adaptive_rolling_defect_ratio_by_cluster
                           > arguments.adaptive_local_depth_high)
                    )
                    decrease_depth = (
                        (iteration_local_steps
                         == arguments.adaptive_local_depth_maximum)
                        & (adaptive_depth_dwell_remaining == 0)
                        & finite_rolling
                        & (adaptive_rolling_defect_ratio_by_cluster
                           < arguments.adaptive_local_depth_low)
                    )
                    next_local_steps[increase_depth] = (
                        arguments.adaptive_local_depth_maximum
                    )
                    next_local_steps[decrease_depth] = arguments.local_steps
                    changed_clusters = increase_depth | decrease_depth
                    adaptive_depth_changed = bool(np.any(changed_clusters))
                    adaptive_depth_dwell_remaining[changed_clusters] = (
                        arguments.adaptive_local_depth_dwell
                    )
                    adaptive_rolling_defect_ratio = float(np.nanmax(
                        adaptive_rolling_defect_ratio_by_cluster
                    ))
            if arguments.adaptive_local_depth:
                adaptive_local_steps = next_local_steps.copy()


            row = {
                "iteration": iteration,
                "localCameraStepScale": selected_local_camera_step_scale,
                "localLandmarkStepScale": selected_local_landmark_step_scale,
                "overallSeconds": time.perf_counter() - started_at,
                "optimizationSeconds": (
                    time.perf_counter() - optimization_started_at
                ),
                "sumSquaredError": metrics["sumSquaredError"],
                "candidateSumSquaredError": candidate_sse,
                "proximalPointSumSquaredError": proximal_point_sse,
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
                "metricProposalDisagreementRatio": (
                    metric_proposal_disagreement_ratio
                ),
                "appliedMetricProposalDisagreementScale": (
                    applied_metric_proposal_disagreement_scale
                ),
                "metricProposalRotationEnergyFraction": (
                    metric_proposal_subspace_energy_fractions[0]
                ),
                "metricProposalTranslationEnergyFraction": (
                    metric_proposal_subspace_energy_fractions[1]
                ),
                "metricProposalIntrinsicsEnergyFraction": (
                    metric_proposal_subspace_energy_fractions[2]
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
                "bootstrapBasinGuardActive": bootstrap_basin_guard_active,
                "bootstrapBasinGuardTriggered": (
                    bootstrap_basin_guard_triggered
                ),
                "bootstrapBasinGuardReleased": (
                    bootstrap_basin_guard_released
                ),
                "bootstrapBasinGuardCeiling": bootstrap_basin_guard_ceiling,
                "recoveryAction": recovery_action,
                "relaxedAcceptanceApplied": relaxed_acceptance_applied,
                "relaxedExplorationRemaining": relaxed_exploration_remaining,
                "bestCheckpointRestarts": best_checkpoint_restarts,
                "cameraCopyDiagnostics": camera_copy_diagnostics,
                "sharedCameraCompatibilityRatio": (
                    shared_camera_compatibility
                ),
                "schurAlignmentDiagnostics": schur_alignment_diagnostics,
                "uniqueCameraMetricScale": (
                    applied_unique_camera_metric_scale
                ),
                "nextUniqueCameraMetricScale": (
                    next_unique_camera_metric_scale
                ),
                "uniqueMetricSelectorAttempted": (
                    unique_metric_selector_attempted
                ),
                "uniqueMetricSelectorSelected": (
                    unique_metric_selector_selected
                ),
                "uniqueMetricSelectorSSE": unique_metric_selector_sse,
                "uniqueMetricSelectorDRE": unique_metric_selector_dre,
                "uniqueMetricSelectorRejected": (
                    unique_metric_selector_rejected
                ),
                "collectiveTrustTrialActive": collective_trust_trial_active,
                "collectiveTrustTrialEvaluated": (
                    collective_trust_trial_evaluated
                ),
                "collectiveTrustTrialSelected": (
                    collective_trust_trial_selected
                ),
                "collectiveTrustTrialRadius": (
                    collective_trust_trial_radius_value
                ),
                "collectiveTrustTrialSSE": collective_trust_trial_sse,
                "collectiveTrustTrialDRE": collective_trust_trial_dre,
                "collectiveTrustTrialRejected": (
                    collective_trust_trial_rejected
                ),
                "outerAcceleration": arguments.outer_acceleration,
                "outerAccelerationRestartApplied": (
                    outer_acceleration_restart_applied
                ),
                "localStateRebaseApplied": local_state_rebase_applied,
                "midSharedSchurTriggered": mid_shared_schur_triggered,
                "midSharedSchurAccepted": (
                    mid_shared_schur_accepted_this_iteration
                ),
                "midSharedSchurProductStateTransported": (
                    mid_shared_schur_product_state_transported
                    and mid_shared_schur_accepted_this_iteration
                ),
                "forcedLocalTrustRegionRadius": (
                    float(np.exp(np.mean(np.log(
                        np.asarray(
                            iteration_forced_trust_region_radius,
                            dtype=np.float64,
                        )
                    ))))
                    if iteration_forced_trust_region_radius is not None
                    else float("nan")
                ),
                "enhancedInnerActive": enhanced_inner_active,
                "diagonalTrustActive": diagonal_trust_active,
                "relativeResidualActive": relative_residual_active,
                "sharedTrustRegionActive": shared_trust_region_active,
                "sharedTrustRegionRadius": (
                    iteration_shared_trust_region_radius
                    if shared_trust_region_active else float("nan")
                ),
                "nextSharedTrustRegionRadius": (
                    shared_trust_region_radius
                    if shared_trust_region_active else float("nan")
                ),
                "sharedTrustRegionLogSpread": (
                    shared_trust_region_log_spread
                ),
                "localTrustRegionRadiusMinimum": float(np.nanmin(
                    nominal_trust_region_radii
                )),
                "localTrustRegionRadiusGeometricMean": float(np.exp(
                    np.nanmean(np.log(nominal_trust_region_radii))
                )),
                "localTrustRegionRadiusMaximum": float(np.nanmax(
                    nominal_trust_region_radii
                )),
                "nesterovMaximumUsed": iteration_nesterov_maximum,
                "localLinearIterationsMinimum": float(np.nanmin(
                    local_linear_iterations
                )),
                "localLinearIterationsMedian": float(np.nanmedian(
                    local_linear_iterations
                )),
                "localLinearIterationsMaximum": float(np.nanmax(
                    local_linear_iterations
                )),
                "localLinearRelativeResidualMinimum": float(np.nanmin(
                    local_linear_relative_residuals
                )),
                "localLinearRelativeResidualMedian": float(np.nanmedian(
                    local_linear_relative_residuals
                )),
                "localLinearRelativeResidualMaximum": float(np.nanmax(
                    local_linear_relative_residuals
                )),
                "lineSearchGrid": arguments.line_search_grid,
                "acceptedAccelerationWeight": accepted_acceleration_weight,
                "acceleratedTrials": accelerated_trials,
                "oracleCallsThisIteration": oracle_calls_this_iteration,
                "proximalOracleCalls": proximal_oracle_calls,
                "acceleratedAcceptances": accelerated_acceptances,
                "nominalFallbacks": nominal_fallbacks,
                "accelerationStepLimitHits": accelerator.step_limit_hits,
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
                "schurObservabilityFractions": (
                    metric_diagnostics[
                        "schurObservabilityFractions"
                    ].tolist()
                    if metric_diagnostics is not None
                    else []
                ),
                "globalSchurMajorizerDecisionMade": (
                    iteration_schur_observability_diagnostic
                ),
                "globalSchurMajorizerSelected": (
                    global_schur_majorizer_selected
                ),
                "globalSchurMajorizerActive": (
                    iteration_schur_majorizer_active
                ),
                "globalSchurObservabilityStatistic": (
                    global_schur_observability_statistic
                ),
                "globalSchurObservabilityValidClusters": (
                    global_schur_observability_valid_clusters
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
                "uniqueCameraInteriorDefectSquared": (
                    unique_camera_interior_defect_squared
                ),
                "landmarkInteriorDefectSquared": (
                    landmark_interior_defect_squared
                ),
                "interiorDefectSquared": interior_defect_squared,
                "interiorDefectMaximumSquared": (
                    interior_defect_maximum_squared
                ),
                "interiorDefectSquaredByCluster": (
                    metric_diagnostics["interiorDefectSquared"].tolist()
                    if metric_diagnostics is not None else []
                ),
                "localProximalObjectiveByCluster": prox_costs.tolist(),
                "interiorDefectObjectiveRatioByCluster": (
                    interior_defect_objective_ratio_by_cluster.tolist()
                ),
                "localStepsUsed": (
                    iteration_local_steps.tolist()
                    if arguments.adaptive_local_depth
                    else iteration_local_steps
                ),
                "localSolverUsed": iteration_local_solver,
                "nextLocalSteps": (
                    next_local_steps.tolist()
                    if arguments.adaptive_local_depth
                    else next_local_steps
                ),
                "adaptiveDefectRollingMedian": (
                    adaptive_rolling_defect_ratio
                ),
                "adaptiveDefectRollingMedianByCluster": (
                    adaptive_rolling_defect_ratio_by_cluster.tolist()
                ),
                "adaptiveDepthChanged": adaptive_depth_changed,
                "adaptiveDepthDwellRemaining": (
                    int(np.max(adaptive_depth_dwell_remaining))
                ),
                "adaptiveDepthDwellRemainingByCluster": (
                    adaptive_depth_dwell_remaining.tolist()
                ),
                "nextBlockRegularization": block_regularization,
                "transportBytesSent": worker.sent_bytes,
                "transportBytesReceived": worker.received_bytes,
            }
            trajectory.append(row)
            if (
                not rejected
                and reporting_candidate_metrics["objectiveValue"] < best_sse
            ):
                best_sse = reporting_candidate_metrics["objectiveValue"]
                best_metrics = reporting_candidate_metrics.copy()
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
            if (
                arguments.stop_after_iteration > 0
                and iteration + 1 >= arguments.stop_after_iteration
            ):
                termination_reason = "configured_iteration_stop"
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
        if arguments.final_shared_schur_correction:
            final_shared_schur_attempted = True
            final_shared_schur_initial_sse = best_sse
            schur_started = time.perf_counter()
            camera_damping = arguments.shared_schur_camera_damping
            landmark_damping = arguments.shared_schur_landmark_damping
            rejection_damping_factor = 2.0
            final_shared_schur_worker_sse = best_sse
            final_shared_schur_termination = "maximum_corrections"
            stop_schur_corrections = False
            previous_tangent_step = None
            final_shared_schur_bsr_symbolic_cache = SchurBSRSymbolicCache()
            screening_correction = 0
            confirmation_correction = 0
            correction = 0
            confirmation_active = False
            while True:
                if (
                    not confirmation_active
                    and screening_correction
                    >= arguments.shared_schur_maximum_corrections
                ):
                    confirmation_active = True
                    previous_tangent_step = None
                    if (
                        arguments.shared_schur_confirmation_camera_damping
                        > 0.0
                    ):
                        camera_damping = (
                            arguments.
                            shared_schur_confirmation_camera_damping
                        )
                    if (
                        arguments.shared_schur_confirmation_landmark_damping
                        > 0.0
                    ):
                        landmark_damping = (
                            arguments.
                            shared_schur_confirmation_landmark_damping
                        )
                    rejection_damping_factor = 2.0
                if confirmation_active:
                    if confirmation_correction >= (
                        arguments.shared_schur_python_confirmation_corrections
                    ):
                        break
                    correction_operator = "python"
                else:
                    correction_operator = arguments.shared_schur_operator
                correction_accepted = False
                correction_coarse_basis = (
                    similarity_gauge_tangent_basis(best_cameras)
                    if arguments.shared_schur_preconditioner
                    == "gauge_deflated"
                    else None
                )
                for attempt in range(arguments.shared_schur_maximum_attempts):
                    attempt_started = time.perf_counter()
                    attempt_initial_sse = best_sse
                    scaled_best_cameras = to_scaled_cameras(
                        best_cameras, camera_scaling
                    )
                    schur_assembly_started_at = time.perf_counter()
                    schur_systems = worker.build_schur_systems(
                        camera_indices_in_cluster,
                        point_indices_in_cluster,
                        scaled_best_cameras,
                        best_points,
                        cluster_count,
                        landmark_damping,
                    )
                    schur_assembly_seconds = (
                        time.perf_counter() - schur_assembly_started_at
                    )
                    tangent_step, schur_diagnostics = (
                        solve_global_schur_system(
                            schur_systems,
                            camera_count,
                            camera_damping,
                            arguments.shared_schur_step_scale,
                            arguments.shared_schur_linear_solver,
                            arguments.shared_schur_relative_tolerance,
                            arguments.shared_schur_maximum_iterations,
                            (
                                previous_tangent_step
                                if arguments.shared_schur_warm_start
                                else None
                            ),
                            correction_operator,
                            bsr_symbolic_cache=(
                                final_shared_schur_bsr_symbolic_cache
                            ),
                            preconditioner_mode=(
                                arguments.shared_schur_preconditioner
                            ),
                            coarse_basis=correction_coarse_basis,
                        )
                    )
                    previous_tangent_step = tangent_step
                    camera_step_started_at = time.perf_counter()
                    (
                        corrected_costs,
                        corrected_scaled_cameras,
                        corrected_points,
                    ) = worker.apply_camera_step(
                        camera_indices_in_cluster,
                        point_indices_in_cluster,
                        scaled_best_cameras,
                        best_points,
                        tangent_step,
                        cluster_count,
                        arguments.shared_schur_landmark_refinement_steps,
                    )
                    camera_step_seconds = (
                        time.perf_counter() - camera_step_started_at
                    )
                    candidate_worker_sse = float(np.sum(corrected_costs))
                    corrected_cameras = to_physical_cameras(
                        corrected_scaled_cameras, camera_scaling
                    )
                    evaluation_started_at = time.perf_counter()
                    corrected_metrics = evaluate_state(
                        corrected_cameras,
                        corrected_points,
                        camera_indices,
                        point_indices,
                        observations,
                    )
                    evaluation_seconds = (
                        time.perf_counter() - evaluation_started_at
                    )
                    candidate_sse = corrected_metrics["objectiveValue"]
                    relative_decrease = (
                        (attempt_initial_sse - candidate_sse)
                        / max(
                            attempt_initial_sse,
                            np.finfo(np.float64).tiny,
                        )
                        if np.isfinite(candidate_sse)
                        else float("-inf")
                    )
                    actual_reduction = 0.5 * (
                        attempt_initial_sse - candidate_sse
                    )
                    damped_predicted_reduction = schur_diagnostics[
                        "dampedPredictedReduction"
                    ]
                    undamped_predicted_reduction = schur_diagnostics[
                        "undampedPredictedReduction"
                    ]
                    damped_gain_ratio = (
                        actual_reduction / damped_predicted_reduction
                        if damped_predicted_reduction > 0.0
                        else float("-inf")
                    )
                    undamped_gain_ratio = (
                        actual_reduction / undamped_predicted_reduction
                        if undamped_predicted_reduction > 0.0
                        else float("-inf")
                    )
                    accepted = (
                        np.isfinite(candidate_sse)
                        and candidate_sse < attempt_initial_sse
                        and schur_diagnostics["linearTermination"] == 0
                        and (
                            arguments.shared_schur_damping_policy
                            != "model_ratio"
                            or damped_gain_ratio
                            > arguments.shared_schur_minimum_gain_ratio
                        )
                    )
                    final_shared_schur_diagnostics = schur_diagnostics
                    final_shared_schur_attempts.append({
                        "correction": correction,
                        "phase": (
                            "confirmation"
                            if confirmation_active else "screening"
                        ),
                        "operator": correction_operator,
                        "attempt": attempt,
                        "cameraDamping": camera_damping,
                        "landmarkDamping": landmark_damping,
                        "initialSSE": attempt_initial_sse,
                        "candidateSSE": candidate_sse,
                        "workerSSE": candidate_worker_sse,
                        "relativeDecrease": relative_decrease,
                        "actualReduction": actual_reduction,
                        "dampedGainRatio": damped_gain_ratio,
                        "undampedGainRatio": undamped_gain_ratio,
                        "accepted": accepted,
                        "seconds": time.perf_counter() - attempt_started,
                        "schurAssemblySeconds": schur_assembly_seconds,
                        "cameraStepSeconds": camera_step_seconds,
                        "evaluationSeconds": evaluation_seconds,
                        "diagnostics": schur_diagnostics,
                    })
                    if accepted:
                        correction_accepted = True
                        final_shared_schur_accepted = True
                        final_shared_schur_accepted_corrections += 1
                        if confirmation_active:
                            final_shared_schur_confirmation_corrections += 1
                            confirmation_correction += 1
                        else:
                            final_shared_schur_screening_corrections += 1
                            screening_correction += 1
                        final_shared_schur_worker_sse = candidate_worker_sse
                        best_sse = candidate_sse
                        best_metrics = corrected_metrics.copy()
                        best_cameras = corrected_cameras
                        best_points = corrected_points
                        if (
                            arguments.shared_schur_damping_policy
                            == "model_ratio"
                        ):
                            damping_factor = model_ratio_damping_factor(
                                damped_gain_ratio
                            )
                            rejection_damping_factor = 2.0
                        else:
                            damping_factor = (
                                arguments.shared_schur_damping_decrease
                            )
                        camera_damping *= damping_factor
                        landmark_damping *= damping_factor
                        if relative_decrease < (
                            arguments.shared_schur_minimum_relative_decrease
                        ):
                            if (
                                not confirmation_active
                                and arguments.
                                shared_schur_python_confirmation_corrections > 0
                            ):
                                if not (
                                    arguments.
                                    shared_schur_confirmation_after_screening_budget
                                ):
                                    final_shared_schur_termination = (
                                        "screening_minimum_relative_decrease"
                                    )
                                    confirmation_active = True
                                    previous_tangent_step = None
                            else:
                                final_shared_schur_termination = (
                                    "minimum_relative_decrease"
                                )
                                stop_schur_corrections = True
                        break
                    if (
                        arguments.shared_schur_damping_policy
                        == "model_ratio"
                    ):
                        damping_factor = rejection_damping_factor
                        rejection_damping_factor *= 2.0
                    else:
                        damping_factor = arguments.shared_schur_damping_increase
                    camera_damping, landmark_damping = rejected_schur_damping(
                        camera_damping,
                        landmark_damping,
                        attempt,
                        damping_factor,
                        arguments.shared_schur_fallback_camera_damping,
                        arguments.shared_schur_fallback_landmark_damping,
                    )
                if stop_schur_corrections:
                    break
                if not correction_accepted:
                    if (
                        not confirmation_active
                        and arguments.
                        shared_schur_python_confirmation_corrections > 0
                    ):
                        final_shared_schur_termination = (
                            "screening_attempts_exhausted"
                        )
                        confirmation_active = True
                        previous_tangent_step = None
                    else:
                        final_shared_schur_termination = "attempts_exhausted"
                        break
                correction += 1
            final_shared_schur_corrected_sse = best_sse
            final_shared_schur_final_camera_damping = camera_damping
            final_shared_schur_final_landmark_damping = landmark_damping
            final_shared_schur_seconds = time.perf_counter() - schur_started
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
            refined_metrics = evaluate_state(
                best_cameras,
                refined_points,
                camera_indices,
                point_indices,
                observations,
            )
            final_polishing_refined_sse = refined_metrics["objectiveValue"]
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

    final_metrics = evaluate_state(
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
        "localCameraStepScale": arguments.local_camera_step_scale,
        "localCameraStepGrid": arguments.local_camera_step_grid,
        "sharedCameraStepGrid": arguments.shared_camera_step_grid,
        "sharedCameraStepScale": arguments.shared_camera_step_scale,
        "sharedCameraDisagreementScale": (
            arguments.shared_camera_disagreement_scale
        ),
        "metricProposalDisagreementScale": (
            arguments.metric_proposal_disagreement_scale
        ),
        "metricProposalDisagreementUntil": (
            arguments.metric_proposal_disagreement_until
        ),
        "metricProposalDisagreementGrid": (
            arguments.metric_proposal_disagreement_grid
        ),
        "metricProposalSubspaceScales": (
            arguments.metric_proposal_subspace_scales
        ),
        "metricProposalDisagreementThreshold": (
            arguments.metric_proposal_disagreement_threshold
        ),
        "metricProposalDisagreementHysteresis": (
            arguments.metric_proposal_disagreement_hysteresis
        ),
        "schurAlignmentDiagnosticIterations": sorted(
            iteration + 1
            for iteration in schur_alignment_diagnostic_iterations
        ),
        "schurAlignmentCameraDamping": (
            arguments.schur_alignment_camera_damping
        ),
        "schurAlignmentLandmarkDamping": (
            arguments.schur_alignment_landmark_damping
        ),
        "schurAlignmentMaximumIterations": (
            arguments.schur_alignment_maximum_iterations
        ),
        "schurModelConsensusClipping": (
            arguments.schur_model_consensus_clipping
        ),
        "schurModelConsensusClippingMinimumScale": (
            arguments.schur_model_consensus_clipping_minimum_scale
        ),
        "schurModelConsensusClippingMaximumScale": (
            arguments.schur_model_consensus_clipping_maximum_scale
        ),
        "sharedCameraMetricBeta": arguments.shared_camera_metric_beta,
        "uniqueCameraMetricScale": arguments.unique_camera_metric_scale,
        "uniqueCameraMetricSelector": arguments.unique_camera_metric_selector,
        "sharedOnlyCameraProximal": arguments.shared_only_camera_proximal,
        "objectiveLoss": "huber" if arguments.huber_delta > 0.0 else "l2",
        "huberDelta": arguments.huber_delta,
        "adaptiveLocalDepth": arguments.adaptive_local_depth,
        "adaptiveLocalDepthStart": arguments.adaptive_local_depth_start,
        "interiorDefectDiagnostic": arguments.interior_defect_diagnostic,
        "sharedFixedInteriorTrial": (
            os.environ.get("BUNDLE_PALM_SHARED_FIXED_INTERIOR_TRIAL", "0")
            == "1"
        ),
        "sharedFixedInteriorTrialMaximumBacktracks": int(os.environ.get(
            "BUNDLE_PALM_SHARED_FIXED_INTERIOR_TRIAL_MAX_BACKTRACKS", "8"
        )),
        "adaptiveLocalDepthMaximum": (
            arguments.adaptive_local_depth_maximum
        ),
        "adaptiveLocalDepthHigh": arguments.adaptive_local_depth_high,
        "adaptiveLocalDepthLow": arguments.adaptive_local_depth_low,
        "adaptiveLocalDepthWindow": arguments.adaptive_local_depth_window,
        "adaptiveLocalDepthDwell": arguments.adaptive_local_depth_dwell,
        "finalLocalSteps": (
            adaptive_local_steps.tolist()
            if arguments.adaptive_local_depth
            else arguments.local_steps
        ),
        "nesterovMaxIterations": arguments.nesterov_max_iterations,
        "enhancedInnerMaxIterations": arguments.enhanced_inner_max_iterations,
        "nesterovMinIterations": arguments.nesterov_min_iterations,
        "nesterovStopTolerance": arguments.nesterov_stop_tolerance,
        "enhancedInnerUntil": arguments.enhanced_inner_until,
        "diagonalTrustUntil": arguments.diagonal_trust_until,
        "relativeResidualUntil": arguments.relative_residual_until,
        "stopAfterIteration": arguments.stop_after_iteration,
        "threadsPerCluster": arguments.threads_per_cluster,
        "localSolver": arguments.local_solver,
        "localSolverSwitchIteration": (
            arguments.local_solver_switch_iteration
        ),
        "localSolverAfterSwitch": arguments.local_solver_after_switch,
        "cameraUpdate": os.environ.get(
            "BUNDLE_PALM_CAMERA_UPDATE", "additive"
        ),
        "trustRegionPolicy": arguments.trust_region_policy,
        "persistentTrustRegion": arguments.persistent_trust_region,
        "trustRegionRecoveryRatio": arguments.trust_region_recovery_ratio,
        "sharedTrustRegionUntil": arguments.shared_trust_region_until,
        "collectiveTrustTrialUntil": arguments.collective_trust_trial_until,
        "localStateRebaseIteration": arguments.local_state_rebase_iteration,
        "sharedTrustRegionInitialRadius": (
            arguments.shared_trust_region_initial_radius
        ),
        "diagonalTrustDamping": (
            os.environ.get("BUNDLE_PALM_DIAGONAL_TRUST_DAMPING", "0") == "1"
        ),
        "baeTrustSchedule": (
            os.environ.get("BUNDLE_PALM_BAE_TRUST_SCHEDULE", "0") == "1"
        ),
        "cumulativeDiagonalDamping": (
            os.environ.get(
                "BUNDLE_PALM_CUMULATIVE_DIAGONAL_DAMPING", "0"
            ) == "1"
        ),
        "directTangentNormalEquations": (
            os.environ.get(
                "BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS", "0"
            ) == "1"
        ),
        "schurProximalMetric": (
            os.environ.get("BUNDLE_PALM_SCHUR_PROXIMAL_METRIC", "0") == "1"
        ),
        "schurProximalMetricBlend": float(
            os.environ.get("BUNDLE_PALM_SCHUR_PROXIMAL_METRIC_BLEND", "1")
        ),
        "schurProximalMetricSubspaceBlends": [
            float(os.environ.get(
                name,
                os.environ.get("BUNDLE_PALM_SCHUR_PROXIMAL_METRIC_BLEND", "1"),
            ))
            for name in (
                "BUNDLE_PALM_SCHUR_PROXIMAL_METRIC_TRANSLATION_BLEND",
                "BUNDLE_PALM_SCHUR_PROXIMAL_METRIC_ROTATION_BLEND",
                "BUNDLE_PALM_SCHUR_PROXIMAL_METRIC_INTRINSICS_BLEND",
            )
        ],
        "schurProximalMetricPreserveRawDiagonal": (
            os.environ.get(
                "BUNDLE_PALM_SCHUR_PROXIMAL_METRIC_PRESERVE_RAW_DIAGONAL",
                "0",
            ) == "1"
        ),
        "schurProximalMetricOffDiagonalMajorizer": (
            os.environ.get(
                "BUNDLE_PALM_SCHUR_PROXIMAL_METRIC_OFFDIAGONAL_MAJORIZE",
                "0",
            ) == "1"
        ),
        "schurProximalMetricOffDiagonalMajorizerScale": float(
            os.environ.get(
                "BUNDLE_PALM_SCHUR_PROXIMAL_METRIC_OFFDIAGONAL_MAJORIZE_SCALE",
                "1",
            )
        ),
        "globalSchurMajorizerObservabilityThreshold": (
            arguments.global_schur_majorizer_observability_threshold
        ),
        "globalSchurMajorizerUntil": (
            arguments.global_schur_majorizer_until
        ),
        "schurObservabilityDiagnostic": (
            os.environ.get(
                "BUNDLE_PALM_SCHUR_OBSERVABILITY_DIAGNOSTIC", "0"
            ) == "1"
        ),
        "disableLocalProximalTerm": (
            os.environ.get("BUNDLE_PALM_DISABLE_LOCAL_PROXIMAL_TERM", "0") == "1"
        ),
        "disableLandmarkPreconditioning": (
            os.environ.get(
                "BUNDLE_PALM_DISABLE_LANDMARK_PRECONDITIONING", "0"
            ) == "1"
        ),
        "pobaBlockRelativeFloor": float(
            os.environ.get("BUNDLE_PALM_POBA_BLOCK_RELATIVE_FLOOR", "0")
        ),
        "pobaDiagnosticIterations": int(
            os.environ.get("BUNDLE_PALM_POBA_DIAGNOSTIC_ITERATIONS", "0")
        ),
        "centralizedCeresJacobiScaling": (
            os.environ.get(
                "BUNDLE_PALM_CENTRALIZED_CERES_JACOBI_SCALING", "1"
            ) == "1"
        ),
        "centralizedCeresSparseSchur": (
            os.environ.get("BUNDLE_PALM_CENTRALIZED_CERES_SPARSE_SCHUR", "0")
            == "1"
        ),
        "centralizedCeresIterations": int(
            os.environ.get("BUNDLE_PALM_CENTRALIZED_CERES_ITERATIONS", "90")
        ),
        "initialTrustRegionRadius": float(
            os.environ.get("BUNDLE_PALM_INITIAL_TRUST_REGION_RADIUS", "10")
        ),
        "maximumTrustRegionRadius": float(
            os.environ.get("BUNDLE_PALM_MAXIMUM_TRUST_REGION_RADIUS", "1000000")
        ),
        "dabaInitialTrustRegionCap": float(
            os.environ.get("BUNDLE_PALM_DABA_INITIAL_TRUST_REGION_CAP", "100")
        ),
        "nesterovSchurLipschitz": float(
            os.environ.get("BUNDLE_PALM_NESTEROV_SCHUR_LIPSCHITZ", "0.9")
        ),
        "nesterovStopCheckInterval": int(
            os.environ.get("BUNDLE_PALM_NESTEROV_STOP_CHECK_INTERVAL", "1")
        ),
        "nesterovRelativeResidual": (
            os.environ.get("BUNDLE_PALM_NESTEROV_RELATIVE_RESIDUAL", "0") == "1"
        ),
        "schurPcgRelativeTolerance": float(
            os.environ.get("BUNDLE_PALM_SCHUR_PCG_RELATIVE_TOLERANCE", "0.01")
        ),
        "schurPcgQTolerance": float(
            os.environ.get("BUNDLE_PALM_SCHUR_PCG_Q_TOLERANCE", "0")
        ),
        "schurPcgJacobiPreconditioner": (
            os.environ.get(
                "BUNDLE_PALM_SCHUR_PCG_JACOBI_PRECONDITIONER", "0"
            ) == "1"
        ),
        "schurPcgMaximumIterations": int(
            os.environ.get("BUNDLE_PALM_SCHUR_PCG_MAX_ITERATIONS", "400")
        ),
        "sceneNormalization": arguments.scene_normalization,
        "initialState": (
            str(Path(arguments.initial_state).resolve())
            if arguments.initial_state else None
        ),
        "initialStateFrame": arguments.initial_state_frame,
        "cameraScaling": arguments.camera_scaling,
        "cameraScalingMaximumRatio": arguments.camera_scaling_maximum_ratio,
        "cameraScalingClippingPercentile": (
            arguments.camera_scaling_clipping_percentile
        ),
        "cameraScalingMinimum": float(np.min(
            camera_coordinate_scale_values(camera_scaling)
        )),
        "cameraScalingMaximum": float(np.max(
            camera_coordinate_scale_values(camera_scaling)
        )),
        "cameraScalingRatio": float(
            np.max(camera_coordinate_scale_values(camera_scaling))
            / np.min(camera_coordinate_scale_values(camera_scaling))
        ),
        "cameraScalingGeometricMean": float(np.exp(np.mean(np.log(
            camera_coordinate_scale_values(camera_scaling)
        )))),
        "workerTranslationZFocalCorrelationMedian": (
            float(np.median(worker_translation_z_focal_correlation))
            if worker_translation_z_focal_correlation.size else float("nan")
        ),
        "workerTranslationZFocalCorrelationP95Absolute": (
            float(np.percentile(
                np.abs(worker_translation_z_focal_correlation), 95
            ))
            if worker_translation_z_focal_correlation.size else float("nan")
        ),
        "workerTranslationZFocalCorrelationMaximumAbsolute": (
            float(np.max(np.abs(worker_translation_z_focal_correlation)))
            if worker_translation_z_focal_correlation.size else float("nan")
        ),
        "cameraDiagonalRelativeFloor": arguments.camera_diagonal_relative_floor,
        "cameraDiagonalTranslationFloor": float(os.environ.get(
            "BUNDLE_PALM_CAMERA_DIAGONAL_TRANSLATION_FLOOR",
            arguments.camera_diagonal_relative_floor,
        )),
        "cameraDiagonalRotationFloor": float(os.environ.get(
            "BUNDLE_PALM_CAMERA_DIAGONAL_ROTATION_FLOOR",
            arguments.camera_diagonal_relative_floor,
        )),
        "cameraDiagonalIntrinsicsFloor": float(os.environ.get(
            "BUNDLE_PALM_CAMERA_DIAGONAL_INTRINSICS_FLOOR",
            arguments.camera_diagonal_relative_floor,
        )),
        "cameraTrustDiagonalScale": arguments.camera_trust_diagonal_scale,
        "cameraDiagonalMetricScale": arguments.camera_diagonal_metric_scale,
        "so3TranslationMetricRatio": float(
            os.environ.get("BUNDLE_PALM_SO3_TRANSLATION_METRIC_RATIO", "1")
        ),
        "scalingSeconds": scaling_seconds,
        "initializationSeconds": initialization_seconds,
        "optimizationSeconds": optimization_seconds,
        "relaxation": arguments.relaxation,
        "outerAcceleration": arguments.outer_acceleration,
        "outerAccelerationUntil": arguments.outer_acceleration_until,
        "outerAccelerationRestartIteration": (
            arguments.outer_acceleration_restart_iteration
        ),
        "singleClusterProximal": arguments.single_cluster_proximal,
        "lineSearchGrid": arguments.line_search_grid,
        "accelerationRestartAfter": arguments.acceleration_restart_after,
        "acceleratedAcceptances": accelerated_acceptances,
        "nominalFallbacks": nominal_fallbacks,
        "accelerationMaximumStepRatio": accelerator.max_step_ratio,
        "accelerationStepLimitHits": accelerator.step_limit_hits,
        "proximalOracleCalls": proximal_oracle_calls,
        "proximalMetric": arguments.proximal_metric,
        "consensusMetric": arguments.consensus_metric,
        "consensusUnflooredCameraDiagonal": (
            os.environ.get(
                "BUNDLE_PALM_CONSENSUS_UNFLOORED_CAMERA_DIAGONAL", "0"
            ) == "1"
        ),
        "consensusSharedFloorPriorScale": (
            arguments.consensus_shared_floor_prior_scale
        ),
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
        "recoveryExhaustionPolicy": arguments.recovery_exhaustion_policy,
        "recoveryRelaxedIterations": arguments.recovery_relaxed_iterations,
        "bestCheckpointRestarts": best_checkpoint_restarts,
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
        "finalSharedSchurAttempted": final_shared_schur_attempted,
        "finalSharedSchurAccepted": final_shared_schur_accepted,
        "finalSharedSchurInitialSSE": final_shared_schur_initial_sse,
        "finalSharedSchurCorrectedSSE": final_shared_schur_corrected_sse,
        "finalSharedSchurWorkerSSE": final_shared_schur_worker_sse,
        "finalSharedSchurSeconds": final_shared_schur_seconds,
        "finalSharedSchurDiagnostics": final_shared_schur_diagnostics,
        "finalSharedSchurAttempts": final_shared_schur_attempts,
        "finalSharedSchurAcceptedCorrections": (
            final_shared_schur_accepted_corrections
        ),
        "finalSharedSchurScreeningCorrections": (
            final_shared_schur_screening_corrections
        ),
        "finalSharedSchurConfirmationCorrections": (
            final_shared_schur_confirmation_corrections
        ),
        "finalSharedSchurTermination": final_shared_schur_termination,
        "finalSharedSchurFinalCameraDamping": (
            final_shared_schur_final_camera_damping
        ),
        "finalSharedSchurFinalLandmarkDamping": (
            final_shared_schur_final_landmark_damping
        ),
        "initialSharedSchurCorrection": (
            arguments.initial_shared_schur_correction
        ),
        "initialSharedSchurOperator": (
            arguments.shared_schur_operator
            if arguments.initial_shared_schur_operator == "inherit"
            else arguments.initial_shared_schur_operator
        ),
        "initialSharedSchurMaximumIterations": (
            arguments.initial_shared_schur_maximum_iterations
            or arguments.shared_schur_maximum_iterations
        ),
        "initialSharedSchurMaximumCorrections": (
            arguments.initial_shared_schur_maximum_corrections
        ),
        "initialSharedSchurDampingPolicy": (
            arguments.initial_shared_schur_damping_policy
        ),
        "initialSharedSchurModelRatioMinimumFactor": (
            arguments.initial_shared_schur_model_ratio_minimum_factor
        ),
        "initialSharedSchurAttempted": initial_shared_schur_attempted,
        "initialSharedSchurAccepted": initial_shared_schur_accepted,
        "initialSharedSchurInitialSSE": initial_shared_schur_initial_sse,
        "initialSharedSchurCandidateSSE": initial_shared_schur_candidate_sse,
        "initialSharedSchurWorkerSSE": initial_shared_schur_worker_sse,
        "initialSharedSchurSeconds": initial_shared_schur_seconds,
        "initialSharedSchurDiagnostics": initial_shared_schur_diagnostics,
        "initialSharedSchurAttempts": initial_shared_schur_attempts,
        "initialSharedSchurAcceptedCorrections": (
            initial_shared_schur_accepted_corrections
        ),
        "initialSharedSchurTermination": initial_shared_schur_termination,
        "initialSharedSchurTrustRebase": (
            arguments.initial_shared_schur_rebase_trust_state
        ),
        "initialSharedSchurTrustRebased": initial_shared_schur_trust_rebased,
        "initialSharedSchurPreRebaseTrustRadii": (
            initial_shared_schur_pre_rebase_trust_radii
        ),
        "initialSharedSchurPostRebaseTrustRadii": (
            initial_shared_schur_post_rebase_trust_radii
        ),
        "midSharedSchurCorrectionIteration": (
            arguments.mid_shared_schur_correction_iteration
        ),
        "midSharedSchurAttempted": mid_shared_schur_attempted,
        "midSharedSchurAccepted": mid_shared_schur_accepted,
        "midSharedSchurInitialSSE": mid_shared_schur_initial_sse,
        "midSharedSchurCandidateSSE": mid_shared_schur_candidate_sse,
        "midSharedSchurWorkerSSE": mid_shared_schur_worker_sse,
        "midSharedSchurSeconds": mid_shared_schur_seconds,
        "midSharedSchurDiagnostics": mid_shared_schur_diagnostics,
        "midSharedSchurTrustRebased": mid_shared_schur_trust_rebased,
        "midSharedSchurTransportProductState": (
            arguments.mid_shared_schur_transport_product_state
        ),
        "midSharedSchurProductStateTransported": (
            mid_shared_schur_product_state_transported
        ),
        "midSharedSchurTransportWorkerSSE": (
            mid_shared_schur_transport_worker_sse
        ),
        "midSharedSchurTransportOffsetError": (
            mid_shared_schur_transport_offset_error
        ),
        "midSharedSchurPreRebaseTrustRadii": (
            mid_shared_schur_pre_rebase_trust_radii
        ),
        "midSharedSchurPostRebaseTrustRadii": (
            mid_shared_schur_post_rebase_trust_radii
        ),
        "initialSharedSchurBasinGuard": (
            arguments.initial_shared_schur_basin_guard
        ),
        "bootstrapBasinGuardRejections": bootstrap_basin_guard_rejections,
        "bootstrapBasinGuardReleaseIteration": (
            bootstrap_basin_guard_release_iteration + 1
            if bootstrap_basin_guard_release_iteration >= 0
            else -1
        ),
        "sharedSchurLandmarkDamping": (
            arguments.shared_schur_landmark_damping
        ),
        "sharedSchurCameraDamping": arguments.shared_schur_camera_damping,
        "sharedSchurStepScale": arguments.shared_schur_step_scale,
        "sharedSchurLinearSolver": arguments.shared_schur_linear_solver,
        "sharedSchurRelativeTolerance": (
            arguments.shared_schur_relative_tolerance
        ),
        "sharedSchurMaximumIterations": (
            arguments.shared_schur_maximum_iterations
        ),
        "sharedSchurMaximumCorrections": (
            arguments.shared_schur_maximum_corrections
        ),
        "sharedSchurPythonConfirmationCorrections": (
            arguments.shared_schur_python_confirmation_corrections
        ),
        "sharedSchurConfirmationCameraDamping": (
            arguments.shared_schur_confirmation_camera_damping
        ),
        "sharedSchurConfirmationLandmarkDamping": (
            arguments.shared_schur_confirmation_landmark_damping
        ),
        "sharedSchurConfirmationAfterScreeningBudget": (
            arguments.shared_schur_confirmation_after_screening_budget
        ),
        "sharedSchurMaximumAttempts": arguments.shared_schur_maximum_attempts,
        "sharedSchurDampingIncrease": arguments.shared_schur_damping_increase,
        "sharedSchurDampingDecrease": arguments.shared_schur_damping_decrease,
        "sharedSchurFallbackCameraDamping": (
            arguments.shared_schur_fallback_camera_damping
        ),
        "sharedSchurFallbackLandmarkDamping": (
            arguments.shared_schur_fallback_landmark_damping
        ),
        "sharedSchurDampingPolicy": arguments.shared_schur_damping_policy,
        "sharedSchurMinimumGainRatio": (
            arguments.shared_schur_minimum_gain_ratio
        ),
        "sharedSchurMinimumRelativeDecrease": (
            arguments.shared_schur_minimum_relative_decrease
        ),
        "sharedSchurWarmStart": arguments.shared_schur_warm_start,
        "sharedSchurPreconditioner": (
            arguments.shared_schur_preconditioner
        ),
        "sharedSchurOperator": arguments.shared_schur_operator,
        "sharedSchurLandmarkRefinementSteps": (
            arguments.shared_schur_landmark_refinement_steps
        ),
        "targetTransformedLipschitz": (
            arguments.target_transformed_lipschitz
        ),
        "initialPenalty": (
            arguments.penalty_multiplier * 2.5 * len(observations) / camera_count
        ),
        "initialBlockRegularization": arguments.block_regularization,
        "initialBlockDiagonalCoefficient": (
            arguments.camera_diagonal_metric_scale
            * arguments.block_regularization
            if arguments.proximal_metric == "block"
            else None
        ),
        "finalPenalty": penalty,
        "finalBlockRegularization": block_regularization,
        "finalBlockDiagonalCoefficient": (
            arguments.camera_diagonal_metric_scale * block_regularization
            if arguments.proximal_metric == "block"
            else None
        ),
        "gamma": 1.0 / penalty if arguments.proximal_metric == "scalar" else 1.0,
        "safeguardMode": arguments.safeguard_mode,
        "dreRelativeIncrease": arguments.dre_relative_increase,
        "minimumPrimalRatio": arguments.minimum_primal_ratio,
        "safeguardAnnealingIterations": safeguard_annealing_iterations,
        "safeguardReferenceIteration": arguments.safeguard_reference_iteration,
        "safeguardAnnealingExponent": arguments.safeguard_annealing_exponent,
        "safeguardRelativeDeadband": arguments.safeguard_relative_deadband,
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
