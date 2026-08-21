"""Behavior-neutral coupled consensus oracles from reduced Schur systems."""

import numpy as np

from drs_coupled_metrics import (
    CoupledCameraMetricBlocks,
    project_coupled_consensus,
)


def _floor_symmetric_block(block, relative_floor=1e-12):
    block = 0.5 * (block + block.T)
    eigenvalues, eigenvectors = np.linalg.eigh(block)
    positive = eigenvalues[eigenvalues > 0.0]
    scale = float(np.max(positive)) if positive.size else 1.0
    floor = max(scale * relative_floor, np.finfo(np.float64).tiny)
    floored = np.einsum(
        "ij,j,kj->ik",
        eigenvectors,
        np.maximum(eigenvalues, floor),
        eigenvectors,
    )
    return 0.5 * (floored + floored.T)


def stabilized_coupled_metric_from_schur_systems(
    systems,
    camera_masks,
    camera_count,
    shared_only=True,
):
    """Build the historical Frobenius-stabilized coupled Schur metric."""
    camera_masks = np.asarray(camera_masks, dtype=bool)
    if camera_masks.shape != (len(systems), camera_count):
        raise ValueError("camera masks must match Schur systems")
    shared = np.sum(camera_masks, axis=0) > 1
    cluster_indices = []
    block_rows = []
    block_columns = []
    blocks = []
    identity = np.eye(9)

    for cluster, system in enumerate(systems):
        local_blocks = {}
        for row, column, block in zip(
            system.block_rows, system.block_columns, system.blocks
        ):
            row = int(row)
            column = int(column)
            block = np.asarray(block, dtype=np.float64)
            if row > column:
                row, column = column, row
                block = block.T
            if shared_only and not (shared[row] and shared[column]):
                continue
            key = row, column
            local_blocks[key] = local_blocks.get(key, 0.0) + block

        for (row, column), block in tuple(local_blocks.items()):
            if row == column:
                continue
            bound = float(np.linalg.norm(block, ord="fro"))
            for camera in (row, column):
                key = camera, camera
                local_blocks[key] = local_blocks.get(key, 0.0) + bound * identity

        for (row, column), block in sorted(local_blocks.items()):
            if row == column:
                block = _floor_symmetric_block(block)
            cluster_indices.append(cluster)
            block_rows.append(row)
            block_columns.append(column)
            blocks.append(block)

    if not blocks:
        raise ValueError("coupled Schur metric has no active blocks")
    return CoupledCameraMetricBlocks(
        np.asarray(cluster_indices, dtype=np.int64),
        np.asarray(block_rows, dtype=np.int64),
        np.asarray(block_columns, dtype=np.int64),
        np.asarray(blocks, dtype=np.float64),
        len(systems),
        camera_count,
    )


def project_stabilized_coupled_tangents(
    tangent_copies,
    camera_masks,
    systems,
):
    """Project tangent copies while fixing direct singleton tangents at zero."""
    tangent_copies = np.asarray(tangent_copies, dtype=np.float64)
    metric = stabilized_coupled_metric_from_schur_systems(
        systems,
        camera_masks,
        tangent_copies.shape[1],
        shared_only=True,
    )
    return project_coupled_consensus(
        tangent_copies,
        camera_masks,
        metric,
        direct_singletons=np.zeros_like(tangent_copies),
    )


def jacobi_refine_schur_tangent(
    systems,
    camera_count,
    camera_damping,
    tangent,
    active_cameras,
):
    """Apply one block-Jacobi correction to a restricted Schur tangent."""
    tangent = np.asarray(tangent, dtype=np.float64)
    active_cameras = np.asarray(active_cameras, dtype=bool)
    if tangent.shape != (camera_count, 9):
        raise ValueError("tangent must match the global camera shape")
    if active_cameras.shape != (camera_count,):
        raise ValueError("active camera mask has an invalid shape")
    if camera_damping <= 0.0:
        raise ValueError("camera damping must be positive")

    gradient = np.zeros_like(tangent)
    camera_diagonal = np.zeros((camera_count, 9, 9), dtype=np.float64)
    preconditioner = np.zeros_like(camera_diagonal)
    action = np.zeros_like(tangent)
    for system in systems:
        np.add.at(gradient, system.camera_ids, system.reduced_gradient)
        np.add.at(camera_diagonal, system.camera_ids, system.camera_diagonal)
        block_action = np.einsum(
            "bij,bj->bi", system.blocks, tangent[system.block_columns]
        )
        np.add.at(action, system.block_rows, block_action)
        offdiagonal = system.block_rows != system.block_columns
        if np.any(offdiagonal):
            transpose_action = np.einsum(
                "bji,bj->bi",
                system.blocks[offdiagonal],
                tangent[system.block_rows[offdiagonal]],
            )
            np.add.at(
                action,
                system.block_columns[offdiagonal],
                transpose_action,
            )
        diagonal = ~offdiagonal
        np.add.at(
            preconditioner,
            system.block_rows[diagonal],
            system.blocks[diagonal],
        )

    damping_diagonal = np.diagonal(
        camera_diagonal, axis1=1, axis2=2
    )
    positive = damping_diagonal[damping_diagonal > 0.0]
    floor = (
        float(np.median(positive)) * 1e-12
        if positive.size else 1e-12
    )
    damping_diagonal = np.maximum(damping_diagonal, floor)
    action += camera_damping * damping_diagonal * tangent
    indices = np.arange(9)
    preconditioner[:, indices, indices] += (
        camera_damping * damping_diagonal
    )
    preconditioner = 0.5 * (
        preconditioner + np.swapaxes(preconditioner, 1, 2)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(preconditioner)
    positive = eigenvalues[eigenvalues > 0.0]
    eigenvalue_floor = max(
        float(np.median(positive)) * 1e-12 if positive.size else 0.0,
        np.finfo(np.float64).tiny,
    )
    inverse = np.einsum(
        "bij,bj,bkj->bik",
        eigenvectors,
        1.0 / np.maximum(eigenvalues, eigenvalue_floor),
        eigenvectors,
    )
    residual = -(gradient + action)
    residual[~active_cameras] = 0.0
    correction = np.einsum("bij,bj->bi", inverse, residual)
    correction[~active_cameras] = 0.0
    refined = tangent + correction
    return refined, {
        "activeCameraCount": int(np.count_nonzero(active_cameras)),
        "residualNorm": float(np.linalg.norm(residual[active_cameras])),
        "correctionNorm": float(np.linalg.norm(correction[active_cameras])),
    }