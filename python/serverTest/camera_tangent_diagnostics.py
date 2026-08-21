"""Camera tangent diagnostics matching the left-SE3 worker update."""

import numpy as np
from scipy.spatial.transform import Rotation


def _skew(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    result = np.zeros(vectors.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 1] = -vectors[..., 2]
    result[..., 0, 2] = vectors[..., 1]
    result[..., 1, 0] = vectors[..., 2]
    result[..., 1, 2] = -vectors[..., 0]
    result[..., 2, 0] = -vectors[..., 1]
    result[..., 2, 1] = vectors[..., 0]
    return result


def left_se3_camera_plus(cameras, tangent):
    """Apply `[translation, rotation, intrinsics]` left-SE3 tangents."""
    cameras = np.asarray(cameras, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    if cameras.shape != tangent.shape or cameras.shape[-1] != 9:
        raise ValueError("cameras and tangent must have matching (..., 9) shape")
    rotation = tangent[..., 3:6]
    translation = tangent[..., :3]
    theta_squared = np.sum(rotation * rotation, axis=-1)
    theta = np.sqrt(theta_squared)
    coefficient_a = np.empty_like(theta)
    coefficient_b = np.empty_like(theta)
    regular = theta > 1e-10
    coefficient_a[regular] = (
        (1.0 - np.cos(theta[regular])) / theta_squared[regular]
    )
    coefficient_b[regular] = (
        (theta[regular] - np.sin(theta[regular]))
        / (theta_squared[regular] * theta[regular])
    )
    coefficient_a[~regular] = 0.5 - theta_squared[~regular] / 24.0
    coefficient_b[~regular] = 1.0 / 6.0 - theta_squared[~regular] / 120.0
    cross = np.cross(rotation, translation)
    exponential_translation = (
        translation
        + coefficient_a[..., None] * cross
        + coefficient_b[..., None] * np.cross(rotation, cross)
    )
    delta_rotation = Rotation.from_rotvec(rotation.reshape(-1, 3))
    current_rotation = Rotation.from_rotvec(cameras[..., :3].reshape(-1, 3))
    updated = cameras.copy()
    updated[..., 3:6] = (
        delta_rotation.apply(cameras[..., 3:6].reshape(-1, 3))
        + exponential_translation.reshape(-1, 3)
    ).reshape(cameras.shape[:-1] + (3,))
    updated[..., :3] = (
        delta_rotation * current_rotation
    ).as_rotvec().reshape(cameras.shape[:-1] + (3,))
    updated[..., 6:9] += tangent[..., 6:9]
    return updated


def left_se3_camera_minus(updated, cameras):
    """Return the left-SE3 tangent taking `cameras` to `updated`."""
    updated = np.asarray(updated, dtype=np.float64)
    cameras = np.asarray(cameras, dtype=np.float64)
    if updated.shape != cameras.shape or cameras.shape[-1] != 9:
        raise ValueError("camera arrays must have matching (..., 9) shape")
    updated_rotation = Rotation.from_rotvec(updated[..., :3].reshape(-1, 3))
    current_rotation = Rotation.from_rotvec(cameras[..., :3].reshape(-1, 3))
    relative_rotation = updated_rotation * current_rotation.inv()
    rotation = relative_rotation.as_rotvec().reshape(cameras.shape[:-1] + (3,))
    rotated_translation = relative_rotation.apply(
        cameras[..., 3:6].reshape(-1, 3)
    ).reshape(cameras.shape[:-1] + (3,))
    relative_translation = updated[..., 3:6] - rotated_translation
    theta_squared = np.sum(rotation * rotation, axis=-1)
    theta = np.sqrt(theta_squared)
    coefficient = np.full_like(theta, 1.0 / 12.0)
    regular = theta_squared > 1e-20
    coefficient[regular] = (
        1.0 / theta_squared[regular]
        - (1.0 + np.cos(theta[regular]))
        / (2.0 * theta[regular] * np.sin(theta[regular]))
    )
    rotation_skew = _skew(rotation)
    inverse_left_jacobian = (
        np.eye(3)
        - 0.5 * rotation_skew
        + coefficient[..., None, None]
        * np.matmul(rotation_skew, rotation_skew)
    )
    tangent = np.empty_like(cameras)
    tangent[..., :3] = np.einsum(
        "...ij,...j->...i", inverse_left_jacobian, relative_translation
    )
    tangent[..., 3:6] = rotation
    tangent[..., 6:9] = updated[..., 6:9] - cameras[..., 6:9]
    return tangent


def tangent_alignment(reference, candidate):
    """Return cosine and norm telemetry globally and by camera subspace."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.shape[-1] != 9:
        raise ValueError("tangent arrays must have matching (..., 9) shape")

    def summarize(left, right):
        left = left.ravel()
        right = right.ravel()
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        denominator = left_norm * right_norm
        cosine = (
            float(np.dot(left, right) / denominator)
            if denominator > 0.0
            else float("nan")
        )
        return {
            "cosine": cosine,
            "referenceNorm": left_norm,
            "candidateNorm": right_norm,
        }

    return {
        "global": summarize(reference, candidate),
        "translation": summarize(reference[..., :3], candidate[..., :3]),
        "rotation": summarize(reference[..., 3:6], candidate[..., 3:6]),
        "intrinsics": summarize(reference[..., 6:9], candidate[..., 6:9]),
    }


def project_tangent_orthogonal_to_basis(tangent, basis, diagonal=None):
    """Project a camera tangent off a basis in a diagonal metric."""
    tangent = np.asarray(tangent, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if tangent.ndim < 2 or tangent.shape[-1] != 9:
        raise ValueError("tangent must have shape (..., 9)")
    if basis.ndim != 2 or basis.shape[0] != tangent.size:
        raise ValueError("basis must have one row per tangent coordinate")
    flat_tangent = tangent.ravel()
    if diagonal is None:
        weights = np.ones_like(flat_tangent)
    else:
        diagonal = np.asarray(diagonal, dtype=np.float64)
        if diagonal.shape != tangent.shape:
            raise ValueError("diagonal must match tangent shape")
        positive = diagonal[diagonal > 0.0]
        floor = (
            float(np.median(positive)) * 1e-12
            if positive.size
            else np.finfo(np.float64).tiny
        )
        weights = np.maximum(diagonal.ravel(), floor)
    gram = basis.T @ (weights[:, None] * basis)
    coefficients = np.linalg.solve(
        gram, basis.T @ (weights * flat_tangent)
    )
    component = basis @ coefficients
    residual = flat_tangent - component
    tangent_norm = float(np.sqrt(np.sum(weights * flat_tangent**2)))
    component_norm = float(np.sqrt(np.sum(weights * component**2)))
    residual_norm = float(np.sqrt(np.sum(weights * residual**2)))
    return residual.reshape(tangent.shape), {
        "componentNorm": component_norm,
        "residualNorm": residual_norm,
        "componentFraction": (
            component_norm / tangent_norm if tangent_norm > 0.0 else 0.0
        ),
    }


def consensus_vote_coherence(
    values,
    previous_consensus,
    cluster_indices,
    camera_indices,
    metric_blocks,
    copy_count,
):
    """Measure cancellation of block-metric consensus votes per shared camera."""
    values = np.asarray(values, dtype=np.float64)
    previous_consensus = np.asarray(previous_consensus, dtype=np.float64)
    cluster_indices = np.asarray(cluster_indices)
    camera_indices = np.asarray(camera_indices)
    metric_blocks = np.asarray(metric_blocks, dtype=np.float64)
    copy_count = np.asarray(copy_count)
    if values.ndim != 3 or values.shape[2] != 9:
        raise ValueError("values must have shape (clusters, cameras, 9)")
    if previous_consensus.shape != values.shape[1:]:
        raise ValueError("previous consensus must match global cameras")
    if cluster_indices.shape != camera_indices.shape:
        raise ValueError("active metric indices must match")
    if metric_blocks.shape != (camera_indices.size, 9, 9):
        raise ValueError("metric blocks must match active copies")
    if copy_count.shape != (values.shape[1],):
        raise ValueError("copy count must match global cameras")

    displacements = (
        values[cluster_indices, camera_indices]
        - previous_consensus[camera_indices]
    )
    products = np.einsum("bij,bj->bi", metric_blocks, displacements)
    metric_sum = np.zeros((values.shape[1], 9, 9), dtype=np.float64)
    right_hand_side = np.zeros((values.shape[1], 9), dtype=np.float64)
    np.add.at(metric_sum, camera_indices, metric_blocks)
    np.add.at(right_hand_side, camera_indices, products)
    active_shared = copy_count[camera_indices] > 1
    shared_camera_indices = camera_indices[active_shared]
    contributions = np.linalg.solve(
        metric_sum[shared_camera_indices],
        products[active_shared, ..., None],
    )[..., 0]
    contribution_norms = np.sqrt(np.maximum(
        np.einsum(
            "bi,bij,bj->b",
            contributions,
            metric_sum[shared_camera_indices],
            contributions,
        ),
        0.0,
    ))
    denominator = np.zeros(values.shape[1], dtype=np.float64)
    np.add.at(denominator, shared_camera_indices, contribution_norms)
    shared = copy_count > 1
    aggregate = np.zeros_like(right_hand_side)
    aggregate[shared] = np.linalg.solve(
        metric_sum[shared], right_hand_side[shared, ..., None]
    )[..., 0]
    aggregate_norm = np.sqrt(np.maximum(
        np.einsum("bi,bij,bj->b", aggregate, metric_sum, aggregate),
        0.0,
    ))
    valid = shared & (denominator > 0.0)
    coherence = np.divide(
        aggregate_norm,
        denominator,
        out=np.zeros_like(aggregate_norm),
        where=denominator > 0.0,
    )
    shared_coherence = coherence[valid]
    return {
        "sharedCameraCount": int(np.count_nonzero(shared)),
        "activeSharedCameraCount": int(shared_coherence.size),
        "global": float(
            np.sum(aggregate_norm[shared]) / np.sum(denominator[shared])
        ) if np.sum(denominator[shared]) > 0.0 else 0.0,
        "minimum": float(np.min(shared_coherence))
        if shared_coherence.size else 0.0,
        "median": float(np.median(shared_coherence))
        if shared_coherence.size else 0.0,
        "maximum": float(np.max(shared_coherence))
        if shared_coherence.size else 0.0,
    }


def diagonal_weighted_copy_alignment(
    reference, candidates, diagonal, camera_indices
):
    """Compare individual camera-copy tangents with global reference tangents."""
    reference = np.asarray(reference, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    diagonal = np.asarray(diagonal, dtype=np.float64)
    camera_indices = np.asarray(camera_indices)
    if reference.ndim != 2 or reference.shape[1] != 9:
        raise ValueError("reference must have shape (cameras, 9)")
    if diagonal.shape != reference.shape:
        raise ValueError("diagonal must match reference")
    if candidates.shape != (camera_indices.size, 9):
        raise ValueError("candidates must match camera indices")
    if np.any(camera_indices < 0) or np.any(camera_indices >= reference.shape[0]):
        raise ValueError("camera index is out of range")
    selected_diagonal = diagonal[camera_indices]
    positive = diagonal[diagonal > 0.0]
    floor = (
        float(np.median(positive)) * 1e-12
        if positive.size
        else np.finfo(np.float64).tiny
    )
    weights = np.sqrt(np.maximum(selected_diagonal, floor))
    weighted_reference = reference[camera_indices] * weights
    weighted_candidates = candidates * weights
    reference_norms = np.linalg.norm(weighted_reference, axis=1)
    candidate_norms = np.linalg.norm(weighted_candidates, axis=1)
    denominators = reference_norms * candidate_norms
    copy_cosines = np.divide(
        np.sum(weighted_reference * weighted_candidates, axis=1),
        denominators,
        out=np.full(camera_indices.size, np.nan),
        where=denominators > 0.0,
    )
    finite = np.isfinite(copy_cosines)
    unique_cameras = np.unique(camera_indices)
    best_cosines = np.array([
        np.nanmax(copy_cosines[camera_indices == camera])
        for camera in unique_cameras
        if np.any(np.isfinite(copy_cosines[camera_indices == camera]))
    ])
    global_alignment = tangent_alignment(
        weighted_reference, weighted_candidates
    )["global"]
    return {
        "global": global_alignment,
        "copyCount": int(camera_indices.size),
        "finiteCopyCount": int(np.count_nonzero(finite)),
        "positiveCopyFraction": float(np.mean(copy_cosines[finite] > 0.0))
        if np.any(finite) else 0.0,
        "copyCosineMedian": float(np.median(copy_cosines[finite]))
        if np.any(finite) else float("nan"),
        "cameraBestCosineMedian": float(np.median(best_cosines))
        if best_cosines.size else float("nan"),
        "positiveCameraFraction": float(np.mean(best_cosines > 0.0))
        if best_cosines.size else 0.0,
    }


def diagonal_weighted_tangent_alignment(reference, candidate, diagonal):
    """Return tangent alignment after whitening by a positive diagonal."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    diagonal = np.asarray(diagonal, dtype=np.float64)
    if reference.shape != candidate.shape or reference.shape != diagonal.shape:
        raise ValueError("reference, candidate, and diagonal must match")
    positive = diagonal[diagonal > 0.0]
    floor = (
        float(np.median(positive)) * 1e-12
        if positive.size
        else np.finfo(np.float64).tiny
    )
    weights = np.sqrt(np.maximum(diagonal, floor))
    return tangent_alignment(reference * weights, candidate * weights)