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