"""Common final-state metrics for Bundle Adjustment in the Large datasets."""

import argparse
import bz2
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation


def angle_axis_rotate_points(angle_axis, points):
    """Rotate one point per angle-axis vector using Rodrigues' formula."""
    angle_axis = np.asarray(angle_axis, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    theta_squared = np.sum(angle_axis * angle_axis, axis=1)
    nonzero = theta_squared > 0
    rotated = np.empty_like(points)

    if np.any(nonzero):
        theta = np.sqrt(theta_squared[nonzero])
        axis = angle_axis[nonzero] / theta[:, None]
        cos_theta = np.cos(theta)[:, None]
        sin_theta = np.sin(theta)[:, None]
        selected_points = points[nonzero]
        rotated[nonzero] = (
            selected_points * cos_theta
            + np.cross(axis, selected_points) * sin_theta
            + axis * np.sum(axis * selected_points, axis=1)[:, None]
            * (1.0 - cos_theta)
        )

    if np.any(~nonzero):
        selected_axis = angle_axis[~nonzero]
        selected_points = points[~nonzero]
        rotated[~nonzero] = selected_points + np.cross(
            selected_axis, selected_points)

    return rotated


def angle_axis_rotation_matrices(angle_axis):
    """Convert angle-axis rows to rotation matrices."""
    angle_axis = np.asarray(angle_axis, dtype=np.float64)
    camera_count = angle_axis.shape[0]
    basis = np.broadcast_to(np.eye(3), (camera_count, 3, 3)).copy()
    return np.stack([
        angle_axis_rotate_points(angle_axis, basis[:, :, column])
        for column in range(3)
    ], axis=2)


def project_bal(cameras, points, camera_indices, point_indices):
    """Project BAL points with the Snavely nine-parameter camera model."""
    cameras = np.asarray(cameras, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    point_indices = np.asarray(point_indices, dtype=np.int64)
    observed_cameras = cameras[camera_indices]
    camera_points = angle_axis_rotate_points(
        observed_cameras[:, :3], points[point_indices])
    camera_points += observed_cameras[:, 3:6]
    normalized = -camera_points[:, :2] / camera_points[:, 2, None]
    radius_squared = np.sum(normalized * normalized, axis=1)
    distortion = 1.0 + radius_squared * (
        observed_cameras[:, 7]
        + observed_cameras[:, 8] * radius_squared
    )
    return normalized * (
        observed_cameras[:, 6] * distortion
    )[:, None]


def evaluate_bal_state(
    cameras,
    points,
    camera_indices,
    point_indices,
    observations,
    huber_delta=None,
):
    """Return explicitly named quality metrics for one BAL state."""
    observations = np.asarray(observations, dtype=np.float64)
    residuals = project_bal(
        cameras, points, camera_indices, point_indices) - observations
    if not np.all(np.isfinite(residuals)):
        raise ValueError("BAL state produced non-finite reprojection residuals")
    squared_errors = np.sum(residuals * residuals, axis=1)
    errors = np.sqrt(squared_errors)
    observation_count = int(errors.size)
    if observation_count == 0:
        raise ValueError("cannot evaluate a BAL state without observations")

    sum_squared_error = float(np.sum(squared_errors))
    metrics = {
        "observationCount": observation_count,
        "scalarResidualCount": 2 * observation_count,
        "sumSquaredError": sum_squared_error,
        "ceresCost": 0.5 * sum_squared_error,
        "msePerObservation": sum_squared_error / observation_count,
        "msePerScalarResidual": sum_squared_error / (2 * observation_count),
        "rmsePerObservation": float(np.sqrt(sum_squared_error / observation_count)),
        "rmsePerScalarResidual": float(
            np.sqrt(sum_squared_error / (2 * observation_count))),
        "meanReprojectionError": float(np.mean(errors)),
        "medianReprojectionError": float(np.median(errors)),
        "p90ReprojectionError": float(np.percentile(errors, 90)),
        "p95ReprojectionError": float(np.percentile(errors, 95)),
        "maxReprojectionError": float(np.max(errors)),
    }

    if huber_delta is not None:
        if not np.isfinite(huber_delta) or huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive")
        delta_squared = huber_delta * huber_delta
        rho = np.where(
            squared_errors <= delta_squared,
            squared_errors,
            2.0 * huber_delta * errors - delta_squared,
        )
        metrics["huberDelta"] = float(huber_delta)
        metrics["huberCeresCost"] = 0.5 * float(np.sum(rho))

    return metrics


def evaluate_daba_ray_state(
    initial_cameras,
    cameras,
    points,
    camera_indices,
    point_indices,
    observations,
    fit_intrinsics=True,
):
    """Evaluate DABA's weighted 3D-ray cost per observation.

    Observation normalization and weights remain fixed from the original BAL
    cameras. When requested, the three DABA ray intrinsics are minimized for
    the supplied camera poses and points by independent 3x3 least squares.
    """
    initial_cameras = np.asarray(initial_cameras, dtype=np.float64)
    cameras = np.asarray(cameras, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    point_indices = np.asarray(point_indices, dtype=np.int64)
    observations = np.asarray(observations, dtype=np.float64)
    camera_count = cameras.shape[0]

    initial_focal = initial_cameras[:, 6]
    normalized_observations = observations / -initial_focal[camera_indices, None]
    radius_squared = np.sum(normalized_observations**2, axis=1)
    basis = np.column_stack((
        np.ones_like(radius_squared),
        radius_squared,
        radius_squared**2,
    ))
    sqrt_weights = initial_focal[camera_indices] * np.sqrt(
        radius_squared + 1.0)

    rotations = angle_axis_rotation_matrices(cameras[:, :3])
    transposed_rotations = np.swapaxes(rotations, 1, 2)
    centers = np.einsum(
        "nij,nj->ni", transposed_rotations, cameras[:, 3:6])
    distances = -points[point_indices] - centers[camera_indices]
    camera_rotations = transposed_rotations[camera_indices]
    base_rays = (
        camera_rotations[:, :, 0] * normalized_observations[:, 0, None]
        + camera_rotations[:, :, 1] * normalized_observations[:, 1, None]
    )
    axial_rays = camera_rotations[:, :, 2]
    distance_squared = np.sum(distances**2, axis=1) + 1e-12
    denominator = distance_squared + 1e-6 * np.sqrt(distance_squared)

    def project_from_distances(vectors):
        return vectors - distances * (
            np.sum(distances * vectors, axis=1) / denominator
        )[:, None]

    projected_base = project_from_distances(base_rays)
    projected_axis = project_from_distances(axial_rays)

    if fit_intrinsics:
        axis_squared = np.sum(projected_axis**2, axis=1)
        axis_base = np.sum(projected_axis * projected_base, axis=1)
        weighted_axis_squared = sqrt_weights**2 * axis_squared
        weighted_axis_base = sqrt_weights**2 * axis_base
        normal_matrices = np.zeros((camera_count, 3, 3), dtype=np.float64)
        right_hand_sides = np.zeros((camera_count, 3), dtype=np.float64)
        for row in range(3):
            np.add.at(
                right_hand_sides[:, row],
                camera_indices,
                -weighted_axis_base * basis[:, row],
            )
            for column in range(3):
                np.add.at(
                    normal_matrices[:, row, column],
                    camera_indices,
                    weighted_axis_squared * basis[:, row] * basis[:, column],
                )
        intrinsics = np.empty((camera_count, 3), dtype=np.float64)
        for camera_index in range(camera_count):
            intrinsics[camera_index] = np.linalg.lstsq(
                normal_matrices[camera_index],
                right_hand_sides[camera_index],
                rcond=None,
            )[0]
    else:
        intrinsics = np.column_stack((
            np.ones(camera_count),
            cameras[:, 7] * initial_focal**2,
            cameras[:, 8] * initial_focal**4,
        ))

    ray_depth = np.sum(basis * intrinsics[camera_indices], axis=1)
    residuals = (
        projected_base + projected_axis * ray_depth[:, None]
    ) * sqrt_weights[:, None]
    squared_norms = np.sum(residuals**2, axis=1)
    if not np.all(np.isfinite(squared_norms)):
        raise ValueError("DABA ray evaluation produced non-finite residuals")
    cost = 0.5 * float(np.sum(squared_norms))
    return {
        "observationCount": int(observations.shape[0]),
        "ceresCost": cost,
        "reportedMetric": cost / observations.shape[0],
        "meanRayResidualNorm": float(np.mean(np.sqrt(squared_norms))),
        "maxRayResidualNorm": float(np.sqrt(np.max(squared_norms))),
        "intrinsicsFitted": bool(fit_intrinsics),
    }


def evaluate_encoded_daba_ray_state(
    initial_cameras,
    cameras,
    points,
    camera_indices,
    point_indices,
    observations,
):
    """Evaluate DABA's ray objective in a nine-parameter camera encoding."""
    initial_cameras = np.asarray(initial_cameras, dtype=np.float64)
    cameras = np.asarray(cameras, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    point_indices = np.asarray(point_indices, dtype=np.int64)
    observations = np.asarray(observations, dtype=np.float64)
    initial_focal = initial_cameras[:, 6]
    observed_initial_focal = initial_focal[camera_indices]
    normalized = observations / -observed_initial_focal[:, None]
    radius_squared = np.sum(normalized**2, axis=1)
    observed_cameras = cameras[camera_indices]
    ray_height = (
        observed_cameras[:, 6] / observed_initial_focal
        + observed_cameras[:, 7]
        * observed_initial_focal**2 * radius_squared
        + observed_cameras[:, 8]
        * observed_initial_focal**4 * radius_squared**2
    )
    rays = np.column_stack((normalized, ray_height))
    rotated_points = angle_axis_rotate_points(
        observed_cameras[:, :3], points[point_indices])
    camera_points = rotated_points + observed_cameras[:, 3:6]
    distances = -camera_points
    distance_squared = np.sum(distances**2, axis=1) + 1e-12
    denominator = distance_squared + 1e-6 * np.sqrt(distance_squared)
    residuals = rays - distances * (
        np.sum(distances * rays, axis=1) / denominator)[:, None]
    sqrt_weights = observed_initial_focal * np.sqrt(radius_squared + 1.0)
    residuals *= sqrt_weights[:, None]
    squared_norms = np.sum(residuals**2, axis=1)
    if not np.all(np.isfinite(squared_norms)):
        raise ValueError("encoded DABA ray state produced non-finite residuals")
    cost = 0.5 * float(np.sum(squared_norms))
    return {
        "observationCount": int(observations.shape[0]),
        "ceresCost": cost,
        "reportedMetric": cost / observations.shape[0],
        "meanRayResidualNorm": float(np.mean(np.sqrt(squared_norms))),
        "maxRayResidualNorm": float(np.sqrt(np.max(squared_norms))),
    }


def encoded_daba_ray_state_to_matrix(initial_cameras, cameras, points):
    """Convert the nine-parameter ray encoding to DABA's 3x5 cameras."""
    initial_cameras = np.asarray(initial_cameras, dtype=np.float64)
    cameras = np.asarray(cameras, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    rotations = angle_axis_rotation_matrices(cameras[:, :3])
    transposed_rotations = np.swapaxes(rotations, 1, 2)
    daba_cameras = np.empty((cameras.shape[0], 3, 5), dtype=np.float64)
    daba_cameras[:, :, :3] = transposed_rotations
    daba_cameras[:, :, 3] = np.einsum(
        "nij,nj->ni", transposed_rotations, cameras[:, 3:6])
    initial_focal = initial_cameras[:, 6]
    daba_cameras[:, 0, 4] = cameras[:, 6] / initial_focal
    daba_cameras[:, 1, 4] = cameras[:, 7] * initial_focal**2
    daba_cameras[:, 2, 4] = cameras[:, 8] * initial_focal**4
    return daba_cameras, -points


def daba_matrix_state_to_bal(
    initial_cameras,
    daba_cameras,
    daba_points,
    camera_indices,
    point_indices,
    observations,
):
    """Transfer DABA geometry and fit BAL's forward radial intrinsics."""
    initial_cameras = np.asarray(initial_cameras, dtype=np.float64)
    daba_cameras = np.asarray(daba_cameras, dtype=np.float64)
    daba_points = np.asarray(daba_points, dtype=np.float64)
    if daba_cameras.shape != (len(initial_cameras), 3, 5):
        raise ValueError("DABA camera state has an invalid shape")
    rotations = np.swapaxes(daba_cameras[:, :, :3], 1, 2)
    cameras = np.empty((len(initial_cameras), 9), dtype=np.float64)
    cameras[:, :3] = Rotation.from_matrix(rotations).as_rotvec()
    cameras[:, 3:6] = np.einsum(
        "nij,nj->ni", rotations, daba_cameras[:, :, 3]
    )
    points = -daba_points

    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    point_indices = np.asarray(point_indices, dtype=np.int64)
    observations = np.asarray(observations, dtype=np.float64)
    camera_points = angle_axis_rotate_points(
        cameras[camera_indices, :3], points[point_indices]
    ) + cameras[camera_indices, 3:6]
    normalized = -camera_points[:, :2] / camera_points[:, 2, None]
    radius_squared = np.sum(normalized**2, axis=1)
    radial_basis = np.column_stack((
        np.ones_like(radius_squared),
        radius_squared,
        radius_squared**2,
    ))
    for camera_index in range(len(cameras)):
        selected = camera_indices == camera_index
        design = (normalized[selected, :, None]
                  * radial_basis[selected, None, :]).reshape(-1, 3)
        coefficients, *_ = np.linalg.lstsq(
            design, observations[selected].reshape(-1), rcond=None
        )
        cameras[camera_index, 6] = coefficients[0]
        cameras[camera_index, 7:] = coefficients[1:] / coefficients[0]
    return cameras, points


def save_bal_state(path, cameras, points, metadata=None):
    """Save a solver state and compact provenance in a portable NPZ file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=output_path.name + ".",
        suffix=".tmp",
        dir=output_path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            np.savez_compressed(
                output_file,
                cameras=np.asarray(cameras, dtype=np.float64),
                points=np.asarray(points, dtype=np.float64),
                metadata_json=np.asarray(
                    json.dumps(metadata or {}, sort_keys=True)
                ),
            )
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def read_bal_problem(path):
    """Read raw BAL cameras, points, indices, and observations."""
    input_path = Path(path)
    opener = bz2.open if input_path.suffix == ".bz2" else open
    with opener(input_path, "rt") as input_file:
        camera_count, point_count, observation_count = map(
            int, input_file.readline().split())
        camera_indices = np.empty(observation_count, dtype=np.int64)
        point_indices = np.empty(observation_count, dtype=np.int64)
        observations = np.empty((observation_count, 2), dtype=np.float64)
        for index in range(observation_count):
            camera_index, point_index, x_value, y_value = input_file.readline().split()
            camera_indices[index] = int(camera_index)
            point_indices[index] = int(point_index)
            observations[index] = (float(x_value), float(y_value))
        cameras = np.fromiter(
            (float(input_file.readline()) for _ in range(9 * camera_count)),
            dtype=np.float64,
            count=9 * camera_count,
        ).reshape(camera_count, 9)
        points = np.fromiter(
            (float(input_file.readline()) for _ in range(3 * point_count)),
            dtype=np.float64,
            count=3 * point_count,
        ).reshape(point_count, 3)
    return cameras, points, camera_indices, point_indices, observations


def canonicalize_bal_problem(
    cameras, points, camera_indices, observations, normalize_scene=True
):
    """Apply the focal-sign and scene normalization used by client_acc.py."""
    cameras = np.asarray(cameras, dtype=np.float64).copy()
    points = np.asarray(points, dtype=np.float64).copy()
    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    observations = np.asarray(observations, dtype=np.float64).copy()

    flipped_cameras = cameras[:, 6] < 0
    cameras[flipped_cameras, 6] *= -1
    observations[flipped_cameras[camera_indices]] *= -1

    if not normalize_scene:
        return cameras, points, observations

    median = np.median(points, axis=0)
    points -= median
    camera_centers = -angle_axis_rotate_points(-cameras[:, :3], cameras[:, 3:6])
    camera_centers -= median
    scene_scale = np.percentile(np.linalg.norm(points, axis=1), 95)
    if not np.isfinite(scene_scale) or scene_scale <= np.finfo(float).eps:
        raise ValueError("cannot normalize a scene with zero spatial extent")
    scale = 100.0 / scene_scale
    points *= scale
    camera_centers *= scale
    cameras[:, 3:6] = -angle_axis_rotate_points(
        cameras[:, :3], camera_centers)
    return cameras, points, observations


def write_bal_problem(
    path, cameras, points, camera_indices, point_indices, observations
):
    """Write an uncompressed BAL problem without changing parameter values."""
    with open(path, "wt") as output:
        output.write(f"{len(cameras)} {len(points)} {len(observations)}\n")
        for camera_index, point_index, observation in zip(
            camera_indices, point_indices, observations
        ):
            output.write(
                f"{int(camera_index)} {int(point_index)} "
                f"{observation[0]:.17g} {observation[1]:.17g}\n")
        for value in np.asarray(cameras).ravel():
            output.write(f"{value:.17g}\n")
        for value in np.asarray(points).ravel():
            output.write(f"{value:.17g}\n")


def read_ceres_text_state(path):
    """Read the compact text state emitted by ceres_bal_runner."""
    with open(path, "rt") as input_file:
        camera_count, point_count = map(int, input_file.readline().split())
        values = np.fromiter(
            (float(line) for line in input_file),
            dtype=np.float64,
            count=9 * camera_count + 3 * point_count,
        )
    expected_values = 9 * camera_count + 3 * point_count
    if values.size != expected_values:
        raise ValueError(
            f"Ceres state contains {values.size} values; expected {expected_values}")
    camera_end = 9 * camera_count
    return (
        values[:camera_end].reshape(camera_count, 9),
        values[camera_end:].reshape(point_count, 3),
    )


def read_daba_ceres_state(path):
    """Read the column-major camera state emitted by daba_ceres_bal_runner."""
    with open(path, "rt") as input_file:
        camera_count, point_count = map(int, input_file.readline().split())
        values = np.fromiter(
            (float(line) for line in input_file), dtype=np.float64)
    expected_values = 15 * camera_count + 3 * point_count
    if values.size != expected_values:
        raise ValueError(
            f"DABA Ceres state contains {values.size} values; "
            f"expected {expected_values}")
    camera_end = 15 * camera_count
    cameras = values[:camera_end].reshape(
        camera_count, 5, 3).transpose(0, 2, 1)
    points = values[camera_end:].reshape(point_count, 3)
    return cameras, points


def read_daba_native_state(path):
    """Read a one-rank state emitted by DABA's mpi_daba_bal_dataset."""
    with open(path, "rt") as input_file:
        camera_count, intrinsic_count, point_count, rank_count = map(
            int, input_file.readline().split()
        )
        rank_cameras, rank_intrinsics, rank_points = map(
            int, input_file.readline().split()
        )
        values = np.fromiter(
            (float(value) for line in input_file for value in line.split()),
            dtype=np.float64,
        )
    if rank_count != 1:
        raise ValueError("only one-rank native DABA states are supported")
    if (rank_cameras, rank_intrinsics, rank_points) != (
        camera_count,
        intrinsic_count,
        point_count,
    ):
        raise ValueError("native DABA rank sizes do not match global sizes")
    expected_values = 12 * camera_count + 3 * intrinsic_count + 3 * point_count
    if values.size != expected_values:
        raise ValueError(
            f"native DABA state contains {values.size} values; "
            f"expected {expected_values}"
        )
    extrinsic_end = 12 * camera_count
    intrinsic_end = extrinsic_end + 3 * intrinsic_count
    extrinsics = values[:extrinsic_end].reshape(camera_count, 3, 4)
    intrinsics = values[extrinsic_end:intrinsic_end].reshape(intrinsic_count, 3)
    if intrinsic_count != camera_count:
        raise ValueError("native DABA state must have one intrinsic per camera")
    cameras = np.empty((camera_count, 3, 5), dtype=np.float64)
    cameras[:, :, :4] = extrinsics
    cameras[:, :, 4] = intrinsics
    points = values[intrinsic_end:].reshape(point_count, 3)
    return cameras, points


def evaluate_daba_state_pixel_error(
    initial_cameras,
    daba_cameras,
    daba_points,
    camera_indices,
    point_indices,
    observations,
):
    """Invert DABA's radial ray model and report standard pixel errors."""
    initial_cameras = np.asarray(initial_cameras, dtype=np.float64)
    daba_cameras = np.asarray(daba_cameras, dtype=np.float64)
    daba_points = np.asarray(daba_points, dtype=np.float64)
    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    point_indices = np.asarray(point_indices, dtype=np.int64)
    observations = np.asarray(observations, dtype=np.float64)

    observed_cameras = daba_cameras[camera_indices]
    distances = daba_points[point_indices] - observed_cameras[:, :, 3]
    camera_rays = np.einsum(
        "nij,nj->ni",
        np.swapaxes(observed_cameras[:, :, :3], 1, 2),
        distances,
    )
    planar_rays = camera_rays[:, :2]
    planar_norms = np.linalg.norm(planar_rays, axis=1)
    height_ratios = camera_rays[:, 2] / np.maximum(planar_norms, 1e-30)
    intrinsics = observed_cameras[:, :, 4]
    normalized_observations = observations / -initial_cameras[
        camera_indices, 6, None]
    observed_signed_radii = np.sum(
        normalized_observations * planar_rays, axis=1
    ) / np.maximum(planar_norms, 1e-30)
    radii = observed_signed_radii.copy()

    for _ in range(60):
        constant, quadratic, quartic = (
            intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2])
        polynomial = (
            constant + quadratic * radii**2 + quartic * radii**4
            - height_ratios * radii)
        derivative = (
            2 * quadratic * radii + 4 * quartic * radii**3
            - height_ratios)
        step = np.divide(
            polynomial,
            derivative,
            out=np.zeros_like(polynomial),
            where=np.abs(derivative) > 1e-20,
        )
        limit = 0.5 * np.maximum(np.abs(radii), 1e-3)
        radii -= np.clip(step, -limit, limit)

    polynomial_residual = np.abs(
        intrinsics[:, 0]
        + intrinsics[:, 1] * radii**2
        + intrinsics[:, 2] * radii**4
        - height_ratios * radii)
    unresolved = np.flatnonzero(polynomial_residual > 1e-8)
    noninvertible_count = 0
    for index in unresolved:
        coefficients = [
            intrinsics[index, 2],
            0.0,
            intrinsics[index, 1],
            -height_ratios[index],
            intrinsics[index, 0],
        ]
        roots = np.roots(coefficients)
        real_roots = roots.real[np.abs(roots.imag) < 1e-7]
        if real_roots.size:
            radii[index] = real_roots[np.argmin(
                np.abs(real_roots - observed_signed_radii[index]))]
            continue
        noninvertible_count += 1
        derivative_roots = np.roots([
            4 * intrinsics[index, 2],
            0.0,
            2 * intrinsics[index, 1],
            -height_ratios[index],
        ])
        candidates = derivative_roots.real[
            np.abs(derivative_roots.imag) < 1e-7]
        if candidates.size:
            values = np.abs(
                intrinsics[index, 0]
                + intrinsics[index, 1] * candidates**2
                + intrinsics[index, 2] * candidates**4
                - height_ratios[index] * candidates)
            radii[index] = candidates[np.argmin(values)]

    predicted_normalized = np.divide(
        radii[:, None] * planar_rays,
        planar_norms[:, None],
        out=np.zeros_like(planar_rays),
        where=planar_norms[:, None] > 1e-30,
    )
    predicted = -initial_cameras[camera_indices, 6, None] * predicted_normalized
    errors = np.linalg.norm(predicted - observations, axis=1)
    return {
        "observationCount": int(errors.size),
        "meanReprojectionError": float(np.mean(errors)),
        "medianReprojectionError": float(np.median(errors)),
        "p90ReprojectionError": float(np.percentile(errors, 90)),
        "p95ReprojectionError": float(np.percentile(errors, 95)),
        "rmsePerObservation": float(np.sqrt(np.mean(errors**2))),
        "maxReprojectionError": float(np.max(errors)),
        "noninvertibleObservationCount": int(noninvertible_count),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bal_problem")
    parser.add_argument("--state", help="Ceres text state; default is BAL initial state")
    parser.add_argument("--huber-delta", type=float)
    parser.add_argument(
        "--write-canonical",
        help="write the exact focal-sign/scene-normalized problem used by DRS",
    )
    arguments = parser.parse_args()
    cameras, points, camera_indices, point_indices, observations = read_bal_problem(
        arguments.bal_problem)
    if arguments.write_canonical:
        cameras, points, observations = canonicalize_bal_problem(
            cameras, points, camera_indices, observations)
        write_bal_problem(
            arguments.write_canonical,
            cameras,
            points,
            camera_indices,
            point_indices,
            observations,
        )
    if arguments.state:
        cameras, points = read_ceres_text_state(arguments.state)
    metrics = evaluate_bal_state(
        cameras,
        points,
        camera_indices,
        point_indices,
        observations,
        huber_delta=arguments.huber_delta,
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
