"""Static camera-coordinate scaling for consensus ADMM."""

import warnings

import numpy as np
import torch


def _rotate_angle_axis(angle_axis, points):
    theta_squared = torch.sum(angle_axis * angle_axis, dim=1)
    nonzero = theta_squared > 0
    theta = torch.sqrt(torch.where(nonzero, theta_squared, torch.ones_like(theta_squared)))
    axis = angle_axis / theta[:, None]
    cosine = torch.cos(theta)[:, None]
    sine = torch.sin(theta)[:, None]
    rodrigues = (
        points * cosine
        + torch.linalg.cross(axis, points) * sine
        + axis * torch.sum(axis * points, dim=1)[:, None] * (1.0 - cosine)
    )
    first_order = points + torch.linalg.cross(angle_axis, points)
    return torch.where(nonzero[:, None], rodrigues, first_order)


def _project_observations(cameras, points):
    camera_points = _rotate_angle_axis(cameras[:, :3], points)
    camera_points = camera_points + cameras[:, 3:6]
    normalized = -camera_points[:, :2] / camera_points[:, 2, None]
    radius_squared = torch.sum(normalized * normalized, dim=1)
    distortion = 1.0 + radius_squared * (
        cameras[:, 7] + cameras[:, 8] * radius_squared
    )
    return normalized * (cameras[:, 6] * distortion)[:, None]


def normalize_geometric_mean(
    values, relative_floor=1e-12, maximum_ratio=None
):
    values = np.asarray(values, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError("camera scaling has no finite positive entries")
    floor = max(
        relative_floor * float(np.median(positive)),
        np.finfo(np.float64).tiny,
    )
    safe = np.maximum(np.where(np.isfinite(values), values, floor), floor)
    geometric_mean = float(np.exp(np.mean(np.log(safe))))
    normalized = safe / geometric_mean
    if maximum_ratio is not None:
        if not np.isfinite(maximum_ratio) or maximum_ratio < 1.0:
            raise ValueError("maximum_ratio must be finite and at least one")
        radius = np.sqrt(maximum_ratio)
        normalized = np.clip(normalized, 1.0 / radius, radius)
        normalized /= np.exp(np.mean(np.log(normalized)))
    return normalized


def clip_parameterwise_percentiles(values, clipping_percentile):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("camera scaling values must be a matrix")
    if not 0.0 <= clipping_percentile < 50.0:
        raise ValueError("clipping_percentile must be in the interval [0, 50)")
    lower = np.percentile(values, clipping_percentile, axis=0)
    upper = np.percentile(values, 100.0 - clipping_percentile, axis=0)
    return np.clip(values, lower[None, :], upper[None, :])


def compute_initial_jacobi_scaling(
    cameras,
    points,
    camera_indices,
    point_indices,
    chunk_size=200_000,
    maximum_ratio=None,
    clipping_percentile=None,
):
    """Return geometric-mean-normalized sqrt(diag(J_camera.T J_camera))."""
    cameras = np.asarray(cameras, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    point_indices = np.asarray(point_indices, dtype=np.int64)
    if camera_indices.shape != point_indices.shape:
        raise ValueError("camera_indices and point_indices must have equal shape")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    diagonal = np.zeros_like(cameras)
    for start in range(0, camera_indices.size, chunk_size):
        stop = min(start + chunk_size, camera_indices.size)
        selected_cameras = camera_indices[start:stop]
        observed_cameras = torch.tensor(
            cameras[selected_cameras], dtype=torch.float64, requires_grad=True)
        observed_points = torch.tensor(
            points[point_indices[start:stop]], dtype=torch.float64)
        projections = _project_observations(observed_cameras, observed_points)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "CUDA initialization: The NVIDIA driver on your system "
                    "is too old.*"
                ),
            )
            gradient_x = torch.autograd.grad(
                projections[:, 0].sum(), observed_cameras, retain_graph=True)[0]
            gradient_y = torch.autograd.grad(
                projections[:, 1].sum(), observed_cameras)[0]
        contribution = (
            gradient_x.square() + gradient_y.square()
        ).detach().numpy()
        np.add.at(diagonal, selected_cameras, contribution)

    scaling = np.sqrt(diagonal)
    if clipping_percentile is not None:
        scaling = clip_parameterwise_percentiles(
            scaling, clipping_percentile)
    scaling = normalize_geometric_mean(
        scaling, maximum_ratio=maximum_ratio)
    if not np.all(np.isfinite(scaling)) or np.any(scaling <= 0.0):
        raise ValueError("camera scaling must be finite and positive")
    return scaling


def symmetric_ruiz_scaling_from_blocks(
    blocks, max_iterations=10, tolerance=1e-3
):
    """Equilibrate symmetric camera blocks with diagonal Ruiz scaling."""
    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.ndim != 3 or blocks.shape[1:] != (9, 9):
        raise ValueError("camera Hessian blocks must have shape (camera_count, 9, 9)")
    if max_iterations <= 0:
        raise ValueError("Ruiz maximum iterations must be positive")
    if tolerance <= 0.0:
        raise ValueError("Ruiz tolerance must be positive")
    absolute = np.abs(blocks)
    scaling = np.ones(blocks.shape[:2], dtype=np.float64)
    for _ in range(max_iterations):
        equilibrated = absolute / (
            scaling[:, :, None] * scaling[:, None, :]
        )
        row_norms = np.max(equilibrated, axis=2)
        positive = np.where(
            np.isfinite(row_norms) & (row_norms > 0.0), row_norms, np.nan
        )
        medians = np.nanmedian(positive, axis=1)
        medians = np.where(
            np.isfinite(medians) & (medians > 0.0),
            medians,
            1.0,
        )
        floors = np.maximum(
            1e-12 * medians, np.finfo(np.float64).tiny
        )
        safe_norms = np.maximum(
            np.where(np.isfinite(row_norms), row_norms, floors[:, None]),
            floors[:, None],
        )
        update = np.sqrt(safe_norms)
        scaling *= update
        if np.max(np.abs(np.log(update))) < tolerance:
            break
    return scaling


def compute_initial_ruiz_scaling(
    cameras,
    points,
    camera_indices,
    point_indices,
    chunk_size=200_000,
    maximum_ratio=None,
    clipping_percentile=None,
    max_iterations=10,
    tolerance=1e-3,
):
    """Return full-camera-block-informed diagonal symmetric Ruiz scaling."""
    cameras = np.asarray(cameras, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    point_indices = np.asarray(point_indices, dtype=np.int64)
    if camera_indices.shape != point_indices.shape:
        raise ValueError("camera_indices and point_indices must have equal shape")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    blocks = np.zeros((len(cameras), 9, 9), dtype=np.float64)
    for start in range(0, camera_indices.size, chunk_size):
        stop = min(start + chunk_size, camera_indices.size)
        selected_cameras = camera_indices[start:stop]
        observed_cameras = torch.tensor(
            cameras[selected_cameras], dtype=torch.float64, requires_grad=True
        )
        observed_points = torch.tensor(
            points[point_indices[start:stop]], dtype=torch.float64
        )
        projections = _project_observations(observed_cameras, observed_points)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "CUDA initialization: The NVIDIA driver on your system "
                    "is too old.*"
                ),
            )
            gradient_x = torch.autograd.grad(
                projections[:, 0].sum(), observed_cameras, retain_graph=True
            )[0]
            gradient_y = torch.autograd.grad(
                projections[:, 1].sum(), observed_cameras
            )[0]
        gradient_x = gradient_x.detach().numpy()
        gradient_y = gradient_y.detach().numpy()
        contribution = (
            gradient_x[:, :, None] * gradient_x[:, None, :]
            + gradient_y[:, :, None] * gradient_y[:, None, :]
        )
        np.add.at(blocks, selected_cameras, contribution)

    scaling = symmetric_ruiz_scaling_from_blocks(
        blocks, max_iterations=max_iterations, tolerance=tolerance
    )
    if clipping_percentile is not None:
        scaling = clip_parameterwise_percentiles(
            scaling, clipping_percentile
        )
    scaling = normalize_geometric_mean(
        scaling, maximum_ratio=maximum_ratio
    )
    if not np.all(np.isfinite(scaling)) or np.any(scaling <= 0.0):
        raise ValueError("camera scaling must be finite and positive")
    return scaling


def to_scaled_cameras(physical_cameras, scaling):
    return np.asarray(physical_cameras, dtype=np.float64) * np.asarray(
        scaling, dtype=np.float64)


def to_physical_cameras(scaled_cameras, scaling):
    return np.asarray(scaled_cameras, dtype=np.float64) / np.asarray(
        scaling, dtype=np.float64)
