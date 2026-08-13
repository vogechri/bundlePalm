import numpy as np
import pytest

from admm_scaling import (
    block_jacobi_coordinate_maps,
    camera_coordinate_scale_values,
    clip_parameterwise_percentiles,
    compute_initial_jacobi_scaling,
    compute_initial_ruiz_scaling,
    normalize_geometric_mean,
    symmetric_ruiz_scaling_from_blocks,
    to_physical_cameras,
    to_scaled_cameras,
)
from bal_evaluator import project_bal


def test_jacobi_scaling_is_positive_normalized_and_matches_finite_difference():
    cameras = np.array([
        [0.01, -0.02, 0.03, 0.1, -0.2, 0.3, 800.0, 1e-4, -1e-7],
        [-0.02, 0.01, 0.04, -0.1, 0.1, 0.2, 900.0, -2e-4, 2e-7],
    ])
    points = np.array([
        [0.2, -0.1, -4.0],
        [-0.3, 0.4, -5.0],
        [0.1, 0.2, -6.0],
    ])
    camera_indices = np.array([0, 0, 1, 1])
    point_indices = np.array([0, 1, 1, 2])

    scaling = compute_initial_jacobi_scaling(
        cameras, points, camera_indices, point_indices, chunk_size=2)

    assert np.all(np.isfinite(scaling))
    assert np.all(scaling > 0.0)
    np.testing.assert_allclose(np.exp(np.mean(np.log(scaling))), 1.0)

    diagonal = np.zeros_like(cameras)
    for camera in range(cameras.shape[0]):
        for parameter in range(cameras.shape[1]):
            step = 1e-6 * max(1.0, abs(cameras[camera, parameter]))
            plus = cameras.copy()
            minus = cameras.copy()
            plus[camera, parameter] += step
            minus[camera, parameter] -= step
            derivative = (
                project_bal(plus, points, camera_indices, point_indices)
                - project_bal(minus, points, camera_indices, point_indices)
            ) / (2.0 * step)
            selected = camera_indices == camera
            diagonal[camera, parameter] = np.sum(derivative[selected] ** 2)
    expected = np.sqrt(diagonal)
    expected /= np.exp(np.mean(np.log(expected)))
    np.testing.assert_allclose(scaling, expected, rtol=2e-5, atol=1e-7)


def test_camera_scaling_round_trip():
    cameras = np.arange(18, dtype=np.float64).reshape(2, 9) - 4.0
    scaling = np.geomspace(1e-3, 1e3, 18).reshape(2, 9)

    scaled = to_scaled_cameras(cameras, scaling)

    np.testing.assert_allclose(to_physical_cameras(scaled, scaling), cameras)


def test_camera_block_transform_round_trip():
    cameras = np.arange(18, dtype=np.float64).reshape(2, 9) - 4.0
    generator = np.random.default_rng(7)
    factors = generator.standard_normal((2, 9, 9))
    transforms = np.einsum("...ji,...jk->...ik", factors, factors)
    transforms += 0.5 * np.eye(9)[None, :, :]

    scaled = to_scaled_cameras(cameras, transforms)

    np.testing.assert_allclose(
        to_physical_cameras(scaled, transforms), cameras, rtol=1e-12, atol=1e-12
    )

    repeated = np.repeat(cameras[None, :, :], 3, axis=0)
    repeated_scaled = to_scaled_cameras(repeated, transforms)
    np.testing.assert_allclose(
        to_physical_cameras(repeated_scaled, transforms), repeated,
        rtol=1e-12, atol=1e-12,
    )


def test_camera_coordinate_map_rejects_invalid_shape():
    with pytest.raises(ValueError, match="camera coordinate map"):
        to_scaled_cameras(np.zeros((2, 9)), np.ones((2, 8, 8)))


def test_block_coordinate_scale_summary_uses_inverse_singular_values():
    transforms = np.stack((np.eye(9), 2.0 * np.eye(9)))
    values = camera_coordinate_scale_values(transforms)
    np.testing.assert_allclose(values[0], 1.0)
    np.testing.assert_allclose(values[1], 0.5)


def test_ruiz_scaling_matches_jacobi_for_diagonal_blocks():
    diagonal = np.geomspace(1e-8, 1e8, 18).reshape(2, 9)
    blocks = np.zeros((2, 9, 9))
    blocks[:, np.arange(9), np.arange(9)] = diagonal

    scaling = symmetric_ruiz_scaling_from_blocks(blocks)

    np.testing.assert_allclose(scaling, np.sqrt(diagonal), rtol=1e-12)


def test_ruiz_scaling_reduces_correlated_block_row_norm_spread():
    generator = np.random.default_rng(4)
    basis = generator.standard_normal((9, 9))
    weights = np.geomspace(1e-8, 1e8, 9)
    block = basis @ np.diag(weights) @ basis.T
    blocks = block[None, :, :]

    scaling = symmetric_ruiz_scaling_from_blocks(blocks)
    before = np.max(np.abs(block), axis=1)
    equilibrated = block / (scaling[0, :, None] * scaling[0, None, :])
    after = np.max(np.abs(equilibrated), axis=1)

    assert np.max(after) / np.min(after) < np.max(before) / np.min(before)


def test_initial_ruiz_scaling_is_positive_and_normalized():
    cameras = np.array([
        [0.01, -0.02, 0.03, 0.1, -0.2, 0.3, 800.0, 1e-4, -1e-7],
        [-0.02, 0.01, 0.04, -0.1, 0.1, 0.2, 900.0, -2e-4, 2e-7],
    ])
    points = np.array([
        [0.2, -0.1, -4.0],
        [-0.3, 0.4, -5.0],
        [0.1, 0.2, -6.0],
    ])
    camera_indices = np.array([0, 0, 1, 1])
    point_indices = np.array([0, 1, 1, 2])

    scaling = compute_initial_ruiz_scaling(
        cameras, points, camera_indices, point_indices, chunk_size=2
    )

    assert np.all(np.isfinite(scaling))
    assert np.all(scaling > 0.0)
    np.testing.assert_allclose(np.exp(np.mean(np.log(scaling))), 1.0)


def test_block_jacobi_maps_whiten_camera_blocks():
    generator = np.random.default_rng(9)
    factors = generator.standard_normal((3, 9, 9))
    blocks = np.einsum("...ji,...jk->...ik", factors, factors)
    blocks += 1e-3 * np.eye(9)[None, :, :]

    transforms = block_jacobi_coordinate_maps(blocks)
    transformed = np.einsum(
        "...ji,...jk,...kl->...il", transforms, blocks, transforms
    )
    normalized = transformed / np.trace(transformed, axis1=1, axis2=2)[:, None, None]

    np.testing.assert_allclose(
        normalized,
        np.broadcast_to(np.eye(9) / 9.0, normalized.shape),
        rtol=1e-9,
        atol=1e-10,
    )


def test_scaling_ratio_cap_preserves_geometric_mean():
    scaling = normalize_geometric_mean(
        np.array([1e-12, 1.0, 1e12]), maximum_ratio=1e3)

    np.testing.assert_allclose(np.exp(np.mean(np.log(scaling))), 1.0)
    assert np.max(scaling) / np.min(scaling) <= 1e3 * (1.0 + 1e-12)


def test_parameterwise_percentile_clipping_removes_only_column_outliers():
    values = np.tile(np.arange(1.0, 101.0)[:, None], (1, 9))
    values[0, 0] = 1e-20
    values[-1, 1] = 1e20
    lower = np.percentile(values, 1.0, axis=0)
    upper = np.percentile(values, 99.0, axis=0)
    clipped = clip_parameterwise_percentiles(values, 1.0)

    assert clipped[0, 0] == lower[0]
    assert clipped[-1, 1] == upper[1]
    np.testing.assert_allclose(clipped[50, 2:], values[50, 2:])
