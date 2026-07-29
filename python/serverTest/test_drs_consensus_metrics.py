import numpy as np
import pytest

from drs_consensus_metrics import (
    CAMERA_METRIC_UPPER_INDICES,
    reduce_camera_metric_blocks,
    unpack_symmetric_camera_metric_blocks,
)


def test_full_mode_preserves_blocks_without_aliasing():
    blocks = np.arange(162, dtype=np.float64).reshape(2, 9, 9)
    reduced = reduce_camera_metric_blocks(blocks, "full")
    np.testing.assert_array_equal(reduced, blocks)
    assert reduced is not blocks


def test_arithmetic_mode_returns_identity_per_copy():
    blocks = np.full((2, 9, 9), 7.0)
    reduced = reduce_camera_metric_blocks(blocks, "arithmetic")
    np.testing.assert_array_equal(reduced, np.tile(np.eye(9), (2, 1, 1)))


def test_diagonal_mode_removes_coupling_and_floors_zeros():
    blocks = np.zeros((1, 9, 9))
    blocks[0, 0, 0] = 4.0
    blocks[0, 0, 1] = 3.0
    reduced = reduce_camera_metric_blocks(blocks, "diagonal")
    assert reduced[0, 0, 1] == 0.0
    assert reduced[0, 0, 0] == 4.0
    assert np.all(np.diag(reduced[0])[1:] == pytest.approx(4e-12))


def test_scalar_mode_uses_geometric_mean_diagonal():
    diagonal = np.arange(1.0, 10.0)
    blocks = np.diag(diagonal)[None, :, :]
    reduced = reduce_camera_metric_blocks(blocks, "scalar")
    expected = np.exp(np.mean(np.log(diagonal)))
    np.testing.assert_allclose(reduced[0], expected * np.eye(9))


def test_packed_upper_camera_metrics_reconstruct_symmetric_blocks():
    rows, columns = CAMERA_METRIC_UPPER_INDICES
    packed = np.arange(90, dtype=np.float32).reshape(2, 45)
    blocks = unpack_symmetric_camera_metric_blocks(packed)

    np.testing.assert_array_equal(blocks[:, rows, columns], packed)
    np.testing.assert_array_equal(blocks, np.swapaxes(blocks, 1, 2))