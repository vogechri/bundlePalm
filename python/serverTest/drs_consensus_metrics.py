"""Camera-block reductions for DRS consensus projection ablations."""

import numpy as np


CONSENSUS_METRIC_MODES = ("arithmetic", "scalar", "diagonal", "full")


def reduce_camera_metric_blocks(blocks, mode):
    """Return SPD camera blocks for the requested projection-metric mode."""
    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.ndim != 3 or blocks.shape[1:] != (9, 9):
        raise ValueError("camera metrics must have shape (copies, 9, 9)")
    if mode not in CONSENSUS_METRIC_MODES:
        raise ValueError(f"unknown consensus metric mode: {mode}")
    if mode == "full":
        return blocks.copy()
    symmetric = 0.5 * (blocks + np.swapaxes(blocks, 1, 2))
    diagonals = np.diagonal(symmetric, axis1=1, axis2=2)
    positive_scale = np.maximum(np.max(np.abs(diagonals), axis=1), 1e-32)
    floor = 1e-12 * positive_scale
    if mode == "diagonal":
        reduced = np.zeros_like(symmetric)
        indices = np.arange(9)
        reduced[:, indices, indices] = np.maximum(diagonals, floor[:, None])
        return reduced
    if mode == "scalar":
        clipped = np.maximum(diagonals, floor[:, None])
        scalar = np.exp(np.mean(np.log(clipped), axis=1))
    else:
        scalar = np.ones(blocks.shape[0], dtype=np.float64)
    return scalar[:, None, None] * np.eye(9)[None, :, :]