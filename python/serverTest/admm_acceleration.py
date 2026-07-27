"""Safeguarded outer-center acceleration helpers for consensus ADMM."""

import numpy as np


def nesterov_coefficient(iteration, reset_iteration):
    """Return the mature DRS momentum schedule after the latest restart."""
    relative_iteration = iteration - reset_iteration
    if relative_iteration <= 1:
        return 0.0
    return (relative_iteration - 1.0) / (relative_iteration + 2.0)


def extrapolate_centers(current_centers, previous_centers, coefficient):
    current_centers = np.asarray(current_centers, dtype=np.float64)
    previous_centers = np.asarray(previous_centers, dtype=np.float64)
    if current_centers.shape != previous_centers.shape:
        raise ValueError("center arrays must have matching shapes")
    if not np.isfinite(coefficient) or not 0.0 <= coefficient < 1.0:
        raise ValueError("acceleration coefficient must be in [0, 1)")
    return current_centers + coefficient * (
        current_centers - previous_centers)


def consensus_disagreement_squared(local_cameras, consensus, camera_masks):
    local_cameras = np.asarray(local_cameras, dtype=np.float64)
    consensus = np.asarray(consensus, dtype=np.float64)
    camera_masks = np.asarray(camera_masks, dtype=bool)
    disagreement = camera_masks[:, :, None] * (
        local_cameras - consensus[None, :, :])
    return float(np.sum(disagreement * disagreement))


def augmented_consensus_merit(pixel_sse, disagreement_squared, penalty):
    """Return a global objective-plus-consensus merit for trial comparison."""
    if not all(np.isfinite(value) for value in (
        pixel_sse, disagreement_squared, penalty
    )):
        return float("inf")
    if pixel_sse < 0 or disagreement_squared < 0 or penalty <= 0:
        raise ValueError("merit inputs must be nonnegative and penalty positive")
    return pixel_sse + penalty * disagreement_squared


def should_fallback_acceleration(
    candidate_sse,
    reference_sse,
    best_sse,
    candidate_merit,
    reference_merit,
    cost_ratio,
    merit_ratio,
    best_cost_ratio,
):
    """Reject acceleration only when both relative quality signals worsen."""
    values = (
        candidate_sse,
        reference_sse,
        best_sse,
        candidate_merit,
        reference_merit,
        cost_ratio,
        merit_ratio,
        best_cost_ratio,
    )
    if not all(np.isfinite(value) for value in values):
        return True
    if reference_sse <= 0 or best_sse <= 0 or reference_merit <= 0:
        raise ValueError("acceleration references must be nonnegative")
    if cost_ratio < 1.0 or merit_ratio < 1.0 or best_cost_ratio < 1.0:
        raise ValueError("acceleration ratios must be at least one")
    return (
        candidate_sse > best_cost_ratio * best_sse
        or (
            candidate_sse > cost_ratio * reference_sse
            and candidate_merit > merit_ratio * reference_merit
        )
    )