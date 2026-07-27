"""Reference scaled-ADMM camera consensus updates."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdmmResiduals:
    primal_squared: float
    dual_squared: float


def validate_admm_arrays(local_cameras, scaled_duals, camera_masks, consensus):
    local_cameras = np.asarray(local_cameras, dtype=np.float64)
    scaled_duals = np.asarray(scaled_duals, dtype=np.float64)
    camera_masks = np.asarray(camera_masks, dtype=bool)
    consensus = np.asarray(consensus, dtype=np.float64)
    if local_cameras.ndim != 3:
        raise ValueError("local_cameras must have shape (clusters, cameras, parameters)")
    if scaled_duals.shape != local_cameras.shape:
        raise ValueError("scaled_duals must match local_cameras")
    if camera_masks.shape != local_cameras.shape[:2]:
        raise ValueError("camera_masks must have shape (clusters, cameras)")
    if consensus.shape != local_cameras.shape[1:]:
        raise ValueError("consensus must have shape (cameras, parameters)")
    if np.any(np.sum(camera_masks, axis=0) == 0):
        raise ValueError("every camera must occur in at least one cluster")
    return local_cameras, scaled_duals, camera_masks, consensus


def consensus_update(
    local_cameras,
    scaled_duals,
    camera_masks,
    previous_consensus,
    penalties=None,
):
    """Compute DABA's occurrence-aware arithmetic camera reference."""
    local_cameras, scaled_duals, camera_masks, previous_consensus = (
        validate_admm_arrays(
            local_cameras, scaled_duals, camera_masks, previous_consensus))
    if penalties is None:
        weights = camera_masks.astype(np.float64)
    else:
        penalties = np.asarray(penalties, dtype=np.float64)
        if penalties.shape == (local_cameras.shape[0],):
            penalties = penalties[:, None]
        if penalties.shape not in (
            (local_cameras.shape[0], 1), camera_masks.shape
        ):
            raise ValueError("penalties must be per cluster or cluster-camera")
        if np.any(penalties <= 0) or not np.all(np.isfinite(penalties)):
            raise ValueError("ADMM penalties must be finite and positive")
        weights = camera_masks * penalties
    denominator = np.sum(weights, axis=0)
    consensus = np.sum(
        weights[:, :, None] * local_cameras, axis=0
    ) / denominator[:, None]
    return consensus


def dual_update(
    scaled_duals, local_cameras, consensus, camera_masks, alpha=1.5
):
    if not 0 < alpha < 2:
        raise ValueError("ADMM over-relaxation alpha must be in (0, 2)")
    scaled_duals = np.asarray(scaled_duals, dtype=np.float64)
    local_cameras = np.asarray(local_cameras, dtype=np.float64)
    camera_masks = np.asarray(camera_masks, dtype=bool)
    updated = scaled_duals.copy()
    updated += alpha * camera_masks[:, :, None] * (
        local_cameras - consensus[None, :, :])
    updated[~camera_masks] = 0.0
    return updated


def admm_residuals(
    previous_local_cameras,
    consensus,
    previous_consensus,
    camera_masks,
):
    previous_local_cameras = np.asarray(
        previous_local_cameras, dtype=np.float64)
    consensus = np.asarray(consensus, dtype=np.float64)
    previous_consensus = np.asarray(previous_consensus, dtype=np.float64)
    camera_masks = np.asarray(camera_masks, dtype=bool)
    primal = camera_masks[:, :, None] * (
        previous_local_cameras - previous_consensus[None, :, :])
    dual = camera_masks[:, :, None] * (
        consensus[None, :, :] - previous_consensus[None, :, :])
    return AdmmResiduals(
        primal_squared=float(np.sum(primal * primal)),
        dual_squared=float(np.sum(dual * dual)),
    )


def plain_drs_step(local_cameras, centers, camera_masks, previous_consensus):
    """Apply one product-space DRS projection and fixed-point update."""
    local_cameras, centers, camera_masks, previous_consensus = (
        validate_admm_arrays(
            local_cameras, centers, camera_masks, previous_consensus))
    reflected = 2.0 * local_cameras - centers
    consensus = consensus_update(
        reflected,
        np.zeros_like(reflected),
        camera_masks,
        previous_consensus,
    )
    step = camera_masks[:, :, None] * (
        consensus[None, :, :] - local_cameras)
    next_centers = centers + step
    consensus_change = camera_masks[:, :, None] * (
        consensus[None, :, :] - previous_consensus[None, :, :])
    residuals = AdmmResiduals(
        primal_squared=float(np.sum(step * step)),
        dual_squared=float(np.sum(consensus_change * consensus_change)),
    )
    return consensus, next_centers, residuals


def adapt_penalty(
    penalty,
    residuals,
    initial_penalty,
    increasing_ratio=1.5,
    decreasing_ratio=0.8,
):
    """Apply DABA ADMM's residual-balanced scalar penalty rule."""
    if penalty <= 0 or initial_penalty <= 0:
        raise ValueError("penalties must be positive")
    ratio = 1.005
    if initial_penalty * residuals.primal_squared > 2.5 * residuals.dual_squared:
        ratio = increasing_ratio
    elif residuals.dual_squared > (
        10.0 * initial_penalty * residuals.primal_squared
    ):
        ratio = decreasing_ratio
    return penalty * ratio, ratio


def rescale_scaled_duals(scaled_duals, penalty_ratio):
    if penalty_ratio <= 0 or not np.isfinite(penalty_ratio):
        raise ValueError("penalty_ratio must be finite and positive")
    return np.asarray(scaled_duals, dtype=np.float64) / penalty_ratio


def combine_proximal_terms(consensus_center, accepted_local, penalty, damping):
    """Combine ADMM and temporary recovery quadratics into one proximal term."""
    if penalty <= 0 or not np.isfinite(penalty):
        raise ValueError("penalty must be finite and positive")
    if damping < 0 or not np.isfinite(damping):
        raise ValueError("damping must be finite and nonnegative")
    consensus_center = np.asarray(consensus_center, dtype=np.float64)
    accepted_local = np.asarray(accepted_local, dtype=np.float64)
    if consensus_center.shape != accepted_local.shape:
        raise ValueError("proximal centers must have matching shapes")
    if damping == 0:
        return consensus_center, penalty
    effective_penalty = penalty + damping
    effective_center = (
        penalty * consensus_center + damping * accepted_local
    ) / effective_penalty
    return effective_center, effective_penalty
