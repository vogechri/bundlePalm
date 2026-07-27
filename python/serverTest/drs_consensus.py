"""Product-space Douglas-Rachford consensus operators."""

from dataclasses import dataclass

import numpy as np

from drs_consensus_metrics import (
    CONSENSUS_METRIC_MODES,
    reduce_camera_metric_blocks,
)


@dataclass(frozen=True)
class DrsResiduals:
    fixed_point_squared: float
    proximal_displacement_squared: float
    reflection_projection_squared: float
    center_step_squared: float


def validate_drs_arrays(local_cameras, centers, camera_masks, previous_consensus):
    local_cameras = np.asarray(local_cameras, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    camera_masks = np.asarray(camera_masks, dtype=bool)
    previous_consensus = np.asarray(previous_consensus, dtype=np.float64)
    if local_cameras.ndim != 3:
        raise ValueError("local_cameras must have shape (clusters, cameras, parameters)")
    if centers.shape != local_cameras.shape:
        raise ValueError("centers must match local_cameras")
    if camera_masks.shape != local_cameras.shape[:2]:
        raise ValueError("camera_masks must have shape (clusters, cameras)")
    if previous_consensus.shape != local_cameras.shape[1:]:
        raise ValueError("previous_consensus must have shape (cameras, parameters)")
    if np.any(np.sum(camera_masks, axis=0) == 0):
        raise ValueError("every camera must occur in at least one cluster")
    return local_cameras, centers, camera_masks, previous_consensus


def reduce_metric_tensor(metric_blocks, camera_masks, mode, parameter_count=9):
    """Reduce active raw camera blocks to one selected metric tensor."""
    camera_masks = np.asarray(camera_masks, dtype=bool)
    if mode not in CONSENSUS_METRIC_MODES:
        raise ValueError(f"unknown consensus metric mode: {mode}")
    if metric_blocks is None:
        if mode != "arithmetic":
            raise ValueError("non-arithmetic consensus requires camera metric blocks")
        reduced = np.broadcast_to(
            np.eye(parameter_count),
            camera_masks.shape + (parameter_count, parameter_count),
        ).copy()
        reduced[~camera_masks] = 0.0
        return reduced
    metric_blocks = np.asarray(metric_blocks, dtype=np.float64)
    if metric_blocks.shape != camera_masks.shape + (9, 9):
        raise ValueError(
            "metric_blocks must have shape (clusters, cameras, 9, 9)"
        )
    reduced = np.zeros_like(metric_blocks)
    reduced[camera_masks] = reduce_camera_metric_blocks(
        metric_blocks[camera_masks], mode
    )
    return reduced


def metric_quadratic_sum(vectors, metric_blocks, camera_masks):
    """Return the sum of active per-copy block quadratic forms."""
    vectors = np.asarray(vectors, dtype=np.float64)
    camera_masks = np.asarray(camera_masks, dtype=bool)
    metric_blocks = np.asarray(metric_blocks, dtype=np.float64)
    if vectors.shape[:2] != camera_masks.shape:
        raise ValueError("vectors must have shape (clusters, cameras, parameters)")
    if metric_blocks.shape != camera_masks.shape + (vectors.shape[2],) * 2:
        raise ValueError("metric blocks do not match vectors")
    active_vectors = vectors[camera_masks]
    active_blocks = metric_blocks[camera_masks]
    return float(np.einsum("bi,bij,bj->", active_vectors, active_blocks, active_vectors))


def project_consensus(
    values, camera_masks, previous_consensus, metric_blocks=None
):
    """Project product-space camera copies onto block-metric consensus."""
    values = np.asarray(values, dtype=np.float64)
    camera_masks = np.asarray(camera_masks, dtype=bool)
    previous_consensus = np.asarray(previous_consensus, dtype=np.float64)
    if metric_blocks is None:
        metric_blocks = reduce_metric_tensor(
            None, camera_masks, "arithmetic", values.shape[2]
        )
    metric_blocks = np.asarray(metric_blocks, dtype=np.float64)
    denominator = np.sum(camera_masks, axis=0)
    if np.any(denominator == 0):
        raise ValueError("every camera must occur in at least one cluster")
    metric_sum = np.sum(metric_blocks, axis=0)
    right_hand_side = np.einsum("kcij,kcj->ci", metric_blocks, values)
    consensus = previous_consensus.copy()
    for camera in range(values.shape[1]):
        consensus[camera] = np.linalg.solve(
            metric_sum[camera], right_hand_side[camera]
        )
    return consensus


def drs_step(
    local_cameras,
    centers,
    camera_masks,
    previous_consensus,
    relaxation=1.0,
    metric_blocks=None,
    metric_mode="arithmetic",
):
    """Apply prox-G to the reflection and update the DRS fixed-point center.

    The caller supplies ``u = prox_F(s)`` as ``local_cameras``. This function
    computes ``v = P_C(2u-s)`` and ``s_next = s + relaxation * (v-u)``.
    """
    if not np.isfinite(relaxation) or not 0.0 < relaxation < 2.0:
        raise ValueError("relaxation must be finite and in (0, 2)")
    local_cameras, centers, camera_masks, previous_consensus = validate_drs_arrays(
        local_cameras, centers, camera_masks, previous_consensus
    )
    active = camera_masks[:, :, None]
    selected_metrics = reduce_metric_tensor(
        metric_blocks, camera_masks, metric_mode, local_cameras.shape[2]
    )
    reflected = 2.0 * local_cameras - centers
    consensus = project_consensus(
        reflected, camera_masks, previous_consensus, selected_metrics
    )
    fixed_point = active * (local_cameras - consensus[None, :, :])
    proximal_displacement = active * (local_cameras - centers)
    reflection_projection = active * (reflected - consensus[None, :, :])
    center_step = -relaxation * fixed_point
    next_centers = centers + center_step
    residuals = DrsResiduals(
        fixed_point_squared=metric_quadratic_sum(
            fixed_point, selected_metrics, camera_masks
        ),
        proximal_displacement_squared=metric_quadratic_sum(
            proximal_displacement, selected_metrics, camera_masks
        ),
        reflection_projection_squared=metric_quadratic_sum(
            reflection_projection, selected_metrics, camera_masks
        ),
        center_step_squared=metric_quadratic_sum(
            center_step, selected_metrics, camera_masks
        ),
    )
    return consensus, next_centers, reflected, residuals, selected_metrics


def dre_splitting_term(
    local_cameras,
    consensus,
    centers,
    camera_masks,
    penalty=1.0,
    metric_blocks=None,
):
    """Return only the quadratic DRE splitting term, excluding F(u)."""
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("penalty must be finite and positive")
    active = np.asarray(camera_masks, dtype=bool)[:, :, None]
    local_cameras = np.asarray(local_cameras, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    consensus = np.asarray(consensus, dtype=np.float64)
    if metric_blocks is None:
        metric_blocks = reduce_metric_tensor(
            None, camera_masks, "arithmetic", local_cameras.shape[2]
        )
    metric_blocks = np.asarray(metric_blocks, dtype=np.float64)
    u_minus_v = active * (local_cameras - consensus[None, :, :])
    u_minus_s = active * (local_cameras - centers)
    return 0.5 * penalty * float(
        np.einsum(
            "kci,kcij,kcj->",
            u_minus_v,
            metric_blocks,
            u_minus_v + 2.0 * u_minus_s,
        )
    )


def recover_local_data_objective(
    proximal_objective, penalty, proximal_displacement_squared
):
    """Recover F(u) from F(u) + rho * ||u-s||^2."""
    if not np.isfinite(proximal_objective):
        raise ValueError("proximal objective must be finite")
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("penalty must be finite and positive")
    if (
        not np.isfinite(proximal_displacement_squared)
        or proximal_displacement_squared < 0.0
    ):
        raise ValueError("proximal displacement must be finite and nonnegative")
    return float(
        proximal_objective - penalty * proximal_displacement_squared
    )


def complete_douglas_rachford_envelope(
    local_data_objective, splitting_term, consensus_objective
):
    """Return the model envelope and its f(v) sandwich used by client_acc.py."""
    values = np.asarray(
        [local_data_objective, splitting_term, consensus_objective],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("DRE inputs must be finite")
    model_envelope = float(local_data_objective + splitting_term)
    return model_envelope, max(model_envelope, float(consensus_objective))


def reset_to_consensus(consensus, cluster_count):
    """Return coherent DRS local copies and centers with s = u = v."""
    consensus = np.asarray(consensus, dtype=np.float64)
    if consensus.ndim != 2 or cluster_count <= 0:
        raise ValueError("consensus must be a matrix and cluster_count positive")
    copies = np.repeat(consensus[None, :, :], cluster_count, axis=0)
    return copies.copy(), copies.copy(), consensus.copy()
