"""Residual metrics for the plain distributed Douglas-Rachford map."""

import math

import numpy as np


def drs_residual_metrics(current, mapped, local_cameras, consensus, metrics):
    """Measure the plain DRS fixed-point residual in Euclidean and metric norms."""
    residual = mapped - current
    residual_metric_squared = 0.0
    consensus_squared = 0.0
    consensus_metric_squared = 0.0
    for cluster_id, metric in enumerate(metrics):
        start = cluster_id * consensus.size
        stop = start + consensus.size
        cluster_residual = residual[start:stop]
        disagreement = (consensus - local_cameras[cluster_id]).reshape(-1)
        residual_metric_squared += float(cluster_residual.dot(metric * cluster_residual))
        consensus_squared += float(disagreement.dot(disagreement))
        consensus_metric_squared += float(disagreement.dot(metric * disagreement))

    residual_norm = float(np.linalg.norm(residual))
    current_norm = float(np.linalg.norm(current))
    scale = max(current_norm, np.finfo(float).eps)
    return {
        "fixed_point_residual_norm": residual_norm,
        "fixed_point_residual_relative": residual_norm / scale,
        "fixed_point_residual_metric_norm": math.sqrt(max(residual_metric_squared, 0.0)),
        "consensus_residual_norm": math.sqrt(max(consensus_squared, 0.0)),
        "consensus_residual_metric_norm": math.sqrt(max(consensus_metric_squared, 0.0)),
    }
