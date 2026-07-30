import numpy as np
import pytest

from client_admm import SingleNodeConsensusSummary
from client_drs import apply_single_node_consensus, validate_worker_consensus_rhs
from drs_consensus import (
    ActiveCameraMetricBlocks,
    dre_splitting_term,
    drs_step,
    project_consensus,
)


def _shadow_fixture():
    cluster_indices = np.array([0, 0, 1, 1], dtype=np.uint16)
    camera_indices = np.array([0, 1, 0, 1], dtype=np.uint16)
    diagonal = np.array([
        np.arange(1.0, 10.0),
        np.arange(2.0, 11.0),
        np.arange(3.0, 12.0),
        np.arange(4.0, 13.0),
    ])
    blocks = np.zeros((4, 9, 9), dtype=np.float64)
    rows = np.arange(9)
    blocks[:, rows, rows] = diagonal
    metrics = ActiveCameraMetricBlocks(
        cluster_indices, camera_indices, blocks, 2, 2
    )
    local = np.arange(36.0).reshape(2, 2, 9) / 7.0
    centers = np.arange(36.0, 72.0).reshape(2, 2, 9) / 11.0
    reflection = 2.0 * local - centers
    rhs = np.einsum(
        "bij,bj->bi",
        blocks,
        reflection[cluster_indices, camera_indices],
    )
    consensus = project_consensus(
        reflection,
        np.ones((2, 2), dtype=bool),
        np.zeros((2, 9)),
        metrics,
    )
    return local, centers, metrics, rhs, consensus


def test_worker_consensus_rhs_matches_reference_reduction():
    local, centers, metrics, rhs, consensus = _shadow_fixture()
    rhs_error, consensus_error = validate_worker_consensus_rhs(
        local, centers, metrics, rhs, consensus
    )
    assert rhs_error == 0.0
    assert consensus_error == 0.0


def test_worker_consensus_rhs_rejects_modified_contribution():
    local, centers, metrics, rhs, consensus = _shadow_fixture()
    rhs = rhs.copy()
    rhs[2, 4] += 1.0
    with pytest.raises(RuntimeError, match="worker consensus shadow mismatch"):
        validate_worker_consensus_rhs(
            local, centers, metrics, rhs, consensus
        )


def test_single_node_summary_matches_coordinator_state_update():
    local, centers, metrics, _, _ = _shadow_fixture()
    masks = np.ones((2, 2), dtype=bool)
    relaxation = 1.25
    consensus, next_centers, reflected, residuals, selected = drs_step(
        local,
        centers,
        masks,
        np.zeros((2, 9)),
        relaxation=relaxation,
        metric_blocks=metrics,
        metric_mode="full",
    )
    summary = SingleNodeConsensusSummary(
        consensus=consensus,
        fixed_point_squared=residuals.fixed_point_squared,
        proximal_displacement_squared=(
            residuals.proximal_displacement_squared
        ),
        reflection_projection_squared=(
            residuals.reflection_projection_squared
        ),
        center_step_squared=residuals.center_step_squared,
        splitting_term=dre_splitting_term(
            local, consensus, centers, masks, metric_blocks=selected
        ),
    )
    actual = apply_single_node_consensus(
        local, centers, masks, summary, relaxation
    )
    np.testing.assert_array_equal(actual[0], consensus)
    np.testing.assert_array_equal(actual[1], next_centers)
    np.testing.assert_array_equal(actual[2], reflected)
    assert actual[3] == residuals