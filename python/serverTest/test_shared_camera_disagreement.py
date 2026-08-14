import numpy as np

from client_admm import ClusterSchurSystem
from client_drs import (
    camera_copy_disagreement_diagnostics,
    damp_metric_projected_camera_proposals,
    damp_metric_projected_camera_proposals_hysteresis,
    damp_metric_projected_camera_subspaces,
    damp_shared_camera_disagreement,
    schur_system_observability_fractions,
    select_metric_proposal_hysteresis_scale,
    select_metric_proposal_scale,
)
from drs_consensus import project_consensus


def test_schur_observability_uses_removed_trace_per_parameter_group():
    raw = np.array([10.0 * np.eye(9)])
    schur = np.array([4.0 * np.eye(9)])
    system = ClusterSchurSystem(
        camera_ids=np.array([7]),
        block_rows=np.array([7]),
        block_columns=np.array([7]),
        blocks=schur,
        reduced_gradient=np.zeros((1, 9)),
        camera_diagonal=raw,
    )

    np.testing.assert_allclose(
        schur_system_observability_fractions([system]),
        np.full((1, 3), 0.6),
    )


def test_camera_copy_diagnostic_uses_weighted_projection_and_metric():
    local_cameras = np.zeros((2, 1, 9), dtype=np.float64)
    local_cameras[1, 0] = 4.0
    camera_masks = np.ones((2, 1), dtype=bool)
    metric_blocks = np.zeros((2, 1, 9, 9), dtype=np.float64)
    metric_blocks[0, 0] = np.eye(9)
    metric_blocks[1, 0] = 3.0 * np.eye(9)

    diagnostics = camera_copy_disagreement_diagnostics(
        local_cameras,
        camera_masks,
        np.zeros((1, 9)),
        metric_blocks,
        "full",
        np.ones((1, 9)),
        (0,),
    )

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["copies"] == 2
    assert diagnostic["translationRange"] == 4.0
    assert diagnostic["translationRms"] == np.sqrt(5.0)
    assert diagnostic["metricEnergy"] == 108.0
    assert diagnostic["metricSumEigenvalueMinimum"] == 4.0
    assert diagnostic["metricSumEigenvalueMaximum"] == 4.0
    assert diagnostic["metricSumCondition"] == 1.0


def test_disagreement_scale_one_preserves_all_camera_copies():
    local_cameras = np.arange(54, dtype=np.float64).reshape(2, 3, 9)
    original = local_cameras.copy()
    centers = np.zeros_like(local_cameras)
    camera_masks = np.array([[True, True, False], [True, False, True]])

    damp_shared_camera_disagreement(
        local_cameras, centers, camera_masks, scale=1.0
    )

    np.testing.assert_array_equal(local_cameras, original)


def test_zero_scale_preserves_mean_motion_and_inactive_copies():
    centers = np.zeros((3, 3, 9), dtype=np.float64)
    local_cameras = np.zeros_like(centers)
    camera_masks = np.array(
        [
            [True, True, False],
            [True, False, True],
            [False, False, True],
        ]
    )
    local_cameras[0, 0] = 2.0
    local_cameras[1, 0] = 6.0
    local_cameras[2, 0] = 99.0
    local_cameras[0, 1] = 3.0
    local_cameras[1, 1] = 77.0
    local_cameras[0, 2] = 88.0
    local_cameras[1, 2] = 4.0
    local_cameras[2, 2] = 8.0

    damp_shared_camera_disagreement(
        local_cameras, centers, camera_masks, scale=0.0
    )

    np.testing.assert_array_equal(local_cameras[0, 0], np.full(9, 4.0))
    np.testing.assert_array_equal(local_cameras[1, 0], np.full(9, 4.0))
    np.testing.assert_array_equal(local_cameras[1, 2], np.full(9, 6.0))
    np.testing.assert_array_equal(local_cameras[2, 2], np.full(9, 6.0))
    np.testing.assert_array_equal(local_cameras[0, 1], np.full(9, 3.0))
    np.testing.assert_array_equal(local_cameras[2, 0], np.full(9, 99.0))
    np.testing.assert_array_equal(local_cameras[1, 1], np.full(9, 77.0))
    np.testing.assert_array_equal(local_cameras[0, 2], np.full(9, 88.0))


def test_metric_proposal_damping_preserves_weighted_projection():
    local_cameras = np.zeros((2, 2, 9), dtype=np.float64)
    local_cameras[1, 0] = 4.0
    local_cameras[0, 1] = 7.0
    local_cameras[1, 1] = 99.0
    camera_masks = np.array([[True, True], [True, False]])
    metric_blocks = np.zeros((2, 2, 9, 9), dtype=np.float64)
    metric_blocks[0, 0] = np.eye(9)
    metric_blocks[1, 0] = 3.0 * np.eye(9)
    metric_blocks[0, 1] = 2.0 * np.eye(9)

    ratio, applied_scale = damp_metric_projected_camera_proposals(
        local_cameras,
        np.zeros_like(local_cameras),
        camera_masks,
        np.zeros((2, 9)),
        metric_blocks,
        "full",
        scale=0.5,
    )

    np.testing.assert_allclose(local_cameras[0, 0], 1.5)
    np.testing.assert_allclose(local_cameras[1, 0], 3.5)
    np.testing.assert_array_equal(local_cameras[0, 1], np.full(9, 7.0))
    np.testing.assert_array_equal(local_cameras[1, 1], np.full(9, 99.0))
    assert ratio > 0.0
    assert applied_scale == 0.5


def test_metric_proposal_threshold_preserves_low_disagreement_proposals():
    local_cameras = np.zeros((2, 1, 9), dtype=np.float64)
    local_cameras[1, 0] = 0.01
    original = local_cameras.copy()
    centers = np.zeros_like(local_cameras)
    camera_masks = np.ones((2, 1), dtype=bool)
    metric_blocks = np.broadcast_to(
        np.eye(9), (2, 1, 9, 9)
    ).copy()

    ratio, applied_scale = damp_metric_projected_camera_proposals(
        local_cameras,
        centers,
        camera_masks,
        np.zeros((1, 9)),
        metric_blocks,
        "full",
        scale=0.6,
        disagreement_threshold=2.0,
    )

    assert ratio < 2.0
    assert applied_scale == 1.0
    np.testing.assert_array_equal(local_cameras, original)


def test_metric_proposal_hysteresis_switches_at_both_thresholds():
    assert select_metric_proposal_hysteresis_scale(
        0.20, 0.8, 0.6, 0.8, 0.03, 0.10
    ) == 0.6
    assert select_metric_proposal_hysteresis_scale(
        0.05, 0.6, 0.6, 0.8, 0.03, 0.10
    ) == 0.6
    assert select_metric_proposal_hysteresis_scale(
        0.02, 0.6, 0.6, 0.8, 0.03, 0.10
    ) == 0.8


def test_metric_proposal_hysteresis_preserves_weighted_projection():
    local_cameras = np.zeros((2, 1, 9), dtype=np.float64)
    local_cameras[1, 0] = 4.0
    camera_masks = np.ones((2, 1), dtype=bool)
    metric_blocks = np.zeros((2, 1, 9, 9), dtype=np.float64)
    metric_blocks[0, 0] = np.eye(9)
    metric_blocks[1, 0] = 3.0 * np.eye(9)
    projection_before = project_consensus(
        local_cameras,
        camera_masks,
        np.zeros((1, 9)),
        metric_blocks,
    )

    ratio, scale = damp_metric_projected_camera_proposals_hysteresis(
        local_cameras,
        np.zeros_like(local_cameras),
        camera_masks,
        np.zeros((1, 9)),
        metric_blocks,
        "full",
        current_scale=0.8,
        strong_scale=0.6,
        normal_scale=0.8,
        low_threshold=0.03,
        high_threshold=0.10,
    )

    projection_after = project_consensus(
        local_cameras,
        camera_masks,
        np.zeros((1, 9)),
        metric_blocks,
    )
    assert ratio >= 0.10
    assert scale == 0.6
    np.testing.assert_allclose(projection_after, projection_before)


def test_single_scale_selector_is_exact_identity_without_cost_evaluation():
    local_cameras = np.arange(18, dtype=np.float64).reshape(2, 1, 9)
    metric_blocks = np.broadcast_to(np.eye(9), (2, 1, 9, 9)).copy()

    selected, scale, ratio, objective = select_metric_proposal_scale(
        local_cameras,
        np.zeros_like(local_cameras),
        np.ones((2, 1), dtype=bool),
        np.zeros((1, 9)),
        metric_blocks,
        "full",
        (1.0,),
        12.5,
        lambda _: (_ for _ in ()).throw(AssertionError("unexpected evaluation")),
    )

    np.testing.assert_array_equal(selected, local_cameras)
    assert scale == 1.0
    assert np.isnan(ratio)
    assert objective == 12.5


def test_selector_can_choose_lower_corrected_dre_damped_proposal():
    local_cameras = np.zeros((2, 1, 9), dtype=np.float64)
    local_cameras[1, 0] = 4.0
    metric_blocks = np.broadcast_to(np.eye(9), (2, 1, 9, 9)).copy()

    selected, scale, ratio, objective = select_metric_proposal_scale(
        local_cameras,
        np.zeros_like(local_cameras),
        np.ones((2, 1), dtype=bool),
        np.zeros((1, 9)),
        metric_blocks,
        "full",
        (0.6, 1.0),
        100.0,
        lambda candidate: float(np.sum(candidate**2)),
    )

    assert scale == 0.6
    assert ratio > 0.0
    assert objective == np.sum(selected**2)
    np.testing.assert_allclose(np.mean(selected, axis=0), 2.0)


def test_subspace_damping_preserves_projection_with_metric_cross_terms():
    local_cameras = np.zeros((2, 1, 9), dtype=np.float64)
    local_cameras[0, 0, :6] = [1.0, -2.0, 0.5, 3.0, -1.0, 2.0]
    local_cameras[1, 0, :6] = [-0.5, 1.0, 2.0, -2.0, 4.0, 1.0]
    camera_masks = np.ones((2, 1), dtype=bool)
    metric_blocks = np.zeros((2, 1, 9, 9), dtype=np.float64)
    for cluster, coupling in enumerate((0.2, -0.15)):
        block = np.eye(9)
        block[:3, 3:6] = coupling * np.eye(3)
        block[3:6, :3] = coupling * np.eye(3)
        metric_blocks[cluster, 0] = block
    before = project_consensus(
        local_cameras, camera_masks, np.zeros((1, 9)), metric_blocks
    )

    fractions = damp_metric_projected_camera_subspaces(
        local_cameras,
        camera_masks,
        np.zeros((1, 9)),
        metric_blocks,
        "full",
        (0.6, 1.0, 1.0),
    )
    after = project_consensus(
        local_cameras, camera_masks, np.zeros((1, 9)), metric_blocks
    )

    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.sum(fractions), 1.0)
    assert np.all(fractions >= 0.0)