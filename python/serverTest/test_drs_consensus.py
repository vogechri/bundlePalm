import numpy as np
import pytest

from admm_consensus import plain_drs_step
from drs_consensus import (
    ActiveCameraMetricBlocks,
    complete_douglas_rachford_envelope,
    dre_splitting_term,
    drs_step,
    proximal_point_step,
    recover_local_data_objective,
    reset_to_consensus,
    shared_floor_prior_blocks,
)


def test_shared_floor_prior_is_added_once_per_camera():
    camera_masks = np.array([[True], [True]])
    cluster_indices = np.array([0, 1])
    camera_indices = np.array([0, 0])
    vote = ActiveCameraMetricBlocks(
        cluster_indices, camera_indices,
        np.tile(np.eye(9), (2, 1, 1)), 2, 1,
    )
    local = ActiveCameraMetricBlocks(
        cluster_indices, camera_indices,
        np.tile(3.0 * np.eye(9), (2, 1, 1)), 2, 1,
    )

    prior = shared_floor_prior_blocks(
        local, vote, camera_masks, scale=0.25
    )

    np.testing.assert_allclose(prior[0], 0.5 * np.eye(9))


def test_drs_step_matches_writeup_equations():
    centers = np.array([[[1.0]], [[5.0]]])
    local = np.array([[[2.0]], [[4.0]]])
    masks = np.ones((2, 1), dtype=bool)

    consensus, next_centers, reflected, residuals, _ = drs_step(
        local, centers, masks, np.array([[0.0]])
    )

    np.testing.assert_allclose(reflected[:, 0, 0], [3.0, 3.0])
    np.testing.assert_allclose(consensus, [[3.0]])
    np.testing.assert_allclose(next_centers[:, 0, 0], [2.0, 4.0])
    assert residuals.fixed_point_squared == pytest.approx(2.0)
    assert residuals.proximal_displacement_squared == pytest.approx(2.0)
    assert residuals.reflection_projection_squared == pytest.approx(0.0)
    assert residuals.center_step_squared == pytest.approx(2.0)


def test_relaxed_drs_scales_only_center_update():
    centers = np.array([[[1.0]], [[5.0]]])
    local = np.array([[[2.0]], [[4.0]]])
    masks = np.ones((2, 1), dtype=bool)

    consensus, next_centers, reflected, residuals, _ = drs_step(
        local, centers, masks, np.array([[0.0]]), relaxation=1.5
    )

    np.testing.assert_allclose(reflected[:, 0, 0], [3.0, 3.0])
    np.testing.assert_allclose(consensus, [[3.0]])
    np.testing.assert_allclose(next_centers[:, 0, 0], [2.5, 3.5])
    assert residuals.center_step_squared == pytest.approx(4.5)


def test_single_cluster_proximal_point_advances_directly_to_local_solution():
    centers = np.array([[[1.0], [5.0]]])
    local = np.array([[[2.0], [3.0]]])
    masks = np.ones((1, 2), dtype=bool)

    consensus, next_centers, residuals, _ = proximal_point_step(
        local, centers, masks
    )

    np.testing.assert_allclose(consensus, local[0])
    np.testing.assert_allclose(next_centers, local)
    assert residuals.fixed_point_squared == pytest.approx(5.0)
    assert residuals.proximal_displacement_squared == pytest.approx(5.0)
    assert residuals.reflection_projection_squared == pytest.approx(0.0)
    assert residuals.center_step_squared == pytest.approx(5.0)


def test_drs_projection_ignores_absent_camera_copies():
    centers = np.zeros((2, 2, 1))
    local = np.array([[[1.0], [99.0]], [[3.0], [7.0]]])
    masks = np.array([[True, False], [True, True]])

    consensus, _, _, _, _ = drs_step(local, centers, masks, np.zeros((2, 1)))

    np.testing.assert_allclose(consensus[:, 0], [4.0, 14.0])


def test_dre_splitting_term_matches_explicit_formula():
    local = np.array([[[2.0]], [[4.0]]])
    centers = np.array([[[1.0]], [[5.0]]])
    consensus = np.array([[3.0]])
    masks = np.ones((2, 1), dtype=bool)

    term = dre_splitting_term(local, consensus, centers, masks, penalty=2.0)

    expected = 0.5 * 2.0 * ((-1.0) * (-1.0 + 2.0) + 1.0 * (1.0 - 2.0))
    assert term == pytest.approx(expected)


def test_drs_relaxation_must_be_in_open_interval():
    arrays = np.ones((1, 1, 1))
    masks = np.ones((1, 1), dtype=bool)
    for relaxation in (0.0, 2.0, np.inf):
        with pytest.raises(ValueError, match="relaxation"):
            drs_step(arrays, arrays, masks, arrays[0], relaxation=relaxation)


def test_canonical_drs_matches_previous_embedded_plain_step():
    rng = np.random.default_rng(20260727)
    local = rng.normal(size=(3, 4, 2))
    centers = rng.normal(size=(3, 4, 2))
    masks = np.array([
        [True, True, False, True],
        [True, False, True, True],
        [False, True, True, True],
    ])
    previous = rng.normal(size=(4, 2))

    old_consensus, old_centers, old_residuals = plain_drs_step(
        local, centers, masks, previous
    )
    consensus, next_centers, _, residuals, _ = drs_step(
        local, centers, masks, previous
    )

    np.testing.assert_allclose(consensus, old_consensus)
    np.testing.assert_allclose(next_centers, old_centers)
    assert residuals.fixed_point_squared == pytest.approx(
        old_residuals.primal_squared
    )
    assert residuals.center_step_squared == pytest.approx(
        old_residuals.primal_squared
    )


def test_complete_dre_recovers_data_term_and_applies_sandwich():
    local_data = recover_local_data_objective(
        proximal_objective=130.0,
        penalty=5.0,
        proximal_displacement_squared=6.0,
    )
    model, envelope = complete_douglas_rachford_envelope(
        local_data, splitting_term=-10.0, consensus_objective=95.0
    )

    assert local_data == pytest.approx(100.0)
    assert model == pytest.approx(90.0)
    assert envelope == pytest.approx(95.0)


def test_consensus_reset_collapses_all_drs_variables():
    consensus = np.array([[1.0, 2.0], [3.0, 4.0]])
    local, centers, reset_consensus = reset_to_consensus(consensus, 3)

    np.testing.assert_allclose(local, np.repeat(consensus[None], 3, axis=0))
    np.testing.assert_allclose(centers, local)
    np.testing.assert_allclose(reset_consensus, consensus)
    local[0, 0, 0] = 99.0
    assert centers[0, 0, 0] == 1.0


def test_scalar_metric_projection_uses_per_copy_geometric_mean():
    local = np.zeros((2, 1, 9))
    local[:, 0, 0] = [1.0, 3.0]
    centers = np.zeros_like(local)
    masks = np.ones((2, 1), dtype=bool)
    raw = np.zeros((2, 1, 9, 9))
    raw[0, 0] = 2.0 * np.eye(9)
    raw[1, 0] = 8.0 * np.eye(9)

    consensus, _, _, residuals, selected = drs_step(
        local,
        centers,
        masks,
        np.zeros((1, 9)),
        metric_blocks=raw,
        metric_mode="scalar",
    )

    # Reflections are 2 and 6, weighted by scalar blocks 2 and 8.
    assert consensus[0, 0] == pytest.approx(5.2)
    np.testing.assert_allclose(selected, raw)
    expected = 2.0 * (1.0 - 5.2) ** 2 + 8.0 * (3.0 - 5.2) ** 2
    assert residuals.fixed_point_squared == pytest.approx(expected)


def test_diagonal_metric_projection_is_coordinate_wise():
    local = np.zeros((2, 1, 9))
    local[0, 0, :2] = [1.0, 2.0]
    local[1, 0, :2] = [3.0, 4.0]
    centers = np.zeros_like(local)
    masks = np.ones((2, 1), dtype=bool)
    raw = np.zeros((2, 1, 9, 9))
    raw[0, 0] = np.diag([1.0, 9.0] + [1.0] * 7)
    raw[1, 0] = np.diag([3.0, 1.0] + [1.0] * 7)

    consensus, _, _, _, selected = drs_step(
        local,
        centers,
        masks,
        np.zeros((1, 9)),
        metric_blocks=raw,
        metric_mode="diagonal",
    )

    assert consensus[0, 0] == pytest.approx((1.0 * 2.0 + 3.0 * 6.0) / 4.0)
    assert consensus[0, 1] == pytest.approx((9.0 * 4.0 + 1.0 * 8.0) / 10.0)
    np.testing.assert_allclose(selected, raw)


def test_full_metric_projection_preserves_parameter_coupling():
    local = np.zeros((2, 1, 9))
    local[0, 0, :2] = [1.0, 0.0]
    local[1, 0, :2] = [0.0, 1.0]
    centers = np.zeros_like(local)
    masks = np.ones((2, 1), dtype=bool)
    raw = np.zeros((2, 1, 9, 9))
    raw[0, 0] = np.eye(9)
    raw[1, 0] = np.eye(9)
    raw[1, 0, :2, :2] = [[2.0, 0.5], [0.5, 1.5]]

    consensus, _, reflected, _, selected = drs_step(
        local,
        centers,
        masks,
        np.zeros((1, 9)),
        metric_blocks=raw,
        metric_mode="full",
    )

    expected = np.linalg.solve(
        raw[0, 0] + raw[1, 0],
        raw[0, 0] @ reflected[0, 0] + raw[1, 0] @ reflected[1, 0],
    )
    np.testing.assert_allclose(consensus[0], expected)
    np.testing.assert_allclose(selected, raw)


def test_full_metric_projection_batched_solve_matches_camera_loop():
    rng = np.random.default_rng(20260729)
    clusters = 4
    cameras = 7
    parameters = 9
    masks = rng.random((clusters, cameras)) > 0.35
    masks[0] = True
    values = rng.normal(size=(clusters, cameras, parameters))
    factors = rng.normal(size=(clusters, cameras, parameters, parameters))
    blocks = np.einsum("kcji,kcjl->kcil", factors, factors)
    blocks += 0.1 * np.eye(parameters)[None, None]
    blocks[~masks] = 0.0

    metric_sum = np.sum(blocks, axis=0)
    right_hand_side = np.einsum("kcij,kcj->ci", blocks, values)
    expected = np.stack([
        np.linalg.solve(metric_sum[camera], right_hand_side[camera])
        for camera in range(cameras)
    ])

    from drs_consensus import project_consensus

    actual = project_consensus(
        values, masks, np.zeros((cameras, parameters)), blocks
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_active_metric_storage_matches_dense_drs_step():
    rng = np.random.default_rng(20260730)
    clusters = 4
    cameras = 7
    masks = rng.random((clusters, cameras)) > 0.35
    masks[0] = True
    local = rng.normal(size=(clusters, cameras, 9))
    centers = rng.normal(size=(clusters, cameras, 9))
    previous = rng.normal(size=(cameras, 9))
    factors = rng.normal(size=(clusters, cameras, 9, 9))
    dense_blocks = np.einsum("kcji,kcjl->kcil", factors, factors)
    dense_blocks += 0.1 * np.eye(9)[None, None]
    dense_blocks[~masks] = 0.0
    cluster_indices, camera_indices = np.nonzero(masks)
    active_blocks = ActiveCameraMetricBlocks(
        cluster_indices.astype(np.uint16),
        camera_indices.astype(np.uint16),
        dense_blocks[masks],
        clusters,
        cameras,
    )

    dense = drs_step(
        local, centers, masks, previous,
        metric_blocks=dense_blocks, metric_mode="full",
    )
    active = drs_step(
        local, centers, masks, previous,
        metric_blocks=active_blocks, metric_mode="full",
    )

    np.testing.assert_allclose(active[0], dense[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(active[1], dense[1], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(active[2], dense[2], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        tuple(vars(active[3]).values()),
        tuple(vars(dense[3]).values()),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(active[4].blocks, dense[4][masks])
    assert dre_splitting_term(
        local, active[0], centers, masks, metric_blocks=active[4]
    ) == pytest.approx(
        dre_splitting_term(
            local, dense[0], centers, masks, metric_blocks=dense[4]
        ),
        rel=1e-13,
        abs=1e-13,
    )
