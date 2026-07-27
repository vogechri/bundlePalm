import numpy as np
import pytest

from admm_consensus import (
    AdmmResiduals,
    adapt_penalty,
    admm_residuals,
    consensus_update,
    combine_proximal_terms,
    dual_update,
    rescale_scaled_duals,
    plain_drs_step,
)


def test_plain_drs_step_matches_product_space_formula():
    centers = np.array([[[1.0]], [[5.0]]])
    local_cameras = np.array([[[2.0]], [[4.0]]])
    camera_masks = np.ones((2, 1), dtype=bool)

    consensus, next_centers, residuals = plain_drs_step(
        local_cameras,
        centers,
        camera_masks,
        np.array([[0.0]]),
    )

    # P_C(2 prox_f(z) - z) = mean(3, 3) = 3.
    np.testing.assert_allclose(consensus, [[3.0]])
    np.testing.assert_allclose(next_centers, [[[2.0]], [[4.0]]])
    assert residuals.primal_squared == 2.0
    assert residuals.dual_squared == 18.0


def test_consensus_uses_only_present_camera_copies_and_scaled_duals():
    local = np.array([
        [[1.0], [10.0]],
        [[3.0], [20.0]],
    ])
    dual = np.array([
        [[0.5], [0.0]],
        [[-0.5], [2.0]],
    ])
    masks = np.array([[True, False], [True, True]])
    previous = np.array([[2.0], [18.0]])

    consensus = consensus_update(local, dual, masks, previous)

    np.testing.assert_allclose(consensus, [[2.0], [20.0]])


def test_over_relaxed_dual_update_preserves_zero_sum_at_consensus():
    local = np.array([[[1.0]], [[3.0]]])
    dual = np.zeros_like(local)
    masks = np.ones((2, 1), dtype=bool)
    previous = np.array([[0.0]])

    consensus = consensus_update(local, dual, masks, previous)
    updated = dual_update(dual, local, consensus, masks, alpha=1.5)

    np.testing.assert_allclose(consensus, [[2.0]])
    np.testing.assert_allclose(updated[:, 0, 0], [-1.5, 1.5])
    assert np.sum(updated[:, 0, 0]) == pytest.approx(0.0)


def test_residuals_and_penalty_adaptation_match_daba_rules():
    local = np.array([[[0.0]], [[2.0]]])
    masks = np.ones((2, 1), dtype=bool)
    consensus = np.array([[1.0]])
    previous = np.array([[0.5]])

    residuals = admm_residuals(
        local, consensus, previous, masks)

    assert residuals.primal_squared == pytest.approx(2.5)
    assert residuals.dual_squared == pytest.approx(0.5)
    penalty, ratio = adapt_penalty(2.0, residuals, initial_penalty=2.0)
    assert ratio == pytest.approx(1.5)
    assert penalty == pytest.approx(3.0)

    increased, ratio = adapt_penalty(
        2.0, AdmmResiduals(100.0, 1.0), initial_penalty=2.0)
    assert increased == pytest.approx(3.0)
    assert ratio == pytest.approx(1.5)

    decreased, ratio = adapt_penalty(
        2.0, AdmmResiduals(1.0, 100.0), initial_penalty=2.0)
    assert decreased == pytest.approx(1.6)
    assert ratio == pytest.approx(0.8)
    np.testing.assert_allclose(
        rescale_scaled_duals(np.array([2.0, 4.0]), ratio), [2.5, 5.0])


def test_single_cluster_has_exact_consensus_and_zero_residuals():
    local = np.array([[[1.0, 2.0]]])
    dual = np.zeros_like(local)
    masks = np.array([[True]])
    previous = local[0].copy()

    consensus = consensus_update(local, dual, masks, previous)
    residuals = admm_residuals(
        local, consensus, previous, masks)

    np.testing.assert_allclose(consensus, local[0])
    np.testing.assert_allclose(
        dual_update(dual, local, consensus, masks, alpha=1.5), 0)
    assert residuals.primal_squared == pytest.approx(0.0)
    assert residuals.dual_squared == pytest.approx(0.0)


def test_combined_proximal_term_preserves_quadratic_up_to_constant():
    consensus_center = np.array([[[-2.0, 3.0]]])
    accepted_local = np.array([[[4.0, -1.0]]])
    penalty = 5.0
    damping = 7.0
    effective_center, effective_penalty = combine_proximal_terms(
        consensus_center, accepted_local, penalty, damping)

    offsets = []
    for candidate in (np.array([[[0.0, 2.0]]]), np.array([[[8.0, -5.0]]])):
        original = (
            penalty * np.sum((candidate - consensus_center) ** 2)
            + damping * np.sum((candidate - accepted_local) ** 2)
        )
        combined = effective_penalty * np.sum(
            (candidate - effective_center) ** 2)
        offsets.append(original - combined)

    assert effective_penalty == pytest.approx(12.0)
    assert offsets[0] == pytest.approx(offsets[1])


def test_zero_recovery_damping_preserves_original_proximal_term():
    center = np.array([[[1.0, 2.0]]])
    effective_center, effective_penalty = combine_proximal_terms(
        center, np.array([[[9.0, 8.0]]]), 3.0, 0.0)

    assert effective_center is center
    assert effective_penalty == 3.0
