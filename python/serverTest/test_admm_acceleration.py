import numpy as np
import pytest

from admm_acceleration import (
    augmented_consensus_merit,
    consensus_disagreement_squared,
    extrapolate_centers,
    nesterov_coefficient,
    should_fallback_acceleration,
)
from outer_acceleration import (
    AndersonAcceleration,
    LBFGSAcceleration,
    LegacyNesterov,
    interpolate_line_search_center,
)


def test_nesterov_coefficient_restarts_with_two_plain_steps():
    assert nesterov_coefficient(4, 4) == 0.0
    assert nesterov_coefficient(5, 4) == 0.0
    assert nesterov_coefficient(6, 4) == pytest.approx(0.25)


def test_center_extrapolation_uses_current_fixed_point_direction():
    previous = np.array([[[1.0, 2.0]]])
    current = np.array([[[3.0, 6.0]]])
    np.testing.assert_allclose(
        extrapolate_centers(current, previous, 0.25),
        [[[3.5, 7.0]]],
    )


def test_consensus_disagreement_ignores_absent_copies():
    local = np.array([[[1.0]], [[9.0]]])
    masks = np.array([[True], [False]])
    assert consensus_disagreement_squared(local, np.array([[3.0]]), masks) == 4.0


def test_augmented_merit_combines_pixel_quality_and_consensus_penalty():
    assert augmented_consensus_merit(10.0, 4.0, 2.5) == 20.0


def test_acceleration_fallback_requires_both_relative_failures():
    assert should_fallback_acceleration(
        12.0, 10.0, 9.0, 15.0, 10.0, 1.01, 1.01, 100.0)
    assert not should_fallback_acceleration(
        9.0, 10.0, 9.0, 15.0, 10.0, 1.01, 1.01, 100.0)
    assert not should_fallback_acceleration(
        12.0, 10.0, 9.0, 9.0, 10.0, 1.01, 1.01, 100.0)


def test_acceleration_falls_back_after_leaving_established_best_basin():
    assert should_fallback_acceleration(
        1.1e6, 2.0e6, 100.0, 1.1e6, 2.0e6, 1.01, 1.01, 1e4)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_accelerated_candidate_falls_back(value):
    assert should_fallback_acceleration(
        value, 10.0, 5.0, 1.0, 1.0, 1.01, 1.01, 1e4)


def test_legacy_nesterov_has_two_nominal_startup_proposals():
    accelerator = LegacyNesterov()
    current = np.array([0.0])
    mapped = np.array([2.0])
    first, first_accelerated = accelerator.propose(current, mapped, 0)
    second, second_accelerated = accelerator.propose(mapped, np.array([3.0]), 1)
    third, third_accelerated = accelerator.propose(
        np.array([3.0]), np.array([4.0]), 2)
    np.testing.assert_allclose(first, [2.0])
    np.testing.assert_allclose(second, [3.0])
    np.testing.assert_allclose(third, [4.25])
    assert not first_accelerated
    assert not second_accelerated
    assert third_accelerated


@pytest.mark.parametrize(
    "accelerator_type", [LBFGSAcceleration, AndersonAcceleration]
)
def test_secant_accelerator_uses_observed_first_trial(accelerator_type):
    accelerator = accelerator_type()
    current = np.array([0.0])
    mapped = np.array([1.0])

    first, first_accelerated = accelerator.propose(current, mapped, 0)
    accelerator.observe_first_trial(np.array([0.5]))
    second, second_accelerated = accelerator.propose(
        np.array([1.0]), np.array([2.0]), 1
    )

    np.testing.assert_allclose(first, mapped)
    assert not first_accelerated
    assert second_accelerated
    assert not np.array_equal(second, np.array([2.0]))


@pytest.mark.parametrize(
    ("weight", "expected"),
    [(0.0, [2.0, 4.0]), (0.5, [3.0, 6.0]), (1.0, [4.0, 8.0])],
)
def test_line_search_center_weights(weight, expected):
    result = interpolate_line_search_center(
        np.array([2.0, 4.0]), np.array([4.0, 8.0]), weight)
    np.testing.assert_allclose(result, expected)