import sys

import numpy as np
import pytest

import client_drs
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


def test_drs_cli_exposes_themelis_fast_drs(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["client_drs.py", "unused.bal", "--outer-acceleration", "themelis_nesterov"],
    )
    assert client_drs.parse_arguments().outer_acceleration == "themelis_nesterov"


def test_outer_acceleration_cutoff_is_active_only_before_until():
    assert client_drs.scheduled_outer_acceleration_active(30, 0)
    assert client_drs.scheduled_outer_acceleration_active(30, 29)
    assert not client_drs.scheduled_outer_acceleration_active(30, 30)
    assert client_drs.scheduled_outer_acceleration_active(0, 90)


def test_outer_acceleration_restart_is_active_once():
    assert not client_drs.outer_acceleration_restart_is_active(100, 99)
    assert client_drs.outer_acceleration_restart_is_active(100, 100)
    assert not client_drs.outer_acceleration_restart_is_active(100, 101)
    assert not client_drs.outer_acceleration_restart_is_active(0, 100)


def test_metric_proposal_cutoff_is_active_only_before_until():
    assert client_drs.scheduled_metric_proposal_scale(0.5, 30, 0) == 0.5
    assert client_drs.scheduled_metric_proposal_scale(0.5, 30, 29) == 0.5
    assert client_drs.scheduled_metric_proposal_scale(0.5, 30, 30) == 1.0
    assert client_drs.scheduled_metric_proposal_scale(0.5, 0, 90) == 0.5


def test_local_state_rebase_is_active_once():
    assert not client_drs.local_state_rebase_is_active(30, 29)
    assert client_drs.local_state_rebase_is_active(30, 30)
    assert not client_drs.local_state_rebase_is_active(30, 31)
    assert not client_drs.local_state_rebase_is_active(0, 0)


def test_factorized_metric_allows_themelis_acceleration(monkeypatch):
    monkeypatch.setenv("BUNDLE_PALM_CAMERA_UPDATE", "se3_left")
    monkeypatch.setenv("BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--local-solver",
            "schur_pcg",
            "--proximal-metric",
            "block",
            "--consensus-metric",
            "full",
            "--shared-only-camera-proximal",
            "--factorized-coupled-schur-proximal-metric",
            "--global-schur-majorizer-observability-threshold",
            "0.55",
            "--outer-acceleration",
            "themelis_nesterov",
        ],
    )
    client_drs.validate_arguments(client_drs.parse_arguments())


def test_factorized_metric_allows_adaptive_local_depth(monkeypatch):
    monkeypatch.setenv("BUNDLE_PALM_CAMERA_UPDATE", "se3_left")
    monkeypatch.setenv("BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--local-solver",
            "schur_pcg",
            "--proximal-metric",
            "block",
            "--consensus-metric",
            "full",
            "--shared-only-camera-proximal",
            "--factorized-coupled-schur-proximal-metric",
            "--global-schur-majorizer-observability-threshold",
            "0.55",
            "--adaptive-local-depth",
        ],
    )
    client_drs.validate_arguments(client_drs.parse_arguments())


def test_shared_only_proximal_allows_fixed_shared_proposal_damping(monkeypatch):
    monkeypatch.setenv("BUNDLE_PALM_CAMERA_UPDATE", "se3_left")
    monkeypatch.setenv("BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--proximal-metric",
            "block",
            "--consensus-metric",
            "full",
            "--shared-only-camera-proximal",
            "--metric-proposal-disagreement-scale",
            "0.5",
        ],
    )
    client_drs.validate_arguments(client_drs.parse_arguments())


def test_shared_proposal_damping_allows_iteration_cutoff(monkeypatch):
    monkeypatch.setenv("BUNDLE_PALM_CAMERA_UPDATE", "se3_left")
    monkeypatch.setenv("BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--iterations",
            "90",
            "--proximal-metric",
            "block",
            "--consensus-metric",
            "full",
            "--shared-only-camera-proximal",
            "--metric-proposal-disagreement-scale",
            "0.5",
            "--metric-proposal-disagreement-until",
            "30",
        ],
    )
    client_drs.validate_arguments(client_drs.parse_arguments())


def test_bootstrap_trust_rebase_cli_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--initial-shared-schur-correction",
            "--initial-shared-schur-maximum-corrections",
            "3",
            "--initial-shared-schur-rebase-trust-state",
        ],
    )
    arguments = client_drs.parse_arguments()
    assert arguments.initial_shared_schur_correction
    assert arguments.initial_shared_schur_maximum_corrections == 3
    assert arguments.initial_shared_schur_rebase_trust_state


def test_initial_shared_schur_rejects_nonpositive_correction_cap(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--initial-shared-schur-maximum-corrections",
            "0",
        ],
    )
    with pytest.raises(
        ValueError,
        match="initial shared Schur maximum corrections must be positive",
    ):
        client_drs.validate_arguments(client_drs.parse_arguments())


def test_ruiz_camera_scaling_cli_choice(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["client_drs.py", "unused.bal", "--camera-scaling", "ruiz_initial"],
    )
    arguments = client_drs.parse_arguments()
    assert arguments.camera_scaling == "ruiz_initial"


def test_collective_trust_trial_radius_uses_geometric_mean():
    radius = client_drs.collective_trust_trial_radius(
        np.array([4.0, 16.0]), 0.5
    )
    assert radius == pytest.approx(4.0)


@pytest.mark.parametrize("radii", [[], [1.0, np.nan], [1.0, 0.0]])
def test_collective_trust_trial_radius_rejects_invalid_radii(radii):
    with pytest.raises(ValueError, match="collective trust radii"):
        client_drs.collective_trust_trial_radius(radii, 0.5)


def test_collective_trust_trial_requires_material_sse_improvement():
    assert not client_drs.prefer_collective_trust_trial(
        False, 100.0, False, 100.0 - 1e-11
    )
    assert client_drs.prefer_collective_trust_trial(
        False, 100.0, False, 99.0
    )
    assert client_drs.prefer_collective_trust_trial(
        True, 100.0, False, 101.0
    )
    assert not client_drs.prefer_collective_trust_trial(
        False, 100.0, True, 99.0
    )


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