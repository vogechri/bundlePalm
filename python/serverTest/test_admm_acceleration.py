import sys
from types import SimpleNamespace

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


def test_acceleration_step_limit_hits_are_counted(monkeypatch):
    monkeypatch.setenv("BUNDLE_PALM_ACCEL_MAX_STEP_RATIO", "1")
    accelerator = LegacyNesterov()

    accelerator.propose(np.array([0.0]), np.array([2.0]), 0)
    accelerator.propose(np.array([2.0]), np.array([3.0]), 1)
    candidate, accelerated = accelerator.propose(
        np.array([3.0]), np.array([4.0]), 2
    )
    accelerator.reset()

    np.testing.assert_allclose(candidate, [4.0])
    assert not accelerated
    assert accelerator.step_limit_hits == 1


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
            "--initial-shared-schur-damping-policy",
            "model_ratio",
            "--initial-shared-schur-rebase-trust-state",
        ],
    )
    arguments = client_drs.parse_arguments()
    assert arguments.initial_shared_schur_correction
    assert arguments.initial_shared_schur_maximum_corrections == 3
    assert arguments.initial_shared_schur_damping_policy == "model_ratio"
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


def test_mid_shared_schur_cli_iteration(monkeypatch):
    monkeypatch.setenv("BUNDLE_PALM_CAMERA_UPDATE", "se3_left")
    monkeypatch.setenv("BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--iterations",
            "30",
            "--proximal-metric",
            "block",
            "--consensus-metric",
            "full",
            "--shared-only-camera-proximal",
            "--mid-shared-schur-correction-iteration",
            "10",
            "--mid-shared-schur-transport-product-state",
        ],
    )

    arguments = client_drs.parse_arguments()
    client_drs.validate_arguments(arguments)

    assert arguments.mid_shared_schur_correction_iteration == 10
    assert arguments.mid_shared_schur_transport_product_state


def test_mid_shared_schur_transport_requires_iteration(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--mid-shared-schur-transport-product-state",
        ],
    )
    with pytest.raises(
        ValueError,
        match="product transport requires a correction iteration",
    ):
        client_drs.validate_arguments(client_drs.parse_arguments())


def test_mid_shared_schur_rejects_iteration_at_budget(monkeypatch):
    monkeypatch.setenv("BUNDLE_PALM_CAMERA_UPDATE", "se3_left")
    monkeypatch.setenv("BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--iterations",
            "10",
            "--proximal-metric",
            "block",
            "--consensus-metric",
            "full",
            "--shared-only-camera-proximal",
            "--mid-shared-schur-correction-iteration",
            "10",
        ],
    )
    with pytest.raises(
        ValueError,
        match="mid shared Schur correction must be in",
    ):
        client_drs.validate_arguments(client_drs.parse_arguments())


@pytest.mark.parametrize("accept_candidate", [False, True])
def test_mid_shared_schur_rebase_restores_or_commits(
    monkeypatch, accept_candidate
):
    class FakeWorker:
        def __init__(self):
            self.last_trust_region_radii = np.array([10.0, 20.0])
            self.steps = []

        def build_schur_systems(self, *args):
            return []

        def apply_camera_step(
            self,
            camera_indices_in_cluster,
            point_indices_in_cluster,
            consensus,
            landmarks,
            tangent_step,
            cluster_count,
            landmark_refinement_steps,
            rebase_trust_state=False,
        ):
            self.steps.append((tangent_step.copy(), rebase_trust_state))
            if np.any(tangent_step):
                cameras = consensus + 1.0
                points = landmarks + 1.0
            else:
                cameras = consensus.copy()
                points = landmarks.copy()
            self.last_trust_region_radii = np.array(
                [30.0, 30.0] if rebase_trust_state else [5.0, 6.0]
            )
            return np.array([40.0, 40.0]), cameras, points

    monkeypatch.setattr(
        client_drs,
        "solve_global_schur_system",
        lambda *args, **kwargs: (
            np.ones((2, 9)),
            {"linearTermination": 0},
        ),
    )
    worker = FakeWorker()
    arguments = SimpleNamespace(
        worker_owned_landmarks=False,
        mid_shared_schur_transport_product_state=False,
        shared_schur_landmark_damping=3.0,
        shared_schur_camera_damping=3.0,
        shared_schur_step_scale=1.0,
        shared_schur_linear_solver="cg",
        shared_schur_relative_tolerance=1e-6,
        shared_schur_maximum_iterations=500,
        shared_schur_operator="python",
        shared_schur_preconditioner="jacobi",
        shared_schur_landmark_refinement_steps=0,
    )
    consensus = np.zeros((2, 9))
    landmarks = np.zeros((1, 3))

    def evaluate_state(cameras, points, *_):
        changed = np.any(cameras != 0.0)
        return {
            "objectiveValue": (
                80.0 if accept_candidate else 120.0
            ) if changed else 100.0
        }

    result = client_drs.run_mid_shared_schur_rebase(
        worker,
        arguments,
        [np.array([0]), np.array([1])],
        [np.array([0]), np.array([0])],
        consensus,
        landmarks,
        np.ones_like(consensus),
        2,
        2,
        evaluate_state,
        np.array([0, 1]),
        np.array([0, 0]),
        np.zeros((2, 2)),
        1,
    )

    assert result["accepted"] is accept_candidate
    assert len(worker.steps) == 2
    if accept_candidate:
        assert result["trustRebased"]
        assert worker.steps[1][1]
        np.testing.assert_array_equal(result["consensus"], 1.0)
        np.testing.assert_array_equal(result["landmarks"], 1.0)
    else:
        assert not result["trustRebased"]
        assert not worker.steps[1][1]
        np.testing.assert_array_equal(result["consensus"], consensus)
        np.testing.assert_array_equal(result["landmarks"], landmarks)


def test_product_camera_transport_preserves_scaled_dual_offsets():
    local_cameras = np.zeros((2, 2, 9))
    local_cameras[0, 0, :6] = [0.01, 0.02, 0.03, 1.0, 2.0, 3.0]
    local_cameras[1, 0, :6] = [-0.02, 0.01, 0.04, 1.1, 1.9, 3.2]
    centers = local_cameras - 0.25
    tangent_step = np.zeros((2, 9))
    tangent_step[:, :6] = [0.1, -0.2, 0.3, 0.01, 0.02, -0.03]
    scaling = np.full((2, 9), 2.0)

    transported, transported_centers = (
        client_drs.transport_product_camera_state(
            local_cameras,
            centers,
            tangent_step,
            scaling,
        )
    )

    assert not np.array_equal(transported, local_cameras)
    np.testing.assert_allclose(
        transported - transported_centers,
        local_cameras - centers,
        rtol=0.0,
        atol=2e-16,
    )


def test_mid_shared_schur_transports_product_state_without_trust_reset(
    monkeypatch,
):
    class FakeWorker:
        def __init__(self):
            self.last_trust_region_radii = np.array([10.0, 20.0])

        def build_schur_systems(self, *args):
            return []

        def apply_camera_step(
            self,
            camera_indices_in_cluster,
            point_indices_in_cluster,
            consensus,
            landmarks,
            tangent_step,
            cluster_count,
            landmark_refinement_steps,
            rebase_trust_state=False,
        ):
            assert not rebase_trust_state
            return (
                np.array([40.0, 40.0]),
                consensus + tangent_step,
                landmarks + 1.0,
            )

        def transport_product_state(
            self,
            camera_indices_in_cluster,
            point_indices_in_cluster,
            local_cameras,
            centers,
            landmarks,
            tangent_step,
            cluster_count,
        ):
            repeated_step = np.broadcast_to(tangent_step, local_cameras.shape)
            transported = client_drs.left_se3_camera_plus(
                local_cameras,
                repeated_step,
            )
            active = np.zeros(local_cameras.shape[:2], dtype=bool)
            for cluster_id, indices in enumerate(camera_indices_in_cluster):
                active[cluster_id, np.unique(indices)] = True
            transported = np.where(
                active[..., None], transported, local_cameras
            )
            return np.array([30.0, 30.0]), transported, landmarks.copy()

    tangent_step = np.zeros((2, 9))
    tangent_step[:, 6:] = 0.5
    monkeypatch.setattr(
        client_drs,
        "solve_global_schur_system",
        lambda *args, **kwargs: (tangent_step, {"linearTermination": 0}),
    )
    worker = FakeWorker()
    arguments = SimpleNamespace(
        worker_owned_landmarks=False,
        mid_shared_schur_transport_product_state=True,
        shared_schur_landmark_damping=3.0,
        shared_schur_camera_damping=3.0,
        shared_schur_step_scale=1.0,
        shared_schur_linear_solver="cg",
        shared_schur_relative_tolerance=1e-6,
        shared_schur_maximum_iterations=500,
        shared_schur_operator="python",
        shared_schur_preconditioner="jacobi",
        shared_schur_landmark_refinement_steps=0,
    )
    consensus = np.zeros((2, 9))
    landmarks = np.zeros((1, 3))
    local_cameras = np.zeros((2, 2, 9))
    local_cameras[0] += 0.1
    local_cameras[1] -= 0.2
    centers = local_cameras - 0.3

    result = client_drs.run_mid_shared_schur_rebase(
        worker,
        arguments,
        [np.array([0]), np.array([1])],
        [np.array([0]), np.array([0])],
        consensus,
        landmarks,
        np.ones_like(consensus),
        2,
        2,
        lambda cameras, points, *_: {
            "objectiveValue": 80.0 if np.any(cameras) else 100.0
        },
        np.array([0, 1]),
        np.array([0, 0]),
        np.zeros((2, 2)),
        1,
        local_cameras=local_cameras,
        centers=centers,
    )

    assert result["accepted"]
    assert result["productStateTransported"]
    assert not result["trustRebased"]
    assert result["preRebaseTrustRadii"] == [10.0, 20.0]
    assert result["postRebaseTrustRadii"] == [10.0, 20.0]
    assert result["transportOffsetError"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        result["localCameras"] - result["centers"],
        local_cameras - centers,
    )


def test_schur_trial_model_ratio_requires_model_agreement():
    diagnostics = {
        "linearTermination": 0,
        "dampedPredictedReduction": 100.0,
        "undampedPredictedReduction": 80.0,
    }

    geometric = client_drs.assess_schur_trial(
        1000.0, 990.0, diagnostics, "geometric", 0.1
    )
    model_ratio = client_drs.assess_schur_trial(
        1000.0, 990.0, diagnostics, "model_ratio", 0.1
    )

    assert geometric["accepted"]
    assert not model_ratio["accepted"]
    assert model_ratio["actualReduction"] == 5.0
    assert model_ratio["dampedGainRatio"] == 0.05


def test_schur_trial_model_ratio_accepts_converged_predictive_step():
    assessment = client_drs.assess_schur_trial(
        1000.0,
        900.0,
        {
            "linearTermination": 0,
            "dampedPredictedReduction": 100.0,
            "undampedPredictedReduction": 125.0,
        },
        "model_ratio",
        0.1,
    )

    assert assessment["accepted"]
    assert assessment["dampedGainRatio"] == 0.5
    assert assessment["undampedGainRatio"] == 0.4


def test_model_ratio_damping_respects_startup_floor():
    assert client_drs.model_ratio_damping_factor(2.0) == pytest.approx(1.0 / 3.0)
    assert client_drs.model_ratio_damping_factor(2.0, 0.5) == 0.5


def test_canonical_initial_state_cli_choice(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--initial-state",
            "stage.npz",
            "--initial-state-frame",
            "canonical",
        ],
    )

    arguments = client_drs.parse_arguments()

    assert arguments.initial_state == "stage.npz"
    assert arguments.initial_state_frame == "canonical"


def test_local_solver_switch_cli(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--iterations",
            "30",
            "--local-solver",
            "schur_pcg",
            "--local-solver-switch-iteration",
            "10",
            "--local-solver-after-switch",
            "nesterov",
        ],
    )
    arguments = client_drs.parse_arguments()
    client_drs.validate_arguments(arguments)
    assert arguments.local_solver_switch_iteration == 10
    assert arguments.local_solver_after_switch == "nesterov"


def test_local_solver_switch_requires_second_solver(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--iterations",
            "30",
            "--local-solver-switch-iteration",
            "10",
        ],
    )
    with pytest.raises(ValueError, match="requires --local-solver-after-switch"):
        client_drs.validate_arguments(client_drs.parse_arguments())


def test_zero_iteration_final_schur_allows_disabled_solver_switch(monkeypatch):
    monkeypatch.setenv("BUNDLE_PALM_CAMERA_UPDATE", "se3_left")
    monkeypatch.setenv("BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--iterations",
            "0",
            "--final-shared-schur-correction",
            "--shared-only-camera-proximal",
            "--proximal-metric",
            "block",
            "--consensus-metric",
            "full",
        ],
    )

    arguments = client_drs.parse_arguments()
    client_drs.validate_arguments(arguments)

    assert arguments.local_solver_switch_iteration == 0
    assert arguments.local_state_rebase_iteration == 0
    assert arguments.outer_acceleration_restart_iteration == 0


def test_zero_iteration_final_schur_rejects_solver_switch(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--iterations",
            "0",
            "--final-shared-schur-correction",
            "--local-solver-switch-iteration",
            "1",
            "--local-solver-after-switch",
            "schur_pcg",
        ],
    )

    with pytest.raises(ValueError, match="switch must be in"):
        client_drs.validate_arguments(client_drs.parse_arguments())


def test_ruiz_camera_scaling_cli_choice(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["client_drs.py", "unused.bal", "--camera-scaling", "ruiz_initial"],
    )
    arguments = client_drs.parse_arguments()
    assert arguments.camera_scaling == "ruiz_initial"


def test_block_jacobi_camera_scaling_cli_choice(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--camera-scaling",
            "block_jacobi_initial",
        ],
    )
    arguments = client_drs.parse_arguments()
    assert arguments.camera_scaling == "block_jacobi_initial"


def test_worker_block_jacobi_camera_scaling_cli_choice(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--camera-scaling",
            "worker_block_jacobi_initial",
        ],
    )
    arguments = client_drs.parse_arguments()
    assert arguments.camera_scaling == "worker_block_jacobi_initial"


def test_worker_diagonal_jacobi_camera_scaling_cli_choice(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--camera-scaling",
            "worker_diagonal_jacobi_initial",
        ],
    )
    arguments = client_drs.parse_arguments()
    assert arguments.camera_scaling == "worker_diagonal_jacobi_initial"


def test_worker_z_f_block_camera_scaling_cli_choice(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "client_drs.py",
            "unused.bal",
            "--camera-scaling",
            "worker_z_f_block_jacobi_initial",
        ],
    )
    arguments = client_drs.parse_arguments()
    assert arguments.camera_scaling == "worker_z_f_block_jacobi_initial"


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