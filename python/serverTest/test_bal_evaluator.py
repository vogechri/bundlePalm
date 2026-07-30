import json

import numpy as np
import pytest

from bal_evaluator import (
    canonicalize_bal_problem,
    encoded_daba_ray_state_to_matrix,
    evaluate_encoded_daba_ray_state,
    evaluate_daba_ray_state,
    evaluate_daba_state_pixel_error,
    evaluate_bal_state,
    project_bal,
    read_bal_problem,
    read_ceres_text_state,
    read_daba_ceres_state,
    save_bal_state,
    write_bal_problem,
)


def test_evaluate_bal_state_reports_explicit_metric_conventions():
    cameras = np.array([[0, 0, 0, 0, 0, 0, 2, 0, 0]], dtype=float)
    points = np.array([[1, 2, -4]], dtype=float)
    camera_indices = np.array([0])
    point_indices = np.array([0])
    observations = np.array([[0, 0]], dtype=float)

    metrics = evaluate_bal_state(
        cameras, points, camera_indices, point_indices, observations,
        huber_delta=0.5)

    np.testing.assert_allclose(
        project_bal(cameras, points, camera_indices, point_indices),
        [[0.5, 1.0]],
    )
    assert metrics["sumSquaredError"] == pytest.approx(1.25)
    assert metrics["ceresCost"] == pytest.approx(0.625)
    assert metrics["msePerObservation"] == pytest.approx(1.25)
    assert metrics["msePerScalarResidual"] == pytest.approx(0.625)
    assert metrics["meanReprojectionError"] == pytest.approx(np.sqrt(1.25))
    assert metrics["huberCeresCost"] == pytest.approx(
        0.5 * (np.sqrt(1.25) - 0.25))


def test_save_bal_state_round_trips_arrays_and_metadata(tmp_path):
    path = tmp_path / "state.npz"
    cameras = np.arange(18, dtype=float).reshape(2, 9)
    points = np.arange(9, dtype=float).reshape(3, 3)

    save_bal_state(path, cameras, points, {"solver": "drs", "iteration": 4})

    with np.load(path) as state:
        assert state["cameras"] == pytest.approx(cameras)
        assert state["points"] == pytest.approx(points)
        assert json.loads(str(state["metadata_json"])) == {
            "iteration": 4,
            "solver": "drs",
        }


def test_raw_bal_and_ceres_text_state_readers(tmp_path):
    bal_path = tmp_path / "problem.txt"
    bal_path.write_text(
        "1 1 1\n0 0 0.5 1.0\n" + "\n".join(
            map(str, [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 2, -4])) + "\n")
    state_path = tmp_path / "state.txt"
    state_path.write_text(
        "1 1\n" + "\n".join(
            map(str, [0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 2, -4])) + "\n")

    cameras, points, camera_indices, point_indices, observations = read_bal_problem(
        bal_path)
    state_cameras, state_points = read_ceres_text_state(state_path)

    np.testing.assert_allclose(cameras, state_cameras)
    np.testing.assert_allclose(points, state_points)
    metrics = evaluate_bal_state(
        state_cameras, state_points, camera_indices, point_indices, observations)
    assert metrics["sumSquaredError"] == pytest.approx(0.0)


def test_canonicalization_preserves_per_observation_squared_errors(tmp_path):
    cameras = np.array([
        [0.1, -0.2, 0.05, 1, 2, 3, -2, 0.01, -0.001],
        [-0.1, 0.05, 0.2, -2, 1, 4, 3, -0.02, 0.002],
    ])
    points = np.array([[1, 2, -4], [-2, 1, -6], [3, -1, -5]], dtype=float)
    camera_indices = np.array([0, 1, 0, 1])
    point_indices = np.array([0, 1, 2, 0])
    predictions = project_bal(cameras, points, camera_indices, point_indices)
    observations = predictions + np.array([[1, 2], [3, 4], [-2, 1], [1, -3]])

    canonical_cameras, canonical_points, canonical_observations = (
        canonicalize_bal_problem(
            cameras, points, camera_indices, observations))
    original_errors = np.sum(
        (predictions - observations) ** 2, axis=1)
    canonical_errors = np.sum(
        (project_bal(
            canonical_cameras,
            canonical_points,
            camera_indices,
            point_indices,
        ) - canonical_observations) ** 2,
        axis=1,
    )
    np.testing.assert_allclose(
        original_errors, canonical_errors, rtol=1e-10, atol=1e-10)

    output_path = tmp_path / "canonical.txt"
    write_bal_problem(
        output_path,
        canonical_cameras,
        canonical_points,
        camera_indices,
        point_indices,
        canonical_observations,
    )
    round_trip = read_bal_problem(output_path)
    np.testing.assert_allclose(round_trip[0], canonical_cameras)
    np.testing.assert_allclose(round_trip[1], canonical_points)
    np.testing.assert_allclose(round_trip[4], canonical_observations)


def test_canonicalization_can_preserve_raw_scene_coordinates():
    cameras = np.array([
        [0.1, -0.2, 0.05, 1, 2, 3, -2, 0.01, -0.001],
        [-0.1, 0.05, 0.2, -2, 1, 4, 3, -0.02, 0.002],
    ])
    points = np.array([[1, 2, -4], [-2, 1, -6], [3, -1, -5]], dtype=float)
    camera_indices = np.array([0, 1, 0, 1])
    observations = np.array([[1, 2], [3, 4], [-2, 1], [1, -3]], dtype=float)

    canonical_cameras, canonical_points, canonical_observations = (
        canonicalize_bal_problem(
            cameras,
            points,
            camera_indices,
            observations,
            normalize_scene=False,
        )
    )

    np.testing.assert_array_equal(canonical_points, points)
    np.testing.assert_array_equal(canonical_cameras[:, :6], cameras[:, :6])
    np.testing.assert_array_equal(canonical_cameras[:, 6], [2, 3])
    np.testing.assert_array_equal(
        canonical_observations,
        np.array([[-1, -2], [3, 4], [2, -1], [1, -3]], dtype=float),
    )


def test_daba_ray_metric_matches_direct_source_formula():
    cameras = np.array([[0, 0, 0, 0, 0, 1, 2, 0, 0]], dtype=float)
    points = np.array([[1, 2, -4]], dtype=float)
    camera_indices = np.array([0])
    point_indices = np.array([0])
    observations = np.array([[0.5, 1.0]], dtype=float)

    metrics = evaluate_daba_ray_state(
        cameras,
        cameras,
        points,
        camera_indices,
        point_indices,
        observations,
        fit_intrinsics=False,
    )

    normalized = observations[0] / -2.0
    ray = np.array([normalized[0], normalized[1], 1.0])
    distance = -points[0] - cameras[0, 3:6]
    distance_squared = distance.dot(distance) + 1e-12
    residual = ray - (
        distance.dot(ray)
        / (distance_squared + 1e-6 * np.sqrt(distance_squared))
    ) * distance
    sqrt_weight = 2.0 * np.sqrt(normalized.dot(normalized) + 1.0)
    expected_cost = 0.5 * (sqrt_weight * residual).dot(
        sqrt_weight * residual)
    assert metrics["ceresCost"] == pytest.approx(expected_cost)


def test_encoded_daba_ray_metric_matches_direct_source_formula():
    initial_cameras = np.array(
        [[0, 0, 0, 0, 0, 1, 2, 0, 0]], dtype=float)
    cameras = initial_cameras.copy()
    points = np.array([[1, 2, -4]], dtype=float)
    observations = np.array([[0.5, 1.0]], dtype=float)
    metrics = evaluate_encoded_daba_ray_state(
        initial_cameras,
        cameras,
        points,
        np.array([0]),
        np.array([0]),
        observations,
    )
    expected = evaluate_daba_ray_state(
        initial_cameras,
        cameras,
        points,
        np.array([0]),
        np.array([0]),
        observations,
        fit_intrinsics=False,
    )
    assert metrics["ceresCost"] == pytest.approx(expected["ceresCost"])


def test_encoded_daba_ray_matrix_conversion_preserves_pinhole_pixels():
    initial_cameras = np.array(
        [[0, 0, 0, 0, 0, 0, 2, 0, 0]], dtype=float)
    cameras = initial_cameras.copy()
    points = np.array([[-1, -2, -4]], dtype=float)
    daba_cameras, daba_points = encoded_daba_ray_state_to_matrix(
        initial_cameras, cameras, points)
    metrics = evaluate_daba_state_pixel_error(
        initial_cameras,
        daba_cameras,
        daba_points,
        np.array([0]),
        np.array([0]),
        np.array([[-0.5, -1.0]]),
    )
    assert metrics["meanReprojectionError"] == pytest.approx(0.0, abs=1e-10)


def test_daba_inverse_ray_pixel_evaluator_for_pinhole_state(tmp_path):
    state_path = tmp_path / "daba-state.txt"
    camera = np.column_stack((np.eye(3), np.zeros(3), np.array([1, 0, 0])))
    point = np.array([1, 2, 4], dtype=float)
    state_path.write_text(
        "1 1\n"
        + "\n".join(map(str, camera.T.ravel()))
        + "\n"
        + "\n".join(map(str, point))
        + "\n")
    daba_cameras, daba_points = read_daba_ceres_state(state_path)
    initial_cameras = np.array([[0, 0, 0, 0, 0, 0, 2, 0, 0]], dtype=float)
    observations = np.array([[-0.5, -1.0]], dtype=float)

    metrics = evaluate_daba_state_pixel_error(
        initial_cameras,
        daba_cameras,
        daba_points,
        np.array([0]),
        np.array([0]),
        observations,
    )

    assert metrics["meanReprojectionError"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["noninvertibleObservationCount"] == 0
