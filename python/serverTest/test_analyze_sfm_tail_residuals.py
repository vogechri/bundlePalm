import numpy as np
import pytest

from analyze_sfm_tail_residuals import analyze, concentration


def test_concentration_reports_top_fraction_energy():
    values = np.zeros(100)
    values[0] = 9.0
    values[1:] = 1.0 / 99.0

    result = concentration(values)

    assert result["0.01"] == pytest.approx(0.9)
    assert result["0.05"] > result["0.01"]
    assert result["0.1"] > result["0.05"]


def test_identical_state_has_zero_tangent_and_excess(tmp_path):
    cameras = np.zeros((2, 9))
    cameras[:, 6] = 1.0
    cameras[1, 3] = 1.0
    points = np.array([[0.0, 0.0, -2.0]])
    camera_indices = np.array([0, 1], dtype=np.int64)
    point_indices = np.array([0, 0], dtype=np.int64)
    observations = np.array([[0.0, 0.0], [0.5, 0.0]])
    problem = (
        cameras,
        points,
        camera_indices,
        point_indices,
        observations,
    )
    state = tmp_path / "state.npz"
    np.savez(state, cameras=cameras, points=points)

    result = analyze(problem, state, state)

    assert result["candidateSSE"] == result["referenceSSE"]
    assert all(
        value == 0.0
        for value in result["positiveCameraExcessConcentration"].values()
    )
    assert all(
        value == 0.0 for value in result["tangentEnergyFractions"].values()
    )
