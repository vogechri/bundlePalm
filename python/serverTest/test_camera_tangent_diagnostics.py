import numpy as np

from camera_tangent_diagnostics import (
    diagonal_weighted_tangent_alignment,
    left_se3_camera_minus,
    left_se3_camera_plus,
    project_tangent_orthogonal_to_basis,
    tangent_alignment,
)


def test_left_se3_camera_minus_round_trips_plus():
    cameras = np.array([
        [0.1, -0.2, 0.05, 1.0, -2.0, 3.0, 800.0, 0.01, -0.001],
        [-0.05, 0.03, 0.2, -4.0, 2.0, 1.0, 500.0, -0.02, 0.003],
    ])
    tangent = np.array([
        [0.02, -0.03, 0.01, 0.005, -0.004, 0.003, 1.0, 1e-4, -2e-5],
        [-0.01, 0.04, -0.02, -0.002, 0.006, -0.005, -0.5, 2e-4, 1e-5],
    ])

    updated = left_se3_camera_plus(cameras, tangent)
    recovered = left_se3_camera_minus(updated, cameras)

    np.testing.assert_allclose(recovered, tangent, rtol=1e-10, atol=1e-10)


def test_tangent_alignment_reports_subspace_cosines():
    reference = np.zeros((2, 9))
    reference[:, :3] = 1.0
    reference[:, 3:6] = 2.0
    reference[:, 6:9] = 3.0
    candidate = reference.copy()
    candidate[:, 3:6] *= -1.0

    diagnostics = tangent_alignment(reference, candidate)

    np.testing.assert_allclose(diagnostics["translation"]["cosine"], 1.0)
    np.testing.assert_allclose(diagnostics["rotation"]["cosine"], -1.0)
    np.testing.assert_allclose(diagnostics["intrinsics"]["cosine"], 1.0)
    assert diagnostics["global"]["cosine"] < 1.0


def test_diagonal_weighted_alignment_removes_parameter_unit_bias():
    reference = np.ones((1, 9))
    candidate = reference.copy()
    candidate[0, 6] = -1.0
    diagonal = np.ones((1, 9))
    diagonal[0, 6] = 1e-12

    unweighted = tangent_alignment(reference, candidate)
    weighted = diagonal_weighted_tangent_alignment(
        reference, candidate, diagonal
    )

    assert weighted["global"]["cosine"] > unweighted["global"]["cosine"]
    np.testing.assert_allclose(weighted["translation"]["cosine"], 1.0)


def test_tangent_projection_removes_orthonormal_basis_component():
    tangent = np.arange(1.0, 19.0).reshape(2, 9)
    basis = np.zeros((tangent.size, 2))
    basis[0, 0] = 1.0
    basis[10, 1] = 1.0

    residual, diagnostics = project_tangent_orthogonal_to_basis(
        tangent, basis
    )

    assert residual.ravel()[0] == 0.0
    assert residual.ravel()[10] == 0.0
    np.testing.assert_allclose(
        diagnostics["componentNorm"], np.hypot(tangent.ravel()[0], tangent.ravel()[10])
    )
    np.testing.assert_allclose(
        diagnostics["residualNorm"], np.linalg.norm(residual)
    )


def test_tangent_projection_is_orthogonal_in_diagonal_metric():
    tangent = np.zeros((1, 9))
    tangent[0, :2] = [2.0, 3.0]
    basis = np.zeros((tangent.size, 1))
    basis[:2, 0] = [1.0, 1.0]
    diagonal = np.ones_like(tangent)
    diagonal[0, 1] = 4.0

    residual, _ = project_tangent_orthogonal_to_basis(
        tangent, basis, diagonal
    )

    np.testing.assert_allclose(
        basis[:, 0] @ (diagonal.ravel() * residual.ravel()), 0.0, atol=1e-12
    )