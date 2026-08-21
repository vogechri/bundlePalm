from types import SimpleNamespace

import numpy as np

from drs_coupled_metrics import assemble_coupled_metric
from schur_consensus_oracle import (
    jacobi_refine_schur_tangent,
    project_stabilized_coupled_tangents,
    stabilized_coupled_metric_from_schur_systems,
)


def make_system(blocks):
    rows, columns, values = zip(*blocks)
    return SimpleNamespace(
        block_rows=np.asarray(rows),
        block_columns=np.asarray(columns),
        blocks=np.asarray(values),
    )


def test_stabilized_coupled_metric_adds_offdiagonal_frobenius_bounds():
    identity = np.eye(9)
    coupling = np.zeros((9, 9))
    coupling[0, 0] = -2.0
    system = make_system([
        (0, 0, 3.0 * identity),
        (0, 1, coupling),
        (1, 1, 4.0 * identity),
    ])

    metric = stabilized_coupled_metric_from_schur_systems(
        [system], np.ones((1, 2), dtype=bool), 2, shared_only=False
    )
    matrix = assemble_coupled_metric(
        metric, np.ones((1, 2), dtype=bool)
    ).toarray()

    np.testing.assert_allclose(matrix[:9, :9], 5.0 * identity)
    np.testing.assert_allclose(matrix[9:, 9:], 6.0 * identity)
    np.testing.assert_allclose(matrix[:9, 9:], coupling)
    assert np.min(np.linalg.eigvalsh(matrix)) > 0.0


def test_coupled_tangent_oracle_preserves_direct_singletons():
    identity = np.eye(9)
    systems = [
        make_system([
            (0, 0, 2.0 * identity),
            (0, 1, -0.25 * identity),
            (1, 1, 2.0 * identity),
            (2, 2, identity),
        ]),
        make_system([
            (0, 0, identity),
            (0, 1, -0.1 * identity),
            (1, 1, identity),
        ]),
    ]
    masks = np.array([[True, True, True], [True, True, False]])
    tangents = np.zeros((2, 3, 9))
    tangents[0, :, 0] = [1.0, 3.0, 7.0]
    tangents[1, :2, 0] = [2.0, -1.0]

    projected = project_stabilized_coupled_tangents(
        tangents, masks, systems
    )

    assert projected[2, 0] == 0.0
    assert np.all(np.isfinite(projected))


def test_jacobi_refinement_solves_diagonal_schur_system_in_one_step():
    identity = np.eye(9)
    system = SimpleNamespace(
        camera_ids=np.array([0, 1]),
        block_rows=np.array([0, 1]),
        block_columns=np.array([0, 1]),
        blocks=np.array([2.0 * identity, 3.0 * identity]),
        reduced_gradient=np.array([
            np.full(9, -5.0),
            np.full(9, -7.0),
        ]),
        camera_diagonal=np.array([identity, identity]),
    )

    refined, diagnostics = jacobi_refine_schur_tangent(
        [system],
        camera_count=2,
        camera_damping=0.5,
        tangent=np.zeros((2, 9)),
        active_cameras=np.array([True, False]),
    )

    np.testing.assert_allclose(refined[0], 2.0)
    np.testing.assert_allclose(refined[1], 0.0)
    assert diagnostics["activeCameraCount"] == 1


def test_second_jacobi_refinement_contracts_coupled_residual():
    identity = np.eye(9)
    system = SimpleNamespace(
        camera_ids=np.array([0, 1]),
        block_rows=np.array([0, 0, 1]),
        block_columns=np.array([0, 1, 1]),
        blocks=np.array([
            2.0 * identity,
            -0.5 * identity,
            2.0 * identity,
        ]),
        reduced_gradient=np.array([
            np.full(9, -1.0),
            np.full(9, -2.0),
        ]),
        camera_diagonal=np.array([identity, identity]),
    )

    refined, diagnostics = jacobi_refine_schur_tangent(
        [system],
        camera_count=2,
        camera_damping=0.5,
        tangent=np.zeros((2, 9)),
        active_cameras=np.array([True, True]),
        refinement_steps=2,
    )

    assert diagnostics["refinementSteps"] == 2
    assert diagnostics["steps"][1]["residualNorm"] < (
        diagnostics["steps"][0]["residualNorm"]
    )
    assert np.all(np.isfinite(refined))


def test_model_optimal_second_step_maximizes_directional_model_decrease():
    identity = np.eye(9)
    system = SimpleNamespace(
        camera_ids=np.array([0, 1]),
        block_rows=np.array([0, 0, 1]),
        block_columns=np.array([0, 1, 1]),
        blocks=np.array([
            2.0 * identity,
            -0.5 * identity,
            2.0 * identity,
        ]),
        reduced_gradient=np.array([
            np.full(9, -1.0),
            np.full(9, -2.0),
        ]),
        camera_diagonal=np.array([identity, identity]),
    )
    arguments = (
        [system],
        2,
        0.5,
        np.zeros((2, 9)),
        np.array([True, True]),
    )

    _, unit = jacobi_refine_schur_tangent(
        *arguments, refinement_steps=2
    )
    refined, optimal = jacobi_refine_schur_tangent(
        *arguments,
        refinement_steps=2,
        model_optimal_after_first=True,
    )

    assert optimal["steps"][1]["stepScale"] > 0.0
    assert optimal["steps"][1]["modelDecrease"] >= (
        unit["steps"][1]["modelDecrease"]
    )
    assert np.all(np.isfinite(refined))