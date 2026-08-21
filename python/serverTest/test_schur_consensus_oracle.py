from types import SimpleNamespace

import numpy as np

from drs_coupled_metrics import assemble_coupled_metric
from schur_consensus_oracle import (
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