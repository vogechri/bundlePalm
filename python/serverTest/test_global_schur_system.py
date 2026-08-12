import numpy as np

from client_admm import ClusterSchurSystem
from client_drs import (
    SchurBSRSymbolicCache,
    similarity_gauge_tangent_basis,
    solve_global_schur_system,
)


def make_system(scale=1.0, extra_edge=False):
    identity = np.eye(9)
    rows = [0, 0, 1]
    columns = [0, 1, 1]
    blocks = [
        4.0 * identity,
        -0.25 * identity,
        3.0 * identity,
    ]
    if extra_edge:
        rows.append(1)
        columns.append(2)
        blocks.append(-0.1 * identity)
    return ClusterSchurSystem(
        camera_ids=np.arange(3, dtype=np.int64),
        block_rows=np.asarray(rows, dtype=np.int64),
        block_columns=np.asarray(columns, dtype=np.int64),
        blocks=scale * np.asarray(blocks),
        reduced_gradient=(
            scale * np.arange(27, dtype=np.float64).reshape(3, 9)
        ),
        camera_diagonal=scale * np.repeat(identity[None], 3, axis=0),
        landmark_model_reduction=1.0,
    )


def solve(systems, operator_mode, cache=None):
    return solve_global_schur_system(
        systems,
        camera_count=3,
        camera_damping=3.0,
        step_scale=1.0,
        linear_solver="cg",
        relative_tolerance=1e-10,
        maximum_iterations=200,
        operator_mode=operator_mode,
        bsr_symbolic_cache=cache,
    )


def test_cached_bsr_matches_python_and_reuses_symbolic_pattern():
    systems = [make_system()]
    expected, _ = solve(systems, "python")
    cache = SchurBSRSymbolicCache()

    first, first_diagnostics = solve(systems, "bsr_low_memory", cache)
    second, second_diagnostics = solve(
        [make_system(scale=1.25)], "bsr_low_memory", cache
    )
    second_expected, _ = solve([make_system(scale=1.25)], "python")

    np.testing.assert_allclose(first, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(second, second_expected, rtol=1e-11, atol=1e-11)
    assert not first_diagnostics["symbolicCacheHit"]
    assert second_diagnostics["symbolicCacheHit"]
    assert cache.builds == 1
    assert cache.hits == 1


def test_cached_bsr_rebuilds_when_sparsity_changes():
    cache = SchurBSRSymbolicCache()
    solve([make_system()], "bsr_low_memory", cache)
    actual, diagnostics = solve(
        [make_system(extra_edge=True)], "bsr_low_memory", cache
    )
    expected, _ = solve([make_system(extra_edge=True)], "python")

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert not diagnostics["symbolicCacheHit"]
    assert cache.builds == 2


def test_cached_bsr_duplicate_source_uses_safe_accumulation():
    system = make_system()
    system.block_rows = np.concatenate((system.block_rows, [0]))
    system.block_columns = np.concatenate((system.block_columns, [1]))
    system.blocks = np.concatenate((system.blocks, system.blocks[1:2]))
    cache = SchurBSRSymbolicCache()

    actual, _ = solve([system], "bsr_low_memory", cache)
    expected, _ = solve([system], "python")

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert not all(cache.source_is_unique)


def test_gauge_deflated_preconditioner_matches_jacobi_solution():
    systems = [make_system()]
    expected, _ = solve(systems, "bsr_low_memory")
    coarse_basis = np.zeros((27, 2))
    coarse_basis[::9, 0] = 1.0
    coarse_basis[1::9, 1] = 1.0
    actual, diagnostics = solve_global_schur_system(
        systems,
        camera_count=3,
        camera_damping=3.0,
        step_scale=1.0,
        linear_solver="cg",
        relative_tolerance=1e-10,
        maximum_iterations=200,
        operator_mode="bsr_low_memory",
        preconditioner_mode="gauge_deflated",
        coarse_basis=coarse_basis,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert diagnostics["preconditioner"] == "gauge_deflated"
    assert diagnostics["coarseBasisRank"] == 2


def test_similarity_gauge_basis_has_seven_orthonormal_modes():
    cameras = np.zeros((4, 9))
    cameras[:, :3] = np.array([
        [0.1, -0.2, 0.3],
        [-0.2, 0.1, 0.05],
        [0.05, 0.2, -0.1],
        [0.3, 0.1, -0.2],
    ])
    cameras[:, 3:6] = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 3.0],
        [-1.0, 0.5, 2.5],
        [0.5, -1.0, 4.0],
    ])

    basis = similarity_gauge_tangent_basis(cameras)

    assert basis.shape == (36, 7)
    np.testing.assert_allclose(basis.T @ basis, np.eye(7), atol=1e-12)