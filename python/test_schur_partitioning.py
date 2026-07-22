import numpy as np

import palm_ba


def test_schur_refinement_recovers_crossed_couplings() -> None:
    problem = palm_ba.BALProblem(
        cameras=np.zeros((4, 9)),
        points=np.zeros((4, 3)),
        camera_indices=np.array([0, 1, 2, 3]),
        point_indices=np.array([0, 1, 2, 3]),
        observations=np.zeros((4, 2)),
    )
    camera_owner = np.array([0, 0, 1, 1], dtype=np.int64)
    point_owner = camera_owner.copy()
    edge_weights = {(0, 2): 10.0, (1, 3): 10.0}

    refined, _ = palm_ba._refine_partition_schur(
        problem, camera_owner, point_owner, 2, 1, 0.0, 256,
        edge_weights)

    retained = sum(
        weight for (first, second), weight in edge_weights.items()
        if refined[first] == refined[second]
    )
    assert retained == 20.0
    assert np.array_equal(np.bincount(refined, minlength=2), [2, 2])