import numpy as np

import palm_ba
from palm_partition_portfolio import (
    Candidate, make_candidate, select_candidates, select_pilot_winner)


def test_make_candidate_computes_shared_landmark_features() -> None:
    problem = palm_ba.BALProblem(
        cameras=np.zeros((2, 9)),
        points=np.zeros((2, 3)),
        camera_indices=np.array([0, 1, 0, 1]),
        point_indices=np.array([0, 0, 1, 1]),
        observations=np.zeros((4, 2)),
    )

    candidate = make_candidate(problem, 2, None, 0, 0)

    assert candidate.shared_point_count == 2
    assert candidate.shared_point_degree_mean == 2.0


def test_select_candidates_prioritizes_static_partition_score() -> None:
    def candidate(name: str, shared_count: int,
                  shared_degree: float) -> Candidate:
        return Candidate(
            partitioner="overlap",
            seed=None,
            refinement_passes=0,
            camera_owner=np.array([0, 0, 1, 1]),
            point_owner=np.array([0]),
            ownership_hash=name,
            cut_observations=20,
            shared_point_count=shared_count,
            shared_point_degree_mean=shared_degree,
            included_load_ratio=1.02,
            camera_load_ratio=1.01,
        )

    selected = select_candidates([
        candidate("balanced", 10, 2.0),
        candidate("static", 4, 8.0),
        candidate("intermediate", 6, 4.0),
    ], 1)

    assert selected[0].ownership_hash == "static"


def test_analysis_refinement_prefers_high_degree_shared_landmark() -> None:
    problem = palm_ba.BALProblem(
        cameras=np.zeros((4, 9)),
        points=np.zeros((3, 3)),
        camera_indices=np.array([0, 2, 0, 0, 0, 0, 0, 1, 1, 2]),
        point_indices=np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2]),
        observations=np.zeros((10, 2)),
    )
    camera_owner = np.array([0, 0, 1, 1], dtype=np.int64)
    point_owner = np.array([0, 0, 0], dtype=np.int64)

    overlap_cameras, _ = palm_ba._refine_partition_overlap(
        problem, camera_owner.copy(), point_owner.copy(), 2, 1, 0.0, 256)
    analysis_cameras, analysis_points = palm_ba._refine_partition_overlap(
        problem, camera_owner.copy(), point_owner.copy(), 2, 1, 0.0, 256,
        "analysis")

    assert np.array_equal(overlap_cameras, camera_owner)
    assert not np.array_equal(analysis_cameras, camera_owner)
    degrees = np.bincount(problem.point_indices, minlength=len(problem.points))
    shared = np.zeros(len(problem.points), dtype=bool)
    cut = (
        analysis_cameras[problem.camera_indices]
        != analysis_points[problem.point_indices])
    shared[problem.point_indices[cut]] = True
    assert np.sum(1.0 / degrees[shared]) == 2.0 / 3.0


def test_analysis_pilot_requires_confident_improvement() -> None:
    def result(partitioner: str, cost: float) -> dict:
        candidate = Candidate(
            partitioner=partitioner,
            seed=None,
            refinement_passes=0,
            camera_owner=np.array([0]),
            point_owner=np.array([0]),
            ownership_hash=partitioner,
            cut_observations=0,
            shared_point_count=0,
            shared_point_degree_mean=0.0,
            included_load_ratio=1.0,
            camera_load_ratio=1.0,
        )
        return {
            "candidate": candidate,
            "pilot_best_cost": cost,
            "pilot_tail_cost": cost,
        }

    overlap = result("overlap", 100.0)
    close_analysis = result("analysis", 99.6)
    strong_analysis = result("analysis", 99.4)

    assert select_pilot_winner([overlap, close_analysis], 0.5) is overlap
    assert select_pilot_winner([overlap, strong_analysis], 0.5) is strong_analysis