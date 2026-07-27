from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sfm_init_1dsfm import (
    CameraFeatures,
    TriangulationResult,
    average_global_rotations,
    build_translation_graph,
    clean_triangulated_graph,
    inventory_dataset,
    iter_coordinate_cameras,
    iter_tracks,
    mfas_broken_weights,
    one_dsfm_broken_weights,
    sample_projection_directions,
    select_track_cover,
    solve_translation_directions,
    sift_to_bundler,
    triangulate_tracks,
)


def write_dataset(root: Path, *, bad_reference: bool = False) -> None:
    (root / "cc.txt").write_text("2\n7\n", encoding="utf-8")
    (root / "coords.txt").write_text(
        "#index = 2, name = images/a.jpg, keys = 2, px = 4.0, py = 3.0, focal = 10.0\n"
        "0 1.0 1.0 0 0 1 2 3\n"
        "1 8.0 6.0 0 0 4 5 6\n"
        "#index = 7, name = images/b.jpg, keys = 1, px = 5.0, py = 4.0, focal = 12.0\n"
        "0 2.0 3.0 0 0 7 8 9\n",
        encoding="utf-8",
    )
    feature = 2 if bad_reference else 0
    (root / "tracks.txt").write_text(
        f"2\n2 2 0 7 {feature}\n1 2 1\n", encoding="utf-8"
    )
    identity = "1 0 0 0 1 0 0 0 1"
    (root / "EGs.txt").write_text(
        f"2 7 {identity} 1 0 0\n7 9 {identity} 0 1 0\n",
        encoding="utf-8",
    )


def test_sift_to_bundler_matches_sfminit_half_pixel_convention() -> None:
    converted = sift_to_bundler(np.array([[1.0, 1.0], [8.0, 6.0]]), (8.0, 6.0))
    np.testing.assert_allclose(converted, [[-3.5, 2.5], [3.5, -2.5]])


def test_streaming_parsers_and_inventory(tmp_path: Path) -> None:
    write_dataset(tmp_path)
    cameras = list(iter_coordinate_cameras(tmp_path / "coords.txt"))
    tracks = list(iter_tracks(tmp_path / "tracks.txt"))
    assert [(camera.index, camera.key_count) for camera in cameras] == [(2, 2), (7, 1)]
    assert [track.tolist() for track in tracks] == [[[2, 0], [7, 0]], [[2, 1]]]

    inventory = inventory_dataset(tmp_path)
    assert inventory.component_cameras == 2
    assert inventory.coordinate_features == 3
    assert inventory.epipolar_geometries == 2
    assert inventory.component_epipolar_geometries == 1
    assert inventory.declared_tracks == 2
    assert inventory.track_observations == 3
    assert inventory.component_track_observations == 3
    assert inventory.tracks_with_two_component_views == 1
    assert inventory.tracks_with_three_component_views == 0
    assert inventory.invalid_track_references == 0
    assert set(inventory.source_sha256) == {"cc.txt", "coords.txt", "tracks.txt", "EGs.txt"}


def test_inventory_counts_invalid_track_references(tmp_path: Path) -> None:
    write_dataset(tmp_path, bad_reference=True)
    assert inventory_dataset(tmp_path).invalid_track_references == 1


def test_track_declared_degree_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "tracks.txt"
    path.write_text("1\n2 2 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid degree"):
        list(iter_tracks(path))


def test_mfas_broken_weights_preserves_original_ordering_rule() -> None:
    edges = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
    weights = np.array([2.0, 1.0, 0.5, -3.0])
    np.testing.assert_allclose(mfas_broken_weights(edges, weights), [0.0, 0.0, 0.5, 0.0])


def test_projection_sampling_and_voting_are_deterministic() -> None:
    edges = np.array([[10, 20], [20, 30], [30, 10], [30, 40]], dtype=np.int64)
    poses = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    first_directions = sample_projection_directions(poses, 8, seed=17)
    second_directions = sample_projection_directions(poses, 8, seed=17)
    np.testing.assert_allclose(first_directions, second_directions)
    np.testing.assert_allclose(np.linalg.norm(first_directions, axis=1), 1.0)
    first = one_dsfm_broken_weights(edges, poses, 8, seed=17)
    second = one_dsfm_broken_weights(edges, poses, 8, seed=17)
    np.testing.assert_allclose(first, second)
    assert np.all(first >= 0.0)


def make_epipolar_graph(
    rotations: np.ndarray, edges: np.ndarray, outlier_edge: int | None = None
) -> np.ndarray:
    rows = []
    for index, (source, destination) in enumerate(edges):
        relative = rotations[destination] @ rotations[source].T
        if index == outlier_edge:
            relative = Rotation.from_rotvec([1.2, -0.7, 0.4]).as_matrix()
        rows.append(np.concatenate(([source, destination], relative.T.reshape(-1), [1, 0, 0])))
    return np.asarray(rows)


def test_rotation_averaging_recovers_exact_graph_with_sfminit_transpose() -> None:
    expected = Rotation.from_rotvec([
        [0.0, 0.0, 0.0],
        [0.1, -0.2, 0.05],
        [-0.15, 0.02, 0.25],
        [0.07, 0.12, -0.18],
    ]).as_matrix()
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]])
    _, actual, diagnostics = average_global_rotations(
        make_epipolar_graph(expected, edges), np.arange(4)
    )
    errors = Rotation.from_matrix(actual @ np.swapaxes(expected, 1, 2)).magnitude()
    assert np.max(errors) < 1e-8
    assert diagnostics["finalMedianResidualDegrees"] < 1e-8


def test_rotation_averaging_rejects_disconnected_graph() -> None:
    rotations = np.repeat(np.eye(3)[np.newaxis], 4, axis=0)
    edges = np.array([[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="disconnected"):
        average_global_rotations(make_epipolar_graph(rotations, edges), np.arange(4))


def reference_track_cover(tracks: list[np.ndarray], coverage: int) -> set[int]:
    maximum_camera = max(int(np.max(track[:, 0])) for track in tracks)
    remaining = np.full(maximum_camera + 1, coverage)
    improvement = [len(track) for track in tracks]
    lookup = [[] for _ in range(maximum_camera + 1)]
    for index, track in enumerate(tracks):
        for camera in track[:, 0]:
            lookup[int(camera)].append(index)
    selected = set()
    while True:
        index = int(np.argmax(improvement))
        if improvement[index] <= 0:
            break
        selected.add(index)
        improvement[index] = 0
        for camera_value in tracks[index][:, 0]:
            camera = int(camera_value)
            if remaining[camera] == 1:
                for track_index in lookup[camera]:
                    improvement[track_index] = max(0, improvement[track_index] - 1)
            remaining[camera] = max(0, remaining[camera] - 1)
    return selected


def test_track_cover_matches_original_greedy_algorithm() -> None:
    tracks = [
        np.array([[0, 0], [1, 0], [2, 0]]),
        np.array([[0, 1], [1, 1]]),
        np.array([[1, 2], [2, 1]]),
        np.array([[0, 2], [2, 2]]),
    ]
    assert set(select_track_cover(tracks, 2)) == reference_track_cover(tracks, 2)


def test_translation_graph_uses_world_directions_and_minimum_views() -> None:
    camera_ids = np.array([0, 1])
    rotations = Rotation.from_rotvec([
        [0.0, 0.0, np.pi / 2.0],
        [0.0, 0.0, 0.0],
    ]).as_matrix()
    relative = rotations[1] @ rotations[0].T
    geometry = np.asarray([
        np.concatenate(([0, 1], relative.T.reshape(-1), [1.0, 0.0, 0.0]))
    ])
    tracks = [np.array([[0, 0], [1, 0]]), np.array([[0, 1]])]
    metadata = CameraFeatures(0, "a", 2, 8.0, 6.0, 10.0)
    metadata_one = CameraFeatures(1, "b", 1, 8.0, 6.0, 10.0)
    coordinates = {
        0: (metadata, np.array([[4.5, 3.5], [1.0, 1.0]])),
        1: (metadata_one, np.array([[4.5, 3.5]])),
    }
    edges, poses, weights, retained, diagnostics = build_translation_graph(
        geometry, camera_ids, rotations, tracks, coordinates, track_coverage=1
    )
    np.testing.assert_allclose(poses[0], rotations[0].T @ [1.0, 0.0, 0.0])
    assert retained.tolist() == [0]
    assert diagnostics["cameraPointEdges"] == 2
    assert edges.shape == (3, 2)
    np.testing.assert_allclose(weights, [1.0, 0.25, 0.25])


def test_translation_solver_recovers_exact_directions() -> None:
    expected = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ])
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]])
    poses = expected[edges[:, 1]] - expected[edges[:, 0]]
    poses /= np.linalg.norm(poses, axis=1)[:, np.newaxis]
    node_ids, actual, diagnostics = solve_translation_directions(
        edges, poses, np.ones(len(edges)), maximum_iterations=200
    )
    predicted = actual[edges[:, 1]] - actual[edges[:, 0]]
    predicted /= np.linalg.norm(predicted, axis=1)[:, np.newaxis]
    np.testing.assert_allclose(predicted, poses, atol=1e-6)
    np.testing.assert_array_equal(node_ids, np.arange(4))
    assert diagnostics["medianAngularResidualDegrees"] < 1e-5


def test_multiview_triangulation_recovers_exact_point() -> None:
    camera_ids = np.array([0, 1, 2])
    rotations = np.repeat(np.eye(3)[np.newaxis], 3, axis=0)
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    point = np.array([0.2, 0.3, -3.0])
    focal = 100.0
    coordinates = {}
    track = []
    for camera, center in enumerate(centers):
        local = point - center
        bundler = -focal * local[:2] / local[2]
        feature = np.array([50.5 + bundler[0], 40.5 - bundler[1]])
        metadata = CameraFeatures(camera, str(camera), 1, 100.0, 80.0, focal)
        coordinates[camera] = (metadata, feature[np.newaxis])
        track.append((camera, 0))
    result = triangulate_tracks(
        camera_ids,
        rotations,
        centers,
        [np.asarray(track)],
        coordinates,
    )
    np.testing.assert_allclose(result.points[0], point, atol=1e-10)
    assert result.positive_depth_fractions[0] == 1.0
    assert result.maximum_reprojection_errors[0] < 1e-10


def test_degree_cleanup_cascades_from_cameras_to_points() -> None:
    triangulation = TriangulationResult(
        track_ids=np.arange(3),
        points=np.zeros((3, 3)),
        view_counts=np.full(3, 2),
        positive_depth_fractions=np.ones(3),
        maximum_parallax_degrees=np.full(3, 2.0),
        condition_numbers=np.full(3, 10.0),
        mean_reprojection_errors=np.full(3, 1.0),
        median_reprojection_errors=np.full(3, 1.0),
        maximum_reprojection_errors=np.full(3, 2.0),
    )
    tracks = [
        np.array([[0, 0], [1, 0]]),
        np.array([[0, 1], [1, 1]]),
        np.array([[1, 2], [2, 0]]),
    ]
    cameras, indices, observations, diagnostics = clean_triangulated_graph(
        triangulation,
        tracks,
        np.arange(3),
        minimum_camera_points=2,
    )
    np.testing.assert_array_equal(cameras, [0, 1])
    np.testing.assert_array_equal(indices, [0, 1])
    assert len(observations) == 2
    assert diagnostics["retainedObservations"] == 4
