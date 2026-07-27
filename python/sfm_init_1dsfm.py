"""Parse and inventory public 1DSfM datasets using SfM_Init conventions."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import heapq
import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation
from scipy.stats import gaussian_kde


_COORDS_HEADER = re.compile(
    r"^#index = (?P<index>\d+), name = (?P<name>\S+), "
    r"keys = (?P<keys>\d+), px = (?P<px>[-+\d.eE]+), "
    r"py = (?P<py>[-+\d.eE]+), focal = (?P<focal>[-+\d.eE]+)$"
)


@dataclass(frozen=True)
class CameraFeatures:
    index: int
    name: str
    key_count: int
    width: float
    height: float
    focal: float


@dataclass(frozen=True)
class DatasetInventory:
    dataset: str
    component_cameras: int
    coordinate_cameras: int
    coordinate_features: int
    epipolar_geometries: int
    component_epipolar_geometries: int
    declared_tracks: int
    track_observations: int
    component_track_observations: int
    tracks_with_two_component_views: int
    tracks_with_three_component_views: int
    invalid_track_references: int
    source_sha256: dict[str, str]


@dataclass(frozen=True)
class TriangulationResult:
    track_ids: np.ndarray
    points: np.ndarray
    view_counts: np.ndarray
    positive_depth_fractions: np.ndarray
    maximum_parallax_degrees: np.ndarray
    condition_numbers: np.ndarray
    mean_reprojection_errors: np.ndarray
    median_reprojection_errors: np.ndarray
    maximum_reprojection_errors: np.ndarray


def sift_to_bundler(
    coordinates: np.ndarray | tuple[float, float],
    image_dimensions: np.ndarray | tuple[float, float],
) -> np.ndarray:
    """Convert SIFT coordinates to centered Bundler coordinates."""
    point = np.asarray(coordinates, dtype=np.float64)
    dimensions = np.asarray(image_dimensions, dtype=np.float64)
    if point.shape[-1] != 2 or dimensions.shape != (2,):
        raise ValueError("coordinates and image_dimensions must have size two")
    result = np.empty_like(point, dtype=np.float64)
    result[..., 0] = point[..., 0] - 0.5 * (dimensions[0] + 1.0)
    result[..., 1] = 0.5 * (dimensions[1] + 1.0) - point[..., 1]
    return result


def sample_projection_directions(
    poses: np.ndarray,
    number_of_samples: int = 48,
    seed: int = 0,
    maximum_kde_inputs: int = 2000,
) -> np.ndarray:
    """Sample deterministic 1DSfM projection directions from pose density."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 2:
        raise ValueError("poses must have shape (N, 3) with N >= 2")
    if number_of_samples <= 0 or maximum_kde_inputs <= 1:
        raise ValueError("sample counts must be positive")
    norms = np.linalg.norm(poses, axis=1)
    if np.any(~np.isfinite(poses)) or np.any(norms <= 0):
        raise ValueError("poses must contain finite nonzero vectors")
    unit_poses = poses / norms[:, np.newaxis]
    zenith = np.arccos(np.clip(unit_poses[:, 1], -1.0, 1.0))
    azimuth = np.arctan2(unit_poses[:, 0], -unit_poses[:, 2])
    rng = np.random.default_rng(seed)
    if len(poses) > maximum_kde_inputs:
        selected = rng.choice(len(poses), maximum_kde_inputs, replace=False)
        azimuth = azimuth[selected]
        zenith = zenith[selected]
    samples = gaussian_kde(np.vstack((azimuth, zenith))).resample(
        size=number_of_samples, seed=rng
    )
    sampled_azimuth, sampled_zenith = samples
    directions = np.column_stack((
        np.sin(sampled_azimuth) * np.sin(sampled_zenith),
        np.cos(sampled_zenith),
        -np.cos(sampled_azimuth) * np.sin(sampled_zenith),
    ))
    return directions


def mfas_broken_weights(
    edges: np.ndarray,
    weights: np.ndarray,
    library_path: str | Path | None = None,
) -> np.ndarray:
    """Run SfM_Init's weighted MFAS heuristic and return broken-edge weights."""
    edges = np.ascontiguousarray(edges, dtype=np.int64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    if edges.ndim != 2 or edges.shape[1] != 2 or weights.shape != (len(edges),):
        raise ValueError("edges must have shape (N, 2) and weights shape (N,)")
    if np.any(edges < 0) or np.any(~np.isfinite(weights)):
        raise ValueError("edges and weights contain invalid values")
    if library_path is None:
        library_path = Path(__file__).with_name("libsfm_init_mfas.so")
    library = ctypes.CDLL(str(Path(library_path).resolve()))
    function = library.sfm_init_mfas_broken_weights
    function.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int64,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ]
    function.restype = ctypes.c_int
    broken = np.empty(len(edges), dtype=np.float64)
    status = function(edges, weights, len(edges), broken)
    if status != 0:
        raise RuntimeError(f"MFAS kernel failed with status {status}")
    return broken


def one_dsfm_broken_weights(
    edges: np.ndarray,
    poses: np.ndarray,
    number_of_samples: int = 48,
    seed: int = 0,
    library_path: str | Path | None = None,
) -> np.ndarray:
    """Compute deterministic SfM_Init broken-edge voting weights."""
    edges = np.asarray(edges, dtype=np.int64)
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (len(edges), 3):
        raise ValueError("poses must have shape (N, 3) matching edges")
    unique_nodes, reindexed = np.unique(edges, return_inverse=True)
    if unique_nodes.size == 0:
        return np.empty(0, dtype=np.float64)
    reindexed_edges = reindexed.reshape(edges.shape)
    directions = sample_projection_directions(
        poses, number_of_samples=number_of_samples, seed=seed
    )
    broken = np.zeros(len(edges), dtype=np.float64)
    for direction in directions:
        broken += mfas_broken_weights(
            reindexed_edges, poses @ direction, library_path=library_path
        )
    return broken / number_of_samples


def solve_translation_directions(
    edges: np.ndarray,
    poses: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int = 20260727,
    maximum_iterations: int = 1000,
    function_tolerance: float = 1e-14,
    parameter_tolerance: float = 1e-14,
    library_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Solve SfM_Init's weighted chordal translation-direction objective."""
    edges = np.asarray(edges, dtype=np.int64)
    poses = np.asarray(poses, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (N, 2)")
    if poses.shape != (len(edges), 3) or weights.shape != (len(edges),):
        raise ValueError("poses and weights must match edges")
    node_ids, reindexed = np.unique(edges, return_inverse=True)
    reindexed_edges = np.ascontiguousarray(reindexed.reshape(edges.shape), dtype=np.int64)
    poses = np.ascontiguousarray(poses, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    if library_path is None:
        library_path = Path(__file__).with_name("libsfm_init_translation.so")
    library = ctypes.CDLL(str(Path(library_path).resolve()))
    function = library.sfm_init_solve_translations
    function.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    function.restype = ctypes.c_int
    positions = np.empty((len(node_ids), 3), dtype=np.float64)
    final_cost = ctypes.c_double()
    completed_iterations = ctypes.c_int()
    status = function(
        reindexed_edges,
        poses,
        weights,
        len(edges),
        len(node_ids),
        seed,
        maximum_iterations,
        function_tolerance,
        parameter_tolerance,
        positions,
        ctypes.byref(final_cost),
        ctypes.byref(completed_iterations),
    )
    if status != 0:
        raise RuntimeError(f"translation solver failed with status {status}")
    positions -= positions[0]
    edge_lengths = np.linalg.norm(
        positions[reindexed_edges[:, 1]] - positions[reindexed_edges[:, 0]], axis=1
    )
    positive_lengths = edge_lengths[edge_lengths > 0]
    if not np.all(np.isfinite(positions)) or positive_lengths.size == 0:
        raise RuntimeError("translation solver returned a degenerate solution")
    scale = float(np.median(positive_lengths))
    positions /= scale
    predicted = positions[reindexed_edges[:, 1]] - positions[reindexed_edges[:, 0]]
    predicted /= np.linalg.norm(predicted, axis=1)[:, np.newaxis]
    angular_errors = np.arccos(np.clip(np.sum(predicted * poses, axis=1), -1.0, 1.0))
    diagnostics: dict[str, float | int] = {
        "nodes": len(node_ids),
        "edges": len(edges),
        "randomSeed": seed,
        "iterations": completed_iterations.value,
        "finalCeresCost": final_cost.value,
        "gaugeMedianEdgeLength": scale,
        "medianAngularResidualDegrees": float(np.degrees(np.median(angular_errors))),
        "p95AngularResidualDegrees": float(np.degrees(np.percentile(angular_errors, 95))),
    }
    return node_ids, positions, diagnostics


def triangulate_tracks(
    camera_ids: np.ndarray,
    rotations: np.ndarray,
    camera_positions: np.ndarray,
    tracks: list[np.ndarray],
    coordinates: dict[int, tuple[CameraFeatures, np.ndarray]],
    *,
    minimum_views: int = 2,
) -> TriangulationResult:
    """Triangulate all eligible tracks by multi-ray least squares."""
    if rotations.shape != (len(camera_ids), 3, 3):
        raise ValueError("rotations do not match camera_ids")
    if camera_positions.shape != (len(camera_ids), 3):
        raise ValueError("camera_positions do not match camera_ids")
    if minimum_views < 2:
        raise ValueError("minimum_views must be at least two")
    camera_lookup = {
        int(camera): index for index, camera in enumerate(camera_ids)
    }
    track_ids: list[int] = []
    points: list[np.ndarray] = []
    view_counts: list[int] = []
    positive_fractions: list[float] = []
    parallaxes: list[float] = []
    conditions: list[float] = []
    means: list[float] = []
    medians: list[float] = []
    maxima: list[float] = []
    identity = np.eye(3)
    for track_id, track in enumerate(tracks):
        observations: list[tuple[int, np.ndarray]] = []
        seen_cameras: set[int] = set()
        for camera_value, feature_value in track:
            camera = int(camera_value)
            feature = int(feature_value)
            if (
                camera in seen_cameras
                or camera not in camera_lookup
                or camera not in coordinates
            ):
                continue
            metadata, features = coordinates[camera]
            if metadata.focal <= 0 or not 0 <= feature < len(features):
                continue
            observations.append((camera, features[feature]))
            seen_cameras.add(camera)
        if len(observations) < minimum_views:
            continue
        directions = []
        centers = []
        for camera, feature in observations:
            camera_index = camera_lookup[camera]
            metadata, _ = coordinates[camera]
            x, y = sift_to_bundler(feature, (metadata.width, metadata.height))
            local_direction = np.array((x, y, -metadata.focal))
            local_direction /= np.linalg.norm(local_direction)
            directions.append(rotations[camera_index].T @ local_direction)
            centers.append(camera_positions[camera_index])
        directions_array = np.asarray(directions)
        centers_array = np.asarray(centers)
        projectors = identity - directions_array[:, :, np.newaxis] * directions_array[:, np.newaxis, :]
        system = np.sum(projectors, axis=0)
        condition = float(np.linalg.cond(system))
        if not np.isfinite(condition):
            continue
        point = np.linalg.solve(system, np.einsum("nij,nj->i", projectors, centers_array))
        depths = np.sum((point - centers_array) * directions_array, axis=1)
        dot_products = np.clip(directions_array @ directions_array.T, -1.0, 1.0)
        maximum_parallax = float(np.degrees(np.arccos(np.min(dot_products))))
        errors = []
        for camera, feature in observations:
            camera_index = camera_lookup[camera]
            metadata, _ = coordinates[camera]
            camera_point = rotations[camera_index] @ (point - camera_positions[camera_index])
            if camera_point[2] == 0:
                errors.append(np.inf)
                continue
            prediction = -metadata.focal * camera_point[:2] / camera_point[2]
            observation = sift_to_bundler(feature, (metadata.width, metadata.height))
            errors.append(float(np.linalg.norm(prediction - observation)))
        errors_array = np.asarray(errors)
        track_ids.append(track_id)
        points.append(point)
        view_counts.append(len(observations))
        positive_fractions.append(float(np.mean(depths > 0)))
        parallaxes.append(maximum_parallax)
        conditions.append(condition)
        means.append(float(np.mean(errors_array)))
        medians.append(float(np.median(errors_array)))
        maxima.append(float(np.max(errors_array)))
    return TriangulationResult(
        track_ids=np.asarray(track_ids, dtype=np.int64),
        points=np.asarray(points, dtype=np.float64),
        view_counts=np.asarray(view_counts, dtype=np.int64),
        positive_depth_fractions=np.asarray(positive_fractions, dtype=np.float64),
        maximum_parallax_degrees=np.asarray(parallaxes, dtype=np.float64),
        condition_numbers=np.asarray(conditions, dtype=np.float64),
        mean_reprojection_errors=np.asarray(means, dtype=np.float64),
        median_reprojection_errors=np.asarray(medians, dtype=np.float64),
        maximum_reprojection_errors=np.asarray(maxima, dtype=np.float64),
    )


def clean_triangulated_graph(
    triangulation: TriangulationResult,
    tracks: list[np.ndarray],
    valid_camera_ids: np.ndarray,
    *,
    minimum_parallax_degrees: float = 1.0,
    maximum_condition_number: float = 1e6,
    maximum_median_reprojection_error: float = 20.0,
    maximum_reprojection_error: float = 100.0,
    minimum_point_views: int = 2,
    minimum_camera_points: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], dict[str, int | float]]:
    """Filter triangulations and iteratively enforce camera/point degrees."""
    valid_cameras = set(map(int, valid_camera_ids))
    keep = (
        (triangulation.positive_depth_fractions == 1.0)
        & (triangulation.maximum_parallax_degrees >= minimum_parallax_degrees)
        & (triangulation.condition_numbers <= maximum_condition_number)
        & (
            triangulation.median_reprojection_errors
            <= maximum_median_reprojection_error
        )
        & (triangulation.maximum_reprojection_errors <= maximum_reprojection_error)
    )
    candidate_indices = np.flatnonzero(keep)
    active_cameras = valid_cameras.copy()
    observations: list[np.ndarray] = []
    while True:
        retained_indices = []
        observations = []
        camera_degrees: dict[int, int] = {}
        for result_index in candidate_indices:
            track_id = int(triangulation.track_ids[result_index])
            track_observations = np.asarray([
                [int(camera), int(feature)]
                for camera, feature in tracks[track_id]
                if int(camera) in active_cameras
            ], dtype=np.int64)
            if len(track_observations) < minimum_point_views:
                continue
            retained_indices.append(result_index)
            observations.append(track_observations)
            for camera in np.unique(track_observations[:, 0]):
                camera_degrees[int(camera)] = camera_degrees.get(int(camera), 0) + 1
        retained_cameras = {
            camera for camera, degree in camera_degrees.items()
            if degree >= minimum_camera_points
        }
        if retained_cameras == active_cameras:
            candidate_indices = np.asarray(retained_indices, dtype=np.int64)
            break
        active_cameras = retained_cameras
        candidate_indices = np.asarray(retained_indices, dtype=np.int64)
    retained_track_ids = triangulation.track_ids[candidate_indices]
    diagnostics: dict[str, int | float] = {
        "minimumParallaxDegrees": minimum_parallax_degrees,
        "maximumConditionNumber": maximum_condition_number,
        "maximumMedianReprojectionError": maximum_median_reprojection_error,
        "maximumReprojectionError": maximum_reprojection_error,
        "minimumPointViews": minimum_point_views,
        "minimumCameraPoints": minimum_camera_points,
        "retainedCameras": len(active_cameras),
        "retainedPoints": len(retained_track_ids),
        "retainedObservations": sum(map(len, observations)),
    }
    return (
        np.asarray(sorted(active_cameras), dtype=np.int64),
        candidate_indices,
        observations,
        diagnostics,
    )


def write_bal_problem(
    path: str | Path,
    source_camera_ids: np.ndarray,
    rotations: np.ndarray,
    camera_positions: np.ndarray,
    retained_camera_ids: np.ndarray,
    triangulation: TriangulationResult,
    retained_result_indices: np.ndarray,
    observations: list[np.ndarray],
    coordinates: dict[int, tuple[CameraFeatures, np.ndarray]],
) -> None:
    """Write a compact standard BAL problem from cleaned SfM_Init geometry."""
    camera_source_lookup = {
        int(camera): index for index, camera in enumerate(source_camera_ids)
    }
    compact_camera_lookup = {
        int(camera): index for index, camera in enumerate(retained_camera_ids)
    }
    cameras = np.zeros((len(retained_camera_ids), 9), dtype=np.float64)
    for compact_index, camera in enumerate(retained_camera_ids):
        source_index = camera_source_lookup[int(camera)]
        rotation = rotations[source_index]
        cameras[compact_index, :3] = Rotation.from_matrix(rotation).as_rotvec()
        cameras[compact_index, 3:6] = -rotation @ camera_positions[source_index]
        cameras[compact_index, 6] = coordinates[int(camera)][0].focal
    points = triangulation.points[retained_result_indices]
    observation_count = sum(map(len, observations))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write(f"{len(cameras)} {len(points)} {observation_count}\n")
        for point_index, point_observations in enumerate(observations):
            for camera_value, feature_value in point_observations:
                camera = int(camera_value)
                feature = int(feature_value)
                metadata, features = coordinates[camera]
                x, y = sift_to_bundler(
                    features[feature], (metadata.width, metadata.height)
                )
                stream.write(
                    f"{compact_camera_lookup[camera]} {point_index} "
                    f"{x:.17g} {y:.17g}\n"
                )
        for value in cameras.reshape(-1):
            stream.write(f"{value:.17g}\n")
        for value in points.reshape(-1):
            stream.write(f"{value:.17g}\n")


def read_component_ids(path: str | Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("cc.txt must contain at least one camera index")
    if np.any(values < 0) or np.unique(values).size != values.size:
        raise ValueError("cc.txt contains negative or duplicate camera indices")
    return values


def iter_coordinate_cameras(path: str | Path) -> Iterator[CameraFeatures]:
    for camera, _ in iter_coordinate_blocks(path):
        yield camera


def iter_coordinate_blocks(
    path: str | Path,
) -> Iterator[tuple[CameraFeatures, np.ndarray]]:
    with Path(path).open(encoding="utf-8") as stream:
        while header := stream.readline():
            match = _COORDS_HEADER.fullmatch(header.rstrip("\n"))
            if match is None:
                raise ValueError(f"invalid coords header: {header.rstrip()!r}")
            camera = CameraFeatures(
                index=int(match["index"]),
                name=match["name"],
                key_count=int(match["keys"]),
                width=2.0 * float(match["px"]),
                height=2.0 * float(match["py"]),
                focal=float(match["focal"]),
            )
            coordinates = np.empty((camera.key_count, 2), dtype=np.float64)
            for expected_feature in range(camera.key_count):
                line = stream.readline()
                if not line:
                    raise ValueError(
                        f"coords camera {camera.index} ends before feature "
                        f"{expected_feature}"
                    )
                fields = line.split()
                if len(fields) != 8 or int(fields[0]) != expected_feature:
                    raise ValueError(
                        f"invalid feature {expected_feature} for camera {camera.index}"
                    )
                coordinates[expected_feature] = float(fields[1]), float(fields[2])
            yield camera, coordinates


def iter_tracks(path: str | Path) -> Iterator[np.ndarray]:
    with Path(path).open(encoding="utf-8") as stream:
        first_line = stream.readline()
        if not first_line:
            raise ValueError("tracks.txt is empty")
        declared_tracks = int(first_line)
        for track_index in range(declared_tracks):
            line = stream.readline()
            if not line:
                raise ValueError(f"tracks.txt ends before track {track_index}")
            fields = np.fromstring(line, sep=" ", dtype=np.int64)
            if fields.size == 0:
                raise ValueError(f"track {track_index} is empty")
            degree = int(fields[0])
            if degree < 0 or fields.size != 1 + 2 * degree:
                raise ValueError(f"track {track_index} has an invalid degree")
            yield fields[1:].reshape((-1, 2))
        if stream.readline():
            raise ValueError("tracks.txt contains rows beyond its declared count")


def read_tracks(path: str | Path) -> list[np.ndarray]:
    return list(iter_tracks(path))


def select_track_cover(
    tracks: list[np.ndarray], coverage: int = 6
) -> np.ndarray:
    """Reproduce SfM_Init's greedy track cover with deterministic tie handling."""
    if coverage <= 0 or not tracks:
        raise ValueError("coverage must be positive and tracks nonempty")
    maximum_camera = max(int(np.max(track[:, 0])) for track in tracks)
    remaining = np.full(maximum_camera + 1, coverage, dtype=np.int64)
    improvements = np.asarray([len(track) for track in tracks], dtype=np.int64)
    camera_tracks: list[list[int]] = [[] for _ in range(maximum_camera + 1)]
    for track_index, track in enumerate(tracks):
        for camera in track[:, 0]:
            camera_tracks[int(camera)].append(track_index)
    heap = [(-int(value), index) for index, value in enumerate(improvements)]
    heapq.heapify(heap)
    selected: list[int] = []
    while heap:
        negative_value, track_index = heapq.heappop(heap)
        if -negative_value != improvements[track_index]:
            continue
        if improvements[track_index] <= 0:
            break
        selected.append(track_index)
        improvements[track_index] = 0
        for camera_value in tracks[track_index][:, 0]:
            camera = int(camera_value)
            if remaining[camera] == 1:
                for affected_track in camera_tracks[camera]:
                    if improvements[affected_track] > 0:
                        improvements[affected_track] -= 1
                        heapq.heappush(
                            heap,
                            (-int(improvements[affected_track]), affected_track),
                        )
            if remaining[camera] > 0:
                remaining[camera] -= 1
    return np.asarray(selected, dtype=np.int64)


def build_translation_graph(
    epipolar_geometries: np.ndarray,
    camera_ids: np.ndarray,
    rotations: np.ndarray,
    tracks: list[np.ndarray],
    coordinates: dict[int, tuple[CameraFeatures, np.ndarray]],
    *,
    track_coverage: int = 6,
    minimum_track_views: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Build SfM_Init camera-camera and selected camera-point directions."""
    if rotations.shape != (len(camera_ids), 3, 3):
        raise ValueError("rotations do not match camera_ids")
    rotation_lookup = {
        int(camera): rotations[index] for index, camera in enumerate(camera_ids)
    }
    camera_edges: list[tuple[int, int]] = []
    camera_poses: list[np.ndarray] = []
    for row in np.asarray(epipolar_geometries):
        source, destination = int(row[0]), int(row[1])
        if source in rotation_lookup and destination in rotation_lookup:
            direction = rotation_lookup[source].T @ row[11:14]
            direction /= np.linalg.norm(direction)
            camera_edges.append((source, destination))
            camera_poses.append(direction)
    if not camera_edges:
        raise ValueError("translation graph contains no camera-camera edges")
    number_of_images = max(max(edge) for edge in camera_edges) + 1
    selected_tracks = select_track_cover(tracks, coverage=track_coverage)
    point_edges: list[tuple[int, int]] = []
    point_poses: list[np.ndarray] = []
    retained_tracks: list[int] = []
    for track_index in selected_tracks:
        candidate_edges: list[tuple[int, int]] = []
        candidate_poses: list[np.ndarray] = []
        for camera_value, feature_value in tracks[int(track_index)]:
            camera = int(camera_value)
            feature = int(feature_value)
            if camera not in rotation_lookup or camera not in coordinates:
                continue
            metadata, features = coordinates[camera]
            if metadata.focal <= 0 or not 0 <= feature < len(features):
                continue
            x, y = sift_to_bundler(
                features[feature], (metadata.width, metadata.height)
            )
            local_direction = np.array((x, y, -metadata.focal))
            local_direction /= np.linalg.norm(local_direction)
            candidate_edges.append((camera, int(track_index) + number_of_images))
            candidate_poses.append(rotation_lookup[camera].T @ local_direction)
        if len(candidate_edges) >= minimum_track_views:
            point_edges.extend(candidate_edges)
            point_poses.extend(candidate_poses)
            retained_tracks.append(int(track_index))
    edges = np.asarray(camera_edges + point_edges, dtype=np.int64)
    poses = np.asarray(camera_poses + point_poses, dtype=np.float64)
    poses /= np.linalg.norm(poses, axis=1)[:, np.newaxis]
    camera_point_weight = 0.5 * len(camera_edges) / len(point_edges)
    weights = np.concatenate((
        np.ones(len(camera_edges), dtype=np.float64),
        np.full(len(point_edges), camera_point_weight, dtype=np.float64),
    ))
    diagnostics = {
        "cameraEdges": len(camera_edges),
        "selectedTracks": len(selected_tracks),
        "retainedTracks": len(retained_tracks),
        "cameraPointEdges": len(point_edges),
        "totalEdges": len(edges),
        "numberOfImages": number_of_images,
    }
    return (
        edges,
        poses,
        weights,
        np.asarray(retained_tracks, dtype=np.int64),
        diagnostics,
    )


def read_epipolar_geometries(path: str | Path) -> np.ndarray:
    geometries = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if geometries.shape[1] != 14:
        raise ValueError("EGs.txt rows must contain 14 fields")
    camera_indices = geometries[:, :2]
    if np.any(camera_indices < 0) or np.any(camera_indices != np.floor(camera_indices)):
        raise ValueError("EGs.txt contains invalid camera indices")
    return geometries


def rotation_residual_vectors(
    rotations: np.ndarray,
    edges: np.ndarray,
    relative_rotations: np.ndarray,
) -> np.ndarray:
    """Return log-map residuals for R_j R_i^T = relative_rotation."""
    errors = (
        np.swapaxes(rotations[edges[:, 1]], 1, 2)
        @ relative_rotations
        @ rotations[edges[:, 0]]
    )
    return Rotation.from_matrix(errors).as_rotvec()


def _spanning_tree_rotations(
    node_count: int,
    edges: np.ndarray,
    relative_rotations: np.ndarray,
) -> np.ndarray:
    rotations = np.repeat(np.eye(3)[np.newaxis, :, :], node_count, axis=0)
    known = np.zeros(node_count, dtype=bool)
    known[0] = True
    while not np.all(known):
        changed = False
        for edge, (source, destination) in enumerate(edges):
            if known[source] and not known[destination]:
                rotations[destination] = relative_rotations[edge] @ rotations[source]
                known[destination] = True
                changed = True
            elif not known[source] and known[destination]:
                rotations[source] = relative_rotations[edge].T @ rotations[destination]
                known[source] = True
                changed = True
        if not changed:
            raise ValueError("relative-rotation graph is disconnected")
    return rotations


def _rotation_incidence(edges: np.ndarray, node_count: int) -> csr_matrix:
    edge_count = len(edges)
    rows = np.repeat(np.arange(edge_count), 2)
    columns = edges.reshape(-1)
    values = np.tile((-1.0, 1.0), edge_count)
    incidence = coo_matrix(
        (values, (rows, columns)), shape=(edge_count, node_count)
    ).tocsr()
    return incidence[:, 1:]


def average_global_rotations(
    epipolar_geometries: np.ndarray,
    component_ids: np.ndarray,
    *,
    l1_iterations: int = 10,
    robust_iterations: int = 100,
    sigma_degrees: float = 5.0,
    tolerance: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Estimate component rotations with sparse L1-IRLS and robust refinement."""
    geometries = np.asarray(epipolar_geometries, dtype=np.float64)
    component_ids = np.asarray(component_ids, dtype=np.int64)
    if geometries.ndim != 2 or geometries.shape[1] != 14:
        raise ValueError("epipolar_geometries must have shape (N, 14)")
    if l1_iterations < 0 or robust_iterations < 0 or sigma_degrees <= 0:
        raise ValueError("rotation solver options are invalid")
    lookup = {int(camera): index for index, camera in enumerate(component_ids)}
    selected = np.array([
        int(row[0]) in lookup and int(row[1]) in lookup for row in geometries
    ])
    geometries = geometries[selected]
    edges = np.array([
        [lookup[int(row[0])], lookup[int(row[1])]] for row in geometries
    ], dtype=np.int64)
    if len(edges) == 0:
        raise ValueError("component contains no epipolar geometries")
    source_relative = geometries[:, 2:11].reshape((-1, 3, 3))
    relative_rotations = Rotation.from_matrix(
        np.swapaxes(source_relative, 1, 2)
    ).as_matrix()
    node_count = len(component_ids)
    rotations = _spanning_tree_rotations(node_count, edges, relative_rotations)
    incidence = _rotation_incidence(edges, node_count)
    initial_residuals = rotation_residual_vectors(
        rotations, edges, relative_rotations
    )

    for _ in range(l1_iterations):
        residuals = rotation_residual_vectors(rotations, edges, relative_rotations)
        weights = 1.0 / np.maximum(np.linalg.norm(residuals, axis=1), 1e-4)
        weighted_incidence = diags(weights) @ incidence
        corrections = np.zeros((node_count, 3), dtype=np.float64)
        for dimension in range(3):
            corrections[1:, dimension] = lsqr(
                weighted_incidence,
                weights * residuals[:, dimension],
                atol=1e-10,
                btol=1e-10,
            )[0]
        rotations = rotations @ Rotation.from_rotvec(corrections).as_matrix()
        if np.max(np.linalg.norm(corrections, axis=1)) < tolerance:
            break

    sigma = np.deg2rad(sigma_degrees)
    completed_robust_iterations = 0
    for iteration in range(robust_iterations):
        residuals = rotation_residual_vectors(rotations, edges, relative_rotations)
        weights = sigma / (np.sum(residuals * residuals, axis=1) + sigma * sigma)
        weighted_incidence = diags(weights) @ incidence
        corrections = np.zeros((node_count, 3), dtype=np.float64)
        for dimension in range(3):
            corrections[1:, dimension] = lsqr(
                weighted_incidence,
                weights * residuals[:, dimension],
                atol=1e-10,
                btol=1e-10,
            )[0]
        rotations = rotations @ Rotation.from_rotvec(corrections).as_matrix()
        completed_robust_iterations = iteration + 1
        if np.mean(np.linalg.norm(corrections, axis=1)) < tolerance:
            break

    rotations = Rotation.from_matrix(rotations).as_matrix()

    final_residuals = rotation_residual_vectors(
        rotations, edges, relative_rotations
    )
    diagnostics: dict[str, float | int] = {
        "edges": len(edges),
        "l1Iterations": l1_iterations,
        "robustIterations": completed_robust_iterations,
        "initialMedianResidualDegrees": float(np.degrees(np.median(
            np.linalg.norm(initial_residuals, axis=1)
        ))),
        "finalMedianResidualDegrees": float(np.degrees(np.median(
            np.linalg.norm(final_residuals, axis=1)
        ))),
        "finalP95ResidualDegrees": float(np.degrees(np.percentile(
            np.linalg.norm(final_residuals, axis=1), 95
        ))),
    }
    return component_ids.copy(), rotations, diagnostics


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_dataset(dataset_directory: str | Path) -> DatasetInventory:
    directory = Path(dataset_directory)
    paths = {
        name: directory / name
        for name in ("cc.txt", "coords.txt", "tracks.txt", "EGs.txt")
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    component_ids = read_component_ids(paths["cc.txt"])
    component = set(map(int, component_ids))
    coordinate_cameras = list(iter_coordinate_cameras(paths["coords.txt"]))
    feature_counts: dict[int, int] = {}
    for camera in coordinate_cameras:
        if camera.index in feature_counts:
            raise ValueError(f"duplicate coords camera {camera.index}")
        feature_counts[camera.index] = camera.key_count

    geometries = read_epipolar_geometries(paths["EGs.txt"])
    geometry_cameras = geometries[:, :2].astype(np.int64)
    component_geometries = int(np.sum(
        np.isin(geometry_cameras[:, 0], component_ids)
        & np.isin(geometry_cameras[:, 1], component_ids)
    ))

    declared_tracks = 0
    track_observations = 0
    component_track_observations = 0
    tracks_with_two_component_views = 0
    tracks_with_three_component_views = 0
    invalid_track_references = 0
    for track in iter_tracks(paths["tracks.txt"]):
        declared_tracks += 1
        track_observations += len(track)
        component_views = 0
        for camera_index, feature_index in track:
            camera = int(camera_index)
            feature = int(feature_index)
            if camera not in feature_counts or not 0 <= feature < feature_counts[camera]:
                invalid_track_references += 1
            if camera in component:
                component_views += 1
        component_track_observations += component_views
        tracks_with_two_component_views += component_views >= 2
        tracks_with_three_component_views += component_views >= 3

    return DatasetInventory(
        dataset=directory.name,
        component_cameras=len(component_ids),
        coordinate_cameras=len(coordinate_cameras),
        coordinate_features=sum(feature_counts.values()),
        epipolar_geometries=len(geometries),
        component_epipolar_geometries=component_geometries,
        declared_tracks=declared_tracks,
        track_observations=track_observations,
        component_track_observations=component_track_observations,
        tracks_with_two_component_views=tracks_with_two_component_views,
        tracks_with_three_component_views=tracks_with_three_component_views,
        invalid_track_references=invalid_track_references,
        source_sha256={name: sha256_file(path) for name, path in paths.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    inventory = inventory_dataset(arguments.dataset_directory)
    text = json.dumps(asdict(inventory), indent=2, sort_keys=True)
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
