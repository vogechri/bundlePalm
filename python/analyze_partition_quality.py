"""Evaluate BA partitions through their induced local observation problems."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parent
SERVER_TEST = WORKSPACE / "serverTest"
sys.path.insert(0, str(SERVER_TEST))

from bal_evaluator import read_bal_problem  # noqa: E402


def parse_arguments():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("dataset")
	parser.add_argument("--clusters", type=int, nargs="+", default=[10, 20, 30])
	parser.add_argument(
		"--partitioner", choices=("ours", "daba_labels"), default="ours")
	parser.add_argument(
		"--labels",
		help="NPZ with camera_owner and point_owner arrays for daba_labels",
	)
	parser.add_argument("--residual-balance-slack", type=float, default=0.01)
	parser.add_argument("--minimum-camera-landmarks", type=int, default=20)
	parser.add_argument("--max-refinement-passes", type=int, default=3)
	parser.add_argument("--output")
	return parser.parse_args()


def load_current_partitioner():
	previous_directory = Path.cwd()
	try:
		os.chdir(SERVER_TEST)
		from clustering import cluster_by_landmark_scalable_stable
	finally:
		os.chdir(previous_directory)
	return cluster_by_landmark_scalable_stable


def current_memberships(
	camera_indices,
	point_indices,
	observations,
	camera_count,
	point_count,
	cluster_count,
	residual_balance_slack,
	minimum_camera_landmarks,
	max_refinement_passes,
):
	partition = load_current_partitioner()
	started = time.perf_counter()
	camera_clusters, point_clusters, _, returned_count = partition(
		camera_indices,
		observations,
		point_indices,
		cluster_count,
		camera_count,
		point_count,
		residual_balance_slack,
		minimum_camera_landmarks,
		max_refinement_passes,
	)
	if returned_count != cluster_count:
		raise RuntimeError("partitioner changed the requested cluster count")
	memberships = []
	point_owner = np.full(point_count, -1, dtype=np.int64)
	for cluster, local_points in enumerate(point_clusters):
		for point in np.unique(local_points):
			if point_owner[point] not in (-1, cluster):
				raise RuntimeError("landmark has multiple owners")
			point_owner[point] = cluster
		memberships.append(np.full(local_points.size, cluster, dtype=np.int64))
	if np.any(point_owner < 0):
		raise RuntimeError("partition left landmarks unassigned")
	observation_memberships = [[int(point_owner[point])] for point in point_indices]
	return observation_memberships, None, point_owner, time.perf_counter() - started


def daba_memberships(camera_indices, point_indices, labels_path, cluster_count):
	if labels_path is None:
		raise ValueError("--labels is required for daba_labels")
	labels_path = Path(labels_path)
	if labels_path.suffix == ".json":
		labels = json.loads(labels_path.read_text(encoding="utf-8"))
	else:
		labels = np.load(labels_path)
	camera_owner = np.asarray(labels["camera_owner"], dtype=np.int64)
	point_owner = np.asarray(labels["point_owner"], dtype=np.int64)
	if camera_owner.size <= int(np.max(camera_indices)):
		raise ValueError("camera_owner has the wrong length")
	if point_owner.size <= int(np.max(point_indices)):
		raise ValueError("point_owner has the wrong length")
	if np.any(camera_owner < 0) or np.any(camera_owner >= cluster_count):
		raise ValueError("camera_owner contains an invalid cluster")
	if np.any(point_owner < 0) or np.any(point_owner >= cluster_count):
		raise ValueError("point_owner contains an invalid cluster")
	observation_memberships = [
		sorted({int(camera_owner[camera]), int(point_owner[point])})
		for camera, point in zip(camera_indices, point_indices)
	]
	return (
		observation_memberships,
		camera_owner,
		point_owner,
		float(labels.get("partitionSeconds", 0.0)),
	)


def evaluate_memberships(
	camera_indices,
	point_indices,
	observation_memberships,
	camera_count,
	point_count,
	cluster_count,
	camera_owner,
	point_owner,
	partition_seconds,
	partitioner,
):
	local_observations = [[] for _ in range(cluster_count)]
	for observation, memberships in enumerate(observation_memberships):
		for cluster in memberships:
			local_observations[cluster].append(observation)

	observations_per_cluster = np.array(
		[len(indices) for indices in local_observations], dtype=np.int64)
	cameras_per_cluster = np.zeros(cluster_count, dtype=np.int64)
	points_per_cluster = np.zeros(cluster_count, dtype=np.int64)
	weak_counts = {threshold: 0 for threshold in (5, 10, 20)}
	camera_cluster_incidences = 0
	degree_histogram = {}
	for cluster, indices in enumerate(local_observations):
		if not indices:
			continue
		indices = np.asarray(indices, dtype=np.int64)
		local_cameras = camera_indices[indices]
		local_points = point_indices[indices]
		cameras_per_cluster[cluster] = np.unique(local_cameras).size
		points_per_cluster[cluster] = np.unique(local_points).size
		order = np.lexsort((local_points, local_cameras))
		pairs = np.column_stack((local_cameras[order], local_points[order]))
		unique_pairs = np.unique(pairs, axis=0)
		cameras, degrees = np.unique(unique_pairs[:, 0], return_counts=True)
		camera_cluster_incidences += cameras.size
		for degree in degrees:
			degree_histogram[str(int(degree))] = (
				degree_histogram.get(str(int(degree)), 0) + 1)
		for threshold in weak_counts:
			weak_counts[threshold] += int(np.sum(degrees < threshold))

	total_copies = int(np.sum(observations_per_cluster))
	cross_owner = sum(len(memberships) > 1 for memberships in observation_memberships)
	mean_observations = float(np.mean(observations_per_cluster))
	assignment_bytes = np.asarray(
		[cluster for memberships in observation_memberships for cluster in memberships],
		dtype=np.int32,
	).tobytes()
	result = {
		"partitioner": partitioner,
		"clusters": cluster_count,
		"partitionSeconds": partition_seconds,
		"observations": int(camera_indices.size),
		"localObservationCopies": total_copies,
		"additionalObservationCopies": total_copies - int(camera_indices.size),
		"crossOwnerObservations": int(cross_owner),
		"observationsPerCluster": observations_per_cluster.tolist(),
		"minimumObservations": int(np.min(observations_per_cluster)),
		"maximumObservations": int(np.max(observations_per_cluster)),
		"observationImbalanceRatio": (
			float(np.max(observations_per_cluster) / mean_observations)),
		"observationCoefficientOfVariation": (
			float(np.std(observations_per_cluster) / mean_observations)),
		"camerasPerCluster": cameras_per_cluster.tolist(),
		"maximumCameras": int(np.max(cameras_per_cluster)),
		"cameraClusterIncidences": int(camera_cluster_incidences),
		"additionalCameraCopies": int(camera_cluster_incidences - camera_count),
		"pointsPerCluster": points_per_cluster.tolist(),
		"maximumPoints": int(np.max(points_per_cluster)),
		"weakCameraClusterIncidences": {
			f"degreeLessThan{threshold}": count
			for threshold, count in weak_counts.items()
		},
		"cameraClusterLandmarkDegreeHistogram": degree_histogram,
		"partitionHash": hashlib.sha256(assignment_bytes).hexdigest(),
	}
	if camera_owner is not None:
		result["ownedCamerasPerCluster"] = np.bincount(
			camera_owner, minlength=cluster_count).tolist()
	if point_owner is not None:
		result["ownedPointsPerCluster"] = np.bincount(
			point_owner, minlength=cluster_count).tolist()
	return result


def main():
	arguments = parse_arguments()
	if any(cluster_count <= 0 for cluster_count in arguments.clusters):
		raise ValueError("cluster counts must be positive")
	cameras, points, camera_indices, point_indices, observations = (
		read_bal_problem(arguments.dataset))
	results = []
	for cluster_count in arguments.clusters:
		if arguments.partitioner == "ours":
			membership_data = current_memberships(
				camera_indices,
				point_indices,
				observations,
				cameras.shape[0],
				points.shape[0],
				cluster_count,
				arguments.residual_balance_slack,
				arguments.minimum_camera_landmarks,
				arguments.max_refinement_passes,
			)
		else:
			membership_data = daba_memberships(
				camera_indices, point_indices, arguments.labels, cluster_count)
		observation_memberships, camera_owner, point_owner, partition_seconds = (
			membership_data)
		results.append(evaluate_memberships(
			camera_indices,
			point_indices,
			observation_memberships,
			cameras.shape[0],
			points.shape[0],
			cluster_count,
			camera_owner,
			point_owner,
			partition_seconds,
			partitioner=arguments.partitioner,
		))
	document = {
		"dataset": str(Path(arguments.dataset).resolve()),
		"results": results,
	}
	text = json.dumps(document, indent=2, sort_keys=True)
	if arguments.output:
		output = Path(arguments.output)
		output.parent.mkdir(parents=True, exist_ok=True)
		output.write_text(text + "\n", encoding="utf-8")
	print(text)


if __name__ == "__main__":
	main()
