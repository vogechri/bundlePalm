#!/usr/bin/env python3
"""Analyze camera-subspace and residual concentration between BA states."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_sfm_solver_state_gap import (
    camera_centers,
    similarity_transform,
    spectral_energy_fractions,
    transform_reconstruction,
)
from bal_evaluator import project_bal, read_bal_problem
from camera_tangent_diagnostics import left_se3_camera_minus


def load_state(path, problem):
    if path.suffix == ".npz":
        with np.load(path) as state:
            return (
                np.asarray(state["cameras"], dtype=np.float64),
                np.asarray(state["points"], dtype=np.float64),
            )
    camera_count, point_count = map(int, path.read_text().splitlines()[0].split())
    values = np.loadtxt(path, skiprows=1)
    split = 9 * camera_count
    cameras = values[:split].reshape(camera_count, 9)
    points = values[split:].reshape(point_count, 3)
    if cameras.shape != problem[0].shape or points.shape != problem[1].shape:
        raise ValueError("text state shape does not match the BAL problem")
    return cameras, points


def concentration(values, fractions=(0.01, 0.05, 0.10)):
    values = np.sort(np.asarray(values, dtype=np.float64).ravel())[::-1]
    total = float(np.sum(values))
    if total <= 0.0:
        return {str(fraction): 0.0 for fraction in fractions}
    return {
        str(fraction): float(np.sum(values[: max(1, int(np.ceil(fraction * len(values))))]) / total)
        for fraction in fractions
    }


def analyze(problem, reference_state, candidate_state):
    reference_cameras, reference_points = load_state(reference_state, problem)
    candidate_cameras, candidate_points = load_state(candidate_state, problem)
    scale, rotation, translation = similarity_transform(
        camera_centers(candidate_cameras), camera_centers(reference_cameras)
    )
    aligned_cameras, aligned_points = transform_reconstruction(
        candidate_cameras, candidate_points, scale, rotation, translation
    )
    tangent = left_se3_camera_minus(aligned_cameras, reference_cameras)
    tangent_energy = {
        "translation": np.sum(tangent[:, :3] ** 2, axis=1),
        "rotation": np.sum(tangent[:, 3:6] ** 2, axis=1),
        "intrinsics": np.sum(tangent[:, 6:9] ** 2, axis=1),
    }
    coordinate_scale = np.median(np.abs(reference_cameras), axis=0)
    positive_scale = coordinate_scale[coordinate_scale > 0.0]
    scale_floor = (
        float(np.median(positive_scale)) * 1e-6
        if positive_scale.size else 1e-6
    )
    normalized_tangent = tangent / np.maximum(coordinate_scale, scale_floor)
    normalized_tangent_energy = {
        "translation": np.sum(normalized_tangent[:, :3] ** 2, axis=1),
        "rotation": np.sum(normalized_tangent[:, 3:6] ** 2, axis=1),
        "intrinsics": np.sum(normalized_tangent[:, 6:9] ** 2, axis=1),
    }
    coordinate_names = (
        "translationX", "translationY", "translationZ",
        "rotationX", "rotationY", "rotationZ",
        "focal", "radialK1", "radialK2",
    )
    _, _, camera_indices, point_indices, observations = problem

    def residual_energy(cameras, points):
        residual = project_bal(
            cameras, points, camera_indices, point_indices
        ) - observations
        return np.sum(residual * residual, axis=1)

    candidate_residual = residual_energy(aligned_cameras, aligned_points)
    reference_residual = residual_energy(reference_cameras, reference_points)
    camera_candidate = np.bincount(
        camera_indices, weights=candidate_residual, minlength=len(reference_cameras)
    )
    camera_reference = np.bincount(
        camera_indices, weights=reference_residual, minlength=len(reference_cameras)
    )
    camera_observations = np.bincount(
        camera_indices, minlength=len(reference_cameras)
    )
    excess = camera_candidate - camera_reference
    mean_excess = excess / np.maximum(camera_observations, 1)
    worst_cameras = np.argsort(mean_excess)[::-1][:10]
    center_delta = camera_centers(aligned_cameras) - camera_centers(reference_cameras)
    total_tangent_energy = float(np.sum(tangent * tangent))
    total_normalized_tangent_energy = float(
        np.sum(normalized_tangent * normalized_tangent)
    )
    return {
        "referenceSSE": float(np.sum(reference_residual)),
        "candidateSSE": float(np.sum(candidate_residual)),
        "observationResidualConcentration": concentration(candidate_residual),
        "referenceObservationResidualConcentration": concentration(
            reference_residual
        ),
        "cameraResidualConcentration": concentration(camera_candidate),
        "referenceCameraResidualConcentration": concentration(camera_reference),
        "positiveCameraExcessConcentration": concentration(np.maximum(excess, 0.0)),
        "tangentEnergyFractions": {
            key: (
                float(np.sum(values) / total_tangent_energy)
                if total_tangent_energy > 0.0 else 0.0
            )
            for key, values in tangent_energy.items()
        },
        "tangentCameraConcentration": {
            key: concentration(values) for key, values in tangent_energy.items()
        },
        "normalizedTangentEnergyFractions": {
            key: (
                float(np.sum(values) / total_normalized_tangent_energy)
                if total_normalized_tangent_energy > 0.0 else 0.0
            )
            for key, values in normalized_tangent_energy.items()
        },
        "normalizedTangentCameraConcentration": {
            key: concentration(values)
            for key, values in normalized_tangent_energy.items()
        },
        "normalizedTangentCoordinateRms": {
            name: float(np.sqrt(np.mean(normalized_tangent[:, coordinate] ** 2)))
            for coordinate, name in enumerate(coordinate_names)
        },
        "coordinateScales": {
            name: float(max(coordinate_scale[coordinate], scale_floor))
            for coordinate, name in enumerate(coordinate_names)
        },
        "centerSpectrum": spectral_energy_fractions(problem, center_delta),
        "worstExcessCameras": [
            {
                "camera": int(camera),
                "candidateResidualSSE": float(camera_candidate[camera]),
                "referenceResidualSSE": float(camera_reference[camera]),
                "excessSSE": float(excess[camera]),
                "observations": int(camera_observations[camera]),
                "excessSSEPerObservation": float(mean_excess[camera]),
                "translationTangentNorm": float(np.linalg.norm(tangent[camera, :3])),
                "rotationTangentDegrees": float(
                    np.rad2deg(np.linalg.norm(tangent[camera, 3:6]))
                ),
                "intrinsicsTangentNorm": float(np.linalg.norm(tangent[camera, 6:9])),
            }
            for camera in worst_cameras
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("reference_state", type=Path)
    parser.add_argument("candidate_state", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = analyze(
        read_bal_problem(arguments.problem),
        arguments.reference_state,
        arguments.candidate_state,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
