#!/usr/bin/env python3
"""Partitioned PALM-style bundle adjustment with epoch acceleration.

Each block owns disjoint cameras and landmarks. Local sparse Gauss-Newton solves
use fixed values for variables owned by other blocks. Blocks are scheduled by
predicted gain, age, and runtime; every block is updated once per epoch. A
complete epoch defines a coherent fixed-point map that can be accelerated and
safeguarded by a global objective line search.

Example, from the python directory:
    serverTest/.venv/bin/python -u palm_ba.py \
        http://grail.cs.washington.edu/projects/bal/data/ladybug/ \
        problem-49-7776-pre.txt.bz2 30 6 --accelerator anderson
"""

from __future__ import annotations

import argparse
import bz2
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import block_diag, coo_matrix, eye, lil_matrix
import torch


@dataclass(frozen=True)
class BALProblem:
    cameras: np.ndarray
    points: np.ndarray
    camera_indices: np.ndarray
    point_indices: np.ndarray
    observations: np.ndarray


@dataclass
class State:
    cameras: np.ndarray
    points: np.ndarray

    def copy(self) -> "State":
        return State(self.cameras.copy(), self.points.copy())

    def vector(self) -> np.ndarray:
        return np.concatenate((self.cameras.ravel(), self.points.ravel()))

    @classmethod
    def from_vector(cls, vector: np.ndarray, n_cameras: int) -> "State":
        camera_size = 9 * n_cameras
        return cls(vector[:camera_size].reshape(-1, 9).copy(),
                   vector[camera_size:].reshape(-1, 3).copy())


@dataclass(frozen=True)
class GlobalJacobiPreconditioner:
    mode: str
    scale: np.ndarray

    def transform(self, vector: np.ndarray) -> np.ndarray:
        return self.scale * vector

    def inverse_transform(self, vector: np.ndarray) -> np.ndarray:
        return vector / self.scale


@dataclass
class Block:
    block_id: int
    cameras: np.ndarray
    points: np.ndarray
    observation_indices: np.ndarray
    local_camera_indices: np.ndarray
    local_point_indices: np.ndarray
    sparsity: object
    age: int = 0
    predicted_gain: float = 1.0
    runtime_seconds: float = 1.0
    damping: float = 1.0
    damping_reject_multiplier: float = 4.0
    local_acceptance: str = "palm"


@dataclass
class LocalResult:
    block_id: int
    cameras: np.ndarray
    points: np.ndarray
    local_cost_before: float
    local_cost_after: float
    runtime_seconds: float
    attempts: int
    inner_iterations: int
    model_quality: float
    step_weight: float = 1.0

    @property
    def gain(self) -> float:
        return self.local_cost_before - self.local_cost_after


@dataclass
class Accelerator:
    name: str
    memory: int
    regularization: float
    max_step_ratio: float
    momentum: float
    momentum_schedule: str
    previous_mapped: np.ndarray | None = None
    velocity: np.ndarray | None = None
    states: list[np.ndarray] = field(default_factory=list)
    residuals: list[np.ndarray] = field(default_factory=list)
    lbfgs_steps: list[np.ndarray] = field(default_factory=list)
    lbfgs_differences: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.previous_mapped = None
        self.velocity = None
        self.states.clear()
        self.residuals.clear()
        self.lbfgs_steps.clear()
        self.lbfgs_differences.clear()

    def propose(self, current: np.ndarray, mapped: np.ndarray,
                epoch: int) -> np.ndarray:
        residual = current - mapped
        if self.name == "none":
            candidate = mapped
        elif self.name == "heavy_ball":
            nominal_step = mapped - current
            beta = (max(0.0, (epoch - 1.0) / (epoch + 2.0))
                    if self.momentum_schedule == "palm" else self.momentum)
            self.velocity = (nominal_step if self.velocity is None else
                             nominal_step + beta * self.velocity)
            candidate = current + self.velocity
        elif self.name == "nesterov":
            beta = max(0.0, (epoch - 1.0) / (epoch + 2.0))
            candidate = mapped if self.previous_mapped is None else (
                mapped + beta * (mapped - self.previous_mapped))
        elif self.name == "anderson":
            candidate = self._anderson(current, mapped, residual)
        elif self.name in ("lbfgs", "bfgs"):
            candidate = current - self._lbfgs_inverse(residual)
        else:
            raise ValueError(f"unknown accelerator: {self.name}")
        self.previous_mapped = mapped.copy()
        self._limit_step(current, mapped, candidate)
        return candidate

    def observe(self, accepted: np.ndarray, mapped_from_accepted: np.ndarray,
                previous_accepted: np.ndarray,
                previous_residual: np.ndarray | None) -> np.ndarray:
        residual = accepted - mapped_from_accepted
        self.states.append(accepted.copy())
        self.residuals.append(residual.copy())
        self.states[:] = self.states[-(self.memory + 1):]
        self.residuals[:] = self.residuals[-(self.memory + 1):]
        if previous_residual is not None:
            step = accepted - previous_accepted
            difference = residual - previous_residual
            curvature = float(step @ difference)
            if curvature > 1e-12 * np.linalg.norm(step) * np.linalg.norm(difference):
                self.lbfgs_steps.append(step)
                self.lbfgs_differences.append(difference)
                self.lbfgs_steps[:] = self.lbfgs_steps[-self.memory:]
                self.lbfgs_differences[:] = self.lbfgs_differences[-self.memory:]
        return residual

    def _anderson(self, current: np.ndarray, mapped: np.ndarray,
                  residual: np.ndarray) -> np.ndarray:
        if not self.states:
            return mapped
        states = self.states + [current]
        residuals = self.residuals + [residual]
        delta_x = np.column_stack([
            states[index + 1] - states[index]
            for index in range(len(states) - 1)
        ])
        delta_r = np.column_stack([
            residuals[index + 1] - residuals[index]
            for index in range(len(residuals) - 1)
        ])
        system = delta_r.T @ delta_r
        system.flat[::system.shape[0] + 1] += self.regularization
        coefficients = np.linalg.solve(system, delta_r.T @ residual)
        return mapped - (delta_x - delta_r) @ coefficients

    def _lbfgs_inverse(self, residual: np.ndarray) -> np.ndarray:
        if not self.lbfgs_steps:
            return residual.copy()
        value = residual.copy()
        alphas = []
        for step, difference in reversed(list(zip(
                self.lbfgs_steps, self.lbfgs_differences))):
            rho = 1.0 / float(step @ difference)
            alpha = rho * float(step @ value)
            alphas.append(alpha)
            value -= alpha * difference
        last_step = self.lbfgs_steps[-1]
        last_difference = self.lbfgs_differences[-1]
        scale = float(last_step @ last_difference) / float(last_difference @ last_difference)
        value *= np.clip(scale, 1e-6, 1e6)
        for (step, difference), alpha in zip(
                zip(self.lbfgs_steps, self.lbfgs_differences), reversed(alphas)):
            rho = 1.0 / float(step @ difference)
            value += step * (alpha - rho * float(difference @ value))
        return value

    def _limit_step(self, current: np.ndarray, mapped: np.ndarray,
                    candidate: np.ndarray) -> None:
        nominal_norm = np.linalg.norm(mapped - current)
        candidate_step = candidate - current
        candidate_norm = np.linalg.norm(candidate_step)
        limit = self.max_step_ratio * max(nominal_norm, 1e-12)
        if candidate_norm > limit:
            candidate[:] = current + candidate_step * (limit / candidate_norm)


def read_bal(path: Path) -> BALProblem:
    with bz2.open(path, "rt") as handle:
        n_cameras, n_points, n_observations = map(int, handle.readline().split())
        camera_indices = np.empty(n_observations, dtype=np.int64)
        point_indices = np.empty(n_observations, dtype=np.int64)
        observations = np.empty((n_observations, 2), dtype=np.float64)
        for index in range(n_observations):
            camera, point, x_value, y_value = handle.readline().split()
            camera_indices[index] = int(camera)
            point_indices[index] = int(point)
            observations[index] = float(x_value), float(y_value)
        cameras = np.fromiter((float(handle.readline())
                               for _ in range(9 * n_cameras)),
                              dtype=np.float64).reshape(n_cameras, 9)
        points = np.fromiter((float(handle.readline())
                              for _ in range(3 * n_points)),
                             dtype=np.float64).reshape(n_points, 3)
    return BALProblem(cameras, points, camera_indices, point_indices,
                      observations)


def ensure_problem(base_url: str, file_name: str, directory: Path) -> Path:
    path = directory / file_name
    if not path.exists():
        print(f"Downloading {base_url + file_name}")
        temporary_path = path.with_suffix(path.suffix + ".part")
        try:
            urllib.request.urlretrieve(base_url + file_name, temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return path


def rotate(points: np.ndarray, rotation_vectors: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotation_vectors, axis=1)[:, None]
    directions = np.divide(rotation_vectors, theta, out=np.zeros_like(rotation_vectors),
                           where=theta != 0)
    dot = np.sum(points * directions, axis=1)[:, None]
    return (np.cos(theta) * points
            + np.sin(theta) * np.cross(directions, points)
            + dot * (1.0 - np.cos(theta)) * directions)


def project(points: np.ndarray, cameras: np.ndarray) -> np.ndarray:
    transformed = rotate(points, cameras[:, :3]) + cameras[:, 3:6]
    normalized = -transformed[:, :2] / transformed[:, 2, None]
    radius_squared = np.sum(normalized * normalized, axis=1)
    distortion = 1.0 + cameras[:, 7] * radius_squared + cameras[:, 8] * radius_squared**2
    return normalized * (cameras[:, 6] * distortion)[:, None]


def residuals(problem: BALProblem, state: State,
              observation_indices: np.ndarray | None = None) -> np.ndarray:
    indices = (np.arange(problem.observations.shape[0])
               if observation_indices is None else observation_indices)
    projected = project(state.points[problem.point_indices[indices]],
                        state.cameras[problem.camera_indices[indices]])
    return (projected - problem.observations[indices]).ravel()


def objective(problem: BALProblem, state: State) -> float:
    values = residuals(problem, state)
    return float(values @ values)


def _assign_partition_owners(problem: BALProblem, count: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 1 or count > problem.cameras.shape[0]:
        raise ValueError("partitions must be between 1 and the number of cameras")
    camera_load = np.bincount(problem.camera_indices,
                              minlength=problem.cameras.shape[0])
    block_load = np.zeros(count, dtype=np.int64)
    camera_owner = np.empty(problem.cameras.shape[0], dtype=np.int64)
    for camera in np.argsort(-camera_load, kind="stable"):
        owner = int(np.argmin(block_load))
        camera_owner[camera] = owner
        block_load[owner] += camera_load[camera]

    point_owner = np.empty(problem.points.shape[0], dtype=np.int64)
    point_load = np.zeros(count, dtype=np.int64)
    order = np.lexsort((camera_owner[problem.camera_indices], problem.point_indices))
    sorted_points = problem.point_indices[order]
    sorted_owners = camera_owner[problem.camera_indices[order]]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_points)) + 1]
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops):
        point = int(sorted_points[start])
        counts = np.bincount(sorted_owners[start:stop], minlength=count)
        best_count = counts.max()
        choices = np.flatnonzero(counts == best_count)
        owner = int(choices[np.argmin(point_load[choices])])
        point_owner[point] = owner
        point_load[owner] += stop - start

    return camera_owner, point_owner


def _refine_partition_overlap(problem: BALProblem, camera_owner: np.ndarray,
                              point_owner: np.ndarray, count: int,
                              max_passes: int,
                              balance_slack: float,
                              max_swap_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    camera_count = problem.cameras.shape[0]
    point_count = problem.points.shape[0]
    camera_load = np.bincount(problem.camera_indices, minlength=camera_count)
    point_degree = np.bincount(problem.point_indices, minlength=point_count)
    votes = np.zeros((point_count, count), dtype=np.int32)
    np.add.at(votes, (problem.point_indices,
                      camera_owner[problem.camera_indices]), 1)

    camera_points = []
    camera_point_counts = []
    for camera in range(camera_count):
        mask = problem.camera_indices == camera
        points, counts = np.unique(problem.point_indices[mask], return_counts=True)
        camera_points.append(points)
        camera_point_counts.append(counts)

    pair_strength = {}
    observation_order = np.argsort(problem.point_indices, kind="stable")
    sorted_points = problem.point_indices[observation_order]
    sorted_cameras = problem.camera_indices[observation_order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_points)) + 1]
    stops = np.r_[starts[1:], len(observation_order)]
    for start, stop in zip(starts, stops):
        cameras = np.unique(sorted_cameras[start:stop])
        for first_index in range(len(cameras)):
            for second_index in range(first_index + 1, len(cameras)):
                pair = (int(cameras[first_index]), int(cameras[second_index]))
                pair_strength[pair] = pair_strength.get(pair, 0) + 1
    candidate_pairs = sorted(
        pair_strength, key=lambda pair: (-pair_strength[pair], pair))
    if max_swap_candidates:
        candidate_pairs = candidate_pairs[:max_swap_candidates]

    camera_block_load = np.bincount(
        camera_owner, weights=camera_load, minlength=count).astype(np.int64)
    point_block_load = np.bincount(
        point_owner, weights=point_degree, minlength=count).astype(np.int64)
    matched_load = np.bincount(
        point_owner,
        weights=votes[np.arange(point_count), point_owner],
        minlength=count).astype(np.int64)
    baseline_loads = camera_block_load + point_block_load - matched_load
    load_limit = int(math.ceil(baseline_loads.max() * (1.0 + balance_slack)))

    for _ in range(max_passes):
        best = None
        for first_camera, second_camera in candidate_pairs:
            first_block = camera_owner[first_camera]
            second_block = camera_owner[second_camera]
            if first_block == second_block:
                continue
            affected_points = np.union1d(
                camera_points[first_camera], camera_points[second_camera])
            trial_votes = votes[affected_points].copy()
            first_rows = np.searchsorted(
                affected_points, camera_points[first_camera])
            second_rows = np.searchsorted(
                affected_points, camera_points[second_camera])
            trial_votes[first_rows, first_block] -= camera_point_counts[first_camera]
            trial_votes[first_rows, second_block] += camera_point_counts[first_camera]
            trial_votes[second_rows, second_block] -= camera_point_counts[second_camera]
            trial_votes[second_rows, first_block] += camera_point_counts[second_camera]

            old_point_owner = point_owner[affected_points]
            maxima = trial_votes.max(axis=1)
            keep_owner = (trial_votes[np.arange(len(affected_points)),
                                      old_point_owner] == maxima)
            new_point_owner = np.where(
                keep_owner, old_point_owner, np.argmax(trial_votes, axis=1))
            old_matched = votes[affected_points, old_point_owner]
            new_matched = trial_votes[
                np.arange(len(affected_points)), new_point_owner]
            overlap_gain = int(new_matched.sum() - old_matched.sum())
            if overlap_gain <= 0:
                continue

            trial_camera_load = camera_block_load.copy()
            trial_camera_load[first_block] += (
                camera_load[second_camera] - camera_load[first_camera])
            trial_camera_load[second_block] += (
                camera_load[first_camera] - camera_load[second_camera])
            trial_point_load = point_block_load.copy()
            trial_matched_load = matched_load.copy()
            np.add.at(trial_point_load, old_point_owner,
                      -point_degree[affected_points])
            np.add.at(trial_point_load, new_point_owner,
                      point_degree[affected_points])
            np.add.at(trial_matched_load, old_point_owner, -old_matched)
            np.add.at(trial_matched_load, new_point_owner, new_matched)
            trial_loads = (trial_camera_load + trial_point_load
                           - trial_matched_load)
            if trial_loads.max() > load_limit:
                continue
            key = (overlap_gain, -int(trial_loads.max()))
            if best is None or key > best[0]:
                best = (key, first_camera, second_camera, first_block,
                        second_block, affected_points, trial_votes,
                        new_point_owner, trial_camera_load,
                        trial_point_load, trial_matched_load)
        if best is None:
            break
        (_, first_camera, second_camera, first_block, second_block,
         affected_points, trial_votes, new_point_owner, camera_block_load,
         point_block_load, matched_load) = best
        camera_owner[first_camera] = second_block
        camera_owner[second_camera] = first_block
        votes[affected_points] = trial_votes
        point_owner[affected_points] = new_point_owner
    return camera_owner, point_owner


def build_partitions(problem: BALProblem, count: int,
                     partitioner: str = "load", refinement_passes: int = 20,
                     balance_slack: float = 0.0,
                     max_swap_candidates: int = 256) -> list[Block]:
    camera_owner, point_owner = _assign_partition_owners(problem, count)
    if partitioner == "overlap":
        camera_owner, point_owner = _refine_partition_overlap(
            problem, camera_owner, point_owner, count,
            refinement_passes, balance_slack, max_swap_candidates)
    elif partitioner != "load":
        raise ValueError(f"unknown partitioner: {partitioner}")

    blocks = []
    for block_id in range(count):
        owned_cameras = np.flatnonzero(camera_owner == block_id)
        owned_points = np.flatnonzero(point_owner == block_id)
        included = ((camera_owner[problem.camera_indices] == block_id)
                    | (point_owner[problem.point_indices] == block_id))
        observation_indices = np.flatnonzero(included)
        camera_map = np.full(problem.cameras.shape[0], -1, dtype=np.int64)
        point_map = np.full(problem.points.shape[0], -1, dtype=np.int64)
        camera_map[owned_cameras] = np.arange(len(owned_cameras))
        point_map[owned_points] = np.arange(len(owned_points))
        local_cameras = camera_map[problem.camera_indices[observation_indices]]
        local_points = point_map[problem.point_indices[observation_indices]]
        sparsity = local_sparsity(local_cameras, local_points,
                                  len(owned_cameras), len(owned_points))
        blocks.append(Block(block_id, owned_cameras, owned_points,
                            observation_indices, local_cameras, local_points,
                            sparsity.tocsr()))
    return blocks


def local_sparsity(camera_indices: np.ndarray, point_indices: np.ndarray,
                   n_cameras: int, n_points: int):
    matrix = lil_matrix((2 * len(camera_indices), 9 * n_cameras + 3 * n_points),
                        dtype=np.int8)
    rows = np.arange(len(camera_indices))
    camera_mask = camera_indices >= 0
    point_mask = point_indices >= 0
    for offset in range(9):
        columns = 9 * camera_indices[camera_mask] + offset
        matrix[2 * rows[camera_mask], columns] = 1
        matrix[2 * rows[camera_mask] + 1, columns] = 1
    point_offset = 9 * n_cameras
    for offset in range(3):
        columns = point_offset + 3 * point_indices[point_mask] + offset
        matrix[2 * rows[point_mask], columns] = 1
        matrix[2 * rows[point_mask] + 1, columns] = 1
    return matrix


def torch_rotate(points: torch.Tensor,
                 rotation_vectors: torch.Tensor) -> torch.Tensor:
    theta_squared = torch.sum(rotation_vectors * rotation_vectors, dim=1)
    mask = theta_squared > 0.0
    theta = torch.sqrt(torch.where(mask, theta_squared,
                                   torch.ones_like(theta_squared)))
    directions = rotation_vectors / theta[:, None]
    dot = torch.sum(points * directions, dim=1)[:, None]
    rotated = (torch.cos(theta)[:, None] * points
               + torch.sin(theta)[:, None] * torch.linalg.cross(
                   directions, points)
               + dot * (1.0 - torch.cos(theta))[:, None] * directions)
    first_order = points + torch.linalg.cross(rotation_vectors, points)
    return torch.where(mask[:, None], rotated, first_order)


def torch_project(points: torch.Tensor,
                  cameras: torch.Tensor) -> torch.Tensor:
    transformed = torch_rotate(points, cameras[:, :3]) + cameras[:, 3:6]
    normalized = -transformed[:, :2] / transformed[:, 2, None]
    radius_squared = torch.sum(normalized * normalized, dim=1)
    distortion = (1.0 + cameras[:, 7] * radius_squared
                  + cameras[:, 8] * radius_squared**2)
    return normalized * (cameras[:, 6] * distortion)[:, None]


def build_sparse_jacobian(dx: np.ndarray, dy: np.ndarray,
                          variable_indices: np.ndarray,
                          variable_size: int, variable_count: int):
    present = variable_indices >= 0
    observation_indices = np.flatnonzero(present)
    local_indices = variable_indices[present]
    columns = (variable_size * local_indices[:, None]
               + np.arange(variable_size)).ravel()
    x_rows = np.repeat(observation_indices, variable_size)
    y_rows = x_rows + len(variable_indices)
    rows = np.concatenate((x_rows, y_rows))
    data = np.concatenate((dx[present].ravel(), dy[present].ravel()))
    return coo_matrix(
        (data, (rows, np.concatenate((columns, columns)))),
        shape=(2 * len(variable_indices), variable_size * variable_count),
    ).tocsr()


def compute_derivative_matrices(cameras: np.ndarray, points: np.ndarray,
                                observations: np.ndarray,
                                camera_indices: np.ndarray,
                                point_indices: np.ndarray,
                                n_cameras: int, n_points: int):
    camera_dx, camera_dy, point_dx, point_dy, fx0 = (
        compute_observation_derivatives(cameras, points, observations))

    J_pose = build_sparse_jacobian(
        camera_dx, camera_dy, camera_indices, 9, n_cameras)
    J_land = build_sparse_jacobian(
        point_dx, point_dy, point_indices, 3, n_points)
    return J_pose, J_land, fx0


def compute_observation_derivatives(cameras: np.ndarray, points: np.ndarray,
                                    observations: np.ndarray):
    torch_cameras = torch.from_numpy(cameras.copy()).requires_grad_(True)
    torch_points = torch.from_numpy(points.copy()).requires_grad_(True)
    torch_observations = torch.from_numpy(observations)
    projected = torch_project(torch_points, torch_cameras)
    residual = projected - torch_observations

    residual[:, 0].sum().backward(retain_graph=True)
    camera_dx = torch_cameras.grad.detach().numpy().copy()
    point_dx = torch_points.grad.detach().numpy().copy()
    torch_cameras.grad.zero_()
    torch_points.grad.zero_()
    residual[:, 1].sum().backward()
    camera_dy = torch_cameras.grad.detach().numpy().copy()
    point_dy = torch_points.grad.detach().numpy().copy()

    fx0 = np.concatenate((residual[:, 0].detach().numpy(),
                          residual[:, 1].detach().numpy()))
    return camera_dx, camera_dy, point_dx, point_dy, fx0


def build_global_jacobi_preconditioner(
        problem: BALProblem, mode: str, batch_size: int,
        relative_floor: float) -> GlobalJacobiPreconditioner:
    camera_diagonal = np.zeros_like(problem.cameras)
    point_diagonal = np.zeros_like(problem.points)

    if mode != "none":
        for start in range(0, len(problem.observations), batch_size):
            stop = min(start + batch_size, len(problem.observations))
            camera_indices = problem.camera_indices[start:stop]
            point_indices = problem.point_indices[start:stop]
            camera_dx, camera_dy, point_dx, point_dy, _ = (
                compute_observation_derivatives(
                    problem.cameras[camera_indices],
                    problem.points[point_indices],
                    problem.observations[start:stop]))
            np.add.at(camera_diagonal, camera_indices,
                      camera_dx**2 + camera_dy**2)
            if mode == "full":
                np.add.at(point_diagonal, point_indices,
                          point_dx**2 + point_dy**2)

    def diagonal_scale(diagonal: np.ndarray) -> np.ndarray:
        positive = diagonal[np.isfinite(diagonal) & (diagonal > 0.0)]
        if positive.size == 0:
            return np.ones(diagonal.size)
        floor = max(relative_floor * float(np.median(positive)),
                    np.finfo(float).tiny)
        return np.sqrt(np.maximum(diagonal, floor)).ravel()

    camera_scale = (np.ones(problem.cameras.size) if mode == "none" else
                    diagonal_scale(camera_diagonal))
    point_scale = (diagonal_scale(point_diagonal) if mode == "full" else
                   np.ones(problem.points.size))
    return GlobalJacobiPreconditioner(
        mode, np.concatenate((camera_scale, point_scale)))


def block_inverse(matrix, block_size: int):
    blocks = []
    for start in range(0, matrix.shape[0], block_size):
        block = matrix[start:start + block_size,
                       start:start + block_size].toarray()
        blocks.append(coo_matrix(np.linalg.pinv(block)))
    return block_diag(blocks, format="csr")


def stop_criterion(delta: float, delta_i: float, iteration: int,
                   tolerance: float) -> bool:
    return ((iteration + 1) * delta_i / max(delta, 1e-30)
            < tolerance)


def solveByGDNesterov(U, S, bS: np.ndarray,
                      max_iterations: int = 200,
                      tolerance: float = 1e-2,
                      min_iterations: int = 10) -> tuple[np.ndarray, int]:
    Lip = 0.9
    lambda0 = (1.0 + np.sqrt(5.0)) / 2.0
    Uli = block_inverse(U, 9)
    ubs = -Uli @ bS
    xk = -ubs
    y0 = -ubs

    for iteration in range(max_iterations):
        lambda1 = (1.0 + np.sqrt(1.0 + 4.0 * lambda0**2)) / 2.0
        gamma = (1.0 - lambda0) / lambda1
        lambda0 = lambda1
        gradient = Uli @ (S @ xk - bS)
        yk = xk - gradient / Lip
        xk = (1.0 - gamma) * yk + gamma * y0
        y0 = yk
        if ((iteration + 1) >= min_iterations
            and stop_criterion(np.linalg.norm(xk),
                np.linalg.norm(gradient / Lip), iteration,
                tolerance)):
            return np.asarray(xk).ravel(), iteration + 1
    return np.asarray(xk).ravel(), max_iterations


def solve_block(problem: BALProblem, block: Block, snapshot: State,
                max_nfev: int, local_solver: str = "cpu",
                device: str = "cuda:0", inner_iterations: int = 200,
                inner_check_interval: int = 10,
                inner_tolerance: float = 1e-2,
                inner_min_iterations: int = 10,
                inner_solver: str = "nesterov",
                inner_jacobi: bool = False) -> LocalResult:
    start = time.monotonic()
    observation_indices = block.observation_indices
    camera_mask = block.local_camera_indices >= 0
    point_mask = block.local_point_indices >= 0
    if local_solver == "bae" and np.all(camera_mask):
        fixed_cameras = np.empty((0, 9), dtype=snapshot.cameras.dtype)
    else:
        fixed_cameras = snapshot.cameras[
            problem.camera_indices[observation_indices]].copy()
    if local_solver == "bae" and np.all(point_mask):
        fixed_points = np.empty((0, 3), dtype=snapshot.points.dtype)
    else:
        fixed_points = snapshot.points[
            problem.point_indices[observation_indices]].copy()
    local_cameras = snapshot.cameras[block.cameras].copy()
    local_points = snapshot.points[block.points].copy()

    def local_residual() -> np.ndarray:
        cameras = fixed_cameras.copy()
        points = fixed_points.copy()
        cameras[camera_mask] = local_cameras[block.local_camera_indices[camera_mask]]
        points[point_mask] = local_points[block.local_point_indices[point_mask]]
        return (project(points, cameras)
                - problem.observations[observation_indices]).ravel()

    before_values = local_residual() if local_solver == "cpu" else None
    local_cost_before = float(before_values @ before_values) if before_values is not None else None
    local_cost_after = local_cost_before
    successful_steps = 0
    attempts = 0
    total_inner_iterations = 0
    model_quality = -1.0
    while successful_steps < max(1, max_nfev - 1) and attempts < 8 * max_nfev:
        attempts += 1
        if local_solver == "cpu":
            cameras = fixed_cameras.copy()
            points = fixed_points.copy()
            cameras[camera_mask] = local_cameras[
                block.local_camera_indices[camera_mask]]
            points[point_mask] = local_points[
                block.local_point_indices[point_mask]]
        else:
            cameras = fixed_cameras
            points = fixed_points
        epsilon = 1e-4
        if local_solver == "bae":
            from bae_local_solver import solve_local_step
            gpu_step = solve_local_step(
                local_cameras, local_points, cameras, points,
                problem.observations[observation_indices],
                block.local_camera_indices, block.local_point_indices,
                block.damping, epsilon=epsilon, device=device,
                cache_key=(id(problem), block.block_id),
                max_iterations=inner_iterations,
                tolerance=inner_tolerance,
                min_iterations=inner_min_iterations,
                inner_solver=inner_solver,
                jacobi_preconditioner=inner_jacobi,
                check_interval=inner_check_interval)
            total_inner_iterations += gpu_step.iterations
            delta_p = gpu_step.delta_p
            delta_l = gpu_step.delta_l
            fx0 = gpu_step.residual
            if local_cost_before is None:
                local_cost_before = gpu_step.cost_before
                local_cost_after = local_cost_before
            quadratic_penalty = block.damping * (
                gpu_step.camera_penalty + gpu_step.point_penalty)
            predicted_reduction = gpu_step.predicted_reduction
        else:
            J_pose, J_land, fx0 = compute_derivative_matrices(
                cameras, points, problem.observations[observation_indices],
                block.local_camera_indices, block.local_point_indices,
                len(block.cameras), len(block.points))

            camera_metric = J_pose.T @ J_pose
            landmark_metric = J_land.T @ J_land
            U_metric = camera_metric + epsilon * eye(camera_metric.shape[0])
            V_metric = landmark_metric + epsilon * eye(landmark_metric.shape[0])
            U = camera_metric + block.damping * U_metric
            V = landmark_metric + block.damping * V_metric
            W = J_pose.T @ J_land
            Vli = block_inverse(V, 3)
            S = U - W @ Vli @ W.T
            bp = np.asarray(J_pose.T @ fx0).ravel()
            bl = np.asarray(J_land.T @ fx0).ravel()
            bS = bp - np.asarray(W @ (Vli @ bl)).ravel()
            delta_p, inner_count = solveByGDNesterov(
                U, S, bS, inner_iterations, inner_tolerance,
                inner_min_iterations)
            total_inner_iterations += inner_count
            delta_p = -delta_p
            delta_l = -np.asarray(Vli @ (W.T @ delta_p + bl)).ravel()
            quadratic_penalty = block.damping * (
                float(delta_p @ (U_metric @ delta_p))
                + float(delta_l @ (V_metric @ delta_l)))
            linearized_change = J_pose @ delta_p + J_land @ delta_l
            predicted_reduction = float(-(
                2.0 * (fx0 @ linearized_change)
                + linearized_change @ linearized_change))

        candidate_cameras = local_cameras + delta_p.reshape(-1, 9)
        candidate_points = local_points + delta_l.reshape(-1, 3)
        old_cameras, old_points = local_cameras, local_points
        local_cameras, local_points = candidate_cameras, candidate_points
        if local_solver == "bae":
            cost_before = gpu_step.cost_before
            cost_after = gpu_step.cost_after
        else:
            after_values = local_residual()
            cost_before = float(fx0 @ fx0)
            cost_after = float(after_values @ after_values)
        accepted_cost = (cost_after + quadratic_penalty
                         if block.local_acceptance == "palm"
                         else cost_after)
        actual_reduction = cost_before - cost_after
        model_quality = (actual_reduction / predicted_reduction
                 if predicted_reduction > 0.0 else -1.0)
        if np.isfinite(cost_after) and accepted_cost <= cost_before:
            successful_steps += 1
            local_cost_after = cost_after
            block.damping = max(1e-6, block.damping / 2.0)
        else:
            local_cameras, local_points = old_cameras, old_points
            block.damping *= block.damping_reject_multiplier

    if local_solver == "cpu":
        after_values = local_residual()
        local_cost_after = float(after_values @ after_values)
    return LocalResult(
        block.block_id,
        local_cameras,
        local_points,
        float(local_cost_before),
        float(local_cost_after),
        time.monotonic() - start,
        attempts,
        total_inner_iterations,
        model_quality,
    )


def choose_block(remaining: set[int], blocks: list[Block], age_weight: float,
                 runtime_weight: float) -> int:
    def score(block_id: int) -> float:
        block = blocks[block_id]
        numerator = max(block.predicted_gain, 0.0) + age_weight * block.age
        denominator = max(block.runtime_seconds, 1e-3) ** runtime_weight
        return numerator / denominator
    return max(remaining, key=score)


def apply_local_result(state: State, block: Block,
                       result: LocalResult) -> None:
    if result.gain < 0.0:
        return
    state.cameras[block.cameras] = result.cameras
    state.points[block.points] = result.points


def overrelax_local_result(problem: BALProblem, state: State, block: Block,
                           result: LocalResult, factor: float,
                           backtracks: int, safeguard: bool) -> None:
    if result.gain < 0.0 or factor <= 1.0:
        return
    camera_step = result.cameras - state.cameras[block.cameras]
    point_step = result.points - state.points[block.points]
    if not safeguard:
        result.cameras = state.cameras[block.cameras] + factor * camera_step
        result.points = state.points[block.points] + factor * point_step
        result.step_weight = factor
        return
    observation_indices = block.observation_indices
    camera_mask = block.local_camera_indices >= 0
    point_mask = block.local_point_indices >= 0
    for backtrack in range(backtracks + 1):
        weight = 1.0 + (factor - 1.0) * 0.5**backtrack
        trial_cameras = state.cameras[block.cameras] + weight * camera_step
        trial_points = state.points[block.points] + weight * point_step
        cameras = state.cameras[
            problem.camera_indices[observation_indices]].copy()
        points = state.points[
            problem.point_indices[observation_indices]].copy()
        cameras[camera_mask] = trial_cameras[
            block.local_camera_indices[camera_mask]]
        points[point_mask] = trial_points[
            block.local_point_indices[point_mask]]
        values = (project(points, cameras)
                  - problem.observations[observation_indices]).ravel()
        trial_cost = float(values @ values)
        if (np.isfinite(trial_cost)
                and trial_cost <= result.local_cost_after):
            result.cameras = trial_cameras
            result.points = trial_points
            result.local_cost_after = trial_cost
            result.step_weight = weight
            return


def run_epoch(problem: BALProblem, blocks: list[Block], anchor: State,
              max_nfev: int, age_weight: float,
              runtime_weight: float, execution: str,
              workers: int, local_solver: str = "cpu",
              device: str = "cuda:0", inner_iterations: int = 200,
              inner_check_interval: int = 10,
              inner_tolerance: float = 1e-2,
              inner_min_iterations: int = 10,
              inner_solver: str = "nesterov",
              inner_jacobi: bool = False,
              block_overrelaxation: float = 1.0,
              block_backtracks: int = 3,
              block_safeguard: bool = True) -> tuple[State, list[LocalResult]]:
    state = anchor.copy()
    if execution == "parallel":
        order = sorted(
            range(len(blocks)),
            key=lambda block_id: choose_block(
                {block_id}, blocks, age_weight, runtime_weight),
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(solve_block, problem, blocks[block_id],
                                anchor, max_nfev, local_solver, device,
                                inner_iterations, inner_check_interval,
                                inner_tolerance, inner_min_iterations,
                                inner_solver, inner_jacobi)
                for block_id in order
            ]
            results = [future.result() for future in futures]
        for result in results:
            block = blocks[result.block_id]
            apply_local_result(state, block, result)
            update_block_statistics(blocks, block, result)
            print(f"  block {block.block_id:3d}: gain={result.gain: .6e}, "
                f"time={result.runtime_seconds:.2f}s, attempts={result.attempts}, "
                f"inner={result.inner_iterations}, quality={result.model_quality:.3e}, "
                f"damping={block.damping:.3e}")
        return state, results

    remaining = set(range(len(blocks)))
    results = []
    while remaining:
        block_id = choose_block(remaining, blocks, age_weight, runtime_weight)
        block = blocks[block_id]
        result = solve_block(problem, block, state.copy(), max_nfev,
                             local_solver, device, inner_iterations,
                             inner_check_interval, inner_tolerance,
                             inner_min_iterations, inner_solver,
                             inner_jacobi)
        overrelaxation_started = time.monotonic()
        overrelax_local_result(
            problem, state, block, result,
            block_overrelaxation, block_backtracks, block_safeguard)
        result.runtime_seconds += time.monotonic() - overrelaxation_started
        apply_local_result(state, block, result)
        update_block_statistics(blocks, block, result)
        remaining.remove(block_id)
        results.append(result)
        print(f"  block {block_id:3d}: gain={result.gain: .6e}, "
              f"time={result.runtime_seconds:.2f}s, attempts={result.attempts}, "
              f"inner={result.inner_iterations}, quality={result.model_quality:.3e}, "
              f"damping={block.damping:.3e}, step_weight={result.step_weight:.3f}")
    return state, results


def update_block_statistics(blocks: list[Block], block: Block,
                            result: LocalResult) -> None:
    block.predicted_gain = 0.7 * block.predicted_gain + 0.3 * result.gain
    block.runtime_seconds = 0.7 * block.runtime_seconds + 0.3 * result.runtime_seconds
    for other in blocks:
        other.age += 1
    block.age = 0


def safeguard(problem: BALProblem, nominal: State, candidate: State,
              recent_costs: list[float], backtracks: int,
              nonmonotone_memory: int,
              max_acceleration_increase: float) -> tuple[State, float, float]:
    nominal_cost = objective(problem, nominal)
    envelope = max(recent_costs[-nonmonotone_memory:]) if recent_costs else math.inf
    acceptance_limit = min(
        envelope, nominal_cost * (1.0 + max_acceleration_increase))
    anchor_vector = nominal.vector()
    direction = candidate.vector() - anchor_vector
    n_cameras = nominal.cameras.shape[0]
    for backtrack in range(backtracks + 1):
        weight = 0.5**backtrack
        trial = State.from_vector(anchor_vector + weight * direction, n_cameras)
        trial_cost = objective(problem, trial)
        if np.isfinite(trial_cost) and trial_cost <= acceptance_limit:
            return trial, trial_cost, weight
    return nominal, nominal_cost, 0.0


def safeguard_nominal(problem: BALProblem, anchor: State, nominal: State,
                      anchor_cost: float,
                      backtracks: int) -> tuple[State, float, float]:
    anchor_vector = anchor.vector()
    direction = nominal.vector() - anchor_vector
    n_cameras = anchor.cameras.shape[0]
    for backtrack in range(backtracks + 1):
        weight = 0.5**backtrack
        trial = State.from_vector(anchor_vector + weight * direction, n_cameras)
        trial_cost = objective(problem, trial)
        if np.isfinite(trial_cost) and trial_cost <= anchor_cost:
            return trial, trial_cost, weight
    return anchor.copy(), anchor_cost, 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("file_name")
    parser.add_argument("epochs", nargs="?", type=int, default=30)
    parser.add_argument("partitions", nargs="?", type=int, default=6)
    parser.add_argument("--partitioner", choices=("load", "overlap"), default="load",
                        help="camera assignment: load balance or cut refinement")
    parser.add_argument("--partition-refinement-passes", type=int, default=20,
                        help="maximum accepted camera swaps for overlap partitioning")
    parser.add_argument("--partition-balance-slack", type=float, default=0.0,
                        help="allowed increase over the load partitioner's largest block")
    parser.add_argument("--partition-swap-candidates", type=int, default=256,
                        help="strongest camera pairs scored per pass; 0 uses all pairs")
    parser.add_argument("--execution", choices=("sequential", "parallel"), default="sequential")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel workers; 0 uses the partition count")
    parser.add_argument("--accelerator", choices=("none", "heavy_ball", "nesterov", "anderson", "lbfgs", "bfgs"), default="anderson")
    parser.add_argument("--momentum", type=float, default=0.8)
    parser.add_argument(
        "--momentum-schedule", choices=("constant", "palm"), default="constant",
        help="heavy-ball beta: fixed --momentum or historical (epoch-1)/(epoch+2)")
    parser.add_argument("--memory", type=int, default=5)
    parser.add_argument("--local-nfev", "--local-iterations", dest="local_nfev",
                        type=int, default=3,
                        help="local Schur work budget; 2 performs one accepted step")
    parser.add_argument("--local-solver", choices=("cpu", "bae"), default="cpu",
                        help="Schur backend; bae requires the dedicated BAE environment")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device used by the BAE local solver")
    parser.add_argument("--inner-iterations", type=int, default=200,
                        help="maximum iterations per Schur solve")
    parser.add_argument("--inner-solver", choices=("nesterov", "pcg"),
                        default="nesterov",
                        help="BAE Schur solver")
    parser.add_argument(
        "--inner-jacobi", action=argparse.BooleanOptionalAction, default=None,
        help="scalar Jacobi-normalize the BAE Schur system (default: enabled)")
    parser.add_argument("--inner-check-interval", type=int, default=10,
                        help="check GPU inner-solver convergence every N iterations")
    parser.add_argument("--inner-tolerance", type=float, default=1e-2,
                        help="relative inner-solver stopping tolerance; 0 disables early stopping")
    parser.add_argument("--inner-min-iterations", type=int, default=10,
                        help="minimum inner iterations before convergence stopping")
    parser.add_argument("--initial-damping", type=float, default=1.0,
                        help="initial local LM/PALM damping")
    parser.add_argument("--damping-reject-multiplier", type=float, default=4.0,
                        help="damping growth after a rejected local step")
    parser.add_argument("--local-acceptance", choices=("palm", "decrease"),
                        default="palm",
                        help="require proximal-model decrease or only nonlinear decrease")
    parser.add_argument("--block-overrelaxation", type=float, default=1.0,
                        help="sequential block step factor; values above 1 are safeguarded")
    parser.add_argument("--block-backtracks", type=int, default=3,
                        help="backtracks for sequential block over-relaxation")
    parser.add_argument(
        "--block-safeguard", action=argparse.BooleanOptionalAction, default=True,
        help="evaluate and safeguard extrapolated block steps (disable for no extra evaluations)")
    parser.add_argument("--backtracks", type=int, default=3)
    parser.add_argument("--nonmonotone-memory", type=int, default=5)
    parser.add_argument("--max-accel-increase", type=float, default=0.02)
    parser.add_argument("--age-weight", type=float, default=1.0)
    parser.add_argument("--runtime-weight", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=1e-8)
    parser.add_argument("--max-step-ratio", type=float, default=5.0)
    parser.add_argument("--restart-failures", type=int, default=3)
    parser.add_argument(
        "--global-jacobi", choices=("none", "camera", "full"), default="none",
        help="outer coordinate scaling from the initial global Jacobian diagonal")
    parser.add_argument("--global-jacobi-batch-size", type=int, default=50000)
    parser.add_argument("--global-jacobi-floor", type=float, default=1e-12,
                        help="relative floor applied to Jacobi diagonal entries")
    parser.add_argument("--output", type=Path, default=Path("palm_ba_results.jsonl"))
    parser.add_argument("--live-summary", type=Path,
                        help="single JSON object atomically refreshed each epoch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inner_jacobi is None:
        args.inner_jacobi = args.local_solver == "bae"
    if args.epochs < 1 or args.partitions < 1 or args.local_nfev < 2:
        raise ValueError("epochs and partitions must be positive; local-nfev must be at least 2")
    if args.memory < 1 or args.backtracks < 0 or args.nonmonotone_memory < 1:
        raise ValueError("memory values must be positive and backtracks nonnegative")
    if args.max_accel_increase < 0.0 or args.restart_failures < 1:
        raise ValueError("max-accel-increase must be nonnegative and restart-failures positive")
    if args.global_jacobi_batch_size < 1 or args.global_jacobi_floor <= 0.0:
        raise ValueError("global Jacobi batch size and floor must be positive")
    if (args.inner_iterations < 1 or args.inner_check_interval < 1
            or args.inner_tolerance < 0.0 or args.inner_min_iterations < 1
            or args.inner_min_iterations > args.inner_iterations):
        raise ValueError("inner iterations/check interval must be positive and tolerance nonnegative")
    if args.initial_damping <= 0.0 or args.damping_reject_multiplier <= 1.0:
        raise ValueError("initial damping must be positive and reject multiplier greater than one")
    if args.block_overrelaxation < 1.0 or args.block_backtracks < 0:
        raise ValueError("block over-relaxation must be at least one and backtracks nonnegative")
    if args.execution != "sequential" and args.block_overrelaxation != 1.0:
        raise ValueError("block over-relaxation is available only for sequential execution")
    if (args.partition_refinement_passes < 0 or args.partition_balance_slack < 0.0
            or args.partition_swap_candidates < 0):
        raise ValueError("partition refinement controls must be nonnegative")
    if not 0.0 <= args.momentum < 1.0 or args.workers < 0:
        raise ValueError("momentum must be in [0, 1) and workers nonnegative")
    if args.local_solver == "bae" and args.workers not in (0, 1):
        raise ValueError("the BAE local solver requires --workers 1")
    script_directory = Path(__file__).resolve().parent
    problem_path = ensure_problem(args.base_url, args.file_name, script_directory)
    problem = read_bal(problem_path)
    state = State(problem.cameras.copy(), problem.points.copy())
    global_preconditioner = build_global_jacobi_preconditioner(
        problem, args.global_jacobi, args.global_jacobi_batch_size,
        args.global_jacobi_floor)
    blocks = build_partitions(
        problem, args.partitions, args.partitioner,
        args.partition_refinement_passes, args.partition_balance_slack,
        args.partition_swap_candidates)
    processed_observations = sum(len(block.observation_indices) for block in blocks)
    partition_loads = np.array([len(block.observation_indices) for block in blocks])
    cut_observations = processed_observations - len(problem.observations)
    duplication_factor = processed_observations / len(problem.observations)
    partition_load_ratio = partition_loads.max() / partition_loads.mean()
    print(f"partitioner={args.partitioner}; cuts={cut_observations}; "
          f"duplication={duplication_factor:.4f}; "
          f"block observations={partition_loads.min()}..{partition_loads.max()}")
    for block in blocks:
        block.damping = args.initial_damping
        block.damping_reject_multiplier = args.damping_reject_multiplier
        block.local_acceptance = args.local_acceptance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(args.accelerator, args.memory,
                              args.regularization, args.max_step_ratio,
                              args.momentum, args.momentum_schedule)
    current_cost = objective(problem, state)
    best_state = state.copy()
    best_cost = current_cost
    recent_costs = [current_cost]
    failures = 0
    previous_residual = None
    previous_accepted = global_preconditioner.transform(state.vector())
    started = time.monotonic()

    print(f"BAL: {len(problem.cameras)} cameras, {len(problem.points)} points, "
        f"{len(problem.observations)} observations")
    print(f"PALM: {len(blocks)} blocks, execution={args.execution}, "
        f"accelerator={accelerator.name}, "
        f"initial cost={current_cost:.9e}")
    print(f"Local solver: {args.local_solver}, inner={args.inner_solver}, "
          f"inner_jacobi={args.inner_jacobi}, device={args.device}")
    print(f"Block over-relaxation: {args.block_overrelaxation:.3f}, "
            f"safeguard={args.block_safeguard}, "
            f"backtracks={args.block_backtracks}")
    print(f"Global Jacobi: {global_preconditioner.mode}, "
        f"scale=[{global_preconditioner.scale.min():.3e}, "
        f"{global_preconditioner.scale.max():.3e}]")

    for epoch in range(args.epochs):
        anchor = state.copy()
        anchor_vector = global_preconditioner.transform(anchor.vector())
        nominal, local_results = run_epoch(
            problem, blocks, anchor, args.local_nfev,
            args.age_weight, args.runtime_weight, args.execution,
            1 if args.local_solver == "bae" else args.workers or args.partitions,
            args.local_solver, args.device, args.inner_iterations,
            args.inner_check_interval, args.inner_tolerance,
            args.inner_min_iterations, args.inner_solver,
            args.inner_jacobi, args.block_overrelaxation,
            args.block_backtracks, args.block_safeguard)
        if len(blocks) == 1:
            nominal_cost = local_results[0].local_cost_after
            nominal_weight = 1.0
        else:
            nominal, nominal_cost, nominal_weight = safeguard_nominal(
                problem, anchor, nominal, current_cost, args.backtracks)
        nominal_vector = global_preconditioner.transform(nominal.vector())
        candidate_vector = accelerator.propose(anchor_vector, nominal_vector, epoch)
        candidate = State.from_vector(
            global_preconditioner.inverse_transform(candidate_vector),
            len(problem.cameras))
        if accelerator.name == "none":
            state, current_cost, weight = nominal, nominal_cost, 1.0
        else:
            state, current_cost, weight = safeguard(
                problem, nominal, candidate, recent_costs,
                args.backtracks, args.nonmonotone_memory,
                args.max_accel_increase)
        recent_costs.append(current_cost)

        if weight == 0.0 and not np.array_equal(candidate_vector, nominal_vector):
            failures += 1
        else:
            failures = max(0, failures - 1)
        if failures >= args.restart_failures:
            accelerator.reset()
            previous_residual = None
            failures = 0
            print("  acceleration history restarted")

        accepted_vector = global_preconditioner.transform(state.vector())
        # One extra epoch is too expensive solely for a residual. Use the
        # coherent residual from the just-completed nominal epoch for history.
        epoch_residual = anchor_vector - nominal_vector
        accelerator.states.append(anchor_vector.copy())
        accelerator.residuals.append(epoch_residual.copy())
        accelerator.states[:] = accelerator.states[-(args.memory + 1):]
        accelerator.residuals[:] = accelerator.residuals[-(args.memory + 1):]
        if previous_residual is not None:
            step = accepted_vector - previous_accepted
            difference = epoch_residual - previous_residual
            curvature = float(step @ difference)
            if curvature > 1e-12 * np.linalg.norm(step) * np.linalg.norm(difference):
                accelerator.lbfgs_steps.append(step)
                accelerator.lbfgs_differences.append(difference)
                accelerator.lbfgs_steps[:] = accelerator.lbfgs_steps[-args.memory:]
                accelerator.lbfgs_differences[:] = accelerator.lbfgs_differences[-args.memory:]
        previous_residual = epoch_residual
        previous_accepted = accepted_vector.copy()

        if current_cost < best_cost:
            best_cost = current_cost
            best_state = state.copy()
        elapsed = time.monotonic() - started
        print(f"epoch {epoch:3d}: cost={current_cost:.9e}, "
              f"nominal={nominal_cost:.9e}, nominal_weight={nominal_weight:.3f}, "
              f"accel_weight={weight:.3f}, best={best_cost:.9e}, "
              f"elapsed={elapsed:.1f}s")

        record = {
            "file_name": args.file_name,
            "epoch": epoch,
            "cost": current_cost,
            "best_cost": best_cost,
            "nominal_cost": nominal_cost,
            "nominal_weight": nominal_weight,
            "acceleration_weight": weight,
            "accelerator": accelerator.name,
            "momentum": args.momentum,
            "momentum_schedule": args.momentum_schedule,
            "global_jacobi": args.global_jacobi,
            "global_jacobi_floor": args.global_jacobi_floor,
            "local_solver": args.local_solver,
            "device": args.device,
            "inner_iterations": args.inner_iterations,
            "inner_solver": args.inner_solver,
            "inner_jacobi": args.inner_jacobi,
            "inner_check_interval": args.inner_check_interval,
            "inner_tolerance": args.inner_tolerance,
            "inner_min_iterations": args.inner_min_iterations,
            "initial_damping": args.initial_damping,
            "damping_reject_multiplier": args.damping_reject_multiplier,
            "local_acceptance": args.local_acceptance,
            "block_overrelaxation": args.block_overrelaxation,
            "block_backtracks": args.block_backtracks,
            "block_safeguard": args.block_safeguard,
            "execution": args.execution,
            "partitions": args.partitions,
            "partitioner": args.partitioner,
            "partition_refinement_passes": args.partition_refinement_passes,
            "partition_balance_slack": args.partition_balance_slack,
            "partition_swap_candidates": args.partition_swap_candidates,
            "partition_cut_observations": cut_observations,
            "partition_duplication_factor": duplication_factor,
            "partition_max_mean_load_ratio": partition_load_ratio,
            "partition_block_observations": partition_loads.tolist(),
            "elapsed_seconds": elapsed,
            "block_order": [result.block_id for result in local_results],
            "block_attempts": [result.attempts for result in local_results],
            "block_inner_iterations": [
                result.inner_iterations for result in local_results],
            "block_model_quality": [
                result.model_quality for result in local_results],
            "block_step_weights": [
                result.step_weight for result in local_results],
            "block_damping": [block.damping for block in blocks],
        }
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if args.live_summary is not None:
            live_record = {
                **record,
                "algorithm": "palm_ba",
                "epochs_completed": epoch + 1,
                "epochs_requested": args.epochs,
                "status": "running" if epoch + 1 < args.epochs else "completed",
                "trajectory": str(args.output),
            }
            temporary_summary = args.live_summary.with_suffix(".tmp")
            with temporary_summary.open("w", encoding="utf-8") as handle:
                json.dump(live_record, handle, indent=2)
                handle.write("\n")
            temporary_summary.replace(args.live_summary)

    state = best_state
    np.savez_compressed(
        args.output.with_suffix(".npz"), cameras=state.cameras,
        points=state.points, best_cost=best_cost,
        global_jacobi=args.global_jacobi,
        global_jacobi_scale=global_preconditioner.scale)
    print(f"best cost={best_cost:.9e}; state saved to {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
