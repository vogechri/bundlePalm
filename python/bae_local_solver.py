"""Optional BAE/Warp CUDA backend for one PALM local Schur step."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pypose as pp
import torch
import torch.nn as nn
import warp as wp
from pypose.autograd.function import psjac
from warp import sparse

from bae.autograd.graph import jacobian
from bae.sparse.py_ops import inv_op
from bae.sparse.warp_wrappers import format_vec_for_bsr, torchbsr2wp, wp2torchbsr

_MODEL_CACHE: dict[tuple[str, object], "LocalResidual"] = {}


def clear_model_cache(device: str | None = None) -> None:
    if device is None:
        _MODEL_CACHE.clear()
        return
    for key in [key for key in _MODEL_CACHE if key[0] == device]:
        del _MODEL_CACHE[key]


@dataclass(frozen=True)
class BaeStep:
    delta_p: np.ndarray
    delta_l: np.ndarray
    residual: np.ndarray
    cost_before: float
    cost_after: float
    predicted_reduction: float
    iterations: int
    camera_penalty: float
    point_penalty: float
    timings: dict[str, float]


@psjac
def project_angle_axis(points: torch.Tensor,
                       cameras: torch.Tensor) -> torch.Tensor:
    rotation_vectors = cameras[..., :3]
    theta_squared = torch.sum(rotation_vectors * rotation_vectors, dim=-1)
    mask = theta_squared > 0.0
    theta = torch.sqrt(torch.where(mask, theta_squared,
                                   torch.ones_like(theta_squared)))
    directions = rotation_vectors / theta[..., None]
    dot = torch.sum(points * directions, dim=-1)[..., None]
    cross = torch.linalg.cross(directions, points, dim=-1)
    rotated = (torch.cos(theta)[..., None] * points
               + torch.sin(theta)[..., None] * cross
               + dot * (1.0 - torch.cos(theta))[..., None] * directions)
    first_order = points + torch.linalg.cross(rotation_vectors, points, dim=-1)
    transformed = torch.where(mask[..., None], rotated, first_order)
    transformed = transformed + cameras[..., 3:6]
    normalized = -transformed[..., :2] / transformed[..., 2, None]
    radius_squared = torch.sum(normalized * normalized, dim=-1)
    distortion = (1.0 + cameras[..., 7] * radius_squared
                  + cameras[..., 8] * radius_squared**2)
    return normalized * (cameras[..., 6] * distortion)[..., None]


class LocalResidual(nn.Module):
    def __init__(self, local_cameras: np.ndarray, local_points: np.ndarray,
                 observation_cameras: np.ndarray,
                 observation_points: np.ndarray,
                 observations: np.ndarray,
                 camera_indices: np.ndarray,
                 point_indices: np.ndarray,
                 device: str,
                 shared_cameras: torch.Tensor | None = None,
                 shared_points: torch.Tensor | None = None,
                 observation_camera_indices: torch.Tensor | None = None,
                 observation_point_indices: torch.Tensor | None = None):
        super().__init__()
        dtype = torch.float64
        self.cameras = pp.Parameter(torch.as_tensor(
            local_cameras, dtype=dtype, device=device), sjac=True)
        self.points = pp.Parameter(torch.as_tensor(
            local_points, dtype=dtype, device=device), sjac=True)
        self.shared_cameras = shared_cameras
        self.shared_points = shared_points
        self.observation_camera_indices = observation_camera_indices
        self.observation_point_indices = observation_point_indices
        if ((shared_cameras is None) != (shared_points is None)
                or (shared_cameras is None) != (observation_camera_indices is None)
                or (shared_cameras is None) != (observation_point_indices is None)):
            raise ValueError("shared state and observation indices must be provided together")
        stored_observation_cameras = (observation_cameras
            if shared_cameras is None else np.empty((0, 9), dtype=np.float64))
        stored_observation_points = (observation_points
            if shared_points is None else np.empty((0, 3), dtype=np.float64))
        self.register_buffer("observation_cameras", torch.as_tensor(
            stored_observation_cameras, dtype=dtype, device=device))
        self.register_buffer("observation_points", torch.as_tensor(
            stored_observation_points, dtype=dtype, device=device))
        self.register_buffer("observations", torch.as_tensor(
            observations, dtype=dtype, device=device))
        self.register_buffer("camera_indices", torch.as_tensor(
            camera_indices, dtype=torch.long, device=device))
        self.register_buffer("point_indices", torch.as_tensor(
            point_indices, dtype=torch.long, device=device))
        self._use_cached_groups = (
            os.environ.get("BUNDLE_PALM_BAE_CACHED_GROUPS", "1") != "0")
        camera_owned = camera_indices >= 0
        point_owned = point_indices >= 0
        self._cached_groups: list[tuple[str, bool, bool]] = []
        for group_id, (mask, local_camera, local_point) in enumerate((
                (camera_owned & point_owned, True, True),
                (camera_owned & ~point_owned, True, False),
                (~camera_owned & point_owned, False, True))):
            name = f"ownership_group_{group_id}"
            self.register_buffer(name, torch.as_tensor(
                np.flatnonzero(mask), dtype=torch.long, device=device))
            self._cached_groups.append((name, local_camera, local_point))

    @torch.no_grad()
    def update(self, local_cameras: np.ndarray, local_points: np.ndarray,
               observation_cameras: np.ndarray,
               observation_points: np.ndarray) -> None:
        self.cameras.copy_(torch.as_tensor(
            local_cameras, dtype=self.cameras.dtype,
            device=self.cameras.device))
        self.points.copy_(torch.as_tensor(
            local_points, dtype=self.points.dtype, device=self.points.device))
        if self.shared_cameras is None:
            self.observation_cameras.copy_(torch.as_tensor(
                observation_cameras, dtype=self.observation_cameras.dtype,
                device=self.observation_cameras.device))
            self.observation_points.copy_(torch.as_tensor(
                observation_points, dtype=self.observation_points.dtype,
                device=self.observation_points.device))

    def _fixed_cameras(self, indices: torch.Tensor) -> torch.Tensor:
        if self.shared_cameras is None:
            return self.observation_cameras[indices]
        return self.shared_cameras[self.observation_camera_indices[indices]]

    def _fixed_points(self, indices: torch.Tensor) -> torch.Tensor:
        if self.shared_points is None:
            return self.observation_points[indices]
        return self.shared_points[self.observation_point_indices[indices]]

    def forward(self) -> torch.Tensor:
        if not self._use_cached_groups:
            return self._forward_dynamic()
        groups = []
        for name, local_camera, local_point in self._cached_groups:
            indices = getattr(self, name)
            if indices.numel() == 0:
                continue
            cameras = (self.cameras[self.camera_indices[indices]]
                       if local_camera else self._fixed_cameras(indices))
            points = (self.points[self.point_indices[indices]]
                      if local_point else self._fixed_points(indices))
            groups.append(project_angle_axis(points, cameras)
                          - self.observations[indices])
        return torch.cat(groups, dim=0)

    def _forward_dynamic(self) -> torch.Tensor:
        camera_owned = self.camera_indices >= 0
        point_owned = self.point_indices >= 0
        groups = []
        for mask in (camera_owned & point_owned,
                     camera_owned & ~point_owned,
                     ~camera_owned & point_owned):
            if not bool(mask.any()):
                continue
            cameras = (self.cameras[self.camera_indices[mask]]
                       if bool((self.camera_indices[mask] >= 0).all())
                       else self._fixed_cameras(mask))
            points = (self.points[self.point_indices[mask]]
                      if bool((self.point_indices[mask] >= 0).all())
                      else self._fixed_points(mask))
            groups.append(project_angle_axis(points, cameras)
                          - self.observations[mask])
        return torch.cat(groups, dim=0)


def _damp_block_diagonal(matrix: torch.Tensor, damping: float,
                         epsilon: float) -> None:
    values = matrix.values()
    columns = matrix.col_indices()
    block_count = matrix.crow_indices().numel() - 1
    expected_columns = torch.arange(block_count, device=columns.device)
    if values.shape[0] != block_count or not torch.equal(columns, expected_columns):
        raise RuntimeError("expected a block-diagonal metric")
    values.mul_(1.0 + damping)
    values.diagonal(dim1=-2, dim2=-1).add_(damping * epsilon)


def _block_pseudoinverse(matrix: torch.Tensor) -> torch.Tensor:
    return torch.sparse_bsr_tensor(
        matrix.crow_indices(), matrix.col_indices(),
        torch.linalg.pinv(matrix.values()), size=matrix.shape,
        dtype=matrix.dtype, device=matrix.device)


def _jacobi_scale(matrix: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(matrix.values().diagonal(
        dim1=-2, dim2=-1).abs()).clamp_min_(1e-10).reshape(-1).reciprocal_()


def _scale_block_diagonal(matrix: torch.Tensor,
                          inverse_scale: torch.Tensor) -> None:
    block_scale = inverse_scale.reshape(matrix.values().shape[0], -1)
    matrix.values().mul_(block_scale[:, :, None] * block_scale[:, None, :])


def _scale_bsr(matrix: torch.Tensor, row_inverse_scale: torch.Tensor,
               column_inverse_scale: torch.Tensor) -> None:
    values = matrix.values()
    row_block_size, column_block_size = values.shape[-2:]
    row_scale = row_inverse_scale.reshape(-1, row_block_size)
    column_scale = column_inverse_scale.reshape(-1, column_block_size)
    row_counts = matrix.crow_indices()[1:] - matrix.crow_indices()[:-1]
    row_indices = torch.repeat_interleave(
        torch.arange(len(row_counts), device=values.device), row_counts)
    values.mul_(row_scale[row_indices, :, None]
                * column_scale[matrix.col_indices(), None, :])


def _solve_nesterov(U: torch.Tensor, W, Wt, V_i, rhs_c, rhs_p,
                     max_iterations: int, tolerance: float,
                     check_interval: int,
                     min_iterations: int) -> tuple[object, int]:
    Lip = 1.0
    lambda0 = (1.0 + math.sqrt(5.0)) / 2.0
    U_i = torchbsr2wp(_block_pseudoinverse(U))
    ubs = wp.zeros_like(rhs_c)
    sparse.bsr_mv(U_i, rhs_c, y=ubs, beta=0.0)
    ubs_t = wp.to_torch(ubs)
    ubs_t.neg_()
    xk = wp.empty_like(rhs_c)
    xk_t = wp.to_torch(xk)
    xk_t.copy_(ubs_t).neg_()
    y0_t = xk_t.clone()
    scratch_points = wp.empty_like(rhs_p)
    scratch_points_2 = wp.empty_like(rhs_p)
    temporary = wp.empty_like(rhs_c)
    gradient = wp.empty_like(rhs_c)
    gradient_t = wp.to_torch(gradient)
    yk_t = torch.empty_like(xk_t)
    inverse_lipschitz = 1.0 / Lip

    for iteration in range(max_iterations):
        lambda1 = (1.0 + math.sqrt(1.0 + 4.0 * lambda0**2)) / 2.0
        gamma = (1.0 - lambda0) / lambda1
        lambda0 = lambda1
        sparse.bsr_mv(Wt, xk, y=scratch_points, beta=0.0)
        sparse.bsr_mv(V_i, scratch_points, y=scratch_points_2, beta=0.0)
        sparse.bsr_mv(W, scratch_points_2, y=temporary, beta=0.0)
        sparse.bsr_mv(U_i, temporary, y=gradient, beta=0.0)
        gradient_t.copy_(xk_t - gradient_t + ubs_t)
        yk_t.copy_(xk_t - inverse_lipschitz * gradient_t)
        xk_t.copy_((1.0 - gamma) * yk_t + gamma * y0_t)
        y0_t.copy_(yk_t)
        if ((iteration + 1) >= min_iterations
            and (iteration + 1) % check_interval == 0):
            step_norm = torch.linalg.norm(inverse_lipschitz * gradient_t)
            if ((iteration + 1) * step_norm
                    < tolerance * torch.linalg.norm(xk_t)):
                return xk, iteration + 1
    return xk, max_iterations


def _solve_pcg(U: torch.Tensor, W, Wt, V_i, rhs_c, rhs_p,
               max_iterations: int, tolerance: float,
               check_interval: int,
               min_iterations: int) -> tuple[object, int]:
    U_wp = torchbsr2wp(U)
    U_i = torchbsr2wp(_block_pseudoinverse(U))
    scratch_points = wp.empty_like(rhs_p)
    scratch_points_2 = wp.empty_like(rhs_p)

    def schur_matvec(value, output) -> None:
        sparse.bsr_mv(Wt, value, y=scratch_points, beta=0.0)
        sparse.bsr_mv(V_i, scratch_points, y=scratch_points_2, beta=0.0)
        sparse.bsr_mv(W, scratch_points_2, y=output, beta=0.0)
        sparse.bsr_mv(U_wp, value, y=output, alpha=1.0, beta=-1.0)

    solution = wp.zeros_like(rhs_c)
    solution_t = wp.to_torch(solution)
    residual = wp.empty_like(rhs_c)
    wp.copy(src=rhs_c, dest=residual)
    residual_t = wp.to_torch(residual)
    preconditioned = wp.empty_like(rhs_c)
    sparse.bsr_mv(U_i, residual, y=preconditioned, beta=0.0)
    preconditioned_t = wp.to_torch(preconditioned)
    direction = wp.empty_like(rhs_c)
    wp.copy(src=preconditioned, dest=direction)
    direction_t = wp.to_torch(direction)
    residual_dot = torch.dot(
        residual_t.reshape(-1), preconditioned_t.reshape(-1))
    product = wp.empty_like(rhs_c)
    product_t = wp.to_torch(product)

    for iteration in range(max_iterations):
        schur_matvec(direction, product)
        curvature = torch.dot(
            direction_t.reshape(-1), product_t.reshape(-1))
        if not torch.isfinite(curvature) or curvature <= 0.0:
            return solution, iteration
        alpha = residual_dot / curvature
        solution_t.add_(direction_t, alpha=float(alpha))
        residual_t.add_(product_t, alpha=-float(alpha))
        completed = iteration + 1
        if (completed >= min_iterations
                and completed % check_interval == 0
                and torch.linalg.norm(residual_t)
                < tolerance * torch.linalg.norm(solution_t)):
            return solution, completed
        sparse.bsr_mv(U_i, residual, y=preconditioned, beta=0.0)
        next_residual_dot = torch.dot(
            residual_t.reshape(-1), preconditioned_t.reshape(-1))
        if not torch.isfinite(next_residual_dot) or residual_dot == 0.0:
            return solution, completed
        beta = next_residual_dot / residual_dot
        direction_t.mul_(float(beta)).add_(preconditioned_t)
        residual_dot = next_residual_dot
    return solution, max_iterations


@torch.no_grad()
def solve_local_step(local_cameras: np.ndarray, local_points: np.ndarray,
                     observation_cameras: np.ndarray,
                     observation_points: np.ndarray,
                     observations: np.ndarray,
                     camera_indices: np.ndarray,
                     point_indices: np.ndarray,
                     damping: float, epsilon: float = 1e-4,
                     max_iterations: int = 200, tolerance: float = 1e-2,
                     check_interval: int = 10,
                     min_iterations: int = 10,
                     inner_solver: str = "nesterov",
                     jacobi_preconditioner: bool = False,
                     device: str = "cuda",
                     cache_key: object | None = None,
                     profile: bool = False,
                     shared_cameras: torch.Tensor | None = None,
                     shared_points: torch.Tensor | None = None,
                     observation_camera_indices: torch.Tensor | None = None,
                     observation_point_indices: torch.Tensor | None = None) -> BaeStep:
    profile = profile or os.environ.get("BUNDLE_PALM_PROFILE_LOCAL") == "1"
    timings: dict[str, float] = {}
    previous_time = time.perf_counter()

    def record_timing(name: str) -> None:
        nonlocal previous_time
        if profile:
            wp.synchronize()
            current_time = time.perf_counter()
            timings[name] = current_time - previous_time
            previous_time = current_time

    shared_state = shared_cameras is not None
    key = ((device, cache_key, shared_state)
           if cache_key is not None else None)
    model = _MODEL_CACHE.get(key) if key is not None else None
    if model is None:
        model = LocalResidual(
            local_cameras, local_points, observation_cameras,
            observation_points, observations, camera_indices,
            point_indices, device, shared_cameras, shared_points,
            observation_camera_indices, observation_point_indices)
        if key is not None:
            _MODEL_CACHE[key] = model
    else:
        model.update(local_cameras, local_points, observation_cameras,
                     observation_points)
    record_timing("model")
    residual = model()
    record_timing("residual")
    J_pose, J_land = jacobian(residual, [model.cameras, model.points])
    record_timing("jacobian")
    J_pose_wp = torchbsr2wp(J_pose)
    J_land_wp = torchbsr2wp(J_land)
    J_pose_t = sparse.bsr_transposed(J_pose_wp)
    J_land_t = sparse.bsr_transposed(J_land_wp)
    record_timing("jacobian_conversion")
    U_wp = sparse.bsr_mm(J_pose_t, J_pose_wp)
    record_timing("assemble_U")
    V_wp = sparse.bsr_mm(J_land_t, J_land_wp)
    record_timing("assemble_V")
    W = sparse.bsr_mm(J_pose_t, J_land_wp)
    Wt = sparse.bsr_transposed(W)
    record_timing("assemble_W")
    U = wp2torchbsr(U_wp)
    V = wp2torchbsr(V_wp)
    camera_inverse_scale = None
    point_inverse_scale = None
    if jacobi_preconditioner:
        camera_inverse_scale = _jacobi_scale(U)
        point_inverse_scale = _jacobi_scale(V)
        _scale_block_diagonal(U, camera_inverse_scale)
        _scale_block_diagonal(V, point_inverse_scale)
        W_torch = wp2torchbsr(W)
        _scale_bsr(W_torch, camera_inverse_scale, point_inverse_scale)
        W = torchbsr2wp(W_torch)
        Wt = sparse.bsr_transposed(W)
    record_timing("metric_conversion")
    _damp_block_diagonal(U, damping, epsilon)
    _damp_block_diagonal(V, damping, epsilon)
    U_wp = torchbsr2wp(U)
    V_i = torchbsr2wp(inv_op(V))
    record_timing("damping_inverse")

    residual_flat = residual.tensor().reshape(-1).contiguous()
    residual_wp = format_vec_for_bsr(
        residual_flat, (J_pose_wp.block_shape[1], J_pose_wp.block_shape[0]))
    rhs_camera = sparse.bsr_mv(
        J_pose_wp, residual_wp, alpha=-1.0, transpose=True)
    rhs_point = sparse.bsr_mv(
        J_land_wp, residual_wp, alpha=-1.0, transpose=True)
    if jacobi_preconditioner:
        wp.to_torch(rhs_camera).reshape(-1).mul_(camera_inverse_scale)
        wp.to_torch(rhs_point).reshape(-1).mul_(point_inverse_scale)
    reduced_rhs = wp.empty_like(rhs_camera)
    scratch_points = wp.empty_like(rhs_point)
    wp.copy(src=rhs_camera, dest=reduced_rhs)
    sparse.bsr_mv(V_i, rhs_point, y=scratch_points, beta=0.0)
    sparse.bsr_mv(W, scratch_points, y=reduced_rhs,
                  alpha=-1.0, beta=1.0)
    record_timing("rhs")

    if inner_solver == "nesterov":
        delta_camera, iterations = _solve_nesterov(
            U, W, Wt, V_i, reduced_rhs, rhs_point,
            max_iterations, tolerance, check_interval, min_iterations)
    elif inner_solver == "pcg":
        delta_camera, iterations = _solve_pcg(
            U, W, Wt, V_i, reduced_rhs, rhs_point,
            max_iterations, tolerance, check_interval, min_iterations)
    else:
        raise ValueError(f"unknown inner solver: {inner_solver}")
    record_timing("inner_solve")
    point_rhs = wp.empty_like(rhs_point)
    wp.copy(src=rhs_point, dest=point_rhs)
    sparse.bsr_mv(Wt, delta_camera, y=point_rhs, alpha=-1.0, beta=1.0)
    delta_point = wp.zeros_like(rhs_point)
    sparse.bsr_mv(V_i, point_rhs, y=delta_point, beta=0.0)
    normalized_camera_norm_squared = float(
        wp.to_torch(delta_camera).reshape(-1).square().sum())
    normalized_point_norm_squared = float(
        wp.to_torch(delta_point).reshape(-1).square().sum())
    if jacobi_preconditioner:
        wp.to_torch(delta_camera).reshape(-1).mul_(camera_inverse_scale)
        wp.to_torch(delta_point).reshape(-1).mul_(point_inverse_scale)
    wp.synchronize()

    delta_camera_t = wp.to_torch(delta_camera).reshape(-1)
    delta_point_t = wp.to_torch(delta_point).reshape(-1)
    camera_linearized_change = sparse.bsr_mv(J_pose_wp, delta_camera)
    point_linearized_change = sparse.bsr_mv(J_land_wp, delta_point)
    camera_linearized_change_t = wp.to_torch(
        camera_linearized_change).reshape(-1)
    point_linearized_change_t = wp.to_torch(
        point_linearized_change).reshape(-1)
    linearized_change = wp.empty_like(camera_linearized_change)
    wp.copy(src=camera_linearized_change, dest=linearized_change)
    sparse.bsr_mv(J_land_wp, delta_point, y=linearized_change,
                  alpha=1.0, beta=1.0)
    linearized_change_t = wp.to_torch(linearized_change).reshape(-1)
    predicted_reduction = float(-(
        2.0 * (residual_flat @ linearized_change_t)
        + linearized_change_t @ linearized_change_t))
    camera_penalty = float(
        camera_linearized_change_t @ camera_linearized_change_t
        + epsilon * normalized_camera_norm_squared)
    point_penalty = float(
        point_linearized_change_t @ point_linearized_change_t
        + epsilon * normalized_point_norm_squared)
    cost_before = float(residual_flat @ residual_flat)
    model.cameras.add_(delta_camera_t.reshape_as(model.cameras))
    model.points.add_(delta_point_t.reshape_as(model.points))
    candidate_residual = model()
    cost_after = float(candidate_residual.tensor().square().sum())
    record_timing("backsub_penalty")

    if profile:
        print("BAE_PROFILE " + json.dumps({
            "iterations": iterations,
            "timings": timings,
        }, separators=(",", ":")), file=sys.stderr, flush=True)

    return BaeStep(
        delta_camera_t.cpu().numpy().copy(),
        delta_point_t.cpu().numpy().copy(),
        residual_flat.cpu().numpy().copy(),
        cost_before,
        cost_after,
        predicted_reduction,
        iterations,
        camera_penalty,
        point_penalty,
        timings,
    )
