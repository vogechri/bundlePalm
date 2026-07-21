"""
Jacobi (diagonal) preconditioner for Bundle Adjustment.

Computes a diagonal scaling from the Jacobian structure:
  D_c = sqrt(|diag(J_c^T J_c)|)   (per camera parameter)
  D_p = sqrt(|diag(J_p^T J_p)|)   (per point parameter)

Rescales parameters: y = D * x.  After optimization, recover x = D^{-1} * y.

Designed for axis-angle parameterization where all camera parameters are free.
For quaternion/SE3 parameterizations, use scale_cameras=False or restrict to
intrinsics only.

Usage (axis-angle, all params free):
  precond = JacobiPreconditioner(model, input_dict, num_camera_params,
                                  scale_cameras=True, scale_points=True)
  precond.apply(model)
  # ... optimize ...
  precond.unapply(model)

Usage (quaternion/SE3, only points):
  precond = JacobiPreconditioner(model, input_dict, num_camera_params,
                                  scale_cameras=False, scale_points=True)
"""

import torch
import torch.nn as nn
import numpy as np


def _get_scaling(min_val, max_val):
    """Compute scaling factor so that min*s * max*s ~ 1, i.e. s = 1/sqrt(min*max)."""
    return np.sqrt(1.0 / (min_val * max_val))


def compute_jacobi_diagonal(model, input_dict, num_camera_params, project_fn=None):
    """Compute sqrt(|diag(J^T J)|) for cameras and points from the residual Jacobian.

    Args:
        model: Residual module with .pose and .points parameters.
        input_dict: dict with 'observes', 'cidx', 'pidx' keys.
        num_camera_params: number of camera parameters.
        project_fn: optional projection function(points, cameras) -> projected.
                    If None, uses a default SE3 projection.

    Returns:
        diag_c: (num_cameras, cam_block_size) sqrt of diagonal of J_c^T J_c.
        diag_p: (num_points, 3) sqrt of diagonal of J_p^T J_p.
    """
    observes = input_dict["observes"]
    cidx = input_dict["cidx"]
    pidx = input_dict["pidx"]

    num_cameras = model.pose.shape[0]
    num_points = model.points.shape[0]
    residual_dim = 2

    with torch.enable_grad():
        pose = model.pose.detach().clone().requires_grad_(True)
        points = model.points.detach().clone().requires_grad_(True)

        cam_sel = pose[cidx]
        pts_sel = points[pidx]

        if project_fn is not None:
            projected = project_fn(pts_sel, cam_sel)
        else:
            import pypose as pp
            projection = pp.SE3(cam_sel[..., :7]).Act(pts_sel)
            projection = -projection[..., :2] / projection[..., [2]]
            f = cam_sel[..., [-3]]
            k1 = cam_sel[..., [-2]]
            k2 = cam_sel[..., [-1]]
            n = torch.sum(projection**2, axis=-1, keepdim=True)
            r = 1 + k1 * n + k2 * n**2
            projected = projection * r * f

        residual = projected - observes

    diag_c = torch.zeros(num_cameras, num_camera_params, device=pose.device, dtype=pose.dtype)
    diag_p = torch.zeros(num_points, 3, device=points.device, dtype=points.dtype)

    for r_idx in range(residual_dim):
        grad_out = torch.zeros_like(residual)
        grad_out[:, r_idx] = 1.0
        grads = torch.autograd.grad(residual, (cam_sel, pts_sel), grad_outputs=grad_out,
                                    retain_graph=True, allow_unused=True)
        if grads[0] is not None:
            diag_c.scatter_add_(0, cidx.unsqueeze(1).expand_as(grads[0]), grads[0] ** 2)
        if grads[1] is not None:
            diag_p.scatter_add_(0, pidx.unsqueeze(1).expand_as(grads[1]), grads[1] ** 2)

    diag_c = torch.sqrt(diag_c.abs())
    diag_p = torch.sqrt(diag_p.abs())

    return diag_c, diag_p


def _scale_and_clamp(diag, clamp_min, clamp_max):
    """Apply percentile-based scaling then clamp."""
    flat = diag.flatten()
    nonzero_mask = flat > 0
    if nonzero_mask.any():
        nonzero_vals = flat[nonzero_mask]
        min_val = torch.quantile(nonzero_vals, 0.0001).item()
        max_val = nonzero_vals.max().item()
        if min_val > 0 and max_val > 0:
            s = np.sqrt(1.0 / (min_val * max_val))
            diag = diag * s
    diag = diag.clamp(min=clamp_min, max=clamp_max)
    return diag


class JacobiPreconditioner:
    """Jacobi (diagonal) preconditioner for BA.

    Scales parameters by sqrt(diag(J^T J)) to normalize Jacobian columns.

    For axis-angle (all params free): use scale_cameras=True, scale_points=True.
    For quaternion/SE3: use scale_cameras=False (or scale only intrinsics).
    """

    def __init__(self, model, input_dict, num_camera_params,
                 clamp_min=1e-18, clamp_max=1e18,
                 scale_cameras=True, scale_points=True,
                 project_fn=None):
        """
        Args:
            scale_cameras: scale all camera parameters (for axis-angle).
            scale_points: scale point parameters.
            project_fn: custom projection function for Jacobian computation.
        """
        diag_c, diag_p = compute_jacobi_diagonal(model, input_dict, num_camera_params,
                                                  project_fn=project_fn)

        if scale_cameras:
            self.D_c = _scale_and_clamp(diag_c, clamp_min, clamp_max)
        else:
            self.D_c = torch.ones_like(diag_c)

        if scale_points:
            self.D_p = _scale_and_clamp(diag_p, clamp_min, clamp_max)
        else:
            self.D_p = torch.ones_like(diag_p)

        self.D_c_inv = 1.0 / self.D_c
        self.D_p_inv = 1.0 / self.D_p
        self.num_camera_params = num_camera_params
        self.scale_cameras = scale_cameras
        self.scale_points = scale_points

    def apply(self, model):
        """Scale parameters: y = D * x."""
        with torch.no_grad():
            if self.scale_cameras:
                model.pose[:, :self.num_camera_params] *= self.D_c[:, :self.num_camera_params]
            if self.scale_points:
                model.points *= self.D_p

    def unapply(self, model):
        """Unscale parameters: x = D^{-1} * y."""
        with torch.no_grad():
            if self.scale_cameras:
                model.pose[:, :self.num_camera_params] *= self.D_c_inv[:, :self.num_camera_params]
            if self.scale_points:
                model.points *= self.D_p_inv

    def to(self, device):
        """Move preconditioner tensors to device."""
        self.D_c = self.D_c.to(device)
        self.D_p = self.D_p.to(device)
        self.D_c_inv = self.D_c_inv.to(device)
        self.D_p_inv = self.D_p_inv.to(device)
        return self
