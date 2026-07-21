"""
Power Bundle Adjustment example.

Implements the power series expansion method from:
  Weber et al., "Power Bundle Adjustment for Large-Scale 3D Reconstruction", ECCV 2022.
  https://arxiv.org/abs/2204.12834

Instead of solving the Schur complement system S * Dc = rhs with PCG,
we use the power series expansion:
  S^{-1} = sum_{k=0}^{K} (U^{-1} W V^{-1} W^T)^k U^{-1}

which amounts to the iteration:
  Dc^{(0)} = U^{-1} rhs
  Dc^{(k+1)} = U^{-1} (rhs + W V^{-1} W^T Dc^{(k)})

This avoids an inner CG loop entirely — each LM iteration just does K
sparse matrix-vector products.
"""
import argparse
import json
import math
from time import perf_counter
from datetime import datetime
from pathlib import Path
from functools import partial
import inspect
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datapipes.bal_io import DTYPE
import pypose as pp
import torch
import torch.nn as nn
import warp as wp
from pypose.autograd.function import psjac

from bae.optim.optimizer import Schur
from bae.autograd.graph import jacobian
from bae.sparse.py_ops import diagonal_op_, inv_op
from bae.sparse.warp_wrappers import format_vec_for_bsr, torchbsr2wp, wp2torchbsr
from bae.utils.parameter import parameter_update_shape
from bae.utils.jacobi_preconditioner import JacobiPreconditioner
from datapipes.bal_loader import get_problem
from bae.optim import LM
from bae.optim.strategy import TrustRegion
from warp import sparse
from warp.optim import linear

###########
# TARGET_DATASET = "trafalgar"
# TARGET_PROBLEM = "problem-257-65132-pre"

# Total time: 3.639s
# CG max iters: 400, eps: 0.0001, LM iters: 20, solver: PCG, jacobi: True, normalize: False
# Peak CUDA memory: 289.2 MiB
# Ending MSE loss: 0.8536947696305146
# Total time: 1.816s
# CG max iters: 400, eps: 0.001, LM iters: 20, solver: PCG, jacobi: True, normalize: False
# Peak CUDA memory: 289.2 MiB
# Ending MSE loss: 0.8669610138729859

###########
# TARGET_DATASET = "ladybug"
# TARGET_PROBLEM = "problem-1723-156502-pre"

# Total time: 3.323s
# CG max iters: 400, eps: 0.0001, LM iters: 20, solver: PCG, jacobi: True, normalize: False
# Peak CUDA memory: 830.9 MiB
# Ending MSE loss: 1.1136686842978223

# TARGET_DATASET = "ladybug"
# TARGET_PROBLEM = "problem-1266-132593-pre"

###########
# TARGET_DATASET = "dubrovnik"
# TARGET_PROBLEM = "problem-356-226730-pre"

# Total time: 6.520s
# CG max iters: 400, eps: 0.0001, LM iters: 20, solver: PCG, jacobi: True, normalize: False
# Peak CUDA memory: 1518.6 MiB
# Ending MSE loss: 0.791057801360159
###########

# TARGET_DATASET = "venice"
# TARGET_PROBLEM = "problem-951-708276-pre"

# Total time: 13.517s
# CG max iters: 400, eps: 0.0001, LM iters: 20, solver: PCG, jacobi: True, normalize: False
# Peak CUDA memory: 4487.6 MiB
# Ending MSE loss: 0.8919742598589668
###########

TARGET_DATASET = "final"
TARGET_PROBLEM = "problem-871-527480-pre"

# too large! for 16 GB GPU!
# TARGET_DATASET = "final"
# TARGET_PROBLEM = "problem-4585-1324582-pre"

# TARGET_DATASET = "final"
# TARGET_PROBLEM = "problem-1936-649673-pre"
# Total time: 15.039s
# CG max iters: 400, eps: 0.0001, LM iters: 20, solver: PCG, jacobi: True, normalize: False
# Peak CUDA memory: 6219.6 MiB -- 11GB allocated
# Ending MSE loss: 1.7840453817100266
# Total time: 71.136s
# CG max iters: 400, eps: 0.0001, LM iters: 20, solver: PCG, jacobi: False, normalize: False
# Peak CUDA memory: 6219.6 MiB
# Ending MSE loss: 1.7840453464447124
# Total time: 15.078s
# CG max iters: 400, eps: 0.0001, LM iters: 20, solver: PCG, jacobi: True, normalize: True
# Peak CUDA memory: 6219.6 MiB
# Ending MSE loss: 1.7840453817100261

#
TARGET_DATASET = "venice"
TARGET_PROBLEM = "problem-52-64053-pre"

TARGET_DATASET = "ladybug"
TARGET_PROBLEM = "problem-1064-113655-pre"

TARGET_DATASET = "venice"
TARGET_PROBLEM = "problem-245-198739-pre"

#######################################
DEVICE = "cuda"
OPTIMIZE_INTRINSICS = True
NUM_CAMERA_PARAMS = 10 if OPTIMIZE_INTRINSICS else 7
CG_MAX_ITERS = 400  # Max CG iterations per Schur solve
EARLY_STOPPING_EPS = 1e-4  # Convergence threshold for CG residual needs to be smaller
GLOBAL_ITERATIONS = 20  # Number of LM iterations
USE_JACOBI_PRECOND = True # Precondition the linear system (not the parameters) Must do for cg or normalize!
NORMALIZE_DATASET = False
SCENE_SCALE = 100.0  # Scale the scene so that 95% of points are within this distance from the median
PROFILE = False  # Enable pyinstrument profiling

import numpy as np

def invert_focal_distance(camera_params_, camera_indices_, points_2d_):
    flipIndices = camera_params_[:,6] < 0
    flipCamIds = np.arange(camera_params_.shape[0])[flipIndices]
    camera_params_[flipCamIds,6] *= -1
    flip_point_ids = np.isin(camera_indices_, flipCamIds)
    points_2d_[flip_point_ids] *= -1
    return camera_params_, points_2d_

def AngleAxisRotatePoint(angleAxis, pt):
    theta2 = np.sum(angleAxis * angleAxis, axis=1)

    mask = (theta2 > 0).astype(float)

    theta = np.sqrt(theta2 + (1 - mask))

    mask = np.hstack([mask[:, np.newaxis], mask[:, np.newaxis], mask[:, np.newaxis]])

    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    thetaInverse = 1.0 / theta

    w0 = angleAxis[:, 0] * thetaInverse
    w1 = angleAxis[:, 1] * thetaInverse
    w2 = angleAxis[:, 2] * thetaInverse

    wCrossPt0 = w1 * pt[:, 2] - w2 * pt[:, 1]
    wCrossPt1 = w2 * pt[:, 0] - w0 * pt[:, 2]
    wCrossPt2 = w0 * pt[:, 1] - w1 * pt[:, 0]

    tmp_ = (w0 * pt[:, 0] + w1 * pt[:, 1] + w2 * pt[:, 2]) * (1.0 - costheta)

    r0 = pt[:, 0] * costheta + wCrossPt0 * sintheta + w0 * tmp_
    r1 = pt[:, 1] * costheta + wCrossPt1 * sintheta + w1 * tmp_
    r2 = pt[:, 2] * costheta + wCrossPt2 * sintheta + w2 * tmp_

    res1 = np.vstack([r0, r1, r2]).transpose()

    wCrossPt0 = angleAxis[:, 1] * pt[:, 2] - angleAxis[:, 2] * pt[:, 1]
    wCrossPt1 = angleAxis[:, 2] * pt[:, 0] - angleAxis[:, 0] * pt[:, 2]
    wCrossPt2 = angleAxis[:, 0] * pt[:, 1] - angleAxis[:, 1] * pt[:, 0]

    r00 = pt[:, 0] + wCrossPt0
    r01 = pt[:, 1] + wCrossPt1
    r02 = pt[:, 2] + wCrossPt2

    res2 = np.vstack([r00, r01, r02]).transpose()

    return res1 * mask + res2 * (1 - mask)

def QuaternionRotatePoint(q, pt):
    """Rotate points by quaternions (xyzw format). Batch operation.

    q:  (N, 4) quaternion [qx, qy, qz, qw]
    pt: (N, 3) points
    Returns: (N, 3) rotated points
    """
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # t = 2 * cross(q_xyz, pt)
    tx = 2.0 * (qy * pt[:, 2] - qz * pt[:, 1])
    ty = 2.0 * (qz * pt[:, 0] - qx * pt[:, 2])
    tz = 2.0 * (qx * pt[:, 1] - qy * pt[:, 0])
    # result = pt + qw * t + cross(q_xyz, t)
    rx = pt[:, 0] + qw * tx + (qy * tz - qz * ty)
    ry = pt[:, 1] + qw * ty + (qz * tx - qx * tz)
    rz = pt[:, 2] + qw * tz + (qx * ty - qy * tx)
    return np.column_stack([rx, ry, rz])


def quat_conjugate(q):
    """Conjugate of quaternion (xyzw format): negate xyz, keep w."""
    return np.column_stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]])


def quat_rotate_pt_torch(q, pt):
    """Rotate points by quaternions (xyzw format). Torch batch operation."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    tx = 2.0 * (qy * pt[:, 2] - qz * pt[:, 1])
    ty = 2.0 * (qz * pt[:, 0] - qx * pt[:, 2])
    tz = 2.0 * (qx * pt[:, 1] - qy * pt[:, 0])
    rx = pt[:, 0] + qw * tx + (qy * tz - qz * ty)
    ry = pt[:, 1] + qw * ty + (qz * tx - qx * tz)
    rz = pt[:, 2] + qw * tz + (qx * ty - qy * tx)
    return torch.stack([rx, ry, rz], dim=-1)


def normalize_by_points_torch(points_3d_, cameras_):
    """Normalize scene: center at median, scale so 95th percentile is at 1000 units.
    Works on torch tensors. Camera params layout: [tx, ty, tz, qx, qy, qz, qw, f, k1, k2]
    """
    t = cameras_[:, 0:3]
    q = cameras_[:, 3:7]
    q_conj = q.clone()
    q_conj[:, :3] = -q_conj[:, :3]

    median = points_3d_.median(dim=0).values
    points_3d_ = points_3d_ - median

    cam_loc = -quat_rotate_pt_torch(q_conj, t)
    cam_loc = cam_loc - median

    norm = torch.linalg.norm(points_3d_, dim=1)
    scene_scale = torch.quantile(norm, 0.95)
    scale = SCENE_SCALE / scene_scale
    points_3d_ = points_3d_ * scale
    cam_loc = cam_loc * scale

    cameras_ = cameras_.clone()
    cameras_[:, 0:3] = -quat_rotate_pt_torch(q, cam_loc)
    return points_3d_, cameras_


# idea: median ste to 0, scale set to 100: let 95% fall into < 100 distance to center.
def normalize_by_points(points_3d_, cameras_):
    # Camera params layout (use_quat=True): [tx, ty, tz, qx, qy, qz, qw, f, k1, k2]
    # Camera model: P_cam = R(q) * P_world + t
    # Camera center in world: cam_loc = -R(q)^{-1} * t = -R(q_conj) * t
    t = cameras_[:, 0:3]
    q = cameras_[:, 3:7]

    # 1. get median in each direction.
    median = np.median(points_3d_, axis=0)
    points_3d_ = points_3d_ - median

    # camera center in world frame
    cam_loc = -QuaternionRotatePoint(quat_conjugate(q), t)
    cam_loc = cam_loc - median

    norm = np.linalg.norm(points_3d_, axis=1)
    scene_scale = np.percentile(norm, 95)
    scale = 1000 / scene_scale
    points_3d_ = points_3d_ * scale
    cam_loc = cam_loc * scale

    # Recover translation: t_new = -R(q) * cam_loc_new
    cameras_ = cameras_.copy()
    cameras_[:, 0:3] = -QuaternionRotatePoint(q, cam_loc)
    return points_3d_, cameras_

@psjac
def project(points, camera_params):
    projection = pp.SE3(camera_params[..., :7]).Act(points)
    projection = -projection[..., :2] / projection[..., [2]]

    f = camera_params[..., [-3]]
    k1 = camera_params[..., [-2]]
    k2 = camera_params[..., [-1]]

    n = torch.sum(projection**2, axis=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n**2
    return projection * r * f


class Residual(nn.Module):
    def __init__(self, camera_params, points):
        super().__init__()
        self.pose = pp.Parameter(camera_params, sjac=True)
        self.points = pp.Parameter(points, sjac=True)
        self.pose.trim_SE3_grad = True

    def forward(self, observes, cidx, pidx):
        points_proj = project(self.points[pidx], self.pose[cidx])
        return points_proj - observes


class PowerSchur(Schur):
    """
    Power Bundle Adjustment: replaces the PCG solve on the Schur complement
    with a power series expansion of S^{-1}.

    S = U - W V^{-1} W^T
    S^{-1} ≈ sum_{k=0}^{K} (U^{-1} W V^{-1} W^T)^k U^{-1}

    Iteration:
      Dc^{(0)} = U^{-1} * rhs_c
      Dc^{(k+1)} = U^{-1} * (rhs_c + W V^{-1} W^T * Dc^{(k)})
    """

    def __init__(self, *args, power_iters: int = 10,
                 use_jacobi_precond: bool = False, **kwargs):
        self.power_iters = power_iters
        self.use_jacobi_precond = use_jacobi_precond
        super().__init__(*args, **kwargs)

    @staticmethod
    def _extract_block_diag(bsr_pt):
        """Extract diagonal of a BSR matrix as a flat vector (one value per scalar row)."""
        diag = diagonal_op_(bsr_pt)  # (n_blocks, block_size) flattened to (n_blocks * block_size,)
        return diag

    @staticmethod
    def _compute_jacobi_scale(bsr_pt, clamp_min=1e-10):
        """Compute D = sqrt(diag(A)), clamped, for Jacobi preconditioning."""
        diag = diagonal_op_(bsr_pt).clone()
        diag = torch.sqrt(diag.abs()).clamp(min=clamp_min)
        return diag

    @staticmethod
    def _scale_bsr_diag(bsr_pt, D_inv):
        """Scale a BSR matrix in-place: A' = diag(D_inv) * A * diag(D_inv).

        D_inv is a flat vector of length n (matching total scalar rows/cols).
        """
        crow = bsr_pt.crow_indices()
        col = bsr_pt.col_indices()
        vals = bsr_pt.values()  # (nnz, bs, bs)
        bs = vals.shape[-1]

        D_inv_blocks = D_inv.view(-1, bs)  # (n_block_rows, bs)
        n_block_rows = crow.shape[0] - 1

        # Compute block row index for each nnz block
        counts = crow[1:] - crow[:-1]  # blocks per row
        row_idx = torch.repeat_interleave(torch.arange(n_block_rows, device=vals.device), counts)

        # row and col scaling: vals[k] *= D_row[k][:,None] * D_col[k][None,:]
        d_row = D_inv_blocks[row_idx]  # (nnz, bs)
        d_col = D_inv_blocks[col]       # (nnz, bs)
        vals *= d_row.unsqueeze(2) * d_col.unsqueeze(1)

    @staticmethod
    def _scale_bsr_lr(bsr_pt, D_inv_row, D_inv_col):
        """Scale a BSR matrix in-place: A' = diag(D_inv_row) * A * diag(D_inv_col).

        For non-square blocks (W), row and col scalings differ.
        """
        crow = bsr_pt.crow_indices()
        col = bsr_pt.col_indices()
        vals = bsr_pt.values()  # (nnz, bs_r, bs_c)
        bs_r, bs_c = vals.shape[-2], vals.shape[-1]
        n_block_rows = crow.shape[0] - 1

        D_row_blocks = D_inv_row.view(-1, bs_r)  # (n_block_rows, bs_r)
        D_col_blocks = D_inv_col.view(-1, bs_c)  # (n_block_cols, bs_c)

        counts = crow[1:] - crow[:-1]
        row_idx = torch.repeat_interleave(torch.arange(n_block_rows, device=vals.device), counts)

        d_row = D_row_blocks[row_idx]  # (nnz, bs_r)
        d_col = D_col_blocks[col]       # (nnz, bs_c)
        vals *= d_row.unsqueeze(2) * d_col.unsqueeze(1)

    def solvePowerIts(self, Upt, W, Wt, V_i, rhs_c, Ip, power_iters):
        """Solve the Schur complement system S*x = rhs_c via Preconditioned CG.

        S = U - W V^{-1} W^T  (implicit, applied via matvec)
        Preconditioner M^{-1} = U^{-1} (block-diagonal, cheap to apply)
        """
        U_i = torchbsr2wp(inv_op(Upt))

        scratch_pts = wp.empty_like(Ip)
        scratch_pts2 = wp.empty_like(Ip)
        tmp = wp.empty_like(rhs_c)

        def schur_matvec(x, out):
            """Compute out = S * x = U*x - W*V^{-1}*W^T*x"""
            # W^T * x
            sparse.bsr_mv(Wt, x, y=scratch_pts, beta=0.0)
            # V^{-1} * W^T * x
            sparse.bsr_mv(V_i, scratch_pts, y=scratch_pts2, beta=0.0)
            # W * V^{-1} * W^T * x
            sparse.bsr_mv(W, scratch_pts2, y=out, beta=0.0)
            # U*x - W*V^{-1}*W^T*x  (out = U*x - out)
            sparse.bsr_mv(torchbsr2wp(Upt), x, y=out, alpha=1.0, beta=-1.0)

        # Initial guess: x = 0
        x = wp.zeros_like(rhs_c)
        x_t = wp.to_torch(x)

        # r = rhs_c - S*x = rhs_c (since x=0)
        r = wp.empty_like(rhs_c)
        wp.copy(src=rhs_c, dest=r)
        r_t = wp.to_torch(r)

        # z = M^{-1} r = U^{-1} r
        z = wp.empty_like(rhs_c)
        sparse.bsr_mv(U_i, r, y=z, beta=0.0)
        z_t = wp.to_torch(z)

        # p = z
        p = wp.empty_like(rhs_c)
        wp.copy(src=z, dest=p)
        p_t = wp.to_torch(p)

        # rz = r^T z
        rz = torch.dot(r_t.view(-1), z_t.view(-1))

        # Ap buffer
        Ap = wp.empty_like(rhs_c)
        Ap_t = wp.to_torch(Ap)

        for _k in range(power_iters):
            # Ap = S * p
            schur_matvec(p, Ap)

            # alpha = rz / (p^T Ap)
            pAp = torch.dot(p_t.view(-1), Ap_t.view(-1))
            if pAp.item() <= 0:
                break  # Negative curvature, stop
            alpha = rz / pAp

            # x = x + alpha * p
            x_t.add_(p_t, alpha=alpha.item())

            # r = r - alpha * Ap
            r_t.add_(Ap_t, alpha=-alpha.item())

            # Check convergence every 10 iterations
            if _k % 10 == 9:
                r_norm = torch.linalg.norm(r_t).item()
                x_norm = torch.linalg.norm(x_t).item()
                if r_norm < EARLY_STOPPING_EPS * x_norm:
                    break

            # z = M^{-1} r = U^{-1} r
            sparse.bsr_mv(U_i, r, y=z, beta=0.0)

            # rz_new = r^T z
            rz_new = torch.dot(r_t.view(-1), z_t.view(-1))

            # beta = rz_new / rz
            beta = rz_new / rz
            rz = rz_new

            # p = z + beta * p
            p_t.mul_(beta.item()).add_(z_t)

        return x

    # Keep solveByGDNesterov as alias for compatibility
    solveByGDNesterov = solvePowerIts

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            self.reject_count = 0
            weight = self.weight if weight is None else weight
            R = self.model(input, target)[0]
            J = jacobian(R, pg['params'])

            self.last = self.loss = self.loss if hasattr(self, 'loss') else self.model.loss(input, target)

            # Build the normal equation blocks
            J0wp = torchbsr2wp(J[0])
            J1wp = torchbsr2wp(J[1])
            J0twp = sparse.bsr_transposed(J0wp)
            J1twp = sparse.bsr_transposed(J1wp)
            U = sparse.bsr_mm(J0twp, J0wp)   # camera normal: J_c^T J_c
            V = sparse.bsr_mm(J1twp, J1wp)   # point normal:  J_p^T J_p
            W = sparse.bsr_mm(J0twp, J1wp)   # cross term:    J_c^T J_p
            Wt = sparse.bsr_transposed(W)    # W^T = J_p^T J_c
            del J0twp, J1twp

            Upt = wp2torchbsr(U)
            Vpt = wp2torchbsr(V)
            diagonal_op_(Upt, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            diagonal_op_(Vpt, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))

            # --- Jacobi preconditioning of the linear system ---
            # Compute D_c = sqrt(diag(U)), D_p = sqrt(diag(V))
            # Scale: U' = D_c^{-1} U D_c^{-1}, V' = D_p^{-1} V D_p^{-1}
            #        W' = D_c^{-1} W D_p^{-1}, Ic' = D_c^{-1} Ic, Ip' = D_p^{-1} Ip
            # After solving, recover: delta = D^{-1} * delta_precond
            D_c_inv = None
            D_p_inv = None
            if self.use_jacobi_precond:
                D_c_scale = self._compute_jacobi_scale(Upt)
                D_p_scale = self._compute_jacobi_scale(Vpt)
                D_c_inv = 1.0 / D_c_scale
                D_p_inv = 1.0 / D_p_scale

                # Scale U, V (symmetric: left and right by same D_inv)
                self._scale_bsr_diag(Upt, D_c_inv)
                self._scale_bsr_diag(Vpt, D_p_inv)

                # Scale W (non-square: D_c^{-1} * W * D_p^{-1})
                # W is in warp format, need to convert, scale, convert back
                Wpt = wp2torchbsr(W)
                self._scale_bsr_lr(Wpt, D_c_inv, D_p_inv)
                W = torchbsr2wp(Wpt)
                Wt = sparse.bsr_transposed(W)

            # Right-hand side vectors
            R_flat = R.reshape(-1).contiguous()
            Rwp = format_vec_for_bsr(R_flat, (J0wp.block_shape[1], J0wp.block_shape[0]))
            Ic = sparse.bsr_mv(J0wp, Rwp, alpha=-1.0, transpose=True)  # -J_c^T r
            Ip = sparse.bsr_mv(J1wp, Rwp, alpha=-1.0, transpose=True)  # -J_p^T r

            # Scale RHS by D^{-1}
            if self.use_jacobi_precond:
                Ic_t = wp.to_torch(Ic).view(-1)
                Ip_t = wp.to_torch(Ip).view(-1)
                Ic_t.mul_(D_c_inv)
                Ip_t.mul_(D_p_inv)

            rhs_c = wp.empty_like(Ic)
            rhs_p = wp.empty_like(Ip)
            scratch_pts = wp.empty_like(Ip)

            while self.last <= self.loss:
                # Apply damping
                damp = partial(torch.mul, other=1 + pg['damping'])
                diagonal_op_(Upt, op=damp)
                diagonal_op_(Vpt, op=damp)

                # Invert V (block-diagonal)
                V_i = torchbsr2wp(inv_op(Vpt))

                # Build rhs_c = Ic - W V^{-1} Ip
                wp.copy(src=Ic, dest=rhs_c)
                sparse.bsr_mv(V_i, Ip, y=scratch_pts, beta=0.0)
                sparse.bsr_mv(W, scratch_pts, y=rhs_c, alpha=-1.0, beta=1.0)

                # Solve Schur complement via Preconditioned CG
                D_c = self.solvePowerIts(Upt, W, Wt, V_i, rhs_c, Ip, self.power_iters)

                # --- Back-substitution for points ---
                # Dp = V^{-1} (Ip - W^T Dc)
                wp.copy(src=Ip, dest=rhs_p)
                sparse.bsr_mv(Wt, D_c, y=rhs_p, alpha=-1.0, beta=1.0)
                D_p = wp.zeros_like(rhs_p)
                sparse.bsr_mv(V_i, rhs_p, y=D_p, beta=0.0)

                # Apply update — unscale if preconditioned
                D_c_t = wp.to_torch(D_c).flatten()
                D_p_t = wp.to_torch(D_p).flatten()
                if self.use_jacobi_precond:
                    D_c_t = D_c_t * D_c_inv
                    D_p_t = D_p_t * D_p_inv
                D = torch.cat([D_c_t, D_p_t])
                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                print(f"Loss: {self.loss}  Last: {self.last}  "
                      f"Reject: {self.reject_count}  Damping: {pg['damping']}")

                self.strategy.update(
                    pg,
                    last=self.last,
                    loss=self.loss,
                    J=J,
                    Jwp=[J0wp, J1wp],
                    D=[D_c_t, D_p_t],
                    R=R_flat.view(-1, 1),
                )

                if self.last < self.loss and self.reject_count < self.reject:
                    self.update_parameter(params=pg['params'], step=-D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break

        return self.loss


def least_square_error(camera_params, points, cidx, pidx, observes):
    model = Residual(camera_params, points)
    loss = model(observes, cidx, pidx)
    return torch.sum(loss**2, dim=-1).mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Run CG Schur BA on one BAL problem.")
    parser.add_argument("dataset", nargs="?", default=TARGET_DATASET)
    parser.add_argument("problem", nargs="?", default=TARGET_PROBLEM)
    parser.add_argument("--iterations", type=int, default=GLOBAL_ITERATIONS)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "bal_data")
    parser.add_argument("--base-url")
    parser.add_argument("--file-name")
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main(args):
    dataset = get_problem(args.problem, args.dataset, cache_dir=args.cache_dir, use_quat=True)
    print(f"Fetched {args.problem} from {args.dataset}")

    # dataset: can i normalize the data? median to 0, scale to 100? (95% of points within 100 units)
    # dataset["camera_params"], dataset["points_2d"] = invert_focal_distance(
    #     dataset["camera_params"].cpu().numpy(),
    #     dataset["camera_index_of_observations"].cpu().numpy(),
    #     dataset["points_2d"].cpu().numpy(),
    # )
    # dataset["points_3d"], dataset["camera_params"] = normalize_by_points(
    #     dataset["points_3d"].cpu().numpy(), dataset["camera_params"].cpu().numpy()
    # )
    # # convert back to torch
    # dataset["camera_params"] = torch.tensor(dataset["camera_params"], dtype=DTYPE)
    # dataset["points_3d"] = torch.tensor(dataset["points_3d"], dtype=DTYPE)
    # dataset["points_2d"] = torch.tensor(dataset["points_2d"], dtype=DTYPE)

    if NORMALIZE_DATASET:
        dataset["points_3d"], dataset["camera_params"] = normalize_by_points_torch(
            dataset["points_3d"], dataset["camera_params"])

    dataset = {
        key: value.to(DEVICE)
        for key, value in dataset.items()
        if isinstance(value, torch.Tensor)
    }
    input = {
        "observes": dataset["points_2d"],
        "cidx": dataset["camera_index_of_observations"],
        "pidx": dataset["point_index_of_observations"],
    }

    model = Residual(
        dataset["camera_params"][:, :NUM_CAMERA_PARAMS].clone(),
        dataset["points_3d"].clone(),
    ).to(DEVICE)

    strategy = TrustRegion(up=2.0, down=0.5**4)
    optimizer = PowerSchur(
        model,
        strategy=strategy,
        solver=None,
        reject=30,
        power_iters=CG_MAX_ITERS,
        use_jacobi_precond=USE_JACOBI_PRECOND,
    )

    print('Initial MSE loss:', least_square_error(
        model.pose, model.points,
        dataset["camera_index_of_observations"],
        dataset["point_index_of_observations"],
        dataset["points_2d"],
    ).item())

    print("Initial loss:", optimizer.model.loss(input, None).item())

    start = perf_counter()
    profiler = None
    if PROFILE:
        from pyinstrument import Profiler
        profiler = Profiler(interval=0.1)
        profiler.start()

    costs = []
    trajectory_file = None
    if args.trajectory is not None:
        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
        trajectory_file = args.trajectory.open("w")

    for idx in range(args.iterations):
        loss = optimizer.step(input)
        cost = loss.item()
        elapsed = perf_counter() - start
        costs.append(cost)
        print(f"Iteration {idx}  loss {cost:.6f}  time {elapsed:.3f}s")
        if trajectory_file is not None:
            trajectory_file.write(json.dumps({"iteration": idx, "cost": cost, "elapsedSeconds": elapsed}) + "\n")
            trajectory_file.flush()

    if trajectory_file is not None:
        trajectory_file.close()

    torch.cuda.synchronize()
    end = perf_counter()

    if profiler is not None:
        profiler.stop()
        profiler.print()

    print(f"\nTotal time: {end - start:.3f}s")
    print(f"CG max iters: {CG_MAX_ITERS}, eps: {EARLY_STOPPING_EPS}, "
            f"LM iters: {args.iterations}, solver: PCG, "
          f"jacobi: {USE_JACOBI_PRECOND}, normalize: {NORMALIZE_DATASET}")

    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        print(f"Peak CUDA memory: {peak / 1024**2:.1f} MiB")

    print('Ending MSE loss:', least_square_error(
        model.pose,
        model.points,
        dataset["camera_index_of_observations"],
        dataset["point_index_of_observations"],
        dataset["points_2d"],
    ).item())

    if args.summary is not None:
        best_cost = min(costs)
        summary = {
            "algorithm": "schur_pcg_cg",
            "base_url": args.base_url or f"http://grail.cs.washington.edu/projects/bal/data/{args.dataset}/",
            "file_name": args.file_name or f"{args.problem}.txt.bz2",
            "iterations": args.iterations,
            "bestCost": best_cost,
            "bestIt": costs.index(best_cost),
            "bestCost30": min(costs[:30]),
            "bestCost60": min(costs[:60]) if len(costs) >= 60 else None,
            "status": "completed",
            "epochsCompleted": len(costs),
            "linearSolver": "pcg",
            "globalJacobi": "full" if USE_JACOBI_PRECOND else "none",
            "trajectory": str(args.trajectory.resolve()) if args.trajectory is not None else None,
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("a") as summary_file:
            summary_file.write(json.dumps(summary, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main(parse_args())
