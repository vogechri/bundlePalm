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
# [EIG] distribution (200 Lanczos eigs): p0=0.0000, p1=0.0013, p5=0.0072, p10=0.0290, p25=0.1514, p50=0.5012, p75=0.8516, p90=0.9743, p95=0.9935, p99=0.9997, p100=1.0000
# TARGET_DATASET = "trafalgar"
# TARGET_PROBLEM = "problem-257-65132-pre"
# Total time: 5.333s
# Power iterations per solve: 400, SCENE_SCALE = 10.0, EARLY_STOPPING_EPS = 1e-2
# Peak CUDA memory: 289.2 MiB
# Ending MSE loss: 0.8544272616672501

# Total time: 2.547s
# Power iterations per solve: 400, EARLY_STOPPING_EPS = 1e-3, no pcg, no normalize
# Peak CUDA memory: 289.2 MiB
# Ending MSE loss: 0.8573146649285095

# Total time: 4.979s
# Power iterations per solve: 400, no nesterov, pcg, 1e-3
# Peak CUDA memory: 289.2 MiB
# Ending MSE loss: 0.8765088943104147

###########
# TARGET_DATASET = "ladybug"
# TARGET_PROBLEM = "problem-1723-156502-pre"
# Total time: 5.134s
# Power iterations per solve: 400, SCENE_SCALE = 10.0, EARLY_STOPPING_EPS = 1e-2
# Peak CUDA memory: 830.9 MiB
# Ending MSE loss: 1.1097130100420383

# Total time: 4.564s
# Power iterations per solve: 400, EARLY_STOPPING_EPS = 1e-3, no pcg, no normalize
# Peak CUDA memory: 830.9 MiB
# Ending MSE loss: 1.1097342513398254

# Total time: 5.256s
# Power iterations per solve: 400, no nesterov, pcg, 1e-3
# Peak CUDA memory: 830.9 MiB
# Ending MSE loss: 1.1347273589007938

# TARGET_DATASET = "ladybug"
# TARGET_PROBLEM = "problem-1266-132593-pre"
# Total time: 3.903s
# Power iterations: 400, eps: 0.01, iters: 20, nesterov: True, jacobi: True, normalize: False
# Peak CUDA memory: 720.9 MiB
# Ending MSE loss: 1.125553305786994

###########
# TARGET_DATASET = "dubrovnik"
# TARGET_PROBLEM = "problem-356-226730-pre"
# Total time: 10.652s
# Power iterations per solve: 400, SCENE_SCALE = 10.0, EARLY_STOPPING_EPS = 1e-2, pcg
# Peak CUDA memory: 1518.6 MiB
# Ending MSE loss: 0.7911538386346951

# Total time: 8.246s
# Power iterations per solve: 400, EARLY_STOPPING_EPS = 1e-3, no pcg, no normalize
# Peak CUDA memory: 1518.6 MiB
# Ending MSE loss: 0.79154366543767

# Total time: 14.637s
# Power iterations per solve: 400, no nesterov, pcg, 1e-3
# Peak CUDA memory: 1518.6 MiB
# Ending MSE loss: 0.8099138904383044
###########

# {"base_url": "http://grail.cs.washington.edu/projects/bal/data/venice/", "file_name": "problem-951-708276-pre.txt.bz2", "iterations": 90, "bestCost": 3192479, "bestIt": 89, "kClusters": 30, "bestCost60": 3200260, "bestCost30": 3265533},
# TARGET_DATASET = "venice"
# TARGET_PROBLEM = "problem-951-708276-pre"
# Total time: 11.978s
# Power iterations: 400, eps: 0.001, iters: 20, nesterov: True, jacobi: False, normalize: True
# Peak CUDA memory: 4487.6 MiB
# Ending MSE loss: 0.8931065819782235
# Total time: 48.067s!!!
# Power iterations: 400, eps: 0.001, iters: 20, nesterov: True, jacobi: True, normalize: False
# Peak CUDA memory: 4487.6 MiB
# Ending MSE loss: 0.8918893259895316

# {"base_url": "http://grail.cs.washington.edu/projects/bal/data/final/", "file_name": "problem-871-527480-pre.txt.bz2", "iterations": 90, "bestCost": 3493829, "bestIt": 89, "kClusters": 30, "bestCost60": 3509618, "bestCost30": 3564338}
TARGET_DATASET = "final"
TARGET_PROBLEM = "problem-871-527480-pre"
# Total time: 9.299s
# Power iterations: 400, eps: 0.001, iters: 20, nesterov: True, jacobi: False, normalize: True
# Peak CUDA memory: 3342.2 MiB
# Ending MSE loss: 1.2664379589341337

# Total time: 14.206s
# Power iterations: 400, eps: 0.01, iters: 20, nesterov: True, jacobi: True, normalize: False
# Peak CUDA memory: 3342.2 MiB
# Ending MSE loss: 1.2409794067200923

# Total time: 42.612s
# Power iterations: 400, eps: 0.001, iters: 60, nesterov: True, jacobi: False, normalize: True
# Peak CUDA memory: 3342.2 MiB
# Ending MSE loss: 1.2409259861489468, loss 3457191.256113

# Total time: 80.476s -> better not recompute all the time .. or why so much slower?
# Power iterations: 400, eps: 0.001, iters: 60, nesterov: True, jacobi: True, normalize: False
# Peak CUDA memory: 3342.2 MiB
# Ending MSE loss: 1.2394141319394207
# Total time: 38.513s -> lower stopping criterion to 1e-2:better & faster
# Power iterations: 400, eps: 0.01, iters: 60, nesterov: True, jacobi: True, normalize: False
# Peak CUDA memory: 3342.2 MiB
# Ending MSE loss: 1.2388764522965545

# too large! for 16 GB GPU!
# TARGET_DATASET = "final"
# TARGET_PROBLEM = "problem-4585-1324582-pre"

# TARGET_DATASET = "final"
# TARGET_PROBLEM = "problem-1936-649673-pre"
# Total time: 18.714s
# Power iterations: 400, eps: 0.01, iters: 20, nesterov: True, jacobi: True, normalize: False
# Peak CUDA memory: 6219.6 MiB
# Ending MSE loss: 1.7840457403144379
# Now jacobi & smaller step gains -- same with Normalize
# Total time: 27.534s
# Power iterations: 400, eps: 0.001, iters: 20, nesterov: True, jacobi: False, normalize: False
# Peak CUDA memory: 6219.6 MiB
# Ending MSE loss: 1.7840463467180994

# super slow wo. nesterov
# Total time: 68.137s
# Power iterations: 400, eps: 0.01, iters: 20, nesterov: False, jacobi: True, normalize: False
# Peak CUDA memory: 6219.6 MiB
# Ending MSE loss: 1.7840727652984674

# this is so tough : BUT no prec and no normalize -> loss reaches much lower levels
TARGET_DATASET = "venice"
TARGET_PROBLEM = "problem-52-64053-pre"

TARGET_DATASET = "ladybug"
TARGET_PROBLEM = "problem-1064-113655-pre"

TARGET_DATASET = "venice"
TARGET_PROBLEM = "problem-245-198739-pre"
# 1870055

TARGET_DATASET = "dubrovnik"
TARGET_PROBLEM = "problem-356-226730-pre"
TARGET_PROBLEM = "problem-173-111908-pre"

#######################################
DEVICE = "cuda"
OPTIMIZE_INTRINSICS = True
NUM_CAMERA_PARAMS = 10 if OPTIMIZE_INTRINSICS else 7
USE_NESTEROV = True  # Use Nesterov acceleration in the power series solver
POWER_ITERATIONS = 400  # Number of power series terms (K)
EARLY_STOPPING_EPS = 1e-2  # Convergence threshold for inner solver
GLOBAL_ITERATIONS = 20  # Number of LM iterations
USE_JACOBI_PRECOND = False #True # Precondition the linear system (not the parameters), we might want to lower stopping criterion if on, e.g. 1e-2
NORMALIZE_DATASET = False
SCENE_SCALE = 100.0  # Scale the scene so that 95% of points are within this distance from the median
PROFILE = False  # Enable pyinstrument profiling
LOG_EIGENVALUES = False  # Enable eigenvalue logging (expensive, for analysis only)
USE_DEFLATED = False  # Use deflated solver (Lanczos + Nesterov on easy subspace)

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

    def __init__(self, *args, power_iters: int = 10, use_nesterov: bool = False,
                 use_jacobi_precond: bool = False, log_eigenvalues: bool = False,
                 use_deflated: bool = False, **kwargs):
        self.power_iters = power_iters
        self.use_nesterov = use_nesterov
        self.use_jacobi_precond = use_jacobi_precond
        self.log_eigenvalues = log_eigenvalues
        self.use_deflated = use_deflated
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
        """Solve the Schur complement system via power series expansion.

        xk = U^{-1} rhs_c;  g = xk
        for k:
            g = U^{-1} (W V^{-1} W^T g)
            xk = xk + g
        """
        U_i = torchbsr2wp(inv_op(Upt))

        # xk = U^{-1} rhs_c
        xk = wp.zeros_like(rhs_c)
        sparse.bsr_mv(U_i, rhs_c, y=xk, beta=0.0)

        # g = xk (copy)
        g = wp.empty_like(rhs_c)
        wp.copy(src=xk, dest=g)

        scratch_pts = wp.empty_like(Ip)
        scratch_pts2 = wp.empty_like(Ip)
        tmp = wp.empty_like(rhs_c)

        for _k in range(power_iters):
            # g = U^{-1} (W (V^{-1} (W^T g)))
            sparse.bsr_mv(Wt, g, y=scratch_pts, beta=0.0)              # W^T g
            sparse.bsr_mv(V_i, scratch_pts, y=scratch_pts2, beta=0.0)  # V^{-1} W^T g
            sparse.bsr_mv(W, scratch_pts2, y=tmp, beta=0.0)            # W V^{-1} W^T g
            sparse.bsr_mv(U_i, tmp, y=g, beta=0.0)                     # U^{-1} (...)

            # xk = xk + g
            xk_t = wp.to_torch(xk)
            g_t = wp.to_torch(g)
            xk_t.add_(g_t)

            # Early stopping: (k+1) * ||g|| < eps * ||xk||
            # Only check every 10 iterations to avoid GPU sync overhead
            if _k % 10 == 9:
                if (_k + 1) * torch.linalg.norm(g_t).item() < EARLY_STOPPING_EPS * torch.linalg.norm(xk_t).item():
                    break

        return xk

    def solveByGDNesterov(self, Upt, W, Wt, V_i, rhs_c, Ip, power_iters):
        """Solve the Schur complement via Nesterov-accelerated gradient descent.

        Solves (I - U^{-1} W V^{-1} W^T) x = U^{-1} rhs_c  using FISTA-style
        momentum with fixed Lipschitz estimate.
        """
        Lip = 0.9
        lambda0 = (1.0 + math.sqrt(5.0)) / 2.0

        U_i = torchbsr2wp(inv_op(Upt))

        # ubs = -U^{-1} rhs_c
        ubs = wp.zeros_like(rhs_c)
        sparse.bsr_mv(U_i, rhs_c, y=ubs, beta=0.0)
        ubs_t = wp.to_torch(ubs)
        ubs_t.neg_()

        # xk = -ubs = U^{-1} rhs_c
        xk = wp.empty_like(rhs_c)
        xk_t = wp.to_torch(xk)
        xk_t.copy_(ubs_t).neg_()

        # y0 = xk
        y0 = wp.empty_like(rhs_c)
        y0_t = wp.to_torch(y0)
        y0_t.copy_(xk_t)

        scratch_pts = wp.empty_like(Ip)
        scratch_pts2 = wp.empty_like(Ip)
        tmp = wp.empty_like(rhs_c)
        g = wp.empty_like(rhs_c)
        yk_t = wp.to_torch(wp.empty_like(rhs_c))

        inv_Lip = 1.0 / Lip

        for _k in range(power_iters):
            lambda1 = (1.0 + math.sqrt(1.0 + 4.0 * lambda0**2)) / 2.0
            gamma = (1.0 - lambda0) / lambda1
            lambda0 = lambda1

            # g = xk - U^{-1}*(W*(V^{-1}*(W^T * xk))) + ubs
            sparse.bsr_mv(Wt, xk, y=scratch_pts, beta=0.0)              # W^T xk
            sparse.bsr_mv(V_i, scratch_pts, y=scratch_pts2, beta=0.0)   # V^{-1} W^T xk
            sparse.bsr_mv(W, scratch_pts2, y=tmp, beta=0.0)             # W V^{-1} W^T xk
            sparse.bsr_mv(U_i, tmp, y=g, beta=0.0)                      # U^{-1} (...)
            g_t = wp.to_torch(g)
            g_t.copy_(xk_t - g_t + ubs_t)

            # yk = xk - (1/Lip) * g
            yk_t.copy_(xk_t - inv_Lip * g_t)

            # xk = (1-gamma) * yk + gamma * y0
            xk_t.copy_((1.0 - gamma) * yk_t + gamma * y0_t)

            # y0 = yk
            y0_t.copy_(yk_t)

            # Early stopping: (k+1) * ||(1/Lip)*g|| < eps * ||xk||
            # Only check every 10 iterations to avoid GPU sync overhead
            if _k % 10 == 9:
                if (_k + 1) * inv_Lip * torch.linalg.norm(g_t).item() < EARLY_STOPPING_EPS * torch.linalg.norm(xk_t).item():
                    break

        return xk

    def solveDeflated(self, Upt, W, Wt, V_i, rhs_c, Ip, power_iters):
        """Solve the Schur complement via deflated Nesterov.

        1. Run Lanczos to find top-k eigenvectors of M = U^{-1} W V^{-1} W^T
        2. Project the hard subspace out and solve it exactly (small dense system)
        3. Solve the deflated (easy) system with Nesterov in few iterations

        The system is: S x = b, where S = U - W V^{-1} W^T, equivalently
        (I - M) x = U^{-1} b, where M = U^{-1} W V^{-1} W^T.
        """
        N_DEFLATE = 25  # number of eigenvectors to deflate
        U_i = torchbsr2wp(inv_op(Upt))

        scratch_pts = wp.empty_like(Ip)
        scratch_pts2 = wp.empty_like(Ip)
        tmp = wp.empty_like(rhs_c)

        def apply_M_wp(x_wp):
            """Apply M = U^{-1} W V^{-1} W^T to a warp vector."""
            out = wp.empty_like(x_wp)
            sparse.bsr_mv(Wt, x_wp, y=scratch_pts, beta=0.0)
            sparse.bsr_mv(V_i, scratch_pts, y=scratch_pts2, beta=0.0)
            sparse.bsr_mv(W, scratch_pts2, y=tmp, beta=0.0)
            sparse.bsr_mv(U_i, tmp, y=out, beta=0.0)
            return out

        def apply_M_torch(x_t):
            """Apply M to a torch tensor, return torch tensor."""
            x_wp = wp.from_torch(x_t.contiguous())
            out_wp = apply_M_wp(x_wp)
            return wp.to_torch(out_wp).view(-1)

        n = wp.to_torch(rhs_c).numel()

        # --- Step 1: Lanczos to get top eigenvectors of M ---
        n_lanczos = min(N_DEFLATE + 10, power_iters, n)
        v_t = torch.randn(n, dtype=torch.float64, device='cuda')
        v_t /= torch.linalg.norm(v_t)

        # Store Lanczos vectors for eigenvector recovery
        Q = torch.zeros(n_lanczos, n, dtype=torch.float64, device='cuda')
        alphas = []
        betas_list = [0.0]
        v_prev_t = torch.zeros_like(v_t)

        for j in range(n_lanczos):
            Q[j] = v_t
            w_t = apply_M_torch(v_t)

            alpha_j = torch.dot(v_t, w_t).item()
            alphas.append(alpha_j)

            w_t -= alpha_j * v_t + betas_list[-1] * v_prev_t

            beta_jp1 = torch.linalg.norm(w_t).item()
            if beta_jp1 < 1e-14:
                n_lanczos = j + 1
                break
            betas_list.append(beta_jp1)

            v_prev_t = v_t.clone()
            v_t = w_t / beta_jp1

        # Build tridiagonal and get eigenpairs
        k = len(alphas)
        T = torch.zeros(k, k, dtype=torch.float64, device='cuda')
        for i in range(k):
            T[i, i] = alphas[i]
            if i > 0:
                T[i, i-1] = betas_list[i]
                T[i-1, i] = betas_list[i]

        eig_vals, eig_vecs_T = torch.linalg.eigh(T)  # eigenvalues sorted ascending

        # Take the top-k (largest) eigenvalues/vectors
        n_deflate = min(N_DEFLATE, k)
        top_indices = torch.arange(k - n_deflate, k, device='cuda')
        lam_top = eig_vals[top_indices]  # (n_deflate,) eigenvalues of M
        # Clamp eigenvalues to (0, 1-eps) to avoid division by zero
        lam_top = lam_top.clamp(min=0.0, max=1.0 - 1e-8)
        # Ritz vectors: V_k @ s_i (eigenvectors in original space)
        Z = eig_vecs_T[:, top_indices]  # (k, n_deflate) Lanczos eigvecs
        # P = Q^T Z gives the deflation basis in original space: (n, n_deflate)
        P = (Q[:k].T @ Z)  # (n, n_deflate)
        # Orthonormalize P (Lanczos can lose orthogonality)
        P, _ = torch.linalg.qr(P)  # (n, n_deflate)

        # --- Step 2: Solve the hard subspace exactly ---
        # In the deflation subspace, (I - M) restricted to P gives:
        # (I - diag(lam_top)) * coeffs = P^T * U^{-1} b
        rhs_t = wp.to_torch(rhs_c).view(-1).clone()
        # Compute U^{-1} b
        Uinv_b_wp = wp.zeros_like(rhs_c)
        sparse.bsr_mv(U_i, rhs_c, y=Uinv_b_wp, beta=0.0)
        Uinv_b = wp.to_torch(Uinv_b_wp).view(-1)

        # Project RHS onto deflation space
        rhs_proj = P.T @ Uinv_b  # (n_deflate,)
        # Solve (I - Λ) c = rhs_proj → c = rhs_proj / (1 - λ_i)
        coeffs = rhs_proj / (1.0 - lam_top)
        # Hard solution component
        x_hard = P @ coeffs  # (n,)

        # --- Step 3: Solve the deflated (easy) system with Nesterov ---
        # Deflated RHS: U^{-1} b - (I-M)^{-1}_hard contribution projected out
        # Actually: solve (I-M) x_easy = U^{-1}b - (I-M) x_hard in complement of P
        # Equivalently: solve with deflated operator where top eigenvalues are removed

        # Deflated RHS: subtract the hard component from Uinv_b
        rhs_deflated = Uinv_b - x_hard + apply_M_torch(x_hard)  # = U^{-1}b - (I-M)x_hard... no
        # Actually simpler: full solution = x_hard + x_easy
        # where x_easy solves (I-M_deflated) x_easy = (I - P P^T)(U^{-1}b)
        # with M_deflated having spectral radius ≈ p75 = 0.85

        # Deflated RHS = project out the hard subspace from U^{-1}b
        rhs_easy = Uinv_b - P @ (P.T @ Uinv_b)  # orthogonal complement

        # Run Nesterov on deflated system (same operator M, but starting from
        # projected initial guess — the hard directions won't grow)
        Lip_deflated = 0.99  # effective max eigenvalue after deflation ~0.97
        lambda0 = (1.0 + math.sqrt(5.0)) / 2.0

        xk_t = rhs_easy.clone()  # initial guess = U^{-1} projected rhs
        y0_t = xk_t.clone()
        inv_Lip = 1.0 / Lip_deflated

        # Fewer iterations needed — deflated spectrum is much better conditioned
        max_deflated_iters = min(power_iters, 50)

        for _k in range(max_deflated_iters):
            lambda1 = (1.0 + math.sqrt(1.0 + 4.0 * lambda0**2)) / 2.0
            gamma = (1.0 - lambda0) / lambda1
            lambda0 = lambda1

            # g = xk - M*xk - U^{-1}b (but in deflated space)
            Mxk = apply_M_torch(xk_t)
            # Project out hard directions from Mxk to stay in deflated subspace
            Mxk -= P @ (P.T @ Mxk)
            g_t = xk_t - Mxk - rhs_easy

            yk_t = xk_t - inv_Lip * g_t
            xk_new = (1.0 - gamma) * yk_t + gamma * y0_t
            y0_t = yk_t
            xk_t = xk_new

            # Early stopping
            if _k % 10 == 9:
                if (_k + 1) * torch.linalg.norm(inv_Lip * g_t).item() < EARLY_STOPPING_EPS * torch.linalg.norm(xk_t).item():
                    break

        # --- Combine: x = x_hard + x_easy ---
        x_full = x_hard + xk_t

        # Fallback: if NaN, use plain Nesterov
        if torch.isnan(x_full).any():
            print("  [DEFLATED] NaN detected, falling back to Nesterov")
            return self.solveByGDNesterov(Upt, W, Wt, V_i, rhs_c, Ip, power_iters)

        # Convert to warp — match the dtype/shape of rhs_c
        rhs_t = wp.to_torch(rhs_c)
        rhs_t.view(-1).copy_(x_full)
        return rhs_c

    def solveByGDNesterov_logging(self, Upt, W, Wt, V_i, rhs_c, Ip, power_iters):
        """Like solveByGDNesterov but estimates min/max eigenvalues of
        M = U^{-1/2} W V^{-1} W^T U^{-1/2} (symmetric form, same eigenvalues as
        U^{-1} W V^{-1} W^T) via Lanczos.

        Stores results in self._eigenvalue_log (list of dicts).
        Expensive — use only for analysis.
        """
        # Compute symmetric U^{-1/2} per block via eigendecomposition
        # U = Q Λ Q^T → U^{-1/2} = Q Λ^{-1/2} Q^T
        U_vals = Upt.values()  # (n_blocks, bs, bs)
        eigvals, eigvecs = torch.linalg.eigh(U_vals)  # symmetric eigendecomposition
        eigvals_inv_sqrt = 1.0 / torch.sqrt(eigvals.clamp(min=1e-12))
        U_inv_half_vals = eigvecs * eigvals_inv_sqrt.unsqueeze(-2)  # Q * diag(λ^{-1/2})
        U_inv_half_vals = U_inv_half_vals @ eigvecs.transpose(-1, -2)  # Q Λ^{-1/2} Q^T

        # Build U^{-1/2} as BSR
        U_inv_half_pt = torch.sparse_bsr_tensor(
            crow_indices=Upt.crow_indices(),
            col_indices=Upt.col_indices(),
            values=U_inv_half_vals,
            size=Upt.shape, dtype=Upt.dtype, device=Upt.device,
        )
        U_inv_half = torchbsr2wp(U_inv_half_pt)

        scratch_pts = wp.empty_like(Ip)
        scratch_pts2 = wp.empty_like(Ip)
        tmp = wp.empty_like(rhs_c)
        tmp2 = wp.empty_like(rhs_c)

        def apply_M_sym(x_wp):
            """Apply M_sym = U^{-1/2} W V^{-1} W^T U^{-1/2} x (symmetric)."""
            out = wp.empty_like(x_wp)
            # U^{-1/2} x
            sparse.bsr_mv(U_inv_half, x_wp, y=tmp, beta=0.0)
            # W^T (U^{-1/2} x)
            sparse.bsr_mv(Wt, tmp, y=scratch_pts, beta=0.0)
            # V^{-1} W^T U^{-1/2} x
            sparse.bsr_mv(V_i, scratch_pts, y=scratch_pts2, beta=0.0)
            # W V^{-1} W^T U^{-1/2} x
            sparse.bsr_mv(W, scratch_pts2, y=tmp2, beta=0.0)
            # U^{-1/2} W V^{-1} W^T U^{-1/2} x
            sparse.bsr_mv(U_inv_half, tmp2, y=out, beta=0.0)
            return out

        # --- Estimate eigenvalues via Lanczos (symmetric operator) ---
        n_lanczos = min(200, power_iters)
        n = wp.to_torch(rhs_c).numel()

        v_t = torch.randn(n, dtype=torch.float64, device='cuda')
        v_t /= torch.linalg.norm(v_t)

        alphas = []
        betas = [0.0]
        v_prev_t = torch.zeros_like(v_t)

        for j in range(n_lanczos):
            v_wp = wp.from_torch(v_t.contiguous())
            w_wp = apply_M_sym(v_wp)
            w_t = wp.to_torch(w_wp).view(-1)

            alpha_j = torch.dot(v_t, w_t).item()
            alphas.append(alpha_j)

            w_t -= alpha_j * v_t + betas[-1] * v_prev_t

            beta_jp1 = torch.linalg.norm(w_t).item()
            if beta_jp1 < 1e-14:
                break
            betas.append(beta_jp1)

            v_prev_t = v_t.clone()
            v_t = w_t / beta_jp1

        # Build tridiagonal matrix and compute eigenvalues
        k = len(alphas)
        T = torch.zeros(k, k, dtype=torch.float64)
        for i in range(k):
            T[i, i] = alphas[i]
            if i > 0:
                T[i, i-1] = betas[i]
                T[i-1, i] = betas[i]

        eigs = torch.linalg.eigvalsh(T)
        lambda_min = eigs[0].item()
        lambda_max = eigs[-1].item()
        spectral_radius = max(abs(lambda_min), abs(lambda_max))
        condition = lambda_max / max(lambda_min, 1e-15) if lambda_min > 0 else float('inf')

        if not hasattr(self, '_eigenvalue_log'):
            self._eigenvalue_log = []
        self._eigenvalue_log.append({
            'lambda_min': lambda_min,
            'lambda_max': lambda_max,
            'spectral_radius': spectral_radius,
            'condition_number': condition,
            'lanczos_steps': k,
            'eigs_sample': eigs.cpu().tolist(),
        })

        print(f"  [EIG] M_sym = U^{{-1/2}}WV^{{-1}}W^TU^{{-1/2}}: "
              f"λ_min={lambda_min:.6e}, λ_max={lambda_max:.6e}, "
              f"ρ={spectral_radius:.6f}, κ={condition:.1f}")

        # Detailed eigenvalue distribution
        eigs_np = eigs.cpu().numpy()
        percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        pct_vals = np.percentile(eigs_np, percentiles)
        pct_str = ", ".join([f"p{p}={v:.4f}" for p, v in zip(percentiles, pct_vals)])
        print(f"  [EIG] distribution ({k} Lanczos eigs): {pct_str}")
        # How many are > 0.9, > 0.99, > 0.999?
        n_above_09 = (eigs_np > 0.9).sum()
        n_above_099 = (eigs_np > 0.99).sum()
        n_above_0999 = (eigs_np > 0.999).sum()
        print(f"  [EIG] >0.9: {n_above_09}/{k}, >0.99: {n_above_099}/{k}, >0.999: {n_above_0999}/{k}")

        # Now solve normally with Nesterov
        return self.solveByGDNesterov(Upt, W, Wt, V_i, rhs_c, Ip, power_iters)

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

                # Solve Schur complement
                if self.log_eigenvalues:
                    solver_fn = self.solveByGDNesterov_logging
                elif self.use_deflated:
                    solver_fn = self.solveDeflated
                elif self.use_nesterov:
                    solver_fn = self.solveByGDNesterov
                else:
                    solver_fn = self.solvePowerIts
                D_c = solver_fn(Upt, W, Wt, V_i, rhs_c, Ip, self.power_iters)

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
    parser = argparse.ArgumentParser(description="Run Nesterov-accelerated Schur BA on one BAL problem.")
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
        power_iters=POWER_ITERATIONS,
        use_nesterov=USE_NESTEROV,
        use_jacobi_precond=USE_JACOBI_PRECOND,
        log_eigenvalues=LOG_EIGENVALUES,
        use_deflated=USE_DEFLATED,
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
    print(f"Power iterations: {POWER_ITERATIONS}, eps: {EARLY_STOPPING_EPS}, "
            f"iters: {args.iterations}, nesterov: {USE_NESTEROV}, "
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
            "algorithm": "schur_pcg_nesterov",
            "base_url": args.base_url or f"http://grail.cs.washington.edu/projects/bal/data/{args.dataset}/",
            "file_name": args.file_name or f"{args.problem}.txt.bz2",
            "iterations": args.iterations,
            "bestCost": best_cost,
            "bestIt": costs.index(best_cost),
            "bestCost30": min(costs[:30]),
            "bestCost60": min(costs[:60]) if len(costs) >= 60 else None,
            "status": "completed",
            "epochsCompleted": len(costs),
            "accelerator": "nesterov",
            "trajectory": str(args.trajectory.resolve()) if args.trajectory is not None else None,
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("a") as summary_file:
            summary_file.write(json.dumps(summary, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main(parse_args())
