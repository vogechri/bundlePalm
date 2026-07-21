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


TARGET_DATASET = "trafalgar"
TARGET_PROBLEM = "problem-257-65132-pre"
# Total time: 3.273s
# Ending MSE loss: 0.8582973378453572
# Total time: 3.264s
# Power iterations per solve: 400, EARLY_STOPPING_EPS = 1e-3, SCENE_SCALE = 10.0
# Peak CUDA memory: 289.2 MiB
# Ending MSE loss: 0.8569803905513893

# other options:
# TARGET_DATASET = "ladybug"
# TARGET_PROBLEM = "problem-1723-156502-pre"
# Total time: 2.012s
# Ending MSE loss: 1.1266520056504714
# Peak CUDA memory: 830.9 MiB
# Total time: 4.624s
# Power iterations per solve: 400, EARLY_STOPPING_EPS = 1e-3, SCENE_SCALE = 10.0
# Peak CUDA memory: 830.9 MiB
# Ending MSE loss: 1.109734251318925

TARGET_DATASET = "dubrovnik"
TARGET_PROBLEM = "problem-356-226730-pre"
# Total time: 15.599s
# Power iterations per solve: 1000
# Peak CUDA memory: 1518.6 MiB
# Ending MSE loss: 0.7910905885774738
# Total time: 10.056s
# Power iterations per solve: 400, EARLY_STOPPING_EPS = 1e-3, SCENE_SCALE = 10.0
# Peak CUDA memory: 1518.6 MiB
# Ending MSE loss: 0.7872517499582448

DEVICE = "cuda"
OPTIMIZE_INTRINSICS = True
NUM_CAMERA_PARAMS = 10 if OPTIMIZE_INTRINSICS else 7
POWER_ITERATIONS = 400  # Number of power series terms (K)
EARLY_STOPPING_EPS = 1e-3  # Convergence threshold for inner solver
GLOBAL_ITERATIONS = 20  # Number of LM iterations
SCENE_SCALE = 10.0  # Scale the scene so that 95% of points are within this distance from the median
NORMALIZE_DATASET = True  # False = identity (disabled), True = normalize_by_points_torch (center at median, scale to SCENE_SCALE)

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

    def __init__(self, *args, power_iters: int = 10, use_nesterov: bool = False, **kwargs):
        self.power_iters = power_iters
        self.use_nesterov = use_nesterov
        super().__init__(*args, **kwargs)

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
        yk = wp.empty_like(rhs_c)

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
            yk_t = wp.to_torch(yk)
            yk_t.copy_(xk_t - inv_Lip * g_t)

            # xk = (1-gamma) * yk + gamma * y0
            xk_t.copy_((1.0 - gamma) * yk_t + gamma * y0_t)

            # y0 = yk
            y0_t.copy_(yk_t)

            # Early stopping: (k+1) * ||(1/Lip)*g|| < eps * ||xk||
            if _k % 10 == 9:
                if (_k + 1) * torch.linalg.norm(inv_Lip * g_t).item() < EARLY_STOPPING_EPS * torch.linalg.norm(xk_t).item():
                    break

        return xk

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

            # Right-hand side vectors
            R_flat = R.reshape(-1).contiguous()
            Rwp = format_vec_for_bsr(R_flat, (J0wp.block_shape[1], J0wp.block_shape[0]))
            Ic = sparse.bsr_mv(J0wp, Rwp, alpha=-1.0, transpose=True)  # -J_c^T r
            Ip = sparse.bsr_mv(J1wp, Rwp, alpha=-1.0, transpose=True)  # -J_p^T r
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
                solver_fn = self.solveByGDNesterov if self.use_nesterov else self.solvePowerIts
                D_c = solver_fn(Upt, W, Wt, V_i, rhs_c, Ip, self.power_iters)

                # --- Back-substitution for points ---
                # Dp = V^{-1} (Ip - W^T Dc)
                wp.copy(src=Ip, dest=rhs_p)
                sparse.bsr_mv(Wt, D_c, y=rhs_p, alpha=-1.0, beta=1.0)
                D_p = wp.zeros_like(rhs_p)
                sparse.bsr_mv(V_i, rhs_p, y=D_p, beta=0.0)

                # Apply update
                D_c_t = wp.to_torch(D_c).flatten()
                D_p_t = wp.to_torch(D_p).flatten()
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


def main():
    dataset = get_problem(TARGET_PROBLEM, TARGET_DATASET, use_quat=True)
    print(f"Fetched {TARGET_PROBLEM} from {TARGET_DATASET}")

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
        use_nesterov=True,  # False = power series, True = Nesterov-accelerated
    )

    print('Initial MSE loss:', least_square_error(
        model.pose, model.points,
        dataset["camera_index_of_observations"],
        dataset["point_index_of_observations"],
        dataset["points_2d"],
    ).item())

    print("Initial loss:", optimizer.model.loss(input, None).item())

    start = perf_counter()
    for idx in range(GLOBAL_ITERATIONS):
        loss = optimizer.step(input)
        print(f"Iteration {idx}  loss {loss.item():.6f}  time {perf_counter() - start:.3f}s")

    torch.cuda.synchronize()
    end = perf_counter()
    print(f"\nTotal time: {end - start:.3f}s")
    print(f"Power iterations per solve: {POWER_ITERATIONS}")

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


if __name__ == "__main__":
    main()
