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
from time import perf_counter
from datetime import datetime
from pathlib import Path
from functools import partial
import inspect
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from datapipes.bal_loader import get_problem
from bae.optim import LM
from bae.optim.strategy import TrustRegion
from warp import sparse
from warp.optim import linear


TARGET_DATASET = "trafalgar"
TARGET_PROBLEM = "problem-257-65132-pre"
# other options:
# TARGET_DATASET = "ladybug"
# TARGET_PROBLEM = "problem-1723-156502-pre"
# TARGET_DATASET = "dubrovnik"
# TARGET_PROBLEM = "problem-356-226730-pre"

DEVICE = "cuda"
OPTIMIZE_INTRINSICS = True
NUM_CAMERA_PARAMS = 10 if OPTIMIZE_INTRINSICS else 7
POWER_ITERATIONS = 100  # Number of power series terms (K)


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

    def __init__(self, *args, power_iters: int = 10, **kwargs):
        self.power_iters = power_iters
        super().__init__(*args, **kwargs)

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
            scratch_pts2 = wp.empty_like(Ip)

            while self.last <= self.loss:
                # Apply damping
                damp = partial(torch.mul, other=1 + pg['damping'])
                diagonal_op_(Upt, op=damp)
                diagonal_op_(Vpt, op=damp)

                # Invert the block-diagonal matrices
                U_i = torchbsr2wp(inv_op(Upt))
                V_i = torchbsr2wp(inv_op(Vpt))

                # Build rhs_c = Ic - W V^{-1} Ip
                wp.copy(src=Ic, dest=rhs_c)
                sparse.bsr_mv(V_i, Ip, y=scratch_pts, beta=0.0)
                sparse.bsr_mv(W, scratch_pts, y=rhs_c, alpha=-1.0, beta=1.0)

                # --- Power series iteration ---
                # Dc^{(0)} = U^{-1} rhs_c
                D_c = wp.zeros_like(rhs_c)
                sparse.bsr_mv(U_i, rhs_c, y=D_c, beta=0.0)

                # Dc^{(k+1)} = U^{-1} (rhs_c + W V^{-1} W^T Dc^{(k)})
                tmp = wp.empty_like(rhs_c)
                D_c_prev = wp.empty_like(rhs_c)
                for _k in range(self.power_iters - 1):
                    wp.copy(src=D_c, dest=D_c_prev)
                    # Compute W V^{-1} W^T Dc^{(k)}
                    sparse.bsr_mv(Wt, D_c, y=scratch_pts2, beta=0.0)   # W^T Dc
                    sparse.bsr_mv(V_i, scratch_pts2, y=scratch_pts, beta=0.0)  # V^{-1} W^T Dc
                    wp.copy(src=rhs_c, dest=tmp)
                    sparse.bsr_mv(W, scratch_pts, y=tmp, alpha=1.0, beta=1.0)  # rhs_c + W V^{-1} W^T Dc
                    sparse.bsr_mv(U_i, tmp, y=D_c, beta=0.0)  # U^{-1} (...)

                    # Early stopping: (k+1) * ||D_c - D_c_prev|| < eps * ||D_c||
                    D_c_t = wp.to_torch(D_c)
                    delta_i = torch.linalg.norm(D_c_t - wp.to_torch(D_c_prev)).item()
                    delta = torch.linalg.norm(D_c_t).item()
                    if (_k + 1) * delta_i < 1e-3 * delta:
                        break

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
    dataset = get_problem(TARGET_PROBLEM, TARGET_DATASET)
    print(f"Fetched {TARGET_PROBLEM} from {TARGET_DATASET}")

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
        solver=None,  # No iterative solver needed — power series replaces it
        reject=30,
        power_iters=POWER_ITERATIONS,
    )

    print('Initial MSE loss:', least_square_error(
        model.pose,
        model.points,
        dataset["camera_index_of_observations"],
        dataset["point_index_of_observations"],
        dataset["points_2d"],
    ).item())

    print("Initial loss:", optimizer.model.loss(input, None).item())

    start = perf_counter()
    for idx in range(20):
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
