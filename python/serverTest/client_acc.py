"""Coordinate distributed bundle adjustment over the ZeroMQ C++ server."""

import bz2
import faulthandler
import importlib
import itertools
import json
import os
import secrets
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import zmq

from bal_evaluator import (
    encoded_daba_ray_state_to_matrix,
    evaluate_bal_state,
    evaluate_daba_state_pixel_error,
    evaluate_encoded_daba_ray_state,
    save_bal_state,
)
from clustering import (
    cluster_by_landmark_clean,
    cluster_by_landmark_scalable,
    cluster_by_landmark_scalable_stable,
    cluster_deg_by_landmark,
)
from drs_safeguards import should_reject_trial
from drs_consensus_metrics import (
    CONSENSUS_METRIC_MODES,
    reduce_camera_metric_blocks,
)
from numpy.linalg import inv as inv_nonHermetian
from scipy.sparse import bsr_matrix, csr_array, csr_matrix
from scipy.sparse import diags as diag_sparse

faulthandler.enable(all_threads=True)
client_started_at = time.perf_counter()
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY / "build" / "generated" / "proto"))
test_pb2 = importlib.import_module("test_pb2")
context = zmq.Context()
async_run_id = secrets.randbits(63) or 1
async_phase_ids = itertools.count(1)
transport_bytes_sent = 0
transport_bytes_received = 0
explicit_cost_landmarks = os.environ.get(
    "BUNDLE_PALM_EXPLICIT_COST_LANDMARKS", "0") == "1"
strict_trial_safeguard = os.environ.get(
    "BUNDLE_PALM_STRICT_TRIAL_SAFEGUARD", "0") == "1"
require_common_cost_match = os.environ.get(
    "BUNDLE_PALM_REQUIRE_COMMON_COST_MATCH", "1") == "1"
DRS_SCALING_METHOD = os.environ.get(
    "BUNDLE_PALM_DRS_SCALING", "jacobi_gmean").lower()
DRS_CONSENSUS_METRIC = os.environ.get(
    "BUNDLE_PALM_DRS_CONSENSUS_METRIC", "full").lower()
if DRS_CONSENSUS_METRIC not in CONSENSUS_METRIC_MODES:
    raise ValueError(
        "BUNDLE_PALM_DRS_CONSENSUS_METRIC must be one of "
        + ", ".join(CONSENSUS_METRIC_MODES))
DRS_OBJECTIVE_MODEL = os.environ.get(
    "BUNDLE_PALM_DRS_OBJECTIVE", "pixel").lower()
if DRS_OBJECTIVE_MODEL not in ("pixel", "daba_ray"):
    raise ValueError(
        "BUNDLE_PALM_DRS_OBJECTIVE must be 'pixel' or 'daba_ray'")
BLOCK_JACOBI_EIGENVALUE_FLOOR = float(os.environ.get(
    "BUNDLE_PALM_BLOCK_JACOBI_EIGENVALUE_FLOOR", "1e-6"))
if not 0 < BLOCK_JACOBI_EIGENVALUE_FLOOR <= 1:
    raise ValueError("BUNDLE_PALM_BLOCK_JACOBI_EIGENVALUE_FLOOR must be in (0, 1]")
DRS_SCALING_METHODS = {
    "none", "jacobi", "jacobi_gmean", "pock", "pock_gmean",
    "pock_alpha0", "pock_alpha0_gmean",
    "jacobi_q025_gmean", "jacobi_q0375_gmean", "jacobi_q0625_gmean",
    "jacobi_damped_gmean", "symmetric_ruiz_gmean",
    "block_jacobi_gmean"
}
if DRS_SCALING_METHOD not in DRS_SCALING_METHODS:
    raise ValueError(
        "BUNDLE_PALM_DRS_SCALING must be one of "
        + ", ".join(sorted(DRS_SCALING_METHODS)))


class InputProgress:
    def __init__(self):
        self.enabled = os.environ.get("BUNDLE_PALM_PARTITION_TRACE", "1") != "0"
        self.interval = max(0, int(os.environ.get(
            "BUNDLE_PALM_PARTITION_TRACE_INTERVAL_SECONDS", "5")))
        self.start = time.monotonic()
        self.last_report = self.start
        self.phase = "input"
        self.total = None
        self.unit = "items"
        self.completed = 0
        self.ticks = 0

    def begin(self, phase, total=None, unit="items"):
        self.phase = phase
        self.total = total
        self.unit = unit
        self.completed = 0
        self.ticks = 0

    def tick(self, amount=1, check_every=1024):
        self.completed += amount
        self.ticks += 1
        if (not self.enabled or
                (self.interval > 0 and self.ticks % check_every != 0)):
            return
        now = time.monotonic()
        if now - self.last_report < self.interval:
            return
        message = (
            f"input progress: {now - self.start:.1f} s, phase {self.phase}, "
            f"completed {self.completed} {self.unit}")
        if self.total:
            message += f" of {self.total} ({100 * self.completed / self.total:.1f}%)"
        print(message, file=sys.stderr, flush=True)
        self.last_report = now


# download ceres, edit CMakeList EXPORT_dir : On, cmake ../ceres-solver-2.2.0

# sudo apt-get install libeigen3-dev
# sudo apt install libzmq3-dev
# sudo apt install protobuf-compiler
# pip3 install zmq

# pip install protobuf==4.25.9
# in /proto:
# cd proto; protoc --python_out=. test.proto; cd -
# protoc --cpp_out=./output_directory your_file.proto
# cd build; CC=clang-15 CXX=clang++-15 cmake .. ; cd -

# python client.py

###########################################

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

# idea: median ste to 0, scale set to 100: let 95% fall into < 100 distance to center.
def normalize_by_points(points_3d_, cameras_):
    """Center the scene and scale its 95th-percentile point radius to 100."""
    # 1. get median in each direction.
    median = np.median(points_3d_, axis=0)
    points_3d_ = points_3d_ - median
    # simpler: rot median (still per camera)
    cam_loc = -AngleAxisRotatePoint(-cameras_[:,0:3], cameras_[:,3:6])
    cam_loc = cam_loc - median
    #cam_tra = AngleAxisRotatePoint(cameras[:,0:3], cam_loc)
    #cameras_[:,3:6] = cameras_[:,3:6] - median
    norm = np.linalg.norm(points_3d_, axis=1)
    scene_scale = np.percentile(norm, 95)
    if not np.isfinite(scene_scale) or scene_scale <= np.finfo(float).eps:
        raise ValueError("cannot normalize a scene with zero spatial extent")
    scale = 100 / scene_scale # scale = 1 to turn off, median always on?
    points_3d_ = points_3d_ * scale
    #cameras_[:,3:6] = cameras_[:,3:6] * scale
    cam_loc = cam_loc * scale
    cameras_[:,3:6] = -AngleAxisRotatePoint(cameras_[:,0:3], cam_loc)
    return points_3d_, cameras_

def read_bal_data(file_name):
    progress = InputProgress()
    with bz2.open(file_name, "rt") as file:
        n_cameras_, n_points_, n_observations = map(int, file.readline().split())

        camera_indices_ = np.empty(n_observations, dtype=int)
        point_indices_ = np.empty(n_observations, dtype=int)
        points_2d_ = np.empty((n_observations, 2))

        progress.begin("parse observations", n_observations, "observations")
        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices_[i] = int(camera_index)
            point_indices_[i] = int(point_index)
            points_2d_[i] = [float(x), float(y)]
            progress.tick()

        camera_params = np.empty(n_cameras_ * 9)
        progress.begin("parse cameras", n_cameras_ * 9, "values")
        for i in range(n_cameras_ * 9):
            camera_params[i] = float(file.readline())
            progress.tick()
        camera_params = camera_params.reshape((n_cameras_, -1))

        points_3d_ = np.empty(n_points_ * 3)
        progress.begin("parse landmarks", n_points_ * 3, "values")
        for i in range(n_points_ * 3):
            points_3d_[i] = float(file.readline())
            progress.tick()
        points_3d_ = points_3d_.reshape((n_points_, -1))

    # invert points_2d_ and focal distance if needed
    (camera_params, points_2d_) = \
        invert_focal_distance(camera_params, camera_indices_, points_2d_)

    (points_3d_ ,camera_params) = normalize_by_points(points_3d_ ,camera_params)

    return camera_params, points_3d_, camera_indices_, point_indices_, points_2d_

def is_valid_bz2(file_name):
    if not os.path.isfile(file_name):
        return False
    progress = InputProgress()
    progress.begin("validate BAL archive", unit="MiB decompressed")
    try:
        with bz2.open(file_name, "rb") as file:
            while chunk := file.read(1024 * 1024):
                progress.tick(len(chunk) / (1024 * 1024), check_every=1)
        return True
    except (EOFError, OSError):
        return False

def get_bal_file(base_url, file_name):
    candidates = [file_name, os.path.join("..", file_name)]
    for candidate in candidates:
        if is_valid_bz2(candidate):
            return candidate

    target = file_name
    temporary = target + ".part"
    if os.path.exists(temporary):
        os.remove(temporary)
    print("Downloading", base_url + file_name, "to", target)
    try:
        urllib.request.urlretrieve(base_url + file_name, temporary)
        if not is_valid_bz2(temporary):
            raise RuntimeError("downloaded BAL archive is incomplete")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target

def recv_message(socket_, operation):
    """Receive one message, adding context when the configured timeout expires."""
    global transport_bytes_received
    try:
        message = socket_.recv()
        transport_bytes_received += len(message)
        return message
    except zmq.Again as error:
        raise TimeoutError(f"timed out while {operation}") from error

def send_request(push_socket_, request_serialized, operation):
    """Send one asynchronous server request."""
    global transport_bytes_sent
    try:
        push_socket_.send(request_serialized)
        transport_bytes_sent += len(request_serialized)
    except zmq.Again as error:
        raise TimeoutError(f"timed out while {operation}") from error

def consume_cluster_reply(pending_cluster_ids, cluster_id, operation):
    """Ensure each asynchronous batch contains exactly one reply per cluster."""
    if cluster_id not in pending_cluster_ids:
        raise RuntimeError(
            f"unexpected or duplicate cluster {cluster_id} while {operation}")
    pending_cluster_ids.remove(cluster_id)

def is_current_async_reply(reply, phase_id):
    return reply.run_id == async_run_id and reply.phase_id == phase_id

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def blockInverse(M, bs):
    Mi = M.copy()
    if bs > 1:
        bs2 = bs * bs

        symmetric = True
        mat = M.data[0 : bs2].reshape(bs, bs)
        if not check_symmetric(mat):
            symmetric = False

        for i_ in range(int(M.data.shape[0] / bs2)):
            mat = Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs)
            if not symmetric:
                mat = np.fliplr(mat)
                #imat = inv_dense(mat, hermitian=True)
                imat = inv_nonHermetian(mat) # faster.
                imat = np.fliplr(imat)
            else:
                #imat = inv_dense(mat, hermitian=True)
                imat = inv_nonHermetian(mat)
            Mi.data[bs2 * i_ : bs2 * i_ + bs2] = imat.flatten()
    else:
        Mi = M.copy()
        for i_ in range(int(M.data.shape[0])):
            Mi.data[i_ : i_ + 1] = 1.0 / Mi.data[i_ : i_ + 1]
    return Mi

# cost is fuk * fuk + rho_k/2 |u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k>
#                     rho_k/2 <u_k - v_k, u_k - v_k> - rho_k <s_k - u_k, u_k - v_k>
#                     rho_k/2 <u_k - v_k - 2s_k + 2u_k, u_k - v_k>
#                     rho_k/2 <3u_k - v_k - 2s_k, u_k - v_k>
#                     rho_k/2 {v^tv - 2vT[2uk-sk] + uk^T[3uk-2sk]}
def cost_DRE(
    #camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, pose_v
    camera_indices_in_cluster_,  poses_in_cluster_, poses_s_in_cluster_, L_in_cluster_, Ul_in_cluster_, pose_v_, nabla_p_in_cluster_):
    """Evaluate the Douglas-Rachford envelope contribution by cluster."""
    num_cams =  poses_in_cluster_[0].shape[0]
    #sum_Ds_2u = np.zeros(num_cams * 9)
    #sum_constant_term = 0
    sum_u_s = 0
    sum_u_v = 0
    sum_u_v_ = 0
    sum_2u_s_v = 0
    cost_dre = 0
    dre_per_part = []
    # for i in range(len(Ul_in_cluster_)):
    for i, Ul_in_cluster_i in enumerate(Ul_in_cluster_):
        camera_indices_ = np.unique(camera_indices_in_cluster_[i])
        indices = np.repeat(np.array([9 * camera_indices_ + j for j in range(9)]).transpose(), 9, axis=0).flatten()

        indptr = [np.array([0])]
        j = 0
        for q in range(num_cams):
            if j < camera_indices_.shape[0] and camera_indices_[j] == q:
                indptr.append(np.array([81 * j + 9, 81 * j + 18, 81 * j + 27, 
                                        81 * j + 36, 81 * j + 45, 81 * j + 54, 
                                        81 * j + 63, 81 * j + 72, 81 * j + 81]).flatten())
                j = j + 1
            else:
                indptr.append(np.array([81 * j, 81 * j, 81 * j, 81 * j, 
                                        81 * j, 81 * j, 81 * j, 81 * j, 81 * j]).flatten())
        indptr = np.concatenate(indptr)

        U_pose = csr_matrix(
            (Ul_in_cluster_i.data, indices, indptr),
            shape=(9 * num_cams, 9 * num_cams),
        )
        u2_s = (2 *  poses_in_cluster_[i].flatten() - poses_s_in_cluster_[i].flatten())
        #sum_Ds_2u += U_pose * u2_s # has 0's for those not present
        #sum_constant_term +=  poses_in_cluster_[i].flatten().dot(U_pose * (poses_in_cluster_[i].flatten() + u2_s - poses_s_in_cluster_[i].flatten()))

        u_s =  poses_in_cluster_[i].flatten() - poses_s_in_cluster_[i].flatten()
        # do not do this: never accepted extrapolation
        #u_s =  2 * poses_in_cluster_[i].flatten() - poses_s_in_cluster_[i].flatten() - pose_v_.flatten() # next s : s + v-u

        u_v =  poses_in_cluster_[i].flatten() - pose_v_.flatten()
        v_u2_s = u2_s - pose_v_.flatten()
        sum_u_s += u_s.dot(U_pose * u_s)
        sum_u_v += u_v.dot(U_pose * u_v)
        sum_u_v_ += u_v.dot(u_v)
        sum_2u_s_v += v_u2_s.dot(U_pose * v_u2_s)

        # rho_k/2 |u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k>
        # rho_k/2 ( uu  + vv - 2uv - 2su + 2uu + 2sv - 2uv)
        # rho_k/2 ( 3uu + vv - 4uv - 2su + 2sv)
        # v only
        # rho_k/2 ( vv + v(2s-4u)   - 2su + 3uu)
        # deriv
        # rho_k (2v + 2s-4u), same
        #
        # rho_k/2 |u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k>
        # deriv
        # sum_k rho_k v + rho_k (s_k - 2 u_k) = 0
        # v = (sum_k rho_k)^-1 (sum_k rho_k (2 u_k - s_k))

        prox_solution = True # does not matter
        if prox_solution:
            # rho_k/2 |u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k>
            # rho_k/2 <u_k - v_k - 2s_k + 2u_k , u_k - v_k>
            local_cost = 0.5 * u_v.dot(U_pose * (u_v + 2 * u_s))
        else: # assuming we do not solve the problem exactly
            nabla_p = np.zeros(num_cams * 9)
            nabla_p[np.array([9 * camera_indices_ + j for j in range(9)]).transpose().flatten()] = nabla_p_in_cluster_[i]
            local_cost = 0.5 * u_v.dot(U_pose * u_v) - u_v.dot(nabla_p)

        cost_dre  += local_cost
        dre_per_part.append(
            round(local_cost.copy()) if np.isfinite(local_cost) else local_cost)

    # TODO: I use a different Vl to compute the cost here than in the update of prox u.
    #       Since I want to work with a new Vl already. Problem.
    # i want |u-s|_D |u-v|_D, also |v-2u-s|_D
    #cost_input  = 0.5 * (pose_v_.flatten().dot(Ul_all * pose_v_.flatten() - 2 * sum_Ds_2u) + sum_constant_term)
    print("---- |u-s|^2_D ", round(sum_u_s), "|u-v|^2_D ", round(sum_u_v), "|2u-s-v|^2_D ", round(sum_2u_s_v),
          "|u-v|^2 ", round(sum_u_v_), " cost_dre ", cost_dre, file=sys.stderr)
    print("---- dre_per_part --- ", dre_per_part, file=sys.stderr) # must be < 0.
    return cost_dre, dre_per_part

def average_cameras_new(
    camera_indices_in_cluster_, poses_in_cluster_, poses_s_in_cluster_, L_in_cluster_, UL_in_cluster_, nabla_p_in_cluster_):
    """Compute the block-metric weighted consensus camera parameters."""
    num_cameras = poses_in_cluster_[0].shape[0]
    projection_sum = np.zeros((num_cameras, 9, 9))
    projection_rhs = np.zeros((num_cameras, 9))
    UL_zeros_in_cluster_ = []

    # Here or per part.
    # compressedData = False # idea we would send a compressed version of the stepsize.
    # if compressedData:
    #     for i in range(len(UL_in_cluster_)):
    #         UL_in_cluster_[i] = PostCompressBlockMatrix(UL_in_cluster_[i], 9) # this would be send in quantized form. We would need ensure its spd.

    #for i in range(len(UL_in_cluster_)):
    for i, UL_in_cluster_i in enumerate(UL_in_cluster_):
        # Lc = L_in_cluster_[i]
        camera_indices_ = np.unique(camera_indices_in_cluster_[i])
        block_values = np.asarray(UL_in_cluster_i)
        expected_values = 81 * len(camera_indices_)
        if block_values.size != expected_values:
            raise ValueError(
                f"cluster {i} returned {block_values.size} step-size values; "
                f"expected {expected_values}")
        block_indptr = np.zeros(num_cameras + 1, dtype=np.int64)
        block_indptr[camera_indices_ + 1] = 1
        np.cumsum(block_indptr, out=block_indptr)
        projection_blocks = reduce_camera_metric_blocks(
            block_values.reshape(-1, 9, 9), DRS_CONSENSUS_METRIC)
        reflected_cameras = (
            2 * poses_in_cluster_[i][camera_indices_]
            - poses_s_in_cluster_[i][camera_indices_])
        projection_sum[camera_indices_] += projection_blocks
        projection_rhs[camera_indices_] += np.einsum(
            "bij,bj->bi", projection_blocks, reflected_cameras)
        U_pose = bsr_matrix(
            (projection_blocks, camera_indices_, block_indptr),
            shape=(9 * num_cameras, 9 * num_cameras),
        ).tocsr()
        UL_zeros_in_cluster_.append(U_pose)
        # print(mean_points.shape, " " , V_land.shape, points_3d_in_cluster_[i].shape)
        # print cost after/before.
        # cost old v is where ? (v-2u+s)^T V_land (v-2u+s) = v^T V_land v + 2 v^T V_land (-2u+s) + (2u-s)^T V_land (2u-s)
        # derivative 2 V_land v + 2 V_land (-2u+s) = 0 <-> sum (V_land) v = sum (V_land (2u-s))

        # print(i, "averaging 3d ", points_3d_in_cluster_[i][globalSingleLandmarksB_in_c[i], :]) # indeed 1 changed rest is constant
        # print(i, "averaging vl ", V_land.data.reshape(-1,9)[globalSingleLandmarksA_in_c[i],:])  # indeed diagonal

        raw_U_pose = bsr_matrix(
            (block_values.reshape(-1, 9, 9), camera_indices_, block_indptr),
            shape=(9 * num_cameras, 9 * num_cameras),
        ).tocsr()
        if i == 0:
            raw_Up_all = raw_U_pose
        else:
            raw_Up_all += raw_U_pose

    # rho_k/2 |u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k> is actually
    # For phi(v) = sum_k 1/2 |u_k-v|^2_{D_k}
    #                         - <g_k, u_k-v>,
    # stationarity gives sum_k [D_k(v-u_k) + g_k] = 0, hence
    # v = (sum_k D_k)^-1 sum_k (D_k u_k - g_k).
    # Here g_k = D_k(s_k-u_k), so
    # v = (sum_k D_k)^-1 sum_k D_k(2u_k-s_k).
    pose_v_out = poses_in_cluster_[0].copy()
    active_cameras = np.flatnonzero(
        np.any(projection_sum != 0.0, axis=(1, 2)))
    for camera in active_cameras:
        pose_v_out[camera] = np.linalg.solve(
            projection_sum[camera], projection_rhs[camera])

    return pose_v_out, raw_Up_all, UL_zeros_in_cluster_


# Operates sequentially.
def prox_f_push_pull(camera_indices_in_cluster_, point_indices_in_cluster_, local_camera_indices_in_cluster_,
           local_landmark_indices_in_cluster_, points_2d_in_cluster_, poses_in_cluster_, landmarks_,
           poses_s_in_cluster_, L_in_cluster_, Vl_in_cluster_, blockEig_in_cluster_, kClusters_,
           LipJ_, innerIts_ = 1, revert_lm = 0) :
    """Run one proximal solve per cluster and collect out-of-order results."""
    cost_ = np.zeros(kClusters_)
    nabla_p_in_cluster_ = [0 for _ in range(kClusters_)]

    # ignore for now:
    # num_poses = poses_in_cluster_[0].shape[0]
    # pose_occurences = np.zeros(num_poses)
    # for ci_ in range(kClusters):
    #     unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci_])
    #     pose_occurences[unique_poses_in_c_] +=1

    global global_init
    #global global_iteration
    global push_socket
    global pull_socket
    phase_id = next(async_phase_ids)

    for ci in range(kClusters_):
        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci])
        if global_init:
            # print("Sending program …", ci)
            request = test_pb2.request_proto()
            #request.program.SetInParent()
            #program = request.program
            request.program.cameras[:] = poses_in_cluster_[ci][unique_poses_in_c_].ravel()
            #request.program.cameras_s[:] = cameras.ravel() # set later in prox_cluster_proto
            request.program.landmarks[:] = landmarks_[unique_points_in_c_].ravel()
            request.program.observations[:] = points_2d_in_cluster_[ci].ravel()
            request.program.cam_id[:] = local_camera_indices_in_cluster_[ci].ravel()
            request.program.lm_id[:] = local_landmark_indices_in_cluster_[ci].ravel()
            request.program.iterations = innerIts_
            request.program.be = blockEig_in_cluster_[ci]
            request.program.num_clusters = kClusters_
            request.program.cluster_id = ci
            request.program.run_id = async_run_id
            request.program.phase_id = phase_id
            request.program.init_l = LipJ_
            request.program.unorm[:] = np.ones(9 * unique_poses_in_c_.shape[0])
            request.program.vnorm[:] = np.ones(3 * unique_points_in_c_.shape[0])
            request.program.camera_transform[:] = np.tile(
                np.eye(9), (unique_poses_in_c_.shape[0], 1, 1)
            ).ravel()
            request.program.objective_model = (
                1 if DRS_OBJECTIVE_MODEL == "daba_ray" else 0)
            request.program.initial_focal[:] = initial_ray_focal[
                unique_poses_in_c_]
            if request.program.objective_model != (
                1 if DRS_OBJECTIVE_MODEL == "daba_ray" else 0
            ):
                raise RuntimeError("failed to encode worker objective model")
        else: # just update
            #print("Sending request …", ci)
            request = test_pb2.request_proto()
            #cameras = test_pb2.camera_proto()
            #temp = program_deserialized.cameras[:]
            #temp = [i * 10 for i in temp]
            #request.cameras.cameras[:] = temp # ok, program works with changed data.
            request.update.cameras[:] = poses_in_cluster_[ci][unique_poses_in_c_].ravel()
            request.update.cameras_s[:] = poses_s_in_cluster_[ci][unique_poses_in_c_].ravel()
            request.update.be = blockEig_in_cluster_[ci]
            request.update.cluster_id = ci
            request.update.run_id = async_run_id
            request.update.phase_id = phase_id
            # 0 accepts the current landmarks, 1 restores the previous trial,
            # and 2 restores the landmarks saved with the global best cost.
            if revert_lm == 1:
                request.update.revert_lm = 1
            elif revert_lm == 2:
                request.update.revert_lm = 2
                # here send lms as well, else those are empty -- or better last cost was best -> keep lms.
            else:
                request.update.revert_lm = 0 # next step

        request_serialized_ = request.SerializeToString() # SerializeToArray() does not exist
        send_request(push_socket, request_serialized_, f"starting cluster {ci}")

    pending_cluster_ids = set(range(kClusters_))
    uncorrelated_replies = 0
    while pending_cluster_ids:
        #print("Receiving return …", k)
        return_proto_ = test_pb2.return_cluster_proto()
        message_in_bytes_ = recv_message(
            pull_socket, "waiting for a cluster result")

        #message_out_str = "Ok" # this might not be needed if this socket is pull not REC
        #message_out_bytes = message_out_str.encode("utf-8")
        #pull_socket.send(message_out_bytes)

        return_proto_.ParseFromString(message_in_bytes_)# ParseFromArray(message_in_bytes_)
        if not is_current_async_reply(return_proto_, phase_id):
            uncorrelated_replies += (
                return_proto_.run_id == 0 and return_proto_.phase_id == 0)
            if uncorrelated_replies >= kClusters_:
                raise RuntimeError(
                    "server replies do not contain correlation IDs; restart "
                    "the rebuilt serverTest/build/zeromq_cpp_server_ex")
            print(
                "Discarding stale cluster result for run/phase "
                f"{return_proto_.run_id}/{return_proto_.phase_id}",
                file=sys.stderr,
            )
            continue
        ci = return_proto_.cluster_id
        consume_cluster_reply(
            pending_cluster_ids, ci, "receiving cluster results")
        expected_objective_model = (
            1 if DRS_OBJECTIVE_MODEL == "daba_ray" else 0)
        if return_proto_.objective_model != expected_objective_model:
            raise RuntimeError(
                "worker objective model does not match coordinator: "
                f"{return_proto_.objective_model} versus "
                f"{expected_objective_model}")
        #print("Return for cluster …", ci)

        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci])
        expected_camera_values = 9 * unique_poses_in_c_.size
        expected_landmark_values = 3 * unique_points_in_c_.size
        expected_step_values = 81 * unique_poses_in_c_.size
        if len(return_proto_.cameras) != expected_camera_values:
            raise RuntimeError(
                f"cluster {ci} returned {len(return_proto_.cameras)} camera "
                f"values; expected {expected_camera_values}")
        if len(return_proto_.landmarks) != expected_landmark_values:
            raise RuntimeError(
                f"cluster {ci} returned {len(return_proto_.landmarks)} landmark "
                f"values; expected {expected_landmark_values}")
        if len(return_proto_.step_size) != expected_step_values:
            raise RuntimeError(
                f"cluster {ci} returned {len(return_proto_.step_size)} step-size "
                f"values; expected {expected_step_values}")
        cost_[ci] = return_proto_.cost
        # L_in_cluster_[ci] = LipJ_ # unsused anyway
        Vl_in_cluster_[ci] = np.array(return_proto_.step_size[:]) #.copy() # stepsize
        # print(ci, " UL_in_cluster_[i].shape", (Vl_in_cluster_[ci]).shape)
        # print(ci , " Vl_in_cluster_[ci].data " , (Vl_in_cluster_[ci]).data)
        poses_in_cluster_[ci][unique_poses_in_c_, :] = np.asarray(return_proto_.cameras[:]).reshape((-1, 9))
        #poses_in_cluster_[ci][unique_poses_in_c_, :] = np.asarray(return_proto_.cameras[:], dtype = 'float').reshape((-1, 9))
        landmarks_[unique_points_in_c_,:] = np.array(return_proto_.landmarks[:]).reshape((-1, 3))
        #blockEig_in_cluster_[ci] = blockEig_in_c_ # not done

    # print("exit prox_f")
    global_init = False
    return (cost_, L_in_cluster_, Vl_in_cluster_, poses_in_cluster_, landmarks_, nabla_p_in_cluster_, blockEig_in_cluster_)

# Operates sequentially.
def primal_cost_push_pull(
    camera_indices_in_cluster_, point_indices_in_cluster_, poses_in_cluster_,
    landmarks_, k_clusters, single_pose=False, revert_lm_=False
):

    global push_socket
    global pull_socket
    phase_id = next(async_phase_ids)

    for ci in range(k_clusters):
        unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci])
        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        #print("Sending cost query …", ci)
        request = test_pb2.request_proto()
        #request.program.SetInParent()
        #program = request.program
        if single_pose:
            request.cost_update.cameras[:] = poses_in_cluster_[unique_poses_in_c_].ravel()
        else:
            request.cost_update.cameras[:] = poses_in_cluster_[ci][unique_poses_in_c_].ravel()
        request.cost_update.cluster_id = ci
        request.cost_update.run_id = async_run_id
        request.cost_update.phase_id = phase_id
        if revert_lm_:
            request.cost_update.revert_lm = 2
        else:
            request.cost_update.revert_lm = 0
            if explicit_cost_landmarks:
                request.cost_update.landmarks[:] = landmarks_[
                    unique_points_in_c_].ravel()

        request_serialized_ = request.SerializeToString() # SerializeToArray() does not exist
        send_request(push_socket, request_serialized_, f"requesting cost for cluster {ci}")

    cost_ = np.zeros(k_clusters)
    cost_landmarks_ = landmarks_.copy()
    pending_cluster_ids = set(range(k_clusters))
    uncorrelated_replies = 0
    while pending_cluster_ids:
        #print("Receiving return …", k)
        return_proto_ = test_pb2.return_cost_proto()
        message_in_bytes_ = recv_message(
            pull_socket, "waiting for a cluster cost")
        return_proto_.ParseFromString(message_in_bytes_)
        if not is_current_async_reply(return_proto_, phase_id):
            uncorrelated_replies += (
                return_proto_.run_id == 0 and return_proto_.phase_id == 0)
            if uncorrelated_replies >= k_clusters:
                raise RuntimeError(
                    "server replies do not contain correlation IDs; restart "
                    "the rebuilt serverTest/build/zeromq_cpp_server_ex")
            print(
                "Discarding stale cluster cost for run/phase "
                f"{return_proto_.run_id}/{return_proto_.phase_id}",
                file=sys.stderr,
            )
            continue
        ci = return_proto_.cluster_id
        consume_cluster_reply(
            pending_cluster_ids, ci, "receiving cluster costs")
        # print("Receiving cost for cluster …", ci, " = ", return_proto_.cost)
        cost_[ci] = return_proto_.cost
        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        expected_landmark_values = 3 * unique_points_in_c_.size
        if len(return_proto_.landmarks) != expected_landmark_values:
            raise RuntimeError(
                f"cluster {ci} returned {len(return_proto_.landmarks)} cost "
                f"landmark values; expected {expected_landmark_values}")
        cost_landmarks_[unique_points_in_c_] = np.asarray(
            return_proto_.landmarks, dtype=np.float64).reshape(-1, 3)

    # print("exit primal_cost_push_pull")
    return cost_, cost_landmarks_


def retrieve_best_landmarks(
    camera_indices_in_cluster_, point_indices_in_cluster_, best_poses_,
    landmarks_, k_clusters
):
    """Retrieve the physical worker landmarks saved with the native best cost."""
    global push_socket
    global pull_socket
    phase_id = next(async_phase_ids)
    retrieved = landmarks_.copy()

    for ci in range(k_clusters):
        unique_poses = np.unique(camera_indices_in_cluster_[ci])
        request = test_pb2.request_proto()
        request.cost_update.cameras[:] = best_poses_[unique_poses].ravel()
        request.cost_update.cluster_id = ci
        request.cost_update.run_id = async_run_id
        request.cost_update.phase_id = phase_id
        request.cost_update.revert_lm = 2
        send_request(
            push_socket,
            request.SerializeToString(),
            f"retrieving best landmarks for cluster {ci}",
        )

    pending_cluster_ids = set(range(k_clusters))
    while pending_cluster_ids:
        reply = test_pb2.return_cost_proto()
        reply.ParseFromString(recv_message(
            pull_socket, "waiting for best worker landmarks"))
        if not is_current_async_reply(reply, phase_id):
            continue
        ci = reply.cluster_id
        consume_cluster_reply(
            pending_cluster_ids, ci, "receiving best worker landmarks")
        unique_points = np.unique(point_indices_in_cluster_[ci])
        expected_values = 3 * unique_points.size
        if len(reply.landmarks) != expected_values:
            raise RuntimeError(
                f"cluster {ci} returned {len(reply.landmarks)} best landmark "
                f"values; expected {expected_values}")
        retrieved[unique_points] = np.asarray(
            reply.landmarks, dtype=np.float64).reshape(-1, 3)

    return retrieved

# Operates sequentially.
def best_cost_found_push_pull(
    point_indices_in_cluster_, k_clusters, costs, cost_landmarks_
) :

    global push_socket
    global pull_socket

    for ci in range(k_clusters):
        #print("Sending cost query …", ci)
        request = test_pb2.request_proto()
        request.best_cost.cost = costs[ci]
        request.best_cost.cluster_id = ci
        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        request.best_cost.landmarks[:] = cost_landmarks_[
            unique_points_in_c_].ravel()

        request_serialized_ = request.SerializeToString() # SerializeToArray() does not exist
        send_request(
            push_socket, request_serialized_, f"updating best cost for cluster {ci}")
    return

def rounded_cost(value):
    return round(value) if np.isfinite(value) else float(value)

def normalize_geometric_mean(values, floor=1e-12):
    values = np.asarray(values, dtype=np.float64)
    safe_values = np.maximum(
        np.where(np.isfinite(values), values, floor),
        floor,
    )
    geometric_mean = np.exp(np.mean(np.log(safe_values)))
    return values / geometric_mean

def getScaling(min_, max_): # aim at max * min = 1. So max * x = 1/(min * x). x^2 = 1/(min * max)
    return np.sqrt(1. / (min_ * max_))

def get_symmetric_ruiz_scaling(matrix, max_iterations=10, tolerance=1e-3):
    absolute = np.abs(matrix).tocsr()
    scaling = np.ones(matrix.shape[0], dtype=np.float64)
    for _ in range(max_iterations):
        inverse = np.reciprocal(scaling)
        equilibrated = absolute.multiply(inverse[:, None]).multiply(inverse[None, :])
        row_norms = np.asarray(equilibrated.max(axis=1).toarray()).ravel()
        valid = row_norms[np.isfinite(row_norms) & (row_norms > 0)]
        if valid.size == 0:
            break
        floor = max(1e-12 * np.median(valid), np.finfo(float).tiny)
        update = np.sqrt(np.maximum(
            np.where(np.isfinite(row_norms), row_norms, floor), floor
        ))
        scaling *= update
        if np.max(np.abs(np.log(update))) < tolerance:
            break
    return scaling

def GetPcgScalingDiag(JtJ, W, method=DRS_SCALING_METHOD):
    if method in {"none", "block_jacobi_gmean"}:
        return np.ones(JtJ.shape[1])

    if method == "symmetric_ruiz_gmean":
        temp_ = get_symmetric_ruiz_scaling(JtJ)
        return normalize_geometric_mean(temp_)

    baseVersion = False
    if baseVersion:
        temp_  = np.squeeze(np.asarray((np.abs(JtJ)).sum(axis=0) )) # ATTENTION: must adjust / add sqrt on lms here below. CCC
    else:
        squared = False
        if squared:
            JtJ_ = JtJ.copy()
            W_ = W.copy()
            JtJ_.data = np.square(JtJ_.data)
            W_.data = np.square(W_.data)
            temp_ = np.squeeze(np.asarray((np.abs(JtJ_)).sum(axis=0) ))
            temp_W = np.squeeze(np.asarray((np.abs(W_)).sum(axis=0) ))
            temp_ = np.sqrt(temp_ + temp_W)
        else: # just jacobi, looks best?
            #temp_ = np.squeeze(np.asarray((np.abs(JtJ)).sum(axis=0) ))
            #temp_W = np.squeeze(np.asarray((np.abs(W)).sum(axis=0) ))
            #temp_  = temp_ + temp_W # + 1e-6 does nothing
            if method.startswith("pock_alpha0"):
                squared = JtJ.copy()
                squared.data = np.square(np.abs(squared.data))
                temp_ = np.asarray(squared.sum(axis=0)).ravel()
            elif method.startswith("pock"):
                temp_ = np.asarray(np.abs(JtJ).sum(axis=0)).ravel()
            else:
                temp_ = np.abs(JtJ.diagonal())
            positive = temp_[np.isfinite(temp_) & (temp_ > 0)]
            if positive.size == 0:
                return np.ones(JtJ.shape[1])
            floor = max(1e-12 * np.median(positive), np.finfo(float).tiny)
            temp_ = np.where(np.isfinite(temp_), temp_, floor)
            if method == "jacobi_damped_gmean":
                temp_ += 1e-3 * np.median(positive)
            jacobi_exponents = {
                "jacobi_q025_gmean": 0.25,
                "jacobi_q0375_gmean": 0.375,
                "jacobi_q0625_gmean": 0.625,
            }
            exponent = jacobi_exponents.get(method, 0.5)
            temp_ = np.power(np.maximum(temp_, floor), exponent)

    if False: # clamp and rescale
        print("min/max Unorm before ", np.min(temp_), np.max(temp_))
        minTemp = np.percentile(temp_[np.nonzero(temp_)], 0.0001) # not sure..
        t = getScaling(minTemp, np.max(temp_))
        temp_  = temp_ * t
        print("min/max Unorm after ", np.min(temp_), np.max(temp_), " t ", t, " min*max= ", np.min(temp_) * np.max(temp_))
        print("Preconditioners min/max Unorm ", np.min(temp_), np.max(temp_))
        minTresh = 1e-18 # 12 -> 14 for 245 and scale!
        maxTresh = 1e18
        temp_ = np.fmin(np.fmax(temp_, minTresh), maxTresh) #np.sqrt(np.minimum(np.maximum(t, minTresh), maxTresh))
        print("Preconditioners min/max Unorm after thresholding ", np.min(temp_), np.max(temp_))

    scaleToHaveValuesAroundOneForHess = False #True # cosmetics mostly.
    if scaleToHaveValuesAroundOneForHess:
        absDiagJtJ = np.abs(JtJ.diagonal())
        guess = diag_sparse(1./temp_.flatten()) * absDiagJtJ * diag_sparse(1./temp_.flatten())
        print("Preconditioners min/max guess ", np.min(guess), np.max(guess))
        scale = np.sqrt(np.median(guess)) # same as 1e-1 * np.sqrt(np.median(guess))
        print("scale ", scale) # there has to be a stepsize issue?
        temp_ = temp_ * scale # * 1e5 works but not as well ()
        print("Preconditioners min/max Unorm after scaling 2: ", np.min(temp_), np.max(temp_))
        guess = diag_sparse(1./temp_.flatten()) * absDiagJtJ * diag_sparse(1./temp_.flatten())
        print("Preconditioners min/max guess ", np.min(guess), np.max(guess))

    if method.endswith("_gmean"):
        temp_ = normalize_geometric_mean(temp_)
    return temp_


def get_camera_block_transforms(
    matrix, block_size=9,
    eigenvalue_floor=BLOCK_JACOBI_EIGENVALUE_FLOOR):
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % block_size != 0:
        raise ValueError("camera curvature matrix must contain complete square blocks")
    num_cameras = matrix.shape[0] // block_size
    transforms = np.empty((num_cameras, block_size, block_size))
    for camera_id in range(num_cameras):
        start = block_size * camera_id
        block = matrix[start:start + block_size, start:start + block_size].toarray()
        block = 0.5 * (block + block.T)
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        largest = max(float(np.max(eigenvalues)), np.finfo(float).tiny)
        eigenvalues = np.maximum(eigenvalues, eigenvalue_floor * largest)
        transforms[camera_id] = (
            eigenvectors * np.reciprocal(np.sqrt(eigenvalues))
        ) @ eigenvectors.T

    determinant_signs, log_determinants = np.linalg.slogdet(transforms)
    if np.any(determinant_signs <= 0) or not np.all(np.isfinite(log_determinants)):
        raise ValueError("camera block transform is not finite positive definite")
    geometric_mean_singular_value = np.exp(
        np.mean(log_determinants) / block_size
    )
    transforms /= geometric_mean_singular_value
    if not np.all(np.isfinite(transforms)):
        raise ValueError("camera block transform contains non-finite values")
    return transforms


def apply_camera_transforms(poses, transforms, inverse=False):
    if inverse:
        return np.linalg.solve(transforms, poses[..., None])[..., 0]
    return np.einsum("nij,nj->ni", transforms, poses)


def physical_cameras_from_preconditioned(
    poses, unorm_, camera_transforms_, cluster_count
):
    if camera_transforms_ is None:
        return poses / unorm_.reshape(-1, 9)
    return apply_camera_transforms(
        poses, camera_transforms_
    ) / (2.0 / np.sqrt(cluster_count))


def preconditioning_push(poses_v_, poses_in_cluster_, poses_s_in_cluster_, camera_indices_in_cluster_,
                         point_indices_in_cluster_, unorm_, vnorm_, kClusters_,
                         camera_transforms_=None) :

    global push_socket

    unorm_ *= 2. / np.sqrt(kClusters_) # TODO: check on small example

    if camera_transforms_ is None:
        poses_v_ = (unorm_ * poses_v_.ravel()).reshape(-1,9)
    else:
        poses_v_ = apply_camera_transforms(
            poses_v_, camera_transforms_, inverse=True
        ) * (2. / np.sqrt(kClusters_))

    for ci in range(kClusters_):
        unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci])
        unique_landmarks_in_c_ = np.unique(point_indices_in_cluster_[ci])
        #print("Sending preconditioning query …", ci)
        request = test_pb2.request_proto()

        if camera_transforms_ is None:
            poses_in_cluster_[ci] = (unorm_ * poses_in_cluster_[ci].ravel()).reshape(-1,9)
            poses_s_in_cluster_[ci] = (unorm_ * poses_s_in_cluster_[ci].ravel()).reshape(-1,9)
        else:
            poses_in_cluster_[ci] = apply_camera_transforms(
                poses_in_cluster_[ci], camera_transforms_, inverse=True
            ) * (2. / np.sqrt(kClusters_))
            poses_s_in_cluster_[ci] = apply_camera_transforms(
                poses_s_in_cluster_[ci], camera_transforms_, inverse=True
            ) * (2. / np.sqrt(kClusters_))

        if camera_transforms_ is None:
            request.preconditioning_update.unorm[:] = 1./(unorm_.reshape(-1,9)[unique_poses_in_c_,:]).ravel()
        else:
            request.preconditioning_update.unorm[:] = np.full(
                9 * unique_poses_in_c_.shape[0],
                1. / (2. / np.sqrt(kClusters_)),
            )
            request.preconditioning_update.camera_transform[:] = (
                camera_transforms_[unique_poses_in_c_]
            ).ravel()
        #request.preconditioning_update.vnorm[:] = vnorm_[unique_landmarks_in_c_].ravel()
        request.preconditioning_update.vnorm[:] = np.ones(3 * unique_landmarks_in_c_.shape[0])
        request.preconditioning_update.cluster_id = ci
        request_serialized_ = request.SerializeToString() # SerializeToArray() does not exist
        send_request(
            push_socket, request_serialized_, f"preconditioning cluster {ci}")
    return poses_v_, poses_in_cluster_, poses_s_in_cluster_

def GetLocalIndices(point_indices_in_cluster, camera_indices_in_cluster):
    if len(point_indices_in_cluster) != len(camera_indices_in_cluster):
        raise ValueError("point and camera cluster lists must have equal length")
    local_landmark_indices_in_cluster = [
        np.unique(indices, return_inverse=True)[1]
        for indices in point_indices_in_cluster
    ]
    local_camera_indices_in_cluster = [
        np.unique(indices, return_inverse=True)[1]
        for indices in camera_indices_in_cluster
    ]

    return (local_landmark_indices_in_cluster, local_camera_indices_in_cluster)

#################
def AngleAxisRotatePointT(angleAxis, pt):
    theta2 = (angleAxis * angleAxis).sum(dim=1)

    mask = (theta2 > 0).float()  # ? == 0 is alternative? check other repo

    theta = torch.sqrt(theta2 + (1 - mask))

    mask = mask.reshape((mask.shape[0], 1))
    mask = torch.cat([mask, mask, mask], dim=1)

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
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

    r0 = r0.reshape((r0.shape[0], 1))
    r1 = r1.reshape((r1.shape[0], 1))
    r2 = r2.reshape((r2.shape[0], 1))

    res1 = torch.cat([r0, r1, r2], dim=1)

    wCrossPt0 = angleAxis[:, 1] * pt[:, 2] - angleAxis[:, 2] * pt[:, 1]
    wCrossPt1 = angleAxis[:, 2] * pt[:, 0] - angleAxis[:, 0] * pt[:, 2]
    wCrossPt2 = angleAxis[:, 0] * pt[:, 1] - angleAxis[:, 1] * pt[:, 0]

    r00 = pt[:, 0] + wCrossPt0
    r01 = pt[:, 1] + wCrossPt1
    r02 = pt[:, 2] + wCrossPt2

    r00 = r00.reshape((r00.shape[0], 1))
    r01 = r01.reshape((r01.shape[0], 1))
    r02 = r02.reshape((r02.shape[0], 1))

    res2 = torch.cat([r00, r01, r02], dim=1)

    return res1 * mask + res2 * (1 - mask)

def buildMatrixNew(dx, dy, v_indices, sz=9) :
    data = []
    indptr = []
    indices = []

    start = 0
    end = v_indices.shape[0]

    data.append(dx.flatten())
    data.append(dy.flatten())
    # print("dx datavals ", dx)
    # print("dy datavals ", dy)
    indptr.append(np.arange(2*start*sz, 2*end*sz, sz).flatten())
    indices.append(np.array([sz * v_indices[start:end] + j for j in range(sz)]).transpose().flatten())
    indices.append(np.array([sz * v_indices[start:end] + j for j in range(sz)]).transpose().flatten())
    indptr.append(np.array([sz+ indptr[-1][-1]])) # closing

    datavals = np.concatenate(data)
    crs_pose = csr_array((datavals, np.concatenate(indices), np.concatenate(indptr)))

    J_pose = csr_matrix(crs_pose)
    return J_pose

def buildResiduumNew(resX, resY) :
    data = []
    data.append(resX.flatten().numpy())
    data.append(resY.flatten().numpy())
    res = np.concatenate(data)
    return res

def torchSingleResiduumX(camera_params, point_params, p2d) :
    angle_axis = camera_params[:,:3]
    points_cam = AngleAxisRotatePointT(angle_axis, point_params)
    points_cam[:,0:2] = points_cam[:,0:2] + camera_params[:, 3:5]
    points_cam[:,2] = points_cam[:,2] + camera_params[:, 5]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f  = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    r2 = points_projX*points_projX + points_projY*points_projY
    distortion = 1. + r2 * (k1 + k2 * r2)
    points_reprojX = points_projX * distortion * f # if f is negative, points_reprojX is as well. -> negate p2d and f.
    # distortion = f + r2 * (k1 + k2 * r2)
    # points_reprojX = points_projX * distortion
    resX = (points_reprojX-p2d[:,0])
    return resX

def torchSingleResiduumY(camera_params, point_params, p2d) :
    angle_axis = camera_params[:,:3]
    points_cam = AngleAxisRotatePointT(angle_axis, point_params)
    points_cam[:,0:2] = points_cam[:,0:2] + camera_params[:, 3:5]
    points_cam[:,2] = points_cam[:,2] + camera_params[:, 5]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f  = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    r2 = points_projX*points_projX + points_projY*points_projY
    distortion = 1 + r2 * (k1 + k2 * r2)
    points_reprojY = points_projY * distortion * f
    # distortion = f + r2 * (k1 + k2 * r2)
    # points_reprojY = points_projY * distortion
    resY = (points_reprojY-p2d[:,1])
    return resY

def ComputeDerivativeMatrixInit(x0_c_, x0_l_, points_2d, camera_indices, point_indices):
    def residual_x(camera_values, landmark_values, observations):
        return torchSingleResiduumX(
            camera_values.view(-1, 9), landmark_values.view(-1, 3),
            observations.view(-1, 2))

    def residual_y(camera_values, landmark_values, observations):
        return torchSingleResiduumY(
            camera_values.view(-1, 9), landmark_values.view(-1, 3),
            observations.view(-1, 2))

    torch_cams = torch.from_numpy(x0_c_.reshape(-1,9)[camera_indices[:],:])
    torch_lands = torch.from_numpy(x0_l_.reshape(-1,3)[point_indices[:],:])
    torch_lands.requires_grad_()
    torch_cams.requires_grad_()
    torch_cams.retain_grad()
    torch_lands.retain_grad()

    torch_points_2d = torch.from_numpy(points_2d)
    torch_points_2d.requires_grad_(False)

    resX = residual_x(torch_cams, torch_lands, torch_points_2d[:,:]).flatten()
    lossX = torch.sum(resX)
    lossX.backward()

    cam_grad_x = torch_cams.grad.detach().numpy().copy()
    land_grad_x = torch_lands.grad.detach().numpy().copy()

    torch_cams.grad.zero_()
    torch_lands.grad.zero_()
    resY = residual_y(torch_cams, torch_lands, torch_points_2d[:,:]).flatten()
    lossY = torch.sum(resY)
    lossY.backward()
    cam_grad_y = torch_cams.grad.detach().numpy().copy()
    land_grad_y = torch_lands.grad.detach().numpy().copy()

    J_pose = buildMatrixNew(cam_grad_x, cam_grad_y, camera_indices, sz=9)
    J_land = buildMatrixNew(land_grad_x, land_grad_y, point_indices, sz=3)
    fx0 = buildResiduumNew(resX.detach(), resY.detach())

    return (J_pose, J_land, fx0)

# per camera print.
def print_selected_cameras(poses_, poses_v_, selected_cameras, cluster_set_few_obs_, k_clusters):
    if os.environ.get("BUNDLE_PALM_PRINT_SELECTED_CAMERAS", "0") != "1":
        return
    CRED = '\033[91m'
    CGREEN = '\033[92m'
    CEND = '\033[0m'
    for i in range(selected_cameras.shape[0]):
        for ci in range(k_clusters):
            if ci in cluster_set_few_obs_:
                print(CRED + "u cluster ", ci , ". camera ", selected_cameras[i], " ", poses_[ci][selected_cameras[i]], CEND + "")
            else:
                print("u cluster  ", ci , ". camera ", selected_cameras[i], " ", poses_[ci][selected_cameras[i]])
        print(CGREEN + "v         ", ". camera ", selected_cameras[i], " ", poses_v_[selected_cameras[i]], CEND + "")
    return

# todo: median + scale, unorm, acceleration + adjust.

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
# BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/venice/"
# FILE_NAME = "problem-52-64053-pre.txt.bz2"
# FILE_NAME = "../problem-173-111908-pre.txt.bz2" # check if compute not only in jacobian

# bug checking
# BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
# FILE_NAME = "problem-646-73584-pre.txt.bz2"
# FILE_NAME = "problem-1064-113655-pre.txt.bz2"

kClusters = 1 # todo: will still die if too many (0 in jac?)
global_iterations = 30

num_args = len(sys.argv)
if num_args not in (1, 3, 4, 5):
    raise ValueError(
        "usage: client_acc.py [BASE_URL FILE_NAME [ITERATIONS [CLUSTERS]]]")
if num_args > 2:
    print("Total arguments passed:", num_args)
    # Arguments passed
    print("\nName of Python script:", sys.argv[0], "url ", sys.argv[1], "file ", sys.argv[2])
    BASE_URL =  sys.argv[1]
    FILE_NAME = sys.argv[2]

    if num_args > 3:
        global_iterations = int(sys.argv[3])
    if num_args > 4:
        kClusters = int(sys.argv[4])

if global_iterations < 0:
    raise ValueError("global iterations must be nonnegative")
if kClusters <= 0:
    raise ValueError("cluster count must be positive")

bal_file = get_bal_file(BASE_URL, FILE_NAME)
cameras, points_3d, camera_indices, point_indices, points_2d = read_bal_data(bal_file)
initial_ray_focal = cameras[:, 6].copy()
initial_pixel_quality_metrics = evaluate_bal_state(
    cameras, points_3d, camera_indices, point_indices, points_2d)
if DRS_OBJECTIVE_MODEL == "daba_ray":
    initial_quality_metrics = evaluate_encoded_daba_ray_state(
        cameras, cameras, points_3d,
        camera_indices, point_indices, points_2d)
    initial_quality_metrics["sumSquaredError"] = (
        2.0 * initial_quality_metrics["ceresCost"])
else:
    initial_quality_metrics = initial_pixel_quality_metrics
objective_trajectory = [{
    "overallSeconds": time.perf_counter() - client_started_at,
    "sumSquaredError": initial_quality_metrics["sumSquaredError"],
    "source": "initial",
}]
n_cameras = cameras.shape[0]
n_points = points_3d.shape[0]

# simple! clustering
startL = 1
innerIts = 1 # does shit, keep at 1
LipJ = 1 # unused
global_init = True
resetIt = 0
globalBlockEigUpperLimit = 5e-1 # 1e-1, 1e-3?
blockEig_in_cluster = 5e-5 * np.ones(kClusters) # 1e-4 or 1e-5, 5e-5?
failedNesterovAcceleration = 0
maxFailedNesterovAcceleration = 3 # TODO: 2 or 3?
outer_acceleration = os.environ.get(
    "BUNDLE_PALM_DRS_ACCELERATION", "nesterov").lower()
if outer_acceleration not in {"none", "nesterov"}:
    raise ValueError(
        "BUNDLE_PALM_DRS_ACCELERATION must be 'none' or 'nesterov'")
print("input blockEig_in_cluster[ci] ", blockEig_in_cluster[0])
print("DRS camera scaling:", DRS_SCALING_METHOD)
print("DRS consensus projection metric:", DRS_CONSENSUS_METRIC)
print("DRS objective model:", DRS_OBJECTIVE_MODEL)
print(
    "DRS trial modes:",
    f"explicit_cost_landmarks={explicit_cost_landmarks}",
    f"strict_trial_safeguard={strict_trial_safeguard}",
    f"require_common_cost_match={require_common_cost_match}",
)

# Connect to the server
print("Connecting to cpp server…")
#socket = context.socket(zmq.REQ)
#socket.connect("tcp://localhost:5555")

# new idea.
push_socket = context.socket(zmq.PUSH)
#pull_socket = context.socket(zmq.REP)#.PULL)
pull_socket = context.socket(zmq.PULL)

# Bound transport failures while allowing long cluster solves to complete.
zmq_timeout_ms = int(os.environ.get("BUNDLE_PALM_ZMQ_TIMEOUT_MS", "600000"))
if zmq_timeout_ms <= 0:
    raise ValueError("BUNDLE_PALM_ZMQ_TIMEOUT_MS must be positive")
for active_socket in (push_socket, pull_socket):
    active_socket.setsockopt(zmq.RCVTIMEO, zmq_timeout_ms)
    active_socket.setsockopt(zmq.SNDTIMEO, zmq_timeout_ms)
    active_socket.setsockopt(zmq.LINGER, 0)

request_port = os.environ.get("BUNDLE_PALM_REQUEST_PORT", "5556")
result_port = os.environ.get("BUNDLE_PALM_RESULT_PORT", "5557")
push_socket.connect(f"tcp://localhost:{request_port}")
pull_socket.connect(f"tcp://localhost:{result_port}")

#lib = ctypes.CDLL("./libprocess_clusters.so")
# init_lib() # ?
partition_started_at = time.perf_counter()
clustering_mode = os.environ.get("BUNDLE_PALM_CLUSTERING", "landmark")
if clustering_mode == "landmark":
    (
        camera_indices_in_cluster,
        point_indices_in_cluster,
        points_2d_in_cluster,
        kClusters,
    ) = cluster_deg_by_landmark(
        camera_indices, points_2d, point_indices, kClusters)
elif clustering_mode == "landmark_clean":
    residual_balance_slack = float(
        os.environ.get("BUNDLE_PALM_RESIDUAL_BALANCE_SLACK", "0.02"))
    minimum_camera_landmarks = int(
        os.environ.get("BUNDLE_PALM_MIN_CAMERA_LANDMARKS", "20"))
    max_refinement_passes = int(
        os.environ.get("BUNDLE_PALM_MAX_REFINEMENT_PASSES", "2"))
    batch_repair_scans = os.environ.get(
        "BUNDLE_PALM_BATCH_REPAIR_SCANS", "0") == "1"
    repair_restart_interval = int(os.environ.get(
        "BUNDLE_PALM_REPAIR_RESTART_INTERVAL",
        "0" if batch_repair_scans else "32"))
    max_repair_work_per_phase = int(os.environ.get(
        "BUNDLE_PALM_MAX_REPAIR_WORK_PER_PHASE", "0"))
    hard_group_max_cameras = int(os.environ.get(
        "BUNDLE_PALM_HARD_GROUP_MAX_CAMERAS", "2"))
    optimize_max_camera_count = os.environ.get(
        "BUNDLE_PALM_OPTIMIZE_MAX_CAMERAS", "0") == "1"
    (
        camera_indices_in_cluster,
        point_indices_in_cluster,
        points_2d_in_cluster,
        kClusters,
    ) = cluster_by_landmark_clean(
        camera_indices, points_2d, point_indices, kClusters,
        n_cameras, n_points, residual_balance_slack,
        minimum_camera_landmarks, max_refinement_passes,
        repair_restart_interval, max_repair_work_per_phase,
        hard_group_max_cameras,
        optimize_max_camera_count)
elif clustering_mode == "landmark_scalable":
    residual_balance_slack = float(
        os.environ.get("BUNDLE_PALM_RESIDUAL_BALANCE_SLACK", "0.02"))
    minimum_camera_landmarks = int(
        os.environ.get("BUNDLE_PALM_MIN_CAMERA_LANDMARKS", "20"))
    max_refinement_passes = int(
        os.environ.get("BUNDLE_PALM_MAX_REFINEMENT_PASSES", "2"))
    (
        camera_indices_in_cluster,
        point_indices_in_cluster,
        points_2d_in_cluster,
        kClusters,
    ) = cluster_by_landmark_scalable(
        camera_indices, points_2d, point_indices, kClusters,
        n_cameras, n_points, residual_balance_slack,
        minimum_camera_landmarks, max_refinement_passes)
elif clustering_mode == "landmark_scalable_stable":
    residual_balance_slack = float(
        os.environ.get("BUNDLE_PALM_RESIDUAL_BALANCE_SLACK", "0.02"))
    minimum_camera_landmarks = int(
        os.environ.get("BUNDLE_PALM_MIN_CAMERA_LANDMARKS", "20"))
    max_refinement_passes = int(
        os.environ.get("BUNDLE_PALM_MAX_REFINEMENT_PASSES", "2"))
    (
        camera_indices_in_cluster,
        point_indices_in_cluster,
        points_2d_in_cluster,
        kClusters,
    ) = cluster_by_landmark_scalable_stable(
        camera_indices, points_2d, point_indices, kClusters,
        n_cameras, n_points, residual_balance_slack,
        minimum_camera_landmarks, max_refinement_passes)
else:
    raise ValueError(
        "BUNDLE_PALM_CLUSTERING must be 'landmark', 'landmark_clean', "
        "'landmark_scalable', or 'landmark_scalable_stable'")
partition_seconds = time.perf_counter() - partition_started_at
print("==========", clustering_mode, "clustering took", partition_seconds,
      "s ===========")

nonempty_cluster_ids = [
    ci for ci in range(kClusters)
    if camera_indices_in_cluster[ci].size > 0
]
if not nonempty_cluster_ids:
    raise RuntimeError("clustering produced no nonempty clusters")
if len(nonempty_cluster_ids) != kClusters:
    print("Dropping empty clusters:", kClusters - len(nonempty_cluster_ids))
    camera_indices_in_cluster = [
        camera_indices_in_cluster[ci] for ci in nonempty_cluster_ids]
    point_indices_in_cluster = [
        point_indices_in_cluster[ci] for ci in nonempty_cluster_ids]
    points_2d_in_cluster = [
        points_2d_in_cluster[ci] for ci in nonempty_cluster_ids]
    blockEig_in_cluster = blockEig_in_cluster[nonempty_cluster_ids]
    kClusters = len(nonempty_cluster_ids)

(local_landmark_indices_in_cluster, local_camera_indices_in_cluster) = \
    GetLocalIndices(point_indices_in_cluster, camera_indices_in_cluster)

# find cameras with < 3 observations in a cluster. print the evolvement of those cameras / also residuals?
# can do in c++?
cameras_with_few_observations = []
cluster_set_few_obs = set()
min_cam_obs = 1
for ci in range(kClusters):
    values, counts = np.unique(camera_indices_in_cluster[ci], return_counts=True)
    print(ci, ". minimum camera observations in cluster ", np.min(counts), " cams with < 5 landmarks ", np.sum(counts < 5))
    cameras_with_few_observations.append(values[counts <= min_cam_obs])
    if np.min(counts) <= min_cam_obs:
        cluster_set_few_obs.add(ci)
cameras_with_few_observations = np.unique(np.concatenate(cameras_with_few_observations))
print("cameras with less than ", min_cam_obs, " observations in any cluster ", cameras_with_few_observations)

# preconditioner?

# 1.st version, implement prox_f. extension 1. polling and threads on server side. parameter k threads, etc.
L_in_cluster = [startL for _ in range(kClusters)]
Ul_in_cluster = [0 for _ in range(kClusters)]
#blockEig_in_cluster = [1e-4 for _ in range(kClusters)]
# Unorm/Vnorm missing
# fairly stupid
poses_s_in_cluster = [cameras.copy() for _ in range(kClusters)]
poses_in_cluster = [cameras.copy() for _ in range(kClusters)]
landmarks = points_3d.copy()

prevGap = 0
lastCost = 1e12 # tem
lastCostDRE = lastCost
best_poses_v = cameras.copy()
best_landmarks = points_3d.copy()
bestCost = lastCost
bestIt = -1
bestCost60 = bestCost
bestCost30 = bestCost
tau = 1 # 2 is best ? does not generalize!
revert_lm = 0

J_pose, _, __ = ComputeDerivativeMatrixInit(cameras, points_3d, points_2d, camera_indices, point_indices)
unorm_t = GetPcgScalingDiag(J_pose.transpose() * J_pose, 0)

start = time.time() # this is not working at all. Slower then iteratively
(
    cost,
    L_in_cluster,
    Ul_in_cluster,
    poses_in_cluster,
    landmarks,
    nabla_p_in_cluster,
    blockEig_in_cluster
) = prox_f_push_pull( #prox_f(
    camera_indices_in_cluster, point_indices_in_cluster, local_camera_indices_in_cluster,
    local_landmark_indices_in_cluster, points_2d_in_cluster, poses_in_cluster, landmarks,
    poses_s_in_cluster, L_in_cluster, Ul_in_cluster, blockEig_in_cluster, kClusters,
    LipJ, innerIts_=innerIts,
    )
end = time.time() # this is not working at all. Slower then iteratively

currentCost = np.sum(cost)
print(-1, " ", round(currentCost), " gain ", round(lastCost - currentCost),
    ". ============= sum fk update takes ", end - start," s",)
#print(Ul_in_cluster)
#print(Ul_in_cluster[0])
# TODO: DO NOT UPDATE poses_v since usig bad? Ul's?
# poses_v, U_all, Up_cluster =
poses_v = best_poses_v
_, U_all, Up_cluster = average_cameras_new(camera_indices_in_cluster, poses_in_cluster,
                                                 poses_s_in_cluster, L_in_cluster, Ul_in_cluster, nabla_p_in_cluster)

#DRE cost BEFORE s update, always lower than AFTER update.
dre, dre_per_part = cost_DRE(camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, \
                            L_in_cluster, Ul_in_cluster, poses_v, nabla_p_in_cluster)

camera_transforms = None
if DRS_SCALING_METHOD == "block_jacobi_gmean":
    camera_transforms = get_camera_block_transforms(U_all)
unorm = GetPcgScalingDiag(U_all, 0)
validate_each_best = os.environ.get(
    "BUNDLE_PALM_VALIDATE_EACH_BEST", "0") == "1"
Unorm_ = diag_sparse(unorm.flatten())
relative_diff  = np.abs(Unorm_.data.flatten() - unorm_t.flatten()) / unorm_t.flatten()
relative_diff2 = np.abs(Unorm_.data.flatten()) / np.fmin(unorm_t.flatten(),Unorm_.data.flatten())
print("All badly pcg cams ", np.arange(relative_diff.shape[0]) [relative_diff > 1e1] // 9)
print("All badly pcg cams2 ", np.arange(relative_diff2.shape[0]) [relative_diff2 > 1e1] // 9)
#print(relative_diff.reshape((-1,9)))
print(np.max(relative_diff))
print(np.max( (Unorm_.data.flatten() - unorm_t.flatten()) / Unorm_.data.flatten()) )
amax  = np.argmax(relative_diff)
amax2 = np.argmax( (Unorm_.data.flatten() - unorm_t.flatten()) / Unorm_.data.flatten())
print(amax)
print(amax2)
print( (Unorm_.data.flatten())[amax], " - ", (unorm_t.flatten())[amax] )
print( (Unorm_.data.flatten())[amax2], " - ", (unorm_t.flatten())[amax2] )
print("cam1 ", amax//9, " :" , (Unorm_.data.flatten().reshape((-1,9)))[amax // 9, :]) # smaller
print("cam2 ", amax//9, " :" , (unorm_t.flatten().reshape((-1,9)))[amax // 9, :]) # larger
# print("cam1 ", 1+amax//9, " :" , (Unorm_.data.flatten().reshape((-1,9)))[1+amax // 9, :])
# print("cam2 ", 1+amax//9, " :" , (unorm_t.flatten().reshape((-1,9)))[1+amax // 9, :])
# print("cam1 ", amax//9 - 1, " :" , (Unorm_.data.flatten().reshape((-1,9)))[amax // 9 -1, :])
# print("cam2 ", amax//9 -1, " :" , (unorm_t.flatten().reshape((-1,9)))[amax // 9 -1, :])

# camMax = 645; print("cam1 ", amax//9, " :" , (Unorm_.data.flatten().reshape((-1,9)))[camMax, :]); print("cam2 ", amax//9, " :" , (unorm_t.flatten().reshape((-1,9)))[camMax, :])

#print(Unorm_.data)
#print(unorm_t)
# exit()
# use pcg from python. Differs from ceres in few? cameras
# unorm = Unorm_.data.flatten()

poses_v, poses_in_cluster, poses_s_in_cluster = preconditioning_push(poses_v, poses_in_cluster, poses_s_in_cluster,
                                                                     camera_indices_in_cluster, point_indices_in_cluster, unorm, 0, kClusters,
                                                                     camera_transforms)

poses_s_in_cluster_pre = [0 for x in range(kClusters)] # dummy fill list
for ci in range(kClusters):
    temp = Up_cluster[ci].diagonal()
    temp[temp != 0] = 1
    poses_s_in_cluster_pre[ci] = poses_s_in_cluster[ci] + tau * (poses_v - poses_in_cluster[ci]) # update s = s + v - u.
    poses_s_in_cluster[ci] = temp.reshape(-1,9) * poses_s_in_cluster[ci] # set to zero if not in cluster.
    poses_s_in_cluster_pre[ci] = temp.reshape(-1,9) * poses_s_in_cluster_pre[ci] # set to zero if not in cluster.

primal_costs_u = cost
primal_cost_u = currentCost
# primal_costs_u = primal_cost_push_pull(camera_indices_in_cluster, poses_in_cluster, kClusters)
# primal_cost_u = 0
# for ci in range(kClusters):
#     primal_cost_u += primal_costs_u[ci]
dre += primal_cost_u

primal_costs_v, primal_cost_landmarks_v = primal_cost_push_pull(
    camera_indices_in_cluster, point_indices_in_cluster, poses_v, landmarks,
    kClusters, True)
primal_cost_v = np.sum(primal_costs_v)
primal_cost_v_before = primal_cost_v
dre = max( primal_cost_v, dre ) # sandwich lemma, prevent maybe chaos
print( -1, " ======== DRE ====== ", round(dre) , " ========= gain " , \
    round(lastCostDRE - dre), "==== f(v)= ", round(primal_cost_v), " f(u)= ",
    round(primal_cost_u), " BE ", blockEig_in_cluster)

# if lastCostDRE < dre:
#     #LipJ += 0.2 * np.ones(kClusters)
#     partid = np.argmax(dre_per_part)
#     blockEig_in_cluster[partid] = np.minimum(blockEig_in_cluster[partid] * np.sqrt(2), 1e-0)
# # if f(u) < f(v) also raise? not lip but smth else.

lastCost = currentCost
lastCostDRE = dre
search_direction = [0 for x in range(kClusters)] # dummy fill list
poses_s_in_cluster_bfgs = [0 for x in range(kClusters)] # dummy fill list
lastCostDRE_bfgs = lastCostDRE
restartIteration = 0
if primal_cost_v < bestCost:
    best_poses_v = poses_v.copy()
    best_landmarks = primal_cost_landmarks_v.copy()
    bestCost = primal_cost_v
    bestCost60 = bestCost
    bestCost30 = bestCost
    best_cost_found_push_pull(
        point_indices_in_cluster, kClusters, primal_costs_v,
        primal_cost_landmarks_v)
    objective_trajectory.append({
        "overallSeconds": time.perf_counter() - client_started_at,
        "sumSquaredError": float(bestCost),
        "source": "initial_prox",
    })

# init state
print_selected_cameras(poses_in_cluster, best_poses_v, cameras_with_few_observations, cluster_set_few_obs, kClusters)

# Nesterov history is flattened across all cluster-camera parameter blocks.
outer_transport_bytes_sent_start = transport_bytes_sent
outer_transport_bytes_received_start = transport_bytes_received
acceleration_state_size = kClusters * 9 * n_cameras
s_prev = np.zeros(acceleration_state_size)
delta_s_old_ = np.zeros(acceleration_state_size)
prev_dk = np.zeros(acceleration_state_size)
for global_iteration in range(global_iterations):

    delta_s  = np.zeros(kClusters * 9 * n_cameras)
    s_new = np.zeros(kClusters * 9 * n_cameras)
    s_cur  = np.zeros(kClusters * 9 * n_cameras)
    for ci in range(kClusters):
        #delta_s1[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = (poses_v - poses_in_cluster[ci]).flatten()
        delta_s [ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = (poses_v - poses_in_cluster[ci]).flatten()
        s_new[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = poses_s_in_cluster_pre[ci].flatten()
        s_cur[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = poses_s_in_cluster[ci].flatten()

    if global_iteration <= 0: # s_prev is known
        dk = s_new - s_cur #+ delta_s
        lambda_0 = 1.0
        lambda_1 = 1.0
    else:
        delta_s_ = s_new - s_prev
        dk = s_new - s_cur + delta_s_
        if global_iteration > 1:
            dk = delta_s_ + delta_s_old_ # last 3 + 2nd step (so 2nd twice).

        delta_s_old_ = delta_s_.copy()

        # momentum simple, same for v? about same
        beta_nesterov = (global_iteration-resetIt-1) / (global_iteration-resetIt+2) # 0.7
        dk = s_new - s_cur + beta_nesterov * prev_dk
        if False: # conventional nesterov
            lambda_1 = (1. + np.sqrt(1. + 4. * lambda_0**2)) / 2.
            gamma = (lambda_0 - 1.) / lambda_1
            dk = s_new - s_cur + gamma * (s_new - s_cur)
            lambda_0 = lambda_1

    prev_dk = dk.copy()
    dk_stepLength = np.linalg.norm(dk, 2)
    for ci in range(kClusters):
        search_direction[ci] = dk[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras].reshape(n_cameras, 9)
    s_prev = s_cur.copy() # access to old s.

    #line_search_iterations = 1 # is pure DRS (forced, see below set tk == 1)
    line_search_iterations = 2 if outer_acceleration == "nesterov" else 1
    if global_iteration <= resetIt + 1:
        line_search_iterations = 1

    # print(" bfgs step ", dk_stepLength, " ratio ", file=sys.stderr )
    for ls_it in range(line_search_iterations):
        if line_search_iterations > 1:
            tk = ls_it / (line_search_iterations-1)
        else:
            tk = 1 # 0: line-search 1: drs
        for ci in range(kClusters):
            poses_s_in_cluster_bfgs[ci] = tk * poses_s_in_cluster_pre[ci] + (1-tk) * (poses_s_in_cluster[ci] + search_direction[ci])

        if True: # linearize at v / average solution, same issue I suppose. Yes. solution is too return the new gradient, s.t. update of v is wrt to current situation.
            poses_in_cluster_bfgs = [elem.copy() for elem in poses_in_cluster]
        else: # does not work well here.
            poses_in_cluster_bfgs = [poses_v.copy() for _ in range(kClusters)]

        # if revert_lm == 2:
        #     print("----------------------------------------------")
        #     for ci in range(kClusters):
        #         print(ci, " best ps   bfgs: ", poses_in_cluster_bfgs[ci][0:2])
        #         print(ci, " best ps-s bfgs: ", poses_s_in_cluster_bfgs[ci][0:2])
        #     print("bestlm: ", landmarks[0:18])
        #     print("----------------------------------------------")

        # print(poses_s_in_cluster_pre[0].flatten()[0:10], " ==? " , poses_in_cluster_bfgs[0].flatten()[0:10], " vs " , poses_s_in_cluster[0].flatten()[0:10] )
        if ls_it > 0:
            revert_lm = 1 # revert landmark to last step (as the pose as well)
            print("-------------------- revert_lm ", revert_lm, "--------------------")

        L_in_cluster_bfgs = L_in_cluster.copy()
        Ul_in_cluster_bfgs = [elem.copy() for elem in Ul_in_cluster]
        blockEig_in_cluster_bfgs = [elem for elem in blockEig_in_cluster]
        (
            cost_bfgs,
            L_in_cluster_bfgs,
            Ul_in_cluster_bfgs,
            poses_in_cluster_bfgs,
            landmarks_bfgs,
            nabla_p_in_cluster_bfgs,
            blockEig_in_cluster_bfgs
        ) = prox_f_push_pull( # revert_lm
            camera_indices_in_cluster, point_indices_in_cluster, local_camera_indices_in_cluster,
            local_landmark_indices_in_cluster, points_2d_in_cluster, poses_in_cluster_bfgs, landmarks.copy(),
            poses_s_in_cluster_bfgs, L_in_cluster_bfgs, Ul_in_cluster_bfgs, blockEig_in_cluster_bfgs, kClusters,
            LipJ, innerIts_=innerIts, revert_lm = revert_lm)

        #print("2. x0_p", "points_3d_in_cluster", points_3d_in_cluster)
        currentCost_bfgs = np.sum(cost_bfgs)
        poses_v_bfgs, Ul_all_bfgs, U_cluster_zeros = average_cameras_new(
            camera_indices_in_cluster, poses_in_cluster_bfgs, poses_s_in_cluster_bfgs, L_in_cluster_bfgs, Ul_in_cluster_bfgs, nabla_p_in_cluster_bfgs)

        # eval cost
        dre_bfgs, dre_per_part = cost_DRE(camera_indices_in_cluster, poses_in_cluster_bfgs,
            poses_s_in_cluster_bfgs, L_in_cluster_bfgs, Ul_in_cluster_bfgs, poses_v_bfgs, nabla_p_in_cluster_bfgs)
        dre_bfgs += currentCost_bfgs

        # debugging cost block ################
        primal_cost_v_costs, primal_cost_landmarks_v = primal_cost_push_pull(
            camera_indices_in_cluster, point_indices_in_cluster,
            poses_v_bfgs, landmarks_bfgs, kClusters, True)
        primal_cost_v = np.sum(primal_cost_v_costs)
        primal_cost_v_all = [rounded_cost(cost) for cost in primal_cost_v_costs]

        #primal_cost_u_all = []
        #primal_cost_u_all = primal_cost_push_pull(camera_indices_in_cluster, poses_in_cluster_bfgs, kClusters, False)
        primal_cost_u_all = cost_bfgs
        primal_cost_u = np.sum(primal_cost_u_all)
        if currentCost_bfgs != primal_cost_u:
            print("Costs do not match line 740 ", currentCost_bfgs, " u:" , primal_cost_u, " v:", primal_cost_v)
        primal_cost_u_all = [rounded_cost(cost) for cost in primal_cost_u_all]
        ###

        dre_bfgs = max(dre_bfgs, primal_cost_v) # sandwich lemma
        blockEigLastIt = blockEig_in_cluster
        #blockEigLastIt = getBlockEigUsed() # the actual used not the one written into memory or whatever blockEig_in_cluster_bfgs is.
        dre_gain = float(lastCostDRE_bfgs - dre_bfgs)
        primal_gap = float(primal_cost_v - primal_cost_u)
        diffToGain = max(primal_gap - dre_gain, 0.) / float(primal_cost_u)
        gapToGain = max(primal_gap - dre_gain, 1.) / max(dre_gain, 1.)
        currentGap = max(primal_gap, 1.) #- round(lastCostDRE_bfgs - dre_bfgs), 1) # not sure ..
        differentialGap = prevGap - currentGap
        costGain = lastCostDRE_bfgs - dre_bfgs
        gain_to_gap = costGain / currentGap
        G2C = (
            round(1000 * gain_to_gap) / 1000
            if np.isfinite(gain_to_gap) else gain_to_gap)
        print( global_iteration, "/", ls_it, " ======== DRE BFGS ====== ", rounded_cost(dre_bfgs) , " ========= gain " , \
            rounded_cost(costGain), "==== f(v)= ", rounded_cost(primal_cost_v), " f(u)= ", rounded_cost(primal_cost_u),
            " G ", currentGap , " dG ", differentialGap, " ", differentialGap / np.maximum(costGain, 1.), #" D2G ", diffToGain, "G2G ", gapToGain, 
            " G2C ", G2C, " BE ", blockEigLastIt[0]) #, " L ", L_in_cluster_bfgs) #blockEig_in_cluster_bfgs)
        print( global_iteration, "/", ls_it, " f(v) = ", primal_cost_v_all, " f(u) = ", primal_cost_u_all)
        prevGap = currentGap

        if primal_cost_v < bestCost:
            if validate_each_best:
                candidate_cameras = physical_cameras_from_preconditioned(
                    poses_v_bfgs, unorm, camera_transforms, kClusters)
                if DRS_OBJECTIVE_MODEL == "daba_ray":
                    prox_candidate_metrics = evaluate_encoded_daba_ray_state(
                        cameras, candidate_cameras, landmarks_bfgs,
                        camera_indices, point_indices, points_2d)
                    candidate_metrics = evaluate_encoded_daba_ray_state(
                        cameras, candidate_cameras, primal_cost_landmarks_v,
                        camera_indices, point_indices, points_2d)
                    candidate_metrics["sumSquaredError"] = (
                        2.0 * candidate_metrics["ceresCost"])
                    prox_candidate_metrics["sumSquaredError"] = (
                        2.0 * prox_candidate_metrics["ceresCost"])
                else:
                    prox_candidate_metrics = evaluate_bal_state(
                        candidate_cameras, landmarks_bfgs,
                        camera_indices, point_indices, points_2d)
                    candidate_metrics = evaluate_bal_state(
                        candidate_cameras, primal_cost_landmarks_v,
                        camera_indices, point_indices, points_2d)
                candidate_relative_error = abs(
                    candidate_metrics["sumSquaredError"] - primal_cost_v
                ) / max(1.0, abs(primal_cost_v))
                if candidate_relative_error > 1e-6:
                    raise RuntimeError(
                        "candidate state does not match native DRS cost at "
                        f"iteration {global_iteration}: "
                        f'{candidate_metrics["sumSquaredError"]} versus '
                        f"{primal_cost_v} (relative error "
                        f"{candidate_relative_error}; prox landmarks give "
                        f'{prox_candidate_metrics["sumSquaredError"]})')
            best_poses_v = poses_v_bfgs.copy()
            best_landmarks = primal_cost_landmarks_v.copy()
            bestCost = primal_cost_v
            bestIt = global_iteration
            best_cost_found_push_pull(
                point_indices_in_cluster, kClusters, primal_cost_v_costs,
                primal_cost_landmarks_v)
            objective_trajectory.append({
                "overallSeconds": time.perf_counter() - client_started_at,
                "sumSquaredError": float(bestCost),
                "source": "outer_iteration",
                "iteration": global_iteration,
            })
            # send best proto
        if global_iteration < 60:
            bestCost60 = bestCost
        if global_iteration < 30:
            bestCost30 = bestCost

        # TODO: inc blockEig_in_cluster[cluster_id] *=2 -- if possible u/v
        beMin = np.minimum(globalBlockEigUpperLimit, np.min(blockEig_in_cluster))

        iteration_factor      = (1 - (global_iteration / global_iterations))**4 # 1 at start, ~0 at end.
        reference_iteration = min(5, global_iterations - 1)
        iteration_factor_five = (1 - (reference_iteration / global_iterations))**4 #, could also running mean of gains and use 5% of those (positive gains, if < - (5% of mean) ).
        maxPct = 1 + 0.01 * iteration_factor / iteration_factor_five # aim at 1% at 5 iterations?

        disable_best_pose = False
        if disable_best_pose:
            best_poses_v = poses_v.copy()
            best_landmarks = landmarks.copy()

        # idea accept if primal v cost is very close.
        # can happen that best primal cost is about same as current and dre was set to this as correction.
        # TODO if dre < primal_v also increase LipJ or so.
        # recompute if we will reject.  this just reproduces best cost: yes. Can be removed / replacing with bestCost if no new idea here.
        # todo: in original likely a bug is making this necessary.
        if (beMin < globalBlockEigUpperLimit) and (ls_it == line_search_iterations-1) and (maxPct * lastCostDRE_bfgs < dre_bfgs):
            # best_landmarks not present. use poses_v_bfgs. This should be the best solution? last solution?
            # best. problem lms not reset here. cost should also able to revert lms.
            # this is fishy. i want cost with best lms, but only 'once'. revert 1 or 2 or 0 is possible.
            # eval with best or last? set back after.
            primal_cost_v_before = bestCost
            # primal_cost_v_before_ = primal_cost_push_pull(camera_indices_in_cluster, best_poses_v, kClusters, True, True)
            # # primal_cost_v_before = primal_cost_push_pull(camera_indices_in_cluster, poses_v_bfgs, kClusters, True)
            # print("Check ================= New primal_cost_v_before ", primal_cost_v_before, " vs. ", np.sum(primal_cost_v_before_), " vs ", primal_cost_v , " vs ", bestCost)
            # primal_cost_v_before = np.sum(primal_cost_v_before_)

        # Reset acceleration if fails 6 times in a row
        if ls_it == line_search_iterations - 1 and line_search_iterations > 1:
            failedNesterovAcceleration += 1
            lambda_0 = np.maximum(1., lambda_1 / 2.) # reset acceleration
            if False:
                #lambda_0 = np.maximum(1., lambda_1 / np.sqrt(5)) # reset acceleration, less flickering can be worse results.
                prev_dk = s_new - s_cur
                print('lambda_0 reset ', lambda_0)

            if failedNesterovAcceleration >= maxFailedNesterovAcceleration:
                prev_dk = 0 * prev_dk
                resetIt = global_iteration
                failedNesterovAcceleration = 0
                restartIteration = global_iteration # reset RNA has no effect.
                print("Reset Nesterov acceleration after ", maxFailedNesterovAcceleration, " consecutive failures.")

        maxPctV = np.maximum(1.001, np.sqrt(maxPct)) # max 0.1 % AAA
        nonfinite_trial = (
            not np.isfinite(dre_bfgs) or not np.isfinite(primal_cost_v))
        if strict_trial_safeguard:
            reject_trial = should_reject_trial(
                ls_it,
                line_search_iterations,
                dre_bfgs,
                primal_cost_v,
                lastCostDRE_bfgs,
                primal_cost_v_before,
                maxPct,
                maxPctV,
            )
        else:
            reject_trial = (
                ls_it == line_search_iterations - 1
                and (
                    nonfinite_trial
                    or (
                        beMin < globalBlockEigUpperLimit
                        and dre_bfgs > maxPct * lastCostDRE_bfgs
                        and primal_cost_v > maxPctV * primal_cost_v_before
                    )
                )
            )
        if reject_trial:
            print("Rejected is primal cost (v) bad or what", primal_cost_v, " > ", maxPctV * primal_cost_v_before, " > ", primal_cost_v_before, " * ", maxPctV)
            print("Rejected is dre cost (v) bad or what", dre_bfgs, " > ", maxPct * lastCostDRE_bfgs, " > ", lastCostDRE_bfgs, " * ", maxPct)

            print_selected_cameras(poses_in_cluster_bfgs, poses_v_bfgs, cameras_with_few_observations, cluster_set_few_obs, kClusters)

            # A rejected final trial resets every local copy and its landmarks
            # to the state associated with the best global primal cost.
            poses_in_cluster = [best_poses_v.copy() for _ in poses_in_cluster]
            for ci in range(kClusters):
                poses_s_in_cluster_pre[ci] = best_poses_v.copy() # s + u-v = s in this case, do we use the best s?
                poses_s_in_cluster[ci] = best_poses_v.copy()
            landmarks = best_landmarks.copy()
            revert_lm = 2 # revert to best landmark / pose.
            print("-------------------- revert_lm ", revert_lm, "--------------------")
            # TODO: s-> best_v & u=v after reset ? landmark match best v? -- we can/could compute lms from v only: yes: VLi * Jl * res, poses fixed.

            # what are the best poses and lms? What does the model think. This is so weird always off.
            # for ci in range(kClusters):
            #     print(ci, " bestps: ", poses_in_cluster[ci][0:2])
            #     print(ci, " bestps: ", poses_s_in_cluster[ci][0:2])
            # print("bestlm: ", landmarks[0:18])

            # IDEA: verify cost here.
            CheckCost = True # temporal test. it appears odd that this is so bad. Maybe set be very strict for one iteration?
            if CheckCost:
                primal_cost_v_check, _ = primal_cost_push_pull(
                    camera_indices_in_cluster, point_indices_in_cluster,
                    poses_in_cluster, landmarks, kClusters)
                primal_cost_v_check = np.sum(primal_cost_v_check)
                print(
                    "Checking cost after reset: f(v_best)=",
                    rounded_cost(primal_cost_v_check),
                    " vs. f(v)=", rounded_cost(primal_cost_v),
                    " vs. f(u)=", rounded_cost(primal_cost_u))

            ############
            # VERSION U is doing nothing actually. This is differetn if taking actula steps as below.
            # poses_s_in_cluster = [elem.copy() for elem in poses_s_in_cluster_pre]
            # poses_in_cluster_test = [elem.copy() for elem in poses_in_cluster]
            ############
            # TODO: mult in dependence on GAP / cost jump.
            be_mult = 2 # if we have *= sqrt idea below this can be lower?
            # be_mult = 4 #for JtJ? CCC # TODO: 4 was best, try to do 3 here.
            for ci in range(kClusters):
                blockEig_in_cluster[ci] = np.minimum(blockEig_in_cluster[ci] * be_mult, globalBlockEigUpperLimit)
            print("Be *= ", be_mult, " -> Be= ", blockEig_in_cluster, " LipJ " , np.mean(LipJ))

            # TODO: equalize / reset nesterov(acceleration) here.
            AlsoResetNesterovAcceleration = True # test on 646, 1266, 1064, 961, 427, 1778 -> no conclusion.
            if AlsoResetNesterovAcceleration:
                prev_dk = 0 * prev_dk
                resetIt = global_iteration
                print("Reset Nesterov acceleration after ", failedNesterovAcceleration, " failures.")
                failedNesterovAcceleration = 0
            lastCostDRE_bfgs = (
                bestCost
                if strict_trial_safeguard or nonfinite_trial
                else dre_bfgs)
            # THIS AVOIDS INSTANT REJECTION OF NEXT STEP (resetting this best dre cost) unless very bad result in next step.

            print(" ************** REVERTED iteration **************, DRE cost set to ", lastCostDRE_bfgs, \
                " before ", primal_cost_v_before, " min LipJ", np.min(LipJ))

        else: # normal case

            print("Not rejected primal cost (v) not bad or what", primal_cost_v, " > ", maxPctV * primal_cost_v_before, " > ", primal_cost_v_before, " * ", maxPctV)
            print("Not rejected dre cost (v) not bad or what", dre_bfgs, " > ", maxPct * lastCostDRE_bfgs, " > ", lastCostDRE_bfgs, " * ", maxPct)

            # differentialGap / np.maximum(costGain, 1)
            # if diffToGain > 0.2: CCC: turn off for JtJ stepsize?
            # gets ignored mostly if we enter condition below.
            if costGain < 0 and differentialGap < 0 and currentGap > 1 and (ls_it == line_search_iterations-1): # gap present fv - fu >0, gets wider and cost higher than best
                be_mult__ = np.sqrt(2)
                for ci in range(kClusters):
                    blockEig_in_cluster[ci] = np.minimum(blockEig_in_cluster[ci] * be_mult__, globalBlockEigUpperLimit)
                print("Be *= ", be_mult__, " Be ", blockEig_in_cluster, " ", np.mean(LipJ))

            if dre_bfgs <= lastCostDRE_bfgs or 10000 * (dre_bfgs-lastCostDRE_bfgs) <= lastCostDRE_bfgs \
                or (ls_it == line_search_iterations-1 and line_search_iterations > 1): # not correct yet, must be <= last - c/gamma |u-v|

                for ci in range(kClusters):
                    poses_s_in_cluster[ci] = poses_s_in_cluster_bfgs[ci].copy()
                    s_step_cluster = poses_v_bfgs - poses_in_cluster_bfgs[ci]
                    poses_s_in_cluster_pre[ci] = poses_s_in_cluster[ci] + tau * s_step_cluster # update s = s + v - u.

                for ci in range(kClusters):
                    poses_in_cluster[ci] = poses_in_cluster_bfgs[ci].copy()
                    # Ul_in_cluster[ci] = Ul_in_cluster_bfgs[ci].copy() # unused
                    blockEig_in_cluster[ci] = blockEig_in_cluster_bfgs[ci] # why, nullifies times sqrt 2 above.

                # here poses_s_in_cluster_pre == poses_in_cluster_bfgs
                # single cluster v=u step =0?
                # print("Accept: ", poses_s_in_cluster_pre[0].flatten()[0:10], " ==? " ,
                #       poses_in_cluster_bfgs[0].flatten()[0:10], " vs " , poses_s_in_cluster[0].flatten()[0:10],
                #       " step ", s_step_cluster.flatten()[0:10], " " )

                L_in_cluster = L_in_cluster_bfgs.copy()
                landmarks = landmarks_bfgs.copy()
                lastCostDRE_bfgs = dre_bfgs.copy()
                poses_v = poses_v_bfgs.copy()

                if ls_it != line_search_iterations-1:
                    #print("Reset counter after ", failedNesterovAcceleration , " failed acceleration steps")
                    failedNesterovAcceleration = np.maximum(0, failedNesterovAcceleration - 1) # success, reset counter / reduce counter TODO: reset more often 2 fails?
                revert_lm = 0 # normal case accept

                break # next full iteration

overall_seconds = time.perf_counter() - client_started_at
outer_transport_bytes_sent = (
    transport_bytes_sent - outer_transport_bytes_sent_start)
outer_transport_bytes_received = (
    transport_bytes_received - outer_transport_bytes_received_start)
best_physical_cameras = physical_cameras_from_preconditioned(
    best_poses_v, unorm, camera_transforms, kClusters)
pixel_cross_metrics = None
if DRS_OBJECTIVE_MODEL == "daba_ray":
    quality_metrics = evaluate_encoded_daba_ray_state(
        cameras, best_physical_cameras, best_landmarks,
        camera_indices, point_indices, points_2d)
    quality_metrics["sumSquaredError"] = 2.0 * quality_metrics["ceresCost"]
    daba_cameras, daba_points = encoded_daba_ray_state_to_matrix(
        cameras, best_physical_cameras, best_landmarks)
    pixel_cross_metrics = evaluate_daba_state_pixel_error(
        cameras, daba_cameras, daba_points,
        camera_indices, point_indices, points_2d)
else:
    quality_metrics = evaluate_bal_state(
        best_physical_cameras,
        best_landmarks,
        camera_indices,
        point_indices,
        points_2d,
    )
native_cost_relative_error = abs(
    quality_metrics["sumSquaredError"] - bestCost
) / max(
    1.0, abs(bestCost))
if require_common_cost_match and native_cost_relative_error > 1e-6:
    raise RuntimeError(
        "common evaluator cost does not match native DRS cost: "
        f'{quality_metrics["sumSquaredError"]} versus {bestCost} '
        f"(relative error {native_cost_relative_error})")
objective_trajectory.append({
    "overallSeconds": overall_seconds,
    "sumSquaredError": quality_metrics["sumSquaredError"],
    "source": "final_common_evaluator",
})
result_dict = {"base_url": BASE_URL, "file_name": FILE_NAME, "iterations" : global_iterations, \
            "bestCost" : round(bestCost), "bestIt": bestIt, "kClusters" : kClusters, \
            "bestCost60" : round(bestCost60), "bestCost30" : round(bestCost30), \
            "status": "completed", "accelerator": "nesterov", \
            "drsScaling": DRS_SCALING_METHOD, \
            "drsConsensusMetric": DRS_CONSENSUS_METRIC, \
            "objectiveModel": DRS_OBJECTIVE_MODEL, \
            "partitionSeconds": partition_seconds, \
            "overallSeconds": overall_seconds, \
            "qualityMetrics": quality_metrics, \
            "pixelCrossMetrics": pixel_cross_metrics, \
            "initialQualityMetrics": initial_quality_metrics, \
            "initialPixelQualityMetrics": initial_pixel_quality_metrics, \
            "objectiveTrajectory": objective_trajectory, \
            "transportBytesSent": transport_bytes_sent, \
            "transportBytesReceived": transport_bytes_received, \
            "outerTransportBytesSent": outer_transport_bytes_sent, \
            "outerTransportBytesReceived": outer_transport_bytes_received, \
            "outerTransportBytesPerIteration": \
                (outer_transport_bytes_sent + outer_transport_bytes_received) \
                / max(1, global_iterations), \
            "explicitCostLandmarks": explicit_cost_landmarks, \
            "strictTrialSafeguard": strict_trial_safeguard, \
            "nativeCostRelativeError": native_cost_relative_error }
results_file = os.environ.get("BUNDLE_PALM_RESULTS_FILE", "results_server.json")
state_file = os.environ.get("BUNDLE_PALM_STATE_FILE")
if state_file:
    save_bal_state(
        state_file,
        best_physical_cameras,
        best_landmarks,
        {
            "base_url": BASE_URL,
            "file_name": FILE_NAME,
            "solver": "drs",
            "objective_model": DRS_OBJECTIVE_MODEL,
            "iterations": global_iterations,
            "clusters": kClusters,
            "best_iteration": bestIt,
            "quality_metrics": quality_metrics,
        },
    )
    result_dict["stateFile"] = str(Path(state_file).resolve())
with open(results_file, 'a') as json_file:
    json.dump(result_dict, json_file)
    json_file.write('\n')
