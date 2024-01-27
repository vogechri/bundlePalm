from __future__ import print_function
from termios import CINTR
import urllib
import bz2
import os
import time
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_array, csr_matrix, issparse
from scipy.sparse import diags as diag_sparse
from scipy.sparse.linalg import inv as inv_sparse
from numpy.linalg import pinv as inv_dense
from numpy.linalg import eigvalsh
# idea reimplement projection with torch to get a jacobian -> numpy then
import torch
import math
import ctypes
from torch.autograd.functional import jacobian
from torch import tensor, from_numpy

# look at website. This is the smallest problem. guess: pytoch cpu is pure python?
BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
#FILE_NAME = "problem-73-11032-pre.txt.bz2"

#BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/dubrovnik/"
# FILE_NAME = "problem-16-22106-pre.txt.bz2"
#FILE_NAME = "problem-356-226730-pre.txt.bz2" # large dub, play with ideas: cover, etc
#FILE_NAME = "problem-237-154414-pre.txt.bz2"
#FILE_NAME = "problem-173-111908-pre.txt.bz2"
#FILE_NAME = "problem-135-90642-pre.txt.bz2"

#BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/trafalgar/"
# FILE_NAME = "problem-21-11315-pre.txt.bz2"
#FILE_NAME = "problem-257-65132-pre.txt.bz2"

URL = BASE_URL + FILE_NAME

if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)

def remove_large_points(points_3d, camera_indices, points_2d, point_indices):
    remove_ids = np.arange(points_3d.shape[0])[np.sum(points_3d**2, 1) > 1e6]
    if remove_ids.shape[0] >0:
        points_3d = np.delete(points_3d, remove_ids, axis=0)
        num_all_res = camera_indices.shape[0]
        res_remove_ids = np.isin(point_indices, remove_ids)
        camera_indices = camera_indices[~res_remove_ids]
        points_2d = points_2d[~res_remove_ids]
        point_indices = point_indices[~res_remove_ids]
        unique_numbers = np.unique(point_indices)
        # Step 2: Create a dictionary for mapping
        mapping = {number: i for i, number in enumerate(unique_numbers)}    
        # Step 3: Apply the mapping to the array
        vfunc = np.vectorize(mapping.get)
        point_indices = vfunc(point_indices)
        print("Removed ", remove_ids.shape[0], " points")
        # alot points far away. so likely present in many parts.
        print("Removed ", num_all_res - camera_indices.shape[0], " residuals, ", \
            (num_all_res - camera_indices.shape[0]) / remove_ids.shape[0], " observations in removed landmarks")
        print(np.max(point_indices))
        print(points_3d.shape)
    return points_3d, camera_indices, points_2d, point_indices

def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras_, n_points_, n_observations = map(int, file.readline().split())

        camera_indices_ = np.empty(n_observations, dtype=int)
        point_indices_ = np.empty(n_observations, dtype=int)
        points_2d_ = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices_[i] = int(camera_index)
            point_indices_[i] = int(point_index)
            points_2d_[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras_ * 9)
        for i in range(n_cameras_ * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras_, -1))

        points_3d_ = np.empty(n_points_ * 3)
        for i in range(n_points_ * 3):
            points_3d_[i] = float(file.readline())
        points_3d_ = points_3d_.reshape((n_points_, -1))

    # currently must do for drs. turn off to fix issues? better debug
    (points_3d_, camera_indices_, points_2d_, point_indices_) = \
        remove_large_points(points_3d_, camera_indices_, points_2d_, point_indices_)

    return camera_params, points_3d_, camera_indices_, point_indices_, points_2d_

def fillPythonVec(out, sizes_out, kClusters):
    ret = []
    start = 0
    for i__ in range (kClusters):
        tmp = []
        k = 0
        #print("lib.vector_get(sizes_out, i) ", i__, " : ", lib.vector_get(sizes_out, i__))
        for j in range(lib.vector_get(sizes_out, i__)):
            tmp.append(lib.vector_get(out, start+j))
            k += 1
        start += k
        ret.append(np.array(tmp))
        #print(i__, " fillPythonVec ", len(ret), " ", ret[i__].shape)
    return ret

def fillPythonVecSimple(out):
    tmp = []
    for j in range(lib.vector_size(out)):
        tmp.append(lib.vector_get(out, j))
    return np.array(tmp)

def init_lib():
    lib.new_vector.restype = ctypes.c_void_p
    lib.new_vector.argtypes = None
    lib.new_vector_of_size.restype = ctypes.c_void_p
    lib.new_vector_of_size.argtypes = [ctypes.c_int]
    lib.vector_set.restype = None
    lib.vector_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.vector_get.restype = ctypes.c_int
    lib.vector_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.vector_size.restype = ctypes.c_int
    lib.vector_size.argtypes = [ctypes.c_void_p]
    lib.new_vector_by_copy.restype = ctypes.c_void_p
    lib.new_vector_by_copy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.delete_vector.restype = None
    lib.delete_vector.argtypes = [ctypes.c_void_p]
    lib.process_clusters_test.restype = None

    lib.process_clusters.restype = None #[ctypes.c_void_p, ctypes.c_void_p]
    lib.process_clusters.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p, # out:
                                    ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p]

    lib.cluster_covis.restype = None
    lib.cluster_covis.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p] # out]


def cluster_covis_lib(kClusters, camera_indices__, point_indices__):
    c_kClusters_ = ctypes.c_int(kClusters)
    pre_merges_ = 0
    c_pre_merges_ = ctypes.c_int(pre_merges_)
    c_max_vol_part = 4
    c_max_vol_part_ = ctypes.c_int(c_max_vol_part)

    camera_indices_list = camera_indices__.tolist()
    point_indices_list = point_indices__.tolist()

    #c_point_indices_ptr = (ctypes.c_int * len(point_indices__))(*point_indices__)
    c_point_indices_ptr = (ctypes.c_int * len(point_indices_list))(*point_indices_list)
    c_point_indices_cpp = lib.new_vector_by_copy(c_point_indices_ptr, len(c_point_indices_ptr))
    c_cam_indices_ptr = (ctypes.c_int * len(camera_indices_list))(*camera_indices_list)
    c_cam_indices_cpp = lib.new_vector_by_copy(c_cam_indices_ptr, len(c_cam_indices_ptr))

    res_to_cluster_c_out = lib.new_vector()
    res_to_cluster_c_sizes = lib.new_vector_of_size(kClusters)

    old_vtxsToPart_ = 0
    if (isinstance(old_vtxsToPart_, list)):
        c_old_vtxsToPart_ptr = (ctypes.c_int * len(old_vtxsToPart_))(*old_vtxsToPart_)
        old_vtxsToPart_cpp = lib.new_vector_by_copy(c_old_vtxsToPart_ptr, len(c_old_vtxsToPart_ptr))
    else:
        old_vtxsToPart_cpp = lib.new_vector()

    lib.cluster_covis(c_kClusters_, c_pre_merges_, c_max_vol_part_, c_cam_indices_cpp, c_point_indices_cpp, res_to_cluster_c_out, res_to_cluster_c_sizes, old_vtxsToPart_cpp)

    #old_vtxsToPart_ = fillPythonVecSimple(old_vtxsToPart_cpp).tolist()
    kClusters = lib.vector_size(res_to_cluster_c_sizes)

    res_indices_in_cluster__ = fillPythonVec(res_to_cluster_c_out, res_to_cluster_c_sizes, kClusters)
    return res_indices_in_cluster__, kClusters
    # copy data, free c++ mem

def process_cluster_lib(num_lands_, num_res_, kClusters__, point_indices_in_cluster__, res_indices_in_cluster__, point_indices__):
    # Flatten the nested lists and get the sizes of the sublists
    point_indices_in_cluster_flat = [item for sublist in point_indices_in_cluster__ for item in sublist]
    point_indices_in_cluster_sizes = [len(sublist) for sublist in point_indices_in_cluster__]

    res_indices_in_cluster_flat = [item for sublist in res_indices_in_cluster__ for item in sublist]
    res_indices_in_cluster_sizes = [len(sublist) for sublist in res_indices_in_cluster__]

    # Convert the input arguments to C types
    c_num_lands_ = ctypes.c_int(num_lands_)
    c_num_res_ = ctypes.c_int(num_res_)
    c_kClusters_ = ctypes.c_int(kClusters__)

    c_point_indices_in_cluster_flat_ptr = (ctypes.c_int * len(point_indices_in_cluster_flat))(*point_indices_in_cluster_flat)
    c_point_indices_in_cluster_sizes_ptr = (ctypes.c_int * len(point_indices_in_cluster_sizes))(*point_indices_in_cluster_sizes)

    c_point_indices_ptr = (ctypes.c_int * len(point_indices__))(*point_indices__)

    c_res_indices_in_cluster_flat_ptr = (ctypes.c_int * len(res_indices_in_cluster_flat))(*res_indices_in_cluster_flat)
    c_res_indices_in_cluster_sizes_ptr = (ctypes.c_int * len(res_indices_in_cluster_sizes))(*res_indices_in_cluster_sizes)

    #c_res_indices_in_cluster_flat = lib.new_vector_of_size(len(c_res_indices_in_cluster_flat))
    c_point_indices_in_cluster_flat_cpp = lib.new_vector_by_copy(c_point_indices_in_cluster_flat_ptr, len(c_point_indices_in_cluster_flat_ptr))
    c_point_indices_in_cluster_sizes_cpp = lib.new_vector_by_copy(c_point_indices_in_cluster_sizes_ptr, len(c_point_indices_in_cluster_sizes_ptr))
    c_point_indices_cpp = lib.new_vector_by_copy(c_point_indices_ptr, len(c_point_indices_ptr))
    c_res_indices_in_cluster_flat_cpp = lib.new_vector_by_copy(c_res_indices_in_cluster_flat_ptr, len(c_res_indices_in_cluster_flat_ptr))
    c_res_indices_in_cluster_sizes_cpp = lib.new_vector_by_copy(c_res_indices_in_cluster_sizes_ptr, len(c_res_indices_in_cluster_sizes_ptr))

    #lib.vector_set(c_res_indices_in_cluster_flat, i, value)

    res_toadd_out = lib.new_vector()
    res_toadd_sizes_out = lib.new_vector_of_size(kClusters__)

    point_indices_already_covered_out = lib.new_vector_of_size(kClusters__)
    point_indices_already_covered_sizes = lib.new_vector_of_size(kClusters__)

    # print("point_indices_already_covered_outsiez ", lib.vector_size(point_indices_already_covered_out))
    # print("point_indices_already_covered_sizes siez ", lib.vector_size(point_indices_already_covered_sizes))

    covered_landmark_indices_c_out = lib.new_vector()
    covered_landmark_indices_c_sizes = lib.new_vector_of_size(kClusters__)

    res_to_cluster_by_landmark_out = lib.new_vector()

    lib.process_clusters(c_num_lands_, c_num_res_, c_kClusters_,
                        c_point_indices_in_cluster_flat_cpp, c_point_indices_in_cluster_sizes_cpp,
                        c_point_indices_cpp,
                        c_res_indices_in_cluster_flat_cpp,c_res_indices_in_cluster_sizes_cpp,
                        res_toadd_out, res_toadd_sizes_out,
                        point_indices_already_covered_out, point_indices_already_covered_sizes,
                        covered_landmark_indices_c_out, covered_landmark_indices_c_sizes,
                        res_to_cluster_by_landmark_out)

    # print("lib.vector_get(res_toadd_sizes_out, i) ", 0, " : ", lib.vector_get(res_toadd_sizes_out, 0))

    #res_toadd_to_c_ = fillPythonVec(res_toadd_out, res_toadd_sizes_out, kClusters__)
    point_indices_already_covered_ = fillPythonVec(point_indices_already_covered_out, point_indices_already_covered_sizes, kClusters__)
    covered_landmark_indices_c_ = fillPythonVec(covered_landmark_indices_c_out, covered_landmark_indices_c_sizes, kClusters__)
    res_to_cluster_by_landmark_out_ = fillPythonVecSimple(res_to_cluster_by_landmark_out)

    return res_to_cluster_by_landmark_out_, point_indices_already_covered_, covered_landmark_indices_c_

def AngleAxisRotatePoint(angleAxis, pt):
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

    tmp = (w0 * pt[:, 0] + w1 * pt[:, 1] + w2 * pt[:, 2]) * (1.0 - costheta)

    r0 = pt[:, 0] * costheta + wCrossPt0 * sintheta + w0 * tmp
    r1 = pt[:, 1] * costheta + wCrossPt1 * sintheta + w1 * tmp
    r2 = pt[:, 2] * costheta + wCrossPt2 * sintheta + w2 * tmp

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


def torchSingleResiduum(camera_params_, point_params_, p2d):
    angle_axis = camera_params_[:, :3]
    points_cam = AngleAxisRotatePoint(angle_axis, point_params_)
    points_cam = points_cam + camera_params_[:, 3:6]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f = camera_params_[:, 6]
    k1 = camera_params_[:, 7]
    k2 = camera_params_[:, 8]
    r2 = points_projX * points_projX + points_projY * points_projY
    distortion = 1.0 + r2 * (k1 + k2 * r2)
    points_reprojX = points_projX * distortion * f
    points_reprojY = points_projY * distortion * f
    resX = (points_reprojX - p2d[:, 0]).reshape((p2d.shape[0], 1))
    resY = (points_reprojY - p2d[:, 1]).reshape((p2d.shape[0], 1))
    residual = torch.cat([resX[:,], resY[:,]], dim=1)
    return residual

def torchSingleResiduumX(camera_params, point_params, p2d) :
    angle_axis = camera_params[:,:3]
    points_cam = AngleAxisRotatePoint(angle_axis, point_params)
    points_cam = points_cam + camera_params[:,3:6]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f  = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    r2 = points_projX*points_projX + points_projY*points_projY
    distortion = 1. + r2 * (k1 + k2 * r2)
    points_reprojX = points_projX * distortion * f
    resX = (points_reprojX-p2d[:,0])
    return resX

def torchSingleResiduumY(camera_params, point_params, p2d) :
    angle_axis = camera_params[:,:3]
    points_cam = AngleAxisRotatePoint(angle_axis, point_params)
    points_cam = points_cam + camera_params[:,3:6]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f  = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    r2 = points_projX*points_projX + points_projY*points_projY
    distortion = 1. + r2 * (k1 + k2 * r2)
    points_reprojY = points_projY * distortion * f
    resY = (points_reprojY-p2d[:,1])
    return resY

def getJacSin(
    start_,
    end_,
    camera_indices_,
    point_indices_,
    camera_params_,
    point_params_,
    torch_points_2d_,
):
    end_ = min(end_, camera_indices_.shape[0])
    funx0_st1 = lambda X0, X1, X2: torchSingleResiduum(
        X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2)
    )
    jac = jacobian(
        funx0_st1,
        (
            camera_params_[camera_indices_[start_:end_], :],
            point_params_[point_indices_[start_:end_], :],
            torch_points_2d_[start_:end_, :],
        ),
        create_graph=False,
        vectorize=True,
        strategy="reverse-mode",
    )

    # print(start_, " ", end_, ", jac shapes : ", jac[0].shape, " x ", jac[1].shape)
    res = funx0_st1(
        camera_params_[camera_indices_[start_:end_], :],
        point_params_[point_indices_[start_:end_], :],
        torch_points_2d_[start_:end_, :],
    )
    # print("res", res.shape) # 200,2, so N, x/y
    return (jac[0], jac[1], res)

# bs : blocksize, eg 9 -> 9x9 or 3 -> 3x3 per block
def blockInverse(M, bs):
    Mi = M.copy()
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            mat = Mi.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs)
            # print(i, " ", mat)
            imat = inv_dense(mat) # inv or pinv?
            Mi.data[bs2 * i : bs2 * i + bs2] = imat.flatten()
    else:
        Mi = M.copy()
        for i in range(int(M.data.shape[0])):
            Mi.data[i : i + 1] = 1.0 / Mi.data[i : i + 1]
    return Mi

def ComputeDerivativeMatricesNew(x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d
):
    verbose = False
    if verbose:
        start = time.time() # this is not working at all. Slower then iteratively

    funx0_st1 = lambda X0, X1, X2: torchSingleResiduumX(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d fucntion -> grad possible
    funy0_st1 = lambda X0, X1, X2: torchSingleResiduumY(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d fucntion -> grad possible

    torch_cams = x0_t_cam[camera_indices_[:],:] #x0_t[:n_cameras*9].reshape(n_cameras,9)[camera_indices[:],:]
    torch_lands = x0_t_land[point_indices_[:],:] #x0_t[n_cameras*9:].reshape(n_points,3)[point_indices[:],:]
    torch_lands.requires_grad_()
    torch_cams.requires_grad_()
    torch_cams.retain_grad()
    torch_lands.retain_grad()

    resX = funx0_st1(torch_cams, torch_lands, torch_points_2d[:,:]).flatten()
    lossX = torch.sum(resX)
    lossX.backward()

    cam_grad_x = torch_cams.grad.detach().numpy().copy()
    #cam_grad_x.detach()
    land_grad_x = torch_lands.grad.detach().numpy().copy()
    #land_grad_x.detach()
    #print("torch_lands.grad X ", land_grad_x)

    torch_cams.grad.zero_()
    torch_lands.grad.zero_()
    resY = funy0_st1(torch_cams, torch_lands, torch_points_2d[:,:]).flatten()
    lossY = torch.sum(resY)
    lossY.backward()
    cam_grad_y = torch_cams.grad.detach().numpy().copy()
    land_grad_y = torch_lands.grad.detach().numpy().copy()
    #print("torch_lands.grad Y ", land_grad_y)

    if verbose:
        end = time.time()
        print("All torch grads take ", end - start, "s")
        start = time.time()

    J_pose = buildMatrixNew(cam_grad_x, cam_grad_y, camera_indices_, sz=9)
    if verbose:
        end = time.time()
        print(" build Matrix & residuum took ", end-start, "s")
        start = time.time()
    J_land = buildMatrixNew(land_grad_x, land_grad_y, point_indices_, sz=3)

    fx0 = buildResiduumNew(resX.detach(), resY.detach())

    if verbose:
        print(" build Matrix & residuum took ", end-start, "s")
        end = time.time()

    return (J_pose, J_land, fx0)

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
    # debug: set all inner parameters to 0
    if False:
        #datavals[0:end:9] = 0
        #datavals[1:end:9] = 0
        #datavals[2:end:9] = 0

        #datavals[3:end:9] = 0
        #datavals[4:end:9] = 0
        #datavals[5:end:9] = 0

        datavals[6:end:9] = 0
        datavals[7:end:9] = 0
        datavals[8:end:9] = 0

    crs_pose = csr_array((datavals, np.concatenate(indices), np.concatenate(indptr)))
    J_pose = csr_matrix(crs_pose)
    return J_pose

def buildResiduumNew(resX, resY) :
    data = []
    data.append(resX.flatten().numpy())
    data.append(resY.flatten().numpy())
    res = np.concatenate(data)
    return res

def blockEigenvalue(M, bs):
    Ei = np.zeros(M.shape[0])
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            mat = M.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs)
            # print(i, " ", mat)
            evs = eigvalsh(mat)
            Ei[bs*i:bs*i+bs] = evs[bs-1]
        Ei = diag_sparse(Ei)
    else:
        Ei = M.copy()

    # todo: eval super safe method since symmetric FAIL again.
    # idea was that non isotropic structure might hurting bfgs. but as bad as without
    #maxEv = np.max(Ei.data)
    #Ei.data[:] = 0.1 * maxEv # 0.1 worked
    #Ei.data[:] *= 0.125 #0.25 promising # 0.1 already too low. could try line-search idea
    return Ei

def copy_selected_blocks(M, block_selection_, bs):
    Mi = M.copy()
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            if block_selection_[i] == True:
                Mi.data[bs2 * i : bs2 * i + bs2] = 1e-12
    else:
        Mi = M.copy()
        for i in range(int(M.data.shape[0])):
            if block_selection_[i] == True:
                Mi.data[i : i + 1] = 1e-12
    return Mi

def stop_criterion(delta, delta_i, i):
    eps = 1e-2 #1e-2 used in paper, tune. might allow smaller as faster?
    return (i+1) * delta_i / delta < eps

def solvePowerIts(Ul, W, Vli, bS, m_):
    # costk = np.sum( bS**2 )
    # print("start gd cost ", costk)

    Uli = blockInverse(Ul, 9)
    xk = Uli * bS
    g = xk

    for it in range(m_):
        # here uli^1/2 * M uli^1/2 * 'uli^1/2 * g' could be a symmetric split.
        # to the power of k uli^1/2 * uli^1/2 = uli
        g = Uli * (W * (Vli * (W.transpose() * g)))
        xk = xk + g
        if False:
            # eq is Ul [I - Uli * W * Vli * W.transpose()] x = b
            costk = np.sum(((Ul - W * Vli * W.transpose()) * xk - bS) ** 2)
            print(it, " gd cost ", costk)
        if stop_criterion(np.linalg.norm(xk, 2), np.linalg.norm(g, 2), it):
           return xk
    return xk


def cluster_by_camera(
    camera_indices_, points_3d_, points_2d_, point_indices_, kClusters_, startL_
):
    # sort by res-indices by camera indices
    res_sorted = np.argsort(camera_indices_)
    # camera_indices_[res_sorted]
    num_res = camera_indices_.shape[0]
    print("number of residuum: ", num_res)
    # now split by cameras. list [] of nparrays with camera_ind
    cluster_to_camera_ = np.array_split(np.arange(n_cameras), kClusters_)
    # cluster_to_residuum =

    camera_indices_in_cluster_ = []
    point_indices_in_cluster_ = []
    points_2d_in_cluster_ = []
    for c in range(kClusters_):
        indices_in_cluster = np.logical_and(
            camera_indices_[res_sorted] <= cluster_to_camera_[c][-1],
            camera_indices_[res_sorted] >= cluster_to_camera_[c][0],
        )
        res_indices_in_cluster = res_sorted[indices_in_cluster]
        res_indices_in_cluster = np.sort(res_indices_in_cluster)
        points_2d_in_cluster_.append(points_2d_[res_indices_in_cluster])
        camera_indices_in_cluster_.append(camera_indices_[res_indices_in_cluster])
        point_indices_in_cluster_.append(point_indices_[res_indices_in_cluster])
        print("cams in ",c," " , np.unique(camera_indices_[res_indices_in_cluster]))

    points_3d_in_cluster_ = []
    L_in_cluster_ = []
    for _ in range(kClusters_):
        points_3d_in_cluster_.append(points_3d_.copy())
        L_in_cluster_.append(startL_)
    return (
        camera_indices_in_cluster_,
        point_indices_in_cluster_,
        points_2d_in_cluster_,
        cluster_to_camera_,
        points_3d_in_cluster_,
        L_in_cluster_,
    )

def cluster_by_camera_smarter(
    camera_indices_, points_3d_, points_2d_, point_indices_, kClusters_, startL_,init_cam_id=0, init_lm_id=0
):

    cluster_to_camera_ = [] # list of sets
    cluster_to_landmarks_ = []
    num_cams = np.unique(camera_indices_).shape[0]
    num_res = camera_indices_.shape[0]

    baseline_clustering = False #True

    if baseline_clustering:

        # make map cam id to landmark indices seen by cam as set.
        # smarter might be to pick cam 'furthest' apart
        cam_idx_to_lms = [ set(point_indices_[camera_indices_ == cam_idx]) for cam_idx in range(num_cams) ]

        # init pick
        cameras_available = set(range(num_cams))
        #print("cameras_available ", cameras_available, " num_cams ", num_cams)
        cluster_to_camera_.append(set([init_cam_id])) # cam 0 to set 0
        cluster_to_landmarks_.append(cam_idx_to_lms[init_lm_id])
        cameras_available.remove(init_cam_id)
        for i in range(kClusters_-1):
            best_cam = min(cameras_available, key=lambda candidate: sum(np.random.normal(0,1,1)**2 + len(set.intersection(cluster_to_landmarks_[set_id], cam_idx_to_lms[candidate])) for set_id in range(len(cluster_to_camera_))))
            cluster_to_camera_.append(set([best_cam]))
            cameras_available.remove(best_cam)
            cluster_to_landmarks_.append(cam_idx_to_lms[best_cam])
        # init done

        while cameras_available:
            cid = np.argmin(np.array([len(x) for x in cluster_to_landmarks_]))
            best_score = 0
            best_cam = -1
            for candidate in cameras_available:
                overlap = len(set.intersection(cluster_to_landmarks_[cid], cam_idx_to_lms[candidate]))
                total_overlap = sum(len(set.intersection(cluster_to_landmarks_[set_id], cam_idx_to_lms[candidate])) for set_id in range(len(cluster_to_camera_)))
                score = overlap / (total_overlap + 1)
                if score > best_score:
                    best_score = score
                    best_cam = candidate
            cluster_to_camera_[cid].add(best_cam)
            cameras_available.remove(best_cam)
            cluster_to_landmarks_[cid] = cluster_to_landmarks_[cid].union(cam_idx_to_lms[best_cam])
            #print("cluster_to_camera_smart_ mid ", cluster_to_camera_smart_)
        #print("cluster_to_camera_smart_ end ", cluster_to_camera_smart_)

        camera_indices_in_cluster_ = []
        point_indices_in_cluster_ = []
        points_2d_in_cluster_ = []
        res_indices_in_cluster_ = []
        for c in range(kClusters_):
            #res_indices_in_cluster = np.sort(indices_in_cluster)
            indices_in_cluster = np.zeros(num_res, dtype=bool)
            for camid_in_c_ in cluster_to_camera_[c]:
                #print(indices_in_cluster.shape, " ", indices_in_cluster.shape, " ", camid_in_c_)
                indices_in_cluster = np.logical_or(indices_in_cluster, camera_indices_==camid_in_c_)
            res_indices_in_cluster = np.arange(num_res)[indices_in_cluster]
            print(res_indices_in_cluster.shape," ", res_indices_in_cluster)

            res_indices_in_cluster = np.sort(res_indices_in_cluster)
            # res_indices_in_cluster are the residuums of th e cameras ids in cluster, so map cam id to res id
            points_2d_in_cluster_.append(points_2d_[res_indices_in_cluster])
            camera_indices_in_cluster_.append(camera_indices_[res_indices_in_cluster])
            point_indices_in_cluster_.append(point_indices_[res_indices_in_cluster])
            res_indices_in_cluster_.append(res_indices_in_cluster.copy())
            print("cams in ",c," " , np.unique(camera_indices_[res_indices_in_cluster]))
    else:
        res_indices_in_cluster_, kClusters_ = cluster_covis_lib(kClusters_, camera_indices_, point_indices_)
        kClusters = kClusters_
        camera_indices_in_cluster_ = []
        point_indices_in_cluster_ = []
        points_2d_in_cluster_ = []
        cluster_to_camera_ = []
        for c in range(kClusters_):
            res_indices_in_c_ = np.sort(res_indices_in_cluster_[c])
            points_2d_in_cluster_.append(points_2d_[res_indices_in_c_])
            camera_indices_in_cluster_.append(camera_indices_[res_indices_in_c_])
            point_indices_in_cluster_.append(point_indices_[res_indices_in_c_])
            cluster_to_camera_.append(np.unique(camera_indices_[res_indices_in_c_]))

            #res_indices_in_cluster_.append(res_indices_in_cluster.copy())

    points_3d_in_cluster_ = []
    L_in_cluster_ = []
    for _ in range(kClusters_):
        points_3d_in_cluster_.append(points_3d_.copy())
        L_in_cluster_.append(startL_)

    return (
        camera_indices_in_cluster_,
        point_indices_in_cluster_,
        points_2d_in_cluster_,
        cluster_to_camera_,
        points_3d_in_cluster_,
        L_in_cluster_,
        kClusters
    )


def cluster_by_landmark(
    camera_indices_, points_2d_, point_indices_, kClusters_, pre_merges, old_vtxsToPart=0
):
    num_res = camera_indices_.shape[0]
    num_cams = np.unique(camera_indices_).shape[0]
    num_lands = np.unique(point_indices_).shape[0] #points_3d_.shape[0]
    print("number of residuum: ", num_res)

    res_indices_in_cluster_, kClusters = cluster_covis_lib(kClusters_, camera_indices_, point_indices_)
    kClusters_ = kClusters
    camera_indices_in_cluster_ = []
    point_indices_in_cluster_ = []
    points_2d_in_cluster_ = []
    cluster_to_camera_ = []
    for c in range(kClusters_):
        res_indices_in_c_ = np.sort(res_indices_in_cluster_[c])
        points_2d_in_cluster_.append(points_2d_[res_indices_in_c_])
        camera_indices_in_cluster_.append(camera_indices_[res_indices_in_c_])
        point_indices_in_cluster_.append(point_indices_[res_indices_in_c_])
        cluster_to_camera_.append(np.unique(camera_indices_[res_indices_in_c_]))

    res_to_cluster_by_landmark_, point_indices_already_covered_, covered_landmark_indices_c_ = \
        process_cluster_lib(num_lands, num_res, kClusters, point_indices_in_cluster_, res_indices_in_cluster_, point_indices_)
    
    # we only case about covered_landmark_indices_c_
    # 1. distribute residuals by occurence of above per cluster: res_to_cluster_by_landmark_: res -> cluster covers all landmarks exculsively. to test.
    # 2. landmarks per cluster are exclusive, but use whole cams per cluster (simpler)
    # 3. need indices of cameras utilized in cluster and stepsizes per cam in cluster
    # point_indices_already_covered_ : here exclusively covered by cluster
    for ci in range(kClusters):
        point_indices_already_covered_[ci] = np.union1d(point_indices_already_covered_[ci], covered_landmark_indices_c_[ci])
        ids_of_res_in_cluster = res_to_cluster_by_landmark_ == ci
        camera_indices_in_cluster_[ci] = camera_indices_[ids_of_res_in_cluster]
        points_2d_in_cluster_[ci] = points_2d_[ids_of_res_in_cluster]
        point_indices_in_cluster_[ci] = point_indices_[ids_of_res_in_cluster]
        print("===== Cluster ", ci , " covers ", points_2d_in_cluster_[ci].shape, "residuals ",
              np.unique(point_indices_in_cluster_[ci]).shape, " of ", num_lands, " landmarks ",
              np.unique(camera_indices_in_cluster_[ci]).shape, " of ", num_cams, "cameras ")

    # check if indices are disjoint / local vs global / sums up to n_points
    sum_cams_cover = 0
    sum_landmarks_cover = 0
    for ci in range(kClusters):
        sum_cams_cover += np.unique(camera_indices_in_cluster_[ci]).shape[0]
        sum_landmarks_cover += np.unique(point_indices_already_covered_[ci]).shape[0]
    if sum_landmarks_cover < num_lands or sum_cams_cover < num_cams:
        print("sum_cams_cover ", sum_cams_cover, " / ", num_cams)
        print("sum_landmarks_cover ", sum_landmarks_cover, " / ", num_lands)
        return

    return (
        camera_indices_in_cluster_,
        point_indices_in_cluster_,
        points_2d_in_cluster_,
        kClusters
    )


# Afterwards landmarks only present in 1 cluster should equal points_3d_in_cluster_ afterwards.
# For those input s = old_lm. out put is 2 new - old.
# s is updated as s+ = s + 2new - old - new = s + new - old = new: ok.
# so should work without indexing.
#
# idea from bundle return:
# just to check, problem last uk is not present.
# or 2u+ - s - u to get delta. then add to v. or use real uk instead.
#
# poses_v, _ = average_cameras_new(
#     camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster) # old_poses for costs?
def average_cameras_new(
    camera_indices_in_cluster_, poses_in_cluster, poses_s_in_cluster, L_in_cluster_, UL_in_cluster_#, pose_v_
):
    num_cameras = poses_in_cluster[0].shape[0]
    sum_Ds_2u = np.zeros(num_cameras * 9)
    sum_constant_term = 0
    for i in range(len(L_in_cluster_)):
        # Lc = L_in_cluster_[i]
        camera_indices_ = np.unique(camera_indices_in_cluster[i])
        # mean_points[point_indices_,:] = mean_points[point_indices_,:] + points_3d_in_cluster_[i][point_indices_,:] * Lc
        # num_clusters[point_indices_] = num_clusters[point_indices_] + Lc
        # fill Vl with 0s 3x3 blocks? somehow ..
        # sparse matrix is 0,3,6, ... data (0s of Vl data), 012,012,012,345,345,345, etc ?
        #
        # data can remain, indptr can remain, indices must be adjusted / new
        # point_indices_
        #print(UL_in_cluster_[i].data.shape, " ", camera_indices_.shape)

        indices = np.repeat(
            np.array([9 * camera_indices_ + j for j in range(9)]).transpose(), 9, axis=0
        ).flatten()
        # indices.append(np.array([3 * point_indices_ + j for j in range(3)]).transpose().flatten())
        # indptr is to be set to have empty lines by 0 3 3 -> no entries in row 3. 0:0-3, row 1:3-3

        indptr = [np.array([0])]
        j = 0
        for q in range(num_cameras):
            # print(q, " ", j, " ", point_indices_.shape[0], " ", np.array([9*j+3, 9*j+6, 9*j+9]) )
            if j < camera_indices_.shape[0] and camera_indices_[j] == q:
                indptr.append(np.array([81 * j + 9, 81 * j + 18, 81 * j + 27, 
                                        81 * j + 36, 81 * j + 45, 81 * j + 54, 
                                        81 * j + 63, 81 * j + 72, 81 * j + 81]).flatten())
                j = j + 1
            else: # 9x9 block of "0's" not present in data
                indptr.append(np.array([81 * j, 81 * j, 81 * j, 81 * j, 
                                        81 * j, 81 * j, 81 * j, 81 * j, 81 * j]).flatten())
        indptr = np.concatenate(indptr)
        U_pose = csr_matrix(
            (UL_in_cluster_[i].data, indices, indptr),
            shape=(9 * num_cameras, 9 * num_cameras),
        )
        # print(mean_points.shape, " " , V_land.shape, points_3d_in_cluster_[i].shape)
        # print cost after/before.
        # cost old v is where ? (v-2u+s)^T V_land (v-2u+s) = v^T V_land v + 2 v^T V_land (-2u+s) + (2u-s)^T V_land (2u-s)
        # derivative 2 V_land v + 2 V_land (-2u+s) = 0 <-> sum (V_land) v = sum (V_land (2u-s))

        # print(i, "averaging 3d ", points_3d_in_cluster_[i][globalSingleLandmarksB_in_c[i], :]) # indeed 1 changed rest is constant
        # print(i, "averaging vl ", V_land.data.reshape(-1,9)[globalSingleLandmarksA_in_c[i],:])  # indeed diagonal

        # TODO change 3, claim  2u+-s = 2 * (s+u)/2 - s  -  2 * (vli/2 .. ), so subtract u to get delta only
        u2_s = (2 * poses_in_cluster[i].flatten() - poses_s_in_cluster[i].flatten())
        #u2_s = (2 * points_3d_in_cluster_[i].flatten() - landmark_s_in_cluster_[i].flatten()) - (points_3d_in_cluster_[i].flatten() - delta_l_in_cluster[i].flatten())

        sum_Ds_2u += U_pose * u2_s # has 0's for those not present

        # print(i, "averaging u2_s ", u2_s.reshape(-1,3)[globalSingleLandmarksB_in_c[i], :]) # indeed 1 changed rest is constant

        sum_constant_term += poses_in_cluster[i].flatten().dot(U_pose * (poses_in_cluster[i].flatten() + u2_s - poses_s_in_cluster[i].flatten()))
        if i == 0:
            Up_all = U_pose
        else:
            Up_all += U_pose
    Upi_all = blockInverse(Up_all, 9)
    # TODO change 1
    pose_v_out = Upi_all * sum_Ds_2u
    verbose = False
    if verbose:
        cost_input  = 0.5 * (landmark_v_.flatten().dot(Vl_all * landmark_v_.flatten() - 2 * sum_Ds_2u) + sum_constant_term)
        cost_output = 0.5 * (landmark_v_out.dot(       Vl_all * landmark_v_out        - 2 * sum_Ds_2u) + sum_constant_term)

        #cost_simpler_out = landmark_v_out.dot(       Vl_all * landmark_v_out)        * 0.5 - landmark_v_out.dot(       sum_Ds_2u)
        #cost_simpler_in =  landmark_v_.flatten().dot(Vl_all * landmark_v_.flatten()) * 0.5 - landmark_v_.flatten().dot(sum_Ds_2u)
        print("========== update v: ", round(cost_input), " -> ", round(cost_output), " gain: ", round(cost_input - cost_output) )
        #print("======================== update v: ", round(cost_simpler_in), " -> ", round(cost_simpler_out), " gain: ", round(cost_simpler_in - cost_simpler_out) )

    return pose_v_out.reshape(num_cameras, 9), Up_all
    # Then use this for fixing landmarks / updating. enven use VLi / Vl instead? argmin_x sum_y=lm_in_cluster (x-y) VL (x-y) of last Vl.
    # ==> x^t sum Vl x - 2 x sum (Vl y) + const =>  x = (sum Vl)^-1 [sum (Vl*y)].
    # above should lead to better? solutions at least. At next local updates we work with Vl?
    # 1. return last Vl. recall Vl * delta = nabla l is solved. or use L * diag Vl and return it (cheaper) or L * Vl ?
    # return 3x3 matrix per lm.
    # f(x) < f(y) + <nabla fy , x-y> + (x-y)^ Vl (x-y). Vl is making this strongly convex by design. s.t. this descent lemma holds. Even by design.

# cost is fuk * fuk + rho_k/2 |u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k>
#                     rho_k/2 <u_k - v_k, u_k - v_k> - rho_k <s_k - u_k, u_k - v_k>
#                     rho_k/2 <u_k - v_k - 2s_k + 2u_k, u_k - v_k>
#                     rho_k/2 <3u_k - v_k - 2s_k, u_k - v_k>
#                     rho_k/2 {v^tv - vT[2uk-sk] + uk^T[3uk-2sk]}
def cost_DRE(
    #camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, pose_v
    camera_indices_in_cluster_,  poses_in_cluster_, poses_s_in_cluster_, L_in_cluster_, Ul_in_cluster_, pose_v_
):
    num_cams =  poses_in_cluster_[0].shape[0]
    sum_Ds_2u = np.zeros(num_cams * 9)
    sum_constant_term = 0
    sum_u_s =0
    sum_u_v = 0
    sum_u_v_ = 0
    sum_2u_s_v = 0
    for i in range(len(L_in_cluster_)):
        point_indices_ = np.unique(camera_indices_in_cluster_[i])
        indices = np.repeat(
            np.array([9 * point_indices_ + j for j in range(9)]).transpose(), 9, axis=0
        ).flatten()

        indptr = [np.array([0])]
        j = 0
        for q in range(num_cams):
            if j < point_indices_.shape[0] and point_indices_[j] == q:
                indptr.append(np.array([81 * j + 9, 81 * j + 18, 81 * j + 27, 
                                        81 * j + 36, 81 * j + 45, 81 * j + 54, 
                                        81 * j + 63, 81 * j + 72, 81 * j + 81]).flatten())
                j = j + 1
            else:
                indptr.append(np.array([81 * j, 81 * j, 81 * j, 81 * j, 
                                        81 * j, 81 * j, 81 * j, 81 * j, 81 * j]).flatten())
        indptr = np.concatenate(indptr)

        U_pose = csr_matrix(
            (Ul_in_cluster_[i].data, indices, indptr),
            shape=(9 * num_cams, 9 * num_cams),
        )
        u2_s = (2 *  poses_in_cluster_[i].flatten() - poses_s_in_cluster_[i].flatten())
        sum_Ds_2u += U_pose * u2_s # has 0's for those not present
        sum_constant_term +=  poses_in_cluster_[i].flatten().dot(U_pose * (poses_in_cluster_[i].flatten() + u2_s - poses_s_in_cluster_[i].flatten()))

        u_s =  poses_in_cluster_[i].flatten() - poses_s_in_cluster_[i].flatten()
        u_v =  poses_in_cluster_[i].flatten() - pose_v_.flatten()
        v_u2_s = u2_s - pose_v_.flatten()
        sum_u_s += u_s.dot(U_pose * u_s)
        sum_u_v += u_v.dot(U_pose * u_v)
        sum_u_v_ += u_v.dot(u_v)
        sum_2u_s_v += v_u2_s.dot(U_pose * v_u2_s)
        if i == 0:
            Ul_all = U_pose
        else:
            Ul_all += U_pose

    # TODO: I use a different Vl to compute the cost here than in the update of prox u.
    #       Since I want to work with a new Vl already. Problem.
    # i want |u-s|_D |u-v|_D, also |v-2u-s|_D
    cost_input  = 0.5 * (pose_v_.flatten().dot(Ul_all * pose_v_.flatten() - 2 * sum_Ds_2u) + sum_constant_term)
    print("---- |u-s|^2_D ", round(sum_u_s), "|u-v|^2_D ", round(sum_u_v), "|2u-s-v|^2_D ", round(sum_2u_s_v), "|u-v|^2 ", round(sum_u_v_))
    return cost_input

# TODO: shorten
def primal_cost(
    poses_in_cluster_,
    camera_indices_in_cluster_,
    point_indices_in_cluster_,
    points_2d_in_cluster_,
    points_3d_in_cluster_,
):
    cameras_indices_in_c_ = np.unique(camera_indices_in_cluster_)
    cameras_in_c = poses_in_cluster_[cameras_indices_in_c_]
    torch_points_2d_in_c = from_numpy(points_2d_in_cluster_)
    torch_points_2d_in_c.requires_grad_(False)

    unique_points_in_c_ = np.unique(point_indices_in_cluster_)
    inverse_point_indices = -np.ones(np.max(unique_points_in_c_) + 1)  # all -1
    for i in range(unique_points_in_c_.shape[0]):
        inverse_point_indices[unique_points_in_c_[i]] = i

    point_indices_in_c = point_indices_in_cluster_.copy()
    for i in range(point_indices_in_cluster_.shape[0]):
        point_indices_in_c[i] = inverse_point_indices[point_indices_in_c[i]]

    points_3d_in_c = points_3d_in_cluster_[unique_points_in_c_]

    camera_indices_ = np.zeros(camera_indices_in_cluster_.shape[0], dtype=int)
    for i in range(cameras_indices_in_c_.shape[0]):
        camera_indices_[camera_indices_in_cluster_ == cameras_indices_in_c_[i]] = i

    point_indices_ = point_indices_in_c
    torch_points_2d = torch_points_2d_in_c
    cameras_in = cameras_in_c
    points_3d_in = points_3d_in_c

    x0_l_ = points_3d_in.flatten()
    # holds all cameras, only use fraction, camera_indices_ can be adjusted - min index
    x0_p_ = cameras_in.flatten()
    x0 = np.hstack((x0_p_, x0_l_))
    x0_t = from_numpy(x0)
    # torch_points_2d = from_numpy(points_2d)
    n_cameras_ = int(x0_p_.shape[0] / 9)
    n_points_ = int(x0_l_.shape[0] / 3)
    x0_t_cam = x0_t[: n_cameras_ * 9].reshape(n_cameras_, 9) # not needed?
    x0_t_land = x0_t[n_cameras_ * 9 :].reshape(n_points_, 3)
    funx0_st1 = lambda X0, X1, X2: \
        torchSingleResiduum(X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2))
    fx1 = funx0_st1(
        x0_t_cam[camera_indices_[:]],
        x0_t_land[point_indices_[:]],
        torch_points_2d)
    costEnd = np.sum(fx1.numpy() ** 2)
    return costEnd


# cost_, x0_p_c_, x0_l_c_, Lnew_c_, Vl_c_ = bundle_adjust(
#     local_landmark_indices_in_cluster, # these are indexing into landmarks_in_c, a subset of all landmarks, directly.
#     pose_indices_in_c,
#     poses_only_in_cluster_, # input those poses not present anywhere else to relax hold on those.
#     torch_points_2d_in_c,
#     landmarks_in_c,
#     poses_in_c,
#     poses_s_in_c,
#     Vl_in_cluster_, # these are for those poses in cluster only. 
#     L_in_cluster_,
#     its_,
# )

def bundle_adjust(
    point_indices_,
    camera_indices_,
    poses_only_in_cluster_,
    torch_points_2d,
    points_3d_in,
    cameras_in,
    cameras_s_, # taylor expand at point_3d_in -> prox on landmark_s_ - points_3d = lambda (multiplier)
    Ul_in_c_,
    L_in_cluster_,
    successfull_its_=1,
):
    # print("landmarks_only_in_cluster_  ", landmarks_only_in_cluster_, " ", np.sum(landmarks_only_in_cluster_), " vs ", np.sum(1 - landmarks_only_in_cluster_) )
    newForUnique = False
    blockEigMult = 1e-3 # 1e-3 was used before
    L = L_in_cluster_
    minimumL = 1e-6
    updateJacobian = True
    # holds all! landmarks, only use fraction likely no matter not present in cams anyway.
    x0_l_ = points_3d_in.flatten()
    # holds all cameras, only use fraction, camera_indices_ can be adjusted - min index
    x0_p_ = cameras_in.flatten()
    x0 = np.hstack((x0_p_, x0_l_))
    x0_t = from_numpy(x0)
    s_p_ = cameras_s_.flatten()
    # torch_points_2d = from_numpy(points_2d)
    n_cameras_ = int(x0_p_.shape[0] / 9)
    n_points_ = int(x0_l_.shape[0] / 3)
    powerits = 20 # kind of any value works here? > =5?
    tr_eta_1 = 0.8
    tr_eta_2 = 0.25

    it_ = 0
    funx0_st1 = lambda X0, X1, X2: \
        torchSingleResiduum(X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2))

    #stepSize = diag_sparse(np.zeros(n_points_))
    # make diagonal again.
    if issparse(Ul_in_c_): # only increase -- if needed.
        stepSize = diag_sparse(Ul_in_c_.diagonal())

    steSizeTouched = False

    while it_ < successfull_its_:

        if updateJacobian:  # not needed if rejected
            #
            x0_t_cam = x0_t[: n_cameras_ * 9].reshape(n_cameras_, 9)
            x0_t_land = x0_t[n_cameras_ * 9 :].reshape(n_points_, 3)
            J_pose, J_land, fx0 = ComputeDerivativeMatricesNew (
                x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d )

            JtJ = J_pose.transpose() * J_pose
            #JtJDiag = diag_sparse(np.fmax(JtJ.diagonal(), 1e-4))

            blockEigenvalueJtJ = blockEigenvalue(JtJ, 9)
            # TODO does work with 1.. ?
            stepSize = 1. * (blockEigMult * blockEigenvalueJtJ + 1 * JtJ.copy()) # Todo '2 *' vs 1 by convex
            # if not issparse(Vl_in_c_) and it_ < 1:
            #     stepSize = blockEigenvalueJltJl
            # else: # increase where needed -- this here is WAY too slow?
            #     stepSize.data = np.maximum(0.05 * stepSize.data, blockEigenvalueJltJl.data) # else diagSparse of it

            JltJl = J_land.transpose() * J_land
            J_eps = 1e-4
            JltJlDiag = JltJl + J_eps * diag_sparse(np.ones(JltJl.shape[0]))
          
            JtJDiag = stepSize.copy() # max 1, 1/L, line-search dre fails -> increase
            if newForUnique:
                JtJDiag = copy_selected_blocks(JtJDiag, poses_only_in_cluster_, 3)
                JtJDiag = JtJDiag + L * blockEigMult * blockEigenvalueJtJ

            # maybe better: 3x3 matrix sqrt(|M|_1 |M|inf) as diag. Yet this removes effect of 'L' getting small = large steps.
            # do i need to keep memory to ensure it remains >? or pre compute grad (and store)?
            print(" min/max JtJ.diagonal() ", np.min(JtJ.diagonal()), " ", np.max(JtJ.diagonal()), " adjusted ", np.min(JtJDiag.diagonal()), " ", np.max(JtJDiag.diagonal()))
            #print("JltJlDiag.shape ", JltJlDiag.shape, JltJlDiag.shape[0]/3)

            JtJDiag = 1/L * JtJDiag # max 1, 1/L, line-search dre fails -> increase

            W = J_pose.transpose() * J_land
            bp = J_pose.transpose() * fx0
            bl = J_land.transpose() * fx0

            prox_rhs = x0_p_ - s_p_
            if newForUnique: # alternative turn off completely, use 2u-s -> return (u-s)/2 to average u+k = uk + delta uk
                landmarks_in_many_cluster_ = np.invert(landmarks_only_in_cluster_)
                diag_present = diag_sparse( np.repeat((np.ones(n_points_) * landmarks_in_many_cluster_).reshape(-1,1), 3).flatten() )
                prox_rhs = 1 * diag_present * prox_rhs

            costStart = np.sum(fx0**2)
            penaltyStartConst = prox_rhs.dot(JtJDiag * prox_rhs)

        # start_ = time.time()
        Vl = JltJl + L * JltJlDiag
        Ul = JtJ + L * JtJDiag
        penaltyStart = L * penaltyStartConst
        # cost added is + L * (delta_v - s_l_ + x0_l_)^T  JltJlDiag * (delta_v - s_l_ + x0_l_)
        # + L * (delta_v)^T  JltJlDiag * (delta_v) + 2 L * (delta_v^T JltJlDiag * (x0_l_ - s_l_) + L * (s_l_ - x0_l_)^T  JltJlDiag * (s_l_ - x0_l_)
        # derivative
        # L * 2 * JltJlDiag * (delta_v) + 2 L * JltJlDiag * (x0_l_ - s_l_) = 0
        # added cost is, 2 L * (delta_v^T JltJlDiag * (x0_l_ - s_l_) + L * (s_l_ - x0_l_)^T  JltJlDiag * (s_l_ - x0_l_)

        Vli = blockInverse(Vl, 3)
        bp_s = bp + L * JtJDiag * prox_rhs # TODO: + or -. '+', see above
        bS = (bp_s - W * Vli * bl).flatten()

        delta_p = -solvePowerIts(Ul, W, Vli, bS, powerits)
        delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl)

        penaltyL = L * (delta_l).dot(JltJlDiag * delta_l)
        penaltyP = L * (delta_p + prox_rhs).dot(JtJDiag * (delta_p + prox_rhs))
        # end_ = time.time()
        # print("Lm step took ", end - start, "s")

        fx0_new = fx0 + (J_pose * delta_p + J_land * delta_l)
        costQuad = np.sum(fx0_new**2)
        print(it_, "it. cost 0     ", round(costStart)," cost + penalty ", round(costStart + penaltyStart), " === using L = ", L)
        print(it_, "it. cost 0/new ", round(costQuad), " cost + penalty ", round(costQuad + penaltyL + penaltyP),)

        # update and compute cost
        x0_p_ = x0_p_ + delta_p
        x0_l_ = x0_l_ + delta_l

        x0 = np.hstack((x0_p_, x0_l_))
        x0_t = from_numpy(x0)
        x0_t_cam = x0_t[: n_cameras_ * 9].reshape(n_cameras_, 9)
        x0_t_land = x0_t[n_cameras_ * 9 :].reshape(n_points_, 3)

        fx1 = funx0_st1(
            x0_t_cam[camera_indices_[:]],
            x0_t_land[point_indices_[:]],
            torch_points_2d)
        costEnd = np.sum(fx1.numpy() ** 2)
        print(it_, "it. cost 1     ", round(costEnd), "      + penalty ", round(costEnd + penaltyL + penaltyP),)

        tr_check = (costStart + penaltyStart - costEnd - penaltyL) / (costStart + penaltyStart - costQuad - penaltyL)

        old_descent_lemma = False
        if old_descent_lemma:
            # old descent lemma test.
            #nablaXp = L * JtJDiag * delta_p  # actual gradient. discussable
            nablaXp = L * JtJDiag * delta_p  # actual gradient. discussable TODO
            nablaXl = JltJlDiag * delta_l  # actual gradient: J^t fx0 = bp|bl
            # before
            LfkDiagonal = \
                2 * (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l)) \
                / (delta_l.dot(nablaXl) + delta_p.dot(nablaXp))
            LfkDistance  = costEnd - (costStart + bp.dot(delta_p) + bl.dot(delta_l) + (delta_l.dot(nablaXl) + delta_p.dot(nablaXp)) / 2)
            LfkViolated = LfkDistance > 0
            LfkSafe = (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l)) < 0 # for any phi ok.
        else:
            # after:
            # f(x) <= f(y) + <nabla(f(y) x-y> + Lf/2 |x-y|^2
            # we demand stepsize phi >= 2 Lf. Then even
            # f(x) <= f(y) + <nabla(f(y) x-y> + phi/4 |x-y|^2 
            # (f(x) - f(y) - <nabla(f(y) x-y>) * 4 / |x-y|^2  <= phi
            #nablaXp = L * JtJDiag * delta_p    # actual gradient: J^t fx0 = bp|bl , no L. grad at f(x) is L * JtJDiag * delta_p - s
            #nablaXl = JltJlDiag * delta_l  # actual gradient is b. L since JltJlDiag multiplied by 1/L. 
            descent_rhs_l = delta_l.dot(JltJlDiag * delta_l)
            descent_rhs_p = L * delta_p.dot(JtJDiag * delta_p)

            LfkDistance  = costEnd - (costStart + bp.dot(delta_p) + bl.dot(delta_l) + (descent_rhs_l + descent_rhs_p) / 4)
            LfkViolated = LfkDistance > 0
            LfkSafe = (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l)) < 0 # for any phi ok.

        if tr_check < tr_eta_2:
            print(" //////  tr_check " , tr_check, " Lfk distance ", LfkDistance, " -nabla^Tdelta=" , -bp.dot(delta_p) - bl.dot(delta_l), " /////")
            L = L * 2
            JltJlDiag = 1/2 * JltJlDiag

        if tr_check >= tr_eta_2 and LfkViolated: # violated -- should revert update.
            steSizeTouched = True
            print(" |||||||  Lfk distance ", LfkDistance, " -nabla^Tdelta=" , -bp.dot(delta_p) - bl.dot(delta_l), " |||||||")
            #stepSize = stepSize * 2
            # other idea, initially we only add 1/2^k eg 0.125, times the needed value and inc if necessary, maybe do not add anything if not needed.

            blockEigenvalueJtJ.data *= 2 # appears slow but safe
            #stepSize.data = np.maximum(stepSize.data, blockEigenvalueJltJl.data) # else diagSparse of it
            stepSize = blockEigMult * blockEigenvalueJtJ + JtJ.copy()

            JtJDiag = 1/L * stepSize.copy()
            penaltyStartConst = prox_rhs.dot(JtJDiag * prox_rhs)

        if LfkSafe and not steSizeTouched:
            L = L / 2

        # version with penalty check for ADMM convergence / descent lemma. Problem: slower?
        if costStart + penaltyStart < costEnd + penaltyL or LfkViolated:
            # revert -- or linesearch
            x0_p_ = x0_p_ - delta_p
            x0_l_ = x0_l_ - delta_l
            x0 = np.hstack((x0_p_, x0_l_))
            x0_t = from_numpy(x0)
            x0_t_cam = x0_t[: n_cameras_ * 9].reshape(n_cameras_, 9)
            x0_t_land = x0_t[n_cameras_ * 9 :].reshape(n_points_, 3)
            updateJacobian = False
            continue # avoids de/increasing L below.
        else:
            it_ = it_ + 1
            updateJacobian = True

        print(" ------- Lfk distance ", LfkDistance, " -nabla^Tdelta=" , -bp.dot(delta_p) - bl.dot(delta_l),  " tr_check ", tr_check, " -------- ")

    x0_p_ = x0_p_.reshape(n_cameras_, 9)
    x0_l_ = x0_l_.reshape(n_points_, 3)

    # idea use latest for better control ?
    getBetterStepSize = False # this is used as approx of f in update of v and thus s. maybe change there u-v should be small. 
    if getBetterStepSize: # needs to set L correctly
        J_pose, J_land, fx0 = ComputeDerivativeMatricesNew(
            x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d
        )
        JltJl = J_land.transpose() * J_land
        stepSize.data = np.maximum(stepSize.data, blockEigenvalue(JltJl, 3).data) # else diagSparse of it
        JltJlDiag = 1/L * stepSize.copy() # max 1, 1/L, line-search dre fails -> increase

    Rho = L * JtJDiag + 1e-12 * Ul

    # TODO change 2
    if newForUnique:
        diag_present = 0.5 * diag_sparse( np.repeat((np.ones(n_points_) * landmarks_only_in_cluster_).reshape(-1,1), 3).flatten() )
        # see above, compensate for averaging
        xTest = x0_l_ - (diag_present*(x0_l_.flatten() - s_l_)).reshape(-1,3)
        return costEnd, x0_p_, xTest, L, diag_present + Rho

    L_out = np.maximum(minimumL, np.minimum(L_in_cluster_ * 2, L)) # not clear if generally ok, or 2 or 4 should be used.
    return costEnd, x0_p_, x0_l_, L_out, Rho

    # recall solution wo. splitting is 
    # solve Vl x + bS + Vd () 
    #bl_s = bl + L * JltJlDiag * (x0_l_ - s_l_) # TODO: + or -. '+', see above
    #delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl_s).flatten()
    # sum_k delta_l Vlk delta_l + delta_l ((Wk.transpose() * delta_p) + bl_sk) argmin
    # all this is local per block. k blocks: sum_k Vlk = 3x3, we could instead 
    # return Vlk and bk = ((Wk.transpose() * delta_p) + bl_sk): 4 times #landmarks floats.
    # and compute the update, summing each and solving v = (sumk vlk)^-1 (sum_k bk).
    # would be cool if we could do n iterations locally -- with a gain.
    # this shows one core can do all this in parallel / local network can -> problem is parallelizable trivially
    # large network, better send minimal information. problem still send landmarks.
    # and diagonal? would be good if can work n steps locally.
    # what is missing? maybe accumulate 'Vl' on the way or upper bound
    # as max eigenvalue per landmark 3x3, or just sum row/col -> blockEigen, and 

def updateCluster(
    poses_in_cluster_,
    camera_indices_in_cluster_,
    landmark_indices_in_cluster_,
    points_2d_in_cluster_,
    landmarks_,
    poses_s_in_cluster_,
    Vl_in_cluster_,
    L_in_cluster_,
    pose_occurences,
    its_,
):
    landmark_indices_in_c_ = np.unique(landmark_indices_in_cluster_)
    landmarks_in_c = landmarks_[landmark_indices_in_c_]
    local_landmark_indices_in_cluster = np.zeros(landmark_indices_in_cluster_.shape[0], dtype=int)
    for i in range(landmark_indices_in_c_.shape[0]):
        local_landmark_indices_in_cluster[landmark_indices_in_cluster_ == landmark_indices_in_c_[i]] = i

    torch_points_2d_in_c = from_numpy(points_2d_in_cluster_)
    torch_points_2d_in_c.requires_grad_(False)

    # take point_indices_in_cluster[ci] unique:
    unique_poses_in_c_ = np.unique(camera_indices_in_cluster_)
    # unique_points_in_c_[i] -> i, map each pi : point_indices_in_cluster[ci] to position in unique_points_in_c_[i]
    inverse_pose_indices = -np.ones(np.max(unique_poses_in_c_) + 1)  # all -1
    for i in range(unique_poses_in_c_.shape[0]):
        inverse_pose_indices[unique_poses_in_c_[i]] = i

    poses_only_in_cluster_ = pose_occurences[unique_poses_in_c_] == 1
    #print("Unique landmarks  ", landmark_occurences, " ", landmark_occurences.shape, " ", np.min(landmark_occurences), " ", np.max(landmark_occurences))
    #print("Unique landmarks  ", landmarks_only_in_cluster_, " ", np.sum(landmarks_only_in_cluster_), " vs ", np.sum(1 - landmarks_only_in_cluster_) )

    pose_indices_in_c = camera_indices_in_cluster_.copy()  # np.zeros(point_indices_in_cluster_.shape)
    for i in range(camera_indices_in_cluster_.shape[0]):
        pose_indices_in_c[i] = inverse_pose_indices[pose_indices_in_c[i]]

    # put in unique points, adjust point_indices_in_cluster[ci] by id in unique_points_in_c_
    poses_in_c = poses_in_cluster_[unique_poses_in_c_]
    poses_s_in_c = poses_s_in_cluster_[unique_poses_in_c_] # same as landmarks

    cost_, x0_p_c_, x0_l_c_, Lnew_c_, Vl_c_ = bundle_adjust(
        local_landmark_indices_in_cluster, # these are indexing into landmarks_in_c, a subset of all landmarks, directly.
        pose_indices_in_c,
        poses_only_in_cluster_, # input those poses not present anywhere else to relax hold on those.
        torch_points_2d_in_c,
        landmarks_in_c,
        poses_in_c,
        poses_s_in_c,
        Vl_in_cluster_, # these are for those poses in cluster only. 
        L_in_cluster_,
        its_,
    )

    return (
        cost_,
        x0_p_c_,
        x0_l_c_, # out side globa lm [landmark_indices_in_c_] = x0_l_c_
        Lnew_c_,
        Vl_c_,
        unique_poses_in_c_,
        landmark_indices_in_c_
    )

def prox_f(camera_indices_in_cluster_, point_indices_in_cluster_, points_2d_in_cluster_,
    poses_in_cluster_, landmarks_, poses_s_in_cluster_, L_in_cluster_, Vl_in_cluster_, kClusters, innerIts=1, sequential=True) :
    cost_ = np.zeros(kClusters)

    num_poses = poses_in_cluster_[0].shape[0]
    pose_occurences = np.zeros(num_poses)
    for ci in range(kClusters):
        unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci])
        pose_occurences[unique_poses_in_c_] +=1

    # for ci in range(kClusters):
    #     print(ci, " 3d " ,points_3d_in_cluster_[ci][landmark_occurences==1, :])

    for ci in range(kClusters):
        (
            cost_c_,
            x0_p_c_,
            x0_l_c_,
            Lnew_c_,
            Vl_c_,
            unique_poses_in_c_,
            landmark_indices_in_c_
        ) = updateCluster(
            poses_in_cluster_[ci],
            camera_indices_in_cluster_[ci],
            point_indices_in_cluster_[ci],
            points_2d_in_cluster_[ci],
            landmarks_,
            poses_s_in_cluster_[ci],
            Vl_in_cluster_[ci],
            L_in_cluster_[ci],
            pose_occurences, # haeh?
            its_=innerIts,
        )
        cost_[ci] = cost_c_
        L_in_cluster_[ci] = Lnew_c_
        Vl_in_cluster_[ci] = Vl_c_
        poses_in_cluster_[ci][unique_poses_in_c_, :] = x0_p_c_
        landmarks_[landmark_indices_in_c_] = x0_l_c_

    for ci in range(kClusters):
        #vl = Vl_in_cluster_[ci]
        unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci])
        poses_only_in_cluster_ = pose_occurences[unique_poses_in_c_] == 1
        #globalSingleLandmarksA_in_c[ci] = poses_only_in_cluster_.copy()
        #globalSingleLandmarksB_in_c[ci] = poses_only_in_cluster_==1

        # print(ci, " 3d ", points_3d_in_cluster_[ci][landmark_occurences==1, :]) # indeed 1 changed rest is constant
        # print(ci, " vl ", vl.data.reshape(-1,9)[landmarks_only_in_cluster_,:])  # indeed diagonal
    #return (cost_, L_in_cluster_, Vl_in_cluster_, points_3d_in_cluster_, x0_p_, delta_l_in_cluster_, globalSingleLandmarksA_in_c, globalSingleLandmarksB_in_c)
    return (cost, L_in_cluster_, Vl_in_cluster_, poses_in_cluster_, landmarks_)


# fill lists G and F, with g and f = g - old g, sets of size m, 
# at position it % m, c^t compute F^tF c + lamda (c - 1/k)^2, sum c=1
# g = x0, f = delta. Actullay xnew = xold + delta.
def RNA(G, F, g, f, it_, m_, Fe, fe, lamda):
    lamda = 0.05 # reasonable 0.01-0.1
    crefVersion = True
    lamda = 0.05 # cref version needs larger 
    h = -1 #-0.1 # 2 / (L+mu) -- should 1/diag * F^t F * c
    id_ = it_ % m_
    if len(G) >= m_:
        #print("it, it%m", it, " ", it % m)
        G[id_] = np.squeeze(g)
        F[id_] = np.squeeze(f)
        Fe[id_] = np.squeeze(fe)
    else:
        G.append(np.squeeze(g))
        F.append(np.squeeze(f))
        Fe.append(np.squeeze(fe))
    mg = len(G)
    cref = np.zeros(mg)
    if mg >= m_:
        cref[id_] = 1
    else:
        cref[mg-1] = 1

    Gs_ = np.concatenate(G).reshape(mg, -1).transpose()
    Fs_ = np.concatenate(F).reshape(mg, -1).transpose()
    Fes_ = np.concatenate(Fe).reshape(mg, -1).transpose()
    #print("Fs ", Fs.shape)
    FtF = Fs_.transpose().dot(Fs_) # why dot?
    fTfNorm = np.linalg.norm(FtF, 2)
    #print("FtF ", FtF.shape, " |FtF|_2=", fTfNorm)

    FtF = FtF * (1. / fTfNorm) + lamda * np.eye(mg)
    if crefVersion:
        #print("cref ", cref, " ", cref.shape)
        w = np.linalg.solve(FtF, lamda * cref)
        z = np.linalg.solve(FtF, np.ones(mg))
        #print("w ", w, " ", w.shape)
        #print("z ", z, " ", z.shape)
        #print(w.transpose().dot(np.ones(mg)))
        c = w + z * (1 - w.transpose().dot(np.ones(mg))) / (z.transpose().dot(np.ones(mg)))
        #print("c ", c, " ", c.shape)
    else:
        z = np.linalg.solve(FtF, np.ones(mg) / mg)
        c = z / z.transpose().dot(np.ones(mg)) # sums to 1
    extrapolation = Gs_.dot(c) #+ 0.1 * Fs_.dot(c)
    extrapolationF = Fes_.dot(c)

    print("c ", c, " ", c.shape, id_)

    # shape utter non sense
    #print("extrapolation ", extrapolation.shape, " ", g.shape)
    return (G, F, Fe, np.squeeze(extrapolation - h * extrapolationF))


def BFGS_direction(r, ps, qs, rhos, k, mem, mu):
    # r = -r # not needed with below
    # lookup k-1, k-mem entries. 
    alpha = np.zeros([mem,1])
    r = np.squeeze(r)
    for i in range(k-1, np.maximum(k-mem,-1), -1):
    #for i in range(k-1, np.maximum(k-mem-1,-1), -1): # todo might be correct.
        #print("1i", i) # k-1, .. k-mem usually
        j = i % mem # j>=0
        #print("1j", j)
        #print("j ", j, " ", r.shape, ps[j].shape)
        alpha[j] = np.dot(r, ps[j]) * rhos[j]
        r = r - alpha[j]*qs[j]
        if (rhos[j]>0):
            print(j, " 1st. al ", alpha[j], " rh ", rhos[j], " qs " , np.linalg.norm(qs[j],2), " ps " , np.linalg.norm(ps[j],2) )

    dk_ = mu * r

    for i in range(np.maximum(k-mem, 0), k):
        #print("2i", i) # k-1, .. k-mem usually
        j = i % mem # j>=0
        #print("2j", j)
        beta = rhos[j] * np.dot(dk_, qs[j])
        dk_ = dk_ + ps[j] * (alpha[j] - beta)
        if (rhos[j]>0):
            print(j, " 2nd. al ", alpha[j], " rh ", rhos[j], " be ", beta, " qs " , np.linalg.norm(qs[j],2), " ps " , np.linalg.norm(ps[j],2) )

    return dk_

##############################################################################
cameras, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

n_cameras = cameras.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

write_output = False #True
read_output =  False
if read_output:
    camera_params_np = np.fromfile("camera_params_drs.dat", dtype=np.float64)
    point_params_np = np.fromfile("point_params_drs.dat", dtype=np.float64)
    x0_p = camera_params_np.reshape(-1)
    x0_l = point_params_np.reshape(-1)
    #x0 = np.concatenate([x0_p, x0_l])
    x0 = np.hstack((x0_p, x0_l))
    x0_t = from_numpy(x0)
    cameras   = x0_p.reshape(n_cameras,9)
    points_3d = x0_l.reshape(n_points,3)

np.set_printoptions(formatter={"float": "{: 0.2f}".format})

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))

x0_p = cameras.ravel()
# x0_l = points_3d.ravel()
# x0 = np.hstack((x0_p, x0_l))
x0_p = x0_p.reshape(n_cameras, 9)
# x0_l = x0_l.reshape(n_points, 3)

# torch_points_2d = from_numpy(points_2d)
# torch_points_2d.requires_grad_(False)
# x0_t = from_numpy(x0)

# init. Duplicate only non unique variables. s and u contain only those duplicates. 
# initially all variables are contained, but selected by indices .. 
# Init s = v = u. = dupe all landmarks, now twice.
# 1. s+ = s + v-u = s.
# prox on (s+)
# prox on 2u+ - s+. Adjust average fct.
# update s+ = s + v+ - u+ = s + 2u+ - s - u+ = u+ .. Actually not since v+ != 2u-s
# u := points_3d_in_cluster
# s := landmark_s_in_cluster
# v := output, temporary

# 1. take problem and split, sort indices by camera, define local global map and test it.
startL = 1
kClusters = 3
innerIts = 1  # change to get an update, not 1 iteration
its = 60
cost = np.zeros(kClusters)
lastCost = 1e20
lastCostDRE = 1e20
basic_version = False # accelerated or basic
sequential = True
linearize_at_last_solution = True # linearize at uk or v. maybe best to check energy. at u or v. DRE:
lib = ctypes.CDLL("./libprocess_clusters.so")
init_lib()

# f(x) + 1/2 (v^T Vlk * v - v^T 2 * Vlk (2x - sk) ) for x=u/v. Does not look right .. haeh
# 
# The Lagrangian is sum_k f_k (x_k^t) - <mu^t_k, x_k^t - Bk z^t> + rho_k/2 | Bk z^t-x_k^t |^2
# Apparently the difference is only (for Y = x_k^t or Bk z):
# f_k (Y) + rho_k/2 | Y - Bk z^t |^2 - <mu^t_k, Y - Bk z^t>.
# and we recall lambda = s^+ - v = s + (v-u) - v = s - u.
#
# Hence it is fv' * fv versus fu' * fu + rho_k|u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k> in DRS variables.
# how did this turn into my dre cost? 
# insert v -> only fk(v) remains. insert u: 

(
    camera_indices_in_cluster,
    point_indices_in_cluster,
    points_2d_in_cluster,
    kClusters,
) = cluster_by_landmark(
    camera_indices, points_2d, point_indices, kClusters, pre_merges=0)

L_in_cluster = []
for _ in range(kClusters):
    L_in_cluster.append(startL)

print(L_in_cluster)
Ul_in_cluster = [0 for x in range(kClusters)] # dummy fill list
#poses = cameras.copy()
poses_s_in_cluster = [cameras.copy() for _ in range(kClusters)]
poses_in_cluster = [cameras.copy() for _ in range(kClusters)]
landmarks = points_3d.copy()

if basic_version:

    for it in range(its):
        start = time.time()
        (
            cost,
            L_in_cluster,
            Ul_in_cluster,
            poses_in_cluster,
            landmarks
        ) = prox_f(
            camera_indices_in_cluster, point_indices_in_cluster, points_2d_in_cluster,
            poses_in_cluster, landmarks, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, kClusters, innerIts=innerIts, sequential=True,
            )
        end = time.time()

        #print("++++++++++++++++++ globalSingleLandmarksB_in_c[0].shape ", globalSingleLandmarksB_in_c[0].shape)

        currentCost = np.sum(cost)
        print(it, " ", round(currentCost), " gain ", round(lastCost - currentCost), ". ============= sum fk update takes ", end - start," s",)

        poses_v, _ = average_cameras_new(
            camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster) # old_poses for costs?

        #DRE cost BEFORE s update
        dre = cost_DRE(camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, poses_v) + currentCost

        tau = 1 # 2 is best ? does not generalize!
        for ci in range(kClusters):
            poses_s_in_cluster[ci] = poses_s_in_cluster[ci] + tau * (poses_v - poses_in_cluster[ci]) # update s = s + v - u.

        #DRE cost AFTER s update
        #dre = cost_DRE(camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, poses_v) + currentCost

        primal_cost_v = 0
        for ci in range(kClusters):
            primal_cost_v += primal_cost(
                poses_v,
                camera_indices_in_cluster[ci],
                point_indices_in_cluster[ci],
                points_2d_in_cluster[ci],
                landmarks)
        primal_cost_u = 0
        for ci in range(kClusters):
            primal_cost_u += primal_cost(
                poses_in_cluster[ci],
                camera_indices_in_cluster[ci],
                point_indices_in_cluster[ci],
                points_2d_in_cluster[ci],
                landmarks)

        print( it, " ======== DRE ====== ", round(dre) , " ========= gain " , \
            round(lastCostDRE - dre), "==== f(v)= ", round(primal_cost_v), " f(u)= ", round(primal_cost_u))

        lastCost = currentCost
        # print(" output shapes ", x0_p_c.shape, " ", x0_l_c.shape, " takes ", end-start , " s")
        if False and lastCostDRE - dre < 1:
            break
        lastCostDRE = dre

        # fill variables for update: linearize at u or v.
        for ci in range(kClusters):
            if not linearize_at_last_solution: # linearize at v / average solution, same issue I suppose. Yes. solution is too return the new gradient, s.t. update of v is wrt to current situation.
                poses_in_cluster[ci]  = poses_v.copy() # init at v, above at u

else:

    bfgs_mem = 6 # 2:Cost @50:  -12.87175888983266, 6: cost @ 50: 12.871757400143322
    bfgs_mu = 1.0
    bfgs_qs = np.zeros([bfgs_mem, kClusters * 9 * n_cameras]) # access/write with % mem
    bfgs_ps = np.zeros([bfgs_mem, kClusters * 9 * n_cameras])
    bfgs_rhos = np.zeros([bfgs_mem, 1])
    poses_s_in_cluster_pre = [0 for x in range(kClusters)] # dummy fill list
    search_direction = [0 for x in range(kClusters)] # dummy fill list
    poses_s_in_cluster_bfgs = [0 for x in range(kClusters)] # dummy fill list
    steplength = [0 for x in range(kClusters)]
    lastCostDRE_bfgs = lastCostDRE
    Gs = [] 
    Fs = []
    Fes = []
    rnaBufferSize = 6

    # Only it 0: update s,u,v.
    start = time.time()
    (
        cost,
        L_in_cluster,
        Ul_in_cluster,
        poses_in_cluster,
        landmarks
    ) = prox_f(
        camera_indices_in_cluster, point_indices_in_cluster, points_2d_in_cluster,
        poses_in_cluster, landmarks, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, kClusters, innerIts=innerIts, sequential=True,
        )
    end = time.time()
    currentCost = np.sum(cost)
    print(-1, " ", round(currentCost), " gain ", round(lastCost - currentCost), ". ============= sum fk update takes ", end - start," s",)

    poses_v, U_all = average_cameras_new(
        camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster)

    for it in range(its):

        steplength = 0
        tau = 1 # todo sqrt(2), not sure what is happening here.
        for ci in range(kClusters):
            poses_s_in_cluster_pre[ci] = poses_s_in_cluster[ci] + tau * (poses_v - poses_in_cluster[ci]) # update s = s + v - u.
            steplength += np.linalg.norm(poses_s_in_cluster_pre[ci] - poses_s_in_cluster[ci], 2)
            #update_flat = (poses_s_in_cluster_pre[ci] - poses_s_in_cluster[ci]).flatten()
            #steplength += update_flat.dot(Ul_all * update_flat)
        #steplength = np.sqrt(steplength)

        # get line search direction and update bfgs data
        # operate with np concatenate to get large vector and reshape search_direction here?
        bfgs_r = np.zeros(kClusters * 9 * n_cameras)
        rna_s  = np.zeros(kClusters * 9 * n_cameras)
        for ci in range(kClusters): #bfgs_r = u-v
            bfgs_r[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = (poses_v - poses_in_cluster[ci]).flatten()
            #bfgs_r[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = (poses_v - poses_in_cluster[ci]).flatten()
            rna_s[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = poses_s_in_cluster_pre[ci].flatten()

        use_bfgs = False # maybe full u,v?
        if use_bfgs:
            dk = BFGS_direction(bfgs_r, bfgs_ps, bfgs_qs, bfgs_rhos, it, bfgs_mem, bfgs_mu)
            dk_stepLength = np.linalg.norm(dk, 2)
            # step length by using Vl, also above computing steplength!
            #dk_stepLength = 0
            #for ci in range(kClusters): #bfgs_r = u-v
                #dk_stepLength += (dk[ci * 3 * n_points: (ci+1) * 3 * n_points]).dot(Ul_all * (dk[ci * 3 * n_points: (ci+1) * 3 * n_points]))
            #dk_stepLength = np.sqrt(dk_stepLength)
            multiplier = steplength / dk_stepLength
        else:
            #L_rna = max(L_in_cluster) , L_rna * bfgs_r
            Gs, Fs, Fes, dk = RNA(Gs, Fs, rna_s, bfgs_r, it, rnaBufferSize, Fes, bfgs_r, lamda = 1)
            dk = dk - (rna_s - bfgs_r)
            dk_stepLength = np.linalg.norm(dk, 2)
            multiplier = 1

        for ci in range(kClusters):
            search_direction[ci] = dk[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras].reshape(n_cameras, 9)

        # need a check to reject idiotic proposals:
        # rho(u-s)^2 is gigantic
        # line search:
        line_search_iterations = 3
        print(" ..... step length ", steplength, " bfgs step ", dk_stepLength, " ratio ", multiplier)
        for ls_it in range(line_search_iterations):
            tk = ls_it / (line_search_iterations-1)
            for ci in range(kClusters):
                poses_s_in_cluster_bfgs[ci] = tk * poses_s_in_cluster_pre[ci] + (1-tk) * (poses_s_in_cluster[ci] + multiplier * search_direction[ci])
                #print(" bfgs_r ", bfgs_r[ci * 3 * n_points: (ci+1) * 3 * n_points].reshape(n_points, 3))
                #print(" search_direction[ci] ", search_direction[ci])

            # prox on line search s:
            #print("1. x0_p", "points_3d_in_cluster", points_3d_in_cluster)
            poses_in_cluster_bfgs = [elem.copy() for elem in poses_in_cluster]
            L_in_cluster_bfgs = L_in_cluster.copy()
            Ul_in_cluster_bfgs = [elem.copy() for elem in Ul_in_cluster]
            (   cost_bfgs,
                L_in_cluster_bfgs,
                Ul_in_cluster_bfgs,
                poses_in_cluster_bfgs,
                landmarks_bfgs,
            ) = prox_f(
                camera_indices_in_cluster, point_indices_in_cluster, points_2d_in_cluster, 
                poses_in_cluster_bfgs, landmarks.copy(), poses_s_in_cluster_bfgs, L_in_cluster_bfgs, Ul_in_cluster_bfgs, kClusters, innerIts=innerIts, sequential=True,
                )
            
            #print("2. x0_p", "points_3d_in_cluster", points_3d_in_cluster)
            currentCost_bfgs = np.sum(cost_bfgs)
            poses_v_bfgs, Ul_all_bfgs = average_cameras_new(
                camera_indices_in_cluster, poses_in_cluster_bfgs, poses_s_in_cluster_bfgs, L_in_cluster_bfgs, Ul_in_cluster_bfgs)

            # update buffers
            if ls_it == 0: # todo: the one we accept put here, no?
                #bfgs_ps[it % bfgs_mem] = -dk * multiplier
                bfgs_ps[it % bfgs_mem] = -bfgs_r # this is not so much overshooting as dk
                bfgs_rr = np.zeros(kClusters * 9 * n_cameras)
                for ci in range(kClusters):
                    bfgs_rr[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = poses_v_bfgs.flatten() - poses_in_cluster_bfgs[ci].flatten() # flatten?
                bfgs_qs[it % bfgs_mem] = bfgs_rr - bfgs_r
                bfgs_rhos[it % bfgs_mem] = np.maximum(0., 1./ bfgs_qs[it % bfgs_mem].dot(bfgs_ps[it % bfgs_mem]))

            # eval cost
            dre_bfgs = cost_DRE(camera_indices_in_cluster, poses_in_cluster_bfgs, poses_s_in_cluster_bfgs, L_in_cluster_bfgs, Ul_in_cluster_bfgs, poses_v_bfgs)
            dre_bfgs += currentCost_bfgs

            # debugging cost block ################
            primal_cost_v = 0
            for ci in range(kClusters):
                primal_cost_v += primal_cost(
                    poses_v_bfgs,
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    landmarks_bfgs) #points_3d_in_cluster[ci]) # v not u
            primal_cost_u = 0
            for ci in range(kClusters):
                primal_cost_u += primal_cost(
                    poses_in_cluster_bfgs[ci],
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    landmarks_bfgs)

            print( it, "/", ls_it, " ======== DRE BFGS ====== ", round(dre_bfgs) , " ========= gain " , \
                round(lastCostDRE_bfgs - dre_bfgs), "==== f(v)= ", round(primal_cost_v), " f(u)= ", round(primal_cost_u), " ~= ", currentCost_bfgs)

            # accept / reject, reject all but drs and see
            # if ls_it == line_search_iterations-1 :
            if dre_bfgs <= lastCostDRE_bfgs or ls_it == line_search_iterations-1 : # not correct yet, must be <= last - c/gamma |u-v|
                for ci in range(kClusters):
                    poses_s_in_cluster[ci] = poses_s_in_cluster_bfgs[ci].copy()
                    poses_in_cluster[ci] = poses_in_cluster_bfgs[ci].copy()
                    Ul_in_cluster[ci] = Ul_in_cluster_bfgs[ci].copy()
                L_in_cluster = L_in_cluster_bfgs.copy()
                landmarks = landmarks_bfgs.copy()
                lastCostDRE_bfgs = dre_bfgs.copy()
                poses_v = poses_v_bfgs.copy()
                #print("A landmark_s_in_cluster", landmark_s_in_cluster)
                break
# here bfgs is better, but dre has better cost for the drs solution.

# either adjust points_3d_in_cluster -> copy over output.
# or let output as input to ba -- and Lfk
# descent lemma for landmarks only
# f(x) <= f(y) + <nabla f(y), x-y> + Lf/2 |x-y|^2
# e.g. update from y to x, nabla f(y) Jac_l * res(y)
# x=y+delta, [f(x) - f(y) - b^T delta] / delta^2 <= Lf/2, defines Lf
# trust region
#        fx0_new = fx0 + (J_pose * delta_p + J_land * delta_l) = fx0 + J^t H^-1 nabla fx0
#        costQuad = np.sum(fx0_new**2)
#        tr_check = (costStart - costEnd) / (costStart - costQuad)
#
# TODO:
# maybe estimate locally per lm -> change in f(x) from LOCAL residuums. x->y. gradient from nabla -> get local Lfk per lm!
# will fix close ones more strongly.
# xl -> xl + delta == sum of residduums^2 of that landmark involved. Fix cam, use new gradient L* diag * delta!
# new residuum f(x) is fast.
#
# Then use this for fixing landmarks / updating. enven use VLi / Vl instead? argmin_x sum_y=lm_in_cluster (x-y) VL (x-y) of last Vl.
# ==> x^t sum Vl x - 2 x sum (Vl y) + const =>  x = (sum Vl)^-1 [sum (Vl*y)].
# above should lead to better? solutions at least. At next local updates we work with Vl?
# 1. return last Vl. recall Vl * delta = nabla l is solved. or use L * diag Vl and return it (cheaper) or L * Vl ?
# return 3x3 matrix per lm.
#     f(x) < f(y) + <nabla fy , x-y> + (x-y)^ Vl (x-y). Vl is making this strongly convex by design. s.t. this descent lemma holds. Even by design.
# or  f(x) < f(y) + <nabla fy , x-y> + (x-y)^ JJl (x-y). New solution < old + penalty + <nabla fy, delta>
# <=> f(x) < f(y) + <nabla fy + nabla fx, x-y>

if write_output:
    x0_p.tofile("camera_params_drs.dat")
    landmark_v.tofile("point_params_drs.dat")
