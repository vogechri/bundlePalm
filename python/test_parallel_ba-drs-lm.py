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
from scipy.sparse import hstack as sparse_hstack
#from scipy.sparse.linalg import splu # slow as FUCK
from scipy.sparse.linalg import spsolve # slow as FUCK
#from scipy.linalg import cholesky, cho_solve, cho_factor
#from sksparse.cholmod import cholesky # install suitesparse and ... and ..
from scipy.sparse.linalg import inv as inv_sparse # Slowest ever.
from numpy.linalg import pinv as inv_dense
from numpy.linalg import inv as inv_nonHermetian
from numpy.linalg import eigvalsh, eigh
# idea reimplement projection with torch to get a jacobian -> numpy then
import torch
import math
import ctypes
from torch.autograd.functional import jacobian
from torch import tensor, from_numpy
#import open3d as o3d
from pyinstrument import Profiler
#import pyinstrument
# look at website. This is the smallest problem. guess: pytoch cpu is pure python?
BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
#FILE_NAME = "problem-73-11032-pre.txt.bz2"
FILE_NAME = "problem-138-19878-pre.txt.bz2"

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/dubrovnik/"
FILE_NAME = "problem-16-22106-pre.txt.bz2"
#FILE_NAME = "problem-88-64298-pre.txt.bz2"
#FILE_NAME = "problem-356-226730-pre.txt.bz2" # large dub, play with ideas: cover, etc
#FILE_NAME = "problem-237-154414-pre.txt.bz2"
# acc. 59 / 0  ======== DRE BFGS ======  514126  ========= gain
# acc. x 100: fluctuates
FILE_NAME = "problem-173-111908-pre.txt.bz2"
#FILE_NAME = "problem-135-90642-pre.txt.bz2" # this appears incredibly bad
# problem-253-163691-pre.txt.bz2 90 10  uneven

#BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/trafalgar/"
# 71k
#FILE_NAME = "problem-21-11315-pre.txt.bz2"
#59 / 0  ======== DRE BFGS ======  207468  ========= gain  102
# newVersion worse .. 60 / 0  ======== DRE BFGS ======  208439, 100 its 204k
#FILE_NAME = "problem-257-65132-pre.txt.bz2"
# uneven problem-257-65132-pre.txt.bz2 90 10

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/venice/"
FILE_NAME = "problem-52-64053-pre.txt.bz2"
# TODO: fake: take best not last line search
# restart s ? if dre - f(u) > 0.1 * f(u). s=v?
# restart s ? if f(v) - dre > 0.1 * dre.  s=u? BOTH .. ? This is likely total BS -> stepsize?
# DRE: 78539 |2u-s-v|^2_D per component or <s-u,u-v>_D + |u-v|_D

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/final/"
FILE_NAME = "problem-93-61203-pre.txt.bz2"

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-138-19878-pre.txt.bz2"
FILE_NAME = "problem-646-73584-pre.txt.bz2"

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/venice/"
FILE_NAME = "problem-52-64053-pre.txt.bz2"

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/dubrovnik/"
FILE_NAME = "problem-173-111908-pre.txt.bz2"
# without remove_large_points: 1e-5!
# 71 / 0  ======== DRE BFGS ======  520390  ========= gain  -47 ==== f(v)=  520387  f(u)=

URL = BASE_URL + FILE_NAME

# now it stalls early. haeh? check pcg again.
F_SCALE = 1e0 # idea: scale focal distance and 2d points by this factor -> better numerics. 1e-1 ok, 1e-2 not
# Note all cost computation must be adjusted. |fs * res|^2 = fs^2 * |res|^2, so divide cost by fs^2.
# still easy to do :)

if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)

def remove_large_points(points_3d, camera_indices, points_2d, point_indices):
    remove_ids = np.arange(points_3d.shape[0])[np.sum(points_3d**2, 1) > 1e6]
    if remove_ids.shape[0] >0:
        return points_3d, camera_indices, points_2d, point_indices # do not remove anything
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
    #exit() # if nothing is removed == same performance anyway
    return points_3d, camera_indices, points_2d, point_indices

def invert_focal_distance(camera_params_, camera_indices_, points_2d_):
    flipIndices = camera_params_[:,6] < 0
    flipCamIds = np.arange(camera_params_.shape[0])[flipIndices]
    camera_params_[flipCamIds,6] *= -1
    flip_point_ids = np.isin(camera_indices_, flipCamIds)
    points_2d_[flip_point_ids] *= -1
    return camera_params_, points_2d_

# adjust also residual computation. Gain should be numbers more even, f large, k's small.
def combine_focal_distance_and_kappas(camera_params_):
    camera_params_[:,7] *= camera_params_[:,6]
    camera_params_[:,8] *= camera_params_[:,6]
    return camera_params_

def adjust_focal_scale(camera_params_, points_2d_):
    camera_params_[:,6] *= F_SCALE
    points_2d_[:] *= F_SCALE
    return camera_params_, points_2d_

def scale_adjust_focal_distance(camera_params_, camera_indices_, points_2d_, maxF = 2000):
    flipIndices = camera_params_[:,6] > maxF
    flipCamIds = np.arange(camera_params_.shape[0])[flipIndices]
    scale_flipCamIds = maxF / camera_params_[flipCamIds, 6]
    scale_focal_distance = np.ones(len(flipIndices))
    scale_focal_distance[flipCamIds] = scale_flipCamIds
    camera_params_[flipCamIds, 6] *= scale_flipCamIds
    points_2d_[:,0] *= scale_focal_distance[camera_indices_]
    points_2d_[:,1] *= scale_focal_distance[camera_indices_]
    #flip_point_ids = np.isin(camera_indices_, flipCamIds)
    #points_2d_[flip_point_ids] *= -1
    return camera_params_, points_2d_

def scale_adjust_small_focal_distance(camera_params_, camera_indices_, points_2d_, minF = 1):
    flipIndices = camera_params_[:,6] < minF
    flipCamIds = np.arange(camera_params_.shape[0])[flipIndices]
    scale_flipCamIds = minF / camera_params_[flipCamIds, 6]
    scale_focal_distance = np.ones(len(flipIndices))
    scale_focal_distance[flipCamIds] = scale_flipCamIds
    camera_params_[flipCamIds, 6] *= scale_flipCamIds
    points_2d_[:,0] *= scale_focal_distance[camera_indices_]
    points_2d_[:,1] *= scale_focal_distance[camera_indices_]
    #flip_point_ids = np.isin(camera_indices_, flipCamIds)
    #points_2d_[flip_point_ids] *= -1
    return camera_params_, points_2d_

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

    # invert points_2d_ and focal distance if needed
    (camera_params, points_2d_) = \
        invert_focal_distance(camera_params, camera_indices_, points_2d_)

    (camera_params, points_2d_) = \
        adjust_focal_scale(camera_params, points_2d_)

    # why is this so bad?
    # camera_params = combine_focal_distance_and_kappas(camera_params)
    # avoid super large focal distance values for numerical sanity.
    if False: # changes the cost apparently.
        (camera_params, points_2d_) = \
            scale_adjust_focal_distance(camera_params, camera_indices_, points_2d_)
        # avoid super small focal distance (if those exist?) values for numerical sanity.
        (camera_params, points_2d_) = \
            scale_adjust_small_focal_distance(camera_params, camera_indices_, points_2d_)

    return camera_params, points_3d_, camera_indices_, point_indices_, points_2d_

def round_int(x):
    if x in [float('inf'), float('-inf')]:
        return x
    else:
        return int(round(x))

def float_to_rgb(f):
    a=(1-f)/0.2
    x = np.floor(a)
    y = np.floor(255*(a-x))
    match x:
        case 0: 
            r=1;g=y/255;b=0
        case 1: 
            r=1-y/255;g=1;b=0
        case 2: 
            r=0;g=1;b=y/255
        case 3: 
            r=0;g=1-y/255;b=1
        case 4: 
            r=y/255;g=0;b=1
        case 5: 
            r=1;g=0;b=1
    return [r,g,b]

def render_points_cameras(camera_indices_in_cluster, point_indices_in_cluster, cameras, landmark_v):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    landmarks_vis = []
    cameras_vis1 = []
    cam_loc = -AngleAxisRotatePoint(-from_numpy(cameras[:,0:3]), from_numpy(cameras[:,3:6])).numpy()
   
    for ci in range(kClusters):

        alpha = (kClusters-1 - ci) / (kClusters-1)
        col = float_to_rgb(alpha)

        #cameras_ci = cameras[np.unique(camera_indices_in_cluster[ci]), 3:6].copy()
        cameras_ci = cam_loc[np.unique(camera_indices_in_cluster[ci]), :].copy()
        landmarks_vis.append(o3d.geometry.PointCloud())
        landmarks_ci = landmark_v[np.unique(point_indices_in_cluster[ci]),:]
        landmarks_vis[ci].points = o3d.utility.Vector3dVector(landmarks_ci)
        pc = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = np.repeat(np.array([i-1,j-1,k-1]), cameras_ci.shape[0]) * 0.5
                    pc.append(cameras_ci + a.copy().reshape(3, cameras_ci.shape[0]).transpose())
        cameras_vis1.append(o3d.geometry.PointCloud())
        cameras_vis1[ci].points = o3d.utility.Vector3dVector(np.concatenate(pc))
        cameras_vis1[ci].paint_uniform_color(col) # make larger or what ?
        landmarks_vis[ci].paint_uniform_color(col)

        if ci ==0:
            vis.add_geometry(landmarks_vis[ci])
        else:
            vis.add_geometry(landmarks_vis[ci],  reset_bounding_box=False)
        vis.add_geometry(cameras_vis1[ci], reset_bounding_box=False)

    vis.get_render_option().point_size = 2.0
    vis.run()
    return vis, cameras_vis1, landmarks_vis

def rerender(vis, camera_indices_in_cluster, point_indices_in_cluster, poses_in_cluster, landmark_v, save_image=False):
    for ci in range(kClusters):
        alpha = (kClusters-1 - ci) / (kClusters-1)
        col = float_to_rgb(alpha)
        #cameras_ci = poses_in_cluster[ci][np.unique(camera_indices_in_cluster[ci]), 3:6].copy()
        cam_loc = -AngleAxisRotatePoint(-from_numpy(poses_in_cluster[ci][:,0:3]), from_numpy(poses_in_cluster[ci][:,3:6])).numpy()
        cameras_ci = cam_loc[np.unique(camera_indices_in_cluster[ci]), :].copy()
        landmarks_ci = landmark_v[np.unique(point_indices_in_cluster[ci]),:]
        landmarks_vis[ci].points = o3d.utility.Vector3dVector(landmarks_ci)
        pc = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = np.repeat(np.array([i-1,j-1,k-1]), cameras_ci.shape[0]) * 0.5
                    pc.append(cameras_ci + a.copy().reshape(3, cameras_ci.shape[0]).transpose())
        cameras_vis1[ci].points = o3d.utility.Vector3dVector(np.concatenate(pc))
        cameras_vis1[ci].paint_uniform_color(col)
        vis.update_geometry(cameras_vis1[ci])

        landmarks_vis[ci].paint_uniform_color(col)
        vis.update_geometry(landmarks_vis[ci])

    vis.poll_events()
    vis.update_renderer()
    vis.run()

    if save_image:
        vis.capture_screen_image("temp_%04d.jpg" % i)
    #vis.destroy_window()

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

    lib.recluster_cameras.restype = None
    lib.recluster_cameras.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_void_p] # in&out]

    lib.cluster_cameras_degeneracy.restype = None
    lib.cluster_cameras_degeneracy.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_void_p] # in&out]

def cluster_covis_lib(kClusters, pre_merges_, camera_indices__, point_indices__):
    c_kClusters_ = ctypes.c_int(kClusters)
    #pre_merges_ = 0
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

def post_process_cluster_lib(num_lands_, num_res_, kClusters__, point_indices_in_cluster__, res_indices_in_cluster__, point_indices__, camera_indices__):

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
    c_camera_indices_ptr = (ctypes.c_int * len(camera_indices__))(*camera_indices__)

    c_res_indices_in_cluster_flat_ptr = (ctypes.c_int * len(res_indices_in_cluster_flat))(*res_indices_in_cluster_flat)
    c_res_indices_in_cluster_sizes_ptr = (ctypes.c_int * len(res_indices_in_cluster_sizes))(*res_indices_in_cluster_sizes)

    #c_res_indices_in_cluster_flat = lib.new_vector_of_size(len(c_res_indices_in_cluster_flat))
    c_point_indices_in_cluster_flat_cpp = lib.new_vector_by_copy(c_point_indices_in_cluster_flat_ptr, len(c_point_indices_in_cluster_flat_ptr))
    c_point_indices_in_cluster_sizes_cpp = lib.new_vector_by_copy(c_point_indices_in_cluster_sizes_ptr, len(c_point_indices_in_cluster_sizes_ptr))
    c_point_indices_cpp = lib.new_vector_by_copy(c_point_indices_ptr, len(c_point_indices_ptr))
    c_camera_indices_cpp = lib.new_vector_by_copy(c_camera_indices_ptr, len(c_camera_indices_ptr))

    c_res_indices_in_cluster_flat_cpp = lib.new_vector_by_copy(c_res_indices_in_cluster_flat_ptr, len(c_res_indices_in_cluster_flat_ptr))
    c_res_indices_in_cluster_sizes_cpp = lib.new_vector_by_copy(c_res_indices_in_cluster_sizes_ptr, len(c_res_indices_in_cluster_sizes_ptr))

    #lib.vector_set(c_res_indices_in_cluster_flat, i, value)

    res_toadd_out = lib.new_vector()
    res_toadd_sizes_out = lib.new_vector_of_size(kClusters)

    point_indices_already_covered_out = lib.new_vector_of_size(kClusters)
    point_indices_already_covered_sizes = lib.new_vector_of_size(kClusters)

    # print("point_indices_already_covered_outsiez ", lib.vector_size(point_indices_already_covered_out))
    # print("point_indices_already_covered_sizes siez ", lib.vector_size(point_indices_already_covered_sizes))

    covered_landmark_indices_c_out = lib.new_vector()
    covered_landmark_indices_c_sizes = lib.new_vector_of_size(kClusters)

    res_to_cluster_by_landmark_out = lib.new_vector()

    lib.process_clusters(c_num_lands_, c_num_res_, c_kClusters_, 
                         c_point_indices_in_cluster_flat_cpp, c_point_indices_in_cluster_sizes_cpp,
                         c_point_indices_cpp,
                         c_res_indices_in_cluster_flat_cpp,c_res_indices_in_cluster_sizes_cpp,
                         res_toadd_out, res_toadd_sizes_out,
                         point_indices_already_covered_out, point_indices_already_covered_sizes,
                         covered_landmark_indices_c_out, covered_landmark_indices_c_sizes,
                         res_to_cluster_by_landmark_out)

    lib.recluster_cameras(c_kClusters_, c_camera_indices_cpp, c_point_indices_cpp,
                          res_to_cluster_by_landmark_out)

    #print("lib.vector_get(res_toadd_sizes_out, i) ", 0, " : ", lib.vector_get(res_toadd_sizes_out, 0))
    #res_toadd_to_c_ = fillPythonVec(res_toadd_out, res_toadd_sizes_out, kClusters)
    point_indices_already_covered_ = fillPythonVec(point_indices_already_covered_out, point_indices_already_covered_sizes, kClusters)
    covered_landmark_indices_c_ = fillPythonVec(covered_landmark_indices_c_out, covered_landmark_indices_c_sizes, kClusters)
    #num_res_per_c_ = fillPythonVecSimple(res_toadd_out)

    res_to_cluster_by_landmark_out_ = fillPythonVecSimple(res_to_cluster_by_landmark_out)
    # only first needed: res_to_cluster_by_landmark_out_
    return res_to_cluster_by_landmark_out_, point_indices_already_covered_, covered_landmark_indices_c_

def deg_process_cluster_lib(kClusters__, point_indices__, camera_indices__):

    c_kClusters_ = ctypes.c_int(kClusters__)

    c_point_indices_ptr = (ctypes.c_int * len(point_indices__))(*point_indices__)
    c_camera_indices_ptr = (ctypes.c_int * len(camera_indices__))(*camera_indices__)

    c_point_indices_cpp = lib.new_vector_by_copy(c_point_indices_ptr, len(c_point_indices_ptr))
    c_camera_indices_cpp = lib.new_vector_by_copy(c_camera_indices_ptr, len(c_camera_indices_ptr))

    # todo should operate similarly, put in own
    res_to_cluster_by_landmark_deg_out = lib.new_vector()
    lib.cluster_cameras_degeneracy(c_kClusters_, c_camera_indices_cpp, c_point_indices_cpp, res_to_cluster_by_landmark_deg_out)
    lib.recluster_cameras(c_kClusters_, c_camera_indices_cpp, c_point_indices_cpp, res_to_cluster_by_landmark_deg_out)

    res_to_cluster_by_landmark_out_ = fillPythonVecSimple(res_to_cluster_by_landmark_deg_out)

    return res_to_cluster_by_landmark_out_

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

def torchSingleResiduum(camera_params_, point_params_, p2d):
    angle_axis = camera_params_[:, :3] * c02_mult
    points_cam = AngleAxisRotatePoint(angle_axis, point_params_)
    points_cam[:,0:2] = points_cam[:,0:2] + camera_params_[:, 3:5] * c34_mult
    points_cam[:,2] = points_cam[:,2] + camera_params_[:, 5] * c5_mult
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f = camera_params_[:, 6] * c6_mult
    k1 = camera_params_[:, 7] * c7_mult
    k2 = camera_params_[:, 8] * c8_mult
    r2 = points_projX * points_projX + points_projY * points_projY
    distortion = 1.0 + r2 * (k1 + k2 * r2)
    points_reprojX = points_projX * distortion * f
    points_reprojY = points_projY * distortion * f
    # distortion = f + r2 * (k1 + k2 * r2)
    # points_reprojX = points_projX * distortion
    # points_reprojY = points_projY * distortion
    resX = (points_reprojX - p2d[:, 0]).reshape((p2d.shape[0], 1))
    resY = (points_reprojY - p2d[:, 1]).reshape((p2d.shape[0], 1))
    residual = torch.cat([resX[:,], resY[:,]], dim=1)
    return residual

def torchSingleResiduumX(camera_params, point_params, p2d) :
    angle_axis = camera_params[:,:3] * c02_mult
    points_cam = AngleAxisRotatePoint(angle_axis, point_params)
    points_cam[:,0:2] = points_cam[:,0:2] + camera_params[:, 3:5] * c34_mult
    points_cam[:,2] = points_cam[:,2] + camera_params[:, 5] * c5_mult
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f  = camera_params[:, 6] * c6_mult
    k1 = camera_params[:, 7] * c7_mult
    k2 = camera_params[:, 8] * c8_mult
    r2 = points_projX*points_projX + points_projY*points_projY
    distortion = 1. + r2 * (k1 + k2 * r2)
    points_reprojX = points_projX * distortion * f # if f is negative, points_reprojX is as well. -> negate p2d and f.
    # distortion = f + r2 * (k1 + k2 * r2)
    # points_reprojX = points_projX * distortion
    resX = (points_reprojX-p2d[:,0])
    return resX

def torchSingleResiduumY(camera_params, point_params, p2d) :
    angle_axis = camera_params[:,:3] * c02_mult
    points_cam = AngleAxisRotatePoint(angle_axis, point_params)
    points_cam[:,0:2] = points_cam[:,0:2] + camera_params[:, 3:5] * c34_mult
    points_cam[:,2] = points_cam[:,2] + camera_params[:, 5] * c5_mult
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f  = camera_params[:, 6] * c6_mult
    k1 = camera_params[:, 7] * c7_mult
    k2 = camera_params[:, 8] * c8_mult
    r2 = points_projX*points_projX + points_projY*points_projY
    distortion = 1 + r2 * (k1 + k2 * r2)
    points_reprojY = points_projY * distortion * f
    # distortion = f + r2 * (k1 + k2 * r2)
    # points_reprojY = points_projY * distortion
    resY = (points_reprojY-p2d[:,1])
    return resY

# scaling should be per UNorm.data.reshape(9,-1)[cam index,:], even torch no grad
def torchSingleResiduumScaled(camera_params_, point_params_, p2d, scaling, scalingP):
    camera_params_ = camera_params_ * scaling
    point_params_ = point_params_ * scalingP
    angle_axis = camera_params_[:, :3] #* scaling[:,:3]
    points_cam = AngleAxisRotatePoint(angle_axis, point_params_)
    points_cam[:,0:2] = points_cam[:,0:2] + camera_params_[:, 3:5] #* scaling[:, 3:5]
    points_cam[:,2] = points_cam[:,2] + camera_params_[:, 5] #* scaling[:, 5]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f  = camera_params_[:, 6] #* scaling[:, 6]
    k1 = camera_params_[:, 7] #* scaling[:, 7]
    k2 = camera_params_[:, 8] #* scaling[:, 8]
    r2 = points_projX * points_projX + points_projY * points_projY
    distortion = 1.0 + r2 * (k1 + k2 * r2)
    points_reprojX = points_projX * distortion * f
    points_reprojY = points_projY * distortion * f
    # distortion = f + r2 * (k1 + k2 * r2)
    # points_reprojX = points_projX * distortion
    # points_reprojY = points_projY * distortion
    resX = (points_reprojX - p2d[:, 0]).reshape((p2d.shape[0], 1))
    resY = (points_reprojY - p2d[:, 1]).reshape((p2d.shape[0], 1))
    residual = torch.cat([resX[:,], resY[:,]], dim=1)
    return residual

def torchSingleResiduumXScaled(camera_params, point_params, p2d, scaling, scalingP) :
    angle_axis = camera_params[:,:3] * scaling[:,:3]
    point_params = point_params * scalingP
    points_cam = AngleAxisRotatePoint(angle_axis, point_params)
    points_cam[:,0:2] = points_cam[:,0:2] + camera_params[:, 3:5] * scaling[:, 3:5]
    points_cam[:,2] = points_cam[:,2] + camera_params[:, 5] * scaling[:, 5]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f  = camera_params[:, 6] * scaling[:, 6]
    k1 = camera_params[:, 7] * scaling[:, 7]
    k2 = camera_params[:, 8] * scaling[:, 8]
    r2 = points_projX*points_projX + points_projY*points_projY
    # distortion = f + r2 * (k1 + k2 * r2)
    # points_reprojX = points_projX * distortion
    distortion = 1. + r2 * (k1 + k2 * r2)
    points_reprojX = points_projX * distortion * f
    resX = points_reprojX-p2d[:,0]
    return resX

def torchSingleResiduumYScaled(camera_params, point_params, p2d, scaling, scalingP) :
    angle_axis = camera_params[:,:3] * scaling[:,:3]
    point_params = point_params * scalingP
    points_cam = AngleAxisRotatePoint(angle_axis, point_params)
    points_cam[:,0:2] = points_cam[:,0:2] + camera_params[:, 3:5] * scaling[:, 3:5]
    points_cam[:,2] = points_cam[:,2] + camera_params[:, 5] * scaling[:, 5]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f  = camera_params[:, 6] * scaling[:, 6]
    k1 = camera_params[:, 7] * scaling[:, 7]
    k2 = camera_params[:, 8] * scaling[:, 8]
    r2 = points_projX*points_projX + points_projY*points_projY
    # distortion = f + r2 * (k1 + k2 * r2)
    # points_reprojY = points_projY * distortion
    distortion = 1. + r2 * (k1 + k2 * r2)
    points_reprojY = points_projY * distortion * f
    resY = points_reprojY-p2d[:,1]
    return resY

def ComputeDerivativeMatrixInit(x0_c_, x0_l_, points_2d, camera_indices, point_indices):
    funx0_st1 = lambda X0, X1, X2: torchSingleResiduumX(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d fucntion -> grad possible
    funy0_st1 = lambda X0, X1, X2: torchSingleResiduumY(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d fucntion -> grad possible

    torch_cams = from_numpy(x0_c_.reshape(-1,9)[camera_indices[:],:])
    torch_lands = from_numpy(x0_l_.reshape(-1,3)[point_indices[:],:])
    torch_lands.requires_grad_()
    torch_cams.requires_grad_()
    torch_cams.retain_grad()
    torch_lands.retain_grad()

    torch_points_2d = from_numpy(points_2d)
    torch_points_2d.requires_grad_(False)

    resX = funx0_st1(torch_cams, torch_lands, torch_points_2d[:,:]).flatten()
    lossX = torch.sum(resX)
    lossX.backward()

    cam_grad_x = torch_cams.grad.detach().numpy().copy()
    land_grad_x = torch_lands.grad.detach().numpy().copy()

    torch_cams.grad.zero_()
    torch_lands.grad.zero_()
    resY = funy0_st1(torch_cams, torch_lands, torch_points_2d[:,:]).flatten()
    lossY = torch.sum(resY)
    lossY.backward()
    cam_grad_y = torch_cams.grad.detach().numpy().copy()
    land_grad_y = torch_lands.grad.detach().numpy().copy()

    J_pose = buildMatrixNew(cam_grad_x, cam_grad_y, camera_indices, sz=9)
    J_land = buildMatrixNew(land_grad_x, land_grad_y, point_indices, sz=3)
    fx0 = buildResiduumNew(resX.detach(), resY.detach())

    return (J_pose, J_land, fx0)

def ComputeDerivativeMatricesNew(x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d, camScale, landScale #, unique_poses_in_c_, unique_landmarks_in_c_,
):
    verbose = False
    if verbose:
        start = time.time() # this is not working at all. Slower then iteratively

    #funx0_st1 = lambda X0, X1, X2: torchSingleResiduumX(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d function -> grad possible
    #funy0_st1 = lambda X0, X1, X2: torchSingleResiduumY(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d function -> grad possible

    # camScale = 1./Unorm.data.reshape(-1,9)
    # camScale = camScale[unique_poses_in_c_]
    # camScale = from_numpy(camScale[camera_indices_[:]])
    # camScale.requires_grad_(False)

    # landScale = 1./Vnorm.data.reshape(-1,3)
    # landScale = landScale[unique_landmarks_in_c_]
    # landScale = from_numpy(landScale[point_indices_[:]]) # here direct, or not?
    # landScale.requires_grad_(False)

    funx0_st1 = lambda X0, X1, X2: torchSingleResiduumXScaled(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2), camScale, landScale)
    funy0_st1 = lambda X0, X1, X2: torchSingleResiduumYScaled(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2), camScale, landScale)

    torch_cams = x0_t_cam[camera_indices_[:],:] #x0_t[:n_cameras*9].reshape(n_cameras,9)[camera_indices[:],:]
    torch_lands = x0_t_land[point_indices_[:],:] #x0_t[n_cameras*9:].reshape(n_points,3)[point_indices[:],:]
    torch_lands.requires_grad_()
    torch_cams.requires_grad_()
    torch_cams.retain_grad()
    torch_lands.retain_grad()

    # print("camScale ", camScale)
    # print("torch_cams ", torch_cams)

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

    fx0_ = buildResiduumNew(resX.detach(), resY.detach())

    if verbose:
        print(" build Matrix & residuum took ", end-start, "s")
        end = time.time()

    return (J_pose, J_land, fx0_)

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

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

# idea is to simulate compression here only. in: 9x9 block matrix, out: 9x9 block matrix compressed. to 255 values and scale.
# we send max s as float and 8 bits per entry. since mat is symmetric -> 45 * 8 bit + 4 byte float ~ 4x less data.
# 1. find max in block 9x9 block. divide block by max.
# 2. convert block to 255 values: mult by 127, round to int, divide by 127 must by max.
def CompressBlockMatrix(M, bs):
    #Ei = np.zeros(M.shape[0])
    Mi = M.copy()
    diag = M.diagonal()
    # could be random effects also. not clear that larger is much better.
    val = 1023.#127. # 255 is better ? or lower is same ? 255: 12 floats for 45. 511: 14 floats for 45 1/3rd. 127: 11.25 floats for 45: 10 floats
    values = []
    if bs > 1:
        bs2 = bs * bs

        symmetric = True
        mat = M.data[0 : bs2].reshape(bs, bs)
        if not check_symmetric(mat):
            symmetric = False

        for i_ in range(int(M.data.shape[0] / bs2)):
            mat = Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs)

            d = diag[bs * i_ : bs * i_ + bs] # > 0 by definition.
            scale = np.max(np.abs(d)) / np.min(np.abs(d)) # diagonal .. or hmm. overestimate wrt diag is ok.
            # maxrow = np.sum(np.abs(mat), axis=0)
            # scale = np.max(maxrow) / np.min(maxrow) # ?

            adaptive = False #True # maye not even needed adaptively then we might use more bits?
            logScale = np.round(np.log2(scale))
            adaptive_limit = 128. # if < 128: 1723 fails.
            if adaptive:
                val = np.maximum(adaptive_limit, np.power(2., 4 + logScale)) - 1 # 6: ok, 4: ok for s * val * 1e-2 ?
                if val > adaptive_limit:
                    values.append(val)

            s = np.max(np.abs(mat)) / val # maybe even 63/ 31? save bit2

            # temp = mat.copy()
            mat = s * np.round(mat / s) # float?
            # see quantization: can learn given data: H*x ~ H^*x for data distribution x. |Hx - H^x + V(x)|, V in [0,1]
            # diffRow = np.sum(np.abs(mat-temp), axis=0) # add diff to diag?
            # diffRow = np.sum(np.fmax(np.abs(mat)-np.abs(temp), 0), axis=0) # off diag  rounded up.

            #print(np.sum(np.abs(mat - temp) * val)) # small diff: ok
            # diff = np.abs(mat - temp) / s
            # if np.max(diff) > 1.01: # or logScale > 11:
            #     print(logScale, "--------", s)
            #     print(diff) # small diff: ok
            #     print(Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs))
            #     print(mat)
            #     print(Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs) / s)
            #     print(np.round(mat / s))
            #     print("--------", logScale )

            # as in max is > 127 times larger than smallest diag value. This DEFINES the scale we need.
            # problem is that last 3x3 part is so small not interacting with rest.
            # hence compression all entries leasd to issues there. would need to compress each block separately.
            # [[ 2.96  0.87 -2.96 -0.03  2.96  2.01  0.01  0.00  0.00]
            # [ 0.87  2.91 -0.92 -2.81  0.85 -1.33 -0.00  0.00  0.00]
            # [-2.96 -0.92  2.96  0.07 -2.96 -1.98 -0.01 -0.00 -0.00]
            # [-0.03 -2.81  0.07  2.96 -0.00  2.02  0.01  0.00 -0.00]
            # [ 2.96  0.85 -2.96 -0.00  2.96  2.03  0.01  0.00  0.00]
            # [ 2.01 -1.33 -1.98  2.02  2.03  2.76  0.02  0.01  0.01]
            # [ 0.01 -0.00 -0.01  0.01  0.01  0.02  0.03  0.03  0.02]
            # [ 0.00  0.00 -0.00  0.00  0.00  0.01  0.03  0.03  0.03]
            # [ 0.00  0.00 -0.00 -0.00  0.00  0.01  0.02  0.03  0.03]]

            # compress can lead to singular. add to diagonal!
            if False:
                for j_ in range(bs): # speed?
                    # mat[j_, j_] += np.max(np.abs(d)) * 1e-2 # works well -- not clear this makes more sense.
                    # mat[j_, j_] += s * val * 1e-2 # more stable perf. cmp 52, 1723: this is really bad
                    # would make most sense: idea pos definiteness is lost by rounding. add a bit to diag could be sufficient.
                    # below went through. not really sure makes sense. but it works.

                    # BEST SO FAR. 4 best so far. better than 3 & 5. 3 sucks : 1723. 5:
                    # assumption need to add a bit to ensure it stays positive definite. Here on value on diag -- not any value.
                    mat[j_, j_] = np.maximum((mat[j_, j_]) * (1. + 3. / val), 1e-8)

                    # mat[j_, j_] += np.maximum(diffRow[j_], 1e-8) + np.maximum(0, np.abs(temp[j_,j_])-np.abs(mat[j_,j_]))
                    # mat[j_, j_] += np.maximum(np.max(diffRow[j_]), 1e-8)
                    # mat[j_, j_] += np.maximum(diffRow[j_], 1e-8) + 2 * np.maximum(0, np.abs(temp[j_,j_]) - np.abs(mat[j_,j_]))

                    # aehm. we could inc off diag by 1/2 * quant error * 8 (elements)
                    # = + 4 / s = 4 * val / maxs no 1 / 2 * max / val is quant error.
                    # + 4 * s . maybe just + s, or a bit more.
                    #mat[j_, j_] += 1. * s # shoulld've added to beg with.

                    # sum abs row on diff. add diff to diag?

                    # might still be non positive? What to do? check - if not add more?
                nev = True
                if not symmetric:
                    mat = np.fliplr(mat)
                while nev: # usually 1 iteration.
                    nev = False
                    if False: # TODO test true/false
                        #evs, evv = eigh(mat)
                        evs = eigvalsh(mat)
                        if np.min(evs) < 0: # hope it is rare. we round up?
                            nev = True
                            #print("negative eigenvalue in compression ", evs)
                            for j_ in range(bs):
                                mat[j_, j_] += s # miss 3068, 646, 1266 .. could also mult by (1+1/val) ? in general could mult by 1+be compared add be .. or alawys add s here.
                    else: # advantage just do once, disadvatage comute evs as well.
                        evs, evv = eigh(mat)
                        evs = np.fmax(evs, np.abs(evs[bs-1]) * 1e-4) # SO WEIRD 1e-4 is HUGE and fails? 1e-3 - 5e-4?
                        #evs = np.fmax(evs, 1e-6) # ?
                        mat = evv.dot(diag_sparse(evs) * evv.transpose())
                if not symmetric:
                    mat = np.fliplr(mat)

            else:
                if not symmetric:
                    mat = np.fliplr(mat)
                evs, evv = eigh(mat)
                evs = np.fmax(1.0 * evs, np.abs(evs[bs-1]) * 3e-4) # SO WEIRD 1e-4 is HUGE and fails? 1e-3 - 5e-4?
                #evs = np.fmax(evs, 1e-6) # ?
                #print("evs ", evs[bs-1] / evs)
                #print("evv ", evv[bs-1])
                #print("evv ", evv)
                mat = evv.dot(diag_sparse(evs) * evv.transpose())
                if not symmetric:
                    mat = np.fliplr(mat)

            mat += np.minimum(1e-2, s) * 1e-16 * np.ones([bs, bs]) # some 0's problem. adding temp 'worked'. failing 245
            #print(i_, " ", mat.shape, " ", Mi.data[bs2 * i_ : bs2 * i_ + bs2].shape)
            Mi.data[bs2 * i_ : bs2 * i_ + bs2] = mat.flatten()
            #print(i_, " \n", mat, " \n", Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs))

            #Ei[bs * i_ : bs * i_ + bs] = s * 127. * 1e-6
        #print("Mi ", Mi)
        #print("Ei ", diag_sparse(Ei))
        #Mi = Mi + diag_sparse(Ei) # 0 becomes 0 by this must add in data
        #print("Mi ", Mi)
        # add diag mat.
        #print(len(values))
    return Mi

def PreCompressBlockMatrix(M, bs):
    #Ei = np.zeros(M.shape[0])
    Mi = M.copy()
    diag = M.diagonal()
    # could be random effects also. not clear that larger is much better.
    # 1023: 10 bits for 32 bits (float) + 1 = 10/32 * 45 + 1 = 15. add/reduce 45 bits per +- 1 bit
    val = 1023. # 255 is better ? or lower is same ? 255: 12 floats for 45. 511: 13.6 floats for 45 1/3rd. 127: 11.25 floats for 45: 10 floats, 1023: 15 floats for 45
    values = []
    if bs > 1:
        bs2 = bs * bs

        symmetric = True
        mat = M.data[0 : bs2].reshape(bs, bs)
        if not check_symmetric(mat):
            symmetric = False

        for i_ in range(int(M.data.shape[0] / bs2)):
            mat = Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs)

            d = diag[bs * i_ : bs * i_ + bs] # > 0 by definition.
            scale = np.max(np.abs(d)) / np.min(np.abs(d)) # diagonal .. log appears All between 1 and 2. (at end maybe not)
            #maxrow = np.sum(np.abs(mat), axis=0) # some become HUGE.
            #scale = np.max(maxrow) / np.min(maxrow) # ?
            #scale = np.max(np.abs(mat))

            adaptive = False # maye not even needed adaptively then we might use more bits?
            logScale = np.round(np.log2(scale))
            adaptive_limit = 128. # if < 128: 1723 fails.
            if adaptive:
                val = np.maximum(adaptive_limit, np.power(2., 8 + logScale)) - 1 # 6: ok, 4: ok for s * val * 1e-2 ?
                if True or val > adaptive_limit:
                    values.append(val)

            s = np.max(np.abs(mat)) / val # maybe even 63/ 31? save bit2

            #temp = mat.copy()
            mat = s * np.round(mat / s) # float?
            #print(np.sum(np.abs(mat - temp) * val)) # small diff: ok
            # diff = np.abs(mat - temp) / s
            # if np.max(diff) > 1.01: # or logScale > 11:
            #     print(logScale, "--------", s)
            #     print(diff) # small diff: ok
            #     print(Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs))
            #     print(mat)
            #     print(Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs) / s)
            #     print(np.round(mat / s))
            #     print("--------", logScale )

            # compress can lead to singular. add to diagonal!
            if False:
                for j_ in range(bs): # speed?
                    # mat[j_, j_] += np.max(np.abs(d)) * 1e-2 # works well -- not clear this makes more sense.
                    # mat[j_, j_] += s * val * 1e-2 # more stable perf. cmp 52, 1723: this is really bad
                    # would make most sense: idea pos definiteness is lost by rounding. add a bit to diag could be sufficient.
                    # below went through. not really sure makes sense. but it works.
                    mat[j_, j_] = np.abs(mat[j_, j_]) * (1. + 4. / val) # assumption need to add a bit to ensure it stays positive definite. Here on value on diag -- not any value.
                    # aehm. we could inc off diag by 1/2 * quant error * 8 (elements)
                    # = + 4 / s = 4 * val / maxs no 1 / 2 * max / val is quant error.
                    # + 4 * s . maybe just + s, or a bit more.
                    # mat[j_, j_] += 1.0 * s # shoulld've added to beg with.
            else:
                if not symmetric:
                    mat = np.fliplr(mat)
                evs, evv = eigh(mat)
                # 127 -> 1e-3
                # 255, 511 -> 5e-4
                # 1023-> 2e-4 / 3e-4 .. use globalIt, small at first biger late?
                evs = np.fmax(evs, np.abs(evs[bs-1]) * 3e-4) # SO WEIRD 1e-4 is HUGE and fails? 1e-3 - 5e-4? Test on 52.
                # evs = np.fmax(evs, 1e-4) # i do not get it.
                #print("evs ", evs[bs-1] / evs)
                #print("evv ", evv[bs-1])
                #print("evv ", evv)
                mat = evv.dot(diag_sparse(evs) * evv.transpose())
                if not symmetric:
                    mat = np.fliplr(mat)

            mat += np.minimum(1e-4, s) * 1e-16 * np.ones([bs, bs]) # some 0's problem. adding temp 'worked'. failing 245
            Mi.data[bs2 * i_ : bs2 * i_ + bs2] = mat.flatten()
            #Ei[bs * i_ : bs * i_ + bs] = s * 127. * 1e-6
        #print("Mi ", Mi)
        #print("Ei ", diag_sparse(Ei))
        #Mi = Mi + diag_sparse(Ei) # 0 becomes 0 by this must add in data
        #print("Mi ", Mi)
        # add diag mat.
        #print((values))
    return Mi

def PostCompressBlockMatrix(M, bs):
    #Ei = np.zeros(M.shape[0])
    Mi = M.copy()
    diag = M.diagonal()
    # could be random effects also. not clear that larger is much better.
    val = 1023.#127. # 255 is better ? or lower is same ? 255: 12 floats for 45. 511: 14 floats for 45 1/3rd. 127: 11.25 floats for 45: 10 floats
    values = []
    if bs > 1:
        bs2 = bs * bs

        symmetric = True
        mat = M.data[0 : bs2].reshape(bs, bs)
        if not check_symmetric(mat):
            symmetric = False

        for i_ in range(int(M.data.shape[0] / bs2)):
            mat = Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs)

            d = diag[bs * i_ : bs * i_ + bs] # > 0 by definition.
            scale = np.max(np.abs(d)) / np.min(np.abs(d)) # diagonal .. or hmm. overestimate wrt diag is ok.
            # maxrow = np.sum(np.abs(mat), axis=0)
            # scale = np.max(maxrow) / np.min(maxrow) # ?

            adaptive = False #True # maye not even needed adaptively then we might use more bits?
            logScale = np.round(np.log2(scale))
            adaptive_limit = 128. # if < 128: 1723 fails.
            if adaptive:
                val = np.maximum(adaptive_limit, np.power(2., 4 + logScale)) - 1 # 6: ok, 4: ok for s * val * 1e-2 ?
                if val > adaptive_limit:
                    values.append(val)

            s = np.max(np.abs(mat)) / val # maybe even 63/ 31? save bit2

            # temp = mat.copy()
            mat = s * np.round(mat / s) # float?
            # see quantization: can learn given data: H*x ~ H^*x for data distribution x. |Hx - H^x + V(x)|, V in [0,1]
            # diffRow = np.sum(np.abs(mat-temp), axis=0) # add diff to diag?
            # diffRow = np.sum(np.fmax(np.abs(mat)-np.abs(temp), 0), axis=0) # off diag  rounded up.

            #print(np.sum(np.abs(mat - temp) * val)) # small diff: ok
            # diff = np.abs(mat - temp) / s
            # if np.max(diff) > 1.01: # or logScale > 11:
            #     print(logScale, "--------", s)
            #     print(diff) # small diff: ok
            #     print(Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs))
            #     print(mat)
            #     print(Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs) / s)
            #     print(np.round(mat / s))
            #     print("--------", logScale )

            # as in max is > 127 times larger than smallest diag value. This DEFINES the scale we need.
            # problem is that last 3x3 part is so small not interacting with rest.
            # hence compression all entries leasd to issues there. would need to compress each block separately.
            # [[ 2.96  0.87 -2.96 -0.03  2.96  2.01  0.01  0.00  0.00]
            # [ 0.87  2.91 -0.92 -2.81  0.85 -1.33 -0.00  0.00  0.00]
            # [-2.96 -0.92  2.96  0.07 -2.96 -1.98 -0.01 -0.00 -0.00]
            # [-0.03 -2.81  0.07  2.96 -0.00  2.02  0.01  0.00 -0.00]
            # [ 2.96  0.85 -2.96 -0.00  2.96  2.03  0.01  0.00  0.00]
            # [ 2.01 -1.33 -1.98  2.02  2.03  2.76  0.02  0.01  0.01]
            # [ 0.01 -0.00 -0.01  0.01  0.01  0.02  0.03  0.03  0.02]
            # [ 0.00  0.00 -0.00  0.00  0.00  0.01  0.03  0.03  0.03]
            # [ 0.00  0.00 -0.00 -0.00  0.00  0.01  0.02  0.03  0.03]]

            # compress can lead to singular. add to diagonal!
            if False:
                for j_ in range(bs): # speed?
                    # mat[j_, j_] += np.max(np.abs(d)) * 1e-2 # works well -- not clear this makes more sense.
                    # mat[j_, j_] += s * val * 1e-2 # more stable perf. cmp 52, 1723: this is really bad
                    # would make most sense: idea pos definiteness is lost by rounding. add a bit to diag could be sufficient.
                    # below went through. not really sure makes sense. but it works.

                    # BEST SO FAR. 4 best so far. better than 3 & 5. 3 sucks : 1723. 5:
                    # assumption need to add a bit to ensure it stays positive definite. Here on value on diag -- not any value.
                    mat[j_, j_] = np.maximum((mat[j_, j_]) * (1. + 3. / val), 1e-8)

                    # mat[j_, j_] += np.maximum(diffRow[j_], 1e-8) + np.maximum(0, np.abs(temp[j_,j_])-np.abs(mat[j_,j_]))
                    # mat[j_, j_] += np.maximum(np.max(diffRow[j_]), 1e-8)
                    # mat[j_, j_] += np.maximum(diffRow[j_], 1e-8) + 2 * np.maximum(0, np.abs(temp[j_,j_]) - np.abs(mat[j_,j_]))

                    # aehm. we could inc off diag by 1/2 * quant error * 8 (elements)
                    # = + 4 / s = 4 * val / maxs no 1 / 2 * max / val is quant error.
                    # + 4 * s . maybe just + s, or a bit more.
                    #mat[j_, j_] += 1. * s # shoulld've added to beg with.

                    # sum abs row on diff. add diff to diag?

                    # might still be non positive? What to do? check - if not add more?
                nev = True
                if not symmetric:
                    mat = np.fliplr(mat)
                while nev: # usually 1 iteration.
                    nev = False
                    if False: # TODO test true/false
                        #evs, evv = eigh(mat)
                        evs = eigvalsh(mat)
                        if np.min(evs) < 0: # hope it is rare. we round up?
                            nev = True
                            #print("negative eigenvalue in compression ", evs)
                            for j_ in range(bs):
                                mat[j_, j_] += s # miss 3068, 646, 1266 .. could also mult by (1+1/val) ? in general could mult by 1+be compared add be .. or alawys add s here.
                    else: # advantage just do once, disadvatage comute evs as well.
                        evs, evv = eigh(mat)
                        evs = np.fmax(evs, np.abs(evs[bs-1]) * 1e-4) # SO WEIRD 1e-4 is HUGE and fails? 1e-3 - 5e-4?
                        #evs = np.fmax(evs, 1e-6) # ?
                        mat = evv.dot(diag_sparse(evs) * evv.transpose())
                if not symmetric:
                    mat = np.fliplr(mat)

            else:
                if not symmetric:
                    mat = np.fliplr(mat)
                evs, evv = eigh(mat)
                evs = np.fmax(evs, np.abs(evs[bs-1]) * 1e-4) # 3e-4 is base. 1e-6 too small. 1e-5? - 1e-3?
                #evs = np.fmax(evs, 1e-6) # ?
                #print("evs ", evs[bs-1] / evs)
                #print("evv ", evv[bs-1])
                #print("evv ", evv)
                mat = evv.dot(diag_sparse(evs) * evv.transpose())
                if not symmetric:
                    mat = np.fliplr(mat)

            mat += np.minimum(1e-2, s) * 1e-16 * np.ones([bs, bs]) # some 0's problem. adding temp 'worked'. failing 245
            #print(i_, " ", mat.shape, " ", Mi.data[bs2 * i_ : bs2 * i_ + bs2].shape)
            Mi.data[bs2 * i_ : bs2 * i_ + bs2] = mat.flatten()
            #print(i_, " \n", mat, " \n", Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs))

            #Ei[bs * i_ : bs * i_ + bs] = s * 127. * 1e-6
        #print("Mi ", Mi)
        #print("Ei ", diag_sparse(Ei))
        #Mi = Mi + diag_sparse(Ei) # 0 becomes 0 by this must add in data
        #print("Mi ", Mi)
        # add diag mat.
        #print(len(values))
    return Mi

# Always round down.
def LowCompressBlockMatrix(M, bs, val = 127.):
    Mi = M.copy()
    Bi = M.copy()
    # val = 127. # 255 is better ? or lower is same ? 255: 12 floats for 45. 511: 14 floats for 45 1/3rd. 127: 11.25 floats for 45: 10 floats
    bs2 = bs * bs
    scales = np.zeros(int(M.data.shape[0] / bs2))
    if bs > 1:

        symmetric = True
        mat = M.data[0 : bs2].reshape(bs, bs)
        if not check_symmetric(mat):
            symmetric = False

        for i_ in range(int(M.data.shape[0] / bs2)):
            mat = Mi.data[bs2 * i_ : bs2 * i_ + bs2].reshape(bs, bs)

            s = np.max(np.abs(mat)) / val
            scales[i_] = s
            matRound = np.round(mat / s)
            mat = np.round(mat / s - 0.5) #
            # see quantization: can learn given data: H*x ~ H^*x for data distribution x. |Hx - H^x + V(x)|, V in [0,1]

            # compress can lead to singular. add to diagonal!
            if False:
                for i_ in range(bs): # speed?
                    # BEST SO FAR. 4 best so far.
                    mat[i_, i_] = np.maximum((mat[i_, i_]) * (1. + 4. / val), 1e-8) # assumption need to add a bit to ensure it stays positive definite. Here on value on diag -- not any value.

            mat += 1e-16 * np.ones([bs, bs]) # some 0's problem. adding temp 'worked'. failing 245
            matRound += 1e-16 * np.ones([bs, bs])
            Mi.data[bs2 * i_ : bs2 * i_ + bs2] = mat.flatten()
            Bi.data[bs2 * i_ : bs2 * i_ + bs2] = matRound.flatten()
    return Mi, scales, Bi

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

def blockEigenvalue(M, bs):
    Ei = np.zeros(M.shape[0])
    if bs > 1:
        bs2 = bs * bs

        symmetric = True
        mat = M.data[0 : bs2].reshape(bs, bs)
        if not check_symmetric(mat):
            symmetric = False

        for i_ in range(int(M.data.shape[0] / bs2)):
            mat = M.data[bs2 * i_ : bs2 * i_ + bs2].copy().reshape(bs, bs)
            if not symmetric:
                mat = np.fliplr(mat)
            evs = eigvalsh(mat)
            Ei[bs * i_ : bs * i_ + bs] = evs[bs - 1] # largest
            # Ei[bs * i_ : bs * i_ + bs] = np.fmax(evs, 1e-3 * evs[bs - 1]), bs but maybe max diag is ok
        Ei = diag_sparse(Ei)
    else:
        Ei = M.copy()
    return Ei

def maxDiagA(M, bs):
    Ei = np.zeros(M.shape[0])
    if bs > 1:
        diag = M.diagonal()
        for i_ in range(int(diag.shape[0] / bs)):
            maxDiag_ = diag[bs * i_ : bs * i_ + bs].copy()
            Ei[bs * i_ : bs * i_ + bs] = np.max(maxDiag_) # largest
        Ei = diag_sparse(Ei)
    else:
        Ei = diag_sparse(M.diag())
    return Ei

def maxDiag(M, bs): # this should be an advantage now ?!
    Ei = np.zeros(M.shape[0])
    if bs > 1:
        diag = M.diagonal()
        for i_ in range(int(diag.shape[0] / bs)):
            diag_ = diag[bs * i_ : bs * i_ + bs].copy()
            #Ei[bs * i_ : bs * i_ + bs] = np.max(diag_) # largest
            # absurd but diag is worse no matter what.
            Ei[bs * i_ : bs * i_ + bs] = diag_ + 1e-2 * np.max(diag_) # 1e-4 >> 1e-2, 1e-8
        Ei = diag_sparse(Ei)
    else:
        Ei = diag_sparse(M.diag())
    return Ei

# What about max on row not diag. if diagonally dominat its the same.
def maxRow(M, bs):
    Ei = np.zeros(M.shape[0])
    if bs > 1:
        bs2 = bs * bs

        symmetric = True
        mat = M.data[0 : bs2].reshape(bs, bs)
        if not check_symmetric(mat):
            symmetric = False

        for i in range(int(M.data.shape[0] / bs2)):
            mat = M.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs).copy()
            if not symmetric:
                mat = np.fliplr(mat)
            # if bs == 9:
            #     print(mat)
            maxrow = np.sum(np.abs(mat), axis=0) # symmetric: axis does not matter
            # Degen cam, 1e-2 significant better start than 1e-3/1e-4. but end result is the same.
            maxrow = maxrow + 1e-2 * np.max(maxrow) # singular cam (<5 lms?) want higher 1e-k?
            Ei[bs * i : bs * i + bs] = maxrow
        Ei = diag_sparse(Ei)
    else:
        Ei = diag_sparse(Ei)
    return Ei

# analysis
def blockEigenvalueSet(M, bs):
    Ei = np.zeros((bs, int(M.data.shape[0] / bs)))
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            mat = M.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs).copy()
            if not check_symmetric(mat):
                mat = np.fliplr(mat)
            # print(i, " ", mat)
            evs = eigvalsh(mat)
            # if evs[0] <0:
            #    mat = np.fliplr(mat)
            #    evs = eigvalsh(mat)
            Ei[:, i] = evs
    return Ei

def minEigenvalues(M, bs, get_max_evs=False):
    bs2 = bs * bs
    Ei = np.zeros(int(M.data.shape[0] / bs2))
    for i in range(int(M.data.shape[0] / bs2)):
        mat = M.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs).copy()
        if not check_symmetric(mat):
            mat = np.fliplr(mat)
        # print(i, " ", mat)
        evs = eigvalsh(mat)
        if get_max_evs:
            Ei[i] = np.maximum(1e-16, evs[bs-1])
        else:
            Ei[i] = np.maximum(1e-16, evs[0])
    return Ei

def blockEigenvalueWhereNeeded(M, bs, thresh = 1e-6):
    Ei = np.zeros(M.shape[0])
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            mat = M.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs).copy()
            if not check_symmetric(mat):
                mat = np.fliplr(mat)
            # print(i, " ", mat)
            evs = eigvalsh(mat)
            # if evs[0] <0:
            #    mat = np.fliplr(mat)
            #    evs = eigvalsh(mat)
            if evs[0] < thresh * evs[bs-1]: # all smaller horror.
                Ei[bs*i:bs*i+bs] = evs[bs-1] # largest
            else:
                Ei[bs*i:bs*i+bs] = evs[0] # smallest

        Ei = diag_sparse(Ei)
    else:
        Ei = M.copy()

    # todo: eval super safe method since symmetric FAIL again.
    # idea was that non isotropic structure might hurting bfgs. but as bad as without
    #maxEv = np.max(Ei.data)
    #Ei.data[:] = 0.1 * maxEv # 0.1 worked
    #Ei.data[:] *= 0.125 #0.25 promising # 0.1 already too low. could try line-search idea
    return Ei

def minmaxEv(M, bs):
    maxE = np.zeros(int(M.shape[0]/bs))
    minE = np.zeros(int(M.shape[0]/bs))
    if bs > 1:
        bs2 = bs * bs

        symmetric = True
        mat = M.data[0 : bs2].reshape(bs, bs)
        if not check_symmetric(mat):
            symmetric = False

        for i in range(int(M.data.shape[0] / bs2)):
            mat = M.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs).copy()
            if not symmetric:
                mat = np.fliplr(mat)
            evs = eigvalsh(mat)
            maxE[i] = evs[bs-1]
            minE[i] = evs[0]
            # if evs[0] <0:
            #    #print("evs[0] ", evs[0], " " ,mat)
            #    mat = np.fliplr(mat)
            #    evs = eigvalsh(mat)
            #    maxE[i] = evs[bs-1]
            #    minE[i] = evs[0]

    return maxE, minE

def blockEigenvalueFull(M, bs, x0_t_cam_):
    Ei = M.copy()
    if bs > 1:
        bs2 = bs * bs

        symmetric = True
        mat = M.data[0 : bs2].reshape(bs, bs)
        if not check_symmetric(mat):
            symmetric = False

        for i in range(int(M.data.shape[0] / bs2)):
            mat = M.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs)
            flip = False
            if not symmetric:
                mat = np.fliplr(mat)
                flip = True
            evs, evv = eigh(mat)
            evs = np.fmax(evs, evs[bs-1] * 1e-6) # e.g. ?
            #print("evs ", evs[bs-1] / evs)
            #print("evv ", evv[bs-1])
            #print("evv ", evv)
            #print(" cam " , x0_t_cam_[i,:])
            mat = evv.dot(diag_sparse(evs) * evv.transpose())
            if flip:
                mat = np.fliplr(mat)
            Ei.data[bs2 * i : bs2 * i + bs2] = mat.flatten()
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

def mult_selected_blocks(M, block_selection_, v, bs):
    Mi = M.copy()
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            if block_selection_[i] == True:
                Mi.data[bs2 * i : bs2 * i + bs2] *= v
    else:
        Mi = M.copy()
        for i in range(int(M.data.shape[0])):
            if block_selection_[i] == True:
                Mi.data[i : i + 1] *= v
    return Mi

def stop_criterion(delta, delta_i, i):
    # lower (1e-4) can be worse? maybe just the parts / how parts are.
    eps = 1e-3 #1e-2 used in paper, tune. might allow smaller as faster?
    return (i+1) * delta_i < eps * delta

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

# test Loop over L0=x, L=y here. Likely best to do grid search to get an idea. model as exp(-poly(L,it))
def solveByGDNesterov(Ul, W, Vli, bS, m):
    Lip = 0.9 # 100 -> 1. # TODO: play, find out how to progress over time.
    lambda0 = (1.+np.sqrt(5.)) / 2. # l=0 g=1, 0, .. L0=1 g = 0,..

    Uli = blockInverse(Ul, 9)
    ubs = - Uli * bS
    xk = - ubs
    y0 = - ubs

    verbose = False
    if verbose:
        costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
        print("-1 gd cost ", costk)

    for it__ in range(m):
        lambda1 = (1 + np.sqrt(1 + 4 * lambda0**2)) / 2
        gamma = (1-lambda0) / lambda1
        lambda0 = lambda1

        #( I - Uli * W * Vli * W.transpose())
        g = xk - Uli * ( W * (Vli * (W.transpose() * xk))) + ubs
        yk = xk - 1/Lip * g
        xk = (1-gamma) * yk + gamma * y0
        y0 = yk

        if verbose:
            # eq is Ul [I - Uli * W * Vli * W.transpose()] x = b
            costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
            print(it__, " gd cost ", costk)

        if stop_criterion(np.linalg.norm(xk, 2), np.linalg.norm(1/Lip * g, 2), it__):
            return xk, it__
    return xk, it__

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

    res_indices_in_cluster_, kClusters = cluster_covis_lib(kClusters_, pre_merges, camera_indices_, point_indices_)
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

    if False:
        res_to_cluster_by_landmark_, point_indices_already_covered_, covered_landmark_indices_c_ = \
            process_cluster_lib(num_lands, num_res, kClusters, point_indices_in_cluster_, res_indices_in_cluster_, point_indices_)
    else: # avoid cameras with few evidence / singular updates.
        res_to_cluster_by_landmark_, point_indices_already_covered_, covered_landmark_indices_c_ = \
            post_process_cluster_lib(num_lands, num_res, kClusters, point_indices_in_cluster_, res_indices_in_cluster_, point_indices_, camera_indices_)

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

def cluster_deg_by_landmark(camera_indices_, points_2d_, point_indices_, kClusters_):
    num_res = camera_indices_.shape[0]
    num_cams = np.unique(camera_indices_).shape[0]
    num_lands = np.unique(point_indices_).shape[0]
    print("number of residuum: ", num_res)

    res_to_cluster_by_landmark_ = deg_process_cluster_lib(kClusters_, point_indices_, camera_indices_)

    # we only case about covered_landmark_indices_c_
    # 1. distribute residuals by occurence of above per cluster: res_to_cluster_by_landmark_: res -> cluster covers all landmarks exculsively. to test.
    # 2. landmarks per cluster are exclusive, but use whole cams per cluster (simpler)
    # 3. need indices of cameras utilized in cluster and stepsizes per cam in cluster
    # point_indices_already_covered_ : here exclusively covered by cluster
    camera_indices_in_cluster_ = []
    points_2d_in_cluster_ = []
    point_indices_in_cluster_ = []
    for ci in range(kClusters):
        ids_of_res_in_cluster = res_to_cluster_by_landmark_ == ci
        camera_indices_in_cluster_.append(camera_indices_[ids_of_res_in_cluster])
        points_2d_in_cluster_.append(points_2d_[ids_of_res_in_cluster])
        point_indices_in_cluster_.append(point_indices_[ids_of_res_in_cluster])
        print("===== Cluster ", ci , " covers ", points_2d_in_cluster_[ci].shape, "residuals ",
              np.unique(point_indices_in_cluster_[ci]).shape, " of ", num_lands, " landmarks ",
              np.unique(camera_indices_in_cluster_[ci]).shape, " of ", num_cams, "cameras ")

    # check if indices are disjoint / local vs global / sums up to n_points
    sum_cams_cover = 0
    sum_landmarks_cover = 0
    for ci in range(kClusters):
        sum_cams_cover += np.unique(camera_indices_in_cluster_[ci]).shape[0]
        sum_landmarks_cover += np.unique(point_indices_in_cluster_[ci]).shape[0]
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
def average_cameras_new(
    camera_indices_in_cluster_, poses_in_cluster_, poses_s_in_cluster_, L_in_cluster_, UL_in_cluster_, nabla_p_in_cluster_):
    num_cameras = poses_in_cluster_[0].shape[0]
    sum_D_u2_s = np.zeros(num_cameras * 9)
    sum_constant_term = 0
    UL_zeros_in_cluster_ = []

    # Here or per part.
    compressedData = False # idea we would send a compressed version of the stepsize.
    if compressedData:
        for i in range(len(UL_in_cluster_)):
            UL_in_cluster_[i] = PostCompressBlockMatrix(UL_in_cluster_[i], 9) # this would be send in quantized form. We would need ensure its spd.

    for i in range(len(UL_in_cluster_)):
        # Lc = L_in_cluster_[i]
        camera_indices_ = np.unique(camera_indices_in_cluster_[i])
        # mean_points[point_indices_,:] = mean_points[point_indices_,:] + points_3d_in_cluster_[i][point_indices_,:] * Lc
        # num_clusters[point_indices_] = num_clusters[point_indices_] + Lc
        # fill Vl with 0s 3x3 blocks? somehow ..
        # sparse matrix is 0,3,6, ... data (0s of Vl data), 012,012,012,345,345,345, etc ?
        #
        # data can remain, indptr can remain, indices must be adjusted / new
        # point_indices_
        # print(UL_in_cluster_[i].data.shape, " flip? ", camera_indices_.shape)
        # print(UL_in_cluster_[i].data[0:81].reshape(9,9), " datA")
        # print(UL_in_cluster_[i].indices[0:81].reshape(9,9), " indices")

        indices = np.repeat(
            np.array([9 * camera_indices_ + j for j in range(9)]).transpose(), 9, axis=0).flatten()
        # indices.append(np.array([3 * point_indices_ + j for j in range(3)]).transpose().flatten())
        # indptr is to be set to have empty lines by 0 3 3 -> no entries in row 3. 0:0-3, row 1:3-3

        indptr = [np.array([0])]
        j = 0
        for q in range(num_cameras):
            # print(q, " ", j, " ", point_indices_.shape[0], " ", np.array([9*j+3, 9*j+6, 9*j+9]) )
            if j < camera_indices_.shape[0] and camera_indices_[j] == q:
                indptr.append(np.array([81 * j +  9, 81 * j + 18, 81 * j + 27,
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
        UL_zeros_in_cluster_.append(U_pose)
        # print(mean_points.shape, " " , V_land.shape, points_3d_in_cluster_[i].shape)
        # print cost after/before.
        # cost old v is where ? (v-2u+s)^T V_land (v-2u+s) = v^T V_land v + 2 v^T V_land (-2u+s) + (2u-s)^T V_land (2u-s)
        # derivative 2 V_land v + 2 V_land (-2u+s) = 0 <-> sum (V_land) v = sum (V_land (2u-s))

        # print(i, "averaging 3d ", points_3d_in_cluster_[i][globalSingleLandmarksB_in_c[i], :]) # indeed 1 changed rest is constant
        # print(i, "averaging vl ", V_land.data.reshape(-1,9)[globalSingleLandmarksA_in_c[i],:])  # indeed diagonal

        prox_solution = True # does not matter
        if prox_solution:
            # TODO change 3, claim  2u+-s = 2 * (s+u)/2 - s  -  2 * (vli/2 .. ), so subtract u to get delta only
            u2_s = (2 * poses_in_cluster_[i].flatten() - poses_s_in_cluster_[i].flatten())
            #u2_s = (2 * points_3d_in_cluster_[i].flatten() - landmark_s_in_cluster_[i].flatten()) - (points_3d_in_cluster_[i].flatten() - delta_l_in_cluster[i].flatten())
            sum_D_u2_s += U_pose * u2_s # has 0's for those not present
        else: # assuming we do not solve the problem exactly
            nabla_p = np.zeros(num_cameras * 9)
            nabla_p[np.array([9 * camera_indices_ + j for j in range(9)]).transpose().flatten()] = nabla_p_in_cluster_[i]
            sum_D_u2_s += U_pose * poses_in_cluster_[i].flatten() - nabla_p
            # So nabla_p = U_pose * (u-s) , see bundle .. kind of.
            # Issue! need to send num cams data (matrix and pose) back == 6+9*9 floats, vs 3 floats per lm. no gain.
            # pose = 3 angle, 3 pos, 3 focal+distort. compress U_pose?
            # compared to 1/3 floats per cam. palm needs to send landmarks. 87 floats * cam vs 3 floats * lm, still factor 30.
            # lms are owned per node by drs. Also send more cams + lms in palm overlap.
            # DRS: send lms once back, once forth. Send poses, average, send back to compute cost, send cost, compare, resend maybe.
            #

        # print(i, "averaging u2_s ", u2_s.reshape(-1,3)[globalSingleLandmarksB_in_c[i], :]) # indeed 1 changed rest is constant

        #sum_constant_term += poses_in_cluster_[i].flatten().dot(U_pose * (poses_in_cluster_[i].flatten() + u2_s - poses_s_in_cluster_[i].flatten()))
        if i == 0:
            Up_all = U_pose
        else:
            Up_all += U_pose
    Upi_all = blockInverse(Up_all, 9)

    # rho_k/2 |u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k> is actually
    # rho_k/2 |u_k - v_k|^2 - <nabla_k, u_k - v_k>
    # and solution
    # sum_k rho_k (u_k - v_k) + nabla_k = 0 ->
    # v = sum_k (rho_k)^-1 * sum_k (rho_k u_k + nabla_k)

    # rho_k/2 |u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k>
    # deriv
    # sum_k rho_k v + rho_k (s_k - 2 u_k) = 0
    # v = (sum_k rho_k)^-1 (sum_k rho_k (2 u_k - s_k))
    pose_v_out = Upi_all * sum_D_u2_s
    verbose = False
    if verbose:
        cost_input  = 0.5 * (landmark_v_.flatten().dot(Vl_all * landmark_v_.flatten() - 2 * sum_D_u2_s) + sum_constant_term)
        cost_output = 0.5 * (landmark_v_out.dot(       Vl_all * landmark_v_out        - 2 * sum_D_u2_s) + sum_constant_term)

        #cost_simpler_out = landmark_v_out.dot(       Vl_all * landmark_v_out)        * 0.5 - landmark_v_out.dot(       sum_Ds_2u)
        #cost_simpler_in =  landmark_v_.flatten().dot(Vl_all * landmark_v_.flatten()) * 0.5 - landmark_v_.flatten().dot(sum_Ds_2u)
        print("========== update v: ", round(cost_input), " -> ", round(cost_output), " gain: ", round(cost_input - cost_output) )
        #print("======================== update v: ", round(cost_simpler_in), " -> ", round(cost_simpler_out), " gain: ", round(cost_simpler_in - cost_simpler_out) )

    return pose_v_out.reshape(num_cameras, 9), Up_all, UL_zeros_in_cluster_
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
#                     rho_k/2 {v^tv - 2vT[2uk-sk] + uk^T[3uk-2sk]}
def cost_DRE(
    #camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, pose_v
    camera_indices_in_cluster_,  poses_in_cluster_, poses_s_in_cluster_, L_in_cluster_, Ul_in_cluster_, pose_v_, nabla_p_in_cluster_
):
    num_cams =  poses_in_cluster_[0].shape[0]
    #sum_Ds_2u = np.zeros(num_cams * 9)
    #sum_constant_term = 0
    sum_u_s = 0
    sum_u_v = 0
    sum_u_v_ = 0
    sum_2u_s_v = 0
    cost_dre = 0
    dre_per_part = []
    penalty_per_cluster = []
    EV = []
    for i in range(len(Ul_in_cluster_)):
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
            (Ul_in_cluster_[i].data, indices, indptr),
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
        dre_per_part.append(round(local_cost.copy()))

        # if i == 0:
        #     Ul_all = U_pose
        # else:
        #     Ul_all += U_pose

    # analyis 646 small and large mixed.
    #     EV.append(blockEigenvalueSet(U_pose, 9))
    # EV.append(blockEigenvalueSet(Ul_all, 9))
    # print("-----------")
    # for c in range(num_cams):
    #     #evs = np.zeros(len(EV))
    #     for ci in range(len(EV)):
    #         print(EV[ci][:,c])
    #     print("-----------")

    # TODO: I use a different Vl to compute the cost here than in the update of prox u.
    #       Since I want to work with a new Vl already. Problem.
    # i want |u-s|_D |u-v|_D, also |v-2u-s|_D
    #cost_input  = 0.5 * (pose_v_.flatten().dot(Ul_all * pose_v_.flatten() - 2 * sum_Ds_2u) + sum_constant_term)
    print("---- |u-s|^2_D ", round(sum_u_s), "|u-v|^2_D ", round(sum_u_v), "|2u-s-v|^2_D ", round(sum_2u_s_v),
          "|u-v|^2 ", round(sum_u_v_), " cost_dre ", cost_dre, file=sys.stderr)
    print("---- dre_per_part --- ", dre_per_part, file=sys.stderr) # must be < 0.
    return cost_dre, dre_per_part
    # Why is f(v) supposed to be < f(u), check: since insert u for w into definition of envelope. argmin_w <s-u, u-w> + |u-w|_H^2, so argmin_w should be lower.
    # so envelope < f(u) should be true.
    # Here we see that local_cost should be < 0 <-> <u_k - v_k - 2s_k + 2u_k , u_k - v_k> < 0.
    #
    # 2. v in prox (2u-s) = prox(u - (s-u)), and optimality: 0 in nabla f(u) + rho(s-u), from u = argmin (f(uk) + <nabla f(uk), u-uk> + rho/2 |u-s|_2^2),
    # k-> inf (until convergence) -> nabla f(uk) + rho (u-s) = 0. -> s-u = nabla f(uk) / rho
    # v = prox(u - (s-u)) = prox(u  - nabla f(uk) / rho) = argmin_v sum_k  rho_k |v - (u - nabla f(uk) / rho_k)|^2
    #
    # def envelope in v is sum_k f_k ( u_k ) + <nabla f_k, vk-uk> + rho_k/2 |uk-sk|_2^2)
    # by Lf smoothness f_k(v_k) <= f_k ( u_k ) + <nabla f_k(uk), vk - uk> + Lfk/2 |uk-vk|^2, so
    # f_k(v_k) - Lfk/2 |uk-vk|^2 <= f_k ( u_k ) + <nabla f_k(uk), vk - uk> insert
    # <= f_k(v_k) - Lfk/2 |uk-vk|^2 + rho_k/2 |uk-sk|_2^2) = f_k(v_k) + (rho_k-L_fk)/2 |uk-sk|_2^2.
    # So, envelope in s >= cost in v + (rho_k-L_fk)/2 |uk-sk|_2^2 something positive if rho_k > L_fk.
    #
    # In other words, if we desire this, then we assume --
    # f_k(v_k) <= f_k ( u_k ) + <nabla f_k(uk), vk - uk> + Lfk/2 |uk-vk|^2, ok, but Lf not known
    # and we assume rho_k >= L_fk.
    # Can we use the above to compute/approx Lk?


# TODO: shorten
def primal_cost(
    poses_in_cluster_,
    camera_indices_in_cluster_,
    point_indices_in_cluster_,
    local_camera_indices_in_cluster_,
    local_landmark_indices_in_cluster_,
    points_2d_in_cluster_,
    points_3d_in_cluster_,
):
    cameras_indices_in_c__ = np.unique(camera_indices_in_cluster_)
    cameras_in_c = poses_in_cluster_[cameras_indices_in_c__]
    torch_points_2d_in_c = from_numpy(points_2d_in_cluster_)
    torch_points_2d_in_c.requires_grad_(False)

    unique_points_in_c_ = np.unique(point_indices_in_cluster_)
    # inverse_point_indices = -np.ones(np.max(unique_points_in_c_) + 1)  # all -1
    # for i in range(unique_points_in_c_.shape[0]):
    #     inverse_point_indices[unique_points_in_c_[i]] = i

    # point_indices_in_c = point_indices_in_cluster_.copy()
    # for i in range(point_indices_in_cluster_.shape[0]):
    #     point_indices_in_c[i] = inverse_point_indices[point_indices_in_c[i]]

    # camera_indices_ = np.zeros(camera_indices_in_cluster_.shape[0], dtype=int)
    # for i in range(cameras_indices_in_c_.shape[0]): # TODO: this might be slow if many cameras. make global, do once
    #     camera_indices_[camera_indices_in_cluster_ == cameras_indices_in_c_[i]] = i

    if False:
        x0_l_ = points_3d_in_cluster_[unique_points_in_c_].flatten()
        # holds all cameras, only use fraction, camera_indices_ can be adjusted - min index
        x0_p_ = cameras_in_c.flatten()
        x0__ = np.hstack((x0_p_, x0_l_))
        x0_t__ = from_numpy(x0__)
        n_cameras_ = int(x0_p_.shape[0] / 9)
        n_points_ = int(x0_l_.shape[0] / 3)
        x0_t_cam = x0_t__[: n_cameras_ * 9].reshape(n_cameras_, 9) # not needed?
        x0_t_land = x0_t__[n_cameras_ * 9 :].reshape(n_points_, 3)
        # funx0_st1 = lambda X0, X1, X2: \
        #     torchSingleResiduum(X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2))
    else:
        # simpler:
        x0_t_cam = from_numpy(cameras_in_c)
        x0_t_land = from_numpy(points_3d_in_cluster_[unique_points_in_c_])

    #camScale = 1./Unorm.data.reshape(-1,9)
    camScale = from_numpy(Unorm[cameras_indices_in_c__])
    camScale.requires_grad_(False)
    # print("camScale ", camScale.shape)
    # print("x0_t_cam ", x0_t_cam.shape)
    # print("cameras ", camScale * x0_t_cam)
    #landScale = 1./Vnorm.data.reshape(-1,3)
    landScale = from_numpy(Vnorm[unique_points_in_c_])
    landScale.requires_grad_(False)

    funx0_st1 = lambda X0, X1, X2: \
        torchSingleResiduumScaled(X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2), \
                                  camScale[local_camera_indices_in_cluster_[:]], \
                                  landScale[local_landmark_indices_in_cluster_[:]])

    fx1 = funx0_st1(
        x0_t_cam[local_camera_indices_in_cluster_[:]],
        x0_t_land[local_landmark_indices_in_cluster_[:]],
        torch_points_2d_in_c)
    costEnd = np.sum(fx1.numpy() ** 2) / (F_SCALE* F_SCALE)
    return costEnd

# there are cams with < 5 -- even 1 landmark only.
# must invert 9x9 in ok manner. those also constrain the cam vectors to lie at s.
# what if we constrain it to lie in BS place? k1,k2,f>0 e.g.
#
# TODO maybe need to split prox and tr terms? So prox always JtJ and some other part is
# to make underconstrained VLi work at to diag to invert? pseudo inverse?
# cams are not full rank. swap cams? fill based on #cams present --
# recall clustering 1. cams disjoint, 2. add res to make complete landmarks
# TODO: what does not work:
# trust region binds to s-u = delta but we eval f at u.
# then TR binds closer to f(u-s), which can be very bad.
# we get jac at dist s-u, so |nabla f(u) - nabla f(s)|^2 <= L/2 |s-u|^2
# upper bound f(s) <= f(u) + <nabla f(u), s-u> + L/2|s-u|^2. in other words f(s) should be accepted, delta_u = s-u.
# in particular L > 2 nabla f(u) on s-u. nabla f(u) = J^t f(u).
# So disable TR? and understand why my J fails?
def bundle_adjust(
    point_indices_,
    camera_indices_,
    poses_only_in_cluster_, # those cameras should be excluded from prox part. But those cams do not exist.
    torch_points_2d,
    points_3d_in,
    cameras_in,
    cameras_s_, # taylor expand at point_3d_in -> prox on landmark_s_ - points_3d = lambda (multiplier)
    Ul_in_c_,
    L_in_cluster_,
    LipJ_, # start with 1.0. externally increase if dre increases
    blockEig_in_c_,
    unique_poses_in_c_, # global indices, needed for pcg
    unique_landmarks_in_c_,
    cluster_id,
    successfull_its_ = 1,
):
    successfull_its_ = 1 # indeed works well.
    LipJ_ = 1.005 # less jumping never better. maybe best to inc this when failing ?! not really. some fail very early for no reason?
    blockEigMult = 1e-5 # 1e-3 was used before, too high low precision.
    # 1e-8 fluctuates but faster 1e-6. increase JJ_mult?
    # problem dies at 173 example. 1e-5 ok more not.
    #J_eps = 1e-4
    minimumL = 1e-6 #1e-6 # 1e-6 # 1e-8 also ok, maybe 1e-5
    #minDiag = 1e-5
    L = max(minimumL, L_in_cluster_)
    JJ_mult = 4 # TODO 4 / 2. 4 should suffice everywhere?
    updateJacobian = True
    # holds all! landmarks, only use fraction likely no matter not present in cams anyway.
    x0_l_ = points_3d_in.flatten()
    # holds all cameras, only use fraction, camera_indices_ can be adjusted - min index
    x0_p_ = cameras_in.flatten()
    x0_ = np.hstack((x0_p_, x0_l_))
    x0_t_ = from_numpy(x0_)
    s_p_ = cameras_s_.flatten()
    # torch_points_2d = from_numpy(points_2d)
    n_cameras_ = int(x0_p_.shape[0] / 9)
    n_points_ = int(x0_l_.shape[0] / 3)
    powerits = 100 # kind of any value works here? > =5?
    tr_eta_1 = 0.8
    tr_eta_2 = 0.25
    blockEigMultGain = 4 # 4 better than 2 at least if allowDecreaseBlockEig, feels random and weird
    threshWhereNeeded = 1e-6
    verbose_Jac = False # faster if False, True only debug

    newVersion = True
    jointVersion = False
    # TODO: This parameter block is ok blockEigMultJtJ 1e-5, LipJ = 2, blockEigenvalueWhereNeeded 1e-2,
    # might be slightly better than blockEigMultJtJ 1e-4 ? / use LipJ = 2 * np.ones(kClusters) appears safe.
    if newVersion:
        # JJ_mult = 2 less flipping much slower in ladybug646,
        JJ_mult = 1 #+ L # YES! .. should i add tr thing instead, no? use first in computing deriv 2nd time (simple test to lower this here.)
        blockEigMult = 1e-5 # 1e-7 fails with venice'52' 173 demands 1e-3/1e-4/1e-5? 52:1e-6 totally fails. Maybe also True below (recomp jacobian): yes stable
        # TODO: set to 1 and play with Limit. Set higher. is 1e-2 same 1e-1? is 1e-3 worse?
        threshWhereNeeded = 1e-4 # this higher -> blockEigMult, blockEigMultLimit lower?
        blockEigMultJtJ = 1e-4 # 173: little effect 1e-6/4/8. just 173 or always not mattering much?
        blockEigMultLimit = 1e-5 # TODO: 8: no, 6: maybe, 4: no (52 fails)
        #globalBlockEigUpperLimit = 1e-1 # 1e-2? # same as globalBlockEigUpperLimit
        decent_lemma_divisor = 2 # 2/4: higher does indeed delay flow over, but result is worse.
        Derivative_at_end = False # maybe negative to have update v and updte u differ.
        # limit lower -> major effect from partitioning only?
        # 1e-5 even better than 1e-7.
        # 1. fix rng, 2. is there structure, how identify if good or bad part?
        # blockEigMult not import for 173 but 52 yes
        # adapt blockEigMult based on check? pass up and down hierarchy?
        # problem 1e-3/4/5 good for 173, not for 52. 52: better for 1e-6 bad for 1e-5 etc.
        blockEigMult = blockEig_in_c_
        print("blockEig_in_c_ ", blockEig_in_c_, file=sys.stderr)

    use_be_memory = True
    if use_be_memory and len(tempBlockEigen[cluster_id]) > 1:
        if len(tempBlockEigen[cluster_id]) > globalIt % memory_be:
            tempBlockEigen[cluster_id][globalIt % memory_be] = 0 # remove current, either inner iteration or beyond memory
        print(globalIt, " cluster_id ", cluster_id, " 1000 * tmp ", 1000 * np.array(tempBlockEigen[cluster_id]),\
              " maxbe ", np.max(np.array(tempBlockEigen[cluster_id]), axis=0), " globalIt % memory_be ", globalIt % memory_be, file=sys.stderr)
        #blockEigMult = np.maximum(blockEigMult, np.max(tmp, axis=0)) # else always larger / pointless same as base version
        blockEigMult = np.max(np.array(tempBlockEigen[cluster_id]), axis=0) # correct but 52 much worse ?Z

    it_ = 0
    funx0_st1 = lambda X0, X1, X2: \
        torchSingleResiduum(X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2))

    #camScale = 1./Unorm.data.reshape(-1,9)
    camScale = Unorm[unique_poses_in_c_] # 1st, problem
    camScale = from_numpy(camScale[camera_indices_[:]]) # 2nd
    camScale.requires_grad_(False)

    #landScale = 1./Vnorm.data.reshape(-1,3)
    landScale = Vnorm[unique_landmarks_in_c_]
    landScale = from_numpy(landScale[point_indices_[:]]) # here direct?
    landScale.requires_grad_(False)

    funx0_st1 = lambda X0, X1, X2: \
        torchSingleResiduumScaled(X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2), camScale, landScale)

    # if issparse(Ul_in_c_): # only increase -- if needed.
    #     stepSize = diag_sparse(Ul_in_c_.diagonal())

    steSizeTouched = False
    while it_ < successfull_its_:

        if updateJacobian:  # not needed if rejected
            x0_t_cam  = x0_t_[: n_cameras_ * 9].reshape(n_cameras_, 9)
            x0_t_land = x0_t_[n_cameras_ * 9 :].reshape(n_points_, 3)
            #start = time.time()

            J_pose, J_land, fx0 = ComputeDerivativeMatricesNew (
                x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d, camScale, landScale)
            #print("Jac time ", time.time() - start )

            # 2 * JtJ majorizes, note JtJ:=(UW|W^TV), so W part majorized by *2:
            # clearly: 2a^2+b^2 > (a+b)^2 = a^2 + b^2 + 2ab. Since (a-b)^2 = a^2 + b^2 - 2ab > 0, so a^2 + b^2 > 2ab.
            # a^2 = p^t* Jp^TJp * p , b^2 = l^tJl^TJl l. ab = p^tJp^T Jl*l.
            #(Jl | Jp) (l,p)^T = Jl l + Jp p and |(Jl | Jp) (l,p)^T|^2 = l^t Jl^t Jl l + p^t Jp^t Jp p + 2 p^t Jp^t Jl l.
            # So 2 JtJ  + 2 JltJl shuold majorize |J^t x|^2 for all x.

            JltJl = J_land.transpose() * J_land
            if False:
                # JltJlDiag = JltJl + J_eps * diag_sparse(np.ones(JltJl.shape[0]))
                #blockEigenvalueJltJl = blockEigenvalueWhereNeeded(JltJl, 3) # nope not at all.
                blockEigenvalueJltJl = 1e-3 * blockEigenvalue(JltJl, 3)
                JltJlDiag = JltJl + 1e-6 * blockEigenvalueJltJl
            else:
                # could do only where needed? smallest ev is indeed small?
                # JltJlDiag = JltJl + blockEigenvalue(JltJl, 3) # with normalization better? lower here, higher on JtJ ? or larger even?
                # TODO: blockEigenvalue likely expensive here. avoid?
                # JltJlDiag = JltJl + 2.5 * diag_sparse(np.fmax(JltJl.diagonal(), 1e-6)) #, worse for 52, 3068

                JltJlDiag = JltJl + maxDiagA(JltJl, 3) # ok.
                # JltJlDiag = JltJl + maxDiag(JltJl, 3) # hmm worse? Why? very unclear.
                # JltJlDiag = JltJl + maxRow(JltJl, 3) # ok.

                # experiments with this, as it does not cancel easily in A^TA and looks wrong.
                #JltJlDiag = JltJl # ? singular matrix ugh
                # JltJlDiag = maxDiag(JltJl, 3) # nope

            if verbose_Jac:
                absDiagJltJl = np.abs(JltJl.diagonal()).reshape(-1,3)
                print( "Diag pseudo HessL (max/min/med)",  np.max(absDiagJltJl, axis=0), " ", np.min(absDiagJltJl, axis=0),  " ", np.median(absDiagJltJl, axis=0), file=sys.stderr )
                maxE, minE = minmaxEv(JltJl, 3)
                #print("minmax ev JltJl ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spec ", np.max(maxE/minE) )
                JltJlSpec = [round_int(np.max(maxE/minE)), np.min(maxE/minE), round_int(np.median(maxE/minE))]
                maxE, minE = minmaxEv(JltJlDiag, 3)
                JltJlDiagSpec = [round_int(np.max(maxE/minE)), np.min(maxE/minE), round_int(np.median(maxE/minE))]
                print("minmax ev JltJlD ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spec ", JltJlSpec, " -> ", JltJlDiagSpec, file=sys.stderr )

            prox_rhs = x0_p_ - s_p_
            costStart = np.sum(fx0**2) #/ (F_SCALE* F_SCALE)
            W = J_pose.transpose() * J_land
            bp = J_pose.transpose() * fx0
            bl = J_land.transpose() * fx0
            JtJ = J_pose.transpose() * J_pose

            if verbose_Jac:
                # TODO: cam hessian scaled awfully. degenerate.
                maxE, minE = minmaxEv(JtJ, 9)
                JtJSpec = [round_int(np.max(maxE/minE)), np.min(maxE/minE), round_int(np.median(maxE/minE))]
                print("minmax ev JtJ ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spec ", JtJSpec, file=sys.stderr )

                print( "Mean diagonal of pseudo Hessian ",  np.sum(np.abs(JtJ.diagonal()).reshape(-1,9) / (1000 * n_cameras_), 0), file=sys.stderr )
                absDiagJtJ = np.abs(JtJ.diagonal()).reshape(-1,9)
                print( "Diag pseudo HessP (max/min/med)",  np.max(absDiagJtJ, axis=0), " ", np.min(absDiagJtJ, axis=0),  " ", np.median(absDiagJtJ, axis=0), file=sys.stderr )

            if newVersion:
                # traditional ADMM: not good -- also return
                # blockEigenvalueJtJ = blockEigenvalue(JtJ, 9) + 1e-15 * JtJ
                # stepSize = blockEigenvalueJtJ #

                # default 1e-4. 1e-6 for 173: slightly worse| 1e-6 for 52: also worse now (maybe was the other way round).
                #               1e-3 for 173: | 1e-3 for 52: ok 1e-3 or 1e-4? might be highly random based on clustering.
                # 1e-3 appears a bit better.
                #blockEigenvalueJtJ = blockEigenvalueWhereNeeded(JtJ, 9, threshWhereNeeded) # ! 1e-2, 1e-4 1e-5 als works. problem 52, 1e-6 does not 173 performance bad if not 1e-6?
                #blockEigenvalueJtJ = 1e-1 * blockEigenvalue(JtJ, 9) # ! try 173 & 52, fails at 1e-2, 10 clusters. 1e-1 ok for 173 & 52. (not dead)

                if False:
                    # TODO: check this out?
                    blockEigenvalueJtJ = 1e2 * threshWhereNeeded * blockEigenvalue(JtJ, 9) # 173 to set.

                    # TODO: LipJ for both? or only JJ?
                    #stepSize = JJ_mult * JtJ.copy() + blockEigMult * blockEigenvalueJtJ
                    stepSize = LipJ_ * JtJ.copy() + blockEigMult * blockEigenvalueJtJ
                    JtJDiag = JtJ.copy() + blockEigMultJtJ * blockEigenvalueJtJ
                    #JtJDiag = blockEigMultJtJ * blockEigenvalueJtJ # this is likely almost same as above. Todo: check/find value.
                else:
                    # paper: why is this needed? since nearby hess are different especially for small eigen values -> add max ev.
                    # blockEigenvalueJtJ = 1e1 * blockEigenvalue(JtJ, 9) # a bit better, maybe random.
                    blockEigenvalueJtJ = 1e1 * maxDiagA(JtJ, 9) # almost ..? maybe just random
                    # blockEigenvalueJtJ = 1e1 * maxRow(JtJ, 9) # ? does it matter?

                    CompressHere = False
                    if CompressHere:
                        preCompress = True
                        if not preCompress:
                            stepSize = LipJ_ * JtJ.copy() + blockEigMult * blockEigenvalueJtJ # 12 already does not jump, but some results are not good: 52
                            stepSize = CompressBlockMatrix(stepSize, 9) # can i bring + blockEigMult * blockEigenvalueJtJ inside compress?
                            # blockEigenvalueJtJ = 1e1 * maxDiagA(stepSize, 9) # Trust region cannot catch up, unclear. this here is same?
                        else: # compress first adjust later? Issue is stll negative definite mat?
                            stepSize = PreCompressBlockMatrix( JtJ.copy(), 9 ) # add max( 0, neg min ev) on diagonal? + blockEigMult * maxDiag(JtJ, 9)
                            blockEigenvalueJtJ = 1e1 * maxDiagA(stepSize, 9)
                            stepSize = LipJ_ * stepSize + blockEigMult * blockEigenvalueJtJ
                    else:
                        stepSize = LipJ_ * 1e-1 * JtJ.copy() + blockEigMult * blockEigenvalueJtJ # 12 already does not jump, but some results are not good: 52
    
                stepSize = mult_selected_blocks(stepSize, poses_only_in_cluster_, 1e-6, 9) # those do not exist -- only if 1 cluster.

                # best? or 8 for my single .. above is producing less jumps.
                # try as new simple step 16 or .. ?
                # blockEigenvalueJtJ = 1e1 * maxDiagA(JtJ, 9) #maxRow(JtJ, 9) # ? does it matter?
                # stepSize = 16 * blockEigMult * blockEigenvalueJtJ + 1e-16 * JtJ.copy() # ? * 2 appear better. larger rather not.
                # stepSize = 1e-0 * (LipJ_ * JtJ.copy() + blockEigMult * blockEigenvalueJtJ) # 1 cluster. why not above? will it jump?
                JtJDiag = JtJ.copy() + blockEigMultJtJ * blockEigenvalueJtJ # new 1e-2 * same as for  JltJlDiag

                # Todo: eval. this might be very similar but simpler.
                # JtJDiag = blockEigMultJtJ * 5e1 * maxDiag(JtJ, 9) # why adding jtj here? 3068
                # JtJDiag = 1e-1 * maxDiag(JtJ, 9) # why adding jtj here? 3068

                # how does diag value change over iterations? mean/max of last k iterations?
                # stable? but i change a tiny bit only to get above argh.
                # if cluster_id == 0:
                #     print("1e1 * JtJ.diagonal() ", 1e1 * JtJ.diagonal().reshape(-1,9)[0:100,:])#, file=sys.stderr)
                #     print("blockEigenvalueJtJ.diagonal() ", blockEigenvalueJtJ.diagonal().reshape(-1,9)[0:100,:])#, file=sys.stderr)
                #     print("stepSize.diagonal() ", stepSize.diagonal().reshape(-1,9)[0:100,:])#, file=sys.stderr)

                # maxE, minE = minmaxEv(JtJ, 9)
                # print("JtJ spectral ", (maxE/minE))
                # maxE, minE = minmaxEv(stepSize, 9)
                # print("stepSize spectral ", (maxE/minE))


                if False: # play better use annealing? It really sucks.
                    # SBlock in -127 to 127. scale * SBlock = stepSize.
                    SBlock, scales, SBlockRound = LowCompressBlockMatrix(stepSize, 9, 127)
                    # Best Fit adding 1 more bit to above.
                    # Fitting to be done PER 9x9 block.
                    id = 23
                    ss = stepSize.data[81 * id : 81 * id + 81].reshape(9, 9)
                    ss_ = SBlock.data[81 * id : 81 * id + 81].reshape(9, 9)
                    ssR_ = SBlockRound.data[81 * id : 81 * id + 81].reshape(9, 9)
                    print(ss, "\n s", scales[id], "\n ss ", ss_)
                    M = ss * (1./scales[id]) - ss_ # can add 1 to stepsize -- maybe more to diag. Idea is to best approx JtJ.
                    M = np.matrix(M)
                    cR = np.matrix(ssR_ - ss_ ) # rounding solution.
                    print("M ", M)

                    # Here  W^ = SBlock and W:= stepSize / scale.
                    # From sum_x | W X - (W^x + V x) |_F^2 + R(V), V in {0,1} -- can add diag later.
                    # x: examples, here cam parameters before optimization
                    # M:= W - W^, set X = x1|x2|..|xn->  1/2m | M x - V X|^2_F + R(V) = 1/2n |X^T M^T - X^T V^T|^2_F + R(V)
                    # symmetry V: = 1/2n | X^T M - X^T V |^2_F + R(V).
                    # Derivative is 1/n X (X^T M - X^T V) -> G:= X X^T := 1/n sum_i x_i^T x_i in R^9x9
                    # with derivative G V + G M, since V symmetric, add symmetric part of derivatives.
                    # Regularizer is r(x):= k|x|_1 + l|x|^2 per component.
                    # proxmap is solvable in closed form
                    # prox_r(k,l) [x] := 1/(1+l) * { x - k sign(x) if x<=1/2, x - k sign(x-1)+ l, x > 1/2}
                    # x^k := argmin_x 1/2 tk | x - (x^k-1 - tk nabla f(x^k-1))  |^2 + R(x)
                    # tk: 1/L, L Lipshitz of f.
                    # f: = 1/2n |X^T M - X^T V|^2_F = 1/2n [X^T (M - V)]^T [X^T (M - V)] = 1/2n (M - V)^T G (M - V) = 1/2n V^T G V - V G M - M G V + M G M  .. -> 1/n maxEV(G) is our L.
                    # note deriv of M*V by vij is column i of M := Mi at column index j and deriv of V M by vij is ith row of M := Mi at row j.
                    # Since we sum entries of 1/2n {V^T G V - V G M - M G V + M G M} the palcement does not matter, we sum the respective row/column.
                    # d/dij V^T G V = sum of entries in V_i and V_j, and from V G M + M G V we get sum of entries in M_i and M_j.
                    # -> Conpute G hadamard (V+M), then sum rows and columns, then repeat along col/row to get original size. this is the per entry derivative.
                    #
                    # 1. compute G := X^T X: could also be just 1 example for each entry / pose, currently i use all poses for each fit / stepSize.

                    G = np.matrix(x0_t_cam[id,:]) / x0_t_cam.shape[0] # single example.
                    #G = np.matrix(x0_t_cam[]) / x0_t_cam.shape[0] # all examples

                    G = G.transpose().dot(G)
                    #evs = eigvalsh(G)
                    L = 1 #000
                    # G = G / np.max(evs) * L * 0.9 # L = 1000
                    # G = G / np.max(np.abs(G)) * L * 0.1 # Hadamard means max entry is Lipshitz.
                    # GScale = np.max(np.sum(np.abs(G), axis=0)) # symmetric
                    GScale = np.sum(np.abs(G)) # symmetric
                    G = G / GScale * L * 1 # Hadamard means max entry is Lipshitz. # guess 0.499 is ok.

                    # evs = eigvalsh(G)
                    #print("eigenvals: ", evs)
                    #print(G.shape)
                    print(G)

                    # X^0 = 0
                    x = 0.5 #* (M +0.5) #* (cR + 0.5) #0.5 * np.ones([9,9]) # INIT or M? or M
                    ka = 0.001 #0.5 # regularizer
                    la = 0.02 # 0.5 # regularizer, increase with time. This is the rate with which we move, i.e. la = 0.1 0.5, 0.4, etc.
                    # x^k := argmin_x 1/2 tk | x - (x^k-1 - tk nabla f(x^k-1))  |^2 + R(x)

                    c = np.matrix(x) #np.round(x)
                    #initCost = c * G * c - c * G * M - c * M * G + M * G * M # this is not correct still can be negative.
                    #initCost = np.multiply(G, c * c) - np.multiply(G, c * M) - np.multiply(c * M, G ) + np.multiply(G,M * M) # this is not correct still can be negative.
                    initCost2 = np.multiply(G, (c-M).transpose().dot(c-M)) * scales[id] * 10000000
                    initCostR = np.multiply(G, (cR-M).transpose().dot(cR-M)) * scales[id] * 10000000
                    print(0, ". x ", x, " cost ", np.sum( initCostR ), " cost2 ", np.sum( initCost2 )) # regularizer not printed.

                    for it_ in range(100):
                        # deriv = 1/n X (X^T V - X^T M) = G V - G M
                        # d = -np.multiply(G, (M - x)) # 1 hadamard producr.

                        # d = np.multiply(G, (x - M))
                        # print(" d-norm ", np.sum(np.abs(d)))
                        # rows = np.sum(d, axis=0)
                        # cols = np.sum(d, axis=1)
                        # d = np.repeat(rows, 9, axis=0) + np.repeat(cols, 9, axis=1)

                        d = np.zeros([9,9])
                        Z = (x - M)
                        #print("Z", Z)
                        for k in range(9):
                            Zk = np.repeat(Z[:,k], 9, axis=1) # kth column
                            #print("Zk", Zk)
                            Gzk = np.sum(np.multiply(G, Zk), axis = 0)
                            #print("Gzk", Gzk)
                            d[k,:] = Gzk
                            #print("d k", d)
                        d = 0.5 * (d + d.transpose()) # solution must be symmetric ? Actually not if I send 45 bits more (+1).
                        print(" d-norm ", np.sum(np.abs(d)))

                        y = x - 1/L * d
                        # solve proxmap with regularizer.
                        # prox_r(k,l) [x] := 1/(1+l) * { x - k sign(x) if x<=1/2, x - k sign(x-1)+ l, x > 1/2}
                        select = np.where(y<=1./2., 0, 1)
                        #print("y ", y)
                        #print("s ", select)
                        t0 = 1./(1. + 2. * la) * (y - ka * np.sign(y)) # 0.5 -> 1/2 * [ 0.5 - 1 ] - 0.25?
                        t1 = 1./(1. + 2. * la) * (y - ka * np.sign(y-1) + 2. * la) # 0.5 -> 1/2 * [0.5 + 1 + 1] = 1.25
                        #print("t0/t1 ", t0, " \n", t1)
                        x = np.multiply(np.ones([9,9]) - select, t0) + np.multiply(select, t1)
                        # x = np.fmax(0,x) # never good?
                        # x = y # to ignoere regularizer. should converge faster.
                        c = np.matrix(x) #
                        c = np.round(c)
                        # cost1 = c * G * c - c * G * M - c * M * G + M * G * M # this is not correct still can be negative.
                        #cost2 = G * (M-c).transpose().dot(M-c)
                        cost2 = np.multiply(G, (c-M).transpose().dot(c-M)) * scales[id] * 10000000
                        # cost = np.multiply(G, x * x - x * M - M * x + M * M)
                        # simpler cost (derivative) is this (see above deriv is G)
                        # cost = np.multiply(G, x - M) # DOES NOT WORK SINCE GOES TO 0 of course.
                        print(it_+1, ". x ", x, " cost round ", np.sum(initCostR), " cost2 ", np.sum(cost2)) # regularizer not printed.
                        la = la * 1.05 # slower is better .. maybe start slower raise stronger?

                    # maxdiag * 10 * blockEigMult is added.
                    # let s = maxdiag then 1e-4 * s is added. quantization means s/127 is 1 unit we would observe.
                    # so mult of be by 2/4 leads to effect? rarely effect only low likelihood that round goes beyond 1 unit.
                    # BUT we can do this AFTER quant. we know be in part & globally.
                    # so we compress JtJ, then mult by LipJ and add be * diag.

                penaltyStartConst = prox_rhs.dot(stepSize * prox_rhs)
                if jointVersion:
                    penaltyStartConst += L * prox_rhs.dot(JtJDiag * prox_rhs)

                if verbose_Jac:
                    maxE, minE = minmaxEv(stepSize, 9)
                    StepSizeSpec = [round_int(np.max(maxE/minE)), np.min(maxE/minE), round_int(np.median(maxE/minE))]
                    print("minmax ev stepSz ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spec ", StepSizeSpec, file=sys.stderr)
                    #maxE, minE = minmaxEv(JtJDiag, 9)
                    #print("minmax ev JtJDiag ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spec ", np.max(maxE/minE))
            else: # not newversion
                #blockEigenvalueJtJ = blockEigenvalue(JtJ, 9) # TODO: what if this is only needed for 0-eigen directions? return !=0 only if in small eigendir
                blockEigenvalueJtJ = blockEigenvalueWhereNeeded(JtJ, 9, threshWhereNeeded) # here ok? 173: 1e-6
                stepSize = blockEigMult * blockEigenvalueJtJ + JJ_mult * JtJ.copy() # Todo '2 *' vs 1 by convex.

                #blockEigenvalueJtJ = blockEigenvalueFull(JtJ, 9, x0_t_cam)
                #stepSize = blockEigenvalueJtJ + JJ_mult * JtJ.copy() # Todo '2 *' vs 1 by convex.

                JtJDiag = stepSize.copy() # max 1, 1/L, line-search dre fails -> increase
                JtJDiag = 1/L * JtJDiag # max 1, 1/L, line-search dre fails -> increase
                penaltyStartConst = prox_rhs.dot(JtJDiag * prox_rhs)

                # TODO: here, also write min/max Eigenvec and spec to debug
                # blockEigenvalueJtJ = blockEigenvalueFull(JtJ, 9) # print eval/vec structure
                # stepSize = blockEigenvalueJtJ + JJ_mult * JtJ.copy() # Todo '2 *' vs 1 by convex.

                # try this. maybe eigenvals very far apart?
                #stepSize = JJ_mult * JtJ.copy() + minDiag * diag_sparse(np.fmax(JtJ.diagonal(), 1e-4))

                # stepSize = 1. * (blockEigMult * blockEigenvalueJtJ + 1.4 * JtJ.copy()) # Todo '2 *' vs 1 by convex.
                #stepSize = 1. * (1e-1 * diag_sparse(np.ones(JtJ.shape[0])) + 1.1 * JtJ.copy()) # not at all working
                # both of these are faster (accelerated only? or anyways?)
                # todo: maybe adjust factor on JtJ instead? or check extrapolation of s wrt. cost / penalty.
                # faster for normal, non accelerated runs
                #stepSize = diag_sparse(np.fmax(blockEigMult * JtJ.diagonal(), 1e-2)) + 1.1 * JtJ.copy() # stable 27.9 non-acc. with unstable but faster.
                #stepSize = diag_sparse(np.fmax(blockEigMult * JtJ.diagonal(), 1e-1)) + 2.0 * JtJ.copy()# stable 28.1 non-acc. with unstable but faster.
                # this is what dre test is for, no? maybe cannot compare if we alter RELATIVE weight of step size.

                # if not issparse(Vl_in_c_) and it_ < 1:
                #     stepSize = blockEigenvalueJltJl
                # else: # increase where needed -- this here is WAY too slow?
                #     stepSize.data = np.maximum(0.05 * stepSize.data, blockEigenvalueJltJl.data) # else diagSparse of it

                # shoudl not depend on eigenvalue of block. the small ones should be increased, since we invert the matrix.
                #stepSize = LipJ * JtJ.copy() + J_eps2 * diag_sparse(np.ones(JtJ.shape[0])) # ?
                #stepSize = LipJ * JtJ.copy() + diag_sparse(np.fmax(JtJ.diagonal(), 1e-4))

                #maxE, minE = minmaxEv(stepSize, 9)
                #print("minmax ev stepSz ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spec ", np.max(maxE/minE) )

            #print("Full Jac time ", time.time() - start )

        # TODO: solve the whole! thing with cholesky and compare. maybe this is better.
        # Advantage DRS in parts: can be parallelized, no memory issues. Disadvantage: not as good as a whole -- maybe.

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
        #etst = W * Vli * W.transpose() # Ul - W * Vli * W.transpose() # the matrix is DENSE? look at 1st 468 entries:
        #print(etst.shape, " row 1: ", etst.data[etst.indptr[0] : etst.indptr[1]])
        #print(etst.shape, " ", np.max(etst.diagonal()), " ", etst.data.shape, " ", etst.indices[0:100], " ", etst.indptr[0:100])
        # We wanted Ul - W * Vli * W.transpose()), yet this is dense.
        # Ul - W * Vli * W.transpose()) = Ul * ( I - Uli * W * Vli * W.transpose()),
        # where eigen value of ( I - Uli * W * Vli * W.transpose()) is <1 but > 0.

        # Todo: check this out, slightly more than the REAL Hessian of our approximation.
        # Since we set it here should work? LipJ_ = 1.005: too small
        # before:
        # stepSize = LipJ_ * JtJ.copy() + blockEigMult * blockEigenvalueJtJ
        # JtJDiag = JtJ.copy() + blockEigMultJtJ * blockEigenvalueJtJ # new 1e-2 * same as for  JltJlDiag

        # stepSize = 1.01 * JtJ + np.minimum(0.1, L) * blockEigMultJtJ * blockEigenvalueJtJ # unstable at 1064 compared to current one althoug almost the same.

        #stepSize = 1 * (1.5 - 1. / (1. + L)**2) * JtJ.copy() + blockEigMult * blockEigenvalueJtJ

        #stepSize = (2. - 1. / (1. + L)**2) * JtJ.copy() + blockEigMult * blockEigenvalueJtJ
        # stepSize = 2.0 * JtJ.copy() + blockEigMult * blockEigenvalueJtJ # how low can we go .. see above breaking point.

        # stepSize = 2 * (1.2 - 1. / (1. + L)**2) * JtJ.copy() + blockEigMult * blockEigenvalueJtJ
        # 1.05, 1.1 adjust ..? at least 0.9, 0.8 best?
        # stepSize = 2 * (1.0 - np.minimum( 0.75, 1. / (1. + L)**2)) * JtJ.copy() + blockEigMult * blockEigenvalueJtJ

        penaltyStartConst = prox_rhs.dot(stepSize * prox_rhs)

        if newVersion:
            Ul = JtJ + L * JtJDiag + stepSize
            penaltyStart = penaltyStartConst

        bp_s = bp + L * JtJDiag * prox_rhs # TODO: + or -. '+', see above
        if newVersion:
            bp_s = bp + stepSize * prox_rhs
        if jointVersion:
            bp_s = bp + (L * JtJDiag + stepSize) * prox_rhs # AAA

        bS = (bp_s - W * Vli * bl).flatten() # see XX equals 2 * (bp - W * Vli * bl)
        # look in power its paper.

        # Lesson: |f(x0) + Jp^t dp + Jl^t dl|^2 = |f(x0) + Jp^t dp - Jl^t Vli * (W^T * dp) - Jl^t Vli * Jl f(x0))) |^2
        # with W^T = Jl^T Jp, A: = (I - Jl Vli Jt^T)
        # = |A f(x0) + A Jp^t dp|^2, hence
        # A = A^T A, if Vli = (Jl^T Jl)^-1 since
        # (I - Jl Vli Jl^T) (I - Jl Vli Jl^T) = I - 2 Jl Vli Jl^T - Jl Vli Jl^T + Jl Vli Jl^T Jl Vli Jl^T = I - Jl Vli Jl^T
        # Then quadratic part is actually == Jp^T * Jp - W * VL^-1 * W^T,
        # linear part is == 2 f(x0)^T A Jp^t dp = 2 f(x0)^T Jp^t dp - 2 (f(x0)^T Jl Vli W)^T dp (XX)
        #
        # possibly Ul replaces Jp^T * Jp, see above.
        # we add stepSize * prox_rhs and Ul + stepSize to incorporate the prox term.
        # How does this interact with how we solve the equation system.
        # 1. Replace stepsize using Jp^T Jp - W * VL^-1 * W^T as basis?
        # increasing L -> stepsize ==

        #delta_p = -solvePowerIts(Ul, W, Vli, bS, powerits)
        delta_p, powerits_run = solveByGDNesterov(Ul, W, Vli, bS, powerits)
        delta_p = -delta_p
        delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl)

        penaltyL = L * (delta_l).dot(JltJlDiag * delta_l)
        penaltyP = L * (delta_p + prox_rhs).dot(JtJDiag * (delta_p + prox_rhs))
        if newVersion:
            penaltyP = L * delta_p.dot(JtJDiag * delta_p) + (delta_p + prox_rhs).dot(stepSize * (delta_p + prox_rhs))

        if jointVersion:
            penaltyP = (delta_p + prox_rhs).dot((L * JtJDiag + stepSize) * (delta_p + prox_rhs)) # AAA

        # end_ = time.time()
        # print("Lm step took ", end - start, "s")

        fx0_new = fx0 + (J_pose * delta_p + J_land * delta_l)
        costQuad = np.sum(fx0_new**2) #/ (F_SCALE* F_SCALE)
        print(it_, "it. cost 0     ", round(costStart)," cost + penalty ", round(costStart + penaltyStart), " === using L = ", L, file=sys.stderr)
        print(it_, "it. cost 0/new ", round(costQuad), " cost + penalty ", round(costQuad + penaltyL + penaltyP), " Pits ", powerits_run, file=sys.stderr)

        # update and compute cost
        x0_p_ = x0_p_ + delta_p
        x0_l_ = x0_l_ + delta_l

        x0_ = np.hstack((x0_p_, x0_l_))
        x0_t_ = from_numpy(x0_)
        x0_t_cam = x0_t_[: n_cameras_ * 9].reshape(n_cameras_, 9)
        x0_t_land = x0_t_[n_cameras_ * 9 :].reshape(n_points_, 3)

        fx1 = funx0_st1(
            x0_t_cam[camera_indices_[:]],
            x0_t_land[point_indices_[:]],
            torch_points_2d)
        costEnd = np.sum(fx1.numpy() ** 2) #/ (F_SCALE* F_SCALE)
        print(it_, "it. cost 1     ", round(costEnd), "      + penalty ", round(costEnd + penaltyL + penaltyP), file=sys.stderr,)

        # v1: also must adjust below line 2013, same.
        # tr_check = (costStart + penaltyStart - costEnd - penaltyP - penaltyL) / (costStart + penaltyStart - costQuad - penaltyP - penaltyL)
        # v3: also must adjust below line 2013, same.
        # tr_check = (costStart - costEnd) / np.maximum(0.1, costStart - costQuad) # much worse for 1266, 3068, rest similar.
        # v2: also must adjust below line 2013, same.
        if not jointVersion:
            penaltyP = (delta_p + prox_rhs).dot(stepSize * (delta_p + prox_rhs))
        tr_check = (costStart - costEnd + penaltyStart - penaltyP) / np.maximum(0.1, costStart - costQuad + penaltyStart - penaltyP)

        # quadratic is an overestimator of cost.
        # f(x) <= f(y) + <nabla(f(y) x-y> + Lf/2 |x-y|^2
        # we demand stepsize phi >= 2 Lf. Then even
        # f(x) <= f(y) + <nabla(f(y) x-y> + phi/4 |x-y|^2
        # (f(x) - f(y) - <nabla(f(y) x-y>) * 4 / |x-y|^2  <= phi, recall actual gradient:
        # J^t fx0 = bp|bl , no L. grad at f(x) is L * JtJDiag * delta_p - s
        # f(x) is costEnd
        # still runs into 646 problem
        # another problem when dre is set to prim_v (since dre<primv) this can be lower than
        # true dre. so we should inc or not use this as best dre?

        # The tr part? Deriv of (delta_l).dot(JltJlDiag * delta_l) at delta_l=0 is 0
        #if newVersion:
            #nablaXp = JtJDiag * delta_p # idea here 1/L is dropped so no need to correct this.
            # TODO This does not belong here, no? DL is not including prox term. It is plain function only.
            #Lfklin = Lfklin + (delta_p + prox_rhs).dot(stepSize * (delta_p + prox_rhs)) - penaltyStartConst

        # TODO actually not clear. I look at at as if l part is not there or implicit.
        # LfkQuad = (delta_l.dot(nablaXl) + delta_p.dot(stepSize * delta_p)) / decent_lemma_divisor

        # Model could be f(x(y), y), where x(y) := argmax_x f(x,y), with p=y, l=x we have
        # (recall x is linear in y: delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl))
        # x=x-delta_l <=> x = x - Vli * ((W.transpose() * delta_p).flatten() + bl), bl = Jl^t f0
        # derivative as df/dx * dx/dy = <bl, Vli * W^T * dy> = <xl, delta_l + Vli * bl> ? looks
        # with df/dx = bl as before and, see above, dx/dy = Vli * W^T * dy (no bl, since not multiplied by delta_p)
        # = delta_l + Vli * bl (added back).
        # also follows from delta_l + Vli * bl = -Vli * ((W.transpose() * delta_p).flatten())
        # weird but assume nabla_l = 0. Then changing y only has this effect.
        # now look at changing y only leads to a change in x as follows
        # Lfklin = (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l + Vli * bl))
        # LfkQuad = delta_p.dot(stepSize * delta_p)) / decent_lemma_divisor
        only_function_of_p = True
        if newVersion and only_function_of_p:
            # TODO: - looks ok but maybe worse. try. also different divisors and 646?
            Lfklin = bp.dot(delta_p) + bl.dot(delta_l + Vli * bl) # '+' or '-'? in +/- bl.dot
            #Lfklin = bp.dot(delta_p) + bl.dot(Vli * ((W.transpose() * delta_p).flatten())) # look also reasonable?
            # the gradient only wrt. delta_p. this should be the rhs of system ignoring the prox and tr part.
            # == bp - W * Vli * bl, yet delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl)
            # hence -(W * Vli * bl)^T delta_p = bl^T (delta_l - Vli * bl). which is not above: +- switch
            Lfklin = (bp - W * Vli * bl).dot(delta_p)
            LfkQuad = delta_p.dot(stepSize * delta_p) / decent_lemma_divisor # my estimate for Lf.
            if jointVersion:
                LfkQuad = delta_p.dot((L * JtJDiag + stepSize) * delta_p) / decent_lemma_divisor

        else:
            #nablaXp = L * JtJDiag * delta_p  # actual gradient. discussable TODO
            nablaXl = JltJlDiag * delta_l  # actual gradient: J^t fx0 = bp|bl
            Lfklin = bp.dot(delta_p) + bl.dot(delta_l)
            LfkQuad = (delta_l.dot(nablaXl) + delta_p.dot(stepSize * delta_p)) / decent_lemma_divisor

        Lfkconst = costEnd - costStart
        LfkDistance  = Lfkconst - Lfklin - LfkQuad
        LfkViolated = LfkDistance > 0
        if True: # TODO test this. Also try to improve this. look at palm vs drs, worst cases.
            LfkViolated = False
            # steSizeTouched: lower blockEigMult and L - if smaller minimum value.
            # so could move lower, to only block print(" |||||||
        LfkSafe = Lfklin < 0 # for any phi ok.

        if tr_check < tr_eta_2: # and False: # TR should not help here. Maybe apply differently? TR checks if approx w. JtJ is ok within region.
            print(" //////  tr_check " , tr_check, " tr_check f: ", (costStart - costEnd) / np.maximum(0.1, costStart - costQuad), " Lfk distance ", LfkDistance, " -nabla^Tdelta=" , -Lfklin, " /////", file=sys.stderr)
            L = L * 2
            if not newVersion:
                JtJDiag = 1/2 * JtJDiag # why that? tr only for landmarks here..
            # else:
            #     stepSize = JtJ.copy() + L * JtJDiag
            #     penaltyStartConst = prox_rhs.dot(stepSize * prox_rhs)
            if jointVersion:
                penaltyStartConst += L * prox_rhs.dot(JtJDiag * prox_rhs)

            # revive idea: tr steered by prox term.
            # stepSize += L * what is added
            # penaltyStartConst += prox_rhs.dot(L * what is added * prox_rhs)

        if tr_check >= tr_eta_2 and LfkViolated:
            steSizeTouched = True
        # LfkViolated = False # todo remove? What happens? all tested ok so far. example 646 failed without.
        if tr_check >= tr_eta_2 and LfkViolated and blockEigMult < globalBlockEigUpperLimit: # violated -- should revert update.
        #if tr_check >= tr_eta_2 and LfkViolated and not steSizeTouched or (steSizeTouched and costStart + penaltyStart < costEnd + penaltyL + penaltyP): # violated -- should revert update.
        #if LfkViolated and not steSizeTouched or (steSizeTouched and costStart + penaltyStart < costEnd + penaltyL): # violated -- should revert update.
            steSizeTouched = True
            print(" |||||||  Lfk distance ", LfkDistance, " -nabla^Tdelta=" , -bp.dot(delta_p) - bl.dot(delta_l), " LipJ ", \
                  LipJ_, " blockEigMult ", blockEigMult , " tr_check, " , tr_check, "  |||||||", file=sys.stderr)
            #print(" |||||||  f(x) <= f(y) + <nabla(f(y) x-y> + Lf/2 |x-y|^2: ", costEnd, " <= ", costStart, " + ", Lfklin, " + ", LfkQuad, " |||||||")

            #stepSize = stepSize * 2
            # other idea, initially we only add 1/2^k eg 0.125, times the needed value and inc if necessary, maybe do not add anything if not needed.

            modThis = False #True # e.g 3068: completely stuck! shit.
            if modThis:
                L = L * 2 # TODO: does this change behaviour?
            else:
                # indeed reliable to get over -- yet not better cost? appears to behave better though.
                blockEigMult_old = blockEigMult
                blockEigMult = np.minimum(globalBlockEigUpperLimit, np.maximum(blockEigMultLimit, blockEigMultGain * blockEigMult))
                stepSize += (blockEigMult - blockEigMult_old) * blockEigenvalueJtJ
                #blockEigenvalueJtJ.data *= 2 # appears slow but safe

            # try this
            #minDiag *= 2
            #stepSize = JJ_mult * JtJ.copy() + diag_sparse(np.fmax(JtJ.diagonal(), minDiag))
            #stepSize = JJ_mult * JtJ.copy() + minDiag * diag_sparse(np.fmax(JtJ.diagonal(), 1e-4))

            # try this, should memorize if works (exists scale s.t. fulfilled)
            # rather if dre is violated increase this.
            #LipJ *= np.sqrt(2) EXTERNALLY -- we do not know if dre increases.
            #stepSize = 1. * (blockEigMult * blockEigenvalueJtJ + J_scale * JtJ.copy()) # Todo '2 *' vs 1 by convex.

            if not newVersion:
                JtJDiag = 1/L * stepSize.copy()
                penaltyStartConst = prox_rhs.dot(JtJDiag * prox_rhs)
            else:
                penaltyStartConst = prox_rhs.dot(stepSize * prox_rhs)
        else:
            LfkViolated = False # hack, also above , or (steSizeTouched and costStart + penaltyStart < costEnd + penaltyL) is hack

        # TODO: go in here despite DL not fulfilled?
        if (newVersion and tr_check >= tr_eta_1 and not LfkViolated) or (not newVersion and LfkSafe and not steSizeTouched):
            L = L / 2
            if not newVersion:
                JtJDiag = 2 * JtJDiag # we return this maybe -- of course stupid to do in a release version

        # TODO: this basically disables lowering blockEigMult !?
        # maybe decrease with *4 off with*2
        allowDecreaseBlockEig = False #True #False # CCC, not sure here. JtJ stepsize vs single value ?
        if (newVersion and LfkSafe and not steSizeTouched) and allowDecreaseBlockEig: # 394 escalates if True here.
            blockEigMult = np.minimum(globalBlockEigUpperLimit, np.maximum(blockEigMultLimit, blockEigMult / 2))

        # version with penalty check for ADMM convergence / descent lemma. Problem: slower?
        #if costStart + penaltyStart < costEnd + penaltyL + penaltyP or LfkViolated: # v1
        if costStart + penaltyStart < costEnd + penaltyP or LfkViolated: # v2
        #if costStart + penaltyStart < costEnd or LfkViolated: # v3
            # revert -- or linesearch
            x0_p_ = x0_p_ - delta_p
            x0_l_ = x0_l_ - delta_l
            x0_ = np.hstack((x0_p_, x0_l_))
            x0_t_ = from_numpy(x0_)
            x0_t_cam = x0_t_[: n_cameras_ * 9].reshape(n_cameras_, 9)
            x0_t_land = x0_t_[n_cameras_ * 9 :].reshape(n_points_, 3)
            updateJacobian = False
            continue # avoids de/increasing L below.
        else:
            it_ = it_ + 1
            updateJacobian = True

        print(" ------- Lfk distance ", LfkDistance, " tr_check ", tr_check,  " LipJ ", LipJ_, " blockEig ", blockEigMult, " -------- ", file=sys.stderr)

    x0_p_ = x0_p_.reshape(n_cameras_, 9)
    x0_l_ = x0_l_.reshape(n_points_, 3)

    # idea use latest for better control ?
    getBetterStepSize = False # this is used as approx of f in update of v and thus s. maybe change there u-v should be small.
    if getBetterStepSize: # needs to set L correctly
        J_pose, J_land, fx0 = ComputeDerivativeMatricesNew(
            x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d, unique_poses_in_c_, unique_landmarks_in_c_)
        bp = J_pose.transpose() * fx0
        JtJ = J_pose.transpose() * J_pose
        #stepSize.data = np.maximum(stepSize.data, blockEigenvalue(JltJl, 3).data) # else diagSparse of it

        nabla_p_approx = JtJDiag * (delta_p + prox_rhs)

        stepSize = blockEigMult * blockEigenvalueJtJ + JJ_mult * JtJ.copy()

        #stepSize = LipJ * JtJ.copy() + J_eps2 * diag_sparse(np.ones(JtJ.shape[0])) # ?
        #stepSize = LipJ * JtJ.copy() + diag_sparse(np.fmax(JtJ.diagonal(), 1e-4))
        JtJDiag = 1/L * stepSize.copy() # max 1, 1/L, line-search dre fails -> increase

        nabla_p_approx2 = JtJDiag * (delta_p + prox_rhs)

        diff_to_nabla_l2_2 = np.linalg.norm(L*nabla_p_approx2+bp, 2)
        diff_to_nabla_l2   = np.linalg.norm(L*nabla_p_approx+bp, 2)
        diff_to_nabla_l4_2 = np.linalg.norm(2*L*nabla_p_approx2+bp, 2)
        diff_to_nabla_l4   = np.linalg.norm(2*L*nabla_p_approx+bp, 2)

        print("diff_to_nabla_p *2 ", diff_to_nabla_l4, " | ", "diff_to_nabla_p ", diff_to_nabla_l2, " |")
        print("diff_to_nabla2_p *2 ", diff_to_nabla_l4_2, " | ", "diff_to_nabla2_p ", diff_to_nabla_l2_2, " |")

        print(" nablas 1", - L * nabla_p_approx ) # So nabla_l_approx = JtJDiag * (u-s), hence return (i use s-u), 2 * JtJDiag * L, the 2 DELIVERS a better cost!
        print(" nablas 2", - L * nabla_p_approx2) # So nabla_l_approx = JtJDiag * (u-s), hence return (i use s-u), JtJDiag * L, the 2 DELIVERS a better cost!
        print(" nablas b", bp) # TINY

    # acceptance this can be reused, rejection the above can be reused.
    if Derivative_at_end: # 646 more constrained but not better. maybe can lower some stuff.
        J_pose, J_land, fx0 = ComputeDerivativeMatricesNew (
            x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d, unique_poses_in_c_, unique_landmarks_in_c_)
        JtJ = J_pose.transpose() * J_pose
        bp = J_pose.transpose() * fx0
        #JJ_mult = 1 + np.maximum(minimumL, np.minimum(L_in_cluster_ * 2, L)) # might have changed .. must be off sigh

        #blockEigenvalueJtJ = threshWhereNeeded * blockEigenvalue(JtJ, 9) # TODO: what if this is only needed for 0-eigen directions? return !=0 only if in small eigendir
        blockEigenvalueJtJ = blockEigenvalueWhereNeeded(JtJ, 9, threshWhereNeeded) # ! 1e-2, 1e-4 1e-5 als works. problem 52, 1e-6 does not 173 performance bad if not 1e-6?

        #stepSize = JJ_mult * JtJ.copy() + blockEigMult * blockEigenvalueJtJ
        stepSize = LipJ_ * JtJ.copy() + blockEigMult * blockEigenvalueJtJ # todo blockEigMult was maybe adjusted above, ok?

        if not newVersion:
            JtJDiag = 1/L * stepSize.copy() # max 1, 1/L, line-search dre fails -> increase
        # else:
        #     JtJDiag = JtJ.copy() + blockEigMultJtJ * blockEigenvalueJtJ
        #     stepSize = JtJ.copy() + L * JtJDiag

    nabla_p = bp.copy()
    Rho = L * JtJDiag #+ 1e-12 * Ul

    if use_be_memory:
        if globalIt < memory_be and globalIt >= len(tempBlockEigen[cluster_id]):
            tempBlockEigen[cluster_id].append(np.minimum(globalBlockEigUpperLimit, np.maximum(blockEigMultLimit, blockEigMult)))
            #print("Appending  cl:", cluster_id, " it: ", globalIt, " mem:", memory_be, " -> ", tempBlockEigen[cluster_id])
        else:
            tempBlockEigen[cluster_id][globalIt % memory_be] = np.minimum(globalBlockEigUpperLimit, np.maximum(blockEigMultLimit, blockEigMult))

    ###################
    # TODO this leads to blockEigMult not shrinking at all?!
    # if important we maybe should never lower it to begin with !? see line 2019
    #    blockEigMult = np.minimum(globalBlockEigUpperLimit, np.maximum(blockEigMultLimit, blockEigMult / 2))
    # if (newVersion and LfkSafe and not steSizeTouched):
    #     blockEigMult = np.minimum(globalBlockEigUpperLimit, np.maximum(blockEigMultLimit, 2*blockEigMult)) # TODOL if ok remove this.
    # else:
    #     blockEigMult = np.minimum(globalBlockEigUpperLimit, np.maximum(blockEigMultLimit, blockEigMult))
    ###################

    if newVersion:
        # stepSize = LipJ_ * JtJ.copy() + blockEigMult * blockEigenvalueJtJ # TODO unclear why this? here it's capped.
        Rho = stepSize # is this an issue if we adjust stepsize?
    if jointVersion:
        Rho = L * JtJDiag + stepSize # AAA

    # TODO: preconditioning should influence this.
    # this should be less communication, where do we get to? 518k vs 875k. With PCG does work now.
    # Rho = blockEigenvalueJtJ + 1e-16 * Rho # issue: needs to be same as drs penalty. else slow ?!
    # Rho = 2 * JtJ.copy() + 1e-12 * Ul # this here means we just use safe bet. Likely multiplier, '2' does not matter anyway. stable but slower. FUCK

    if False: # leads to numerical issues -> influences the results (can only be numerical?).
        Vnorm_ = diag_sparse(np.squeeze(np.asarray(np.sqrt( (np.abs(JltJl)/10).sum(axis=0) ))))
        temp   = Vnorm_.data
        x0_l_ = Vnorm_ * x0_l_.flatten()
        x0_l_ = x0_l_.reshape(-1,3)
        temp_ = Vnorm.data.reshape(-1,3)
        #print( temp_.shape, " ", unique_landmarks_in_c_.shape, " ", temp.shape, " ", points_3d_in.shape, " ",x0_l_.shape)
        temp_[unique_landmarks_in_c_,:] *= temp.reshape(-1,3)
        Vnorm.data = temp_.flatten()
        # breaks if rejected. must use last Vnorm then

    #print("Output cluster_id ", cluster_id, " blockEigMult ", blockEigMult)
    #L_out = np.maximum(minimumL, np.minimum(L_in_cluster_ * 2, L)) # not clear if generally ok, or 2 or 4 should be used.
    L_out = np.maximum(minimumL, (L_in_cluster_ + L) / 2) # not clear if generally ok, or 2 or 4 should be used.
    # L_out = np.maximum(minimumL, L) # WHY THE MIXING ABOVE .. why is this worse LOL? 
    #print(" L " , L_out, L_in_cluster_, L)
    return costEnd / (F_SCALE* F_SCALE), x0_p_, x0_l_, L_out, Rho, nabla_p, blockEigMult

    # recall solution wo. splitting is
    # solve Vl x + bS + Vd ()
    # bl_s = bl + L * JltJlDiag * (x0_l_ - s_l_) # TODO: + or -. '+', see above
    # delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl_s).flatten()
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

#@pyinstrument.profile()
def updateCluster(
    poses_in_cluster_,
    camera_indices_in_cluster_,
    landmark_indices_in_cluster_,
    local_camera_indices_in_cluster,
    local_landmark_indices_in_cluster,
    points_2d_in_cluster_,
    landmarks_,
    poses_s_in_cluster_,
    Vl_in_cluster_,
    L_in_cluster_,
    pose_occurences,
    LipJ,
    blockEig_in_c_,
    cluster_id,
    its_,
):
    landmark_indices_in_c_ = np.unique(landmark_indices_in_cluster_) # 3 input param
    landmarks_in_c = landmarks_[landmark_indices_in_c_]
    #local_landmark_indices_in_cluster = np.zeros(landmark_indices_in_cluster_.shape[0], dtype=int)
    #for i in range(landmark_indices_in_c_.shape[0]): # TODO: precompute THESE: slow!
    #    local_landmark_indices_in_cluster[landmark_indices_in_cluster_ == landmark_indices_in_c_[i]] = i

    torch_points_2d_in_c = from_numpy(points_2d_in_cluster_)
    torch_points_2d_in_c.requires_grad_(False)

    # take point_indices_in_cluster[ci] unique:
    unique_poses_in_c_ = np.unique(camera_indices_in_cluster_)
    # unique_points_in_c_[i] -> i, map each pi : point_indices_in_cluster[ci] to position in unique_points_in_c_[i]
    inverse_pose_indices = -np.ones(np.max(unique_poses_in_c_) + 1)  # all -1
    for i in range(unique_poses_in_c_.shape[0]): # TODO: precompute THESE
        inverse_pose_indices[unique_poses_in_c_[i]] = i

    poses_only_in_cluster_ = pose_occurences[unique_poses_in_c_] == 1
    #print("Unique landmarks  ", landmark_occurences, " ", landmark_occurences.shape, " ", np.min(landmark_occurences), " ", np.max(landmark_occurences))
    #print("Unique landmarks  ", landmarks_only_in_cluster_, " ", np.sum(landmarks_only_in_cluster_), " vs ", np.sum(1 - landmarks_only_in_cluster_) )

    pose_indices_in_c = camera_indices_in_cluster_.copy() # TODO: precompute THESE
    for i in range(camera_indices_in_cluster_.shape[0]):
        pose_indices_in_c[i] = inverse_pose_indices[pose_indices_in_c[i]]

    # put in unique points, adjust point_indices_in_cluster[ci] by id in unique_points_in_c_
    poses_in_c = poses_in_cluster_[unique_poses_in_c_]
    poses_s_in_c = poses_s_in_cluster_[unique_poses_in_c_] # same as landmarks

    cost_, x0_p_c_, x0_l_c_, Lnew_c_, Vl_c_, nabla_p_c_, blockEig_in_c_, = bundle_adjust(
        local_landmark_indices_in_cluster, # these are indexing into landmarks_in_c, a subset of all landmarks, directly.
        pose_indices_in_c,
        poses_only_in_cluster_, # input those poses not present anywhere else to relax hold on those.
        torch_points_2d_in_c,
        landmarks_in_c,
        poses_in_c,
        poses_s_in_c,
        Vl_in_cluster_, # these are for those poses in cluster only.
        L_in_cluster_,
        LipJ,
        blockEig_in_c_,
        unique_poses_in_c_,
        landmark_indices_in_c_, # same as unique poses for pcg
        cluster_id,
        its_,
    )

    return (
        cost_,
        x0_p_c_,
        x0_l_c_, # out side globa lm [landmark_indices_in_c_] = x0_l_c_
        Lnew_c_,
        Vl_c_,
        unique_poses_in_c_,
        landmark_indices_in_c_,
        nabla_p_c_,
        blockEig_in_c_
    )

def prox_f(camera_indices_in_cluster_, point_indices_in_cluster_, local_camera_indices_in_cluster_, local_landmark_indices_in_cluster_,
    points_2d_in_cluster_, poses_in_cluster_, landmarks_, poses_s_in_cluster_, L_in_cluster_, Vl_in_cluster_, blockEig_in_cluster_,
    kClusters, LipJ, innerIts=1, sequential=True) :
    cost_ = np.zeros(kClusters)
    nabla_p_in_cluster_ = [0 for _ in range(kClusters)]

    num_poses = poses_in_cluster_[0].shape[0]
    pose_occurences = np.zeros(num_poses)
    for ci_ in range(kClusters):
        unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci_])
        pose_occurences[unique_poses_in_c_] +=1

    # for ci in range(kClusters):
    #     print(ci, " 3d " ,points_3d_in_cluster_[ci][landmark_occurences==1, :])

    for ci_ in range(kClusters):
        (
            cost_c_,
            x0_p_c_,
            x0_l_c_,
            Lnew_c_,
            Vl_c_,
            unique_poses_in_c_,
            landmark_indices_in_c_,
            nabla_p_c_,
            blockEig_in_c_
        ) = updateCluster(
            poses_in_cluster_[ci_],
            camera_indices_in_cluster_[ci_],
            point_indices_in_cluster_[ci_],
            local_camera_indices_in_cluster_[ci],
            local_landmark_indices_in_cluster_[ci_],
            points_2d_in_cluster_[ci_],
            landmarks_,
            poses_s_in_cluster_[ci_],
            Vl_in_cluster_[ci_],
            # np.max(L_in_cluster_), #L_in_cluster_[ci_], # AAA
            L_in_cluster_[ci_],
            pose_occurences, # haeh?
            LipJ[ci_],
            blockEig_in_cluster_[ci_],
            ci_,
            its_=innerIts,
        )
        cost_[ci_] = cost_c_
        L_in_cluster_[ci_] = Lnew_c_
        Vl_in_cluster_[ci_] = Vl_c_
        poses_in_cluster_[ci_][unique_poses_in_c_, :] = x0_p_c_
        landmarks_[landmark_indices_in_c_] = x0_l_c_
        nabla_p_in_cluster_[ci_] = nabla_p_c_
        blockEig_in_cluster_[ci_] = blockEig_in_c_

    # for ci_ in range(kClusters):
    #     #vl = Vl_in_cluster_[ci_]
    #     unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci_])
    #     poses_only_in_cluster_ = pose_occurences[unique_poses_in_c_] == 1
    #     #globalSingleLandmarksA_in_c[ci] = poses_only_in_cluster_.copy()
    #     #globalSingleLandmarksB_in_c[ci] = poses_only_in_cluster_==1

        # print(ci, " 3d ", points_3d_in_cluster_[ci][landmark_occurences==1, :]) # indeed 1 changed rest is constant
        # print(ci, " vl ", vl.data.reshape(-1,9)[landmarks_only_in_cluster_,:])  # indeed diagonal
    #return (cost_, L_in_cluster_, Vl_in_cluster_, points_3d_in_cluster_, x0_p_, delta_l_in_cluster_, globalSingleLandmarksA_in_c, globalSingleLandmarksB_in_c)
    return (cost_, L_in_cluster_, Vl_in_cluster_, poses_in_cluster_, landmarks_, nabla_p_in_cluster_, blockEig_in_cluster_)


# fill lists G and F, with g and f = g - old g, sets of size m, 
# at position it % m, c^t compute F^tF c + lamda (c - 1/k)^2, sum c=1
# g = x0, f = delta. Actullay xnew = xold + delta.
def RNA(G, F, g, f, it_, m_, Fe, fe, lamda, h, res_pcg):
    #lamda = 0.05 # reasonable 0.01-0.1
    crefVersion = True # TODO
    #lamda = 0.05 # cref version needs larger 
    # h = -1 #-0.1 # 2 / (L+mu) -- should 1/diag * F^t F * c
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

    #FtF = Fs_.transpose().dot(Fs_)
    FtF = Fs_.transpose().dot(res_pcg * Fs_) # why dot?

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
    extrapolation = Gs_.dot(c)
    extrapolationF = Fes_.dot(c)

    print("c ", c, " ", c.shape, id_, file=sys.stderr)

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
        if rhos[j]>0:
            print(j, " 1st. al ", alpha[j], " rh ", rhos[j], " qs " , np.linalg.norm(qs[j],2), " ps " , np.linalg.norm(ps[j],2) )

    dk_ = mu * r

    for i in range(np.maximum(k-mem, 0), k):
        #print("2i", i) # k-1, .. k-mem usually
        j = i % mem # j>=0
        #print("2j", j)
        beta = rhos[j] * np.dot(dk_, qs[j])
        dk_ = dk_ + ps[j] * (alpha[j] - beta)
        if rhos[j]>0:
            print(j, " 2nd. al ", alpha[j], " rh ", rhos[j], " be ", beta, " qs " , np.linalg.norm(qs[j],2), " ps " , np.linalg.norm(ps[j],2) )

    return dk_

def perform_full_iteration(camera_indices_in_cluster_, point_indices_in_cluster_, local_camera_indices_in_cluster_, local_landmark_indices_in_cluster_,
            points_2d_in_cluster_, poses_in_cluster_, landmarks_, poses_s_in_cluster_, L_in_cluster_, Ul_in_cluster_, blockEig_in_cluster__,
            kClusters_, LipJ_, innerIts_, lastCost_, outerit = -1):
    # Only it 0: update s,u,v.
    startT = time.time()
    (
        primal_cost_,
        L_in_cluster_,
        Ul_in_cluster_,
        poses_in_cluster_,
        landmarks_,
        nabla_p_in_cluster_,
        blockEig_in_cluster__
    ) = prox_f(
        camera_indices_in_cluster_, point_indices_in_cluster_, local_camera_indices_in_cluster_, local_landmark_indices_in_cluster_,
        points_2d_in_cluster_, poses_in_cluster_, landmarks_, poses_s_in_cluster_, L_in_cluster_, Ul_in_cluster_, blockEig_in_cluster__,
        kClusters_, LipJ_, innerIts=innerIts_, sequential=True,
        )
    endT = time.time()
    primalCost_u = np.sum(primal_cost_)
    print(outerit, " ", round(primalCost_u), " gain ", round(lastCost_ - primalCost_u), ". ============= sum fk update takes ", endT - startT," s",)

    poses_v_, _, U_cluster_zeros_ = average_cameras_new(
        camera_indices_in_cluster_, poses_in_cluster_, poses_s_in_cluster_, L_in_cluster_, Ul_in_cluster_, nabla_p_in_cluster_)
    # TODO: not updated poses are treated how? v - v old.
    #DRE cost BEFORE s update, always lower than AFTER update.
    dre_, dre_per_part__ = cost_DRE(camera_indices_in_cluster_, poses_in_cluster_, poses_s_in_cluster_, \
                                    L_in_cluster_, Ul_in_cluster_, poses_v_, nabla_p_in_cluster_)
    dre_ += primalCost_u
    steplength_ = 0
    poses_s_in_cluster_pre_ = [0 for e in range(kClusters_)]
    tau_ = 1 # todo sqrt(2), not sure what is happening here.
    for ci in range(kClusters_):
        s_step_cluster_ = poses_v_ - poses_in_cluster_[ci]
        poses_s_in_cluster_pre_[ci] = poses_s_in_cluster_[ci] + tau_ * s_step_cluster_ # update s = s + v - u.
        steplength_ += np.linalg.norm(s_step_cluster_.flatten(), 2)**2
        #update_flat = (poses_s_in_cluster_pre[ci] - poses_s_in_cluster[ci]).flatten()
        #steplength += update_flat.dot(Ul_all * update_flat)
    steplength_ = np.sqrt(steplength_)

    primal_cost_v_ = 0
    for ci in range(kClusters_):
        primal_cost_v_ += primal_cost(
            poses_v_,
            camera_indices_in_cluster_[ci],
            point_indices_in_cluster_[ci],
            local_camera_indices_in_cluster_[ci],
            local_landmark_indices_in_cluster_[ci],
            points_2d_in_cluster_[ci],
            landmarks_)
    primal_cost_u_ = 0
    for ci in range(kClusters_):
        primal_cost_u_ += primal_cost(
            poses_in_cluster_[ci],
            camera_indices_in_cluster_[ci],
            point_indices_in_cluster_[ci],
            local_camera_indices_in_cluster_[ci],
            local_landmark_indices_in_cluster_[ci],
            points_2d_in_cluster_[ci],
            landmarks_)
    dre_ = max( primal_cost_v_, dre_ ) # sandwich lemma, prevent maybe chaos
    print("=== DRE = ", dre_, " ==== f(v)= ", round(primal_cost_v_), " f(u)= ", round(primal_cost_u_), "dre_per_part__ ", dre_per_part__)

    return primalCost_u, dre_, L_in_cluster_, Ul_in_cluster_, poses_in_cluster_, poses_v_, landmarks_, \
        nabla_p_in_cluster_, blockEig_in_cluster__, poses_s_in_cluster_pre_, U_cluster_zeros_, steplength_, primal_cost_v_

def getBlockEigUsed():
    retBlock = []
    tempBlockEigenCopy = [elem.copy() for elem in tempBlockEigen]
    for ci in range(kClusters):
        if len(tempBlockEigenCopy[ci]) > 1:
            if len(tempBlockEigenCopy[ci]) > globalIt % memory_be:
                tempBlockEigenCopy[ci][globalIt % memory_be] = 0 # remove current
            retBlock.append(np.max(np.array(tempBlockEigen[ci]), axis=0))
    return retBlock

def getScaling(min_, max_): # aim at max * min = 1. So max * x = 1/(min * x). x^2 = 1/(min * max)
    # max * np.sqrt(1. / (min * max)) = np.sqrt(max^2 / (min * max)) = np.sqrt(max / min)
    # 1/ (min * np.sqrt(1. / (min * max)) = np.sqrt(min * max / min^2) = np.sqrt(max / min).
    return np.sqrt(1. / (min_ * max_))
    #return np.sqrt(1. / np.maximum(1e-16, (min_ * max_) ))

# Looking at entangled variables, what if we use JtJ + eps * diag(JtJ)^-1/2 as preconditioner?
# JtJ^-1/2 * JtJ * JtJ^-1/2 = I
# send to node once (need also to send lms once, poses all the time)
# how to? compute JtJ+e*diag(JtJ), eigendecomposition, 1/sqrt eigenvalues on diag.
# Let P := JtJ^1/2, Q = JtJ^-1/2
# New variables are y := JtJ^1/2 x
# Yet. use old vars in bundle, apply intenally Q * JtJ * Q? also apply on W.
# problem is what happens to Ws non zero pattern. As I would need to apply on JtJ and W.
# Then when averaging we need to apply P on the input, solve system and apply Q on the output.

# next a local version of this? keep relative weight?
def GetPcgScalingDiag(JtJ, W):
    baseVersion = False #True #False
    if baseVersion:
        temp_  = np.squeeze(np.asarray((np.abs(JtJ)).sum(axis=0) )) # ATTENTION: must adjust / add sqrt on lms here below. CCC
        #[[ 0.11  0.00 -0.07 -0.00  0.11 -0.10 -0.09 -0.09 -0.10]
        # [ 0.00  0.12 -0.03 -0.12  0.00 -0.02 -0.02 -0.03 -0.03]
        # [-0.07 -0.03  0.12  0.03 -0.07  0.07  0.07  0.07  0.07]
        # [-0.00 -0.12  0.03  0.12 -0.00  0.03  0.02  0.03  0.03]
        # [ 0.11  0.00 -0.07 -0.00  0.11 -0.10 -0.09 -0.09 -0.09]
        # [-0.10 -0.02  0.07  0.03 -0.10  0.15  0.14  0.15  0.16]
        # [-0.09 -0.02  0.07  0.02 -0.09  0.14  0.12  0.13  0.14]
        # [-0.09 -0.03  0.07  0.03 -0.09  0.15  0.13  0.17  0.20]
        # [-0.10 -0.03  0.07  0.03 -0.09  0.16  0.14  0.20  0.27]]
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
            #[[ 0.07 -0.00 -0.06 -0.00  0.07 -0.07 -0.07 -0.07 -0.06]
            # [-0.00  0.07 -0.03 -0.07 -0.00 -0.02 -0.02 -0.02 -0.02]
            # [-0.06 -0.03  0.13  0.03 -0.07  0.07  0.07  0.06  0.06]
            # [-0.00 -0.07  0.03  0.07 -0.00  0.02  0.02  0.02  0.02]
            # [ 0.07 -0.00 -0.07 -0.00  0.07 -0.07 -0.07 -0.06 -0.06]
            # [-0.07 -0.02  0.07  0.02 -0.07  0.12  0.12  0.11  0.11]
            # [-0.07 -0.02  0.07  0.02 -0.07  0.12  0.11  0.11  0.11]
            # [-0.07 -0.02  0.06  0.02 -0.06  0.11  0.11  0.13  0.15]
            # [-0.06 -0.02  0.06  0.02 -0.06  0.11  0.11  0.15  0.18]]
        else: # just jacobi, looks best?
            temp_ = np.squeeze(np.asarray((np.abs(JtJ)).sum(axis=0) ))
            temp_W = np.squeeze(np.asarray((np.abs(W)).sum(axis=0) ))
            temp_  = temp_ + temp_W # + 1e-6 does nothing
            # Jacobi pcg: here sqrt here on both, not only on landm. externally
            temp_ = np.squeeze(np.asarray((np.abs(JtJ.diagonal()))))
            temp_ = np.sqrt(temp_) # works on Jacobi, not on rest ?
            #[[ 0.10 -0.01  0.08  0.00  0.10 -0.03 -0.03 -0.03 -0.03]
            # [-0.01  0.10 -0.06 -0.10 -0.01 -0.01 -0.01 -0.01 -0.01]
            # [ 0.08 -0.06  0.10  0.05  0.08 -0.02 -0.02 -0.02 -0.02]
            # [ 0.00 -0.10  0.05  0.10  0.00  0.01  0.01  0.01  0.00]
            # [ 0.10 -0.01  0.08  0.00  0.10 -0.03 -0.02 -0.03 -0.03]
            # [-0.03 -0.01 -0.02  0.01 -0.03  0.10  0.10  0.09  0.08]
            # [-0.03 -0.01 -0.02  0.01 -0.02  0.10  0.10  0.09  0.08]
            # [-0.03 -0.01 -0.02  0.01 -0.03  0.09  0.09  0.10  0.09]
            # [-0.03 -0.01 -0.02  0.00 -0.03  0.08  0.08  0.09  0.09]]

    print("min/max Unorm before ", np.min(temp_), np.max(temp_))
    minTemp = np.percentile(temp_, 0.0001) # not sure..
    t = getScaling(minTemp, np.max(temp_))
    temp_  = temp_ * t
    # temp_  = np.squeeze(np.asarray((np.abs(t * JtJ_)).sum(axis=0) ))
    # temp_W = np.squeeze(np.asarray((np.abs(t * W)).sum(axis=1) ))
    # temp_  = temp_ + temp_W
    print("min/max Unorm after ", np.min(temp_), np.max(temp_), " t ", t, " min*max= ", np.min(temp_) * np.max(temp_))
    #temp_  = temp_.reshape(-1,9)
    print("Preconditioners min/max Unorm ", np.min(temp_), np.max(temp_))
    # e-14 to e16 at -2. -6 ->
    minTresh = 1e-18 # 12 -> 14 for 245 and scale!
    maxTresh = 1e18
    temp_ = np.fmin(np.fmax(temp_, minTresh), maxTresh) #np.sqrt(np.minimum(np.maximum(t, minTresh), maxTresh))
    #temp_ = np.fmin(np.fmax(temp_, 1e-14), 1e16) # TODO. pick most singular example? 646? 173 maybe / any dubrovnik
    print("Preconditioners min/max Unorm after thresholding ", np.min(temp_), np.max(temp_))

    scaleToHaveValuesAroundOneForHess = True # cosmetics mostly.
    if scaleToHaveValuesAroundOneForHess:
        #temp_ /= np.sqrt(t) #np.sqrt(np.minimum(np.maximum(t, minTresh), maxTresh))
        #print("Preconditioners min/max Unorm after scaling 1", np.min(temp_), np.max(temp_))
        absDiagJtJ = np.abs(JtJ.diagonal())
        #print("absDiagJtJ ", absDiagJtJ.shape, " ", absDiagJtJ)
        #print("temp_ ", temp_.shape, " ", temp_)
        guess = diag_sparse(1./temp_.flatten()) * absDiagJtJ * diag_sparse(1./temp_.flatten())
        print("Preconditioners min/max guess ", np.min(guess), np.max(guess))
        #scale = 1. / np.maximum(1, 1./ np.sqrt(np.max(guess)))
        #scale = 1. / np.maximum(1, 1./ np.sqrt(np.mean(guess)))
        #scale = 1. / np.maximum(1, 1./ np.sqrt(np.median(guess)))
        #scale = 1. / np.maximum(1, 1./ np.sqrt(np.min(guess)))
        #scale = 1e-1 * np.sqrt(np.median(guess)) # 245 with scale worse/stalls. 646 wo. max(1, *).
        #scale = np.sqrt(np.max(guess)) # 245 with scale worse/stalls. 646 wo. max(1, *).
        scale = np.sqrt(np.median(guess)) # same as 1e-1 * np.sqrt(np.median(guess))
        print("scale ", scale) # there has to be a stepsize issue?
        temp_ = temp_ * scale # * 1e5 works but not as well ()
        print("Preconditioners min/max Unorm after scaling 2: ", np.min(temp_), np.max(temp_))
        guess = diag_sparse(1./temp_.flatten()) * absDiagJtJ * diag_sparse(1./temp_.flatten())
        print("Preconditioners min/max guess ", np.min(guess), np.max(guess))
        #exit()
        # i could also thresh AGAIN? does not make sense!? more updating? adjust vnorm? stronger descent lemma correction / more?
        #temp_ = np.fmin(np.fmax(temp_, minTresh), maxTresh) #np.sqrt(np.minimum(np.maximum(t, minTresh), maxTresh))
        #print("Preconditioners min/max Unorm after thresholding ", np.min(temp_), np.max(temp_))

    return temp_

def GetPreconditioners(cameras_, points_3d_, points_2d_, camera_indices_, point_indices_):
    J_pose, J_land, fx0_ = ComputeDerivativeMatrixInit(cameras_, points_3d_, points_2d_, camera_indices_, point_indices_)

    JtJ = J_pose.transpose() * J_pose
    #W = J_pose.transpose() * J_land
    orig = False
    if orig:
        temp_ = np.squeeze(np.asarray(0.0001 * ( (np.abs(JtJ)/1000).sum(axis=0) )))
        temp_  = temp_.reshape(-1,9)
        temp_[:,0:5] *= 0.4
        temp_ = np.fmin(np.fmax(temp_, 1e-14), 1e14) # e-14 to e16 at 1e-2.
        print("GetPreconditioners min/max Unorm ", np.min(temp_), np.max(temp_))
        #temp_ = np.fmin(np.fmax(temp_, 1e-14), 1e16) # TODO. pick most singular example? 646? 173 maybe / any dubrovnik
        print("GetPreconditioners min/max Unorm ", np.min(temp_), np.max(temp_))
    else:
        temp_ = GetPcgScalingDiag(JtJ, J_land.transpose() * J_pose)

    # TODO: eval thresh here. lower higher, use 173 maybe w. all lms. Also: redo every 10 iterations?
    # temp_ = np.ones(temp_.shape) # e.g. 173: worse. Likely all w landmarks far away?
    #temp_ = np.sqrt(temp_) # AVOID here.
    Unorm_ = diag_sparse(temp_.copy().flatten())
    #print(Unorm.shape, " ", Unorm.data.shape)
    #Unorm = diag_sparse(np.squeeze(np.asarray(0.01 * np.sqrt( (np.abs(JtJ)/1000).sum(axis=0) ))))
    #Unorm = diag_sparse(np.ones(cameras.flatten().shape[0])) * 100 # ok.
    # print("Unorm.data.reshape(-1,9)", Unorm.data.reshape(-1,9))
    # print("np.sum(fx0**2) ", np.sum(fx0**2))
    # print("cameras ", cameras )

    #print("cameras ", cameras ) # looks ok ..

    # could also compute locally / all the time! 542: appears to 'go crazy' after 20 its.
    JltJl = J_land.transpose() * J_land
    if False:
        Vnorm_ = diag_sparse(np.squeeze(np.asarray((np.abs(JltJl)).sum(axis=0) )))
        temp_  = Vnorm_.data.reshape(-1,3)

        temp_ = np.sqrt(temp_) # does not matter?

        print("min/max Vnorm ", np.min(temp_), np.max(temp_))
        temp_ = 1e-1 * np.fmin(np.fmax(temp_, 1e-10), 1e10) # TODO. pick most singular example? 646 and 52? 1-10 was ok on 52 clust 1e-1, 1e-3 bad? check
        #temp = np.max(np.sqrt(temp), axis=1) # max or mean? sqrt
        #temp_ = np.repeat(temp_[:,np.newaxis], 3, axis=1)
        Vnorm_ = diag_sparse(temp_.flatten())
        #Vnorm_ = diag_sparse(np.ones(points_3d.flatten().shape[0])) # 52: this is much better -- could be random
    else:
        temp_ = GetPcgScalingDiag(JltJl, J_pose.transpose() * J_land)
        # temp_ = np.sqrt(temp_) # a bit better with sqrt (especially for diag prox).
        Vnorm_ = diag_sparse(temp_.flatten())

    return Unorm_, Vnorm_, fx0_

def UpdatePreconditioners(cameras_, points_3d_, points_2d_, camera_indices_, point_indices_, Unorm_old, Vnorm_old):
    Unorm_old_ = Unorm_old.copy()
    Unorm_old_.data = 1. / Unorm_old.data
    cameras_ = (Unorm_old_ * cameras_.flatten()).reshape(-1,9)

    Vnorm_old_ = Vnorm_old.copy()
    Vnorm_old_.data = 1. / Vnorm_old.data
    points_3d_ = (Vnorm_old_ * points_3d_.flatten()).reshape(-1,3)

    # torch_cams = from_numpy(cameras_.reshape(-1,9))
    # # torch_cams.requires_grad_()
    # # torch_cams.retain_grad()
    # torch_lands = from_numpy(points_3d_.reshape(-1,3))
    # # torch_lands.requires_grad_()
    # # torch_lands.retain_grad()
    # torch_points_2d = from_numpy(points_2d_)
    # torch_points_2d.requires_grad_(False)

    # J_pose, J_land, fx0_ = ComputeDerivativeMatricesNew (
    #     torch_cams, torch_lands, camera_indices_, point_indices_,
    #     torch_points_2d, range(torch_cams.shape[0]), range(torch_lands.shape[0]) )

    J_pose, J_land, fx0_ = ComputeDerivativeMatrixInit(cameras_.flatten(), \
        points_3d_, points_2d_, camera_indices_, point_indices_)

    #temp_old  = Unorm_.data.reshape(-1,9)
    JtJ = J_pose.transpose() * J_pose
    # W = J_pose.transpose() * J_land
    # temp_W = np.squeeze(np.asarray((np.abs(1e-6 * W)).sum(axis=1) ))
    # temp_  = temp_ + temp_W
    temp_ = GetPcgScalingDiag(JtJ)
    Unorm_ = diag_sparse(temp_.flatten())
    # print("np.sum(fx0_**2) ", np.sum(fx0_**2))
    # print("cameras ", cameras )
    return Unorm_, fx0_

##############################################################################

# DEFAULT DEBUG examples
BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
#BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/venice/"
#FILE_NAME = "problem-52-64053-pre.txt.bz2"

kClusters = 5 # 10
its = 30

import sys
# total arguments
num_args = len(sys.argv)
if num_args > 2:
    print("Total arguments passed:", num_args)
    # Arguments passed
    print("\nName of Python script:", sys.argv[0], "url ", sys.argv[1], "file ", sys.argv[2])
    BASE_URL =  sys.argv[1]
    FILE_NAME = sys.argv[2]

    if num_args > 3:
        its = int(sys.argv[3])
    if num_args > 4:
        kClusters = int(sys.argv[4])

    URL = BASE_URL + FILE_NAME
    if not os.path.isfile(FILE_NAME):
        urllib.request.urlretrieve(URL, FILE_NAME)

cameras, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

n_cameras = cameras.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

write_output = False
read_output =  False
if read_output:
    # continue with this
    camera_params_np = np.fromfile("camera_params_drs-lm.dat", dtype=np.float64)
    point_params_np = np.fromfile("point_params_drs-lm.dat", dtype=np.float64)
    #camera_params_np = np.fromfile("camera_params_base.dat", dtype=np.float64)
    #point_params_np = np.fromfile("point_params_base.dat", dtype=np.float64)
    x0_p = camera_params_np.reshape(-1)
    x0_l = point_params_np.reshape(-1)
    #x0 = np.concatenate([x0_p, x0_l])
    x0 = np.hstack((x0_p, x0_l))
    x0_t = from_numpy(x0)
    cameras   = x0_p.reshape(n_cameras,9)
    points_3d = x0_l.reshape(n_points,3)
    print("READ DATA")

print("min focal distance ", np.min(cameras[:,6].flatten()), " ", np.max(cameras[:,6].flatten()) )
print("min k1 distance ", np.min(cameras[:,7].flatten()), " ", np.max(cameras[:,7].flatten()) )
print("min k2 distance ", np.min(cameras[:,8].flatten()), " ", np.max(cameras[:,8].flatten()) )

# eval with blockEigenvalueFull, also run 1 it get VL -> evs. use sqrt(|m|_1 * |m|_inf) as rhs mult
# hack: comp Jac HERE. JtJ, define diag matrix 9 x #cam.
# adjust cameras here. define diag in torch, use in res computation.
# torchSingleResiduum X/Y alternative with diag term as input.
# alternative: this defines a basis. then we return not 9x9 but 9 values wrt basis bounding the actual

c02_mult = 1; c34_mult = 1; c5_mult = 1; c6_mult = 1; c7_mult = 1; c8_mult = 1
Unorm, Vnorm, fx0 = GetPreconditioners(cameras, points_3d, points_2d, camera_indices, point_indices)
cameras = (Unorm * cameras.flatten()).reshape(-1,9)
points_3d = (Vnorm * points_3d.flatten()).reshape(-1,3)
Vnorm = 1./Vnorm.data.reshape(-1,3)
Unorm = 1./Unorm.data.reshape(-1,9)

#print("cameras output ", cameras)
#print(Unorm)

# print("cameras ", cameras)
# print("Unorm ", Unorm)

# can i do this?
if False:
    # this is not working so well, print eigs/eval, eg. 8th param, 'k2' is max or min ev / eig.
    # motivaes pcg style treatment, .. Init
    # c34_mult = 20
    # c5_mult = 80
    # c6_mult = 3000
    # c7_mult = 10
    # c8_mult = 20 # amazingly smallest and largest ev are this component in subsequent cams

    c02_mult = 0.01
    c34_mult = 1
    c5_mult = 10
    c6_mult = 100
    c7_mult = 1
    c8_mult = 10 # amazingly smallest and largest ev are this component in subsequent cams

    cameras[:,0:3] = cameras[:,0:3] / c02_mult
    cameras[:,3:6] = cameras[:,3:6] / c34_mult
    cameras[:,5] = cameras[:,5] / c5_mult
    cameras[:,6] = cameras[:,6] / c6_mult
    cameras[:,7] = cameras[:,7] / c7_mult
    cameras[:,8] = cameras[:,8] / c8_mult

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

# todo: return mean reprojectionerror and combine f * k1, f * k2 replacing k1, k2.
# needs adjusted preconditioner avoid too small rescale.

# 1. take problem and split, sort indices by camera, define local global map and test it.
startL = 1e-1 # crazy idea: this lower -> stepsize can be lower as well (less jumps?). It is always worse to not start at 1.
innerIts = 1  # change to get an update, not 1 iteration
cost = np.zeros(kClusters)
lastCost = np.sum(fx0**2)/ (F_SCALE* F_SCALE)
lastCostDRE = np.sum(fx0**2)/ (F_SCALE* F_SCALE)
bestCost = np.sum(fx0**2)/ (F_SCALE* F_SCALE)
bestIt = 0
globalIt = 0
resetIt = 0
failedNesterovAcceleration = 0 # count after k consecutive misses, restart (RNA might not need this)
maxFailedNesterovAcceleration = 3 # 3 or 4
basic_version = False #True # accelerated or basic .. line_search_iterations = 1 (if set below tk==1) is pure DRS
sequential = True
linearize_at_last_solution = True # linearize at uk or v. maybe best to check energy. at u or v. DRE:
lib = ctypes.CDLL("./libprocess_clusters.so")
init_lib()

# todo LipJ_ = ? 1.005? globalBlockEigUpperLimit, globalBlockEigUpperLimit
LipJ = 1 * np.ones(kClusters)
globalBlockEigUpperLimit = 5e-1 # 1e-1, 1e1? # simple stepsize vs JtJ + eps * diag: 1e-3
#globalBlockEigUpperLimit = 1e-3 #1e-3 # 13k cam dataset needs more than 1e-3 and maybe alsobetter partitioning. CCC
blockEig_in_cluster = 1e-5 * np.ones(kClusters) # 1e-4 or 1e-5
memory_be = 4 # here can shrink, below this only grow.
print("input blockEig_in_cluster[ci] ", blockEig_in_cluster[0])

tempEigen = [[] for i in range(kClusters)]
tempBlockEigen = [[] for i in range(kClusters)] # last k multipliers for DL. take max
pre_merges = 0
#pre_merges = int(0.4 * n_cameras) # play to get 'best' cluster. Depends quite a lot
#pre_merges = int(0.02 * n_cameras) # play to get 'best' cluster. This little can induce no deg parts. Those showed 173 & 52 like non deg parts.

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

values, counts = np.unique(camera_indices, return_counts=True)
print(". minimum camera observations in total ", np.min(counts), " cams with < 5 landmarks ", np.sum(counts < 5))

# what if clustering must avoid degenrate clusters?
# e.g. 173 with 6 clusters is much better than with 5! but 5 with! good distribution is better than 6.
# max_c min_i,j in c #(cam_i, lm_j).
# could pick max c s.t. at least 10 are present.
# alternative cams occur in least # clusters.

# (
#     camera_indices_in_cluster,
#     point_indices_in_cluster,
#     points_2d_in_cluster,
#     kClusters,
# ) = cluster_by_landmark(
#     camera_indices, points_2d, point_indices, kClusters, pre_merges)

start = time.time() # this is not working at all. Slower then iteratively
(
    camera_indices_in_cluster,
    point_indices_in_cluster,
    points_2d_in_cluster,
    kClusters,
) = cluster_deg_by_landmark(
    camera_indices, points_2d, point_indices, kClusters)
end = time.time() # this is not working at all. Slower then iteratively
print("========== clustering took ", end - start, " s ==========")

# test, yes much faster if precompute:
local_landmark_indices_in_cluster = []
for ci in range(kClusters):
    landmark_indices_in_c_ = np.unique(point_indices_in_cluster[ci])
    #landmarks_in_c = landmarks_[landmark_indices_in_c_]
    local_landmark_indices_in_cluster.append(np.zeros(point_indices_in_cluster[ci].shape[0], dtype=int))
    for i in range(landmark_indices_in_c_.shape[0]): # TODO: precompute THESE: slow!
        local_landmark_indices_in_cluster[ci][point_indices_in_cluster[ci] == landmark_indices_in_c_[i]] = i

local_camera_indices_in_cluster = []
for ci in range(kClusters):
    cameras_indices_in_c_ = np.unique(camera_indices_in_cluster[ci])
    local_camera_indices_in_cluster.append( np.zeros(camera_indices_in_cluster[ci].shape[0], dtype=int) )
    for i in range(cameras_indices_in_c_.shape[0]): # TODO: precompute these, now if many cams this is slow ?!
        local_camera_indices_in_cluster[ci][camera_indices_in_cluster[ci] == cameras_indices_in_c_[i]] = i

for ci in range(kClusters):
    values, counts = np.unique(camera_indices_in_cluster[ci], return_counts=True)
    print(ci, ". minimum camera observations in cluster ", np.min(counts), " cams with < 5 landmarks ", np.sum(counts < 5))

L_in_cluster = []
for _ in range(kClusters):
    L_in_cluster.append(startL)

print(L_in_cluster)
Ul_in_cluster = [0 for x in range(kClusters)] # dummy fill list
#poses = cameras.copy()
poses_s_in_cluster = [cameras.copy() for _ in range(kClusters)]
poses_in_cluster = [cameras.copy() for _ in range(kClusters)]
landmarks = points_3d.copy()

primal_cost_v = 0
for ci in range(kClusters):
    primal_cost_v += primal_cost(
        poses_in_cluster[ci],
        camera_indices_in_cluster[ci],
        point_indices_in_cluster[ci],
        local_camera_indices_in_cluster[ci],
        local_landmark_indices_in_cluster[ci],
        points_2d_in_cluster[ci],
        landmarks)
print("DEBUG scaled cost ", primal_cost_v)
best_poses_v = poses_in_cluster[0].copy()
best_landmarks = landmarks.copy()
bestCost = primal_cost_v
prevGap = 0
differentialGap = 0
prev_dk = 0

o3d_defined = False
if o3d_defined:
    vis, cameras_vis1, landmarks_vis = render_points_cameras(camera_indices_in_cluster, point_indices_in_cluster, cameras, landmarks)

#with Profiler(interval=0.1) as profiler:

if basic_version:

    for globalIt in range(its):
        start = time.time()
        (
            cost,
            L_in_cluster,
            Ul_in_cluster,
            poses_in_cluster,
            landmarks,
            nabla_p_in_cluster,
            blockEig_in_cluster
        ) = prox_f(
            camera_indices_in_cluster, point_indices_in_cluster, local_camera_indices_in_cluster, local_landmark_indices_in_cluster,
            points_2d_in_cluster, poses_in_cluster, landmarks, poses_s_in_cluster, L_in_cluster, Ul_in_cluster,
            blockEig_in_cluster, kClusters, LipJ, innerIts=innerIts, sequential=True,
            )
        end = time.time()

        #print("++++++++++++++++++ globalSingleLandmarksB_in_c[0].shape ", globalSingleLandmarksB_in_c[0].shape)

        currentCost = np.sum(cost)
        print(globalIt, " ", round(currentCost), " gain ", round(lastCost - currentCost), ". ============= sum fk update takes ", end - start," s",)

        poses_v, _, Up_cluster = average_cameras_new(
            camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, nabla_p_in_cluster) # old_poses for costs?

        #DRE cost BEFORE s update, always lower than AFTER update.
        dre, dre_per_part = cost_DRE(camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, \
                                    L_in_cluster, Ul_in_cluster, poses_v, nabla_p_in_cluster)
        dre += currentCost

        tau = 1 # 2 is best ? does not generalize!
        for ci in range(kClusters):
            temp = Up_cluster[ci].diagonal()
            temp[temp != 0] = 1
            poses_s_in_cluster[ci] = poses_s_in_cluster[ci] + tau * (poses_v - poses_in_cluster[ci]) # update s = s + v - u.
            poses_s_in_cluster[ci] = temp.reshape(-1,9) * poses_s_in_cluster[ci] # set to zero if not in cluster.

        #DRE cost AFTER s update
        #dre = cost_DRE(camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, poses_v, nabla_p_in_cluster) + currentCost

        primal_cost_v = 0
        for ci in range(kClusters):
            primal_cost_v += primal_cost(
                poses_v,
                camera_indices_in_cluster[ci],
                point_indices_in_cluster[ci],
                local_camera_indices_in_cluster[ci],
                local_landmark_indices_in_cluster[ci],
                points_2d_in_cluster[ci],
                landmarks)
        primal_cost_u = 0
        for ci in range(kClusters):
            primal_cost_u += primal_cost(
                poses_in_cluster[ci],
                camera_indices_in_cluster[ci],
                point_indices_in_cluster[ci],
                local_camera_indices_in_cluster[ci],
                local_landmark_indices_in_cluster[ci],
                points_2d_in_cluster[ci],
                landmarks)

        dre = max( primal_cost_v, dre ) # sandwich lemma, prevent maybe chaos
        print( globalIt, " ======== DRE ====== ", round(dre) , " ========= gain " , \
            round(lastCostDRE - dre), "==== f(v)= ", round(primal_cost_v), " f(u)= ", round(primal_cost_u), " BE ", blockEig_in_cluster)

        if lastCostDRE < dre:
            #LipJ += 0.2 * np.ones(kClusters)
            partid = np.argmax(dre_per_part)
            LipJ[partid] = np.minimum(LipJ[partid] * np.sqrt(2), 6)
        # if f(u) < f(v) also raise? not lip but smth else.

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

    tau = 1
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

    (cost, dre, L_in_cluster, Ul_in_cluster, poses_in_cluster, poses_v, landmarks, nabla_p_in_cluster,
     blockEig_in_cluster, poses_s_in_cluster_pre, U_cluster_zeros, steplength, primal_cost_v) = \
        perform_full_iteration(camera_indices_in_cluster, point_indices_in_cluster,
            local_camera_indices_in_cluster, local_landmark_indices_in_cluster,
            points_2d_in_cluster, poses_in_cluster, landmarks, poses_s_in_cluster, L_in_cluster,
            Ul_in_cluster, blockEig_in_cluster, kClusters, LipJ, innerIts, lastCost)
    restartIteration = 0

    if primal_cost_v < bestCost:
        best_poses_v = poses_v.copy()
        best_landmarks = landmarks.copy()
        bestCost = primal_cost_v

    # Only it 0: update s,u,v.
    # start = time.time()
    # (
    #     cost,
    #     L_in_cluster,
    #     Ul_in_cluster,
    #     poses_in_cluster,
    #     landmarks,
    #     nabla_p_in_cluster,
    #     blockEig_in_cluster
    # ) = prox_f(
    #     camera_indices_in_cluster, point_indices_in_cluster, points_2d_in_cluster,
    #     poses_in_cluster, landmarks, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, blockEig_in_cluster,
    #     kClusters, LipJ, innerIts=innerIts, sequential=True,
    #     )
    # end = time.time()
    # currentCost = np.sum(cost)
    # print(-1, " ", round(currentCost), " gain ", round(lastCost - currentCost), ". ============= sum fk update takes ", end - start," s",)

    # poses_v, _, U_cluster_zeros = average_cameras_new(
    #     camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, nabla_p_in_cluster)
    # # TODO: not updated poses are treated how? v - v old.

    # steplength = 0
    # tau = 1 # todo sqrt(2), not sure what is happening here.
    # restartIteration = 0
    # for ci in range(kClusters):
    #     s_step_cluster = poses_v - poses_in_cluster[ci]
    #     poses_s_in_cluster_pre[ci] = poses_s_in_cluster[ci] + tau * s_step_cluster # update s = s + v - u.
    #     steplength += np.linalg.norm(s_step_cluster.flatten(), 2)**2
    #     #update_flat = (poses_s_in_cluster_pre[ci] - poses_s_in_cluster[ci]).flatten()
    #     #steplength += update_flat.dot(Ul_all * update_flat)
    # steplength = np.sqrt(steplength)

    for globalIt in range(its): ##########################################################################################
        # get line search direction and update bfgs data
        # operate with np concatenate to get large vector and reshape search_direction here?
        RNA_or_bfgs = False # RNA is best here ?! ok. else nesterov: False
        if RNA_or_bfgs:
            use_bfgs = False # maybe full u,v?
            bfgs_r = np.zeros(kClusters * 9 * n_cameras)
            rna_s  = np.zeros(kClusters * 9 * n_cameras)
            if use_bfgs:
                use_s_in_rna = False # better False problem is fluctuation. False for RNA works good.
            else:
                use_s_in_rna = False # better False problem is fluctuation. False for RNA works good.

            for ci in range(kClusters): #bfgs_r = u-v
                temp = U_cluster_zeros[ci].diagonal()
                temp[temp != 0] = 1

                bfgs_r[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = temp * (poses_v - poses_in_cluster[ci]).flatten()

                # somewhat unclear: s+ or s to use?
                if not use_s_in_rna:
                    rna_s[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = temp * poses_s_in_cluster_pre[ci].flatten() # TODO: check if not s is used.
                else:
                    # also not stable:
                    # YET this is clearly with h=-1 and lambda high delivering s+.
                    rna_s[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = poses_s_in_cluster[ci].flatten() # TODO: worse. clearly s not s+ should be used, no?
                # recall best working was maybe similar to problem this was on v! directly.
                # sk+1 + (dk + dk-1) or so.

            print("Debug info bfgs_r=u-v: |bfgs_r| ", np.linalg.norm(bfgs_r, 2), file=sys.stderr)

            if use_bfgs:
                dk = BFGS_direction(bfgs_r, bfgs_ps, bfgs_qs, bfgs_rhos, globalIt, bfgs_mem, bfgs_mu)
                dk_stepLength = np.linalg.norm(dk, 2)
                # debug Hessian H fulfills H * (xt+1-xt) = nabla f (xt+1) - nabla f (xt)
                # learn H^-1 and apply on nabla f(x): update delta = - eta * H^-1 nabla f. xt+1 = xt + delta
                # inverse Hess does here fulfill? (test)  delta = (st+1 - st) = H^-1 * ()
                # how does this make sense actually?
                # implemented is    H^-1 * bfgs_qs[it % bfgs_mem] = bfgs_ps[it % bfgs_mem]
                # test:
                #ps_maybe = BFGS_direction(bfgs_qs[(it-1) % bfgs_mem], bfgs_ps, bfgs_qs, bfgs_rhos, it, bfgs_mem, bfgs_mu)
                #print("bfgs test ", np.linalg.norm(ps_maybe - bfgs_ps[(it-1) % bfgs_mem], 2) ) # yes.
                # so ps = r = delta

                # step length by using Vl, also above computing steplength!
                #dk_stepLength = 0
                #for ci in range(kClusters): #bfgs_r = u-v
                    #dk_stepLength += (dk[ci * 3 * n_points: (ci+1) * 3 * n_points]).dot(Ul_all * (dk[ci * 3 * n_points: (ci+1) * 3 * n_points]))
                #dk_stepLength = np.sqrt(dk_stepLength)
                multiplier = 1 #steplength / dk_stepLength
            else:
                #L_rna = max(L_in_cluster) , L_rna * bfgs_r
                # Ui_all = blockInverse(U_all, 9)
                # dk = rna_s.copy()
                # print("dk", dk, " ", bfgs_r, " |bfgs_r| ", np.linalg.norm(bfgs_r, 2), " |dk| ", np.linalg.norm(dk, 2), " |rna_s| ", np.linalg.norm(rna_s, 2))

                # idea does not work. likely too different over time.
                # could input U_all and use for all instead.

                # for ci in range(kClusters):
                #     rna_s[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = U_all * rna_s[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras]
                #     bfgs_r[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = U_all * bfgs_r[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras]
                # print("dk", dk, " ", bfgs_r, " |bfgs_r| ", np.linalg.norm(bfgs_r, 2), " |dk| ", np.linalg.norm(dk, 2), " |rna_s| ", np.linalg.norm(rna_s, 2))
                # print("dk - bfgs_r - rna_s ", dk - bfgs_r - rna_s, " |dk - bfgs_r - rna_s| ", np.linalg.norm(dk - bfgs_r - rna_s, 2))

                U_diag = np.zeros(rna_s.shape)
                RNA_thresh = 1e-6 # 1e-6? - no clue. U_cluster should be around 1 .. could also use Unorm here.
                for ci in range(kClusters):
                    #U_diag[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = blockEigenvalue(U_cluster_zeros[ci], 9).diagonal()
                    #CCC , inverse as done now or 1?
                    U_diag[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = np.fmax(1./np.fmax(np.abs(maxDiagA(U_cluster_zeros[ci], 9).diagonal()), RNA_thresh), RNA_thresh)

                U_diag = diag_sparse(U_diag)
                #print(U_diag) # 
                U_diag.data = np.ones(U_diag.data.shape) # appears better .. ? Does not matter / same (with pcg once)
                # U_diag.data = np.fmax(1./U_diag.data, RNA_thresh) # somewhat better ? maybe best/ or random

                #U_diag = np.ones(rna_s.shape)
                #lambdaScale = np.sqrt(np.mean(U_diag.diagonal()))
                #lambdaScale *= 10.
                #lambdaScale = np.mean(U_diag.diagonal()) # does this make any sense? scale by smth. done inside.
                lambdaScale = 1 # inc if n times? success, dec if fail once -- does this even matter? NO any value not too large works here.
                # h= -1 vs anything else does it matter? -0.1 is worse? -10 is better? Haeh? 
                Gs, Fs, Fes, dk = RNA(Gs, Fs, rna_s, bfgs_r, globalIt - restartIteration, rnaBufferSize, Fes, bfgs_r,
                                    lamda = 0.001 * lambdaScale, h = -5., res_pcg = U_diag) # has changed likely, 0.001 before
                # dk = rna_s + bfgs_r
                # Ui_all = blockInverse(U_all, 9)
                # for ci in range(kClusters):
                #     dk[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = Ui_all * dk[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras]
                #     bfgs_r[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = Ui_all * bfgs_r[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras]
                #     rna_s[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = Ui_all * rna_s[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras]

                if not use_s_in_rna:
                    dk = dk - (rna_s - bfgs_r)
                else:
                    dk = dk - rna_s
                print("Stepsizes taken dk |bfgs_r| ", np.linalg.norm(bfgs_r, 2), " |dk| ", np.linalg.norm(dk, 2), file=sys.stderr )
                dk_stepLength = np.linalg.norm(dk, 2)
                multiplier = 1

            for ci in range(kClusters):
                search_direction[ci] = dk[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras].reshape(n_cameras, 9)
        else:
            # flexible delta_s, delta_s1
            #delta_s1 = np.zeros(kClusters * 9 * n_cameras)
            delta_s  = np.zeros(kClusters * 9 * n_cameras)
            s_new = np.zeros(kClusters * 9 * n_cameras)
            s_cur  = np.zeros(kClusters * 9 * n_cameras)
            for ci in range(kClusters):
                #delta_s1[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = (poses_v - poses_in_cluster[ci]).flatten()
                delta_s [ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = (poses_v - poses_in_cluster[ci]).flatten()
                s_new[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = poses_s_in_cluster_pre[ci].flatten()
                s_cur[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = poses_s_in_cluster[ci].flatten()

            if globalIt <= 0: # s_prev is known
                dk = s_new - s_cur #+ delta_s
                lambda_0 = 1.0
            else:
                delta_s_ = s_new - s_prev
                dk = s_new - s_cur + delta_s_
                if globalIt > 1: # delta_s_old_ is known
                    #dk = delta_s + delta_s_old_ # last 3 steps this is similar to palm. s^k+1 = sk + sum_i=0^2 delta^k-i
                    dk = delta_s_ + delta_s_old_ # last 3 + 2nd step (so 2nd twice).
                # if it > 2: # delta_s_old_ is known
                #     dk = delta_s_ + delta_s_old2_ # last 4? better but fluctuates -- might need to figure out why / how to avoid.
                # if it > 1: # delta_s_old_ is known
                #     delta_s_old2_ = delta_s_old_.copy()

                delta_s_old_ = delta_s_.copy()

                # momentum simple, same for v? about same
                beta_nesterov = (globalIt-resetIt-1) / (globalIt-resetIt+2) # 0.7
                #beta_nesterov = 0.7
                dk = s_new - s_cur + beta_nesterov * prev_dk
                # dk = s_new - s_cur + beta_nesterov * (s_new - s_cur) 
                #vk = s_new - s_cur + 0.7 * prev_vk
                # other idea is 
                if False:
                    lambda_1 = (1. + np.sqrt(1. + 4. * lambda_0**2)) / 2.
                    gamma = (lambda_0 - 1.) / lambda_1
                    dk = s_new - s_cur + gamma * (s_new - s_cur)
                    lambda_0 = lambda_1

            # other idea: treat delta as momentum gradient.
            # later do rms prop?
            prev_dk = dk.copy()

            #dk = s_new - s_cur + 5 * delta_s # all of this is worse for mult = 1 .. 4
            dk_stepLength = np.linalg.norm(dk, 2)
            multiplier = 1 # steplength / dk_stepLength # Haeh?
            for ci in range(kClusters):
                search_direction[ci] = dk[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras].reshape(n_cameras, 9)

            s_prev = s_cur.copy() # access to old s.

        #line_search_iterations = 1 # is pure DRS (forced, see below set tk == 1)
        line_search_iterations = 2 # 3 appears ok
        if globalIt <= resetIt + 1:
            line_search_iterations = 1

        print(" ..... step length ", steplength, " bfgs step ", dk_stepLength, " ratio ", multiplier, file=sys.stderr )
        Vnorm_safe = Vnorm.copy()
        for ls_it in range(line_search_iterations):
            Vnorm = Vnorm_safe.copy()
            if line_search_iterations >1:
                tk = ls_it / (line_search_iterations-1)
            else: # debug
                tk = 1 # 0: line-search 1: drs
            for ci in range(kClusters):
                poses_s_in_cluster_bfgs[ci] = tk * poses_s_in_cluster_pre[ci] + (1-tk) * (poses_s_in_cluster[ci] + multiplier * search_direction[ci])
                #print(" bfgs_r ", bfgs_r[ci * 3 * n_points: (ci+1) * 3 * n_points].reshape(n_points, 3))
                #print(" search_direction[ci] ", search_direction[ci])

                if False: # totally stcuk solution in base has BS values as solution k1 = 600k, k2 =-200
                    print("s focal distance ", np.min(poses_s_in_cluster_bfgs[ci][:,6]), " - ", np.max(poses_s_in_cluster_bfgs[ci][:,6]))
                    print("s k1 distance ", np.min(poses_s_in_cluster_bfgs[ci][:,7]), " - ", np.max(poses_s_in_cluster_bfgs[ci][:,7]))
                    print("s k2 distance ", np.min(poses_s_in_cluster_bfgs[ci][:,8]), " - ", np.max(poses_s_in_cluster_bfgs[ci][:,8]))

            # prox on line search s:
            #print("1. x0_p", "points_3d_in_cluster", points_3d_in_cluster)
            if True or linearize_at_last_solution: # linearize at v / average solution, same issue I suppose. Yes. solution is too return the new gradient, s.t. update of v is wrt to current situation.
                poses_in_cluster_bfgs = [elem.copy() for elem in poses_in_cluster]
            else: # does not work well here.
                poses_in_cluster_bfgs = [poses_v.copy() for _ in range(kClusters)]

            L_in_cluster_bfgs = L_in_cluster.copy()
            Ul_in_cluster_bfgs = [elem.copy() for elem in Ul_in_cluster]
            blockEig_in_cluster_bfgs = [elem.copy() for elem in blockEig_in_cluster]
            (   cost_bfgs,
                L_in_cluster_bfgs,
                Ul_in_cluster_bfgs,
                poses_in_cluster_bfgs,
                landmarks_bfgs,
                nabla_p_in_cluster_bfgs,
                blockEig_in_cluster_bfgs
            ) = prox_f(
                camera_indices_in_cluster, point_indices_in_cluster, local_camera_indices_in_cluster, local_landmark_indices_in_cluster,
                points_2d_in_cluster, poses_in_cluster_bfgs, landmarks.copy(), poses_s_in_cluster_bfgs, L_in_cluster_bfgs,
                Ul_in_cluster_bfgs, blockEig_in_cluster_bfgs, kClusters, LipJ, innerIts=innerIts, sequential=True,
                )
            
            #print("2. x0_p", "points_3d_in_cluster", points_3d_in_cluster)
            currentCost_bfgs = np.sum(cost_bfgs)
            poses_v_bfgs, Ul_all_bfgs, U_cluster_zeros = average_cameras_new(
                camera_indices_in_cluster, poses_in_cluster_bfgs, poses_s_in_cluster_bfgs, L_in_cluster_bfgs, Ul_in_cluster_bfgs, nabla_p_in_cluster_bfgs)

            # update buffers
            if RNA_or_bfgs and use_bfgs and ls_it == 0: # todo: the one we accept put here, no?
                #bfgs_ps[globalIt % bfgs_mem] = -dk * multiplier
                bfgs_ps[globalIt % bfgs_mem] = -bfgs_r # this is not so much overshooting as dk
                bfgs_rr = np.zeros(kClusters * 9 * n_cameras)
                for ci in range(kClusters):
                    bfgs_rr[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = poses_v_bfgs.flatten() - poses_in_cluster_bfgs[ci].flatten() # flatten?
                bfgs_qs[globalIt % bfgs_mem] = bfgs_rr - bfgs_r
                bfgs_rhos[globalIt % bfgs_mem] = np.maximum(0., 1./ bfgs_qs[globalIt % bfgs_mem].dot(bfgs_ps[globalIt % bfgs_mem]))

            # eval cost
            dre_bfgs, dre_per_part = cost_DRE(camera_indices_in_cluster, poses_in_cluster_bfgs, poses_s_in_cluster_bfgs,
                                L_in_cluster_bfgs, Ul_in_cluster_bfgs, poses_v_bfgs, nabla_p_in_cluster_bfgs)
            dre_bfgs += currentCost_bfgs

            # debugging cost block ################
            primal_cost_v = 0
            primal_cost_v_all = []
            primal_cost_u_all = []
            for ci in range(kClusters):
                primal_cost_v_all.append(primal_cost(
                    poses_v_bfgs, # v not u
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    local_camera_indices_in_cluster[ci],
                    local_landmark_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    landmarks_bfgs))
            primal_cost_v = np.sum(primal_cost_v_all)
            primal_cost_v_all = [round(cost) for cost in primal_cost_v_all]
            primal_cost_u = 0 # = currentCost_bfgs
            for ci in range(kClusters):
                primal_cost_u_all.append(primal_cost(
                    poses_in_cluster_bfgs[ci],
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    local_camera_indices_in_cluster[ci],
                    local_landmark_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    landmarks_bfgs))
            primal_cost_u = np.sum(primal_cost_u_all)
            primal_cost_u_all = [round(cost) for cost in primal_cost_u_all]

            dre_bfgs = max(dre_bfgs, primal_cost_v) # sandwich lemma
            blockEigLastIt = getBlockEigUsed() # the actual used not the one written into memory or whatever blockEig_in_cluster_bfgs is.
            # diffToGain too large -> raise 'be' e.g. * 2, only if accepted.
            diffToGain = np.maximum(round(primal_cost_v) - round(primal_cost_u) - round(lastCostDRE_bfgs - dre_bfgs), 0.) / round(primal_cost_u)
            #gapToGain = np.maximum(1, round(lastCostDRE_bfgs - dre_bfgs)) / np.maximum(round(primal_cost_v) - round(primal_cost_u), 1)
            # ~its to fill gap
            gapToGain = np.maximum(1. * round(primal_cost_v- primal_cost_u) - round(lastCostDRE_bfgs - dre_bfgs), 1.) / np.maximum(1, round(lastCostDRE_bfgs - dre_bfgs))
            currentGap = np.maximum(1. * round(primal_cost_v - primal_cost_u), 1. ) #- round(lastCostDRE_bfgs - dre_bfgs), 1) # not sure .. 
            differentialGap = prevGap - currentGap
            costGain = lastCostDRE_bfgs - dre_bfgs
            G2C = round(1000 * costGain / currentGap) / 1000
            print( globalIt, "/", ls_it, " ======== DRE BFGS ====== ", round(dre_bfgs) , " ========= gain " , \
                round(costGain), "==== f(v)= ", round(primal_cost_v), " f(u)= ", round(primal_cost_u),
                " G ", currentGap , " dG ", differentialGap, " ", differentialGap / np.maximum(costGain, 1.), #" D2G ", diffToGain, "G2G ", gapToGain, 
                " G2C ", G2C, " BE ", blockEigLastIt, " L ", L_in_cluster_bfgs) #blockEig_in_cluster_bfgs)
            print( globalIt, "/", ls_it, " f(v) = ", primal_cost_v_all, " f(u) = ", primal_cost_u_all)
            prevGap = currentGap.copy()

            if primal_cost_v < bestCost:
                best_poses_v = poses_v_bfgs.copy()
                best_landmarks = landmarks_bfgs.copy()
            bestCost = np.minimum(primal_cost_v, bestCost)
            bestIt = globalIt
            if globalIt < 60:
                bestCost60 = bestCost
            if globalIt < 30:
                bestCost30 = bestCost

            # TODO: inc tempBlockEigen[cluster_id] *=2 -- if possible u/v
            beMin = globalBlockEigUpperLimit
            for ci in range(kClusters):
                beMin = np.minimum(beMin, tempBlockEigen[ci][globalIt % memory_be])

            LipJMax = 16 # 6.x was computed
            iteration_factor      = (1 - (globalIt / its))**4 # 1 at start, ~0 at end.
            iteration_factor_five = (1 - (5 / its))**4 #, could also running mean of gains and use 5% of those (positive gains, if < - (5% of mean) ).
            maxPct = 1 + 0.01 * iteration_factor/iteration_factor_five # aim at 1% at 5 iterations?
            startFromU = False
            reject = True # the general act.
            # normally use u,v. Forgot example why / where v is better.
            # likely 'things' go 'wild'. hmm.

            disable_best_pose = False
            if disable_best_pose:
                best_poses_v = poses_v.copy()
                best_landmarks = landmarks.copy()

            # idea accept if primal v cost is very close.
            # can happen that best primal cost is about same as current and dre was set to this as correction. 
            # TODO if dre < primal_v also increase LipJ or so.
            if (beMin < globalBlockEigUpperLimit) and (ls_it == line_search_iterations-1) and (maxPct * lastCostDRE_bfgs < dre_bfgs):
            #if (np.min(LipJ) < LipJMax) and (ls_it == line_search_iterations-1) and (maxPct * lastCostDRE_bfgs < dre_bfgs):
                primal_cost_v_before = 0 # should be fixed also
                for ci in range(kClusters):
                    primal_cost_v_before += primal_cost(
                        best_poses_v, # v not u
                        camera_indices_in_cluster[ci],
                        point_indices_in_cluster[ci],
                        local_camera_indices_in_cluster[ci],
                        local_landmark_indices_in_cluster[ci],
                        points_2d_in_cluster[ci],
                        best_landmarks)
            # do not if primal_v best and current are about the same.

            # Reset acceleration if fails 6 times in a row
            if RNA_or_bfgs == False and ls_it == line_search_iterations - 1 and line_search_iterations > 1:
                failedNesterovAcceleration += 1
                if False:
                    lambda_0 = np.maximum(1., lambda_1 / 2.) # reset acceleration
                    #lambda_0 = np.maximum(1., lambda_1 / np.sqrt(5)) # reset acceleration, less flickering can be worse results.
                    prev_dk = s_new - s_cur
                    print('lambda_0 reset ', lambda_0)

                if failedNesterovAcceleration >= maxFailedNesterovAcceleration:
                    prev_dk = 0 * prev_dk
                    resetIt = globalIt
                    failedNesterovAcceleration = 0
                    restartIteration = globalIt # reset RNA has no effect.
                    Gs = []
                    Fs = []
                    Fes = []
                    print("Reset Nesterov acceleration after ", maxFailedNesterovAcceleration, " consecutive failures.")

            maxPctV = np.maximum(1.001, np.sqrt(maxPct)) # max 0.1 % AAA

            #if reject and (np.min(LipJ) < LipJMax) and (ls_it == line_search_iterations-1) and (maxPct * lastCostDRE_bfgs < dre_bfgs) and (primal_cost_v > maxPctV * primal_cost_v_before): # or primal_cost_v > maxPct * primal_cost_u):
            # if reject and (beMin < globalBlockEigUpperLimit) and (ls_it == line_search_iterations-1 and line_search_iterations > 1) and (maxPct * lastCostDRE_bfgs < dre_bfgs) and (primal_cost_v > maxPctV * primal_cost_v_before): # or primal_cost_v > maxPct * primal_cost_u):
            if reject and (beMin < globalBlockEigUpperLimit) and (ls_it == line_search_iterations-1) and (maxPct * lastCostDRE_bfgs < dre_bfgs) and (primal_cost_v > maxPctV * primal_cost_v_before): 
                print("Why enter is primal cost (v) bad or what", primal_cost_v, " > ", maxPctV * primal_cost_v_before, " > ", primal_cost_v_before, " * ", maxPctV)

                # revert ! Not clear how to do this.
                # before, _ = cost_DRE(camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster,
                #                    L_in_cluster, Ul_in_cluster, poses_v, nabla_p_in_cluster_bfgs)
                # # should be same as poses_v, so not needed./debug: ok.
                # poses_v_before, Ul_all_bfgs, U_cluster_zeros = average_cameras_new(
                #     camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, nabla_p_in_cluster_bfgs)
                # print("|poses_v_before - poses_v| = ", np.linalg.norm(poses_v_before - poses_v, 2))

                num_positive_elements = sum(1 for element in dre_per_part if element > 0)
                if False and maxPct * lastCostDRE_bfgs > dre_bfgs and 4 * num_positive_elements < kClusters: # only adjust few parts.
                    print("dre_per_part ", dre_per_part, " adjust those >0.")
                    for ci in range(kClusters):
                        if dre_per_part[ci] > 0:
                            LipJ[ci] *= np.sqrt(2)
                            #Ul_in_cluster[ci] *= np.sqrt(2) # approx
                            print("LipJ[" , ci, "] *= sqrt(2)") # 

                else: # adjust all
                    # if i use u, the cost is likely to grow instead, since i increase the penalty?
                    # for ci in range(kClusters):
                    #     poses_in_cluster[ci] = poses_v.copy() # 'best_v' instead != last_v? this alone is bfgs_r = u-v =0.
                    #     poses_s_in_cluster[ci] = poses_v.copy() # this alone is killing RNA fnorm = 0.

                    # TODO: either LipJ does not work or other ? 
                    oneRound = True # loop has an issue that it is not only dependent on the input, but more. blockEig ?
                    # so more needs to be reset than I do.

                    # super basic?
                    # VERSION v
                    poses_in_cluster = [best_poses_v.copy() for _ in poses_in_cluster]
                    for ci in range(kClusters):
                        if RNA_or_bfgs:
                            poses_in_cluster[ci][:,5] += 1e-6 # 1e-6 is enough to make it different.
                        poses_s_in_cluster_pre[ci] = best_poses_v.copy() # s + u-v = s in this case, do we use the best s?
                        poses_s_in_cluster[ci] = best_poses_v.copy()
                    landmarks = best_landmarks.copy()
                    # TODO: s-> best_v & u=v after reset ? landmark match best v? -- we can/could compute lms from v only: yes: VLi * Jl * res, poses fixed.

                    # IDEA: verify cost here.
                    CheckCost = True # temporal test. it appears odd that this is so bad. Maybe set be very strict for one iteration?
                    if CheckCost:
                        primal_cost_v_check = 0
                        for ci in range(kClusters):
                            primal_cost_v_check += primal_cost(
                                poses_in_cluster[ci], # v not u
                                camera_indices_in_cluster[ci],
                                point_indices_in_cluster[ci],
                                local_camera_indices_in_cluster[ci],
                                local_landmark_indices_in_cluster[ci],
                                points_2d_in_cluster[ci],
                                landmarks)
                        print("Checking cost after rest: ", round(primal_cost_v_check), " vs. ", round(primal_cost_v), " vs. ", round(primal_cost_u))


                    ############
                    # VERSION U is doing nothing actually. This is differetn if taking actula steps as below.
                    # poses_s_in_cluster = [elem.copy() for elem in poses_s_in_cluster_pre]
                    # poses_in_cluster_test = [elem.copy() for elem in poses_in_cluster]
                    ############
                    oneRound = False
                    # TODO: LipJ or tempBlockEigen.
                    #LipJ *= np.sqrt(2)
                    tmp = []
                    # innerIts = 2 # BBB inc temporaily at failure, avoid direct failure again? can we?
                    # eval wrt. # of total its.
                    be_mult = 2 # if we have *= sqrt idea below this can be lower?
                    # be_mult = 4 #for JtJ? CCC
                    for ci in range(kClusters):
                        tempBlockEigen[ci][globalIt % memory_be] = \
                            np.minimum(tempBlockEigen[ci][globalIt % memory_be] * be_mult, globalBlockEigUpperLimit)
                        tmp.append(tempBlockEigen[ci][globalIt % memory_be])
                    print("Be *= ", be_mult, " -> Be= ", tmp, " LipJ " , np.mean(LipJ))

                    # TODO: equalize / reset nesterov(acceleration) here.
                    AlsoResetNesterovAcceleration = False #True # test on 646, 1266, 1064, 961, 427, 1778 -> no conclusion.
                    if AlsoResetNesterovAcceleration:
                        prev_dk = 0 * prev_dk
                        resetIt = globalIt
                        print("Reset Nesterov acceleration after ", failedNesterovAcceleration, " failures.")
                        failedNesterovAcceleration = 0

                    # if 1 fails alawya will.
                    while oneRound and (np.min(LipJ) < LipJMax) and (maxPct * lastCostDRE_bfgs < dre_bfgs) and (primal_cost_v > maxPctV * primal_cost_v_before): # while since LipJ must be large enough.
                        oneRound = False
                        LipJ *= np.sqrt(2) # there could one 1 particularly bad one
                        # if np.min(LipJ) > 128:
                        #     exit()
                        # for cin in range(kClusters):
                        #     Ul_in_cluster[ci] *= np.sqrt(2) # approx
                        print("executed LipJ *= sqrt(2) = ", np.mean(LipJ))
                        # all of this fails.
                        primal_cost_v_before = 0 # TODO out of loop if correct
                        for ci in range(kClusters):
                            primal_cost_v_before += primal_cost(
                                poses_v, # v not u
                                camera_indices_in_cluster[ci],
                                point_indices_in_cluster[ci],
                                local_camera_indices_in_cluster[ci],
                                local_landmark_indices_in_cluster[ci],
                                points_2d_in_cluster[ci],
                                landmarks)
                        # this outside maybe? maybe not even enter here at all.
                        # if primal_cost_v_before > primal_cost_v: # primal_cost_v is last bfgs step.
                        #     poses_v = poses_v_bfgs.copy()
                        #     landmarks = landmarks_bfgs.copy()
                        #     primal_cost_v_before = primal_cost_v 
                        #     print("Copy over bfgs cost, new primal_cost_v_before ", primal_cost_v_before)

                        # copy wither all the time as those get overwritten.
                        if not startFromU: # else from v
                            poses_in_cluster_test = [poses_v.copy() for _ in poses_in_cluster]
                            for ci in range(kClusters):
                                poses_s_in_cluster[ci] = poses_v.copy()
                            # clearly the best one can reach is the primal v cost.
                        else:
                            poses_s_in_cluster = [elem.copy() for elem in poses_s_in_cluster_pre]
                            poses_in_cluster_test = [elem.copy() for elem in poses_in_cluster]
                        blockEig_in_cluster_in = [elem.copy() for elem in blockEig_in_cluster]

                        # TODO: better was to not do this additional step for some reason. test 427, also look 646
                        print("TODO: Big riddle, redo it primal_cost_v_before changes in loop ", round(primal_cost_v_before) )
                        # TODO: reset RNA memory to 0 = forget past ?!
                        # correct: 1 it wo. linesearch. skip somehow.
                        # restartIteration = it;Gs = [Gs[it%rnaBufferSize]]; Fs = [[it%rnaBufferSize]]; Fes = [[it%rnaBufferSize]] # ? it+1?
                        # TODO: poses_v can be much worse than the input poses_v.
                        # Also poses_v may not be the best solution observed. (poses/landmarks)
                        landmarks_test = landmarks.copy() # below overwrites landmarks internally.
                        (cost, dre, L_in_cluster, Ul_in_cluster, poses_in_cluster_test, poses_v_test, landmarks_test, \
                        nabla_p_in_cluster, blockEig_in_cluster_test, poses_s_in_cluster_pre_test, U_cluster_zeros, steplength) = \
                            perform_full_iteration(camera_indices_in_cluster, point_indices_in_cluster, local_camera_indices_in_cluster,
                                local_landmark_indices_in_cluster, points_2d_in_cluster, poses_in_cluster_test, landmarks_test,
                                poses_s_in_cluster, L_in_cluster, Ul_in_cluster, blockEig_in_cluster_in, kClusters, LipJ, innerIts,
                                lastCostDRE_bfgs, outerit = globalIt)
                        print("========== dre/dre_bfgs/lastCostDRE_bfgs  ===========", round(dre), " / " , round(dre_bfgs), " / ", round(lastCostDRE_bfgs), " BE ", blockEig_in_cluster_bfgs)
                        if maxPct * lastCostDRE_bfgs >= dre and (primal_cost_v > maxPctV * primal_cost_v_before): # better: overwrite
                            print("Overwrite poses/landmarks/bl-eig")
                            poses_v = poses_v_test
                            landmarks = landmarks_test
                            blockEig_in_cluster = [elem.copy() for elem in blockEig_in_cluster_test]
                            poses_in_cluster = [elem.copy() for elem in poses_in_cluster_test]
                            poses_s_in_cluster_pre = [elem.copy() for elem in poses_s_in_cluster_pre_test]

                        dre_bfgs = dre # while loop uses this
                    if True: # is True better maybe -> Does not do ANYTHING? 52/89, test smth else.
                        lastCostDRE_bfgs = dre_bfgs # does not happen? needed to not keep running in it. Yet demanding Lip < X to enter should work

                # revert whole iteration.
                #it = it - 1
                #its = its - 1
                print(" ************** REVERTED iteration **************, DRE cost set to ", lastCostDRE_bfgs, \
                    " before ", primal_cost_v_before, " min LipJ", np.min(LipJ))
                # must update costs as well. if LipJ increased, then costs are not valid. Recompute how?
                # u and v and s known, but not step size change.

                # poses_v, Ul_all_bfgs, U_cluster_zeros = average_cameras_new(# same if all U are adjusted / multiplied by a factor.
                #     camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster, L_in_cluster, Ul_in_cluster, nabla_p_in_cluster_bfgs)
                # print("|poses_v_before - poses_v| = ", np.linalg.norm(poses_v_before - poses_v, 2)) # v does not change Since multiplied by same value!
                # after, _ = cost_DRE(camera_indices_in_cluster, poses_in_cluster, poses_s_in_cluster,
                #                    L_in_cluster, Ul_in_cluster, poses_v, nabla_p_in_cluster_bfgs)
                # primal_cost_v_before = 0
                # for ci in range(kClusters):
                #     primal_cost_v_before += primal_cost(
                #         poses_v, # pose are different now? Should be after changing U.
                #         camera_indices_in_cluster[ci],
                #         point_indices_in_cluster[ci],
                #         points_2d_in_cluster[ci],
                #         landmarks)
                #print("lastCostDRE_bfgs", lastCostDRE_bfgs, " primal_cost_v_before ", primal_cost_v_before, " dre after/before changing U ", after, " - ", before, " = ", after - before )
                # if primal_cost_v_before > lastCostDRE_bfgs: # cost did not update. use v to replace u.
                #     for ci in range(kClusters):
                #         poses_in_cluster[ci] = poses_v.copy()


                #lastCostDRE_bfgs = max(lastCostDRE_bfgs + after - before, primal_cost_v_before) # this should be different. Canot be the same. but it is.

                # what is different?
                # s = (v-u). U is U * sqrt(2)?
                # if f(v) < f(u), and f(u) + (u-s)^T U (u-s) ->
                # pulls towards v, f(v) has lower cost, so good? Haeh?
                # s can be far beyond v though
                # u ... v .... s "drawing a line"
                # I can do a restart at u=v, s=0 and start at v. for all part with f(v)<f(u) ?
                # use s = s + (u-v)? could also work.

                # ************** REVERTED iteration **************
                # ---- |u-s|^2_D  1131028 |u-v|^2_D  1294132 |2u-s-v|^2_D  167405 |u-v|^2  43936  cost_dre  -481811.5150287446
                # ---- dre_per_part ---  [-980, -5893, -204140, -6156, -5824, -6608, -6747, -5595, -6263, -233605]
                # lastCostDRE_bfgs 719617.989097168  primal_cost_v_before  930026.222556065  dre after/before changing U  -481811.5150287446   -264417.30638654006
                # ********************** NEW AT IT   10  /  89  **********************

            else:

                # differentialGap / np.maximum(costGain, 1)
                # if diffToGain > 0.2: CCC: turn off for JtJ stepsize?
                if costGain < 0 and differentialGap < 0 and currentGap > 1 and (ls_it == line_search_iterations-1): # gap present fv - fu >0, gets wider and cost higher than best
                    be_mult__ = np.sqrt(2)
                    tmp__ = []
                    for ci in range(kClusters):
                        tempBlockEigen[ci][globalIt % memory_be] = \
                            np.minimum(tempBlockEigen[ci][globalIt % memory_be] * be_mult__, globalBlockEigUpperLimit)
                        tmp__.append(tempBlockEigen[ci][globalIt % memory_be])
                    print("Be *= ", be_mult__, " Be ", tmp__, " ", np.mean(LipJ))
                    # innerIts = 2 BBB

                # if ls_it == line_search_iterations-1:
                #     innerIts = 1 # BBB

                # if lastCostDRE_bfgs < dre_bfgs and ls_it == line_search_iterations-1:
                #     #LipJ += 0.2 * np.ones(kClusters)
                #     partid = np.argmax(dre_per_part)
                #     LipJ[partid] *= np.sqrt(2) # maybe just the LARGEST dre cost.

                # accept / reject, reject all but drs and see
                # if ls_it == line_search_iterations-1 :
                if dre_bfgs <= lastCostDRE_bfgs or 10000 * (dre_bfgs-lastCostDRE_bfgs) <= lastCostDRE_bfgs or (ls_it == line_search_iterations-1 and line_search_iterations > 1): # not correct yet, must be <= last - c/gamma |u-v|
                    # TODO: save and use best wrt. cost: new fct?
                    steplength = 0 # currently printed
                    for ci in range(kClusters):
                        poses_s_in_cluster[ci] = poses_s_in_cluster_bfgs[ci].copy()
                        s_step_cluster = poses_v_bfgs - poses_in_cluster_bfgs[ci]
                        poses_s_in_cluster_pre[ci] = poses_s_in_cluster[ci] + tau * s_step_cluster # update s = s + v - u.
                        steplength += np.linalg.norm(s_step_cluster.flatten(), 2)**2
                        #update_flat = (poses_s_in_cluster_pre[ci] - poses_s_in_cluster[ci]).flatten()
                        #steplength += update_flat.dot(Ul_all * update_flat)
                    steplength = np.sqrt(steplength)

                    for ci in range(kClusters):
                        poses_in_cluster[ci] = poses_in_cluster_bfgs[ci].copy()
                        Ul_in_cluster[ci] = Ul_in_cluster_bfgs[ci].copy()
                        blockEig_in_cluster[ci] = blockEig_in_cluster_bfgs[ci].copy()
                    L_in_cluster = L_in_cluster_bfgs.copy()
                    landmarks = landmarks_bfgs.copy()
                    lastCostDRE_bfgs = dre_bfgs.copy()
                    poses_v = poses_v_bfgs.copy()

                    # for ci in range(kClusters):
                    #     print(ci, " poses_v ", poses_in_cluster[ci])

                    equalizeBlockEig = False #True
                    if equalizeBlockEig:
                        # all mem are forced equal in all clusters / maybe even better in time (so only grow or very long memory):
                        for m in range(len(tempBlockEigen[ci])):
                            maxM = 0
                            for ci in range(kClusters):
                                maxM = np.maximum(tempBlockEigen[ci][m], maxM)
                            for ci in range(kClusters):
                                tempBlockEigen[ci][m] = maxM

                    if False and globalIt % 10 == 9: # after bigger analysis: not needed.
                    #if True and globalIt % 3 == 2: # debatable, bigger analysis needed. Just random?
                        Unorm = diag_sparse(1./Unorm.data) #.reshape(-1,9)
                        Unorm_update, fx0 = UpdatePreconditioners(poses_v, landmarks, points_2d, camera_indices, point_indices, Unorm, diag_sparse(1./Vnorm.data))

                        # 1. update poses, etc.
                        Unorm.data = 1. / Unorm.data
                        poses_v = (Unorm_update * (Unorm * poses_v.flatten())).reshape(-1,9)
                        poses_s_in_cluster = [(Unorm_update * (Unorm * poses_s.flatten())).reshape(-1,9) for poses_s in poses_s_in_cluster]
                        poses_s_in_cluster_pre = [(Unorm_update * (Unorm * poses_s.flatten())).reshape(-1,9) for poses_s in poses_s_in_cluster_pre]
                        poses_in_cluster = [(Unorm_update * (Unorm * poses_u.flatten())).reshape(-1,9) for poses_u in poses_in_cluster]
                        best_poses_v = (Unorm_update * (Unorm * best_poses_v.flatten())).reshape(-1,9)

                        # avoid total chaos, adjust RNA buffer along.
                        if RNA_or_bfgs:
                            for pos in range(len(Gs)):
                                for ci in range(kClusters):
                                    Gs[pos][ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = Unorm_update * (Unorm * Gs[pos][ci * 9 * n_cameras: (ci+1) * 9 * n_cameras])
                                    Fes[pos][ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = Unorm_update * (Unorm * Fes[pos][ci * 9 * n_cameras: (ci+1) * 9 * n_cameras])
                                    Fs[pos][ci * 9 * n_cameras: (ci+1) * 9 * n_cameras] = Unorm_update * (Unorm * Fs[pos][ci * 9 * n_cameras: (ci+1) * 9 * n_cameras])
                        else:
                            for ci in range(kClusters):
                                prev_dk[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras]  = Unorm_update * (Unorm *  prev_dk[ci * 9 * n_cameras: (ci+1) * 9 * n_cameras])

                        Unorm = Unorm_update
                        Unorm = 1./Unorm.data.reshape(-1,9)

                    #print("poses_v ", poses_v)
                    if o3d_defined:
                        rerender(vis, camera_indices_in_cluster, point_indices_in_cluster, poses_in_cluster, landmarks, save_image=False)

                    if RNA_or_bfgs == False and ls_it != line_search_iterations-1:
                        #print("Reset counter after ", failedNesterovAcceleration , " failed acceleration steps")
                        failedNesterovAcceleration = 0 # success, reset counter

                    #print("A landmark_s_in_cluster", landmark_s_in_cluster)
                    break # next full iteration

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

if o3d_defined:
    vis, cameras_vis1, landmarks_vis = \
        render_points_cameras(camera_indices_in_cluster, point_indices_in_cluster, poses_v, landmarks)

if write_output:
    poses_v.tofile("camera_params_drs-lm.dat")
    landmarks.tofile("point_params_drs-lm.dat")

import json
result_dict = {"base_url": BASE_URL, "file_name": FILE_NAME, "iterations" : its, \
            "bestCost" : round(bestCost), "bestIt": bestIt, "kClusters" : kClusters, \
            "bestCost60" : round(bestCost60), "bestCost30" : round(bestCost30) }
with open('results_lm.json', 'a') as json_file:
    json.dump(result_dict, json_file)

# Another issue 646 occurs, likely in focal length vs z-coord or kappa?
# local optimization jumps big in 1 part. there is a huge gap parameter space from 1 part to the rest.
# profiler.print()
# profiler.open_in_browser()