from __future__ import print_function
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
#import open3d as o3d

# look at website. This is the smallest problem. guess: pytoch cpu is pure python?
BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
#FILE_NAME = "problem-73-11032-pre.txt.bz2"
# 59 / 2  ======== DRE BFGS ======  118872
FILE_NAME = "problem-138-19878-pre.txt.bz2"

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/dubrovnik/"
FILE_NAME = "problem-16-22106-pre.txt.bz2"
# large on1: 59 / 1  ======== DRE BFGS ======  9222786 FUCK, got stuck
# 36 / 0  ======== DRE BFGS ======  885868  ========= gain  973 with JtJ blockEig !
# 28 it. cost 0      841047 test_base
#FILE_NAME = "problem-356-226730-pre.txt.bz2" # large dub, play with ideas: cover, etc
#FILE_NAME = "problem-237-154414-pre.txt.bz2"
#59 / 0  ======== DRE BFGS ======  502663  ========= gain  122 ====
# 59 / 0  ======== DRE BFGS ======  499360  ========= gain  81 w 'pose pcg' at 500k much earlier.
FILE_NAME = "problem-173-111908-pre.txt.bz2"
#FILE_NAME = "problem-135-90642-pre.txt.bz2"

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/trafalgar/"
# FILE_NAME = "problem-21-11315-pre.txt.bz2"
# 59 / 0  ======== DRE BFGS ======  197984
FILE_NAME = "problem-257-65132-pre.txt.bz2"

# 52 / 2  ======== DRE BFGS ======  481560  ========= gain  175
# 52 / 2  ======== DRE BFGS ======  480851  ========= gain  3452 w pcg .. 'last turn' not clear
# 59 / 2  ======== DRE BFGS ======  475798  ========= gain  1601
BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/venice/"
FILE_NAME = "problem-52-64053-pre.txt.bz2"

# 59 / 0  ======== DRE BFGS ======  291177  ========= gain  938
#BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/final/"
#FILE_NAME = "problem-93-61203-pre.txt.bz2"


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
    #cameras_vis = []
    cameras_vis1 = []

    cam_loc = -AngleAxisRotatePoint(-from_numpy(cameras[:,0:3]), from_numpy(cameras[:,3:6])).numpy()
    #print(cam_loc.shape, " ", cam_loc)

    for ci in range(kClusters):

        alpha = (kClusters-1 - ci) / (kClusters-1)
        col = float_to_rgb(alpha)

        #cameras_vis.append(o3d.geometry.PointCloud())
        #cameras_ci = cameras[np.unique(camera_indices_in_cluster[ci]), 3:6].copy()
        cameras_ci = cam_loc[np.unique(camera_indices_in_cluster[ci]), :].copy()
        #cameras_vis[ci].points = o3d.utility.Vector3dVector(cameras_ci)

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
        #cameras_vis1[ci].paint_uniform_color(col) # make larger or what ?
        #cameras_vis[ci].paint_uniform_color(col) # make larger or what ?
        #geometry.points = o3d.utility.Vector3dVector(points_3d)
        #geometry_cam = o3d.geometry.PointCloud()
        #geometry_cam.points = o3d.utility.Vector3dVector(cameras[:,3:6])

        if ci ==0:
            vis.add_geometry(landmarks_vis[ci])
        else:
            vis.add_geometry(landmarks_vis[ci],  reset_bounding_box=False)
        vis.add_geometry(cameras_vis1[ci], reset_bounding_box=False)

        #vis.add_geometry(cameras_vis[ci],  reset_bounding_box=False)

        # matl = o3d.visualization.rendering.MaterialRecord()
        # matl.shader = 'defaultUnlit'
        # matl.point_size = 1.0
        # if ci ==0:
        #     pcs.append({'name': 'lm'+str(ci), 'geometry': landmarks_vis[ci], 'material': matl, 'reset_bounding_box':True})
        # else:
        #     pcs.append({'name': 'lm'+str(ci), 'geometry': landmarks_vis[ci], 'material': matl, 'reset_bounding_box':False})

        # mat = o3d.visualization.rendering.MaterialRecord()
        # mat.shader = 'defaultUnlit'
        # mat.point_size = 3.0
        # pcs.append({'name': 'pcd'+str(ci), 'geometry': cameras_vis[ci], 'material': mat, 'reset_bounding_box':True})

        #o3d.geometry.create_mesh_sphere(radius=10, resolution = 20)

    #o3d.visualization.draw(pcs, show_skybox=True)
    #o3d.visualization.Visualizer().get_view_control().set_zoom(0.5)
    #o3d.visualization.Visualizer().get_view_control().scale(0.2)

    vis.get_render_option().point_size = 2.0
    vis.run()
    return vis, cameras_vis1, landmarks_vis
    #o3d.visualization.draw_geometries([geometry])    # Visualize point cloud 
    save_image = False
    #exit()

def rerender(vis, camera_indices_in_cluster, point_indices_in_cluster, cameras, landmark_v, save_image=False):
    cam_loc = -AngleAxisRotatePoint(-from_numpy(cameras[:,0:3]), from_numpy(cameras[:,3:6])).numpy()
    for ci in range(kClusters):

        alpha = (kClusters-1 - ci) / (kClusters-1)
        col = float_to_rgb(alpha)

        #cameras_ci = cameras[np.unique(camera_indices_in_cluster[ci]), 3:6].copy()
        cameras_ci = cam_loc[np.unique(camera_indices_in_cluster[ci]), :].copy()

        #landmarks_vis.append(o3d.geometry.PointCloud())
        landmarks_ci = landmark_v[np.unique(point_indices_in_cluster[ci]),:]
        landmarks_vis[ci].points = o3d.utility.Vector3dVector(landmarks_ci)
        pc = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = np.repeat(np.array([i-1,j-1,k-1]), cameras_ci.shape[0]) * 0.5
                    pc.append(cameras_ci + a.copy().reshape(3, cameras_ci.shape[0]).transpose())
        #cameras_vis1.append(o3d.geometry.PointCloud())
        cameras_vis1[ci].points = o3d.utility.Vector3dVector(np.concatenate(pc))
        cameras_vis1[ci].paint_uniform_color(col) # make larger or what ?
        vis.update_geometry(cameras_vis1[ci])#, reset_bounding_box=False)

        landmarks_vis[ci].paint_uniform_color(col)
        #cameras_vis[ci].paint_uniform_color(col) # make larger or what ?
        #geometry.points = o3d.utility.Vector3dVector(points_3d)
        #geometry_cam = o3d.geometry.PointCloud()
        #geometry_cam.points = o3d.utility.Vector3dVector(cameras[:,3:6])

        vis.update_geometry(landmarks_vis[ci])#,  reset_bounding_box=False)
        #vis.update_geometry(cameras_vis[ci])#,  reset_bounding_box=False)

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

    res_toadd_to_c_ = fillPythonVec(res_toadd_out, res_toadd_sizes_out, kClusters__)
    point_indices_already_covered_ = fillPythonVec(point_indices_already_covered_out, point_indices_already_covered_sizes, kClusters__)
    covered_landmark_indices_c_ = fillPythonVec(covered_landmark_indices_c_out, covered_landmark_indices_c_sizes, kClusters__)
    num_res_per_c_ = fillPythonVecSimple(res_toadd_out)

    return res_toadd_to_c_, point_indices_already_covered_, covered_landmark_indices_c_, num_res_per_c_

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
    points_reprojX = points_projX * distortion * f
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
    distortion = 1. + r2 * (k1 + k2 * r2)
    points_reprojY = points_projY * distortion * f
    resY = (points_reprojY-p2d[:,1])
    return resY

def torchSingleResiduumScaled(camera_params_, point_params_, p2d, scalingP):
    point_params_ = point_params_ * scalingP
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
    resX = (points_reprojX - p2d[:, 0]).reshape((p2d.shape[0], 1))
    resY = (points_reprojY - p2d[:, 1]).reshape((p2d.shape[0], 1))
    residual = torch.cat([resX[:,], resY[:,]], dim=1)
    return residual

def torchSingleResiduumXScaled(camera_params, point_params, p2d, scalingP) :
    point_params = point_params * scalingP
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
    points_reprojX = points_projX * distortion * f
    resX = (points_reprojX-p2d[:,0])
    return resX

def torchSingleResiduumYScaled(camera_params, point_params, p2d, scalingP) :
    point_params = point_params * scalingP
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
        X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2) )
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


### could be removed / not used
def torchResiduum(x0T, n_cameras_, n_points_, camera_indices_, point_indices_, p2d_):
    # x0T = from_numpy(x0)
    # p2d = from_numpy(points_2d) # outside is better?
    # p2d.requires_grad_(False)
    camera_params = x0T[: n_cameras_ * 9].reshape(n_cameras_, 9)
    point_params = x0T[n_cameras_ * 9 :].reshape(n_points_, 3)
    angle_axis = camera_params[:, :3]

    # likely better to create per point/cam representation 1st. no slower
    # rot_matrix = AngleAxisToRotationMatrix(angle_axis) #
    # points_cam = rot_matrix[camera_indices_, :, 0] * point_params[point_indices_,0].view(-1,1) + rot_matrix[camera_indices_, :, 1] * point_params[point_indices_,1].view(-1,1) + rot_matrix[camera_indices_, :, 2] * point_params[point_indices_,2].view(-1,1)

    points_cam = AngleAxisRotatePoint(
        angle_axis[camera_indices_, :], point_params[point_indices_, :]
    )

    points_cam = points_cam + camera_params[camera_indices_, 3:6]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]

    f = camera_params[camera_indices_, 6]
    k1 = camera_params[camera_indices_, 7]
    k2 = camera_params[camera_indices_, 8]
    # f.requires_grad_(False) # not leaf?!
    # k1.requires_grad_(False)
    # k2.requires_grad_(False)
    r2 = points_projX * points_projX + points_projY * points_projY
    distortion = 1.0 + r2 * (k1 + k2 * r2)
    points_reprojX = -points_projX * distortion * f
    points_reprojY = -points_projY * distortion * f
    resX = (points_reprojX - p2d_[:, 0]).reshape((p2d_.shape[0], 1))
    resY = (points_reprojY - p2d_[:, 1]).reshape((p2d_.shape[0], 1))
    residual = torch.cat([resX[:,], resY[:,]], dim=1)
    return residual


def getJac(
    start_,
    end_,
    camera_indices_,
    point_indices_,
    camera_params_,
    point_params_,
    torch_points_2d_,
):
    end_ = min(end_, camera_indices.shape[0])
    n_cameras_ = camera_params_.shape[0]
    n_points_ = point_params_.shape[0]
    funx0_t = lambda X0: torchResiduum(
        X0,
        n_cameras_,
        n_points_,
        camera_indices_[start_:end_],
        point_indices_[start_:end_],
        torch_points_2d_[start_:end_],
    )
    x0_t_ = torch.hstack((camera_params_.flatten(), point_params_.flatten()))
    jac = jacobian(
        funx0_t, x0_t_, create_graph=False, vectorize=True, strategy="reverse-mode"
    )  # forward-mode
    print(start_, " ", end_, " : ")
    return jac


def buildMatrix(step, full, results_, camera_indices_, point_indices_, varset=0):
    data = []
    indptr = []
    indices = []
    if varset == 0:
        v_indices = camera_indices_
        sz = 9
    if varset == 1:
        v_indices = point_indices_
        sz = 3
    dataIndexSet = []

    for j in range(step):
        dataIndexSet.append(
            j * 2 * step * sz + 0 * sz * step + j * sz + np.arange(0, sz)
        )
    for j in range(step):
        dataIndexSet.append(
            j * 2 * step * sz + 1 * sz * step + j * sz + np.arange(0, sz)
        )
    dataIndices = np.concatenate(dataIndexSet)

    for i in np.arange(0, full, step):  # i, i + step: (0, 400), etc
        start_ = i
        end_ = min(i + step, v_indices.shape[0])
        # numLocalRows = results_[int(i/step)][varset].shape[0]
        # results_ hold res as resId, x/y, cam/point id, 9 or 3. only current cam if holds data!
        # print(i/step, " ", start, "-", end)
        # print(results_[int(i/step)][varset].shape) # 200,2,200,9 | 43,2,43,9
        # print(np.arange(start*sz, end*sz, sz).shape) # 200 | 43
        # print(np.array([sz * camera_indices[start:end] + j for j in range(sz)]).transpose().flatten().shape) # 1800 | 387

        # bottleneck? how to improve though? maybe 1 x one y and combine later.
        # can have index set?
        if end_ != i + step:  # else uber slow
            for j in range(end_ - start_):
                data.append(
                    results_[int(i / step)][varset][j, 0, j, :]
                    .flatten()
                    .detach()
                    .numpy()
                )
            for j in range(end_ - start_):
                data.append(
                    results_[int(i / step)][varset][j, 1, j, :]
                    .flatten()
                    .detach()
                    .numpy()
                )
                # print(data[-1].shape) # 18, so x/y x/y ... now first all x res then all y res.
        else:
            # print(i/step, " ", dataIndices.shape)
            data.append(
                results_[int(i / step)][varset].flatten()[dataIndices].detach().numpy()
            )
        # data is a list of ALL 9 arrays.
        # indptr and indices are not, last part is not of same size

        # assume data holds 1st all res for x then res for y, see below. Is this correct depends of result is indexed.
        # (end-start)*sz*2 elements (x/y).
        # in 2 parts -> 2(e-s)*sz/2 per part.
        # part starts in 0, 2*(e-s), 4*(e-s), .. = 2*start (start = k*(e-s))
        # 2*start*sz -
        # 2*s*sz -- 2(e)*sz in 2 parts ->
        # 2(e-s)*sz overall, 2(e-s)*sz/2 per part ->
        # 2(e-s)*sz/2 + 2*s*sz = (e-s)*sz + 2*s*sz = (e + s)*sz
        # so sz blocks of data pre res.
        indptr.append(np.arange(2 * start_ * sz, 2 * end_ * sz, sz).flatten())
        # indptr.append(np.arange((2*start+1)*sz, (sz*end+1)*sz, sz))
        # x&y: operate on same cam variables
        # first all x res, then all y res here.
        indices.append(
            np.array([sz * v_indices[start_:end_] + j for j in range(sz)])
            .transpose()
            .flatten()
        )
        indices.append(
            np.array([sz * v_indices[start_:end_] + j for j in range(sz)])
            .transpose()
            .flatten()
        )

        # n = 9 * n_cameras + 3 * n_points
        # m = 2 * points_2d.shape[0]
        # crs_pose = csr_array((np.array(data).flatten(), np.array(indices).flatten(), np.array(indptr).flatten()), shape=(2*full, 9 * n_cameras)).toarray()
    indptr.append(np.array([sz + indptr[-1][-1]]))  # closing
    if False:
        print("indptr ", np.concatenate(indptr))
        print("indptr len ", len(indptr))
        print("---------")
        print(
            "data ",
            np.concatenate(data).shape,
            " ",
            min(np.concatenate(data)),
            " ",
            max(np.concatenate(data)),
        )
        print(
            "indptr ",
            np.concatenate(indptr).shape,
            " ",
            min(np.concatenate(indptr)),
            " ",
            max(np.concatenate(indptr)),
        )
        print(
            "indices ",
            np.concatenate(indices).shape,
            " ",
            min(np.concatenate(indices)),
            " ",
            max(np.concatenate(indices)),
        )

    # toarray is slow and does what? nothing? makes a dense matrix .. OMG
    datavals = np.concatenate(data)
    # debug: set all inner parameters to 0
    if False and varset == 0:
        # datavals[0:end:9] = 0
        # datavals[1:end:9] = 0
        # datavals[2:end:9] = 0

        # datavals[3:end:9] = 0
        # datavals[4:end:9] = 0
        # datavals[5:end:9] = 0

        datavals[6:end:9] = 0
        datavals[7:end:9] = 0
        datavals[8:end:9] = 0
    crs_pose = csr_array((datavals, np.concatenate(indices), np.concatenate(indptr)))
    J_pose = csr_matrix(crs_pose)
    return J_pose


def buildResiduum(step, full, results_, varset=2):
    data = []
    for i in np.arange(0, full, step):  # i, i + step: (0, 400), etc
        # start = i
        # end = min(i + step, camera_indices_.shape[0])
        # results_ hold res as resId, x/y, cam/point id, 9 or 3. only current cam if holds data!
        # print(i/step, " ", start, "-", end)
        # print(results_[int(i/step)][varset].shape) # 200,2,200,9 | 43,2,43,9
        # print(np.arange(start*sz, end*sz, sz).shape) # 200 | 43
        # print(np.array([sz * camera_indices[start:end] + j for j in range(sz)]).transpose().flatten().shape) # 1800 | 387
        # print("res shape", results_[int(i/step)][varset][:,0].flatten().numpy().shape())
        # print("res ", results_[int(i/step)][varset][0].numpy())
        # for j in range(end-start):
        #    data.append(results_[int(i/step)][varset][0,:].flatten().numpy())
        # for j in range(end-start):
        #    data.append(results_[int(i/step)][varset][1,:].flatten().numpy())
        data.append(results_[int(i / step)][varset][:, 0].flatten().numpy())
        data.append(results_[int(i / step)][varset][:, 1].flatten().numpy())

    # print("---------")
    # print(np.concatenate(data).shape)
    res = np.concatenate(data)
    # print(res)
    return res


# fx0 + J delta x, J is Jp|Jl * xp|xl
def ComputeDerivativeMatrices(
    x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d
):
    start_ = time.time()
    step = 250
    full = camera_indices_.shape[0]
    # camera_params = x0_t[:n_cameras*9].reshape(n_cameras,9)
    # point_params  = x0_t[n_cameras*9:].reshape(n_points,3)
    # funx0_st1 = lambda X0, X1, X2: torchSingleResiduum(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2))
    # results = Parallel(n_jobs=8,prefer="threads")(delayed(getJac)(i, i + step) for i in np.arange(0, full, step))

    # parallel version
    # results = Parallel(n_jobs=8,prefer="threads")(delayed(getJacSin)(i, i + step, camera_indices_, point_indices_, x0_t_cam, x0_t_land, torch_points_2d) for i in np.arange(0, full, step))
    # not parallel
    results_ = []
    for i in np.arange(0, full, step):
        result = getJacSin(
            i,
            i + step,
            camera_indices_,
            point_indices_,
            x0_t_cam,
            x0_t_land,
            torch_points_2d,
        )
        # result = getJac(i, i + step, camera_indices_, point_indices_, x0_t_cam, x0_t_land, torch_points_2d) # awfully slow
        results_.append(result)

    end_ = time.time()
    print("Parallel ", len(results_), " ", step, " its take ", end_ - start_, "s")
    # print(results_[0][0].shape, " x ", results_[0][1].shape)
    # now compose sparse! jacobian
    # define sparsity pattern and add data
    # results_ hold res as resId, x/y, cam/point id, 9 or 3

    start = time.time()  # bottleneck now
    J_pose = buildMatrix(
        step, full, results_, camera_indices_, point_indices_, varset=0
    )
    end = time.time()
    # print(" build Matrix & residuum took ", end-start, "s")
    start = time.time()  # bottleneck now
    J_land = buildMatrix(
        step, full, results_, camera_indices_, point_indices_, varset=1
    )  # slow
    fx0 = buildResiduum(step, full, results_, varset=2)  # takes 1%
    end = time.time()
    # print(" build Matrix & residuum took ", end-start, "s")

    # print(J_pose.shape)
    return (J_pose, J_land, fx0)

def ComputeDerivativeMatricesNew(x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d, unique_landmarks_in_c_
):
    verbose = False
    if verbose:
        start = time.time() # this is not working at all. Slower then iteratively

    funx0_st1 = lambda X0, X1, X2: torchSingleResiduumX(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d fucntion -> grad possible
    funy0_st1 = lambda X0, X1, X2: torchSingleResiduumY(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d fucntion -> grad possible

    # TODO new, check
    landScale = 1./Vnorm.data.reshape(-1,3)
    landScale = landScale[unique_landmarks_in_c_]
    landScale = from_numpy(landScale[point_indices_[:]]) # here direct, or not?
    landScale.requires_grad_(False)

    funx0_st1 = lambda X0, X1, X2: torchSingleResiduumXScaled(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2), landScale)
    funy0_st1 = lambda X0, X1, X2: torchSingleResiduumYScaled(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2), landScale)

    torch_cams = x0_t_cam[camera_indices_[:],:] #x0_t[:n_cameras*9].reshape(n_cameras,9)[camera_indices[:],:]
    torch_lands = x0_t_land[point_indices_[:],:] #x0_t[n_cameras*9:].reshape(n_points,3)[point_indices[:],:]
    torch_lands.requires_grad_()
    torch_cams.requires_grad_()
    torch_cams.retain_grad()
    torch_lands.retain_grad()

    resX = funx0_st1(torch_cams, torch_lands, torch_points_2d[:,:]).flatten()
    lossX = torch.sum(resX)
    lossX.backward()
    # print(torch_cams.grad.shape)
    # print(torch_lands.grad.shape)
    #print(torch_cams.grad)

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
    # print("build res")
    # print(resX.flatten().numpy())
    # print(resY.flatten().numpy())
    res = np.concatenate(data)
    return res

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

# bs : blocksize, eg 9 -> 9x9 or 3 -> 3x3 per block
def blockInverse(M, bs):
    Mi = M.copy()
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            mat = Mi.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs)
            #mat = np.fliplr(mat)
            if not check_symmetric(mat):
                #print(i, " ", mat) # kind of flipped, so eigenval is crap.
                mat = np.fliplr(mat)
                imat = np.fliplr(inv_dense(mat)) # inv or pinv?
            else:
                # print(i, " ", mat)
                imat = inv_dense(mat) # inv or pinv?
            Mi.data[bs2 * i : bs2 * i + bs2] = imat.flatten()
    else:
        Mi = M.copy()
        for i in range(int(M.data.shape[0])):
            Mi.data[i : i + 1] = 1.0 / Mi.data[i : i + 1]
    return Mi

def blockEigenvalue(M, bs):
    Ei = np.zeros(M.shape[0])
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            mat = M.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs).copy()
            mat = np.fliplr(mat)
            if not check_symmetric(mat):
                mat = np.fliplr(mat)
            #print(i, " ", mat) # kind of flipped, so eigenval is crap.
            # [[ 4575.01 -1272.34  6458.94]
            #  [ 1029.28  8855.05 -1272.34]
            #  [ 4838.40  1029.28  4575.01]]
            evs = eigvalsh(mat)
            # if evs[0] <0:
            #     mat = np.fliplr(mat)
            #     evs = eigvalsh(mat)
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

def minmaxEv(M, bs):
    maxE = np.zeros(int(M.shape[0]/bs))
    minE = np.zeros(int(M.shape[0]/bs))
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            mat = M.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs).copy()
            if not check_symmetric(mat):
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

def copy_selected_blocks(M, block_selection_, bs):
    Mi = M.copy()
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            if block_selection_[i] == True:
                Mi.data[bs2 * i : bs2 * i + bs2] = 1e-15
    else:
        Mi = M.copy()
        for i in range(int(M.data.shape[0])):
            if block_selection_[i] == True:
                Mi.data[i : i + 1] = 1e-12
    return Mi

def stop_criterion(delta, delta_i, i):
    eps = 1e-2 #1e-2 used in paper, tune. might allow smaller as faster?
    #print(i+1, " ", delta_i, " ", delta)
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
        g = xk - Uli*(W*(Vli*(W.transpose() * xk))) + ubs
        yk = xk - 1/Lip * g
        xk = (1-gamma) * yk + gamma * y0
        y0 = yk

        if verbose:
            # test:
            # eq is Ul [I - Uli * W * Vli * W.transpose()] x = b
            costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
            print(it, " gd cost ", costk)

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

    res_toadd_to_c_, point_indices_already_covered_, covered_landmark_indices_c_, res_to_cluster_by_landmark_ = \
        process_cluster_lib(num_lands, num_res, kClusters, point_indices_in_cluster_, res_indices_in_cluster_, point_indices)
    
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
def average_landmarks_new(
    point_indices_in_cluster_, points_3d_in_cluster_, landmark_s_in_cluster_, L_in_cluster_, VL_in_cluster_, landmark_v_, delta_l_in_cluster
):
    num_points = points_3d_in_cluster_[0].shape[0]
    sum_Ds_2u = np.zeros(num_points * 3)
    sum_constant_term = 0
    V_cluster_zeros = []
    for i in range(len(L_in_cluster_)):
        # Lc = L_in_cluster_[i]
        point_indices_ = np.unique(point_indices_in_cluster_[i])
        # mean_points[point_indices_,:] = mean_points[point_indices_,:] + points_3d_in_cluster_[i][point_indices_,:] * Lc
        # num_clusters[point_indices_] = num_clusters[point_indices_] + Lc
        # fill Vl with 0s 3x3 blocks? somehow ..
        # sparse matrix is 0,3,6, ... data (0s of Vl data), 012,012,012,345,345,345, etc ?
        #
        # data can remain, indptr can remain, indices must be adjusted / new
        # point_indices_
        indices = np.repeat(
            np.array([3 * point_indices_ + j for j in range(3)]).transpose(), 3, axis=0
        ).flatten()
        # indices.append(np.array([3 * point_indices_ + j for j in range(3)]).transpose().flatten())
        # indptr is to be set to have empty lines by 0 3 3 -> no entries in row 3. 0:0-3, row 1:3-3

        indptr = [np.array([0])]
        j = 0
        for q in range(num_points):
            # print(q, " ", j, " ", point_indices_.shape[0], " ", np.array([9*j+3, 9*j+6, 9*j+9]) )
            if j < point_indices_.shape[0] and point_indices_[j] == q:
                indptr.append(np.array([9 * j + 3, 9 * j + 6, 9 * j + 9]).flatten())
                j = j + 1
            else:
                indptr.append(np.array([9 * j, 9 * j, 9 * j]).flatten())
        # print(indptr)
        indptr = np.concatenate(indptr)

        # print(indices)
        # print(VL_in_cluster_[i].indptr)
        # print(indptr)

        # print(indices.shape,  " ", VL_in_cluster_[i].data.shape, " ", VL_in_cluster_[i].indptr.shape, " ", indptr.shape, " ", 3*num_points, " ", 3 * np.max(point_indices_) )
        V_land = csr_matrix(
            (VL_in_cluster_[i].data, indices, indptr),
            shape=(3 * num_points, 3 * num_points),
        )
        V_cluster_zeros.append(V_land)
        # print(mean_points.shape, " " , V_land.shape, points_3d_in_cluster_[i].shape)
        # print cost after/before.
        # cost old v is where ? (v-2u+s)^T V_land (v-2u+s) = v^T V_land v + 2 v^T V_land (-2u+s) + (2u-s)^T V_land (2u-s)
        # derivative 2 V_land v + 2 V_land (-2u+s) = 0 <-> sum (V_land) v = sum (V_land (2u-s))

        # print(i, "averaging 3d ", points_3d_in_cluster_[i][globalSingleLandmarksB_in_c[i], :]) # indeed 1 changed rest is constant
        # print(i, "averaging vl ", V_land.data.reshape(-1,9)[globalSingleLandmarksA_in_c[i],:])  # indeed diagonal

        # TODO change 3, claim  2u+-s = 2 * (s+u)/2 - s  -  2 * (vli/2 .. ), so subtract u to get delta only
        u2_s = (2 * points_3d_in_cluster_[i].flatten() - landmark_s_in_cluster_[i].flatten())
        #u2_s = (2 * points_3d_in_cluster_[i].flatten() - landmark_s_in_cluster_[i].flatten()) - (points_3d_in_cluster_[i].flatten() - delta_l_in_cluster[i].flatten())

        sum_Ds_2u += V_land * u2_s # has 0's for those not present

        # print(i, "averaging u2_s ", u2_s.reshape(-1,3)[globalSingleLandmarksB_in_c[i], :]) # indeed 1 changed rest is constant

        #sum_constant_term += points_3d_in_cluster_[i].flatten().dot(V_land * (points_3d_in_cluster_[i].flatten() + u2_s - landmark_s_in_cluster_[i].flatten()))
        if i == 0:
            Vl_all = V_land
        else:
            Vl_all += V_land
    Vli_all = blockInverse(Vl_all, 3)
    # TODO change 1
    landmark_v_out = Vli_all * sum_Ds_2u
    #landmark_v_out = landmark_v_.flatten() + Vli_all * sum_Ds_2u
    verbose = False
    if verbose:
        cost_input  = 0.5 * (landmark_v_.flatten().dot(Vl_all * landmark_v_.flatten() - 2 * sum_Ds_2u) + sum_constant_term)
        cost_output = 0.5 * (landmark_v_out.dot(       Vl_all * landmark_v_out        - 2 * sum_Ds_2u) + sum_constant_term)

        #cost_simpler_out = landmark_v_out.dot(       Vl_all * landmark_v_out)        * 0.5 - landmark_v_out.dot(       sum_Ds_2u)
        #cost_simpler_in =  landmark_v_.flatten().dot(Vl_all * landmark_v_.flatten()) * 0.5 - landmark_v_.flatten().dot(sum_Ds_2u)
        print("========== update v: ", round(cost_input), " -> ", round(cost_output), " gain: ", round(cost_input - cost_output) )
        #print("======================== update v: ", round(cost_simpler_in), " -> ", round(cost_simpler_out), " gain: ", round(cost_simpler_in - cost_simpler_out) )

    return landmark_v_out.reshape(num_points, 3), Vl_all, V_cluster_zeros
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
#                     rho_k/2 {v^tv -  vT[2uk-sk] + uk^T[3uk-2sk]}
#                     rho_k/2 {v^tv - 2vT[2uk-sk] + uk^T[3uk-2sk]} # different
# 2nd try: 
#                     rho_k/2 <3u_k - v_k - 2s_k, u_k - v_k>
# SOO dumb. compute per part, help to select?! u or v
# insert v for uk -> R/2 (-v^tv + v^t sk + 3v^tv - 2 v^ts) = R/2 (2v^tv - v^ts)
# insert v=u above -> '0'
def cost_DRE(
    point_indices_in_cluster_, points_3d_in_cluster_, landmark_s_in_cluster_, L_in_cluster_, VL_in_cluster_, landmark_v_
):
    num_points = points_3d_in_cluster_[0].shape[0]
    sum_Ds_2u = np.zeros(num_points * 3)
    sum_constant_term = 0
    sum_u_s =0
    sum_u_v = 0
    sum_u_v_ = 0
    sum_2u_s_v = 0
    cost_dre = 0
    dre_per_part = []
    for i in range(len(L_in_cluster_)):
        point_indices_ = np.unique(point_indices_in_cluster_[i])
        indices = np.repeat(
            np.array([3 * point_indices_ + j for j in range(3)]).transpose(), 3, axis=0
        ).flatten()

        indptr = [np.array([0])]
        j = 0
        for q in range(num_points):
            if j < point_indices_.shape[0] and point_indices_[j] == q:
                indptr.append(np.array([9 * j + 3, 9 * j + 6, 9 * j + 9]).flatten())
                j = j + 1
            else:
                indptr.append(np.array([9 * j, 9 * j, 9 * j]).flatten())
        indptr = np.concatenate(indptr)

        V_land = csr_matrix(
            (VL_in_cluster_[i].data, indices, indptr),
            shape=(3 * num_points, 3 * num_points),
        )
        u2_s = (2 * points_3d_in_cluster_[i].flatten() - landmark_s_in_cluster_[i].flatten())
        #sum_Ds_2u += V_land * u2_s # has 0's for those not present
        #sum_constant_term += points_3d_in_cluster_[i].flatten().dot(V_land * (points_3d_in_cluster_[i].flatten() + u2_s - landmark_s_in_cluster_[i].flatten()))

        u_s = points_3d_in_cluster_[i].flatten() - landmark_s_in_cluster_[i].flatten()
        u_v = points_3d_in_cluster_[i].flatten() - landmark_v_.flatten()
        v_u2_s = u2_s - landmark_v_.flatten()
        sum_u_s += u_s.dot(V_land * u_s)
        sum_u_v += u_v.dot(V_land * u_v)
        sum_u_v_ += u_v.dot(u_v)
        sum_2u_s_v += v_u2_s.dot(V_land * v_u2_s)

        # rho_k/2 {vT[4uk-2sk-vk] + uk^T[3uk-2sk]}
        # rho_k/2 <3u_k - v_k - 2s_k, u_k - v_k>
        # same as my dre cost here.
        #local_cost = 0.5 * (u_v.dot(V_land * (u2_s + u_v - landmark_s_in_cluster_[i].flatten())))

        # different omg
        # rho_k/2 |u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k>
        # rho_k/2 |u_k - v_k|^2 - rho_k/2 <2(s_k - u_k), u_k - v_k>
        # rho_k/2 <u_k - v_k - 2(s_k - u_k), u_k - v_k>
        local_cost = 0.5 * u_v.dot(V_land * (u_v + 2 * u_s))
        #dre_per_part.append(local_cost.copy())

        # rho_k/2 |u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k>
        # rho_k/2 ( uu + vv - 2uv -2su + 2uu + 2sv - 2uv)
        # rho_k/2 ( 3uu + vv - 4uv - 2su + 2sv)
        # v only
        # rho_k/2 ( vv + v(2s-4u)   - 2su + 3uu)
        # deriv
        # rho_k (2v + 2s-4u) = 0

        # let rho_k = rho, the 2v = 4u-2s at min with cost 
        # r/2|u-2u+s|^2 - r<s-u, u-2u+s>
        # r/2|s-u|^2 - r|s-u, s-u|^2 = -r/2|s-u|^2 <0 Haeh? 

        #local_cost = min(0, local_cost.copy()) # well prevent chaos. sandwich lemma
        cost_dre += local_cost
        dre_per_part.append(local_cost) #.copy())

        if i == 0:
            Vl_all = V_land
        else:
            Vl_all += V_land

    # TODO: I use a different Vl to compute the cost here than in the update of prox u.
    #       Since I want to work with a new Vl already. Problem.
    # i want |u-s|_D |u-v|_D, also |v-2u-s|_D
    #cost_input  = 0.5 * (landmark_v_.flatten().dot(Vl_all * landmark_v_.flatten() - 2 * sum_Ds_2u) + sum_constant_term)
    
    print("---- |u-s|^2_D ", round(sum_u_s), "|u-v|^2_D ", round(sum_u_v), "|2u-s-v|^2_D ", round(sum_2u_s_v), 
          "|u-v|^2 ", round(sum_u_v_), " cost_dre ", cost_dre)
    # print("dre_per_part ",dre_per_part)
    #print(np.sum(np.array(dre_per_part)), " ", cost_input)
    return cost_dre, dre_per_part


def average_landmarks(point_indices_in_cluster_, points_3d_in_cluster_, L_in_cluster_):
    mean_points = np.zeros(points_3d_in_cluster_[0].shape)
    num_clusters = np.zeros(points_3d_in_cluster_[0].shape[0])
    kClusters_ = len(L_in_cluster_)
    for i in range(kClusters_):
        Lc = L_in_cluster_[i]
        point_indices_ = np.unique(point_indices_in_cluster_[i])
        mean_points[point_indices_, :] = (
            mean_points[point_indices_, :]
            + points_3d_in_cluster_[i][point_indices_, :] * Lc
        )
        num_clusters[point_indices_] = num_clusters[point_indices_] + Lc
    mean_points[:, 0] = mean_points[:, 0] / num_clusters
    mean_points[:, 1] = mean_points[:, 1] / num_clusters
    mean_points[:, 2] = mean_points[:, 2] / num_clusters
    return mean_points

# TODO: shorten
def primal_cost(
    x0_p_,
    camera_indices_in_cluster_,
    point_indices_in_cluster_,
    points_2d_in_cluster_,
    points_3d_in_cluster_,
):
    cameras_indices_in_c_ = np.unique(camera_indices_in_cluster_)
    cameras_in_c = x0_p_[cameras_indices_in_c_]
    torch_points_2d_in_c = from_numpy(points_2d_in_cluster_)
    torch_points_2d_in_c.requires_grad_(False)

    unique_points_in_c_ = np.unique(point_indices_in_cluster_)
    inverse_point_indices = -np.ones(np.max(unique_points_in_c_) + 1)  # all -1
    for i in range(unique_points_in_c_.shape[0]):
        inverse_point_indices[unique_points_in_c_[i]] = i

    point_indices_in_c = point_indices_in_cluster_.copy()
    for i in range(point_indices_in_cluster_.shape[0]):
        point_indices_in_c[i] = inverse_point_indices[point_indices_in_c[i]]

    min_cam_index_in_c = np.min(camera_indices_in_cluster_)
    points_3d_in_c = points_3d_in_cluster_[unique_points_in_c_]

    #camera_indices_ = camera_indices_in_cluster_ - min_cam_index_in_c
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

    landScale = 1. / Vnorm.data.reshape(-1,3)
    landScale = from_numpy(landScale[unique_points_in_c_])
    #landScale = from_numpy(landScale)
    #print("landScale ", landScale)
    landScale.requires_grad_(False)
    funx0_st1 = lambda X0, X1, X2: \
        torchSingleResiduumScaled(X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2), landScale[point_indices_[:]])

    fx1 = funx0_st1(
        x0_t_cam[camera_indices_[:]],
        x0_t_land[point_indices_[:]],
        torch_points_2d)
    costEnd = np.sum(fx1.numpy() ** 2)
    return costEnd

def bundle_adjust(
    camera_indices_,
    point_indices_,
    landmarks_only_in_cluster_,
    torch_points_2d,
    cameras_in,
    points_3d_in,
    landmark_s_, # taylor expand at point_3d_in -> prox on landmark_s_ - points_3d = lambda (multiplier)
    unique_points_in_c_,
    Vl_in_c_,
    L_in_cluster_,
    LipJ, # start with 1.0. externally increase if dre increases
    blockLip, 
    successfull_its_=1,
):
    # print("landmarks_only_in_cluster_  ", landmarks_only_in_cluster_, " ", np.sum(landmarks_only_in_cluster_), " vs ", np.sum(1 - landmarks_only_in_cluster_) )
    newForUnique = True # debug nable_approx 
    #blockEigMult = 1e-3 # 1e-3 was used before, solid, less hicups smaller 1e-3. recall last val .. 1e-4 explodes for 173 example.
    blockEigMult = 1e-3 # maybe sub-problem singular? for landmarks? how to test?
    # define x0_t, x0_p, x0_l, L # todo: missing Lb: inner L for bundle, Lc: to fix duplicates
    minimumL = 1e-6
    JJ_mult = 1 # 1/2: affects problem 52. 477K vs 499k or so, is 2 needed for something -- maybe 10 clusters?
    L = max(minimumL, L_in_cluster_)
    updateJacobian = True
    # holds all! landmarks, only use fraction likely no matter not present in cams anyway.
    x0_l_ = points_3d_in.flatten()
    s_l_ = landmark_s_.flatten()

    #x0_l_ = 0.5*(x0_l_ + s_l_) # TODO EXPERIMENT, join TR and DL, does it help? A bit worse? performance, chokes in a different manner. init s becomes too far from 'u' solution at some point.

    # holds all cameras, only use fraction, camera_indices_ can be adjusted - min index
    x0_p_ = cameras_in.flatten()
    x0 = np.hstack((x0_p_, x0_l_))
    x0_t = from_numpy(x0)
    # torch_points_2d = from_numpy(points_2d)
    n_cameras_ = int(x0_p_.shape[0] / 9)
    n_points_ = int(x0_l_.shape[0] / 3)
    powerits = 100 # kind of any value works here? > =5?
    tr_eta_1 = 0.8
    tr_eta_2 = 0.25

    it_ = 0
    funx0_st1 = lambda X0, X1, X2: \
        torchSingleResiduum(X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2))

    landScale = 1./Vnorm.data.reshape(-1,3)
    landScale = landScale[unique_points_in_c_]
    landScale = from_numpy(landScale[point_indices_[:]]) # here direct?
    landScale.requires_grad_(False)
    funx0_st1 = lambda X0, X1, X2: \
        torchSingleResiduumScaled(X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2), landScale)

    #stepSize = diag_sparse(np.zeros(n_points_))
    # make diagonal again.
    if issparse(Vl_in_c_):
        stepSize = diag_sparse(Vl_in_c_.diagonal())

    steSizeTouched = False
    newVersion = True
    newForUnique = True
    if newVersion:
        blockEigMult = 1e-3
    # works with delivering updated J's, s.t. averagin and NEXT update use same matrix.
    blockLip = 1e-3 # fixed is better? I do not get it. 1e-4  @ 173 fails -- why ? with post grad: ok -- BUT 1e-4 is worse?

    while it_ < successfull_its_:
        # to torch here ?

        if updateJacobian:  # not needed if rejected
            x0_t_cam = x0_t[: n_cameras_ * 9].reshape(n_cameras_, 9) # not needed?
            x0_t_land = x0_t[n_cameras_ * 9 :].reshape(n_points_, 3) # another idea, start at s, see below, unifies TR and Descent Lemma check. See above replace u_l with sl
            J_pose, J_land, fx0 = ComputeDerivativeMatricesNew(
                x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d, unique_points_in_c_
            )

            JtJ = J_pose.transpose() * J_pose
            # TODO: flip if needed. likely not needed. since  adding a flipped and a non flipped matrix should deliver a non flipped one. This might lead to issues in every 'block' method
            #JtJ = flipIfNeeded(JtJ)

            # in test_base
            #JtJDiag = diag_sparse(JtJ.diagonal())

            #J_eps = 1e-3 # does this matter? 1e-3 2 / 0  ======== DRE BFGS ======  861672  ========= gain  486744 ==== f(v)=  735282  f(u)=  971304  ~=  971304.0525993105
            # test CSU: 2a^2+b^2 > (a+b)^2 = a^2 + b^2 + 2ab. Since (a-b)^2 = a^2 + b^2 - 2ab > 0, so a^2 + b^2 > 2ab. 
            # a^2 = p^t* Jp^TJp * p , b^2 = l^tJl^TJl l. ab = p^tJp^T Jl*l.
            #(Jl | Jp) (l,p)^T = Jl l + Jp p and |(Jl | Jp) (l,p)^T|^2 = l^t Jl^t Jl l + p^t Jp^t Jp p + 2 p^t Jp^t Jl l.
            # So 2 JtJ  + 2 JltJl shuold majorize |J^t x|^2 for all x.
            # why relevant here.
            #JtJDiag = JtJ + J_eps * diag_sparse(np.ones(JtJ.shape[0])) # fails if JtJ is too large or small, example dubrovnik, 356 cams
            blockEigenvalueJtJ = blockEigenvalue(JtJ, 9)
            JtJDiag = JtJ + 1e-9 * blockEigenvalueJtJ # alot worse at 1e6, bit better 1e-9, but hmm 59 / 0  ======== DRE BFGS ======  501565

            maxE, minE = minmaxEv(JtJDiag, 9)
            print("min max ev JtJDiag ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spectral ", np.max(maxE/minE), file=sys.stderr )

            #print(" min/max JtJ.diagonal() ", np.min(JtJ.diagonal()), " ", np.max(JtJ.diagonal()), " adjusted ", np.min(JtJDiag.diagonal()), " ", np.max(JtJDiag.diagonal()), " ", np.min(blockEigenvalueJtJ.diagonal()) ," ", np.max(blockEigenvalueJtJ.diagonal()) )

            #JtJDiag = diag_sparse(np.fmax(JtJ.diagonal(), 1e1)) # sensible not clear maybe lower.
            JltJl = J_land.transpose() * J_land
            absDiagJltJl = np.abs(JltJl.diagonal()).reshape(-1,3)
            print( "Diag pseudo HessL (max/min/med)",  np.max(absDiagJltJl, axis=0), " ", np.min(absDiagJltJl, axis=0),  " ", np.median(absDiagJltJl, axis=0))

            blockEigenvalueJltJl = blockEigenvalue(JltJl, 3)
            maxE, minE = minmaxEv(JltJl, 3)
            print("min max ev JltJl ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spectral ", np.max(maxE/minE), file=sys.stderr)
            # if not issparse(Vl_in_c_) and it_ < 1:
            #     stepSize = blockEigenvalueJltJl
            # else: # increase where needed -- this here is WAY too slow?
            #     stepSize.data = np.maximum(0.05 * stepSize.data, blockEigenvalueJltJl.data) # else diagSparse of it
            
            # '2 *' smallest example dies with < 2 ok with 2.
            # 'hangs at this value.' Check new variant with recomputing Jacobian at end.
            #68 / 0  ======== DRE BFGS ======  501664  ========= gain  247 ==== f(v)=  502614  f(u)=  500821  ~=  500704.09297090676
            #81 / 0  ======== DRE BFGS ======  501076  ========= gain  139 ==== f(v)=  502339  f(u)=  499758  ~=  499713.50253538677
            stepSize = blockEigMult * blockEigenvalueJltJl + JJ_mult * JltJl.copy() # Todo '2 *' vs 1 by convex.
            maxE, minE = minmaxEv(stepSize, 3)
            print("min max ev stepSize ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spectral ", np.max(maxE/minE), file=sys.stderr )
            #stepSize = 1. * (blockEigMult * blockEigenvalueJltJl + LipJ * JltJl.copy()) # hmm
            
            #stepSize = 1. * (1e-0 * diag_sparse(np.ones(n_points_*3)) + 1.0 * JltJl.copy()) # not so good
            #stepSize = 1. * (blockEigMult * LipJ * blockEigenvalueJltJl + JltJl.copy()) # test use Lip here. did not work?

            maxE, minE = minmaxEv(JltJl, 3)
            JltJlSpec = [round_int(np.max(maxE/minE)), np.min(maxE/minE), round_int(np.median(maxE/minE))]
            maxE, minE = minmaxEv(stepSize, 3)
            JltJlDiagSpec = [round_int(np.max(maxE/minE)), np.min(maxE/minE), round_int(np.median(maxE/minE))]
            print("minmax ev JltJlD ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spec ", JltJlSpec, " -> ", JltJlDiagSpec)

            # problem blockinverse assume data is set in full 3x3
            # Problem: gets stuck. guess. a new, very large value for a coordinate hits a previously low one.
            # averaging leaves a gap here, if gap is too large we fail.
            # should still be caverable by (pre!)-computing deriv in advance. Then jump is limited by inc in L.
            #JltJlDiag = diag_sparse(np.fmax(JltJl.diagonal(), 1e1)) # startL defines this. 
            #maxDiag = np.max(JltJl.diagonal())
            # here larger x, eg 5: 1ex leasd  to stallment, small to v,u divergence.
            #JltJlDiag = diag_sparse(np.fmax(JltJl.diagonal(), np.minimum(maxDiag, 1e5 * np.maximum(1., 1./L)))) # startL defines this. 
            # theoretic value is 2 * diag. [modula change to diag]. Here we use 1/2 diag. but No dependence on L.
            #JltJlDiag = np.maximum(1, 0.5 / L) * diag_sparse(np.fmax(JltJl.diagonal(), 1e1)) # this should ensure (small L) condition.
            # improvement: only for constrained variables.
            # improvement: acceleration, maybe needs ALL variables to be effective.

            JltJlDiag = stepSize.copy() # max 1, 1/L, line-search dre fails -> increase

            print(" min/max JltJl.diagonal() ", np.min(JltJl.diagonal()), " ", np.max(JltJl.diagonal()), " adjusted ", np.min(JltJlDiag.diagonal()), " ", np.max(JltJlDiag.diagonal()), file=sys.stderr)
            #print("JltJlDiag.shape ", JltJlDiag.shape, JltJlDiag.shape[0]/3)

            # set to 1e-12+L * blockEigMult * blockEigenvalueJltJl all entries of completely covered lms.
            if newForUnique:
                JltJlDiag = copy_selected_blocks(JltJlDiag, landmarks_only_in_cluster_, 3)
                JltJlDiag = JltJlDiag + L * blockEigMult * blockEigenvalueJltJl
 
                #JltJlDiag = JltJlDiag + L * diag_sparse(JltJl.diagonal()) # original
                #JltJlDiag = 1/L * JltJlDiag + diag_sparse(JltJl.diagonal()) #diag_sparse(np.fmax(JltJl.diagonal(), 1e0))

            # theory states roughly JltJlDiag * max(1,1/L) should be used. This does not converge u,v are coupled but no progress is made on primal.
            # should use descent lemma to define this 'L' not TR? pose L can be anything
            # Above to slow with 1/L.
            # maybe use Dl/L instead, Dl=1 init, check descent lemma and lower Dl if possible (increase if not)
            # but Dl is fulfilled with L. So would lower Dl just like, or? 
            # maybe better: 3x3 matrix sqrt(|M|_1 |M|inf) as diag. Yet this removes effect of 'L' getting small = large steps.
            # do i need to keep memory to ensure it remains >? or pre compute grad (and store)?

            # based on unique_pointindices_ make diag_1L that is 1 at points 
            # only present within this part.

            # 0.2 was working
            JltJlDiag = 1/L * JltJlDiag # max 1, 1/L, line-search dre fails -> increase

            W = J_pose.transpose() * J_land
            bp = J_pose.transpose() * fx0
            bl = J_land.transpose() * fx0

            prox_rhs = x0_l_ - s_l_
            # remove prox term for covered landmarks
            if newForUnique: # alternative turn off completely, use 2u-s -> return (u-s)/2 to average u+k = uk + delta uk
                landmarks_in_many_cluster_ = np.invert(landmarks_only_in_cluster_)
                diag_present = diag_sparse( np.repeat((np.ones(n_points_) * landmarks_in_many_cluster_).reshape(-1,1), 3).flatten() )
                prox_rhs = diag_present * prox_rhs

            costStart = np.sum(fx0**2)
            penaltyStartConst = prox_rhs.dot(JltJlDiag * prox_rhs)

            # idea stepSize is EXTRA on prox part and 'fixed'. unless DL fails we need to add something. use blockEigenvalueJltJl as before? or increase JJ_mult?
            if newVersion:
                #stepSize = JJ_mult * JltJl.copy() + blockEigMult * blockEigenvalueJltJl
                stepSize = JJ_mult * JltJl.copy() + blockLip * blockEigenvalueJltJl
                JltJlDiag = JltJl.copy() + 1e-6 * blockEigMult * blockEigenvalueJltJl
                maxE, minE = minmaxEv(stepSize, 3)
                print("min max ev stepSize ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spectral ", np.max(maxE/minE), file=sys.stderr)
                maxE, minE = minmaxEv(JltJlDiag, 3)
                print("min max ev JltJlDiag ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spectral ", np.max(maxE/minE), file=sys.stderr)
                if newForUnique:
                    stepSize = copy_selected_blocks(stepSize, landmarks_only_in_cluster_, 3)
                penaltyStartConst = prox_rhs.dot(stepSize * prox_rhs)

        # start_ = time.time()
        Vl = JltJl + L * JltJlDiag
        Ul = JtJ + L * JtJDiag
        penaltyStart = L * penaltyStartConst

        if newVersion:
            Vl = JltJl + L * JltJlDiag + stepSize
            penaltyStart = penaltyStartConst
            #maxE, minE = minmaxEv(Vl, 3)
            #print("min max ev Vl ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spectral ", np.max(maxE/minE))

        # cost added is + L * (delta_v - s_l_ + x0_l_)^T  JltJlDiag * (delta_v - s_l_ + x0_l_)
        # + L * (delta_v)^T  JltJlDiag * (delta_v) + 2 L * (delta_v^T JltJlDiag * (x0_l_ - s_l_) + L * (s_l_ - x0_l_)^T  JltJlDiag * (s_l_ - x0_l_)
        # derivative
        # L * 2 * JltJlDiag * (delta_v) + 2 L * JltJlDiag * (x0_l_ - s_l_) = 0
        # added cost is, 2 L * (delta_v^T JltJlDiag * (x0_l_ - s_l_) + L * (s_l_ - x0_l_)^T  JltJlDiag * (s_l_ - x0_l_)

        Vli = blockInverse(Vl, 3)
        maxE, minE = minmaxEv(Vli, 3)
        #print("min max ev Vli ", np.max(maxE), " ", np.max(minE), " ", np.min(maxE), " ",  np.min(minE), " spectral ", np.max(maxE/minE))

        bl_s = bl + L * JltJlDiag * prox_rhs # TODO: + or -. '+', see above
        if newVersion:
            bl_s = bl + stepSize * prox_rhs # TODO: + or -. '+', see above
        bS = (bp - W * Vli * bl_s).flatten()

        #delta_p = -solvePowerIts(Ul, W, Vli, bS, powerits)
        delta_p, powerits_run = solveByGDNesterov(Ul, W, Vli, bS, powerits)
        delta_p = -delta_p
        delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl_s)
        penaltyL = L * (delta_l + prox_rhs).dot(JltJlDiag * (delta_l + prox_rhs))
        penaltyP = L * delta_p.dot(JtJDiag * delta_p)
        if newVersion:
            penaltyL = L * delta_l.dot(JltJlDiag * delta_l) + (delta_l + prox_rhs).dot(stepSize * (delta_l + prox_rhs))

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

        fx1 = funx0_st1(x0_t_cam[camera_indices_[:]], x0_t_land[point_indices_[:]], torch_points_2d)
        costEnd = np.sum(fx1.numpy() ** 2)
        print(it_, "it. cost 1     ", round(costEnd), "      + penalty ", round(costEnd + penaltyL + penaltyP),)

        tr_check = (costStart + penaltyStart - costEnd - penaltyL) / (costStart + penaltyStart - costQuad - penaltyL)
        if newVersion:
            tr_check = (costStart + penaltyStart - costEnd - penaltyL - penaltyP) / (costStart + penaltyStart - costQuad - penaltyL - penaltyP)

        old_descent_lemma = True # appears not to bring value with 4x
        if old_descent_lemma:
            # old descent lemma test.
            #nablaXp = L * JtJDiag * delta_p  # actual gradient. discussable
            nablaXp = JtJDiag * delta_p  # actual gradient. discussable TODO
            nablaXl = L * JltJlDiag * delta_l  # actual gradient
            # before
            # LfkDiagonal = \
            #     2 * (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l)) \
            #     / (delta_l.dot(nablaXl) + delta_p.dot(nablaXp))

            Lfklin = (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l))
            if newVersion:
                #nablaXp = L * JtJDiag * delta_p # TODO: right?
                nablaXl = JltJlDiag * delta_l # TODO: or this?
                Lfklin = Lfklin + (delta_l + prox_rhs).dot(stepSize * (delta_l + prox_rhs)) - penaltyStartConst

            LfkDistance  = Lfklin - (delta_l.dot(nablaXl) + delta_p.dot(nablaXp)) / 2
            LfkViolated = LfkDistance > 0
            LfkSafe = Lfklin < 0 # for any phi ok.
        else:
            # after:
            # f(x) <= f(y) + <nabla(f(y) x-y> + Lf/2 |x-y|^2
            # we demand stepsize phi >= 2 Lf. Then even
            # f(x) <= f(y) + <nabla(f(y) x-y> + phi/4 |x-y|^2
            # (f(x) - f(y) - <nabla(f(y) x-y>) * 4 / |x-y|^2  <= phi
            nablaXp = JtJDiag * delta_p    # actual gradient: J^t fx0 = bp|bl , no L. grad at f(x) is L * JtJDiag * delta_p - s
            nablaXl = L * JltJlDiag * delta_l  # actual gradient is b. L since JltJlDiag multiplied by 1/L. 
            descent_rhs_l = L * delta_l.dot(JltJlDiag * delta_l)
            descent_rhs_p = delta_p.dot(JtJDiag * delta_p)

            LfkDistance  = costEnd - (costStart + bp.dot(delta_p) + bl.dot(delta_l) + (descent_rhs_l + descent_rhs_p) / 4)
            LfkViolated = LfkDistance > 0
            LfkSafe = (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l)) < 0 # for any phi ok.

        # TODO: this does not work: problem-356, this also explains why J_mult *2 is ok here. 
        # Does not solve the 356 problem as one subproblem gets stcuk with huge L
        # tr only affects cams, binds cam to 'u'
        # descent -lem only affects landmarks, binds lms to s
        # tr vioation could be either lms OR cams or both.
        # unlcear what to do
        # 1. do either, now i block descent lemma update if tr violation (why?)
        # 2. if tr violation increase eith, not only lms. problem s could be wrong
        # 3. introduce 'L' * for lms to ensure tr does something there see 2.
        # 4. something smart. eval all and take best decision ?
        # for 3: so our cost is prox f(s) instead. Easy.

        if tr_check < tr_eta_2:
            print(" //////  tr_check " , tr_check, " Lfk distance ", LfkDistance, " -nabla^Tdelta=" , -bp.dot(delta_p) - bl.dot(delta_l), " /////", file=sys.stderr)
            L = L * 2
            if not newVersion:
                JltJlDiag = 1/2 * JltJlDiag

        # todo: store last blockEigenvalueJltJl multiplier in/out current L used but not really.
        # option 1: does not work, now subproblem gets 's' forcing much worse solutions.
        # this is still fishy. turn off totally if cost improves -- or consider to utilize blockEigMult everywhere. 
        # I return stepsize now with blockEigMult baked in  -- so why not kepp in memory and lower even is possible?
        # 
        #
        # MAYBE DL is even wrong, we could treat as function of landmarks only.
        if False: # so fixed 1e-3 is better. Hence why bother at all?
        #if tr_check >= tr_eta_2 and LfkViolated and not steSizeTouched or (steSizeTouched and costStart + penaltyStart < costEnd + penaltyL): # violated -- should revert update.
        #if tr_check >= tr_eta_2 and (LfkViolated and costStart + penaltyStart < costEnd + penaltyL): # violated -- should revert update.
        #if LfkViolated and not steSizeTouched or (steSizeTouched and costStart + penaltyStart < costEnd + penaltyL): # violated -- should revert update.
            steSizeTouched = True
            #stepSize = stepSize * 2
            # other idea, initially we only add 1/2^k eg 0.125, times the needed value and inc if necessary, maybe do not add anything if not needed.

            if not newVersion:
                stepSize += blockEigMult * blockEigenvalueJltJl
                blockEigMult *= 2
            else:
                stepSize += blockLip * blockEigenvalueJltJl
                blockLip *= 2
            print(" |||||||  Lfk distance ", LfkDistance, " -nabla^Tdelta=" , -bp.dot(delta_p) - bl.dot(delta_l), " LipJ ", LipJ, " blockLip ", blockLip, " |||||||", file=sys.stderr)

            #blockEigenvalueJltJl.data *= 2 # appears slow but safe
            #stepSize.data = np.maximum(stepSize.data, blockEigenvalueJltJl.data) # else diagSparse of it

            if not newVersion:
                JltJlDiag = 1/L * stepSize.copy()
                penaltyStartConst = prox_rhs.dot(JltJlDiag * prox_rhs)
            else:
                penaltyStartConst = prox_rhs.dot(stepSize * prox_rhs)
        else:
            LfkViolated = False # hack, also above , or (steSizeTouched and costStart + penaltyStart < costEnd + penaltyL) is hack
            # replace by following: track cost, if next has lower cost -> continue. if next has higher cost return current.

        #if LfkSafe and not steSizeTouched:
        if (newVersion and tr_check >= tr_eta_1) or (not newVersion and LfkSafe and not steSizeTouched):
            L = L / 2
            if not newVersion:
                JltJlDiag = 2 * JltJlDiag # we return this maybe -- of course stupid to do in a release version

        if (newVersion and LfkSafe and not steSizeTouched):
            blockLip = blockLip / 2
            print(" _________  Lfk distance ", LfkDistance, " -nabla^Tdelta=" , -bp.dot(delta_p) - bl.dot(delta_l), " LipJ ", LipJ, " blockLip ", blockLip, " _________", file=sys.stderr)

        # if LfkDiagonal < -2 and steSizeTouched:
        #     LfkDiagonal = -2
        # if LfkDiagonal < -2 and not steSizeTouched: # can we increase convergence in those cases? Problme if this fluctuates of course. Maybe only do if above did not enter
        #     steSizeTouched = True
        #     print(" |||||||  Lfk estimate ", LfkDiagonal, " -nabla^Tdelta=" , -bp.dot(delta_p) - bl.dot(delta_l), " |||||||")
        #     stepSize.data = stepSize.data / 1.5
        #     JltJlDiag = 1/L * stepSize.copy()
        #     penaltyStartConst = (x0_l_ - delta_l - s_l_).dot(JltJlDiag * (x0_l_ - delta_l - s_l_))

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

        # Descent lemma print # descent lemma for landmarks only?
        # f(x) <= f(y) + <nabla f(y), x-y> + Lf/2 |x-y|^2
        # e.g. update from y to x, nabla f(y) = Jac * res(y) = b
        # x=y+delta, f(x) <= f(y) + <nabla f(y), x-y> + Lf/2 |x-y|^2
        #       <=>  f(x) - f(y) <= <nabla f(y), delta> + Lf/2 |delta|^2
        #       <=> [f(x) - f(y) - b^T delta] / delta^2 <= Lf/2, defines Lf
        # 2 * (costEnd - costStart - b^t (delta) / |delta|^2)
        # ok delta ~ - nabla -> b^Tdelta > 0. Even delta = - JtJ^-1 b.
        # Hence - b^T JtJ^-1 b. Since J^tJ>=0 it follows b^Tdelta < 0. Ok see below.
        # Note that nabla f = b and delta = - J^tJ ^-2 b -> b = JtJ delta
        # Hence f(x) <= f(y) + delta^T [l/2 D^tD + J^tJ] delta. for all x=delta+y.
        # Also J^tJ = K^tK + q * D^tD with D = diag(K) and q by TR.
        # So (ignore 1/2) (l-q) D^TD > J^TJ. Maybe set l = 2q, where q is our TR choice.
        # At least should not diverge
        if False:
            Lfk = \
                2 * (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l)) \
                / (delta_l.dot(delta_l) + delta_p.dot(delta_p))
            # Not the same we have diag in there: Lf/2 delta^t D delta.
            # so divide by this to receive update for Lf. This is given below.
            # Do i need the same for only landmarks/dupe lms?
            LfkX = \
                2 * (costStart - costEnd - nablaXp.dot(delta_p) - nablaXl.dot(delta_l)) \
                / (delta_l.dot(delta_l) + delta_p.dot(delta_p))
        # TODO descent lemma with L * JltJlDiag only for landmarks.
        # test joint.
        # We use diagonal matrix L * JtJDiag, JltJlDiag instead of scalar L:
        # we now get -- if accepted update < 2 always
        # nablaXp = L * JtJDiag * delta_p  # actual gradient
        # nablaXl = L * JltJlDiag * delta_l  # actual gradient
        # LfkDiagonal = \
        #     2 * (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l)) \
        #     / (delta_l.dot( nablaXl) + delta_p.dot( nablaXp ))
        # this becomes x: new, y: old
        # descent lemma is now, y + delta = x, delta = x-y
        # f(x) <= f(y) + <nabla(f(y) x-y> + Lf/2 nablaf(x)^T (x-y)
        # f(x) <= f(y) + <nabla(f(y) + Lf/2 nablaf(x)^T, x-y> must hold.
        # find Lf s.t. this holds. 
        #print("Lfk estimate ", Lfk, " and ", LfkX, " or ", LfkDiagonal)
        # relation to check f(x) > f(y) + penaltyL to accept step is:
        # f(x) > f(y) + delta^t Diag delta = f(y) + <nabla f(x), x-y>
        # this implies descent lemma ALWAYS fulfilled.

        print(" ------- Lfk distance ", LfkDistance, " tr_check ", tr_check, " blockLip ", blockLip, " -------- ", file=sys.stderr)
        # update TR -- 
        #if costStart + penaltyStart > costEnd + penaltyL
        #tr_check = (costStart - costEnd) / (costStart - costQuad)

        # TODO: unclear if this does nay bad/good.?
        if False and it_ < successfull_its_ and L > 1e-6: # lowering leads to, see below averaging affected, can trigger multiple increases
            #print( "A JltJlDiag-bun ", JltJlDiag.data.reshape(-1,9)[landmarks_only_in_cluster_,:])
            if tr_check > tr_eta_1: # and L > 0.1: # needs a limit else with my L * Vl, or 1/L in diag?
                L = L / 2
                JltJlDiag = 2 * JltJlDiag
            if tr_check < tr_eta_2 or LfkDiagonal > 2: # tr check becomes descent lemma, might need > 1?
                L = L * 2
                JltJlDiag = 1/2 * JltJlDiag

        if LfkViolated: # violated -- should revert update.
            print("=========================== SHOULD NOT ENTER ==========================")
            stepSize = stepSize * 2
            JltJlDiag = 1/L * stepSize.copy()

    x0_p_ = x0_p_.reshape(n_cameras_, 9)
    x0_l_ = x0_l_.reshape(n_points_, 3)

    # 173 with more stable, still dre can increase later many more of 'accept' first line search.
    # still not much faster.
    # both gradient-approx appear very close to each other so maybe not needed.
    getBetterStepSize = False # this is used as approx of f in update of v and thus s. maybe change there u-v should be small. 
    if getBetterStepSize: # needs to set L correctly
        J_pose, J_land, fx0 = ComputeDerivativeMatricesNew(
            x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d, unique_points_in_c_
        )
        # this is nabla, compare to estimation old (u-s)^T Jdiag (u-s)
        #bp = J_pose.transpose() * fx0
        bl = J_land.transpose() * fx0

        JltJl = J_land.transpose() * J_land
        # #JltJlDiag = diag_sparse(np.fmax(JltJl.diagonal(), 1e1)) # should be same as above
        # #JltJlDiag = np.maximum(1, 0.5 / L) * diag_sparse(np.fmax(JltJl.diagonal(), 1e1)) # this should ensure (small L) condition.
        # #stepSize = blockEigenvalue(JltJl, 3)
        # stepSize.data = np.maximum(stepSize.data, blockEigenvalue(JltJl, 3).data) # else diagSparse of it

        nabla_l_approx = JltJlDiag * (delta_l + prox_rhs)

        # fairly good. x2 tested. x1 ?
        #stepSize = 1. * (blockEigMult * blockEigenvalueJltJl + LipJ * JltJl.copy()) # hmm
        stepSize = 1. * (blockEigMult * blockEigenvalueJltJl + JJ_mult * JltJl.copy())
        #stepSize = 1. * (blockEigMult * blockEigenvalueJltJl +     JltJl.copy()) # obviously the same as above, right.
        JltJlDiag = 1/L * stepSize.copy() # max 1, 1/L, line-search dre fails -> increase

        # ok maybe yes. I want to 'set those to a small value' only.
        if newForUnique: # does this make sense at all? just clear and manipulate s->u, s.t. v = 2u-s = u. new s = u (old), will set to new later.
            JltJlDiag = copy_selected_blocks(JltJlDiag, landmarks_only_in_cluster_, 3)
            JltJlDiag = JltJlDiag + L * blockEigMult * blockEigenvalueJltJl

        nabla_l_approx2 = JltJlDiag * (delta_l + prox_rhs)

        if newForUnique:
            nabla_l_approx2 = diag_present * nabla_l_approx2
            bl = diag_present * bl

        diff_to_nabla_l2_2 = np.linalg.norm(L*nabla_l_approx2+bl, 2)
        diff_to_nabla_l2   = np.linalg.norm(L*nabla_l_approx+bl, 2)
        #diff_to_nabla_p1 = np.linalg.norm(nablaXp-bp, 2)
        #diff_to_nabla_p2 = np.linalg.norm(nablaXp+bp, 2)
        diff_to_nabla_l4_2 = np.linalg.norm(2*L*nabla_l_approx2+bl, 2)
        diff_to_nabla_l4   = np.linalg.norm(2*L*nabla_l_approx+bl, 2)

        print("diff_to_nabla_l *2 ", diff_to_nabla_l4, " | ", "diff_to_nabla_l ", diff_to_nabla_l2, " |")
        print("diff_to_nabla2_l *2 ", diff_to_nabla_l4_2, " | ", "diff_to_nabla2_l ", diff_to_nabla_l2_2, " |")

        print(" nablas 1", - L * nabla_l_approx ) # So nabla_l_approx = JtJDiag * (u-s), hence return (i use s-u), 2 * JtJDiag * L, the 2 DELIVERS a better cost!
        print(" nablas 2", - L * nabla_l_approx2) # So nabla_l_approx = JtJDiag * (u-s), hence return (i use s-u), JtJDiag * L, the 2 DELIVERS a better cost!
        print(" nablas b", bl) # TINY


        # Also ok, but slower. Haeh? numerically unstable. at some point dre estimate wrong -> collapse.
        # JltJlDiag = 2/L * JltJl.copy() # could use as precomputed

    if True:
        J_pose, J_land, fx0 = ComputeDerivativeMatricesNew(
            x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d, unique_points_in_c_
        )
        JltJl = J_land.transpose() * J_land
        # ok maybe yes. I want to 'set those to a small value' only.
        blockEigenvalueJltJl = blockEigenvalue(JltJl, 3)
        stepSize = JJ_mult * JltJl.copy() + blockLip * blockEigenvalueJltJl

        if newVersion:
            if newForUnique:
                stepSize = copy_selected_blocks(stepSize, landmarks_only_in_cluster_, 3)
        else:
            JltJlDiag = 1/L * stepSize.copy() # max 1, 1/L, line-search dre fails -> increase
            if newForUnique: # does this make sense at all? just clear and manipulate s->u, s.t. v = 2u-s = u. new s = u (old), will set to new later.
                JltJlDiag = copy_selected_blocks(JltJlDiag, landmarks_only_in_cluster_, 3)
                JltJlDiag = JltJlDiag + L * blockEigMult * blockEigenvalueJltJl


    # in the averaging step we do 
    # min_v 1/2 sum_k (v - (2uk-sk)) Vlk (v - (2uk-sk)) with solution (sum_k(Vlk))^-1 * sum_k Vlk (2uk-sk)
    # <=> min_v 1/2 sum_k v^T Vlk v - v^T sum_k Vlk (2uk-sk) + const
    # {1} 
    #    
    # the delta_l update here (and even without! paralellism) is
    # bl = J_land.transpose() * fx0
    # bl_s = bl + L * JltJlDiag * (x0_l_ - s_l_) <=>
    # bl_s = bl +           Vl  * (x0_l_ - s_l_)
    # delta_l = - Vli * ((W.transpose() * delta_p).flatten() + bl_s)
    # solves what. then add xo_l to it (delta_v -> v)
    # recall x0_l + delta_l = 'v'
    # 1/2 delta_l^T Vl delta_l + delta_l^T [ (W.transpose() * delta_p) + bl_S]
    # insert x0_l + delta_l = v, recall x0_l is constant/known
    # 1/2 v^T Vl v + v^T [ (W.transpose() * delta_p) + bl_S] + 
    # 1/2 x0_l^T Vl x0_l + v^T Vl x0_l + x0_l^T [ (W.transpose() * delta_p) + bl_S]
    # remove constant parts
    # 1/2 v^T Vl v + v^T [ (W.transpose() * delta_p) + bl_S + Vl x0_l]
    # {2}
    # 1/2 v^T Vl v + v^T [ (W.transpose() * delta_p) + bl] + v^T Vl * (2 * x0_l_ - s_l_)] 
    # WEIRD: this is the same as {1} but + v^t b and +! v^T Vl * (2 * x0_l_ - s_l_)] not '-'!
    # Again. Correct u in non-split BA would be to sum over k for {2} and solve it. !Wo x-s term!
    # back to {1}
    # in {1} we have u+ for uk, so u+ = u + delta. we can insert 
    # min_v 1/2 sum_k v^T Vlk v - v^T sum_k Vlk (2(x0+delta) - sk)
    # 
    # So idea would be do {2} style updates wo. any s anywhere.
    # interpret as stochastic BA with updates on local parts. 
    # update for l done with sarah style updates.
    # how does this turn out for local updates. I do! need some adjustment from non considered parts.
    # actually the last gradient for these. Last + something.
    # 
    # delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl_s)
    # Compute insert solution here here we use Vli = (2*Vl)^-1
    # u+ = u - Vli/2 [(W.transpose() * delta_p).flatten() + bl_s]
    # u+ = u - Vli/2 [(W.transpose() * delta_p).flatten() + bl + Vl * (u - s)]
    # u+ = u - Vli/2 [(W.transpose() * delta_p).flatten() + bl] - 1/2(u - s)
    # u+ = (s+u)/2 - Vli/2 [(W.transpose() * delta_p).flatten() + bl] or 
    # u+ = u - (x-s)/2 - Vli/2 [(W.transpose() * delta_p).flatten() + bl] with x=u or v
    # 
    # v is prox on Vl/2 |v- 2u+ + s |^2 = 
    # 2u+-s = 2 * (s+u)/2 - s - 2 * (vli/2 .. )
    # 2u+-s = u - Vli [(W.transpose() * delta_p).flatten() + bl]
    #       = u + delta original for each k.
    # 
    # averaging over this is
    # (sum_k Vlk)^-1 { sum_k Vlk uk - [(W.transpose() * delta_p).flatten() + bl]k }
    # (sum_k Vlk)^-1 { sum_k Vlk uk} + real delta
    # why not v + real data instead of this mix of u's?
    # simulation: replace 2u+-s by 2u+ - s -u + v
    # or 2u+ - s - u to get delta. then add to v. or use real uk instead.
    # TODO! 

    # My tries to use Vl + eps * diag instead of 2Vl has issues, see above
    # we average over 2u-s not u, so, if we the add to the rhs Vl + eps * diag * (u-s) 
    # as L * JltJlDiag * (x0_l_ - s_l_) and invert with JltJl + L * JltJlDiag
    # with 0s in l we do not want to constrain we must return ..
    # shorter: we use 2u+ - s to compute average, return u + delta (desired solution)
    # we get 2u+-s as new solution. 
    # So we must return (u + delta + s)/2 to get 2 * (u + delta + s)/2 - s = u + delta as new v.
    # 
    # compute  we need to return   
    # Compute insert solution here here we use Vli = (2*Vl)^-1
    # u+ = u - Vli/2 [(W.transpose() * delta_p).flatten() + bl_s]
    # u+ = u - Vli/2 [(W.transpose() * delta_p).flatten() + bl + Vl * (u - s)]

    # see diff_to_nabla_l3 = np.linalg.norm(2*L*nabla_l_approx-bl, 2) is smallest. only for computing DRE.
    Rho = L * JltJlDiag #+ 1e-12 * Vl
    if newVersion:
        Rho = stepSize #+ 1e-12 * Vl

    # TODO change 2
    # It appears as if we set entries to 1 for covered landmarks in Rho (those were set to almost 0)
    # and hack the outpout xl to ensure 'averaging' will later operate on v = 2s-u
    # here these have 1 on diagonal and i return u := s, then v = 2s-s = s. This leads to a cost of 0 in dre.
    # but how do i get true u in there? heah? Better reset s.
    # that is not smart since we also reuse those xl later (now wrong) and also compute dre from it (wrong now)
    if newForUnique:
        #print( "B JltJlDiag-bun ", JltJlDiag.data.reshape(-1,9)[landmarks_only_in_cluster_,:])
        #landmarks_in_many_cluster_ = np.invert(landmarks_only_in_cluster_)
        #print("landmarks_only_in_cluster_  ", landmarks_only_in_cluster_, " ", np.sum(landmarks_only_in_cluster_), " vs ", np.sum(1 - landmarks_only_in_cluster_) )
        #print("landmarks_in_many_cluster_  ", landmarks_in_many_cluster_, " ", np.sum(landmarks_in_many_cluster_), " vs ", np.sum(1 - landmarks_in_many_cluster_) )
        # print(np.repeat((np.ones(n_points_) * landmarks_in_many_cluster_)[..., np.newaxis], 3, axis=1).shape)
        # print(np.repeat((np.ones(n_points_) * landmarks_in_many_cluster_)[..., np.newaxis], 3, axis=1))
        #print("1. ", (np.ones(n_points_) * landmarks_in_many_cluster_))
        #print("2. ", np.repeat((np.ones(n_points_) * landmarks_in_many_cluster_).reshape(1,-1), 3))
        diag_present = 0.5 * diag_sparse( np.repeat((np.ones(n_points_) * landmarks_only_in_cluster_).reshape(-1,1), 3).flatten() )
        #diag_present = copy_selected_blocks(0.5 * diag_sparse(np.ones(n_points_*3)), landmarks_in_many_cluster_, 1)
        #print("diag_present ", diag_present.shape, " " , np.sum(diag_present.data==0.5), " " , np.sum(diag_present.data!=0.5) )
        #print( "Y ", (L * JltJlDiag).data.reshape(-1,9)[landmarks_only_in_cluster_,:])

        # see above, compensate for averaging
        if False:
            xTest = x0_l_ - (diag_present*(x0_l_.flatten() - s_l_)).reshape(-1,3)
        else:
            xTest = x0_l_ # settting s to u after prox

        # print("xt ", xTest[landmarks_only_in_cluster_,:])
        # print("x0 ", x0_l_[landmarks_only_in_cluster_,:])

        #test = diag_present + L * JltJlDiag + 1e-12 * Vl
        #print( "C vl-bun ", test.data.reshape(-1,9)[landmarks_only_in_cluster_,:]) # is diag mat off line basically 0

        # do for all, remove rhs completely
        # diag_presentB = 0.5 * diag_sparse( np.repeat((np.ones(n_points_)).reshape(-1,1), 3).flatten() )
        # xTest = x0_l_ - (diag_presentB*(x0_l_.flatten() - s_l_)).reshape(-1,3)

        return costEnd, x0_p_, xTest, L, diag_present + Rho, delta_l.reshape(n_points_, 3), blockLip

    L_out = np.maximum(minimumL, np.minimum(L_in_cluster_ * 2, L)) # not clear if generally ok, or 2 or 4 should be used.
    return costEnd, x0_p_, x0_l_, L_out, Rho, delta_l.reshape(n_points_, 3), blockLip

    # recall solution wo. splitting is 
    # solve Vl x + bS + Vd () 
    #bl_s = bl + L * JltJlDiag * (x0_l_ - s_l_) # TODO: + or -. '+', see above
    #delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl_s).flatten()
    # sum_k delta_l Vlk delta_l + delta_l ((Wk.transpose() * delta_p) + bl_sk) argmin
    # all this is local per block. k blocks: sum_k Vlk = 3x3, we could instead 
    # return Vlk and bk = ((Wk.transpose() * delta_p) + bl_sk): 4 times #landmarks floats.
    # and compute the update, summing each and solving v = (sumk vlk)^-1 (sum_k bk).
    # would be cool if we could do n iterations locally -- with a gain.
    # much less cams then lms. do the inverse?
    # send only parts each time?single lms = worst?
    # i need to send v anyway and lms as well. above is 4x more.
    # this shows one core can do all this in parallel / local network can -> problem is parallelizable trivially
    # large network, better send minimal information. problem still send landmarks.
    # and diagonal? would be good if can work n steps locally.
    # what is missing? maybe accumulate 'Vl' on the way or upper bound
    # as max eigenvalue per landmark 3x3, or just sum row/col -> blockEigen, and 

def updateCluster(
    x0_p_,
    camera_indices_in_cluster_,
    point_indices_in_cluster_,
    points_2d_in_cluster_,
    points_3d_in_cluster_,
    landmark_s_in_cluster_,
    Vl_in_cluster_,
    L_in_cluster_,
    landmark_occurences,
    LipJ,
    blockLip_c,
    its_,
):
    cameras_indices_in_c_ = np.unique(camera_indices_in_cluster_)
    cameras_in_c = x0_p_[cameras_indices_in_c_]
    #local_camera_indices_in_cluster = camera_indices_in_cluster_ - min_cam_index_in_c
    # alternative 
    local_camera_indices_in_cluster = np.zeros(camera_indices_in_cluster_.shape[0], dtype=int)
    for i in range(cameras_indices_in_c_.shape[0]):
        local_camera_indices_in_cluster[camera_indices_in_cluster_ == cameras_indices_in_c_[i]] = i

    # torch_points_2d_in_c = torch_points_2d[points_2d_in_cluster[ci], :]
    torch_points_2d_in_c = from_numpy(points_2d_in_cluster_)
    torch_points_2d_in_c.requires_grad_(False)

    # take point_indices_in_cluster[ci] unique:
    unique_points_in_c_ = np.unique(point_indices_in_cluster_)
    # unique_points_in_c_[i] -> i, map each pi : point_indices_in_cluster[ci] to position in unique_points_in_c_[i]
    inverse_point_indices = -np.ones(np.max(unique_points_in_c_) + 1)  # all -1
    for i in range(unique_points_in_c_.shape[0]):
        inverse_point_indices[unique_points_in_c_[i]] = i

    landmarks_only_in_cluster_ = landmark_occurences[unique_points_in_c_] == 1
    #print("Unique landmarks  ", landmark_occurences, " ", landmark_occurences.shape, " ", np.min(landmark_occurences), " ", np.max(landmark_occurences))
    #print("Unique landmarks  ", landmarks_only_in_cluster_, " ", np.sum(landmarks_only_in_cluster_), " vs ", np.sum(1 - landmarks_only_in_cluster_) )

    point_indices_in_c = (
        point_indices_in_cluster_.copy()
    )  # np.zeros(point_indices_in_cluster_.shape)
    for i in range(point_indices_in_cluster_.shape[0]):
        point_indices_in_c[i] = inverse_point_indices[point_indices_in_c[i]]

    # put in unique points, adjust point_indices_in_cluster[ci] by id in unique_points_in_c_
    min_cam_index_in_c = np.min(camera_indices_in_cluster_)
    points_3d_in_c = points_3d_in_cluster_[unique_points_in_c_]
    landmark_s_in_c = landmark_s_in_cluster_[unique_points_in_c_]

    cost_, x0_p_c_, x0_l_c_, Lnew_c_, Vl_c_, delta_l_c_, blockLip_c_ = bundle_adjust(
        local_camera_indices_in_cluster,
        point_indices_in_c,
        landmarks_only_in_cluster_, # input those lms not present anywhere else to relax hold on those.
        torch_points_2d_in_c,
        cameras_in_c,
        points_3d_in_c,
        landmark_s_in_c,
        unique_points_in_c_,
        Vl_in_cluster_,
        L_in_cluster_,
        LipJ,
        blockLip_c,
        its_,
    )

    return (
        cost_,
        x0_p_c_,
        x0_l_c_,
        Lnew_c_,
        Vl_c_,
        unique_points_in_c_,
        cameras_indices_in_c_,
        delta_l_c_,
        blockLip_c_
    )

def prox_f(x0_p_, camera_indices_in_cluster_, point_indices_in_cluster_, 
           points_2d_in_cluster_, points_3d_in_cluster_, landmark_s_in_cluster_, 
           L_in_cluster_, Vl_in_cluster_, LipJ, blockLip_in_cluster_, kClusters, innerIts, sequential) :
    cost_ = np.zeros(kClusters)
    delta_l_in_cluster_ = [0 for elem in range(kClusters)]

    num_landmarks = points_3d_in_cluster_[0].shape[0]
    landmark_occurences = np.zeros(num_landmarks)
    for ci in range(kClusters):
        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        landmark_occurences[unique_points_in_c_] +=1

    # for ci in range(kClusters):
    #     print(ci, " 3d " ,points_3d_in_cluster_[ci][landmark_occurences==1, :])

    if sequential:
        for ci in range(kClusters):

            delta_l_in_cluster_[ci] = np.zeros(points_3d_in_cluster_[ci].shape)

            (
                cost_c_,
                x0_p_c_,
                x0_l_c_,
                Lnew_c_,
                Vl_c_,
                unique_points_in_c_,
                cameras_indices_in_c_,
                delta_l_c_,
                blockLip_c_
            ) = updateCluster(
                x0_p_,
                camera_indices_in_cluster_[ci],
                point_indices_in_cluster_[ci],
                points_2d_in_cluster_[ci],
                points_3d_in_cluster_[ci],
                landmark_s_in_cluster_[ci],
                Vl_in_cluster_[ci],
                L_in_cluster_[ci],
                landmark_occurences,
                LipJ[ci],
                blockLip_in_cluster_[ci],
                its_=innerIts,
            )
            cost_[ci] = cost_c_
            L_in_cluster_[ci] = Lnew_c_
            Vl_in_cluster_[ci] = Vl_c_
            points_3d_in_cluster_[ci][unique_points_in_c_, :] = x0_l_c_
            x0_p_[cameras_indices_in_c_] = x0_p_c_
            delta_l_in_cluster_[ci][unique_points_in_c_, :] = delta_l_c_
            blockLip_in_cluster_[ci] = blockLip_c_
    else:
        # not not ,prefer="threads" ?
        # results = Parallel(n_jobs=8,prefer="threads")(delayed(getJacSin)(i, i + step, camera_indices, point_indices, x0_t_cam, x0_t_land, torch_points_2d) for i in np.arange(0, full, step))
        results = Parallel(n_jobs=3)(
            delayed(updateCluster)(
                x0_p_,
                camera_indices_in_cluster_[ci],
                point_indices_in_cluster_[ci],
                points_2d_in_cluster_[ci],
                points_3d_in_cluster_[ci],
                landmark_s_in_cluster_[ci],
                Vl_in_cluster_[ci],
                L_in_cluster_[ci],
                landmark_occurences,
                LipJ[ci],
                its_=innerIts,
            )
            for ci in range(kClusters)
        )
        for ci in range(kClusters):
            cost_[ci] = results[ci][0]
            L_in_cluster_[ci] = results[ci][3]
            Vl_in_cluster_[ci] = results[ci][4]
            points_3d_in_cluster_[ci][results[ci][5], :] = results[ci][2]
            x0_p_[results[ci][6]] = results[ci][1]

    for ci in range(kClusters):
        vl = Vl_in_cluster_[ci]
        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        landmarks_only_in_cluster_ = landmark_occurences[unique_points_in_c_] == 1
        globalSingleLandmarksA_in_c[ci] = landmarks_only_in_cluster_.copy()
        globalSingleLandmarksB_in_c[ci] = landmark_occurences==1

        # print(ci, " 3d ", points_3d_in_cluster_[ci][landmark_occurences==1, :]) # indeed 1 changed rest is constant
        # print(ci, " vl ", vl.data.reshape(-1,9)[landmarks_only_in_cluster_,:])  # indeed diagonal
    return (cost_, L_in_cluster_, Vl_in_cluster_, points_3d_in_cluster_, x0_p_, delta_l_in_cluster_, blockLip_in_cluster_, globalSingleLandmarksA_in_c, globalSingleLandmarksB_in_c)

# fill lists G and F, with g and f = g - old g, sets of size m, 
# at position it % m, c^t compute F^tF c + lamda (c - 1/k)^2, sum c=1
# g = x0, f = delta. Actullay xnew = xold + delta.
def RNA(G, F, g, f, it_, m_, Fe, fe, res_pcg, lamda=0.05, h=-1):
    #lamda = 0.05 # reasonable 0.01-0.1
    crefVersion = True #False
    #lamda = 0.05 # cref version needs larger 
    #h = -1 #-0.1 # 2 / (L+mu) -- should 1/diag * F^t F * c
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

    #FtF = Fs_.transpose().dot(Fs_) # why dot?
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
    extrapolation = Gs_.dot(c) #+ 0.1 * Fs_.dot(c)
    extrapolationF = Fes_.dot(c)

    print("c ", c, " ", c.shape, id_)

    # shape utter non sense
    #print("extrapolation ", extrapolation.shape, " ", g.shape)
    return (G, F, Fe, np.squeeze(extrapolation - h * extrapolationF))


def BFGS_direction(r, ps, qs, rhos, k, mem, mu, U_diag):
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

    #dk_ = mu * r
    dk_ = U_diag * r

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

write_output = False #True
read_output =  False
if read_output:
    # camera_params_np = np.fromfile("camera_params_drs.dat", dtype=np.float64)
    # point_params_np = np.fromfile("point_params_drs.dat", dtype=np.float64)
    camera_params_np = np.fromfile("camera_params_base.dat", dtype=np.float64)
    point_params_np = np.fromfile("point_params_base.dat", dtype=np.float64)

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

c02_mult = 1
c34_mult = 1
c5_mult = 1
c6_mult = 1
c7_mult = 1
c8_mult = 1
if True:
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

############################################
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

# could also compute locally / all the time! 542: appears to 'go crazy' after 20 its.
J_pose, J_land, fx0 = ComputeDerivativeMatrixInit(cameras, points_3d, points_2d, camera_indices, point_indices)
JltJl = J_land.transpose() * J_land
Vnorm = diag_sparse(np.squeeze(np.asarray( ( (np.abs(JltJl)/100).sum(axis=0) ))))
temp = Vnorm.data.reshape(-1,3)
temp = np.sqrt(temp)
Vnorm = diag_sparse(temp.flatten())
#Vnorm = 100 * diag_sparse(np.ones(points_3d.flatten().shape[0])) # 52: eval
points_3d = (Vnorm * points_3d.flatten()).reshape(-1,3)
############################################

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
kClusters = 10 #5
innerIts = 1  # change to get an update, not 1 iteration
its = 60
cost = np.zeros(kClusters)
bestCost = np.sum(fx0**2) #1e20
lastCost = bestCost
lastCostDRE = bestCost
basic_version = False #True # accelerated or basic
newForUnique = True
sequential = True
# 173 true is better ?! maybe cluster?
# might be violating lipshitz condition from last Jacobian if disabled.
# acceleration does not work anymore if false! aha.
linearize_at_last_solution = True # linearize at uk or v. maybe best to check energy. at u or v. DRE:
lib = ctypes.CDLL("./libprocess_clusters.so")
init_lib()

# for part k: given Vk, uk and v,  
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
# local cost is fv^2 + (v-s) Vl (v-s) or fu^2 + (u-s) Vl (u-s)

globalSingleLandmarksA_in_c = [0 for x in range(kClusters)]
globalSingleLandmarksB_in_c = [0 for x in range(kClusters)]

values, counts = np.unique(camera_indices, return_counts=True)
minCount = np.min(counts)
print(". minimum camera observations in total ", minCount, " cams with < 5 landmarks ", np.sum(counts < 5))

if False:
    (
        camera_indices_in_cluster,
        point_indices_in_cluster,
        points_2d_in_cluster,
        cluster_to_camera,
        points_3d_in_cluster,
        L_in_cluster,
    ) = cluster_by_camera(
        camera_indices, points_3d, points_2d, point_indices, kClusters, startL
    )
else:
    (
        camera_indices_in_cluster,
        point_indices_in_cluster,
        points_2d_in_cluster,
        cluster_to_camera,
        points_3d_in_cluster,
        L_in_cluster,
        kClusters,
    ) = cluster_by_camera_smarter(
        camera_indices, points_3d, points_2d, point_indices, kClusters, startL, init_cam_id=0, init_lm_id=0
    )

for ci in range(kClusters):
    values, counts = np.unique(camera_indices_in_cluster[ci], return_counts=True)
    minCount = np.min(counts)
    print(ci, ". minimum camera observations in cluster ", minCount, " cams with < 5 landmarks ", np.sum(counts < 5))

print(L_in_cluster)
Vl_in_cluster = [0 for x in range(kClusters)] # dummy fill list
landmark_s_in_cluster = [elem.copy() for elem in points_3d_in_cluster]
landmark_v = points_3d_in_cluster[0].copy()
LipJ = np.ones(kClusters)
blockLip_in_cluster = 1e-3 * np.ones(kClusters)

primal_cost_v = 0
for ci in range(kClusters):
    primal_cost_v += primal_cost(
        x0_p,
        camera_indices_in_cluster[ci],
        point_indices_in_cluster[ci],
        points_2d_in_cluster[ci],
        landmark_v)
print("DEBUG scaled cost ", primal_cost_v)
#exit()

o3d_defined = False
if o3d_defined:
    vis, cameras_vis1, landmarks_vis = render_points_cameras(camera_indices_in_cluster, point_indices_in_cluster, cameras, landmark_v)

if basic_version:

    for it in range(its):
        start = time.time()
        (
            cost,
            L_in_cluster,
            Vl_in_cluster,
            points_3d_in_cluster,
            x0_p,
            delta_l_in_cluster,
            blockLip_in_cluster,
            globalSingleLandmarksA_in_c,
            globalSingleLandmarksB_in_c
        ) = prox_f(
            x0_p, camera_indices_in_cluster, point_indices_in_cluster, points_2d_in_cluster, 
            points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, 
            LipJ, blockLip_in_cluster, kClusters, innerIts=innerIts, sequential=True,
            )
        end = time.time()

        if newForUnique:
            for ci in range(kClusters):
                landmark_s_in_cluster[ci][globalSingleLandmarksB_in_c[ci]] = points_3d_in_cluster[ci][globalSingleLandmarksB_in_c[ci]]

        #print("++++++++++++++++++ globalSingleLandmarksB_in_c[0].shape ", globalSingleLandmarksB_in_c[0].shape)

        currentCost = np.sum(cost)
        print(it, " ", round(currentCost), " gain ", round(lastCost - currentCost), ". ============= sum fk update takes ", end - start," s",)

        landmark_v, _, _ = average_landmarks_new(
            point_indices_in_cluster, points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, landmark_v, delta_l_in_cluster
        )

        #DRE cost BEFORE s update
        #dre, dre_per_part = cost_DRE(point_indices_in_cluster, points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, landmark_v)

        tau = 1 # 2 is best ? does not generalize?!
        for ci in range(kClusters):
            landmark_s_in_cluster[ci] = landmark_s_in_cluster[ci] + tau * (landmark_v - points_3d_in_cluster[ci]) # update s = s + v - u.

        #DRE cost AFTER s update
        dre, dre_per_part = cost_DRE(point_indices_in_cluster, points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, landmark_v)
        dre += currentCost

        primal_cost_us = [0 for _ in range(kClusters)]
        primal_cost_vs = [0 for _ in range(kClusters)]
        for ci in range(kClusters):
            primal_cost_vs[ci] = primal_cost(
                x0_p,
                camera_indices_in_cluster[ci],
                point_indices_in_cluster[ci],
                points_2d_in_cluster[ci],
                landmark_v) #points_3d_in_cluster[ci]) # v not u
        primal_cost_u = 0
        for ci in range(kClusters):
            primal_cost_us[ci] = primal_cost(
                x0_p,
                camera_indices_in_cluster[ci],
                point_indices_in_cluster[ci],
                points_2d_in_cluster[ci],
                points_3d_in_cluster[ci]) # v not u
        primal_cost_v = np.sum(np.array(primal_cost_vs))
        primal_cost_u = np.sum(np.array(primal_cost_us))

        dre = max( primal_cost_v, dre ) # sandwich lemma, prevent maybe chaos

        print( it, " ======== DRE ====== ", round(dre) , " ========= gain " , \
            round(lastCostDRE - dre), "==== f(v)= ", round(primal_cost_v), " f(u)= ", round(primal_cost_u))

        if lastCostDRE < dre:
            LipJ += 0.2 * np.ones(kClusters)

        lastCost = currentCost
        # print(" output shapes ", x0_p_c.shape, " ", x0_l_c.shape, " takes ", end-start , " s")
        if False and lastCostDRE - dre < 1:
            break
        lastCostDRE = dre

        # fill variables for update: linearize at u or v.
        for ci in range(kClusters):
            # point_indices = np.unique(point_indices_in_cluster[ci])
            # print ("output.shape ", output.shape, " ", points_3d_in_cluster[ci].shape, " ", point_indices_in_cluster[ci].shape, " ", point_indices.shape )
            # points_3d_in_cluster[ci][point_indices,:] = output[point_indices,:]
            if not linearize_at_last_solution: # linearize at v / average solution, same issue I suppose. Yes. solution is too return the new gradient, s.t. update of v is wrt to current situation.
                # points_3d_in_cluster[ci]  = landmark_v.copy() # init at v, above at u
                #compare cost of part and dre_per_part to pick best option
                if primal_cost_vs[ci] < primal_cost_us[ci] + dre_per_part[ci]:
                    print(primal_cost_vs[ci], " < ", primal_cost_us[ci], " + ", dre_per_part[ci])
                    points_3d_in_cluster[ci] = landmark_v.copy()
                else:
                    print(primal_cost_vs[ci], " > ", primal_cost_us[ci], " + ", dre_per_part[ci])

else:

    bfgs_mem = 6 # 2:Cost @50:  -12.87175888983266, 6: cost @ 50: 12.871757400143322
    bfgs_mu = 1.0
    bfgs_qs = np.zeros([bfgs_mem, kClusters * 3 * n_points]) # access/write with % mem
    bfgs_ps = np.zeros([bfgs_mem, kClusters * 3 * n_points])
    bfgs_rhos = np.zeros([bfgs_mem, 1])
    landmark_s_in_cluster_pre = [0 for x in range(kClusters)] # dummy fill list
    search_direction = [0 for x in range(kClusters)] # dummy fill list
    landmark_s_in_cluster_bfgs = [0 for x in range(kClusters)] # dummy fill list
    steplength = [0 for x in range(kClusters)]
    lastCostDRE_bfgs = lastCostDRE
    lastCostDRE_bfgs_per_part = [1e20 for x in range(kClusters)] 
    Gs = []
    Fs = []
    Fes = []
    rnaBufferSize = 6

    # Only it 0: update s,u,v.
    start = time.time()
    (
        cost,
        L_in_cluster,
        Vl_in_cluster,
        points_3d_in_cluster,
        x0_p,
        delta_l_in_cluster,
        blockLip_in_cluster,
        globalSingleLandmarksA_in_c,
        globalSingleLandmarksB_in_c
    ) = prox_f(
        x0_p, camera_indices_in_cluster, point_indices_in_cluster, points_2d_in_cluster,
        points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, 
        LipJ, blockLip_in_cluster, kClusters, innerIts=innerIts, sequential=True,
        )
    if newForUnique:
        for ci in range(kClusters):
            landmark_s_in_cluster[ci][globalSingleLandmarksB_in_c[ci]] = points_3d_in_cluster[ci][globalSingleLandmarksB_in_c[ci]]

    end = time.time()
    currentCost = np.sum(cost)
    print(-1, " ", round(currentCost), " gain ", round(lastCost - currentCost), ". ============= sum fk update takes ", end - start," s",)

    landmark_v, Vl_all, V_cluster_zeros = average_landmarks_new( # v update
        point_indices_in_cluster, points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, landmark_v, delta_l_in_cluster
    )

    # to enable using v in prox map (else s is not updated properly) we do this here, once and always below.
    steplength = 0
    tau = 1 # todo sqrt(2), not sure what is happening here.
    for ci in range(kClusters):
        landmark_s_in_cluster_pre[ci] = landmark_s_in_cluster[ci] + tau * (landmark_v - points_3d_in_cluster[ci]) # update s = s + v - u.
        steplength += np.linalg.norm(landmark_s_in_cluster_pre[ci] - landmark_s_in_cluster[ci], 2)**2
        #update_flat = (landmark_v - points_3d_in_cluster[ci]).flatten()
        #steplength += update_flat.dot(Vl_all * update_flat)
    steplength = np.sqrt(steplength)

    for it in range(its):

        # debugging cost block ################
        if False:
            dre = cost_DRE(point_indices_in_cluster, points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, landmark_v) + currentCost
            # here potentially skip line search and increase weight of step sizes (all?).

            primal_cost_v = 0
            for ci in range(kClusters):
                primal_cost_v += primal_cost(
                    x0_p,
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    landmark_v) #points_3d_in_cluster[ci]) # v not u
            primal_cost_u = 0
            for ci in range(kClusters):
                primal_cost_u += primal_cost(
                    x0_p,
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    points_3d_in_cluster[ci])

            print( it, " ======== DRE ====== ", round(dre) , " ========= gain " , \
                round(lastCostDRE - dre), "==== f(v)= ", round(primal_cost_v), " f(u)= ", round(primal_cost_u))
            # end debugging cost block ################

        # get line search direction and update bfgs data
        # operate with np concatenate to get large vector and reshape search_direction here?
        bfgs_r = np.zeros(kClusters * 3 * n_points)
        rna_s  = np.zeros(kClusters * 3 * n_points)
        rna_s_reg  = np.zeros(kClusters * 3 * n_points)
        for ci in range(kClusters): #bfgs_r = u-v

            temp = V_cluster_zeros[ci].diagonal()
            temp[temp != 0] = 1

            bfgs_r[ci * 3 * n_points: (ci+1) * 3 * n_points] = temp * (landmark_v - points_3d_in_cluster[ci]).flatten()
            #bfgs_r[ci * 3 * n_points: (ci+1) * 3 * n_points] = (landmark_s_in_cluster_pre[ci] - landmark_s_in_cluster[ci]).flatten() # with tau NO likely due to + h * nab
            rna_s[ci * 3 * n_points: (ci+1) * 3 * n_points] = temp * landmark_s_in_cluster_pre[ci].flatten()
            rna_s_reg[ci * 3 * n_points: (ci+1) * 3 * n_points] = temp * landmark_s_in_cluster[ci].flatten()

        use_bfgs = False #True # maybe full u,v?
        if use_bfgs:
            if False: # awful

                U_diag = np.zeros(rna_s.shape)
                for ci in range(kClusters):
                    U_diag[ci * 3 * n_points: (ci+1) * 3 * n_points] = blockEigenvalue(V_cluster_zeros[ci], 3).diagonal()
                U_diag = diag_sparse(U_diag)
                temp = U_diag.diagonal()[:]
                temp[temp < 1e-4] = 1e10
                U_diag.diagonal()[:] = 1. / temp[:]
                U_diag.diagonal()[temp >= 1e10] = 0

                # ok what happens if we replace mu by the hessian we have? Vl_all or even the Vl we have.
                dk = BFGS_direction(bfgs_r, bfgs_ps, bfgs_qs, bfgs_rhos, it, bfgs_mem, bfgs_mu, U_diag)
                dk_stepLength = np.linalg.norm(dk, 2)
                # step length by using Vl, also above computing steplength!
                #dk_stepLength = 0
                # for ci in range(kClusters): #bfgs_r = u-v
                #     bfgs_r[ci * 3 * n_points: (ci+1) * 3 * n_points] = landmark_v.flatten() - points_3d_in_cluster[ci].flatten()
                    #dk_stepLength += (bfgs_r[ci * 3 * n_points: (ci+1) * 3 * n_points]).dot(Vl_all * (bfgs_r[ci * 3 * n_points: (ci+1) * 3 * n_points]))
                #dk_stepLength = np.sqrt(dk_stepLength)
                multiplier = steplength / dk_stepLength

            else: # nesterov, best here.
                xk1 = rna_s
                xk05= rna_s - bfgs_r
                if it > 0:
                    #t_k1 = 1 + np.sqrt(1 + 4 * t_k)/2
                    #beta_nesterov = (tk-1) / (t_k1+2) # 488120, 52 but 1k gain
                    #t_k = t_k1
                    beta_nesterov = (it-1) / (it+2) # usual
                    delta_v = xk1 - xk05 + beta_nesterov * delta_v
                else:
                    t_k = 1
                    delta_v = xk1 - xk05 # for momentum
                dk = delta_v.copy()
                dk_stepLength = np.linalg.norm(dk, 2)
                multiplier = 1
        else: # RNA works, yet not good enough
            #L_rna = max(L_in_cluster) # nope?

            U_diag = np.zeros(rna_s.shape)
            for ci in range(kClusters):
                U_diag[ci * 3 * n_points: (ci+1) * 3 * n_points] = blockEigenvalue(V_cluster_zeros[ci], 3).diagonal()
            U_diag = diag_sparse(U_diag)
            #0.001 * lambdaScale
            # 59 / 0  ======== DRE BFGS ======  503077  ========= gain  7 ==== f(v)=  503075
            U_diag.data = np.ones(U_diag.data.shape) # appears better .. ? why?
            #U_diag = np.ones(rna_s.shape) ;U_diag = diag_sparse(U_diag)

            # try also inverse from some reason
            # same performance so no clear if needed at all. likely not :)
            # temp = U_diag.diagonal()[:]
            # temp[temp < 1e-4] = 1e10
            # U_diag.diagonal()[:] = 1. / temp[:]
            # U_diag.diagonal()[temp >= 1e10] = 0

            # stable but a bit slow.
            lambdaScale = np.sqrt(np.mean(U_diag.diagonal())) # sqrt?
            #Gs, Fs, Fes, dk = RNA(Gs, Fs, rna_s_reg, bfgs_r, it, rnaBufferSize, Fes, bfgs_r, res_pcg = U_diag, lamda = 0.001 * lambdaScale, h = -1)
            #dk = dk - rna_s_reg

            # other option also stable with - rna_s.
            #0.01: or 0.0001? 59 / 0  ======== DRE BFGS ======  504493
            # 0.1: 488k, 0.2: 497k
            Gs, Fs, Fes, dk = RNA(Gs, Fs, rna_s, bfgs_r, it, rnaBufferSize, Fes, bfgs_r, res_pcg = U_diag, lamda = 0.1 * lambdaScale, h = -1)
            if it < 5:
                dk = dk - rna_s # start at s, not s+. stable.
            else: # ok maybe the '-1' should be 0 then .. ?
                #dk = dk - (rna_s - bfgs_r) # wrose? haeh
                dk = dk - rna_s_reg # appears since dk is extrapolated point and ls is reg + dk this should be used, but just not stable at all, crazy overshoot in 1st run(s)?

            dk_stepLength = np.linalg.norm(dk, 2)
            multiplier = 1

        for ci in range(kClusters):
            search_direction[ci] = dk[ci * 3 * n_points: (ci+1) * 3 * n_points].reshape(n_points, 3) # reshape 3, n_points ?

        # need a check to reject idiotic proposals:
        # rho(u-s)^2 is gigantic
        # line search:
        line_search_iterations = 3
        print(" ..... step length ", steplength, " bfgs step ", dk_stepLength, " ratio ", multiplier, file=sys.stderr)
        for ls_it in range(line_search_iterations):
            tk = ls_it / max(1,line_search_iterations-1)
            for ci in range(kClusters):
                landmark_s_in_cluster_bfgs[ci] = tk * landmark_s_in_cluster_pre[ci] + (1-tk) * (landmark_s_in_cluster[ci] + multiplier * search_direction[ci])
                #print(" bfgs_r ", bfgs_r[ci * 3 * n_points: (ci+1) * 3 * n_points].reshape(n_points, 3))
                #print(" search_direction[ci] ", search_direction[ci])

            # prox on line search s:
            #print("1. x0_p", "points_3d_in_cluster", points_3d_in_cluster)
            points_3d_in_cluster_bfgs = [elem.copy() for elem in points_3d_in_cluster]
            L_in_cluster_bfgs = L_in_cluster.copy()
            Vl_in_cluster_bfgs = [elem.copy() for elem in Vl_in_cluster]
            blockLip_in_cluster_bfgs = blockLip_in_cluster.copy()
            (   cost_bfgs,
                L_in_cluster_bfgs,
                Vl_in_cluster_bfgs,
                points_3d_in_cluster_bfgs,
                x0_p_bfgs,
                delta_l_in_cluster,
                blockLip_in_cluster_bfgs,
                globalSingleLandmarksA_in_c,
                globalSingleLandmarksB_in_c # fixed
            ) = prox_f(
                x0_p.copy(), camera_indices_in_cluster, point_indices_in_cluster, points_2d_in_cluster,
                points_3d_in_cluster_bfgs, landmark_s_in_cluster_bfgs, L_in_cluster_bfgs, Vl_in_cluster_bfgs, 
                LipJ, blockLip_in_cluster_bfgs, kClusters, innerIts=innerIts, sequential=True,
                )
            if newForUnique:
                for ci in range(kClusters):
                    landmark_s_in_cluster_bfgs[ci][globalSingleLandmarksB_in_c[ci]] = points_3d_in_cluster_bfgs[ci][globalSingleLandmarksB_in_c[ci]]

            #print("2. x0_p", "points_3d_in_cluster", points_3d_in_cluster)
            currentCost_bfgs = np.sum(cost_bfgs)
            landmark_v_bfgs, _, V_cluster_zeros = average_landmarks_new( # v update
                point_indices_in_cluster, points_3d_in_cluster_bfgs, landmark_s_in_cluster_bfgs, L_in_cluster_bfgs, Vl_in_cluster_bfgs, landmark_v, delta_l_in_cluster
                )
            #print("3. x0_p", x0_p, "points_3d_in_cluster", points_3d_in_cluster)
            # update buffers
            if ls_it == 0: # todo: the one we accept put here, no?
                bfgs_ps[it % bfgs_mem] = -dk #* multiplier
                #bfgs_ps[it % bfgs_mem] = -bfgs_r # this is not so much overshooting as dk
                bfgs_rr = np.zeros(kClusters * 3 * n_points)
                for ci in range(kClusters):
                    bfgs_rr[ci * 3 * n_points: (ci+1) * 3 * n_points] = landmark_v_bfgs.flatten() - points_3d_in_cluster_bfgs[ci].flatten() # flatten?
                bfgs_qs[it % bfgs_mem] = bfgs_rr - bfgs_r
                bfgs_rhos[it % bfgs_mem] = np.maximum(0., 1./ bfgs_qs[it % bfgs_mem].dot(bfgs_ps[it % bfgs_mem]))

            # eval cost
            dre_bfgs, dre_per_part = cost_DRE(point_indices_in_cluster, points_3d_in_cluster_bfgs, landmark_s_in_cluster_bfgs, L_in_cluster_bfgs, Vl_in_cluster_bfgs, landmark_v_bfgs)
            dre_bfgs += currentCost_bfgs

            # debugging cost block ################
            primal_cost_us = [0 for _ in range(kClusters)]
            primal_cost_vs = [0 for _ in range(kClusters)]
            for ci in range(kClusters):
                primal_cost_vs[ci] = primal_cost(
                    x0_p_bfgs,
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    landmark_v_bfgs) #points_3d_in_cluster[ci]) # v not u
            primal_cost_u = 0
            for ci in range(kClusters):
                primal_cost_us[ci] += primal_cost(
                    x0_p_bfgs,
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    points_3d_in_cluster_bfgs[ci])
            primal_cost_v = np.sum(np.array(primal_cost_vs))
            primal_cost_u = np.sum(np.array(primal_cost_us))

            dre_bfgs = max(dre_bfgs, primal_cost_v) # sandwich lemma 
            print( it, "/", ls_it, " ======== DRE BFGS ====== ", round(dre_bfgs) , " ========= gain " , \
                round(lastCostDRE_bfgs - dre_bfgs), "==== f(v)= ", round(primal_cost_v), " f(u)= ", round(primal_cost_u), " ~= ", currentCost_bfgs)
            bestCost = np.minimum(primal_cost_v, bestCost)
            bestIt = it
            if it < 60:
                bestCost60 = bestCost
            if it < 30:
                bestCost30 = bestCost

            # appears better to do globally .. hm. not used any more.
            if lastCostDRE_bfgs < dre_bfgs and ls_it == line_search_iterations-1:
                LipJ += 0.2 * np.ones(kClusters) # need to skip once .. 
            # if ls_it == line_search_iterations-1:
            #     for ci in range(kClusters):
            #         if lastCostDRE_bfgs_per_part[ci] < dre_per_part[ci] + cost_bfgs[ci] :
            #             LipJ[ci] += 0.2

            # accept / reject, reject all but drs and see
            # if ls_it == line_search_iterations-1 :
            if dre_bfgs <= lastCostDRE_bfgs or ls_it == line_search_iterations-1 : # not correct yet, must be <= last - c/gamma |u-v|

                steplength = 0
                for ci in range(kClusters):
                    landmark_s_in_cluster[ci] = landmark_s_in_cluster_bfgs[ci].copy()
                    s_step_cluster = landmark_v_bfgs - points_3d_in_cluster_bfgs[ci]
                    landmark_s_in_cluster_pre[ci] = landmark_s_in_cluster[ci] + tau * s_step_cluster # update s = s + v - u.
                    steplength += np.linalg.norm(s_step_cluster, 2)**2
                    #update_flat = (landmark_v - points_3d_in_cluster[ci]).flatten()
                    #steplength += update_flat.dot(Vl_all * update_flat)
                steplength = np.sqrt(steplength)

                for ci in range(kClusters):
                    if linearize_at_last_solution: # better with acceleration.
                        points_3d_in_cluster[ci] = points_3d_in_cluster_bfgs[ci].copy()
                    else:
                        #compare cost of part and dre_per_part to pick best option
                        if primal_cost_vs[ci] < primal_cost_us[ci] + dre_per_part[ci]:
                            print(primal_cost_vs[ci], " < ", primal_cost_us[ci], " + ", dre_per_part[ci])
                            points_3d_in_cluster[ci] = landmark_v_bfgs.copy()
                        else:
                            print(primal_cost_vs[ci], " > ", primal_cost_us[ci], " + ", dre_per_part[ci])
                            points_3d_in_cluster[ci] = points_3d_in_cluster_bfgs[ci].copy()
                    Vl_in_cluster[ci] = Vl_in_cluster_bfgs[ci].copy()
                L_in_cluster = L_in_cluster_bfgs.copy()
                blockLip_in_cluster = blockLip_in_cluster_bfgs.copy()
                landmark_v = landmark_v_bfgs.copy()
                lastCostDRE_bfgs = dre_bfgs.copy()
                for ci in range(kClusters):
                    lastCostDRE_bfgs_per_part[ci] = dre_per_part[ci] + cost_bfgs[ci]

                x0_p = x0_p_bfgs.copy()
                #print("A landmark_s_in_cluster", landmark_s_in_cluster)
                #print("x0_p ", x0_p, file=sys.stderr)
                if o3d_defined:
                    rerender(vis, camera_indices_in_cluster, point_indices_in_cluster, x0_p, landmark_v)

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

if o3d_defined:
    vis, cameras_vis1, landmarks_vis = render_points_cameras(camera_indices_in_cluster, point_indices_in_cluster, x0_p, landmark_v)

if write_output:
    x0_p.tofile("camera_params_drs.dat")
    landmark_v.tofile("point_params_drs.dat")

import json
result_dict = {"base_url": BASE_URL, "file_name": FILE_NAME, "iterations" : its, \
               "bestCost" : round(bestCost), "bestIt": bestIt, "kClusters" : kClusters, \
               "bestCost60" : round(bestCost60), "bestCost30" : round(bestCost30) }
with open('results_drs.json', 'a') as json_file:
    json.dump(result_dict, json_file)