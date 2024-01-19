from __future__ import print_function

import os
import urllib.request
import bz2
import ctypes
import numpy as np

# Step 1 & 2: You've already done this part by writing your code.
# Step 3: Compile your .cpp file into an object file.
# g++ -c -fPIC process_clusters.cpp -o process_clusters.o
# Step 4: Convert the object file into a shared library.
# g++ -shared -o libprocess_clusters.so process_clusters.o
#The function process_clusters is not being exported. On some platforms, you need to explicitly export the symbols that should be available in the shared library. You can do this by adding __declspec(dllexport) before the function definition on Windows, or by using the -Wl,--export-dynamic option when compiling the library on Linux:
#g++ -shared -o libprocess_clusters.so process_clusters.o -Wl,--export-dynamic

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/dubrovnik/"
FILE_NAME = "problem-16-22106-pre.txt.bz2"
FILE_NAME = "problem-135-90642-pre.txt.bz2"

#BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
#FILE_NAME = "problem-49-7776-pre.txt.bz2"
#FILE_NAME = "problem-73-11032-pre.txt.bz2"

#BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/dubrovnik/"

# BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/final/"
# FILE_NAME = "problem-93-61203-pre.txt.bz2"

# BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/venice/"
# FILE_NAME = "problem-52-64053-pre.txt.bz2"


URL = BASE_URL + FILE_NAME

if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)


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
    return ret

def fillPythonVecSimple(out):
    tmp = []
    for j in range(lib.vector_size(out)):
        tmp.append(lib.vector_get(out, j))
    return np.array(tmp)

def init_lib():
    # lib.process_clusters.restype = ctypes.POINTER(ctypes.c_int)
    # lib.process_clusters.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
    #                                 ctypes.POINTER(ctypes.c_int),
    #                                 ctypes.POINTER(ctypes.c_int),
    #                                 ctypes.POINTER(ctypes.c_int),
    #                                 ctypes.POINTER(ctypes.c_int),
    #                                 ctypes.POINTER(ctypes.c_int)]
    # before?                        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
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
    #lib.process_clusters_test.restype = None
    #lib.process_clusters_test.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    #lib.process_clusters_test(c_num_lands, c_num_res, c_kClusters, c_res_indices_in_cluster_flat_cpp)
    #lib.delete_vector(c_res_indices_in_cluster_flat_cpp)

    lib.process_clusters.restype = None #[ctypes.c_void_p, ctypes.c_void_p]
    lib.process_clusters.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p, # out
                                    ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p]

    lib.cluster_covis.restype = None
    lib.cluster_covis.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p] #,
    #                                  ctypes.c_void_p, ctypes.c_void_p] # out]


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
    res_toadd_sizes_out = lib.new_vector_of_size(kClusters)

    point_indices_already_covered_out = lib.new_vector_of_size(kClusters)
    point_indices_already_covered_sizes = lib.new_vector_of_size(kClusters)

    # print("point_indices_already_covered_outsiez ", lib.vector_size(point_indices_already_covered_out))
    # print("point_indices_already_covered_sizes siez ", lib.vector_size(point_indices_already_covered_sizes))

    covered_landmark_indices_c_out = lib.new_vector()
    covered_landmark_indices_c_sizes = lib.new_vector_of_size(kClusters)

    num_res_per_c_out = lib.new_vector()

    lib.process_clusters(c_num_lands_, c_num_res_, c_kClusters_,
                        c_point_indices_in_cluster_flat_cpp, c_point_indices_in_cluster_sizes_cpp,
                        c_point_indices_cpp,
                        c_res_indices_in_cluster_flat_cpp,c_res_indices_in_cluster_sizes_cpp,
                        res_toadd_out, res_toadd_sizes_out,
                        point_indices_already_covered_out, point_indices_already_covered_sizes,
                        covered_landmark_indices_c_out, covered_landmark_indices_c_sizes,
                        num_res_per_c_out)

    #print("lib.vector_get(res_toadd_sizes_out, i) ", 0, " : ", lib.vector_get(res_toadd_sizes_out, 0))
    res_toadd_to_c_ = fillPythonVec(res_toadd_out, res_toadd_sizes_out, kClusters)
    point_indices_already_covered_ = fillPythonVec(point_indices_already_covered_out, point_indices_already_covered_sizes, kClusters)
    covered_landmark_indices_c_ = fillPythonVec(covered_landmark_indices_c_out, covered_landmark_indices_c_sizes, kClusters)
    num_res_per_c_ = fillPythonVecSimple(res_toadd_out)

    return res_toadd_to_c_, point_indices_already_covered_, covered_landmark_indices_c_, num_res_per_c_

def cluster_covis_lib(num_res_, kClusters, n_points, n_cameras, camera_indices__, point_indices__):
    c_num_lands_ = ctypes.c_int(n_points)
    c_num_cams_ = ctypes.c_int(n_cameras)
    c_num_res_ = ctypes.c_int(num_res_)
    c_kClusters_ = ctypes.c_int(kClusters)

    camera_indices_list = camera_indices__.tolist()
    point_indices_list = point_indices__.tolist()

    c_point_indices_ptr = (ctypes.c_int * len(point_indices_list))(*point_indices_list)
    c_point_indices_cpp = lib.new_vector_by_copy(c_point_indices_ptr, len(c_point_indices_ptr))
    c_cam_indices_ptr = (ctypes.c_int * len(camera_indices_list))(*camera_indices_list)
    c_cam_indices_cpp = lib.new_vector_by_copy(c_cam_indices_ptr, len(c_cam_indices_ptr))

    #c_point_indices_ptr = (ctypes.c_int * len(point_indices__))(*point_indices__)
    #c_point_indices_cpp = lib.new_vector_by_copy(c_point_indices_ptr, len(c_point_indices_ptr))

    # res_to_cluster_c_out = lib.new_vector()
    # res_to_cluster_c_sizes = lib.new_vector_of_size(kClusters)

    lib.cluster_covis(c_num_res_, c_kClusters_, c_num_lands_, c_num_cams_, 
        c_cam_indices_cpp, c_point_indices_cpp)#, res_to_cluster_c_out, res_to_cluster_c_sizes)
    
    # copy data, free c++ mem


cameras, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)
n_cameras = cameras.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

np.set_printoptions(formatter={"float": "{: 0.2f}".format})

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))
kClusters = 3 # 6 cluster also not bad at all !

# Load the shared library
#lib = ctypes.CDLL("/path/to/your/library.so")
lib = ctypes.CDLL("/home/vogechri/bupa/python/libprocess_clusters.so")
init_lib()
cluster_covis_lib(m, kClusters, n_points, n_cameras, camera_indices, point_indices)

camera_indices_ = camera_indices
points_3d_  = points_3d
points_2d_  = points_2d
point_indices_ = point_indices
kClusters_ = kClusters
startL_ = 1
init_cam_id = 0 
init_lm_id = 0 
seed =0

np.random.seed(seed)
# sort by res-indices by camera indices
res_sorted = np.argsort(camera_indices_)
num_res = camera_indices_.shape[0]
num_cams = np.unique(camera_indices_).shape[0]
num_lands = points_3d_.shape[0]
print("number of residuum: ", num_res)
# now split by cameras. list [] of nparrays with camera_ind
cluster_to_camera_ = np.array_split(np.arange(n_cameras), kClusters_)

# smarter cluster to camera. in loop pick cam for smallest cluster the one with most overlap
# to current set or better ratio 'in set'+1 / 'not other sets'+1
cluster_to_camera_smart_ = [] # list of sets
cluster_to_landmarks_ = []
# make map cam id to landmark indices seen by cam as set.
cam_idx_to_lms = [ set(point_indices_[camera_indices_ == cam_idx]) for cam_idx in range(num_cams) ]

# init pick
cameras_available = set(range(num_cams))
#print("cameras_available ", cameras_available, " num_cams ", num_cams)
cluster_to_camera_smart_.append(set([init_cam_id])) # cam 0 to set 0
cluster_to_landmarks_.append(cam_idx_to_lms[init_lm_id])
cameras_available.remove(init_cam_id)
for i in range(kClusters-1):
    best_cam = min(cameras_available, key=lambda candidate: sum(np.random.normal(0,1,1)**2 + len(set.intersection(cluster_to_landmarks_[set_id], cam_idx_to_lms[candidate])) for set_id in range(len(cluster_to_camera_smart_))))
    cluster_to_camera_smart_.append(set([best_cam]))
    cameras_available.remove(best_cam)
    cluster_to_landmarks_.append(cam_idx_to_lms[best_cam])
# init done

while cameras_available:
    cid = np.argmin(np.array([len(x) for x in cluster_to_landmarks_]))
    best_score = 0
    best_cam = -1
    for candidate in cameras_available:
        overlap = len(set.intersection(cluster_to_landmarks_[cid], cam_idx_to_lms[candidate]))
        total_overlap = sum(len(set.intersection(cluster_to_landmarks_[set_id], cam_idx_to_lms[candidate])) for set_id in range(len(cluster_to_camera_smart_)))
        score = overlap / (total_overlap + 1)
        if score > best_score:
            best_score = score
            best_cam = candidate
    cluster_to_camera_smart_[cid].add(best_cam)
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
    for camid_in_c_ in cluster_to_camera_smart_[c]:
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

# based on this i must add lms(and res) to cluster EXTRA.
# each lm should have one cluster with all cam ids (+res) it occurs in
# notion of 
# 1. find incomplete lms -> not all res in single cluster. 
# 2. lm -> res missing by id (only incomplete)
# 3. distribute equally, pick cluster w least res. pick lm with least res to add, add (bunch)

print(point_indices_in_cluster_[0].shape)
print(point_indices_.shape)
print(res_indices_in_cluster_[0].shape)

point_indices_in_cluster = point_indices_in_cluster_
point_indices = point_indices_ #np.array(point_indices_)
#res_indices_in_cluster = res_indices_in_cluster_

res_toadd_to_c, point_indices_already_covered, covered_landmark_indices_c, num_res_per_c = \
    process_cluster_lib(num_lands, num_res, kClusters, point_indices_in_cluster_, res_indices_in_cluster_, point_indices)

exit()

# Convert the input arguments to C types
c_num_lands = ctypes.c_int(num_lands)
c_num_res = ctypes.c_int(num_res)
c_kClusters = ctypes.c_int(kClusters)

c_point_indices_in_cluster = (ctypes.POINTER(ctypes.c_int) * len(point_indices_in_cluster_))()
for i, sublist in enumerate(point_indices_in_cluster_):
    np_sublist = np.array(sublist, dtype=np.int32)
    c_point_indices_in_cluster[i] = np_sublist.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

c_res_indices_in_cluster = (ctypes.POINTER(ctypes.c_int) * len(res_indices_in_cluster_))()
for i, sublist in enumerate(res_indices_in_cluster_):
    np_sublist = np.array(sublist, dtype=np.int32)
    c_res_indices_in_cluster[i] = np_sublist.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

#c_point_indices_in_cluster = point_indices_in_cluster.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))
c_point_indices = point_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
#c_res_indices_in_cluster = res_indices_in_cluster.ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))

# Call the C++ function
result = lib.process_clusters(c_num_lands, c_num_res, c_kClusters, c_point_indices_in_cluster, c_point_indices, c_res_indices_in_cluster)

# Convert the result back to Python types
# TODO: Convert the result to the desired Python types

# Free the memory allocated by the C++ function
lib.free(result)