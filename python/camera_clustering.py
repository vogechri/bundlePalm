from __future__ import print_function
from termios import CINTR
import ctypes
import numpy as np


# more smart and new. After partitioning by camera non overlapping, we add
# residuals and cameras to complete (certain) landamrks until each lm is covered in some part.
def cluster_by_camera(
    camera_indices_, points_3d_, points_2d_, point_indices_, kClusters_, startL_
):
    n_cameras = np.max(camera_indices_)
    # sort by res-indices by camera indices
    res_sorted = np.argsort(camera_indices_)
    # camera_indices_[res_sorted]
    num_res = camera_indices_.shape[0]
    num_cams = np.unique(camera_indices_).shape[0]
    num_lands = points_3d_.shape[0]
    print("number of residuum: ", num_res)
    # now split by cameras. list [] of nparrays with camera_ind
    cluster_to_camera_ = np.array_split(np.arange(n_cameras), kClusters_)
    # cluster_to_residuum =
    # smarter cluster to camera. in loop pick cam for smallest cluster the one with most overlap
    # to current set or better ratio 'in set'+1 / 'not other sets'+1
    cluster_to_camera_smart_ = [] # list of sets
    cluster_to_landmarks_ = []
    # make map cam id to landmark indices seen by cam as set.
    cam_idx_to_lms = [ set(point_indices_[camera_indices_ == cam_idx]) for cam_idx in range(num_cams) ]

    # init pick
    cameras_available = set(range(num_cams))
    #print("cameras_available ", cameras_available, " num_cams ", num_cams)
    cluster_to_camera_smart_.append(set([0])) # cam 0 to set 0
    cluster_to_landmarks_.append(cam_idx_to_lms[0])
    cameras_available.remove(0)
    for i in range(kClusters_-1):
        # find other cam with lowest overlap, append as cluster
        lowest_overlap = num_lands+1
        best_cam = -1
        for candidate in cameras_available:
            # overlap with other sets = landmarks seen from cams in set, also seen by some camera
            sum_overlap = 0
            for set_id in range(len(cluster_to_camera_smart_)):
                # overlap:
                sum_overlap += len(set.intersection(cluster_to_landmarks_[set_id], cam_idx_to_lms[candidate]))
            if lowest_overlap > sum_overlap:
                lowest_overlap = sum_overlap
                best_cam = candidate
        cluster_to_camera_smart_.append(set([best_cam]))
        cameras_available.remove(best_cam)
        cluster_to_landmarks_.append(cam_idx_to_lms[best_cam])
    # init done
    #print("cluster_to_camera_smart_ ini ", cluster_to_camera_smart_)
    while cameras_available: # not empty
        # pick smallest cluster, find cam with largest overlap / sum overlap
        cid = np.argmin(np.array([len(x) for x in cluster_to_landmarks_]))
        #print("cid ", cid, " ", len(cluster_to_landmarks_[cid]))
        best_score = 0
        best_cam = -1
        for candidate in cameras_available:
            #print("candidate ", candidate)
            # overlap with other sets = landmarks seen from cams in set, also seen by some camera
            nominator   = len(set.intersection(cluster_to_landmarks_[cid], cam_idx_to_lms[candidate]))
            denominator = 1
            for set_id in range(len(cluster_to_camera_smart_)):
                # overlap:
                denominator += len(set.intersection(cluster_to_landmarks_[set_id], cam_idx_to_lms[candidate]))
            if nominator/denominator > best_score:
                best_score = nominator/denominator
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
        indices_in_cluster = np.logical_and( # indices of residums in cluster, unclear why indirection below or sorting is needed?
            camera_indices_[res_sorted] <= cluster_to_camera_[c][-1],
            camera_indices_[res_sorted] >= cluster_to_camera_[c][0],
        )
        res_indices_in_cluster = res_sorted[indices_in_cluster]
        print(res_indices_in_cluster.shape," ", res_indices_in_cluster)
        #res_indices_in_cluster = np.sort(indices_in_cluster)
        if True: # other version of clustering
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

    landmark_occurences = np.zeros(num_lands)
    for ci in range(kClusters_):
        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        landmark_occurences[unique_points_in_c_] +=1
    landmarks_divided_over_cluster_ = landmark_occurences > 1
    _, idx, counts = np.unique(point_indices_in_cluster_[cid], return_inverse=True, return_counts=True)
    counts[idx]
    # find lm with lowest missing and cluster
    #per cl sort by occurence argsort of misses
    #pick cluster, add res remove
    # landmarks_divided_over_cluster_ -> residuals ids

    # inverse_point_indices = -np.ones(np.max(unique_points_in_c_) + 1)  # all -1
    # for i in range(num_lands)[landmarks_divided_over_cluster_]:
    #         inverse_point_indices[unique_points_in_c_[i]] = i

    point_indices_to_complete = np.arange(num_lands)[landmarks_divided_over_cluster_]
    point_to_res_id_ = [set() for i in range(num_lands)]
    for i in range(num_res):
        if landmarks_divided_over_cluster_[point_indices_[i]]: # non unique add res-number
            point_to_res_id_[point_indices_[i]].add(i)
    # point to res id holds all residuum ids. most know which id is missing per cluster and point
    # add the missing ones.could add those
    #
    # i wanted to add only few cam ids for some reason -> bandwidth when sending/loading data.
    # loop over andmarks_divided_over_cluster_

    _, idx, res_per_lm = np.unique(point_indices_, return_inverse=True, return_counts=True)
    missing_res_per_lm_c_ = []
    num_res_per_c_ = np.zeros(kClusters_)
    for ci in range(kClusters_):
        _, idx, counts = np.unique(point_indices_in_cluster_[ci], return_inverse=True, return_counts=True)
        counts = np.hstack([counts, np.zeros(res_per_lm.shape[0] - counts.shape[0])])
        missing_res_per_lm_c_.append(res_per_lm - counts)
        num_res_per_c_[ci] = np.sum(counts)

        # from res of lm in total we also need missing res of lm in cluster
        #res_of_lm_c_ = {value: np.where(point_indices_ == value) for value in np.unique(point_indices_)}

    res_of_lm = {value: np.where(point_indices_ == value) for value in np.unique(point_indices_)}
    #print(res_of_lm) # ok

    res_of_lm_notin_c_ = []
    res_of_lm_in_c_ = []
    for ci in range(kClusters_):
        res_notin_c = np.ones(num_res, dtype=bool)
        res_notin_c[res_indices_in_cluster_[ci]] = False
        point_indices_temp = point_indices_.copy()
        point_indices_temp[res_notin_c] = -1
        tmp = {value: np.where(point_indices_temp == value) for value in np.unique(point_indices_temp)}
        res_of_lm_in_c_.append(tmp)

        res_notin_c = np.zeros(num_res, dtype=bool)
        res_notin_c[res_indices_in_cluster_[ci]] = True
        point_indices_temp = point_indices_.copy()
        point_indices_temp[res_notin_c] = -1
        tmp = {value: np.where(point_indices_temp == value) for value in np.unique(point_indices_temp)}
        res_of_lm_notin_c_.append(tmp)

        #print(ci, " " ,  res_of_lm_notin_c_[ci])

    res_toadd_to_c_ = [[] for i in range(kClusters_)]
    for i in point_indices_to_complete:
        #print(i, " complete ")
        cost = np.zeros(kClusters_)
        for ci in range(kClusters_):
            cost[ci] += res_of_lm_notin_c_[ci][i][0].shape[0] * num_res
            cost[ci] += num_res_per_c_[ci]
            #print(ci, " ", i, " res to add ", res_of_lm_notin_c_[ci][i][0].shape[0], " ", num_res_per_c_[ci] )
            #print(res_of_lm_notin_c_[ci][i][0])

            # if missing_res_per_lm_c_[ci][i] == 0:
            #     cost[ci] = num_res * 100
            # missing is new cam added (i do not know the res at all)
        ci = np.argmin(cost)
        # add res missing to cluster
        #print(ci, " ", i, " res to add ", missing_res_per_lm_c_[ci][i], " ", num_res_per_c_[ci] )
        #print("best ci ", ci, " ", i, " res to add ", res_of_lm_notin_c_[ci][i][0].shape[0], " of ", res_of_lm[i][0].shape[0], " ", num_res_per_c_[ci] )
        #print(res_of_lm_notin_c_[ci][i][0])
        #print("all res_of_lm[i] ", res_of_lm[i])

        res_toadd_to_c_[ci].append(res_of_lm_notin_c_[ci][i][0])
        num_res_per_c_[ci] += res_of_lm_notin_c_[ci][i][0].shape[0]

    for ci in range(kClusters_):
        print(ci, " adding " , np.concatenate(res_toadd_to_c_[ci]).shape, " residuals to ", \
            point_indices_in_cluster_[ci].shape, " original residuals")

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

# returns covered_landmark_indices_c: those in addition! -- which is weird -- to those covered by main residual split
# the latter are point_indices_already_covered and also returned
def process_clusters(num_lands, num_res, kClusters_, point_indices_in_cluster_, point_indices_, res_indices_in_cluster_):
    landmark_occurrences = np.zeros(num_lands)
    for ci in range(kClusters_):
        unique_points_in_cluster = np.unique(point_indices_in_cluster_[ci])
        landmark_occurrences[unique_points_in_cluster] += 1
    print("uncovered are ", np.sum(landmark_occurrences < 1), " landmarks ", " present in single ", np.sum(landmark_occurrences==1),
          " present in multiple ", np.sum(landmark_occurrences>1))

    # attention. omg makes no sense.
    landmarks_divided_over_cluster = landmark_occurrences > 1
    point_indices_to_complete = np.arange(num_lands)[landmarks_divided_over_cluster]

    print("point_indices_to_complete ", point_indices_to_complete.shape)

    # also return those covered in main, disjoint from covered_landmark_indices better rename.
    # 
    point_indices_already_covered = []
    sum_points_covered = 0
    for ci in range(kClusters_):
        unique_points_in_cluster = np.unique(point_indices_in_cluster_[ci])
        # assumes that landmarks only present in cluster are present with all observations. since cameras are disjoint it follows that all cameras must be present in cluster for those -> ok
        point_indices_already_covered.append(np.intersect1d(unique_points_in_cluster, np.arange(num_lands)[landmark_occurrences == 1] ))
        print(ci, " point_indices_already_covered ", point_indices_already_covered[ci].shape)
        sum_points_covered += point_indices_already_covered[ci].shape[0]
    print("Together covered points ", point_indices_to_complete.shape[0] + sum_points_covered, "  sum_points_covered: " ,  sum_points_covered)

    point_to_res_id = [set() for _ in range(num_lands)]
    for i in range(num_res):
        if landmarks_divided_over_cluster[point_indices_[i]]:
            point_to_res_id[point_indices_[i]].add(i)

    _, _, res_per_lm = np.unique(point_indices_, return_inverse=True, return_counts=True)
    missing_res_per_lm_c = []
    num_res_per_c = np.zeros(kClusters_)

    for ci in range(kClusters_):
        _, _, counts = np.unique(point_indices_in_cluster_[ci], return_inverse=True, return_counts=True)
        counts = np.hstack([counts, np.zeros(res_per_lm.shape[0] - counts.shape[0])])
        missing_res_per_lm_c.append(res_per_lm - counts)
        num_res_per_c[ci] = np.sum(counts)

    res_of_lm = {value: np.where(point_indices_ == value) for value in np.unique(point_indices_)}

    res_of_lm_notin_c = []
    res_of_lm_in_c = []

    for ci in range(kClusters_):
        res_notin_c = np.ones(num_res, dtype=bool)
        res_notin_c[res_indices_in_cluster_[ci]] = False
        point_indices_temp = point_indices_.copy()
        point_indices_temp[res_notin_c] = -1
        tmp = {value: np.where(point_indices_temp == value) for value in np.unique(point_indices_temp)}
        res_of_lm_in_c.append(tmp)

        res_notin_c = np.zeros(num_res, dtype=bool)
        res_notin_c[res_indices_in_cluster_[ci]] = True
        point_indices_temp = point_indices_.copy()
        point_indices_temp[res_notin_c] = -1
        tmp = {value: np.where(point_indices_temp == value) for value in np.unique(point_indices_temp)}
        res_of_lm_notin_c.append(tmp)

    res_toadd_to_c = [[] for _ in range(kClusters_)]
    covered_landmark_indices_c = [[] for _ in range(kClusters_)]

    # also consider cam linked by residuum already present in cluster or not
    for i in point_indices_to_complete:
        cost = np.zeros(kClusters_)
        for ci in range(kClusters_):
            cost[ci] += len(res_of_lm_notin_c[ci][i][0]) * num_res
            cost[ci] += num_res_per_c[ci]

        ci = np.argmin(cost)
        res_toadd_to_c[ci].append(res_of_lm_notin_c[ci][i][0])
        num_res_per_c[ci] += len(res_of_lm_notin_c[ci][i][0])
        covered_landmark_indices_c[ci].append(i)
    for ci in range(kClusters_):
        res_toadd_to_c[ci] = np.concatenate(res_toadd_to_c[ci])

    return res_toadd_to_c, point_indices_already_covered, covered_landmark_indices_c, num_res_per_c

def fillPythonVec(out, sizes_out, kClusters_):
    ret = []
    start = 0
    for i__ in range (kClusters_):
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

def cluster_covis_lib(kClusters, pre_merges_, camera_indices__, point_indices__, old_vtxsToPart_=0):
    c_kClusters_ = ctypes.c_int(kClusters)
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

    if (isinstance(old_vtxsToPart_, list)):
        c_old_vtxsToPart_ptr = (ctypes.c_int * len(old_vtxsToPart_))(*old_vtxsToPart_)
        old_vtxsToPart_cpp = lib.new_vector_by_copy(c_old_vtxsToPart_ptr, len(c_old_vtxsToPart_ptr))
    else:
        old_vtxsToPart_cpp = lib.new_vector()

    lib.cluster_covis(c_kClusters_, c_pre_merges_, c_max_vol_part_, c_cam_indices_cpp, c_point_indices_cpp, res_to_cluster_c_out, res_to_cluster_c_sizes, old_vtxsToPart_cpp)

    old_vtxsToPart_ = fillPythonVecSimple(old_vtxsToPart_cpp).tolist()
    kClusters = lib.vector_size(res_to_cluster_c_sizes)

    res_indices_in_cluster__ = fillPythonVec(res_to_cluster_c_out, res_to_cluster_c_sizes, kClusters)
    return res_indices_in_cluster__, kClusters, old_vtxsToPart_
    # copy data, free c++ mem

def cluster_by_camera_gpt(
    camera_indices_, points_2d_, point_indices_, kClusters_, pre_merges, old_vtxsToPart=0, baseline_clustering=False, init_cam_id=0, init_lm_id=0, seed=0
):
    np.random.seed(seed)
    # sort by res-indices by camera indices
    res_sorted = np.argsort(camera_indices_)
    num_res = camera_indices_.shape[0]
    num_cams = np.unique(camera_indices_).shape[0]
    num_lands = np.unique(point_indices_).shape[0] #points_3d_.shape[0]
    print("number of residuum: ", num_res)
    n_cameras = np.max(camera_indices_)

    if baseline_clustering:

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
        for i in range(kClusters_-1):
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
            cluster_to_camera_[c] = cluster_to_camera_smart_[c]
            if True: # other version of clustering
                indices_in_cluster = np.zeros(num_res, dtype=bool)
                for camid_in_c_ in cluster_to_camera_smart_[c]:
                    #print(indices_in_cluster.shape, " ", indices_in_cluster.shape, " ", camid_in_c_)
                    indices_in_cluster = np.logical_or(indices_in_cluster, camera_indices_==camid_in_c_)
                res_indices_in_cluster = np.arange(num_res)[indices_in_cluster]
                print(res_indices_in_cluster.shape," ", res_indices_in_cluster)
            else:
                indices_in_cluster = np.logical_and( # indices of residums in cluster, unclear why indirection below or sorting is needed?
                camera_indices_[res_sorted] <= cluster_to_camera_[c][-1],
                camera_indices_[res_sorted] >= cluster_to_camera_[c][0],
                )
                res_indices_in_cluster = res_sorted[indices_in_cluster]
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

    else:
        res_indices_in_cluster_, kClusters, old_vtxsToPart = cluster_covis_lib(kClusters_, pre_merges, camera_indices_, point_indices_, old_vtxsToPart)
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

            #res_indices_in_cluster_.append(res_indices_in_cluster.copy())


    res_toadd_to_c_, point_indices_already_covered_, covered_landmark_indices_c_, res_to_cluster_by_landmark_ = \
        process_cluster_lib(num_lands, num_res, kClusters, point_indices_in_cluster_, res_indices_in_cluster_, point_indices_)

    # (res_toadd_to_c_, point_indices_already_covered_, covered_landmark_indices_c_, num_res_per_c) = \
    #     process_clusters(num_lands, num_res, kClusters, \
    #         point_indices_in_cluster_, point_indices_, res_indices_in_cluster_)
    
    for ci in range(kClusters):
        print(ci, " adding " , res_toadd_to_c_[ci].shape, " residuals to ", \
            point_indices_in_cluster_[ci].shape, " original residuals")
    
    additional_point_indices_in_cluster_ = [0 for _ in range(kClusters)] # variables to add, just unique (additional cameras -> not needed)
    additional_camera_indices_in_cluster_ = [0 for _ in range(kClusters)] # index into var per res
    additional_points_2d_in_cluster_ = [0 for _ in range(kClusters)] # essentially rhs for res
    # point_indices_already_covered: landmarks to be updated, present in main res only not in additional res (since complete)
    # covered_landmark_indices_c_: landmarks to be updated in additional!!! set of res since completely contained in cluster

    for ci in range(kClusters):
        #print("camera_indices_ ", camera_indices_.shape)
        con_res = res_toadd_to_c_[ci]
        #print("camera_indices_ ", con_res.shape)
        new_cam_indices_ = np.unique(camera_indices_[con_res])
        #print(ci, " new cam indices ", new_cam_indices_, " " , new_cam_indices_.shape)
        #print(ci, " old cam indices ", np.unique(camera_indices_in_cluster_[ci]), " " , np.unique(camera_indices_in_cluster_[ci]).shape)
        new_cam_indices_ = np.setdiff1d(new_cam_indices_, np.unique(camera_indices_in_cluster_[ci])) # not needed since disjoint anyway
        #print(ci, " new cam indices ", new_cam_indices_, " " , new_cam_indices_.shape)
        additional_camera_indices_in_cluster_[ci] = camera_indices_[con_res]
        additional_points_2d_in_cluster_[ci] = points_2d_[con_res]
        covered_landmark_indices_c_[ci] = np.array(covered_landmark_indices_c_[ci]) #?
        additional_point_indices_in_cluster_[ci] = point_indices_[con_res]
        point_indices_already_covered_[ci] = np.union1d(point_indices_already_covered_[ci], covered_landmark_indices_c_[ci])
        print("===== Cluster ", ci , " covers ", point_indices_already_covered_[ci].shape, "landmarks ", " of ", num_lands)

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
        res_indices_in_cluster_,
        additional_point_indices_in_cluster_, additional_camera_indices_in_cluster_, additional_points_2d_in_cluster_, point_indices_already_covered_, covered_landmark_indices_c_,
        old_vtxsToPart, kClusters
    )

lib = ctypes.CDLL("./libprocess_clusters.so")
init_lib()