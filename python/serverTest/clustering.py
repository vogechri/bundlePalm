from __future__ import print_function
from termios import CINTR
import ctypes
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np


def partition_observations_by_point_owner(
    camera_indices, points_2d, point_indices, point_owner, cluster_count
):
    point_owner = np.asarray(point_owner, dtype=np.int64)
    if point_owner.shape != (int(np.max(point_indices)) + 1,):
        raise ValueError("DABA point ownership has an invalid shape")
    if np.any(point_owner < 0) or np.any(point_owner >= cluster_count):
        raise ValueError("DABA point ownership contains an invalid cluster")
    camera_clusters = []
    point_clusters = []
    observation_clusters = []
    for cluster in range(cluster_count):
        mask = point_owner[point_indices] == cluster
        camera_clusters.append(camera_indices[mask])
        point_clusters.append(point_indices[mask])
        observation_clusters.append(points_2d[mask])
    return camera_clusters, point_clusters, observation_clusters, cluster_count


def cluster_by_daba_louvain(
    camera_indices_, points_2d_, point_indices_, kClusters_, n_cameras_, n_points_,
    residual_balance_slack=0.05, minimum_camera_landmarks=20,
    max_refinement_passes=2,
):
    del residual_balance_slack, minimum_camera_landmarks, max_refinement_passes
    workspace = Path(__file__).resolve().parents[1]
    exporter = Path(os.environ.get(
        "BUNDLE_PALM_DABA_PARTITION_EXPORTER",
        workspace / "third_party/DABA/build-clustering/daba_partition_exporter",
    ))
    if not exporter.is_file():
        raise FileNotFoundError(exporter)
    cuda_directory = Path(os.environ.get(
        "BUNDLE_PALM_DABA_CUDA_DIRECTORY",
        Path.home() / "bae/.venv/targets/x86_64-linux/lib",
    ))
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(cuda_directory) + (
        ":" + environment["LD_LIBRARY_PATH"]
        if environment.get("LD_LIBRARY_PATH") else ""
    )
    with tempfile.TemporaryDirectory(prefix="bundle-palm-daba-partition-") as temp:
        problem = Path(temp) / "graph.txt"
        output = Path(temp) / "partition.json"
        with problem.open("w", encoding="utf-8") as stream:
            stream.write(
                f"{n_cameras_} {n_points_} {len(camera_indices_)}\n"
            )
            for camera, point, observation in zip(
                camera_indices_, point_indices_, points_2d_
            ):
                stream.write(
                    f"{int(camera)} {int(point)} "
                    f"{float(observation[0]):.17g} "
                    f"{float(observation[1]):.17g}\n"
                )
        subprocess.run(
            [str(exporter), str(problem), str(kClusters_), str(output), "0"],
            check=True,
            env=environment,
        )
        partition = json.loads(output.read_text(encoding="utf-8"))
    if partition["clusters"] != kClusters_:
        raise RuntimeError("DABA returned a different cluster count")
    if len(partition["camera_owner"]) != n_cameras_:
        raise RuntimeError("DABA returned invalid camera ownership")
    if len(partition["point_owner"]) != n_points_:
        raise RuntimeError("DABA returned invalid point ownership")
    return partition_observations_by_point_owner(
        camera_indices_, points_2d_, point_indices_,
        partition["point_owner"], kClusters_
    )

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

    lib.cluster_cameras_hypergraph.restype = ctypes.c_int
    lib.cluster_cameras_hypergraph.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_double, ctypes.c_void_p,
                                               ctypes.c_void_p, ctypes.c_void_p]

    lib.cluster_landmarks_clean.restype = ctypes.c_int
    lib.cluster_landmarks_clean.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int64, ctypes.c_int, ctypes.c_bool,
                                            ctypes.c_double,
                                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    lib.cluster_landmarks_scalable.restype = ctypes.c_int
    lib.cluster_landmarks_scalable.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_int, ctypes.c_int,
                                               ctypes.c_double, ctypes.c_void_p,
                                               ctypes.c_void_p, ctypes.c_void_p]
    lib.cluster_landmarks_scalable_stable.restype = ctypes.c_int
    lib.cluster_landmarks_scalable_stable.argtypes = [ctypes.c_int, ctypes.c_int,
                                                      ctypes.c_int, ctypes.c_int,
                                                      ctypes.c_int, ctypes.c_double,
                                                      ctypes.c_void_p, ctypes.c_void_p,
                                                      ctypes.c_void_p]

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

    lib.recluster_cameras(c_kClusters_, c_camera_indices_cpp, c_point_indices_cpp,
                          res_to_cluster_by_landmark_out)

    #print("lib.vector_get(res_toadd_sizes_out, i) ", 0, " : ", lib.vector_get(res_toadd_sizes_out, 0))
    #res_toadd_to_c_ = fillPythonVec(res_toadd_out, res_toadd_sizes_out, kClusters)
    point_indices_already_covered_ = fillPythonVec(point_indices_already_covered_out, point_indices_already_covered_sizes, kClusters__)
    covered_landmark_indices_c_ = fillPythonVec(covered_landmark_indices_c_out, covered_landmark_indices_c_sizes, kClusters__)
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

def cluster_by_camera(
    camera_indices_, points_3d_, points_2d_, point_indices_, kClusters_, startL_, n_cameras_
):
    # sort by res-indices by camera indices
    res_sorted = np.argsort(camera_indices_)
    # camera_indices_[res_sorted]
    num_res = camera_indices_.shape[0]
    print("number of residuum: ", num_res)
    # now split by cameras. list [] of nparrays with camera_ind
    cluster_to_camera_ = np.array_split(np.arange(n_cameras_), kClusters_)
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
    for ci in range(kClusters_):
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
    for ci in range(kClusters_):
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
        kClusters_
    )

def cluster_by_camera_hypergraph(
    camera_indices_, points_2d_, point_indices_, kClusters_, n_cameras_, n_points_,
    camera_balance_slack=0.0
):
    camera_indices_list = camera_indices_.tolist()
    point_indices_list = point_indices_.tolist()
    c_camera_indices = (ctypes.c_int * len(camera_indices_list))(*camera_indices_list)
    c_point_indices = (ctypes.c_int * len(point_indices_list))(*point_indices_list)
    c_camera_indices_cpp = lib.new_vector_by_copy(c_camera_indices, len(c_camera_indices))
    c_point_indices_cpp = lib.new_vector_by_copy(c_point_indices, len(c_point_indices))
    camera_to_cluster_cpp = lib.new_vector()

    try:
        status = lib.cluster_cameras_hypergraph(
            ctypes.c_int(kClusters_), ctypes.c_int(n_cameras_), ctypes.c_int(n_points_),
            ctypes.c_double(camera_balance_slack), c_camera_indices_cpp,
            c_point_indices_cpp, camera_to_cluster_cpp)
        if status != 0:
            raise RuntimeError("camera hypergraph partitioning failed")
        camera_to_cluster = fillPythonVecSimple(camera_to_cluster_cpp)
    finally:
        lib.delete_vector(camera_to_cluster_cpp)
        lib.delete_vector(c_point_indices_cpp)
        lib.delete_vector(c_camera_indices_cpp)

    if camera_to_cluster.shape[0] != n_cameras_:
        raise RuntimeError("camera hypergraph partitioning returned an invalid assignment")

    camera_indices_in_cluster_ = []
    point_indices_in_cluster_ = []
    points_2d_in_cluster_ = []
    low_degree_counts = np.zeros(10, dtype=np.int64)
    point_cluster_occurrences = np.zeros(n_points_, dtype=np.int64)
    for cluster in range(kClusters_):
        residual_mask = camera_to_cluster[camera_indices_] == cluster
        cluster_camera_indices = camera_indices_[residual_mask]
        cluster_point_indices = point_indices_[residual_mask]
        camera_indices_in_cluster_.append(cluster_camera_indices)
        point_indices_in_cluster_.append(cluster_point_indices)
        points_2d_in_cluster_.append(points_2d_[residual_mask])

        unique_points, point_degrees = np.unique(cluster_point_indices, return_counts=True)
        point_cluster_occurrences[unique_points] += 1
        degree_histogram = np.bincount(point_degrees, minlength=10)
        low_degree_counts += degree_histogram[:10]

    camera_counts = np.bincount(camera_to_cluster, minlength=kClusters_)
    copied_points = np.maximum(point_cluster_occurrences - 1, 0).sum()
    print("camera hypergraph cameras per cluster:", camera_counts.tolist())
    print("camera hypergraph additional point copies:", int(copied_points))
    print("camera hypergraph point-cluster degree counts 1..9:",
          low_degree_counts[1:10].tolist())

    return (
        camera_indices_in_cluster_,
        point_indices_in_cluster_,
        points_2d_in_cluster_,
        kClusters_
    )

def cluster_by_landmark_clean(
    camera_indices_, points_2d_, point_indices_, kClusters_, n_cameras_, n_points_,
    residual_balance_slack=0.05, minimum_camera_landmarks=20,
    max_refinement_passes=10, repair_restart_interval=1,
    max_repair_work_per_phase=0, hard_group_max_cameras=0,
    optimize_max_camera_count=False
):
    camera_indices_list = camera_indices_.tolist()
    point_indices_list = point_indices_.tolist()
    c_camera_indices = (ctypes.c_int * len(camera_indices_list))(*camera_indices_list)
    c_point_indices = (ctypes.c_int * len(point_indices_list))(*point_indices_list)
    c_camera_indices_cpp = lib.new_vector_by_copy(c_camera_indices, len(c_camera_indices))
    c_point_indices_cpp = lib.new_vector_by_copy(c_point_indices, len(c_point_indices))
    landmark_to_cluster_cpp = lib.new_vector()

    try:
        status = lib.cluster_landmarks_clean(
            ctypes.c_int(kClusters_), ctypes.c_int(n_cameras_), ctypes.c_int(n_points_),
            ctypes.c_int(minimum_camera_landmarks),
            ctypes.c_int(max_refinement_passes),
            ctypes.c_int(repair_restart_interval),
            ctypes.c_int64(max_repair_work_per_phase),
            ctypes.c_int(hard_group_max_cameras),
            ctypes.c_bool(optimize_max_camera_count),
            ctypes.c_double(residual_balance_slack), c_camera_indices_cpp,
            c_point_indices_cpp, landmark_to_cluster_cpp)
        if status != 0:
            raise RuntimeError("clean landmark partitioning failed")
        landmark_to_cluster = fillPythonVecSimple(landmark_to_cluster_cpp)
    finally:
        lib.delete_vector(landmark_to_cluster_cpp)
        lib.delete_vector(c_point_indices_cpp)
        lib.delete_vector(c_camera_indices_cpp)

    if landmark_to_cluster.shape[0] != n_points_:
        raise RuntimeError("clean landmark partitioning returned an invalid assignment")

    camera_indices_in_cluster_ = []
    point_indices_in_cluster_ = []
    points_2d_in_cluster_ = []
    for cluster in range(kClusters_):
        residual_mask = landmark_to_cluster[point_indices_] == cluster
        camera_indices_in_cluster_.append(camera_indices_[residual_mask])
        point_indices_in_cluster_.append(point_indices_[residual_mask])
        points_2d_in_cluster_.append(points_2d_[residual_mask])

    return (
        camera_indices_in_cluster_,
        point_indices_in_cluster_,
        points_2d_in_cluster_,
        kClusters_
    )

def cluster_by_landmark_scalable(
    camera_indices_, points_2d_, point_indices_, kClusters_, n_cameras_, n_points_,
    residual_balance_slack=0.05, minimum_camera_landmarks=20,
    max_refinement_passes=2
):
    camera_indices_list = camera_indices_.tolist()
    point_indices_list = point_indices_.tolist()
    c_camera_indices = (ctypes.c_int * len(camera_indices_list))(*camera_indices_list)
    c_point_indices = (ctypes.c_int * len(point_indices_list))(*point_indices_list)
    c_camera_indices_cpp = lib.new_vector_by_copy(c_camera_indices, len(c_camera_indices))
    c_point_indices_cpp = lib.new_vector_by_copy(c_point_indices, len(c_point_indices))
    landmark_to_cluster_cpp = lib.new_vector()

    try:
        status = lib.cluster_landmarks_scalable(
            ctypes.c_int(kClusters_), ctypes.c_int(n_cameras_), ctypes.c_int(n_points_),
            ctypes.c_int(minimum_camera_landmarks),
            ctypes.c_int(max_refinement_passes),
            ctypes.c_double(residual_balance_slack), c_camera_indices_cpp,
            c_point_indices_cpp, landmark_to_cluster_cpp)
        if status != 0:
            raise RuntimeError("scalable landmark partitioning failed")
        landmark_to_cluster = fillPythonVecSimple(landmark_to_cluster_cpp)
    finally:
        lib.delete_vector(landmark_to_cluster_cpp)
        lib.delete_vector(c_point_indices_cpp)
        lib.delete_vector(c_camera_indices_cpp)

    if landmark_to_cluster.shape[0] != n_points_:
        raise RuntimeError("scalable landmark partitioning returned an invalid assignment")

    camera_indices_in_cluster_ = []
    point_indices_in_cluster_ = []
    points_2d_in_cluster_ = []
    for cluster in range(kClusters_):
        residual_mask = landmark_to_cluster[point_indices_] == cluster
        camera_indices_in_cluster_.append(camera_indices_[residual_mask])
        point_indices_in_cluster_.append(point_indices_[residual_mask])
        points_2d_in_cluster_.append(points_2d_[residual_mask])

    return (
        camera_indices_in_cluster_,
        point_indices_in_cluster_,
        points_2d_in_cluster_,
        kClusters_
    )

def cluster_by_landmark_scalable_stable(
    camera_indices_, points_2d_, point_indices_, kClusters_, n_cameras_, n_points_,
    residual_balance_slack=0.05, minimum_camera_landmarks=20,
    max_refinement_passes=2
):
    camera_indices_list = camera_indices_.tolist()
    point_indices_list = point_indices_.tolist()
    c_camera_indices = (ctypes.c_int * len(camera_indices_list))(*camera_indices_list)
    c_point_indices = (ctypes.c_int * len(point_indices_list))(*point_indices_list)
    c_camera_indices_cpp = lib.new_vector_by_copy(c_camera_indices, len(c_camera_indices))
    c_point_indices_cpp = lib.new_vector_by_copy(c_point_indices, len(c_point_indices))
    landmark_to_cluster_cpp = lib.new_vector()

    try:
        status = lib.cluster_landmarks_scalable_stable(
            ctypes.c_int(kClusters_), ctypes.c_int(n_cameras_), ctypes.c_int(n_points_),
            ctypes.c_int(minimum_camera_landmarks),
            ctypes.c_int(max_refinement_passes),
            ctypes.c_double(residual_balance_slack), c_camera_indices_cpp,
            c_point_indices_cpp, landmark_to_cluster_cpp)
        if status != 0:
            raise RuntimeError("stability-focused scalable landmark partitioning failed")
        landmark_to_cluster = fillPythonVecSimple(landmark_to_cluster_cpp)
    finally:
        lib.delete_vector(landmark_to_cluster_cpp)
        lib.delete_vector(c_point_indices_cpp)
        lib.delete_vector(c_camera_indices_cpp)

    if landmark_to_cluster.shape[0] != n_points_:
        raise RuntimeError("stability-focused scalable partitioning returned an invalid assignment")

    camera_indices_in_cluster_ = []
    point_indices_in_cluster_ = []
    points_2d_in_cluster_ = []
    for cluster in range(kClusters_):
        residual_mask = landmark_to_cluster[point_indices_] == cluster
        camera_indices_in_cluster_.append(camera_indices_[residual_mask])
        point_indices_in_cluster_.append(point_indices_[residual_mask])
        points_2d_in_cluster_.append(points_2d_[residual_mask])

    return (
        camera_indices_in_cluster_,
        point_indices_in_cluster_,
        points_2d_in_cluster_,
        kClusters_
    )

lib = ctypes.CDLL("../libprocess_clusters.so")
init_lib()
