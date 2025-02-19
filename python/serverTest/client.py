"""
Original code from https://zeromq.org/languages/python/
"""

import zmq
#from proto import test_pb2 #import ImageVector #, Image
import sys
sys.path.insert(0, './generated/proto/')
#from test import test_pb2
import test_pb2

import numpy as np
context = zmq.Context()

import bz2
import time

from clustering import init_lib, cluster_deg_by_landmark
from scipy.sparse import csr_array, csr_matrix, issparse
from numpy.linalg import inv as inv_nonHermetian

# download ceres, edit CMakeList EXPORT_dir : On, cmake ../ceres-solver-2.2.0

# sudo apt-get install libeigen3-dev
# sudo apt install libzmq3-dev
# sudo apt install protobuf-compiler
# pip3 install zmq

# pip install protobuf==3.20.3
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

    # invert points_2d_ and focal distance if needed
    (camera_params, points_2d_) = \
        invert_focal_distance(camera_params, camera_indices_, points_2d_)

    return camera_params, points_3d_, camera_indices_, point_indices_, points_2d_

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

def average_cameras_new(
    camera_indices_in_cluster_, poses_in_cluster_, poses_s_in_cluster_, L_in_cluster_, UL_in_cluster_, nabla_p_in_cluster_):
    num_cameras = poses_in_cluster_[0].shape[0]
    sum_D_u2_s = np.zeros(num_cameras * 9)
    sum_constant_term = 0
    UL_zeros_in_cluster_ = []

    # Here or per part.
    # compressedData = False # idea we would send a compressed version of the stepsize.
    # if compressedData:
    #     for i in range(len(UL_in_cluster_)):
    #         UL_in_cluster_[i] = PostCompressBlockMatrix(UL_in_cluster_[i], 9) # this would be send in quantized form. We would need ensure its spd.

    for i in range(len(UL_in_cluster_)):
        # Lc = L_in_cluster_[i]
        camera_indices_ = np.unique(camera_indices_in_cluster_[i])
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

    return pose_v_out.reshape(num_cameras, 9), Up_all, UL_zeros_in_cluster_


# pollin: blocking receive for parellel message receiving. low cpu
# get program back. right now we also need landmarks. Likely not needed in smart implementation.
# We need to eval cost / send back cost. problem f(v) also needed now. DRE as well.
# master: compute fv, send s = 2u-v -> send v to slaves, they send cost back (need anyway to do step).
# can send both f(v) and f(u)! can do acceleration locally i guess or with minimal information.
def prox_f(camera_indices_in_cluster_, point_indices_in_cluster_, local_camera_indices_in_cluster_,
           local_landmark_indices_in_cluster_, points_2d_in_cluster_, poses_in_cluster_, landmarks_,
           poses_s_in_cluster_, L_in_cluster_, Vl_in_cluster_, blockEig_in_cluster_, kClusters_,
           LipJ_, innerIts_=1, sequential_=True) :
    cost_ = np.zeros(kClusters_)
    nabla_p_in_cluster_ = [0 for _ in range(kClusters_)]

    # ignore for now:
    # num_poses = poses_in_cluster_[0].shape[0]
    # pose_occurences = np.zeros(num_poses)
    # for ci_ in range(kClusters):
    #     unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci_])
    #     pose_occurences[unique_poses_in_c_] +=1

    global global_iteration
    global socket
    for ci in range(kClusters):
        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci])
        if global_iteration == 0:
            # print("Sending program …", ci)
            request = test_pb2.request_proto()
            #request.program.SetInParent()
            #program = request.program
            request.program.cameras[:] = poses_in_cluster_[ci].ravel()
            #request.program.cameras_s[:] = cameras.ravel() # set later in prox_cluster_proto
            request.program.landmarks[:] = landmarks_[unique_points_in_c_].ravel()
            request.program.observations[:] = points_2d_in_cluster_[ci].ravel()
            request.program.cam_id[:] = local_camera_indices_in_cluster_[ci].ravel()
            request.program.lm_id[:] = local_landmark_indices_in_cluster_[ci].ravel()
            request.program.iterations = innerIts_
            request.program.be = blockEig_in_cluster_[ci]
            request.program.cluster_id = ci
            request.program.init_l = LipJ_
            #request.unorm
            #request.vnorm
        else: # just update
            # print("Sending request …", ci)
            request = test_pb2.request_proto()
            #cameras = test_pb2.camera_proto()
            #temp = program_deserialized.cameras[:]
            #temp = [i * 10 for i in temp]
            #request.cameras.cameras[:] = temp # ok, program works with changed data.
            request.update.cameras[:] = poses_in_cluster_[ci].ravel()
            request.update.cameras_s[:] = poses_s_in_cluster_[ci].ravel()
            request.update.be = blockEig_in_cluster_[ci]
            request.update.cluster_id = ci

        request_serialized_ = request.SerializeToString()
        # request_serialized_ = request.SerializeToArray() #?
        socket.send(request_serialized_) # ? HOW THE FUCK DOES IT KNOW WHAT MESSAGE TYPE IT IS?

        message_in_bytes_ = socket.recv()
        return_proto_ = test_pb2.return_cluster_proto()
        return_proto_.ParseFromString(message_in_bytes_)
        # return_proto_.ParseFromArray(message_in_bytes_)

        # output should be:
        cost_[ci] = return_proto_.cost
        # L_in_cluster_[ci] = LipJ_ # unsused anyway
        Vl_in_cluster_[ci] = np.array(return_proto_.step_size[:]) # stepsize
        poses_in_cluster_[ci][unique_poses_in_c_, :] = np.array(return_proto_.cameras[:]).reshape((-1, 9))
        landmarks_[unique_points_in_c_,:] = np.array(return_proto_.landmarks[:]).reshape((-1, 3))
        #blockEig_in_cluster_[ci] = blockEig_in_c_ # not done

    global_iteration = global_iteration + 1
    return (cost_, L_in_cluster_, Vl_in_cluster_, poses_in_cluster_, landmarks_, nabla_p_in_cluster_, blockEig_in_cluster_)

# Operates sequentially.
def prox_f_push_pull(camera_indices_in_cluster_, point_indices_in_cluster_, local_camera_indices_in_cluster_,
           local_landmark_indices_in_cluster_, points_2d_in_cluster_, poses_in_cluster_, landmarks_,
           poses_s_in_cluster_, L_in_cluster_, Vl_in_cluster_, blockEig_in_cluster_, kClusters_,
           LipJ_, innerIts_=1, sequential_=True) :
    cost_ = np.zeros(kClusters_)
    nabla_p_in_cluster_ = [0 for _ in range(kClusters_)]

    # ignore for now:
    # num_poses = poses_in_cluster_[0].shape[0]
    # pose_occurences = np.zeros(num_poses)
    # for ci_ in range(kClusters):
    #     unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci_])
    #     pose_occurences[unique_poses_in_c_] +=1

    global global_iteration
    global push_socket
    global pull_socket

    for ci in range(kClusters):
        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci])
        if global_iteration == 0:
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
            request.program.cluster_id = ci
            request.program.init_l = LipJ_
            #request.unorm
            #request.vnorm
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

        request_serialized_ = request.SerializeToString() # SerializeToArray() does not exist
        push_socket.send(request_serialized_)
        temp = push_socket.recv() # ok back, blocking to wait for thread start.
        # do i need to send back a 'yes'?/ack?

    for k in range(kClusters):
        #print("Receiving return …", k)
        message_in_bytes_ = pull_socket.recv()
        return_proto_ = test_pb2.return_cluster_proto()

        #message_out_str = "Ok" # this might not be needed if this socket is pull not REC
        #message_out_bytes = message_out_str.encode("utf-8")
        #pull_socket.send(message_out_bytes)

        return_proto_.ParseFromString(message_in_bytes_)# ParseFromArray(message_in_bytes_)
        ci = return_proto_.cluster_id
        #print("Return for cluster …", ci)

        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        unique_poses_in_c_ = np.unique(camera_indices_in_cluster_[ci])
        cost_[ci] = return_proto_.cost
        # L_in_cluster_[ci] = LipJ_ # unsused anyway
        Vl_in_cluster_[ci] = np.array(return_proto_.step_size[:]) # stepsize
        poses_in_cluster_[ci][unique_poses_in_c_, :] = np.array(return_proto_.cameras[:]).reshape((-1, 9))
        landmarks_[unique_points_in_c_,:] = np.array(return_proto_.landmarks[:]).reshape((-1, 3))
        #blockEig_in_cluster_[ci] = blockEig_in_c_ # not done

    global_iteration = global_iteration + 1
    # print("exit prox_f")
    return (cost_, L_in_cluster_, Vl_in_cluster_, poses_in_cluster_, landmarks_, nabla_p_in_cluster_, blockEig_in_cluster_)

def GetLocalIndices(point_indices_in_cluster, camera_indices_in_cluster):
    # test, yes much faster if precompute:
    local_landmark_indices_in_cluster = [] # for residuals in cluster. local indices for landmark data send to cluster.
    for ci in range(kClusters):
        # can be used to index out global to local data. local/cluster = global[landmark_indices_in_c_]
        landmark_indices_in_c_ = np.unique(point_indices_in_cluster[ci])
        # print("local landmarks in ", ci, " " ,landmark_indices_in_c_.shape[0])
        #landmarks_in_c = landmarks_[landmark_indices_in_c_]
        # or global[landmark_indices_in_c_]  = local
        local_landmark_indices_in_cluster.append(np.zeros(point_indices_in_cluster[ci].shape[0], dtype=int))
        for i in range(landmark_indices_in_c_.shape[0]): # TODO: precompute THESE: slow!
            local_landmark_indices_in_cluster[ci][point_indices_in_cluster[ci] == landmark_indices_in_c_[i]] = i

    local_camera_indices_in_cluster = [] # for residuals in cluster. local indices for pose data send to cluster.
    for ci in range(kClusters):
        # can be used to index out global to local data. local/cluster = global[landmark_indices_in_c_]
        # or global[landmark_indices_in_c_]  = local
        cameras_indices_in_c_ = np.unique(camera_indices_in_cluster[ci])
        # print("local cameras in ", ci, " " ,cameras_indices_in_c_.shape[0])
        local_camera_indices_in_cluster.append( np.zeros(camera_indices_in_cluster[ci].shape[0], dtype=int) )
        for i in range(cameras_indices_in_c_.shape[0]): # TODO: precompute these, now if many cams this is slow ?!
            local_camera_indices_in_cluster[ci][camera_indices_in_cluster[ci] == cameras_indices_in_c_[i]] = i
        # print(local_camera_indices_in_cluster[ci])

    return (local_landmark_indices_in_cluster, local_camera_indices_in_cluster)

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "../problem-49-7776-pre.txt.bz2"
#FILE_NAME = "../problem-52-64053-pre.txt.bz2"
# FILE_NAME = "../problem-173-111908-pre.txt.bz2" # check if compute not only in jacobian

cameras, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)
n_cameras = cameras.shape[0]
n_points = points_3d.shape[0]

# simple! clustering
kClusters = 10 # todo: will still die if too many (0 in jac?)
startL = 1
innerIts = 1
LipJ = 1 # unused
global_iteration = 0
global_iterations = 10

#  Connect to the server
print("Connecting to cpp server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# new idea.
push_socket = context.socket(zmq.REQ)#PUSH)
push_socket.connect("tcp://localhost:5556")
#pull_socket = context.socket(zmq.REP)#.PULL)
pull_socket = context.socket(zmq.PULL)
pull_socket.connect("tcp://localhost:5557")

#lib = ctypes.CDLL("./libprocess_clusters.so")
init_lib()
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

(local_landmark_indices_in_cluster, local_camera_indices_in_cluster) = \
    GetLocalIndices(point_indices_in_cluster, camera_indices_in_cluster)

for ci in range(kClusters):
    values, counts = np.unique(camera_indices_in_cluster[ci], return_counts=True)
    print(ci, ". minimum camera observations in cluster ", np.min(counts), " cams with < 5 landmarks ", np.sum(counts < 5))
# preconditioner?

# 1.st version, implement prox_f. extension 1. polling and threads on server side. parameter k threads, etc.
L_in_cluster = [startL for _ in range(kClusters)]
Ul_in_cluster = [0 for _ in range(kClusters)]
blockEig_in_cluster = [1e-4 for _ in range(kClusters)]
# Unorm/Vnorm missing
# fairly stupid
poses_s_in_cluster = [cameras.copy() for _ in range(kClusters)]
poses_in_cluster = [cameras.copy() for _ in range(kClusters)]
landmarks = points_3d.copy()

lastCost = 1e12 # tem
lastCostDRE = lastCost

for global_iteration in range(global_iterations):

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
        LipJ, innerIts_=innerIts, sequential_=True,
        )
    end = time.time() # this is not working at all. Slower then iteratively

    if global_iteration == 1:
        print("landmarks ", landmarks)
        for ci in range(kClusters):
            print("poses_in_cluster ", ci, " : ", poses_in_cluster[ci])

    currentCost = np.sum(cost)
    print(global_iteration, " ", round(currentCost), " gain ", round(lastCost - currentCost),
          ". ============= sum fk update takes ", end - start," s",)
    poses_v, _, Up_cluster = average_cameras_new(camera_indices_in_cluster, poses_in_cluster,
                                                 poses_s_in_cluster, L_in_cluster, Ul_in_cluster, nabla_p_in_cluster)

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

    # primal_cost_v = 0
    # for ci in range(kClusters):
    #     primal_cost_v += primal_cost(
    #         poses_v,
    #         camera_indices_in_cluster[ci],
    #         point_indices_in_cluster[ci],
    #         local_camera_indices_in_cluster[ci],
    #         local_landmark_indices_in_cluster[ci],
    #         points_2d_in_cluster[ci],
    #         landmarks)
    # primal_cost_u = 0
    # for ci in range(kClusters):
    #     primal_cost_u += primal_cost(
    #         poses_in_cluster[ci],
    #         camera_indices_in_cluster[ci],
    #         point_indices_in_cluster[ci],
    #         local_camera_indices_in_cluster[ci],
    #         local_landmark_indices_in_cluster[ci],
    #         points_2d_in_cluster[ci],
    #         landmarks)
    primal_cost_u = currentCost
    primal_cost_v = primal_cost_u # must send.

    dre = max( primal_cost_v, dre ) # sandwich lemma, prevent maybe chaos
    print( global_iteration, " ======== DRE ====== ", round(dre) , " ========= gain " , \
        round(lastCostDRE - dre), "==== f(v)= ", round(primal_cost_v), " f(u)= ",
        round(primal_cost_u), " BE ", blockEig_in_cluster)

    if lastCostDRE < dre:
        #LipJ += 0.2 * np.ones(kClusters)
        partid = np.argmax(dre_per_part)
        blockEig_in_cluster[partid] = np.minimum(blockEig_in_cluster[partid] * np.sqrt(2), 1e-0)
    # if f(u) < f(v) also raise? not lip but smth else.

    lastCost = currentCost
    lastCostDRE = dre

###########################################
# updateCluster(
#             poses_in_cluster_[ci_],
#             camera_indices_in_cluster_[ci_],
#             point_indices_in_cluster_[ci_],
#             local_camera_indices_in_cluster_[ci],
#             local_landmark_indices_in_cluster_[ci_],
#             points_2d_in_cluster_[ci_],
#             landmarks_,
#             poses_s_in_cluster_[ci_],
#             Vl_in_cluster_[ci_],
#             # np.max(L_in_cluster_), #L_in_cluster_[ci_], # AAA
#             L_in_cluster_[ci_],
#             pose_occurences, # haeh?
#             LipJ[ci_],
#             blockEig_in_cluster_[ci_],
#             ci_,
#             its_=innerIts,
#         )

# landmark_indices_in_c_
# local_landmark_indices_in_cluster # precomputed: for residuals, order changed due to exclusive clustering
# point_indices_in_cluster[ci] == landmark_indices_in_cluster_, the clustered indices of the residuals.
# landmark_indices_in_c_ = np.unique(landmark_indices_in_cluster_) # used for global -> local ids == VNorm[landmark_indices_in_c_], landmarks_[landmark_indices_in_c_]
# landmarks_in_c = landmarks_[landmark_indices_in_c_] # the 3d landmarks in a cluster (global -> local)
# VNorm[landmark_indices_in_c_]
# for landmarks we need
#  local landmarks 3d = landmarks_in_c, indices = local_landmark_indices_in_cluster. VNorm[landmark_indices_in_c_]

# for poses we need
# local_camera_indices_in_cluster[ci] # precomputed for residuals
# cameras_indices_in_c_ = np.unique(camera_indices_in_cluster[ci]) # global -> local.
# poses_in_c = poses_in_cluster_[cameras_indices_in_c_] # poses: global -> local
# poses_s_in_c = poses_s_in_cluster_[cameras_indices_in_c_] # poses_s: global -> local
# UNorm[cameras_indices_in_c_]
# local poses 9d = poses_in_c/poses_s_in_c, indices = local_camera_indices_in_cluster. UNorm[cameras_indices_in_c_], poses_only_in_cluster_: optional -> step[poses_only_in_cluster_] = 1e-16
#
# 2d reprojections per cluster

# send
if False:
    cameras[ci] = cameras[np.unique(camera_indices_in_cluster[ci])]
    landmarks = points_3d[np.unique(point_indices_in_cluster[ci])]
    observations = points_2d_in_cluster[ci]
    cam_id = local_camera_indices_in_cluster[ci]
    lm_id = local_landmark_indices_in_cluster[ci]
# later
#unorm
#vnorm

# and back (ugly is to rebuild the stepsize matrix) especially to sum up and invert:
# maybe explicitly, c++ would be simple loop and add up by
# cameras[np.unique(camera_indices_in_cluster[ci][k])] = mat[ci][k] * pose[ci][k]
#
# Can I send back data, indices, indptr -- rather row ind col ind
# indptr [81 * j +  9, 81 * j + 18, 81 * j + 27, 81 * j + 36, 81 * j + 45, 81 * j + 54, 81 * j + 63, 81 * j + 72, 81 * j + 81]
# indices = 9 * camera_indices_ + j, 9 * camera_indices_ + j, ..
#
# another possibility is
# for row i indices are stored as indices[indptr[i]:indptr[i+1]] and
# values are stored in data[indptr[i]:indptr[i+1]].
# indices : in row 1, .. i, i+1, .. N. indptr is 0,9,18, .. empty rows as indptr n times -> empty block 9,9,9,9,9,9,9,9,9
# so i send values, indptr, indices. indices is also
# the uttermost problem is we must receive the data also in line and send it -- not parallel.

# original code uses UL_in_cluster_[i].data .. so we can send just that? convert to np.array with python code?

##############
# we get back: cameras_back, landmarks_back, stepSize
# ci
# cameras[np.unique(camera_indices_in_cluster[ci])] = cameras_back[ci]
# landmarks[np.unique(point_indices_in_cluster[ci])] = landmarks_back[ci]
# step_size.data

# the client should replace this function !?
# out:
#     primal_cost_: V
#     L_in_cluster_: X
#     Ul_in_cluster_: V
#     poses_in_cluster_: V
#     landmarks_: V
#     nabla_p_in_cluster_: X
#     blockEig_in_cluster__: X
# (
#     primal_cost_,
#     L_in_cluster_,
#     Ul_in_cluster_,
#     poses_in_cluster_,
#     landmarks_,
#     nabla_p_in_cluster_,
#     blockEig_in_cluster__
# ) = prox_f(
#     camera_indices_in_cluster_, point_indices_in_cluster_, local_camera_indices_in_cluster_, local_landmark_indices_in_cluster_,
#     points_2d_in_cluster_, poses_in_cluster_, landmarks_, poses_s_in_cluster_, L_in_cluster_, Ul_in_cluster_, blockEig_in_cluster__,
#     kClusters_, LipJ_, innerIts=innerIts_, sequential=True,
#     )
# in program:
# camera_indices_in_cluster_, point_indices_in_cluster_, local_camera_indices_in_cluster_,
# local_landmark_indices_in_cluster_, points_2d_in_cluster_, landmarks_
# in update:
# poses_in_cluster_, poses_s_in_cluster_, innerIts, be
#
# out cost ul poses, landmarks
#
# 1. cluster
# 2. pre ci send program. What to get back?
###################

#Send data
#message_out_str = "Hi from Python ZeroMQ Client"
#message_out_bytes = message_out_str.encode("utf-8")

#print("Sending request …")
#socket.send(message_out_bytes)
#message_in_bytes = socket.recv() # MUST receive something -- handshake? i do not understand.

#################################