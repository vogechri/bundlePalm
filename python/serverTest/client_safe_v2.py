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


BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "../problem-49-7776-pre.txt.bz2"
#FILE_NAME = "../problem-52-64053-pre.txt.bz2"
# FILE_NAME = "../problem-173-111908-pre.txt.bz2" # check if compute not only in jacobian

cameras, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)
n_cameras = cameras.shape[0]
n_points = points_3d.shape[0]

# simple! clustering
kClusters = 2

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

# test, yes much faster if precompute:
local_landmark_indices_in_cluster = [] # for residuals in cluster. local indices for landmark data send to cluster.
for ci in range(kClusters):
    # can be used to index out global to local data. local/cluster = global[landmark_indices_in_c_]
    landmark_indices_in_c_ = np.unique(point_indices_in_cluster[ci])
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
    local_camera_indices_in_cluster.append( np.zeros(camera_indices_in_cluster[ci].shape[0], dtype=int) )
    for i in range(cameras_indices_in_c_.shape[0]): # TODO: precompute these, now if many cams this is slow ?!
        local_camera_indices_in_cluster[ci][camera_indices_in_cluster[ci] == cameras_indices_in_c_[i]] = i

for ci in range(kClusters):
    values, counts = np.unique(camera_indices_in_cluster[ci], return_counts=True)
    print(ci, ". minimum camera observations in cluster ", np.min(counts), " cams with < 5 landmarks ", np.sum(counts < 5))
# preconditioner?
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

#  Connect to the server
print("Connecting to cpp server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

#Send data
message_out_str = "Hi from Python ZeroMQ Client"
message_out_bytes = message_out_str.encode("utf-8")

print("Sending request …")
socket.send(message_out_bytes)
message_in_bytes = socket.recv() # MUST receive something -- handshake? i do not understand.

#################################
print("Sending program …")
request = test_pb2.request_proto()
#request.program.SetInParent()
#program = request.program
request.program.cameras[:] = cameras.ravel()
#request.program.cameras_s[:] = cameras.ravel() # set later in prox_cluster_proto
request.program.landmarks[:] = points_3d.ravel()
request.program.observations[:] = points_2d.ravel()
request.program.cam_id[:] = camera_indices.ravel()
request.program.lm_id[:] = point_indices.ravel()
request.program.iterations = 1
request.program.be = 1e-5
global_iterations = 1
request_serialized = request.SerializeToString()
socket.send(request_serialized) # ? HOW THE FUCK DOES IT KNOW WHAT MESSAGE TYPE IT IS?

# get program back. right now we also need landmarks. Likely not needed in smart implementation.
# We need to eval cost / send back cost. problem f(v) also needed now. DRE as well.
# master: compute fv, send s = 2u-v -> send v to slaves, they send cost back (need anyway to do step).
# can send both f(v) and f(u)! can do acceleration locally i guess or with minimal information.
message_in_bytes = socket.recv()

return_proto = test_pb2.return_cluster_proto()
return_proto.ParseFromString(message_in_bytes)
print(-1, " cameras " , return_proto.cameras[0:9], " cost ", return_proto.cost)

#program_deserialized = test_pb2.program_proto()
#program_deserialized.ParseFromString(message_in_bytes)
#print(-1, " cameras " , program_deserialized.cameras[0:9])

# how to defuse oneof return:
# field = config.WhichOneof('config')
# if field = 'name_of_message?': ..

#################################
# next iteration. send cameras again, receive update, etc.
request = test_pb2.request_proto()
#cameras = test_pb2.camera_proto()
#temp = program_deserialized.cameras[:]
#temp = [i * 10 for i in temp]
#request.cameras.cameras[:] = temp # ok, program works with changed data.

request.update.cameras[:] = return_proto.cameras[:]
request.update.cameras_s[:] = return_proto.cameras[:]
request.update.be = 1e-5
request.update.cluster_id = 0
#request.cameras.cameras[:] = program_deserialized.cameras[:]
for i in range(global_iterations):
    request_serialized = request.SerializeToString()
    socket.send(request_serialized) # ? HOW THE FUCK DOES IT KNOW WHAT MESSAGE TYPE IT IS? -> oneof, case

    message_in_bytes = socket.recv()
    #return_proto = test_pb2.return_cluster_proto()
    return_proto.ParseFromString(message_in_bytes) # parse all data ?!
    #request.cameras.ParseFromString(message_in_bytes)
    print(i, " cameras = " , return_proto.cameras[0:9], " cost ", return_proto.cost)
    del request.update.cameras[:]
    request.update.cameras.extend(return_proto.cameras)
    #request.cameras.cameras[:] = cameras[:]

# Serialize the message to a string
#image_serialized = image.SerializeToString()
#print("Serialized message ", image_serialized)
# Deserialize the message from a string
#image_deserialized = test_pb2.Image()
#image_deserialized.ParseFromString(image_serialized)

# image_vector.images.ids.extend([1, 32, 43432])
# message.values.extend(numpy_array) ?
# image_vector.images.data

#Get the server response
# message_in_bytes = socket.recv()
# Convert message bytes to string
#message_in_str = message_in_bytes.decode("utf-8")
#print("Received [ %s ]" % message_in_str)