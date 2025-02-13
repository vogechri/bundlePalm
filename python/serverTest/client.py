"""
Original code from https://zeromq.org/languages/python/
"""

import zmq
from proto import test_pb2 #import ImageVector #, Image 
import numpy as np
context = zmq.Context()

import bz2

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
FILE_NAME = "../problem-52-64053-pre.txt.bz2"
FILE_NAME = "../problem-173-111908-pre.txt.bz2" # check if compute not only in jacobian

cameras, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

# preconditioner?

# program = test_pb2.program_proto()
# program.cameras[:] = cameras.ravel()
# program.landmarks[:] = points_3d.ravel()
# program.observations[:] = points_2d.ravel()
# program.cam_id[:] = camera_indices.ravel()
# program.lm_id[:] = point_indices.ravel()
# program.iterations = 1

###########################################

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
request.program.landmarks[:] = points_3d.ravel()
request.program.observations[:] = points_2d.ravel()
request.program.cam_id[:] = camera_indices.ravel()
request.program.lm_id[:] = point_indices.ravel()
request.program.iterations = 10

request_serialized = request.SerializeToString()
socket.send(request_serialized) # ? HOW THE FUCK DOES IT KNOW WHAT MESSAGE TYPE IT IS?

# get program back.
message_in_bytes = socket.recv()
program_deserialized = test_pb2.program_proto()
program_deserialized.ParseFromString(message_in_bytes)

print(0, " cameras " , program_deserialized.cameras[0:9])
# how to defuse oneof return:
# field = config.WhichOneof('config')
# if field = 'name_of_message?': ..

#################################
# next iteration. send cameras again, receive update, etc.
request = test_pb2.request_proto()
#cameras = test_pb2.camera_proto()
request.cameras.cameras[:] = program_deserialized.cameras[:] #?
for i in range(5):
    request_serialized = request.SerializeToString()
    socket.send(request_serialized) # ? HOW THE FUCK DOES IT KNOW WHAT MESSAGE TYPE IT IS?

    message_in_bytes = socket.recv()
    request.cameras.ParseFromString(message_in_bytes)
    print(i, " cameras = " , request.cameras.cameras[0:9])
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