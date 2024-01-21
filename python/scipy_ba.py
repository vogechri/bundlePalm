from __future__ import print_function
import urllib
import urllib.request
import bz2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/venice/"
#FILE_NAME = "problem-52-64053-pre.txt.bz2"
FILE_NAME = "problem-1778-993923-pre.txt.bz2"

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/dubrovnik/"
#FILE_NAME = "problem-16-22106-pre.txt.bz2"
#FILE_NAME = "problem-356-226730-pre.txt.bz2"
FILE_NAME = "problem-173-111908-pre.txt.bz2"
# n_cameras: 173
# n_points: 111908
# Total number of parameters: 337281
# Total number of residuals: 1269140
# Residuum initial  [-7.42211774  8.14923825 -0.80491448 ... -0.25041295 -0.16280334
#   0.39991869]   (1269140,)
# Residuum equals initially  120750865.30234598
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality
#        0              1         6.0375e+07                                    1.47e+08
#        1              2         2.8597e+05      6.01e+07       2.91e+05       7.17e+06
#        2              3         2.6250e+05      2.35e+04       3.70e+05       6.22e+06
#        3              4         2.6215e+05      3.52e+02       4.15e+04       3.35e+05
#        4              5         2.6214e+05      1.08e+01       2.79e+04       5.27e+04
#        5              6         2.6213e+05      9.06e+00       2.68e+04       7.95e+03
#        6              7         2.6212e+05      4.56e+00       1.98e+04       4.55e+03
#        7              8         2.6212e+05      4.49e+00       2.05e+04       4.00e+03
#        8              9         2.6212e+05      3.37e+00       1.73e+04       2.70e+03
#        9             10         2.6211e+05      2.95e+00       1.29e+04       1.67e+03
#       10             11         2.6211e+05      1.62e+00       4.06e+03       1.32e+03
# `ftol` termination condition is satisfied.
# Function evaluations 11, initial cost 6.0375e+07, final cost 2.6211e+05, first-order optimality 1.32e+03.
# Optimization took 331 seconds
# Residuum final  [ 0.79748542 -0.09474595  0.10791841 ...  0.00582406  0.00141867
#  -0.00343391]   (1269140,)
# Residuum equals finally  524221.45726861514

URL = BASE_URL + FILE_NAME

if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)

def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d

camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
#plt.plot(f0)

print("Residuum initial ", f0, " ", f0.shape)
print("Residuum equals initially ", np.sum(f0**2))

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

t0 = time.time()
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-5, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
t1 = time.time()

print("Optimization took {0:.0f} seconds".format(t1 - t0))
#plt.plot(res.fun)
print("Residuum final ", res.fun, " ", res.fun.shape)
print("Residuum equals finally ", np.sum(res.fun**2))
