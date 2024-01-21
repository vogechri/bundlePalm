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

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/trafalgar/"
# FILE_NAME = "problem-21-11315-pre.txt.bz2"
FILE_NAME = "problem-257-65132-pre.txt.bz2" # 193123.30793304814
# n_cameras: 257
# n_points: 65132
# Total number of parameters: 197709
# Total number of residuals: 451822
# Residuum initial  [ 67.42412924  21.14971817  -0.30177141 ...   1.62791101 -25.82724353
#  -21.90961561]   (451822,)
# Residuum equals initially  49124846.839772195
#    Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality
#        0              1         2.4562e+07                                    1.43e+08
# ^[[A^[[B       1              2         8.6002e+05      2.37e+07       2.70e+07       5.78e+08
#        2              3         3.1974e+05      5.40e+05       7.49e+06       9.75e+07
#        3              4         1.0676e+05      2.13e+05       2.05e+06       8.22e+06
#        4              5         9.7804e+04      8.95e+03       4.00e+05       9.74e+05
#        5              6         9.7031e+04      7.73e+02       5.00e+05       3.01e+05
#        6              7         9.6948e+04      8.25e+01       4.00e+05       6.81e+04
#        7              9         9.6855e+04      9.36e+01       1.40e+05       8.09e+04
#        8             10         9.6809e+04      4.55e+01       3.14e+05       2.69e+04
#        9             11         9.6794e+04      1.51e+01       3.00e+05       3.00e+04
#       10             12         9.6782e+04      1.24e+01       3.35e+05       2.78e+04
#       11             13         9.6770e+04      1.24e+01       2.75e+05       3.25e+04
#       12             14         9.6741e+04      2.83e+01       5.95e+04       1.16e+04
#       13             15         9.6730e+04      1.08e+01       2.46e+05       1.55e+04
#       14             16         9.6713e+04      1.79e+01       5.52e+05       8.42e+04
#       15             18         9.6705e+04      7.44e+00       1.56e+05       3.28e+04
#       16             19         9.6693e+04      1.25e+01       3.74e+05       4.35e+04
    #   17             20         9.6669e+04      2.36e+01       7.48e+05       1.46e+05
    #   18             21         9.6668e+04      5.65e-01       1.47e+06       5.77e+05
    #   19             22         9.6641e+04      2.75e+01       8.15e+04       1.60e+05
    #   20             23         9.6621e+04      2.00e+01       3.77e+04       4.16e+04
    #   21             24         9.6612e+04      9.18e+00       3.83e+05       5.01e+04
    #   22             25         9.6609e+04      3.21e+00       1.64e+05       5.39e+04
    #   23             26         9.6606e+04      2.30e+00       1.81e+05       3.33e+04
    #   24             28         9.6601e+04      5.34e+00       2.78e+04       3.46e+04
    #   25             29         9.6598e+04      3.10e+00       7.29e+04       4.43e+03
    #   26             30         9.6594e+04      3.87e+00       1.89e+05       1.70e+04
    #   27             32         9.6592e+04      2.23e+00       3.13e+04       1.27e+04
    #   28             33         9.6589e+04      2.42e+00       7.91e+04       3.49e+03
    #   29             34         9.6585e+04      4.21e+00       1.71e+05       9.18e+03
    #   30             36         9.6583e+04      1.70e+00       4.26e+04       3.40e+03
    #   31             37         9.6581e+04      2.70e+00       9.73e+04       3.40e+03
    #   32             38         9.6575e+04      5.32e+00       1.96e+05       1.14e+04
    #   33             39         9.6565e+04      1.03e+01       3.88e+05       4.61e+04
    #   34             41         9.6563e+04      2.48e+00       9.49e+04       1.16e+04
    #   35             42         9.6562e+04      9.57e-01       1.79e+05       5.44e+04
# `ftol` termination condition is satisfied.
# Function evaluations 42, initial cost 2.4562e+07, final cost 9.6562e+04, first-order optimality 5.44e+04.
# Optimization took 515 seconds
# Residuum final  [-0.61468939  0.57072604  0.61096167 ...  0.08962477 -0.00145848
#  -0.05093602]   (451822,)
# Residuum equals finally  193123.30793304814

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
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-5, method='trf', # test method = 'lm' or 'dogbox'
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
t1 = time.time()

print("Optimization took {0:.0f} seconds".format(t1 - t0))
#plt.plot(res.fun)
print("Residuum final ", res.fun, " ", res.fun.shape)
print("Residuum equals finally ", np.sum(res.fun**2))
