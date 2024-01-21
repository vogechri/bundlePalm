from __future__ import print_function
import urllib
import bz2
import os
import numpy as np
import time
from joblib import Parallel, delayed
from scipy.sparse import csr_array, csr_matrix
from scipy.sparse import diags as diag_sparse
from scipy.sparse.linalg import inv as inv_sparse
from numpy.linalg import inv as inv_dense

# idea reimplement projection with torch to get a jacobian -> numpy then 
import torch
import math
from torch.autograd.functional import jacobian
from torch import tensor, from_numpy

# look at website. This is the smallest problem. guess: pytoch cpu is pure python? 
BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
#FILE_NAME = "problem-49-7776-pre.txt.bz2"
#FILE_NAME = "problem-73-11032-pre.txt.bz2"
FILE_NAME = "problem-138-19878-pre.txt.bz2"
FILE_NAME = "problem-646-73584-pre.txt.bz2"

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/trafalgar/"
# FILE_NAME = "problem-21-11315-pre.txt.bz2"
FILE_NAME = "problem-257-65132-pre.txt.bz2"

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/dubrovnik/"
#FILE_NAME = "problem-16-22106-pre.txt.bz2"
FILE_NAME = "problem-356-226730-pre.txt.bz2"
#FILE_NAME = "problem-237-154414-pre.txt.bz2"
FILE_NAME = "problem-173-111908-pre.txt.bz2" # ex where power its are worse


#BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/venice/"
#FILE_NAME = "problem-52-64053-pre.txt.bz2"
#FILE_NAME = "problem-1778-993923-pre.txt.bz2"

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/final/"
FILE_NAME = "problem-93-61203-pre.txt.bz2"
FILE_NAME = "problem-871-527480-pre.txt.bz2"
#FILE_NAME = "problem-394-100368-pre.txt.bz2"

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/trafalgar/"
# FILE_NAME = "problem-21-11315-pre.txt.bz2"
FILE_NAME = "problem-257-65132-pre.txt.bz2"
#
# solver -> super slow cholesky is more accurate. Why? float-?double or need more iterations?
# better stopping crit?
# TODO: understand why not working. scipy -> 193k
# Lm step took  1.0648386478424072 s
# 99 it. cost 0      198024
# 99 it. cost 0/new  198023  cost with penalty  198023
# 99 it. cost 1      198023       with penalty  198023  cost compute took  0.10829901695251465 s
# c  [ 0.01516885 -0.02060484  0.00302279  0.98726985  0.05791101 -0.04276767]   (6,)
# 100 it. cost ext    198023       with penalty  198023
# 100 it. cost ext    197271       with penalty  197271 # 50 its, stop at 1e-6 (==off?)
# 100 it. cost ext    196614       with penalty  196614 with extrapolation
# 61 it. cost 1      197215       with penalty  197216  cost compute took  0.10839438438415527 s
# todo: manipulate / copy make test with 10 & compare
# need to solve hanging (n times increasing L, failing TR)
# powerits = 100 Lm step took  2.429298162460327 s, 46 it. cost 1      196634
# 99 it. cost 1      195253       with penalty  195254  cost compute took  0.10700201988220215 s
# powerits = 200: 99 it. cost 1      193227

URL = BASE_URL + FILE_NAME
old_c = 0

if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)

def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2), dtype=np.float64)

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9, dtype=np.float64)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3, dtype=np.float64)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d

def RotationMatrixToQuaternion(R):
  trace = R[0, 0] + R[1, 1] + R[2, 2]
  q = torch.zeros(4, dtype=torch.float64)
  if trace > 0.0:
    t = math.sqrt(trace + 1.0)
    q[0] = 0.5 * t
    t = 0.5 / t
    q[1] = (R[2, 1] - R[1, 2]) * t
    q[2] = (R[0, 2] - R[2, 0]) * t
    q[3] = (R[1, 0] - R[0, 1]) * t
  else:
    i = 0
    if R[1, 1] > R[0, 0]:
      i = 1
    if R[2, 2] > R[i, i]:
      i = 2

    j = (i + 1) % 3
    k = (j + 1) % 3
    t = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1)

    q[i + 1] = 0.5 * t
    t = 0.5 / t
    q[0] = (R[k, j] - R[j, k]) * t
    q[j + 1] = (R[j, i] + R[i, j]) * t
    q[k + 1] = (R[k, i] + R[i, k]) * t
  return q

def QuaternionToAngleAxis(Q):
  q1 = Q[1]
  q2 = Q[2]
  q3 = Q[3]
  sinSquaredTheta = q1 * q1 + q2 * q2 + q3 * q3
  if sinSquaredTheta > 0:
    sinTheta = math.sqrt(sinSquaredTheta)
    cosTheta = Q[0]
    if cosTheta < 0:
      twoTheta = np.arctan2(-sinTheta, -cosTheta) * 2
    else:
      twoTheta = np.arctan2(sinTheta, cosTheta) * 2
    k = twoTheta / sinTheta
  else:
    k = 2
  return torch.from_numpy(np.array([q1 * k, q2 * k, q3 * k]))

def AngleAxisToRotationMatrix(angleAxis):
  kOne = 1.0
  theta2 = torch.sum(angleAxis * angleAxis, dim=1).view(-1, 1)

  mask0 = (theta2 > 0).float()

  theta = torch.sqrt(theta2 + (1 - mask0))
  wx = angleAxis[:,0:1] / theta
  wy = angleAxis[:,1:2] / theta
  wz = angleAxis[:,2:3] / theta
  costheta = torch.cos(theta)
  sintheta = torch.sin(theta)

  R00 =     (costheta   + wx*wx*(kOne -    costheta)) * mask0\
    + kOne * (1 - mask0)
  R10 =  (wz*sintheta   + wx*wy*(kOne -    costheta)) * mask0\
    + angleAxis[:,2:3] * (1 - mask0)
  R20 = (-wy*sintheta   + wx*wz*(kOne -    costheta)) * mask0\
    - angleAxis[:,1:2] * (1 - mask0)
  R01 =  (wx*wy*(kOne - costheta)     - wz*sintheta) * mask0\
    - angleAxis[:,2:3] * (1 - mask0)
  R11 =     (costheta   + wy*wy*(kOne -    costheta)) * mask0\
    + kOne * (1 - mask0)
  R21 =  (wx*sintheta   + wy*wz*(kOne -    costheta)) * mask0\
    + angleAxis[:,0:1] * (1 - mask0)
  R02 =  (wy*sintheta   + wx*wz*(kOne -    costheta)) * mask0\
    + angleAxis[:,1:2] * (1 - mask0)
  R12 = (-wx*sintheta   + wy*wz*(kOne -    costheta)) * mask0\
    - angleAxis[:,0:1] * (1 - mask0)
  R22 =     (costheta   + wz*wz*(kOne -    costheta)) * mask0\
    + kOne * (1 - mask0)

  return torch.cat((R00, R01, R02, R10, R11, R12, R20, R21, R22),\
    dim=1).view(-1, 3, 3)


def RotationMatrixToAngleAxis(R):
  res = torch.zeros(R.shape[0], 3, dtype=torch.float64)
  for i in range(R.shape[0]):
    res[i] = QuaternionToAngleAxis(RotationMatrixToQuaternion(R[i]))
  return res

def AngleAxisRotatePoint(angleAxis, pt):
  theta2 = (angleAxis * angleAxis).sum(dim=1)

  mask = (theta2 > 0).float() #? == 0 is alternative? check other repo

  theta = torch.sqrt(theta2 + (1 - mask) )

  mask = mask.reshape((mask.shape[0], 1))
  mask = torch.cat([mask, mask, mask], dim=1)

  costheta = torch.cos(theta)
  sintheta = torch.sin(theta)
  thetaInverse = 1.0 / theta

  w0 = angleAxis[:,0] * thetaInverse
  w1 = angleAxis[:,1] * thetaInverse
  w2 = angleAxis[:,2] * thetaInverse

  wCrossPt0 = w1 * pt[:,2] - w2 * pt[:,1]
  wCrossPt1 = w2 * pt[:,0] - w0 * pt[:,2]
  wCrossPt2 = w0 * pt[:,1] - w1 * pt[:,0]

  tmp = (w0 * pt[:,0] + w1 * pt[:,1] + w2 * pt[:,2]) * (1.0 - costheta)

  r0 = pt[:,0] * costheta + wCrossPt0 * sintheta + w0 * tmp
  r1 = pt[:,1] * costheta + wCrossPt1 * sintheta + w1 * tmp
  r2 = pt[:,2] * costheta + wCrossPt2 * sintheta + w2 * tmp

  r0 = r0.reshape((r0.shape[0], 1))
  r1 = r1.reshape((r1.shape[0], 1))
  r2 = r2.reshape((r2.shape[0], 1))
  
  res1 = torch.cat([r0, r1, r2], dim=1)

  wCrossPt0 = angleAxis[:,1] * pt[:,2] - angleAxis[:,2] * pt[:,1]
  wCrossPt1 = angleAxis[:,2] * pt[:,0] - angleAxis[:,0] * pt[:,2]
  wCrossPt2 = angleAxis[:,0] * pt[:,1] - angleAxis[:,1] * pt[:,0]

  r00 = pt[:,0] + wCrossPt0;
  r01 = pt[:,1] + wCrossPt1;
  r02 = pt[:,2] + wCrossPt2;

  r00 = r00.reshape((r00.shape[0], 1))
  r01 = r01.reshape((r01.shape[0], 1))
  r02 = r02.reshape((r02.shape[0], 1))

  res2 = torch.cat([r00, r01, r02], dim=1)

  return res1 * mask + res2 * (1 - mask)

def Normalize(points):
  l = torch.sqrt(torch.sum(points * points, dim=1) + 1e-10)
  l = l.reshape((l.shape[0], 1))
  l = torch.cat([l, l, l], dim=1)
  points_normalized = points / l

  return points_normalized

def EquirectangularProjection(points, width = 5760.0):
  pn = Normalize(points)
  lon = torch.atan2(pn[:,0], pn[:,2])
  hypot = torch.sqrt(pn[:,0]*pn[:,0] + pn[:,2]*pn[:,2])
  lat = torch.atan2(-pn[:,1], hypot)

  x = lon / (2.0 * np.pi)
  mask = (x < 0).float()
  x = mask * (-0.5 - x) + (1 - mask) * (0.5 - x)

  y = lat / (-2.0 * np.pi)

  x = x.reshape((x.shape[0], 1))
  y = y.reshape((y.shape[0], 1))

  return torch.cat([x,y], dim=1) * width

def torchResiduum(x0T, n_cameras, n_points, camera_indices, point_indices, p2d) :
    # x0T = from_numpy(x0)
    #p2d = from_numpy(points_2d) # outside is better?
    #p2d.requires_grad_(False)
    camera_params = x0T[:n_cameras*9].reshape(n_cameras,9)
    point_params  = x0T[n_cameras*9:].reshape(n_points,3)
    angle_axis = camera_params[:,:3]

    # likely better to create per point/cam representation 1st. no slower
    #rot_matrix = AngleAxisToRotationMatrix(angle_axis) # 
    #points_cam = rot_matrix[camera_indices, :, 0] * point_params[point_indices,0].view(-1,1) + rot_matrix[camera_indices, :, 1] * point_params[point_indices,1].view(-1,1) + rot_matrix[camera_indices, :, 2] * point_params[point_indices,2].view(-1,1) 

    points_cam = AngleAxisRotatePoint(angle_axis[camera_indices,:], point_params[point_indices,:])

    points_cam = points_cam + camera_params[camera_indices,3:6]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]

    f  = camera_params[camera_indices, 6]
    k1 = camera_params[camera_indices, 7]
    k2 = camera_params[camera_indices, 8]
    #f.requires_grad_(False) # not leaf?!
    #k1.requires_grad_(False)
    #k2.requires_grad_(False)
    r2 = points_projX*points_projX + points_projY*points_projY
    distortion = 1. + r2 * (k1 + k2 * r2)
    points_reprojX = -points_projX * distortion * f
    points_reprojY = -points_projY * distortion * f
    resX = (points_reprojX-p2d[:,0]).reshape((p2d.shape[0], 1))
    resY = (points_reprojY-p2d[:,1]).reshape((p2d.shape[0], 1))
    residual = torch.cat([resX[:,], resY[:,]], dim=1)
    return residual

def torchSingleResiduum(camera_params, point_params, p2d) :
    angle_axis = camera_params[:,:3]
    points_cam = AngleAxisRotatePoint(angle_axis, point_params)
    points_cam = points_cam + camera_params[:,3:6]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f  = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    r2 = points_projX*points_projX + points_projY*points_projY
    distortion = 1. + r2 * (k1 + k2 * r2)
    points_reprojX = points_projX * distortion * f
    points_reprojY = points_projY * distortion * f
    resX = (points_reprojX-p2d[:,0]).reshape((p2d.shape[0], 1))
    resY = (points_reprojY-p2d[:,1]).reshape((p2d.shape[0], 1))
    residual = torch.cat([resX[:,], resY[:,]], dim=1)
    return residual

def torchSingleResiduumX(camera_params, point_params, p2d) :
    angle_axis = camera_params[:,:3]
    points_cam = AngleAxisRotatePoint(angle_axis, point_params)
    points_cam = points_cam + camera_params[:,3:6]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f  = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    r2 = points_projX*points_projX + points_projY*points_projY
    distortion = 1. + r2 * (k1 + k2 * r2)
    points_reprojX = points_projX * distortion * f
    resX = (points_reprojX-p2d[:,0]) #.reshape((p2d.shape[0], 1))
    return resX

def torchSingleResiduumY(camera_params, point_params, p2d) :
    angle_axis = camera_params[:,:3]
    points_cam = AngleAxisRotatePoint(angle_axis, point_params)
    points_cam = points_cam + camera_params[:,3:6]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f  = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    r2 = points_projX*points_projX + points_projY*points_projY
    distortion = 1. + r2 * (k1 + k2 * r2)
    points_reprojY = points_projY * distortion * f
    resY = (points_reprojY-p2d[:,1])
    return resY

# import jax
# import jax.numpy as jnp

# def AngleAxisRotatePointJax(angleAxis, pt):
#   theta2 = jnp.sum(angleAxis * angleAxis, keepdims=True, axis=1) # sums out the dimension in numpy, argh
    
#   #mask = (theta2 > 0).float() #? == 0 is alternative? check other repo
#   mask = (theta2 > 0) * 1. #? == 0 is alternative? check other repo

#   theta = jnp.sqrt(theta2 + (1 - mask) )

#   #mask = mask.reshape((mask.shape[0], 1))
#   mask = jnp.concatenate([mask, mask, mask], axis=1)

#   costheta = jnp.cos(theta)
#   sintheta = jnp.sin(theta)
#   thetaInverse = 1.0 / theta
    
#   w0 = angleAxis[:,[0]] * thetaInverse
#   w1 = angleAxis[:,[1]] * thetaInverse
#   w2 = angleAxis[:,[2]] * thetaInverse

#   wCrossPt0 = w1 * pt[:,[2]] - w2 * pt[:,[1]]
#   wCrossPt1 = w2 * pt[:,[0]] - w0 * pt[:,[2]]
#   wCrossPt2 = w0 * pt[:,[1]] - w1 * pt[:,[0]]

#   tmp = (w0 * pt[:,[0]] + w1 * pt[:,[1]] + w2 * pt[:,[2]]) * (1.0 - costheta)

#   r0 = pt[:,[0]] * costheta + wCrossPt0 * sintheta + w0 * tmp
#   r1 = pt[:,[1]] * costheta + wCrossPt1 * sintheta + w1 * tmp
#   r2 = pt[:,[2]] * costheta + wCrossPt2 * sintheta + w2 * tmp

#   r0 = r0.reshape((r0.shape[0], 1))
#   r1 = r1.reshape((r1.shape[0], 1))
#   r2 = r2.reshape((r2.shape[0], 1))
  
#   res1 = jnp.concatenate([r0, r1, r2], axis=1)

#   wCrossPt0 = angleAxis[:,1] * pt[:,2] - angleAxis[:,2] * pt[:,1]
#   wCrossPt1 = angleAxis[:,2] * pt[:,0] - angleAxis[:,0] * pt[:,2]
#   wCrossPt2 = angleAxis[:,0] * pt[:,1] - angleAxis[:,1] * pt[:,0]

#   r00 = pt[:,0] + wCrossPt0;
#   r01 = pt[:,1] + wCrossPt1;
#   r02 = pt[:,2] + wCrossPt2;

#   r00 = r00.reshape((r00.shape[0], 1))
#   r01 = r01.reshape((r01.shape[0], 1))
#   r02 = r02.reshape((r02.shape[0], 1))

#   res2 = jnp.concatenate([r00, r01, r02], axis=1)

#   return res1 * mask + res2 * (1 - mask)

# def jaxResiduum(camera_params, point_params, p2d) :
#     angle_axis = camera_params[:,:3]
#     points_cam = AngleAxisRotatePointJax(angle_axis, point_params)
#     points_cam = points_cam + camera_params[:,3:6]
#     points_projX = -points_cam[:, 0] / points_cam[:, 2]
#     points_projY = -points_cam[:, 1] / points_cam[:, 2]
#     f  = camera_params[:, 6]
#     k1 = camera_params[:, 7]
#     k2 = camera_params[:, 8]
#     r2 = points_projX*points_projX + points_projY*points_projY
#     distortion = 1. + r2 * (k1 + k2 * r2)
#     points_reprojX = -points_projX * distortion * f
#     points_reprojY = -points_projY * distortion * f
#     resX = (points_reprojX-p2d[:,0]).reshape((p2d.shape[0], 1))
#     resY = (points_reprojY-p2d[:,1]).reshape((p2d.shape[0], 1))
#     residual = jnp.concatenate([resX[:,], resY[:,]], axis=1)
#     return residual

cameras, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

# # better define function. data in as batch -> normal gradient.
# # Dp this with the jacobian ?!
# # jac as function -> [which:0/1][var index, row, var_index, dim]
# # so maybe j(x) = jac(x)[0][i,:,i,:]
# # then call j(x)? no idea how it works. small example.
# # Create a tensor with some non-zero elements
# x = torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]]).flatten()
# # Get the indices of the non-zero elements
# indices = x.nonzero()
# # Use torch.index_add_ to increase the values of the non-zero elements by 1
# #torch.index_add_(x, 0, indices, torch.ones(indices.size(0)))
# print(x)  # tensor([[2, 0, 0], [0, 3, 0], [0, 0, 4]])
# print(x[indices[:]])
# print("indices: ", indices)

n_cameras = cameras.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))

x0_p = cameras.ravel()
x0_l = points_3d.ravel()
x0 = np.hstack((x0_p, x0_l), dtype=np.float64)
torch_points_2d = from_numpy(points_2d)
torch_points_2d.requires_grad_(False)
x0_t = from_numpy(x0)

if False:
    # idea compute in 500 steps, 
    numRes = 400 # 500: 0.33, 1000: 1.1 -- 63686 res. 32*0.66, 20s, 400: 0.25 -> 40s
    start = time.time()
    funx0_t = lambda X0: torchResiduum(X0, n_cameras, n_points, camera_indices[:numRes], point_indices[:numRes], torch_points_2d[:numRes])
    jac = jacobian(funx0_t, x0_t, create_graph=False, vectorize=True, strategy='reverse-mode') #forward-mode
    end = time.time()
    print(numRes, "its take ", end - start, "s")
    # print(jac.shape)
    # for i in range(0,numRes, 100):
    #     print(jac[i,0,9*camera_indices[i]:9*(camera_indices[i]+1)])
    #     print(jac[i,0,9*n_cameras+3*point_indices[i]:9*n_cameras+3*(point_indices[i]+1)])

if False:
    camera_params = x0_t[:n_cameras*9].reshape(n_cameras,9)
    point_params  = x0_t[n_cameras*9:].reshape(n_points,3)
    funx0_st1 = lambda X0, X1, X2: torchSingleResiduum(X0.view(1,9), X1.view(1,3), X2.view(1,2))
    start = time.time()
    JAC = torch.zeros(size=(numRes,9))
    for i in range(numRes):
        jac = jacobian(funx0_st1, (camera_params[camera_indices[i],:], point_params[point_indices[i],:], torch_points_2d[i,:]), create_graph=False, vectorize=True, strategy='reverse-mode') #forward-mode
        #jac = jacfwd(funx0_st1, has_aux=False)(camera_params[camera_indices[i],:], point_params[point_indices[i],:], torch_points_2d[i,:])
        # fill real jac
        JAC[i,:] = jac[0][0,1,:]
    end = time.time()
    print(numRes, "its take ", end - start, "s")

if False:
  #cameras, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)
  jax_cam = jax.numpy.asarray(cameras)
  jax_3d  = jax.numpy.asarray(points_3d)
  jax_2d  = jax.numpy.asarray(points_2d)
  start = time.time()
  #print(torchSingleResiduumJax(jax_cam[camera_indices[:numRes],:], jax_3d[point_indices[:numRes],:], jax_2d[np.arange(numRes),:] ))
  jaxjac = jax.jacrev(jaxResiduum, (0))(jax_cam[camera_indices[:numRes],:], jax_3d[point_indices[:numRes],:], jax_2d[np.arange(numRes),:] )
  end = time.time()
  print("Jax ", numRes, "its take ", end - start, "s")

if False:
    nres = 60000
    camera_params = x0_t[:n_cameras*9].reshape(n_cameras,9)
    point_params  = x0_t[n_cameras*9:].reshape(n_points,3)
    # camera_params.requires_grad_()
    # point_params.requires_grad_()

    funx0_st1 = lambda X0, X1, X2: torchSingleResiduumX(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d fucntion -> grad possible
    funy0_st1 = lambda X0, X1, X2: torchSingleResiduumY(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d fucntion -> grad possible
    start = time.time()
    torch_cams = camera_params[camera_indices[0:nres],:]
    torch_lands = point_params[point_indices[0:nres],:]
    torch_lands.requires_grad_()
    torch_cams.requires_grad_()
    torch_cams.retain_grad()
    torch_lands.retain_grad()
    resX = funx0_st1(torch_cams, torch_lands, torch_points_2d[0:nres,:]).flatten()
    lossX = torch.sum(resX)
    lossX.backward()
    print(resX)
    print(resX.shape)
    print(torch_cams.grad.shape)
    print(torch_lands.grad.shape)
    #print(torch_cams.grad)
    print(torch_lands.grad)
    #torch_cams.detach()
    #torch_lands.detach()
    torch_cams.grad.zero_()
    torch_lands.grad.zero_()
    resY = funy0_st1(torch_cams, torch_lands, torch_points_2d[0:nres,:]).flatten()
    lossY = torch.sum(resY)
    lossY.backward()
    print(torch_cams.grad.shape)
    print(torch_lands.grad.shape)
    #print(torch_cams.grad)
    print(torch_lands.grad)
    # ?
    #nabla = grad(funx0_st1, (camera_params[camera_indices[1:nres],:], point_params[point_indices[1:nres],:], torch_points_2d[1:nres,:]), create_graph=False) #forward-mode
    end = time.time()
    print(nres, "its take ", end - start, "s")

def getJac(start, end):
  end = min(end, camera_indices.shape[0])
  #print(start, " ", end, " ", camera_indices.shape[0])
  # why not input X0 reduced directly
  funx0_t = lambda X0: torchResiduum(X0, n_cameras, n_points, camera_indices[start:end], point_indices[start:end], torch_points_2d[start:end])
  jac = jacobian(funx0_t, x0_t, create_graph=False, vectorize=True, strategy='reverse-mode') #forward-mode
  #print(jac.shape)
  print(start, " ", end, " : ")
  #   for i in range(0,end-start, 100):
  #       print(jac[i,0,9*camera_indices[start+i]:9*(camera_indices[start+i]+1)])
  #       print(jac[i,0,9*n_cameras+3*point_indices[start+i]:9*n_cameras+3*(point_indices[start+i]+1)])

  return jac
  return start

def getJacSin(start, end, camera_params, point_params, torch_points_2d, funx0_st1):
  end = min(end, camera_indices.shape[0])
  #funx0_st1 = lambda X0, X1, X2: torchSingleResiduum(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2))
  jac = jacobian(funx0_st1, (camera_params[camera_indices[start:end],:], point_params[point_indices[start:end],:], torch_points_2d[start:end,:]), create_graph=False, vectorize=True, strategy='reverse-mode')

  #print(start, " ", end, ", jac shapes : ", jac[0].shape, " x ", jac[1].shape)
  res = funx0_st1(camera_params[camera_indices[start:end],:], point_params[point_indices[start:end],:], torch_points_2d[start:end,:])
  # print("res", res.shape) # 200,2, so N, x/y
  return (jac[0], jac[1], res)

def buildMatrix(step, full, results, varset=0) :
    data = []
    indptr = []
    indices = []
    if varset == 0:
        v_indices = camera_indices
        sz = 9
    if varset == 1:
        v_indices = point_indices
        sz = 3
    dataIndexSet = []

    for j in range(step):
        dataIndexSet.append(j * 2*step*sz + 0 * sz*step + j*sz + np.arange(0,sz))
    for j in range(step):
        dataIndexSet.append(j * 2*step*sz + 1 * sz*step + j*sz + np.arange(0,sz))
    dataIndices = np.concatenate(dataIndexSet)

    for i in np.arange(0, full, step): # i, i + step: (0, 400), etc
        start = i
        end = min(i+step, v_indices.shape[0])
        # numLocalRows = results[int(i/step)][varset].shape[0]
        # results hold res as resId, x/y, cam/point id, 9 or 3. only current cam if holds data!
        #print(i/step, " ", start, "-", end)
        #print(results[int(i/step)][varset].shape) # 200,2,200,9 | 43,2,43,9
        #print(np.arange(start*sz, end*sz, sz).shape) # 200 | 43
        #print(np.array([sz * camera_indices[start:end] + j for j in range(sz)]).transpose().flatten().shape) # 1800 | 387

        # bottleneck? how to improve though? maybe 1 x one y and combine later.
        # can have index set?
        if end!=i+step: # else uber slow
            for j in range(end-start):
                data.append(results[int(i/step)][varset][j,0,j,:].flatten().detach().numpy())
            for j in range(end-start):
                data.append(results[int(i/step)][varset][j,1,j,:].flatten().detach().numpy())
                #print(data[-1].shape) # 18, so x/y x/y ... now first all x res then all y res.
        else:
            #print(i/step, " ", dataIndices.shape)
            data.append(results[int(i/step)][varset].flatten()[dataIndices].detach().numpy())
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
        indptr.append(np.arange(2*start*sz, 2*end*sz, sz).flatten())
        #indptr.append(np.arange((2*start+1)*sz, (sz*end+1)*sz, sz))
        # x&y: operate on same cam variables
        # first all x res, then all y res here.
        indices.append(np.array([sz * v_indices[start:end] + j for j in range(sz)]).transpose().flatten())
        indices.append(np.array([sz * v_indices[start:end] + j for j in range(sz)]).transpose().flatten())

        #n = 9 * n_cameras + 3 * n_points
        #m = 2 * points_2d.shape[0]
        #crs_pose = csr_array((np.array(data).flatten(), np.array(indices).flatten(), np.array(indptr).flatten()), shape=(2*full, 9 * n_cameras)).toarray()
    indptr.append(np.array([sz+ indptr[-1][-1]])) # closing
    if False:
        print("indptr ",np.concatenate(indptr))
        print("indptr len ",len(indptr))
        print("---------")
        print("data ", np.concatenate(data).shape, " ", min(np.concatenate(data)), " ", max(np.concatenate(data)))
        print("indptr ",np.concatenate(indptr).shape, " ", min(np.concatenate(indptr)), " ", max(np.concatenate(indptr)))
        print("indices ",np.concatenate(indices).shape, " ", min(np.concatenate(indices)), " ", max(np.concatenate(indices)))

    # toarray is slow and does what? nothing? makes a dense matrix .. OMG
    datavals = np.concatenate(data)
    # debug: set all inner parameters to 0
    if False and varset == 0:
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

def buildResiduum(step, full, results, varset=2, sz =1) :
    data = []
    for i in np.arange(0, full, step): # i, i + step: (0, 400), etc
        start = i
        end = min(i+step, camera_indices.shape[0])
        # results hold res as resId, x/y, cam/point id, 9 or 3. only current cam if holds data!
        #print(i/step, " ", start, "-", end)
        #print(results[int(i/step)][varset].shape) # 200,2,200,9 | 43,2,43,9
        #print(np.arange(start*sz, end*sz, sz).shape) # 200 | 43
        #print(np.array([sz * camera_indices[start:end] + j for j in range(sz)]).transpose().flatten().shape) # 1800 | 387
        #print("res shape", results[int(i/step)][varset][:,0].flatten().numpy().shape())
        #print("res ", results[int(i/step)][varset][0].numpy())
        #for j in range(end-start):
        #    data.append(results[int(i/step)][varset][0,:].flatten().numpy())
        #for j in range(end-start):
        #    data.append(results[int(i/step)][varset][1,:].flatten().numpy())
        data.append(results[int(i/step)][varset][:,0].flatten().numpy())
        data.append(results[int(i/step)][varset][:,1].flatten().numpy())

    #print("---------")
    #print(np.concatenate(data).shape)
    res = np.concatenate(data)
    #print(res)
    return res

# fx0 + J delta x, J is Jp|Jl * xp|xl
def ComputeDerivativeMatrices():
    start = time.time() # this is not working at all. Slower then iteratively
    step = 150 # 400, 8 th appears optimal: 17s -- 200 SingleResiduum 5.5s
    full = camera_indices.shape[0]
    camera_params = x0_t[:n_cameras*9].reshape(n_cameras,9)
    point_params  = x0_t[n_cameras*9:].reshape(n_points,3)
    funx0_st1 = lambda X0, X1, X2: torchSingleResiduum(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2))
    #funx0_st1 = lambda X0, X1, X2: torchSingleResiduum(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2))
    #results = Parallel(n_jobs=8,prefer="threads")(delayed(getJac)(i, i + step) for i in np.arange(0, full, step))
    results = Parallel(n_jobs=8,prefer="threads")(delayed(getJacSin)(i, i + step, camera_params, point_params, torch_points_2d, funx0_st1) for i in np.arange(0, full, step))
    end = time.time()
    print("Parallel ", len(results) ," ", step,   " its take ", end - start, "s")
    # print(results[0][0].shape, " x ", results[0][1].shape)
    # now compose sparse! jacobian
    # define sparsity pattern and add data
    # results hold res as resId, x/y, cam/point id, 9 or 3

    start = time.time() # bottleneck now
    J_pose = buildMatrix(step, full, results, varset=0)
    end = time.time()
    print(" build Matrix & residuum took ", end-start, "s")
    start = time.time() # bottleneck now
    J_land = buildMatrix(step, full, results, varset=1) # slow
    fx0 = buildResiduum(step, full, results, varset=2, sz =1) # takes 1%
    end = time.time()
    print(" build Matrix & residuum took ", end-start, "s")

    #print(J_pose.shape)
    return (J_pose, J_land, fx0)

def ComputeDerivativeMatricesNew():
    start = time.time() # this is not working at all. Slower then iteratively
    funx0_st1 = lambda X0, X1, X2: torchSingleResiduumX(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d fucntion -> grad possible
    funy0_st1 = lambda X0, X1, X2: torchSingleResiduumY(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d fucntion -> grad possible

    torch_cams = x0_t[:n_cameras*9].reshape(n_cameras,9)[camera_indices[:],:]
    torch_lands = x0_t[n_cameras*9:].reshape(n_points,3)[point_indices[:],:]
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

    end = time.time()
    if verbose:
        print("All torch grads take ", end - start, "s")
        start = time.time()

    J_pose = buildMatrixNew(cam_grad_x, cam_grad_y, varset=0)
    if verbose:
        end = time.time()
        print(" build Matrix & residuum took ", end-start, "s")
        start = time.time()
    J_land = buildMatrixNew(land_grad_x, land_grad_y, varset=1)

    fx0 = buildResiduumNew(resX.detach(), resY.detach())

    if verbose:
        print(" build Matrix & residuum took ", end-start, "s")
        end = time.time()

    return (J_pose, J_land, fx0)

def buildMatrixNew(dx, dy, varset=0) :
    data = []
    indptr = []
    indices = []
    if varset == 0:
        v_indices = camera_indices
        sz = 9
    if varset == 1:
        v_indices = point_indices
        sz = 3

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
    if False and varset == 0:
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


# bs : blocksize, eg 9 -> 9x9 or 3 -> 3x3 per block
def blockInverse(M,bs):
    Mi = M.copy()
    if bs>1:
        bs2 = bs*bs
        for i in range(int(M.data.shape[0]/bs2)):
            mat = Mi.data[bs2*i:bs2*i+bs2].reshape(bs,bs)
            # print(i, " ", mat)
            imat = inv_dense(mat)
            Mi.data[bs2*i:bs2*i+bs2] = imat.flatten()
    else:
        Mi = M.copy()
        for i in range(int(M.data.shape[0])):
            Mi.data[i:i+1] = 1./ Mi.data[i:i+1]
    return Mi
#S = Ul - W * Vli * W.transpose()
# alternative power its on Ul - W * Vli * W.transpose()
# better on Ul [I - Uli * W * Vli * W.transpose()] or rather
# sum_k [Uli * W * Vli * W.transpose()]^k (Ul^-1 * b)
# oh no. this is not! GD. it is not a gradient in any sense
# this is Ax = b. not GD A is NOt at all symmetric.
# Ul - W * Vli * W.transpose() is symmetric
# S = Ul - W * Vli * W^t is symmetric. 1/2 xSx + bx -> gd ok
# solve as Sx + b=0, now eigenvalues of S?
#
# S = Ul ( I - Uli * W * Vli * W.transpose())
# g = (Ul - W * Vli * W^t) * xk - bb
# xk = xl - gamma g
#
# But trick where we know L:
# Uli * W * Vli * W.transpose() is not symmetric. 
#
# S = Ul ( I - Uli * W * Vli * W.transpose()) has eigen values eig(U) * (0,1]
# what is L of U ?
#
# interpreted as GD:
#     xk = gamma * bb
# for it in range(m):
#    g = (WW.dot(xk) - bb)
#    xk = xk - gamma * g # xk = xk-1 + l * nabla f(xk-1).
# power iterations: gamma = 1

def solvePowerIts(Ul, W, Vli, bS, m):
    #costk = np.sum( bS**2 )
    #print("start gd cost ", costk)

    Uli = blockInverse(Ul, 9)
    xk  = Uli * bS
    g   = xk

    for it in range(m):
       # here uli^1/2 * M uli^1/2 * 'uli^1/2 * g' could be a symmetric split.
       # to the power of k uli^1/2 * uli^1/2 = uli
       g = Uli*(W*(Vli*(W.transpose() * g)))
       xk = xk + g

       if False: 
       # eq is Ul [I - Uli * W * Vli * W.transpose()] x = b
         costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
         print(it, " gd cost ", costk)

       if stop_criterion(np.linalg.norm(xk, 2), np.linalg.norm(g, 2), it):
          return xk
    return xk

# symmetric version, correct but how to find L?
# S x = -b -> x^t(Sx+b) -> min, S = Ul - W * Vli * W.transpose()
# matrix is 1/2 x^T [ Ul - W * Vli * W.transpose() ] x + b^T x ->min
def solveByGD(Ul, W, Vli, bS, m, gamma):
    verbose = False
    gamma = 2 / 1.6 # 2/L+mu 
    # assume without adding diag L=1 mu=0 here. 
    # adding diag is about 1, then ? the more we add on diag the bigger mu

    #Unorm1 = diag_sparse(np.squeeze(np.asarray( 1./np.sqrt(np.abs(Ul).sum(axis=0)))))
    #Unorm1 = diag_sparse(np.squeeze(np.asarray( 1./(np.abs(Ul).sum(axis=0)))))
    S = Ul - W * Vli * W.transpose()
    Unorm1 = diag_sparse(np.squeeze(np.asarray( 1./np.sqrt( (np.abs(S)).sum(axis=0) ))))
    # Best pcg is Ul^-1 applied on both sides as Uli^1/2.

    #print("Unorm1 ", 1./np.abs(Ul).sum(axis=0).ravel())
    # Ax = b <-> K A K x = K b, then return K x since K^-1 x is computed as solution of [K A K] x = K b
    # (K A K)^-1 K b = z , K^-1 A^-1 b = z <-> A^-1 b = K z
    # preconditioning
    Ul_ = Unorm1 * Ul * Unorm1
    W_ = Unorm1 * W
    bS_ = -Unorm1 * bS
    #Uli = blockInverse(Ul, 9)
    # x0 = 0 -> g = M*0 + bS, x1 = 0 - gamma bs
    xk = -gamma * bS_ # test set to 0 * bS_.

    if verbose:
        costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
        costkB = np.sum( (Unorm1*xk).transpose() * (0.5*(S * (Unorm1*xk)) + bS) )
        print("-1 gd cost ", costkB, " ", costk)
    
    for it in range(m):
        g = Ul_ * xk - W_*(Vli*(W_.transpose() * xk)) + bS_
        xk = xk - gamma * g

        if verbose:
            # test: 
            # eq is Ul [I - Uli * W * Vli * W.transpose()] x = b
            costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
            costkB = np.sum( (Unorm1*xk).transpose() * (0.5*(S * (Unorm1*xk)) + bS) )
            print(it, " gd cost ", costkB, " ", costk)
    return Unorm1 * xk

# S = Ul ( I - Uli * W * Vli * W.transpose()) 
# uli is fantastic pcg -> (I - Uli * W * Vli * W.transpose())  is not symmetric so not GD 
# uli ^1/2 needs eigen 9x9 would work. likely too slow.
# U * M * M * U, squared kappa. already fail.
# Ui M 
# 1/2 x^t S x + x^t b is cost. |Sx-b|^2 = xt S*S 
# But
# 1/2 x^t U S x + x^t b -> min
# 1/2 x^t x - 1/2 U S x + 1/2 x^t U S + b = x - 1/2 (U S x + S^t U ^T x) + b = x - U S x + b
# 
def solveByGDPower(Ul, W, Vli, bS, m, gamma):
    # by global it decrease gamma, start with 1.5 go to 1.0
    # maybe learn by L and it number?
    gamma = 1.2 # 2/(l+mu) clearly smaller for larger L. gamma==1 -> power its

    #L0=0.1, 
    #gamma = 1, 9  gd cost  1357866037366478.8
    #gamma = 2, 9  gd cost  1986622831238522.5
    #gamma = 1.25, gd cost  747212056312938.4
    #gamma = 1.42  gd cost  486363655246602.06
    #gamma = 1.5   gd cost  409126119073686.75
    #gamma = 1.6   gd cost  320975775794483.6
    #gamma = 1.7   gd cost  253410838385286.88
    #gamma = 1.8   gd cost  216552942444294.06
    #gamma = 1.9   gd cost  349992174153889.6 # wha?
    #gamma = 2.0   gd cost  1986622831238522.5
    #L0=1,
    # gamma = 1.6  gd cost  101086086136.3841
    # gamma = 1.5  gd cost  1625528970.087006
    # gamma = 1.4  gd cost  11306535.37204616 !
    # gamma = 1.3  gd cost  14668444.073417133
    # gamma = 1.2  gd cost  302646198.864862
    # gamma = 1.0  gd cost  49948313572.82567

    # can i have .. + L |delta|^2 for better control. replacing UTU + L D(UtU)
    # problem JU|JV * delta. Maybe JutJu + L*diag(JutJu)
    # Add L*I to original problem, how does it appear
    # Ul = (U+LI)
    # S = Ul ( I - Uli * W * Vli * W.transpose()) = [U - W * Vli * W.transpose())] + L I
    # for all L Ul^-1[..] in [0,1). == I - Uli^-1 W VLI Wt, for all l
    # Uli^-1 W VLI Wt in [0,1] for all l, so between max 1/mu(Uli) * L(W VLI Wt) and min 1/L(Uli) * mu(W VLI Wt)   
    # 1/mu(Uli) l-> 0 or inf is 1/mu(U) or 0? So L(W VLI Wt) < mu(Uli) then also < mu(U)
    # Bullshit it follws max 1/mu(Uli) * L(W VLI Wt) -> 1 for l -> inf
    # likewise it should be 1/L(Uli) * mu(W VLI Wt) -> 1
    # much simpler 1/(U+l) * 1/(V+l)

    Uli = blockInverse(Ul, 9)
    ubs = - Uli * bS
    xk = - gamma * ubs

    verbose = False
    if verbose:
        costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
        print("-1 gd cost ", costk)
    
    for it in range(m):
        #( I - Uli * W * Vli * W.transpose()) 
        g = xk - Uli*(W*(Vli*(W.transpose() * xk))) + ubs
        xk = xk - gamma * g

        if verbose:
            # test: 
            # eq is Ul [I - Uli * W * Vli * W.transpose()] x = b
            costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
            print(it, " gd cost ", costk)

        if stop_criterion(np.linalg.norm(xk, 2), np.linalg.norm(gamma * g, 2), it):
           return xk
    return xk

# test Loop over L0=x, L=y here. Likely best to do grid search to get an idea. model as exp(-poly(L,it))
def solveByGDNesterov(Ul, W, Vli, bS, m, L):
    L = 0.9 # 100 -> 1. 
    #lambda0 = 1 #?
    lambda0 = (1.+np.sqrt(5.)) / 2. # l=0 g=1, 0, .. L0=1 g = 0,..

    #0    -> 0.66
    #0.01 -> 0.66
    #0.1  -> 0.

    # L0=0.01, L=0.66 is best, L0=1 L=0.8. L0=10 L=1. 1 - 0.2 / L0
    # L < 1, L0=0.1
    # 9  gd cost  35243429520496.086 L = 1
    # 8  gd cost  26813733250342.566 L = 0.8
    # 9  gd cost  27865486374890.02 L = 0.9

    #L0=1
    #9  gd cost  254586553117.66028 L=1
    #9  gd cost  10877888816.105503 L=0.9
    #9  gd cost  5208501071.905113  L=0.8 !

    #L0=10
    #9  gd cost  3732.7242597387008
    #9  gd cost  0.0388449355871202 L=1
    #9  gd cost  4042720329.1696663 L=0.8

    Uli = blockInverse(Ul, 9)
    ubs = - Uli * bS
    xk = - ubs
    y0 = - ubs

    verbose = False
    if verbose:
        costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
        print("-1 gd cost ", costk)
    
    for it in range(m):
       lambda1 = (1 + np.sqrt(1 + 4 * lambda0**2)) / 2
       gamma = (1-lambda0) / lambda1
       lambda0 = lambda1

       #( I - Uli * W * Vli * W.transpose()) 
       g = xk - Uli*(W*(Vli*(W.transpose() * xk))) + ubs
       yk = xk - 1/L * g
       xk = (1-gamma) * yk + gamma * y0
       y0 = yk

       if verbose:
         # test: 
         # eq is Ul [I - Uli * W * Vli * W.transpose()] x = b
         costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
         print(it, " gd cost ", costk)

       if stop_criterion(np.linalg.norm(xk, 2), np.linalg.norm(1/L * g, 2), it):
           return xk
    return xk

def solveByGDPolak(Ul, W, Vli, bS, m, L, mu):
    #L = 1 # 100 -> 1. 
    #mu = 0.9
    L = 0.7 # 100 -> 1. 
    mu = 0.4
    #L0=0.1
    #0.7/0.4 gd cost  181600100398952.8
    #0.7/0.5 gd cost  258197042164083.25
    #0.7/0.3 gd cost  1747953609506390.5 TOTAL BS!
    #0.8/0.4 gd cost  213004761838213.38

    #L0=1
    # L =0.8, 1.4=2/(L+mu), 1/0.7 - L = mu = 0.6
    L = 0.9
    mu = 0.6
    #L = 0.8, mu = 0.6 cost    29209723.640828945
    #L = 0.8, mu = 0.5 cost  3867287715.704476
    #L = 0.9, mu = 0.6 cost      728035.5904369259
    #L = 0.8, mu = 0.7 cost     4319868.471749563
    #L = 0.9, mu = 0.7 cost    41151320.04702702

    deltaInf = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
    delta1 = 4 * deltaInf / (L-mu)
    delta2 = (1 - 2 * deltaInf * (L+mu)/(L-mu))
    gamma = 2 / (L+mu)

    Uli = blockInverse(Ul, 9)
    ubs = - Uli * bS
    xk0 = - ubs * 0
    xk = - ubs * gamma

    verbose = False
    if verbose:
        costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
        print("-1 gd cost ", costk)
    
    for it in range(m):

       #( I - Uli * W * Vli * W.transpose()) 
       g = xk - Uli*(W*(Vli*(W.transpose() * xk))) + ubs
       #zk = xk - delta1 * g + delta2 * (xk0 - xk) # xk-1 - delta1 nabla f + delta2 (xk-2 -xk-1)
       xk_i = delta1 * g + delta2 * (xk0 - xk)
       zk = xk - xk_i # xk-1 - delta1 nabla f + delta2 (xk-2 -xk-1)
       xk0 = xk
       xk = zk

       if verbose:
         # test: 
         # eq is Ul [I - Uli * W * Vli * W.transpose()] x = b
         costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
         print(it, " gd cost ", costk)

       if stop_criterion(np.linalg.norm(xk, 2), np.linalg.norm(xk_i, 2), it):
           return xk
    return xk

# stop criterion is 
# stop_criterion(np.linalg.norm(delta, 2), np.linalg.norm(delta_i, 2), it)
def stop_criterion(delta, delta_i, i):
    eps = 1e-6 #1e-2 used in paper, tune. might allow smaller as faster? TODO: confusing first step, not av step
    return (i+1) * delta_i / delta < eps

# |f(x0) + J(xo) * delta|^2 + lambda | [Ip,Il] * delta|^2
# iterate over this: 

# Lm step took  0.11277008056640625 s
# 3 it. cost 0      3535975824.167471
# 3 it. cost 0/new  3290971702.6593885
# 3 it. cost 1      3315161329.207933  cost compute took  0.00623321533203125 s
#  Test  258.96993952164456   148.9847721007778   538.8031130609922

# fill lists G and F, with g and f = g - old g, sets of size m, 
# at position it % m, c^t compute F^tF c + lamda (c - 1/k)^2, sum c=1
# g = x0, f = delta. Actullay xnew = xold + delta.
def RNA(G, F, g, f, it, m, Fe, fe, lamda, old_c):
    lamda = 0.005 # reasonable 0.01-0.1
    crefVersion = True
    lamda = 0.05 # cref version needs larger 
    h = -1. #-0.25 # 2 / (L+mu) -- should 1/diag * F^t F * c. Maybe if L too large -> go longer == h=-1: 2 delta not 1 delta
    # todo: all depend on value of L, lamda nut more importantly f/delta! 
    # xk+1 = prox_f(xk)
    # recall u = prox_f(s) = argmin f(u) + L/2 |u-s|^2 => nabla f(u) = L(s-u). Here s==0
    # then our solution nabla f(delta u) = - L delta(u) .. Haeh?
    # i have prox is fixpoint equation g, f = gk - gk-1 = delta. we want delta small
    # I should multiply the deltas by L?
    # what do i have if we do not use a delta update but uk+1 = prox(uk)
    # |f(xk) + nabla f * delta xk|^2 + L/2|delta|^2, here nabla f is the Jacobian.
    # |f(xk) + nabla f * (x - xk)|^2 + L/2|x - xk|^2
    # |f(xk) - nabla f * xk + nabla f * x|^2 + L/2|x - xk|^2
    # then for u = prox f(x), u-xk = nabla f(u), again u = xk+1 = xk + delta
    # nabla f(xk+1) = L(xk-xk+1) = L delta. So we should try to get L delta small, find c : |Gc| ->min
    # same solution / algorithm. Can give as idea? 
    # correct Needs large L however.
    # check if R/t/inner work differently. How? set HUGE L for those!

    if len(G) >= m:
        #print("it, it%m", it, " ", it % m)
        G[it % m] = np.squeeze(g)
        F[it % m] = np.squeeze(f)
        Fe[it % m] = np.squeeze(fe)
    else:
        G.append(np.squeeze(g))
        F.append(np.squeeze(f))
        Fe.append(np.squeeze(fe))
    mg = len(G)
    cref = np.zeros(mg)
    if mg >= m:
        cref[it % m] = 1
    else:
        cref[mg-1] = 1

    Gs = np.concatenate(G).reshape(mg, -1).transpose()
    Fs = np.concatenate(F).reshape(mg, -1).transpose()
    Fes = np.concatenate(Fe).reshape(mg, -1).transpose()
    #print("Fs ", Fs.shape)
    FtF = Fs.transpose().dot(Fs) # why dot?
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
    extrapolation = Gs.dot(c) #+ 0.1 * Fs.dot(c)
    extrapolationF = Fes.dot(c)

    # test
    verbose = False
    if verbose:
      if it > 0:
          if it < m:
              #print("old_c pre append", old_c)
              old_c = np.append(old_c, np.array([0]), axis=0)
              #print("old_c append", old_c)
          else:
             old_c[it % m] = 0
          #print("old_c ", old_c)
          #print("old_c ", old_c.shape)
          old_extr_cost = np.linalg.norm(Fs.dot(old_c), 2)
          old_c = c # current
      else:
         old_extr_cost = -1
         old_c = c
      print(" Test Ex/Last/old", np.linalg.norm(extrapolationF,2), " ",  np.linalg.norm(f,2), " ", old_extr_cost)
    print("c ", c, " ", c.shape)

    # shape utter non sense
    #print("extrapolation ", extrapolation.shape, " ", g.shape)
    return (G, F, Fe, np.squeeze(extrapolation - h * extrapolationF), old_c)

def is_topk(a, k=1):
    rix = np.argsort(-a)
    return rix[:k]
    return np.where(rix < k, 1, 0).reshape(a.shape)

# Hk+1 := Hk +  [ <pk,qk> + <Hkqk,qk> ] / <pk,qk>^2 * [pk*pk^T] - 
# exactly algorithm 7.4 of nocedal (compute Hk*q):
# set q = r^k; s=p; y=q. rho_k = 1/<sk,yk>
# H0 is a diagonal approximation of the hessian, eg. mu*Identity
# call with BFGS_direction(u-v, ps:=direction-buffer, qs:=[(uc-vc) - (u-v)]-buffer, rhos, iter, mem-size, mu)
# r: u-v. not clear / residuum
# ps: direction/output buffer, qs: update/gradients buffer, rhos buffer scalar products
# k iteration number, mem: memory, mu: initial approx mu=1 usually
# maybe q current residual (of computed update) , qs:gradient/residual difference, ps: target/update of target, eg s^+ - s -> H(s,y) * q
# differs from drs version significantly.
def BFGS_direction(r, pk, qk, ps, qs, rhos, k, mem, mu):
    # r = -r # not needed with below
    # lookup k-1, k-mem entries. 
    alpha = np.zeros([mem,1])
    r = np.squeeze(r)
    for i in range(k-1, np.maximum(k-mem,-1), -1):
        #print("1i", i) # k-1, .. k-mem usually
        j = i % mem # j>=0
        #print("1j", j)
        #print("j ", j, " ", r.shape, ps[j].shape)
        alpha[j] = np.dot(r, ps[j]) * rhos[j]
        r = r - alpha[j]*qs[j]
        print(j, " 1st. al ", alpha[j], " rh ", rhos[j], " qs " , np.linalg.norm(qs[j],2), " ps " , np.linalg.norm(ps[j],2) )

    dk = mu * r

    for i in range(np.maximum(k-mem, 0), k):
        #print("2i", i) # k-1, .. k-mem usually
        j = i % mem # j>=0
        #print("2j", j)
        beta = rhos[j] * np.dot(dk, qs[j])
        dk = dk + ps[j] * (alpha[j] - beta)
        print(j, " 2nd. al ", alpha[j], " rh ", rhos[j], " be ", beta, " qs " , np.linalg.norm(qs[j],2), " ps " , np.linalg.norm(ps[j],2) )
        
    # update ps and qs
    #
    pk = -dk # yes minus, but pk is '+'

    #pkStepLength = np.linalg.norm(pk,2)
    #dkStepLength = np.linalg.norm(dk,2)
    #pk = pkStepLength/dkStepLength * dk # or this version ? might be better but likely needs more changes
    #print(k, " it |pk - dk|^2 " , np.linalg.norm(pk - dk, 2))
    #print(k, " it |r - dk|^2 " , np.linalg.norm(r - dk, 2))
    rhok = np.sum(pk.dot(qk))
    rhok = np.maximum(0, 1./rhok)
    #print("rhok ", rhok)
    ps[k%mem,:] = np.squeeze(pk) # or store dk or -dk? dk: prediction from here, not form prox
    qs[k%mem,:] = np.squeeze(qk) # store old r minus this r, so diiference of residuum
    rhos[k%mem,:] = rhok

    return (dk, ps, qs, rhos)

write_output = False #True
read_output = False
if read_output:
    camera_params_np = np.fromfile("camera_params.dat", dtype=np.float64)
    point_params_np = np.fromfile("point_params.dat", dtype=np.float64)
    x0_p = camera_params_np.reshape(-1)
    x0_l = point_params_np.reshape(-1)
    #x0 = np.concatenate([x0_p, x0_l])
    x0 = np.hstack((x0_p, x0_l))
    x0_t = from_numpy(x0)
    camera_params = x0_t[:n_cameras*9].reshape(n_cameras,9)
    point_params  = x0_t[n_cameras*9:].reshape(n_points,3)


# parameter bfgs
bfgs_mem = 6 # 2:Cost @50:  -12.87175888983266, 6: cost @ 50: 12.871757400143322
bfgs_mu = 1.0
bfgs_qs = np.zeros([bfgs_mem, n]) # access/write with % mem
bfgs_ps = np.zeros([bfgs_mem, n])
bfgs_rhos = np.zeros([bfgs_mem, 1])

# compute f(x0) and Jacobians given x0
# do not understand can elad to hicup despite being better per step.
useExtInCost = True # with using extrapolation, does it lead to lower cost, how much? -- must use with TR as well?
L0 = 1.0 #e-3
L = L0
iterations = 100
verbose = False
debug = False
useInvSolver = False
# VENICE
#SOLVER 14 it. cost 1      780196.0678238661
#        9 it. cost 1      857776.3171311354
# power its 10
#  9                 1116206.857252748       with penalty  1129433.474213173
# 14 it. cost 1      1080683.541304447       with penalty  1083019
# power 20 
#9 it. cost 1      981319.3290127576       with penalty  986289.6641950987
#14 it. cost 1      965701.3471037014       with penalty  967042.
# powerits = 30
# 9 it.  cost 1      863494.3923218106       with penalty  889793.9636650706
# 14 it. cost 1      649813.2902730353       with penalty  657871.82424686
# issue is L grows too large sometimes.
#Polak 1o its
#9 it. cost 1      1082638.0196162288       with penalty  1098187.86145675
#14 it. cost 1      701979.4950357398

powerits = 200
gamma = 1/1
Gs = [] 
Fs = []
Fes = []
rnaBufferSize = 6
sit = 0
updateJacobian = True
#gamma = 1/10000000000000 # recall 2/(L+mu), adding L*diag as U *(I-X) -> L+=k*L and mu + k*L, 
# so 2 / ((k+1)L + k L+mu) vs 2/(L+mu) becomes 2/(k*L + mu)
# 1. divide Ul and other part so divide S by diag(Ul).
# 2. then gamma~2/1+almost 1? hmm.

camera_params = x0_t[:n_cameras*9].reshape(n_cameras,9)
if debug:
    print("init x0_p rot", camera_params[:,0:3]) # to check validity -- what goes wrong?
    print("init x0_p tra", camera_params[:,3:6]) # to check validity -- what goes wrong?
    print("init x0_p int", camera_params[:,6:9]) # to check validity -- what goes wrong?

it = 0
while it < iterations:
    print("Using L = ", L)
    if updateJacobian: # not needed if rejected
        start = time.time()
        J_pose, J_land, fx0 = ComputeDerivativeMatricesNew()
        #J_pose, J_land, fx0 = ComputeDerivativeMatrices() # takes x0_t

        JtJ = J_pose.transpose() * J_pose # can use the block diagonal for precond. i don't know .. 
        JtJDiag = diag_sparse(JtJ.diagonal())
        #JtJDiag = diag_sparse(np.fmax(JtJ.diagonal(), 1e0))
        JltJl = J_land.transpose() * J_land
        #JltJlDiag = diag_sparse(JltJl.diagonal())
        JltJlDiag = diag_sparse(np.fmax(JltJl.diagonal(), 1e-3)) # todo: does this help == lower total energy? no but |delta_l| is much smaller

        # awful
        #JtJDiag = diag_sparse(np.ones(9 * n_cameras))
        #JltJlDiag = diag_sparse(np.ones(3 * n_points))

        # for both? 
        J_eps = 1e-9
        JltJlDiag = JltJl + J_eps * diag_sparse(np.ones(JltJl.shape[0]))
        JtJDiag   = JtJ + J_eps * diag_sparse(np.ones(JtJ.shape[0]))

        if verbose:
            print("J_pose ", J_pose.shape)
            print("J_land ", J_land.shape)
            print("fX0 ", fx0.shape)
            print("JtJ ", JtJ.shape)
            print("J_land ",J_land.shape)
            print("JltJl ",JltJl.shape)

        debugNonlinearity = False
        if debugNonlinearity:
            jtjd = JtJ.diagonal()
            jtjd[6:-1:9] = jtjd[6:-1:9] * 100
            jtjd[7:-1:9] = jtjd[7:-1:9] * 100
            jtjd[8:-1:9] = jtjd[8:-1:9] * 100
            JtJDiag = diag_sparse(jtjd)
        
        end = time.time()
        print("Computing Derivative and JtJ took ", end - start, "s")


    # one iteration, schur solver.
    # W= jp^t Jl
    # Ul = JtJ + l * JtJDiag
    # Vl = JltJl + l * JltJlDiag
    # bp = Jp^T * fx0, bl = Jl^t * fxo
    # S = Ul - W^t Vl^-1 W, bS = bp - W Vl^-1 bl
    # later above can be solved with GD. given L and mu. 
    # or 
    # delta xp = - S^-1 bS
    # delta xl = - V^-1 (W^T delta xp - bl) 

    costStart = np.sum(fx0**2)
    #print("cost ", it, " ", round(costStart))
    #print("Residuum initial ", fx0, " ", fx0.shape)
    start = time.time()
    W = J_pose.transpose() * J_land
    Vl = JltJl + L * JltJlDiag
    Ul = JtJ + L * JtJDiag
    bp = J_pose.transpose() * fx0
    bl = J_land.transpose() * fx0

    if verbose:
       print("W", W.shape)
       print("Vl", Vl.shape, " ", Vl.data.shape, "non zeros in 3x3 blocks")
       print("Ul", Ul.shape)
       print("bp", bp.shape)
       print("bl", bl.shape)

    Vli = blockInverse(Vl, 3)
    bS = (bp - W * Vli * bl).flatten()

    # S x = bS. 1/2*(JtJDiag)^-1 * S x = 1/2*(JtJDiag)^-1 b
    # recall the pcg trick row sum col sum from left / right
    # then L <= 1? diag_sparse(1./np.abs(A).sum(axis=0)) and diag_sparse(1./np.abs(A).sum(axis=1))
    # but since Ul ( I - bla ) is sym -> both are same !?
    # even only U suffices? 

    if useInvSolver:
        S = Ul - W * Vli * W.transpose() # = Ul - Jp^t J_l * (Jl^t Jl)^-1 J_l^T Jp^t
        #Snorm1   = diag_sparse(1./np.abs(S).sum(axis=0))
        #Snorm1   = 1./np.abs(S).sum(axis=0).ravel() # maybe not neeed?
        #same SnormInf = 1./np.abs(S).sum(axis=1)
        #print("Snorm1 ", Snorm1)
        #print("Unorm1 ", 1./np.abs(Ul).sum(axis=0).ravel())
        delta_p = - inv_sparse(S) * bS
        # Ax = b <-> K A K x = K b, then return K x since K^-1 x is computed as solution of [K A K] x = K b
    else:
        #delta_p = - solveByGDPolak(Ul, W, Vli, bS, powerits, L, 0.9)
        delta_p = - solveByGDNesterov(Ul, W, Vli, bS, powerits, L)
        #delta_p = - solveByGDPower(Ul,W, Vli, bS, powerits, gamma/L)
        #delta_p = - solvePowerIts(Ul,W, Vli, bS, powerits)
        #delta_p = - solveByGD(Ul, W, Vli, bS, powerits, gamma/L) # likely does not work well

    if verbose:
        print("bS:", bS.shape, "=", W.shape, " x ", Vl.shape,  " x ", bl.shape)
        # (441,) = (441, 147)  x  (147, 147)  x  (147,)
        print("S:", S.shape, " = ", Ul.shape, "x", Vl.shape, "x", W.shape, " bS: ", bS.shape, " " )
        # (441, 441)  =  (441, 441) x (147, 147) x (441, 147)  bS:  (441,)
        print("delta_p: ", delta_p.shape)
        print(Vl.shape, "x", W.shape, "x", delta_p.shape, "- ", bl.shape)

    # paper appears wrong this is correct: W^t delta_p + Vl delta_l = -bl -> - bl - W^t delta_p = Vl delta_l
    # -> delta_l= - Vli (bl + W^t delta_p)
    delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl).flatten()  # experimental 1st best but last delta_l=1e9
       
    end = time.time()
    print("Lm step took ", end - start, "s")

    fx0_new = fx0 + (J_pose * delta_p + J_land * delta_l)
    costQuad = np.sum(fx0_new**2)
    costQuadPenalty = costQuad + L * (delta_p.dot(JtJDiag * delta_p) + delta_l.dot(JltJlDiag *delta_l))
    print(it, "it. cost 0     ", round(costStart) )
    print(it, "it. cost 0/new ", round(costQuad), " cost with penalty ", round(costQuadPenalty) )

    # update and compute cost
    x0_p = x0_p + delta_p
    x0_l = x0_l + delta_l

    x0 = np.hstack((x0_p, x0_l), dtype=np.float64)
    x0_t = from_numpy(x0)
    camera_params = x0_t[:n_cameras*9].reshape(n_cameras,9)
    point_params  = x0_t[n_cameras*9:].reshape(n_points,3)

    if debug:
        print("x0_p rot", camera_params[:,0:3]) # to check validity -- what goes wrong?
        print("x0_p tra", camera_params[:,3:6]) # to check validity -- what goes wrong?
        print("x0_p int", camera_params[:,6:9]) # to check validity -- what goes wrong?

    start = time.time()
    funx0_st1 = lambda X0, X1, X2: torchSingleResiduum(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2))
    fx1 = funx0_st1(camera_params[camera_indices[:]], point_params[point_indices[:]], torch_points_2d[:,:])
    end = time.time()
    costEnd = np.sum(fx1.numpy()**2)
    costEndPenalty = costEnd + L * (delta_p.dot(JtJDiag * delta_p) + delta_l.dot(JltJlDiag *delta_l))
    print(it, "it. cost 1     ", round(costEnd), "      with penalty ", round(costEndPenalty), " cost compute took ", end - start, "s")

    if costStart < costEndPenalty:
        # revert -- or linesearch
        # in theory should be a constant (1+l)/(1+2*l), unclear why it is not
        # dx JtJ dx + L * dx JtJ dx + 2 J^tb <=> (1+L) * dx JtJ dx + 2J^tb -> min
        # <=> (1+L) JtJ dx = J^tb <=> (1+L) dx = (JtJ)^-1 J^tb
        # sol 1: (1+L)  dx1 = (JtJ)^-1 J^tb
        # sol 2: (1+2L) dx2 = (JtJ)^-1 J^tb
        # dx2 = dx1 (1+L)/(1+2L). Incorrect W is missing.

        x0_p = x0_p - delta_p
        x0_l = x0_l - delta_l
        x0 = np.hstack((x0_p, x0_l), dtype=np.float64)
        x0_t = from_numpy(x0)
        camera_params = x0_t[:n_cameras*9].reshape(n_cameras,9)
        point_params  = x0_t[n_cameras*9:].reshape(n_points,3)
        updateJacobian = False
    else:

        updateJacobian = True
        it = it + 1
        # just test
        RNAVersion = True
        if not RNAVersion: # does work
            # x_k+1 = g(xk), delta = g(xk) - x_k, residual
            #deltaL = np.hstack((JtJDiag * delta_p, JltJlDiag * delta_l)) # actual gradient
            delta = np.hstack((delta_p, delta_l))
            deltaL = np.hstack((bp,bl))
            # normal
            deltaL = delta.copy()
            deltaS = delta.copy()

            #deltaL = L /L0 * delta.copy()
            #deltaS = delta * L0 / L

            #deltaL = delta # todo: not corrcet?
            # it. cost ext    2258203006.7622504, still better setting deltaL = delta for no reason
            # now what is what
            # recall it is modeling H^1 * nabla. x^tHx + nabla x + b -> Hx 
            # drs: r=s+-s. qs = r+-r, ps: prediction, namely r^
            # r = u-v
            # s+ = s - r
            # q+ = r+ - r
            # s+ = s + dk -> dk = s+ -s = -r
            # and r -> 0 since s fixpoint 
            # conclusion: r: delta, pk = delta or dk? qk = r-r_old
            #BFGS_direction(r, pk, qk ..
            # TODO if success use dk for delta
            if sit > 0:
                qk = deltaL - deltaL_old
                bfgs_r = deltaS
                print(" |Delta| ", np.linalg.norm(delta,1), " |Delta_p| ", np.linalg.norm(delta_p,1), " |Delta_l| ", np.linalg.norm(delta_l,1))
                print(" |deltaL - deltaL_old| ", np.linalg.norm(qk,1))
                print("<deltaL , deltaL_old> ", qk.dot(delta))
                pk = deltaS
                #bfgs_mu = diag_sparse(1./(np.concatenate([JtJ.diagonal(), np.fmax(JltJl.diagonal(), 1e-3)])))
                #bfgs_mu = diag_sparse((np.concatenate([JtJ.diagonal(), np.fmax(JltJl.diagonal(), 1e-3)]))) # rho is 0 here.

                # todo: maybe this gets disturbed by few super large landmark deltas / unconstrained landmarks.
                # mega todo: of course we cannot fill the buffer before we accept in this case.
                dk, bfgs_ps, bfgs_qs, bfgs_rhos = BFGS_direction(bfgs_r, pk, qk, bfgs_ps, bfgs_qs, bfgs_rhos, sit-1, bfgs_mem, bfgs_mu)
                #print(sit-1, " it |pk - dk|^2 " , np.linalg.norm(pk - dk, 2))
                # DeltaDiag = diag_sparse(np.concatenate([np.squeeze(JtJ.diagonal()), np.squeeze(JltJl.diagonal())] ))
                # DeltaDiag = blockInverse(DeltaDiag, 1)
                # todo: large L -> fails. why? always stuck in smaller energies
                # maybe just line search *eta would work here. last eta:  if > eta/2 else try eta*2 until > 
                #extr = x0 + 1 * (dk - delta) # original version. 0.0125
                #extr = x0 + L * 0.5 * (2 * dk - delta) # with deltaL = delta
                #extr = x0 + 1/(2*L) * deltaL
                #extr = x0 + L0*L0/(2*L) * delta # the bitter truth working well. only once. like L0=1 version.
                #extr = x0 + 0.051 * (dk - delta)

                # line search or check step length in line with L / delta
                if False:
                    deltaStepLength = np.linalg.norm(delta,2)
                    dkStepLength = np.linalg.norm(dk,2)
                    # different #extr = x0 + 1. * (deltaStepLength / dkStepLength * dk - delta)
                    extr = x0 + 0.3 * deltaStepLength / dkStepLength * (dk - delta)
                    print("1. deltaStepLength", deltaStepLength, " dkStepLength ", dkStepLength, " ratio ", deltaStepLength / dkStepLength)
                if False:
                    DeltaDiag = diag_sparse(np.concatenate([np.squeeze(JtJ.diagonal()), np.squeeze(JltJl.diagonal())] ))
                    #DeltaDiag = blockInverse(DeltaDiag, 1)
                    #DeltaDiag = diag_sparse(np.ones(n))
                    deltaStepLength = delta.dot(DeltaDiag * delta)
                    dkStepLength = dk.dot(DeltaDiag * dk)
                if True:
                    deltaStepLength = np.sqrt(delta_p.dot(JtJDiag * delta_p) + delta_l.dot(JltJlDiag * delta_l))
                    dkStepLength = np.sqrt(dk[:n_cameras*9].dot(JtJDiag * dk[:n_cameras*9]) + dk[n_cameras*9:].dot(JltJlDiag * dk[n_cameras*9:]))
                    # step length is 1e-12 or so. ALWAYS WRONG.

                ##extr = x0 + 1. * (deltaStepLength / dkStepLength * dk - delta)
                print("2. deltaStepLength", deltaStepLength, " dkStepLength ", dkStepLength, " ratio ", deltaStepLength / dkStepLength)

                # works but slow w.o. 239 -> *1 218 normal: 214, dk not pk: 215, 213 l0/l, this does not work with lo/l
                #extr = x0 + 1 * deltaStepLength / dkStepLength * (dk - delta)
                extr = x0 + 0.5 * (deltaStepLength / dkStepLength * dk - delta)
            else:
                #extr = x0 + delta
                extr = x0
            deltaL_old = deltaL
        else: # works for small steps / continuously a bit
            lamdaRNA = 1 # likely just using '- h * lambda' to get a benefit.
            # deltaL does not appear to matter much
            deltaL = L * np.hstack((JtJDiag * delta_p, JltJlDiag * delta_l)) # ? no L * is same as lambdaRna * L?
            deltaL = L * np.hstack((delta_p, delta_l)) # much difference over not 'L * '. maybe linesearch
            #deltaL = L * np.hstack((bp,bl)) # also not working gradient direction
            delta = np.hstack((delta_p, delta_l))
            # maybe min in delta norm? |sum_i nabla fi ci|^2-> min
            Gs, Fs, Fes, extr, old_c = RNA(Gs, Fs, x0, deltaL, sit, rnaBufferSize, Fes, delta, lamda = lamdaRNA, old_c = old_c)
            #_, _, _, extr, _ = RNA(Gs, Fs, x0, deltaL, sit+1, rnaBufferSize, Fes, delta, lamda = lamdaRNA, old_c = old_c) # better with hmm

            # test just 'acceleration' also working. Not so well though / significantly worse.
            #extr = x0 - (1./np.sqrt(2)-1) * delta
            #extr = x0 + (np.sqrt(2)-1) * delta

        sit = sit + 1 # succesful iteration number

        # test/compare
        ext_t = from_numpy(extr)
        camera_ext = ext_t[:n_cameras*9].reshape(n_cameras,9)
        point_ext  = ext_t[n_cameras*9:].reshape(n_points,3)

        fx1 = funx0_st1(camera_ext[camera_indices[:]], point_ext[point_indices[:]], torch_points_2d[:,:])
        costExt = np.sum(fx1.numpy()**2)
        delta_ext_p = extr[:n_cameras*9] - (x0_p - delta_p)
        delta_ext_l = extr[n_cameras*9:] - (x0_l - delta_l)
        costExtPenalty = costExt + L * (delta_ext_p.dot(JtJDiag * delta_ext_p) + delta_ext_l.dot(JltJlDiag * delta_ext_l))
        print(it, "it. cost ext   ", round(costExt), "      with penalty ", round(costExtPenalty))
        #L = max(L0, L/2)
        # unsure about exact check. Does one ever work, why does the other lead to issues.
        if useExtInCost and costExtPenalty < costEndPenalty: #costStart: # todo: + penalty again
        #if useExtInCost and costExt < costEnd and costExtPenalty < costStart: # todo: + penalty again
            x0 = extr
            x0_p = x0[:9 * n_cameras]
            x0_l = x0[9 * n_cameras:]
            x0_t = ext_t
            camera_params = camera_ext
            point_params  = point_ext
        if False and costExt > 1000*costEnd: # maybe just clean last step?
            print("Reset BFGS buffer") # nope
            sit = 0
            bfgs_qs = np.zeros([bfgs_mem, n]) # access/write with % mem
            bfgs_ps = np.zeros([bfgs_mem, n])
            bfgs_rhos = np.zeros([bfgs_mem, 1])

    # nope: use smart tr check not penalty
    tr_check = (costStart - costEndPenalty) / (costStart - costQuadPenalty)
    eta_1 = 0.8
    eta_2 = 0.25
    if tr_check > eta_1:
        L = L / 2
    if tr_check < eta_2:
        L = L * 2

    if write_output:
        camera_params.numpy().tofile("camera_params.dat")
        point_params.numpy().tofile("point_params.dat")
