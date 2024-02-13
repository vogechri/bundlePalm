from __future__ import print_function
import urllib
import bz2
import os
import numpy as np
import time
import copy
from joblib import Parallel, delayed
from scipy.sparse import csr_array, csr_matrix
from scipy.sparse import diags as diag_sparse
from scipy.sparse.linalg import inv as inv_sparse
from numpy.linalg import inv as inv_dense
from numpy.linalg import eigvalsh, eigh

# idea reimplement projection with torch to get a jacobian -> numpy then 
import torch
import math
import open3d as o3d
from torch.autograd.functional import jacobian
from torch import tensor, from_numpy


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
    points_projX = points_cam[:, 0] / points_cam[:, 2]
    points_projY = points_cam[:, 1] / points_cam[:, 2]

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

def rerender(vis, geometry, landmarks, save_image):
    geometry.points = o3d.utility.Vector3dVector(landmarks) # ?
    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    if save_image:
        vis.capture_screen_image("temp_%04d.jpg" % i)
    #vis.destroy_window()

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

# R(p-c), c: cam location in world = Rp -Rc = R|t * p|1. t=-Rc
# estimate R|t, then - R^t * t =- R^t * -Rc = c
# also make line from c to 0.
def torchResiduum_(rot, tra, cam, p3d, p2d, camera_indices, point_indices) :
    points_cam = AngleAxisRotatePoint(rot[camera_indices,:], p3d[point_indices,:])

    points_cam = points_cam + tra[camera_indices,:]
    points_projX = points_cam[:, 0] / points_cam[:, 2]
    points_projY = points_cam[:, 1] / points_cam[:, 2]

    f  = cam[camera_indices, 0]
    k1 = cam[camera_indices, 1]
    k2 = cam[camera_indices, 2]

    r2 = points_projX*points_projX + points_projY*points_projY
    distortion = 1. + r2 * (k1 + k2 * r2)
    points_reprojX = points_projX * distortion * f
    points_reprojY = points_projY * distortion * f
    resX = (points_reprojX-p2d[:,0]).reshape((p2d.shape[0], 1))
    resY = (points_reprojY-p2d[:,1]).reshape((p2d.shape[0], 1))
    residual = torch.cat([resX[:,], resY[:,]], dim=1)
    return residual

# derivative of rotation around point becomes ?
def torchRotation_(rot, p3d, camera_indices, point_indices) :
    points_cam = AngleAxisRotatePoint(rot[camera_indices,:], p3d[point_indices,:])
    return points_cam

def getPrecisionMatrix(points_3d_t, cam_model_t, angle = 20, dist = 200):
    verbose = False
    # observing form 20 degree angle at 200 cm distance. 
    rot = np.array([0,math.pi/180 * angle, 0], dtype=np.float64) # as axis times angle, z is negative 
    tra = np.array([0,0,dist], dtype=np.float64) # 2 meter away but in front of anchor / QR
    # cam location is 
    cam = AngleAxisRotatePoint(from_numpy(-rot[np.newaxis, :]), from_numpy(tra[np.newaxis, :])).numpy().flatten()
    # t is -R*c
    tra = -AngleAxisRotatePoint(from_numpy(rot[np.newaxis, :]), from_numpy(cam[np.newaxis, :])).numpy().flatten()

    print("getPrecisionMatrix cam ", cam, " tra ", tra, " angle ", angle, " z-dist ", dist)
    rot = rot.reshape((1,3))
    tra = tra.reshape((1,3))
    rot_init = rot.copy().flatten()
    tra_init = tra.copy().flatten()

    verbose = False
    # f * (R*p +t / |R*p +t|) no minus!
    rot_t = from_numpy(rot)
    tra_t = from_numpy(tra)
    camIds = np.array([0,0,0])

    # compute 2d observations for gt, later add noise. depends on current rot, tra
    points_cam = AngleAxisRotatePoint(rot_t[camIds,:], points_3d_t)
    points_cam = points_cam + tra_t[camIds,:]
    points_projX = f*points_cam[:, 0] / points_cam[:, 2]
    points_projY = f*points_cam[:, 1] / points_cam[:, 2]
    points_2d = np.hstack((points_projX.numpy(), points_projY.numpy())).reshape(2,3).transpose()

    #print("points_2d ", points_2d)
    if verbose: # 0,0 image center
        print("points_2d ", points_2d[:,:])

    rot_t = from_numpy(rot_init.copy().reshape(1,-1))
    tra_t = from_numpy(tra_init.copy().reshape(1,-1))

    points_2d_t = from_numpy(points_2d)
    points_2d_t = points_2d_t.requires_grad_(False)
    camera_indices = [0,0,0]
    point_indices = [0,1,2]
    funx0_t = lambda X0, X1: torchResiduum_(X0, X1, cam_model_t[camera_indices,:], 
                                            points_3d_t[point_indices,:], points_2d_t, 
                                            camera_indices, point_indices)

    jac = jacobian(funx0_t, (rot_t, tra_t), create_graph=False, vectorize=True, strategy='reverse-mode') #forward-mode
    J = torch.cat( [jac[0].flatten().reshape(-1,3), jac[1].flatten().reshape(-1,3) ], dim=1).numpy()
    # res = funx0_t(rot_t, tra_t)
    # fx0 = res.flatten().numpy()[:,np.newaxis]
    # by hand to compare to !
    # ∂R(v)u/∂v =−R[u]× (vv⊤+(R⊤−Id)[v]× ) / v 2

    funx0_t = lambda X0: torchRotation_(X0, points_3d_t[point_indices,:], camera_indices, [0])
    jac = jacobian(funx0_t, (rot_t), create_graph=False, vectorize=True, strategy='reverse-mode') #forward-mode
    print("J_rot ", jac[0])

    funx0_t = lambda X0: torchRotation_(X0, points_3d_t[point_indices,:], camera_indices, [1])
    jac = jacobian(funx0_t, (rot_t), create_graph=False, vectorize=True, strategy='reverse-mode') #forward-mode
    print("J_rot ", jac[0])

    funx0_t = lambda X0: torchRotation_(X0, points_3d_t[point_indices,:], camera_indices, [2])
    jac = jacobian(funx0_t, (rot_t), create_graph=False, vectorize=True, strategy='reverse-mode') #forward-mode
    print("J_rot ", jac[0])

    print("sum fron those points_3d_t ", points_3d_t)

    #JtJ = J.transpose() * J
    print("J ", J)
    #JtJ = J.dot(J)
    JtJ = J.transpose().dot(J)
    #print("JtJ ", JtJ) # ok?
    #print("JtJ ", J.transpose()*J)
    return JtJ

def fit_pose(rot_init, tra_init, points_3d_t, cam_model_t, points_2d, sigma=1, iterations=20):

    print_debug = False
    verbose = False

    noise = np.random.normal(0, sigma, 6).reshape(points_2d.shape)
    rot_t = from_numpy(rot_init.copy().reshape(1,-1))
    tra_t = from_numpy(tra_init.copy().reshape(1,-1))

    L = 1000
    # optimize for R,t from (noisy) 2d points and cam geometry. init from gt?
    points_2d_t = from_numpy(points_2d + noise)
    points_2d_t = points_2d_t.requires_grad_(False)
    #print("points_2d_t ", points_2d_t)

    camera_indices = [0,0,0]
    point_indices = [0,1,2]
    funx0_t = lambda X0, X1: torchResiduum_(X0, X1, cam_model_t[camera_indices,:], 
                                            points_3d_t[point_indices,:], points_2d_t, 
                                            camera_indices, point_indices)
    totalCostBegin = 0
    totalCostEnd = 0
    for it in range(iterations):
        jac = jacobian(funx0_t, (rot_t, tra_t), create_graph=False, vectorize=True, strategy='reverse-mode') #forward-mode
        J = torch.cat( [jac[0].flatten().reshape(-1,3), jac[1].flatten().reshape(-1,3) ], dim=1).numpy()
        res = funx0_t(rot_t, tra_t)
        fx0 = res.flatten().numpy()[:,np.newaxis]

        if it ==0:
           totalCostBegin = np.sum(fx0**2)
        if it ==iterations-1:
           totalCostEnd = np.sum(fx0**2)

        if verbose:
            print(jac[0].shape) # R 3, 2, 1, 3, 0.#res 1.x/y 3. var ids.
            print(jac[1].shape) # t
            print(res.shape) # 3 x 2
            # fx + (r,t)
            print(jac[0].flatten().reshape(-1,3))
            print(jac[1].flatten().reshape(-1,3))

        #JtJ = J.transpose() * J
        JtJ = J.transpose().dot(J)

        if verbose:
            print("J", J.shape, " \n", J)
            print("fx0 ",fx0.shape, fx0)
            print("JtJ", JtJ.shape, " \n", JtJ)

        # is this better? or not?
        #J_eps = 1e-4
        #JtJDiag = JtJ + J_eps * diag_sparse(np.ones(JtJ.shape[0]))
        #JtJDiag = diag_sparse(np.fmax(JtJ.diagonal(), 1e-4))

        # JtJDiag = diag_sparse(np.ones(JtJ.shape[0]))
        JtJDiag = diag_sparse(np.ones(JtJ.shape[0]))

        JJ = JtJ + L * JtJDiag
        b = J.transpose().dot(fx0)
        JJi = inv_dense(JJ)
        delta = -JJi.dot(b)
        costStart = np.sum(fx0**2)
        fx0_new = fx0 + J.dot(delta)
        costQuad = np.sum(fx0_new.transpose().dot(fx0_new)) #np.sum(fx0_new**2)
        penalty = L * np.sum(delta.transpose().dot(JtJDiag.dot(delta)))
        costQuadPenalty = costQuad + penalty

        if verbose:
            print("JtJDiag ", JJ.shape, " ", JJ)
            print("JJi ", JJi.shape, " ", JJi)
            print("b ", b.shape, " ", b)
            print( "JtJ * delta ", JtJ * delta) 
            print("delta ", delta.shape, " ", delta)

            print("fx0_new ", fx0_new.shape)
            print("fx0 ", fx0.shape)
            print(costQuad)
            print("penalty ", penalty)
            print(rot_t.shape)
            print(delta.shape)
            print(from_numpy(delta[:3,:].transpose()).shape)
            print(costStart.shape, " ", costQuad.shape, " ", costEnd.shape)

        rot_t += from_numpy(delta[:3,:].transpose())
        tra_t += from_numpy(delta[3:,:].transpose())
        fx1 = funx0_t(rot_t, tra_t)
        costEnd = np.sum(fx1.numpy()**2)
        costEndPenalty = costEnd + penalty

        if print_debug:
            print(it, "it. cost 0:     ", costStart)
            print(it, "it. cost 0 new: ", costQuad, "     with penalty ", costQuadPenalty)
            print(it, "it. cost 1:     ", costEnd,  "     with penalty ", costEndPenalty)

        if costStart < costEnd:
            rot_t -= from_numpy(delta[:3,:].transpose())
            tra_t -= from_numpy(delta[3:,:].transpose())
            updateJacobian = False
        else:
            updateJacobian = True
            it = it + 1

        tr_check = (costStart - costEndPenalty) / (costStart - costQuadPenalty)
        eta_1 = 0.8
        eta_2 = 0.25
        if tr_check > eta_1 and costStart > costEndPenalty:
            L = L / 2
        if tr_check < eta_2 or costStart < costEndPenalty:
            L = L * 2
        if print_debug:
            print("costStart < costEndPenalty", costStart < costEndPenalty, " tr_check ", tr_check, " L ", L)
        #print("rot", rot, " tra ", tra, "| rot_init ", rot_init, " tra_init ", tra_init)
        # store solution -- store JTJ from no noise case. 
        # plot
    print("cost gain ",  totalCostBegin - totalCostEnd, "costBegin", totalCostBegin, " totalCostEnd ", totalCostEnd)
    return rot_t.numpy().flatten(), tra_t.numpy().flatten()

def getModels(points_3d_t, cam_model_t, examples = 20, angle = 20, dist = 200, sigma = 1.0, iterations = 40):
    verbose = False
    # from 0,0-200 straight onto 0,0,0 z looks at
    #rot = np.array([0,1e-12,0], dtype=np.float64) # as axis times angle, z is negative 
    #tra = np.array([0,0,200], dtype=np.float64) # 2 meter away but in front of anchor / QR

    # observing form 20 degree angle at 200 cm distance. 
    rot = np.array([0,math.pi/180 * angle, 0], dtype=np.float64) # as axis times angle, z is negative 
    tra = np.array([0,0,dist], dtype=np.float64) # 2 meter away but in front of anchor / QR
    # cam location is 
    cam = AngleAxisRotatePoint(from_numpy(-rot[np.newaxis, :]), from_numpy(tra[np.newaxis, :])).numpy().flatten()
    # t is -R*c
    tra = -AngleAxisRotatePoint(from_numpy(rot[np.newaxis, :]), from_numpy(cam[np.newaxis, :])).numpy().flatten()

    print("cam ", cam, " tra ", tra, " angle ", angle, " z-dist ", dist)

    rot = rot.reshape((1,3))
    tra = tra.reshape((1,3))
    rot_init = rot.copy().flatten()
    tra_init = tra.copy().flatten()
    cam_init = cam.copy().flatten()
    cameras = np.concatenate((rot, tra))

    if False: # move cam to origin here. does nothing.
        for i in range(points_3d.shape[0]):
            points_3d[i,:] = points_3d[i,:] - tra
        tra -= tra

    verbose = False
    # model / cost FIXED
    # f * (R*p +t / |R*p +t|) no minus!
    # 
    rot_t = from_numpy(rot)
    tra_t = from_numpy(tra)

    camIds = np.array([0,0,0])

    # compute 2d observations for gt, later add noise. depends on current rot, tra
    points_cam = AngleAxisRotatePoint(rot_t[camIds,:], points_3d_t)
    points_cam = points_cam + tra_t[camIds,:]
    points_projX = f*points_cam[:, 0] / points_cam[:, 2]
    points_projY = f*points_cam[:, 1] / points_cam[:, 2]
    points_2d = np.hstack((points_projX.numpy(), points_projY.numpy())).reshape(2,3).transpose()

    print("points_2d ", points_2d)
    if verbose: # 0,0 image center
        print("points_2d ", points_2d[:,:])

    # given noise fit n times and visualize ()
    rot_fits = []
    tra_fits = []
    np.random.seed(42)
    for runs in range(examples):
        rot_fit, tra_fit = fit_pose(rot_init, tra_init, points_3d_t, cam_model_t, points_2d, sigma=sigma, iterations=iterations)
        rot_fits.append(rot_fit.copy())
        tra_fits.append(tra_fit.copy())

        # return is R, -Rc ->  -R^t t = c. 
        # should plot location GT and where the anchor actually is, plot also fitted locations.
        # better (R_gt|t_gt)^-1 

        #print(runs, " rot", rot_fit, " tra ", tra_fit, "| rot_init ", rot_init, " tra_init ", tra_init)
        print(runs, " rot", np.abs(rot_fit-rot_init), " tra ", np.abs(tra_fit-tra_init))

    # plot test
    if True or o3d_defined:
        # should plot here

        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        #vis = o3d.visualization.Visualizer()
        #vis.create_window()
        #geometry = o3d.geometry.PointCloud()

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.6, origin=[0, 0, 0])
        gt_location = copy.deepcopy(origin)
        #source_mesh.rotate(source_mesh.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi / 4)), center=(0, 0, 0))
        gt_location.rotate(gt_location.get_rotation_matrix_from_xyz(-rot_init), center=(0, 0, 0)) # ok
        gt_location.translate(cam_init) #  / 10 does not work / show geometry any more

        meshes = []
        for id in range(len(rot_fits)):
            fit_location = copy.deepcopy(origin)
            fit_location.rotate(fit_location.get_rotation_matrix_from_xyz(-rot_fits[id]), center=(0, 0, 0)) # ok
            # pose location is t = -Rc, c = -R^t*t
            cam_fit = -AngleAxisRotatePoint(from_numpy(-rot_fits[id][np.newaxis, :]), from_numpy(tra_fits[id][np.newaxis, :])).numpy().flatten()
            fit_location.translate(cam_fit)
            #fit_location.translate(tra_fits[id])
            # mix color, eg. yellow - different shades of yellow

            da = (angle + 60.) / 120. # -60 to 60
            fit_location.paint_uniform_color([da, 1.0-da, 0.3])

            #rot_fits = np.vstack([rot_fits, rot_fits[id]])
            #tra_fits = np.vstack([tra_fits, rot_fits[id]])
            meshes.append(fit_location)
        meshes.append(gt_location)
        meshes.append(origin)

        #o3d.visualization.draw_geometries([mesh_frame, mesh_frame_b, geometry])
        # avoid draw of origin since visualization sucks if

        # line to orient my self.
        #gt to origin.

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([[0,0,0],[cam_init[0], cam_init[1], cam_init[2]]]),
            lines=o3d.utility.Vector2iVector([[0, 1]]) )
        meshes.append(line_set)
    return meshes

# test with our simple case. use optimization to find 'best' R|t, drop fd. 
# GT R|t, points are fix and known, pixel computed from ideal no noise case
# add noise, solve -> R', t'
# plot t and maybe R? for many runs.
# plot axis  on 3d point location as assumed. 
# pi 3d points known, p0 = 0, p1 = p0 + x * 20cm, p2 = p0 + y * 20cm
# can also plot where pi are assumed to be located, use gt t, use compute R',t' to define the 3d points as assumed to be
# plot axis at p0 as observed from cam. 

# 20 cm 3 corners
points_3d = np.array([[-10,-10,0], [10,-10,0], [-10,10,0]], dtype=np.float64)
#points_3d = np.array([[0,0,0], [20,0,0], [0,20,0]], dtype=np.float64)
points_3d_t = from_numpy(points_3d)
points_3d_t.requires_grad_(False)

# define focal distance and compute observations wo. noise
f = 600 # fx=fy=f
c = 300 # cx=cy=c # maybe not needed, not modeled anyway
cam_model = np.array([[f,0,0]], dtype=np.float64) # no distortion! also fix it!
cam_model_t = from_numpy(cam_model)
cam_model_t.requires_grad_(False)

## non fixed stuff, rot and tra gt. from where do we look at the qr code. ##
# desire to do from n views: define view -> get and plot primitives.

iterations = 40
sigma = 0.5
dist = 100
examples = 40


J = getPrecisionMatrix(points_3d_t, cam_model_t, angle = -40, dist = 100)
eigenvalues, eigenvectors = eigh(J[:3,:3])
print("Evals ", eigenvalues)
print("Evecs \n", eigenvectors)
# 1. 3x3 block. 
J = getPrecisionMatrix(points_3d_t, cam_model_t, angle = -20, dist = 100)
eigenvalues, eigenvectors = eigh(J[:3,:3])
print("Evals ", eigenvalues)
print("Evecs \n", eigenvectors)
J = getPrecisionMatrix(points_3d_t, cam_model_t, angle =   0, dist = 100)
eigenvalues, eigenvectors = eigh(J[:3,:3])
print("J ", J)
print("Evals ", eigenvalues)
print("Evecs \n", eigenvectors)
J = getPrecisionMatrix(points_3d_t, cam_model_t, angle =  20, dist = 100)
eigenvalues, eigenvectors = eigh(J[:3,:3])
print("Evals ", eigenvalues)
print("Evecs \n", eigenvectors)
J = getPrecisionMatrix(points_3d_t, cam_model_t, angle =  40, dist = 100)
eigenvalues, eigenvectors = eigh(J[:3,:3])
print("Evals ", eigenvalues)
print("Evecs \n", eigenvectors)

meshes3 = getModels(points_3d_t, cam_model_t, examples = examples, angle = -40, dist = dist, sigma=sigma, iterations=iterations)
meshes2 = getModels(points_3d_t, cam_model_t, examples = examples, angle = -20, dist = dist, sigma=sigma, iterations=iterations)
meshes  = getModels(points_3d_t, cam_model_t, examples = examples, angle =   0, dist = dist, sigma=sigma, iterations=iterations)
meshes1 = getModels(points_3d_t, cam_model_t, examples = examples, angle =  20, dist = dist, sigma=sigma, iterations=iterations)
meshes4 = getModels(points_3d_t, cam_model_t, examples = examples, angle =  40, dist = dist, sigma=sigma, iterations=iterations)

o3d.visualization.draw_geometries(meshes + meshes1 + meshes2 + meshes3 + meshes4)
#o3d.visualization.draw_geometries(meshes2)

################### 
#o3d.visualization.draw_geometries([gt_location, fit_location])

# test = np.vstack([tra_init, tra])
# rerender(vis, geometry, test, False)
#vis.destroy_window()
#exit()

# import time
# time.sleep(2.5)

# solve (x-mu1)^t L1 (x-mu1) + (x-mu2)^t L2 (x-mu2), mu_i solutions, L_i: precision of problem i
# L1 (x-mu1) + L2 (x-mu2) = 0  <=> x = [sum_i L_i ]^-1 (sum_i Li mu_i)