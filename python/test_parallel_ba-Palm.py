from __future__ import print_function
import urllib
import bz2
import os
import time
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_array, csr_matrix, issparse, vstack
from scipy.sparse import diags as diag_sparse
from scipy.sparse.linalg import inv as inv_sparse
from numpy.linalg import pinv as inv_dense
from numpy.linalg import eigvalsh 
# idea reimplement projection with torch to get a jacobian -> numpy then
import torch
import math
import ctypes
from torch.autograd.functional import jacobian
from torch import tensor, from_numpy, flatten

# look at website. This is the smallest problem. guess: pytoch cpu is pure python?
BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
#FILE_NAME = "problem-49-7776-pre.txt.bz2"
FILE_NAME = "problem-73-11032-pre.txt.bz2"
#FILE_NAME = "problem-138-19878-pre.txt.bz2"

# BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/trafalgar/"
# FILE_NAME = "problem-21-11315-pre.txt.bz2"

#BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/dubrovnik/"
#FILE_NAME = "problem-16-22106-pre.txt.bz2"
#FILE_NAME = "problem-135-90642-pre.txt.bz2"
#FILE_NAME = "problem-173-111908-pre.txt.bz2"
#FILE_NAME = "problem-237-154414-pre.txt.bz2"
#FILE_NAME = "problem-356-226730-pre.txt.bz2" # large dub, play with ideas: cover, etc

# BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/venice/"
# FILE_NAME = "problem-52-64053-pre.txt.bz2"

# BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/final/"
# FILE_NAME = "problem-93-61203-pre.txt.bz2"

#BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/dubrovnik/"
#FILE_NAME = "problem-16-22106-pre.txt.bz2"

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


def AngleAxisRotatePoint(angleAxis, pt):
    theta2 = (angleAxis * angleAxis).sum(dim=1)

    mask = (theta2 > 0).float()  # ? == 0 is alternative? check other repo

    theta = torch.sqrt(theta2 + (1 - mask))

    mask = mask.reshape((mask.shape[0], 1))
    mask = torch.cat([mask, mask, mask], dim=1)

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    thetaInverse = 1.0 / theta

    w0 = angleAxis[:, 0] * thetaInverse
    w1 = angleAxis[:, 1] * thetaInverse
    w2 = angleAxis[:, 2] * thetaInverse

    wCrossPt0 = w1 * pt[:, 2] - w2 * pt[:, 1]
    wCrossPt1 = w2 * pt[:, 0] - w0 * pt[:, 2]
    wCrossPt2 = w0 * pt[:, 1] - w1 * pt[:, 0]

    tmp = (w0 * pt[:, 0] + w1 * pt[:, 1] + w2 * pt[:, 2]) * (1.0 - costheta)

    r0 = pt[:, 0] * costheta + wCrossPt0 * sintheta + w0 * tmp
    r1 = pt[:, 1] * costheta + wCrossPt1 * sintheta + w1 * tmp
    r2 = pt[:, 2] * costheta + wCrossPt2 * sintheta + w2 * tmp

    r0 = r0.reshape((r0.shape[0], 1))
    r1 = r1.reshape((r1.shape[0], 1))
    r2 = r2.reshape((r2.shape[0], 1))

    res1 = torch.cat([r0, r1, r2], dim=1)

    wCrossPt0 = angleAxis[:, 1] * pt[:, 2] - angleAxis[:, 2] * pt[:, 1]
    wCrossPt1 = angleAxis[:, 2] * pt[:, 0] - angleAxis[:, 0] * pt[:, 2]
    wCrossPt2 = angleAxis[:, 0] * pt[:, 1] - angleAxis[:, 1] * pt[:, 0]

    r00 = pt[:, 0] + wCrossPt0
    r01 = pt[:, 1] + wCrossPt1
    r02 = pt[:, 2] + wCrossPt2

    r00 = r00.reshape((r00.shape[0], 1))
    r01 = r01.reshape((r01.shape[0], 1))
    r02 = r02.reshape((r02.shape[0], 1))

    res2 = torch.cat([r00, r01, r02], dim=1)

    return res1 * mask + res2 * (1 - mask)


def torchSingleResiduum(camera_params_, point_params_, p2d):
    angle_axis = camera_params_[:, :3]
    points_cam = AngleAxisRotatePoint(angle_axis, point_params_)
    points_cam = points_cam + camera_params_[:, 3:6]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]
    f = camera_params_[:, 6]
    k1 = camera_params_[:, 7]
    k2 = camera_params_[:, 8]
    r2 = points_projX * points_projX + points_projY * points_projY
    distortion = 1.0 + r2 * (k1 + k2 * r2)
    points_reprojX = points_projX * distortion * f
    points_reprojY = points_projY * distortion * f
    resX = (points_reprojX - p2d[:, 0]).reshape((p2d.shape[0], 1))
    resY = (points_reprojY - p2d[:, 1]).reshape((p2d.shape[0], 1))
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

def getJacSin(
    start_,
    end_,
    camera_indices_,
    point_indices_,
    camera_params_,
    point_params_,
    torch_points_2d_,
):
    end_ = min(end_, camera_indices_.shape[0])
    funx0_st1 = lambda X0, X1, X2: torchSingleResiduum(
        X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2)
    )
    jac = jacobian(
        funx0_st1,
        (
            camera_params_[camera_indices_[start_:end_], :],
            point_params_[point_indices_[start_:end_], :],
            torch_points_2d_[start_:end_, :],
        ),
        create_graph=False,
        vectorize=True,
        strategy="reverse-mode",
    )

    # print(start_, " ", end_, ", jac shapes : ", jac[0].shape, " x ", jac[1].shape)
    res = funx0_st1(
        camera_params_[camera_indices_[start_:end_], :],
        point_params_[point_indices_[start_:end_], :],
        torch_points_2d_[start_:end_, :],
    )
    # print("res", res.shape) # 200,2, so N, x/y
    return (jac[0], jac[1], res)


### could be removed / not used
def torchResiduum(x0T, n_cameras_, n_points_, camera_indices_, point_indices_, p2d_):
    # x0T = from_numpy(x0)
    # p2d = from_numpy(points_2d) # outside is better?
    # p2d.requires_grad_(False)
    camera_params = x0T[: n_cameras_ * 9].reshape(n_cameras_, 9)
    point_params = x0T[n_cameras_ * 9 :].reshape(n_points_, 3)
    angle_axis = camera_params[:, :3]

    # likely better to create per point/cam representation 1st. no slower
    # rot_matrix = AngleAxisToRotationMatrix(angle_axis) #
    # points_cam = rot_matrix[camera_indices_, :, 0] * point_params[point_indices_,0].view(-1,1) + rot_matrix[camera_indices_, :, 1] * point_params[point_indices_,1].view(-1,1) + rot_matrix[camera_indices_, :, 2] * point_params[point_indices_,2].view(-1,1)

    points_cam = AngleAxisRotatePoint(
        angle_axis[camera_indices_, :], point_params[point_indices_, :]
    )

    points_cam = points_cam + camera_params[camera_indices_, 3:6]
    points_projX = -points_cam[:, 0] / points_cam[:, 2]
    points_projY = -points_cam[:, 1] / points_cam[:, 2]

    f = camera_params[camera_indices_, 6]
    k1 = camera_params[camera_indices_, 7]
    k2 = camera_params[camera_indices_, 8]
    # f.requires_grad_(False) # not leaf?!
    # k1.requires_grad_(False)
    # k2.requires_grad_(False)
    r2 = points_projX * points_projX + points_projY * points_projY
    distortion = 1.0 + r2 * (k1 + k2 * r2)
    points_reprojX = -points_projX * distortion * f
    points_reprojY = -points_projY * distortion * f
    resX = (points_reprojX - p2d_[:, 0]).reshape((p2d_.shape[0], 1))
    resY = (points_reprojY - p2d_[:, 1]).reshape((p2d_.shape[0], 1))
    residual = torch.cat([resX[:,], resY[:,]], dim=1)
    return residual


def getJac(
    start_,
    end_,
    camera_indices_,
    point_indices_,
    camera_params_,
    point_params_,
    torch_points_2d_,
):
    end_ = min(end_, camera_indices.shape[0])
    n_cameras_ = camera_params_.shape[0]
    n_points_ = point_params_.shape[0]
    funx0_t = lambda X0: torchResiduum(
        X0,
        n_cameras_,
        n_points_,
        camera_indices_[start_:end_],
        point_indices_[start_:end_],
        torch_points_2d_[start_:end_],
    )
    x0_t_ = torch.hstack((camera_params_.flatten(), point_params_.flatten()))
    jac = jacobian(
        funx0_t, x0_t_, create_graph=False, vectorize=True, strategy="reverse-mode"
    )  # forward-mode
    print(start_, " ", end_, " : ")
    return jac


def buildMatrix(step, full, results_, camera_indices_, point_indices_, varset=0):
    data = []
    indptr = []
    indices = []
    if varset == 0:
        v_indices = camera_indices_
        sz = 9
    if varset == 1:
        v_indices = point_indices_
        sz = 3
    dataIndexSet = []

    for j in range(step):
        dataIndexSet.append(
            j * 2 * step * sz + 0 * sz * step + j * sz + np.arange(0, sz)
        )
    for j in range(step):
        dataIndexSet.append(
            j * 2 * step * sz + 1 * sz * step + j * sz + np.arange(0, sz)
        )
    dataIndices = np.concatenate(dataIndexSet)

    for i in np.arange(0, full, step):  # i, i + step: (0, 400), etc
        start_ = i
        end_ = min(i + step, v_indices.shape[0])
        # numLocalRows = results_[int(i/step)][varset].shape[0]
        # results_ hold res as resId, x/y, cam/point id, 9 or 3. only current cam if holds data!
        # print(i/step, " ", start, "-", end)
        # print(results_[int(i/step)][varset].shape) # 200,2,200,9 | 43,2,43,9
        # print(np.arange(start*sz, end*sz, sz).shape) # 200 | 43
        # print(np.array([sz * camera_indices[start:end] + j for j in range(sz)]).transpose().flatten().shape) # 1800 | 387

        # bottleneck? how to improve though? maybe 1 x one y and combine later.
        # can have index set?
        if end_ != i + step:  # else uber slow
            for j in range(end_ - start_):
                data.append(
                    results_[int(i / step)][varset][j, 0, j, :]
                    .flatten()
                    .detach()
                    .numpy()
                )
            for j in range(end_ - start_):
                data.append(
                    results_[int(i / step)][varset][j, 1, j, :]
                    .flatten()
                    .detach()
                    .numpy()
                )
                # print(data[-1].shape) # 18, so x/y x/y ... now first all x res then all y res.
        else:
            # print(i/step, " ", dataIndices.shape)
            data.append(
                results_[int(i / step)][varset].flatten()[dataIndices].detach().numpy()
            )
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
        indptr.append(np.arange(2 * start_ * sz, 2 * end_ * sz, sz).flatten())
        # indptr.append(np.arange((2*start+1)*sz, (sz*end+1)*sz, sz))
        # x&y: operate on same cam variables
        # first all x res, then all y res here.
        indices.append(
            np.array([sz * v_indices[start_:end_] + j for j in range(sz)])
            .transpose()
            .flatten()
        )
        indices.append(
            np.array([sz * v_indices[start_:end_] + j for j in range(sz)])
            .transpose()
            .flatten()
        )

        # n = 9 * n_cameras + 3 * n_points
        # m = 2 * points_2d.shape[0]
        # crs_pose = csr_array((np.array(data).flatten(), np.array(indices).flatten(), np.array(indptr).flatten()), shape=(2*full, 9 * n_cameras)).toarray()
    indptr.append(np.array([sz + indptr[-1][-1]]))  # closing
    if False:
        print("indptr ", np.concatenate(indptr))
        print("indptr len ", len(indptr))
        print("---------")
        print(
            "data ",
            np.concatenate(data).shape,
            " ",
            min(np.concatenate(data)),
            " ",
            max(np.concatenate(data)),
        )
        print(
            "indptr ",
            np.concatenate(indptr).shape,
            " ",
            min(np.concatenate(indptr)),
            " ",
            max(np.concatenate(indptr)),
        )
        print(
            "indices ",
            np.concatenate(indices).shape,
            " ",
            min(np.concatenate(indices)),
            " ",
            max(np.concatenate(indices)),
        )

    # toarray is slow and does what? nothing? makes a dense matrix .. OMG
    datavals = np.concatenate(data)
    # debug: set all inner parameters to 0
    if False and varset == 0:
        # datavals[0:end:9] = 0
        # datavals[1:end:9] = 0
        # datavals[2:end:9] = 0

        # datavals[3:end:9] = 0
        # datavals[4:end:9] = 0
        # datavals[5:end:9] = 0

        datavals[6:end:9] = 0
        datavals[7:end:9] = 0
        datavals[8:end:9] = 0
    crs_pose = csr_array((datavals, np.concatenate(indices), np.concatenate(indptr)))
    J_pose = csr_matrix(crs_pose)
    return J_pose


def buildResiduum(step, full, results_, varset=2):
    data = []
    for i in np.arange(0, full, step):  # i, i + step: (0, 400), etc
        # start = i
        # end = min(i + step, camera_indices_.shape[0])
        # results_ hold res as resId, x/y, cam/point id, 9 or 3. only current cam if holds data!
        # print(i/step, " ", start, "-", end)
        # print(results_[int(i/step)][varset].shape) # 200,2,200,9 | 43,2,43,9
        # print(np.arange(start*sz, end*sz, sz).shape) # 200 | 43
        # print(np.array([sz * camera_indices[start:end] + j for j in range(sz)]).transpose().flatten().shape) # 1800 | 387
        # print("res shape", results_[int(i/step)][varset][:,0].flatten().numpy().shape())
        # print("res ", results_[int(i/step)][varset][0].numpy())
        # for j in range(end-start):
        #    data.append(results_[int(i/step)][varset][0,:].flatten().numpy())
        # for j in range(end-start):
        #    data.append(results_[int(i/step)][varset][1,:].flatten().numpy())
        data.append(results_[int(i / step)][varset][:, 0].flatten().numpy())
        data.append(results_[int(i / step)][varset][:, 1].flatten().numpy())

    # print("---------")
    # print(np.concatenate(data).shape)
    res = np.concatenate(data)
    # print(res)
    return res


# fx0 + J delta x, J is Jp|Jl * xp|xl
def ComputeDerivativeMatrices(
    x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d
):
    start_ = time.time()
    step = 250
    full = camera_indices_.shape[0]
    # camera_params = x0_t[:n_cameras*9].reshape(n_cameras,9)
    # point_params  = x0_t[n_cameras*9:].reshape(n_points,3)
    # funx0_st1 = lambda X0, X1, X2: torchSingleResiduum(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2))
    # results = Parallel(n_jobs=8,prefer="threads")(delayed(getJac)(i, i + step) for i in np.arange(0, full, step))

    # parallel version
    # results = Parallel(n_jobs=8,prefer="threads")(delayed(getJacSin)(i, i + step, camera_indices_, point_indices_, x0_t_cam, x0_t_land, torch_points_2d) for i in np.arange(0, full, step))
    # not parallel
    results_ = []
    for i in np.arange(0, full, step):
        result = getJacSin(
            i,
            i + step,
            camera_indices_,
            point_indices_,
            x0_t_cam,
            x0_t_land,
            torch_points_2d,
        )
        # result = getJac(i, i + step, camera_indices_, point_indices_, x0_t_cam, x0_t_land, torch_points_2d) # awfully slow
        results_.append(result)

    end_ = time.time()
    print("Parallel ", len(results_), " ", step, " its take ", end_ - start_, "s")
    # print(results_[0][0].shape, " x ", results_[0][1].shape)
    # now compose sparse! jacobian
    # define sparsity pattern and add data
    # results_ hold res as resId, x/y, cam/point id, 9 or 3

    start = time.time()  # bottleneck now
    J_pose = buildMatrix(
        step, full, results_, camera_indices_, point_indices_, varset=0
    )
    end = time.time()
    # print(" build Matrix & residuum took ", end-start, "s")
    start = time.time()  # bottleneck now
    J_land = buildMatrix(
        step, full, results_, camera_indices_, point_indices_, varset=1
    )  # slow
    fx0 = buildResiduum(step, full, results_, varset=2)  # takes 1%
    end = time.time()
    # print(" build Matrix & residuum took ", end-start, "s")

    # print(J_pose.shape)
    return (J_pose, J_land, fx0)

def ComputeDerivativeMatricesNew(x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d
):
    verbose = False
    if verbose:
        start = time.time() # this is not working at all. Slower then iteratively

    funx0_st1 = lambda X0, X1, X2: torchSingleResiduumX(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d fucntion -> grad possible
    funy0_st1 = lambda X0, X1, X2: torchSingleResiduumY(X0.view(-1,9), X1.view(-1,3), X2.view(-1,2)) # 1d fucntion -> grad possible

    torch_cams = x0_t_cam[camera_indices_[:],:] #x0_t[:n_cameras*9].reshape(n_cameras,9)[camera_indices[:],:]
    torch_lands = x0_t_land[point_indices_[:],:] #x0_t[n_cameras*9:].reshape(n_points,3)[point_indices[:],:]
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

    if verbose:
        end = time.time()
        print("All torch grads take ", end - start, "s")
        start = time.time()

    J_pose = buildMatrixNew(cam_grad_x, cam_grad_y, camera_indices_, sz=9)
    if verbose:
        end = time.time()
        print(" build Matrix & residuum took ", end-start, "s")
        start = time.time()
    J_land = buildMatrixNew(land_grad_x, land_grad_y, point_indices_, sz=3)

    fx0 = buildResiduumNew(resX.detach(), resY.detach())

    if verbose:
        print(" build Matrix & residuum took ", end-start, "s")
        end = time.time()

    return (J_pose, J_land, fx0)

def buildMatrixNew(dx, dy, v_indices, sz=9) :
    data = []
    indptr = []
    indices = []

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
    if False:
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
def blockInverse(M, bs):
    Mi = M.copy()
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            mat = Mi.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs)
            # print(i, " ", mat)
            imat = inv_dense(mat) # inv or pinv?
            Mi.data[bs2 * i : bs2 * i + bs2] = imat.flatten()
    else:
        Mi = M.copy()
        for i in range(int(M.data.shape[0])):
            Mi.data[i : i + 1] = 1.0 / Mi.data[i : i + 1]
    return Mi

def blockEigenvalue(M, bs):
    Ei = np.zeros(M.shape[0])
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            mat = M.data[bs2 * i : bs2 * i + bs2].reshape(bs, bs)
            # print(i, " ", mat)
            evs = eigvalsh(mat)  # inv or pinv?
            Ei[bs*i:bs*i+bs] = evs[bs-1]
        Ei = diag_sparse(Ei)
    else:
        Ei = M.copy()

    # todo: eval super safe method since symmetric FAIL again.
    # idea was that non isotropic structure might hurting bfgs. but as bad as without
    #maxEv = np.max(Ei.data)
    #Ei.data[:] = 0.1 * maxEv # 0.1 worked
    #Ei.data[:] *= 0.125 #0.25 promising # 0.1 already too low. could try line-search idea
    return Ei

def copy_selected_blocks(M, block_selection_, bs):
    Mi = M.copy()
    if bs > 1:
        bs2 = bs * bs
        for i in range(int(M.data.shape[0] / bs2)):
            if block_selection_[i] == True:
                Mi.data[bs2 * i : bs2 * i + bs2] = 1e-12
    else:
        Mi = M.copy()
        for i in range(int(M.data.shape[0])):
            if block_selection_[i] == True:
                Mi.data[i : i + 1] = 1e-12
    return Mi

def stop_criterion(delta, delta_i, i):
    eps = 1e-2 #1e-2 used in paper, tune. might allow smaller as faster?
    return (i+1) * delta_i / delta < eps

def solvePowerIts(Ul, W, Vli, bS, m_):
    # costk = np.sum( bS**2 )
    # print("start gd cost ", costk)

    Uli = blockInverse(Ul, 9)
    xk = Uli * bS
    g = xk

    for _ in range(m_):
        # here uli^1/2 * M uli^1/2 * 'uli^1/2 * g' could be a symmetric split.
        # to the power of k uli^1/2 * uli^1/2 = uli
        g = Uli * (W * (Vli * (W.transpose() * g)))
        xk = xk + g
        if False:
            # eq is Ul [I - Uli * W * Vli * W.transpose()] x = b
            costk = np.sum(((Ul - W * Vli * W.transpose()) * xk - bS) ** 2)
            print(it, " gd cost ", costk)

        if stop_criterion(np.linalg.norm(xk, 2), np.linalg.norm(g, 2), it):
           return xk

    return xk

# more smart and new. After partitioning by camera non overlapping, we add
# residuals and cameras to complete (certain) landamrks until each lm is covered in some part.
#  
def cluster_by_camera(
    camera_indices_, points_3d_, points_2d_, point_indices_, kClusters_, startL_
):
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
    for i in range(kClusters-1):
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
    for ci in range(kClusters):
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
    num_res_per_c_ = np.zeros(kClusters)
    for ci in range(kClusters):
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
    for ci in range(kClusters):
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

    res_toadd_to_c_ = [[] for i in range(kClusters)]
    for i in point_indices_to_complete:
        #print(i, " complete ")
        cost = np.zeros(kClusters)
        for ci in range(kClusters):
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

    
    for ci in range(kClusters):
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
def process_clusters(num_lands, num_res, kClusters, point_indices_in_cluster_, point_indices_, res_indices_in_cluster_):
    landmark_occurrences = np.zeros(num_lands)
    for ci in range(kClusters):
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
    for ci in range(kClusters):
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
    num_res_per_c = np.zeros(kClusters)

    for ci in range(kClusters):
        _, _, counts = np.unique(point_indices_in_cluster_[ci], return_inverse=True, return_counts=True)
        counts = np.hstack([counts, np.zeros(res_per_lm.shape[0] - counts.shape[0])])
        missing_res_per_lm_c.append(res_per_lm - counts)
        num_res_per_c[ci] = np.sum(counts)

    res_of_lm = {value: np.where(point_indices_ == value) for value in np.unique(point_indices_)}

    res_of_lm_notin_c = []
    res_of_lm_in_c = []

    for ci in range(kClusters):
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

    res_toadd_to_c = [[] for _ in range(kClusters)]
    covered_landmark_indices_c = [[] for _ in range(kClusters)]

    # also consider cam linked by residuum already present in cluster or not
    for i in point_indices_to_complete:
        cost = np.zeros(kClusters)
        for ci in range(kClusters):
            cost[ci] += len(res_of_lm_notin_c[ci][i][0]) * num_res
            cost[ci] += num_res_per_c[ci]

        ci = np.argmin(cost)
        res_toadd_to_c[ci].append(res_of_lm_notin_c[ci][i][0])
        num_res_per_c[ci] += len(res_of_lm_notin_c[ci][i][0])
        covered_landmark_indices_c[ci].append(i)

    return res_toadd_to_c, point_indices_already_covered, covered_landmark_indices_c, num_res_per_c

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
    #lib.process_clusters_test.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    #lib.process_clusters_test(c_num_lands, c_num_res, c_kClusters, c_res_indices_in_cluster_flat_cpp)
    #lib.delete_vector(c_res_indices_in_cluster_flat_cpp)

    lib.process_clusters.restype = None #[ctypes.c_void_p, ctypes.c_void_p]
    lib.process_clusters.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p, # out:
                                    ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_void_p]

    lib.cluster_covis.restype = None
    lib.cluster_covis.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p] # out]


def process_cluster_lib(num_lands_, num_res_, kClusters__, point_indices_in_cluster__, res_indices_in_cluster__, point_indices__):
    init_lib()
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

def cluster_covis_lib(kClusters, camera_indices__, point_indices__):
    c_kClusters_ = ctypes.c_int(kClusters)

    camera_indices_list = camera_indices__.tolist()
    point_indices_list = point_indices__.tolist()

    c_point_indices_ptr = (ctypes.c_int * len(point_indices_list))(*point_indices_list)
    c_point_indices_cpp = lib.new_vector_by_copy(c_point_indices_ptr, len(c_point_indices_ptr))
    c_cam_indices_ptr = (ctypes.c_int * len(camera_indices_list))(*camera_indices_list)
    c_cam_indices_cpp = lib.new_vector_by_copy(c_cam_indices_ptr, len(c_cam_indices_ptr))

    #c_point_indices_ptr = (ctypes.c_int * len(point_indices__))(*point_indices__)
    #c_point_indices_cpp = lib.new_vector_by_copy(c_point_indices_ptr, len(c_point_indices_ptr))

    res_to_cluster_c_out = lib.new_vector()
    res_to_cluster_c_sizes = lib.new_vector_of_size(kClusters)

    lib.cluster_covis(c_kClusters_, c_cam_indices_cpp, c_point_indices_cpp, res_to_cluster_c_out, res_to_cluster_c_sizes)

    res_indices_in_cluster__ = fillPythonVec(res_to_cluster_c_out, res_to_cluster_c_sizes, kClusters)
    return res_indices_in_cluster__
    # copy data, free c++ mem

def cluster_by_camera_gpt(
    camera_indices_, points_3d_, points_2d_, point_indices_, kClusters_, startL_, init_cam_id=0, init_lm_id=0, seed=0
):
    np.random.seed(seed)
    # sort by res-indices by camera indices
    res_sorted = np.argsort(camera_indices_)
    num_res = camera_indices_.shape[0]
    num_cams = np.unique(camera_indices_).shape[0]
    num_lands = points_3d_.shape[0]
    print("number of residuum: ", num_res)

    baseline_clustering = True
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
        res_indices_in_cluster_ = cluster_covis_lib(kClusters, camera_indices, point_indices)
        camera_indices_in_cluster_ = []
        point_indices_in_cluster_ = []
        points_2d_in_cluster_ = []
        for c in range(kClusters_):
            res_indices_in_c_ = np.sort(res_indices_in_cluster_[c])
            points_2d_in_cluster_.append(points_2d_[res_indices_in_c_])
            camera_indices_in_cluster_.append(camera_indices_[res_indices_in_c_])
            point_indices_in_cluster_.append(point_indices_[res_indices_in_c_])
            #res_indices_in_cluster_.append(res_indices_in_cluster.copy())


    print(point_indices_in_cluster_.shape)
    print(point_indices_.shape)
    print(res_indices_in_cluster_.shape)

    res_toadd_to_c_, point_indices_already_covered_, covered_landmark_indices_c_, num_res_per_c_ = \
        process_cluster_lib(num_lands, num_res, kClusters, point_indices_in_cluster_, res_indices_in_cluster_, point_indices)

    # (res_toadd_to_c_, point_indices_already_covered_, covered_landmark_indices_c_, num_res_per_c) = \
    #     process_clusters(num_lands, num_res, kClusters, \
    #         point_indices_in_cluster_, point_indices_, res_indices_in_cluster_)
    
    for ci in range(kClusters):
        print(ci, " adding " , np.concatenate(res_toadd_to_c_[ci]).shape, " residuals to ", \
            point_indices_in_cluster_[ci].shape, " original residuals")
    
    additional_point_indices_in_cluster_ = [0 for _ in range(kClusters)] # variables to add, just unique (additional cameras -> not needed)
    additional_camera_indices_in_cluster_ = [0 for _ in range(kClusters)] # index into var per res
    additional_points_2d_in_cluster_ = [0 for _ in range(kClusters)] # essentially rhs for res
    # point_indices_already_covered: landmarks to be updated, present in main res only not in additional res (since complete)
    # covered_landmark_indices_c_: landmarks to be updated in additional!!! set of res since completely contained in cluster

    for ci in range(kClusters):
        #print("camera_indices_ ", camera_indices_.shape)
        con_res = np.concatenate(res_toadd_to_c_[ci])
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
        additional_point_indices_in_cluster_, additional_camera_indices_in_cluster_, additional_points_2d_in_cluster_, point_indices_already_covered_, covered_landmark_indices_c_,
        L_in_cluster_,
    )


# Afterwards landmarks only present in 1 cluster should equal points_3d_in_cluster_ afterwards.
# For those input s = old_lm. out put is 2 new - old.
# s is updated as s+ = s + 2new - old - new = s + new - old = new: ok.
# so should work without indexing.
#
# idea from bundle return:
# just to check, problem last uk is not present.
# or 2u+ - s - u to get delta. then add to v. or use real uk instead.
#
def average_landmarks_new(
    point_indices_in_cluster_, points_3d_in_cluster_, landmark_s_in_cluster_, L_in_cluster_, VL_in_cluster_, landmark_v_, delta_l_in_cluster
):
    num_points = points_3d_in_cluster_[0].shape[0]
    sum_Ds_2u = np.zeros(num_points * 3)
    sum_constant_term = 0
    for i in range(len(L_in_cluster_)):
        # Lc = L_in_cluster_[i]
        point_indices_ = np.unique(point_indices_in_cluster_[i])
        # mean_points[point_indices_,:] = mean_points[point_indices_,:] + points_3d_in_cluster_[i][point_indices_,:] * Lc
        # num_clusters[point_indices_] = num_clusters[point_indices_] + Lc
        # fill Vl with 0s 3x3 blocks? somehow ..
        # sparse matrix is 0,3,6, ... data (0s of Vl data), 012,012,012,345,345,345, etc ?
        #
        # data can remain, indptr can remain, indices must be adjusted / new
        # point_indices_
        indices = np.repeat(
            np.array([3 * point_indices_ + j for j in range(3)]).transpose(), 3, axis=0
        ).flatten()
        # indices.append(np.array([3 * point_indices_ + j for j in range(3)]).transpose().flatten())
        # indptr is to be set to have empty lines by 0 3 3 -> no entries in row 3. 0:0-3, row 1:3-3

        indptr = [np.array([0])]
        j = 0
        for q in range(num_points):
            # print(q, " ", j, " ", point_indices_.shape[0], " ", np.array([9*j+3, 9*j+6, 9*j+9]) )
            if j < point_indices_.shape[0] and point_indices_[j] == q:
                indptr.append(np.array([9 * j + 3, 9 * j + 6, 9 * j + 9]).flatten())
                j = j + 1
            else:
                indptr.append(np.array([9 * j, 9 * j, 9 * j]).flatten())
        # print(indptr)
        indptr = np.concatenate(indptr)

        # print(indices)
        # print(VL_in_cluster_[i].indptr)
        # print(indptr)

        # print(indices.shape,  " ", VL_in_cluster_[i].data.shape, " ", VL_in_cluster_[i].indptr.shape, " ", indptr.shape, " ", 3*num_points, " ", 3 * np.max(point_indices_) )
        V_land = csr_matrix(
            (VL_in_cluster_[i].data, indices, indptr),
            shape=(3 * num_points, 3 * num_points),
        )
        # print(mean_points.shape, " " , V_land.shape, points_3d_in_cluster_[i].shape)
        # print cost after/before.
        # cost old v is where ? (v-2u+s)^T V_land (v-2u+s) = v^T V_land v + 2 v^T V_land (-2u+s) + (2u-s)^T V_land (2u-s)
        # derivative 2 V_land v + 2 V_land (-2u+s) = 0 <-> sum (V_land) v = sum (V_land (2u-s))

        # print(i, "averaging 3d ", points_3d_in_cluster_[i][globalSingleLandmarksB_in_c[i], :]) # indeed 1 changed rest is constant
        # print(i, "averaging vl ", V_land.data.reshape(-1,9)[globalSingleLandmarksA_in_c[i],:])  # indeed diagonal

        # TODO change 3, claim  2u+-s = 2 * (s+u)/2 - s  -  2 * (vli/2 .. ), so subtract u to get delta only
        u2_s = (2 * points_3d_in_cluster_[i].flatten() - landmark_s_in_cluster_[i].flatten())
        #u2_s = (2 * points_3d_in_cluster_[i].flatten() - landmark_s_in_cluster_[i].flatten()) - (points_3d_in_cluster_[i].flatten() - delta_l_in_cluster[i].flatten())

        sum_Ds_2u += V_land * u2_s # has 0's for those not present

        # print(i, "averaging u2_s ", u2_s.reshape(-1,3)[globalSingleLandmarksB_in_c[i], :]) # indeed 1 changed rest is constant

        sum_constant_term += points_3d_in_cluster_[i].flatten().dot(V_land * (points_3d_in_cluster_[i].flatten() + u2_s - landmark_s_in_cluster_[i].flatten()))
        if i == 0:
            Vl_all = V_land
        else:
            Vl_all += V_land
    Vli_all = blockInverse(Vl_all, 3)
    # TODO change 1
    landmark_v_out = Vli_all * sum_Ds_2u
    #landmark_v_out = landmark_v_.flatten() + Vli_all * sum_Ds_2u
    verbose = False
    if verbose:
        cost_input  = 0.5 * (landmark_v_.flatten().dot(Vl_all * landmark_v_.flatten() - 2 * sum_Ds_2u) + sum_constant_term)
        cost_output = 0.5 * (landmark_v_out.dot(       Vl_all * landmark_v_out        - 2 * sum_Ds_2u) + sum_constant_term)

        #cost_simpler_out = landmark_v_out.dot(       Vl_all * landmark_v_out)        * 0.5 - landmark_v_out.dot(       sum_Ds_2u)
        #cost_simpler_in =  landmark_v_.flatten().dot(Vl_all * landmark_v_.flatten()) * 0.5 - landmark_v_.flatten().dot(sum_Ds_2u)
        print("========== update v: ", round(cost_input), " -> ", round(cost_output), " gain: ", round(cost_input - cost_output) )
        #print("======================== update v: ", round(cost_simpler_in), " -> ", round(cost_simpler_out), " gain: ", round(cost_simpler_in - cost_simpler_out) )

    return landmark_v_out.reshape(num_points, 3)
    # Then use this for fixing landmarks / updating. enven use VLi / Vl instead? argmin_x sum_y=lm_in_cluster (x-y) VL (x-y) of last Vl.
    # ==> x^t sum Vl x - 2 x sum (Vl y) + const =>  x = (sum Vl)^-1 [sum (Vl*y)].
    # above should lead to better? solutions at least. At next local updates we work with Vl?
    # 1. return last Vl. recall Vl * delta = nabla l is solved. or use L * diag Vl and return it (cheaper) or L * Vl ?
    # return 3x3 matrix per lm.
    # f(x) < f(y) + <nabla fy , x-y> + (x-y)^ Vl (x-y). Vl is making this strongly convex by design. s.t. this descent lemma holds. Even by design.

# cost is fuk * fuk + rho_k/2 |u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k>
#                     rho_k/2 <u_k - v_k, u_k - v_k> - rho_k <s_k - u_k, u_k - v_k>
#                     rho_k/2 <u_k - v_k - 2s_k + 2u_k, u_k - v_k>
#                     rho_k/2 <3u_k - v_k - 2s_k, u_k - v_k>
#                     rho_k/2 {v^tv - vT[2uk-sk] + uk^T[3uk-2sk]}
def cost_DRE(
    point_indices_in_cluster_, points_3d_in_cluster_, landmark_s_in_cluster_, L_in_cluster_, VL_in_cluster_, landmark_v_
):
    num_points = points_3d_in_cluster_[0].shape[0]
    sum_Ds_2u = np.zeros(num_points * 3)
    sum_constant_term = 0
    sum_u_s =0
    sum_u_v = 0
    sum_u_v_ = 0
    sum_2u_s_v = 0
    for i in range(len(L_in_cluster_)):
        point_indices_ = np.unique(point_indices_in_cluster_[i])
        indices = np.repeat(
            np.array([3 * point_indices_ + j for j in range(3)]).transpose(), 3, axis=0
        ).flatten()

        indptr = [np.array([0])]
        j = 0
        for q in range(num_points):
            if j < point_indices_.shape[0] and point_indices_[j] == q:
                indptr.append(np.array([9 * j + 3, 9 * j + 6, 9 * j + 9]).flatten())
                j = j + 1
            else:
                indptr.append(np.array([9 * j, 9 * j, 9 * j]).flatten())
        indptr = np.concatenate(indptr)

        V_land = csr_matrix(
            (VL_in_cluster_[i].data, indices, indptr),
            shape=(3 * num_points, 3 * num_points),
        )
        u2_s = (2 * points_3d_in_cluster_[i].flatten() - landmark_s_in_cluster_[i].flatten())
        sum_Ds_2u += V_land * u2_s # has 0's for those not present
        sum_constant_term += points_3d_in_cluster_[i].flatten().dot(V_land * (points_3d_in_cluster_[i].flatten() + u2_s - landmark_s_in_cluster_[i].flatten()))

        u_s = points_3d_in_cluster_[i].flatten() - landmark_s_in_cluster_[i].flatten()
        u_v = points_3d_in_cluster_[i].flatten() - landmark_v_.flatten()
        v_u2_s = u2_s - landmark_v_.flatten()
        sum_u_s += u_s.dot(V_land * u_s)
        sum_u_v += u_v.dot(V_land * u_v)
        sum_u_v_ =+ u_v.dot(u_v)
        sum_2u_s_v += v_u2_s.dot(V_land * v_u2_s)
        if i == 0:
            Vl_all = V_land
        else:
            Vl_all += V_land

    # TODO: I use a different Vl to compute the cost here than in the update of prox u.
    #       Since I want to work with a new Vl already. Problem.
    # i want |u-s|_D |u-v|_D, also |v-2u-s|_D
    cost_input  = 0.5 * (landmark_v_.flatten().dot(Vl_all * landmark_v_.flatten() - 2 * sum_Ds_2u) + sum_constant_term)
    print("---- |u-s|^2_D ", round(sum_u_s), "|u-v|^2_D ", round(sum_u_v), "|2u-s-v|^2_D ", round(sum_2u_s_v), "|u-v|^2 ", round(sum_u_v_))
    return cost_input


def average_landmarks(point_indices_in_cluster_, points_3d_in_cluster_, L_in_cluster_):
    mean_points = np.zeros(points_3d_in_cluster_[0].shape)
    num_clusters = np.zeros(points_3d_in_cluster_[0].shape[0])
    kClusters_ = len(L_in_cluster_)
    for i in range(kClusters_):
        Lc = L_in_cluster_[i]
        point_indices_ = np.unique(point_indices_in_cluster_[i])
        mean_points[point_indices_, :] = (
            mean_points[point_indices_, :]
            + points_3d_in_cluster_[i][point_indices_, :] * Lc
        )
        num_clusters[point_indices_] = num_clusters[point_indices_] + Lc
    mean_points[:, 0] = mean_points[:, 0] / num_clusters
    mean_points[:, 1] = mean_points[:, 1] / num_clusters
    mean_points[:, 2] = mean_points[:, 2] / num_clusters
    return mean_points

# TODO: shorten
def primal_cost(
    x0_p_,
    camera_indices_in_cluster_,
    point_indices_in_cluster_,
    points_2d_in_cluster_,
    points_3d_in_cluster_,
):
    cameras_indices_in_c_ = np.unique(camera_indices_in_cluster_)
    cameras_in_c = x0_p_[cameras_indices_in_c_]
    torch_points_2d_in_c = from_numpy(points_2d_in_cluster_)
    torch_points_2d_in_c.requires_grad_(False)

    unique_points_in_c_ = np.unique(point_indices_in_cluster_)
    inverse_point_indices = -np.ones(np.max(unique_points_in_c_) + 1)  # all -1
    for i in range(unique_points_in_c_.shape[0]):
        inverse_point_indices[unique_points_in_c_[i]] = i

    point_indices_in_c = point_indices_in_cluster_.copy()
    for i in range(point_indices_in_cluster_.shape[0]):
        point_indices_in_c[i] = inverse_point_indices[point_indices_in_c[i]]

    min_cam_index_in_c = np.min(camera_indices_in_cluster_)
    points_3d_in_c = points_3d_in_cluster_[unique_points_in_c_]

    #camera_indices_ = camera_indices_in_cluster_ - min_cam_index_in_c # outdated
    camera_indices_ = np.zeros(camera_indices_in_cluster_.shape[0], dtype=int)
    for i in range(cameras_indices_in_c_.shape[0]):
        camera_indices_[camera_indices_in_cluster_ == cameras_indices_in_c_[i]] = i

    point_indices_ = point_indices_in_c
    torch_points_2d = torch_points_2d_in_c
    cameras_in = cameras_in_c
    points_3d_in = points_3d_in_c

    x0_l_ = points_3d_in.flatten()
    # holds all cameras, only use fraction, camera_indices_ can be adjusted - min index
    x0_p_ = cameras_in.flatten()
    x0 = np.hstack((x0_p_, x0_l_))
    x0_t = from_numpy(x0)
    # torch_points_2d = from_numpy(points_2d)
    n_cameras_ = int(x0_p_.shape[0] / 9)
    n_points_ = int(x0_l_.shape[0] / 3)
    x0_t_cam = x0_t[: n_cameras_ * 9].reshape(n_cameras_, 9) # not needed?
    x0_t_land = x0_t[n_cameras_ * 9 :].reshape(n_points_, 3)
    funx0_st1 = lambda X0, X1, X2: \
        torchSingleResiduum(X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2))
    fx1 = funx0_st1(
        x0_t_cam[camera_indices_[:]],
        x0_t_land[point_indices_[:]],
        torch_points_2d)
    costEnd = np.sum(fx1.numpy() ** 2)
    return costEnd


# return cost_, x0_p_c_, x0_l_c_, Lnew_c_, Vl_c_ =
def local_bundle_adjust(
    local_camera_indices_in_, # LOCAL 1st res
    point_indices_in_,   # LOCAL 1st res
    torch_points_2d_in_, # 1st res
    cameras_in_,         # LOCAL 1st res
    points_3d_in_,       # LOCAL all res
    additional_local_camera_indices_, # local cam  ids 2nd res
    additional_point_indices_in_,              # local land ids 2nd res
    additional_torch_points_2d_in_,            # 2nd res
    torch_additional_cameras_in_c,             # LOCAL 2nd part
    # MY IDEA WAS TO CLUSTER BY POINTS without gap and complete cameras, so that all points get updated
    # that does not even exist, right .. now need to set delta_l =0 for those fixed
    covered_landmark_indices_, # those will be returned AND UPDATED FUCK, picking a subset of unique(point_indices_in_) (not!? points_3d_in_c) to update current estimate
    # very dangerous to confuse. it is not those global col indices but indices of present col indices.
    # So if jacobian has GLOBAL landmark indices in its column we must x2 index those relevant ones out.
    # BUT point_indices_in_ are 'local indices' -- what does this mean? 
    # if these indices are local how could i ever argh
    additional_covered_landmark_indices_, # those are returned and updated, but present in additional res, so picking a subset of unique(additional_point_indices_in_)
    # point is since additional_point_indices_in_ are only those to complete, it is all of them.
    Vl_in_cluster_,
    L_in_cluster_,
    successfull_its_=1,
):
    # define x0_t, x0_p, x0_l, L # todo: missing Lb: inner L for bundle, Lc: to fix duplicates
    L = L_in_cluster_
    updateJacobian = True
    x0_l_ = from_numpy(points_3d_in_) # only update covered landmarks!
    x0_p_ = from_numpy(cameras_in_)
    x0_p_a = torch_additional_cameras_in_c
    #x0_t = np.hstack((x0_p_, x0_l_))
    #x0_a = torch.hstack(x0_c_a, x0_l_)
    #x0_t = from_numpy(x0)

    # torch_points_2d = from_numpy(points_2d)
    n_cameras_ = int(x0_p_.shape[0])
    n_points_ = int(x0_l_.shape[0])
    powerits = 20 # kind of any value works here? > =5?

    it_ = 0
    funx0_st1 = lambda X0, X1, X2: \
        torchSingleResiduum(X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2))

    verbose = False
    unused = True

    while it_ < successfull_its_:

        if updateJacobian:  # not needed if rejected
            #
            x0_t_cam = x0_p_
            x0_t_land = x0_l_
            # J_land_o has more columns than landmarks exist -> 3* num points ok
            if verbose:
                print("points_3d_in_ shape", points_3d_in_.shape)
                print("min max column indices = landamrk index", np.min(point_indices_in_), " ", np.max(point_indices_in_), " ", np.unique(point_indices_in_).shape)
                print("min max column indices = landamrk index", 3*np.min(point_indices_in_), " ", 3*np.max(point_indices_in_)+2)
            J_pose, J_land_o, fx0_o = ComputeDerivativeMatricesNew(
                x0_t_cam, x0_t_land, local_camera_indices_in_, point_indices_in_, torch_points_2d_in_
            )
            if verbose:
                print("min max J column indices = landamrk index", np.min(J_land_o.indices), " ", np.max(J_land_o.indices))

            # J_land_o is sparse matrix
            # covered_landmark_indices_ holds indices to keep = columns to keep
            # option 1 set data to 0 for relevant columns, option 2 remove relevant columns (not last one!)
            # TODO: does not work like this
            # data is holding nx3 rows per variable. if we option 1 remove cols, we also need to remove more
            if False:
                print("J_land_o " , J_land_o.shape)
                covered_indices_3d = np.vstack((3*covered_landmark_indices_, 3*covered_landmark_indices_ +1, 3*covered_landmark_indices_+2)).transpose().flatten()
                covered_indices = np.vstack((9*covered_landmark_indices_, 9*covered_landmark_indices_ +1, 9*covered_landmark_indices_+2, \
                                            9*covered_landmark_indices_+3, 9*covered_landmark_indices_ +4, 9*covered_landmark_indices_+5, \
                                            9*covered_landmark_indices_+6, 9*covered_landmark_indices_ +7, 9*covered_landmark_indices_+8)).transpose().flatten()
                uncovered_indices = np.setdiff1d( np.arange(9 * n_points_), covered_indices)
                uncovered_indices_3d = np.setdiff1d( np.arange(3 * n_points_), covered_indices_3d)
                # keep_data = J_land_o.data.copy()
                # keep_data[covered_indices] = 0
                # J_land_o.data -= keep_data
                J_land_o.data[uncovered_indices] = 0
                print("J_land_o.data[covered_indices] ", J_land_o.data[covered_indices])
                print("J_land_o.data[uncovered_indices] ", J_land_o.data[uncovered_indices])
                #print(J_land_o)
                print("covered_indices ", covered_indices)
                print("uncovered_indices ", uncovered_indices)
            else:
                covered_indices_3d = np.vstack((3*covered_landmark_indices_, 3*covered_landmark_indices_ +1, 3*covered_landmark_indices_+2)).transpose().flatten()
                uncovered_indices_3d = np.setdiff1d( np.arange(3 * n_points_), covered_indices_3d)

            #orig_shape = J_land_o.shape
            #print("J_land_o " , orig_shape)
            # keep only columns that are changed
            # 1st concat with a then delete cols
            # J_land_o = J_land_o[:,covered_indices_3d] # problem deletes all columns and shrinks matrix in place, not leaving 0 columns where they should be
            #print("covered J_land_o " , J_land_o.shape)
            #J_land_o.resize(orig_shape)
            if verbose:
                print("covered J_land_o " , J_land_o.shape)
                print("covered ", covered_landmark_indices_) # a subset of covered

                print("points_3d_in_ shape", points_3d_in_.shape)
                print("min max column indices additional = landamrk index", np.min(additional_point_indices_in_), " ", np.max(additional_point_indices_in_), " ", np.unique(additional_point_indices_in_).shape)
                print("min max column indices additional = landamrk index", 3*np.min(additional_point_indices_in_), " ", 3*np.max(additional_point_indices_in_)+2)

            _, J_land_a, fx0_a = ComputeDerivativeMatricesNew(
                x0_p_a, x0_t_land, additional_local_camera_indices_, additional_point_indices_in_, additional_torch_points_2d_in_
            )
            if verbose:
                print(J_land_a.data)
                print("min max Ja column indices = landamrk index", np.min(J_land_a.indices), " ", np.max(J_land_a.indices))
                print("unique in a and u ", np.unique(additional_point_indices_in_).shape, np.unique(point_indices_in_).shape)
                print("J_land_a " ,J_land_a.shape)

            J_land_a.resize((J_land_a.shape[0], J_land_o.shape[1])) # same number of columns
            if verbose:
                print("resized J_land_a " ,J_land_a.shape)
            J_land = vstack((J_land_o, J_land_a))
            if verbose:
                print("stacked J_land " ,J_land.shape)

            # Fixing landmarks not covered completely
            J_land = J_land[:,covered_indices_3d] # problem deletes all columns and shrinks matrix in place, not leaving 0 columns where they should be
            if verbose:
                print("stacked J_land cols removed" ,J_land.shape)

            JtJ = J_pose.transpose() * J_pose
            JtJDiag = diag_sparse(JtJ.diagonal())
            #JtJDiag = diag_sparse(np.fmax(JtJ.diagonal(), 1e-3)) # sensible not clear maybe lower.
            JltJl = J_land_o.transpose() * J_land_o + J_land_a.transpose() * J_land_a
            JltJl = J_land.transpose() * J_land
            if unused:
                ones_at_uncovered_indices = np.ones(3*n_points_)
                ones_at_uncovered_indices[covered_indices_3d] = 0
                blockEigenvalueJltJl = blockEigenvalue(JltJl, 3) #+ diag_sparse(ones_at_uncovered_indices)
                
                # todo 0.5 appars to be ok still & faster. lower leads to hickups and slow down
                stepSize = 1. * (1e-3 * blockEigenvalueJltJl + JltJl.copy()) # this already suffices
                #stepSize = 1. * (1e-0 * diag_sparse(np.ones(n_points_*3)) + 1.0 * JltJl.copy()) # not so good
                JltJlDiag = stepSize.copy() # max 1, 1/L, line-search dre fails -> increase
            
            # JltJlDiag = JltJl.copy()
            # JtJDiag = JtJ.copy() # TODO: new about same
            # JtJDiag = diag_sparse(JtJ.diagonal()) # leads to increase of L slowing down everything
            #JltJlDiag = diag_sparse(JltJl.diagonal()) # absolutely not

            J_eps = 1e-9
            JltJlDiag = JltJl + J_eps * diag_sparse(np.ones(JltJl.shape[0]))
            JtJDiag   = JtJ + J_eps * diag_sparse(np.ones(JtJ.shape[0]))

            if verbose:
                print(" min/max JltJl.diagonal() ", np.min(JltJl.diagonal()), " adjusted ", np.min(JltJlDiag.diagonal()), " ", np.max(JltJlDiag.diagonal()))
                print("JltJlDiag.shape ", JltJlDiag.shape, JltJlDiag.shape[0]/3)

            W = J_pose.transpose() * J_land_o
            #W = J_pose.transpose() * J_land # adding 0s to J_pose can be avoided here (missing residuals since f is stacked)

            # Fixing landmarks not covered completely
            W = W[:,covered_indices_3d]
            bp = J_pose.transpose() * fx0_o
            # bl = J_land_o.transpose() * fx0_o + J_land_a.transpose() * fx0_a
            if verbose:
                print("fx shapes ", fx0_o.shape, " ", fx0_a.shape)
                #print("fx stacked shapes ", np.hstack((fx0_o, fx0_a)))
            bl = J_land.transpose() * np.hstack((fx0_o, fx0_a))

            costStart = np.sum(fx0_o**2) + np.sum(fx0_a**2)

        Vl = JltJl + L * JltJlDiag
        Ul = JtJ + L * JtJDiag

        Vli = blockInverse(Vl, 3)
        bS = (bp - W * Vli * bl).flatten()

        delta_p = -solvePowerIts(Ul, W, Vli, bS, powerits)
        delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl)
        # delta_l = delta_l * 0 # test ok error in landmark update.
        #print("delta_l[covered_indices] ", delta_l[covered_indices_3d])
        #print("delta_l[uncovered_indices] ", delta_l[uncovered_indices_3d])
        #delta_l[uncovered_indices_3d] = 0

        penaltyL = L * delta_l.dot(JltJlDiag * delta_l)
        penaltyP = L * delta_p.dot(JtJDiag * delta_p)
 
        delta_l_full = np.zeros(3*n_points_)
        # Fixing landmarks not covered completely
        delta_l_full[covered_indices_3d] = delta_l
        #delta_l_full = delta_l

        fx0_o_new = fx0_o + (J_pose * delta_p + J_land_o * delta_l_full)
        fx0_a_new = fx0_a + J_land_a * delta_l_full
        if verbose:
            print("J_land_a * delta_l ", J_land_a * delta_l_full, " |delta_l| ", np.linalg.norm(delta_l), " |delta_p| ", np.linalg.norm(delta_p))
        costQuad = np.sum(fx0_o_new**2) + np.sum(fx0_a_new**2)
        #print(it_, "it. cost o/a   ", round(np.sum(fx0_o_new**2)), " / ", round(np.sum(fx0_a_new**2)))
        print(it_, "it. cost 0     ", round(costStart)," cost + penalty ", round(costStart), " === using L = ", L)
        print(it_, "it. cost 0/new ", round(costQuad), " cost + penalty ", round(costQuad + penaltyL + penaltyP),)

        # update and compute cost
        x0_p_ = x0_p_ + from_numpy(delta_p.reshape(n_cameras_, 9))
        x0_l_ = x0_l_ + from_numpy(delta_l_full.reshape(n_points_, 3))

        fx1_o = funx0_st1(
            x0_p_[local_camera_indices_in_,:],
            x0_l_[point_indices_in_,:],
            torch_points_2d_in_)

        fx1_a = funx0_st1(
            x0_p_a[additional_local_camera_indices_,:],
            x0_l_[additional_point_indices_in_,:],
            additional_torch_points_2d_in_)

        costEnd = np.sum(fx1_o.numpy() ** 2) + np.sum(fx1_a.numpy() ** 2)
        print(it_, "it. cost 1     ", round(costEnd), "      + penalty ", round(costEnd + penaltyL + penaltyP),)

        # descent lemms test again.
        nablaXp = L * JtJDiag * delta_p  # actual gradient
        nablaXl = L * JltJlDiag * delta_l  # actual gradient
        LfkDiagonal = \
            2 * (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l)) \
            / (delta_l.dot(nablaXl) + delta_p.dot(nablaXp))
        # descent lemma. 
        if costStart < costEnd + penaltyL + penaltyP or LfkDiagonal > 2:
            # revert -- or linesearch
            x0_p_ = x0_p_ - from_numpy(delta_p.reshape(n_cameras_, 9))
            x0_l_ = x0_l_ - from_numpy(delta_l_full.reshape(n_points_, 3))
            updateJacobian = False
        else:
            it_ = it_ + 1
            updateJacobian = True

        tr_check = (costStart - costEnd - penaltyL - penaltyP) / (costStart - costQuad - penaltyL - penaltyP)
        print(" ------- Lfk estimate ", LfkDiagonal, " -nabla^Tdelta=" , -bp.dot(delta_p) - bl.dot(delta_l),  " tr_check ", tr_check, " -------- ")
        # update TR -- not now
        if it_ <= successfull_its_:# and L > 1e-6: # lowering leads to, see below averaging affected, can trigger multiple increases
            #print( "A JltJlDiag-bun ", JltJlDiag.data.reshape(-1,9)[landmarks_only_in_cluster_,:])
            eta_1 = 0.8
            eta_2 = 0.25
            if LfkDiagonal > 2 or costStart < costEnd + penaltyL + penaltyP: #tr_check > eta_1: # and L > 0.1: # needs a limit else with my L * Vl, or 1/L in diag?
                L = L * 2
            if LfkDiagonal < -1: #tr_check < eta_2 or LfkDiagonal > 2: # tr check becomes descent lemma, might need > 1?
                L = L / 2
    print("LfkDiagonal ", LfkDiagonal, " L ", L)

    return costEnd, x0_p_.numpy(), x0_l_[covered_landmark_indices_,:].numpy(), L, L * JltJlDiag + 1e-12 * Vl


# TODO: to run multiple iterations in distributed version we need to use f(x) + L/2|x -  z|
# where z is the mean of the landmarks. likely L can be set via descent lemma or better as diag Vl
# there is the notion of whether to start from x or z.
# using landmarks_only_in_cluster_: issue is s + v-u is now wrong, but since unique lms only ok. when extrapolating set to 0. 
# 21  ======== DRE ======  27571  ========= gain  44 ==== f(v)=  27532  f(u)=  27571
# 30  ======== DRE ======  27286  ========= gain  25 ==== f(v)=  27268  f(u)=  27275
# 38  ======== DRE ======  27169  ========= gain  14 ==== f(v)=  27157  f(u)=  27159
# without not much, still some its.
# 21  ======== DRE ======  27795  ========= gain  58 ==== f(v)=  27752  f(u)=  27799
# 30  ======== DRE ======  27415  ========= gain  34 ==== f(v)=  27399  f(u)=  27405
# 38  ======== DRE ======  27260  ========= gain  20 ==== f(v)=  27249  f(u)=  27250
def bundle_adjust(
    camera_indices_,
    point_indices_,
    landmarks_only_in_cluster_,
    torch_points_2d,
    cameras_in,
    points_3d_in,
    landmark_s_, # taylor expand at point_3d_in -> prox on landmark_s_ - points_3d = lambda (multiplier)
    Vl_in_c_,
    Lc_,
    successfull_its_=1,
):
    # print("landmarks_only_in_cluster_  ", landmarks_only_in_cluster_, " ", np.sum(landmarks_only_in_cluster_), " vs ", np.sum(1 - landmarks_only_in_cluster_) )
    newForUnique = False
    # define x0_t, x0_p, x0_l, L # todo: missing Lb: inner L for bundle, Lc: to fix duplicates
    L = Lc_
    updateJacobian = True
    # holds all! landmarks, only use fraction likely no matter not present in cams anyway.
    x0_l_ = points_3d_in.flatten()
    s_l_ = landmark_s_.flatten()
    # holds all cameras, only use fraction, camera_indices_ can be adjusted - min index
    x0_p_ = cameras_in.flatten()
    x0 = np.hstack((x0_p_, x0_l_))
    x0_t = from_numpy(x0)
    # torch_points_2d = from_numpy(points_2d)
    n_cameras_ = int(x0_p_.shape[0] / 9)
    n_points_ = int(x0_l_.shape[0] / 3)
    powerits = 20 # kind of any value works here? > =5?

    it_ = 0
    funx0_st1 = lambda X0, X1, X2: \
        torchSingleResiduum(X0.view(-1, 9), X1.view(-1, 3), X2.view(-1, 2))

    #stepSize = diag_sparse(np.zeros(n_points_))
    # make diagonal again.
    if issparse(Vl_in_c_):
        stepSize = diag_sparse(Vl_in_c_.diagonal())

    steSizeTouched = False

    while it_ < successfull_its_:
        # to torch here ?

        if updateJacobian:  # not needed if rejected
            #
            x0_t_cam = x0_t[: n_cameras_ * 9].reshape(n_cameras_, 9) # not needed?
            x0_t_land = x0_t[n_cameras_ * 9 :].reshape(n_points_, 3)

            print("points_3d_in_ shape", points_3d_in.shape, " n_points_", n_points_)
            print("min max column indices = landamrk index", np.min(point_indices_), " ", np.max(point_indices_), " ", np.unique(point_indices_).shape)
            J_pose, J_land, fx0 = ComputeDerivativeMatricesNew(
                x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d
            )
            print("min max J column indices = landamrk index", np.min(J_land.indices), " ", np.max(J_land.indices))
            print("J_land " , J_land.shape)

            JtJ = J_pose.transpose() * J_pose
            JtJDiag = diag_sparse(JtJ.diagonal())
            #JtJDiag = diag_sparse(np.fmax(JtJ.diagonal(), 1e1)) # sensible not clear maybe lower.
            JltJl = J_land.transpose() * J_land
            blockEigenvalueJltJl = blockEigenvalue(JltJl, 3)
            # if not issparse(Vl_in_c_) and it_ < 1:
            #     stepSize = blockEigenvalueJltJl
            # else: # increase where needed -- this here is WAY too slow?
            #     stepSize.data = np.maximum(0.05 * stepSize.data, blockEigenvalueJltJl.data) # else diagSparse of it
            
            # todo 0.5 appars to be ok still & faster. lower leads to hickups and slow down
            stepSize = 1. * (1e-3 * blockEigenvalueJltJl + JltJl.copy()) # this already suffices
            #stepSize = 1. * (1e-0 * diag_sparse(np.ones(n_points_*3)) + 1.0 * JltJl.copy()) # not so good

            # problem blockinverse assume data is set in full 3x3
            # Problem: gets stuck. guess. a new, very large value for a coordinate hits a previously low one.
            # averaging leaves a gap here, if gap is too large we fail.
            # should still be caverable by (pre!)-computing deriv in advance. Then jump is limited by inc in L.
            #JltJlDiag = diag_sparse(np.fmax(JltJl.diagonal(), 1e1)) # startL defines this. 
            maxDiag = np.max(JltJl.diagonal())
            # here larger x, eg 5: 1ex leasd  to stallment, small to v,u divergence.
            #JltJlDiag = diag_sparse(np.fmax(JltJl.diagonal(), np.minimum(maxDiag, 1e5 * np.maximum(1., 1./L)))) # startL defines this. 
            # theoretic value is 2 * diag. [modula change to diag]. Here we use 1/2 diag. but No dependence on L.
            JltJlDiag = np.maximum(1, 0.5 / L) * diag_sparse(np.fmax(JltJl.diagonal(), 1e1)) # this should ensure (small L) condition.
            # improvement: only for constrained variables.
            # improvement: acceleration, maybe needs ALL variables to be effective.

            JltJlDiag = stepSize.copy() # max 1, 1/L, line-search dre fails -> increase

            if newForUnique:
                JltJlDiag = copy_selected_blocks(JltJlDiag, landmarks_only_in_cluster_, 3)
                JltJlDiag = JltJlDiag + L * 1e-3 * blockEigenvalueJltJl
                #JltJlDiag = JltJlDiag + L * diag_sparse(JltJl.diagonal()) # original
                #JltJlDiag = 1/L * JltJlDiag + diag_sparse(JltJl.diagonal()) #diag_sparse(np.fmax(JltJl.diagonal(), 1e0))

            # theory states roughly JltJlDiag * max(1,1/L) should be used. This does not converge u,v are coupled but no progress is made on primal.
            # should use descent lemma to define this 'L' not TR? pose L can be anything
            # Above to slow with 1/L.
            # maybe use Dl/L instead, Dl=1 init, check descent lemma and lower Dl if possible (increase if not)
            # but Dl is fulfilled with L. So would lower Dl just like, or? 
            # maybe better: 3x3 matrix sqrt(|M|_1 |M|inf) as diag. Yet this removes effect of 'L' getting small = large steps.
            # do i need to keep memory to ensure it remains >? or pre compute grad (and store)?
            print(" min/max JltJl.diagonal() ", np.min(JltJl.diagonal()), " ", maxDiag, " adjusted ", np.min(JltJlDiag.diagonal()), " ", np.max(JltJlDiag.diagonal()))
            print("JltJlDiag.shape ", JltJlDiag.shape, JltJlDiag.shape[0]/3)

            # based on unique_pointindices_ make diag_1L that is 1 at points 
            # only present within this part.

            # 0.2 was working
            JltJlDiag = 1/L * JltJlDiag # max 1, 1/L, line-search dre fails -> increase

            W = J_pose.transpose() * J_land
            bp = J_pose.transpose() * fx0
            bl = J_land.transpose() * fx0

            prox_rhs = x0_l_ - s_l_
            if newForUnique: # alternative turn off completely, use 2u-s -> return (u-s)/2 to average u+k = uk + delta uk
                landmarks_in_many_cluster_ = np.invert(landmarks_only_in_cluster_)
                diag_present = diag_sparse( np.repeat((np.ones(n_points_) * landmarks_in_many_cluster_).reshape(-1,1), 3).flatten() )
                prox_rhs = 1 * diag_present * prox_rhs

            costStart = np.sum(fx0**2)
            penaltyStartConst = prox_rhs.dot(JltJlDiag * prox_rhs)

        # start_ = time.time()
        Vl = JltJl + L * JltJlDiag
        Ul = JtJ + L * JtJDiag
        penaltyStart = L * penaltyStartConst
        # cost added is + L * (delta_v - s_l_ + x0_l_)^T  JltJlDiag * (delta_v - s_l_ + x0_l_)
        # + L * (delta_v)^T  JltJlDiag * (delta_v) + 2 L * (delta_v^T JltJlDiag * (x0_l_ - s_l_) + L * (s_l_ - x0_l_)^T  JltJlDiag * (s_l_ - x0_l_)
        # derivative
        # L * 2 * JltJlDiag * (delta_v) + 2 L * JltJlDiag * (x0_l_ - s_l_) = 0
        # added cost is, 2 L * (delta_v^T JltJlDiag * (x0_l_ - s_l_) + L * (s_l_ - x0_l_)^T  JltJlDiag * (s_l_ - x0_l_)

        Vli = blockInverse(Vl, 3)
        bl_s = bl + L * JltJlDiag * prox_rhs # TODO: + or -. '+', see above
        bS = (bp - W * Vli * bl_s).flatten()

        delta_p = -solvePowerIts(Ul, W, Vli, bS, powerits)
        delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl_s)
        penaltyL = L * (delta_l + prox_rhs).dot(JltJlDiag * (delta_l + prox_rhs))
        penaltyP = L * delta_p.dot(JtJDiag * delta_p)
        # end_ = time.time()
        # print("Lm step took ", end - start, "s")

        fx0_new = fx0 + (J_pose * delta_p + J_land * delta_l)
        costQuad = np.sum(fx0_new**2)
        print(it_, "it. cost 0     ", round(costStart)," cost + penalty ", round(costStart + penaltyStart), " === using L = ", L)
        print(it_, "it. cost 0/new ", round(costQuad), " cost + penalty ", round(costQuad + penaltyL + penaltyP),)

        # update and compute cost
        x0_p_ = x0_p_ + delta_p
        x0_l_ = x0_l_ + delta_l

        x0 = np.hstack((x0_p_, x0_l_))
        x0_t = from_numpy(x0)
        x0_t_cam = x0_t[: n_cameras_ * 9].reshape(n_cameras_, 9)
        x0_t_land = x0_t[n_cameras_ * 9 :].reshape(n_points_, 3)

        fx1 = funx0_st1(
            x0_t_cam[camera_indices_[:]],
            x0_t_land[point_indices_[:]],
            torch_points_2d)
        costEnd = np.sum(fx1.numpy() ** 2)
        print(it_, "it. cost 1     ", round(costEnd), "      + penalty ", round(costEnd + penaltyL + penaltyP),)

        # descent lemms test again.
        nablaXp = L * JtJDiag * delta_p  # actual gradient
        nablaXl = L * JltJlDiag * delta_l  # actual gradient
        LfkDiagonal = \
            2 * (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l)) \
            / (delta_l.dot(nablaXl) + delta_p.dot(nablaXp))
        if LfkDiagonal > 2: # violated -- should revert update.
            steSizeTouched = True
            print(" |||||||  Lfk estimate ", LfkDiagonal, " -nabla^Tdelta=" , -bp.dot(delta_p) - bl.dot(delta_l), " |||||||")
            #stepSize = stepSize * 2
            # other idea, initially we only add 1/2^k eg 0.125, times the needed value and inc if necessary, maybe do not add anything if not needed.

            blockEigenvalueJltJl.data *= 2 # appears slow but safe
            #stepSize.data = np.maximum(stepSize.data, blockEigenvalueJltJl.data) # else diagSparse of it
            stepSize = 1e-3 * blockEigenvalueJltJl + JltJl.copy()

            JltJlDiag = 1/L * stepSize.copy()
            penaltyStartConst = (prox_rhs - delta_l).dot(JltJlDiag * (prox_rhs - delta_l))
            #penaltyStartConst *= 2
        # if LfkDiagonal < -2 and steSizeTouched:
        #     LfkDiagonal = -2
        # if LfkDiagonal < -2 and not steSizeTouched: # can we increase convergence in those cases? Problme if this fluctuates of course. Maybe only do if above did not enter
        #     steSizeTouched = True
        #     print(" |||||||  Lfk estimate ", LfkDiagonal, " -nabla^Tdelta=" , -bp.dot(delta_p) - bl.dot(delta_l), " |||||||")
        #     stepSize.data = stepSize.data / 1.5
        #     JltJlDiag = 1/L * stepSize.copy()
        #     penaltyStartConst = (x0_l_ - delta_l - s_l_).dot(JltJlDiag * (x0_l_ - delta_l - s_l_))

        # version with penalty check for ADMM convergence / descent lemma. Problem: slower?
        if costStart + penaltyStart < costEnd + penaltyL or LfkDiagonal > 2: # or LfkDiagonal < -2:
            # revert -- or linesearch
            x0_p_ = x0_p_ - delta_p
            x0_l_ = x0_l_ - delta_l
            x0 = np.hstack((x0_p_, x0_l_))
            x0_t = from_numpy(x0)
            x0_t_cam = x0_t[: n_cameras_ * 9].reshape(n_cameras_, 9)
            x0_t_land = x0_t[n_cameras_ * 9 :].reshape(n_points_, 3)
            updateJacobian = False
            continue # avoids de/increasing L below.
        else:
            it_ = it_ + 1
            updateJacobian = True

        # Descent lemma print # descent lemma for landmarks only?
        # f(x) <= f(y) + <nabla f(y), x-y> + Lf/2 |x-y|^2
        # e.g. update from y to x, nabla f(y) = Jac * res(y) = b
        # x=y+delta, f(x) <= f(y) + <nabla f(y), x-y> + Lf/2 |x-y|^2
        #       <=>  f(x) - f(y) <= <nabla f(y), delta> + Lf/2 |delta|^2
        #       <=> [f(x) - f(y) - b^T delta] / delta^2 <= Lf/2, defines Lf
        # 2 * (costEnd - costStart - b^t (delta) / |delta|^2)
        # ok delta ~ - nabla -> b^Tdelta > 0. Even delta = - JtJ^-1 b.
        # Hence - b^T JtJ^-1 b. Since J^tJ>=0 it follows b^Tdelta < 0. Ok see below.
        # Note that nabla f = b and delta = - J^tJ ^-2 b -> b = JtJ delta
        # Hence f(x) <= f(y) + delta^T [l/2 D^tD + J^tJ] delta. for all x=delta+y.
        # Also J^tJ = K^tK + q * D^tD with D = diag(K) and q by TR.
        # So (ignore 1/2) (l-q) D^TD > J^TJ. Maybe set l = 2q, where q is our TR choice.
        # At least should not diverge
        if False:
            Lfk = \
                2 * (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l)) \
                / (delta_l.dot(delta_l) + delta_p.dot(delta_p))
            # Not the same we have diag in there: Lf/2 delta^t D delta.
            # so divide by this to receive update for Lf. This is given below.
            # Do i need the same for only landmarks/dupe lms?
            LfkX = \
                2 * (costStart - costEnd - nablaXp.dot(delta_p) - nablaXl.dot(delta_l)) \
                / (delta_l.dot(delta_l) + delta_p.dot(delta_p))
        # TODO descent lemma with L * JltJlDiag only for landmarks.
        # test joint.
        # We use diagonal matrix L * JtJDiag, JltJlDiag instead of scalar L:
        # we now get -- if accepted update < 2 always
        # nablaXp = L * JtJDiag * delta_p  # actual gradient
        # nablaXl = L * JltJlDiag * delta_l  # actual gradient
        # LfkDiagonal = \
        #     2 * (costEnd - costStart - bp.dot(delta_p) - bl.dot(delta_l)) \
        #     / (delta_l.dot( nablaXl) + delta_p.dot( nablaXp ))
        # this becomes x: new, y: old
        # descent lemma is now, y + delta = x, delta = x-y
        # f(x) <= f(y) + <nabla(f(y) x-y> + Lf/2 nablaf(x)^T (x-y)
        # f(x) <= f(y) + <nabla(f(y) + Lf/2 nablaf(x)^T, x-y> must hold.
        # find Lf s.t. this holds. 
        #print("Lfk estimate ", Lfk, " and ", LfkX, " or ", LfkDiagonal)
        # relation to check f(x) > f(y) + penaltyL to accept step is:
        # f(x) > f(y) + delta^t Diag delta = f(y) + <nabla f(x), x-y>
        # this implies descent lemma ALWAYS fulfilled.

        tr_check = (costStart + penaltyStart - costEnd - penaltyL) / (costStart + penaltyStart - costQuad - penaltyL)
        print(" ------- Lfk estimate ", LfkDiagonal, " -nabla^Tdelta=" , -bp.dot(delta_p) - bl.dot(delta_l),  " tr_check ", tr_check, " -------- ")
        # update TR -- 
        #if costStart + penaltyStart > costEnd + penaltyL
        #tr_check = (costStart - costEnd) / (costStart - costQuad)
        if True or it_ < successfull_its_ and L > 1e-6: # lowering leads to, see below averaging affected, can trigger multiple increases
            #print( "A JltJlDiag-bun ", JltJlDiag.data.reshape(-1,9)[landmarks_only_in_cluster_,:])
            eta_1 = 0.9
            eta_2 = 0.25
            if tr_check > eta_1: # and L > 0.1: # needs a limit else with my L * Vl, or 1/L in diag?
                L = L / 2
                JltJlDiag = 2 * JltJlDiag
            if tr_check < eta_2 or LfkDiagonal > 2: # tr check becomes descent lemma, might need > 1?
                L = L * 2
                JltJlDiag = 1/2 * JltJlDiag

        if LfkDiagonal > 2: # violated -- should revert update.
            print("=========================== SHOULD NOT ENTER ==========================")
            stepSize = stepSize * 2
            JltJlDiag = 1/L * stepSize.copy()

    x0_p_ = x0_p_.reshape(n_cameras_, 9)
    x0_l_ = x0_l_.reshape(n_points_, 3)

    # maybe only if likely that is differs? Still insufficient.
    # needs to estimate L as well. only if accept. no effect.
    getBetterStepSize = False # this is used as approx of f in update of v and thus s. maybe change there u-v should be small. 
    if getBetterStepSize: # needs to set L correctly
        J_pose, J_land, fx0 = ComputeDerivativeMatricesNew(
            x0_t_cam, x0_t_land, camera_indices_, point_indices_, torch_points_2d
        )
        JltJl = J_land.transpose() * J_land
        #JltJlDiag = diag_sparse(np.fmax(JltJl.diagonal(), 1e1)) # should be same as above
        #JltJlDiag = np.maximum(1, 0.5 / L) * diag_sparse(np.fmax(JltJl.diagonal(), 1e1)) # this should ensure (small L) condition.
        #stepSize = blockEigenvalue(JltJl, 3)
        stepSize.data = np.maximum(stepSize.data, blockEigenvalue(JltJl, 3).data) # else diagSparse of it
        JltJlDiag = 1/L * stepSize.copy() # max 1, 1/L, line-search dre fails -> increase

    # in the averaging step we do 
    # min_v 1/2 sum_k (v - (2uk-sk)) Vlk (v - (2uk-sk)) with solution (sum_k(Vlk))^-1 * sum_k Vlk (2uk-sk)
    # <=> min_v 1/2 sum_k v^T Vlk v - v^T sum_k Vlk (2uk-sk) + const
    # {1} 
    #    
    # the delta_l update here (and even without! paralellism) is
    # bl = J_land.transpose() * fx0
    # bl_s = bl + L * JltJlDiag * (x0_l_ - s_l_) <=>
    # bl_s = bl +           Vl  * (x0_l_ - s_l_)
    # delta_l = - Vli * ((W.transpose() * delta_p).flatten() + bl_s)
    # solves what. then add xo_l to it (delta_v -> v)
    # recall x0_l + delta_l = 'v'
    # 1/2 delta_l^T Vl delta_l + delta_l^T [ (W.transpose() * delta_p) + bl_S]
    # insert x0_l + delta_l = v, recall x0_l is constant/known
    # 1/2 v^T Vl v + v^T [ (W.transpose() * delta_p) + bl_S] + 
    # 1/2 x0_l^T Vl x0_l + v^T Vl x0_l + x0_l^T [ (W.transpose() * delta_p) + bl_S]
    # remove constant parts
    # 1/2 v^T Vl v + v^T [ (W.transpose() * delta_p) + bl_S + Vl x0_l]
    # {2}
    # 1/2 v^T Vl v + v^T [ (W.transpose() * delta_p) + bl] + v^T Vl * (2 * x0_l_ - s_l_)] 
    # WEIRD: this is the same as {1} but + v^t b and +! v^T Vl * (2 * x0_l_ - s_l_)] not '-'!
    # Again. Correct u in non-split BA would be to sum over k for {2} and solve it. !Wo x-s term!
    # back to {1}
    # in {1} we have u+ for uk, so u+ = u + delta. we can insert 
    # min_v 1/2 sum_k v^T Vlk v - v^T sum_k Vlk (2(x0+delta) - sk)
    # 
    # So idea would be do {2} style updates wo. any s anywhere.
    # interpret as stochastic BA with updates on local parts. 
    # update for l done with sarah style updates.
    # how does this turn out for local updates. I do! need some adjustment from non considered parts.
    # actually the last gradient for these. Last + something.
    # 
    # delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl_s)
    # Compute insert solution here here we use Vli = (2*Vl)^-1
    # u+ = u - Vli/2 [(W.transpose() * delta_p).flatten() + bl_s]
    # u+ = u - Vli/2 [(W.transpose() * delta_p).flatten() + bl + Vl * (u - s)]
    # u+ = u - Vli/2 [(W.transpose() * delta_p).flatten() + bl] - 1/2(u - s)
    # u+ = (s+u)/2 - Vli/2 [(W.transpose() * delta_p).flatten() + bl] or 
    # u+ = u - (x-s)/2 - Vli/2 [(W.transpose() * delta_p).flatten() + bl] with x=u or v
    # 
    # v is prox on Vl/2 |v- 2u+ + s |^2 = 
    # 2u+-s = 2 * (s+u)/2 - s - 2 * (vli/2 .. )
    # 2u+-s = u - Vli [(W.transpose() * delta_p).flatten() + bl]
    #       = u + delta original for each k.
    # 
    # averaging over this is
    # (sum_k Vlk)^-1 { sum_k Vlk uk - [(W.transpose() * delta_p).flatten() + bl]k }
    # (sum_k Vlk)^-1 { sum_k Vlk uk} + real delta
    # why not v + real data instead of this mix of u's?
    # simulation: replace 2u+-s by 2u+ - s -u + v
    # or 2u+ - s - u to get delta. then add to v. or use real uk instead.
    # TODO! 

    # My tries to use Vl + eps * diag instead of 2Vl has issues, see above
    # we average over 2u-s not u, so, if we the add to the rhs Vl + eps * diag * (u-s) 
    # as L * JltJlDiag * (x0_l_ - s_l_) and invert with JltJl + L * JltJlDiag
    # with 0s in l we do not want to constrain we must return ..
    # shorter: we use 2u+ - s to compute average, return u + delta (desired solution)
    # we get 2u+-s as new solution. 
    # So we must return (u + delta + s)/2 to get 2 * (u + delta + s)/2 - s = u + delta as new v.
    # 
    # compute  we need to return   
    # Compute insert solution here here we use Vli = (2*Vl)^-1
    # u+ = u - Vli/2 [(W.transpose() * delta_p).flatten() + bl_s]
    # u+ = u - Vli/2 [(W.transpose() * delta_p).flatten() + bl + Vl * (u - s)]


    # TODO change 2
    if newForUnique:
        #print( "B JltJlDiag-bun ", JltJlDiag.data.reshape(-1,9)[landmarks_only_in_cluster_,:])
        #landmarks_in_many_cluster_ = np.invert(landmarks_only_in_cluster_)
        #print("landmarks_only_in_cluster_  ", landmarks_only_in_cluster_, " ", np.sum(landmarks_only_in_cluster_), " vs ", np.sum(1 - landmarks_only_in_cluster_) )
        #print("landmarks_in_many_cluster_  ", landmarks_in_many_cluster_, " ", np.sum(landmarks_in_many_cluster_), " vs ", np.sum(1 - landmarks_in_many_cluster_) )
        # print(np.repeat((np.ones(n_points_) * landmarks_in_many_cluster_)[..., np.newaxis], 3, axis=1).shape)
        # print(np.repeat((np.ones(n_points_) * landmarks_in_many_cluster_)[..., np.newaxis], 3, axis=1))
        #print("1. ", (np.ones(n_points_) * landmarks_in_many_cluster_))
        #print("2. ", np.repeat((np.ones(n_points_) * landmarks_in_many_cluster_).reshape(1,-1), 3))
        diag_present = 0.5 * diag_sparse( np.repeat((np.ones(n_points_) * landmarks_only_in_cluster_).reshape(-1,1), 3).flatten() )
        #diag_present = copy_selected_blocks(0.5 * diag_sparse(np.ones(n_points_*3)), landmarks_in_many_cluster_, 1)
        #print("diag_present ", diag_present.shape, " " , np.sum(diag_present.data==0.5), " " , np.sum(diag_present.data!=0.5) )
        #print( "Y ", (L * JltJlDiag).data.reshape(-1,9)[landmarks_only_in_cluster_,:])

        # see above, compensate for averaging
        xTest = x0_l_ - (diag_present*(x0_l_.flatten() - s_l_)).reshape(-1,3)
        # print("xt ", xTest[landmarks_only_in_cluster_,:])
        # print("x0 ", x0_l_[landmarks_only_in_cluster_,:])

        #test = diag_present + L * JltJlDiag + 1e-12 * Vl
        #print( "C vl-bun ", test.data.reshape(-1,9)[landmarks_only_in_cluster_,:]) # is diag mat off line basically 0

        # do for all, remove rhs completely
        # diag_presentB = 0.5 * diag_sparse( np.repeat((np.ones(n_points_)).reshape(-1,1), 3).flatten() )
        # xTest = x0_l_ - (diag_presentB*(x0_l_.flatten() - s_l_)).reshape(-1,3)

        return costEnd, x0_p_, xTest, L, diag_present + L * JltJlDiag + 1e-12 * Vl, delta_l.reshape(n_points_, 3)

    return costEnd, x0_p_, x0_l_, L, L * JltJlDiag + 1e-12 * Vl, delta_l.reshape(n_points_, 3)

    # recall solution wo. splitting is 
    # solve Vl x + bS + Vd () 
    #bl_s = bl + L * JltJlDiag * (x0_l_ - s_l_) # TODO: + or -. '+', see above
    #delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl_s).flatten()
    # sum_k delta_l Vlk delta_l + delta_l ((Wk.transpose() * delta_p) + bl_sk) argmin
    # all this is local per block. k blocks: sum_k Vlk = 3x3, we could instead 
    # return Vlk and bk = ((Wk.transpose() * delta_p) + bl_sk): 4 times #landmarks floats.
    # and compute the update, summing each and solving v = (sumk vlk)^-1 (sum_k bk).
    # would be cool if we could do n iterations locally -- with a gain.
    # much less cams then lms. do the inverse?
    # send only parts each time?single lms = worst?
    # i need to send v anyway and lms as well. above is 4x more.
    # this shows one core can do all this in parallel / local network can -> problem is parallelizable trivially
    # large network, better send minimal information. problem still send landmarks.
    # and diagonal? would be good if can work n steps locally.
    # what is missing? maybe accumulate 'Vl' on the way or upper bound
    # as max eigenvalue per landmark 3x3, or just sum row/col -> blockEigen, and 

def updateCluster(
    x0_p_,
    camera_indices_in_cluster_,
    point_indices_in_cluster_,
    points_2d_in_cluster_,
    points_3d_in_cluster_,
    landmark_s_in_cluster_,
    Vl_in_cluster_,
    L_in_cluster_,
    landmark_occurences,
    its_,
):
    # currently 
    cameras_indices_in_c_ = np.unique(camera_indices_in_cluster_) # indices into large vector from local: 0, .. , npart
    cameras_in_c = x0_p_[cameras_indices_in_c_] # first index -> 0, second -> 1 etc. 
    min_cam_index_in_c = np.min(camera_indices_in_cluster_)
    local_camera_indices_in_cluster_ = camera_indices_in_cluster_ - min_cam_index_in_c
    #print("local_camera_indices_in_cluster_", local_camera_indices_in_cluster_)
    # alternative 
    local_camera_indices_in_cluster_2 = np.zeros(camera_indices_in_cluster_.shape[0], dtype=int)
    for i in range(cameras_indices_in_c_.shape[0]):
        local_camera_indices_in_cluster_2[camera_indices_in_cluster_ == cameras_indices_in_c_[i]] = i
    #print("local_camera_indices_in_cluster_2", local_camera_indices_in_cluster_2)

    # torch_points_2d_in_c = torch_points_2d[points_2d_in_cluster[ci], :]
    torch_points_2d_in_c = from_numpy(points_2d_in_cluster_)
    torch_points_2d_in_c.requires_grad_(False)

    # take point_indices_in_cluster[ci] unique:
    unique_points_in_c_ = np.unique(point_indices_in_cluster_)
    # unique_points_in_c_[i] -> i, map each pi : point_indices_in_cluster[ci] to position in unique_points_in_c_[i]
    inverse_point_indices = -np.ones(np.max(unique_points_in_c_) + 1)  # all -1
    for i in range(unique_points_in_c_.shape[0]):
        inverse_point_indices[unique_points_in_c_[i]] = i

    landmarks_only_in_cluster_ = landmark_occurences[unique_points_in_c_] == 1
    #print("Unique landmarks  ", landmark_occurences, " ", landmark_occurences.shape, " ", np.min(landmark_occurences), " ", np.max(landmark_occurences))
    #print("Unique landmarks  ", landmarks_only_in_cluster_, " ", np.sum(landmarks_only_in_cluster_), " vs ", np.sum(1 - landmarks_only_in_cluster_) )

    point_indices_in_c = (
        point_indices_in_cluster_.copy()
    )  # np.zeros(point_indices_in_cluster_.shape)
    for i in range(point_indices_in_cluster_.shape[0]):
        point_indices_in_c[i] = inverse_point_indices[point_indices_in_c[i]]

    # put in unique points, adjust point_indices_in_cluster[ci] by id in unique_points_in_c_
    points_3d_in_c = points_3d_in_cluster_[unique_points_in_c_]
    landmark_s_in_c = landmark_s_in_cluster_[unique_points_in_c_]

    cost_, x0_p_c_, x0_l_c_, Lnew_c_, Vl_c_, delta_l_c_ = bundle_adjust(
        local_camera_indices_in_cluster_2,
        point_indices_in_c,
        landmarks_only_in_cluster_, # input those lms not present anywhere else to relax hold on those.
        torch_points_2d_in_c,
        cameras_in_c,
        points_3d_in_c,
        landmark_s_in_c,
        Vl_in_cluster_,
        L_in_cluster_,
        its_,
    )

    return (
        cost_,
        x0_p_c_,
        x0_l_c_,
        Lnew_c_,
        Vl_c_,
        unique_points_in_c_,
        cameras_indices_in_c_,
        delta_l_c_
    )

# 2 different derivative runs for additional res and original res. additional: only deriv wrt landmarks.
def updateCluster_palm(
    x0_p_,
    camera_indices_in_cluster_, # updated and all per res
    point_indices_in_cluster_,  # present, per res
    points_2d_in_cluster_,      # pre res
    points_3d_in_cluster_,      # all, just all, we select subset, todo: stupid
    Vl_in_cluster_,             # maybe storage for latest Vl. then next turn use it?
    L_in_cluster_,
    additional_point_indices_in_cluster_, # for the additional residuals.
    additional_camera_indices_in_cluster_,# camera indices for additional residuals. needed to compute f, not gradient -> set grad to false
    additional_points_2d_in_cluster_,     # additional residuals (rhs of them)
    covered_landmark_indices_in_cluster_, # those landmarks get updated in cluster, contains all of them, subset of (point_indices_in_cluster_) that are totally covered by res+additional res in cluster.
    additional_covered_landmark_indices_in_cluster_, # those landmarks get updated and are present in additional residual
    its_,
):
    cameras_indices_in_c_ = np.unique(camera_indices_in_cluster_) # indices into large vector from local: 0, .. , npart
    cameras_in_c = x0_p_[cameras_indices_in_c_] # first index -> 0, second -> 1 etc. 

    local_camera_indices_in_cluster = np.zeros(camera_indices_in_cluster_.shape[0], dtype=int)
    for i in range(cameras_indices_in_c_.shape[0]):
        local_camera_indices_in_cluster[camera_indices_in_cluster_ == cameras_indices_in_c_[i]] = i
    #print("local_camera_indices_in_cluster", local_camera_indices_in_cluster)

    # for the additional one we also need indices wrt. additional residuals to compute jacobian for landmarks observed from there
    additional_cameras_indices_in_c_ = np.unique(additional_camera_indices_in_cluster_) # indices into large vector from local: 0, .. , npart
    additional_local_camera_indices_in_cluster = np.zeros(additional_camera_indices_in_cluster_.shape[0], dtype=int)
    for i in range(additional_cameras_indices_in_c_.shape[0]):
        additional_local_camera_indices_in_cluster[additional_camera_indices_in_cluster_ == additional_cameras_indices_in_c_[i]] = i
    # also the values of those cameras are needed
    additional_cameras_in_c = x0_p_[additional_cameras_indices_in_c_] # first index -> 0, second -> 1, etc
    torch_additional_cameras_in_c = from_numpy(additional_cameras_in_c)
    torch_additional_cameras_in_c.requires_grad_(False)
    # res rhs:
    torch_points_2d_in_c = from_numpy(points_2d_in_cluster_)
    torch_points_2d_in_c.requires_grad_(False)
    additional_torch_points_2d_in_c = from_numpy(additional_points_2d_in_cluster_)
    additional_torch_points_2d_in_c.requires_grad_(False)

    # TODO: now landmarks. those local indices are the same! TODO_HERE Note: indices start at 0 for additional entities as well
    # need to remap, not just from 0.

    # take point_indices_in_cluster[ci] unique:
    unique_points_in_c_ = np.unique(point_indices_in_cluster_)
    # unique_points_in_c_[i] -> i, map each pi : point_indices_in_cluster[ci] to position in unique_points_in_c_[i]
    inverse_point_indices = -np.ones(np.max(unique_points_in_c_) + 1)  # all -1
    for i in range(unique_points_in_c_.shape[0]):
        inverse_point_indices[unique_points_in_c_[i]] = i # should be the index of all points -> index in cluster [as jacobian knows]

    #print("point in c minus covered ", np.setdiff1d(unique_points_in_c_, covered_landmark_indices_in_cluster_))
    verbose = False
    if verbose:
        print("covered minus points in c ", np.setdiff1d(covered_landmark_indices_in_cluster_, unique_points_in_c_))
        print("shapes u,c,a", unique_points_in_c_.shape, " ", covered_landmark_indices_in_cluster_.shape, " ", additional_covered_landmark_indices_in_cluster_.shape)
        print("shapes u-c, u-a, c-a", np.setdiff1d(unique_points_in_c_, covered_landmark_indices_in_cluster_).shape, " ", \
            np.setdiff1d(unique_points_in_c_,additional_covered_landmark_indices_in_cluster_).shape, " ", \
            np.setdiff1d(covered_landmark_indices_in_cluster_, additional_covered_landmark_indices_in_cluster_).shape)

        print("unique in a and u ", np.unique(additional_point_indices_in_cluster_).shape, np.unique(point_indices_in_cluster_).shape)

    point_indices_in_c = point_indices_in_cluster_.copy()  # np.zeros(point_indices_in_cluster_.shape)
    for i in range(point_indices_in_cluster_.shape[0]):
        point_indices_in_c[i] = inverse_point_indices[point_indices_in_c[i]]
    additional_point_indices_in_c = additional_point_indices_in_cluster_.copy() # np.zeros(point_indices_in_cluster_.shape)
    for i in range(additional_point_indices_in_c.shape[0]):
        additional_point_indices_in_c[i] = inverse_point_indices[additional_point_indices_in_c[i]] # for res
    covered_landmark_indices_c_ = covered_landmark_indices_in_cluster_.copy()
    for i in range(covered_landmark_indices_c_.shape[0]):
        covered_landmark_indices_c_[i] = inverse_point_indices[covered_landmark_indices_c_[i]] # 3dp[ids] = lm_c[ids]
    # later 3dp[covered_landmark_indices_in_cluster_] = lm_c_returned[covered_landmark_indices_c_]

    #########  need covered indices for the landmarks present in the additional subset of landmarks
    unique_additional_points_in_c_ = np.unique(additional_point_indices_in_cluster_) # only these!
    # unique_points_in_c_[i] -> i, map each pi : point_indices_in_cluster[ci] to position in unique_points_in_c_[i]
    inverse_additional_point_indices = -np.ones(np.max(unique_additional_points_in_c_) + 1)  # all -1
    for i in range(unique_additional_points_in_c_.shape[0]):
        inverse_additional_point_indices[unique_additional_points_in_c_[i]] = i
    # remove those not present in additional res 
    #additional_covered_landmark_indices = np.setdiff1d(covered_landmark_indices_in_cluster_, np.unique(point_indices_in_cluster_))
    additional_covered_landmark_indices_c_ = additional_covered_landmark_indices_in_cluster_.copy()
    for i in range(additional_covered_landmark_indices_c_.shape[0]):
        additional_covered_landmark_indices_c_[i] = inverse_additional_point_indices[additional_covered_landmark_indices_c_[i]] # 3dp[ids] = lm_c[ids]
    ################
    # delivers these additional_covered_landmark_indices_c_ -> from the landmarks present in additional, those that are covered by cluster
    # covered == indices that are updated from cluster alone, thus completely covered and 

    # put in unique points, adjust point_indices_in_cluster[ci] by id in unique_points_in_c_
    points_3d_in_c = points_3d_in_cluster_[unique_points_in_c_]

    cost_, x0_p_c_, x0_l_c_, Lnew_c_, Vl_c_ = local_bundle_adjust(
        local_camera_indices_in_cluster, # LOCAL 1st res
        point_indices_in_c,   # LOCAL 1st res
        torch_points_2d_in_c, # 1st res
        cameras_in_c,         # LOCAL 1st res
        points_3d_in_c,       # LOCAL all res
        additional_local_camera_indices_in_cluster, # local cam  ids 2nd res
        additional_point_indices_in_c,              # local land ids 2nd res
        additional_torch_points_2d_in_c,            # 2nd res
        torch_additional_cameras_in_c,              # LOCAL 2nd part
        covered_landmark_indices_c_, # those will be returned, subset of points_3d_in_c to update current estimate
        additional_covered_landmark_indices_c_, # covered landmarks present in additional res, subset of covered. 
        Vl_in_cluster_,
        L_in_cluster_,
        its_,
    )
    return (
            cost_,
            x0_p_c_,
            x0_l_c_,
            Lnew_c_,
            Vl_c_,
            covered_landmark_indices_in_cluster_,
            cameras_indices_in_c_,
    )

# difference: more input, update directly.
def palm_f(x0_p_, camera_indices_in_cluster_, point_indices_in_cluster_, 
           points_2d_in_cluster_, points_3d_in_cluster_, 
           additional_point_indices_in_cluster, additional_camera_indices_in_cluster, additional_points_2d_in_cluster, point_indices_already_covered_c, covered_landmark_indices_c,
           L_in_cluster_, Vl_in_cluster_, kClusters, innerIts, sequential) :
    cost_ = np.zeros(kClusters)
    landmark_v_ = points_3d_in_cluster_[0].copy()

    for ci in range(kClusters):
    #for ci in np.random.permutation(kClusters):
        (
            cost_c_,
            x0_p_c_,
            x0_l_c_,
            Lnew_c_,
            Vl_c_,
            update_point_indices_in_c_,
            update_cameras_indices_in_c_,
        ) = updateCluster_palm(
            x0_p_,
            camera_indices_in_cluster_[ci], # updated and all per res
            point_indices_in_cluster_[ci],  # present, per res
            points_2d_in_cluster_[ci],      # pre res
            points_3d_in_cluster_[ci],      # all, just all, we select subset
            Vl_in_cluster_[ci],             # maybe storage for latest Vl. then next turn use it?
            L_in_cluster_[ci],              # maybe use for L*diag + Vl on prox linear term.
            #additional_cameras_in_cluster[ci], # just unique additional_camera_indices
            additional_point_indices_in_cluster[ci], # for the additional residuals.
            additional_camera_indices_in_cluster[ci], # camera indices for additional residuals. needed to compute f, not gradient -> set grad to false
            additional_points_2d_in_cluster[ci], # additional residuals (rhs of them)
            point_indices_already_covered_c[ci],
            covered_landmark_indices_c[ci], # those landmarks get updated in addition to unique (point_indices_in_cluster_) since totally covered
            its_=innerIts,
        )
        cost_[ci] = cost_c_
        L_in_cluster_[ci] = Lnew_c_
        Vl_in_cluster_[ci] = Vl_c_
        landmark_v_[update_point_indices_in_c_, :] = x0_l_c_.copy() # global ensure disjoint
        use_inertia = False # bad
        if use_inertia:
            x0_p_[update_cameras_indices_in_c_] = x0_p_[update_cameras_indices_in_c_] + tau * (x0_p_c_- x0_p_[update_cameras_indices_in_c_])
        else:
            x0_p_[update_cameras_indices_in_c_] = x0_p_c_.copy() # this is also instant update.
        if sequential: # can be parallel since all use same input. Then update from outside.
            if use_inertia:
                for ci in range(kClusters): # update for all
                    tau = np.sqrt(1.2)
                    points_3d_in_cluster_[ci][update_point_indices_in_c_, :] = points_3d_in_cluster_[ci][update_point_indices_in_c_, :] \
                        + tau * (x0_l_c_ - points_3d_in_cluster_[ci][update_point_indices_in_c_, :])
            else:
                for ci in range(kClusters): # update for all
                    points_3d_in_cluster_[ci][update_point_indices_in_c_, :] = x0_l_c_.copy()

        # print(ci, " 3d ", points_3d_in_cluster_[ci][landmark_occurences==1, :]) # indeed 1 changed rest is constant
        # print(ci, " vl ", vl.data.reshape(-1,9)[landmarks_only_in_cluster_,:])  # indeed diagonal
    return (cost_, L_in_cluster_, Vl_in_cluster_, landmark_v_, x0_p_)


def prox_f(x0_p_, camera_indices_in_cluster_, point_indices_in_cluster_, 
           points_2d_in_cluster_, points_3d_in_cluster_, landmark_s_in_cluster_, 
           L_in_cluster_, Vl_in_cluster_, kClusters, innerIts, sequential) :
    cost_ = np.zeros(kClusters)
    delta_l_in_cluster_ = [0 for elem in range(kClusters)]

    num_landmarks = points_3d_in_cluster_[0].shape[0]
    landmark_occurences = np.zeros(num_landmarks)
    for ci in range(kClusters):
        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        landmark_occurences[unique_points_in_c_] +=1

    # for ci in range(kClusters):
    #     print(ci, " 3d " ,points_3d_in_cluster_[ci][landmark_occurences==1, :])

    if sequential:
        for ci in range(kClusters):

            delta_l_in_cluster_[ci] = np.zeros(points_3d_in_cluster_[ci].shape)

            (
                cost_c_,
                x0_p_c_,
                x0_l_c_,
                Lnew_c_,
                Vl_c_,
                unique_points_in_c_,
                cameras_indices_in_c_,
                delta_l_c_,
            ) = updateCluster(
                x0_p_,
                camera_indices_in_cluster_[ci],
                point_indices_in_cluster_[ci],
                points_2d_in_cluster_[ci],
                points_3d_in_cluster_[ci],
                landmark_s_in_cluster_[ci],
                Vl_in_cluster_[ci],
                L_in_cluster_[ci],
                landmark_occurences,
                its_=innerIts,
            )
            cost_[ci] = cost_c_
            L_in_cluster_[ci] = Lnew_c_
            Vl_in_cluster_[ci] = Vl_c_
            points_3d_in_cluster_[ci][unique_points_in_c_, :] = x0_l_c_
            x0_p_[cameras_indices_in_c_] = x0_p_c_
            delta_l_in_cluster_[ci][unique_points_in_c_, :] = delta_l_c_
    else:
        # not not ,prefer="threads" ?
        # results = Parallel(n_jobs=8,prefer="threads")(delayed(getJacSin)(i, i + step, camera_indices, point_indices, x0_t_cam, x0_t_land, torch_points_2d) for i in np.arange(0, full, step))
        results = Parallel(n_jobs=3)(
            delayed(updateCluster)(
                x0_p_,
                camera_indices_in_cluster_[ci],
                point_indices_in_cluster_[ci],
                points_2d_in_cluster_[ci],
                points_3d_in_cluster_[ci],
                landmark_s_in_cluster_[ci],
                Vl_in_cluster_[ci],
                L_in_cluster_[ci],
                landmark_occurences,
                its_=innerIts,
            )
            for ci in range(kClusters)
        )
        for ci in range(kClusters):
            cost_[ci] = results[ci][0]
            L_in_cluster_[ci] = results[ci][3]
            Vl_in_cluster_[ci] = results[ci][4]
            points_3d_in_cluster_[ci][results[ci][5], :] = results[ci][2]
            x0_p_[results[ci][6]] = results[ci][1]

    for ci in range(kClusters):
        vl = Vl_in_cluster_[ci]
        unique_points_in_c_ = np.unique(point_indices_in_cluster_[ci])
        landmarks_only_in_cluster_ = landmark_occurences[unique_points_in_c_] == 1
        globalSingleLandmarksA_in_c[ci] = landmarks_only_in_cluster_.copy()
        globalSingleLandmarksB_in_c[ci] = landmark_occurences==1

        # print(ci, " 3d ", points_3d_in_cluster_[ci][landmark_occurences==1, :]) # indeed 1 changed rest is constant
        # print(ci, " vl ", vl.data.reshape(-1,9)[landmarks_only_in_cluster_,:])  # indeed diagonal
    return (cost_, L_in_cluster_, Vl_in_cluster_, points_3d_in_cluster_, x0_p_, delta_l_in_cluster_, globalSingleLandmarksA_in_c, globalSingleLandmarksB_in_c)

# fill lists G and F, with g and f = g - old g, sets of size m, 
# at position it % m, c^t compute F^tF c + lamda (c - 1/k)^2, sum c=1
# g = x0, f = delta. Actullay xnew = xold + delta.
def RNA(G, F, g, f, it_, m_, Fe, fe, lamda):
    crefVersion = True
    lamda = 15.0 # cref version needs larger 
    h = 1 - np.sqrt(1.1) #-0.1 # 2 / (L+mu) -- should 1/diag * F^t F * c
    id_ = it_ % m_
    if len(G) >= m_:
        #print("it, it%m", it, " ", it % m)
        G[id_] = np.squeeze(g)
        F[id_] = np.squeeze(f)
        Fe[id_] = np.squeeze(fe)
    else:
        G.append(np.squeeze(g))
        F.append(np.squeeze(f))
        Fe.append(np.squeeze(fe))
    mg = len(G)
    cref = np.zeros(mg)
    if mg >= m_:
        cref[id_] = 1
    else:
        cref[mg-1] = 1

    Gs_ = np.concatenate(G).reshape(mg, -1).transpose()
    Fs_ = np.concatenate(F).reshape(mg, -1).transpose()
    Fes_ = np.concatenate(Fe).reshape(mg, -1).transpose()
    #print("Fs ", Fs.shape)
    FtF = Fs_.transpose().dot(Fs_) # why dot?
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
    extrapolation = Gs_.dot(c) #+ 0.1 * Fs_.dot(c)
    extrapolationF = Fes_.dot(c)

    print("c ", c, " ", c.shape, id_)

    # shape utter non sense
    #print("extrapolation ", extrapolation.shape, " ", g.shape)
    return (G, F, Fe, np.squeeze(extrapolation - h * extrapolationF))


def BFGS_direction(r, ps, qs, rhos, k, mem, mu):
    # r = -r # not needed with below
    # lookup k-1, k-mem entries. 
    alpha = np.zeros([mem,1])
    r = np.squeeze(r)
    for i in range(k-1, np.maximum(k-mem,-1), -1):
    #for i in range(k-1, np.maximum(k-mem-1,-1), -1): # todo might be correct.
        #print("1i", i) # k-1, .. k-mem usually
        j = i % mem # j>=0
        #print("1j", j)
        #print("j ", j, " ", r.shape, ps[j].shape)
        alpha[j] = np.dot(r, ps[j]) * rhos[j]
        r = r - alpha[j]*qs[j]
        if (rhos[j]>0):
            print(j, " 1st. al ", alpha[j], " rh ", rhos[j], " qs " , np.linalg.norm(qs[j],2), " ps " , np.linalg.norm(ps[j],2) )

    dk_ = mu * r

    for i in range(np.maximum(k-mem, 0), k):
        #print("2i", i) # k-1, .. k-mem usually
        j = i % mem # j>=0
        #print("2j", j)
        beta = rhos[j] * np.dot(dk_, qs[j])
        dk_ = dk_ + ps[j] * (alpha[j] - beta)
        if (rhos[j]>0):
            print(j, " 2nd. al ", alpha[j], " rh ", rhos[j], " be ", beta, " qs " , np.linalg.norm(qs[j],2), " ps " , np.linalg.norm(ps[j],2) )

    return dk_

##############################################################################
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

x0_p = cameras.ravel()
# x0_l = points_3d.ravel()
# x0 = np.hstack((x0_p, x0_l))
x0_p = x0_p.reshape(n_cameras, 9)
# x0_l = x0_l.reshape(n_points, 3)

# torch_points_2d = from_numpy(points_2d)
# torch_points_2d.requires_grad_(False)
# x0_t = from_numpy(x0)

# init. Duplicate only non unique variables. s and u contain only those duplicates. 
# initially all variables are contained, but selected by indices .. 
# Init s = v = u. = dupe all landmarks, now twice.
# 1. s+ = s + v-u = s.
# prox on (s+)
# prox on 2u+ - s+. Adjust average fct.
# update s+ = s + v+ - u+ = s + 2u+ - s - u+ = u+ .. Actually not since v+ != 2u-s
# u := points_3d_in_cluster
# s := landmark_s_in_cluster
# v := output, temporary

# 1. take problem and split, sort indices by camera, define local global map and test it.
startL = 1
kClusters = 3 # 6 cluster also not bad at all !
innerIts = 1  # change to get an update, not 1 iteration
its = 20
cost = np.zeros(kClusters)
lastCost = 1e20
lastCostDRE = 1e20
basic_version = True # accelerated or basic
sequential = True
linearize_at_last_solution = False # linearize at uk or v. Maybe check energy at u or v. Currently with energy check: always pick v (False here)
Gs = [] 
Fs = []
Fes = []
rnaBufferSize = 6

# DRE:
# f(x) + 1/2 (v^T Vlk * v - v^T 2 * Vlk (2x - sk) ) for x=u/v. Does not look right .. haeh
# 
# The Lagrangian is sum_k f_k (x_k^t) - <mu^t_k, x_k^t - Bk z^t> + rho_k/2 | Bk z^t-x_k^t |^2
# Apparently the difference is only (for Y = x_k^t or Bk z):
# f_k (Y) + rho_k/2 | Y - Bk z^t |^2 - <mu^t_k, Y - Bk z^t>.
# and we recall lambda = s^+ - v = s + (v-u) - v = s - u.
#
# Hence it is fv' * fv versus fu' * fu + rho_k|u_k - v_k|^2 - rho_k <s_k - u_k, u_k - v_k> in DRS variables.
# how did this turn into my dre cost? 
# insert v -> only fk(v) remains. insert u: 

globalSingleLandmarksA_in_c = [0 for x in range(kClusters)]
globalSingleLandmarksB_in_c = [0 for x in range(kClusters)]

read_cluster = True
write_cluster = False

if read_cluster:
    L_in_cluster = [0 for x in range(kClusters)] # dummy fill list
    camera_indices_in_cluster = [0 for x in range(kClusters)] # dummy fill list
    point_indices_in_cluster = [0 for x in range(kClusters)] # dummy fill list
    points_2d_in_cluster = [0 for x in range(kClusters)] # dummy fill list
    cluster_to_camera = [0 for x in range(kClusters)] # dummy fill list
    points_3d_in_cluster = [0 for x in range(kClusters)] # dummy fill list
    additional_point_indices_in_cluster = [0 for x in range(kClusters)] # dummy fill list
    additional_camera_indices_in_cluster = [0 for x in range(kClusters)] # dummy fill list
    additional_points_2d_in_cluster = [0 for x in range(kClusters)] # dummy fill list
    point_indices_already_covered_c = [0 for x in range(kClusters)] # dummy fill list
    covered_landmark_indices_c = [0 for x in range(kClusters)] # dummy fill list
    for ci in range(kClusters):
        camera_indices_in_cluster[ci] = np.fromfile("camera_indices_in_cluster" + str(ci) + ".dat", dtype=int)
        point_indices_in_cluster[ci] = np.fromfile("point_indices_in_cluster" + str(ci) + ".dat", dtype=int)
        print("ci", point_indices_in_cluster[ci].shape)
        points_2d_in_cluster[ci] = np.fromfile("points_2d_in_cluster" + str(ci) + ".dat", dtype=float).reshape(-1,2)
        cluster_to_camera[ci] = np.fromfile("cluster_to_camera" + str(ci) + ".dat", dtype=int)
        points_3d_in_cluster[ci] = np.fromfile("points_3d_in_cluster" + str(ci) + ".dat", dtype=float).reshape(-1,3)
        print("ci", points_3d_in_cluster[ci].shape)
        print("ci", points_2d_in_cluster[ci].shape)
        additional_point_indices_in_cluster[ci] = np.fromfile("additional_point_indices_in_cluster" + str(ci) + ".dat", dtype=int)
        additional_camera_indices_in_cluster[ci] = np.fromfile("additional_camera_indices_in_cluster" + str(ci) + ".dat", dtype=int)
        additional_points_2d_in_cluster[ci] = np.fromfile("additional_points_2d_in_cluster" + str(ci) + ".dat", dtype=float).reshape(-1,2)
        point_indices_already_covered_c[ci] = np.fromfile("point_indices_already_covered_c" + str(ci) + ".dat", dtype=int)
        covered_landmark_indices_c[ci] = np.fromfile("covered_landmark_indices_c" + str(ci) + ".dat", dtype=int)
        L_in_cluster[ci] = startL
    #exit()
else:
    (
        camera_indices_in_cluster,
        point_indices_in_cluster,
        points_2d_in_cluster,
        cluster_to_camera,
        points_3d_in_cluster,
        additional_point_indices_in_cluster, additional_camera_indices_in_cluster, additional_points_2d_in_cluster, point_indices_already_covered_c, covered_landmark_indices_c,
        L_in_cluster,
    ) = cluster_by_camera_gpt(
        camera_indices, points_3d, points_2d, point_indices, kClusters, startL
    )

if write_cluster:
    for ci in range(kClusters):
        camera_indices_in_cluster[ci].tofile("camera_indices_in_cluster" + str(ci) + ".dat")
        point_indices_in_cluster[ci].tofile("point_indices_in_cluster" + str(ci) + ".dat")
        points_2d_in_cluster[ci].tofile("points_2d_in_cluster"+ str(ci) +".dat")
        cluster_to_camera[ci].tofile("cluster_to_camera"+ str(ci) +".dat")
        points_3d_in_cluster[ci].tofile("points_3d_in_cluster"+ str(ci) +".dat")
        print("ci", points_3d_in_cluster[ci].shape)
        print("ci", points_2d_in_cluster[ci].shape)
        additional_point_indices_in_cluster[ci].tofile("additional_point_indices_in_cluster"+ str(ci) +".dat")
        additional_camera_indices_in_cluster[ci].tofile("additional_camera_indices_in_cluster"+ str(ci) +".dat")
        additional_points_2d_in_cluster[ci].tofile("additional_points_2d_in_cluster"+ str(ci) +".dat")
        point_indices_already_covered_c[ci].tofile("point_indices_already_covered_c"+ str(ci) +".dat")
        covered_landmark_indices_c[ci].tofile("covered_landmark_indices_c"+ str(ci) +".dat")
        print("ci", point_indices_in_cluster[ci].shape)
    #exit()

multiCluster = False # verify overlap ? setting L delayed is a slow down. Tried to increase variance, about same. Also same as non clustered. Maybe each step different clusters?
if multiCluster:
    (
        camera_indices_in_cluster_2,
        point_indices_in_cluster_2,
        points_2d_in_cluster_2,
        cluster_to_camera_2,
        points_3d_in_cluster_2,
        additional_point_indices_in_cluster_2, additional_camera_indices_in_cluster_2, additional_points_2d_in_cluster_2, point_indices_already_covered_c_2, covered_landmark_indices_c_2,
        L_in_cluster_2,
    ) = cluster_by_camera_gpt(
        camera_indices, points_3d, points_2d, point_indices, kClusters, startL, init_cam_id=30, init_lm_id=5, seed=1234
    )
    Vl_in_cluster_2 = [0 for x in range(kClusters)] # dummy fill list

print(L_in_cluster)
Vl_in_cluster = [0 for x in range(kClusters)] # dummy fill list
landmark_s_in_cluster = [elem.copy() for elem in points_3d_in_cluster]
landmark_v = points_3d_in_cluster[0].copy()

if basic_version:

    for it in range(its):
        run_DRS = False
        if run_DRS:
            start = time.time()
            (
                cost,
                L_in_cluster,
                Vl_in_cluster,
                points_3d_in_cluster,
                x0_p,
                delta_l_in_cluster,
                globalSingleLandmarksA_in_c,
                globalSingleLandmarksB_in_c
            ) = prox_f(
                x0_p, camera_indices_in_cluster, point_indices_in_cluster, points_2d_in_cluster, 
                points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, 
                kClusters, innerIts=innerIts, sequential=True,
                )
            end = time.time()

            currentCost = np.sum(cost)
            print(it, " ", round(currentCost), " gain ", round(lastCost - currentCost), ". ============= sum fk update takes ", end - start," s",)

            landmark_v = average_landmarks_new(
                point_indices_in_cluster, points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, landmark_v, delta_l_in_cluster
            )

            #DRE cost BEFORE s update
            dre = cost_DRE(point_indices_in_cluster, points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, landmark_v) + currentCost

            tau = 2
            for ci in range(kClusters):
                landmark_s_in_cluster[ci] = landmark_s_in_cluster[ci] + tau*(landmark_v - points_3d_in_cluster[ci]) # update s = s + v - u.

            #DRE cost AFTER s update
            #dre = cost_DRE(point_indices_in_cluster, points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, landmark_v) + currentCost

            primal_cost_v = 0
            for ci in range(kClusters):
                primal_cost_v += primal_cost(
                    x0_p,
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    landmark_v) #points_3d_in_cluster[ci]) # v not u
            primal_cost_u = 0
            for ci in range(kClusters):
                primal_cost_u += primal_cost(
                    x0_p,
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    points_3d_in_cluster[ci]) # v not u

            print( it, " ======== DRE ====== ", round(dre) , " ========= gain " , \
                round(lastCostDRE - dre), "==== f(v)= ", round(primal_cost_v), " f(u)= ", round(primal_cost_u))

            lastCost = currentCost
            # print(" output shapes ", x0_p_c.shape, " ", x0_l_c.shape, " takes ", end-start , " s")
            if False and lastCostDRE - dre < 1:
                break
            lastCostDRE = dre

            # fill variables for update: linearize at u or v.
            for ci in range(kClusters):
                # point_indices = np.unique(point_indices_in_cluster[ci])
                # print ("output.shape ", output.shape, " ", points_3d_in_cluster[ci].shape, " ", point_indices_in_cluster[ci].shape, " ", point_indices.shape )
                # points_3d_in_cluster[ci][point_indices,:] = output[point_indices,:]
                if not linearize_at_last_solution: # linearize at v / average solution, same issue I suppose. Yes. solution is too return the new gradient, s.t. update of v is wrt to current situation.
                    points_3d_in_cluster[ci]  = landmark_v.copy() # init at v, above at u
                    #points_3d_in_cluster[ci]  = landmark_s_in_cluster[ci].copy() # penalty is 0 but init position is WRONG
        else:
            if multiCluster:
                # need to map points_3d_in_cluster to points_3d_in_cluster_ just copy
                if it %2 ==0:
                    (
                        cost,
                        L_in_cluster,
                        Vl_in_cluster,
                        landmark_v,
                        x0_p_new
                    ) = palm_f(
                        x0_p, camera_indices_in_cluster, point_indices_in_cluster, points_2d_in_cluster, 
                        points_3d_in_cluster,
                        additional_point_indices_in_cluster, additional_camera_indices_in_cluster, additional_points_2d_in_cluster, 
                        point_indices_already_covered_c, covered_landmark_indices_c,
                        L_in_cluster, Vl_in_cluster, kClusters, innerIts=innerIts, sequential=True,
                        )
                else:
                    (
                        cost,
                        L_in_cluster_2,
                        Vl_in_cluster_2,
                        landmark_v,
                        x0_p_new
                    ) = palm_f(
                        x0_p, camera_indices_in_cluster_2, point_indices_in_cluster_2, points_2d_in_cluster_2, 
                        points_3d_in_cluster, # ?!?!
                        additional_point_indices_in_cluster_2, additional_camera_indices_in_cluster_2, additional_points_2d_in_cluster_2, 
                        point_indices_already_covered_c_2, covered_landmark_indices_c_2,
                        L_in_cluster_2, Vl_in_cluster_2, kClusters, innerIts=innerIts, sequential=True,
                        )
            else:
                (
                    cost,
                    L_in_cluster,
                    Vl_in_cluster,
                    landmark_v,
                    x0_p_new
                ) = palm_f(
                    x0_p, camera_indices_in_cluster, point_indices_in_cluster, points_2d_in_cluster, 
                    points_3d_in_cluster, 
                    additional_point_indices_in_cluster, additional_camera_indices_in_cluster, additional_points_2d_in_cluster, point_indices_already_covered_c, covered_landmark_indices_c,
                    L_in_cluster, Vl_in_cluster, kClusters, innerIts=innerIts, sequential=True,
                    )

            primal_cost_v = 0
            for ci in range(kClusters):
                primal_cost_v += primal_cost(
                    x0_p_new,
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    landmark_v)
            print( it, "==== f(v)= ", round(primal_cost_v), " and ", round(np.sum(cost)))

            extrapolate = False
            if extrapolate:
                # L_rna = 1 # max(L_in_cluster)
                # delta_l = landmark_v - points_3d_in_cluster[0]
                # delta_p = x0_p_new - x0_p
                # delta = np.hstack((delta_p.flatten(), delta_l.flatten()))
                # v__ = np.hstack((x0_p_new.flatten(), landmark_v.flatten()))
                # Gs, Fs, Fes, extr = RNA(Gs, Fs, v__, L_rna * delta, it, rnaBufferSize, Fes, delta, lamda = 1)
                # _, _, _, extr = RNA(Gs, Fs, v__, L_rna * delta, it+1, rnaBufferSize, Fes, delta, lamda = 1) # here not helpful.
                # camera_ext = extr[:n_cameras * 9].reshape(n_cameras, 9)
                # point_ext  = extr[n_cameras * 9:].reshape(n_points, 3)

                # pure acc, maybe sqrt(1.1)
                tau = np.sqrt(1.1)
                point_ext  = points_3d_in_cluster[0] + tau * (landmark_v - points_3d_in_cluster[0])
                camera_ext = x0_p + tau * (x0_p_new -x0_p)

                primal_cost_ext = 0
                for ci in range(kClusters):
                    primal_cost_ext += primal_cost(
                        camera_ext,
                        camera_indices_in_cluster[ci],
                        point_indices_in_cluster[ci],
                        points_2d_in_cluster[ci],
                        point_ext)
                print( it, "==== accelerated f(v)= ", round(primal_cost_ext), " basic ", round(primal_cost_v), " gain ", round(primal_cost_v-primal_cost_ext) )

                if primal_cost_ext < primal_cost_v:
                    for ci in range(kClusters):
                        points_3d_in_cluster[ci] = point_ext.copy()
                    x0_p = camera_ext.copy()
                else:
                    for ci in range(kClusters):
                        points_3d_in_cluster[ci] = landmark_v.copy()
                    x0_p = x0_p_new.copy()

            else:
                use_inertia_simple = False # does a bit with small tau. Beware to use when done in sequential update!
                if use_inertia_simple: # RNA works, so maybe use here with L*delta, see acceleration
                    tau = np.sqrt(1.1)
                    for ci in range(kClusters):
                        points_3d_in_cluster[ci]  = points_3d_in_cluster[ci] + tau * (landmark_v - points_3d_in_cluster[ci])
                    x0_p = x0_p + tau * (x0_p_new -x0_p)

                    primal_cost_v = 0
                    for ci in range(kClusters):
                        primal_cost_v += primal_cost(
                            x0_p,
                            camera_indices_in_cluster[ci],
                            point_indices_in_cluster[ci],
                            points_2d_in_cluster[ci],
                            points_3d_in_cluster[ci])
                    print( it, "==== + inertia f(v)= ", round(primal_cost_v), " and ", round(np.sum(cost)))

                else:
                    for ci in range(kClusters):
                        points_3d_in_cluster[ci] = landmark_v.copy()
                    x0_p = x0_p_new.copy()



else:

    bfgs_mem = 6 # 2:Cost @50:  -12.87175888983266, 6: cost @ 50: 12.871757400143322
    bfgs_mu = 1.0
    bfgs_qs = np.zeros([bfgs_mem, kClusters * 3 * n_points]) # access/write with % mem
    bfgs_ps = np.zeros([bfgs_mem, kClusters * 3 * n_points])
    bfgs_rhos = np.zeros([bfgs_mem, 1])
    landmark_s_in_cluster_pre = [0 for x in range(kClusters)] # dummy fill list
    search_direction = [0 for x in range(kClusters)] # dummy fill list
    landmark_s_in_cluster_bfgs = [0 for x in range(kClusters)] # dummy fill list
    steplength = [0 for x in range(kClusters)]
    lastCostDRE_bfgs = lastCostDRE
    Gs = [] 
    Fs = []
    Fes = []
    rnaBufferSize = 6

    # Only it 0: update s,u,v.
    start = time.time()
    (
        cost,
        L_in_cluster,
        Vl_in_cluster,
        points_3d_in_cluster,
        x0_p,
        delta_l_in_cluster,
    ) = prox_f(
        x0_p, camera_indices_in_cluster, point_indices_in_cluster, points_2d_in_cluster, 
        points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, kClusters, innerIts=innerIts, sequential=True,
        )
    end = time.time()
    currentCost = np.sum(cost)
    print(-1, " ", round(currentCost), " gain ", round(lastCost - currentCost), ". ============= sum fk update takes ", end - start," s",)

    landmark_v = average_landmarks_new( # v update
        point_indices_in_cluster, points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, landmark_v, delta_l_in_cluster
    )

    for it in range(its):

        steplength = 0
        for ci in range(kClusters):
            landmark_s_in_cluster_pre[ci] = landmark_s_in_cluster[ci] + landmark_v - points_3d_in_cluster[ci] # update s = s + v - u.
            steplength += np.linalg.norm(landmark_v - points_3d_in_cluster[ci], 2)

        # debugging cost block ################
        if False:
            dre = cost_DRE(point_indices_in_cluster, points_3d_in_cluster, landmark_s_in_cluster, L_in_cluster, Vl_in_cluster, landmark_v) + currentCost
            # here potentially skip line search and increase weight of step sizes (all?).

            primal_cost_v = 0
            for ci in range(kClusters):
                primal_cost_v += primal_cost(
                    x0_p,
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    landmark_v) #points_3d_in_cluster[ci]) # v not u
            primal_cost_u = 0
            for ci in range(kClusters):
                primal_cost_u += primal_cost(
                    x0_p,
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    points_3d_in_cluster[ci])

            print( it, " ======== DRE ====== ", round(dre) , " ========= gain " , \
                round(lastCostDRE - dre), "==== f(v)= ", round(primal_cost_v), " f(u)= ", round(primal_cost_u))
            # end debugging cost block ################

        # get line search direction and update bfgs data
        # operate with np concatenate to get large vector and reshape search_direction here?
        bfgs_r = np.zeros(kClusters * 3 * n_points)
        rna_s  = np.zeros(kClusters * 3 * n_points)
        for ci in range(kClusters): #bfgs_r = u-v
            bfgs_r[ci * 3 * n_points: (ci+1) * 3 * n_points] = landmark_v.flatten() - points_3d_in_cluster[ci].flatten()
            rna_s[ci * 3 * n_points: (ci+1) * 3 * n_points] = landmark_s_in_cluster_pre[ci].flatten()

        use_bfgs = True
        if use_bfgs:
            dk = BFGS_direction(bfgs_r, bfgs_ps, bfgs_qs, bfgs_rhos, it, bfgs_mem, bfgs_mu)
            dk_stepLength = np.linalg.norm(dk, 2)
            multiplier = steplength / dk_stepLength
        else:
            Gs, Fs, Fes, dk = RNA(Gs, Fs, rna_s, bfgs_r, it, rnaBufferSize, Fes, bfgs_r, lamda = 1)
            dk = dk - (rna_s - bfgs_r)
            dk_stepLength = np.linalg.norm(dk, 2)
            multiplier = 1

        for ci in range(kClusters):
            search_direction[ci] = dk[ci * 3 * n_points: (ci+1) * 3 * n_points].reshape(n_points, 3) # reshape 3, n_points ?

        # need a check to reject idiotic proposals:
        # rho(u-s)^2 is gigantic
        # line search:
        line_search_iterations = 3
        print(" ..... step length ", steplength, " bfgs step ", dk_stepLength, " ratio ", multiplier)
        for ls_it in range(line_search_iterations):
            tk = (1 - ls_it) / (line_search_iterations-1) # i=2: 1, 0 | i=3: 1, 1/2, 1 | .. or multiplier * 
            #---- |u-s|^2_D  12854 |u-v|^2_D  131 |2u-s-v|^2_D  13090 |u-v|^2  1234
            #10 / 0  ======== DRE BFGS ======  29010  ========= gain  513 ==== f(v)=  29336  f(u)=  28616  ~=  28615.853615225344
            #13 / 0  ======== DRE BFGS ======  28612  ========= gain  469 ==== f(v)=  28672  f(u)=  28494  ~=  28493.593312067547
            for ci in range(kClusters):
                landmark_s_in_cluster_bfgs[ci] = (1-tk) * landmark_s_in_cluster_pre[ci] + tk * (landmark_s_in_cluster[ci] + multiplier * search_direction[ci])
                #print(" bfgs_r ", bfgs_r[ci * 3 * n_points: (ci+1) * 3 * n_points].reshape(n_points, 3))
                #print(" search_direction[ci] ", search_direction[ci])

            # prox on line search s:
            #print("1. x0_p", "points_3d_in_cluster", points_3d_in_cluster)
            points_3d_in_cluster_bfgs = [elem.copy() for elem in points_3d_in_cluster]
            L_in_cluster_bfgs = L_in_cluster.copy()
            Vl_in_cluster_bfgs = [elem.copy() for elem in Vl_in_cluster]
            (   cost_bfgs,
                L_in_cluster_bfgs,
                Vl_in_cluster_bfgs,
                points_3d_in_cluster_bfgs,
                x0_p_bfgs,
                delta_l_in_cluster
            ) = prox_f(
                x0_p.copy(), camera_indices_in_cluster, point_indices_in_cluster, points_2d_in_cluster, 
                points_3d_in_cluster_bfgs, landmark_s_in_cluster_bfgs, L_in_cluster_bfgs, Vl_in_cluster_bfgs, kClusters, innerIts=innerIts, sequential=True,
                )
            #print("2. x0_p", "points_3d_in_cluster", points_3d_in_cluster)
            currentCost_bfgs = np.sum(cost_bfgs)
            landmark_v_bfgs = average_landmarks_new( # v update
                point_indices_in_cluster, points_3d_in_cluster_bfgs, landmark_s_in_cluster_bfgs, L_in_cluster_bfgs, Vl_in_cluster_bfgs, landmark_v, delta_l_in_cluster
                )
            #print("3. x0_p", x0_p, "points_3d_in_cluster", points_3d_in_cluster)
            # update buffers
            if ls_it == 0: # todo: the one we accept put here, no?
                #bfgs_ps[it % bfgs_mem] = -dk * multiplier
                bfgs_ps[it % bfgs_mem] = -bfgs_r # this is not so much overshooting as dk
                bfgs_rr = np.zeros(kClusters * 3 * n_points)
                for ci in range(kClusters):
                    bfgs_rr[ci * 3 * n_points: (ci+1) * 3 * n_points] = landmark_v_bfgs.flatten() - points_3d_in_cluster_bfgs[ci].flatten() # flatten?
                bfgs_qs[it % bfgs_mem] = bfgs_rr - bfgs_r
                bfgs_rhos[it % bfgs_mem] = np.maximum(0., 1./ bfgs_qs[it % bfgs_mem].dot(bfgs_ps[it % bfgs_mem]))

            # eval cost
            dre_bfgs = cost_DRE(point_indices_in_cluster, points_3d_in_cluster_bfgs, landmark_s_in_cluster_bfgs, L_in_cluster_bfgs, Vl_in_cluster_bfgs, landmark_v_bfgs)
            dre_bfgs += currentCost_bfgs

            # debugging cost block ################
            primal_cost_v = 0
            for ci in range(kClusters):
                primal_cost_v += primal_cost(
                    x0_p_bfgs,
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    landmark_v_bfgs) #points_3d_in_cluster[ci]) # v not u
            primal_cost_u = 0
            for ci in range(kClusters):
                primal_cost_u += primal_cost(
                    x0_p_bfgs,
                    camera_indices_in_cluster[ci],
                    point_indices_in_cluster[ci],
                    points_2d_in_cluster[ci],
                    points_3d_in_cluster_bfgs[ci])

            print( it, "/", ls_it, " ======== DRE BFGS ====== ", round(dre_bfgs) , " ========= gain " , \
                round(lastCostDRE_bfgs - dre_bfgs), "==== f(v)= ", round(primal_cost_v), " f(u)= ", round(primal_cost_u), " ~= ", currentCost_bfgs)

            # accept / reject, reject all but drs and see
            # if ls_it == line_search_iterations-1 :
            if dre_bfgs <= lastCostDRE_bfgs or ls_it == line_search_iterations-1 : # not correct yet, must be <= last - c/gamma |u-v|
                for ci in range(kClusters):
                    landmark_s_in_cluster[ci] = landmark_s_in_cluster_bfgs[ci].copy()
                    points_3d_in_cluster[ci] = points_3d_in_cluster_bfgs[ci].copy()
                    Vl_in_cluster_bfgs[ci] = Vl_in_cluster_bfgs[ci].copy()
                L_in_cluster = L_in_cluster_bfgs.copy()
                landmark_v = landmark_v_bfgs.copy()
                lastCostDRE_bfgs = dre_bfgs.copy()
                x0_p = x0_p_bfgs.copy()
                #print("A landmark_s_in_cluster", landmark_s_in_cluster)
                break
# here bfgs is better, but dre has better cost for the drs solution.

# either adjust points_3d_in_cluster -> copy over output.
# or let output as input to ba -- and Lfk
# descent lemma for landmarks only
# f(x) <= f(y) + <nabla f(y), x-y> + Lf/2 |x-y|^2
# e.g. update from y to x, nabla f(y) Jac_l * res(y)
# x=y+delta, [f(x) - f(y) - b^T delta] / delta^2 <= Lf/2, defines Lf
# trust region
#        fx0_new = fx0 + (J_pose * delta_p + J_land * delta_l) = fx0 + J^t H^-1 nabla fx0
#        costQuad = np.sum(fx0_new**2)
#        tr_check = (costStart - costEnd) / (costStart - costQuad)
#
# TODO:
# maybe estimate locally per lm -> change in f(x) from LOCAL residuums. x->y. gradient from nabla -> get local Lfk per lm!
# will fix close ones more strongly.
# xl -> xl + delta == sum of residduums^2 of that landmark involved. Fix cam, use new gradient L* diag * delta!
# new residuum f(x) is fast.
#
# Then use this for fixing landmarks / updating. enven use VLi / Vl instead? argmin_x sum_y=lm_in_cluster (x-y) VL (x-y) of last Vl.
# ==> x^t sum Vl x - 2 x sum (Vl y) + const =>  x = (sum Vl)^-1 [sum (Vl*y)].
# above should lead to better? solutions at least. At next local updates we work with Vl?
# 1. return last Vl. recall Vl * delta = nabla l is solved. or use L * diag Vl and return it (cheaper) or L * Vl ?
# return 3x3 matrix per lm.
#     f(x) < f(y) + <nabla fy , x-y> + (x-y)^ Vl (x-y). Vl is making this strongly convex by design. s.t. this descent lemma holds. Even by design.
# or  f(x) < f(y) + <nabla fy , x-y> + (x-y)^ JJl (x-y). New solution < old + penalty + <nabla fy, delta>
# <=> f(x) < f(y) + <nabla fy + nabla fx, x-y>
