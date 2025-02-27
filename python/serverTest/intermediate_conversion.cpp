pdate step size 
    Jp, Jl
    SetStepSize:
        JpJ -> stepsize
        BlockSqrt<9>(JpJ)

Solve
    cers solve -> fill cameras, landmarks
    return cost


def stop_criterion(delta, delta_i, i):
    # lower (1e-4) can be worse? maybe just the parts / how parts are.
    eps = 1e-3 #1e-2 used in paper, tune. might allow smaller as faster?
    return (i+1) * delta_i < eps * delta

def solveByGDNesterov(Ul, W, Vli, bS, m):
    Lip = 0.9 # 100 -> 1. # TODO: play, find out how to progress over time.
    lambda0 = (1.+np.sqrt(5.)) / 2. # l=0 g=1, 0, .. L0=1 g = 0,..

    Uli = blockInverse(Ul, 9)
    ubs = - Uli * bS
    xk = - ubs
    y0 = - ubs

    verbose = False
    if verbose:
        costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
        print("-1 gd cost ", costk)

    for it__ in range(m):
        lambda1 = (1 + np.sqrt(1 + 4 * lambda0**2)) / 2
        gamma = (1-lambda0) / lambda1
        lambda0 = lambda1

        #( I - Uli * W * Vli * W.transpose())
        g = xk - Uli * ( W * (Vli * (W.transpose() * xk))) + ubs
        yk = xk - 1/Lip * g
        xk = (1-gamma) * yk + gamma * y0
        y0 = yk

        if verbose:
            # eq is Ul [I - Uli * W * Vli * W.transpose()] x = b
            costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
            print(it__, " gd cost ", costk)

        if stop_criterion(np.linalg.norm(xk, 2), np.linalg.norm(1/Lip * g, 2), it__):
            return xk, it__
    return xk, it__

# each new L, need access to:
# JtJ, JlTJl, Jl and Jp diag: (1+L) * X + L * XDiag
# compute VLi
# stepSize once for all L. Also returned & stored.

# compute cost start and Jp, Jl, gradient, using ceres
# in solve: store Vl, Ul, diag(Jl / Jp)
# compute stepSize and store
# prox penalty can be compute from cam, cams stepsize
# compute VLi, W, bp bl (== gradient)
# conmpute bS solve for deltas. 
# TR check -> update cam landmarks with deltas or new L and redo.

// Not sure if this copies or not.
void Diagonal(SparseMatrix<double, Eigen::RowMajor>& mat) {
    DiagonalMatrix<double> diag = mat.diagonal().asDiagonal(); // ?
    // diag.diagonal() = mat.diagonal(); // copy?
#ifdef _const_diag_
    for (int b = 0; b < numCameras; ++b) { // block
      double mv = diag(9*b);
      for (int id = 1; id < 9; ++id) {
        mv = std::max(mv, diag(9*b + id));
      }
      for (int id = 0; id < 9; ++id) {
        diag(9*b + id) += be * mv;
      }
    }
#else
    //diag.diagonal().array() *= (1. + be * scale);
#endif
    //diag.diagonal().array() += 1e-18; // TODO: this is not good.
}

void UpdatePreconditioningCameras(SparseMatrix<double, Eigen::RowMajor> JpJ) {
    full_stepSize.resize(stepSize.size(), 0);
    const double *values = JpJ.valuePtr();
    std::copy(values, values + full_stepSize.size(), full_stepSize.data());
    // ToDo: Is this ok or an issue to be resolved differently?
    for (int b = 0; b < numCameras; ++b) {
      for(int id = 0; id < 81; id += 10) { // diagonal entries !?
        full_stepSize[81*b + id] = std::max(1e-36, full_stepSize[81*b + id]);
      }
    }
}

// StepSize as matrix is needed for multiplication with vectors. Still should be easy as blockMult std vector with other vector.
template<int N>
std::vector<double> blockMult(const std::vector<double>& blockMat, const std::vector<double>& vec) {
    std::vector<double> res(blockMat.size() / N, 0);
    for (int id = 0; id < blockMat.size(); ++id) {
        res[id / N] += blockMat[id] * vec[id / N];
    }
    return res;
}

bool stop_criterion(double delta, double delta_i, int i) {
    // lower (1e-4) can be worse? maybe just the parts / how parts are.
    const double eps = 1e-3 //#1e-2 used in paper, tune. might allow smaller as faster?
    return (i+1) * delta_i < eps * delta;
}

std::pair<Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::Matrix<double, Eigen::Dynamic, 1>>
SolveByGDNesterov(const SparseMatrix<double, Eigen::RowMajor>& Ul, const SparseMatrix<double, Eigen::RowMajor>& Vl, 
                  const SparseMatrix<double, Eigen::RowMajor>& Jp, const SparseMatrix<double, Eigen::RowMajor>& Jl, 
                  const Eigen::Matrix<double, Eigen::Dynamic, 1>& res, int power_iterations) {
    // compute bS, Vli, W
    const double Lip = 0.9;
    const double lambda0 = (1. + std::sqrt(5.)) / 2.;
    const auto Vli = BlockInverse<3>(Vl);
    const auto Uli = BlockInverse<9>(Ul);
    const SparseMatrix<double, Eigen::RowMajor> W = Jp.transpose() * Jl;
    Eigen::Matrix<double, Eigen::Dynamic, 1> bS;
    // bS = (bp + stepSize * prox_rhs - W * Vli * bl).flatten() # see XX equals 2 * (bp - W * Vli * bl)
    bS = Jp.transpose() * res;
    bS = bS + Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> >(blockMult(full_stepSize, cameras).data());
    bS = bS - Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> >(blockMult(full_stepSize, cameras_s).data());
    bS = bS - W * (Vli * (Jl.transpose() * res));
    
    Eigen::Matrix<double, Eigen::Dynamic, 1> ubs = -Uli * bS;
    Eigen::Matrix<double, Eigen::Dynamic, 1> xk = bS;
    Eigen::Matrix<double, Eigen::Dynamic, 1> y0 = bS;
    // Lip = 0.9 # 100 -> 1. # TODO: play, find out how to progress over time.
    // lambda0 = (1.+np.sqrt(5.)) / 2. # l=0 g=1, 0, .. L0=1 g = 0,..

    for(int i=0;i<power_iterations; ++i) {
        const double lambda1 = 1. + std::sqrt(1. + 4. * lambda0**2) / 2.;
        const double gamma = (1.-lambda0) / lambda1;
        lambda0 = lambda1;

        //     g = xk - Uli * ( W * (Vli * (W.transpose() * xk))) + ubs
        //     yk = xk - 1/Lip * g
        //     xk = (1-gamma) * yk + gamma * y0
        //     y0 = yk
        const Eigen::Matrix<double, Eigen::Dynamic, 1> g = xk - Uli * (W * (Vli * (W.transpose() * xk))) + ubs;
        const Eigen::Matrix<double, Eigen::Dynamic, 1> yk = xk - 1. / Lip * g;
        xk = (1. - gamma) * yk + gamma * y0;
        y0 = yk;

        if(stop_criterion(xk.norm(), 1. / Lip * g.norm(), i)) { // array().real().norm();?
            break;
        }
    }
    delta_p = -delta_p
    Eigen::Matrix<double, Eigen::Dynamic, 1> delta_l = Vli * ((W.transpose() * xk).flatten() + bl);
    return {-xk, delta_l};
}

void UpdateStepSizeAndSolve() { // Recompute.
    const auto [Jp, Jl, res] = GetJacobianAndResidual(); // also return sorted! residuals.
    SparseMatrix<double, Eigen::RowMajor> Vl(3 * numLandmarks, 3 * numLandmarks);
    Vl.reserve(VectorXi::Constant(3 * numLandmarks, 3));
    Vl = Jl.transpose() * Jl;
    if (firstIteration) { // preconditioning
        const auto diag = JlJ.diagonal().array().cwiseAbs().cwiseSqrt().cwiseMax(1e-10);
        THROW_IF(diag.size() != vnorm.size());
        std::cout << " Update vnorm " << cluster_id << " " << diag.size() << " == " << vnorm.size() << "\n";
        for (int id = 0; id < vnorm.size(); ++id) {
            landmarks[id] *= diag[id];
            vnorm[id] = 1. / diag(id);
        }
    }
    DiagonalMatrix<double> diagVL = Diagonal(Vl); // Vl = VL + L * diagVL

    // JpJ, StepSize, diag JpJ
    if(Jp.nonZeros() != 9 * Jp.rows())
        std::cout << "Jp " << cluster_id << " | " << Jp.nonZeros() << " =? " << Jp.rows() * 9 << "\n";
    THROW_IF(Jp.nonZeros() != 9 * Jp.rows());
    SparseMatrix<double, Eigen::RowMajor> Ul(9 * numCameras, 9 * numCameras);
    Ul.reserve(VectorXi::Constant(9 * numCameras, 9));
    Ul = Jp.transpose() * Jp;
    if (firstIteration) { // also handled setting be = 0 in 1st step.
        UpdatePreconditioningCameras(Ul);
    }
    DiagonalMatrix<double> diagUP = Diagonal(Ul); // Vp = Vp + L * diagVp

    const double scale = 1e-1; // 1e0: @29: 501k, no jump. 1e1 many jumps. 473k
    Ul += scale * Ul + current_be * diagUP;
    if (firstIteration) { // also handled setting be = 0 in 1st step.
        const double* values = Ul.valuePtr();
        std::copy(values, values + full_stepSize.size(), full_stepSize.data());
    }

    // Loop until ok or adjust tr_region
    double tr_radius = std::min(max_trust_region_radius, summary.iterations.back().trust_region_radius);
    double old_tr_radius = 0;

    Eigen::Matrix<double, Eigen::Dynamic, 1> res = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> >(residuals.data());
    const int power_iterations = 100;
    const double costStart = res.squared_norm();

    while ( true ) {
        Ul += (1 ./ tr_radius - 1 ./ old_tr_radius) * diagUP;
        Vl += (1 ./ tr_radius - 1 ./ old_tr_radius) * diagVL;
        old_tr_radius = tr_radius;

        const auto [delta_p, delta_l] = SolveByGDNesterov(Ul, Vl, Jp, Jl, res, power_iterations);
        // compute cost / tr_check
        //fx0_new = fx0 + (J_pose * delta_p + J_land * delta_l)
        const double costQuad = (res + Jp * delta_p + J_l * delta_l).squared_norm();
        penaltyStart = prox_rhs.dot(stepSize * prox_rhs) # fixed per run
        
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> >(blockMult(full_stepSize, cameras).data());

        std::vector<double> temp(9 * num_cameras, 0.); // same size as camera vector
        Eigen::Matrix<double, Eigen::Dynamic, 1> prox_rhs = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> > (temp.data());
        prox_rhs = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> > (cameras.data()) - Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> > (cameras_s.data()); 
        const double penaltyStart = prox_rhs.dot( Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> >(blockMult(full_stepSize, temp)) );
        prox_rhs += delta_p;
        const double penaltyEnd = prox_rhs.dot( Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> >(blockMult(full_stepSize, temp)) );


        for (int id = 0; id < delta_p.size(); ++id) {
            cameras[id] += delta_p[id];
        }
        for (int id = 0; id < delta_l.size(); ++id) {
            landmarks[id] += delta_l[id];
        }
        const double costEnd = 2 * GetCost(); // demands cameras , landmarks already updated.

        const double tr_check = (costStart - costEnd + penaltyStart - penaltyEnd) / std::max(0.1, costStart - costQuad + penaltyStart - penaltyEnd);

        if(tr_check < 0.25) {
            for (int id = 0; id < delta_p.size(); ++id) {
                cameras[id] -= delta_p[id];
            }
            for (int id = 0; id < delta_l.size(); ++id) {
                landmarks[id] -= delta_l[id];
            }
            tr_check /= 2;
        }
        if(tr_check > 0.25) {
            if(tr_check > 0.8) { 
                tr_check *= 2;
            }
            break;
        }
    }
}

// CUDA_VISIBLE_DEVICES={GPU ID} python demo/inference_on_a_image.py -c groundingdino/config/GroundingDINO_SwinT_OGC.py -p weights/groundingdino_swint_ogc.pth -i cars.jpg -o ./output -t "car"

# 1. JltJl, Jtl. diag(JltJl), diag(JtJ), => Vl & diag JL
# 2.                                        stepSize and Ul and diag(Jp)
# 3. bp, bl, W
JltJlDiag = JltJl + maxDiagA(JltJl, 3) # need to adjust tr. is JltJl part needed? 
Vl = JltJl + L * JltJlDiag # Vl new = Vl + (Lnew - Lold) * JlJ Diag
Vli = blockInverse(Vl, 3) # anyway for each L need to invert. store JlJ and Vli.

prox_rhs = x0_p_ - s_p_
costStart = np.sum(fx0**2) # once

W = J_pose.transpose() * J_land
bp = J_pose.transpose() * fx0
bl = J_land.transpose() * fx0
JtJ = J_pose.transpose() * J_pose

blockEigenvalueJtJ = 1e1 * maxDiagA(JtJ, 9)
stepSize = LipJ_ * 1e-1 * JtJ.copy() + blockEigMult * blockEigenvalueJtJ
stepSize = mult_selected_blocks(stepSize, poses_only_in_cluster_, 1e-6, 9)

JtJDiag = JtJ.copy() + blockEigMultJtJ * blockEigenvalueJtJ

Ul = JtJ + L * JtJDiag + stepSize # new Ul = Ul + (Lnew-Lold) * JtJDiag
penaltyStart = prox_rhs.dot(stepSize * prox_rhs) # fixed per run

# could compute bs intern in solveNesterov. Also Vli, returns delta l/p
# In: bp, bl, W (check sparsity to Jl, Jp), Vl,Ul, rest stored in class.
bS = (bp + stepSize * prox_rhs - W * Vli * bl).flatten() # see XX equals 2 * (bp - W * Vli * bl)
delta_p, powerits_run = solveByGDNesterov(Ul, W, Vli, bS, powerits)
delta_p = -delta_p
delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl)
# out: delta p,l 
#-------------

#penaltyL = L * (delta_l).dot(JltJlDiag * delta_l) # not needed for basic
#penaltyP = L * delta_p.dot(JtJDiag * delta_p) + (delta_p + prox_rhs).dot(stepSize * (delta_p + prox_rhs))

# |fx0 + J * delta|^2 = fx0^2 + delta J^TJ * delta + 2 fx0^T J * delta, store fx0^T J = nabla f = bp,bl.
fx0_new = fx0 + (J_pose * delta_p + J_land * delta_l)
costQuad = np.sum(fx0_new**2)

# update and compute cost
costEnd = np.sum(fx1.numpy() ** 2) 

penaltyP = (delta_p + prox_rhs).dot(stepSize * (delta_p + prox_rhs))
tr_check = (costStart - costEnd + penaltyStart - penaltyP) / np.maximum(0.1, costStart - costQuad + penaltyStart - penaltyP)

If tr_check ok 
x0_p_ = x0_p_ + delta_p
x0_l_ = x0_l_ + delta_l
else repeat

Update step size 
    Jp, Jl
    SetStepSize:
        JpJ -> stepsize
        BlockSqrt<9>(JpJ)

Solve
    cers solve -> fill cameras, landmarks
    return cost


def stop_criterion(delta, delta_i, i):
    # lower (1e-4) can be worse? maybe just the parts / how parts are.
    eps = 1e-3 #1e-2 used in paper, tune. might allow smaller as faster?
    return (i+1) * delta_i < eps * delta

def solveByGDNesterov(Ul, W, Vli, bS, m):
    Lip = 0.9 # 100 -> 1. # TODO: play, find out how to progress over time.
    lambda0 = (1.+np.sqrt(5.)) / 2. # l=0 g=1, 0, .. L0=1 g = 0,..

    Uli = blockInverse(Ul, 9)
    ubs = - Uli * bS
    xk = - ubs
    y0 = - ubs

    verbose = False
    if verbose:
        costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
        print("-1 gd cost ", costk)

    for it__ in range(m):
        lambda1 = (1 + np.sqrt(1 + 4 * lambda0**2)) / 2
        gamma = (1-lambda0) / lambda1
        lambda0 = lambda1

        #( I - Uli * W * Vli * W.transpose())
        g = xk - Uli * ( W * (Vli * (W.transpose() * xk))) + ubs
        yk = xk - 1/Lip * g
        xk = (1-gamma) * yk + gamma * y0
        y0 = yk

        if verbose:
            # eq is Ul [I - Uli * W * Vli * W.transpose()] x = b
            costk = xk.dot(Ul * xk - W * (Vli * (W.transpose() * xk)) - 2 * bS)
            print(it__, " gd cost ", costk)

        if stop_criterion(np.linalg.norm(xk, 2), np.linalg.norm(1/Lip * g, 2), it__):
            return xk, it__
    return xk, it__

# each new L, need access to:
# JtJ, JlTJl, Jl and Jp diag: (1+L) * X + L * XDiag
# compute VLi
# stepSize once for all L. Also returned & stored.

# compute cost start and Jp, Jl, gradient, using ceres
# in solve: store Vl, Ul, diag(Jl / Jp)
# compute stepSize and store
# prox penalty can be compute from cam, cams stepsize
# compute VLi, W, bp bl (== gradient)
# conmpute bS solve for deltas. 
# TR check -> update cam landmarks with deltas or new L and redo.


# 1. JltJl, Jtl. diag(JltJl), diag(JtJ), => Vl & diag JL
# 2.                                        stepSize and Ul and diag(Jp)
# 3. bp, bl, W
JltJlDiag = JltJl + maxDiagA(JltJl, 3) # need to adjust tr. is JltJl part needed? 
Vl = JltJl + L * JltJlDiag # Vl new = Vl + (Lnew - Lold) * JlJ Diag
Vli = blockInverse(Vl, 3) # anyway for each L need to invert. store JlJ and Vli.

prox_rhs = x0_p_ - s_p_
costStart = np.sum(fx0**2) # once

W = J_pose.transpose() * J_land
bp = J_pose.transpose() * fx0
bl = J_land.transpose() * fx0
JtJ = J_pose.transpose() * J_pose

blockEigenvalueJtJ = 1e1 * maxDiagA(JtJ, 9)
stepSize = LipJ_ * 1e-1 * JtJ.copy() + blockEigMult * blockEigenvalueJtJ
stepSize = mult_selected_blocks(stepSize, poses_only_in_cluster_, 1e-6, 9)

JtJDiag = JtJ.copy() + blockEigMultJtJ * blockEigenvalueJtJ

Ul = JtJ + L * JtJDiag + stepSize # new Ul = Ul + (Lnew-Lold) * JtJDiag
penaltyStart = prox_rhs.dot(stepSize * prox_rhs) # fixed per run

# could compute bs intern in solveNesterov. Also Vli, returns delta l/p
# In: bp, bl, W (check sparsity to Jl, Jp), Vl,Ul, rest stored in class.
bS = (bp + stepSize * prox_rhs - W * Vli * bl).flatten() # see XX equals 2 * (bp - W * Vli * bl)
delta_p, powerits_run = solveByGDNesterov(Ul, W, Vli, bS, powerits)
delta_p = -delta_p
delta_l = -Vli * ((W.transpose() * delta_p).flatten() + bl)
# out: delta p,l 
#-------------

#penaltyL = L * (delta_l).dot(JltJlDiag * delta_l) # not needed for basic
#penaltyP = L * delta_p.dot(JtJDiag * delta_p) + (delta_p + prox_rhs).dot(stepSize * (delta_p + prox_rhs))

# |fx0 + J * delta|^2 = fx0^2 + delta J^TJ * delta + 2 fx0^T J * delta, store fx0^T J = nabla f = bp,bl.
fx0_new = fx0 + (J_pose * delta_p + J_land * delta_l)
costQuad = np.sum(fx0_new**2)

# update and compute cost
costEnd = np.sum(fx1.numpy() ** 2) 

penaltyP = (delta_p + prox_rhs).dot(stepSize * (delta_p + prox_rhs))
tr_check = (costStart - costEnd + penaltyStart - penaltyP) / np.maximum(0.1, costStart - costQuad + penaltyStart - penaltyP)

If tr_check ok 
x0_p_ = x0_p_ + delta_p
x0_l_ = x0_l_ + delta_l
else repeat
