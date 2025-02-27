// #define _ceres_num_threads_ 1
// #define __unweighted_system__
#define _num_threads_machine_ 31

#include <zmq.hpp>
#include <string>
#include <iostream>
#include <thread>
#include <mutex>
#include <omp.h>
//#include <chrono>
#include "generated/proto/test.pb.h"
#include <google/protobuf/message_lite.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 

#include "ceres/ceres.h"
//#include "ceres/normal_prior.h"
#include "ceres/rotation.h"

using Eigen::SparseMatrix;
using Eigen::VectorXi;

#ifdef _WIN32
#include<Windows.h>
#elif defined __unix__
#include <unistd.h>
#endif

#define THROW_IF(cond)                                                    \
  do {                                                                    \
    if (cond) [[unlikely]] {                                              \
      std::ostringstream error;                                           \
      error << "Check failed in " << __FILE__ << ":" << __LINE__ << " - " \
            << #cond;                                                     \
      throw std::runtime_error(error.str());                              \
    }                                                                     \
  } while (false)

template <typename Proto>
Proto ParseProto(const std::string& serialized) {
  Proto proto;
  THROW_IF(!proto.ParseFromString(serialized));
  return proto;
}

void delay(int msec)
{
#ifdef _WIN32
    Sleep(msec);
#elif defined __unix__
    sleep(msec);
#endif
}

struct ProxStepPrior {
    ProxStepPrior() {}
    // ProxStepPrior(int block_): block(block_) {}
    // |sqrt(S) * (x-s) |^2 = (x-s)^T S (x-s)
    template <typename T>
    bool operator()(const T *const matBlock,
                    const T *const camera,
                    const T *const camera_s,
                    T *residuals) const {
         Eigen::Matrix<T, 9, 1> d = Eigen::Map<const Eigen::Matrix<T, 9, 1>>(camera) - Eigen::Map<const Eigen::Matrix<T, 9, 1>>(camera_s);
        Eigen::Map<Eigen::Matrix<T, 9, 1>> residualsVector(residuals);
        residualsVector = Eigen::Map<const Eigen::Matrix<T, 9, 9>>(matBlock) * d;
        return true;
    }

    // Factory to hide the construction of the CostFunction object from the client code.
    static ceres::CostFunction *Create() {
        return (new ceres::AutoDiffCostFunction<ProxStepPrior, 9, 81, 9, 9>(new ProxStepPrior()));
    }
};
#ifdef __unweighted_system__
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = -p[0] / p[2]; // that means focal length is flipped.
    T yp = -p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    T r2 = xp * xp + yp * yp;
    T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T& focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
        new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};
#else
struct SnavelyReprojectionErrorWeighted {
    SnavelyReprojectionErrorWeighted(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    const T* const cameraWeight,
                    const T* const pointWeight,
                    T* residuals) const {

      T cameraW[9];
      cameraW[0] = camera[0] * cameraWeight[0];
      cameraW[1] = camera[1] * cameraWeight[1];
      cameraW[2] = camera[2] * cameraWeight[2];
      cameraW[3] = camera[3] * cameraWeight[3];
      cameraW[4] = camera[4] * cameraWeight[4];
      cameraW[5] = camera[5] * cameraWeight[5];
      cameraW[6] = camera[6] * cameraWeight[6];
      cameraW[7] = camera[7] * cameraWeight[7];
      cameraW[8] = camera[8] * cameraWeight[8];
      T pointW[3];
      pointW[0] = point[0] * pointWeight[0];
      pointW[1] = point[1] * pointWeight[1];
      pointW[2] = point[2] * pointWeight[2];
      // camera[0,1,2] are the angle-axis rotation.
      T p[3];
      ceres::AngleAxisRotatePoint(cameraW, pointW, p);

      // camera[3,4,5] are the translation.
      p[0] += cameraW[3];
      p[1] += cameraW[4];
      p[2] += cameraW[5];

      // Compute the center of distortion. The sign change comes from
      // the camera model that Noah Snavely's Bundler assumes, whereby
      // the camera coordinate system has a negative z axis.
      T xp = -p[0] / p[2]; // that means focal length is flipped.
      T yp = -p[1] / p[2];

      // Apply second and fourth order radial distortion.
      const T& l1 = cameraW[7];
      const T& l2 = cameraW[8];
      T r2 = xp * xp + yp * yp;
      T distortion = 1.0 + r2 * (l1 + l2 * r2);

      // Compute final projected point position.
      const T& focal = cameraW[6];
      T predicted_x = focal * distortion * xp;
      T predicted_y = focal * distortion * yp;

      // The error is the difference between the predicted and observed position.
      residuals[0] = predicted_x - observed_x;
      residuals[1] = predicted_y - observed_y;

      return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y) {
      return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorWeighted, 2, 9, 3, 9, 3>(
          new SnavelyReprojectionErrorWeighted(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;
  };
#endif

template<int N>
void BlockSqrt(SparseMatrix<double, Eigen::RowMajor>& mat) {
    const int numrows = mat.rows();
    const int numNonZeros = mat.nonZeros();
    if(numNonZeros != mat.rows() * N)
        std::cout << "BlockSqrt " << mat.nonZeros() << " ?= " << numrows * N << "\n";
    THROW_IF(numrows != mat.cols());
    THROW_IF(mat.nonZeros() != numrows * N);
    double* values = mat.valuePtr();
    //std::cout << "before  "<< values[0]<< " " << values[1]<< " " << values[2]<< " " << values[3] << "\n";
#pragma omp parallel for num_threads(options.num_threads)
    for (int i = 0; i < numrows / N; i++) {
        auto mat9x9 = Eigen::Map< Eigen::Matrix<double, N, N> > (&(values[i * N*N]));//,  Eigen::Stride<0, 0>);
        //std::cout << "before "<< mat9x9 << " \n";

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,N,N> > eigensolver;
        eigensolver.computeDirect(mat9x9, Eigen::DecompositionOptions::ComputeEigenvectors);
        //VPQ_EXPECT_EQ(eigensolver.info(), Eigen::Success);

        // SqrtCovEigenValues are sorted in decreasing order.
        const Eigen::Vector<double, N> sqrtEigenValues = eigensolver.eigenvalues().cwiseAbs().cwiseSqrt().cwiseMax(1e-16);//.cwiseMax(lowerBoundSquared).cwiseSqrt().cwiseInverse();
        // recall : i had here min ev >= 1e-6 * maxEv. Could return a diag matrix
        mat9x9 = eigensolver.eigenvectors() * sqrtEigenValues.asDiagonal() * eigensolver.eigenvectors().transpose();
        
        //auto mat9x9_out = Eigen::Map< Eigen::Matrix<double,N,N> > (&(values[i * N*N]));//,  Eigen::Stride<0, 0>);
        //std::cout << "after  "<< mat9x9_out.transpose() * mat9x9_out << " \n";
    }
    //std::cout << "after  "<< values[0]<< " " << values[1]<< " " << values[2]<< " " << values[3] << "\n";
}

template<int N>
void BlockInverse(SparseMatrix<double, Eigen::RowMajor>& mat) {
    const int numrows = mat.rows();
    THROW_IF(mat.rows() != mat.cols());
    //THROW_IF(mat.);
    double* values = mat.valuePtr();
    //std::cout << "before  "<< values[0]<< " " << values[1]<< " " << values[2]<< " " << values[3] << "\n";
#pragma omp parallel for num_threads(options.num_threads)
    for (int i = 0; i < numrows / N; i++) {
        auto matNxN = Eigen::Map< Eigen::Matrix<double,N,N> > (&(values[i * N*N]));//,  Eigen::Stride<0, 0>);
        //std::cout << "before "<< matNxN << " \n";
        matNxN = matNxN.inverse().eval();
        //std::cout << "after "<< matNxN << " \n";        
    }
    //std::cout << "after  "<< values[0]<< " " << values[1]<< " " << values[2]<< " " << values[3] << "\n";
}

void WriteJacobian(ceres::Problem& problem, int numCameras, int numLandmarks) {
    std::cout << "Write Jac\n";
    ceres::Problem::EvaluateOptions evalOptions;
    evalOptions.apply_loss_function = true;
    evalOptions.num_threads = 1;

    ceres::CRSMatrix jacobian; // likely unordered as shit.
    problem.Evaluate(evalOptions, nullptr, nullptr, nullptr, &jacobian);
    const size_t numUnknowns = jacobian.num_cols;

    // 12 per row, 9 cams, 3 landmark indices. problem: order is as follows by cam_id and lm_id.
    // J_pose is given by going over jac and id. 
    // J_pose is n_res x 9 * # cams
    // J_land is n_res x 3 * # land
    for (Eigen::Index r = 0; r < 100; ++ r) { //jacobian.num_rows; ++r) {
        std::cout << r << ": ";
        for (Eigen::Index idx = jacobian.rows[r]; idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];
            ++idx) {
        const Eigen::Index c = jacobian.cols[idx];
        std::cout << c << ", ";// << " = " << jacobian.values[idx] << " | ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Not sure if this copies or not.
template<int N>
Eigen::DiagonalMatrix<double, Eigen::Dynamic>
Diagonal(SparseMatrix<double, Eigen::RowMajor>& mat) {
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> diag = mat.diagonal().asDiagonal(); // ?
//#define _const_diag_
#ifdef _const_diag_
  auto diagdiag = diag.diagonal();
  for (int b = 0; b < mat.rows() / N; ++b) { // block
    double mv = diagdiag(N*b);
    for (int id = 1; id < N; ++id) {
      mv = std::max(mv, diagdiag(N*b + id));
    }
    for (int id = 0; id < N; ++id) {
        diagdiag(N*b + id) = mv;
    }
  }
#else
  //diag.diagonal().array() *= (1. + be * scale);
#endif
  // diag.diagonal().array() += 1e-18; // TODO: this is not good.
  // std::cout << diag.diagonal() << "\n";
  return diag;
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
  const double eps = 1e-3; //#1e-2 used in paper, tune. might allow smaller as faster?
  return (i+1) * delta_i < eps * delta;
}

// Ensure we can have a vector of programs. by id. maybe just a map : clusterid-> program.
///////////////////////////////////////////////////////
class CeresProgram {
public:
    CeresProgram() {
      Init(1);
    }

    int ClusterId() { return cluster_id; }

    void ResetProgram(const program_proto &pro) {
      Init(pro.num_clusters());
      numCameras = pro.cameras_size() / 9;
      numLandmarks = pro.landmarks_size() / 3;
      numResiduals =  pro.observations_size() / 2;
      //std::cout << numCameras << " " << numLandmarks << "\n";
      cameras.clear();
      cameras.reserve(9 * numCameras);
      for (const auto &v : pro.cameras()) {
        cameras.push_back(v);
      }
      //std::cout << "cameras.push_back\n";
      landmarks.clear();
      landmarks.reserve(3 * numLandmarks);
      for (const auto &v : pro.landmarks()) {
        landmarks.push_back(v);
      }
      last_landmarks = landmarks;
      //std::cout << "landmarks.push_back\n";
      cameras_s.clear();
      cameras_s.reserve(9 * numCameras);
      for (const auto &v : pro.cameras()) {
        cameras_s.push_back(v);
      }
      //std::cout << "cameras_s.push_back\n";
      stepSize.clear(); // all 0 to ensure jacobian is reasonable.
      stepSize.resize(81 * numCameras, 0);

      lm_obs.clear();
      lm_obs.reserve(pro.lm_id_size());
      for (const int &id : pro.lm_id()) {
        lm_obs.push_back(id);
      }
      //std::cout << "lm_obs.push_back\n";
      cam_obs.clear();
      cam_obs.reserve(pro.cam_id_size());
      for (const int &id : pro.cam_id()) {
        cam_obs.push_back(id);
      }
      unorm.clear();
      unorm.reserve(9 * numCameras);
      for (const auto &v : pro.unorm()) {
        unorm.push_back(v);
      }
      //std::cout << "cameras.push_back\n";
      vnorm.clear();
      vnorm.reserve(3 * numLandmarks);
      for (const auto &v : pro.vnorm()) {
        vnorm.push_back(v);
      }
      //std::cout << "data updated\n";

      options.max_num_iterations = std::max(0, std::min(10, pro.iterations()));
      options.initial_trust_region_radius = tr_radius;
      problem = ceres::Problem();
      cluster_id = pro.cluster_id();
      current_be = pro.be();
      start_be = current_be;
      std::cout << " be set to c_be " << current_be << " s_be:" << start_be<< "\n";
      //std::cout << "problem.AddParameterBlock\n";
      for (int i = 0; i < cameras.size(); i += 9) {
        problem.AddParameterBlock(&cameras[i], 9);
      }
      for (int i = 0; i < landmarks.size(); i += 3) {
        problem.AddParameterBlock(&landmarks[i], 3);
      }
      for (int i = 0; i < cameras_s.size(); i += 9) {
        problem.AddParameterBlock(&cameras_s[i], 9);
        problem.SetParameterBlockConstant(&cameras_s[i]);
      }
      std::cout << cluster_id << ". Parameter blocks added wo step c:" << cameras.size()
                << " s:" << cameras_s.size() << " l: " << landmarks.size()
                << " | " << numCameras << " " << numLandmarks << "\n";
      for (int i = 0; i < stepSize.size(); i += 81) {
        problem.AddParameterBlock(&stepSize[i], 81);
        problem.SetParameterBlockConstant(&stepSize[i]);
      }

      for (int i = 0; i < unorm.size(); i += 9) {
        problem.AddParameterBlock(&unorm[i], 9);
        problem.SetParameterBlockConstant(&unorm[i]);
      }
      for (int i = 0; i < vnorm.size(); i += 3) {
        problem.AddParameterBlock(&vnorm[i], 3);
        problem.SetParameterBlockConstant(&vnorm[i]);
      }
      //std::cout << "All Parameter blocks added\n";

#ifdef __unweighted_system__
      // SetParameterBlockVariable. optional set cameras constant, use for 's'.
      for (int i = 0; i < pro.observations_size() / 2; ++i) {
        // Each Residual block takes a point and a camera as input and outputs a
        // 2 dimensional residual. Internally, the cost function stores the
        // observed image location and compares the reprojection against the
        // observation.

        // *fMessage.mutable_samples() = {fData.begin(), fData.end()}; // copies? OMG

        // google::protobuf::RepeatedField<float> data(fData.begin(),
        // fData.end()); fMessage.mutable_samples()->Swap(&data); Parse 1 by 1
        // -- omg this is bad.
        ceres::CostFunction *cost_function = SnavelyReprojectionError::Create(
            pro.observations(2 * i + 0), pro.observations(2 * i + 1));
        problem.AddResidualBlock(cost_function, nullptr /* squared loss */,
                                 &(cameras[9 * pro.cam_id(i)]),
                                 &(landmarks[3 * pro.lm_id(i)]));
      }
#else
      for (int i = 0; i < pro.observations_size() / 2; ++i) {
        ceres::CostFunction *cost_function = SnavelyReprojectionErrorWeighted::Create(
            pro.observations(2 * i + 0), pro.observations(2 * i + 1));
        problem.AddResidualBlock(cost_function, nullptr /* squared loss */,
                                 &(cameras[9 * pro.cam_id(i)]),
                                 &(landmarks[3 * pro.lm_id(i)]),
                                 &(unorm[9 * pro.cam_id(i)]),
                                 &(vnorm[3 * pro.lm_id(i)]));
      }
#endif
      std::cout << "Added " << pro.observations_size() / 2 << " Residual blocks\n";
      function_residual_blocks.clear();
      problem.GetResidualBlocks(&function_residual_blocks);

      for (int cam_id = 0; cam_id < numCameras; ++cam_id) {
        // double* values = JpJ.valuePtr();
        //  ceres::Matrix block9x9 = Eigen::Map< Eigen::Matrix<double,9,9> >
        //  (&(stepSize[cam_id * 9*9])); ceres::Vector block9 = Eigen::Map<
        //  Eigen::Matrix<double,9,1> > (&(cameras_s[cam_id * 9]));

        // ceres::CostFunction* cost_function = new
        // ceres::AutoDiffCostFunction<ceres::NormalPrior, 9, 9>(new
        // ceres::NormalPrior(block9x9, block9));
        ceres::CostFunction *cost_function = ProxStepPrior::Create();
        // std::cout << "ptr " << &(stepSize [81 * cam_id]) << " " << &(cameras
        // [9 * cam_id]) << " " << &(cameras_s[9 * cam_id]) << "\n";
        problem.AddResidualBlock(
            cost_function, nullptr /* squared loss */,
            &(stepSize[81 * cam_id]), // it thinks this block size is 9, not 81.
            &(cameras[9 * cam_id]), &(cameras_s[9 * cam_id]));
      }
      std::cout << "Added " << numCameras << " Stepsize Residual blocks\n";
    }

    double GetCost() {
      // 1st get Jacobian(s):
      ceres::Problem::EvaluateOptions evalOptions;
      evalOptions.apply_loss_function = true;
      // evalOpt.parameter_blocks = {};
      evalOptions.residual_blocks = function_residual_blocks;
      evalOptions.num_threads = options.num_threads;
      std::vector<double> residuals;
      double cost;
      problem.Evaluate(evalOptions, &cost, &residuals, nullptr, nullptr);
      if (cost < best_cost) {
        best_landmarks = landmarks;
        best_cost = cost;
      }
      return cost;
    }

    //void SetBe(double be) { be = be; }
    
    return_cluster_proto FillReturnProto() {
      return_cluster_proto return_proto = return_cluster_proto();
      for (const double &v : cameras) {
        //return_proto.add_cameras(static_cast<float>(v));
        return_proto.add_cameras(v);
      }
      for (const double &v : landmarks) {
        //return_proto.add_landmarks(static_cast<float>(v));
        return_proto.add_landmarks(v);
      }
      for (const double &v : full_stepSize) {
        //return_proto.add_step_size(static_cast<float>(v));
        return_proto.add_step_size(v);
      }
      return_proto.set_cluster_id(cluster_id);
      return_proto.set_cost(cost);
      return return_proto;
    }

    double Solve() {
      ceres::Solver::Summary summary; // return?
      ceres::Solve(options, &problem, &summary);
      std::cout << summary.BriefReport() << "\n"; // .FullReport()
      // TODO: use this trust region size : store and reuse later.
      //std::vector<IterationSummary> Solver::Summary::iterations
      tr_radius = std::min(max_trust_region_radius, summary.iterations.back().trust_region_radius);
      // TODO: -ordering=user for schur (maybe cameras 1st then landmarks)
      cost = summary.final_cost * 2;
      std::cout << cluster_id << ". Update TR: " << tr_radius << ". be: " << current_be
                << ". Mycost: " << summary.final_cost * 2 << "\n";
      return cost;
    }

    void UpdateCameras(const cost_proto& costProto) {
      // std::cout << "Update cluster " << cluster_id << " update proto id:" << update.cluster_id() << "\n";
      THROW_IF(costProto.cameras_size() != cameras.size());
      THROW_IF(costProto.cluster_id() != cluster_id);

      int id = 0; // fill existing buffer
      for (const auto &v : costProto.cameras()) {
        cameras[id++] = v;
      }
    }

    void UpdateData(const prox_cluster_proto &update) {
      // std::cout << "Update cluster " << cluster_id << " update proto id:" << update.cluster_id() << "\n";
      THROW_IF(update.cameras_size() != cameras.size());
      THROW_IF(update.cameras_s_size() != cameras_s.size());
      THROW_IF(update.cluster_id() != cluster_id);

      int id = 0; // fill existing buffer
      for (const auto &v : update.cameras()) {
        cameras[id++] = v;
      }
      id = 0;
      for (const auto &v : update.cameras_s()) {
        cameras_s[id++] = v;
      }
      current_be = update.be();
      options.initial_trust_region_radius = tr_radius; // use from last solve.
      firstIteration = false;

      if (update.revert_lm() == 1) {
        std::cout << cluster_id << ". Revert landmarks\n";
        landmarks = last_landmarks;
      } else if (update.revert_lm() == 2) {
        std::cout << cluster_id << ". Revert landmarks to best cost lms\n";
        landmarks = best_landmarks; // hmm could be same as last_landmarks.
      }
      else {
        last_landmarks = landmarks;
      }
    }

    void UpdateStepSize() { // Recompute.
      const auto [Jp, Jl] = GetJacobian();
      if (firstIteration) {
        SparseMatrix<double, Eigen::RowMajor> JlJ(3 * numLandmarks, 3 * numLandmarks);
        JlJ.reserve(VectorXi::Constant(3 * numLandmarks, 3));
        JlJ = Jl.transpose() * Jl;
        const auto diag = JlJ.diagonal().array().cwiseAbs().cwiseSqrt().cwiseMax(1e-10);
        std::cout << " Update vnorm " << cluster_id << " " << diag.size() << " == " << vnorm.size() << "\n";
        THROW_IF(diag.size() != vnorm.size());
        for (int id = 0; id < vnorm.size(); ++id) {
          landmarks[id] *= diag[id];
          vnorm[id] = 1. / diag(id);
        }
      }
      stepSize.clear(); // all 0 to ensure jacobian is reasonable.
      stepSize.resize(81 * numCameras, 0);
      SetStepSize(Jp);
    }

    void UpdatePreconditioning(const preconditioning_proto& preconditioningProto) {
        // std::cout << "Update cluster " << cluster_id << " update proto id:" << update.cluster_id() << "\n";
        THROW_IF(preconditioningProto.cluster_id() != cluster_id);
        THROW_IF(preconditioningProto.unorm_size() != unorm.size());
        THROW_IF(preconditioningProto.vnorm_size() != vnorm.size());

        int id = 0; // fill existing buffer
        for (const auto &v : preconditioningProto.unorm()) {
            unorm[id++] = v;
        }
        // id = 0; // fill existing buffer
        // for (const float &v : preconditioningProto.vnorm()) {
        //     vnorm[id++] = v;
        // }
    }

////////////////////////////////////////
// new stuff for self optimization

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

std::pair<Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::Matrix<double, Eigen::Dynamic, 1>>
SolveByGDNesterov(SparseMatrix<double, Eigen::RowMajor> Uli, SparseMatrix<double, Eigen::RowMajor> Vli, 
                const SparseMatrix<double, Eigen::RowMajor>& Jp, const SparseMatrix<double, Eigen::RowMajor>& Jl, 
                const Eigen::Matrix<double, Eigen::Dynamic, 1>& res, int power_iterations) {
  // compute bS, Vli, W
  BlockInverse<3>(Vli);
  BlockInverse<9>(Uli);
  const double Lip = 0.9;
  double lambda0 = (1. + std::sqrt(5.)) / 2.;
  const SparseMatrix<double, Eigen::RowMajor> W = Jp.transpose() * Jl;
  Eigen::Matrix<double, Eigen::Dynamic, 1> bS;
  // bS = (bp_s                     - W * Vli * bl).flatten() # see XX equals 2 * (bp - W * Vli * bl)
  //       bp_s = bp + stepSize * prox_rhs
  // bS = (bp + stepSize * prox_rhs - W * Vli * bl).flatten() # see XX equals 2 * (bp - W * Vli * bl)
  bS = Jp.transpose() * res;
  bS = bS + Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> >(blockMult<9>(full_stepSize, cameras).data(), 9*numCameras);
  bS = bS - Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> >(blockMult<9>(full_stepSize, cameras_s).data(), 9*numCameras);
  bS = bS - W * (Vli * (Jl.transpose() * res));

  //std::cout << " Jl " << Jl.valuePtr()[0] << " " << Jl.valuePtr()[1] << " " << Jl.valuePtr()[2] << "\n";
//   std::cout << "bS :" << bS.array() << "\n";
//   std::cout << "bp :" << (Jp.transpose() * res).array() << "\n";
//   std::cout << "bl :" << (Jl.transpose() * res).array() << "\n";
  // std::cout << "res :" << res.array() << "\n"; //ok

  Eigen::Matrix<double, Eigen::Dynamic, 1> ubs = -Uli * bS;
  Eigen::Matrix<double, Eigen::Dynamic, 1> xk = - ubs;
  Eigen::Matrix<double, Eigen::Dynamic, 1> y0 = - ubs;
  // Lip = 0.9 # 100 -> 1. # TODO: play, find out how to progress over time.
  // lambda0 = (1.+np.sqrt(5.)) / 2. # l=0 g=1, 0, .. L0=1 g = 0,..

  std::cout << "xk :" << xk.squaredNorm() << "\n";

  for(int i = 0;i < power_iterations; ++i) {
      const double lambda1 = (1. + std::sqrt(1. + 4. * lambda0*lambda0)) / 2.;
      const double gamma = (1. - lambda0) / lambda1;
      lambda0 = lambda1;

      //     g = xk - Uli * ( W * (Vli * (W.transpose() * xk))) + ubs
      //     yk = xk - 1/Lip * g
      //     xk = (1-gamma) * yk + gamma * y0
      //     y0 = yk
      const Eigen::Matrix<double, Eigen::Dynamic, 1> g = xk - Uli * (W * (Vli * (W.transpose() * xk))) + ubs;
      const Eigen::Matrix<double, Eigen::Dynamic, 1> yk = xk - 1. / Lip * g;
      xk = (1. - gamma) * yk + gamma * y0;
      y0 = yk;

      //std::cout << i << ". xk :" << xk.squaredNorm() << "\n";

      if(stop_criterion(xk.norm(), 1. / Lip * g.norm(), i)) { // array().real().norm();?
          break;
      }
  }
  Eigen::Matrix<double, Eigen::Dynamic, 1> delta_l = Vli * ((W.transpose() * xk) - (Jl.transpose() * res));
  return {-xk, delta_l};
}

void UpdateStepSizeAndSolve() { // Recompute.
  auto [Jp, Jl, res] = GetJacobianAndResidual(); // also return sorted! residuals.
  SparseMatrix<double, Eigen::RowMajor> Vl(3 * numLandmarks, 3 * numLandmarks);
  Vl.reserve(VectorXi::Constant(3 * numLandmarks, 3));
  Vl = Jl.transpose() * Jl;
  if (firstIteration) { // preconditioning
      // diag is a reference .. why? i do stuff on it.
      const auto diag = Vl.diagonal().array().cwiseMax(1e-24).cwiseSqrt().cwiseInverse();
      THROW_IF(diag.size() != vnorm.size());
      std::cout << " Update vnorm " << cluster_id << " " << diag.size() << " == " << vnorm.size() << "\n";
      for (int id = 0; id < vnorm.size(); ++id) {
          landmarks[id] /= diag[id];
          vnorm[id] = diag(id);
      }
      // Update Vl as well.
      std::cout << " Jl " << Jl.valuePtr()[0] << " " << Jl.valuePtr()[1] << " " << Jl.valuePtr()[2] << "\n";
      Jl = (Jl * diag.matrix().asDiagonal()).eval(); // This does not happen as diag is diag of Vl. That gets changed. diag is not copied but reference.
      std::cout << " Jl " << Jl.valuePtr()[0] << " " << Jl.valuePtr()[1] << " " << Jl.valuePtr()[2] << "\n";

      Vl = diag.matrix().asDiagonal() * Vl * diag.matrix().asDiagonal();
  }
  const Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagVL = Diagonal<3>(Vl); // Vl = VL + L * diagVL

  // std::cout << " diagVL " << diagVL.diagonal() << "\n"; // 1's

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
  const Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagUP = Diagonal<9>(Ul); // Vp = Vp + L * diagVp

  const double scale = 1e-1; // 1e0: @29: 501k, no jump. 1e1 many jumps. 473k
  if (!firstIteration) { // also handled setting be = 0 in 1st step.
    SparseMatrix<double, Eigen::RowMajor> stepSize = scale * Ul;
    stepSize += diagUP * current_be;
    const double* values = stepSize.valuePtr();
    std::copy(values, values + full_stepSize.size(), full_stepSize.data());
    Ul += stepSize;
  } else {
    Ul += scale * Ul;
    Ul += diagUP * current_be;
  }
  // Loop until ok or adjust tr_region
  tr_radius = std::min(max_trust_region_radius, tr_radius);
  double inv_tr_radius = 0;

  const Eigen::Matrix<double, Eigen::Dynamic, 1> residual = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> >(res.data(), 2 * numResiduals);
  const int power_iterations = 100;
  const double costStart = residual.squaredNorm();
  std::cout << " coststart " << costStart << "\n";

  while ( true ) {
    //   std::cout << " diagUP " << Ul.diagonal()[0] << " " << Ul.diagonal()[1] << " " << Ul.diagonal()[2] << "\n";
    //   std::cout << " diagVL " << Vl.diagonal()[0] << " " << Vl.diagonal()[1] << " " << Vl.diagonal()[2] << "\n";// TOTALLY OFF after tr_check fails.
      Ul += (1. / tr_radius - inv_tr_radius) * (diagUP + Jp.transpose() * Jp);
      Vl += (1. / tr_radius - inv_tr_radius) * (diagVL + Jl.transpose() * Jl);
    //   std::cout << " diagUp " << Ul.diagonal()[0] << " " << Ul.diagonal()[1] << " " << Ul.diagonal()[2] << "\n";
    //   std::cout << " diagVL " << Vl.diagonal()[0] << " " << Vl.diagonal()[1] << " " << Vl.diagonal()[2] << "\n";
      
      //std::cout << " VL " << Vl.diagonal() << "\n";
      inv_tr_radius = 1. / tr_radius;

      const auto [delta_p, delta_l] = SolveByGDNesterov(Ul, Vl, Jp, Jl, residual, power_iterations);
      // compute cost / tr_check
      //fx0_new = fx0 + (J_pose * delta_p + J_land * delta_l)
      const double costQuad = (residual + Jp * delta_p + Jl * delta_l).squaredNorm();
      
      const double costQuad2 = (residual - Jp * delta_p - Jl * delta_l).squaredNorm();
      const double costQuad3 = (residual - Jp * delta_p + Jl * delta_l).squaredNorm();
      const double costQuad4 = (residual + Jp * delta_p - Jl * delta_l).squaredNorm();

        std::cout << costStart << " > " << costQuad << " " << costQuad2 << " " << costQuad3 << " " << costQuad4 << "\n";

      // std::cout << "res/dl/dp :" << residual.squaredNorm() << " " << delta_p.squaredNorm() << " " << delta_l.squaredNorm() << "\n";

      // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> >(blockMult<9>(full_stepSize, cameras).data());

      std::vector<double> temp(9 * numCameras, 0.); // same size as camera vector
      Eigen::Matrix<double, Eigen::Dynamic, 1> prox_rhs = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> > (temp.data(), 9 * numCameras);
      prox_rhs = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> > (cameras.data(), 9 * numCameras) -
                 Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> > (cameras_s.data(), 9 * numCameras);
      const double penaltyStart = prox_rhs.dot( Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> >(blockMult<9>(full_stepSize, temp).data(), 9 * numCameras) );
      prox_rhs += delta_p;
      const double penaltyEnd = prox_rhs.dot( Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> >(blockMult<9>(full_stepSize, temp).data(), 9 * numCameras) );

      for (int id = 0; id < delta_p.size(); ++id) {
          cameras[id] += delta_p[id];
      }
      for (int id = 0; id < delta_l.size(); ++id) {
          landmarks[id] += delta_l[id];
      }
      const double costEnd = 2 * GetCost(); // demands cameras , landmarks already updated.
      std::cout << " costs " << costStart << " " << costQuad << " " << costEnd << "\n";

      const double tr_check = (costStart - costEnd + penaltyStart - penaltyEnd) / std::max(0.1, costStart - costQuad + penaltyStart - penaltyEnd);
      std::cout << " tr_check " << tr_check << "\n";

      if(tr_check < 0.25) {
          for (int id = 0; id < delta_p.size(); ++id) {
              cameras[id] -= delta_p[id];
          }
          for (int id = 0; id < delta_l.size(); ++id) {
              landmarks[id] -= delta_l[id];
          }
          tr_radius /= 2;
          std::cout << "decrease TR radius " << tr_radius << "\n";
      }
      if(tr_check > 0.25) {
          if(tr_check > 0.8) {
            tr_radius = std::min(max_trust_region_radius, 1.5 * tr_radius);
            std::cout << "increase TR radius " << tr_radius << "\n";
          }
          break;
      }
  }
}


///////////////////////////////////

private:

    // Currently this is set 'stepsize' from Jp only.
    void SetStepSize(const SparseMatrix<double, Eigen::RowMajor> &Jp) {
      // std::cout << "Set step size " << cluster_id << "\n";
      if(Jp.nonZeros() != 9 * Jp.rows())
          std::cout << "Jp " << cluster_id << " | " << Jp.nonZeros() << " =? " << Jp.rows() * 9 << "\n";
      THROW_IF(Jp.nonZeros() != 9 * Jp.rows());
      SparseMatrix<double, Eigen::RowMajor> JpJ(9 * numCameras, 9 * numCameras);
      JpJ.reserve(VectorXi::Constant(9 * numCameras, 9));
      JpJ = Jp.transpose() * Jp;
      JpJ.makeCompressed();

      // SparseMatrix<double, Eigen::RowMajor> JlJ(3 * numLandmarks, 3 * numLandmarks);
      // JlJ.reserve(VectorXi::Constant(3 * numLandmarks, 3));
      // JlJ = Jl.transpose() * Jl;
      // auto JpJ_diag = JpJ.diagonal().array();
      if(JpJ.nonZeros() != stepSize.size() || JpJ.rows() * 9 != numCameras * 81)
        std::cout << "JpJ " << cluster_id << " | " << JpJ.nonZeros() << " =? " << JpJ.rows() * 9
                  << " " << stepSize.size() << " " << numCameras * 81 << "\n";
      THROW_IF(JpJ.nonZeros() != numCameras * 81);

      if (firstIteration) { // also handled setting be = 0 in 1st step.
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

      // Allow to scale JtJ as well? 
      const double scale = 1e1;
      // TODO.
      //const double scale = std::max(1. / 1.005, 1e1 * std::sqrt(start_be / current_be)); // 1e0: @29: 501k, no jump. 1e1 many jumps. 473k
      std::cout << "scale " << scale << " " << start_be << " " << current_be << "\n";
      JpJ = JpJ * (1. / scale); // optional to test. in theory should almost always suffice.
#define _const_diag_
#ifdef _const_diag_
      auto diag = JpJ.diagonal().array();
      for (int b = 0; b < numCameras; ++b) { // block
        double mv = diag(9*b);
        for (int id = 1; id < 9; ++id) {
          mv = std::max(mv, diag(9*b + id));
        }
        mv *= scale;
        for (int id = 0; id < 9; ++id) {
          diag(9*b + id) += current_be * mv;
        }
      }
#else
      JpJ.diagonal().array() *= (1. + current_be * scale);
#endif
      // JpJ.diagonal().array() += 1e-18; // TODO: this is not good.

      //JpJ.diagonal().array() *= (1. + be); //+= be * JpJ.diagonal().array();
      // JpJ = JpJ * 3; // optional to test. in theory should almost always suffice.
      //   const auto JpJDiagonal = JpJ.diagonal();//.array();
      //   JpJ = JpJ * 0.5 * 1e-12;
      //   //JpJ.diagonal() = JpJ.diagonal() + be * JpJDiagonal;
      //   JpJ.diagonal() = JpJDiagonal * (1. + be);
      //JpJ.diagonal() += be * JpJ.diagonal();
      //JpJ.diagonal().array().cwise

      if (!firstIteration) {
        full_stepSize.resize(stepSize.size(), 0);
        const double *values = JpJ.valuePtr();
        std::copy(values, values + full_stepSize.size(), full_stepSize.data());
      }

      // std::cout << "BlockSqrt " << cluster_id << "\n";
      BlockSqrt<9>(JpJ); // need templated fct.
      // instead reset variable block(s) JpJ and s to sqrt(Stepsize)
      const double* values = JpJ.valuePtr();
      std::copy(values, values + stepSize.size(), stepSize.data());
    }

    std::tuple<SparseMatrix<double, Eigen::RowMajor>,
               SparseMatrix<double, Eigen::RowMajor>, 
               std::vector<double>>
      GetJacobianAndResidual() {
      // 1st get Jacobian(s):
      ceres::Problem::EvaluateOptions evalOptions;
      evalOptions.apply_loss_function = true;
      // evalOpt.parameter_blocks = {}; // TODO only poses.
      evalOptions.residual_blocks = function_residual_blocks;
      evalOptions.num_threads = options.num_threads;
      ceres::CRSMatrix jacobian;
      std::vector<double> residuals;
      // std::cout << "GetJacobian: Evaluate " << cluster_id << "\n"; 
      problem.Evaluate(evalOptions, &startCost, &residuals, nullptr, &jacobian);
      const size_t numUnknowns = jacobian.num_cols;
      //   std::cout << "GetJacobian: " << cluster_id << " Finished eval problem "
      //             << jacobian.num_rows << "-" << 9 * numCameras << "\n";

      // Now. I need JpTJp, hence.
      const int relevantRows = jacobian.num_rows;// - 9 * numCameras; // since I use residual_blocks
      SparseMatrix<double, Eigen::RowMajor> Jp(relevantRows, 9 * numCameras);
      SparseMatrix<double, Eigen::RowMajor> Jl(relevantRows, 3 * numLandmarks);
      Jp.reserve(VectorXi::Constant(relevantRows, 9));
      Jl.reserve(VectorXi::Constant(relevantRows, 3));
      // JP.setFromTriplets(coefficients.begin(), coefficients.end());

      // std::vector<double> JpJ_cam(9,0);
      for (Eigen::Index r = 0; r < relevantRows; ++r) {
        const int lm_id = lm_obs[r/2];
        const int cam_id = cam_obs[r/2];
        //std::cout << r << ":";
        Eigen::Index idx = jacobian.rows[r];
        // const Eigen::Index c = jacobian.cols[idx]; // index of variable.
        for (int i = 0; i < 9 && idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];++idx, ++i) {
            // if (9 * cam_id + i != jacobian.cols[idx])
            //     std::cout << cluster_id << " jacobian cam/idx do not match " << 9 * cam_id + i << " = " << jacobian.cols[idx] << " | ";
            Jp.insert(r, 9 * cam_id + i) = jacobian.values[idx];
            // if (cam_id == 543) {
            //   std::cout << i << " " << " r " << r << " < " << relevantRows << " " << jacobian.values[idx] << "\n";
            //   JpJ_cam[i] += jacobian.values[idx] * jacobian.values[idx];
            // }
        }
        for (int i = 0; i < 3 && idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];++idx, ++i) {
            // if (cameras.size() + 3 * lm_id + i != jacobian.cols[idx])
            //     std::cout << 3 * lm_id + i << " = " << jacobian.cols[idx];
            Jl.insert(r, 3 * lm_id + i) = jacobian.values[idx];
        }
      }
      // std::cout << " Cam 543: " << std::sqrt(JpJ_cam[0]) << " " << std::sqrt(JpJ_cam[1]) << " " << std::sqrt(JpJ_cam[2]) << " " << std::sqrt(JpJ_cam[3]) << " " << std::sqrt(JpJ_cam[4]) << " " << std::sqrt(JpJ_cam[5]) << " " << std::sqrt(JpJ_cam[6]) << " " << std::sqrt(JpJ_cam[7]) << " " << std::sqrt(JpJ_cam[8]) << "\n";
      Jp.makeCompressed();
      Jl.makeCompressed();
      return std::make_tuple(Jp, Jl, residuals);
    }

    // Also delivers residuals and gradient.
    std::pair<SparseMatrix<double, Eigen::RowMajor>,
              SparseMatrix<double, Eigen::RowMajor>>
    GetJacobian() {
      // 1st get Jacobian(s):
      ceres::Problem::EvaluateOptions evalOptions;
      evalOptions.apply_loss_function = true;
      // evalOpt.parameter_blocks = {}; // TODO only poses.
      evalOptions.residual_blocks = function_residual_blocks;
      evalOptions.num_threads = options.num_threads;
      ceres::CRSMatrix jacobian;
      //std::vector<double> residuals;
      // std::cout << "GetJacobian: Evaluate " << cluster_id << "\n"; 
      problem.Evaluate(evalOptions, &startCost, nullptr, nullptr, &jacobian); //&residuals
      const size_t numUnknowns = jacobian.num_cols;
      //   std::cout << "GetJacobian: " << cluster_id << " Finished eval problem "
      //             << jacobian.num_rows << "-" << 9 * numCameras << "\n";

      // Now. I need JpTJp, hence.
      const int relevantRows = jacobian.num_rows;// - 9 * numCameras; // since I use residual_blocks
      SparseMatrix<double, Eigen::RowMajor> Jp(relevantRows, 9 * numCameras);
      SparseMatrix<double, Eigen::RowMajor> Jl(relevantRows, 3 * numLandmarks);
      Jp.reserve(VectorXi::Constant(relevantRows, 9));
      Jl.reserve(VectorXi::Constant(relevantRows, 3));
      // JP.setFromTriplets(coefficients.begin(), coefficients.end());

      // std::vector<double> JpJ_cam(9,0);
      for (Eigen::Index r = 0; r < relevantRows; ++r) {
        const int lm_id = lm_obs[r/2];
        const int cam_id = cam_obs[r/2];
        //std::cout << r << ":";
        Eigen::Index idx = jacobian.rows[r];
        // const Eigen::Index c = jacobian.cols[idx]; // index of variable.
        for (int i = 0; i < 9 && idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];++idx, ++i) {
            // if (9 * cam_id + i != jacobian.cols[idx])
            //     std::cout << cluster_id << " jacobian cam/idx do not match " << 9 * cam_id + i << " = " << jacobian.cols[idx] << " | ";
            Jp.insert(r, 9 * cam_id + i) = jacobian.values[idx];
            // if (cam_id == 543) {
            //   std::cout << i << " " << " r " << r << " < " << relevantRows << " " << jacobian.values[idx] << "\n";
            //   JpJ_cam[i] += jacobian.values[idx] * jacobian.values[idx];
            // }
        }
        for (int i = 0; i < 3 && idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];++idx, ++i) {
            // if (cameras.size() + 3 * lm_id + i != jacobian.cols[idx])
            //     std::cout << 3 * lm_id + i << " = " << jacobian.cols[idx];
            Jl.insert(r, 3 * lm_id + i) = jacobian.values[idx];
        }
      }
      // std::cout << " Cam 543: " << std::sqrt(JpJ_cam[0]) << " " << std::sqrt(JpJ_cam[1]) << " " << std::sqrt(JpJ_cam[2]) << " " << std::sqrt(JpJ_cam[3]) << " " << std::sqrt(JpJ_cam[4]) << " " << std::sqrt(JpJ_cam[5]) << " " << std::sqrt(JpJ_cam[6]) << " " << std::sqrt(JpJ_cam[7]) << " " << std::sqrt(JpJ_cam[8]) << "\n";
      Jp.makeCompressed();
      Jl.makeCompressed();
      return {Jp,Jl};
    }

    void Init(int numClusters = 1) {
        numCameras = 0;
        numLandmarks = 0;
        numResiduals = 0;
        firstIteration = true;
        current_be = init_be;
        start_be = init_be;
        tr_radius = std::min(max_trust_region_radius, init_trust_region_radius);
        startCost = 1e20;
        cost = 1e20;
        best_cost = cost;
        cluster_id = -1;
        function_residual_blocks.clear();
        // Solve
        // Make Ceres automatically detect the bundle structure. Note that the
        // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is
        // slower for standard bundle adjustment problems.
        //options.linear_solver_type = ceres::DENSE_SCHUR; // SPARSE_SCHUR;// same
        options.linear_solver_type = ceres::ITERATIVE_SCHUR; // same ceres::CGNR;//
        // options.linear_solver_type = ceres::CGNR;
        // options.linear_solver_type = ceres::DENSE_QR; // SHIT
        // options.max_linear_solver_iterations = 100;
        const int threads_per_cluster = std::max(1, _num_threads_machine_ / numClusters);
        options.num_threads = threads_per_cluster; // _ceres_num_threads_; // single cpu -> still slow / bottleneck.
        // options.preconditioner_type = ceres::IDENTITY; // Sucks if CGNR of course. 
        // options.preconditioner_type = ceres::JACOBI; // CGNR -> jacobi anyway.
        options.max_num_iterations = 1;
        // options.minimizer_progress_to_stdout = true;
        options.minimizer_progress_to_stdout = false;
        // options.logging_type = ceres::SILENT;  
    }

    int cluster_id;
    int numCameras = 0;
    int numLandmarks = 0;
    int numResiduals = 0;
    const double init_be = 1e-4;
    double current_be = init_be;
    double start_be = init_be;
    const double init_trust_region_radius = 1e1; // Todo: set to 1?
    double tr_radius = init_trust_region_radius; // 1e4 is ceres standard. -> Init()
    const double max_trust_region_radius = 1e6;
    double startCost;
    double cost;
    double best_cost;
    bool firstIteration = true; // full step is wo. diag part to acc.
    std::vector<ceres::ResidualBlockId> function_residual_blocks;
    std::vector<double> cameras;
    std::vector<double> cameras_s;
    std::vector<double> landmarks;// todo: either revert or send landmarkss all the time.
    std::vector<double> last_landmarks;
    std::vector<double> best_landmarks;
    std::vector<double> stepSize; // internally modelling prox term. 'sqrt' of full_stepSize 
    std::vector<double> full_stepSize; // returned to compute s update in DRS.
    std::vector<double> unorm;
    std::vector<double> vnorm;
    std::vector<int> cam_obs;
    std::vector<int> lm_obs;
    ceres::Problem problem;
    ceres::Solver::Options options;
};
///////////////////////////////////////////////////////

int main() {
    // Initialize the context
    zmq::context_t context(1);

    // Create a socket of type REP (reply)
    zmq::socket_t socket(context, ZMQ_REP);

    //zmq::socket_t push_socket(context, ZMQ_REQ);// ZMQ_PUSH);
    zmq::socket_t push_socket(context, ZMQ_PUSH);
    zmq::socket_t pull_socket(context, ZMQ_REP);// ZMQ_PULL);

    std::mutex mtx; // Mutex for critical section.

    // Bind the socket to a TCP address
    std::cout << "Starting the server on port 5555..." << std::endl;
    socket.bind("tcp://*:5555");
    pull_socket.bind("tcp://*:5556");
    push_socket.bind("tcp://*:5557");

    std::map<int, CeresProgram> cluster_to_program;

    while (true) {
      zmq::message_t request;

      // Wait for the next request from a client
      // socket.recv(&request);
      pull_socket.recv(&request);
      // std::cout << "Received pull request \n";

      // ParseFromString expects a byte string.
      // ParseFromArray expects a byte array and the size of the array.
      // ParseFromString(value) is the same as ParseFromArray(value.data(), value.size()). 

      // const std::string received_message(static_cast<char*>(request.data()), request.size());
      request_proto request_p;
      // request_p.ParseFromString(received_message);
      request_p.ParseFromArray(request.data(), request.size());
      // std::cout << "Request ParseFromArray\n";

      switch(request_p.options_case()) {

        case request_proto::OptionsCase::kUpdate: {
          const prox_cluster_proto update = request_p.update(); // we get an update for the cameras only -- update buffer, run its iterations.
          const int cluster_id = update.cluster_id();
          std::cout << "request_proto::OptionsCase::kUpdate " << cluster_id << std::endl;
          THROW_IF(cluster_to_program.find(cluster_id) == cluster_to_program.end());
          CeresProgram& program = cluster_to_program[cluster_id];
          program.UpdateData(update); // update is local, we nned to fill data in main thread.

          // Define a Lambda Expression
          auto update_lambda = [&push_socket, &cluster_to_program, &mtx](int cluster_id) {
              CeresProgram& program = cluster_to_program[cluster_id];
              // std::cout << cluster_id << " Update "<< "\n";

            //   program.UpdateStepSize();
            //   program.Solve();
              program.UpdateStepSizeAndSolve();

              return_cluster_proto return_proto = program.FillReturnProto();
              const double cost = 2 * program.GetCost();
              return_proto.set_cost(cost);
              //std::cout << "Cost from update " << cost <<"\n";
              // SerializeToArray saves memory and time?
              size_t bytes = return_proto.ByteSizeLong();
              zmq::message_t reply(bytes);
              return_proto.SerializeToArray(reply.data(), bytes);
              std::lock_guard<std::mutex> lock(mtx);
              push_socket.send(reply, zmq::send_flags::none);
              // std::cout << cluster_id << ". Update send" << std::endl;
          };

          //std::thread update_thread(update_lambda, std::ref(program), std::cref(update));
          std::thread update_thread(update_lambda, cluster_id);
          update_thread.detach();
          ///update_thread.join();

          break;
        }

        // one idea would be to receive, start a thread to compute result, send the result.
        // Problem: ZMQ_REP is blocking -- zmq.REQ is alos blocking in python.
        // Dealer is like an assync Req socket. Router is like an assync Rep Socket.
        // Request (REQ) / reply (REP).
        // If we replace REP with ROUTER. This gives us an asynchronous server that can talk to multiple REQ clients
        //Push/Pull Pattern. 
        // client pushes to port A, server listens/pull to port A in loop
        // server does threaded work and sends/pushes result to port B, client listens/pulls to port B 

        // if we get program we setup new program. if we get cam & prox we update cams (?) and prox term only! do one more it, etc.
        case request_proto::OptionsCase::kProgram : {
          const program_proto pro = request_p.program();
          const int cluster_id =  pro.cluster_id();
          std::cout << "request_proto::OptionsCase::kProgram " << cluster_id << std::endl;
          //if(cluster_to_program.find(cluster_id) == cluster_to_program.end())
          CeresProgram& program = cluster_to_program[pro.cluster_id()];
          program.ResetProgram(pro);

          auto program_lambda = [&cluster_to_program, &push_socket, &mtx](int cluster_id) {
            CeresProgram& program = cluster_to_program[cluster_id];

            // program.UpdateStepSize();
            // program.Solve();
            //std::this_thread::sleep_for(std::chrono::seconds(5));
            program.UpdateStepSizeAndSolve();

            return_cluster_proto return_proto = program.FillReturnProto();
            const double cost = 2 * program.GetCost();
            return_proto.set_cost(cost);
            // SerializeToArray saves memory and time?
            const size_t bytes = return_proto.ByteSizeLong();
            zmq::message_t reply(bytes);
            return_proto.SerializeToArray(reply.data(), bytes);
            std::lock_guard<std::mutex> lock(mtx);
            push_socket.send(reply, zmq::send_flags::none);
          };
          //std::thread program_thread(program_lambda, std::ref(program), std::cref(pro));
          std::thread program_thread(program_lambda, cluster_id);
          program_thread.detach();
          //program_thread.join();// ok, so proto pro runs out of scope / gets deleted.
          //std::this_thread::sleep_for(std::chrono::seconds(0)); // >5 s ok, so .. haeh?
          break;
        }

        case request_proto::OptionsCase::kCostUpdate: {
          const cost_proto costUpdate = request_p.cost_update();
          const int cluster_id = costUpdate.cluster_id();
          std::cout << "request_proto::OptionsCase::kCostUpdate " << cluster_id << std::endl;
          THROW_IF(cluster_to_program.find(cluster_id) == cluster_to_program.end());
          CeresProgram& program = cluster_to_program[cluster_id];
          program.UpdateCameras(costUpdate); // update is local, we need to fill data in main thread.
          
          // Define a Lambda Expression
          auto cost_lambda = [&push_socket, &cluster_to_program, &mtx](int cluster_id) {
            CeresProgram& program = cluster_to_program[cluster_id];
            const double cost = 2 * program.GetCost();
            return_cost_proto return_proto;
            return_proto.set_cost(cost);
            
            return_proto.set_cluster_id(cluster_id);
            // SerializeToArray saves memory and time?
            const size_t bytes = return_proto.ByteSizeLong();
            zmq::message_t reply(bytes);
            return_proto.SerializeToArray(reply.data(), bytes);
            std::lock_guard<std::mutex> lock(mtx);
            push_socket.send(reply, zmq::send_flags::none);
          };
          std::thread cost_thread(cost_lambda, cluster_id);
          cost_thread.detach();
          break;
        }

        case request_proto::OptionsCase::kPreconditioningUpdate : {
            std::cout << "request_proto::OptionsCase::kPreconditioningUpdate" << std::endl;
            const preconditioning_proto ppro = request_p.preconditioning_update();
            // Define a Lambda Expression
            CeresProgram& program = cluster_to_program[ppro.cluster_id()];
            program.UpdatePreconditioning(ppro);
        }

        default: {
        break;
        }
      }

      // Here one would send back 'ack' / 'ok'. Unclear if necessary, blocks on sender -- maybe good idea.
      // std::cout << "Sending message acknowledged to pull_socket\n";
      const std::string reply_message = "Ok";
      zmq::message_t reply(reply_message.size());
      memcpy(reply.data(), reply_message.data(), reply_message.size());
      pull_socket.send(reply, zmq::send_flags::none);
    } // end while 

    return 0;
}