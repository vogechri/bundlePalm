// Original Code from http://kr.zeromq.org/cpp:hwserver
// kongineer.com

#define _ceres_num_threads_ 1
// #define __unweighted_system__

#include <zmq.hpp>
#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include "generated/proto/test.pb.h"
#include <google/protobuf/message_lite.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include<Eigen/Dense>
#include <Eigen/Eigenvalues> 

#include "ceres/ceres.h"
#include "ceres/normal_prior.h"
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
        // camera[0,1,2] are the angle-axis rotation.
        //   T d[9];
        //   d[0] = camera[0] - camera_s[0];
        //   d[1] = camera[1] - camera_s[1];
        //   d[2] = camera[2] - camera_s[2];
        //   d[3] = camera[3] - camera_s[3];
        //   d[4] = camera[4] - camera_s[4];
        //   d[5] = camera[5] - camera_s[5];
        //   d[6] = camera[6] - camera_s[6];
        //   d[7] = camera[7] - camera_s[7];
        //   d[8] = camera[8] - camera_s[8];

        Eigen::Matrix<T, 9, 1> d = Eigen::Map<const Eigen::Matrix<T, 9, 1>>(camera) - Eigen::Map<const Eigen::Matrix<T, 9, 1>>(camera_s);
        Eigen::Map<Eigen::Matrix<T, 9, 1>> residualsVector(residuals);
        residualsVector = Eigen::Map<const Eigen::Matrix<T, 9, 9>>(matBlock) * d;

        // The error is the difference between the predicted and observed position.
        //   residuals[0] = predicted_x - observed_x;
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
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
    int numrows = mat.rows();
    int numNonZeros = mat.nonZeros();
    if(numNonZeros != mat.rows() * N)
        std::cout << "BlockSqrt " << mat.nonZeros() << " ?= " << mat.rows() * N << "\n";
    THROW_IF(mat.rows() != mat.cols());
    THROW_IF(mat.nonZeros() != mat.rows() * N);
    double* values = mat.valuePtr();
    //std::cout << "before  "<< values[0]<< " " << values[1]<< " " << values[2]<< " " << values[3] << "\n";
    for (int i = 0; i < numrows / N; i++) {
        auto mat9x9 = Eigen::Map< Eigen::Matrix<double,N,N> > (&(values[i * N*N]));//,  Eigen::Stride<0, 0>);
        //std::cout << "before "<< mat9x9 << " \n";

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,N,N> > eigensolver;
        eigensolver.computeDirect(mat9x9, Eigen::DecompositionOptions::ComputeEigenvectors);
        //VPQ_EXPECT_EQ(eigensolver.info(), Eigen::Success);

        // SqrtCovEigenValues are sorted in decreasing order.
        Eigen::Vector<double, N> sqrtEigenValues = eigensolver.eigenvalues().cwiseSqrt();//.cwiseMax(lowerBoundSquared).cwiseSqrt().cwiseInverse();
        mat9x9 = eigensolver.eigenvectors() * sqrtEigenValues.asDiagonal() * eigensolver.eigenvectors().transpose();
        //std::cout << "after  "<< mat9x9.transpose() * mat9x9 << " \n";
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
    for (int i = 0; i < numrows / N; i++) {
        auto mat9x9 = Eigen::Map< Eigen::Matrix<double,N,N> > (&(values[i * N*N]));//,  Eigen::Stride<0, 0>);
        //std::cout << "before "<< mat9x9 << " \n";
        mat9x9 = mat9x9.inverse();
        //std::cout << "after "<< mat9x9 << " \n";        
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

// Ensure we can have a vector of programs. by id. maybe just a map : clusterid-> program.
///////////////////////////////////////////////////////
class CeresProgram {
public:
    CeresProgram() {
      Init();
    }

    int ClusterId() { return cluster_id; }

    void ResetProgram(const program_proto &pro) {
      Init();
      numCameras = pro.cameras_size() / 9;
      numLandmarks = pro.landmarks_size() / 3;
      //std::cout << numCameras << " " << numLandmarks << "\n";
      cameras.clear();
      cameras.reserve(9 * numCameras);
      for (const float &v : pro.cameras()) {
        cameras.push_back(v);
      }
      //std::cout << "cameras.push_back\n";
      landmarks.clear();
      landmarks.reserve(3 * numLandmarks);
      for (const float &v : pro.landmarks()) {
        landmarks.push_back(v);
      }
      //std::cout << "landmarks.push_back\n";
      cameras_s.clear();
      cameras_s.reserve(9 * numCameras);
      for (const float &v : pro.cameras()) {
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
      for (const float &v : pro.unorm()) {
        unorm.push_back(v);
      }
      //std::cout << "cameras.push_back\n";
      vnorm.clear();
      vnorm.reserve(3 * numLandmarks);
      for (const float &v : pro.vnorm()) {
        vnorm.push_back(v);
      }
      //std::cout << "data updated\n";

      options.max_num_iterations = std::max(0, std::min(10, pro.iterations()));
      options.initial_trust_region_radius = tr_radius;
      problem = ceres::Problem();
      cluster_id = pro.cluster_id();
      be = pro.be();
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

    // Also delivers residuals and gradient.
    std::pair<SparseMatrix<double, Eigen::RowMajor>,
              SparseMatrix<double, Eigen::RowMajor>>
    GetJacobian() {
      // 1st get Jacobian(s):
      ceres::Problem::EvaluateOptions evalOptions;
      evalOptions.apply_loss_function = true;
      // evalOpt.parameter_blocks = {}; // TODO only poses.
      evalOptions.residual_blocks = function_residual_blocks;
      evalOptions.num_threads = 1;
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
      Jp.reserve(VectorXi::Constant(2 * relevantRows, 9));
      Jl.reserve(VectorXi::Constant(2 * relevantRows, 3));
      // JP.setFromTriplets(coefficients.begin(), coefficients.end());
      for (Eigen::Index r = 0; r < relevantRows; ++r) {
            const int lm_id = lm_obs[r/2];
            const int cam_id = cam_obs[r/2];
            //std::cout << r << ":";
            Eigen::Index idx = jacobian.rows[r];
            // const Eigen::Index c = jacobian.cols[idx]; // index of variable.
            for (int i = 0; i < 9 && idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];++idx,++i) {
                // if (9 * cam_id + i != jacobian.cols[idx])
                //     std::cout << 9 * cam_id + i << " = " << jacobian.cols[idx] << " | ";
                Jp.insert(r, 9 * cam_id + i) = jacobian.values[idx];
            }
            for (int i = 0; i < 3 && idx < jacobian.rows[r + static_cast<Eigen::Index>(1)];++idx, ++i) {
                // if (cameras.size() + 3 * lm_id + i != jacobian.cols[idx])
                //     std::cout << 3 * lm_id + i << " = " << jacobian.cols[idx];
                Jl.insert(r, 3 * lm_id + i) = jacobian.values[idx];
            }
        }
        Jp.makeCompressed();
        Jl.makeCompressed();
        return {Jp,Jl};
    }

    double GetCost() {
      // 1st get Jacobian(s):
      ceres::Problem::EvaluateOptions evalOptions;
      evalOptions.apply_loss_function = true;
      // evalOpt.parameter_blocks = {};
      evalOptions.residual_blocks = function_residual_blocks;
      evalOptions.num_threads = 1;
      std::vector<double> residuals;
      double cost;
      problem.Evaluate(evalOptions, &cost, &residuals, nullptr, nullptr);
      return cost;
    }

    void SetBe(double be) { be = be; }

    // Currently this is set 'stepsize' from Jp only.
    void SetStepSize(const SparseMatrix<double, Eigen::RowMajor> &Jp) {
      // std::cout << "Set step size " << cluster_id << "\n";
      if(Jp.nonZeros() != 9 * Jp.rows())
          std::cout << "Jp " << cluster_id << " | " << Jp.nonZeros() << " =? " << Jp.rows() * 9 << "\n";
      THROW_IF(Jp.nonZeros() != 9 * Jp.rows());
      SparseMatrix<double, Eigen::RowMajor> JpJ(9 * numCameras, 9 * numCameras);
      JpJ.reserve(VectorXi::Constant(9 * numCameras, 9));
      JpJ = Jp.transpose() * Jp;
      // SparseMatrix<double, Eigen::RowMajor> JlJ(3 * numLandmarks, 3 * numLandmarks);
      // JlJ.reserve(VectorXi::Constant(3 * numLandmarks, 3));
      // JlJ = Jl.transpose() * Jl;
      // auto JpJ_diag = JpJ.diagonal().array();
      if(JpJ.nonZeros() != stepSize.size() || JpJ.rows() * 9 != numCameras * 81)
        std::cout << "JpJ " << cluster_id << " | " << JpJ.nonZeros() << " =? " << JpJ.rows() * 9
                  << " " << stepSize.size() << " " << numCameras * 81 << "\n";
      THROW_IF(JpJ.nonZeros() != numCameras * 81);

      full_stepSize.resize(stepSize.size(), 0);
      const double *values = JpJ.valuePtr();
      for (int id = 0; id < 81 * numCameras; ++id) {
        full_stepSize[id] = values[id]; // this is returned, the other is just used in the eq.
      }

      auto diag = JpJ.diagonal().array();
      for (int b = 0; b < numCameras; ++b) { // block
        double mv = diag(9*b);
        for (int id = 1; id < 9; ++id) {
          mv = std::max(mv, diag(9*b + id));
        }
        for (int id = 0; id < 9; ++id) {
          diag(9*b + id) = mv;
        }
      }
      JpJ.diagonal().array() += be * diag;

      //JpJ.diagonal().array() *= (1. + be); //+= be * JpJ.diagonal().array();
      // JpJ = JpJ * 3; // optional to test. in theory should almost always suffice.
    //   const auto JpJDiagonal = JpJ.diagonal();//.array();
    //   JpJ = JpJ * 0.5 * 1e-12;
    //   //JpJ.diagonal() = JpJ.diagonal() + be * JpJDiagonal;
    //   JpJ.diagonal() = JpJDiagonal * (1. + be);
    //JpJ.diagonal() += be * JpJ.diagonal();
    //JpJ.diagonal().array().cwise
      // std::cout << "BlockSqrt " << cluster_id << "\n";
      BlockSqrt<9>(JpJ); // need templated fct.
      // instead reset variable block(s) JpJ and s to sqrt(Stepsize)
      values = JpJ.valuePtr();
      for (int id = 0; id < 81 * numCameras; ++id) {
        stepSize[id] = values[id]; // = 1000 -> different cost: so ok
      }
    }
    
    return_cluster_proto FillReturnProto() {
      return_cluster_proto return_proto = return_cluster_proto();
      for (const double &v : cameras) {
        //return_proto.set_cameras(id++, static_cast<float>(v));
        return_proto.add_cameras(static_cast<float>(v));
      }
      for (const double &v : landmarks) {
        return_proto.add_landmarks(static_cast<float>(v));
      }
      for (const double &v : full_stepSize) {
        return_proto.add_step_size(static_cast<float>(v));
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
      tr_radius = summary.iterations.back().trust_region_radius;
      //tr_radius = summary.trust_region_radius();
      // TODO: -ordering=user for schur (maybe cameras 1st then landmarks)
      cost = summary.final_cost * 2;
      // std::cout << "\nUpdate Mycost: " << summary.final_cost * 2 << "\n";
      return cost;
    }

    void UpdateCameras(const cost_proto& costProto) {
      // std::cout << "Update cluster " << cluster_id << " update proto id:" << update.cluster_id() << "\n";
      THROW_IF(costProto.cameras_size() != cameras.size());
      THROW_IF(costProto.cluster_id() != cluster_id);

      int id = 0; // fill existing buffer
      for (const float &v : costProto.cameras()) {
        cameras[id++] = v;
      }
    }

    void UpdateData(const prox_cluster_proto &update) {
        // std::cout << "Update cluster " << cluster_id << " update proto id:" << update.cluster_id() << "\n";
      THROW_IF(update.cameras_size() != cameras.size());
      THROW_IF(update.cameras_s_size() != cameras_s.size());
      THROW_IF(update.cluster_id() != cluster_id);

      int id = 0; // fill existing buffer
      for (const float &v : update.cameras()) {
        cameras[id++] = v;
      }
      id = 0;
      for (const float &v : update.cameras_s()) {
        cameras_s[id++] = v;
      }
      be = update.be();
      options.initial_trust_region_radius = tr_radius; // use from last solve.
    }

    void UpdateStepSize() {
      // Recompute.
      stepSize.clear(); // all 0 to ensure jacobian is reasonable.
      stepSize.resize(81 * numCameras, 0);
      const auto [Jp, Jl] = GetJacobian();
      SetStepSize(Jp);
    }

    void UpdatePreconditioning(const preconditioning_proto& preconditioningProto) {
        // std::cout << "Update cluster " << cluster_id << " update proto id:" << update.cluster_id() << "\n";
        THROW_IF(preconditioningProto.unorm_size() != unorm.size());
        THROW_IF(preconditioningProto.vnorm_size() != vnorm.size());

        int id = 0; // fill existing buffer
        for (const float &v : preconditioningProto.unorm()) {
            unorm[id++] = v;
        }
        id = 0; // fill existing buffer
        for (const float &v : preconditioningProto.vnorm()) {
            vnorm[id++] = v;
        }
    }

private:

    void Init() {
        numCameras = 0;
        numLandmarks = 0;
        be = 1e-4;
        tr_radius = 1e4;
        startCost = 1e12;
        cost = 1e12;
        cluster_id = -1;
        function_residual_blocks.clear();
        // Solve
        // Make Ceres automatically detect the bundle structure. Note that the
        // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is
        // slower for standard bundle adjustment problems.
        options.linear_solver_type = ceres::DENSE_SCHUR; // SPARSE_SCHUR;// same
        // options.linear_solver_type = ITERATIVE_SCHUR; // same ceres::CGNR;//
        // options.linear_solver_type = ceres::CGNR;
        // options.linear_solver_type = ceres::DENSE_QR; // SHIT
        // options.max_linear_solver_iterations = 100;
        options.num_threads = _ceres_num_threads_; // single cpu -> still slow / bottleneck.
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
    double be = 1e-4;
    double tr_radius = 1e4;
    double startCost;
    double cost;
    std::vector<ceres::ResidualBlockId> function_residual_blocks;
    std::vector<double> cameras_s;
    std::vector<double> cameras;
    std::vector<double> landmarks;// todo: either revert or send landmarkss all the time.
    std::vector<double> stepSize;
    std::vector<double> full_stepSize;
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
                //std::cout << "request_proto::OptionsCase::kUpdate" << std::endl;
                const prox_cluster_proto update = request_p.update(); // we get an update for the cameras only -- update buffer, run its iterations.
                const int cluster_id = update.cluster_id();
                THROW_IF(cluster_to_program.find(cluster_id) == cluster_to_program.end());
                CeresProgram& program = cluster_to_program[cluster_id];
                program.UpdateData(update); // update is local, we nned to fill data in main thread.

                // Define a Lambda Expression
                auto update_lambda = [&push_socket, &cluster_to_program](int cluster_id) {
                    CeresProgram& program = cluster_to_program[cluster_id];
                    // std::cout << cluster_id << " Update "<< "\n";
                    program.UpdateStepSize();
                    // const auto [Jp, Jl] = program.GetJacobian();
                    // program.SetStepSize(Jp);
                    program.Solve();
                    return_cluster_proto return_proto = program.FillReturnProto();
                    const double cost = 2 * program.GetCost();
                    return_proto.set_cost(cost);
                    //std::cout << "Cost from update " << cost <<"\n";
                    // SerializeToArray saves memory and time?
                    size_t bytes = return_proto.ByteSizeLong();
                    zmq::message_t reply(bytes);
                    return_proto.SerializeToArray(reply.data(), bytes);
                    push_socket.send(reply, zmq::send_flags::none);
                    // std::cout << cluster_id << ". Update send" << std::endl;
                };

                //std::thread update_thread(update_lambda, std::ref(program), std::cref(update));
                std::thread update_thread(update_lambda, cluster_id);
                update_thread.detach();
                //update_thread.join();

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
                //std::cout << "request_proto::OptionsCase::kProgram" << std::endl;
                const program_proto pro = request_p.program();

                //if(cluster_to_program.find(cluster_id) == cluster_to_program.end())
                CeresProgram& program = cluster_to_program[pro.cluster_id()];
                //std::cout << pro.cluster_id() << " Program "<< "\n";
                program.ResetProgram(pro);

                auto program_lambda = [&cluster_to_program, &push_socket](int cluster_id) {
                    CeresProgram& program = cluster_to_program[cluster_id];
                    const auto [Jp, Jl] = program.GetJacobian();
                    program.SetStepSize(Jp);
                    program.Solve();
                    //std::this_thread::sleep_for(std::chrono::seconds(5)); // sleep here, pollin / block pull/push, no send? dies before sleep ends.
                    return_cluster_proto return_proto = program.FillReturnProto();
                    const double cost = 2 * program.GetCost();
                    return_proto.set_cost(cost);
                    // SerializeToArray saves memory and time?
                    const size_t bytes = return_proto.ByteSizeLong();
                    zmq::message_t reply(bytes);
                    return_proto.SerializeToArray(reply.data(), bytes);
                    push_socket.send(reply, zmq::send_flags::none);
                };
                //std::thread program_thread(program_lambda, std::ref(program), std::cref(pro));
                std::thread program_thread(program_lambda, pro.cluster_id());
                program_thread.detach();
                //program_thread.join();// ok, so proto pro runs out of scope / gets deleted.
                //std::this_thread::sleep_for(std::chrono::seconds(0)); // >5 s ok, so .. haeh?
                break;
            }

            case request_proto::OptionsCase::kCostUpdate: {
                //std::cout << "request_proto::OptionsCase::kCostUpdate" << std::endl;
                const cost_proto costUpdate = request_p.cost_update();
                const int cluster_id = costUpdate.cluster_id();
                THROW_IF(cluster_to_program.find(cluster_id) == cluster_to_program.end());
                CeresProgram& program = cluster_to_program[cluster_id];
                program.UpdateCameras(costUpdate); // update is local, we need to fill data in main thread.

                // Define a Lambda Expression
                auto cost_lambda = [&push_socket, &cluster_to_program](int cluster_id) {
                    CeresProgram& program = cluster_to_program[cluster_id];
                    const double cost = 2 * program.GetCost();
                    return_cost_proto return_proto;
                    return_proto.set_cost(cost);

                    return_proto.set_cluster_id(cluster_id);
                    // SerializeToArray saves memory and time?
                    const size_t bytes = return_proto.ByteSizeLong();
                    zmq::message_t reply(bytes);
                    return_proto.SerializeToArray(reply.data(), bytes);
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