// Concurrency contract: one client drives this server and waits for all
// asynchronous cluster results before starting the next phase. Worker replies
// share the PUSH socket under a mutex. Revisit socket ownership and program
// synchronization before allowing multiple clients or overlapping phases.

// #define _ceres_num_threads_ 1
#define _num_threads_machine_ 31
// #define _const_diag_
// #define __ceresVersion__

#include <zmq.hpp>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <array>
#include <algorithm>
#include <string>
#include <iostream>
#include <thread>
#include <mutex>
#include <omp.h>
#include <chrono>
#include "proto/test.pb.h"
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
using Eigen::RowMajor;
using Eigen::Matrix;
using Eigen::Map;

#ifdef BUNDLE_PALM_VERBOSE_LOGGING
#define WORKER_LOG(expression) do { std::cout << expression; } while (false)
#else
#define WORKER_LOG(expression) do { } while (false)
#endif

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

double EnvironmentDouble(const char* name, double default_value,
                          double minimum, double maximum) {
  const char* text = std::getenv(name);
  if (text == nullptr || *text == '\0') {
    return default_value;
  }
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(text, &end);
  if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value) ||
      value < minimum || value > maximum) {
    std::ostringstream error;
    error << name << " must be a finite value in [" << minimum << ", "
          << maximum << "]";
    throw std::runtime_error(error.str());
  }
  return value;
}

double CameraDiagonalRelativeFloor() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CAMERA_DIAGONAL_FLOOR", 1e-48, 0.0, 1.0);
  return value;
}

double LocalAcceptanceRatio() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_LOCAL_ACCEPTANCE_RATIO", 0.9999, 0.0, 1.0);
  return value;
}

bool LocalSolveMetricsEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("BUNDLE_PALM_LOCAL_SOLVE_METRICS");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}

bool BatchedEvaluationEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("BUNDLE_PALM_BATCHED_EVALUATION");
    return value == nullptr || std::string(value) != "0";
  }();
  return enabled;
}

void EmitLocalSolveMetric(const std::string& metric) {
  static std::mutex mutex;
  const std::lock_guard<std::mutex> lock(mutex);
  std::cerr.write(metric.data(), metric.size());
  std::cerr.flush();
}

using TimingClock = std::chrono::steady_clock;

double ElapsedSeconds(const TimingClock::time_point& start) {
  return std::chrono::duration<double>(TimingClock::now() - start).count();
}

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
        Matrix<T, 9, 1> d = Map<const Matrix<T, 9, 1>>(camera) - Map<const Matrix<T, 9, 1>>(camera_s);
        Map<Matrix<T, 9, 1>> residualsVector(residuals);
        residualsVector = Map<const Matrix<T, 9, 9>>(matBlock) * d;
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
template <typename T>
bool EvaluateWeightedProjection(const T* const camera,
                                const T* const point,
                                double observed_x, double observed_y,
                                T* residuals) {
  T p[3];
  ceres::AngleAxisRotatePoint(camera, point, p);

  p[0] += camera[3];
  p[1] += camera[4];
  p[2] += camera[5];

  T xp = -p[0] / p[2];
  T yp = -p[1] / p[2];
  const T& l1 = camera[7];
  const T& l2 = camera[8];
  T r2 = xp * xp + yp * yp;
  T distortion = 1.0 + r2 * (l1 + l2 * r2);
  const T& focal = camera[6];
  T predicted_x = focal * distortion * xp;
  T predicted_y = focal * distortion * yp;

  residuals[0] = predicted_x - observed_x;
  residuals[1] = predicted_y - observed_y;
  return true;
}

struct SnavelyReprojectionErrorWeighted {
    SnavelyReprojectionErrorWeighted(
        double observed_x, double observed_y,
        const double* camera_weight, const double* point_weight,
        const double* camera_transform)
        : observed_x(observed_x), observed_y(observed_y),
          camera_weight(camera_weight), point_weight(point_weight),
          camera_transform(camera_transform) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {

      T cameraW[9];
      for (int row = 0; row < 9; ++row) {
        cameraW[row] = T(0);
        for (int col = 0; col < 9; ++col) {
          cameraW[row] += T(camera_transform[9 * row + col])
                          * camera[col] * T(camera_weight[col]);
        }
      }
      T pointW[3];
      pointW[0] = point[0] * T(point_weight[0]);
      pointW[1] = point[1] * T(point_weight[1]);
      pointW[2] = point[2] * T(point_weight[2]);
        return EvaluateWeightedProjection(
          cameraW, pointW, observed_x, observed_y, residuals);
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                       const double observed_y,
                       const double* camera_weight,
                       const double* point_weight,
                       const double* camera_transform) {
      return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorWeighted, 2, 9, 3>(
        new SnavelyReprojectionErrorWeighted(
          observed_x, observed_y, camera_weight, point_weight,
          camera_transform)));
    }

    double observed_x;
    double observed_y;
    const double* camera_weight;
    const double* point_weight;
    const double* camera_transform;
  };
#endif

template<int N>
void BlockSqrt(SparseMatrix<double, RowMajor>& mat) {
    const int numrows = mat.rows();
    const int numNonZeros = mat.nonZeros();
    if(numNonZeros != mat.rows() * N)
        WORKER_LOG("BlockSqrt " << mat.nonZeros() << " ?= " << numrows * N << "\n");
    THROW_IF(numrows != mat.cols());
    THROW_IF(mat.nonZeros() != numrows * N);
    double* values = mat.valuePtr();
    //std::cout << "before  "<< values[0]<< " " << values[1]<< " " << values[2]<< " " << values[3] << "\n";
#pragma omp parallel for num_threads(options.num_threads)
    for (int i = 0; i < numrows / N; i++) {
        auto matNxN = Map< Matrix<double, N, N> > (&(values[i * N*N]));//,  Eigen::Stride<0, 0>);
        //std::cout << "before "<< matNxN << " \n";

        Eigen::SelfAdjointEigenSolver<Matrix<double,N,N> > eigensolver;
        eigensolver.computeDirect(matNxN, Eigen::DecompositionOptions::ComputeEigenvectors);
        //VPQ_EXPECT_EQ(eigensolver.info(), Eigen::Success);

        // SqrtCovEigenValues are sorted in decreasing order.
        const Eigen::Vector<double, N> sqrtEigenValues = eigensolver.eigenvalues().cwiseAbs().cwiseSqrt().cwiseMax(1e-16);//.cwiseMax(lowerBoundSquared).cwiseSqrt().cwiseInverse();
        // recall : i had here min ev >= 1e-6 * maxEv. Could return a diag matrix
        matNxN = eigensolver.eigenvectors() * sqrtEigenValues.asDiagonal() * eigensolver.eigenvectors().transpose();

        //auto matNxN_out = Map< Matrix<double,N,N> > (&(values[i * N*N]));//,  Eigen::Stride<0, 0>);
        //std::cout << "after  "<< matNxN_out.transpose() * matNxN_out << " \n";
    }
    //std::cout << "after  "<< values[0]<< " " << values[1]<< " " << values[2]<< " " << values[3] << "\n";
}

template<int N>
void BlockInverse(SparseMatrix<double, RowMajor>& mat) {
    const int numrows = mat.rows();
    THROW_IF(mat.rows() != mat.cols());
    double* values = mat.valuePtr();
#pragma omp parallel for num_threads(options.num_threads)
    for (int i = 0; i < numrows / N; i++) {
        auto matNxN = Map< Matrix<double,N,N> > (&(values[i * N*N]));//,  Eigen::Stride<0, 0>);
    matNxN = matNxN.inverse().eval();
    }
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
Diagonal(SparseMatrix<double, RowMajor>& mat, int cluster_id = -1) {
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> diag = mat.diagonal().asDiagonal(); // ?
#ifdef _const_diag_
  if (N == 9) {
    // const double target = 1e-2; // the larger the stricter regularized.
    // const double eps = 1e-20;
    // const double d = -std::log10(eps);
    // const double t = 1e-1; // should map to target == 1e-3. 1e-3 is best value that works if too small.

    // const double f_t = -std::log10(t + eps) / d;
    // const double power = std::log(target) / std::log(f_t);

    // const double test = std::pow( -std::log10(std::min(1., t) + eps) / d, power);
    // std::cout << "t=" << t << " should map to " << target << " == f(t) = " << test << "\n";

    auto& diagdiag = diag.diagonal();
#pragma omp parallel for num_threads(options.num_threads)
    for (int b = 0; b < mat.rows() / N; ++b) { // block
      double mv = diagdiag(N*b);
      for (int id = 1; id < N; ++id) {
        mv = std::max(mv, diagdiag(N*b + id));
      }
      mv = std::max(1e-32, mv);
      // TODO: stricter if max is very small?
      // With PCG, The GN-Hessian aprox should have a scale of #clusters (1/#clusters?). So we can say if max is < t we must correct. 
      // t = 1e-4. We can even scale. if 1 -> 1e-10, 1e-4: 1e-3. maybe log() -- could also be global min (max cam block)
      // log(1e-4)

      // f(mv): with f(t)=target, f(1) = 0, f(0) = 1.
      // const double variableThresh = std::pow( -std::log10(std::min( 1., mv) + eps) / d, power);
      // if (N == 9 && b < 10) {
      //   std::cout << "mv " << b << ": f(" << mv << ") = " << variableThresh << "\n";
      // }
      for (int id = 0; id < N; ++id) {
          // diagdiag(N*b + id) = mv;

          // This does ok in handling cases with degenerate cameras (observing only 1/2/3 landmarks in cluster).
          // Bad clustering 1e-3, not 1e-4. so < 1e-4, but 1e-2 a bit better than 1e-3? a bit random.
          diagdiag(N*b + id) = std::max(1e-3 * mv, diagdiag(N*b + id)); // not < 1/inf. also 1e10: ok, 1e-6: bit worse

          //diagdiag(N*b + id) = std::max(variableThresh * mv, diagdiag(N*b + id)); // not < 1/inf. also 1e10: ok, 1e-6: bit worse
      }
    }
  }
#else
  if (N == 9) {
    auto& blockDiagonal = diag.diagonal();
    const bool collectMetrics = LocalSolveMetricsEnabled();
    int flooredEntries = 0;
    int zeroEntries = 0;
    double minimumRelativeDiagonal = 1.0;
#pragma omp parallel for num_threads(options.num_threads) \
    reduction(+:flooredEntries, zeroEntries) reduction(min:minimumRelativeDiagonal)
    for (int block = 0; block < mat.rows() / N; ++block) {
      const auto cameraDiagonal = blockDiagonal.template segment<N>(N * block);
      const double maxDiagonal = std::max(1e-32, cameraDiagonal.maxCoeff());
      const double floor = CameraDiagonalRelativeFloor() * maxDiagonal;
      if (collectMetrics) {
        for (int coordinate = 0; coordinate < N; ++coordinate) {
          zeroEntries += cameraDiagonal[coordinate] == 0.0;
          flooredEntries += cameraDiagonal[coordinate] < floor;
          minimumRelativeDiagonal = std::min(
              minimumRelativeDiagonal,
              cameraDiagonal[coordinate] / maxDiagonal);
        }
      }
      blockDiagonal.template segment<N>(N * block) = 
        cameraDiagonal.cwiseMax(floor);
      // if (cameraDiagonal.minCoeff() < 1e-24 * maxDiagonal) {
      //   blockDiagonal.template segment<N>(N * block) =
      //       cameraDiagonal.cwiseMax(1e-24 * maxDiagonal);
      // }
    }
    if (collectMetrics) {
      std::ostringstream metric;
      metric << "CAMERA_DIAGONAL cluster=" << cluster_id
         << " blocks=" << mat.rows() / N
         << " entries=" << blockDiagonal.size()
         << " floored=" << flooredEntries
         << " zeros=" << zeroEntries
         << " min_relative=" << minimumRelativeDiagonal
         << " floor=" << CameraDiagonalRelativeFloor() << "\n";
      EmitLocalSolveMetric(metric.str());
    }
  }
#endif
  // diag.diagonal().array() += 1e-16; // TODO: this is not good/ ok?. does something? test 3086
  // diag.diagonal().array() = diag.diagonal().array().cwiseMax(1e-10); 

  // just like const diag, add 1e-16 * max value to all diagonal entries / cwiseMax(1e-16 * mv).

  // std::cout << diag.diagonal() << "\n";
  return diag;
}

// TODO: Add a best landmarks from fixed poses (best, all same globally). Run after reset to restart with optimal landmarks to poses.
// same as set s to large value, yet faster: no need for solving eqs == set iterations to 0.

// StepSize as matrix is needed for multiplication with vectors. Still should be easy as blockMult std vector with other vector.
template<int N>
std::vector<double> blockMult(const std::vector<double>& blockMat, const std::vector<double>& vec) {
  std::vector<double> res(blockMat.size() / N, 0);
#pragma omp parallel for num_threads(options.num_threads)
  for (int id = 0; id < res.size(); ++id) {
    for (int k = 0; k < N; ++k) {
      res[id] += blockMat[id * N + k] * vec[(id / N ) * N + k];
    }
  }
  return res;
}

template<int N>
Eigen::VectorXd blockMult(const std::vector<double>& blockMat, const Eigen::VectorXd& vec) {
  const int num = blockMat.size() / N;
  Eigen::VectorXd res = Eigen::VectorXd::Zero(num); // Eigen::Vector<double, Eigen::Dynamic> res(num, 0);
#pragma omp parallel for num_threads(options.num_threads)
  for (int id = 0; id < res.size(); ++id) {
    for (int k = 0; k < N; ++k) {
      res[id] += blockMat[id * N + k] * vec[(id / N ) * N + k];
    }
  }
  return res;
}

template<int N>
void blockMult(const std::vector<double>& blockMat, const Eigen::VectorXd& vec, Eigen::VectorXd& res) {
  const int num = blockMat.size() / N;
  res.setZero(num);
#pragma omp parallel for num_threads(options.num_threads)
  for (int id = 0; id < res.size(); ++id) {
    for (int k = 0; k < N; ++k) {
      res[id] += blockMat[id * N + k] * vec[(id / N ) * N + k];
    }
  }
}

template<int N>
SparseMatrix<double, RowMajor> BlockDiagonalJtJ(
    const SparseMatrix<double, RowMajor>& jacobian, int num_blocks) {
  std::vector<double> blocks(num_blocks * N * N, 0.);
  for (int row = 0; row < jacobian.rows(); ++row) {
    const int begin = jacobian.outerIndexPtr()[row];
    const int end = jacobian.outerIndexPtr()[row + 1];
    THROW_IF(begin == end);
    const int block = jacobian.innerIndexPtr()[begin] / N;
    THROW_IF(block < 0 || block >= num_blocks);
    for (int left = begin; left < end; ++left) {
      THROW_IF(jacobian.innerIndexPtr()[left] / N != block);
      const int block_row = jacobian.innerIndexPtr()[left] % N;
      const double left_value = jacobian.valuePtr()[left];
      for (int right = begin; right < end; ++right) {
        const int block_col = jacobian.innerIndexPtr()[right] % N;
        blocks[(block * N + block_row) * N + block_col] +=
            left_value * jacobian.valuePtr()[right];
      }
    }
  }

  SparseMatrix<double, RowMajor> result(N * num_blocks, N * num_blocks);
  result.reserve(VectorXi::Constant(N * num_blocks, N));
  for (int block = 0; block < num_blocks; ++block) {
    for (int row = 0; row < N; ++row) {
      for (int col = 0; col < N; ++col) {
        result.insert(block * N + row, block * N + col) =
            blocks[(block * N + row) * N + col];
      }
    }
  }
  result.makeCompressed();
  return result;
}

struct CameraLandmarkEdge {
  int camera;
  int landmark;
  std::array<double, 27> values;
};

class BlockEdgeMatrix {
 public:
  void Initialize(int num_cameras, int num_landmarks,
                  const std::vector<int>& cameras,
                  const std::vector<int>& landmarks) {
    THROW_IF(cameras.size() != landmarks.size());
    num_cameras_ = num_cameras;
    num_landmarks_ = num_landmarks;
    std::vector<int> observation_order(cameras.size());
    std::iota(observation_order.begin(), observation_order.end(), 0);
    std::stable_sort(observation_order.begin(), observation_order.end(),
        [&cameras, &landmarks](int left, int right) {
          return cameras[left] < cameras[right] ||
              (cameras[left] == cameras[right] &&
               landmarks[left] < landmarks[right]);
        });
    edges_.clear();
    edges_.reserve(cameras.size());
    observation_edges_.resize(cameras.size());
    for (int observation : observation_order) {
      if (edges_.empty() || edges_.back().camera != cameras[observation] ||
          edges_.back().landmark != landmarks[observation]) {
        edges_.push_back(
            CameraLandmarkEdge{cameras[observation], landmarks[observation], {}});
      }
      observation_edges_[observation] = edges_.size() - 1;
    }
    for (const CameraLandmarkEdge& edge : edges_) {
      THROW_IF(edge.camera < 0 || edge.camera >= num_cameras_ ||
               edge.landmark < 0 || edge.landmark >= num_landmarks_);
    }
  }

  void ClearValues() {
    for (CameraLandmarkEdge& edge : edges_) {
      edge.values.fill(0.);
    }
  }

  std::array<double, 27>& ObservationValues(int observation) {
    return edges_[observation_edges_[observation]].values;
  }

  int rows() const { return 9 * num_cameras_; }
  int cols() const { return 3 * num_landmarks_; }

  void Multiply(const Eigen::VectorXd& landmark_vector,
                Eigen::VectorXd& camera_result) const {
    THROW_IF(landmark_vector.size() != cols());
    camera_result.setZero(rows());
    for (const CameraLandmarkEdge& edge : edges_) {
      const int camera_offset = 9 * edge.camera;
      const int landmark_offset = 3 * edge.landmark;
      const double landmark0 = landmark_vector[landmark_offset];
      const double landmark1 = landmark_vector[landmark_offset + 1];
      const double landmark2 = landmark_vector[landmark_offset + 2];
      for (int camera_row = 0; camera_row < 9; ++camera_row) {
        camera_result[camera_offset + camera_row] +=
            edge.values[3 * camera_row] * landmark0 +
            edge.values[3 * camera_row + 1] * landmark1 +
            edge.values[3 * camera_row + 2] * landmark2;
      }
    }
  }

  void TransposeMultiply(const Eigen::VectorXd& camera_vector,
                         Eigen::VectorXd& landmark_result) const {
    THROW_IF(camera_vector.size() != rows());
    landmark_result.setZero(cols());
    for (const CameraLandmarkEdge& edge : edges_) {
      const int camera_offset = 9 * edge.camera;
      const int landmark_offset = 3 * edge.landmark;
      for (int landmark_row = 0; landmark_row < 3; ++landmark_row) {
        double sum = 0.;
        for (int camera_row = 0; camera_row < 9; ++camera_row) {
          sum += edge.values[3 * camera_row + landmark_row] *
                 camera_vector[camera_offset + camera_row];
        }
        landmark_result[landmark_offset + landmark_row] += sum;
      }
    }
  }

 private:
  int num_cameras_ = 0;
  int num_landmarks_ = 0;
  std::vector<CameraLandmarkEdge> edges_;
  std::vector<int> observation_edges_;
};

struct NormalEquations {
  SparseMatrix<double, RowMajor> camera_hessian;
  SparseMatrix<double, RowMajor> landmark_hessian;
  Eigen::VectorXd camera_gradient;
  Eigen::VectorXd landmark_gradient;
  Eigen::VectorXd residual;
};


// template<int N>
// void blockAdd(std::vector<double>& dest, std::vector<double>& add) {
//   for (int id = 0; id < dest.size(); ++id) {
//     dest[id] += add[id];
//   }
// }

// template<int N>
// void blockSub(std::vector<double>& dest, std::vector<double>& sub) {
//   for (int id = 0; id < dest.size(); ++id) {
//     dest[id] -= sub[id];
//   }
// }

bool stop_criterion(double x_squared_norm, double gradient_squared_norm, double lip, int i) {
  // lower (1e-4) can be worse? maybe just the parts / how parts are.
  const double eps = 1e-2; //#1e-2 used in paper, tune. might allow smaller as faster?
  const double iterations = i + 1.;
  const double scaled_eps = eps * lip;
  return iterations * iterations * gradient_squared_norm
      < scaled_eps * scaled_eps * x_squared_norm;
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
      observed_x.resize(numResiduals);
      observed_y.resize(numResiduals);
      for (int observation = 0; observation < numResiduals; ++observation) {
        observed_x[observation] = pro.observations(2 * observation);
        observed_y[observation] = pro.observations(2 * observation + 1);
      }
      weighted_cameras.resize(9 * numCameras);
      weighted_landmarks.resize(3 * numLandmarks);
      camera_landmark_hessian.Initialize(
          numCameras, numLandmarks, cam_obs, lm_obs);
      unorm.clear();
      unorm.reserve(9 * numCameras);
      for (const auto &v : pro.unorm()) {
        unorm.push_back(v);
      }
      cameraTransform.clear();
      cameraTransform.reserve(81 * numCameras);
      for (const auto &v : pro.camera_transform()) {
        cameraTransform.push_back(v);
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
      WORKER_LOG(" be set to c_be " << current_be << " s_be:" << start_be<< "\n");
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
      WORKER_LOG(cluster_id << ". Parameter blocks added wo step c:" << cameras.size()
                << " s:" << cameras_s.size() << " l: " << landmarks.size()
            << " | " << numCameras << " " << numLandmarks << "\n");
      for (int i = 0; i < stepSize.size(); i += 81) {
        problem.AddParameterBlock(&stepSize[i], 81);
        problem.SetParameterBlockConstant(&stepSize[i]);
      }

      THROW_IF(unorm.size() != 9 * numCameras);
      THROW_IF(vnorm.size() != 3 * numLandmarks);
      THROW_IF(cameraTransform.size() != 81 * numCameras);
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
        const int camera_id = pro.cam_id(i);
        const int landmark_id = pro.lm_id(i);
        ceres::CostFunction *cost_function = SnavelyReprojectionErrorWeighted::Create(
          pro.observations(2 * i + 0), pro.observations(2 * i + 1),
          &(unorm[9 * camera_id]), &(vnorm[3 * landmark_id]),
          &(cameraTransform[81 * camera_id]));
        problem.AddResidualBlock(cost_function, nullptr /* squared loss */,
                     &(cameras[9 * camera_id]),
                     &(landmarks[3 * landmark_id]));
      }
#endif
      WORKER_LOG("Added " << pro.observations_size() / 2 << " Residual blocks\n");
      function_residual_blocks.clear();
      problem.GetResidualBlocks(&function_residual_blocks);
      normal_equation_evaluate_options.apply_loss_function = true;
      normal_equation_evaluate_options.residual_blocks = function_residual_blocks;
      normal_equation_evaluate_options.num_threads = options.num_threads;
      normal_equation_residuals.reserve(2 * numResiduals);
      normal_equation_camera_blocks.resize(numCameras * 81);
      normal_equation_landmark_blocks.resize(numLandmarks * 9);

      for (int cam_id = 0; cam_id < numCameras; ++cam_id) {
        // double* values = JpJ.valuePtr();
        //  ceres::Matrix block9x9 = Map< Matrix<double,9,9> >
        //  (&(stepSize[cam_id * 9*9])); ceres::Vector block9 = Map<
        //  Matrix<double,9,1> > (&(cameras_s[cam_id * 9]));

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
      WORKER_LOG("Added " << numCameras << " Stepsize Residual blocks\n");
    }

    double GetCost(bool revert_lm = false) {
#ifndef __unweighted_system__
      if (BatchedEvaluationEnabled()) {
        if (revert_lm) {
          std::vector<double> temp_landmarks = landmarks;
          landmarks = best_landmarks;
          const double result = GetBatchedCost();
          landmarks = temp_landmarks;
          return result;
        }
        return GetBatchedCost();
      }
#endif
      // 1st get Jacobian(s):
      ceres::Problem::EvaluateOptions evalOptions;
      evalOptions.apply_loss_function = true;
      // evalOpt.parameter_blocks = {};
      evalOptions.residual_blocks = function_residual_blocks;
      evalOptions.num_threads = options.num_threads;
      double cost;
      if (revert_lm) {
        std::vector<double> temp_landmarks = landmarks;
        landmarks = best_landmarks;
        problem.Evaluate(evalOptions, &cost, nullptr, nullptr, nullptr);
        landmarks = temp_landmarks;
        WORKER_LOG(cluster_id << ". Eval cost with best landmarks: " << 2 * cost << "\n");
      } else {
        problem.Evaluate(evalOptions, &cost, nullptr, nullptr, nullptr);
        WORKER_LOG(cluster_id << ". Eval cost: " << 2 * cost << "\n");
      }

      // if (cost < best_cost) { // by v! not by u. -- not working like this .. we must send GLOBAL best cost signal -> save best lms. now those are individual
      //   best_landmarks = landmarks;
      //   //best_poses = cameras;
      //   best_cost = cost;
      //   std::cout << cluster_id << ". Best cost: " << 2 * cost << "\n";

      //   std::cout << "best poses: ";
      //   for(int i=0; i< 18; ++i)
      //   std::cout << cameras[i] << ", ";
      //   std::cout << "\n";

      //   std::cout << "best lms: ";
      //   for(int i=0; i< 18; ++i)
      //   std::cout << best_landmarks[i] << ", ";
      //   std::cout << "\n";
      // }

      return cost;
    }

#ifndef __unweighted_system__
  void UpdateWeightedParameters() {
      for (int camera = 0; camera < numCameras; ++camera) {
        const int cameraOffset = 9 * camera;
        const int transformOffset = 81 * camera;
        for (int row = 0; row < 9; ++row) {
          double value = 0.;
          for (int col = 0; col < 9; ++col) {
            value += cameraTransform[transformOffset + 9 * row + col]
                     * cameras[cameraOffset + col]
                     * unorm[cameraOffset + col];
          }
          weighted_cameras[cameraOffset + row] = value;
        }
      }
      for (int landmark = 0; landmark < numLandmarks; ++landmark) {
        const int offset = 3 * landmark;
        weighted_landmarks[offset] = landmarks[offset] * vnorm[offset];
        weighted_landmarks[offset + 1] =
            landmarks[offset + 1] * vnorm[offset + 1];
        weighted_landmarks[offset + 2] =
            landmarks[offset + 2] * vnorm[offset + 2];
      }
    }

    double GetBatchedCost() {
      UpdateWeightedParameters();
      double result = 0.;
      for (int observation = 0; observation < numResiduals; ++observation) {
        double residuals[2];
        EvaluateWeightedProjection(
            &weighted_cameras[9 * cam_obs[observation]],
            &weighted_landmarks[3 * lm_obs[observation]],
            observed_x[observation], observed_y[observation], residuals);
        result += 0.5 * (residuals[0] * residuals[0]
                        + residuals[1] * residuals[1]);
      }
      return result;
    }
#endif

    //void SetBe(double be) { be = be; }
    
    return_cluster_proto FillReturnProto() {
      return_cluster_proto return_proto = return_cluster_proto();
      for (const double &v : cameras) {
        //return_proto.add_cameras(static_cast<float>(v));
        return_proto.add_cameras(v);
      }
      THROW_IF(landmarks.size() != vnorm.size());
      for (int landmark_value = 0; landmark_value < landmarks.size();
           ++landmark_value) {
        return_proto.add_landmarks(
        landmarks[landmark_value] * vnorm[landmark_value]);
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
      WORKER_LOG(summary.BriefReport() << "\n"); // .FullReport()
      // TODO: use this trust region size : store and reuse later.
      //std::vector<IterationSummary> Solver::Summary::iterations
      tr_radius = std::min(max_trust_region_radius, summary.iterations.back().trust_region_radius);
      // TODO: -ordering=user for schur (maybe cameras 1st then landmarks)
      cost = summary.final_cost * 2;
      WORKER_LOG(cluster_id << ". Update TR: " << tr_radius << ". be: " << current_be
            << ". Mycost: " << summary.final_cost * 2 << "\n");
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
        WORKER_LOG(cluster_id << ". Revert landmarks\n");
        landmarks = last_landmarks;
      } else if (update.revert_lm() == 2) {
        WORKER_LOG(cluster_id << ". Revert landmarks to best cost lms\n");
        landmarks = best_landmarks; // hmm could be same as last_landmarks.
        //cameras = best_poses;
        //cameras_s = best_poses;

        // Can I fix cameras and compute best landmarks

        // TODO: new idea: No.
        // tr_radius = init_trust_region_radius;
        //tr_radius = 1e-0; // reset as well. TODO. Does nothing for some reason.

        // Get new cost: compare to best cost 
        const double maybe_best_cost = 2 * GetCost(); // demands cameras , landmarks already updated.

        if (maybe_best_cost != best_cost) {
          WORKER_LOG(cluster_id << "\n=============== Best cost changed: " << best_cost << " vs " << maybe_best_cost << " ===================\n");

          WORKER_LOG("best poses: ");
          for(int i=0; i< 18; ++i)
          WORKER_LOG(cameras[i] << ", ");
          WORKER_LOG("\n");
  
          WORKER_LOG("lms: ");
          for(int i=0; i< 18; ++i)
          WORKER_LOG(best_landmarks[i] << ", ");
          WORKER_LOG("\n");
        } else {
          WORKER_LOG(cluster_id << "\n!!!!!!!!!!!!!!!! Best cost matched: " << best_cost << " vs " << maybe_best_cost << " !!!!!!!!!!!!!!!!!!\n");
        }

      }
      else {
        last_landmarks = landmarks;
      }
    }

    // With that Jl changes but it does not matter.
    void UpdateStepSize() { // Recompute.
      const auto [Jp, Jl] = GetJacobian();
      if (firstIteration) {
        const SparseMatrix<double, RowMajor> JlJ =
            BlockDiagonalJtJ<3>(Jl, numLandmarks);
        const auto diag = JlJ.diagonal().array().cwiseAbs().cwiseSqrt().cwiseMax(1e-10);
        WORKER_LOG(" Update vnorm " << cluster_id << " " << diag.size() << " == " << vnorm.size() << "\n");
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
        WORKER_LOG("Preconditioning update " << cluster_id << " " << unorm.size() << " " << vnorm.size() << "\n");
        for (const auto &v : preconditioningProto.unorm()) {
            unorm[id++] = v;
        }
        if (preconditioningProto.camera_transform_size() > 0) {
          THROW_IF(preconditioningProto.camera_transform_size() != cameraTransform.size());
          id = 0;
          for (const auto &v : preconditioningProto.camera_transform()) {
            cameraTransform[id++] = v;
          }
        }
        // id = 0; // fill existing buffer
        // for (const float &v : preconditioningProto.vnorm()) {
        //     vnorm[id++] = v;
        // }
    }

    void UpdateBestCost(best_cost_proto ppro) {
      THROW_IF(ppro.cluster_id() != cluster_id);
      WORKER_LOG("Best cost update " << cluster_id << " " << ppro.cost() << " < " << best_cost << "\n");
//      if (ppro.cost() < best_cost) {
        best_cost = ppro.cost();
        best_landmarks = landmarks;
        //new_best_cost = true;
//      }
    }

////////////////////////////////////////
// new stuff for self optimization

void UpdatePreconditioningCameras(SparseMatrix<double, RowMajor> JpJ) {
  full_stepSize.resize(81 * numCameras, 0);
  const double *values = JpJ.valuePtr();
  std::copy(values, values + full_stepSize.size(), full_stepSize.data());
  // ToDo: Is this ok or an issue to be resolved differently?
  for (int b = 0; b < numCameras; ++b) {
    for(int id = 0; id < 81; id += 10) { // diagonal entries !?
      full_stepSize[81*b + id] = std::max(1e-36, full_stepSize[81*b + id]);
    }
  }
}

std::pair<Matrix<double, Eigen::Dynamic, 1>, Matrix<double, Eigen::Dynamic, 1>>
SolveByGDNesterov(SparseMatrix<double, RowMajor> Uli, SparseMatrix<double, RowMajor> Vli, 
                const BlockEdgeMatrix& W,
                const Matrix<double, Eigen::Dynamic, 1>& bp,
                const Matrix<double, Eigen::Dynamic, 1>& bl,
                const Matrix<double, Eigen::Dynamic, 1>& proximalGradient,
                int power_iterations) {

  // compute bS, Vli, W
  BlockInverse<3>(Vli);

  if (power_iterations == 0) {  // quick hack: xk = delta_p = 0
    Matrix<double, Eigen::Dynamic, 1> ubs = Uli * bp;
    Matrix<double, Eigen::Dynamic, 1> xk = 0 * ubs;
    Matrix<double, Eigen::Dynamic, 1> delta_l = Vli * (-bl);
    return {xk, delta_l};
   }

  BlockInverse<9>(Uli);
  const double Lip = 0.9;
  double lambda0 = (1. + std::sqrt(5.)) / 2.;
  Matrix<double, Eigen::Dynamic, 1> bS = bp;
  // bS = (bp_s                     - W * Vli * bl).flatten() # see XX equals 2 * (bp - W * Vli * bl)
  //       bp_s = bp + stepSize * prox_rhs
  // bS = (bp + stepSize * prox_rhs - W * Vli * bl).flatten() # see XX equals 2 * (bp - W * Vli * bl)
  bS += proximalGradient;
  Eigen::VectorXd landmarkWorkspace = Vli * bl;
  Eigen::VectorXd cameraWorkspace(W.rows());
  W.Multiply(landmarkWorkspace, cameraWorkspace);
  bS -= cameraWorkspace;

  // std::cout << " Jl " << Jl.valuePtr()[0] << " " << Jl.valuePtr()[1] << " " << Jl.valuePtr()[2] << "\n";
  // std::cout << "bS :" << bS.array() << "\n";
  // std::cout << "bp :" << (Jp.transpose() * res).array() << "\n";
  // std::cout << "bl :" << (Jl.transpose() * res).array() << "\n";
  // std::cout << "res :" << res.array() << "\n"; //ok

  Matrix<double, Eigen::Dynamic, 1> ubs = -Uli * bS;
  // Todo : * 1. / Lip ? or not
  Matrix<double, Eigen::Dynamic, 1> xk = - 1. / Lip * ubs; // xk =0, g = ubs, yk = -1. / Lip * g = - 1. / Lip * ubs; xk = (1-gamma) yk + gamma y0, gamma = 0
  Matrix<double, Eigen::Dynamic, 1> y0 = - 1. / Lip * ubs; // xk =0, g = ubs, yk = -1. / Lip * g = - 1. / Lip * ubs; y0 = yk.
  Matrix<double, Eigen::Dynamic, 1> wtX(W.cols());
  Matrix<double, Eigen::Dynamic, 1> vinvWtX(W.cols());
  Matrix<double, Eigen::Dynamic, 1> wVinvWtX(W.rows());
  Matrix<double, Eigen::Dynamic, 1> uinvWVinvWtX(W.rows());
  Matrix<double, Eigen::Dynamic, 1> g(W.rows());
  Matrix<double, Eigen::Dynamic, 1> yk(W.rows());
  // Lip = 0.9 # 100 -> 1. # TODO: play, find out how to progress over time.
  // lambda0 = (1.+np.sqrt(5.)) / 2. # l=0 g=1, 0, .. L0=1 g = 0,..

  // std::cout << "xk :" << xk.squaredNorm() << "\n";

  for (int i = 0; i < power_iterations; ++i) {
      const double lambda1 = (1. + std::sqrt(1. + 4. * lambda0 * lambda0)) / 2.;
      const double gamma = (1. - lambda0) / lambda1;
      lambda0 = lambda1;

      //     g = xk - Uli * ( W * (Vli * (W.transpose() * xk))) + ubs
      //     yk = xk - 1/Lip * g
      //     xk = (1-gamma) * yk + gamma * y0
      //     y0 = yk
      // const Matrix<double, Eigen::Dynamic, 1> g = (xk - Uli * (W * (Vli * (W.transpose() * xk).eval()).eval()).eval() + ubs).eval();
      W.TransposeMultiply(xk, wtX);
      vinvWtX.noalias() = Vli * wtX;
      W.Multiply(vinvWtX, wVinvWtX);
      uinvWVinvWtX.noalias() = Uli * wVinvWtX;
      g = xk;
      g -= uinvWVinvWtX;
      g += ubs;
      yk = xk;
      yk -= 1. / Lip * g;
      xk = (1. - gamma) * yk + gamma * y0;
      y0 = yk;

      //std::cout << i << ". xk :" << xk.squaredNorm() << "\n";

        if (stop_criterion(xk.squaredNorm(), g.squaredNorm(), Lip, i)) {
          break;
      }
  }

  W.TransposeMultiply(xk, wtX);
  Matrix<double, Eigen::Dynamic, 1> delta_l = Vli * (wtX - bl);
  return {-xk, delta_l};
}

void UpdateStepSizeAndSolve() {//bool keep_cameras_fixed = false) { // Recompute.
  // if (keep_cameras_fixed) {
  //   keep_cameras_fixed = new_best_cost;
  // }

  const bool collectTiming = LocalSolveMetricsEnabled();
  const auto localSolveStart = collectTiming ? TimingClock::now() : TimingClock::time_point{};
  double nesterovSeconds = 0.;
  double costEvaluationSeconds = 0.;
  NormalEquations normalEquations = GetNormalEquations();
  const auto assemblyStart = collectTiming ? TimingClock::now() : TimingClock::time_point{};
  const Eigen::VectorXd& residual = normalEquations.residual;
  SparseMatrix<double, RowMajor> Vl = normalEquations.landmark_hessian;
  if (firstIteration) { // preconditioning
      // diag is a reference .. why? i do stuff on it.
      const auto diag = Vl.diagonal().array().cwiseMax(1e-24).cwiseSqrt().cwiseInverse().eval();
      THROW_IF(diag.size() != vnorm.size());
      WORKER_LOG(" Update vnorm " << cluster_id << " " << diag.size() << " == " << vnorm.size() << "\n");
      for (int id = 0; id < vnorm.size(); ++id) {
          landmarks[id] /= diag[id];
          vnorm[id] = diag(id);
      }
      best_landmarks = landmarks;
      // Update Vl as well. we return below in 1st it, so not needed.
      // std::cout << " Jl " << Jl.valuePtr()[0] << " " << Jl.valuePtr()[1] << " " << Jl.valuePtr()[2] << "\n";
      // Jl = (Jl * diag.matrix().asDiagonal()).eval(); // This does not happen as diag is diag of Vl. That gets changed. diag is not copied but reference.
      // std::cout << " Jl " << Jl.valuePtr()[0] << " " << Jl.valuePtr()[1] << " " << Jl.valuePtr()[2] << "\n";
      // Vl = diag.matrix().asDiagonal() * Vl * diag.matrix().asDiagonal();
  }
  
  // std::cout << " diagVL " << diagVL.diagonal() << "\n"; // 1's
  
  // JpJ, StepSize, diag JpJ
  SparseMatrix<double, RowMajor> Ul = normalEquations.camera_hessian;
  if (firstIteration) { // also handled setting be = 0 in 1st step.
    UpdatePreconditioningCameras(Ul);
    // Debug: write cost
    const auto costEvaluationStart = collectTiming ? TimingClock::now() : TimingClock::time_point{};
    const double costEnd = 2 * GetCost(); // demands cameras , landmarks already updated.
    if (collectTiming) {
      costEvaluationSeconds += ElapsedSeconds(costEvaluationStart);
    }
    cost = costEnd;
    WORKER_LOG(cluster_id << ". Sending no update but pcg. costStart == costend: " << residual.squaredNorm() << " == "  << costEnd << "\n");
    if (collectTiming) {
      std::ostringstream metric;
      metric << "LOCAL_SOLVE_TIMING cluster=" << cluster_id
         << " initial=1"
         << " jacobian_evaluate=" << last_jacobian_evaluate_seconds
         << " jacobian_conversion=" << last_jacobian_conversion_seconds
         << " assembly=" << ElapsedSeconds(assemblyStart) - costEvaluationSeconds
         << " nesterov=0"
         << " cost_evaluate=" << costEvaluationSeconds
         << " total=" << ElapsedSeconds(localSolveStart)
         << " attempts=0\n";
      EmitLocalSolveMetric(metric.str());
    }
    return; // 1st step only preconditioning as it can go very wrong?
  }
  const Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagVL = Diagonal<3>(Vl); // Vl = VL + L * diagVL
  const Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagUP = 1e1 * Diagonal<9>(Ul, cluster_id); // Vp = Vp + L * diagVp
  const SparseMatrix<double, RowMajor>& cameraHessian = normalEquations.camera_hessian;
  const SparseMatrix<double, RowMajor>& landmarkHessian = normalEquations.landmark_hessian;

  //const double scale = 1e-1; // 1e0: @29: 501k, no jump. 1e1 many jumps. 473k
  // TODO
  const double scale = std::min(1.005, 1e-1 * std::sqrt(current_be / start_be));

  if (!firstIteration) { // also handled setting be = 0 in 1st step.
    SparseMatrix<double, RowMajor> stepSize = scale * Ul;
    stepSize += diagUP * current_be;
    const double* values = stepSize.valuePtr();
    std::copy(values, values + full_stepSize.size(), full_stepSize.data()); 
    Ul += stepSize;
  } else {
    Ul += scale * Ul;
    Ul += diagUP * current_be;
    // let full_Stepsize define setpsize always. else confusing to debug: cost optimized differs from cost evaluated.
    // const SparseMatrix<double, RowMajor> stepSize = Ul;
    // Ul += stepSize;
  }
  // Loop until ok or adjust tr_region
  tr_radius = std::min(max_trust_region_radius, tr_radius);
  double inv_tr_radius = 0;

  const int power_iterations = 100; //keep_cameras_fixed ? 0 : 100;
  const double costStart = residual.squaredNorm();
  const BlockEdgeMatrix& W = camera_landmark_hessian;
  const Eigen::VectorXd& bp = normalEquations.camera_gradient;
  const Eigen::VectorXd& bl = normalEquations.landmark_gradient;
  Matrix<double, Eigen::Dynamic, 1> proximalOffset(9 * numCameras);
  for (int id = 0; id < proximalOffset.size(); ++id) {
    proximalOffset[id] = cameras[id] - cameras_s[id];
  }
  Matrix<double, Eigen::Dynamic, 1> proximalGradient(9 * numCameras);
  blockMult<9>(full_stepSize, proximalOffset, proximalGradient);
  const double penaltyStart = proximalOffset.dot(proximalGradient);
  Matrix<double, Eigen::Dynamic, 1> proximalStep(9 * numCameras);
  Matrix<double, Eigen::Dynamic, 1> crossProduct(9 * numCameras);
  const double assemblySeconds = collectTiming ? ElapsedSeconds(assemblyStart) : 0.;
  int trust_region_attempts = 0;
  int trust_region_rejections = 0;
  // options.max_num_iterations 
  while ( true ) { // if costStart + penaltyStart < costEnd + penaltyP
    ++trust_region_attempts;

    //   std::cout << " diagUP " << Ul.diagonal()[0] << " " << Ul.diagonal()[1] << " " << Ul.diagonal()[2] << "\n";
    //   std::cout << " diagVL " << Vl.diagonal()[0] << " " << Vl.diagonal()[1] << " " << Vl.diagonal()[2] << "\n";// TOTALLY OFF after tr_check fails.

    // if not complicated this will lead to total chaos, likely the 
    Ul += (1. / tr_radius - inv_tr_radius) * (1e-4 * diagUP);// + Jp.transpose() * Jp);
    Ul += cameraHessian * (1. / tr_radius - inv_tr_radius);

    if (inv_tr_radius != 0) {
      Vl = landmarkHessian;
    }
    Vl *= 1. + 1. / tr_radius;
    Vl += (1. / tr_radius) * diagVL;
    inv_tr_radius = 1. / tr_radius;
    //   std::cout << " diagUp " << Ul.diagonal()[0] << " " << Ul.diagonal()[1] << " " << Ul.diagonal()[2] << "\n";
    //   std::cout << " diagVL " << Vl.diagonal()[0] << " " << Vl.diagonal()[1] << " " << Vl.diagonal()[2] << "\n";

    //std::cout << " VL " << Vl.diagonal() << "\n";

    const auto nesterovStart = collectTiming ? TimingClock::now() : TimingClock::time_point{};
    const auto [delta_p, delta_l] = SolveByGDNesterov(
      Ul, Vl, W, bp, bl, proximalGradient, power_iterations);
    if (collectTiming) {
      nesterovSeconds += ElapsedSeconds(nesterovStart);
    }
    // compute cost / tr_check
    //fx0_new = fx0 + (J_pose * delta_p + J_land * delta_l)
    W.Multiply(delta_l, crossProduct);
    const double costQuad = costStart
      + 2. * bp.dot(delta_p)
      + 2. * bl.dot(delta_l)
      + delta_p.dot(cameraHessian * delta_p)
      + 2. * delta_p.dot(crossProduct)
      + delta_l.dot(landmarkHessian * delta_l);
    // const double costQuad2 = (residual - Jp * delta_p - Jl * delta_l).squaredNorm();
    // const double costQuad3 = (residual - Jp * delta_p + Jl * delta_l).squaredNorm();
    // const double costQuad4 = (residual + Jp * delta_p - Jl * delta_l).squaredNorm();
    // This is wrong: costQuad cannot be greater costStart by definition. it can if s is in the wrong direction. 
    // yet then penalties should have changed as well?
    // std::cout << costStart << " > " << costQuad << " " << costQuad2 << " " << costQuad3 << " " << costQuad4 << "\n";

    // std::cout << "res/dl/dp :" << residual.squaredNorm() << " " << delta_p.squaredNorm() << " " << delta_l.squaredNorm() << "\n";
    // Map<Matrix<double, Eigen::Dynamic, 1> >(blockMult<9>(full_stepSize, cameras).data());

    blockMult<9>(full_stepSize, delta_p, proximalStep);
    const double penaltyEnd = penaltyStart
        + 2. * delta_p.dot(proximalGradient)
        + delta_p.dot(proximalStep);
    //const double penaltyEnd2 = (Jp * prox_rhs).squaredNorm();

    //SparseMatrix<double, RowMajor> Ul_(9 * numCameras, 9 * numCameras);
    //Ul_.reserve(VectorXi::Constant(9 * numCameras, 9));
    //Ul_ = Jp.transpose() * Jp;
    // const double penaltyEnd6 = prox_rhs.dot( Ul_ * prox_rhs );
    //std::cout << "==Penalties end/end2: " << penaltyEnd << " ?= " << penaltyEnd2 << " == " << penaltyEnd6 << "\n"; // since full_step differs from Jp cna differ  
    
    // Needs to be done due to GetCost.
    for (int id = 0; id < delta_p.size(); ++id) {
        cameras[id] += delta_p[id];
    }
    for (int id = 0; id < delta_l.size(); ++id) {
        landmarks[id] += delta_l[id];
    }
    const auto costEvaluationStart = collectTiming ? TimingClock::now() : TimingClock::time_point{};
    const double costEnd = 2 * GetCost(); // demands cameras , landmarks already updated.
    if (collectTiming) {
      costEvaluationSeconds += ElapsedSeconds(costEvaluationStart);
    }
    WORKER_LOG("==Costs start/quad/end: " << costStart << " " << costQuad << " " << costEnd << "\n");
    WORKER_LOG("==Penalties start/end: " << penaltyStart << " " << penaltyEnd << "\n");

    const double tr_check = (costStart - costEnd + penaltyStart - penaltyEnd) / std::max(0.1, costStart - costQuad + penaltyStart - penaltyEnd);
    if (tr_check < 0.25) {
      tr_radius /= 2;
      WORKER_LOG(tr_check << ": decrease TR radius " << tr_radius << "\n");
    }
    if (tr_check > 0.8) {
      tr_radius = std::min(max_trust_region_radius, 2 * tr_radius);//1.5
      WORKER_LOG(tr_check << ": increase TR radius " << tr_radius << "\n");
    }

    if (costStart + penaltyStart > costQuad + penaltyEnd)
      WORKER_LOG("==Start Cost < estimated cost: " << costStart + penaltyStart << " < " << costQuad + penaltyEnd << "\n");

    if (costStart + penaltyStart <
        (costEnd + penaltyEnd) * LocalAcceptanceRatio()) { // revert if cost does not improve
      ++trust_region_rejections;
      WORKER_LOG("Reject Start Cost < end cost: "<< costStart + penaltyStart << " < " << costEnd + penaltyEnd << "\n");
      for (int id = 0; id < delta_p.size(); ++id) {
          cameras[id] -= delta_p[id];
      }
      for (int id = 0; id < delta_l.size(); ++id) {
        landmarks[id] -= delta_l[id];
      }
      continue;
    }
    cost = costEnd;
    WORKER_LOG("Accept Start Cost > end cost: "<< costStart + penaltyStart << " > " << costEnd + penaltyEnd << "\n");
    if (LocalSolveMetricsEnabled()) {
      std::ostringstream metric;
      metric << "LOCAL_SOLVE cluster=" << cluster_id
         << " be=" << current_be
         << " attempts=" << trust_region_attempts
         << " rejections=" << trust_region_rejections
         << " start=" << costStart + penaltyStart
         << " end=" << costEnd + penaltyEnd
         << " predicted=" << costStart - costQuad + penaltyStart - penaltyEnd
         << " rho=" << tr_check
         << " radius=" << tr_radius
         << " diagonal_floor=" << CameraDiagonalRelativeFloor()
         << " acceptance_ratio=" << LocalAcceptanceRatio() << "\n";
      EmitLocalSolveMetric(metric.str());
      std::ostringstream timingMetric;
      timingMetric << "LOCAL_SOLVE_TIMING cluster=" << cluster_id
        << " initial=0"
        << " jacobian_evaluate=" << last_jacobian_evaluate_seconds
        << " jacobian_conversion=" << last_jacobian_conversion_seconds
        << " assembly=" << assemblySeconds
        << " nesterov=" << nesterovSeconds
        << " cost_evaluate=" << costEvaluationSeconds
        << " total=" << ElapsedSeconds(localSolveStart)
        << " attempts=" << trust_region_attempts << "\n";
      EmitLocalSolveMetric(timingMetric.str());
    }

    // if (keep_cameras_fixed) {
    //   best_landmarks = landmarks; // landmarks optimal for fixed cameras.
    //   best_cost = costEnd;
    //   new_best_cost = false; // internal true if new est cost found -> new global cameras, landmarks not optimized for those.
    // }
  
    break;
  }
}
///////////////////////////////////

private:

  // Currently this is set 'stepsize' from Jp only.
  void SetStepSize(const SparseMatrix<double, RowMajor> &Jp) {
    // std::cout << "Set step size " << cluster_id << "\n";
    if(Jp.nonZeros() != 9 * Jp.rows())
        std::cout << "Jp " << cluster_id << " | " << Jp.nonZeros() << " =? " << Jp.rows() * 9 << "\n";
    THROW_IF(Jp.nonZeros() != 9 * Jp.rows());
    SparseMatrix<double, RowMajor> JpJ = BlockDiagonalJtJ<9>(Jp, numCameras);

    // SparseMatrix<double, RowMajor> JlJ(3 * numLandmarks, 3 * numLandmarks);
    // JlJ.reserve(VectorXi::Constant(3 * numLandmarks, 3));
    // JlJ = Jl.transpose() * Jl;
    // auto JpJ_diag = JpJ.diagonal().array();
    if(JpJ.nonZeros() != stepSize.size() || JpJ.rows() * 9 != numCameras * 81)
      std::cout << "JpJ " << cluster_id << " | " << JpJ.nonZeros() << " =? " << JpJ.rows() * 9
                << " " << stepSize.size() << " " << numCameras * 81 << "\n";
    THROW_IF(JpJ.nonZeros() != numCameras * 81);

    if (firstIteration) { // also handled setting be = 0 in 1st step.
      full_stepSize.resize(81 * numCameras, 0);
      const double *values = JpJ.valuePtr();
      std::copy(values, values + full_stepSize.size(), full_stepSize.data());
      // ToDo: Is this ok or an issue to be resolved differently?
      for (int b = 0; b < numCameras; ++b) {
        for(int id = 0; id < 81; id += 10) { // diagonal entries !?
          full_stepSize[81*b + id] = std::max(1e-36, full_stepSize[81*b + id]);
        }
      }
      best_landmarks = landmarks; // !
      //best_poses = cameras; // ? 
      //return; // 1st iteration only preconditioning. do not solve! needs other stuff do be dones below.
    }

    // Allow to scale JtJ as well?
    const double scale = 1e1;
    // TODO.
    //const double scale = std::max(1. / 1.005, 1e1 * std::sqrt(start_be / current_be)); // 1e0: @29: 501k, no jump. 1e1 many jumps. 473k
    WORKER_LOG("scale " << scale << " " << start_be << " " << current_be << "\n");
    JpJ = JpJ * (1. / scale); // optional to test. in theory should almost always suffice.
#ifdef _const_diag_
    auto diag = JpJ.diagonal().array();
#pragma omp parallel for num_threads(options.num_threads)
    for (int b = 0; b < numCameras; ++b) { // block
      double mv = diag(9*b);
      for (int id = 1; id < 9; ++id) {
        mv = std::max(mv, diag(9*b + id));
      }
      mv *= scale;
      for (int id = 0; id < 9; ++id) {
        diag(9*b + id) += current_be * std::max(mv * 1e-3, diag(9*b + id));
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
      full_stepSize.resize(81 * numCameras, 0);
      const double *values = JpJ.valuePtr();
      std::copy(values, values + full_stepSize.size(), full_stepSize.data());
    }

    // std::cout << "BlockSqrt " << cluster_id << "\n";
    BlockSqrt<9>(JpJ); // need templated fct.
    // instead reset variable block(s) JpJ and s to sqrt(Stepsize)
    const double* values = JpJ.valuePtr();
    std::copy(values, values + stepSize.size(), stepSize.data());
  }

#ifndef __unweighted_system__
  NormalEquations GetBatchedNormalEquations() {
    using ObservationJet = ceres::Jet<double, 12>;
    const bool collectTiming = LocalSolveMetricsEnabled();
    const auto evaluateStart = collectTiming ? TimingClock::now() : TimingClock::time_point{};
    UpdateWeightedParameters();
    std::fill(normal_equation_camera_blocks.begin(),
              normal_equation_camera_blocks.end(), 0.);
    std::fill(normal_equation_landmark_blocks.begin(),
              normal_equation_landmark_blocks.end(), 0.);
    camera_landmark_hessian.ClearValues();
    normal_equation_residuals.resize(2 * numResiduals);

    NormalEquations result;
    result.camera_gradient = Eigen::VectorXd::Zero(9 * numCameras);
    result.landmark_gradient = Eigen::VectorXd::Zero(3 * numLandmarks);
    double squaredResidualNorm = 0.;

    for (int observation = 0; observation < numResiduals; ++observation) {
      const int cameraId = cam_obs[observation];
      const int landmarkId = lm_obs[observation];
      const int cameraOffset = 9 * cameraId;
      const int landmarkOffset = 3 * landmarkId;
      const int transformOffset = 81 * cameraId;
      ObservationJet camera[9];
      ObservationJet landmark[3];
      for (int row = 0; row < 9; ++row) {
        camera[row].a = weighted_cameras[cameraOffset + row];
        camera[row].v.setZero();
        for (int col = 0; col < 9; ++col) {
          camera[row].v[col] =
              cameraTransform[transformOffset + 9 * row + col]
              * unorm[cameraOffset + col];
        }
      }
      for (int row = 0; row < 3; ++row) {
        landmark[row].a = weighted_landmarks[landmarkOffset + row];
        landmark[row].v.setZero();
        landmark[row].v[9 + row] = vnorm[landmarkOffset + row];
      }

      ObservationJet residuals[2];
      EvaluateWeightedProjection(camera, landmark, observed_x[observation],
                                 observed_y[observation], residuals);
      std::array<double, 27>& edgeValues =
          camera_landmark_hessian.ObservationValues(observation);
      for (int component = 0; component < 2; ++component) {
        const double residual = residuals[component].a;
        normal_equation_residuals[2 * observation + component] = residual;
        squaredResidualNorm += residual * residual;
        for (int row = 0; row < 9; ++row) {
          const double cameraValue = residuals[component].v[row];
          result.camera_gradient[cameraOffset + row] += cameraValue * residual;
          for (int col = 0; col < 9; ++col) {
            normal_equation_camera_blocks[81 * cameraId + 9 * row + col] +=
                cameraValue * residuals[component].v[col];
          }
          for (int col = 0; col < 3; ++col) {
            edgeValues[3 * row + col] +=
                cameraValue * residuals[component].v[9 + col];
          }
        }
        for (int row = 0; row < 3; ++row) {
          const double landmarkValue = residuals[component].v[9 + row];
          result.landmark_gradient[landmarkOffset + row] +=
              landmarkValue * residual;
          for (int col = 0; col < 3; ++col) {
            normal_equation_landmark_blocks[9 * landmarkId + 3 * row + col] +=
                landmarkValue * residuals[component].v[9 + col];
          }
        }
      }
    }
    startCost = 0.5 * squaredResidualNorm;
    last_jacobian_evaluate_seconds = collectTiming ? ElapsedSeconds(evaluateStart) : 0.;
    const auto conversionStart = collectTiming ? TimingClock::now() : TimingClock::time_point{};
    result.residual = Eigen::Map<Eigen::VectorXd>(normal_equation_residuals.data(),
                            normal_equation_residuals.size());

    result.camera_hessian.resize(9 * numCameras, 9 * numCameras);
    result.camera_hessian.reserve(VectorXi::Constant(9 * numCameras, 9));
    for (int cameraId = 0; cameraId < numCameras; ++cameraId) {
      for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
          result.camera_hessian.insert(9 * cameraId + row, 9 * cameraId + col) =
              normal_equation_camera_blocks[81 * cameraId + 9 * row + col];
        }
      }
    }
    result.camera_hessian.makeCompressed();

    result.landmark_hessian.resize(3 * numLandmarks, 3 * numLandmarks);
    result.landmark_hessian.reserve(VectorXi::Constant(3 * numLandmarks, 3));
    for (int landmarkId = 0; landmarkId < numLandmarks; ++landmarkId) {
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          result.landmark_hessian.insert(3 * landmarkId + row,
                                         3 * landmarkId + col) =
              normal_equation_landmark_blocks[9 * landmarkId + 3 * row + col];
        }
      }
    }
    result.landmark_hessian.makeCompressed();
    last_jacobian_conversion_seconds =
        collectTiming ? ElapsedSeconds(conversionStart) : 0.;
    return result;
  }
#endif

  NormalEquations GetNormalEquations() {
#ifndef __unweighted_system__
    if (BatchedEvaluationEnabled()) {
      return GetBatchedNormalEquations();
    }
#endif
    // 1st get Jacobian(s):
    // std::cout << "GetJacobian: Evaluate " << cluster_id << "\n"; 
    const bool collectTiming = LocalSolveMetricsEnabled();
    const auto evaluateStart = collectTiming ? TimingClock::now() : TimingClock::time_point{};
    problem.Evaluate(normal_equation_evaluate_options, &startCost,
                     &normal_equation_residuals, nullptr,
                     &normal_equation_jacobian);
    last_jacobian_evaluate_seconds = collectTiming ? ElapsedSeconds(evaluateStart) : 0.;
    const auto conversionStart = collectTiming ? TimingClock::now() : TimingClock::time_point{};
    THROW_IF(normal_equation_jacobian.num_rows != 2 * numResiduals);
    THROW_IF(normal_equation_residuals.size() != normal_equation_jacobian.num_rows);
    std::fill(normal_equation_camera_blocks.begin(),
              normal_equation_camera_blocks.end(), 0.);
    std::fill(normal_equation_landmark_blocks.begin(),
              normal_equation_landmark_blocks.end(), 0.);
    camera_landmark_hessian.ClearValues();
    NormalEquations result;
    result.camera_gradient = Eigen::VectorXd::Zero(9 * numCameras);
    result.landmark_gradient = Eigen::VectorXd::Zero(3 * numLandmarks);
    result.residual = Eigen::Map<Eigen::VectorXd>(normal_equation_residuals.data(),
                                                  normal_equation_residuals.size());

    for (int observation = 0; observation < numResiduals; ++observation) {
      const int lm_id = lm_obs[observation];
      const int cam_id = cam_obs[observation];
        std::array<double, 27>& edgeValues =
          camera_landmark_hessian.ObservationValues(observation);
      for (int component = 0; component < 2; ++component) {
        const Eigen::Index row = 2 * observation + component;
        const Eigen::Index begin = normal_equation_jacobian.rows[row];
        const Eigen::Index end = normal_equation_jacobian.rows[row + 1];
        THROW_IF(end - begin != 12);
        const double residual = normal_equation_residuals[row];
        for (int i = 0; i < 9; ++i) {
          const double cameraValue = normal_equation_jacobian.values[begin + i];
          result.camera_gradient[9 * cam_id + i] += cameraValue * residual;
          for (int j = 0; j < 9; ++j) {
            normal_equation_camera_blocks[81 * cam_id + 9 * i + j] +=
                cameraValue * normal_equation_jacobian.values[begin + j];
          }
          for (int j = 0; j < 3; ++j) {
            edgeValues[3 * i + j] +=
                cameraValue * normal_equation_jacobian.values[begin + 9 + j];
          }
        }
        for (int i = 0; i < 3; ++i) {
          const double landmarkValue = normal_equation_jacobian.values[begin + 9 + i];
          result.landmark_gradient[3 * lm_id + i] += landmarkValue * residual;
          for (int j = 0; j < 3; ++j) {
            normal_equation_landmark_blocks[9 * lm_id + 3 * i + j] +=
                landmarkValue * normal_equation_jacobian.values[begin + 9 + j];
          }
        }
      }
    }

    result.camera_hessian.resize(9 * numCameras, 9 * numCameras);
    result.camera_hessian.reserve(VectorXi::Constant(9 * numCameras, 9));
    for (int camera = 0; camera < numCameras; ++camera) {
      for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
          result.camera_hessian.insert(9 * camera + row, 9 * camera + col) =
              normal_equation_camera_blocks[81 * camera + 9 * row + col];
        }
      }
    }
    result.camera_hessian.makeCompressed();

    result.landmark_hessian.resize(3 * numLandmarks, 3 * numLandmarks);
    result.landmark_hessian.reserve(VectorXi::Constant(3 * numLandmarks, 3));
    for (int landmark = 0; landmark < numLandmarks; ++landmark) {
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          result.landmark_hessian.insert(3 * landmark + row, 3 * landmark + col) =
              normal_equation_landmark_blocks[9 * landmark + 3 * row + col];
        }
      }
    }
    result.landmark_hessian.makeCompressed();

    last_jacobian_conversion_seconds = collectTiming ? ElapsedSeconds(conversionStart) : 0.;
    return result;
  }

  // Also delivers residuals and gradient.
  std::pair<SparseMatrix<double, RowMajor>,
            SparseMatrix<double, RowMajor>>
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
    SparseMatrix<double, RowMajor> Jp(relevantRows, 9 * numCameras);
    SparseMatrix<double, RowMajor> Jl(relevantRows, 3 * numLandmarks);
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
      //new_best_cost = false;
      current_be = init_be;
      start_be = init_be;
      tr_radius = std::min(max_trust_region_radius, init_trust_region_radius);
      startCost = 1e20;
      cost = 1e20;
      best_cost = cost;
      last_jacobian_evaluate_seconds = 0.;
      last_jacobian_conversion_seconds = 0.;
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
      Eigen::setNbThreads(threads_per_cluster);
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
  double last_jacobian_evaluate_seconds = 0.;
  double last_jacobian_conversion_seconds = 0.;
  bool firstIteration = true; // full step is wo. diag part to acc.
  //bool new_best_cost = false;
  std::vector<ceres::ResidualBlockId> function_residual_blocks;
  std::vector<double> cameras;
  //std::vector<double> best_poses;
  std::vector<double> cameras_s;
  std::vector<double> landmarks;// todo: either revert or send landmarkss all the time.
  std::vector<double> last_landmarks;
  std::vector<double> best_landmarks;
  std::vector<double> stepSize; // internally modelling prox term. 'sqrt' of full_stepSize 
  std::vector<double> full_stepSize; // returned to compute s update in DRS.
  std::vector<double> unorm;
  std::vector<double> vnorm;
  std::vector<double> cameraTransform;
  std::vector<int> cam_obs;
  std::vector<int> lm_obs;
  std::vector<double> observed_x;
  std::vector<double> observed_y;
  std::vector<double> weighted_cameras;
  std::vector<double> weighted_landmarks;
  BlockEdgeMatrix camera_landmark_hessian;
  ceres::Problem::EvaluateOptions normal_equation_evaluate_options;
  ceres::CRSMatrix normal_equation_jacobian;
  std::vector<double> normal_equation_residuals;
  std::vector<double> normal_equation_camera_blocks;
  std::vector<double> normal_equation_landmark_blocks;
  ceres::Problem problem;
  ceres::Solver::Options options;
};
///////////////////////////////////////////////////////

int main() {
  // Initialize the context
  zmq::context_t context(1);

  zmq::socket_t push_socket(context, ZMQ_PUSH);
  zmq::socket_t pull_socket(context, ZMQ_PULL);

  std::mutex mtx; // Mutex for critical section.

  // Bind the socket to a TCP address
  std::cout << "Starting the server on ports 5556 and 5557..." << std::endl;
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
    // ParseFromString(value) is the same as ParseFromArray(value.data(),
    // value.size()).

    // const std::string received_message(static_cast<char*>(request.data()),
    // request.size());
    request_proto request_p;
    // request_p.ParseFromString(received_message);
    request_p.ParseFromArray(request.data(), request.size());
    // std::cout << "Request ParseFromArray\n";

    switch (request_p.options_case()) {

    case request_proto::OptionsCase::kUpdate: {
      const prox_cluster_proto& update =
          request_p.update(); // we get an update for the cameras only -- update
                              // buffer, run its iterations.
      const int cluster_id = update.cluster_id();
      WORKER_LOG("request_proto::OptionsCase::kUpdate " << cluster_id
            << std::endl);
      THROW_IF(cluster_to_program.find(cluster_id) == cluster_to_program.end());
      CeresProgram &program = cluster_to_program[cluster_id];

      //bool keep_cameras_fixed = (update.revert_lm() == 2) ? true : false;

      program.UpdateData(update); // update is local, we nned to fill data in main thread.

      // Define a Lambda Expression
      auto update_lambda = [&push_socket, &cluster_to_program,
                &mtx](int cluster_id, std::uint64_t run_id,
                  std::uint64_t phase_id) {
        CeresProgram &program = cluster_to_program[cluster_id];
        // std::cout << cluster_id << " Update "<< "\n";
#ifdef __ceresVersion__
        program.UpdateStepSize();
        program.Solve();
#else
        program.UpdateStepSizeAndSolve();//keep_cameras_fixed);
#endif
        return_cluster_proto return_proto = program.FillReturnProto();
        return_proto.set_run_id(run_id);
        return_proto.set_phase_id(phase_id);
        const double cost = return_proto.cost();
        WORKER_LOG(cluster_id << ". Cost from update: " << cost << "\n");
        // std::cout << "Cost from update " << cost <<"\n";
        //  SerializeToArray saves memory and time?
        size_t bytes = return_proto.ByteSizeLong();
        zmq::message_t reply(bytes);
        return_proto.SerializeToArray(reply.data(), bytes);
        std::lock_guard<std::mutex> lock(mtx);
        push_socket.send(reply, zmq::send_flags::none);
        // std::cout << cluster_id << ". Update send" << std::endl;
      };

      // std::thread update_thread(update_lambda, std::ref(program),
      // std::cref(update));
      std::thread update_thread(update_lambda, cluster_id, update.run_id(),
                update.phase_id());//, keep_cameras_fixed);
      update_thread.detach();
      /// update_thread.join();

      break;
    }

    // one idea would be to receive, start a thread to compute result, send the
    // result. Problem: ZMQ_REP is blocking -- zmq.REQ is alos blocking in
    // python. Dealer is like an assync Req socket. Router is like an assync Rep
    // Socket. Request (REQ) / reply (REP). If we replace REP with ROUTER. This
    // gives us an asynchronous server that can talk to multiple REQ clients
    // Push/Pull Pattern.
    // client pushes to port A, server listens/pull to port A in loop
    // server does threaded work and sends/pushes result to port B, client
    // listens/pulls to port B

    // if we get program we setup new program. if we get cam & prox we update
    // cams (?) and prox term only! do one more it, etc.
    case request_proto::OptionsCase::kProgram: {
      const program_proto& pro = request_p.program();
      const int cluster_id = pro.cluster_id();
      WORKER_LOG("request_proto::OptionsCase::kProgram " << cluster_id
            << std::endl);
      // if(cluster_to_program.find(cluster_id) == cluster_to_program.end())
      CeresProgram &program = cluster_to_program[pro.cluster_id()];
      program.ResetProgram(pro);

      auto program_lambda = [&cluster_to_program, &push_socket,
                 &mtx](int cluster_id, std::uint64_t run_id,
                   std::uint64_t phase_id) {
        CeresProgram &program = cluster_to_program[cluster_id];

#ifdef __ceresVersion__
        program.UpdateStepSize();
        // program.Solve(); // only pcg!
        // std::this_thread::sleep_for(std::chrono::seconds(5));
#else
        program.UpdateStepSizeAndSolve();
#endif
        return_cluster_proto return_proto = program.FillReturnProto();
        return_proto.set_run_id(run_id);
        return_proto.set_phase_id(phase_id);
        const double cost = return_proto.cost();
        WORKER_LOG(cluster_id << ". Cost from program: " << cost << "\n");
        // SerializeToArray saves memory and time?
        const size_t bytes = return_proto.ByteSizeLong();
        zmq::message_t reply(bytes);
        return_proto.SerializeToArray(reply.data(), bytes);
        std::lock_guard<std::mutex> lock(mtx);
        push_socket.send(reply, zmq::send_flags::none);
      };
      // std::thread program_thread(program_lambda, std::ref(program),
      // std::cref(pro));
      std::thread program_thread(program_lambda, cluster_id, pro.run_id(),
                 pro.phase_id());
      program_thread.detach();
      // program_thread.join();// ok, so proto pro runs out of scope / gets
      // deleted. std::this_thread::sleep_for(std::chrono::seconds(0)); // >5 s
      // ok, so .. haeh?
      break;
    }

    case request_proto::OptionsCase::kCostUpdate: {
      const cost_proto& costUpdate = request_p.cost_update();
      const int cluster_id = costUpdate.cluster_id();
      WORKER_LOG("request_proto::OptionsCase::kCostUpdate " << cluster_id
            << std::endl);
      THROW_IF(cluster_to_program.find(cluster_id) == cluster_to_program.end());
      CeresProgram &program = cluster_to_program[cluster_id];
      program.UpdateCameras(
          costUpdate); // update is local, we need to fill data in main thread.
      bool revert_lms = costUpdate.revert_lm() == 2 ? true : false;
      // Define a Lambda Expression
      auto cost_lambda = [&push_socket, &cluster_to_program,
              &mtx](int cluster_id, bool revert_lms,
                std::uint64_t run_id,
                std::uint64_t phase_id) {
        CeresProgram &program = cluster_to_program[cluster_id];
        const double cost = 2 * program.GetCost(revert_lms);
        WORKER_LOG(cluster_id << ". Cost from cost: " << cost << "\n");
        return_cost_proto return_proto;
        return_proto.set_cost(cost);
        return_proto.set_cluster_id(cluster_id);
        return_proto.set_run_id(run_id);
        return_proto.set_phase_id(phase_id);
        // SerializeToArray saves memory and time?
        const size_t bytes = return_proto.ByteSizeLong();
        zmq::message_t reply(bytes);
        return_proto.SerializeToArray(reply.data(), bytes);
        std::lock_guard<std::mutex> lock(mtx);
        push_socket.send(reply, zmq::send_flags::none);
      };
      std::thread cost_thread(cost_lambda, cluster_id, revert_lms,
              costUpdate.run_id(), costUpdate.phase_id());
      cost_thread.detach();
      break;
    }

    case request_proto::OptionsCase::kPreconditioningUpdate: {
      WORKER_LOG("request_proto::OptionsCase::kPreconditioningUpdate"
            << std::endl);
      const preconditioning_proto& ppro = request_p.preconditioning_update();
      // Define a Lambda Expression
      CeresProgram &program = cluster_to_program[ppro.cluster_id()];
      program.UpdatePreconditioning(ppro);
      break;
    }

    case request_proto::OptionsCase::kBestCost: {
      WORKER_LOG("request_proto::OptionsCase::kBestCost" << std::endl);
      const best_cost_proto& ppro = request_p.best_cost();
      // Define a Lambda Expression
      CeresProgram &program = cluster_to_program[ppro.cluster_id()];
      program.UpdateBestCost(ppro);
    }

    default: {
      break;
    }
    }

  } // end while

  return 0;
}