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
#include <cstring>
#include <cstdlib>
#include <array>
#include <algorithm>
#include <string>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <omp.h>
#include <chrono>
#include "test.pb.h"
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

struct SingleNodeConsensusContribution {
  std::vector<std::uint32_t> global_camera_ids;
  std::vector<std::array<double, 81>> metrics;
  std::vector<std::array<double, 9>> cameras;
  std::vector<std::array<double, 9>> centers;
};

struct SingleNodeConsensusResult {
  std::vector<double> consensus;
  double fixed_point_squared = 0.;
  double proximal_displacement_squared = 0.;
  double reflection_projection_squared = 0.;
  double center_step_squared = 0.;
  double splitting_term = 0.;
};

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

int EnvironmentInteger(const char* name, int default_value,
                       int minimum, int maximum) {
  const double value = EnvironmentDouble(
      name, static_cast<double>(default_value),
      static_cast<double>(minimum), static_cast<double>(maximum));
  if (value != std::floor(value)) {
    std::ostringstream error;
    error << name << " must be an integer";
    throw std::runtime_error(error.str());
  }
  return static_cast<int>(value);
}

double CameraDiagonalRelativeFloor() { // 1e-48 worked well, lower maybe more stable for different cluster sizes, example problem 1723. 40 already delivered inferior costs for 30 clusters 
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

struct DabaRayErrorWeighted {
  DabaRayErrorWeighted(
      double observed_x, double observed_y, double initial_focal,
      const double* camera_weight, const double* point_weight,
      const double* camera_transform)
      : observed_x(observed_x), observed_y(observed_y),
        initial_focal(initial_focal), camera_weight(camera_weight),
        point_weight(point_weight), camera_transform(camera_transform) {}

  template <typename T>
  bool operator()(const T* const camera, const T* const point,
                  T* residuals) const {
    T physical_camera[9];
    for (int row = 0; row < 9; ++row) {
      physical_camera[row] = T(0);
      for (int col = 0; col < 9; ++col) {
        physical_camera[row] += T(camera_transform[9 * row + col])
                                * camera[col] * T(camera_weight[col]);
      }
    }
    T physical_point[3] = {
        point[0] * T(point_weight[0]),
        point[1] * T(point_weight[1]),
        point[2] * T(point_weight[2])};
    T camera_point[3];
    ceres::AngleAxisRotatePoint(physical_camera, physical_point, camera_point);
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      camera_point[coordinate] += physical_camera[3 + coordinate];
    }
    const T normalized_x = T(observed_x / -initial_focal);
    const T normalized_y = T(observed_y / -initial_focal);
    const T radius_squared =
        normalized_x * normalized_x + normalized_y * normalized_y;
    T ray[3] = {
        normalized_x,
        normalized_y,
        physical_camera[6] / T(initial_focal)
            + physical_camera[7] * T(initial_focal * initial_focal)
                * radius_squared
            + physical_camera[8]
                * T(initial_focal * initial_focal * initial_focal
                    * initial_focal)
                * radius_squared * radius_squared};
    T distance[3] = {
        -camera_point[0], -camera_point[1], -camera_point[2]};
    const T distance_squared = distance[0] * distance[0]
        + distance[1] * distance[1] + distance[2] * distance[2] + T(1e-12);
    const T denominator =
        distance_squared + T(1e-6) * sqrt(distance_squared);
    const T projection = (distance[0] * ray[0] + distance[1] * ray[1]
                          + distance[2] * ray[2]) / denominator;
    const T sqrt_weight = T(initial_focal) * sqrt(radius_squared + T(1));
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      residuals[coordinate] =
          sqrt_weight * (ray[coordinate] - projection * distance[coordinate]);
    }
    return true;
  }

  static ceres::CostFunction* Create(
      double observed_x, double observed_y, double initial_focal,
      const double* camera_weight, const double* point_weight,
      const double* camera_transform) {
    return new ceres::AutoDiffCostFunction<DabaRayErrorWeighted, 3, 9, 3>(
        new DabaRayErrorWeighted(
            observed_x, observed_y, initial_focal, camera_weight,
            point_weight, camera_transform));
  }

  double observed_x;
  double observed_y;
  double initial_focal;
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

struct MetricDiagnostic {
  double transformed_lipschitz = std::numeric_limits<double>::quiet_NaN();
  double relative_residual = std::numeric_limits<double>::quiet_NaN();
  double camera_proximal_defect_squared =
      std::numeric_limits<double>::quiet_NaN();
  double landmark_proximal_defect_squared =
      std::numeric_limits<double>::quiet_NaN();
  double proximal_defect_squared = std::numeric_limits<double>::quiet_NaN();
  int iterations = 0;
};

void EstimateProximalDefect(
    const NormalEquations& normal_equations,
    const std::vector<double>& metric_blocks,
    const std::vector<double>& cameras,
    const std::vector<double>& centers,
    MetricDiagnostic& diagnostic) {
  THROW_IF(cameras.size() != centers.size());
  THROW_IF(metric_blocks.size() != 9 * cameras.size());
  Eigen::VectorXd offset(cameras.size());
  for (int index = 0; index < offset.size(); ++index) {
    offset[index] = cameras[index] - centers[index];
  }
  Eigen::VectorXd camera_defect = normal_equations.camera_gradient;
  Eigen::VectorXd proximal_gradient(camera_defect.size());
  blockMult<9>(metric_blocks, offset, proximal_gradient);
  camera_defect += proximal_gradient;

  std::vector<double> metric_inverse = metric_blocks;
#pragma omp parallel for num_threads(options.num_threads)
  for (int block = 0; block < camera_defect.size() / 9; ++block) {
    auto inverse_block = Map<Matrix<double, 9, 9>>(
        &(metric_inverse[81 * block]));
    inverse_block = inverse_block.inverse().eval();
  }
  Eigen::VectorXd preconditioned_camera(camera_defect.size());
  blockMult<9>(metric_inverse, camera_defect, preconditioned_camera);
  diagnostic.camera_proximal_defect_squared = std::max(
      0., camera_defect.dot(preconditioned_camera));

  SparseMatrix<double, RowMajor> landmark_inverse =
      normal_equations.landmark_hessian;
  BlockInverse<3>(landmark_inverse);
  diagnostic.landmark_proximal_defect_squared = std::max(
      0., normal_equations.landmark_gradient.dot(
          landmark_inverse * normal_equations.landmark_gradient));
  diagnostic.proximal_defect_squared =
      diagnostic.camera_proximal_defect_squared
      + diagnostic.landmark_proximal_defect_squared;
}

MetricDiagnostic EstimateTransformedLipschitz(
    const SparseMatrix<double, RowMajor>& camera_hessian,
    const SparseMatrix<double, RowMajor>& landmark_hessian,
    const BlockEdgeMatrix& cross_hessian,
    const std::vector<double>& metric_blocks,
    int iterations) {
  MetricDiagnostic diagnostic;
  if (iterations <= 0) {
    return diagnostic;
  }
  THROW_IF(camera_hessian.rows() != cross_hessian.rows());
  THROW_IF(landmark_hessian.rows() != cross_hessian.cols());
  THROW_IF(metric_blocks.size() !=
           static_cast<std::size_t>(9 * camera_hessian.rows()));

  SparseMatrix<double, RowMajor> landmark_inverse = landmark_hessian;
  BlockInverse<3>(landmark_inverse);
  std::vector<double> metric_inverse = metric_blocks;
#pragma omp parallel for num_threads(options.num_threads)
  for (int block = 0; block < camera_hessian.rows() / 9; ++block) {
    auto inverse_block = Map<Matrix<double, 9, 9>>(
        &(metric_inverse[81 * block]));
    inverse_block = inverse_block.inverse().eval();
  }

  Eigen::VectorXd vector(camera_hessian.rows());
  for (int index = 0; index < vector.size(); ++index) {
    vector[index] = std::sin(0.5 + static_cast<double>(index + 1));
  }
  Eigen::VectorXd metric_vector(vector.size());
  blockMult<9>(metric_blocks, vector, metric_vector);
  const double initial_norm_squared = vector.dot(metric_vector);
  if (!(initial_norm_squared > 0.) || !std::isfinite(initial_norm_squared)) {
    return diagnostic;
  }
  vector /= std::sqrt(initial_norm_squared);

  Eigen::VectorXd landmark_workspace(cross_hessian.cols());
  Eigen::VectorXd camera_workspace(cross_hessian.rows());
  Eigen::VectorXd schur_vector(camera_hessian.rows());
  Eigen::VectorXd next_vector(camera_hessian.rows());
  for (int iteration = 0; iteration < iterations; ++iteration) {
    cross_hessian.TransposeMultiply(vector, landmark_workspace);
    landmark_workspace = landmark_inverse * landmark_workspace;
    cross_hessian.Multiply(landmark_workspace, camera_workspace);
    schur_vector.noalias() = camera_hessian * vector;
    schur_vector -= camera_workspace;
    blockMult<9>(metric_inverse, schur_vector, next_vector);
    blockMult<9>(metric_blocks, next_vector, metric_vector);
    const double norm_squared = next_vector.dot(metric_vector);
    if (!(norm_squared > 0.) || !std::isfinite(norm_squared)) {
      return diagnostic;
    }
    vector = next_vector / std::sqrt(norm_squared);
    diagnostic.iterations = iteration + 1;
  }

  cross_hessian.TransposeMultiply(vector, landmark_workspace);
  landmark_workspace = landmark_inverse * landmark_workspace;
  cross_hessian.Multiply(landmark_workspace, camera_workspace);
  schur_vector.noalias() = camera_hessian * vector;
  schur_vector -= camera_workspace;
  blockMult<9>(metric_blocks, vector, metric_vector);
  const double denominator = vector.dot(metric_vector);
  if (!(denominator > 0.) || !std::isfinite(denominator)) {
    return diagnostic;
  }
  diagnostic.transformed_lipschitz = vector.dot(schur_vector) / denominator;
  const Eigen::VectorXd residual =
      schur_vector - diagnostic.transformed_lipschitz * metric_vector;
  diagnostic.relative_residual = residual.norm() /
      std::max(schur_vector.norm(), std::numeric_limits<double>::min());
  return diagnostic;
}

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

    int ClusterCount() const { return num_clusters; }

    SingleNodeConsensusContribution BuildSingleNodeConsensusContribution()
        const {
      THROW_IF(global_camera_ids.size() != static_cast<size_t>(numCameras));
      THROW_IF(cameras.size() != static_cast<size_t>(9 * numCameras));
      THROW_IF(cameras_s.size() != cameras.size());
      THROW_IF(full_stepSize.size() != static_cast<size_t>(81 * numCameras));
      SingleNodeConsensusContribution contribution;
      contribution.global_camera_ids = global_camera_ids;
      contribution.metrics.resize(numCameras);
      contribution.cameras.resize(numCameras);
      contribution.centers.resize(numCameras);
      for (int camera = 0; camera < numCameras; ++camera) {
        for (int row = 0; row < 9; ++row) {
          contribution.cameras[camera][row] = cameras[9 * camera + row];
          contribution.centers[camera][row] = cameras_s[9 * camera + row];
          for (int column = 0; column < 9; ++column) {
            contribution.metrics[camera][9 * row + column] =
              static_cast<float>(
                full_stepSize[81 * camera + 9 * row + column]);
          }
        }
      }
      return contribution;
    }

    void ResetProgram(const program_proto &pro) {
      Init(pro.num_clusters());
      numCameras = pro.cameras_size() / 9;
      numLandmarks = pro.landmarks_size() / 3;
      numResiduals =  pro.observations_size() / 2;
      global_camera_ids.assign(
        pro.global_camera_id().begin(), pro.global_camera_id().end());
      THROW_IF(!global_camera_ids.empty()
          && global_camera_ids.size() != static_cast<size_t>(numCameras));
      local_iterations = std::max(1, std::min(20, pro.iterations()));
      scalar_proximal_prior = pro.scalar_proximal_prior();
      block_curvature_multiplier = pro.block_curvature_multiplier();
      metric_diagnostic_iterations = pro.metric_diagnostic_iterations();
      landmark_refinement_steps = pro.landmark_refinement_steps();
      THROW_IF(metric_diagnostic_iterations < 0 ||
           metric_diagnostic_iterations > 100);
      THROW_IF(landmark_refinement_steps < 0 ||
           landmark_refinement_steps > 20);
      THROW_IF(block_curvature_multiplier < 0. ||
           !std::isfinite(block_curvature_multiplier));
      proximal_rho = pro.proximal_rho();
      split_camera_penalty = pro.split_camera_penalty();
      proximal_rho_intrinsics = pro.proximal_rho_intrinsics();
      local_linear_solver = pro.local_linear_solver();
      trust_region_policy = pro.trust_region_policy();
      persistent_trust_region = pro.persistent_trust_region();
      persistent_trust_region_active = persistent_trust_region;
      if (trust_region_policy == 1 && persistent_trust_region) {
        tr_radius = std::min(100., max_trust_region_radius);
      }
      ceres_local_solver = pro.ceres_local_solver();
      objective_model = pro.objective_model();
      residual_dimension = objective_model == 1 ? 3 : 2;
      THROW_IF(objective_model != 0 && objective_model != 1);
      if (LocalSolveMetricsEnabled()) {
        std::ostringstream metric;
        metric << "WORKER_OBJECTIVE cluster=" << pro.cluster_id()
               << " model=" << objective_model
               << " focals=" << pro.initial_focal_size() << "\n";
        EmitLocalSolveMetric(metric.str());
      }
      if (scalar_proximal_prior) {
        THROW_IF(!(proximal_rho > 0.) || !std::isfinite(proximal_rho));
        if (split_camera_penalty) {
          THROW_IF(!(proximal_rho_intrinsics > 0.) ||
                   !std::isfinite(proximal_rho_intrinsics));
        }
      }
      //std::cout << numCameras << " " << numLandmarks << "\n";
      cameras.clear();
      cameras.reserve(9 * numCameras);
      for (const auto &v : pro.cameras()) {
        cameras.push_back(v);
      }
      last_cameras = cameras;
      //std::cout << "cameras.push_back\n";
      landmarks.clear();
      landmarks.reserve(3 * numLandmarks);
      for (const auto &v : pro.landmarks()) {
        landmarks.push_back(v);
      }
      last_landmarks = landmarks;
      last_tr_radius = tr_radius;
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
      initial_focal.assign(pro.initial_focal().begin(), pro.initial_focal().end());
      if (initial_focal.empty()) {
        initial_focal.resize(numCameras);
        for (int camera = 0; camera < numCameras; ++camera) {
          double focal = 0.;
          for (int col = 0; col < 9; ++col) {
            focal += cameraTransform[81 * camera + 9 * 6 + col]
                     * cameras[9 * camera + col] * unorm[9 * camera + col];
          }
          initial_focal[camera] = focal;
        }
      }
      THROW_IF(initial_focal.size() != numCameras);
      //std::cout << "cameras.push_back\n";
      vnorm.clear();
      vnorm.reserve(3 * numLandmarks);
      for (const auto &v : pro.vnorm()) {
        vnorm.push_back(v);
      }
      //std::cout << "data updated\n";

      options.max_num_iterations = std::max(0, std::min(20, pro.iterations()));
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
        ceres::CostFunction *cost_function = objective_model == 1
          ? DabaRayErrorWeighted::Create(
              pro.observations(2 * i + 0), pro.observations(2 * i + 1),
              initial_focal[camera_id], &(unorm[9 * camera_id]),
              &(vnorm[3 * landmark_id]), &(cameraTransform[81 * camera_id]))
          : SnavelyReprojectionErrorWeighted::Create(
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
      normal_equation_residuals.reserve(residual_dimension * numResiduals);
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
      if (objective_model == 0 && BatchedEvaluationEnabled()) {
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

    void AddPhysicalLandmarks(return_cost_proto &return_proto) const {
      THROW_IF(landmarks.size() != vnorm.size());
      for (int landmark_value = 0; landmark_value < landmarks.size();
           ++landmark_value) {
        return_proto.add_landmarks(
            landmarks[landmark_value] * vnorm[landmark_value]);
      }
    }

    void AddPhysicalLandmarks(
        const std::vector<double>& source_landmarks,
        landmark_state_reply_proto& return_proto) const {
      THROW_IF(source_landmarks.size() != vnorm.size());
      std::vector<double> physical_landmarks(source_landmarks.size());
      for (int landmark_value = 0; landmark_value < source_landmarks.size();
           ++landmark_value) {
        physical_landmarks[landmark_value] =
            source_landmarks[landmark_value] * vnorm[landmark_value];
      }
      return_proto.set_landmarks_f64(
          reinterpret_cast<const char*>(physical_landmarks.data()),
          physical_landmarks.size() * sizeof(double));
    }

    void MaterializeCurrentLandmarkState(
        landmark_state_reply_proto& return_proto) const {
      AddPhysicalLandmarks(landmarks, return_proto);
    }

    double EvaluateRefinedLandmarkCost(
        int refinement_steps, return_cost_proto& return_proto) {
      const std::vector<double> local_landmarks = landmarks;
      const double local_cost = cost;
      RefineLandmarksWithFixedCameras(refinement_steps);
      const double refined_cost = 2. * GetCost();
      AddPhysicalLandmarks(return_proto);
      landmarks = local_landmarks;
      cost = local_cost;
      return refined_cost;
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
    
    return_cluster_proto FillReturnProto(
      bool include_landmarks = true,
      bool include_consensus_rhs = false,
      bool include_metric_blocks = true) {
      return_cluster_proto return_proto = return_cluster_proto();
      return_proto.set_cameras_f64(
          reinterpret_cast<const char*>(cameras.data()),
          cameras.size() * sizeof(double));
      if (include_landmarks) {
      THROW_IF(landmarks.size() != vnorm.size());
      std::vector<double> physical_landmarks(landmarks.size());
      for (int landmark_value = 0; landmark_value < landmarks.size();
         ++landmark_value) {
        physical_landmarks[landmark_value] =
          landmarks[landmark_value] * vnorm[landmark_value];
      }
      return_proto.set_landmarks_f64(
        reinterpret_cast<const char*>(physical_landmarks.data()),
        physical_landmarks.size() * sizeof(double));
      }
      if (include_metric_blocks) {
        std::vector<float> metric_upper_blocks;
        metric_upper_blocks.reserve(45 * numCameras);
        for (int camera = 0; camera < numCameras; ++camera) {
          const int offset = 81 * camera;
          for (int row = 0; row < 9; ++row) {
            for (int column = row; column < 9; ++column) {
              const float upper = static_cast<float>(
                  full_stepSize[offset + 9 * row + column]);
              const float lower = static_cast<float>(
                  full_stepSize[offset + 9 * column + row]);
              THROW_IF(upper != lower);
              metric_upper_blocks.push_back(upper);
            }
          }
        }
        return_proto.set_step_size_upper_f32(
            reinterpret_cast<const char*>(metric_upper_blocks.data()),
            metric_upper_blocks.size() * sizeof(float));
      }
      if (include_consensus_rhs) {
        std::vector<double> consensus_rhs(9 * numCameras, 0.);
        for (int camera = 0; camera < numCameras; ++camera) {
          const int camera_offset = 9 * camera;
          const int metric_offset = 81 * camera;
          for (int row = 0; row < 9; ++row) {
            double value = 0.;
            for (int column = 0; column < 9; ++column) {
              const double metric_value = static_cast<float>(
                  full_stepSize[metric_offset + 9 * row + column]);
              value += metric_value * (
                  2. * cameras[camera_offset + column]
                  - cameras_s[camera_offset + column]);
            }
            consensus_rhs[camera_offset + row] = value;
          }
        }
        return_proto.set_consensus_rhs_f64(
            reinterpret_cast<const char*>(consensus_rhs.data()),
            consensus_rhs.size() * sizeof(double));
      }
      return_proto.set_cluster_id(cluster_id);
      return_proto.set_cost(cost);
      return_proto.set_objective_model(objective_model);
        return_proto.set_transformed_lipschitz_estimate(
          metric_diagnostic.transformed_lipschitz);
        return_proto.set_transformed_lipschitz_residual(
          metric_diagnostic.relative_residual);
        return_proto.set_metric_diagnostic_iterations(
          metric_diagnostic.iterations);
          return_proto.set_camera_proximal_defect_squared(
            metric_diagnostic.camera_proximal_defect_squared);
          return_proto.set_landmark_proximal_defect_squared(
            metric_diagnostic.landmark_proximal_defect_squared);
          return_proto.set_proximal_defect_squared(
            metric_diagnostic.proximal_defect_squared);
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
      if (LocalSolveMetricsEnabled()) {
        std::ostringstream metric;
        metric << "CERES_LOCAL_SOLVE cluster=" << cluster_id
               << " initial=" << 2 * summary.initial_cost
               << " final=" << 2 * summary.final_cost
               << " successful=" << summary.num_successful_steps
               << " unsuccessful=" << summary.num_unsuccessful_steps
               << " iterations=" << summary.iterations.size()
               << " termination=" << summary.termination_type
               << " radius=" << tr_radius << "\n";
        EmitLocalSolveMetric(metric.str());
      }
      return cost;
    }

    void UpdateCostState(const cost_proto& costProto) {
      // std::cout << "Update cluster " << cluster_id << " update proto id:" << update.cluster_id() << "\n";
      const bool packed_cameras = !costProto.cameras_f64().empty();
      THROW_IF(packed_cameras && costProto.cameras_size() != 0);
      THROW_IF(packed_cameras
          ? costProto.cameras_f64().size()
              != cameras.size() * sizeof(double)
          : costProto.cameras_size() != cameras.size());
      THROW_IF(costProto.cluster_id() != cluster_id);

      int id = 0; // fill existing buffer
      if (packed_cameras) {
        std::memcpy(cameras.data(), costProto.cameras_f64().data(),
            costProto.cameras_f64().size());
      } else {
        for (const auto &v : costProto.cameras()) {
          cameras[id++] = v;
        }
      }
      if (costProto.landmarks_size() > 0) {
        THROW_IF(costProto.landmarks_size() != landmarks.size());
        id = 0;
        for (const auto &v : costProto.landmarks()) {
          landmarks[id] = v / vnorm[id];
          ++id;
        }
      }
    }

    const std::vector<double>& CurrentCameras() const {
      return cameras;
    }

    void RestoreCameras(const std::vector<double>& saved_cameras) {
      THROW_IF(saved_cameras.size() != cameras.size());
      cameras = saved_cameras;
    }

    void SaveNominalLandmarkState(std::uint64_t state_id) {
      THROW_IF(state_id == 0 || nominal_landmark_state_id != 0);
      nominal_cameras = cameras;
      nominal_landmarks = landmarks;
      nominal_landmark_state_id = state_id;
    }

    void ValidateNominalLandmarkState(std::uint64_t state_id) const {
      THROW_IF(state_id == 0 || state_id != nominal_landmark_state_id);
      THROW_IF(nominal_cameras.size() != cameras.size());
      THROW_IF(nominal_landmarks.size() != landmarks.size());
    }

    void RestoreNominalLandmarkState(std::uint64_t state_id) {
      ValidateNominalLandmarkState(state_id);
      cameras = nominal_cameras;
      landmarks = nominal_landmarks;
    }

    void RoundTripPhysicalLandmarks() {
      for (int id = 0; id < landmarks.size(); ++id) {
        landmarks[id] = (landmarks[id] * vnorm[id]) / vnorm[id];
      }
    }

    void RestoreNominalLandmarkStateWithRoundTrip(std::uint64_t state_id) {
      RestoreNominalLandmarkState(state_id);
      RoundTripPhysicalLandmarks();
    }

    void DiscardNominalLandmarkState(std::uint64_t state_id) {
      THROW_IF(state_id == 0 || state_id != nominal_landmark_state_id);
      nominal_cameras.clear();
      nominal_landmarks.clear();
      nominal_landmark_state_id = 0;
    }

    void SaveAcceptedLandmarkState() {
      accepted_landmarks = landmarks;
    }

    void RestoreAcceptedLandmarkState() {
      THROW_IF(accepted_landmarks.size() != landmarks.size());
      landmarks = accepted_landmarks;
    }

    void SaveBestOutputLandmarkState() {
      best_output_landmarks = landmarks;
    }

    void MaterializeBestOutputLandmarkState(
        landmark_state_reply_proto& return_proto) const {
      AddPhysicalLandmarks(best_output_landmarks, return_proto);
    }

    void UpdateData(const prox_cluster_proto &update) {
      // std::cout << "Update cluster " << cluster_id << " update proto id:" << update.cluster_id() << "\n";
      const bool packed_cameras = !update.cameras_f64().empty();
      const bool packed_centers = !update.cameras_s_f64().empty();
      THROW_IF(packed_cameras && update.cameras_size() != 0);
      THROW_IF(packed_centers && update.cameras_s_size() != 0);
      THROW_IF(update.retain_cameras()
          ? packed_cameras || update.cameras_size() != 0
          : (packed_cameras
              ? update.cameras_f64().size()
                  != cameras.size() * sizeof(double)
              : update.cameras_size() != cameras.size()));
      THROW_IF(packed_centers
          ? update.cameras_s_f64().size()
              != cameras_s.size() * sizeof(double)
          : update.cameras_s_size() != cameras_s.size());
      THROW_IF(update.cluster_id() != cluster_id);

      int id = 0; // fill existing buffer
      if (!update.retain_cameras()) {
        if (packed_cameras) {
          std::memcpy(cameras.data(), update.cameras_f64().data(),
              update.cameras_f64().size());
        } else {
          for (const auto &v : update.cameras()) {
            cameras[id++] = v;
          }
        }
      }
      if (packed_centers) {
        std::memcpy(cameras_s.data(), update.cameras_s_f64().data(),
            update.cameras_s_f64().size());
      } else {
        id = 0;
        for (const auto &v : update.cameras_s()) {
          cameras_s[id++] = v;
        }
      }
      current_be = update.be();
      scalar_proximal_prior = update.scalar_proximal_prior();
      block_curvature_multiplier = update.block_curvature_multiplier();
      metric_diagnostic_iterations = update.metric_diagnostic_iterations();
      landmark_refinement_steps = update.landmark_refinement_steps();
      THROW_IF(metric_diagnostic_iterations < 0 ||
           metric_diagnostic_iterations > 100);
      THROW_IF(landmark_refinement_steps < 0 ||
           landmark_refinement_steps > 20);
      THROW_IF(block_curvature_multiplier < 0. ||
           !std::isfinite(block_curvature_multiplier));
      proximal_rho = update.proximal_rho();
      split_camera_penalty = update.split_camera_penalty();
      proximal_rho_intrinsics = update.proximal_rho_intrinsics();
      local_linear_solver = update.local_linear_solver();
      trust_region_policy = update.trust_region_policy();
      persistent_trust_region = update.persistent_trust_region();
      ceres_local_solver = update.ceres_local_solver();
      if (update.landmarks_size() > 0) {
        THROW_IF(update.landmarks_size() != landmarks.size());
        for (int id = 0; id < update.landmarks_size(); ++id) {
          landmarks[id] = update.landmarks(id) / vnorm[id];
        }
      }
      if (scalar_proximal_prior) {
        THROW_IF(!(proximal_rho > 0.) || !std::isfinite(proximal_rho));
        if (split_camera_penalty) {
          THROW_IF(!(proximal_rho_intrinsics > 0.) ||
                   !std::isfinite(proximal_rho_intrinsics));
        }
      }
      firstIteration = false;

      if (update.revert_lm() == 1) {
        WORKER_LOG(cluster_id << ". Revert landmarks\n");
        if (update.revert_cameras()) {
          THROW_IF(last_cameras.size() != cameras.size());
          cameras = last_cameras;
        }
        landmarks = last_landmarks;
        tr_radius = last_tr_radius;
        if (persistent_trust_region) {
          persistent_trust_region_active = true;
          tr_radius *= update.trust_region_recovery_ratio();
          last_tr_radius = tr_radius;
        }
      } else if (update.revert_lm() == 3) {
        WORKER_LOG(cluster_id << ". Keep restored accepted landmarks\n");
        RoundTripPhysicalLandmarks();
        if (persistent_trust_region) {
          persistent_trust_region_active = true;
          tr_radius = last_tr_radius * update.trust_region_recovery_ratio();
          tr_radius = std::max(1e-4, std::min(
              max_trust_region_radius, tr_radius));
          last_tr_radius = tr_radius;
        }
        last_cameras = cameras;
        last_landmarks = landmarks;
        last_tr_radius = tr_radius;
      } else if (update.revert_lm() == 2) {
        WORKER_LOG(cluster_id << ". Revert landmarks to best cost lms\n");
        if (persistent_trust_region) {
          persistent_trust_region_active = true;
          tr_radius = last_tr_radius * update.trust_region_recovery_ratio();
          tr_radius = std::max(1e-4, std::min(
              max_trust_region_radius, tr_radius));
          last_tr_radius = tr_radius;
        }
        if (update.landmarks_size() == 0) {
          landmarks = best_landmarks;
        }
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
        last_cameras = cameras;
        last_landmarks = landmarks;
        last_tr_radius = tr_radius;
      }
      options.initial_trust_region_radius = tr_radius;
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

    double CameraPenalty(int parameter) const {
      return split_camera_penalty && parameter >= 6
          ? proximal_rho_intrinsics : proximal_rho;
    }

    void PrepareScalarCeresPrior() {
      THROW_IF(!scalar_proximal_prior || !(proximal_rho > 0.));
      stepSize.assign(81 * numCameras, 0.);
      for (int camera = 0; camera < numCameras; ++camera) {
        for (int parameter = 0; parameter < 9; ++parameter) {
          stepSize[81 * camera + 10 * parameter] =
              std::sqrt(CameraPenalty(parameter));
        }
      }
    }

    bool UsesCeresLocalSolver() const { return ceres_local_solver; }
    int LocalIterations() const { return local_iterations; }

    void UpdatePreconditioning(const preconditioning_proto& preconditioningProto) {
        // std::cout << "Update cluster " << cluster_id << " update proto id:" << update.cluster_id() << "\n";
        THROW_IF(preconditioningProto.cluster_id() != cluster_id);
        THROW_IF(preconditioningProto.unorm_size() != unorm.size());
        THROW_IF(preconditioningProto.vnorm_size() != vnorm.size());
        if (preconditioningProto.cameras_size() > 0) {
          THROW_IF(preconditioningProto.cameras_size() != cameras.size());
          cameras.assign(
              preconditioningProto.cameras().begin(),
              preconditioningProto.cameras().end());
          last_cameras = cameras;
        }

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
      THROW_IF(ppro.landmarks_size() != landmarks.size());
      WORKER_LOG("Best cost update " << cluster_id << " " << ppro.cost() << " < " << best_cost << "\n");
//      if (ppro.cost() < best_cost) {
        best_cost = ppro.cost();
        for (int id = 0; id < ppro.landmarks_size(); ++id) {
          best_landmarks[id] = ppro.landmarks(id) / vnorm[id];
        }
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

std::pair<Matrix<double, Eigen::Dynamic, 1>, Matrix<double, Eigen::Dynamic, 1>>
SolveBySchurPCG(
    SparseMatrix<double, RowMajor> Uli,
    SparseMatrix<double, RowMajor> Vli,
    const BlockEdgeMatrix& W,
    const Matrix<double, Eigen::Dynamic, 1>& bp,
    const Matrix<double, Eigen::Dynamic, 1>& bl,
    const Matrix<double, Eigen::Dynamic, 1>& proximalGradient,
    int* iterations_out) {
  SparseMatrix<double, RowMajor> Vinv = Vli;
  SparseMatrix<double, RowMajor> Uinv = Uli;
  BlockInverse<3>(Vinv);
  BlockInverse<9>(Uinv);

  Eigen::VectorXd vinvBl = Vinv * bl;
  Eigen::VectorXd wVinvBl(W.rows());
  W.Multiply(vinvBl, wVinvBl);
  Eigen::VectorXd right_hand_side = -bp - proximalGradient + wVinvBl;

  Eigen::VectorXd solution = Eigen::VectorXd::Zero(W.rows());
  Eigen::VectorXd residual = right_hand_side;
  Eigen::VectorXd preconditioned = Uinv * residual;
  Eigen::VectorXd direction = preconditioned;
  Eigen::VectorXd wtDirection(W.cols());
  Eigen::VectorXd vinvWtDirection(W.cols());
  Eigen::VectorXd wVinvWtDirection(W.rows());
  Eigen::VectorXd schurDirection(W.rows());
  double residual_preconditioned = residual.dot(preconditioned);
  const double initial_residual_norm = std::max(
      residual.norm(), std::numeric_limits<double>::min());
  int iterations = 0;

  for (; iterations < 400; ++iterations) {
    W.TransposeMultiply(direction, wtDirection);
    vinvWtDirection.noalias() = Vinv * wtDirection;
    W.Multiply(vinvWtDirection, wVinvWtDirection);
    schurDirection.noalias() = Uli * direction;
    schurDirection -= wVinvWtDirection;
    const double denominator = direction.dot(schurDirection);
    if (!(denominator > 0.) || !std::isfinite(denominator)) {
      break;
    }
    const double alpha = residual_preconditioned / denominator;
    solution += alpha * direction;
    residual -= alpha * schurDirection;
    if (residual.norm() <= 1e-2 * initial_residual_norm) {
      ++iterations;
      break;
    }
    preconditioned.noalias() = Uinv * residual;
    const double next_residual_preconditioned = residual.dot(preconditioned);
    if (!(next_residual_preconditioned >= 0.) ||
        !std::isfinite(next_residual_preconditioned)) {
      break;
    }
    const double beta = next_residual_preconditioned /
        std::max(residual_preconditioned, std::numeric_limits<double>::min());
    direction = preconditioned + beta * direction;
    residual_preconditioned = next_residual_preconditioned;
  }
  *iterations_out = iterations;

  Eigen::VectorXd wtSolution(W.cols());
  W.TransposeMultiply(solution, wtSolution);
  Eigen::VectorXd delta_l = Vinv * (wtSolution - bl);
  return {solution, delta_l};
}

void UpdateStepSizeAndSolve() {//bool keep_cameras_fixed = false) { // Recompute.
  // if (keep_cameras_fixed) {
  //   keep_cameras_fixed = new_best_cost;
  // }

  const bool collectTiming = LocalSolveMetricsEnabled();
  if (collectTiming) {
    const auto summarize = [](const std::vector<double>& values) {
      std::pair<double, double> summary{0., 0.};
      for (const double value : values) {
        summary.first += value;
        summary.second += value * value;
      }
      return summary;
    };
    const auto camera_summary = summarize(cameras);
    const auto center_summary = summarize(cameras_s);
    const auto landmark_summary = summarize(landmarks);
    const auto camera_scale_summary = summarize(unorm);
    const auto landmark_scale_summary = summarize(vnorm);
    std::ostringstream metric;
    metric << "PROX_INPUT cluster=" << cluster_id
           << " sequence=" << local_solve_sequence++
           << " first=" << firstIteration
           << " cameras_sum=" << camera_summary.first
           << " cameras_squared=" << camera_summary.second
           << " centers_sum=" << center_summary.first
           << " centers_squared=" << center_summary.second
           << " landmarks_sum=" << landmark_summary.first
           << " landmarks_squared=" << landmark_summary.second
           << " unorm_sum=" << camera_scale_summary.first
           << " unorm_squared=" << camera_scale_summary.second
           << " vnorm_sum=" << landmark_scale_summary.first
           << " vnorm_squared=" << landmark_scale_summary.second
           << " be=" << current_be
           << " curvature=" << block_curvature_multiplier
           << " trust=" << trust_region_policy << "\n";
    EmitLocalSolveMetric(metric.str());
  }
  metric_diagnostic = MetricDiagnostic();
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
  if (firstIteration && !scalar_proximal_prior) { // also handled setting be = 0 in 1st step.
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
    const double legacy_scale =
      std::min(1.005, 1e-1 * std::sqrt(current_be / start_be));
    const double scale = block_curvature_multiplier > 0.
      ? block_curvature_multiplier
      : legacy_scale;

  if (!firstIteration) { // also handled setting be = 0 in 1st step.
    SparseMatrix<double, RowMajor> stepSize;
    if (scalar_proximal_prior) {
      stepSize = SparseMatrix<double, RowMajor>(9 * numCameras, 9 * numCameras);
      stepSize.reserve(Eigen::VectorXi::Constant(9 * numCameras, 1));
      for (int parameter = 0; parameter < 9 * numCameras; ++parameter) {
        stepSize.insert(parameter, parameter) = CameraPenalty(parameter % 9);
      }
      stepSize.makeCompressed();
    } else {
      stepSize = scale * Ul;
      stepSize += diagUP * current_be;
    }
    const double* values = stepSize.valuePtr();
    if (scalar_proximal_prior) {
      std::fill(full_stepSize.begin(), full_stepSize.end(), 0.);
      for (int camera = 0; camera < numCameras; ++camera) {
        for (int parameter = 0; parameter < 9; ++parameter) {
            full_stepSize[81 * camera + 10 * parameter] =
              CameraPenalty(parameter);
        }
      }
    } else {
      std::copy(values, values + full_stepSize.size(), full_stepSize.data());
    }
    Ul += stepSize;
  } else {
    if (scalar_proximal_prior) {
      full_stepSize.assign(81 * numCameras, 0.);
      for (int camera = 0; camera < numCameras; ++camera) {
        for (int parameter = 0; parameter < 9; ++parameter) {
            full_stepSize[81 * camera + 10 * parameter] =
              CameraPenalty(parameter);
          Ul.coeffRef(9 * camera + parameter, 9 * camera + parameter) +=
              CameraPenalty(parameter);
        }
      }
    } else {
      Ul += scale * Ul;
      Ul += diagUP * current_be;
    }
    // let full_Stepsize define setpsize always. else confusing to debug: cost optimized differs from cost evaluated.
    // const SparseMatrix<double, RowMajor> stepSize = Ul;
    // Ul += stepSize;
  }
  if (!scalar_proximal_prior && metric_diagnostic_iterations > 0) {
    metric_diagnostic = EstimateTransformedLipschitz(
        normalEquations.camera_hessian,
        normalEquations.landmark_hessian,
        camera_landmark_hessian,
        full_stepSize,
        metric_diagnostic_iterations);
    if (collectTiming) {
      std::ostringstream metric;
      metric << "METRIC_DIAGNOSTIC cluster=" << cluster_id
             << " transformed_lipschitz="
             << metric_diagnostic.transformed_lipschitz
             << " relative_residual=" << metric_diagnostic.relative_residual
             << " iterations=" << metric_diagnostic.iterations << "\n";
      EmitLocalSolveMetric(metric.str());
    }
  }
  // Loop until ok or adjust tr_region
  if (trust_region_policy == 1 && !persistent_trust_region_active) {
    tr_radius = std::min(100., max_trust_region_radius);
  } else {
    tr_radius = std::min(max_trust_region_radius, tr_radius);
  }
  double trust_region_decreasing_ratio = 0.5;
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
  int linear_iterations = 0;
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
    std::pair<Eigen::VectorXd, Eigen::VectorXd> step;
    if (local_linear_solver == 1) {
      step = SolveBySchurPCG(
          Ul, Vl, W, bp, bl, proximalGradient, &linear_iterations);
    } else {
      step = SolveByGDNesterov(
          Ul, Vl, W, bp, bl, proximalGradient, power_iterations);
      linear_iterations = power_iterations;
    }
    const Eigen::VectorXd& delta_p = step.first;
    const Eigen::VectorXd& delta_l = step.second;
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

    const double actual_decrease =
        costStart - costEnd + penaltyStart - penaltyEnd;
    const double predicted_decrease =
        costStart - costQuad + penaltyStart - penaltyEnd;
    const double tr_check = actual_decrease /
        std::max(0.1, predicted_decrease);
    bool accept_step = false;
    if (trust_region_policy == 1) {
      accept_step = actual_decrease > 0.;
      if (accept_step) {
        const double radius_divisor = std::max(
            1. / 3., 1. - std::pow(2. * tr_check - 1., 3));
        tr_radius = std::min(
            max_trust_region_radius, tr_radius / radius_divisor);
        trust_region_decreasing_ratio = 0.5;
      } else {
        tr_radius *= trust_region_decreasing_ratio;
        trust_region_decreasing_ratio *= 0.5;
      }
    } else {
      if (tr_check < 0.25) {
        tr_radius /= 2;
        WORKER_LOG(tr_check << ": decrease TR radius " << tr_radius << "\n");
      }
      if (tr_check > 0.8) {
        tr_radius = std::min(max_trust_region_radius, 2 * tr_radius);//1.5
        WORKER_LOG(tr_check << ": increase TR radius " << tr_radius << "\n");
      }
      accept_step = costStart + penaltyStart >=
          (costEnd + penaltyEnd) * LocalAcceptanceRatio();
    }

    if (costStart + penaltyStart > costQuad + penaltyEnd)
      WORKER_LOG("==Start Cost < estimated cost: " << costStart + penaltyStart << " < " << costQuad + penaltyEnd << "\n");

    if (!accept_step) {
      ++trust_region_rejections;
      WORKER_LOG("Reject Start Cost < end cost: "<< costStart + penaltyStart << " < " << costEnd + penaltyEnd << "\n");
      for (int id = 0; id < delta_p.size(); ++id) {
          cameras[id] -= delta_p[id];
      }
      for (int id = 0; id < delta_l.size(); ++id) {
        landmarks[id] -= delta_l[id];
      }
      if (trust_region_attempts >= 20 || tr_radius < 1e-4) {
        cost = costStart;
        break;
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
         << " linear_solver=" << local_linear_solver
         << " linear_iterations=" << linear_iterations
         << " trust_policy=" << trust_region_policy
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
  if (landmark_refinement_steps > 0) {
    RefineLandmarksWithFixedCameras(landmark_refinement_steps);
  }
  if (!scalar_proximal_prior && metric_diagnostic_iterations > 0) {
    const NormalEquations final_normal_equations = GetNormalEquations();
    EstimateProximalDefect(
        final_normal_equations,
        full_stepSize,
        cameras,
        cameras_s,
        metric_diagnostic);
    if (collectTiming) {
      std::ostringstream metric;
      metric << "PROXIMAL_DEFECT cluster=" << cluster_id
             << " camera_squared="
             << metric_diagnostic.camera_proximal_defect_squared
             << " landmark_squared="
             << metric_diagnostic.landmark_proximal_defect_squared
             << " total_squared="
             << metric_diagnostic.proximal_defect_squared << "\n";
      EmitLocalSolveMetric(metric.str());
    }
  }
}

void RefineLandmarksWithFixedCameras(int refinement_steps) {
  for (int refinement = 0; refinement < refinement_steps;
       ++refinement) {
    const NormalEquations normal_equations = GetNormalEquations();
    SparseMatrix<double, RowMajor> landmark_inverse =
        normal_equations.landmark_hessian;
    BlockInverse<3>(landmark_inverse);
    const Eigen::VectorXd direction =
        -landmark_inverse * normal_equations.landmark_gradient;
    if (!direction.allFinite() || direction.squaredNorm() == 0.) {
      break;
    }

    const double cost_before = 2. * GetCost();
    double step_length = 1.;
    bool accepted = false;
    for (int backtrack = 0; backtrack < 8; ++backtrack) {
      for (int index = 0; index < direction.size(); ++index) {
        landmarks[index] += step_length * direction[index];
      }
      const double cost_after = 2. * GetCost();
      if (std::isfinite(cost_after) && cost_after < cost_before) {
        cost = cost_after;
        accepted = true;
        if (LocalSolveMetricsEnabled()) {
          std::ostringstream metric;
          metric << "LANDMARK_REFINEMENT cluster=" << cluster_id
                 << " refinement=" << refinement
                 << " backtracks=" << backtrack
                 << " step_length=" << step_length
                 << " before=" << cost_before
                 << " after=" << cost_after << "\n";
          EmitLocalSolveMetric(metric.str());
        }
        break;
      }
      for (int index = 0; index < direction.size(); ++index) {
        landmarks[index] -= step_length * direction[index];
      }
      step_length *= 0.5;
    }
    if (!accepted) {
      break;
    }
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
    if (objective_model == 0 && BatchedEvaluationEnabled()) {
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
    THROW_IF(normal_equation_jacobian.num_rows !=
         residual_dimension * numResiduals);
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
      for (int component = 0; component < residual_dimension; ++component) {
        const Eigen::Index row = residual_dimension * observation + component;
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
      num_clusters = numClusters;
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
      options.preconditioner_type = ceres::SCHUR_JACOBI;
      // options.linear_solver_type = ceres::CGNR;
      // options.linear_solver_type = ceres::DENSE_QR; // SHIT
      // options.max_linear_solver_iterations = 100;
        const int default_threads_per_cluster =
          std::max(1, _num_threads_machine_ / numClusters);
        const int threads_per_cluster = EnvironmentInteger(
          "BUNDLE_PALM_THREADS_PER_CLUSTER",
          default_threads_per_cluster, 1, _num_threads_machine_);
      Eigen::setNbThreads(threads_per_cluster);
      options.num_threads = threads_per_cluster; // _ceres_num_threads_; // single cpu -> still slow / bottleneck.
        WORKER_LOG("Threads per cluster: " << threads_per_cluster << "\n");
      // options.preconditioner_type = ceres::IDENTITY; // Sucks if CGNR of course. 
      // options.preconditioner_type = ceres::JACOBI; // CGNR -> jacobi anyway.
      options.max_num_iterations = 1;
      // options.minimizer_progress_to_stdout = true;
      options.minimizer_progress_to_stdout = false;
      // options.logging_type = ceres::SILENT;  
  }

  int cluster_id;
  int num_clusters = 1;
  int numCameras = 0;
  std::vector<std::uint32_t> global_camera_ids;
  int local_solve_sequence = 0;
  int numLandmarks = 0;
  int numResiduals = 0;
  const double init_be = 1e-4;
  double current_be = init_be;
  double block_curvature_multiplier = 0.;
  int metric_diagnostic_iterations = 0;
  int landmark_refinement_steps = 0;
  MetricDiagnostic metric_diagnostic;
  bool scalar_proximal_prior = false;
  double proximal_rho = 1.;
  bool split_camera_penalty = false;
  double proximal_rho_intrinsics = 1.;
  int local_linear_solver = 0;
  int trust_region_policy = 0;
  int objective_model = 0;
  int residual_dimension = 2;
  bool persistent_trust_region = false;
  bool persistent_trust_region_active = false;
  bool ceres_local_solver = false;
  int local_iterations = 1;
  double start_be = init_be;
  const double init_trust_region_radius = 1e1; // Todo: set to 1?
  double tr_radius = init_trust_region_radius; // 1e4 is ceres standard. -> Init()
  double last_tr_radius = init_trust_region_radius;
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
  std::vector<double> last_cameras;
  std::vector<double> nominal_cameras;
  //std::vector<double> best_poses;
  std::vector<double> cameras_s;
  std::vector<double> landmarks;// todo: either revert or send landmarkss all the time.
  std::vector<double> last_landmarks;
  std::vector<double> nominal_landmarks;
  std::uint64_t nominal_landmark_state_id = 0;
  std::vector<double> accepted_landmarks;
  std::vector<double> best_output_landmarks;
  std::vector<double> best_landmarks;
  std::vector<double> stepSize; // internally modelling prox term. 'sqrt' of full_stepSize 
  std::vector<double> full_stepSize; // returned to compute s update in DRS.
  std::vector<double> unorm;
  std::vector<double> vnorm;
  std::vector<double> cameraTransform;
  std::vector<double> initial_focal;
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

class SingleNodeConsensusReducer {
public:
  std::shared_ptr<const SingleNodeConsensusResult> Submit(
      std::uint64_t run_id, std::uint64_t phase_id, int cluster_id,
      int cluster_count, double relaxation,
      SingleNodeConsensusContribution contribution) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (contributions_.empty()) {
      run_id_ = run_id;
      phase_id_ = phase_id;
      cluster_count_ = cluster_count;
      relaxation_ = relaxation;
      result_.reset();
      readers_ = 0;
    }
    THROW_IF(run_id != run_id_ || phase_id != phase_id_);
    THROW_IF(cluster_count != cluster_count_ || cluster_count <= 0);
    THROW_IF(cluster_id < 0 || cluster_id >= cluster_count);
    THROW_IF(relaxation != relaxation_ || !(relaxation > 0.)
        || !(relaxation < 2.));
    THROW_IF(contributions_.count(cluster_id) != 0);
    contributions_.emplace(cluster_id, std::move(contribution));
    if (contributions_.size() == static_cast<size_t>(cluster_count_)) {
      result_ = std::make_shared<SingleNodeConsensusResult>(Reduce());
      condition_.notify_all();
    } else {
      condition_.wait(lock, [this]() { return result_ != nullptr; });
    }
    const auto result = result_;
    if (++readers_ == cluster_count_) {
      contributions_.clear();
      result_.reset();
    }
    return result;
  }

private:
  SingleNodeConsensusResult Reduce() const {
    std::uint32_t maximum_camera_id = 0;
    bool has_camera = false;
    for (int cluster = 0; cluster < cluster_count_; ++cluster) {
      const auto& contribution = contributions_.at(cluster);
      THROW_IF(contribution.global_camera_ids.size()
          != contribution.metrics.size());
      THROW_IF(contribution.cameras.size() != contribution.metrics.size());
      THROW_IF(contribution.centers.size() != contribution.metrics.size());
      for (const std::uint32_t camera : contribution.global_camera_ids) {
        maximum_camera_id = std::max(maximum_camera_id, camera);
        has_camera = true;
      }
    }
    THROW_IF(!has_camera);
    const size_t camera_count = static_cast<size_t>(maximum_camera_id) + 1;
    std::vector<Eigen::Matrix<double, 9, 9>> metric_sums(
      camera_count, Eigen::Matrix<double, 9, 9>::Zero());
    std::vector<Eigen::Matrix<double, 9, 1>> right_hand_sides(
      camera_count, Eigen::Matrix<double, 9, 1>::Zero());
    std::vector<bool> present(camera_count, false);
    for (int cluster = 0; cluster < cluster_count_; ++cluster) {
      const auto& contribution = contributions_.at(cluster);
      for (size_t local_camera = 0;
           local_camera < contribution.global_camera_ids.size();
           ++local_camera) {
        const size_t camera = contribution.global_camera_ids[local_camera];
        Eigen::Matrix<double, 9, 9> metric;
        Eigen::Matrix<double, 9, 1> reflection;
        for (int row = 0; row < 9; ++row) {
          reflection(row) = 2. * contribution.cameras[local_camera][row]
            - contribution.centers[local_camera][row];
          for (int column = 0; column < 9; ++column) {
            metric(row, column) =
              contribution.metrics[local_camera][9 * row + column];
          }
        }
        metric_sums[camera] += metric;
        right_hand_sides[camera] += metric * reflection;
        present[camera] = true;
      }
    }
    SingleNodeConsensusResult result;
    result.consensus.resize(9 * camera_count);
    for (size_t camera = 0; camera < camera_count; ++camera) {
      THROW_IF(!present[camera]);
      const Eigen::Matrix<double, 9, 1> consensus =
        metric_sums[camera].partialPivLu().solve(right_hand_sides[camera]);
      THROW_IF(!consensus.allFinite());
      for (int row = 0; row < 9; ++row) {
        result.consensus[9 * camera + row] = consensus(row);
      }
    }
    for (int cluster = 0; cluster < cluster_count_; ++cluster) {
      const auto& contribution = contributions_.at(cluster);
      for (size_t local_camera = 0;
           local_camera < contribution.global_camera_ids.size();
           ++local_camera) {
        const size_t camera = contribution.global_camera_ids[local_camera];
        Eigen::Matrix<double, 9, 9> metric;
        Eigen::Matrix<double, 9, 1> local;
        Eigen::Matrix<double, 9, 1> center;
        Eigen::Matrix<double, 9, 1> consensus;
        for (int row = 0; row < 9; ++row) {
          local(row) = contribution.cameras[local_camera][row];
          center(row) = contribution.centers[local_camera][row];
          consensus(row) = result.consensus[9 * camera + row];
          for (int column = 0; column < 9; ++column) {
            metric(row, column) =
              contribution.metrics[local_camera][9 * row + column];
          }
        }
        const Eigen::Matrix<double, 9, 1> fixed_point = local - consensus;
        const Eigen::Matrix<double, 9, 1> displacement = local - center;
        const Eigen::Matrix<double, 9, 1> reflection_projection =
          2. * local - center - consensus;
        const Eigen::Matrix<double, 9, 1> center_step =
          -relaxation_ * fixed_point;
        result.fixed_point_squared += fixed_point.dot(metric * fixed_point);
        result.proximal_displacement_squared +=
          displacement.dot(metric * displacement);
        result.reflection_projection_squared +=
          reflection_projection.dot(metric * reflection_projection);
        result.center_step_squared += center_step.dot(metric * center_step);
        result.splitting_term += 0.5 * fixed_point.dot(
          metric * (fixed_point + 2. * displacement));
      }
    }
    return result;
  }

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::map<int, SingleNodeConsensusContribution> contributions_;
  std::shared_ptr<const SingleNodeConsensusResult> result_;
  std::uint64_t run_id_ = 0;
  std::uint64_t phase_id_ = 0;
  int cluster_count_ = 0;
  int readers_ = 0;
  double relaxation_ = 1.;
};

int main() {
  // Initialize the context
  zmq::context_t context(1);

  zmq::socket_t push_socket(context, ZMQ_PUSH);
  zmq::socket_t pull_socket(context, ZMQ_PULL);

  std::mutex mtx; // Mutex for critical section.

  // Bind the socket to configurable addresses so isolated benchmark workers
  // can run alongside a user-owned server.
  const char* request_port_env = std::getenv("BUNDLE_PALM_REQUEST_PORT");
  const char* result_port_env = std::getenv("BUNDLE_PALM_RESULT_PORT");
  const std::string request_port = request_port_env ? request_port_env : "5556";
  const std::string result_port = result_port_env ? result_port_env : "5557";
  std::cout << "Starting the server on ports " << request_port << " and "
            << result_port << "..." << std::endl;
  pull_socket.bind("tcp://*:" + request_port);
  push_socket.bind("tcp://*:" + result_port);

  std::map<int, CeresProgram> cluster_to_program;
  SingleNodeConsensusReducer single_node_consensus_reducer;

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
                &single_node_consensus_reducer,
                &mtx](int cluster_id, bool omit_landmarks,
                  bool return_consensus_rhs,
                  bool single_node_consensus,
                  double consensus_relaxation,
                  std::uint64_t run_id, std::uint64_t phase_id) {
        CeresProgram &program = cluster_to_program[cluster_id];
        // std::cout << cluster_id << " Update "<< "\n";
  if (program.UsesCeresLocalSolver()) {
    program.PrepareScalarCeresPrior();
    program.Solve();
  } else {
#ifdef __ceresVersion__
    program.UpdateStepSize();
    program.Solve();
#else
          for (int local_iteration = 0;
               local_iteration < program.LocalIterations();
               ++local_iteration) {
            program.UpdateStepSizeAndSolve();//keep_cameras_fixed);
          }
#endif
  }
        std::shared_ptr<const SingleNodeConsensusResult> consensus_result;
        if (single_node_consensus) {
          consensus_result = single_node_consensus_reducer.Submit(
            run_id, phase_id, cluster_id, program.ClusterCount(),
            consensus_relaxation,
            program.BuildSingleNodeConsensusContribution());
        }
        return_cluster_proto return_proto = program.FillReturnProto(
          !omit_landmarks, return_consensus_rhs, !single_node_consensus);
        if (single_node_consensus && cluster_id == 0) {
          return_proto.set_consensus_f64(
            reinterpret_cast<const char*>(consensus_result->consensus.data()),
            consensus_result->consensus.size() * sizeof(double));
          return_proto.set_consensus_fixed_point_squared(
            consensus_result->fixed_point_squared);
          return_proto.set_consensus_proximal_displacement_squared(
            consensus_result->proximal_displacement_squared);
          return_proto.set_consensus_reflection_projection_squared(
            consensus_result->reflection_projection_squared);
          return_proto.set_consensus_center_step_squared(
            consensus_result->center_step_squared);
          return_proto.set_consensus_splitting_term(
            consensus_result->splitting_term);
          return_proto.set_has_single_node_consensus(true);
        }
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
      std::thread update_thread(update_lambda, cluster_id,
        update.omit_landmarks(), update.return_consensus_rhs(),
        update.single_node_consensus(), update.consensus_relaxation(),
        update.run_id(),
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

  if (program.UsesCeresLocalSolver()) {
    program.PrepareScalarCeresPrior();
    program.Solve();
  } else {
#ifdef __ceresVersion__
    program.UpdateStepSize();
    // program.Solve(); // only pcg!
#else
          for (int local_iteration = 0;
               local_iteration < program.LocalIterations();
               ++local_iteration) {
            program.UpdateStepSizeAndSolve();
          }
#endif
  }
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
      const std::vector<double> saved_cameras =
          costUpdate.preserve_cameras()
          ? program.CurrentCameras() : std::vector<double>();
        program.UpdateCostState(
          costUpdate); // update is local, we need to fill data in main thread.
      bool revert_lms = costUpdate.revert_lm() == 2 ? true : false;
        const bool omit_landmarks = costUpdate.omit_landmarks();
        const int landmark_refinement_steps =
          costUpdate.landmark_refinement_steps();
        THROW_IF(landmark_refinement_steps < 0 ||
             landmark_refinement_steps > 20);
      // Define a Lambda Expression
      auto cost_lambda = [&push_socket, &cluster_to_program,
              &mtx](int cluster_id, bool revert_lms,
                int landmark_refinement_steps,
                bool omit_landmarks,
                bool preserve_cameras,
                std::vector<double> saved_cameras,
                std::uint64_t run_id,
                std::uint64_t phase_id) {
        CeresProgram &program = cluster_to_program[cluster_id];
        return_cost_proto return_proto;
        const double cost = landmark_refinement_steps > 0 && !revert_lms
          ? program.EvaluateRefinedLandmarkCost(
            landmark_refinement_steps, return_proto)
          : 2 * program.GetCost(revert_lms);
        WORKER_LOG(cluster_id << ". Cost from cost: " << cost << "\n");
        return_proto.set_cost(cost);
        return_proto.set_precise_cost(cost);
        return_proto.set_cluster_id(cluster_id);
        return_proto.set_run_id(run_id);
        return_proto.set_phase_id(phase_id);
        if (!omit_landmarks &&
            (landmark_refinement_steps == 0 || revert_lms)) {
          program.AddPhysicalLandmarks(return_proto);
        }
        if (preserve_cameras) {
          program.RestoreCameras(saved_cameras);
        }
        // SerializeToArray saves memory and time?
        const size_t bytes = return_proto.ByteSizeLong();
        zmq::message_t reply(bytes);
        return_proto.SerializeToArray(reply.data(), bytes);
        std::lock_guard<std::mutex> lock(mtx);
        push_socket.send(reply, zmq::send_flags::none);
      };
            std::thread cost_thread(cost_lambda, cluster_id, revert_lms,
              landmark_refinement_steps,
              omit_landmarks,
              costUpdate.preserve_cameras(), saved_cameras,
              costUpdate.run_id(), costUpdate.phase_id());
      cost_thread.detach();
      break;
    }

    case request_proto::OptionsCase::kLandmarkState: {
      const landmark_state_proto& state = request_p.landmark_state();
      const int cluster_id = state.cluster_id();
      THROW_IF(cluster_to_program.find(cluster_id) == cluster_to_program.end());
      CeresProgram& program = cluster_to_program[cluster_id];
      landmark_state_reply_proto reply_proto;
      bool materialize_landmarks = false;
      switch (state.operation()) {
        case landmark_state_proto::SAVE_NOMINAL:
          program.SaveNominalLandmarkState(state.state_id());
          break;
        case landmark_state_proto::RESTORE_NOMINAL:
          program.RestoreNominalLandmarkState(state.state_id());
          break;
        case landmark_state_proto::DISCARD_NOMINAL:
          program.DiscardNominalLandmarkState(state.state_id());
          break;
        case landmark_state_proto::MATERIALIZE_CURRENT:
          program.ValidateNominalLandmarkState(state.state_id());
          materialize_landmarks = true;
          break;
        case landmark_state_proto::SAVE_ACCEPTED:
          program.SaveAcceptedLandmarkState();
          break;
        case landmark_state_proto::RESTORE_ACCEPTED:
          program.RestoreAcceptedLandmarkState();
          break;
        case landmark_state_proto::SAVE_BEST:
          program.SaveBestOutputLandmarkState();
          break;
        case landmark_state_proto::MATERIALIZE_BEST:
          program.MaterializeBestOutputLandmarkState(reply_proto);
          break;
        case landmark_state_proto::RESTORE_NOMINAL_ROUNDTRIP:
          program.RestoreNominalLandmarkStateWithRoundTrip(state.state_id());
          break;
        default:
          THROW_IF(true);
      }
      reply_proto.set_cluster_id(cluster_id);
      reply_proto.set_run_id(state.run_id());
      reply_proto.set_phase_id(state.phase_id());
      reply_proto.set_state_id(state.state_id());
      if (materialize_landmarks) {
        program.MaterializeCurrentLandmarkState(reply_proto);
      }
      const size_t bytes = reply_proto.ByteSizeLong();
      zmq::message_t reply(bytes);
      reply_proto.SerializeToArray(reply.data(), bytes);
      push_socket.send(reply, zmq::send_flags::none);
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