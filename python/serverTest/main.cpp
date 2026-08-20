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
#include <limits>
#include <memory>
#include <unordered_set>
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

bool SharedFixedInteriorTrialEnabled() {
  return EnvironmentInteger(
      "BUNDLE_PALM_SHARED_FIXED_INTERIOR_TRIAL", 0, 0, 1) == 1;
}

int SharedFixedInteriorTrialMaximumBacktracks() {
  return EnvironmentInteger(
      "BUNDLE_PALM_SHARED_FIXED_INTERIOR_TRIAL_MAX_BACKTRACKS", 8, 0, 8);
}

enum class CameraUpdateMode {
  kAdditive,
  kAngleAxisLeft,
  kSo3Left,
  kSo3CenterLeft,
  kSe3Left,
  kSe3Right,
};

CameraUpdateMode GetCameraUpdateMode() {
  static const CameraUpdateMode mode = [] {
    const char* value = std::getenv("BUNDLE_PALM_CAMERA_UPDATE");
    if (value == nullptr || *value == '\0' || std::strcmp(value, "additive") == 0) {
      return CameraUpdateMode::kAdditive;
    }
    if (std::strcmp(value, "angle_axis_left") == 0) {
      return CameraUpdateMode::kAngleAxisLeft;
    }
    if (std::strcmp(value, "so3_left") == 0) {
      return CameraUpdateMode::kSo3Left;
    }
    if (std::strcmp(value, "so3_center_left") == 0) {
      return CameraUpdateMode::kSo3CenterLeft;
    }
    if (std::strcmp(value, "se3_left") == 0) {
      return CameraUpdateMode::kSe3Left;
    }
    if (std::strcmp(value, "se3_right") == 0) {
      return CameraUpdateMode::kSe3Right;
    }
    throw std::runtime_error(
      "BUNDLE_PALM_CAMERA_UPDATE must be additive, angle_axis_left, "
      "so3_left, so3_center_left, se3_left, or se3_right");
  }();
  return mode;
}
bool FreezeBlockMetricEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_FREEZE_BLOCK_METRIC", 0, 0, 1) == 1;
  return enabled;
}

bool ConsensusUnflooredCameraDiagonalEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_CONSENSUS_UNFLOORED_CAMERA_DIAGONAL", 0, 0, 1) == 1;
  return enabled;
}

bool ManifoldCameraUpdatesEnabled() {
  return GetCameraUpdateMode() != CameraUpdateMode::kAdditive;
}

// Highly active: changing this floor alters convergence paths and final costs.
double CameraDiagonalRelativeFloor() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CAMERA_DIAGONAL_FLOOR", 1e-48, 0.0, 1.0);
  return value;
}

double CameraDiagonalTranslationFloor() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CAMERA_DIAGONAL_TRANSLATION_FLOOR",
      CameraDiagonalRelativeFloor(), 0.0, 1.0);
  return value;
}

double CameraDiagonalRotationFloor() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CAMERA_DIAGONAL_ROTATION_FLOOR",
      CameraDiagonalRelativeFloor(), 0.0, 1.0);
  return value;
}

double CameraDiagonalIntrinsicsFloor() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CAMERA_DIAGONAL_INTRINSICS_FLOOR",
      CameraDiagonalRelativeFloor(), 0.0, 1.0);
  return value;
}

const std::unordered_set<std::uint32_t>& CameraDiagonalExcludedCameraIds() {
  static const std::unordered_set<std::uint32_t> ids = [] {
    std::unordered_set<std::uint32_t> result;
    const char* text = std::getenv(
        "BUNDLE_PALM_CAMERA_DIAGONAL_EXCLUDED_CAMERA_IDS");
    if (text == nullptr || *text == '\0') {
      return result;
    }
    std::istringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
      if (token.empty()) {
        throw std::runtime_error(
            "BUNDLE_PALM_CAMERA_DIAGONAL_EXCLUDED_CAMERA_IDS contains an empty ID");
      }
      char* end = nullptr;
      errno = 0;
      const unsigned long value = std::strtoul(token.c_str(), &end, 10);
      if (errno != 0 || end == token.c_str() || *end != '\0' ||
          value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "BUNDLE_PALM_CAMERA_DIAGONAL_EXCLUDED_CAMERA_IDS must be comma-separated camera IDs");
      }
      result.insert(static_cast<std::uint32_t>(value));
    }
    return result;
  }();
  return ids;
}

// Inactive in the standard build; only used by the __ceresVersion__ path.
double BlockSqrtEigenvalueFloor() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_BLOCK_SQRT_EIGENVALUE_FLOOR", 1e-16, 0.0, 1.0);
  return value;
}

// Never hit in measured runs; reasonable perturbations had no effect.
double LandmarkPreconditionerFloor() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_LANDMARK_PRECONDITIONER_FLOOR", 1e-24, 0.0, 1.0);
  return value;
}

bool DisableLandmarkPreconditioningEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_DISABLE_LANDMARK_PRECONDITIONING", 0, 0, 1) == 1;
  return enabled;
}

// Highly active: changes early progress, final costs, and runtime.
double CameraTrustDiagonalScale() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CAMERA_TRUST_DIAGONAL_SCALE", 1e-4, 0.0, 1.0);
  return value;
}

// Inactive in the standard build; only used by the __ceresVersion__ path.
double CameraBlockScale() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CAMERA_BLOCK_SCALE", 1e1, 1e-12, 1e12);
  return value;
}

// Inactive in the standard build; only used by the _const_diag_ path.
double ConstantDiagonalMaximumFloor() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CONST_DIAGONAL_MAXIMUM_FLOOR", 1e-32, 0.0, 1.0);
  return value;
}

// Inactive in the standard build; only used by the _const_diag_ path.
double ConstantDiagonalRelativeFloor() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CONST_DIAGONAL_RELATIVE_FLOOR", 1e-3, 0.0, 1.0);
  return value;
}

// Never hit in measured runs; observed block maxima were many orders larger.
double CameraDiagonalMaximumGuard() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CAMERA_DIAGONAL_MAXIMUM_GUARD", 1e-32, 0.0, 1.0);
  return value;
}

// Little or no effect for reasonable values in measured runs.
double MinimumTrustRegionRadius() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_MINIMUM_TRUST_REGION_RADIUS", 1e-4, 0.0, 1e6);
  return value;
}

// Inactive in the standard build; only used by the __ceresVersion__ path.
double LegacyLandmarkJacobianSqrtFloor() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_LEGACY_LANDMARK_JACOBIAN_SQRT_FLOOR", 1e-10, 0.0, 1.0);
  return value;
}

// Hit by near-zero entries, but reasonable perturbations had no observed effect.
double CameraPreconditionerDiagonalFloor() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CAMERA_PRECONDITIONER_DIAGONAL_FLOOR", 1e-36, 0.0, 1.0);
  return value;
}

// Highly active: strongly changes convergence quality, rejection count, and cost.
double CameraDiagonalMetricScale() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CAMERA_DIAGONAL_METRIC_SCALE", 25.0, 0.0, 1e12);
  return value;
}

double So3TranslationMetricRatio() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_SO3_TRANSLATION_METRIC_RATIO", 1.0, 1e-6, 1e6);
  return value;
}

double InitialTrustRegionRadius() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_INITIAL_TRUST_REGION_RADIUS", 1e1, 1e-12, 1e12);
  return value;
}

double MaximumTrustRegionRadius() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_MAXIMUM_TRUST_REGION_RADIUS", 1e6, 1e-12, 1e16);
  return value;
}

double DabaInitialTrustRegionCap() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_DABA_INITIAL_TRUST_REGION_CAP", 100., 1e-12, 1e16);
  return value;
}

double NesterovSchurLipschitz() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_NESTEROV_SCHUR_LIPSCHITZ", 0.9, 1e-6, 1e6);
  return value;
}

int NesterovStopCheckInterval() {
  static const int value = EnvironmentInteger(
      "BUNDLE_PALM_NESTEROV_STOP_CHECK_INTERVAL", 1, 1, 10000);
  return value;
}

int SchurPcgMaximumIterations() {
  static const int value = EnvironmentInteger(
      "BUNDLE_PALM_SCHUR_PCG_MAX_ITERATIONS", 400, 1, 10000);
  return value;
}

double SchurPcgRelativeTolerance() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_SCHUR_PCG_RELATIVE_TOLERANCE", 1e-2, 1e-12, 1.0);
  return value;
}

double SchurPcgQTolerance() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_SCHUR_PCG_Q_TOLERANCE", 0., 0., 1.0);
  return value;
}

bool SchurPcgJacobiPreconditionerEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_SCHUR_PCG_JACOBI_PRECONDITIONER", 0, 0, 1) == 1;
  return enabled;
}

int CentralizedCeresIterations() {
  static const int value = EnvironmentInteger(
      "BUNDLE_PALM_CENTRALIZED_CERES_ITERATIONS", 90, 1, 1000);
  return value;
}

double CentralizedCeresInitialRadius() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_CENTRALIZED_CERES_INITIAL_RADIUS", 1e4, 1e-12, 1e12);
  return value;
}

int CentralizedCeresMaximumLinearIterations() {
  static const int value = EnvironmentInteger(
      "BUNDLE_PALM_CENTRALIZED_CERES_MAX_LINEAR_ITERATIONS", 500, 1, 10000);
  return value;
}

bool CentralizedCeresJacobiScalingEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_CENTRALIZED_CERES_JACOBI_SCALING", 1, 0, 1) == 1;
  return enabled;
}

bool CentralizedCeresTangentDiagnosticEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_CENTRALIZED_CERES_TANGENT_DIAGNOSTIC", 0, 0, 1) == 1;
  return enabled;
}

bool CentralizedCeresSparseSchurEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_CENTRALIZED_CERES_SPARSE_SCHUR", 0, 0, 1) == 1;
  return enabled;
}

bool DiagonalTrustDampingEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_DIAGONAL_TRUST_DAMPING", 0, 0, 1) == 1;
  return enabled;
}

bool BaeTrustScheduleEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_BAE_TRUST_SCHEDULE", 0, 0, 1) == 1;
  return enabled;
}

bool CumulativeDiagonalDampingEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_CUMULATIVE_DIAGONAL_DAMPING", 0, 0, 1) == 1;
  return enabled || BaeTrustScheduleEnabled();
}
bool NesterovRelativeResidualEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_NESTEROV_RELATIVE_RESIDUAL", 0, 0, 1) == 1;
  return enabled;
}

bool DirectTangentNormalEquationsEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS", 0, 0, 1) == 1;
  return enabled;
}
bool SchurProximalMetricEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_SCHUR_PROXIMAL_METRIC", 0, 0, 1) == 1;
  return enabled;
}

bool CoupledSchurProximalMetricEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_COUPLED_SCHUR_PROXIMAL_METRIC", 0, 0, 1) == 1;
  return enabled;
}

bool FactorizedCoupledSchurProximalMetricEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_FACTORIZED_COUPLED_SCHUR_PROXIMAL_METRIC",
      0, 0, 1) == 1;
  return enabled;
}

double CoupledSchurProximalMetricStabilization() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_COUPLED_SCHUR_PROXIMAL_METRIC_STABILIZATION",
      1., 0., 16.);
  return value;
}

int FactorizedSchurProximalMetricStabilizationBuckets() {
  static const int value = EnvironmentInteger(
      "BUNDLE_PALM_FACTORIZED_SCHUR_PROXIMAL_METRIC_STABILIZATION_BUCKETS",
      32, 1, 4096);
  return value;
}

bool DisableLocalProximalTermEnabled() {
  static const bool enabled = EnvironmentInteger(
      "BUNDLE_PALM_DISABLE_LOCAL_PROXIMAL_TERM", 0, 0, 1) == 1;
  return enabled;
}

int PobaDiagnosticIterations() {
  static const int value = EnvironmentInteger(
      "BUNDLE_PALM_POBA_DIAGNOSTIC_ITERATIONS", 0, 0, 100);
  return value;
}

double PobaBlockRelativeFloor() {
  static const double value = EnvironmentDouble(
      "BUNDLE_PALM_POBA_BLOCK_RELATIVE_FLOOR", 0., 0., 1.);
  return value;
}

template <int BlockSize>
void FloorSymmetricBlocks(SparseMatrix<double, RowMajor>& matrix,
                          double relative_floor) {
  if (!(relative_floor > 0.)) {
    return;
  }
  const int block_count = matrix.rows() / BlockSize;
  for (int block_index = 0; block_index < block_count; ++block_index) {
    Eigen::Matrix<double, BlockSize, BlockSize> block;
    for (int row = 0; row < BlockSize; ++row) {
      for (int column = 0; column < BlockSize; ++column) {
        block(row, column) = matrix.coeff(
            BlockSize * block_index + row,
            BlockSize * block_index + column);
      }
    }
    block = 0.5 * (block + block.transpose()).eval();
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<double, BlockSize, BlockSize>> solver(block);
    Eigen::Matrix<double, BlockSize, 1> eigenvalues = solver.eigenvalues();
    double floor = relative_floor * eigenvalues.maxCoeff();
    if (!(floor > 0.) || !std::isfinite(floor)) {
      floor = relative_floor;
    }
    eigenvalues = eigenvalues.array().max(floor);
    block = solver.eigenvectors() * eigenvalues.asDiagonal()
        * solver.eigenvectors().transpose();
    for (int row = 0; row < BlockSize; ++row) {
      for (int column = 0; column < BlockSize; ++column) {
        matrix.coeffRef(BlockSize * block_index + row,
                        BlockSize * block_index + column) = block(row, column);
      }
    }
  }
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

double Quantile(std::vector<double> values, double probability) {
  THROW_IF(values.empty() || probability < 0.0 || probability > 1.0);
  const size_t index = static_cast<size_t>(
      std::floor(probability * static_cast<double>(values.size() - 1)));
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return values[index];
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
    const double eigenvalueFloor = BlockSqrtEigenvalueFloor();
    double minimumSqrtEigenvalue = std::numeric_limits<double>::infinity();
    long long flooredEigenvalues = 0;
    //std::cout << "before  "<< values[0]<< " " << values[1]<< " " << values[2]<< " " << values[3] << "\n";
  #pragma omp parallel for num_threads(options.num_threads) \
    reduction(min:minimumSqrtEigenvalue) reduction(+:flooredEigenvalues)
    for (int i = 0; i < numrows / N; i++) {
        auto matNxN = Map< Matrix<double, N, N> > (&(values[i * N*N]));//,  Eigen::Stride<0, 0>);
        //std::cout << "before "<< matNxN << " \n";

        Eigen::SelfAdjointEigenSolver<Matrix<double,N,N> > eigensolver;
        eigensolver.computeDirect(matNxN, Eigen::DecompositionOptions::ComputeEigenvectors);
        //VPQ_EXPECT_EQ(eigensolver.info(), Eigen::Success);

        // SqrtCovEigenValues are sorted in decreasing order.
        const Eigen::Vector<double, N> rawSqrtEigenValues =
          eigensolver.eigenvalues().cwiseAbs().cwiseSqrt();
        minimumSqrtEigenvalue =
          std::min(minimumSqrtEigenvalue, rawSqrtEigenValues.minCoeff());
        flooredEigenvalues +=
          (rawSqrtEigenValues.array() < eigenvalueFloor).count();
        const Eigen::Vector<double, N> sqrtEigenValues =
          rawSqrtEigenValues.cwiseMax(eigenvalueFloor);//.cwiseMax(lowerBoundSquared).cwiseSqrt().cwiseInverse();
        // recall : i had here min ev >= 1e-6 * maxEv. Could return a diag matrix
        matNxN = eigensolver.eigenvectors() * sqrtEigenValues.asDiagonal() * eigensolver.eigenvectors().transpose();

        //auto matNxN_out = Map< Matrix<double,N,N> > (&(values[i * N*N]));//,  Eigen::Stride<0, 0>);
        //std::cout << "after  "<< matNxN_out.transpose() * matNxN_out << " \n";
    }
    if (LocalSolveMetricsEnabled()) {
      std::ostringstream metric;
      metric << "BLOCK_SQRT_FLOOR blocks=" << numrows / N
             << " entries=" << numrows
             << " floored=" << flooredEigenvalues
             << " minimum=" << minimumSqrtEigenvalue
             << " floor=" << eigenvalueFloor << "\n";
      EmitLocalSolveMetric(metric.str());
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
    const Matrix<double, N, N> original = matNxN;
        Matrix<double, N, N> inverse = original.inverse().eval();
    if (!inverse.allFinite()) {
          const double scale = original.cwiseAbs().maxCoeff();
          if (scale > 0. && std::isfinite(scale)) {
            const Matrix<double, N, N> scaled = original / scale;
            inverse = scaled.ldlt().solve(
                Matrix<double, N, N>::Identity()) / scale;
          }
        }
        if (!inverse.allFinite()) {
      const Eigen::Matrix<double, N, 1> eigenvalues =
        Eigen::SelfAdjointEigenSolver<Matrix<double, N, N>>(
          0.5 * (original + original.transpose())).eigenvalues();
#pragma omp critical
      std::cerr << "Non-finite block inverse: block_size=" << N
          << " block=" << i
          << " input_finite=" << original.allFinite()
          << " input_max=" << original.cwiseAbs().maxCoeff()
          << " determinant=" << original.determinant()
          << " eigenvalues=" << eigenvalues.transpose()
          << std::endl;
    }
    matNxN = inverse;
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
Diagonal(SparseMatrix<double, RowMajor>& mat, int cluster_id = -1,
         bool collect_camera_metrics = false, int outer_iteration = -1,
         int oracle_kind = 0, const char* metric_source = "unspecified",
         const std::vector<std::uint32_t>* global_camera_ids = nullptr) {
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
      mv = std::max(ConstantDiagonalMaximumFloor(), mv);
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
          diagdiag(N*b + id) = std::max(ConstantDiagonalRelativeFloor() * mv, diagdiag(N*b + id)); // not < 1/inf. also 1e10: ok, 1e-6: bit worse

          //diagdiag(N*b + id) = std::max(variableThresh * mv, diagdiag(N*b + id)); // not < 1/inf. also 1e10: ok, 1e-6: bit worse
      }
    }
  }
#else
  if (N == 9) {
    auto& blockDiagonal = diag.diagonal();
    const bool collectMetrics = collect_camera_metrics;
    std::vector<double> relativeDiagonals(
      collectMetrics ? blockDiagonal.size() : 0);
    std::vector<double> blockMaxima(
      collectMetrics ? mat.rows() / N : 0);
    int flooredEntries = 0;
    int zeroEntries = 0;
    int guardedBlocks = 0;
    double minimumRelativeDiagonal = 1.0;
    double minimumBlockMaximum = std::numeric_limits<double>::infinity();
#pragma omp parallel for num_threads(options.num_threads) \
    reduction(+:flooredEntries, zeroEntries, guardedBlocks) \
    reduction(min:minimumRelativeDiagonal, minimumBlockMaximum)
    for (int block = 0; block < mat.rows() / N; ++block) {
      const auto cameraDiagonal = blockDiagonal.template segment<N>(N * block);
      const double rawMaxDiagonal = cameraDiagonal.maxCoeff();
      minimumBlockMaximum = std::min(minimumBlockMaximum, rawMaxDiagonal);
      const double maxDiagonal =
          std::max(CameraDiagonalMaximumGuard(), rawMaxDiagonal);
        const bool excluded = global_camera_ids != nullptr &&
          CameraDiagonalExcludedCameraIds().count(
            (*global_camera_ids)[block]) != 0;
        const std::array<double, 3> relativeFloors = excluded
          ? std::array<double, 3>{0., 0., 0.}
          : std::array<double, 3>{
            CameraDiagonalTranslationFloor(), CameraDiagonalRotationFloor(),
            CameraDiagonalIntrinsicsFloor()};
      if (collectMetrics) {
        blockMaxima[block] = rawMaxDiagonal;
        guardedBlocks += rawMaxDiagonal < CameraDiagonalMaximumGuard();
        for (int coordinate = 0; coordinate < N; ++coordinate) {
          relativeDiagonals[N * block + coordinate] =
              cameraDiagonal[coordinate] / maxDiagonal;
          zeroEntries += cameraDiagonal[coordinate] == 0.0;
            flooredEntries += cameraDiagonal[coordinate]
              < relativeFloors[coordinate / 3] * maxDiagonal;
          minimumRelativeDiagonal = std::min(
              minimumRelativeDiagonal,
              cameraDiagonal[coordinate] / maxDiagonal);
        }
      }
      for (int coordinate = 0; coordinate < N; ++coordinate) {
        blockDiagonal[N * block + coordinate] = std::max(
            cameraDiagonal[coordinate],
            relativeFloors[coordinate / 3] * maxDiagonal);
      }
      // if (cameraDiagonal.minCoeff() < 1e-24 * maxDiagonal) {
      //   blockDiagonal.template segment<N>(N * block) =
      //       cameraDiagonal.cwiseMax(1e-24 * maxDiagonal);
      // }
    }
    if (collectMetrics) {
      THROW_IF(global_camera_ids != nullptr &&
          global_camera_ids->size() != static_cast<size_t>(mat.rows() / N));
      std::array<std::vector<double>, 3> groupRelativeDiagonals;
      std::vector<int> cameraHitCounts(mat.rows() / N, 0);
      std::array<int, 3> groupHits = {0, 0, 0};
      std::ostringstream hitCoordinates;
      bool firstHitCoordinate = true;
      for (int block = 0; block < mat.rows() / N; ++block) {
        for (int coordinate = 0; coordinate < N; ++coordinate) {
          const double relative = relativeDiagonals[N * block + coordinate];
          const int group = coordinate / 3;
          groupRelativeDiagonals[group].push_back(relative);
            const bool excluded = global_camera_ids != nullptr &&
              CameraDiagonalExcludedCameraIds().count(
                (*global_camera_ids)[block]) != 0;
            const std::array<double, 3> relativeFloors = excluded
              ? std::array<double, 3>{0., 0., 0.}
              : std::array<double, 3>{
                CameraDiagonalTranslationFloor(), CameraDiagonalRotationFloor(),
                CameraDiagonalIntrinsicsFloor()};
          if (relative < relativeFloors[group]) {
            ++cameraHitCounts[block];
            ++groupHits[group];
            if (!firstHitCoordinate) {
              hitCoordinates << ",";
            }
            hitCoordinates
              << (global_camera_ids == nullptr
                    ? static_cast<std::uint32_t>(block)
                    : (*global_camera_ids)[block])
              << ":" << coordinate << ":" << relative;
            firstHitCoordinate = false;
          }
        }
      }
      std::vector<double> sortedRelativeDiagonals = relativeDiagonals;
      std::sort(sortedRelativeDiagonals.begin(), sortedRelativeDiagonals.end());
      const int bottomCount = std::min<int>(8, sortedRelativeDiagonals.size());
      const double bottomMean = bottomCount > 0
          ? std::accumulate(sortedRelativeDiagonals.begin(),
              sortedRelativeDiagonals.begin() + bottomCount, 0.) / bottomCount
          : std::numeric_limits<double>::quiet_NaN();
      std::vector<int> sortedCameraHitCounts = cameraHitCounts;
      std::sort(sortedCameraHitCounts.begin(), sortedCameraHitCounts.end(),
          std::greater<int>());
      const int hitCameras = std::count_if(
          cameraHitCounts.begin(), cameraHitCounts.end(),
          [](int count) { return count > 0; });
      const int topCameraCount = std::max<int>(
          1, (sortedCameraHitCounts.size() + 9) / 10);
      const int topCameraHits = std::accumulate(
          sortedCameraHitCounts.begin(),
          sortedCameraHitCounts.begin() + topCameraCount, 0);
      double hitHhi = 0.;
      if (flooredEntries > 0) {
        for (const int count : cameraHitCounts) {
          const double share = static_cast<double>(count) / flooredEntries;
          hitHhi += share * share;
        }
      }
      std::ostringstream metric;
      metric << "CAMERA_DIAGONAL cluster=" << cluster_id
        << " outer_iteration=" << outer_iteration
        << " oracle_kind=" << oracle_kind
        << " source=" << metric_source
         << " blocks=" << mat.rows() / N
         << " entries=" << blockDiagonal.size()
         << " floored=" << flooredEntries
         << " zeros=" << zeroEntries
         << " guarded_blocks=" << guardedBlocks
         << " min_relative=" << minimumRelativeDiagonal
            << " q001_relative=" << Quantile(relativeDiagonals, 0.001)
         << " q01_relative=" << Quantile(relativeDiagonals, 0.01)
            << " bottom8_mean=" << bottomMean
            << " translation_min=" << *std::min_element(
              groupRelativeDiagonals[0].begin(), groupRelativeDiagonals[0].end())
            << " translation_q001=" << Quantile(groupRelativeDiagonals[0], 0.001)
            << " translation_q01=" << Quantile(groupRelativeDiagonals[0], 0.01)
            << " translation_hits=" << groupHits[0]
            << " rotation_min=" << *std::min_element(
              groupRelativeDiagonals[1].begin(), groupRelativeDiagonals[1].end())
            << " rotation_q001=" << Quantile(groupRelativeDiagonals[1], 0.001)
            << " rotation_q01=" << Quantile(groupRelativeDiagonals[1], 0.01)
            << " rotation_hits=" << groupHits[1]
            << " intrinsics_min=" << *std::min_element(
              groupRelativeDiagonals[2].begin(), groupRelativeDiagonals[2].end())
            << " intrinsics_q001=" << Quantile(groupRelativeDiagonals[2], 0.001)
            << " intrinsics_q01=" << Quantile(groupRelativeDiagonals[2], 0.01)
            << " intrinsics_hits=" << groupHits[2]
            << " hit_cameras=" << hitCameras
            << " max_hits_per_camera=" << (sortedCameraHitCounts.empty()
              ? 0 : sortedCameraHitCounts.front())
            << " top10_camera_hit_share=" << (flooredEntries > 0
              ? static_cast<double>(topCameraHits) / flooredEntries : 0.)
            << " hit_hhi=" << hitHhi
            << " hit_coordinates=" << (firstHitCoordinate
              ? "none" : hitCoordinates.str())
         << " min_block_maximum=" << minimumBlockMaximum
         << " q01_block_maximum=" << Quantile(blockMaxima, 0.01)
         << " floor=" << CameraDiagonalRelativeFloor()
         << " translation_floor=" << CameraDiagonalTranslationFloor()
         << " rotation_floor=" << CameraDiagonalRotationFloor()
         << " intrinsics_floor=" << CameraDiagonalIntrinsicsFloor()
         << " maximum_guard=" << CameraDiagonalMaximumGuard() << "\n";
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
    landmark_edges_.assign(num_landmarks_, {});
    for (int edge = 0; edge < edges_.size(); ++edge) {
      landmark_edges_[edges_[edge].landmark].push_back(edge);
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
  size_t EdgeCount() const { return edges_.size(); }
  int LandmarkCount() const { return num_landmarks_; }
  const std::vector<CameraLandmarkEdge>& Edges() const { return edges_; }

  double SquaredNorm() const {
    double result = 0.;
    for (const CameraLandmarkEdge& edge : edges_) {
      for (double value : edge.values) {
        result += value * value;
      }
    }
    return result;
  }

  void LeftMultiplyByCameraBlockTransposes(
      const std::vector<Eigen::Matrix<double, 9, 9>>& blocks) {
    THROW_IF(blocks.size() != num_cameras_);
    for (CameraLandmarkEdge& edge : edges_) {
      Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> values(
          edge.values.data());
      values = blocks[edge.camera].transpose() * values;
    }
  }

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

  void SubtractSchurDiagonal(
      const SparseMatrix<double, RowMajor>& landmark_inverse,
      SparseMatrix<double, RowMajor>& camera_blocks) const {
    for (const CameraLandmarkEdge& edge : edges_) {
      Eigen::Matrix3d inverse_block;
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          inverse_block(row, column) = landmark_inverse.coeff(
              3 * edge.landmark + row, 3 * edge.landmark + column);
        }
      }
      const Eigen::Map<
          const Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> cross(
              edge.values.data());
      const Eigen::Matrix<double, 9, 9> contribution =
          cross * inverse_block * cross.transpose();
      for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
          camera_blocks.coeffRef(9 * edge.camera + row,
              9 * edge.camera + column) -= contribution(row, column);
        }
      }
    }
  }

  void AddBucketedSchurOffDiagonalFrobeniusBounds(
      const SparseMatrix<double, RowMajor>& landmark_inverse,
      const std::vector<double>& camera_multipliers,
      double scale,
      int bucket_count,
      SparseMatrix<double, RowMajor>& camera_blocks) const {
    THROW_IF(landmark_inverse.rows() != cols() ||
             camera_blocks.rows() != rows() ||
             camera_multipliers.size() !=
                 static_cast<size_t>(num_cameras_) ||
             scale < 0. || !std::isfinite(scale) || bucket_count <= 0);
    if (scale == 0.) {
      return;
    }
    for (int bucket = 0; bucket < bucket_count; ++bucket) {
      std::map<std::pair<int, int>, Eigen::Matrix<double, 9, 9>> pair_blocks;
      for (int landmark = 0; landmark < num_landmarks_; ++landmark) {
        Eigen::Matrix3d inverse_block;
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            inverse_block(row, column) = landmark_inverse.coeff(
                3 * landmark + row, 3 * landmark + column);
          }
        }
        const std::vector<int>& landmark_edges = landmark_edges_[landmark];
        for (int left_index = 0; left_index < landmark_edges.size();
             ++left_index) {
          const CameraLandmarkEdge& left = edges_[landmark_edges[left_index]];
          if (!(camera_multipliers[left.camera] > 0.)) {
            continue;
          }
          const Eigen::Map<
              const Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> left_cross(
                  left.values.data());
          for (int right_index = left_index + 1;
               right_index < landmark_edges.size(); ++right_index) {
            const CameraLandmarkEdge& right =
                edges_[landmark_edges[right_index]];
            if (!(camera_multipliers[right.camera] > 0.)) {
              continue;
            }
            const int row_camera = std::min(left.camera, right.camera);
            const int column_camera = std::max(left.camera, right.camera);
            const std::uint64_t key =
                static_cast<std::uint64_t>(row_camera) * num_cameras_
                + column_camera;
            if (key % bucket_count !=
                static_cast<std::uint64_t>(bucket)) {
              continue;
            }
            const Eigen::Map<
                const Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> right_cross(
                    right.values.data());
            Eigen::Matrix<double, 9, 9> contribution =
                left_cross * inverse_block * right_cross.transpose();
            if (left.camera > right.camera) {
              contribution.transposeInPlace();
            }
            const std::pair<int, int> camera_pair{
                row_camera, column_camera};
            auto block = pair_blocks.find(camera_pair);
            if (block == pair_blocks.end()) {
              block = pair_blocks.emplace(
                  camera_pair, Eigen::Matrix<double, 9, 9>::Zero()).first;
            }
            block->second -= contribution;
          }
        }
      }
      for (const auto& [camera_pair, block] : pair_blocks) {
        const double bound = scale * block.norm();
        for (const int camera : {camera_pair.first, camera_pair.second}) {
          for (int parameter = 0; parameter < 9; ++parameter) {
            camera_blocks.coeffRef(
                9 * camera + parameter,
                9 * camera + parameter) += bound;
          }
        }
      }
    }
  }

  void ScaleCameraRows(const std::vector<double>& scales) {
    THROW_IF(scales.size() != static_cast<size_t>(num_cameras_));
    for (CameraLandmarkEdge& edge : edges_) {
      THROW_IF(scales[edge.camera] < 0. ||
               !std::isfinite(scales[edge.camera]));
      for (double& value : edge.values) {
        value *= scales[edge.camera];
      }
    }
  }

  void ZeroLandmarksWithFewerThanTwoActiveEdges() {
    for (const std::vector<int>& landmark_edges : landmark_edges_) {
      int active_edges = 0;
      for (const int edge_index : landmark_edges) {
        const CameraLandmarkEdge& edge = edges_[edge_index];
        active_edges += std::any_of(
            edge.values.begin(), edge.values.end(),
            [](double value) { return value != 0.; });
      }
      if (active_edges >= 2) {
        continue;
      }
      for (const int edge_index : landmark_edges) {
        edges_[edge_index].values.fill(0.);
      }
    }
  }

  std::map<std::pair<int, int>, Eigen::Matrix<double, 9, 9>>
  MaterializeSchurComplement(
      const SparseMatrix<double, RowMajor>& camera_hessian,
      const SparseMatrix<double, RowMajor>& landmark_inverse) const {
    THROW_IF(camera_hessian.rows() != rows() ||
             landmark_inverse.rows() != cols());
    std::map<std::pair<int, int>, Eigen::Matrix<double, 9, 9>> blocks;
    for (int camera = 0; camera < num_cameras_; ++camera) {
      Eigen::Matrix<double, 9, 9>& diagonal = blocks[{camera, camera}];
      for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
          diagonal(row, column) = camera_hessian.coeff(
              9 * camera + row, 9 * camera + column);
        }
      }
    }
    for (int landmark = 0; landmark < num_landmarks_; ++landmark) {
      Eigen::Matrix3d inverse_block;
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          inverse_block(row, column) = landmark_inverse.coeff(
              3 * landmark + row, 3 * landmark + column);
        }
      }
      const std::vector<int>& landmark_edges = landmark_edges_[landmark];
      for (int left_index = 0; left_index < landmark_edges.size(); ++left_index) {
        const CameraLandmarkEdge& left = edges_[landmark_edges[left_index]];
        const Eigen::Map<const Eigen::Matrix<double, 9, 3, Eigen::RowMajor>>
            left_cross(left.values.data());
        for (int right_index = left_index;
             right_index < landmark_edges.size(); ++right_index) {
          const CameraLandmarkEdge& right = edges_[landmark_edges[right_index]];
          const Eigen::Map<
              const Eigen::Matrix<double, 9, 3, Eigen::RowMajor>>
              right_cross(right.values.data());
          Eigen::Matrix<double, 9, 9> contribution =
              left_cross * inverse_block * right_cross.transpose();
          const int row_camera = std::min(left.camera, right.camera);
          const int column_camera = std::max(left.camera, right.camera);
          if (left.camera > right.camera) {
            contribution.transposeInPlace();
          }
          const std::pair<int, int> key{row_camera, column_camera};
          auto block = blocks.find(key);
          if (block == blocks.end()) {
            block = blocks.emplace(
                key, Eigen::Matrix<double, 9, 9>::Zero()).first;
          }
          block->second -= contribution;
        }
      }
    }
    return blocks;
  }

 private:
  int num_cameras_ = 0;
  int num_landmarks_ = 0;
  std::vector<CameraLandmarkEdge> edges_;
  std::vector<int> observation_edges_;
  std::vector<std::vector<int>> landmark_edges_;
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
    double unique_camera_interior_defect_squared =
      std::numeric_limits<double>::quiet_NaN();
    double landmark_interior_defect_squared =
      std::numeric_limits<double>::quiet_NaN();
    double interior_defect_squared =
      std::numeric_limits<double>::quiet_NaN();
  int iterations = 0;
};

  void EstimateInteriorDefect(
    const NormalEquations& normal_equations,
    const std::vector<double>& camera_proximal_multipliers,
    MetricDiagnostic& diagnostic) {
    SparseMatrix<double, RowMajor> camera_inverse =
      normal_equations.camera_hessian;
    FloorSymmetricBlocks<9>(camera_inverse, 1e-16);
    BlockInverse<9>(camera_inverse);
    diagnostic.unique_camera_interior_defect_squared = 0.;
    for (int camera = 0; camera < camera_proximal_multipliers.size(); ++camera) {
    if (camera_proximal_multipliers[camera] != 0.) {
      continue;
    }
    const Eigen::VectorXd gradient =
      normal_equations.camera_gradient.segment<9>(9 * camera);
    diagnostic.unique_camera_interior_defect_squared += gradient.dot(
      camera_inverse.block(9 * camera, 9 * camera, 9, 9) * gradient);
    }
    SparseMatrix<double, RowMajor> landmark_inverse =
      normal_equations.landmark_hessian;
    FloorSymmetricBlocks<3>(landmark_inverse, 1e-16);
    BlockInverse<3>(landmark_inverse);
    diagnostic.landmark_interior_defect_squared = std::max(
      0., normal_equations.landmark_gradient.dot(
        landmark_inverse * normal_equations.landmark_gradient));
    diagnostic.unique_camera_interior_defect_squared = std::max(
      0., diagnostic.unique_camera_interior_defect_squared);
    diagnostic.interior_defect_squared =
      diagnostic.unique_camera_interior_defect_squared
      + diagnostic.landmark_interior_defect_squared;
  }

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

bool stop_criterion(double x_squared_norm, double gradient_squared_norm,
                    double lip, int i, double tolerance) {
  const double iterations = i + 1.;
  const double scaled_eps = tolerance * lip;
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
      const std::vector<double>& consensusMetric = ConsensusMetricBlocks();
      THROW_IF(consensusMetric.size() != static_cast<size_t>(81 * numCameras));
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
              consensusMetric[81 * camera + 9 * row + column];
          }
        }
      }
      return contribution;
    }

    void ResetProgram(const program_proto &pro) {
      Init(pro.num_clusters());
      last_linear_iterations = 0;
      last_linear_relative_residual =
          std::numeric_limits<double>::quiet_NaN();
      numCameras = pro.cameras_size() / 9;
      numLandmarks = pro.landmarks_size() / 3;
      numResiduals =  pro.observations_size() / 2;
      global_camera_ids.assign(
        pro.global_camera_id().begin(), pro.global_camera_id().end());
      THROW_IF(!global_camera_ids.empty()
          && global_camera_ids.size() != static_cast<size_t>(numCameras));
      camera_proximal_multipliers.assign(
          pro.camera_proximal_multiplier().begin(),
          pro.camera_proximal_multiplier().end());
      if (camera_proximal_multipliers.empty()) {
        camera_proximal_multipliers.assign(numCameras, 1.);
      }
      THROW_IF(camera_proximal_multipliers.size()
          != static_cast<size_t>(numCameras));
      local_iterations = std::max(1, std::min(20, pro.iterations()));
      frozen_block_metric_initialized = false;
      camera_proximal_metric.resize(0, 0);
      factorized_proximal_active = false;
      factorized_proximal_diagonal.resize(0, 0);
      factorized_proximal_landmark_inverse.resize(0, 0);
      scalar_proximal_prior = pro.scalar_proximal_prior();
      block_curvature_multiplier = pro.block_curvature_multiplier();
      metric_diagnostic_iterations = pro.metric_diagnostic_iterations();
      outer_iteration = pro.outer_iteration();
      oracle_kind = pro.oracle_kind();
      collect_camera_diagonal_metrics =
        pro.collect_camera_diagonal_metrics();
      factorized_coupled_schur_metric =
        pro.factorized_coupled_schur_metric();
      proximal_defect_diagnostic = pro.proximal_defect_diagnostic();
        diagonal_trust_damping = pro.diagonal_trust_damping()
          || DiagonalTrustDampingEnabled();
        nesterov_relative_residual = pro.nesterov_relative_residual()
          || NesterovRelativeResidualEnabled();
      landmark_refinement_steps = pro.landmark_refinement_steps();
        nesterov_max_iterations = pro.nesterov_max_iterations();
        nesterov_min_iterations = pro.nesterov_min_iterations();
        nesterov_stop_tolerance = pro.nesterov_stop_tolerance();
        THROW_IF(nesterov_max_iterations <= 0 || nesterov_max_iterations > 1000);
        THROW_IF(nesterov_min_iterations <= 0
          || nesterov_min_iterations > nesterov_max_iterations);
        THROW_IF(!(nesterov_stop_tolerance > 0.)
          || !(nesterov_stop_tolerance < 1.));
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
        tr_radius = std::min(
            DabaInitialTrustRegionCap(), max_trust_region_radius);
      }
      if (pro.forced_trust_region_radius() > 0.) {
        tr_radius = std::max(
        MinimumTrustRegionRadius(),
        std::min(max_trust_region_radius, pro.forced_trust_region_radius()));
      }
      ceres_local_solver = pro.ceres_local_solver();
      objective_model = pro.objective_model();
      huber_delta = pro.huber_delta();
      THROW_IF(huber_delta < 0. || !std::isfinite(huber_delta));
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
        ceres::LossFunction* loss = huber_delta > 0.
          ? new ceres::HuberLoss(huber_delta) : nullptr;
        problem.AddResidualBlock(cost_function, loss,
                     &(cameras[9 * camera_id]),
                     &(landmarks[3 * landmark_id]));
      }
#endif
      WORKER_LOG("Added " << pro.observations_size() / 2 << " Residual blocks\n");
      function_residual_blocks.clear();
      problem.GetResidualBlocks(&function_residual_blocks);
      normal_equation_evaluate_options.apply_loss_function = huber_delta == 0.;
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

    double GetCost(bool revert_lm = false, bool apply_loss = true) {
#ifndef __unweighted_system__
      if (objective_model == 0 && (!apply_loss || huber_delta == 0.)
        && BatchedEvaluationEnabled()) {
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
      evalOptions.apply_loss_function = apply_loss;
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
  static Eigen::Matrix3d Skew(const Eigen::Vector3d& vector) {
    Eigen::Matrix3d result;
    result << 0., -vector.z(), vector.y(),
              vector.z(), 0., -vector.x(),
              -vector.y(), vector.x(), 0.;
    return result;
  }

  static Eigen::Matrix3d So3LeftJacobian(const Eigen::Vector3d& rotation) {
    const double squared_angle = rotation.squaredNorm();
    const Eigen::Matrix3d rotation_cross = Skew(rotation);
    if (squared_angle < 1e-12) {
      return Eigen::Matrix3d::Identity() + 0.5 * rotation_cross
          + (1. / 6.) * rotation_cross * rotation_cross;
    }
    const double angle = std::sqrt(squared_angle);
    return Eigen::Matrix3d::Identity()
        + ((1. - std::cos(angle)) / squared_angle) * rotation_cross
        + ((angle - std::sin(angle)) / (squared_angle * angle))
            * rotation_cross * rotation_cross;
  }

  static Eigen::Matrix3d RotationMatrix(const Eigen::Vector3d& angle_axis) {
    const double angle = angle_axis.norm();
    if (angle < 1e-14) {
      return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(angle, angle_axis / angle).toRotationMatrix();
  }

  static Eigen::Vector3d ContinuousAngleAxis(
      const Eigen::Matrix3d& rotation_matrix,
      const Eigen::Vector3d& reference) {
    const double two_pi = 2. * std::acos(-1.);
    const Eigen::AngleAxisd canonical(rotation_matrix);
    if (canonical.angle() < 1e-14) {
      const double reference_norm = reference.norm();
      if (reference_norm < 1e-14) {
        return Eigen::Vector3d::Zero();
      }
      const double winding = std::round(reference_norm / two_pi);
      return winding * two_pi * reference / reference_norm;
    }
    const Eigen::Vector3d base = canonical.angle() * canonical.axis();
    const double winding = std::round(
        (canonical.axis().dot(reference) - canonical.angle()) / two_pi);
    return base + winding * two_pi * canonical.axis();
  }

  static Eigen::Matrix<double, 9, 9> PhysicalTangentJacobian(
      const Eigen::Matrix<double, 9, 1>& physical_camera,
      CameraUpdateMode update_mode) {
    THROW_IF(update_mode == CameraUpdateMode::kAdditive);
    Eigen::Matrix<double, 9, 9> result =
        Eigen::Matrix<double, 9, 9>::Zero();
    if (update_mode == CameraUpdateMode::kAngleAxisLeft) {
      result.topLeftCorner<3, 3>() =
          So3LeftJacobian(physical_camera.head<3>()).inverse();
      result.block<3, 3>(3, 3).setIdentity();
    } else {
      Eigen::Vector3d rotation_for_jacobian = physical_camera.head<3>();
      if (update_mode == CameraUpdateMode::kSe3Right) {
        rotation_for_jacobian = -rotation_for_jacobian;
      }
      result.block<3, 3>(0, 3) =
          So3LeftJacobian(rotation_for_jacobian).inverse();
      if (update_mode == CameraUpdateMode::kSe3Right) {
        result.block<3, 3>(3, 0) =
            RotationMatrix(physical_camera.head<3>());
      } else if (update_mode == CameraUpdateMode::kSo3CenterLeft) {
        result.block<3, 3>(3, 0) =
            -RotationMatrix(physical_camera.head<3>());
        result.block<3, 3>(3, 3) =
            -Skew(physical_camera.segment<3>(3));
      } else {
        result.block<3, 3>(3, 0).setIdentity();
        if (update_mode == CameraUpdateMode::kSe3Left) {
          result.block<3, 3>(3, 3) =
              -Skew(physical_camera.segment<3>(3));
        }
      }
    }
    result.bottomRightCorner<3, 3>().setIdentity();
    return result;
  }

  std::vector<Eigen::Matrix<double, 9, 9>>
  TangentToScaledJacobians() const {
    const CameraUpdateMode update_mode = GetCameraUpdateMode();
    THROW_IF(update_mode == CameraUpdateMode::kAdditive);
    std::vector<Eigen::Matrix<double, 9, 9>> result(numCameras);
    for (int camera = 0; camera < numCameras; ++camera) {
      const int camera_offset = 9 * camera;
      const int transform_offset = 81 * camera;
      Eigen::Matrix<double, 9, 9> physical_transform;
      for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
          physical_transform(row, column) =
              cameraTransform[transform_offset + 9 * row + column]
              * unorm[camera_offset + column];
        }
      }
      const Eigen::Map<const Eigen::Matrix<double, 9, 1>> scaled_camera(
          &cameras[camera_offset]);
      const Eigen::Matrix<double, 9, 1> physical_camera =
          physical_transform * scaled_camera;
      const Eigen::Matrix<double, 9, 9> physical_jacobian =
          PhysicalTangentJacobian(physical_camera, update_mode);
      result[camera] =
          physical_transform.partialPivLu().solve(physical_jacobian);
      THROW_IF(!result[camera].allFinite());
    }
    return result;
  }

  void ApplySo3SubspaceMetricRatio(
      SparseMatrix<double, RowMajor>& metric) const {
    const double ratio = So3TranslationMetricRatio();
    if ((GetCameraUpdateMode() != CameraUpdateMode::kSo3Left &&
       GetCameraUpdateMode() != CameraUpdateMode::kSo3CenterLeft) ||
      ratio == 1.0) {
      return;
    }
    THROW_IF(metric.rows() != 9 * numCameras ||
             metric.cols() != 9 * numCameras);
    const auto tangent_to_scaled = TangentToScaledJacobians();
    Eigen::Matrix<double, 9, 9> subspace_scale =
        Eigen::Matrix<double, 9, 9>::Identity();
    subspace_scale.topLeftCorner<3, 3>() *= std::sqrt(ratio);
    for (int camera = 0; camera < numCameras; ++camera) {
      Eigen::Matrix<double, 9, 9> stored_metric;
      for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
          stored_metric(row, column) = metric.coeff(
              9 * camera + row, 9 * camera + column);
        }
      }
      const Eigen::Matrix<double, 9, 9> inverse_tangent =
          tangent_to_scaled[camera].inverse();
      const Eigen::Matrix<double, 9, 9> tangent_metric =
          tangent_to_scaled[camera].transpose()
          * stored_metric * tangent_to_scaled[camera];
      Eigen::Matrix<double, 9, 9> adjusted =
          inverse_tangent.transpose()
          * subspace_scale * tangent_metric * subspace_scale
          * inverse_tangent;
      adjusted = 0.5 * (adjusted + adjusted.transpose()).eval();
      THROW_IF(!adjusted.allFinite());
      for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
          metric.coeffRef(9 * camera + row, 9 * camera + column) =
              adjusted(row, column);
        }
      }
    }
  }

  static SparseMatrix<double, RowMajor> TransformCameraBlockMatrix(
      const SparseMatrix<double, RowMajor>& input,
      const std::vector<Eigen::Matrix<double, 9, 9>>& blocks) {
    SparseMatrix<double, RowMajor> result(input.rows(), input.cols());
    result.reserve(VectorXi::Constant(input.rows(), 9));
    for (int camera = 0; camera < blocks.size(); ++camera) {
      Eigen::Matrix<double, 9, 9> input_block;
      for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
          input_block(row, column) =
              input.coeff(9 * camera + row, 9 * camera + column);
        }
      }
      const Eigen::Matrix<double, 9, 9> output_block =
          blocks[camera].transpose() * input_block * blocks[camera];
      for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
          result.insert(9 * camera + row, 9 * camera + column) =
              output_block(row, column);
        }
      }
    }
    result.makeCompressed();
    return result;
  }
  static SparseMatrix<double, RowMajor> CameraBlocksToSparse(
      const std::vector<double>& blocks) {
    THROW_IF(blocks.size() % 81 != 0);
    const int block_count = blocks.size() / 81;
    SparseMatrix<double, RowMajor> result(
        9 * block_count, 9 * block_count);
    result.reserve(VectorXi::Constant(9 * block_count, 9));
    for (int block = 0; block < block_count; ++block) {
      for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
          result.insert(9 * block + row, 9 * block + column) =
              blocks[81 * block + 9 * row + column];
        }
      }
    }
    result.makeCompressed();
    return result;
  }

  static SparseMatrix<double, RowMajor> CameraPairBlocksToSparse(
      int camera_count,
      const std::map<std::pair<int, int>,
          Eigen::Matrix<double, 9, 9>>& blocks) {
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(81 * (2 * blocks.size() - camera_count));
    for (const auto& [camera_pair, block] : blocks) {
      const int row_camera = camera_pair.first;
      const int column_camera = camera_pair.second;
      THROW_IF(row_camera < 0 || row_camera >= camera_count ||
               column_camera < row_camera || column_camera >= camera_count);
      for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
          entries.emplace_back(
              9 * row_camera + row,
              9 * column_camera + column,
              block(row, column));
          if (row_camera != column_camera) {
            entries.emplace_back(
                9 * column_camera + column,
                9 * row_camera + row,
                block(row, column));
          }
        }
      }
    }
    SparseMatrix<double, RowMajor> result(
        9 * camera_count, 9 * camera_count);
    result.setFromTriplets(entries.begin(), entries.end());
    result.makeCompressed();
    return result;
  }

  void SetCameraProximalMetric(
      const SparseMatrix<double, RowMajor>& metric) {
    THROW_IF(metric.rows() != 9 * numCameras ||
             metric.cols() != 9 * numCameras);
    camera_proximal_metric = metric;
    camera_proximal_metric.makeCompressed();
    full_stepSize.assign(81 * numCameras, 0.);
    for (int camera = 0; camera < numCameras; ++camera) {
      for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
          full_stepSize[81 * camera + 9 * row + column] =
              camera_proximal_metric.coeff(
                  9 * camera + row, 9 * camera + column);
        }
      }
    }
  }

  void SetFactorizedCameraProximalMetric(
      const SparseMatrix<double, RowMajor>& diagonal,
      const BlockEdgeMatrix& cross,
      const SparseMatrix<double, RowMajor>& landmark_inverse) {
    THROW_IF(diagonal.rows() != 9 * numCameras ||
             diagonal.cols() != 9 * numCameras ||
             cross.rows() != 9 * numCameras ||
             cross.cols() != 3 * numLandmarks ||
             landmark_inverse.rows() != 3 * numLandmarks ||
             landmark_inverse.cols() != 3 * numLandmarks);
    factorized_proximal_diagonal = diagonal;
    factorized_proximal_cross = cross;
    factorized_proximal_landmark_inverse = landmark_inverse;
    for (int camera = 0; camera < numCameras; ++camera) {
      for (int row = 0; row < 9; ++row) {
        for (int column = row; column < 9; ++column) {
          const int global_row = 9 * camera + row;
          const int global_column = 9 * camera + column;
          const double value = 0.5 * (
              factorized_proximal_diagonal.coeff(global_row, global_column)
              + factorized_proximal_diagonal.coeff(
                  global_column, global_row));
          factorized_proximal_diagonal.coeffRef(
              global_row, global_column) = value;
          factorized_proximal_diagonal.coeffRef(
              global_column, global_row) = value;
        }
      }
    }
    factorized_proximal_active = true;
    SparseMatrix<double, RowMajor> block_diagonal =
        factorized_proximal_diagonal;
    factorized_proximal_cross.SubtractSchurDiagonal(
        factorized_proximal_landmark_inverse, block_diagonal);
    SetCameraProximalMetric(block_diagonal);
  }

  Eigen::VectorXd ApplyFactorizedCameraProximalMetric(
      const Eigen::VectorXd& vector) const {
    THROW_IF(!factorized_proximal_active ||
             vector.size() != factorized_proximal_diagonal.rows());
    Eigen::VectorXd factor_workspace(factorized_proximal_cross.cols());
    factorized_proximal_cross.TransposeMultiply(vector, factor_workspace);
    factor_workspace =
        factorized_proximal_landmark_inverse * factor_workspace;
    Eigen::VectorXd camera_workspace(factorized_proximal_cross.rows());
    factorized_proximal_cross.Multiply(
        factor_workspace, camera_workspace);
    return factorized_proximal_diagonal * vector - camera_workspace;
  }

  Eigen::VectorXd ApplyCameraProximalMetric(
      const Eigen::VectorXd& vector) const {
    if (factorized_proximal_active) {
      return ApplyFactorizedCameraProximalMetric(vector);
    }
    THROW_IF(camera_proximal_metric.rows() != vector.size());
    return camera_proximal_metric * vector;
  }

  Eigen::VectorXd ApplyCameraProximalMetricInTangent(
      const Eigen::VectorXd& vector,
      const std::vector<Eigen::Matrix<double, 9, 9>>& tangent_to_scaled) const {
    Eigen::VectorXd scaled(vector.size());
    for (int camera = 0; camera < numCameras; ++camera) {
      scaled.segment<9>(9 * camera) =
          tangent_to_scaled[camera] * vector.segment<9>(9 * camera);
    }
    return TransformCameraVector(
        ApplyCameraProximalMetric(scaled), tangent_to_scaled);
  }

  SparseMatrix<double, RowMajor> FactorizedProximalBlockDiagonalInTangent(
      const std::vector<Eigen::Matrix<double, 9, 9>>& tangent_to_scaled) const {
    THROW_IF(!factorized_proximal_active);
    return TransformCameraBlockMatrix(
        camera_proximal_metric, tangent_to_scaled);
  }
  static Eigen::VectorXd TransformCameraVector(
      const Eigen::VectorXd& input,
      const std::vector<Eigen::Matrix<double, 9, 9>>& blocks) {
    Eigen::VectorXd result(input.size());
    for (int camera = 0; camera < blocks.size(); ++camera) {
      result.segment<9>(9 * camera) =
          blocks[camera].transpose() * input.segment<9>(9 * camera);
    }
    return result;
  }

  void ApplyCameraStep(const Eigen::VectorXd& step) {
    THROW_IF(step.size() != cameras.size());
    const CameraUpdateMode update_mode = GetCameraUpdateMode();
    if (update_mode == CameraUpdateMode::kAdditive) {
      for (int index = 0; index < step.size(); ++index) {
        cameras[index] += step[index];
      }
      return;
    }

    for (int camera = 0; camera < numCameras; ++camera) {
      const int camera_offset = 9 * camera;
      const int transform_offset = 81 * camera;
      Eigen::Matrix<double, 9, 9> physical_transform;
      for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
          physical_transform(row, column) =
              cameraTransform[transform_offset + 9 * row + column]
              * unorm[camera_offset + column];
        }
      }

      const Eigen::Map<const Eigen::Matrix<double, 9, 1>> scaled_camera(
          &cameras[camera_offset]);
      const Eigen::Matrix<double, 9, 1> physical_camera =
          physical_transform * scaled_camera;
      const Eigen::Vector3d rotation = physical_camera.head<3>();
      const Eigen::Vector3d translation = physical_camera.segment<3>(3);
      const bool se3_left = update_mode == CameraUpdateMode::kSe3Left;
      const bool se3_right = update_mode == CameraUpdateMode::kSe3Right;
        const bool so3_left = update_mode == CameraUpdateMode::kSo3Left;
        const bool so3_center_left =
          update_mode == CameraUpdateMode::kSo3CenterLeft;
        const Eigen::Vector3d left_rotation =
          (se3_left || se3_right || so3_left || so3_center_left)
            ? step.segment<3>(camera_offset + 3)
            : step.segment<3>(camera_offset);
      const Eigen::Matrix3d rotation_increment = RotationMatrix(left_rotation);

      Eigen::Matrix<double, 9, 1> updated_physical = physical_camera;
        updated_physical.head<3>() = ContinuousAngleAxis(
          se3_right
              ? RotationMatrix(rotation) * rotation_increment
              : rotation_increment * RotationMatrix(rotation),
          rotation);
        if (se3_left) {
        const Eigen::Vector3d left_translation =
          step.segment<3>(camera_offset);
        updated_physical.segment<3>(3) = rotation_increment * translation
          + So3LeftJacobian(left_rotation) * left_translation;
        } else if (se3_right) {
        const Eigen::Vector3d right_translation =
          step.segment<3>(camera_offset);
        updated_physical.segment<3>(3) = translation
          + RotationMatrix(rotation) * So3LeftJacobian(left_rotation)
              * right_translation;
        } else if (so3_left) {
        updated_physical.segment<3>(3) +=
          step.segment<3>(camera_offset);
        } else if (so3_center_left) {
        const Eigen::Vector3d camera_center =
          -RotationMatrix(rotation).transpose() * translation;
        const Eigen::Vector3d updated_center =
          camera_center + step.segment<3>(camera_offset);
        updated_physical.segment<3>(3) =
          -rotation_increment * RotationMatrix(rotation) * updated_center;
        } else {
        updated_physical.segment<3>(3) +=
          step.segment<3>(camera_offset + 3);
        }
        updated_physical.tail<3>() += step.segment<3>(camera_offset + 6);

      const Eigen::Matrix<double, 9, 1> updated_scaled =
          physical_transform.partialPivLu().solve(updated_physical);
        if (!updated_scaled.allFinite()) {
        const Eigen::Matrix<double, 9, 1> camera_step =
          step.segment<9>(camera_offset);
        const Eigen::Matrix<double, 9, 1> singular_values =
          physical_transform.jacobiSvd().singularValues();
        std::cerr << "Non-finite camera back-transform: camera=" << camera
              << " radius=" << tr_radius
              << " step_norm=" << camera_step.norm()
              << " step_max=" << camera_step.cwiseAbs().maxCoeff()
              << " physical_norm=" << physical_camera.norm()
              << " updated_physical_norm=" << updated_physical.norm()
              << " transform_sigma_min=" << singular_values.minCoeff()
              << " transform_sigma_max=" << singular_values.maxCoeff()
              << " scaled_max=" << updated_scaled.cwiseAbs().maxCoeff()
              << std::endl;
        }
      THROW_IF(!updated_scaled.allFinite());
      Eigen::Map<Eigen::Matrix<double, 9, 1>> scaled_camera_output(
          &cameras[camera_offset]);
      scaled_camera_output = updated_scaled;
    }
  }

  class ScaledLeftSe3Manifold final : public ceres::Manifold {
   public:
    explicit ScaledLeftSe3Manifold(
        const Eigen::Matrix<double, 9, 9>& physical_transform)
        : physical_transform_(physical_transform),
          scaled_transform_(physical_transform.inverse()) {}

    int AmbientSize() const override { return 9; }
    int TangentSize() const override { return 9; }

    bool Plus(const double* x, const double* delta,
              double* x_plus_delta) const override {
      const Eigen::Map<const Eigen::Matrix<double, 9, 1>> scaled(x);
      const Eigen::Matrix<double, 9, 1> physical = physical_transform_ * scaled;
      const Eigen::Map<const Eigen::Vector3d> translation_delta(delta);
      const Eigen::Map<const Eigen::Vector3d> rotation_delta(delta + 3);
      const Eigen::Matrix3d rotation_increment = RotationMatrix(rotation_delta);
      Eigen::Matrix<double, 9, 1> updated = physical;
      updated.head<3>() = ContinuousAngleAxis(
          rotation_increment * RotationMatrix(physical.head<3>()),
          physical.head<3>());
      updated.segment<3>(3) =
          rotation_increment * physical.segment<3>(3)
          + So3LeftJacobian(rotation_delta) * translation_delta;
      updated.tail<3>() += Eigen::Map<const Eigen::Vector3d>(delta + 6);
        Eigen::Map<Eigen::Matrix<double, 9, 1>> output(x_plus_delta);
        output = scaled_transform_ * updated;
      return true;
    }

    bool PlusJacobian(const double* x, double* jacobian) const override {
      const Eigen::Map<const Eigen::Matrix<double, 9, 1>> scaled(x);
      const Eigen::Matrix<double, 9, 1> physical = physical_transform_ * scaled;
      Eigen::Matrix<double, 9, 9> physical_jacobian =
          Eigen::Matrix<double, 9, 9>::Zero();
      physical_jacobian.block<3, 3>(0, 3) =
          So3LeftJacobian(physical.head<3>()).inverse();
      physical_jacobian.block<3, 3>(3, 0).setIdentity();
      physical_jacobian.block<3, 3>(3, 3) =
          -Skew(physical.segment<3>(3));
      physical_jacobian.bottomRightCorner<3, 3>().setIdentity();
      Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> result(jacobian);
      result = scaled_transform_ * physical_jacobian;
      return result.allFinite();
    }

    bool Minus(const double* y, const double* x,
               double* y_minus_x) const override {
      const Eigen::Map<const Eigen::Matrix<double, 9, 1>> scaled_y(y);
      const Eigen::Map<const Eigen::Matrix<double, 9, 1>> scaled_x(x);
      const Eigen::Matrix<double, 9, 1> physical_y = physical_transform_ * scaled_y;
      const Eigen::Matrix<double, 9, 1> physical_x = physical_transform_ * scaled_x;
      const Eigen::Matrix3d relative_rotation =
          RotationMatrix(physical_y.head<3>())
          * RotationMatrix(physical_x.head<3>()).transpose();
      const Eigen::Vector3d rotation = ContinuousAngleAxis(
          relative_rotation, Eigen::Vector3d::Zero());
      const Eigen::Vector3d relative_translation =
          physical_y.segment<3>(3)
          - relative_rotation * physical_x.segment<3>(3);
        Eigen::Map<Eigen::Vector3d> translation_output(y_minus_x);
        Eigen::Map<Eigen::Vector3d> rotation_output(y_minus_x + 3);
        Eigen::Map<Eigen::Vector3d> intrinsics_output(y_minus_x + 6);
        translation_output =
          So3LeftJacobian(rotation).inverse() * relative_translation;
        rotation_output = rotation;
        intrinsics_output = physical_y.tail<3>() - physical_x.tail<3>();
      return true;
    }

    bool MinusJacobian(const double* x, double* jacobian) const override {
      double plus_jacobian[81];
      PlusJacobian(x, plus_jacobian);
      const Eigen::Map<const Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> plus(
          plus_jacobian);
      Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> result(jacobian);
      result = plus.inverse();
      return result.allFinite();
    }

   private:
    Eigen::Matrix<double, 9, 9> physical_transform_;
    Eigen::Matrix<double, 9, 9> scaled_transform_;
  };

  class CameraBlockProximalCost final
      : public ceres::SizedCostFunction<9, 9> {
   public:
    CameraBlockProximalCost(
        const Eigen::Matrix<double, 9, 9>& metric,
        const double* center) {
      const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> solver(
          0.5 * (metric + metric.transpose()));
      THROW_IF(solver.info() != Eigen::Success);
      square_root_ = solver.eigenvectors()
          * solver.eigenvalues().cwiseMax(0.).cwiseSqrt().asDiagonal()
          * solver.eigenvectors().transpose();
      center_ = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(center);
    }

    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override {
      const Eigen::Map<const Eigen::Matrix<double, 9, 1>> camera(parameters[0]);
        Eigen::Map<Eigen::Matrix<double, 9, 1>> residual_map(residuals);
        residual_map = square_root_ * (camera - center_);
      if (jacobians != nullptr && jacobians[0] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> jacobian(
            jacobians[0]);
        jacobian = square_root_;
      }
      return true;
    }

   private:
    Eigen::Matrix<double, 9, 9> square_root_;
    Eigen::Matrix<double, 9, 1> center_;
  };

  void ScaleCameraProximalMetric(
      SparseMatrix<double, RowMajor>& metric) const {
    THROW_IF(camera_proximal_multipliers.size()
        != static_cast<size_t>(numCameras));
    for (int row = 0; row < metric.outerSize(); ++row) {
      const double multiplier = camera_proximal_multipliers[row / 9];
      THROW_IF(multiplier < 0. || !std::isfinite(multiplier));
      for (SparseMatrix<double, RowMajor>::InnerIterator entry(metric, row);
           entry; ++entry) {
        entry.valueRef() *= multiplier;
      }
    }
  }

  bool PrepareCentralizedProximalMetric() {
    NormalEquations normal_equations = GetNormalEquations();
    SparseMatrix<double, RowMajor> camera_hessian =
        normal_equations.camera_hessian;
    if (firstIteration) {
      if (!DisableLandmarkPreconditioningEnabled()) {
        const Eigen::VectorXd diagonal =
            normal_equations.landmark_hessian.diagonal().array()
                .cwiseMax(LandmarkPreconditionerFloor())
                .cwiseSqrt().cwiseInverse();
        for (int index = 0; index < diagonal.size(); ++index) {
          landmarks[index] /= diagonal[index];
          vnorm[index] = diagonal[index];
        }
      }
      UpdatePreconditioningCameras(camera_hessian);
      best_landmarks = landmarks;
      cost = 2. * GetCost();
      firstIteration = false;
      return false;
    }

    const Eigen::DiagonalMatrix<double, Eigen::Dynamic> consensus_diagonal =
        CameraDiagonalMetricScale() * Diagonal<9>(
          camera_hessian, cluster_id, collect_camera_diagonal_metrics,
          outer_iteration, oracle_kind, "centralized_consensus",
          &global_camera_ids);
    const double legacy_scale =
        std::min(1.005, 1e-1 * std::sqrt(current_be / start_be));
    const double scale = block_curvature_multiplier > 0.
        ? block_curvature_multiplier : legacy_scale;
    SparseMatrix<double, RowMajor> metric = scale * camera_hessian;
    metric += consensus_diagonal * current_be;
    ScaleCameraProximalMetric(metric);
    const double* values = metric.valuePtr();
    full_stepSize.assign(values, values + 81 * numCameras);
    return true;
  }

  double SolveCentralizedLeftSe3(bool include_proximal = false) {
    THROW_IF((!include_proximal && num_clusters != 1) ||
             GetCameraUpdateMode() != CameraUpdateMode::kSe3Left);
    if (include_proximal && !PrepareCentralizedProximalMetric()) {
      return cost;
    }
    ceres::Problem centralized_problem;
    const std::vector<double> diagnostic_initial_cameras = cameras;
    const std::vector<double> diagnostic_initial_landmarks = landmarks;
    for (int observation = 0; observation < numResiduals; ++observation) {
      const int camera_id = cam_obs[observation];
      const int landmark_id = lm_obs[observation];
      centralized_problem.AddResidualBlock(
          SnavelyReprojectionErrorWeighted::Create(
              observed_x[observation], observed_y[observation],
              &unorm[9 * camera_id], &vnorm[3 * landmark_id],
              &cameraTransform[81 * camera_id]),
          nullptr, &cameras[9 * camera_id], &landmarks[3 * landmark_id]);
    }
    for (int camera = 0; camera < numCameras; ++camera) {
      Eigen::Matrix<double, 9, 9> physical_transform;
      for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
          physical_transform(row, column) =
              cameraTransform[81 * camera + 9 * row + column]
              * unorm[9 * camera + column];
        }
      }
      centralized_problem.SetManifold(
          &cameras[9 * camera], new ScaledLeftSe3Manifold(physical_transform));
      if (include_proximal) {
        Eigen::Matrix<double, 9, 9> metric;
        for (int row = 0; row < 9; ++row) {
          for (int column = 0; column < 9; ++column) {
            metric(row, column) =
                full_stepSize[81 * camera + 9 * row + column];
          }
        }
        centralized_problem.AddResidualBlock(
            new CameraBlockProximalCost(metric, &cameras_s[9 * camera]),
            nullptr, &cameras[9 * camera]);
      }
    }
    ceres::Solver::Options centralized_options;
    centralized_options.max_num_iterations = CentralizedCeresIterations();
    centralized_options.num_threads = options.num_threads;
    centralized_options.linear_solver_type = CentralizedCeresSparseSchurEnabled()
        ? ceres::SPARSE_SCHUR : ceres::ITERATIVE_SCHUR;
    if (!CentralizedCeresSparseSchurEnabled()) {
      centralized_options.preconditioner_type = ceres::SCHUR_JACOBI;
    }
    centralized_options.initial_trust_region_radius =
      CentralizedCeresInitialRadius();
    centralized_options.max_linear_solver_iterations =
      CentralizedCeresMaximumLinearIterations();
    centralized_options.function_tolerance = 1e-12;
    centralized_options.gradient_tolerance = 1e-12;
    centralized_options.parameter_tolerance = 1e-12;
    centralized_options.eta = 0.1;
    centralized_options.jacobi_scaling =
      CentralizedCeresJacobiScalingEnabled();
    centralized_options.minimizer_progress_to_stdout = false;
    if (const char* dump_directory = std::getenv(
        "BUNDLE_PALM_CENTRALIZED_CERES_DUMP_DIRECTORY")) {
      centralized_options.trust_region_minimizer_iterations_to_dump = {0, 1};
      centralized_options.trust_region_problem_dump_directory = dump_directory;
      centralized_options.trust_region_problem_dump_format_type =
        ceres::TEXTFILE;
    }
    if (CentralizedCeresTangentDiagnosticEnabled()) {
      ceres::Problem::EvaluateOptions evaluate_options;
      for (int camera = 0; camera < numCameras; ++camera) {
        evaluate_options.parameter_blocks.push_back(&cameras[9 * camera]);
      }
      for (int landmark = 0; landmark < numLandmarks; ++landmark) {
        evaluate_options.parameter_blocks.push_back(&landmarks[3 * landmark]);
      }
      double diagnostic_cost = 0.;
      std::vector<double> gradient;
      ceres::CRSMatrix jacobian;
      centralized_problem.Evaluate(
          evaluate_options, &diagnostic_cost, nullptr, &gradient, &jacobian);
      std::array<double, 4> gradient_squared = {0., 0., 0., 0.};
      std::vector<double> column_squared(jacobian.num_cols, 0.);
      for (int row = 0; row < jacobian.num_rows; ++row) {
        for (int index = jacobian.rows[row];
             index < jacobian.rows[row + 1]; ++index) {
          column_squared[jacobian.cols[index]] +=
              jacobian.values[index] * jacobian.values[index];
        }
      }
      std::array<double, 4> normalized_gradient_squared = {0., 0., 0., 0.};
      std::array<double, 4> minimum_column_squared = {
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::infinity()};
      for (int camera = 0; camera < numCameras; ++camera) {
        for (int parameter = 0; parameter < 9; ++parameter) {
          const double value = gradient[9 * camera + parameter];
            const int group = parameter / 3;
            gradient_squared[group] += value * value;
            minimum_column_squared[group] = std::min(
              minimum_column_squared[group],
              column_squared[9 * camera + parameter]);
            normalized_gradient_squared[group] += value * value /
              std::max(column_squared[9 * camera + parameter],
                std::numeric_limits<double>::min());
        }
      }
      const int landmark_offset = 9 * numCameras;
      for (int parameter = landmark_offset;
           parameter < gradient.size(); ++parameter) {
        gradient_squared[3] += gradient[parameter] * gradient[parameter];
        minimum_column_squared[3] = std::min(
          minimum_column_squared[3], column_squared[parameter]);
        normalized_gradient_squared[3] += gradient[parameter] * gradient[parameter] /
          std::max(column_squared[parameter],
            std::numeric_limits<double>::min());
      }
      double camera_quadratic = 0.;
      double landmark_quadratic = 0.;
      double cross_quadratic = 0.;
      double gradient_probe = 0.;
      for (int column = 0; column < gradient.size(); ++column) {
        gradient_probe += gradient[column]
        * std::sin(0.6180339887498949 * (column + 1));
      }
      for (int row = 0; row < jacobian.num_rows; ++row) {
        double camera_action = 0.;
        double landmark_action = 0.;
        for (int index = jacobian.rows[row];
             index < jacobian.rows[row + 1]; ++index) {
          const int column = jacobian.cols[index];
          const double probe = std::sin(0.6180339887498949 * (column + 1));
          if (column < landmark_offset) {
            camera_action += jacobian.values[index] * probe;
          } else {
            landmark_action += jacobian.values[index] * probe;
          }
        }
        camera_quadratic += camera_action * camera_action;
        landmark_quadratic += landmark_action * landmark_action;
        cross_quadratic += 2. * camera_action * landmark_action;
      }
      std::ostringstream metric;
      metric << "CERES_TANGENT_GRADIENT cluster=" << cluster_id
             << " cost=" << 2. * diagnostic_cost
             << " translation=" << std::sqrt(gradient_squared[0])
             << " rotation=" << std::sqrt(gradient_squared[1])
             << " intrinsics=" << std::sqrt(gradient_squared[2])
             << " landmarks=" << std::sqrt(gradient_squared[3])
             << " normalized_translation="
             << std::sqrt(normalized_gradient_squared[0])
             << " normalized_rotation="
             << std::sqrt(normalized_gradient_squared[1])
             << " normalized_intrinsics="
             << std::sqrt(normalized_gradient_squared[2])
             << " normalized_landmarks="
             << std::sqrt(normalized_gradient_squared[3])
             << " minimum_translation_diagonal=" << minimum_column_squared[0]
             << " minimum_rotation_diagonal=" << minimum_column_squared[1]
             << " minimum_intrinsics_diagonal=" << minimum_column_squared[2]
             << " minimum_landmark_diagonal=" << minimum_column_squared[3]
             << " camera_quadratic=" << camera_quadratic
             << " landmark_quadratic=" << landmark_quadratic
             << " cross_quadratic=" << cross_quadratic
             << " gradient_probe=" << gradient_probe
             << " columns=" << jacobian.num_cols << "\n";
      EmitLocalSolveMetric(metric.str());
    }
    NormalEquations diagnostic_normal_equations;
    BlockEdgeMatrix diagnostic_cross_hessian;
    if (CentralizedCeresTangentDiagnosticEnabled()) {
      diagnostic_normal_equations = GetBatchedNormalEquations(true);
      diagnostic_cross_hessian = camera_landmark_hessian;
    }
    ceres::Solver::Summary summary;
    ceres::Solve(centralized_options, &centralized_problem, &summary);
    if (include_proximal && LocalSolveMetricsEnabled()) {
      Eigen::Map<const Eigen::VectorXd> initial_camera_vector(
        diagnostic_initial_cameras.data(), diagnostic_initial_cameras.size());
      Eigen::Map<const Eigen::VectorXd> final_camera_vector(
        cameras.data(), cameras.size());
      Eigen::Map<const Eigen::VectorXd> initial_landmark_vector(
        diagnostic_initial_landmarks.data(), diagnostic_initial_landmarks.size());
      Eigen::Map<const Eigen::VectorXd> final_landmark_vector(
        landmarks.data(), landmarks.size());
      Eigen::VectorXd proximal_offset(cameras.size());
      for (int index = 0; index < proximal_offset.size(); ++index) {
      proximal_offset[index] = cameras[index] - cameras_s[index];
      }
      const double proximal_cost = proximal_offset.dot(
        blockMult<9>(full_stepSize, proximal_offset));
      std::ostringstream metric;
      metric << "CENTRALIZED_CERES_PROXIMAL_ENDPOINT cluster=" << cluster_id
         << " camera_displacement="
         << (final_camera_vector - initial_camera_vector).norm()
         << " landmark_displacement="
         << (final_landmark_vector - initial_landmark_vector).norm()
         << " reprojection=" << 2. * GetCost()
         << " proximal=" << proximal_cost
         << " total=" << 2. * GetCost() + proximal_cost << "\n";
      EmitLocalSolveMetric(metric.str());
    }
    if (CentralizedCeresTangentDiagnosticEnabled()) {
      const NormalEquations& normal_equations = diagnostic_normal_equations;
      const BlockEdgeMatrix& cross_hessian = diagnostic_cross_hessian;
      Eigen::VectorXd camera_step(9 * numCameras);
      for (int camera = 0; camera < numCameras; ++camera) {
        Eigen::Matrix<double, 9, 9> physical_transform;
        for (int row = 0; row < 9; ++row) {
          for (int column = 0; column < 9; ++column) {
            physical_transform(row, column) =
                cameraTransform[81 * camera + 9 * row + column]
                * unorm[9 * camera + column];
          }
        }
        ScaledLeftSe3Manifold manifold(physical_transform);
        manifold.Minus(&cameras[9 * camera],
                       &diagnostic_initial_cameras[9 * camera],
                       &camera_step[9 * camera]);
      }
      const Eigen::Map<const Eigen::VectorXd> initial_landmarks(
          diagnostic_initial_landmarks.data(), diagnostic_initial_landmarks.size());
      const Eigen::Map<const Eigen::VectorXd> final_landmarks(
          landmarks.data(), landmarks.size());
      const Eigen::VectorXd landmark_step = final_landmarks - initial_landmarks;
      double solve_radius = centralized_options.initial_trust_region_radius;
      for (int iteration = 1; iteration < summary.iterations.size(); ++iteration) {
        if (summary.iterations[iteration].step_is_successful) {
          solve_radius = summary.iterations[iteration - 1].trust_region_radius;
          break;
        }
      }
      Eigen::VectorXd camera_cross(camera_step.size());
      Eigen::VectorXd landmark_cross(landmark_step.size());
      cross_hessian.Multiply(landmark_step, camera_cross);
      cross_hessian.TransposeMultiply(camera_step, landmark_cross);
      Eigen::VectorXd camera_residual =
          normal_equations.camera_hessian * camera_step + camera_cross
          + normal_equations.camera_gradient;
      Eigen::VectorXd landmark_residual =
          normal_equations.landmark_hessian * landmark_step + landmark_cross
          + normal_equations.landmark_gradient;
      camera_residual += (1. / solve_radius)
          * normal_equations.camera_hessian.diagonal().cwiseMax(1e-6)
              .cwiseMin(1e32).cwiseProduct(camera_step);
      landmark_residual += (1. / solve_radius)
          * normal_equations.landmark_hessian.diagonal().cwiseMax(1e-6)
              .cwiseMin(1e32).cwiseProduct(landmark_step);
      const double gradient_norm = std::hypot(
          normal_equations.camera_gradient.norm(),
          normal_equations.landmark_gradient.norm());
      std::ostringstream metric;
        double translation_step_squared = 0.;
        double rotation_step_squared = 0.;
        double intrinsics_step_squared = 0.;
        for (int camera = 0; camera < numCameras; ++camera) {
        translation_step_squared +=
          camera_step.segment<3>(9 * camera).squaredNorm();
        rotation_step_squared +=
          camera_step.segment<3>(9 * camera + 3).squaredNorm();
        intrinsics_step_squared +=
          camera_step.segment<3>(9 * camera + 6).squaredNorm();
        }
      metric << "CERES_STEP_NORMAL_RESIDUAL cluster=" << cluster_id
             << " radius=" << solve_radius
             << " camera_step=" << camera_step.norm()
           << " translation_step=" << std::sqrt(translation_step_squared)
           << " rotation_step=" << std::sqrt(rotation_step_squared)
           << " intrinsics_step=" << std::sqrt(intrinsics_step_squared)
             << " landmark_step=" << landmark_step.norm()
             << " camera_residual=" << camera_residual.norm()
             << " landmark_residual=" << landmark_residual.norm()
             << " relative=" << std::hypot(
                  camera_residual.norm(), landmark_residual.norm())
                  / gradient_norm << "\n";
      EmitLocalSolveMetric(metric.str());
    }
    if (LocalSolveMetricsEnabled()) {
      EmitLocalSolveMetric(
          "CENTRALIZED_CERES_SCHUR_STRUCTURE cluster="
          + std::to_string(cluster_id)
          + " used=" + summary.schur_structure_used + "\n");
      for (const ceres::IterationSummary& iteration : summary.iterations) {
        std::ostringstream metric;
        metric << "CENTRALIZED_CERES_ITERATION cluster=" << cluster_id
               << " iteration=" << iteration.iteration
               << " cost=" << 2. * iteration.cost
               << " cost_change=" << 2. * iteration.cost_change
               << " relative_decrease=" << iteration.relative_decrease
               << " radius=" << iteration.trust_region_radius
               << " step_norm=" << iteration.step_norm
               << " step_successful=" << iteration.step_is_successful
               << " step_valid=" << iteration.step_is_valid
               << " linear_iterations=" << iteration.linear_solver_iterations
               << "\n";
        EmitLocalSolveMetric(metric.str());
      }
    }
    cost = 2. * GetCost();
    if (!summary.iterations.empty()) {
      tr_radius = std::min(
          max_trust_region_radius,
          summary.iterations.back().trust_region_radius);
    }
    return cost;
  }

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

    void SetExternalState(const std::string& camera_bytes,
                          const std::string& landmark_bytes) {
      THROW_IF(camera_bytes.size() != cameras.size() * sizeof(double));
      THROW_IF(landmark_bytes.size() != landmarks.size() * sizeof(double));
      std::memcpy(cameras.data(), camera_bytes.data(), camera_bytes.size());
      const double* physical_landmarks =
          reinterpret_cast<const double*>(landmark_bytes.data());
      for (int index = 0; index < landmarks.size(); ++index) {
        landmarks[index] = physical_landmarks[index] / vnorm[index];
      }
    }

    return_schur_system_proto BuildSchurSystem(
        const schur_system_proto& request) {
      THROW_IF(!ManifoldCameraUpdatesEnabled() ||
               !DirectTangentNormalEquationsEnabled());
      THROW_IF(request.cluster_id() != cluster_id ||
               request.landmark_damping() < 0.);
      const std::vector<double> saved_cameras = cameras;
      const std::vector<double> saved_landmarks = landmarks;
      const double saved_cost = cost;
      SetExternalState(request.cameras_f64(), request.landmarks_f64());
      const NormalEquations normal_equations = GetBatchedNormalEquations(true);
      SparseMatrix<double, RowMajor> landmark_inverse =
          normal_equations.landmark_hessian;
      if (request.landmark_damping() > 0.) {
        SparseMatrix<double, RowMajor> landmark_diagonal_source =
            normal_equations.landmark_hessian;
        landmark_inverse += request.landmark_damping()
            * Diagonal<3>(landmark_diagonal_source);
      }
        FloorSymmetricBlocks<3>(
          landmark_inverse, std::max(PobaBlockRelativeFloor(), 1e-12));
      BlockInverse<3>(landmark_inverse);

      Eigen::VectorXd landmark_workspace =
          landmark_inverse * normal_equations.landmark_gradient;
      const double landmark_model_reduction = 0.5
          * normal_equations.landmark_gradient.dot(landmark_workspace);
      Eigen::VectorXd camera_workspace(camera_landmark_hessian.rows());
      camera_landmark_hessian.Multiply(landmark_workspace, camera_workspace);
      const Eigen::VectorXd reduced_gradient =
          normal_equations.camera_gradient - camera_workspace;
      const auto schur_blocks =
          camera_landmark_hessian.MaterializeSchurComplement(
              normal_equations.camera_hessian, landmark_inverse);

      std::vector<std::uint32_t> block_rows;
      std::vector<std::uint32_t> block_columns;
      std::vector<double> block_values;
      block_rows.reserve(schur_blocks.size());
      block_columns.reserve(schur_blocks.size());
      block_values.reserve(81 * schur_blocks.size());
      for (const auto& entry : schur_blocks) {
        std::uint32_t row_camera = global_camera_ids[entry.first.first];
        std::uint32_t column_camera = global_camera_ids[entry.first.second];
        Eigen::Matrix<double, 9, 9> block = entry.second;
        if (row_camera > column_camera) {
          std::swap(row_camera, column_camera);
          block.transposeInPlace();
        }
        block_rows.push_back(row_camera);
        block_columns.push_back(column_camera);
        for (int row = 0; row < 9; ++row) {
          for (int column = 0; column < 9; ++column) {
            block_values.push_back(block(row, column));
          }
        }
      }

      std::vector<double> camera_diagonal;
      camera_diagonal.reserve(81 * numCameras);
      for (int camera = 0; camera < numCameras; ++camera) {
        for (int row = 0; row < 9; ++row) {
          for (int column = 0; column < 9; ++column) {
            camera_diagonal.push_back(normal_equations.camera_hessian.coeff(
                9 * camera + row, 9 * camera + column));
          }
        }
      }

      return_schur_system_proto response;
      response.set_cluster_id(cluster_id);
      response.set_run_id(request.run_id());
      response.set_phase_id(request.phase_id());
      response.set_camera_ids_u32(
          reinterpret_cast<const char*>(global_camera_ids.data()),
          global_camera_ids.size() * sizeof(std::uint32_t));
      response.set_block_rows_u32(
          reinterpret_cast<const char*>(block_rows.data()),
          block_rows.size() * sizeof(std::uint32_t));
      response.set_block_columns_u32(
          reinterpret_cast<const char*>(block_columns.data()),
          block_columns.size() * sizeof(std::uint32_t));
      response.set_blocks_f64(
          reinterpret_cast<const char*>(block_values.data()),
          block_values.size() * sizeof(double));
      response.set_reduced_gradient_f64(
          reinterpret_cast<const char*>(reduced_gradient.data()),
          reduced_gradient.size() * sizeof(double));
      response.set_camera_diagonal_f64(
          reinterpret_cast<const char*>(camera_diagonal.data()),
          camera_diagonal.size() * sizeof(double));
      response.set_landmark_model_reduction(landmark_model_reduction);
      cameras = saved_cameras;
      landmarks = saved_landmarks;
      cost = saved_cost;
      return response;
    }

    return_cluster_proto ApplyExternalCameraStep(
        const camera_step_proto& request) {
      THROW_IF(request.cluster_id() != cluster_id ||
               request.landmark_refinement_steps() < 0 ||
               request.landmark_refinement_steps() > 20);
      THROW_IF(request.transport_product_state() &&
               (request.rebase_trust_state() ||
                request.landmark_refinement_steps() != 0 ||
                request.centers_f64().size()
                    != cameras_s.size() * sizeof(double)));
      SetExternalState(request.cameras_f64(), request.landmarks_f64());
        Eigen::VectorXd product_offsets;
        if (request.transport_product_state()) {
        const Eigen::Map<const Eigen::VectorXd> requested_centers(
          reinterpret_cast<const double*>(request.centers_f64().data()),
          cameras_s.size());
        product_offsets = Eigen::Map<const Eigen::VectorXd>(
          cameras.data(), cameras.size()) - requested_centers;
        }
      THROW_IF(request.tangent_step_f64().size()
               != cameras.size() * sizeof(double));
      const Eigen::Map<const Eigen::VectorXd> tangent_step(
          reinterpret_cast<const double*>(request.tangent_step_f64().data()),
          cameras.size());
      ApplyCameraStep(tangent_step);
      if (request.transport_product_state()) {
        Eigen::Map<Eigen::VectorXd>(cameras_s.data(), cameras_s.size()) =
            Eigen::Map<const Eigen::VectorXd>(cameras.data(), cameras.size())
            - product_offsets;
      }
      RefineLandmarksWithFixedCameras(request.landmark_refinement_steps());
      cost = 2. * GetCost();
      if (request.transport_product_state()) {
        last_cameras = cameras;
        last_landmarks = landmarks;
        accepted_landmarks = landmarks;
        last_tr_radius = tr_radius;
      } else if (request.rebase_trust_state()) {
        tr_radius = std::min(max_trust_region_radius, init_trust_region_radius);
        if (trust_region_policy == 1 && persistent_trust_region) {
          tr_radius = std::min(
              DabaInitialTrustRegionCap(), max_trust_region_radius);
        }
        last_cameras = cameras;
        last_landmarks = landmarks;
        last_tr_radius = tr_radius;
        persistent_trust_region_active = persistent_trust_region;
        last_linear_iterations = 0;
        last_linear_relative_residual =
            std::numeric_limits<double>::quiet_NaN();
        options.initial_trust_region_radius = tr_radius;
      }
      return FillReturnProto(true, false, false);
    }

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
        std::vector<double> metric_upper_blocks;
        metric_upper_blocks.reserve(45 * numCameras);
        for (int camera = 0; camera < numCameras; ++camera) {
          const int offset = 81 * camera;
          for (int row = 0; row < 9; ++row) {
            for (int column = row; column < 9; ++column) {
              const double upper = full_stepSize[offset + 9 * row + column];
              const double lower = full_stepSize[offset + 9 * column + row];
              THROW_IF(!std::isfinite(upper) || !std::isfinite(lower));
              metric_upper_blocks.push_back(0.5 * (upper + lower));
            }
          }
        }
        return_proto.set_step_size_upper_f64(
            reinterpret_cast<const char*>(metric_upper_blocks.data()),
            metric_upper_blocks.size() * sizeof(double));
        if (factorized_proximal_active) {
          return_proto.set_has_factorized_metric(true);
          std::vector<std::uint32_t> camera_ids;
          std::vector<double> diagonal_values;
          for (int camera = 0; camera < numCameras; ++camera) {
            if (!(camera_proximal_multipliers[camera] > 0.)) {
              continue;
            }
            camera_ids.push_back(global_camera_ids[camera]);
            for (int row = 0; row < 9; ++row) {
              for (int column = 0; column < 9; ++column) {
                diagonal_values.push_back(
                    factorized_proximal_diagonal.coeff(
                        9 * camera + row, 9 * camera + column));
              }
            }
          }
          std::vector<int> factor_map(
              factorized_proximal_cross.LandmarkCount(), -1);
          const auto& factor_edges = factorized_proximal_cross.Edges();
          for (const CameraLandmarkEdge& edge : factor_edges) {
            const Eigen::Map<
                const Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> block(
                    edge.values.data());
            if (block.squaredNorm() > 0. && factor_map[edge.landmark] < 0) {
              factor_map[edge.landmark] = 0;
            }
          }
          std::vector<double> inverse_values;
          int factor_count = 0;
          for (int landmark = 0; landmark < factor_map.size(); ++landmark) {
            if (factor_map[landmark] < 0) {
              continue;
            }
            factor_map[landmark] = factor_count++;
            for (int row = 0; row < 3; ++row) {
              for (int column = 0; column < 3; ++column) {
                inverse_values.push_back(
                    factorized_proximal_landmark_inverse.coeff(
                        3 * landmark + row, 3 * landmark + column));
              }
            }
          }
          std::vector<std::uint32_t> edge_factors;
          std::vector<std::uint32_t> edge_cameras;
          std::vector<double> edge_values;
          for (const CameraLandmarkEdge& edge : factor_edges) {
            if (factor_map[edge.landmark] < 0) {
              continue;
            }
            const Eigen::Map<
                const Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> block(
                    edge.values.data());
            if (block.squaredNorm() == 0.) {
              continue;
            }
            edge_factors.push_back(factor_map[edge.landmark]);
            edge_cameras.push_back(global_camera_ids[edge.camera]);
            edge_values.insert(
                edge_values.end(), edge.values.begin(), edge.values.end());
          }
          return_proto.set_factorized_metric_camera_ids_u32(
              reinterpret_cast<const char*>(camera_ids.data()),
              camera_ids.size() * sizeof(std::uint32_t));
          return_proto.set_factorized_metric_diagonal_blocks_f64(
              reinterpret_cast<const char*>(diagonal_values.data()),
              diagonal_values.size() * sizeof(double));
          return_proto.set_factorized_metric_inverse_blocks_f64(
              reinterpret_cast<const char*>(inverse_values.data()),
              inverse_values.size() * sizeof(double));
          return_proto.set_factorized_metric_edge_factors_u32(
              reinterpret_cast<const char*>(edge_factors.data()),
              edge_factors.size() * sizeof(std::uint32_t));
          return_proto.set_factorized_metric_edge_cameras_u32(
              reinterpret_cast<const char*>(edge_cameras.data()),
              edge_cameras.size() * sizeof(std::uint32_t));
          return_proto.set_factorized_metric_edge_blocks_f64(
              reinterpret_cast<const char*>(edge_values.data()),
              edge_values.size() * sizeof(double));
        }
        if (ConsensusUnflooredCameraDiagonalEnabled()) {
          const std::vector<double>& consensusMetric = ConsensusMetricBlocks();
          metric_upper_blocks.clear();
          for (int camera = 0; camera < numCameras; ++camera) {
            const int offset = 81 * camera;
            for (int row = 0; row < 9; ++row) {
              for (int column = row; column < 9; ++column) {
                metric_upper_blocks.push_back(
                    consensusMetric[offset + 9 * row + column]);
              }
            }
          }
          return_proto.set_consensus_step_size_upper_f64(
              reinterpret_cast<const char*>(metric_upper_blocks.data()),
              metric_upper_blocks.size() * sizeof(double));
        }
      }
      if (include_consensus_rhs) {
        const std::vector<double>& consensusMetric = ConsensusMetricBlocks();
        std::vector<double> consensus_rhs(9 * numCameras, 0.);
        for (int camera = 0; camera < numCameras; ++camera) {
          const int camera_offset = 9 * camera;
          const int metric_offset = 81 * camera;
          for (int row = 0; row < 9; ++row) {
            double value = 0.;
            for (int column = 0; column < 9; ++column) {
                const double metric_value =
                  consensusMetric[metric_offset + 9 * row + column];
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
      return_proto.set_trust_region_radius(tr_radius);
        return_proto.set_linear_iterations(last_linear_iterations);
        return_proto.set_linear_relative_residual(
          last_linear_relative_residual);
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
          return_proto.set_unique_camera_interior_defect_squared(
            metric_diagnostic.unique_camera_interior_defect_squared);
          return_proto.set_landmark_interior_defect_squared(
            metric_diagnostic.landmark_interior_defect_squared);
          return_proto.set_interior_defect_squared(
            metric_diagnostic.interior_defect_squared);
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

    void MaterializeAcceptedLandmarkState(
        landmark_state_reply_proto& return_proto) const {
      THROW_IF(accepted_landmarks.size() != landmarks.size());
      AddPhysicalLandmarks(accepted_landmarks, return_proto);
    }

    void RestoreBestOutputLandmarkState() {
      THROW_IF(best_output_landmarks.size() != landmarks.size());
      landmarks = best_output_landmarks;
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
      if (update.camera_proximal_multiplier_size() > 0) {
        THROW_IF(update.camera_proximal_multiplier_size() != numCameras);
        camera_proximal_multipliers.assign(
        update.camera_proximal_multiplier().begin(),
        update.camera_proximal_multiplier().end());
      }
      block_curvature_multiplier = update.block_curvature_multiplier();
      metric_diagnostic_iterations = update.metric_diagnostic_iterations();
      outer_iteration = update.outer_iteration();
      oracle_kind = update.oracle_kind();
      collect_camera_diagonal_metrics =
        update.collect_camera_diagonal_metrics();
      factorized_coupled_schur_metric =
        update.factorized_coupled_schur_metric();
      if (collect_camera_diagonal_metrics) {
        std::ostringstream metric;
        metric << "CAMERA_DIAGONAL_REQUEST cluster=" << cluster_id
               << " outer_iteration=" << outer_iteration
               << " oracle_kind=" << oracle_kind << "\n";
        EmitLocalSolveMetric(metric.str());
      }
        proximal_defect_diagnostic = update.proximal_defect_diagnostic();
        diagonal_trust_damping = update.diagonal_trust_damping();
        nesterov_relative_residual = update.nesterov_relative_residual();
        THROW_IF(update.local_iterations() <= 0 ||
          update.local_iterations() > 20);
        local_iterations = update.local_iterations();
      landmark_refinement_steps = update.landmark_refinement_steps();
        nesterov_max_iterations = update.nesterov_max_iterations();
        nesterov_min_iterations = update.nesterov_min_iterations();
        nesterov_stop_tolerance = update.nesterov_stop_tolerance();
        THROW_IF(nesterov_max_iterations <= 0 || nesterov_max_iterations > 1000);
        THROW_IF(nesterov_min_iterations <= 0
          || nesterov_min_iterations > nesterov_max_iterations);
        THROW_IF(!(nesterov_stop_tolerance > 0.)
          || !(nesterov_stop_tolerance < 1.));
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
          tr_radius = std::max(MinimumTrustRegionRadius(), std::min(
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
          tr_radius = std::max(MinimumTrustRegionRadius(), std::min(
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
      if (update.forced_trust_region_radius() > 0.) {
        tr_radius = std::max(
            MinimumTrustRegionRadius(),
            std::min(
                max_trust_region_radius,
                update.forced_trust_region_radius()));
        last_tr_radius = tr_radius;
        persistent_trust_region_active = true;
      }
      options.initial_trust_region_radius = tr_radius;
    }

    // With that Jl changes but it does not matter.
    void UpdateStepSize() { // Recompute.
      const auto [Jp, Jl] = GetJacobian();
      if (firstIteration) {
        const SparseMatrix<double, RowMajor> JlJ =
            BlockDiagonalJtJ<3>(Jl, numLandmarks);
        const auto diag = JlJ.diagonal().array().cwiseAbs().cwiseSqrt().cwiseMax(LegacyLandmarkJacobianSqrtFloor());
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

    const std::vector<double>& ConsensusMetricBlocks() const {
      return ConsensusUnflooredCameraDiagonalEnabled()
          && consensus_stepSize.size() == full_stepSize.size()
          ? consensus_stepSize : full_stepSize;
    }
    bool UsesCentralizedLeftSe3Solver() const {
      return local_linear_solver == 2 || local_linear_solver == 3;
    }
    bool UsesCentralizedProximalLeftSe3Solver() const {
      return local_linear_solver == 3;
    }
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
  int flooredEntries = 0;
  double minimumDiagonal = std::numeric_limits<double>::infinity();
  std::vector<double> cameraDiagonals;
  if (LocalSolveMetricsEnabled()) {
    cameraDiagonals.reserve(9 * numCameras);
  }
  // ToDo: Is this ok or an issue to be resolved differently?
  for (int b = 0; b < numCameras; ++b) {
    for(int id = 0; id < 81; id += 10) { // diagonal entries !?
      const double diagonal = full_stepSize[81*b + id];
      if (LocalSolveMetricsEnabled()) {
        cameraDiagonals.push_back(diagonal);
      }
      minimumDiagonal = std::min(minimumDiagonal, diagonal);
      flooredEntries += diagonal < CameraPreconditionerDiagonalFloor();
      full_stepSize[81*b + id] =
          std::max(CameraPreconditionerDiagonalFloor(), diagonal);
    }
  }
  if (LocalSolveMetricsEnabled()) {
    std::ostringstream metric;
    metric << "CAMERA_PRECONDITIONER_FLOOR cluster=" << cluster_id
           << " entries=" << 9 * numCameras
           << " floored=" << flooredEntries
           << " minimum=" << minimumDiagonal
           << " q01=" << Quantile(cameraDiagonals, 0.01)
           << " floor=" << CameraPreconditionerDiagonalFloor() << "\n";
    EmitLocalSolveMetric(metric.str());
  }
}

struct NesterovInnerTiming {
  double inverse_landmark_blocks = 0.;
  double inverse_camera_blocks = 0.;
  double rhs_landmark_multiply = 0.;
  double rhs_w_multiply = 0.;
  double rhs_vector = 0.;
  double initialize = 0.;
  double iter_w_transpose = 0.;
  double iter_landmark_multiply = 0.;
  double iter_w_multiply = 0.;
  double iter_camera_multiply = 0.;
  double iter_vector = 0.;
  double iter_stop = 0.;
  double final_w_transpose = 0.;
  double final_landmark_multiply = 0.;
  double inner_total = 0.;
  std::uint64_t iterative_edge_visits = 0;
  int calls = 0;
  int iterations = 0;
};

std::pair<Matrix<double, Eigen::Dynamic, 1>, Matrix<double, Eigen::Dynamic, 1>>
SolveByGDNesterov(SparseMatrix<double, RowMajor> Uli, SparseMatrix<double, RowMajor> Vli, 
                const BlockEdgeMatrix& W,
                const Matrix<double, Eigen::Dynamic, 1>& bp,
                const Matrix<double, Eigen::Dynamic, 1>& bl,
                const Matrix<double, Eigen::Dynamic, 1>& proximalGradient,
                int power_iterations, int minimum_iterations,
                double stop_tolerance, bool relative_residual,
                int* completed_iterations, double* completed_relative_residual,
                NesterovInnerTiming* timing) {

  *completed_iterations = 0;
  *completed_relative_residual = std::numeric_limits<double>::quiet_NaN();
  const SparseMatrix<double, RowMajor> camera_system = Uli;
  const bool collect_timing = timing != nullptr;
  const auto inner_start = collect_timing
      ? TimingClock::now() : TimingClock::time_point{};
  if (collect_timing) {
    ++timing->calls;
  }

  // compute bS, Vli, W
  auto operation_start = collect_timing
      ? TimingClock::now() : TimingClock::time_point{};
    const Eigen::Map<const Eigen::VectorXd> landmark_block_values_before(
      Vli.valuePtr(), Vli.nonZeros());
    const bool landmark_blocks_finite_before =
      landmark_block_values_before.allFinite();
    const double landmark_blocks_max_before =
      landmark_block_values_before.cwiseAbs().maxCoeff();
  BlockInverse<3>(Vli);
  if (collect_timing) {
    timing->inverse_landmark_blocks += ElapsedSeconds(operation_start);
  }

  if (power_iterations == 0) {  // quick hack: xk = delta_p = 0
    operation_start = collect_timing
        ? TimingClock::now() : TimingClock::time_point{};
    Matrix<double, Eigen::Dynamic, 1> ubs = Uli * bp;
    Matrix<double, Eigen::Dynamic, 1> xk = 0 * ubs;
    Matrix<double, Eigen::Dynamic, 1> delta_l = Vli * (-bl);
    if (collect_timing) {
      timing->initialize += ElapsedSeconds(operation_start);
      timing->inner_total += ElapsedSeconds(inner_start);
    }
    return {xk, delta_l};
   }

  operation_start = collect_timing
      ? TimingClock::now() : TimingClock::time_point{};
    const Eigen::Map<const Eigen::VectorXd> camera_block_values_before(
      Uli.valuePtr(), Uli.nonZeros());
    const bool camera_blocks_finite_before =
      camera_block_values_before.allFinite();
    const double camera_blocks_max_before =
      camera_block_values_before.cwiseAbs().maxCoeff();
  BlockInverse<9>(Uli);
  if (collect_timing) {
    timing->inverse_camera_blocks += ElapsedSeconds(operation_start);
  }
  const double Lip = NesterovSchurLipschitz();
  double lambda0 = (1. + std::sqrt(5.)) / 2.;
  operation_start = collect_timing
      ? TimingClock::now() : TimingClock::time_point{};
  Matrix<double, Eigen::Dynamic, 1> bS = bp;
  // bS = (bp_s                     - W * Vli * bl).flatten() # see XX equals 2 * (bp - W * Vli * bl)
  //       bp_s = bp + stepSize * prox_rhs
  // bS = (bp + stepSize * prox_rhs - W * Vli * bl).flatten() # see XX equals 2 * (bp - W * Vli * bl)
  bS += proximalGradient;
  if (collect_timing) {
    timing->rhs_vector += ElapsedSeconds(operation_start);
  }
  operation_start = collect_timing
      ? TimingClock::now() : TimingClock::time_point{};
  Eigen::VectorXd landmarkWorkspace = Vli * bl;
  if (collect_timing) {
    timing->rhs_landmark_multiply += ElapsedSeconds(operation_start);
  }
  Eigen::VectorXd cameraWorkspace(W.rows());
  operation_start = collect_timing
      ? TimingClock::now() : TimingClock::time_point{};
  W.Multiply(landmarkWorkspace, cameraWorkspace);
  if (collect_timing) {
    timing->rhs_w_multiply += ElapsedSeconds(operation_start);
  }
  operation_start = collect_timing
      ? TimingClock::now() : TimingClock::time_point{};
  bS -= cameraWorkspace;
  if (collect_timing) {
    timing->rhs_vector += ElapsedSeconds(operation_start);
  }

  // std::cout << " Jl " << Jl.valuePtr()[0] << " " << Jl.valuePtr()[1] << " " << Jl.valuePtr()[2] << "\n";
  // std::cout << "bS :" << bS.array() << "\n";
  // std::cout << "bp :" << (Jp.transpose() * res).array() << "\n";
  // std::cout << "bl :" << (Jl.transpose() * res).array() << "\n";
  // std::cout << "res :" << res.array() << "\n"; //ok

  operation_start = collect_timing
      ? TimingClock::now() : TimingClock::time_point{};
  Matrix<double, Eigen::Dynamic, 1> ubs = -Uli * bS;
  const double initial_gradient_squared_norm = std::max(
      ubs.squaredNorm(), std::numeric_limits<double>::min());
  // Todo : * 1. / Lip ? or not
  Matrix<double, Eigen::Dynamic, 1> xk = - 1. / Lip * ubs; // xk =0, g = ubs, yk = -1. / Lip * g = - 1. / Lip * ubs; xk = (1-gamma) yk + gamma y0, gamma = 0
  Matrix<double, Eigen::Dynamic, 1> y0 = - 1. / Lip * ubs; // xk =0, g = ubs, yk = -1. / Lip * g = - 1. / Lip * ubs; y0 = yk.
  if (!ubs.allFinite()) {
    const Eigen::Map<const Eigen::VectorXd> camera_inverse_values(
        Uli.valuePtr(), Uli.nonZeros());
    const Eigen::Map<const Eigen::VectorXd> landmark_inverse_values(
        Vli.valuePtr(), Vli.nonZeros());
    std::cerr << "Non-finite Nesterov initialization:"
              << " camera_blocks_finite_before="
              << camera_blocks_finite_before
              << " camera_blocks_max_before="
              << camera_blocks_max_before
              << " camera_inverse_finite="
              << camera_inverse_values.allFinite()
              << " landmark_blocks_finite_before="
              << landmark_blocks_finite_before
              << " landmark_blocks_max_before="
              << landmark_blocks_max_before
              << " landmark_inverse_finite="
              << landmark_inverse_values.allFinite()
              << " bp_finite=" << bp.allFinite()
              << " bl_finite=" << bl.allFinite()
              << " proximal_gradient_finite="
              << proximalGradient.allFinite()
              << " landmark_workspace_finite="
              << landmarkWorkspace.allFinite()
              << " camera_workspace_finite="
              << cameraWorkspace.allFinite()
              << " schur_rhs_finite=" << bS.allFinite()
              << std::endl;
  }
  Matrix<double, Eigen::Dynamic, 1> wtX(W.cols());
  Matrix<double, Eigen::Dynamic, 1> vinvWtX(W.cols());
  Matrix<double, Eigen::Dynamic, 1> wVinvWtX(W.rows());
  Matrix<double, Eigen::Dynamic, 1> uinvWVinvWtX(W.rows());
  Matrix<double, Eigen::Dynamic, 1> g(W.rows());
  Matrix<double, Eigen::Dynamic, 1> yk(W.rows());
  if (collect_timing) {
    timing->initialize += ElapsedSeconds(operation_start);
  }
  // Lip = 0.9 # 100 -> 1. # TODO: play, find out how to progress over time.
  // lambda0 = (1.+np.sqrt(5.)) / 2. # l=0 g=1, 0, .. L0=1 g = 0,..

  // std::cout << "xk :" << xk.squaredNorm() << "\n";

  for (int i = 0; i < power_iterations; ++i) {
      *completed_iterations = i + 1;
      if (collect_timing) {
        ++timing->iterations;
        timing->iterative_edge_visits += 2 * W.EdgeCount();
      }
      operation_start = collect_timing
          ? TimingClock::now() : TimingClock::time_point{};
      const double lambda1 = (1. + std::sqrt(1. + 4. * lambda0 * lambda0)) / 2.;
      const double gamma = (1. - lambda0) / lambda1;
      lambda0 = lambda1;

      //     g = xk - Uli * ( W * (Vli * (W.transpose() * xk))) + ubs
      //     yk = xk - 1/Lip * g
      //     xk = (1-gamma) * yk + gamma * y0
      //     y0 = yk
      // const Matrix<double, Eigen::Dynamic, 1> g = (xk - Uli * (W * (Vli * (W.transpose() * xk).eval()).eval()).eval() + ubs).eval();
      if (collect_timing) {
        timing->iter_vector += ElapsedSeconds(operation_start);
      }
      operation_start = collect_timing
          ? TimingClock::now() : TimingClock::time_point{};
      W.TransposeMultiply(xk, wtX);
      if (collect_timing) {
        timing->iter_w_transpose += ElapsedSeconds(operation_start);
      }
      operation_start = collect_timing
          ? TimingClock::now() : TimingClock::time_point{};
      vinvWtX.noalias() = Vli * wtX;
      if (collect_timing) {
        timing->iter_landmark_multiply += ElapsedSeconds(operation_start);
      }
      operation_start = collect_timing
          ? TimingClock::now() : TimingClock::time_point{};
      W.Multiply(vinvWtX, wVinvWtX);
      if (collect_timing) {
        timing->iter_w_multiply += ElapsedSeconds(operation_start);
      }
      operation_start = collect_timing
          ? TimingClock::now() : TimingClock::time_point{};
      uinvWVinvWtX.noalias() = Uli * wVinvWtX;
      if (collect_timing) {
        timing->iter_camera_multiply += ElapsedSeconds(operation_start);
      }
      operation_start = collect_timing
          ? TimingClock::now() : TimingClock::time_point{};
      g = xk;
      g -= uinvWVinvWtX;
      g += ubs;
      yk = xk;
      yk -= 1. / Lip * g;
      xk = (1. - gamma) * yk + gamma * y0;
      y0 = yk;
      if (collect_timing) {
        timing->iter_vector += ElapsedSeconds(operation_start);
      }
      if (!xk.allFinite()) {
        std::cerr << "Non-finite Nesterov iterate: iteration=" << i + 1
                  << " lipschitz=" << Lip
                  << " ubs_finite=" << ubs.allFinite()
                  << " wt_finite=" << wtX.allFinite()
                  << " vinv_wt_finite=" << vinvWtX.allFinite()
                  << " w_vinv_wt_finite=" << wVinvWtX.allFinite()
                  << " uinv_w_vinv_wt_finite="
                  << uinvWVinvWtX.allFinite()
                  << " gradient_finite=" << g.allFinite()
                  << " extrapolated_finite=" << yk.allFinite()
                  << std::endl;
        break;
      }

      //std::cout << i << ". xk :" << xk.squaredNorm() << "\n";

      operation_start = collect_timing
          ? TimingClock::now() : TimingClock::time_point{};
        const bool should_stop = relative_residual
          ? g.squaredNorm() < stop_tolerance * stop_tolerance
            * initial_gradient_squared_norm
          : stop_criterion(
            xk.squaredNorm(), g.squaredNorm(), Lip, i, stop_tolerance);
      if (collect_timing) {
        timing->iter_stop += ElapsedSeconds(operation_start);
      }
        if (i + 1 >= minimum_iterations
          && (i + 1) % NesterovStopCheckInterval() == 0
          && should_stop) {
          break;
      }
  }

  operation_start = collect_timing
      ? TimingClock::now() : TimingClock::time_point{};
  W.TransposeMultiply(xk, wtX);
  if (collect_timing) {
    timing->final_w_transpose += ElapsedSeconds(operation_start);
  }
  operation_start = collect_timing
      ? TimingClock::now() : TimingClock::time_point{};
  Matrix<double, Eigen::Dynamic, 1> delta_l = Vli * (wtX - bl);
    const Eigen::VectorXd solution = -xk;
    W.TransposeMultiply(solution, wtX);
    vinvWtX.noalias() = Vli * wtX;
    W.Multiply(vinvWtX, wVinvWtX);
    const Eigen::VectorXd reduced_residual =
      -bS - (camera_system * solution - wVinvWtX);
    *completed_relative_residual = reduced_residual.norm() /
      std::max(bS.norm(), std::numeric_limits<double>::min());
  if (collect_timing) {
    timing->final_landmark_multiply += ElapsedSeconds(operation_start);
    timing->inner_total += ElapsedSeconds(inner_start);
  }
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
    const std::vector<Eigen::Matrix<double, 9, 9>>* factorized_tangent,
    int* iterations_out, int* termination_out,
    double* relative_residual_out) {
  SparseMatrix<double, RowMajor> Vinv = Vli;
  SparseMatrix<double, RowMajor> Uinv = Uli;
  BlockInverse<3>(Vinv);
  if (factorized_tangent != nullptr) {
    Uinv += FactorizedProximalBlockDiagonalInTangent(
        *factorized_tangent);
  }
  if (SchurPcgJacobiPreconditionerEnabled()) {
    const double landmark_trace = Vli.diagonal().sum();
    const double landmark_inverse_trace = Vinv.diagonal().sum();
    W.SubtractSchurDiagonal(Vinv, Uinv);
    FloorSymmetricBlocks<9>(Uinv, PobaBlockRelativeFloor());
    if (LocalSolveMetricsEnabled()) {
      Eigen::VectorXd probe(Uinv.rows());
      for (int parameter = 0; parameter < probe.size(); ++parameter) {
        probe[parameter] =
            std::sin(0.6180339887498949 * (parameter + 1));
      }
      double trace = 0.;
      double squared_norm = 0.;
      double minimum_eigenvalue = std::numeric_limits<double>::infinity();
      for (int camera = 0; camera < Uinv.rows() / 9; ++camera) {
        Eigen::Matrix<double, 9, 9> block;
        for (int row = 0; row < 9; ++row) {
          for (int column = 0; column < 9; ++column) {
            block(row, column) = Uinv.coeff(
                9 * camera + row, 9 * camera + column);
          }
        }
        trace += block.trace();
        squared_norm += block.squaredNorm();
        minimum_eigenvalue = std::min(minimum_eigenvalue,
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>>(block)
                .eigenvalues().minCoeff());
      }
      std::ostringstream metric;
      metric << "SCHUR_JACOBI_MATRIX cluster=" << cluster_id
              << " camera_trace=" << Uli.diagonal().sum()
               << " cross_frobenius=" << std::sqrt(W.SquaredNorm())
               << " landmark_trace=" << landmark_trace
               << " landmark_inverse_trace=" << landmark_inverse_trace
              << " elimination_trace="
              << Uli.diagonal().sum() - Uinv.diagonal().sum()
             << " trace=" << trace
             << " frobenius=" << std::sqrt(squared_norm)
             << " probe=" << probe.dot(Uinv * probe)
             << " minimum_eigenvalue=" << minimum_eigenvalue << "\n";
      EmitLocalSolveMetric(metric.str());
    }
  }
  BlockInverse<9>(Uinv);

  Eigen::VectorXd vinvBl = Vinv * bl;
  Eigen::VectorXd wVinvBl(W.rows());
  W.Multiply(vinvBl, wVinvBl);
  Eigen::VectorXd right_hand_side = -bp - proximalGradient + wVinvBl;

  Eigen::VectorXd solution = Eigen::VectorXd::Zero(W.rows());
  Eigen::VectorXd residual = right_hand_side;
  Eigen::VectorXd preconditioned = Uinv * residual;
  Eigen::VectorXd direction = preconditioned;
  if (LocalSolveMetricsEnabled()) {
    double translation_squared = 0.;
    double rotation_squared = 0.;
    double intrinsics_squared = 0.;
    for (int camera = 0; camera < preconditioned.size() / 9; ++camera) {
      translation_squared +=
          preconditioned.segment<3>(9 * camera).squaredNorm();
      rotation_squared +=
          preconditioned.segment<3>(9 * camera + 3).squaredNorm();
      intrinsics_squared +=
          preconditioned.segment<3>(9 * camera + 6).squaredNorm();
    }
    std::ostringstream metric;
    metric << "SCHUR_PCG_INITIAL cluster=" << cluster_id
           << " rhs=" << right_hand_side.norm()
           << " preconditioned=" << preconditioned.norm()
           << " translation=" << std::sqrt(translation_squared)
           << " rotation=" << std::sqrt(rotation_squared)
           << " intrinsics=" << std::sqrt(intrinsics_squared) << "\n";
    EmitLocalSolveMetric(metric.str());
  }
  Eigen::VectorXd wtDirection(W.cols());
  Eigen::VectorXd vinvWtDirection(W.cols());
  Eigen::VectorXd wVinvWtDirection(W.rows());
  Eigen::VectorXd schurDirection(W.rows());
  double residual_preconditioned = residual.dot(preconditioned);
  double quadratic_model = 0.;
  const double initial_residual_norm = std::max(
      residual.norm(), std::numeric_limits<double>::min());
  int iterations = 0;
  *termination_out = 0;

  for (; iterations < SchurPcgMaximumIterations(); ++iterations) {
    W.TransposeMultiply(direction, wtDirection);
    vinvWtDirection.noalias() = Vinv * wtDirection;
    W.Multiply(vinvWtDirection, wVinvWtDirection);
    schurDirection.noalias() = Uli * direction;
    if (factorized_tangent != nullptr) {
      schurDirection += ApplyCameraProximalMetricInTangent(
          direction, *factorized_tangent);
    }
    schurDirection -= wVinvWtDirection;
    const double denominator = direction.dot(schurDirection);
    if (!(denominator > 0.) || !std::isfinite(denominator)) {
      *termination_out = 2;
      break;
    }
    const double alpha = residual_preconditioned / denominator;
    if (iterations == 0 && LocalSolveMetricsEnabled()) {
      std::ostringstream metric;
      metric << "SCHUR_PCG_FIRST_ALPHA cluster=" << cluster_id
             << " alpha=" << alpha
             << " numerator=" << residual_preconditioned
             << " denominator=" << denominator << "\n";
      EmitLocalSolveMetric(metric.str());
    }
    solution += alpha * direction;
    residual -= alpha * schurDirection;
    const double next_quadratic_model =
        -solution.dot(right_hand_side + residual);
    const double zeta = (iterations + 1)
        * (next_quadratic_model - quadratic_model) / next_quadratic_model;
    if (SchurPcgQTolerance() > 0. &&
        zeta < SchurPcgQTolerance()) {
      ++iterations;
      *termination_out = 4;
      break;
    }
    quadratic_model = next_quadratic_model;
    if (SchurPcgQTolerance() == 0. && residual.norm() <=
      SchurPcgRelativeTolerance() * initial_residual_norm) {
      ++iterations;
      *termination_out = 1;
      break;
    }
    preconditioned.noalias() = Uinv * residual;
    const double next_residual_preconditioned = residual.dot(preconditioned);
    if (!(next_residual_preconditioned >= 0.) ||
        !std::isfinite(next_residual_preconditioned)) {
      *termination_out = 3;
      break;
    }
    const double beta = next_residual_preconditioned /
        std::max(residual_preconditioned, std::numeric_limits<double>::min());
    direction = preconditioned + beta * direction;
    residual_preconditioned = next_residual_preconditioned;
  }
  *iterations_out = iterations;
  *relative_residual_out = residual.norm() / initial_residual_norm;

  Eigen::VectorXd wtSolution(W.cols());
  W.TransposeMultiply(solution, wtSolution);
  Eigen::VectorXd delta_l = Vinv * (-wtSolution - bl);
  return {solution, delta_l};
}

std::pair<Matrix<double, Eigen::Dynamic, 1>, Matrix<double, Eigen::Dynamic, 1>>
SolveByPobaPowerSeries(
    SparseMatrix<double, RowMajor> camera_hessian,
    SparseMatrix<double, RowMajor> landmark_hessian,
    const BlockEdgeMatrix& W,
    const Matrix<double, Eigen::Dynamic, 1>& bp,
    const Matrix<double, Eigen::Dynamic, 1>& bl,
    const Matrix<double, Eigen::Dynamic, 1>& proximal_gradient,
    int maximum_order, double tolerance, int* iterations_out,
    double* relative_residual_out) {
  BlockInverse<9>(camera_hessian);
  BlockInverse<3>(landmark_hessian);

  Eigen::VectorXd landmark_workspace = landmark_hessian * bl;
  Eigen::VectorXd camera_workspace(W.rows());
  W.Multiply(landmark_workspace, camera_workspace);
  Eigen::VectorXd reduced_gradient = bp + proximal_gradient - camera_workspace;

  Eigen::VectorXd term = -camera_hessian * reduced_gradient;
  Eigen::VectorXd solution = term;
  Eigen::VectorXd wt_term(W.cols());
  Eigen::VectorXd vinv_wt_term(W.cols());
  Eigen::VectorXd w_vinv_wt_term(W.rows());
  int completed_orders = 0;
  for (int order = 1; order <= maximum_order; ++order) {
    W.TransposeMultiply(term, wt_term);
    vinv_wt_term.noalias() = landmark_hessian * wt_term;
    W.Multiply(vinv_wt_term, w_vinv_wt_term);
    term.noalias() = camera_hessian * w_vinv_wt_term;
    solution += term;
    completed_orders = order;
    if ((order + 1.) * term.norm()
        < tolerance * std::max(solution.norm(),
            std::numeric_limits<double>::min())) {
      break;
    }
  }
  *iterations_out = completed_orders;
  *relative_residual_out = (completed_orders + 1.) * term.norm() /
      std::max(solution.norm(), std::numeric_limits<double>::min());

  Eigen::VectorXd wt_solution(W.cols());
  W.TransposeMultiply(solution, wt_solution);
  Eigen::VectorXd delta_l = landmark_hessian * (-bl - wt_solution);
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
  if (firstIteration && !DisableLandmarkPreconditioningEnabled()) {
      // diag is a reference .. why? i do stuff on it.
      const Eigen::VectorXd landmarkDiagonal = Vl.diagonal();
      const double landmarkFloor = LandmarkPreconditionerFloor();
      if (LocalSolveMetricsEnabled()) {
        std::vector<double> landmarkDiagonals(
            landmarkDiagonal.data(),
            landmarkDiagonal.data() + landmarkDiagonal.size());
        std::ostringstream metric;
        metric << "LANDMARK_PRECONDITIONER_FLOOR cluster=" << cluster_id
               << " entries=" << landmarkDiagonal.size()
               << " floored="
               << (landmarkDiagonal.array() < landmarkFloor).count()
               << " minimum=" << landmarkDiagonal.minCoeff()
               << " q01=" << Quantile(landmarkDiagonals, 0.01)
               << " floor=" << landmarkFloor << "\n";
        EmitLocalSolveMetric(metric.str());
      }
      const auto diag = landmarkDiagonal.array().cwiseMax(landmarkFloor).cwiseSqrt().cwiseInverse().eval();
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
    firstIteration = false;
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
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagVL = Diagonal<3>(Vl); // Vl = VL + L * diagVL
  SparseMatrix<double, RowMajor> metricCameraHessian = Ul;
  const bool factorized_coupled_schur_metric_active =
      factorized_coupled_schur_metric;
  SparseMatrix<double, RowMajor> metricLandmarkInverse;
  if (SchurProximalMetricEnabled() ||
      factorized_coupled_schur_metric_active) {
    metricLandmarkInverse = Vl;
    BlockInverse<3>(metricLandmarkInverse);
    camera_landmark_hessian.SubtractSchurDiagonal(
        metricLandmarkInverse, metricCameraHessian);
    FloorSymmetricBlocks<9>(metricCameraHessian,
      std::max(PobaBlockRelativeFloor(), 1e-16));
  }
  const Eigen::DiagonalMatrix<double, Eigen::Dynamic> consensusDiagUP =
    CameraDiagonalMetricScale() * Diagonal<9>(
      metricCameraHessian, cluster_id, collect_camera_diagonal_metrics,
      outer_iteration, oracle_kind, "consensus",
      &global_camera_ids); // Vp = Vp + L * diagVp
  const SparseMatrix<double, RowMajor>& landmarkHessian = normalEquations.landmark_hessian;

  //const double scale = 1e-1; // 1e0: @29: 501k, no jump. 1e1 many jumps. 473k
  // TODO
    const double legacy_scale =
      std::min(1.005, 1e-1 * std::sqrt(current_be / start_be));
    const double scale = block_curvature_multiplier > 0.
      ? block_curvature_multiplier
      : legacy_scale;

    const bool freeze_current_metric = FreezeBlockMetricEnabled() &&
      frozen_block_metric_initialized &&
      (!factorized_coupled_schur_metric_active ||
       factorized_proximal_active);
    const bool factorized_metric_refresh =
      factorized_coupled_schur_metric_active && !freeze_current_metric;
    if (factorized_metric_refresh && !firstIteration) {
    SparseMatrix<double, RowMajor> factorized_base_diagonal = Ul;
    camera_landmark_hessian.AddBucketedSchurOffDiagonalFrobeniusBounds(
      metricLandmarkInverse,
      camera_proximal_multipliers,
      CoupledSchurProximalMetricStabilization(),
      FactorizedSchurProximalMetricStabilizationBuckets(),
      factorized_base_diagonal);
    SparseMatrix<double, RowMajor> factorized_base_block_diagonal =
      factorized_base_diagonal;
    camera_landmark_hessian.SubtractSchurDiagonal(
      metricLandmarkInverse, factorized_base_block_diagonal);
    SparseMatrix<double, RowMajor> factorized_floored_block_diagonal =
      factorized_base_block_diagonal;
    FloorSymmetricBlocks<9>(factorized_floored_block_diagonal,
      std::max(PobaBlockRelativeFloor(), 1e-16));
    factorized_base_diagonal +=
      factorized_floored_block_diagonal
      - factorized_base_block_diagonal;
    SparseMatrix<double, RowMajor> factorized_diagonal =
      scale * factorized_base_diagonal;
    factorized_diagonal += consensusDiagUP * current_be;
    ScaleCameraProximalMetric(factorized_diagonal);
    BlockEdgeMatrix factorized_cross = camera_landmark_hessian;
    std::vector<double> factor_scales(numCameras);
    for (int camera = 0; camera < numCameras; ++camera) {
      factor_scales[camera] = std::sqrt(
        scale * camera_proximal_multipliers[camera]);
    }
    factorized_cross.ScaleCameraRows(factor_scales);
    SetFactorizedCameraProximalMetric(
      factorized_diagonal,
      factorized_cross,
      metricLandmarkInverse);
    }

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
      if (freeze_current_metric ||
          factorized_coupled_schur_metric_active) {
        stepSize = camera_proximal_metric;
      } else {
        stepSize = scale * metricCameraHessian;
        stepSize += consensusDiagUP * current_be;
        ScaleCameraProximalMetric(stepSize);
        ApplySo3SubspaceMetricRatio(stepSize);
      }
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
      if (!factorized_coupled_schur_metric_active) {
        SetCameraProximalMetric(stepSize);
      }
      frozen_block_metric_initialized = true;
    }
    consensus_stepSize = full_stepSize;
    if (!scalar_proximal_prior &&
        ConsensusUnflooredCameraDiagonalEnabled()) {
      SparseMatrix<double, RowMajor> voteMetric = scale * Ul;
      if (SchurProximalMetricEnabled()) {
        voteMetric = scale * metricCameraHessian;
      }
      voteMetric += current_be * CameraDiagonalMetricScale()
          * metricCameraHessian.diagonal().asDiagonal();
      ScaleCameraProximalMetric(voteMetric);
        ApplySo3SubspaceMetricRatio(voteMetric);
      const double* voteValues = voteMetric.valuePtr();
      std::copy(voteValues, voteValues + consensus_stepSize.size(),
          consensus_stepSize.data());
    }
    if (!factorized_coupled_schur_metric_active) {
      Ul += stepSize;
    }
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
      Ul += scale * metricCameraHessian;
      Ul += consensusDiagUP * current_be;
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
  SparseMatrix<double, RowMajor> cameraHessian =
      normalEquations.camera_hessian;
  Eigen::VectorXd bp = normalEquations.camera_gradient;
  BlockEdgeMatrix W = camera_landmark_hessian;
  std::vector<Eigen::Matrix<double, 9, 9>> tangentToScaled;
  if (ManifoldCameraUpdatesEnabled()) {
    tangentToScaled = TangentToScaledJacobians();
    if (DirectTangentNormalEquationsEnabled()) {
      const NormalEquations tangent_normal_equations =
          GetBatchedNormalEquations(true);
      cameraHessian = tangent_normal_equations.camera_hessian;
      bp = tangent_normal_equations.camera_gradient;
        Vl = tangent_normal_equations.landmark_hessian;
        diagVL = Diagonal<3>(Vl);
        normalEquations.landmark_hessian =
          tangent_normal_equations.landmark_hessian;
        normalEquations.landmark_gradient =
          tangent_normal_equations.landmark_gradient;
      W = camera_landmark_hessian;
      Ul = cameraHessian;
      if (DisableLocalProximalTermEnabled()) {
        // K1 BA reference: LM damping supplies SPD regularization.
      } else {
        if (!factorized_proximal_active) {
          Ul += TransformCameraBlockMatrix(
              camera_proximal_metric, tangentToScaled);
        }
      }
    } else {
      cameraHessian = TransformCameraBlockMatrix(
          cameraHessian, tangentToScaled);
      Ul = TransformCameraBlockMatrix(Ul, tangentToScaled);
      bp = TransformCameraVector(bp, tangentToScaled);
      W.LeftMultiplyByCameraBlockTransposes(tangentToScaled);
    }
  }
  if (collectTiming) {
    double translation_gradient_squared = 0.;
    double rotation_gradient_squared = 0.;
    double intrinsics_gradient_squared = 0.;
    double normalized_translation_squared = 0.;
    double normalized_rotation_squared = 0.;
    double normalized_intrinsics_squared = 0.;
    std::array<double, 3> minimum_diagonal = {
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()};
    std::array<int, 3> nonpositive_diagonal = {0, 0, 0};
    for (int camera = 0; camera < numCameras; ++camera) {
      const int offset = 9 * camera;
      translation_gradient_squared += bp.segment<3>(offset).squaredNorm();
      rotation_gradient_squared += bp.segment<3>(offset + 3).squaredNorm();
      intrinsics_gradient_squared += bp.segment<3>(offset + 6).squaredNorm();
      for (int parameter = 0; parameter < 9; ++parameter) {
        const double diagonal = std::max(
            cameraHessian.coeff(offset + parameter, offset + parameter),
            std::numeric_limits<double>::min());
        const double raw_diagonal =
          cameraHessian.coeff(offset + parameter, offset + parameter);
        const int group = parameter / 3;
        minimum_diagonal[group] = std::min(
          minimum_diagonal[group], raw_diagonal);
        nonpositive_diagonal[group] += raw_diagonal <= 0.;
        const double normalized = bp[offset + parameter] / std::sqrt(diagonal);
        if (parameter < 3) {
          normalized_translation_squared += normalized * normalized;
        } else if (parameter < 6) {
          normalized_rotation_squared += normalized * normalized;
        } else {
          normalized_intrinsics_squared += normalized * normalized;
        }
      }
    }
    double normalized_landmarks_squared = 0.;
    for (int parameter = 0;
         parameter < normalEquations.landmark_gradient.size(); ++parameter) {
      const double diagonal = std::max(
          normalEquations.landmark_hessian.coeff(parameter, parameter),
          std::numeric_limits<double>::min());
      const double normalized =
          normalEquations.landmark_gradient[parameter] / std::sqrt(diagonal);
      normalized_landmarks_squared += normalized * normalized;
    }
    std::ostringstream metric;
    metric << "GRADIENT_GROUPS cluster=" << cluster_id
           << " translation=" << std::sqrt(translation_gradient_squared)
           << " rotation=" << std::sqrt(rotation_gradient_squared)
           << " intrinsics=" << std::sqrt(intrinsics_gradient_squared)
           << " landmarks=" << normalEquations.landmark_gradient.norm()
           << " normalized_translation="
           << std::sqrt(normalized_translation_squared)
           << " normalized_rotation=" << std::sqrt(normalized_rotation_squared)
           << " normalized_intrinsics="
           << std::sqrt(normalized_intrinsics_squared)
           << " normalized_landmarks="
           << std::sqrt(normalized_landmarks_squared)
           << " minimum_translation_diagonal=" << minimum_diagonal[0]
           << " minimum_rotation_diagonal=" << minimum_diagonal[1]
           << " minimum_intrinsics_diagonal=" << minimum_diagonal[2]
           << " nonpositive_translation_diagonal="
           << nonpositive_diagonal[0]
           << " nonpositive_rotation_diagonal=" << nonpositive_diagonal[1]
           << " nonpositive_intrinsics_diagonal=" << nonpositive_diagonal[2]
           << "\n";
    EmitLocalSolveMetric(metric.str());

    Eigen::VectorXd camera_probe(cameraHessian.rows());
    Eigen::VectorXd landmark_probe(normalEquations.landmark_hessian.rows());
    for (int parameter = 0; parameter < camera_probe.size(); ++parameter) {
      camera_probe[parameter] =
          std::sin(0.6180339887498949 * (parameter + 1));
    }
    for (int parameter = 0; parameter < landmark_probe.size(); ++parameter) {
      landmark_probe[parameter] = std::sin(
          0.6180339887498949 * (camera_probe.size() + parameter + 1));
    }
    Eigen::VectorXd cross_action(camera_probe.size());
    W.Multiply(landmark_probe, cross_action);
    std::ostringstream quadratic_metric;
    quadratic_metric << "TANGENT_QUADRATIC cluster=" << cluster_id
        << " camera=" << camera_probe.dot(cameraHessian * camera_probe)
        << " landmark=" << landmark_probe.dot(
             normalEquations.landmark_hessian * landmark_probe)
         << " cross=" << 2. * camera_probe.dot(cross_action)
         << " gradient_probe="
         << camera_probe.dot(bp)
           + landmark_probe.dot(normalEquations.landmark_gradient)
         << "\n";
    EmitLocalSolveMetric(quadratic_metric.str());
  }
  const Eigen::DiagonalMatrix<double, Eigen::Dynamic> trustDiagUP =
      CameraDiagonalMetricScale() * Diagonal<9>(
        cameraHessian, cluster_id, collect_camera_diagonal_metrics,
        outer_iteration, oracle_kind, "trust", &global_camera_ids);
      Eigen::DiagonalMatrix<double, Eigen::Dynamic> cameraTrustDiagonal =
        Diagonal<9>(cameraHessian, cluster_id);
      Eigen::DiagonalMatrix<double, Eigen::Dynamic> landmarkTrustDiagonal = diagVL;
  if (collectTiming) {
    std::ostringstream metric;
    metric << "LANDMARK_TRUST_INPUT cluster=" << cluster_id
           << " base_trace=" << Vl.diagonal().sum()
           << " damping_trace=" << landmarkTrustDiagonal.diagonal().sum()
           << " minimum=" << Vl.diagonal().minCoeff()
           << " maximum=" << Vl.diagonal().maxCoeff() << "\n";
    EmitLocalSolveMetric(metric.str());
  }
  // Loop until ok or adjust tr_region
  if (trust_region_policy == 1 && !persistent_trust_region_active) {
    tr_radius = std::min(DabaInitialTrustRegionCap(), max_trust_region_radius);
  } else {
    tr_radius = std::min(max_trust_region_radius, tr_radius);
  }
  double trust_region_decreasing_ratio =
      BaeTrustScheduleEnabled() ? 1. / 16. : 0.5;
  double inv_tr_radius = 0;

  const int power_iterations = nesterov_max_iterations;
  const double costStart = huber_delta > 0.
      ? 2. * GetCost() : residual.squaredNorm();
  const Eigen::VectorXd& bl = normalEquations.landmark_gradient;
  Matrix<double, Eigen::Dynamic, 1> proximalOffset(9 * numCameras);
  for (int id = 0; id < proximalOffset.size(); ++id) {
    proximalOffset[id] = cameras[id] - cameras_s[id];
  }
  Matrix<double, Eigen::Dynamic, 1> proximalGradient(9 * numCameras);
  if (scalar_proximal_prior) {
    blockMult<9>(full_stepSize, proximalOffset, proximalGradient);
  } else {
    proximalGradient = ApplyCameraProximalMetric(proximalOffset);
  }
  double penaltyStart = proximalOffset.dot(proximalGradient);
  if (DisableLocalProximalTermEnabled()) {
    proximalGradient.setZero();
    penaltyStart = 0.;
  }
  if (ManifoldCameraUpdatesEnabled()) {
    proximalGradient = TransformCameraVector(
        proximalGradient, tangentToScaled);
  }
  Matrix<double, Eigen::Dynamic, 1> proximalStep(9 * numCameras);
  Matrix<double, Eigen::Dynamic, 1> crossProduct(9 * numCameras);
  const double assemblySeconds = collectTiming ? ElapsedSeconds(assemblyStart) : 0.;
  int trust_region_attempts = 0;
  int trust_region_rejections = 0;
  int linear_iterations = 0;
  int linear_termination = 0;
  double linear_relative_residual = std::numeric_limits<double>::quiet_NaN();
  NesterovInnerTiming nesterov_inner_timing;
  // options.max_num_iterations 
  while ( true ) { // if costStart + penaltyStart < costEnd + penaltyP
    ++trust_region_attempts;

    //   std::cout << " diagUP " << Ul.diagonal()[0] << " " << Ul.diagonal()[1] << " " << Ul.diagonal()[2] << "\n";
    //   std::cout << " diagVL " << Vl.diagonal()[0] << " " << Vl.diagonal()[1] << " " << Vl.diagonal()[2] << "\n";// TOTALLY OFF after tr_check fails.

    // if not complicated this will lead to total chaos, likely the 
    const bool cumulative_damping = CumulativeDiagonalDampingEnabled()
      || BaeTrustScheduleEnabled();
    const double damping_change = cumulative_damping
      ? 1. / tr_radius : 1. / tr_radius - inv_tr_radius;
    if (diagonal_trust_damping) {
      Ul += damping_change * (cumulative_damping
          ? Diagonal<9>(Ul, cluster_id) : cameraTrustDiagonal);
    } else {
      Ul += damping_change * (CameraTrustDiagonalScale() * trustDiagUP);
      Ul += cameraHessian * damping_change;
    }

    if (inv_tr_radius != 0 && !BaeTrustScheduleEnabled()) {
      Vl = landmarkHessian;
    }
    if (diagonal_trust_damping) {
        Vl += (1. / tr_radius) * (cumulative_damping
          ? Diagonal<3>(Vl) : landmarkTrustDiagonal);
    } else {
      Vl *= 1. + 1. / tr_radius;
      Vl += (1. / tr_radius) * diagVL;
    }
    inv_tr_radius = 1. / tr_radius;
    FloorSymmetricBlocks<9>(Ul, PobaBlockRelativeFloor());
    FloorSymmetricBlocks<3>(Vl, PobaBlockRelativeFloor());
    if (collectTiming) {
      std::ostringstream metric;
      metric << "LANDMARK_TRUST_UPDATED cluster=" << cluster_id
             << " radius=" << tr_radius
             << " trace=" << Vl.diagonal().sum() << "\n";
      EmitLocalSolveMetric(metric.str());
    }

    if (PobaDiagnosticIterations() > 0) {
      double minimum_u_eigenvalue = std::numeric_limits<double>::infinity();
      double minimum_v_eigenvalue = std::numeric_limits<double>::infinity();
      int nonpositive_u_blocks = 0;
      int nonpositive_v_blocks = 0;
      for (int camera = 0; camera < numCameras; ++camera) {
        Eigen::Matrix<double, 9, 9> block;
        for (int row = 0; row < 9; ++row) {
          for (int column = 0; column < 9; ++column) {
            block(row, column) = Ul.coeff(
                9 * camera + row, 9 * camera + column);
          }
        }
        const double eigenvalue = Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix<double, 9, 9>>(block).eigenvalues().minCoeff();
        minimum_u_eigenvalue = std::min(minimum_u_eigenvalue, eigenvalue);
        nonpositive_u_blocks += !(eigenvalue > 0.);
      }
      for (int landmark = 0; landmark < numLandmarks; ++landmark) {
        Eigen::Matrix3d block;
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            block(row, column) = Vl.coeff(
                3 * landmark + row, 3 * landmark + column);
          }
        }
        const double eigenvalue = Eigen::SelfAdjointEigenSolver<
            Eigen::Matrix3d>(block).eigenvalues().minCoeff();
        minimum_v_eigenvalue = std::min(minimum_v_eigenvalue, eigenvalue);
        nonpositive_v_blocks += !(eigenvalue > 0.);
      }

      double spectral_radius = std::numeric_limits<double>::quiet_NaN();
      if (nonpositive_u_blocks == 0 && nonpositive_v_blocks == 0) {
        SparseMatrix<double, RowMajor> u_inverse = Ul;
        SparseMatrix<double, RowMajor> v_inverse = Vl;
        BlockInverse<9>(u_inverse);
        BlockInverse<3>(v_inverse);
        Eigen::VectorXd vector = Eigen::VectorXd::Ones(W.rows());
        vector /= std::sqrt(vector.dot(Ul * vector));
        Eigen::VectorXd landmark_workspace(W.cols());
        Eigen::VectorXd camera_workspace(W.rows());
        Eigen::VectorXd next(W.rows());
        for (int iteration = 0;
             iteration < PobaDiagnosticIterations(); ++iteration) {
          W.TransposeMultiply(vector, landmark_workspace);
          landmark_workspace = v_inverse * landmark_workspace;
          W.Multiply(landmark_workspace, camera_workspace);
            spectral_radius = vector.dot(camera_workspace) /
              vector.dot(Ul * vector);
          next = u_inverse * camera_workspace;
          const double norm = std::sqrt(std::max(0., next.dot(Ul * next)));
          if (!(norm > 0.) || !std::isfinite(norm)) {
            break;
          }
          vector = next / norm;
        }
      }
      std::ostringstream metric;
      metric << "POBA_INVARIANT cluster=" << cluster_id
             << " attempt=" << trust_region_attempts
             << " minimum_u_eigenvalue=" << minimum_u_eigenvalue
             << " minimum_v_eigenvalue=" << minimum_v_eigenvalue
             << " nonpositive_u_blocks=" << nonpositive_u_blocks
             << " nonpositive_v_blocks=" << nonpositive_v_blocks
             << " spectral_radius=" << spectral_radius << "\n";
      EmitLocalSolveMetric(metric.str());
    }
    //   std::cout << " diagUp " << Ul.diagonal()[0] << " " << Ul.diagonal()[1] << " " << Ul.diagonal()[2] << "\n";
    //   std::cout << " diagVL " << Vl.diagonal()[0] << " " << Vl.diagonal()[1] << " " << Vl.diagonal()[2] << "\n";

    //std::cout << " VL " << Vl.diagonal() << "\n";

    const auto nesterovStart = collectTiming ? TimingClock::now() : TimingClock::time_point{};
    std::pair<Eigen::VectorXd, Eigen::VectorXd> step;
    if (local_linear_solver == 1) {
      step = SolveBySchurPCG(
          Ul, Vl, W, bp, bl, proximalGradient,
          factorized_proximal_active ? &tangentToScaled : nullptr,
          &linear_iterations,
          &linear_termination, &linear_relative_residual);
    } else if (local_linear_solver == 5) {
      step = SolveByPobaPowerSeries(
          Ul, Vl, W, bp, bl, proximalGradient, power_iterations,
          nesterov_stop_tolerance, &linear_iterations,
          &linear_relative_residual);
      linear_termination = 1;
    } else {
      step = SolveByGDNesterov(
          Ul, Vl, W, bp, bl, proximalGradient, power_iterations,
          nesterov_min_iterations, nesterov_stop_tolerance,
          nesterov_relative_residual, &linear_iterations,
          &linear_relative_residual,
          collectTiming ? &nesterov_inner_timing : nullptr);
    }
    Eigen::VectorXd delta_p = step.first;
    Eigen::VectorXd delta_l = step.second;
    if (collectTiming) {
      nesterovSeconds += ElapsedSeconds(nesterovStart);
      const double camera_diagonal_norm = std::sqrt(std::max(
          0., delta_p.dot(Diagonal<9>(cameraHessian, cluster_id) * delta_p)));
      const double landmark_diagonal_norm = std::sqrt(std::max(
          0., delta_l.dot(diagVL * delta_l)));
        double translation_step_squared = 0.;
        double rotation_step_squared = 0.;
        double intrinsics_step_squared = 0.;
        for (int camera = 0; camera < numCameras; ++camera) {
        translation_step_squared +=
          delta_p.segment<3>(9 * camera).squaredNorm();
        rotation_step_squared +=
          delta_p.segment<3>(9 * camera + 3).squaredNorm();
        intrinsics_step_squared +=
          delta_p.segment<3>(9 * camera + 6).squaredNorm();
        }
      std::ostringstream metric;
      metric << "FULL_STEP_NORM cluster=" << cluster_id
             << " attempt=" << trust_region_attempts
             << " camera_euclidean=" << delta_p.norm()
             << " landmark_euclidean=" << delta_l.norm()
             << " full_euclidean="
             << std::sqrt(delta_p.squaredNorm() + delta_l.squaredNorm())
             << " translation=" << std::sqrt(translation_step_squared)
             << " rotation=" << std::sqrt(rotation_step_squared)
             << " intrinsics=" << std::sqrt(intrinsics_step_squared)
             << " camera_diagonal=" << camera_diagonal_norm
             << " landmark_diagonal=" << landmark_diagonal_norm
             << " full_diagonal=" << std::hypot(
                    camera_diagonal_norm, landmark_diagonal_norm)
             << " radius=" << tr_radius << "\n";
      EmitLocalSolveMetric(metric.str());
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

    if (!scalar_proximal_prior) {
      proximalStep = ManifoldCameraUpdatesEnabled()
        ? ApplyCameraProximalMetricInTangent(delta_p, tangentToScaled)
          : ApplyCameraProximalMetric(delta_p);
    } else if (ManifoldCameraUpdatesEnabled()) {
      for (int camera = 0; camera < numCameras; ++camera) {
        const Eigen::Map<const Eigen::Matrix<double, 9, 9, Eigen::RowMajor>>
            scaledProximal(&full_stepSize[81 * camera]);
        proximalStep.segment<9>(9 * camera) =
            tangentToScaled[camera].transpose() * scaledProximal
            * tangentToScaled[camera] * delta_p.segment<9>(9 * camera);
      }
    } else {
      blockMult<9>(full_stepSize, delta_p, proximalStep);
    }
    if (DisableLocalProximalTermEnabled()) {
      proximalStep.setZero();
    }
    const double penaltyModelEnd = penaltyStart
        + 2. * delta_p.dot(proximalGradient)
        + delta_p.dot(proximalStep);
    //const double penaltyEnd2 = (Jp * prox_rhs).squaredNorm();

    //SparseMatrix<double, RowMajor> Ul_(9 * numCameras, 9 * numCameras);
    //Ul_.reserve(VectorXi::Constant(9 * numCameras, 9));
    //Ul_ = Jp.transpose() * Jp;
    // const double penaltyEnd6 = prox_rhs.dot( Ul_ * prox_rhs );
    //std::cout << "==Penalties end/end2: " << penaltyEnd << " ?= " << penaltyEnd2 << " == " << penaltyEnd6 << "\n"; // since full_step differs from Jp cna differ  
    
    // Needs to be done due to GetCost.
    const std::vector<double> cameras_before_step = cameras;
    ApplyCameraStep(delta_p);
    for (int id = 0; id < delta_l.size(); ++id) {
        landmarks[id] += delta_l[id];
    }
    Matrix<double, Eigen::Dynamic, 1> actualProximalOffset(9 * numCameras);
    for (int id = 0; id < actualProximalOffset.size(); ++id) {
      actualProximalOffset[id] = cameras[id] - cameras_s[id];
    }
    Matrix<double, Eigen::Dynamic, 1> actualProximalGradient(9 * numCameras);
    if (scalar_proximal_prior) {
      blockMult<9>(full_stepSize, actualProximalOffset,
          actualProximalGradient);
    } else {
      actualProximalGradient =
          ApplyCameraProximalMetric(actualProximalOffset);
    }
    const double penaltyEnd = DisableLocalProximalTermEnabled()
      ? 0. : actualProximalOffset.dot(actualProximalGradient);
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
      costStart - costQuad + penaltyStart - penaltyModelEnd;
    const double tr_check = actual_decrease /
        std::max(0.1, predicted_decrease);
    if (LocalSolveMetricsEnabled()) {
      std::ostringstream metric;
      metric << "LOCAL_TRUST_ATTEMPT cluster=" << cluster_id
             << " attempt=" << trust_region_attempts
             << " radius=" << tr_radius
             << " start=" << costStart + penaltyStart
             << " end=" << costEnd + penaltyEnd
             << " actual_decrease=" << actual_decrease
             << " predicted_decrease=" << predicted_decrease
             << " rho=" << tr_check
             << " linear_iterations=" << linear_iterations
             << " linear_relative_residual=" << linear_relative_residual
             << "\n";
      EmitLocalSolveMetric(metric.str());
    }
    bool accept_step = false;
    if (trust_region_policy == 1) {
      accept_step = actual_decrease > 0.;
      if (BaeTrustScheduleEnabled() && accept_step) {
        if (tr_check > 0.5) {
          tr_radius = std::min(max_trust_region_radius, 2. * tr_radius);
        } else if (!(tr_check > 1e-3)) {
          tr_radius *= trust_region_decreasing_ratio;
        }
        trust_region_decreasing_ratio = 1. / 16.;
      } else if (accept_step) {
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

    if (costStart + penaltyStart > costQuad + penaltyModelEnd)
      WORKER_LOG("==Start Cost < estimated cost: " << costStart + penaltyStart << " < " << costQuad + penaltyModelEnd << "\n");

    if (!accept_step) {
      ++trust_region_rejections;
      WORKER_LOG("Reject Start Cost < end cost: "<< costStart + penaltyStart << " < " << costEnd + penaltyEnd << "\n");
        cameras = cameras_before_step;
      for (int id = 0; id < delta_l.size(); ++id) {
        landmarks[id] -= delta_l[id];
      }
        if (trust_region_attempts >= 20 ||
          tr_radius < MinimumTrustRegionRadius()) {
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
         << " predicted=" << costStart - costQuad + penaltyStart - penaltyModelEnd
         << " rho=" << tr_check
         << " radius=" << tr_radius
         << " linear_solver=" << local_linear_solver
         << " linear_iterations=" << linear_iterations
         << " linear_termination=" << linear_termination
         << " linear_relative_residual=" << linear_relative_residual
         << " trust_policy=" << trust_region_policy
         << " diagonal_floor=" << CameraDiagonalRelativeFloor()
         << " block_sqrt_floor=" << BlockSqrtEigenvalueFloor()
         << " landmark_preconditioner_floor=" << LandmarkPreconditionerFloor()
         << " landmark_preconditioning_disabled="
         << DisableLandmarkPreconditioningEnabled()
         << " camera_trust_diagonal_scale=" << CameraTrustDiagonalScale()
         << " camera_diagonal_maximum_guard=" << CameraDiagonalMaximumGuard()
         << " minimum_trust_region_radius=" << MinimumTrustRegionRadius()
         << " camera_preconditioner_floor="
         << CameraPreconditionerDiagonalFloor()
         << " camera_diagonal_metric_scale=" << CameraDiagonalMetricScale()
         << " camera_block_scale=" << CameraBlockScale()
         << " acceptance_ratio=" << LocalAcceptanceRatio() << "\n";
      EmitLocalSolveMetric(metric.str());
      std::ostringstream timingMetric;
      timingMetric << "LOCAL_SOLVE_TIMING cluster=" << cluster_id
        << " initial=0"
        << " jacobian_evaluate=" << last_jacobian_evaluate_seconds
        << " jacobian_conversion=" << last_jacobian_conversion_seconds
        << " assembly=" << assemblySeconds
        << " nesterov=" << nesterovSeconds
        << " nesterov_calls=" << nesterov_inner_timing.calls
        << " nesterov_iterations=" << nesterov_inner_timing.iterations
        << " nesterov_inverse_landmark_blocks="
        << nesterov_inner_timing.inverse_landmark_blocks
        << " nesterov_inverse_camera_blocks="
        << nesterov_inner_timing.inverse_camera_blocks
        << " nesterov_rhs_landmark_multiply="
        << nesterov_inner_timing.rhs_landmark_multiply
        << " nesterov_rhs_w_multiply="
        << nesterov_inner_timing.rhs_w_multiply
        << " nesterov_rhs_vector=" << nesterov_inner_timing.rhs_vector
        << " nesterov_initialize=" << nesterov_inner_timing.initialize
        << " nesterov_iter_w_transpose="
        << nesterov_inner_timing.iter_w_transpose
        << " nesterov_iter_landmark_multiply="
        << nesterov_inner_timing.iter_landmark_multiply
        << " nesterov_iter_w_multiply="
        << nesterov_inner_timing.iter_w_multiply
        << " nesterov_iter_camera_multiply="
        << nesterov_inner_timing.iter_camera_multiply
        << " nesterov_iter_vector=" << nesterov_inner_timing.iter_vector
        << " nesterov_iter_stop=" << nesterov_inner_timing.iter_stop
        << " nesterov_final_w_transpose="
        << nesterov_inner_timing.final_w_transpose
        << " nesterov_final_landmark_multiply="
        << nesterov_inner_timing.final_landmark_multiply
        << " nesterov_inner_total=" << nesterov_inner_timing.inner_total
        << " nesterov_iterative_edge_visits="
        << nesterov_inner_timing.iterative_edge_visits
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
  last_linear_iterations = linear_iterations;
  last_linear_relative_residual = linear_relative_residual;
  if (landmark_refinement_steps > 0) {
    RefineLandmarksWithFixedCameras(landmark_refinement_steps);
  }
  if (SharedFixedInteriorTrialEnabled()) {
    RunSharedFixedInteriorTrial();
  }
  if (!scalar_proximal_prior &&
      (metric_diagnostic_iterations > 0 || proximal_defect_diagnostic)) {
    const NormalEquations final_normal_equations = GetNormalEquations();
    EstimateProximalDefect(
        final_normal_equations,
        full_stepSize,
        cameras,
        cameras_s,
        metric_diagnostic);
    EstimateInteriorDefect(
      final_normal_equations,
      camera_proximal_multipliers,
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

bool RunSharedFixedInteriorTrial() {
#ifdef __unweighted_system__
  return false;
#else
  THROW_IF(!DirectTangentNormalEquationsEnabled());
  const std::vector<double> cameras_before = cameras;
  const std::vector<double> landmarks_before = landmarks;
  const double cost_before = 2. * GetCost();
  const NormalEquations normal_equations = GetBatchedNormalEquations(true);

  SparseMatrix<double, RowMajor> camera_inverse =
      normal_equations.camera_hessian;
  SparseMatrix<double, RowMajor> landmark_inverse =
      normal_equations.landmark_hessian;
  FloorSymmetricBlocks<9>(camera_inverse, 1e-16);
  FloorSymmetricBlocks<3>(landmark_inverse, 1e-16);
  BlockInverse<9>(camera_inverse);
  BlockInverse<3>(landmark_inverse);
  Eigen::VectorXd camera_direction =
      -camera_inverse * normal_equations.camera_gradient;
  const Eigen::VectorXd landmark_direction =
      -landmark_inverse * normal_equations.landmark_gradient;
  int unique_cameras = 0;
  int shared_cameras = 0;
  for (int camera = 0; camera < numCameras; ++camera) {
    if (camera_proximal_multipliers[camera] > 0.) {
      camera_direction.segment<9>(9 * camera).setZero();
      ++shared_cameras;
    } else {
      ++unique_cameras;
    }
  }

  bool accepted = false;
  int accepted_backtrack = -1;
  double accepted_cost = cost_before;
  double step_length = 1.;
  if (camera_direction.allFinite() && landmark_direction.allFinite() &&
      camera_direction.squaredNorm() + landmark_direction.squaredNorm() > 0.) {
        for (int backtrack = 0;
          backtrack < SharedFixedInteriorTrialMaximumBacktracks();
          ++backtrack) {
      cameras = cameras_before;
      landmarks = landmarks_before;
      ApplyCameraStep(step_length * camera_direction);
      for (int camera = 0; camera < numCameras; ++camera) {
        if (camera_proximal_multipliers[camera] > 0.) {
          std::copy_n(
              cameras_before.begin() + 9 * camera, 9,
              cameras.begin() + 9 * camera);
        }
      }
      for (int index = 0; index < landmark_direction.size(); ++index) {
        landmarks[index] += step_length * landmark_direction[index];
      }
      const double cost_after = 2. * GetCost();
      if (std::isfinite(cost_after) && cost_after < cost_before) {
        accepted = true;
        accepted_backtrack = backtrack;
        accepted_cost = cost_after;
        cost = cost_after;
        break;
      }
      step_length *= 0.5;
    }
  }
  if (!accepted) {
    cameras = cameras_before;
    landmarks = landmarks_before;
    cost = cost_before;
  }
  double shared_camera_maximum_change = 0.;
  for (int camera = 0; camera < numCameras; ++camera) {
    if (camera_proximal_multipliers[camera] > 0.) {
      for (int parameter = 0; parameter < 9; ++parameter) {
        shared_camera_maximum_change = std::max(
            shared_camera_maximum_change,
            std::abs(cameras[9 * camera + parameter]
                     - cameras_before[9 * camera + parameter]));
      }
    }
  }
  THROW_IF(shared_camera_maximum_change != 0.);
  if (LocalSolveMetricsEnabled()) {
    std::ostringstream metric;
    metric << "SHARED_FIXED_INTERIOR_TRIAL cluster=" << cluster_id
           << " accepted=" << accepted
           << " unique_cameras=" << unique_cameras
           << " shared_cameras=" << shared_cameras
           << " shared_camera_maximum_change="
           << shared_camera_maximum_change
           << " backtracks=" << accepted_backtrack
           << " step_length=" << (accepted ? step_length : 0.)
           << " before=" << cost_before
           << " after=" << accepted_cost << "\n";
    EmitLocalSolveMetric(metric.str());
  }
  return accepted;
#endif
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
            full_stepSize[81*b + id] = std::max(
              CameraPreconditionerDiagonalFloor(),
              full_stepSize[81*b + id]);
        }
      }
      best_landmarks = landmarks; // !
      //best_poses = cameras; // ? 
      //return; // 1st iteration only preconditioning. do not solve! needs other stuff do be dones below.
    }

    // Allow to scale JtJ as well?
    const double scale = CameraBlockScale();
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
  NormalEquations GetBatchedNormalEquations(bool tangent_coordinates = false) {
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
    std::vector<Eigen::Matrix<double, 9, 9>> tangent_to_scaled;

    for (int observation = 0; observation < numResiduals; ++observation) {
      const int cameraId = cam_obs[observation];
      const int landmarkId = lm_obs[observation];
      const int cameraOffset = 9 * cameraId;
      const int landmarkOffset = 3 * landmarkId;
      const int transformOffset = 81 * cameraId;
      ObservationJet camera[9];
      ObservationJet landmark[3];
      Eigen::Matrix<double, 9, 9> physical_jacobian =
          Eigen::Matrix<double, 9, 9>::Zero();
      if (tangent_coordinates) {
        const Eigen::Map<const Eigen::Matrix<double, 9, 1>> physical_camera(
            &weighted_cameras[cameraOffset]);
        physical_jacobian = PhysicalTangentJacobian(
            physical_camera, GetCameraUpdateMode());
      }
      for (int row = 0; row < 9; ++row) {
        camera[row].a = weighted_cameras[cameraOffset + row];
        camera[row].v.setZero();
        for (int col = 0; col < 9; ++col) {
          if (tangent_coordinates) {
            camera[row].v[col] = physical_jacobian(row, col);
          } else {
            camera[row].v[col] =
                cameraTransform[transformOffset + 9 * row + col]
                * unorm[cameraOffset + col];
          }
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
    if (objective_model == 0 && huber_delta == 0. && BatchedEvaluationEnabled()) {
      return GetBatchedNormalEquations(false);
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
    Eigen::VectorXd robust_sqrt_weight = Eigen::VectorXd::Ones(numResiduals);
    if (huber_delta > 0.) {
      for (int observation = 0; observation < numResiduals; ++observation) {
        const double x = normal_equation_residuals[2 * observation];
        const double y = normal_equation_residuals[2 * observation + 1];
        const double norm = std::hypot(x, y);
        if (norm > huber_delta) {
          robust_sqrt_weight[observation] = std::sqrt(huber_delta / norm);
        }
        result.residual.segment<2>(2 * observation) *=
            robust_sqrt_weight[observation];
      }
    }

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
        const double sqrt_weight = robust_sqrt_weight[observation];
        const double residual = sqrt_weight * normal_equation_residuals[row];
        for (int i = 0; i < 9; ++i) {
          const double cameraValue =
              sqrt_weight * normal_equation_jacobian.values[begin + i];
          result.camera_gradient[9 * cam_id + i] += cameraValue * residual;
          for (int j = 0; j < 9; ++j) {
            normal_equation_camera_blocks[81 * cam_id + 9 * i + j] +=
                cameraValue * sqrt_weight
                * normal_equation_jacobian.values[begin + j];
          }
          for (int j = 0; j < 3; ++j) {
            edgeValues[3 * i + j] +=
                cameraValue * sqrt_weight
                * normal_equation_jacobian.values[begin + 9 + j];
          }
        }
        for (int i = 0; i < 3; ++i) {
            const double landmarkValue =
              sqrt_weight * normal_equation_jacobian.values[begin + 9 + i];
          result.landmark_gradient[3 * lm_id + i] += landmarkValue * residual;
          for (int j = 0; j < 3; ++j) {
            normal_equation_landmark_blocks[9 * lm_id + 3 * i + j] +=
                landmarkValue * sqrt_weight
                * normal_equation_jacobian.values[begin + 9 + j];
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
  std::vector<double> camera_proximal_multipliers;
  int local_solve_sequence = 0;
  int numLandmarks = 0;
  int numResiduals = 0;
  const double init_be = 1e-4;
  double current_be = init_be;
  double block_curvature_multiplier = 0.;
  int metric_diagnostic_iterations = 0;
  int outer_iteration = -1;
  int oracle_kind = 0;
  bool collect_camera_diagonal_metrics = false;
  bool frozen_block_metric_initialized = false;
  bool proximal_defect_diagnostic = false;
  bool diagonal_trust_damping = false;
  bool nesterov_relative_residual = false;
  int landmark_refinement_steps = 0;
  int nesterov_max_iterations = 100;
  int nesterov_min_iterations = 1;
  double nesterov_stop_tolerance = 1e-2;
  MetricDiagnostic metric_diagnostic;
  bool scalar_proximal_prior = false;
  bool factorized_coupled_schur_metric = false;
  double proximal_rho = 1.;
  bool split_camera_penalty = false;
  double proximal_rho_intrinsics = 1.;
  int local_linear_solver = 0;
  int trust_region_policy = 0;
  int objective_model = 0;
  double huber_delta = 0.;
  int residual_dimension = 2;
  bool persistent_trust_region = false;
  bool persistent_trust_region_active = false;
  bool ceres_local_solver = false;
  int local_iterations = 1;
  double start_be = init_be;
  const double init_trust_region_radius = InitialTrustRegionRadius();
  double tr_radius = init_trust_region_radius; // 1e4 is ceres standard. -> Init()
  double last_tr_radius = init_trust_region_radius;
  const double max_trust_region_radius = MaximumTrustRegionRadius();
  double startCost;
  double cost;
  double best_cost;
  double last_jacobian_evaluate_seconds = 0.;
  double last_jacobian_conversion_seconds = 0.;
  int last_linear_iterations = 0;
  double last_linear_relative_residual =
      std::numeric_limits<double>::quiet_NaN();
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
  std::vector<double> consensus_stepSize; // optional unfloored projection metric.
  SparseMatrix<double, RowMajor> camera_proximal_metric;
  bool factorized_proximal_active = false;
  SparseMatrix<double, RowMajor> factorized_proximal_diagonal;
  SparseMatrix<double, RowMajor> factorized_proximal_landmark_inverse;
  BlockEdgeMatrix factorized_proximal_cross;
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
    const bool collect_request_timing = LocalSolveMetricsEnabled();
    const auto request_parse_start = collect_request_timing
      ? TimingClock::now() : TimingClock::time_point{};
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
    const double request_parse_seconds = collect_request_timing
      ? ElapsedSeconds(request_parse_start) : 0.;
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

      const auto update_data_start = collect_request_timing
        ? TimingClock::now() : TimingClock::time_point{};
      program.UpdateData(update); // update is local, we nned to fill data in main thread.
      const double update_data_seconds = collect_request_timing
        ? ElapsedSeconds(update_data_start) : 0.;

      // Define a Lambda Expression
      auto update_lambda = [&push_socket, &cluster_to_program,
                &single_node_consensus_reducer,
                &mtx](int cluster_id, bool omit_landmarks,
                  bool return_consensus_rhs,
                  bool single_node_consensus,
                  double consensus_relaxation,
                  std::uint64_t run_id, std::uint64_t phase_id,
                  double request_parse_seconds,
                  double update_data_seconds,
                  TimingClock::time_point launch_start) {
        const bool collect_completion_timing = LocalSolveMetricsEnabled();
        const double launch_wait_seconds = collect_completion_timing
          ? ElapsedSeconds(launch_start) : 0.;
        CeresProgram &program = cluster_to_program[cluster_id];
        // std::cout << cluster_id << " Update "<< "\n";
  if (program.UsesCentralizedLeftSe3Solver()) {
    program.SolveCentralizedLeftSe3(
        program.UsesCentralizedProximalLeftSe3Solver());
  } else if (program.UsesCeresLocalSolver()) {
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
        const auto completion_start = collect_completion_timing
          ? TimingClock::now() : TimingClock::time_point{};
        double contribution_seconds = 0.;
        double reduction_seconds = 0.;
        std::shared_ptr<const SingleNodeConsensusResult> consensus_result;
        if (single_node_consensus) {
          const auto contribution_start = collect_completion_timing
            ? TimingClock::now() : TimingClock::time_point{};
          SingleNodeConsensusContribution contribution =
            program.BuildSingleNodeConsensusContribution();
          if (collect_completion_timing) {
            contribution_seconds = ElapsedSeconds(contribution_start);
          }
          const auto reduction_start = collect_completion_timing
            ? TimingClock::now() : TimingClock::time_point{};
          consensus_result = single_node_consensus_reducer.Submit(
            run_id, phase_id, cluster_id, program.ClusterCount(),
            consensus_relaxation, std::move(contribution));
          if (collect_completion_timing) {
            reduction_seconds = ElapsedSeconds(reduction_start);
          }
        }
        const auto reply_pack_start = collect_completion_timing
          ? TimingClock::now() : TimingClock::time_point{};
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
        const double reply_pack_seconds = collect_completion_timing
          ? ElapsedSeconds(reply_pack_start) : 0.;
        const double cost = return_proto.cost();
        WORKER_LOG(cluster_id << ". Cost from update: " << cost << "\n");
        // std::cout << "Cost from update " << cost <<"\n";
        //  SerializeToArray saves memory and time?
        const auto serialize_start = collect_completion_timing
          ? TimingClock::now() : TimingClock::time_point{};
        size_t bytes = return_proto.ByteSizeLong();
        zmq::message_t reply(bytes);
        return_proto.SerializeToArray(reply.data(), bytes);
        const double serialize_seconds = collect_completion_timing
          ? ElapsedSeconds(serialize_start) : 0.;
        const auto mutex_start = collect_completion_timing
          ? TimingClock::now() : TimingClock::time_point{};
        double mutex_wait_seconds = 0.;
        double send_seconds = 0.;
        {
          std::unique_lock<std::mutex> lock(mtx);
          if (collect_completion_timing) {
            mutex_wait_seconds = ElapsedSeconds(mutex_start);
          }
          const auto send_start = collect_completion_timing
            ? TimingClock::now() : TimingClock::time_point{};
          push_socket.send(reply, zmq::send_flags::none);
          if (collect_completion_timing) {
            send_seconds = ElapsedSeconds(send_start);
          }
        }
        if (collect_completion_timing) {
          std::ostringstream timing;
          timing << "SOLVE_COMPLETION_TIMING cluster=" << cluster_id
            << " run=" << run_id << " phase=" << phase_id
            << " single_node=" << single_node_consensus
            << " bytes=" << bytes
            << " request_parse=" << request_parse_seconds
            << " update_data=" << update_data_seconds
            << " launch_wait=" << launch_wait_seconds
            << " contribution=" << contribution_seconds
            << " reduction=" << reduction_seconds
            << " reply_pack=" << reply_pack_seconds
            << " serialize=" << serialize_seconds
            << " mutex_wait=" << mutex_wait_seconds
            << " send=" << send_seconds
            << " total=" << ElapsedSeconds(completion_start) << "\n";
          EmitLocalSolveMetric(timing.str());
        }
        // std::cout << cluster_id << ". Update send" << std::endl;
      };

      // std::thread update_thread(update_lambda, std::ref(program),
      // std::cref(update));
      const auto launch_start = collect_request_timing
        ? TimingClock::now() : TimingClock::time_point{};
      std::thread update_thread(update_lambda, cluster_id,
        update.omit_landmarks(), update.return_consensus_rhs(),
        update.single_node_consensus(), update.consensus_relaxation(),
        update.run_id(),
        update.phase_id(), request_parse_seconds, update_data_seconds,
        launch_start);//, keep_cameras_fixed);
      update_thread.detach();
      /// update_thread.join();

      break;
    }

    case request_proto::OptionsCase::kSchurSystem: {
      const schur_system_proto schur_request = request_p.schur_system();
      const int cluster_id = schur_request.cluster_id();
      THROW_IF(cluster_to_program.find(cluster_id) == cluster_to_program.end());
      auto schur_lambda = [&cluster_to_program, &push_socket, &mtx](
          int cluster_id, schur_system_proto request) {
        return_schur_system_proto response =
            cluster_to_program[cluster_id].BuildSchurSystem(request);
        const size_t bytes = response.ByteSizeLong();
        zmq::message_t reply(bytes);
        response.SerializeToArray(reply.data(), bytes);
        std::lock_guard<std::mutex> lock(mtx);
        push_socket.send(reply, zmq::send_flags::none);
      };
      std::thread schur_thread(
          schur_lambda, cluster_id, std::move(schur_request));
      schur_thread.detach();
      break;
    }

    case request_proto::OptionsCase::kCameraStep: {
      const camera_step_proto camera_step_request = request_p.camera_step();
      const int cluster_id = camera_step_request.cluster_id();
      THROW_IF(cluster_to_program.find(cluster_id) == cluster_to_program.end());
      auto camera_step_lambda = [&cluster_to_program, &push_socket, &mtx](
          int cluster_id, camera_step_proto request) {
        return_cluster_proto response =
            cluster_to_program[cluster_id].ApplyExternalCameraStep(request);
        response.set_run_id(request.run_id());
        response.set_phase_id(request.phase_id());
        const size_t bytes = response.ByteSizeLong();
        zmq::message_t reply(bytes);
        response.SerializeToArray(reply.data(), bytes);
        std::lock_guard<std::mutex> lock(mtx);
        push_socket.send(reply, zmq::send_flags::none);
      };
      std::thread camera_step_thread(
          camera_step_lambda, cluster_id, std::move(camera_step_request));
      camera_step_thread.detach();
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
        const double l2_cost = 2 * program.GetCost(revert_lms, false);
        WORKER_LOG(cluster_id << ". Cost from cost: " << cost << "\n");
        return_proto.set_cost(cost);
        return_proto.set_precise_cost(cost);
        return_proto.set_precise_l2_cost(l2_cost);
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
        case landmark_state_proto::RESTORE_BEST:
          program.RestoreBestOutputLandmarkState();
          break;
        case landmark_state_proto::MATERIALIZE_ACCEPTED:
          program.MaterializeAcceptedLandmarkState(reply_proto);
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