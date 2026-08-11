#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct ObjectiveTraceCallback : public ceres::IterationCallback {
  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override {
    iterations.push_back(summary.iteration);
    solve_seconds.push_back(summary.cumulative_time_in_seconds);
    sum_squared_errors.push_back(2.0 * summary.cost);
    return ceres::SOLVER_CONTINUE;
  }

  std::vector<int> iterations;
  std::vector<double> solve_seconds;
  std::vector<double> sum_squared_errors;
};

struct ReprojectionError {
  ReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera, const T* const point,
                  T* residuals) const {
    T camera_point[3];
    ceres::AngleAxisRotatePoint(camera, point, camera_point);
    camera_point[0] += camera[3];
    camera_point[1] += camera[4];
    camera_point[2] += camera[5];
    const T projected_x = -camera_point[0] / camera_point[2];
    const T projected_y = -camera_point[1] / camera_point[2];
    const T radius_squared =
        projected_x * projected_x + projected_y * projected_y;
    const T distortion =
        T(1.0) + radius_squared * (camera[7] + camera[8] * radius_squared);
    residuals[0] = camera[6] * distortion * projected_x - observed_x;
    residuals[1] = camera[6] * distortion * projected_y - observed_y;
    return true;
  }

  static ceres::CostFunction* Create(double observed_x, double observed_y) {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 9, 3>(
        new ReprojectionError(observed_x, observed_y));
  }

  double observed_x;
  double observed_y;
};

template <typename T>
void QuaternionProductXyzw(const T* lhs, const T* rhs, T* product) {
  product[0] = lhs[3] * rhs[0] + lhs[0] * rhs[3] +
               lhs[1] * rhs[2] - lhs[2] * rhs[1];
  product[1] = lhs[3] * rhs[1] - lhs[0] * rhs[2] +
               lhs[1] * rhs[3] + lhs[2] * rhs[0];
  product[2] = lhs[3] * rhs[2] + lhs[0] * rhs[1] -
               lhs[1] * rhs[0] + lhs[2] * rhs[3];
  product[3] = lhs[3] * rhs[3] - lhs[0] * rhs[0] -
               lhs[1] * rhs[1] - lhs[2] * rhs[2];
}

template <typename T>
void RotatePointXyzw(const T* quaternion, const T* point, T* rotated) {
  const T twice_x = T(2.0) *
      (quaternion[1] * point[2] - quaternion[2] * point[1]);
  const T twice_y = T(2.0) *
      (quaternion[2] * point[0] - quaternion[0] * point[2]);
  const T twice_z = T(2.0) *
      (quaternion[0] * point[1] - quaternion[1] * point[0]);
  rotated[0] = point[0] + quaternion[3] * twice_x +
               quaternion[1] * twice_z - quaternion[2] * twice_y;
  rotated[1] = point[1] + quaternion[3] * twice_y +
               quaternion[2] * twice_x - quaternion[0] * twice_z;
  rotated[2] = point[2] + quaternion[3] * twice_z +
               quaternion[0] * twice_y - quaternion[1] * twice_x;
}

void AngleAxisToQuaternionXyzw(const double* angle_axis, double* quaternion) {
  double quaternion_wxyz[4];
  ceres::AngleAxisToQuaternion(angle_axis, quaternion_wxyz);
  quaternion[0] = quaternion_wxyz[1];
  quaternion[1] = quaternion_wxyz[2];
  quaternion[2] = quaternion_wxyz[3];
  quaternion[3] = quaternion_wxyz[0];
}

void QuaternionXyzwToAngleAxis(const double* quaternion, double* angle_axis) {
  const double quaternion_wxyz[4] = {
      quaternion[3], quaternion[0], quaternion[1], quaternion[2]};
  ceres::QuaternionToAngleAxis(quaternion_wxyz, angle_axis);
}

class LeftSe3CameraManifold final : public ceres::Manifold {
 public:
  int AmbientSize() const override { return 10; }
  int TangentSize() const override { return 9; }

  bool Plus(const double* x, const double* delta,
            double* x_plus_delta) const override {
    const double* translation_delta = delta;
    const double* rotation_delta = delta + 3;
    const double theta_squared =
        rotation_delta[0] * rotation_delta[0] +
        rotation_delta[1] * rotation_delta[1] +
        rotation_delta[2] * rotation_delta[2];
    const double theta = std::sqrt(theta_squared);
    double quaternion_delta[4];
    double coefficient_a;
    double coefficient_b;
    if (theta > 1e-10) {
      const double half_theta = 0.5 * theta;
      const double quaternion_scale = std::sin(half_theta) / theta;
      quaternion_delta[0] = quaternion_scale * rotation_delta[0];
      quaternion_delta[1] = quaternion_scale * rotation_delta[1];
      quaternion_delta[2] = quaternion_scale * rotation_delta[2];
      quaternion_delta[3] = std::cos(half_theta);
      coefficient_a = (1.0 - std::cos(theta)) / theta_squared;
      coefficient_b = (theta - std::sin(theta)) / (theta_squared * theta);
    } else {
      quaternion_delta[0] = 0.5 * rotation_delta[0];
      quaternion_delta[1] = 0.5 * rotation_delta[1];
      quaternion_delta[2] = 0.5 * rotation_delta[2];
      quaternion_delta[3] = 1.0 - theta_squared / 8.0;
      coefficient_a = 0.5 - theta_squared / 24.0;
      coefficient_b = 1.0 / 6.0 - theta_squared / 120.0;
    }
    const Eigen::Map<const Eigen::Vector3d> rotation(rotation_delta);
    const Eigen::Map<const Eigen::Vector3d> translation(translation_delta);
    const Eigen::Vector3d algebra_cross = rotation.cross(translation);
    const Eigen::Vector3d exponential_translation =
        translation + coefficient_a * algebra_cross +
        coefficient_b * rotation.cross(algebra_cross);
    double rotated_translation[3];
    RotatePointXyzw(quaternion_delta, x, rotated_translation);
    for (int index = 0; index < 3; ++index) {
      x_plus_delta[index] = rotated_translation[index] +
                            exponential_translation[index];
    }
    QuaternionProductXyzw(quaternion_delta, x + 3, x_plus_delta + 3);
    for (int index = 0; index < 3; ++index) {
      x_plus_delta[7 + index] = x[7 + index] + delta[6 + index];
    }
    return true;
  }

  bool PlusJacobian(const double* x, double* jacobian) const override {
    Eigen::Map<Eigen::Matrix<double, 10, 9, Eigen::RowMajor>> result(jacobian);
    result.setZero();
    result.block<3, 3>(0, 0).setIdentity();
    const Eigen::Map<const Eigen::Vector3d> translation(x);
    Eigen::Matrix3d translation_skew;
    translation_skew << 0.0, -translation.z(), translation.y(),
                        translation.z(), 0.0, -translation.x(),
                        -translation.y(), translation.x(), 0.0;
    result.block<3, 3>(0, 3) = -translation_skew;
    const Eigen::Map<const Eigen::Vector3d> quaternion_vector(x + 3);
    Eigen::Matrix3d quaternion_skew;
    quaternion_skew << 0.0, -quaternion_vector.z(), quaternion_vector.y(),
                       quaternion_vector.z(), 0.0, -quaternion_vector.x(),
                       -quaternion_vector.y(), quaternion_vector.x(), 0.0;
    result.block<3, 3>(3, 3) =
        0.5 * (x[6] * Eigen::Matrix3d::Identity() - quaternion_skew);
    result.block<1, 3>(6, 3) = -0.5 * quaternion_vector.transpose();
    result.block<3, 3>(7, 6).setIdentity();
    return true;
  }

  bool Minus(const double* y, const double* x,
             double* y_minus_x) const override {
    const double inverse_x[4] = {-x[3], -x[4], -x[5], x[6]};
    double relative_quaternion[4];
    QuaternionProductXyzw(y + 3, inverse_x, relative_quaternion);
    if (relative_quaternion[3] < 0.0) {
      for (double& value : relative_quaternion) value = -value;
    }
    QuaternionXyzwToAngleAxis(relative_quaternion, y_minus_x + 3);
    double rotated_x_translation[3];
    RotatePointXyzw(relative_quaternion, x, rotated_x_translation);
    Eigen::Vector3d relative_translation;
    for (int index = 0; index < 3; ++index) {
      relative_translation[index] = y[index] - rotated_x_translation[index];
    }
    const Eigen::Map<const Eigen::Vector3d> rotation(y_minus_x + 3);
    const double theta_squared = rotation.squaredNorm();
    Eigen::Matrix3d rotation_skew;
    rotation_skew << 0.0, -rotation.z(), rotation.y(),
                     rotation.z(), 0.0, -rotation.x(),
                     -rotation.y(), rotation.x(), 0.0;
    double coefficient = 1.0 / 12.0;
    if (theta_squared > 1e-20) {
      const double theta = std::sqrt(theta_squared);
      coefficient = 1.0 / theta_squared -
          (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
    }
    Eigen::Map<Eigen::Vector3d> tangent_translation(y_minus_x);
    tangent_translation =
      (Eigen::Matrix3d::Identity() - 0.5 * rotation_skew +
       coefficient * rotation_skew * rotation_skew) * relative_translation;
    for (int index = 0; index < 3; ++index) {
      y_minus_x[6 + index] = y[7 + index] - x[7 + index];
    }
    return true;
  }

  bool MinusJacobian(const double* x, double* jacobian) const override {
    double plus_jacobian[90];
    PlusJacobian(x, plus_jacobian);
    const Eigen::Map<const Eigen::Matrix<double, 10, 9, Eigen::RowMajor>> plus(
        plus_jacobian);
    Eigen::Map<Eigen::Matrix<double, 9, 10, Eigen::RowMajor>> minus(jacobian);
    minus = (plus.transpose() * plus).ldlt().solve(plus.transpose());
    return true;
  }
};

class RightSe3CameraManifold final : public ceres::Manifold {
 public:
  int AmbientSize() const override { return 10; }
  int TangentSize() const override { return 9; }

  bool Plus(const double* x, const double* delta,
            double* x_plus_delta) const override {
    const double* translation_delta = delta;
    const double* rotation_delta = delta + 3;
    const double theta_squared =
        rotation_delta[0] * rotation_delta[0] +
        rotation_delta[1] * rotation_delta[1] +
        rotation_delta[2] * rotation_delta[2];
    const double theta = std::sqrt(theta_squared);
    double quaternion_delta[4];
    double coefficient_a;
    double coefficient_b;
    if (theta > 1e-10) {
      const double half_theta = 0.5 * theta;
      const double quaternion_scale = std::sin(half_theta) / theta;
      quaternion_delta[0] = quaternion_scale * rotation_delta[0];
      quaternion_delta[1] = quaternion_scale * rotation_delta[1];
      quaternion_delta[2] = quaternion_scale * rotation_delta[2];
      quaternion_delta[3] = std::cos(half_theta);
      coefficient_a = (1.0 - std::cos(theta)) / theta_squared;
      coefficient_b = (theta - std::sin(theta)) / (theta_squared * theta);
    } else {
      quaternion_delta[0] = 0.5 * rotation_delta[0];
      quaternion_delta[1] = 0.5 * rotation_delta[1];
      quaternion_delta[2] = 0.5 * rotation_delta[2];
      quaternion_delta[3] = 1.0 - theta_squared / 8.0;
      coefficient_a = 0.5 - theta_squared / 24.0;
      coefficient_b = 1.0 / 6.0 - theta_squared / 120.0;
    }
    const Eigen::Map<const Eigen::Vector3d> rotation(rotation_delta);
    const Eigen::Map<const Eigen::Vector3d> translation(translation_delta);
    const Eigen::Vector3d algebra_cross = rotation.cross(translation);
    const Eigen::Vector3d exponential_translation =
        translation + coefficient_a * algebra_cross +
        coefficient_b * rotation.cross(algebra_cross);
    double rotated_translation[3];
    RotatePointXyzw(x + 3, exponential_translation.data(), rotated_translation);
    for (int index = 0; index < 3; ++index) {
      x_plus_delta[index] = x[index] + rotated_translation[index];
    }
    QuaternionProductXyzw(x + 3, quaternion_delta, x_plus_delta + 3);
    for (int index = 0; index < 3; ++index) {
      x_plus_delta[7 + index] = x[7 + index] + delta[6 + index];
    }
    return true;
  }

  bool PlusJacobian(const double* x, double* jacobian) const override {
    Eigen::Map<Eigen::Matrix<double, 10, 9, Eigen::RowMajor>> result(jacobian);
    result.setZero();
    const double basis[3][3] = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    for (int column = 0; column < 3; ++column) {
      double rotated[3];
      RotatePointXyzw(x + 3, basis[column], rotated);
      for (int row = 0; row < 3; ++row) result(row, column) = rotated[row];
    }
    const Eigen::Map<const Eigen::Vector3d> quaternion_vector(x + 3);
    Eigen::Matrix3d quaternion_skew;
    quaternion_skew << 0.0, -quaternion_vector.z(), quaternion_vector.y(),
                       quaternion_vector.z(), 0.0, -quaternion_vector.x(),
                       -quaternion_vector.y(), quaternion_vector.x(), 0.0;
    result.block<3, 3>(3, 3) =
        0.5 * (x[6] * Eigen::Matrix3d::Identity() + quaternion_skew);
    result.block<1, 3>(6, 3) = -0.5 * quaternion_vector.transpose();
    result.block<3, 3>(7, 6).setIdentity();
    return true;
  }

  bool Minus(const double* y, const double* x,
             double* y_minus_x) const override {
    const double inverse_x[4] = {-x[3], -x[4], -x[5], x[6]};
    double relative_quaternion[4];
    QuaternionProductXyzw(inverse_x, y + 3, relative_quaternion);
    if (relative_quaternion[3] < 0.0) {
      for (double& value : relative_quaternion) value = -value;
    }
    QuaternionXyzwToAngleAxis(relative_quaternion, y_minus_x + 3);
    const double translation_difference[3] = {
        y[0] - x[0], y[1] - x[1], y[2] - x[2]};
    Eigen::Vector3d relative_translation;
    RotatePointXyzw(inverse_x, translation_difference,
                    relative_translation.data());
    const Eigen::Map<const Eigen::Vector3d> rotation(y_minus_x + 3);
    const double theta_squared = rotation.squaredNorm();
    Eigen::Matrix3d rotation_skew;
    rotation_skew << 0.0, -rotation.z(), rotation.y(),
                     rotation.z(), 0.0, -rotation.x(),
                     -rotation.y(), rotation.x(), 0.0;
    double coefficient = 1.0 / 12.0;
    if (theta_squared > 1e-20) {
      const double theta = std::sqrt(theta_squared);
      coefficient = 1.0 / theta_squared -
          (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
    }
    Eigen::Map<Eigen::Vector3d> tangent_translation(y_minus_x);
    tangent_translation =
        (Eigen::Matrix3d::Identity() - 0.5 * rotation_skew +
         coefficient * rotation_skew * rotation_skew) * relative_translation;
    for (int index = 0; index < 3; ++index) {
      y_minus_x[6 + index] = y[7 + index] - x[7 + index];
    }
    return true;
  }

  bool MinusJacobian(const double* x, double* jacobian) const override {
    double plus_jacobian[90];
    PlusJacobian(x, plus_jacobian);
    const Eigen::Map<const Eigen::Matrix<double, 10, 9, Eigen::RowMajor>> plus(
        plus_jacobian);
    Eigen::Map<Eigen::Matrix<double, 9, 10, Eigen::RowMajor>> minus(jacobian);
    minus = (plus.transpose() * plus).ldlt().solve(plus.transpose());
    return true;
  }
};

struct QuaternionReprojectionError {
  QuaternionReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera, const T* const point,
                  T* residuals) const {
    T camera_point[3];
    RotatePointXyzw(camera + 3, point, camera_point);
    camera_point[0] += camera[0];
    camera_point[1] += camera[1];
    camera_point[2] += camera[2];
    const T projected_x = -camera_point[0] / camera_point[2];
    const T projected_y = -camera_point[1] / camera_point[2];
    const T radius_squared =
        projected_x * projected_x + projected_y * projected_y;
    const T distortion = T(1.0) + radius_squared *
        (camera[8] + camera[9] * radius_squared);
    residuals[0] = camera[7] * distortion * projected_x - observed_x;
    residuals[1] = camera[7] * distortion * projected_y - observed_y;
    return true;
  }

  static ceres::CostFunction* Create(double observed_x, double observed_y) {
    return new ceres::AutoDiffCostFunction<QuaternionReprojectionError, 2, 10, 3>(
        new QuaternionReprojectionError(observed_x, observed_y));
  }

  double observed_x;
  double observed_y;
};

struct BalProblem {
  int camera_count;
  int point_count;
  int observation_count;
  std::vector<int> camera_indices;
  std::vector<int> point_indices;
  std::vector<double> observations;
  std::vector<double> cameras;
  std::vector<double> points;
};

BalProblem ReadBalProblem(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open BAL problem: " + path);
  }
  BalProblem problem;
  input >> problem.camera_count >> problem.point_count >> problem.observation_count;
  problem.camera_indices.resize(problem.observation_count);
  problem.point_indices.resize(problem.observation_count);
  problem.observations.resize(2 * problem.observation_count);
  for (int index = 0; index < problem.observation_count; ++index) {
    input >> problem.camera_indices[index] >> problem.point_indices[index]
          >> problem.observations[2 * index]
          >> problem.observations[2 * index + 1];
  }
  problem.cameras.resize(9 * problem.camera_count);
  problem.points.resize(3 * problem.point_count);
  for (double& value : problem.cameras) {
    input >> value;
  }
  for (double& value : problem.points) {
    input >> value;
  }
  if (!input) {
    throw std::runtime_error("invalid or truncated BAL problem: " + path);
  }
  return problem;
}

void WriteState(const std::string& path, const BalProblem& problem) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot write Ceres state: " + path);
  }
  output << problem.camera_count << ' ' << problem.point_count << '\n';
  output << std::setprecision(17);
  for (double value : problem.cameras) {
    output << value << '\n';
  }
  for (double value : problem.points) {
    output << value << '\n';
  }
}

int main(int argc, char** argv) {
  if (argc < 3 || argc > 10) {
    std::cerr << "usage: ceres_bal_runner INPUT_BAL STATE_OUTPUT "
                 "[ITERATIONS=40] [THREADS=16] [HUBER_DELTA=0] "
                 "[LINEAR_SOLVER=iterative_schur] [ETA=0.1] "
                 "[INITIAL_TRUST_RADIUS=1e4] [CAMERA_MODE=angle_axis]\n";
    return 2;
  }
  const int max_iterations = argc >= 4 ? std::stoi(argv[3]) : 40;
  const int thread_count = argc >= 5 ? std::stoi(argv[4]) : 16;
  const double huber_delta = argc >= 6 ? std::stod(argv[5]) : 0.0;
  const std::string linear_solver = argc >= 7 ? argv[6] : "iterative_schur";
  const double eta = argc >= 8 ? std::stod(argv[7]) : 1e-1;
  const double initial_trust_radius = argc >= 9 ? std::stod(argv[8]) : 1e4;
  const std::string camera_mode = argc >= 10 ? argv[9] : "angle_axis";
  if (max_iterations <= 0 || thread_count <= 0 || huber_delta < 0) {
    throw std::invalid_argument("iterations/threads must be positive and Huber nonnegative");
  }

  const auto setup_started = std::chrono::steady_clock::now();
  BalProblem bal = ReadBalProblem(argv[1]);
  ceres::Problem problem;
  std::vector<double> quaternion_cameras;
  if (camera_mode == "se3_left" || camera_mode == "se3_right") {
    quaternion_cameras.resize(10 * bal.camera_count);
    for (int camera_index = 0; camera_index < bal.camera_count; ++camera_index) {
      const double* input_camera = &bal.cameras[9 * camera_index];
      double* output_camera = &quaternion_cameras[10 * camera_index];
      output_camera[0] = input_camera[3];
      output_camera[1] = input_camera[4];
      output_camera[2] = input_camera[5];
      AngleAxisToQuaternionXyzw(input_camera, output_camera + 3);
      output_camera[7] = input_camera[6];
      output_camera[8] = input_camera[7];
      output_camera[9] = input_camera[8];
    }
  } else if (camera_mode != "angle_axis") {
    throw std::invalid_argument(
        "camera mode must be angle_axis, se3_left, or se3_right");
  }
  for (int index = 0; index < bal.observation_count; ++index) {
    ceres::LossFunction* loss =
        huber_delta > 0 ? new ceres::HuberLoss(huber_delta) : nullptr;
    if (camera_mode == "angle_axis") {
      problem.AddResidualBlock(
          ReprojectionError::Create(
              bal.observations[2 * index], bal.observations[2 * index + 1]),
          loss, &bal.cameras[9 * bal.camera_indices[index]],
          &bal.points[3 * bal.point_indices[index]]);
    } else {
      problem.AddResidualBlock(
          QuaternionReprojectionError::Create(
              bal.observations[2 * index], bal.observations[2 * index + 1]),
          loss, &quaternion_cameras[10 * bal.camera_indices[index]],
          &bal.points[3 * bal.point_indices[index]]);
    }
  }
  if (camera_mode != "angle_axis") {
    for (int camera_index = 0; camera_index < bal.camera_count; ++camera_index) {
      ceres::Manifold* manifold = camera_mode == "se3_left"
          ? static_cast<ceres::Manifold*>(new LeftSe3CameraManifold())
          : static_cast<ceres::Manifold*>(new RightSe3CameraManifold());
      problem.SetManifold(&quaternion_cameras[10 * camera_index], manifold);
    }
  }
  const auto solve_started = std::chrono::steady_clock::now();

  ceres::Solver::Options options;
  options.max_num_iterations = max_iterations;
  options.num_threads = thread_count;
  if (linear_solver == "iterative_schur") {
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
  } else if (linear_solver == "sparse_schur") {
    options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {
    throw std::invalid_argument(
        "linear solver must be iterative_schur or sparse_schur");
  }
  options.minimizer_progress_to_stdout = false;
  options.function_tolerance = 1e-12;
  options.gradient_tolerance = 1e-12;
  options.parameter_tolerance = 1e-12;
  options.eta = eta;
  options.initial_trust_region_radius = initial_trust_radius;
  ObjectiveTraceCallback trace;
  options.callbacks.push_back(&trace);
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  const auto solve_finished = std::chrono::steady_clock::now();
  if (camera_mode != "angle_axis") {
    for (int camera_index = 0; camera_index < bal.camera_count; ++camera_index) {
      const double* input_camera = &quaternion_cameras[10 * camera_index];
      double* output_camera = &bal.cameras[9 * camera_index];
      QuaternionXyzwToAngleAxis(input_camera + 3, output_camera);
      output_camera[3] = input_camera[0];
      output_camera[4] = input_camera[1];
      output_camera[5] = input_camera[2];
      output_camera[6] = input_camera[7];
      output_camera[7] = input_camera[8];
      output_camera[8] = input_camera[9];
    }
  }
  WriteState(argv[2], bal);
  const auto finished = std::chrono::steady_clock::now();

  const std::chrono::duration<double> setup_seconds = solve_started - setup_started;
  const std::chrono::duration<double> solve_seconds = solve_finished - solve_started;
  const std::chrono::duration<double> overall_seconds = finished - setup_started;
  std::cout << std::setprecision(17)
            << "{\"solver\":\"ceres-iterative-schur\","
            << "\"input\":\"" << argv[1] << "\","
            << "\"stateFile\":\"" << argv[2] << "\","
            << "\"threads\":" << thread_count << ','
            << "\"linearSolver\":\"" << linear_solver << "\","
            << "\"eta\":" << eta << ','
            << "\"initialTrustRegionRadius\":" << initial_trust_radius << ','
            << "\"cameraMode\":\"" << camera_mode << "\","
            << "\"maxIterations\":" << max_iterations << ','
            << "\"successfulSteps\":" << summary.num_successful_steps << ','
            << "\"unsuccessfulSteps\":" << summary.num_unsuccessful_steps << ','
            << "\"initialCeresCost\":" << summary.initial_cost << ','
            << "\"finalCeresCost\":" << summary.final_cost << ','
            << "\"setupSeconds\":" << setup_seconds.count() << ','
            << "\"solveSeconds\":" << solve_seconds.count() << ','
            << "\"overallSeconds\":" << overall_seconds.count() << ','
            << "\"objectiveTrajectory\":[";
  for (std::size_t index = 0; index < trace.iterations.size(); ++index) {
    if (index > 0) {
      std::cout << ',';
    }
    std::cout << "{\"iteration\":" << trace.iterations[index]
              << ",\"solveSeconds\":" << trace.solve_seconds[index]
              << ",\"overallSeconds\":"
              << setup_seconds.count() + trace.solve_seconds[index]
              << ",\"sumSquaredError\":"
              << trace.sum_squared_errors[index] << '}';
  }
  std::cout << "],"
            << "\"termination\":\"" << ceres::TerminationTypeToString(summary.termination_type)
            << "\"}\n";
  return summary.IsSolutionUsable() ? 0 : 1;
}
