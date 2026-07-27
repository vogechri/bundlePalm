#include <ceres/ceres.h>
#include <ceres/rotation.h>

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
  if (argc < 3 || argc > 6) {
    std::cerr << "usage: ceres_bal_runner INPUT_BAL STATE_OUTPUT "
                 "[ITERATIONS=40] [THREADS=16] [HUBER_DELTA=0]\n";
    return 2;
  }
  const int max_iterations = argc >= 4 ? std::stoi(argv[3]) : 40;
  const int thread_count = argc >= 5 ? std::stoi(argv[4]) : 16;
  const double huber_delta = argc >= 6 ? std::stod(argv[5]) : 0.0;
  if (max_iterations <= 0 || thread_count <= 0 || huber_delta < 0) {
    throw std::invalid_argument("iterations/threads must be positive and Huber nonnegative");
  }

  const auto setup_started = std::chrono::steady_clock::now();
  BalProblem bal = ReadBalProblem(argv[1]);
  ceres::Problem problem;
  for (int index = 0; index < bal.observation_count; ++index) {
    ceres::LossFunction* loss =
        huber_delta > 0 ? new ceres::HuberLoss(huber_delta) : nullptr;
    problem.AddResidualBlock(
        ReprojectionError::Create(
            bal.observations[2 * index], bal.observations[2 * index + 1]),
        loss,
        &bal.cameras[9 * bal.camera_indices[index]],
        &bal.points[3 * bal.point_indices[index]]);
  }
  const auto solve_started = std::chrono::steady_clock::now();

  ceres::Solver::Options options;
  options.max_num_iterations = max_iterations;
  options.num_threads = thread_count;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.minimizer_progress_to_stdout = false;
  options.function_tolerance = 1e-12;
  options.gradient_tolerance = 1e-12;
  options.parameter_tolerance = 1e-12;
  ObjectiveTraceCallback trace;
  options.callbacks.push_back(&trace);
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  const auto solve_finished = std::chrono::steady_clock::now();
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
