#include <ceres/ceres.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <examples/ceres/camera.h>
#include <examples/ceres/math.h>
#include <examples/ceres/reprojection.h>

struct Measurement {
  int camera_index;
  int point_index;
  Ceres::Vector2 position;
  double sqrt_weight;
};

struct DabaBalProblem {
  std::vector<Measurement> measurements;
  std::vector<Ceres::Matrix<3, 5>> cameras;
  std::vector<Ceres::Vector3> points;
};

DabaBalProblem ReadDabaBalProblem(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open BAL problem: " + path);
  }

  int camera_count;
  int point_count;
  int observation_count;
  input >> camera_count >> point_count >> observation_count;
  DabaBalProblem problem;
  problem.measurements.resize(observation_count);
  for (Measurement& measurement : problem.measurements) {
    input >> measurement.camera_index >> measurement.point_index
          >> measurement.position[0] >> measurement.position[1];
    measurement.sqrt_weight = 1.0;
  }

  problem.cameras.resize(camera_count);
  std::vector<Ceres::Vector3> raw_intrinsics(camera_count);
  for (int camera_index = 0; camera_index < camera_count; ++camera_index) {
    Ceres::Vector3 angle_axis;
    Ceres::Vector3 translation;
    input >> angle_axis[0] >> angle_axis[1] >> angle_axis[2]
          >> translation[0] >> translation[1] >> translation[2]
          >> raw_intrinsics[camera_index][0]
          >> raw_intrinsics[camera_index][1]
          >> raw_intrinsics[camera_index][2];
    Ceres::Matrix3 rotation;
    Ceres::math::SO3::exp(angle_axis, rotation);
    problem.cameras[camera_index].leftCols<3>() = rotation.transpose();
    problem.cameras[camera_index].col(3) = rotation.transpose() * translation;
  }

  problem.points.resize(point_count);
  for (Ceres::Vector3& point : problem.points) {
    input >> point[0] >> point[1] >> point[2];
    point = -point;
  }
  if (!input) {
    throw std::runtime_error("invalid or truncated BAL problem: " + path);
  }

  for (Measurement& measurement : problem.measurements) {
    const double focal = raw_intrinsics[measurement.camera_index][0];
    measurement.position /= -focal;
    measurement.sqrt_weight =
        focal * std::sqrt(measurement.position.squaredNorm() + 1.0);
  }
  for (int camera_index = 0; camera_index < camera_count; ++camera_index) {
    const double focal = raw_intrinsics[camera_index][0];
    Ceres::Vector3 intrinsics = raw_intrinsics[camera_index];
    intrinsics[1] *= focal * focal;
    intrinsics[2] *= focal * focal * focal * focal;
    intrinsics[0] /= focal;
    problem.cameras[camera_index].col(4) = intrinsics;
  }
  return problem;
}

struct ObjectiveTraceCallback : public ceres::IterationCallback {
  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override {
    iterations.push_back(summary.iteration);
    cumulative_seconds.push_back(summary.cumulative_time_in_seconds);
    costs.push_back(summary.cost);
    return ceres::SOLVER_CONTINUE;
  }

  std::vector<int> iterations;
  std::vector<double> cumulative_seconds;
  std::vector<double> costs;
};

void WriteState(const std::string& path, const DabaBalProblem& problem) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot write DABA Ceres state: " + path);
  }
  output << problem.cameras.size() << ' ' << problem.points.size() << '\n';
  output << std::setprecision(17);
  for (const auto& camera : problem.cameras) {
    for (int column = 0; column < camera.cols(); ++column) {
      for (int row = 0; row < camera.rows(); ++row) {
        output << camera(row, column) << '\n';
      }
    }
  }
  for (const auto& point : problem.points) {
    output << point[0] << '\n' << point[1] << '\n' << point[2] << '\n';
  }
}

int main(int argc, char** argv) {
  if (argc < 3 || argc > 6) {
    std::cerr << "usage: daba_ceres_bal_runner INPUT_BAL STATE_OUTPUT "
                 "[LOSS=trivial] [THREADS=64] [ITERATIONS=40]\n";
    return 2;
  }
  const std::string loss_name = argc >= 4 ? argv[3] : "trivial";
  const int thread_count = argc >= 5 ? std::stoi(argv[4]) : 64;
  const int iteration_count = argc >= 6 ? std::stoi(argv[5]) : 40;
  if (thread_count <= 0 || iteration_count <= 0 ||
      (loss_name != "trivial" && loss_name != "huber")) {
    throw std::invalid_argument(
        "threads and iterations must be positive; loss must be trivial or huber");
  }

  const auto setup_started = std::chrono::steady_clock::now();
  DabaBalProblem bal = ReadDabaBalProblem(argv[1]);
  ceres::Problem problem;
  auto* manifold = new Ceres::Camera();
  ceres::LossFunction* loss =
      loss_name == "huber" ? new ceres::HuberLoss(32.0) : nullptr;
  for (const Measurement& measurement : bal.measurements) {
    problem.AddResidualBlock(
        new Ceres::ReprojectionError(
            measurement.position, measurement.sqrt_weight),
        loss,
        bal.cameras[measurement.camera_index].data(),
        bal.points[measurement.point_index].data());
  }
  for (auto& camera : bal.cameras) {
    problem.SetManifold(camera.data(), manifold);
  }
  const auto solve_started = std::chrono::steady_clock::now();

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = iteration_count;
  options.num_threads = thread_count;
  options.parameter_tolerance = 0;
  options.function_tolerance = 0;
  options.gradient_tolerance = 0;
  options.max_solver_time_in_seconds = 14400;
  options.minimizer_progress_to_stdout = false;
  ObjectiveTraceCallback trace;
  options.callbacks.push_back(&trace);
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  const auto solve_finished = std::chrono::steady_clock::now();
  WriteState(argv[2], bal);
  const auto finished = std::chrono::steady_clock::now();

  const double observation_count = bal.measurements.size();
  const std::chrono::duration<double> setup_seconds = solve_started - setup_started;
  const std::chrono::duration<double> solve_seconds = solve_finished - solve_started;
  const std::chrono::duration<double> overall_seconds = finished - setup_started;
  std::cout << std::setprecision(17)
            << "{\"solver\":\"daba-ceres\","
            << "\"input\":\"" << argv[1] << "\","
            << "\"stateFile\":\"" << argv[2] << "\","
            << "\"loss\":\"" << loss_name << "\","
            << "\"threads\":" << thread_count << ','
            << "\"maxIterations\":" << iteration_count << ','
            << "\"observationCount\":" << bal.measurements.size() << ','
            << "\"initialCeresCost\":" << summary.initial_cost << ','
            << "\"finalCeresCost\":" << summary.final_cost << ','
            << "\"initialMeanHalfWeightedRayCost\":"
            << summary.initial_cost / observation_count << ','
            << "\"finalMeanHalfWeightedRayCost\":"
            << summary.final_cost / observation_count << ','
            << "\"initialReportedMetric\":"
            << summary.initial_cost / observation_count << ','
            << "\"finalReportedMetric\":"
            << summary.final_cost / observation_count << ','
            << "\"setupSeconds\":" << setup_seconds.count() << ','
            << "\"solveSeconds\":" << solve_seconds.count() << ','
            << "\"overallSeconds\":" << overall_seconds.count() << ','
            << "\"objectiveTrajectory\":[";
  double best_cost = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < trace.iterations.size(); ++index) {
    best_cost = std::min(best_cost, trace.costs[index]);
    if (index > 0) {
      std::cout << ',';
    }
    std::cout << "{\"iteration\":" << trace.iterations[index]
              << ",\"cumulativeSeconds\":" << trace.cumulative_seconds[index]
              << ",\"ceresCost\":" << trace.costs[index]
              << ",\"bestReportedMetric\":"
              << best_cost / observation_count << '}';
  }
  std::cout << "],\"termination\":\""
            << ceres::TerminationTypeToString(summary.termination_type)
            << "\"}\n";
  return summary.IsSolutionUsable() ? 0 : 1;
}
