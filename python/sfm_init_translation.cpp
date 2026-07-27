#include <ceres/ceres.h>

#include <cmath>
#include <cstdint>
#include <random>

namespace {

struct ChordFunctor {
  ChordFunctor(const double* direction, double weight)
      : direction_(direction), weight_(weight) {}

  template <typename T>
  bool operator()(const T* const source, const T* const destination,
                  T* residual) const {
    const T dx = destination[0] - source[0];
    const T dy = destination[1] - source[1];
    const T dz = destination[2] - source[2];
    const T norm = sqrt(dx * dx + dy * dy + dz * dz);
    residual[0] = T(weight_) * (dx / norm - T(direction_[0]));
    residual[1] = T(weight_) * (dy / norm - T(direction_[1]));
    residual[2] = T(weight_) * (dz / norm - T(direction_[2]));
    return true;
  }

  const double* direction_;
  double weight_;
};

}  // namespace

extern "C" int sfm_init_solve_translations(
    const std::int64_t* edges, const double* directions, const double* weights,
    std::int64_t edge_count, std::int64_t node_count, std::uint64_t seed,
    int maximum_iterations, double function_tolerance,
    double parameter_tolerance, double* positions, double* final_cost,
    int* completed_iterations) {
  if (edges == nullptr || directions == nullptr || weights == nullptr ||
      positions == nullptr || final_cost == nullptr ||
      completed_iterations == nullptr || edge_count <= 0 || node_count <= 1 ||
      maximum_iterations <= 0) {
    return 1;
  }

  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (std::int64_t index = 0; index < 3 * node_count; ++index) {
    positions[index] = distribution(generator);
  }

  ceres::Problem problem;
  for (std::int64_t node = 0; node < node_count; ++node) {
    problem.AddParameterBlock(positions + 3 * node, 3);
  }
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const std::int64_t source = edges[2 * edge];
    const std::int64_t destination = edges[2 * edge + 1];
    if (source < 0 || destination < 0 || source >= node_count ||
        destination >= node_count || source == destination ||
        !std::isfinite(weights[edge])) {
      return 2;
    }
    auto* cost = new ceres::AutoDiffCostFunction<ChordFunctor, 3, 3, 3>(
        new ChordFunctor(directions + 3 * edge, weights[edge]));
    problem.AddResidualBlock(cost, nullptr, positions + 3 * source,
                             positions + 3 * destination);
  }

  ceres::Solver::Options options;
  options.num_threads = 16;
  options.max_num_iterations = maximum_iterations;
  options.function_tolerance = function_tolerance;
  options.parameter_tolerance = parameter_tolerance;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  *final_cost = summary.final_cost;
  *completed_iterations = static_cast<int>(summary.iterations.size());
  return summary.IsSolutionUsable() ? 0 : 3;
}
