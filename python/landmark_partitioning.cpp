#include "landmark_partitioning.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace bundle_palm {
namespace {

struct Objective {
  std::int64_t weak_camera_penalty = 0;
  std::int64_t weak_camera_count = 0;
  std::int64_t severe_weak_camera_count = 0;
  std::int64_t residual_imbalance = 0;
  std::int64_t copied_camera_count = 0;
  std::int64_t maximum_camera_count = 0;
};

struct State {
  State(const BipartiteCameraPointGraph& graph,
        const LandmarkPartitioningOptions& options)
      : camera_cluster_degree(graph.camera_count * options.cluster_count, 0),
        camera_cluster_count(graph.camera_count, 0),
        cameras_per_cluster(options.cluster_count, 0),
        residuals_per_cluster(options.cluster_count, 0),
        objective{} {}

  std::vector<int> camera_cluster_degree;
  std::vector<int> camera_cluster_count;
  std::vector<int> cameras_per_cluster;
  std::vector<int> residuals_per_cluster;
  Objective objective;
};

class ProgressTrace {
 public:
  ProgressTrace()
  : enabled_(TraceEnabled()),
        start_(std::chrono::steady_clock::now()),
        last_report_(start_) {}

  void Begin(const char* phase, int pass) {
    phase_ = phase;
    pass_ = pass;
    work_units_ = 0;
  }

  void Tick(const Objective& objective, std::int64_t work_units = 1) {
    work_units_ += work_units;
    if (!enabled_ || (++ticks_since_check_ & 1023) != 0) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report_ < std::chrono::seconds(5)) {
      return;
    }
    Report(objective, now);
  }

  void Finish(const Objective& objective) {
    if (enabled_) {
      Report(objective, std::chrono::steady_clock::now());
    }
  }

  std::int64_t work_units() const { return work_units_; }

 private:
  static bool TraceEnabled() {
    const char* setting = std::getenv("BUNDLE_PALM_PARTITION_TRACE");
    return setting == nullptr || std::strcmp(setting, "0") != 0;
  }

  void Report(const Objective& objective,
              std::chrono::steady_clock::time_point now) {
    const double elapsed =
        std::chrono::duration<double>(now - start_).count();
    std::cerr << "landmark partition progress: " << elapsed << " s, phase "
              << phase_;
    if (pass_ >= 0) {
      std::cerr << ", pass " << pass_;
    }
    std::cerr << ", work " << work_units_
              << ", weak " << objective.weak_camera_count
              << ", severe weak " << objective.severe_weak_camera_count
              << ", weak penalty " << objective.weak_camera_penalty
              << ", max cameras " << objective.maximum_camera_count
              << ", camera copies " << objective.copied_camera_count
              << ", imbalance " << objective.residual_imbalance << '\n';
    last_report_ = now;
  }

  bool enabled_;
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point last_report_;
  const char* phase_ = "initialization";
  int pass_ = -1;
  std::int64_t work_units_ = 0;
  std::uint64_t ticks_since_check_ = 0;
};

struct MoveRecord {
  int landmark;
  int source_cluster;
  int target_cluster;
  Objective objective_before;
};

constexpr bool IsBetter(const Objective& left, const Objective& right) {
  if (left.severe_weak_camera_count != right.severe_weak_camera_count) {
    return left.severe_weak_camera_count < right.severe_weak_camera_count;
  }
  if (left.weak_camera_count != right.weak_camera_count) {
    return left.weak_camera_count < right.weak_camera_count;
  }
  if (left.weak_camera_penalty != right.weak_camera_penalty) {
    return left.weak_camera_penalty < right.weak_camera_penalty;
  }
  if (left.maximum_camera_count != right.maximum_camera_count) {
    return left.maximum_camera_count < right.maximum_camera_count;
  }
  if (left.copied_camera_count != right.copied_camera_count) {
    return left.copied_camera_count < right.copied_camera_count;
  }
  return left.residual_imbalance < right.residual_imbalance;
}

bool IsEquivalent(const Objective& left, const Objective& right) {
  return !IsBetter(left, right) && !IsBetter(right, left);
}

constexpr std::int64_t WeakCameraPenalty(int degree, int degree_limit) {
  if (degree <= 0 || degree >= degree_limit) {
    return 0;
  }
  const std::int64_t deficit = degree_limit - degree;
  return deficit * deficit * deficit;
}

static_assert(WeakCameraPenalty(0, 20) == 0);
static_assert(WeakCameraPenalty(1, 20) == 6859);
static_assert(WeakCameraPenalty(10, 20) == 1000);
static_assert(WeakCameraPenalty(19, 20) == 1);
static_assert(WeakCameraPenalty(20, 20) == 0);
static_assert(IsBetter(Objective{2000, 1, 0, 0, 0, 0},
                       Objective{1000, 2, 0, 0, 0, 0}));
static_assert(IsBetter(Objective{1, 1, 0, 0, 0, 0},
                       Objective{1000, 1, 0, 0, 0, 0}));
static_assert(IsBetter(Objective{1000, 2, 0, 0, 0, 0},
                       Objective{1, 1, 1, 0, 0, 0}));
static_assert(IsBetter(Objective{0, 0, 0, 1000000, 9, 0},
                       Objective{0, 0, 0, 0, 10, 0}));
static_assert(IsBetter(Objective{0, 0, 0, 0, 1000, 9},
                       Objective{0, 0, 0, 0, 0, 10}));
static_assert(IsBetter(Objective{0, 0, 0, 9, 10, 0},
                       Objective{0, 0, 0, 10, 10, 0}));

void ReplaceWeakDegree(int old_degree,
                       int new_degree,
                       int degree_limit,
                       Objective& objective) {
  if (old_degree > 0 && old_degree < degree_limit) {
    --objective.weak_camera_count;
    if (old_degree < degree_limit / 2) {
      --objective.severe_weak_camera_count;
    }
    objective.weak_camera_penalty -=
        WeakCameraPenalty(old_degree, degree_limit);
  }
  if (new_degree > 0 && new_degree < degree_limit) {
    ++objective.weak_camera_count;
    if (new_degree < degree_limit / 2) {
      ++objective.severe_weak_camera_count;
    }
    objective.weak_camera_penalty +=
        WeakCameraPenalty(new_degree, degree_limit);
  }
}

int ResidualViolationForCluster(int residuals,
                                int minimum_residuals,
                                int maximum_residuals) {
  return std::max(0, minimum_residuals - residuals) +
         std::max(0, residuals - maximum_residuals);
}

int ResidualViolation(const std::vector<int>& residuals_per_cluster,
                      int minimum_residuals,
                      int maximum_residuals) {
  int violation = 0;
  for (int residuals : residuals_per_cluster) {
    violation += ResidualViolationForCluster(
        residuals, minimum_residuals, maximum_residuals);
  }
  return violation;
}

std::int64_t Square(int value) {
  return static_cast<std::int64_t>(value) * value;
}

int MaximumCameraCount(const State& state,
                       int first_cluster,
                       int first_delta,
                       int second_cluster,
                       int second_delta) {
  int maximum = 0;
  for (int cluster = 0;
       cluster < static_cast<int>(state.cameras_per_cluster.size());
       ++cluster) {
    int count = state.cameras_per_cluster[cluster];
    if (cluster == first_cluster) {
      count += first_delta;
    }
    if (cluster == second_cluster) {
      count += second_delta;
    }
    maximum = std::max(maximum, count);
  }
  return maximum;
}

int PointMultiplicity(const BipartiteCameraPointGraph& graph, int landmark) {
  return graph.point_multiplicity.empty()
             ? 1
             : graph.point_multiplicity[landmark];
}

int LandmarkWeight(const BipartiteCameraPointGraph& graph, int landmark) {
  return PointMultiplicity(graph, landmark) *
         static_cast<int>(graph.cameras_from_point[landmark].size());
}

std::pair<int, int> ResidualBounds(
    const BipartiteCameraPointGraph& graph,
    const LandmarkPartitioningOptions& options) {
  int residual_count = 0;
  for (int landmark = 0; landmark < graph.point_count; ++landmark) {
    residual_count += LandmarkWeight(graph, landmark);
  }
  const double target = static_cast<double>(residual_count) /
                        static_cast<double>(options.cluster_count);
  int largest_landmark = 0;
  for (int landmark = 0; landmark < graph.point_count; ++landmark) {
    largest_landmark = std::max(
        largest_landmark, LandmarkWeight(graph, landmark));
  }
  const int minimum_residuals = static_cast<int>(
      std::floor(target * (1.0 - options.residual_balance_slack)));
  const int maximum_residuals = std::max(
      largest_landmark,
      static_cast<int>(
          std::ceil(target * (1.0 + options.residual_balance_slack))));
  return {minimum_residuals, maximum_residuals};
}

Objective ScoreMove(const BipartiteCameraPointGraph& graph,
                    const LandmarkPartitioningOptions& options,
                    const State& state,
                    int landmark,
                    int source_cluster,
                    int target_cluster) {
  Objective candidate = state.objective;
  const int multiplicity = PointMultiplicity(graph, landmark);
  const int weight = LandmarkWeight(graph, landmark);
  int source_camera_delta = 0;
  int target_camera_delta = 0;
  if (source_cluster >= 0) {
    const int source_residuals = state.residuals_per_cluster[source_cluster];
    candidate.residual_imbalance +=
        Square(source_residuals - weight) - Square(source_residuals);
  }
  const int target_residuals = state.residuals_per_cluster[target_cluster];
  candidate.residual_imbalance +=
      Square(target_residuals + weight) - Square(target_residuals);
  for (int camera : graph.cameras_from_point[landmark]) {
    const int occurrences_before = state.camera_cluster_count[camera];
    int occurrences_after = occurrences_before;
    if (source_cluster >= 0) {
      const int source_offset = camera * options.cluster_count + source_cluster;
      const int source_degree = state.camera_cluster_degree[source_offset];
      ReplaceWeakDegree(source_degree, source_degree - multiplicity,
                        options.weak_camera_degree_limit, candidate);
      if (source_degree == multiplicity) {
        --occurrences_after;
        --source_camera_delta;
      }
    }
    const int target_offset = camera * options.cluster_count + target_cluster;
    const int target_degree = state.camera_cluster_degree[target_offset];
    ReplaceWeakDegree(target_degree, target_degree + multiplicity,
                      options.weak_camera_degree_limit, candidate);
    if (target_degree == 0) {
      ++occurrences_after;
      ++target_camera_delta;
    }
    candidate.copied_camera_count +=
        std::max(0, occurrences_after - 1) -
        std::max(0, occurrences_before - 1);
  }
        if (options.optimize_max_camera_count) {
          candidate.maximum_camera_count = MaximumCameraCount(
          state, source_cluster, source_camera_delta,
          target_cluster, target_camera_delta);
        }
  return candidate;
}

void ApplyMoveToState(const BipartiteCameraPointGraph& graph,
                      const LandmarkPartitioningOptions& options,
                      int landmark,
                      int source_cluster,
                      int target_cluster,
                      const Objective& objective,
                      State& state) {
  const int multiplicity = PointMultiplicity(graph, landmark);
  const int weight = LandmarkWeight(graph, landmark);
  for (int camera : graph.cameras_from_point[landmark]) {
    if (source_cluster >= 0) {
      const int source_offset = camera * options.cluster_count + source_cluster;
      state.camera_cluster_degree[source_offset] -= multiplicity;
      if (state.camera_cluster_degree[source_offset] == 0) {
        --state.camera_cluster_count[camera];
        --state.cameras_per_cluster[source_cluster];
      }
    }
    const int target_offset = camera * options.cluster_count + target_cluster;
    if (state.camera_cluster_degree[target_offset] == 0) {
      ++state.camera_cluster_count[camera];
      ++state.cameras_per_cluster[target_cluster];
    }
    state.camera_cluster_degree[target_offset] += multiplicity;
  }
  if (source_cluster >= 0) {
    state.residuals_per_cluster[source_cluster] -= weight;
  }
  state.residuals_per_cluster[target_cluster] += weight;
  state.objective = objective;
}

void ApplyMove(const BipartiteCameraPointGraph& graph,
               const LandmarkPartitioningOptions& options,
               int landmark,
               int source_cluster,
               int target_cluster,
               const Objective& objective,
               State& state,
               std::vector<int>& landmark_to_cluster) {
  ApplyMoveToState(graph, options, landmark, source_cluster, target_cluster,
                   objective, state);
  landmark_to_cluster[landmark] = target_cluster;
}

void ApplyRecordedMove(const BipartiteCameraPointGraph& graph,
                       const LandmarkPartitioningOptions& options,
                       int landmark,
                       int source_cluster,
                       int target_cluster,
                       const Objective& objective,
                       State& state,
                       std::vector<int>& landmark_to_cluster,
                         std::vector<MoveRecord>& move_log,
                         std::vector<std::vector<int>>* landmarks_per_cluster =
                           nullptr) {
  move_log.push_back(
      {landmark, source_cluster, target_cluster, state.objective});
  ApplyMove(graph, options, landmark, source_cluster, target_cluster,
            objective, state, landmark_to_cluster);
              if (landmarks_per_cluster != nullptr) {
              auto& source_landmarks = (*landmarks_per_cluster)[source_cluster];
              source_landmarks.erase(std::lower_bound(
                source_landmarks.begin(), source_landmarks.end(), landmark));
              auto& target_landmarks = (*landmarks_per_cluster)[target_cluster];
              target_landmarks.insert(std::lower_bound(
                target_landmarks.begin(), target_landmarks.end(), landmark),
                landmark);
              }
}

void RollbackMoves(const BipartiteCameraPointGraph& graph,
                   const LandmarkPartitioningOptions& options,
                   std::size_t first_move,
                   State& state,
                   std::vector<int>& landmark_to_cluster,
                     std::vector<MoveRecord>& move_log,
                     std::vector<std::vector<int>>* landmarks_per_cluster =
                       nullptr) {
  for (std::size_t index = move_log.size(); index > first_move; --index) {
    const MoveRecord& move = move_log[index - 1];
    ApplyMove(graph, options, move.landmark, move.target_cluster,
              move.source_cluster, move.objective_before, state,
              landmark_to_cluster);
            if (landmarks_per_cluster != nullptr) {
              auto& target_landmarks =
                (*landmarks_per_cluster)[move.target_cluster];
              target_landmarks.erase(std::lower_bound(
                target_landmarks.begin(), target_landmarks.end(), move.landmark));
              auto& source_landmarks =
                (*landmarks_per_cluster)[move.source_cluster];
              source_landmarks.insert(std::lower_bound(
                source_landmarks.begin(), source_landmarks.end(), move.landmark),
                move.landmark);
            }
  }
  move_log.resize(first_move);
}

bool TryMoveLandmarkOutOfWeakCamera(
    const BipartiteCameraPointGraph& graph,
    const LandmarkPartitioningOptions& options,
    int camera,
    int landmark,
    int source_cluster,
    int minimum_residuals,
    int maximum_residuals,
    State& state,
    std::vector<int>& landmark_to_cluster,
    std::vector<std::vector<int>>& landmarks_per_cluster,
    std::vector<MoveRecord>& move_log,
    ProgressTrace& progress) {

  Objective best_first_objective;
  Objective best_objective;
  int best_target = -1;
  int best_outbound = -1;
  bool found = false;
  for (int target = 0; target < options.cluster_count; ++target) {
    if (target == source_cluster) {
      continue;
    }

    Objective first_objective = ScoreMove(
      graph, options, state, landmark, source_cluster, target);
    Objective final_objective = first_objective;

    int outbound_for_target = -1;
    const int weight = LandmarkWeight(graph, landmark);
    if (state.residuals_per_cluster[source_cluster] - weight <
        minimum_residuals ||
      state.residuals_per_cluster[target] + weight > maximum_residuals) {
      const Objective original_objective = state.objective;
      ApplyMoveToState(graph, options, landmark, source_cluster, target,
                       first_objective, state);
      int best_outbound = -1;
      int tried = 0;
      Objective best_exchange_objective;
      for (int outbound : landmarks_per_cluster[target]) {
        progress.Tick(state.objective);
        if (options.repair_restart_interval != 1 &&
            options.max_swap_candidates_per_landmark > 0 &&
            tried++ >= options.max_swap_candidates_per_landmark) {
          break;
        }
        if (std::binary_search(graph.cameras_from_point[outbound].begin(),
                               graph.cameras_from_point[outbound].end(),
                               camera)) {
          continue;
        }
        const int outbound_weight = LandmarkWeight(graph, outbound);
        const int source_after =
          state.residuals_per_cluster[source_cluster] +
            outbound_weight;
        const int target_after =
          state.residuals_per_cluster[target] - outbound_weight;
        if (source_after < minimum_residuals ||
            source_after > maximum_residuals ||
            target_after < minimum_residuals ||
            target_after > maximum_residuals) {
          continue;
        }
        Objective exchange_objective = ScoreMove(
          graph, options, state, outbound, target, source_cluster);
        if (best_outbound < 0 ||
            IsBetter(exchange_objective, best_exchange_objective)) {
          best_outbound = outbound;
          best_exchange_objective = std::move(exchange_objective);
        }
      }
      ApplyMoveToState(graph, options, landmark, target, source_cluster,
                       original_objective, state);
      if (best_outbound < 0) {
        continue;
      }
      outbound_for_target = best_outbound;
      final_objective = best_exchange_objective;
    }

    if (!found || IsBetter(final_objective, best_objective)) {
      found = true;
      best_first_objective = first_objective;
      best_objective = final_objective;
      best_target = target;
      best_outbound = outbound_for_target;
    }
  }
  if (!found) {
    return false;
  }
  ApplyRecordedMove(graph, options, landmark, source_cluster, best_target,
                    best_first_objective, state, landmark_to_cluster,
                    move_log, &landmarks_per_cluster);
  if (best_outbound >= 0) {
    ApplyRecordedMove(graph, options, best_outbound, best_target,
                      source_cluster, best_objective, state,
                      landmark_to_cluster, move_log,
                      &landmarks_per_cluster);
  }
  return true;
}

bool TryEvacuateWeakCamera(
    const BipartiteCameraPointGraph& graph,
    const LandmarkPartitioningOptions& options,
    int camera,
    int source_cluster,
    int minimum_residuals,
    int maximum_residuals,
    State& state,
    std::vector<int>& landmark_to_cluster,
    std::vector<std::vector<int>>& landmarks_per_cluster,
    ProgressTrace& progress) {
  std::vector<int> landmarks;
  for (int landmark : graph.points_from_camera[camera]) {
    if (landmark_to_cluster[landmark] == source_cluster) {
      landmarks.push_back(landmark);
    }
  }
  if (landmarks.empty()) {
    return false;
  }

  const Objective objective_before = state.objective;
  std::vector<MoveRecord> move_log;
  for (int landmark : landmarks) {
      progress.Tick(state.objective);
    if (!TryMoveLandmarkOutOfWeakCamera(
            graph, options, camera, landmark, source_cluster,
            minimum_residuals, maximum_residuals, state,
            landmark_to_cluster, landmarks_per_cluster,
        move_log, progress)) {
          RollbackMoves(graph, options, 0, state, landmark_to_cluster, move_log,
                &landmarks_per_cluster);
      return false;
    }
  }

  if (!IsBetter(state.objective, objective_before) ||
      state.camera_cluster_degree[
          camera * options.cluster_count + source_cluster] != 0) {
      RollbackMoves(graph, options, 0, state, landmark_to_cluster, move_log,
              &landmarks_per_cluster);
    return false;
  }
  return true;
}

bool ReinforceWeakCameraInPlace(
    const BipartiteCameraPointGraph& graph,
    const LandmarkPartitioningOptions& options,
    int camera,
    int target_cluster,
    int minimum_residuals,
    int maximum_residuals,
    State& state,
    std::vector<int>& landmark_to_cluster,
    std::vector<MoveRecord>& move_log,
    std::vector<std::vector<int>>& landmarks_per_cluster,
    ProgressTrace& progress) {
  const int offset = camera * options.cluster_count + target_cluster;
  int capacity_moves = 0;
  while (state.camera_cluster_degree[offset] <
         options.weak_camera_degree_limit) {
    int best_landmark = -1;
    int best_source = -1;
    Objective best_objective;
    for (int landmark : graph.points_from_camera[camera]) {
      progress.Tick(state.objective);
      const int source = landmark_to_cluster[landmark];
      if (source == target_cluster) {
        continue;
      }
      const int weight = LandmarkWeight(graph, landmark);
        if (state.residuals_per_cluster[source] - weight <
              minimum_residuals ||
          state.residuals_per_cluster[target_cluster] + weight >
              maximum_residuals) {
        continue;
      }
      Objective candidate = ScoreMove(
          graph, options, state, landmark, source, target_cluster);
      if (best_landmark < 0 || IsBetter(candidate, best_objective)) {
        best_landmark = landmark;
        best_source = source;
        best_objective = std::move(candidate);
      }
    }
    if (best_landmark < 0) {
        if (options.repair_restart_interval != 1 &&
          capacity_moves >= options.weak_camera_degree_limit) {
        return false;
      }
      int outbound_landmark = -1;
      int outbound_target = -1;
      Objective outbound_objective;
      for (int landmark : landmarks_per_cluster[target_cluster]) {
        progress.Tick(state.objective);
        if (std::binary_search(graph.cameras_from_point[landmark].begin(),
                               graph.cameras_from_point[landmark].end(),
                               camera)) {
          continue;
        }
        const int weight = LandmarkWeight(graph, landmark);
        if (state.residuals_per_cluster[target_cluster] - weight <
            minimum_residuals) {
          continue;
        }
        for (int target = 0; target < options.cluster_count; ++target) {
          if (target == target_cluster ||
              state.residuals_per_cluster[target] + weight >
                  maximum_residuals) {
            continue;
          }
          Objective candidate = ScoreMove(
              graph, options, state, landmark, target_cluster, target);
          if (outbound_landmark < 0 ||
              IsBetter(candidate, outbound_objective)) {
            outbound_landmark = landmark;
            outbound_target = target;
            outbound_objective = std::move(candidate);
          }
        }
      }
      if (outbound_landmark < 0) {
        return false;
      }
      ApplyRecordedMove(graph, options, outbound_landmark, target_cluster,
                        outbound_target, outbound_objective, state,
                        landmark_to_cluster, move_log,
                        &landmarks_per_cluster);
      ++capacity_moves;
      continue;
    }
    ApplyRecordedMove(graph, options, best_landmark, best_source,
                      target_cluster, best_objective, state,
                      landmark_to_cluster, move_log,
                      &landmarks_per_cluster);
  }
  return true;
}

bool TryReinforceWeakCamera(
    const BipartiteCameraPointGraph& graph,
    const LandmarkPartitioningOptions& options,
    int camera,
    int target_cluster,
    int minimum_residuals,
    int maximum_residuals,
    State& state,
    std::vector<int>& landmark_to_cluster,
    std::vector<std::vector<int>>& landmarks_per_cluster,
    ProgressTrace& progress) {
  const Objective objective_before = state.objective;
  std::vector<MoveRecord> move_log;
  if (!ReinforceWeakCameraInPlace(
          graph, options, camera, target_cluster, minimum_residuals,
          maximum_residuals, state, landmark_to_cluster, move_log,
          landmarks_per_cluster, progress)) {
    RollbackMoves(graph, options, 0, state, landmark_to_cluster, move_log,
                  &landmarks_per_cluster);
    return false;
  }

  std::vector<bool> blocked(
      graph.camera_count * options.cluster_count, false);
    const int repair_limit = options.repair_restart_interval != 1
      ? options.weak_camera_degree_limit
      : graph.camera_count;
  for (int repair = 0;
       !IsBetter(state.objective, objective_before) &&
       repair < repair_limit;
       ++repair) {
    int next_offset = -1;
    int next_degree = -1;
    for (int candidate_camera = 0;
         candidate_camera < graph.camera_count; ++candidate_camera) {
      for (int cluster = 0; cluster < options.cluster_count; ++cluster) {
        progress.Tick(state.objective);
        const int candidate_offset =
            candidate_camera * options.cluster_count + cluster;
        const int degree =
          state.camera_cluster_degree[candidate_offset];
        if (!blocked[candidate_offset] && degree > 0 &&
            degree < options.weak_camera_degree_limit &&
            degree > next_degree) {
          next_offset = candidate_offset;
          next_degree = degree;
        }
      }
    }
    if (next_offset < 0) {
      RollbackMoves(graph, options, 0, state, landmark_to_cluster, move_log,
                    &landmarks_per_cluster);
      return false;
    }
    const std::size_t first_trial_move = move_log.size();
    if (!ReinforceWeakCameraInPlace(
            graph, options, next_offset / options.cluster_count,
            next_offset % options.cluster_count, minimum_residuals,
                maximum_residuals, state, landmark_to_cluster, move_log,
              landmarks_per_cluster, progress)) {
      RollbackMoves(graph, options, first_trial_move, state,
                  landmark_to_cluster, move_log, &landmarks_per_cluster);
      blocked[next_offset] = true;
      continue;
    }
    std::fill(blocked.begin(), blocked.end(), false);
  }
  if (!IsBetter(state.objective, objective_before)) {
    RollbackMoves(graph, options, 0, state, landmark_to_cluster, move_log,
                  &landmarks_per_cluster);
    return false;
  }
  return true;
}

Objective ScoreSwap(const BipartiteCameraPointGraph& graph,
                    const LandmarkPartitioningOptions& options,
                    const State& state,
                    int first_landmark,
                    int second_landmark,
                    int first_cluster,
                    int second_cluster) {
  Objective candidate = state.objective;
  const int first_multiplicity = PointMultiplicity(graph, first_landmark);
  const int second_multiplicity = PointMultiplicity(graph, second_landmark);
  const int first_weight = LandmarkWeight(graph, first_landmark);
  const int second_weight = LandmarkWeight(graph, second_landmark);
  const int first_residuals = state.residuals_per_cluster[first_cluster];
  const int second_residuals = state.residuals_per_cluster[second_cluster];
  candidate.residual_imbalance +=
      Square(first_residuals - first_weight + second_weight) -
      Square(first_residuals) +
      Square(second_residuals - second_weight + first_weight) -
      Square(second_residuals);
  const auto& first_cameras = graph.cameras_from_point[first_landmark];
  const auto& second_cameras = graph.cameras_from_point[second_landmark];
  int first_camera_delta = 0;
  int second_camera_delta = 0;
  std::size_t first_index = 0;
  std::size_t second_index = 0;
  while (first_index < first_cameras.size() ||
         second_index < second_cameras.size()) {
    const bool observes_first = first_index < first_cameras.size() &&
        (second_index == second_cameras.size() ||
         first_cameras[first_index] <= second_cameras[second_index]);
    const bool observes_second = second_index < second_cameras.size() &&
        (first_index == first_cameras.size() ||
         second_cameras[second_index] <= first_cameras[first_index]);
    const int camera = observes_first
                           ? first_cameras[first_index]
                           : second_cameras[second_index];
    if (observes_first) {
      ++first_index;
    }
    if (observes_second) {
      ++second_index;
    }
    const int first_offset = camera * options.cluster_count + first_cluster;
    const int second_offset = camera * options.cluster_count + second_cluster;
    const int first_delta =
        (observes_second ? second_multiplicity : 0) -
        (observes_first ? first_multiplicity : 0);
    const int second_delta = -first_delta;
    if (first_delta == 0) {
      continue;
    }
    const int old_first = state.camera_cluster_degree[first_offset];
    const int old_second = state.camera_cluster_degree[second_offset];
    first_camera_delta += (old_first + first_delta > 0) - (old_first > 0);
    second_camera_delta +=
      (old_second + second_delta > 0) - (old_second > 0);
    ReplaceWeakDegree(old_first, old_first + first_delta,
                      options.weak_camera_degree_limit, candidate);
    ReplaceWeakDegree(old_second, old_second + second_delta,
                      options.weak_camera_degree_limit, candidate);
    const int occurrences_before = state.camera_cluster_count[camera];
    const int occurrences_after =
        occurrences_before + (old_first + first_delta > 0) - (old_first > 0) +
        (old_second + second_delta > 0) - (old_second > 0);
    candidate.copied_camera_count +=
        std::max(0, occurrences_after - 1) -
        std::max(0, occurrences_before - 1);
  }
        if (options.optimize_max_camera_count) {
          candidate.maximum_camera_count = MaximumCameraCount(
          state, first_cluster, first_camera_delta,
          second_cluster, second_camera_delta);
        }
  return candidate;
}

void ApplySwap(const BipartiteCameraPointGraph& graph,
               const LandmarkPartitioningOptions& options,
               int first_landmark,
               int second_landmark,
               int first_cluster,
               int second_cluster,
               const Objective& objective,
               State& state,
               std::vector<int>& landmark_to_cluster) {
  const int first_multiplicity = PointMultiplicity(graph, first_landmark);
  const int second_multiplicity = PointMultiplicity(graph, second_landmark);
  const auto& first_cameras = graph.cameras_from_point[first_landmark];
  const auto& second_cameras = graph.cameras_from_point[second_landmark];
  std::size_t first_index = 0;
  std::size_t second_index = 0;
  while (first_index < first_cameras.size() ||
         second_index < second_cameras.size()) {
    const bool observes_first = first_index < first_cameras.size() &&
        (second_index == second_cameras.size() ||
         first_cameras[first_index] <= second_cameras[second_index]);
    const bool observes_second = second_index < second_cameras.size() &&
        (first_index == first_cameras.size() ||
         second_cameras[second_index] <= first_cameras[first_index]);
    const int camera = observes_first
                           ? first_cameras[first_index]
                           : second_cameras[second_index];
    if (observes_first) {
      ++first_index;
    }
    if (observes_second) {
      ++second_index;
    }
    const int first_delta =
        (observes_second ? second_multiplicity : 0) -
        (observes_first ? first_multiplicity : 0);
    if (first_delta == 0) {
      continue;
    }
    const int first_offset = camera * options.cluster_count + first_cluster;
    const int second_offset = camera * options.cluster_count + second_cluster;
    const int old_first = state.camera_cluster_degree[first_offset];
    const int old_second = state.camera_cluster_degree[second_offset];
    state.camera_cluster_degree[first_offset] += first_delta;
    state.camera_cluster_degree[second_offset] -= first_delta;
    state.cameras_per_cluster[first_cluster] +=
      (state.camera_cluster_degree[first_offset] > 0) - (old_first > 0);
    state.cameras_per_cluster[second_cluster] +=
      (state.camera_cluster_degree[second_offset] > 0) - (old_second > 0);
    state.camera_cluster_count[camera] +=
        (state.camera_cluster_degree[first_offset] > 0) - (old_first > 0) +
        (state.camera_cluster_degree[second_offset] > 0) - (old_second > 0);
  }
  const int first_weight = LandmarkWeight(graph, first_landmark);
  const int second_weight = LandmarkWeight(graph, second_landmark);
  state.residuals_per_cluster[first_cluster] += second_weight - first_weight;
  state.residuals_per_cluster[second_cluster] += first_weight - second_weight;
  landmark_to_cluster[first_landmark] = second_cluster;
  landmark_to_cluster[second_landmark] = first_cluster;
  state.objective = objective;
}

void Validate(const BipartiteCameraPointGraph& graph,
              const LandmarkPartitioningOptions& options) {
  if (options.cluster_count <= 0 || options.cluster_count > graph.point_count) {
    throw std::invalid_argument(
        "cluster_count must be between one and landmark_count");
  }
  if (options.weak_camera_degree_limit < 2) {
    throw std::invalid_argument(
        "weak_camera_degree_limit must be at least two");
  }
  if (options.residual_balance_slack < 0.0 ||
      options.residual_balance_slack > 0.05) {
    throw std::invalid_argument(
        "residual_balance_slack must be in the range [0, 0.05]");
  }
  if (options.max_refinement_passes < 0 ||
      options.max_swap_candidates_per_landmark < 0) {
    throw std::invalid_argument("refinement limits cannot be negative");
  }
  if (options.repair_restart_interval < 0) {
    throw std::invalid_argument("repair restart interval cannot be negative");
  }
  if (options.max_repair_work_per_phase < 0) {
    throw std::invalid_argument("repair work limit cannot be negative");
  }
  if (options.hard_group_max_camera_count < 0) {
    throw std::invalid_argument("hard-group camera limit cannot be negative");
  }
}

struct HardGroupedGraph {
  BipartiteCameraPointGraph graph;
  std::vector<std::vector<int>> original_landmarks;
};

HardGroupedGraph BuildHardGroupedGraph(
    const BipartiteCameraPointGraph& original,
    const LandmarkPartitioningOptions& options) {
  const int maximum_residuals = ResidualBounds(original, options).second;
  std::map<std::vector<int>, std::vector<int>> landmarks_by_signature;
  for (int landmark = 0; landmark < original.point_count; ++landmark) {
    landmarks_by_signature[original.cameras_from_point[landmark]].push_back(
        landmark);
  }

  HardGroupedGraph grouped;
  grouped.graph.camera_count = original.camera_count;
  grouped.graph.points_from_camera.resize(original.camera_count);
  for (const auto& [signature, landmarks] : landmarks_by_signature) {
    const bool eligible = !signature.empty() && landmarks.size() > 1 &&
        static_cast<int>(signature.size()) <=
            options.hard_group_max_camera_count;
    const int maximum_chunk_size = eligible
        ? std::max(1, maximum_residuals /
                          static_cast<int>(signature.size()))
        : 1;
    for (std::size_t begin = 0; begin < landmarks.size();
         begin += maximum_chunk_size) {
      const std::size_t end = std::min(
          landmarks.size(), begin + maximum_chunk_size);
      const int grouped_landmark = grouped.graph.point_count++;
      grouped.graph.cameras_from_point.push_back(signature);
      grouped.graph.point_multiplicity.push_back(
          static_cast<int>(end - begin));
      grouped.original_landmarks.emplace_back(
          landmarks.begin() + begin, landmarks.begin() + end);
      for (int camera : signature) {
        grouped.graph.points_from_camera[camera].push_back(grouped_landmark);
      }
    }
  }
  return grouped;
}

}  // namespace

LandmarkPartitioner::LandmarkPartitioner(LandmarkPartitioningOptions options)
    : options_(std::move(options)) {}

LandmarkPartition LandmarkPartitioner::Partition(
    const BipartiteCameraPointGraph& graph) const {
  Validate(graph, options_);
  if (options_.hard_group_max_camera_count > 0) {
    HardGroupedGraph grouped = BuildHardGroupedGraph(graph, options_);
    if (grouped.graph.point_count < options_.cluster_count) {
      throw std::invalid_argument(
          "hard grouping leaves fewer landmark groups than clusters");
    }
    LandmarkPartitioningOptions grouped_options = options_;
    grouped_options.hard_group_max_camera_count = 0;
    LandmarkPartition grouped_result =
        LandmarkPartitioner(grouped_options).Partition(grouped.graph);

    LandmarkPartition result;
    result.landmark_to_cluster.assign(graph.point_count, -1);
    for (int grouped_landmark = 0;
         grouped_landmark < grouped.graph.point_count; ++grouped_landmark) {
      const int cluster =
          grouped_result.landmark_to_cluster[grouped_landmark];
      for (int original_landmark :
           grouped.original_landmarks[grouped_landmark]) {
        result.landmark_to_cluster[original_landmark] = cluster;
      }
    }
    result.metrics = Evaluate(graph, result.landmark_to_cluster, options_);
    result.accepted_refinement_moves =
        grouped_result.accepted_refinement_moves;
    std::cout << "Hard-grouped " << graph.point_count << " landmarks into "
              << grouped.graph.point_count << " weighted groups\n";
    return result;
  }
  ProgressTrace progress;
  const auto [minimum_residuals, maximum_residuals] =
      ResidualBounds(graph, options_);
  State state(graph, options_);
  std::vector<int> landmark_to_cluster(graph.point_count, -1);
  std::map<std::vector<int>, std::vector<int>> landmarks_by_signature;
  for (int landmark = 0; landmark < graph.point_count; ++landmark) {
    landmarks_by_signature[graph.cameras_from_point[landmark]].push_back(
        landmark);
  }
  std::vector<const std::vector<int>*> signature_groups;
  signature_groups.reserve(landmarks_by_signature.size());
  for (const auto& [signature, landmarks] : landmarks_by_signature) {
    static_cast<void>(signature);
    signature_groups.push_back(&landmarks);
  }
  std::stable_sort(
      signature_groups.begin(), signature_groups.end(),
      [&graph](const std::vector<int>* left, const std::vector<int>* right) {
        const auto group_weight = [&graph](const std::vector<int>* group) {
          return group->size() *
                 LandmarkWeight(graph, group->front());
        };
        return group_weight(left) > group_weight(right);
      });
  std::vector<int> order;
  order.reserve(graph.point_count);
  for (const std::vector<int>* group : signature_groups) {
    order.insert(order.end(), group->begin(), group->end());
  }

  progress.Begin("initialization", -1);
  for (int landmark : order) {
    progress.Tick(state.objective);
    const int weight = LandmarkWeight(graph, landmark);
    int best_cluster = -1;
    Objective best_objective;
    for (int cluster = 0; cluster < options_.cluster_count; ++cluster) {
      if (state.residuals_per_cluster[cluster] + weight > maximum_residuals) {
        continue;
      }
      Objective candidate = ScoreMove(
          graph, options_, state, landmark, -1, cluster);
      if (best_cluster < 0 || IsBetter(candidate, best_objective) ||
          (IsEquivalent(candidate, best_objective) &&
           state.residuals_per_cluster[cluster] <
               state.residuals_per_cluster[best_cluster])) {
        best_cluster = cluster;
        best_objective = std::move(candidate);
      }
    }
    if (best_cluster < 0) {
      best_cluster = static_cast<int>(std::min_element(
          state.residuals_per_cluster.begin(),
          state.residuals_per_cluster.end()) -
          state.residuals_per_cluster.begin());
      best_objective = ScoreMove(
          graph, options_, state, landmark, -1, best_cluster);
    }
    ApplyMove(graph, options_, landmark, -1, best_cluster, best_objective,
              state, landmark_to_cluster);
  }
  progress.Finish(state.objective);

  int accepted_moves = 0;
  for (int pass = 0; pass < options_.max_refinement_passes; ++pass) {
    bool changed = false;
    progress.Begin("direct moves", pass);
    for (int landmark = 0; landmark < graph.point_count; ++landmark) {
      progress.Tick(state.objective);
      const int source = landmark_to_cluster[landmark];
      const int weight = LandmarkWeight(graph, landmark);
      int best_target = source;
      Objective best_objective = state.objective;
        const int current_violation = ResidualViolation(
          state.residuals_per_cluster, minimum_residuals, maximum_residuals);
        int best_violation = current_violation;
      for (int target = 0; target < options_.cluster_count; ++target) {
        if (target == source) {
          continue;
        }
        const int source_residuals = state.residuals_per_cluster[source];
        const int target_residuals = state.residuals_per_cluster[target];
        const int violation =
          current_violation -
          ResidualViolationForCluster(
            source_residuals, minimum_residuals, maximum_residuals) -
          ResidualViolationForCluster(
            target_residuals, minimum_residuals, maximum_residuals) +
          ResidualViolationForCluster(
            source_residuals - weight,
            minimum_residuals, maximum_residuals) +
          ResidualViolationForCluster(
            target_residuals + weight,
            minimum_residuals, maximum_residuals);
        if (violation > best_violation) {
          continue;
        }
        Objective candidate = ScoreMove(
            graph, options_, state, landmark, source, target);
        if (violation < best_violation || IsBetter(candidate, best_objective)) {
          best_target = target;
          best_violation = violation;
          best_objective = std::move(candidate);
        }
      }
      if (best_target != source) {
        ApplyMove(graph, options_, landmark, source, best_target,
                  best_objective, state, landmark_to_cluster);
        ++accepted_moves;
        changed = true;
      }
    }
    progress.Finish(state.objective);

    progress.Begin("swaps", pass);
    for (int first = 0;
          pass == 0 && state.objective.weak_camera_count > 0 &&
          first < graph.point_count;
         ++first) {
          progress.Tick(state.objective);
      const int first_cluster = landmark_to_cluster[first];
      int tried = 0;
      for (int second = 0; second < graph.point_count; ++second) {
        if (landmark_to_cluster[second] == first_cluster) {
          continue;
        }
        if (options_.max_swap_candidates_per_landmark > 0 &&
            tried++ >= options_.max_swap_candidates_per_landmark) {
          break;
        }
        const int second_cluster = landmark_to_cluster[second];
        const int first_weight = LandmarkWeight(graph, first);
        const int second_weight = LandmarkWeight(graph, second);
        const int new_first_residuals =
            state.residuals_per_cluster[first_cluster] - first_weight +
            second_weight;
        const int new_second_residuals =
            state.residuals_per_cluster[second_cluster] - second_weight +
            first_weight;
        if (new_first_residuals < minimum_residuals ||
            new_first_residuals > maximum_residuals ||
            new_second_residuals < minimum_residuals ||
            new_second_residuals > maximum_residuals) {
          continue;
        }
        Objective candidate = ScoreSwap(
            graph, options_, state, first, second, first_cluster,
            second_cluster);
        if (IsBetter(candidate, state.objective)) {
          ApplySwap(graph, options_, first, second, first_cluster,
                    second_cluster, candidate, state, landmark_to_cluster);
          ++accepted_moves;
          changed = true;
          break;
        }
      }
    }
    progress.Finish(state.objective);

    std::vector<std::vector<int>> landmarks_per_cluster(
        options_.cluster_count);
    for (int landmark = 0; landmark < graph.point_count; ++landmark) {
      landmarks_per_cluster[landmark_to_cluster[landmark]].push_back(landmark);
    }
    progress.Begin("evacuation", pass);
    bool evacuated = true;
    while (evacuated && state.objective.weak_camera_count > 0 &&
         (options_.max_repair_work_per_phase == 0 ||
        progress.work_units() < options_.max_repair_work_per_phase)) {
      evacuated = false;
      bool restart_scan = false;
      int repairs_this_scan = 0;
      for (int degree = 1;
          degree < options_.weak_camera_degree_limit && !restart_scan;
          ++degree) {
        for (int camera = 0;
           camera < graph.camera_count && !restart_scan;
           ++camera) {
          for (int cluster = 0; cluster < options_.cluster_count; ++cluster) {
            if (options_.max_repair_work_per_phase > 0 &&
                progress.work_units() >=
                    options_.max_repair_work_per_phase) {
              restart_scan = true;
              evacuated = false;
              break;
            }
            if (state.camera_cluster_degree[
                    camera * options_.cluster_count + cluster] != degree) {
              continue;
            }
            if (TryEvacuateWeakCamera(
                    graph, options_, camera, cluster, minimum_residuals,
                  maximum_residuals, state, landmark_to_cluster,
                  landmarks_per_cluster, progress)) {
              ++accepted_moves;
              changed = true;
              evacuated = true;
              ++repairs_this_scan;
              if (options_.repair_restart_interval > 0 &&
                  repairs_this_scan >= options_.repair_restart_interval) {
                restart_scan = true;
                break;
              }
            }
            progress.Tick(state.objective);
          }
        }
      }
    }
    progress.Finish(state.objective);
    progress.Begin("reinforcement", pass);
    bool reinforced = true;
    while (reinforced && state.objective.weak_camera_count > 0 &&
         (options_.max_repair_work_per_phase == 0 ||
        progress.work_units() < options_.max_repair_work_per_phase)) {
      reinforced = false;
      bool restart_scan = false;
      int repairs_this_scan = 0;
      for (int degree = options_.weak_camera_degree_limit - 1;
          degree > 0 && !restart_scan;
          --degree) {
        for (int camera = 0;
           camera < graph.camera_count && !restart_scan;
           ++camera) {
          for (int cluster = 0; cluster < options_.cluster_count; ++cluster) {
            if (options_.max_repair_work_per_phase > 0 &&
                progress.work_units() >=
                    options_.max_repair_work_per_phase) {
              restart_scan = true;
              reinforced = false;
              break;
            }
            if (state.camera_cluster_degree[
                    camera * options_.cluster_count + cluster] != degree) {
              continue;
            }
            if (TryReinforceWeakCamera(
                    graph, options_, camera, cluster, minimum_residuals,
                  maximum_residuals, state, landmark_to_cluster,
                  landmarks_per_cluster, progress)) {
              ++accepted_moves;
              changed = true;
              reinforced = true;
              ++repairs_this_scan;
              if (options_.repair_restart_interval > 0 &&
                  repairs_this_scan >= options_.repair_restart_interval) {
                restart_scan = true;
                break;
              }
            }
            progress.Tick(state.objective);
          }
        }
      }
    }
    progress.Finish(state.objective);
    if (!changed) {
      break;
    }
  }

  LandmarkPartition result;
  result.landmark_to_cluster = std::move(landmark_to_cluster);
  result.metrics = Evaluate(graph, result.landmark_to_cluster, options_);
  result.accepted_refinement_moves = accepted_moves;
  return result;
}

LandmarkPartitionMetrics LandmarkPartitioner::Evaluate(
    const BipartiteCameraPointGraph& graph,
    const std::vector<int>& landmark_to_cluster,
    const LandmarkPartitioningOptions& options) {
  Validate(graph, options);
  if (landmark_to_cluster.size() !=
      static_cast<std::size_t>(graph.point_count)) {
    throw std::invalid_argument("landmark assignment has the wrong size");
  }
  LandmarkPartitionMetrics metrics;
  metrics.weak_camera_counts.assign(options.weak_camera_degree_limit, 0);
  metrics.residuals_per_cluster.assign(options.cluster_count, 0);
  metrics.landmarks_per_cluster.assign(options.cluster_count, 0);
  metrics.cameras_per_cluster.assign(options.cluster_count, 0);
  std::vector<int> camera_cluster_degree(
      graph.camera_count * options.cluster_count, 0);
  std::vector<int> camera_cluster_count(graph.camera_count, 0);

  for (int landmark = 0; landmark < graph.point_count; ++landmark) {
    const int cluster = landmark_to_cluster[landmark];
    if (cluster < 0 || cluster >= options.cluster_count) {
      throw std::invalid_argument(
          "landmark assignment contains an invalid cluster");
    }
    ++metrics.landmarks_per_cluster[cluster];
    metrics.residuals_per_cluster[cluster] += LandmarkWeight(graph, landmark);
    for (int camera : graph.cameras_from_point[landmark]) {
      camera_cluster_degree[camera * options.cluster_count + cluster] +=
          PointMultiplicity(graph, landmark);
    }
  }
  for (int camera = 0; camera < graph.camera_count; ++camera) {
    for (int cluster = 0; cluster < options.cluster_count; ++cluster) {
      const int degree =
          camera_cluster_degree[camera * options.cluster_count + cluster];
      if (degree == 0) {
        continue;
      }
      ++camera_cluster_count[camera];
      ++metrics.cameras_per_cluster[cluster];
      if (degree < options.weak_camera_degree_limit) {
        ++metrics.weak_camera_counts[degree];
      }
    }
    metrics.copied_camera_count += std::max(0, camera_cluster_count[camera] - 1);
  }
  const auto [minimum_residuals, maximum_residuals] =
      ResidualBounds(graph, options);
  metrics.minimum_residuals = minimum_residuals;
  metrics.maximum_residuals = maximum_residuals;
  metrics.residual_balance_violation = ResidualViolation(
      metrics.residuals_per_cluster, minimum_residuals, maximum_residuals);
  return metrics;
}

}  // namespace bundle_palm

extern "C" int cluster_landmarks_clean(
    int cluster_count,
    int camera_count,
    int landmark_count,
    int minimum_camera_landmarks,
    int max_refinement_passes,
    int repair_restart_interval,
    std::int64_t max_repair_work_per_phase,
    int hard_group_max_camera_count,
    bool optimize_max_camera_count,
    double residual_balance_slack,
    const std::vector<int>& camera_indices,
    const std::vector<int>& landmark_indices,
    std::vector<int>& landmark_to_cluster_out) {
  try {
    const auto graph = bundle_palm::BipartiteCameraPointGraph::FromObservations(
        camera_count, landmark_count, camera_indices, landmark_indices);
    bundle_palm::LandmarkPartitioningOptions options;
    options.cluster_count = cluster_count;
    options.weak_camera_degree_limit = minimum_camera_landmarks;
    options.max_refinement_passes = max_refinement_passes;
    options.repair_restart_interval = repair_restart_interval;
    options.max_repair_work_per_phase = max_repair_work_per_phase;
    options.hard_group_max_camera_count = hard_group_max_camera_count;
    options.optimize_max_camera_count = optimize_max_camera_count;
    options.residual_balance_slack = residual_balance_slack;
    const auto result =
        bundle_palm::LandmarkPartitioner(options).Partition(graph);
    landmark_to_cluster_out = result.landmark_to_cluster;
    for (int cluster = 0; cluster < cluster_count; ++cluster) {
      std::cout << "Cluster " << cluster << " covers ("
                << result.metrics.residuals_per_cluster[cluster]
                << ", 2) residuals ("
                << result.metrics.landmarks_per_cluster[cluster] << ",) of "
                << landmark_count << " landmarks ("
                << result.metrics.cameras_per_cluster[cluster] << ",) of "
                << camera_count << " cameras\n";
    }
    std::cout << "Residual balance bounds ["
              << result.metrics.minimum_residuals << ", "
              << result.metrics.maximum_residuals << "], violation "
              << result.metrics.residual_balance_violation << "\n";
    std::cout << "Weak camera-cluster counts 1.."
          << options.weak_camera_degree_limit - 1 << ":";
    for (int degree = 1; degree < options.weak_camera_degree_limit; ++degree) {
      std::cout << " " << result.metrics.weak_camera_counts[degree];
    }
    std::cout << "\nAdditional camera copies: "
              << result.metrics.copied_camera_count << "\n";
        std::cout << "Maximum cameras in a cluster: "
        << *std::max_element(
          result.metrics.cameras_per_cluster.begin(),
          result.metrics.cameras_per_cluster.end())
        << " (objective "
        << (options.optimize_max_camera_count ? "enabled" : "disabled")
        << ")\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Clean landmark partitioning failed: " << error.what() << "\n";
    landmark_to_cluster_out.clear();
    return 1;
  }
}