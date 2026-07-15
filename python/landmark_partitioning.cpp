#include "landmark_partitioning.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace bundle_palm {
namespace {

struct Objective {
  std::int64_t weak_camera_count = 0;
  std::int64_t weak_camera_distance = 0;
  std::int64_t residual_imbalance = 0;
  std::int64_t copied_camera_count = 0;
};

struct State {
  State(const BipartiteCameraPointGraph& graph,
        const LandmarkPartitioningOptions& options)
      : camera_cluster_degree(graph.camera_count * options.cluster_count, 0),
        camera_cluster_count(graph.camera_count, 0),
        residuals_per_cluster(options.cluster_count, 0),
        objective{} {}

  std::vector<int> camera_cluster_degree;
  std::vector<int> camera_cluster_count;
  std::vector<int> residuals_per_cluster;
  Objective objective;
};

struct MoveRecord {
  int landmark;
  int source_cluster;
  int target_cluster;
  Objective objective_before;
};

bool IsBetter(const Objective& left, const Objective& right) {
  if (left.weak_camera_count != right.weak_camera_count) {
    return left.weak_camera_count < right.weak_camera_count;
  }
  if (left.weak_camera_distance != right.weak_camera_distance) {
    return left.weak_camera_distance < right.weak_camera_distance;
  }
  if (left.residual_imbalance != right.residual_imbalance) {
    return left.residual_imbalance < right.residual_imbalance;
  }
  return left.copied_camera_count < right.copied_camera_count;
}

bool IsEquivalent(const Objective& left, const Objective& right) {
  return !IsBetter(left, right) && !IsBetter(right, left);
}

void ReplaceWeakDegree(int old_degree,
                       int new_degree,
                       int degree_limit,
                       Objective& objective) {
  if (old_degree > 0 && old_degree < degree_limit) {
    --objective.weak_camera_count;
    objective.weak_camera_distance -=
        std::min(old_degree, degree_limit - old_degree);
  }
  if (new_degree > 0 && new_degree < degree_limit) {
    ++objective.weak_camera_count;
    objective.weak_camera_distance +=
        std::min(new_degree, degree_limit - new_degree);
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

std::pair<int, int> ResidualBounds(
    const BipartiteCameraPointGraph& graph,
    const LandmarkPartitioningOptions& options) {
  const int residual_count = std::accumulate(
      graph.cameras_from_point.begin(), graph.cameras_from_point.end(), 0,
      [](int sum, const std::vector<int>& cameras) {
        return sum + static_cast<int>(cameras.size());
      });
  const double target = static_cast<double>(residual_count) /
                        static_cast<double>(options.cluster_count);
  int largest_landmark = 0;
  for (const auto& cameras : graph.cameras_from_point) {
    largest_landmark =
        std::max(largest_landmark, static_cast<int>(cameras.size()));
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
  const int weight = graph.cameras_from_point[landmark].size();
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
      ReplaceWeakDegree(source_degree, source_degree - 1,
                        options.weak_camera_degree_limit, candidate);
      if (source_degree == 1) {
        --occurrences_after;
      }
    }
    const int target_offset = camera * options.cluster_count + target_cluster;
    const int target_degree = state.camera_cluster_degree[target_offset];
    ReplaceWeakDegree(target_degree, target_degree + 1,
                      options.weak_camera_degree_limit, candidate);
    if (target_degree == 0) {
      ++occurrences_after;
    }
    candidate.copied_camera_count +=
        std::max(0, occurrences_after - 1) -
        std::max(0, occurrences_before - 1);
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
  const int weight = graph.cameras_from_point[landmark].size();
  for (int camera : graph.cameras_from_point[landmark]) {
    if (source_cluster >= 0) {
      const int source_offset = camera * options.cluster_count + source_cluster;
      if (--state.camera_cluster_degree[source_offset] == 0) {
        --state.camera_cluster_count[camera];
      }
    }
    const int target_offset = camera * options.cluster_count + target_cluster;
    if (state.camera_cluster_degree[target_offset]++ == 0) {
      ++state.camera_cluster_count[camera];
    }
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
                       std::vector<MoveRecord>& move_log) {
  move_log.push_back(
      {landmark, source_cluster, target_cluster, state.objective});
  ApplyMove(graph, options, landmark, source_cluster, target_cluster,
            objective, state, landmark_to_cluster);
}

void RollbackMoves(const BipartiteCameraPointGraph& graph,
                   const LandmarkPartitioningOptions& options,
                   std::size_t first_move,
                   State& state,
                   std::vector<int>& landmark_to_cluster,
                   std::vector<MoveRecord>& move_log) {
  for (std::size_t index = move_log.size(); index > first_move; --index) {
    const MoveRecord& move = move_log[index - 1];
    ApplyMove(graph, options, move.landmark, move.target_cluster,
              move.source_cluster, move.objective_before, state,
              landmark_to_cluster);
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
    bool& landmarks_per_cluster_ready) {

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
    const int weight = graph.cameras_from_point[landmark].size();
    if (state.residuals_per_cluster[source_cluster] - weight <
        minimum_residuals ||
      state.residuals_per_cluster[target] + weight > maximum_residuals) {
      const Objective original_objective = state.objective;
      ApplyMoveToState(graph, options, landmark, source_cluster, target,
                       first_objective, state);
      if (!landmarks_per_cluster_ready) {
        for (int outbound = 0; outbound < graph.point_count; ++outbound) {
          const int cluster = landmark_to_cluster[outbound];
          landmarks_per_cluster[cluster].push_back(outbound);
        }
        landmarks_per_cluster_ready = true;
      }
      int best_outbound = -1;
      Objective best_exchange_objective;
      for (int outbound : landmarks_per_cluster[target]) {
        if (std::binary_search(graph.cameras_from_point[outbound].begin(),
                               graph.cameras_from_point[outbound].end(),
                               camera)) {
          continue;
        }
        const int outbound_weight = graph.cameras_from_point[outbound].size();
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
  ApplyMove(graph, options, landmark, source_cluster, best_target,
            best_first_objective, state, landmark_to_cluster);
  if (landmarks_per_cluster_ready) {
    auto& source_landmarks = landmarks_per_cluster[source_cluster];
    source_landmarks.erase(std::lower_bound(
        source_landmarks.begin(), source_landmarks.end(), landmark));
    auto& target_landmarks = landmarks_per_cluster[best_target];
    target_landmarks.insert(std::lower_bound(
        target_landmarks.begin(), target_landmarks.end(), landmark),
        landmark);
  }
  if (best_outbound >= 0) {
    ApplyMove(graph, options, best_outbound, best_target, source_cluster,
              best_objective, state, landmark_to_cluster);
    auto& target_landmarks = landmarks_per_cluster[best_target];
    target_landmarks.erase(std::lower_bound(
        target_landmarks.begin(), target_landmarks.end(), best_outbound));
    auto& source_landmarks = landmarks_per_cluster[source_cluster];
    source_landmarks.insert(std::lower_bound(
        source_landmarks.begin(), source_landmarks.end(), best_outbound),
        best_outbound);
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
    std::vector<int>& landmark_to_cluster) {
  std::vector<int> landmarks;
  for (int landmark : graph.points_from_camera[camera]) {
    if (landmark_to_cluster[landmark] == source_cluster) {
      landmarks.push_back(landmark);
    }
  }
  if (landmarks.empty()) {
    return false;
  }

  State candidate_state = state;
  std::vector<int> candidate_assignment = landmark_to_cluster;
  std::vector<std::vector<int>> landmarks_per_cluster(options.cluster_count);
  bool landmarks_per_cluster_ready = false;
  for (int landmark : landmarks) {
    if (!TryMoveLandmarkOutOfWeakCamera(
            graph, options, camera, landmark, source_cluster,
            minimum_residuals, maximum_residuals, candidate_state,
            candidate_assignment, landmarks_per_cluster,
            landmarks_per_cluster_ready)) {
      return false;
    }
  }

  if (!IsBetter(candidate_state.objective, state.objective) ||
      candidate_state.camera_cluster_degree[
          camera * options.cluster_count + source_cluster] != 0) {
    return false;
  }
  state = std::move(candidate_state);
  landmark_to_cluster = std::move(candidate_assignment);
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
    std::vector<MoveRecord>& move_log) {
  const int offset = camera * options.cluster_count + target_cluster;
  while (state.camera_cluster_degree[offset] <
         options.weak_camera_degree_limit) {
    int best_landmark = -1;
    int best_source = -1;
    Objective best_objective;
    for (int landmark : graph.points_from_camera[camera]) {
      const int source = landmark_to_cluster[landmark];
      if (source == target_cluster) {
        continue;
      }
      const int weight = graph.cameras_from_point[landmark].size();
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
      int outbound_landmark = -1;
      int outbound_target = -1;
      Objective outbound_objective;
      for (int landmark = 0; landmark < graph.point_count; ++landmark) {
        if (landmark_to_cluster[landmark] != target_cluster ||
            std::binary_search(graph.cameras_from_point[landmark].begin(),
                               graph.cameras_from_point[landmark].end(),
                               camera)) {
          continue;
        }
        const int weight = graph.cameras_from_point[landmark].size();
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
                        landmark_to_cluster, move_log);
      continue;
    }
    ApplyRecordedMove(graph, options, best_landmark, best_source,
                      target_cluster, best_objective, state,
                      landmark_to_cluster, move_log);
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
    std::vector<int>& landmark_to_cluster) {
  State candidate_state = state;
  std::vector<int> candidate_assignment = landmark_to_cluster;
    std::vector<MoveRecord> move_log;
  if (!ReinforceWeakCameraInPlace(
          graph, options, camera, target_cluster, minimum_residuals,
      maximum_residuals, candidate_state, candidate_assignment,
      move_log)) {
    return false;
  }

  std::vector<bool> blocked(
      graph.camera_count * options.cluster_count, false);
  for (int repair = 0;
       !IsBetter(candidate_state.objective, state.objective) &&
       repair < graph.camera_count;
       ++repair) {
    int next_offset = -1;
    int next_degree = -1;
    for (int candidate_camera = 0;
         candidate_camera < graph.camera_count; ++candidate_camera) {
      for (int cluster = 0; cluster < options.cluster_count; ++cluster) {
        const int candidate_offset =
            candidate_camera * options.cluster_count + cluster;
        const int degree =
            candidate_state.camera_cluster_degree[candidate_offset];
        if (!blocked[candidate_offset] && degree > 0 &&
            degree < options.weak_camera_degree_limit &&
            degree > next_degree) {
          next_offset = candidate_offset;
          next_degree = degree;
        }
      }
    }
    if (next_offset < 0) {
      return false;
    }
        const std::size_t first_trial_move = move_log.size();
    if (!ReinforceWeakCameraInPlace(
            graph, options, next_offset / options.cluster_count,
            next_offset % options.cluster_count, minimum_residuals,
        maximum_residuals, candidate_state, candidate_assignment,
                move_log)) {
      RollbackMoves(graph, options, first_trial_move, candidate_state,
                  candidate_assignment, move_log);
      blocked[next_offset] = true;
      continue;
    }
    std::fill(blocked.begin(), blocked.end(), false);
  }
  if (!IsBetter(candidate_state.objective, state.objective)) {
    return false;
  }
  state = std::move(candidate_state);
  landmark_to_cluster = std::move(candidate_assignment);
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
  const int first_weight = graph.cameras_from_point[first_landmark].size();
  const int second_weight = graph.cameras_from_point[second_landmark].size();
  const int first_residuals = state.residuals_per_cluster[first_cluster];
  const int second_residuals = state.residuals_per_cluster[second_cluster];
  candidate.residual_imbalance +=
      Square(first_residuals - first_weight + second_weight) -
      Square(first_residuals) +
      Square(second_residuals - second_weight + first_weight) -
      Square(second_residuals);
  std::vector<int> affected_cameras = graph.cameras_from_point[first_landmark];
  affected_cameras.insert(affected_cameras.end(),
                          graph.cameras_from_point[second_landmark].begin(),
                          graph.cameras_from_point[second_landmark].end());
  std::sort(affected_cameras.begin(), affected_cameras.end());
  affected_cameras.erase(
      std::unique(affected_cameras.begin(), affected_cameras.end()),
      affected_cameras.end());

  for (int camera : affected_cameras) {
    const bool observes_first = std::binary_search(
        graph.cameras_from_point[first_landmark].begin(),
        graph.cameras_from_point[first_landmark].end(), camera);
    const bool observes_second = std::binary_search(
        graph.cameras_from_point[second_landmark].begin(),
        graph.cameras_from_point[second_landmark].end(), camera);
    if (observes_first == observes_second) {
      continue;
    }
    const int first_offset = camera * options.cluster_count + first_cluster;
    const int second_offset = camera * options.cluster_count + second_cluster;
    const int first_delta = observes_first ? -1 : 1;
    const int second_delta = -first_delta;
    const int old_first = state.camera_cluster_degree[first_offset];
    const int old_second = state.camera_cluster_degree[second_offset];
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
  for (int camera : graph.cameras_from_point[first_landmark]) {
    const int first_offset = camera * options.cluster_count + first_cluster;
    const int second_offset = camera * options.cluster_count + second_cluster;
    if (--state.camera_cluster_degree[first_offset] == 0) {
      --state.camera_cluster_count[camera];
    }
    if (state.camera_cluster_degree[second_offset]++ == 0) {
      ++state.camera_cluster_count[camera];
    }
  }
  for (int camera : graph.cameras_from_point[second_landmark]) {
    const int second_offset = camera * options.cluster_count + second_cluster;
    const int first_offset = camera * options.cluster_count + first_cluster;
    if (--state.camera_cluster_degree[second_offset] == 0) {
      --state.camera_cluster_count[camera];
    }
    if (state.camera_cluster_degree[first_offset]++ == 0) {
      ++state.camera_cluster_count[camera];
    }
  }
  const int first_weight = graph.cameras_from_point[first_landmark].size();
  const int second_weight = graph.cameras_from_point[second_landmark].size();
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
}

}  // namespace

LandmarkPartitioner::LandmarkPartitioner(LandmarkPartitioningOptions options)
    : options_(std::move(options)) {}

LandmarkPartition LandmarkPartitioner::Partition(
    const BipartiteCameraPointGraph& graph) const {
  Validate(graph, options_);
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
                 graph.cameras_from_point[group->front()].size();
        };
        return group_weight(left) > group_weight(right);
      });
  std::vector<int> order;
  order.reserve(graph.point_count);
  for (const std::vector<int>* group : signature_groups) {
    order.insert(order.end(), group->begin(), group->end());
  }

  for (int landmark : order) {
    const int weight = graph.cameras_from_point[landmark].size();
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

  int accepted_moves = 0;
  for (int pass = 0; pass < options_.max_refinement_passes; ++pass) {
    bool changed = false;
    for (int landmark = 0; landmark < graph.point_count; ++landmark) {
      const int source = landmark_to_cluster[landmark];
      const int weight = graph.cameras_from_point[landmark].size();
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

    for (int first = 0;
          pass == 0 && state.objective.weak_camera_count > 0 &&
          first < graph.point_count;
         ++first) {
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
        const int first_weight = graph.cameras_from_point[first].size();
        const int second_weight = graph.cameras_from_point[second].size();
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

    bool evacuated = true;
    while (evacuated && state.objective.weak_camera_count > 0) {
      evacuated = false;
      for (int degree = 1;
           degree < options_.weak_camera_degree_limit && !evacuated;
           ++degree) {
        for (int camera = 0; camera < graph.camera_count && !evacuated;
             ++camera) {
          for (int cluster = 0; cluster < options_.cluster_count; ++cluster) {
            if (state.camera_cluster_degree[
                    camera * options_.cluster_count + cluster] != degree) {
              continue;
            }
            if (TryEvacuateWeakCamera(
                    graph, options_, camera, cluster, minimum_residuals,
                  maximum_residuals, state, landmark_to_cluster)) {
              ++accepted_moves;
              changed = true;
              evacuated = true;
              break;
            }
          }
        }
      }
    }
    bool reinforced = true;
    while (reinforced && state.objective.weak_camera_count > 0) {
      reinforced = false;
      for (int degree = options_.weak_camera_degree_limit - 1;
           degree > 0 && !reinforced; --degree) {
        for (int camera = 0; camera < graph.camera_count && !reinforced;
             ++camera) {
          for (int cluster = 0; cluster < options_.cluster_count; ++cluster) {
            if (state.camera_cluster_degree[
                    camera * options_.cluster_count + cluster] != degree) {
              continue;
            }
            if (TryReinforceWeakCamera(
                    graph, options_, camera, cluster, minimum_residuals,
                    maximum_residuals, state, landmark_to_cluster)) {
              ++accepted_moves;
              changed = true;
              reinforced = true;
              break;
            }
          }
        }
      }
    }
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
    metrics.residuals_per_cluster[cluster] +=
        graph.cameras_from_point[landmark].size();
    for (int camera : graph.cameras_from_point[landmark]) {
      ++camera_cluster_degree[camera * options.cluster_count + cluster];
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
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Clean landmark partitioning failed: " << error.what() << "\n";
    landmark_to_cluster_out.clear();
    return 1;
  }
}