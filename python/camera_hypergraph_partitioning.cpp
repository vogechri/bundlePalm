#include "camera_hypergraph_partitioning.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace bundle_palm {
namespace {

struct Objective {
  std::vector<std::int64_t> low_degree_point_counts;
  std::int64_t copied_point_count = 0;
};

struct PartitionState {
  explicit PartitionState(const BipartiteCameraPointGraph& graph,
                          const PartitioningOptions& options)
      : point_cluster_degree(
            graph.point_count * options.cluster_count, 0),
        point_cluster_count(graph.point_count, 0),
        cameras_per_cluster(options.cluster_count, 0),
        objective{std::vector<std::int64_t>(options.low_degree_limit, 0), 0} {}

  std::vector<int> point_cluster_degree;
  std::vector<int> point_cluster_count;
  std::vector<int> cameras_per_cluster;
  Objective objective;
};

bool IsBetter(const Objective& left, const Objective& right) {
  for (std::size_t degree = 1;
       degree < left.low_degree_point_counts.size(); ++degree) {
    if (left.low_degree_point_counts[degree] !=
        right.low_degree_point_counts[degree]) {
      return left.low_degree_point_counts[degree] <
             right.low_degree_point_counts[degree];
    }
  }
  return left.copied_point_count < right.copied_point_count;
}

void ReplaceDegreeContribution(int old_degree,
                               int new_degree,
                               int low_degree_limit,
                               Objective& objective) {
  if (old_degree > 0 && old_degree < low_degree_limit) {
    --objective.low_degree_point_counts[old_degree];
  }
  if (new_degree > 0 && new_degree < low_degree_limit) {
    ++objective.low_degree_point_counts[new_degree];
  }
}

Objective ScoreCameraMove(const BipartiteCameraPointGraph& graph,
                          const PartitioningOptions& options,
                          const PartitionState& state,
                          int camera,
                          int source_cluster,
                          int target_cluster) {
  Objective candidate = state.objective;
  for (int point : graph.points_from_camera[camera]) {
    if (source_cluster >= 0) {
      const int source_offset = point * options.cluster_count + source_cluster;
      const int old_source_degree = state.point_cluster_degree[source_offset];
      ReplaceDegreeContribution(old_source_degree, old_source_degree - 1,
                                options.low_degree_limit, candidate);
      if (old_source_degree == 1) {
        --candidate.copied_point_count;
      }
    }

    const int target_offset = point * options.cluster_count + target_cluster;
    const int old_target_degree = state.point_cluster_degree[target_offset];
    ReplaceDegreeContribution(old_target_degree, old_target_degree + 1,
                              options.low_degree_limit, candidate);
    if (old_target_degree == 0 && state.point_cluster_count[point] > 0) {
      ++candidate.copied_point_count;
    }
  }
  return candidate;
}

void ApplyCameraMove(const BipartiteCameraPointGraph& graph,
                     const PartitioningOptions& options,
                     int camera,
                     int source_cluster,
                     int target_cluster,
                     const Objective& new_objective,
                     PartitionState& state,
                     std::vector<int>& camera_to_cluster) {
  for (int point : graph.points_from_camera[camera]) {
    if (source_cluster >= 0) {
      const int source_offset = point * options.cluster_count + source_cluster;
      if (--state.point_cluster_degree[source_offset] == 0) {
        --state.point_cluster_count[point];
      }
    }
    const int target_offset = point * options.cluster_count + target_cluster;
    if (state.point_cluster_degree[target_offset]++ == 0) {
      ++state.point_cluster_count[point];
    }
  }
  if (source_cluster >= 0) {
    --state.cameras_per_cluster[source_cluster];
  }
  ++state.cameras_per_cluster[target_cluster];
  camera_to_cluster[camera] = target_cluster;
  state.objective = new_objective;
}

Objective ScoreCameraSwap(const BipartiteCameraPointGraph& graph,
                          const PartitioningOptions& options,
                          const PartitionState& state,
                          int first_camera,
                          int second_camera,
                          int first_cluster,
                          int second_cluster) {
  Objective candidate = state.objective;
  std::vector<int> affected_points = graph.points_from_camera[first_camera];
  affected_points.insert(affected_points.end(),
                         graph.points_from_camera[second_camera].begin(),
                         graph.points_from_camera[second_camera].end());
  std::sort(affected_points.begin(), affected_points.end());
  affected_points.erase(
      std::unique(affected_points.begin(), affected_points.end()),
      affected_points.end());

  for (int point : affected_points) {
    const bool first_observes = std::binary_search(
        graph.points_from_camera[first_camera].begin(),
        graph.points_from_camera[first_camera].end(), point);
    const bool second_observes = std::binary_search(
        graph.points_from_camera[second_camera].begin(),
        graph.points_from_camera[second_camera].end(), point);
    if (first_observes == second_observes) {
      continue;
    }

    const int first_offset = point * options.cluster_count + first_cluster;
    const int second_offset = point * options.cluster_count + second_cluster;
    const int first_delta = first_observes ? -1 : 1;
    const int second_delta = -first_delta;
    const int old_first_degree = state.point_cluster_degree[first_offset];
    const int old_second_degree = state.point_cluster_degree[second_offset];
    ReplaceDegreeContribution(old_first_degree,
                              old_first_degree + first_delta,
                              options.low_degree_limit, candidate);
    ReplaceDegreeContribution(old_second_degree,
                              old_second_degree + second_delta,
                              options.low_degree_limit, candidate);
    candidate.copied_point_count +=
        (old_first_degree + first_delta > 0) - (old_first_degree > 0);
    candidate.copied_point_count +=
        (old_second_degree + second_delta > 0) - (old_second_degree > 0);
  }
  return candidate;
}

void ApplyCameraSwap(const BipartiteCameraPointGraph& graph,
                     const PartitioningOptions& options,
                     int first_camera,
                     int second_camera,
                     int first_cluster,
                     int second_cluster,
                     const Objective& new_objective,
                     PartitionState& state,
                     std::vector<int>& camera_to_cluster) {
  for (int point : graph.points_from_camera[first_camera]) {
    const int first_offset = point * options.cluster_count + first_cluster;
    const int second_offset = point * options.cluster_count + second_cluster;
    if (--state.point_cluster_degree[first_offset] == 0) {
      --state.point_cluster_count[point];
    }
    if (state.point_cluster_degree[second_offset]++ == 0) {
      ++state.point_cluster_count[point];
    }
  }
  for (int point : graph.points_from_camera[second_camera]) {
    const int second_offset = point * options.cluster_count + second_cluster;
    const int first_offset = point * options.cluster_count + first_cluster;
    if (--state.point_cluster_degree[second_offset] == 0) {
      --state.point_cluster_count[point];
    }
    if (state.point_cluster_degree[first_offset]++ == 0) {
      ++state.point_cluster_count[point];
    }
  }
  camera_to_cluster[first_camera] = second_cluster;
  camera_to_cluster[second_camera] = first_cluster;
  state.objective = new_objective;
}

std::pair<int, int> CameraCountBounds(int camera_count,
                                      const PartitioningOptions& options) {
  const double ideal = static_cast<double>(camera_count) /
                       static_cast<double>(options.cluster_count);
  const int exact_min = camera_count / options.cluster_count;
  const int exact_max =
      (camera_count + options.cluster_count - 1) / options.cluster_count;
  const int lower = std::min(
      exact_min,
      static_cast<int>(std::ceil(ideal * (1.0 - options.camera_balance_slack))));
  const int upper = std::max(
      exact_max,
      static_cast<int>(std::floor(ideal * (1.0 + options.camera_balance_slack))));
  return {std::max(0, lower), std::max(exact_max, upper)};
}

void ValidateOptions(const BipartiteCameraPointGraph& graph,
                     const PartitioningOptions& options) {
  if (options.cluster_count <= 0 || options.cluster_count > graph.camera_count) {
    throw std::invalid_argument(
        "cluster_count must be between one and camera_count");
  }
  if (options.low_degree_limit < 2) {
    throw std::invalid_argument("low_degree_limit must be at least two");
  }
  if (options.camera_balance_slack < 0.0 ||
      options.camera_balance_slack >= 1.0) {
    throw std::invalid_argument(
        "camera_balance_slack must be in the range [0, 1)");
  }
  if (options.max_refinement_passes < 0 ||
      options.max_swap_candidates_per_camera < 0) {
    throw std::invalid_argument("refinement limits cannot be negative");
  }
}

}  // namespace

BipartiteCameraPointGraph BipartiteCameraPointGraph::FromObservations(
    int camera_count,
    int point_count,
    const std::vector<int>& camera_indices,
    const std::vector<int>& point_indices) {
  if (camera_count < 0 || point_count < 0) {
    throw std::invalid_argument("graph sizes cannot be negative");
  }
  if (camera_indices.size() != point_indices.size()) {
    throw std::invalid_argument(
        "camera_indices and point_indices must have equal length");
  }

  BipartiteCameraPointGraph graph;
  graph.camera_count = camera_count;
  graph.point_count = point_count;
  graph.points_from_camera.resize(camera_count);
  graph.cameras_from_point.resize(point_count);
  graph.point_multiplicity.assign(point_count, 1);
  for (std::size_t observation = 0; observation < camera_indices.size();
       ++observation) {
    const int camera = camera_indices[observation];
    const int point = point_indices[observation];
    if (camera < 0 || camera >= camera_count ||
        point < 0 || point >= point_count) {
      throw std::invalid_argument("observation index is outside the graph");
    }
    graph.points_from_camera[camera].push_back(point);
    graph.cameras_from_point[point].push_back(camera);
  }

  for (auto& points : graph.points_from_camera) {
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
  }
  for (auto& cameras : graph.cameras_from_point) {
    std::sort(cameras.begin(), cameras.end());
    cameras.erase(std::unique(cameras.begin(), cameras.end()), cameras.end());
  }
  return graph;
}

CameraHypergraphPartitioner::CameraHypergraphPartitioner(
    PartitioningOptions options)
    : options_(std::move(options)) {}

CameraPartition CameraHypergraphPartitioner::Partition(
    const BipartiteCameraPointGraph& graph) const {
  ValidateOptions(graph, options_);

  PartitionState state(graph, options_);
  std::vector<int> camera_to_cluster(graph.camera_count, -1);
  std::vector<int> target_sizes(options_.cluster_count,
                                graph.camera_count / options_.cluster_count);
  for (int cluster = 0;
       cluster < graph.camera_count % options_.cluster_count; ++cluster) {
    ++target_sizes[cluster];
  }

  std::vector<int> camera_order(graph.camera_count);
  std::iota(camera_order.begin(), camera_order.end(), 0);
  std::stable_sort(camera_order.begin(), camera_order.end(),
                   [&graph](int left, int right) {
                     return graph.points_from_camera[left].size() >
                            graph.points_from_camera[right].size();
                   });

  for (int camera : camera_order) {
    int best_cluster = -1;
    Objective best_objective;
    for (int cluster = 0; cluster < options_.cluster_count; ++cluster) {
      if (state.cameras_per_cluster[cluster] >= target_sizes[cluster]) {
        continue;
      }
      Objective candidate = ScoreCameraMove(
          graph, options_, state, camera, -1, cluster);
      if (best_cluster < 0 || IsBetter(candidate, best_objective) ||
          (!IsBetter(best_objective, candidate) &&
           state.cameras_per_cluster[cluster] <
               state.cameras_per_cluster[best_cluster])) {
        best_cluster = cluster;
        best_objective = std::move(candidate);
      }
    }
    ApplyCameraMove(graph, options_, camera, -1, best_cluster,
                    best_objective, state, camera_to_cluster);
  }

  const auto [minimum_cameras, maximum_cameras] =
      CameraCountBounds(graph.camera_count, options_);
  int accepted_moves = 0;
  for (int pass = 0; pass < options_.max_refinement_passes; ++pass) {
    bool changed = false;
    for (int camera = 0; camera < graph.camera_count; ++camera) {
      const int source = camera_to_cluster[camera];
      int best_target = source;
      Objective best_objective = state.objective;
      for (int target = 0; target < options_.cluster_count; ++target) {
        if (target == source ||
            state.cameras_per_cluster[source] - 1 < minimum_cameras ||
            state.cameras_per_cluster[target] + 1 > maximum_cameras) {
          continue;
        }
        Objective candidate = ScoreCameraMove(
            graph, options_, state, camera, source, target);
        if (IsBetter(candidate, best_objective)) {
          best_target = target;
          best_objective = std::move(candidate);
        }
      }
      if (best_target != source) {
        ApplyCameraMove(graph, options_, camera, source, best_target,
                        best_objective, state, camera_to_cluster);
        ++accepted_moves;
        changed = true;
      }
    }

    for (int first_camera = 0; first_camera < graph.camera_count;
         ++first_camera) {
      std::set<int> candidates;
      for (int point : graph.points_from_camera[first_camera]) {
        for (int second_camera : graph.cameras_from_point[point]) {
          if (camera_to_cluster[second_camera] !=
              camera_to_cluster[first_camera]) {
            candidates.insert(second_camera);
          }
        }
      }
      int tried = 0;
      for (int second_camera : candidates) {
        if (options_.max_swap_candidates_per_camera > 0 &&
            tried++ >= options_.max_swap_candidates_per_camera) {
          break;
        }
        const int first_cluster = camera_to_cluster[first_camera];
        const int second_cluster = camera_to_cluster[second_camera];
        Objective candidate = ScoreCameraSwap(
            graph, options_, state, first_camera, second_camera,
            first_cluster, second_cluster);
        if (IsBetter(candidate, state.objective)) {
          ApplyCameraSwap(graph, options_, first_camera, second_camera,
                          first_cluster, second_cluster, candidate, state,
                          camera_to_cluster);
          ++accepted_moves;
          changed = true;
          break;
        }
      }
    }
    if (!changed) {
      break;
    }
  }

  CameraPartition result;
  result.camera_to_cluster = std::move(camera_to_cluster);
  result.metrics = Evaluate(graph, result.camera_to_cluster,
                            options_.cluster_count,
                            options_.low_degree_limit);
  result.accepted_refinement_moves = accepted_moves;
  return result;
}

PartitionMetrics CameraHypergraphPartitioner::Evaluate(
    const BipartiteCameraPointGraph& graph,
    const std::vector<int>& camera_to_cluster,
    int cluster_count,
    int low_degree_limit) {
  if (camera_to_cluster.size() != static_cast<std::size_t>(graph.camera_count)) {
    throw std::invalid_argument("camera assignment has the wrong size");
  }
  if (cluster_count <= 0 || low_degree_limit < 2) {
    throw std::invalid_argument("invalid metric dimensions");
  }

  PartitionMetrics metrics;
  metrics.low_degree_point_counts.assign(low_degree_limit, 0);
  metrics.cameras_per_cluster.assign(cluster_count, 0);
  metrics.points_per_cluster.assign(cluster_count, 0);
  std::vector<int> point_cluster_degree(graph.point_count * cluster_count, 0);
  for (int camera = 0; camera < graph.camera_count; ++camera) {
    const int cluster = camera_to_cluster[camera];
    if (cluster < 0 || cluster >= cluster_count) {
      throw std::invalid_argument("camera assignment contains invalid cluster");
    }
    ++metrics.cameras_per_cluster[cluster];
    for (int point : graph.points_from_camera[camera]) {
      ++point_cluster_degree[point * cluster_count + cluster];
    }
  }

  for (int point = 0; point < graph.point_count; ++point) {
    int cluster_occurrences = 0;
    for (int cluster = 0; cluster < cluster_count; ++cluster) {
      const int degree = point_cluster_degree[point * cluster_count + cluster];
      if (degree == 0) {
        continue;
      }
      ++cluster_occurrences;
      ++metrics.points_per_cluster[cluster];
      if (degree < low_degree_limit) {
        ++metrics.low_degree_point_counts[degree];
      }
    }
    metrics.copied_point_count += std::max(0, cluster_occurrences - 1);
  }
  return metrics;
}

}  // namespace bundle_palm

extern "C" int cluster_cameras_hypergraph(
    int cluster_count,
    int camera_count,
    int point_count,
    double camera_balance_slack,
    const std::vector<int>& camera_indices,
    const std::vector<int>& point_indices,
    std::vector<int>& camera_to_cluster_out) {
  try {
    const auto graph =
        bundle_palm::BipartiteCameraPointGraph::FromObservations(
            camera_count, point_count, camera_indices, point_indices);
    bundle_palm::PartitioningOptions options;
    options.cluster_count = cluster_count;
    options.camera_balance_slack = camera_balance_slack;
    camera_to_cluster_out =
        bundle_palm::CameraHypergraphPartitioner(options)
            .Partition(graph)
            .camera_to_cluster;
    return 0;
  } catch (const std::exception&) {
    camera_to_cluster_out.clear();
    return 1;
  }
}