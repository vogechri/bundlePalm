#include "scalable_landmark_partitioning.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace bundle_palm {
namespace {

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
  int largest_landmark = 0;
  for (int landmark = 0; landmark < graph.point_count; ++landmark) {
    const int weight = LandmarkWeight(graph, landmark);
    residual_count += weight;
    largest_landmark = std::max(largest_landmark, weight);
  }
  const double target = static_cast<double>(residual_count) /
                        static_cast<double>(options.cluster_count);
  return {
      static_cast<int>(std::floor(
          target * (1.0 - options.residual_balance_slack))),
      std::max(largest_landmark,
               static_cast<int>(std::ceil(
                   target * (1.0 + options.residual_balance_slack))))};
}

struct State {
  State(const BipartiteCameraPointGraph& graph, int cluster_count)
      : camera_cluster_degree(graph.camera_count * cluster_count, 0),
        residuals_per_cluster(cluster_count, 0),
        cameras_per_cluster(cluster_count, 0) {}

  void SetLandmarkOrder(const std::vector<int>& order) {
    landmark_by_rank = order;
    rank_by_landmark.resize(order.size());
    for (int rank = 0; rank < static_cast<int>(order.size()); ++rank) {
      rank_by_landmark[order[rank]] = rank;
    }
    const std::size_t word_count = (order.size() + 63) / 64;
    landmark_rank_bits.assign(
        residuals_per_cluster.size(), std::vector<std::uint64_t>(word_count));
  }

  void MoveLandmarkRank(int landmark, int source, int target) {
    const int rank = rank_by_landmark[landmark];
    const std::size_t word = static_cast<std::size_t>(rank) / 64;
    const std::uint64_t bit = std::uint64_t{1} << (rank % 64);
    if (source >= 0) {
      landmark_rank_bits[source][word] &= ~bit;
    }
    landmark_rank_bits[target][word] |= bit;
  }

  std::vector<int> camera_cluster_degree;
  std::vector<int> residuals_per_cluster;
  std::vector<int> cameras_per_cluster;
  std::vector<int> landmark_by_rank;
  std::vector<int> rank_by_landmark;
  std::vector<std::vector<std::uint64_t>> landmark_rank_bits;
};

struct MoveDelta {
  int copied_cameras = 0;
  int severe_weak = 0;
  int weak = 0;
  std::int64_t weak_penalty = 0;
  int target_camera_count = 0;
  int degree_one_two = 0;
  int degree_three = 0;
  int degree_four_nine = 0;
  int moderate_weak = 0;
  int maximum_camera_count = 0;
  std::int64_t low_support_penalty = 0;
};

struct SourceMoveDelta {
  MoveDelta delta;
  int removed_cameras = 0;
};

std::int64_t WeakPenalty(int degree, int limit) {
  if (degree <= 0 || degree >= limit) {
    return 0;
  }
  const std::int64_t deficit = limit - degree;
  return deficit * deficit * deficit;
}

int IsWeak(int degree, int limit) {
  return degree > 0 && degree < limit;
}

int IsSevereWeak(int degree, int limit) {
  return degree > 0 && degree < limit / 2;
}

std::int64_t LowSupportPenalty(int degree) {
  if (degree <= 0 || degree >= 10) {
    return 0;
  }
  if (degree == 1) {
    return 10000;
  }
  if (degree == 2) {
    return 3000;
  }
  const std::int64_t deficit = 10 - degree;
  return deficit * deficit * deficit;
}

int InDegreeRange(int degree, int first, int last) {
  return degree >= first && degree <= last;
}

bool BetterDelta(const MoveDelta& left, const MoveDelta& right,
                 ScalableLandmarkObjective objective) {
  if (objective == ScalableLandmarkObjective::kStability) {
    if (left.severe_weak != right.severe_weak) {
      return left.severe_weak < right.severe_weak;
    }
    if (left.low_support_penalty != right.low_support_penalty) {
      return left.low_support_penalty < right.low_support_penalty;
    }
    if (left.copied_cameras != right.copied_cameras) {
      return left.copied_cameras < right.copied_cameras;
    }
    if (left.maximum_camera_count != right.maximum_camera_count) {
      return left.maximum_camera_count < right.maximum_camera_count;
    }
    if (left.moderate_weak != right.moderate_weak) {
      return left.moderate_weak < right.moderate_weak;
    }
    if (left.weak_penalty != right.weak_penalty) {
      return left.weak_penalty < right.weak_penalty;
    }
    return left.target_camera_count < right.target_camera_count;
  }
  if (left.severe_weak != right.severe_weak) {
    return left.severe_weak < right.severe_weak;
  }
  if (left.copied_cameras != right.copied_cameras) {
    return left.copied_cameras < right.copied_cameras;
  }
  if (left.weak != right.weak) {
    return left.weak < right.weak;
  }
  if (left.weak_penalty != right.weak_penalty) {
    return left.weak_penalty < right.weak_penalty;
  }
  return left.target_camera_count < right.target_camera_count;
}

void AccumulateDelta(const MoveDelta& delta, MoveDelta& total) {
  total.copied_cameras += delta.copied_cameras;
  total.severe_weak += delta.severe_weak;
  total.weak += delta.weak;
  total.weak_penalty += delta.weak_penalty;
  total.degree_one_two += delta.degree_one_two;
  total.degree_three += delta.degree_three;
  total.degree_four_nine += delta.degree_four_nine;
  total.moderate_weak += delta.moderate_weak;
  total.low_support_penalty += delta.low_support_penalty;
}

int MaximumCameraCount(const State& state) {
  return *std::max_element(state.cameras_per_cluster.begin(),
                           state.cameras_per_cluster.end());
}

SourceMoveDelta ScoreMoveSource(
    const BipartiteCameraPointGraph& graph,
    const LandmarkPartitioningOptions& options,
    const State& state,
    int landmark,
    int source,
  ScalableLandmarkObjective objective) {
  SourceMoveDelta source_delta;
  const int multiplicity = PointMultiplicity(graph, landmark);
  for (int camera : graph.cameras_from_point[landmark]) {
    const int source_offset = camera * options.cluster_count + source;
    const int old_source = state.camera_cluster_degree[source_offset];
    const int new_source = old_source - multiplicity;
    source_delta.delta.copied_cameras +=
        (new_source > 0) - (old_source > 0);
    source_delta.delta.severe_weak +=
      IsSevereWeak(new_source, options.weak_camera_degree_limit) -
      IsSevereWeak(old_source, options.weak_camera_degree_limit);
    source_delta.delta.weak +=
      IsWeak(new_source, options.weak_camera_degree_limit) -
      IsWeak(old_source, options.weak_camera_degree_limit);
    source_delta.delta.weak_penalty +=
      WeakPenalty(new_source, options.weak_camera_degree_limit) -
      WeakPenalty(old_source, options.weak_camera_degree_limit);
    if (objective == ScalableLandmarkObjective::kStability) {
      source_delta.delta.low_support_penalty +=
        LowSupportPenalty(new_source) - LowSupportPenalty(old_source);
      source_delta.delta.degree_one_two +=
        InDegreeRange(new_source, 1, 2) -
        InDegreeRange(old_source, 1, 2);
      source_delta.delta.degree_three +=
        (new_source == 3) - (old_source == 3);
      source_delta.delta.degree_four_nine +=
        InDegreeRange(new_source, 4, 9) -
        InDegreeRange(old_source, 4, 9);
      source_delta.delta.moderate_weak +=
        InDegreeRange(new_source, 10,
              options.weak_camera_degree_limit - 1) -
        InDegreeRange(old_source, 10,
              options.weak_camera_degree_limit - 1);
    }
    source_delta.removed_cameras += new_source == 0;
  }
  return source_delta;
}

  MoveDelta ScoreMoveTarget(
    const BipartiteCameraPointGraph& graph,
    const LandmarkPartitioningOptions& options,
    const State& state,
    int landmark,
    int source,
    int target,
    ScalableLandmarkObjective objective,
    const SourceMoveDelta& source_delta) {
  MoveDelta delta = source_delta.delta;
  const int multiplicity = PointMultiplicity(graph, landmark);
  int new_target_cameras = 0;
  for (int camera : graph.cameras_from_point[landmark]) {
    const int target_offset = camera * options.cluster_count + target;
    const int old_target = state.camera_cluster_degree[target_offset];
    const int new_target = old_target + multiplicity;
    delta.copied_cameras += (new_target > 0) - (old_target > 0);
    delta.severe_weak +=
      IsSevereWeak(new_target, options.weak_camera_degree_limit) -
      IsSevereWeak(old_target, options.weak_camera_degree_limit);
    delta.weak += IsWeak(new_target, options.weak_camera_degree_limit) -
            IsWeak(old_target, options.weak_camera_degree_limit);
    delta.weak_penalty +=
      WeakPenalty(new_target, options.weak_camera_degree_limit) -
      WeakPenalty(old_target, options.weak_camera_degree_limit);
    if (objective == ScalableLandmarkObjective::kStability) {
      delta.low_support_penalty +=
        LowSupportPenalty(new_target) - LowSupportPenalty(old_target);
      delta.degree_one_two += InDegreeRange(new_target, 1, 2) -
                  InDegreeRange(old_target, 1, 2);
      delta.degree_three += (new_target == 3) - (old_target == 3);
      delta.degree_four_nine += InDegreeRange(new_target, 4, 9) -
                  InDegreeRange(old_target, 4, 9);
      delta.moderate_weak +=
        InDegreeRange(new_target, 10,
              options.weak_camera_degree_limit - 1) -
        InDegreeRange(old_target, 10,
              options.weak_camera_degree_limit - 1);
    }
    new_target_cameras += old_target == 0;
  }
  delta.target_camera_count =
      state.cameras_per_cluster[target] + new_target_cameras;
  if (objective == ScalableLandmarkObjective::kStability) {
    const int current_maximum = MaximumCameraCount(state);
    int candidate_maximum = 0;
    for (int cluster = 0; cluster < options.cluster_count; ++cluster) {
      int camera_count = state.cameras_per_cluster[cluster];
      if (cluster == source) {
        camera_count -= source_delta.removed_cameras;
      }
      if (cluster == target) {
        camera_count += new_target_cameras;
      }
      candidate_maximum = std::max(candidate_maximum, camera_count);
    }
    delta.maximum_camera_count = candidate_maximum - current_maximum;
  }
  return delta;
}

MoveDelta ScoreMove(const BipartiteCameraPointGraph& graph,
                    const LandmarkPartitioningOptions& options,
                    const State& state,
                    int landmark,
                    int source,
                    int target,
                    ScalableLandmarkObjective objective) {
  MoveDelta delta;
  const int multiplicity = PointMultiplicity(graph, landmark);
  int new_target_cameras = 0;
  int removed_source_cameras = 0;
  for (int camera : graph.cameras_from_point[landmark]) {
    const int source_offset = camera * options.cluster_count + source;
    const int target_offset = camera * options.cluster_count + target;
    const int old_source = state.camera_cluster_degree[source_offset];
    const int old_target = state.camera_cluster_degree[target_offset];
    const int new_source = old_source - multiplicity;
    const int new_target = old_target + multiplicity;
    delta.copied_cameras += (new_source > 0) - (old_source > 0) +
                            (new_target > 0) - (old_target > 0);
    delta.severe_weak += IsSevereWeak(new_source, options.weak_camera_degree_limit) -
                         IsSevereWeak(old_source, options.weak_camera_degree_limit) +
                         IsSevereWeak(new_target, options.weak_camera_degree_limit) -
                         IsSevereWeak(old_target, options.weak_camera_degree_limit);
    delta.weak += IsWeak(new_source, options.weak_camera_degree_limit) -
                  IsWeak(old_source, options.weak_camera_degree_limit) +
                  IsWeak(new_target, options.weak_camera_degree_limit) -
                  IsWeak(old_target, options.weak_camera_degree_limit);
    delta.weak_penalty +=
        WeakPenalty(new_source, options.weak_camera_degree_limit) -
        WeakPenalty(old_source, options.weak_camera_degree_limit) +
        WeakPenalty(new_target, options.weak_camera_degree_limit) -
        WeakPenalty(old_target, options.weak_camera_degree_limit);
    if (objective == ScalableLandmarkObjective::kStability) {
      delta.low_support_penalty +=
          LowSupportPenalty(new_source) - LowSupportPenalty(old_source) +
          LowSupportPenalty(new_target) - LowSupportPenalty(old_target);
      delta.degree_one_two += InDegreeRange(new_source, 1, 2) -
                  InDegreeRange(old_source, 1, 2) +
                  InDegreeRange(new_target, 1, 2) -
                  InDegreeRange(old_target, 1, 2);
      delta.degree_three += (new_source == 3) - (old_source == 3) +
                (new_target == 3) - (old_target == 3);
      delta.degree_four_nine += InDegreeRange(new_source, 4, 9) -
                  InDegreeRange(old_source, 4, 9) +
                  InDegreeRange(new_target, 4, 9) -
                  InDegreeRange(old_target, 4, 9);
      delta.moderate_weak +=
        InDegreeRange(new_source, 10,
              options.weak_camera_degree_limit - 1) -
        InDegreeRange(old_source, 10,
              options.weak_camera_degree_limit - 1) +
        InDegreeRange(new_target, 10,
              options.weak_camera_degree_limit - 1) -
        InDegreeRange(old_target, 10,
              options.weak_camera_degree_limit - 1);
    }
    new_target_cameras += old_target == 0;
    removed_source_cameras += new_source == 0;
  }
  delta.target_camera_count =
      state.cameras_per_cluster[target] + new_target_cameras;
  if (objective == ScalableLandmarkObjective::kStability) {
    const int current_maximum = *std::max_element(
        state.cameras_per_cluster.begin(), state.cameras_per_cluster.end());
    int candidate_maximum = 0;
    for (int cluster = 0; cluster < options.cluster_count; ++cluster) {
      int camera_count = state.cameras_per_cluster[cluster];
      if (cluster == source) {
        camera_count -= removed_source_cameras;
      }
      if (cluster == target) {
        camera_count += new_target_cameras;
      }
      candidate_maximum = std::max(candidate_maximum, camera_count);
    }
    delta.maximum_camera_count = candidate_maximum - current_maximum;
  }
  return delta;
}

void ApplyMove(const BipartiteCameraPointGraph& graph,
               int cluster_count,
               int landmark,
               int source,
               int target,
               State& state,
               std::vector<int>& landmark_to_cluster) {
  const int multiplicity = PointMultiplicity(graph, landmark);
  const int weight = LandmarkWeight(graph, landmark);
  for (int camera : graph.cameras_from_point[landmark]) {
    if (source >= 0) {
      const int source_offset = camera * cluster_count + source;
      if (state.camera_cluster_degree[source_offset] == multiplicity) {
        --state.cameras_per_cluster[source];
      }
      state.camera_cluster_degree[source_offset] -= multiplicity;
    }
    const int target_offset = camera * cluster_count + target;
    if (state.camera_cluster_degree[target_offset] == 0) {
      ++state.cameras_per_cluster[target];
    }
    state.camera_cluster_degree[target_offset] += multiplicity;
  }
  if (source >= 0) {
    state.residuals_per_cluster[source] -= weight;
  }
  state.residuals_per_cluster[target] += weight;
  state.MoveLandmarkRank(landmark, source, target);
  landmark_to_cluster[landmark] = target;
}

class ScalableProgressTrace {
 public:
  ScalableProgressTrace(const BipartiteCameraPointGraph& graph,
                        const LandmarkPartitioningOptions& options)
      : graph_(graph),
        options_(options),
        enabled_(TraceEnabled()),
        interval_(TraceInterval()),
        start_(std::chrono::steady_clock::now()),
        last_report_(start_) {}

  void Begin(const char* phase, int round = -1, int pass = -1) {
    phase_ = phase;
    round_ = round;
    pass_ = pass;
    work_units_ = 0;
  }

  void Tick(const State& state, int accepted_moves,
            std::int64_t work_units = 1) {
    work_units_ += work_units;
    if (!enabled_ ||
      (interval_.count() > 0 && (++ticks_since_check_ & 1023) != 0)) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report_ < interval_) {
      return;
    }
    Report(state, accepted_moves, now);
  }

 private:
  static bool TraceEnabled() {
    const char* setting = std::getenv("BUNDLE_PALM_PARTITION_TRACE");
    return setting == nullptr || std::strcmp(setting, "0") != 0;
  }

  static std::chrono::seconds TraceInterval() {
    const char* setting =
        std::getenv("BUNDLE_PALM_PARTITION_TRACE_INTERVAL_SECONDS");
    return std::chrono::seconds(
        setting == nullptr ? 5 : std::max(0, std::atoi(setting)));
  }

  void Report(const State& state, int accepted_moves,
              std::chrono::steady_clock::time_point now) {
    std::int64_t weak = 0;
    std::int64_t severe = 0;
    std::int64_t degree_one_to_nine = 0;
    std::int64_t copied_cameras = 0;
    for (int camera = 0; camera < graph_.camera_count; ++camera) {
      int cluster_incidence_count = 0;
      for (int cluster = 0; cluster < options_.cluster_count; ++cluster) {
        const int degree = state.camera_cluster_degree[
            camera * options_.cluster_count + cluster];
        weak += IsWeak(degree, options_.weak_camera_degree_limit);
        severe += IsSevereWeak(degree, options_.weak_camera_degree_limit);
        degree_one_to_nine += degree > 0 && degree < 10;
        cluster_incidence_count += degree > 0;
      }
      copied_cameras += std::max(0, cluster_incidence_count - 1);
    }
    const auto [minimum_residual, maximum_residual] = std::minmax_element(
        state.residuals_per_cluster.begin(),
        state.residuals_per_cluster.end());
    const int maximum_cameras = *std::max_element(
        state.cameras_per_cluster.begin(), state.cameras_per_cluster.end());
    const double elapsed = std::chrono::duration<double>(now - start_).count();
    std::cerr << "scalable landmark progress: " << elapsed << " s, phase "
              << phase_;
    if (round_ >= 0) {
      std::cerr << ", round " << round_;
    }
    if (pass_ >= 0) {
      std::cerr << ", pass " << pass_;
    }
    std::cerr << ", work " << work_units_
              << ", accepted moves " << accepted_moves
              << ", severe " << severe
              << ", degree 1..9 " << degree_one_to_nine
              << ", weak " << weak
              << ", camera copies " << copied_cameras
              << ", max cameras " << maximum_cameras
              << ", residual range [" << *minimum_residual << ", "
              << *maximum_residual << "]\n";
    last_report_ = now;
  }

  const BipartiteCameraPointGraph& graph_;
  const LandmarkPartitioningOptions& options_;
  bool enabled_;
  std::chrono::seconds interval_;
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point last_report_;
  const char* phase_ = "initialization";
  int round_ = -1;
  int pass_ = -1;
  std::int64_t work_units_ = 0;
  std::uint64_t ticks_since_check_ = 0;
};

}  // namespace

ScalableLandmarkPartitioner::ScalableLandmarkPartitioner(
  LandmarkPartitioningOptions options,
  ScalableLandmarkObjective objective)
  : options_(std::move(options)), objective_(objective) {}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-reorder-blocks-and-partition")))
#endif
LandmarkPartition ScalableLandmarkPartitioner::Partition(
    const BipartiteCameraPointGraph& graph) const {
  if (options_.cluster_count <= 0 ||
      options_.cluster_count > graph.point_count) {
    throw std::invalid_argument("invalid cluster count");
  }
  if (options_.weak_camera_degree_limit <= 0 ||
      options_.residual_balance_slack < 0.0) {
    throw std::invalid_argument("invalid scalable partitioning options");
  }

  const auto [minimum_residuals, maximum_residuals] =
      ResidualBounds(graph, options_);
  PartitioningOptions camera_options;
  camera_options.cluster_count = options_.cluster_count;
  camera_options.low_degree_limit = options_.weak_camera_degree_limit;
  camera_options.camera_balance_slack = 0.0;
  camera_options.max_refinement_passes = 2;
  const CameraPartition camera_partition =
      CameraHypergraphPartitioner(camera_options).Partition(graph);

  State state(graph, options_.cluster_count);
  ScalableProgressTrace progress(graph, options_);
  std::vector<int> assignment(graph.point_count, -1);
  std::vector<int> order(graph.point_count);
  std::iota(order.begin(), order.end(), 0);
  std::vector<int> vote_margin(graph.point_count, 0);
  std::vector<int> votes(options_.cluster_count, 0);
  for (int landmark = 0; landmark < graph.point_count; ++landmark) {
    std::fill(votes.begin(), votes.end(), 0);
    for (int camera : graph.cameras_from_point[landmark]) {
      ++votes[camera_partition.camera_to_cluster[camera]];
    }
    const auto best = std::max_element(votes.begin(), votes.end());
    const int best_votes = *best;
    *best = -1;
    vote_margin[landmark] =
        best_votes - *std::max_element(votes.begin(), votes.end());
  }
  std::stable_sort(order.begin(), order.end(),
                   [&graph, &vote_margin](int left, int right) {
    if (vote_margin[left] != vote_margin[right]) {
      return vote_margin[left] > vote_margin[right];
    }
    return graph.cameras_from_point[left].size() >
           graph.cameras_from_point[right].size();
  });
  state.SetLandmarkOrder(order);

  progress.Begin("initialization");
  for (int landmark : order) {
    const int weight = LandmarkWeight(graph, landmark);
    std::fill(votes.begin(), votes.end(), 0);
    for (int camera : graph.cameras_from_point[landmark]) {
      ++votes[camera_partition.camera_to_cluster[camera]];
    }
    int best_cluster = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    int best_new_cameras = std::numeric_limits<int>::max();
    for (int cluster = 0; cluster < options_.cluster_count; ++cluster) {
      if (state.residuals_per_cluster[cluster] + weight > maximum_residuals) {
        continue;
      }
      int present_cameras = 0;
      int camera_strength = 0;
      for (int camera : graph.cameras_from_point[landmark]) {
        const int degree = state.camera_cluster_degree[
            camera * options_.cluster_count + cluster];
        present_cameras += degree > 0;
        camera_strength += degree;
      }
      const int new_cameras =
          static_cast<int>(graph.cameras_from_point[landmark].size()) -
          present_cameras;
        const double capacity_factor =
          1.0 - static_cast<double>(state.residuals_per_cluster[cluster]) /
              static_cast<double>(maximum_residuals);
        const double score =
          (1000.0 * votes[cluster] + present_cameras +
           0.01 * camera_strength) * capacity_factor;
      if (best_cluster < 0 || score > best_score ||
          (score == best_score && new_cameras < best_new_cameras) ||
          (score == best_score && new_cameras == best_new_cameras &&
           state.residuals_per_cluster[cluster] <
               state.residuals_per_cluster[best_cluster])) {
        best_cluster = cluster;
        best_score = score;
        best_new_cameras = new_cameras;
      }
    }
    if (best_cluster < 0) {
      best_cluster = static_cast<int>(std::min_element(
          state.residuals_per_cluster.begin(),
          state.residuals_per_cluster.end()) -
          state.residuals_per_cluster.begin());
    }
    ApplyMove(graph, options_.cluster_count, landmark, -1, best_cluster,
              state, assignment);
    progress.Tick(state, 0);
  }

  int accepted_moves = 0;
  for (int pass = 0; pass < options_.max_refinement_passes; ++pass) {
    progress.Begin("refinement", -1, pass);
    bool changed = false;
    for (int landmark : order) {
      const int source = assignment[landmark];
      const int weight = LandmarkWeight(graph, landmark);
      if (state.residuals_per_cluster[source] - weight < minimum_residuals) {
        progress.Tick(state, accepted_moves);
        continue;
      }
      const SourceMoveDelta source_delta = ScoreMoveSource(
          graph, options_, state, landmark, source, objective_);
      int best_target = source;
      MoveDelta best_delta;
      for (int target = 0; target < options_.cluster_count; ++target) {
        if (target == source ||
            state.residuals_per_cluster[target] + weight > maximum_residuals) {
          continue;
        }
        const MoveDelta candidate = ScoreMoveTarget(
          graph, options_, state, landmark, source, target, objective_,
          source_delta);
        if (best_target == source ||
          BetterDelta(candidate, best_delta, objective_)) {
          best_target = target;
          best_delta = candidate;
        }
      }
      const MoveDelta unchanged;
        if (best_target != source &&
          BetterDelta(best_delta, unchanged, objective_)) {
        ApplyMove(graph, options_.cluster_count, landmark, source, best_target,
                  state, assignment);
        ++accepted_moves;
        changed = true;
      }
      progress.Tick(state, accepted_moves);
    }
    if (!changed) {
      break;
    }
  }

  progress.Begin("home consolidation");
  for (int camera = 0; camera < graph.camera_count; ++camera) {
    const int target = camera_partition.camera_to_cluster[camera];
    for (int source = 0; source < options_.cluster_count; ++source) {
      const int degree = state.camera_cluster_degree[
          camera * options_.cluster_count + source];
      progress.Tick(state, accepted_moves);
      if (source == target || degree <= 0 ||
          degree >= options_.weak_camera_degree_limit / 2) {
        continue;
      }
      std::vector<int> landmarks;
      int moved_weight = 0;
      for (int landmark : graph.points_from_camera[camera]) {
        if (assignment[landmark] == source) {
          landmarks.push_back(landmark);
          moved_weight += LandmarkWeight(graph, landmark);
        }
      }
      if (landmarks.empty() ||
          state.residuals_per_cluster[source] - moved_weight <
              minimum_residuals ||
          state.residuals_per_cluster[target] + moved_weight >
              maximum_residuals) {
        continue;
      }

      MoveDelta batch_delta;
      const int initial_maximum_cameras = MaximumCameraCount(state);
      for (int landmark : landmarks) {
        const MoveDelta delta = ScoreMove(
          graph, options_, state, landmark, source, target, objective_);
        AccumulateDelta(delta, batch_delta);
        ApplyMove(graph, options_.cluster_count, landmark, source, target,
                  state, assignment);
      }
      batch_delta.target_camera_count = state.cameras_per_cluster[target];
      batch_delta.maximum_camera_count =
          MaximumCameraCount(state) - initial_maximum_cameras;
      const MoveDelta unchanged;
      if (BetterDelta(batch_delta, unchanged, objective_)) {
        accepted_moves += static_cast<int>(landmarks.size());
      } else {
        for (auto landmark = landmarks.rbegin();
             landmark != landmarks.rend(); ++landmark) {
          ApplyMove(graph, options_.cluster_count, *landmark, target, source,
                    state, assignment);
        }
      }
      progress.Tick(state, accepted_moves);
    }
  }

  struct ReinforcementMove {
    int landmark;
    int source;
  };
  const int severe_degree_limit = options_.weak_camera_degree_limit / 2;
  for (int recovery_round = 0; recovery_round < 3; ++recovery_round) {
  for (int repair_pass = 0; repair_pass < 8; ++repair_pass) {
    progress.Begin("reinforcement", recovery_round, repair_pass);
    bool repaired = false;
    for (int camera = 0; camera < graph.camera_count; ++camera) {
      for (int target = 0; target < options_.cluster_count; ++target) {
        const int target_offset = camera * options_.cluster_count + target;
        const int initial_degree = state.camera_cluster_degree[target_offset];
        progress.Tick(state, accepted_moves);
        if (initial_degree <= 0 || initial_degree >= severe_degree_limit) {
          continue;
        }

        MoveDelta batch_delta;
        const int initial_maximum_cameras = MaximumCameraCount(state);
        std::vector<ReinforcementMove> moves;
        while (state.camera_cluster_degree[target_offset] <
               severe_degree_limit) {
          int best_landmark = -1;
          int best_source = -1;
          MoveDelta best_delta;
          for (int landmark : graph.points_from_camera[camera]) {
            const int source = assignment[landmark];
            if (source == target) {
              continue;
            }
            const int multiplicity = PointMultiplicity(graph, landmark);
            const int donor_degree = state.camera_cluster_degree[
                camera * options_.cluster_count + source];
            const int new_donor_degree = donor_degree - multiplicity;
            if (new_donor_degree > 0 &&
                new_donor_degree < severe_degree_limit) {
              continue;
            }
            const int weight = LandmarkWeight(graph, landmark);
            if (state.residuals_per_cluster[source] - weight <
                    minimum_residuals ||
                state.residuals_per_cluster[target] + weight >
                    maximum_residuals) {
              continue;
            }
            const MoveDelta candidate = ScoreMove(
              graph, options_, state, landmark, source, target, objective_);
            if (best_landmark < 0 ||
              BetterDelta(candidate, best_delta, objective_)) {
              best_landmark = landmark;
              best_source = source;
              best_delta = candidate;
            }
          }
          if (best_landmark < 0) {
            break;
          }
          AccumulateDelta(best_delta, batch_delta);
          moves.push_back({best_landmark, best_source});
          ApplyMove(graph, options_.cluster_count, best_landmark, best_source,
                    target, state, assignment);
        }

        batch_delta.target_camera_count = state.cameras_per_cluster[target];
        batch_delta.maximum_camera_count =
          MaximumCameraCount(state) - initial_maximum_cameras;
        const MoveDelta unchanged;
        if (state.camera_cluster_degree[target_offset] >=
                severe_degree_limit &&
            BetterDelta(batch_delta, unchanged, objective_)) {
          accepted_moves += static_cast<int>(moves.size());
          repaired = true;
        } else {
          for (auto move = moves.rbegin(); move != moves.rend(); ++move) {
            ApplyMove(graph, options_.cluster_count, move->landmark, target,
                      move->source, state, assignment);
          }
        }
        progress.Tick(state, accepted_moves);
      }
    }
    if (!repaired) {
      break;
    }
  }

  for (int evacuation_pass = 0; evacuation_pass < 4; ++evacuation_pass) {
    progress.Begin("evacuation", recovery_round, evacuation_pass);
    bool evacuated = false;
    for (int camera = 0; camera < graph.camera_count; ++camera) {
      for (int source = 0; source < options_.cluster_count; ++source) {
        const int source_offset = camera * options_.cluster_count + source;
        const int source_degree = state.camera_cluster_degree[source_offset];
        progress.Tick(state, accepted_moves);
        if (source_degree <= 0 || source_degree >= severe_degree_limit) {
          continue;
        }

        std::vector<int> landmarks;
        int moved_weight = 0;
        for (int landmark : graph.points_from_camera[camera]) {
          if (assignment[landmark] == source) {
            landmarks.push_back(landmark);
            moved_weight += LandmarkWeight(graph, landmark);
          }
        }
        if (landmarks.empty()) {
          continue;
        }

        int best_target = -1;
        MoveDelta best_batch_delta;
        std::vector<int> best_backfill;
        for (int target = 0; target < options_.cluster_count; ++target) {
          if (target == source ||
              state.camera_cluster_degree[
                  camera * options_.cluster_count + target] <
                  severe_degree_limit) {
            continue;
          }

          MoveDelta batch_delta;
          const int initial_maximum_cameras = MaximumCameraCount(state);
          for (int landmark : landmarks) {
            const MoveDelta delta = ScoreMove(
              graph, options_, state, landmark, source, target, objective_);
            AccumulateDelta(delta, batch_delta);
            ApplyMove(graph, options_.cluster_count, landmark, source, target,
                      state, assignment);
          }

          std::vector<int> backfill;
          while ((state.residuals_per_cluster[source] < minimum_residuals ||
                  state.residuals_per_cluster[target] > maximum_residuals) &&
                 backfill.size() < 32) {
            int best_landmark = -1;
            MoveDelta best_delta;
            int inspected = 0;
            const auto& rank_bits = state.landmark_rank_bits[target];
            bool inspection_limit_reached = false;
            for (std::size_t word_index = 0;
                 word_index < rank_bits.size() && !inspection_limit_reached;
                 ++word_index) {
              std::uint64_t bits = rank_bits[word_index];
              while (bits != 0) {
                const int bit = __builtin_ctzll(bits);
                bits &= bits - 1;
                const int rank = static_cast<int>(64 * word_index) + bit;
                const int landmark = state.landmark_by_rank[rank];
                if (std::find(graph.cameras_from_point[landmark].begin(),
                              graph.cameras_from_point[landmark].end(),
                              camera) != graph.cameras_from_point[landmark].end()) {
                  continue;
                }
                if (++inspected > 4096) {
                  inspection_limit_reached = true;
                  break;
                }
                const int weight = LandmarkWeight(graph, landmark);
                if (state.residuals_per_cluster[source] + weight >
                        maximum_residuals ||
                    state.residuals_per_cluster[target] - weight <
                        minimum_residuals) {
                  continue;
                }
                const MoveDelta candidate = ScoreMove(
                    graph, options_, state, landmark, target, source,
                    objective_);
                if (best_landmark < 0 ||
                    BetterDelta(candidate, best_delta, objective_)) {
                  best_landmark = landmark;
                  best_delta = candidate;
                }
              }
            }
            if (best_landmark < 0) {
              break;
            }
            AccumulateDelta(best_delta, batch_delta);
            backfill.push_back(best_landmark);
            ApplyMove(graph, options_.cluster_count, best_landmark, target,
                      source, state, assignment);
          }
          batch_delta.target_camera_count = state.cameras_per_cluster[target];
            batch_delta.maximum_camera_count =
              MaximumCameraCount(state) - initial_maximum_cameras;
          const bool balanced =
              state.residuals_per_cluster[source] >= minimum_residuals &&
              state.residuals_per_cluster[source] <= maximum_residuals &&
              state.residuals_per_cluster[target] >= minimum_residuals &&
              state.residuals_per_cluster[target] <= maximum_residuals;
          for (auto landmark = backfill.rbegin();
               landmark != backfill.rend(); ++landmark) {
            ApplyMove(graph, options_.cluster_count, *landmark, source, target,
                      state, assignment);
          }
          for (auto landmark = landmarks.rbegin();
               landmark != landmarks.rend(); ++landmark) {
            ApplyMove(graph, options_.cluster_count, *landmark, target, source,
                      state, assignment);
          }

          const MoveDelta unchanged;
            if (balanced &&
              BetterDelta(batch_delta, unchanged, objective_) &&
              (best_target < 0 ||
               BetterDelta(batch_delta, best_batch_delta, objective_))) {
            best_target = target;
            best_batch_delta = batch_delta;
            best_backfill = std::move(backfill);
          }
        }
        if (best_target < 0) {
          continue;
        }
        for (int landmark : landmarks) {
          ApplyMove(graph, options_.cluster_count, landmark, source,
                    best_target, state, assignment);
        }
        for (int landmark : best_backfill) {
          ApplyMove(graph, options_.cluster_count, landmark, best_target,
                    source, state, assignment);
        }
        accepted_moves += static_cast<int>(landmarks.size() +
                                           best_backfill.size());
        evacuated = true;
        progress.Tick(state, accepted_moves);
      }
    }
    if (!evacuated) {
      break;
    }
  }

  for (int cleanup_pass = 0;
       cleanup_pass < options_.max_refinement_passes; ++cleanup_pass) {
    progress.Begin("cleanup", recovery_round, cleanup_pass);
    bool changed = false;
    for (int landmark : order) {
      const int source = assignment[landmark];
      const int weight = LandmarkWeight(graph, landmark);
      if (state.residuals_per_cluster[source] - weight < minimum_residuals) {
        progress.Tick(state, accepted_moves);
        continue;
      }
      const SourceMoveDelta source_delta = ScoreMoveSource(
          graph, options_, state, landmark, source, objective_);
      int best_target = source;
      MoveDelta best_delta;
      for (int target = 0; target < options_.cluster_count; ++target) {
        if (target == source ||
            state.residuals_per_cluster[target] + weight > maximum_residuals) {
          continue;
        }
        const MoveDelta candidate = ScoreMoveTarget(
          graph, options_, state, landmark, source, target, objective_,
          source_delta);
        if (best_target == source ||
          BetterDelta(candidate, best_delta, objective_)) {
          best_target = target;
          best_delta = candidate;
        }
      }
      const MoveDelta unchanged;
        if (best_target != source &&
          BetterDelta(best_delta, unchanged, objective_)) {
        ApplyMove(graph, options_.cluster_count, landmark, source, best_target,
                  state, assignment);
        ++accepted_moves;
        changed = true;
      }
      progress.Tick(state, accepted_moves);
    }
    if (!changed) {
      break;
    }
  }
  }

  LandmarkPartition result;
  result.landmark_to_cluster = std::move(assignment);
  result.metrics = LandmarkPartitioner::Evaluate(
      graph, result.landmark_to_cluster, options_);
  result.accepted_refinement_moves = accepted_moves;
  return result;
}

}  // namespace bundle_palm

namespace {

int ClusterLandmarksScalable(
    int cluster_count,
    int camera_count,
    int landmark_count,
    int minimum_camera_landmarks,
    int max_refinement_passes,
    double residual_balance_slack,
    const std::vector<int>& camera_indices,
    const std::vector<int>& landmark_indices,
    std::vector<int>& landmark_to_cluster_out,
    bundle_palm::ScalableLandmarkObjective objective,
    const char* label) {
  try {
    const auto graph = bundle_palm::BipartiteCameraPointGraph::FromObservations(
        camera_count, landmark_count, camera_indices, landmark_indices);
    bundle_palm::LandmarkPartitioningOptions options;
    options.cluster_count = cluster_count;
    options.weak_camera_degree_limit = minimum_camera_landmarks;
    options.max_refinement_passes = max_refinement_passes;
    options.residual_balance_slack = residual_balance_slack;
    const auto result =
      bundle_palm::ScalableLandmarkPartitioner(options, objective).Partition(
        graph);
    landmark_to_cluster_out = result.landmark_to_cluster;
    std::cout << label << "\n";
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
              << *std::max_element(result.metrics.cameras_per_cluster.begin(),
                                   result.metrics.cameras_per_cluster.end())
              << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Scalable landmark partitioning failed: "
              << error.what() << "\n";
    landmark_to_cluster_out.clear();
    return 1;
  }
}

}  // namespace

extern "C" int cluster_landmarks_scalable(
    int cluster_count,
    int camera_count,
    int landmark_count,
    int minimum_camera_landmarks,
    int max_refinement_passes,
    double residual_balance_slack,
    const std::vector<int>& camera_indices,
    const std::vector<int>& landmark_indices,
    std::vector<int>& landmark_to_cluster_out) {
  return ClusterLandmarksScalable(
      cluster_count, camera_count, landmark_count, minimum_camera_landmarks,
      max_refinement_passes, residual_balance_slack, camera_indices,
      landmark_indices, landmark_to_cluster_out,
      bundle_palm::ScalableLandmarkObjective::kLegacy,
      "Scalable landmark partition");
}

extern "C" int cluster_landmarks_scalable_stable(
    int cluster_count,
    int camera_count,
    int landmark_count,
    int minimum_camera_landmarks,
    int max_refinement_passes,
    double residual_balance_slack,
    const std::vector<int>& camera_indices,
    const std::vector<int>& landmark_indices,
    std::vector<int>& landmark_to_cluster_out) {
  return ClusterLandmarksScalable(
      cluster_count, camera_count, landmark_count, minimum_camera_landmarks,
      max_refinement_passes, residual_balance_slack, camera_indices,
      landmark_indices, landmark_to_cluster_out,
      bundle_palm::ScalableLandmarkObjective::kStability,
      "Stability-focused scalable landmark partition");
}