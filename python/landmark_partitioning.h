#ifndef LANDMARK_PARTITIONING_H
#define LANDMARK_PARTITIONING_H

#include "camera_hypergraph_partitioning.h"

#include <cstdint>
#include <vector>

namespace bundle_palm {

struct LandmarkPartitioningOptions {
  int cluster_count = 1;
  int weak_camera_degree_limit = 20;
  double residual_balance_slack = 0.05;
  int max_refinement_passes = 10;
  int max_swap_candidates_per_landmark = 256;
  int repair_restart_interval = 1;
  std::int64_t max_repair_work_per_phase = 0;
  int hard_group_max_camera_count = 0;
  bool optimize_max_camera_count = false;
};

struct LandmarkPartitionMetrics {
  std::vector<std::int64_t> weak_camera_counts;
  std::int64_t copied_camera_count = 0;
  std::vector<int> residuals_per_cluster;
  std::vector<int> landmarks_per_cluster;
  std::vector<int> cameras_per_cluster;
  int minimum_residuals = 0;
  int maximum_residuals = 0;
  int residual_balance_violation = 0;
};

struct LandmarkPartition {
  std::vector<int> landmark_to_cluster;
  LandmarkPartitionMetrics metrics;
  int accepted_refinement_moves = 0;
};

class LandmarkPartitioner {
 public:
  explicit LandmarkPartitioner(LandmarkPartitioningOptions options);

  LandmarkPartition Partition(const BipartiteCameraPointGraph& graph) const;

  static LandmarkPartitionMetrics Evaluate(
      const BipartiteCameraPointGraph& graph,
      const std::vector<int>& landmark_to_cluster,
      const LandmarkPartitioningOptions& options);

 private:
  LandmarkPartitioningOptions options_;
};

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
    std::vector<int>& landmark_to_cluster_out);

#endif  // LANDMARK_PARTITIONING_H