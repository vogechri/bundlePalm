#ifndef CAMERA_HYPERGRAPH_PARTITIONING_H
#define CAMERA_HYPERGRAPH_PARTITIONING_H

#include <cstdint>
#include <vector>

namespace bundle_palm {

struct BipartiteCameraPointGraph {
  int camera_count = 0;
  int point_count = 0;
  std::vector<std::vector<int>> points_from_camera;
  std::vector<std::vector<int>> cameras_from_point;
  std::vector<int> point_multiplicity;

  static BipartiteCameraPointGraph FromObservations(
      int camera_count,
      int point_count,
      const std::vector<int>& camera_indices,
      const std::vector<int>& point_indices);
};

struct PartitioningOptions {
  int cluster_count = 1;
  int low_degree_limit = 10;
  double camera_balance_slack = 0.0;
  int max_refinement_passes = 10;
  int max_swap_candidates_per_camera = 256;
};

struct PartitionMetrics {
  std::vector<std::int64_t> low_degree_point_counts;
  std::int64_t copied_point_count = 0;
  std::vector<int> cameras_per_cluster;
  std::vector<int> points_per_cluster;
};

struct CameraPartition {
  std::vector<int> camera_to_cluster;
  PartitionMetrics metrics;
  int accepted_refinement_moves = 0;
};

class CameraHypergraphPartitioner {
 public:
  explicit CameraHypergraphPartitioner(PartitioningOptions options);

  CameraPartition Partition(const BipartiteCameraPointGraph& graph) const;

  static PartitionMetrics Evaluate(
      const BipartiteCameraPointGraph& graph,
      const std::vector<int>& camera_to_cluster,
      int cluster_count,
      int low_degree_limit);

 private:
  PartitioningOptions options_;
};

}  // namespace bundle_palm

extern "C" int cluster_cameras_hypergraph(
  int cluster_count,
  int camera_count,
  int point_count,
  double camera_balance_slack,
  const std::vector<int>& camera_indices,
  const std::vector<int>& point_indices,
  std::vector<int>& camera_to_cluster_out);

#endif  // CAMERA_HYPERGRAPH_PARTITIONING_H