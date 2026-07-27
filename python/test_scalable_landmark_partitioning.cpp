#include "scalable_landmark_partitioning.h"

#include <cassert>
#include <vector>

int main() {
  const auto graph = bundle_palm::BipartiteCameraPointGraph::FromObservations(
      4, 8,
      {0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3},
      {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7});
  bundle_palm::LandmarkPartitioningOptions options;
  options.cluster_count = 2;
  options.weak_camera_degree_limit = 3;
  options.residual_balance_slack = 0.0;
  options.max_refinement_passes = 2;

  const auto result =
      bundle_palm::ScalableLandmarkPartitioner(options).Partition(graph);
  assert(result.metrics.residual_balance_violation == 0);
  assert(result.metrics.residuals_per_cluster == std::vector<int>({8, 8}));
  assert(result.metrics.copied_camera_count == 0);

  const auto stable_result = bundle_palm::ScalableLandmarkPartitioner(
      options, bundle_palm::ScalableLandmarkObjective::kStability).Partition(
          graph);
  assert(stable_result.metrics.residual_balance_violation == 0);
  assert(stable_result.metrics.residuals_per_cluster ==
         std::vector<int>({8, 8}));
  assert(stable_result.metrics.copied_camera_count == 0);

  return 0;
}