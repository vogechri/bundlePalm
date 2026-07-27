#include "camera_hypergraph_partitioning.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using bundle_palm::BipartiteCameraPointGraph;
using bundle_palm::CameraHypergraphPartitioner;
using bundle_palm::PartitioningOptions;

void TestSeparatesTwoCameraCommunities() {
  const std::vector<int> cameras = {
      0, 1, 2, 0, 1,
      3, 4, 5, 4, 5,
  };
  const std::vector<int> points = {
      0, 0, 0, 1, 1,
      2, 2, 2, 3, 3,
  };
  const auto graph = BipartiteCameraPointGraph::FromObservations(
      6, 4, cameras, points);
  PartitioningOptions options;
  options.cluster_count = 2;

  const auto result = CameraHypergraphPartitioner(options).Partition(graph);

  assert(result.metrics.cameras_per_cluster == std::vector<int>({3, 3}));
  assert(result.metrics.copied_point_count == 0);
  assert(result.camera_to_cluster[0] == result.camera_to_cluster[1]);
  assert(result.camera_to_cluster[1] == result.camera_to_cluster[2]);
  assert(result.camera_to_cluster[3] == result.camera_to_cluster[4]);
  assert(result.camera_to_cluster[4] == result.camera_to_cluster[5]);
  assert(result.camera_to_cluster[0] != result.camera_to_cluster[3]);
}

void TestKeepsExactBalanceWhenDivisibilityRequiresOneExtraCamera() {
  const auto graph = BipartiteCameraPointGraph::FromObservations(
      5, 3,
      {0, 1, 1, 2, 3, 4},
      {0, 0, 1, 1, 2, 2});
  PartitioningOptions options;
  options.cluster_count = 2;

  const auto result = CameraHypergraphPartitioner(options).Partition(graph);
  auto cluster_sizes = result.metrics.cameras_per_cluster;
  std::sort(cluster_sizes.begin(), cluster_sizes.end());
  assert(cluster_sizes == std::vector<int>({2, 3}));
  assert(result.camera_to_cluster.size() == 5);
  for (int cluster : result.camera_to_cluster) {
    assert(cluster == 0 || cluster == 1);
  }
}

void TestDeduplicatesRepeatedObservations() {
  const auto graph = BipartiteCameraPointGraph::FromObservations(
      2, 1, {0, 0, 1}, {0, 0, 0});
  assert(graph.points_from_camera[0].size() == 1);
  assert(graph.cameras_from_point[0].size() == 2);

  const auto metrics = CameraHypergraphPartitioner::Evaluate(
      graph, {0, 0}, 1, 10);
  assert(metrics.low_degree_point_counts[2] == 1);
  assert(metrics.copied_point_count == 0);
}

void TestReportedMetricsMatchFullRecomputation() {
  const auto graph = BipartiteCameraPointGraph::FromObservations(
      8, 6,
      {0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 1, 6},
      {0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 4, 5});
  PartitioningOptions options;
  options.cluster_count = 2;
  options.max_refinement_passes = 5;

  const auto result = CameraHypergraphPartitioner(options).Partition(graph);
  const auto recomputed = CameraHypergraphPartitioner::Evaluate(
      graph, result.camera_to_cluster, options.cluster_count,
      options.low_degree_limit);
  assert(result.metrics.low_degree_point_counts ==
         recomputed.low_degree_point_counts);
  assert(result.metrics.copied_point_count == recomputed.copied_point_count);
  assert(result.metrics.cameras_per_cluster == recomputed.cameras_per_cluster);
  assert(result.metrics.points_per_cluster == recomputed.points_per_cluster);
}

void TestRejectsInvalidObservationIndices() {
  bool rejected = false;
  try {
    BipartiteCameraPointGraph::FromObservations(2, 1, {0, 2}, {0, 0});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void TestCInterfaceReturnsCameraAssignment() {
  std::vector<int> camera_to_cluster;
  const int status = cluster_cameras_hypergraph(
      2, 4, 2, 0.0,
      {0, 1, 2, 3}, {0, 0, 1, 1}, camera_to_cluster);
  assert(status == 0);
  assert(camera_to_cluster.size() == 4);
  assert(camera_to_cluster[0] == camera_to_cluster[1]);
  assert(camera_to_cluster[2] == camera_to_cluster[3]);
  assert(camera_to_cluster[0] != camera_to_cluster[2]);
}

}  // namespace

int main() {
  TestSeparatesTwoCameraCommunities();
  TestKeepsExactBalanceWhenDivisibilityRequiresOneExtraCamera();
  TestDeduplicatesRepeatedObservations();
  TestReportedMetricsMatchFullRecomputation();
  TestRejectsInvalidObservationIndices();
  TestCInterfaceReturnsCameraAssignment();
  std::cout << "camera hypergraph partitioning tests passed\n";
  return 0;
}