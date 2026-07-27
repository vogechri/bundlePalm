#include "landmark_partitioning.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

namespace {

using bundle_palm::BipartiteCameraPointGraph;
using bundle_palm::LandmarkPartitioner;
using bundle_palm::LandmarkPartitioningOptions;

void TestLandmarksAreExclusiveAndResidualsAreBalanced() {
  const auto graph = BipartiteCameraPointGraph::FromObservations(
      4, 8,
      {0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3},
      {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7});
  LandmarkPartitioningOptions options;
  options.cluster_count = 2;
  options.residual_balance_slack = 0.0;

  const auto result = LandmarkPartitioner(options).Partition(graph);
  assert(result.landmark_to_cluster.size() == 8);
  assert(result.metrics.residuals_per_cluster == std::vector<int>({8, 8}));
  assert(result.metrics.residual_balance_violation == 0);
  for (int cluster : result.landmark_to_cluster) {
    assert(cluster == 0 || cluster == 1);
  }
}

void TestGroupsLandmarksObservedByTheSameCameras() {
  const auto graph = BipartiteCameraPointGraph::FromObservations(
      4, 8,
      {0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3},
      {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7});
  LandmarkPartitioningOptions options;
  options.cluster_count = 2;
  options.weak_camera_degree_limit = 3;
  options.residual_balance_slack = 0.0;

  const auto result = LandmarkPartitioner(options).Partition(graph);
  for (int landmark = 1; landmark < 4; ++landmark) {
    assert(result.landmark_to_cluster[landmark] ==
           result.landmark_to_cluster[0]);
  }
  for (int landmark = 5; landmark < 8; ++landmark) {
    assert(result.landmark_to_cluster[landmark] ==
           result.landmark_to_cluster[4]);
  }
  assert(result.landmark_to_cluster[0] != result.landmark_to_cluster[4]);
  assert(result.metrics.copied_camera_count == 0);
  assert(std::all_of(result.metrics.weak_camera_counts.begin() + 1,
                     result.metrics.weak_camera_counts.end(),
                     [](std::int64_t count) { return count == 0; }));
}

void TestHardGroupsLowCameraSignatures() {
  const auto graph = BipartiteCameraPointGraph::FromObservations(
      4, 8,
      {0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3},
      {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7});
  LandmarkPartitioningOptions options;
  options.cluster_count = 2;
  options.weak_camera_degree_limit = 3;
  options.residual_balance_slack = 0.0;
  options.hard_group_max_camera_count = 4;

  const auto result = LandmarkPartitioner(options).Partition(graph);
  for (int landmark = 1; landmark < 4; ++landmark) {
    assert(result.landmark_to_cluster[landmark] ==
           result.landmark_to_cluster[0]);
  }
  for (int landmark = 5; landmark < 8; ++landmark) {
    assert(result.landmark_to_cluster[landmark] ==
           result.landmark_to_cluster[4]);
  }
  assert(result.landmark_to_cluster[0] != result.landmark_to_cluster[4]);
  assert(result.metrics.residual_balance_violation == 0);
}

void TestSplitsOversizedHardGroupsIntoCapacityChunks() {
  const auto graph = BipartiteCameraPointGraph::FromObservations(
      2, 8,
      {0, 0, 0, 0, 0, 0, 1, 1},
      {0, 1, 2, 3, 4, 5, 6, 7});
  LandmarkPartitioningOptions options;
  options.cluster_count = 2;
  options.residual_balance_slack = 0.0;
  options.hard_group_max_camera_count = 4;

  const auto result = LandmarkPartitioner(options).Partition(graph);
  for (int landmark = 1; landmark < 4; ++landmark) {
    assert(result.landmark_to_cluster[landmark] ==
           result.landmark_to_cluster[0]);
  }
  assert(result.landmark_to_cluster[4] ==
         result.landmark_to_cluster[5]);
  assert(result.landmark_to_cluster[0] !=
         result.landmark_to_cluster[4]);
  assert(result.metrics.residuals_per_cluster == std::vector<int>({4, 4}));
  assert(result.metrics.residual_balance_violation == 0);
}

void TestMetricsMatchFullEvaluation() {
  const auto graph = BipartiteCameraPointGraph::FromObservations(
      5, 6,
      {0, 1, 2, 0, 2, 3, 1, 3, 4, 0, 4},
      {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4});
  LandmarkPartitioningOptions options;
  options.cluster_count = 2;
  const auto result = LandmarkPartitioner(options).Partition(graph);
  const auto recomputed = LandmarkPartitioner::Evaluate(
      graph, result.landmark_to_cluster, options);
  assert(result.metrics.weak_camera_counts == recomputed.weak_camera_counts);
  assert(result.metrics.copied_camera_count == recomputed.copied_camera_count);
  assert(result.metrics.residuals_per_cluster ==
         recomputed.residuals_per_cluster);
  assert(result.metrics.landmarks_per_cluster ==
         recomputed.landmarks_per_cluster);
  assert(result.metrics.cameras_per_cluster == recomputed.cameras_per_cluster);
}

void TestOptionalMaximumCameraObjective() {
  BipartiteCameraPointGraph graph;
  graph.camera_count = 10;
  graph.point_count = 4;
  graph.cameras_from_point = {
      {0, 1}, {0, 1}, {2, 3, 4, 5}, {6, 7, 8, 9}};
  graph.points_from_camera = {
      {0, 1}, {0, 1}, {2}, {2}, {2}, {2}, {3}, {3}, {3}, {3}};
  graph.point_multiplicity = {2, 2, 1, 1};
  LandmarkPartitioningOptions baseline_options;
  baseline_options.cluster_count = 2;
  baseline_options.weak_camera_degree_limit = 2;
  baseline_options.residual_balance_slack = 0.0;

  const auto baseline = LandmarkPartitioner(baseline_options).Partition(graph);
  LandmarkPartitioningOptions balanced_options = baseline_options;
  balanced_options.optimize_max_camera_count = true;
  const auto balanced = LandmarkPartitioner(balanced_options).Partition(graph);

  assert(*std::max_element(balanced.metrics.cameras_per_cluster.begin(),
                           balanced.metrics.cameras_per_cluster.end()) <
         *std::max_element(baseline.metrics.cameras_per_cluster.begin(),
                           baseline.metrics.cameras_per_cluster.end()));
  assert(balanced.metrics.residual_balance_violation == 0);
  assert(balanced.metrics.weak_camera_counts ==
         baseline.metrics.weak_camera_counts);
  assert(balanced.metrics.copied_camera_count >
         baseline.metrics.copied_camera_count);
}

void TestCInterfaceReturnsLandmarkAssignment() {
  for (bool optimize_max_camera_count : {false, true}) {
    for (int restart_interval : {0, 1, 32}) {
      for (std::int64_t max_repair_work_per_phase : {0, 1}) {
        std::vector<int> assignment;
        const int status = cluster_landmarks_clean(
        2, 4, 4, 20, 10, restart_interval,
          max_repair_work_per_phase, 0,
          optimize_max_camera_count, 0.05,
          {0, 1, 0, 1, 2, 3, 2, 3},
          {0, 0, 1, 1, 2, 2, 3, 3}, assignment);
        assert(status == 0);
        assert(assignment.size() == 4);
        assert(assignment[0] == assignment[1]);
        assert(assignment[2] == assignment[3]);
        assert(assignment[0] != assignment[2]);
      }
    }
  }

  std::vector<int> assignment;
  assert(cluster_landmarks_clean(
      2, 4, 4, 20, 10, -1, 0, 0, false, 0.05,
      {0, 1, 0, 1, 2, 3, 2, 3},
      {0, 0, 1, 1, 2, 2, 3, 3}, assignment) != 0);
    assert(cluster_landmarks_clean(
      2, 4, 4, 20, 10, 1, -1, 0, false, 0.05,
      {0, 1, 0, 1, 2, 3, 2, 3},
      {0, 0, 1, 1, 2, 2, 3, 3}, assignment) != 0);
}

}  // namespace

int main() {
  TestLandmarksAreExclusiveAndResidualsAreBalanced();
  TestGroupsLandmarksObservedByTheSameCameras();
  TestHardGroupsLowCameraSignatures();
  TestSplitsOversizedHardGroupsIntoCapacityChunks();
  TestMetricsMatchFullEvaluation();
  TestOptionalMaximumCameraObjective();
  TestCInterfaceReturnsLandmarkAssignment();
  std::cout << "landmark partitioning tests passed\n";
  return 0;
}