#ifndef SCALABLE_LANDMARK_PARTITIONING_H
#define SCALABLE_LANDMARK_PARTITIONING_H

#include "landmark_partitioning.h"

namespace bundle_palm {

class ScalableLandmarkPartitioner {
 public:
  explicit ScalableLandmarkPartitioner(LandmarkPartitioningOptions options);

  LandmarkPartition Partition(const BipartiteCameraPointGraph& graph) const;

 private:
  LandmarkPartitioningOptions options_;
};

}  // namespace bundle_palm

extern "C" int cluster_landmarks_scalable(
    int cluster_count,
    int camera_count,
    int landmark_count,
    int minimum_camera_landmarks,
    int max_refinement_passes,
    double residual_balance_slack,
    const std::vector<int>& camera_indices,
    const std::vector<int>& landmark_indices,
    std::vector<int>& landmark_to_cluster_out);

#endif  // SCALABLE_LANDMARK_PARTITIONING_H