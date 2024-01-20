// returns covered_landmark_indices_c: those in addition! -- which is weird -- to those covered by main residual split
// the latter are point_indices_already_covered and also returned
#include "process_clusters.h"

#include <iostream>
#include <vector>
#include <unordered_set>
#include <map>
#include <queue>
#include <limits>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>

// Speedup, very little worse, else retries rejected merges.
#define inheritInvalidEdgeness_
#define simpleConstraint_  // if undefined works well with max-sum
// if undefined: maxWeight (much faster & better) - sum overestimates overlap.
// #define sumWeight_
#define debug_out_
// Permanent recomputation does not help unless sumWeight / maxWeight are
// replaced by 2nd order (expensive) estimate [maxWeight: 1st order approx].
// #define recomputeVolumeAllTheTime_

std::pair<std::vector<int>*, std::vector<int>*> vec_and_size_from_vex(const std::vector<std::vector<int>>& vec) {
    auto* sizes   = new std::vector<int>(vec.size());
    int total_size = std::accumulate(vec.begin(), vec.end(), 0, [](int sum, const std::vector<int>& v) { return sum + v.size(); });
    auto* content = new std::vector<int>(total_size);
    int offset=0;
    for(const auto &v : vec) {
        std::copy(v.begin(), v.end(), content->begin() + offset);
        offset += v.size();
    }
    return std::make_pair(content, sizes);
}

void fill_vec_and_size(std::vector<int>& content, std::vector<int>& sizes, const std::vector<std::vector<int>>& vec) {
    //std::cout << " sizes resize to " << vec.size() << " sizes size:" << sizes.size() << "\n";
    //sizes.resize(vec.size());
    //std::cout << " sizes resized to " << sizes.size() << "\n";
    sizes.clear();
    int total_size = std::accumulate(vec.begin(), vec.end(), 0, [](int sum, const std::vector<int>& v) { return sum + v.size(); });
    content.resize(total_size);
    int offset = 0;
    for(const auto &v : vec) {
        //std::cout << "v.size() "  << v.size() << "\n";
        std::copy(v.begin(), v.end(), content.begin() + offset);
        offset += v.size();
        sizes.push_back(v.size());
    }
    // for(int sz : sizes) {
    //     std::cout << "filled sizes with shit: " << sz << "\n";
    // }
}

void fill_vec_and_size(std::vector<int>* content, std::vector<int>* sizes, const std::vector<std::vector<int>>& vec) {
    //std::cout << " sizes resize to " << vec.size() << " sizes size:" << sizes->size() << "\n";
    //sizes->resize(vec.size());
    //std::cout << " sizes resized to " << sizes->size() << "\n";
    int total_size = std::accumulate(vec.begin(), vec.end(), 0, [](int sum, const std::vector<int>& v) { return sum + v.size(); });
    content->resize(total_size);
    int offset = 0;
    for(const auto &v : vec) {
        std::copy(v.begin(), v.end(), content->begin() + offset);
        offset += v.size();
        sizes->push_back(v.size());
    }
}

void fill_vec(std::vector<int>& content, const std::vector<int>& vec) {
    content.resize(vec.size());
    std::copy(vec.begin(), vec.end(), content.begin());
}

// nm -D libprocess_clusters.so | grep clusters
extern "C"
void //std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<int>> 
process_clusters(
    int num_lands,
    int num_res,
    int kClusters, 
    const std::vector<int>& point_indices_in_cluster_flat, 
    const std::vector<int>& point_indices_in_cluster_sizes,
    const std::vector<int>& point_indices_, 
    const std::vector<int>& res_indices_in_cluster_flat,
    const std::vector<int>& res_indices_in_cluster_sizes,
    std::vector<int>& res_toadd_out, std::vector<int>& res_toadd_sizes,
    std::vector<int>& point_indices_already_covered_out, std::vector<int>& point_indices_already_covered_sizes,
    std::vector<int>& covered_landmark_indices_c_out, std::vector<int>& covered_landmark_indices_c_sizes, 
    std::vector<int>& num_res_per_c_out)
{
    std::cout << "nl " << num_lands << " nr " << num_res << " kc " << kClusters << std::endl;
    std::cout << point_indices_in_cluster_flat.size() << "  " << point_indices_.size()
              << " " << res_indices_in_cluster_flat.size() << std::endl;
    std::cout << "point_indices_in_cluster_sizes " << point_indices_in_cluster_sizes.size() 
              << " res_indices_in_cluster_sizes " << res_indices_in_cluster_sizes.size() << "\n";

    // Reconstruct the nested vectors
    std::vector<std::vector<int>> point_indices_in_cluster_;
    int start = 0;
    for (int size : point_indices_in_cluster_sizes) {
        std::vector<int> sublist(point_indices_in_cluster_flat.begin() + start, point_indices_in_cluster_flat.begin() + start + size);
        point_indices_in_cluster_.push_back(sublist);
        start += size;
    }
    std::vector<std::vector<int>> res_indices_in_cluster_;
    start = 0;
    for (int size : res_indices_in_cluster_sizes) {
        std::vector<int> sublist(res_indices_in_cluster_flat.begin() + start, res_indices_in_cluster_flat.begin() + start + size);
        res_indices_in_cluster_.push_back(sublist);
        start += size;
    }

    std::vector<int> landmark_occurrences(num_lands, 0);
    for (int ci = 0; ci < kClusters; ci++) {
        std::unordered_set<int> unique_points_in_cluster(point_indices_in_cluster_[ci].begin(), point_indices_in_cluster_[ci].end());
        for (int point : unique_points_in_cluster) {
            landmark_occurrences[point]++;
        }
    }
    std::cout << "uncovered are " << std::count(landmark_occurrences.begin(), landmark_occurrences.end(), 0) << " landmarks, present in single " << std::count(landmark_occurrences.begin(), landmark_occurrences.end(), 1) << ", present in multiple " << std::count_if(landmark_occurrences.begin(), landmark_occurrences.end(), [](int val) { return val > 1; }) << std::endl;

    std::vector<int> point_indices_to_complete;
    std::vector<int> point_indices_completed;

    for (int i = 0; i < num_lands; i++) {
        if (landmark_occurrences[i] > 1) {
            point_indices_to_complete.push_back(i);
        }
        if (landmark_occurrences[i] == 1) {
            point_indices_completed.push_back(i);
        }
    }
    std::cout << "point_indices_to_complete " << point_indices_to_complete.size() << std::endl;

    std::vector<std::vector<int>> point_indices_already_covered(kClusters);
    int sum_points_covered = 0;
    for (int ci = 0; ci < kClusters; ci++) {
        std::set<int> unique_points_in_cluster(point_indices_in_cluster_[ci].begin(), point_indices_in_cluster_[ci].end());
        std::cout << ci << " unique_points_in_cluster " << unique_points_in_cluster.size() << "\n";
        std::vector<int> intersection;
        std::set_intersection(unique_points_in_cluster.begin(), unique_points_in_cluster.end(), point_indices_completed.begin(), point_indices_completed.end(), std::back_inserter(intersection));
        point_indices_already_covered[ci] = intersection;
        // Wrong here
        std::cout << ci << " point_indices_already_covered " << point_indices_already_covered[ci].size() << std::endl;
        sum_points_covered += point_indices_already_covered[ci].size();
    }
    std::cout << "Together covered points " << point_indices_to_complete.size() + sum_points_covered << "  sum_points_covered: " << sum_points_covered << std::endl;
    // looks ok unordered yet?
    std::unordered_map<int, std::unordered_set<int>> point_to_res_id;
    for (int i = 0; i < num_res; i++) {
        if (landmark_occurrences[point_indices_[i]] > 1) {
            point_to_res_id[point_indices_[i]].insert(i);
        }
    }

    // can this be any slower?
    std::vector<int> res_per_lm(num_lands, 0);
    for(int i :point_indices_) {
        res_per_lm[i]++;
    }
    // std::vector<int> res_per_lm(num_res);
    // for (int i = 0; i < num_res; i++) {
    //     res_per_lm[i] = std::count(point_indices_.begin(), point_indices_.end(), point_indices_[i]);
    // }
 
    std::vector<std::vector<int>> missing_res_per_lm_c(kClusters, std::vector<int>(num_res));
    std::vector<int> num_res_per_c(kClusters);

    for (int ci = 0; ci < kClusters; ci++) {
        std::unordered_map<int, int> counts;
        for (int point : point_indices_in_cluster_[ci]) {
            counts[point]++;
        }
        for (int i = 0; i < num_res; i++) {
            const int lm_index =point_indices_[i];
            const auto count_itr = counts.find(point_indices_[i]);
            if (count_itr != counts.end()) {
                missing_res_per_lm_c[ci][lm_index] = res_per_lm[lm_index] - count_itr->second;//counts[point_indices_[i]];
            } else {
                missing_res_per_lm_c[ci][lm_index] = res_per_lm[lm_index];
            }
        }
        num_res_per_c[ci] = point_indices_in_cluster_.size();//std::accumulate(counts.begin(), counts.end(), 0, [](int sum, const std::pair<int, int>& p) { return sum + p.second; });
    }

    // std::unordered_map<int, std::vector<int>> res_of_lm;
    // for (int i = 0; i < num_res; i++) {
    //     res_of_lm[point_indices_[i]].push_back(i);
    // }

    // We want the set of residual indices of lms for esiduals in the 
    // cluster / not in the cluster sum if total res of lm index = cams that see point.
    std::vector<std::unordered_map<int, std::vector<int>>> res_of_lm_notin_c(kClusters);

    for (int ci = 0; ci < kClusters; ci++) {
        std::vector<bool> res_notin_c(num_res, true); // true for res index not in c
        for (int res_index : res_indices_in_cluster_[ci]) {
            res_notin_c[res_index] = false;
        }
        std::unordered_map<int, std::vector<int>>& tmp = res_of_lm_notin_c[ci];
        for (int i = 0; i < num_res; i++) {
            if (res_notin_c[i]) {
                tmp[point_indices_[i]].push_back(i);
            }
        }
    }

    std::vector<std::vector<int>> res_toadd_to_c(kClusters);
    std::vector<std::vector<int>> covered_landmark_indices_c(kClusters);

    for (int i : point_indices_to_complete) {
        std::vector<int> cost(kClusters, 0);
        for (int ci = 0; ci < kClusters; ci++) {
            cost[ci] += res_of_lm_notin_c[ci][i].size() * num_res;
            cost[ci] += num_res_per_c[ci]; // tie breaker.
        }
        int ci = std::min_element(cost.begin(), cost.end()) - cost.begin();
        res_toadd_to_c[ci].insert(res_toadd_to_c[ci].end(), res_of_lm_notin_c[ci][i].begin(), res_of_lm_notin_c[ci][i].end());
        num_res_per_c[ci] += res_of_lm_notin_c[ci][i].size();
        covered_landmark_indices_c[ci].push_back(i);
    }

    fill_vec_and_size(point_indices_already_covered_out, point_indices_already_covered_sizes, point_indices_already_covered);
    fill_vec_and_size(res_toadd_out, res_toadd_sizes, res_toadd_to_c);
    fill_vec(num_res_per_c_out, num_res_per_c);
    fill_vec_and_size(covered_landmark_indices_c_out, covered_landmark_indices_c_sizes, covered_landmark_indices_c);

    //std::pair<std::vector<int> *, std::vector<int> *> res_toadd_to_c_return = vec_and_size(res_toadd_to_c);

    // (res_toadd_to_c_, point_indices_already_covered_, covered_landmark_indices_c_, num_res_per_c)
    // hmm how to return? make new vectors & fill?
    return;
    // all are per cluster even. so vector<vector<int>>
    //return std::make_tuple(res_toadd_to_c, point_indices_already_covered, covered_landmark_indices_c, num_res_per_c);
}

struct VolumePartitionOptions {
  float maxVolumeOfUnion = 1000.;  // Limit of a partition.
  int maxNumKfsInPart = 100;       // Limit of a partition.
};

float GetCostGain(const std::pair<int, int>& edge,
                  float edgeWeight,
                  const std::vector<float>& weightPerVtx,
                  const std::vector<int>& vtxToPart,
                  const std::vector<int>& numVtxsInUnion,
                  const std::vector<float>& volumeOfUnions,
                  const float totalVolume,
                  const float desiredVol) {
  constexpr float kEps = 1e-6;
  const auto [vtxI, vtxJ] = edge;

  const int numVtxs = std::max(1, static_cast<int>(weightPerVtx.size()));
  const int partSizeI = numVtxsInUnion[vtxToPart[vtxI]];
  const int partSizeJ = numVtxsInUnion[vtxToPart[vtxJ]];
  const float fracPij =
      static_cast<float>(partSizeI) / static_cast<float>(partSizeI + partSizeJ);
  const float fracPijTotal =
      static_cast<float>(partSizeI + partSizeJ) / static_cast<float>(numVtxs);
  const float kldivPQ = fracPijTotal * -0.66f * 2.f * fracPij * (1.f - fracPij);

  const float volI = volumeOfUnions[vtxToPart[vtxI]];
  const float volJ = volumeOfUnions[vtxToPart[vtxJ]];
  // TODO(chvogel): high weight on kldivQP -> can remove strict volume check to
  // allow violations if gain in volume compression is strong, or strict if x2
  // the desired size? -- or higher weight iff volI + volJ >= or ~ desiredVol?
  // const float scale =
  //     std::min(0.1f, std::max(0.f, volI + volJ) - desiredVol) / desiredVol );
  const float kldivQP = desiredVol / totalVolume *
                        std::log(std::max(kEps, (volI + volJ) / (volI * volJ)));

  const float prior = 0.0001 * kldivPQ + 0.001 * kldivQP;

  const float wi = weightPerVtx[vtxI];
  const float wj = weightPerVtx[vtxJ];

  const float gain = edgeWeight / std::min(wi, wj) + prior;
  return gain;
}

std::pair<int, float> GetBestCostGainForPart(
    const std::map<int, float>& adjacentPartToWeight,
    const std::set<std::pair<int, int>>& invalidEdges,
    int partId,
    const std::vector<float>& weightPerVtx,   
    const std::vector<int>& vtxsToPart,
    const std::vector<int>& numVtxsInUnion,   // num cameras in part
    const std::vector<float>& volumeOfUnions, // that is sum of weights = # res covered
    const float totalVolume,
    const VolumePartitionOptions& options) {
  int bestPartId = -1;
  float bestCostGain = std::numeric_limits<float>::lowest();
  for (const auto& [adjPartId, weight] : adjacentPartToWeight) {
    const std::pair<int, int> edge(std::min(partId, adjPartId),
                                   std::max(partId, adjPartId));
    const int numVtxs = numVtxsInUnion[partId] + numVtxsInUnion[adjPartId];
    if (numVtxs > options.maxNumKfsInPart || invalidEdges.find(edge) != invalidEdges.end()) {
      continue;
    }

    const float costOfNewEdge = GetCostGain(edge,
                                            weight,
                                            weightPerVtx,
                                            vtxsToPart,
                                            numVtxsInUnion,
                                            volumeOfUnions,
                                            totalVolume,
                                            options.maxVolumeOfUnion);
    if (bestCostGain < costOfNewEdge) {
      bestCostGain = costOfNewEdge;
      bestPartId = adjPartId;
    }
  }
  //std::cout << "bestPartId " << bestPartId << " bestCostGain " << bestCostGain << "\n";
  return {bestPartId, bestCostGain};
}

int FindRootInVtxToPartMap(const std::vector<int>& vtxsToPart, int start) {
  int rootInPart = start;
  while (vtxsToPart[rootInPart] != rootInPart) {
    rootInPart = vtxsToPart[rootInPart];
  }
  return rootInPart;
}

void RemapToRootInVtxToPartMap(std::vector<int>* vtxsToPart,
                               int start,
                               int rootInPart) {
  while ((*vtxsToPart)[start] != rootInPart) {
    int const temp = (*vtxsToPart)[start];
    (*vtxsToPart)[start] = rootInPart;
    start = temp;
  }
}

void RedoAdjacentPartToWeight(
    const std::map<int, float>& oldAdjacentPartToWeight,
    int vtxIdx,
    const std::vector<int>& vtxsToPart,
#ifdef inheritInvalidEdgeness_
    std::set<std::pair<int, int>>* invalidEdges,
#else
    std::set<std::pair<int, int>>* /*invalidEdges*/,
#endif
    std::map<int, float>* adjacentPartToWeight) {
  for (const auto& [adjvtxIdx, weight] : oldAdjacentPartToWeight) {
    // adjacentParts must also have weight, else we must have old edges to
    // get weight and new edges to get new connections. can the PQ then also
    // take unordered edges and use this cost / memory?
    const int adjPartIdx = FindRootInVtxToPartMap(vtxsToPart, adjvtxIdx);
    if (adjPartIdx != vtxIdx) {
      auto itr = adjacentPartToWeight->try_emplace(adjPartIdx, weight);
      if (!itr.second) {
#ifdef sumWeight_
        itr.first->second += weight;  // better compute [maybe just top 3 max?]
#else
        itr.first->second = std::max(itr.first->second, weight);
#endif
      }
    }
#ifdef inheritInvalidEdgeness_
    if (invalidEdges->find({std::min(vtxIdx, adjvtxIdx), std::max(vtxIdx, adjvtxIdx)}) != invalidEdges->end()) {
      invalidEdges->emplace(std::min(vtxIdx, adjPartIdx),
                            std::max(vtxIdx, adjPartIdx));
    }
#endif
  }
};

void PlotStatistics(const std::vector<int>& vtxsToPart,
                    const std::vector<float>& volumeOfUnions,
                    const std::vector<int>& numVtxsInUnion,
                    const std::vector<int>& idToConsecutiveId,
                    int fullComputes,
                    int merges,
                    int totalVolume,
                    int numEdges) {
  const int numVtxs = vtxsToPart.size();
  float mergedVolume = 0;
  float mergedVolume2 = 0;
  int numVolumesOverall = 0;
  int numParts = 0;
#ifdef debug_out_
  std::cout << "Statistics ============================================\n";
#endif
  for (int bbId = 0; bbId < numVtxs; ++bbId) {
    const int partId = vtxsToPart[bbId];
    if (partId == bbId) {  // keep order of bbIds in consecutive partIds.
      numParts++;
      mergedVolume += volumeOfUnions[partId];
    //   numVolumesOverall += (*boundingvolumes)[bbId].NumVolumes();
    //   const float volumeOfUnion =
    //       (*boundingvolumes)[bbId].RecomputeVolume(5 * numSamplesPerBox);
    //   mergedVolume2 += volumeOfUnion;
#ifdef debug_out_
      std::cout << "Part " << idToConsecutiveId[partId]
                << " vol: " << volumeOfUnions[partId] //<< " ~= " << volumeOfUnion
                << ". vtxs: " << numVtxsInUnion[partId] << "\n";
#endif
    }
  }

  std::cout << "Statistics ============================================\n";
  std::cout << "NumEdges tested " << numEdges
            << " fullComputes: " << fullComputes << "\n";
  std::cout << "Num Parts: " << numParts << " = " << numVtxs - merges
            << " mergedVolume: " << mergedVolume << " ~ " << mergedVolume2
            << " of total vol:" << 100 * mergedVolume2 / totalVolume << "\% \n";
  std::cout << " Num volumes " << numVolumesOverall << " in " << numVtxs
            << " vertices\n";
}

// same sa volume:
// cams are cams, # landmakrs seen by cam are volume, union is number of lms seen by boths sets of cams.
extern "C" 
void cluster_covis(
    int num_res,
    int kClusters, 
    int num_lands,
    int num_cams,
    const std::vector<int>& camera_indices_in, 
    const std::vector<int>& landmark_indices_in,
    std::vector<int>& res_to_cluster, std::vector<int>& res_to_cluster_sizes) {

    const bool verbose = false;

    if (verbose) {
    std::cout << "Start #res " << num_res<< " " << kClusters << " #lnds" << num_lands << "  #cams " << num_cams << "\n";
    std::cout << " camera_indices_in " <<"\n";
    std::cout << " camera_indices_in " << camera_indices_in.size() << "\n";
    std::cout << " landmark_indices_in  " << landmark_indices_in.size() << "\n";
    }
    VolumePartitionOptions options = {.maxVolumeOfUnion = (2 * num_lands) / static_cast<float>(kClusters), .maxNumKfsInPart = static_cast<int>((2 * num_cams) / kClusters)};

struct OrderedEdge {
  int srcIdx;
  int tgtIdx;

  OrderedEdge() : srcIdx(-1), tgtIdx(-1) {}
  OrderedEdge(int idx1, int idx2) {srcIdx = std::min(idx1, idx2);tgtIdx = std::max(idx1, idx2);};

  bool operator<(const OrderedEdge& other) const{
    if (srcIdx < other.srcIdx) {
        return true;
    } else if (srcIdx == other.srcIdx) {
        return tgtIdx < other.tgtIdx;
    } else {
        return false;
    }
  }
  bool operator==(const OrderedEdge& other) const {
    return srcIdx == other.srcIdx && tgtIdx == other.tgtIdx;
  };
};

    // 0. check that vertex indices are in range 0-num_cams
    // 1. define graph 
    // cam = vertex. edge-weight = covisible number of landmarks. edge weight = overlap of bounding boxes.
    //  std::map<OrderedEdge, float> edgeWeightMap =
    //      ComputeOverlapGraphEdges(options.spatialHashingOptions, boundingVolumes);
    // OrderedEdge(int idx1, int idx2)
    // define cam to lms and lm to cams. 
    // define edge weights
    std::vector<std::set<int>> cams_from_lm(num_lands);
    std::vector<std::set<int>> lms_from_cam(num_cams);
    std::vector<std::set<int>> lms_in_part(num_cams);
    for (int res_id = 0; res_id < landmark_indices_in.size(); ++res_id) {
        const int lm_id = landmark_indices_in[res_id];
        const int cam_id = camera_indices_in[res_id];
        cams_from_lm[lm_id].insert(cam_id);
        lms_from_cam[cam_id].insert(lm_id);
        lms_in_part[cam_id].insert(lm_id);
    }

    // int cm = 0;
    // for (const std::set<int>& lm_from_cam : lms_from_cam) {
    //     std::cout << "cam "<< cm++ << " sees " << lm_from_cam.size() << " landmarks\n";
    // }
    // int lm = 0;
    // for (const std::set<int>& cam_from_lm : cams_from_lm) {
    //     std::cout << "lm "<< lm++ << " sees " << cam_from_lm.size() << " cams\n";
    // }

    // edge if cam share lm. weight = number of lms shared.
    std::map<OrderedEdge, float> edgeWeightMap;
    for (const std::set<int>& cams_of_lm : cams_from_lm) {
        //std::cout << "cams of lm: " << cams_of_lm.size() << "\n";
        for(const int cam1 :cams_of_lm) {
            for(const int cam2 :cams_of_lm) {
                if (cam1 != cam2){
                    edgeWeightMap[{cam1, cam2}] += 0.5f;
                }
            }
        }
    }

   std::vector<float> weightPerVtx(num_cams, 0);
   std::vector<float> weightPerVtxSelected(num_cams, 0);
   std::vector<std::map<int, float>> adjacentPartAndEdgeWeight(num_cams);

   // The costs used in the heap based greedy strategy.
   for (const auto& [edge, weight] : edgeWeightMap) {
     if (edge.srcIdx != edge.tgtIdx) {
      weightPerVtx[edge.srcIdx] += weight;
      weightPerVtx[edge.tgtIdx] += weight;
      adjacentPartAndEdgeWeight[edge.srcIdx].insert({edge.tgtIdx, weight});
      adjacentPartAndEdgeWeight[edge.tgtIdx].insert({edge.srcIdx, weight});
    }
  }

  std::vector<int> vtxsToPart(num_cams);
  std::iota(vtxsToPart.begin(), vtxsToPart.end(), 0);

  // Helpers to supervise state of clustering.
  std::vector<float> volumeOfUnions(num_cams);  // per part.
  std::vector<int> numVtxsInUnion(num_cams);    // per part.
  for (int bbId = 0; bbId < num_cams; ++bbId) {
    volumeOfUnions[bbId] = lms_from_cam[bbId].size(); // lms in part!
    numVtxsInUnion[bbId] = 1; // cam
  }
  const float totalVolume = num_lands;
    //   std::accumulate(volumeOfUnions.begin(), volumeOfUnions.end(), 0.f);

  std::vector<float> costVector;
  auto cmp = [&costVector](int left, int right) {
    return costVector[left] < costVector[right];
  };
  std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);
  std::vector<std::pair<int, int>> edgeVector;
  std::vector<float> edgeWeightVector;
  std::set<std::pair<int, int>> invalidEdges;

  for (int partId = 0; partId < num_cams; ++partId) {
    const auto [bestAdjPartId, bestCostGain] =
        GetBestCostGainForPart(adjacentPartAndEdgeWeight[partId],
                               invalidEdges, // not used?
                               partId,
                               weightPerVtx,
                               vtxsToPart,
                               numVtxsInUnion,
                               volumeOfUnions,
                               totalVolume,
                               options);
    if (bestAdjPartId >= 0) {
      const int id = edgeVector.size();
      edgeVector.emplace_back(partId, bestAdjPartId);
      edgeWeightVector.push_back(
          adjacentPartAndEdgeWeight[partId][bestAdjPartId]);
      costVector.push_back(bestCostGain);
      pq.push(id);
    }
  }
  //PlotWeightPercentiles(edgeWeightVector); // good debug?

  int merges = 0;
  int fullComputes = 0;

  while (!pq.empty()) {
    const int edgeId = pq.top();
    const auto& [vtxA, vtxB] = edgeVector[edgeId];
    const float intersectionVolumeEstimate = edgeWeightVector[edgeId];
    pq.pop();

    if (pq.size() % 100 == 0) {
      std::cout << "pq.size: " << pq.size() << " fullComputes: " << fullComputes
                << " weight/cost: " << intersectionVolumeEstimate << " / "
                << costVector[edgeId] << " merges: " << merges
                << " edge: " << vtxA << "->" << vtxB << " totalVolume "
                << totalVolume << "\r";
    }

    const int rootVtxInPartA = FindRootInVtxToPartMap(vtxsToPart, vtxA);
    const int rootVtxInPartB = FindRootInVtxToPartMap(vtxsToPart, vtxB);

    if (rootVtxInPartA != vtxA) {
      RemapToRootInVtxToPartMap(&vtxsToPart, vtxA, rootVtxInPartA);
      //std::cout << "RemapToRootInVtxToPartMap " << rootVtxInPartA << " != " <<  vtxA << "\n";
      continue;  // Part does not exist anymore.
    }
    if (vtxA == rootVtxInPartB) {
       //std::cout << "vtxA == rootVtxInPartB " << vtxA << " == " <<  rootVtxInPartB << "\n";
      continue;  // Already merged, we built a new edge for the surving partId.
    }
    const int numVtxsInCover = // num cameras in cover
        numVtxsInUnion[vtxA] + numVtxsInUnion[rootVtxInPartB];
    // Merge would have a part consist of too many vertices.
    if (numVtxsInCover > options.maxNumKfsInPart) {
        //std::cout << "numVtxsInCover > options.maxNumKfsInPart " << numVtxsInCover << " > " << options.maxNumKfsInPart << "\n";
      invalidEdges.emplace(std::min(vtxA, rootVtxInPartB),
                           std::max(vtxA, rootVtxInPartB));
    }

    std::map<int, float> adjacentPartToWeight;
    RedoAdjacentPartToWeight(adjacentPartAndEdgeWeight[vtxA],
                             vtxA,
                             vtxsToPart,
                             &invalidEdges,
                             &adjacentPartToWeight);
    adjacentPartAndEdgeWeight[vtxA] = adjacentPartToWeight;

    // Recompte best cost again. check cost and part to be the same.
    const auto [vtxABestAdjPartId, vtxABestCostGain] =
        GetBestCostGainForPart(adjacentPartAndEdgeWeight[vtxA],
                               invalidEdges,
                               vtxA,
                               weightPerVtx,
                               vtxsToPart,
                               numVtxsInUnion,
                               volumeOfUnions,
                               totalVolume,
                               options);

    if (vtxABestCostGain > costVector[edgeId]) {  // very rare.
      std::cout << "\n Better cost at different vtx" << vtxABestCostGain << ">"
                << costVector[edgeId] << " " << vtxABestAdjPartId
                << "!=" << rootVtxInPartB << "\n";
    }

    if (vtxABestAdjPartId < 0 || vtxABestAdjPartId == vtxA) {
      //std::cout << "vtxABestAdjPartId < 0  || vtxABestAdjPartId == vtxA, vtxABestAdjPartId:" << vtxABestAdjPartId << " vtxA " << vtxA << "\n";
      continue;  // latter exists since both a->b and b->a can exist.
    }

    if (vtxABestAdjPartId >= 0 && (vtxABestCostGain != costVector[edgeId] ||
                                   vtxABestAdjPartId != rootVtxInPartB)) {
      edgeVector[edgeId] = {vtxA, vtxABestAdjPartId};
      edgeWeightVector[edgeId] = adjacentPartToWeight[vtxABestAdjPartId];
      costVector[edgeId] = vtxABestCostGain;
      pq.push(edgeId);
      continue;
    }

    const int rootVtxInPart = std::min(vtxA, rootVtxInPartB);
    const int largestVtxInPart = std::max(vtxA, rootVtxInPartB);

    const float intersectionVolumeLowerBound = 0;
    fullComputes++;
    // Todo: simple in this case: size of set of landmarks in intersection.
    // float volumeOfUnion = (*boundingVolumes)[vtxA].ComputeVolumeOfUnion(
    //       (*boundingVolumes)[rootVtxInPartB], numSamplesPerBox);
    std::set<int> intersection;
    std::set_union(lms_in_part[vtxA].begin(), lms_in_part[vtxA].end(), 
        lms_in_part[rootVtxInPartB].begin(), lms_in_part[rootVtxInPartB].end(),
        std::inserter(intersection, intersection.begin()));
    const int volumeOfUnion = intersection.size();

    // std::cout << "vtxA,b lms in part, union " << vtxA << "-" << rootVtxInPartB << " : " 
    //     << lms_in_part[vtxA].size() << " " << lms_in_part[rootVtxInPartB].size() << " " << volumeOfUnion << "\n";

#ifndef simpleConstraint_
    // const float maxCandidateVolume = std::max(
    //     volumeOfUnions[vtxA], volumeOfUnions[rootVtxInPartB]);
    // const float relativeVolumeGain =
    //     (volumeOfUnion - maxCandidateVolume) / maxCandidateVolume;
    const float minCandidateVolume =
        std::min(volumeOfUnions[vtxA], volumeOfUnions[rootVtxInPartB]);
    const float sumCandidateVolume =
        volumeOfUnions[vtxA] + volumeOfUnions[rootVtxInPartB];
    const float relativeVolumeGain =
        (sumCandidateVolume - volumeOfUnion) / minCandidateVolume;  // Standard
    // intersection/union or inter/min variants:
    // 1.f - (sumCandidateVolume - volumeOfUnion) / volumeOfUnion;
    // 1.f - (volumeOfUnion - maxCandidateVolume) / maxCandidateVolume;

    const float relativeVolumeViolation =
        (volumeOfUnion - options.maxVolumeOfUnion) / options.maxVolumeOfUnion;
    // Below is relativeVolumeGain wrt to max.
    // const float relativeVolumeViolation = // variant too soft / grows away.
    //     (volumeOfUnion -
    //      std::max(maxCandidateVolume, options.maxVolumeOfUnion)) /
    //     options.maxVolumeOfUnion;
    const float volumeConstraintMultiplicator = 7.f;

#ifdef __rejDebugInfo__
    if (volumeOfUnion > options.maxVolumeOfUnion &&
        relativeVolumeViolation * volumeConstraintMultiplicator <
            relativeVolumeGain) {
      std::cout << "Vol vio/inc in \%: " << 100 * relativeVolumeViolation << " "
                << 100 * relativeVolumeGain << "\n";
      std::cout << "Acc (vol) " << volumeOfUnion << " > "
                << options.maxVolumeOfUnion
                << " vol: " << (*boundingVolumes)[vtxA].Volume() << " V "
                << (*boundingVolumes)[rootVtxInPartB].Volume() << "\n";
    }
#endif
    if (volumeOfUnion > options.maxVolumeOfUnion &&
        relativeVolumeViolation * volumeConstraintMultiplicator >=
            1. - relativeVolumeGain &&
        relativeVolumeGain > 0) {
#else
    // Constraints fail, abandon merge !?
    if (volumeOfUnion > options.maxVolumeOfUnion) {
#endif
    if (verbose) {
      std::cout << " volumeOfUnion > options.maxVolumeOfUnion " << volumeOfUnion << " " <<  options.maxVolumeOfUnion<< "\n";
    }
      // Mark edge as invalid.
      invalidEdges.emplace(rootVtxInPart, largestVtxInPart);
      // Find new best option and reinsert.
      // TODO(chvogel): Could avoid recomputation by storing 2nd best cost above
      const auto [bestAdjPartId, bestCostGain] =
          GetBestCostGainForPart(adjacentPartAndEdgeWeight[vtxA],
                                 invalidEdges,
                                 vtxA,
                                 weightPerVtx,
                                 vtxsToPart,
                                 numVtxsInUnion,
                                 volumeOfUnions,
                                 totalVolume,
                                 options);
      if (bestAdjPartId >= 0) {
        edgeVector[edgeId] = {vtxA, bestAdjPartId};
        edgeWeightVector[edgeId] =
            adjacentPartAndEdgeWeight[vtxA][bestAdjPartId];
        costVector[edgeId] = bestCostGain;
        pq.push(edgeId);
      }
      continue;
    }
    //////////////////////////////// Merge parts:
    ++merges;
    if (verbose) {
    std::cout << "Merge parts: " << vtxA << " " << rootVtxInPartB << "\n";
    }
    volumeOfUnions[rootVtxInPart] = volumeOfUnion;
    // merge sets of landmarks
    // (*boundingVolumes)[rootVtxInPart].Merge(
    //     (*boundingVolumes)[largestVtxInPart], volumeOfUnion);
    lms_in_part[rootVtxInPart]= intersection;
    numVtxsInUnion[rootVtxInPart] = numVtxsInCover;

#ifdef recomputeVolumeAllTheTime_
    (*boundingVolumes)[rootVtxInPart].Merge(
        (*boundingVolumes)[largestVtxInPart]);
    volumeOfUnions[rootVtxInPart] = (*boundingVolumes)[rootVtxInPart].Volume();
#endif
    RemapToRootInVtxToPartMap(&vtxsToPart, vtxA, rootVtxInPart);
    RemapToRootInVtxToPartMap(&vtxsToPart, vtxB, rootVtxInPart);

    // Merge edges, compute new ones from the new part, push into queue.
    adjacentPartToWeight.clear();
    for (const int oldPartId : {vtxA, rootVtxInPartB}) {
      RedoAdjacentPartToWeight(adjacentPartAndEdgeWeight[oldPartId],
                               oldPartId,
                               vtxsToPart,
                               &invalidEdges,
                               &adjacentPartToWeight);
    }

    weightPerVtxSelected[rootVtxInPart] +=
        weightPerVtxSelected[largestVtxInPart] + intersectionVolumeEstimate;
    weightPerVtx[rootVtxInPart] +=
        weightPerVtx[largestVtxInPart] - intersectionVolumeEstimate;

    const auto [bestAdjPartId, bestCostGain] =
        GetBestCostGainForPart(adjacentPartToWeight,
                               invalidEdges,
                               rootVtxInPart,
                               weightPerVtx,
                               vtxsToPart,
                               numVtxsInUnion,
                               volumeOfUnions,
                               totalVolume,
                               options);

    if (bestCostGain == costVector[edgeId]) {
      std::cout << "Same cost after merge, bef/after: " << costVector[edgeId]
                << " / " << bestCostGain << "\n";
    }

    if (bestAdjPartId >= 0) {
      edgeVector[edgeId] = {rootVtxInPart, bestAdjPartId};
      edgeWeightVector[edgeId] = adjacentPartToWeight[bestAdjPartId];
      costVector[edgeId] = bestCostGain;
      pq.push(edgeId);
    }

    adjacentPartAndEdgeWeight[rootVtxInPart] = adjacentPartToWeight;
    //////////////////////////////

  }  // end pq

  //std::cout << "End pq " << "pq.size: " << pq.size() << "\n";

  for (auto& vtxToPart : vtxsToPart) {
    vtxToPart = vtxsToPart[vtxToPart];
  }
  int numParts = 0;
  std::vector<int> idToConsecutiveId(num_cams, -1);
  for (int bbId = 0; bbId < num_cams; ++bbId) {
    const int partId = vtxsToPart[bbId];
    if (partId == bbId) {  // Keep order of bbIds in consecutive partIds.
      idToConsecutiveId[partId] = numParts++;
    }
  }

//   for (int bbId = 0; bbId < num_cams; ++bbId) {
//     std::cout << "vtx " << bbId << " in part " << vtxsToPart[bbId] << " land in part " << " volumeOfUnions " << volumeOfUnions[vtxsToPart[bbId]] <<"\n";
//   }

  PlotStatistics(vtxsToPart,
                 volumeOfUnions,
                 numVtxsInUnion,
                 idToConsecutiveId,
                 fullComputes,
                 merges,
                 totalVolume,
                 edgeWeightMap.size());
                //  numSamplesPerBox,
                //  boundingVolumes);

  std::vector<std::set<int>> lms_in_part_(kClusters);
  for (int bbId = 0; bbId < num_cams; ++bbId) {
    vtxsToPart[bbId] = idToConsecutiveId[vtxsToPart[bbId]];
    lms_in_part_[vtxsToPart[bbId]].insert(lms_from_cam[bbId].begin(), lms_from_cam[bbId].end());
  }

//   for (int bbId = 0; bbId < num_cams; ++bbId) {
//     std::cout << "vtx " << bbId << " in part " << vtxsToPart[bbId] << " land in part " << " volumeOfUnions " << volumeOfUnions[vtxsToPart[bbId]] << " " << lms_in_part_[vtxsToPart[bbId]].size() <<"\n";
//   }

    //return vtxsToPart;

}