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
#include <random> // for std::mt19937
#include <chrono> // for std::chrono

// Speedup, very little worse, else retries rejected merges.
#define inheritInvalidEdgeness_
// if undefined: maxWeight (much faster & better than sumWeight)? - sum overestimates overlap.
//#define explicitWeight_ // exact overlap
// if defined -> explicit weight -> sum weight
//#define sumWeight_
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
    std::vector<int>& res_to_cluster_by_landmark_out)
{
    if (false) {
        std::cout << "nl " << num_lands << " nr " << num_res << " kc " << kClusters << std::endl;
        std::cout << point_indices_in_cluster_flat.size() << "  " << point_indices_.size()
                << " " << res_indices_in_cluster_flat.size() << std::endl;
        std::cout << "point_indices_in_cluster_sizes " << point_indices_in_cluster_sizes.size() 
                << " res_indices_in_cluster_sizes " << res_indices_in_cluster_sizes.size() << "\n";
    }
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
    std::cout << "uncovered are " << std::count(landmark_occurrences.begin(), landmark_occurrences.end(), 0) 
              << " landmarks, present in single " << std::count(landmark_occurrences.begin(), landmark_occurrences.end(), 1) 
              << ", present in multiple " << std::count_if(landmark_occurrences.begin(), landmark_occurrences.end(), [](int val) { return val > 1; })
              << "point_indices_to_complete " << point_indices_to_complete.size() << std::endl;

    std::vector<std::vector<int>> point_indices_already_covered(kClusters);
    int sum_points_covered = 0;
    for (int ci = 0; ci < kClusters; ci++) {
        std::set<int> unique_points_in_cluster(point_indices_in_cluster_[ci].begin(), point_indices_in_cluster_[ci].end());
        std::vector<int> intersection;
        std::set_intersection(unique_points_in_cluster.begin(), unique_points_in_cluster.end(), point_indices_completed.begin(), point_indices_completed.end(), std::back_inserter(intersection));
        point_indices_already_covered[ci] = intersection;
        // Wrong here
        std::cout << ci << " unique_points_in_cluster " << unique_points_in_cluster.size()
                  << " point_indices_already_covered " << point_indices_already_covered[ci].size() << std::endl;
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

    std::vector<int> res_per_lm(num_lands, 0);
    for(int i :point_indices_) {
        res_per_lm[i]++;
    }
 
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
                missing_res_per_lm_c[ci][lm_index] = res_per_lm[lm_index] - count_itr->second;
            } else {
                missing_res_per_lm_c[ci][lm_index] = res_per_lm[lm_index];
            }
        }
        num_res_per_c[ci] = point_indices_in_cluster_.size();
    }

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
    fill_vec_and_size(covered_landmark_indices_c_out, covered_landmark_indices_c_sizes, covered_landmark_indices_c);

    // so all point indices in cluster are covered_landmark_indices_c AND point_indices_already_covered_out
    // must union on these.
    // res_of_all_covered_landmarks shoudl be returned,
    // per cluster go over points set int to part : landmark to cluster
    // go over res add res following landmark to cluster.
    // landmark_res_in_cluster can be used in python: res -> cluster id
    std::vector<int> res_to_cluster_by_landmark(num_res, -1);
    std::vector<int> landmark_to_cluster(num_lands, -1);
    for (int ci = 0; ci < kClusters; ci++) {
        for (int i : covered_landmark_indices_c[ci]) {
          landmark_to_cluster[i] = ci;
        }
        for (int i : point_indices_already_covered[ci]) {
          landmark_to_cluster[i] = ci;
        }
    }
    // lms are disjointly distributed to cluster. residuals are assigned by lm present in residual.
    for (int res_id=0; res_id < point_indices_.size(); res_id++) {
        const int lm_id = point_indices_[res_id];
        res_to_cluster_by_landmark[res_id] = landmark_to_cluster[lm_id];
    }

  fill_vec(res_to_cluster_by_landmark_out, res_to_cluster_by_landmark);
  return;
}

struct VolumePartitionOptions {
  float targetVolumeOfUnion = 1000.;  // Limit of a partition.
  float maxVolumeOfUnion = 1000.;  // Limit of a partition.
  int maxNumKfsInPart = 100;       // Limit of a partition.
  int targetKfsInPart = 100;
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

    // Original:
  // const float prior = 0.0001 * kldivPQ + 0.001 * kldivQP;
  // kldivPQ -> equal cams , kldivQP -> equal volume/landmarks/edges.
  const float prior = 0.5 * kldivPQ + 0.1 * kldivQP; // ToDo play around here.

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
                                            options.targetVolumeOfUnion);
    if (bestCostGain < costOfNewEdge) { // bias to small indxs?
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
    const std::vector<std::set<int>>& lms_in_part, // alternative way to compute edge weights.
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
    float newWeight = weight;
    const int adjPartIdx = FindRootInVtxToPartMap(vtxsToPart, adjvtxIdx);
    if (adjPartIdx == vtxIdx) {
        continue;
    }
#ifdef inheritInvalidEdgeness_
    if (invalidEdges->find({std::min(vtxIdx, adjvtxIdx), std::max(vtxIdx, adjvtxIdx)}) != invalidEdges->end()) {
      invalidEdges->emplace(std::min(vtxIdx, adjPartIdx), std::max(vtxIdx, adjPartIdx));
      newWeight = std::numeric_limits<float>::max();
      // overwrite and continue. faster?
      adjacentPartToWeight->insert_or_assign(adjPartIdx, newWeight);
      continue;
    }
#endif
      auto itr = adjacentPartToWeight->try_emplace(adjPartIdx, newWeight);
      if (!itr.second) {
#ifdef explicitWeight_
#ifdef sumWeight_
        itr.first->second += newWeight;  // better compute [maybe just top 3 max?]
#else // slow
        std::set<int> union_new_part;
        std::set_union(lms_in_part[vtxIdx].begin(), lms_in_part[vtxIdx].end(), 
            lms_in_part[adjPartIdx].begin(), lms_in_part[adjPartIdx].end(),
            std::inserter(union_new_part, union_new_part.begin()));
        const int volumeOfUnion = union_new_part.size();
        itr.first->second = volumeOfUnion;
#endif
#else
        itr.first->second = std::max(itr.first->second, newWeight);
#endif
      }
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

// idea, draw edge, maybe based on weight (more likely to draw), maybe just pairs? not already merged ones?
int RandomMerge(int number_merges,
     std::vector<float>& volumeOfUnions,
     std::vector<int>& numVtxsInUnion,
     std::vector<int>& vtxsToPart,
     std::vector<std::set<int>>& lms_in_part,
     std::set<std::pair<int, int>>& invalidEdges,
     std::vector<float>& weightPerVtx,
     //std::vector<float>& weightPerVtxSelected,
     const std::vector<int>& old_vtxsToPart,
     const std::map<std::pair<int,int>, float>& edgeWeightMap,
     const VolumePartitionOptions& options) {
    int num_merges = 0;
    int num_vtx = vtxsToPart.size();
    int num_edges = edgeWeightMap.size();

    std::mt19937 mt{ static_cast<std::mt19937::result_type>(
		std::chrono::steady_clock::now().time_since_epoch().count() ) };
    std::uniform_int_distribution dist{ 0, num_edges-1 };
    // build prefered set, permute, pick from until merges reached or empty
    std::vector<int> preferedEdgeIds;
    //std::cout << "old_vtxsToPart " << old_vtxsToPart.size() << "\n";
    preferedEdgeIds.reserve(old_vtxsToPart.size() * 10);
    if(!old_vtxsToPart.empty()) {
        int id = 0;
        for (const auto& ordered_edge : edgeWeightMap) {
            // better go over edges -> go over prefered set. merge until numVtxs too large, permute order       
            const int old_part_src = old_vtxsToPart[ordered_edge.first.first];
            const int old_part_tgt = old_vtxsToPart[ordered_edge.first.second];
            if(old_part_src != old_part_tgt) {
                preferedEdgeIds.push_back(id);
            }
            ++id;
        }
        std::cout << "preferedEdgeIds " << preferedEdgeIds.size() << " \n";
    }
    std::shuffle(preferedEdgeIds.begin(), preferedEdgeIds.end(), mt);
    int prefered_edge_id = 0;
    preferedEdgeIds.resize(std::min(preferedEdgeIds.size(), size_t(150)));
    while (prefered_edge_id < preferedEdgeIds.size() || num_merges < std::min(number_merges, static_cast<int>(0.5 * (num_vtx-1)))) {

        // draw edge seed from time.
        auto elementId = edgeWeightMap.begin();

        if(prefered_edge_id < preferedEdgeIds.size()) {
            std::advance(elementId, preferedEdgeIds[prefered_edge_id++]);
        }
        else {
            std::advance(elementId, dist(mt));
        }

        const auto& [ordered_edge, weight] = *elementId; //edgeWeightMap[id];

        const int old_part_src = old_vtxsToPart[ordered_edge.first];
        const int old_part_tgt = old_vtxsToPart[ordered_edge.second];

        const int rootVtxInPartA = FindRootInVtxToPartMap(vtxsToPart, ordered_edge.first);
        const int rootVtxInPartB = FindRootInVtxToPartMap(vtxsToPart, ordered_edge.second);

        if (rootVtxInPartA == rootVtxInPartB) {continue;}
        const int rootVtxInPart = std::min(rootVtxInPartA, rootVtxInPartB);
        const int largestVtxInPart = std::max(rootVtxInPartA, rootVtxInPartB);

        std::pair<int,int> edge (std::min(rootVtxInPartA, rootVtxInPartB), std::max(rootVtxInPartA, rootVtxInPartB));
        RemapToRootInVtxToPartMap(&vtxsToPart, edge.first, rootVtxInPart);
        RemapToRootInVtxToPartMap(&vtxsToPart, edge.second, rootVtxInPart);
        const int numVtxs = numVtxsInUnion[rootVtxInPartA] + numVtxsInUnion[rootVtxInPartB];
        if (numVtxs > options.targetKfsInPart || invalidEdges.find(edge) != invalidEdges.end()) {
        //if (numVtxs > options.maxNumKfsInPart || invalidEdges.find(edge) != invalidEdges.end()) {
            continue;
        }
        // std::cout << "edge " << ordered_edge.srcIdx << "/" << ordered_edge.tgtIdx
        //           << " Old parts " << old_part_src << " " << old_part_tgt << " num_merges " << num_merges << "\n";

        num_merges++;
        std::set<int> union_new_part;
        std::set_union(lms_in_part[rootVtxInPartA].begin(), lms_in_part[rootVtxInPartA].end(), 
            lms_in_part[rootVtxInPartB].begin(), lms_in_part[rootVtxInPartB].end(),
            std::inserter(union_new_part, union_new_part.begin()));
        const int volumeOfUnion = union_new_part.size();
        const int numVtxsInCover = numVtxsInUnion[rootVtxInPart] + numVtxsInUnion[largestVtxInPart];
        int intersectionVolume = lms_in_part[rootVtxInPartA].size() - volumeOfUnion;

        volumeOfUnions[rootVtxInPartA] = volumeOfUnion;
        lms_in_part[rootVtxInPart]= union_new_part;
        numVtxsInUnion[rootVtxInPart] = numVtxsInCover;

        // TODO: intersectionVolumeEstimate or better intersectionVolume ?
        const float intersectionVolumeEstimate = intersectionVolume;//edgeWeightVector[edgeId];
        weightPerVtx[rootVtxInPart] +=
            weightPerVtx[largestVtxInPart] - intersectionVolumeEstimate;
        // weightPerVtxSelected[rootVtxInPart] +=
        //     weightPerVtxSelected[largestVtxInPart] + intersectionVolumeEstimate; // WHY '+'. not used for anything
        invalidEdges.emplace(rootVtxInPart, largestVtxInPart);
      }
      return num_merges;
  }

// same sa volume:
// cams are cams, # landmakrs seen by cam are volume, union is number of lms seen by boths sets of cams.
// TODO: input landmark ids that must be in same cluster.
// go over cam ids per lm and connect them in advance.
// Then the lm cannot occur in any other cluster. and both move together.
// Else one observing cam is fixed when optimizing the lm -- maybe a blocker.
extern "C" 
void cluster_covis(
    int kClusters,
    int random_pre_number_merges,
    int maxVolPerPart, // = kClusters if landmarks are clustered.
    const std::vector<int>& camera_indices_in,
    const std::vector<int>& landmark_indices_in,
    std::vector<int>& res_to_cluster, std::vector<int>& res_to_cluster_sizes,
    std::vector<int>& old_vtxsToPart) {

    const bool skip_vol_constraint = true; // might be slow, not 'worth' it
    const bool verbose = false;
    const int num_res = landmark_indices_in.size();
    const int num_cams = std::set<double>( camera_indices_in.begin(), camera_indices_in.end() ).size();
    const int num_lands = std::set<double>( landmark_indices_in.begin(), landmark_indices_in.end() ).size();

    std::cout << "Starting cluster_covis " << kClusters << " clusters, pre merges: " << random_pre_number_merges << "\n";

    if (camera_indices_in.size() != num_res || verbose) {
    std::cout << "Start #res " << num_res<< " " << kClusters << " #lnds" << num_lands << "  #cams " << num_cams << "\n";
    std::cout << " camera_indices_in " <<"\n";
    std::cout << " camera_indices_in " << camera_indices_in.size() << "\n";
    std::cout << " landmark_indices_in  " << landmark_indices_in.size() << "\n";
    }
    VolumePartitionOptions options = {.targetVolumeOfUnion = 1.5f * static_cast<float>(num_lands) / static_cast<float>(kClusters), 
        .maxVolumeOfUnion = (std::min(maxVolPerPart, kClusters) * num_lands) / static_cast<float>(kClusters), // hard constraint can lead to invalid # clusters
        .maxNumKfsInPart = static_cast<int>((2 * num_cams) / kClusters),
        .targetKfsInPart = static_cast<int>(num_cams / kClusters)};

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

    // edge if cam share lm. weight = number of lms shared.
    std::map<std::pair<int,int>, float> edgeWeightMap; // slow |landmarks| * |cams| ^2 or worse |cams| * |landmarks| ^2.
    for (const std::set<int>& cams_of_lm : cams_from_lm) {
        for (auto c1_itr = cams_of_lm.cbegin(); c1_itr != cams_of_lm.cend(); c1_itr++) {
            for (auto c2_itr = std::next(c1_itr); c2_itr != cams_of_lm.end(); c2_itr++) {
                    edgeWeightMap[{*c1_itr, *c2_itr}] += 1.f;
            }
        }
    }

   std::vector<float> weightPerVtx(num_cams, 0);
   //std::vector<float> weightPerVtxSelected(num_cams, 0);
   std::vector<std::map<int, float>> adjacentPartAndEdgeWeight(num_cams); // This is edgeweight used in pq. keep up to date.

   // The costs used in the heap based greedy strategy.
   for (const auto& [edge, weight] : edgeWeightMap) {
     if (edge.first != edge.second) {
      weightPerVtx[edge.first] += weight;
      weightPerVtx[edge.second] += weight;
      adjacentPartAndEdgeWeight[edge.first].insert({edge.second, weight});
      adjacentPartAndEdgeWeight[edge.second].insert({edge.first, weight});
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
  std::set<std::pair<int, int>> invalidEdges;
  const float totalVolume = num_lands;
    //   std::accumulate(volumeOfUnions.begin(), volumeOfUnions.end(), 0.f);

 int num_random_merged = RandomMerge(random_pre_number_merges,
     volumeOfUnions,
     numVtxsInUnion,
     vtxsToPart,
     lms_in_part,
     invalidEdges, // neded to set or will automatically?
     weightPerVtx,
     //weightPerVtxSelected,
     old_vtxsToPart,
     edgeWeightMap,
     options);

  std::vector<float> costVector;
  auto cmp = [&costVector](int left, int right) {
    return costVector[left] < costVector[right];
  };
  std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);
  std::vector<std::pair<int, int>> edgeVector;
  std::vector<float> edgeWeightVector;

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

  // stochastic: perform random merges at start, 10% / 20% of cameras ?
  // always group

  // random method: sample k existing part ids.
  // compute score. Some can be invalid update invalid, meybe set of active parts, remove deleted one sample iterator.
  // keep track of best (5) candidates?
  // merge
  // reuse best 5 again. Faster. cluster cams to get COVERDED landmarks.
  // those define the cut.
  // for sake of simplicity dist ALL cams to all clusters at first.
  // differs from palm only in cams can move aspect and are averaged later.

  std::vector<int> partReEvaluatedAtMerge(num_cams, -1);

  int merges = num_random_merged;
  int fullComputes = 0;
  while (!pq.empty() && (num_cams - merges > kClusters)) {
    const int edgeId = pq.top();
    const auto& [vtxA, vtxB] = edgeVector[edgeId];
    float intersectionVolumeEstimate = edgeWeightVector[edgeId];
    pq.pop();

    if (pq.size() % 20 == 0) {
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
      continue;  // Already merged, we built a new edge for the surviving partId.
    }

    const int numVtxsInCover = // num cameras in cover
        numVtxsInUnion[vtxA] + numVtxsInUnion[rootVtxInPartB];
    // Merge would have a part consist of too many vertices.
    bool edgeInvalid = false;
    if (numVtxsInCover > options.maxNumKfsInPart) {
        //std::cout << "numVtxsInCover > options.maxNumKfsInPart " << numVtxsInCover << " > " << options.maxNumKfsInPart << "\n";
      invalidEdges.emplace(std::min(vtxA, rootVtxInPartB),
                           std::max(vtxA, rootVtxInPartB));
      edgeInvalid = true;
    }
       
    // slow: this is done too often and pushed back. must go over all adjacent parts.
    std::map<int, float> adjacentPartToWeight;
    bool consideredAtCurrentState = false;
    if(partReEvaluatedAtMerge[rootVtxInPartA] != merges) { // else reevaluated recently, best cost.
        partReEvaluatedAtMerge[rootVtxInPartA] = merges;

        RedoAdjacentPartToWeight(adjacentPartAndEdgeWeight[vtxA],
                                vtxA,
                                vtxsToPart,
                                lms_in_part,
                                &invalidEdges,
                                &adjacentPartToWeight);
        adjacentPartAndEdgeWeight[vtxA] = adjacentPartToWeight;
    } else {
        adjacentPartToWeight = adjacentPartAndEdgeWeight[vtxA];
        consideredAtCurrentState = true;
    }

    // Recompute best cost again. check cost and part to be the same.
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
                    << "!=" << rootVtxInPartB << "\r";
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
        // Do i need to push if part was looked at at merge?
        // unclear why cost would change if updated wo merge in between. if not we push same edge multiple times?
        // it can change. 
        // if (!consideredAtCurrentState || edgeInvalid || adjacentPartToWeight[vtxABestAdjPartId] != vtxABestCostGain)
        pq.push(edgeId);
        continue;
    }

    const int rootVtxInPart = std::min(vtxA, rootVtxInPartB);
    const int largestVtxInPart = std::max(vtxA, rootVtxInPartB);

    fullComputes++;
    // Todo: simple in this case: size of set of landmarks in intersection.
    // float volumeOfUnion = (*boundingVolumes)[vtxA].ComputeVolumeOfUnion(
    //       (*boundingVolumes)[rootVtxInPartB], numSamplesPerBox);
    std::set<int> union_new_part;
    std::set_union(lms_in_part[vtxA].begin(), lms_in_part[vtxA].end(), 
        lms_in_part[rootVtxInPartB].begin(), lms_in_part[rootVtxInPartB].end(),
        std::inserter(union_new_part, union_new_part.begin()));
    const int volumeOfUnion = union_new_part.size();
    // TODO:? better?
    intersectionVolumeEstimate = volumeOfUnion - lms_in_part[vtxA].size();

    // std::cout << "vtxA,b lms in part, union " << vtxA << "-" << rootVtxInPartB << " : " 
    //     << lms_in_part[vtxA].size() << " " << lms_in_part[rootVtxInPartB].size() << " " << volumeOfUnion << "\n";

    // Constraints fail, abandon merge !?
    if (volumeOfUnion > options.maxVolumeOfUnion) {
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
    lms_in_part[rootVtxInPart]= union_new_part;
    numVtxsInUnion[rootVtxInPart] = numVtxsInCover;

    RemapToRootInVtxToPartMap(&vtxsToPart, vtxA, rootVtxInPart);
    RemapToRootInVtxToPartMap(&vtxsToPart, vtxB, rootVtxInPart);

    // Merge edges, compute new ones from the new part, push into queue.
    adjacentPartToWeight.clear();
    for (const int oldPartId : {vtxA, rootVtxInPartB}) {
      RedoAdjacentPartToWeight(adjacentPartAndEdgeWeight[oldPartId],
                               oldPartId,
                               vtxsToPart,
                               lms_in_part,
                               &invalidEdges,
                               &adjacentPartToWeight);
    }

    // weightPerVtxSelected[rootVtxInPart] +=
    //     weightPerVtxSelected[largestVtxInPart] + intersectionVolumeEstimate;
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

////////// out put should be.
// const std::vector<int>& res_indices_in_cluster_flat,
// const std::vector<int>& res_indices_in_cluster_sizes,
// I got vtxsToPart[bbId] : camid to part.
// this is sufficient to map res to part (cam id contained: camera_indices_in)
//std::vector<int> res_in_cluster(num_res);

//std::cout << " res size " << camera_indices_in.size() << " num_res " << num_res << "\n";
std::vector<std::vector<int>> res_indices_in_cluster(numParts);
//res_to_cluster_sizes.clear();
//res_to_cluster_sizes.resize(kClusters, 0);
old_vtxsToPart.resize(vtxsToPart.size());
for(int res_id = 0; res_id < num_res; ++res_id) {
    const int cam_id = camera_indices_in[res_id];
    const int part_id = vtxsToPart[cam_id];
    old_vtxsToPart[cam_id] = part_id;
    //std::cout << res_id << " residual " << cam_id << " to " << part_id << "\n";  
    //res_to_cluster_sizes[part_id]++;
    //res_in_cluster[res_id] = vtxsToPart[cam_id]; // could also be output, simpler
    res_indices_in_cluster[part_id].push_back(res_id);
}

fill_vec_and_size(res_to_cluster, res_to_cluster_sizes, res_indices_in_cluster);

// from here use
// res_to_cluster but need res -> landmark / camera
}

///////////////////////

// relevantCameras hold part and camId
void FillRelevantCameras(const std::vector<std::map<int, std::set<int>>> &landmarkFromCameraPerPart,
                         const std::vector<std::set<int>>& lms_from_cam,
                         int maxLmPerCam,
                         std::vector<std::vector<std::pair<int, int>>> &relevantCameras) {
  relevantCameras.clear();
  relevantCameras.resize(maxLmPerCam);
  for (int partId = 0; partId < landmarkFromCameraPerPart.size(); partId++) {
    std::map<int, std::set<int>> camToLmsInPart = landmarkFromCameraPerPart[partId];
    for (const auto &[camId, lms] : camToLmsInPart) {
      if (lms.size() < std::min(static_cast<int>(lms_from_cam[camId].size()), maxLmPerCam) && lms.size() > 0 )  {
        relevantCameras[lms.size()].push_back({partId, camId});
        // std::cout << "Cam " << camId << " in part " << partId << " with " 
        //           << lms.size() << "/" << lms_from_cam[camId].size() << " observations\n";
      }
    }
  } // try to move all lms of above.
}

// for(int  relevantCameras

double CostGain(int numLmsBefore, int numLmsAfter, 
                int maxLmPerCam, double temperature) {
  if (std::min(numLmsBefore, numLmsAfter) > maxLmPerCam ||
      numLmsBefore == 0 && numLmsAfter <= 0) {
    return 0.0;
  }
  // 0/1: 1 - exp(-t/5), should be bad. 1/0 should be good.
  // 1/2 exp(-t/5) - exp(-2t/5)
  return std::exp(-numLmsAfter / static_cast<double>(maxLmPerCam) * temperature) - 
         std::exp(-numLmsBefore / static_cast<double>(maxLmPerCam) * temperature);
}

// cost with known lmid. try to move ANY of the lmids. move the best.
std::pair<double, int> GetMoveCost(int lmId, int partFrom, int kClusters,
                                   const std::vector<std::map<int, std::set<int>>> &landmarkFromCameraPerPart,
                                   const std::vector<std::set<int>>& cams_from_lm,
                                   const std::vector<int>& res_per_cluster,
                                   int maxLmPerCam, double temperature) {
  // per part getCostGain: here just the new min number of landmarks in the part.
  std::vector<double> moveCost(kClusters); // cost per part to move to?
  double gainFrom = 0;
  constexpr double eps = 1e-6;// prefer cluster with fewer res, tiny bias.
  int min_res_per_cluster = res_per_cluster[partFrom];
  for (int partId = 0; partId < kClusters; ++partId) {
    min_res_per_cluster = std::min(min_res_per_cluster, res_per_cluster[partId]);
  }
  for (int partId = 0; partId < kClusters; ++partId) {
    for (int camId : cams_from_lm[lmId]) {
      const auto landmarkFromCameraIt = landmarkFromCameraPerPart[partId].find(camId);
      int lmOfCaminPart = landmarkFromCameraIt == landmarkFromCameraPerPart[partId].end() ? 0 : landmarkFromCameraIt->second.size();
      if (partId == partFrom) {
        gainFrom += CostGain(lmOfCaminPart, lmOfCaminPart - 1, maxLmPerCam, temperature) - eps * lmOfCaminPart / min_res_per_cluster;
      }
      moveCost[partId] += CostGain(lmOfCaminPart, lmOfCaminPart + 1, maxLmPerCam, temperature) + eps * lmOfCaminPart / res_per_cluster[partId];
    }
  }

  int argmax = distance(moveCost.begin(), std::max_element(moveCost.begin(), moveCost.end()));
  return {gainFrom + moveCost[argmax], argmax};
}

// move from partId to return first part lmid return second. as the cost is positive.
std::pair<int, int> GetBestMoveCost(int partId, int camId, int kClusters,
                                    const std::vector<std::map<int, std::set<int>>> &landmarkFromCameraPerPart,
                                    const std::vector<std::set<int>>& cams_from_lm,
                                    const std::vector<int>& res_per_cluster,
                                    int maxLmPerCam, double temperature) {
  std::pair<int, int> bestLmAndPartId = {-1, -1};
  double bestCost = 0;
  const auto landmarkFromCameraIt = landmarkFromCameraPerPart[partId].find(camId);
  if (landmarkFromCameraIt == landmarkFromCameraPerPart[partId].end()){
    return bestLmAndPartId; // invalid since no landmarks observed by cam in part.
  };

  for (int lmId : landmarkFromCameraIt->second) {
    std::pair<double, int> costAndPart = GetMoveCost(lmId, partId, kClusters, landmarkFromCameraPerPart, cams_from_lm, res_per_cluster, maxLmPerCam, temperature);
    if (costAndPart.first > bestCost) {
      bestCost = costAndPart.first;
      bestLmAndPartId.first = costAndPart.second;
      bestLmAndPartId.second = lmId;
    }
  }
  return bestLmAndPartId;
}

void ApplyMove(int lmId, int partFrom, int partTo,
               const std::vector<std::set<int>>& cams_from_lm,
               std::vector<std::map<int, std::set<int>>> &landmarkFromCameraPerPart) {
  for (int camId : cams_from_lm[lmId])
  {
    landmarkFromCameraPerPart[partFrom][camId].erase(lmId);
    landmarkFromCameraPerPart[partTo][camId].insert(lmId);
  }
}

// res_to_cluster_by_landmark from post cluster. 
void recluster_cameras(
    int kClusters,
    const std::vector<int>& camera_indices_in,
    const std::vector<int>& landmark_indices_in,
    std::vector<int>& res_to_cluster_by_landmark) {

    const bool verbose = false;
    const int num_res = landmark_indices_in.size();
    const int num_cams = std::set<double>( camera_indices_in.begin(), camera_indices_in.end() ).size();
    const int num_lands = std::set<double>( landmark_indices_in.begin(), landmark_indices_in.end() ).size();

    if (camera_indices_in.size() != num_res || verbose) {
      std::cout << "Start #res " << num_res<< " " << kClusters << " #lnds" << num_lands << "  #cams " << num_cams << "\n";
      std::cout << " camera_indices_in " <<"\n";
      std::cout << " camera_indices_in " << camera_indices_in.size() << "\n";
      std::cout << " landmark_indices_in  " << landmark_indices_in.size() << "\n";
    }

    // 1. maps from lm to cameras and from cameras to landmarks
    // find cameras in part with few landmarks. So per part: map cam id -> landmarks seen and in part.
    std::vector<std::set<int>> cams_from_lm(num_lands);
    std::vector<std::set<int>> lms_from_cam(num_cams);
    std::vector<std::map<int, std::set<int>>> landmarkFromCameraPerPart(kClusters);
    std::vector<int> res_per_cluster(kClusters, 0);
    for (int res_id = 0; res_id < landmark_indices_in.size(); ++res_id) {
        const int lm_id = landmark_indices_in[res_id];
        const int cam_id = camera_indices_in[res_id];
        cams_from_lm[lm_id].insert(cam_id);
        lms_from_cam[cam_id].insert(lm_id);
        int partId = res_to_cluster_by_landmark[res_id];
        res_per_cluster[partId]++;
        landmarkFromCameraPerPart[partId][cam_id].insert(lm_id);
    }

    // simple step find part and camera with fewest landmarks.
    // try to move these (all res with landmark) -> all cameras would get landmark removed.
    // 1. count new landmarks per cam.
    // 2. per landmark find best fitting cluster. or jointly?
    //
    // could do annealing procedure cost before / after if above threshold apply.
    // only do with cams with few landmarks, pick the landmark.
    // eg move one lm from cam with 5 landmarks brings +1. 4: +2 3: +3 etc. 
    // accept with 1-exp(-cost * temperature), pick with  ? prevent x2 picking .. sigh.

    // A relevant cameras / parts. 
    static int maxLmPerCam = 6;
    static double temperature = 10;
    std::vector<std::vector<std::pair<int,int>>> relevantCameras(maxLmPerCam);

    // relevantCameras hold part and camId
    int movable = 0;
    int repeats = 30;
    std::vector<int> started(maxLmPerCam, 0);
    std::vector<int> finished(maxLmPerCam, 0);
    FillRelevantCameras(landmarkFromCameraPerPart, lms_from_cam, maxLmPerCam, relevantCameras);
    for(int camObservations = 1 ; camObservations < relevantCameras.size(); ++camObservations) {
      started[camObservations] = relevantCameras[camObservations].size();
      movable += started[camObservations];
    }

    while(movable > 0 && repeats >= 0) {
      --repeats;
      for (std::vector<std::pair<int, int>> relevantCamerasPerLm : relevantCameras) {
        for (std::pair<int, int> partAndCamIdx : relevantCamerasPerLm) {
          int fromPartId = partAndCamIdx.first;
          int camId = partAndCamIdx.second;
          const auto [toPartId, lmIdx] =
            GetBestMoveCost(fromPartId, camId, kClusters, // should consider # res in cluster as tie breaker.
              landmarkFromCameraPerPart, cams_from_lm, res_per_cluster, maxLmPerCam, temperature);
          if(toPartId>=0 && lmIdx>=0) {
            //std::cout << " Moving " << lmIdx << " from " << fromPartId << " observed by cam " << camId << " to " << toPartId << "\n";
            ApplyMove(lmIdx, fromPartId, toPartId, cams_from_lm, landmarkFromCameraPerPart);
            // could break if enters here after for loop end, to ensure we process small 1st. also could use pq.
          }
        }
      }
      movable = 0;
      FillRelevantCameras(landmarkFromCameraPerPart, lms_from_cam, maxLmPerCam, relevantCameras);
      for(int camObservations = 1 ; camObservations < relevantCameras.size(); ++camObservations) {
        finished[camObservations] = relevantCameras[camObservations].size();
        movable += finished[camObservations];
      }
      if (movable == 0 && maxLmPerCam < 20) {
        movable = 1;
        maxLmPerCam +=5;
        relevantCameras.resize(maxLmPerCam);
        finished.resize(maxLmPerCam);
      }
  }

  std::cout << "Cam observations started/finished: ";
  for(int camObservations = 1 ; camObservations < relevantCameras.size(); ++camObservations) {
      std::cout << camObservations << " : " << started[camObservations] << "/" << finished[camObservations] << ", ";
  }
  std::cout << " left " << movable << std::endl;


  // output res_to_cluster (_by_landmark)

  // map from cam/lm id to resid:
  std::map<int, int> camIdTimesLmIdToResId; 
  for(int res_id = 0; res_id < num_res; ++res_id) {
    const int lm_id = landmark_indices_in[res_id];
    const int cam_id = camera_indices_in[res_id];
    camIdTimesLmIdToResId[cam_id * num_lands + lm_id] = res_id;
  }

  for (int partId = 0; partId < landmarkFromCameraPerPart.size(); ++partId) {
    for (const auto [camIdx, lmIdSet] : landmarkFromCameraPerPart[partId]) {
      for (const int lmIdx : lmIdSet) {
        res_to_cluster_by_landmark[camIdTimesLmIdToResId[camIdx * num_lands + lmIdx]] = partId;
      }
    }
  }

  // fill vector : see above, done
// posst process clusters : in res_to_cluster_by_landmark, res to cam and res to lm.
// cluster should be whole landmark to cluster with all res in 1 cluster.
// problem cameras are split over clusters and some cameras and up with < 5 lms observed.
// idea is to move cams around a posteriori.
// per cluster -> cam to #lms. 0 is good, 1-5 is bad. > 5 ok.
// need to map from lm to all cams observing
// and from all cams to lms observed.
// find cams with < 5 lms and try move to other clusters.
// all of its lms must be moved now.
// for each of those lms, find cluster with 

// for smallest cluster find all cams with < 5 observations.
// try to add landmark from other clusters observed by that camera: 

// - for all lms observed by cam, get current cluster and cameras observing, 
// - check minimal # lms observed by any camera in this cluster. 
// - if # is large enough move landmark to this cluster.

// try to move landmarks observed by the cam to other clusters:
// - for lm find cluster with least cams observing it, move to that cluster.
// - if for all we do not introduce a new cam to the cluster : ok.
// - moving the landmarks reduces # observation for cam in cluster, ensure we do not create a new cam with < 5 lms.
// avoid to grow clsuter endlessly

}

    // for (int camId : cams_from_lm[lmId]) {
    //   const auto landmarkFromCameraIt = landmarkFromCameraPerPart[partId].find(camId);
    //   int lmOfCaminPart = landmarkFromCameraIt == landmarkFromCameraPerPart[partId].end() ? 0 : landmarkFromCameraIt->second.size();
    //   if (partId == partFrom) {
    //     gainFrom += CostGain(lmOfCaminPart, lmOfCaminPart - 1, maxLmPerCam, temperature) - eps * lmOfCaminPart / min_res_per_cluster;
    //   }

// for a part receive landmark from camera: cam -> landmarks oberserved.
// go over cams, for landmarks with only few observations -> merge
double GetCost(const std::map<int, std::set<int>> &landmarkFromCameraOfPart,
               int maxLmPerCam, double temperature) {
  double cost = 0;
  for(const auto& [cam, landmarksFromCam] : landmarkFromCameraOfPart){
    const int numLandmarks = landmarksFromCam.size();
    if (numLandmarks > maxLmPerCam) {continue;}
    cost += std::exp(-numLandmarks / static_cast<double>(maxLmPerCam) * temperature);
  }
  return cost; // could also return mean cost
}

double GetAverageCost(const std::map<int, std::set<int>> &landmarkFromCameraOfPart,
                      int maxLmPerCam, double temperature) {
  double cost = 0;
  int entries = 0;
  for(const auto& [cam, landmarksFromCam] : landmarkFromCameraOfPart){
    const int numLandmarks = landmarksFromCam.size();
    if (numLandmarks > maxLmPerCam) {continue;}
    cost += std::exp(-numLandmarks / static_cast<double>(maxLmPerCam) * temperature);
    entries++;
  }
  return cost / static_cast<double>(std::max(1,entries));
}

// return cams in part with fewest landamrk observations.
std::vector<int> GetLowestKCameras(const std::map<int, std::set<int>> &landmarkFromCameraOfPart, int topK, const std::vector<std::set<int>>& lms_from_cam) {
  std::vector<std::pair<int, int>> lmsInPartOfCam;
  // map cost to partId ? update by set cost to inf / update cost = heap.
  auto cmp = [&lmsInPartOfCam](int left, int right) {
  return lmsInPartOfCam[left].second > lmsInPartOfCam[right].second; // largest first
  };
  std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);

  for(const auto& [cam, landmarksFromCam] : landmarkFromCameraOfPart) {
    const int numLandmarks = landmarksFromCam.size(); // 0 cannot happen.

    // std::cout << "Cam  " << cam << " in part with " << numLandmarks << " landmarks\n";

    if(lms_from_cam[cam].size() <= numLandmarks) {continue;} // skip fully covered.

    if(pq.size() < topK) { //always push if less than desired
      lmsInPartOfCam.push_back({cam, numLandmarks});
      pq.push(lmsInPartOfCam.size() - 1);
    }
    if(pq.top() > numLandmarks) { // new is better (== random pick?)
      const int id = pq.top();
      pq.pop();
      lmsInPartOfCam[id] = {cam, numLandmarks}; // overwrite
      pq.push(id); // re enter in new place
    }
  }
  // best cams are in pq.
  std::vector<int> ids;
  while (!pq.empty()) {
    std::cout << "Top k Cam  " << lmsInPartOfCam[pq.top()].first << " with " << lmsInPartOfCam[pq.top()].second << " lms in part selected\n";
    ids.push_back(lmsInPartOfCam[pq.top()].first);
    pq.pop();
  }
  return ids;
}

// 2nd part and CostGain
std::pair<int, double> FindbestMatchForPart(int partId, const std::vector<std::set<int>>& cams_from_lm, 
  const std::vector<std::set<int>>& lms_from_cam, const std::map<int, std::set<int>> &landmarkFromCameraOfPart, 
  const std::vector<std::map<int, std::set<int>>>& landmarkFromCameraPerPart,
  const std::vector<int>& lmToPart, const std::vector<double>& costOfPart, int maxLmPerCam, double temperature) {
  // idea: find k cams with fewest landmarks in part.
  const int topK = 3;
  const int topL = 10;
  // cam in part with fewest landamrk observations in part.
  std::vector<int> camsToTryInPart = GetLowestKCameras(landmarkFromCameraOfPart, topK, lms_from_cam);
  // for each cam find a second different part also observing it, pick one with low costs

  std::cout << "Lowest k ";
  for(const auto& cam: camsToTryInPart) {
    std::cout << " cam " << cam <<  " ";
  }
  std::cout << "\n";

  std::set<int> partToTryMerge;

  // Again keep top k possibilities in Q ?
  for(int camToTry : camsToTryInPart) { // this cam has these 
    std::cout << "At camToTry " << camToTry << std::endl;
    std::set<int> landmarksInPart = landmarkFromCameraOfPart.at(camToTry); // the landmarks observed by the cam, can be just 1
    std::cout << "Ok\n";
    // for each lm here, find a potential partner.
    std::set<int> otherLandmarksObservedByCam = lms_from_cam[camToTry];

    std::vector<std::pair<int, double>> otherPartCost;
    // map cost to partId ? update by set cost to inf / update cost = heap.
    auto cmp = [&otherPartCost](int left, int right) {
      return otherPartCost[left].second > otherPartCost[right].second; // smallest first
    };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);

    std::cout << "Go over other landmarks observed by cam " << camToTry << "\n";
    std::set<int> checkedParts = {partId}; // TODO could keep outside! and / or even fill pq from all at once.
    for(int lmId : otherLandmarksObservedByCam) {
      int otherPartId = lmToPart[lmId];
      if (checkedParts.find(otherPartId) != checkedParts.end()) {continue;}
      checkedParts.insert(otherPartId);
      const double cost = costOfPart[otherPartId];
      if (cost < 0) {continue;}

      //std::cout << "lm " << lmId << " " << cost << "\n";

      if(pq.size() < topL) { // always push if less than desired
        otherPartCost.push_back({otherPartId, cost});
        //std::cout << "Pushing "  << otherPartId << " with " << cost << " for merge\n";
        pq.push(otherPartCost.size() - 1);
        continue;
      }

      if(otherPartCost[pq.top()].second < cost) { // new is better (== random pick?)
        const int id = pq.top();
        pq.pop();
        //std::cout << "Considering "  << otherPartId << " with " << cost << " for merge replacing " << otherPartCost[id].first << " c: " << otherPartCost[id].second << "\n";
        otherPartCost[id] = {otherPartId, cost}; // overwrite
        pq.push(id); // re enter in a new place
      }
    }

    for (const auto [id, cost] : otherPartCost) {
      partToTryMerge.insert(id);
      std::cout << "Using part "  << id << " with " << cost << " for possible merge top cost: " << otherPartCost[pq.top()].second << "\n";
      pq.pop();
    }
  }
  
  // compute merge gain
  int bestPartToMerge = -1;
  double bestCostGain = -1000;
  for(int otherPartId : partToTryMerge) {
    const double oldCost = costOfPart[partId] + costOfPart[otherPartId];
    std::map<int, std::set<int>> landmarkFromCameraOfOtherPart = landmarkFromCameraPerPart[otherPartId];
    // merge landmarkFromCameraOfOtherPart and landmarkFromCameraOfPart and compute new cost.
    for(const auto&[cam, lms] : landmarkFromCameraOfPart) {
      landmarkFromCameraOfOtherPart[cam].insert(lms.begin(), lms.end());
    }
    const double newCost = GetCost(landmarkFromCameraOfOtherPart, maxLmPerCam, temperature);
    const double costGain = oldCost - newCost;
    std::cout << "Own part "  << partId << " with cost " << costOfPart[partId] << " and merge  Part " << otherPartId 
              << " with cost " << costOfPart[otherPartId] << " for possible merge with new cost " << newCost << " = " << costGain << " vs " << bestCostGain << "\n";

    if(oldCost - newCost > bestCostGain){
      bestCostGain = oldCost - newCost;
      bestPartToMerge = otherPartId;
    }
  }
  std::cout << "Found Best part " << bestPartToMerge << " and cost gain " << bestCostGain << "\n";
   return {bestPartToMerge, bestCostGain};
  }

// update a lot. costs, invalidate one part, cma to lms, .. 
double MergeParts(int partId, int otherPartId,
      std::vector<std::map<int, std::set<int>>>& landmarkFromCameraPerPart,
      std::vector<int>& res_per_cluster,
      std::vector<int>& lmToPart, 
      std::vector<double>& costOfPart,
      int maxLmPerCam, double temperature) {
    std::map<int, std::set<int>>& landmarkFromCameraOfPart = landmarkFromCameraPerPart[partId];
    std::map<int, std::set<int>>& landmarkFromCameraOfOtherPart = landmarkFromCameraPerPart[otherPartId];
    // merge landmarkFromCameraOfOtherPart and landmarkFromCameraOfPart and compute new cost.
    for(const auto&[cam, lms] : landmarkFromCameraOfOtherPart) {
      landmarkFromCameraOfPart[cam].insert(lms.begin(), lms.end());
      for(const int lm : lms) { // way too many != lms in part
        lmToPart[lm] = partId;
      }
    }
    
    costOfPart[otherPartId] = -100;
    landmarkFromCameraOfOtherPart.clear();
    res_per_cluster[partId] += res_per_cluster[otherPartId];
    return GetCost(landmarkFromCameraPerPart[partId], maxLmPerCam, temperature);
    // 'disable' one part. update the other. 
    // i popped one already, push this back in. Set cost to -inf for other part [we can drop this one, when popping it just looking up its cost]
}

void cluster_cameras_degeneracy(
    int kClusters,
    const std::vector<int>& camera_indices_in,  // per res -> cam involved
    const std::vector<int>& landmark_indices_in,// per res -> landmark involved
    std::vector<int>& res_to_cluster_by_landmark_out) {

    const bool verbose = false;
    const int num_res = landmark_indices_in.size();
    const int num_cams = std::set<int>( camera_indices_in.begin(), camera_indices_in.end() ).size();
    const int num_lands = std::set<int>( landmark_indices_in.begin(), landmark_indices_in.end() ).size();

    if (true || camera_indices_in.size() != num_res || verbose) {
      std::cout << "Start #res " << num_res<< " " << kClusters << " #lnds" << num_lands << "  #cams " << num_cams << "\n";
      std::cout << " camera_indices_in " <<"\n";
      std::cout << " camera_indices_in " << camera_indices_in.size() << "\n";
      std::cout << " landmark_indices_in  " << landmark_indices_in.size() << "\n";
    }

    static int maxLmPerCam = 10;
    static double temperature = 20;
    static int nLowestPartsToTry = 1;

    // 2. map lm to cluster index. start each lm is a cluster.
    // RemapToRootInVtxToPartMap
    std::vector<int> lmToPart(num_lands);
    std::iota(lmToPart.begin(), lmToPart.end(), 0);

    // 1. maps from lm to cameras and from cameras to landmarks
    // find cameras in part with few landmarks. So per part: map cam id -> landmarks seen and in part.
    std::vector<std::set<int>> cams_from_lm(num_lands);
    std::vector<std::set<int>> lms_from_cam(num_cams);
    std::vector<std::map<int, std::set<int>>> landmarkFromCameraPerPart(num_lands);
    std::vector<int> res_per_cluster(num_lands, 0);
    for (int res_id = 0; res_id < landmark_indices_in.size(); ++res_id) {
        const int lm_id = landmark_indices_in[res_id];
        const int cam_id = camera_indices_in[res_id];
        cams_from_lm[lm_id].insert(cam_id);
        lms_from_cam[cam_id].insert(lm_id);
        int partId = lmToPart[lm_id]; // identity at start
        res_per_cluster[partId]++;
        landmarkFromCameraPerPart[partId][cam_id].insert(lm_id);
    }
    std::cout << " landmarkFromCameraPerPart  " << landmarkFromCameraPerPart.size() << " done\n";

    // 3. cluster to cam involved and counts use landmarkFromCameraPerPart
    // 4. compute cost per part. init.
    std::vector<double> costOfPart;
    // map cost to partId ? update by set cost to inf / update cost = heap.
    auto cmp = [&costOfPart](int left, int right) {
    return costOfPart[left] < costOfPart[right]; // highest cost 1st best merge candidates
    };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);

    for(int partId=0;partId < num_lands; partId++ ) {
      const double cost = GetCost(landmarkFromCameraPerPart[partId], maxLmPerCam, temperature);
      //std::cout << "Insert PartId  " << partId << " cost " << cost << " \n";
      costOfPart.push_back(cost);
      pq.push(partId);
    }

    int num_parts = num_lands;
    while (!pq.empty() && num_parts > kClusters) {

      const int partId = pq.top();
      if(costOfPart[partId] < 0) {pq.pop();continue;} // invalid.

      std::cout << "PartId  " << partId << " cost " << costOfPart[partId] << " \n";

      // 0. select part to try for a merge
      // 1. select parts to merge     
      // 2. merge & update costs
      std::map<double, std::pair<int,int>> partsToTry;
      for(int partToTryId = 0; partToTryId < std::min(nLowestPartsToTry, num_parts-1); partToTryId++){
        const int partId = pq.top();
        pq.pop();
        std::cout << "Try PartId  " << partId << " cost " << costOfPart[partId] << " \n";
        std::pair<int, double> partAndGain = // 2nd part and CostGain
          FindbestMatchForPart(partId, cams_from_lm, lms_from_cam, landmarkFromCameraPerPart[partId], 
            landmarkFromCameraPerPart, lmToPart, costOfPart, maxLmPerCam,temperature);
        std::cout << " partAndGain  " << partAndGain.first << " gain " << partAndGain.second << " \n";
        partsToTry[partAndGain.second] = {partId, partAndGain.first};
      }

      // per part to try find best match(es).
      if(!partsToTry.empty() && (partsToTry.begin()->second.first >=0)) {
      std::pair<int,int> partsToMerge = partsToTry.begin()->second; // 
      std::cout << " Merge  " << partsToMerge.first << " " << partsToMerge.second << " gain " << partsToTry.begin()->first << " \n";
      const double newCost = MergeParts(partsToMerge.first, partsToMerge.second,
            landmarkFromCameraPerPart,
            res_per_cluster,
            lmToPart, 
            costOfPart,
            maxLmPerCam, 
            temperature);

      costOfPart[partId] = newCost;
      pq.push(partId);
      }
      else {
        std::cout << "Did not find overlap !? for part " <<  partId << " to cover\n"; // 0 cost parts should be allowed?
      }
    }

  for (auto& lmToP : lmToPart) {
    lmToP = lmToPart[lmToP];
  }
  int numParts = 0;
  std::vector<int> idToConsecutiveId(num_lands, -1);
  for (int bbId = 0; bbId < num_lands; ++bbId) {
    const int partId = lmToPart[bbId];
    if (partId == bbId) {  // Keep order of bbIds in consecutive partIds.
      idToConsecutiveId[partId] = numParts++;
    }
  }
  for (int bbId = 0; bbId < num_lands; ++bbId) {
    lmToPart[bbId] = idToConsecutiveId[lmToPart[bbId]];
  }

  // go over res -> lmid use lmid to look up part id
  std::vector<int> res_to_cluster_by_landmark(num_res, -1);
  for (int res_id=0; res_id < landmark_indices_in.size(); res_id++) {
      const int lm_id = landmark_indices_in[res_id];
      res_to_cluster_by_landmark[res_id] = lmToPart[lm_id];
  }

  fill_vec(res_to_cluster_by_landmark_out, res_to_cluster_by_landmark);

  }
