// returns covered_landmark_indices_c: those in addition! -- which is weird -- to those covered by main residual split
// the latter are point_indices_already_covered and also returned
// Order landmark shift by 
#define _select_by_even_cost_

// Compltetely new idea:
// define landmark weight li, lj as number of cams shared.
// could also be relative? random walk idea?
// problem: this is very slow.
// again all res of lm go into cluster, also evenly distributed.
// 
// take cam w. fewest lms if e.g. < 10 lms. 
// all lms merge into 1 cluster.
// while cams with < 10 lms exist.

// Should work .. but it does not. Different random seeds needed .. lol?
// with    Cam observations started/finished: 1 : 312/107, 2 : 1017/74, 3 : 1411/45, 4 : 1478/39, 5 : 1271/25,  left 290
//         Cam observations started/finished: 1 : 10/3, 2 : 7/3, 3 : 4/4, 4 : 2/7, 5 : 1/8, 6 : 24/27, 7 : 43/40, 8 : 68/72, 9 : 72/70, 10 : 92/91, 11 : 116/118, 12 : 133/126, 13 : 162/166, 14 : 213/207,  left 942
// without Cam observations started/finished: 1 : 254/101, 2 : 867/68, 3 : 1275/37, 4 : 1446/27, 5 : 1084/29,  left 262
//         Cam observations started/finished: 1 : 15/5, 2 : 9/7, 3 : 3/3, 4 : 2/7, 5 : 0/6, 6 : 16/21, 7 : 35/36, 8 : 68/68, 9 : 76/74, 10 : 87/88, 11 : 154/152, 12 : 155/154, 13 : 220/222, 14 : 290/281,  left 1124
#define __clusteridentical_lms_early__ // with this 3068 is AWFUL? without also.

#define __testThisNew__

#include "process_clusters.h"

#include <omp.h>
#include <assert.h>
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

//#define _rngseed_ 999 // 529k
//#define _rngseed_ 123 // 527k, but intermediate (60 its) results bad. Still: BEST overall
//#define _rngseed_ 456 works for large DL
//#define _rngseed_ 666 // be2 552k
#define _rngseed_ 4567 // be2 547k // BEST
// find new best seed for __clusteridentical_lms_early__ could also compare 9 clusters or 11 with wither.
//#define _rngseed_ 9753 // SHIT at 3068 -- all

#define _order_div_mult_  1e-1

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

    // std::mt19937 mt{ static_cast<std::mt19937::result_type>(
		// std::chrono::steady_clock::now().time_since_epoch().count() ) };
    std::mt19937 mt{ static_cast<std::mt19937::result_type>( _rngseed_ ) };
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

// relevantCameras hold part and camId, maps from #lms obsered by cam in part. 
void FillRelevantCameras(const std::vector<std::map<int, std::set<int>>> &landmarkFromCameraPerPart,
                         const std::vector<std::set<int>>& lms_from_cam,
                         int maxLmPerCam,
                         std::vector<std::vector<std::pair<int, int>>> &relevantCameras) {
  relevantCameras.clear();
  relevantCameras.resize(maxLmPerCam);
  for (int partId = 0; partId < landmarkFromCameraPerPart.size(); partId++) {
    const std::map<int, std::set<int>>& camToLmsInPart = landmarkFromCameraPerPart[partId];
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

// this is twisted for removal of lms towards 0 (ideal), yet 1 has to better than 2, to reach 0
double CostGainDecrease(int numLmsBefore, int numLmsAfter, 
                int maxLmPerCam, double temperature) {
  if (std::min(numLmsBefore, numLmsAfter) > maxLmPerCam ||
      numLmsBefore == 0 && numLmsAfter <= 0) {
    return 0.0;
  }
  // 0/1: 1 - exp(-t/5), should be bad. 1/0 should be good.
  // 1/2 exp(-t/5) - exp(-2t/5)
  // exp(-6) - exp(-5) is negative. 
  return std::exp(-numLmsAfter / static_cast<double>(maxLmPerCam) * temperature) - 
         std::exp(-numLmsBefore / static_cast<double>(maxLmPerCam) * temperature);
}

// adding a lm is positive 5 better 4 but 0 better 1 (much better)
double CostGainIncrease(int numLmsBefore, int numLmsAfter, 
                int maxLmPerCam, double temperature) {
  if (std::min(numLmsBefore, numLmsAfter) > maxLmPerCam ||
      numLmsBefore == 0 && numLmsAfter <= 0) {
    return 0.0;
  }

  if (numLmsBefore == 0) return - 100;// - std::exp(-numLmsAfter / static_cast<double>(maxLmPerCam) * temperature);
  // 0/1: 1 - exp(-t/5), should be bad. 1/0 should be good.
  // 1/2 exp(-t/5) - exp(-2t/5)
  // exp(-6) - exp(-5) is negative. 
  return std::exp(-numLmsBefore / static_cast<double>(maxLmPerCam) * temperature) - 
         std::exp(-numLmsAfter / static_cast<double>(maxLmPerCam) * temperature);
}

// cost with known lmid. move lmId from partFrom to the best cluster wrt. cost.
// if receive == true: move lmId TO partFromTo from the best cluster wrt. cost.
std::pair<double, int> GetMoveCost(int lmId, int partFromTo, int kClusters,
                                   const std::vector<std::map<int, std::set<int>>> &landmarkFromCameraPerPart,
                                   const std::vector<std::set<int>>& cams_from_lm,
                                   const std::vector<int>& res_per_cluster,
                                   int maxLmPerCam, double temperature) {
  // per part getCostGain: here just the new min number of landmarks in the part.
  std::vector<double> moveCost(kClusters); // cost per part to move to?
  double gainFromTo = 0;
  constexpr double eps = 1e-6; // prefer cluster with fewer res, tiny bias? 
  int min_res_per_cluster = res_per_cluster[partFromTo]; // TODO: this appears weird.
  for (int partId = 0; partId < kClusters; ++partId) {
    min_res_per_cluster = std::min(min_res_per_cluster, res_per_cluster[partId]);
  }
  // cost to move lm to partId is defined by cam observations and 
  // the change in cost by the change in cam obs, so sum over cam obesrving the lm.
  for (int partId = 0; partId < kClusters; ++partId) {
    for (int camId : cams_from_lm[lmId]) { // go over all cams observing lm.
      const auto landmarkFromCameraIt = landmarkFromCameraPerPart[partId].find(camId);
      int lmOfCaminPart = (landmarkFromCameraIt == landmarkFromCameraPerPart[partId].end()) ? 0 : landmarkFromCameraIt->second.size();
      // lmOfCaminPart holds #observations of cam in part.
      if (partId == partFromTo) { // remove the lm from part yields this cost.
          gainFromTo += CostGainDecrease(lmOfCaminPart, lmOfCaminPart - 1, maxLmPerCam, temperature) - eps * lmOfCaminPart / min_res_per_cluster;
      }
      moveCost[partId] += CostGainDecrease(lmOfCaminPart, lmOfCaminPart + 1, maxLmPerCam, temperature) - eps * lmOfCaminPart / res_per_cluster[partId];
    }
  }

  moveCost[partFromTo] = -gainFromTo; // TODO was this important to get better results?
  int argmax = distance(moveCost.begin(), std::max_element(moveCost.begin(), moveCost.end()));
  return {gainFromTo + moveCost[argmax], argmax};
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

  for (int lmId : landmarkFromCameraIt->second) { // for all lms seen by the camera.
    // if receive: cost to receive lm to partId from anthoer part.
    // else cost to move lm from partId to another part.
    std::pair<double, int> costAndPart = GetMoveCost(lmId, partId, kClusters, landmarkFromCameraPerPart, 
                                                     cams_from_lm, res_per_cluster, maxLmPerCam, temperature);

    // cost to move lmId from partId to another part OR to receive lmId from another part.
    if (costAndPart.first > bestCost) {
      bestCost = costAndPart.first;
      bestLmAndPartId.first = costAndPart.second;
      bestLmAndPartId.second = lmId;
    }
  }
  return bestLmAndPartId;
}

// move from partId to return first part lmid return second. as the cost is positive.
std::pair<int, int> GetBestMoveCostToCamInPart(int partId, int inCamId, int kClusters,
                                    const std::vector<std::map<int, std::set<int>>> &landmarkToCameraPerPart,
                                    const std::vector<std::set<int>>& cams_from_lm,
                                    const std::vector<int>& res_per_cluster,
                                    int maxLmPerCam, double temperature) {
  std::pair<int, int> bestLmAndPartId = {-1, -1};
  double bestCost = 0;
  //const double eps = 0;
  const auto landmarkToCameraIt = landmarkToCameraPerPart[partId].find(inCamId);
  if (landmarkToCameraIt == landmarkToCameraPerPart[partId].end()){
    return bestLmAndPartId; // invalid since no landmarks observed by cam in part.
  };

  // go over lms not present in cluster, but in seen by cam. go over cluster find cam, go over the landmarks in this cluster, compute the cost
  for (int fromPartId = 0; fromPartId < kClusters; ++fromPartId) {
    if (fromPartId == partId) {continue;}
    const auto landmarkFromCameraIt = landmarkToCameraPerPart[fromPartId].find(inCamId);
    if (landmarkFromCameraIt == landmarkToCameraPerPart[fromPartId].end()) { continue;}

    // only consider lms that exist in other part. 
    for (int lmId : landmarkFromCameraIt->second) { // cost change remove lm from frompart, add to to partId
      double costForMovingLm = 0;
      for (int camId : cams_from_lm[lmId]) { // go over all cams observing lm.

        const auto landmarkFromCameraIt = landmarkToCameraPerPart[fromPartId].find(camId);
        const auto landmarkToCameraIt = landmarkToCameraPerPart[partId].find(camId);

        int lmFromCaminPart = (landmarkFromCameraIt == landmarkToCameraPerPart[fromPartId].end()) ? 0 : landmarkFromCameraIt->second.size();
        int lmToCaminPart = (landmarkToCameraIt == landmarkToCameraPerPart[partId].end()) ? 0 : landmarkToCameraIt->second.size();

        costForMovingLm += CostGainIncrease(lmFromCaminPart, lmFromCaminPart - 1, maxLmPerCam, temperature);// - eps * lmOfCaminPart / min_res_per_cluster;
        costForMovingLm += CostGainIncrease(lmToCaminPart, lmToCaminPart + 1, maxLmPerCam, temperature);// - eps * lmOfCaminPart / res_per_cluster[partId];

         //std::cout << "Cam " << camId << " lmFromCaminPart " << lmFromCaminPart << "-1 " << " lmToCaminPart " << lmToCaminPart << "+1 , costForMovingLm: " <<  costForMovingLm<< std::endl;

        }
      // cost to move lmId from partId to another part OR to receive lmId from another part.
      //std::cout << " Total Cost " << costForMovingLm << std::endl;
      if (costForMovingLm > bestCost) {
        bestCost = costForMovingLm;
        bestLmAndPartId.first = fromPartId;
        bestLmAndPartId.second = lmId;
      }
    }
  }
  return bestLmAndPartId;
}

// res_per_cluster adjust !
void ApplyMove(int lmId, int partFrom, int partTo,
               const std::vector<std::set<int>>& cams_from_lm,
               std::vector<std::map<int, std::set<int>>> &landmarkFromCameraPerPart,
               std::vector<int>& res_per_cluster) {
  if(partFrom ==partTo) return;
  for (int camId : cams_from_lm[lmId]) { // all cams from lm are in partFrom. 
    landmarkFromCameraPerPart[partFrom][camId].erase(lmId);
    landmarkFromCameraPerPart[partTo][camId].insert(lmId); // can be new part, but unlikely as cost high.
    if (landmarkFromCameraPerPart[partFrom][camId].empty()) {
      landmarkFromCameraPerPart[partFrom].erase(camId);
    }
  }
  res_per_cluster[partFrom] -= cams_from_lm[lmId].size();
  res_per_cluster[partTo] += cams_from_lm[lmId].size();
}

int GetClusterByNumRes(int res_id, std::vector<int> res_per_cluster) {
  const int nCluster = res_per_cluster.size();
  int sum_res = 0;
  for(int i = 0; i < nCluster; ++i) {
    sum_res += res_per_cluster[i];
    if (res_id < sum_res) {
      return i;
    }
  }
  return nCluster - 1;
}

template <typename T>
std::vector<int> SortIndices(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  std::stable_sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) {return v[i1] < v[i2];});

  return idx;
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
    static int maxLmPerCam = 6; // ?
    static double temperature = 10;
    std::vector<std::vector<std::pair<int,int>>> relevantCameras(maxLmPerCam); // cam obs in part -> cam id.

    // relevantCameras hold part and camId
    int movable = 0;
    int repeats = 30;
    std::vector<int> started(maxLmPerCam, 0);
    std::vector<int> finished(maxLmPerCam, 0);

//#ifdef __disabled__for__testing__
// temperature = 30;  // does something -- but, if not good start here not much gain.
for(int runs = 0; runs < 3; ++runs) { // more uns do not change things WTF?

    // new:
    maxLmPerCam = 6;
    relevantCameras.clear();
    relevantCameras.resize(maxLmPerCam); // cam obs in part -> part and camId
    temperature -= runs;
    movable = 0;
    repeats = 30;
    started.clear();started.resize(maxLmPerCam, 0);
    finished.clear();finished.resize(maxLmPerCam, 0);
    // end new

    // relevantCameras maps from #lms obsered by cam in part. e.g. relevantCameras[1] -> all cams and part seeing 1 lm only in part.
    // Those are candidayes to receive / send lms. Likely we should try BOTH here? 
    // curently we HERE only send from cam with few to other cam/part. 
    FillRelevantCameras(landmarkFromCameraPerPart, lms_from_cam, maxLmPerCam, relevantCameras);
    for(int camObservations = 1 ; camObservations < relevantCameras.size(); ++camObservations) {
      started[camObservations] = relevantCameras[camObservations].size();
      movable += started[camObservations];
    }

    // TODO: lower parity in #residuals over time here. lower temperature over time? residuals cost is done how -- why not entropy?
    while(movable > 0 && repeats >= 0) {
      --repeats;
      for (std::vector<std::pair<int, int>> relevantCamerasPerLm : relevantCameras) {
        for (std::pair<int, int> partAndCamIdx : relevantCamerasPerLm) {
          int fromPartId = partAndCamIdx.first;
          int camId = partAndCamIdx.second;
          do { // loop here until no more moves possible?
          const auto [toPartId, lmIdx] =
            GetBestMoveCost(fromPartId, camId, kClusters, // should consider # res in cluster as tie breaker.
              landmarkFromCameraPerPart, cams_from_lm, res_per_cluster, maxLmPerCam, temperature);
          if(toPartId>=0 && lmIdx>=0) {
            //std::cout << " Moving " << lmIdx << " from " << fromPartId << " observed by cam " << camId << " to " << toPartId << "\n";
            ApplyMove(lmIdx, fromPartId, toPartId, cams_from_lm, landmarkFromCameraPerPart, res_per_cluster);
            // could break if enters here after for loop end, to ensure we process small 1st. also could use pq.
          }else{break;}
          } while(false); // set to false to recive original behavior.
        }
        // Maybe we need to treat 2 lms by cam 1st?
        //Cam observations started/finished: 1 : 347/35, 2 : 431/25, 3 : 380/14, 4 : 382/11, 5 : 337/9,  left 94: true
        //Cam observations started/finished: 1 : 347/34, 2 : 431/28, 3 : 380/11, 4 : 382/10, 5 : 337/5,  left 88: false
      }

      movable = 0;
      FillRelevantCameras(landmarkFromCameraPerPart, lms_from_cam, maxLmPerCam, relevantCameras);
      for(int camObservations = 1 ; camObservations < relevantCameras.size(); ++camObservations) {
        finished[camObservations] = relevantCameras[camObservations].size();
        movable += finished[camObservations];
      }
      if (movable == 0 && maxLmPerCam < 10) {
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

//#endif

//#ifdef __disabled__for__testing__

  // TODO:
  // 1.
  // even first: cams with < 10 lms in cluster:
  // go over lms seen from camera
  // go over cluster of those lm.
  // move the lm to the critical cluster does not introduce new lm? Yes -> move


    // a. find cluster with cam with < K obs in cluster
    // b. go over .. see above
    maxLmPerCam = 15;
    relevantCameras.clear();
    relevantCameras.resize(maxLmPerCam); // cam obs in part -> part and camId
    movable = 0;
    repeats = 30;
    started.clear();started.resize(maxLmPerCam, 0);
    finished.clear();finished.resize(maxLmPerCam, 0);
    FillRelevantCameras(landmarkFromCameraPerPart, lms_from_cam, maxLmPerCam, relevantCameras);
    for(int camObservations = 1 ; camObservations < relevantCameras.size(); ++camObservations) {
      started[camObservations] = relevantCameras[camObservations].size();
      movable += started[camObservations];
    }

    while(movable > 0 && repeats >= 0) {
      --repeats;
      for (std::vector<std::pair<int, int>> relevantCamerasPerLm : relevantCameras) {
        // now inverse: find a lm in other part to move into this part.
        for (std::pair<int, int> partAndCamIdx : relevantCamerasPerLm) {
          int toPartId = partAndCamIdx.first;
          int camId = partAndCamIdx.second; // this might belong to toPartId, not from partid now. So add a lm seen by this cam in this part to the part, from some other part.
          const auto [fromPartId, lmIdx] =
            GetBestMoveCostToCamInPart(toPartId, camId, kClusters, // should consider # res in cluster as tie breaker.
              landmarkFromCameraPerPart, cams_from_lm, res_per_cluster, maxLmPerCam, temperature);
          if (toPartId >= 0 && lmIdx >= 0) {
  
            // std::cout << "Part " << fromPartId << " contains in cam " << camId << " " << (landmarkFromCameraPerPart[fromPartId].at(camId).find(lmIdx) != landmarkFromCameraPerPart[fromPartId].at(camId).end()) <<
            // " and Part " << toPartId << " contains in cam " << camId << " " << (landmarkFromCameraPerPart[toPartId].at(camId).find(lmIdx) != landmarkFromCameraPerPart[toPartId].at(camId).end()) << "\n";

            // std::cout << " Moving " << lmIdx << " from part " << fromPartId << " observed by cam " << camId << " to part " << toPartId << 
            // " from cam obs " << landmarkFromCameraPerPart[fromPartId].at(camId).size() << " to cam obs  " << landmarkFromCameraPerPart[toPartId].at(camId).size() << "\n";
            
            ApplyMove(lmIdx, fromPartId, toPartId, cams_from_lm, landmarkFromCameraPerPart, res_per_cluster);

            // std::cout << " After Moving " << lmIdx << " from part " << fromPartId << " observed by cam " << camId << " to part " << toPartId << 
            // " from cam obs " << landmarkFromCameraPerPart[fromPartId].at(camId).size() << " to cam obs  " << landmarkFromCameraPerPart[toPartId].at(camId).size() << "\n";

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
      if (movable == 0 && maxLmPerCam < 10) {
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

} // repeats
//#endif

  // inverse: find cams in small clusters. those cams observe lms in other clusters.
  // those lms are candidates to move over to the small cluster if
  // all cams observing the lm are in the small cluster and in the big cluster the cams observing the lm are ok with removeg the lm.
  // 
  // pick cluster at random
  // candidate cl are larger cluster now identify set
  // go over cams of cl. [cam, lms] map
  // pick one with lm not in cluster but: lm -> cluster and lm->cam all these cams are already present in cluster
  // needs lm -> cluster map = vector
  const int minLmObsPerCamInPart = maxLmPerCam;
  //const int num_res_tresh = 1.1 * num_res / kClusters; // consider as balanced.
  //const int av_res_tresh = num_res / kClusters; // consider as balanced.
  // std::mt19937 mt{ static_cast<std::mt19937::result_type>(
  //   std::chrono::steady_clock::now().time_since_epoch().count() ) };
  std::mt19937 mt{ static_cast<std::mt19937::result_type>( _rngseed_ ) };
  std::uniform_int_distribution sample_cl{ 0, kClusters-1 };
  //std::uniform_int_distribution sample_cam{ 0, num_cams-1 };
  std::uniform_int_distribution sample_lm{ 0, num_lands-1 };

  std::vector<int> lm_to_part(num_lands, 0);
  for (int partId = 0; partId < landmarkFromCameraPerPart.size(); ++partId) {
    for (const auto& [camIdx, lmIdSet] : landmarkFromCameraPerPart[partId]) {
      for (const int lmIdx : lmIdSet) {
        lm_to_part[lmIdx] = partId;
      }
    }
  }

  for (int partId = 0; partId < res_per_cluster.size(); ++partId) {
    std::cout << "Part " << partId << " with " << res_per_cluster[partId] << " residuals\n";
  }
#ifdef __old_version__
  constexpr int n_LmSamples = 10; // will hit often a lm in larger cl. 
  constexpr int max_consecutive_failures = 100;
  int consecutive_failures = 0;
  const int maxMoves = 500000;
  int moves = 0;
  int passes = 0;
  std::vector<int> res_order = SortIndices(res_per_cluster);
  while (consecutive_failures < max_consecutive_failures && moves < maxMoves) {
    ++consecutive_failures;
    //const int receive_candidate_cluster_id = sample_cl(mt); // maybe better in order, smallest 1st?
    const int receive_candidate_cluster_id = res_order[passes++ % res_order.size()];

    //std::cout << " Receive Candidate cl " << receive_candidate_cluster_id << " with " << res_per_cluster[receive_candidate_cluster_id] << " res \n";    
    bool noCandidates = true;
    for (int c = 0; c < kClusters; ++c) {
      if (res_per_cluster[receive_candidate_cluster_id] + 3 < res_per_cluster[c]) {
        //std::cout << "Candidate cl " << c << " with " << res_per_cluster[c] << " res \n";
        noCandidates = false;
        break;
      }
    }
    if (noCandidates) {continue;}

    // N times: sample lmid, or cand + lm from cand. test all cams present in this cl.
    // if yes check if cams of cand cl would remain ok.
    int consecutive_samples_fail = 0;
    while(consecutive_samples_fail < n_LmSamples && moves < maxMoves) {
      consecutive_samples_fail++;
      const int candidate_lm_id = sample_lm(mt);
      const int move_candidate_cluster_id = lm_to_part[candidate_lm_id];

      // why not sample lm -> moving part
      // sample receiving part / or try all possible? 

      if (move_candidate_cluster_id == receive_candidate_cluster_id || 
          res_per_cluster[receive_candidate_cluster_id] + 2 * cams_from_lm[candidate_lm_id].size() > res_per_cluster[move_candidate_cluster_id]) {
        //std::cout << " move_candidate_cluster_id " << move_candidate_cluster_id  << " skipped\n";
        continue;
      }
      // else{
      //   std::cout << " move_candidate_cluster_id " << move_candidate_cluster_id  << " accepted \n";
      // }

      const int num_cams_in_receive_cluster = landmarkFromCameraPerPart[receive_candidate_cluster_id].size();
      if (num_cams_in_receive_cluster <= cams_from_lm[candidate_lm_id].size()) {
          continue;
      };

      bool fail = false;
      // test 1 are ALL cams observed by lm present in receive cluster. Those cams are rare i suppose. 
      for (int camIdx : cams_from_lm[candidate_lm_id]) {
        if (landmarkFromCameraPerPart[receive_candidate_cluster_id].find(camIdx) ==
            landmarkFromCameraPerPart[receive_candidate_cluster_id].end()) {
              fail = true;
              break; // not a candidate.
          }
      }
      if (fail) {continue;}

      // test 2: removing the lm from this cluster does not lead to degenerate situations.
      for (const auto& [camIdx, lmIdSet] : landmarkFromCameraPerPart[move_candidate_cluster_id]) {
        if (lmIdSet.size() < minLmObsPerCamInPart + 1 && lmIdSet.find(candidate_lm_id) != lmIdSet.end()) {
          fail = true;
          break; // not a candidate.
        }
      }
      if(fail) {continue;}

      // std::cout << consecutive_failures << "/" << consecutive_samples_fail 
      //           << " Moving lm " << candidate_lm_id << " from "<< move_candidate_cluster_id << " to " << receive_candidate_cluster_id 
      //           << " #res: " << cams_from_lm[candidate_lm_id].size() << " of " << res_per_cluster[move_candidate_cluster_id] 
      //           << " -> " << res_per_cluster[receive_candidate_cluster_id] << "\n";

      // per form move 
      ApplyMove(candidate_lm_id, move_candidate_cluster_id, receive_candidate_cluster_id, 
                cams_from_lm, landmarkFromCameraPerPart, res_per_cluster);
      lm_to_part[candidate_lm_id] = receive_candidate_cluster_id;
      consecutive_failures = 0;
      moves++;
      consecutive_samples_fail = 0;
      // break; // or maybe not as this cluster appears movable?
    } // samples

    std::cout << "Expanding " << receive_candidate_cluster_id << "\n";
    for (int partId = 0; partId < res_per_cluster.size(); ++partId) {
      std::cout << "Part " << partId << " with " << res_per_cluster[partId] << " residuals\n";
    }

  } // while trying to move landmarks.

#else

  constexpr int max_consecutive_failures = 500;
  int consecutive_failures = 0;
  const int maxMoves = 500000;
  int moves = 0;
  int last_receive_cluster = 0;
  while (consecutive_failures < max_consecutive_failures && moves < maxMoves) {
    ++consecutive_failures;

    const int candidate_lm_id = sample_lm(mt);
    const int move_candidate_cluster_id = lm_to_part[candidate_lm_id];

    // try last succesful cl again?
    const int receive_candidate_cluster_id = (consecutive_failures == 1) ? last_receive_cluster : sample_cl(mt);

    if (move_candidate_cluster_id == receive_candidate_cluster_id || 
        res_per_cluster[receive_candidate_cluster_id] + 2 * cams_from_lm[candidate_lm_id].size() > res_per_cluster[move_candidate_cluster_id]) {
      //std::cout << " move_candidate_cluster_id " << move_candidate_cluster_id  << " skipped\n";
      continue;
    }
      // else{
      //   std::cout << " move_candidate_cluster_id " << move_candidate_cluster_id  << " accepted \n";
      // }

      const int num_cams_in_receive_cluster = landmarkFromCameraPerPart[receive_candidate_cluster_id].size();
      if (num_cams_in_receive_cluster <= cams_from_lm[candidate_lm_id].size()) {
          continue;
      };

      bool fail = false;
      // test 1 are ALL cams observed by lm present in receive cluster. Those cams are rare i suppose. 
      for (int camIdx : cams_from_lm[candidate_lm_id]) {
        if (landmarkFromCameraPerPart[receive_candidate_cluster_id].find(camIdx) ==
            landmarkFromCameraPerPart[receive_candidate_cluster_id].end()) {
              fail = true;
              break; // not a candidate.
          }
      }
      if (fail) {continue;}

      // test 2: removing the lm from this cluster does not lead to degenerate situations.
      for (const auto& [camIdx, lmIdSet] : landmarkFromCameraPerPart[move_candidate_cluster_id]) {
        if (lmIdSet.size() < minLmObsPerCamInPart + 1 && lmIdSet.find(candidate_lm_id) != lmIdSet.end()) {
          fail = true;
          break; // not a candidate.
        }
      }
      if(fail) {continue;}

      // std::cout << consecutive_failures << "/" << consecutive_samples_fail 
      //           << " Moving lm " << candidate_lm_id << " from "<< move_candidate_cluster_id << " to " << receive_candidate_cluster_id 
      //           << " #res: " << cams_from_lm[candidate_lm_id].size() << " of " << res_per_cluster[move_candidate_cluster_id] 
      //           << " -> " << res_per_cluster[receive_candidate_cluster_id] << "\n";

      // per form move 
      ApplyMove(candidate_lm_id, move_candidate_cluster_id, receive_candidate_cluster_id, 
                cams_from_lm, landmarkFromCameraPerPart, res_per_cluster);
      lm_to_part[candidate_lm_id] = receive_candidate_cluster_id;
      consecutive_failures = 0;
      last_receive_cluster = receive_candidate_cluster_id;
      moves++;

    // std::cout << "Expanding " << receive_candidate_cluster_id << "\n";
    // for (int partId = 0; partId < res_per_cluster.size(); ++partId) {
    //   std::cout << "Part " << partId << " with " << res_per_cluster[partId] << " residuals\n";
    // }

  } // while trying to move landmarks.

#endif

  std::cout << "After " << moves << " moves\n";
  for(int partId=0;partId < res_per_cluster.size(); ++partId) {
    std::cout << "Part " << partId << " with " << res_per_cluster[partId] << " residuals\n";
  }

  /////////////// Finish output /////////////
  // map from cam/lm id to resid:
  std::map<int, int> camIdTimesLmIdToResId; 
  for(int res_id = 0; res_id < num_res; ++res_id) {
    const int lm_id = landmark_indices_in[res_id];
    const int cam_id = camera_indices_in[res_id];
    camIdTimesLmIdToResId[cam_id * num_lands + lm_id] = res_id;
  }

  for (int partId = 0; partId < landmarkFromCameraPerPart.size(); ++partId) {
    for (const auto& [camIdx, lmIdSet] : landmarkFromCameraPerPart[partId]) {
      for (const int lmIdx : lmIdSet) {
        res_to_cluster_by_landmark[camIdTimesLmIdToResId[camIdx * num_lands + lmIdx]] = partId;
      }
    }
  }
  //////////////////////
  
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
// Leads to 1 big cluster in some cases trafalgar problem-257 and problem-253 dubrovnik
// Order cost uses mean of 'cost' here.
//
// Where do I prefer to have fewer cameras in part? its in here. as is sum over cameras with few lms. this prefers cams to have > maxLmPerCam = 20 lms. then the cam does not count.
//#define __testThis__
double GetCost(const std::map<int, std::set<int>> &landmarkFromCameraOfPart, 
               int maxLmPerCam, double temperature, int res_in_cluster, int total_res, int kClusters) {
  double cost = 0;
  temperature = 15; // 20
  // 10 Cam observations started/finished: 1 : 403/135, 2 : 715/70, 3 : 1110/46, 4 : 1112/37, 5 : 1045/29,  left 317
  // 15 Cam observations started/finished: 1 : 312/107, 2 : 1017/74, 3 : 1411/45, 4 : 1478/39, 5 : 1271/25,  left 290
  // 20 Cam observations started/finished: 1 : 248/118, 2 : 1331/88, 3 : 1611/46, 4 : 1425/42, 5 : 1297/39,  left 333 
  //    Cam observations started/finished: 1 : 352/150, 2 : 1033/92, 3 : 1426/70, 4 : 1361/55, 5 : 1293/24,  left 391
  // maxLmPerCam = 12; // 10 ?

  // Keep only worst 5/10/? to define cost
  // auto cmp = [](double left, double right) {
  //   return left > right; // '<' :largest top,  '>' : smallest top. top one is exchanged.
  // };
  // std::priority_queue<double, std::vector<double>, decltype(cmp)> pq(cmp);

  // Wasserstein distance to 0 for all 1 on > 10 lms in cam.
  // How does this become a probl? transportation is from to.
  // cost is 

  for (const auto& [cam, landmarksFromCam] : landmarkFromCameraOfPart) {
    // TODO: EVAL THE CHANGE.
#ifdef __testThis__
    temperature = 20; // make fit here. lower: more weight on not having isolated lms in cam.
    const int numLandmarks = std::min(static_cast<int>(landmarksFromCam.size()), maxLmPerCam);
    // if (numLandmarks > maxLmPerCam) {continue;} // excluding this -> num cams does matter as well.
    if (numLandmarks == 0) {continue;}  // no cost.
    // orig:
    // cost += std::exp(-(numLandmarks-1) / static_cast<double>(maxLmPerCam-1) * temperature); // gain is exp(-t) ->exp(-2t) etc. 
    // new:
    cost += std::pow(static_cast<double>(maxLmPerCam-1) / static_cast<double>(numLandmarks-0.5) * temperature, 1.5); // gain is exp(-t) ->exp(-2t) etc.     
#else

    //const int numLandmarks = landmarksFromCam.size();
    // if (numLandmarks > maxLmPerCam) {continue;} // ? with: prefer cluster to overlap also for cams seeb by many lms.

    // minor effetcwith below or not. without prefers many lms seen by cam: imbalance vs only < maxLmPerCam observed are considered
    // numLandmarks = std::min(static_cast<int>(landmarksFromCam.size()), maxLmPerCam); // 3068, 30 cl. VERY sensitive to this.
    // if (numLandmarks == 0) {continue;}  // no cost.

    const int numLandmarks = std::min(static_cast<int>(landmarksFromCam.size()), maxLmPerCam);
    //if (numLandmarks > maxLmPerCam) {continue;} // Why not? todo: never happens? also above is wierdly needed for 3068 @ 30 cl. to work fix..
    if (numLandmarks == 0) {continue;}  // no cost.

    cost += std::exp(-numLandmarks / static_cast<double>(maxLmPerCam) * temperature); // should 0 be a cost? not correct to skip 0.. hmm. does not do anything.
#endif

    // cost += std::exp(static_cast<double>(maxLmPerCam-numLandmarks) * temperature); // '-' -> ? same

    // const double v = std::exp(-numLandmarks / static_cast<double>(maxLmPerCam) * temperature);
    // constexpr int maxK = 25;
    // if (pq.size() < maxK) {
    //   pq.push(v);
    // } else if (pq.top() < v) {
    //   pq.pop();
    //   pq.push(v);
    // }
  }

  // cost / entries
  // Cam observations started/finished: 1 : 2419/120, 2 : 2653/77, 3 : 2156/52, 4 : 1901/32, 5 : 1702/23,  left 304
  // Cam observations started/finished: 1 : 11/2, 2 : 5/2, 3 : 3/2, 4 : 3/6, 5 : 0/11, 6 : 19/22, 7 : 44/47, 8 : 72/68, 9 : 88/89, 10 : 84/83, 11 : 89/91, 12 : 112/108, 13 : 125/125, 14 : 187/182,  left 838
  // cost:
  // Cam observations started/finished: 1 : 101/2, 2 : 89/6, 3 : 55/3, 4 : 48/14, 5 : 31/13, 6 : 683/27, 7 : 475/36, 8 : 552/57, 9 : 664/84, 10 : 659/80, 11 : 639/134, 12 : 663/164, 13 : 656/184, 14 : 680/233,  left 1037
  // Cam observations started/finished: 1 : 2/9, 2 : 6/10, 3 : 3/5, 4 : 14/4, 5 : 13/3,  left 31

  // effective:
  // cost = 0;
  // for(int i = 0; i < pq.size(); ++i) {
  //   cost += pq.top();
  //   pq.pop();
  // }

  double costKlDivEquality = 0;
  const int target_residual = total_res / kClusters;
  int overflow_res = res_in_cluster - target_residual;

  // too large reduce costs, s.t. we pick small to merge
  if (false && overflow_res > target_residual) { // has no positive effect
    costKlDivEquality = -cost / 2;// / static_cast<double>(kClusters);
  }

  // does not work, we should try to always merge with 1 small / pick the small one.
  // 
  if (false || overflow_res > 0) {
    double p = static_cast<double>(res_in_cluster) / static_cast<double>(total_res);
    double costKlDivEquality1 = - std::log(p * static_cast<double>(kClusters)) / static_cast<double>(kClusters); // quite strong yet impacts degeneracy
    double costKlDivEquality2 = - p * std::log(p * kClusters); // for some reason alway leads to some balance but degenerate or unbalance & not deg
    // similar to res_per_cluster / total_res - 1 / kCluster as probability.
    // distributions: 1 / kCluster kl div in either direction. p * log(p/q)
    // 1st divergence
    // 10 clusters: 1/10 * log( 10 * total_res / res_in_cl) = 1/10 * log( 10 * total_res) - log( 1 / res)
    // 1st part is constant = - 1/10 * log( res )
    // 2nd divergence is 
    // res_in_cl / total_res * log (10 * res_in_cl / total_res) 
    // costKlDivEquality = 1e-2 * costKlDivEquality1 + 1e-1 * costKlDivEquality2;
    // costKlDivEquality = 1e-3 * costKlDivEquality2;
    costKlDivEquality = 1e-3 * costKlDivEquality1; // does something, but should merge small clusters -> desired 10 -> out 8 or so ..
  }

#ifdef __testThis__ // might not make ANY difference
  // test this long term. done. worse

  double targetRes = static_cast<double>(total_res) / static_cast<double>(kClusters);
  // Might need to mult by #cams / kClusters? 
  double mult = 1; // the larger the more even. Can still fail to do very even. I guess before i only had this on but very soft. as + 1/res.
  // I guess for even we would need to grow such that small still exist AT THE END to fill gaps.
  // orig:
  // return cost + 1 * std::exp( mult * std::max(0., (static_cast<double>(res_in_cluster) - targetRes / 2.))  / targetRes / 2. ); // at > T/2 -> penalty. very large at the end.
  // now lin cost new:
  return cost + std::pow(1e0 / static_cast<double>(res_in_cluster), 1.0); // could also return mean cost

  // Maybe other way round so log(sum(exp())) + 1/ res? 1/res: merge 2 small ones + do not like large clusters. 
  const double target = 3. * targetRes / 4.;
  return 1e2 * std::log(cost) + 1e-0 / static_cast<double>(res_in_cluster) + mult * std::max(0., (static_cast<double>(res_in_cluster) - target)) / target; // at > T/2 -> penalty. very large at the end.

  return cost + costKlDivEquality + 1e-0 / static_cast<double>(res_in_cluster); // could also return mean cost
  return cost + costKlDivEquality + 10 * 1e-0 / static_cast<double>(res_in_cluster); // could also return mean cost
#else

  // Hmm aim is to minimize this cost. So, 1/res -> prefers to combine 2 small cl. not 1 small one large.
  // p: indifferent? vs log(q): prefer small to be merged.

  // 1723: never ever any value times div. 
  // double p = static_cast<double>(res_in_cluster) / static_cast<double>(total_res);
  // Cam observations started/finished: 1 : 1124/156, 2 : 1446/91, 3 : 1603/61, 4 : 1507/40, 5 : 1474/28,  left 376
  // return cost - 25. * p;
  // //costKlDivEquality = - p * std::log(p * kClusters); // prefers growing large.
  //costKlDivEquality = -std::log(p * static_cast<double>(kClusters)) / static_cast<double>(kClusters);
  // // 3068: even res but bad -- haeh? run now ..
  // Cam observations started/finished: 1 : 1037/129, 2 : 1494/88, 3 : 1616/64, 4 : 1539/42, 5 : 1393/42,  left 365
  // return cost;// + 0.00001 * std::abs(costKlDivEquality) + 1e-0 / static_cast<double>(res_in_cluster); // 2nd 10. Part 24 with 302951 residuals

  // return 1e-0 / static_cast<double>(res_in_cluster);
  // Cam observations started/finished: 1 : 7985/183, 2 : 6359/87, 3 : 5161/71, 4 : 4594/49, 5 : 3917/32,  left 422
  // Cam observations started/finished: 1 : 183/0, 2 : 87/2, 3 : 71/3, 4 : 49/9, 5 : 32/17, 6 : 783/48, 7 : 516/63, 8 : 668/78, 9 : 718/102, 10 : 726/89, 11 : 728/86, 12 : 710/112, 13 : 700/146, 14 : 691/217,  left 972
  // Cam observations started/finished: 1 : 0/14, 2 : 2/9, 3 : 3/2, 4 : 9/1, 5 : 17/2,  left 28
  // Cam observations started/finished: 1 : 14/0, 2 : 9/2, 3 : 2/3, 4 : 1/7, 5 : 2/14, 6 : 46/48, 7 : 59/63, 8 : 79/81, 9 : 102/101, 10 : 87/88, 11 : 84/85, 12 : 112/114, 13 : 151/146, 14 : 223/215,  left 967
  // Cam observations started/finished: 1 : 0/12, 2 : 2/9, 3 : 3/2, 4 : 7/1, 5 : 14/2,  left 26
  // Cam observations started/finished: 1 : 12/0, 2 : 9/2, 3 : 2/3, 4 : 1/7, 5 : 2/14, 6 : 44/48, 7 : 61/63, 8 : 81/81, 9 : 102/101, 10 : 89/88, 11 : 84/84, 12 : 113/116, 13 : 149/147, 14 : 214/204,  left 958

  // 1/ res -> get rid of small clusters first. e.g. merge small into 1 is better than merge 2 middle sized ones.
  return cost + costKlDivEquality + 1e-0 / static_cast<double>(res_in_cluster); // could also return mean cost
  //return cost + costKlDivEquality + 1e-0 / std::sqrt(static_cast<double>(res_in_cluster));
#endif
}

// Cost Gain per landmark:
// wij = exp(-sij/S), neg. similaity sij measures cams not in common, ie. let C(p): set of cams in part P?, pij = wij/wT, wT constant hmm.
// or sij := 

// prefers landmarks seen by few cameras with few observations in those cameras on average -- so not few observations first.
// IDEA: highest cost first!
double GetOrderCost(const std::map<int, std::set<int>> &landmarkFromCameraOfPart, 
               int maxLmPerCam, double temperature, int res_in_cluster, int total_res, int kClusters) {
  double cost = 0;
  int entries = 0;
  // maybe take average of worst ten? not average of all?

  // auto cmp = [](double left, double right) {
  //   return left < right; // smallest first?
  // };
  // std::priority_queue<double, std::vector<double>, decltype(cmp)> pq(cmp);

  for(const auto& [cam, landmarksFromCam] : landmarkFromCameraOfPart) {
    const int numLandmarks = landmarksFromCam.size();
    if (numLandmarks > maxLmPerCam) {continue;}
    cost += std::exp(-numLandmarks / static_cast<double>(maxLmPerCam) * temperature);
    //cost += std::exp(-(maxLmPerCam-numLandmarks) / static_cast<double>(maxLmPerCam) * temperature);

    //pq.push(std::exp(-numLandmarks / static_cast<double>(maxLmPerCam) * temperature));
    // pq.push(std::exp(-(maxLmPerCam-numLandmarks) / static_cast<double>(maxLmPerCam) * temperature));
    // if (pq.size() > 10) {pq.pop();}
    entries++;
  }
  cost /= static_cast<double>(std::max(1, entries));
  // cost = 0;
  // for(int i = 0; i < pq.size(); ++i) {
  //   cost += pq.top() / static_cast<double>(pq.size());
  //   pq.pop();
  // }

  // 1. Kl-div(p,q) = sum p_i log(p_i / q_i) = - sum p_i log(q_i/p_i).
  // 2. Kl-div(q,p) = sum q_i log(q_i / p_i) = - sum q_i log(p_i/q_i).
  // q_i = 1 / static_cast<double>(kClusters)
  // 2. - log (p * kClusters) / kClusters
  // loves cl size res to equal to 1/kClusters, prefers even larger ones.
  // what do i want to pick small clusters first? prefers large clusters first.
  double p = static_cast<double>(res_in_cluster) / static_cast<double>(total_res);
  double costKlDivEquality = - std::log(p * static_cast<double>(kClusters)) / static_cast<double>(kClusters); // quite strong yet impacts degeneracy
  // for 3086 result is so bad. why not just 

  // mean 
  // TODO: 356 was 1e-3 one component remains. 1e-2: better, still 4 large 6 small cluster.
  // Could also use 1e-3, eval if not recompute with 1e-2, etc.

  // Slow: merges only small ones. why slow then ? no idea.
  // return cost + /// static_cast<double>(std::max(1, entries)) +
  //        (1e-0 / static_cast<double>(res_in_cluster) +
  //         _order_div_mult_ * costKlDivEquality);

//#undef __testThisNew__
#ifdef __testThisNew__
  // Try this version for order
  //std::cout << "Order cost " << cost + 1e-1 * _order_div_mult_ * costKlDivEquality << " = " << cost << " + " << 1e-0 / static_cast<double>(res_in_cluster) << " + " << _order_div_mult_ * costKlDivEquality << " " << res_in_cluster << " < " << total_res<< "\n";
  //return cost + 1e-1 * _order_div_mult_ * costKlDivEquality2;
  //return cost - 1e-0 / static_cast<double>(res_in_cluster);
  // maybe should be distributed as log? same as kl div then.
  // cost -  .1 * std::log(p): Cam observations started/finished: 1 : 1992/80, 2 : 2178/59, 3 : 1849/32, 4 : 1560/19, 5 : 1417/13,  left 203
  // cost - .01 * std::log(p): Cam observations started/finished: 1 : 1513/95, 2 : 1855/56, 3 : 1699/42, 4 : 1562/26, 5 : 1306/29,  left 248 .. maybe its not this?
  // return cost - .1 * std::log(p); // ? cost in [0,1]. at which p factor 100: p +- 0.01 -> prefer 100 times smaller res 
  // 1. become independent of # clusters. R: all res, r: num res -> replace R/r by -r/R here. -p is linear in percent of res. vs 1/r is stronger on small r.
  // p*p -> more smaller cost for larger parts.
  // overindex
  // 100 is fast enough. 10 also still ok. - X*p with idea: indifferent if within factor of X. if factor is larger (|T| > X * |S|) -> S comes before T.
  // cost - 50. * p: Cam observations started/finished: 1 : 1866/105, 2 : 2101/66, 3 : 2042/52, 4 : 1757/31, 5 : 1505/21,  left 275
  // cost - 10. * p: Cam observations started/finished: 1 : 1409/93,  2 : 1834/57, 3 : 2025/39, 4 : 1689/33, 5 : 1507/24,  left 246
  // cost - 25. * p:Cam observations started/finished:  1 : 1745/83,  2 : 2258/51, 3 : 2173/40, 4 : 1809/20, 5 : 1646/19,  left 213
  // maybe 10 is bad for small problems. must be smaller then?
  // 10 is slow. why actually?
  return cost - 25. * p;// + _order_div_mult_ * costKlDivEquality; // from 0 to 1 -- should be more general small vs large ba datasets?
  // or even pure. if this does not work, the order cost is ok, the cost is not.
  // return -p;//costKlDivEquality; // ok sorts both as desired.
#endif

  //    1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10 .. twice an many res: 1/r vs 1/2r and r/R vs 2r/R. 
  // vs 1/R, 2/R, 3/R, 4/R, 5/R, 6/R, 7/R, 8/R, 9/R, 10/R
  // Highest cost first == biggest estimated cost gain possible => low cost if 
  // So picks small guys first as cost is high 1/1 vs 1/ 10000 or so. costKlDivEquality is same idea. cost wants bad configs go first for merge.
  return cost + 1e-0 / static_cast<double>(res_in_cluster) + _order_div_mult_ * costKlDivEquality;
  // return cost / std::sqrt(static_cast<double>(std::max(1, entries))) + 1e-0 / static_cast<double>(res_in_cluster);
}

// return cams in part with fewest landmark observations. Foolows definition of GetCost. cma with 1 lm has high cost, picking cam seeing this can lead to good merge candidate.
std::vector<int> GetLowestKCameras(const std::map<int, std::set<int>> &landmarkFromCameraOfPart, 
                                   int topK, const std::vector<std::vector<int>>& lms_from_cam) {
  std::vector<std::pair<int, int>> lmsInPartOfCam;
  lmsInPartOfCam.reserve(topK);
  // map cost to partId ? update by set cost to inf / update cost = heap.
  auto cmp = [&lmsInPartOfCam](int left, int right) {
    return lmsInPartOfCam[left].second < lmsInPartOfCam[right].second; // largest top
  };
  std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);

  for (const auto& [cam, landmarksFromCam] : landmarkFromCameraOfPart) {
    const int numLandmarks = landmarksFromCam.size(); // 0 cannot happen.

    // std::cout << "Cam  " << cam << " in part with " << numLandmarks << " landmarks\n";
    // TODO; could also be cam with most observations not in part.
    if(lms_from_cam[cam].size() <= numLandmarks) {continue;} // skip fully covered.

    // baseline Cam observations started/finished: 1 : 781/123, 2 : 1063/60, 3 : 1302/55, 4 : 1289/25, 5 : 1244/32,  left 295
    // *5         m observations started/finished: 1 : 472/111, 2 : 718/60, 3 : 1074/56, 4 : 1102/42, 5 : 1078/35,  left 304
    // *2       Cam observations started/finished: 1 :  528/88, 2 : 725/64, 3 : 1033/32, 4 : 1103/32, 5 : 1091/17,  left 233 -- still worse.
    // /2       Cam observations started/finished: 1 : 507/128, 2 : 655/91, 3 : 1051/42, 4 : 1025/41, 5 : 1022/37,  left 339
    // 1:1      Cam observations started/finished: 1 : 543/118, 2 : 784/58, 3 : 1008/43, 4 : 1158/34, 5 : 1076/24,  left 277

    const int cost = lms_from_cam[cam].size() + 3 * numLandmarks; // prefer few lms in part and few lms in total.

    if (pq.size() < topK) { // always push if less than desired
      lmsInPartOfCam.push_back({cam, cost});
      pq.push(lmsInPartOfCam.size() - 1);
      // std::cout << "Insert " << cam << " with " << numLandmarks
      //           << " lms in part\n";
    } else {
      const int id = pq.top();
      // TODO: this is just crazy:
      // if(  id > numLandmarks) { // new is better (== random pick?)
      if (lmsInPartOfCam[id].second > cost) { // if new is smaller than largest of pq, replace.
        pq.pop();
        // std::cout << "Replacing " << lmsInPartOfCam[id].first << "/"
        //           << lmsInPartOfCam[id].second << " with " << cam << " with "
        //           << numLandmarks << " lms in part selected\n";
        lmsInPartOfCam[id] = {cam, cost}; // overwrite
        pq.push(id);                              // re enter in new place
      }
    }
  }

  // std::cout <<"----\n";
  // best cams are in pq.
  std::vector<int> ids;
  while (!pq.empty()) {
    //std::cout << "Top k Cam  " << lmsInPartOfCam[pq.top()].first << " with " << lmsInPartOfCam[pq.top()].second << " lms in part selected\n";
    ids.push_back(lmsInPartOfCam[pq.top()].first);
    pq.pop();
  }
  return ids;
}

// 2nd part and CostGain
std::pair<int, double> FindbestMatchForPart(int partId, //const std::vector<std::set<int>>& cams_from_lm, 
  const std::vector<std::vector<int>>& lms_from_cam, 
  const std::vector<std::map<int, std::set<int>>>& landmarkFromCameraPerPart, const std::vector<int>& res_per_cluster,
  const std::vector<int>& lmToPart, const std::vector<double>& costOfPart, 
  int maxLmPerCam, double temperature, int num_res, int kClusters, bool verbose = false) {
  // idea: find k cams with fewest landmarks in part.
  constexpr int topK = 3;//4; // 1 -> 3: 24 -> 38s, before used 1. maybe 3 is better 3068. not clear what defines better for 3068.
  constexpr int topL = 30;//20;  // change to 20 does 4s -> 5s .. + 25% likely trade off with below.
 
  // 1/ 30   Cam observations started/finished: 1 : 1445/101, 2 : 1595/89, 3 : 1684/55, 4 : 1521/48, 5 : 1456/31,  left 324
  // 3 / 30  Cam observations started/finished: 1 :  776/83,  2 : 1057/56, 3 : 1212/38, 4 : 1249/29, 5 : 1164/19,  left 225
  // 4 20 Cam observations started/finished: 1 : 827/110, 2 : 1121/50, 3 : 1338/49, 4 : 1292/32, 5 : 1188/23,  left 264

  // 4/25      Cam observations started/finished: 1 : 985/128, 2 : 1214/80, 3 : 1544/55, 4 : 1486/47, 5 : 1383/32,  left 342
  // 6/15/     Cam observations started/finished: 1 : 1028/96, 2 : 1367/82, 3 : 1540/68, 4 : 1522/31, 5 : 1358/27,  left 304
  // 4/20/1000 Cam observations started/finished: 1 : 953/100, 2 : 1315/70, 3 : 1495/48, 4 : 1472/43, 5 : 1333/24,  left 285
  // 2/30/1500 Cam observations started/finished: 1 : 1113/109, 2 : 1385/67, 3 : 1484/40, 4 : 1451/32, 5 : 1362/18,  left 266
  // 15/5/1000 Cam observations started/finished: 1 : 1237/95, 2 : 1472/62, 3 : 1483/43, 4 : 1476/46, 5 : 1365/25,  left 271
  // new Get Cost .. worse again Cam observations started/finished: 1 : 1017/154, 2 : 1412/88, 3 : 1503/50, 4 : 1450/47, 5 : 1416/21,  left 360

  // could have 2 q's, could have n random + selected.
  // otherPartCost[id].second > cost: 4/25: 

  // 4 22 Cam observations started/finished: 1 : 1/5, 2 : 144/5, 3 : 438/5, 4 : 531/0, 5 : 500/0,  left 15
  // 3 22
  // Cam observations started/finished: 1 : 4/10, 2 : 176/5, 3 : 467/1, 4 : 548/1, 5 : 458/0,  left 17
  // 7 7 
  // Cam observations started/finished: 1 : 24/15, 2 : 399/1, 3 : 586/3, 4 : 670/1, 5 : 529/1,  left 21
   // 3 17
  //Cam observations started/finished: 1 : 23/13, 2 : 264/4, 3 : 548/1, 4 : 577/1, 5 : 484/4,  left 23
  // 4 17
  // Cam observations started/finished: 1 : 5/19, 2 : 197/2, 3 : 523/9, 4 : 594/1, 5 : 458/2,  left 33

  // 4 10
  // Cam observations started/finished: 1 : 17/22, 2 : 390/6, 3 : 581/5, 4 : 608/0, 5 : 517/2,  left 35
  // 2 15
  //  Cam observations started/finished: 1 : 27/17, 2 : 401/4, 3 : 623/10, 4 : 662/2, 5 : 595/2,  left 35
  // 5 6
  // Cam observations started/finished: 1 : 64/16, 2 : 472/6, 3 : 556/8, 4 : 684/2, 5 : 573/0,  left 32
  // 3 10
  // Cam observations started/finished: 1 : 40/10, 2 : 369/3, 3 : 549/6, 4 : 561/2, 5 : 496/1,  left 22
  // 10, 3
  // Cam observations started/finished: 1 : 110/12, 2 : 586/6, 3 : 560/2, 4 : 652/1, 5 : 526/0,  left 21
  // 10 4
  // Cam observations started/finished: 1 : 69/15, 2 : 497/5, 3 : 512/8, 4 : 614/2, 5 : 537/0,  left 30
  // 10, 5, better but does not work still.
  // Cam observations started/finished: 1 : 42/11, 2 : 425/4, 3 : 511/12, 4 : 588/4, 5 : 478/2,  left 33
  // 12 3
  //Cam observations started/finished: 1 : 92/13, 2 : 539/2, 3 : 560/4, 4 : 666/4, 5 : 517/0,  left 23
  // 8 4
  // Cam observations started/finished: 1 : 97/16, 2 : 579/4, 3 : 578/4, 4 : 741/0, 5 : 562/2,  left 26

  //Cam observations started/finished: 1 : 1553/129, 2 : 2023/57, 3 : 2022/55, 4 : 1792/33, 5 : 1518/25,  left 299
  //Cam observations started/finished: 1 : 1707/114, 2 : 2207/77, 3 : 2111/54, 4 : 1890/41, 5 : 1590/22,  left 308 # this is better yet?
  // test: 3000 replacing 1500. run on probem 52. small value: slow since we do not find small cluster to merge.
  constexpr int topM = 1000;//1000;// 1500 -> 4500: 24s ->27s // we do not use all but 700 random landmarks for a cam -- there can be 30k.
  //Cam observations started/finished: 1 : 37/5, 2 : 22/2, 3 : 44/1, 4 : 252/0, 5 : 288/2,  left 10  - 1500, largest
  //Cam observations started/finished: 1 : 96/2, 2 : 45/0, 3 : 132/0, 4 : 242/0, 5 : 294/0,  left 2 -- random 30
  
  // Cam in part with fewest landamrk observations in part.
  const std::map<int, std::set<int>> &landmarkFromCameraOfPart = landmarkFromCameraPerPart[partId];
  std::vector<int> camsToTryInPart = GetLowestKCameras(landmarkFromCameraOfPart, topK, lms_from_cam);
  // for each cam find a second different part also observing it, pick one with low costs

  if (verbose) {
    std::cout << "Lowest k ";
    for(const auto& cam: camsToTryInPart) {
      std::cout << " cam " << cam <<  " ";
    }
    std::cout << "\n";
  }

  // std::mt19937 mt{ static_cast<std::mt19937::result_type>(
  // std::chrono::steady_clock::now().time_since_epoch().count() ) }; // maybe repeatable -> seed =0.
  std::mt19937 mt{ static_cast<std::mt19937::result_type>( _rngseed_ ) };

  std::set<int> partToTryMerge;
  // Again keep top k possibilities in Q ?
// #pragma omp parallel for num_threads(4) // slower
//    for (int i = 0; i < topK; ++i) { // this cam has these 
//      const int camToTry = camsToTryInPart[i];
  for (int camToTry : camsToTryInPart) { // this cam has these 
    //std::cout << "At camToTry " << camToTry << std::endl;
    // const std::set<int>& landmarksInPart = landmarkFromCameraOfPart.at(camToTry); // the landmarks observed by the cam, can be just 1
    //std::cout << "Ok\n";
    // for each lm here, find a potential partner.
    const std::vector<int>& otherLandmarksObservedByCam = lms_from_cam[camToTry]; // do part from cam
    const int numObservations = otherLandmarksObservedByCam.size();

    std::vector<std::pair<int, double>> otherPartCost;
    otherPartCost.reserve(topL);
    // map cost to partId ? update by set cost to inf / update cost = heap.
    auto cmp = [&otherPartCost](int left, int right) {
      // This is not the best way. even 30 random was better than 1500 selected ones !! 
      // Maybe use random 30, maybe ? 
      return otherPartCost[left].second > otherPartCost[right].second; // smallest on top, note that 1500 os WORSE than 3000 for 3068. -- should pick based on cost gain. high cost + overlap ~ highest gain possible.
      // for speed reasons we likely want small go first. Indeed '<' here is TOTALLY SUPER SLOW.
      //return otherPartCost[left].second < otherPartCost[right].second; // TODO should it not be largest first here, since large ones are the trouble makers, since we use .. wait what? what should it be actually extra cost for this?
    };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);
   
    int num = 0;
    // TODO: Paralel, smarter implementation.
    // Preselect topL parts to try merge with.
    #define _simpler_
    #ifdef _simpler_
    std::set<int> checkedParts = {partId}; // TODO could keep outside! and / or even fill pq from all at once.
    std::vector<int> landmarkIds(numObservations, 0);
    std::iota(landmarkIds.begin(), landmarkIds.end(), 0);
    if (topM < numObservations) {
      std::shuffle(landmarkIds.begin(), landmarkIds.end(), mt);
    }
    for (int lmIdd = 0; lmIdd < std::min(topM, numObservations); ++lmIdd ) {
      const int lmId = otherLandmarksObservedByCam[landmarkIds[lmIdd]];
      
      const int otherPartId = lmToPart[lmId]; ///////////////////////////////////////////////////////////////////////////// FindRootInVtxToPartMap(const std::vector<int>& vtxsToPart, int start)
      const auto& [it, inserted] = checkedParts.insert(otherPartId);
      if (!inserted) {continue;}
      
      if (res_per_cluster[otherPartId] <= 0) {continue;}
      
      const double cost = costOfPart[otherPartId]; // The order cost, not the cost!
      
      //std::cout << "lm " << lmId << " " << cost << "\n";
      if (pq.size() < topL) { // always push if less than desired
        otherPartCost.push_back({otherPartId, cost});
        //std::cout << "Pushing "  << otherPartId << " with " << cost << " for merge\n";
        pq.push(otherPartCost.size() - 1);
        continue;
      }
      
      const int id = pq.top(); // if here is largest first, then check should lead to replace act if smaller. (queue holds smallest parts) and vice versa. CORRECT BUT SLOW -- why?
      if (otherPartCost[id].second < cost) { // sign different than above is correct, also need many trials topM = 1500 is best -- any other number fails at 1068, also rng.
        pq.pop();
        //std::cout << "Considering "  << otherPartId << " with " << cost << " for merge replacing " << otherPartCost[id].first << " c: " << otherPartCost[id].second << "\n";
        otherPartCost[id] = {otherPartId, cost}; // overwrite
        pq.push(id); // re enter in a new place
      }
    }
    //std::cout << "===\n";
    #else
    auto landmarkIt = otherLandmarksObservedByCam.begin();
    std::set<int> checkedParts = {partId}; // TODO could keep outside! and / or even fill pq from all at once.
    std::uniform_int_distribution dist{ 0, numObservations-1 };
    while (num < std::min(topM, numObservations) ) { // 5k on average, uber-BOTTLENECK. have a map cam to part?
      if (numObservations < topM) {
        if (num>0) {std::advance(landmarkIt, 1);}
      }
      else { // pick at random if more than topM exist.
        landmarkIt = otherLandmarksObservedByCam.begin();
        std::advance(landmarkIt, dist(mt));
      }
      ++num;
      const int lmId = *landmarkIt;
      const int otherPartId = lmToPart[lmId]; ///////////////////////////////////////////////////////////////////////////// FindRootInVtxToPartMap(const std::vector<int>& vtxsToPart, int start)

      const auto& [it, inserted] = checkedParts.insert(otherPartId);
      if (!inserted) {continue;}
      if (res_per_cluster[otherPartId] <= 0) {continue;} 

      const double cost = costOfPart[otherPartId];
      //std::cout << "lm " << lmId << " " << cost << "\n";
      if (pq.size() < topL) { // always push if less than desired
        otherPartCost.push_back({otherPartId, cost});
        //std::cout << "Pushing "  << otherPartId << " with " << cost << " for merge\n";
        pq.push(otherPartCost.size() - 1);
        continue;
      }

      const int id = pq.top();
      if (otherPartCost[id].second < cost) { // new is better (== random pick?)
        pq.pop();
        //std::cout << "Considering "  << otherPartId << " with " << cost << " for merge replacing " << otherPartCost[id].first << " c: " << otherPartCost[id].second << "\n";
        otherPartCost[id] = {otherPartId, cost}; // overwrite
        pq.push(id); // re enter in a new place
      }
    }
#endif

//#pragma omp critical
    for (const auto [id, cost] : otherPartCost) {
      partToTryMerge.insert(id);
      //std::cout << "Using part "  << id << " with " << cost << " for possible merge top cost: " << otherPartCost[pq.top()].second << "\n";
      pq.pop();
    }
  } // over different camsToTryInPart
  
  // compute merge gain
  int bestPartToMerge = -1;
  double bestCostGain = -1000;
#ifdef _select_by_even_cost_
  // cost GAIN be regular cost
  const double partCost = GetCost(landmarkFromCameraPerPart[partId], maxLmPerCam, temperature, res_per_cluster[partId], num_res, kClusters);
#else
  const double partCost = costOfPart[partId];
#endif

// TODO: parallel? below set best part not parallel.
#pragma omp parallel num_threads(10) // not for
  for (int otherPartId : partToTryMerge) {
#ifdef _select_by_even_cost_
    // cost GAIN be regular cost
    const double oldCost = partCost + GetCost(landmarkFromCameraPerPart[otherPartId], maxLmPerCam, temperature, res_per_cluster[otherPartId], num_res, kClusters);
#else
    const double oldCost = partCost + costOfPart[otherPartId];
#endif

    // TODO maybe faster by using a vector with set sizes directly. not copy the map.
    std::map<int, std::set<int>> landmarkFromCameraOfOtherPart = landmarkFromCameraPerPart[otherPartId];
    // merge landmarkFromCameraOfOtherPart and landmarkFromCameraOfPart and compute new cost.
    for(const auto&[cam, lms] : landmarkFromCameraOfPart) {
      auto it = landmarkFromCameraOfOtherPart.find(cam);
      if (it != landmarkFromCameraOfOtherPart.end() && landmarkFromCameraOfOtherPart.at(cam).size() > maxLmPerCam) {continue;} // already maximized for cost compute.
      if (lms.size() > maxLmPerCam) { // too large would not count in cost anyway.
        if (it == landmarkFromCameraOfOtherPart.end()) {continue;} // just skip.
        landmarkFromCameraOfOtherPart.erase(cam); // erase is faster then insert many?
      } else {
        landmarkFromCameraOfOtherPart[cam].insert(lms.begin(), lms.end()); // slow
      }
    }

#ifdef _select_by_even_cost_
    // cost GAIN be regular cost
    const double newCost = GetCost(landmarkFromCameraOfOtherPart, maxLmPerCam, temperature, res_per_cluster[partId] + res_per_cluster[otherPartId], num_res, kClusters);
#else
    const double newCost = GetCost(landmarkFromCameraOfOtherPart, maxLmPerCam, temperature, res_per_cluster[partId] + res_per_cluster[otherPartId], num_res, kClusters);
#endif
    const double costGain = oldCost - newCost;
    // std::cout << "Own part "  << partId << " with cost " << costOfPart[partId] << " and merge  Part " << otherPartId 
    //           << " with cost " << costOfPart[otherPartId] << " for possible merge with new cost " << newCost << " = " << costGain << " vs " << bestCostGain << "\n";

#pragma omp critical
    if(costGain > bestCostGain) {
      bestCostGain = costGain;
      bestPartToMerge = otherPartId;
      //  std::cout << "Own part "  << partId << " with cost " << costOfPart[partId] << " and merge  Part " << otherPartId 
      //            << " with cost " << costOfPart[otherPartId] << " for possible merge with new cost " << newCost << " = " << costGain << " vs " << bestCostGain << "\n";
    }
  }

  if (verbose)
    std::cout << "Found Best part to " << partId << " cost " << costOfPart[partId] << " " << bestPartToMerge 
              << " cost " << costOfPart[bestPartToMerge] << " and cost gain " << bestCostGain << "\n";
   return {bestPartToMerge, bestCostGain};
  }

// update a lot. costs, invalidate one part, cma to lms, .. 
double MergeParts(int partId, int otherPartId,
      std::vector<std::map<int, std::set<int>>>& landmarkFromCameraPerPart,
      std::vector<int>& res_per_cluster,
      std::vector<int>& lmToPart, 
      //std::vector<double>& costOfPart,
      int maxLmPerCam, double temperature, int num_res, int kClusters) {
    std::map<int, std::set<int>>& landmarkFromCameraOfPart = landmarkFromCameraPerPart[partId];
    std::map<int, std::set<int>>& landmarkFromCameraOfOtherPart = landmarkFromCameraPerPart[otherPartId];
    // merge landmarkFromCameraOfOtherPart and landmarkFromCameraOfPart and compute new cost.
    for(const auto&[cam, lms] : landmarkFromCameraOfOtherPart) {
      landmarkFromCameraOfPart[cam].insert(lms.begin(), lms.end());
      for(const int lm : lms) { // way too many != lms in part ~ could be handled by here only lmToPart[otherPartId] = partId and using root to .. 10%
        lmToPart[lm] = partId;
      }
    }
    
    //costOfPart[otherPartId] = -100; // invalidats q invariant: do not do like this, 3 lines below.
    landmarkFromCameraOfOtherPart.clear();
    res_per_cluster[partId] += res_per_cluster[otherPartId];
    res_per_cluster[otherPartId] = 0; // invalid
  #ifdef _select_by_even_cost_
    // order by order cost, prefer small parts.
    return GetOrderCost(landmarkFromCameraPerPart[partId], maxLmPerCam, temperature, res_per_cluster[partId], num_res, kClusters); // this could be a different cost
  #else
    return GetCost(landmarkFromCameraPerPart[partId], maxLmPerCam, temperature, res_per_cluster[partId], num_res, kClusters); // this could be a different cost
  #endif
    // 'disable' one part. update the other. 
    // i popped one already, push this back in. Set cost to -inf for other part [we can drop this one, when popping it just looking up its cost]
}

bool do_entries_match(const std::vector<int> &a, const std::vector<int> &b) {
  if (a.size() != a.size()) {
    return false;
  }
  for (int i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

// Find out if hashing works to find lms that see the same cameras.
std::vector<std::set<int>> find_identical_lms(const std::vector<std::vector<int>>& cams_from_lm, int num_cams, int num_lms) {
  // cams_from_lm[lm_id] is a vector of cameras observing the lm.
  // idea was to find mean vector. Identify this as a 01 vector: v[c] = 1: c observers lm, v[c] = 0 else 
  //const int num_lms = cams_from_lm.size();
  std::vector<double> mean(num_cams, 0);
  int mean_cams_seen = 0;
  const double val = 1./ static_cast<double> (num_lms);
  for (const auto& cam_vec : cams_from_lm) {
    for(int c: cam_vec) {
      mean[c] += val;
      mean_cams_seen++;
    }
  }
  const int cams_seen = mean_cams_seen;
  mean_cams_seen *= val;
  // sample n normals with mean
  std::mt19937 mt{ static_cast<std::mt19937::result_type>( _rngseed_ ) };
  //std::uniform_int_distribution dist{ 0, num_cams-1 };
  std::vector<int> n_ids(num_cams);
  std::iota(n_ids.begin(), n_ids.end(), 0);
  // sample normal, not clear if 1s and 0's or +- 1's
  // 64 bits
  int num_hashes = 2;
  std::vector<std::vector<unsigned long>> lm_to_hashes(num_hashes);
  for (int hash_id = 0; hash_id < num_hashes; ++hash_id) {
    std::vector<unsigned long> &lm_to_hash = lm_to_hashes[hash_id];
    lm_to_hash.resize(num_lms, 0);
    const int n_normals = 31;
    for (unsigned int n = 0; n < n_normals; ++n) {
      const unsigned long ul(1ul << n);
      // std::cout << " ul " << ul << std::endl;

      // const int num_normal_entries = mean_cams_seen;
      std::shuffle(n_ids.begin(), n_ids.end(), mt);
      const int num_normal_entries = num_cams / 2;
      std::vector<int> normal_ids(num_normal_entries);
      std::vector<int> normal(num_cams, -1);
      for (int i = 0; i < num_normal_entries; ++i) {
        const int n_id = n_ids[i];
        normal[n_id] = 1;
        normal_ids[i] = n_id;
      }
      // normal ^T 1 = 0. mean_cams_seen 1s, num_cams - mean_cams_seen: - mean_cams_seen / (num_cams - mean_cams_seen)
      // compute n times mean: all -1 -> n^t mean = -1 * sum mean + 2 * sum n_entry mean[nentry]
      // double mean_times_normal = -cams_seen * mean_cams_seen / static_cast<double>(num_cams - mean_cams_seen);
      // for (int n_id : normal_ids) {
      //   mean_times_normal += mean[n_id] * (1 + mean_cams_seen / static_cast<double>(num_cams - mean_cams_seen));
      // }
      // now 1 full pass.
      double mean_times_normal = 0;
      for (int n_id = 0; n_id < num_cams; ++n_id) {
        mean_times_normal += mean[n_id] * normal[n_id];
      }

      // compute dot product for all, compute part of hash of lm.
      for (int l = 0; l < num_lms; ++l) {
        double dot = -mean_times_normal;
        for (int cam_id : cams_from_lm[l]) {
          dot += normal[cam_id]; // normal + or -1, lm[cam_id] =1, 0 else.
        }
        // now per lm compute hash / side of normal it falls onto. problem both sparse vectors.
        if (dot > 0) {
          lm_to_hash[l] += ul;
        } // 2^n
      }
    }
  }

  // hash codes.
  // 1. write entries per bin. num_cams / 32. Not so much / 64: better
  // std::vector<int> entries(1 << n_normals);
  std::map<std::pair<int, int>, std::set<int>> hash_to_lmSet;
  for (int l = 0; l < lm_to_hashes[0].size(); ++l) {
    hash_to_lmSet[{lm_to_hashes[0][l], lm_to_hashes[1][l]}].insert(l);
  }

  // std::cout << "Mean_cams_seen " << mean_cams_seen << " Num hashes " << hash_to_lmSet.size() << "\n";
  // for (const auto& [hash, lm_set] : hash_to_lmSet) {
  //   if (lm_set.size() <2) {continue;}
  //   std::cout << "(" << hash.first << " " << hash.second << ") " << lm_set.size() << "\n";
  //   for( const int lm_id : lm_set) {
  //     std::cout << lm_id << " : ";
  //     for(const int c : cams_from_lm[lm_id]) {
  //       std::cout << c << " ";
  //     } 
  //     std::cout << std::endl;
  //   }
  // }

  // return .. a vec of sets of lmids -> 1
  int numDuplicateLms = 0;
  std::vector<std::set<int>> duplicate_lm_ids;
  for (const auto& [hash, lm_set] : hash_to_lmSet) {
    if (lm_set.size() <2) {continue;}
    // Compare lm set exhaustively and put inot set (identical ones)
    std::set<int> dupe_set = lm_set;
    for(const int lm : lm_set) {
      std::set<int> identical_set;
      identical_set.insert(lm);
      const std::vector<int>& cam_vec_lm = cams_from_lm[lm];
      dupe_set.erase(lm);
      for(const int lm2 : dupe_set) {
        if (do_entries_match(cam_vec_lm, cams_from_lm[lm2])){
          // else match to each other.
          identical_set.insert(lm2);
        }
      }
      if (identical_set.size() > 1) {
        duplicate_lm_ids.push_back(identical_set);
        numDuplicateLms += identical_set.size() - 1;
        for(const int lm3 : identical_set)
          dupe_set.erase(lm3);
      }
    }
  }
  std::cout << "Number of landmarks with identical cameras " << numDuplicateLms << " / " << num_lms << std::endl;
  return duplicate_lm_ids;
}


// Slected part to be merged by metric GetOrderCost.
// pop next part.
// for that part find best part to merge with.
// minimize what. GetCost(part) = sum _c in part exp-|lms(c)| / maxLmPerCam * temperature, if lms(c) < maxLmPerCam, 0 else.
// High cost for few lms seen. 0 for none! not sure this works for greedy. no idea to approx as well.
void cluster_cameras_degeneracy(
    int kClusters,
    const std::vector<int>& camera_indices_in,  // per res -> cam involved
    const std::vector<int>& landmark_indices_in,// per res -> landmark involved
    std::vector<int>& res_to_cluster_by_landmark_out) {

    const bool verbose = false;
    const int num_res = landmark_indices_in.size();
    const int num_cams = std::set<int>( camera_indices_in.begin(), camera_indices_in.end() ).size();
    const int num_lands = std::set<int>( landmark_indices_in.begin(), landmark_indices_in.end() ).size();

    if (camera_indices_in.size() != num_res || verbose) {
      std::cout << "Start #res " << num_res<< " " << kClusters << " #lnds" << num_lands << "  #cams " << num_cams << "\n";
      std::cout << " camera_indices_in " <<"\n";
      std::cout << " camera_indices_in " << camera_indices_in.size() << "\n";
      std::cout << " landmark_indices_in  " << landmark_indices_in.size() << "\n";
    }

    static int maxLmPerCam = 10;
    // Guess this is hard to trade off. Will lead to 1 large many small cluster
    static double temperature = 20; // todo: lower -> more weight on few cams. maybe slower?
    static int nLowestPartsToTry = 1;

    // 2. map lm to cluster index. start each lm is a cluster.
    // RemapToRootInVtxToPartMap
    std::vector<int> lmToPart(num_lands);
    std::iota(lmToPart.begin(), lmToPart.end(), 0);

    // 1. maps from lm to cameras and from cameras to landmarks
    // find cameras in part with few landmarks. So per part: map cam id -> landmarks seen and in part.
    std::vector<std::vector<int>> cams_from_lm(num_lands);
    std::vector<std::vector<int>> lms_from_cam(num_cams);
    std::vector<std::map<int, std::set<int>>> landmarkFromCameraPerPart(num_lands);
    std::vector<int> res_per_cluster(num_lands, 0);
    for (int res_id = 0; res_id < landmark_indices_in.size(); ++res_id) {
        const int lm_id = landmark_indices_in[res_id];
        const int cam_id = camera_indices_in[res_id];
        cams_from_lm[lm_id].push_back(cam_id);
        lms_from_cam[cam_id].push_back(lm_id);
        int partId = lmToPart[lm_id]; // identity at start
        res_per_cluster[partId]++;
        landmarkFromCameraPerPart[partId][cam_id].insert(lm_id);
    }
    if (verbose)
      std::cout << " landmarkFromCameraPerPart  " << landmarkFromCameraPerPart.size() << " done\n";

    // 3. cluster to cam involved and counts use landmarkFromCameraPerPart
    // 4. compute cost per part. init.
    int num_parts = num_lands;
    std::vector<double> costOfPart(num_parts, 0);

    // TODO: Merge parts that are full subsets of another part.
    // maybe hash all. take hash of one. find similar one. small to large would suffice. also can be ordered simply.
    // n hashes. P(same ) hash subsets for all but smallest parts (those have no subsets?).
    // problem 2 subsets are quadratic already in total size.
    // per view this is not so bad. go over all parts that share a view (for smaller one). I.e. 2 elements -> 2 views.
#ifdef __clusteridentical_lms_early__
    // Would be faster to first remove identical lms. then build the queue. 
    // Clamp landmarks with identical camera set into one part.
    // Likely better to make code believe only single lm is in part (searches voer cams .. ?) Does it do anything?
    std::vector<std::set<int>> list_of_identical_lms = find_identical_lms(cams_from_lm, num_cams, num_lands);
    for (const auto& set_of_idential_lms : list_of_identical_lms) {
      const int keptPartId = *(set_of_idential_lms.begin());
      for (const int deletedPartId : set_of_idential_lms) {
        if (keptPartId == deletedPartId) {continue;}
        // Todo : ineffective to compute cost here.
        const double newCost = MergeParts(keptPartId, deletedPartId,
              landmarkFromCameraPerPart,
              res_per_cluster,
              lmToPart, 
              //costOfPart, // if set to -1 blocks other parts to go up in q. must invalidate extra.
              maxLmPerCam, 
              temperature,
              num_res, 
              kClusters);
        //costOfPart[keptPartId] = newCost; // should suffice. no pop needed as merging set num res to 0 of 
        num_parts--;
      }
    }
#endif

    // map cost to partId ? update by set cost to inf / update cost = heap.
    auto cmp = [&costOfPart](int left, int right) {
      return costOfPart[left] < costOfPart[right]; // highest cost 1st best merge candidates
      };
      std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);
  
    for (int partId = 0;partId < num_lands; partId++ ) {
      if (res_per_cluster[partId] <= 0) {continue;} // invalid / merged
      // This could be a different cost.
#ifdef _select_by_even_cost_
      // order by order cost. prefer small parts.
      const double cost = GetOrderCost(landmarkFromCameraPerPart[partId], maxLmPerCam, temperature, res_per_cluster[partId], num_res, kClusters); // also prefer small parts ?!
#else
      const double cost = GetCost(landmarkFromCameraPerPart[partId], maxLmPerCam, temperature, res_per_cluster[partId], num_res, kClusters); // also prefer small parts ?!
#endif
      //std::cout << "Insert PartId  " << partId << " cost " << cost << " \n";
      costOfPart[partId] = cost;
      pq.push(partId);
    }
    //std::cout << "Parts " << num_parts << " #res " << num_res << " #lms " << num_lands << " #cams " << num_cams << " \n";

    // 5. merge parts until kClusters are left.
    while (!pq.empty() && num_parts > kClusters) {
      const int partId = pq.top();
      // std::cout << "PartId  " << partId << " cost " << costOfPart[partId] << " #parts " 
      //           << num_parts << " #pq" << pq.size() << " " << res_per_cluster[partId] << " \n";

      if (res_per_cluster[partId] <= 0) {pq.pop();continue;} // invalid / merged

      // 0. select part to try for a merge
      // 1. select parts to merge     
      // 2. merge & update costs
      std::map<double, std::pair<int,int>> partsToTry; // nLowestPartsToTry is 1.
      for (int partToTryId = 0; partToTryId < std::min(nLowestPartsToTry, num_parts-1); partToTryId++) {
        const int partId = pq.top();
        pq.pop();
        if (verbose)
          std::cout << "#P:" << num_parts << " Try PartId  " << partId << " cost " << costOfPart[partId] 
                  << " pq-size " << pq.size() << " #res:" << res_per_cluster[partId] << " \n";
          // We try the same partId very often repeatedly. can i just skip it if it failed once?
          // it failed if 
        std::pair<int, double> partAndGain = // 2nd part and CostGain
          FindbestMatchForPart(partId, lms_from_cam, landmarkFromCameraPerPart, res_per_cluster, 
                               lmToPart, costOfPart, maxLmPerCam,temperature, num_res, kClusters, verbose);
        if (verbose)
          std::cout << " partAndGain  " << partAndGain.first << " gain " << partAndGain.second << " \n";
        partsToTry[partAndGain.second] = {partId, partAndGain.first};
      }

      // per part to try find best match(es).
      if(!partsToTry.empty() && (partsToTry.begin()->second.second >=0)) {
      std::pair<int,int> partsToMerge = partsToTry.begin()->second;
      const double gain = partsToTry.begin()->first;
      if (verbose)
        std::cout << " Merge  " << partsToMerge.first << " " << partsToMerge.second 
                  << " gain " << gain << " parts " << num_parts-1 << " #res "
                  << res_per_cluster[partsToMerge.first] << " + "<<  res_per_cluster[partsToMerge.second] << " = "
                  << res_per_cluster[partsToMerge.first] + res_per_cluster[partsToMerge.second] << " \n";

      const int numCamsBefore1 = landmarkFromCameraPerPart[partId].size();
      const int numCamsBefore2 = landmarkFromCameraPerPart[partsToMerge.second ].size();

      const double newCost = MergeParts(partsToMerge.first, partsToMerge.second,
            landmarkFromCameraPerPart,
            res_per_cluster,
            lmToPart, 
            //costOfPart, // if set to -1 blocks other parts to go up in q. must invalidate extra.
            maxLmPerCam, 
            temperature,
            num_res, 
            kClusters);

      // res_per_cluster[partsToMerge.second] == 0 after merge.

      if (verbose) {
        const int numCamsAfter = landmarkFromCameraPerPart[partId].size();
        std::cout << " parts " << num_parts-1 << " Merge num Cams " << numCamsBefore1 << " & " << numCamsBefore2 << " = " << numCamsAfter << " newCost " << newCost << " \n";
      }

      assert(partsToMerge.first == partId);

      costOfPart[partId] = newCost;
      pq.push(partId);
      num_parts--;
      }
      else {
        if (verbose)
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
