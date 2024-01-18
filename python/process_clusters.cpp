// returns covered_landmark_indices_c: those in addition! -- which is weird -- to those covered by main residual split
// the latter are point_indices_already_covered and also returned
#include "process_clusters.h"

#include <iostream>
#include <vector>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <numeric>

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