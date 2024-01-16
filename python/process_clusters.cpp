// returns covered_landmark_indices_c: those in addition! -- which is weird -- to those covered by main residual split
// the latter are point_indices_already_covered and also returned
#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<int>> process_clusters(int num_lands, int num_res, int kClusters, const std::vector<std::vector<int>>& point_indices_in_cluster_, const std::vector<int>& point_indices_, const std::vector<std::vector<int>>& res_indices_in_cluster_) {
    std::vector<int> landmark_occurrences(num_lands, 0);
    for (int ci = 0; ci < kClusters; ci++) {
        std::unordered_set<int> unique_points_in_cluster(point_indices_in_cluster_[ci].begin(), point_indices_in_cluster_[ci].end());
        for (int point : unique_points_in_cluster) {
            landmark_occurrences[point]++;
        }
    }
    std::cout << "uncovered are " << std::count(landmark_occurrences.begin(), landmark_occurrences.end(), 0) << " landmarks, present in single " << std::count(landmark_occurrences.begin(), landmark_occurrences.end(), 1) << ", present in multiple " << std::count_if(landmark_occurrences.begin(), landmark_occurrences.end(), [](int val) { return val > 1; }) << std::endl;

    std::vector<int> point_indices_to_complete;
    for (int i = 0; i < num_lands; i++) {
        if (landmark_occurrences[i] > 1) {
            point_indices_to_complete.push_back(i);
        }
    }
    std::cout << "point_indices_to_complete " << point_indices_to_complete.size() << std::endl;

    std::vector<std::vector<int>> point_indices_already_covered(kClusters);
    int sum_points_covered = 0;
    for (int ci = 0; ci < kClusters; ci++) {
        std::unordered_set<int> unique_points_in_cluster(point_indices_in_cluster_[ci].begin(), point_indices_in_cluster_[ci].end());
        std::vector<int> intersection;
        std::set_intersection(unique_points_in_cluster.begin(), unique_points_in_cluster.end(), point_indices_.begin(), point_indices_.end(), std::back_inserter(intersection));
        point_indices_already_covered[ci] = intersection;
        std::cout << ci << " point_indices_already_covered " << point_indices_already_covered[ci].size() << std::endl;
        sum_points_covered += point_indices_already_covered[ci].size();
    }
    std::cout << "Together covered points " << point_indices_to_complete.size() + sum_points_covered << "  sum_points_covered: " << sum_points_covered << std::endl;

    std::unordered_map<int, std::unordered_set<int>> point_to_res_id;
    for (int i = 0; i < num_res; i++) {
        if (landmark_occurrences[point_indices_[i]] > 1) {
            point_to_res_id[point_indices_[i]].insert(i);
        }
    }

    std::vector<int> res_per_lm(num_res);
    for (int i = 0; i < num_res; i++) {
        res_per_lm[i] = std::count(point_indices_.begin(), point_indices_.end(), point_indices_[i]);
    }

    std::vector<std::vector<int>> missing_res_per_lm_c(kClusters, std::vector<int>(num_res));
    std::vector<int> num_res_per_c(kClusters);

    for (int ci = 0; ci < kClusters; ci++) {
        std::unordered_map<int, int> counts;
        for (int point : point_indices_in_cluster_[ci]) {
            counts[point]++;
        }
        for (int i = 0; i < num_res; i++) {
            if (counts.find(point_indices_[i]) != counts.end()) {
                missing_res_per_lm_c[ci][i] = res_per_lm[i] - counts[point_indices_[i]];
            } else {
                missing_res_per_lm_c[ci][i] = res_per_lm[i];
            }
        }
        num_res_per_c[ci] = std::accumulate(counts.begin(), counts.end(), 0, [](int sum, const std::pair<int, int>& p) { return sum + p.second; });
    }

    std::unordered_map<int, std::vector<int>> res_of_lm;
    for (int i = 0; i < num_res; i++) {
        res_of_lm[point_indices_[i]].push_back(i);
    }

    std::vector<std::unordered_map<int, std::vector<int>>> res_of_lm_notin_c(kClusters);
    std::vector<std::unordered_map<int, std::vector<int>>> res_of_lm_in_c(kClusters);

    for (int ci = 0; ci < kClusters; ci++) {
        std::vector<bool> res_notin_c(num_res, true);
        for (int res_index : res_indices_in_cluster_[ci]) {
            res_notin_c[res_index] = false;
        }
        std::vector<int> point_indices_temp = point_indices_;
        for (int i = 0; i < num_res; i++) {
            if (res_notin_c[i]) {
                point_indices_temp[i] = -1;
            }
        }
        std::unordered_map<int, std::vector<int>> tmp;
        for (int i = 0; i < num_res; i++) {
            if (point_indices_temp[i] != -1) {
                tmp[point_indices_temp[i]].push_back(i);
            }
        }
        res_of_lm_in_c[ci] = tmp;

        std::fill(res_notin_c.begin(), res_notin_c.end(), false);
        for (int res_index : res_indices_in_cluster_[ci]) {
            res_notin_c[res_index] = true;
        }
        point_indices_temp = point_indices_;
        for (int i = 0; i < num_res; i++) {
            if (res_notin_c[i]) {
                point_indices_temp[i] = -1;
            }
        }
        tmp.clear();
        for (int i = 0; i < num_res; i++) {
            if (point_indices_temp[i] != -1) {
                tmp[point_indices_temp[i]].push_back(i);
            }
        }
        res_of_lm_notin_c[ci] = tmp;
    }

    std::vector<std::vector<int>> res_toadd_to_c(kClusters);
    std::vector<std::vector<int>> covered_landmark_indices_c(kClusters);

    for (int i : point_indices_to_complete) {
        std::vector<int> cost(kClusters, 0);
        for (int ci = 0; ci < kClusters; ci++) {
            cost[ci] += res_of_lm_notin_c[ci][i].size() * num_res;
            cost[ci] += num_res_per_c[ci];
        }
        int ci = std::min_element(cost.begin(), cost.end()) - cost.begin();
        res_toadd_to_c[ci].insert(res_toadd_to_c[ci].end(), res_of_lm_notin_c[ci][i].begin(), res_of_lm_notin_c[ci][i].end());
        num_res_per_c[ci] += res_of_lm_notin_c[ci][i].size();
        covered_landmark_indices_c[ci].push_back(i);
    }

    return std::make_tuple(res_toadd_to_c, point_indices_already_covered, covered_landmark_indices_c, num_res_per_c);
}