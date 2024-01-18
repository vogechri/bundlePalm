#ifndef PROCESS_CLUSTERS_H
#define PROCESS_CLUSTERS_H

#include <vector>
#include <iostream>
#include <tuple>

// check that fct is exported with name 
// nm -D libprocess_clusters.so | grep clusters
// output -> 000000000002c68b T process_clusters
// extern "C" std::tuple<std::vector<std::vector<int>>, 
// std::vector<std::vector<int>>, 
// std::vector<std::vector<int>>, 
// std::vector<int>>

extern "C" void
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
    std::vector<int>& num_res_per_c_out);

extern "C" void process_clusters_test(
    int num_lands,
    int num_res,
    int kClusters,
    const std::vector<int>& point_indices_in_cluster_flat){
    std::cout << "nl " << num_lands << " nr " << num_res << " kc " << kClusters << std::endl;
    std::cout << "point_indices_in_cluster_flat " << point_indices_in_cluster_flat.size() << "\n";
    return;
};

extern "C" {
    std::vector<int>* new_vector(){
        return new std::vector<int>;
    }
    std::vector<int>* new_vector_of_size(int size){
        return new std::vector<int>(size);
    }
    std::vector<int>* new_vector_by_copy(int* v, int size){
        auto* vec = new std::vector<int>(size);
        std::copy(v, v + size, vec->data());
        return vec;
    }
    void delete_vector(std::vector<int>* v){
        std::cout << "destructor called in C++ for " << v << std::endl;
        delete v;
    }
    int vector_size(std::vector<int>* v){
        return v->size();
    }
    int vector_get(std::vector<int>* v, int i){
        return v->at(i);
    }
    void vector_set(std::vector<int>* v, int i, int j){
        v->at(i) = j;
        return;
    }
    void vector_push_back(std::vector<int>* v, int i){
        v->push_back(i);
    }
}
#endif // PROCESS_CLUSTERS_H

// g++ -c -fPIC process_clusters.cpp -o process_clusters.o
// g++ -shared -o libprocess_clusters.so process_clusters.o -Wl,--export-dynamic