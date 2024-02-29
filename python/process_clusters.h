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

// maybe add cam_to_cluster and landmark_to_cluster to mark the variables to update
extern "C"
void cluster_covis(
    int kClusters,
    int random_pre_number_merges,
    int maxVolPerPart,
    const std::vector<int>& camera_indices_in, 
    const std::vector<int>& landmark_indices_in,
    std::vector<int>& res_to_cluster, std::vector<int>& res_to_cluster_sizes, std::vector<int>& old_vtxsToPart);

extern "C" void
process_clusters(
    int num_lands,
    int num_res,
    int kClusters, 
    const std::vector<int>& point_indices_in_cluster_flat, //those inclusters
    const std::vector<int>& point_indices_in_cluster_sizes,
    const std::vector<int>& point_indices_, // all points
    const std::vector<int>& res_indices_in_cluster_flat,
    const std::vector<int>& res_indices_in_cluster_sizes,
    std::vector<int>& res_toadd_out, std::vector<int>& res_toadd_sizes,
    std::vector<int>& point_indices_already_covered_out, std::vector<int>& point_indices_already_covered_sizes,
    std::vector<int>& covered_landmark_indices_c_out, std::vector<int>& covered_landmark_indices_c_sizes, 
    std::vector<int>& res_to_cluster_by_landmark_out);

extern "C" void
recluster_cameras(
    int kClusters,
    const std::vector<int>& camera_indices_in,
    const std::vector<int>& landmark_indices_in,
    std::vector<int>& res_to_cluster_by_landmark);

extern "C" void
cluster_cameras_degeneracy(
    int kClusters,
    const std::vector<int>& camera_indices_in,  // per res -> cam involved
    const std::vector<int>& landmark_indices_in,// per res -> landmark involved
    std::vector<int>& res_to_cluster_by_landmark_out);

// simplify removing inputs
// extern "C"
// void cluster_covis_full(
//     int num_res,
//     int kClusters,
//     int num_lands,
//     int num_cams,
//     const std::vector<int>& camera_indices_in,
//     const std::vector<int>& landmark_indices_in,
//     std::vector<int>& res_to_cluster, std::vector<int>& res_to_cluster_sizes,
//     // out
//     std::vector<int>& res_toadd_out, std::vector<int>& res_toadd_sizes,
//     std::vector<int>& point_indices_already_covered_out, std::vector<int>& point_indices_already_covered_sizes,
//     std::vector<int>& covered_landmark_indices_c_out, std::vector<int>& covered_landmark_indices_c_sizes,
//     std::vector<int>& num_res_per_c_out);

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
        //std::cout << vec->size() << " v[0]" << (*vec)[0] << std::endl; 
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

// g++ -c -O3 -fPIC process_clusters.cpp -o process_clusters.o
// g++ -shared -o libprocess_clusters.so process_clusters.o -Wl,--export-dynamic
// clang++-15 -O3 -c -fPIC -std=c++17 process_clusters.cpp -o process_clusters.o;clang++-15 -shared -o libprocess_clusters.so process_clusters.o -Wl,--export-dynamic

// clang++-15 -std=c++17 -g process_clusters.cpp -fprofile-instr-generate -fcoverage-mapping -o process_clusters

// clang++-15 -g -fprofile-instr-generate -std=c++17 process_clusters.cpp -o process_clusters.o ??
// clang++-15  -fcoverage-mapping -shared -o libprocess_clusters.so process_clusters.o -Wl,--export-dynamic ??
// -fprofile-instr-generate

// maybe 
// clang++-15 -c -std=c++17 -fPIC -fprofile-instr-generate -fcoverage-mapping process_clusters.cpp -o process_clusters.o 
// clang++-15 -c -std=c++17 -fPIC -fprofile-generate -fcoverage-mapping process_clusters.cpp -o process_clusters.o 
// clang++-15 -fcoverage-mapping -shared -o libprocess_clusters.so process_clusters.o -Wl,--export-dynamic
// export LLVM_PROFILE_FILE=./llvm_new.prof
// llvm-profdata-15 merge -output=merge_mew.out -instr llvm_new.prof
// llvm-profdata-15 show -all-functions -counts -ic-targets  merge_mew.out >  profdata_show.log
// llvm-cov-15 show profile_coverage  -instr-profile=merge_mew.out

// no: clang++-15 -c -std=c++17 -o process_clusters.o process_clusters.cpp -fPIC -fprofile-instr-generate -femit-coverage-data -femit-coverage-notes