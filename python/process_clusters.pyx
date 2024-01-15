# distutils: language = c++
# distutils: sources = process_clusters.cpp

cdef extern from "process_clusters.cpp":
    cdef cppclass VectorInt "std::vector<int>":
        VectorInt()
        void push_back(int)
        int size()
    cdef cppclass VectorVectorInt "std::vector<std::vector<int>>":
        VectorVectorInt()
        void push_back(VectorInt)
        int size()
    cdef cppclass TupleVectorVectorInt "std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<int>>":
        TupleVectorVectorInt()
    TupleVectorVectorInt process_clusters(int, int, int, VectorVectorInt&, VectorInt&, VectorVectorInt&)

def process_clusters_py(num_lands, num_res, kClusters, point_indices_in_cluster_, point_indices_, res_indices_in_cluster_):
    cdef VectorInt point_indices = VectorInt()
    for point in point_indices_:
        point_indices.push_back(point)

    cdef VectorVectorInt point_indices_in_cluster = VectorVectorInt()
    for cluster in point_indices_in_cluster_:
        cdef VectorInt cluster_vec = VectorInt()
        for point in cluster:
            cluster_vec.push_back(point)
        point_indices_in_cluster.push_back(cluster_vec)

    cdef VectorVectorInt res_indices_in_cluster = VectorVectorInt()
    for cluster in res_indices_in_cluster_:
        cdef VectorInt cluster_vec = VectorInt()
        for res in cluster:
            cluster_vec.push_back(res)
        res_indices_in_cluster.push_back(cluster_vec)

    cdef TupleVectorVectorInt result = process_clusters(num_lands, num_res, kClusters, point_indices_in_cluster, point_indices, res_indices_in_cluster)

    # Convert the result back to Python lists and return
    return [[list(cluster) for cluster in result[0]], [list(cluster) for cluster in result[1]], [list(cluster) for cluster in result[2]], list(result[3])]