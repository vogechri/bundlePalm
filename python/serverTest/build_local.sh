#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_dir=$(cd -- "${script_dir}/.." && pwd)

if ! command -v cmake >/dev/null 2>&1; then
    echo "Native build dependencies are missing." >&2
    echo "On Ubuntu/WSL, install them with:" >&2
    echo "  sudo apt update" >&2
    echo "  sudo apt install -y build-essential cmake protobuf-compiler libprotobuf-dev libzmq3-dev cppzmq-dev libceres-dev libeigen3-dev" >&2
    echo "Then rerun: ./build_local.sh" >&2
    exit 1
fi

g++ -std=c++17 -O3 -fPIC -fopenmp -shared \
    "${python_dir}/process_clusters.cpp" \
    "${python_dir}/camera_hypergraph_partitioning.cpp" \
    "${python_dir}/landmark_partitioning.cpp" \
    -o "${python_dir}/libprocess_clusters.so"

cmake -S "${script_dir}" -B "${script_dir}/build" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${script_dir}/build" --parallel

echo "Built clustering library: ${python_dir}/libprocess_clusters.so"
echo "Built optimization server: ${script_dir}/build/zeromq_cpp_server_ex"