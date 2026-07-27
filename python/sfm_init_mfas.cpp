#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

extern "C" int sfm_init_mfas_broken_weights(
    const std::int64_t* input_edges,
    const double* input_weights,
    std::int64_t edge_count,
    double* broken_weights) {
  if (input_edges == nullptr || input_weights == nullptr ||
      broken_weights == nullptr || edge_count < 0) {
    return 1;
  }

  std::vector<std::pair<int, int>> edges;
  std::vector<double> weights;
  edges.reserve(edge_count);
  weights.reserve(edge_count);
  int node_count = 0;
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    std::int64_t source = input_edges[2 * edge];
    std::int64_t destination = input_edges[2 * edge + 1];
    if (source < 0 || destination < 0 ||
        source > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
        destination > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      return 2;
    }
    double weight = input_weights[edge];
    if (weight < 0.0) {
      std::swap(source, destination);
      weight = -weight;
    }
    edges.emplace_back(static_cast<int>(source), static_cast<int>(destination));
    weights.push_back(weight);
    node_count = std::max(node_count, static_cast<int>(std::max(source, destination)) + 1);
  }

  std::vector<double> weighted_in_degree(node_count, 0.0);
  std::vector<double> weighted_out_degree(node_count, 0.0);
  std::vector<bool> unchosen(node_count, true);
  std::vector<std::vector<std::pair<int, double>>> incoming(node_count);
  std::vector<std::vector<std::pair<int, double>>> outgoing(node_count);
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const auto [source, destination] = edges[edge];
    const double weight = weights[edge];
    weighted_in_degree[destination] += weight;
    weighted_out_degree[source] += weight;
    incoming[destination].emplace_back(source, weight);
    outgoing[source].emplace_back(destination, weight);
  }

  std::vector<int> order;
  order.reserve(node_count);
  while (static_cast<int>(order.size()) < node_count) {
    int choice = -1;
    double maximum_score = 0.0;
    for (int node = 0; node < node_count; ++node) {
      if (!unchosen[node]) {
        continue;
      }
      if (weighted_in_degree[node] < 1e-8) {
        choice = node;
        break;
      }
      const double score =
          (weighted_out_degree[node] + 1.0) / (weighted_in_degree[node] + 1.0);
      if (score > maximum_score) {
        maximum_score = score;
        choice = node;
      }
    }
    if (choice < 0) {
      return 3;
    }
    for (const auto& [neighbor, weight] : incoming[choice]) {
      weighted_out_degree[neighbor] -= weight;
    }
    for (const auto& [neighbor, weight] : outgoing[choice]) {
      weighted_in_degree[neighbor] -= weight;
    }
    order.push_back(choice);
    unchosen[choice] = false;
  }

  std::vector<int> inverse_order(node_count, 0);
  for (int position = 0; position < node_count; ++position) {
    inverse_order[order[position]] = position;
  }
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const auto [source, destination] = edges[edge];
    broken_weights[edge] =
        inverse_order[destination] < inverse_order[source] ? weights[edge] : 0.0;
  }
  return 0;
}
