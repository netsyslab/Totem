/**
 *  Defines common Centrality functions and algorithms.
 *
 *  Created on: 2012-05-07
 *  Author: Greg Redekop
 */

// totem includes
#include "totem_centrality.h"

/**
 * Unweighted BFS single source shortest path kernel using successor stack
 */
__global__
void unweighted_sssp_succs_kernel(graph_t graph, int64_t phase, id_t* sigma,
                                  int32_t* dist, id_t* succ,
                                  uint32_t* succ_count, id_t* stack,
                                  uint32_t* stack_count, bool* finished) {
  const id_t thread_id = THREAD_GLOBAL_INDEX;

  // Each thread corresponds to an edge from a vertex in the phase stack to a
  // destination vertex. We allocate threads in a depth-first order; ie:
  // Let 'v' be the first vertex in the stack, with 'm_0' neighbors. Then
  // threads 0 through (m_0 - 1) correspond to each each from v to its
  // neighbor.  Likewise, threads m_0 through (m_1 - 1) correspond to the
  // edges from second vertex in the stack to all its neighbors
  id_t thread_count = 0;
  id_t v = 0;
  for (uint32_t v_index = 0; v_index < stack_count[phase]; v_index++) {
    v = stack[(graph.vertex_count * phase) + v_index];
    thread_count += (graph.vertices[v + 1] - graph.vertices[v]);
    if (thread_id < thread_count) break;
  }

  // If the thread id is within range, process the vertex associated with it
  if(thread_id < thread_count) {
    // w is the vertex id of the neighbor vertex of v for this thread
    id_t w = graph.edges[graph.vertices[v+1] - (thread_count - thread_id)];

    // Perform the lock-free Brandes SSSP algorithm for the given vertex
    int32_t dw = atomicCAS(&dist[w], (int32_t)-1, phase + 1);
    if (dw == -1) {
      *finished = false;
      id_t p = atomicAdd(&stack_count[phase + 1], 1);
      stack[graph.vertex_count * (phase + 1) + p] = w;
      dw = phase + 1;
    }
    if (dw == phase + 1) {
      id_t p = (id_t)atomicAdd(&succ_count[v], 1);
      succ[graph.vertices[v] + p] = w;
      atomicAdd(&sigma[w], sigma[v]);
    }
  }
}

/**
 * Unweighted BFS single source shortest path kernel with predecessor map
 */
__global__
void unweighted_sssp_preds_kernel(graph_t graph, id_t* r_edges, int32_t dist,
                                  int32_t* dists, id_t* sigma, bool* preds,
                                  bool* finished) {
  const id_t thread_id = THREAD_GLOBAL_INDEX;
  // Each thread corresponds to an edge in the graph
  if (thread_id < graph.edge_count) {
    id_t u = r_edges[thread_id];
    id_t v = graph.edges[thread_id];

    if (dists[u] == dist) {
      if (dists[v] == -1) {
        *finished = false;
        dists[v] = dist + 1;
      }
      if (dists[v] == dist + 1) {
        preds[thread_id] = true;
        atomicAdd(&sigma[v], sigma[u]);
      }
    }
  }
}

/**
 * Unweighted centrality back propagation kernel for predecessor map
 * implementation. Calculates dependence (delta) for each vertex in the graph.
 */
__global__
void unweighted_back_prop_kernel(graph_t graph, id_t* r_edges, int32_t* dists,
                                 id_t* sigma, bool* preds, int32_t dist,
                                 weight_t* delta) {
  const id_t thread_id = THREAD_GLOBAL_INDEX;
  // For each edge (u, v), if u is a predecessor of v, add to its dependence
  if (thread_id < graph.edge_count) {
    id_t u = graph.edges[thread_id];
    id_t v = r_edges[thread_id];
    if (dists[u] == (dist - 1) && preds[thread_id]) {
      atomicAdd(&delta[v], (1.0 * sigma[v] / sigma[u]) * (1.0 + delta[u]));
    }
  }
}
