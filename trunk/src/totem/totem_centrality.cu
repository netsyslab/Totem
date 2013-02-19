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
void unweighted_sssp_succs_kernel(graph_t graph, int64_t phase, vid_t* sigma,
                                  int32_t* dist, vid_t* succ,
                                  uint32_t* succ_count, vid_t* stack,
                                  uint32_t* stack_count, bool* finished) {
  const vid_t thread_id = THREAD_GLOBAL_INDEX;

  // Each thread corresponds to an edge from a vertex in the phase stack to a
  // destination vertex. We allocate threads in a depth-first order; ie:
  // Let 'v' be the first vertex in the stack, with 'm_0' neighbors. Then
  // threads 0 through (m_0 - 1) correspond to each each from v to its
  // neighbor.  Likewise, threads m_0 through (m_1 - 1) correspond to the
  // edges from second vertex in the stack to all its neighbors
  vid_t thread_count = 0;
  vid_t v = 0;
  for (uint32_t v_index = 0; v_index < stack_count[phase]; v_index++) {
    v = stack[(graph.vertex_count * phase) + v_index];
    thread_count += (graph.vertices[v + 1] - graph.vertices[v]);
    if (thread_id < thread_count) break;
  }

  // If the thread id is within range, process the vertex associated with it
  if(thread_id < thread_count) {
    // w is the vertex id of the neighbor vertex of v for this thread
    vid_t w = graph.edges[graph.vertices[v+1] - (thread_count - thread_id)];

    // Perform the lock-free Brandes SSSP algorithm for the given vertex
    int32_t dw = atomicCAS(&dist[w], (int32_t)-1, phase + 1);
    if (dw == -1) {
      *finished = false;
      vid_t p = atomicAdd(&stack_count[phase + 1], 1);
      stack[graph.vertex_count * (phase + 1) + p] = w;
      dw = phase + 1;
    }
    if (dw == phase + 1) {
      vid_t p = (vid_t)atomicAdd(&succ_count[v], 1);
      succ[graph.vertices[v] + p] = w;
      atomicAdd(&sigma[w], sigma[v]);
    }
  }
}

/**
 * Unweighted BFS single source shortest path kernel with predecessor map
 */
__global__
void unweighted_sssp_preds_kernel(graph_t graph, vid_t* r_edges, int32_t dist,
                                  int32_t* dists, vid_t* sigma, bool* preds,
                                  bool* finished) {
  const vid_t thread_id = THREAD_GLOBAL_INDEX;
  // Each thread corresponds to an edge in the graph
  if (thread_id < graph.edge_count) {
    vid_t u = r_edges[thread_id];
    vid_t v = graph.edges[thread_id];

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
void unweighted_back_prop_kernel(graph_t graph, vid_t* r_edges, int32_t* dists,
                                 vid_t* sigma, bool* preds, int32_t dist,
                                 score_t* delta) {
  const vid_t thread_id = THREAD_GLOBAL_INDEX;
  // For each edge (u, v), if u is a predecessor of v, add to its dependence
  if (thread_id < graph.edge_count) {
    vid_t u = graph.edges[thread_id];
    vid_t v = r_edges[thread_id];
    if (dists[u] == (dist - 1) && preds[thread_id]) {
      atomicAdd(&delta[v], (1.0 * sigma[v] / sigma[u]) * (1.0 + delta[u]));
    }
  }
}

/**
 * Construct reverse edges so that we can easily find the source vertex for each
 * edge in the graph.
 */
error_t centrality_construct_r_edges(const graph_t* graph, vid_t** r_edges_p) {
  if (graph == NULL || r_edges_p == NULL) return FAILURE;

  // For every edge in the graph, save its source vertex
  vid_t* r_edges = (vid_t*)mem_alloc(graph->edge_count * sizeof(vid_t));
  vid_t v = 0;
  for (eid_t e = 0; e < graph->edge_count; e++) {
    while (v <= graph->vertex_count
           && !((e >= graph->vertices[v]) && (e < graph->vertices[v+1]))) {
      v++;
    }
    r_edges[e] = v;
  }

  // return the newly allocated reverse edge map
  *r_edges_p = r_edges;
  return SUCCESS;
}
