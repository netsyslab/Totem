/**
 *  Defines Closeness Centrality functions for both CPU and GPU.
 *
 *  Created on: 2012-06-06
 *  Author: Greg Redekop
 */

// totem includes
#include "totem_centrality.h"
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU).
*/
PRIVATE
error_t check_special_cases(const graph_t* graph, bool* finished,
                            weight_t** centrality_score) {
  if (graph == NULL || graph->vertex_count == 0 || centrality_score == NULL) {
    return FAILURE;
  }

  if (graph->edge_count == 0) {
    *centrality_score =
      (weight_t*)mem_alloc(graph->vertex_count * sizeof(weight_t));
    memset(*centrality_score, (weight_t)0.0, graph->vertex_count
           * sizeof(weight_t));
    return SUCCESS;
  }

  *finished = false;
  return SUCCESS;
}

/**
 * Implements the parallel Brandes closeness centrality algorithm using
 * predecessor maps as described in "Fast Network Centrality Analysis Using
 * GPUs" [Shi11]
 */
error_t closeness_unweighted_cpu(const graph_t* graph,
                                 weight_t** centrality_score) {
  // Sanity check on input
  bool finished = true;
  error_t rc = check_special_cases(graph, &finished, centrality_score);
  if (finished) return rc;

  // Allocate space for the results
  weight_t* closeness_centrality =
    (weight_t*)mem_alloc(graph->vertex_count * sizeof(weight_t));
  // Construct the reverse edges list (graph->edges is a list of destination
  // vertices, r_edges is a list of source vertices, indexed by edge id)
  id_t* r_edges = NULL;
  centrality_construct_r_edges(graph, &r_edges);

  // Allocate and initialize state for the problem
  int32_t* dists = (int32_t*)mem_alloc(graph->vertex_count * sizeof(int32_t));
  memset(closeness_centrality, 0.0, graph->vertex_count * sizeof(weight_t));

  for (id_t source = 0; source < graph->vertex_count; source++) {
    // Initialize state for SSSP
    memset(dists, -1, graph->vertex_count * sizeof(int32_t));
    dists[source] = 0;

    // SSSP and path counting
    int32_t dist = 0;
    bool finished = false;
    while (!finished) {
      finished = true;
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif // _OPENMP
      for (id_t e = 0; e < graph->edge_count; e++) {
        id_t e_source = r_edges[e];
        id_t e_dest = graph->edges[e];

        if ((dists[e_source] == dist) && (dists[e_dest] == -1)) {
          finished = false;
          dists[e_dest] = dist + 1;
        }
      }
      dist++;
    }
    // Count connected and calculate closeness centrality
    uint32_t connected = graph->vertex_count;
    uint64_t sum = 0;
    for (id_t v = 0; v < graph->vertex_count; v++) {
      if (dists[v] == -1) {
        connected--;
      } else {
        sum += dists[v];
      }
    }

    // If this node is isolated or has a 0 length path to all other nodes, set
    // its centrality explicitly to 0
    if ((connected == 0) || (sum == 0)) {
      closeness_centrality[source] = (weight_t)0.0;
    } else {
      closeness_centrality[source] =
        (weight_t)(1.0 * (connected - 1) * (connected - 1))
          / ((graph->vertex_count - 1) * sum);
    }
  } // for

  // Cleanup phase
  mem_free(r_edges);
  mem_free(dists);

  // Return the centrality
  *centrality_score = closeness_centrality;
  return SUCCESS;
}
