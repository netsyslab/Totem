/**
 *  Defines common Centrality functions and algorithms.
 *
 *  Created on: 2012-05-07
 *  Author: Greg Redekop
 */

// totem includes
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

/**
 * Implements the parallel Brandes betweenness centrality algorithm, as
 * described in "A Faster Parallel Algorithm and Efficient Multithreaded
 * Implementations for Evaluating Betweenness Centrality on Massive Datasets"
 * [Madduri09]
 */
error_t betweenness_unweighted_cpu(const graph_t* graph,
                                   weight_t** centrality_score) {
  // Sanity check on input
  if (graph == NULL || graph->vertex_count == 0 || centrality_score == NULL) {
    return FAILURE;
  }

  // Allocate memory for the shortest paths problem
  weight_t* betweenness_centrality = (weight_t*)mem_alloc(graph->vertex_count
                                                          * sizeof(weight_t));
  id_t* sigma = (id_t*)mem_alloc(graph->vertex_count * sizeof(id_t));
  int32_t* dist = (int32_t*)mem_alloc(graph->vertex_count * sizeof(int32_t));
  id_t* succ = (id_t*)mem_alloc(graph->edge_count * sizeof(id_t));
  uint32_t* succ_count = (uint32_t*)mem_alloc(graph->vertex_count
                                              * sizeof(uint32_t));
  id_t* stack = (id_t*)mem_alloc(graph->vertex_count * graph->vertex_count
                                 * sizeof(id_t));
  uint32_t* stack_count = (uint32_t*)mem_alloc(graph->vertex_count
                                               * sizeof(uint32_t));
  weight_t* delta = (weight_t*)mem_alloc(graph->vertex_count
                                         * sizeof(weight_t));
  uint64_t count = 0;
  int64_t phase = 0;

  // Initialization stage
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif // _OPENMP
  for (id_t v = 0; v < graph->vertex_count; v++) {
    betweenness_centrality[v] = (weight_t)0.0;
  }

  // Find and count all shortest paths from every source vertex to every other
  // vertex in the graph. These paths and counts are used to determine the
  // betweenness centrality for each vertex
  for (id_t source = 0; source < graph->vertex_count; source++) {
    // Initializations for this iteration
    memset(succ, 0, graph->edge_count * sizeof(id_t));
    memset(succ_count, 0, graph->vertex_count * sizeof(uint32_t));
    memset(stack, 0, graph->vertex_count * graph->vertex_count * sizeof(id_t));
    memset(stack_count, 0,  graph->vertex_count * sizeof(uint32_t));
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif // _OPENMP
    for (id_t t = 0; t < graph->vertex_count; t++) {
      sigma[t] = 0;
      dist[t] = -1;
    }
    sigma[source] = 1;
    dist[source] = 0;
    phase = 0;
    count = 1;
    stack_count[phase] = 1;
    stack[graph->vertex_count * phase] = source;

    // SSSP and path counting
    while (count > 0) {
      count = 0;
      for (id_t v_index = 0; v_index < stack_count[phase]; v_index++) {
        id_t v = stack[graph->vertex_count * phase + v_index];
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif // _OPENMP
        // For all neighbors of v in parallel, iterate over paths
        for (id_t e = graph->vertices[v]; e < graph->vertices[v + 1]; e++) {
          id_t w = graph->edges[e];
          int32_t dw = __sync_val_compare_and_swap(&dist[w], (uint32_t)-1,
                                                   phase + 1);
          if (dw == -1) {
            __sync_fetch_and_add(&count, 1);
            id_t p = __sync_fetch_and_add(&stack_count[phase + 1], 1);
            stack[graph->vertex_count * (phase + 1) + p] = w;
            dw = phase + 1;
          }
          if (dw == phase + 1) {
            id_t p = (id_t)__sync_fetch_and_add(&succ_count[v], 1);
            succ[graph->vertices[v] + p] = w;
            __sync_fetch_and_add(&sigma[w], sigma[v]);
          }
        }
      }
      phase++;
    }
    phase--;

    // Dependency accumulation stage
    memset(delta, (weight_t)0.0, graph->vertex_count * sizeof(id_t));
    phase--;
    while (phase > 0) {
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif // _OPENMP
      for (id_t p = 0; p < stack_count[phase]; p++) {
        id_t w = stack[graph->vertex_count * phase + p];
        weight_t dsw = 0.0;
        weight_t sw = sigma[w];
        for (id_t i = 0; i < succ_count[w]; i++) {
          id_t v = succ[graph->vertices[w] + i];
          dsw = dsw + (sw / sigma[v]) * (1.0 + delta[v]);
        }
        delta[w] = dsw;
        betweenness_centrality[w] = betweenness_centrality[w] + dsw;
      }
      phase--;
    }
  }

  // If the graph is undirected, divide centrality scores by 2
  if (graph->directed == false) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif // _OPENMP
    for (id_t v = 0; v < graph->vertex_count; v++) {
      betweenness_centrality[v] /= 2.0;
    }
  }

  // Cleanup phase
  mem_free(sigma);
  mem_free(dist);
  mem_free(delta);
  mem_free(stack_count);
  mem_free(succ);
  mem_free(succ_count);
  mem_free(stack);

  // Return the centrality
  *centrality_score = betweenness_centrality;
  return SUCCESS;
}
