/**
 *  Defines Clustering Coefficient functions for CPU.
 *
 *  Created on: 2013-07-09
 *  Author: Sidney Pontes Filho
 */

// totem includes
#include "totem_alg.h"
#include "totem_centrality.h"

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces.
 */
PRIVATE
error_t check_special_cases(const graph_t* graph, bool* finished,
                            weight_t** coefficient_score) {
  if ((graph == NULL) || (graph->vertex_count == 0)
      || (coefficient_score == NULL)) {
    return FAILURE;
  }

  if (graph->edge_count == 0) {
    totem_malloc(graph->vertex_count * sizeof(weight_t), TOTEM_MEM_HOST_PINNED, 
                 (void**)coefficient_score);
    memset(*coefficient_score, (weight_t)0.0, graph->vertex_count
           * sizeof(weight_t));
    return SUCCESS;
  }

  *finished = false;
  return SUCCESS;
}

/**
 * Implements the CPU-based local clustering coefficient.
 */
error_t local_clustering_coefficient_cpu(const graph_t* graph,
                                         weight_t** coefficient_score) {
  // Sanity check on input
  bool finished = true;
  error_t rc = check_special_cases(graph, &finished, coefficient_score);
  if (finished) return rc;

  // Allocate space for the results
  weight_t* clustering_coefficient = NULL;
  totem_malloc(graph->vertex_count * sizeof(weight_t), TOTEM_MEM_HOST_PINNED, 
               (void**)&clustering_coefficient);

  // Allocate and initialize state for the problem
  uint32_t* triangles = NULL;
  totem_malloc(graph->vertex_count * sizeof(uint32_t), TOTEM_MEM_HOST,
               (void**)&triangles);
  uint32_t* neighbours = NULL;
  totem_malloc(graph->vertex_count * sizeof(uint32_t), TOTEM_MEM_HOST, 
               (void**)&neighbours);
  memset(clustering_coefficient, 0.0, graph->vertex_count * sizeof(weight_t));
  
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    // Initialize state
    memset(triangles, 0, graph->vertex_count * sizeof(uint32_t));
    triangles[v] = 0;
    memset(neighbours, 0, graph->vertex_count * sizeof(uint32_t));
    neighbours[v] = 0;
    for (eid_t e = graph->vertices[v]; 
         e < graph->vertices[v + 1] - 1; e++) {
      for (eid_t f = e + 1; 
           f < graph->vertices[v + 1]; f++) {
        for (eid_t g = graph->vertices[graph->edges[e]]; 
             g < graph->vertices[graph->edges[e + 1]]; g++) {
          // Counting triangles formed by the neighbours
          if (graph->edges[g] == graph->edges[f] 
              && f != g) {
            triangles[v]++;
          }
        }
      }
    }

    // Counting vertice's neighbours
    neighbours[v] = graph->vertices[v + 1] - graph->vertices[v];
    // Counting local clustering coefficient
    clustering_coefficient[v] = 2.0f * (weight_t)triangles[v] 
      / (((weight_t)neighbours[v] - 1.0f) * (weight_t)neighbours[v]);
    //clustering_coefficient[v] = (weight_t)triangles[v];
  }
  
  // Cleanup phase
  totem_free(neighbours, TOTEM_MEM_HOST);
  totem_free(triangles, TOTEM_MEM_HOST);

  // Return the coefficient
  *coefficient_score = clustering_coefficient;
  return SUCCESS;
}
