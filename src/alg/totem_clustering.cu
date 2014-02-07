/**
 *
 *  Implements Clustering Coefficient algorithm for CPU.
 *
 *  Created on: 2013-07-09
 *  Author: Sidney Pontes Filho
 *
 *  Modified on: 2014-01-20
 *  Author: Tahsin Arafat Reza 
 */

// totem includes
#include "totem_alg.h"
#include "totem_centrality.h"

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (CPU and GPU).
 * @param[in] graph
 * @param[in] finished
 * @param[in] coefficients
 */
PRIVATE
error_t check_special_cases(const graph_t* graph, bool* finished,
                            weight_t** coefficients) {

  // Check whether the graph is null or vertex set is empty
  if ((graph == NULL) || (graph->vertex_count == 0)
      || (coefficients == NULL)) {
    return FAILURE;
  }

  // Check whether the edge set is empty 
  if (graph->edge_count == 0) {
    totem_malloc(graph->vertex_count * sizeof(weight_t), TOTEM_MEM_HOST_PINNED, 
                 (void**)coefficients);
    memset(*coefficients, (weight_t)0.0, graph->vertex_count
           * sizeof(weight_t));
    return SUCCESS;
  }

  *finished = false;
  return SUCCESS;
}

/**
 * Implements the CPU-only clustering coefficient algorithm.
 *
 * Given a graph \f$G = (V, E)\f$, the clustering coefficient (\f$CC\f$) 
 * of a vertex \f$v\inV\f$ with degree \f$d\f$ is defined as 
 * \f$CC = 2*T / d(d-1)f$, where \f$Tf$ is the number of triangles 
 * incident on \f$v\f$.    
 *
 * The outmost loop is parallelized with OpenMP. Each vertex performs 
 * computation in a unique thread. Computaions of triangle
 * counting and clustering coeffienct involves writing to thread 
 * local variables only. Therefore, vertices can carryout computation in 
 * parallel without interrupting each other. Each vertex iterates through 
 * the list of its neighbours (and the list of neighbours of each of its 
 * neighbours) in a sequential manner, to verify presence of a common neigbour; 
 * hence, count trainagles and calculate clustering coefficeint.           
 *
 * @param[in] graph the input graph
 * @param[out] coefficients array containing computed coefficients 
 */
error_t clustering_coefficient_cpu(const graph_t* graph,
                                   weight_t** coefficients) {
  // Check inputs
  bool finished = true;
  error_t rc = check_special_cases(graph, &finished, coefficients);
  if (finished) return rc;

  // Allocate memory for the results
  weight_t* clustering_coefficients = NULL;
  totem_malloc(graph->vertex_count * sizeof(weight_t), TOTEM_MEM_HOST_PINNED, 
               (void**)&clustering_coefficients);
  
  memset(clustering_coefficients, 0.0, graph->vertex_count * sizeof(weight_t));

  OMP(omp parallel for)
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    uint32_t triangle_count_v = 0;
    for (eid_t e = graph->vertices[v]; e < graph->vertices[v + 1]; e++) {
      vid_t e_v = graph->edges[e];  // v's neighbour  
      for (eid_t f = graph->vertices[e_v]; f < graph->vertices[e_v + 1]; f++) {
        vid_t f_e_v = graph->edges[f];  // Neighbour of v's neighbour  
        for (eid_t u = graph->vertices[v]; u < graph->vertices[v + 1]; u++) {
          vid_t u_v = graph->edges[u]; // v's neighbour  
          if (u_v != e_v && u_v == f_e_v) {  // Common neighbour verification  
            triangle_count_v++;
          }
        } // for
      } // for
    } // for

    uint32_t degree_v = graph->vertices[v + 1] - graph->vertices[v]; 

    clustering_coefficients[v] = 0.0f;

    weight_t triangles_v = (weight_t)triangle_count_v/2.0f; 

    if (triangles_v > 0.0f) {      
      clustering_coefficients[v] = 
        (2.0f * triangles_v) / (((weight_t)degree_v - 1.0f) * 
                                (weight_t)degree_v);
    }
  } // parallel for 
   
  *coefficients = clustering_coefficients;
  return SUCCESS;
}
