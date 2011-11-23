/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * All pairs shortest path algorithm based on the Floyd-Warshall algorithm for
 * the CPU implementation.
 * According to [Harish07], running Dijkstra's algorithm for every vertex on the
 * graph, as compared to a parallel Floyd-Warshall algorithm, was both more
 * memory efficient (O(V) as compared to O(V^2)) and faster.
 *
 *  Created on: 2011-11-04
 *      Author: Greg Redekop
 */

// system includes
#include <cuda.h>

// totem includes
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"


/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
 */
PRIVATE
error_t check_special_cases(graph_t* graph, weight_t **distances,
                            bool* finished) {

  *finished = true;
  if ((graph == NULL) || !graph->weighted || graph->vertex_count <= 0) {
    *distances = NULL;
    return FAILURE;
  }
  else if (graph->vertex_count == 1) {
    *distances = (weight_t*)mem_alloc(sizeof(weight_t));
    (*distances)[0] = 0;
    return SUCCESS;
  }

  id_t v_count = graph->vertex_count;
  // Check whether the graph has vertices, but an empty edge set.
  if ((v_count > 0) && (graph->edge_count == 0)) {
    *distances = (weight_t*)mem_alloc(v_count * v_count * sizeof(weight_t));

    for (id_t src = 0; src < v_count; src++) {
      // Initialize path lengths to WEIGHT_MAX.
      weight_t* base = &(*distances)[src * v_count];
      for (id_t dest = 0; dest < v_count; dest++) {
        base[dest] = (weight_t)WEIGHT_MAX;
      }
      // 0 distance to oneself
      base[src] = 0;
    }
    return SUCCESS;
  }

  *finished = false;
  return SUCCESS;
}

/**
 * All Pairs Shortest Path algorithm on the GPU.
 */
error_t apsp_gpu(graph_t* graph, weight_t** path_ret) {
  if ((graph == NULL) || (graph->vertex_count <= 0) || (!graph->weighted)) {
    *path_ret = NULL;
    return FAILURE;
  }
  // TODO(Greg): Avoid extra copies between this and Dijkstra.

  uint32_t v_count = graph->vertex_count;
  // The distances array mimics a static array to avoid the overhead of
  // creating an array of pointers. Thus, accessing index [i][j] will be
  // done as distances[(i * v_count) + j]
  weight_t* distances = (weight_t*)mem_alloc(v_count * v_count *
                                             sizeof(weight_t));
  dim3 block_count, threads_per_block;
  KERNEL_CONFIGURE(graph->vertex_count, block_count, threads_per_block);

  for (id_t i = 0; i < graph->vertex_count; i++) {
    weight_t* paths = NULL;
    // Run SSSP for the selected vertex
    CHK_SUCCESS(dijkstra_gpu(graph, i, &paths), err_free_distances);
    // Copy the result into the adjacency matrix
    memcpy(&(distances[(i * v_count)]), paths, v_count * sizeof(weight_t));
    mem_free(paths);
  }
  *path_ret = distances;
  return SUCCESS;

err_free_distances:
  mem_free(distances);
  *path_ret = NULL;
  return FAILURE;
}

/**
 * CPU implementation of the All-Pairs Shortest Path algorithm.
 */
error_t apsp_cpu(graph_t* graph, weight_t** path_ret) {
  bool finished;
  error_t ret_val = check_special_cases(graph, path_ret, &finished);
  if (finished) return ret_val;

  uint32_t v_count = graph->vertex_count;
  // The distances array mimics a static array to avoid the overhead of
  // creating an array of pointers. Thus, accessing index [i][j] will be
  // done as distances[(i * v_count) + j]
  weight_t* distances = (weight_t*)mem_alloc(v_count * v_count *
                                             sizeof(weight_t));
  // Initialize the path cost from the edge list
  for (id_t src = 0; src < v_count; src++) {
    weight_t* base = &distances[src * v_count];
    // Initialize path lengths to WEIGHT_MAX.
    for (id_t dest = 0; dest < v_count; dest++) {
      base[dest] = (weight_t)WEIGHT_MAX;
    }
    for (id_t edge = graph->vertices[src]; edge < graph->vertices[src + 1];
         edge++) {
      base[graph->edges[edge]] = graph->weights[edge];
    }
    // 0 distance to oneself
    base[src] = 0;
  }

  // Run the main loop |V| times to converge.
  for (id_t mid = 0; mid < v_count; mid++) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif // _OPENMP
    for (id_t src = 0; src < v_count; src++) {
      weight_t* base = &distances[src * v_count];
      weight_t* mid_base = &distances[mid * v_count];
      for (id_t dest = 0; dest < v_count; dest++) {
        base[dest] = min(base[dest], base[mid] + mid_base[dest]);
      }
    }
  }

  *path_ret = distances;
  return SUCCESS;
}
