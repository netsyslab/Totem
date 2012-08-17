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

extern __global__
void vwarp_bfs_kernel(graph_t graph, uint32_t level, bool* finished,
                      uint32_t* cost, uint32_t thread_count);

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU).
*/
PRIVATE
error_t check_special_cases(const graph_t* graph, bool* finished,
                            weight_t** centrality_score) {
  if ((graph == NULL) || (graph->vertex_count == 0)
      || (centrality_score == NULL)) {
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
 * Allocates and initializes memory on the GPU for the predecessors
 * implementation of closeness centrality.
 */
PRIVATE
error_t initialize_gpu(const graph_t* graph, uint64_t vertex_count,
                       graph_t** graph_d, uint32_t** dist_d,
                       bool** finished_d) {
  // Allocate space on GPU
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);
  CHK_CU_SUCCESS(cudaMalloc((void**)dist_d, vertex_count * sizeof(uint32_t)),
                 err_free_graph_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)finished_d, sizeof(bool)),
                 err_free_dist_d);
  return SUCCESS;

 err_free_dist_d:
  cudaFree(dist_d);
 err_free_graph_d:
  graph_finalize_device(*graph_d);
 err:
  return FAILURE;
}

/**
 * Finalize function for the predecessor map GPU implementation. It frees up GPU
 * resources.
 */
PRIVATE
error_t finalize_gpu(graph_t* graph_d, uint32_t* dist_d, bool* finished_d) {
  graph_finalize_device(graph_d);
  cudaFree(dist_d);
  cudaFree(finished_d);
  return SUCCESS;
}

/**
 * Runs the APSP BFS algorithm on the GPU.
 */
PRIVATE
error_t closeness_apsp_gpu(const graph_t* graph, vid_t source, graph_t* graph_d,
                           uint32_t* dist, uint32_t* dist_d, bool* finished_d) {
  // {} used to limit scope and avoid problems with error handles.
  {
  dim3 blocks;
  dim3 threads_per_block;

  CHK_CU_SUCCESS(cudaMemset(dist_d, -1, graph->vertex_count * sizeof(uint32_t)),
                 err);
  CHK_CU_SUCCESS(cudaMemset(&(dist_d[source]), 0, sizeof(uint32_t)), err);
  CHK_CU_SUCCESS(cudaDeviceSynchronize(), err);

  bool finished = false;
  int thread_count = VWARP_WARP_SIZE * VWARP_BATCH_COUNT(graph->vertex_count);
  KERNEL_CONFIGURE(thread_count, blocks, threads_per_block);
  cudaFuncSetCacheConfig(vwarp_bfs_kernel, cudaFuncCachePreferShared);
  while (!finished) {
    CHK_CU_SUCCESS(cudaMemset(finished_d, true, sizeof(bool)), err);
    vwarp_bfs_kernel<<<blocks, threads_per_block>>>
      (*graph_d, *dist, finished_d, dist_d, thread_count);
    CHK_CU_SUCCESS(cudaDeviceSynchronize(), err);
    CHK_CU_SUCCESS(cudaMemcpy(&finished, finished_d, sizeof(bool),
                              cudaMemcpyDeviceToHost), err);
    (*dist)++;
  }}
  return SUCCESS;

 err:
  return FAILURE;
}

/**
 * Calculates closeness centrality scores from the given BFS solution.
 */
PRIVATE
error_t calculate_closeness(const graph_t* graph, vid_t source, uint32_t* dists,
                            weight_t* closeness_centrality) {
  uint32_t connected = graph->vertex_count;
  uint64_t sum = 0;
  // TODO (greg): Move this to the GPU
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    if (dists[v] == (uint32_t)-1) {
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
  return SUCCESS;
}

/**
 * Implements the parallel Brandes closeness centrality algorithm using
 * predecessor maps as described in "Fast Network Centrality Analysis Using
 * GPUs" [Shi11]
 */
error_t closeness_unweighted_gpu(const graph_t* graph,
                                 weight_t** centrality_score) {
  bool finished = true;
  error_t rc = check_special_cases(graph, &finished, centrality_score);
  if (finished) return rc;

  // Allocate space for the results
  weight_t* closeness_centrality =
    (weight_t*)mem_alloc(graph->vertex_count * sizeof(weight_t));
  uint32_t* dists =
    (uint32_t*)mem_alloc(graph->vertex_count * sizeof(uint32_t));

  // Allocate memory and initialize state on the GPU
  graph_t* graph_d;
  uint32_t* dist_d;
  bool* finished_d;

  CHK_SUCCESS(initialize_gpu(graph, graph->vertex_count, &graph_d, &dist_d,
                             &finished_d), err_free_closeness);

  // Find and count all shortest paths from every source vertex to every other
  // vertex in the graph. These paths and counts are used to determine the
  // closeness centrality for each vertex
  for (vid_t source = 0; source < graph->vertex_count; source++) {
    // APSP
    uint32_t dist = 0;
    CHK_SUCCESS(closeness_apsp_gpu(graph, source, graph_d, &dist, dist_d,
                                  finished_d), err_free_all);

    // Count connected and calculate closeness centrality
    CHK_CU_SUCCESS(cudaMemcpy(dists, dist_d, graph->vertex_count
                              * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                   err_free_all);
    CHK_CU_SUCCESS(cudaDeviceSynchronize(), err_free_all);
    CHK_SUCCESS(calculate_closeness(graph, source, dists, closeness_centrality),
                err_free_all);
  }

  CHK_SUCCESS(finalize_gpu(graph_d, dist_d, finished_d), err_free_all);
  mem_free(dists);

  // Return the centrality
  *centrality_score = closeness_centrality;
  return SUCCESS;

 err_free_all:
  graph_finalize_device(graph_d);
  cudaFree(dist_d);
  cudaFree(finished_d);
 err_free_closeness:
  mem_free(closeness_centrality);
  mem_free(dists);
  return FAILURE;
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

  // Allocate and initialize state for the problem
  uint32_t* dists =
    (uint32_t*)mem_alloc(graph->vertex_count * sizeof(uint32_t));
  memset(closeness_centrality, 0.0, graph->vertex_count * sizeof(weight_t));

  for (vid_t source = 0; source < graph->vertex_count; source++) {
    // Initialize state for SSSP
    memset(dists, -1, graph->vertex_count * sizeof(uint32_t));
    dists[source] = 0;

    // SSSP and path counting
    uint32_t dist = 0;
    bool finished = false;
    while (!finished) {
      finished = true;
      OMP(omp parallel for)
      for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
        if (dists[vertex_id] != dist) continue;
        for (eid_t i = graph->vertices[vertex_id];
             i < graph->vertices[vertex_id + 1]; i++) {
          const vid_t neighbor_id = graph->edges[i];
          if (dists[neighbor_id] == (uint32_t)-1) {
            finished = false;
            dists[neighbor_id] = dist + 1;
          }
        }
      }
      dist++;
    }
    // Count connected and calculate closeness centrality
    calculate_closeness(graph, source, dists, closeness_centrality);
  } // for

  // Cleanup phase
  mem_free(dists);

  // Return the centrality
  *centrality_score = closeness_centrality;
  return SUCCESS;
}
