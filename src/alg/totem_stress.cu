/**
 *  Defines Stress Centrality functions for both CPU and GPU.
 *
 *  Created on: 2012-06-06
 *  Author: Greg Redekop
 */

// totem includes
#include "totem_alg.h"
#include "totem_centrality.h"

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
    totem_malloc(graph->vertex_count * sizeof(weight_t), TOTEM_MEM_HOST_PINNED, 
                 (void**)centrality_score);
    memset(*centrality_score, (weight_t)0.0, graph->vertex_count
           * sizeof(weight_t));
    return SUCCESS;
  }

  *finished = false;
  return SUCCESS;
}

/**
 * Allocates and initializes memory on the GPU for the predecessors
 * implementation of stress centrality.
 */
PRIVATE
error_t initialize_gpu(const graph_t* graph, vid_t vertex_count,
                       graph_t** graph_d, uint32_t** sigma_d, uint32_t** dist_d,
                       uint32_t** delta_d, bool** finished_d,
                       weight_t** stress_centrality_d) {
  // Allocate space on GPU
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);
  CHK_CU_SUCCESS(cudaMalloc((void**)sigma_d, vertex_count * sizeof(uint32_t)),
                 err_free_graph_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)dist_d, vertex_count * sizeof(uint32_t)),
                 err_free_sigma_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)delta_d, vertex_count * sizeof(uint32_t)),
                 err_free_dist_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)finished_d, sizeof(bool)),
                 err_free_delta_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)stress_centrality_d, graph->vertex_count
                            * sizeof(weight_t)), err_free_finished_d);

  // Setup initial parameters
  CHK_CU_SUCCESS(cudaMemset(*stress_centrality_d, 0, graph->vertex_count
                            * sizeof(weight_t)), err_free_all);
  return SUCCESS;

 err_free_all:
  cudaFree(stress_centrality_d);
 err_free_finished_d:
  cudaFree(finished_d);
 err_free_delta_d:
  cudaFree(delta_d);
 err_free_dist_d:
  cudaFree(dist_d);
 err_free_sigma_d:
  cudaFree(sigma_d);
 err_free_graph_d:
  graph_finalize_device(*graph_d);
 err:
  return FAILURE;
}

/**
 * Finalize function for the predecessor map GPU implementation. It allocates
 * the host output buffer, moves the final results from GPU to the host buffers
 * and frees up GPU resources.
 */
PRIVATE
error_t finalize_gpu(graph_t* graph_d, uint32_t* sigma_d, uint32_t* dist_d,
                     uint32_t* delta_d, bool* finished_d,
                     weight_t* stress_centrality_d,
                     weight_t* stress_centrality) {
  // Copy back the centrality scores
  CHK_CU_SUCCESS(cudaMemcpy(stress_centrality, stress_centrality_d,
                            graph_d->vertex_count * sizeof(weight_t),
                            cudaMemcpyDeviceToHost), err);
  graph_finalize_device(graph_d);
  cudaFree(sigma_d);
  cudaFree(dist_d);
  cudaFree(delta_d);
  cudaFree(finished_d);
  cudaFree(stress_centrality_d);
  return SUCCESS;

 err:
  return FAILURE;
}

/**
 * Unweighted BFS single source shortest path kernel with predecessor map
 */
__global__
void unweighted_sc_sssp_kernel(graph_t graph, uint32_t dist, uint32_t* dists,
                               uint32_t* sigma, bool* finished) {
  const int vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  if (dists[vertex_id] != dist) return;

  for (eid_t i = graph.vertices[vertex_id];
       i < graph.vertices[vertex_id + 1]; i++) {
    const vid_t neighbor_id = graph.edges[i];
    if (dists[neighbor_id] == (uint32_t)-1) {
      // Threads may update finished and the same position in the cost array
      // concurrently. It does not affect correctness since all
      // threads would update with the same value.
      *finished = false;
      dists[neighbor_id] = dist + 1;
    }
    // Neighboring vertices may be reached by other source vertices and not
    // enter the first if statement. Since we need to set the sigma values for
    // the current edge and neighbor vertex respectively, this check must be
    // seperate.
    if (dists[neighbor_id] == dist + 1) {
      atomicAdd(&sigma[neighbor_id], sigma[vertex_id]);
    }
  } // for
}

/**
 * Unweighted centrality back propagation kernel for predecessor map
 * implementation. Calculates sigma_st for each vertex in the graph.
 */
__global__
void unweighted_sc_back_prop_kernel(graph_t graph, uint32_t* dists,
                                    uint32_t dist, uint32_t* delta) {
  const vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  if (dists[vertex_id] != (dist - 1)) return;

  for (eid_t i = graph.vertices[vertex_id];
       i < graph.vertices[vertex_id + 1]; i++) {
    const vid_t neighbor_id = graph.edges[i];
    if ((dists[neighbor_id] == dist)) {
      delta[vertex_id] += 1 + delta[neighbor_id];
    }
  }
}

/**
 * Unweighted stress centrality back sum kernel for predecessor map
 * implementation. This function calculates the actual stress centrality
 * score by summing path counts for each vertex.
 */
__global__
void unweighted_sc_back_sum_kernel(graph_t graph, vid_t source, uint32_t dist,
                                   uint32_t* dists, uint32_t* delta,
                                   uint32_t* sigma,
                                   weight_t* stress_centrality) {
  const vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;

  if ((vertex_id != source) && (dists[vertex_id] == dist)) {
    stress_centrality[vertex_id] += 1.0 * sigma[vertex_id] * delta[vertex_id];
  }
}

/**
 * Unweighted stress centrality APSP section. This function simultaneously
 * calculates the shortest paths and counts the number of shortest paths. It is
 * called once per source vertex in the graph.
 */
PRIVATE
error_t stress_apsp_gpu(const graph_t* graph, graph_t* graph_d, vid_t source,
                        uint32_t* dist, uint32_t* dist_d, uint32_t* sigma_d,
                        bool* finished_d) {
  // {} used to limit scope and avoid problems with error handles.
  {
  dim3 blocks;
  dim3 threads_per_block;
  // Initialize this iteration of the APSP
  CHK_CU_SUCCESS(cudaMemset(dist_d, -1, graph->vertex_count
                            * sizeof(uint32_t)), err);
  CHK_CU_SUCCESS(cudaMemset(sigma_d, 0, graph->vertex_count
                            * sizeof(uint32_t)), err);
  CHK_CU_SUCCESS(cudaDeviceSynchronize(), err);
  CHK_CU_SUCCESS(cudaMemset(&(dist_d[source]), 0, sizeof(uint32_t)), err);
  CALL_SAFE(totem_memset(&(sigma_d[source]), (uint32_t)1, 1, TOTEM_MEM_DEVICE));
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
  bool finished = false;
  while (!finished) {
    CHK_CU_SUCCESS(cudaMemset(finished_d, true, sizeof(bool)), err);
    unweighted_sc_sssp_kernel<<<blocks, threads_per_block>>>
      (*graph_d, *dist, dist_d, sigma_d, finished_d);
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
 * Unweighted back propagation function for the gpu. This function sums the
 * relative dependence and path counts for each vertex in the graph, calculating
 * stress centrality.
 */
PRIVATE
error_t stress_back_prop_gpu(const graph_t* graph, graph_t* graph_d,
                             vid_t source, uint32_t* dist_d, uint32_t* sigma_d,
                             uint32_t* dist, uint32_t* delta_d,
                             weight_t* stress_centrality_d) {
  // {} used to limit scope and avoid problems with error handles.
  {
  dim3 blocks;
  dim3 threads_per_block;

  // Initialize back propagation
  CHK_CU_SUCCESS(cudaMemset(delta_d, 0, graph->vertex_count
                            * sizeof(uint32_t)), err);
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
  while (*dist > 1) {
    (*dist)--;
    unweighted_sc_back_prop_kernel<<<blocks, threads_per_block>>>
      (*graph_d, dist_d, *dist, delta_d);
    CHK_CU_SUCCESS(cudaDeviceSynchronize(), err);
    unweighted_sc_back_sum_kernel<<<blocks, threads_per_block>>>
      (*graph_d, source, *dist, dist_d, delta_d, sigma_d, stress_centrality_d);
    CHK_CU_SUCCESS(cudaDeviceSynchronize(), err);
  }}
  return SUCCESS;

 err:
  return FAILURE;
}

/**
 * Implements the parallel Brandes stress centrality algorithm modified as
 * described in "On Variants of Shortest-Path Betweenness Centrality and their
 * Generic Computation" [Brandes07].
 */
error_t stress_unweighted_gpu(const graph_t* graph,
                              weight_t** centrality_score) {
  // Sanity check on input
  bool finished = true;
  error_t rc = check_special_cases(graph, &finished, centrality_score);
  if (finished) return rc;

  // Allocate space for the results
  weight_t* stress_centrality = NULL;
  totem_malloc(graph->vertex_count * sizeof(weight_t), TOTEM_MEM_HOST_PINNED, 
               (void**)&stress_centrality);

  // Allocate memory and initialize state on the GPU
  graph_t* graph_d;
  uint32_t* sigma_d;
  uint32_t* dist_d;
  uint32_t* delta_d;
  bool* finished_d;
  weight_t* stress_centrality_d;

  CHK_SUCCESS(initialize_gpu(graph, graph->vertex_count, &graph_d, &sigma_d,
                             &dist_d, &delta_d, &finished_d,
                             &stress_centrality_d), err_free_stress);

  // Find and count all shortest paths from every source vertex to every other
  // vertex in the graph. These paths and counts are used to determine the
  // stress centrality for each vertex
  for (vid_t source = 0; source < graph->vertex_count; source++) {
    // APSP
    uint32_t dist = 0;
    CHK_SUCCESS(stress_apsp_gpu(graph, graph_d, source, &dist, dist_d, sigma_d,
                                finished_d), err_free_all);

    // Back Propagation
    CHK_SUCCESS(stress_back_prop_gpu(graph, graph_d, source, dist_d, sigma_d,
                                     &dist, delta_d, stress_centrality_d),
                err_free_all);
  }

  CHK_SUCCESS(finalize_gpu(graph_d, sigma_d, dist_d, delta_d, finished_d,
                           stress_centrality_d, stress_centrality),
              err_free_all);

  // Return the centrality
  *centrality_score = stress_centrality;
  return SUCCESS;

 err_free_all:
  graph_finalize_device(graph_d);
  cudaFree(sigma_d);
  cudaFree(dist_d);
  cudaFree(delta_d);
  cudaFree(finished_d);
  cudaFree(stress_centrality_d);
 err_free_stress:
  totem_free(stress_centrality, TOTEM_MEM_HOST_PINNED);
  return FAILURE;
}

/**
 * Implements the parallel Brandes stress centrality algorithm modified as
 * described in "On Variants of Shortest-Path Betweenness Centrality and their
 * Generic Computation" [Brandes07].
 */
error_t stress_unweighted_cpu(const graph_t* graph,
                              weight_t** centrality_score) {
  // Sanity check on input
  bool finished = true;
  error_t rc = check_special_cases(graph, &finished, centrality_score);
  if (finished) return rc;

  // Allocate space for the results
  weight_t* stress_centrality = NULL;
  totem_malloc(graph->vertex_count * sizeof(weight_t), TOTEM_MEM_HOST_PINNED, 
               (void**)&stress_centrality);

  // Allocate and initialize state for the problem
  uint32_t* sigma = NULL;
  totem_malloc(graph->vertex_count * sizeof(uint32_t), TOTEM_MEM_HOST, 
               (void**)&sigma);
  uint32_t* delta = NULL;
  totem_malloc(graph->vertex_count * sizeof(uint32_t), TOTEM_MEM_HOST, 
               (void**)&delta);
  uint32_t* dists = NULL;
  totem_malloc(graph->vertex_count * sizeof(uint32_t), TOTEM_MEM_HOST, 
               (void**)&dists);
  memset(stress_centrality, 0.0, graph->vertex_count * sizeof(weight_t));

  for (vid_t source = 0; source < graph->vertex_count; source++) {
    // Initialize state for SSSP
    memset(sigma, 0, graph->vertex_count * sizeof(uint32_t));
    memset(delta, 0, graph->vertex_count * sizeof(uint32_t));
    memset(dists, -1, graph->vertex_count * sizeof(uint32_t));
    dists[source] = 0;
    sigma[source] = 1;

    // SSSP and path counting
    uint32_t dist = 0;
    bool finished = false;
    while (!finished) {
      finished = true;
      OMP(omp parallel for)
      for (vid_t u = 0; u < graph->vertex_count; u++) {
        for (eid_t e = graph->vertices[u]; e < graph->vertices[u + 1]; e++) {
          // For edge (u,v)
          vid_t v = graph->edges[e];

          if (dists[u] == dist) {
            if (dists[v] == (uint32_t)-1) {
              finished = false;
              dists[v] = dist + 1;
            }
            // Neighboring vertices may be reached by other source vertices and
            // not enter the first if statement. Since we need to set the sigma
            // values for the current edge and neighbor vertex respectively,
            // this check must be seperate.
            if (dists[v] == dist + 1) {
              __sync_fetch_and_add(&sigma[v], sigma[u]);
            }
          }
        }
      }
      dist++;
    }

    // Back Propagation
    while (dist > 1) {
      dist--;
      OMP(omp parallel for)
      for (vid_t v = 0; v < graph->vertex_count; v++) {
        if (dists[v] != (dist - 1)) continue;
        for (eid_t e = graph->vertices[v]; e < graph->vertices[v + 1]; e++) {
          // For edge (u,v)
          vid_t u = graph->edges[e];

          if (dists[u] == dist) {
            delta[v] += 1 + delta[u];
          }
        }
      };
      OMP(omp parallel for)
      for (vid_t v = 0; v < graph->vertex_count; v++) {
        if (v != source && dists[v] == dist) {
          stress_centrality[v] += 1.0 * sigma[v] * delta[v];
        }
      }
    }
  } // for

  // Cleanup phase
  totem_free(sigma, TOTEM_MEM_HOST);
  totem_free(delta, TOTEM_MEM_HOST);
  totem_free(dists, TOTEM_MEM_HOST);

  // Return the centrality
  *centrality_score = stress_centrality;
  return SUCCESS;
}
