/**
 *  Defines Betweenness Centrality functions for both CPU and GPU.
 *
 *  Created on: 2012-05-24
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
    *centrality_score = (weight_t*)mem_alloc(graph->vertex_count
                                             * sizeof(weight_t));
    memset(*centrality_score, (weight_t)0.0, graph->vertex_count
           * sizeof(weight_t));
    return SUCCESS;
  }

  *finished = false;
  return SUCCESS;
}

/**
 * Allocates and initializes memory on the GPU for the successors implementation
 * of betweenness centrality.
 */
PRIVATE
error_t initialize_succs_gpu(const graph_t* graph, uint64_t vertex_count,
                             graph_t** graph_d, vid_t** sigma_d,
                             int32_t** dists_d, vid_t** succ_d,
                             uint32_t** succ_count_d, vid_t** stack_d,
                             uint32_t** stack_count_d, weight_t** delta_d,
                             bool** finished_d,
                             weight_t** betweenness_centrality_d) {
  // Allocate space on GPU
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);
  CHK_CU_SUCCESS(cudaMalloc((void**)sigma_d, vertex_count * sizeof(vid_t)),
                 err_free_graph_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)dists_d, vertex_count * sizeof(int32_t)),
                 err_free_sigma_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)succ_d, graph->edge_count * sizeof(vid_t)),
                 err_free_dists_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)succ_count_d, vertex_count
                            * sizeof(uint32_t)), err_free_succ_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)stack_d, vertex_count * vertex_count
                            * sizeof(vid_t)), err_free_succ_count_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)stack_count_d, vertex_count
                            * sizeof(uint32_t)), err_free_stack_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)delta_d, vertex_count * sizeof(weight_t)),
                 err_free_stack_count_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)finished_d, sizeof(bool)),
                 err_free_delta_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)betweenness_centrality_d, vertex_count
                            * sizeof(weight_t)), err_free_finished_d);

  // Setup initial parameters
  CHK_CU_SUCCESS(cudaMemset(*betweenness_centrality_d, (weight_t)0.0,
                            vertex_count * sizeof(weight_t)), err_free_all);
  return SUCCESS;

 err_free_all:
  cudaFree(betweenness_centrality_d);
 err_free_finished_d:
  cudaFree(finished_d);
 err_free_delta_d:
  cudaFree(delta_d);
 err_free_stack_count_d:
  cudaFree(stack_count_d);
 err_free_stack_d:
  cudaFree(stack_d);
 err_free_succ_count_d:
  cudaFree(succ_count_d);
 err_free_succ_d:
  cudaFree(succ_d);
 err_free_dists_d:
  cudaFree(dists_d);
 err_free_sigma_d:
  cudaFree(sigma_d);
 err_free_graph_d:
  graph_finalize_device(*graph_d);
 err:
  return FAILURE;
}

/**
 * Allocates and initializes memory on the GPU for the predecessors
 * implementation of betweenness centrality.
 */
PRIVATE
error_t initialize_preds_gpu(const graph_t* graph, uint64_t vertex_count,
                             vid_t* r_edges, graph_t** graph_d, 
                             vid_t** r_edges_d, bool** preds_d, 
                             vid_t** sigma_d, int32_t** dist_d,
                             weight_t** delta_d, bool** finished_d,
                             weight_t** betweenness_centrality_d) {
  // Allocate space on GPU
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);
  CHK_CU_SUCCESS(cudaMalloc((void**)r_edges_d, graph->edge_count
                            * sizeof(vid_t)), err_free_graph_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)preds_d, graph->edge_count * sizeof(vid_t)),
                 err_free_r_edges_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)sigma_d, vertex_count * sizeof(vid_t)),
                 err_free_preds_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)dist_d, vertex_count * sizeof(int32_t)),
                 err_free_sigma_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)delta_d, vertex_count * sizeof(weight_t)),
                 err_free_dist_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)finished_d, sizeof(bool)),
                 err_free_delta_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)betweenness_centrality_d,
                             graph->vertex_count * sizeof(weight_t)),
                 err_free_finished_d);

  // Setup initial parameters
  CHK_CU_SUCCESS(cudaMemcpy(*r_edges_d, r_edges, graph->edge_count
                            * sizeof(vid_t), cudaMemcpyHostToDevice),
                 err_free_all);
  CHK_CU_SUCCESS(cudaMemset(*betweenness_centrality_d, 0, graph->vertex_count
                            * sizeof(weight_t)), err_free_all);
  return SUCCESS;

 err_free_all:
  cudaFree(betweenness_centrality_d);
 err_free_finished_d:
  cudaFree(finished_d);
 err_free_delta_d:
  cudaFree(delta_d);
 err_free_dist_d:
  cudaFree(dist_d);
 err_free_sigma_d:
  cudaFree(sigma_d);
 err_free_preds_d:
  cudaFree(preds_d);
 err_free_r_edges_d:
  cudaFree(r_edges_d);
 err_free_graph_d:
  graph_finalize_device(*graph_d);
 err:
  return FAILURE;
}

/**
 * Finalize function for the successor stack GPU implementation. It allocates
 * the host output buffer, moves the final results from GPU to the host buffers
 * and frees up GPU resources.
 */
PRIVATE
error_t finalize_succs_gpu(graph_t* graph_d, vid_t* sigma_d, int32_t* dist_d,
                           vid_t* succ_d, uint32_t* succ_count_d, 
                           vid_t* stack_d, uint32_t* stack_count_d, 
                           weight_t* delta_d, bool* finished_d, 
                           weight_t* betweenness_centrality_d,
                           weight_t* betweenness_centrality) {
  // Copy back the centrality scores
  CHK_CU_SUCCESS(cudaMemcpy(betweenness_centrality, betweenness_centrality_d,
                            graph_d->vertex_count * sizeof(weight_t),
                            cudaMemcpyDeviceToHost), err);
  graph_finalize_device(graph_d);
  cudaFree(sigma_d);
  cudaFree(dist_d);
  cudaFree(succ_d);
  cudaFree(succ_count_d);
  cudaFree(stack_d);
  cudaFree(stack_count_d);
  cudaFree(delta_d);
  cudaFree(finished_d);
  cudaFree(betweenness_centrality_d);
  return SUCCESS;

 err:
  return FAILURE;
}

/**
 * Finalize function for the predecessor map GPU implementation. It allocates
 * the host output buffer, moves the final results from GPU to the host buffers
 * and frees up GPU resources.
 */
PRIVATE
error_t finalize_preds_gpu(graph_t* graph_d, vid_t* r_edges_d, bool* preds_d,
                           vid_t* sigma_d, int32_t* dist_d, weight_t* delta_d,
                           bool* finished_d, weight_t* betweenness_centrality_d,
                           weight_t* betweenness_centrality) {
  // Copy back the centrality scores
  CHK_CU_SUCCESS(cudaMemcpy(betweenness_centrality, betweenness_centrality_d,
                            graph_d->vertex_count * sizeof(weight_t),
                            cudaMemcpyDeviceToHost), err);
  graph_finalize_device(graph_d);
  cudaFree(r_edges_d);
  cudaFree(preds_d);
  cudaFree(sigma_d);
  cudaFree(dist_d);
  cudaFree(delta_d);
  cudaFree(finished_d);
  cudaFree(betweenness_centrality_d);
  return SUCCESS;

 err:
  return FAILURE;
}

/**
 * This kernel is invoked for each iteration of the successors GPU betweenness
 * algorithm. It re-initializes variables for the SSSP problem using a different
 * source vertex.
 */
__global__
void unweighted_bc_succs_init_kernel(vid_t source, vid_t* sigma, int32_t* dist,
                                     uint32_t* stack_count, vid_t* stack) {
  sigma[source] = 1;
  dist[source] = 0;
  stack_count[0] = 1;
  stack[0] = source;
}

/**
 * This kernel is invoked for each iteration of the predecessors GPU betweenness
 * algorithm. It re-initializes variables for the SSSP problem using a different
 * source vertex.
 */
__global__
void unweighted_bc_preds_init_kernel(vid_t source, int32_t* dist, 
                                     vid_t* sigma) {
  dist[source] = 0;
  sigma[source] = 1;
}

/**
 * For each iteration of the successors GPU betweenness algorithm, we have to
 * reset all the variables and setup the initial parameters for the SSSP problem
 * using the new source vertex.
 */
PRIVATE
error_t unweighted_succs_init(const graph_t* graph, vid_t source, vid_t* sigma,
                              int32_t* dist, vid_t* succ, uint32_t* succ_count,
                              vid_t* stack, uint32_t* stack_count,
                              weight_t* delta) {
  // Perform the memsets directly on the GPU
  dim3 blocks;
  dim3 threads_per_block;
  CHK_CU_SUCCESS(cudaMemset(succ, 0, graph->edge_count * sizeof(vid_t)), err);
  CHK_CU_SUCCESS(cudaMemset(stack, 0, graph->vertex_count * graph->vertex_count
                            * sizeof(vid_t)), err);
  CHK_CU_SUCCESS(cudaMemset(succ_count, 0, graph->vertex_count
                            * sizeof(uint32_t)), err);
  CHK_CU_SUCCESS(cudaMemset(stack_count, 0, graph->vertex_count
                            * sizeof(uint32_t)), err);
  CHK_CU_SUCCESS(cudaMemset(sigma, 0, graph->vertex_count * sizeof(vid_t)), 
                 err);
  CHK_CU_SUCCESS(cudaMemset(dist, -1, graph->vertex_count * sizeof(int32_t)),
                 err);
  CHK_CU_SUCCESS(cudaDeviceSynchronize(), err);

  // Initialize the appropriate counts for the source vertex
  KERNEL_CONFIGURE(1, blocks, threads_per_block);
  unweighted_bc_succs_init_kernel<<<blocks, threads_per_block>>>
    (source, sigma, dist, stack_count, stack);
  CALL_CU_SAFE(cudaGetLastError());
  return SUCCESS;

 err:
  return FAILURE;
}

/**
 * Unweighted betweenness centrality dependence accumulation kernel for the
 * successors stack GPU implementation. After finding the APSP solution and
 * counts of shortest paths through each node, this function calculates the
 * dependence for each node.
 */
__global__ void
unweighted_dep_acc_kernel(graph_t graph, int64_t phase, uint32_t* stack_count, 
                          vid_t* sigma, vid_t* stack, vid_t* succ, 
                          uint32_t* succ_count, weight_t* delta, 
                          weight_t* betweenness_centrality) {
  const vid_t thread_id = THREAD_GLOBAL_INDEX;

  if (thread_id < stack_count[phase]) {
    vid_t w = stack[graph.vertex_count * phase + thread_id];
    weight_t dsw = 0.0;
    weight_t sw = sigma[w];
    for (vid_t i = 0; i < succ_count[w]; i++) {
      vid_t v = succ[graph.vertices[w] + i];
      dsw = dsw + (sw / sigma[v]) * (1.0 + delta[v]);
    }
    delta[w] = dsw;
    atomicAdd(&betweenness_centrality[w], dsw);
  }
}

/**
 * Unweighted betweenness centrality back sum kernel for predecessor map
 * implementation. This function calculates the actual betweenness centrality
 * score by summing dependences for each vertex.
 */
__global__
void unweighted_back_sum_kernel(graph_t graph, vid_t source, int32_t dist,
                                int32_t* dists, weight_t* delta,
                                weight_t* betweenness_centrality) {
  const vid_t thread_id = THREAD_GLOBAL_INDEX;
  if (thread_id < graph.vertex_count) {
    if (thread_id != source && dists[thread_id] == (dist - 1)) {
      betweenness_centrality[thread_id] += delta[thread_id];
    }
  }
}

/**
 * Implements the parallel Brandes betweenness centrality algorithm using a
 * successor stack, as described in "A Faster Parallel Algorithm and Efficient
 * Multithreaded Implementations for Evaluating Betweenness Centrality on
 * Massive Datasets" [Madduri09]
 */
error_t betweenness_unweighted_gpu(const graph_t* graph,
                                   weight_t** centrality_score) {
  // Sanity check on input
  bool finished = true;
  error_t rc = check_special_cases(graph, &finished, centrality_score);
  if (finished) return rc;

  // Allocate space for the results
  weight_t* betweenness_centrality =
    (weight_t*)mem_alloc(graph->vertex_count * sizeof(weight_t));
  // Allocate memory and initialize state on the GPU
  graph_t* graph_d;
  vid_t* sigma_d;
  int32_t* dist_d;
  vid_t* succ_d;
  uint32_t* succ_count_d;
  vid_t* stack_d;
  uint32_t* stack_count_d;
  weight_t* delta_d;
  bool* finished_d;
  weight_t* betweenness_centrality_d;

  // Initialization stage
  CHK_SUCCESS(initialize_succs_gpu(graph, graph->vertex_count, &graph_d,
                                   &sigma_d, &dist_d, &succ_d, &succ_count_d,
                                   &stack_d, &stack_count_d, &delta_d,
                                   &finished_d, &betweenness_centrality_d),
              err_free_betweenness);

  // {} used to limit scope and avoid problems with error handles.
  {
  dim3 blocks;
  dim3 threads_per_block;

  // Find and count all shortest paths from every source vertex to every other
  // vertex in the graph. These paths and counts are used to determine the
  // betweenness centrality for each vertex
  for (vid_t source = 0; source < graph->vertex_count; source++) {
    // Initializations for this iteration
    CHK_SUCCESS(unweighted_succs_init(graph, source, sigma_d, dist_d, succ_d,
                                      succ_count_d, stack_d, stack_count_d,
                                      delta_d), err_free_all);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);

    // SSSP and path counting stage
    KERNEL_CONFIGURE(graph->edge_count, blocks, threads_per_block);
    bool finished = false;
    int64_t phase = 0;
    // Keep counting distances until the BFS kernel completes
    while (!finished) {
      CHK_CU_SUCCESS(cudaMemset(finished_d, true, sizeof(bool)), err_free_all);
      CHK_CU_SUCCESS(cudaDeviceSynchronize(), err_free_all);
      unweighted_sssp_succs_kernel<<<blocks, threads_per_block>>>
        (*graph_d, phase, sigma_d, dist_d, succ_d, succ_count_d, stack_d,
         stack_count_d, finished_d);
      CHK_CU_SUCCESS(cudaMemcpy(&finished, finished_d, sizeof(bool),
                                cudaMemcpyDeviceToHost), err_free_all);
      phase++;
    }

    // Dependency accumulation stage
    phase -= 2;
    CHK_CU_SUCCESS(cudaMemset(delta_d, (weight_t)0.0,
                              graph->vertex_count * sizeof(vid_t)),
                   err_free_all);
    KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
    while (phase > 0) {
      unweighted_dep_acc_kernel<<<blocks, threads_per_block>>>
        (*graph_d, phase, stack_count_d, sigma_d, stack_d, succ_d, succ_count_d,
         delta_d, betweenness_centrality_d);
      CHK_CU_SUCCESS(cudaDeviceSynchronize(), err_free_all);
      CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
      phase--;
    }
  }}

  // Cleanup phase
  CHK_SUCCESS(finalize_succs_gpu(graph_d, sigma_d, dist_d, succ_d, succ_count_d,
                                 stack_d, stack_count_d, delta_d, finished_d,
                                 betweenness_centrality_d,
                                 betweenness_centrality), err_free_all);

  // If the graph is undirected, divide centrality scores by 2
  if (graph->directed == false) {
    for (vid_t v = 0; v < graph->vertex_count; v++) {
      betweenness_centrality[v] /= 2.0;
    }
  }

  // Return the centrality
  *centrality_score = betweenness_centrality;
  return SUCCESS;

 err_free_all:
  graph_finalize_device(graph_d);
  cudaFree(sigma_d);
  cudaFree(dist_d);
  cudaFree(succ_d);
  cudaFree(succ_count_d);
  cudaFree(stack_d);
  cudaFree(stack_count_d);
  cudaFree(delta_d);
  cudaFree(finished_d);
  cudaFree(betweenness_centrality_d);
 err_free_betweenness:
  mem_free(betweenness_centrality);
  return FAILURE;
}

/**
 * Implements the parallel Brandes betweenness centrality algorithm using
 * predecessor maps as described in "Fast Network Centrality Analysis Using
 * GPUs" [Shi11]
 */
error_t betweenness_unweighted_shi_gpu(const graph_t* graph,
                                       weight_t** centrality_score) {
  // Sanity check on input
  bool finished = true;
  error_t rc = check_special_cases(graph, &finished, centrality_score);
  if (finished) return rc;

  // Allocate space for the results
  weight_t* betweenness_centrality =
    (weight_t*)mem_alloc(graph->vertex_count * sizeof(weight_t));
  // Construct the reverse edges list (graph->edges is a list of destination
  // vertices, r_edges is a list of source vertices, indexed by edge id)
  vid_t* r_edges = (vid_t*)mem_alloc(graph->edge_count * sizeof(vid_t));
  vid_t v = 0;
  for (eid_t e = 0; e < graph->edge_count; e++) {
    while (v <= graph->vertex_count &&
           !(e >= graph->vertices[v] && e < graph->vertices[v+1])) {
      v++;
    }
    r_edges[e] = v;
  }

  // Allocate memory and initialize state on the GPU
  graph_t* graph_d;
  vid_t* r_edges_d;
  bool* preds_d;
  vid_t* sigma_d;
  int32_t* dist_d;
  weight_t* delta_d;
  bool* finished_d;
  weight_t* betweenness_centrality_d;


  CHK_SUCCESS(initialize_preds_gpu(graph, graph->vertex_count, r_edges,
                                   &graph_d, &r_edges_d, &preds_d, &sigma_d,
                                   &dist_d, & delta_d, &finished_d,
                                   &betweenness_centrality_d),
              err_free_betweenness);

  // {} used to limit scope and avoid problems with error handles.
  {
  dim3 blocks;
  dim3 threads_per_block;

  // Find and count all shortest paths from every source vertex to every other
  // vertex in the graph. These paths and counts are used to determine the
  // betweenness centrality for each vertex
  for (vid_t source = 0; source < graph->vertex_count; source++) {
    // APSP
    int32_t dist = 0;
    CHK_CU_SUCCESS(cudaMemset(dist_d, -1, graph->vertex_count
                                          * sizeof(int32_t)), err_free_all);
    CHK_CU_SUCCESS(cudaMemset(preds_d, false, graph->edge_count * sizeof(bool)),
                   err_free_all);
    CHK_CU_SUCCESS(cudaMemset(sigma_d, 0, graph->vertex_count * sizeof(vid_t)),
                   err_free_all);
    CHK_CU_SUCCESS(cudaMemset(delta_d, 0, graph->vertex_count
                              * sizeof(weight_t)), err_free_all);
    KERNEL_CONFIGURE(1, blocks, threads_per_block);
    unweighted_bc_preds_init_kernel<<<blocks, threads_per_block>>>
      (source, dist_d, sigma_d);
    CHK_CU_SUCCESS(cudaDeviceSynchronize(), err_free_all);

    KERNEL_CONFIGURE(graph->edge_count, blocks, threads_per_block);
    bool finished = false;
    while (!finished) {
      CHK_CU_SUCCESS(cudaMemset(finished_d, true, sizeof(bool)), err_free_all);
      unweighted_sssp_preds_kernel<<<blocks, threads_per_block>>>
        (*graph_d, r_edges_d, dist, dist_d, sigma_d, preds_d, finished_d);
      CHK_CU_SUCCESS(cudaDeviceSynchronize(), err_free_all);
      CHK_CU_SUCCESS(cudaMemcpy(&finished, finished_d, sizeof(bool),
                                cudaMemcpyDeviceToHost), err_free_all);
      dist++;
    }
    // Back Propogation
    while (dist > 1) {
      KERNEL_CONFIGURE(graph->edge_count, blocks, threads_per_block);
      unweighted_back_prop_kernel<<<blocks, threads_per_block>>>
        (*graph_d, r_edges_d, dist_d, sigma_d, preds_d, dist, delta_d);
      KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
      unweighted_back_sum_kernel<<<blocks, threads_per_block>>>
        (*graph_d, source, dist, dist_d, delta_d, betweenness_centrality_d);
      dist--;
    }
  }}

  CHK_SUCCESS(finalize_preds_gpu(graph_d, r_edges_d, preds_d, sigma_d, dist_d,
                                 delta_d, finished_d, betweenness_centrality_d,
                                 betweenness_centrality), err_free_all);
  mem_free(r_edges);

  // If the graph is undirected, divide all the centrality scores by two
  if (graph->directed == false) {
    for (vid_t v = 0; v < graph->vertex_count; v++) {
      betweenness_centrality[v] /= 2.0;
    }
  }

  // Return the centrality
  *centrality_score = betweenness_centrality;
  return SUCCESS;

 err_free_all:
  graph_finalize_device(graph_d);
  cudaFree(r_edges_d);
  cudaFree(preds_d);
  cudaFree(sigma_d);
  cudaFree(dist_d);
  cudaFree(delta_d);
  cudaFree(finished_d);
  cudaFree(betweenness_centrality_d);
 err_free_betweenness:
  mem_free(betweenness_centrality);
  mem_free(r_edges);
  return FAILURE;
}

/**
 * Implements the parallel Brandes betweenness centrality algorithm using a
 * successor stack, as described in "A Faster Parallel Algorithm and Efficient
 * Multithreaded Implementations for Evaluating Betweenness Centrality on
 * Massive Datasets" [Madduri09]
 */
error_t betweenness_unweighted_cpu(const graph_t* graph,
                                   weight_t** centrality_score) {
  // Sanity check on input
  bool finished = true;
  error_t rc = check_special_cases(graph, &finished, centrality_score);
  if (finished) return rc;

  // Allocate memory for the shortest paths problem
  weight_t* betweenness_centrality =
    (weight_t*)mem_alloc(graph->vertex_count * sizeof(weight_t));
  uint32_t* sigma = 
    (uint32_t*)mem_alloc(graph->vertex_count * sizeof(uint32_t));
  int32_t* dist = (int32_t*)mem_alloc(graph->vertex_count * sizeof(int32_t));
  vid_t* succ = (vid_t*)mem_alloc(graph->edge_count * sizeof(vid_t));
  vid_t* succ_count = (vid_t*)mem_alloc(graph->vertex_count * sizeof(vid_t));
  vid_t* stack = (vid_t*)mem_alloc(graph->vertex_count * graph->vertex_count
                                 * sizeof(vid_t));
  vid_t* stack_count = (vid_t*)mem_alloc(graph->vertex_count * sizeof(vid_t));
  weight_t* delta =
    (weight_t*)mem_alloc(graph->vertex_count * sizeof(weight_t));
  int64_t phase = 0;

  // Initialization stage
  OMP(omp parallel for)
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    betweenness_centrality[v] = (weight_t)0.0;
  }

  // Find and count all shortest paths from every source vertex to every other
  // vertex in the graph. These paths and counts are used to determine the
  // betweenness centrality for each vertex
  for (vid_t source = 0; source < graph->vertex_count; source++) {
    // Initializations for this iteration
    memset(succ, 0, graph->edge_count * sizeof(vid_t));
    memset(succ_count, 0, graph->vertex_count * sizeof(vid_t));
    memset(stack, 0, graph->vertex_count * graph->vertex_count * sizeof(vid_t));
    memset(stack_count, 0,  graph->vertex_count * sizeof(vid_t));
    OMP(omp parallel for)
    for (vid_t t = 0; t < graph->vertex_count; t++) {
      sigma[t] = 0;
      dist[t] = -1;
    }
    sigma[source] = 1;
    dist[source] = 0;
    phase = 0;
    stack_count[phase] = 1;
    stack[graph->vertex_count * phase] = source;

    // SSSP and path counting
    bool finished = false;
    while (!finished) {
      finished = true;
      for (vid_t v_index = 0; v_index < stack_count[phase]; v_index++) {
        vid_t v = stack[graph->vertex_count * phase + v_index];
        // For all neighbors of v in parallel, iterate over paths
        OMP(omp parallel for)
        for (eid_t e = graph->vertices[v]; e < graph->vertices[v + 1]; e++) {
          vid_t w = graph->edges[e];
          int32_t dw = __sync_val_compare_and_swap(&dist[w], (uint32_t)-1,
                                                   phase + 1);
          if (dw == -1) {
            finished = false;
            vid_t p = __sync_fetch_and_add(&stack_count[phase + 1], 1);
            stack[graph->vertex_count * (phase + 1) + p] = w;
            dw = phase + 1;
          }
          if (dw == phase + 1) {
            vid_t p = (vid_t)__sync_fetch_and_add(&succ_count[v], 1);
            succ[graph->vertices[v] + p] = w;
            __sync_fetch_and_add(&sigma[w], sigma[v]);
          }
        }
      }
      phase++;
    }
    phase--;

    // Dependency accumulation stage
    memset(delta, (weight_t)0.0, graph->vertex_count * sizeof(vid_t));
    phase--;
    while (phase > 0) {
      OMP(omp parallel for)
      for (vid_t p = 0; p < stack_count[phase]; p++) {
        vid_t w = stack[graph->vertex_count * phase + p];
        weight_t dsw = 0.0;
        weight_t sw = sigma[w];
        for (vid_t i = 0; i < succ_count[w]; i++) {
          vid_t v = succ[graph->vertices[w] + i];
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
    OMP(omp parallel for)
    for (vid_t v = 0; v < graph->vertex_count; v++) {
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

/**
 * Implements the forward propagation phase of the Betweenness Centrality
 * Algorithm described in Chapter 2 of GPU Computing Gems. Utilized by:
 * error_t betweenness_cpu(const graph_t* graph, weight_t** centrality_score)
 * @param[in] graph the graph for which the centrality measure is calculated
 * @param[in] source the source node for the shortest paths
 * @param[in] level the shared level variable between backward and forward
 *            propagations
 * @param[in] numSPs an array which counts the number of shortest paths in
 *            which each node is involved 
 * @param[in] distance an array which stores the distance of the shortest
 *            path for each node
 * @return void
 */
PRIVATE void betweenness_cpu_forward_propagation(const graph_t* graph, 
                                        vid_t source, int64_t& level,
                                        uint32_t* numSPs, int32_t* distance) {
  // Initialize the shortest path count to 0 and distance to infinity given
  // this source node
  OMP(omp parallel for)
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    numSPs[v] = 0;
    distance[v] = -1;
  }
  // Set the distance from source to itself to 0
  distance[source] = 0;
  // Set the shortest path count to 1 (from source to itself)
  numSPs[source] = 1;

  bool done = false;
  while (!done) {
    done = true;
    // In parallel, iterate over vertices which are at the current level
    OMP(omp parallel for)
    for (vid_t v = 0; v < graph->vertex_count; v++) {
      if (distance[v] == level) {
        // For all neighbors of v, iterate over paths
        for (eid_t e = graph->vertices[v]; e < graph->vertices[v + 1]; e++) {
          vid_t w = graph->edges[e];
          if (distance[w] == -1) {
            distance[w] = level + 1;
            done = false;
          }
          if (distance[w] == level + 1) {
            __sync_fetch_and_add(&numSPs[w], numSPs[v]);
          }
        }
      }
    }
    level++;
  }
}

/**
 * Implements the backward propagation phase of the Betweenness Centrality
 * Algorithm described in Chapter 2 of GPU Computing Gems. Utilized by:
 * error_t betweenness_cpu(const graph_t* graph, weight_t** centrality_score)
 * @param[in] graph the graph for which the centrality measure is calculated
 * @param[in] source the source node for the shortest paths
 * @param[in] level the shared level variable between backward and forward
 *            propagations
 * @param[in] numSPs an array which counts the number of shortest paths in
 *            which each node is involved 
 * @param[in] distance an array which stores the distance of the shortest
 *            path for each node
 * @param[in] delta an array of the dependencies for each node, which are used
 *            to compute the betweenness centrality measure
 * @param[out] betweenness_centrality the output list which contains the
 *             betweenness centrality values computed for each node
 * @return void
 */
PRIVATE void betweenness_cpu_backward_propagation(const graph_t* graph,
                                          vid_t source, int64_t& level, 
                                          uint32_t* numSPs, int32_t* distance,
                                          weight_t* delta, 
                                          weight_t* betweenness_centrality) {
  // Set deltas to 0 for every input node
  memset(delta, 0, graph->vertex_count * sizeof(vid_t));
  while (level > 1) {
    level--;
    // In parallel, iterate over vertices which are at the current level
    OMP(omp parallel for)
    for (vid_t v = 0; v < graph->vertex_count; v++) {
      if (distance[v] == level) {
        // For all neighbors of v, iterate over paths
        for (eid_t e = graph->vertices[v]; e < graph->vertices[v + 1]; e++) {
          vid_t w = graph->edges[e];
          if (distance[w] == level + 1) {
            delta[v] = (delta[v] + ((((weight_t)numSPs[v]) /
                       ((weight_t)numSPs[w]))*(delta[w] + 1)));
          }
        }
        // Add the dependency to the BC sum
        betweenness_centrality[v] = betweenness_centrality[v] + delta[v];
      }
    }
  }
}

/**
 * Parallel CPU implementation of  Bewteenness Centrality algorithm described
 * in Chapter 2 of GPU Computing Gems (Algorithm 1 - Sequential BC Computation)
 * @param[in] graph the graph for which the centrality measure is calculated
 * @param[out] centrality_score the output list of betweenness centrality scores
 *             per vertex
 * @return generic success or failure
 */
error_t betweenness_cpu(const graph_t* graph, weight_t** centrality_score) {
  // Sanity check on input
  bool finished = true;
  error_t rc = check_special_cases(graph, &finished, centrality_score);
  if (finished) return rc;

  // Allocate memory for the shortest paths problem
  int32_t* distance = (int32_t*)malloc(graph->vertex_count * sizeof(int32_t));
  uint32_t* numSPs = (uint32_t*)malloc(graph->vertex_count * sizeof(uint32_t));
  weight_t* delta = (weight_t*)malloc(graph->vertex_count * sizeof(weight_t));
  weight_t* betweenness_centrality = (weight_t*)mem_alloc(graph->vertex_count
                                                          * sizeof(weight_t));

  // Initialization stage
  // Set BC(v) to 0 for every input node
  memset(betweenness_centrality, 0, graph->vertex_count * sizeof(vid_t));

  // Main loop - iterate over every node and perform both forward propagation
  // and backward propagation for that node, which in turn computes the
  // Betweenness Centrality, as described by the reference algorithm
  for (vid_t source = 0; source < graph->vertex_count; source++) {
    // Initialize variable to keep track of level
    int64_t level = 0;

    // Perform the forward propagation phase for this source node
    betweenness_cpu_forward_propagation(graph, source, level, numSPs, distance);

    // Perform the backward propagation phase for this source node
    betweenness_cpu_backward_propagation(graph, source, level, numSPs, distance,
                                         delta, betweenness_centrality);
  }

  // Cleanup the allocated memory
  free(numSPs);
  free(distance);
  free(delta);

  // Return the centrality
  *centrality_score = betweenness_centrality;
  return SUCCESS;
}

