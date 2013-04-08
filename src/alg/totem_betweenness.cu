/**
 *  Defines Betweenness Centrality functions for both CPU and GPU.
 *
 *  Created on: 2012-05-24
 *  Author: Greg Redekop
 */

// totem includes
#include "totem_alg.h"
#include "totem_centrality.h"

/**
 * Allocates and initializes memory on the GPU for the successors implementation
 * of betweenness centrality.
 */
PRIVATE
error_t initialize_succs_gpu(const graph_t* graph, uint64_t vertex_count,
                             graph_t** graph_d, vid_t** sigma_d,
                             int32_t** dists_d, vid_t** succ_d,
                             uint32_t** succ_count_d, vid_t** stack_d,
                             uint32_t** stack_count_d, score_t** delta_d,
                             bool** finished_d,
                             score_t** betweenness_centrality_d) {
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
  CHK_CU_SUCCESS(cudaMalloc((void**)delta_d, vertex_count * sizeof(score_t)),
                 err_free_stack_count_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)finished_d, sizeof(bool)),
                 err_free_delta_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)betweenness_centrality_d, vertex_count
                            * sizeof(score_t)), err_free_finished_d);

  // Setup initial parameters
  CHK_CU_SUCCESS(cudaMemset(*betweenness_centrality_d, (score_t)0.0,
                            vertex_count * sizeof(score_t)), err_free_all);
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
                             score_t** delta_d, bool** finished_d,
                             score_t** betweenness_centrality_d) {
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
  CHK_CU_SUCCESS(cudaMalloc((void**)delta_d, vertex_count * sizeof(score_t)),
                 err_free_dist_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)finished_d, sizeof(bool)),
                 err_free_delta_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)betweenness_centrality_d,
                             graph->vertex_count * sizeof(score_t)),
                 err_free_finished_d);

  // Setup initial parameters
  CHK_CU_SUCCESS(cudaMemcpy(*r_edges_d, r_edges, graph->edge_count
                            * sizeof(vid_t), cudaMemcpyHostToDevice),
                 err_free_all);
  CHK_CU_SUCCESS(cudaMemset(*betweenness_centrality_d, 0, graph->vertex_count
                            * sizeof(score_t)), err_free_all);
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
                           score_t* delta_d, bool* finished_d, 
                           score_t* betweenness_centrality_d,
                           score_t* betweenness_centrality) {
  // Copy back the centrality scores
  CHK_CU_SUCCESS(cudaMemcpy(betweenness_centrality, betweenness_centrality_d,
                            graph_d->vertex_count * sizeof(score_t),
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
                           vid_t* sigma_d, int32_t* dist_d, score_t* delta_d,
                           bool* finished_d, score_t* betweenness_centrality_d,
                           score_t* betweenness_centrality) {
  // Copy back the centrality scores
  CHK_CU_SUCCESS(cudaMemcpy(betweenness_centrality, betweenness_centrality_d,
                            graph_d->vertex_count * sizeof(score_t),
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
                              score_t* delta) {
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
                          uint32_t* succ_count, score_t* delta, 
                          score_t* betweenness_centrality) {
  const vid_t thread_id = THREAD_GLOBAL_INDEX;

  if (thread_id < stack_count[phase]) {
    vid_t w = stack[graph.vertex_count * phase + thread_id];
    score_t dsw = 0.0;
    score_t sw = sigma[w];
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
                                int32_t* dists, score_t* delta,
                                score_t* betweenness_centrality) {
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
                                   score_t* betweenness_centrality) {
  // Sanity check on input
  bool finished = true;
  error_t rc = betweenness_check_special_cases(graph, &finished, 
                                               betweenness_centrality);
  if (finished) return rc;

  // Allocate memory and initialize state on the GPU
  graph_t* graph_d;
  vid_t* sigma_d;
  int32_t* dist_d;
  vid_t* succ_d;
  uint32_t* succ_count_d;
  vid_t* stack_d;
  uint32_t* stack_count_d;
  score_t* delta_d;
  bool* finished_d;
  score_t* betweenness_centrality_d;

  // Initialization stage
  CHK_SUCCESS(initialize_succs_gpu(graph, graph->vertex_count, &graph_d,
                                   &sigma_d, &dist_d, &succ_d, &succ_count_d,
                                   &stack_d, &stack_count_d, &delta_d,
                                   &finished_d, &betweenness_centrality_d),
              err);

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
    CHK_CU_SUCCESS(cudaMemset(delta_d, (score_t)0.0,
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
 err:
  return FAILURE;
}

/**
 * Implements the parallel Brandes betweenness centrality algorithm using
 * predecessor maps as described in "Fast Network Centrality Analysis Using
 * GPUs" [Shi11]
 */
error_t betweenness_unweighted_shi_gpu(const graph_t* graph,
                                       score_t* betweenness_centrality) {
  // Sanity check on input
  bool finished = true;
  error_t rc = betweenness_check_special_cases(graph, &finished, 
                                               betweenness_centrality);
  if (finished) return rc;

  // Construct the reverse edges list (graph->edges is a list of destination
  // vertices, r_edges is a list of source vertices, indexed by edge id)
  vid_t* r_edges;
  totem_malloc(graph->edge_count * sizeof(vid_t), 
               TOTEM_MEM_HOST_PINNED, (void**)&r_edges);
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
  score_t* delta_d;
  bool* finished_d;
  score_t* betweenness_centrality_d;


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
                              * sizeof(score_t)), err_free_all);
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
  totem_free(r_edges, TOTEM_MEM_HOST_PINNED);

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
  totem_free(r_edges, TOTEM_MEM_HOST_PINNED);
  return FAILURE;
}

/**
 * Implements the parallel Brandes betweenness centrality algorithm using a
 * successor stack, as described in "A Faster Parallel Algorithm and Efficient
 * Multithreaded Implementations for Evaluating Betweenness Centrality on
 * Massive Datasets" [Madduri09]
 */
error_t betweenness_unweighted_cpu(const graph_t* graph, 
                                   score_t* betweenness_centrality) {
  // Sanity check on input
  bool finished = true;
  error_t rc = betweenness_check_special_cases(graph, &finished, 
                                               betweenness_centrality);
  if (finished) return rc;

  // Allocate memory for the shortest paths problem
  uint32_t* sigma = 
    (uint32_t*)malloc(graph->vertex_count * sizeof(uint32_t));
  int32_t* dist = (int32_t*)malloc(graph->vertex_count * sizeof(int32_t));
  vid_t* succ = (vid_t*)malloc(graph->edge_count * sizeof(vid_t));
  vid_t* succ_count = (vid_t*)malloc(graph->vertex_count * sizeof(vid_t));
  vid_t* stack = (vid_t*)malloc(graph->vertex_count * graph->vertex_count
                                 * sizeof(vid_t));
  vid_t* stack_count = (vid_t*)malloc(graph->vertex_count * sizeof(vid_t));
  score_t* delta =
    (score_t*)malloc(graph->vertex_count * sizeof(score_t));
  int64_t phase = 0;

  // Initialization stage
  OMP(omp parallel for)
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    betweenness_centrality[v] = (score_t)0.0;
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
    memset(delta, (score_t)0.0, graph->vertex_count * sizeof(vid_t));
    phase--;
    while (phase > 0) {
      OMP(omp parallel for)
      for (vid_t p = 0; p < stack_count[phase]; p++) {
        vid_t w = stack[graph->vertex_count * phase + p];
        score_t dsw = 0.0;
        score_t sw = sigma[w];
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

  // Cleanup phase
  free(sigma);
  free(dist);
  free(delta);
  free(stack_count);
  free(succ);
  free(succ_count);
  free(stack);
  return SUCCESS;
}

/**
 * Implements the forward propagation phase of the Betweenness Centrality
 * Algorithm described in Chapter 2 of GPU Computing Gems
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
inline PRIVATE 
void betweenness_cpu_forward_propagation(const graph_t* graph, 
                                         vid_t source, cost_t& level,
                                         uint32_t* numSPs, cost_t* distance) {
  // Initialize the shortest path count to 0 and distance to infinity given
  // this source node
  totem_memset(numSPs, (uint32_t)0, graph->vertex_count, TOTEM_MEM_HOST);
  totem_memset(distance, (cost_t)INF_COST, graph->vertex_count, TOTEM_MEM_HOST);
  // Set the distance from source to itself to 0
  distance[source] = 0;
  // Set the shortest path count to 1 (from source to itself)
  numSPs[source] = 1;

  bool done = false;
  while (!done) {
    done = true;
    // In parallel, iterate over vertices which are at the current level
    OMP(omp parallel for schedule(runtime) reduction(& : done))
    for (vid_t v = 0; v < graph->vertex_count; v++) {
      if (distance[v] == level) {
        // For all neighbors of v, iterate over paths
        for (eid_t e = graph->vertices[v]; e < graph->vertices[v + 1]; e++) {
          vid_t w = graph->edges[e];
          if (distance[w] == INF_COST) {
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
 * Algorithm described in Chapter 2 of GPU Computing Gems
 * @param[in] graph the graph for which the centrality measure is calculated
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
inline PRIVATE 
void betweenness_cpu_backward_propagation(const graph_t* graph,
                                          cost_t& level, uint32_t* numSPs, 
                                          cost_t* distance, score_t* delta, 
                                          score_t* betweenness_centrality) {
  // Set deltas to 0 for every input node
  memset(delta, 0, graph->vertex_count * sizeof(vid_t));
  while (level > 1) {
    level--;
    // In parallel, iterate over vertices which are at the current level
    OMP(omp parallel for  schedule(runtime))
    for (vid_t v = 0; v < graph->vertex_count; v++) {
      if (distance[v] == level) {
        // For all neighbors of v, iterate over paths
        for (eid_t e = graph->vertices[v]; e < graph->vertices[v + 1]; e++) {
          vid_t w = graph->edges[e];
          if (distance[w] == level + 1) {
            delta[v] = (delta[v] + ((((score_t)numSPs[v]) /
                       ((score_t)numSPs[w]))*(delta[w] + 1)));
          }
        }
        // Add the dependency to the BC sum
        betweenness_centrality[v] = betweenness_centrality[v] + delta[v];
      }
    }
  }
}

/**
 * Implements the core functionality for computing Betweenness Centrality
 * @param[in] graph the graph for which the centrality measure is calculated
 * @param[in] source the source node for the shortest paths
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
inline PRIVATE void betweenness_cpu_core(const graph_t* graph, vid_t source, 
                                         uint32_t* numSPs, cost_t* distance, 
                                         score_t* delta, 
                                         score_t* betweenness_score) {
  // Initialize variable to keep track of level
  cost_t level = 0;
  // Perform the forward propagation phase for this source node
  betweenness_cpu_forward_propagation(graph, source, level, numSPs, distance);
  // Perform the backward propagation phase for this source node
  betweenness_cpu_backward_propagation(graph, level, numSPs, distance, delta,
                                       betweenness_score);
}

/**
 * Parallel CPU implementation of  Bewteenness Centrality algorithm described
 * in Chapter 2 of GPU Computing Gems (Algorithm 1 - Sequential BC Computation)
 * @param[in] graph the graph for which the centrality measure is calculated
 * @param[in] epsilon determines how precise the results of the algorithm will
 *            be, and thus also how long it will take to compute
 * @param[out] betweenness_score the output list of betweenness centrality
 *             scores per vertex
 * @return generic success or failure
 */
error_t betweenness_cpu(const graph_t* graph, double epsilon,
                        score_t* betweenness_score) {
  // Sanity check on input
  bool finished = true;
  error_t rc = betweenness_check_special_cases(graph, &finished, 
                                               betweenness_score);
  if (finished) return rc;

  // Allocate memory for the shortest paths problem
  cost_t* distance = (cost_t*)malloc(graph->vertex_count * sizeof(cost_t));
  uint32_t* numSPs = (uint32_t*)malloc(graph->vertex_count * sizeof(uint32_t));
  score_t* delta = (score_t*)malloc(graph->vertex_count * sizeof(score_t));

  // Initialization stage
  // Set BC(v) to 0 for every input node
  memset(betweenness_score, 0, graph->vertex_count * sizeof(vid_t));

 // determine whether we will compute exact or approximate BC values
  if (epsilon == CENTRALITY_EXACT) {
    // Compute exact values for Betweenness Centrality
    for (vid_t source = 0; source < graph->vertex_count; source++) { 
      // Perform forward and backward propagation with source node
      betweenness_cpu_core(graph, source, numSPs, distance, delta,
                           betweenness_score);  
    }
  } else {
    // Compute approximate values based on the value of epsilon provided
    // Select a subset of source nodes to make the computation faster
    int num_samples = centrality_get_number_sample_nodes(graph->vertex_count,
                                                         epsilon);
    // Populate the array of indices to sample
    vid_t* sample_nodes = centrality_select_sampling_nodes(graph,
                                                           num_samples);
 
    for (int source_index = 0; source_index < num_samples; source_index++) {
      // Get the next sample node in the array to use as a source
      vid_t source = sample_nodes[source_index];
      // Perform forward and backward propagation with source node
      betweenness_cpu_core(graph, source, numSPs, distance, delta,
                           betweenness_score);   
    }
    
    // Scale the computed Betweenness Centrality metrics since they were
    // computed using a subset of the total nodes within the graph
    // The scaling value is: (Total Number of Nodes / Subset of Nodes Used)
    OMP(omp parallel for) 
    for (vid_t v = 0; v < graph->vertex_count; v++) {
      betweenness_score[v] *= (score_t)((double)(graph->vertex_count) /
                                        (double)num_samples);
    }
 
    // Clean up the allocated memory
    free(sample_nodes);
  }

  // Clean up the allocated memory
  free(numSPs);
  free(distance);
  free(delta);

  return SUCCESS;
}

/**
 * Scales the computed betweenness centrality scores
 * when computing approximate values
 */
__global__
void betweenness_gpu_scale_scores_kernel(vid_t vertex_count, int num_samples,
                                         score_t* betweenness_score_d) { 
  const vid_t vid = THREAD_GLOBAL_INDEX; 
  if (vid < vertex_count) {
    betweenness_score_d[vid] = (score_t)(((double)(vertex_count) / 
                                          num_samples) * 
                                         betweenness_score_d[vid]);
  }
}

/**
 * Initializes the distance and number of shortest paths for each
 * source node before forward propagation, as well as to initialize 
 * the distance and number of shortest paths  for the specified
 * source node.
 */
__global__
void betweenness_gpu_forward_init_kernel(vid_t source, bool* done_d, 
                                         vid_t vertex_count, cost_t* distance_d,
                                         uint32_t* numSPs_d) {
  const vid_t vid = THREAD_GLOBAL_INDEX;
  if (vid >= vertex_count) return;
  if (vid == source) {
    distance_d[vid] = 0;
    numSPs_d[vid] = 1;
    *done_d = false;
  } else {
    distance_d[vid] = INF_COST;
    numSPs_d[vid] = 0;
  } 
}

/**
 * This structure is used by the virtual warp-based implementation. It stores a
 * batch of work. It is allocated on shared memory and is processed by a single
 * virtual warp. Basically it caches the state of the vertices to be processed.
 */
typedef struct {
  eid_t    vertices[VWARP_BATCH_SIZE + 1];
  cost_t   distance[VWARP_BATCH_SIZE];
  uint32_t numSPs[VWARP_BATCH_SIZE];
} batch_mem_t;

/**
 * The neighbors forward propagation processing function. This function sets 
 * the level of the neighbors' vertex to one level more than the parent vertex.
 * The assumption is that the threads of a warp invoke this function to process
 * the warp's batch of work. In each iteration of the for loop, each thread 
 * processes a neighbor. For example, thread 0 in the warp processes neighbors 
 * at indices 0, VWARP_WARP_SIZE, (2 * VWARP_WARP_SIZE) etc. in the edges array,
 * while thread 1 in the warp processes neighbors 1, (1 + VWARP_WARP_SIZE),
 * (1 + 2 * VWARP_WARP_SIZE) and so on.
*/
__device__
void forward_process_neighbors(vid_t warp_offset, vid_t* nbrs, vid_t nbr_count, 
                               uint32_t my_numSPs, uint32_t* numSPs_d,
                               cost_t* distance_d, cost_t level, bool* done_d) {
  for(vid_t i = warp_offset; i < nbr_count; i += VWARP_WARP_SIZE) {
    vid_t nbr = nbrs[i];
    if (distance_d[nbr] == INF_COST) {
      distance_d[nbr] = level + 1;
      *done_d = false;
    }
    if (distance_d[nbr] == level + 1) {
      atomicAdd(&numSPs_d[nbr], my_numSPs);
    }
  }
}

/**
 * Performs forward propagation
 */
__global__
void betweenness_gpu_forward_kernel(const graph_t graph_d, bool* done_d,
                                    cost_t level, uint32_t* numSPs_d, 
                                    cost_t* distance_d, uint32_t thread_count) {
  if (THREAD_GLOBAL_INDEX >= thread_count) return;
  vid_t warp_offset = THREAD_GLOBAL_INDEX % VWARP_WARP_SIZE;
  vid_t warp_id     = THREAD_GLOBAL_INDEX / VWARP_WARP_SIZE;

  __shared__ batch_mem_t batch_s[(MAX_THREADS_PER_BLOCK / VWARP_WARP_SIZE)];
  batch_mem_t* vwarp_batch_s = &batch_s[THREAD_GRID_INDEX / VWARP_WARP_SIZE];
  vid_t base_v = warp_id * VWARP_BATCH_SIZE;
  vwarp_memcpy(vwarp_batch_s->vertices, &(graph_d.vertices[base_v]), 
               VWARP_BATCH_SIZE + 1, warp_offset);
  vwarp_memcpy(vwarp_batch_s->distance, &distance_d[base_v], VWARP_BATCH_SIZE, 
               warp_offset);
  vwarp_memcpy(vwarp_batch_s->numSPs, &numSPs_d[base_v], VWARP_BATCH_SIZE,
               warp_offset);

  // iterate over my work
  for(vid_t v = 0; v < VWARP_BATCH_SIZE; v++) {
    if (vwarp_batch_s->distance[v] == level) {
      vid_t* nbrs = &(graph_d.edges[vwarp_batch_s->vertices[v]]);
      vid_t nbr_count = vwarp_batch_s->vertices[v + 1] - 
        vwarp_batch_s->vertices[v];
      forward_process_neighbors(warp_offset, nbrs, nbr_count, 
                                vwarp_batch_s->numSPs[v], numSPs_d, 
                                distance_d, level, done_d);
    }
  }
}

/**
 * The neighbors backward propagation processing function. This function 
 * computes the delta of a vertex.
 */
__device__
void backward_process_neighbors(vid_t warp_offset, vid_t* nbrs, vid_t nbr_count,
                                uint32_t my_numSPs, score_t* vwarp_delta_s, 
                                uint32_t* numSPs_d, cost_t* distance_d, 
                                score_t* delta_d, cost_t level,
                                score_t* my_delta_d, score_t* my_bc_d) {
  vwarp_delta_s[warp_offset] = 0;
  for(vid_t i = warp_offset; i < nbr_count; i += VWARP_WARP_SIZE) {
    vid_t nbr = nbrs[i];
    if (distance_d[nbr] == level + 1) {
      // Compute an intermediary delta value in shared memory
      vwarp_delta_s[warp_offset] += 
        (((score_t)my_numSPs) / ((score_t)numSPs_d[nbr])) * (delta_d[nbr] + 1);
    }
  }

  // Only one thread in the warp aggregates the final value of delta
  if (warp_offset == 0) {
    score_t delta = 0;
    for (vid_t i = 0; i < VWARP_WARP_SIZE; i++) {
      delta += vwarp_delta_s[i];
    }
    // Add the dependency to the BC sum
    if (delta) {
      *my_delta_d = delta;
      *my_bc_d += delta;
    }
  }
}

/**
 * Performs backward propagation
 */
__global__
void betweenness_gpu_backward_kernel(const graph_t graph_d, cost_t level, 
                                     uint32_t* numSPs_d, cost_t* distance_d,
                                     score_t* delta_d, 
                                     score_t* betweenness_scores_d, 
                                     uint32_t thread_count) {
  if (THREAD_GLOBAL_INDEX >= thread_count) return;
  vid_t warp_offset = THREAD_GLOBAL_INDEX % VWARP_WARP_SIZE;
  vid_t warp_id     = THREAD_GLOBAL_INDEX / VWARP_WARP_SIZE;

  // Each warp has a single entry in the following shared memory array.
  // The entry corresponds to a batch of work which will be processed
  // in parallel by a warp of threads.
  __shared__ batch_mem_t batch_s[(MAX_THREADS_PER_BLOCK / VWARP_WARP_SIZE)];

  // Get a reference to the batch of work of the warp this thread belongs to
  batch_mem_t* vwarp_batch_s = &batch_s[THREAD_GRID_INDEX / VWARP_WARP_SIZE];

  // Calculate the starting vertex of the batch
  vid_t base_v = warp_id * VWARP_BATCH_SIZE;

  // Cache the state of my warp's batch in the shared memory space
  vwarp_memcpy(vwarp_batch_s->vertices, &(graph_d.vertices[base_v]),
               VWARP_BATCH_SIZE + 1, warp_offset);
  vwarp_memcpy(vwarp_batch_s->distance, &distance_d[base_v], VWARP_BATCH_SIZE,
               warp_offset);
  vwarp_memcpy(vwarp_batch_s->numSPs, &numSPs_d[base_v], VWARP_BATCH_SIZE, 
               warp_offset);

  // Each thread in every warp has an entry in the following array which will be
  // used to calcule intermediary delta values in shared memory
  __shared__ score_t delta_s[MAX_THREADS_PER_BLOCK];

  // Get a reference to the entry of the first thread in the warp. This will be
  // indexed later using warp_offset
  int index = THREAD_GRID_INDEX / VWARP_WARP_SIZE;
  score_t* vwarp_delta_s = &delta_s[index * VWARP_WARP_SIZE];

  // Iterate over the warp's batch of work
  for(vid_t v = 0; v < VWARP_BATCH_SIZE; v++) {
    if (vwarp_batch_s->distance[v] == level) {
      vid_t* nbrs = &(graph_d.edges[vwarp_batch_s->vertices[v]]);
      vid_t nbr_count = vwarp_batch_s->vertices[v + 1] - 
        vwarp_batch_s->vertices[v];
      backward_process_neighbors(warp_offset, nbrs, nbr_count, 
                                 vwarp_batch_s->numSPs[v], vwarp_delta_s,
                                 numSPs_d, distance_d, delta_d, level, 
                                 &delta_d[base_v + v], 
                                 &betweenness_scores_d[base_v + v]);
    }
  }
}

/**
 * Implements the core functionality for computing Betweenness Centrality
 * @param[in] graph the graph for which the centrality measure is calculated
 * @param[in] source the source node for the shortest paths
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
inline PRIVATE 
error_t betweenness_gpu_core(graph_t* graph_d, vid_t vertex_count, bool* done_d,
                             vid_t source, uint32_t* numSPs_d, 
                             cost_t* distance_d, score_t* delta_d, 
                             score_t* betweenness_scores_d) {
  // Initialize variables for both forward and backward propagation
  dim3 blocks;
  dim3 threads_per_block;
  bool done = false;
  cost_t level = 0;

  // FORWARD PROPAGATION PHASE
  // Initialize the shortest path count to 0 and distance to infinity given
  // this source node, and also set the distance from source to itself to 0
  // and set the shortest path count to 1 (from source to itself), along 
  // with setting done_d to false
  vid_t state_length = VWARP_BATCH_SIZE * 
    VWARP_BATCH_COUNT(vertex_count);
  KERNEL_CONFIGURE(vertex_count, blocks, threads_per_block);
  betweenness_gpu_forward_init_kernel<<<blocks, threads_per_block>>>
    (source, done_d, state_length, distance_d, numSPs_d);
  CHK_CU_SUCCESS(cudaDeviceSynchronize(), err);
  vid_t thread_count;
  thread_count = VWARP_WARP_SIZE * VWARP_BATCH_COUNT(vertex_count);
  KERNEL_CONFIGURE(thread_count, blocks, threads_per_block);
  while (!done) {
    CHK_CU_SUCCESS(cudaMemset(done_d, true, sizeof(bool)), err);
    // In parallel, iterate over vertices which are at the current level
    betweenness_gpu_forward_kernel<<<blocks, threads_per_block>>>
      (*graph_d, done_d, level, numSPs_d, distance_d, thread_count);
    CHK_CU_SUCCESS(cudaMemcpy(&done, done_d, sizeof(bool), 
                              cudaMemcpyDeviceToHost), err);
    level++;
  }

  // BACKWARD PROPAGATION PHASE
  // Set deltas to 0 for every input node
  CHK_CU_SUCCESS(cudaMemset(delta_d, 0, state_length * sizeof(score_t)), err);
  while (level > 1) {
    level--;
    CHK_CU_SUCCESS(cudaDeviceSynchronize(), err);
    // In parallel, iterate over vertices which are at the current level
    betweenness_gpu_backward_kernel<<<blocks, threads_per_block>>>
      (*graph_d, level, numSPs_d, distance_d, delta_d, betweenness_scores_d, 
       thread_count);
  }

  return SUCCESS;

 err:
  return FAILURE;
}

/**
 * Allocates and initializes memory on the GPU for betweenness_gpu
 */
PRIVATE
error_t initialize_betweenness_gpu(const graph_t* graph, graph_t** graph_d, 
                                   cost_t** distance_d, uint32_t** numSPs_d, 
                                   score_t** delta_d, bool** done_d, 
                                   score_t** betweenness_scores_d) {
  vid_t state_length = VWARP_BATCH_SIZE * 
    VWARP_BATCH_COUNT(graph->vertex_count);
  // Allocate space on GPU
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);
  CHK_CU_SUCCESS(cudaMalloc((void**)distance_d, state_length * sizeof(cost_t)),
                 err_free_graph_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)numSPs_d, state_length * sizeof(uint32_t)),
                 err_free_distance_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)delta_d, state_length * sizeof(score_t)),
                 err_free_numSPs_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)done_d, sizeof(bool)), err_free_delta_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)betweenness_scores_d, state_length
                            * sizeof(score_t)), err_free_done_d);

  // Setup initial parameters
  CHK_CU_SUCCESS(cudaMemset(*betweenness_scores_d, (score_t)0.0,
                            state_length * sizeof(score_t)), err_free_all);
  cudaFuncSetCacheConfig(betweenness_gpu_forward_kernel, 
                         cudaFuncCachePreferShared);
  return SUCCESS;

  // Failure cases for freeing memory
 err_free_all:
  cudaFree(betweenness_scores_d);
 err_free_done_d:
  cudaFree(done_d);
 err_free_delta_d:
  cudaFree(delta_d);
 err_free_numSPs_d:
  cudaFree(numSPs_d);
 err_free_distance_d:
  cudaFree(distance_d);
 err_free_graph_d:
  graph_finalize_device(*graph_d);
 err:
  return FAILURE;
}

/**
 * GPU implementation of  Bewteenness Centrality algorithm described in
 * Chapter 2 of GPU Computing Gems (Algorithm 1 - Sequential BC Computation)
 * @param[in] graph the graph for which the centrality measure is calculated
 * @param[in] epsilon determines how precise the results of the algorithm will
 *            be, and thus also how long it will take to compute
 * @param[out] betweenness_score the output list of betweenness centrality
 *             scores per vertex
 * @return generic success or failure
 */
error_t betweenness_gpu(const graph_t* graph, double epsilon,
                        score_t* betweenness_score) {
  // Sanity check on input
  bool finished = true;
  error_t rc = betweenness_check_special_cases(graph, &finished, 
                                               betweenness_score);
  if (finished) return rc;

  // Initialization stage
  // Set BC(v) to 0 for every input node
  memset(betweenness_score, 0, graph->vertex_count * sizeof(score_t));

  // Create pointers for use with the GPU's memory
  graph_t*  graph_d;
  cost_t*   distance_d;
  uint32_t* numSPs_d;
  score_t*  delta_d;
  bool*     done_d;
  score_t*  betweenness_scores_d;

  // Initialization stage
  CHK_SUCCESS(initialize_betweenness_gpu(graph, &graph_d, &distance_d, 
                                         &numSPs_d, &delta_d, &done_d, 
                                         &betweenness_scores_d), err);
  // determine whether we will compute exact or approximate BC values
  if (epsilon == CENTRALITY_EXACT) {
    // Compute exact values for Betweenness Centrality
    for (vid_t source = 0; source < graph->vertex_count; source++) { 
      // Perform forward and backward propagation with source node
      CHK_SUCCESS(betweenness_gpu_core(graph_d, graph->vertex_count, done_d, 
                                       source, numSPs_d, distance_d, delta_d, 
                                       betweenness_scores_d), err_free_all);
    }
  } else {
    // Compute approximate values based on the value of epsilon provided
    // Select a subset of source nodes to make the computation faster
    int num_samples = centrality_get_number_sample_nodes(graph->vertex_count,
                                                         epsilon);
    // Populate the array of indices to sample
    vid_t* sample_nodes = centrality_select_sampling_nodes(graph,
                                                           num_samples);
 
    for (vid_t source_index = 0; source_index < num_samples; source_index++) {
      // Get the next sample node in the array to use as a source
      vid_t source = sample_nodes[source_index];
      // Perform forward and backward propagation with source node
      CHK_SUCCESS(betweenness_gpu_core(graph_d, graph->vertex_count, done_d, 
                                       source, numSPs_d, distance_d, delta_d,
                                       betweenness_scores_d), err_free_all);
    }
    
    // Scale the computed Betweenness Centrality metrics since they were
    // computed using a subset of the total nodes within the graph
    // The scaling value is: (Total Number of Nodes / Subset of Nodes Used)
    dim3 blocks;
    dim3 threads_per_block;
    KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
    betweenness_gpu_scale_scores_kernel<<<blocks, threads_per_block>>>
      (graph->vertex_count, num_samples, betweenness_scores_d);

    // Clean up the allocated memory
    free(sample_nodes);
  }

  // Copy the calculated betweenness centrality scores back to host memory
  CHK_CU_SUCCESS(cudaMemcpy(betweenness_score, betweenness_scores_d,
                            graph->vertex_count * sizeof(score_t),
                            cudaMemcpyDeviceToHost), err_free_all);

  // Clean up the memory allocated on the GPU
  graph_finalize_device(graph_d);
  cudaFree(distance_d);
  cudaFree(numSPs_d);
  cudaFree(delta_d);
  cudaFree(done_d);
  cudaFree(betweenness_scores_d);
  return SUCCESS;

 err_free_all:
  graph_finalize_device(graph_d);
  cudaFree(distance_d);
  cudaFree(numSPs_d);
  cudaFree(delta_d);
  cudaFree(done_d);
  cudaFree(betweenness_scores_d);
 err:
  return FAILURE;
}
