/**
 * Implements a simplified version of the PageRank algorithm based on the
 * algorithm described by [Malewicz2010]
 * G. Malewicz, M. H. Austern, A. J. C. Bik, J. C. Dehnert, I. Horn, N. Leiser,
 * and G. Czajkowski. Pregel: a system for large-scale graph processing. In
 * Proceedings of the 28th ACM symposium on Principles of distributed computing,
 * PODC 09, page 6, New York, NY, USA, 2009. ACM.
 *
 * Algorithm description [Malewicz2010]:
 * The graph is initialized so that in round 0, the PageRank of each vertex is
 * set to 1 / vertex_count. For PAGE_RANK_ROUNDS rounds, each vertex sends along
 * each outgoing edge its tentative PageRank divided by the number of outgoing
 * edges. The tentative PageRank is calculated as follows: the vertex sums up
 * the values arriving into sum and sets its own tentative PageRank to
 * ((1 - DAMPING_FACTOR) / vertex_count + DAMPING_FACTOR * sum).
 *
 *  Created on: 2011-07-09
 *  Author: Abdullah Gharaibeh
 */

#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

// TODO(abdullah): The following two values should be a parameter in the entry
//                 function to enable more flexibility and experimentation. This
//                 however increases register usage and may affect performance
/**
 * Used to define the number of rounds: a static convergance condition
 * for PageRank
 */
#define PAGE_RANK_ROUNDS 30

/**
 * A probability used in the PageRank algorithm. A probability that models the
 * behavior of the random surfer when she moves from one page to another
 * without following the links on the current page.
 */
#define DAMPING_FACTOR 0.85

/**
 * Produce the sum of the neighbors' ranks. Each vertex atomically 
 * adds its value to the mailbox of the destination neighbor vertex.
 */
__global__
void page_rank_phase1_kernel(graph_t graph, float* rank, float* mailbox) {
  // get the thread's linear index
  id_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;

  float my_rank = rank[vertex_id];
  for (uint64_t i = graph.vertices[vertex_id];
       i < graph.vertices[vertex_id + 1]; i++) {
    const id_t neighbor_id = graph.edges[i];
    atomicAdd(&(mailbox[neighbor_id]), (float)my_rank);
  }
}

/**
 * Produce the rank of each vertex. The sum of ranks coming from the incoming
 * edges is stored in the mailbox of the vertex.
 */
__global__
void page_rank_phase2_kernel(graph_t graph, float* rank, float* mailbox) {
  // get the thread's linear index
  id_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;

  // get sum of incoming neighbors' ranks
  float sum = mailbox[vertex_id];
  mailbox[vertex_id] = 0;

  // calculate my normalized rank
  float my_rank =
    ((1 - DAMPING_FACTOR) / graph.vertex_count) + (DAMPING_FACTOR * sum);
  rank[vertex_id] =  my_rank /
    (graph.vertices[vertex_id + 1] - graph.vertices[vertex_id]);
}

/**
 * This kernel is similar to the page_rank_phase2_kernel. The difference is that
 * it does not normalize the rank (by dividing it by the number of neighbors).
 * It is invoked in the end to get the final, un-normalized, rank.
 */
__global__
void page_rank_phase2_final_kernel(graph_t graph, float* rank, float* mailbox) {
  // get the thread's linear index
  id_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;

  // get sum of neighbors' ranks
  float sum = mailbox[vertex_id];

  // calculate my rank
  rank[vertex_id] = 
    ((1 - DAMPING_FACTOR) / graph.vertex_count) + (DAMPING_FACTOR * sum);
}

error_t page_rank_incoming_gpu(graph_t* graph, float** rank) {
  if (graph == NULL) {
    return FAILURE;
  } else if (graph->vertex_count == 0) {
    return FAILURE;
  } else if (graph->vertex_count == 1) {
    *rank = (float*)mem_alloc(sizeof(float));
    (*rank)[0] = (float)1.0;
    return SUCCESS;
  }

  /* had to define them at the beginning to avoid a compilation problem with
     goto-label error handling mechanism */
  dim3 blocks;
  dim3 threads_per_block;

  // will be passed to the kernel
  graph_t* graph_d;
  CHK_SUCCESS(graph_initialize_device(graph, &graph_d), err);

  // allocate mailbox and outbox device buffers
  float *mailbox_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&mailbox_d, graph->vertex_count * 
                            sizeof(float)), err_free_graph_d);
  float *rank_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&rank_d, graph->vertex_count * 
                            sizeof(float)), err_free_mailbox);

  /* set the number of blocks, TODO(abdullah) handle the case when
     vertex_count is larger than number of threads. */
  assert(graph->vertex_count <= MAX_THREAD_COUNT);
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);

  // initialize the rank of each vertex
  // TODO (elizeu, abdullah): A more realistic version of PageRank could be
  //                          easily implemented as follows:
  //                          1. Have a kernel that determines the in-degree of
  //                             each vertex (i.e., the number of occurrences of
  //                             each vertex id in the edges array).
  //                          2. Divide the in-degree by edge_count and set this
  //                             as the initial rank.

  //                          The rationale behind this is that vertices with
  //                          higher in-degree are more likely to be visited by
  //                          the random surfer.
  float initial_value;
  initial_value = 1 / (float)graph->vertex_count;
  memset_device<<<blocks, threads_per_block>>>
    (rank_d, initial_value, graph->vertex_count);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_rank);

  memset_device<<<blocks, threads_per_block>>>
    (mailbox_d, (float)0.0, graph->vertex_count);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_rank);

  for (uint32_t round = 0; round < PAGE_RANK_ROUNDS - 1; round++) {
    // call the kernel
    page_rank_phase1_kernel<<<blocks, threads_per_block>>>
      (*graph_d, rank_d, mailbox_d);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_rank);

    page_rank_phase2_kernel<<<blocks, threads_per_block>>>
      (*graph_d, rank_d, mailbox_d);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_rank);
  }
  // call the kernel
  page_rank_phase1_kernel<<<blocks, threads_per_block>>>
    (*graph_d, rank_d, mailbox_d);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_rank);
  
  page_rank_phase2_final_kernel<<<blocks, threads_per_block>>>
    (*graph_d, rank_d, mailbox_d);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_rank);

  // copy back the final result from the rank
  *rank = (float*)mem_alloc(graph->vertex_count * sizeof(float));
  CHK_CU_SUCCESS(cudaMemcpy(*rank, rank_d, graph->vertex_count * 
                            sizeof(float), cudaMemcpyDeviceToHost), 
                 err_free_all);

  // we are done! set the output and clean up
  cudaFree(rank_d);
  cudaFree(mailbox_d);
  graph_finalize_device(graph_d);
  return SUCCESS;

  // error handlers
 err_free_all:
 err_free_rank:
  cudaFree(rank_d);
 err_free_mailbox:
  cudaFree(mailbox_d);
 err_free_graph_d:
  graph_finalize_device(graph_d);
 err:
  return FAILURE;
}

error_t page_rank_incoming_cpu(graph_t* graph, float** rank_ret) {
  if (graph == NULL) {
    return FAILURE;
  } else if (graph->vertex_count == 0) {
    return FAILURE;
  } else if (graph->vertex_count == 1) {
    *rank_ret = (float*)mem_alloc(sizeof(float));
    (*rank_ret)[0] = (float)1.0;
    return SUCCESS;
  }

  // allocate buffers
  float* rank    = (float*)mem_alloc(graph->vertex_count * sizeof(float));
  float* mailbox = (float*)mem_alloc(graph->vertex_count * sizeof(float));

  // initialize the rank of each vertex
  float initial_value = 1 / (float)graph->vertex_count;
  for (id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    rank[vertex_id] = initial_value;
    mailbox[vertex_id] = 0;
  }

  for (int round = 0; round < PAGE_RANK_ROUNDS; round++) {
    // iterate over all vertices to calculate the ranks for this round
#ifdef _OPENMP // this is defined if -fopenmp flag is passed to the compiler
#pragma omp parallel for
#endif // _OPENMP
    for(id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      // calculate the sum of all neighbors' rank
      float my_rank = rank[vertex_id];
      for (uint64_t i = graph->vertices[vertex_id];
           i < graph->vertices[vertex_id + 1]; i++) {
        const id_t neighbor_id = graph->edges[i];
        __sync_add_and_fetch_float(&mailbox[neighbor_id], (float)my_rank);
      }
    }

#ifdef _OPENMP // this is defined if -fopenmp flag is passed to the compiler
#pragma omp parallel for
#endif // _OPENMP
    for(id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      // get sum of neighbors' ranks
      float sum = mailbox[vertex_id];
      mailbox[vertex_id] = 0;
      // calculate my rank
      uint64_t neighbors_count =
        graph->vertices[vertex_id + 1] - graph->vertices[vertex_id];
      float my_rank =
        ((1 - DAMPING_FACTOR) / graph->vertex_count) + (DAMPING_FACTOR * sum);
      rank[vertex_id] =
        (round == (PAGE_RANK_ROUNDS - 1)) ? my_rank : my_rank / neighbors_count;
    }
  }

  // we are done! set the output and clean up.
  *rank_ret = rank;
  mem_free(mailbox);
  return SUCCESS;
}
