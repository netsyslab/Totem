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

#include "totem_alg.h"

/**
   This structure is used by virtual warp-based implementation. It stores a
   batch of work. It is typically allocated on shared memory and is processed by
   a single virtual warp.
 */
typedef struct {
  // One is added to make it easy to calculate the number of neighbors of the
  // last vertex. Another one is added to ensure 8 bytes alignment irrespective 
  // whether sizeof(eid_t) is 4 or 8. Alignment is enforced for performance 
  // reasons.
  eid_t  vertices[VWARP_DEFAULT_BATCH_SIZE + 2];
  rank_t rank[VWARP_DEFAULT_BATCH_SIZE];
} vwarp_mem_t;

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
*/
PRIVATE
error_t check_special_cases(const graph_t* graph, rank_t* rank, 
                            bool* finished) {
  *finished = true;
  if (graph == NULL) {
    return FAILURE;
  } else if (graph->vertex_count == 0) {
    return FAILURE;
  } else if (graph->vertex_count == 1) {
    rank[0] = 1.0;
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

/**
 * A common initialization function for GPU implementations. It allocates and
 * initalizes state on the GPU
*/
PRIVATE
error_t initialize_gpu(const graph_t* graph, rank_t* rank_i, vid_t rank_length, 
                       graph_t** graph_d, rank_t **rank_d, rank_t** mailbox_d) {
  totem_mem_t type = TOTEM_MEM_DEVICE;

  // will be passed to the kernel
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);

  // allocate mailbox and outbox device buffers
  CHK_SUCCESS(totem_calloc(graph->vertex_count * sizeof(rank_t), type,
                           (void**)mailbox_d), err_free_graph_d);
  CHK_SUCCESS(totem_malloc(rank_length * sizeof(rank_t), type,
                           (void**)rank_d), err_free_mailbox);

  if (rank_i == NULL) {
    rank_t initial_value = 1 / (rank_t)graph->vertex_count;
    totem_memset(*rank_d, initial_value, rank_length, type);
  } else {
    CHK_CU_SUCCESS(cudaMemcpy(*rank_d, rank_i, rank_length * sizeof(rank_t),
        cudaMemcpyHostToDevice), err_free_all);
  }

  return SUCCESS;

  // error handlers
 err_free_all:
  totem_free(rank_d, type);
 err_free_mailbox:
  totem_free(mailbox_d, type);
 err_free_graph_d:
  graph_finalize_device(*graph_d);
 err:
  return FAILURE;
}

/**
 * A common finalize function for GPU implementations. It allocates the host
 * output buffer, moves the final results from GPU to the host buffers and
 * frees up some resources.
*/
PRIVATE
error_t finalize_gpu(graph_t* graph_d, rank_t* rank_d, rank_t* mailbox_d, 
                     rank_t* rank) {
  // Copy back the final result
  CHK_CU_SUCCESS(cudaMemcpy(rank, rank_d, graph_d->vertex_count *
                            sizeof(rank_t), cudaMemcpyDeviceToHost), err);
  totem_free(rank_d, TOTEM_MEM_DEVICE);
  totem_free(mailbox_d, TOTEM_MEM_DEVICE);
  graph_finalize_device(graph_d);
  return SUCCESS;

 err:
  return FAILURE;
}

/**
 * Phase1 kernel of the original PageRank GPU algorithm (i.e., non-vwarp).
 * Produce the sum of the neighbors' ranks. Each vertex atomically
 * adds its value to the mailbox of the destination neighbor vertex.
 */
__global__
void sum_neighbors_rank_kernel(graph_t graph, rank_t* rank, rank_t* mailbox) {
  // get the thread's linear index
  vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;

  rank_t my_rank = rank[vertex_id];
  for (eid_t i = graph.vertices[vertex_id];
       i < graph.vertices[vertex_id + 1]; i++) {
    const eid_t neighbor_id = graph.edges[i];
    atomicAdd(&(mailbox[neighbor_id]), (rank_t)my_rank);
  }
}

/**
 * Phase2 kernel of the original PageRank GPU algorithm (i.e., non-vwarp).
 * Produce the rank of each vertex. The sum of ranks coming from the incoming
 * edges is stored in the mailbox of the vertex.
 */
__global__
void compute_normalized_rank_kernel(graph_t graph, rank_t* rank,
                                    rank_t* mailbox) {
  // get the thread's linear index
  vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  // get sum of incoming neighbors' ranks
  rank_t sum = mailbox[vertex_id];
  mailbox[vertex_id] = 0;
  // calculate my normalized rank
  rank_t my_rank = ((1 - PAGE_RANK_DAMPING_FACTOR) / graph.vertex_count) + 
    (PAGE_RANK_DAMPING_FACTOR * sum);
  rank[vertex_id] =  my_rank /
    (graph.vertices[vertex_id + 1] - graph.vertices[vertex_id]);
}

/**
 * Phase2 final kernel of the original PageRank GPU algorithm (i.e., non-vwarp).
 * This kernel is similar to the compute_normalized_rank_kernel. The difference
 * is that it does not normalize the rank (by dividing it by the number of 
 * neighbors). It is invoked in the end to get the final, un-normalized, rank.
 */
__global__
void compute_unnormalized_rank_kernel(graph_t graph, rank_t* rank, 
                                      rank_t* mailbox) {
  // get the thread's linear index
  vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  // get sum of neighbors' ranks
  rank_t sum = mailbox[vertex_id];
  // calculate my rank
  rank[vertex_id] = ((1 - PAGE_RANK_DAMPING_FACTOR) / graph.vertex_count) + 
    (PAGE_RANK_DAMPING_FACTOR * sum);
}

/**
 * The neighbors processing function. This function adds the a vertex rank to
 * to the mailbox of all neighbors. The assumption is that the threads of a warp
 * invoke this function to process the warp's batch of work. In each iteration
 * of the for loop, each thread processes a neighbor. For example, thread 0 in
 * the warp processes neighbors at indices 0, VWARP_DEFAULT_WARP_WIDTH,
 * (2 * VWARP_DEFAULT_WARP_WIDTH) etc. in the edges array, while thread 1 in 
 * the warp processes neighbors 1, (1 + VWARP_DEFAULT_WARP_WIDTH), 
 * (1 + 2 * VWARP_DEFAULT_WARP_WIDTH) and so on.
*/
__device__ inline
void vwarp_process_neighbors(vid_t warp_offset, vid_t neighbor_count, 
                             vid_t* neighbors, rank_t my_rank, 
                             rank_t* mailbox) {
  for(vid_t i = warp_offset; i < neighbor_count; 
      i += VWARP_DEFAULT_WARP_WIDTH) {
    const vid_t neighbor_id = neighbors[i];
    atomicAdd(&(mailbox[neighbor_id]), my_rank);
  }
}

/**
 * Phase1 kernel of the vwarp PageRank GPU algorithm.
 * Produce the sum of the neighbors' ranks. Each vertex atomically
 * adds its value to the mailbox of the destination neighbor vertex.
 */
__global__
void vwarp_sum_neighbors_rank_kernel(graph_t graph, rank_t* rank, 
                                     rank_t* mailbox, uint32_t thread_count) {
  if (THREAD_GLOBAL_INDEX >= thread_count) return;
  vid_t warp_offset = THREAD_GLOBAL_INDEX % VWARP_DEFAULT_WARP_WIDTH;
  vid_t warp_id     = THREAD_GLOBAL_INDEX / VWARP_DEFAULT_WARP_WIDTH;

  __shared__ vwarp_mem_t shared_memory[MAX_THREADS_PER_BLOCK /
                                       VWARP_DEFAULT_WARP_WIDTH];
  vwarp_mem_t* my_space = &shared_memory[THREAD_BLOCK_INDEX / 
                                         VWARP_DEFAULT_WARP_WIDTH];

  // copy my work to local space
  vid_t v_ = warp_id * VWARP_DEFAULT_BATCH_SIZE;
  vwarp_memcpy(my_space->rank, &rank[v_], 
               VWARP_DEFAULT_BATCH_SIZE, warp_offset);
  vwarp_memcpy(my_space->vertices, &(graph.vertices[v_]), 
               VWARP_DEFAULT_BATCH_SIZE + 1, warp_offset);

  // iterate over my work
  for(vid_t v = 0; v < VWARP_DEFAULT_BATCH_SIZE; v++) {
    vid_t neighbor_count = my_space->vertices[v + 1] - my_space->vertices[v];
    vid_t* neighbors = &(graph.edges[my_space->vertices[v]]);
    vwarp_process_neighbors(warp_offset, neighbor_count, neighbors,
                            my_space->rank[v], mailbox);
  }
}

/**
 * Phase2 kernel of the vwarp PageRank GPU algorithm.
 * Produce the rank of each vertex. The sum of ranks coming from the incoming
 * edges is stored in the mailbox of the vertex.
 */
__global__
void vwarp_compute_normalized_rank_kernel(graph_t graph, rank_t* rank,
                                          rank_t* mailbox) {
  vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  // get sum of incoming neighbors' ranks
  rank_t sum = mailbox[vertex_id];
  mailbox[vertex_id] = 0;
  // calculate my normalized rank
  rank_t my_rank = ((1 - PAGE_RANK_DAMPING_FACTOR) / graph.vertex_count) + 
    (PAGE_RANK_DAMPING_FACTOR * sum);
  rank[vertex_id] =  my_rank /
    (graph.vertices[vertex_id + 1] - graph.vertices[vertex_id]);
}

/**
 * Phase2 final kernel of the vwarp PageRank GPU algorithm. This kernel is 
 * similar to the compute_normalized_rank_kernel. The difference is that it 
 * does not normalize the rank (by dividing it by the number of neighbors). 
 * It is invoked in the end to get the final, un-normalized, rank.
 */
__global__
void vwarp_compute_unnormalized_rank_kernel(graph_t graph, rank_t* rank,
                                            rank_t* mailbox) {
  vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  // get sum of neighbors' ranks
  rank_t sum = mailbox[vertex_id];
  // calculate my rank
  rank[vertex_id] = ((1 - PAGE_RANK_DAMPING_FACTOR) / graph.vertex_count) + 
    (PAGE_RANK_DAMPING_FACTOR * sum);
}

__host__
error_t page_rank_vwarp_gpu(graph_t* graph, rank_t* rank_i, rank_t* rank) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, rank, &finished);
  if (finished) return rc;

  // Allocate and initialize GPU state
  graph_t* graph_d;
  rank_t* rank_d;
  rank_t* mailbox_d;
  vid_t rank_length;
  rank_length = vwarp_default_state_length(graph->vertex_count);
  CHK_SUCCESS(initialize_gpu(graph, rank_i, rank_length, &graph_d, &rank_d,
                             &mailbox_d), err);

  {// Configure the kernels. Setup the number of threads for phase1 and phase2,
  // configure the on-chip memory as shared memory rather than L1 cache
  dim3 blocks1, threads_per_block1, blocks2, threads_per_block2;
  vid_t phase1_thread_count = 
    vwarp_default_thread_count(graph->vertex_count);
  KERNEL_CONFIGURE(phase1_thread_count, blocks1, threads_per_block1);
  KERNEL_CONFIGURE(graph->vertex_count, blocks2, threads_per_block2);
  cudaFuncSetCacheConfig(vwarp_sum_neighbors_rank_kernel,
                         cudaFuncCachePreferShared);

  // Iterate for a specific number of rounds
  for (int round = 0; round < PAGE_RANK_ROUNDS - 1; round++) {
    vwarp_sum_neighbors_rank_kernel<<<blocks1, threads_per_block1>>>
      (*graph_d, rank_d, mailbox_d, phase1_thread_count);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
    vwarp_compute_normalized_rank_kernel<<<blocks2, threads_per_block2>>>
      (*graph_d, rank_d, mailbox_d);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
  }
  // Final round is seprate. It computes an un-normalized final rank
  vwarp_sum_neighbors_rank_kernel<<<blocks1, threads_per_block1>>>
    (*graph_d, rank_d, mailbox_d, phase1_thread_count);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
  vwarp_compute_unnormalized_rank_kernel<<<blocks2, threads_per_block2>>>
    (*graph_d, rank_d, mailbox_d);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
  }

  // Copy the result back from GPU and clean up
  CHK_SUCCESS(finalize_gpu(graph_d, rank_d, mailbox_d, rank), err_free_all);
  return SUCCESS;

  // error handlers
 err_free_all:
  totem_free(rank_d, TOTEM_MEM_DEVICE);
  totem_free(mailbox_d, TOTEM_MEM_DEVICE);
  graph_finalize_device(graph_d);
 err:
  return FAILURE;
}

__host__
error_t page_rank_gpu(graph_t* graph, rank_t* rank_i, rank_t* rank) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, rank, &finished);
  if (finished) return rc;

  // Allocate and initialize GPU state
  graph_t* graph_d;
  rank_t* rank_d;
  rank_t* mailbox_d;
  CHK_SUCCESS(initialize_gpu(graph, rank_i, graph->vertex_count, 
                             &graph_d, &rank_d, &mailbox_d), err);

  {
  dim3 blocks, threads_per_block;
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
  // Iterate for a specific number of rounds
  for (int round = 0; round < PAGE_RANK_ROUNDS - 1; round++) {
    sum_neighbors_rank_kernel<<<blocks, threads_per_block>>>
      (*graph_d, rank_d, mailbox_d);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
    compute_normalized_rank_kernel<<<blocks, threads_per_block>>>
      (*graph_d, rank_d, mailbox_d);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
  }
  // Final round is seprate. It computes an un-normalized final rank
  sum_neighbors_rank_kernel<<<blocks, threads_per_block>>>
    (*graph_d, rank_d, mailbox_d);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
  compute_unnormalized_rank_kernel<<<blocks, threads_per_block>>>
    (*graph_d, rank_d, mailbox_d);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
  }

  // Copy the result back from GPU and clean up
  CHK_SUCCESS(finalize_gpu(graph_d, rank_d, mailbox_d, rank), err_free_all);
  return SUCCESS;

  // error handlers
 err_free_all:
  totem_free(rank_d, TOTEM_MEM_DEVICE);
  totem_free(mailbox_d, TOTEM_MEM_DEVICE);
  graph_finalize_device(graph_d);
 err:
  return FAILURE;
}

error_t page_rank_cpu(graph_t* graph, rank_t* rank_i, rank_t* rank) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, rank, &finished);
  if (finished) return rc;

  // allocate buffers
  rank_t* mailbox;
  CALL_SAFE(totem_malloc(graph->vertex_count * sizeof(rank_t), TOTEM_MEM_HOST, 
                         (void**)&mailbox));

  // initialize the rank of each vertex
  if (rank_i == NULL) {
    rank_t initial_value = 1 / (rank_t)graph->vertex_count;
    for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      rank[vertex_id] = initial_value;
      mailbox[vertex_id] = 0;
    }
  } else {
    for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      rank[vertex_id] = rank_i[vertex_id];
      mailbox[vertex_id] = 0;
    }
  }

  for (int round = 0; round < PAGE_RANK_ROUNDS; round++) {
    // iterate over all vertices to calculate the ranks for this round
    // The "runtime" scheduling clause defer the choice of thread scheduling
    // algorithm to the choice of the client, either via OS environment variable
    // or omp_set_schedule interface.
    OMP(omp parallel for schedule(runtime))
    for(vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      // calculate the sum of all neighbors' rank
      rank_t my_rank = rank[vertex_id];
      for (eid_t i = graph->vertices[vertex_id];
           i < graph->vertices[vertex_id + 1]; i++) {
        const vid_t neighbor_id = graph->edges[i];
        __sync_fetch_and_add_float(&mailbox[neighbor_id], (rank_t)my_rank);
      }
    }

    // The loop has no load balancing issues, hence the choice of dividing
    // the iterations between the threads statically via the static schedule 
    // clause
    OMP(omp parallel for schedule(static))
    for(vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      // get sum of neighbors' ranks
      rank_t sum = mailbox[vertex_id];
      mailbox[vertex_id] = 0;
      // calculate my rank
      vid_t neighbors_count = 
        graph->vertices[vertex_id + 1] - graph->vertices[vertex_id];
      rank_t my_rank = ((1 - PAGE_RANK_DAMPING_FACTOR) / graph->vertex_count) + 
        (PAGE_RANK_DAMPING_FACTOR * sum);
      rank[vertex_id] =
        (round == (PAGE_RANK_ROUNDS - 1)) ? my_rank : my_rank / neighbors_count;
    }
  }

  // we are done! set the output and clean up.
  totem_free(mailbox, TOTEM_MEM_HOST);
  return SUCCESS;
}
