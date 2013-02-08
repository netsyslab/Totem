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
   This structure is used by virtual warp-based implementation. It stores a
   batch of work. It is typically allocated on shared memory and is processed by
   a single virtual warp.
 */
typedef struct {
  // One is added to make it easy to calculate the number of neighbors of the
  // last vertex. Another one is added to ensure 8 bytes alignment irrespective 
  // whether sizeof(eid_t) is 4 or 8. Alignment is enforced for performance 
  // reasons.
  eid_t  vertices[VWARP_BATCH_SIZE + 2];
  float rank[VWARP_BATCH_SIZE];
} vwarp_mem_t;

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
*/
PRIVATE
error_t check_special_cases(const graph_t* graph, float* rank, bool* finished) {
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
error_t initialize_gpu(const graph_t* graph, float* rank_i, vid_t rank_length, 
                       graph_t** graph_d, float **rank_d, float** mailbox_d) {
  /* had to define them at the beginning to avoid a compilation problem with
     goto-label error handling mechanism */
  dim3 blocks;
  dim3 threads_per_block;

  // will be passed to the kernel
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);

  // allocate mailbox and outbox device buffers
  CHK_CU_SUCCESS(cudaMalloc((void**)mailbox_d, graph->vertex_count *
                            sizeof(float)), err_free_graph_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)rank_d, rank_length * sizeof(float)),
                 err_free_mailbox);

  if (rank_i == NULL) {
    float initial_value = 1 / (float)graph->vertex_count;
    KERNEL_CONFIGURE(rank_length, blocks, threads_per_block);
    memset_device<<<blocks, threads_per_block>>>
        (*rank_d, initial_value, rank_length);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
  } else {
    CHK_CU_SUCCESS(cudaMemcpy(*rank_d, rank_i, rank_length * sizeof(float),
        cudaMemcpyHostToDevice), err_free_all);
  }

  memset_device<<<blocks, threads_per_block>>>
    (*mailbox_d, (float)0.0, graph->vertex_count);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);

  return SUCCESS;

  // error handlers
 err_free_all:
  cudaFree(rank_d);
 err_free_mailbox:
  cudaFree(mailbox_d);
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
error_t finalize_gpu(graph_t* graph_d, float* rank_d, float* mailbox_d, 
                     float* rank) {
  // Copy back the final result
  CHK_CU_SUCCESS(cudaMemcpy(rank, rank_d, graph_d->vertex_count *
                            sizeof(float), cudaMemcpyDeviceToHost), err);
  cudaFree(rank_d);
  cudaFree(mailbox_d);
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
void sum_neighbors_rank_kernel(graph_t graph, float* rank, float* mailbox) {
  // get the thread's linear index
  vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;

  float my_rank = rank[vertex_id];
  for (eid_t i = graph.vertices[vertex_id];
       i < graph.vertices[vertex_id + 1]; i++) {
    const eid_t neighbor_id = graph.edges[i];
    atomicAdd(&(mailbox[neighbor_id]), (float)my_rank);
  }
}

/**
 * Phase2 kernel of the original PageRank GPU algorithm (i.e., non-vwarp).
 * Produce the rank of each vertex. The sum of ranks coming from the incoming
 * edges is stored in the mailbox of the vertex.
 */
__global__
void compute_normalized_rank_kernel(graph_t graph, float* rank,
                                    float* mailbox) {
  // get the thread's linear index
  vid_t vertex_id = THREAD_GLOBAL_INDEX;
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
 * Phase2 final kernel of the original PageRank GPU algorithm (i.e., non-vwarp).
 * This kernel is similar to the compute_normalized_rank_kernel. The difference
 * is that it does not normalize the rank (by dividing it by the number of 
 * neighbors). It is invoked in the end to get the final, un-normalized, rank.
 */
__global__
void compute_unnormalized_rank_kernel(graph_t graph, float* rank, 
                                      float* mailbox) {
  // get the thread's linear index
  vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  // get sum of neighbors' ranks
  float sum = mailbox[vertex_id];
  // calculate my rank
  rank[vertex_id] =
    ((1 - DAMPING_FACTOR) / graph.vertex_count) + (DAMPING_FACTOR * sum);
}

/**
 * The neighbors processing function. This function adds the a vertex rank to
 * to the mailbox of all neighbors. The assumption is that the threads of a warp
 * invoke this function to process the warp's batch of work. In each iteration
 * of the for loop, each thread processes a neighbor. For example, thread 0 in
 * the warp processes neighbors at indices 0, VWARP_WARP_SIZE,
 * (2 * VWARP_WARP_SIZE) etc. in the edges array, while thread 1 in the warp
 * processes neighbors 1, (1 + VWARP_WARP_SIZE), (1 + 2 * VWARP_WARP_SIZE) and
 * so on.
*/
__device__ inline
void vwarp_process_neighbors(vid_t warp_offset, vid_t neighbor_count,
                             vid_t* neighbors, float my_rank, float* mailbox) {
  for(vid_t i = warp_offset; i < neighbor_count; i += VWARP_WARP_SIZE) {
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
void vwarp_sum_neighbors_rank_kernel(graph_t graph, float* rank, float* mailbox,
                                     uint32_t thread_count) {
  if (THREAD_GLOBAL_INDEX >= thread_count) return;
  vid_t warp_offset = THREAD_GLOBAL_INDEX % VWARP_WARP_SIZE;
  vid_t warp_id     = THREAD_GLOBAL_INDEX / VWARP_WARP_SIZE;

  __shared__ vwarp_mem_t shared_memory[(MAX_THREADS_PER_BLOCK /
                                        VWARP_WARP_SIZE)];
  vwarp_mem_t* my_space = shared_memory + (THREAD_GRID_INDEX / VWARP_WARP_SIZE);

  // copy my work to local space
  vid_t v_ = warp_id * VWARP_BATCH_SIZE;
  vwarp_memcpy(my_space->rank, &rank[v_], VWARP_BATCH_SIZE, warp_offset);
  vwarp_memcpy(my_space->vertices, &(graph.vertices[v_]), VWARP_BATCH_SIZE + 1,
               warp_offset);

  // iterate over my work
  for(vid_t v = 0; v < VWARP_BATCH_SIZE; v++) {
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
void vwarp_compute_normalized_rank_kernel(graph_t graph, float* rank,
                                          float* mailbox) {
  vid_t vertex_id = THREAD_GLOBAL_INDEX;
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
 * Phase2 final kernel of the vwarp PageRank GPU algorithm. This kernel is 
 * similar to the compute_normalized_rank_kernel. The difference is that it 
 * does not normalize the rank (by dividing it by the number of neighbors). 
 * It is invoked in the end to get the final, un-normalized, rank.
 */
__global__
void vwarp_compute_unnormalized_rank_kernel(graph_t graph, float* rank,
                                            float* mailbox) {
  vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  // get sum of neighbors' ranks
  float sum = mailbox[vertex_id];
  // calculate my rank
  rank[vertex_id] =
    ((1 - DAMPING_FACTOR) / graph.vertex_count) + (DAMPING_FACTOR * sum);
}

__host__
error_t page_rank_vwarp_gpu(graph_t* graph, float* rank_i, float* rank) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, rank, &finished);
  if (finished) return rc;

  // Allocate and initialize GPU state
  graph_t* graph_d;
  float* rank_d;
  float* mailbox_d;
  vid_t rank_length;
  rank_length = VWARP_BATCH_SIZE * VWARP_BATCH_COUNT(graph->vertex_count);
  CHK_SUCCESS(initialize_gpu(graph, rank_i, rank_length, &graph_d, &rank_d,
                             &mailbox_d), err);

  {// Configure the kernels. Setup the number of threads for phase1 and phase2,
  // configure the on-chip memory as shared memory rather than L1 cache
  dim3 blocks1, threads_per_block1, blocks2, threads_per_block2;
  uint32_t phase1_thread_count = VWARP_WARP_SIZE * 
    VWARP_BATCH_COUNT(graph->vertex_count);
  KERNEL_CONFIGURE(phase1_thread_count, blocks1, threads_per_block1);
  KERNEL_CONFIGURE(graph->vertex_count, blocks2, threads_per_block2);
  cudaFuncSetCacheConfig(vwarp_sum_neighbors_rank_kernel,
                         cudaFuncCachePreferShared);

  // Iterate for a specific number of rounds
  for (uint32_t round = 0; round < PAGE_RANK_ROUNDS - 1; round++) {
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
  cudaFree(rank_d);
  cudaFree(mailbox_d);
  graph_finalize_device(graph_d);
 err:
  return FAILURE;
}

__host__
error_t page_rank_gpu(graph_t* graph, float* rank_i, float* rank) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, rank, &finished);
  if (finished) return rc;

  // Allocate and initialize GPU state
  graph_t* graph_d;
  float* rank_d;
  float* mailbox_d;
  vid_t rank_length;
  rank_length = VWARP_BATCH_SIZE * VWARP_BATCH_COUNT(graph->vertex_count);
  CHK_SUCCESS(initialize_gpu(graph, rank_i, rank_length, &graph_d, &rank_d,
                             &mailbox_d), err);

  {
  dim3 blocks, threads_per_block;
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
  // Iterate for a specific number of rounds
  for (uint32_t round = 0; round < PAGE_RANK_ROUNDS - 1; round++) {
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
  cudaFree(rank_d);
  cudaFree(mailbox_d);
  graph_finalize_device(graph_d);
 err:
  return FAILURE;
}

error_t page_rank_cpu(graph_t* graph, float* rank_i, float* rank) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, rank, &finished);
  if (finished) return rc;

  // allocate buffers
  float* mailbox = (float*)malloc(graph->vertex_count * sizeof(float));

  // initialize the rank of each vertex
  if (rank_i == NULL) {
    float initial_value = 1 / (float)graph->vertex_count;
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

  for (uint32_t round = 0; round < PAGE_RANK_ROUNDS; round++) {
    // iterate over all vertices to calculate the ranks for this round
    OMP(omp parallel for)
    for(vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      // calculate the sum of all neighbors' rank
      float my_rank = rank[vertex_id];
      for (eid_t i = graph->vertices[vertex_id];
           i < graph->vertices[vertex_id + 1]; i++) {
        const vid_t neighbor_id = graph->edges[i];
        __sync_fetch_and_add_float(&mailbox[neighbor_id], (float)my_rank);
      }
    }

    OMP(omp parallel for)
    for(vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      // get sum of neighbors' ranks
      float sum = mailbox[vertex_id];
      mailbox[vertex_id] = 0;
      // calculate my rank
      vid_t neighbors_count = 
        graph->vertices[vertex_id + 1] - graph->vertices[vertex_id];
      float my_rank =
        ((1 - DAMPING_FACTOR) / graph->vertex_count) + (DAMPING_FACTOR * sum);
      rank[vertex_id] =
        (round == (PAGE_RANK_ROUNDS - 1)) ? my_rank : my_rank / neighbors_count;
    }
  }

  // we are done! set the output and clean up.
  free(mailbox);
  return SUCCESS;
}
