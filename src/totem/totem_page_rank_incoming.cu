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
 *  Created on: 2011-03-06
 *  Author: Abdullah Gharaibeh
 */

#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

// TODO(abdullah): The following two values should be a parameter in the entry
//                 function to enable more flexibility and experimentation.
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
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
*/
PRIVATE
error_t check_special_cases(graph_t* graph, float** rank, bool* finished) {
  *finished = true;
  if (graph == NULL) {
    return FAILURE;
  } else if (graph->vertex_count == 0) {
    return FAILURE;
  } else if (graph->vertex_count == 1) {
    *rank = (float*)mem_alloc(sizeof(float));
    (*rank)[0] = (float)1.0;
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

/**
 * Sum the rank of the neighbors.
 * @param[in] graph the graph to apply page rank on
 * @param[in] rank an array storing the current rank of each vertex in the graph
 * @return sum of neighbors' ranks
 */
inline __device__ 
double sum_neighbors_ranks(graph_t* graph, id_t vertex_id, float* ranks) {
  double sum = 0;
  for (uint64_t i = graph->vertices[vertex_id];
       i < graph->vertices[vertex_id + 1]; i++) {
    const id_t neighbor = graph->edges[i];
    sum += ranks[neighbor];
  }
  return sum;
}

/**
 * The PageRank kernel. Based on the algorithm described in [Malewicz2010].
 * For each round, each vertex broadcasts along each outgoing edge its tentative
 * PageRank divided by the number of outgoing edges. The tentative PageRank of
 * vertex is calculated as follows: the vertex sums up the values arriving into
 * sum and sets its own tentative PageRank to
 * ((1 - DAMPING_FACTOR) / vertex_count + DAMPING_FACTOR * sum).
 * Broadcasting messages over outgoing edges is done as follows: the value is
 * placed in the outbox buffer. In the next round the inbox and outbox are
 * swapped, and the message will be accessed in the next round via the
 * inbox buffer. This operation simulates a broadcast because all the neighbors
 * of vertex v will access the same location (i.e., inbox[v]) to get the message
 * (i.e., tentative_PageRank_of_v/neighbor_count). In the last round, outbox
 * will contain the PageRank of each vertex.
 * @param[in] graph the graph to apply page rank on
 * @param[in] inbox messages broadcasted to vertices
 * @param[in] outbox messages to be broadcasted in the next round
 */
__global__
void page_rank_kernel(graph_t graph, float* inbox, float* outbox) {
  // get the thread's linear index
  id_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  // get sum of neighbors' ranks
  double sum = sum_neighbors_ranks(&graph, vertex_id, inbox);
  // calculate my normalized rank
  float my_rank =
    ((1 - DAMPING_FACTOR) / graph.vertex_count) + (DAMPING_FACTOR * sum);
  outbox[vertex_id] =  my_rank /
    (graph.vertices[vertex_id + 1] - graph.vertices[vertex_id]);
}

/**
 * This kernel is similar to the main page_rank_kernel. The difference is that
 * it does not normalize the rank by dividing it by the number of neighbors. It 
 * is invoked in the end to get the final, un-normalized, rank of each vertex.
 */
__global__
void page_rank_final_kernel(graph_t graph, float* inbox, float* outbox) {
  // get the thread's linear index
  id_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  // get sum of neighbors' ranks
  double sum = sum_neighbors_ranks(&graph, vertex_id, inbox);
  // calculate my rank
  outbox[vertex_id] = 
    ((1 - DAMPING_FACTOR) / graph.vertex_count) + (DAMPING_FACTOR * sum);
}

error_t page_rank_incoming_gpu(graph_t* graph, float *rank_i, float** rank) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, rank, &finished);
  if (finished) return rc;

  /* had to define them at the beginning to avoid a compilation problem with
     goto-label error handling mechanism */
  dim3 blocks;
  dim3 threads_per_block;

  // will be passed to the kernel
  graph_t* graph_d;
  CHK_SUCCESS(graph_initialize_device(graph, &graph_d), err);

  // allocate inbox and outbox device buffers
  float *inbox_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&inbox_d, graph->vertex_count * 
                       sizeof(float)), err_free_graph_d);
  float *outbox_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&outbox_d, graph->vertex_count * 
                       sizeof(float)), err_free_inbox);

  /* set the number of blocks, TODO(abdullah) handle the case when
     vertex_count is larger than number of threads. */
  assert(graph->vertex_count <= MAX_THREAD_COUNT);
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);

  // Initialize the intial state of the random walk (i.e., the initial
  // probabilities that the random walker will stop at a given node)
  if (rank_i == NULL) {
    float initial_value;
    initial_value = 1 / (float)graph->vertex_count;
    memset_device<<<blocks, threads_per_block>>>
        (inbox_d, initial_value, graph->vertex_count);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_outbox);
  } else {
    CHK_CU_SUCCESS(cudaMemcpy(inbox_d, rank_i,
        graph->vertex_count * sizeof(float), cudaMemcpyHostToDevice),
        err_free_outbox);
  }

  for (uint32_t round = 0; round < PAGE_RANK_ROUNDS - 1; round++) {
    // call the kernel
    page_rank_kernel<<<blocks, threads_per_block>>>
      (*graph_d, inbox_d, outbox_d);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_outbox);

    // swap the inbox and outbox pointers (simulates passing messages)
    float* tmp = inbox_d;
    inbox_d = outbox_d;
    outbox_d = tmp;
  }
  page_rank_final_kernel<<<blocks, threads_per_block>>>
    (*graph_d, inbox_d, outbox_d);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_outbox);
  cudaThreadSynchronize();

  // copy back the final result from the outbox
  *rank = (float*)mem_alloc(graph->vertex_count * sizeof(float));
  CHK_CU_SUCCESS(cudaMemcpy(*rank, outbox_d, graph->vertex_count * 
                            sizeof(float), cudaMemcpyDeviceToHost), 
                 err_free_all);

  // we are done! set the output and clean up
  cudaFree(outbox_d);
  cudaFree(inbox_d);
  graph_finalize_device(graph_d);
  return SUCCESS;

  // error handlers
 err_free_all:
 err_free_outbox:
  cudaFree(outbox_d);
 err_free_inbox:
  cudaFree(inbox_d);
 err_free_graph_d:
  graph_finalize_device(graph_d);
 err:
  return FAILURE;
}

error_t page_rank_incoming_cpu(graph_t* graph, float *rank_i, float** rank) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, rank, &finished);
  if (finished) return rc;

  // allocate buffers
  float* inbox = (float*)mem_alloc(graph->vertex_count * sizeof(float));
  float* outbox = (float*)mem_alloc(graph->vertex_count * sizeof(float));

  // initialize the rank of each vertex
  if (rank_i == NULL) {
    float initial_value;
    initial_value = 1 / (float)graph->vertex_count;
    for (id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      outbox[vertex_id] = initial_value;
    }
  } else {
    for (id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      outbox[vertex_id] = rank_i[vertex_id];
    }
  }

  for (int round = 0; round < PAGE_RANK_ROUNDS; round++) {
    // swap the inbox and outbox pointers (simulates passing messages!)
    float* tmp = inbox;
    inbox      = outbox;
    outbox     = tmp;

    // iterate over all vertices to calculate the ranks for this round
#ifdef _OPENMP // this is defined if -fopenmp flag is passed to the compiler
#pragma omp parallel for
#endif // _OPENMP
    for(id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      // calculate the sum of all neighbors' rank
      double sum = 0;
      for (uint64_t i = graph->vertices[vertex_id];
           i < graph->vertices[vertex_id + 1]; i++) {
        id_t neighbor  = graph->edges[i];
        sum += inbox[neighbor];
      }

      // calculate my rank
      uint64_t neighbors_count =
        graph->vertices[vertex_id + 1] - graph->vertices[vertex_id];
      float my_rank =
        ((1 - DAMPING_FACTOR) / graph->vertex_count) + (DAMPING_FACTOR * sum);
      outbox[vertex_id] =
        (round == (PAGE_RANK_ROUNDS - 1)) ? my_rank : my_rank / neighbors_count;
    }
  }

  // we are done! set the output and clean up.
  *rank = outbox;
  mem_free(inbox);
  return SUCCESS;
}
