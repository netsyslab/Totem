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

#include "totem_alg.h"

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
*/
PRIVATE
error_t check_special_cases(graph_t* graph, rank_t* rank, bool* finished) {
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
 * Sum the rank of the neighbors.
 * @param[in] graph the graph to apply page rank on
 * @param[in] rank an array storing the current rank of each vertex in the graph
 * @return sum of neighbors' ranks
 */
inline __device__
rank_t sum_neighbors_ranks(graph_t* graph, vid_t vertex_id, rank_t* ranks) {
  rank_t sum = 0;
  for (eid_t i = graph->vertices[vertex_id];
       i < graph->vertices[vertex_id + 1]; i++) {
    const vid_t neighbor = graph->edges[i];
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
void page_rank_kernel(graph_t graph, rank_t* inbox, rank_t* outbox) {
  // get the thread's linear index
  vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  // get sum of neighbors' ranks
  rank_t sum = sum_neighbors_ranks(&graph, vertex_id, inbox);
  // calculate my normalized rank
  rank_t my_rank = ((1 - PAGE_RANK_DAMPING_FACTOR) / graph.vertex_count) + 
    (PAGE_RANK_DAMPING_FACTOR * sum);
  outbox[vertex_id] =  my_rank /
    (graph.vertices[vertex_id + 1] - graph.vertices[vertex_id]);
}

/**
 * This kernel is similar to the main page_rank_kernel. The difference is that
 * it does not normalize the rank by dividing it by the number of neighbors. It
 * is invoked in the end to get the final, un-normalized, rank of each vertex.
 */
__global__
void page_rank_final_kernel(graph_t graph, rank_t* inbox, rank_t* outbox) {
  // get the thread's linear index
  vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  // get sum of neighbors' ranks
  rank_t sum = sum_neighbors_ranks(&graph, vertex_id, inbox);
  // calculate my rank
  outbox[vertex_id] = ((1 - PAGE_RANK_DAMPING_FACTOR) / graph.vertex_count) + 
    (PAGE_RANK_DAMPING_FACTOR * sum);
}

error_t page_rank_incoming_gpu(graph_t* graph, rank_t *rank_i, rank_t* rank) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, rank, &finished);
  if (finished) return rc;

  totem_mem_t type = TOTEM_MEM_DEVICE;

  // will be passed to the kernel
  graph_t* graph_d;
  CHK_SUCCESS(graph_initialize_device(graph, &graph_d), err);

  // allocate inbox and outbox device buffers
  rank_t *inbox_d;
  CHK_SUCCESS(totem_malloc(graph->vertex_count * sizeof(rank_t), type, 
                           (void**)&inbox_d), err_free_graph_d);
  rank_t *outbox_d;
  CHK_SUCCESS(totem_malloc(graph->vertex_count * sizeof(rank_t), type, 
                           (void**)&outbox_d), err_free_inbox);

  // Initialize the intial state of the random walk (i.e., the initial
  // probabilities that the random walker will stop at a given node)
  if (rank_i == NULL) {
    rank_t initial_value = 1 / (rank_t)graph->vertex_count;
    totem_memset(inbox_d, initial_value, graph->vertex_count, TOTEM_MEM_DEVICE);
  } else {
    CHK_CU_SUCCESS(cudaMemcpy(inbox_d, rank_i, graph->vertex_count * 
                              sizeof(rank_t), cudaMemcpyHostToDevice),
                   err_free_outbox);
  }

  {
  /* set the number of blocks, TODO(abdullah) handle the case when
     vertex_count is larger than number of threads. */
  assert(graph->vertex_count <= MAX_THREAD_COUNT);
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
  for (int round = 0; round < PAGE_RANK_ROUNDS - 1; round++) {
    // call the kernel
    page_rank_kernel<<<blocks, threads_per_block>>>
      (*graph_d, inbox_d, outbox_d);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_outbox);

    // swap the inbox and outbox pointers (simulates passing messages)
    rank_t* tmp = inbox_d;
    inbox_d = outbox_d;
    outbox_d = tmp;
  }
  page_rank_final_kernel<<<blocks, threads_per_block>>>
    (*graph_d, inbox_d, outbox_d);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_outbox);
  cudaThreadSynchronize();
  }
  // copy back the final result from the outbox
  CHK_CU_SUCCESS(cudaMemcpy(rank, outbox_d, graph->vertex_count * 
                            sizeof(rank_t), cudaMemcpyDeviceToHost), 
                 err_free_all);

  // we are done! set the output and clean up
  totem_free(outbox_d, type);
  totem_free(inbox_d, type);
  graph_finalize_device(graph_d);
  return SUCCESS;

  // error handlers
 err_free_all:
 err_free_outbox:
  totem_free(outbox_d, type);
 err_free_inbox:
  totem_free(inbox_d, type);
 err_free_graph_d:
  graph_finalize_device(graph_d);
 err:
  return FAILURE;
}

error_t page_rank_incoming_cpu(graph_t* graph, rank_t *rank_i, rank_t* rank) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, rank, &finished);
  if (finished) return rc;

  // allocate buffers
  totem_mem_t type = TOTEM_MEM_HOST;
  rank_t* inbox = NULL;
  CALL_SAFE(totem_malloc(graph->vertex_count * sizeof(rank_t), type, 
                         (void**)&inbox));
  rank_t* outbox = NULL;
  CALL_SAFE(totem_malloc(graph->vertex_count * sizeof(rank_t), type, 
                         (void**)&outbox));

  // initialize the rank of each vertex
  if (rank_i == NULL) {
    rank_t initial_value = 1 / (rank_t)graph->vertex_count;
    totem_memset(outbox, initial_value, graph->vertex_count, type);
  } else {
    OMP(omp parallel for schedule(static))
    for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      outbox[vertex_id] = rank_i[vertex_id];
    }
  }

  for (int round = 0; round < PAGE_RANK_ROUNDS; round++) {
    // swap the inbox and outbox pointers (simulates passing messages!)
    rank_t* tmp = inbox;
    inbox      = outbox;
    outbox     = tmp;

    // iterate over all vertices to calculate the ranks for this round
    // The "runtime" scheduling clause defer the choice of thread scheduling
    // algorithm to the choice of the client, either via OS environment variable
    // or omp_set_schedule interface.
    OMP(omp parallel for schedule(runtime))
    for(vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      // calculate the sum of all neighbors' rank
      rank_t sum = 0;
      for (eid_t i = graph->vertices[vertex_id];
           i < graph->vertices[vertex_id + 1]; i++) {
        vid_t neighbor  = graph->edges[i];
        sum += inbox[neighbor];
      }

      // calculate my rank
      vid_t neighbors_count = 
        graph->vertices[vertex_id + 1] - graph->vertices[vertex_id];
      rank_t my_rank = ((1 - PAGE_RANK_DAMPING_FACTOR) / graph->vertex_count) + 
        (PAGE_RANK_DAMPING_FACTOR * sum);
      outbox[vertex_id] =
        (round == (PAGE_RANK_ROUNDS - 1)) ? my_rank : my_rank / neighbors_count;
    }
  }

  OMP(omp parallel for schedule(static))
  for (vid_t v = 0; v < graph->vertex_count; v++) rank[v] = outbox[v];

  totem_free(inbox, type);
  totem_free(outbox, type);
  return SUCCESS;
}
