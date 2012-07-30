/**
 *
 * Implements a parallel version of a variation of the k-cores algorithm
 * described in [Batagelj2002] V. Batagelj and M. Zaversnik, "An O(m)
 * Algorithm for Cores Decomposition of Networks".
 *
 * The k-core "is a maximal subset of vertices such that each is connected to
 * at least k others in the subset. The word maximal here means that there is
 * no other vertex in the graph that can be added to the subset while preserving
 * the propoerty that every vertex is connected to k other vertices"
 * [Newman2010] M. E. J. Newman, "Networks: An Introduction".
 *
 * This implementation takes into account the weights on the edges instead of
 * the number of edges each vertex is connected to. Hence, we call the
 * algorithm p-core.
 *
 *  Created on: 2011-05-24
 *      Author: Abdullah Gharaibeh (abdullah@ece.ubc.ca)
 */

// totem includes
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

// indicates an active vertex
#define ACTIVE_FLAG ((uint32_t)-1)

// indices in the finish_flags array for the GPU implementation
#define OVERALL_INDEX 0
#define ROUND_INDEX   1

/**
 * Set the initial state of the algorithm. The weights_sum array is initalized
 * with the sum of edge weights each vertex is connected to. The round array is
 * initalized to the flag ACTIVE_FLAG indicating that all vertices are active
 * and that none of them has its round set yet.
 * @param[in] graph an instance of the graph structure
 * @param[in|out] weights_sum sum of edge weights for each vertex
 * @param[in|out] round the round each vertex belongs to if any
 */
__global__
void init_state_kernel(graph_t graph, int* weights_sum, uint32_t* round) {
  // get the thread's linear index
  id_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;

  // all vertices are active at the beginnig
  round[vertex_id] = ACTIVE_FLAG;

  /* initialize the sum of edge weights for this vertex. Since all vertices
     are active at this stage, all edges of a vertex are considered */
  weights_sum[vertex_id] = 0;
  for (uint64_t i = graph.vertices[vertex_id];
       i < graph.vertices[vertex_id + 1]; i++) {
    weights_sum[vertex_id] += (int)graph.weights[i];
  } // for
}

/**
 * Perform a single iteration in the context of a specific round (cur_round).
 * The vertices that belong to this round are set. Also, The finish_flags are
 * set to indicate end of round and overall end of processing.
 * @param[in]  graph an instance of the graph structure
 * @param[in]  weights_sum sum of edge weights for each vertex
 * @param[in]  round the round each vertex belongs to if any
 * @param[in]  cur_round the current round being processed
 * @param[in]  cur_threshold the current threshold considered
 * @param[out] finish_flags represetnts two flags, one indicates if the overall
 *             processing is done, another indicates if the round is done.
 */
__global__
void pcore_kernel(graph_t graph, int* weights_sum, uint32_t* round,
                  uint32_t cur_round,  uint32_t cur_threshold,
                  bool* finish_flags) {
  // get the thread's linear index
  id_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count ||
      round[vertex_id] != ACTIVE_FLAG) return;

  // check if the vertex belongs to the current p-[range]-core
  if (weights_sum[vertex_id] <= cur_threshold) {
    // deactivate the vertex (assign its round value)
    round[vertex_id] = cur_round;
    /* since this vertex has been deactivated, it is virtually not part of
       the graph anymore; therefore, we update the weight_sum of each of
       its neighbors (remember that the graph is undirected). Note that
       the round_finished flag is set to false to guarantee that the
       active neighbors of this vertex will be visited again after their
       weights_sum have been updated */
    finish_flags[ROUND_INDEX] = false;
    for (uint64_t i = graph.vertices[vertex_id];
         i < graph.vertices[vertex_id + 1]; i++) {
      id_t neighbor_id = graph.edges[i];
      // atomically modify the neighbor's sum of edge weights
      atomicSub(&(weights_sum[neighbor_id]), (int)graph.weights[i]);
    }
  } else {
    // some vertices are still active, more rounds are required
    finish_flags[OVERALL_INDEX] = false;
  } // if
}


/**
 * Verify the input to the algorithm.
 * @param[in] graph an instance of the graph structure
 * @param[in] start the start value of p
 * @param[in] step the value used to increment p in each new round
 */
PRIVATE inline
error_t verify_input(const graph_t* graph, uint32_t start, uint32_t step) {
  if (!graph || !graph->weighted || graph->directed || step == 0 ||
      !graph->vertex_count) return FAILURE;
  return SUCCESS;
}

error_t pcore_gpu(const graph_t* graph, uint32_t start, uint32_t step,
                  uint32_t** round_out) {

  // kernel configuration parameters
  dim3 block_count;
  dim3 threads_per_block;

  CHK_SUCCESS(verify_input(graph, start, step), err);

  // simple optimization for a single node graph
  if (graph->vertex_count == 1) {
    *round_out = (uint32_t*)mem_alloc(sizeof(uint32_t));
    (*round_out)[0] = graph->edge_count;
    return SUCCESS;
  }

  // allocate and transfer the graph
  graph_t* graph_d;
  CHK_SUCCESS(graph_initialize_device(graph, &graph_d), err);

  // allocate algorithm-specific state
  int* weights_sum_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&weights_sum_d, graph->vertex_count *
                            sizeof(int)), err_free_graph);
  uint32_t* round_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&round_d, graph->vertex_count *
                            sizeof(uint32_t)), err_free_weights_sum);
  bool* finish_flags_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&finish_flags_d, 2 * sizeof(bool)),
                 err_free_round);

  // compute the number of blocks
  KERNEL_CONFIGURE(graph_d->vertex_count, block_count, threads_per_block);

  // initialize state
  init_state_kernel<<<block_count, threads_per_block>>>
    (*graph_d, weights_sum_d, round_d);
  /* for each round (the outer-most for loop), p-core is computed for the set of
     active vertices for p = cur_threshold + epsilon. The vertices that don't
     belong to current p-core, their round is set (to cur_round), hence they get
     deactivated (i.e., will not be considered in future rounds). The process
     finishes when all vertices are deactivated (i.e., all vertices have been
     assigned a round). */
  bool*    finish_flags;
  finish_flags = (bool*)mem_alloc(2 * sizeof(bool));
  uint32_t cur_round;
  uint32_t cur_threshold;
  finish_flags[OVERALL_INDEX] = false;
  cur_round       = 0;
  cur_threshold   = start;
  while (!finish_flags[OVERALL_INDEX]) {

    finish_flags[ROUND_INDEX] = false;
    while (!finish_flags[ROUND_INDEX]) {
      CHK_CU_SUCCESS(cudaMemset(finish_flags_d, true, 2 * sizeof(bool)),
                     err_free_finish_flags);
      pcore_kernel<<<block_count, threads_per_block>>>
        (*graph_d, weights_sum_d, round_d, cur_round,
         cur_threshold, finish_flags_d);
      CHK_CU_SUCCESS(cudaMemcpy(finish_flags, finish_flags_d, 2 * sizeof(bool),
                                cudaMemcpyDeviceToHost), err_free_finish_flags);
    } // while !round_finished
    // prepare state for the next round
    cur_threshold += step;
    cur_round++;
  } // while !finished

  uint32_t* round;
  round  = (uint32_t*)mem_alloc(graph->vertex_count * sizeof(uint32_t));
  CHK_CU_SUCCESS(cudaMemcpy(round, round_d,
                            graph->vertex_count * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost), err_free_all);
  *round_out = round;

  // release allocated memory
  graph_finalize_device(graph_d);
  cudaFree(round_d);
  cudaFree(weights_sum_d);
  cudaFree(finish_flags_d);
  mem_free(finish_flags);

  return SUCCESS;

 err_free_all:
  mem_free(round);
 err_free_finish_flags:
  cudaFree(finish_flags_d);
  mem_free(finish_flags);
 err_free_round:
  cudaFree(round_d);
 err_free_weights_sum:
  cudaFree(weights_sum_d);
 err_free_graph:
  graph_finalize_device(graph_d);
 err:
  *round_out = NULL;
  return FAILURE;
}

error_t pcore_cpu(const graph_t* graph, uint32_t start, uint32_t step,
                  uint32_t** round_out) {
  CHK_SUCCESS(verify_input(graph, start, step), err);

  // simple optimization for a single node graph
  if (graph->vertex_count == 1) {
    *round_out = (uint32_t*)mem_alloc(sizeof(uint32_t));
    (*round_out)[0] = graph->edge_count;
    return SUCCESS;
  }

  // allocate and initialize state
  int* weights_sum;
  weights_sum = (int*)mem_alloc(graph->vertex_count * sizeof(int));
  uint32_t* round;
  round  = (uint32_t*)mem_alloc(graph->vertex_count * sizeof(uint32_t));

  OMP(omp parallel for)
  for (id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    // all vertices are active at the beginnig
    round[vertex_id] = ACTIVE_FLAG;
    /* initialize the sum of edge weights for this vertex. Since all vertices
       are active at this stage, all edges of a vertex are considered */
    weights_sum[vertex_id] = 0;
    for (uint64_t i = graph->vertices[vertex_id];
         i < graph->vertices[vertex_id + 1]; i++) {
      weights_sum[vertex_id] += (int)graph->weights[i];
    } // for
  } // for

  /* for each round (the outer-most for loop), p-core is computed for the set of
     active vertices for p = cur_threshold + epsilon. The vertices that don't
     belong to current p-core, their round is set (to cur_round), hence they get
     deactivated (i.e., will not be considered in future rounds). The process
     finishes when all vertices are deactivated (i.e., all vertices have been
     assigned a round). */
  bool     finished;
  uint32_t cur_round;
  uint32_t cur_threshold;
  finished      = false;
  cur_round     = 0;
  cur_threshold = start;
  while (!finished) {
    bool round_finished = false;
    while (!round_finished) {
      round_finished = true; // a deactivated vertex will set this back to false
      finished = true; // an active vertex will set this back to false
      OMP(omp parallel for)
      for (id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
        if (round[vertex_id] != ACTIVE_FLAG) {
          continue;
        }

        // check if the vertex belongs to the current p-[range]-core
        if (weights_sum[vertex_id] <= cur_threshold) {
          // deactivate the vertex (assign its round value)
          round[vertex_id] = cur_round;
          /* since this vertex has been deactivated, it is virtually not part of
             the graph anymore; therefore, we update the weight_sum of each of
             its neighbors (remember that the graph is undirected). Note that
             the round_finished flag is set to false to guarantee that the
             active neighbors of this vertex will be visited again after their
             weights_sum have been updated */
          round_finished = false;
          for (uint64_t i = graph->vertices[vertex_id];
               i < graph->vertices[vertex_id + 1]; i++) {
            id_t neighbor_id = graph->edges[i];
            // atomically modify the neighbor's sum of edge weights
            __sync_sub_and_fetch(&(weights_sum[neighbor_id]),
                                 (int)graph->weights[i]);
          }
        } else {
          // some vertices are still active, more rounds are required
          finished = false;
        } // if
      } // for
    } // while !round_finished

    // prepare state for the next round
    cur_threshold += step;
    cur_round++;
  } // while !finished

  *round_out = round;
  return SUCCESS;

 err:
  *round_out = NULL;
  return FAILURE;
}
