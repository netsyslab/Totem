/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * This file contains an implementation of the breadth-first search (BFS) graph
 * search algorithm based as described in [Harish07].
 * [Harish07] P. Harish and P. Narayanan, "Accelerating large graph algorithms
 *   on the GPU using CUDA," in High Performance Computing - HiPC 2007,
 *   LNCS v. 4873, ch. 21, doi: http://dx.doi.org/10.1007/978-3-540-77220-0_21
 *
 *  Created on: 2011-02-28
 *      Author: Lauro Beltrão Costa
 */

// system includes
#include <cuda.h>

// totem includes
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

/* This comment describes implementation details of the next two functions.
 Modified from [Harish07].
 Breadth First Search
 This implementation uses level synchronization. BFS traverses the
 graph in levels; once a level is visited it is not visited again.
 The BFS frontier corresponds to all the nodes being processed at the current
 level.
 Each thread process a vertex (in the following text these terms are used
 interchangeably). An integer array, cost_d, stores the minimal number of edges
 from the source vertex to each vertex. The cost for vertices that have not been
 visited yet is INFINITE. In each iteration, each vertex checks if it belongs to
 the current level by verifying its own cost. If it does, it updates its not yet
 visited neighbors. If the cost of, at least, one neighbor is updated, the
 variable finished_d is set to false and there will be another iteration.
 */
__global__
void bfs_kernel(graph_t graph, uint32_t level, bool* finished, uint32_t* cost) {
  const int vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  if (cost[vertex_id] != level) return;

  // TODO(lauro, abdullah): one optimization is to load the neighbors ids to
  // shared memory to facilitate  coalesced memory access.
  // for all neighbors of vertex_id
  for (id_t i = graph.vertices[vertex_id];
       i < graph.vertices[vertex_id + 1]; i++) {
    const id_t neighbor_id = graph.edges[i];
    if (cost[neighbor_id] == INFINITE) {
      // Threads may update finished and the same position in the cost array
      // concurrently. It does not affect correctness since all
      // threads would update with the same value.
      *finished = false;
      cost[neighbor_id] = level + 1;
    }
  } // for
}

__host__
error_t bfs_gpu(id_t source_id, const graph_t* graph, uint32_t** cost) {
  // TODO(lauro,abdullah): Factor out a validate graph function.
  if((graph == NULL) || (source_id >= graph->vertex_count)) {
    *cost = NULL;
    return FAILURE;
  } else if(graph->vertex_count == 1) {
    *cost = (uint32_t*) mem_alloc(sizeof(uint32_t));
    *cost[0] = 0;
    return SUCCESS;
  }
  // TODO(lauro): More optimizations can be performed here. For example, if
  // there is no edge. It can return the cost array initialize as INFINITE.

  // TODO(lauro) Next four lines are not directly related to this function and
  // should have a better location.
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
  //TODO(abdullah, lauro) handle the case (vertex_count > number of threads).
  assert(graph->vertex_count <= MAX_THREAD_COUNT);

  // Create graph on GPU memory.
  graph_t* graph_d;
  CHK_SUCCESS(graph_initialize_device(graph, &graph_d), err);

  // Create cost array only on GPU.
  uint32_t* cost_d;
  CHK_CU_SUCCESS(cudaMalloc((void**) &cost_d,
                            graph->vertex_count * sizeof(uint32_t)),
                 err_free_graph_d);

  // Initialize cost to INFINITE.
  memset_device<<<blocks, threads_per_block>>>(cost_d, INFINITE,
                                               graph->vertex_count);

  // For the source vertex, initialize cost.
  CHK_CU_SUCCESS(cudaMemset(&(cost_d[source_id]), 0, sizeof(uint32_t)),
                 err_free_cost_d_graph_d);

  // while the current level has vertices to be processed.
  // {} used to limit scope and avoid problems with error handles.
  bool* finished_d;
  {bool finished = false;
  CHK_CU_SUCCESS(cudaMalloc((void**) &finished_d, sizeof(bool)),
                 err_free_cost_d_graph_d);

  for (uint32_t level = 0; !finished; level++) {
    CHK_CU_SUCCESS(cudaMemset(finished_d, true, 1), err_free_all);
    // for each vertex V in parallel do
    bfs_kernel<<<blocks, threads_per_block>>>(*graph_d, level, finished_d,
                                              cost_d);
    CHK_CU_SUCCESS(cudaMemcpy(&finished, finished_d, sizeof(bool),
                              cudaMemcpyDeviceToHost), err_free_all);
  }}

  *cost = (uint32_t*) mem_alloc(graph->vertex_count * sizeof(uint32_t));
  CHK_CU_SUCCESS(cudaMemcpy(*cost, cost_d, graph->vertex_count * 
                            sizeof(uint32_t), cudaMemcpyDeviceToHost),
                 err_free_all);

  graph_finalize_device(graph_d);
  cudaFree(cost_d);
  return SUCCESS;

  // error handlers
  err_free_all:
    cudaFree(finished_d);
  err_free_cost_d_graph_d:
    cudaFree(cost_d);
  err_free_graph_d:
    graph_finalize_device(graph_d);
  err:
  // TODO(lauro, abdullah): This msg is useless. Unless it comes directly to err
  // it always prints cudaFree errors and not the actual error code.
    printf("%d\n", cudaGetLastError());
    *cost = NULL;
    return FAILURE;
}

__host__
error_t bfs_cpu(id_t source_id, const graph_t* graph, uint32_t** cost_ret) {
  if((graph == NULL) || (source_id >= graph->vertex_count)) {
    *cost_ret = NULL;
    return FAILURE;
  }

  uint32_t* cost = (uint32_t*) mem_alloc(graph->vertex_count *
                                         sizeof(uint32_t));
  // Initialize cost to INFINITE.
  memset(cost, 0xFF, graph->vertex_count * sizeof(uint32_t));
  // For the source vertex, initialize cost.
  cost[source_id] = 0;

  // while the current level has vertices to be processed.
  bool finished = false;
  for (uint32_t level = 0; !finished; level++) {
    finished = true;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif // _OPENMP
    for (id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      if (cost[vertex_id] != level) continue;
      for (id_t i = graph->vertices[vertex_id];
           i < graph->vertices[vertex_id + 1]; i++) {
        const id_t neighbor_id = graph->edges[i];
        if (cost[neighbor_id] == INFINITE) {
          finished = false;
          cost[neighbor_id] = level + 1;
        }
      }
    }
  }
  *cost_ret = cost;
  return SUCCESS;
}
