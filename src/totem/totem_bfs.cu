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

// TODO(elizeu): change the type from uint32_t to id_t for graph->vertices and
// graph->edges.

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
  for (uint32_t i = graph.vertices[vertex_id];
       i < graph.vertices[vertex_id + 1]; i++) {
    const uint32_t neighbor_id = graph.edges[i];
    if (cost[neighbor_id] == INFINITE) {
      // Threads may update finished and the same position in the cost array
      // concurrently. It does not affect correctness since all
      // threads would update with the same value.
      *finished = false;
      cost[neighbor_id] = level + 1;
    }
  } // for
}

// TODO(lauro): Add CHECK_ERR for CUDA functions.
// TODO(lauro): Return an error_t and have the a yuck out param.
__host__
uint32_t* bfs_gpu(uint32_t source_id, const graph_t* graph) {
  if( (graph == NULL) || (source_id >= graph->vertex_count) ) {
    return NULL;
  } else if( graph->vertex_count == 1 ) {
    uint32_t* cost = (uint32_t*) mem_alloc(sizeof(uint32_t));
    cost[0] = 0;
    return cost;
  }
  // TODO(lauro): More optimizations can be performed here. For example, if
  // there is no edge. It can return the cost array initialize as INFINITE.

  // Create graph on GPU memory.
  // TODO(lauro): Move to some utility library. We will often need this.
  graph_t graph_d = *graph;
  cudaMalloc((void**) &(graph_d.vertices),
             (graph->vertex_count + 1) * sizeof(uint32_t));
  cudaMalloc((void**) &(graph_d.edges),
             graph->edge_count * sizeof(uint32_t));
  cudaMemcpy(graph_d.vertices, graph->vertices,
            (graph->vertex_count + 1)  * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(graph_d.edges, graph->edges, graph->edge_count * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  // TODO(lauro) Next three lines are not directly related to this function and
  // should have a better location.
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);

  // Create cost array only on GPU.
  uint32_t* cost_d;
  cudaMalloc((void**) &cost_d, graph->vertex_count * sizeof(uint32_t));
  // Initialize cost to INFINITE.
  memset_device<<<blocks, threads_per_block>>>(cost_d, INFINITE,
                                               graph->vertex_count);

  // For the source vertex, initialize cost.
  cudaMemset(&(cost_d[source_id]), 0, sizeof(uint32_t));

  // while the current level has vertices to be processed.
  bool finished = false;
  bool* finished_d;
  cudaMalloc((void**) &finished_d, sizeof(bool));
  for (uint32_t level = 0; !finished; level++) {
    cudaMemset(finished_d, true, 1);
    // for each vertex V in parallel do
    bfs_kernel<<<blocks, threads_per_block>>>(graph_d, level, finished_d,
                                              cost_d);
    cudaMemcpy(&finished, finished_d, sizeof(bool), cudaMemcpyDeviceToHost);
  }

  cudaFree(graph_d.vertices);
  cudaFree(graph_d.edges);

  uint32_t* cost = (uint32_t*) mem_alloc(graph->vertex_count *
                                         sizeof(uint32_t));
  cudaMemcpy(cost, cost_d, graph->vertex_count * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaFree(cost_d);
  return cost;
}

__host__
uint32_t* bfs_cpu(uint32_t source_id, const graph_t* graph) {
  if( (graph == NULL) || (source_id >= graph->vertex_count) ) {
    return NULL;
  }

  uint32_t* cost = (uint32_t* ) mem_alloc(graph->vertex_count *
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
    for (uint32_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      if (cost[vertex_id] != level) continue;
      for (uint32_t i = graph->vertices[vertex_id];
           i < graph->vertices[vertex_id + 1]; i++) {
        const uint32_t neighbor_id = graph->edges[i];
        if (cost[neighbor_id] == INFINITE) {
          finished = false;
          cost[neighbor_id] = level + 1;
        }
      }
    }
  }
  return cost;
}
