/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * This file contains an implementation of the breadth-first search (BFS) graph
 * search algorithm based on description in [Harish07].
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

/**
   This structure is used by the virtual warp-based implementation. It stores a 
   batch of work. It is typically allocated on shared memory and is processed by
   a single virtual warp.
 */
typedef struct {
  uint32_t cost[VWARP_BATCH_SIZE];
  id_t vertices[VWARP_BATCH_SIZE + 1];
  // the following ensures 64-bit alignment, it assumes that the 
  // cost and vertices arrays are of 32-bit elements.
  // TODO(abdullah) a portable way to do this (what if id_t is 64-bit?)
  int pad; 
} vwarp_mem_t;

/**
 * A common initialization function for GPU implementations. It allocates and 
 * initalizes state on the GPU
*/
PRIVATE 
error_t initialize_gpu(const graph_t* graph, id_t source_id, uint64_t cost_len, 
                       graph_t** graph_d, uint32_t** cost_d, 
                       bool** finished_d) {

  // TODO(lauro) Next four lines are not directly related to this function and
  // should have a better location.
  dim3 blocks;
  dim3 threads_per_block;

  // Allocate space on GPU
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);
  CHK_CU_SUCCESS(cudaMalloc((void**) cost_d, cost_len * sizeof(uint32_t)),
                 err_free_graph_d);
  // Initialize cost to INFINITE.
  KERNEL_CONFIGURE(cost_len, blocks, threads_per_block);
  memset_device<<<blocks, threads_per_block>>>((*cost_d), INFINITE, cost_len);
  // For the source vertex, initialize cost.
  CHK_CU_SUCCESS(cudaMemset(&((*cost_d)[source_id]), 0, sizeof(uint32_t)),
                 err_free_cost_d_graph_d);
  // Allocate the termination flag
  CHK_CU_SUCCESS(cudaMalloc((void**) finished_d, sizeof(bool)),
                 err_free_cost_d_graph_d);
  return SUCCESS;

  err_free_cost_d_graph_d:
    cudaFree(cost_d);
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
error_t finalize_gpu(graph_t* graph_d, uint32_t* cost_d, uint32_t** cost) {
  *cost = (uint32_t*) mem_alloc(graph_d->vertex_count * sizeof(uint32_t));
  CHK_CU_SUCCESS(cudaMemcpy(*cost, cost_d, graph_d->vertex_count *
                            sizeof(uint32_t), cudaMemcpyDeviceToHost), err);
  graph_finalize_device(graph_d);
  cudaFree(cost_d);
  return SUCCESS;
 err:
  return FAILURE;
}

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

/**
 * The neighbors processing function. This function sets the level of the 
 * neighbors' vertex to one level more than the vertex. The assumption is that 
 * the threads of a warp invoke this function to process the warp's batch of 
 * work. In each iteration of the for loop, each thread processes a neighbor. 
 * For example, thread 0 in the warp processes neighbors at indices 0, 
 * VWARP_WARP_SIZE, (2 * VWARP_WARP_SIZE) etc. in the edges array, while thread 
 * 1 in the warp processes neighbors 1, (1 + VWARP_WARP_SIZE), 
 * (1 + 2 * VWARP_WARP_SIZE) and so on.
*/
__device__
void vwarp_process_neighbors(int warp_offset, int neighbor_count, id_t* edges, 
                             uint32_t* cost, int level, bool* finished) {
  for(int i = warp_offset; i < neighbor_count; i += VWARP_WARP_SIZE) {
    int neighbor_id = edges[i];
    if (cost[neighbor_id] == INFINITE) {
      cost[neighbor_id] = level + 1;
      *finished = false;
    } 
  }
}

/**
 * A warp-based implementation of the BFS kernel. Please refer to the 
 * description of the warp technique for details. Also, please refer to
 * bfs_kernel for details on the BFS implementation. 
 */
__global__
void vwarp_bfs_kernel(graph_t graph, uint32_t level, bool* finished,
                      uint32_t* cost, uint32_t thread_count) {

  if (THREAD_GLOBAL_INDEX >= thread_count) return;

  int warp_offset = THREAD_GLOBAL_INDEX % VWARP_WARP_SIZE;
  int warp_id     = THREAD_GLOBAL_INDEX / VWARP_WARP_SIZE;
  
  __shared__ vwarp_mem_t shared_memory[(MAX_THREADS_PER_BLOCK / 
                                        VWARP_WARP_SIZE)];
  vwarp_mem_t* my_space = shared_memory + (THREAD_GRID_INDEX / VWARP_WARP_SIZE);

  // copy my work to local space
  int v_ = warp_id * VWARP_BATCH_SIZE;
  vwarp_memcpy(my_space->cost, &cost[v_], VWARP_BATCH_SIZE, warp_offset);
  vwarp_memcpy(my_space->vertices, &(graph.vertices[v_]), VWARP_BATCH_SIZE + 1, 
               warp_offset);

  // iterate over my work
  for(uint32_t v = 0; v < VWARP_BATCH_SIZE; v++) {
    if (my_space->cost[v] == level) {
      int neighbor_count = my_space->vertices[v + 1] - my_space->vertices[v];
      id_t* neighbors = &(graph.edges[my_space->vertices[v]]);
      vwarp_process_neighbors(warp_offset, neighbor_count, neighbors, cost, 
                              level, finished);
    }
  }
}

/**
 * A modified version of bfs_kernel_warp kernel that does not use shared memory.
 * this is a drop-in replacement for the vwarp_bfs_kernel kernel.
 * TODO(Abdullah) This kernel is not currently in use, will be used once we have
 * an elegant soultion for the multi-version algorithms.
*/
__global__
void vwarp_bfs_kernel_no_shared(graph_t graph, uint32_t level, bool* finished, 
                                uint32_t* cost, uint32_t thread_count) {

  if (THREAD_GLOBAL_INDEX >= thread_count) return;

  int warp_offset = THREAD_GLOBAL_INDEX % VWARP_WARP_SIZE;
  int warp_id     = THREAD_GLOBAL_INDEX / VWARP_WARP_SIZE;
  
  // iterate over my work
  int batch_offset = warp_id * VWARP_BATCH_SIZE;
  for(uint64_t v = batch_offset; v < (VWARP_BATCH_SIZE + batch_offset); v++) {
    if (cost[v] == level) {
      int neighbor_count = graph.vertices[v + 1] - graph.vertices[v];
      id_t* neighbors = &(graph.edges[graph.vertices[v]]);
      vwarp_process_neighbors(warp_offset, neighbor_count, neighbors, cost,
                              level, finished);
    }
  }
}

__host__
error_t bfs_vwarp_gpu(graph_t* graph, id_t source_id, uint32_t** cost) {
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

  // Create and initialize state on GPU
  graph_t* graph_d;
  uint32_t* cost_d;
  uint64_t cost_length;
  bool* finished_d;
  cost_length = VWARP_BATCH_SIZE * VWARP_BATCH_COUNT(graph->vertex_count);
  CHK_SUCCESS(initialize_gpu(graph, source_id, cost_length, &graph_d, 
                             &cost_d, &finished_d), err_free_all);

  // {} used to limit scope and avoid problems with error handles.
  {
  // Configure the kernel's threads and on-chip memory. On-ship memory is 
  // configured as shared memory rather than L1 cache
  dim3 blocks;
  dim3 threads_per_block;
  int thread_count = VWARP_WARP_SIZE * VWARP_BATCH_COUNT(graph->vertex_count);
  KERNEL_CONFIGURE(thread_count, blocks, threads_per_block);
  cudaFuncSetCacheConfig(vwarp_bfs_kernel, cudaFuncCachePreferShared);
  bool finished = false;
  // while the current level has vertices to be processed.
  for (uint32_t level = 0; !finished; level++) {
    CHK_CU_SUCCESS(cudaMemset(finished_d, true, 1), err_free_all);
    vwarp_bfs_kernel<<<blocks, threads_per_block>>>(*graph_d, level, finished_d,
                                                    cost_d, thread_count);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
    CHK_CU_SUCCESS(cudaMemcpy(&finished, finished_d, sizeof(bool),
                              cudaMemcpyDeviceToHost), err_free_all);
  }
  }

  CHK_SUCCESS(finalize_gpu(graph_d, cost_d, cost), err_free_all);
  return SUCCESS;

  // error handlers
  err_free_all:
    cudaFree(finished_d);
    cudaFree(cost_d);
    graph_finalize_device(graph_d);
    *cost = NULL;
    return FAILURE;
}

__host__
error_t bfs_gpu(graph_t* graph, id_t source_id, uint32_t** cost) {
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

  // Create and initialize state on GPU
  graph_t* graph_d;
  uint32_t* cost_d;
  bool* finished_d;
  CHK_SUCCESS(initialize_gpu(graph, source_id, graph->vertex_count, 
                             &graph_d, &cost_d, &finished_d), err_free_all);
  
  // {} used to limit scope and avoid problems with error handles.
  {
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
  bool finished = false;
  // while the current level has vertices to be processed.
  for (uint32_t level = 0; !finished; level++) {    
    CHK_CU_SUCCESS(cudaMemset(finished_d, true, 1), err_free_all);
    // for each vertex V in parallel do
    bfs_kernel<<<blocks, threads_per_block>>>(*graph_d, level, finished_d,
                                              cost_d);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
    CHK_CU_SUCCESS(cudaMemcpy(&finished, finished_d, sizeof(bool),
                              cudaMemcpyDeviceToHost), err_free_all);
  }}

  // We are done, get the results back and clean up state
  CHK_SUCCESS(finalize_gpu(graph_d, cost_d, cost), err_free_all);
  return SUCCESS;

  // error handlers
  err_free_all:
    cudaFree(finished_d);
    cudaFree(cost_d);
    graph_finalize_device(graph_d);
    *cost = NULL;
    return FAILURE;
}

__host__
error_t bfs_cpu(graph_t* graph, id_t source_id, uint32_t** cost_ret) {
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
