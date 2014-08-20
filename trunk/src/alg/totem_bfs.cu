/*
 * This file contains an implementation of the breadth-first search (BFS) graph
 * search algorithm based on the algorithms in [Hong2011PPoPP, Hong2011PACT].
 * [Hong2011PPoPP] S. Hong,  S.K. Kim, T. Oguntebi and K. Olukotun, 
 *   "Accelerating CUDA graph algorithms at maximum warp" in PPoPP 2011.
 * [Hong2011PACT] S. Hong, T. Oguntebi and K. Olukotun, "Efficient parallel 
 *   graph exploration on multi-core cpu and gpu" in PACT 2011.
 *
 *  Created on: 2011-02-28
 *      Author: Lauro Beltrão Costa
 *              Abdullah Gharaibeh
 */

// totem includes
#include "totem_alg.h"

/**
 * This structure is used by the virtual warp-based implementation. It stores a
 * batch of work. It is allocated on shared memory and is processed by a single
 * virtual warp.
 */
typedef struct {
  // One is added to make it easy to calculate the number of neighbors of the
  // last vertex. Another one is added to ensure 8Bytes alignment irrespective
  // whether sizeof(eid_t) is 4 or 8. Alignment is enforced for performance
  // reasons.
  eid_t vertices[VWARP_DEFAULT_BATCH_SIZE + 2];
  cost_t cost[VWARP_DEFAULT_BATCH_SIZE];
} vwarp_mem_t;

PRIVATE error_t check_special_cases(graph_t* graph, vid_t src_id,
                                    cost_t* cost, bool* finished) {
  *finished = true;
  if ((graph == NULL) || (src_id >= graph->vertex_count) || (cost == NULL)) {
    return FAILURE;
  } else if (graph->vertex_count == 1) {
    cost[0] = 0;
    return SUCCESS;
  } else if (graph->edge_count == 0) {
    // Initialize cost to INFINITE and zero to the source node
    totem_memset(cost, INF_COST, graph->vertex_count, TOTEM_MEM_HOST);
    cost[src_id] = 0;
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
error_t initialize_gpu(const graph_t* graph, vid_t source_id, vid_t cost_len,
                       graph_t** graph_d, cost_t** cost_d, bool** finished_d) {
  // Allocate space on GPU
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);
  CHK_SUCCESS(totem_malloc(cost_len * sizeof(cost_t), TOTEM_MEM_DEVICE,
                          reinterpret_cast<void**>(cost_d)), err_free_graph_d);
  // Initialize cost to INFINITE.
  totem_memset(*cost_d, INF_COST, cost_len, TOTEM_MEM_DEVICE);
  // For the source vertex, initialize cost.
  totem_memset(&((*cost_d)[source_id]), (cost_t)0, 1, TOTEM_MEM_DEVICE);
  // Allocate the termination flag
  CHK_SUCCESS(totem_malloc(sizeof(bool), TOTEM_MEM_DEVICE,
                           reinterpret_cast<void**>(finished_d)),
              err_free_cost_d_graph_d);
  return SUCCESS;

  err_free_cost_d_graph_d:
    totem_free(cost_d, TOTEM_MEM_DEVICE);
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
error_t finalize_gpu(graph_t* graph_d, bool* finished_d,
                     cost_t* cost_d, cost_t* cost) {
  CHK_CU_SUCCESS(cudaMemcpy(cost, cost_d, graph_d->vertex_count *
                            sizeof(cost_t), cudaMemcpyDeviceToHost), err);
  graph_finalize_device(graph_d);
  totem_free(finished_d, TOTEM_MEM_DEVICE);
  totem_free(cost_d, TOTEM_MEM_DEVICE);
  return SUCCESS;
 err:
  return FAILURE;
}

/* This comment describes implementation details of the next two functions.
 * Modified from [Harish07].
 * Breadth First Search
 * This implementation uses level synchronization. BFS traverses the graph
 * in levels; once a level is visited it is not visited again. The BFs frontier
 * corresponds to all the nodes being processed at the current level.
 * Each thread processes a vertex (in the following text these terms are used
 * interchangeably). An integer array, cost_d, stores the minimal number of 
 * edges from the source vertex to each vertex. The cost for vertices that have
 * not been visited yet is INFINITE. In each iteration, each vertex checks if it
 * belongs to the current level by verifying its own cost. If it does, it
 * updates its not yet visited neighbors. If the cost of, at least, one neighbor
 * is updated, the variable finished_d is set to false and there will be another
 * iteration.
 */
__global__
void bfs_kernel(graph_t graph, cost_t level, bool* finished, cost_t* cost) {
  const vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) return;
  if (cost[vertex_id] != level) return;
  for (eid_t i = graph.vertices[vertex_id];
       i < graph.vertices[vertex_id + 1]; i++) {
    const vid_t neighbor_id = graph.edges[i];
    if (cost[neighbor_id] == INF_COST) {
      // Threads may update finished and the same position in the cost array
      // concurrently. It does not affect correctness since all
      // threads would update with the same value.
      *finished = false;
      cost[neighbor_id] = level + 1;
    }
  }  // for
}

/**
 * The neighbors processing function. This function sets the level of the
 * neighbors' vertex to one level more than the vertex. The assumption is that
 * the threads of a warp invoke this function to process the warp's batch of
 * work. In each iteration of the for loop, each thread processes a neighbor.
 * For example, thread 0 in the warp processes neighbors at indices 0,
 * VWARP_DEFAULT_WARP_WIDTH, (2 * VWARP_DEFAULT_WARP_WIDTH) etc. in the edges
 * array, while thread 1 in the warp processes neighbors 1,
 * (1 + VWARP_DEFAULT_WARP_WIDTH), (1 + 2 * VWARP_DEFAULT_WARP_WIDTH) and so on.
*/
__device__
void vwarp_process_neighbors(vid_t warp_offset, vid_t neighbor_count,
                             vid_t* neighbors, cost_t* cost, cost_t level,
                             bool* finished) {
  for (vid_t i = warp_offset; i < neighbor_count;
      i += VWARP_DEFAULT_WARP_WIDTH) {
    vid_t neighbor_id = neighbors[i];
    if (cost[neighbor_id] == INF_COST) {
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
void vwarp_bfs_kernel(graph_t graph, cost_t level, bool* finished,
                      cost_t* cost, uint32_t thread_count) {
  if (THREAD_GLOBAL_INDEX >= thread_count) return;
  vid_t warp_offset = THREAD_GLOBAL_INDEX % VWARP_DEFAULT_WARP_WIDTH;
  vid_t warp_id     = THREAD_GLOBAL_INDEX / VWARP_DEFAULT_WARP_WIDTH;

  __shared__ vwarp_mem_t shared_memory[(MAX_THREADS_PER_BLOCK /
                                        VWARP_DEFAULT_WARP_WIDTH)];
  vwarp_mem_t* my_space = &shared_memory[THREAD_BLOCK_INDEX /
                                         VWARP_DEFAULT_WARP_WIDTH];

  // copy my work to local space
  vid_t v_ = warp_id * VWARP_DEFAULT_BATCH_SIZE;
  vwarp_memcpy(my_space->cost, &cost[v_], VWARP_DEFAULT_BATCH_SIZE,
               warp_offset);
  vwarp_memcpy(my_space->vertices, &(graph.vertices[v_]),
               VWARP_DEFAULT_BATCH_SIZE + 1, warp_offset);

  // iterate over my work
  for (vid_t v = 0; v < VWARP_DEFAULT_BATCH_SIZE; v++) {
    if (my_space->cost[v] == level) {
      vid_t neighbor_count = my_space->vertices[v + 1] - my_space->vertices[v];
      vid_t* neighbors = &(graph.edges[my_space->vertices[v]]);
      vwarp_process_neighbors(warp_offset, neighbor_count, neighbors, cost,
                              level, finished);
    }
  }
}

__host__
error_t bfs_vwarp_gpu(graph_t* graph, vid_t source_id, cost_t* cost) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, source_id, cost, &finished);
  if (finished) return rc;

  // Create and initialize state on GPU
  graph_t* graph_d;
  cost_t* cost_d;
  bool* finished_d;
  CHK_SUCCESS(initialize_gpu(graph, source_id,
                             vwarp_default_state_length(graph->vertex_count),
                             &graph_d, &cost_d, &finished_d), err_free_all);

  // {} used to limit scope and avoid problems with error handles.
  {
  // Configure the kernel's threads and on-chip memory. On-ship memory is
  // configured as shared memory rather than L1 cache.
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(vwarp_default_thread_count(graph->vertex_count), blocks,
                   threads_per_block);
  cudaFuncSetCacheConfig(vwarp_bfs_kernel, cudaFuncCachePreferShared);
  bool finished = false;
  // while the current level has vertices to be processed.
  for (cost_t level = 0; !finished; level++) {
    CHK_CU_SUCCESS(cudaMemset(finished_d, true, 1), err_free_all);
    vwarp_bfs_kernel<<<blocks, threads_per_block>>>
      (*graph_d, level, finished_d, cost_d,
       vwarp_default_thread_count(graph->vertex_count));
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
    CHK_CU_SUCCESS(cudaMemcpy(&finished, finished_d, sizeof(bool),
                              cudaMemcpyDeviceToHost), err_free_all);
  }
  }

  CHK_SUCCESS(finalize_gpu(graph_d, finished_d, cost_d, cost), err_free_all);
  return SUCCESS;

  // error handlers
  err_free_all:
    totem_free(finished_d, TOTEM_MEM_DEVICE);
    totem_free(cost_d, TOTEM_MEM_DEVICE);
    graph_finalize_device(graph_d);
    return FAILURE;
}

__host__
error_t bfs_gpu(graph_t* graph, vid_t source_id, cost_t* cost) {
  // Check for special cases.
  bool finished = false;
  error_t rc = check_special_cases(graph, source_id, cost, &finished);
  if (finished) return rc;

  // Create and initialize state on GPU.
  graph_t* graph_d;
  cost_t* cost_d;
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
  for (cost_t level = 0; !finished; level++) {
    CHK_CU_SUCCESS(cudaMemset(finished_d, true, 1), err_free_all);
    // for each vertex V in parallel do
    bfs_kernel<<<blocks, threads_per_block>>>(*graph_d, level, finished_d,
                                              cost_d);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
    CHK_CU_SUCCESS(cudaMemcpy(&finished, finished_d, sizeof(bool),
                              cudaMemcpyDeviceToHost), err_free_all);
  }}

  // We are done, get the results back and clean up state.
  CHK_SUCCESS(finalize_gpu(graph_d, finished_d, cost_d, cost), err_free_all);
  return SUCCESS;

  // error handlers
  err_free_all:
    totem_free(finished_d, TOTEM_MEM_DEVICE);
    totem_free(cost_d, TOTEM_MEM_DEVICE);
    graph_finalize_device(graph_d);
    return FAILURE;
}

PRIVATE bitmap_t initialize_cpu(graph_t* graph, vid_t source_id,
                                cost_t* cost) {
  // Initialize cost to INFINITE and create the vertices bitmap.
  totem_memset(cost, INF_COST, graph->vertex_count, TOTEM_MEM_HOST);
  bitmap_t visited = bitmap_init_cpu(graph->vertex_count);

  // Initialize the state of the source vertex.
  cost[source_id] = 0;
  bitmap_set_cpu(visited, source_id);
  return visited;
}

__host__
error_t bfs_cpu(graph_t* graph, vid_t source_id, cost_t* cost) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, source_id, cost, &finished);
  if (finished) return rc;

  bitmap_t visited = initialize_cpu(graph, source_id, cost);

  finished = false;
  // Within the following code segment, all threads execute in parallel the
  // same code (similar to a cuda kernel)
  OMP(omp parallel) {
    // level is a local variable to each thread, having a separate copy per
    // thread reduces the overhead of cache coherency protocol compared to
    // the case where level is shared
    cost_t level = 0;
    // while the current level has vertices to be processed.
    while (!finished) {
      // The following barrier is necessary to ensure that all threads have
      // checked the while condition above using the same "finished" value
      // that resulted from the previous iteration before it is initialized
      // again for the next one.
      OMP(omp barrier)

      // This "single" clause ensures that only one thread sets the variable.
      // Note that this close has an implicit barrier (i.e., all threads will
      // block until the variable is set by the responsible thread)
      OMP(omp single)
      finished = true;

      // The "for" clause instructs openmp to run the loop in parallel. Each
      // thread will be assigned a chunk of work depending on the chosen OMP
      // scheduling algorithm. The reduction clause tells openmp to define a
      // private temporary variable for each thread, and reduce them in the
      // end using an "and" operator and store the value in "finished". Similar
      // to the argument above, this improves performance by reducing cache
      // coherency overhead. The "runtime" scheduling clause defer the choice
      // of thread scheduling algorithm to the choice of the client, either
      // via OS environment variable or omp_set_schedule interface.
      OMP(omp for schedule(runtime) reduction(& : finished))
      for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
        if (cost[vertex_id] != level) continue;
        for (eid_t i = graph->vertices[vertex_id];
             i < graph->vertices[vertex_id + 1]; i++) {
          const vid_t neighbor_id = graph->edges[i];
          if (!bitmap_is_set(visited, neighbor_id)) {
            if (bitmap_set_cpu(visited, neighbor_id)) {
              finished = false;
              cost[neighbor_id] = level + 1;
            }
          }
        }
      }
      level++;
    }
  }  // omp parallel
  bitmap_finalize_cpu(visited);
  return SUCCESS;
}

PRIVATE void allocate_frontiers(graph_t* graph, vid_t** currF, vid_t** nextF,
                                vid_t*** localFs) {
  int thread_count = omp_get_max_threads();
  // Allocate a local queue for each thread.
  *localFs = reinterpret_cast<vid_t**>(malloc(thread_count * sizeof(vid_t*)));
  for (int tid = 0; tid < thread_count; tid++) {
    // allocate space assuming the worst case: all the vertices are
    // pushed to the local queue of a thread.
    // TODO(abdullah): reduce the memory footprint of the local stacks
    //                 (e.g., coarse-grained dynamic expansion of stack size)
    (*localFs)[tid] = reinterpret_cast<vid_t*>(malloc(graph->vertex_count *
                                               sizeof(vid_t)));
    assert((*localFs)[tid]);
  }
  *currF = reinterpret_cast<vid_t*>(malloc(graph->vertex_count *
                                    sizeof(vid_t)));
  *nextF = reinterpret_cast<vid_t*>(malloc(graph->vertex_count *
                                    sizeof(vid_t)));
  assert(*currF && *nextF);
}

PRIVATE void free_frontiers(vid_t* currF, vid_t* nextF, vid_t** localFs) {
  int thread_count = omp_get_max_threads();
  for (int tid = 0; tid < thread_count; tid++) {
    free(localFs[tid]);
  }
  free(localFs);
  free(currF);
  free(nextF);
}

/* Based on the implementation by Agarwal et al.
 * The implementation uses two arrays that maintains the current and next 
 * frontier. The current frontier array contains the vertices that are being 
 * visited in the current level. While the vertices in the current frontier
 * are being processed, their not-visited neighbors are stored in the next 
 * frontier array. Once the current level is done, the next frontier array 
 * becomes the current frontier array, and the processing of the next level
 * starts. To improve performance, this implementation uses what is called local
 * next frontier arrays: each thread has a local next array that in the end 
 * merged into the global next array. This improves performance by getting rid
 * of the required synchronization if the threads were to access the global next
 * array directly.
 */
__host__
error_t bfs_queue_cpu(graph_t* graph, vid_t source_id, cost_t* cost) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, source_id, cost, &finished);
  if (finished) return rc;
  bitmap_t visited = initialize_cpu(graph, source_id, cost);

  // Initialize queues
  vid_t currF_index = 0;
  vid_t nextF_index = 0;
  vid_t* currF = NULL;
  vid_t* nextF = NULL;
  vid_t** localFs = NULL;
  allocate_frontiers(graph, &currF, &nextF, &localFs);

  // Do level 0 separately and parallelize across neighbours.
  // Only the source node is active in level 0
  OMP(omp parallel for schedule(static))
  for (vid_t v = graph->vertices[source_id];
       v < graph->vertices[source_id + 1]; v++) {
    vid_t nbr = graph->edges[v];
    cost[nbr] = 1;
    bitmap_set_cpu(visited, nbr);
    nextF[__sync_fetch_and_add(&nextF_index, 1)] = nbr;
  }


  OMP(omp parallel) {
    // thread-local variables
    cost_t level        = 1;
    vid_t  localF_index = 0;
    vid_t* localF       = localFs[omp_get_thread_num()];

    // while the current level has vertices to be processed.
    while (nextF_index > 0) {
      // The following barrier is necessary to ensure that all threads have
      // checked the while condition before nextF_index is cleared for the next
      // round.
      OMP(omp barrier)

      // This "single" clause ensures that only one thread enters the
      // following block of code. Note that this close has an implicit
      // barrier.
      OMP(omp single) {
        // Swap the current with the next queue.
        vid_t* tmp = currF;
        currF = nextF;
        nextF = tmp;
        currF_index = nextF_index;
        nextF_index = 0;
      }
      localF_index = 0;

      // The "for" clause instructs openmp to run the loop in parallel. Each
      // thread will be assigned a chunk of work depending on the chosen
      // OMP scheduling algorithm. The "runtime" scheduling clause defer the
      // choice of thread scheduling algorithm to the choice of the client,
      // either via OS environment variable or omp_set_schedule interface.
      OMP(omp for schedule(runtime))
      for (vid_t q = 0; q < currF_index; q++) {
        vid_t v = currF[q];
        for (eid_t i = graph->vertices[v]; i < graph->vertices[v + 1]; i++) {
          const vid_t nbr = graph->edges[i];
          if (!bitmap_is_set(visited, nbr)) {
            if (bitmap_set_cpu(visited, nbr)) {
              cost[nbr] = level + 1;
              localF[localF_index++] = nbr;
            }
          }
        }
      }
      if (localF_index > 0) {
        vid_t idx = __sync_fetch_and_add(&nextF_index, localF_index);
        memcpy(&(nextF[idx]), localF, localF_index * sizeof(vid_t));
      }
      level++;

      // The following barrier is necessary to ensure that all threads see the
      // same nextF_index value that is being incremented by the localF_indices
      OMP(omp barrier)
    }
  }  // omp parallel
  bitmap_finalize_cpu(visited);
  free_frontiers(currF, nextF, localFs);
  return SUCCESS;
}

// A classic top down step that iterates over vertices in the frontier
// and tries to add their neighbours to the next frontier.
bool top_down_step(graph_t* graph, cost_t* cost, bitmap_t* visited,
                   frontier_state_t* state, cost_t level) {
  bool finished = true;

  // Iterate across all vertices in frontier.
  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    if (!bitmap_is_set(state->current, vertex_id)) continue;

    // Iterate across all neighbours of this vertex.
    for (eid_t i = graph->vertices[vertex_id];
         i < graph->vertices[vertex_id + 1]; i++) {
      const vid_t neighbor_id = graph->edges[i];

      // If already visited, ignore neighbour.
      if (!bitmap_is_set(*visited, neighbor_id)) {
        if (bitmap_set_cpu(*visited, neighbor_id)) {
          // If a new vertex is now visited, we have a new level
          // of frontier - we are not finished.
          finished = false;
          // Increment the level of this vertex.
          cost[neighbor_id] = level + 1;
        }
      }
    }
  }  // End of omp for

  return finished;
}

// A step that iterates across unvisited vertices and determines
// their status in the next frontier.
bool bottom_up_step(graph_t* graph, cost_t* cost, bitmap_t* visited,
                    frontier_state_t* state, cost_t level) {
  bool finished = true;

  // Iterate across all vertices.
  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    // Ignore vertices that have been visited.
    if (bitmap_is_set(*visited, vertex_id)) { continue; }

    // Iterate across the neighbours of this vertex.
    for (eid_t i = graph->vertices[vertex_id];
         i < graph->vertices[vertex_id + 1]; i++) {
      if (bitmap_is_set(state->current, graph->edges[i])) {
        // Add the vertex we are exploring to the next frontier.
        bitmap_set_cpu(*visited, vertex_id);
        finished = false;
        cost[vertex_id] = level + 1;
        break;
      }
    }  // End of neighbour check - vertex examined.
  }  // All vertices examined in level.
  return finished;
}

/* Similar to the regular BFS for cpu, the difference being choosing
 * a step for each level is now possible.
 * Based off of the work by Scott Beamer et al.
 * Searching for a Parent Instead of Fighting Over Children: A Fast 
 * Breadth-First Search Implementation for Graph500.
 * http://www.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-117.pdf
 */
__host__
error_t bfs_bu_cpu(graph_t* graph, vid_t source_id, cost_t* cost) {
  // TODO(scott): Make this a heuristic instead of a constant.
  const vid_t SMALL_THRESHOLD = graph->vertex_count/(16*16*16*16);

  // Check for special cases.
  bool finished = false;
  error_t rc = check_special_cases(graph, source_id, cost, &finished);
  if (finished) return rc;

  // Initialize frontier set-up.
  frontier_state_t state;
  frontier_init_cpu(&state, graph->vertex_count);

  // Initialize the bitmap.
  bitmap_t visited = initialize_cpu(graph, source_id, cost);

  // Start by assuming we are not done yet.
  finished = false;
  cost_t level = 0;

  // Complete a step for each level, while not finished.
  while (!finished) {
    // Update the frontier.
    frontier_update_bitmap_cpu(&state, visited);

    // Choose a top-down step if the frontier count is small.
    if (bitmap_count_cpu(state.current, graph->vertex_count)
                         < SMALL_THRESHOLD) {
      finished = top_down_step(graph, cost, &visited, &state, level);
    } else {
      // Choose a bottom-up step if the frontier count is large.
      finished = bottom_up_step(graph, cost, &visited, &state, level);
    }
    level++;
  }

  bitmap_finalize_cpu(visited);
  return SUCCESS;
}
