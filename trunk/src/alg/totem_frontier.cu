/**
 * This file contains an implementation of the frontier interface which is used
 * by some traversal-based algorithms such as BFS, Graph500 and Betweenness 
 * Centrality
 *
 *  Created on: 2013-08-3
 *  Author: Abdullah Gharaibeh
 */

#include "totem_alg.h"
#include "totem_engine.cuh"

template<int THREADS_PER_BLOCK>
PRIVATE __global__ void
frontier_build_kernel(frontier_state_t state, const cost_t level,
                      const cost_t* __restrict cost) {
  const vid_t v = THREAD_GLOBAL_INDEX;
  if (v >= state.len) return;

  __shared__ vid_t queue_l[THREADS_PER_BLOCK];
  __shared__ vid_t count_l;
  count_l = 0;
  __syncthreads();

  if (cost[v] == level) {
    int index = atomicAdd(&count_l, 1);
    queue_l[index] = v;
  }
  __syncthreads();

  if (THREAD_BLOCK_INDEX >= count_l) return;

  __shared__ int index;
  if (THREAD_BLOCK_INDEX == 0) {
    index = atomicAdd(state.count, count_l);
  }
  __syncthreads();

  state.list[index + THREAD_BLOCK_INDEX] = queue_l[THREAD_BLOCK_INDEX];  
}

#ifdef FEATURE_SM35
PRIVATE __global__ void
frontier_update_boundaries_kernel(frontier_state_t state,
                                  const eid_t* __restrict vertices) {
  const vid_t index = THREAD_GLOBAL_INDEX;
  if (index >= *state.count) return;

  const vid_t* __restrict frontier = state.list;
  vid_t* boundaries = &state.boundaries[1];

  vid_t v = frontier[index];
  vid_t nbr_count = vertices[v + 1] - vertices[v];

  if (nbr_count > 7 && nbr_count < 32) {
    atomicMin(&boundaries[0], index);
  }
  if (nbr_count > 31 && nbr_count < 128) {
    atomicMin(&boundaries[1], index);
  }
  if (nbr_count > 127 && nbr_count < 256) {
    atomicMin(&boundaries[2], index);
  }
  if (nbr_count > 255 && nbr_count < 1024) {
    atomicMin(&boundaries[3], index);
  }
  if (nbr_count > 1023 && nbr_count < (2 * 1024)) {
    atomicMin(&boundaries[4], index);
  }
  if (nbr_count >= (2 * 1024)) {
    atomicMin(&boundaries[5], index);
  }
}

PRIVATE __global__ void
frontier_update_boundaries_launch_kernel(frontier_state_t state,
                                         const eid_t* __restrict vertices) {
  if (THREAD_GLOBAL_INDEX > 0 || (*state.count == 0)) return;
  dim3 blocks;
  kernel_configure(*state.count, blocks);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  frontier_update_boundaries_kernel
    <<<blocks, DEFAULT_THREADS_PER_BLOCK, 0, stream>>>(state, vertices);
}

PRIVATE __global__ void
frontier_init_boundaries_kernel(frontier_state_t state) {
  const vid_t index = THREAD_GLOBAL_INDEX;
  if (index >= FRONTIER_BOUNDARY_COUNT) return;
  state.boundaries[index + 1] = *state.count;
}

#endif /* FEATURE_SM35  */

void frontier_update_list_gpu(frontier_state_t* state,
                              vid_t level, const cost_t* cost, 
                              const cudaStream_t stream) {
  cudaMemsetAsync(state->count, 0, sizeof(vid_t), stream);
  dim3 blocks;
  kernel_configure(state->len, blocks);
  frontier_build_kernel<DEFAULT_THREADS_PER_BLOCK>
    <<<blocks, DEFAULT_THREADS_PER_BLOCK, 0, stream>>>(*state, level, cost);
  CALL_CU_SAFE(cudaGetLastError());
}

#ifdef FEATURE_SM35
void frontier_update_boundaries_gpu(frontier_state_t* state, 
                                    const graph_t* graph, 
                                    const cudaStream_t stream) {
  if (engine_sorted()) {
    // If the vertices are sorted by degree, build the boundaries array
    // to optimize thread scheduling when launching the traversal kernel
    dim3 blocks;
    kernel_configure(FRONTIER_BOUNDARY_COUNT, blocks);
    frontier_init_boundaries_kernel
      <<<blocks, DEFAULT_THREADS_PER_BLOCK, 0, stream>>>(*state);
    CALL_CU_SAFE(cudaGetLastError());

    frontier_update_boundaries_launch_kernel
      <<<1, 1, 0, stream>>>(*state, graph->vertices);
    CALL_CU_SAFE(cudaGetLastError());
  }
}
#endif /* FEATURE_SM35 */

vid_t frontier_update_bitmap_gpu(frontier_state_t* state, bitmap_t visited,
                                 cudaStream_t stream) {
  bitmap_t tmp = state->current;
  state->current = state->visited_last;
  state->visited_last = tmp;
    bitmap_diff_copy_gpu(visited, state->current, state->visited_last, 
                         state->len, stream);
  return 0;
}
vid_t frontier_update_bitmap_cpu(frontier_state_t* state, bitmap_t visited) {
  // Build the frontier bitmap
  bitmap_t tmp = state->current;
  state->current = state->visited_last;
  state->visited_last = tmp;
  bitmap_diff_copy_cpu(visited, state->current, state->visited_last, 
                       state->len);
  return 0;
}

void frontier_reset_gpu(frontier_state_t* state) {
  bitmap_reset_gpu(state->current, state->len);
  bitmap_reset_gpu(state->visited_last, state->len);
  totem_memset(state->count, (vid_t)0, 1, TOTEM_MEM_DEVICE);  
}
void frontier_reset_cpu(frontier_state_t* state) {
  bitmap_reset_cpu(state->current, state->len);
  bitmap_reset_cpu(state->visited_last, state->len);
}

void frontier_init_gpu(frontier_state_t* state, vid_t vertex_count) {
  assert(state);
  state->len = vertex_count;
  state->current = bitmap_init_gpu(vertex_count);
  state->visited_last = bitmap_init_gpu(vertex_count);
  CALL_SAFE(totem_calloc(sizeof(vid_t), TOTEM_MEM_DEVICE, 
                         (void **)&state->count));
  CALL_SAFE(totem_calloc(vertex_count * sizeof(vid_t), TOTEM_MEM_DEVICE,
                         (void **)&state->list));
  state->list_len = vertex_count;
  if (engine_partition_algorithm() == PAR_SORTED_ASC) {
    // LOW-degree vertices were placed on the GPU. Since there is typically
    // many of them, and the GPU has limited memory, we restrict the frontier
    // array size. If the frontier in a specific level was longer, then the 
    // algorithm will not build a frontier array, and will iterate over all
    // the vertices.
    state->list_len = vertex_count * TRV_MAX_FRONTIER_LEN;
  }
#ifdef FEATURE_SM35
  if (engine_sorted()) {
    CALL_SAFE(totem_calloc((FRONTIER_BOUNDARY_COUNT + 1) * sizeof(vid_t), 
                           TOTEM_MEM_DEVICE, (void**)&state->boundaries));
  }
#endif /* FEATURE_SM35 */
}
void frontier_init_cpu(frontier_state_t* state, vid_t vertex_count) {
  state->len = vertex_count;
  state->current = bitmap_init_cpu(vertex_count);
  state->visited_last = bitmap_init_cpu(vertex_count);
}

void frontier_finalize_gpu(frontier_state_t* state) {
  assert(state);
#ifdef FEATURE_SM35
    if (engine_sorted()) totem_free(state->boundaries, TOTEM_MEM_DEVICE);
#endif /* FEATURE_SM35  */
  totem_free(state->list, TOTEM_MEM_DEVICE);
  totem_free(state->count, TOTEM_MEM_DEVICE);
  bitmap_finalize_gpu(state->visited_last);
  bitmap_finalize_gpu(state->current);
}
void frontier_finalize_cpu(frontier_state_t* state) {
  bitmap_finalize_cpu(state->visited_last);
  bitmap_finalize_cpu(state->current);
}
