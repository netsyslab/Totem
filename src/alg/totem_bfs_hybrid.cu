/**
 * This file contains an implementation of the breadth-first search (BFS) graph
 * search algorithm using the totem framework
 *
 *  Created on: 2012-01-30
 *  Author: Abdullah Gharaibeh
 */

#include "totem_alg.h"
#include "totem_engine.cuh"

/**
 * per-partition specific state
 */
typedef struct bfs_state_s {
  cost_t*   cost;        // one slot per vertex in the partition
  bitmap_t  visited[MAX_PARTITION_COUNT]; // a list of bitmaps one for each 
                                          // remote partition
  bool*     finished;    // points to Totem's finish flag
  cost_t    level;       // current level being processed by the partition
  vid_t* frontier_list;  // maintains the list of vertices that belong to the
                         // current frontier being processed (GPU-based 
                         // partitions only)
  vid_t frontier_max_count; // maximum number of vertices that the frontier 
                            // buffer can hold (GPU-based partitions only)
  vid_t* frontier_count; // number of vertices in the frontier (GPU-based
                         // partitions only)
  bitmap_t frontier;     // current frontier bitmap
  bitmap_t visited_last; // a bitmap of the visited vertices before the start of
                         // the previous round. This is used to quickly compute
                         // the frontier bitmap of the current round by xor-ing 
                         // this bitmap with the visited bitmap (a bitmap of the
                         // visited untill after the end of the previous round
  bool   do_scatter;     // a flag used in the cpu-based partition that 
                         // idicates whether or not to execute the scatter
                         // function. this is set by the CPU kernel as a 
                         // signal to the CPU scatter callback.
} bfs_state_t;

/**
 * state shared between all partitions
 */
typedef struct bfs_global_state_s {
  cost_t*   cost;   // final output buffer
  cost_t*   cost_h; // Used as a temporary buffer to receive the final 
                    // result copied back from GPU partitions before being
                    // copied again to the final output buffer
                    // TODO(abdullah): push this buffer to be managed by Totem
  vid_t     src;    // source vertex id (the id after partitioning)  
  bool      gpu_to_cpu_updates; // a flag set by the gpu-based partitions
                                // are read by the cpu-based one. The flag
                                // is set to true when any gpu-based partition
                                // has potentially sent updates to the cpu-based
                                // one. This is a way to reduce communication 
                                // overhead by avoiding to call the cpu-side 
                                // scatter function
} bfs_global_state_t;
PRIVATE bfs_global_state_t state_g = {NULL, NULL, 0, false};

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
*/
PRIVATE error_t check_special_cases(vid_t src, cost_t* cost, bool* finished) {
  *finished = true;
  if((src >= engine_vertex_count()) || (cost == NULL)) {
    return FAILURE;
  } else if(engine_vertex_count() == 1) {
    cost[0] = 0;
    return SUCCESS;
  } else if(engine_edge_count() == 0) {
    // Initialize cost to INFINITE.
    totem_memset(cost, INF_COST, engine_vertex_count(), TOTEM_MEM_HOST);
    cost[src] = 0;
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

PRIVATE __attribute__((always_inline)) void 
bfs_cpu_process_vertex(graph_t* subgraph, bfs_state_t* state, vid_t v,
                       int pid, bool &finished) {
  for (eid_t i = subgraph->vertices[v]; 
       i < subgraph->vertices[v + 1]; i++) {
    int nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
    vid_t nbr = GET_VERTEX_ID(subgraph->edges[i]);
    bitmap_t visited = state->visited[nbr_pid];
    if (!bitmap_is_set(visited, nbr)) {
      if (bitmap_set_cpu(visited, nbr)) {
        if (nbr_pid == pid) {
          state->cost[nbr] = state->level + 1;
        }
        finished = false;
      }
    }
  }
}

PRIVATE void 
bfs_cpu_sparse_frontier(graph_t* subgraph, bfs_state_t* state, int pid) {
  vid_t words = bitmap_bits_to_words(subgraph->vertex_count);
  // The "runtime" scheduling clause defer the choice of thread scheduling
  // algorithm to the choice of the client, either via OS environment variable
  // or omp_set_schedule interface.
  bool finished = true;
  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t word_index = 0; word_index < words; word_index++) {
    if (!state->frontier[word_index]) continue;
    vid_t v = word_index * BITMAP_BITS_PER_WORD;
    vid_t last_v = (word_index + 1) * BITMAP_BITS_PER_WORD;
    if (last_v > subgraph->vertex_count) last_v = subgraph->vertex_count;
    for (; v < last_v; v++) {
      if (!bitmap_is_set(state->frontier, v)) continue;
      bfs_cpu_process_vertex(subgraph, state, v, pid, finished);
    }
  }
  if (!finished) *(state->finished) = false;
}

PRIVATE void 
bfs_cpu_dense_frontier(graph_t* subgraph, bfs_state_t* state, int pid) {
  bool finished = true;
  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    if (state->cost[v] != state->level) continue;
    bfs_cpu_process_vertex(subgraph, state, v, pid, finished);    
  }
  if (!finished) *(state->finished) = false;
}

void bfs_cpu(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  graph_t* subgraph = &par->subgraph;

  // Indicate whether the scatter function should be executed or not in the next
  // round. The gpu_to_cpu_updates flag is set to false if no vertices in this 
  // partition were set remotely by any of the GPU partitions
  state->do_scatter = true;
  if (!state_g.gpu_to_cpu_updates) {
    state->do_scatter = false;
  }
  state_g.gpu_to_cpu_updates = false;

  // Build the frontier bitmap
  bitmap_t tmp = state->frontier;
  state->frontier = state->visited_last;
  state->visited_last = tmp;
  vid_t frontier_count = 
    bitmap_diff_copy_count_cpu(state->visited[par->id], state->frontier, 
                               state->visited_last, subgraph->vertex_count);
  if (frontier_count == 0) return;

  // If the number of active vertices is lower than a specific threshold, we 
  // consider the frontier sparse
  if (frontier_count <= (subgraph->vertex_count * 
                         TRV_FRONTIER_SPARSE_THRESHOLD)) {
    bfs_cpu_sparse_frontier(subgraph, state, par->id);
  } else {
    bfs_cpu_dense_frontier(subgraph, state, par->id);
  }
}

PRIVATE __global__ void
frontier_build_kernel(const vid_t vertex_count, const cost_t level,
                      const cost_t* __restrict cost,
                      vid_t* frontier_list, vid_t* frontier_count) {
  const vid_t v = THREAD_GLOBAL_INDEX;
  if (v >= vertex_count) return;

  __shared__ vid_t queue_l[MAX_THREADS_PER_BLOCK];
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
    index = atomicAdd(frontier_count, count_l);
  }
  __syncthreads();
  frontier_list[index + THREAD_BLOCK_INDEX] = queue_l[THREAD_BLOCK_INDEX];  
}

PRIVATE inline void frontier_build(partition_t* par, bfs_state_t* state) {
  cudaMemsetAsync(state->frontier_count, 0, sizeof(vid_t), par->streams[1]);
  dim3 blocks;
  kernel_configure(par->subgraph.vertex_count, blocks);
  frontier_build_kernel<<<blocks, DEFAULT_THREADS_PER_BLOCK, 0, 
    par->streams[1]>>>(par->subgraph.vertex_count, state->level, state->cost, 
                       state->frontier_list, state->frontier_count);
  CALL_CU_SAFE(cudaGetLastError());
}

/**
 * A warp-based implementation of the BFS kernel. Please refer to the
 * description of the warp technique for details. Also, please refer to
 * bfs_kernel for details on the BFS implementation.
 */
template<int VWARP_WIDTH, int VWARP_BATCH, bool USE_FRONTIER>
__global__
void bfs_kernel(partition_t par, bfs_state_t state) {
  vid_t count = par.subgraph.vertex_count;
  if (USE_FRONTIER) { // this is evaluated at compile time
    count = *state.frontier_count;
  }

  if (THREAD_GLOBAL_INDEX >= 
      vwarp_thread_count(count, VWARP_WIDTH, VWARP_BATCH)) return;

  const eid_t* __restrict vertices = par.subgraph.vertices;
  const vid_t* __restrict edges = par.subgraph.edges;
  const cost_t* __restrict cost = state.cost;
  const vid_t* __restrict frontier = state.frontier_list;

  // This flag is used to report the finish state of a block of threads. This
  // is useful to avoid having many threads writing to the global finished
  // flag, which can hurt performance (since "finished" is actually allocated
  // on the host, and each write will cause a transfer over the PCI-E bus)
  __shared__ bool finished_block;
  finished_block = true;
  __syncthreads();

  vid_t start_vertex = vwarp_block_start_vertex(VWARP_WIDTH, VWARP_BATCH) + 
    vwarp_warp_start_vertex(VWARP_WIDTH, VWARP_BATCH);
  vid_t end_vertex = start_vertex +
    vwarp_warp_batch_size(count, VWARP_WIDTH, VWARP_BATCH);
  int warp_offset = vwarp_thread_index(VWARP_WIDTH);

  for(vid_t i = start_vertex; i < end_vertex; i++) {
    vid_t v = i;
    if (USE_FRONTIER) { // this is evaluated at compile time
      v = frontier[i];
    }

    // if USE_FRONTIER is true, the if-statement will be removed by the compiler
    if (USE_FRONTIER || (cost[v] == state.level)) { 
      const eid_t nbr_count = vertices[v + 1] - vertices[v];
      const vid_t* __restrict nbrs = &(edges[vertices[v]]);
      for(vid_t i = warp_offset; i < nbr_count; i += VWARP_WIDTH) {
        int nbr_pid = GET_PARTITION_ID(nbrs[i]);
        vid_t nbr = GET_VERTEX_ID(nbrs[i]);
        bitmap_t visited = state.visited[nbr_pid];
        if (!bitmap_is_set(visited, nbr)) {
          if (bitmap_set_gpu(visited, nbr)) {
            if ((nbr_pid == par.id) && state.cost[nbr] == INF_COST) { 
              state.cost[nbr] = state.level + 1;
            }
            finished_block = false;
          }
        }
      }
    }
  }
  __syncthreads();
  if (!finished_block && THREAD_BLOCK_INDEX == 0) *state.finished = false;
}

template<int VWARP_WIDTH, int VWARP_BATCH, bool USE_FRONTIER>
PRIVATE void bfs_gpu_launch(partition_t* par, bfs_state_t* state, 
                            vid_t vertex_count) {
  const int threads = MAX_THREADS_PER_BLOCK;
  dim3 blocks;
  kernel_configure(vwarp_thread_count(vertex_count, VWARP_WIDTH, VWARP_BATCH),
                   blocks, threads);
  bfs_kernel<VWARP_WIDTH, VWARP_BATCH, USE_FRONTIER>
    <<<blocks, threads, 0, par->streams[1]>>>(*par, *state);
  CALL_CU_SAFE(cudaGetLastError());
}

typedef void(*bfs_gpu_func_t)(partition_t*, bfs_state_t*, vid_t);
PRIVATE const bfs_gpu_func_t BFS_GPU_FUNC[][2] = {
  {
    // RANDOM algorithm
    bfs_gpu_launch<VWARP_MEDIUM_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE, false>,
    bfs_gpu_launch<VWARP_MEDIUM_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE, true>
  },
  {
    // HIGH partitioning
    bfs_gpu_launch<VWARP_MEDIUM_WARP_WIDTH, VWARP_LARGE_BATCH_SIZE, false>,
    bfs_gpu_launch<VWARP_MEDIUM_WARP_WIDTH, VWARP_LARGE_BATCH_SIZE, true>
  },
  {
    // LOW partitioning
    bfs_gpu_launch<MAX_THREADS_PER_BLOCK, VWARP_MEDIUM_BATCH_SIZE, false>,
    bfs_gpu_launch<MAX_THREADS_PER_BLOCK, VWARP_MEDIUM_BATCH_SIZE, true>
  }
};

PRIVATE void bfs_gpu(partition_t* par, bfs_state_t* state) {
  // Build the frontier's bitmap (this is a blocking, but fast computation)
  bitmap_t tmp = state->frontier;
  state->frontier = state->visited_last;
  state->visited_last = tmp;
  vid_t frontier_count = 
    bitmap_diff_copy_count_gpu(state->visited[par->id], state->frontier, 
                               state->visited_last, par->subgraph.vertex_count,
                               state->frontier_count, par->streams[1]);
  if (frontier_count == 0) return;

  vid_t vertex_count = par->subgraph.vertex_count;
  int use_frontier = (int)(frontier_count <= state->frontier_max_count);
  if (use_frontier) {
    frontier_build(par, state);
    vertex_count = frontier_count;
  }
  state_g.gpu_to_cpu_updates = true;

  int par_alg = engine_partition_algorithm();
  BFS_GPU_FUNC[par_alg][use_frontier](par, state, vertex_count);
}

PRIVATE void bfs(partition_t* par) {
  if (par->subgraph.vertex_count == 0) return;
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  if (par->processor.type == PROCESSOR_CPU) {
    bfs_cpu(par);
  } else if (par->processor.type == PROCESSOR_GPU) {  
    bfs_gpu(par, state);
  } else {
    assert(false);
  }
  state->level++;
}

PRIVATE inline void bfs_scatter_cpu(grooves_box_table_t* inbox, 
                                    bfs_state_t* state, bitmap_t visited) {
  if (!state->do_scatter) return;
  bitmap_t remotely_visited = (bitmap_t)inbox->push_values;
  OMP(omp parallel for schedule(runtime))
  for (vid_t index = 0; index < inbox->count; index++) {
    if (bitmap_is_set(remotely_visited, index)) {
      vid_t vid = inbox->rmt_nbrs[index];
      if (!bitmap_is_set(visited, vid)) {
        bitmap_set_cpu(visited, vid);
        state->cost[vid] = state->level;
      }
    }
  }
}

__global__ void bfs_scatter_kernel(grooves_box_table_t inbox, bfs_state_t state,
                                   bitmap_t visited) {
  vid_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) return;
  __syncthreads();
  bitmap_t rmt_visited = (bitmap_t)inbox.push_values;
  vid_t vid = inbox.rmt_nbrs[index];
  if (bitmap_is_set(rmt_visited, index) &&
      !bitmap_is_set(visited, vid)) {
    bitmap_set_gpu(visited, vid);
    state.cost[vid] = state.level;
  }
}

PRIVATE void bfs_scatter(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_CPU) {
      bfs_scatter_cpu(inbox, state, state->visited[par->id]);
    } else if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks;
      kernel_configure(inbox->count, blocks);
      bfs_scatter_kernel<<<blocks, DEFAULT_THREADS_PER_BLOCK, 0, 
        par->streams[1]>>>(*inbox, *state, state->visited[par->id]);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(false);
    }
  }
}

PRIVATE void bfs_aggregate(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  bfs_state_t* state    = (bfs_state_t*)par->algo_state;
  graph_t*     subgraph = &par->subgraph;
  cost_t*    src_cost = NULL;
  if (par->processor.type == PROCESSOR_CPU) {
    src_cost = state->cost;
  } else if (par->processor.type == PROCESSOR_GPU) {
    assert(state_g.cost_h);
    CALL_CU_SAFE(cudaMemcpy(state_g.cost_h, state->cost, 
                            subgraph->vertex_count * sizeof(cost_t),
                            cudaMemcpyDefault));
    src_cost = state_g.cost_h;
  } else {
    assert(false);
  }
  // aggregate the results
  assert(state_g.cost);
  OMP(omp parallel for schedule(runtime))
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    state_g.cost[par->map[v]] = src_cost[v];
  }
}

__global__ void bfs_init_kernel(bitmap_t visited, vid_t src) {
  if (THREAD_GLOBAL_INDEX != 0) return;
  bitmap_set_gpu(visited, src);
}

PRIVATE inline void bfs_init_gpu(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  state->frontier = bitmap_init_gpu(par->subgraph.vertex_count);
  state->visited_last = bitmap_init_gpu(par->subgraph.vertex_count);
  state->visited[par->id] = bitmap_init_gpu(par->subgraph.vertex_count);
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      state->visited[pid] = (bitmap_t)par->outbox[pid].push_values;
      bitmap_reset_gpu(state->visited[pid], par->outbox[pid].count);
    }
  }
  // set the source vertex as visited
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    bfs_init_kernel<<<1, 1, 0, par->streams[1]>>>
      (state->visited[par->id], GET_VERTEX_ID(state_g.src));
    CALL_CU_SAFE(cudaGetLastError());
  }
  CALL_SAFE(totem_calloc(sizeof(vid_t), TOTEM_MEM_DEVICE, 
                         (void **)&state->frontier_count));
  state->frontier_max_count = par->subgraph.vertex_count;
  if (engine_partition_algorithm() == PAR_SORTED_ASC) {
    // High-degree vertices were placed on the GPU. Since there is typically
    // many of them, and the GPU has limited memory, we restrict the frontier
    // array size. If the frontier in a specific level was longer, then the 
    // algorithm will not build a frontier array, and will iterate over all
    // the vertices.
    state->frontier_max_count = 
      par->subgraph.vertex_count * TRV_MAX_FRONTIER_LEN;
  }
  CALL_SAFE(totem_calloc(state->frontier_max_count * sizeof(vid_t), 
                         TOTEM_MEM_DEVICE, (void **)&state->frontier_list));
}

PRIVATE inline void bfs_init_cpu(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  state->frontier = bitmap_init_cpu(par->subgraph.vertex_count);
  state->visited_last = bitmap_init_cpu(par->subgraph.vertex_count);
  state->visited[par->id] = bitmap_init_cpu(par->subgraph.vertex_count);
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      state->visited[pid] = (bitmap_t)par->outbox[pid].push_values;
      bitmap_reset_cpu(state->visited[pid], par->outbox[pid].count);
    }
  }
  // set the source vertex as visited
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    bitmap_set_cpu(state->visited[par->id], GET_VERTEX_ID(state_g.src));
  }
}

PRIVATE void bfs_init(partition_t* par) {
  if (par->subgraph.vertex_count == 0) return;
  bfs_state_t* state = (bfs_state_t*)calloc(1, sizeof(bfs_state_t));
  assert(state);
  par->algo_state = state;
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_CPU) {
    bfs_init_cpu(par);
  } else if (par->processor.type == PROCESSOR_GPU) {
    type = TOTEM_MEM_DEVICE;
    bfs_init_gpu(par);
  } else {
    assert(false);
  }
  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(cost_t), type, 
                         (void**)&(state->cost)));
  totem_memset(state->cost, INF_COST, par->subgraph.vertex_count, type, 
               par->streams[1]);
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    // For the source vertex, initialize cost.    
    totem_memset(&((state->cost)[GET_VERTEX_ID(state_g.src)]), (cost_t)0, 1, 
                 type, par->streams[1]);
  }
  state->finished = engine_get_finished_ptr(par->id);
  state->level = 0;
}

PRIVATE void bfs_finalize(partition_t* par) {
  if (par->subgraph.vertex_count == 0) return;
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_CPU) {
    bitmap_finalize_cpu(state->visited[par->id]);
    bitmap_finalize_cpu(state->visited_last);
    bitmap_finalize_cpu(state->frontier);
  } else if (par->processor.type == PROCESSOR_GPU) {
    bitmap_finalize_gpu(state->visited[par->id]);
    bitmap_finalize_gpu(state->visited_last);
    bitmap_finalize_gpu(state->frontier);
    type = TOTEM_MEM_DEVICE;
    totem_free(state->frontier_count, TOTEM_MEM_DEVICE);    
    totem_free(state->frontier_list, TOTEM_MEM_DEVICE);    
  } else {
    assert(false);
  }
  totem_free(state->cost, type);
  free(state);
  par->algo_state = NULL;
}

error_t bfs_hybrid(vid_t src, cost_t* cost) {
  // check for special cases
  bool finished = false;
  error_t rc = check_special_cases(src, cost, &finished);
  if (finished) return rc;

  // initialize the global state
  state_g.cost = cost;
  state_g.src  = engine_vertex_id_in_partition(src);

  // initialize the engine
  engine_config_t config = {
    NULL, bfs, bfs_scatter, NULL, bfs_init, bfs_finalize, bfs_aggregate, 
    GROOVES_PUSH
  };
  engine_config(&config);
  if (engine_largest_gpu_partition()) {
    CALL_SAFE(totem_malloc(engine_largest_gpu_partition() * sizeof(cost_t), 
                           TOTEM_MEM_HOST, (void**)&state_g.cost_h));
  }
  engine_execute();

  // clean up and return
  if (engine_largest_gpu_partition()) {
    totem_free(state_g.cost_h, TOTEM_MEM_HOST);
  }
  memset(&state_g, 0, sizeof(bfs_global_state_t));
  return SUCCESS;
}
