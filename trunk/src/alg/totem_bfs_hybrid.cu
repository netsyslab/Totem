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
  cost_t*   cost;              // one slot per vertex in the partition
  bitmap_t  visited[MAX_PARTITION_COUNT]; // a list of bitmaps one for each 
                                          // remote partition
  bool*     finished;          // points to Totem's finish flag
  cost_t    level;             // current level being processed by the partition
  frontier_state_t frontier;   // frontier management state
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
} bfs_global_state_t;
PRIVATE bfs_global_state_t state_g = {NULL, NULL, 0};

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
  bitmap_t frontier = state->frontier.current;
  bool finished = true;
  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t word_index = 0; word_index < words; word_index++) {
    if (!frontier[word_index]) continue;
    vid_t v = word_index * BITMAP_BITS_PER_WORD;
    vid_t last_v = (word_index + 1) * BITMAP_BITS_PER_WORD;
    if (last_v > subgraph->vertex_count) last_v = subgraph->vertex_count;
    for (; v < last_v; v++) {
      if (!bitmap_is_set(frontier, v)) continue;
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

void bfs_cpu(partition_t* par, bfs_state_t* state) {

  // Copy the current state of the remote vertices bitmap
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
    bitmap_copy_cpu(state->visited[pid], (bitmap_t)par->outbox[pid].push_values,
                    par->outbox[pid].count);
  }

  // Visit the vertices in the frontier
  frontier_update_bitmap_cpu(&state->frontier, state->visited[par->id]);
  graph_t* subgraph = &par->subgraph;
  if (engine_partition_algorithm() == PAR_SORTED_ASC) {
    // High-degree vertices on the CPU: test the frontier count to determine
    // whether the frontier is sparse or dense. If the number of active vertices
    // is lower than a specific threshold, we consider the frontier sparse
    vid_t frontier_count = frontier_count_cpu(&state->frontier);
    if (frontier_count > 0) {
      if (frontier_count <= (subgraph->vertex_count *
                             TRV_FRONTIER_SPARSE_THRESHOLD)) {
        bfs_cpu_sparse_frontier(subgraph, state, par->id);
      } else {
        bfs_cpu_dense_frontier(subgraph, state, par->id);
      }
    }
  } else {
    // LOW-degree vertices on the CPU, the frontier will always be sparse
    bfs_cpu_sparse_frontier(subgraph, state, par->id);
  }

  // Diff the remote vertices bitmaps so that only the vertices who got set
  // in this round are notified
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
    bitmap_diff_cpu(state->visited[pid], (bitmap_t)par->outbox[pid].push_values,
                    par->outbox[pid].count);
  }
}

/**
 * A warp-based implementation of the BFS kernel. Please refer to the
 * description of the warp technique for details. Also, please refer to
 * bfs_kernel for details on the BFS implementation.
 */
template<int VWARP_WIDTH, int VWARP_BATCH, bool USE_FRONTIER>
__global__
void bfs_kernel(partition_t par, bfs_state_t state, 
                const vid_t* __restrict frontier, vid_t count) {

  if (THREAD_GLOBAL_INDEX >= 
      vwarp_thread_count(count, VWARP_WIDTH, VWARP_BATCH)) return;

  const eid_t* __restrict vertices = par.subgraph.vertices;
  const vid_t* __restrict edges = par.subgraph.edges;
  const cost_t* __restrict cost = state.cost;

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
#ifdef FEATURE_SM35
PRIVATE __host__ __device__ 
#else
PRIVATE __host__
#endif /* FEATURE_SM35  */
void bfs_launch_gpu(partition_t* par, bfs_state_t* state, vid_t* frontier, 
                    vid_t vertex_count, cudaStream_t stream) {
  const int threads = MAX_THREADS_PER_BLOCK;
  dim3 blocks;
  kernel_configure(vwarp_thread_count(vertex_count, VWARP_WIDTH, VWARP_BATCH),
                   blocks, threads);
  bfs_kernel<VWARP_WIDTH, VWARP_BATCH, USE_FRONTIER>
    <<<blocks, threads, 0, stream>>>(*par, *state, frontier, vertex_count);
}

typedef void(*bfs_gpu_func_t)(partition_t*, bfs_state_t*, vid_t*, 
                              vid_t, cudaStream_t);

#ifdef FEATURE_SM35
PRIVATE __global__
void bfs_launch_at_boundary_kernel(partition_t par, bfs_state_t state) {
  if (THREAD_GLOBAL_INDEX > 0 || (*state.frontier.count == 0)) {
    return;
  }
  const bfs_gpu_func_t BFS_GPU_FUNC[] = {
    bfs_launch_gpu<1,   2,  true>,   // (0) < 8
    bfs_launch_gpu<8,   8,  true>,   // (1) > 8    && < 32
    bfs_launch_gpu<32,  32, true>,   // (2) > 32   && < 128
    bfs_launch_gpu<128, 32, true>,   // (3) > 128  && < 256
    bfs_launch_gpu<256, 32, true>,   // (4) > 256  && < 1K
    bfs_launch_gpu<512, 32, true>,   // (5) > 1K   && < 2K
    bfs_launch_gpu<MAX_THREADS_PER_BLOCK, 8, true>,  // (6) > 2k
  };

  int64_t end = *(state.frontier.count);
  for (int i = FRONTIER_BOUNDARY_COUNT; i >= 0; i--) {
    int64_t start = state.frontier.boundaries[i];
    int64_t count = end - start;
    if (count > 0) {
      cudaStream_t s;
      cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
      BFS_GPU_FUNC[i](&par, &state, state.frontier.list + start, count, s);
      end = start;
    }
  }
}
#endif /* FEATURE_SM35  */

PRIVATE void bfs_tuned_launch_gpu(partition_t* par, bfs_state_t* state) {
  // Check if it is possible to build a frontier list
  vid_t vertex_count = par->subgraph.vertex_count;
  int use_frontier = true;
  if (engine_partition_algorithm() == PAR_SORTED_ASC) {
    // placing the many low degree nodes on the GPU means the frontier list
    // length is limited, hence we check here if the frontier can fit in the
    // pre-allocated space
    frontier_update_bitmap_gpu(&state->frontier, state->visited[par->id],
                               par->streams[1]);
    vertex_count = frontier_count_gpu(&state->frontier, par->streams[1]);
    use_frontier = (int)(vertex_count <= state->frontier.list_len);
  }

#ifdef FEATURE_SM35
  if (engine_sorted() && use_frontier) {
    // If the vertices are sorted by degree, call a kernel that takes 
    // advantage of that
    frontier_update_list_gpu(&state->frontier, state->level, state->cost, 
                             par->streams[1]);
    frontier_update_boundaries_gpu(&state->frontier, &par->subgraph, 
                                   par->streams[1]);
    bfs_launch_at_boundary_kernel<<<1, 1, 0, par->streams[1]>>>(*par, *state);
    CALL_CU_SAFE(cudaGetLastError());
    return;
  }
#endif /* FEATURE_SM35 */
  
  if (engine_partition_algorithm() != PAR_SORTED_ASC) {
    frontier_update_bitmap_gpu(&state->frontier, state->visited[par->id],
                               par->streams[1]);
    vertex_count = frontier_count_gpu(&state->frontier, par->streams[1]);
  }
  if (vertex_count == 0) return;

  if (use_frontier) {
    frontier_update_list_gpu(&state->frontier, state->level, state->cost, 
                             par->streams[1]);
  }

  // Call the BFS kernel
  const bfs_gpu_func_t BFS_GPU_FUNC[][2] = {{
      // RANDOM algorithm
      bfs_launch_gpu<VWARP_MEDIUM_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE, false>,
      bfs_launch_gpu<VWARP_MEDIUM_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE, true>
    }, {
      // HIGH partitioning
      bfs_launch_gpu<VWARP_MEDIUM_WARP_WIDTH, VWARP_LARGE_BATCH_SIZE, false>,
      bfs_launch_gpu<VWARP_MEDIUM_WARP_WIDTH, VWARP_LARGE_BATCH_SIZE, true>
    }, {
      // LOW partitioning
      bfs_launch_gpu<MAX_THREADS_PER_BLOCK, VWARP_MEDIUM_BATCH_SIZE, false>,
      bfs_launch_gpu<MAX_THREADS_PER_BLOCK, VWARP_MEDIUM_BATCH_SIZE, true>
    }
  };
  int par_alg = engine_partition_algorithm();
  BFS_GPU_FUNC[par_alg][use_frontier]
    (par, state, state->frontier.list, vertex_count, par->streams[1]);
  CALL_CU_SAFE(cudaGetLastError());
}

PRIVATE void bfs_gpu(partition_t* par, bfs_state_t* state) {
  // Copy the current state of the remote vertices
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
    bitmap_copy_gpu(state->visited[pid], 
                    (bitmap_t)par->outbox[pid].push_values,
                    par->outbox[pid].count, par->streams[1]);
  }

  // Call the bfs kernel
  bfs_tuned_launch_gpu(par, state);

  // Diff the remote vertices bitmap so that only the ones who got set in this
  // round are notified
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
    bitmap_diff_gpu(state->visited[pid], (bitmap_t)par->outbox[pid].push_values,
                    par->outbox[pid].count, par->streams[1]);
  }
}

PRIVATE void bfs(partition_t* par) {
  if (par->subgraph.vertex_count == 0) return;
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  if (par->processor.type == PROCESSOR_CPU) {
    bfs_cpu(par, state);
  } else if (par->processor.type == PROCESSOR_GPU) {  
    bfs_gpu(par, state);
  } else {
    assert(false);
  }
  state->level++;
}

PRIVATE inline void bfs_scatter_cpu(grooves_box_table_t* inbox, 
                                    bfs_state_t* state, bitmap_t visited) {
  bitmap_t remotely_visited = (bitmap_t)inbox->push_values;
  OMP(omp parallel for schedule(runtime))
  for (vid_t word_index = 0; word_index < bitmap_bits_to_words(inbox->count); 
       word_index++) {
    if (remotely_visited[word_index]) {
      vid_t bit_index = word_index * BITMAP_BITS_PER_WORD;
      vid_t bit_last_index = (word_index + 1) * BITMAP_BITS_PER_WORD;
      for (; bit_index < bit_last_index; bit_index++) {
        if (bitmap_is_set(remotely_visited, bit_index)) {
          vid_t vid = inbox->rmt_nbrs[bit_index];
          if (!bitmap_is_set(visited, vid)) {
            bitmap_set_cpu(visited, vid);
            state->cost[vid] = state->level;
          }
        }
      }
    }
  }
}

template<int VWARP_WIDTH, int BATCH_SIZE, int THREADS_PER_BLOCK>
__global__ void
bfs_scatter_kernel(const bitmap_t __restrict rmt_visited,
                   const vid_t* __restrict rmt_nbrs, vid_t word_count,
                   bitmap_t visited, cost_t* cost, cost_t level) {
  if (THREAD_GLOBAL_INDEX >= 
      vwarp_thread_count(word_count, VWARP_WIDTH, BATCH_SIZE)) return;
  vid_t start_word = vwarp_warp_start_vertex(VWARP_WIDTH, BATCH_SIZE) +
    vwarp_block_start_vertex(VWARP_WIDTH, BATCH_SIZE, THREADS_PER_BLOCK);
  vid_t end_word = start_word +
    vwarp_warp_batch_size(word_count, VWARP_WIDTH, BATCH_SIZE, 
                          THREADS_PER_BLOCK);
  int warp_offset = vwarp_thread_index(VWARP_WIDTH);
  for(vid_t k = start_word; k < end_word; k++) {
    bitmap_word_t word = rmt_visited[k];
    if (word == 0) continue;
    vid_t start_vertex = k * BITMAP_BITS_PER_WORD;
    for(vid_t i = warp_offset; i < BITMAP_BITS_PER_WORD;
        i += VWARP_WIDTH) {
      if (bitmap_is_set(word, i)) {
        vid_t vid = rmt_nbrs[start_vertex + i];
        if (!bitmap_is_set(visited, vid)) {
          bitmap_set_gpu(visited, vid);
          cost[vid] = level;
        }
      }
    }
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
      vid_t word_count = bitmap_bits_to_words(inbox->count);
      dim3 blocks;
      const int batch_size = 8; const int warp_size = 16;
      const int threads = DEFAULT_THREADS_PER_BLOCK;
      kernel_configure(vwarp_thread_count(word_count, warp_size, batch_size),
                       blocks, threads);
      bfs_scatter_kernel<warp_size, batch_size, threads>
        <<<blocks, threads, 0, par->streams[1]>>>
        ((bitmap_t)inbox->push_values, inbox->rmt_nbrs, word_count,
         state->visited[par->id], state->cost, state->level);
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
  state->visited[par->id] = bitmap_init_gpu(par->subgraph.vertex_count);
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      state->visited[pid] = bitmap_init_gpu(par->outbox[pid].count);
      bitmap_reset_gpu((bitmap_t)par->outbox[pid].push_values,
                       par->outbox[pid].count, par->streams[1]);
    }
  }
  // set the source vertex as visited
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    bfs_init_kernel<<<1, 1, 0, par->streams[1]>>>
      (state->visited[par->id], GET_VERTEX_ID(state_g.src));
    CALL_CU_SAFE(cudaGetLastError());
  }
  frontier_init_gpu(&state->frontier, par->subgraph.vertex_count);
}

PRIVATE inline void bfs_init_cpu(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  state->visited[par->id] = bitmap_init_cpu(par->subgraph.vertex_count);
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      state->visited[pid] = bitmap_init_cpu(par->outbox[pid].count);
      bitmap_reset_cpu((bitmap_t)par->outbox[pid].push_values,
                       par->outbox[pid].count);
    }
  }
  // set the source vertex as visited
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    bitmap_set_cpu(state->visited[par->id], GET_VERTEX_ID(state_g.src));
  }
  frontier_init_cpu(&state->frontier, par->subgraph.vertex_count);
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
    frontier_finalize_cpu(&state->frontier);
  } else if (par->processor.type == PROCESSOR_GPU) {
    bitmap_finalize_gpu(state->visited[par->id]);
    type = TOTEM_MEM_DEVICE;
    frontier_finalize_gpu(&state->frontier);
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
