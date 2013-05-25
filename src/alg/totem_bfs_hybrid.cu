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
  cost_t*   cost;             // one slot per vertex in the partition
  bitmap_t* visited;          // a list of bitmaps one for each remote partition
  bool*     finished;         // points to Totem's finish flag
  vid_t*    est_active_count; // estimated number of active vertices
  cost_t    level;            // current level being processed by the partition
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

void bfs_cpu(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  graph_t* subgraph = &par->subgraph;
  bool finished = true;

  // The "runtime" scheduling clause defer the choice of thread scheduling
  // algorithm to the choice of the client, either via OS environment variable
  // or omp_set_schedule interface.
  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    if (state->cost[v] != state->level) continue;
    for (eid_t i = subgraph->vertices[v]; i < subgraph->vertices[v + 1]; i++) {
      int nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
      vid_t nbr = GET_VERTEX_ID(subgraph->edges[i]);
      bitmap_t visited = state->visited[nbr_pid];
      if (!bitmap_is_set(visited, nbr)) {
        if (bitmap_set_cpu(visited, nbr)) {
          if (nbr_pid == par->id) {
            state->cost[nbr] = state->level + 1;
          }
          finished = false;
        }
      }
    }
  }
  if (!finished) *(state->finished) = false;
}

/**
 * A warp-based implementation of the BFS kernel. Please refer to the
 * description of the warp technique for details. Also, please refer to
 * bfs_kernel for details on the BFS implementation.
 */
template<int VWARP_WIDTH, int VWARP_BATCH>
__global__
void bfs_kernel(partition_t par, bfs_state_t state) {
  vid_t vertex_count = par.subgraph.vertex_count;
  if (vwarp_block_start_vertex(VWARP_WIDTH, VWARP_BATCH) >= vertex_count) {
    return;
  }

  // Copy the work of a thread-block to local space
  const int block_max_batch_size = 
    VWARP_BLOCK_MAX_BATCH_SIZE(VWARP_WIDTH, VWARP_BATCH);
  __shared__ eid_t  vertices_s[block_max_batch_size + 1];
  __shared__ cost_t cost_s[block_max_batch_size];
  const vid_t block_start_vertex = 
    vwarp_block_start_vertex(VWARP_WIDTH, VWARP_BATCH);
  const vid_t block_batch_size = 
    vwarp_block_batch_size(vertex_count, VWARP_WIDTH, VWARP_BATCH);
  vwarp_memcpy(vertices_s, &(par.subgraph.vertices[block_start_vertex]),
               block_batch_size + 1, THREAD_BLOCK_INDEX, blockDim.x);
  vwarp_memcpy(cost_s, &state.cost[block_start_vertex], 
               block_batch_size, THREAD_BLOCK_INDEX, blockDim.x);

  // This flag is used to report the finish state of a block of threads. This
  // is useful to avoid having many threads writing to the global finished
  // flag, which can hurt performance (since "finished" is actually allocated
  // on the host, and each write will cause a transfer over the PCI-E bus)
  __shared__ bool finished_block;
  finished_block = true;
  __shared__ vid_t est_active_count;
  est_active_count = 0;
  __syncthreads();
  if (THREAD_GLOBAL_INDEX >= 
      vwarp_thread_count(vertex_count, VWARP_WIDTH, VWARP_BATCH)) return;

  // Each virtual warp in a thread-block will process its share 
  // of the thread-block batch of work
  vid_t vwarp_start_vertex = vwarp_warp_start_vertex(VWARP_WIDTH, VWARP_BATCH);
  eid_t* vwarp_vertices = &vertices_s[vwarp_start_vertex];
  cost_t* vwarp_cost = &cost_s[vwarp_start_vertex];
  vid_t vwarp_batch_size = vwarp_warp_batch_size(vertex_count, VWARP_WIDTH, 
                                                 VWARP_BATCH);
  vid_t local_est_active_count = 0;
  for(vid_t v = 0; v < vwarp_batch_size; v++) {
    if (vwarp_cost[v] == state.level) {
      eid_t nbr_count = vwarp_vertices[v + 1] - vwarp_vertices[v];
      vid_t* nbrs = &(par.subgraph.edges[vwarp_vertices[v]]);
      for(vid_t i = vwarp_thread_index(VWARP_WIDTH); i < nbr_count; 
          i += VWARP_WIDTH) {
        int nbr_pid = GET_PARTITION_ID(nbrs[i]);
        vid_t nbr = GET_VERTEX_ID(nbrs[i]);
        bitmap_t visited = state.visited[nbr_pid];
        if (!bitmap_is_set(visited, nbr)) {
          if (bitmap_set_gpu(visited, nbr)) {
            if (nbr_pid == par.id) {
              if (state.cost[nbr] == INF_COST) { 
                state.cost[nbr] = state.level + 1;
                local_est_active_count++;
              }
            }
            finished_block = false;
          }
        }
      }
    }
  }

  // If we want to get an exact count, the addition here should be atomic,
  // however to avoid the overhead of atomic operation, and since an estimate
  // count is good enough, we do not use atomic add.
  if (local_est_active_count) est_active_count += local_est_active_count;
  __syncthreads();

  // Similar to the comment above, we accumulate here without using atomic 
  // add to reduce the overhead
  if (est_active_count && THREAD_BLOCK_INDEX == 0) {
    *state.est_active_count += est_active_count;
  }
  if (!finished_block && THREAD_BLOCK_INDEX == 0) *state.finished = false;
}

PRIVATE void bfs_gpu_long_small(partition_t* par, bfs_state_t* state) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(vwarp_thread_count(par->subgraph.vertex_count,
                                      VWARP_LONG_WARP_WIDTH,
                                      VWARP_SMALL_BATCH_SIZE),
                   blocks, threads);
  bfs_kernel<VWARP_LONG_WARP_WIDTH, VWARP_SMALL_BATCH_SIZE>
    <<<blocks, threads, 0, par->streams[1]>>>(*par, *state);
}

PRIVATE void bfs_gpu_short_small(partition_t* par, bfs_state_t* state) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(vwarp_thread_count(par->subgraph.vertex_count,
                                      VWARP_SHORT_WARP_WIDTH,
                                      VWARP_SMALL_BATCH_SIZE),
                   blocks, threads);
  bfs_kernel<VWARP_SHORT_WARP_WIDTH, VWARP_SMALL_BATCH_SIZE>
    <<<blocks, threads, 0, par->streams[1]>>>(*par, *state);
}

PRIVATE void bfs_gpu_medium_large(partition_t* par, bfs_state_t* state) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(vwarp_thread_count(par->subgraph.vertex_count,
                                      VWARP_MEDIUM_WARP_WIDTH,
                                      VWARP_LARGE_BATCH_SIZE),
                   blocks, threads);
  bfs_kernel<VWARP_MEDIUM_WARP_WIDTH, VWARP_LARGE_BATCH_SIZE>
    <<<blocks, threads, 0, par->streams[1]>>>(*par, *state);
}

typedef void(*bfs_gpu_func_t)(partition_t*, bfs_state_t*);
PRIVATE const bfs_gpu_func_t BFS_GPU_FUNC[][2] = {
  {bfs_gpu_medium_large, bfs_gpu_medium_large}, // RANDOM algorithm
  {bfs_gpu_short_small, bfs_gpu_medium_large},  // HIGH partitioning
  {bfs_gpu_long_small, bfs_gpu_long_small}      // LOW partitioning
};

PRIVATE void bfs(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  if (par->processor.type == PROCESSOR_CPU) {
    bfs_cpu(par);
  } else if (par->processor.type == PROCESSOR_GPU) {
    vid_t est_active_count = 0;
    CALL_CU_SAFE(cudaMemcpyAsync(&est_active_count, state->est_active_count,
                                 sizeof(vid_t), cudaMemcpyDefault,
                                 par->streams[1]));
    cudaMemsetAsync(state->est_active_count, 0, sizeof(vid_t), 
                    par->streams[1]);
    int par_alg = engine_partition_algorithm();
    int coarse_fine = (int)(est_active_count > 
                            VWARP_ACTIVE_VERTICES_THRESHOLD);
    BFS_GPU_FUNC[par_alg][coarse_fine](par, state);
    CALL_CU_SAFE(cudaGetLastError());
  } else {
    assert(false);
  }
  state->level++;
}

PRIVATE inline void bfs_scatter_cpu(grooves_box_table_t* inbox, 
                                    bfs_state_t* state, bitmap_t* visited) {
  bitmap_t remotely_visited = (bitmap_t)inbox->push_values;
  OMP(omp parallel for schedule(runtime))
  for (vid_t index = 0; index < inbox->count; index++) {
    vid_t vid = inbox->rmt_nbrs[index];
    if (bitmap_is_set(remotely_visited, index) &&
        !bitmap_is_set(*visited, vid)) {
      bitmap_set_cpu(*visited, vid);
      state->cost[vid] = state->level;
    }
  }
}

__global__ void bfs_scatter_kernel(grooves_box_table_t inbox, bfs_state_t state,
                                   bitmap_t* visited) {
  vid_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) return;
  __shared__ vid_t est_active_count;
  est_active_count = 0;
  __syncthreads();  
  bitmap_t rmt_visited = (bitmap_t)inbox.push_values;
  vid_t vid = inbox.rmt_nbrs[index];
  if (bitmap_is_set(rmt_visited, index) &&
      !bitmap_is_set(*visited, vid)) {
    bitmap_set_gpu(*visited, vid);
    state.cost[vid] = state.level;
    est_active_count += 1;
  }
  if (est_active_count && THREAD_BLOCK_INDEX == 0) {
    *state.est_active_count += est_active_count;
  }
}

PRIVATE void bfs_scatter(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_CPU) {
      bfs_scatter_cpu(inbox, state, &(state->visited[par->id]));
    } else if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks, threads;
      KERNEL_CONFIGURE(inbox->count, blocks, threads);
      bfs_scatter_kernel<<<blocks, threads, 0, par->streams[1]>>>
        (*inbox, *state, &(state->visited[par->id]));
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
  bitmap_t visited[MAX_PARTITION_COUNT];
  visited[par->id] = bitmap_init_gpu(par->subgraph.vertex_count);
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      visited[pid] = (bitmap_t)par->outbox[pid].push_values;
      bitmap_reset_gpu(visited[pid], par->outbox[pid].count);
    }
  }
  CALL_SAFE(totem_malloc(MAX_PARTITION_COUNT * sizeof(bitmap_t), 
                         TOTEM_MEM_DEVICE, (void**)&(state->visited)));
  CALL_CU_SAFE(cudaMemcpy(state->visited, visited, 
                          MAX_PARTITION_COUNT * sizeof(bitmap_t),
                          cudaMemcpyDefault));
  // set the source vertex as visited
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    dim3 blocks, threads;
    KERNEL_CONFIGURE(1, blocks, threads);
    bfs_init_kernel<<<blocks, threads, 0, par->streams[1]>>>
      (visited[par->id], GET_VERTEX_ID(state_g.src));
    CALL_CU_SAFE(cudaGetLastError());
  }
  CALL_SAFE(totem_malloc(sizeof(vid_t), TOTEM_MEM_DEVICE, 
                         (void **)&state->est_active_count));
  totem_memset(state->est_active_count, (vid_t)0, 1, TOTEM_MEM_DEVICE, 
               par->streams[1]);
}

PRIVATE inline void bfs_init_cpu(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  CALL_SAFE(totem_malloc(MAX_PARTITION_COUNT * sizeof(bitmap_t),
                         TOTEM_MEM_HOST, (void**)&(state->visited)));
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
  if (!par->subgraph.vertex_count) return;
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
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_CPU) {
    bitmap_finalize_cpu(state->visited[par->id]);
  } else if (par->processor.type == PROCESSOR_GPU) {
    bitmap_t visited;
    CALL_CU_SAFE(cudaMemcpy(&visited, &(state->visited[par->id]), 
                            sizeof(bitmap_t), cudaMemcpyDefault));
    bitmap_finalize_gpu(visited);
    totem_free(state->est_active_count, TOTEM_MEM_DEVICE);
    type = TOTEM_MEM_DEVICE;
  } else {
    assert(false);
  }
  totem_free(state->visited, type);
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
