/**
 * Contains an implementation of the Graph500 benchmark algorithm
 * using the totem framework
 *
 *  Created on: 2013-05-27
 *  Author: Abdullah Gharaibeh
 */

#include "totem_alg.h"
#include "totem_engine.cuh"

/**
 * per-partition specific state
 */
typedef struct graph500_state_s {
  cost_t*   cost;             // maintains the distance from the source of each 
                              // vertex in the partition
  vid_t*    tree[MAX_PARTITION_COUNT];    // a list of trees one for each
                                          // remote partition
  bitmap_t  visited[MAX_PARTITION_COUNT]; // a list of bitmaps one for each 
                                          // remote partition
  bool*     finished;         // points to Totem's finish flag
  vid_t*    est_active_count; // estimated number of active vertices
  cost_t    level;            // current level being processed in the partition
} graph500_state_t;

/**
 * global state shared among all partitions
 */
typedef struct graph500_global_state_s {
  bool      initialized; // indicates if the global state is initialized
  cost_t*   cost;        // final cost output buffer
  vid_t*    tree;        // final tree output buffer
  vid_t*    tree_h;      // Used as a temporary buffer to receive the final 
                         // result copied back from GPU partitions before being
                         // copied again to the final output buffer
  vid_t     src;         // source vertex id (the id after partitioning)
} graph500_global_state_t;
PRIVATE graph500_global_state_t state_g = {false, NULL, NULL, 0};

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public functions (GPU and CPU)
*/
PRIVATE error_t check_special_cases(vid_t src, vid_t* tree, bool* finished) {
  *finished = true;
  if (!state_g.initialized) return FAILURE;
  if((src >= engine_vertex_count()) || (tree == NULL)) {
    return FAILURE;
  } else if(engine_vertex_count() == 1) {
    tree[0] = src;
    return SUCCESS;
  } else if(engine_edge_count() == 0) {
    // Initialize tree to INFINITE.
    totem_memset(tree, VERTEX_ID_MAX, engine_vertex_count(), TOTEM_MEM_HOST);
    tree[src] = src;
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

void graph500_cpu(partition_t* par) {
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
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
      vid_t* tree = state->tree[nbr_pid];
      if (!bitmap_is_set(visited, nbr)) {
        if (bitmap_set_cpu(visited, nbr)) {
          if (nbr_pid == par->id) {
            state->cost[nbr] = state->level + 1;            
          }
          tree[nbr] = SET_PARTITION_ID(v, par->id);
          finished = false;
        }
      }
    }
  }
  if (!finished) *(state->finished) = false;
}

/**
 * A warp-based implementation of the Graph500 kernel. Please refer to the
 * description of the warp technique for details. Also, please refer to
 * graph500_kernel for details on the algorithm implementation.
 */
template<int VWARP_WIDTH, int VWARP_BATCH>
__global__
void graph500_kernel(partition_t par, graph500_state_t state) {
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
        vid_t* tree = state.tree[nbr_pid];
        if (!bitmap_is_set(visited, nbr)) {
          if (bitmap_set_gpu(visited, nbr)) {
            if (nbr_pid == par.id) {
              if (state.cost[nbr] == INF_COST) { 
                state.cost[nbr] = state.level + 1;
                local_est_active_count++;
              }
            }
            vid_t parent = block_start_vertex + 
              vwarp_start_vertex + v;
            tree[nbr] = SET_PARTITION_ID(parent, par.id);
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

PRIVATE
void graph500_gpu_long_small(partition_t* par, graph500_state_t* state) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(vwarp_thread_count(par->subgraph.vertex_count,
                                      VWARP_LONG_WARP_WIDTH,
                                      VWARP_SMALL_BATCH_SIZE),
                   blocks, threads);
  graph500_kernel<VWARP_LONG_WARP_WIDTH, VWARP_SMALL_BATCH_SIZE>
    <<<blocks, threads, 0, par->streams[1]>>>(*par, *state);
}

PRIVATE
void graph500_gpu_short_small(partition_t* par, graph500_state_t* state) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(vwarp_thread_count(par->subgraph.vertex_count,
                                      VWARP_SHORT_WARP_WIDTH,
                                      VWARP_SMALL_BATCH_SIZE),
                   blocks, threads);
  graph500_kernel<VWARP_SHORT_WARP_WIDTH, VWARP_SMALL_BATCH_SIZE>
    <<<blocks, threads, 0, par->streams[1]>>>(*par, *state);
}

PRIVATE 
void graph500_gpu_medium_large(partition_t* par, graph500_state_t* state) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(vwarp_thread_count(par->subgraph.vertex_count,
                                      VWARP_MEDIUM_WARP_WIDTH,
                                      VWARP_LARGE_BATCH_SIZE),
                   blocks, threads);
  graph500_kernel<VWARP_MEDIUM_WARP_WIDTH, VWARP_LARGE_BATCH_SIZE>
    <<<blocks, threads, 0, par->streams[1]>>>(*par, *state);
}

typedef void(*graph500_gpu_func_t)(partition_t*, graph500_state_t*);
PRIVATE const graph500_gpu_func_t GRAPH500_GPU_FUNC[][2] = {
  {graph500_gpu_medium_large, graph500_gpu_medium_large}, // RANDOM partitioning
  {graph500_gpu_short_small, graph500_gpu_medium_large},  // HIGH partitioning
  {graph500_gpu_long_small, graph500_gpu_long_small}      // LOW partitioning
};

PRIVATE void graph500(partition_t* par) {
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  if (par->processor.type == PROCESSOR_CPU) {
    graph500_cpu(par);
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
    GRAPH500_GPU_FUNC[par_alg][coarse_fine](par, state);
    CALL_CU_SAFE(cudaGetLastError());
  } else {
    fprintf(stderr, "Unsupported processor type\n");
    assert(false);
  }
  state->level++;
}

PRIVATE inline void graph500_scatter_cpu(grooves_box_table_t* inbox, 
                                         graph500_state_t* state, 
                                         bitmap_t visited, vid_t* tree) {
  vid_t* rmt_tree = (vid_t*)inbox->push_values;
  OMP(omp parallel for schedule(runtime))
  for (vid_t index = 0; index < inbox->count; index++) {
    vid_t vid = inbox->rmt_nbrs[index];
    if (!bitmap_is_set(visited, vid)) {
      vid_t parent = rmt_tree[index];
      if (parent != VERTEX_ID_MAX) {
        bitmap_set_cpu(visited, vid);
        state->cost[vid] = state->level;
        tree[vid] = parent;
      }
    }
  }
}

__global__ 
void graph500_scatter_kernel(grooves_box_table_t inbox, graph500_state_t state,
                             bitmap_t visited, vid_t* tree) {
  vid_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) return;
  __shared__ vid_t est_active_count;
  est_active_count = 0;
  __syncthreads();
  vid_t* rmt_tree = (vid_t*)inbox.push_values;
  vid_t vid = inbox.rmt_nbrs[index];
  if (!bitmap_is_set(visited, vid)) {
    vid_t parent = rmt_tree[index];
    if (parent != VERTEX_ID_MAX) {
      bitmap_set_gpu(visited, vid);
      state.cost[vid] = state.level;
      tree[vid] = parent;
      est_active_count += 1;
    }
  }
  if (est_active_count && THREAD_BLOCK_INDEX == 0) {
    *state.est_active_count += est_active_count;
  }
}

PRIVATE void graph500_scatter(partition_t* par) {
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_CPU) {
      graph500_scatter_cpu(inbox, state, state->visited[par->id], 
                           state->tree[par->id]);
    } else if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks, threads;
      KERNEL_CONFIGURE(inbox->count, blocks, threads);
      graph500_scatter_kernel<<<blocks, threads, 0, par->streams[1]>>>
        (*inbox, *state, state->visited[par->id], state->tree[par->id]);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(false);
    }
  }
}

PRIVATE void graph500_aggregate(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  graph_t*     subgraph = &par->subgraph;
  vid_t*       src_tree = NULL;
  if (par->processor.type == PROCESSOR_CPU) {
    src_tree = state->tree[par->id];
  } else if (par->processor.type == PROCESSOR_GPU) {
    assert(state_g.tree_h);
    CALL_CU_SAFE(cudaMemcpy(state_g.tree_h, state->tree[par->id], 
                            subgraph->vertex_count * sizeof(vid_t),
                            cudaMemcpyDefault));
    src_tree = state_g.tree_h;
  } else {
    assert(false);
  }
  // aggregate the results
  assert(state_g.tree);
  OMP(omp parallel for schedule(runtime))
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    if (src_tree[v] != VERTEX_ID_MAX) {
      state_g.tree[par->map[v]] = engine_vertex_id_local_to_global(src_tree[v]);
    }
  }
}

__global__ void graph500_init_kernel(bitmap_t visited, vid_t src) {
  if (THREAD_GLOBAL_INDEX != 0) return;
  bitmap_set_gpu(visited, src);
}

PRIVATE inline void graph500_init_gpu(partition_t* par) {
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  bitmap_reset_gpu(state->visited[par->id], par->subgraph.vertex_count);
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      bitmap_reset_gpu(state->visited[pid], par->outbox[pid].count);
    }
  }
  // set the source vertex as visited
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    dim3 blocks, threads;
    KERNEL_CONFIGURE(1, blocks, threads);
    graph500_init_kernel<<<blocks, threads, 0, par->streams[1]>>>
      (state->visited[par->id], GET_VERTEX_ID(state_g.src));
    CALL_CU_SAFE(cudaGetLastError());
  }
  totem_memset(state->est_active_count, (vid_t)0, 1, TOTEM_MEM_DEVICE, 
               par->streams[1]);
}

PRIVATE inline void graph500_init_cpu(partition_t* par) {
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  bitmap_reset_cpu(state->visited[par->id], par->subgraph.vertex_count);
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      bitmap_reset_cpu(state->visited[pid], par->outbox[pid].count);
    }
  }
  // set the source vertex as visited
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    bitmap_set_cpu(state->visited[par->id], GET_VERTEX_ID(state_g.src));
  }
}

PRIVATE void graph500_init(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_CPU) {
    graph500_init_cpu(par);
  } else if (par->processor.type == PROCESSOR_GPU) {
    type = TOTEM_MEM_DEVICE;
    graph500_init_gpu(par);
  } else {
    assert(false);
  }
  totem_memset(state->cost, INF_COST, par->subgraph.vertex_count, type, 
               par->streams[1]);
  totem_memset(state->tree[par->id], VERTEX_ID_MAX, par->subgraph.vertex_count, 
               type, par->streams[1]);
  engine_set_outbox(par->id, (vid_t)VERTEX_ID_MAX);
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    // For the source vertex, initialize cost.    
    totem_memset(&((state->cost)[GET_VERTEX_ID(state_g.src)]), (cost_t)0, 1, 
                 type, par->streams[1]);
    totem_memset(&((state->tree[par->id])[GET_VERTEX_ID(state_g.src)]), 
                 state_g.src, 1, type, par->streams[1]);
  }
  state->level = 0;
}

void graph500_free(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_CPU) {
    bitmap_finalize_cpu(state->visited[par->id]);
  } else if (par->processor.type == PROCESSOR_GPU) {
    bitmap_finalize_gpu(state->visited[par->id]);
    totem_free(state->est_active_count, TOTEM_MEM_DEVICE);
    type = TOTEM_MEM_DEVICE;
  } else {
    assert(false);
  }
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      if (par->processor.type == PROCESSOR_CPU) {
        bitmap_finalize_cpu(state->visited[pid]);
      } else {
        bitmap_finalize_gpu(state->visited[pid]);
      }
    }
  }
  totem_free(state->cost, type);
  totem_free(state->tree[par->id], type);
  free(state);
  par->algo_state = NULL;
  
  // try to free global state if not already freed
  if (state_g.initialized) {
    if (engine_largest_gpu_partition()) {
      totem_free(state_g.tree_h, TOTEM_MEM_HOST_PINNED);
    }
    totem_free(state_g.cost, TOTEM_MEM_HOST);
    memset(&state_g, 0, sizeof(graph500_global_state_t));
    state_g.initialized = false;
  }
}

void graph500_alloc(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  graph500_state_t* state = 
    (graph500_state_t*)calloc(1, sizeof(graph500_state_t));
  assert(state);
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_CPU) {
    state->visited[par->id] = bitmap_init_cpu(par->subgraph.vertex_count);
  } else if (par->processor.type == PROCESSOR_GPU) {
    state->visited[par->id] = bitmap_init_gpu(par->subgraph.vertex_count);
    CALL_SAFE(totem_malloc(sizeof(vid_t), TOTEM_MEM_DEVICE, 
                           (void **)&state->est_active_count));
    type = TOTEM_MEM_DEVICE;
  } else {
    assert(false);
  }
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      if (par->processor.type == PROCESSOR_CPU) {
        state->visited[pid] = bitmap_init_cpu(par->outbox[pid].count);
      } else {
        state->visited[pid] = bitmap_init_gpu(par->outbox[pid].count);
      }
      state->tree[pid] = (vid_t*)par->outbox[pid].push_values;
    }
  }
  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(cost_t), type, 
                         (void**)&(state->cost)));
  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(vid_t), type, 
                         (void**)&(state->tree[par->id])));
  state->finished = engine_get_finished_ptr(par->id);
  par->algo_state = state;

  // try to initalize the global state if not already initialized
  if (!state_g.initialized) {    
    totem_malloc(engine_vertex_count() * sizeof(cost_t), TOTEM_MEM_HOST, 
                 (void**)&state_g.cost);
    if (engine_largest_gpu_partition()) {
      CALL_SAFE(totem_malloc(engine_largest_gpu_partition() * sizeof(vid_t), 
                             TOTEM_MEM_HOST_PINNED, (void**)&state_g.tree_h));
    }
    state_g.initialized = true;
  }
}

error_t graph500_hybrid(vid_t src, vid_t* tree) {
  // check for special cases
  bool finished = false;
  error_t rc = check_special_cases(src, tree, &finished);
  if (finished) return rc;

  // initialize the global state
  state_g.tree = tree;
  state_g.src  = engine_vertex_id_in_partition(src);

  // initialize the engine
  engine_config_t config = {
    NULL, graph500, graph500_scatter, NULL, graph500_init, NULL, 
    graph500_aggregate, GROOVES_PUSH
  };
  engine_config(&config);
  engine_execute();
  return SUCCESS;
}