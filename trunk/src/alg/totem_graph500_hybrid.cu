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
  vid_t*   tree[MAX_PARTITION_COUNT];    // a list of trees one for each
                                         // remote partition
  bitmap_t visited[MAX_PARTITION_COUNT]; // a list of bitmaps one for each 
                                         // remote partition
  bool*    finished;                     // points to Totem's finish flag
  frontier_state_t frontier;
} graph500_state_t;

/**
 * global state shared among all partitions
 */
typedef struct graph500_global_state_s {
  bool        initialized; // indicates if the global state is initialized
  bfs_tree_t* tree;        // final tree output buffer
  vid_t*      tree_h;      // Used as a temporary buffer to receive the final 
                           // result copied back from GPU partitions before
                           // being copied again to the final output buffer
  vid_t       src;         // source vertex id (the id after partitioning)
} graph500_global_state_t;
PRIVATE graph500_global_state_t state_g = {false, NULL, NULL, 0};

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public functions (GPU and CPU)
*/
PRIVATE error_t check_special_cases(vid_t src, bfs_tree_t* tree, 
                                    bool* finished) {
  *finished = true;
  if (!state_g.initialized) return FAILURE;
  if((src >= engine_vertex_count()) || (tree == NULL)) {
    return FAILURE;
  } else if(engine_vertex_count() == 1) {
    tree[0] = src;
    return SUCCESS;
  } else if(engine_edge_count() == 0) {
    // Initialize tree to INFINITE.
    totem_memset(tree, (bfs_tree_t)(-1), engine_vertex_count(), TOTEM_MEM_HOST);
    tree[src] = src;
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

PRIVATE __attribute__((always_inline)) void 
graph500_cpu_process_vertex(graph_t* subgraph, graph500_state_t* state, 
                            vid_t v, int pid, bool& finished) {
  for (eid_t i = subgraph->vertices[v]; i < subgraph->vertices[v + 1]; i++) {
    int nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
    vid_t nbr = GET_VERTEX_ID(subgraph->edges[i]);
    bitmap_t visited = state->visited[nbr_pid];
    vid_t* tree = state->tree[nbr_pid];
    if (!bitmap_is_set(visited, nbr)) {
      if (bitmap_set_cpu(visited, nbr)) {
        tree[nbr] = SET_PARTITION_ID(v, pid);
        finished = false;
      }
    }
  }
}

PRIVATE void graph500_cpu_sparse_frontier(graph_t* subgraph, 
                                          graph500_state_t* state, int pid) {
  vid_t words = bitmap_bits_to_words(subgraph->vertex_count);
  // The "runtime" scheduling clause defer the choice of thread scheduling
  // algorithm to the choice of the client, either via OS environment variable
  // or omp_set_schedule interface.
  bool finished = true;
  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t word_index = 0; word_index < words; word_index++) {
    if (!state->frontier.current[word_index]) continue;
    vid_t v = word_index * BITMAP_BITS_PER_WORD;
    vid_t last_v = (word_index + 1) * BITMAP_BITS_PER_WORD;
    if (last_v > subgraph->vertex_count) last_v = subgraph->vertex_count;
    for (; v < last_v; v++) {
      if (!bitmap_is_set(state->frontier.current, v)) continue;
      graph500_cpu_process_vertex(subgraph, state, v, pid, finished);
    }
  }
  if (!finished) *(state->finished) = false;
}

void graph500_cpu(partition_t* par, graph500_state_t* state) {

  // Copy the current state of the remote vertices bitmap
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
    bitmap_t rmt_bitmap = (bitmap_t)par->outbox[pid].push_values;
    bitmap_copy_cpu(state->visited[pid], rmt_bitmap, par->outbox[pid].count);
  }

  // Update the frontier bitmap
  frontier_update_bitmap_cpu(&state->frontier, state->visited[par->id]);
  graph500_cpu_sparse_frontier(&par->subgraph, state, par->id);

  // Diff the remote vertices bitmaps so that only the vertices who got set
  // in this round are notified
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
    bitmap_t rmt_bitmap = (bitmap_t)par->outbox[pid].push_values;
    bitmap_diff_cpu(state->visited[pid], rmt_bitmap, par->outbox[pid].count);
  }
}

/**
 * A warp-based implementation of the Graph500 kernel. Please refer to the
 * description of the warp technique for details. Also, please refer to
 * graph500_kernel for details on the algorithm implementation.
 */
template<int VWARP_WIDTH, int VWARP_BATCH, bool USE_FRONTIER>
__global__
void graph500_kernel(partition_t par, graph500_state_t state,
                     const vid_t* __restrict frontier, vid_t count) {
  if ((count == 0) || (THREAD_GLOBAL_INDEX >= 
                       vwarp_thread_count(count, VWARP_WIDTH, VWARP_BATCH))) {
    return;
  }

  const eid_t* __restrict vertices = par.subgraph.vertices;

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
    if (USE_FRONTIER) v = frontier[i]; // this is evaluated at compile time

    // if USE_FRONTIER is true, the if-statement will be removed by the compiler
    if (USE_FRONTIER || bitmap_is_set(state.frontier.current, v)) { 
      const eid_t nbr_count = vertices[v + 1] - vertices[v];
      vid_t* nbrs = par.subgraph.edges + vertices[v];
      if (v >= par.subgraph.vertex_ext) {
        nbrs = par.subgraph.edges_ext + 
          (vertices[v] - par.subgraph.edge_count_ext);
      }
      for(vid_t i = warp_offset; i < nbr_count; i += VWARP_WIDTH) {
        int nbr_pid = GET_PARTITION_ID(nbrs[i]);
        vid_t nbr = GET_VERTEX_ID(nbrs[i]);
        bitmap_t visited = state.visited[nbr_pid];
        vid_t* tree = state.tree[nbr_pid];
        if (!bitmap_is_set(visited, nbr)) {
          if (bitmap_set_gpu(visited, nbr)) {
            tree[nbr] = SET_PARTITION_ID(v, par.id);
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
void graph500_launch_gpu(partition_t* par, graph500_state_t* state,
                         vid_t* frontier, vid_t count, cudaStream_t stream) {
  if (!USE_FRONTIER) count = par->subgraph.vertex_count;
  const int threads = MAX_THREADS_PER_BLOCK;
  dim3 blocks;
  kernel_configure(vwarp_thread_count(count, VWARP_WIDTH, VWARP_BATCH), blocks,
                   threads);
  graph500_kernel<VWARP_WIDTH, VWARP_BATCH, USE_FRONTIER>
    <<<blocks, threads, 0, stream>>>(*par, *state, frontier, count);
}

typedef void(*graph500_gpu_func_t)(partition_t*, graph500_state_t*, vid_t*, 
                                   vid_t, cudaStream_t);
#ifdef FEATURE_SM35
PRIVATE __global__
void graph500_launch_at_boundary_kernel(partition_t par, 
                                        graph500_state_t state) {
  if (THREAD_GLOBAL_INDEX > 0 || (*state.frontier.count == 0)) {
    return;
  }
  const graph500_gpu_func_t GRAPH500_GPU_FUNC[] = {
    graph500_launch_gpu<1,   2,  true>,   // (0) < 8
    graph500_launch_gpu<8,   8,  true>,   // (1) > 8    && < 32
    graph500_launch_gpu<32,  32, true>,   // (2) > 32   && < 128
    graph500_launch_gpu<128, 32, true>,   // (3) > 128  && < 256
    graph500_launch_gpu<256, 32, true>,   // (4) > 256  && < 1K
    graph500_launch_gpu<512, 32, true>,   // (5) > 1K   && < 2K
    graph500_launch_gpu<MAX_THREADS_PER_BLOCK, 8, true>,  // (6) > 2k
  };

  int64_t end = *(state.frontier.count);
  for (int i = FRONTIER_BOUNDARY_COUNT; i >= 0; i--) {
    int64_t start = state.frontier.boundaries[i];
    int64_t count = end - start;
    if (count > 0) {
      cudaStream_t s;
      cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
      GRAPH500_GPU_FUNC[i](&par, &state, state.frontier.list + start, count, s);
      end = start;
    }
  }
}
#endif /* FEATURE_SM35  */

PRIVATE void graph500_tuned_launch_gpu(partition_t* par, 
                                       graph500_state_t* state) {
  // Update the frontier's bitmap
  frontier_update_bitmap_gpu(&state->frontier, state->visited[par->id],
                             par->streams[1]);

  // Check if it is possible to build a frontier list
  vid_t count = par->subgraph.vertex_count;
  int use_frontier = true;
  if (engine_partition_algorithm() == PAR_SORTED_ASC) {
    // placing the many low degree nodes on the GPU means the frontier list
    // length is limited, hence we check here if the frontier can fit in the
    // pre-allocated space
    count = frontier_count_gpu(&state->frontier, par->streams[1]);
    use_frontier = (int)(count <= state->frontier.list_len);
  }

  // If the vertices are sorted by degree, call a kernel that takes 
  // advantage of that
#ifdef FEATURE_SM35
  if (engine_sorted() && use_frontier) {    
    // If the vertices are sorted by degree, call a kernel that takes 
    // advantage of that
    frontier_update_list_gpu(&state->frontier, par->streams[1]);
    frontier_update_boundaries_gpu(&state->frontier, &par->subgraph, 
                                   par->streams[1]);
    graph500_launch_at_boundary_kernel<<<1, 1, 0, par->streams[1]>>>
      (*par, *state);
    CALL_CU_SAFE(cudaGetLastError());
    return;
  }
#endif /* FEATURE_SM35 */  

  if (engine_partition_algorithm() != PAR_SORTED_ASC) {
    count = frontier_count_gpu(&state->frontier, par->streams[1]);
  }
  if (count == 0) return;

  if (use_frontier) {
    frontier_update_list_gpu(&state->frontier, par->streams[1]);
  }

  // Call the GRAPH500 kernel
  const graph500_gpu_func_t GRAPH500_GPU_FUNC[][2] = {{
    // RANDOM algorithm
      graph500_launch_gpu<VWARP_MEDIUM_WARP_WIDTH, 
                          VWARP_MEDIUM_BATCH_SIZE, false>,
      graph500_launch_gpu<VWARP_MEDIUM_WARP_WIDTH, 
                          VWARP_MEDIUM_BATCH_SIZE, true>
    }, {
      // HIGH partitioning
      graph500_launch_gpu<VWARP_MEDIUM_WARP_WIDTH, 
                          VWARP_LARGE_BATCH_SIZE, false>,
      graph500_launch_gpu<VWARP_MEDIUM_WARP_WIDTH, VWARP_LARGE_BATCH_SIZE, true>
    }, {
      // LOW partitioning
      graph500_launch_gpu<MAX_THREADS_PER_BLOCK, 
                          VWARP_MEDIUM_BATCH_SIZE, false>,
      graph500_launch_gpu<MAX_THREADS_PER_BLOCK, VWARP_MEDIUM_BATCH_SIZE, true>
    }
  };
  int par_alg = engine_partition_algorithm();
  GRAPH500_GPU_FUNC[par_alg][use_frontier]
    (par, state, state->frontier.list, count, par->streams[1]);
  CALL_CU_SAFE(cudaGetLastError());
}

void graph500_gpu(partition_t* par, graph500_state_t* state) {

  // Copy the current state of the remote vertices bitmap
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
    bitmap_t rmt_bitmap = (bitmap_t)par->outbox[pid].push_values;
    bitmap_copy_gpu(state->visited[pid], rmt_bitmap, par->outbox[pid].count);
  }

  graph500_tuned_launch_gpu(par, state);

  // Diff the remote vertices bitmaps so that only the vertices who got set
  // in this round are notified
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
    bitmap_t rmt_bitmap = (bitmap_t)par->outbox[pid].push_values;
    bitmap_diff_gpu(state->visited[pid], rmt_bitmap, par->outbox[pid].count);
  }
}

PRIVATE void graph500(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  if (par->processor.type == PROCESSOR_CPU) {
    graph500_cpu(par, state);
  } else if (par->processor.type == PROCESSOR_GPU) {
    graph500_gpu(par, state);
  } else {
    fprintf(stderr, "Unsupported processor type\n");
    assert(false);
  }
}

PRIVATE inline void 
graph500_scatter_cpu(grooves_box_table_t* inbox, graph500_state_t* state, 
                     bitmap_t visited, vid_t* tree, int rmt_pid) {
  bitmap_t rmt_bitmap = (bitmap_t)inbox->push_values;
  OMP(omp parallel for schedule(runtime))
  for (vid_t word_index = 0; word_index < bitmap_bits_to_words(inbox->count); 
       word_index++) {
    if (rmt_bitmap[word_index]) {
      vid_t bit_index = word_index * BITMAP_BITS_PER_WORD;
      vid_t bit_last_index = (word_index + 1) * BITMAP_BITS_PER_WORD;
      for (; bit_index < bit_last_index; bit_index++) {
        if (bitmap_is_set(rmt_bitmap, bit_index)) {
          vid_t vid = inbox->rmt_nbrs[bit_index];
          if (!bitmap_is_set(visited, vid)) {
            bitmap_set_cpu(visited, vid);
            tree[vid] = SET_PARTITION_ID(GET_VERTEX_ID(VERTEX_ID_MAX), rmt_pid);
          }
        }
      }
    }
  }
}

template<int VWARP_WIDTH, int BATCH_SIZE, int THREADS_PER_BLOCK>
PRIVATE __global__ void
graph500_scatter_kernel(const bitmap_t __restrict rmt_bitmap,
                        const vid_t* __restrict rmt_nbrs, vid_t word_count,
                        bitmap_t visited, vid_t* tree, int rmt_pid) {
  if (THREAD_GLOBAL_INDEX >= 
      vwarp_thread_count(word_count, VWARP_WIDTH, BATCH_SIZE)) return;
  vid_t start_word = vwarp_warp_start_vertex(VWARP_WIDTH, BATCH_SIZE) +
    vwarp_block_start_vertex(VWARP_WIDTH, BATCH_SIZE, THREADS_PER_BLOCK);
  vid_t end_word = start_word +
    vwarp_warp_batch_size(word_count, VWARP_WIDTH, BATCH_SIZE, 
                          THREADS_PER_BLOCK);
  int warp_offset = vwarp_thread_index(VWARP_WIDTH);
  for(vid_t k = start_word; k < end_word; k++) {
    bitmap_word_t word = rmt_bitmap[k];
    if (word == 0) continue;
    vid_t start_vertex = k * BITMAP_BITS_PER_WORD;
    for(vid_t i = warp_offset; i < BITMAP_BITS_PER_WORD;
        i += VWARP_WIDTH) {
      if (bitmap_is_set(word, i)) {
        vid_t vid = rmt_nbrs[start_vertex + i];
        if (!bitmap_is_set(visited, vid)) {
          bitmap_set_gpu(visited, vid);
          tree[vid] = SET_PARTITION_ID(GET_VERTEX_ID(VERTEX_ID_MAX), rmt_pid);
        }
      }
    }
  }
}

PRIVATE void graph500_scatter(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_CPU) {
      graph500_scatter_cpu(inbox, state, state->visited[par->id], 
                           state->tree[par->id], rmt_pid);
    } else if (par->processor.type == PROCESSOR_GPU) {
      vid_t word_count = bitmap_bits_to_words(inbox->count);
      dim3 blocks;
      const int batch_size = 8; const int warp_size = 16;
      const int threads = DEFAULT_THREADS_PER_BLOCK;
      kernel_configure(vwarp_thread_count(word_count, warp_size, batch_size),
                       blocks, threads);
      bitmap_t rmt_bitmap = (bitmap_t)inbox->push_values;
      graph500_scatter_kernel<warp_size, batch_size, threads>
        <<<blocks, threads, 0, par->streams[1]>>>
        (rmt_bitmap, inbox->rmt_nbrs, word_count,
         state->visited[par->id], state->tree[par->id], rmt_pid);
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
  bitmap_reset_gpu(state->visited[par->id], par->subgraph.vertex_count, 
                   par->streams[1]);
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      bitmap_reset_gpu(state->visited[pid], par->outbox[pid].count, 
                       par->streams[1]);
      bitmap_reset_gpu((bitmap_t)par->outbox[pid].push_values, 
                       par->outbox[pid].count, par->streams[1]);
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
  frontier_reset_gpu(&state->frontier);
}

PRIVATE inline void graph500_init_cpu(partition_t* par) {
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  bitmap_reset_cpu(state->visited[par->id], par->subgraph.vertex_count);
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      bitmap_reset_cpu(state->visited[pid], par->outbox[pid].count);
      bitmap_reset_cpu((bitmap_t)par->outbox[pid].push_values, 
                       par->outbox[pid].count);
    }
  }
  // set the source vertex as visited
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    bitmap_set_cpu(state->visited[par->id], GET_VERTEX_ID(state_g.src));
  }
  frontier_reset_cpu(&state->frontier);
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
  totem_memset(state->tree[par->id], VERTEX_ID_MAX, par->subgraph.vertex_count, 
               type, par->streams[1]);
  engine_set_outbox(par->id, (vid_t)VERTEX_ID_MAX);
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    totem_memset(&((state->tree[par->id])[GET_VERTEX_ID(state_g.src)]), 
                 state_g.src, 1, type, par->streams[1]);
  }
}

void graph500_free(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_CPU) {
    bitmap_finalize_cpu(state->visited[par->id]);
    frontier_finalize_cpu(&state->frontier);
  } else if (par->processor.type == PROCESSOR_GPU) {
    bitmap_finalize_gpu(state->visited[par->id]);
    frontier_finalize_gpu(&state->frontier);
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
  totem_free(state->tree[par->id], type);
  free(state);
  par->algo_state = NULL;
  
  // try to free global state if not already freed
  if (state_g.initialized) {
    if (engine_largest_gpu_partition()) {
      totem_free(state_g.tree_h, TOTEM_MEM_HOST_PINNED);
    }
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
  frontier_init_cpu(&state->frontier, par->subgraph.vertex_count);
  } else if (par->processor.type == PROCESSOR_GPU) {
    state->visited[par->id] = bitmap_init_gpu(par->subgraph.vertex_count);
    frontier_init_gpu(&state->frontier, par->subgraph.vertex_count);
    type = TOTEM_MEM_DEVICE;
  } else {
    assert(false);
  }
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    vid_t count = par->outbox[pid].count;
    if (pid != par->id && count != 0) {
      if (par->processor.type == PROCESSOR_CPU) {
        state->visited[pid] = bitmap_init_cpu(count);
      } else {
        state->visited[pid] = bitmap_init_gpu(count);
      }
      bitmap_t rmt_bitmap = (bitmap_t)par->outbox[pid].push_values;
      state->tree[pid] = (vid_t*)(&rmt_bitmap[bitmap_bits_to_words(count)]);
    }
  }
  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(vid_t), type, 
                         (void**)&(state->tree[par->id])));
  state->finished = engine_get_finished_ptr(par->id);
  par->algo_state = state;

  // try to initalize the global state if not already initialized
  if (!state_g.initialized) {
    if (engine_largest_gpu_partition()) {
      CALL_SAFE(totem_malloc(engine_largest_gpu_partition() * sizeof(vid_t), 
                             TOTEM_MEM_HOST_PINNED, (void**)&state_g.tree_h));
    }
    state_g.initialized = true;
  }
}

PRIVATE inline void 
rmt_tree_scatter_cpu(grooves_box_table_t* inbox, vid_t* tree, int rmt_pid) {
  bitmap_t rmt_bitmap = (bitmap_t)inbox->push_values;
  vid_t* rmt_tree = (vid_t*)(&rmt_bitmap[bitmap_bits_to_words(inbox->count)]);
  OMP(omp parallel for schedule(runtime))
  for (vid_t index = 0; index < inbox->count; index++) {
    vid_t vid = inbox->rmt_nbrs[index];
    if ((GET_VERTEX_ID(tree[vid]) == GET_VERTEX_ID(VERTEX_ID_MAX)) &&
        (GET_PARTITION_ID(tree[vid]) == rmt_pid)) {
      tree[vid] = rmt_tree[index];
    }
  }
}

PRIVATE __global__ void
rmt_tree_scatter_kernel(const vid_t* __restrict rmt_nbrs, vid_t count, 
                        const vid_t* __restrict rmt_tree, vid_t* tree,
                        int rmt_pid) {
  vid_t index = THREAD_GLOBAL_INDEX;
  if (index >= count) return;
  vid_t vid = rmt_nbrs[index];
  if ((GET_VERTEX_ID(tree[vid]) == GET_VERTEX_ID(VERTEX_ID_MAX)) &&
      (GET_PARTITION_ID(tree[vid]) == rmt_pid)) {
    tree[vid] = rmt_tree[index];
  }
}

PRIVATE void rmt_tree_scatter(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_CPU) {
      rmt_tree_scatter_cpu(inbox, state->tree[par->id], rmt_pid);
    } else if (par->processor.type == PROCESSOR_GPU) {
      const int threads = DEFAULT_THREADS_PER_BLOCK;
      dim3 blocks;
      kernel_configure(inbox->count, blocks, threads);
      bitmap_t rmt_bitmap = (bitmap_t)inbox->push_values;
      vid_t* rmt_tree = 
        (vid_t*)(&rmt_bitmap[bitmap_bits_to_words(inbox->count)]);
      rmt_tree_scatter_kernel<<<blocks, threads, 0, par->streams[1]>>>
        (inbox->rmt_nbrs, inbox->count, rmt_tree, state->tree[par->id], 
         rmt_pid);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(false);
    }
  }
}

void rmt_tree(partition_t* par) {
  if (engine_superstep() == 1) engine_report_not_finished();
}

error_t graph500_hybrid(vid_t src, bfs_tree_t* tree) {
  // check for special cases
  bool finished = false;
  error_t rc = check_special_cases(src, tree, &finished);
  if (finished) return rc;

  // initialize the global state
  state_g.tree = tree;
  state_g.src  = engine_vertex_id_in_partition(src);
  totem_memset(tree, (bfs_tree_t)-1, engine_vertex_count(), TOTEM_MEM_HOST);

  // initialize the engine
  engine_config_t config = {
    NULL, graph500, graph500_scatter, NULL, graph500_init, 
    NULL, NULL, GROOVES_PUSH
  };
  engine_config(&config);
  engine_update_msg_size(GROOVES_PUSH, 1);
  engine_execute();

  // initialize the engine
  engine_config_t config_update_rmt_parents = {
    NULL, rmt_tree, rmt_tree_scatter, NULL, NULL, NULL, 
    graph500_aggregate, GROOVES_PUSH
  };
  engine_config(&config_update_rmt_parents);
  engine_reset_msg_size(GROOVES_PUSH);
  engine_execute();
  return SUCCESS;
}
