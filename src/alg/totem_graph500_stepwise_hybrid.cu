/**
 * This file contains an implementation of the breadth-first search (BFS) graph
 * search algorithm using the totem framework. This is a modified version that
 * performs the algorithm in a Bottom Up fashion.
 *
 * This implementation is modified to output a tree for the Graph500 spec.
 *
 * This implementation only works for undirected graphs.
 *
 * Based off of the work by Scott Beamer et al.
 * Searching for a Parent Instead of Fighting Over Children: A Fast
 * Breadth-First Search Implementation for Graph500.
 * http://www.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-117.pdf
 *
 * TODO(scott): Modify the algorithm to dynamically swap between top down and
 *              bottom up steps.
 *
 *  Created on: 2014-08-26
 *  Authors:    Scott Sallinen
 *              Abdullah Gharaibeh
 */

#include "totem_alg.h"
#include "totem_engine.cuh"

// Per-partition specific state.
typedef struct graph500_state_s {
  vid_t*   tree[MAX_PARTITION_COUNT];    // A list of trees, one for each
                                         // remote partition.
  bitmap_t  visited[MAX_PARTITION_COUNT];   // A list of bitmaps, one for each
                                            // remote partition.
  bitmap_t  frontier[MAX_PARTITION_COUNT];  // A list of bitmaps, one for each
                                            // remote partition.
  bool*     finished;          // Points to Totem's finish flag.
  cost_t    level;             // Current level to process by the partition.
  frontier_state_t frontier_state;   // Frontier management state.
  bool      skip_gather;      // Whether to skip the gather in the round.
  vid_t*   local_to_global[MAX_PARTITION_COUNT];    // Local to global id map.
} graph500_state_t;

// State shared between all partitions.
typedef struct graph500_global_state_s {
  bfs_tree_t*   tree;     // Final tree output buffer.
  vid_t   src;      // Source vertex id. (The id after partitioning.)
  bool    bu_step;  // Whether or not to perform a bottom up step.
  double    switch_parameter;    // Used to determine when to switch from
                                 // top-down.
  vid_t*  tree_h[MAX_PARTITION_COUNT];  // Used as a temporary buffer to receive
                                        // the final result copied back from GPU
                                        // partitions before being copied again
                                        // to the final output buffer.
} graph500_global_state_t;
PRIVATE graph500_global_state_t state_g = {NULL, 0, false, 0};

// Checks for input parameters and special cases. This is invoked at the
// beginning of public interfaces (GPU and CPU)
PRIVATE error_t check_special_cases(vid_t src, bfs_tree_t* tree,
                                    bool* finished) {
  *finished = true;
  if ((src >= engine_vertex_count()) || (tree == NULL)) {
    return FAILURE;
  } else if (engine_vertex_count() == 1) {
    tree[0] = src;
    return SUCCESS;
  } else if (engine_edge_count() == 0) {
    // Initialize tree to INFINITE.
    totem_memset(tree, (bfs_tree_t)(-1), engine_vertex_count(), TOTEM_MEM_HOST);
    tree[src] = src;
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

// A step that iterates across the frontier of vertices and adds their
// neighbours to the next frontier.
PRIVATE void graph500_td_cpu(partition_t* par, graph500_state_t* state) {
  graph_t* subgraph = &par->subgraph;
  bool finished = true;

  const vid_t* local_to_global = state->local_to_global[par->id];

  // Iterate across all of our vertices.
  vid_t edge_frontier_count = 0;
  OMP(omp parallel for schedule(runtime) reduction(& : finished)
      reduction(+ : edge_frontier_count))
  for (vid_t vertex_id = 0; vertex_id < subgraph->vertex_count; vertex_id++) {
    // Ignore the local vertex if it is not in the frontier.
    if (!bitmap_is_set(state->frontier[par->id], vertex_id)) { continue; }

    // Iterate across the neighbours of this vertex.
    for (eid_t i = subgraph->vertices[vertex_id];
         i < subgraph->vertices[vertex_id + 1]; i++) {
      int nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
      vid_t nbr = GET_VERTEX_ID(subgraph->edges[i]);

      // Add the neighbour we are exploring to the next frontier.
      if (!bitmap_is_set(state->visited[nbr_pid], nbr)) {
        if (bitmap_set_cpu(state->visited[nbr_pid], nbr)) {
          // Locate the tree corresponding to the neighbour.
          vid_t* tree = state->tree[nbr_pid];

          // Add the vertex to the tree.
          tree[nbr] = local_to_global[vertex_id];
          if (nbr_pid == par->id) {
            edge_frontier_count +=
                subgraph->vertices[nbr + 1] - subgraph->vertices[nbr];
          }
          finished = false;
        }
      }
    }  // End of neighbour check - vertex examined.
  }  // All vertices examined in level.
  state_g.switch_parameter =
      100.0 * edge_frontier_count / par->subgraph.edge_count;

  // Move over the finished variable.
  if (!finished) *(state->finished) = false;
}

// A step that iterates across unvisited vertices and determines
// their status in the next frontier.
PRIVATE void graph500_bu_cpu(partition_t* par, graph500_state_t* state) {
  graph_t* subgraph = &par->subgraph;
  bool finished = true;
  bitmap_t visited = state->visited[par->id];

  // Locate the tree corresponding to our partition.
  vid_t* tree = state->tree[par->id];

  // Iterate across all of our vertices.
  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t vertex_id = 0; vertex_id < subgraph->vertex_count; vertex_id++) {
    // Ignore the local vertex if it has already been visited.
    if (bitmap_is_set(visited, vertex_id)) { continue; }

    // Iterate across the neighbours of this vertex.
    for (eid_t i = subgraph->vertices[vertex_id];
         i < subgraph->vertices[vertex_id + 1]; i++) {
      int nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
      vid_t nbr = GET_VERTEX_ID(subgraph->edges[i]);

      // Check if the bitmap corresponding to the vertices PID is set.
      // This means the partition that the vertex belongs to, has explored it.
      if (bitmap_is_set(state->frontier[nbr_pid], nbr)) {
        // Add the vertex we are exploring to the next frontier.
        bitmap_set_cpu(visited, vertex_id);

        // Add the vertex to the tree.
        const vid_t* local_to_global = state->local_to_global[nbr_pid];
        tree[vertex_id] = local_to_global[nbr];
        finished = false;
        break;
      }
    }  // End of neighbour check - vertex examined.
  }  // All vertices examined in level.

  // Move over the finished variable.
  if (!finished) *(state->finished) = false;
}

// This is a CPU version of the Bottom-up/Top-down BFS algorithm.
// See file header for full details.
void graph500_stepwise_cpu(partition_t* par, graph500_state_t* state) {
  // Update the frontier.
  frontier_update_bitmap_cpu(&state->frontier_state, state->visited[par->id]);
  state->frontier[par->id] = state->frontier_state.current;

  if (state_g.bu_step) {
    // Execute a bottom up step.
    graph500_bu_cpu(par, state);
  } else {
    // Copy the current state of the remote vertices bitmap.
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      if ((pid == par->id) || (par->outbox[pid].count == 0)) { continue; }
      bitmap_copy_cpu(state->visited[pid],
                      (bitmap_t)par->outbox[pid].push_values,
                      par->outbox[pid].count);
    }

    // Execute a top down step.
    graph500_td_cpu(par, state);

    // Diff the remote vertices bitmaps so that only the vertices who got set
    // in this round are notified.
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      if ((pid == par->id) || (par->outbox[pid].count == 0)) { continue; }
      bitmap_diff_cpu(state->visited[pid],
                      (bitmap_t)par->outbox[pid].push_values,
                      par->outbox[pid].count);
    }
  }
}

template<typename eid_type>
PRIVATE __device__ inline vid_t* get_edges_array(
    const graph_t& graph, const eid_type* __restrict vertices, vid_t v) {
  vid_t* edges = graph.edges + vertices[v];
  if (v >= graph.vertex_ext) {
    edges = graph.edges_ext + (vertices[v] - graph.edge_count_ext);
  }
  return edges;
}

// A gpu version of the Bottom-up step as a kernel.
template<int THREADS_PER_BLOCK, typename eid_type>
__global__ void graph500_bu_kernel(partition_t par, graph500_state_t state,
                                   const eid_type* __restrict vertices) {
  const int kWarpWidth = 1;
  const vid_t kVertexCount = par.subgraph.vertex_count;
  const vid_t kVerticesPerBlock = BITMAP_BITS_PER_WORD * THREADS_PER_BLOCK;
  __shared__ bool nbrs_state[kVerticesPerBlock];

  vid_t block_start_index = BLOCK_GLOBAL_INDEX * kVerticesPerBlock;
  vid_t block_batch = kVerticesPerBlock;
  if (block_start_index + kVerticesPerBlock > kVertexCount) {
    block_batch = kVertexCount - block_start_index;
  }
  const bitmap_t __restrict kVisited = state.visited[par.id];
  for (int i = THREAD_BLOCK_INDEX; i < block_batch; i += THREADS_PER_BLOCK) {
    vid_t v = block_start_index + i;
    nbrs_state[i] = false;
    if (!bitmap_is_set(kVisited, v)) {
      const vid_t* __restrict edges = get_edges_array(
          par.subgraph, vertices, v);
      const eid_t nbr_count = vertices[v + 1] - vertices[v];
      for (eid_t e = 0; e < nbr_count && !nbrs_state[i]; e++) {
        int nbr_pid = GET_PARTITION_ID(edges[e]);
        vid_t nbr = GET_VERTEX_ID(edges[e]);
        if (bitmap_is_set(state.frontier[nbr_pid], nbr)) {
          nbrs_state[i] = true;
          vid_t* tree = state.tree[par.id];
          const vid_t* local_to_global = state.local_to_global[nbr_pid];
          tree[v] = local_to_global[nbr];
        }
      }
    }
  }
  __shared__ bool finished_block;
  finished_block = true;
  __syncthreads();

  if (THREAD_GLOBAL_INDEX >=
      vwarp_thread_count(kVertexCount, kWarpWidth, BITMAP_BITS_PER_WORD)) {
    return;
  }

  vid_t start_vertex =
      vwarp_block_start_vertex(kWarpWidth, BITMAP_BITS_PER_WORD,
                               THREADS_PER_BLOCK) +
      vwarp_warp_start_vertex(kWarpWidth, BITMAP_BITS_PER_WORD);
  vid_t batch_size = vwarp_warp_batch_size(
      kVertexCount, kWarpWidth, BITMAP_BITS_PER_WORD, THREADS_PER_BLOCK);

  const bool* my_nbrs = &nbrs_state[THREAD_BLOCK_INDEX * BITMAP_BITS_PER_WORD];
  bitmap_t visited = state.visited[par.id];
  bitmap_word_t word = visited[start_vertex / BITMAP_BITS_PER_WORD];
  for (vid_t k = 0; k < batch_size; k++) {
    bitmap_word_t mask = (bitmap_word_t)1 << k;
    if (my_nbrs[k]) {
      word |= mask;
      finished_block = false;
    }
  }
  visited[start_vertex / BITMAP_BITS_PER_WORD] = word;

  // Move over the finished variable.
  __syncthreads();
  if (!finished_block && THREAD_BLOCK_INDEX == 0) *state.finished = false;
}

PRIVATE void graph500_bu_gpu(partition_t* par, graph500_state_t* state) {
  // The batch size is fixed at BITMAP_BITS_PER_WORD. This is necessary to avoid
  // using atomic operations when setting the visited bitmap.
  const int threads = DEFAULT_THREADS_PER_BLOCK;
  dim3 blocks;
  kernel_configure(vwarp_thread_count(par->subgraph.vertex_count,
                                      1 /* warp width */,
                                      BITMAP_BITS_PER_WORD), blocks, threads);
  if (par->subgraph.compressed_vertices) {
    graph500_bu_kernel<threads><<<blocks, threads, 0, par->streams[1]>>>
        (*par, *state, par->subgraph.vertices_d);
  } else {
    graph500_bu_kernel<threads><<<blocks, threads, 0, par->streams[1]>>>
        (*par, *state, par->subgraph.vertices);
  }
}

// A warp-based implementation of the top-down graph500 kernel.
template<int VWARP_WIDTH, int VWARP_BATCH, typename eid_type>
__global__ void graph500_td_kernel(partition_t par, graph500_state_t state,
                                   const eid_type* __restrict vertices,
                                   const vid_t* __restrict frontier,
                                   vid_t count) {
  if (THREAD_GLOBAL_INDEX >=
      vwarp_thread_count(count, VWARP_WIDTH, VWARP_BATCH)) { return; }

  // This flag is used to report the finish state of a block of threads. This
  // is useful to avoid having many threads writing to the global finished
  // flag, which can hurt performance (since "finished" is actually allocated
  // on the host, and each write will cause a transfer over the PCI-E bus).
  __shared__ bool finished_block;
  finished_block = true;
  __syncthreads();

  vid_t start_vertex = vwarp_block_start_vertex(VWARP_WIDTH, VWARP_BATCH) +
    vwarp_warp_start_vertex(VWARP_WIDTH, VWARP_BATCH);
  vid_t end_vertex = start_vertex +
    vwarp_warp_batch_size(count, VWARP_WIDTH, VWARP_BATCH);
  int warp_offset = vwarp_thread_index(VWARP_WIDTH);

  const vid_t* local_to_global = state.local_to_global[par.id];

  for (vid_t i = start_vertex; i < end_vertex; i++) {
    vid_t v = frontier[i];
    const eid_t nbr_count = vertices[v + 1] - vertices[v];
    const vid_t* __restrict edges = get_edges_array(par.subgraph, vertices, v);
    for (vid_t i = warp_offset; i < nbr_count; i += VWARP_WIDTH) {
      int nbr_pid = GET_PARTITION_ID(edges[i]);
      vid_t nbr = GET_VERTEX_ID(edges[i]);
      bitmap_t visited = state.visited[nbr_pid];
      if (!bitmap_is_set(visited, nbr)) {
        if (bitmap_set_gpu(visited, nbr)) {
          vid_t* tree = state.tree[nbr_pid];
          tree[nbr] = local_to_global[v];
          finished_block = false;
        }
      }
    }
  }

  __syncthreads();
  if (!finished_block && THREAD_BLOCK_INDEX == 0) *state.finished = false;
}

template<int VWARP_WIDTH, int BATCH_SIZE>
#ifdef FEATURE_SM35
PRIVATE __host__ __device__
#else
PRIVATE __host__
#endif  // FEATURE_SM35
void graph500_td_launch_gpu(partition_t* par, graph500_state_t* state,
                            vid_t* frontier_list, vid_t vertex_count,
                            cudaStream_t stream) {
  const int threads = MAX_THREADS_PER_BLOCK;
  dim3 blocks;
  assert(VWARP_WIDTH <= threads);
  kernel_configure(vwarp_thread_count(vertex_count, VWARP_WIDTH, BATCH_SIZE),
                   blocks, threads);
  if (par->subgraph.compressed_vertices) {
    graph500_td_kernel<VWARP_WIDTH, BATCH_SIZE><<<blocks, threads, 0, stream>>>
        (*par, *state, par->subgraph.vertices_d, frontier_list, vertex_count);
  } else {
    graph500_td_kernel<VWARP_WIDTH, BATCH_SIZE><<<blocks, threads, 0, stream>>>
        (*par, *state, par->subgraph.vertices, frontier_list, vertex_count);
  }
}

typedef void(*graph500_td_gpu_func_t)(partition_t*, graph500_state_t*, vid_t*,
                                      vid_t, cudaStream_t);

#ifdef FEATURE_SM35
PRIVATE __global__
void graph500_td_launch_at_boundary_gpu(
    partition_t par, graph500_state_t state) {
  if (THREAD_GLOBAL_INDEX > 0 || (*state.frontier_state.count == 0)) {
    return;
  }
  const graph500_td_gpu_func_t graph500_GPU_FUNC[] = {
    graph500_td_launch_gpu<1,   2>,   // (0) < 8
    graph500_td_launch_gpu<8,   8>,   // (1) > 8    && < 32
    graph500_td_launch_gpu<32,  32>,   // (2) > 32   && < 128
    graph500_td_launch_gpu<128, 32>,   // (3) > 128  && < 256
    graph500_td_launch_gpu<256, 32>,   // (4) > 256  && < 1K
    graph500_td_launch_gpu<512, 32>,   // (5) > 1K   && < 2K
    graph500_td_launch_gpu<MAX_THREADS_PER_BLOCK, 8>,  // (6) > 2k
  };

  int64_t end = *(state.frontier_state.count);
  for (int i = FRONTIER_BOUNDARY_COUNT; i >= 0; i--) {
    int64_t start = state.frontier_state.boundaries[i];
    int64_t count = end - start;
    if (count > 0) {
      cudaStream_t s;
      cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
      graph500_GPU_FUNC[i](&par, &state, state.frontier_state.list + start,
                      count, s);
      end = start;
    }
  }
}
#endif  // FEATURE_SM35

PRIVATE void graph500_td_gpu(partition_t* par, graph500_state_t* state) {
#ifdef FEATURE_SM35
  if (engine_sorted()) {
    frontier_update_list_gpu(&state->frontier_state, par->streams[1]);
    frontier_update_boundaries_gpu(&state->frontier_state, &par->subgraph,
                                   par->streams[1]);
    graph500_td_launch_at_boundary_gpu<<<1, 1, 0, par->streams[1]>>>
        (*par, *state);
    CALL_CU_SAFE(cudaGetLastError());
    return;
  }
#endif  // FEATURE_SM35

  // Call the graph500 kernel.
  const graph500_td_gpu_func_t graph500_GPU_FUNC[] = {
    // RANDOM algorithm
    graph500_td_launch_gpu<VWARP_MEDIUM_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE>,
    // HIGH partitioning
    graph500_td_launch_gpu<VWARP_MEDIUM_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE>,
    // LOW partitioning
    graph500_td_launch_gpu<MAX_THREADS_PER_BLOCK, VWARP_MEDIUM_BATCH_SIZE>
  };
  int par_alg = engine_partition_algorithm();
  vid_t count = frontier_count_gpu(&state->frontier_state, par->streams[1]);
  if (count == 0) { return; }
  frontier_update_list_gpu(&state->frontier_state, par->streams[1]);
  graph500_GPU_FUNC[par_alg](par, state, state->frontier_state.list, count,
                        par->streams[1]);
  CALL_CU_SAFE(cudaGetLastError());
}

// This is a GPU version of the Bottom-up/Top-down graph500 algorithm.
// See file header for full details.
__host__ void graph500_stepwise_gpu(partition_t* par, graph500_state_t* state) {
  // Update the frontier.
  frontier_update_bitmap_gpu(
      &state->frontier_state, state->visited[par->id], par->streams[1]);
  state->frontier[par->id] = state->frontier_state.current;

  if (state_g.bu_step) {
    // Execute a bottom up step.
    graph500_bu_gpu(par, state);
  } else {
    // Copy the current state of the remote vertices bitmap.
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      if ((pid == par->id) || (par->outbox[pid].count == 0)) { continue; }
      bitmap_copy_gpu(state->visited[pid],
                      (bitmap_t)par->outbox[pid].push_values,
                      par->outbox[pid].count, par->streams[1]);
    }

    // Execute a top down step.
    graph500_td_gpu(par, state);

    // Diff the remote vertices bitmaps so that only the vertices who got set
    // in this round are notified.
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      if ((pid == par->id) || (par->outbox[pid].count == 0)) { continue; }
      bitmap_diff_gpu(state->visited[pid],
                      (bitmap_t)par->outbox[pid].push_values,
                      par->outbox[pid].count, par->streams[1]);
    }
  }
}

// The execution phase - based off of the partition we are, launch an approach.
PRIVATE void graph500(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  graph500_state_t* state =
      reinterpret_cast<graph500_state_t*>(par->algo_state);

  // Ignore the first round - this allows us to communicate the frontier with
  // an updated visited status of the source vertex.
  if (engine_superstep() == 1 && state_g.bu_step) {
    state->skip_gather = false;
    engine_report_not_finished();
    return;
  }

  // The switching thresholds has been determined empirically. Consider looking
  // at them again if they did not work for specific workloads.
  if ((state_g.switch_parameter >= 2 && state_g.bu_step == false) ||
      (engine_superstep() == 5 && state_g.bu_step)) {
    state->skip_gather = true;
    return;
  }

  // Launch the processor specific algorithm.
  if (par->processor.type == PROCESSOR_CPU) {
    graph500_stepwise_cpu(par, state);
  } else if (par->processor.type == PROCESSOR_GPU) {
    graph500_stepwise_gpu(par, state);
  } else {
    assert(false);
  }

  // At the end of the round, increase our BFS level.
  state->level++;
}

PRIVATE void graph500_gather_cpu(partition_t* par, graph500_state_t* state,
                                 grooves_box_table_t* inbox) {
  const vid_t words = bitmap_bits_to_words(inbox->count);
  bitmap_t bitmap = reinterpret_cast<bitmap_t>(inbox->pull_values);

  // Iterate across the items in the inbox.
  OMP(omp parallel for schedule(runtime))
  for (vid_t word_index = 0; word_index < words; word_index++) {
    vid_t start_index = word_index * BITMAP_BITS_PER_WORD;
    bitmap_word_t word = bitmap[word_index];
    for (int i = 0; i < BITMAP_BITS_PER_WORD; i++) {
      vid_t index = start_index + i;
      if (index >= inbox->count) { break; }
      bitmap_word_t mask = ((bitmap_word_t)1) << i;
      if (word & mask) { continue; }
      vid_t vid = inbox->rmt_nbrs[index];
      if (bitmap_is_set(state->visited[par->id], vid)) { word |= mask; }
    }
    bitmap[word_index] = word;
  }
}

// Gather for the GPU bitmap to inbox.
template<int THREADS_PER_BLOCK>
__global__ void graph500_gather_gpu(partition_t par, graph500_state_t state,
                                    grooves_box_table_t inbox) {
  __shared__ bool active[BITMAP_BITS_PER_WORD * THREADS_PER_BLOCK];
  const vid_t kVerticesPerBlock = THREADS_PER_BLOCK * BITMAP_BITS_PER_WORD;
  vid_t block_start_index = BLOCK_GLOBAL_INDEX * kVerticesPerBlock;
  vid_t block_batch = kVerticesPerBlock;
  if (block_start_index + kVerticesPerBlock > inbox.count) {
    block_batch = inbox.count - block_start_index;
  }
  for (int i = THREAD_BLOCK_INDEX; i < block_batch; i += THREADS_PER_BLOCK) {
    vid_t vid = inbox.rmt_nbrs[block_start_index + i];
    active[i] = bitmap_is_set(state.visited[par.id], vid);
  }
  __syncthreads();

  bool* my_active = &active[THREAD_BLOCK_INDEX * BITMAP_BITS_PER_WORD];
  const vid_t word_index = THREAD_GLOBAL_INDEX;
  vid_t start_index = word_index * BITMAP_BITS_PER_WORD;
  if (start_index >= inbox.count) { return; }

  bitmap_t bitmap = reinterpret_cast<bitmap_t>(inbox.pull_values);
  bitmap_word_t word = 0;
  vid_t batch = start_index + BITMAP_BITS_PER_WORD <= inbox.count ?
      BITMAP_BITS_PER_WORD : inbox.count - start_index;
  for (int i = 0; i < batch; i++) {
    bitmap_word_t mask = ((bitmap_word_t)1) << i;
    if (my_active[i]) { word |= mask; }
  }
  bitmap[word_index] = word;
}

// The gather phase - apply values from the inboxes to the partitions' local
// variables.
PRIVATE void graph500_gather(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  graph500_state_t* state =
      reinterpret_cast<graph500_state_t*>(par->algo_state);

  // Skip the communication on the final transitional round.
  if (state->skip_gather) { return; }

  // Across all partitions that are not us.
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) { continue; }

    // Select the inbox to apply to.
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (inbox->count == 0) { continue; }

    // Select a method based off of our processor type.
    if (par->processor.type == PROCESSOR_CPU) {
      graph500_gather_cpu(par, state, inbox);
    } else if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks;
      const int threads = DEFAULT_THREADS_PER_BLOCK;
      kernel_configure(bitmap_bits_to_words(inbox->count), blocks, threads);
      graph500_gather_gpu<threads><<<blocks, threads, 0, par->streams[1]>>>
          (*par, *state, *inbox);
    } else {
      assert(false);
    }
  }
}

// This is a scatter for CPU - copied from the original bfs_hybrid algorithm.
PRIVATE inline void graph500_scatter_cpu(partition_t* par) {
  graph500_state_t* state =
      reinterpret_cast<graph500_state_t*>(par->algo_state);
  bitmap_t visited = state->visited[par->id];
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) { continue; }
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) { continue; }
    bitmap_t remotely_visited = (bitmap_t)inbox->push_values;
    vid_t* tree = state->tree[par->id];
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
              tree[vid] = SET_PARTITION_ID(GET_VERTEX_ID(VERTEX_ID_MAX),
                                           rmt_pid);
            }
          }
        }
      }
    }
  }
}

// This is a scatter for GPU - copied from the original bfs_hybrid algorithm.
template<int VWARP_WIDTH, int BATCH_SIZE, int THREADS_PER_BLOCK>
__global__ void
graph500_scatter_kernel(const bitmap_t __restrict rmt_visited,
                   const vid_t* __restrict rmt_nbrs, vid_t word_count,
                   bitmap_t visited, vid_t* tree, int rmt_pid) {
  if (THREAD_GLOBAL_INDEX >=
      vwarp_thread_count(word_count, VWARP_WIDTH, BATCH_SIZE)) { return; }
  vid_t start_word = vwarp_warp_start_vertex(VWARP_WIDTH, BATCH_SIZE) +
    vwarp_block_start_vertex(VWARP_WIDTH, BATCH_SIZE, THREADS_PER_BLOCK);
  vid_t end_word = start_word +
    vwarp_warp_batch_size(word_count, VWARP_WIDTH, BATCH_SIZE,
                          THREADS_PER_BLOCK);
  int warp_offset = vwarp_thread_index(VWARP_WIDTH);
  for (vid_t k = start_word; k < end_word; k++) {
    bitmap_word_t word = rmt_visited[k];
    if (word == 0) { continue; }
    vid_t start_vertex = k * BITMAP_BITS_PER_WORD;
    for (vid_t i = warp_offset; i < BITMAP_BITS_PER_WORD; i += VWARP_WIDTH) {
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

PRIVATE void graph500_scatter_gpu(partition_t* par) {
  graph500_state_t* state =
      reinterpret_cast<graph500_state_t*>(par->algo_state);
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) { continue; }
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) { continue; }
    vid_t word_count = bitmap_bits_to_words(inbox->count);
    dim3 blocks;
    const int batch_size = 8; const int warp_size = 16;
    const int threads = DEFAULT_THREADS_PER_BLOCK;
    kernel_configure(vwarp_thread_count(word_count, warp_size, batch_size),
                     blocks, threads);
    graph500_scatter_kernel<warp_size, batch_size, threads>
        <<<blocks, threads, 0, par->streams[1]>>>
        ((bitmap_t)inbox->push_values, inbox->rmt_nbrs, word_count,
         state->visited[par->id], state->tree[par->id], rmt_pid);
    CALL_CU_SAFE(cudaGetLastError());
  }
}

// The main scatter function, used in the top down phases.
PRIVATE void graph500_scatter(partition_t* par) {
  if (par->processor.type == PROCESSOR_CPU) {
    graph500_scatter_cpu(par);
  } else if (par->processor.type == PROCESSOR_GPU) {
    graph500_scatter_gpu(par);
  } else {
    assert(false);
  }
}

// A simple kernel that sets the source vertex to visited on the GPU.
__global__ void graph500_init_source_kernel(bitmap_t visited, vid_t src) {
  if (THREAD_GLOBAL_INDEX != 0) { return; }
  bitmap_set_gpu(visited, src);
}

// Initialize the GPU memory - bitmaps and frontier.
PRIVATE inline void graph500_init_gpu(partition_t* par) {
  graph500_state_t* state =
      reinterpret_cast<graph500_state_t*>(par->algo_state);

  // Reset the visited bitmap.
  bitmap_reset_gpu(state->visited[par->id], par->subgraph.vertex_count,
                   par->streams[1]);

  // Initialize other partitions frontier bitmaps.
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    vid_t count = par->outbox[pid].count;
    // Assign the outboxes to our frontier bitmap pointers.
    if (pid != par->id && count != 0) {
      state->frontier[pid] =
        reinterpret_cast<bitmap_t>(par->outbox[pid].pull_values);

      // Reset the visited bitmaps of other partitions.
      bitmap_reset_gpu(state->visited[pid], count, par->streams[1]);

      // Clear the outboxes (push values).
      bitmap_reset_gpu(reinterpret_cast<bitmap_t>(par->outbox[pid].push_values),
                       count, par->streams[1]);

    }

    // Clear the inboxes (pull values), and also their shadows.
    if (pid != par->id && par->inbox[pid].count != 0) {
      bitmap_reset_gpu(reinterpret_cast<bitmap_t>(par->inbox[pid].pull_values),
                       par->inbox[pid].count, par->streams[1]);
      bitmap_reset_gpu(reinterpret_cast<bitmap_t>
                       (par->inbox[pid].pull_values_s),
                       par->inbox[pid].count, par->streams[1]);
    }
  }

  // Reset the local frontier.
  frontier_reset_gpu(&state->frontier_state);

  // Set the source vertex as visited, if it is in our partition.
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    graph500_init_source_kernel<<<1, 1, 0, par->streams[1]>>>
        (state->visited[par->id], GET_VERTEX_ID(state_g.src));
    CALL_CU_SAFE(cudaGetLastError());
  }
}

// Initialize the CPU memory - bitmaps and frontier.
PRIVATE inline void graph500_init_cpu(partition_t* par) {
  graph500_state_t* state =
      reinterpret_cast<graph500_state_t*>(par->algo_state);

  // Clear the visited bitmap.
  bitmap_reset_cpu(state->visited[par->id], par->subgraph.vertex_count);

  // Initialize other partitions bitmaps.
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    vid_t count = par->outbox[pid].count;
    // Assign the outboxes to our frontier bitmap pointers.
    if (pid != par->id && count != 0) {
      state->frontier[pid] =
          reinterpret_cast<bitmap_t>(par->outbox[pid].pull_values);

      // Reset the visited bitmaps of other partitions.
      bitmap_reset_cpu(state->visited[pid], count);

      // Clear the push values.
      bitmap_reset_cpu(
          reinterpret_cast<bitmap_t>(par->outbox[pid].push_values), count);
    }

    // Clear the inboxes, and also their shadows.
    if (pid != par->id && par->inbox[pid].count != 0) {
      bitmap_reset_cpu(reinterpret_cast<bitmap_t>(par->inbox[pid].pull_values),
                       par->inbox[pid].count);
      bitmap_reset_cpu(reinterpret_cast<bitmap_t>
                       (par->inbox[pid].pull_values_s), par->inbox[pid].count);
    }
  }

  // Reset the local frontier.
  frontier_reset_cpu(&state->frontier_state);

  // Set the source vertex as visited, if it is in our partition.
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    bitmap_set_cpu(state->visited[par->id], GET_VERTEX_ID(state_g.src));
  }
}

// The init phase - Set up the memory and statuses.
PRIVATE void graph500_init(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  graph500_state_t* state =
      reinterpret_cast<graph500_state_t*>(par->algo_state);
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_CPU) {
    graph500_init_cpu(par);
  } else if (par->processor.type == PROCESSOR_GPU) {
    type = TOTEM_MEM_DEVICE;
    graph500_init_gpu(par);
  } else {
    assert(false);
  }

  // Reset the local tree.
  totem_memset(state->tree[par->id], VERTEX_ID_MAX, par->subgraph.vertex_count,
               type, par->streams[1]);

  if (GET_PARTITION_ID(state_g.src) == par->id) {
    // For the source vertex, initialize tree.
    totem_memset(&((state->tree[par->id])[GET_VERTEX_ID(state_g.src)]),
                 engine_vertex_id_local_to_global(state_g.src), 1, type,
                 par->streams[1]);
  }

  // Set level 0 to start, and finished pointer.
  state->finished = engine_get_finished_ptr(par->id);
  state->level = 0;
  state_g.switch_parameter = 0;
}

PRIVATE void graph500_aggregate(partition_t* par) {
  if (!par->subgraph.vertex_count) { return; }
  graph500_state_t* state =
      reinterpret_cast<graph500_state_t*>(par->algo_state);
  if (par->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaMemcpyAsync(state_g.tree_h[par->id], state->tree[par->id],
                                 par->subgraph.vertex_count * sizeof(vid_t),
                                 cudaMemcpyDefault, par->streams[1]));
  } else {
    state_g.tree_h[par->id] = state->tree[par->id];
  }
}

PRIVATE void graph500_alloc_prepare_maps(partition_t* par) {
  graph500_state_t* state =
      reinterpret_cast<graph500_state_t*>(par->algo_state);
  if (par->processor.type == PROCESSOR_CPU) {
    state->local_to_global[par->id] = par->map;
  } else {
    CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(vid_t),
                           TOTEM_MEM_DEVICE,
                           (void**)&(state->local_to_global[par->id])));
    CALL_CU_SAFE(cudaMemcpyAsync(state->local_to_global[par->id], par->map,
                                 par->subgraph.vertex_count * sizeof(vid_t),
                                 cudaMemcpyDefault, par->streams[1]));
  }

  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid == par->id) { continue; }
    vid_t* local_to_global;
    if (par->processor.type == PROCESSOR_CPU) {
      local_to_global = par->outbox[pid].rmt_nbrs;
    } else {
      CALL_SAFE(totem_malloc(par->outbox[pid].count * sizeof(vid_t),
                             TOTEM_MEM_HOST, (void**)&(local_to_global)));
      CALL_CU_SAFE(cudaMemcpyAsync(local_to_global,
                                   par->outbox[pid].rmt_nbrs,
                                   par->outbox[pid].count * sizeof(vid_t),
                                   cudaMemcpyDefault, par->streams[1]));
      CALL_CU_SAFE(cudaStreamSynchronize(par->streams[1]));
    }

    for (vid_t i = 0; i < par->outbox[pid].count; i++) {
      local_to_global[i] = engine_vertex_id_local_to_global(
          SET_PARTITION_ID(local_to_global[i], pid));
    }

    if (par->processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaMemcpyAsync(par->outbox[pid].rmt_nbrs,
                                   local_to_global,
                                   par->outbox[pid].count * sizeof(vid_t),
                                   cudaMemcpyDefault, par->streams[1]));
      CALL_CU_SAFE(cudaStreamSynchronize(par->streams[1]));
      totem_free(local_to_global, TOTEM_MEM_HOST);
    }
    state->local_to_global[pid] = par->outbox[pid].rmt_nbrs;
  }
}

// Allocate the GPU memory - bitmaps and frontier.
PRIVATE inline void graph500_alloc_gpu(partition_t* par) {
  graph500_state_t* state =
      reinterpret_cast<graph500_state_t*>(par->algo_state);

  // Allocate memory for the trees.
  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(vid_t),
                         TOTEM_MEM_DEVICE, (void**)&(state->tree[par->id])));

  // Initialize our visited bitmap.
  state->visited[par->id] = bitmap_init_gpu(par->subgraph.vertex_count);

  // Initialize other partitions frontier bitmaps.
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    vid_t count = par->outbox[pid].count;
    // Assign the outboxes to our frontier bitmap pointers.
    if (pid != par->id && count != 0) {
      // Allocate the visited bitmaps for other partitions.
      state->visited[pid] = bitmap_init_gpu(count);
      // Allocate the tree.
      bitmap_t rmt_bitmap = (bitmap_t)par->outbox[pid].push_values;
      state->tree[pid] = (vid_t*)(&rmt_bitmap[bitmap_bits_to_words(count)]);
    }
  }

  // Allocate our local frontier.
  frontier_init_gpu(&state->frontier_state, par->subgraph.vertex_count);

  // Allocate the host pinned buffer that will received the final output
  // from the GPU.
  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(vid_t),
                         TOTEM_MEM_HOST_PINNED,
                         reinterpret_cast<void**>(&state_g.tree_h[par->id])));

}

// Allocate the CPU memory - bitmaps and frontier.
PRIVATE inline void graph500_alloc_cpu(partition_t* par) {
  graph500_state_t* state =
      reinterpret_cast<graph500_state_t*>(par->algo_state);

  // Allocate memory for the trees.
  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(vid_t),
                         TOTEM_MEM_HOST, (void**)&(state->tree[par->id])));

  // Initialize our visited bitmap.
  state->visited[par->id] = bitmap_init_cpu(par->subgraph.vertex_count);

  // Initialize other partitions bitmaps.
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    vid_t count = par->outbox[pid].count;
    // Assign the outboxes to our frontier bitmap pointers.
    if (pid != par->id && count != 0) {
      // Allocate the visited bitmaps for other partitions.
      state->visited[pid] = bitmap_init_cpu(count);
      // Allocate the tree.
      bitmap_t rmt_bitmap = (bitmap_t)par->outbox[pid].push_values;
      state->tree[pid] = (vid_t*)(&rmt_bitmap[bitmap_bits_to_words(count)]);
    }
  }

  // Initialize our local frontier.
  frontier_init_cpu(&state->frontier_state, par->subgraph.vertex_count);
  state_g.tree_h[par->id] = state->tree[par->id];
}

void graph500_stepwise_alloc(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  // Initialize based off of our processor type.
  par->algo_state = calloc(1, sizeof(graph500_state_t));
  assert(par->algo_state);
  if (par->processor.type == PROCESSOR_CPU) {
    graph500_alloc_cpu(par);
  } else {
    graph500_alloc_gpu(par);
  }
  graph500_alloc_prepare_maps(par);
}

// The finalize phase - clean up.
void graph500_stepwise_free(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  graph500_state_t* state =
      reinterpret_cast<graph500_state_t*>(par->algo_state);
  totem_mem_t type = TOTEM_MEM_HOST;

  // Finalize frontiers.
  if (par->processor.type == PROCESSOR_CPU) {
    bitmap_finalize_cpu(state->visited[par->id]);
    frontier_finalize_cpu(&state->frontier_state);
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      if ((par->id == pid) || (par->outbox[pid].count == 0)) { continue; }
      bitmap_finalize_cpu(state->visited[pid]);
    }
  } else if (par->processor.type == PROCESSOR_GPU) {
    bitmap_finalize_gpu(state->visited[par->id]);
    type = TOTEM_MEM_DEVICE;
    frontier_finalize_gpu(&state->frontier_state);
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      if ((par->id == pid) || (par->outbox[pid].count == 0)) { continue; }
      bitmap_finalize_gpu(state->visited[pid]);
    }
    totem_free(state_g.tree_h[par->id], TOTEM_MEM_HOST_PINNED);
  } else {
    assert(false);
  }

  totem_free(state->tree[par->id], type);
  free(state);
  par->algo_state = NULL;
}

PRIVATE inline void graph500_rmt_tree_scatter_cpu(grooves_box_table_t* inbox,
                                               vid_t* tree, int rmt_pid) {
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
graph500_rmt_tree_scatter_kernel(const vid_t* __restrict rmt_nbrs, vid_t count,
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

PRIVATE void graph500_rmt_tree_scatter(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  graph500_state_t* state = (graph500_state_t*)par->algo_state;
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_CPU) {
      graph500_rmt_tree_scatter_cpu(inbox, state->tree[par->id], rmt_pid);
    } else if (par->processor.type == PROCESSOR_GPU) {
      const int threads = DEFAULT_THREADS_PER_BLOCK;
      dim3 blocks;
      kernel_configure(inbox->count, blocks, threads);
      bitmap_t rmt_bitmap = (bitmap_t)inbox->push_values;
      vid_t* rmt_tree =
        (vid_t*)(&rmt_bitmap[bitmap_bits_to_words(inbox->count)]);
      graph500_rmt_tree_scatter_kernel<<<blocks, threads, 0, par->streams[1]>>>
        (inbox->rmt_nbrs, inbox->count, rmt_tree, state->tree[par->id],
         rmt_pid);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(false);
    }
  }
}

void graph500_rmt_tree(partition_t* par) {
  if (engine_superstep() == 1) {
    engine_report_not_finished();
  } else {
    engine_report_no_comm(par->id);
  }
}

PRIVATE void graph500_final_aggregation() {
  vid_t* map = engine_vertex_id_in_partition();
  OMP(omp parallel for schedule(static))
  for (vid_t v = 0; v < engine_vertex_count(); v++) {
    vid_t local = map[v];
    vid_t* tree = state_g.tree_h[GET_PARTITION_ID(local)];
    vid_t parent = tree[GET_VERTEX_ID(local)];
    if (parent != VERTEX_ID_MAX) {
      state_g.tree[v] = parent;
    } else {
      state_g.tree[v] = (bfs_tree_t)-1;
    }
  }
}

// The launch point for the algorithm - set up engine, tree, and launch.
error_t graph500_stepwise_hybrid(vid_t src, bfs_tree_t* tree) {
  // Check for special cases.
  bool finished = false;
  error_t rc = check_special_cases(src, tree, &finished);
  if (finished) { return rc; }

  // Initialize the global state.
  state_g.tree = tree;
  state_g.src  = engine_vertex_id_in_partition(src);

  // Initialize the engines - one for the first top down step, and a second
  // to complete the algorithm with bottom up steps.
  // TODO(scott): Modify the swapping to flip back and forth simpler.

  // During the main execution cycles, only one bit of communication per remote
  // neighbour is needed.
  engine_update_msg_size(GROOVES_PUSH, 1);

  // Begin by executing with top down steps.
  state_g.switch_parameter = 0;
  state_g.bu_step = false;
  engine_config_t config_td = {
    NULL, graph500, graph500_scatter, NULL, graph500_init,
    NULL, NULL, GROOVES_PUSH
  };
  engine_config(&config_td);
  engine_execute();

  // Continue execution with bottom up steps.
  state_g.bu_step = true;
  engine_config_t config_bu = {
    NULL, graph500, NULL, graph500_gather, NULL,
    NULL, NULL, GROOVES_PULL
  };
  engine_config(&config_bu);
  engine_execute();

  // Finalize execution with top down steps.
  state_g.switch_parameter = 0;
  state_g.bu_step = false;
  engine_config_t config_td2 = {
    NULL, graph500, graph500_scatter, NULL, NULL,
    NULL, NULL, GROOVES_PUSH
  };
  engine_config(&config_td2);
  engine_execute();

  // Do a final run for remote tree scatter.
  engine_config_t config_update_rmt_parents = {
    NULL, graph500_rmt_tree, graph500_rmt_tree_scatter, NULL, NULL,
    NULL, graph500_aggregate, GROOVES_PUSH
  };
  engine_config(&config_update_rmt_parents);
  engine_reset_msg_size(GROOVES_PUSH);
  engine_execute();

  // Aggregate the result from the local tree arrays to the global one.
  graph500_final_aggregation();

  return SUCCESS;
}
