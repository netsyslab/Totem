/**
 * This file contains an implementation of the breadth-first search (BFS) graph
 * search algorithm using the totem framework. This is a modified version that
 * performs the algorithm in a Bottom Up fashion.
 *
 * This implementation only works for undirected graphs.
 *
 * Based off of the work by Scott Beamer et al.
 * Searching for a Parent Instead of Fighting Over Children: A Fast
 * Breadth-First Search Implementation for Graph500.
 * http://www.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-117.pdf
 *
 * TODO(scott): Modify the algorithm to swap between top down and bottom up
 *              steps.
 *
 *  Created on: 2014-08-26
 *  Authors:    Scott Sallinen
 *              Abdullah Gharaibeh
 */

#include "totem_alg.h"
#include "totem_engine.cuh"

// Per-partition specific state.
typedef struct bfs_state_s {
  cost_t*   cost;              // One slot per vertex in the partition.
  bitmap_t  visited[MAX_PARTITION_COUNT];   // A list of bitmaps, one for each
                                            // remote partition.
  bitmap_t  frontier[MAX_PARTITION_COUNT];  // A list of bitmaps, one for each
                                            // remote partition.
  bool*     finished;          // Points to Totem's finish flag.
  cost_t    level;             // Current level to process by the partition.
  frontier_state_t frontier_state;   // Frontier management state.
} bfs_state_t;

// State shared between all partitions.
typedef struct bfs_global_state_s {
  cost_t*   cost;     // Final output buffer.
  cost_t*   cost_h;   // Used as a temporary buffer to receive the final
                      // result copied back from GPU partitions before being
                      // copied again to the final output buffer.
                      // TODO(abdullah): push this buffer to be managed by Totem
  vid_t     src;      // Source vertex id. (The id after partitioning.)
  bool      bu_step;  // Whether or not to perform a bottom up step.
} bfs_global_state_t;
PRIVATE bfs_global_state_t state_g = {NULL, NULL, 0};

// Checks for input parameters and special cases. This is invoked at the
// beginning of public interfaces (GPU and CPU)
PRIVATE error_t check_special_cases(vid_t src, cost_t* cost, bool* finished) {
  *finished = true;
  if ((src >= engine_vertex_count()) || (cost == NULL)) {
    return FAILURE;
  } else if (engine_vertex_count() == 1) {
    cost[0] = 0;
    return SUCCESS;
  } else if (engine_edge_count() == 0) {
    // Initialize cost to INFINITE.
    totem_memset(cost, INF_COST, engine_vertex_count(), TOTEM_MEM_HOST);
    cost[src] = 0;
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

// A step that iterates across the frontier of vertices and adds their
// neighbours to the next frontier.
PRIVATE void bfs_td_step(partition_t* par, bfs_state_t* state) {
  graph_t* subgraph = &par->subgraph;
  bool finished = true;

  // Iterate across all of our vertices.
  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t vertex_id = 0; vertex_id < subgraph->vertex_count; vertex_id++) {
    // Ignore the local vertex if it is not in the frontier.
    if (!bitmap_is_set(state->frontier[par->id], vertex_id)) { continue; }

    // Iterate across the neighbours of this vertex.
    for (eid_t i = subgraph->vertices[vertex_id];
         i < subgraph->vertices[vertex_id + 1]; i++) {
      int nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
      int nbr = GET_VERTEX_ID(subgraph->edges[i]);

      // Add the neighbour we are exploring to the next frontier.
      if (!bitmap_is_set(state->visited[nbr_pid], nbr)) {
        if (bitmap_set_cpu(state->visited[nbr_pid], nbr)) {
          // Increment the level of this vertex.
          if (nbr_pid == par->id) {
            state->cost[nbr] = state->level + 1;
          }
          finished = false;
        }
      }
    }  // End of neighbour check - vertex examined.
  }  // All vertices examined in level.

  // Move over the finished variable.
  if (!finished) *(state->finished) = false;
}

// A step that iterates across unvisited vertices and determines
// their status in the next frontier.
PRIVATE void bfs_bu_step(partition_t* par, bfs_state_t* state) {
  graph_t* subgraph = &par->subgraph;
  bool finished = true;

  // Iterate across all of our vertices.
  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t vertex_id = 0; vertex_id < subgraph->vertex_count; vertex_id++) {
    // Ignore the local vertex if it has already been visited.
    if (bitmap_is_set(state->visited[par->id], vertex_id)) { continue; }

    // Iterate across the neighbours of this vertex.
    for (eid_t i = subgraph->vertices[vertex_id];
         i < subgraph->vertices[vertex_id + 1]; i++) {
      int nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
      int nbr = GET_VERTEX_ID(subgraph->edges[i]);

      // Check if the bitmap corresponding to the vertices PID is set.
      // This means the partition that the vertex belongs to, has explored it.
      if (bitmap_is_set(state->frontier[nbr_pid], nbr)) {
        // Add the vertex we are exploring to the next frontier.
        bitmap_set_cpu(state->visited[par->id], vertex_id);

        // Increment the level of this vertex.
        state->cost[vertex_id] = state->level + 1;
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
void bfs_stepwise_cpu(partition_t* par, bfs_state_t* state) {
  if (state_g.bu_step) {
    // Update the frontier.
    frontier_update_bitmap_cpu(&state->frontier_state, state->visited[par->id]);
    state->frontier[par->id] = state->frontier_state.current;

    // Execute a bottom up step.
    bfs_bu_step(par, state);

  } else {
    // Copy the current state of the remote vertices bitmap.
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
      bitmap_copy_cpu(state->visited[pid],
                      (bitmap_t)par->outbox[pid].push_values,
                      par->outbox[pid].count);
    }

    // Update the frontier.
    frontier_update_bitmap_cpu(&state->frontier_state, state->visited[par->id]);
    state->frontier[par->id] = state->frontier_state.current;

    // Execute a top down step.
    bfs_td_step(par, state);

    // Diff the remote vertices bitmaps so that only the vertices who got set
    // in this round are notified.
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
      bitmap_diff_cpu(state->visited[pid],
                      (bitmap_t)par->outbox[pid].push_values,
                      par->outbox[pid].count);
    }
  }
}

// A gpu version of the Top-down step as a kernel.
__global__ void bfs_td_kernel(partition_t par, bfs_state_t state) {
  const vid_t vertex_id = THREAD_GLOBAL_INDEX;
  graph_t subgraph = par.subgraph;
  if (vertex_id >= subgraph.vertex_count) { return; }
  bool finished = true;

  // Ignore the local vertex if it is not in the frontier.
  if (!bitmap_is_set(state.frontier[par.id], vertex_id)) { return; }

  // Iterate across all neighbours of the vertex.
  for (eid_t i = subgraph.vertices[vertex_id];
       i < subgraph.vertices[vertex_id + 1]; i++) {
    int nbr_pid = GET_PARTITION_ID(subgraph.edges[i]);
    int nbr = GET_VERTEX_ID(subgraph.edges[i]);

    // Add the neighbour we are exploring to the next frontier.
    if (!bitmap_is_set(state.visited[nbr_pid], nbr)) {
      if (bitmap_set_gpu(state.visited[nbr_pid], nbr)) {
        // Increment the level of this vertex.
        if (nbr_pid == par.id) {
          state.cost[nbr] = state.level + 1;
        }
        finished = false;
      }
    }
  }

  // Move over the finished variable.
  if (!finished) *(state.finished) = false;
}

// A gpu version of the Bottom-up step as a kernel.
__global__ void bfs_bu_kernel(partition_t par, bfs_state_t state) {
  const vid_t vertex_id = THREAD_GLOBAL_INDEX;
  graph_t subgraph = par.subgraph;
  if (vertex_id >= subgraph.vertex_count) { return; }
  bool finished = true;

  // Ignore the local vertex if it has already been visited.
  if (bitmap_is_set(state.visited[par.id], vertex_id)) { return; }

  // Iterate across all neighbours of the vertex.
  for (eid_t i = subgraph.vertices[vertex_id];
       i < subgraph.vertices[vertex_id + 1]; i++) {
    int nbr_pid = GET_PARTITION_ID(subgraph.edges[i]);
    int nbr = GET_VERTEX_ID(subgraph.edges[i]);

    // Check if neighbour is in the current frontier.
    if (bitmap_is_set(state.frontier[nbr_pid], nbr)) {
      // Add the vertex we are exploring to the next frontier.
      bitmap_set_gpu(state.visited[par.id], vertex_id);

      // Increment the level of this vertex.
      state.cost[vertex_id] = state.level + 1;
      finished = false;
      break;
    }
  }

  // Move over the finished variable.
  if (!finished) *(state.finished) = false;
}

// This is a GPU version of the Bottom-up/Top-down BFS algorithm.
// See file header for full details.
__host__ error_t bfs_stepwise_gpu(partition_t* par, bfs_state_t* state) {
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(par->subgraph.vertex_count, blocks, threads_per_block);

  if (state_g.bu_step) {
    // Update the frontier.
    frontier_update_bitmap_gpu(&state->frontier_state, state->visited[par->id],
                               par->streams[1]);
    state->frontier[par->id] = state->frontier_state.current;

    // Execute a bottom up step.
    bfs_bu_kernel<<<blocks, threads_per_block, 0, par->streams[1]>>>
                 (*par, *state);

  } else {
    // Copy the current state of the remote vertices bitmap.
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
      bitmap_copy_gpu(state->visited[pid],
                      (bitmap_t)par->outbox[pid].push_values,
                      par->outbox[pid].count,
                      par->streams[1]);
    }

    // Update the frontier.
    frontier_update_bitmap_gpu(&state->frontier_state, state->visited[par->id],
                               par->streams[1]);
    state->frontier[par->id] = state->frontier_state.current;

    // Execute a top down step.
    bfs_td_kernel<<<blocks, threads_per_block, 0, par->streams[1]>>>
                 (*par, *state);

    // Diff the remote vertices bitmaps so that only the vertices who got set
    // in this round are notified.
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
      bitmap_diff_gpu(state->visited[pid],
                      (bitmap_t)par->outbox[pid].push_values,
                      par->outbox[pid].count,
                      par->streams[1]);
    }
  }

  return SUCCESS;
}

// The execution phase - based off of the partition we are, launch an approach.
PRIVATE void bfs(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }

  // Ignore the first round - this allows us to communicate the frontier with
  // an updated visited status of the source vertex.
  if (engine_superstep() == 1) {
    engine_report_not_finished();
    return;
  }

  // TODO(scott): Make this not hardcoded - this swaps statically on step 3.
  if (engine_superstep() == 4 && state_g.bu_step == false) {
    return;
  }

  bfs_state_t* state = reinterpret_cast<bfs_state_t*>(par->algo_state);

  // Launch the processor specific algorithm.
  if (par->processor.type == PROCESSOR_CPU) {
    bfs_stepwise_cpu(par, state);
  } else if (par->processor.type == PROCESSOR_GPU) {
    bfs_stepwise_gpu(par, state);
  } else {
    assert(false);
  }

  // At the end of the round, increase our BFS level.
  state->level++;
}

// Gather for the CPU bitmap to inbox.
PRIVATE void bfs_gather_cpu(partition_t* par, bfs_state_t* state,
                            grooves_box_table_t* inbox) {
  // Iterate across the items in the inbox.
  OMP(omp parallel for schedule(static))
  for (vid_t index = 0; index < inbox->count; index++) {
    // Lookup the local vertex it points to.
    vid_t vid = inbox->rmt_nbrs[index];

    // Set the bit state to our local state.
    if (bitmap_is_set(state->visited[par->id], vid)) {
      bitmap_set_cpu(reinterpret_cast<bitmap_t>(inbox->pull_values), index);
    }
  }
}

// Gather for the GPU bitmap to inbox.
__global__ void bfs_gather_gpu(partition_t par, bfs_state_t state,
                               grooves_box_table_t inbox) {
  const vid_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) { return; }

  // Lookup the local vertex it points to.
  vid_t vid = inbox.rmt_nbrs[index];

  // Set the bit state to our local state.
  if (bitmap_is_set(state.visited[par.id], vid)) {
    bitmap_set_gpu(reinterpret_cast<bitmap_t>(inbox.pull_values), index);
  }
}

// The gather phase - apply values from the inboxes to the partitions' local
// variables.
PRIVATE void bfs_gather(partition_t* par) {
  bfs_state_t* state = reinterpret_cast<bfs_state_t*>(par->algo_state);

  // Across all partitions that are not us.
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) { continue; }

    // Select the inbox to apply to.
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (inbox->count == 0) { continue; }

    // Select a method based off of our processor type.
    if (par->processor.type == PROCESSOR_CPU) {
      bfs_gather_cpu(par, state, inbox);
    } else if (par->processor.type == PROCESSOR_GPU) {
      // Set up to iterate across the items in the inbox.
      dim3 blocks;
      dim3 threads_per_block;
      KERNEL_CONFIGURE(inbox->count, blocks, threads_per_block);
      bfs_gather_gpu<<<blocks, threads_per_block, 0, par->streams[1]>>>
        (*par, *state, *inbox);
    } else {
      assert(false);
    }
  }
}

// This is a scatter for CPU - copied from the original bfs_hybrid algorithm.
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

// This is a scatter for GPU - copied from the original bfs_hybrid algorithm.
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
  for (vid_t k = start_word; k < end_word; k++) {
    bitmap_word_t word = rmt_visited[k];
    if (word == 0) continue;
    vid_t start_vertex = k * BITMAP_BITS_PER_WORD;
    for (vid_t i = warp_offset; i < BITMAP_BITS_PER_WORD;
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

// The main scatter function, used in the top down phases.
PRIVATE void bfs_scatter(partition_t* par) {
  bfs_state_t* state = reinterpret_cast<bfs_state_t*>(par->algo_state);
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

// The aggregate phase - combine results to be presented.
PRIVATE void bfs_aggregate(partition_t* par) {
  if (!par->subgraph.vertex_count) { return; }

  bfs_state_t* state    = reinterpret_cast<bfs_state_t*>(par->algo_state);
  graph_t*     subgraph = &par->subgraph;
  cost_t*      src_cost = NULL;

  // Apply the cost from our partition into the final cost array.
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

  // Aggregate the results.
  assert(state_g.cost);
  OMP(omp parallel for schedule(runtime))
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    state_g.cost[par->map[v]] = src_cost[v];
  }
}

// A simple kernel that sets the source vertex to visited on the GPU.
__global__ void bfs_init_bu_kernel(bitmap_t visited, vid_t src) {
  if (THREAD_GLOBAL_INDEX != 0) { return; }
  bitmap_set_gpu(visited, src);
}

// Initialize the GPU memory - bitmaps and frontier.
PRIVATE inline void bfs_init_gpu(partition_t* par) {
  bfs_state_t* state = reinterpret_cast<bfs_state_t*>(par->algo_state);

  // Initialize our visited bitmap.
  state->visited[par->id] = bitmap_init_gpu(par->subgraph.vertex_count);

  // Initialize other partitions frontier bitmaps.
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    // Assign the outboxes to our frontier bitmap pointers.
    if (pid != par->id && par->outbox[pid].count != 0) {
      state->frontier[pid] =
        reinterpret_cast<bitmap_t>(par->outbox[pid].pull_values);

      // Allocate the visited bitmaps for other partitions.
      state->visited[pid] = bitmap_init_gpu(par->outbox[pid].count);

      // Clear the outboxes (push values).
      bitmap_reset_gpu(reinterpret_cast<bitmap_t>
                         (par->outbox[pid].push_values),
                       par->outbox[pid].count,
                       par->streams[1]);
    }

    // Clear the inboxes (pull values), and also their shadows.
    if (pid != par->id && par->inbox[pid].count != 0) {
      bitmap_reset_gpu(reinterpret_cast<bitmap_t>
                         (par->inbox[pid].pull_values),
                       par->inbox[pid].count,
                       par->streams[1]);
      bitmap_reset_gpu(reinterpret_cast<bitmap_t>
                         (par->inbox[pid].pull_values_s),
                       par->inbox[pid].count,
                       par->streams[1]);
    }
  }

  // Set the source vertex as visited, if it is in our partition.
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    bfs_init_bu_kernel<<<1, 1, 0, par->streams[1]>>>
      (state->visited[par->id], GET_VERTEX_ID(state_g.src));
    CALL_CU_SAFE(cudaGetLastError());
  }

  // Initialize our local frontier.
  frontier_init_gpu(&state->frontier_state, par->subgraph.vertex_count);
}

// Initialize the CPU memory - bitmaps and frontier.
PRIVATE inline void bfs_init_cpu(partition_t* par) {
  bfs_state_t* state = reinterpret_cast<bfs_state_t*>(par->algo_state);

  // Initialize our visited bitmap.
  state->visited[par->id] = bitmap_init_cpu(par->subgraph.vertex_count);

  // Initialize other partitions bitmaps.
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    // Assign the outboxes to our frontier bitmap pointers.
    if (pid != par->id && par->outbox[pid].count != 0) {
      state->frontier[pid] =
        reinterpret_cast<bitmap_t>(par->outbox[pid].pull_values);

      // Allocate the visited bitmaps for other partitions.
      state->visited[pid] = bitmap_init_cpu(par->outbox[pid].count);

      // Clear the push values.
      bitmap_reset_cpu(reinterpret_cast<bitmap_t>
                         (par->outbox[pid].push_values),
                      par->outbox[pid].count);
    }

    // Clear the inboxes, and also their shadows.
    if (pid != par->id && par->inbox[pid].count != 0) {
      bitmap_reset_cpu(reinterpret_cast<bitmap_t>
                         (par->inbox[pid].pull_values),
                       par->inbox[pid].count);
      bitmap_reset_cpu(reinterpret_cast<bitmap_t>
                         (par->inbox[pid].pull_values_s),
                       par->inbox[pid].count);
    }
  }

  // Set the source vertex as visited, if it is in our partition.
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    bitmap_set_cpu(state->visited[par->id], GET_VERTEX_ID(state_g.src));
  }

  // Initialize our local frontier.
  frontier_init_cpu(&state->frontier_state, par->subgraph.vertex_count);
}

// The init phase - Set up the memory and statuses.
PRIVATE void bfs_init(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  bfs_state_t* state =
    reinterpret_cast<bfs_state_t*>(calloc(1, sizeof(bfs_state_t)));
  assert(state);

  // Initialize based off of our processor type.
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

  // Allocate memory for the cost array, and set it to INFINITE cost.
  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(cost_t), type,
                         reinterpret_cast<void**>(&(state->cost))));
  totem_memset(state->cost, INF_COST, par->subgraph.vertex_count, type,
               par->streams[1]);

  if (GET_PARTITION_ID(state_g.src) == par->id) {
    // For the source vertex, initialize cost.
    totem_memset(&((state->cost)[GET_VERTEX_ID(state_g.src)]), (cost_t)0, 1,
                 type, par->streams[1]);
  }

  // Set level 0 to start, and finished pointer.
  state->finished = engine_get_finished_ptr(par->id);
  state->level = 0;
}

// The finalize phase - clean up.
PRIVATE void bfs_finalize(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  bfs_state_t* state = reinterpret_cast<bfs_state_t*>(par->algo_state);
  totem_mem_t type = TOTEM_MEM_HOST;

  // Finalize frontiers.
  if (par->processor.type == PROCESSOR_CPU) {
    bitmap_finalize_cpu(state->visited[par->id]);
    frontier_finalize_cpu(&state->frontier_state);
  } else if (par->processor.type == PROCESSOR_GPU) {
    bitmap_finalize_gpu(state->visited[par->id]);
    type = TOTEM_MEM_DEVICE;
    frontier_finalize_gpu(&state->frontier_state);
  } else {
    assert(false);
  }

  // Free memory.
  totem_free(state->cost, type);
  free(state);
  par->algo_state = NULL;
}

// The launch point for the algorithm - set up engine, cost, and launch.
error_t bfs_stepwise_hybrid(vid_t src, cost_t* cost) {
  // Check for special cases.
  bool finished = false;
  error_t rc = check_special_cases(src, cost, &finished);
  if (finished) { return rc; }

  // Initialize the global state.
  state_g.cost = cost;
  state_g.src  = engine_vertex_id_in_partition(src);

  // Initialize the engines - one for the first top down step, and a second
  // to complete the algorithm with bottom up steps.
  // TODO(scott): Modify the swapping to flip back and forth simpler.
  engine_config_t config_td = {
    NULL, bfs, bfs_scatter, NULL, bfs_init, NULL, NULL,
    GROOVES_PUSH
  };

  engine_config_t config_bu = {
    NULL, bfs, NULL, bfs_gather, NULL, bfs_finalize, bfs_aggregate,
    GROOVES_PULL
  };

  engine_config(&config_td);
  if (engine_largest_gpu_partition()) {
    CALL_SAFE(totem_malloc(engine_largest_gpu_partition() * sizeof(cost_t),
                           TOTEM_MEM_HOST,
                           reinterpret_cast<void**>(&state_g.cost_h)));
  }

  // Begin by executing with top down steps.
  state_g.bu_step = false;
  engine_execute();

  // Continue execution with bottom up steps.
  state_g.bu_step = true;
  engine_config(&config_bu);
  engine_execute();

  // Clean up and return.
  if (engine_largest_gpu_partition()) {
    totem_free(state_g.cost_h, TOTEM_MEM_HOST);
  }
  memset(&state_g, 0, sizeof(bfs_global_state_t));
  return SUCCESS;
}
