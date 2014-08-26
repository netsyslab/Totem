/**
 * This file contains an implementation of the breadth-first search (BFS) graph
 * search algorithm using the totem framework. This is a modified version that
 * performs the algorithm in a Bottom Up fashion.
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
  bitmap_t  visited[MAX_PARTITION_COUNT];  // A list of bitmaps, one for each
                                           // remote partition.
  bool*     finished;          // Points to Totem's finish flag.
  cost_t    level;             // Current level to process by the partition.
  frontier_state_t frontier;   // Frontier management state.
} bfs_state_t;

// State shared between all partitions.
typedef struct bfs_global_state_s {
  cost_t*   cost;    // Final output buffer.
  cost_t*   cost_h;  // Used as a temporary buffer to receive the final
                     // result copied back from GPU partitions before being
                     // copied again to the final output buffer.
                     // TODO(abdullah): push this buffer to be managed by Totem
  vid_t     src;     // Source vertex id. (The id after partitioning.)
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

/* Redacted td step.
// A classic top down step that iterates over vertices in the frontier
// and tries to add their neighbours to the next frontier.
PRIVATE void bfs_td_step(graph_t* graph, cost_t* cost, bitmap_t* visited,
                   frontier_state_t* state, cost_t level) {
  //bool finished = true;

  // Iterate across all vertices in frontier.
  OMP(omp parallel for schedule(runtime)) // reduction(& : finished))
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
          //finished = false;
          // Increment the level of this neighbour.
          cost[neighbor_id] = level + 1;
        }
      }
    }
  }  // End of omp for
  //return finished;
}
*/

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

      // Check if the bitmap corresponding to the vertices PID is set.
      // This means the partition that the vertex belongs to, has explored it.
      if (bitmap_is_set(state->visited[nbr_pid], subgraph->edges[i])) {
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

/* Similar to the regular BFS for cpu, the difference being choosing
 * a step for each level is now possible.
 * This implementation only works for undirected graphs.
 * Based off of the work by Scott Beamer et al.
 * Searching for a Parent Instead of Fighting Over Children: A Fast 
 * Breadth-First Search Implementation for Graph500.
 * http://www.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-117.pdf
 */
void bfs_stepwise_cpu(partition_t* par, bfs_state_t* state) {
  // Copy the current state of the remote vertices' bitmaps.
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
    bitmap_copy_cpu(state->visited[pid], (bitmap_t)par->outbox[pid].pull_values,
                    par->outbox[pid].count);
  }

  // Update the frontier.
  frontier_update_bitmap_cpu(&state->frontier, state->visited[par->id]);

  // Execute a step.
  bfs_bu_step(par, state);

  // Apply our visited status to our outboxes.
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
    bitmap_copy_cpu(state->visited[par->id],
                    (bitmap_t)par->outbox[pid].pull_values,
                    par->outbox[pid].count);
  }
}

/* Redacted td kernel.
// A gpu version of the Top-down step as a kernel.
__global__
void bfs_td_kernel(graph_t graph, cost_t level, cost_t* cost,
                   bitmap_t visited, frontier_state_t state) {
  const vid_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) { return; }

  // Ignore vertices not in frontier.
  if (!bitmap_is_set(state.current, vertex_id)) { return; }

  // Iterate across all neighbours of the vertex.
  for (eid_t i = graph.vertices[vertex_id];
       i < graph.vertices[vertex_id + 1]; i++) {
    const vid_t neighbor_id = graph.edges[i];

    // If already visited, ignore neighbour.
    if (!bitmap_is_set(visited, neighbor_id)) {
      if (bitmap_set_gpu(visited, neighbor_id)) {
        // Increment the level of this neighbour.
        cost[neighbor_id] = level + 1;
      }
    }
  }
}
*/

// A gpu version of the Bottom-up step as a kernel.
__global__
void bfs_bu_kernel(partition_t par, bfs_state_t state) {
  const vid_t vertex_id = THREAD_GLOBAL_INDEX;
  graph_t subgraph = par.subgraph;
  if (vertex_id >= subgraph.vertex_count) { return; }
  bool finished = true;

  // Ignore the local vertex if it has already been visited.
  if (bitmap_is_set(state.visited[par.id], vertex_id) ) { return; }

  // Iterate across all neighbours of the vertex.
  for (eid_t i = subgraph.vertices[vertex_id];
       i < subgraph.vertices[vertex_id + 1]; i++) {
    int nbr_pid = GET_PARTITION_ID(subgraph.edges[i]);

    // Check if neighbour is in the current frontier.
    if (bitmap_is_set(state.visited[nbr_pid], subgraph.edges[i])) {
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

// This is a GPU only version of the above Bottom-up/Top-down BFS algorithm.
// See bfs_bu_cpu for full details.
__host__
error_t bfs_stepwise_gpu(partition_t* par, bfs_state_t* state) {
  // Copy the current state of the remote vertices' bitmaps.
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
    bitmap_copy_gpu(state->visited[pid], (bitmap_t)par->outbox[pid].pull_values,
                    par->outbox[pid].count);
  }

  // {} used to limit scope and avoid problems with error handles.
  {
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(par->subgraph.vertex_count, blocks, threads_per_block);

  // Update the frontier.
  frontier_update_bitmap_gpu(&state->frontier, state->visited[par->id],
                             par->streams[1]);

  bfs_bu_kernel<<<blocks, threads_per_block>>>(*par, *state);
  }

  // Apply our visited status to our outboxes.
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if ((pid == par->id) || (par->outbox[pid].count == 0)) continue;
    bitmap_copy_gpu(state->visited[par->id],
                    (bitmap_t)par->outbox[pid].pull_values,
                    par->outbox[pid].count);
  }
  return SUCCESS;
}

// The execution phase - based off of the partition we are, launch an approach.
PRIVATE void bfs(partition_t* par) {
  if (par->subgraph.vertex_count == 0) return;
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

// The gather phase - apply values from the inboxes to the partitions' local
// variables.
PRIVATE void bfs_gather(partition_t* par) {
  bfs_state_t* state = reinterpret_cast<bfs_state_t*>(par->algo_state);

  // Across all partitions that are not us.
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;

    // Select the inbox to apply.
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];

    // Launch a bitmap copy based off of the processing unit we are.
    if (par->processor.type == PROCESSOR_CPU) {
      bitmap_copy_cpu((bitmap_t)inbox->pull_values, state->visited[par->id],
                      inbox->count);
    } else if (par->processor.type == PROCESSOR_GPU) {
      bitmap_copy_gpu((bitmap_t)inbox->pull_values, state->visited[par->id],
                      inbox->count);
    } else {
      assert(false);
    }
  }
}

// The aggregate phase - combine results to be presented.
PRIVATE void bfs_aggregate(partition_t* par) {
  if (!par->subgraph.vertex_count) return;

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
  if (THREAD_GLOBAL_INDEX != 0) return;
  bitmap_set_gpu(visited, src);
}

// Initialize the GPU memory - bitmaps and frontier.
PRIVATE inline void bfs_init_gpu(partition_t* par) {
  bfs_state_t* state = reinterpret_cast<bfs_state_t*>(par->algo_state);

  // Initialize our visited bitmap.
  state->visited[par->id] = bitmap_init_gpu(par->subgraph.vertex_count);

  // Initialize other partitions bitmaps - based off of their size.
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      state->visited[pid] = bitmap_init_gpu(par->outbox[pid].count);
      // Clear our outbox destined for each partition.
      bitmap_reset_gpu((bitmap_t)par->outbox[pid].pull_values,
                       par->outbox[pid].count, par->streams[1]);
    }
  }

  /*
  // Set the source vertex as visited, if it is in our partition.
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    bfs_init_bu_kernel<<<1, 1, 0, par->streams[1]>>>
      (state->visited[par->id], GET_VERTEX_ID(state_g.src));
    CALL_CU_SAFE(cudaGetLastError());
  }
  */

  // Set the source vertex as visited, no matter what partition it is in.
  bfs_init_bu_kernel<<<1, 1, 0, par->streams[1]>>>
    (state->visited[GET_PARTITION_ID(state_g.src)], GET_VERTEX_ID(state_g.src));
  CALL_CU_SAFE(cudaGetLastError());

  // Initialize our local frontier.
  frontier_init_gpu(&state->frontier, par->subgraph.vertex_count);
}

// Initialize the CPU memory - bitmaps and frontier.
PRIVATE inline void bfs_init_cpu(partition_t* par) {
  bfs_state_t* state = reinterpret_cast<bfs_state_t*>(par->algo_state);

  // Initialize our visited bitmap.
  state->visited[par->id] = bitmap_init_cpu(par->subgraph.vertex_count);

  // Initialize other partitions bitmaps - based off of their size.
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      state->visited[pid] = bitmap_init_cpu(par->outbox[pid].count);
      // Clear our outbox destined for each partition.
      bitmap_reset_cpu((bitmap_t)par->outbox[pid].pull_values,
                       par->outbox[pid].count);
    }
  }

  /*
  // Set the source vertex as visited, if it is in our partition.
  if (GET_PARTITION_ID(state_g.src) == par->id) {
    bitmap_set_cpu(state->visited[par->id], GET_VERTEX_ID(state_g.src));
  }
  */

  // Set the source vertex as visited, no matter what partition it is in.
  bitmap_set_cpu(state->visited[GET_PARTITION_ID(state_g.src)], 
                 GET_VERTEX_ID(state_g.src));

  // Initialize our local frontier.
  frontier_init_cpu(&state->frontier, par->subgraph.vertex_count);
}

// The init phase - Set up the memory and statuses.
PRIVATE void bfs_init(partition_t* par) {
  if (par->subgraph.vertex_count == 0) return;
  bfs_state_t* state =
    reinterpret_cast<bfs_state_t*>(calloc(1, sizeof(bfs_state_t)));
  assert(state);

  par->algo_state = state;
  totem_mem_t type = TOTEM_MEM_HOST;

  // Initialize based off of our processor type.
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
  if (par->subgraph.vertex_count == 0) return;
  bfs_state_t* state = reinterpret_cast<bfs_state_t*>(par->algo_state);
  totem_mem_t type = TOTEM_MEM_HOST;

  // Finalize frontiers.
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

  // Free memory.
  totem_free(state->cost, type);
  free(state);
  par->algo_state = NULL;
}

// The launch point for the algorithm - set up engine, cost, and launch.
error_t bfs_bu_hybrid(vid_t src, cost_t* cost) {
  // check for special cases
  bool finished = false;
  error_t rc = check_special_cases(src, cost, &finished);
  if (finished) return rc;

  // initialize the global state
  state_g.cost = cost;
  state_g.src  = engine_vertex_id_in_partition(src);

  // initialize the engine
  engine_config_t config = {
    NULL, bfs, NULL, bfs_gather, bfs_init, bfs_finalize, bfs_aggregate,
    GROOVES_PULL
  };
  engine_config(&config);
  if (engine_largest_gpu_partition()) {
    CALL_SAFE(totem_malloc(engine_largest_gpu_partition() * sizeof(cost_t),
                           TOTEM_MEM_HOST,
                           reinterpret_cast<void**>(&state_g.cost_h)));
  }
  engine_execute();

  // clean up and return
  if (engine_largest_gpu_partition()) {
    totem_free(state_g.cost_h, TOTEM_MEM_HOST);
  }
  memset(&state_g, 0, sizeof(bfs_global_state_t));
  return SUCCESS;
}
