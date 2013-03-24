/**
 * This file contains an implementation of the Betweenness Centrality
 * algorithm using the totem framework
 *
 *  Created on: 2013-03-10
 *  Author: Robert Woff
 */

// totem includes
#include "totem_centrality.h"
#include "totem_engine.cuh"
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

/**
 * per-partition specific state
 */
typedef struct betweenness_state_s {
  cost_t*   distance;    // distance from a source vertex
  uint32_t* numSPs;      // number of shortest paths for a vertex
  score_t*  delta;       // delta BC score for a vertex
  score_t*  betweenness; // betweenness score
  bool*     done;        // pointer to global finish flag
  cost_t    level;       // current level being processed by the partition
  dim3      blocks;      // kernel configuration parameters
  dim3      threads;
} betweenness_state_t;

/**
 * state shared between all partitions
 */
typedef struct betweenness_global_state_s {
  score_t*   betweenness_score;   // final output buffer
  score_t*   betweenness_score_h; // used as a temporary buffer
  vid_t     src;                  // source vertex id (id after partitioning)
} betweenness_global_state_t;
PRIVATE betweenness_global_state_t state_g = {NULL, NULL, 0};

/**
 * This structure is used by the virtual warp-based implementation. It stores a
 * batch of work. It is allocated on shared memory and is processed by a single
 * virtual warp. Basically it caches the state of the vertices to be processed.
 */
typedef struct {
  eid_t    vertices[VWARP_BATCH_SIZE + 1];
  cost_t   distance[VWARP_BATCH_SIZE];
  uint32_t numSPs[VWARP_BATCH_SIZE];
} batch_mem_t;

/**
 * The neighbors forward propagation processing function. This function sets 
 * the level of the neighbors' vertex to one level more than the parent vertex.
 * The assumption is that the threads of a warp invoke this function to process
 * the warp's batch of work. In each iteration of the for loop, each thread 
 * processes a neighbor. For example, thread 0 in the warp processes neighbors 
 * at indices 0, VWARP_WARP_SIZE, (2 * VWARP_WARP_SIZE) etc. in the edges array,
 * while thread 1 in the warp processes neighbors 1, (1 + VWARP_WARP_SIZE),
 * (1 + 2 * VWARP_WARP_SIZE) and so on.
*/
__device__
void forward_process_neighbors(partition_t par, vid_t warp_offset, vid_t* nbrs,
                               vid_t nbr_count, uint32_t my_numSPs,
                               uint32_t* numSPs_d, cost_t* distance_d, 
                               cost_t level, bool& done_d) {
  for(vid_t i = warp_offset; i < nbr_count; i += VWARP_WARP_SIZE) {
    vid_t nbr = nbrs[i];
    uint32_t* dst;
    
    // Check whether the neighbour is local or remote and update accordingly
    int nbr_pid = GET_PARTITION_ID(nbr);                             
    if (nbr_pid != par.id) {  
      // Need to place the updated numSPs value in the outbox  
      uint32_t* values = (uint32_t*)par.outbox[nbr_pid].push_values;         
      dst = &values[GET_VERTEX_ID((nbr))];
      // Done flag and distance will be updated when the scatter function
      // which corresponds to this remote vertex is called
      atomicAdd(dst, my_numSPs);      
    } else {
      // Can just handle the updates locally    
      if (distance_d[nbr] == INF_COST) {
        distance_d[nbr] = level + 1;
        done_d = false;
      }
      if (distance_d[nbr] == level + 1) {
        atomicAdd(&numSPs_d[nbr], my_numSPs);
      }
    }
  }
}

/**
 * Performs forward propagation
 */
__global__
void betweenness_gpu_forward_kernel(partition_t par, bool* done_d,
                                    cost_t level, uint32_t* numSPs_d, 
                                    cost_t* distance_d, uint32_t thread_count) {
  if (THREAD_GLOBAL_INDEX >= thread_count) return;
  vid_t warp_offset = THREAD_GLOBAL_INDEX % VWARP_WARP_SIZE;
  vid_t warp_id     = THREAD_GLOBAL_INDEX / VWARP_WARP_SIZE;
  
  // This flag is used to report the finish state of a block of threads. This
  // is useful to avoid having many threads writing to the global finished
  // flag, which can hurt performance (since "finished" is actually allocated
  // on the host, and each write will cause a transfer over the PCI-E bus)
  __shared__ bool finished_block;
  finished_block = true;
  __syncthreads();

  // copy my work to local space
  __shared__ batch_mem_t batch_s[(MAX_THREADS_PER_BLOCK / VWARP_WARP_SIZE)];
  batch_mem_t* vwarp_batch_s = &batch_s[THREAD_GRID_INDEX / VWARP_WARP_SIZE];
  vid_t base_v = warp_id * VWARP_BATCH_SIZE;
  vwarp_memcpy(vwarp_batch_s->vertices, &(par.subgraph.vertices[base_v]), 
               VWARP_BATCH_SIZE + 1, warp_offset);
  vwarp_memcpy(vwarp_batch_s->distance, &distance_d[base_v], VWARP_BATCH_SIZE, 
               warp_offset);
  vwarp_memcpy(vwarp_batch_s->numSPs, &numSPs_d[base_v], VWARP_BATCH_SIZE,
               warp_offset);

  // iterate over my work
  for(vid_t v = 0; v < VWARP_BATCH_SIZE; v++) {
    if (vwarp_batch_s->distance[v] == level) {
      vid_t* nbrs = &(par.subgraph.edges[vwarp_batch_s->vertices[v]]);
      vid_t nbr_count = vwarp_batch_s->vertices[v + 1] - 
        vwarp_batch_s->vertices[v];
      forward_process_neighbors(par, warp_offset, nbrs, nbr_count, 
                                vwarp_batch_s->numSPs[v], numSPs_d, 
                                distance_d, level, finished_block);
    }
  }
  __syncthreads();
  if (!finished_block && threadIdx.x == 0) *done_d = false;
}

/**
 * Entry point for GPU BC
*/
PRIVATE inline void betweenness_forward_gpu(partition_t* par) {
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  betweenness_gpu_forward_kernel<<<state->blocks, state->threads, 0, 
    par->streams[1]>>>(*par, state->done, state->level, state->numSPs,
                       state->distance, 
                       VWARP_BATCH_COUNT(par->subgraph.vertex_count) *
                       VWARP_WARP_SIZE);
  CALL_CU_SAFE(cudaGetLastError());
}

/**
 * Entry point for CPU BC
*/
void betweenness_forward_cpu(partition_t* par) {
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  graph_t* subgraph = &par->subgraph;
  
  bool done = true;
  // In parallel, iterate over vertices which are at the current level
  OMP(omp parallel for)
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    if (state->distance[v] == state->level) {
      // For all neighbors of v, iterate over paths
      for (eid_t e = subgraph->vertices[v]; e < subgraph->vertices[v + 1];
           e++) {
        vid_t nbr = subgraph->edges[e];
        
        // Check whether the neighbour is local or remote and update accordingly
        int nbr_pid = GET_PARTITION_ID(nbr);                             
        if (nbr_pid != par->id) {  
          // Need to place the updated numSPs value in the outbox  
          uint32_t* values = (uint32_t*)(par->outbox_d)[nbr_pid].push_values;         
          uint32_t* dst = &values[GET_VERTEX_ID((nbr))];
          // Done flag and distance will be updated when the scatter function
          // which corresponds to this remote vertex is called
          __sync_fetch_and_add(dst, state->numSPs[v]); 
        } else {
          // Can just handle the updates locally    
          if (state->distance[nbr] == INF_COST) {
            state->distance[nbr] = state->level + 1;
            done = false;
          }
          if (state->distance[nbr] == state->level + 1) {
            __sync_fetch_and_add(&(state->numSPs[nbr]), state->numSPs[v]);
          }
        }
      }
    }
  }
  if (!done) *(state->done) = false;
}

/**
 * Distributes work to either the CPU or GPU
*/
PRIVATE void betweenness_forward(partition_t* par) {
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  if (par->processor.type == PROCESSOR_CPU) {
    betweenness_forward_cpu(par);
  } else if (par->processor.type == PROCESSOR_GPU) {
    betweenness_forward_gpu(par);
  } else {
    assert(false);
  }
  // increment the level for the next round of forward propagation
  state->level++;
}

/**
 * Update the number of shortest paths from remote vertices
 * Also update distance if it has yet to be initialized
*/
PRIVATE void betweenness_scatter_forward(partition_t* par) {
   // TODO: make a separate CPU and GPU version
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    for (vid_t index = 0; index < inbox->count; index++) {
      vid_t vid = inbox->rmt_nbrs[index];
      uint32_t* inbox_values = (uint32_t*)inbox->push_values;
      if (inbox_values[index] != 0) {
        if (state->distance[vid] == INF_COST) {
          state->distance[vid] = state->level + 1;
          state->done = false;
        }
        if (state->distance[vid] == state->level + 1) {
          state->numSPs[vid] += inbox_values[index];
        }
      }
    }
  }
}

/**
 * Initializes the state for a round of forward propagation
*/
PRIVATE void betweenness_init_forward(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  assert(state);
  id_t src_pid = GET_PARTITION_ID(engine_vertex_id_in_partition(state_g.src));
  id_t src_vid = GET_VERTEX_ID(engine_vertex_id_in_partition(state_g.src));
  vid_t vcount = par->subgraph.vertex_count;

  totem_mem_t type = TOTEM_MEM_HOST; 
  if (par->processor.type == PROCESSOR_GPU) { 
    type = TOTEM_MEM_DEVICE;
    KERNEL_CONFIGURE(VWARP_WARP_SIZE * VWARP_BATCH_COUNT(vcount),
                     state->blocks, state->threads);
  }

  // Initialize the distances to infinity and numSPs to 0
  totem_memset(state->distance, INF_COST, vcount, type, par->streams[1]);
  totem_memset(state->numSPs, (uint32_t)0, vcount, type, par->streams[1]);
  if (src_pid == par->id) {
    // For the source vertex, initialize distance and numSPs.
    totem_memset(&(state->distance[src_vid]), (cost_t)0, 1, type,
                 par->streams[1]);
    totem_memset(&(state->numSPs[src_vid]), (uint32_t)1, 1, type,
                 par->streams[1]);
  } 
  
  // Initialize the outbox to 0 and set the level to 0
  engine_set_outbox(par->id, 0); 
  state->level = 0;
}

/**
 * Allocates and initializes the state for Betweenness Centrality
*/
PRIVATE void betweenness_init(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  betweenness_state_t* state = (betweenness_state_t*)
                               calloc(1, sizeof(betweenness_state_t));
  assert(state); 
  par->algo_state = state;
  vid_t vcount = par->subgraph.vertex_count;

  totem_mem_t type = TOTEM_MEM_HOST; 
  if (par->processor.type == PROCESSOR_GPU) { 
    type = TOTEM_MEM_DEVICE;
  }
  totem_calloc(vcount * sizeof(cost_t), type, (void**)(state->distance));
  totem_calloc(vcount * sizeof(uint32_t), type, (void**)(state->numSPs));
  totem_calloc(vcount * sizeof(score_t), type, (void**)(state->delta));
  totem_calloc(vcount * sizeof(score_t), type, (void**)(state->betweenness));

  // Initialize the state's done flag
  state->done = engine_get_finished_ptr(par->id);

  // Initialize the state
  betweenness_init_forward(par); 
}

/**
 * Cleans up allocated memory on the CPU and GPU
*/
PRIVATE void betweenness_finalize(partition_t* par) {
  // Free the allocated memory
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
 
  totem_mem_t type = TOTEM_MEM_HOST; 
  if (par->processor.type == PROCESSOR_GPU) { 
    type = TOTEM_MEM_DEVICE; 
  }
  totem_free(state->distance, type);
  totem_free(state->numSPs, type);
  totem_free(state->delta, type);
  totem_free(state->betweenness, type);

  free(state);
  par->algo_state = NULL;
}

/**
 * Aggregates the final result to be returned at the end
*/
PRIVATE void betweenness_aggr(partition_t* partition) {
  // Not implemented yet as only have forward propagation currently

  // Scale the computed Betweenness Centrality metrics since they were
  // computed using a subset of the total nodes within the graph
  // The scaling value is: (Total Number of Nodes / Subset of Nodes Used)
  /* TODO: implement CPU and GPU version when backward propagation done
  OMP(omp parallel for) 
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    betweenness_score[v] = (score_t)(((double)(graph->vertex_count)
                           / num_samples)* betweenness_score[v]);
  }*/
}

/**
 * Core functionality for main for loop within the BC computation
*/
void betweenness_hybrid_core(vid_t source, bool is_first_iteration,
                             bool is_last_iteration) {
  // Set the source node for this iteration
  state_g.src  = engine_vertex_id_in_partition(source);

  // Forward propagation
  engine_par_init_func_t init_forward = betweenness_init_forward;
  if (is_first_iteration) {
    init_forward = betweenness_init;
  }
  engine_config_t config_forward = {
    NULL, betweenness_forward, betweenness_scatter_forward, NULL, 
    init_forward, NULL, NULL, GROOVES_PUSH
  };
  engine_config(&config_forward);
  engine_execute();

  // Backward propagation
  // TODO: add proper functions for backward propagation
  engine_par_finalize_func_t finalize_backward = NULL;
  engine_par_aggr_func_t aggr_backward = NULL;
  if (is_last_iteration) {
    finalize_backward = betweenness_finalize;
    aggr_backward = betweenness_aggr;
  }
  engine_config_t config_backward = {
    NULL, NULL /*betweenness_backward*/, NULL,
    NULL /*betweenness_gather_backward*/,
    NULL /*betweenness_backward_init*/,
    finalize_backward, aggr_backward, GROOVES_PULL
  };
  engine_config(&config_backward);
  engine_execute();
}

/**
 * Main function for hybrid betweenness centrality
*/
error_t betweenness_hybrid(double epsilon, score_t* betweenness_score) {                           
  // Sanity check on input
  bool finished = false;
  error_t rc = betweenness_check_special_cases(engine_vertex_count(), 
                                               engine_edge_count(),
                                               &finished, betweenness_score);
  if (finished) return rc;

  // initialize the global state
  state_g.betweenness_score = betweenness_score;
  totem_memset(state_g.betweenness_score, (score_t)0, engine_vertex_count(),
               TOTEM_MEM_HOST);

  if (engine_largest_gpu_partition()) {
      state_g.betweenness_score_h = (score_t*)mem_alloc(
                                    engine_largest_gpu_partition() *
                                    sizeof(score_t));
  }

  // determine whether we will compute exact or approximate BC values
  if (epsilon == CENTRALITY_EXACT) {
    // Compute exact values for Betweenness Centrality
    vid_t vcount = engine_vertex_count();
    for (vid_t source = 0; source < vcount; source++) { 
      betweenness_hybrid_core(source, (source == 0), (source == (vcount-1)));  
    }
  } else {
    // Compute approximate values based on the value of epsilon provided
    // Select a subset of source nodes to make the computation faster
    int num_samples = centrality_get_number_sample_nodes(engine_vertex_count(),
                                                         epsilon);
    // Populate the array of indices to sample
    vid_t* sample_nodes = centrality_select_sampling_nodes(
                          engine_vertex_count(), num_samples);
 
    for (int source_index = 0; source_index < num_samples; source_index++) {
      // Get the next sample node in the array to use as a source
      vid_t source = sample_nodes[source_index];    
      betweenness_hybrid_core(source, (source_index == 0), 
                              (source_index == (num_samples-1)));  
    } 
 
    // Clean up the allocated memory
    free(sample_nodes);
  }
  
  // clean up and return
  if (engine_largest_gpu_partition()) mem_free(state_g.betweenness_score_h);
  memset(&state_g, 0, sizeof(betweenness_global_state_t));
  return SUCCESS;
}
