/**
 * This file contains a hybrid implementation of the Betweenness Centrality
 * algorithm using the Totem framework
 *
 *  Created on: 2013-03-10
 *  Author: Robert Woff
 */

// Totem includes
#include "totem_alg.h"
#include "totem_centrality.h"
#include "totem_engine.cuh"

/**
 * Per-partition specific state
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
 * State shared between all partitions
 */
typedef struct betweenness_global_state_s {
  score_t*   betweenness_score;   // final output buffer
  score_t*   betweenness_score_h; // used as a temporary buffer
  vid_t      src;                 // source vertex id (id after partitioning)
  double     epsilon;             // determines accuracy of BC computation
  int        num_samples;         // number of samples for approximate BC
} betweenness_global_state_t;
PRIVATE betweenness_global_state_t bc_g = 
  {NULL, NULL, 0, CENTRALITY_EXACT, 0};

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
  // Iterate through the portion of work
  for(vid_t i = warp_offset; i < nbr_count; i += VWARP_WARP_SIZE) {
    vid_t nbr = nbrs[i];
    uint32_t* dst;
    
    // Check whether the neighbour is local or remote and update accordingly
    int nbr_pid = GET_PARTITION_ID(nbr);                             
    if (nbr_pid != par.id) {  
      // Need to place the updated numSPs value in the outbox to be sent
      // to the remote partition durin the communication phase 
      uint32_t* values = (uint32_t*)par.outbox_d[nbr_pid].push_values;         
      dst = &values[GET_VERTEX_ID((nbr))];
      // Distance will be updated when the scatter function which
      // corresponds to this remote vertex is called during the
      // communication phase
      done_d = false;
      atomicAdd(dst, my_numSPs);      
    } else {
      // Can just handle the updates locally
      nbr = GET_VERTEX_ID(nbr); 
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
  // Determine the warp parameters for this processing unit
  vid_t warp_offset = THREAD_GLOBAL_INDEX % VWARP_WARP_SIZE;
  vid_t warp_id     = THREAD_GLOBAL_INDEX / VWARP_WARP_SIZE;
  
  // This flag is used to report the finish state of a block of threads. This
  // is useful to avoid having many threads writing to the global finished
  // flag, which can hurt performance (since "finished" is actually allocated
  // on the host, and each write will cause a transfer over the PCI-E bus)
  __shared__ bool finished_block;
  finished_block = true;
  __syncthreads();

  // Copy my work to local space
  __shared__ batch_mem_t batch_s[(MAX_THREADS_PER_BLOCK / VWARP_WARP_SIZE)];
  batch_mem_t* vwarp_batch_s = &batch_s[THREAD_GRID_INDEX / VWARP_WARP_SIZE];
  vid_t base_v = warp_id * VWARP_BATCH_SIZE;
  vwarp_memcpy(vwarp_batch_s->vertices, &(par.subgraph.vertices[base_v]), 
               VWARP_BATCH_SIZE + 1, warp_offset);
  vwarp_memcpy(vwarp_batch_s->distance, &distance_d[base_v], VWARP_BATCH_SIZE, 
               warp_offset);
  vwarp_memcpy(vwarp_batch_s->numSPs, &numSPs_d[base_v], VWARP_BATCH_SIZE,
               warp_offset);

  // Iterate over my work
  for(vid_t v = 0; v < VWARP_BATCH_SIZE; v++) {
    if (vwarp_batch_s->distance[v] == level) {
      // If the distance for this node is equal to the current level, then
      // forward process its neighbours to determine its contribution to
      // the number of shortest paths
      vid_t* nbrs = &(par.subgraph.edges[vwarp_batch_s->vertices[v]]);
      vid_t nbr_count = vwarp_batch_s->vertices[v + 1] - 
        vwarp_batch_s->vertices[v];
      forward_process_neighbors(par, warp_offset, nbrs, nbr_count, 
                                vwarp_batch_s->numSPs[v], numSPs_d, 
                                distance_d, level, finished_block);
    }
  }
  // If there is remaining work to do, set the done flag to false
  __syncthreads();
  if (!finished_block && threadIdx.x == 0) *done_d = false;
}

/**
 * Entry point for forward propagation on the GPU
 */
PRIVATE inline void betweenness_forward_gpu(partition_t* par) {
  // Get the current state of the algorithm
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  // Call the corresponding cuda kernel to perform forward propagation
  // given the current state of the algorithm
  betweenness_gpu_forward_kernel<<<state->blocks, state->threads, 0, 
    par->streams[1]>>>(*par, state->done, state->level, state->numSPs,
                       state->distance, 
                       VWARP_BATCH_COUNT(par->subgraph.vertex_count) *
                       VWARP_WARP_SIZE);
  CALL_CU_SAFE(cudaGetLastError());
}

/**
 * Entry point for forward propagation on the CPU
 */
void betweenness_forward_cpu(partition_t* par) {
  // Get the current state of the algorithm
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
          // Need to place the updated numSPs value in the outbox to be sent
          // to the remote partition during the communication phase
          uint32_t* values = (uint32_t*)(((par->outbox)[nbr_pid]).push_values);
          uint32_t* dst = &values[GET_VERTEX_ID(nbr)];
          // Distance will be updated when the scatter function which
          // corresponds to this remote vertex is called during the
          // communication phase
          done = false;
          __sync_fetch_and_add(dst, state->numSPs[v]); 
        } else {
          // Can just handle the updates locally
          nbr = GET_VERTEX_ID(nbr);
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
  // If there is remaining work to do, set the done flag to false
  if (!done) *(state->done) = false;
}

/**
 * Distributes work to either the CPU or GPU
 */
PRIVATE void betweenness_forward(partition_t* par) {
  // Check if there is no work to be done
  if (!par->subgraph.vertex_count) return;

  // Clear the outbox before the computation phase
  engine_set_outbox(par->id, 0); 

  // Get the current state of the algorithm
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  // Check which kind of processor this partition corresponds to and
  // call the appropriate function to perform forward propagation
  if (par->processor.type == PROCESSOR_CPU) {
    betweenness_forward_cpu(par);
  } else if (par->processor.type == PROCESSOR_GPU) {
    betweenness_forward_gpu(par);
  } else {
    assert(false);
  }
  // Increment the level for the next round of forward propagation
  state->level++;
}

/**
 * The neighbors backward propagation processing function. This function 
 * computes the delta of a vertex.
 */
__device__ inline
void backward_process_neighbors(partition_t* par, vid_t warp_offset, 
                                vid_t* nbrs, vid_t nbr_count, 
                                uint32_t my_numSPs, score_t* vwarp_delta_s, 
                                uint32_t* numSPs_d, cost_t* distance_d, 
                                score_t* delta_d, cost_t level,
                                score_t* my_delta_d, score_t* my_bc_d) {
  vwarp_delta_s[warp_offset] = 0;
  // Iterate through the portion of work
  for(vid_t i = warp_offset; i < nbr_count; i += VWARP_WARP_SIZE) {
    vid_t nbr = nbrs[i];
  
    // Check whether the neighbour is local or remote and update accordingly
    int nbr_pid = GET_PARTITION_ID(nbr);                             
    if (nbr_pid != par->id) {  
      // The neighbour is remote, so we'll need to pull their values as they
      // will not be stored in the processing unit's local memory
      betweenness_backward_t* value = &((betweenness_backward_t*)
        ((par->outbox_d)[nbr_pid].pull_values))[GET_VERTEX_ID(nbr)];
      // In the gather function, if the node's distance is not equal to
      // level + 1, it's delta value is set to INFINITE to encode that the
      // corresponding numSPs and delta values should not be used in 
      // computing the delta for this node.
      if (value->delta != INFINITE) {
        // Compute an intermediary delta value in shared memory
        vwarp_delta_s[warp_offset] += ((((score_t)my_numSPs) / 
                                      ((score_t)(value->numSPs))) * 
                                      (value->delta + 1));
      }
    } else {
      // Can just handle the updates locally
      nbr = GET_VERTEX_ID(nbr); 
      if (distance_d[nbr] == level + 1) {
        // Compute an intermediary delta value in shared memory
        vwarp_delta_s[warp_offset] += ((((score_t)my_numSPs) / 
                                      ((score_t)numSPs_d[nbr])) * 
                                      (delta_d[nbr] + 1));
      }
    }    
  }

  // Only one thread in the warp aggregates the final value of delta
  if (warp_offset == 0) {
    score_t delta = 0;
    for (vid_t i = 0; i < VWARP_WARP_SIZE; i++) {
      delta += vwarp_delta_s[i];
    }
    // Add the dependency to the BC sum
    if (delta) {
      *my_delta_d = delta;
      *my_bc_d += delta;
    }
  }
}

/**
 * CUDA kernel which performs backward propagation
 */
__global__
void betweenness_gpu_backward_kernel(partition_t par, cost_t level, 
                                     uint32_t* numSPs_d, cost_t* distance_d,
                                     score_t* delta_d, 
                                     score_t* betweenness_scores_d, 
                                     uint32_t thread_count) {
  if (THREAD_GLOBAL_INDEX >= thread_count) return;
  // Determine the warp parameters for this processing unit
  vid_t warp_offset = THREAD_GLOBAL_INDEX % VWARP_WARP_SIZE;
  vid_t warp_id     = THREAD_GLOBAL_INDEX / VWARP_WARP_SIZE; 
  
  // Each warp has a single entry in the following shared memory array.
  // The entry corresponds to a batch of work which will be processed
  // in parallel by a warp of threads.
  __shared__ batch_mem_t batch_s[(MAX_THREADS_PER_BLOCK / VWARP_WARP_SIZE)];

  // Get a reference to the batch of work of the warp this thread belongs to
  batch_mem_t* vwarp_batch_s = &batch_s[THREAD_GRID_INDEX / VWARP_WARP_SIZE];

  // Calculate the starting vertex of the batch
  vid_t base_v = warp_id * VWARP_BATCH_SIZE;

  // Cache the state of my warp's batch in the shared memory space
  vwarp_memcpy(vwarp_batch_s->vertices, &(par.subgraph.vertices[base_v]),
               VWARP_BATCH_SIZE + 1, warp_offset);
  vwarp_memcpy(vwarp_batch_s->distance, &distance_d[base_v], VWARP_BATCH_SIZE,
               warp_offset);
  vwarp_memcpy(vwarp_batch_s->numSPs, &numSPs_d[base_v], VWARP_BATCH_SIZE, 
               warp_offset);

  // Each thread in every warp has an entry in the following array which will be
  // used to calculate intermediary delta values in shared memory
  __shared__ score_t delta_s[MAX_THREADS_PER_BLOCK];

  // Get a reference to the entry of the first thread in the warp. This will be
  // indexed later using warp_offset
  int index = THREAD_GRID_INDEX / VWARP_WARP_SIZE;
  score_t* vwarp_delta_s = &delta_s[index * VWARP_WARP_SIZE];

  // Iterate over the warp's batch of work
  for(vid_t v = 0; v < VWARP_BATCH_SIZE; v++) {
    if (vwarp_batch_s->distance[v] == level) {
      // If the vertex is at the current level, determine its contribution
      // to the source vertex's delta value
      vid_t* nbrs = &(par.subgraph.edges[vwarp_batch_s->vertices[v]]);
      vid_t nbr_count = vwarp_batch_s->vertices[v + 1] - 
        vwarp_batch_s->vertices[v];
      backward_process_neighbors(&par, warp_offset, nbrs, nbr_count, 
                                 vwarp_batch_s->numSPs[v], vwarp_delta_s,
                                 numSPs_d, distance_d, delta_d, level, 
                                 &delta_d[base_v + v], 
                                 &betweenness_scores_d[base_v + v]);
    }
  }
}

/**
 * Entry point for backward propagation on GPU
 */
PRIVATE inline void betweenness_backward_gpu(partition_t* par) {
  // Get the current state of the algorithm
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  // Given this state, invoke the CUDA kernel which performs
  // backward propagation
  betweenness_gpu_backward_kernel<<<state->blocks, state->threads, 0, 
    par->streams[1]>>>(*par, state->level, state->numSPs, state->distance,
                       state->delta, state->betweenness,
                       VWARP_BATCH_COUNT(par->subgraph.vertex_count) *
                       VWARP_WARP_SIZE);
  CALL_CU_SAFE(cudaGetLastError());
}

/**
 * Entry point for backward propagation on CPU
 */
void betweenness_backward_cpu(partition_t* par) {
  // Get the current state of the algorithm
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  graph_t* subgraph = &par->subgraph;
  
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
          // The neighbour is remote, so we'll need to pull their values as they
          // will not be stored in the processing unit's local memory
          betweenness_backward_t* value = &((betweenness_backward_t*)
            ((par->outbox)[nbr_pid].pull_values))[GET_VERTEX_ID(nbr)];
          // In the gather function, if the node's distance is not equal to
          // level + 1, it's delta value is set to INFINITE to encode that the
          // corresponding numSPs and delta values should not be used in 
          // computing the delta for this node.
          if (value->delta != INFINITE) {
            state->delta[v] += ((((score_t)(state->numSPs[v])) / 
                               ((score_t)(value->numSPs))) * 
                               (value->delta + 1));
          }
        } else {
          // Can just handle the updates locally    
          nbr = GET_VERTEX_ID(nbr);
          if (state->distance[nbr] == state->level + 1) {
            state->delta[v] += ((((score_t)(state->numSPs[v])) /
                               ((score_t)(state->numSPs[nbr]))) *
                               (state->delta[nbr] + 1));
          }
        }     
      }
      // Add the dependency to the BC sum
      state->betweenness[v] += state->delta[v];
    }
  }
}

/**
 * Distributes work for backward propagation to either the CPU or GPU
 */
PRIVATE void betweenness_backward(partition_t* par) {
  // Check if there is no work to be done
  if (!par->subgraph.vertex_count) return;

  // Get the current state of the algorithm
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  // Check what kind of processing unit corresponds to this partition and
  // then call the appropriate function to perform backward propagation
  if (par->processor.type == PROCESSOR_CPU) {
    betweenness_backward_cpu(par);
  } else if (par->processor.type == PROCESSOR_GPU) {
    betweenness_backward_gpu(par);
  } else {
    assert(false);
  }
  // Decrement the level for the next round of backward propagation
  state->level--;

  // Check whether backward propagation is finished
  if (state->level > 0) {
    engine_report_not_finished();
  }
}

/*
 * Parallel CPU implementation of betweenness scatter function
 */
PRIVATE inline void betweenness_scatter_cpu(grooves_box_table_t* inbox, 
                                            betweenness_state_t* state) {
  OMP(omp parallel for schedule(static))
  for (vid_t index = 0; index < inbox->count; index++) {
    // Get the values that have been pushed to this vertex
    vid_t vid = inbox->rmt_nbrs[index];
    uint32_t* inbox_values = (uint32_t*)inbox->push_values;
    if (inbox_values[index] != 0) {
      // If the distance was previously infinity, initialize it to the
      // current level 
      if (state->distance[vid] == INF_COST) {
        state->distance[vid] = state->level;
      }
      // If the distance is equal to the current level, update the 
      // nodes number of shortest paths with the pushed value
      if (state->distance[vid] == state->level) {
        state->numSPs[vid] += inbox_values[index];
      }
    }
  }
}

/*
 * Kernel for betweenness_scatter_gpu
 */
__global__ void betweenness_scatter_kernel(grooves_box_table_t inbox, 
                                           cost_t* distance, uint32_t* numSPs,
                                           cost_t level, bool* done) {
  vid_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) return;
  // Get the values that have been pushed to this vertex
  vid_t vid = inbox.rmt_nbrs[index];
  uint32_t* inbox_values = (uint32_t*)inbox.push_values;
  if (inbox_values[index] != 0) {
    // If the distance was previously infinity, initialize it to the
    // current level   
    if (distance[vid] == INF_COST) {
      distance[vid] = level;
    }
    // If the distance is equal to the current level, update the 
    // nodes number of shortest paths with the pushed value
    if (distance[vid] == level) {
      numSPs[vid] += inbox_values[index];
    }
  }
}

/*
 * Parallel GPU implementation of betweenness scatter function
 */
PRIVATE inline void betweenness_scatter_gpu(grooves_box_table_t* inbox, 
                                            betweenness_state_t* state) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(inbox->count, blocks, threads);
  // Invoke the appropriate CUDA kernel to perform the scatter functionality
  betweenness_scatter_kernel<<<blocks, threads>>>(*inbox, state->distance, 
                                                  state->numSPs, state->level,
                                                  state->done);
  CALL_CU_SAFE(cudaGetLastError());
}

/**
 * Update the number of shortest paths from remote vertices
 * Also update distance if it has yet to be initialized
 */
PRIVATE void betweenness_scatter_forward(partition_t* par) {
  // Check if there is no work to be done
  if (!par->subgraph.vertex_count) return;

  // Get the current state of the algorithm
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;

  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;
    // For all remote partitions, get the corresponding inbox
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    // If the inbox has some values, determine which type of processing unit
    // corresponds to this partition and call the appropriate scatter function
    if (par->processor.type == PROCESSOR_CPU) {
      betweenness_scatter_cpu(inbox, state);
    } else if (par->processor.type == PROCESSOR_GPU) {
      betweenness_scatter_gpu(inbox, state);
    } else {
      assert(false);
    }
  }
}

/*
 * Parallel CPU implementation of betweenness gather function
 */
PRIVATE inline void betweenness_gather_cpu(grooves_box_table_t* inbox, 
                                           betweenness_state_t* state,
                                           betweenness_backward_t* values) {
  OMP(omp parallel for schedule(static))
  for (vid_t index = 0; index < inbox->count; index++) {
    vid_t vid = inbox->rmt_nbrs[index];
    // Check whether the vertex's distance is equal to level + 1
    if (state->distance[vid] == (state->level + 1)) {
      // If it is, we'll pass the vertex's delta and numSPs values to
      // neighbouring nodes to be used during their backward propagation phase
      values[index].delta  = state->delta[vid];
      values[index].numSPs = state->numSPs[vid];
    } else {
      // If it's not, we'll set the delta value to INFINITE to indicate to
      // neighbouring nodes that this node does not contribute to their
      // delta values during backwards propagation
      values[index].delta  = INFINITE;
    }
  }
}

/*
 * Kernel for betweenness_gather_gpu
 */
__global__ void betweenness_gather_kernel(grooves_box_table_t inbox, 
                                          cost_t* distance, cost_t level,
                                          uint32_t* numSPs, score_t* delta,
                                          betweenness_backward_t* values) {
  vid_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) return;
  vid_t vid = inbox.rmt_nbrs[index];
  // Check whether the vertex's distance is equal to level + 1
  if (distance[vid] == level + 1) {
    // If it is, we'll pass the vertex's delta and numSPs values to
    // neighbouring nodes to be used during their backward propagation phase
    values[index].delta  = delta[vid];
    values[index].numSPs = numSPs[vid];
  } else {
    // If it's not, we'll set the delta value to INFINITE to indicate to
    // neighbouring nodes that this node does not contribute to their
    // delta values during backwards propagation
    values[index].delta  = INFINITE;
  }
}

/*
 * Parallel GPU implementation of betweenness gather function
 */
PRIVATE inline void betweenness_gather_gpu(grooves_box_table_t* inbox, 
                                           betweenness_state_t* state,
                                           betweenness_backward_t* values) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(inbox->count, blocks, threads); 
  // Invoke the appropriate CUDA kernel to perform the gather functionality
  betweenness_gather_kernel<<<blocks, threads>>>(*inbox, state->distance, 
                                                 state->level, state->numSPs,
                                                 state->delta, values);
  CALL_CU_SAFE(cudaGetLastError());
}

/**
 * Pass the number of shortest paths and delta values to neighbouring
 * vertices to be used in the backwards propagation phase
 */
PRIVATE void betweenness_gather_backward(partition_t* par) {
  // Check if there is no work to be done
  if (!par->subgraph.vertex_count) return;

  // Get the current state of the algorithm
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid]; 
    // For all remote partitions, get the corresponding inbox
    if (!inbox->count) continue;
    betweenness_backward_t* values = (betweenness_backward_t*)
                                     inbox->pull_values;
    // If the inbox has some values, determine which type of processing unit
    // corresponds to this partition and call the appropriate gather function
    if (par->processor.type == PROCESSOR_CPU) {
      betweenness_gather_cpu(inbox, state, values);
    } else if (par->processor.type == PROCESSOR_GPU) {
      betweenness_gather_gpu(inbox, state, values);
    } else {
      assert(false);
    }   
  }
}

/**
 * Initializes the state for a round of backward propagation
 */
PRIVATE void betweenness_init_backward(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  // Get the current state of the algorithm
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  assert(state);
  vid_t vcount = par->subgraph.vertex_count;

  // Determine which type of memory this partition corresponds to
  totem_mem_t type = TOTEM_MEM_HOST; 
  if (par->processor.type == PROCESSOR_GPU) { 
    type = TOTEM_MEM_DEVICE;
    // If this is a GPU partition, also initial the kernel parameters
    KERNEL_CONFIGURE(VWARP_WARP_SIZE * VWARP_BATCH_COUNT(vcount),
                     state->blocks, state->threads);
  }

  // Initialize the delta values to 0
  totem_memset(state->delta, (score_t)0, vcount, type, par->streams[1]);
}

/**
 * Initializes the state for a round of forward propagation
 */
PRIVATE void betweenness_init_forward(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  // Get the current state of the algorithm
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  assert(state);
  // Get the source partition and source vertex values
  id_t src_pid = GET_PARTITION_ID(bc_g.src);
  id_t src_vid = GET_VERTEX_ID(bc_g.src);
  vid_t vcount = par->subgraph.vertex_count;

  // Determine which type of memory this partition corresponds to
  totem_mem_t type = TOTEM_MEM_HOST; 
  if (par->processor.type == PROCESSOR_GPU) { 
    type = TOTEM_MEM_DEVICE;
    // If this is a GPU partition, also initialize the kernel parameters
    KERNEL_CONFIGURE(VWARP_WARP_SIZE * VWARP_BATCH_COUNT(vcount),
                     state->blocks, state->threads);
  }

  // Initialize the distances to infinity and numSPs to 0
  totem_memset((state->distance), INF_COST, vcount, type, par->streams[1]);
  totem_memset((state->numSPs), (uint32_t)0, vcount, type, par->streams[1]);
  if (src_pid == par->id) {
    // For the source vertex, initialize its own distance and numSPs
    totem_memset(&((state->distance)[src_vid]), (cost_t)0, 1, type,
                 par->streams[1]);
    totem_memset(&((state->numSPs)[src_vid]), (uint32_t)1, 1, type,
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
  // Allocate memory for the per-partition state
  betweenness_state_t* state = (betweenness_state_t*)
                               calloc(1, sizeof(betweenness_state_t));
  assert(state); 
  // Set the partition's state variable to the previously allocated state
  par->algo_state = state;
  vid_t vcount = par->subgraph.vertex_count;

  // Determine which type of memory this partition corresponds to
  totem_mem_t type = TOTEM_MEM_HOST; 
  if (par->processor.type == PROCESSOR_GPU) { 
    type = TOTEM_MEM_DEVICE;
  }
  
  // Allocate memory for the various pieces of data required for the
  // Betweenness Centrality algorithm
  totem_calloc(vcount * sizeof(cost_t), type, (void**)&(state->distance));
  totem_calloc(vcount * sizeof(uint32_t), type, (void**)&(state->numSPs));
  totem_calloc(vcount * sizeof(score_t), type, (void**)&(state->delta));
  totem_calloc(vcount * sizeof(score_t), type, (void**)&(state->betweenness));

  // Initialize the state's done flag
  state->done = engine_get_finished_ptr(par->id);

  // Initialize the state
  betweenness_init_forward(par); 
}

/**
 * Cleans up allocated memory on the CPU and GPU
 */
PRIVATE void betweenness_finalize(partition_t* par) {
  // Check if there is no work to be done
  if (!par->subgraph.vertex_count) return;  

  // Free the allocated memory
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
 
  // Determine which type of memory this partition corresponds to
  totem_mem_t type = TOTEM_MEM_HOST; 
  if (par->processor.type == PROCESSOR_GPU) { 
    type = TOTEM_MEM_DEVICE; 
  }

  // Free the memory allocated for the algorithm
  totem_free(state->distance, type);
  totem_free(state->numSPs, type);
  totem_free(state->delta, type);
  totem_free(state->betweenness, type);

  // Free the per-partition state and set it to NULL
  free(state);
  par->algo_state = NULL;
}

/**
 * Aggregates the final result to be returned at the end
 */
PRIVATE void betweenness_aggr(partition_t* par) {  
  if (!par->subgraph.vertex_count) return;
  // Get the current state of the algorithm
  betweenness_state_t* state = (betweenness_state_t*)par->algo_state;
  graph_t* subgraph = &par->subgraph;
  score_t* betweenness_values = NULL;
  // Determine which type of processor this partition corresponds to
  if (par->processor.type == PROCESSOR_CPU) {
    // If it is a CPU partition, grab the computed betweenness value directly
    betweenness_values = state->betweenness;
  } else if (par->processor.type == PROCESSOR_GPU) {
    // If it is a GPU partition, copy the computed score back to the host
    assert(bc_g.betweenness_score_h);
    CALL_CU_SAFE(cudaMemcpy(bc_g.betweenness_score_h, state->betweenness, 
                            subgraph->vertex_count * sizeof(score_t),
                            cudaMemcpyDefault));
    betweenness_values = bc_g.betweenness_score_h;
  } else {
    assert(false);
  }
  // Aggregate the results
  assert(bc_g.betweenness_score);
  OMP(omp parallel for schedule(static))
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    // Check whether we are computing exact centrality values
    if (bc_g.epsilon == CENTRALITY_EXACT) {
      // Return the exact values computed
      bc_g.betweenness_score[par->map[v]] = betweenness_values[v];
    }
    else {
      // Scale the computed Betweenness Centrality metrics since they were
      // computed using a subset of the total nodes within the graph
      // The scaling value is: (Total Number of Nodes / Subset of Nodes Used)
      bc_g.betweenness_score[par->map[v]] = betweenness_values[v] *
        (score_t)(((double)(engine_vertex_count())) / bc_g.num_samples); 
    }
  }
}

/**
 * Core functionality for main for loop within the BC computation
 */
void betweenness_hybrid_core(vid_t source, bool is_first_iteration,
                             bool is_last_iteration) {
  // Set the source node for this iteration
  bc_g.src  = engine_vertex_id_in_partition(source);

  // Forward propagation
  engine_par_init_func_t init_forward = betweenness_init_forward;
  if (is_first_iteration) {
    init_forward = betweenness_init;
  }
  // Configure the parameters for forward propagation given the current
  // iteration of the overall computation
  engine_config_t config_forward = {
    NULL, betweenness_forward, betweenness_scatter_forward, NULL, 
    init_forward, NULL, NULL, GROOVES_PUSH
  };
  // Call Totem to begin the computation phase given the specified 
  // configuration
  engine_config(&config_forward);
  engine_execute();

  // Backward propagation
  engine_par_finalize_func_t finalize_backward = NULL;
  engine_par_aggr_func_t aggr_backward = NULL;
  if (is_last_iteration) {
    finalize_backward = betweenness_finalize;
    aggr_backward = betweenness_aggr;
  }
  // Configure the parameters for backward propagation given the current
  // iteration of the overall computation
  engine_config_t config_backward = {
    NULL, betweenness_backward, NULL, betweenness_gather_backward,
    betweenness_init_backward, finalize_backward, aggr_backward, GROOVES_PULL
  };
  // Call Totem to begin the computation phase given the specified 
  // configuration
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

  // Initialize the global state
  bc_g.betweenness_score = betweenness_score;
  totem_memset(bc_g.betweenness_score, (score_t)0, engine_vertex_count(),
               TOTEM_MEM_HOST);
  bc_g.epsilon = epsilon;

  if (engine_largest_gpu_partition()) {
      bc_g.betweenness_score_h = (score_t*)mem_alloc(
                                 engine_largest_gpu_partition() *
                                 sizeof(score_t));
  }

  // Determine whether we will compute exact or approximate BC values
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
    // Store the number of samples used in the global state to be used for 
    // scaling the computed metric during aggregation
    bc_g.num_samples = num_samples;
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
 
  // Clean up and return
  if (engine_largest_gpu_partition()) mem_free(bc_g.betweenness_score_h);
  memset(&bc_g, 0, sizeof(betweenness_global_state_t));
  return SUCCESS;
}
