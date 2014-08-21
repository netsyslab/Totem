/**
 * This file contains an implementation of the single source shortest path 
 * (SSSP) algorithm using the totem framework.
 *
 *  Created on: 2014-05-10
 *  Author: Tahsin Reza
 */

#include "totem_alg.h"
#include "totem_engine.cuh"

/**
 * per-partition specific state
 */
typedef struct sssp_state_s {
  // TODO(treza): Use "bitmap_t" instead of "bool" for "updated".
  bool* updated;  // one slot per vertex in the partition
                  // it indicates whether the corresponding vertex has been
                  // updated and that it should try to update the distances of
                  // its neighbours
  bool* finished;  // points to Totem's finish flag
  weight_t* distance;  // stores results in the partition
} sssp_state_t;

/**
 * state shared between all partitions
 */
typedef struct sssp_global_state_s {
  vid_t src;  // source vertex id (the id after partitioning)
  weight_t* distance;  // stores the final results
  weight_t* distance_h;  // temporary buffer for GPU
} sssp_global_state_t;

PRIVATE sssp_global_state_t state_g = {0, NULL, NULL};

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
 */
PRIVATE error_t check_special_cases(vid_t src, weight_t* distance,
                                    bool* finished) {
  *finished = true;
  if ((src >= engine_vertex_count()) || (distance == NULL)) {
    return FAILURE;
  } else if (engine_vertex_count() == 1) {
    distance[0] = 0.0; 
    return SUCCESS;
  } else if (engine_edge_count() == 0) {
    // Initialize distance
    totem_memset(distance, WEIGHT_MAX, engine_vertex_count(), TOTEM_MEM_HOST);
    distance[src] = 0.0;
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

void sssp_cpu(partition_t* par, sssp_state_t* state) {
  graph_t* subgraph = &par->subgraph;  
  bool finished = true;

  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    if (state->updated[v] == false) { continue; }
    state->updated[v] = false;  

    for (eid_t i = subgraph->vertices[v]; i < subgraph->vertices[v + 1]; 
      i++) { 
      int nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
      vid_t nbr = GET_VERTEX_ID(subgraph->edges[i]);     
      vid_t nbr_dst = subgraph->edges[i];
      weight_t* dst = engine_get_dst_ptr(par->id, nbr_dst, par->outbox, 
                                         state->distance); 
      weight_t new_distance = state->distance[v] + subgraph->weights[i];
      weight_t old_distance =             
         __sync_fetch_and_min_float(dst, new_distance); 
      if (new_distance < old_distance) {
        if (nbr_pid == par->id) {
          state->updated[nbr] = true;
        }
        finished = false;
      }
    }
  } 
  if (!finished) *(state->finished) = false; 
}

__global__
void sssp_kernel(partition_t par, sssp_state_t state) {
  vid_t v = THREAD_GLOBAL_INDEX;
  if (v >= par.subgraph.vertex_count) { return; }

  __shared__ bool finished_block;
  finished_block = true;
  __syncthreads();

  if (state.updated[v] == false) { return; }
  state.updated[v] = false;

  for (eid_t i = par.subgraph.vertices[v]; i < par.subgraph.vertices[v + 1];
      i++) {
    int nbr_pid = GET_PARTITION_ID(par.subgraph.edges[i]);
    vid_t nbr = GET_VERTEX_ID(par.subgraph.edges[i]);
    vid_t nbr_dst = par.subgraph.edges[i];
    weight_t* dst = engine_get_dst_ptr(par.id, nbr_dst, 
                                       par.outbox, state.distance);
    weight_t new_distance = state.distance[v] + par.subgraph.weights[i];
    weight_t old_distance = atomicMin(dst, new_distance);
    if (new_distance < old_distance) {
      if (nbr_pid == par.id) {
          state.updated[nbr] = true;
      }
      finished_block = false;
    }
  }
  __syncthreads();
  if (!finished_block) *state.finished = false; 
}

PRIVATE void sssp_gpu(partition_t* par, sssp_state_t* state) {
  vid_t vertex_count = par->subgraph.vertex_count;
  dim3 blocks, threads;
  KERNEL_CONFIGURE(vertex_count, blocks, threads);  
  sssp_kernel<<<blocks, threads, 1, par->streams[1]>>>(*par, *state); 
  CALL_CU_SAFE(cudaGetLastError()); 
}

PRIVATE void sssp(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  sssp_state_t* state = reinterpret_cast<sssp_state_t*>(par->algo_state);
  if (par->processor.type == PROCESSOR_CPU) {
    sssp_cpu(par, state);
  } else if (par->processor.type == PROCESSOR_GPU) {  
    sssp_gpu(par, state);
  } else {
    assert(false);
  }
}

PRIVATE void sssp_scatter_cpu(grooves_box_table_t* inbox, 
                              sssp_state_t* state) {
  // Get the values that have been pushed to this vertex
  weight_t* inbox_values = reinterpret_cast<weight_t*>(inbox->push_values);
  OMP(omp parallel for schedule(static))
  for (vid_t index = 0; index < inbox->count; index++) {  
    vid_t vid = inbox->rmt_nbrs[index];  
    weight_t old_distance = state->distance[vid];     
    state->distance[vid] = 
      inbox_values[index] < state->distance[vid] ? 
      inbox_values[index] : state->distance[vid]; 
    weight_t new_distance = state->distance[vid];
    if (old_distance > new_distance)  {
      state->updated[vid] = true;
    }      
  }
}

__global__ 
void sssp_scatter_kernel(grooves_box_table_t inbox, sssp_state_t state) {
  vid_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) { return; }

  // Get the values that have been pushed to this vertex
  weight_t* inbox_values = reinterpret_cast<weight_t*>(inbox.push_values);
  vid_t vid = inbox.rmt_nbrs[index];
  weight_t old_distance = state.distance[vid];
  state.distance[vid] =
    inbox_values[index] < state.distance[vid] ?
    inbox_values[index] : state.distance[vid];
  weight_t new_distance = state.distance[vid];
  if (old_distance > new_distance) {
    state.updated[vid] = true;
  }  
}

PRIVATE void sssp_scatter_gpu(partition_t* par, grooves_box_table_t* inbox,
                              sssp_state_t* state) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(inbox->count, blocks, threads);
  // Invoke the appropriate CUDA kernel to perform the scatter functionality
  sssp_scatter_kernel<<<blocks, threads, 0, par->streams[1]>>>
    (*inbox, *state);
  CALL_CU_SAFE(cudaGetLastError());  
}

PRIVATE void sssp_scatter(partition_t* par) {  
  // Check if there is no work to be done
  if (par->subgraph.vertex_count == 0) { return; }  
  // Get the current state of the algorithm
  sssp_state_t* state = reinterpret_cast<sssp_state_t*>(par->algo_state);

  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;
    // For all remote partitions, get the corresponding inbox
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    // If the inbox has some values, determine which type of processing unit
    // corresponds to this partition and call the appropriate scatter function
    if (par->processor.type == PROCESSOR_CPU) {
      sssp_scatter_cpu(inbox, state);
    } else if (par->processor.type == PROCESSOR_GPU) {
      sssp_scatter_gpu(par, inbox, state);
    } else {
      assert(false);
    }
  }
}

PRIVATE void sssp_aggregate(partition_t* par) {
  if (!par->subgraph.vertex_count) { return; }
  sssp_state_t* state = reinterpret_cast<sssp_state_t*>(par->algo_state);
  graph_t* subgraph = &par->subgraph;
  weight_t* src_distance = NULL; 
  if (par->processor.type == PROCESSOR_CPU) {
    src_distance = state->distance; 
  } else if (par->processor.type == PROCESSOR_GPU) {
    assert(state_g.distance_h);
    CALL_CU_SAFE(cudaMemcpy(state_g.distance_h, state->distance, 
                            subgraph->vertex_count * sizeof(weight_t),
                            cudaMemcpyDefault));  
    src_distance = state_g.distance_h;
  } else {
    assert(false);
  }
  assert(state_g.distance);
  OMP(omp parallel for schedule(static))
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    state_g.distance[par->map[v]] = src_distance[v];
  }
}

PRIVATE void sssp_init(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  sssp_state_t* state = reinterpret_cast<sssp_state_t*>
    (calloc(1, sizeof(sssp_state_t)));
  assert(state);
  par->algo_state = state;

  totem_mem_t type;
  if (par->processor.type == PROCESSOR_CPU) {
     type = TOTEM_MEM_HOST;
  } else if (par->processor.type == PROCESSOR_GPU) {
    type = TOTEM_MEM_DEVICE;    
  } else {
    assert(false);
  }

  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(bool), type, 
                         reinterpret_cast<void**>(&(state->updated))));
  totem_memset(state->updated, false, par->subgraph.vertex_count, type, 
               par->streams[1]);

  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(weight_t), type,
                         reinterpret_cast<void**>(&(state->distance))));
  totem_memset(state->distance, WEIGHT_MAX, par->subgraph.vertex_count, type,
               par->streams[1]);

  if (GET_PARTITION_ID(state_g.src) == par->id) {
    // For the source vertex, initialize updated status
    // Please note that instead of simply initializing the updated status of
    // the source using the following expression  
    // "state->updated[GET_VERTEX_ID(state_g.src)] = true", we are using 
    // "memset" becuase the sourec may belongs to a partition which 
    // resides on the GPU.
    totem_memset(&((state->updated)[GET_VERTEX_ID(state_g.src)]), true, 1, 
                 type, par->streams[1]);
    
    // For the source vertex, initialize distance
    // Please note that instead of simply initializing the updated status of
    // the source using the following expression  
    // "state->distance[GET_VERTEX_ID(state_g.src)] = 0.0", we are using 
    // "memset" becuase the sourec may belongs to a partition which 
    // resides on the GPU.
    totem_memset(&((state->distance)[GET_VERTEX_ID(state_g.src)]), (weight_t)0.0, 1, 
                 type, par->streams[1]);
  }

  state->finished = engine_get_finished_ptr(par->id);
  engine_set_outbox(par->id, WEIGHT_MAX); 
}

PRIVATE void sssp_finalize(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  sssp_state_t* state = reinterpret_cast<sssp_state_t*>(par->algo_state);
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_CPU) {
  } else if (par->processor.type == PROCESSOR_GPU) {
    type = TOTEM_MEM_DEVICE;
  } else {
    assert(false);
  }
  totem_free(state->updated, type);
  totem_free(state->distance, type);
  free(state);
  par->algo_state = NULL;
}

error_t sssp_hybrid(vid_t src, weight_t* distance) {
  // check for special cases
  bool finished = false;

  error_t rc = check_special_cases(src, distance, &finished);
  if (finished) return rc;

  // initialize the global state
  state_g.distance = distance;
  state_g.src  = engine_vertex_id_in_partition(src);

  // initialize the engine
  engine_config_t config = {
    NULL, sssp, sssp_scatter, NULL, sssp_init, sssp_finalize, sssp_aggregate,
    GROOVES_PUSH
  };
  engine_config(&config);
  if (engine_largest_gpu_partition()) {
    CALL_SAFE(totem_malloc(engine_largest_gpu_partition() * sizeof(weight_t),
                           TOTEM_MEM_HOST,
                           reinterpret_cast<void**>(&state_g.distance_h)));
  }
  engine_execute();

  // clean up and return
  if (engine_largest_gpu_partition()) {
    totem_free(state_g.distance_h, TOTEM_MEM_HOST);
  }
  memset(&state_g, 0, sizeof(sssp_global_state_t));
  return SUCCESS;
}
