/**
 * This file contains an implementation of the single source shortest path
 * (SSSP) algorithm using the totem framework.
 *
 *  Created on: 2014-05-10
 *  Author: Tahsin Reza
 */

#include "totem_alg.h"
#include "totem_engine.cuh"


// per-partition specific state
typedef struct sssp_state_s {
  bool* finished;                           // points to Totem's finish flag
  weight_t* distance[MAX_PARTITION_COUNT];  // stores results in the partition
  bitmap_t updated[MAX_PARTITION_COUNT];    // a list of bitmaps one for each
                                            // remote partition
  frontier_state_t frontier;
} sssp_state_t;


// state shared between all partitions
typedef struct sssp_global_state_s {
  vid_t src;  // source vertex id (the id after partitioning)
  weight_t* distance;  // stores the final results
  weight_t* distance_h;  // temporary buffer for GPU
} sssp_global_state_t;

PRIVATE sssp_global_state_t state_g = {0, NULL, NULL};


// Checks for input parameters and special cases. This is invoked at the
// beginning of public interfaces (GPU and CPU)
PRIVATE error_t check_special_cases(vid_t src, weight_t* distance,
                                    bool* finished) {
  *finished = true;
  if ((src >= engine_vertex_count()) || (distance == NULL)) {
    return FAILURE;
  } else if (engine_vertex_count() == 1) {
    distance[0] = 0;
    return SUCCESS;
  } else if (engine_edge_count() == 0) {
    // Initialize distance
    totem_memset(distance, WEIGHT_MAX, engine_vertex_count(), TOTEM_MEM_HOST);
    distance[src] = 0;
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

void sssp_cpu(partition_t* par, sssp_state_t* state) {
  graph_t* subgraph = &par->subgraph;
  bool finished = true;

  weight_t* distance = state->distance[par->id];

  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    if (!bitmap_is_set(state->updated[par->id], v)) { continue; }
    bitmap_unset_cpu(state->updated[par->id], v);

    for (eid_t i = subgraph->vertices[v]; i < subgraph->vertices[v + 1]; i++) {
      int nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
      vid_t nbr = GET_VERTEX_ID(subgraph->edges[i]);
      weight_t* nbr_distance = state->distance[nbr_pid];
      bitmap_t nbr_updated = state->updated[nbr_pid];
      weight_t old_distance = nbr_distance[nbr];
      weight_t new_distance = distance[v] + subgraph->weights[i];
      if (new_distance < old_distance) {
        if (old_distance ==
            __sync_fetch_and_min_uint32(&nbr_distance[nbr], new_distance)) {
          bitmap_set_cpu(nbr_updated, nbr);
        }
        finished = false;
      }
    }
  }
  if (!finished) *(state->finished) = false;
}

__global__
void sssp_kernel(partition_t par, sssp_state_t state, vid_t vertex_count) {
  vid_t v = THREAD_GLOBAL_INDEX;
  if (v >= vertex_count) { return; }
  v = state.frontier.list[v];

  __shared__ bool finished_block;
  finished_block = true;
  __syncthreads();

  weight_t* distance = state.distance[par.id];

  if (bitmap_is_set(state.updated[par.id], v)) {
    bitmap_unset_gpu(state.updated[par.id], v);
    for (eid_t i = par.subgraph.vertices[v]; i < par.subgraph.vertices[v + 1];
         i++) {
      int nbr_pid = GET_PARTITION_ID(par.subgraph.edges[i]);
      vid_t nbr = GET_VERTEX_ID(par.subgraph.edges[i]);
      weight_t* nbr_distance = state.distance[nbr_pid];
      bitmap_t nbr_updated = state.updated[nbr_pid];
      weight_t old_distance = nbr_distance[nbr];
      weight_t new_distance = distance[v] + par.subgraph.weights[i];
      if (new_distance < old_distance) {
        if ((old_distance == atomicMin(&nbr_distance[nbr], new_distance))) {
          bitmap_set_gpu(nbr_updated, nbr);
        }
        finished_block = false;
      }
    }
  }

  __syncthreads();
  if (!finished_block && THREAD_BLOCK_INDEX == 0) *state.finished = false;
}

PRIVATE void sssp_gpu(partition_t* par, sssp_state_t* state) {
  vid_t vertex_count = frontier_count_gpu(&state->frontier, par->streams[1]);
  if (vertex_count == 0) { return; }
  assert(state->frontier.list_len >= vertex_count);
  frontier_update_list_gpu(&state->frontier, par->streams[1]);
  dim3 blocks, threads;
  KERNEL_CONFIGURE(vertex_count, blocks, threads);
  sssp_kernel<<<blocks, threads, 1, par->streams[1]>>>(
      *par, *state, vertex_count);
  CALL_CU_SAFE(cudaGetLastError());
}

template<int VWARP_WIDTH, int VWARP_BATCH>
PRIVATE __global__ void sssp_vwarp_kernel(partition_t par, sssp_state_t state,
                                          vid_t vertex_count) {
  if (THREAD_GLOBAL_INDEX >=
      vwarp_thread_count(vertex_count, VWARP_WIDTH, VWARP_BATCH)) { return; }

  const eid_t* __restrict vertices = par.subgraph.vertices;

  __shared__ bool finished_block;
  finished_block = true;
  __syncthreads();

  weight_t* distance = state.distance[par.id];
  vid_t start_vertex = vwarp_block_start_vertex(VWARP_WIDTH, VWARP_BATCH) +
    vwarp_warp_start_vertex(VWARP_WIDTH, VWARP_BATCH);
  vid_t end_vertex = start_vertex +
    vwarp_warp_batch_size(vertex_count, VWARP_WIDTH, VWARP_BATCH);
  int warp_offset = vwarp_thread_index(VWARP_WIDTH);
  for (vid_t i = start_vertex; i < end_vertex; i++) {
    vid_t v = state.frontier.list[i];
    if (bitmap_is_set(state.updated[par.id], v)) {
      bitmap_unset_gpu(state.updated[par.id], v);
      const eid_t nbr_count = vertices[v + 1] - vertices[v];
      vid_t* nbrs = par.subgraph.edges + vertices[v];
      weight_t* weights = par.subgraph.weights + vertices[v];
      if (v >= par.subgraph.vertex_ext) {
        nbrs = par.subgraph.edges_ext +
          (vertices[v] - par.subgraph.edge_count_ext);
      }
      for (vid_t i = warp_offset; i < nbr_count; i += VWARP_WIDTH) {
        int nbr_pid = GET_PARTITION_ID(nbrs[i]);
        vid_t nbr = GET_VERTEX_ID(nbrs[i]);
        weight_t* nbr_distance = state.distance[nbr_pid];
        bitmap_t nbr_updated   = state.updated[nbr_pid];
        weight_t old_distance = nbr_distance[nbr];
        weight_t new_distance = distance[v] + weights[i];
        if (new_distance < old_distance) {
          if (old_distance == atomicMin(&nbr_distance[nbr], new_distance)) {
            bitmap_set_gpu(nbr_updated, nbr);
          }
          finished_block = false;
        }
      }  // for
    }  // if
  }  // for
  __syncthreads();
  if (!finished_block && THREAD_BLOCK_INDEX == 0) *state.finished = false;
}

template<int VWARP_WIDTH, int BATCH_SIZE>
PRIVATE void sssp_gpu_launch(partition_t* par, sssp_state_t* state) {
  vid_t vertex_count = frontier_count_gpu(&state->frontier, par->streams[1]);
  if (vertex_count == 0) { engine_report_no_comm(par->id); return; }
  frontier_update_list_gpu(&state->frontier, par->streams[1]);
  const int threads = MAX_THREADS_PER_BLOCK;
  dim3 blocks;
  assert(VWARP_WIDTH <= threads);
  kernel_configure(vwarp_thread_count(vertex_count, VWARP_WIDTH, BATCH_SIZE),
                   blocks, threads);
  sssp_vwarp_kernel<VWARP_WIDTH, BATCH_SIZE>
      <<<blocks, threads, 0, par->streams[1]>>>(*par, *state, vertex_count);
}

typedef void(*sssp_gpu_func_t)(partition_t*, sssp_state_t*);

PRIVATE void sssp_vwarp_gpu(partition_t* par, sssp_state_t* state) {
  PRIVATE const sssp_gpu_func_t SSSP_GPU_FUNC[] = {
    // RANDOM algorithm
    sssp_gpu_launch<VWARP_MEDIUM_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE>,
    // HIGH partitioning
    sssp_gpu_launch<VWARP_MEDIUM_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE>,
    // LOW partitioning
    // Note that it is not possible to use a virtual warp width longer than the
    // hardware warp width. This is because a vertex may switch from inactive
    // to active state (maintained by the updated array) during the execution
    // of a round. This may lead to the situation where the threads of a
    // virtual warp, which are all supposed to process the neighbours of a
    // vertex, evaluate the vertex's active state differently, and hence part
    // of the neighbours of that vertex will not get processed.
    sssp_gpu_launch<VWARP_HARDWARE_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE>
  };
  int par_alg = engine_partition_algorithm();
  SSSP_GPU_FUNC[par_alg](par, state);
}

PRIVATE void reset_rmt_updated(partition_t* par, sssp_state_t* state) {
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    vid_t count = par->outbox[pid].count;
    if (pid != par->id && count != 0) {
      if (par->processor.type == PROCESSOR_GPU) {
        bitmap_reset_gpu(state->updated[pid], count, par->streams[1]);
      } else {
        bitmap_reset_cpu(state->updated[pid], count);
      }
    }
  }
}

PRIVATE void sssp(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  sssp_state_t* state = reinterpret_cast<sssp_state_t*>(par->algo_state);
  reset_rmt_updated(par, state);
  if (par->processor.type == PROCESSOR_CPU) {
    sssp_cpu(par, state);
  } else if (par->processor.type == PROCESSOR_GPU) {
    sssp_gpu(par, state);
  } else {
    assert(false);
  }
}

PRIVATE void sssp_scatter_cpu(grooves_box_table_t* inbox, bitmap_t updated,
                              weight_t* distance) {
  bitmap_t  rmt_updated = reinterpret_cast<bitmap_t>(inbox->push_values);
  weight_t* rmt_distance =
      (weight_t*)&rmt_updated[bitmap_bits_to_words(inbox->count)];
  /* for (vid_t index = 0; index < inbox->count; index++) { */
  OMP(omp parallel for schedule(static))
  for (vid_t word_index = 0; word_index < bitmap_bits_to_words(inbox->count);
       word_index++) {
    bitmap_word_t word = rmt_updated[word_index];
    if (word) {
      vid_t index = word_index * BITMAP_BITS_PER_WORD;
      vid_t bit_last_index = (word_index + 1) * BITMAP_BITS_PER_WORD;
      vid_t i = 0;
      for (; index < bit_last_index; index++, i++) {
        if (bitmap_is_set(word, i)) {
          vid_t vid = inbox->rmt_nbrs[index];
          if (distance[vid] > rmt_distance[index]) {
            distance[vid] = rmt_distance[index];
            bitmap_set_cpu(updated, vid);
          }
        }
      }
    }
  }
}

__global__
void sssp_scatter_kernel(grooves_box_table_t inbox, bitmap_t updated,
                         weight_t* distance) {
  vid_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) { return; }

  // Get the values that have been pushed to this vertex
  bitmap_t rmt_updated = reinterpret_cast<bitmap_t>(inbox.push_values);
  if (!bitmap_is_set(rmt_updated, index)) { return; }
  weight_t* rmt_distance =
      (weight_t*)&rmt_updated[bitmap_bits_to_words(inbox.count)];
  vid_t vid = inbox.rmt_nbrs[index];
  weight_t old_distance = distance[vid];
  distance[vid] = rmt_distance[index] < distance[vid] ?
      rmt_distance[index] : distance[vid];
  weight_t new_distance = distance[vid];
  if (old_distance > new_distance) {
    bitmap_set_gpu(updated, vid);
  }
}

PRIVATE void sssp_scatter_gpu(partition_t* par, grooves_box_table_t* inbox,
                              sssp_state_t* state) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(inbox->count, blocks, threads);
  // Invoke the appropriate CUDA kernel to perform the scatter functionality
  sssp_scatter_kernel<<<blocks, threads, 0, par->streams[1]>>>
      (*inbox, state->updated[par->id], state->distance[par->id]);
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
      sssp_scatter_cpu(inbox, state->updated[par->id],
                       state->distance[par->id]);
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
    src_distance = state->distance[par->id];
  } else if (par->processor.type == PROCESSOR_GPU) {
    assert(state_g.distance_h);
    CALL_CU_SAFE(cudaMemcpy(state_g.distance_h, state->distance[par->id],
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

__global__ void sssp_init_kernel(bitmap_t visited, vid_t src) {
  if (THREAD_GLOBAL_INDEX != 0) return;
  bitmap_set_gpu(visited, src);
}

void sssp_init_gpu(partition_t* par, sssp_state_t* state) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(1, blocks, threads);
  sssp_init_kernel<<<blocks, threads, 0, par->streams[1]>>>
      (state->updated[par->id], GET_VERTEX_ID(state_g.src));
  CALL_CU_SAFE(cudaGetLastError());
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
    frontier_init_cpu(&state->frontier, par->subgraph.vertex_count);
  } else if (par->processor.type == PROCESSOR_GPU) {
    type = TOTEM_MEM_DEVICE;
    frontier_init_gpu(&state->frontier, par->subgraph.vertex_count);
  } else {
    assert(false);
  }

  state->updated[par->id] = state->frontier.current;
  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(weight_t), type,
                        reinterpret_cast<void**>(&(state->distance[par->id]))));
  totem_memset(state->distance[par->id], WEIGHT_MAX, par->subgraph.vertex_count,
               type, par->streams[1]);

  if (GET_PARTITION_ID(state_g.src) == par->id) {
    // For the source vertex, initialize distance.
    // Please note that instead of simply initializing the updated status of
    // the source using the following expression
    // "state->distance[GET_VERTEX_ID(state_g.src)] = 0.0", we are using
    // "memset" becuase the source may belong to a partition which
    // resides on the GPU.
    totem_memset(&((state->distance[par->id])[GET_VERTEX_ID(state_g.src)]),
                 (weight_t)0, 1, type, par->streams[1]);
    if (par->processor.type == PROCESSOR_GPU) {
      sssp_init_gpu(par, state);
    } else {
      bitmap_set_cpu(state->frontier.current, GET_VERTEX_ID(state_g.src));
    }
  }

  for (int pid = 0; pid < engine_partition_count(); pid++) {
    vid_t count = par->outbox[pid].count;
    if (pid != par->id && count != 0) {
      bitmap_t updated = (bitmap_t)par->outbox[pid].push_values;
      state->updated[pid] = updated;
      state->distance[pid] = (weight_t*)&updated[bitmap_bits_to_words(count)];
      if (par->processor.type == PROCESSOR_GPU) {
        bitmap_reset_gpu(updated, count);
      } else {
        bitmap_reset_cpu(updated, count);
      }
      totem_memset(state->distance[pid],
                   WEIGHT_MAX, count, type, par->streams[1]);
    }
  }

  state->finished = engine_get_finished_ptr(par->id);
}

PRIVATE void sssp_finalize(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  sssp_state_t* state = reinterpret_cast<sssp_state_t*>(par->algo_state);
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_CPU) {
    frontier_finalize_cpu(&state->frontier);
  } else if (par->processor.type == PROCESSOR_GPU) {
    frontier_finalize_gpu(&state->frontier);
    type = TOTEM_MEM_DEVICE;
  } else {
    assert(false);
  }
  totem_free(state->distance[par->id], type);
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
