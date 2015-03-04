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

  bitmap_t updated;  // One bit per vertex in the partition
                     // it indicates whether the corresponding vertex has been
                     // updated and that it should try to update the distances
                     // of its neighbours.
  bool* finished;    // Points to Totem's finish flag.
  weight_t* distance;  // Stores results in the partition.
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
    if (!bitmap_is_set(state->updated, v)) { continue; }
    bitmap_unset_cpu(state->updated, v);

    for (eid_t i = subgraph->vertices[v]; i < subgraph->vertices[v + 1]; i++) {
      int nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
      vid_t nbr = GET_VERTEX_ID(subgraph->edges[i]);
      vid_t nbr_dst = subgraph->edges[i];
      weight_t* dst = engine_get_dst_ptr(
          par->id, nbr_dst, par->outbox, state->distance);
      weight_t new_distance = state->distance[v] + subgraph->weights[i];
      weight_t old_distance = *dst;
      if (new_distance < old_distance) {
        if (old_distance == __sync_fetch_and_min_float(dst, new_distance) &&
            nbr_pid == par->id) {
          bitmap_set_cpu(state->updated, nbr);
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

  if (!bitmap_is_set(state.updated, v)) { return; }
  bitmap_unset_gpu(state.updated, v);

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
        bitmap_set_gpu(state.updated, nbr);
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

template<int VWARP_WIDTH, int VWARP_BATCH>
PRIVATE __global__
void sssp_vwarp_kernel(partition_t par, sssp_state_t state) {
  const vid_t vertex_count = par.subgraph.vertex_count;
  if (THREAD_GLOBAL_INDEX >=
      vwarp_thread_count(vertex_count, VWARP_WIDTH, VWARP_BATCH)) { return; }

  const eid_t* __restrict vertices = par.subgraph.vertices;

  __shared__ bool finished_block;
  finished_block = true;
  __syncthreads();

  vid_t start_vertex = vwarp_block_start_vertex(VWARP_WIDTH, VWARP_BATCH) +
    vwarp_warp_start_vertex(VWARP_WIDTH, VWARP_BATCH);
  vid_t end_vertex = start_vertex +
    vwarp_warp_batch_size(vertex_count, VWARP_WIDTH, VWARP_BATCH);
  int warp_offset = vwarp_thread_index(VWARP_WIDTH);
  for (vid_t v = start_vertex; v < end_vertex; v++) {
    if (bitmap_is_set(state.updated, v)) {
      bitmap_unset_gpu(state.updated, v);
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
        vid_t nbr_dst = nbrs[i];
        weight_t* dst = engine_get_dst_ptr(par.id, nbr_dst, par.outbox,
                                           state.distance);
        weight_t new_distance = state.distance[v] + weights[i];
        weight_t old_distance = atomicMin(dst, new_distance);
        if (new_distance < old_distance) {
          if (nbr_pid == par.id) {
            bitmap_set_gpu(state.updated, nbr);
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
  const vid_t vertex_count = par->subgraph.vertex_count;
  const int threads = MAX_THREADS_PER_BLOCK;
  dim3 blocks;
  assert(VWARP_WIDTH <= threads);
  kernel_configure(vwarp_thread_count(vertex_count, VWARP_WIDTH, BATCH_SIZE),
                   blocks, threads);
  sssp_vwarp_kernel<VWARP_WIDTH, BATCH_SIZE>
    <<<blocks, threads, 0, par->streams[1]>>>(*par, *state);
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

PRIVATE void sssp(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  sssp_state_t* state = reinterpret_cast<sssp_state_t*>(par->algo_state);
  if (par->processor.type == PROCESSOR_CPU) {
    sssp_cpu(par, state);
  } else if (par->processor.type == PROCESSOR_GPU) {
    sssp_vwarp_gpu(par, state);
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
    if (old_distance > new_distance) {
      bitmap_set_cpu(state->updated, vid);
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
    bitmap_set_gpu(state.updated, vid);
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

// A simple kernel that sets the source vertex to visited on the GPU.
__global__ void sssp_init_source_kernel(bitmap_t updated, vid_t src) {
  if (THREAD_GLOBAL_INDEX != 0) { return; }
  bitmap_set_gpu(updated, src);
}

PRIVATE void sssp_init(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  sssp_state_t* state = reinterpret_cast<sssp_state_t*>
    (calloc(1, sizeof(sssp_state_t)));
  assert(state);
  par->algo_state = state;

  totem_mem_t type;
  if (par->processor.type == PROCESSOR_CPU) {
    state->updated = bitmap_init_cpu(par->subgraph.vertex_count);
     type = TOTEM_MEM_HOST;
  } else if (par->processor.type == PROCESSOR_GPU) {
    state->updated = bitmap_init_gpu(par->subgraph.vertex_count);
    type = TOTEM_MEM_DEVICE;
  } else {
    assert(false);
  }

  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(bool), type,
                         reinterpret_cast<void**>(&(state->updated))));
  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(weight_t), type,
                         reinterpret_cast<void**>(&(state->distance))));
  totem_memset(state->distance, WEIGHT_MAX, par->subgraph.vertex_count, type,
               par->streams[1]);

  if (GET_PARTITION_ID(state_g.src) == par->id) {
    // For the source vertex, initialize updated status.
    if (par->processor.type == PROCESSOR_GPU) {
      sssp_init_source_kernel<<<1, 1, 0, par->streams[1]>>>
          (state->updated, GET_VERTEX_ID(state_g.src));
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      bitmap_set_cpu(state->updated, GET_VERTEX_ID(state_g.src));
    }

    // For the source vertex, initialize distance.
    totem_memset(&((state->distance)[GET_VERTEX_ID(state_g.src)]),
                 (weight_t)0.0, 1, type, par->streams[1]);
  }

  state->finished = engine_get_finished_ptr(par->id);
  engine_set_outbox(par->id, WEIGHT_MAX);
}

PRIVATE void sssp_finalize(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  sssp_state_t* state = reinterpret_cast<sssp_state_t*>(par->algo_state);
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_CPU) {
    bitmap_finalize_cpu(state->updated);
  } else if (par->processor.type == PROCESSOR_GPU) {
    bitmap_finalize_gpu(state->updated);
    type = TOTEM_MEM_DEVICE;
  } else {
    assert(false);
  }
  // totem_free(state->updated, type);
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
