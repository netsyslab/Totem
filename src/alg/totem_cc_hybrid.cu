/**
 * This file contains an implementation of the single source shortest path
 * (SSSP) algorithm using the totem framework.
 *
 *  Created on: 2014-05-10
 *  Author: Abdullah Gharaibeh
 */

#include "totem_alg.h"
#include "totem_engine.cuh"

// per-partition specific state
typedef struct cc_state_s {
  bool* finished;                           // points to Totem's finish flag
  vid_t* label[MAX_PARTITION_COUNT];        // stores results in the partition
  bitmap_t updated[MAX_PARTITION_COUNT];    // a list of bitmaps one for each
                                            // remote partition
  frontier_state_t frontier;
} cc_state_t;


// state shared between all partitions
typedef struct cc_global_state_s {
  vid_t* label;  // stores the final results
  vid_t* label_h;  // temporary buffer for GPU
} cc_global_state_t;

PRIVATE cc_global_state_t state_g = {0};


// Checks for input parameters and special cases. This is invoked at the
// beginning of public interfaces (GPU and CPU)
PRIVATE error_t check_special_cases(vid_t* label, bool* finished) {
  *finished = true;
  if (label == NULL) {
    return FAILURE;
  } else if (engine_vertex_count() == 1) {
    label[0] = 0;
    return SUCCESS;
  } else if (engine_edge_count() == 0) {
    // Initialize label
    for (vid_t v = 0; v < engine_vertex_count(); v++) {
      label[v] = v;
    }
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

void cc_cpu(partition_t* par, cc_state_t* state) {
  graph_t* subgraph = &par->subgraph;
  bool finished = true;

  vid_t* label = state->label[par->id];

  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    if (!bitmap_is_set(state->updated[par->id], v)) { continue; }
    bitmap_unset_cpu(state->updated[par->id], v);

    for (eid_t i = subgraph->vertices[v]; i < subgraph->vertices[v + 1]; i++) {
      int nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
      vid_t nbr = GET_VERTEX_ID(subgraph->edges[i]);
      vid_t* nbr_label = state->label[nbr_pid];
      bitmap_t nbr_updated = state->updated[nbr_pid];
      vid_t old_label = nbr_label[nbr];
      vid_t new_label = label[v];
      if (new_label < old_label) {
        if (old_label ==
            __sync_fetch_and_min_uint32(&nbr_label[nbr], new_label)) {
          bitmap_set_cpu(nbr_updated, nbr);
        }
        finished = false;
      }
    }
  }
  if (!finished) *(state->finished) = false;
}

__global__
void cc_kernel(partition_t par, cc_state_t state, vid_t vertex_count) {
  vid_t v = THREAD_GLOBAL_INDEX;
  if (v >= vertex_count) { return; }
  v = state.frontier.list[v];

  __shared__ bool finished_block;
  finished_block = true;
  __syncthreads();

  vid_t* label = state.label[par.id];

  if (bitmap_is_set(state.updated[par.id], v)) {
    bitmap_unset_gpu(state.updated[par.id], v);
    for (eid_t i = par.subgraph.vertices[v]; i < par.subgraph.vertices[v + 1];
         i++) {
      int nbr_pid = GET_PARTITION_ID(par.subgraph.edges[i]);
      vid_t nbr = GET_VERTEX_ID(par.subgraph.edges[i]);
      vid_t* nbr_label = state.label[nbr_pid];
      bitmap_t nbr_updated = state.updated[nbr_pid];
      vid_t old_label = nbr_label[nbr];
      vid_t new_label = label[v];
      if (new_label < old_label) {
        if ((old_label == atomicMin(&nbr_label[nbr], new_label))) {
          bitmap_set_gpu(nbr_updated, nbr);
        }
        finished_block = false;
      }
    }
  }

  __syncthreads();
  if (!finished_block && THREAD_BLOCK_INDEX == 0) *state.finished = false;
}

PRIVATE void cc_gpu(partition_t* par, cc_state_t* state) {
  vid_t vertex_count = frontier_count_gpu(&state->frontier, par->streams[1]);
  if (vertex_count == 0) { return; }
  assert(state->frontier.list_len >= vertex_count);
  frontier_update_list_gpu(&state->frontier, par->streams[1]);
  dim3 blocks, threads;
  KERNEL_CONFIGURE(vertex_count, blocks, threads);
  cc_kernel<<<blocks, threads, 1, par->streams[1]>>>(
      *par, *state, vertex_count);
  CALL_CU_SAFE(cudaGetLastError());
}

template<int VWARP_WIDTH, int VWARP_BATCH>
PRIVATE __global__ void cc_vwarp_kernel(partition_t par, cc_state_t state,
                                        vid_t vertex_count) {
  if (THREAD_GLOBAL_INDEX >=
      vwarp_thread_count(vertex_count, VWARP_WIDTH, VWARP_BATCH)) { return; }

  const eid_t* __restrict vertices = par.subgraph.vertices;

  __shared__ bool finished_block;
  finished_block = true;
  __syncthreads();

  vid_t* label = state.label[par.id];
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
      if (v >= par.subgraph.vertex_ext) {
        nbrs = par.subgraph.edges_ext +
          (vertices[v] - par.subgraph.edge_count_ext);
      }
      for (vid_t i = warp_offset; i < nbr_count; i += VWARP_WIDTH) {
        int nbr_pid = GET_PARTITION_ID(nbrs[i]);
        vid_t nbr = GET_VERTEX_ID(nbrs[i]);
        vid_t* nbr_label = state.label[nbr_pid];
        bitmap_t nbr_updated   = state.updated[nbr_pid];
        vid_t old_label = nbr_label[nbr];
        vid_t new_label = label[v];
        if (new_label < old_label) {
          if (old_label == atomicMin(&nbr_label[nbr], new_label)) {
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
PRIVATE void cc_gpu_launch(partition_t* par, cc_state_t* state) {
  vid_t vertex_count = frontier_count_gpu(&state->frontier, par->streams[1]);
  if (vertex_count == 0) { engine_report_no_comm(par->id); return; }
  assert(state->frontier.list_len >= vertex_count);
  frontier_update_list_gpu(&state->frontier, par->streams[1]);
  const int threads = MAX_THREADS_PER_BLOCK;
  dim3 blocks;
  assert(VWARP_WIDTH <= threads);
  kernel_configure(vwarp_thread_count(vertex_count, VWARP_WIDTH, BATCH_SIZE),
                   blocks, threads);
  cc_vwarp_kernel<VWARP_WIDTH, BATCH_SIZE>
      <<<blocks, threads, 0, par->streams[1]>>>(*par, *state, vertex_count);
}

typedef void(*cc_gpu_func_t)(partition_t*, cc_state_t*);

PRIVATE void cc_vwarp_gpu(partition_t* par, cc_state_t* state) {
  PRIVATE const cc_gpu_func_t CC_GPU_FUNC[] = {
    // RANDOM algorithm
    cc_gpu_launch<VWARP_MEDIUM_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE>,
    // HIGH partitioning
    cc_gpu_launch<VWARP_MEDIUM_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE>,
    // LOW partitioning
    // Note that it is not possible to use a virtual warp width longer than the
    // hardware warp width. This is because a vertex may switch from inactive
    // to active state (maintained by the updated array) during the execution
    // of a round. This may lead to the situation where the threads of a
    // virtual warp, which are all supposed to process the neighbours of a
    // vertex, evaluate the vertex's active state differently, and hence part
    // of the neighbours of that vertex will not get processed.
    cc_gpu_launch<VWARP_HARDWARE_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE>
  };
  int par_alg = engine_partition_algorithm();
  CC_GPU_FUNC[par_alg](par, state);
}

PRIVATE void reset_rmt_updated(partition_t* par, cc_state_t* state) {
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

PRIVATE void cc(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  cc_state_t* state = reinterpret_cast<cc_state_t*>(par->algo_state);
  reset_rmt_updated(par, state);
  if (par->processor.type == PROCESSOR_CPU) {
    cc_cpu(par, state);
  } else if (par->processor.type == PROCESSOR_GPU) {
    cc_vwarp_gpu(par, state);
  } else {
    assert(false);
  }
}

PRIVATE void cc_scatter_cpu(grooves_box_table_t* inbox, bitmap_t updated,
                            vid_t* label) {
  bitmap_t  rmt_updated = reinterpret_cast<bitmap_t>(inbox->push_values);
  vid_t* rmt_label =
      (vid_t*)&rmt_updated[bitmap_bits_to_words(inbox->count)];
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
          if (label[vid] > rmt_label[index]) {
            label[vid] = rmt_label[index];
            bitmap_set_cpu(updated, vid);
          }
        }
      }
    }
  }
}

__global__
void cc_scatter_kernel(grooves_box_table_t inbox, bitmap_t updated,
                       vid_t* label) {
  vid_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) { return; }

  // Get the values that have been pushed to this vertex
  bitmap_t rmt_updated = reinterpret_cast<bitmap_t>(inbox.push_values);
  if (!bitmap_is_set(rmt_updated, index)) { return; }
  vid_t* rmt_label =
      (vid_t*)&rmt_updated[bitmap_bits_to_words(inbox.count)];
  vid_t vid = inbox.rmt_nbrs[index];
  vid_t old_label = label[vid];
  label[vid] = rmt_label[index] < label[vid] ? rmt_label[index] : label[vid];
  vid_t new_label = label[vid];
  if (old_label > new_label) {
    bitmap_set_gpu(updated, vid);
  }
}

PRIVATE void cc_scatter_gpu(partition_t* par, grooves_box_table_t* inbox,
                              cc_state_t* state) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(inbox->count, blocks, threads);
  // Invoke the appropriate CUDA kernel to perform the scatter functionality
  cc_scatter_kernel<<<blocks, threads, 0, par->streams[1]>>>
      (*inbox, state->updated[par->id], state->label[par->id]);
  CALL_CU_SAFE(cudaGetLastError());
}

PRIVATE void cc_scatter(partition_t* par) {
  // Check if there is no work to be done
  if (par->subgraph.vertex_count == 0) { return; }
  // Get the current state of the algorithm
  cc_state_t* state = reinterpret_cast<cc_state_t*>(par->algo_state);

  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;
    // For all remote partitions, get the corresponding inbox
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    // If the inbox has some values, determine which type of processing unit
    // corresponds to this partition and call the appropriate scatter function
    if (par->processor.type == PROCESSOR_CPU) {
      cc_scatter_cpu(inbox, state->updated[par->id],
                       state->label[par->id]);
    } else if (par->processor.type == PROCESSOR_GPU) {
      cc_scatter_gpu(par, inbox, state);
    } else {
      assert(false);
    }
  }
}

PRIVATE void cc_aggregate(partition_t* par) {
  if (!par->subgraph.vertex_count) { return; }
  cc_state_t* state = reinterpret_cast<cc_state_t*>(par->algo_state);
  graph_t* subgraph = &par->subgraph;
  vid_t* src_label = NULL;
  if (par->processor.type == PROCESSOR_CPU) {
    src_label = state->label[par->id];
  } else if (par->processor.type == PROCESSOR_GPU) {
    assert(state_g.label_h);
    CALL_CU_SAFE(cudaMemcpy(state_g.label_h, state->label[par->id],
                            subgraph->vertex_count * sizeof(vid_t),
                            cudaMemcpyDefault));
    src_label = state_g.label_h;
  } else {
    assert(false);
  }
  assert(state_g.label);
  OMP(omp parallel for schedule(static))
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    state_g.label[par->map[v]] = src_label[v];
  }
}

PRIVATE inline void bitmap_set_cpu1(bitmap_t map, size_t len) {
  memset(map, 0xFF, bitmap_bits_to_bytes(len));
  vid_t last_word_index = bitmap_bits_to_words(len) - 1;
  bitmap_word_t last_word = 0;
  if (len % BITMAP_BITS_PER_WORD) {
    last_word = ((bitmap_word_t)-1) >>
        (BITMAP_BITS_PER_WORD - (len % BITMAP_BITS_PER_WORD));
  }
  map[last_word_index] = last_word;
}

PRIVATE inline void bitmap_set_gpu1(bitmap_t map, size_t len,
                                    cudaStream_t stream = 0) {
  CALL_CU_SAFE(cudaMemsetAsync(map, 0xFF, bitmap_bits_to_bytes(len), stream));
  vid_t last_word_index = bitmap_bits_to_words(len) - 1;
  bitmap_word_t last_word = 0;
  if (len % BITMAP_BITS_PER_WORD) {
    last_word = ((bitmap_word_t)-1) >>
        (BITMAP_BITS_PER_WORD - (len % BITMAP_BITS_PER_WORD));
  }
  totem_memset(&map[last_word_index], last_word, 1, TOTEM_MEM_DEVICE);
}

PRIVATE void cc_init(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  cc_state_t* state = reinterpret_cast<cc_state_t*>
    (calloc(1, sizeof(cc_state_t)));
  assert(state);
  par->algo_state = state;

  totem_mem_t type;
  if (par->processor.type == PROCESSOR_CPU) {
    type = TOTEM_MEM_HOST;
    frontier_init_cpu(&state->frontier, par->subgraph.vertex_count);
    bitmap_set_cpu1(state->frontier.current, par->subgraph.vertex_count);
    assert(
        bitmap_count_cpu(state->frontier.current, par->subgraph.vertex_count) ==
        par->subgraph.vertex_count);
  } else if (par->processor.type == PROCESSOR_GPU) {
    type = TOTEM_MEM_DEVICE;
    frontier_init_gpu(&state->frontier, par->subgraph.vertex_count);
    bitmap_set_gpu1(state->frontier.current, par->subgraph.vertex_count);
    assert(
        bitmap_count_gpu(state->frontier.current, par->subgraph.vertex_count) ==
        par->subgraph.vertex_count);
  } else {
    assert(false);
  }

  state->updated[par->id] = state->frontier.current;
  CALL_SAFE(totem_malloc(par->subgraph.vertex_count * sizeof(vid_t), type,
                        reinterpret_cast<void**>(&(state->label[par->id]))));
  cudaMemcpy(state->label[par->id], par->map,
             par->subgraph.vertex_count * sizeof(vid_t),
             cudaMemcpyDefault);

  for (int pid = 0; pid < engine_partition_count(); pid++) {
    vid_t count = par->outbox[pid].count;
    if (pid != par->id && count != 0) {
      bitmap_t updated = (bitmap_t)par->outbox[pid].push_values;
      state->updated[pid] = updated;
      state->label[pid] = (vid_t*)&updated[bitmap_bits_to_words(count)];
      if (par->processor.type == PROCESSOR_GPU) {
        bitmap_reset_gpu(updated, count);
      } else {
        bitmap_reset_cpu(updated, count);
      }
      totem_memset(state->label[pid], VERTEX_ID_MAX, count, type,
                   par->streams[1]);
    }
  }

  state->finished = engine_get_finished_ptr(par->id);
}

PRIVATE void cc_finalize(partition_t* par) {
  if (par->subgraph.vertex_count == 0) { return; }
  cc_state_t* state = reinterpret_cast<cc_state_t*>(par->algo_state);
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_CPU) {
    frontier_finalize_cpu(&state->frontier);
  } else if (par->processor.type == PROCESSOR_GPU) {
    frontier_finalize_gpu(&state->frontier);
    type = TOTEM_MEM_DEVICE;
  } else {
    assert(false);
  }
  totem_free(state->label[par->id], type);
  free(state);
  par->algo_state = NULL;
}

error_t cc_hybrid(vid_t* label) {
  // check for special cases
  bool finished = false;

  error_t rc = check_special_cases(label, &finished);
  if (finished) return rc;

  // initialize the global state
  state_g.label = label;

  // initialize the engine
  engine_config_t config = {
    NULL, cc, cc_scatter, NULL, cc_init, cc_finalize, cc_aggregate,
    GROOVES_PUSH
  };
  engine_config(&config);
  if (engine_largest_gpu_partition()) {
    CALL_SAFE(totem_malloc(engine_largest_gpu_partition() * sizeof(vid_t),
                           TOTEM_MEM_HOST,
                           reinterpret_cast<void**>(&state_g.label_h)));
  }
  engine_execute();

  // clean up and return
  if (engine_largest_gpu_partition()) {
    totem_free(state_g.label_h, TOTEM_MEM_HOST);
  }
  memset(&state_g, 0, sizeof(cc_global_state_t));
  return SUCCESS;
}
