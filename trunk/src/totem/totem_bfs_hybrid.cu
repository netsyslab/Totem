/**
 * This file contains an implementation of the breadth-first search (BFS) graph
 * search algorithm using the totem framework
 *
 *  Created on: 2012-01-30
 *  Author: Abdullah Gharaibeh
 */

#include "totem_engine.cuh"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

/**
 * per-partition specific state
 */
typedef struct bfs_state_s {
  uint32_t* cost;     // one slot per vertex in the partition
  bool*     finished; // points to the global finish flag
  int       level;    // current level being processed by the partition
  dim3      blocks;   // kernel configuration parameters
  dim3      threads;
} bfs_state_t;

/**
 * Stores the final result
 */
uint32_t* cost_g     = NULL;

/**
 * Used as a temporary buffer to host the final result produced by
 * GPU partitions
 */
uint32_t* cost_h     = NULL;

/**
 * A global finish flag. This flag is set to true at the beginning of each
 * superstep. Any partition that still has vertices to process sets this
 * flag to false. There is no need to synchronize access to it since
 * there is only one possible write value (false).
 */
bool* finished_g = NULL;

/**
 * Source vertex
 */
id_t src_g;

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
*/
PRIVATE error_t check_special_cases(graph_t* graph, id_t src, uint32_t** cost,
                                    bool* finished) {
  *finished = true;
  if((graph == NULL) || (src >= graph->vertex_count)) {
    *cost = NULL;
    return FAILURE;
  } else if(graph->vertex_count == 1) {
    *cost = (uint32_t*)mem_alloc(sizeof(uint32_t));
    *cost[0] = 0;
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

/**
   This structure is used by the virtual warp-based implementation. It stores a
   batch of work. It is typically allocated on shared memory and is processed by
   a single virtual warp.
 */
typedef struct {
  uint32_t cost[VWARP_BATCH_SIZE];
  id_t vertices[VWARP_BATCH_SIZE + 1];
  // the following ensures 64-bit alignment, it assumes that the
  // cost and vertices arrays are of 32-bit elements.
  // TODO(abdullah) a portable way to do this (what if id_t is 64-bit?)
  int pad;
} vwarp_mem_t;

/**
 * A warp-based implementation of the BFS kernel. Please refer to the
 * description of the warp technique for details. Also, please refer to
 * bfs_kernel for details on the BFS implementation.
 */
__global__
void vwarp_bfs_kernel(partition_t par, int pc, uint32_t level, bool* finished,
                      uint32_t* cost, int thread_count) {
  if (THREAD_GLOBAL_INDEX >= thread_count) return;
  int warp_offset = THREAD_GLOBAL_INDEX % VWARP_WARP_SIZE;
  int warp_id     = THREAD_GLOBAL_INDEX / VWARP_WARP_SIZE;

  // This flag is used to report the finish state of a block of threads. This
  // is useful to avoid having many threads writing to the global finished
  // flag, which can hurt performance (since "finished" is actually allocated
  // on the host, and each write will cause a transfer over the PCI-E bus)
  __shared__ bool finished_block;
  finished_block = true;
  __syncthreads();

  // copy my work to local space
  __shared__ vwarp_mem_t smem[(MAX_THREADS_PER_BLOCK / VWARP_WARP_SIZE)];
  vwarp_mem_t* my_space = smem + (THREAD_GRID_INDEX / VWARP_WARP_SIZE);
  int v = warp_id * VWARP_BATCH_SIZE;
  int batch_size = (v + VWARP_BATCH_SIZE) > par.subgraph.vertex_count ?
    (par.subgraph.vertex_count - v) : VWARP_BATCH_SIZE;
  vwarp_memcpy(my_space->cost, &cost[v], batch_size, warp_offset);
  vwarp_memcpy(my_space->vertices, &(par.subgraph.vertices[v]),
               batch_size + 1, warp_offset);

  // iterate over my work
  for(uint32_t v = 0; v < batch_size; v++) {
    if (my_space->cost[v] == level) {
      int nbr_count = my_space->vertices[v + 1] - my_space->vertices[v];
      id_t* nbrs = &(par.subgraph.edges[my_space->vertices[v]]);
      for(int i = warp_offset; i < nbr_count; i += VWARP_WARP_SIZE) {
        uint32_t* dst; const id_t nbr = nbrs[i];
        ENGINE_FETCH_DST(par.id, nbr, par.outbox_d, cost, pc, dst, uint32_t);
        if (*dst == INFINITE) {
          *dst = level + 1;
          finished_block = false;
        }
      }
    }
  }
  __syncthreads();
  if (!finished_block && threadIdx.x == 0) *finished = false;
}

__global__
void bfs_kernel(partition_t par, int pc, uint32_t level,
                bool* finished, uint32_t* cost) {
  const int v = THREAD_GLOBAL_INDEX;
  if (v >= par.subgraph.vertex_count || cost[v] != level) return;

  for (id_t i = par.subgraph.vertices[v];
       i < par.subgraph.vertices[v + 1]; i++) {
    uint32_t* dst; id_t nbr = par.subgraph.edges[i];

    ENGINE_FETCH_DST(par.id, nbr, par.outbox_d, cost,
                     pc, dst, uint32_t);
    if (*dst == INFINITE) {
      // Threads may update finished and the same position in the cost array
      // concurrently. It does not affect correctness since all
      // threads would update with the same value.
      *finished = false;
      *dst  = level + 1;
    }
  }
}

PRIVATE void bfs_gpu(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  vwarp_bfs_kernel<<<state->blocks, state->threads, 0,
    par->streams[1]>>>(*par, engine_partition_count(), state->level,
                       state->finished, state->cost,
                       VWARP_BATCH_COUNT(par->subgraph.vertex_count) *
                       VWARP_WARP_SIZE);
  CALL_CU_SAFE(cudaGetLastError());
}

PRIVATE void bfs_cpu(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  graph_t* subgraph = &par->subgraph;
  int pc = engine_partition_count();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (id_t v = 0; v < subgraph->vertex_count; v++) {
    if (state->cost[v] != state->level) continue;
    for (id_t i = subgraph->vertices[v]; i < subgraph->vertices[v + 1]; i++) {
      uint32_t* dst; id_t nbr = subgraph->edges[i];
      ENGINE_FETCH_DST(par->id, nbr, par->outbox, state->cost, pc,
                       dst, uint32_t);
      if (*dst == INFINITE) {
        *dst = state->level + 1;
        *(state->finished) = false;
      }
    }
  }
}

PRIVATE void bfs(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  if (par->processor.type == PROCESSOR_GPU) {
    bfs_gpu(par);
  } else {
    assert(par->processor.type == PROCESSOR_CPU);
    bfs_cpu(par);
  }
  state->level++;
}

PRIVATE void bfs_ss() {
  // This is invoked once at the very beginning of a superstep
  *finished_g = true;
}

PRIVATE void bfs_scatter(partition_t* par) {
  // this callback function is invoked at the end of a superstep. i.e., it is
  // guaranteed at this point that all kernels has finished execution (remember
  // that GPU kernels are inovked asynchronously, and messages to remote
  // vertices has been communicated to the inboxes.
  // The most important point here is that only at this stage it is guaranteed
  // that, if a kernel has set the shared flag "finished_g" to false, this write
  // has been propagated to the host memory.
  if (*finished_g) {
    engine_report_finished(par->id);
    return;
  }
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  engine_scatter_inbox_min(par->id, state->cost);
}

PRIVATE void bfs_aggregate(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  graph_t* subgraph = &par->subgraph;
  uint32_t* src_cost = NULL;
  if (par->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaMemcpy(cost_h, state->cost,
                            subgraph->vertex_count * sizeof(uint32_t),
                            cudaMemcpyDefault));
    src_cost = cost_h;
  } else {
    assert(par->processor.type == PROCESSOR_CPU);
    src_cost = state->cost;
  }
  // aggregate the results
  for (id_t v = 0; v < subgraph->vertex_count; v++) {
    cost_g[par->map[v]] = src_cost[v];
  }
}

PRIVATE void bfs_init(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  bfs_state_t* state = (bfs_state_t*)calloc(1, sizeof(bfs_state_t));
  assert(state);
  id_t src_pid = GET_PARTITION_ID(engine_vertex_id_in_partition(src_g));
  id_t src_vid = GET_VERTEX_ID(engine_vertex_id_in_partition(src_g));
  uint64_t vcount = par->subgraph.vertex_count;
  if (par->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaMalloc((void**)&(state->cost), vcount * sizeof(uint32_t)));
    CALL_CU_SAFE(cudaHostGetDevicePointer((void **)&(state->finished),
                                          (void *)finished_g, 0));
    KERNEL_CONFIGURE(vcount, state->blocks, state->threads);
    memset_device<<<state->blocks, state->threads, 0,
      par->streams[1]>>>(state->cost, INFINITE, vcount);
    CALL_CU_SAFE(cudaGetLastError());
    if (src_pid == par->id) {
      // For the source vertex, initialize cost.
      memset_device<<<state->blocks, state->threads, 0, par->streams[1]>>>
        (&((state->cost)[src_vid]), (uint32_t)0, 1);
      CALL_CU_SAFE(cudaGetLastError());
    }
    KERNEL_CONFIGURE(VWARP_WARP_SIZE * VWARP_BATCH_COUNT(vcount),
                     state->blocks, state->threads);
  } else {
    assert(par->processor.type == PROCESSOR_CPU);
    state->cost = (uint32_t*)calloc(vcount, sizeof(uint32_t));
    state->finished = finished_g;
    assert(state->cost);
    for (id_t v = 0; v < vcount; v++) state->cost[v] = INFINITE;
    if (src_pid == par->id) {
      state->cost[src_vid] = 0;
    }
  }
  engine_set_outbox(par->id, (uint32_t)INFINITE);
  state->level = 0;
  par->algo_state = state;
}

PRIVATE void bfs_finalize(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  if (par->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaFree(state->cost));
  } else {
    assert(par->processor.type == PROCESSOR_CPU);
    free(state->cost);
  }
  free(state);
  par->algo_state = NULL;
}

// TODO(abdullah): Add partitioning algorithm as an input parameter
error_t bfs_hybrid(graph_t* graph, totem_attr_t* attr,
                   id_t src, uint32_t** cost) {
  // check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, src, cost, &finished);
  if (finished) return rc;

  cost_g = (uint32_t*)mem_alloc(graph->vertex_count * sizeof(uint32_t));
  // The global finish flag is allocated on the host using the
  // cudaHostAllocMapped option which allows GPU kernels to access it directly
  // from within the GPU. This flag is initialized to true in the algo_ss_kernel
  // invoked at the beginning of each superstep (before the per-partition kernel
  // callback. Any of the processors (partitions) set this flag to false if it
  // still has work to do.
  CALL_CU_SAFE(cudaHostAlloc((void **)&finished_g, sizeof(bool),
                             cudaHostAllocPortable | cudaHostAllocMapped));

  // initialize the engine
  engine_config_t config = {
    graph,
    attr->platform,
    attr->par_algo,
    attr->cpu_par_share,
    sizeof(uint32_t),
    bfs_ss,
    bfs,
    bfs_scatter,
    bfs_init,
    bfs_finalize,
    bfs_aggregate
  };
  src_g = src;
  engine_init(&config);
  if (engine_largest_gpu_partition()) {
    cost_h = (uint32_t*)mem_alloc(engine_largest_gpu_partition() *
                                  sizeof(uint32_t));
  }
  engine_execute();

  // clean up and return
  *cost = cost_g;
  if (engine_largest_gpu_partition()) mem_free(cost_h);
  CALL_CU_SAFE(cudaFreeHost(finished_g));
  cost_g = NULL; cost_h = NULL; finished_g = NULL;
  return SUCCESS;
}
