/**
 * This file contains an implementation of the breadth-first search (BFS) graph
 * search algorithm using the totem framework
 *
 *  Created on: 2012-01-30
 *  Author: Abdullah Gharaibeh
 */

#include "totem_bitmap.cuh"
#include "totem_engine.cuh"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

/**
 * per-partition specific state
 */
typedef struct bfs_state_s {
  uint32_t* cost;     // one slot per vertex in the partition
  bitmap_t* visited;  // a list of bitmaps, one for each remote partition
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
 * Source vertex partition and local vertex id
 */
id_t src_vid_g;
int  src_pid_g;

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
*/
PRIVATE error_t check_special_cases(id_t src, uint32_t** cost, bool* finished) {
  *finished = true;
  if(src >= engine_vertex_count()) {
    *cost = NULL;
    return FAILURE;
  } else if(engine_vertex_count() == 1) {
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
void bfs_kernel(partition_t par, uint32_t level, bool* finished, 
                uint32_t* cost, bitmap_t* visited_arr, int thread_count) {
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
        int nbr_pid = GET_PARTITION_ID(nbrs[i]);
        id_t nbr = GET_VERTEX_ID(nbrs[i]);
        bitmap_t visited = visited_arr[nbr_pid];
        if (!bitmap_is_set(visited, nbr)) {
          if (bitmap_set_gpu(visited, nbr)) {
            if (nbr_pid == par.id) {
              if (cost[nbr] == INFINITE) cost[nbr] = level + 1;
            }
            finished_block = false;
          }
        }
      }
    }
  }
  __syncthreads();
  if (!finished_block && threadIdx.x == 0) *finished = false;
}

PRIVATE inline void bfs_gpu(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  bfs_kernel<<<state->blocks, state->threads, 0, 
    par->streams[1]>>>(*par, state->level, state->finished, state->cost, 
                       state->visited, 
                       VWARP_BATCH_COUNT(par->subgraph.vertex_count) *
                       VWARP_WARP_SIZE);
  CALL_CU_SAFE(cudaGetLastError());
}

PRIVATE inline void bfs_cpu(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  graph_t* subgraph = &par->subgraph;
  bool finished = true;

  OMP(omp parallel for reduction(& : finished))
  for (id_t v = 0; v < subgraph->vertex_count; v++) {
    if (state->cost[v] != state->level) continue;
    for (id_t i = subgraph->vertices[v]; i < subgraph->vertices[v + 1]; i++) {
      int nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
      id_t nbr = GET_VERTEX_ID(subgraph->edges[i]);
      bitmap_t visited = state->visited[nbr_pid];
      if (!bitmap_is_set(visited, nbr)) {
        if (bitmap_set_cpu(visited, nbr)) {
          if (nbr_pid == par->id) {
            state->cost[nbr] = state->level + 1;
          }
          finished = false;
        }
      }
    }
  }
  if (!finished) *(state->finished) = false;
}

PRIVATE void bfs(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  if (par->processor.type == PROCESSOR_CPU) {
    bfs_cpu(par);
  } else if (par->processor.type == PROCESSOR_GPU) {
    bfs_gpu(par);
  } else {
    assert(false);
  }
  state->level++;
}

PRIVATE void bfs_ss() {
  // This is invoked once at the very beginning of a superstep
  *finished_g = true;
}

PRIVATE inline void bfs_scatter_cpu(grooves_box_table_t* inbox, 
                                    bfs_state_t* state, bitmap_t* visited) {
  bitmap_t remotely_visited = (bitmap_t)inbox->values;
  OMP(omp parallel for)
  for (int index = 0; index < inbox->count; index++) {
    id_t vid = inbox->rmt_nbrs[index];
    if (bitmap_is_set(remotely_visited, index) &&
        !bitmap_is_set(*visited, vid)) {
      assert(bitmap_set_cpu(*visited, vid));
      state->cost[vid] = state->level;
    }
  }
}

__global__ void bfs_scatter_kernel(grooves_box_table_t inbox, uint32_t* cost, 
                                   uint32_t level, bitmap_t* visited) {
  uint32_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) return;
  bitmap_t rmt_visited = (bitmap_t)inbox.values;
  id_t vid = inbox.rmt_nbrs[index];
    if (bitmap_is_set(rmt_visited, index) &&
        !bitmap_is_set(*visited, vid)) {
      bitmap_set_gpu(*visited, vid);
      cost[vid] = level;
    }
}

PRIVATE inline void bfs_scatter_gpu(grooves_box_table_t* inbox, 
                                    bfs_state_t* state, bitmap_t* visited) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(inbox->count, blocks, threads);
  bfs_scatter_kernel<<<blocks, threads>>>(*inbox, state->cost, state->level, 
                                          visited);
  CALL_CU_SAFE(cudaGetLastError());
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
  for (int rmt_pid = 0; rmt_pid < engine_partition_count(); rmt_pid++) {
    if (rmt_pid == par->id) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_CPU) {
      bfs_scatter_cpu(inbox, state, &(state->visited[par->id]));
    } else if (par->processor.type == PROCESSOR_GPU) {
      bfs_scatter_gpu(inbox, state, &(state->visited[par->id]));
    } else {
      assert(false);
    }
  }
}

PRIVATE void bfs_aggregate(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  graph_t* subgraph = &par->subgraph;
  uint32_t* src_cost = NULL;
  if (par->processor.type == PROCESSOR_CPU) {
    src_cost = state->cost;
  } else if (par->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaMemcpy(cost_h, state->cost, 
                            subgraph->vertex_count * sizeof(uint32_t),
                            cudaMemcpyDefault));
    src_cost = cost_h;
  } else {
    assert(false);
  }
  // aggregate the results
  OMP(omp parallel for)
  for (id_t v = 0; v < subgraph->vertex_count; v++) {
    cost_g[par->map[v]] = src_cost[v];
  }
}

__global__ void bfs_init_kernel(bitmap_t visited, id_t src) {
  if (THREAD_GLOBAL_INDEX != 0) return;
  bitmap_set_gpu(visited, src);
}

PRIVATE inline void bfs_init_gpu(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  uint64_t vcount = par->subgraph.vertex_count;
  CALL_CU_SAFE(cudaMalloc((void**)&(state->cost), vcount * sizeof(uint32_t)));
  CALL_CU_SAFE(cudaMalloc((void**)&(state->visited), 
                          MAX_PARTITION_COUNT * sizeof(bitmap_t)));
  bitmap_t visited[MAX_PARTITION_COUNT];
  visited[par->id] = bitmap_init_gpu(vcount);
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      visited[pid] = (bitmap_t)par->outbox[pid].values;
      bitmap_reset_gpu(visited[pid], par->outbox[pid].count);
    }
  }
  CALL_CU_SAFE(cudaMemcpy(state->visited, visited, 
                          MAX_PARTITION_COUNT * sizeof(bitmap_t),
                          cudaMemcpyDefault));
  KERNEL_CONFIGURE(vcount, state->blocks, state->threads);
  memset_device<<<state->blocks, state->threads, 0,
    par->streams[1]>>>(state->cost, INFINITE, vcount);
  CALL_CU_SAFE(cudaGetLastError());
  if (src_pid_g == par->id) {
    // For the source vertex, initialize cost.
    memset_device<<<state->blocks, state->threads, 0, par->streams[1]>>>
      (&((state->cost)[src_vid_g]), (uint32_t)0, 1);
    bfs_init_kernel<<<state->blocks, state->threads, 0, par->streams[1]>>>
      (visited[par->id], src_vid_g);
    CALL_CU_SAFE(cudaGetLastError());
  }
  CALL_CU_SAFE(cudaHostGetDevicePointer((void **)&(state->finished), 
                                        (void *)finished_g, 0));
  KERNEL_CONFIGURE(VWARP_WARP_SIZE * VWARP_BATCH_COUNT(vcount),
                   state->blocks, state->threads);
}

PRIVATE inline void bfs_init_cpu(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  state->cost = (uint32_t*)calloc(par->subgraph.vertex_count, sizeof(uint32_t));
  state->visited = (bitmap_t*)calloc(MAX_PARTITION_COUNT, sizeof(bitmap_t));
  assert(state->cost && state->visited);
  state->visited[par->id] = bitmap_init_cpu(par->subgraph.vertex_count);
  for (int pid = 0; pid < engine_partition_count(); pid++) {
    if (pid != par->id && par->outbox[pid].count != 0) {
      state->visited[pid] = (bitmap_t)par->outbox[pid].values;
      bitmap_reset_cpu(state->visited[pid], par->outbox[pid].count);
    }
  }
  state->finished = finished_g;
  OMP(omp parallel for)
  for (id_t v = 0; v < par->subgraph.vertex_count; v++) {
    state->cost[v] = INFINITE;
  }
  // initialize the cost of the source vertex 
  if (src_pid_g == par->id) {
    state->cost[src_vid_g] = 0;
    bitmap_set_cpu(state->visited[par->id], src_vid_g);
  }
}

PRIVATE void bfs_init(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  bfs_state_t* state = (bfs_state_t*)calloc(1, sizeof(bfs_state_t));
  assert(state);
  state->level = 0;
  par->algo_state = state;
  if (par->processor.type == PROCESSOR_CPU) {
    bfs_init_cpu(par);
  } else if (par->processor.type == PROCESSOR_GPU) {
    bfs_init_gpu(par);
  } else {
    assert(false);
  }
}

PRIVATE void bfs_finalize(partition_t* par) {
  bfs_state_t* state = (bfs_state_t*)par->algo_state;
  if (par->processor.type == PROCESSOR_CPU) {
    bitmap_finalize_cpu(state->visited[par->id]);
    free(state->visited);
    free(state->cost);
  } else if (par->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaFree(state->cost));
    bitmap_t visited;
    CALL_CU_SAFE(cudaMemcpy(&visited, &(state->visited[par->id]), 
                            sizeof(bitmap_t), cudaMemcpyDefault));
    bitmap_finalize_gpu(visited);
    CALL_CU_SAFE(cudaFree(state->visited));    
  } else {
    assert(false);
  }
  free(state);
  par->algo_state = NULL;
}

error_t bfs_hybrid(id_t src, uint32_t** cost) {
  // check for special cases
  bool finished = false;
  error_t rc = check_special_cases(src, cost, &finished);
  if (finished) return rc;

  cost_g = (uint32_t*)mem_alloc(engine_vertex_count() * sizeof(uint32_t));
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
    bfs_ss, bfs, bfs_scatter, bfs_init, bfs_finalize, bfs_aggregate
  };
  src_vid_g = GET_VERTEX_ID(engine_vertex_id_in_partition(src));
  src_pid_g = GET_PARTITION_ID(engine_vertex_id_in_partition(src));
  engine_config(&config);
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
