/**
 * Hybrid implementation of the incoming-based PageRank algorithm
 *
 *  Created on: 2012-09-02
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_alg.h"
#include "totem_engine.cuh"

/**
 * PageRank specific state
 */
typedef struct pagestate_s {
  rank_t* rank;
  rank_t* rank_s[MAX_PARTITION_COUNT];
} page_rank_state_t;

/**
 * final result
 */
PRIVATE rank_t* rank_final = NULL;

/**
 * Used as a temporary buffer to host the final result produced by
 * GPU partitions
 */
PRIVATE rank_t* rank_host = NULL;

/**
 * A constant used as part of calculating the rank in each round. The value
 * depends on the number of vertices in the graph, and is equal to:
 * ((1 - PAGE_RANK_DAMPING_FACTOR) / vertex_count)
 */
PRIVATE rank_t c1 = 0;

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
*/
PRIVATE
error_t check_special_cases(float* rank, bool* finished) {
  *finished = true;
  if (engine_vertex_count() == 0) {
    return FAILURE;
  } else if (engine_vertex_count() == 1) {
    rank[0] = 1.0;
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

template<int VWARP_WIDTH>
PRIVATE __device__ void 
sum_neighbors(const vid_t* __restrict nbrs, const vid_t nbr_count,
              rank_t** rank_s, rank_t* vwarp_rank, int warp_offset) {
  if (VWARP_WIDTH > 32) __syncthreads();
  rank_t sum = 0;
  for (vid_t i = warp_offset; i < nbr_count; i+= VWARP_WIDTH) {
    vid_t nbr = GET_VERTEX_ID(nbrs[i]);
    int nbr_pid = GET_PARTITION_ID(nbrs[i]);
    rank_t* nbr_rank = rank_s[nbr_pid];
    sum += nbr_rank[nbr];
  }
  vwarp_rank[warp_offset] = sum;
  if (VWARP_WIDTH > 32) __syncthreads();

  // completely unrolled parallel reduction
  if (warp_offset < VWARP_WIDTH / 2) {
    if (warp_offset < 32) {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile rank_t *smem = vwarp_rank;
      
      // do reduction in shared mem
      if (VWARP_WIDTH > 1024) assert(false);
      if (VWARP_WIDTH == 1024) {
        if (warp_offset < 512) {
          vwarp_rank[warp_offset] = sum = sum + vwarp_rank[warp_offset + 512];
        }
        __syncthreads();
      }
      
      if (VWARP_WIDTH >= 512) {
        if (warp_offset < 256) {
          vwarp_rank[warp_offset] = sum = sum + vwarp_rank[warp_offset + 256];
        }
        __syncthreads();
      }
      
      if (VWARP_WIDTH >= 256) {
        if (warp_offset < 128) {
          vwarp_rank[warp_offset] = sum = sum + vwarp_rank[warp_offset + 128];
        }
        __syncthreads();
      }
      
      if (VWARP_WIDTH >= 128) {
        if (warp_offset <  64) {
          vwarp_rank[warp_offset] = sum = sum + vwarp_rank[warp_offset + 64];
        }
        __syncthreads();
      }
      
      if (VWARP_WIDTH >= 64) {
        smem[warp_offset] = sum = sum + smem[warp_offset + 32];
      }
      if (VWARP_WIDTH >= 32) {
        smem[warp_offset] = sum = sum + smem[warp_offset + 16];
      }
      if (VWARP_WIDTH >= 16) {
        smem[warp_offset] = sum = sum + smem[warp_offset + 8];
      }
      if (VWARP_WIDTH >= 8) {
        smem[warp_offset] = sum = sum + smem[warp_offset + 4];
      }
      if (VWARP_WIDTH >= 4) {
        smem[warp_offset] = sum = sum + smem[warp_offset + 2];
      }
      if (VWARP_WIDTH >= 2) {
        smem[warp_offset] = sum = sum + smem[warp_offset + 1];
      }
    }
  }
}

/**
 * The PageRank kernel. Based on the algorithm described in [Malewicz2010].
 */
template<int VWARP_WIDTH, int VWARP_BATCH>
PRIVATE __global__
void page_rank_incoming_kernel(partition_t par, page_rank_state_t ps, 
                               float c1, bool last_round) {
  vid_t vertex_count = par.subgraph.vertex_count;
  if (THREAD_GLOBAL_INDEX >= 
      vwarp_thread_count(vertex_count, VWARP_WIDTH, VWARP_BATCH)) return;

  const eid_t* __restrict vertices = par.subgraph.vertices;
  const vid_t* __restrict edges = par.subgraph.edges;

  vid_t start_vertex = vwarp_block_start_vertex(VWARP_WIDTH, VWARP_BATCH) + 
    vwarp_warp_start_vertex(VWARP_WIDTH, VWARP_BATCH);
  vid_t end_vertex = start_vertex +
    vwarp_warp_batch_size(vertex_count, VWARP_WIDTH, VWARP_BATCH);
  int warp_offset = vwarp_thread_index(VWARP_WIDTH);

  // Each thread in every warp has an entry in the following array which will be
  // used to calculate intermediary delta values in shared memory
  __shared__ rank_t block_rank[MAX_THREADS_PER_BLOCK];
  int index = THREAD_BLOCK_INDEX / VWARP_WIDTH;
  rank_t* vwarp_rank = &block_rank[index * VWARP_WIDTH];

  for(vid_t v = start_vertex; v < end_vertex; v++) {
    const eid_t nbr_count = vertices[v + 1] - vertices[v];
    const vid_t* __restrict nbrs = &(edges[vertices[v]]);
    sum_neighbors<VWARP_WIDTH>
      (nbrs, nbr_count, ps.rank_s, vwarp_rank, warp_offset);
    if (warp_offset == 0) {
      rank_t my_rank = c1 + (PAGE_RANK_DAMPING_FACTOR * vwarp_rank[0]);
      if (!last_round) {
        my_rank /= nbr_count;
      }
      ps.rank[v] = my_rank;
    }
  }
}

template<int VWARP_WIDTH, int VWARP_BATCH>
PRIVATE void page_rank_gpu_launch(partition_t* par, page_rank_state_t* ps,
                                  bool last_round) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(vwarp_thread_count(par->subgraph.vertex_count,
                                      VWARP_WIDTH, VWARP_BATCH),
                   blocks, threads);
  page_rank_incoming_kernel<VWARP_WIDTH, VWARP_BATCH><<<blocks, threads, 0,
    par->streams[1]>>>(*par, *ps, c1, last_round);
  CALL_CU_SAFE(cudaGetLastError());
}

typedef void(*page_rank_gpu_func_t)(partition_t*, page_rank_state_t*, bool);
PRIVATE const page_rank_gpu_func_t PAGE_RANK_GPU_FUNC[] = {
  // RANDOM algorithm
  page_rank_gpu_launch<VWARP_MEDIUM_WARP_WIDTH, VWARP_MEDIUM_BATCH_SIZE>,
  // HIGH partitioning
  page_rank_gpu_launch<VWARP_SHORT_WARP_WIDTH, VWARP_SMALL_BATCH_SIZE>,
  // LOW partitioning
  page_rank_gpu_launch<MAX_THREADS_PER_BLOCK, VWARP_MEDIUM_WARP_WIDTH>
};

PRIVATE void page_rank_incoming_gpu(partition_t* par, bool last_round) {
  page_rank_state_t* ps = (page_rank_state_t*)par->algo_state;
  int par_alg = engine_partition_algorithm();
  PAGE_RANK_GPU_FUNC[par_alg](par, ps, last_round);
}

PRIVATE void page_rank_incoming_cpu(partition_t* par, bool last_round) {
  page_rank_state_t* ps = (page_rank_state_t*)par->algo_state;
  graph_t* subgraph = &(par->subgraph);
  vid_t vcount = engine_vertex_count();  

  OMP(omp parallel for schedule(runtime))
  for(vid_t vid = 0; vid < subgraph->vertex_count; vid++) {
    rank_t sum = 0;
    for (eid_t i = subgraph->vertices[vid];
         i < subgraph->vertices[vid + 1]; i++) {
      rank_t* nbr_rank_s = ps->rank_s[GET_PARTITION_ID(subgraph->edges[i])];
      sum += nbr_rank_s[GET_VERTEX_ID(subgraph->edges[i])];
    }
    rank_t my_rank = c1 + (PAGE_RANK_DAMPING_FACTOR * sum);
    if (!last_round) {
      my_rank /= (subgraph->vertices[vid + 1] - subgraph->vertices[vid]); 
    }
    ps->rank[vid] = my_rank;
  }  
}

PRIVATE void page_rank_incoming(partition_t* par) {
  if (engine_superstep() > 1) {
    page_rank_state_t* ps = (page_rank_state_t*)par->algo_state;
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      if (pid != par->id) {
        ps->rank_s[pid] = (rank_t*)par->outbox[pid].pull_values;
      }
    }
    rank_t* tmp = ps->rank;
    ps->rank = ps->rank_s[par->id];
    ps->rank_s[par->id] = tmp;
    bool last_round = (engine_superstep() == (PAGE_RANK_ROUNDS + 1));
    if (par->processor.type == PROCESSOR_GPU) {
      page_rank_incoming_gpu(par, last_round);
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      page_rank_incoming_cpu(par, last_round);
    }
  }
  if (engine_superstep() < (PAGE_RANK_ROUNDS + 1)) {
    engine_report_not_finished();
  }
}

PRIVATE void page_rank_incoming_gather(partition_t* partition) {
  page_rank_state_t* ps = (page_rank_state_t*)partition->algo_state;
  engine_gather_inbox(partition->id, ps->rank);
}

PRIVATE void page_rank_incoming_aggr(partition_t* partition) {
  if (!partition->subgraph.vertex_count) return;
  page_rank_state_t* ps = (page_rank_state_t*)partition->algo_state;
  graph_t* subgraph = &partition->subgraph;
  rank_t* src_rank = NULL;
  if (partition->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaMemcpy(rank_host, ps->rank, 
                            subgraph->vertex_count * sizeof(rank_t),
                            cudaMemcpyDefault));
    src_rank = rank_host;
  } else {
    assert(partition->processor.type == PROCESSOR_CPU);
    src_rank = ps->rank;
  }
  // aggregate the results
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    rank_final[partition->map[v]] = src_rank[v];
  }
}

PRIVATE void page_rank_incoming_init(partition_t* par) {
  vid_t vcount = par->subgraph.vertex_count;
  if (vcount == 0) return;
  page_rank_state_t* ps = NULL;
  CALL_SAFE(totem_calloc(sizeof(page_rank_state_t), TOTEM_MEM_HOST, 
                         (void**)&ps));
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_GPU) {
    type = TOTEM_MEM_DEVICE;
  }

  CALL_SAFE(totem_malloc(vcount * sizeof(rank_t), type, 
                         (void**)&(ps->rank_s[par->id])));
  CALL_SAFE(totem_malloc(vcount * sizeof(rank_t), type, (void**)&(ps->rank)));
  rank_t init_value = 1 / (rank_t)engine_vertex_count();
  totem_memset(ps->rank, init_value, vcount, type, par->streams[1]);
  par->algo_state = ps;
}

PRIVATE void page_rank_incoming_finalize(partition_t* partition) {
  assert(partition->algo_state);
  page_rank_state_t* ps = (page_rank_state_t*)partition->algo_state;
  totem_mem_t type = TOTEM_MEM_HOST;
  if (partition->processor.type == PROCESSOR_GPU) {
    type = TOTEM_MEM_DEVICE;
  } 
  totem_free(ps->rank, type);
  totem_free(ps->rank_s[partition->id], type);
  totem_free(ps, TOTEM_MEM_HOST);
  partition->algo_state = NULL;
}

error_t page_rank_incoming_hybrid(float *rank_i, float* rank) {
  // check for special cases
  bool finished = false;
  error_t rc = check_special_cases(rank, &finished);
  if (finished) return rc;

  // initialize global state
  rank_final = rank;
  c1 = ((1 - PAGE_RANK_DAMPING_FACTOR) / ((double)engine_vertex_count()));

  // initialize the engine
  engine_config_t config = {
    NULL, page_rank_incoming, NULL, page_rank_incoming_gather, 
    page_rank_incoming_init, page_rank_incoming_finalize, 
    page_rank_incoming_aggr, GROOVES_PULL
  };
  engine_config(&config);
  if (engine_largest_gpu_partition()) {
    CALL_SAFE(totem_malloc(engine_largest_gpu_partition() * sizeof(float), 
                           TOTEM_MEM_HOST_PINNED, (void**)&rank_host));
  }
  engine_execute();

  // clean up and return
  if (engine_largest_gpu_partition()) {
    totem_free(rank_host, TOTEM_MEM_HOST_PINNED);
  }
  return SUCCESS;
}
