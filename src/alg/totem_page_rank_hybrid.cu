/**
 * Implements a simplified version of the PageRank algorithm based on the
 * algorithm described by [Malewicz2010]
 * G. Malewicz, M. H. Austern, A. J. C. Bik, J. C. Dehnert, I. Horn, N. Leiser,
 * and G. Czajkowski. Pregel: a system for large-scale graph processing. In
 * Proceedings of the 28th ACM symposium on Principles of distributed computing,
 * PODC 09, page 6, New York, NY, USA, 2009. ACM.
 *
 * Algorithm description [Malewicz2010]:
 * The graph is initialized so that in round 0, the PageRank of each vertex is
 * set to 1 / vertex_count. For PAGE_RANK_ROUNDS rounds, each vertex sends along
 * each outgoing edge its tentative PageRank divided by the number of outgoing
 * edges. The tentative PageRank is calculated as follows: the vertex sums up
 * the values arriving into sum and sets its own tentative PageRank to
 * ((1 - DAMPING_FACTOR) / vertex_count + DAMPING_FACTOR * sum).
 *
 *  Created on: 2012-01-30
 *  Author: Abdullah Gharaibeh
 */

#include "totem_alg.h"
#include "totem_engine.cuh"

/**
 * PageRank specific state
 */
typedef struct page_rank_state_s {
  rank_t* rank;
  rank_t* rank_s;
  dim3 blocks_rank;
  dim3 threads_rank;
  dim3 blocks_sum;
  dim3 threads_sum;
} page_rank_state_t;

/**
 * Stores the final result
 */
rank_t* rank_g = NULL;

/**
 * Used as a temporary buffer to host the final result produced by
 * GPU partitions
 */
rank_t* rank_h = NULL;

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
*/
PRIVATE
error_t check_special_cases(rank_t* rank, bool* finished) {
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

/**
 * This structure is used by virtual warp-based implementation. It stores a
 * batch of work. It is typically allocated on shared memory and is processed by
 * a single virtual warp.
 */
typedef struct {
  eid_t vertices[VWARP_DEFAULT_BATCH_SIZE + 2];
  rank_t rank[VWARP_DEFAULT_BATCH_SIZE];
} vwarp_mem_t;

/**
 * Phase1 kernel of the PageRank GPU algorithm. Produce the sum of
 * the neighbors' ranks. Each vertex atomically adds its value to
 * the temporary rank (rank_s) of the destination neighbor vertex.
 */
__global__
void vwarp_sum_neighbors_rank_kernel(partition_t par, rank_t* rank, 
                                     rank_t* rank_s, vid_t thread_count) {
  if (THREAD_GLOBAL_INDEX >= thread_count) return;
  vid_t warp_offset = THREAD_GLOBAL_INDEX % VWARP_DEFAULT_WARP_WIDTH;
  vid_t warp_id     = THREAD_GLOBAL_INDEX / VWARP_DEFAULT_WARP_WIDTH;

  // copy my work to local space
  __shared__ vwarp_mem_t smem[MAX_THREADS_PER_BLOCK / 
                              VWARP_DEFAULT_WARP_WIDTH];
  vwarp_mem_t* my_space = &smem[THREAD_BLOCK_INDEX / 
                                VWARP_DEFAULT_WARP_WIDTH];
  vid_t v_ = warp_id * VWARP_DEFAULT_BATCH_SIZE;
  int my_batch_size = VWARP_DEFAULT_BATCH_SIZE;
  if (v_ + VWARP_DEFAULT_BATCH_SIZE > par.subgraph.vertex_count) {
    my_batch_size = par.subgraph.vertex_count - v_;
  }
  vwarp_memcpy(my_space->rank, &rank[v_], my_batch_size, warp_offset);
  vwarp_memcpy(my_space->vertices, &(par.subgraph.vertices[v_]),
               my_batch_size + 1, warp_offset);

  // iterate over my work
  for(vid_t v = 0; v < my_batch_size; v++) {
    vid_t nbr_count = my_space->vertices[v + 1] - my_space->vertices[v];
    vid_t* nbrs = &(par.subgraph.edges[my_space->vertices[v]]);
    for(vid_t i = warp_offset; i < nbr_count; i += VWARP_DEFAULT_WARP_WIDTH) {
      const vid_t nbr = nbrs[i];
      rank_t* dst = engine_get_dst_ptr(par.id, nbr, par.outbox_d, rank_s);      
      atomicAdd(dst, my_space->rank[v]);
    }
  }
}

__global__
void compute_normalized_rank_kernel(partition_t par, vid_t vc, rank_t* rank, 
                                    rank_t* rank_s) {
  vid_t v = THREAD_GLOBAL_INDEX;
  if (v >= par.subgraph.vertex_count) return;
  rank_t r = ((1 - PAGE_RANK_DAMPING_FACTOR) / vc) + 
    (PAGE_RANK_DAMPING_FACTOR * rank_s[v]);
  rank[v] = r / (par.subgraph.vertices[v + 1] - par.subgraph.vertices[v]);
  rank_s[v] = 0;
}

__global__
void compute_unnormalized_rank_kernel(partition_t par, vid_t vc, rank_t* rank,
                                      rank_t* rank_s) {
  vid_t v = THREAD_GLOBAL_INDEX;
  if (v >= par.subgraph.vertex_count) return;
  rank[v] = ((1 - PAGE_RANK_DAMPING_FACTOR) / vc) + 
    (PAGE_RANK_DAMPING_FACTOR * rank_s[v]);
}

PRIVATE void page_rank_gpu(partition_t* par) {
  page_rank_state_t* ps = (page_rank_state_t*)par->algo_state;
  if (engine_superstep() > 1) {
    // compute my rank
    if (engine_superstep() != PAGE_RANK_ROUNDS) {
      compute_normalized_rank_kernel<<<ps->blocks_rank, ps->threads_rank, 0,
        par->streams[1]>>>(*par, engine_vertex_count(), ps->rank, ps->rank_s);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      compute_unnormalized_rank_kernel<<<ps->blocks_rank, ps->threads_rank, 0,
        par->streams[1]>>>(*par, engine_vertex_count(), ps->rank, ps->rank_s);
      CALL_CU_SAFE(cudaGetLastError());
    }
  }
  // communicate the ranks
  engine_set_outbox(par->id, 0);
  vwarp_sum_neighbors_rank_kernel<<<ps->blocks_sum, ps->threads_sum, 0,
    par->streams[1]>>>(*par, ps->rank, ps->rank_s, 
                       vwarp_default_thread_count(par->subgraph.vertex_count));
  CALL_CU_SAFE(cudaGetLastError());
}

PRIVATE void page_rank_cpu(partition_t* par) {
  page_rank_state_t* ps = (page_rank_state_t*)par->algo_state;
  graph_t* subgraph = &par->subgraph;
  vid_t vcount = engine_vertex_count();
  int round = engine_superstep();

  if (round > 1) {
    // compute my rank The loop has no load balancing issues, hence the choice
    // of dividing the iterations between the threads statically via the static
    // schedule clause
    OMP(omp parallel for schedule(static))
    for(vid_t v = 0; v < subgraph->vertex_count; v++) {
      vid_t nbrs = subgraph->vertices[v + 1] - subgraph->vertices[v];
      rank_t rank = ((1 - PAGE_RANK_DAMPING_FACTOR) / vcount) +
        (PAGE_RANK_DAMPING_FACTOR * ps->rank_s[v]);
      ps->rank[v] = (round == (PAGE_RANK_ROUNDS)) ? rank : rank / nbrs;
      ps->rank_s[v] = 0;
    }
  }

  // communicate the ranks
  engine_set_outbox(par->id, 0);
  // The "runtime" scheduling clause defer the choice of thread scheduling
  // algorithm to the choice of the client, either via OS environment variable
  // or omp_set_schedule interface.
  OMP(omp parallel for schedule(runtime))
  for(vid_t v = 0; v < subgraph->vertex_count; v++) {
    rank_t my_rank = ps->rank[v];
    for (eid_t i = subgraph->vertices[v]; i < subgraph->vertices[v + 1]; i++) {
      vid_t nbr = subgraph->edges[i];
      rank_t* dst = engine_get_dst_ptr(par->id, nbr, par->outbox, ps->rank_s);
      __sync_fetch_and_add_float(dst, my_rank);
    }
  }
}

PRIVATE void page_rank(partition_t* partition) {
  if (partition->processor.type == PROCESSOR_GPU) {
    page_rank_gpu(partition);
  } else {
    assert(partition->processor.type == PROCESSOR_CPU);
    page_rank_cpu(partition);
  }
  if (engine_superstep() < PAGE_RANK_ROUNDS) {
    engine_report_not_finished();
  }
}

PRIVATE void page_rank_scatter(partition_t* partition) {
  page_rank_state_t* ps = (page_rank_state_t*)partition->algo_state;
  engine_scatter_inbox_add(partition->id, ps->rank_s);
}

PRIVATE void page_rank_aggr(partition_t* partition) {
  if (!partition->subgraph.vertex_count) return;
  page_rank_state_t* ps = (page_rank_state_t*)partition->algo_state;
  graph_t* subgraph = &partition->subgraph;
  rank_t* src_rank = NULL;
  if (partition->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaMemcpy(rank_h, ps->rank,
                            subgraph->vertex_count * sizeof(rank_t),
                            cudaMemcpyDefault));
    src_rank = rank_h;
  } else {
    assert(partition->processor.type == PROCESSOR_CPU);
    src_rank = ps->rank;
  }
  // aggregate the results
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    rank_g[partition->map[v]] = src_rank[v];
  }
}

PRIVATE void page_rank_init(partition_t* par) {
  vid_t vcount = par->subgraph.vertex_count;
  if (vcount == 0) return;
  page_rank_state_t* ps = NULL;
  CALL_SAFE(totem_calloc(sizeof(page_rank_state_t), TOTEM_MEM_HOST, 
                         (void**)&ps));
  totem_mem_t type = TOTEM_MEM_HOST;
  if (par->processor.type == PROCESSOR_GPU) {
    type = TOTEM_MEM_DEVICE;
    KERNEL_CONFIGURE(vwarp_default_thread_count(vcount),
                     ps->blocks_sum, ps->threads_sum);
    KERNEL_CONFIGURE(vcount, ps->blocks_rank, ps->threads_rank);
  }
  CALL_SAFE(totem_calloc(vcount * sizeof(rank_t), type, (void**)&(ps->rank_s)));
  CALL_SAFE(totem_malloc(vcount * sizeof(rank_t), type, (void**)&(ps->rank)));
  rank_t init_value = 1 / (rank_t)engine_vertex_count();
  totem_memset(ps->rank, init_value, vcount, type, par->streams[1]);
  par->algo_state = ps;
}

PRIVATE void page_rank_finalize(partition_t* partition) {
  assert(partition->algo_state);
  page_rank_state_t* ps = (page_rank_state_t*)partition->algo_state;
  totem_mem_t type = TOTEM_MEM_HOST;
  if (partition->processor.type == PROCESSOR_GPU) {
    type = TOTEM_MEM_DEVICE;
  } 
  totem_free(ps->rank, type);
  totem_free(ps->rank_s, type);
  totem_free(ps, TOTEM_MEM_HOST);
  partition->algo_state = NULL;
}

error_t page_rank_hybrid(rank_t *rank_i, rank_t* rank) {
  // check for special cases
  bool finished = false;
  error_t rc = check_special_cases(rank, &finished);
  if (finished) return rc;

  // initialize global state
  rank_g = rank;

  // initialize the engine
  engine_config_t config = {
    NULL, page_rank, page_rank_scatter, NULL, page_rank_init, 
    page_rank_finalize, page_rank_aggr, GROOVES_PUSH
  };
  engine_config(&config);
  if (engine_largest_gpu_partition()) {
    CALL_SAFE(totem_malloc(engine_largest_gpu_partition() * sizeof(rank_t), 
                           TOTEM_MEM_HOST_PINNED, (void**)&rank_h));
  }
  engine_execute();

  // clean up and return
  if (engine_largest_gpu_partition()) totem_free(rank_h, TOTEM_MEM_HOST_PINNED);
  return SUCCESS;
}
