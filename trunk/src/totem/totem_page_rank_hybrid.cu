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

#include "totem_engine.cuh"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

/**
 * Used to define the number of rounds: a static convergance condition
 * for PageRank
 */
#define PAGE_RANK_ROUNDS 30

/**
 * A probability used in the PageRank algorithm. A probability that models the
 * behavior of the random surfer when she moves from one page to another
 * without following the links on the current page.
 */
#define DAMPING_FACTOR 0.85

/**
 * PageRank specific state
 */
typedef struct page_rank_state_s {
  float* rank;
  float* rank_s;
  dim3 blocks1;
  dim3 threads1;
  dim3 blocks2;
  dim3 threads2;
} page_rank_state_t;

/**
 * Stores the final result
 */
float* rank_g = NULL;

/**
 * Used as a temporary buffer to host the final result produced by 
 * GPU partitions
 */
float* rank_h = NULL; 

/**
 * Checks for input parameters and special cases. This is invoked at the 
 * beginning of public interfaces (GPU and CPU)
*/
PRIVATE 
error_t check_special_cases(graph_t* graph, float** rank, bool* finished) {
  *finished = true;
  if (graph == NULL) {
    return FAILURE;
  } else if (graph->vertex_count == 0) {
    return FAILURE;
  } else if (graph->vertex_count == 1) {
    *rank = (float*)mem_alloc(sizeof(float));
    (*rank)[0] = (float)1.0;
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
  float rank[VWARP_BATCH_SIZE];
  id_t vertices[VWARP_BATCH_SIZE + 1];
  // the following ensures 64-bit alignment, it assumes that the
  // cost and vertices arrays are of 32-bit elements.
  // TODO(abdullah) a portable way to do this (what if id_t is 64-bit?)
  int pad;
} vwarp_mem_t;

/**
 * Phase1 kernel of the PageRank GPU algorithm.
 * Produce the sum of the neighbors' ranks. Each vertex atomically
 * adds its value to the mailbox of the destination neighbor vertex.
 */
__global__
void vwarp_sum_neighbors_rank_kernel(partition_t par, int pc, float* rank, 
                                     float* rank_s, int thread_count) {
  if (THREAD_GLOBAL_INDEX >= thread_count) return;
  int warp_offset = THREAD_GLOBAL_INDEX % VWARP_WARP_SIZE;
  int warp_id     = THREAD_GLOBAL_INDEX / VWARP_WARP_SIZE;
  
  // copy my work to local space
  __shared__ vwarp_mem_t smem[(MAX_THREADS_PER_BLOCK / VWARP_WARP_SIZE)];
  vwarp_mem_t* my_space = smem + (THREAD_GRID_INDEX / VWARP_WARP_SIZE);
  int v_ = warp_id * VWARP_BATCH_SIZE;
  int batch_size = v_ + VWARP_BATCH_SIZE > par.subgraph.vertex_count?
    par.subgraph.vertex_count - v_ : VWARP_BATCH_SIZE;
  vwarp_memcpy(my_space->rank, &rank[v_], batch_size, warp_offset);
  vwarp_memcpy(my_space->vertices, &(par.subgraph.vertices[v_]), 
               batch_size + 1, warp_offset);

  // iterate over my work
  for(uint32_t v = 0; v < batch_size; v++) {
    int nbr_count = my_space->vertices[v + 1] - my_space->vertices[v];
    id_t* nbrs = &(par.subgraph.edges[my_space->vertices[v]]);
    for(int i = warp_offset; i < nbr_count; i += VWARP_WARP_SIZE) {
      float* dst; const id_t nbr = nbrs[i];
      ENGINE_FETCH_DST(par.id, nbr, par.outbox_d, rank_s, pc, dst, float);
      atomicAdd(dst, my_space->rank[v]);
    }
  }
}

__global__
void compute_normalized_rank_kernel(partition_t par, uint64_t vc,
                                    float* rank, float* rank_s) {
  id_t v = THREAD_GLOBAL_INDEX;
  if (v >= par.subgraph.vertex_count) return;
  float r = ((1 - DAMPING_FACTOR) / vc) + (DAMPING_FACTOR * rank_s[v]);
  rank[v] = r / (par.subgraph.vertices[v + 1] - par.subgraph.vertices[v]);
  rank_s[v] = 0;
}

__global__
void compute_unnormalized_rank_kernel(partition_t par, uint64_t vc, 
                                      float* rank, float* rank_s) {
  id_t v = THREAD_GLOBAL_INDEX;
  if (v >= par.subgraph.vertex_count) return;
  rank[v] = ((1 - DAMPING_FACTOR) / vc) + (DAMPING_FACTOR * rank_s[v]);
}

PRIVATE void page_rank_gpu(partition_t* par) {
  page_rank_state_t* ps = (page_rank_state_t*)par->algo_state;
  if (engine_superstep() > 1) {
    // compute my rank
    if (engine_superstep() != PAGE_RANK_ROUNDS) {
      compute_normalized_rank_kernel<<<ps->blocks1, ps->threads1, 0, 
        par->streams[1]>>>(*par, engine_vertex_count(), ps->rank, ps->rank_s);
    } else {
      compute_unnormalized_rank_kernel<<<ps->blocks1, ps->threads1, 0, 
        par->streams[1]>>>(*par, engine_vertex_count(), ps->rank, ps->rank_s);
    }
  }
  // communicate the ranks
  engine_set_outbox(par->id, 0);
  vwarp_sum_neighbors_rank_kernel<<<ps->blocks2, ps->threads2, 0,
    par->streams[1]>>>(*par, engine_partition_count(), ps->rank, ps->rank_s,
                       VWARP_BATCH_COUNT(par->subgraph.vertex_count) *
                       VWARP_WARP_SIZE);
}

PRIVATE void page_rank_cpu(partition_t* par) {
  page_rank_state_t* ps = (page_rank_state_t*)par->algo_state;
  graph_t* subgraph = &par->subgraph;
  uint32_t vcount = engine_vertex_count();
  int pc = engine_partition_count();
  int round = engine_superstep();

  if (round > 1) {
    // compute my rank
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(id_t v = 0; v < subgraph->vertex_count; v++) {
      uint32_t nbrs = subgraph->vertices[v + 1] - subgraph->vertices[v];
      float rank = ((1 - DAMPING_FACTOR) / vcount) + 
        (DAMPING_FACTOR * ps->rank_s[v]);
      ps->rank[v] = (round == (PAGE_RANK_ROUNDS)) ? rank : rank / nbrs;
      ps->rank_s[v] = 0;
    }
  }

  // communicate the ranks
  engine_set_outbox(par->id, 0);
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(id_t v = 0; v < subgraph->vertex_count; v++) {
    float my_rank = ps->rank[v];
    for (id_t i = subgraph->vertices[v]; i < subgraph->vertices[v + 1]; i++) {
      float* dst; id_t nbr = subgraph->edges[i];
      ENGINE_FETCH_DST(par->id, nbr, par->outbox, ps->rank_s, pc, dst, float);
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
  if (engine_superstep() == PAGE_RANK_ROUNDS) {
    engine_report_finished(partition->id);
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
  float* src_rank = NULL;
  if (partition->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaMemcpy(rank_h, ps->rank,
                            subgraph->vertex_count * sizeof(float),
                            cudaMemcpyDefault));
    src_rank = rank_h;
  } else {
    assert(partition->processor.type == PROCESSOR_CPU);
    src_rank = ps->rank;
  }
  // aggregate the results
  for (id_t v = 0; v < subgraph->vertex_count; v++) {
    rank_g[partition->map[v]] = src_rank[v];
  }
}

PRIVATE void page_rank_init(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  page_rank_state_t* ps = (page_rank_state_t*)malloc(sizeof(page_rank_state_t));
  assert(ps);
  float init_value = 1 / (float)engine_vertex_count();
  uint64_t vcount = par->subgraph.vertex_count;
  if (par->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaMalloc((void**)&(ps->rank), vcount * sizeof(float)));
    CALL_CU_SAFE(cudaMalloc((void**)&(ps->rank_s), vcount * sizeof(float)));
    KERNEL_CONFIGURE(VWARP_WARP_SIZE * VWARP_BATCH_COUNT(vcount), 
                     ps->blocks2, ps->threads2);
    KERNEL_CONFIGURE(vcount, ps->blocks1, ps->threads1);
    // TODO(abdullah): Use user provided initialization values
    memset_device<<<ps->blocks1, ps->threads1, 0, 
      par->streams[1]>>>(ps->rank, init_value, vcount);
    CALL_CU_SAFE(cudaGetLastError());
    memset_device<<<ps->blocks1, ps->threads1, 0, 
      par->streams[1]>>>(ps->rank_s, (float)0.0, vcount);
    CALL_CU_SAFE(cudaGetLastError());
  } else {
    assert(par->processor.type == PROCESSOR_CPU);    
    ps->rank = (float*)calloc(vcount, sizeof(float));
    ps->rank_s = (float*)calloc(vcount, sizeof(float));
    assert(ps->rank && ps->rank_s);
    for (id_t v = 0; v < vcount; v++) ps->rank[v] = init_value;
  }
  par->algo_state = ps;
}

PRIVATE void page_rank_finalize(partition_t* partition) {
  assert(partition->algo_state);
  page_rank_state_t* ps = (page_rank_state_t*)partition->algo_state;
  if (partition->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaFree(ps->rank));
    CALL_CU_SAFE(cudaFree(ps->rank_s));
  } else {
    assert(partition->processor.type == PROCESSOR_CPU);
    assert(ps->rank && ps->rank_s);
    free(ps->rank);
    free(ps->rank_s);
  }
  free(ps);
  partition->algo_state = NULL;
}

// TODO(abdullah): Add partitioning algorithm as an input parameter
error_t page_rank_hybrid(graph_t* graph, float *rank_i, float** rank) {
  // check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, rank, &finished);
  if (finished) return rc;

  // initialize global state
  rank_g = (float*)mem_alloc(graph->vertex_count * sizeof(float));

  // initialize the engine
  engine_config_t config = {
    graph,
    PAR_RANDOM,
    sizeof(float),
    page_rank,
    page_rank_scatter,
    page_rank_init,
    page_rank_finalize,
    page_rank_aggr
  };
  engine_init(&config);
  if (engine_largest_gpu_partition()) {
    rank_h = (float*)mem_alloc(engine_largest_gpu_partition() * sizeof(float));
  }
  engine_execute();

  // clean up and return
  *rank = rank_g;
  if (engine_largest_gpu_partition()) mem_free(rank_h);
  rank_g = NULL;
  rank_h = NULL;
  return SUCCESS;
}
