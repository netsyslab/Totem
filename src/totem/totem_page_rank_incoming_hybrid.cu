/**
 * Hybrid implementation of the incoming-based PageRank algorithm
 *
 *  Created on: 2012-09-02
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_comkernel.cuh"
#include "totem_engine.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

/**
 * PageRank specific state
 */
typedef struct pagestate_s {
  rank_t* rank;
  rank_t* rank_s;
  dim3 blocks;
  dim3 threads;
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


/**
 * Sum the rank of the neighbors.
 * @param[in] graph the graph to apply page rank on
 * @param[in] rank an array storing the current rank of each vertex in the graph
 * @return sum of neighbors' ranks
 */
inline __device__ __host__
double sum_neighbors_ranks(int pid, grooves_box_table_t* outbox, 
                           graph_t* graph, vid_t vid, rank_t* ranks) {
  double sum = 0;
  for (eid_t i = graph->vertices[vid];
       i < graph->vertices[vid + 1]; i++) {
    rank_t* src;
    ENGINE_FETCH_SRC(pid, graph->edges[i], outbox, ranks, src, rank_t);
    sum += *src;
  }
  return sum;
}

/**
 * The PageRank kernel. Based on the algorithm described in [Malewicz2010].
 * For each round, each vertex broadcasts along each outgoing edge its tentative
 * PageRank divided by the number of outgoing edges. The tentative PageRank of
 * vertex is calculated as follows: the vertex sums up the values arriving into
 * sum and sets its own tentative PageRank to
 * ((1 - DAMPING_FACTOR) / vertex_count + DAMPING_FACTOR * sum).
 * Broadcasting messages over outgoing edges is done as follows: the value is
 * placed in the outbox buffer. In the next round the inbox and outbox are
 * swapped, and the message will be accessed in the next round via the
 * inbox buffer. This operation simulates a broadcast because all the neighbors
 * of vertex v will access the same location (i.e., inbox[v]) to get the message
 * (i.e., tentative_PageRank_of_v/neighbor_count). In the last round, outbox
 * will contain the PageRank of each vertex.
 * @param[in] graph the graph to apply page rank on
 * @param[in] inbox messages broadcasted to vertices
 * @param[in] outbox messages to be broadcasted in the next round
 */
__global__
void page_rank_incoming_kernel(partition_t par, vid_t vcount, rank_t* rank, 
                               rank_t* rank_s) {
  vid_t vid = THREAD_GLOBAL_INDEX;
  if (vid >= par.subgraph.vertex_count) return;
  double sum = sum_neighbors_ranks(par.id, par.outbox_d, &par.subgraph, vid, 
                                   rank_s);
  double my_rank = ((1 - PAGE_RANK_DAMPING_FACTOR) / vcount) + 
    (PAGE_RANK_DAMPING_FACTOR * sum);
  rank[vid] =  my_rank /
    (par.subgraph.vertices[vid + 1] - par.subgraph.vertices[vid]);
}

PRIVATE void page_rank_incoming_gpu(partition_t* par) {
  page_rank_state_t* ps = (page_rank_state_t*)par->algo_state;
  if (engine_superstep() > 1) {
    page_rank_incoming_kernel<<<ps->blocks, ps->threads, 0,
      par->streams[1]>>>(*par, engine_vertex_count(), ps->rank, ps->rank_s);
    CALL_CU_SAFE(cudaGetLastError());
  }
}

PRIVATE void page_rank_incoming_cpu(partition_t* par) {
  page_rank_state_t* ps = (page_rank_state_t*)par->algo_state;
  if (engine_superstep() > 1) {
    vid_t vcount = engine_vertex_count();
    OMP(omp parallel for)
    for(vid_t vid = 0; vid < par->subgraph.vertex_count; vid++) {
      double sum = sum_neighbors_ranks(par->id, par->outbox, &par->subgraph, 
                                       vid, ps->rank_s);
      double my_rank = ((1 - PAGE_RANK_DAMPING_FACTOR) / vcount) + 
        (PAGE_RANK_DAMPING_FACTOR * sum);
      ps->rank[vid] =  my_rank /
        (par->subgraph.vertices[vid + 1] - par->subgraph.vertices[vid]);
    }
  }
}

PRIVATE void page_rank_incoming(partition_t* par) {
  if (par->processor.type == PROCESSOR_GPU) {
    page_rank_incoming_gpu(par);
  } else {
    assert(par->processor.type == PROCESSOR_CPU);
    page_rank_incoming_cpu(par);
  }
  if (engine_superstep() < (PAGE_RANK_ROUNDS + 1)) {
    engine_report_not_finished();
    page_rank_state_t* ps = (page_rank_state_t*)par->algo_state;
    rank_t* tmp = ps->rank;
    ps->rank = ps->rank_s;
    ps->rank_s = tmp;
  }
}

PRIVATE void page_rank_incoming_gather(partition_t* partition) {
  page_rank_state_t* ps = (page_rank_state_t*)partition->algo_state;
  engine_gather_inbox(partition->id, ps->rank_s);
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

PRIVATE void page_rank_incoming_init_gpu(partition_t* par, 
                                         page_rank_state_t* ps,
                                         rank_t init_value) {
  vid_t vcount = par->subgraph.vertex_count;
  CALL_CU_SAFE(cudaMalloc((void**)&(ps->rank), vcount * sizeof(rank_t)));
  CALL_CU_SAFE(cudaMalloc((void**)&(ps->rank_s), vcount * sizeof(rank_t)));
  KERNEL_CONFIGURE(vcount, ps->blocks, ps->threads);
  memset_device<<<ps->blocks, ps->threads, 0, 
    par->streams[1]>>>(ps->rank, init_value, vcount);
  CALL_CU_SAFE(cudaGetLastError());
}

PRIVATE void page_rank_incoming_init_cpu(partition_t* par, 
                                         page_rank_state_t* ps,
                                         rank_t init_value) {
  assert(par->processor.type == PROCESSOR_CPU);
  ps->rank = (rank_t*)calloc(par->subgraph.vertex_count, sizeof(rank_t));
  ps->rank_s = (rank_t*)calloc(par->subgraph.vertex_count, sizeof(rank_t));
  assert(ps->rank && ps->rank_s);
  OMP(omp parallel for)
  for (vid_t v = 0; v < par->subgraph.vertex_count; v++) { 
    ps->rank[v] = init_value;
  }
}

PRIVATE void page_rank_incoming_init(partition_t* par) {
  if (!par->subgraph.vertex_count) return;
  page_rank_state_t* ps = (page_rank_state_t*)malloc(sizeof(page_rank_state_t));
  assert(ps);
  rank_t init_value = 1 / (rank_t)engine_vertex_count();
  if (par->processor.type == PROCESSOR_GPU) {
    page_rank_incoming_init_gpu(par, ps, init_value);
  } else {
    page_rank_incoming_init_cpu(par, ps, init_value);
  }
  par->algo_state = ps;
}

PRIVATE void page_rank_incoming_finalize(partition_t* partition) {
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

error_t page_rank_incoming_hybrid(float *rank_i, float* rank) {
  // check for special cases
  bool finished = false;
  error_t rc = check_special_cases(rank, &finished);
  if (finished) return rc;

  // initialize global state
  rank_final = rank;

  // initialize the engine
  engine_config_t config = {
    NULL, page_rank_incoming, NULL, page_rank_incoming_gather, 
    page_rank_incoming_init, page_rank_incoming_finalize, 
    page_rank_incoming_aggr, GROOVES_PULL
  };
  engine_config(&config);
  if (engine_largest_gpu_partition()) {
    rank_host = (float*)mem_alloc(engine_largest_gpu_partition() * 
                                  sizeof(float));
  }
  engine_execute();

  // clean up and return
  if (engine_largest_gpu_partition()) mem_free(rank_host);
  return SUCCESS;
}
