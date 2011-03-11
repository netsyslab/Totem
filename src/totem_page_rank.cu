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
 *  Created on: 2011-03-06
 *  Author: Abdullah Gharaibeh
 */

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
 * of vertex v will access the same location (i.e., inbox[v]) to get the messege
 * (i.e., tentative_PageRank_of_v/neighbor_count). In the last round, outbox 
 * will contain the PageRank of each vertex.
 * @param[in] graph the graph to apply page rank on
 * @param[in] inbox messeges broadcasted to vertices
 * @param[in] outbox messeges to be broadcasted in the next round
 */
__global__ 
void page_rank_kernel(graph_t graph, float* inbox, float* outbox, 
                      bool last_round) {

  // get the thread's linear index
  uint32_t my_index = THREAD_GLOBAL_INDEX;
  
  // get direct access to graph members
  uint32_t  vertex_count = graph.vertex_count;
  uint32_t* vertices     = graph.vertices;
  uint32_t* edges        = graph.edges;

  if (my_index >= vertex_count) {
    return;
  }
                                                                               
  // get the neighbors
  uint32_t  neighbors_count = vertices[my_index + 1] - vertices[my_index];
  uint32_t* neighbors       = &(edges[vertices[my_index]]);

  // calculate the sum of all neighbors' rank
  double sum = 0;
  for (uint32_t i = 0; i < neighbors_count; i++) {
    uint32_t neighbor = neighbors[i];
    sum += inbox[neighbor];
  }
  
  // calculate my rank
  float my_rank = 
    ((1 - DAMPING_FACTOR) / (double)vertex_count) + (DAMPING_FACTOR * sum);
  outbox[my_index] = last_round? my_rank: my_rank / neighbors_count;
}


error_t page_rank_gpu(graph_t* graph, float** rank) {
  
  /* had to define them at the beginning to avoid a compilation problem with 
     goto-label error handling mechanism */
  dim3 blocks; 
  dim3 threads_per_block;

  // will be passed to the kernel
  graph_t graph_d;  
  memcpy(&graph_d, graph, sizeof(graph_t));

  uint32_t vertex_count = graph->vertex_count;
  uint32_t edge_count   = graph->edge_count;

  // allocate vertices and edges device buffers and move them to the device
  CHECK_ERR(cudaMalloc((void**)&graph_d.vertices, (vertex_count + 1) *
                       sizeof(uint32_t)) == cudaSuccess, err);
  CHECK_ERR(cudaMalloc((void**)&graph_d.edges, edge_count * 
                       sizeof(uint32_t)) == cudaSuccess, err_free_vertices);

  CHECK_ERR(cudaMemcpy(graph_d.vertices, graph->vertices, 
                       (vertex_count + 1) * sizeof(uint32_t), 
                       cudaMemcpyHostToDevice) == cudaSuccess, 
            err_free_edges);
  CHECK_ERR(cudaMemcpy(graph_d.edges, graph->edges, 
                       edge_count * sizeof(uint32_t),
                       cudaMemcpyHostToDevice) == cudaSuccess, 
            err_free_edges);

  // allocate inbox and outbox device buffers
  float *inbox_d;
  CHECK_ERR(cudaMalloc((void**)&inbox_d, vertex_count * 
                       sizeof(float)) == cudaSuccess, err_free_edges);
  float *outbox_d;
  CHECK_ERR(cudaMalloc((void**)&outbox_d, vertex_count * 
                       sizeof(float)) == cudaSuccess, err_free_inbox);

  /* set the number of blocks, TODO(abdullah) handle the case when 
     vertex_count is larger than number of threads. */
  assert(vertex_count <= MAX_THREAD_COUNT);
  KERNEL_CONFIGURE(vertex_count, blocks, threads_per_block);
  
  // initialize the rank of each vertex 
  float initial_value;
  initial_value = 1/(float)vertex_count;
  memset_device<<<blocks, threads_per_block>>>
    (outbox_d, initial_value, vertex_count);
  CHECK_ERR(cudaGetLastError() == cudaSuccess, err_free_outbox);

  uint32_t round;
  for (round = 0; round < PAGE_RANK_ROUNDS; round++) {
    // swap the inbox and outbox pointers (simulates passing messages)
    float* tmp = inbox_d;
    inbox_d = outbox_d;
    outbox_d = tmp;

    // call the kernel
    bool last_round = (round == (PAGE_RANK_ROUNDS - 1));
    page_rank_kernel<<<blocks, threads_per_block>>>
      (graph_d, inbox_d, outbox_d, last_round);
    CHECK_ERR(cudaGetLastError() == cudaSuccess, err_free_outbox);
    
    cudaThreadSynchronize();
    CHECK_ERR(cudaGetLastError() == cudaSuccess, err_free_outbox);
  }

  // copy back the final result from the outbox
  float* my_rank;
  my_rank = (float*)mem_alloc(vertex_count * sizeof(float));
  CHECK_ERR(cudaMemcpy(my_rank, outbox_d, vertex_count * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess, err_free_all);

  // we are done! set the output and clean up
  *rank = my_rank;  
  cudaFree(outbox_d);
  cudaFree(inbox_d);
  cudaFree(graph_d.edges);
  cudaFree(graph_d.vertices);
  return SUCCESS;

  // error handlers
 err_free_all:
 err_free_outbox:
  cudaFree(outbox_d);
 err_free_inbox:
  cudaFree(inbox_d);
 err_free_edges:
  cudaFree(graph_d.edges);
 err_free_vertices:
  cudaFree(graph_d.vertices);
 err:
  printf("%d\n", cudaGetLastError());
  return FAILURE;
}

error_t page_rank_cpu(graph_t* graph, float** rank) {

  // get direct access to graph members
  uint32_t  vertex_count = graph->vertex_count;
  uint32_t* vertices     = graph->vertices;
  uint32_t* edges        = graph->edges;

  // allocate buffers
  float* inbox = (float*)mem_alloc(vertex_count * sizeof(float));
  float* outbox = (float*)mem_alloc(vertex_count * sizeof(float));
  
  // initialize the rank of each vertex
  float initial_value;
  initial_value = 1/(float)vertex_count;
  for (uint32_t vid = 0; vid < vertex_count; vid++) {
    outbox[vid] = initial_value;
  }

  for (uint32_t round = 0; round < PAGE_RANK_ROUNDS; round++) {

    // swap the inbox and outbox pointers (simulates passing messages!)
    float* tmp = inbox;
    inbox = outbox;
    outbox = tmp;

    // iterate over all vertices to calculate the ranks for this round
    for(uint32_t vid = 0; vid < vertex_count; vid++) {
      // get the neighbors
      uint32_t   neighbors_count  = vertices[vid + 1] - vertices[vid];
      uint32_t*  neighbors        = &(edges[vertices[vid]]);

      // calculate the sum of all neighbors' rank
      double sum = 0;
      for (uint32_t i = 0; i < neighbors_count; i++) {
        uint32_t neighbor  = neighbors[i];
        sum               += inbox[neighbor];
      }

      // calculate my rank
      float my_rank = 
        ((1 - DAMPING_FACTOR) / vertex_count) + (DAMPING_FACTOR * sum);
      outbox[vid] = 
        (round == (PAGE_RANK_ROUNDS - 1)) ? my_rank : my_rank / neighbors_count;
    }   
  }

  // we are done! set the output and clean up.
  *rank = outbox;
  mem_free(inbox);
  return SUCCESS;
}
