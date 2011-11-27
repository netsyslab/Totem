/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Implements the push-relabel algorithm for determining the maximum flow
 * through a directed graph. This implementation is based on the algorithms
 * presented by [Hong08] Z. He, B. Hong, "Dynamically Tuned Push-Relabel
 * Algorithm for the Maximum Flow Problem on CPU-GPU-Hybrid Platforms".
 * http://users.ece.gatech.edu/~bhong/papers/ipdps10.pdf
 *
 *  Created on: 2011-10-21
 *      Author: Greg Redekop
 */

// system includes
#include <cuda.h>

// totem includes
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

// For each stage of the algorithm, a kernel loops over each vertex this many
// times attempting pushes and relabels. This will also control the frequency
// of global relabeling.
#define KERNEL_CYCLES 15


/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
 */
PRIVATE
error_t check_special_cases(graph_t* graph, id_t source_id, id_t sink_id) {
  if((graph == NULL) || (graph->vertex_count == 0) || (!graph->weighted) ||
     (!graph->directed) || (source_id >= graph->vertex_count) ||
     (sink_id >= graph->vertex_count) || (source_id == sink_id)) {
    return FAILURE;
  }
  return SUCCESS;
}


/**
 * CPU Push-relabel operation
 * On a particular vertex u, attempt a push operation along any of its edges.
 * If the push operation fails, perform a relabel.
 */
PRIVATE
void push_relabel_cpu(graph_t* graph, id_t u, id_t source_id, id_t sink_id,
                      weight_t* flow, weight_t* excess, uint32_t* height,
                      id_t* reverse_indices, bool* finished) {
  if (excess[u] <= 0 || height[u] >= graph->vertex_count) return;

  weight_t e_prime = excess[u];
  uint32_t h_prime = INFINITE;
  id_t best_edge_id = INFINITE;

  // Find the lowest neighbor connected by a residual edge
  for (id_t edge_id = graph->vertices[u]; edge_id < graph->vertices[u + 1];
       edge_id++) {
    if (graph->weights[edge_id] <= flow[edge_id]) continue;
    uint32_t h_pprime = height[graph->edges[edge_id]];
    if (h_pprime < h_prime) {
      best_edge_id = edge_id;
      h_prime = h_pprime;
    }
  }

  // If a push applies
  if (height[u] > h_prime) {
    weight_t push_amt = min(e_prime, graph->weights[best_edge_id] -
                            flow[best_edge_id]);
    __sync_fetch_and_add_float(&flow[best_edge_id], push_amt);
    __sync_fetch_and_add_float(&flow[reverse_indices[best_edge_id]], -push_amt);
    __sync_fetch_and_add_float(&excess[u], -push_amt);
    __sync_fetch_and_add_float(&excess[graph->edges[best_edge_id]], push_amt);
    *finished = false;
  }
  // Otherwise perform a relabel
  else if (h_prime != INFINITE) {
    height[u] = h_prime + 1;
    *finished = false;
  }
}


error_t maxflow_cpu(graph_t* graph, id_t source_id, id_t sink_id,
                    weight_t* flow_ret) {
  error_t rc = check_special_cases(graph, source_id, sink_id);
  if (rc != SUCCESS) return rc;

  // Setup residual edges. This creates a new graph and updates the graph
  // pointer to point to this new graph. Thus, we have to do this step before
  // any other allocations/initialization.
  id_t* reverse_indices = NULL;
  graph_t* local_graph = graph_create_bidirectional(graph, &reverse_indices);

  weight_t* excess = (weight_t*)mem_alloc(local_graph->vertex_count *
                                          sizeof(weight_t));
  uint32_t* height = (uint32_t*)mem_alloc(local_graph->vertex_count *
                                          sizeof(uint32_t));
  weight_t* flow = (weight_t*)mem_alloc(local_graph->edge_count *
                                        sizeof(weight_t));

  // Initialize flows, height, and excess to 0
  memset(excess, 0, local_graph->vertex_count * sizeof(weight_t));
  memset(height, 0, local_graph->vertex_count * sizeof(uint32_t));
  memset(flow, 0, local_graph->edge_count * sizeof(weight_t));
  // Initialize source's height to the vertex count
  height[source_id] = (uint32_t) local_graph->vertex_count;

  // Initialize preflow
  for (id_t edge_id = local_graph->vertices[source_id];
       edge_id < local_graph->vertices[source_id + 1]; edge_id++) {
    // Don't setup preflow on residual edges
    if (local_graph->weights[edge_id] == 0) continue;
    flow[edge_id] = local_graph->weights[edge_id];
    flow[reverse_indices[edge_id]] = -local_graph->weights[edge_id];
    excess[local_graph->edges[edge_id]] = local_graph->weights[edge_id];
    excess[source_id] -= local_graph->weights[edge_id];
  }

  // While there exists an applicable push or relabel operation, perform it
  bool finished = false;
  while (!finished) {
    finished = true;
    int count = KERNEL_CYCLES;
    while(count--) {
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif // _OPENMP
      for (id_t u = 0; u < local_graph->vertex_count; u++) {
        if (u == sink_id || u == source_id) continue;
        // Perform a push/relabel operation
        push_relabel_cpu(local_graph, u, source_id, sink_id, flow, excess,
                         height, reverse_indices, &finished);
      }
    }
  }

  // The final flow is the sum of all flows into the sink (ie, the excess
  // value at the sink node)
  *flow_ret = excess[sink_id];

  mem_free(reverse_indices);
  mem_free(height);
  mem_free(excess);
  mem_free(flow);
  // Free our modified new graph
  graph_finalize(local_graph);

  return SUCCESS;
}
