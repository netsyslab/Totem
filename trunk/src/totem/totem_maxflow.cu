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

// Static function declarations
__global__
void init_preflow(graph_t graph, id_t edge_base, id_t edge_end, weight_t* flow,
                  weight_t* excess, id_t* reverse_indices);


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
 * A common initialization function for GPU implementations. It allocates and
 * initalizes state on the GPU
 */
PRIVATE
error_t initialize_gpu(graph_t* graph, id_t source_id, id_t* reverse_indices,
                       graph_t** graph_d, weight_t** flow_d,
                       weight_t** excess_d, uint32_t** height_d,
                       id_t** reverse_indices_d, bool** finished_d) {

  dim3 blocks;
  dim3 threads_per_block;

  // Calculate the source excess directly prior to allocation. This prevents
  // compilation errors about variable declaration after a jump
  weight_t source_excess = (weight_t)0;
  for (id_t edge_id = graph->vertices[source_id];
       edge_id < graph->vertices[source_id + 1]; edge_id++) {
    source_excess -= graph->weights[edge_id];
  }
  // Allocate space on GPU
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);
  CHK_CU_SUCCESS(cudaMalloc((void**)flow_d, graph->edge_count *
                            sizeof(weight_t)), err_free_graph_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)reverse_indices_d, graph->edge_count *
                            sizeof(id_t)), err_free_flow_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)excess_d, graph->vertex_count *
                            sizeof(weight_t)), err_free_reverse_indices_d);
  CHK_CU_SUCCESS(cudaMalloc((void**)height_d, graph->vertex_count *
                            sizeof(uint32_t)), err_free_excess_d);
  // Initialize flow, height, and excess to 0.
  KERNEL_CONFIGURE(graph->edge_count, blocks, threads_per_block);
  memset_device<<<blocks, threads_per_block>>>((*flow_d), (weight_t)0,
                                               graph->edge_count);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_all_d);
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
  memset_device<<<blocks, threads_per_block>>>((*excess_d), (weight_t)0,
                                               graph->vertex_count);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_all_d);
  memset_device<<<blocks, threads_per_block>>>((*height_d), (uint32_t)0,
                                               graph->vertex_count);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_all_d);
  CHK_CU_SUCCESS(cudaMemcpy((*reverse_indices_d), reverse_indices,
                            graph->edge_count * sizeof(id_t),
                            cudaMemcpyHostToDevice), err_free_all_d);

  // From the source vertex, initialize preflow
  CHK_CU_SUCCESS(cudaMemset(&((*height_d)[source_id]), graph->vertex_count,
                            sizeof(uint32_t)), err_free_all_d);
  KERNEL_CONFIGURE((graph->vertices[source_id + 1] -
                    graph->vertices[source_id]), blocks, threads_per_block);
  init_preflow<<<blocks, threads_per_block>>>
    (**graph_d, graph->vertices[source_id], graph->vertices[source_id + 1],
     *flow_d, *excess_d, *reverse_indices_d);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_all_d);

  CHK_CU_SUCCESS(cudaMemset(&((*excess_d)[source_id]), source_excess,
                            sizeof(weight_t)), err_free_all_d);
  // Allocate the termination flag
  CHK_CU_SUCCESS(cudaMalloc((void**)finished_d, sizeof(bool)),
                 err_free_all_d);

  return SUCCESS;

  err_free_all_d:
    cudaFree(height_d);
  err_free_excess_d:
    cudaFree(excess_d);
  err_free_reverse_indices_d:
    cudaFree(reverse_indices_d);
  err_free_flow_d:
    cudaFree(flow_d);
  err_free_graph_d:
    graph_finalize_device(*graph_d);
  err:
    return FAILURE;
}


/**
 * Initializes preflow from a given source index. Each edge connected to
 * the source vertex is initialized with its capacity.
 */
__global__
void init_preflow(graph_t graph, id_t edge_base, id_t edge_end, weight_t* flow,
                  weight_t* excess, id_t* reverse_indices) {
  const int offset = THREAD_GLOBAL_INDEX;
  if (offset >= edge_end) return;
  if (graph.weights[edge_base + offset] == 0) return;

  flow[edge_base + offset] = graph.weights[edge_base + offset];
  flow[reverse_indices[edge_base + offset]] = -flow[edge_base + offset];
  excess[graph.edges[edge_base + offset]] = graph.weights[edge_base + offset];
}


/**
 * Implements the push relabel kernel, as per [Hong08]
 */
__global__
void push_relabel_kernel(graph_t graph, weight_t* flow, weight_t* excess,
                         uint32_t* height, id_t* reverse_indices,
                         id_t source_id, id_t sink_id, bool* finished) {
  const id_t u = THREAD_GLOBAL_INDEX;
  if (u >= graph.vertex_count) return;
  if (u == source_id || u == sink_id) return;

  uint32_t count = KERNEL_CYCLES;
  while (count--) {
    if (excess[u] <= 0 || height[u] >= graph.vertex_count) continue;

    weight_t e_prime = excess[u];
    uint32_t h_prime = INFINITE;
    id_t best_edge_id = INFINITE;

    // Find the lowest neighbor connected by a residual edge
    for (id_t edge_id = graph.vertices[u]; edge_id < graph.vertices[u + 1];
         edge_id++) {
      if (graph.weights[edge_id] == flow[edge_id]) continue;
      uint32_t h_pprime = height[graph.edges[edge_id]];
      if (h_pprime < h_prime) {
        best_edge_id = edge_id;
        h_prime = h_pprime;
      }
    }

    // If a push applies
    if (height[u] > h_prime) {
      weight_t push_amt = min(e_prime, graph.weights[best_edge_id] -
                              flow[best_edge_id]);
      atomicAdd(&flow[best_edge_id], push_amt);
      atomicAdd(&flow[reverse_indices[best_edge_id]], -push_amt);
      atomicAdd(&excess[u], -push_amt);
      atomicAdd(&excess[graph.edges[best_edge_id]], push_amt);
      *finished = false;
    }
    // Otherwise perform a relabel
    else if (h_prime != INFINITE) {
      height[u] = h_prime + 1;
      *finished = false;
    }
  }
}


/**
 * A common finalize function for GPU implementations. It allocates the host
 * output buffer, moves the final results from GPU to the host buffers and
 * frees up some resources.
 */
PRIVATE 
error_t finalize_gpu(graph_t* graph_d, weight_t* flow_d, weight_t* excess_d,
                     uint32_t* height_d, id_t* reverse_indices_d,
                     bool* finished_d, weight_t* flow_ret, id_t sink_id) {
  CHK_CU_SUCCESS(cudaMemcpy(flow_ret, (weight_t*)&(excess_d[sink_id]),
                            sizeof(weight_t), cudaMemcpyDeviceToHost), err);
  graph_finalize_device(graph_d);
  cudaFree(flow_d);
  cudaFree(excess_d);
  cudaFree(height_d);
  cudaFree(reverse_indices_d);
  cudaFree(finished_d);
  return SUCCESS;
 err:
  return FAILURE;
}


/**
 * GPU implementation of the Push-Relabel algorithm, as described in [Hong08]
 */
__host__
error_t maxflow_gpu(graph_t* graph, id_t source_id, id_t sink_id,
                    weight_t* flow_ret) {
  error_t rc = check_special_cases(graph, source_id, sink_id);
  if (rc != SUCCESS) return rc;

  // Setup reverse edges. This creates a new graph and updates the graph
  // pointer to point to this new graph. Thus, we have to do this step before
  // any other allocations/initialization.
  id_t* reverse_indices = NULL;
  graph_t* local_graph = graph_create_bidirectional(graph, &reverse_indices);

  uint32_t* height = (uint32_t*)mem_alloc(local_graph->vertex_count *
                                          sizeof(uint32_t));
  weight_t* excess = (weight_t*)mem_alloc(local_graph->vertex_count *
                                         sizeof(weight_t));
  weight_t* flow = (weight_t*)mem_alloc(local_graph->edge_count *
                                        sizeof(weight_t));

  // Create and initialize state on GPU
  graph_t* graph_d;
  weight_t* flow_d;
  weight_t* excess_d;
  uint32_t* height_d;
  id_t* reverse_indices_d;
  bool* finished_d;
  CHK_SUCCESS(initialize_gpu(local_graph, source_id, reverse_indices, &graph_d,
                             &flow_d, &excess_d, &height_d, &reverse_indices_d,
                             &finished_d), err_free_all);

  // {} used to limit scope and avoid problems with error handles.
  {
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(local_graph->edge_count, blocks, threads_per_block);
  memset_device<<<blocks, threads_per_block>>>(flow_d, (weight_t)0,
                                               local_graph->edge_count);

  // While there exists an applicable push or relabel operation, perform it
  bool finished = false;
  KERNEL_CONFIGURE(local_graph->vertex_count, blocks, threads_per_block);
  while (!finished) {
    CHK_CU_SUCCESS(cudaMemset(finished_d, true, sizeof(bool)), err_free_all);
    // Perform push-relabel on each vertex, according to [Hong08]
    push_relabel_kernel<<<blocks, threads_per_block>>>
      (*graph_d, flow_d, excess_d, height_d, reverse_indices_d, source_id,
       sink_id, finished_d);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
    CHK_CU_SUCCESS(cudaMemcpy(&finished, finished_d, sizeof(bool),
                              cudaMemcpyDeviceToHost), err_free_all);
    // TODO (Greg): Copy heights back to main memory and perform the global
    //              relabel operation on the CPU side
  }}

  // We are done, get the results back and clean up state
  CHK_SUCCESS(finalize_gpu(graph_d, flow_d, excess_d, height_d,
                           reverse_indices_d, finished_d, flow_ret, sink_id),
              err_free_all);
  mem_free(height);
  mem_free(excess);
  mem_free(flow);
  mem_free(reverse_indices);
  graph_finalize(local_graph);

  return SUCCESS;

  // error handlers
  err_free_all:
    cudaFree(finished_d);
    cudaFree(height_d);
    cudaFree(excess_d);
    cudaFree(flow_d);
    cudaFree(reverse_indices_d);
    graph_finalize_device(graph_d);
    mem_free(height);
    mem_free(excess);
    mem_free(flow);
    mem_free(reverse_indices);
    graph_finalize(local_graph);
    return FAILURE;
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
