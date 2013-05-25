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

// totem includes
#include "totem_alg.h"

// For each stage of the algorithm, a kernel loops over each vertex this many
// times attempting pushes and relabels. This will also control the frequency
// of global relabeling.
#define KERNEL_CYCLES 35

// Static function declarations
__global__
void init_preflow(graph_t graph, eid_t edge_base, eid_t edge_end, 
                  weight_t* flow, weight_t* excess, eid_t* reverse_indices);

/**
   This structure is used by the virtual warp-based implementation. It stores a
   batch of work. It is typically allocated on shared memory and is processed by
   a single virtual warp.
 */
typedef struct {
  eid_t vertices[VWARP_DEFAULT_BATCH_SIZE + 2];
  uint32_t height[VWARP_DEFAULT_BATCH_SIZE];
} vwarp_mem_t;


/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
 */
PRIVATE
error_t check_special_cases(graph_t* graph, vid_t source_id, vid_t sink_id) {
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
error_t initialize_gpu(graph_t* graph, vid_t source_id, vid_t vwarp_length,
                       eid_t* reverse_indices, graph_t** graph_d,
                       weight_t** flow_d, weight_t** excess_d,
                       uint32_t** height_d, eid_t** reverse_indices_d,
                       bool** finished_d) {

  dim3 blocks;
  dim3 threads_per_block;

  // Calculate the source excess directly prior to allocation. This prevents
  // compilation errors about variable declaration after a jump
  weight_t source_excess = (weight_t)0;
  for (eid_t edge_id = graph->vertices[source_id];
       edge_id < graph->vertices[source_id + 1]; edge_id++) {
    source_excess -= graph->weights[edge_id];
  }
  // Allocate space on GPU
  totem_mem_t mem_type = TOTEM_MEM_DEVICE;
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);
  CHK_SUCCESS(totem_calloc(graph->edge_count * sizeof(eid_t), mem_type, 
                           (void**)reverse_indices_d), err_free_flow_d);
  CHK_SUCCESS(totem_calloc(graph->edge_count * sizeof(weight_t), mem_type,
                           (void**)flow_d), err_free_graph_d);
  CHK_SUCCESS(totem_calloc(graph->vertex_count * sizeof(weight_t), mem_type, 
                           (void**)excess_d), err_free_reverse_indices_d);
  CHK_SUCCESS(totem_calloc(vwarp_length * sizeof(uint32_t), mem_type, 
                           (void**)height_d), err_free_excess_d);
  // Initialize flow, height, and excess to 0
  CHK_CU_SUCCESS(cudaMemcpy((*reverse_indices_d), reverse_indices,
                            graph->edge_count * sizeof(eid_t),
                            cudaMemcpyHostToDevice), err_free_all_d);

  // From the source vertex, initialize preflow
  totem_memset(&((*height_d)[source_id]), (uint32_t)graph->vertex_count,
               1, mem_type);
  KERNEL_CONFIGURE((graph->vertices[source_id + 1] -
                    graph->vertices[source_id]), blocks, threads_per_block);
  init_preflow<<<blocks, threads_per_block>>>
    (**graph_d, graph->vertices[source_id], graph->vertices[source_id + 1],
     *flow_d, *excess_d, *reverse_indices_d);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_all_d);

  totem_memset(&((*excess_d)[source_id]), (weight_t)source_excess, 1, mem_type);

  // Allocate the termination flag
  CHK_SUCCESS(totem_malloc(sizeof(bool), mem_type, (void**)finished_d),
              err_free_all_d);

  return SUCCESS;

  err_free_all_d:
    totem_free(height_d, mem_type);
  err_free_excess_d:
    totem_free(excess_d, mem_type);
  err_free_reverse_indices_d:
    totem_free(reverse_indices_d, mem_type);
  err_free_flow_d:
    totem_free(flow_d, mem_type);
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
void init_preflow(graph_t graph, eid_t edge_base, eid_t edge_end, 
                  weight_t* flow, weight_t* excess, eid_t* reverse_indices) {
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
                         uint32_t* height, eid_t* reverse_indices,
                         vid_t source_id, vid_t sink_id, bool* finished) {
  const vid_t u = THREAD_GLOBAL_INDEX;
  if (u >= graph.vertex_count) return;
  if (u == source_id || u == sink_id) return;

  uint32_t count = KERNEL_CYCLES;
  while (count--) {
    if (excess[u] <= 0 || height[u] >= graph.vertex_count) continue;

    weight_t e_prime = excess[u];
    uint32_t h_prime = INFINITE;
    eid_t best_edge_id = INFINITE;

    // Find the lowest neighbor connected by a residual edge
    for (eid_t edge_id = graph.vertices[u]; edge_id < graph.vertices[u + 1];
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
 * The neighbors processing function. This function finds the smallest neighbor
 * height and sets the corresponding best edge index for the vertex. The
 * assumption is that the threads of a warp invoke this function to process the
 * warp's batch of work. In each iteration of the for loop, each thread
 * processes a neighbor. For example, thread 0 in the warp processes neighbors
 * at indices 0, VWARP_DEFAULT_WARP_WIDTH, (2 * VWARP_DEFAULT_WARP_WIDTH) etc. 
 * in the edges array, while thread 1 in the warp processes neighbors 1, 
 * (1 + VWARP_DEFAULT_WARP_WIDTH), (1 + 2 * VWARP_DEFAULT_WARP_WIDTH) and so on.
*/
__device__ void
vwarp_process_neighbors(vid_t warp_offset, vid_t warp_id, vid_t neighbor_count,
                        vid_t* neighbors, weight_t* flow, weight_t* weight,
                        uint32_t* height, uint32_t* lowest_height,
                        vid_t* best_edge_id) {
  for (vid_t i = warp_offset; i < neighbor_count; i += 
         VWARP_DEFAULT_WARP_WIDTH) {
    vid_t neighbor_id = neighbors[i];
    if (weight[i] > flow[i]) {
      uint32_t h_pprime = height[neighbor_id];
      while (*lowest_height > h_pprime) {
        *lowest_height = h_pprime;
        // TODO: remove synchronization when VWARP_DEFAULT_WARP_WIDTH <= 32
        __threadfence();
        if (height[neighbor_id] == *lowest_height) {
          *best_edge_id = i;
        }
      }
    }
  } // for
}

/**
 * Implements the push relabel kernel, as per [Hong08]. Modified to employ the
 * virtual warp technique.
 */
__global__
void vwarp_push_relabel_kernel(graph_t graph, weight_t* flow, weight_t* excess,
                               uint32_t* height, eid_t* reverse_indices,
                               vid_t source_id, vid_t sink_id, bool* finished,
                               vid_t thread_count) {
  const vid_t thread_id = THREAD_GLOBAL_INDEX;
  if (thread_id >= thread_count) return;

  vid_t warp_offset = thread_id % VWARP_DEFAULT_WARP_WIDTH;
  vid_t warp_id     = thread_id / VWARP_DEFAULT_WARP_WIDTH;

  __shared__ vwarp_mem_t shared_memory[(MAX_THREADS_PER_BLOCK /
                                        VWARP_DEFAULT_WARP_WIDTH)];
  __shared__ vid_t best_edge_ids[MAX_THREADS_PER_BLOCK / 
                                 VWARP_DEFAULT_WARP_WIDTH];
  __shared__ uint32_t lowest_heights[MAX_THREADS_PER_BLOCK / 
                                     VWARP_DEFAULT_WARP_WIDTH];
  vwarp_mem_t* my_space = &shared_memory[THREAD_BLOCK_INDEX / 
                                         VWARP_DEFAULT_WARP_WIDTH];

  // copy my work to local space
  vid_t v_ = warp_id * VWARP_DEFAULT_BATCH_SIZE;
  vwarp_memcpy(my_space->height, &(height[v_]), VWARP_DEFAULT_BATCH_SIZE, 
               warp_offset);
  vwarp_memcpy(my_space->vertices, &(graph.vertices[v_]), 
               VWARP_DEFAULT_BATCH_SIZE + 1, warp_offset);

  int count = KERNEL_CYCLES;
  while(count--) {
    // iterate over my work
    for(vid_t v = 0; v < VWARP_DEFAULT_BATCH_SIZE; v++) {
      vid_t vertex_id = v_ + v;
      if (excess[vertex_id] > 0 && (vertex_id != sink_id) &&
          my_space->height[v] < graph.vertex_count) {
        vid_t* best_edge_id = &(best_edge_ids[(THREAD_BLOCK_INDEX /
                                               VWARP_DEFAULT_WARP_WIDTH)]);
        uint32_t* lowest_height = &(lowest_heights[(THREAD_BLOCK_INDEX /
                                                    VWARP_DEFAULT_WARP_WIDTH)]);
        *best_edge_id = INFINITE;
        *lowest_height = INFINITE;
        // TODO: remove synchronization when VWARP_DEFAULT_WARP_WIDTH <= 32
        __threadfence();

        vid_t* edges = &(graph.edges[my_space->vertices[v]]);
        weight_t* weights = &(graph.weights[my_space->vertices[v]]);
        weight_t* flows = &(flow[my_space->vertices[v]]);

        // Find the lowest neighbor connected by a residual edge
        vid_t nbr_count = my_space->vertices[v + 1] - my_space->vertices[v];
        vwarp_process_neighbors(warp_offset, warp_id, nbr_count, edges,
                                flows, weights, height, lowest_height,
                                best_edge_id);
        // TODO: remove synchronization when VWARP_DEFAULT_WARP_WIDTH <= 32
        __threadfence();

        // Only one thread does this per vertex
        if (warp_offset == 0) {
          eid_t edge = my_space->vertices[v] + *best_edge_id;
          // If a push applies
          if (height[vertex_id] > *lowest_height && *best_edge_id != INFINITE &&
              (graph.weights[edge] != flow[edge])) {
            weight_t push_amt = min(excess[vertex_id],
                                    graph.weights[edge] - flow[edge]);
            atomicAdd(&flow[edge], push_amt);
            atomicAdd(&flow[reverse_indices[edge]], -push_amt);
            atomicAdd(&excess[vertex_id], -push_amt);
            atomicAdd(&excess[graph.edges[edge]], push_amt);
            *finished = false;
          }
          // Otherwise perform a relabel
          else if (*lowest_height != INFINITE) {
            height[vertex_id] = *lowest_height + 1;
            *finished = false;
          }
        }
      }
    } // for
  } // while
}


/**
 * A common finalize function for GPU implementations. It allocates the host
 * output buffer, moves the final results from GPU to the host buffers and
 * frees up some resources.
 */
PRIVATE
error_t finalize_gpu(graph_t* graph_d, weight_t* flow_d, weight_t* excess_d,
                     uint32_t* height_d, eid_t* reverse_indices_d,
                     bool* finished_d, weight_t* flow_ret, vid_t sink_id) {
  totem_mem_t mem_type = TOTEM_MEM_DEVICE;
  CHK_CU_SUCCESS(cudaMemcpy(flow_ret, (weight_t*)&(excess_d[sink_id]),
                            sizeof(weight_t), cudaMemcpyDeviceToHost), err);
  graph_finalize_device(graph_d);
  totem_free(flow_d, mem_type);
  totem_free(excess_d, mem_type);
  totem_free(height_d, mem_type);
  totem_free(reverse_indices_d, mem_type);
  totem_free(finished_d, mem_type);
  return SUCCESS;
 err:
  return FAILURE;
}


/**
 * GPU implementation of the Push-Relabel algorithm, as described in [Hong08],
 * implementing the virtual warping technique.
 */
__host__
error_t maxflow_vwarp_gpu(graph_t* graph, vid_t source_id, vid_t sink_id,
                          weight_t* flow_ret) {
  error_t rc = check_special_cases(graph, source_id, sink_id);
  if (rc != SUCCESS) return rc;

  // Setup reverse edges. This creates a new graph and updates the graph
  // pointer to point to this new graph. Thus, we have to do this step before
  // any other allocations/initialization.
  eid_t* reverse_indices = NULL;
  graph_t* local_graph = graph_create_bidirectional(graph, &reverse_indices);

  // Create and initialize state on GPU
  graph_t* graph_d;
  weight_t* flow_d;
  weight_t* excess_d;
  uint32_t* height_d;
  eid_t* reverse_indices_d;
  bool* finished_d;
  CHK_SUCCESS(initialize_gpu(local_graph, source_id, 
                             vwarp_default_state_length(graph->vertex_count),
                             reverse_indices, &graph_d, &flow_d, &excess_d,
                             &height_d, &reverse_indices_d, &finished_d),
              err_free_all);

  // {} used to limit scope and avoid problems with error handles.
  {
  CALL_SAFE(totem_memset(flow_d, (weight_t)0, local_graph->edge_count, 
                         TOTEM_MEM_DEVICE));
  dim3 blocks;
  dim3 threads_per_block;
  uint32_t thread_count = vwarp_default_thread_count(graph->vertex_count);
  KERNEL_CONFIGURE(thread_count, blocks, threads_per_block);
  cudaFuncSetCacheConfig(vwarp_push_relabel_kernel, cudaFuncCachePreferShared);
  bool finished = false;
  // While there exists an applicable push or relabel operation, perform it
  while (!finished) {
    CHK_CU_SUCCESS(cudaMemset(finished_d, true, sizeof(bool)), err_free_all);
    // Perform push-relabel on each vertex, according to [Hong08]
    vwarp_push_relabel_kernel<<<blocks, threads_per_block>>>
      (*graph_d, flow_d, excess_d, height_d, reverse_indices_d, source_id,
       sink_id, finished_d, thread_count);
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
  totem_free(reverse_indices, TOTEM_MEM_HOST);
  graph_finalize(local_graph);

  return SUCCESS;

  // error handlers
  err_free_all:
    totem_free(flow_d, TOTEM_MEM_DEVICE);
    totem_free(excess_d, TOTEM_MEM_DEVICE);
    totem_free(height_d, TOTEM_MEM_DEVICE);
    totem_free(reverse_indices_d, TOTEM_MEM_DEVICE);
    totem_free(finished_d, TOTEM_MEM_DEVICE);
    graph_finalize_device(graph_d);
    totem_free(reverse_indices, TOTEM_MEM_HOST);
    graph_finalize(local_graph);
    return FAILURE;
}


/**
 * GPU implementation of the Push-Relabel algorithm, as described in [Hong08]
 */
__host__
error_t maxflow_gpu(graph_t* graph, vid_t source_id, vid_t sink_id,
                    weight_t* flow_ret) {
  error_t rc = check_special_cases(graph, source_id, sink_id);
  if (rc != SUCCESS) return rc;

  // Setup reverse edges. This creates a new graph and updates the graph
  // pointer to point to this new graph. Thus, we have to do this step before
  // any other allocations/initialization.
  eid_t* reverse_indices = NULL;
  graph_t* local_graph = graph_create_bidirectional(graph, &reverse_indices);

  // Create and initialize state on GPU
  graph_t* graph_d;
  weight_t* flow_d;
  weight_t* excess_d;
  uint32_t* height_d;
  eid_t* reverse_indices_d;
  bool* finished_d;
  CHK_SUCCESS(initialize_gpu(local_graph, source_id, graph->vertex_count,
                             reverse_indices, &graph_d, &flow_d, &excess_d,
                             &height_d, &reverse_indices_d, &finished_d),
              err_free_all);

  // {} used to limit scope and avoid problems with error handles.
  {
  CALL_SAFE(totem_memset(flow_d, (weight_t)0, local_graph->edge_count, 
                         TOTEM_MEM_DEVICE));
  // While there exists an applicable push or relabel operation, perform it
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(local_graph->vertex_count, blocks, threads_per_block);
  bool finished = false;
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
  totem_free(reverse_indices, TOTEM_MEM_HOST);
  graph_finalize(local_graph);

  return SUCCESS;

  // error handlers
  err_free_all:
    totem_free(flow_d, TOTEM_MEM_DEVICE);
    totem_free(excess_d, TOTEM_MEM_DEVICE);
    totem_free(height_d, TOTEM_MEM_DEVICE);
    totem_free(reverse_indices_d, TOTEM_MEM_DEVICE);
    totem_free(finished_d, TOTEM_MEM_DEVICE);
    graph_finalize_device(graph_d);
    totem_free(reverse_indices, TOTEM_MEM_HOST);
    graph_finalize(local_graph);
    return FAILURE;
}


/**
 * CPU Push-relabel operation
 * On a particular vertex u, attempt a push operation along any of its edges.
 * If the push operation fails, perform a relabel.
 */
PRIVATE
void push_relabel_cpu(graph_t* graph, vid_t u, vid_t source_id, vid_t sink_id,
                      weight_t* flow, weight_t* excess, uint32_t* height,
                      eid_t* reverse_indices, bool* finished) {
  if (excess[u] <= 0 || height[u] >= graph->vertex_count) return;

  weight_t e_prime = excess[u];
  uint32_t h_prime = INFINITE;
  eid_t best_edge_id = INFINITE;

  // Find the lowest neighbor connected by a residual edge
  for (eid_t edge_id = graph->vertices[u]; edge_id < graph->vertices[u + 1];
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


error_t maxflow_cpu(graph_t* graph, vid_t source_id, vid_t sink_id,
                    weight_t* flow_ret) {
  error_t rc = check_special_cases(graph, source_id, sink_id);
  if (rc != SUCCESS) return rc;

  // Setup residual edges. This creates a new graph and updates the graph
  // pointer to point to this new graph. Thus, we have to do this step before
  // any other allocations/initialization.
  eid_t* reverse_indices = NULL;
  graph_t* local_graph = graph_create_bidirectional(graph, &reverse_indices);

  weight_t* excess = NULL;
  CALL_SAFE(totem_calloc(local_graph->vertex_count * sizeof(weight_t), 
                         TOTEM_MEM_HOST, (void**)&excess));
  uint32_t* height = NULL;
  CALL_SAFE(totem_calloc(local_graph->vertex_count * sizeof(uint32_t),
                         TOTEM_MEM_HOST, (void**)&height));
  weight_t* flow = NULL;
  CALL_SAFE(totem_calloc(local_graph->edge_count * sizeof(weight_t), 
                         TOTEM_MEM_HOST, (void**)&flow));

  // Initialize source's height to the vertex count
  height[source_id] = (uint32_t) local_graph->vertex_count;

  // Initialize preflow
  for (eid_t edge_id = local_graph->vertices[source_id];
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
      OMP(omp parallel for)
      for (vid_t u = 0; u < local_graph->vertex_count; u++) {
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

  totem_free(reverse_indices, TOTEM_MEM_HOST);
  totem_free(excess, TOTEM_MEM_HOST);
  totem_free(height, TOTEM_MEM_HOST);
  totem_free(flow, TOTEM_MEM_HOST);
  // Free our modified new graph
  graph_finalize(local_graph);

  return SUCCESS;
}
