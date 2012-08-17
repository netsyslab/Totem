/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Simply counts node degree.
 *
 *  Created on: 2012-04-30
 *      Author: Greg Redekop
 */

// totem includes
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

/**
 * A common initialization function for GPU implementations. It allocates and
 * initalizes state on the GPU.
 */
PRIVATE
error_t initialize_gpu(const graph_t* graph, vid_t** node_degree_d,
                       graph_t** graph_d) {
  dim3 blocks;
  dim3 threads_per_block;

  // Allocate space on GPU
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);
  CHK_CU_SUCCESS(cudaMalloc((void**)node_degree_d, graph->vertex_count *
                            sizeof(vid_t)), err_free_graph_d);
  // Initialize count to 0.
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
  memset_device<<<blocks, threads_per_block>>>((*node_degree_d), (vid_t)0,
                                               graph->vertex_count);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_all_d);

  return SUCCESS;

  err_free_all_d:
    cudaFree(node_degree_d);
  err_free_graph_d:
    graph_finalize_device(*graph_d);
  err:
    return FAILURE;
}


/**
 * Simply counts the degree of each vertex
 */
__global__
void node_degree_kernel(graph_t graph, vid_t* node_degree) {
  const vid_t u = THREAD_GLOBAL_INDEX;
  if (u >= graph.vertex_count) return;
  node_degree[u] = graph.vertices[u + 1] - graph.vertices[u];
}


/**
 * A common finalize function for GPU implementations. It allocates the host
 * output buffer, moves the final results from GPU to the host buffers and
 * frees up some resources.
 */
PRIVATE
error_t finalize_gpu(graph_t* graph_d, vid_t* node_degree_d,
                     vid_t** node_degree) {
  // Allocate space for returned node degree list
  *node_degree = (vid_t*)mem_alloc(graph_d->vertex_count * sizeof(vid_t));
  CHK_CU_SUCCESS(cudaMemcpy(*node_degree, node_degree_d,
                            graph_d->vertex_count * sizeof(vid_t),
                            cudaMemcpyDeviceToHost), err);
  graph_finalize_device(graph_d);
  cudaFree(node_degree_d);
  return SUCCESS;

  err:
    return FAILURE;
}


/**
 * GPU implementation for counting node degree.
 */
__host__
error_t node_degree_gpu(const graph_t* graph, vid_t** node_degree) {
  // Sanity on input parameters
  if (graph == NULL || node_degree == NULL || graph->vertex_count == 0)
    return FAILURE;

  // Create and initialize state on GPU
  graph_t* graph_d;
  vid_t* node_degree_d;

  CHK_SUCCESS(initialize_gpu(graph, &node_degree_d, &graph_d), err_free_all);

  // {} used to limit scope and avoid problems with error handles.
  {
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);

  // Invoke the vertex degree counting kernel
  node_degree_kernel<<<blocks, threads_per_block>>>(*graph_d, node_degree_d);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_all);
  }

  // We are done, get the results back and clean up state
  CHK_SUCCESS(finalize_gpu(graph_d, node_degree_d, node_degree), err_free_all);

  return SUCCESS;

  // error handlers
  err_free_all:
    cudaFree(node_degree_d);
    graph_finalize_device(graph_d);
    return FAILURE;
}


/**
 * CPU implementation of counting node degree
 */
error_t node_degree_cpu(const graph_t* graph, vid_t** node_degree) {
  // Sanity on input parameters
  if (graph == NULL || node_degree == NULL || graph->vertex_count == 0) {
    return FAILURE;
  }

  // Allocate node_degree output list
  *node_degree = (vid_t*)mem_alloc(graph->vertex_count * sizeof(vid_t));
  memset(*node_degree, 0, graph->vertex_count * sizeof(vid_t));

  OMP(omp parallel for)
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    // Count the node degree at the given node
    (*node_degree)[vid] = graph->vertices[vid + 1] - graph->vertices[vid];
  }
  return SUCCESS;
}
