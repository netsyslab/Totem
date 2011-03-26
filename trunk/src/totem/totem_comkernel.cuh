/**
 *  Defines a set of convenience kernels
 *
 *  Created on: 2011-03-07
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_COMKERNEL_CUH
#define TOTEM_COMKERNEL_CUH

// totem includes
#include "totem_comdef.h"
#include "totem_graph.h"

/**
 * A templatized version of memset for device buffers, the assumption is that
 * the caller will dispatch a number of threads at least equal to "size"
 * @param[in] buffer the buffer to be set
 * @param[in] value the value the buffer elements are set to
 * @param[in] size number of elements in the buffer
 */
template<typename T>
__global__ void memset_device(T* buffer, T value, uint32_t size) {

  uint32_t index = THREAD_GLOBAL_INDEX;

  if (index >= size) {
    return;
  }

  buffer[index] = value;
}

/**
 * Initialize a graph structure (graph_d) to be passed as a parameter to GPU
 * kernels. Both graph_d and graph_h structs reside in host memory. The
 * vertices, edges and weights pointers in graph_d will point to buffers in
 * device memory allocated by the routine. Also, the routine will copy-in the
 * data to the aforementioned three buffers from the corresponding buffers in
 * graph_h.
 * @param[in] graph_h source graph which hosts references to main memory buffers
 * @param[out] graph_d allocated graph that hosts references to device buffers
 * @return generic success or failure
 */
inline error_t graph_initialize_device(const graph_t* graph_h,
                                       graph_t** graph_d) {
  assert(graph_h);

  // Allocate the graph struct that will host references to device buffers
  *graph_d = (graph_t*)malloc(sizeof(graph_t));
  if (*graph_d == NULL) return FAILURE;

  /* Copy basic data types within the structure, the buffers pointers will be
     overwritten next with device pointers */
  **graph_d = *graph_h;

  /* Allocate vertices, edges and weights device buffers and move them to
     the device. */
  CHK_CU_SUCCESS(cudaMalloc((void**)&(*graph_d)->vertices,
                            (graph_h->vertex_count + 1) *
                            sizeof(id_t)), err);
  CHK_CU_SUCCESS(cudaMalloc((void**)&(*graph_d)->edges, graph_h->edge_count *
                            sizeof(id_t)), err_free_vertices);
  if (graph_h->weighted) {
    CHK_CU_SUCCESS(cudaMalloc((void**)&(*graph_d)->weights, 
                              graph_h->edge_count * sizeof(weight_t)), 
                   err_free_edges);
  }
  
  CHK_CU_SUCCESS(cudaMemcpy((*graph_d)->vertices, graph_h->vertices,
                            (graph_h->vertex_count + 1) * sizeof(id_t),
                            cudaMemcpyHostToDevice), err_free_weights);
  CHK_CU_SUCCESS(cudaMemcpy((*graph_d)->edges, graph_h->edges,
                            graph_h->edge_count * sizeof(id_t),
                            cudaMemcpyHostToDevice), err_free_weights);
  if (graph_h->weighted) {
    CHK_CU_SUCCESS(cudaMemcpy((*graph_d)->weights, graph_h->weights,
                              graph_h->edge_count * sizeof(weight_t),
                              cudaMemcpyHostToDevice), err_free_weights);
  }
  
  return SUCCESS;
  
 err_free_weights:
  if ((*graph_d)->weighted) cudaFree((*graph_d)->weights);
 err_free_edges:
  cudaFree((*graph_d)->edges);
 err_free_vertices:
  cudaFree((*graph_d)->vertices);
 err:
  free(*graph_d);
  printf("%d\n", cudaGetLastError());
  return FAILURE;
}

/**
 * Free allocated device buffers associated with the graph
 * @param[in] graph_d the graph to be finalized
 */
inline void graph_finalize_device(graph_t* graph_d) {
  assert(graph_d);
  cudaFree(graph_d->edges);
  cudaFree(graph_d->vertices);
  if (graph_d->weighted) cudaFree(graph_d->weights);
  free(graph_d);
}

#endif  // TOTEM_COMKERNEL_CUH
