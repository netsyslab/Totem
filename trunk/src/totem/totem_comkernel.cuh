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
 * A double precision atomic add. Based on the algorithm in the NVIDIA CUDA 
 * Programming Guide V4.0, Section B.11.
 * reads the 64-bit word old located at address in global or shared memory, 
 * computes (old + val), and stores the result back to memory at the same 
 * address atomically.
 * @param[in] address the content is incremented by val
 * @param[in] val the value to be added to the content of address
 * @return old value stored at address
 */
inline __device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

/**
 * A double precision atomic min function. Based on the double precisision
 * atomicAdd algorithm in the NVIDIA CUDA Programming Guide V4.0, Section B.11.
 * reads the 64-bit word old located at address in global or shared memory, 
 * computes the minimum of old and val, and stores the result back to memory at
 * the same address atomically.
 * @param[in] address the content is compared to val, the minimum is stored back
 * @param[in] val the value to be compared with the content of address
 * @return old value stored at address
 */
inline __device__ double atomicMin(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  double min;
  do {
    assumed = old;
    min = (val < __longlong_as_double(assumed)) ? val :
      __longlong_as_double(assumed);
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(min));
  } while (assumed != old);
  return __longlong_as_double(old);
}

/**
 * A single precision atomic min function. Based on the double precisision
 * atomicAdd algorithm in the NVIDIA CUDA Programming Guide V4.0, Section B.11.
 * reads the 32-bit word old located at address in global or shared memory, 
 * computes the minimum of old and val, and stores the result back to memory at
 * the same address atomically.
 * @param[in] address the content is compared to val, the minimum is stored back
 * @param[in] val the value to be compared with the content of address
 * @return old value stored at address
 */
inline __device__ float atomicMin(float* address, float val) {
  uint32_t* address_as_uint = (uint32_t*)address;
  uint32_t old = *address_as_uint, assumed;
  float min;
  do {
    assumed = old;
    min = (val < __int_as_float(assumed)) ? val : __int_as_float(assumed);
    old = atomicCAS(address_as_uint, assumed, __float_as_int(min));
  } while (assumed != old);
  return __int_as_float(old);
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
