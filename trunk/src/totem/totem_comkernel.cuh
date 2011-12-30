/**
 * This header includes common constants, macros and functions used to deal
 * with the GPU.
 * This includes (i) device utility functions (i.e., used only from within a 
 * kernel) such as double precision atomicMin and atomicAdd, (ii) utility 
 * functions to allocate, copy and free a device resident graph structure (e.g.,
 * graph_initialize_device and graph_finalize_device), (iii) utility kernels
 * such as memset_device and (iv) common constants and functions related to the
 * virtual warp technique (described below).
 *
 *  Created on: 2011-03-07
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_COMKERNEL_CUH
#define TOTEM_COMKERNEL_CUH

// totem includes
#include "totem_comdef.h"
#include "totem_graph.h"

// Virtual warp paramters are prefixed with "vwarp". Virtual warp is a technique
// to reduce divergence among threads within a warp. The idea is to have all the
// threads that belong to a warp work as a unit. To this end, instead of 
// dividing the work among threads, the work is divided among warps. A warp goes
// through phases of SISD and SIMD in complete lock-step as if they were all a 
// single thread. The technique divides the work into batches, where each warp 
// is responsible for one batch of work. Typically, the threads of a warp 
// collaborate to fetch their assigned batch data to shared memory, and together
// process the batch. The technique is presented in [Hong11] S. Hong, S. Kim, 
// T. Oguntebi and K.Olukotun "Accelerating CUDA Graph Algorithms at Maximum 
// Warp, PPoPP11.

/**
 * the number of threads a warp consists of
 */
#define VWARP_WARP_SIZE 32

/**
 * the size of the batch of work assigned to each virtual warp
 */
#define VWARP_BATCH_SIZE 32

/**
 * Determines the maximum number of threads per block.
 */
#define MAX_THREADS_PER_BLOCK 192

/**
 * Determines the maximum number of dimensions of a grid block.
 */
#define MAX_BLOCK_DIMENSION 2

/**
 * Determines the maximum number of blocks that fit in a grid dimension.
 */
#define MAX_BLOCK_PER_DIMENSION 1024

/**
 * Determines the maximum number of threads a kernel can be configured with.
 */
#define MAX_THREAD_COUNT \
  MAX_THREADS_PER_BLOCK * pow(MAX_BLOCK_PER_DIMENSION, MAX_BLOCK_DIMENSION)

/**
 * Global linear thread index
 */
#define THREAD_GLOBAL_INDEX (threadIdx.x + blockDim.x                   \
                             * (gridDim.x * blockIdx.y + blockIdx.x))

/**
 * Grid scope linear thread index
 */
#define THREAD_GRID_INDEX (threadIdx.x)

/**
 * number of batches of size "VWARP_BATCH_SIZE" vertices
 */
#define VWARP_BATCH_COUNT(vertex_count) \
  (((vertex_count) / VWARP_BATCH_SIZE) + \
   ((vertex_count) % VWARP_BATCH_SIZE == 0 ? 0 : 1))

/**
 * Computes a kernel configuration based on the number of vertices.
 * It assumes a 2D grid. vertex_count is input paramter, while blocks
 * and threads_per_block are output of type dim3.
 */
#define KERNEL_CONFIGURE(vertex_count, blocks, threads_per_block)       \
  do {                                                                  \
    assert(vertex_count <= MAX_THREAD_COUNT);                           \
    threads_per_block = (vertex_count) >= MAX_THREADS_PER_BLOCK ?       \
      MAX_THREADS_PER_BLOCK : vertex_count;                             \
    uint32_t blocks_left = (((vertex_count) % MAX_THREADS_PER_BLOCK == 0) ? \
                            (vertex_count) / MAX_THREADS_PER_BLOCK :    \
                            (vertex_count) / MAX_THREADS_PER_BLOCK + 1); \
    uint32_t x_blocks = (blocks_left >= MAX_BLOCK_PER_DIMENSION) ?      \
      MAX_BLOCK_PER_DIMENSION : blocks_left;                            \
    blocks_left = (((blocks_left) % x_blocks == 0) ?                    \
                   (blocks_left) / x_blocks :                           \
                   (blocks_left) / x_blocks + 1);                       \
    uint32_t y_blocks = (blocks_left >= MAX_BLOCK_PER_DIMENSION) ?      \
      MAX_BLOCK_PER_DIMENSION : blocks_left;                            \
    dim3 my_blocks(x_blocks, y_blocks);                                 \
    blocks = my_blocks;                                                 \
  } while(0)

/**
 * Check if return value of stmt is cudaSuccess, jump to label and print an 
 * error message if not.
 */
#define CHK_CU_SUCCESS(cuda_call, label)                                \
  do {                                                                  \
    if ((cuda_call) != cudaSuccess) {                                   \
      cudaError_t err = cudaGetLastError();                             \
      fprintf(stderr, "Cuda Error in file '%s' in line %i : %s.\n",     \
              __FILE__, __LINE__, cudaGetErrorString(err));             \
      goto label;                                                       \
    }                                                                   \
  } while(0)

/**
 * A SIMD version of memcpy for the virtual warp technique. The assumption is
 * that the threads of a warp invoke this function to copy their batch of work 
 * from global memory (src) to shared memory (dst). In each iteration of the for
 * loop, each thread copies one element. For example, thread 0 in the warp 
 * copies elements 0, VWARP_WARP_SIZE, (2 * VWARP_WARP_SIZE) etc., while thread 
 * thread 1 in the warp copies elements 1, (1 + VWARP_WARP_SIZE), 
 * (1 + 2 * VWARP_WARP_SIZE) and so on. Finally, this function is called only 
 * from within a kernel.
 * @param[in] dst destination buffer (typically shared memory buffer)
 * @param[in] src the source buffer (typically global memory buffer)
 * @param[in] size number of elements to copy
 * @param[in] thread_offset_in_warp thread index within its warp
 */
template<typename T> 
__device__ void vwarp_memcpy(T* dst, T* src, uint32_t size, 
                             uint32_t thread_offset_in_warp) {
  for(int i = thread_offset_in_warp; i < size; i += VWARP_WARP_SIZE) 
    dst[i] = src[i];
}

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
  if (index >= size) return;
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

  // Copy basic data types within the structure, the buffers pointers will be
  // overwritten next with device pointers
  **graph_d = *graph_h;

  // Vertices will be processed by each warp in batches. To avoid explicitly 
  // checking for end of array boundaries, the vertices array is padded with
  // fake vertices so that its length is multiple of batch size. The fake 
  // vertices has no edges and they don't count in the vertex_count (much
  // like the extra vertex we used to have which enables calculating the number
  // of neighbors for the last vertex). Note that this padding does not affect
  // the algorithms that do not apply the virtual warp technique.
  uint64_t vertex_count_batch_padded = VWARP_BATCH_SIZE *
    VWARP_BATCH_COUNT(graph_h->vertex_count);

  // Allocate device buffers
  CHK_CU_SUCCESS(cudaMalloc((void**)&(*graph_d)->vertices,
                            (vertex_count_batch_padded + 1) * 
                            sizeof(id_t)), err);
  CHK_CU_SUCCESS(cudaMalloc((void**)&(*graph_d)->edges, graph_h->edge_count *
                            sizeof(id_t)), err_free_vertices);
  if (graph_h->weighted) {
    CHK_CU_SUCCESS(cudaMalloc((void**)&(*graph_d)->weights, 
                              graph_h->edge_count * sizeof(weight_t)), 
                   err_free_edges);
  }
  
  // Move data to the GPU
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
  
  // Set the index of the extra vertices to the last actual vertex. This
  // renders the padded fake vertices with zero edges.
  int pad_size;
  pad_size = vertex_count_batch_padded - graph_h->vertex_count;
  if (pad_size > 0) {
    dim3 blocks, threads_per_block;
    KERNEL_CONFIGURE(pad_size, blocks, threads_per_block);
    memset_device<<<blocks, threads_per_block>>>
      (&((*graph_d)->vertices[graph_h->vertex_count + 1]), 
       graph_h->vertices[graph_h->vertex_count], pad_size);
    CHK_CU_SUCCESS(cudaGetLastError(), err_free_weights);
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
