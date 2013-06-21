/**
 * This header includes common constants, macros and functions used to deal
 * with the GPU.
 * This includes (i) device utility functions (i.e., used only from within a
 * kernel) such as double precision atomicMin and atomicAdd, (ii) utility
 * functions to allocate, copy and free a device resident graph structure (e.g.,
 * graph_initialize_device and graph_finalize_device) and (iii) utility kernels
 * such as memset_device.
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
 * Determines the maximum number of threads per block.
 */
const int MAX_THREADS_PER_BLOCK = 1024;

/**
 * Determines the maximum number of dimensions of a grid block.
 */
const int MAX_BLOCK_DIMENSION = 2;

/**
 * Determines the maximum number of blocks that fit in a grid dimension.
 */
const int MAX_BLOCK_PER_DIMENSION = 65535;

/**
 * Determines the maximum number of threads a kernel can be configured with.
 */
const int MAX_THREAD_COUNT =
  (MAX_THREADS_PER_BLOCK * pow(MAX_BLOCK_PER_DIMENSION, MAX_BLOCK_DIMENSION));

/**
 * Minimum percentage of device memory reserved for algorithm state
 */
const double GPU_MIN_ALG_STATE = .05;

/**
 * Global linear thread index
 */
#define THREAD_GLOBAL_INDEX (threadIdx.x + blockDim.x                   \
                             * (gridDim.x * blockIdx.y + blockIdx.x))

/**
 * Block scope linear thread index.
 */
#define THREAD_BLOCK_INDEX (threadIdx.x)

/**
 * Global linear thread-block index
 */
#define BLOCK_GLOBAL_INDEX (gridDim.x * blockIdx.y + blockIdx.x)

/**
 * Computes a kernel configuration based on the number of vertices.
 * It assumes a 2D grid. vertex_count is input paramter, while blocks
 * and threads_per_block are output of type dim3.
 * TODO(abdullah): change this to an inline function
 */
#define KERNEL_CONFIGURE(thread_count, blocks, threads_per_block)       \
  do {                                                                  \
    assert(thread_count <= MAX_THREAD_COUNT);                           \
    threads_per_block = (thread_count) >= MAX_THREADS_PER_BLOCK ?       \
      MAX_THREADS_PER_BLOCK : thread_count;                             \
    uint32_t blocks_left = (((thread_count) % MAX_THREADS_PER_BLOCK == 0) ? \
                            (thread_count) / MAX_THREADS_PER_BLOCK :    \
                            (thread_count) / MAX_THREADS_PER_BLOCK + 1); \
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
 * A wrapper that asserts the success of cuda calls
 */
#define CALL_CU_SAFE(cuda_call)                                         \
  do {                                                                  \
    cudaError_t err = cuda_call;                                        \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "Cuda Error in file '%s' in line %i : %s.\n",     \
              __FILE__, __LINE__, cudaGetErrorString(err));             \
      assert(false);                                                    \
    }                                                                   \
  } while(0)


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

#include "totem_vwarp.cuh"

#endif  // TOTEM_COMKERNEL_CUH
