/**
 *  Memory management
 *
 *  Created on: 2011-03-03
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_MEM_H
#define TOTEM_MEM_H

// totem includes
#include "totem_comkernel.cuh"

/**
 * Buffer types that can be allocated via the memory interface
 */
typedef enum {
  TOTEM_MEM_HOST,        /**< allocates pageable memory buffers using malloc */
  TOTEM_MEM_HOST_PINNED, /**< allocates pinned memory on the host */
  TOTEM_MEM_HOST_MAPPED, /**< allocates pinned memory on the host that is mapped
                              to the address space of the GPUs. Note that the 
                              pointer returned is in the host address space. To 
                              get the corresponding pointer in the GPU address 
                              space, use cudaHostGetDevicePointer function */
  TOTEM_MEM_DEVICE       /**< allocates space on the currently in use device */
} totem_mem_t;

/**
 * Allocates a buffer of size "size". Depending on the type, the buffer is
 * either allocated on the host or on the device. In case of the former,
 * the buffer can be either pinned or not.
 * @param[in] size buffer size to allocate
 * @param[in] type type of memory to allocate
 * @param[out] ptr pointer to the allocated space
 * @return SUCCESS if the buffer is allocated, FALSE otherwise
 */
error_t totem_malloc(size_t size, totem_mem_t type, void** ptr);

/**
 * Similar to totem_malloc with the difference that the allocated space is
 * filled with zeros.
 * the buffer can be either pinned or not.
 * @param[in] size buffer size to allocate
 * @param[in] type type of memory to allocate
 * @param[out] ptr pointer to the allocated space
 * @return SUCCESS if the buffer is allocated, FALSE otherwise
 */
error_t totem_calloc(size_t size, totem_mem_t type, void** ptr);

/**
 * Free a buffer allocated by totem_malloc or totem_calloc. Note that the type
 * must be of the same one used at allocation time.
 * @param[in] ptr pointer to the buffer to be freed
 * @param[in] type type of memory to allocate
 */
void totem_free(void* ptr, totem_mem_t type);

/**
 * A templatized version of memset for device buffers, the assumption is that
 * the caller will dispatch a number of threads at least equal to "size"
 * @param[in] buffer the buffer to be set
 * @param[in] value the value the buffer elements are set to
 * @param[in] size number of elements in the buffer
 */
template<typename T>
__global__ void totem_memset_kernel(T* buffer, T value, uint32_t size) {
  uint32_t index = THREAD_GLOBAL_INDEX;
  if (index >= size) return;
  buffer[index] = value;
}

/**
 * A templatized version of memset.
 * @param[in] ptr pointer the buffer to be set
 * @param[in] value the value the buffer elements are set to
 * @param[in] size number of elements in the buffer
 * @param[in] type type of memory to be set
 * @param[in] stream the cuda stream within which the memset kernel will
 *                   be launch. This is only relevant to device pointers.
 *                   Setting this to 0 launches the kernel in the default
 *                   in the default stream. This argument is not relevant
 *                   to host buffers.
 */
template<typename T>
error_t totem_memset(T* ptr, T value, size_t size, totem_mem_t type, 
                  cudaStream_t stream = 0) {
  switch (type) {
    case TOTEM_MEM_HOST:
    case TOTEM_MEM_HOST_PINNED: 
    case TOTEM_MEM_HOST_MAPPED: {
      OMP(omp parallel for schedule(static))
      for (size_t i = 0; i < size; i++) {
        ptr[i] = value;
      }
      break;
    }
    case TOTEM_MEM_DEVICE: {
      dim3 blocks; dim3 threads;
      KERNEL_CONFIGURE(size, blocks, threads);
      totem_memset_kernel<<<blocks, threads, 0, stream>>>(ptr, value, size);
      if (cudaGetLastError() != cudaSuccess) return FAILURE;      
      break;
    }
    default:
      fprintf(stderr, "Error: invalid memory type\n");
      assert(false);
  }
  return SUCCESS;
}

#endif  // TOTEM_MEM_H
