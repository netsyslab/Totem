/**
 *  Defines memory management functions. The allocated memory is either pinned
 *  or pageable.  The default is pinned memory, unless a compile time flag
 *  MEMORY=PAGEABLE is specified to use pageable memory.
 *
 *  Created on: 2011-03-03
 *  Author: Abdullah Gharaibeh
 */


// totem includes
#include "totem_comkernel.cuh"
#include "totem_mem.h"

error_t totem_malloc(size_t size, totem_mem_t type, void** ptr) {
  error_t err = SUCCESS;
  switch (type) {
    case TOTEM_MEM_HOST:
      *ptr = malloc(size);
      if (*ptr == NULL) { 
        err = FAILURE;
      }
      break;
    case TOTEM_MEM_HOST_PINNED:
      // cudaHostAllocPortable allocates buffers on the host that are
      // accessible by all CUDA contexts (i.e., all devices).
      if (cudaMallocHost(ptr, size, cudaHostAllocPortable) != cudaSuccess) {
        err = FAILURE;
      }
      break;
    case TOTEM_MEM_HOST_MAPPED:
      // cudaHostAllocMapped specifies that the allocated space is mapped
      // to the device address space, which enables the device to directly
      // address that space
      // cudaHostAllocWriteCombined specifies that the buffer is write-combined
      // space, which potentially allows for faster read access from the device
      // at the expense of less efficient read access by the CPU
      if (cudaMallocHost(ptr, size, cudaHostAllocPortable | 
                         cudaHostAllocMapped | cudaHostAllocWriteCombined)
          != cudaSuccess) {
        err = FAILURE;
      }
      break;
    case TOTEM_MEM_DEVICE:
      if (cudaMalloc(ptr, size) != cudaSuccess) {
        size_t available = 0; size_t total = 0;
        CALL_CU_SAFE(cudaMemGetInfo(&available, &total));
        fprintf(stderr, "Error: Could not allocate memory on device. "
                "Requested %llu, Available %llu\n", size, available); 
        fflush(stdout);
        err = FAILURE;
      }
      break;
    default:
      fprintf(stderr, "Error: invalid memory type\n");
      assert(false);
  }
  return err;
}

error_t totem_calloc(size_t size, totem_mem_t type, void** ptr) {
  error_t err = totem_malloc(size, type, ptr);
  if (err != SUCCESS) {
    return FAILURE;
  }
  switch (type) {
    case TOTEM_MEM_HOST:
    case TOTEM_MEM_HOST_PINNED:
    case TOTEM_MEM_HOST_MAPPED:
      memset(*ptr, 0, size);
      break;
    case TOTEM_MEM_DEVICE:
      if (cudaMemset(*ptr, 0, size) != cudaSuccess) {
        err = FAILURE;
      }
      break;
    default:
      fprintf(stderr, "Error: invalid memory type\n");
      assert(false);
  }
  return err;
}

void totem_free(void* ptr, totem_mem_t type) {
  switch (type) {
    case TOTEM_MEM_HOST:
      free(ptr);
      break;
    case TOTEM_MEM_HOST_PINNED:
    case TOTEM_MEM_HOST_MAPPED:
      CALL_CU_SAFE(cudaFreeHost(ptr));
      break;
    case TOTEM_MEM_DEVICE:
      CALL_CU_SAFE(cudaFree(ptr));
      break;
    default:
      fprintf(stderr, "Error: invalid memory type\n");
      assert(false);
  }
}
