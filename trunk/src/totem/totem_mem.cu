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

void* mem_alloc(size_t size) {
  void* buf;
#ifdef FEATURE_PAGEABLE_MEMORY
  buf = malloc(size);
#else
  CALL_CU_SAFE(cudaMallocHost(&buf, size, cudaHostAllocPortable));
#endif // FEATURE_PAGEABLE_MEMORY

  assert(buf);
  return buf;
}

void mem_free(void* buf) {
  assert(buf);
#ifdef FEATURE_PAGEABLE_MEMORY
  free(buf);
#else
  CALL_CU_SAFE(cudaFreeHost(buf));
#endif // FEATURE_PAGEABLE_MEMORY
}

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
      if (cudaMallocHost(ptr, size, cudaHostAllocPortable) != cudaSuccess) {
        err = FAILURE;
      }
      break;
    case TOTEM_MEM_HOST_MAPPED:
      if (cudaMallocHost(ptr, size, cudaHostAllocPortable | cudaHostAllocMapped)
          != cudaSuccess) {
        err = FAILURE;
      }
      break;
    case TOTEM_MEM_DEVICE:
      if (cudaMalloc(ptr, size) != cudaSuccess) {
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
