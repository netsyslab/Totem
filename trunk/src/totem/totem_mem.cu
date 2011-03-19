/**
 *  Defines memory management functions. The allocated memory is either pinned 
 *  or pageable.  The default is pinned memory, unless a compile time flag 
 *  MEMORY=PAGEABLE is specified to use pageable memory.
 *
 *  Created on: 2011-03-03
 *  Author: Abdullah Gharaibeh
 */


// totem includes
#include "totem_mem.h"
#include "cuda.h"

void* mem_alloc(int size) {
  void* buf;
#ifdef FEATURE_PAGEABLE_MEMORY
  buf = malloc(size);
#else
  cudaMallocHost(&buf, size);
#endif // FEATURE_PAGEABLE_MEMORY

  assert(buf);
  return buf;
}

void mem_free(void* buf) {
#ifdef FEATURE_PAGEABLE_MEMORY
  free(buf);
#else
  cudaFreeHost(buf);
#endif // FEATURE_PAGEABLE_MEMORY
}
