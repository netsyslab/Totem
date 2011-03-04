/**
 *  Memory management. The module either allocates pinned or pageable memory.
 *  The default is pageable memory, unless a compile time flag MEMORY=PINNED
 *  is specified to use pinned memory.
 *
 *  Created on: 2011-03-03
 *  Author: Abdullah Gharaibeh
 */


// totem includes
#include "totem_mem.h"
#include "cuda.h"

void* mem_alloc(int size) {

  void* buf;
#ifdef FEATURE_PINNED_MEMORY
  cudaMallocHost(&buf, size);  
#else
  buf = malloc(size);
#endif

  assert(buf);
  return buf;
}

void mem_free(void* buf) {
#ifdef FEATURE_PINNED_MEMORY
  cudaFreeHost(buf);
#else
  free(buf);
#endif
}
