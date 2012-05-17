/**
 *  Memory management
 *
 *  Created on: 2011-03-03
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_MEM_H
#define TOTEM_MEM_H

// totem includes
#include "totem_comdef.h"

/**
 * Allocates a buffer of size "size". Depending on the feature
 * FEATURE_PINNED_MEMORY, the buffer is either allocated in pinned or pageable
 * memory.
 * @param[in] size buffer size to allocate
 * @return allocated buffer
 */
void* mem_alloc(size_t size);

/**
 * Frees a buffer allocated by mem_alloc
 * @param[in] buf to be freed
 */
void mem_free(void* buf);

#endif  // TOTEM_MEM_H

