/**
 *  Common definitions across the modules.
 *
 *  Created on: 2011-02-28
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_COMDEF_H
#define TOTEM_COMDEF_H

// system includes
#include <assert.h>
#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

/**
 *  Function return code types.
 */
typedef enum {
  SUCCESS = 0, /**< generic success return code. */
  FAILURE = -1 /**< generic failure return code. */
} error_t;

/**
 * A wrapper that asserts the success of totem function calls
 */
#define CALL_SAFE(func)                         \
  do {                                          \
    error_t err = func;                         \
    assert(err == SUCCESS);                     \
  } while(0)

/**
 * Simulates simple exceptions: if the statement is not correct, jump to
 * label, typically an error label where you could clean up before exit.
 */
#define CHECK_ERR(stmt, label)                  \
  do {                                          \
    if (!(stmt))                                \
      goto label;                               \
  } while(0)

/**
 * Converts the string to upper case
 * @param[in] str the string to change to upper case
 */
inline void to_upper(char* str) {
  assert(str);
  while (*str != '\0') {
    char c = toupper(*str);
    *str = c;
    str++;
  }
}

/**
 * Checks if the string is a numeric number
 * @param[in] str the string to check
 * @return true if the string represents a numeric number
 */
inline bool is_numeric(char* str) {
  assert(str);
  bool numeric = true;
  while (*str != '\0' && numeric) {
    if (!isdigit(*str++)) {
      numeric = false;
    }
  }
  return numeric;
}

/**
 * Command line options.
 */
typedef struct options_s {
  char* graph_file;
  bool  weighted;
} options_t;

/**
 * Used to define private functions and variables
 */
#define PRIVATE static

/**
 * A constant that represents the integer INFINITE quantity. Useful in several
 * graph algorithms.
 */
const uint32_t INFINITE = UINT_MAX;

/**
 * A macro that determines the number of threads per block passed to a kernel.
 */
#define THREADS_PER_BLOCK 512

/**
 * A macro that determines the maximum number of blocks. Currently it assumes
 * that the grid is just 1D.
 */
#define MAX_BLOCK_COUNT 1024

/**
 * A macro that determines the maximum number of threads a kernel will be 
 * configured with.
 */
#define MAX_THREAD_COUNT (MAX_BLOCK_COUNT * THREADS_PER_BLOCK)

/**
 * Computes the total number of blocks. It assumes a 1D grid.
 */
#define TOTAL_BLOCKS(vertex_count) (((vertex_count) % THREADS_PER_BLOCK == 0) ? \
                                    (vertex_count) / THREADS_PER_BLOCK : \
                                    (vertex_count) / THREADS_PER_BLOCK + 1)

/**
 * Computes the thread id while taking into account the block id and dimenstion
 */
#define THREAD_GLOBAL_INDEX (threadIdx.x + blockDim.x                   \
                             * (gridDim.x * blockIdx.y + blockIdx.x))

#endif  // TOTEM_COMDEF_H
