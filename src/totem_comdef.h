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
#include <sys/time.h>
#include <unistd.h>

/**
 *  Function return code types
 */
typedef enum {
  SUCCESS = 0, /**< generic success return code. */
  FAILURE = -1 /**< generic failure return code. */
} error_t;

/**
 * Command line options
 */
typedef struct options_s {
  char* graph_file;
  bool  weighted;
} options_t;

/**
 * Stopwatch (timer) type
 */
typedef double stopwatch_t;

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
 * Determines the maximum number of threads per block.
 */
#define MAX_THREADS_PER_BLOCK 512

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
 * Computes the linear thread index
 */
#define THREAD_GLOBAL_INDEX (threadIdx.x + blockDim.x                   \
                             * (gridDim.x * blockIdx.y + blockIdx.x))

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
 * Resets the timer to current system time. Called at the moment to 
 * start timing an operation.
 * @param[in] stopwatch the stopwatch handler
 */
inline void stopwatch_start(stopwatch_t* stopwatch) {
  struct timeval tval;
  gettimeofday(&tval, NULL);
  *stopwatch = (tval.tv_sec * 1000 + tval.tv_usec/1000.0);
}

/**
 * Returns the elapsed time since the stopwatch started via stopwatch_start
 * @param[in] stopwatch the stopwatch handler
 * @return elapsed time in milliseconds
 */
inline double stopwatch_elapsed(stopwatch_t* stopwatch) {
  struct timeval tval;
  gettimeofday(&tval, NULL);
  return ((tval.tv_sec * 1000 + tval.tv_usec/1000.0) - *stopwatch);
}

#endif  // TOTEM_COMDEF_H
