/**
 * Defines the benchmark's data types and constants
 *
 *  Created on: 2013-02-09
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_BENCHMARK_H
#define TOTEM_BENCHMARK_H

// totem includes
#include "totem.h"
#include "totem_mem.h"
#include "totem_util.h"

/**
 *  Benchmark algorithm types
 */
typedef enum {
  BENCHMARK_BFS = 0,
  BENCHMARK_PAGERANK,
  BENCHMARK_DIJKSTRA,
  BENCHMARK_BETWEENNESS,
  BENCHMARK_MAX
} benchmark_t;

/**
 *  Benchmark attributes type
 */
typedef struct benchmark_attr_s {
  void(*func)(graph_t*, void*, totem_attr_t*);
  const char* str;
  size_t push_msg_size;
  size_t pull_msg_size;
  size_t output_size;
} benchmark_attr_t;

/**
 *  Benchmark command line options type
 */
typedef struct benchmark_options_s {
  char*                 graph_file; /**<  The file to run the benchmark on */
  benchmark_t           benchmark;  /**<  Benchmark to run */
  platform_t            platform;   /**<  Execution platform */
  int                   gpu_count;  /**<  Number of GPUs to use for hybrid and 
                                          GPU-only platforms */
  int                   repeat;     /**<  Number of times to repeat an
                                          execution (for traversal algorithms,
                                          number of sources used) */
  int                   alpha;      /**< Percentage of edges left on the CPU
                                         for hybrid platforms */
  partition_algorithm_t par_algo;   /**< Partitioning algorithm */
} benchmark_options_t;

/**
 * Parses command line options
 * @param[in] argc number of arguments
 * @param[in] argv argument array
 */
benchmark_options_t* benchmark_cmdline_parse(int argc, char** argv);

#endif // TOTEM_BENCHMARK_H
