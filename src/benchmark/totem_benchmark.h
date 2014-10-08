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
#include "totem_alg.h"
#include "totem_util.h"

// Benchmark algorithm types.
typedef enum {
  BENCHMARK_BFS = 0,
  BENCHMARK_PAGERANK,
  BENCHMARK_SSSP,
  BENCHMARK_BETWEENNESS,
  BENCHMARK_GRAPH500,
  BENCHMARK_CLUSTERING_COEFFICIENT,
  BENCHMARK_BFS_STEPWISE,
  BENCHMARK_GRAPH500_STEPWISE,
  BENCHMARK_MAX
} benchmark_t;

// Benchmark attributes type.
typedef struct benchmark_attr_s {
  void(*func)(graph_t*, void*, totem_attr_t*);  // Benchmark function.
  const char*     name;           // Benchmark name.
  size_t          output_size;    // Per-vertex output size.
  bool            has_totem;      // true if the benchmark has a Totem-based
                                  // implementation
  size_t          push_msg_size;  // Push message size (Totem-based alg.).
  size_t          pull_msg_size;  // Pull message size (Totem-based alg.).
  totem_cb_func_t alloc_func;     // Allocation callback function
                                  // (Totem-based alg.).
  totem_cb_func_t free_func;      // Free callback function (Totem-based alg.).
} benchmark_attr_t;

// Benchmark command line options type.
typedef struct benchmark_options_s {
  char*                 graph_file;     // The file to run the benchmark on.
  benchmark_t           benchmark;      // Benchmark to run.
  platform_t            platform;       // Execution platform.
  int                   gpu_count;      // Number of GPUs to use.
  int                   thread_count;   // Number of CPU threads.
  omp_sched_t           omp_sched;      // OMP scheduling policy.
  int                   repeat;         // Number of times to repeat an
                                        // execution (for traversal algorithms,
                                        // different runs use different seeds.
  int                   alpha;          // The percentage of edges placed on the
                                        // CPU  for hybrid platforms
  partition_algorithm_t par_algo;       // Partitioning algorithm.
  gpu_graph_mem_t       gpu_graph_mem;  // The type of memory used to allocate
                                        // the graph data structure of
                                        // GPU-based partitions.
  bool                  gpu_par_randomized;  // Indicates if the placement of
                                             // vertices across GPUs is random
                                             // or according to par_algo.
  bool                  sorted;       // Maps the vertex ids in sorted order
                                      // by edge degree during the
                                      // partitioning phase.
  bool                  edge_sort_by_degree;  // Sorts the edges by degree
                                              // rather than by id.
  bool                  edge_sort_dsc;  // Makes the direction of edge
                                        // sorting descending order.
} benchmark_options_t;

/**
 * The getter for a reference to the benchmark options.
 */
benchmark_options_t* totem_benchmark_get_options();

/**
 * Parses command line options
 * @param[in] argc number of arguments
 * @param[in] argv argument array
 */
benchmark_options_t* benchmark_cmdline_parse(int argc, char** argv);

/**
 * Prints out the configuration parameters of this benchmark run
 * @param[in] graph the graph being benchmarked
 * @param[in] options benchmark options
 * @param[in] benchmark_name benchmark name
 */
void print_config(graph_t* graph, benchmark_options_t* options,
                  const char* benchmark_name);

/**
 * Prints out the header of the runs' detailed timing
 * @param[in] graph the graph being benchmarked
 * @param[in] totem_based defines whether the benchmark was run via Totem or not
 */
void print_header(graph_t* graph, bool totem_based);

/**
 * Prints out detailed timing of a single run
 * @param[in] graph the graph being benchmarked
 * @param[in] time_total end to end running time
 * @param[in] trv_edges number of edges processed
 * @param[in] totem_based defines whether the benchmark was run via Totem or not
 */
void print_timing(graph_t* graph, double time_total, uint64_t trv_edges,
                  bool totem_based);

#endif  // TOTEM_BENCHMARK_H
